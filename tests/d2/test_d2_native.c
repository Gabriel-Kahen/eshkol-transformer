#define _POSIX_C_SOURCE 200809L

#include "../../native/d2_native.h"
#include "eshkol_transformer/i64_tensor.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
      failures++;                                                              \
    }                                                                          \
  } while (0)

typedef union bytevector_fixture {
  int64_t alignment;
  uint8_t bytes[256];
} bytevector_fixture;

static int read_stage(int64_t result) { return (int)((uint64_t)result >> 32); }

static int read_errno(int64_t result) {
  return (int)((uint64_t)result & UINT32_MAX);
}

static void init_bytevector(bytevector_fixture *fixture, int64_t length) {
  memset(fixture, 0xa5, sizeof(*fixture));
  memcpy(fixture->bytes, &length, sizeof(length));
}

static uint8_t *payload(bytevector_fixture *fixture) {
  return fixture->bytes + sizeof(int64_t);
}

static size_t open_fd_count(void) {
  DIR *directory = opendir("/proc/self/fd");
  struct dirent *entry;
  size_t count = 0u;
  if (directory == NULL) {
    return SIZE_MAX;
  }
  while ((entry = readdir(directory)) != NULL) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
      count++;
    }
  }
  (void)closedir(directory);
  return count;
}

static void test_exact_read(void) {
  char path[] = "/tmp/et-d2-exact-read-XXXXXX";
  uint8_t contents[16];
  bytevector_fixture vector;
  int descriptor;
  int64_t result;
  int64_t header;
  size_t before_fds;
  size_t after_fds;
  size_t index;

  for (index = 0u; index < sizeof(contents); index++) {
    contents[index] = (uint8_t)(index + 11u);
  }
  descriptor = mkstemp(path);
  CHECK(descriptor >= 0);
  if (descriptor < 0) {
    return;
  }
  CHECK(write(descriptor, contents, sizeof(contents)) ==
        (ssize_t)sizeof(contents));
  CHECK(close(descriptor) == 0);

  init_bytevector(&vector, 5);
  CHECK(et_d2_exact_read_v1(path, (int64_t)sizeof(path), 3, &vector, 5) == 0);
  CHECK(memcmp(payload(&vector), contents + 3u, 5u) == 0);
  memcpy(&header, vector.bytes, sizeof(header));
  CHECK(header == 5);

  init_bytevector(&vector, 5);
  CHECK(et_d2_exact_read_v1(path, (int64_t)sizeof(path), 11, &vector, 5) == 0);
  CHECK(memcmp(payload(&vector), contents + 11u, 5u) == 0);
  init_bytevector(&vector, 0);
  CHECK(et_d2_exact_read_v1(path, (int64_t)sizeof(path), 16, &vector, 0) == 0);
  CHECK(et_d2_exact_read_v1(path, (int64_t)sizeof(path), INT64_MAX, &vector,
                            0) == 0);

  init_bytevector(&vector, 5);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), 14, &vector, 5);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_READ);
  CHECK(read_errno(result) == EIO);
  memcpy(&header, vector.bytes, sizeof(header));
  CHECK(header == 5);
  CHECK(memcmp(payload(&vector), contents + 14u, 2u) == 0);
  result =
      et_d2_exact_read_v1(path, (int64_t)sizeof(path), INT64_MAX, &vector, 1);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path) - 1, 0, &vector, 5);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), -1, &vector, 5);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  {
    const char embedded_nul[] = {'x', '\0', 'y', '\0'};
    result = et_d2_exact_read_v1(embedded_nul, 4, 0, &vector, 5);
    CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  }
  init_bytevector(&vector, 4);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), 0, &vector, 5);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), 0, NULL, 5);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  {
    bytevector_fixture overlap;
    int64_t length = 64;
    memset(&overlap, 0, sizeof(overlap));
    memcpy(overlap.bytes, &length, sizeof(length));
    memcpy(overlap.bytes + 8u, path, sizeof(path));
    result = et_d2_exact_read_v1((const char *)overlap.bytes + 8u,
                                 (int64_t)sizeof(path), 0, &overlap, length);
    CHECK(read_stage(result) == ET_D2_EXACT_READ_ARGUMENT);
  }

  before_fds = open_fd_count();
  for (index = 0u; index < 1024u; index++) {
    init_bytevector(&vector, 5);
    result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), 14, &vector, 5);
    CHECK(read_stage(result) == ET_D2_EXACT_READ_READ);
  }
  after_fds = open_fd_count();
  CHECK(before_fds != SIZE_MAX && after_fds == before_fds);

  CHECK(unlink(path) == 0);
  init_bytevector(&vector, 1);
  result = et_d2_exact_read_v1(path, (int64_t)sizeof(path), 0, &vector, 1);
  CHECK(read_stage(result) == ET_D2_EXACT_READ_OPEN);
  CHECK(read_errno(result) == ENOENT);
}

static void encode_i64le(bytevector_fixture *fixture, const int64_t *values,
                         size_t count) {
  size_t value_index;
  init_bytevector(fixture, (int64_t)(8u * count));
  for (value_index = 0u; value_index < count; value_index++) {
    uint64_t bits;
    size_t byte_index;
    memcpy(&bits, &values[value_index], sizeof(bits));
    for (byte_index = 0u; byte_index < 8u; byte_index++) {
      payload(fixture)[value_index * 8u + byte_index] =
          (uint8_t)(bits >> (byte_index * 8u));
    }
  }
}

static void write_span(const void *owner, int64_t generation,
                       int64_t destination, const int64_t *inputs,
                       const int64_t *targets, size_t count) {
  bytevector_fixture input_vector;
  bytevector_fixture target_vector;
  encode_i64le(&input_vector, inputs, count);
  encode_i64le(&target_vector, targets, count);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(
            owner, generation, destination, &input_vector,
            (int64_t)(8u * count), &target_vector, (int64_t)(8u * count),
            (int64_t)count) == 0);
}

static void check_view(const et_kernel_tensor_view_v1 *view, const char *dtype,
                       size_t bytes) {
  CHECK(view != NULL);
  if (view == NULL) {
    return;
  }
  CHECK((uintptr_t)view % _Alignof(et_kernel_tensor_view_v1) == 0u);
  CHECK(view->struct_size == ET_KERNEL_TENSOR_VIEW_V1_0_SIZE);
  CHECK(view->data != NULL && view->byte_length == bytes);
  CHECK(strcmp(view->dtype, dtype) == 0 && strcmp(view->device, "cpu") == 0);
  CHECK(view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  CHECK(view->offset_bytes == 0u && view->rank == 2u);
  CHECK(view->shape != NULL && view->shape[0] == 2u && view->shape[1] == 3u);
  CHECK((uintptr_t)view->shape % _Alignof(uint64_t) == 0u);
  if (strcmp(dtype, "i64") == 0) {
    CHECK((uintptr_t)view->data % _Alignof(int64_t) == 0u);
  }
}

static void test_unpublished_cleanup(void) {
  int owners[8] = {0};
  int64_t stage;
  for (stage = ET_D2_TEST_FAIL_CONTROL; stage <= ET_D2_TEST_FAIL_BEFORE_PUBLISH;
       stage++) {
    et_d2_batch_test_fail_stage_v1(stage);
    CHECK(et_d2_batch_create_v1(&owners[stage], 2, 3, 102) == 0);
    CHECK(et_d2_batch_last_status_v1() ==
          ET_D2_NATIVE_STATUS_ALLOCATION_FAILED);
    CHECK(et_d2_batch_test_live_count_v1() == 0);
    CHECK(et_d2_batch_test_borrow_count_v1() == 0);
  }
  et_d2_batch_test_fail_stage_v1(ET_D2_TEST_FAIL_NONE);
}

static void test_i1_allocation_failures(void) {
  int owners[16] = {0};
  size_t allowed;
  for (allowed = 0u; allowed < 16u; allowed++) {
    int64_t generation;
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    generation = et_d2_batch_create_v1(&owners[allowed], 2, 3, 102);
    if (generation != 0) {
      CHECK(et_d2_batch_release_v1(&owners[allowed], generation) == 0);
    } else {
      CHECK(et_d2_batch_last_status_v1() ==
                ET_D2_NATIVE_STATUS_ALLOCATION_FAILED ||
            et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_INTERNAL);
    }
    CHECK(et_d2_batch_test_live_count_v1() == 0);
  }
  et_i64_tensor_test_reset_allocator_v1();
}

static void test_batch_lifetime(void) {
  int owner = 0;
  int64_t generation;
  int64_t stale_generation;
  int64_t lease;
  int64_t stale_lease;
  int64_t new_lease;
  const et_kernel_tensor_view_v1 *inputs;
  const et_kernel_tensor_view_v1 *targets;
  const et_kernel_tensor_view_v1 *mask;
  const int64_t input_a[] = {INT64_MIN, -1};
  const int64_t target_a[] = {INT64_MAX, 17};
  const int64_t input_b[] = {17, INT64_MAX};
  const int64_t target_b[] = {-1, INT64_MIN};
  const int64_t expected_inputs[] = {INT64_MIN, -1, 0, 0, 17, INT64_MAX};
  const int64_t expected_targets[] = {INT64_MAX, 17, 0, 0, -1, INT64_MIN};
  const uint8_t expected_mask[] = {1u, 1u, 0u, 0u, 1u, 1u};
  bytevector_fixture one;
  const int64_t zero = 0;

  CHECK(et_d2_batch_create_v1(&owner, 2, 3, 101) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_RANGE);
  CHECK(et_d2_batch_create_v1(&owner, 0, 3, 102) == 0);
  CHECK(et_d2_batch_create_v1(NULL, 2, 3, 102) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_create_v1(&owner, INT64_MAX, INT64_MAX, INT64_MAX) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_RANGE);

  generation = et_d2_batch_create_v1(&owner, 2, 3, 102);
  CHECK(generation > 0 && et_d2_batch_last_status_v1() == 0);
  CHECK(et_d2_batch_create_v1(&owner, 2, 3, 102) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_INVALID_STATE);
  CHECK(et_d2_batch_test_payload_bytes_v1(&owner, generation) == 102u);
  CHECK(et_d2_batch_test_control_bytes_v1() > 0u);
  CHECK(et_d2_batch_test_borrow_bytes_v1() == 0u);
  CHECK(et_d2_batch_test_view_metadata_bytes_v1() ==
        3u * sizeof(et_kernel_tensor_view_v1) + 6u * sizeof(uint64_t));
  CHECK(et_d2_batch_seal_v1(&owner, generation) ==
        ET_D2_NATIVE_STATUS_INVALID_STATE);

  encode_i64le(&one, &zero, 1u);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&owner, generation, 0, &one, 8,
                                             &one, 8,
                                             0) == ET_D2_NATIVE_STATUS_RANGE);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&owner, generation, 6, &one, 8,
                                             &one, 8,
                                             1) == ET_D2_NATIVE_STATUS_RANGE);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&owner, generation, 0, &one, 7,
                                             &one, 8,
                                             1) == ET_D2_NATIVE_STATUS_RANGE);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&owner, generation, 0, NULL, 8,
                                             &one, 8, 1) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  write_span(&owner, generation, 0, input_a, target_a, 2u);
  write_span(&owner, generation, 4, input_b, target_b, 2u);
  CHECK(et_d2_batch_seal_v1(&owner, generation) == 0);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&owner, generation, 0, &one, 8,
                                             &one, 8, 1) ==
        ET_D2_NATIVE_STATUS_INVALID_STATE);

  lease = et_d2_batch_borrow_begin_v1(&owner, generation);
  CHECK(lease > 0 && et_d2_batch_last_status_v1() == 0);
  CHECK(et_d2_batch_borrow_begin_v1(&owner, generation) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_INVALID_STATE);
  inputs = et_d2_batch_borrow_inputs_v1(&owner, generation, lease);
  targets = et_d2_batch_borrow_targets_v1(&owner, generation, lease);
  mask = et_d2_batch_borrow_loss_mask_v1(&owner, generation, lease);
  check_view(inputs, "i64", 6u * sizeof(int64_t));
  check_view(targets, "i64", 6u * sizeof(int64_t));
  check_view(mask, "bool", 6u);
  CHECK(memcmp(inputs->data, expected_inputs, sizeof(expected_inputs)) == 0);
  CHECK(memcmp(targets->data, expected_targets, sizeof(expected_targets)) == 0);
  CHECK(memcmp(mask->data, expected_mask, sizeof(expected_mask)) == 0);
  CHECK(et_d2_batch_release_v1(&owner, generation) ==
        ET_D2_NATIVE_STATUS_INVALID_STATE);

  stale_lease = lease;
  CHECK(et_d2_batch_borrow_end_v1(&owner, generation, lease) == 0);
  CHECK(et_d2_batch_borrow_inputs_v1(&owner, generation, stale_lease) == NULL);
  new_lease = et_d2_batch_borrow_begin_v1(&owner, generation);
  CHECK(new_lease > 0 && new_lease != stale_lease);
  CHECK(et_d2_batch_borrow_inputs_v1(&owner, generation, stale_lease) == NULL);
  CHECK(et_d2_batch_borrow_end_v1(&owner, generation, new_lease) == 0);
  stale_generation = generation;
  CHECK(et_d2_batch_release_v1(&owner, generation) == 0);
  CHECK(et_d2_batch_release_v1(&owner, generation) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_seal_v1(&owner, stale_generation) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(et_d2_batch_test_payload_bytes_v1(&owner, stale_generation) == 0u);
}

static void test_live_storage_rejection(void) {
  int first_owner = 0;
  int second_owner = 0;
  int64_t first_generation;
  int64_t second_generation;
  int64_t lease;
  const et_kernel_tensor_view_v1 *view;
  const int64_t values[] = {8, 1};
  bytevector_fixture vector;

  first_generation = et_d2_batch_create_v1(&first_owner, 1, 2, 34);
  write_span(&first_owner, first_generation, 0, values, values, 2u);
  CHECK(et_d2_batch_seal_v1(&first_owner, first_generation) == 0);
  lease = et_d2_batch_borrow_begin_v1(&first_owner, first_generation);
  view = et_d2_batch_borrow_inputs_v1(&first_owner, first_generation, lease);
  second_generation = et_d2_batch_create_v1(&second_owner, 1, 1, 17);
  encode_i64le(&vector, values, 1u);
  CHECK(et_d2_batch_write_pair_i64le_span_v1(&second_owner, second_generation,
                                             0, view->data, 8, &vector, 8, 1) ==
        ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
  CHECK(read_stage(et_d2_exact_read_v1("/dev/zero", 10, 0, view->data, 8)) ==
        ET_D2_EXACT_READ_ARGUMENT);
  CHECK(et_d2_batch_release_v1(&second_owner, second_generation) == 0);
  CHECK(et_d2_batch_borrow_end_v1(&first_owner, first_generation, lease) == 0);
  CHECK(et_d2_batch_release_v1(&first_owner, first_generation) == 0);
}

static void test_long_generation_loop(void) {
  int owner = 0;
  const int64_t value = 7;
  int64_t stale_generation = 0;
  int64_t stale_lease = 0;
  size_t iteration;
  for (iteration = 0u; iteration < 2048u; iteration++) {
    int64_t generation = et_d2_batch_create_v1(&owner, 1, 1, 17);
    int64_t lease;
    CHECK(generation > 0 && generation != stale_generation);
    if (stale_generation != 0) {
      CHECK(et_d2_batch_seal_v1(&owner, stale_generation) ==
            ET_D2_NATIVE_STATUS_INVALID_ARGUMENT);
    }
    write_span(&owner, generation, 0, &value, &value, 1u);
    CHECK(et_d2_batch_seal_v1(&owner, generation) == 0);
    lease = et_d2_batch_borrow_begin_v1(&owner, generation);
    CHECK(lease > 0 && lease != stale_lease);
    if (stale_lease != 0) {
      CHECK(et_d2_batch_borrow_inputs_v1(&owner, generation, stale_lease) ==
            NULL);
    }
    CHECK(et_d2_batch_borrow_end_v1(&owner, generation, lease) == 0);
    stale_generation = generation;
    stale_lease = lease;
    CHECK(et_d2_batch_release_v1(&owner, generation) == 0);
  }
  CHECK(et_d2_batch_test_live_count_v1() == 0);
  CHECK(et_d2_batch_test_borrow_count_v1() == 0);
}

static void test_generation_exhaustion(void) {
  int owner = 0;
  int64_t generation;
  const int64_t value = 1;
  generation = et_d2_batch_create_v1(&owner, 1, 1, 17);
  write_span(&owner, generation, 0, &value, &value, 1u);
  CHECK(et_d2_batch_seal_v1(&owner, generation) == 0);
  et_d2_batch_test_exhaust_generations_v1();
  CHECK(et_d2_batch_borrow_begin_v1(&owner, generation) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_RANGE);
  CHECK(et_d2_batch_release_v1(&owner, generation) == 0);
  CHECK(et_d2_batch_create_v1(&owner, 1, 1, 17) == 0);
  CHECK(et_d2_batch_last_status_v1() == ET_D2_NATIVE_STATUS_RANGE);
}

int main(void) {
  test_exact_read();
  test_unpublished_cleanup();
  test_i1_allocation_failures();
  test_batch_lifetime();
  test_live_storage_rejection();
  test_long_generation_loop();
  test_generation_exhaustion();
  if (failures != 0) {
    (void)fprintf(stderr, "D2 native FAIL: %d of %d checks failed\n", failures,
                  checks);
    return 1;
  }
  (void)printf(
      "D2 native bytes: control=%zu dynamic-borrow=%zu view-metadata=%zu "
      "payload[2,3]=102\n",
      et_d2_batch_test_control_bytes_v1(), et_d2_batch_test_borrow_bytes_v1(),
      et_d2_batch_test_view_metadata_bytes_v1());
  (void)printf("D2 native PASS: %d exact-range, carrier, and lifetime checks\n",
               checks);
  return 0;
}
