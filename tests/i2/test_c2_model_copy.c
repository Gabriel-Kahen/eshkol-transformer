#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t et_c2_private_i2_state_owned_copy_bytes_v1(
    void *owned_carrier, void *destination_bytevector_header,
    int64_t exact_byte_count);
int64_t et_i2_private_last_error_category_v1(void);
int64_t et_i2_private_last_error_code_v1(void);
void *et_i2_private_copy_builder_create_v1(int64_t count);
int64_t et_i2_private_copy_builder_abort_v1(void *builder);
void et_i2_test_fail_alloc_after_v1(size_t allowed);
void et_i2_test_reset_allocator_v1(void);
void et_i2_test_live_builder_counts_v1(size_t *copy_count,
                                        size_t *reset_count,
                                        size_t *decode_count);
void et_i2_test_retired_builder_counts_v1(size_t *copy_count,
                                           size_t *reset_count,
                                           size_t *decode_count,
                                           size_t *retained_control_bytes);

typedef struct bytevector {
  int64_t length;
  uint32_t payload[];
} bytevector;

typedef struct seam_counts {
  et_f32_test_live_counts_v1 live;
  et_f32_test_retired_counts_v1 retired;
  et_f32_test_borrow_event_counts_v1 borrow_events;
  size_t live_copy_builders;
  size_t live_reset_builders;
  size_t live_decode_builders;
  size_t retired_copy_builders;
  size_t retired_reset_builders;
  size_t retired_decode_builders;
  size_t bridge_retained_control_bytes;
} seam_counts;

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                   \
    if (!(condition)) {                                                         \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
      failures++;                                                               \
    }                                                                           \
  } while (0)

static bytevector *bytevector_create(size_t bytes) {
  bytevector *result;
  if (bytes > SIZE_MAX - sizeof(*result) || bytes > INT64_MAX) {
    return NULL;
  }
  result = (bytevector *)malloc(sizeof(*result) + bytes);
  if (result != NULL) {
    result->length = (int64_t)bytes;
    memset(result->payload, 0xa5, bytes);
  }
  return result;
}

static et_f32_tensor *owned_tensor_create(size_t rank, const uint64_t *shape,
                                           const uint32_t *bits,
                                           size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *source = NULL;
  et_f32_tensor *owned = NULL;
  CHECK(et_f32_tensor_create_v1(rank, shape, &source, &error) == 0);
  if (source == NULL) {
    return NULL;
  }
  CHECK(et_f32_tensor_copy_bits_from_v1(source, bits, count, &error) == 0);
  CHECK(et_f32_owned_tensor_clone_v1(source, &owned, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&source, &error) == 0);
  return owned;
}

static void owned_tensor_release(et_f32_tensor *owned) {
  et_f32_tensor_error error;
  CHECK(et_f32_owned_tensor_release_v1(owned, &error) == 0);
}

static seam_counts snapshot(void) {
  seam_counts result = {
      .live = {.struct_size = sizeof(result.live)},
      .retired = {.struct_size = sizeof(result.retired)},
      .borrow_events = {.struct_size = sizeof(result.borrow_events)},
  };
  et_f32_test_live_counts_snapshot_v1(&result.live);
  et_f32_test_retired_counts_snapshot_v1(&result.retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&result.borrow_events);
  et_i2_test_live_builder_counts_v1(
      &result.live_copy_builders, &result.live_reset_builders,
      &result.live_decode_builders);
  et_i2_test_retired_builder_counts_v1(
      &result.retired_copy_builders, &result.retired_reset_builders,
      &result.retired_decode_builders,
      &result.bridge_retained_control_bytes);
  return result;
}

static void check_counts_equal(seam_counts left, seam_counts right) {
  CHECK(left.live.tensors == right.live.tensors);
  CHECK(left.live.parameters == right.live.parameters);
  CHECK(left.live.borrows == right.live.borrows);
  CHECK(left.live.copy_plans == right.live.copy_plans);
  CHECK(left.live.gradient_plans == right.live.gradient_plans);
  CHECK(left.live.reset_plans == right.live.reset_plans);
  CHECK(left.live.owned_clones == right.live.owned_clones);
  CHECK(left.retired.tensors == right.retired.tensors);
  CHECK(left.retired.parameters == right.retired.parameters);
  CHECK(left.retired.borrows == right.retired.borrows);
  CHECK(left.retired.copy_plans == right.retired.copy_plans);
  CHECK(left.retired.gradient_plans == right.retired.gradient_plans);
  CHECK(left.retired.reset_plans == right.retired.reset_plans);
  CHECK(left.retired.retained_control_bytes ==
        right.retired.retained_control_bytes);
  CHECK(left.borrow_events.begin_calls == right.borrow_events.begin_calls);
  CHECK(left.borrow_events.view_calls == right.borrow_events.view_calls);
  CHECK(left.borrow_events.end_calls == right.borrow_events.end_calls);
  CHECK(left.live_copy_builders == right.live_copy_builders);
  CHECK(left.live_reset_builders == right.live_reset_builders);
  CHECK(left.live_decode_builders == right.live_decode_builders);
  CHECK(left.retired_copy_builders == right.retired_copy_builders);
  CHECK(left.retired_reset_builders == right.retired_reset_builders);
  CHECK(left.retired_decode_builders == right.retired_decode_builders);
  CHECK(left.bridge_retained_control_bytes ==
        right.bridge_retained_control_bytes);
}

static void test_exact_bits(void) {
  static const uint32_t scalar_bits[] = {UINT32_C(0x80000000)};
  static const uint32_t matrix_bits[] = {
      UINT32_C(0x00000001), UINT32_C(0x7f800000),
      UINT32_C(0xff800000), UINT32_C(0x7fc12345),
      UINT32_C(0x3f800000), UINT32_C(0xbf800000),
  };
  const uint64_t matrix_shape[] = {2u, 3u};
  et_f32_tensor *scalar = owned_tensor_create(0u, NULL, scalar_bits, 1u);
  et_f32_tensor *matrix =
      owned_tensor_create(2u, matrix_shape, matrix_bits, 6u);
  bytevector *scalar_output = bytevector_create(sizeof(scalar_bits));
  bytevector *matrix_output = bytevector_create(sizeof(matrix_bits));

  CHECK(scalar != NULL && matrix != NULL);
  CHECK(scalar_output != NULL && matrix_output != NULL);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            scalar, scalar_output, (int64_t)sizeof(scalar_bits)) == 0);
  CHECK(memcmp(scalar_output->payload, scalar_bits, sizeof(scalar_bits)) == 0);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            matrix, matrix_output, (int64_t)sizeof(matrix_bits)) == 0);
  CHECK(memcmp(matrix_output->payload, matrix_bits, sizeof(matrix_bits)) == 0);

  free(matrix_output);
  free(scalar_output);
  owned_tensor_release(matrix);
  owned_tensor_release(scalar);
}

static void test_zero_element(void) {
  const uint64_t shape[] = {0u};
  et_f32_tensor *owned = owned_tensor_create(1u, shape, NULL, 0u);
  bytevector *output = bytevector_create(0u);
  CHECK(owned != NULL && output != NULL);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, 0) == 0);
  free(output);
  owned_tensor_release(owned);
}

static void test_exact_eight_mib(void) {
  const size_t bytes = 8u * 1024u * 1024u;
  const size_t count = bytes / sizeof(uint32_t);
  const uint64_t shape[] = {(uint64_t)count};
  uint32_t *bits = (uint32_t *)malloc(bytes);
  bytevector *output = bytevector_create(bytes);
  et_f32_tensor *owned;
  CHECK(bits != NULL && output != NULL);
  if (bits == NULL || output == NULL) {
    free(output);
    free(bits);
    return;
  }
  for (size_t index = 0u; index < count; index++) {
    bits[index] = UINT32_C(0x9e3779b9) ^ (uint32_t)index;
  }
  owned = owned_tensor_create(1u, shape, bits, count);
  CHECK(owned != NULL);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, output, (int64_t)bytes) == 0);
  CHECK(memcmp(output->payload, bits, bytes) == 0);
  owned_tensor_release(owned);
  free(output);
  free(bits);
}

static void check_canary(const bytevector *output, uint32_t canary,
                         size_t count) {
  for (size_t index = 0u; index < count; index++) {
    CHECK(output->payload[index] == canary);
  }
}

static void test_rejections_are_atomic(void) {
  static const uint32_t bits[] = {
      UINT32_C(0x01234567), UINT32_C(0x89abcdef),
      UINT32_C(0x76543210), UINT32_C(0xfedcba98),
  };
  const uint64_t shape[] = {4u};
  const uint64_t alias_shape[] = {6u};
  const uint32_t canary = UINT32_C(0xa5a5a5a5);
  uint32_t alias_bits[6] = {0u};
  uint32_t alias_after[6] = {0u};
  const int64_t alias_length = (int64_t)sizeof(bits);
  et_f32_tensor *owned = owned_tensor_create(1u, shape, bits, 4u);
  et_f32_tensor *alias_tensor = NULL;
  et_f32_tensor_error error;
  bytevector *output = bytevector_create(sizeof(bits));
  unsigned char *misaligned_storage =
      (unsigned char *)malloc(sizeof(*output) + sizeof(bits) + 1u);
  void *builder = et_i2_private_copy_builder_create_v1(1);

  CHECK(owned != NULL && output != NULL && misaligned_storage != NULL);
  CHECK(builder != NULL);
  memcpy(alias_bits, &alias_length, sizeof(alias_length));
  alias_bits[2] = canary;
  alias_bits[3] = canary;
  alias_bits[4] = canary;
  alias_bits[5] = canary;
  CHECK(et_f32_tensor_create_v1(1u, alias_shape, &alias_tensor, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(alias_tensor, alias_bits, 6u, &error) ==
        0);
  memset(output->payload, 0xa5, sizeof(bits));
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, -1) == -1);
  CHECK(et_i2_private_last_error_category_v1() ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(et_i2_private_last_error_code_v1() ==
        ET_F32_TENSOR_CODE_INVALID_BUFFER);
  check_canary(output, canary, 4u);

  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, 15) == -1);
  check_canary(output, canary, 4u);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, 17) == -1);
  check_canary(output, canary, 4u);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, 12) == -1);
  CHECK(et_i2_private_last_error_category_v1() ==
        ET_F32_TENSOR_ERROR_SHAPE_MISMATCH);
  CHECK(et_i2_private_last_error_code_v1() ==
        ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
  check_canary(output, canary, 4u);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(owned, output, 20) == -1);
  check_canary(output, canary, 4u);

  output->length = 12;
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, output, (int64_t)sizeof(bits)) == -1);
  check_canary(output, canary, 4u);
  output->length = (int64_t)sizeof(bits);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, NULL, (int64_t)sizeof(bits)) == -1);
  check_canary(output, canary, 4u);

  memcpy(misaligned_storage + 1u, &output->length, sizeof(output->length));
  memset(misaligned_storage + 1u + sizeof(output->length), 0xa5,
         sizeof(bits));
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, misaligned_storage + 1u, (int64_t)sizeof(bits)) == -1);
  CHECK(memcmp(misaligned_storage + 1u + sizeof(output->length),
               output->payload, sizeof(bits)) == 0);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, (void *)(UINTPTR_MAX - 7u),
            (int64_t)sizeof(bits)) == -1);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, builder, (int64_t)sizeof(bits)) == -1);
  CHECK(et_i2_private_copy_builder_abort_v1(builder) == 0);
  CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
            owned, (void *)et_f32_tensor_test_data_storage_v1(alias_tensor),
            (int64_t)sizeof(bits)) == -1);
  CHECK(et_i2_private_last_error_category_v1() ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(et_i2_private_last_error_code_v1() ==
        ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(et_f32_tensor_copy_bits_to_v1(alias_tensor, alias_after, 6u, &error) ==
        0);
  CHECK(memcmp(alias_after, alias_bits, sizeof(alias_bits)) == 0);
  CHECK(et_f32_tensor_destroy_v1(&alias_tensor, &error) == 0);

  free(misaligned_storage);
  free(output);
  owned_tensor_release(owned);
}

static void test_no_allocation_or_control_churn(void) {
  static const uint32_t bits[] = {
      UINT32_C(0x3f000000), UINT32_C(0xbf000000),
  };
  const uint64_t shape[] = {2u};
  et_f32_tensor *owned = owned_tensor_create(1u, shape, bits, 2u);
  bytevector *output = bytevector_create(sizeof(bits));
  seam_counts before;
  seam_counts after;

  CHECK(owned != NULL && output != NULL);
  before = snapshot();
  et_f32_tensor_test_fail_alloc_after_v1(0u);
  et_i2_test_fail_alloc_after_v1(0u);
  for (size_t iteration = 0u; iteration < 100u; iteration++) {
    CHECK(et_c2_private_i2_state_owned_copy_bytes_v1(
              owned, output, (int64_t)sizeof(bits)) == 0);
  }
  after = snapshot();
  check_counts_equal(before, after);
  CHECK(after.live.borrows == before.live.borrows);
  CHECK(after.retired.borrows == before.retired.borrows);
  CHECK(after.retired.retained_control_bytes ==
        before.retired.retained_control_bytes);
  CHECK(after.bridge_retained_control_bytes ==
        before.bridge_retained_control_bytes);
  et_i2_test_reset_allocator_v1();
  et_f32_tensor_test_reset_allocator_v1();
  free(output);
  owned_tensor_release(owned);
}

int main(void) {
  test_exact_bits();
  test_zero_element();
  test_exact_eight_mib();
  test_rejections_are_atomic();
  test_no_allocation_or_control_churn();
  if (failures != 0) {
    (void)fprintf(stderr, "C2 I2 model copy FAIL: %d/%d checks\n", failures,
                  checks);
    return 1;
  }
  (void)printf("C2 I2 model copy PASS: %d checks\n", checks);
  return 0;
}
