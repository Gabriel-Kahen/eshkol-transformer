#define _GNU_SOURCE

#include "o2_optimizer_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *et_c2_private_o2_state_reconstruct_create_v1(
    int64_t parameter_count, int64_t clip_kind, int64_t clip_max_bits,
    int64_t schedule_kind, int64_t warmup_updates, int64_t total_updates,
    int64_t minimum_ratio_bits, int64_t completed_updates);
int64_t et_c2_private_o2_state_reconstruct_set_v1(
    void *builder, int64_t index, int64_t rank, void *shape_bytevector_header,
    int64_t element_count, void *exp_avg_bytevector_header,
    void *exp_avg_sq_bytevector_header, int64_t learning_rate_bits,
    int64_t beta1_bits, int64_t beta2_bits, int64_t epsilon_bits,
    int64_t weight_decay_bits);
int64_t et_c2_private_o2_state_reconstruct_prepare_v1(void *builder);
void *et_c2_private_o2_state_reconstruct_commit_v1(void *builder);
int64_t et_c2_private_o2_state_reconstruct_abort_v1(void *builder);
int64_t et_c2_private_o2_state_copy_moment_bits_v1(
    void *state, int64_t index, int64_t moment_kind,
    void *destination_bytevector_header, int64_t element_count);
int64_t et_o2_private_last_error_category_v1(void);

static unsigned checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct test_bytevector {
  int64_t length;
  unsigned char bytes[];
} test_bytevector;

static test_bytevector *make_bytevector(size_t bytes, unsigned char fill) {
  test_bytevector *value =
      (test_bytevector *)malloc(sizeof(*value) + (bytes == 0u ? 1u : bytes));
  CHECK(value != NULL);
  value->length = (int64_t)bytes;
  memset(value->bytes, fill, bytes);
  return value;
}

static void store_u32(test_bytevector *value, size_t index, uint32_t bits) {
  memcpy(value->bytes + index * sizeof(bits), &bits, sizeof(bits));
}

static void store_u64_le(test_bytevector *value, size_t index,
                         uint64_t extent) {
  size_t byte;
  for (byte = 0u; byte < sizeof(extent); ++byte) {
    value->bytes[index * sizeof(extent) + byte] =
        (unsigned char)(extent >> (byte * 8u));
  }
}

static void expect_bits(const test_bytevector *value, size_t index,
                        uint32_t expected) {
  uint32_t actual = 0u;
  memcpy(&actual, value->bytes + index * sizeof(actual), sizeof(actual));
  CHECK(actual == expected);
}

static void test_round_trip_and_validation(void) {
  static const uint32_t avg_bits[] = {
      UINT32_C(0x80000000), UINT32_C(0x00000001), UINT32_C(0x3f000000),
      UINT32_C(0xbf000000)};
  static const uint32_t sq_bits[] = {UINT32_C(0x00000000), UINT32_C(0x00800000),
                                     UINT32_C(0x3e800000),
                                     UINT32_C(0x3f800000)};
  test_bytevector *shape = make_bytevector(16u, 0u);
  test_bytevector *avg = make_bytevector(16u, 0u);
  test_bytevector *sq = make_bytevector(16u, 0u);
  test_bytevector *destination = make_bytevector(16u, 0xa5u);
  unsigned char *misaligned_storage = (unsigned char *)malloc(25u);
  void *misaligned = misaligned_storage + 1u;
  int64_t encoded_length = 16;
  void *builder;
  void *state;
  size_t index;

  store_u64_le(shape, 0u, 2u);
  store_u64_le(shape, 1u, 2u);
  for (index = 0u; index < 4u; ++index) {
    store_u32(avg, index, avg_bits[index]);
    store_u32(sq, index, sq_bits[index]);
  }
  CHECK(misaligned_storage != NULL);
  memcpy(misaligned, &encoded_length, sizeof(encoded_length));
  memset((unsigned char *)misaligned + sizeof(encoded_length), 0, 16u);

  CHECK(et_c2_private_o2_state_reconstruct_create_v1(
            -1, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT, 0, 0,
            UINT32_C(0x3f800000), 0) == NULL);
  CHECK(et_o2_private_last_error_category_v1() ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_c2_private_o2_state_reconstruct_create_v1(
            1, INT64_C(4294967296), 0, ET_O2_SCHEDULE_CONSTANT, 0, 0,
            UINT32_C(0x3f800000), 0) == NULL);
  builder = et_c2_private_o2_state_reconstruct_create_v1(
      1, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT, 0, 0,
      UINT32_C(0x3f800000), 17);
  CHECK(builder != NULL);

  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, (void *)(uintptr_t)(UINTPTR_MAX - 4u), 4, avg, sq,
            UINT32_C(0x3dcccccd), UINT32_C(0x3f000000), UINT32_C(0x3f000000),
            UINT32_C(0x3a83126f), 0) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, shape, 4, misaligned, sq, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == ET_O2_STATUS_INVALID_ARGUMENT);
  shape->length = 8;
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, shape, 4, avg, sq, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == ET_O2_STATUS_INVALID_ARGUMENT);
  shape->length = 16;
  store_u64_le(shape, 1u, 3u);
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, shape, 4, avg, sq, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == ET_O2_STATUS_SHAPE_MISMATCH);
  store_u64_le(shape, 1u, 2u);
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, shape, 4, avg, avg, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 2, shape, 4, avg, sq, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == 0);
  CHECK(et_c2_private_o2_state_reconstruct_prepare_v1(builder) == 0);
  et_o2_test_fail_reconstruct_commit_v1(1);
  CHECK(et_c2_private_o2_state_reconstruct_commit_v1(builder) == NULL);
  CHECK(et_o2_private_last_error_category_v1() == ET_O2_STATUS_INTERNAL);
  et_o2_test_fail_reconstruct_commit_v1(0);
  state = et_c2_private_o2_state_reconstruct_commit_v1(builder);
  CHECK(state != NULL);

  destination->length = 12;
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, ET_O2_MOMENT_EXP_AVG, destination, 4) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(destination->bytes[0] == 0xa5u);
  destination->length = 16;
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, 2, destination, 4) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(destination->bytes[0] == 0xa5u);
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, ET_O2_MOMENT_EXP_AVG, misaligned, 4) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  et_o2_test_fail_alloc_after_v1(0u);
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, ET_O2_MOMENT_EXP_AVG, destination, 4) == 0);
  et_o2_test_reset_failpoints_v1();
  for (index = 0u; index < 4u; ++index) {
    expect_bits(destination, index, avg_bits[index]);
  }
  memset(destination->bytes, 0, 16u);
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, ET_O2_MOMENT_EXP_AVG_SQ, destination, 4) == 0);
  for (index = 0u; index < 4u; ++index) {
    expect_bits(destination, index, sq_bits[index]);
  }
  CHECK(et_o2_optimizer_state_release_v1((et_o2_optimizer_state *)state,
                                         NULL) == 0);
  free(misaligned_storage);
  free(destination);
  free(sq);
  free(avg);
  free(shape);
}

static void test_abort_and_zero_element(void) {
  test_bytevector *shape = make_bytevector(8u, 0u);
  test_bytevector *empty_avg = make_bytevector(0u, 0u);
  test_bytevector *empty_sq = make_bytevector(0u, 0u);
  test_bytevector *empty_destination = make_bytevector(0u, 0u);
  void *builder;
  void *state;

  store_u64_le(shape, 0u, 0u);
  builder = et_c2_private_o2_state_reconstruct_create_v1(
      1, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT, 0, 0,
      UINT32_C(0x3f800000), 0);
  CHECK(builder != NULL);
  CHECK(et_c2_private_o2_state_reconstruct_set_v1(
            builder, 0, 1, shape, 0, empty_avg, empty_sq, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f000000), UINT32_C(0x3f000000), UINT32_C(0x3a83126f),
            0) == 0);
  CHECK(et_c2_private_o2_state_reconstruct_prepare_v1(builder) == 0);
  state = et_c2_private_o2_state_reconstruct_commit_v1(builder);
  CHECK(state != NULL);
  CHECK(et_c2_private_o2_state_copy_moment_bits_v1(
            state, 0, ET_O2_MOMENT_EXP_AVG, empty_destination, 0) == 0);
  CHECK(et_o2_optimizer_state_release_v1((et_o2_optimizer_state *)state,
                                         NULL) == 0);

  builder = et_c2_private_o2_state_reconstruct_create_v1(
      1, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT, 0, 0,
      UINT32_C(0x3f800000), 0);
  CHECK(builder != NULL);
  CHECK(et_c2_private_o2_state_reconstruct_abort_v1(builder) == 0);
  CHECK(et_c2_private_o2_state_reconstruct_abort_v1(NULL) == 0);
  free(empty_destination);
  free(empty_sq);
  free(empty_avg);
  free(shape);
}

int main(void) {
  et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_o2_test_reset_failpoints_v1();
  et_o2_test_live_counts_snapshot_v1(&before);
  test_round_trip_and_validation();
  test_abort_and_zero_element();
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.reconstruct_builders == before.reconstruct_builders);
  CHECK(after.owned_state_clones == before.owned_state_clones);
  CHECK(after.live_states == before.live_states);
  printf("C2 O2 reconstruction bridge PASS: %u checks\n", checks);
  return 0;
}
