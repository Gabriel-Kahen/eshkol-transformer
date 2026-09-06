#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>

void *et_i2_private_decode_builder_create_v1(int64_t rank, void *payload,
                                              int64_t payload_size);
int64_t et_i2_private_decode_builder_set_v1(void *builder, int64_t index,
                                             int64_t extent);
void *et_i2_private_decode_builder_finish_v1(void *builder);
int64_t et_i2_private_decode_builder_abort_v1(void *builder);
void *et_i2_private_copy_builder_create_v1(int64_t count);
int64_t et_i2_private_copy_builder_set_v1(
    void *builder, int64_t index, void *destination, int64_t destination_role,
    void *destination_handle, void *source, int64_t source_role,
    void *source_handle);
int64_t et_i2_private_copy_builder_prepare_v1(void *builder);
int64_t et_i2_private_copy_builder_commit_v1(void *builder);
int64_t et_i2_private_copy_builder_abort_v1(void *builder);
void *et_i2_private_reset_builder_create_v1(int64_t count);
int64_t et_i2_private_reset_builder_set_v1(void *builder, int64_t index,
                                            void *parameter);
int64_t et_i2_private_reset_builder_prepare_v1(void *builder);
int64_t et_i2_private_reset_builder_commit_v1(void *builder);
int64_t et_i2_private_reset_builder_abort_v1(void *builder);
int64_t et_i2_private_owned_release_v1(void *owned);
int64_t et_i2_private_tensor_copy_bytes_v1(void *carrier, int64_t role,
                                            void *p1_handle,
                                            void *destination,
                                            int64_t destination_size);
int64_t et_i2_private_parameter_accumulate_uniform_v1(
    void *parameter, void *p1_handle, int64_t bits, int64_t expected_ordinal,
    int64_t weight_bits);
void *et_i2_private_tensor_create_uniform_v1(int64_t count, int64_t bits);
#ifdef ET_F32_TENSOR_TESTING
void et_i2_test_fail_alloc_after_v1(size_t allowed);
void et_i2_test_reset_allocator_v1(void);
void et_i2_test_live_builder_counts_v1(size_t *copy_count,
                                        size_t *reset_count,
                                        size_t *decode_count);
#endif

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

static void check_builder_bytevector_alias_rejected(void *retired,
                                                    et_f32_tensor *empty) {
  CHECK(et_i2_private_decode_builder_create_v1(0, retired, 0) == NULL);
  CHECK(et_i2_private_tensor_copy_bytes_v1(empty, 2, NULL, retired, 0) == -1);
}

#ifdef ET_F32_TENSOR_TESTING
typedef struct bridge_live_snapshot {
  et_f32_test_live_counts_v1 native;
  size_t copy_builders;
  size_t reset_builders;
  size_t decode_builders;
} bridge_live_snapshot;

static bridge_live_snapshot bridge_snapshot(void) {
  bridge_live_snapshot snapshot = {
      .native = {.struct_size = sizeof(snapshot.native)},
  };
  et_f32_test_live_counts_snapshot_v1(&snapshot.native);
  et_i2_test_live_builder_counts_v1(&snapshot.copy_builders,
                                     &snapshot.reset_builders,
                                     &snapshot.decode_builders);
  return snapshot;
}

static void check_bridge_snapshot(bridge_live_snapshot expected) {
  bridge_live_snapshot actual = bridge_snapshot();
  CHECK(actual.native.tensors == expected.native.tensors);
  CHECK(actual.native.parameters == expected.native.parameters);
  CHECK(actual.native.borrows == expected.native.borrows);
  CHECK(actual.native.copy_plans == expected.native.copy_plans);
  CHECK(actual.native.gradient_plans == expected.native.gradient_plans);
  CHECK(actual.native.reset_plans == expected.native.reset_plans);
  CHECK(actual.native.owned_clones == expected.native.owned_clones);
  CHECK(actual.copy_builders == expected.copy_builders);
  CHECK(actual.reset_builders == expected.reset_builders);
  CHECK(actual.decode_builders == expected.decode_builders);
}

static void test_bridge_allocation_failpoints(
    et_f32_tensor *source, et_f32_tensor *destination,
    et_f32_parameter *parameter, void *identity, void *bytevector) {
  bridge_live_snapshot baseline = bridge_snapshot();
  et_f32_gradient_metadata_v1 metadata = {
      .struct_size = sizeof(metadata),
  };
  et_f32_tensor_error error;

  for (size_t allowed = 0u; allowed <= 1u; allowed++) {
    et_f32_tensor *tensor;
    et_i2_test_fail_alloc_after_v1(allowed);
    tensor = (et_f32_tensor *)et_i2_private_tensor_create_uniform_v1(
        1, INT64_C(1065353216));
    if (allowed == 0u) {
      CHECK(tensor == NULL);
    } else {
      CHECK(tensor != NULL);
    }
    et_i2_test_reset_allocator_v1();
    if (tensor != NULL) {
      CHECK(et_f32_tensor_destroy_v1(&tensor, &error) == 0);
    }
    check_bridge_snapshot(baseline);
  }

  for (size_t allowed = 0u; allowed <= 3u; allowed++) {
    void *builder;
    et_i2_test_fail_alloc_after_v1(allowed);
    builder = et_i2_private_decode_builder_create_v1(1, bytevector, 4);
    if (allowed < 3u) {
      CHECK(builder == NULL);
    } else {
      CHECK(builder != NULL);
    }
    et_i2_test_reset_allocator_v1();
    if (builder != NULL) {
      CHECK(et_i2_private_decode_builder_abort_v1(builder) == 0);
    }
    check_bridge_snapshot(baseline);
  }

  for (size_t allowed = 0u; allowed <= 2u; allowed++) {
    void *builder;
    et_i2_test_fail_alloc_after_v1(allowed);
    builder = et_i2_private_copy_builder_create_v1(1);
    if (allowed < 2u) {
      CHECK(builder == NULL);
    } else {
      CHECK(builder != NULL);
    }
    et_i2_test_reset_allocator_v1();
    if (builder != NULL) {
      CHECK(et_i2_private_copy_builder_set_v1(
                builder, 0, destination, 2, NULL, source, 2, NULL) == 0);
      CHECK(et_i2_private_copy_builder_abort_v1(builder) == 0);
    }
    check_bridge_snapshot(baseline);
  }

  for (size_t allowed = 0u; allowed <= 2u; allowed++) {
    void *builder;
    et_i2_test_fail_alloc_after_v1(allowed);
    builder = et_i2_private_reset_builder_create_v1(1);
    if (allowed < 2u) {
      CHECK(builder == NULL);
    } else {
      CHECK(builder != NULL);
    }
    et_i2_test_reset_allocator_v1();
    if (builder != NULL) {
      CHECK(et_i2_private_reset_builder_set_v1(builder, 0, parameter) == 0);
      CHECK(et_i2_private_reset_builder_abort_v1(builder) == 0);
    }
    check_bridge_snapshot(baseline);
  }

  et_i2_test_fail_alloc_after_v1(0u);
  CHECK(et_i2_private_parameter_accumulate_uniform_v1(
            parameter, identity, 0, 0, INT64_C(1065353216)) == -1);
  et_i2_test_reset_allocator_v1();
  check_bridge_snapshot(baseline);
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
        0);
  CHECK(metadata.state == ET_F32_GRADIENT_ABSENT);
  CHECK(metadata.contribution_count == 0u);
  CHECK(metadata.normalization_weight_bits == 0u);
}
#endif

int main(void) {
  const uint64_t shape[1] = {1u};
  const uint64_t empty_shape[1] = {0u};
  const uint32_t source_bits[1] = {UINT32_C(0x3f800000)};
  et_f32_tensor_error error;
  et_f32_tensor *source = NULL;
  et_f32_tensor *destination = NULL;
  et_f32_tensor *empty = NULL;
  et_f32_parameter *parameter = NULL;
  void *copy_builder;
  void *reset_builder;
  void *decode_builder;
  void *stale_builder;
  et_f32_tensor *decoded;
  int identity = 0;
  struct {
    int64_t length;
    uint32_t bits;
  } bytevector = {4, UINT32_C(0x40000000)};
#ifdef ET_F32_TENSOR_TESTING
  bridge_live_snapshot repeated_baseline;
#endif

  CHECK(et_f32_tensor_create_v1(1u, shape, &source, &error) == 0);
  CHECK(et_f32_tensor_create_v1(1u, shape, &destination, &error) == 0);
  CHECK(et_f32_tensor_create_v1(1u, empty_shape, &empty, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(source, source_bits, 1u, &error) == 0);
  CHECK(et_f32_parameter_create_v1(source, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);

#ifdef ET_F32_TENSOR_TESTING
  test_bridge_allocation_failpoints(source, destination, parameter, &identity,
                                    &bytevector);
#endif

  copy_builder = et_i2_private_copy_builder_create_v1(1);
  CHECK(copy_builder != NULL);
  CHECK(et_i2_private_copy_builder_set_v1(copy_builder, 0, destination, 2,
                                           NULL, source, 2, NULL) == 0);
  CHECK(et_i2_private_copy_builder_prepare_v1(copy_builder) == 0);
  CHECK(et_i2_private_copy_builder_commit_v1(copy_builder) == 0);
  CHECK(et_i2_private_copy_builder_commit_v1(copy_builder) == -1);
  CHECK(et_i2_private_copy_builder_abort_v1(copy_builder) == -1);
  CHECK(et_i2_private_copy_builder_commit_v1((void *)(uintptr_t)1u) == -1);

  copy_builder = et_i2_private_copy_builder_create_v1(1);
  CHECK(copy_builder != NULL);
  CHECK(et_i2_private_copy_builder_abort_v1(copy_builder) == 0);
  CHECK(et_i2_private_copy_builder_abort_v1(copy_builder) == -1);

  reset_builder = et_i2_private_reset_builder_create_v1(1);
  CHECK(reset_builder != NULL);
  CHECK(et_i2_private_parameter_accumulate_uniform_v1(
            parameter, &identity, 0, 0, INT64_C(1065353216)) == 0);
  CHECK(et_i2_private_reset_builder_set_v1(reset_builder, 0, parameter) == 0);
  CHECK(et_i2_private_reset_builder_prepare_v1(reset_builder) == 0);
  CHECK(et_i2_private_reset_builder_commit_v1(reset_builder) == 0);
  CHECK(et_i2_private_reset_builder_commit_v1(reset_builder) == -1);
  CHECK(et_i2_private_reset_builder_abort_v1(reset_builder) == -1);

  reset_builder = et_i2_private_reset_builder_create_v1(1);
  CHECK(reset_builder != NULL);
  CHECK(et_i2_private_reset_builder_abort_v1(reset_builder) == 0);
  CHECK(et_i2_private_reset_builder_abort_v1(reset_builder) == -1);

  decode_builder = et_i2_private_decode_builder_create_v1(1, &bytevector, 4);
  CHECK(decode_builder != NULL);
  CHECK(et_i2_private_decode_builder_set_v1(decode_builder, 0, 1) == 0);
  decoded = (et_f32_tensor *)
      et_i2_private_decode_builder_finish_v1(decode_builder);
  CHECK(decoded != NULL);
  CHECK(et_i2_private_decode_builder_finish_v1(decode_builder) == NULL);
  CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == -1);
  CHECK(et_i2_private_owned_release_v1(decoded) == 0);

  decode_builder = et_i2_private_decode_builder_create_v1(1, &bytevector, 4);
  CHECK(decode_builder != NULL);
  CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == 0);
  CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == -1);

#ifdef ET_F32_TENSOR_TESTING
  decode_builder = et_i2_private_decode_builder_create_v1(1, &bytevector, 4);
  CHECK(decode_builder != NULL);
  CHECK(et_i2_private_decode_builder_set_v1(decode_builder, 0, 1) == 0);
  et_f32_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_i2_private_decode_builder_finish_v1(decode_builder) == NULL);
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == 0);
  CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == -1);
#endif

#ifdef ET_F32_TENSOR_TESTING
  repeated_baseline = bridge_snapshot();
#endif
  stale_builder = et_i2_private_copy_builder_create_v1(1);
  CHECK(stale_builder != NULL);
  CHECK(et_i2_private_copy_builder_abort_v1(stale_builder) == 0);
  check_builder_bytevector_alias_rejected(stale_builder, empty);
#ifdef ET_F32_TENSOR_TESTING
  check_bridge_snapshot(repeated_baseline);
#endif
  for (size_t repetition = 0u; repetition < 32u; repetition++) {
    copy_builder = et_i2_private_copy_builder_create_v1(1);
    CHECK(copy_builder != NULL);
    CHECK(copy_builder != stale_builder);
    CHECK(et_i2_private_copy_builder_abort_v1(stale_builder) == -1);
    CHECK(et_i2_private_copy_builder_abort_v1(copy_builder) == 0);
#ifdef ET_F32_TENSOR_TESTING
    check_bridge_snapshot(repeated_baseline);
#endif
  }

  stale_builder = et_i2_private_reset_builder_create_v1(1);
  CHECK(stale_builder != NULL);
  CHECK(et_i2_private_reset_builder_abort_v1(stale_builder) == 0);
  check_builder_bytevector_alias_rejected(stale_builder, empty);
#ifdef ET_F32_TENSOR_TESTING
  check_bridge_snapshot(repeated_baseline);
#endif
  for (size_t repetition = 0u; repetition < 32u; repetition++) {
    reset_builder = et_i2_private_reset_builder_create_v1(1);
    CHECK(reset_builder != NULL);
    CHECK(reset_builder != stale_builder);
    CHECK(et_i2_private_reset_builder_abort_v1(stale_builder) == -1);
    CHECK(et_i2_private_reset_builder_abort_v1(reset_builder) == 0);
#ifdef ET_F32_TENSOR_TESTING
    check_bridge_snapshot(repeated_baseline);
#endif
  }

  stale_builder = et_i2_private_decode_builder_create_v1(1, &bytevector, 4);
  CHECK(stale_builder != NULL);
  CHECK(et_i2_private_decode_builder_abort_v1(stale_builder) == 0);
  check_builder_bytevector_alias_rejected(stale_builder, empty);
#ifdef ET_F32_TENSOR_TESTING
  check_bridge_snapshot(repeated_baseline);
#endif
  for (size_t repetition = 0u; repetition < 32u; repetition++) {
    decode_builder =
        et_i2_private_decode_builder_create_v1(1, &bytevector, 4);
    CHECK(decode_builder != NULL);
    CHECK(decode_builder != stale_builder);
    CHECK(et_i2_private_decode_builder_set_v1(stale_builder, 0, 1) == -1);
    CHECK(et_i2_private_decode_builder_abort_v1(decode_builder) == 0);
#ifdef ET_F32_TENSOR_TESTING
    check_bridge_snapshot(repeated_baseline);
#endif
  }

  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&empty, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&destination, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&source, &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  {
    et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
    et_f32_test_live_counts_snapshot_v1(&counts);
    CHECK(counts.tensors == 0u);
    CHECK(counts.parameters == 0u);
    CHECK(counts.borrows == 0u);
    CHECK(counts.copy_plans == 0u);
    CHECK(counts.gradient_plans == 0u);
    CHECK(counts.reset_plans == 0u);
    CHECK(counts.owned_clones == 0u);
  }
#endif
  if (failures != 0) {
    return 1;
  }
  (void)printf("I2 Wave2 bridge PASS: %d checks\n", checks);
  return 0;
}
