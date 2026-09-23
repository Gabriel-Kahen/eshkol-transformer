#ifndef ESHKOL_TRANSFORMER_F32_PARAMETER_INTERNAL_H
#define ESHKOL_TRANSFORMER_F32_PARAMETER_INTERNAL_H

#include "eshkol_transformer/f32_tensor.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_F32_GRADIENT_ABSENT 0u
#define ET_F32_GRADIENT_PRESENT 1u
#define ET_F32_PARAMETER_MAX_BATCH 4096u

typedef struct et_f32_parameter et_f32_parameter;
typedef struct et_f32_gradient_plan et_f32_gradient_plan;
typedef struct et_f32_gradient_reset_plan et_f32_gradient_reset_plan;

#ifdef ET_I2_PRIVATE_OWNED_CLONE_MATCH
enum {
  ET_I2_PRIVATE_OWNED_RESULT_OK = 0,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_ARGUMENT = 1,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_STATE = 2,
  ET_I2_PRIVATE_OWNED_RESULT_SHAPE_MISMATCH = 3,
  ET_I2_PRIVATE_OWNED_RESULT_NONFINITE = 4,
  ET_I2_PRIVATE_OWNED_RESULT_INVALID_VALUE = 5,
  ET_I2_PRIVATE_OWNED_RESULT_OWNER_CONFLICT = 6,
  ET_I2_PRIVATE_OWNED_RESULT_VERSION_MISMATCH = 7,
  ET_I2_PRIVATE_OWNED_RESULT_ALLOCATION_FAILED = 8,
  ET_I2_PRIVATE_OWNED_RESULT_INTERNAL = 9
};

enum {
  ET_I2_PRIVATE_OWNED_VALUE_FINITE = 1,
  ET_I2_PRIVATE_OWNED_VALUE_FINITE_NONNEGATIVE = 2
};

#if defined(__cplusplus)
static_assert(ET_I2_PRIVATE_OWNED_RESULT_OK == 0 &&
                  ET_I2_PRIVATE_OWNED_RESULT_INTERNAL == 9,
              "private owned result values changed");
#else
_Static_assert(ET_I2_PRIVATE_OWNED_RESULT_OK == 0 &&
                   ET_I2_PRIVATE_OWNED_RESULT_INTERNAL == 9,
               "private owned result values changed");
#endif

int32_t et_i2_private_owned_clone_match_v1(
    const et_f32_tensor *candidate, const et_f32_tensor *destination,
    uint32_t value_policy);
#endif

typedef struct et_f32_gradient_metadata_v1 {
  size_t struct_size;
  uint32_t state;
  uint32_t normalization_weight_bits;
  uint64_t contribution_count;
} et_f32_gradient_metadata_v1;

typedef struct et_f32_gradient_contribution_v1 {
  size_t struct_size;
  et_f32_parameter *destination;
  const et_f32_tensor *weighted_numerator;
  uint64_t expected_ordinal;
} et_f32_gradient_contribution_v1;

int32_t et_f32_tensor_is_live_v1(const et_f32_tensor *tensor);
int32_t et_f32_parameter_is_live_v1(const et_f32_parameter *parameter);
const et_f32_tensor *
et_f32_tensor_canonical_owner_v1(const et_f32_tensor *tensor);
const et_f32_tensor *
et_f32_parameter_canonical_owner_v1(const et_f32_parameter *parameter);
int32_t et_f32_tensor_storage_owner_identical_v1(
    const et_f32_tensor *left, const et_f32_tensor *right);
int32_t et_f32_owned_tensor_clone_v1(const et_f32_tensor *source,
                                     et_f32_tensor **owned_clone,
                                     et_f32_tensor_error *error);
int32_t et_f32_owned_tensor_release_v1(et_f32_tensor *owned_clone,
                                       et_f32_tensor_error *error);

int32_t et_f32_parameter_create_v1(const et_f32_tensor *initial_value,
                                   et_f32_parameter **parameter,
                                   et_f32_tensor_error *error);
int32_t et_f32_parameter_destroy_v1(et_f32_parameter **parameter,
                                    et_f32_tensor_error *error);
int32_t et_f32_parameter_bind_identity_v1(et_f32_parameter *parameter,
                                          const void *identity,
                                          et_f32_tensor_error *error);
int32_t et_f32_parameter_validate_identity_v1(
    const et_f32_parameter *parameter, const void *expected_identity,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_identity_v1(const et_f32_parameter *parameter,
                                     const void **identity,
                                     et_f32_tensor_error *error);

int32_t et_f32_parameter_value_snapshot_v1(const et_f32_parameter *parameter,
                                           et_f32_tensor **snapshot,
                                           et_f32_tensor_error *error);
int32_t et_f32_parameter_value_tensor_v1(
    const et_f32_parameter *parameter, const et_f32_tensor **value,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_value_borrow_begin_v1(
    et_f32_parameter *parameter, et_f32_tensor_borrow **borrow,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_gradient_metadata_v1(
    const et_f32_parameter *parameter,
    et_f32_gradient_metadata_v1 *metadata, et_f32_tensor_error *error);
int32_t et_f32_parameter_gradient_finite_v1(
    const et_f32_parameter *parameter, int32_t *finite,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_gradient_exact_positive_zero_v1(
    const et_f32_parameter *parameter, int32_t *exact_zero,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_gradient_snapshot_v1(
    const et_f32_parameter *parameter, et_f32_tensor **snapshot,
    et_f32_tensor_error *error);
int32_t et_f32_parameter_gradient_borrow_begin_v1(
    et_f32_parameter *parameter, et_f32_tensor_borrow **borrow,
    et_f32_tensor_error *error);

/* One contribution is added to every unique destination in canonical order. */
int32_t et_f32_gradient_plan_prepare_v1(
    size_t contribution_count,
    const et_f32_gradient_contribution_v1 *contributions,
    uint32_t normalization_weight_increment_bits,
    et_f32_gradient_plan **plan, et_f32_tensor_error *error);
int32_t et_f32_gradient_plan_commit_v1(et_f32_gradient_plan *plan,
                                       et_f32_tensor_error *error);
int32_t et_f32_gradient_plan_release_v1(et_f32_gradient_plan **plan,
                                        et_f32_tensor_error *error);

int32_t et_f32_gradient_reset_plan_prepare_v1(
    size_t parameter_count, et_f32_parameter *const *parameters,
    et_f32_gradient_reset_plan **plan, et_f32_tensor_error *error);
int32_t et_f32_gradient_reset_plan_commit_v1(
    et_f32_gradient_reset_plan *plan, et_f32_tensor_error *error);
int32_t et_f32_gradient_reset_plan_release_v1(
    et_f32_gradient_reset_plan **plan, et_f32_tensor_error *error);

#ifdef ET_F32_TENSOR_TESTING
typedef struct et_f32_test_live_counts_v1 {
  size_t struct_size;
  size_t tensors;
  size_t parameters;
  size_t borrows;
  size_t copy_plans;
  size_t gradient_plans;
  size_t reset_plans;
  size_t owned_clones;
} et_f32_test_live_counts_v1;

typedef struct et_f32_test_retired_counts_v1 {
  size_t struct_size;
  size_t tensors;
  size_t parameters;
  size_t borrows;
  size_t copy_plans;
  size_t gradient_plans;
  size_t reset_plans;
  size_t retained_control_bytes;
} et_f32_test_retired_counts_v1;

typedef struct et_f32_test_borrow_event_counts_v1 {
  size_t struct_size;
  size_t begin_calls;
  size_t view_calls;
  size_t end_calls;
} et_f32_test_borrow_event_counts_v1;

enum {
  ET_F32_TEST_TENSOR_METADATA_RANK = 1,
  ET_F32_TEST_TENSOR_METADATA_ELEMENT_COUNT = 2,
  ET_F32_TEST_TENSOR_METADATA_BYTE_LENGTH = 3,
  ET_F32_TEST_TENSOR_METADATA_SHAPE = 4,
  ET_F32_TEST_TENSOR_METADATA_STRIDE = 5,
  ET_F32_TEST_TENSOR_METADATA_DATA = 6
};

typedef struct et_f32_test_tensor_metadata_v1 {
  size_t struct_size;
  size_t rank;
  size_t element_count;
  size_t byte_length;
  size_t plan_pins;
  uint64_t *shape_storage;
  size_t *stride_storage;
  float *data_storage;
  uint64_t shape[ET_KERNEL_MAX_RANK];
  size_t strides[ET_KERNEL_MAX_RANK];
} et_f32_test_tensor_metadata_v1;

void et_f32_parameter_test_set_metadata_v1(et_f32_parameter *parameter,
                                           uint32_t state,
                                           uint64_t contribution_count,
                                           uint32_t weight_bits);
void et_f32_parameter_test_set_gradient_bits_v1(
    et_f32_parameter *parameter, const uint32_t *bits, size_t count);
void et_f32_test_live_counts_snapshot_v1(et_f32_test_live_counts_v1 *counts);
void et_f32_test_retired_counts_snapshot_v1(
    et_f32_test_retired_counts_v1 *counts);
void et_f32_test_borrow_event_counts_snapshot_v1(
    et_f32_test_borrow_event_counts_v1 *counts);
size_t et_f32_tensor_test_successful_allocations_v1(void);
int32_t et_f32_tensor_test_metadata_snapshot_v1(
    const et_f32_tensor *tensor, et_f32_test_tensor_metadata_v1 *snapshot);
int32_t et_f32_tensor_test_metadata_corrupt_v1(et_f32_tensor *tensor,
                                               uint32_t member,
                                               size_t dimension);
int32_t et_f32_tensor_test_metadata_restore_v1(
    et_f32_tensor *tensor, const et_f32_test_tensor_metadata_v1 *snapshot);
int32_t et_f32_tensor_test_plan_pins_v1(const et_f32_tensor *tensor,
                                        size_t *pins);
#endif

#ifdef __cplusplus
}
#endif

#endif
