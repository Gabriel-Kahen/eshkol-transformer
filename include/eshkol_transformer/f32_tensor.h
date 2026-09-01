#ifndef ESHKOL_TRANSFORMER_F32_TENSOR_H
#define ESHKOL_TRANSFORMER_F32_TENSOR_H

#include "eshkol_transformer/kernel_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define ET_F32_TENSOR_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define ET_F32_TENSOR_API __attribute__((visibility("default")))
#else
#define ET_F32_TENSOR_API
#endif

#define ET_F32_TENSOR_ABI_MAJOR 1u
#define ET_F32_TENSOR_ABI_MINOR 0u
#define ET_F32_TENSOR_ERROR_SOURCE_DOMAIN "f32-tensor"
#define ET_F32_TENSOR_ERROR_OPERATION_CAPACITY 64u
#define ET_F32_TENSOR_ERROR_MESSAGE_CAPACITY 192u

enum {
  ET_F32_TENSOR_ERROR_NONE = 0,
  ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
  ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
  ET_F32_TENSOR_ERROR_NONCONTIGUOUS,
  ET_F32_TENSOR_ERROR_INVALID_STATE,
  ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
  ET_F32_TENSOR_ERROR_INTERNAL
};
typedef uint32_t et_f32_tensor_error_category;

enum {
  ET_F32_TENSOR_CODE_OK = 0,
  ET_F32_TENSOR_CODE_NULL_ARGUMENT,
  ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
  ET_F32_TENSOR_CODE_INVALID_SHAPE,
  ET_F32_TENSOR_CODE_INVALID_BUFFER,
  ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH,
  ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
  ET_F32_TENSOR_CODE_INVALID_HANDLE,
  ET_F32_TENSOR_CODE_ACTIVE_BORROW,
  ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH,
  ET_F32_TENSOR_CODE_PROVIDER_REJECTED,
  ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT
};
typedef uint32_t et_f32_tensor_error_code;

typedef struct et_f32_tensor_error {
  et_f32_tensor_error_category category;
  et_f32_tensor_error_code code;
  char operation[ET_F32_TENSOR_ERROR_OPERATION_CAPACITY];
  char message[ET_F32_TENSOR_ERROR_MESSAGE_CAPACITY];
} et_f32_tensor_error;

#if defined(__cplusplus)
#define ET_F32_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ET_F32_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif
ET_F32_STATIC_ASSERT(sizeof(et_f32_tensor_error) == 264u,
                     "I2 error record size changed");
ET_F32_STATIC_ASSERT(offsetof(et_f32_tensor_error, category) == 0u,
                     "I2 error category offset changed");
ET_F32_STATIC_ASSERT(offsetof(et_f32_tensor_error, code) == 4u,
                     "I2 error code offset changed");
ET_F32_STATIC_ASSERT(offsetof(et_f32_tensor_error, operation) == 8u,
                     "I2 error operation offset changed");
ET_F32_STATIC_ASSERT(offsetof(et_f32_tensor_error, message) == 72u,
                     "I2 error message offset changed");
#undef ET_F32_STATIC_ASSERT

typedef struct et_f32_tensor et_f32_tensor;
typedef struct et_f32_tensor_borrow et_f32_tensor_borrow;
typedef struct et_f32_tensor_copy_plan et_f32_tensor_copy_plan;

typedef struct et_f32_tensor_copy_assignment_v1 {
  size_t struct_size;
  et_f32_tensor *destination;
  const et_f32_tensor *source;
} et_f32_tensor_copy_assignment_v1;

ET_F32_TENSOR_API int32_t et_f32_tensor_abi_major_v1(void);
ET_F32_TENSOR_API int32_t et_f32_tensor_abi_minor_v1(void);
ET_F32_TENSOR_API int32_t et_f32_tensor_abi_require_v1(
    uint32_t major, uint32_t minimum_minor, et_f32_tensor_error *error);
ET_F32_TENSOR_API void
et_f32_tensor_error_clear_v1(et_f32_tensor_error *error);

/*
 * Shapes are deep-copied. Pointer outputs must initially contain NULL and are
 * byte-preserved on failure. All calls require caller serialization.
 */
ET_F32_TENSOR_API int32_t et_f32_tensor_create_v1(
    size_t rank, const uint64_t *shape, et_f32_tensor **tensor,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t
et_f32_tensor_destroy_v1(et_f32_tensor **tensor, et_f32_tensor_error *error);

ET_F32_TENSOR_API int32_t et_f32_tensor_rank_v1(
    const et_f32_tensor *tensor, size_t *rank, et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_shape_at_v1(
    const et_f32_tensor *tensor, size_t dimension, uint64_t *extent,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_stride_bytes_at_v1(
    const et_f32_tensor *tensor, size_t dimension, size_t *stride,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_element_count_v1(
    const et_f32_tensor *tensor, size_t *element_count,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_byte_length_v1(
    const et_f32_tensor *tensor, size_t *byte_length,
    et_f32_tensor_error *error);

/* Exact physical binary32 representations; no numeric conversion occurs. */
ET_F32_TENSOR_API int32_t et_f32_tensor_copy_bits_from_v1(
    et_f32_tensor *tensor, const uint32_t *source_bits, size_t element_count,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_copy_bits_to_v1(
    const et_f32_tensor *tensor, uint32_t *destination_bits,
    size_t element_count, et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_clone_v1(
    const et_f32_tensor *tensor, et_f32_tensor **clone,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_storage_identical_v1(
    const et_f32_tensor *left, const et_f32_tensor *right, int32_t *identical,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_bits_equal_v1(
    const et_f32_tensor *left, const et_f32_tensor *right, int32_t *equal,
    et_f32_tensor_error *error);

ET_F32_TENSOR_API int32_t et_f32_tensor_copy_plan_prepare_v1(
    size_t assignment_count,
    const et_f32_tensor_copy_assignment_v1 *assignments,
    et_f32_tensor_copy_plan **plan, et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_copy_plan_commit_v1(
    et_f32_tensor_copy_plan *plan, et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_copy_plan_release_v1(
    et_f32_tensor_copy_plan **plan, et_f32_tensor_error *error);

ET_F32_TENSOR_API int32_t et_f32_tensor_borrow_begin_v1(
    et_f32_tensor *tensor, et_f32_tensor_borrow **borrow,
    et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_borrow_view_v1(
    const et_f32_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 **view, et_f32_tensor_error *error);
ET_F32_TENSOR_API int32_t et_f32_tensor_borrow_end_v1(
    et_f32_tensor_borrow **borrow, et_f32_tensor_error *error);

/* Explicit composition accessor; the canonical K1 provider symbol is absent. */
ET_F32_TENSOR_API const et_kernel_provider_v1 *
et_f32_tensor_provider_v1(void);

#ifdef ET_F32_TENSOR_TESTING
void et_f32_tensor_test_fail_alloc_after_v1(size_t successful_allocations);
void et_f32_tensor_test_reset_allocator_v1(void);
const uint64_t *et_f32_tensor_test_shape_storage_v1(
    const et_f32_tensor *tensor);
const size_t *et_f32_tensor_test_stride_storage_v1(
    const et_f32_tensor *tensor);
const float *et_f32_tensor_test_data_storage_v1(
    const et_f32_tensor *tensor);
size_t et_f32_tensor_test_control_bytes_v1(void);
size_t et_f32_tensor_test_borrow_bytes_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#undef ET_F32_TENSOR_API

#endif
