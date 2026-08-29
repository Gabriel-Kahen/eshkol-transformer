#ifndef ESHKOL_TRANSFORMER_I64_TENSOR_H
#define ESHKOL_TRANSFORMER_I64_TENSOR_H

#include "eshkol_transformer/kernel_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_I64_TENSOR_ABI_MAJOR 1u
#define ET_I64_TENSOR_ABI_MINOR 0u
#define ET_I64_TENSOR_ERROR_SOURCE_DOMAIN "i64-tensor"
#define ET_I64_TENSOR_ERROR_OPERATION_CAPACITY 64u
#define ET_I64_TENSOR_ERROR_MESSAGE_CAPACITY 192u

enum {
  ET_I64_TENSOR_ERROR_NONE = 0,
  ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
  ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
  ET_I64_TENSOR_ERROR_NONCONTIGUOUS,
  ET_I64_TENSOR_ERROR_INVALID_STATE,
  ET_I64_TENSOR_ERROR_VERSION_MISMATCH,
  ET_I64_TENSOR_ERROR_INTERNAL
};
typedef uint32_t et_i64_tensor_error_category;

enum {
  ET_I64_TENSOR_CODE_OK = 0,
  ET_I64_TENSOR_CODE_NULL_ARGUMENT,
  ET_I64_TENSOR_CODE_INVALID_STRUCT_SIZE,
  ET_I64_TENSOR_CODE_INTEGER_OVERFLOW,
  ET_I64_TENSOR_CODE_INVALID_SHAPE,
  ET_I64_TENSOR_CODE_INVALID_BUFFER,
  ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH,
  ET_I64_TENSOR_CODE_ALLOCATION_FAILED,
  ET_I64_TENSOR_CODE_INVALID_HANDLE,
  ET_I64_TENSOR_CODE_ACTIVE_BORROW,
  ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH,
  ET_I64_TENSOR_CODE_PROVIDER_REJECTED
};
typedef uint32_t et_i64_tensor_error_code;

typedef struct et_i64_tensor_error {
  et_i64_tensor_error_category category;
  et_i64_tensor_error_code code;
  char operation[ET_I64_TENSOR_ERROR_OPERATION_CAPACITY];
  char message[ET_I64_TENSOR_ERROR_MESSAGE_CAPACITY];
} et_i64_tensor_error;

#if defined(__cplusplus)
#define ET_I64_TENSOR_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ET_I64_TENSOR_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

ET_I64_TENSOR_STATIC_ASSERT(sizeof(et_i64_tensor_error) == 264u,
                            "I1 error record size changed");
ET_I64_TENSOR_STATIC_ASSERT(offsetof(et_i64_tensor_error, category) == 0u,
                            "I1 error category offset changed");
ET_I64_TENSOR_STATIC_ASSERT(offsetof(et_i64_tensor_error, code) == 4u,
                            "I1 error code offset changed");
ET_I64_TENSOR_STATIC_ASSERT(offsetof(et_i64_tensor_error, operation) == 8u,
                            "I1 error operation offset changed");
ET_I64_TENSOR_STATIC_ASSERT(offsetof(et_i64_tensor_error, message) == 72u,
                            "I1 error message offset changed");
ET_I64_TENSOR_STATIC_ASSERT(
    sizeof(((et_i64_tensor_error *)0)->category) == 4u,
    "I1 error category size changed");
ET_I64_TENSOR_STATIC_ASSERT(sizeof(((et_i64_tensor_error *)0)->code) == 4u,
                            "I1 error code size changed");
ET_I64_TENSOR_STATIC_ASSERT(
    sizeof(((et_i64_tensor_error *)0)->operation) == 64u,
    "I1 error operation size changed");
ET_I64_TENSOR_STATIC_ASSERT(
    sizeof(((et_i64_tensor_error *)0)->message) == 192u,
    "I1 error message size changed");

#undef ET_I64_TENSOR_STATIC_ASSERT

typedef struct et_i64_tensor et_i64_tensor;
typedef struct et_i64_tensor_borrow et_i64_tensor_borrow;

int32_t et_i64_tensor_abi_major_v1(void);
int32_t et_i64_tensor_abi_minor_v1(void);
int32_t et_i64_tensor_abi_require_v1(uint32_t major, uint32_t minimum_minor,
                                     et_i64_tensor_error *error);
void et_i64_tensor_error_clear_v1(et_i64_tensor_error *error);

/* Pointer-valued output slots must contain NULL and remain NULL on failure. */
int32_t et_i64_tensor_create_v1(size_t rank, const uint64_t *shape,
                                et_i64_tensor **tensor,
                                et_i64_tensor_error *error);
int32_t et_i64_tensor_destroy_v1(et_i64_tensor **tensor,
                                 et_i64_tensor_error *error);

int32_t et_i64_tensor_rank_v1(const et_i64_tensor *tensor, size_t *rank,
                              et_i64_tensor_error *error);
int32_t et_i64_tensor_shape_at_v1(const et_i64_tensor *tensor,
                                  size_t dimension, uint64_t *extent,
                                  et_i64_tensor_error *error);
int32_t et_i64_tensor_stride_bytes_at_v1(const et_i64_tensor *tensor,
                                         size_t dimension, size_t *stride,
                                         et_i64_tensor_error *error);
int32_t et_i64_tensor_element_count_v1(const et_i64_tensor *tensor,
                                       size_t *element_count,
                                       et_i64_tensor_error *error);
int32_t et_i64_tensor_byte_length_v1(const et_i64_tensor *tensor,
                                     size_t *byte_length,
                                     et_i64_tensor_error *error);

int32_t et_i64_tensor_copy_from_v1(et_i64_tensor *tensor,
                                   const int64_t *source,
                                   size_t element_count,
                                   et_i64_tensor_error *error);
int32_t et_i64_tensor_copy_to_v1(const et_i64_tensor *tensor,
                                 int64_t *destination,
                                 size_t element_count,
                                 et_i64_tensor_error *error);

int32_t et_i64_tensor_borrow_begin_v1(et_i64_tensor *tensor,
                                      et_i64_tensor_borrow **borrow,
                                      et_i64_tensor_error *error);
/* The lease permits mutation of view->data; descriptor metadata stays const. */
int32_t et_i64_tensor_borrow_view_v1(
    const et_i64_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 **view, et_i64_tensor_error *error);
int32_t et_i64_tensor_borrow_end_v1(et_i64_tensor_borrow **borrow,
                                    et_i64_tensor_error *error);

/* Returned provider metadata is static and immutable for process lifetime. */
const et_kernel_provider_v1 *et_i64_tensor_provider_v1(void);

#ifdef ET_I64_TENSOR_TESTING
/* Allow this many successful I1 allocations, then fail subsequent ones. */
void et_i64_tensor_test_fail_alloc_after_v1(size_t successful_allocations);
void et_i64_tensor_test_reset_allocator_v1(void);
/* Test-only corruption hooks use live handles and never dereference forgeries. */
void et_i64_tensor_test_set_borrow_owner_v1(et_i64_tensor_borrow *borrow,
                                            et_i64_tensor *owner);
void et_i64_tensor_test_set_active_borrow_v1(et_i64_tensor *tensor,
                                             et_i64_tensor_borrow *borrow);
#endif

#ifdef __cplusplus
}
#endif

#endif
