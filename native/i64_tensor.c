#include "eshkol_transformer/i64_tensor.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ET_I64_TENSOR_MAGIC UINT64_C(0x455449363454454e)
#define ET_I64_BORROW_MAGIC UINT64_C(0x45544936424f5252)
#define ET_I64_RANK1_MAX ((uint64_t)(SIZE_MAX / sizeof(int64_t)))

struct et_i64_tensor {
  uint64_t magic;
  size_t rank;
  size_t element_count;
  size_t byte_length;
  et_i64_tensor_borrow *active_borrow;
  uint64_t *shape;
  size_t *strides;
  int64_t *data;
};

struct et_i64_tensor_borrow {
  uint64_t magic;
  et_i64_tensor *owner;
  et_kernel_tensor_view_v1 view;
};

_Static_assert(CHAR_BIT == 8 && sizeof(int64_t) == 8u,
               "I1 requires exact 64-bit int64_t storage");
_Static_assert(sizeof(et_i64_tensor_error) == 264u,
               "I1 error ABI layout changed");

#ifdef ET_I64_TENSOR_TESTING
static size_t allocation_limit = SIZE_MAX;
static size_t successful_allocations;

void et_i64_tensor_test_fail_alloc_after_v1(size_t allowed) {
  allocation_limit = allowed;
  successful_allocations = 0u;
}

void et_i64_tensor_test_reset_allocator_v1(void) {
  allocation_limit = SIZE_MAX;
  successful_allocations = 0u;
}

void et_i64_tensor_test_set_borrow_owner_v1(et_i64_tensor_borrow *borrow,
                                            et_i64_tensor *owner) {
  if (borrow != NULL && borrow->magic == ET_I64_BORROW_MAGIC) {
    borrow->owner = owner;
  }
}

void et_i64_tensor_test_set_active_borrow_v1(et_i64_tensor *tensor,
                                             et_i64_tensor_borrow *borrow) {
  if (tensor != NULL && tensor->magic == ET_I64_TENSOR_MAGIC) {
    tensor->active_borrow = borrow;
  }
}
#endif

static void *i64_calloc(size_t count, size_t size) {
  void *allocation;
#ifdef ET_I64_TENSOR_TESTING
  if (successful_allocations >= allocation_limit) {
    return NULL;
  }
#endif
  allocation = calloc(count, size);
#ifdef ET_I64_TENSOR_TESTING
  if (allocation != NULL) {
    successful_allocations++;
  }
#endif
  return allocation;
}

void et_i64_tensor_error_clear_v1(et_i64_tensor_error *error) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
}

static int32_t set_i64_error(et_i64_tensor_error *error,
                             et_i64_tensor_error_category category,
                             et_i64_tensor_error_code code,
                             const char *operation, const char *message) {
  if (error != NULL) {
    et_i64_tensor_error_clear_v1(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int32_t success(et_i64_tensor_error *error) {
  et_i64_tensor_error_clear_v1(error);
  return 0;
}

static int aligned_pointer(const void *pointer, size_t alignment) {
  return pointer != NULL && (uintptr_t)pointer % alignment == 0u;
}

static int pointer_span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int ranges_overlap(const void *left, size_t left_bytes,
                          const void *right, size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u) {
    return 0;
  }
  if (!pointer_span_fits(left, left_bytes) ||
      !pointer_span_fits(right, right_bytes)) {
    return 1;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static int valid_tensor(const et_i64_tensor *tensor) {
  return tensor != NULL && tensor->magic == ET_I64_TENSOR_MAGIC;
}

static int32_t require_tensor(const et_i64_tensor *tensor,
                              const char *operation,
                              et_i64_tensor_error *error) {
  if (tensor == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT, operation,
                         "tensor handle is null");
  }
  if (!valid_tensor(tensor)) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_INVALID_HANDLE, operation,
                         "tensor handle is invalid or no longer live");
  }
  return 0;
}

int32_t et_i64_tensor_abi_major_v1(void) {
  return (int32_t)ET_I64_TENSOR_ABI_MAJOR;
}

int32_t et_i64_tensor_abi_minor_v1(void) {
  return (int32_t)ET_I64_TENSOR_ABI_MINOR;
}

int32_t et_i64_tensor_abi_require_v1(uint32_t major, uint32_t minimum_minor,
                                     et_i64_tensor_error *error) {
  if (major != ET_I64_TENSOR_ABI_MAJOR ||
      minimum_minor > ET_I64_TENSOR_ABI_MINOR) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_VERSION_MISMATCH,
                         ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH,
                         "i64-tensor-abi-require",
                         "requested I1 tensor ABI version is unavailable");
  }
  return success(error);
}

static int32_t validate_shape(size_t rank, const uint64_t *shape,
                              size_t *element_count, size_t *byte_length,
                              int *empty, et_i64_tensor_error *error) {
  size_t count = 1u;
  int has_zero = 0;
  if (rank > ET_KERNEL_MAX_RANK) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_I64_TENSOR_CODE_INTEGER_OVERFLOW,
                         "i64-tensor-create",
                         "tensor rank exceeds the I1 ABI limit");
  }
  if (rank > 0u && shape == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_I64_TENSOR_CODE_INVALID_SHAPE,
                         "i64-tensor-create",
                         "nonzero rank requires a shape array");
  }
  if (rank > 0u &&
      (!aligned_pointer(shape, _Alignof(uint64_t)) ||
       rank > SIZE_MAX / sizeof(*shape) ||
       !pointer_span_fits(shape, rank * sizeof(*shape)))) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER,
                         "i64-tensor-create",
                         "shape storage is misaligned or its span overflows");
  }
  for (size_t index = 0; index < rank; index++) {
    has_zero = has_zero || shape[index] == 0u;
  }
  if (!has_zero) {
    for (size_t index = 0; index < rank; index++) {
      if (shape[index] > SIZE_MAX ||
          count > SIZE_MAX / (size_t)shape[index]) {
        return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                             ET_I64_TENSOR_CODE_INTEGER_OVERFLOW,
                             "i64-tensor-create",
                             "tensor element count overflows address space");
      }
      count *= (size_t)shape[index];
    }
    if (count > SIZE_MAX / sizeof(int64_t)) {
      return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                           ET_I64_TENSOR_CODE_INTEGER_OVERFLOW,
                           "i64-tensor-create",
                           "tensor byte length overflows address space");
    }
  } else {
    count = 0u;
  }
  *element_count = count;
  *byte_length = count * sizeof(int64_t);
  *empty = has_zero;
  return 0;
}

int32_t et_i64_tensor_create_v1(size_t rank, const uint64_t *shape,
                                et_i64_tensor **output,
                                et_i64_tensor_error *error) {
  et_i64_tensor *tensor = NULL;
  size_t element_count;
  size_t byte_length;
  int empty;
  int output_was_nonnull;
  int32_t result;
  if (output == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-create",
                         "tensor output pointer is null");
  }
  output_was_nonnull = *output != NULL;
  *output = NULL;
  if (output_was_nonnull) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER,
                         "i64-tensor-create",
                         "tensor output must be null before creation");
  }
  result = validate_shape(rank, shape, &element_count, &byte_length, &empty,
                          error);
  if (result != 0) {
    return result;
  }
  tensor = (et_i64_tensor *)i64_calloc(1u, sizeof(*tensor));
  if (tensor == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INTERNAL,
                         ET_I64_TENSOR_CODE_ALLOCATION_FAILED,
                         "i64-tensor-create",
                         "cannot allocate the tensor control block");
  }
  tensor->rank = rank;
  tensor->element_count = element_count;
  tensor->byte_length = byte_length;
  if (rank > 0u) {
    tensor->shape = (uint64_t *)i64_calloc(rank, sizeof(*tensor->shape));
    tensor->strides = (size_t *)i64_calloc(rank, sizeof(*tensor->strides));
    if (tensor->shape == NULL || tensor->strides == NULL) {
      free(tensor->strides);
      free(tensor->shape);
      free(tensor);
      return set_i64_error(error, ET_I64_TENSOR_ERROR_INTERNAL,
                           ET_I64_TENSOR_CODE_ALLOCATION_FAILED,
                           "i64-tensor-create",
                           "cannot allocate tensor shape metadata");
    }
    memcpy(tensor->shape, shape, rank * sizeof(*shape));
    if (!empty) {
      size_t running = sizeof(int64_t);
      for (size_t index = rank; index > 0u; index--) {
        const size_t dimension = index - 1u;
        tensor->strides[dimension] = running;
        running *= (size_t)shape[dimension];
      }
    }
  }
  if (byte_length > 0u) {
    tensor->data = (int64_t *)i64_calloc(element_count, sizeof(*tensor->data));
    if (tensor->data == NULL) {
      free(tensor->strides);
      free(tensor->shape);
      free(tensor);
      return set_i64_error(error, ET_I64_TENSOR_ERROR_INTERNAL,
                           ET_I64_TENSOR_CODE_ALLOCATION_FAILED,
                           "i64-tensor-create",
                           "cannot allocate tensor element storage");
    }
  }
  tensor->magic = ET_I64_TENSOR_MAGIC;
  *output = tensor;
  return success(error);
}

int32_t et_i64_tensor_destroy_v1(et_i64_tensor **tensor,
                                 et_i64_tensor_error *error) {
  et_i64_tensor *value;
  if (tensor == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-destroy",
                         "tensor handle pointer is null");
  }
  value = *tensor;
  if (value == NULL) {
    return success(error);
  }
  if (!valid_tensor(value)) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_INVALID_HANDLE,
                         "i64-tensor-destroy",
                         "tensor handle is invalid or no longer live");
  }
  if (value->active_borrow != NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_ACTIVE_BORROW,
                         "i64-tensor-destroy",
                         "tensor cannot be destroyed while a view is borrowed");
  }
  value->magic = 0u;
  free(value->data);
  free(value->strides);
  free(value->shape);
  free(value);
  *tensor = NULL;
  return success(error);
}

int32_t et_i64_tensor_rank_v1(const et_i64_tensor *tensor, size_t *rank,
                              et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-rank", error);
  if (result != 0) {
    return result;
  }
  if (rank == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-rank", "rank output pointer is null");
  }
  *rank = tensor->rank;
  return success(error);
}

int32_t et_i64_tensor_shape_at_v1(const et_i64_tensor *tensor,
                                  size_t dimension, uint64_t *extent,
                                  et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-shape-at", error);
  if (result != 0) {
    return result;
  }
  if (extent == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-shape-at",
                         "shape output pointer is null");
  }
  if (dimension >= tensor->rank) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_I64_TENSOR_CODE_INVALID_SHAPE,
                         "i64-tensor-shape-at",
                         "shape dimension is out of range");
  }
  *extent = tensor->shape[dimension];
  return success(error);
}

int32_t et_i64_tensor_stride_bytes_at_v1(const et_i64_tensor *tensor,
                                         size_t dimension, size_t *stride,
                                         et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-stride-at", error);
  if (result != 0) {
    return result;
  }
  if (stride == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-stride-at",
                         "stride output pointer is null");
  }
  if (dimension >= tensor->rank) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_I64_TENSOR_CODE_INVALID_SHAPE,
                         "i64-tensor-stride-at",
                         "stride dimension is out of range");
  }
  *stride = tensor->strides[dimension];
  return success(error);
}

int32_t et_i64_tensor_element_count_v1(const et_i64_tensor *tensor,
                                       size_t *element_count,
                                       et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-element-count", error);
  if (result != 0) {
    return result;
  }
  if (element_count == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-element-count",
                         "element-count output pointer is null");
  }
  *element_count = tensor->element_count;
  return success(error);
}

int32_t et_i64_tensor_byte_length_v1(const et_i64_tensor *tensor,
                                     size_t *byte_length,
                                     et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-byte-length", error);
  if (result != 0) {
    return result;
  }
  if (byte_length == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-byte-length",
                         "byte-length output pointer is null");
  }
  *byte_length = tensor->byte_length;
  return success(error);
}

static int32_t validate_copy_buffer(const et_i64_tensor *tensor,
                                    const void *buffer, size_t element_count,
                                    const char *operation,
                                    et_i64_tensor_error *error) {
  if (element_count != tensor->element_count) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH, operation,
                         "copy element count does not match the tensor");
  }
  if (tensor->byte_length == 0u) {
    return 0;
  }
  if (buffer == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT, operation,
                         "nonempty copy buffer is null");
  }
  if (!aligned_pointer(buffer, _Alignof(int64_t)) ||
      !pointer_span_fits(buffer, tensor->byte_length)) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER, operation,
                         "copy buffer is misaligned or its span overflows");
  }
  if (ranges_overlap(buffer, tensor->byte_length, tensor->data,
                     tensor->byte_length)) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER, operation,
                         "copy buffer aliases tensor storage");
  }
  return 0;
}

int32_t et_i64_tensor_copy_from_v1(et_i64_tensor *tensor,
                                   const int64_t *source,
                                   size_t element_count,
                                   et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-copy-from", error);
  if (result != 0) {
    return result;
  }
  if (tensor->active_borrow != NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_ACTIVE_BORROW,
                         "i64-tensor-copy-from",
                         "tensor cannot be mutated while a view is borrowed");
  }
  result = validate_copy_buffer(tensor, source, element_count,
                                "i64-tensor-copy-from", error);
  if (result != 0) {
    return result;
  }
  if (tensor->byte_length != 0u) {
    memcpy(tensor->data, source, tensor->byte_length);
  }
  return success(error);
}

int32_t et_i64_tensor_copy_to_v1(const et_i64_tensor *tensor,
                                 int64_t *destination,
                                 size_t element_count,
                                 et_i64_tensor_error *error) {
  int32_t result = require_tensor(tensor, "i64-tensor-copy-to", error);
  if (result != 0) {
    return result;
  }
  result = validate_copy_buffer(tensor, destination, element_count,
                                "i64-tensor-copy-to", error);
  if (result != 0) {
    return result;
  }
  if (tensor->byte_length != 0u) {
    memcpy(destination, tensor->data, tensor->byte_length);
  }
  return success(error);
}

int32_t et_i64_tensor_borrow_begin_v1(et_i64_tensor *tensor,
                                      et_i64_tensor_borrow **output,
                                      et_i64_tensor_error *error) {
  et_i64_tensor_borrow *borrow;
  int output_was_nonnull;
  int32_t result;
  if (output == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-borrow-begin",
                         "borrow output pointer is null");
  }
  output_was_nonnull = *output != NULL;
  *output = NULL;
  if (output_was_nonnull) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER,
                         "i64-tensor-borrow-begin",
                         "borrow output must be null before acquisition");
  }
  result = require_tensor(tensor, "i64-tensor-borrow-begin", error);
  if (result != 0) {
    return result;
  }
  if (tensor->active_borrow != NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_ACTIVE_BORROW,
                         "i64-tensor-borrow-begin",
                         "tensor already has an active borrow");
  }
  borrow = (et_i64_tensor_borrow *)i64_calloc(1u, sizeof(*borrow));
  if (borrow == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INTERNAL,
                         ET_I64_TENSOR_CODE_ALLOCATION_FAILED,
                         "i64-tensor-borrow-begin",
                         "cannot allocate a tensor borrow lease");
  }
  borrow->owner = tensor;
  borrow->view.struct_size = sizeof(borrow->view);
  borrow->view.data = tensor->data;
  borrow->view.byte_length = tensor->byte_length;
  borrow->view.dtype = "i64";
  borrow->view.device = "cpu";
  borrow->view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  borrow->view.offset_bytes = 0u;
  borrow->view.rank = tensor->rank;
  borrow->view.shape = tensor->shape;
  borrow->magic = ET_I64_BORROW_MAGIC;
  tensor->active_borrow = borrow;
  *output = borrow;
  return success(error);
}

int32_t et_i64_tensor_borrow_view_v1(
    const et_i64_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 **view, et_i64_tensor_error *error) {
  int output_was_nonnull;
  if (view == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-borrow-view",
                         "borrow and view output are required");
  }
  output_was_nonnull = *view != NULL;
  *view = NULL;
  if (output_was_nonnull) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_INVALID_BUFFER,
                         "i64-tensor-borrow-view",
                         "view output must be null before access");
  }
  if (borrow == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-borrow-view",
                         "borrow and view output are required");
  }
  if (borrow->magic != ET_I64_BORROW_MAGIC ||
      !valid_tensor(borrow->owner) ||
      borrow->owner->active_borrow != borrow) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_INVALID_HANDLE,
                         "i64-tensor-borrow-view",
                         "borrow lease is invalid or no longer active");
  }
  *view = &borrow->view;
  return success(error);
}

int32_t et_i64_tensor_borrow_end_v1(et_i64_tensor_borrow **borrow,
                                    et_i64_tensor_error *error) {
  et_i64_tensor_borrow *value;
  if (borrow == NULL) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_I64_TENSOR_CODE_NULL_ARGUMENT,
                         "i64-tensor-borrow-end",
                         "borrow handle pointer is null");
  }
  value = *borrow;
  if (value == NULL) {
    return success(error);
  }
  if (value->magic != ET_I64_BORROW_MAGIC || !valid_tensor(value->owner) ||
      value->owner->active_borrow != value) {
    return set_i64_error(error, ET_I64_TENSOR_ERROR_INVALID_STATE,
                         ET_I64_TENSOR_CODE_INVALID_HANDLE,
                         "i64-tensor-borrow-end",
                         "borrow lease is invalid or no longer active");
  }
  value->owner->active_borrow = NULL;
  value->magic = 0u;
  free(value);
  *borrow = NULL;
  return success(error);
}

static int32_t set_kernel_error(et_kernel_error *error,
                                et_kernel_error_category category,
                                et_kernel_error_code code,
                                const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   "storage.copy");
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int provider_shape_supported(size_t rank, const uint64_t *shape) {
  if (rank == 0u) {
    return shape == NULL;
  }
  if (shape == NULL || rank != 1u) {
    return 0;
  }
  return shape[0] <= ET_I64_RANK1_MAX;
}

static int32_t provider_validate_table(size_t count, size_t stride,
                                       size_t bytes, const void *base,
                                       const char *message,
                                       const et_kernel_tensor_view_v1 **view,
                                       et_kernel_error *error) {
  if (count != 1u || base == NULL) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER, message);
  }
  if (stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return set_kernel_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                            ET_KERNEL_CODE_INVALID_STRUCT_SIZE, message);
  }
  if (stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      !aligned_pointer(base, _Alignof(et_kernel_tensor_view_v1)) ||
      bytes != stride || !pointer_span_fits(base, bytes)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER, message);
  }
  *view = (const et_kernel_tensor_view_v1 *)base;
  if ((*view)->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      (*view)->struct_size > stride) {
    return set_kernel_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                            ET_KERNEL_CODE_INVALID_STRUCT_SIZE, message);
  }
  return 0;
}

static int32_t provider_validate_view(const et_kernel_tensor_view_v1 *view,
                                      const et_kernel_request_v1 *request,
                                      et_kernel_error *error) {
  size_t expected_bytes = sizeof(int64_t);
  if (!exact_text(view->dtype, "i64")) {
    return set_kernel_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_TEXT,
                            "storage.copy accepts only i64 tensor views");
  }
  if (!exact_text(view->device, "cpu") ||
      !exact_text(request->device, "cpu")) {
    return set_kernel_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy accepts only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_kernel_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy requires dense row-major zero-offset views");
  }
  if (view->rank != request->rank ||
      (view->rank > 0u && view->shape == NULL) ||
      !provider_shape_supported(view->rank, view->shape)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_SHAPE,
                            "storage.copy tensor shape is unsupported");
  }
  if (view->rank > 0u &&
      (!aligned_pointer(view->shape, _Alignof(uint64_t)) ||
       !pointer_span_fits(view->shape, view->rank * sizeof(*view->shape)))) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy shape storage is invalid");
  }
  for (size_t index = 0; index < view->rank; index++) {
    if (view->shape[index] != request->shape[index]) {
      return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                              ET_KERNEL_CODE_INVALID_SHAPE,
                              "storage.copy tensor shape differs from its request");
    }
    if (view->shape[index] == 0u) {
      expected_bytes = 0u;
      break;
    }
    expected_bytes *= (size_t)view->shape[index];
  }
  if (view->byte_length != expected_bytes ||
      (expected_bytes > 0u &&
       (!aligned_pointer(view->data, _Alignof(int64_t)) ||
        !pointer_span_fits(view->data, expected_bytes)))) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy data alignment or byte span is invalid");
  }
  return 0;
}

static int32_t provider_validate_call(const et_kernel_call_v1 *call,
                                      et_kernel_error *error) {
  const et_kernel_tensor_view_v1 *input;
  const et_kernel_tensor_view_v1 *output;
  int32_t result;
  if (call == NULL || call->request == NULL) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_NULL_ARGUMENT,
                            "storage.copy call and request are required");
  }
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      call->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE) {
    return set_kernel_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                            ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                            "storage.copy call or request is truncated");
  }
  if (!exact_text(call->capability, "tensor.i64") ||
      !exact_text(call->request->operation, "storage.copy") ||
      !exact_text(call->request->dtype, "i64") ||
      !provider_shape_supported(call->request->rank,
                                call->request->shape)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                            ET_KERNEL_CODE_PROVIDER_REJECTED,
                            "storage.copy request is outside I1 evidence");
  }
  result = provider_validate_table(call->input_count, call->input_stride,
                                   call->input_bytes, call->inputs,
                                   "storage.copy requires exactly one valid input",
                                   &input, error);
  if (result != 0) {
    return result;
  }
  result = provider_validate_table(
      call->output_count, call->output_stride, call->output_bytes,
      call->outputs, "storage.copy requires exactly one valid output", &output,
      error);
  if (result != 0) {
    return result;
  }
  result = provider_validate_view(input, call->request, error);
  if (result != 0) {
    return result;
  }
  result = provider_validate_view(output, call->request, error);
  if (result != 0) {
    return result;
  }
  if (input->rank != output->rank) {
    return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_SHAPE,
                            "storage.copy input and output ranks differ");
  }
  for (size_t index = 0; index < input->rank; index++) {
    if (input->shape[index] != output->shape[index]) {
      return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                              ET_KERNEL_CODE_INVALID_SHAPE,
                              "storage.copy input and output shapes differ");
    }
  }
  if (ranges_overlap(input->data, input->byte_length, output->data,
                     output->byte_length)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_ALIASING_OUTPUT,
                            "storage.copy output aliases its input");
  }
  et_kernel_error_clear(error);
  return 0;
}

static void provider_invoke_call(const et_kernel_call_v1 *call) {
  const et_kernel_tensor_view_v1 *input =
      (const et_kernel_tensor_view_v1 *)call->inputs;
  et_kernel_tensor_view_v1 *output =
      (et_kernel_tensor_view_v1 *)call->outputs;
  if (output->byte_length != 0u) {
    memcpy(output->data, input->data, output->byte_length);
  }
}

static const et_kernel_dimension_range_v1 rank1_dimensions[] = {
    {.minimum = 0u,
     .maximum = ET_I64_RANK1_MAX,
     .maximum_unbounded = 0u,
     .reserved = {0}},
};
static const et_kernel_shape_range_v1 provider_ranges[] = {
    {.rank = 0u, .dimensions = NULL},
    {.rank = 1u, .dimensions = rank1_dimensions},
};
static const char *const provider_operations[] = {"storage.copy"};
static const char *const provider_dtypes[] = {"i64"};
static const char *const provider_devices[] = {"cpu"};
static const et_kernel_capability_v1 provider_capability = {
    .struct_size = sizeof(et_kernel_capability_v1),
    .name = "tensor.i64",
    .status = ET_KERNEL_CAPABILITY_VERIFIED,
    .implementation = "eshkol-transformer-i64",
    .version = "1.0",
    .evidence = "I1:bounded-exact-i64-storage.copy-v1",
    .deterministic = 1u,
    .reserved = {0},
    .operation_count = 1u,
    .operations = provider_operations,
    .dtype_count = 1u,
    .dtypes = provider_dtypes,
    .device_count = 1u,
    .devices = provider_devices,
    .shape_range_count = 2u,
    .shape_ranges = provider_ranges,
};
static const et_kernel_provider_v1 provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = ET_KERNEL_ABI_MINOR,
    .required_features = 0u,
    .name = "eshkol-transformer-i64",
    .version = "1.0",
    .evidence = "I1:bounded-exact-i64-storage.copy-v1",
    .capability_count = 1u,
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(et_kernel_capability_v1),
    .capabilities = &provider_capability,
    .validate_call = provider_validate_call,
    .invoke_call = provider_invoke_call,
};

const et_kernel_provider_v1 *et_i64_tensor_provider_v1(void) {
  return &provider;
}
