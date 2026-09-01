#include "f32_parameter_internal.h"

#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#define ET_F32_TENSOR_MAGIC UINT64_C(0x455446333254454e)
#define ET_F32_BORROW_MAGIC UINT64_C(0x45544633424f5252)
#define ET_F32_RANK1_MAX ((uint64_t)(SIZE_MAX / sizeof(float)))

struct et_f32_tensor {
  uint64_t magic;
  et_f32_tensor *registry_next;
  size_t rank;
  size_t element_count;
  size_t byte_length;
  et_f32_tensor_borrow *active_borrow;
  size_t plan_pins;
  uint64_t *shape;
  size_t *strides;
  float *data;
};

struct et_f32_tensor_borrow {
  uint64_t magic;
  et_f32_tensor_borrow *registry_next;
  et_f32_tensor *owner;
  et_kernel_tensor_view_v1 view;
};

struct et_f32_tensor_copy_plan {
  uint64_t magic;
  et_f32_tensor_copy_plan *registry_next;
  size_t count;
  et_f32_tensor_copy_assignment_v1 *assignments;
  uint32_t consumed;
};

struct et_f32_parameter {
  uint64_t magic;
  et_f32_parameter *registry_next;
  et_f32_tensor *value;
  et_f32_tensor *gradient;
  const void *identity;
  uint64_t contribution_count;
  uint32_t normalization_weight_bits;
  uint32_t gradient_state;
  size_t plan_pins;
};

typedef struct et_f32_gradient_plan_entry {
  et_f32_parameter *parameter;
  const et_f32_tensor *source;
  float *prepared;
} et_f32_gradient_plan_entry;

struct et_f32_gradient_plan {
  uint64_t magic;
  et_f32_gradient_plan *registry_next;
  size_t count;
  et_f32_gradient_plan_entry *entries;
  uint64_t next_count;
  uint32_t next_weight_bits;
  uint32_t consumed;
};

struct et_f32_gradient_reset_plan {
  uint64_t magic;
  et_f32_gradient_reset_plan *registry_next;
  size_t count;
  et_f32_parameter **parameters;
  uint32_t consumed;
};

#define ET_F32_COPY_PLAN_MAGIC UINT64_C(0x4554463343504c4e)
#define ET_F32_PARAMETER_MAGIC UINT64_C(0x455446335041524d)
#define ET_F32_GRAD_PLAN_MAGIC UINT64_C(0x4554463347504c4e)
#define ET_F32_RESET_PLAN_MAGIC UINT64_C(0x4554463352504c4e)

static et_f32_tensor *live_tensors;
static et_f32_tensor_borrow *live_borrows;
static et_f32_tensor_copy_plan *live_copy_plans;
static et_f32_parameter *live_parameters;
static et_f32_gradient_plan *live_gradient_plans;
static et_f32_gradient_reset_plan *live_reset_plans;

_Static_assert(CHAR_BIT == 8, "I2 requires 8-bit bytes");
_Static_assert(sizeof(float) == 4u && sizeof(uint32_t) == 4u,
               "I2 requires 32-bit float and bit carriers");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128 && FLT_MIN_EXP == -125,
               "I2 requires IEEE-754 binary32 parameters");
_Static_assert(FLT_EVAL_METHOD == 0,
               "I2 requires binary32 expression evaluation");

#ifdef ET_F32_TENSOR_TESTING
static size_t allocation_limit = SIZE_MAX;
static size_t successful_allocations;

void et_f32_tensor_test_fail_alloc_after_v1(size_t allowed) {
  allocation_limit = allowed;
  successful_allocations = 0u;
}

void et_f32_tensor_test_reset_allocator_v1(void) {
  allocation_limit = SIZE_MAX;
  successful_allocations = 0u;
}
#endif

static void *f32_calloc(size_t count, size_t size) {
  void *allocation;
#ifdef ET_F32_TENSOR_TESTING
  if (successful_allocations >= allocation_limit) {
    return NULL;
  }
#endif
  allocation = calloc(count, size);
#ifdef ET_F32_TENSOR_TESTING
  if (allocation != NULL) {
    successful_allocations++;
  }
#endif
  return allocation;
}

static int pointer_span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int ranges_overlap(const void *left, size_t left_bytes,
                          const void *right, size_t right_bytes) {
  uintptr_t left_start;
  uintptr_t right_start;
  if (left_bytes == 0u || right_bytes == 0u) {
    return 0;
  }
  if (!pointer_span_fits(left, left_bytes) ||
      !pointer_span_fits(right, right_bytes)) {
    return 1;
  }
  left_start = (uintptr_t)left;
  right_start = (uintptr_t)right;
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static int aligned_pointer(const void *pointer, size_t alignment) {
  return pointer != NULL && (uintptr_t)pointer % alignment == 0u;
}

static et_f32_tensor *find_tensor(const void *candidate) {
  et_f32_tensor *cursor = live_tensors;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_f32_tensor_borrow *find_borrow(const void *candidate) {
  et_f32_tensor_borrow *cursor = live_borrows;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_f32_parameter *find_parameter(const void *candidate) {
  et_f32_parameter *cursor = live_parameters;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_f32_tensor_copy_plan *find_copy_plan(const void *candidate) {
  et_f32_tensor_copy_plan *cursor = live_copy_plans;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_f32_gradient_plan *find_gradient_plan(const void *candidate) {
  et_f32_gradient_plan *cursor = live_gradient_plans;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static et_f32_gradient_reset_plan *find_reset_plan(const void *candidate) {
  et_f32_gradient_reset_plan *cursor = live_reset_plans;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->registry_next;
  }
  return NULL;
}

static int valid_tensor(const et_f32_tensor *candidate) {
  et_f32_tensor *tensor = find_tensor(candidate);
  return tensor != NULL && tensor->magic == ET_F32_TENSOR_MAGIC;
}

static int valid_borrow(const et_f32_tensor_borrow *candidate) {
  et_f32_tensor_borrow *borrow = find_borrow(candidate);
  return borrow != NULL && borrow->magic == ET_F32_BORROW_MAGIC &&
         valid_tensor(borrow->owner) &&
         borrow->owner->active_borrow == borrow;
}

static void register_tensor(et_f32_tensor *tensor) {
  tensor->registry_next = live_tensors;
  live_tensors = tensor;
}

static void unregister_tensor(et_f32_tensor *tensor) {
  et_f32_tensor **cursor = &live_tensors;
  while (*cursor != NULL && *cursor != tensor) {
    cursor = &(*cursor)->registry_next;
  }
  if (*cursor == tensor) {
    *cursor = tensor->registry_next;
  }
}

static void register_borrow(et_f32_tensor_borrow *borrow) {
  borrow->registry_next = live_borrows;
  live_borrows = borrow;
}

static void unregister_borrow(et_f32_tensor_borrow *borrow) {
  et_f32_tensor_borrow **cursor = &live_borrows;
  while (*cursor != NULL && *cursor != borrow) {
    cursor = &(*cursor)->registry_next;
  }
  if (*cursor == borrow) {
    *cursor = borrow->registry_next;
  }
}

static int storage_aliases_live(const void *storage, size_t bytes) {
  const et_f32_tensor *tensor;
  const et_f32_tensor_borrow *borrow;
  const et_f32_tensor_copy_plan *copy_plan;
  const et_f32_parameter *parameter;
  const et_f32_gradient_plan *gradient_plan;
  const et_f32_gradient_reset_plan *reset_plan;
  for (tensor = live_tensors; tensor != NULL;
       tensor = tensor->registry_next) {
    if (ranges_overlap(storage, bytes, tensor, sizeof(*tensor)) ||
        ranges_overlap(storage, bytes, tensor->shape,
                       tensor->rank * sizeof(*tensor->shape)) ||
        ranges_overlap(storage, bytes, tensor->strides,
                       tensor->rank * sizeof(*tensor->strides)) ||
        ranges_overlap(storage, bytes, tensor->data, tensor->byte_length)) {
      return 1;
    }
  }
  for (borrow = live_borrows; borrow != NULL;
       borrow = borrow->registry_next) {
    if (ranges_overlap(storage, bytes, borrow, sizeof(*borrow))) {
      return 1;
    }
  }
  for (copy_plan = live_copy_plans; copy_plan != NULL;
       copy_plan = copy_plan->registry_next) {
    if (ranges_overlap(storage, bytes, copy_plan, sizeof(*copy_plan)) ||
        ranges_overlap(storage, bytes, copy_plan->assignments,
                       copy_plan->count * sizeof(*copy_plan->assignments))) {
      return 1;
    }
  }
  for (parameter = live_parameters; parameter != NULL;
       parameter = parameter->registry_next) {
    if (ranges_overlap(storage, bytes, parameter, sizeof(*parameter))) {
      return 1;
    }
  }
  for (gradient_plan = live_gradient_plans; gradient_plan != NULL;
       gradient_plan = gradient_plan->registry_next) {
    if (ranges_overlap(storage, bytes, gradient_plan, sizeof(*gradient_plan)) ||
        ranges_overlap(storage, bytes, gradient_plan->entries,
                       gradient_plan->count * sizeof(*gradient_plan->entries))) {
      return 1;
    }
    for (size_t index = 0u; index < gradient_plan->count; index++) {
      if (ranges_overlap(storage, bytes,
                         gradient_plan->entries[index].prepared,
                         gradient_plan->entries[index].parameter->gradient
                             ->byte_length)) {
        return 1;
      }
    }
  }
  for (reset_plan = live_reset_plans; reset_plan != NULL;
       reset_plan = reset_plan->registry_next) {
    if (ranges_overlap(storage, bytes, reset_plan, sizeof(*reset_plan)) ||
        ranges_overlap(storage, bytes, reset_plan->parameters,
                       reset_plan->count * sizeof(*reset_plan->parameters))) {
      return 1;
    }
  }
  return 0;
}

void et_f32_tensor_error_clear_v1(et_f32_tensor_error *error) {
  if (error != NULL && !storage_aliases_live(error, sizeof(*error))) {
    memset(error, 0, sizeof(*error));
  }
}

static int32_t set_error(et_f32_tensor_error *error,
                         et_f32_tensor_error_category category,
                         et_f32_tensor_error_code code,
                         const char *operation, const char *message) {
  if (error != NULL && storage_aliases_live(error, sizeof(*error))) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (error != NULL) {
    et_f32_tensor_error_clear_v1(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int32_t success(et_f32_tensor_error *error) {
  et_f32_tensor_error_clear_v1(error);
  return 0;
}

static int32_t preflight_error_operand(const et_f32_tensor_error *error,
                                       const void *operand, size_t bytes) {
  if (error != NULL &&
      (storage_aliases_live(error, sizeof(*error)) ||
       ranges_overlap(error, sizeof(*error), operand, bytes))) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  return 0;
}

static int32_t preflight_output(void *output, size_t bytes,
                                et_f32_tensor_error *error,
                                const char *operation) {
  int32_t result = preflight_error_operand(error, output, bytes);
  if (result != 0) {
    return result;
  }
  if (output != NULL && storage_aliases_live(output, bytes)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER, operation,
                     "output storage aliases a live I2 object");
  }
  return 0;
}

static int caller_shape_span(size_t rank, const uint64_t *shape,
                             size_t *bytes) {
  *bytes = 0u;
  if (rank == 0u || shape == NULL) {
    return 1;
  }
  if (rank > SIZE_MAX / sizeof(*shape)) {
    return 0;
  }
  *bytes = rank * sizeof(*shape);
  return pointer_span_fits(shape, *bytes);
}

static int32_t require_tensor(const et_f32_tensor *candidate,
                              const char *operation,
                              et_f32_tensor_error *error) {
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (candidate == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, operation,
                     "tensor handle is null");
  }
  if (!valid_tensor(candidate)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE, operation,
                     "tensor handle is foreign, stale, or invalid");
  }
  return 0;
}

static int32_t validate_shape(size_t rank, const uint64_t *shape,
                              size_t *count, size_t *bytes, int *empty,
                              et_f32_tensor_error *error) {
  size_t product = 1u;
  int has_zero = 0;
  if (rank > ET_KERNEL_MAX_RANK) {
    return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                     ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                     "f32-tensor-create", "tensor rank exceeds 64");
  }
  if (rank > 0u && shape == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                     ET_F32_TENSOR_CODE_INVALID_SHAPE,
                     "f32-tensor-create", "nonzero rank requires shape");
  }
  if (rank > 0u &&
      (!aligned_pointer(shape, _Alignof(uint64_t)) ||
       !pointer_span_fits(shape, rank * sizeof(*shape)))) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-tensor-create", "shape storage is invalid");
  }
  for (size_t index = 0u; index < rank; index++) {
    has_zero = has_zero || shape[index] == 0u;
  }
  if (has_zero) {
    product = 0u;
  } else {
    for (size_t index = 0u; index < rank; index++) {
      if (shape[index] > SIZE_MAX ||
          product > SIZE_MAX / (size_t)shape[index]) {
        return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                         ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                         "f32-tensor-create", "element count overflows");
      }
      product *= (size_t)shape[index];
    }
    if (product > SIZE_MAX / sizeof(float)) {
      return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                       ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                       "f32-tensor-create", "byte length overflows");
    }
  }
  *count = product;
  *bytes = product * sizeof(float);
  *empty = has_zero;
  return 0;
}

int32_t et_f32_tensor_abi_major_v1(void) {
  return (int32_t)ET_F32_TENSOR_ABI_MAJOR;
}

int32_t et_f32_tensor_abi_minor_v1(void) {
  return (int32_t)ET_F32_TENSOR_ABI_MINOR;
}

int32_t et_f32_tensor_abi_require_v1(uint32_t major, uint32_t minimum_minor,
                                     et_f32_tensor_error *error) {
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (major != ET_F32_TENSOR_ABI_MAJOR ||
      minimum_minor > ET_F32_TENSOR_ABI_MINOR) {
    return set_error(error, ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
                     ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH,
                     "f32-tensor-abi-require", "requested ABI is unavailable");
  }
  return success(error);
}

int32_t et_f32_tensor_create_v1(size_t rank, const uint64_t *shape,
                                et_f32_tensor **output,
                                et_f32_tensor_error *error) {
  et_f32_tensor *tensor;
  size_t count;
  size_t bytes;
  size_t shape_bytes;
  int empty;
  int32_t result;
  if (!caller_shape_span(rank, shape, &shape_bytes)) {
    if (error != NULL) {
      return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
    }
    shape_bytes = 0u;
  }
  result = preflight_error_operand(error, shape, shape_bytes);
  if (result != 0) {
    return result;
  }
  result = preflight_output(output, output == NULL ? 0u : sizeof(*output),
                            error, "f32-tensor-create");
  if (result != 0) {
    return result;
  }
  if (output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, "f32-tensor-create",
                     "tensor output is null");
  }
  if (*output != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER, "f32-tensor-create",
                     "tensor output must initially be null");
  }
  result = validate_shape(rank, shape, &count, &bytes, &empty, error);
  if (result != 0) {
    return result;
  }
  tensor = (et_f32_tensor *)f32_calloc(1u, sizeof(*tensor));
  if (tensor == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-tensor-create", "cannot allocate tensor control");
  }
  tensor->rank = rank;
  tensor->element_count = count;
  tensor->byte_length = bytes;
  if (rank > 0u) {
    tensor->shape = (uint64_t *)f32_calloc(rank, sizeof(*tensor->shape));
    tensor->strides = (size_t *)f32_calloc(rank, sizeof(*tensor->strides));
    if (tensor->shape == NULL || tensor->strides == NULL) {
      free(tensor->strides);
      free(tensor->shape);
      free(tensor);
      return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                       ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                       "f32-tensor-create", "cannot allocate tensor metadata");
    }
    memcpy(tensor->shape, shape, rank * sizeof(*shape));
    if (!empty) {
      size_t running = sizeof(float);
      for (size_t index = rank; index > 0u; index--) {
        size_t dimension = index - 1u;
        tensor->strides[dimension] = running;
        running *= (size_t)shape[dimension];
      }
    }
  }
  if (bytes > 0u) {
    tensor->data = (float *)f32_calloc(count, sizeof(*tensor->data));
    if (tensor->data == NULL) {
      free(tensor->strides);
      free(tensor->shape);
      free(tensor);
      return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                       ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                       "f32-tensor-create", "cannot allocate tensor storage");
    }
  }
  tensor->magic = ET_F32_TENSOR_MAGIC;
  (void)success(error);
  register_tensor(tensor);
  *output = tensor;
  return 0;
}

int32_t et_f32_tensor_destroy_v1(et_f32_tensor **slot,
                                 et_f32_tensor_error *error) {
  et_f32_tensor *tensor;
  int32_t result = preflight_output(
      slot, slot == NULL ? 0u : sizeof(*slot), error, "f32-tensor-destroy");
  if (result != 0) {
    return result;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, "f32-tensor-destroy",
                     "tensor handle slot is null");
  }
  tensor = *slot;
  if (tensor == NULL) {
    return success(error);
  }
  result = require_tensor(tensor, "f32-tensor-destroy", error);
  if (result != 0) {
    return result;
  }
  if (tensor->active_borrow != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_ACTIVE_BORROW, "f32-tensor-destroy",
                     "tensor has an active borrow");
  }
  if (tensor->plan_pins != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE, "f32-tensor-destroy",
                     "tensor is retained by a prepared plan");
  }
  (void)success(error);
  unregister_tensor(tensor);
  tensor->magic = 0u;
  free(tensor->data);
  free(tensor->strides);
  free(tensor->shape);
  free(tensor);
  *slot = NULL;
  return 0;
}

static int32_t scalar_output(const et_f32_tensor *tensor, void *output,
                             size_t output_bytes, const char *operation,
                             et_f32_tensor_error *error) {
  int32_t result = preflight_output(output, output_bytes, error, operation);
  if (result != 0) {
    return result;
  }
  result = require_tensor(tensor, operation, error);
  if (result != 0) {
    return result;
  }
  if (output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, operation,
                     "scalar output is null");
  }
  return 0;
}

int32_t et_f32_tensor_rank_v1(const et_f32_tensor *tensor, size_t *rank,
                              et_f32_tensor_error *error) {
  int32_t result = scalar_output(tensor, rank,
                                 rank == NULL ? 0u : sizeof(*rank),
                                 "f32-tensor-rank", error);
  if (result == 0) {
    (void)success(error);
    *rank = tensor->rank;
  }
  return result;
}

int32_t et_f32_tensor_shape_at_v1(const et_f32_tensor *tensor,
                                  size_t dimension, uint64_t *extent,
                                  et_f32_tensor_error *error) {
  int32_t result = scalar_output(tensor, extent,
                                 extent == NULL ? 0u : sizeof(*extent),
                                 "f32-tensor-shape-at", error);
  if (result != 0) {
    return result;
  }
  if (dimension >= tensor->rank) {
    return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                     ET_F32_TENSOR_CODE_INVALID_SHAPE,
                     "f32-tensor-shape-at", "dimension is out of range");
  }
  (void)success(error);
  *extent = tensor->shape[dimension];
  return 0;
}

int32_t et_f32_tensor_stride_bytes_at_v1(const et_f32_tensor *tensor,
                                         size_t dimension, size_t *stride,
                                         et_f32_tensor_error *error) {
  int32_t result = scalar_output(tensor, stride,
                                 stride == NULL ? 0u : sizeof(*stride),
                                 "f32-tensor-stride-at", error);
  if (result != 0) {
    return result;
  }
  if (dimension >= tensor->rank) {
    return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                     ET_F32_TENSOR_CODE_INVALID_SHAPE,
                     "f32-tensor-stride-at", "dimension is out of range");
  }
  (void)success(error);
  *stride = tensor->strides[dimension];
  return 0;
}

int32_t et_f32_tensor_element_count_v1(const et_f32_tensor *tensor,
                                       size_t *count,
                                       et_f32_tensor_error *error) {
  int32_t result = scalar_output(tensor, count,
                                 count == NULL ? 0u : sizeof(*count),
                                 "f32-tensor-element-count", error);
  if (result == 0) {
    (void)success(error);
    *count = tensor->element_count;
  }
  return result;
}

int32_t et_f32_tensor_byte_length_v1(const et_f32_tensor *tensor,
                                     size_t *bytes,
                                     et_f32_tensor_error *error) {
  int32_t result = scalar_output(tensor, bytes,
                                 bytes == NULL ? 0u : sizeof(*bytes),
                                 "f32-tensor-byte-length", error);
  if (result == 0) {
    (void)success(error);
    *bytes = tensor->byte_length;
  }
  return result;
}

static int32_t validate_bits_buffer(const et_f32_tensor *tensor,
                                    const void *buffer, size_t count,
                                    const char *operation,
                                    et_f32_tensor_error *error) {
  if (count != tensor->element_count) {
    return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                     ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH, operation,
                     "copy element count differs from tensor");
  }
  if (tensor->byte_length == 0u) {
    return 0;
  }
  if (buffer == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, operation,
                     "nonempty copy buffer is null");
  }
  if (!aligned_pointer(buffer, _Alignof(uint32_t)) ||
      !pointer_span_fits(buffer, tensor->byte_length) ||
      storage_aliases_live(buffer, tensor->byte_length)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER, operation,
                     "copy buffer is invalid or aliases I2 storage");
  }
  return 0;
}

int32_t et_f32_tensor_copy_bits_from_v1(et_f32_tensor *tensor,
                                        const uint32_t *source,
                                        size_t count,
                                        et_f32_tensor_error *error) {
  size_t declared_bytes =
      count <= SIZE_MAX / sizeof(*source) ? count * sizeof(*source) : SIZE_MAX;
  int32_t result = preflight_error_operand(error, source, declared_bytes);
  if (result != 0) {
    return result;
  }
  result = require_tensor(tensor, "f32-tensor-copy-bits-from", error);
  if (result != 0) {
    return result;
  }
  result = validate_bits_buffer(tensor, source, count,
                                "f32-tensor-copy-bits-from", error);
  if (result != 0) {
    return result;
  }
  if (tensor->active_borrow != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_ACTIVE_BORROW,
                     "f32-tensor-copy-bits-from",
                     "tensor cannot be mutated while borrowed");
  }
  if (tensor->plan_pins != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-tensor-copy-bits-from",
                     "tensor is retained by a prepared plan");
  }
  if (tensor->byte_length > 0u) {
    memcpy(tensor->data, source, tensor->byte_length);
  }
  return success(error);
}

int32_t et_f32_tensor_copy_bits_to_v1(const et_f32_tensor *tensor,
                                      uint32_t *destination, size_t count,
                                      et_f32_tensor_error *error) {
  size_t declared_bytes = count <= SIZE_MAX / sizeof(*destination)
                              ? count * sizeof(*destination)
                              : SIZE_MAX;
  int32_t result = preflight_output(
      destination, destination == NULL ? 0u : declared_bytes, error,
      "f32-tensor-copy-bits-to");
  if (result != 0) {
    return result;
  }
  result = require_tensor(tensor, "f32-tensor-copy-bits-to", error);
  if (result != 0) {
    return result;
  }
  result = validate_bits_buffer(tensor, destination, count,
                                "f32-tensor-copy-bits-to", error);
  if (result != 0) {
    return result;
  }
  if (tensor->byte_length > 0u) {
    memcpy(destination, tensor->data, tensor->byte_length);
  }
  return success(error);
}

static int same_shape(const et_f32_tensor *left,
                      const et_f32_tensor *right) {
  return left->rank == right->rank &&
         left->byte_length == right->byte_length &&
         (left->rank == 0u ||
          memcmp(left->shape, right->shape,
                 left->rank * sizeof(*left->shape)) == 0);
}

int32_t et_f32_tensor_clone_v1(const et_f32_tensor *tensor,
                               et_f32_tensor **output,
                               et_f32_tensor_error *error) {
  et_f32_tensor *clone = NULL;
  int32_t result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error,
      "f32-tensor-clone");
  if (result != 0) {
    return result;
  }
  if (output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, "f32-tensor-clone",
                     "clone output is null");
  }
  if (*output != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER, "f32-tensor-clone",
                     "clone output must initially be null");
  }
  result = require_tensor(tensor, "f32-tensor-clone", error);
  if (result != 0) {
    return result;
  }
  result = et_f32_tensor_create_v1(tensor->rank, tensor->shape, &clone, error);
  if (result != 0) {
    return result;
  }
  if (tensor->byte_length > 0u) {
    memcpy(clone->data, tensor->data, tensor->byte_length);
  }
  *output = clone;
  return success(error);
}

int32_t et_f32_tensor_storage_identical_v1(const et_f32_tensor *left,
                                           const et_f32_tensor *right,
                                           int32_t *identical,
                                           et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      identical, identical == NULL ? 0u : sizeof(*identical), error,
      "f32-tensor-storage-identical");
  if (result == 0) {
    result = require_tensor(left, "f32-tensor-storage-identical", error);
  }
  if (result == 0) {
    result = require_tensor(right, "f32-tensor-storage-identical", error);
  }
  if (result != 0) {
    return result;
  }
  if (identical == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-tensor-storage-identical", "output is null");
  }
  *identical = left == right ? 1 : 0;
  return success(error);
}

int32_t et_f32_tensor_bits_equal_v1(const et_f32_tensor *left,
                                    const et_f32_tensor *right,
                                    int32_t *equal,
                                    et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      equal, equal == NULL ? 0u : sizeof(*equal), error,
      "f32-tensor-bits-equal");
  if (result == 0) {
    result = require_tensor(left, "f32-tensor-bits-equal", error);
  }
  if (result == 0) {
    result = require_tensor(right, "f32-tensor-bits-equal", error);
  }
  if (result != 0) {
    return result;
  }
  if (equal == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-tensor-bits-equal", "output is null");
  }
  *equal = same_shape(left, right) &&
                   (left->byte_length == 0u ||
                    memcmp(left->data, right->data, left->byte_length) == 0)
               ? 1
               : 0;
  return success(error);
}

int32_t et_f32_tensor_copy_plan_prepare_v1(
    size_t count, const et_f32_tensor_copy_assignment_v1 *assignments,
    et_f32_tensor_copy_plan **output, et_f32_tensor_error *error) {
  et_f32_tensor_copy_plan *plan;
  size_t assignment_bytes;
  int32_t result;
  result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error,
      "f32-copy-plan-prepare");
  if (result != 0) {
    return result;
  }
  if (count == 0u || count > ET_F32_PARAMETER_MAX_BATCH ||
      count > SIZE_MAX / sizeof(*assignments)) {
    if (assignments != NULL && count > 0u) {
      if (count > SIZE_MAX / sizeof(*assignments) ||
          !pointer_span_fits(assignments, count * sizeof(*assignments))) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
      if (preflight_error_operand(error, assignments,
                                  count * sizeof(*assignments)) != 0) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
    }
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                     "f32-copy-plan-prepare", "assignment count is invalid");
  }
  assignment_bytes = count * sizeof(*assignments);
  result = preflight_error_operand(error, assignments, assignment_bytes);
  if (result != 0) {
    return result;
  }
  if (assignments == NULL || output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-copy-plan-prepare", "assignments and output required");
  }
  if (*output != NULL ||
      !aligned_pointer(assignments, _Alignof(et_f32_tensor_copy_assignment_v1)) ||
      !pointer_span_fits(assignments, assignment_bytes)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-copy-plan-prepare", "assignment storage is invalid");
  }
  for (size_t index = 0u; index < count; index++) {
    const et_f32_tensor_copy_assignment_v1 *item = &assignments[index];
    if (item->struct_size != sizeof(*item)) {
      return set_error(error, ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
                       ET_F32_TENSOR_CODE_INVALID_BUFFER,
                       "f32-copy-plan-prepare", "assignment struct size differs");
    }
    result = require_tensor(item->destination, "f32-copy-plan-prepare", error);
    if (result != 0) {
      return result;
    }
    result = require_tensor(item->source, "f32-copy-plan-prepare", error);
    if (result != 0) {
      return result;
    }
    if (!same_shape(item->destination, item->source)) {
      return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                       ET_F32_TENSOR_CODE_INVALID_SHAPE,
                       "f32-copy-plan-prepare", "source shape differs");
    }
    if (item->destination->active_borrow != NULL ||
        item->source->active_borrow != NULL ||
        item->destination->plan_pins != 0u || item->source->plan_pins != 0u) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-copy-plan-prepare", "tensor is borrowed or pinned");
    }
    for (size_t prior = 0u; prior < index; prior++) {
      if (item->destination == assignments[prior].destination ||
          item->source == assignments[prior].source) {
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-copy-plan-prepare", "batch storage is duplicated");
      }
    }
    for (size_t other = 0u; other < count; other++) {
      if (item->destination == assignments[other].source) {
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-copy-plan-prepare", "source aliases destination");
      }
    }
  }
  plan = (et_f32_tensor_copy_plan *)f32_calloc(1u, sizeof(*plan));
  if (plan == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-copy-plan-prepare", "cannot allocate copy plan");
  }
  plan->assignments = (et_f32_tensor_copy_assignment_v1 *)
      f32_calloc(count, sizeof(*plan->assignments));
  if (plan->assignments == NULL) {
    free(plan);
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-copy-plan-prepare", "cannot allocate assignments");
  }
  memcpy(plan->assignments, assignments, assignment_bytes);
  plan->count = count;
  plan->magic = ET_F32_COPY_PLAN_MAGIC;
  for (size_t index = 0u; index < count; index++) {
    plan->assignments[index].destination->plan_pins++;
    ((et_f32_tensor *)plan->assignments[index].source)->plan_pins++;
  }
  plan->registry_next = live_copy_plans;
  live_copy_plans = plan;
  *output = plan;
  return success(error);
}

int32_t et_f32_tensor_copy_plan_commit_v1(
    et_f32_tensor_copy_plan *candidate, et_f32_tensor_error *error) {
  et_f32_tensor_copy_plan *plan = find_copy_plan(candidate);
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (plan == NULL || plan->magic != ET_F32_COPY_PLAN_MAGIC ||
      plan->consumed != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-copy-plan-commit", "plan is foreign, stale, or consumed");
  }
  plan->consumed = 1u;
  for (size_t index = 0u; index < plan->count; index++) {
    et_f32_tensor *destination = plan->assignments[index].destination;
    const et_f32_tensor *source = plan->assignments[index].source;
    if (destination->byte_length > 0u) {
      memcpy(destination->data, source->data, destination->byte_length);
    }
  }
  return success(error);
}

int32_t et_f32_tensor_copy_plan_release_v1(
    et_f32_tensor_copy_plan **slot, et_f32_tensor_error *error) {
  et_f32_tensor_copy_plan **cursor;
  et_f32_tensor_copy_plan *plan;
  int32_t preflight = preflight_output(
      slot, slot == NULL ? 0u : sizeof(*slot), error,
      "f32-copy-plan-release");
  if (preflight != 0) {
    return preflight;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-copy-plan-release", "plan slot is null");
  }
  plan = *slot;
  if (plan == NULL) {
    return success(error);
  }
  if (find_copy_plan(plan) == NULL || plan->magic != ET_F32_COPY_PLAN_MAGIC) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-copy-plan-release", "plan is foreign or stale");
  }
  cursor = &live_copy_plans;
  while (*cursor != plan) {
    cursor = &(*cursor)->registry_next;
  }
  *cursor = plan->registry_next;
  for (size_t index = 0u; index < plan->count; index++) {
    plan->assignments[index].destination->plan_pins--;
    ((et_f32_tensor *)plan->assignments[index].source)->plan_pins--;
  }
  plan->magic = 0u;
  free(plan->assignments);
  free(plan);
  *slot = NULL;
  return success(error);
}

static int32_t require_parameter(const et_f32_parameter *candidate,
                                 const char *operation,
                                 et_f32_tensor_error *error) {
  et_f32_parameter *parameter = find_parameter(candidate);
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (candidate == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, operation,
                     "parameter is null");
  }
  if (parameter == NULL || parameter->magic != ET_F32_PARAMETER_MAGIC) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE, operation,
                     "parameter is foreign or stale");
  }
  return 0;
}

int32_t et_f32_tensor_is_live_v1(const et_f32_tensor *tensor) {
  return valid_tensor(tensor) ? 1 : 0;
}

int32_t et_f32_parameter_is_live_v1(const et_f32_parameter *parameter) {
  et_f32_parameter *found = find_parameter(parameter);
  return found != NULL && found->magic == ET_F32_PARAMETER_MAGIC ? 1 : 0;
}

static int bits_finite(uint32_t bits) {
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int bits_positive_finite(uint32_t bits) {
  return (bits & UINT32_C(0x80000000)) == 0u &&
         (bits & UINT32_C(0x7fffffff)) != 0u && bits_finite(bits);
}

static int tensor_finite(const et_f32_tensor *tensor) {
  for (size_t index = 0u; index < tensor->element_count; index++) {
    uint32_t bits;
    memcpy(&bits, &tensor->data[index], sizeof(bits));
    if (!bits_finite(bits)) {
      return 0;
    }
  }
  return 1;
}

static int tensor_exact_positive_zero(const et_f32_tensor *tensor) {
  for (size_t index = 0u; index < tensor->element_count; index++) {
    uint32_t bits;
    memcpy(&bits, &tensor->data[index], sizeof(bits));
    if (bits != 0u) {
      return 0;
    }
  }
  return 1;
}

int32_t et_f32_parameter_create_v1(const et_f32_tensor *initial,
                                   et_f32_parameter **output,
                                   et_f32_tensor_error *error) {
  et_f32_parameter *parameter;
  et_f32_tensor *value = NULL;
  et_f32_tensor *gradient = NULL;
  int32_t result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error,
      "f32-parameter-create");
  if (result != 0) {
    return result;
  }
  if (output == NULL || initial == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-parameter-create", "initial value and output required");
  }
  if (*output != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-parameter-create", "output must initially be null");
  }
  result = require_tensor(initial, "f32-parameter-create", error);
  if (result != 0) {
    return result;
  }
  result = et_f32_tensor_clone_v1(initial, &value, error);
  if (result != 0) {
    return result;
  }
  result = et_f32_tensor_create_v1(initial->rank, initial->shape, &gradient,
                                   error);
  if (result != 0) {
    (void)et_f32_tensor_destroy_v1(&value, NULL);
    return result;
  }
  parameter = (et_f32_parameter *)f32_calloc(1u, sizeof(*parameter));
  if (parameter == NULL) {
    (void)et_f32_tensor_destroy_v1(&gradient, NULL);
    (void)et_f32_tensor_destroy_v1(&value, NULL);
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-parameter-create", "cannot allocate parameter carrier");
  }
  parameter->value = value;
  parameter->gradient = gradient;
  parameter->magic = ET_F32_PARAMETER_MAGIC;
  parameter->registry_next = live_parameters;
  live_parameters = parameter;
  *output = parameter;
  return success(error);
}

int32_t et_f32_parameter_destroy_v1(et_f32_parameter **slot,
                                    et_f32_tensor_error *error) {
  et_f32_parameter **cursor;
  et_f32_parameter *parameter;
  int32_t result;
  result = preflight_output(slot, slot == NULL ? 0u : sizeof(*slot), error,
                            "f32-parameter-destroy");
  if (result != 0) {
    return result;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-parameter-destroy", "parameter slot is null");
  }
  parameter = *slot;
  if (parameter == NULL) {
    return success(error);
  }
  result = require_parameter(parameter, "f32-parameter-destroy", error);
  if (result != 0) {
    return result;
  }
  if (parameter->plan_pins != 0u ||
      parameter->value->active_borrow != NULL ||
      parameter->gradient->active_borrow != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-parameter-destroy", "parameter is borrowed or pinned");
  }
  cursor = &live_parameters;
  while (*cursor != parameter) {
    cursor = &(*cursor)->registry_next;
  }
  *cursor = parameter->registry_next;
  parameter->magic = 0u;
  (void)et_f32_tensor_destroy_v1(&parameter->gradient, NULL);
  (void)et_f32_tensor_destroy_v1(&parameter->value, NULL);
  free(parameter);
  *slot = NULL;
  return success(error);
}

int32_t et_f32_parameter_bind_identity_v1(et_f32_parameter *parameter,
                                          const void *identity,
                                          et_f32_tensor_error *error) {
  int32_t result = require_parameter(parameter, "f32-parameter-bind", error);
  if (result != 0) {
    return result;
  }
  if (identity == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-parameter-bind", "identity is null");
  }
  if (parameter->identity != NULL && parameter->identity != identity) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-parameter-bind", "parameter is already bound");
  }
  for (et_f32_parameter *cursor = live_parameters; cursor != NULL;
       cursor = cursor->registry_next) {
    if (cursor != parameter && cursor->identity == identity) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-parameter-bind", "identity is already bound");
    }
  }
  parameter->identity = identity;
  return success(error);
}

int32_t et_f32_parameter_identity_v1(const et_f32_parameter *parameter,
                                     const void **identity,
                                     et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      (void *)identity, identity == NULL ? 0u : sizeof(*identity), error,
      "f32-parameter-identity");
  if (result == 0) {
    result = require_parameter(parameter, "f32-parameter-identity", error);
  }
  if (result != 0) {
    return result;
  }
  if (identity == NULL || parameter->identity == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-parameter-identity", "parameter is unbound");
  }
  *identity = parameter->identity;
  return success(error);
}

int32_t et_f32_parameter_value_snapshot_v1(const et_f32_parameter *parameter,
                                           et_f32_tensor **snapshot,
                                           et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      snapshot, snapshot == NULL ? 0u : sizeof(*snapshot), error,
      "f32-value-snapshot");
  if (result == 0) {
    result = require_parameter(parameter, "f32-value-snapshot", error);
  }
  return result != 0 ? result
                     : et_f32_tensor_clone_v1(parameter->value, snapshot, error);
}

int32_t et_f32_parameter_value_tensor_v1(
    const et_f32_parameter *parameter, const et_f32_tensor **value,
    et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      (void *)value, value == NULL ? 0u : sizeof(*value), error,
      "f32-parameter-value-tensor");
  if (result == 0) {
    result = require_parameter(parameter, "f32-parameter-value-tensor", error);
  }
  if (result != 0) {
    return result;
  }
  if (value == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-parameter-value-tensor", "value output is null");
  }
  if (*value != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-parameter-value-tensor",
                     "value output must initially be null");
  }
  *value = parameter->value;
  return success(error);
}

int32_t et_f32_parameter_value_borrow_begin_v1(
    et_f32_parameter *parameter, et_f32_tensor_borrow **borrow,
    et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      borrow, borrow == NULL ? 0u : sizeof(*borrow), error,
      "f32-value-borrow");
  if (result == 0) {
    result = require_parameter(parameter, "f32-value-borrow", error);
  }
  return result != 0 ? result
                     : et_f32_tensor_borrow_begin_v1(parameter->value, borrow,
                                                     error);
}

int32_t et_f32_parameter_gradient_metadata_v1(
    const et_f32_parameter *parameter,
    et_f32_gradient_metadata_v1 *metadata, et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      metadata, metadata == NULL ? 0u : sizeof(*metadata), error,
      "f32-gradient-metadata");
  if (result == 0) {
    result = require_parameter(parameter, "f32-gradient-metadata", error);
  }
  if (result != 0) {
    return result;
  }
  if (metadata == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-gradient-metadata", "metadata output is null");
  }
  if (metadata->struct_size != sizeof(*metadata)) {
    return set_error(error, ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-gradient-metadata", "metadata struct size differs");
  }
  metadata->state = parameter->gradient_state;
  metadata->normalization_weight_bits = parameter->normalization_weight_bits;
  metadata->contribution_count = parameter->contribution_count;
  return success(error);
}

static int32_t require_present(const et_f32_parameter *parameter,
                               const char *operation,
                               et_f32_tensor_error *error) {
  int32_t result = require_parameter(parameter, operation, error);
  if (result != 0) {
    return result;
  }
  if (parameter->gradient_state != ET_F32_GRADIENT_PRESENT) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE, operation,
                     "gradient is absent");
  }
  return 0;
}

static int32_t gradient_boolean(const et_f32_parameter *parameter,
                                int32_t *output, int exact_zero,
                                const char *operation,
                                et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error, operation);
  if (result == 0) {
    result = require_present(parameter, operation, error);
  }
  if (result != 0) {
    return result;
  }
  if (output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT, operation,
                     "inspection output is null");
  }
  *output = exact_zero ? tensor_exact_positive_zero(parameter->gradient)
                       : tensor_finite(parameter->gradient);
  return success(error);
}

int32_t et_f32_parameter_gradient_finite_v1(
    const et_f32_parameter *parameter, int32_t *finite,
    et_f32_tensor_error *error) {
  return gradient_boolean(parameter, finite, 0, "f32-gradient-finite", error);
}

int32_t et_f32_parameter_gradient_exact_positive_zero_v1(
    const et_f32_parameter *parameter, int32_t *exact_zero,
    et_f32_tensor_error *error) {
  return gradient_boolean(parameter, exact_zero, 1,
                          "f32-gradient-exact-zero", error);
}

int32_t et_f32_parameter_gradient_snapshot_v1(
    const et_f32_parameter *parameter, et_f32_tensor **snapshot,
    et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      snapshot, snapshot == NULL ? 0u : sizeof(*snapshot), error,
      "f32-gradient-snapshot");
  if (result == 0) {
    result = require_present(parameter, "f32-gradient-snapshot", error);
  }
  return result != 0
             ? result
             : et_f32_tensor_clone_v1(parameter->gradient, snapshot, error);
}

int32_t et_f32_parameter_gradient_borrow_begin_v1(
    et_f32_parameter *parameter, et_f32_tensor_borrow **borrow,
    et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      borrow, borrow == NULL ? 0u : sizeof(*borrow), error,
      "f32-gradient-borrow");
  if (result == 0) {
    result = require_present(parameter, "f32-gradient-borrow", error);
  }
  return result != 0
             ? result
             : et_f32_tensor_borrow_begin_v1(parameter->gradient, borrow,
                                             error);
}

static int parameter_metadata_valid(const et_f32_parameter *parameter) {
  if (parameter->gradient_state == ET_F32_GRADIENT_ABSENT) {
    return parameter->contribution_count == 0u &&
           parameter->normalization_weight_bits == 0u &&
           tensor_exact_positive_zero(parameter->gradient);
  }
  return parameter->gradient_state == ET_F32_GRADIENT_PRESENT &&
         parameter->contribution_count > 0u &&
         parameter->contribution_count <= (uint64_t)INT64_MAX &&
         bits_positive_finite(parameter->normalization_weight_bits) &&
         tensor_finite(parameter->gradient);
}

static int f32_environment_admitted(uint32_t control) {
  /* x86-64 Clang evaluates scalar float arithmetic through SSE. Require
   * round-to-nearest-even, masked exceptions, and no DAZ/FTZ substitution. */
  return (control & UINT32_C(0x0000ffc0)) == UINT32_C(0x00001f80);
}

static int add_f32_bits(uint32_t left_bits, uint32_t right_bits,
                        uint32_t *result) {
  float left;
  float right;
  volatile float sum;
  float rounded;
  uint32_t control = _mm_getcsr();
  if (!f32_environment_admitted(control)) {
    return 0;
  }
  memcpy(&left, &left_bits, sizeof(left));
  memcpy(&right, &right_bits, sizeof(right));
  sum = left + right;
  rounded = sum;
  memcpy(result, &rounded, sizeof(*result));
  _mm_setcsr(control);
  return 1;
}

int32_t et_f32_gradient_plan_prepare_v1(
    size_t count, const et_f32_gradient_contribution_v1 *contributions,
    uint32_t weight_increment_bits, et_f32_gradient_plan **output,
    et_f32_tensor_error *error) {
  et_f32_gradient_plan *plan;
  uint32_t expected_state = 0u;
  uint32_t expected_weight = 0u;
  uint64_t expected_count = 0u;
  size_t contribution_bytes;
  int32_t result;
  result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error,
      "f32-gradient-prepare");
  if (result != 0) {
    return result;
  }
  if (count == 0u || count > ET_F32_PARAMETER_MAX_BATCH ||
      count > SIZE_MAX / sizeof(*contributions)) {
    if (contributions != NULL && count > 0u) {
      if (count > SIZE_MAX / sizeof(*contributions) ||
          !pointer_span_fits(contributions,
                             count * sizeof(*contributions))) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
      if (preflight_error_operand(error, contributions,
                                  count * sizeof(*contributions)) != 0) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
    }
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                     "f32-gradient-prepare", "contribution count is invalid");
  }
  contribution_bytes = count * sizeof(*contributions);
  result = preflight_error_operand(error, contributions, contribution_bytes);
  if (result != 0) {
    return result;
  }
  if (contributions == NULL || output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-gradient-prepare", "contributions and output required");
  }
  if (*output != NULL ||
      !aligned_pointer(contributions,
                       _Alignof(et_f32_gradient_contribution_v1)) ||
      !pointer_span_fits(contributions, contribution_bytes)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-gradient-prepare", "contribution storage is invalid");
  }
  if (!bits_positive_finite(weight_increment_bits)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-gradient-prepare", "weight increment is not positive finite");
  }
  if (!f32_environment_admitted(_mm_getcsr())) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT,
                     "f32-gradient-prepare",
                     "binary32 environment is not deterministic");
  }
  for (size_t index = 0u; index < count; index++) {
    const et_f32_gradient_contribution_v1 *item = &contributions[index];
    et_f32_parameter *parameter;
    if (item->struct_size != sizeof(*item)) {
      return set_error(error, ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
                       ET_F32_TENSOR_CODE_INVALID_BUFFER,
                       "f32-gradient-prepare", "contribution struct size differs");
    }
    result = require_parameter(item->destination, "f32-gradient-prepare", error);
    if (result != 0) {
      return result;
    }
    result = require_tensor(item->weighted_numerator, "f32-gradient-prepare",
                            error);
    if (result != 0) {
      return result;
    }
    parameter = item->destination;
    if (parameter->identity == NULL) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-prepare", "parameter metadata is invalid");
    }
    if (parameter->contribution_count >= (uint64_t)INT64_MAX) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                       "f32-gradient-prepare", "contribution count overflows");
    }
    if (!parameter_metadata_valid(parameter)) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-prepare", "parameter metadata is invalid");
    }
    if (item->expected_ordinal != parameter->contribution_count) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-prepare", "contribution ordinal differs");
    }
    if (!same_shape(parameter->gradient, item->weighted_numerator)) {
      return set_error(error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                       ET_F32_TENSOR_CODE_INVALID_SHAPE,
                       "f32-gradient-prepare", "gradient shape differs");
    }
    if (!tensor_finite(item->weighted_numerator)) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                       ET_F32_TENSOR_CODE_INVALID_BUFFER,
                       "f32-gradient-prepare", "contribution is nonfinite");
    }
    if (parameter->plan_pins != 0u ||
        parameter->gradient->active_borrow != NULL ||
        parameter->gradient->plan_pins != 0u ||
        item->weighted_numerator->active_borrow != NULL ||
        item->weighted_numerator->plan_pins != 0u) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-prepare", "gradient storage is borrowed or pinned");
    }
    if (index == 0u) {
      expected_state = parameter->gradient_state;
      expected_weight = parameter->normalization_weight_bits;
      expected_count = parameter->contribution_count;
    } else if (parameter->gradient_state != expected_state ||
               parameter->normalization_weight_bits != expected_weight ||
               parameter->contribution_count != expected_count) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-prepare", "batch pre-state metadata differs");
    }
    for (size_t prior = 0u; prior < index; prior++) {
      if (parameter == contributions[prior].destination ||
          parameter->identity == contributions[prior].destination->identity ||
          item->weighted_numerator ==
              contributions[prior].weighted_numerator) {
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-gradient-prepare", "batch destination or source repeats");
      }
    }
    for (size_t other = 0u; other < count; other++) {
      et_f32_parameter *other_parameter =
          find_parameter(contributions[other].destination);
      if (other_parameter != NULL &&
          (item->weighted_numerator == other_parameter->value ||
           item->weighted_numerator == other_parameter->gradient)) {
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-gradient-prepare",
                         "contribution aliases parameter storage");
      }
    }
  }
  {
    uint32_t next_weight;
    if (!add_f32_bits(expected_weight, weight_increment_bits, &next_weight)) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT,
                       "f32-gradient-prepare",
                       "binary32 environment changed during preparation");
    }
    if (!bits_positive_finite(next_weight)) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                       ET_F32_TENSOR_CODE_INVALID_BUFFER,
                       "f32-gradient-prepare", "normalization weight became nonfinite");
    }
    plan = (et_f32_gradient_plan *)f32_calloc(1u, sizeof(*plan));
    if (plan == NULL) {
      return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                       ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                       "f32-gradient-prepare", "cannot allocate gradient plan");
    }
    plan->entries = (et_f32_gradient_plan_entry *)
        f32_calloc(count, sizeof(*plan->entries));
    if (plan->entries == NULL) {
      free(plan);
      return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                       ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                       "f32-gradient-prepare", "cannot allocate gradient entries");
    }
    plan->count = count;
    plan->next_count = expected_count + 1u;
    plan->next_weight_bits = next_weight;
  }
  for (size_t index = 0u; index < count; index++) {
    et_f32_parameter *parameter = contributions[index].destination;
    const et_f32_tensor *source = contributions[index].weighted_numerator;
    plan->entries[index].parameter = parameter;
    plan->entries[index].source = source;
    if (parameter->gradient->byte_length > 0u) {
      plan->entries[index].prepared =
          (float *)f32_calloc(parameter->gradient->element_count, sizeof(float));
      if (plan->entries[index].prepared == NULL) {
        for (size_t prior = 0u; prior < index; prior++) {
          free(plan->entries[prior].prepared);
        }
        free(plan->entries);
        free(plan);
        return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                         ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                         "f32-gradient-prepare", "cannot allocate gradient scratch");
      }
    }
    for (size_t element = 0u; element < parameter->gradient->element_count;
         element++) {
      uint32_t current_bits = 0u;
      uint32_t source_bits;
      uint32_t sum_bits;
      if (parameter->gradient_state == ET_F32_GRADIENT_PRESENT) {
        memcpy(&current_bits, &parameter->gradient->data[element],
               sizeof(current_bits));
      }
      memcpy(&source_bits, &source->data[element], sizeof(source_bits));
      if (!add_f32_bits(current_bits, source_bits, &sum_bits)) {
        for (size_t prior = 0u; prior <= index; prior++) {
          free(plan->entries[prior].prepared);
        }
        free(plan->entries);
        free(plan);
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                         ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT,
                         "f32-gradient-prepare",
                         "binary32 environment changed during preparation");
      }
      if (!bits_finite(sum_bits)) {
        for (size_t prior = 0u; prior <= index; prior++) {
          free(plan->entries[prior].prepared);
        }
        free(plan->entries);
        free(plan);
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-gradient-prepare", "gradient sum became nonfinite");
      }
      memcpy(&plan->entries[index].prepared[element], &sum_bits,
             sizeof(sum_bits));
    }
  }
  plan->magic = ET_F32_GRAD_PLAN_MAGIC;
  for (size_t index = 0u; index < count; index++) {
    plan->entries[index].parameter->plan_pins++;
    ((et_f32_tensor *)plan->entries[index].source)->plan_pins++;
  }
  plan->registry_next = live_gradient_plans;
  live_gradient_plans = plan;
  *output = plan;
  return success(error);
}

int32_t et_f32_gradient_plan_commit_v1(et_f32_gradient_plan *candidate,
                                       et_f32_tensor_error *error) {
  et_f32_gradient_plan *plan = find_gradient_plan(candidate);
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (plan == NULL || plan->magic != ET_F32_GRAD_PLAN_MAGIC ||
      plan->consumed != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-gradient-commit", "plan is foreign, stale, or consumed");
  }
  plan->consumed = 1u;
  for (size_t index = 0u; index < plan->count; index++) {
    et_f32_parameter *parameter = plan->entries[index].parameter;
    if (parameter->gradient->byte_length > 0u) {
      memcpy(parameter->gradient->data, plan->entries[index].prepared,
             parameter->gradient->byte_length);
    }
    parameter->gradient_state = ET_F32_GRADIENT_PRESENT;
    parameter->contribution_count = plan->next_count;
    parameter->normalization_weight_bits = plan->next_weight_bits;
  }
  return success(error);
}

int32_t et_f32_gradient_plan_release_v1(et_f32_gradient_plan **slot,
                                        et_f32_tensor_error *error) {
  et_f32_gradient_plan **cursor;
  et_f32_gradient_plan *plan;
  int32_t preflight = preflight_output(
      slot, slot == NULL ? 0u : sizeof(*slot), error,
      "f32-gradient-release");
  if (preflight != 0) {
    return preflight;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-gradient-release", "plan slot is null");
  }
  plan = *slot;
  if (plan == NULL) {
    return success(error);
  }
  if (find_gradient_plan(plan) == NULL || plan->magic != ET_F32_GRAD_PLAN_MAGIC) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-gradient-release", "plan is foreign or stale");
  }
  cursor = &live_gradient_plans;
  while (*cursor != plan) {
    cursor = &(*cursor)->registry_next;
  }
  *cursor = plan->registry_next;
  for (size_t index = 0u; index < plan->count; index++) {
    plan->entries[index].parameter->plan_pins--;
    ((et_f32_tensor *)plan->entries[index].source)->plan_pins--;
    free(plan->entries[index].prepared);
  }
  plan->magic = 0u;
  free(plan->entries);
  free(plan);
  *slot = NULL;
  return success(error);
}

int32_t et_f32_gradient_reset_plan_prepare_v1(
    size_t count, et_f32_parameter *const *parameters,
    et_f32_gradient_reset_plan **output, et_f32_tensor_error *error) {
  et_f32_gradient_reset_plan *plan;
  size_t bytes;
  int32_t result;
  result = preflight_output(output, output == NULL ? 0u : sizeof(*output),
                            error, "f32-gradient-reset-prepare");
  if (result != 0) {
    return result;
  }
  if (count == 0u || count > ET_F32_PARAMETER_MAX_BATCH ||
      count > SIZE_MAX / sizeof(*parameters)) {
    if (parameters != NULL && count > 0u) {
      if (count > SIZE_MAX / sizeof(*parameters) ||
          !pointer_span_fits(parameters, count * sizeof(*parameters))) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
      if (preflight_error_operand(error, parameters,
                                  count * sizeof(*parameters)) != 0) {
        return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
      }
    }
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INTEGER_OVERFLOW,
                     "f32-gradient-reset-prepare", "parameter count is invalid");
  }
  bytes = count * sizeof(*parameters);
  result = preflight_error_operand(error, parameters, bytes);
  if (result != 0) {
    return result;
  }
  if (parameters == NULL || output == NULL || *output != NULL ||
      !aligned_pointer(parameters, _Alignof(et_f32_parameter *)) ||
      !pointer_span_fits(parameters, bytes)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-gradient-reset-prepare", "parameter table is invalid");
  }
  for (size_t index = 0u; index < count; index++) {
    result = require_parameter(parameters[index],
                               "f32-gradient-reset-prepare", error);
    if (result != 0) {
      return result;
    }
    if (parameters[index]->identity == NULL ||
        !parameter_metadata_valid(parameters[index]) ||
        parameters[index]->plan_pins != 0u ||
        parameters[index]->gradient->active_borrow != NULL) {
      return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                       ET_F32_TENSOR_CODE_INVALID_HANDLE,
                       "f32-gradient-reset-prepare",
                       "parameter metadata, borrow, or pin is invalid");
    }
    for (size_t prior = 0u; prior < index; prior++) {
      if (parameters[index] == parameters[prior] ||
          parameters[index]->identity == parameters[prior]->identity) {
        return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                         ET_F32_TENSOR_CODE_INVALID_BUFFER,
                         "f32-gradient-reset-prepare", "parameter repeats");
      }
    }
  }
  plan = (et_f32_gradient_reset_plan *)f32_calloc(1u, sizeof(*plan));
  if (plan == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-gradient-reset-prepare", "cannot allocate reset plan");
  }
  plan->parameters =
      (et_f32_parameter **)f32_calloc(count, sizeof(*plan->parameters));
  if (plan->parameters == NULL) {
    free(plan);
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-gradient-reset-prepare", "cannot allocate reset table");
  }
  memcpy(plan->parameters, parameters, bytes);
  plan->count = count;
  plan->magic = ET_F32_RESET_PLAN_MAGIC;
  for (size_t index = 0u; index < count; index++) {
    plan->parameters[index]->plan_pins++;
  }
  plan->registry_next = live_reset_plans;
  live_reset_plans = plan;
  *output = plan;
  return success(error);
}

int32_t et_f32_gradient_reset_plan_commit_v1(
    et_f32_gradient_reset_plan *candidate, et_f32_tensor_error *error) {
  et_f32_gradient_reset_plan *plan = find_reset_plan(candidate);
  if (preflight_error_operand(error, NULL, 0u) != 0) {
    return ET_F32_TENSOR_ERROR_INVALID_ARGUMENT;
  }
  if (plan == NULL || plan->magic != ET_F32_RESET_PLAN_MAGIC ||
      plan->consumed != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-gradient-reset-commit",
                     "plan is foreign, stale, or consumed");
  }
  plan->consumed = 1u;
  for (size_t index = 0u; index < plan->count; index++) {
    et_f32_parameter *parameter = plan->parameters[index];
    if (parameter->gradient->byte_length > 0u) {
      memset(parameter->gradient->data, 0, parameter->gradient->byte_length);
    }
    parameter->gradient_state = ET_F32_GRADIENT_ABSENT;
    parameter->contribution_count = 0u;
    parameter->normalization_weight_bits = 0u;
  }
  return success(error);
}

int32_t et_f32_gradient_reset_plan_release_v1(
    et_f32_gradient_reset_plan **slot, et_f32_tensor_error *error) {
  et_f32_gradient_reset_plan **cursor;
  et_f32_gradient_reset_plan *plan;
  int32_t preflight = preflight_output(
      slot, slot == NULL ? 0u : sizeof(*slot), error,
      "f32-gradient-reset-release");
  if (preflight != 0) {
    return preflight;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-gradient-reset-release", "plan slot is null");
  }
  plan = *slot;
  if (plan == NULL) {
    return success(error);
  }
  if (find_reset_plan(plan) == NULL || plan->magic != ET_F32_RESET_PLAN_MAGIC) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-gradient-reset-release", "plan is foreign or stale");
  }
  cursor = &live_reset_plans;
  while (*cursor != plan) {
    cursor = &(*cursor)->registry_next;
  }
  *cursor = plan->registry_next;
  for (size_t index = 0u; index < plan->count; index++) {
    plan->parameters[index]->plan_pins--;
  }
  plan->magic = 0u;
  free(plan->parameters);
  free(plan);
  *slot = NULL;
  return success(error);
}

#ifdef ET_F32_TENSOR_TESTING
void et_f32_parameter_test_set_metadata_v1(et_f32_parameter *parameter,
                                           uint32_t state,
                                           uint64_t count,
                                           uint32_t weight_bits) {
  if (find_parameter(parameter) != NULL) {
    parameter->gradient_state = state;
    parameter->contribution_count = count;
    parameter->normalization_weight_bits = weight_bits;
  }
}

void et_f32_parameter_test_set_gradient_bits_v1(
    et_f32_parameter *parameter, const uint32_t *bits, size_t count) {
  if (find_parameter(parameter) != NULL && bits != NULL &&
      count == parameter->gradient->element_count) {
    memcpy(parameter->gradient->data, bits,
           count * sizeof(*bits));
  }
}

void et_f32_test_live_counts_snapshot_v1(et_f32_test_live_counts_v1 *counts) {
  et_f32_test_live_counts_v1 snapshot = {
      .struct_size = sizeof(snapshot),
  };
  if (counts == NULL || counts->struct_size != sizeof(*counts)) {
    return;
  }
  for (const et_f32_tensor *item = live_tensors; item != NULL;
       item = item->registry_next) {
    snapshot.tensors++;
  }
  for (const et_f32_parameter *item = live_parameters; item != NULL;
       item = item->registry_next) {
    snapshot.parameters++;
  }
  for (const et_f32_tensor_borrow *item = live_borrows; item != NULL;
       item = item->registry_next) {
    snapshot.borrows++;
  }
  for (const et_f32_tensor_copy_plan *item = live_copy_plans; item != NULL;
       item = item->registry_next) {
    snapshot.copy_plans++;
  }
  for (const et_f32_gradient_plan *item = live_gradient_plans; item != NULL;
       item = item->registry_next) {
    snapshot.gradient_plans++;
  }
  for (const et_f32_gradient_reset_plan *item = live_reset_plans; item != NULL;
       item = item->registry_next) {
    snapshot.reset_plans++;
  }
  *counts = snapshot;
}
#endif

int32_t et_f32_tensor_borrow_begin_v1(et_f32_tensor *tensor,
                                      et_f32_tensor_borrow **output,
                                      et_f32_tensor_error *error) {
  et_f32_tensor_borrow *borrow;
  int32_t result = preflight_output(
      output, output == NULL ? 0u : sizeof(*output), error,
      "f32-tensor-borrow-begin");
  if (result != 0) {
    return result;
  }
  if (output == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-tensor-borrow-begin", "borrow output is null");
  }
  if (*output != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-tensor-borrow-begin",
                     "borrow output must initially be null");
  }
  result = require_tensor(tensor, "f32-tensor-borrow-begin", error);
  if (result != 0) {
    return result;
  }
  if (tensor->active_borrow != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_ACTIVE_BORROW,
                     "f32-tensor-borrow-begin", "tensor is already borrowed");
  }
  if (tensor->plan_pins != 0u) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-tensor-borrow-begin",
                     "tensor is retained by a prepared plan");
  }
  borrow = (et_f32_tensor_borrow *)f32_calloc(1u, sizeof(*borrow));
  if (borrow == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INTERNAL,
                     ET_F32_TENSOR_CODE_ALLOCATION_FAILED,
                     "f32-tensor-borrow-begin", "cannot allocate borrow lease");
  }
  borrow->owner = tensor;
  borrow->view.struct_size = sizeof(borrow->view);
  borrow->view.data = tensor->data;
  borrow->view.byte_length = tensor->byte_length;
  borrow->view.dtype = "f32";
  borrow->view.device = "cpu";
  borrow->view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  borrow->view.offset_bytes = 0u;
  borrow->view.rank = tensor->rank;
  borrow->view.shape = tensor->shape;
  borrow->magic = ET_F32_BORROW_MAGIC;
  (void)success(error);
  register_borrow(borrow);
  tensor->active_borrow = borrow;
  *output = borrow;
  return 0;
}

int32_t et_f32_tensor_borrow_view_v1(
    const et_f32_tensor_borrow *candidate,
    const et_kernel_tensor_view_v1 **view, et_f32_tensor_error *error) {
  int32_t result = preflight_output(
      (void *)view, view == NULL ? 0u : sizeof(*view), error,
      "f32-tensor-borrow-view");
  if (result != 0) {
    return result;
  }
  if (view == NULL || candidate == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-tensor-borrow-view", "borrow and output are required");
  }
  if (*view != NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_INVALID_BUFFER,
                     "f32-tensor-borrow-view",
                     "view output must initially be null");
  }
  if (!valid_borrow(candidate)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-tensor-borrow-view", "borrow is foreign or stale");
  }
  (void)success(error);
  *view = &candidate->view;
  return 0;
}

int32_t et_f32_tensor_borrow_end_v1(et_f32_tensor_borrow **slot,
                                    et_f32_tensor_error *error) {
  et_f32_tensor_borrow *borrow;
  int32_t result = preflight_output(
      slot, slot == NULL ? 0u : sizeof(*slot), error,
      "f32-tensor-borrow-end");
  if (result != 0) {
    return result;
  }
  if (slot == NULL) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                     ET_F32_TENSOR_CODE_NULL_ARGUMENT,
                     "f32-tensor-borrow-end", "borrow slot is null");
  }
  borrow = *slot;
  if (borrow == NULL) {
    return success(error);
  }
  if (!valid_borrow(borrow)) {
    return set_error(error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                     ET_F32_TENSOR_CODE_INVALID_HANDLE,
                     "f32-tensor-borrow-end", "borrow is foreign or stale");
  }
  (void)success(error);
  unregister_borrow(borrow);
  borrow->owner->active_borrow = NULL;
  borrow->magic = 0u;
  free(borrow);
  *slot = NULL;
  return 0;
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
  if (rank != 1u || shape == NULL ||
      !aligned_pointer(shape, _Alignof(uint64_t)) ||
      !pointer_span_fits(shape, sizeof(*shape))) {
    return 0;
  }
  return shape[0] <= ET_F32_RANK1_MAX;
}

static int32_t provider_view(const et_kernel_tensor_view_v1 *view,
                             const et_kernel_request_v1 *request,
                             et_kernel_error *error) {
  size_t expected = sizeof(float);
  if (!exact_text(view->dtype, "f32")) {
    return set_kernel_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_TEXT,
                            "storage.copy requires f32 views");
  }
  if (!exact_text(view->device, "cpu")) {
    return set_kernel_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy requires CPU views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_kernel_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy requires dense zero-offset views");
  }
  if (view->rank != request->rank ||
      !provider_shape_supported(view->rank, view->shape)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                            ET_KERNEL_CODE_INVALID_SHAPE,
                            "storage.copy shape is unsupported");
  }
  if (view->rank > 0u) {
    size_t count = 1u;
    int zero = 0;
    for (size_t index = 0u; index < view->rank; index++) {
      if (view->shape[index] != request->shape[index]) {
        return set_kernel_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                                ET_KERNEL_CODE_INVALID_SHAPE,
                                "storage.copy shape differs from request");
      }
      zero = zero || view->shape[index] == 0u;
    }
    if (zero) {
      expected = 0u;
    } else {
      for (size_t index = 0u; index < view->rank; index++) {
        count *= (size_t)view->shape[index];
      }
      expected = count * sizeof(float);
    }
  }
  if (view->byte_length != expected ||
      (expected > 0u &&
       (!aligned_pointer(view->data, _Alignof(float)) ||
        !pointer_span_fits(view->data, expected)))) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy data span is invalid");
  }
  return 0;
}

static int32_t provider_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  const et_kernel_tensor_view_v1 *input;
  const et_kernel_tensor_view_v1 *output;
  int32_t result;
  if (call == NULL || call->request == NULL || call->inputs == NULL ||
      call->outputs == NULL) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_NULL_ARGUMENT,
                            "storage.copy call is incomplete");
  }
  if (!exact_text(call->capability, "tensor.f32") ||
      !exact_text(call->request->operation, "storage.copy") ||
      !exact_text(call->request->dtype, "f32") ||
      !exact_text(call->request->device, "cpu") ||
      !provider_shape_supported(call->request->rank,
                                call->request->shape)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                            ET_KERNEL_CODE_PROVIDER_REJECTED,
                            "storage.copy request is outside I2 evidence");
  }
  if (call->input_count != 1u || call->output_count != 1u ||
      call->input_stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      call->output_stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      call->input_bytes != call->input_stride ||
      call->output_bytes != call->output_stride) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_INVALID_BUFFER,
                            "storage.copy requires one input and one output");
  }
  input = (const et_kernel_tensor_view_v1 *)call->inputs;
  output = (const et_kernel_tensor_view_v1 *)call->outputs;
  result = provider_view(input, call->request, error);
  if (result != 0) {
    return result;
  }
  result = provider_view(output, call->request, error);
  if (result != 0) {
    return result;
  }
  if (ranges_overlap(input->data, input->byte_length, output->data,
                     output->byte_length)) {
    return set_kernel_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                            ET_KERNEL_CODE_ALIASING_OUTPUT,
                            "storage.copy input and output overlap");
  }
  et_kernel_error_clear(error);
  return 0;
}

static void provider_invoke(const et_kernel_call_v1 *call) {
  const et_kernel_tensor_view_v1 *input =
      (const et_kernel_tensor_view_v1 *)call->inputs;
  et_kernel_tensor_view_v1 *output =
      (et_kernel_tensor_view_v1 *)call->outputs;
  if (output->byte_length > 0u) {
    memcpy(output->data, input->data, output->byte_length);
  }
}

static const et_kernel_dimension_range_v1 provider_rank1_dimensions[] = {
    {.minimum = 0u,
     .maximum = ET_F32_RANK1_MAX,
     .maximum_unbounded = 0u,
     .reserved = {0}},
};
static const et_kernel_shape_range_v1 provider_ranges[] = {
    {.rank = 0u, .dimensions = NULL},
    {.rank = 1u, .dimensions = provider_rank1_dimensions},
};
static const char *const provider_operations[] = {"storage.copy"};
static const char *const provider_dtypes[] = {"f32"};
static const char *const provider_devices[] = {"cpu"};
static const et_kernel_capability_v1 provider_capability = {
    .struct_size = sizeof(et_kernel_capability_v1),
    .name = "tensor.f32",
    .status = ET_KERNEL_CAPABILITY_VERIFIED,
    .implementation = "eshkol-transformer-f32",
    .version = "1.0",
    .evidence = "I2:bounded-exact-f32-storage.copy-v1",
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
    .name = "eshkol-transformer-f32",
    .version = "1.0",
    .evidence = "I2:bounded-exact-f32-storage.copy-v1",
    .capability_count = 1u,
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(et_kernel_capability_v1),
    .capabilities = &provider_capability,
    .validate_call = provider_call,
    .invoke_call = provider_invoke,
};

const et_kernel_provider_v1 *et_f32_tensor_provider_v1(void) {
  return &provider;
}

#ifdef ET_F32_TENSOR_TESTING
const uint64_t *et_f32_tensor_test_shape_storage_v1(
    const et_f32_tensor *tensor) {
  return valid_tensor(tensor) ? tensor->shape : NULL;
}

const size_t *et_f32_tensor_test_stride_storage_v1(
    const et_f32_tensor *tensor) {
  return valid_tensor(tensor) ? tensor->strides : NULL;
}

const float *et_f32_tensor_test_data_storage_v1(
    const et_f32_tensor *tensor) {
  return valid_tensor(tensor) ? tensor->data : NULL;
}

size_t et_f32_tensor_test_control_bytes_v1(void) {
  return sizeof(et_f32_tensor);
}

size_t et_f32_tensor_test_borrow_bytes_v1(void) {
  return sizeof(et_f32_tensor_borrow);
}
#endif
