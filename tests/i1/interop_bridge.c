#include "eshkol_transformer/i64_tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define I1_ESHKOL_CONTEXT_MAGIC UINT64_C(0x4931455348435458)

typedef struct {
  uint64_t magic;
  et_i64_tensor *tensor;
  et_i64_tensor_borrow *borrow;
  const et_kernel_tensor_view_v1 *view;
  et_i64_tensor_error error;
  size_t rank;
  uint64_t shape[ET_KERNEL_MAX_RANK];
  int64_t *input;
  int64_t *output;
  size_t element_count;
  int64_t scalar;
  uint64_t call_count;
} i1_eshkol_context;

static int context_valid(const i1_eshkol_context *context) {
  return context != NULL && context->magic == I1_ESHKOL_CONTEXT_MAGIC;
}

static int64_t bridge_error(i1_eshkol_context *context,
                            et_i64_tensor_error_category category,
                            et_i64_tensor_error_code code,
                            const char *operation, const char *message) {
  et_i64_tensor_error_clear_v1(&context->error);
  context->error.category = category;
  context->error.code = code;
  (void)snprintf(context->error.operation, sizeof(context->error.operation),
                 "%s", operation);
  (void)snprintf(context->error.message, sizeof(context->error.message), "%s",
                 message);
  return (int64_t)category;
}

static int64_t require_context(i1_eshkol_context *context,
                               const char *operation) {
  if (!context_valid(context)) {
    return ET_I64_TENSOR_ERROR_INVALID_STATE;
  }
  context->call_count++;
  if (context->tensor == NULL &&
      strcmp(operation, "interop-prepare") != 0 &&
      strcmp(operation, "interop-shape-set") != 0 &&
      strcmp(operation, "interop-commit") != 0 &&
      strcmp(operation, "interop-destroy") != 0) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_STATE,
                        ET_I64_TENSOR_CODE_INVALID_HANDLE, operation,
                        "tensor handle is not live");
  }
  return 0;
}

void *et_i64_tensor_eshkol_context_create_v1(void) {
  i1_eshkol_context *context =
      (i1_eshkol_context *)calloc(1u, sizeof(*context));
  if (context != NULL) {
    context->magic = I1_ESHKOL_CONTEXT_MAGIC;
    context->call_count = 1u;
  }
  return context;
}

int64_t et_i64_tensor_eshkol_context_release_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  if (!context_valid(context)) {
    return ET_I64_TENSOR_ERROR_INVALID_STATE;
  }
  if (context->borrow != NULL) {
    (void)et_i64_tensor_borrow_end_v1(&context->borrow, &context->error);
  }
  if (context->tensor != NULL) {
    (void)et_i64_tensor_destroy_v1(&context->tensor, &context->error);
  }
  free(context->output);
  free(context->input);
  context->magic = 0u;
  free(context);
  return 0;
}

int64_t et_i64_tensor_eshkol_context_prepare_v1(void *opaque, int64_t rank) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-prepare");
  if (result != 0) {
    return result;
  }
  if (context->tensor != NULL || rank < 0 || rank > ET_KERNEL_MAX_RANK) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                        ET_I64_TENSOR_CODE_INVALID_SHAPE, "interop-prepare",
                        "rank is invalid or a tensor is already live");
  }
  context->rank = (size_t)rank;
  memset(context->shape, 0, sizeof(context->shape));
  return 0;
}

int64_t et_i64_tensor_eshkol_context_shape_set_v1(void *opaque, int64_t index,
                                                   int64_t extent) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-shape-set");
  if (result != 0) {
    return result;
  }
  if (context->tensor != NULL || index < 0 || (size_t)index >= context->rank ||
      extent < 0) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                        ET_I64_TENSOR_CODE_INVALID_SHAPE,
                        "interop-shape-set", "shape index or extent is invalid");
  }
  context->shape[(size_t)index] = (uint64_t)extent;
  return 0;
}

int64_t et_i64_tensor_eshkol_context_commit_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  size_t count = 0u;
  int32_t status;
  int64_t result = require_context(context, "interop-commit");
  if (result != 0) {
    return result;
  }
  if (context->tensor != NULL) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_STATE,
                        ET_I64_TENSOR_CODE_INVALID_HANDLE, "interop-commit",
                        "a tensor is already live");
  }
  status = et_i64_tensor_create_v1(
      context->rank, context->rank == 0u ? NULL : context->shape,
      &context->tensor, &context->error);
  if (status != 0) {
    return status;
  }
  status = et_i64_tensor_element_count_v1(context->tensor, &count,
                                          &context->error);
  if (status != 0) {
    (void)et_i64_tensor_destroy_v1(&context->tensor, &context->error);
    return status;
  }
  context->input = (int64_t *)calloc(count == 0u ? 1u : count, sizeof(int64_t));
  context->output =
      (int64_t *)calloc(count == 0u ? 1u : count, sizeof(int64_t));
  if (context->input == NULL || context->output == NULL) {
    free(context->output);
    free(context->input);
    context->output = NULL;
    context->input = NULL;
    (void)et_i64_tensor_destroy_v1(&context->tensor, &context->error);
    return bridge_error(context, ET_I64_TENSOR_ERROR_INTERNAL,
                        ET_I64_TENSOR_CODE_ALLOCATION_FAILED, "interop-commit",
                        "cannot allocate aligned caller-owned copy buffers");
  }
  context->element_count = count;
  return 0;
}

int64_t et_i64_tensor_eshkol_context_destroy_tensor_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-destroy");
  if (result != 0) {
    return result;
  }
  return et_i64_tensor_destroy_v1(&context->tensor, &context->error);
}

int64_t et_i64_tensor_eshkol_context_rank_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  size_t value = 0u;
  int64_t result = require_context(context, "interop-rank");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_rank_v1(context->tensor, &value, &context->error);
  if (result == 0) {
    context->scalar = (int64_t)value;
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_shape_at_v1(void *opaque,
                                                  int64_t dimension) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  uint64_t value = 0u;
  int64_t result = require_context(context, "interop-shape-at");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_shape_at_v1(context->tensor, (size_t)dimension, &value,
                                     &context->error);
  if (result == 0) {
    context->scalar = (int64_t)value;
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_stride_at_v1(void *opaque,
                                                   int64_t dimension) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  size_t value = 0u;
  int64_t result = require_context(context, "interop-stride-at");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_stride_bytes_at_v1(
      context->tensor, (size_t)dimension, &value, &context->error);
  if (result == 0) {
    context->scalar = (int64_t)value;
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_element_count_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  size_t value = 0u;
  int64_t result = require_context(context, "interop-element-count");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_element_count_v1(context->tensor, &value,
                                          &context->error);
  if (result == 0) {
    context->scalar = (int64_t)value;
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_byte_length_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  size_t value = 0u;
  int64_t result = require_context(context, "interop-byte-length");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_byte_length_v1(context->tensor, &value,
                                        &context->error);
  if (result == 0) {
    context->scalar = (int64_t)value;
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_scalar_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? context->scalar : 0;
}

int64_t et_i64_tensor_eshkol_context_copy_from4_v1(
    void *opaque, int64_t a, int64_t b, int64_t c, int64_t d) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-copy-from4");
  if (result != 0) {
    return result;
  }
  if (context->element_count != 4u) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                        ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH,
                        "interop-copy-from4",
                        "fixed interop evidence requires exactly four elements");
  }
  context->input[0] = a;
  context->input[1] = b;
  context->input[2] = c;
  context->input[3] = d;
  return et_i64_tensor_copy_from_v1(context->tensor, context->input, 4u,
                                    &context->error);
}

int64_t et_i64_tensor_eshkol_context_copy_to_v1(void *opaque, int64_t count) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-copy-to");
  if (result != 0) {
    return result;
  }
  return et_i64_tensor_copy_to_v1(context->tensor, context->output,
                                  (size_t)count, &context->error);
}

int64_t et_i64_tensor_eshkol_context_seed_output4_v1(
    void *opaque, int64_t a, int64_t b, int64_t c, int64_t d) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-seed-output4");
  if (result != 0) {
    return result;
  }
  if (context->element_count != 4u) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                        ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH,
                        "interop-seed-output4",
                        "fixed interop evidence requires exactly four elements");
  }
  context->output[0] = a;
  context->output[1] = b;
  context->output[2] = c;
  context->output[3] = d;
  return 0;
}

int64_t et_i64_tensor_eshkol_context_output_at_v1(void *opaque,
                                                   int64_t index) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-output-at");
  if (result != 0) {
    return result;
  }
  if (index < 0 || (size_t)index >= context->element_count) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                        ET_I64_TENSOR_CODE_INVALID_SHAPE, "interop-output-at",
                        "caller output index is out of range");
  }
  context->scalar = context->output[(size_t)index];
  return 0;
}

int64_t et_i64_tensor_eshkol_context_borrow_begin_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-borrow-begin");
  if (result != 0) {
    return result;
  }
  if (context->borrow != NULL) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_STATE,
                        ET_I64_TENSOR_CODE_ACTIVE_BORROW,
                        "interop-borrow-begin", "bridge borrow is already active");
  }
  result = et_i64_tensor_borrow_begin_v1(context->tensor, &context->borrow,
                                         &context->error);
  if (result == 0) {
    result = et_i64_tensor_borrow_view_v1(context->borrow, &context->view,
                                          &context->error);
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_borrow_begin_again_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  et_i64_tensor_borrow *second = NULL;
  int64_t result = require_context(context, "interop-borrow-begin-again");
  if (result != 0) {
    return result;
  }
  if (context->borrow == NULL || context->view == NULL) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_STATE,
                        ET_I64_TENSOR_CODE_INVALID_HANDLE,
                        "interop-borrow-begin-again",
                        "the first borrow is not active");
  }
  result = et_i64_tensor_borrow_begin_v1(context->tensor, &second,
                                         &context->error);
  if (result == 0) {
    (void)et_i64_tensor_borrow_end_v1(&second, &context->error);
    return bridge_error(context, ET_I64_TENSOR_ERROR_INTERNAL,
                        ET_I64_TENSOR_CODE_PROVIDER_REJECTED,
                        "interop-borrow-begin-again",
                        "native tensor unexpectedly allowed a second borrow");
  }
  return result;
}

int64_t et_i64_tensor_eshkol_context_borrow_write_first_v1(void *opaque,
                                                            int64_t value) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-borrow-write-first");
  if (result != 0) {
    return result;
  }
  if (context->view == NULL || context->view->data == NULL ||
      context->view->byte_length < sizeof(value)) {
    return bridge_error(context, ET_I64_TENSOR_ERROR_INVALID_STATE,
                        ET_I64_TENSOR_CODE_INVALID_HANDLE,
                        "interop-borrow-write-first",
                        "a nonempty live borrowed view is required");
  }
  ((int64_t *)context->view->data)[0] = value;
  return 0;
}

int64_t et_i64_tensor_eshkol_context_borrow_end_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  int64_t result = require_context(context, "interop-borrow-end");
  if (result != 0) {
    return result;
  }
  result = et_i64_tensor_borrow_end_v1(&context->borrow, &context->error);
  if (result == 0) {
    context->view = NULL;
  }
  return result;
}

const char *et_i64_tensor_eshkol_context_borrow_dtype_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL ? context->view->dtype
                                                         : "";
}

const char *et_i64_tensor_eshkol_context_borrow_device_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL ? context->view->device
                                                         : "";
}

int64_t et_i64_tensor_eshkol_context_borrow_layout_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL
             ? (int64_t)context->view->layout
             : -1;
}

int64_t et_i64_tensor_eshkol_context_borrow_offset_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL
             ? (int64_t)context->view->offset_bytes
             : -1;
}

int64_t et_i64_tensor_eshkol_context_borrow_rank_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL
             ? (int64_t)context->view->rank
             : -1;
}

int64_t et_i64_tensor_eshkol_context_borrow_byte_length_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) && context->view != NULL
             ? (int64_t)context->view->byte_length
             : -1;
}

int64_t et_i64_tensor_eshkol_context_error_category_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? (int64_t)context->error.category : 6;
}

int64_t et_i64_tensor_eshkol_context_error_code_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? (int64_t)context->error.code : 0;
}

const char *et_i64_tensor_eshkol_context_error_operation_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? context->error.operation : "interop-invalid";
}

const char *et_i64_tensor_eshkol_context_error_message_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? context->error.message
                                : "interop context is invalid";
}

int64_t et_i64_tensor_eshkol_context_call_count_v1(void *opaque) {
  i1_eshkol_context *context = (i1_eshkol_context *)opaque;
  return context_valid(context) ? (int64_t)context->call_count : -1;
}

static const et_kernel_capability_v1 *capability(void) {
  const et_kernel_provider_v1 *provider = et_i64_tensor_provider_v1();
  return provider != NULL && provider->capability_count == 1u
             ? (const et_kernel_capability_v1 *)provider->capabilities
             : NULL;
}

int64_t et_i64_tensor_eshkol_abi_major_v1(void) {
  return et_i64_tensor_abi_major_v1();
}

int64_t et_i64_tensor_eshkol_abi_minor_v1(void) {
  return et_i64_tensor_abi_minor_v1();
}

const char *et_i64_tensor_eshkol_capability_name_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? value->name : "";
}

int64_t et_i64_tensor_eshkol_capability_status_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? (int64_t)value->status : -1;
}

const char *et_i64_tensor_eshkol_capability_operation_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->operation_count == 1u ? value->operations[0]
                                                       : "";
}

const char *et_i64_tensor_eshkol_capability_dtype_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->dtype_count == 1u ? value->dtypes[0] : "";
}

const char *et_i64_tensor_eshkol_capability_device_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->device_count == 1u ? value->devices[0] : "";
}

int64_t et_i64_tensor_eshkol_capability_deterministic_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? (int64_t)value->deterministic : -1;
}

const char *et_i64_tensor_eshkol_capability_version_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? value->version : "";
}

const char *et_i64_tensor_eshkol_capability_evidence_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? value->evidence : "";
}

int64_t et_i64_tensor_eshkol_capability_shape_count_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL ? (int64_t)value->shape_range_count : -1;
}

int64_t et_i64_tensor_eshkol_capability_shape0_rank_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->shape_range_count >= 1u
             ? (int64_t)value->shape_ranges[0].rank
             : -1;
}

int64_t et_i64_tensor_eshkol_capability_shape1_rank_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->shape_range_count >= 2u
             ? (int64_t)value->shape_ranges[1].rank
             : -1;
}

int64_t et_i64_tensor_eshkol_capability_shape1_min_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->shape_range_count >= 2u &&
                 value->shape_ranges[1].rank == 1u &&
                 value->shape_ranges[1].dimensions != NULL
             ? (int64_t)value->shape_ranges[1].dimensions[0].minimum
             : -1;
}

int64_t et_i64_tensor_eshkol_capability_shape1_max_v1(void) {
  const et_kernel_capability_v1 *value = capability();
  return value != NULL && value->shape_range_count >= 2u &&
                 value->shape_ranges[1].rank == 1u &&
                 value->shape_ranges[1].dimensions != NULL
             ? (int64_t)value->shape_ranges[1].dimensions[0].maximum
             : -1;
}
