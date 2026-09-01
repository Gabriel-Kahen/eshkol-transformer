#include "eshkol_transformer/indexed_cross_entropy.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct l2_eshkol_context {
  et_kernel_runtime *runtime;
  et_kernel_error error;
  float losses[2];
  float gradients[6];
} l2_eshkol_context;

static const et_kernel_provider_v1 *l2_resolver(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_l2_indexed_cross_entropy_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 l2_view(void *data, size_t bytes,
                                        const char *dtype, size_t rank,
                                        const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = rank,
      .shape = shape,
  };
  return result;
}

void *et_l2_eshkol_context_create_v1(void) {
  l2_eshkol_context *context =
      (l2_eshkol_context *)calloc(1u, sizeof(*context));
  if (context == NULL ||
      et_kernel_runtime_discover(l2_resolver, NULL, &context->runtime,
                                 &context->error) != 0) {
    free(context);
    return NULL;
  }
  return context;
}

int64_t et_l2_eshkol_context_run_v1(void *opaque) {
  static const uint64_t logits_shape[] = {1u, 2u, 3u};
  static const uint64_t leading_shape[] = {1u, 2u};
  float logits[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  int64_t targets[] = {2, 0};
  float upstream[] = {0.7f, -1.25f};
  l2_eshkol_context *context = (l2_eshkol_context *)opaque;
  et_kernel_tensor_view_v1 inputs[3];
  et_kernel_tensor_view_v1 output;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  if (context == NULL || context->runtime == NULL) {
    return -1;
  }
  inputs[0] = l2_view(logits, sizeof(logits), "f32", 3u, logits_shape);
  inputs[1] = l2_view(targets, sizeof(targets), "i64", 2u, leading_shape);
  inputs[2] = l2_view(upstream, sizeof(upstream), "f32", 2u, leading_shape);
  output = l2_view(context->losses, sizeof(context->losses), "f32", 2u,
                   leading_shape);
  memset(&request, 0, sizeof(request));
  request.struct_size = sizeof(request);
  request.operation = ET_L2_INDEXED_CROSS_ENTROPY_FORWARD;
  request.dtype = "f32";
  request.device = "cpu";
  request.rank = 3u;
  request.shape = logits_shape;
  request.deterministic = 1u;
  memset(&call, 0, sizeof(call));
  call.struct_size = sizeof(call);
  call.capability = ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY;
  call.request = &request;
  call.input_count = 2u;
  call.input_stride = sizeof(inputs[0]);
  call.input_bytes = 2u * sizeof(inputs[0]);
  call.inputs = inputs;
  call.output_count = 1u;
  call.output_stride = sizeof(output);
  call.output_bytes = sizeof(output);
  call.outputs = &output;
  if (et_kernel_runtime_dispatch(context->runtime, &call, &context->error) != 0) {
    return -2;
  }
  request.operation = ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD;
  output = l2_view(context->gradients, sizeof(context->gradients), "f32", 3u,
                   logits_shape);
  call.input_count = 3u;
  call.input_bytes = sizeof(inputs);
  call.output_bytes = sizeof(output);
  call.outputs = &output;
  return et_kernel_runtime_dispatch(context->runtime, &call, &context->error);
}

double et_l2_eshkol_context_loss_v1(void *opaque, int64_t index) {
  const l2_eshkol_context *context = (const l2_eshkol_context *)opaque;
  return context == NULL || index < 0 || index >= 2 ? 0.0
                                                    : context->losses[index];
}

double et_l2_eshkol_context_gradient_v1(void *opaque, int64_t index) {
  const l2_eshkol_context *context = (const l2_eshkol_context *)opaque;
  return context == NULL || index < 0 || index >= 6
             ? 0.0
             : context->gradients[index];
}

int64_t et_l2_eshkol_context_invalid_atomic_v1(void *opaque) {
  static const uint64_t logits_shape[] = {1u, 1u, 2u};
  static const uint64_t leading_shape[] = {1u, 1u};
  float logits[] = {1.0f, 2.0f};
  int64_t targets[] = {-1};
  float output_bits;
  uint32_t before = UINT32_C(0x5a5aa5a5);
  l2_eshkol_context *context = (l2_eshkol_context *)opaque;
  et_kernel_tensor_view_v1 inputs[2];
  et_kernel_tensor_view_v1 output;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  int32_t status;
  if (context == NULL || context->runtime == NULL) {
    return 0;
  }
  memcpy(&output_bits, &before, sizeof(before));
  inputs[0] = l2_view(logits, sizeof(logits), "f32", 3u, logits_shape);
  inputs[1] = l2_view(targets, sizeof(targets), "i64", 2u, leading_shape);
  output = l2_view(&output_bits, sizeof(output_bits), "f32", 2u,
                   leading_shape);
  memset(&request, 0, sizeof(request));
  request.struct_size = sizeof(request);
  request.operation = ET_L2_INDEXED_CROSS_ENTROPY_FORWARD;
  request.dtype = "f32";
  request.device = "cpu";
  request.rank = 3u;
  request.shape = logits_shape;
  request.deterministic = 1u;
  memset(&call, 0, sizeof(call));
  call.struct_size = sizeof(call);
  call.capability = ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY;
  call.request = &request;
  call.input_count = 2u;
  call.input_stride = sizeof(inputs[0]);
  call.input_bytes = sizeof(inputs);
  call.inputs = inputs;
  call.output_count = 1u;
  call.output_stride = sizeof(output);
  call.output_bytes = sizeof(output);
  call.outputs = &output;
  status = et_kernel_runtime_dispatch(context->runtime, &call, &context->error);
  return status == ET_KERNEL_ERROR_SHAPE_MISMATCH &&
                 memcmp(&output_bits, &before, sizeof(before)) == 0
             ? 1
             : 0;
}

int64_t et_l2_eshkol_context_destroy_v1(void *opaque) {
  l2_eshkol_context *context = (l2_eshkol_context *)opaque;
  if (context == NULL) {
    return -1;
  }
  et_kernel_runtime_destroy(context->runtime);
  free(context);
  return 0;
}
