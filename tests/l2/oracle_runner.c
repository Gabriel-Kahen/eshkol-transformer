#include "eshkol_transformer/indexed_cross_entropy.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_l2(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_l2_indexed_cross_entropy_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 tensor(void *data, size_t bytes,
                                       const char *dtype, size_t rank,
                                       const uint64_t *shape) {
  et_kernel_tensor_view_v1 value = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = rank,
      .shape = shape,
  };
  return value;
}

static void print_bits(const char *name, const float *values, size_t count) {
  printf("%s", name);
  for (size_t index = 0u; index < count; index++) {
    uint32_t bits;
    memcpy(&bits, &values[index], sizeof(bits));
    printf(" %08" PRIx32, bits);
  }
  putchar('\n');
}

int main(void) {
  static const uint64_t logits_shape[] = {1u, 2u, 3u};
  static const uint64_t leading_shape[] = {1u, 2u};
  float logits[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  int64_t targets[] = {2, 0};
  float upstream[] = {0.7f, -1.25f};
  float losses[2];
  float gradients[6];
  et_kernel_tensor_view_v1 inputs[3];
  et_kernel_tensor_view_v1 output;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  inputs[0] = tensor(logits, sizeof(logits), "f32", 3u, logits_shape);
  inputs[1] = tensor(targets, sizeof(targets), "i64", 2u, leading_shape);
  inputs[2] = tensor(upstream, sizeof(upstream), "f32", 2u, leading_shape);
  output = tensor(losses, sizeof(losses), "f32", 2u, leading_shape);
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
  if (et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) != 0 ||
      et_kernel_runtime_dispatch(runtime, &call, &error) != 0) {
    fprintf(stderr, "L2 oracle forward failed: %s\n", error.message);
    et_kernel_runtime_destroy(runtime);
    return EXIT_FAILURE;
  }
  request.operation = ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD;
  output = tensor(gradients, sizeof(gradients), "f32", 3u, logits_shape);
  call.input_count = 3u;
  call.input_bytes = sizeof(inputs);
  call.output_bytes = sizeof(output);
  call.outputs = &output;
  if (et_kernel_runtime_dispatch(runtime, &call, &error) != 0) {
    fprintf(stderr, "L2 oracle backward failed: %s\n", error.message);
    et_kernel_runtime_destroy(runtime);
    return EXIT_FAILURE;
  }
  print_bits("loss", losses, 2u);
  print_bits("gradient", gradients, 6u);
  et_kernel_runtime_destroy(runtime);
  return EXIT_SUCCESS;
}
