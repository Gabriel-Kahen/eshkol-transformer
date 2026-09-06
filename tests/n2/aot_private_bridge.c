#include "eshkol_transformer/n2_primitives_abi.h"

#include <stdint.h>
#include <string.h>

/* Test-only fixed-buffer transport.  This is not a tensor carrier API. */

static et_kernel_tensor_view_v1 make_view(void *data, size_t bytes,
                                          const char *dtype, size_t rank,
                                          const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = rank,
      .shape = shape};
  return result;
}

static et_kernel_request_v1 make_request(const char *operation, size_t rank,
                                         const uint64_t *shape) {
  et_kernel_request_v1 result = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = operation,
      .dtype = "f32",
      .device = "cpu",
      .rank = rank,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0}};
  return result;
}

static et_kernel_call_v1 make_call(
    const char *capability, const et_kernel_request_v1 *request,
    et_kernel_tensor_view_v1 *inputs, size_t input_count,
    et_kernel_tensor_view_v1 *outputs, size_t output_count) {
  et_kernel_call_v1 result = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = capability,
      .request = request,
      .input_count = input_count,
      .input_stride = sizeof(et_kernel_tensor_view_v1),
      .input_bytes = input_count * sizeof(et_kernel_tensor_view_v1),
      .inputs = inputs,
      .output_count = output_count,
      .output_stride = sizeof(et_kernel_tensor_view_v1),
      .output_bytes = output_count * sizeof(et_kernel_tensor_view_v1),
      .outputs = outputs};
  return result;
}

static const et_kernel_provider_v1 *resolve_n2(void *context,
                                               const char *symbol) {
  (void)context;
  if (symbol == NULL || strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0)
    return NULL;
  return et_n2_kernel_provider_v1();
}

static int dispatch(et_kernel_runtime *runtime, const char *capability,
                    const char *operation, size_t rank, const uint64_t *shape,
                    et_kernel_tensor_view_v1 *inputs, size_t input_count,
                    et_kernel_tensor_view_v1 *outputs, size_t output_count) {
  et_kernel_error error;
  et_kernel_request_v1 request = make_request(operation, rank, shape);
  et_kernel_call_v1 call = make_call(capability, &request, inputs, input_count,
                                     outputs, output_count);
  return et_kernel_runtime_dispatch(runtime, &call, &error) == 0;
}

int64_t et_n2_test_provider_transport_v1(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  int success = 0;
#define REQUIRE(condition)                                                      \
  do {                                                                          \
    if (!(condition)) goto done;                                                 \
  } while (0)

  REQUIRE(et_n2_kernel_provider_v1() != NULL);
  REQUIRE(et_kernel_runtime_discover(resolve_n2, NULL, &runtime, &error) == 0);
  REQUIRE(runtime != NULL);

  {
    uint64_t request_shape[] = {1u, 1u, 1u, 1u};
    uint64_t ids_shape[] = {1u, 1u};
    uint64_t weight_shape[] = {1u, 1u};
    uint64_t value_shape[] = {1u, 1u, 1u};
    int64_t ids[] = {0};
    float weight[] = {2.0f};
    float upstream[] = {3.0f};
    float output[] = {0.0f};
    float dweight[] = {0.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(ids, sizeof(ids), "i64", 2u, ids_shape),
        make_view(weight, sizeof(weight), "f32", 2u, weight_shape)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, value_shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(ids, sizeof(ids), "i64", 2u, ids_shape),
        make_view(upstream, sizeof(upstream), "f32", 3u, value_shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dweight, sizeof(dweight), "f32", 2u, weight_shape)};
    REQUIRE(dispatch(runtime, "kernel.embedding-forward", "embedding.forward",
                     4u, request_shape, forward_inputs, 2u, forward_outputs,
                     1u));
    REQUIRE(output[0] == 2.0f);
    REQUIRE(dispatch(runtime, "kernel.embedding-backward",
                     "embedding.backward", 4u, request_shape, backward_inputs,
                     2u, backward_outputs, 1u));
    REQUIRE(dweight[0] == 3.0f);
  }

  {
    uint64_t request_shape[] = {1u, 1u, 1u, 1u};
    uint64_t value_shape[] = {1u, 1u, 1u};
    uint64_t weight_shape[] = {1u, 1u};
    uint64_t bias_shape[] = {1u};
    float x[] = {2.0f};
    float weight[] = {4.0f};
    float bias[] = {1.0f};
    float upstream[] = {3.0f};
    float output[] = {0.0f};
    float dx[] = {0.0f};
    float dweight[] = {0.0f};
    float dbias[] = {0.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, value_shape),
        make_view(weight, sizeof(weight), "f32", 2u, weight_shape),
        make_view(bias, sizeof(bias), "f32", 1u, bias_shape)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, value_shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, value_shape),
        make_view(weight, sizeof(weight), "f32", 2u, weight_shape),
        make_view(upstream, sizeof(upstream), "f32", 3u, value_shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dx, sizeof(dx), "f32", 3u, value_shape),
        make_view(dweight, sizeof(dweight), "f32", 2u, weight_shape),
        make_view(dbias, sizeof(dbias), "f32", 1u, bias_shape)};
    REQUIRE(dispatch(runtime, "kernel.matmul", "linear.forward", 4u,
                     request_shape, forward_inputs, 3u, forward_outputs, 1u));
    REQUIRE(output[0] == 9.0f);
    REQUIRE(dispatch(runtime, "kernel.matmul", "linear.backward", 4u,
                     request_shape, backward_inputs, 3u, backward_outputs,
                     3u));
    REQUIRE(dx[0] == 12.0f && dweight[0] == 6.0f && dbias[0] == 3.0f);
  }

  {
    uint64_t shape[] = {1u, 1u, 1u};
    uint64_t affine_shape[] = {1u};
    float x[] = {5.0f};
    float gamma[] = {2.0f};
    float beta[] = {7.0f};
    float epsilon = 1.0e-5f;
    float upstream[] = {3.0f};
    float output[] = {0.0f};
    float dx[] = {1.0f};
    float dgamma[] = {1.0f};
    float dbeta[] = {0.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, shape),
        make_view(gamma, sizeof(gamma), "f32", 1u, affine_shape),
        make_view(beta, sizeof(beta), "f32", 1u, affine_shape),
        make_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, shape),
        make_view(gamma, sizeof(gamma), "f32", 1u, affine_shape),
        make_view(&epsilon, sizeof(epsilon), "f32", 0u, NULL),
        make_view(upstream, sizeof(upstream), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dx, sizeof(dx), "f32", 3u, shape),
        make_view(dgamma, sizeof(dgamma), "f32", 1u, affine_shape),
        make_view(dbeta, sizeof(dbeta), "f32", 1u, affine_shape)};
    REQUIRE(dispatch(runtime, "kernel.norm", "layer-norm.forward", 3u, shape,
                     forward_inputs, 4u, forward_outputs, 1u));
    REQUIRE(output[0] == 7.0f);
    REQUIRE(dispatch(runtime, "kernel.norm", "layer-norm.backward", 3u,
                     shape, backward_inputs, 4u, backward_outputs, 3u));
    REQUIRE(dx[0] == 0.0f && dgamma[0] == 0.0f && dbeta[0] == 3.0f);
  }

  {
    uint64_t shape[] = {1u, 1u, 1u};
    float zero[] = {0.0f};
    float upstream[] = {3.0f};
    float output[] = {1.0f};
    float dx[] = {1.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(zero, sizeof(zero), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(zero, sizeof(zero), "f32", 3u, shape),
        make_view(upstream, sizeof(upstream), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dx, sizeof(dx), "f32", 3u, shape)};
    REQUIRE(dispatch(runtime, "kernel.activation", "gelu.forward", 3u, shape,
                     forward_inputs, 1u, forward_outputs, 1u));
    REQUIRE(output[0] == 0.0f);
    REQUIRE(dispatch(runtime, "kernel.activation", "gelu.backward", 3u,
                     shape, backward_inputs, 2u, backward_outputs, 1u));
    REQUIRE(dx[0] == 1.5f);
    output[0] = 1.0f;
    dx[0] = 1.0f;
    REQUIRE(dispatch(runtime, "kernel.activation", "relu.forward", 3u, shape,
                     forward_inputs, 1u, forward_outputs, 1u));
    REQUIRE(output[0] == 0.0f);
    REQUIRE(dispatch(runtime, "kernel.activation", "relu.backward", 3u,
                     shape, backward_inputs, 2u, backward_outputs, 1u));
    REQUIRE(dx[0] == 0.0f);
  }

  {
    uint64_t shape[] = {1u, 1u, 1u};
    float x[] = {2.0f};
    float branch[] = {3.0f};
    float upstream[] = {4.0f};
    float output[] = {0.0f};
    float dx[] = {0.0f};
    float dbranch[] = {0.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, shape),
        make_view(branch, sizeof(branch), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(upstream, sizeof(upstream), "f32", 3u, shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dx, sizeof(dx), "f32", 3u, shape),
        make_view(dbranch, sizeof(dbranch), "f32", 3u, shape)};
    REQUIRE(dispatch(runtime, "kernel.residual", "residual.forward", 3u,
                     shape, forward_inputs, 2u, forward_outputs, 1u));
    REQUIRE(output[0] == 5.0f);
    REQUIRE(dispatch(runtime, "kernel.residual", "residual.backward", 3u,
                     shape, backward_inputs, 1u, backward_outputs, 2u));
    REQUIRE(dx[0] == 4.0f && dbranch[0] == 4.0f);
  }

  {
    uint64_t shape[] = {1u, 1u, 4u};
    uint64_t state_shape[] = {4u};
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float upstream[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float probability = 0.5f;
    int64_t state[] = {1, 0, 0, 0};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t next[4] = {0, 0, 0, 0};
    float dx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    et_kernel_tensor_view_v1 forward_inputs[] = {
        make_view(x, sizeof(x), "f32", 3u, shape),
        make_view(&probability, sizeof(probability), "f32", 0u, NULL),
        make_view(state, sizeof(state), "i64", 1u, state_shape)};
    et_kernel_tensor_view_v1 forward_outputs[] = {
        make_view(output, sizeof(output), "f32", 3u, shape),
        make_view(next, sizeof(next), "i64", 1u, state_shape)};
    et_kernel_tensor_view_v1 backward_inputs[] = {
        make_view(upstream, sizeof(upstream), "f32", 3u, shape),
        make_view(&probability, sizeof(probability), "f32", 0u, NULL),
        make_view(state, sizeof(state), "i64", 1u, state_shape)};
    et_kernel_tensor_view_v1 backward_outputs[] = {
        make_view(dx, sizeof(dx), "f32", 3u, shape)};
    REQUIRE(dispatch(runtime, "kernel.dropout", "dropout.train.forward", 3u,
                     shape, forward_inputs, 3u, forward_outputs, 2u));
    REQUIRE(output[0] == 0.0f && output[1] == 2.0f && output[2] == 2.0f &&
            output[3] == 2.0f);
    REQUIRE(next[0] == 1 && next[1] == 0 && next[2] == 1 && next[3] == 0);
    REQUIRE(dispatch(runtime, "kernel.dropout", "dropout.train.backward", 3u,
                     shape, backward_inputs, 3u, backward_outputs, 1u));
    REQUIRE(dx[0] == 0.0f && dx[1] == 4.0f && dx[2] == 4.0f &&
            dx[3] == 4.0f);
  }

  success = 1;
done:
  et_kernel_runtime_destroy(runtime);
#undef REQUIRE
  return success;
}

#if defined(ET_N2_AOT_BRIDGE_TESTING)
int main(void) {
  return et_n2_test_provider_transport_v1() == 1 ? 0 : 1;
}
#endif
