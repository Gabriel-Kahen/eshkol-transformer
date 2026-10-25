#include "test_common.h"
#include "../../native/m3t_c4_gelu_vjp_private.h"
#include <fenv.h>
#include <float.h>

static const uint64_t shape[3] = {1u, 4u, 8u};
static float input_data[32], upstream_data[32], gradient_data[32], output_data[32];

static et_kernel_tensor_view_v1 view(float *data) {
  return tv(data, 128u, "f32", 3u, shape);
}

static double loss(et_kernel_runtime *runtime) {
  et_kernel_tensor_view_v1 input = view(input_data);
  et_kernel_tensor_view_v1 output = view(output_data);
  et_kernel_request_v1 request = rq("g3c4.gelu.forward", 3u, shape);
  et_kernel_call_v1 call = kc("g3c4.gelu", &request, &input, 1u, &output, 1u);
  run(runtime, &call);
  double result = 0.0;
  for (size_t i = 0; i < 32u; ++i)
    result += (double)output_data[i] * (double)upstream_data[i];
  return result;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *input,
                             et_kernel_tensor_view_v1 *upstream,
                             et_kernel_tensor_view_v1 *gradient,
                             int expected) {
  unsigned char before[128];
  memcpy(before, gradient->data, sizeof(before));
  CHECK(et_m3t_c4_gelu_vjp_private_v1(input, upstream, gradient) == expected);
  CHECK(memcmp(before, gradient->data, sizeof(before)) == 0);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  et_kernel_runtime *runtime = runtime_new();
  for (size_t i = 0; i < 32u; ++i) {
    input_data[i] = (float)((int)i - 16) / 8.0f;
    upstream_data[i] = (float)((int)(i % 7u) - 3) / 8.0f + 0.0625f;
    gradient_data[i] = 37.0f;
  }
  et_kernel_tensor_view_v1 input = view(input_data);
  et_kernel_tensor_view_v1 upstream = view(upstream_data);
  et_kernel_tensor_view_v1 gradient = view(gradient_data);
  unsigned char before_input[128], before_upstream[128];
  memcpy(before_input, input_data, 128u);
  memcpy(before_upstream, upstream_data, 128u);
  CHECK(et_m3t_c4_gelu_vjp_private_v1(&input, &upstream, &gradient) ==
        ET_M3T_C4_GELU_VJP_OK);
  CHECK(memcmp(before_input, input_data, 128u) == 0);
  CHECK(memcmp(before_upstream, upstream_data, 128u) == 0);
  for (size_t i = 0; i < 32u; ++i) {
    const float original = input_data[i];
    input_data[i] = original + 0.0078125f;
    const double high = loss(runtime);
    input_data[i] = original - 0.0078125f;
    const double low = loss(runtime);
    input_data[i] = original;
    CHECK(fabs((high - low) / 0.015625 - (double)gradient_data[i]) < 1e-4);
  }
  CHECK(fabsf(gradient_data[16] - 0.5f * upstream_data[16]) < 1e-7f);

  input.struct_size = 0u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT);
  input = view(input_data); input.byte_length = 124u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_SHAPE);
  input = view(input_data); uint64_t wrong_shape[3] = {1u, 3u, 8u};
  input.shape = wrong_shape;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_SHAPE);
  input = view(input_data); input.dtype = "i64";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DTYPE);
  input = view(input_data); input.device = "gpu";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DEVICE);
  input = view(input_data); input.offset_bytes = 4u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_LAYOUT);
  input = view(input_data); upstream.rank = 2u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_SHAPE);
  upstream = view(upstream_data); upstream.dtype = "i64";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DTYPE);
  upstream = view(upstream_data); upstream.device = "gpu";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DEVICE);
  upstream = view(upstream_data); upstream.layout = 0u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_LAYOUT);
  upstream = view(upstream_data); gradient.rank = 2u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_SHAPE);
  gradient = view(gradient_data); gradient.byte_length = 124u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_SHAPE);
  gradient = view(gradient_data); gradient.dtype = "i64";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DTYPE);
  gradient = view(gradient_data); gradient.device = "gpu";
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_DEVICE);
  gradient = view(gradient_data); gradient.layout = 0u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_LAYOUT);
  gradient = view(gradient_data);
  _Alignas(4) unsigned char unaligned[129] = {0};
  gradient.data = unaligned + 1u;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_LAYOUT);
  gradient = view(gradient_data); gradient.data = input_data;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_OVERLAP);
  gradient = view(gradient_data); gradient.data = upstream_data;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_OVERLAP);
  float partial_alias[33] = {0};
  upstream.data = partial_alias;
  gradient.data = partial_alias + 1;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_OVERLAP);
  upstream = view(upstream_data); gradient = view(gradient_data);
  input_data[3] = NAN;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_VALUE);
  input_data[3] = -1.625f; upstream_data[9] = INFINITY;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_VALUE);
  upstream_data[9] = -0.0625f; input_data[24] = 1.0f; upstream_data[24] = FLT_MAX;
  reject_unchanged(&input, &upstream, &gradient, ET_M3T_C4_GELU_VJP_VALUE);
  input_data[24] = 1.0f; upstream_data[24] = (float)((int)(24u % 7u) - 3) / 8.0f + 0.0625f;
  unsigned char before_null[128];
  memcpy(before_null, gradient_data, sizeof(before_null));
  CHECK(et_m3t_c4_gelu_vjp_private_v1(NULL, &upstream, &gradient) ==
        ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_gelu_vjp_private_v1(&input, NULL, &gradient) ==
        ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_gelu_vjp_private_v1(&input, &upstream, NULL) ==
        ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT);
  CHECK(memcmp(before_null, gradient_data, sizeof(before_null)) == 0);
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 GELU VJP checks=%zu\n", checks);
  return 0;
}
