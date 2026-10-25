#include "test_common.h"
#include "../../native/m3t_c4_residual_vjp_private.h"
#include <fenv.h>
#include <float.h>

static const uint64_t shape[3] = {1u, 4u, 4u};
static const uint64_t request_shape[3] = {1u, 4u, 4u};
static float left_data[16], right_data[16], upstream_data[16];
static float left_gradient_data[16], right_gradient_data[16], output_data[16];

static et_kernel_tensor_view_v1 view(float *data) {
  return tv(data, 64u, "f32", 3u, shape);
}

static double loss(et_kernel_runtime *runtime) {
  et_kernel_tensor_view_v1 inputs[2] = {view(left_data), view(right_data)};
  et_kernel_tensor_view_v1 output = view(output_data);
  et_kernel_request_v1 request = rq("g3c4.residual.forward", 3u, request_shape);
  et_kernel_call_v1 call = kc("g3c4.residual", &request, inputs, 2u, &output, 1u);
  run(runtime, &call);
  double result = 0.0;
  for (size_t i = 0; i < 16u; ++i)
    result += (double)output_data[i] * (double)upstream_data[i];
  return result;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *left,
                             et_kernel_tensor_view_v1 *right,
                             et_kernel_tensor_view_v1 *upstream,
                             et_kernel_tensor_view_v1 *left_gradient,
                             et_kernel_tensor_view_v1 *right_gradient,
                             int expected) {
  unsigned char before_left[64], before_right[64];
  memcpy(before_left, left_gradient->data, 64u);
  memcpy(before_right, right_gradient->data, 64u);
  CHECK(et_m3t_c4_residual_vjp_private_v1(left, right, upstream,
         left_gradient, right_gradient) == expected);
  CHECK(memcmp(before_left, left_gradient->data, 64u) == 0);
  CHECK(memcmp(before_right, right_gradient->data, 64u) == 0);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  et_kernel_runtime *runtime = runtime_new();
  for (size_t i = 0; i < 16u; ++i) {
    left_data[i] = (float)((int)i - 8) / 8.0f;
    right_data[i] = (float)((int)(i % 7u) - 3) / 16.0f;
    upstream_data[i] = (float)((int)(i % 5u) - 2) / 8.0f;
    left_gradient_data[i] = 37.0f;
    right_gradient_data[i] = -41.0f;
  }
  et_kernel_tensor_view_v1 left = view(left_data), right = view(right_data);
  et_kernel_tensor_view_v1 upstream = view(upstream_data);
  et_kernel_tensor_view_v1 left_gradient = view(left_gradient_data);
  et_kernel_tensor_view_v1 right_gradient = view(right_gradient_data);
  unsigned char original_left[64], original_right[64], original_upstream[64];
  memcpy(original_left, left_data, 64u);
  memcpy(original_right, right_data, 64u);
  memcpy(original_upstream, upstream_data, 64u);
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, &right, &upstream,
        &left_gradient, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_OK);
  CHECK(memcmp(original_left, left_data, 64u) == 0);
  CHECK(memcmp(original_right, right_data, 64u) == 0);
  CHECK(memcmp(original_upstream, upstream_data, 64u) == 0);
  for (size_t operand = 0; operand < 2u; ++operand) {
    float *data = operand == 0u ? left_data : right_data;
    float *gradient = operand == 0u ? left_gradient_data : right_gradient_data;
    for (size_t i = 0; i < 16u; ++i) {
      const float original = data[i];
      data[i] = original + 0.0625f;
      const double high = loss(runtime);
      data[i] = original - 0.0625f;
      const double low = loss(runtime);
      data[i] = original;
      CHECK(fabs((high - low) / 0.125 - (double)gradient[i]) < 1e-6);
      CHECK(fbits(gradient[i]) == fbits(upstream_data[i]));
    }
  }

  left.struct_size = 0u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  left = view(left_data); left.byte_length = 60u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_SHAPE);
  left = view(left_data); uint64_t wrong_shape[3] = {1u, 3u, 4u};
  left.shape = wrong_shape;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_SHAPE);
  left = view(left_data); left.dtype = "i64";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DTYPE);
  left = view(left_data); left.device = "gpu";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DEVICE);
  left = view(left_data); left.offset_bytes = 4u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_LAYOUT);
  left = view(left_data); right.rank = 2u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_SHAPE);
  right = view(right_data); right.layout = 0u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_LAYOUT);
  right = view(right_data); upstream.byte_length = 60u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_SHAPE);
  upstream = view(upstream_data); upstream.dtype = "i64";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DTYPE);
  upstream = view(upstream_data); upstream.device = "gpu";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DEVICE);
  upstream = view(upstream_data); left_gradient.rank = 2u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_SHAPE);
  left_gradient = view(left_gradient_data); left_gradient.dtype = "i64";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DTYPE);
  left_gradient = view(left_gradient_data); right_gradient.device = "gpu";
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_DEVICE);
  right_gradient = view(right_gradient_data); right_gradient.offset_bytes = 4u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_LAYOUT);
  right_gradient = view(right_gradient_data);
  upstream.data = NULL;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  upstream = view(upstream_data); right_gradient.shape = NULL;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  right_gradient = view(right_gradient_data);
  _Alignas(4) unsigned char unaligned[65] = {0};
  left_gradient.data = unaligned + 1u;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_LAYOUT);
  left_gradient = view(left_gradient_data); left_gradient.data = left_data;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_OVERLAP);
  left_gradient = view(left_gradient_data); right_gradient.data = right_data;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_OVERLAP);
  right_gradient = view(right_gradient_data); right_gradient.data = left_gradient_data;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_OVERLAP);
  right_gradient = view(right_gradient_data);
  float partial_alias[17] = {0};
  upstream.data = partial_alias;
  left_gradient.data = partial_alias + 1;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_OVERLAP);
  upstream = view(upstream_data); left_gradient = view(left_gradient_data);
  left_data[2] = NAN;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_VALUE);
  left_data[2] = -0.75f; right_data[4] = INFINITY;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_VALUE);
  right_data[4] = 0.0625f; upstream_data[9] = NAN;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_VALUE);
  upstream_data[9] = 0.25f; left_data[0] = FLT_MAX; right_data[0] = FLT_MAX;
  reject_unchanged(&left, &right, &upstream, &left_gradient, &right_gradient,
                   ET_M3T_C4_RESIDUAL_VJP_VALUE);
  left_data[0] = -1.0f; right_data[0] = -0.1875f;
  unsigned char before_left[64], before_right[64];
  memcpy(before_left, left_gradient_data, 64u);
  memcpy(before_right, right_gradient_data, 64u);
  CHECK(et_m3t_c4_residual_vjp_private_v1(NULL, &right, &upstream,
        &left_gradient, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, NULL, &upstream,
        &left_gradient, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, &right, NULL,
        &left_gradient, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, &right, &upstream,
        NULL, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, &right, &upstream,
        &left_gradient, NULL) == ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT);
  CHECK(memcmp(before_left, left_gradient_data, 64u) == 0);
  CHECK(memcmp(before_right, right_gradient_data, 64u) == 0);
  right.data = left_data;
  CHECK(et_m3t_c4_residual_vjp_private_v1(&left, &right, &upstream,
        &left_gradient, &right_gradient) == ET_M3T_C4_RESIDUAL_VJP_OK);
  CHECK(memcmp(left_gradient_data, upstream_data, 64u) == 0);
  CHECK(memcmp(right_gradient_data, upstream_data, 64u) == 0);
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 residual VJP checks=%zu\n", checks);
  return 0;
}
