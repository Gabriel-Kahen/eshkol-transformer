#include "test_common.h"
#include "../../native/m3t_c4_position_vjp_private.h"
#include <stddef.h>

static const uint64_t id_shape[2] = {1, 4};
static const uint64_t output_shape[3] = {1, 4, 4};
static const uint64_t weight_shape[2] = {4, 4};
static const uint64_t request_shape[4] = {1, 4, 4, 4};
static int64_t positions[4] = {0, 1, 2, 3};
static float weights[16], upstream_data[16], gradient_data[16], output_data[16];

static et_kernel_tensor_view_v1 ids_view(void) {
  return tv(positions, sizeof(positions), "i64", 2, id_shape);
}
static et_kernel_tensor_view_v1 upstream_view(void) {
  return tv(upstream_data, sizeof(upstream_data), "f32", 3, output_shape);
}
static et_kernel_tensor_view_v1 gradient_view(void) {
  return tv(gradient_data, sizeof(gradient_data), "f32", 2, weight_shape);
}

static double forward_loss(et_kernel_runtime *runtime) {
  et_kernel_tensor_view_v1 inputs[2] = {
    ids_view(), tv(weights, sizeof(weights), "f32", 2, weight_shape)
  };
  et_kernel_tensor_view_v1 output =
      tv(output_data, sizeof(output_data), "f32", 3, output_shape);
  et_kernel_request_v1 request =
      rq("g3c4.embedding.forward", 4, request_shape);
  et_kernel_call_v1 call = kc("g3c4.embedding-forward", &request,
                              inputs, 2, &output, 1);
  run(runtime, &call);
  double sum = 0.0;
  for (size_t i = 0; i < 16; ++i)
    sum += (double)output_data[i] * (double)upstream_data[i];
  return sum;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *ids,
                             et_kernel_tensor_view_v1 *upstream,
                             et_kernel_tensor_view_v1 *gradient,
                             int expected) {
  unsigned char prior[64];
  memcpy(prior, gradient->data, sizeof(prior));
  CHECK(et_m3t_c4_position_vjp_private_v1(ids, upstream, gradient) == expected);
  CHECK(memcmp(prior, gradient->data, sizeof(prior)) == 0);
}

int main(void) {
  et_kernel_runtime *runtime = runtime_new();
  for (size_t i = 0; i < 16; ++i) {
    weights[i] = (float)((int)i - 8) / 8.0f;
    upstream_data[i] = (float)((int)(i % 5) - 2) / 8.0f;
    gradient_data[i] = 37.0f;
  }
  et_kernel_tensor_view_v1 ids = ids_view();
  et_kernel_tensor_view_v1 upstream = upstream_view();
  et_kernel_tensor_view_v1 gradient = gradient_view();
  CHECK(et_m3t_c4_position_vjp_private_v1(&ids, &upstream, &gradient) ==
        ET_M3T_C4_POSITION_VJP_OK);
  for (size_t i = 0; i < 16; ++i) {
    float old = weights[i];
    weights[i] = old + 0.0625f;
    double high = forward_loss(runtime);
    weights[i] = old - 0.0625f;
    double low = forward_loss(runtime);
    weights[i] = old;
    CHECK(fabs((high - low) / 0.125 - (double)gradient_data[i]) < 1e-6);
    CHECK(fbits(gradient_data[i]) == fbits(upstream_data[i]));
  }

  ids.struct_size = 0;
  reject_unchanged(&ids, &upstream, &gradient,
                   ET_M3T_C4_POSITION_VJP_INVALID_ARGUMENT);
  ids = ids_view();
  ids.byte_length = 24;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  ids = ids_view();
  uint64_t wrong_ids_shape[2] = {1, 3};
  ids.shape = wrong_ids_shape;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  ids = ids_view();
  ids.dtype = "f32";
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_DTYPE);
  ids = ids_view();
  ids.offset_bytes = 8;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_LAYOUT);
  ids = ids_view();
  upstream.rank = 2;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  upstream = upstream_view();
  uint64_t wrong_up_shape[3] = {1, 3, 4};
  upstream.shape = wrong_up_shape;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  upstream = upstream_view();
  upstream.offset_bytes = 4;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_LAYOUT);
  upstream = upstream_view();
  gradient.rank = 3;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  gradient = gradient_view();
  gradient.byte_length = 60;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_SHAPE);
  gradient = gradient_view();
  gradient.dtype = "i64";
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_DTYPE);
  gradient = gradient_view();
  gradient.device = "gpu";
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_DEVICE);
  gradient = gradient_view();
  gradient.layout = 0;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_LAYOUT);
  gradient = gradient_view();
  _Alignas(4) unsigned char unaligned[65] = {0};
  gradient.data = unaligned + 1;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_LAYOUT);
  gradient = gradient_view();
  gradient.data = upstream_data;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_OVERLAP);
  float partial_alias[17] = {0};
  upstream.data = partial_alias;
  gradient.data = partial_alias + 1;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_OVERLAP);
  upstream = upstream_view();
  gradient = gradient_view();
  positions[2] = 1;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_VALUE);
  positions[2] = 2;
  upstream_data[9] = NAN;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_VALUE);
  upstream_data[9] = INFINITY;
  reject_unchanged(&ids, &upstream, &gradient, ET_M3T_C4_POSITION_VJP_VALUE);
  upstream_data[9] = 0.25f;
  unsigned char before_nulls[sizeof(gradient_data)];
  memcpy(before_nulls, gradient_data, sizeof(before_nulls));
  CHECK(et_m3t_c4_position_vjp_private_v1(NULL, &upstream, &gradient) ==
        ET_M3T_C4_POSITION_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_position_vjp_private_v1(&ids, NULL, &gradient) ==
        ET_M3T_C4_POSITION_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_position_vjp_private_v1(&ids, &upstream, NULL) ==
        ET_M3T_C4_POSITION_VJP_INVALID_ARGUMENT);
  CHECK(memcmp(before_nulls, gradient_data, sizeof(before_nulls)) == 0);
  CHECK(et_m3t_c4_position_vjp_private_v1(&ids, &upstream, &gradient) ==
        ET_M3T_C4_POSITION_VJP_OK);
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 position VJP checks=%zu\n", checks);
  return 0;
}
