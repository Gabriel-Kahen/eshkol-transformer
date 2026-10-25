#include "test_common.h"
#include "../../native/m3t_c4_linear_vjp_private.h"
#include <fenv.h>
#include <float.h>

static float x[32], w[1024], u[1024], dx[32], dw[1024], output[1024];
static uint64_t xs[3], ws[2], us[3], request_shape[4];
static size_t input_width, output_width;
static et_kernel_runtime *runtime;

static et_kernel_tensor_view_v1 xv(void) {
  return tv(x, 16u * input_width, "f32", 3u, xs);
}
static et_kernel_tensor_view_v1 wv(void) {
  return tv(w, 4u * input_width * output_width, "f32", 2u, ws);
}
static et_kernel_tensor_view_v1 uv(void) {
  return tv(u, 16u * output_width, "f32", 3u, us);
}
static et_kernel_tensor_view_v1 dxv(void) {
  return tv(dx, 16u * input_width, "f32", 3u, xs);
}
static et_kernel_tensor_view_v1 dwv(void) {
  return tv(dw, 4u * input_width * output_width, "f32", 2u, ws);
}

static double loss(void) {
  et_kernel_tensor_view_v1 inputs[2] = {xv(), wv()};
  et_kernel_tensor_view_v1 result = tv(output, 16u * output_width,
                                       "f32", 3u, us);
  et_kernel_request_v1 request =
      rq("g3c4.linear.forward-no-bias", 4u, request_shape);
  et_kernel_call_v1 call =
      kc("g3c4.linear", &request, inputs, 2u, &result, 1u);
  run(runtime, &call);
  double sum = 0.0;
  for (size_t i = 0; i < 4u * output_width; ++i)
    sum += (double)output[i] * (double)u[i];
  return sum;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *a,
                             et_kernel_tensor_view_v1 *b,
                             et_kernel_tensor_view_v1 *g,
                             et_kernel_tensor_view_v1 *da,
                             et_kernel_tensor_view_v1 *db, int expected) {
  unsigned char before_dx[sizeof(dx)], before_dw[sizeof(dw)];
  unsigned char before_da[sizeof(dx)], before_db[sizeof(dw)];
  memcpy(before_dx, dx, sizeof(dx));
  memcpy(before_dw, dw, sizeof(dw));
  memcpy(before_da, da->data, 16u * input_width);
  memcpy(before_db, db->data, 4u * input_width * output_width);
  CHECK(et_m3t_c4_linear_vjp_private_v1(a, b, g, da, db) == expected);
  CHECK(memcmp(before_dx, dx, sizeof(dx)) == 0);
  CHECK(memcmp(before_dw, dw, sizeof(dw)) == 0);
  CHECK(memcmp(before_da, da->data, 16u * input_width) == 0);
  CHECK(memcmp(before_db, db->data, 4u * input_width * output_width) == 0);
}

static void run_row(size_t in_width, size_t out_width) {
  input_width = in_width;
  output_width = out_width;
  xs[0] = 1u; xs[1] = 4u; xs[2] = in_width;
  ws[0] = out_width; ws[1] = in_width;
  us[0] = 1u; us[1] = 4u; us[2] = out_width;
  request_shape[0] = 1u; request_shape[1] = 4u;
  request_shape[2] = in_width; request_shape[3] = out_width;
  for (size_t i = 0; i < 4u * in_width; ++i)
    x[i] = (float)((int)(i % 11u) - 5) / 32.0f;
  for (size_t i = 0; i < in_width * out_width; ++i)
    w[i] = (float)((int)(i % 17u) - 8) / 64.0f;
  for (size_t i = 0; i < 4u * out_width; ++i)
    u[i] = (float)((int)(i % 13u) - 6) / 32.0f + 0.0078125f;
  memset(dx, 0x5a, sizeof(dx));
  memset(dw, 0xa5, sizeof(dw));
  et_kernel_tensor_view_v1 a = xv(), b = wv(), g = uv(), da = dxv(), db = dwv();
  unsigned char before_x[sizeof(x)], before_w[sizeof(w)], before_u[sizeof(u)];
  memcpy(before_x, x, sizeof(x));
  memcpy(before_w, w, sizeof(w));
  memcpy(before_u, u, sizeof(u));
  CHECK(et_m3t_c4_linear_vjp_private_v1(&a, &b, &g, &da, &db) ==
        ET_M3T_C4_LINEAR_VJP_OK);
  CHECK(memcmp(before_x, x, sizeof(x)) == 0);
  CHECK(memcmp(before_w, w, sizeof(w)) == 0);
  CHECK(memcmp(before_u, u, sizeof(u)) == 0);
  const float step = 0.0078125f;
  for (size_t i = 0; i < 4u * in_width; ++i) {
    const float original = x[i];
    x[i] = original + step; const double high = loss();
    x[i] = original - step; const double low = loss();
    x[i] = original;
    CHECK(fabs((high - low) / (2.0 * step) - (double)dx[i]) < 0.001);
  }
  const size_t count = in_width * out_width;
  const size_t tested = count <= 32u ? count : 32u;
  for (size_t sample = 0; sample < tested; ++sample) {
    const size_t i = count <= 32u ? sample :
                     sample * (count - 1u) / (tested - 1u);
    const float original = w[i];
    w[i] = original + step; const double high = loss();
    w[i] = original - step; const double low = loss();
    w[i] = original;
    CHECK(fabs((high - low) / (2.0 * step) - (double)dw[i]) < 0.001);
  }
  CHECK(isfinite(loss()));
}

static void negatives(void) {
  input_width = 4u; output_width = 4u;
  xs[0] = 1u; xs[1] = 4u; xs[2] = 4u;
  ws[0] = 4u; ws[1] = 4u;
  us[0] = 1u; us[1] = 4u; us[2] = 4u;
  et_kernel_tensor_view_v1 a = xv(), b = wv(), g = uv(), da = dxv(), db = dwv();
  a.struct_size = 0u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  a = xv(); a.rank = 2u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  a = xv(); a.byte_length = 60u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_SHAPE);
  a = xv(); _Alignas(8) unsigned char shape_bytes[25] = {0};
  a.shape = (const uint64_t *)(shape_bytes + 1u);
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  a = xv(); uint64_t wrong[3] = {1u, 3u, 4u}; a.shape = wrong;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_SHAPE);
  a = xv(); ws[0] = 5u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_SHAPE);
  ws[0] = 4u;
  a = xv(); a.dtype = "i64";
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_DTYPE);
  a = xv(); b.device = "gpu";
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_DEVICE);
  b = wv(); g.layout = 0u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_LAYOUT);
  g = uv(); da.offset_bytes = 4u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_LAYOUT);
  da = dxv(); db.byte_length = 60u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_SHAPE);
  db = dwv(); da.data = x;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_OVERLAP);
  da = dxv(); db.data = dx;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_OVERLAP);
  db = dwv(); float partial_alias[17] = {0};
  a.data = partial_alias; da.data = partial_alias + 1u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_OVERLAP);
  a = xv(); da = dxv();
  db = dwv(); _Alignas(4) unsigned char unaligned[65] = {0};
  db.data = unaligned + 1u;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_LAYOUT);
  db = dwv(); x[0] = NAN;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_VALUE);
  x[0] = 0.0f; w[0] = INFINITY;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_VALUE);
  w[0] = 0.0f; u[0] = NAN;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_VALUE);
  u[0] = 0.0f; x[0] = FLT_MAX; w[0] = 2.0f;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_VALUE);
  x[0] = 0.0f; w[0] = 0.0f; u[0] = FLT_MAX; w[0] = 2.0f;
  reject_unchanged(&a, &b, &g, &da, &db, ET_M3T_C4_LINEAR_VJP_VALUE);
  CHECK(et_m3t_c4_linear_vjp_private_v1(NULL, &b, &g, &da, &db) ==
        ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_linear_vjp_private_v1(&a, NULL, &g, &da, &db) ==
        ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_linear_vjp_private_v1(&a, &b, NULL, &da, &db) ==
        ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_linear_vjp_private_v1(&a, &b, &g, NULL, &db) ==
        ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_linear_vjp_private_v1(&a, &b, &g, &da, NULL) ==
        ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  runtime = runtime_new();
  run_row(4u, 4u);
  run_row(4u, 8u);
  run_row(8u, 4u);
  run_row(4u, 256u);
  negatives();
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 linear VJP checks=%zu\n", checks);
  return 0;
}
