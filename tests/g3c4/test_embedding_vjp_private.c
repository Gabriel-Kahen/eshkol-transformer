#include "test_common.h"
#include "../../native/m3t_c4_embedding_vjp_private.h"
#include <fenv.h>
#include <float.h>

static const uint64_t id_shape[2] = {1u, 4u};
static const uint64_t output_shape[3] = {1u, 4u, 4u};
static const uint64_t weight_shape[2] = {256u, 4u};
static const uint64_t request_shape[4] = {1u, 4u, 256u, 4u};
static int64_t ids[4] = {17, 17, 255, 0};
static float weights[1024], u[16], grad[1024], output[16];
static et_kernel_runtime *runtime;

static et_kernel_tensor_view_v1 idv(void) {
  return tv(ids, sizeof(ids), "i64", 2u, id_shape);
}
static et_kernel_tensor_view_v1 uv(void) {
  return tv(u, sizeof(u), "f32", 3u, output_shape);
}
static et_kernel_tensor_view_v1 gv(void) {
  return tv(grad, sizeof(grad), "f32", 2u, weight_shape);
}

static double loss(void) {
  et_kernel_tensor_view_v1 inputs[2] = {
    idv(), tv(weights, sizeof(weights), "f32", 2u, weight_shape)};
  et_kernel_tensor_view_v1 result =
      tv(output, sizeof(output), "f32", 3u, output_shape);
  et_kernel_request_v1 request =
      rq("g3c4.embedding.forward", 4u, request_shape);
  et_kernel_call_v1 call =
      kc("g3c4.embedding-forward", &request, inputs, 2u, &result, 1u);
  run(runtime, &call);
  double sum = 0.0;
  for (size_t i = 0; i < 16u; ++i) sum += (double)output[i] * (double)u[i];
  return sum;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *a,
                             et_kernel_tensor_view_v1 *up,
                             et_kernel_tensor_view_v1 *g, int expected) {
  unsigned char before[sizeof(grad)], before_u[sizeof(u)], before_ids[sizeof(ids)];
  memcpy(before, grad, sizeof(grad));
  memcpy(before_u, u, sizeof(u));
  memcpy(before_ids, ids, sizeof(ids));
  CHECK(et_m3t_c4_embedding_vjp_private_v1(a, up, g) == expected);
  CHECK(memcmp(before, grad, sizeof(grad)) == 0);
  CHECK(memcmp(before_u, u, sizeof(u)) == 0);
  CHECK(memcmp(before_ids, ids, sizeof(ids)) == 0);
}

static void positive(void) {
  for (size_t i = 0; i < 1024u; ++i)
    weights[i] = (float)((int)(i % 29u) - 14) / 32.0f;
  for (size_t i = 0; i < 16u; ++i)
    u[i] = (float)((int)(i * 7u % 17u) - 8) / 16.0f;
  memset(grad, 0xa5, sizeof(grad));
  et_kernel_tensor_view_v1 a = idv(), up = uv(), g = gv();
  CHECK(et_m3t_c4_embedding_vjp_private_v1(&a, &up, &g) ==
        ET_M3T_C4_EMBEDDING_VJP_OK);
  for (size_t row = 0; row < 256u; ++row)
    for (size_t d = 0; d < 4u; ++d) {
      float expected = 0.0f;
      for (size_t t = 0; t < 4u; ++t)
        if (ids[t] == (int64_t)row) expected += u[t * 4u + d];
      CHECK(fbits(grad[row * 4u + d]) == fbits(expected));
    }
  const float step = 0.015625f;
  for (size_t t = 0; t < 4u; ++t)
    for (size_t d = 0; d < 4u; ++d) {
      const size_t i = (size_t)ids[t] * 4u + d;
      const float original = weights[i];
      weights[i] = original + step; const double high = loss();
      weights[i] = original - step; const double low = loss();
      weights[i] = original;
      CHECK(fabs((high - low) / (2.0 * step) - (double)grad[i]) < 1e-6);
    }
  for (size_t sample = 0; sample < 32u; ++sample) {
    const size_t i = sample * 31u;
    if (i / 4u == 0u || i / 4u == 17u || i / 4u == 255u) continue;
    const float original = weights[i];
    weights[i] = original + step; const double high = loss();
    weights[i] = original - step; const double low = loss();
    weights[i] = original;
    CHECK(fabs((high - low) / (2.0 * step) - (double)grad[i]) < 1e-6);
  }
}

static void negatives(void) {
  et_kernel_tensor_view_v1 a = idv(), up = uv(), g = gv();
  a.struct_size = 0;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT);
  a = idv(); a.byte_length = 24u;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_SHAPE);
  a = idv(); uint64_t wrong[2] = {1u, 3u}; a.shape = wrong;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_SHAPE);
  a = idv(); a.dtype = "f32";
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_DTYPE);
  a = idv(); a.offset_bytes = 8u;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_LAYOUT);
  a = idv(); up.rank = 2u;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_SHAPE);
  up = uv(); g.device = "gpu";
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_DEVICE);
  g = gv(); g.data = u;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_OVERLAP);
  g = gv(); ids[0] = -1;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_VALUE);
  ids[0] = 256;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_VALUE);
  ids[0] = 17; u[0] = NAN;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_VALUE);
  u[0] = INFINITY;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_VALUE);
  u[0] = FLT_MAX; u[4] = FLT_MAX;
  reject_unchanged(&a, &up, &g, ET_M3T_C4_EMBEDDING_VJP_VALUE);
  u[0] = 0.0f; u[4] = 0.0f;
  CHECK(et_m3t_c4_embedding_vjp_private_v1(NULL, &up, &g) ==
        ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_embedding_vjp_private_v1(&a, NULL, &g) ==
        ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_embedding_vjp_private_v1(&a, &up, NULL) ==
        ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  runtime = runtime_new();
  positive();
  negatives();
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 token embedding VJP checks=%zu\n", checks);
  return 0;
}
