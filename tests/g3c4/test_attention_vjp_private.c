#include "test_common.h"
#include "../../native/m3t_c4_attention_vjp_private.h"
#include <fenv.h>
#include <float.h>

static float q[16], k[16], v[16], u[16], dq[16], dk[16], dv[16], out[16];
static int64_t qp[4] = {0, 1, 2, 3}, kp[4] = {0, 1, 2, 3};
static uint8_t keep[16];
static const uint64_t head_shape[4] = {1u, 2u, 4u, 2u};
static const uint64_t position_shape[2] = {1u, 4u};
static const uint64_t mask_shape[3] = {1u, 4u, 4u};
static const uint64_t request_shape[6] = {1u, 2u, 2u, 4u, 4u, 2u};
static et_kernel_runtime *runtime;

static void views(et_kernel_tensor_view_v1 a[10]) {
  a[0] = tv(q, sizeof(q), "f32", 4u, head_shape);
  a[1] = tv(k, sizeof(k), "f32", 4u, head_shape);
  a[2] = tv(v, sizeof(v), "f32", 4u, head_shape);
  a[3] = tv(qp, sizeof(qp), "i64", 2u, position_shape);
  a[4] = tv(kp, sizeof(kp), "i64", 2u, position_shape);
  a[5] = tv(keep, sizeof(keep), "bool", 3u, mask_shape);
  a[6] = tv(u, sizeof(u), "f32", 4u, head_shape);
  a[7] = tv(dq, sizeof(dq), "f32", 4u, head_shape);
  a[8] = tv(dk, sizeof(dk), "f32", 4u, head_shape);
  a[9] = tv(dv, sizeof(dv), "f32", 4u, head_shape);
}

static int backward(et_kernel_tensor_view_v1 a[10]) {
  return et_m3t_c4_attention_vjp_private_v1(
      &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6],
      &a[7], &a[8], &a[9]);
}

static double loss(void) {
  et_kernel_tensor_view_v1 a[10];
  views(a);
  et_kernel_tensor_view_v1 result = tv(out, sizeof(out), "f32", 4u, head_shape);
  et_kernel_request_v1 request =
      rq("g3c4.causal-attention.forward", 6u, request_shape);
  et_kernel_call_v1 call = kc("g3c4.causal-attention", &request, a, 6u,
                              &result, 1u);
  run(runtime, &call);
  double sum = 0.0;
  for (size_t i = 0; i < 16u; ++i) sum += (double)out[i] * (double)u[i];
  return sum;
}

static void positive(int sparse) {
  for (size_t i = 0; i < 16u; ++i) {
    q[i] = (float)((int)(i * 7u % 13u) - 6) / 16.0f;
    k[i] = (float)((int)(i * 5u % 17u) - 8) / 16.0f;
    v[i] = (float)((int)(i * 3u % 11u) - 5) / 8.0f;
    u[i] = (float)((int)(i * 11u % 19u) - 9) / 32.0f;
    keep[i] = (uint8_t)(!sparse || (i / 4u != 2u && i != 13u));
  }
  memset(dq, 0xa5, sizeof(dq));
  memset(dk, 0xa5, sizeof(dk));
  memset(dv, 0xa5, sizeof(dv));
  et_kernel_tensor_view_v1 a[10];
  views(a);
  CHECK(backward(a) == ET_M3T_C4_ATTENTION_VJP_OK);
  const float step = 0.00390625f;
  float *fields[3] = {q, k, v};
  const float *grads[3] = {dq, dk, dv};
  for (size_t field = 0; field < 3u; ++field)
    for (size_t i = 0; i < 16u; ++i) {
      const float original = fields[field][i];
      fields[field][i] = original + step;
      const double high = loss();
      fields[field][i] = original - step;
      const double low = loss();
      fields[field][i] = original;
      CHECK(fabs((high - low) / (2.0 * step) - (double)grads[field][i]) <
            0.002);
    }
  if (sparse)
    for (size_t h = 0; h < 2u; ++h)
      for (size_t d = 0; d < 2u; ++d)
        CHECK(fbits(dq[(h * 4u + 2u) * 2u + d]) == fbits(0.0f));
}

static void reject_unchanged(et_kernel_tensor_view_v1 a[10], int expected) {
  unsigned char before[3][sizeof(dq)];
  memcpy(before[0], dq, sizeof(dq));
  memcpy(before[1], dk, sizeof(dk));
  memcpy(before[2], dv, sizeof(dv));
  CHECK(backward(a) == expected);
  CHECK(memcmp(before[0], dq, sizeof(dq)) == 0);
  CHECK(memcmp(before[1], dk, sizeof(dk)) == 0);
  CHECK(memcmp(before[2], dv, sizeof(dv)) == 0);
}

static void negatives(void) {
  et_kernel_tensor_view_v1 a[10];
  views(a);
  a[0].rank = 3u;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_SHAPE);
  views(a); a[1].byte_length = 60u;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_SHAPE);
  views(a); uint64_t wrong[4] = {1u, 2u, 3u, 2u}; a[2].shape = wrong;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_SHAPE);
  views(a); a[5].dtype = "i64";
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_DTYPE);
  views(a); a[8].device = "gpu";
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_DEVICE);
  views(a); a[9].offset_bytes = 4u;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_LAYOUT);
  views(a); a[7].data = q;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_OVERLAP);
  views(a); a[8].data = dq;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_OVERLAP);
  views(a); q[0] = NAN;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  q[0] = 0.0f; k[0] = INFINITY;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  k[0] = 0.0f; v[0] = NAN;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  v[0] = 0.0f; u[0] = INFINITY;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  u[0] = 0.0f; qp[0] = 1;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  qp[0] = 0; kp[1] = 3;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  kp[1] = 1; keep[0] = 2;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  keep[0] = 1; q[0] = FLT_MAX; k[0] = FLT_MAX;
  reject_unchanged(a, ET_M3T_C4_ATTENTION_VJP_VALUE);
  q[0] = 0.0f; k[0] = 0.0f;
  CHECK(et_m3t_c4_attention_vjp_private_v1(NULL, &a[1], &a[2], &a[3],
      &a[4], &a[5], &a[6], &a[7], &a[8], &a[9]) ==
      ET_M3T_C4_ATTENTION_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  runtime = runtime_new();
  positive(0);
  positive(1);
  negatives();
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 attention VJP checks=%zu\n", checks);
  return 0;
}
