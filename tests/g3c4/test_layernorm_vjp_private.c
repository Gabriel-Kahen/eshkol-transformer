#include "test_common.h"
#include "../../native/m3t_c4_layernorm_vjp_private.h"
#include <fenv.h>
#include <float.h>

static float x[16], gamma[4], beta[4], epsilon = 1e-5f, u[16];
static float dx[16], dgamma[4], dbeta[4], output[16];
static const uint64_t tensor_shape[3] = {1u, 4u, 4u};
static const uint64_t channel_shape[1] = {4u};
static et_kernel_runtime *runtime;

static et_kernel_tensor_view_v1 xv(void) {
  return tv(x, sizeof(x), "f32", 3u, tensor_shape);
}
static et_kernel_tensor_view_v1 gv(void) {
  return tv(gamma, sizeof(gamma), "f32", 1u, channel_shape);
}
static et_kernel_tensor_view_v1 bv(void) {
  return tv(beta, sizeof(beta), "f32", 1u, channel_shape);
}
static et_kernel_tensor_view_v1 ev(void) {
  return tv(&epsilon, sizeof(epsilon), "f32", 0u, channel_shape);
}
static et_kernel_tensor_view_v1 uv(void) {
  return tv(u, sizeof(u), "f32", 3u, tensor_shape);
}
static et_kernel_tensor_view_v1 dxv(void) {
  return tv(dx, sizeof(dx), "f32", 3u, tensor_shape);
}
static et_kernel_tensor_view_v1 dgv(void) {
  return tv(dgamma, sizeof(dgamma), "f32", 1u, channel_shape);
}
static et_kernel_tensor_view_v1 dbv(void) {
  return tv(dbeta, sizeof(dbeta), "f32", 1u, channel_shape);
}

static int vjp(et_kernel_tensor_view_v1 *a, et_kernel_tensor_view_v1 *g,
               et_kernel_tensor_view_v1 *b, et_kernel_tensor_view_v1 *e,
               et_kernel_tensor_view_v1 *up, et_kernel_tensor_view_v1 *da,
               et_kernel_tensor_view_v1 *dg, et_kernel_tensor_view_v1 *db) {
  return et_m3t_c4_layernorm_vjp_private_v1(a, g, b, e, up, da, dg, db);
}

static double loss(void) {
  et_kernel_tensor_view_v1 inputs[4] = {xv(), gv(), bv(), ev()};
  et_kernel_tensor_view_v1 result = tv(output, sizeof(output),
                                       "f32", 3u, tensor_shape);
  et_kernel_request_v1 request =
      rq("g3c4.layer-norm.forward", 3u, tensor_shape);
  et_kernel_call_v1 call =
      kc("g3c4.layer-norm", &request, inputs, 4u, &result, 1u);
  run(runtime, &call);
  double sum = 0.0;
  for (size_t i = 0; i < 16u; ++i)
    sum += (double)output[i] * (double)u[i];
  return sum;
}

static void reject_unchanged(et_kernel_tensor_view_v1 *a,
                             et_kernel_tensor_view_v1 *g,
                             et_kernel_tensor_view_v1 *b,
                             et_kernel_tensor_view_v1 *e,
                             et_kernel_tensor_view_v1 *up,
                             et_kernel_tensor_view_v1 *da,
                             et_kernel_tensor_view_v1 *dg,
                             et_kernel_tensor_view_v1 *db, int expected) {
  unsigned char before_x[sizeof(x)], before_g[sizeof(gamma)];
  unsigned char before_b[sizeof(beta)], before_dx[sizeof(dx)];
  unsigned char before_dg[sizeof(dgamma)], before_db[sizeof(dbeta)];
  memcpy(before_x, x, sizeof(x));
  memcpy(before_g, gamma, sizeof(gamma));
  memcpy(before_b, beta, sizeof(beta));
  memcpy(before_dx, dx, sizeof(dx));
  memcpy(before_dg, dgamma, sizeof(dgamma));
  memcpy(before_db, dbeta, sizeof(dbeta));
  CHECK(vjp(a, g, b, e, up, da, dg, db) == expected);
  CHECK(memcmp(before_x, x, sizeof(x)) == 0);
  CHECK(memcmp(before_g, gamma, sizeof(gamma)) == 0);
  CHECK(memcmp(before_b, beta, sizeof(beta)) == 0);
  CHECK(memcmp(before_dx, dx, sizeof(dx)) == 0);
  CHECK(memcmp(before_dg, dgamma, sizeof(dgamma)) == 0);
  CHECK(memcmp(before_db, dbeta, sizeof(dbeta)) == 0);
}

static void positive(void) {
  for (size_t i = 0; i < 16u; ++i) {
    x[i] = (float)((int)((i * 7u) % 19u) - 9) / 32.0f;
    u[i] = (float)((int)((i * 5u) % 13u) - 6) / 32.0f + 0.0078125f;
  }
  for (size_t i = 0; i < 4u; ++i) {
    gamma[i] = 0.75f + (float)i / 8.0f;
    beta[i] = (float)((int)i - 2) / 16.0f;
  }
  memset(dx, 0x5a, sizeof(dx));
  memset(dgamma, 0xa5, sizeof(dgamma));
  memset(dbeta, 0x3c, sizeof(dbeta));
  et_kernel_tensor_view_v1 a=xv(), g=gv(), b=bv(), e=ev(), up=uv();
  et_kernel_tensor_view_v1 da=dxv(), dg=dgv(), db=dbv();
  unsigned char before_x[sizeof(x)], before_g[sizeof(gamma)];
  unsigned char before_b[sizeof(beta)], before_u[sizeof(u)];
  memcpy(before_x, x, sizeof(x));
  memcpy(before_g, gamma, sizeof(gamma));
  memcpy(before_b, beta, sizeof(beta));
  memcpy(before_u, u, sizeof(u));
  CHECK(vjp(&a,&g,&b,&e,&up,&da,&dg,&db) == ET_M3T_C4_LAYERNORM_VJP_OK);
  CHECK(memcmp(before_x,x,sizeof(x)) == 0);
  CHECK(memcmp(before_g,gamma,sizeof(gamma)) == 0);
  CHECK(memcmp(before_b,beta,sizeof(beta)) == 0);
  CHECK(memcmp(before_u,u,sizeof(u)) == 0);
  const float step = 0.00390625f;
  for (size_t i = 0; i < 16u; ++i) {
    const float original = x[i];
    x[i] = original + step; const double high = loss();
    x[i] = original - step; const double low = loss();
    x[i] = original;
    CHECK(fabs((high-low)/(2.0*step) - (double)dx[i]) < 0.002);
  }
  for (size_t i = 0; i < 4u; ++i) {
    const float original = gamma[i];
    gamma[i] = original + step; const double high = loss();
    gamma[i] = original - step; const double low = loss();
    gamma[i] = original;
    CHECK(fabs((high-low)/(2.0*step) - (double)dgamma[i]) < 0.002);
  }
  for (size_t i = 0; i < 4u; ++i) {
    const float original = beta[i];
    beta[i] = original + step; const double high = loss();
    beta[i] = original - step; const double low = loss();
    beta[i] = original;
    CHECK(fabs((high-low)/(2.0*step) - (double)dbeta[i]) < 0.002);
  }
  CHECK(isfinite(loss()));
}

static void negatives(void) {
  et_kernel_tensor_view_v1 a=xv(), g=gv(), b=bv(), e=ev(), up=uv();
  et_kernel_tensor_view_v1 da=dxv(), dg=dgv(), db=dbv();
  a.struct_size = 0u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_INVALID_ARGUMENT);
  a=xv(); a.rank=2u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_SHAPE);
  a=xv(); uint64_t wrong[3]={1u,3u,4u}; a.shape=wrong;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_SHAPE);
  a=xv(); g.byte_length=12u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_SHAPE);
  g=gv(); b.dtype="i64";
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_DTYPE);
  b=bv(); e.device="gpu";
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_DEVICE);
  e=ev(); up.layout=0u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_LAYOUT);
  up=uv(); da.offset_bytes=4u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_LAYOUT);
  da=dxv(); dg.data=x;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_OVERLAP);
  dg=dgv(); db.data=dgamma;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_OVERLAP);
  db=dbv(); da.data=u;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_OVERLAP);
  da=dxv(); epsilon=0.0f;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  epsilon=-1e-5f;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  epsilon=NAN;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  epsilon=1e-5f; x[0]=INFINITY;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  x[0]=FLT_MAX; x[1]=-FLT_MAX;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  x[0]=0.0f; x[1]=0.0f; gamma[0]=NAN;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  gamma[0]=1.0f; beta[0]=INFINITY;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  beta[0]=0.0f; u[0]=NAN;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  gamma[0]=FLT_MAX; u[0]=2.0f;
  reject_unchanged(&a,&g,&b,&e,&up,&da,&dg,&db,
                   ET_M3T_C4_LAYERNORM_VJP_VALUE);
  CHECK(vjp(NULL,&g,&b,&e,&up,&da,&dg,&db) ==
        ET_M3T_C4_LAYERNORM_VJP_INVALID_ARGUMENT);
  CHECK(vjp(&a,&g,&b,&e,&up,NULL,&dg,&db) ==
        ET_M3T_C4_LAYERNORM_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  runtime = runtime_new();
  positive();
  negatives();
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 layernorm VJP checks=%zu\n", checks);
  return 0;
}
