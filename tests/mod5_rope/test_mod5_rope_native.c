#include "eshkol_transformer/mod5_rope_abi.h"
#include "mod5_rope_eshkol_bridge.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 RoPE native test requires x86-64"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Trap any heap allocation during provider/direct/bridge dispatch. */
static int dispatching = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!dispatching);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!dispatching);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!dispatching);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!dispatching);
  __real_free(ptr);
}

static const et_kernel_provider_v1 *resolve_rope(void *context,
                                                 const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_rope_kernel_provider_v1()
             : NULL;
}

static et_kernel_runtime *discover_rope(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_rope, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t rank, const uint64_t *shape) {
  size_t bytes = strcmp(dtype, "i64") == 0 ? sizeof(int64_t) : sizeof(float);
  for (size_t d = 0; d < rank; ++d)
    bytes *= (size_t)shape[d];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static int32_t dispatch_rope_k1(const et_kernel_runtime *runtime, int backward,
                                size_t n, size_t h, size_t t, size_t dh,
                                void *in_buf, void *positions, float *theta,
                                float *scale, void *out_buf,
                                et_kernel_error *error) {
  const uint64_t shape[] = {n, h, t, dh};
  const uint64_t pos_shape[] = {n, t};
  const et_kernel_tensor_view_v1 in[] = {
      make_view(in_buf, "f32", 4u, shape),
      make_view(positions, "i64", 2u, pos_shape),
      make_view(theta, "f32", 0u, NULL), make_view(scale, "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 out[] = {make_view(out_buf, "f32", 4u, shape)};
  const et_kernel_request_v1 request = {
      sizeof(request), backward ? "rope.backward" : "rope.forward",
      "f32",           "cpu",
      4u,              shape,
      1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.rope", &request, 4u, stride, 4u * stride, in,
      1u,           stride,        stride,   out};
  return et_kernel_runtime_dispatch(runtime, &call, error);
}

static void fill_deterministic_f32(float *buf, size_t count, int salt) {
  for (size_t i = 0; i < count; ++i) {
    const int v = (int)((29u * (unsigned)salt + 17u * (unsigned)i +
                         (unsigned)(i / 5u)) %
                        41u) -
                  20;
    buf[i] = (float)v * 0.0625f;
  }
}

static void fill_positions(int64_t *pos, size_t n, size_t t, int mode) {
  for (size_t b = 0; b < n; ++b) {
    for (size_t i = 0; i < t; ++i) {
      if (mode == 0) {
        pos[b * t + i] = (int64_t)(i + b * 3u);
      } else if (mode == 1) {
        pos[b * t + i] = (int64_t)(i * 7u + b * 19u);
      } else {
        /* Includes 0 and near-max boundary position. */
        pos[b * t + i] = (i == t - 1u && b == n - 1u)
                             ? ET_MOD5_ROPE_MAX_POSITION
                             : (int64_t)(i * 131u + b * 17u);
      }
    }
  }
}

typedef struct shape_case {
  size_t n, h, t, dh;
  double theta;
  double scale;
  int pos_mode;
} shape_case;

static const shape_case k_cases[] = {
    {1, 1, 1, 2, 10000.0, 1.0, 0},
    {1, 2, 4, 4, 10000.0, 1.0, 0},
    {2, 4, 8, 6, 10000.0, 1.0, 1},
    {1, 4, 8, 8, 500000.0, 2.0, 1},
    {2, 2, 16, 10, 10000.0, 0.5, 0},
    {1, 3, 16, 12, 500000.0, 1.0, 2},
    {2, 4, 32, 14, 10000.0, 2.0, 1},
    {2, 4, 64, 16, 500000.0, 2.0, 2},
    {2, 4, 64, 16, 10000.0, 1.0, 0},
};

static double dot_f64(const float *a, const float *b, size_t count) {
  double acc = 0.0;
  for (size_t i = 0; i < count; ++i)
    acc += (double)a[i] * (double)b[i];
  return acc;
}

static void check_orthogonality_inverse_and_fd(const et_kernel_runtime *runtime,
                                               const shape_case *sc,
                                               double *max_ortho_err,
                                               double *max_inv_err,
                                               double *max_fd_err,
                                               size_t *fd_checks) {
  const size_t total = sc->n * sc->h * sc->t * sc->dh;
  const size_t pos_count = sc->n * sc->t;
  float *x = (float *)calloc(total, sizeof(float));
  float *y = (float *)calloc(total, sizeof(float));
  float *y_direct = (float *)calloc(total, sizeof(float));
  float *y_repeat = (float *)calloc(total, sizeof(float));
  float *dy = (float *)calloc(total, sizeof(float));
  float *dx = (float *)calloc(total, sizeof(float));
  float *dx_direct = (float *)calloc(total, sizeof(float));
  float *x_rec = (float *)calloc(total, sizeof(float));
  float *x_rec2 = (float *)calloc(total, sizeof(float));
  float *x_tmp = (float *)calloc(total, sizeof(float));
  float *y_plus = (float *)calloc(total, sizeof(float));
  float *y_minus = (float *)calloc(total, sizeof(float));
  int64_t *positions = (int64_t *)calloc(pos_count, sizeof(int64_t));
  CHECK(x && y && y_direct && y_repeat && dy && dx && dx_direct && x_rec &&
        x_rec2 && x_tmp && y_plus && y_minus && positions);

  fill_deterministic_f32(x, total, (int)(sc->dh + sc->t));
  fill_deterministic_f32(dy, total, (int)(sc->dh + sc->t + 101));
  fill_positions(positions, sc->n, sc->t, sc->pos_mode);

  float theta_f = (float)sc->theta;
  float scale_f = (float)sc->scale;
  et_kernel_error error;

  dispatching = 1;
  CHECK(dispatch_rope_k1(runtime, 0, sc->n, sc->h, sc->t, sc->dh, x, positions,
                         &theta_f, &scale_f, y, &error) == 0);
  CHECK(dispatch_rope_k1(runtime, 0, sc->n, sc->h, sc->t, sc->dh, x, positions,
                         &theta_f, &scale_f, y_repeat, &error) == 0);
  CHECK(et_mod5_rope_forward_f32(sc->n, sc->h, sc->t, sc->dh, x, positions,
                                 sc->theta, sc->scale, y_direct, &error) == 0);

  CHECK(dispatch_rope_k1(runtime, 1, sc->n, sc->h, sc->t, sc->dh, dy, positions,
                         &theta_f, &scale_f, dx, &error) == 0);
  CHECK(et_mod5_rope_backward_f32(sc->n, sc->h, sc->t, sc->dh, dy, positions,
                                  sc->theta, sc->scale, dx_direct,
                                  &error) == 0);

  /* Inverse: backward(forward(x)) == x and forward(backward(x)) == x */
  CHECK(dispatch_rope_k1(runtime, 1, sc->n, sc->h, sc->t, sc->dh, y, positions,
                         &theta_f, &scale_f, x_rec, &error) == 0);
  CHECK(dispatch_rope_k1(runtime, 1, sc->n, sc->h, sc->t, sc->dh, x, positions,
                         &theta_f, &scale_f, x_tmp, &error) == 0);
  CHECK(dispatch_rope_k1(runtime, 0, sc->n, sc->h, sc->t, sc->dh, x_tmp,
                         positions, &theta_f, &scale_f, x_rec2, &error) == 0);
  dispatching = 0;

  /* Bitwise determinism & K1-vs-direct parity. */
  CHECK(memcmp(y, y_repeat, total * sizeof(float)) == 0);
  CHECK(memcmp(y, y_direct, total * sizeof(float)) == 0);
  CHECK(memcmp(dx, dx_direct, total * sizeof(float)) == 0);

  /* Exact orthogonality: ||y[n,h,t,:]|| == ||x[n,h,t,:]||. */
  const size_t slices = sc->n * sc->h * sc->t;
  for (size_t s = 0; s < slices; ++s) {
    double nx = 0.0, ny = 0.0;
    for (size_t d = 0; d < sc->dh; ++d) {
      const double xv = (double)x[s * sc->dh + d];
      const double yv = (double)y[s * sc->dh + d];
      nx += xv * xv;
      ny += yv * yv;
    }
    const double err = fabs(sqrt(ny) - sqrt(nx));
    if (err > *max_ortho_err)
      *max_ortho_err = err;
    CHECK(err <= 2e-6 + 2e-6 * sqrt(nx));
  }

  /* Exact inverse property. */
  for (size_t i = 0; i < total; ++i) {
    const double e1 = fabs((double)x_rec[i] - (double)x[i]);
    const double e2 = fabs((double)x_rec2[i] - (double)x[i]);
    if (e1 > *max_inv_err)
      *max_inv_err = e1;
    if (e2 > *max_inv_err)
      *max_inv_err = e2;
    CHECK(e1 <= 1e-6);
    CHECK(e2 <= 1e-6);
  }

  /* Central finite-difference check on L(x) = sum_i dy[i] * y(x)[i]. */
  const float eps = 0.0078125f; /* exact power of two: 2^-7 */
  const size_t stride = total <= 128u ? 1u : (total / 48u);
  memcpy(x_tmp, x, total * sizeof(float));
  for (size_t idx = 0; idx < total; idx += stride) {
    const float orig = x_tmp[idx];
    x_tmp[idx] = orig + eps;
    dispatching = 1;
    CHECK(et_mod5_rope_forward_f32(sc->n, sc->h, sc->t, sc->dh, x_tmp,
                                   positions, sc->theta, sc->scale, y_plus,
                                   &error) == 0);
    x_tmp[idx] = orig - eps;
    CHECK(et_mod5_rope_forward_f32(sc->n, sc->h, sc->t, sc->dh, x_tmp,
                                   positions, sc->theta, sc->scale, y_minus,
                                   &error) == 0);
    dispatching = 0;
    x_tmp[idx] = orig;
    const double lp = dot_f64(dy, y_plus, total);
    const double lm = dot_f64(dy, y_minus, total);
    const double num = (lp - lm) / (2.0 * (double)eps);
    const double ana = (double)dx[idx];
    const double diff = fabs(ana - num);
    if (diff > *max_fd_err)
      *max_fd_err = diff;
    CHECK(diff <= 5e-5 + 5e-4 * fabs(num));
    (*fd_checks)++;
  }

  free(x);
  free(y);
  free(y_direct);
  free(y_repeat);
  free(dy);
  free(dx);
  free(dx_direct);
  free(x_rec);
  free(x_rec2);
  free(x_tmp);
  free(y_plus);
  free(y_minus);
  free(positions);
}

static void check_negative_suite(const et_kernel_runtime *runtime) {
  et_kernel_runtime *baseline = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);

  /* Oversized aligned backing buffers for misalignment, wide bad shapes, & overlap. */
  unsigned char raw_x[1024] __attribute__((aligned(16)));
  unsigned char raw_y[1024] __attribute__((aligned(16)));
  unsigned char raw_pos[1024] __attribute__((aligned(16)));
  unsigned char sentinel[1024];
  memset(raw_x, 0, sizeof(raw_x));
  memset(raw_y, 0xa5, sizeof(raw_y));
  memcpy(sentinel, raw_y, sizeof(sentinel));
  memset(raw_pos, 0, sizeof(raw_pos));

  float *x = (float *)raw_x;
  float *y = (float *)raw_y;
  int64_t *pos = (int64_t *)raw_pos;
  for (size_t i = 0; i < 32u; ++i)
    x[i] = 0.25f * (float)(i + 1u);
  for (size_t i = 0; i < 8u; ++i)
    pos[i] = (int64_t)i;

  float theta_ok = 10000.0f;
  float scale_ok = 1.0f;

  dispatching = 1;

  /* 1. Baseline runtime has no verified kernel.rope capability. */
  CHECK(dispatch_rope_k1(baseline, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 2. Odd Dh rejected by both K1 and direct helpers. */
  const size_t odd_dhs[] = {1, 3, 5, 7, 15, 17};
  for (size_t i = 0; i < ARRAY_COUNT(odd_dhs); ++i) {
    CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, odd_dhs[i], x, pos, &theta_ok,
                           &scale_ok, y,
                           &error) == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(et_mod5_rope_forward_f32(1, 1, 2, odd_dhs[i], x, pos, 10000.0, 1.0, y,
                                   &error) == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(et_mod5_rope_backward_f32(1, 1, 2, odd_dhs[i], x, pos, 10000.0, 1.0,
                                    y, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  }
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 3. Out-of-range N, H, T, Dh. */
  const size_t bad_shapes[][4] = {
      {0, 1, 1, 2}, {3, 1, 1, 2}, {1, 0, 1, 2},  {1, 5, 1, 2},
      {1, 1, 0, 2}, {1, 1, 65, 2}, {1, 1, 1, 0}, {1, 1, 1, 18}};
  for (size_t i = 0; i < ARRAY_COUNT(bad_shapes); ++i) {
    CHECK(et_mod5_rope_forward_f32(bad_shapes[i][0], bad_shapes[i][1],
                                   bad_shapes[i][2], bad_shapes[i][3], x, pos,
                                   10000.0, 1.0, y,
                                   &error) == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(dispatch_rope_k1(runtime, 0, bad_shapes[i][0], bad_shapes[i][1],
                           bad_shapes[i][2], bad_shapes[i][3], x, pos,
                           &theta_ok, &scale_ok, y,
                           &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  }
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 4. NULL pointers. */
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, NULL, pos, 10000.0, 1.0, y,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, NULL, 10000.0, 1.0, y,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, NULL,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 5. Misaligned pointers via void* (zero UBSan alignment violation!). */
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, raw_x + 1, pos, 10000.0, 1.0, y,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, raw_pos + 2, 10000.0, 1.0, y,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, raw_y + 1,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, raw_x + 1, pos, &theta_ok,
                         &scale_ok, y,
                         &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, raw_pos + 4, &theta_ok,
                         &scale_ok, y,
                         &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         raw_y + 2,
                         &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 6. Aliasing (exact & partial overlap between input/output/positions). */
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, x, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, x + 2,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, pos,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         x, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         x + 4, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 7. Negative and out-of-range positions. */
  pos[1] = -1;
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  pos[1] = ET_MOD5_ROPE_MAX_POSITION + 1;
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  pos[1] = 1;
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 8. Invalid theta & scale (<= 1.0, <= 0.0, NaN, Inf, overflow scale). */
  const double bad_thetas[] = {1.0, 0.5, 0.0, -10000.0, NAN, INFINITY,
                               -INFINITY};
  for (size_t i = 0; i < ARRAY_COUNT(bad_thetas); ++i) {
    float tf = (float)bad_thetas[i];
    CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, bad_thetas[i], 1.0, y,
                                   &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &tf, &scale_ok, y,
                           &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }
  const double bad_scales[] = {0.0, -0.0, -1.0, NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < ARRAY_COUNT(bad_scales); ++i) {
    float sf = (float)bad_scales[i];
    CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, bad_scales[i],
                                   y,
                                   &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &sf, y,
                           &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }
  /* Tiny scale causing pos/scale binary64 overflow. */
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1e-320, y,
                                 &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 9. Non-finite inputs and binary32 overflow during rotation. */
  const float bad_vals[] = {NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < ARRAY_COUNT(bad_vals); ++i) {
    x[5] = bad_vals[i];
    CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y,
                                   &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(dispatch_rope_k1(runtime, 1, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                           y, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }
  /* Rotation overflow: x0 = 3e38, x1 = -3e38 at pos=1 -> x0*cos(1)-x1*sin(1) > FLT_MAX */
  x[4] = 3.0e38f;
  x[5] = -3.0e38f;
  errno = EAGAIN;
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(errno == EAGAIN);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(errno == EAGAIN);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);
  x[4] = 1.25f;
  x[5] = 1.5f;

  /* 10. Non-standard FP environments (rounding mode, MXCSR FTZ, MXCSR DAZ). */
  fenv_t saved_env;
  CHECK(fegetenv(&saved_env) == 0);
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(fesetenv(&saved_env) == 0);

  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_mod5_rope_forward_f32(1, 1, 2, 4, x, pos, 10000.0, 1.0, y, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(saved_mxcsr | 0x0040u); /* DAZ */
  CHECK(dispatch_rope_k1(runtime, 0, 1, 1, 2, 4, x, pos, &theta_ok, &scale_ok,
                         y, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(saved_mxcsr);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);

  /* 11. Direct provider validate_call negative descriptor checks. */
  const et_kernel_provider_v1 *provider = et_mod5_rope_kernel_provider_v1();
  CHECK(provider->validate_call(NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  dispatching = 0;
  et_kernel_runtime_destroy(baseline);
}

static void check_bridge(void) {
  et_kernel_runtime *rt = NULL;
  CHECK(et_mod5_rope_aot_open(NULL) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_rope_aot_open(&rt) == 0 && rt != NULL);

  unsigned char raw_x[128] __attribute__((aligned(16)));
  unsigned char raw_y[128] __attribute__((aligned(16)));
  unsigned char raw_dx[128] __attribute__((aligned(16)));
  unsigned char raw_pos[64] __attribute__((aligned(16)));
  unsigned char sentinel[128];
  memset(raw_y, 0x5a, sizeof(raw_y));
  memcpy(sentinel, raw_y, sizeof(sentinel));

  for (int64_t i = 0; i < 16; ++i)
    et_mod5_rope_aot_store_f32(raw_x, i, 0.125 * (double)(i - 7));
  for (int64_t t = 0; t < 4; ++t)
    et_mod5_rope_aot_store_i64(raw_pos, t, t * 3);
  CHECK(et_mod5_rope_aot_load_i64(raw_pos, 3) == 9);

  dispatching = 1;
  CHECK(et_mod5_rope_aot_forward(rt, 1, 1, 4, 4, raw_x, raw_pos, 10000.0, 1.0,
                                 raw_y) == 0);
  CHECK(et_mod5_rope_aot_backward(rt, 1, 1, 4, 4, raw_y, raw_pos, 10000.0, 1.0,
                                  raw_dx) == 0);
  for (int64_t i = 0; i < 16; ++i) {
    CHECK(fabs(et_mod5_rope_aot_load_f32(raw_dx, i) -
               et_mod5_rope_aot_load_f32(raw_x, i)) <= 1e-6);
  }

  /* Bridge negative checks: odd Dh, misaligned pointer, non-finite store. */
  memcpy(raw_y, sentinel, sizeof(raw_y));
  CHECK(et_mod5_rope_aot_forward(rt, 1, 1, 4, 3, raw_x, raw_pos, 10000.0, 1.0,
                                 raw_y) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mod5_rope_aot_forward(rt, 1, 1, 4, 4, raw_x + 1, raw_pos, 10000.0,
                                 1.0, raw_y) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  et_mod5_rope_aot_store_f32(raw_x, 0, DBL_MAX);
  CHECK(et_mod5_rope_aot_forward(rt, 1, 1, 4, 4, raw_x, raw_pos, 10000.0, 1.0,
                                 raw_y) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(raw_y, sentinel, sizeof(raw_y)) == 0);
  dispatching = 0;

  et_mod5_rope_aot_close(rt);
}

static void emit_oracle(const et_kernel_runtime *runtime) {
  for (size_t c = 0; c < ARRAY_COUNT(k_cases); ++c) {
    const shape_case *sc = &k_cases[c];
    const size_t total = sc->n * sc->h * sc->t * sc->dh;
    const size_t pos_count = sc->n * sc->t;
    float *x = (float *)calloc(total, sizeof(float));
    float *dy = (float *)calloc(total, sizeof(float));
    float *y = (float *)calloc(total, sizeof(float));
    float *dx = (float *)calloc(total, sizeof(float));
    int64_t *positions = (int64_t *)calloc(pos_count, sizeof(int64_t));
    CHECK(x && dy && y && dx && positions);

    fill_deterministic_f32(x, total, (int)(sc->dh + sc->t));
    fill_deterministic_f32(dy, total, (int)(sc->dh + sc->t + 101));
    fill_positions(positions, sc->n, sc->t, sc->pos_mode);

    float theta_f = (float)sc->theta;
    float scale_f = (float)sc->scale;
    et_kernel_error error;
    CHECK(dispatch_rope_k1(runtime, 0, sc->n, sc->h, sc->t, sc->dh, x,
                           positions, &theta_f, &scale_f, y, &error) == 0);
    CHECK(dispatch_rope_k1(runtime, 1, sc->n, sc->h, sc->t, sc->dh, dy,
                           positions, &theta_f, &scale_f, dx, &error) == 0);

    printf("CASE %zu %zu %zu %zu %a %a\n", sc->n, sc->h, sc->t, sc->dh,
           (double)theta_f, (double)scale_f);
    for (size_t i = 0; i < pos_count; ++i)
      printf("%s%lld", i ? " " : "", (long long)positions[i]);
    putchar('\n');
    for (size_t i = 0; i < total; ++i)
      printf("%s%a", i ? " " : "", (double)x[i]);
    putchar('\n');
    for (size_t i = 0; i < total; ++i)
      printf("%s%a", i ? " " : "", (double)dy[i]);
    putchar('\n');
    for (size_t i = 0; i < total; ++i)
      printf("%s%a", i ? " " : "", (double)y[i]);
    putchar('\n');
    for (size_t i = 0; i < total; ++i)
      printf("%s%a", i ? " " : "", (double)dx[i]);
    putchar('\n');

    free(x);
    free(dy);
    free(y);
    free(dx);
    free(positions);
  }
}

int main(int argc, char **argv) {
  et_kernel_runtime *runtime = discover_rope();
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    emit_oracle(runtime);
    et_kernel_runtime_destroy(runtime);
    return 0;
  }

  /* Verify K1 capability metadata. */
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, "kernel.rope");
  CHECK(cap != NULL);
  CHECK(cap->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(cap->deterministic == 1u);
  CHECK(cap->operation_count == 2u);
  CHECK(cap->shape_range_count == 8u);

  double max_ortho_err = 0.0;
  double max_inv_err = 0.0;
  double max_fd_err = 0.0;
  size_t fd_checks = 0u;
  for (size_t i = 0; i < ARRAY_COUNT(k_cases); ++i) {
    check_orthogonality_inverse_and_fd(runtime, &k_cases[i], &max_ortho_err,
                                       &max_inv_err, &max_fd_err, &fd_checks);
  }
  check_negative_suite(runtime);
  check_bridge();

  et_kernel_runtime_destroy(runtime);
  printf("MOD5 RoPE native suite: PASS (cases=%zu fd_checks=%zu "
         "max_ortho_err=%.3e max_inv_err=%.3e max_fd_err=%.3e)\n",
         ARRAY_COUNT(k_cases), fd_checks, max_ortho_err, max_inv_err,
         max_fd_err);
  return 0;
}
