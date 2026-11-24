#include "eshkol_transformer/mod5_mla_abi.h"
#include "mod5_mla_eshkol_bridge.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 MLA native test requires x86-64"
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

/* Trap any heap allocation during provider/direct/cache/bridge dispatch. */
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

static const et_kernel_provider_v1 *resolve_mla(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_mla_kernel_provider_v1()
             : NULL;
}

static et_kernel_runtime *discover_mla(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mla, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static et_kernel_tensor_view_v1 make_f32_view(void *data, size_t rank,
                                              const uint64_t *shape) {
  size_t bytes = sizeof(float);
  for (size_t d = 0; d < rank; ++d)
    bytes *= (size_t)shape[d];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, "f32", "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static int32_t dispatch_mla_k1(const et_kernel_runtime *runtime, int backward,
                               const et_mod5_mla_dims *d, void *x, void *w_q,
                               void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                               void *dy, void *out_or_dx, void *dw_q,
                               void *dw_dkv, void *dw_uk, void *dw_uv,
                               void *dw_o, et_kernel_error *error) {
  const uint64_t req_shape[] = {d->batch, d->tokens,   d->model_dim,
                                d->heads, d->head_dim, d->latent_dim};
  const uint64_t hd = (uint64_t)(d->heads * d->head_dim);
  const uint64_t x_shape[] = {d->batch, d->tokens, d->model_dim};
  const uint64_t wq_shape[] = {hd, d->model_dim};
  const uint64_t wdkv_shape[] = {d->latent_dim, d->model_dim};
  const uint64_t wuk_shape[] = {hd, d->latent_dim};
  const uint64_t wo_shape[] = {d->model_dim, hd};

  const et_kernel_tensor_view_v1 in[] = {
      make_f32_view(x, 3u, x_shape),     make_f32_view(w_q, 2u, wq_shape),
      make_f32_view(w_dkv, 2u, wdkv_shape), make_f32_view(w_uk, 2u, wuk_shape),
      make_f32_view(w_uv, 2u, wuk_shape),   make_f32_view(w_o, 2u, wo_shape),
      make_f32_view(dy, 3u, x_shape)};
  et_kernel_tensor_view_v1 out[] = {
      make_f32_view(out_or_dx, 3u, x_shape),
      make_f32_view(dw_q, 2u, wq_shape),
      make_f32_view(dw_dkv, 2u, wdkv_shape),
      make_f32_view(dw_uk, 2u, wuk_shape),
      make_f32_view(dw_uv, 2u, wuk_shape),
      make_f32_view(dw_o, 2u, wo_shape)};

  const size_t ni = backward ? 7u : 6u;
  const size_t no = backward ? 6u : 1u;
  const size_t stride = sizeof(*in);
  const et_kernel_request_v1 request = {
      sizeof(request), backward ? "mla.backward" : "mla.forward",
      "f32",           "cpu",
      6u,              req_shape,
      1u,              {0}};
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.mla", &request, ni, stride, ni * stride, in,
      no,           stride,       no * stride, out};
  return et_kernel_runtime_dispatch(runtime, &call, error);
}

static void fill_deterministic_f32(float *buf, size_t count, int salt,
                                   float scale) {
  for (size_t i = 0; i < count; ++i) {
    const int v = (int)((31u * (unsigned)salt + 17u * (unsigned)i +
                         7u * (unsigned)(i / 3u)) %
                        33u) -
                  16;
    const float jitter = 1.0f + 0.013f * (float)(i % 5u);
    buf[i] = (float)v * scale * jitter;
  }
}

static void fill_sentinel(unsigned char *buf, size_t bytes,
                          unsigned char seed) {
  for (size_t i = 0; i < bytes; ++i)
    buf[i] = (unsigned char)((seed + i * 13u) & 0xffu);
}

static double dot_f64(const float *a, const float *b, size_t count) {
  double acc = 0.0;
  for (size_t i = 0; i < count; ++i)
    acc += (double)a[i] * (double)b[i];
  return acc;
}

typedef struct mla_case {
  et_mod5_mla_dims d;
  size_t prefill_tokens;
} mla_case;

static const mla_case k_cases[] = {
    {{1, 1, 4, 1, 2, 1}, 1},     /* min extents, 4x compression */
    {{1, 4, 8, 2, 4, 4}, 2},     /* 4x compression */
    {{1, 8, 16, 4, 4, 4}, 3},    /* 8x compression */
    {{1, 16, 16, 4, 8, 8}, 6},   /* prompt example: 8x compression */
    {{2, 8, 12, 3, 4, 6}, 3},    /* batch=2, odd H=3, 4x compression */
    {{2, 12, 24, 4, 6, 8}, 5},   /* 6x compression */
    {{1, 16, 16, 4, 16, 8}, 7},  /* 16x compression */
    {{2, 32, 32, 4, 16, 16}, 10} /* max extents, 8x compression */
};

typedef struct mla_buffers {
  size_t x_n, wq_n, wdkv_n, wuk_n, wo_n;
  float *x, *w_q, *w_dkv, *w_uk, *w_uv, *w_o, *dy;
  float *o, *o_direct, *o_repeat, *o_cached, *o_full_prefill;
  float *dx, *dw_q, *dw_dkv, *dw_uk, *dw_uv, *dw_o;
  float *dx_d, *dw_q_d, *dw_dkv_d, *dw_uk_d, *dw_uv_d, *dw_o_d;
  float *o_tmp;
  float *c_kv;
} mla_buffers;

static void alloc_buffers(const et_mod5_mla_dims *d, mla_buffers *b) {
  const size_t hd = d->heads * d->head_dim;
  b->x_n = d->batch * d->tokens * d->model_dim;
  b->wq_n = hd * d->model_dim;
  b->wdkv_n = d->latent_dim * d->model_dim;
  b->wuk_n = hd * d->latent_dim;
  b->wo_n = d->model_dim * hd;
  const size_t c_n = d->batch * d->tokens * d->latent_dim;

  b->x = (float *)calloc(b->x_n, sizeof(float));
  b->w_q = (float *)calloc(b->wq_n, sizeof(float));
  b->w_dkv = (float *)calloc(b->wdkv_n, sizeof(float));
  b->w_uk = (float *)calloc(b->wuk_n, sizeof(float));
  b->w_uv = (float *)calloc(b->wuk_n, sizeof(float));
  b->w_o = (float *)calloc(b->wo_n, sizeof(float));
  b->dy = (float *)calloc(b->x_n, sizeof(float));
  b->o = (float *)calloc(b->x_n, sizeof(float));
  b->o_direct = (float *)calloc(b->x_n, sizeof(float));
  b->o_repeat = (float *)calloc(b->x_n, sizeof(float));
  b->o_cached = (float *)calloc(b->x_n, sizeof(float));
  b->o_full_prefill = (float *)calloc(b->x_n, sizeof(float));
  b->dx = (float *)calloc(b->x_n, sizeof(float));
  b->dw_q = (float *)calloc(b->wq_n, sizeof(float));
  b->dw_dkv = (float *)calloc(b->wdkv_n, sizeof(float));
  b->dw_uk = (float *)calloc(b->wuk_n, sizeof(float));
  b->dw_uv = (float *)calloc(b->wuk_n, sizeof(float));
  b->dw_o = (float *)calloc(b->wo_n, sizeof(float));
  b->dx_d = (float *)calloc(b->x_n, sizeof(float));
  b->dw_q_d = (float *)calloc(b->wq_n, sizeof(float));
  b->dw_dkv_d = (float *)calloc(b->wdkv_n, sizeof(float));
  b->dw_uk_d = (float *)calloc(b->wuk_n, sizeof(float));
  b->dw_uv_d = (float *)calloc(b->wuk_n, sizeof(float));
  b->dw_o_d = (float *)calloc(b->wo_n, sizeof(float));
  b->o_tmp = (float *)calloc(b->x_n, sizeof(float));
  b->c_kv = (float *)calloc(c_n, sizeof(float));
  CHECK(b->x && b->w_q && b->w_dkv && b->w_uk && b->w_uv && b->w_o && b->dy &&
        b->o && b->o_direct && b->o_repeat && b->o_cached &&
        b->o_full_prefill && b->dx && b->dw_q && b->dw_dkv && b->dw_uk &&
        b->dw_uv && b->dw_o && b->dx_d && b->dw_q_d && b->dw_dkv_d &&
        b->dw_uk_d && b->dw_uv_d && b->dw_o_d && b->o_tmp && b->c_kv);

  const int salt = (int)(d->batch * 3u + d->tokens * 5u + d->model_dim * 7u +
                         d->heads * 11u + d->head_dim * 13u +
                         d->latent_dim * 17u);
  fill_deterministic_f32(b->x, b->x_n, salt + 1, 0.03125f);
  fill_deterministic_f32(b->w_q, b->wq_n, salt + 2, 0.015625f);
  fill_deterministic_f32(b->w_dkv, b->wdkv_n, salt + 3, 0.015625f);
  fill_deterministic_f32(b->w_uk, b->wuk_n, salt + 4, 0.015625f);
  fill_deterministic_f32(b->w_uv, b->wuk_n, salt + 5, 0.015625f);
  fill_deterministic_f32(b->w_o, b->wo_n, salt + 6, 0.015625f);
  fill_deterministic_f32(b->dy, b->x_n, salt + 7, 0.03125f);
}

static void free_buffers(mla_buffers *b) {
  free(b->x);
  free(b->w_q);
  free(b->w_dkv);
  free(b->w_uk);
  free(b->w_uv);
  free(b->w_o);
  free(b->dy);
  free(b->o);
  free(b->o_direct);
  free(b->o_repeat);
  free(b->o_cached);
  free(b->o_full_prefill);
  free(b->dx);
  free(b->dw_q);
  free(b->dw_dkv);
  free(b->dw_uk);
  free(b->dw_uv);
  free(b->dw_o);
  free(b->dx_d);
  free(b->dw_q_d);
  free(b->dw_dkv_d);
  free(b->dw_uk_d);
  free(b->dw_uv_d);
  free(b->dw_o_d);
  free(b->o_tmp);
  free(b->c_kv);
}

static double eval_scalar_loss(const et_mod5_mla_dims *d, mla_buffers *b) {
  et_kernel_error error;
  dispatching = 1;
  CHECK(et_mod5_mla_forward_f32(d, b->x, b->w_q, b->w_dkv, b->w_uk, b->w_uv,
                                b->w_o, b->o_tmp, &error) == 0);
  dispatching = 0;
  return dot_f64(b->o_tmp, b->dy, b->x_n);
}

static void check_tensor_fd(const et_mod5_mla_dims *d, mla_buffers *b,
                            float *param, const float *grad, size_t count,
                            size_t max_samples, double *max_fd_err,
                            size_t *fd_checks) {
  const size_t step_stride =
      count <= max_samples ? 1u : (count + max_samples - 1u) / max_samples;
  const float h1 = 1.0f / 32.0f;
  const float h2 = 1.0f / 64.0f;
  for (size_t idx = 0; idx < count; idx += step_stride) {
    const float orig = param[idx];
    param[idx] = orig + h1;
    const double p1 = eval_scalar_loss(d, b);
    param[idx] = orig - h1;
    const double m1 = eval_scalar_loss(d, b);
    const double d1 = (p1 - m1) / (2.0 * (double)h1);

    param[idx] = orig + h2;
    const double p2 = eval_scalar_loss(d, b);
    param[idx] = orig - h2;
    const double m2 = eval_scalar_loss(d, b);
    const double d2 = (p2 - m2) / (2.0 * (double)h2);
    param[idx] = orig;

    /* Richardson extrapolation cancels O(h^2) truncation error. */
    const double numerical = (4.0 * d2 - d1) / 3.0;
    const double analytic = (double)grad[idx];
    const double err = fabs(analytic - numerical);
    if (err > *max_fd_err)
      *max_fd_err = err;
    CHECK(err <= 2e-5 + 1e-3 * fabs(numerical));
    ++(*fd_checks);
  }
}

static void run_cached_prefill_decode(const et_mod5_mla_dims *d,
                                      size_t prefill_tokens,
                                      const mla_buffers *b, float *o_out) {
  const size_t N = d->batch;
  const size_t T = d->tokens;
  const size_t D = d->model_dim;
  const size_t c_bytes =
      et_mod5_mla_latent_kv_bytes(d->batch, d->tokens, d->latent_dim);
  float *x_prefill = (float *)calloc(N * prefill_tokens * D, sizeof(float));
  float *o_prefill = (float *)calloc(N * prefill_tokens * D, sizeof(float));
  float *x_step = (float *)calloc(N * D, sizeof(float));
  float *o_step = (float *)calloc(N * D, sizeof(float));
  CHECK(x_prefill && o_prefill && x_step && o_step);

  for (size_t n = 0; n < N; ++n) {
    memcpy(x_prefill + n * prefill_tokens * D, b->x + n * T * D,
           prefill_tokens * D * sizeof(float));
  }

  et_mod5_mla_kv_cache cache;
  et_kernel_error error;

  dispatching = 1;
  CHECK(et_mod5_mla_cache_init(&cache, d, b->c_kv, c_bytes, &error) == 0);
  CHECK(cache.seq_len == 0u);
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, prefill_tokens, x_prefill, b->w_q,
                                      b->w_dkv, b->w_uk, b->w_uv, b->w_o,
                                      o_prefill, &error) == 0);
  CHECK(cache.seq_len == prefill_tokens);
  dispatching = 0;

  for (size_t n = 0; n < N; ++n) {
    memcpy(o_out + n * T * D, o_prefill + n * prefill_tokens * D,
           prefill_tokens * D * sizeof(float));
  }

  for (size_t t = prefill_tokens; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      memcpy(x_step + n * D, b->x + (n * T + t) * D, D * sizeof(float));
    }
    dispatching = 1;
    CHECK(et_mod5_mla_cache_step_f32(&cache, x_step, b->w_q, b->w_dkv, b->w_uk,
                                     b->w_uv, b->w_o, o_step, &error) == 0);
    CHECK(cache.seq_len == t + 1u);
    dispatching = 0;
    for (size_t n = 0; n < N; ++n) {
      memcpy(o_out + (n * T + t) * D, o_step + n * D, D * sizeof(float));
    }
  }

  free(x_prefill);
  free(o_prefill);
  free(x_step);
  free(o_step);
}

static void check_case(const et_kernel_runtime *runtime, const mla_case *mc,
                       double *max_fd_err, size_t *fd_checks,
                       double *max_cache_diff) {
  const et_mod5_mla_dims *d = &mc->d;
  mla_buffers b;
  alloc_buffers(d, &b);

  /* Verify exact KV-cache memory telemetry & compression ratio. */
  const size_t latent_bytes =
      et_mod5_mla_latent_kv_bytes(d->batch, d->tokens, d->latent_dim);
  const size_t full_bytes =
      et_mod5_mla_full_kv_bytes(d->batch, d->tokens, d->heads, d->head_dim);
  const double ratio =
      et_mod5_mla_kv_compression_ratio(d->heads, d->head_dim, d->latent_dim);
  CHECK(latent_bytes == d->batch * d->tokens * d->latent_dim * sizeof(float));
  CHECK(full_bytes ==
        d->batch * d->tokens * 2u * d->heads * d->head_dim * sizeof(float));
  CHECK(full_bytes > latent_bytes);
  CHECK(fabs(ratio - (double)full_bytes / (double)latent_bytes) <= 1e-12);
  if (d->heads == 4u && d->head_dim == 8u && d->latent_dim == 8u) {
    CHECK(fabs(ratio - 8.0) <= 1e-12);
  }

  et_kernel_error error;
  dispatching = 1;
  CHECK(dispatch_mla_k1(runtime, 0, d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, NULL, b.o, NULL, NULL, NULL, NULL, NULL,
                        &error) == 0);
  CHECK(dispatch_mla_k1(runtime, 0, d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, NULL, b.o_repeat, NULL, NULL, NULL, NULL, NULL,
                        &error) == 0);
  CHECK(et_mod5_mla_forward_f32(d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o_direct, &error) == 0);

  CHECK(dispatch_mla_k1(runtime, 1, d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, b.dy, b.dx, b.dw_q, b.dw_dkv, b.dw_uk, b.dw_uv,
                        b.dw_o, &error) == 0);
  CHECK(et_mod5_mla_backward_f32(d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                 b.dy, b.dx_d, b.dw_q_d, b.dw_dkv_d, b.dw_uk_d,
                                 b.dw_uv_d, b.dw_o_d, &error) == 0);
  dispatching = 0;

  /* Bitwise determinism & K1-vs-direct parity across forward + all 6 grads. */
  CHECK(memcmp(b.o, b.o_repeat, b.x_n * sizeof(float)) == 0);
  CHECK(memcmp(b.o, b.o_direct, b.x_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dx, b.dx_d, b.x_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dw_q, b.dw_q_d, b.wq_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dw_dkv, b.dw_dkv_d, b.wdkv_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dw_uk, b.dw_uk_d, b.wuk_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dw_uv, b.dw_uv_d, b.wuk_n * sizeof(float)) == 0);
  CHECK(memcmp(b.dw_o, b.dw_o_d, b.wo_n * sizeof(float)) == 0);

  /* Central FD gradient checks on ALL 6 gradients. */
  const size_t sample_cap = (d->tokens <= 4u) ? 16u : 8u;
  check_tensor_fd(d, &b, b.x, b.dx, b.x_n, sample_cap, max_fd_err, fd_checks);
  check_tensor_fd(d, &b, b.w_q, b.dw_q, b.wq_n, sample_cap, max_fd_err,
                  fd_checks);
  check_tensor_fd(d, &b, b.w_dkv, b.dw_dkv, b.wdkv_n, sample_cap, max_fd_err,
                  fd_checks);
  check_tensor_fd(d, &b, b.w_uk, b.dw_uk, b.wuk_n, sample_cap, max_fd_err,
                  fd_checks);
  check_tensor_fd(d, &b, b.w_uv, b.dw_uv, b.wuk_n, sample_cap, max_fd_err,
                  fd_checks);
  check_tensor_fd(d, &b, b.w_o, b.dw_o, b.wo_n, sample_cap, max_fd_err,
                  fd_checks);

  /* Compressed latent KV-cache incremental prefill+decode parity. */
  run_cached_prefill_decode(d, mc->prefill_tokens, &b, b.o_cached);
  /* Full-sequence prefill must match prefill+step bit-for-bit! */
  run_cached_prefill_decode(d, d->tokens, &b, b.o_full_prefill);
  CHECK(memcmp(b.o_cached, b.o_full_prefill, b.x_n * sizeof(float)) == 0);
  /* And cached decode must match full-context binary64 MLA within <= 1e-6. */
  for (size_t i = 0; i < b.x_n; ++i) {
    const double diff = fabs((double)b.o_cached[i] - (double)b.o[i]);
    if (diff > *max_cache_diff)
      *max_cache_diff = diff;
    CHECK(diff <= 1e-6);
  }
  /* Also verify token-by-token (prefill=1 + T-1 steps) matches bit-for-bit! */
  if (d->tokens > 1u && mc->prefill_tokens != 1u) {
    run_cached_prefill_decode(d, 1u, &b, b.o_tmp);
    CHECK(memcmp(b.o_cached, b.o_tmp, b.x_n * sizeof(float)) == 0);
  }

  free_buffers(&b);
}

static void check_negative_and_atomicity(const et_kernel_runtime *runtime) {
  const et_mod5_mla_dims d = {1, 4, 8, 2, 4, 4}; /* HD=8, 4x compression */
  mla_buffers b;
  alloc_buffers(&d, &b);

  unsigned char o_sentinel[1 * 4 * 8 * sizeof(float)];
  unsigned char dx_sentinel[1 * 4 * 8 * sizeof(float)];
  unsigned char dwq_sentinel[8 * 8 * sizeof(float)];
  unsigned char dwdkv_sentinel[4 * 8 * sizeof(float)];
  unsigned char dwuk_sentinel[8 * 4 * sizeof(float)];
  unsigned char dwuv_sentinel[8 * 4 * sizeof(float)];
  unsigned char dwo_sentinel[8 * 8 * sizeof(float)];
  fill_sentinel(o_sentinel, sizeof(o_sentinel), 0x11u);
  fill_sentinel(dx_sentinel, sizeof(dx_sentinel), 0x22u);
  fill_sentinel(dwq_sentinel, sizeof(dwq_sentinel), 0x33u);
  fill_sentinel(dwdkv_sentinel, sizeof(dwdkv_sentinel), 0x44u);
  fill_sentinel(dwuk_sentinel, sizeof(dwuk_sentinel), 0x55u);
  fill_sentinel(dwuv_sentinel, sizeof(dwuv_sentinel), 0x66u);
  fill_sentinel(dwo_sentinel, sizeof(dwo_sentinel), 0x77u);

  memcpy(b.o, o_sentinel, sizeof(o_sentinel));
  memcpy(b.dx, dx_sentinel, sizeof(dx_sentinel));
  memcpy(b.dw_q, dwq_sentinel, sizeof(dwq_sentinel));
  memcpy(b.dw_dkv, dwdkv_sentinel, sizeof(dwdkv_sentinel));
  memcpy(b.dw_uk, dwuk_sentinel, sizeof(dwuk_sentinel));
  memcpy(b.dw_uv, dwuv_sentinel, sizeof(dwuv_sentinel));
  memcpy(b.dw_o, dwo_sentinel, sizeof(dwo_sentinel));

  et_kernel_error error;
  /* 1. Invalid dimensions (bounds + dc >= 2*H*Dh compression rule).
   * Use dedicated oversized disjoint spans for K1 bad_dims dispatch so larger
   * out-of-range extents cannot accidentally overlap neighboring heap chunks. */
  const size_t bad_span_elems = 4u * 64u * 64u;
  float *bad_pool = (float *)calloc(7u * bad_span_elems, sizeof(float));
  CHECK(bad_pool != NULL);
  float *bx = bad_pool + 0u * bad_span_elems;
  float *bwq = bad_pool + 1u * bad_span_elems;
  float *bwdkv = bad_pool + 2u * bad_span_elems;
  float *bwuk = bad_pool + 3u * bad_span_elems;
  float *bwuv = bad_pool + 4u * bad_span_elems;
  float *bwo = bad_pool + 5u * bad_span_elems;
  float *bo = bad_pool + 6u * bad_span_elems;
  memcpy(bo, o_sentinel, sizeof(o_sentinel));

  const et_mod5_mla_dims bad_dims[] = {
      {0, 4, 8, 2, 4, 4},   {3, 4, 8, 2, 4, 4},   {1, 0, 8, 2, 4, 4},
      {1, 33, 8, 2, 4, 4},  {1, 4, 3, 2, 4, 4},   {1, 4, 33, 2, 4, 4},
      {1, 4, 8, 0, 4, 4},   {1, 4, 8, 5, 4, 4},   {1, 4, 8, 2, 1, 1},
      {1, 4, 8, 2, 17, 4},  {1, 4, 8, 2, 4, 0},   {1, 4, 8, 2, 4, 17},
      {1, 4, 8, 1, 2, 4},   /* dc == 2*H*Dh (4 == 4) */
      {1, 4, 8, 1, 2, 5},   /* dc > 2*H*Dh */
      {1, 4, 8, 2, 2, 8},   /* dc == 2*H*Dh (8 == 8) */
      {1, 4, 8, 4, 2, 16},  /* dc == 2*H*Dh (16 == 16) */
  };
  dispatching = 1;
  for (size_t i = 0; i < ARRAY_COUNT(bad_dims); ++i) {
    CHECK(et_mod5_mla_forward_f32(&bad_dims[i], bx, bwq, bwdkv, bwuk, bwuv, bwo,
                                  bo, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(dispatch_mla_k1(runtime, 0, &bad_dims[i], bx, bwq, bwdkv, bwuk, bwuv,
                          bwo, NULL, bo, NULL, NULL, NULL, NULL, NULL,
                          &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  }
  CHECK(memcmp(bo, o_sentinel, sizeof(o_sentinel)) == 0);
  CHECK(et_mod5_mla_kv_compression_ratio(1u, 2u, 4u) == 0.0);
  CHECK(et_mod5_mla_latent_kv_bytes(0u, 4u, 4u) == 0u);
  CHECK(et_mod5_mla_full_kv_bytes(1u, 33u, 2u, 4u) == 0u);

  /* 2. Null pointers. */
  CHECK(et_mod5_mla_forward_f32(NULL, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                                b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_forward_f32(&d, NULL, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_backward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                 b.dy, b.dx, b.dw_q, b.dw_dkv, b.dw_uk, b.dw_uv,
                                 NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* 3. Misaligned pointers (via untyped char* offset +1/+2). */
  unsigned char raw_buf[512];
  memset(raw_buf, 0, sizeof(raw_buf));
  CHECK(et_mod5_mla_forward_f32(raw_buf + 1, b.x, b.w_q, b.w_dkv, b.w_uk,
                                b.w_uv, b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_forward_f32(&d, (unsigned char *)b.x + 1, b.w_q, b.w_dkv,
                                b.w_uk, b.w_uv, b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                (unsigned char *)b.o + 2, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_mla_k1(runtime, 0, &d, (unsigned char *)b.x + 1, b.w_q,
                        b.w_dkv, b.w_uk, b.w_uv, b.w_o, NULL, b.o, NULL, NULL,
                        NULL, NULL, NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* 4. Aliased inputs / outputs. */
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uk, b.w_o,
                                b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.x, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_backward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                 b.dy, b.dx, b.dw_q, b.dw_dkv, b.dw_uk, b.dw_uk,
                                 b.dw_o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(dispatch_mla_k1(runtime, 0, &d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, NULL, b.x, NULL, NULL, NULL, NULL, NULL,
                        &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* 5. Non-finite inputs (NaN / Inf) and finite-input overflow. */
  const float saved_x0 = b.x[0];
  b.x[0] = NAN;
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_backward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                 b.dy, b.dx, b.dw_q, b.dw_dkv, b.dw_uk, b.dw_uv,
                                 b.dw_o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  b.x[0] = INFINITY;
  CHECK(dispatch_mla_k1(runtime, 0, &d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, NULL, b.o, NULL, NULL, NULL, NULL, NULL,
                        &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  b.x[0] = saved_x0;

  /* Overflow in output projection from extreme finite weight. */
  const float saved_wo0 = b.w_o[0];
  const float saved_wuv0 = b.w_uv[0];
  b.w_o[0] = 1e38f;
  b.w_uv[0] = 1e38f;
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  b.w_o[0] = saved_wo0;
  b.w_uv[0] = saved_wuv0;

  /* 6. x87 / MXCSR FP-environment checks (non-RN, FTZ, DAZ, unmasked). */
  fenv_t saved_env;
  CHECK(fegetenv(&saved_env) == 0);
  errno = EAGAIN;

  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(errno == EAGAIN);
  CHECK(fesetenv(&saved_env) == 0);

  const unsigned int orig_mxcsr = _mm_getcsr();
  _mm_setcsr(orig_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(orig_mxcsr | 0x0040u); /* DAZ */
  CHECK(et_mod5_mla_forward_f32(&d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv, b.w_o,
                                b.o, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(orig_mxcsr & ~0x0080u); /* Unmask invalid op */
  CHECK(dispatch_mla_k1(runtime, 0, &d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                        b.w_o, NULL, b.o, NULL, NULL, NULL, NULL, NULL,
                        &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  _mm_setcsr(orig_mxcsr);
  CHECK(fesetenv(&saved_env) == 0);

  /* Verify EVERY output sentinel remained byte-for-byte untouched! */
  CHECK(memcmp(b.o, o_sentinel, sizeof(o_sentinel)) == 0);
  CHECK(memcmp(b.dx, dx_sentinel, sizeof(dx_sentinel)) == 0);
  CHECK(memcmp(b.dw_q, dwq_sentinel, sizeof(dwq_sentinel)) == 0);
  CHECK(memcmp(b.dw_dkv, dwdkv_sentinel, sizeof(dwdkv_sentinel)) == 0);
  CHECK(memcmp(b.dw_uk, dwuk_sentinel, sizeof(dwuk_sentinel)) == 0);
  CHECK(memcmp(b.dw_uv, dwuv_sentinel, sizeof(dwuv_sentinel)) == 0);
  CHECK(memcmp(b.dw_o, dwo_sentinel, sizeof(dwo_sentinel)) == 0);

  /* 7. KV-cache negative tests & failure atomicity. */
  et_mod5_mla_kv_cache cache;
  const size_t c_bytes =
      et_mod5_mla_latent_kv_bytes(d.batch, d.tokens, d.latent_dim);
  /* Wrong byte length / misaligned c_kv / aliased buffers. */
  CHECK(et_mod5_mla_cache_init(&cache, &d, b.c_kv, c_bytes - 4u, &error) ==
        ET_KERNEL_ERROR_SHAPE_MISMATCH);
  CHECK(et_mod5_mla_cache_init(&cache, &d, (unsigned char *)b.c_kv + 1, c_bytes,
                               &error) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_cache_init(&cache, &d, &cache, c_bytes, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_cache_init(&cache, &d, b.c_kv, c_bytes, &error) == 0);

  /* Invalid prefill counts (0 and > max_tokens). */
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, 0u, b.x, b.w_q, b.w_dkv, b.w_uk,
                                      b.w_uv, b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, 5u, b.x, b.w_q, b.w_dkv, b.w_uk,
                                      b.w_uv, b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache.seq_len == 0u);
  CHECK(memcmp(b.o, o_sentinel, sizeof(o_sentinel)) == 0);

  /* Fill cache completely, then verify overflow & double-prefill rejection. */
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, 4u, b.x, b.w_q, b.w_dkv, b.w_uk,
                                      b.w_uv, b.w_o, b.o_cached, &error) == 0);
  CHECK(cache.seq_len == 4u);
  /* Second prefill without reset rejected. */
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, 1u, b.x, b.w_q, b.w_dkv, b.w_uk,
                                      b.w_uv, b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  /* Step overflow rejected with BUFFER_TOO_SMALL and untouched output. */
  CHECK(et_mod5_mla_cache_step_f32(&cache, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                                   b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_BUFFER_TOO_SMALL);
  CHECK(cache.seq_len == 4u);
  CHECK(memcmp(b.o, o_sentinel, sizeof(o_sentinel)) == 0);

  /* Corrupted non-finite cached C_kv rejected cleanly on next step. */
  CHECK(et_mod5_mla_cache_reset(&cache, &error) == 0);
  CHECK(et_mod5_mla_cache_prefill_f32(&cache, 2u, b.x, b.w_q, b.w_dkv, b.w_uk,
                                      b.w_uv, b.w_o, b.o_cached, &error) == 0);
  b.c_kv[0] = NAN;
  CHECK(et_mod5_mla_cache_step_f32(&cache, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                                   b.w_o, b.o, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(cache.seq_len == 2u);
  CHECK(memcmp(b.o, o_sentinel, sizeof(o_sentinel)) == 0);

  /* Bridge negative tests. */
  CHECK(et_mod5_mla_aot_set_dims(raw_buf + 1, 1, 4, 8, 2, 4, 4) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_mla_aot_cache_seq_len(raw_buf + 1) == -1);

  dispatching = 0;
  free(bad_pool);
  free_buffers(&b);
}

static void dump_hex_line(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)buf[i]);
  }
  putchar('\n');
}

static int run_oracle_dump(const et_kernel_runtime *runtime) {
  et_kernel_error error;
  for (size_t ci = 0; ci < ARRAY_COUNT(k_cases); ++ci) {
    const mla_case *mc = &k_cases[ci];
    const et_mod5_mla_dims *d = &mc->d;
    mla_buffers b;
    alloc_buffers(d, &b);

    dispatching = 1;
    CHECK(dispatch_mla_k1(runtime, 0, d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                          b.w_o, NULL, b.o, NULL, NULL, NULL, NULL, NULL,
                          &error) == 0);
    CHECK(dispatch_mla_k1(runtime, 1, d, b.x, b.w_q, b.w_dkv, b.w_uk, b.w_uv,
                          b.w_o, b.dy, b.dx, b.dw_q, b.dw_dkv, b.dw_uk, b.dw_uv,
                          b.dw_o, &error) == 0);
    dispatching = 0;

    run_cached_prefill_decode(d, mc->prefill_tokens, &b, b.o_cached);

    printf("CASE %zu %zu %zu %zu %zu %zu %zu\n", d->batch, d->tokens,
           d->model_dim, d->heads, d->head_dim, d->latent_dim,
           mc->prefill_tokens);
    dump_hex_line(b.x, b.x_n);
    dump_hex_line(b.w_q, b.wq_n);
    dump_hex_line(b.w_dkv, b.wdkv_n);
    dump_hex_line(b.w_uk, b.wuk_n);
    dump_hex_line(b.w_uv, b.wuk_n);
    dump_hex_line(b.w_o, b.wo_n);
    dump_hex_line(b.dy, b.x_n);
    dump_hex_line(b.o, b.x_n);
    dump_hex_line(b.dx, b.x_n);
    dump_hex_line(b.dw_q, b.wq_n);
    dump_hex_line(b.dw_dkv, b.wdkv_n);
    dump_hex_line(b.dw_uk, b.wuk_n);
    dump_hex_line(b.dw_uv, b.wuk_n);
    dump_hex_line(b.dw_o, b.wo_n);
    dump_hex_line(b.o_cached, b.x_n);
    free_buffers(&b);
  }
  return 0;
}

int main(int argc, char **argv) {
  et_kernel_runtime *runtime = discover_mla();

  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    const int rc = run_oracle_dump(runtime);
    et_kernel_runtime_destroy(runtime);
    return rc;
  }

  /* Verify K1 capability metadata & 16 exact shape rows. */
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, "kernel.mla");
  CHECK(cap != NULL);
  CHECK(cap->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(cap->deterministic == 1u);
  CHECK(cap->shape_range_count == 16u);

  double max_fd_err = 0.0;
  double max_cache_diff = 0.0;
  size_t fd_checks = 0u;
  for (size_t i = 0; i < ARRAY_COUNT(k_cases); ++i) {
    check_case(runtime, &k_cases[i], &max_fd_err, &fd_checks, &max_cache_diff);
  }

  check_negative_and_atomicity(runtime);

  et_kernel_runtime_destroy(runtime);
  printf("MOD5 MLA native suite: PASS (cases=%zu, fd_checks=%zu, "
         "max_fd_err=%.3e, max_cache_diff=%.3e, "
         "h4_dh8_dc8_ratio=%.1fx)\n",
         ARRAY_COUNT(k_cases), fd_checks, max_fd_err, max_cache_diff,
         et_mod5_mla_kv_compression_ratio(4u, 8u, 8u));
  return 0;
}
