#include "eshkol_transformer/mamba_ssm_abi.h"
#include "mamba_ssm_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "Mamba S6 native test requires x86-64"
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

/* Trap any heap allocation during Mamba core / cache / bridge hot dispatch. */
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

static void fill_det(float *buf, size_t count, int salt, float scale) {
  for (size_t i = 0; i < count; ++i) {
    const int v = (int)((37u * (unsigned)salt + 19u * (unsigned)i +
                         11u * (unsigned)(i / 3u)) %
                        33u) -
                  16;
    const float jitter = 1.0f + 0.017f * (float)(i % 5u);
    buf[i] = (float)v * scale * jitter;
  }
}

static void fill_a_log(float *buf, size_t d, size_t s, int salt) {
  for (size_t di = 0; di < d; ++di) {
    for (size_t si = 0; si < s; ++si) {
      const double base = log((double)(si + 1u));
      const double off =
          0.07 * (double)((int)((salt * 13u + di * 7u + si * 5u) % 9u) - 4);
      buf[di * s + si] = (float)(base + off);
    }
  }
}

static void fill_sentinel(unsigned char *buf, size_t bytes,
                          unsigned char seed) {
  for (size_t i = 0; i < bytes; ++i)
    buf[i] = (unsigned char)(seed ^ (unsigned char)((i * 29u + 7u) & 0xffu));
}

static int sentinel_intact(const unsigned char *buf, size_t bytes,
                           unsigned char seed) {
  for (size_t i = 0; i < bytes; ++i) {
    const unsigned char expected =
        (unsigned char)(seed ^ (unsigned char)((i * 29u + 7u) & 0xffu));
    if (buf[i] != expected)
      return 0;
  }
  return 1;
}

static et_mamba_config_v1 make_cfg(size_t n, size_t t, size_t d, size_t s,
                                   size_t k) {
  et_mamba_config_v1 cfg;
  cfg.struct_size = sizeof(cfg);
  cfg.batch = n;
  cfg.seq_len = t;
  cfg.d_inner = d;
  cfg.d_state = s;
  cfg.k_conv = k;
  return cfg;
}

static double dot_f32_f64(const float *a, const float *b, size_t count) {
  double acc = 0.0;
  for (size_t i = 0; i < count; ++i)
    acc += (double)a[i] * (double)b[i];
  return acc;
}

static void print_hex_array(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)buf[i]);
  }
  putchar('\n');
}

/* ============================================================================
 * 1. Config & O(1) Constant State-Byte Verification
 * ============================================================================ */
static void test_config_and_o1_state_bytes(void) {
  const size_t seq_lens[] = {1u, 8u, 16u, 32u, 64u};
  size_t ref_conv = 0u, ref_ssm = 0u, ref_total = 0u;

  for (size_t i = 0; i < ARRAY_COUNT(seq_lens); ++i) {
    const et_mamba_config_v1 cfg = make_cfg(2u, seq_lens[i], 16u, 8u, 4u);
    dispatching = 1;
    CHECK(et_mamba_validate_config_v1(&cfg) == ET_MAMBA_OK);
    const size_t cb = et_mamba_cache_conv_state_bytes_v1(&cfg);
    const size_t sb = et_mamba_cache_ssm_state_bytes_v1(&cfg);
    const size_t tb = et_mamba_cache_total_state_bytes_v1(&cfg);
    dispatching = 0;

    CHECK(cb == 2u * 3u * 16u * sizeof(float));
    CHECK(sb == 2u * 16u * 8u * sizeof(double));
    CHECK(tb == cb + sb);
    if (i == 0u) {
      ref_conv = cb;
      ref_ssm = sb;
      ref_total = tb;
    } else {
      CHECK(cb == ref_conv);
      CHECK(sb == ref_ssm);
      CHECK(tb == ref_total);
    }
  }
  printf("MAMBA_SSM O(1) state bytes (N=2,D=16,S=8,K=4,T=1..64): "
         "conv=%zu ssm=%zu total=%zu PASS\n",
         ref_conv, ref_ssm, ref_total);
}

/* ============================================================================
 * 2. Stage A Causal Conv1D + SiLU Finite-Difference Gradient Checks
 * ============================================================================ */
static void test_conv1d_silu_fd(void) {
  const struct {
    size_t n, t, d, k;
  } cases[] = {
      {1u, 1u, 2u, 2u},
      {1u, 2u, 4u, 4u}, /* T < K */
      {2u, 7u, 6u, 3u},
      {2u, 12u, 8u, 4u},
  };
  const float eps = 1.0f / 256.0f;
  double max_err = 0.0;
  size_t checked = 0u;

  for (size_t c = 0; c < ARRAY_COUNT(cases); ++c) {
    const et_mamba_config_v1 cfg =
        make_cfg(cases[c].n, cases[c].t, cases[c].d, 2u, cases[c].k);
    const size_t ntd = cfg.batch * cfg.seq_len * cfg.d_inner;
    const size_t dk = cfg.d_inner * cfg.k_conv;
    const size_t d = cfg.d_inner;

    float u[256], w[64], b[16], dx[256];
    float x_plus[256], x_minus[256], du[256], dw[64], db[16];

    fill_det(u, ntd, (int)(10 + c), 0.0625f);
    fill_det(w, dk, (int)(20 + c), 0.0625f);
    fill_det(b, d, (int)(30 + c), 0.03125f);
    fill_det(dx, ntd, (int)(40 + c), 0.0625f);

    dispatching = 1;
    CHECK(et_mamba_causal_conv1d_silu_backward_v1(&cfg, u, w, b, dx, du, dw,
                                                  db) == ET_MAMBA_OK);
    dispatching = 0;

    /* Check all elements of u, w, b (or strided subset for larger tensors). */
    for (size_t i = 0; i < ntd; i += (ntd > 32u ? 5u : 1u)) {
      const float orig = u[i];
      u[i] = orig + eps;
      dispatching = 1;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_plus) ==
            ET_MAMBA_OK);
      u[i] = orig - eps;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_minus) ==
            ET_MAMBA_OK);
      dispatching = 0;
      u[i] = orig;

      const double lp = dot_f32_f64(x_plus, dx, ntd);
      const double lm = dot_f32_f64(x_minus, dx, ntd);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double err = fabs((double)du[i] - num);
      CHECK(err <= 1e-4 + 1e-3 * fabs(num));
      if (err > max_err)
        max_err = err;
      ++checked;
    }

    for (size_t i = 0; i < dk; ++i) {
      const float orig = w[i];
      w[i] = orig + eps;
      dispatching = 1;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_plus) ==
            ET_MAMBA_OK);
      w[i] = orig - eps;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_minus) ==
            ET_MAMBA_OK);
      dispatching = 0;
      w[i] = orig;

      const double lp = dot_f32_f64(x_plus, dx, ntd);
      const double lm = dot_f32_f64(x_minus, dx, ntd);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double err = fabs((double)dw[i] - num);
      CHECK(err <= 1e-4 + 1e-3 * fabs(num));
      if (err > max_err)
        max_err = err;
      ++checked;
    }

    for (size_t i = 0; i < d; ++i) {
      const float orig = b[i];
      b[i] = orig + eps;
      dispatching = 1;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_plus) ==
            ET_MAMBA_OK);
      b[i] = orig - eps;
      CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, w, b, x_minus) ==
            ET_MAMBA_OK);
      dispatching = 0;
      b[i] = orig;

      const double lp = dot_f32_f64(x_plus, dx, ntd);
      const double lm = dot_f32_f64(x_minus, dx, ntd);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double err = fabs((double)db[i] - num);
      CHECK(err <= 1e-4 + 1e-3 * fabs(num));
      if (err > max_err)
        max_err = err;
      ++checked;
    }
  }
  printf("MAMBA_SSM Stage A CausalConv1D+SiLU FD: checked=%zu "
         "max_abs_err=%.3e PASS\n",
         checked, max_err);
}

/* ============================================================================
 * 3. Stage B Selective Scan S6 Finite-Difference Gradient Checks (All 7 Ops)
 * ============================================================================ */
static void check_s6_fd_tensor(const et_mamba_config_v1 *cfg, float *target_buf,
                               const float *analytic_grad, size_t count,
                               size_t stride, float *x, float *dt_raw,
                               float *a_log, float *b_in, float *c_in,
                               float *d_skip, float *z_opt, const float *dout,
                               double *max_err, size_t *checked) {
  const size_t ntd = cfg->batch * cfg->seq_len * cfg->d_inner;
  const float eps = 1.0f / 256.0f;
  float out_p[256], out_m[256];

  for (size_t i = 0; i < count; i += stride) {
    const float orig = target_buf[i];
    target_buf[i] = orig + eps;
    dispatching = 1;
    CHECK(et_mamba_selective_scan_forward_v1(cfg, x, dt_raw, a_log, b_in, c_in,
                                             d_skip, z_opt, out_p) ==
          ET_MAMBA_OK);
    target_buf[i] = orig - eps;
    CHECK(et_mamba_selective_scan_forward_v1(cfg, x, dt_raw, a_log, b_in, c_in,
                                             d_skip, z_opt, out_m) ==
          ET_MAMBA_OK);
    dispatching = 0;
    target_buf[i] = orig;

    const double lp = dot_f32_f64(out_p, dout, ntd);
    const double lm = dot_f32_f64(out_m, dout, ntd);
    const double num = (lp - lm) / (2.0 * (double)eps);
    const double err = fabs((double)analytic_grad[i] - num);
    CHECK(err <= 1e-4 + 1e-3 * fabs(num));
    if (err > *max_err)
      *max_err = err;
    ++(*checked);
  }
}

static void test_selective_scan_fd(void) {
  const struct {
    size_t n, t, d, s;
    int gated;
  } cases[] = {
      {1u, 1u, 2u, 2u, 1},
      {1u, 6u, 4u, 3u, 1},
      {2u, 8u, 4u, 4u, 0},
      {2u, 10u, 6u, 4u, 1},
  };
  double max_err = 0.0;
  size_t checked = 0u;

  for (size_t c = 0; c < ARRAY_COUNT(cases); ++c) {
    const et_mamba_config_v1 cfg =
        make_cfg(cases[c].n, cases[c].t, cases[c].d, cases[c].s, 3u);
    const size_t ntd = cfg.batch * cfg.seq_len * cfg.d_inner;
    const size_t nts = cfg.batch * cfg.seq_len * cfg.d_state;
    const size_t ds = cfg.d_inner * cfg.d_state;
    const size_t d = cfg.d_inner;

    float x[256], dt_raw[256], a_log[64], b_in[256], c_in[256], d_skip[16];
    float z_in[256], dout[256];
    float dx[256], ddt[256], dalog[64], db[256], dc[256], ddskip[16], dz[256];

    fill_det(x, ntd, (int)(50 + c), 0.0625f);
    fill_det(dt_raw, ntd, (int)(60 + c), 0.09375f);
    fill_a_log(a_log, cfg.d_inner, cfg.d_state, (int)(70 + c));
    fill_det(b_in, nts, (int)(80 + c), 0.0625f);
    fill_det(c_in, nts, (int)(90 + c), 0.0625f);
    fill_det(d_skip, d, (int)(100 + c), 0.0625f);
    fill_det(z_in, ntd, (int)(110 + c), 0.0625f);
    fill_det(dout, ntd, (int)(120 + c), 0.0625f);

    float *z_ptr = cases[c].gated ? z_in : NULL;
    float *dz_ptr = cases[c].gated ? dz : NULL;

    dispatching = 1;
    CHECK(et_mamba_selective_scan_backward_v1(&cfg, x, dt_raw, a_log, b_in,
                                              c_in, d_skip, z_ptr, dout, dx,
                                              ddt, dalog, db, dc, ddskip,
                                              dz_ptr) == ET_MAMBA_OK);
    dispatching = 0;

    const size_t st = (ntd > 32u ? 5u : 1u);
    const size_t st_s = (nts > 32u ? 5u : 1u);
    check_s6_fd_tensor(&cfg, x, dx, ntd, st, x, dt_raw, a_log, b_in, c_in,
                       d_skip, z_ptr, dout, &max_err, &checked);
    check_s6_fd_tensor(&cfg, dt_raw, ddt, ntd, st, x, dt_raw, a_log, b_in, c_in,
                       d_skip, z_ptr, dout, &max_err, &checked);
    check_s6_fd_tensor(&cfg, a_log, dalog, ds, 1u, x, dt_raw, a_log, b_in, c_in,
                       d_skip, z_ptr, dout, &max_err, &checked);
    check_s6_fd_tensor(&cfg, b_in, db, nts, st_s, x, dt_raw, a_log, b_in, c_in,
                       d_skip, z_ptr, dout, &max_err, &checked);
    check_s6_fd_tensor(&cfg, c_in, dc, nts, st_s, x, dt_raw, a_log, b_in, c_in,
                       d_skip, z_ptr, dout, &max_err, &checked);
    check_s6_fd_tensor(&cfg, d_skip, ddskip, d, 1u, x, dt_raw, a_log, b_in,
                       c_in, d_skip, z_ptr, dout, &max_err, &checked);
    if (cases[c].gated) {
      check_s6_fd_tensor(&cfg, z_in, dz, ntd, st, x, dt_raw, a_log, b_in, c_in,
                         d_skip, z_ptr, dout, &max_err, &checked);
    }
  }

  /* Extreme dt_raw stability test (-30, 0, +30) for overflow-safe softplus. */
  {
    const et_mamba_config_v1 cfg = make_cfg(1u, 3u, 2u, 2u, 2u);
    const float x[6] = {0.5f, -0.25f, 0.75f, 0.1f, -0.4f, 0.3f};
    const float dt_raw[6] = {-30.0f, 30.0f, 0.0f, -50.0f, 50.0f, 100.0f};
    const float a_log[4] = {-1.0f, 0.0f, -0.5f, 0.5f};
    const float b_in[6] = {0.2f, -0.3f, 0.4f, 0.1f, -0.2f, 0.5f};
    const float c_in[6] = {0.3f, 0.2f, -0.1f, 0.4f, 0.2f, -0.3f};
    const float d_skip[2] = {0.5f, -0.25f};
    const float dout[6] = {1.0f, -0.5f, 0.25f, 0.75f, -1.0f, 0.5f};
    float out[6], dx[6], ddt[6], dalog[4], db[6], dc[6], ddskip[2];

    dispatching = 1;
    CHECK(et_mamba_selective_scan_forward_v1(&cfg, x, dt_raw, a_log, b_in, c_in,
                                             d_skip, NULL, out) == ET_MAMBA_OK);
    CHECK(et_mamba_selective_scan_backward_v1(&cfg, x, dt_raw, a_log, b_in,
                                              c_in, d_skip, NULL, dout, dx, ddt,
                                              dalog, db, dc, ddskip,
                                              NULL) == ET_MAMBA_OK);
    dispatching = 0;
    for (size_t i = 0; i < 6u; ++i) {
      CHECK(isfinite(out[i]) && isfinite(dx[i]) && isfinite(ddt[i]) &&
            isfinite(db[i]) && isfinite(dc[i]));
    }
  }

  printf("MAMBA_SSM Stage B SelectiveScan S6 FD: checked=%zu "
         "max_abs_err=%.3e PASS\n",
         checked, max_err);
}

/* ============================================================================
 * 4. Stage C O(1) Recurrent Cache vs Full-Sequence Parity
 * ============================================================================ */
static void test_cache_parity(void) {
  const struct {
    size_t n, t, d, s, k;
    int gated;
  } cases[] = {
      {1u, 1u, 2u, 2u, 2u, 1},
      {1u, 8u, 8u, 4u, 4u, 1},
      {2u, 16u, 8u, 4u, 3u, 0},
      {4u, 64u, 32u, 16u, 4u, 1}, /* max admitted shape */
  };
  double max_diff = 0.0;

  for (size_t c = 0; c < ARRAY_COUNT(cases); ++c) {
    const et_mamba_config_v1 cfg =
        make_cfg(cases[c].n, cases[c].t, cases[c].d, cases[c].s, cases[c].k);
    const size_t n = cfg.batch, t = cfg.seq_len, d = cfg.d_inner,
                 s = cfg.d_state, k = cfg.k_conv;
    const size_t ntd = n * t * d;
    const size_t nts = n * t * s;
    const size_t dk = d * k;
    const size_t ds = d * s;

    float *u = (float *)malloc(ntd * sizeof(float));
    float *conv_w = (float *)malloc(dk * sizeof(float));
    float *conv_b = (float *)malloc(d * sizeof(float));
    float *x_full = (float *)malloc(ntd * sizeof(float));
    float *dt_raw = (float *)malloc(ntd * sizeof(float));
    float *a_log = (float *)malloc(ds * sizeof(float));
    float *b_in = (float *)malloc(nts * sizeof(float));
    float *c_in = (float *)malloc(nts * sizeof(float));
    float *d_skip = (float *)malloc(d * sizeof(float));
    float *z_in = (float *)malloc(ntd * sizeof(float));
    float *out_full = (float *)malloc(ntd * sizeof(float));
    float *u_step = (float *)malloc(n * d * sizeof(float));
    float *dt_step = (float *)malloc(n * d * sizeof(float));
    float *z_step = (float *)malloc(n * d * sizeof(float));
    float *b_step = (float *)malloc(n * s * sizeof(float));
    float *c_step = (float *)malloc(n * s * sizeof(float));
    float *out_step = (float *)malloc(n * d * sizeof(float));

    const size_t conv_bytes = et_mamba_cache_conv_state_bytes_v1(&cfg);
    const size_t ssm_bytes = et_mamba_cache_ssm_state_bytes_v1(&cfg);
    float *conv_state = (float *)malloc(conv_bytes);
    double *ssm_state = (double *)malloc(ssm_bytes);

    CHECK(u && conv_w && conv_b && x_full && dt_raw && a_log && b_in && c_in &&
          d_skip && z_in && out_full && u_step && dt_step && z_step && b_step &&
          c_step && out_step && conv_state && ssm_state);

    fill_det(u, ntd, (int)(200 + c), 0.1f);
    fill_det(conv_w, dk, (int)(210 + c), 0.125f);
    fill_det(conv_b, d, (int)(220 + c), 0.05f);
    fill_det(dt_raw, ntd, (int)(230 + c), 0.15f);
    fill_a_log(a_log, d, s, (int)(240 + c));
    fill_det(b_in, nts, (int)(250 + c), 0.1f);
    fill_det(c_in, nts, (int)(260 + c), 0.1f);
    fill_det(d_skip, d, (int)(270 + c), 0.125f);
    fill_det(z_in, ntd, (int)(280 + c), 0.125f);

    const float *z_full_ptr = cases[c].gated ? z_in : NULL;

    dispatching = 1;
    CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, conv_w, conv_b,
                                                 x_full) == ET_MAMBA_OK);
    CHECK(et_mamba_selective_scan_forward_v1(&cfg, x_full, dt_raw, a_log, b_in,
                                             c_in, d_skip, z_full_ptr,
                                             out_full) == ET_MAMBA_OK);

    et_mamba_cache_v1 cache;
    CHECK(et_mamba_cache_init_v1(&cache, &cfg, conv_state, conv_bytes,
                                 ssm_state, ssm_bytes) == ET_MAMBA_OK);
    CHECK(cache.step_count == 0u);

    /* Run twice (second pass after reset) to verify reset repeatability. */
    for (int pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
        CHECK(et_mamba_cache_reset_v1(&cache) == ET_MAMBA_OK);
        CHECK(cache.step_count == 0u);
      }
      for (size_t ti = 0; ti < t; ++ti) {
        for (size_t ni = 0; ni < n; ++ni) {
          memcpy(&u_step[ni * d], &u[(ni * t + ti) * d], d * sizeof(float));
          memcpy(&dt_step[ni * d], &dt_raw[(ni * t + ti) * d],
                 d * sizeof(float));
          memcpy(&z_step[ni * d], &z_in[(ni * t + ti) * d], d * sizeof(float));
          memcpy(&b_step[ni * s], &b_in[(ni * t + ti) * s], s * sizeof(float));
          memcpy(&c_step[ni * s], &c_in[(ni * t + ti) * s], s * sizeof(float));
        }
        const float *z_st_ptr = cases[c].gated ? z_step : NULL;
        CHECK(et_mamba_cache_step_v1(&cache, u_step, conv_w, conv_b, dt_step,
                                     a_log, b_step, c_step, d_skip, z_st_ptr,
                                     out_step) == ET_MAMBA_OK);
        CHECK(cache.step_count == (uint64_t)(ti + 1u));
        for (size_t ni = 0; ni < n; ++ni) {
          for (size_t di = 0; di < d; ++di) {
            const double diff =
                fabs((double)out_step[ni * d + di] -
                     (double)out_full[(ni * t + ti) * d + di]);
            CHECK(diff <= 1e-6);
            if (diff > max_diff)
              max_diff = diff;
          }
        }
      }
    }
    dispatching = 0;

    free(u);
    free(conv_w);
    free(conv_b);
    free(x_full);
    free(dt_raw);
    free(a_log);
    free(b_in);
    free(c_in);
    free(d_skip);
    free(z_in);
    free(out_full);
    free(u_step);
    free(dt_step);
    free(z_step);
    free(b_step);
    free(c_step);
    free(out_step);
    free(conv_state);
    free(ssm_state);
  }
  printf("MAMBA_SSM Stage C O(1) cache vs full-seq parity (incl max "
         "4x64x32x16x4): max_abs_err=%.3e PASS\n",
         max_diff);
}

/* ============================================================================
 * 5. Negative Tests: Non-finite, A_log Overflow, Bad Dims, Alignment, Aliasing,
 *    FP Environment, Sentinel Output & Cache Preservation
 * ============================================================================ */
static void test_negative_contracts(void) {
  const et_mamba_config_v1 cfg = make_cfg(1u, 4u, 4u, 2u, 3u);
  const size_t ntd = 16u, nts = 8u, dk = 12u, d = 4u;

  float u[16], conv_w[12], conv_b[4], dt_raw[16], a_log[8];
  float b_in[8], c_in[8], d_skip[4], z_in[16], dout[16];
  fill_det(u, ntd, 301, 0.1f);
  fill_det(conv_w, dk, 302, 0.1f);
  fill_det(conv_b, d, 303, 0.1f);
  fill_det(dt_raw, ntd, 304, 0.1f);
  fill_a_log(a_log, d, 2u, 305);
  fill_det(b_in, nts, 306, 0.1f);
  fill_det(c_in, nts, 307, 0.1f);
  fill_det(d_skip, d, 308, 0.1f);
  fill_det(z_in, ntd, 309, 0.1f);
  fill_det(dout, ntd, 310, 0.1f);

  float out_buf[16], dx_buf[16], ddt_buf[16], dalog_buf[8];
  float db_buf[8], dc_buf[8], ddskip_buf[4], dz_buf[16];
  fill_sentinel((unsigned char *)out_buf, sizeof(out_buf), 0xA5u);
  fill_sentinel((unsigned char *)dx_buf, sizeof(dx_buf), 0xB6u);
  fill_sentinel((unsigned char *)ddt_buf, sizeof(ddt_buf), 0xC7u);
  fill_sentinel((unsigned char *)dalog_buf, sizeof(dalog_buf), 0xD8u);
  fill_sentinel((unsigned char *)db_buf, sizeof(db_buf), 0xE9u);
  fill_sentinel((unsigned char *)dc_buf, sizeof(dc_buf), 0xFAu);
  fill_sentinel((unsigned char *)ddskip_buf, sizeof(ddskip_buf), 0x1Bu);
  fill_sentinel((unsigned char *)dz_buf, sizeof(dz_buf), 0x2Cu);

  dispatching = 1;

  /* 5a. Invalid dims / kernel size / bad struct_size. */
  et_mamba_config_v1 bad = cfg;
  bad.struct_size = 0u;
  CHECK(et_mamba_validate_config_v1(&bad) == ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_validate_config_v1(NULL) == ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_cache_total_state_bytes_v1(&bad) == 0u);

  const et_mamba_config_v1 bad_dims[] = {
      make_cfg(0u, 4u, 4u, 2u, 3u),  make_cfg(5u, 4u, 4u, 2u, 3u),
      make_cfg(1u, 0u, 4u, 2u, 3u),  make_cfg(1u, 65u, 4u, 2u, 3u),
      make_cfg(1u, 4u, 1u, 2u, 3u),  make_cfg(1u, 4u, 33u, 2u, 3u),
      make_cfg(1u, 4u, 4u, 1u, 3u),  make_cfg(1u, 4u, 4u, 17u, 3u),
      make_cfg(1u, 4u, 4u, 2u, 1u),  make_cfg(1u, 4u, 4u, 2u, 5u),
  };
  for (size_t i = 0; i < ARRAY_COUNT(bad_dims); ++i) {
    CHECK(et_mamba_validate_config_v1(&bad_dims[i]) ==
          ET_MAMBA_ERR_UNSUPPORTED);
    CHECK(et_mamba_causal_conv1d_silu_forward_v1(&bad_dims[i], u, conv_w,
                                                 conv_b, out_buf) ==
          ET_MAMBA_ERR_UNSUPPORTED);
    CHECK(et_mamba_selective_scan_forward_v1(&bad_dims[i], u, dt_raw, a_log,
                                             b_in, c_in, d_skip, z_in,
                                             out_buf) ==
          ET_MAMBA_ERR_UNSUPPORTED);
  }

  /* 5b. A_log overflow (exp(A_log) huge -> ET_MAMBA_ERR_OVERFLOW). */
  float a_log_huge[8];
  memcpy(a_log_huge, a_log, sizeof(a_log));
  a_log_huge[3] = 85.0f;
  CHECK(et_mamba_selective_scan_forward_v1(&cfg, u, dt_raw, a_log_huge, b_in,
                                           c_in, d_skip, z_in, out_buf) ==
        ET_MAMBA_ERR_OVERFLOW);
  a_log_huge[3] = 800.0f;
  CHECK(et_mamba_selective_scan_backward_v1(
            &cfg, u, dt_raw, a_log_huge, b_in, c_in, d_skip, z_in, dout, dx_buf,
            ddt_buf, dalog_buf, db_buf, dc_buf, ddskip_buf, dz_buf) ==
        ET_MAMBA_ERR_OVERFLOW);

  /* 5c. Non-finite inputs (+Inf, -Inf, NaN). */
  float bad_u[16];
  memcpy(bad_u, u, sizeof(u));
  bad_u[5] = INFINITY;
  CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, bad_u, conv_w, conv_b,
                                               out_buf) ==
        ET_MAMBA_ERR_NONFINITE);
  bad_u[5] = NAN;
  CHECK(et_mamba_selective_scan_forward_v1(&cfg, bad_u, dt_raw, a_log, b_in,
                                           c_in, d_skip, z_in, out_buf) ==
        ET_MAMBA_ERR_NONFINITE);
  bad_u[5] = -INFINITY;
  CHECK(et_mamba_selective_scan_backward_v1(
            &cfg, bad_u, dt_raw, a_log, b_in, c_in, d_skip, z_in, dout, dx_buf,
            ddt_buf, dalog_buf, db_buf, dc_buf, ddskip_buf, dz_buf) ==
        ET_MAMBA_ERR_NONFINITE);

  /* 5d. Misaligned pointers. */
  unsigned char unaligned_buf[sizeof(u) + 8u];
  CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, unaligned_buf + 1u, conv_w,
                                               conv_b, out_buf) ==
        ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_selective_scan_forward_v1(&cfg, u, dt_raw, a_log, b_in, c_in,
                                           d_skip, z_in, unaligned_buf + 2u) ==
        ET_MAMBA_ERR_INVALID_ARG);

  /* 5e. Aliased outputs (output overlaps input or another output). */
  CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, conv_w, conv_b, u) ==
        ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_selective_scan_forward_v1(&cfg, u, dt_raw, a_log, b_in, c_in,
                                           d_skip, z_in, u) ==
        ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_selective_scan_backward_v1(
            &cfg, u, dt_raw, a_log, b_in, c_in, d_skip, z_in, dout, dx_buf,
            dx_buf, dalog_buf, db_buf, dc_buf, ddskip_buf, dz_buf) ==
        ET_MAMBA_ERR_INVALID_ARG);

  /* 5f. Mismatched z_opt / dz_out_opt. */
  CHECK(et_mamba_selective_scan_backward_v1(
            &cfg, u, dt_raw, a_log, b_in, c_in, d_skip, z_in, dout, dx_buf,
            ddt_buf, dalog_buf, db_buf, dc_buf, ddskip_buf, NULL) ==
        ET_MAMBA_ERR_INVALID_ARG);
  CHECK(et_mamba_selective_scan_backward_v1(
            &cfg, u, dt_raw, a_log, b_in, c_in, d_skip, NULL, dout, dx_buf,
            ddt_buf, dalog_buf, db_buf, dc_buf, ddskip_buf, dz_buf) ==
        ET_MAMBA_ERR_INVALID_ARG);

  /* 5g. Cache negative checks & failure-atomic state preservation. */
  et_mamba_cache_v1 cache;
  float conv_state[8];
  double ssm_state[8];
  CHECK(et_mamba_cache_init_v1(&cache, &cfg, conv_state, sizeof(conv_state) - 4,
                               ssm_state, sizeof(ssm_state)) ==
        ET_MAMBA_ERR_SHAPE_MISMATCH);
  CHECK(et_mamba_cache_init_v1(&cache, &cfg, conv_state, sizeof(conv_state),
                               ssm_state, sizeof(ssm_state) - 8) ==
        ET_MAMBA_ERR_SHAPE_MISMATCH);
  CHECK(et_mamba_cache_init_v1(&cache, &cfg, conv_state, sizeof(conv_state),
                               ssm_state, sizeof(ssm_state)) == ET_MAMBA_OK);
  /* Perform one valid step, snapshot state, then inject A_log overflow. */
  float step_out[4];
  CHECK(et_mamba_cache_step_v1(&cache, u, conv_w, conv_b, dt_raw, a_log, b_in,
                               c_in, d_skip, z_in, step_out) == ET_MAMBA_OK);
  CHECK(cache.step_count == 1u);
  float saved_conv[8];
  double saved_ssm[8];
  memcpy(saved_conv, conv_state, sizeof(conv_state));
  memcpy(saved_ssm, ssm_state, sizeof(ssm_state));
  fill_sentinel((unsigned char *)step_out, sizeof(step_out), 0x7Eu);

  CHECK(et_mamba_cache_step_v1(&cache, u, conv_w, conv_b, dt_raw, a_log_huge,
                               b_in, c_in, d_skip, z_in, step_out) ==
        ET_MAMBA_ERR_OVERFLOW);
  CHECK(cache.step_count == 1u);
  CHECK(memcmp(saved_conv, conv_state, sizeof(conv_state)) == 0);
  CHECK(memcmp(saved_ssm, ssm_state, sizeof(ssm_state)) == 0);
  CHECK(sentinel_intact((unsigned char *)step_out, sizeof(step_out), 0x7Eu));

  /* Corrupted cache state (NaN injected) fails closed. */
  ssm_state[2] = NAN;
  CHECK(et_mamba_cache_step_v1(&cache, u, conv_w, conv_b, dt_raw, a_log, b_in,
                               c_in, d_skip, z_in, step_out) ==
        ET_MAMBA_ERR_NONFINITE);
  CHECK(sentinel_intact((unsigned char *)step_out, sizeof(step_out), 0x7Eu));

  /* 5h. Unsupported FP rounding mode and MXCSR FTZ checks. */
  {
    fenv_t saved_env;
    fegetenv(&saved_env);
    CHECK(fesetround(FE_UPWARD) == 0);
    CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, conv_w, conv_b,
                                                 out_buf) ==
          ET_MAMBA_ERR_FP_ENV);
    CHECK(et_mamba_selective_scan_forward_v1(&cfg, u, dt_raw, a_log, b_in, c_in,
                                             d_skip, z_in, out_buf) ==
          ET_MAMBA_ERR_FP_ENV);
    fesetenv(&saved_env);

    const unsigned int orig_mxcsr = _mm_getcsr();
    _mm_setcsr(orig_mxcsr | 0x8000u); /* Set FTZ */
    CHECK(et_mamba_selective_scan_forward_v1(&cfg, u, dt_raw, a_log, b_in, c_in,
                                             d_skip, z_in, out_buf) ==
          ET_MAMBA_ERR_FP_ENV);
    _mm_setcsr(orig_mxcsr);
  }

  dispatching = 0;

  /* Verify ALL output sentinels are byte-for-byte untouched! */
  CHECK(sentinel_intact((unsigned char *)out_buf, sizeof(out_buf), 0xA5u));
  CHECK(sentinel_intact((unsigned char *)dx_buf, sizeof(dx_buf), 0xB6u));
  CHECK(sentinel_intact((unsigned char *)ddt_buf, sizeof(ddt_buf), 0xC7u));
  CHECK(sentinel_intact((unsigned char *)dalog_buf, sizeof(dalog_buf), 0xD8u));
  CHECK(sentinel_intact((unsigned char *)db_buf, sizeof(db_buf), 0xE9u));
  CHECK(sentinel_intact((unsigned char *)dc_buf, sizeof(dc_buf), 0xFAu));
  CHECK(sentinel_intact((unsigned char *)ddskip_buf, sizeof(ddskip_buf),
                        0x1Bu));
  CHECK(sentinel_intact((unsigned char *)dz_buf, sizeof(dz_buf), 0x2Cu));

  printf("MAMBA_SSM negative contracts & failure-atomic sentinels: PASS\n");
}

/* ============================================================================
 * 6. Oracle Data Emission Mode (--oracle)
 * ============================================================================ */
static void run_oracle_mode(void) {
  const struct {
    size_t n, t, d, s, k;
    int gated;
  } cases[] = {
      {1u, 1u, 2u, 2u, 2u, 1},   {1u, 3u, 4u, 2u, 4u, 1},
      {1u, 8u, 8u, 4u, 4u, 1},   {1u, 8u, 8u, 4u, 3u, 0},
      {2u, 6u, 4u, 3u, 2u, 1},   {2u, 12u, 6u, 4u, 3u, 0},
      {3u, 10u, 8u, 6u, 4u, 1},  {4u, 16u, 12u, 8u, 4u, 1},
  };

  for (size_t c = 0; c < ARRAY_COUNT(cases); ++c) {
    const et_mamba_config_v1 cfg =
        make_cfg(cases[c].n, cases[c].t, cases[c].d, cases[c].s, cases[c].k);
    const size_t n = cfg.batch, t = cfg.seq_len, d = cfg.d_inner,
                 s = cfg.d_state, k = cfg.k_conv;
    const size_t ntd = n * t * d, nts = n * t * s, dk = d * k, ds = d * s;

    float *u = (float *)malloc(ntd * sizeof(float));
    float *conv_w = (float *)malloc(dk * sizeof(float));
    float *conv_b = (float *)malloc(d * sizeof(float));
    float *dt_raw = (float *)malloc(ntd * sizeof(float));
    float *a_log = (float *)malloc(ds * sizeof(float));
    float *b_in = (float *)malloc(nts * sizeof(float));
    float *c_in = (float *)malloc(nts * sizeof(float));
    float *d_skip = (float *)malloc(d * sizeof(float));
    float *z_in = (float *)malloc(ntd * sizeof(float));
    float *dout = (float *)malloc(ntd * sizeof(float));

    float *x_conv = (float *)malloc(ntd * sizeof(float));
    float *out_full = (float *)malloc(ntd * sizeof(float));
    float *out_cached = (float *)malloc(ntd * sizeof(float));
    float *dx_scan = (float *)malloc(ntd * sizeof(float));
    float *ddt_raw = (float *)malloc(ntd * sizeof(float));
    float *da_log = (float *)malloc(ds * sizeof(float));
    float *db = (float *)malloc(nts * sizeof(float));
    float *dc = (float *)malloc(nts * sizeof(float));
    float *dd_skip = (float *)malloc(d * sizeof(float));
    float *dz = (float *)malloc(ntd * sizeof(float));
    float *du = (float *)malloc(ntd * sizeof(float));
    float *dconv_w = (float *)malloc(dk * sizeof(float));
    float *dconv_b = (float *)malloc(d * sizeof(float));

    float *u_step = (float *)malloc(n * d * sizeof(float));
    float *dt_step = (float *)malloc(n * d * sizeof(float));
    float *z_step = (float *)malloc(n * d * sizeof(float));
    float *b_step = (float *)malloc(n * s * sizeof(float));
    float *c_step = (float *)malloc(n * s * sizeof(float));
    float *out_step = (float *)malloc(n * d * sizeof(float));

    const size_t conv_bytes = et_mamba_cache_conv_state_bytes_v1(&cfg);
    const size_t ssm_bytes = et_mamba_cache_ssm_state_bytes_v1(&cfg);
    float *conv_state = (float *)malloc(conv_bytes);
    double *ssm_state = (double *)malloc(ssm_bytes);

    fill_det(u, ntd, (int)(400 + c), 0.05f);
    fill_det(conv_w, dk, (int)(410 + c), 0.0625f);
    fill_det(conv_b, d, (int)(420 + c), 0.03125f);
    fill_det(dt_raw, ntd, (int)(430 + c), 0.075f);
    fill_a_log(a_log, d, s, (int)(440 + c));
    fill_det(b_in, nts, (int)(450 + c), 0.0625f);
    fill_det(c_in, nts, (int)(460 + c), 0.0625f);
    fill_det(d_skip, d, (int)(470 + c), 0.0625f);
    fill_det(z_in, ntd, (int)(480 + c), 0.0625f);
    fill_det(dout, ntd, (int)(490 + c), 0.0625f);

    const float *z_ptr = cases[c].gated ? z_in : NULL;
    float *dz_ptr = cases[c].gated ? dz : NULL;

    dispatching = 1;
    CHECK(et_mamba_causal_conv1d_silu_forward_v1(&cfg, u, conv_w, conv_b,
                                                 x_conv) == ET_MAMBA_OK);
    CHECK(et_mamba_selective_scan_forward_v1(&cfg, x_conv, dt_raw, a_log, b_in,
                                             c_in, d_skip, z_ptr,
                                             out_full) == ET_MAMBA_OK);
    CHECK(et_mamba_selective_scan_backward_v1(
              &cfg, x_conv, dt_raw, a_log, b_in, c_in, d_skip, z_ptr, dout,
              dx_scan, ddt_raw, da_log, db, dc, dd_skip, dz_ptr) ==
          ET_MAMBA_OK);
    CHECK(et_mamba_causal_conv1d_silu_backward_v1(&cfg, u, conv_w, conv_b,
                                                  dx_scan, du, dconv_w,
                                                  dconv_b) == ET_MAMBA_OK);

    et_mamba_cache_v1 cache;
    CHECK(et_mamba_cache_init_v1(&cache, &cfg, conv_state, conv_bytes,
                                 ssm_state, ssm_bytes) == ET_MAMBA_OK);
    for (size_t ti = 0; ti < t; ++ti) {
      for (size_t ni = 0; ni < n; ++ni) {
        memcpy(&u_step[ni * d], &u[(ni * t + ti) * d], d * sizeof(float));
        memcpy(&dt_step[ni * d], &dt_raw[(ni * t + ti) * d], d * sizeof(float));
        memcpy(&z_step[ni * d], &z_in[(ni * t + ti) * d], d * sizeof(float));
        memcpy(&b_step[ni * s], &b_in[(ni * t + ti) * s], s * sizeof(float));
        memcpy(&c_step[ni * s], &c_in[(ni * t + ti) * s], s * sizeof(float));
      }
      const float *z_st_ptr = cases[c].gated ? z_step : NULL;
      CHECK(et_mamba_cache_step_v1(&cache, u_step, conv_w, conv_b, dt_step,
                                   a_log, b_step, c_step, d_skip, z_st_ptr,
                                   out_step) == ET_MAMBA_OK);
      for (size_t ni = 0; ni < n; ++ni)
        memcpy(&out_cached[(ni * t + ti) * d], &out_step[ni * d],
               d * sizeof(float));
    }
    dispatching = 0;

    printf("CASE %zu %zu %zu %zu %zu %d\n", n, t, d, s, k, cases[c].gated);
    print_hex_array(u, ntd);
    print_hex_array(conv_w, dk);
    print_hex_array(conv_b, d);
    print_hex_array(dt_raw, ntd);
    print_hex_array(a_log, ds);
    print_hex_array(b_in, nts);
    print_hex_array(c_in, nts);
    print_hex_array(d_skip, d);
    print_hex_array(z_in, ntd);
    print_hex_array(dout, ntd);

    print_hex_array(x_conv, ntd);
    print_hex_array(out_full, ntd);
    print_hex_array(out_cached, ntd);
    print_hex_array(dx_scan, ntd);
    print_hex_array(ddt_raw, ntd);
    print_hex_array(da_log, ds);
    print_hex_array(db, nts);
    print_hex_array(dc, nts);
    print_hex_array(dd_skip, d);
    if (cases[c].gated)
      print_hex_array(dz, ntd);
    print_hex_array(du, ntd);
    print_hex_array(dconv_w, dk);
    print_hex_array(dconv_b, d);

    free(u);
    free(conv_w);
    free(conv_b);
    free(dt_raw);
    free(a_log);
    free(b_in);
    free(c_in);
    free(d_skip);
    free(z_in);
    free(dout);
    free(x_conv);
    free(out_full);
    free(out_cached);
    free(dx_scan);
    free(ddt_raw);
    free(da_log);
    free(db);
    free(dc);
    free(dd_skip);
    free(dz);
    free(du);
    free(dconv_w);
    free(dconv_b);
    free(u_step);
    free(dt_step);
    free(z_step);
    free(b_step);
    free(c_step);
    free(out_step);
    free(conv_state);
    free(ssm_state);
  }
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_mode();
    return 0;
  }
  test_config_and_o1_state_bytes();
  test_conv1d_silu_fd();
  test_selective_scan_fd();
  test_cache_parity();
  test_negative_contracts();
  printf("MAMBA_SSM native C11 zero-alloc suite: PASS\n");
  return 0;
}
