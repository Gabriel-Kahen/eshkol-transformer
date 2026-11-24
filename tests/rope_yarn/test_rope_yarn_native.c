#include "eshkol_transformer/rope_yarn_abi.h"
#include "rope_yarn_eshkol_bridge.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__x86_64__)
#error "RoPE YaRN native test requires x86-64"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/*
 * Allocation trap linked via -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free.
 * Any heap allocation inside et_rope_yarn_* or et_rope_yarn_aot_* while
 * hot_dispatch != 0 fails immediately.
 */
static int hot_dispatch = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!hot_dispatch);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!hot_dispatch);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!hot_dispatch);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!hot_dispatch);
  __real_free(ptr);
}

typedef struct rope_case {
  uint32_t mode;
  uint32_t layout;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint64_t orig_max_pos;
  double base;
  double scale_factor;
  double beta_fast;
  double beta_slow;
  size_t batch;
  size_t tokens;
  size_t heads;
  int64_t inv_m;
  int64_t inv_n;
  int64_t inv_shift;
} rope_case;

static const rope_case TEST_CASES[] = {
    /* 0: STANDARD, NEOX, full rotary */
    {ET_ROPE_SCALE_STANDARD, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 16u, 16u, 2048u,
     10000.0, 1.0, 32.0, 1.0, 2u, 6u, 3u, 17, 5, 1024},
    /* 1: STANDARD, INTERLEAVED, partial rotary */
    {ET_ROPE_SCALE_STANDARD, ET_ROPE_LAYOUT_INTERLEAVED, 12u, 8u, 2048u,
     500000.0, 1.0, 32.0, 1.0, 1u, 5u, 2u, 42, 11, 512},
    /* 2: LINEAR_PI s=4.0, NEOX, full rotary, extended pos up to 8192 */
    {ET_ROPE_SCALE_LINEAR_PI, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 16u, 16u, 2048u,
     10000.0, 4.0, 32.0, 1.0, 2u, 5u, 2u, 17, 5, 8175},
    /* 3: LINEAR_PI s=2.0, INTERLEAVED, partial rotary */
    {ET_ROPE_SCALE_LINEAR_PI, ET_ROPE_LAYOUT_INTERLEAVED, 14u, 8u, 2048u,
     10000.0, 2.0, 32.0, 1.0, 1u, 6u, 3u, 19, 31, 4000},
    /* 4: NTK_AWARE s=4.0, NEOX, full rotary, extended pos up to 8192 */
    {ET_ROPE_SCALE_NTK_AWARE, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 16u, 16u, 2048u,
     10000.0, 4.0, 32.0, 1.0, 2u, 5u, 2u, 23, 7, 8169},
    /* 5: NTK_AWARE s=8.0, INTERLEAVED, partial rotary (d_rot=4 minimum) */
    {ET_ROPE_SCALE_NTK_AWARE, ET_ROPE_LAYOUT_INTERLEAVED, 8u, 4u, 1024u,
     10000.0, 8.0, 32.0, 1.0, 1u, 6u, 2u, 5, 19, 8100},
    /* 6: YARN s=4.0, NEOX, full rotary (hits all 3 gamma regimes), pos=8192 */
    {ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 16u, 16u, 2048u,
     10000.0, 4.0, 32.0, 1.0, 2u, 6u, 3u, 17, 5, 8175},
    /* 7: YARN s=4.0, INTERLEAVED, partial rotary */
    {ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_INTERLEAVED, 20u, 16u, 2048u, 10000.0,
     4.0, 32.0, 1.0, 1u, 6u, 2u, 13, 29, 8160},
    /* 8: YARN s=8.0, NEOX, large head_dim=64 partial rotary d_rot=32, pos=16384 */
    {ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 64u, 32u, 2048u,
     10000.0, 8.0, 32.0, 1.0, 1u, 4u, 2u, 100, 37, 16284},
    /* 9: YARN s=1.0 boundary (reduces exactly to STANDARD, mscale=1.0) */
    {ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_INTERLEAVED, 16u, 16u, 2048u, 10000.0,
     1.0, 32.0, 1.0, 1u, 4u, 2u, 33, 12, 2000},
    /* 10: YARN custom beta_fast/beta_slow, s=16.0 */
    {ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF, 16u, 12u, 1024u,
     50000.0, 16.0, 48.0, 2.5, 1u, 5u, 2u, 11, 3, 16300},
};

static et_rope_yarn_config_t case_to_config(const rope_case *c) {
  et_rope_yarn_config_t cfg;
  cfg.mode = c->mode;
  cfg.layout = c->layout;
  cfg.head_dim = c->head_dim;
  cfg.rotary_dim = c->rotary_dim;
  cfg.orig_max_pos = c->orig_max_pos;
  cfg.base = c->base;
  cfg.scale_factor = c->scale_factor;
  cfg.beta_fast = c->beta_fast;
  cfg.beta_slow = c->beta_slow;
  return cfg;
}

static void fill_deterministic_f32(float *buf, size_t count, int salt) {
  for (size_t i = 0u; i < count; ++i) {
    const int v = (int)((31u * (unsigned)salt + 17u * (unsigned)i +
                         (unsigned)(i / 3u)) %
                        41u) -
                  20;
    buf[i] = (float)v * 0.0625f;
  }
}

static void fill_positions(int64_t *pos, size_t n, size_t t,
                           const et_rope_yarn_config_t *cfg, int salt) {
  const int64_t max_pos =
      (int64_t)floor((double)cfg->orig_max_pos * cfg->scale_factor);
  for (size_t b = 0u; b < n; ++b) {
    for (size_t i = 0u; i < t; ++i) {
      int64_t p = 0;
      if (i == 0u) {
        p = (b == 0u) ? 0 : (max_pos / 3);
      } else if (i + 1u == t) {
        p = max_pos;
      } else {
        /* Non-contiguous / shifted / extended positions across [0, max_pos]. */
        const int64_t stride = (max_pos > 100) ? (max_pos / (int64_t)t) : 1;
        p = (int64_t)i * stride + (int64_t)((b * 7u + (size_t)salt * 3u) % 19u);
        if (p > max_pos)
          p = max_pos - (int64_t)i;
        if (p < 0)
          p = 0;
      }
      pos[b * t + i] = p;
    }
  }
}

static double dot_f32(const float *a, const float *b, size_t count) {
  double acc = 0.0;
  for (size_t i = 0u; i < count; ++i)
    acc += (double)a[i] * (double)b[i];
  return acc;
}

static void test_s1_equivalence_and_yarn_bands(void) {
  et_rope_yarn_config_t base_cfg = {
      ET_ROPE_SCALE_STANDARD, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF,
      16u,                    16u,
      2048u,                  10000.0,
      1.0,                    32.0,
      1.0};
  double freq_std[8] = {0};
  double mscale_std = 0.0;
  hot_dispatch = 1;
  CHECK(et_rope_yarn_compute_freqs_v1(&base_cfg, freq_std, 8u, &mscale_std) ==
        ET_ROPE_YARN_OK);
  hot_dispatch = 0;
  CHECK(mscale_std == 1.0);

  for (uint32_t m = ET_ROPE_SCALE_LINEAR_PI; m <= ET_ROPE_SCALE_YARN; ++m) {
    et_rope_yarn_config_t cfg = base_cfg;
    cfg.mode = m;
    double freq[8] = {0};
    double mscale = 0.0;
    hot_dispatch = 1;
    CHECK(et_rope_yarn_compute_freqs_v1(&cfg, freq, 8u, &mscale) ==
          ET_ROPE_YARN_OK);
    hot_dispatch = 0;
    CHECK(mscale == 1.0);
    for (size_t k = 0u; k < 8u; ++k)
      CHECK(fabs(freq[k] - freq_std[k]) <= 1e-15);
  }

  /* Verify YaRN s=4.0 hits all 3 frequency regimes (gamma=1, 0<gamma<1, gamma=0). */
  et_rope_yarn_config_t yarn_cfg = base_cfg;
  yarn_cfg.mode = ET_ROPE_SCALE_YARN;
  yarn_cfg.scale_factor = 4.0;
  double freq_yarn[8] = {0};
  double mscale_yarn = 0.0;
  hot_dispatch = 1;
  CHECK(et_rope_yarn_compute_freqs_v1(&yarn_cfg, freq_yarn, 8u, &mscale_yarn) ==
        ET_ROPE_YARN_OK);
  hot_dispatch = 0;

  CHECK(fabs(mscale_yarn - (0.1 * log(4.0) + 1.0)) <= 1e-15);
  /* k=0,1,2: high-freq band (r >= 32 -> gamma == 1.0 -> unscaled). */
  for (size_t k = 0u; k < 3u; ++k)
    CHECK(fabs(freq_yarn[k] - freq_std[k]) <= 1e-15);
  /* k=3,4,5: smooth linear ramp (0 < gamma < 1 -> strictly between orig/s and orig). */
  for (size_t k = 3u; k < 6u; ++k) {
    CHECK(freq_yarn[k] > freq_std[k] / 4.0);
    CHECK(freq_yarn[k] < freq_std[k]);
  }
  /* k=6,7: low-freq band (r <= 1 -> gamma == 0.0 -> full linear PI orig/s). */
  for (size_t k = 6u; k < 8u; ++k)
    CHECK(fabs(freq_yarn[k] - freq_std[k] / 4.0) <= 1e-15);
}

static void test_forward_backward_fd_and_invariance(void) {
  double max_norm_err = 0.0;
  double max_inv_err = 0.0;
  double max_fd_err = 0.0;
  double max_shift_err = 0.0;
  double max_rel_err = 0.0;
  size_t fd_checks = 0u;

  for (size_t c = 0u; c < ARRAY_COUNT(TEST_CASES); ++c) {
    const rope_case *tc = &TEST_CASES[c];
    const et_rope_yarn_config_t cfg = case_to_config(tc);
    const size_t elems = tc->batch * tc->tokens * tc->heads * tc->head_dim;
    const size_t nt = tc->batch * tc->tokens;

    float *x = (float *)malloc(elems * sizeof(float));
    float *dy = (float *)malloc(elems * sizeof(float));
    float *y = (float *)malloc(elems * sizeof(float));
    float *dx = (float *)malloc(elems * sizeof(float));
    float *rt = (float *)malloc(elems * sizeof(float));
    float *xp = (float *)malloc(elems * sizeof(float));
    float *yp = (float *)malloc(elems * sizeof(float));
    float *ym = (float *)malloc(elems * sizeof(float));
    int64_t *pos = (int64_t *)malloc(nt * sizeof(int64_t));
    CHECK(x && dy && y && dx && rt && xp && yp && ym && pos);

    fill_deterministic_f32(x, elems, (int)(c * 3u + 1u));
    fill_deterministic_f32(dy, elems, (int)(c * 3u + 2u));
    fill_positions(pos, tc->batch, tc->tokens, &cfg, (int)c);

    double inv_freq[ET_ROPE_YARN_MAX_HEAD_DIM / 2u];
    double mscale = 0.0;

    hot_dispatch = 1;
    CHECK(et_rope_yarn_compute_freqs_v1(&cfg, inv_freq, tc->rotary_dim / 2u,
                                        &mscale) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, x,
                                  pos, y) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_backward_v1(&cfg, tc->batch, tc->tokens, tc->heads, dy,
                                   pos, dx) == ET_ROPE_YARN_OK);
    /* Roundtrip: backward(forward(x)) == mscale^2 * x_rot (and x_tail unchanged). */
    CHECK(et_rope_yarn_backward_v1(&cfg, tc->batch, tc->tokens, tc->heads, y,
                                   pos, rt) == ET_ROPE_YARN_OK);
    hot_dispatch = 0;

    /* Verify partial rotary tail exactness and norm scaling. */
    const double m2 = mscale * mscale;
    for (size_t n = 0u; n < tc->batch; ++n) {
      for (size_t t = 0u; t < tc->tokens; ++t) {
        for (size_t h = 0u; h < tc->heads; ++h) {
          const size_t base =
              ((n * tc->tokens + t) * tc->heads + h) * tc->head_dim;
          double x_rot_sq = 0.0;
          double y_rot_sq = 0.0;
          for (size_t d = 0u; d < tc->rotary_dim; ++d) {
            x_rot_sq += (double)x[base + d] * (double)x[base + d];
            y_rot_sq += (double)y[base + d] * (double)y[base + d];
            const double inv_err = fabs((double)rt[base + d] - m2 * x[base + d]);
            if (inv_err > max_inv_err)
              max_inv_err = inv_err;
            CHECK(inv_err <= 2e-5);
          }
          const double norm_err = fabs(y_rot_sq - m2 * x_rot_sq);
          if (norm_err > max_norm_err)
            max_norm_err = norm_err;
          CHECK(norm_err <= 5e-5);

          /* Trailing unrotated channels must be bit-for-bit identical! */
          for (size_t d = tc->rotary_dim; d < tc->head_dim; ++d) {
            CHECK(y[base + d] == x[base + d]);
            CHECK(dx[base + d] == dy[base + d]);
            CHECK(rt[base + d] == x[base + d]);
          }
        }
      }
    }

    /* Central finite-difference gradient check on sampled rotary + tail indices. */
    memcpy(xp, x, elems * sizeof(float));
    const float eps = 1.0f / 256.0f;
    const size_t step = (elems > 37u) ? (elems / 37u) : 1u;
    for (size_t idx = 0u; idx < elems; idx += step) {
      const float orig = xp[idx];
      xp[idx] = orig + eps;
      hot_dispatch = 1;
      CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, xp,
                                    pos, yp) == ET_ROPE_YARN_OK);
      xp[idx] = orig - eps;
      CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, xp,
                                    pos, ym) == ET_ROPE_YARN_OK);
      hot_dispatch = 0;
      xp[idx] = orig;

      const double lp = dot_f32(yp, dy, elems);
      const double lm = dot_f32(ym, dy, elems);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double ana = (double)dx[idx];
      const double err = fabs(ana - num);
      if (err > max_fd_err)
        max_fd_err = err;
      CHECK(err <= 5e-4);
      ++fd_checks;
    }
    /* Always also check the very last element (covers partial rotary tail!). */
    {
      const size_t idx = elems - 1u;
      const float orig = xp[idx];
      xp[idx] = orig + eps;
      hot_dispatch = 1;
      CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, xp,
                                    pos, yp) == ET_ROPE_YARN_OK);
      xp[idx] = orig - eps;
      CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, xp,
                                    pos, ym) == ET_ROPE_YARN_OK);
      hot_dispatch = 0;
      xp[idx] = orig;
      const double num =
          (dot_f32(yp, dy, elems) - dot_f32(ym, dy, elems)) / (2.0 * eps);
      const double err = fabs((double)dx[idx] - num);
      if (err > max_fd_err)
        max_fd_err = err;
      CHECK(err <= 5e-4);
      ++fd_checks;
    }

    /* Relative-position QK dot-product invariance check. */
    et_rope_yarn_invariance_result_t inv;
    memset(&inv, 0, sizeof(inv));
    hot_dispatch = 1;
    CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&cfg, x, dy, tc->inv_m,
                                                  tc->inv_n, tc->inv_shift,
                                                  &inv) == ET_ROPE_YARN_OK);
    hot_dispatch = 0;
    CHECK(fabs(inv.mscale - mscale) <= 1e-15);
    CHECK(inv.shift_abs_err <= 2e-5);
    CHECK(inv.relative_abs_err <= 2e-5);
    if (inv.shift_abs_err > max_shift_err)
      max_shift_err = inv.shift_abs_err;
    if (inv.relative_abs_err > max_rel_err)
      max_rel_err = inv.relative_abs_err;

    /* Also verify AOT bridge matches core bit-for-bit with zero allocations! */
    float *y_aot = (float *)malloc(elems * sizeof(float));
    float *dx_aot = (float *)malloc(elems * sizeof(float));
    double metrics_aot[6] = {0};
    CHECK(y_aot && dx_aot);
    hot_dispatch = 1;
    CHECK(et_rope_yarn_aot_forward(
              (int64_t)cfg.mode, (int64_t)cfg.layout, (int64_t)cfg.head_dim,
              (int64_t)cfg.rotary_dim, (int64_t)cfg.orig_max_pos, cfg.base,
              cfg.scale_factor, cfg.beta_fast, cfg.beta_slow,
              (int64_t)tc->batch, (int64_t)tc->tokens, (int64_t)tc->heads, x,
              pos, y_aot) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_aot_backward(
              (int64_t)cfg.mode, (int64_t)cfg.layout, (int64_t)cfg.head_dim,
              (int64_t)cfg.rotary_dim, (int64_t)cfg.orig_max_pos, cfg.base,
              cfg.scale_factor, cfg.beta_fast, cfg.beta_slow,
              (int64_t)tc->batch, (int64_t)tc->tokens, (int64_t)tc->heads, dy,
              pos, dx_aot) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_aot_qk_invariance(
              (int64_t)cfg.mode, (int64_t)cfg.layout, (int64_t)cfg.head_dim,
              (int64_t)cfg.rotary_dim, (int64_t)cfg.orig_max_pos, cfg.base,
              cfg.scale_factor, cfg.beta_fast, cfg.beta_slow, x, dy, tc->inv_m,
              tc->inv_n, tc->inv_shift, metrics_aot) == ET_ROPE_YARN_OK);
    hot_dispatch = 0;

    CHECK(memcmp(y, y_aot, elems * sizeof(float)) == 0);
    CHECK(memcmp(dx, dx_aot, elems * sizeof(float)) == 0);
    CHECK(metrics_aot[0] == inv.dot_mn && metrics_aot[1] == inv.dot_shifted &&
          metrics_aot[2] == inv.dot_relative_ref &&
          metrics_aot[5] == inv.mscale);

    free(y_aot);
    free(dx_aot);
    free(x);
    free(dy);
    free(y);
    free(dx);
    free(rt);
    free(xp);
    free(yp);
    free(ym);
    free(pos);
  }

  printf("RoPE YaRN native forward/backward/FD/invariance: PASS "
         "(cases=%zu, fd_checks=%zu, max_norm_err=%.3e, max_inv_err=%.3e, "
         "max_fd_err=%.3e, max_shift_err=%.3e, max_rel_err=%.3e)\n",
         ARRAY_COUNT(TEST_CASES), fd_checks, max_norm_err, max_inv_err,
         max_fd_err, max_shift_err, max_rel_err);
}

static void check_sentinel_untouched(const unsigned char *buf,
                                     const unsigned char *ref, size_t bytes) {
  CHECK(memcmp(buf, ref, bytes) == 0);
}

static void test_negative_and_fpenv(void) {
  et_rope_yarn_config_t good = {
      ET_ROPE_SCALE_YARN, ET_ROPE_LAYOUT_NEOX_SPLIT_HALF,
      16u,                16u,
      2048u,              10000.0,
      4.0,                32.0,
      1.0};

  float x[64];
  float y[64];
  float y_ref[64];
  int64_t pos[4] = {0, 100, 4096, 8192};
  double freqs[8];
  double freqs_ref[8];
  double mscale = 99.25;
  double mscale_ref = 99.25;
  et_rope_yarn_invariance_result_t inv;
  et_rope_yarn_invariance_result_t inv_ref;

  fill_deterministic_f32(x, 64u, 7);
  memset(y, 0xA5, sizeof(y));
  memcpy(y_ref, y, sizeof(y));
  memset(freqs, 0x5A, sizeof(freqs));
  memcpy(freqs_ref, freqs, sizeof(freqs));
  memset(&inv, 0x3C, sizeof(inv));
  memcpy(&inv_ref, &inv, sizeof(inv));

  hot_dispatch = 1;

  /* 1. NULL pointers */
  CHECK(et_rope_yarn_validate_config_v1(NULL) == ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_compute_freqs_v1(NULL, freqs, 8u, &mscale) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_compute_freqs_v1(&good, NULL, 8u, &mscale) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_compute_freqs_v1(&good, freqs, 8u, NULL) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, NULL, pos, y) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, NULL, y) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, pos, NULL) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&good, NULL, x, 1, 0, 10,
                                                &inv) ==
        ET_ROPE_YARN_ERR_ARGUMENT);
  CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&good, x, x, 1, 0, 10, NULL) ==
        ET_ROPE_YARN_ERR_ARGUMENT);

  /* 2. Invalid mode / layout / base / scale / beta / orig_max_pos */
  et_rope_yarn_config_t bad = good;
  bad.mode = 4u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.layout = 2u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.scale_factor = 0.5;
  CHECK(et_rope_yarn_compute_freqs_v1(&bad, freqs, 8u, &mscale) ==
        ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.mode = ET_ROPE_SCALE_STANDARD;
  bad.scale_factor = 2.0; /* STANDARD forbids silent ignoring of s != 1.0 */
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.base = 1.0;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);
  bad.base = NAN;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.beta_fast = 1.0;
  bad.beta_slow = 1.0;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);
  bad.beta_fast = 0.5;
  bad.beta_slow = 1.0;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);
  bad.beta_fast = 32.0;
  bad.beta_slow = 0.0;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);
  bad.beta_slow = -1.0;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  bad = good;
  bad.orig_max_pos = 0u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_ARGUMENT);

  /* 3. Invalid shapes: odd rotary_dim, rotary_dim > head_dim, NTK d_rot=2, bad dims */
  bad = good;
  bad.rotary_dim = 7u;
  CHECK(et_rope_yarn_forward_v1(&bad, 1u, 4u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_SHAPE);

  bad = good;
  bad.rotary_dim = 18u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_SHAPE);

  bad = good;
  bad.head_dim = 1u;
  bad.rotary_dim = 2u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_SHAPE);

  bad = good;
  bad.head_dim = ET_ROPE_YARN_MAX_HEAD_DIM + 2u;
  CHECK(et_rope_yarn_validate_config_v1(&bad) == ET_ROPE_YARN_ERR_SHAPE);

  bad = good;
  bad.mode = ET_ROPE_SCALE_NTK_AWARE;
  bad.head_dim = 4u;
  bad.rotary_dim = 2u; /* NTK requires d_rot >= 4 */
  CHECK(et_rope_yarn_compute_freqs_v1(&bad, freqs, 1u, &mscale) ==
        ET_ROPE_YARN_ERR_SHAPE);

  CHECK(et_rope_yarn_compute_freqs_v1(&good, freqs, 7u, &mscale) ==
        ET_ROPE_YARN_ERR_SHAPE);
  CHECK(et_rope_yarn_forward_v1(&good, 0u, 4u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_SHAPE);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 0u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_SHAPE);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 0u, x, pos, y) ==
        ET_ROPE_YARN_ERR_SHAPE);

  /* 4. Out-of-range positions (negative, > orig_max_pos * s, negative shift, overflow) */
  int64_t bad_pos[4] = {0, 100, -1, 8192};
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, bad_pos, y) ==
        ET_ROPE_YARN_ERR_POSITION);
  bad_pos[2] = 8193; /* > 2048 * 4 */
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, bad_pos, y) ==
        ET_ROPE_YARN_ERR_POSITION);
  CHECK(et_rope_yarn_backward_v1(&good, 1u, 4u, 1u, x, bad_pos, y) ==
        ET_ROPE_YARN_ERR_POSITION);
  CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&good, x, x, 10, 5, -1, &inv) ==
        ET_ROPE_YARN_ERR_POSITION);
  CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&good, x, x, 10, 5, 8183,
                                                &inv) ==
        ET_ROPE_YARN_ERR_POSITION);

  /* 5. Misaligned raw pointers */
  unsigned char raw_buf[512];
  memset(raw_buf, 0, sizeof(raw_buf));
  CHECK(et_rope_yarn_forward_raw_v1(&good, 1u, 4u, 1u, raw_buf + 1, pos, y) ==
        ET_ROPE_YARN_ERR_ALIGNMENT);
  CHECK(et_rope_yarn_forward_raw_v1(&good, 1u, 4u, 1u, x, raw_buf + 4, y) ==
        ET_ROPE_YARN_ERR_ALIGNMENT);
  CHECK(et_rope_yarn_forward_raw_v1(&good, 1u, 4u, 1u, x, pos, raw_buf + 2) ==
        ET_ROPE_YARN_ERR_ALIGNMENT);
  CHECK(et_rope_yarn_compute_freqs_raw_v1(&good, raw_buf + 4, 8u, &mscale) ==
        ET_ROPE_YARN_ERR_ALIGNMENT);
  CHECK(et_rope_yarn_compute_freqs_raw_v1(&good, freqs, 8u, raw_buf + 4) ==
        ET_ROPE_YARN_ERR_ALIGNMENT);
  CHECK(et_rope_yarn_qk_dot_invariance_check_raw_v1(
            &good, x, x, 10, 5, 100, raw_buf + 4) == ET_ROPE_YARN_ERR_ALIGNMENT);

  /* 6. Aliasing / overlapping spans */
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, pos, x) ==
        ET_ROPE_YARN_ERR_ALIAS);
  CHECK(et_rope_yarn_backward_v1(&good, 1u, 4u, 1u, x, pos, x) ==
        ET_ROPE_YARN_ERR_ALIAS);
  /* Partial overlap */
  float big[80];
  fill_deterministic_f32(big, 80u, 13);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, big, pos, big + 4) ==
        ET_ROPE_YARN_ERR_ALIAS);
  CHECK(et_rope_yarn_compute_freqs_raw_v1(&good, freqs, 8u, &freqs[7]) ==
        ET_ROPE_YARN_ERR_ALIAS);
  CHECK(et_rope_yarn_qk_dot_invariance_check_raw_v1(
            &good, (const float *)&inv, x, 10, 5, 100, &inv) ==
        ET_ROPE_YARN_ERR_ALIAS);

  /* 7. Non-finite inputs (including LAST element -> failure-atomic!) & f32 overflow */
  float bad_x[64];
  memcpy(bad_x, x, sizeof(x));
  bad_x[63] = NAN;
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, bad_x, pos, y) ==
        ET_ROPE_YARN_ERR_NONFINITE);
  bad_x[63] = INFINITY;
  CHECK(et_rope_yarn_backward_v1(&good, 1u, 4u, 1u, bad_x, pos, y) ==
        ET_ROPE_YARN_ERR_NONFINITE);
  /* YaRN mscale > 1.0 multiplies FLT_MAX -> overflows binary32 -> rejected! */
  memcpy(bad_x, x, sizeof(x));
  bad_x[0] = FLT_MAX;
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, bad_x, pos, y) ==
        ET_ROPE_YARN_ERR_NONFINITE);
  /* Non-finite element in partial-rotary unrotated tail must also be rejected! */
  et_rope_yarn_config_t part = good;
  part.rotary_dim = 8u;
  memcpy(bad_x, x, sizeof(x));
  bad_x[63] = -INFINITY;
  CHECK(et_rope_yarn_forward_v1(&part, 1u, 4u, 1u, bad_x, pos, y) ==
        ET_ROPE_YARN_ERR_NONFINITE);

  hot_dispatch = 0;

  /* Verify every single output sentinel remained byte-for-byte untouched! */
  check_sentinel_untouched((const unsigned char *)y,
                           (const unsigned char *)y_ref, sizeof(y));
  check_sentinel_untouched((const unsigned char *)freqs,
                           (const unsigned char *)freqs_ref, sizeof(freqs));
  CHECK(memcmp(&mscale, &mscale_ref, sizeof(mscale)) == 0);
  check_sentinel_untouched((const unsigned char *)&inv,
                           (const unsigned char *)&inv_ref, sizeof(inv));

  /* 8. Non-standard x87/MXCSR FP environments must be rejected and restored. */
  fenv_t saved;
  CHECK(fegetenv(&saved) == 0);

  /* Non-RN rounding mode */
  CHECK(fesetround(FE_UPWARD) == 0);
  errno = EAGAIN;
  hot_dispatch = 1;
  CHECK(et_rope_yarn_validate_config_v1(&good) == ET_ROPE_YARN_ERR_FP_ENV);
  CHECK(et_rope_yarn_compute_freqs_v1(&good, freqs, 8u, &mscale) ==
        ET_ROPE_YARN_ERR_FP_ENV);
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_FP_ENV);
  hot_dispatch = 0;
  CHECK(errno == EAGAIN);
  CHECK(fesetenv(&saved) == 0);

  /* FTZ bit in MXCSR */
  const unsigned int orig_mxcsr = _mm_getcsr();
  _mm_setcsr(orig_mxcsr | 0x8000u);
  hot_dispatch = 1;
  CHECK(et_rope_yarn_forward_v1(&good, 1u, 4u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_FP_ENV);
  hot_dispatch = 0;
  _mm_setcsr(orig_mxcsr);

  /* DAZ bit in MXCSR */
  _mm_setcsr(orig_mxcsr | 0x0040u);
  hot_dispatch = 1;
  CHECK(et_rope_yarn_backward_v1(&good, 1u, 4u, 1u, x, pos, y) ==
        ET_ROPE_YARN_ERR_FP_ENV);
  hot_dispatch = 0;
  _mm_setcsr(orig_mxcsr);

  check_sentinel_untouched((const unsigned char *)y,
                           (const unsigned char *)y_ref, sizeof(y));

  printf("RoPE YaRN native negative & FP-env & failure-atomicity: PASS\n");
}

static void emit_oracle(void) {
  for (size_t c = 0u; c < ARRAY_COUNT(TEST_CASES); ++c) {
    const rope_case *tc = &TEST_CASES[c];
    const et_rope_yarn_config_t cfg = case_to_config(tc);
    const size_t elems = tc->batch * tc->tokens * tc->heads * tc->head_dim;
    const size_t nt = tc->batch * tc->tokens;
    const size_t half = tc->rotary_dim / 2u;

    float *x = (float *)malloc(elems * sizeof(float));
    float *dy = (float *)malloc(elems * sizeof(float));
    float *y = (float *)malloc(elems * sizeof(float));
    float *dx = (float *)malloc(elems * sizeof(float));
    int64_t *pos = (int64_t *)malloc(nt * sizeof(int64_t));
    CHECK(x && dy && y && dx && pos);

    fill_deterministic_f32(x, elems, (int)(c * 5u + 3u));
    fill_deterministic_f32(dy, elems, (int)(c * 5u + 4u));
    fill_positions(pos, tc->batch, tc->tokens, &cfg, (int)c);

    double inv_freq[ET_ROPE_YARN_MAX_HEAD_DIM / 2u];
    double mscale = 0.0;
    et_rope_yarn_invariance_result_t inv;
    memset(&inv, 0, sizeof(inv));

    hot_dispatch = 1;
    CHECK(et_rope_yarn_compute_freqs_v1(&cfg, inv_freq, half, &mscale) ==
          ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_forward_v1(&cfg, tc->batch, tc->tokens, tc->heads, x,
                                  pos, y) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_backward_v1(&cfg, tc->batch, tc->tokens, tc->heads, dy,
                                   pos, dx) == ET_ROPE_YARN_OK);
    CHECK(et_rope_yarn_qk_dot_invariance_check_v1(&cfg, x, dy, tc->inv_m,
                                                  tc->inv_n, tc->inv_shift,
                                                  &inv) == ET_ROPE_YARN_OK);
    hot_dispatch = 0;

    printf("CASE %u %u %u %u %llu %a %a %a %a %zu %zu %zu\n", cfg.mode,
           cfg.layout, cfg.head_dim, cfg.rotary_dim,
           (unsigned long long)cfg.orig_max_pos, cfg.base, cfg.scale_factor,
           cfg.beta_fast, cfg.beta_slow, tc->batch, tc->tokens, tc->heads);

    for (size_t k = 0u; k < half; ++k)
      printf("%s%a", k ? " " : "", inv_freq[k]);
    printf(" %a\n", mscale);

    for (size_t i = 0u; i < nt; ++i)
      printf("%s%lld", i ? " " : "", (long long)pos[i]);
    printf("\n");

    for (size_t i = 0u; i < elems; ++i)
      printf("%s%a", i ? " " : "", (double)x[i]);
    printf("\n");

    for (size_t i = 0u; i < elems; ++i)
      printf("%s%a", i ? " " : "", (double)dy[i]);
    printf("\n");

    for (size_t i = 0u; i < elems; ++i)
      printf("%s%a", i ? " " : "", (double)y[i]);
    printf("\n");

    for (size_t i = 0u; i < elems; ++i)
      printf("%s%a", i ? " " : "", (double)dx[i]);
    printf("\n");

    printf("INV %lld %lld %lld %a %a %a %a %a %a\n", (long long)tc->inv_m,
           (long long)tc->inv_n, (long long)tc->inv_shift, inv.dot_mn,
           inv.dot_shifted, inv.dot_relative_ref, inv.shift_abs_err,
           inv.relative_abs_err, inv.mscale);

    free(x);
    free(dy);
    free(y);
    free(dx);
    free(pos);
  }

  /* Emit negative status witnesses for independent oracle verification. */
  et_rope_yarn_config_t cfg = {ET_ROPE_SCALE_YARN,
                               ET_ROPE_LAYOUT_NEOX_SPLIT_HALF,
                               16u,
                               16u,
                               2048u,
                               10000.0,
                               4.0,
                               32.0,
                               1.0};
  float buf[16] = {0};
  float out[16] = {0};
  int64_t pos[1] = {0};

  et_rope_yarn_config_t c1 = cfg;
  c1.rotary_dim = 5u;
  printf("NEG odd_d_rot %d\n",
         et_rope_yarn_forward_v1(&c1, 1u, 1u, 1u, buf, pos, out));

  et_rope_yarn_config_t c2 = cfg;
  c2.scale_factor = 0.75;
  printf("NEG scale_lt_1 %d\n",
         et_rope_yarn_forward_v1(&c2, 1u, 1u, 1u, buf, pos, out));

  et_rope_yarn_config_t c3 = cfg;
  c3.beta_fast = 1.0;
  c3.beta_slow = 2.0;
  printf("NEG beta_order %d\n",
         et_rope_yarn_forward_v1(&c3, 1u, 1u, 1u, buf, pos, out));

  et_rope_yarn_config_t c4 = cfg;
  c4.mode = ET_ROPE_SCALE_NTK_AWARE;
  c4.rotary_dim = 2u;
  printf("NEG ntk_d_rot_2 %d\n",
         et_rope_yarn_forward_v1(&c4, 1u, 1u, 1u, buf, pos, out));

  printf("NEG alias %d\n",
         et_rope_yarn_forward_v1(&cfg, 1u, 1u, 1u, buf, pos, buf));

  buf[0] = NAN;
  printf("NEG nonfinite %d\n",
         et_rope_yarn_forward_v1(&cfg, 1u, 1u, 1u, buf, pos, out));

  buf[0] = 0.0f;
  pos[0] = 8193;
  printf("NEG pos_oob %d\n",
         et_rope_yarn_forward_v1(&cfg, 1u, 1u, 1u, buf, pos, out));
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    emit_oracle();
    return EXIT_SUCCESS;
  }
  CHECK(argc == 1);
  test_s1_equivalence_and_yarn_bands();
  test_forward_backward_fd_and_invariance();
  test_negative_and_fpenv();
  return EXIT_SUCCESS;
}
