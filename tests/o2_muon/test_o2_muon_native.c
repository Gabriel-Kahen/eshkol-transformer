#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/o2_muon_abi.h"
#include "o2_muon_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));        \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static float det_matrix_val(uint32_t r, uint32_t c, uint32_t cols,
                            uint32_t salt) {
  const uint32_t idx = r * cols + c;
  const int32_t bucket =
      (int32_t)((idx * 31u + salt * 17u + (idx / 5u) * 13u) % 41u) - 20;
  float v = (float)bucket * 0.03125f;
  if (r == c)
    v += 0.75f;
  return v;
}

static void fill_matrix(float *dst, uint32_t rows, uint32_t cols,
                        uint32_t salt) {
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c)
      dst[(size_t)r * cols + c] = det_matrix_val(r, c, cols, salt);
  }
}

static void print_f32_hex_vec(const float *data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)data[i]);
  }
  putchar('\n');
}

/* ---- Oracle emission mode (checked by tests/o2_muon/test_o2_muon_oracle.py) ---- */

static int emit_oracle(void) {
  /* 1. Newton-Schulz5 cases: square, tall, wide, 1x1, zero matrix, 3-step. */
  const struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t ns_steps;
    uint32_t salt;
    int zero;
  } ns_cases[] = {
      {4u, 4u, 5u, 11u, 0},   {8u, 4u, 5u, 19u, 0},  {4u, 8u, 5u, 23u, 0},
      {12u, 6u, 5u, 29u, 0},  {6u, 12u, 5u, 37u, 0}, {1u, 1u, 5u, 41u, 0},
      {5u, 5u, 3u, 43u, 0},   {4u, 6u, 5u, 0u, 1},
  };

  for (size_t i = 0; i < sizeof(ns_cases) / sizeof(ns_cases[0]); ++i) {
    const uint32_t rows = ns_cases[i].rows;
    const uint32_t cols = ns_cases[i].cols;
    const size_t n = (size_t)rows * (size_t)cols;
    float *g = (float *)calloc(n, sizeof(float));
    float *u = (float *)calloc(n, sizeof(float));
    if (g == NULL || u == NULL)
      return 2;
    if (!ns_cases[i].zero)
      fill_matrix(g, rows, cols, ns_cases[i].salt);
    const double eps = 1e-7;
    if (et_muon_newton_schulz5_f32(rows, cols, g, ns_cases[i].ns_steps, eps,
                                   u) != ET_MUON_OK)
      return 3;
    double ortho = 0.0;
    if (et_muon_orthogonality_error_f32(rows, cols, u, &ortho) != ET_MUON_OK)
      return 4;
    printf("NS5 %u %u %u %a %a\n", rows, cols, ns_cases[i].ns_steps, eps,
           ortho);
    print_f32_hex_vec(g, n);
    print_f32_hex_vec(u, n);
    free(g);
    free(u);
  }

  /* 2. WSD schedule cases: linear and cosine sweeps covering every boundary. */
  const et_wsd_config wsd_cases[] = {
      {4u, 5u, 14u, 0.05, 0.005, ET_WSD_DECAY_LINEAR, 0u},
      {4u, 5u, 14u, 0.04, 0.004, ET_WSD_DECAY_COSINE, 0u},
      {0u, 3u, 8u, 0.02, 0.001, ET_WSD_DECAY_COSINE, 0u},
      {3u, 0u, 9u, 0.03, 0.003, ET_WSD_DECAY_LINEAR, 0u},
      {3u, 4u, 7u, 0.01, 0.002, ET_WSD_DECAY_LINEAR, 0u},
  };
  for (size_t i = 0; i < sizeof(wsd_cases) / sizeof(wsd_cases[0]); ++i) {
    const et_wsd_config *cfg = &wsd_cases[i];
    const uint64_t end_step = cfg->total_steps + 3u;
    printf("WSD %llu %llu %llu %a %a %u %llu\n",
           (unsigned long long)cfg->warmup_steps,
           (unsigned long long)cfg->stable_steps,
           (unsigned long long)cfg->total_steps, cfg->max_lr, cfg->min_lr,
           cfg->decay_kind, (unsigned long long)end_step);
    for (uint64_t step = 0; step <= end_step; ++step) {
      double lr = 0.0;
      uint32_t phase = 0u;
      if (et_wsd_lr(cfg, step, &lr, &phase) != ET_MUON_OK)
        return 5;
      printf("WSD_PT %llu %a %u\n", (unsigned long long)step, lr, phase);
    }
  }

  /* 3. Hybrid Muon + AdamW multi-step oracle (Nesterov=1 and Nesterov=0). */
  for (uint32_t nesterov = 0u; nesterov <= 1u; ++nesterov) {
    float w_wide[24], g_wide[24], b_wide[24];
    float w_tall[24], g_tall[24], b_tall[24];
    float w_norm[6], g_norm[6], m_norm[6], v_norm[6];
    float w_vec[6], g_vec[6], m_vec[6], v_vec[6];

    fill_matrix(w_wide, 4u, 6u, 5u + nesterov);
    fill_matrix(w_tall, 6u, 4u, 9u + nesterov);
    for (uint32_t i = 0; i < 6u; ++i) {
      w_norm[i] = 1.0f + 0.0625f * (float)i;
      w_vec[i] = -0.25f + 0.125f * (float)i;
    }
    memset(b_wide, 0, sizeof(b_wide));
    memset(b_tall, 0, sizeof(b_tall));
    memset(m_norm, 0, sizeof(m_norm));
    memset(v_norm, 0, sizeof(v_norm));
    memset(m_vec, 0, sizeof(m_vec));
    memset(v_vec, 0, sizeof(v_vec));

    const et_wsd_config wsd = {2u, 2u, 6u, 0.04, 0.008, ET_WSD_DECAY_COSINE,
                               0u};
    const et_muon_step_config base_cfg = {
        0.04, 1u, 0.9, 1e-7, 0.01, 5u, nesterov, 0.9, 0.999, 1e-8, 0.02, 0.5};

    printf("HYBRID %u %a %a %a %u %a %a %a %a %a %llu %llu %llu %a %a %u 4\n",
           nesterov, base_cfg.muon_beta, base_cfg.muon_eps,
           base_cfg.muon_weight_decay, base_cfg.muon_ns_steps,
           base_cfg.adamw_beta1, base_cfg.adamw_beta2, base_cfg.adamw_eps,
           base_cfg.adamw_weight_decay, base_cfg.adamw_lr_scale,
           (unsigned long long)wsd.warmup_steps,
           (unsigned long long)wsd.stable_steps,
           (unsigned long long)wsd.total_steps, wsd.max_lr, wsd.min_lr,
           wsd.decay_kind);
    print_f32_hex_vec(w_wide, 24u);
    print_f32_hex_vec(w_tall, 24u);
    print_f32_hex_vec(w_norm, 6u);
    print_f32_hex_vec(w_vec, 6u);

    const et_muon_param_group groups[4] = {
        {ET_MUON_GROUP_MUON_2D, 4u, 6u, 1u, w_wide, g_wide, b_wide, NULL},
        {ET_MUON_GROUP_MUON_2D, 6u, 4u, 0u, w_tall, g_tall, b_tall, NULL},
        {ET_MUON_GROUP_ADAMW_1D, 6u, 1u, 0u, w_norm, g_norm, m_norm, v_norm},
        {ET_MUON_GROUP_ADAMW_1D, 6u, 1u, 1u, w_vec, g_vec, m_vec, v_vec},
    };

    for (uint64_t step = 0; step < 4u; ++step) {
      fill_matrix(g_wide, 4u, 6u, (uint32_t)(51u + step * 7u + nesterov));
      fill_matrix(g_tall, 6u, 4u, (uint32_t)(71u + step * 11u + nesterov));
      for (uint32_t i = 0; i < 6u; ++i) {
        g_norm[i] = det_matrix_val(i, 0u, 1u, (uint32_t)(91u + step));
        g_vec[i] = det_matrix_val(i, 0u, 1u, (uint32_t)(103u + step));
      }
      print_f32_hex_vec(g_wide, 24u);
      print_f32_hex_vec(g_tall, 24u);
      print_f32_hex_vec(g_norm, 6u);
      print_f32_hex_vec(g_vec, 6u);

      double lr = 0.0;
      uint32_t phase = 0u;
      if (et_muon_hybrid_step_wsd(&wsd, step, &base_cfg, 4u, groups, &lr,
                                  &phase) != ET_MUON_OK)
        return 6;
      printf("STEP_OUT %llu %a %u\n", (unsigned long long)step, lr, phase);
      print_f32_hex_vec(w_wide, 24u);
      print_f32_hex_vec(b_wide, 24u);
      print_f32_hex_vec(w_tall, 24u);
      print_f32_hex_vec(b_tall, 24u);
      print_f32_hex_vec(w_norm, 6u);
      print_f32_hex_vec(m_norm, 6u);
      print_f32_hex_vec(v_norm, 6u);
      print_f32_hex_vec(w_vec, 6u);
      print_f32_hex_vec(m_vec, 6u);
      print_f32_hex_vec(v_vec, 6u);
    }
  }
  return 0;
}

/* ---- Unit tests ---- */

static void test_newton_schulz5_shapes(void) {
  const struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t salt;
  } shapes[] = {
      {1u, 1u, 3u},   {4u, 4u, 7u},   {16u, 16u, 13u}, {64u, 64u, 17u},
      {12u, 4u, 21u}, {32u, 8u, 29u}, {4u, 12u, 31u},  {8u, 32u, 37u},
  };

  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
    const uint32_t rows = shapes[i].rows;
    const uint32_t cols = shapes[i].cols;
    const uint32_t r = rows < cols ? rows : cols;
    const size_t n = (size_t)rows * (size_t)cols;
    float *g = (float *)calloc(n, sizeof(float));
    float *u = (float *)calloc(n, sizeof(float));
    CHECK(g != NULL && u != NULL, "alloc NS5 shape buffers");
    fill_matrix(g, rows, cols, shapes[i].salt);

    CHECK(et_muon_newton_schulz5_f32(rows, cols, g, ET_MUON_DEFAULT_NS_STEPS,
                                     1e-7, u) == ET_MUON_OK,
          "NS5 succeeds on valid square/tall/wide matrix");

    for (size_t k = 0; k < n; ++k)
      CHECK(isfinite(u[k]), "NS5 output entry is finite");

    double ortho_err = -1.0;
    CHECK(et_muon_orthogonality_error_f32(rows, cols, u, &ortho_err) ==
              ET_MUON_OK,
          "orthogonality error succeeds");
    /* Quintic Muon coeffs (3.4445, -4.7750, 2.0315) drive singular values into
     * ~[0.7, 1.15], so ||U U^T - I||_F <= 0.65 * sqrt(min(rows, cols)). */
    const double bound = 0.65 * sqrt((double)r);
    CHECK(isfinite(ortho_err) && ortho_err >= 0.0 && ortho_err <= bound,
          "NS5 orthogonality error is finite and bounded");

    /* Verify transpose symmetry: NS5(G^T) == NS5(G)^T closely. */
    float *gt = (float *)calloc(n, sizeof(float));
    float *ut = (float *)calloc(n, sizeof(float));
    for (uint32_t a = 0; a < rows; ++a)
      for (uint32_t b = 0; b < cols; ++b)
        gt[(size_t)b * rows + a] = g[(size_t)a * cols + b];
    CHECK(et_muon_newton_schulz5_f32(cols, rows, gt, ET_MUON_DEFAULT_NS_STEPS,
                                     1e-7, ut) == ET_MUON_OK,
          "NS5 transpose succeeds");
    double max_t_diff = 0.0;
    for (uint32_t a = 0; a < rows; ++a) {
      for (uint32_t b = 0; b < cols; ++b) {
        const double d = fabs((double)u[(size_t)a * cols + b] -
                              (double)ut[(size_t)b * rows + a]);
        if (d > max_t_diff)
          max_t_diff = d;
      }
    }
    CHECK(max_t_diff <= 1e-6, "NS5(G^T) matches NS5(G)^T");
    free(gt);
    free(ut);
    free(g);
    free(u);
  }
}

static void test_zero_gradient_safety(void) {
  float g[12] = {0.0f};
  float u[12];
  for (size_t i = 0; i < 12u; ++i)
    u[i] = 9.0f;

  CHECK(et_muon_newton_schulz5_f32(3u, 4u, g, 5u, 1e-7, u) == ET_MUON_OK,
        "NS5 accepts all-zero matrix safely");
  for (size_t i = 0; i < 12u; ++i)
    CHECK(u[i] == 0.0f, "NS5 of zero matrix is exact zero");

  double ortho_err = -1.0;
  CHECK(et_muon_orthogonality_error_f32(3u, 4u, u, &ortho_err) == ET_MUON_OK,
        "orthogonality error on zero matrix succeeds");
  CHECK(fabs(ortho_err - sqrt(3.0)) < 1e-12,
        "orthogonality error of zero matrix equals sqrt(r)");

  /* Zero-gradient hybrid step with apply_weight_decay=0 leaves weights intact. */
  float w2d[12], b2d[12] = {0.0f}, w2d_orig[12];
  fill_matrix(w2d, 3u, 4u, 77u);
  memcpy(w2d_orig, w2d, sizeof(w2d));

  const et_muon_param_group grp = {
      ET_MUON_GROUP_MUON_2D, 3u, 4u, 0u, w2d, g, b2d, NULL};
  const et_muon_step_config cfg = {
      0.05, 1u, 0.9, 1e-7, 0.1, 5u, 1u, 0.9, 0.999, 1e-8, 0.1, 1.0};
  CHECK(et_muon_hybrid_step(&cfg, 1u, &grp) == ET_MUON_OK,
        "zero-grad Muon step succeeds");
  CHECK(memcmp(w2d, w2d_orig, sizeof(w2d)) == 0,
        "zero-grad Muon step with wd_mask=0 preserves weights bitwise");
}

static void test_wsd_boundaries(void) {
  /* Linear WSD: warmup=4, stable=5, total=14 (stable_end=9, decay_steps=5). */
  const et_wsd_config lin = {4u, 5u, 14u, 0.1, 0.01, ET_WSD_DECAY_LINEAR, 0u};
  double lr = 0.0;
  uint32_t phase = 99u;

  CHECK(et_wsd_lr(&lin, 0u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_WARMUP && fabs(lr - 0.025) < 1e-15,
        "WSD warmup step 0");
  CHECK(et_wsd_lr(&lin, 3u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_WARMUP && fabs(lr - 0.1) < 1e-15,
        "WSD warmup final step");
  CHECK(et_wsd_lr(&lin, 4u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_STABLE && lr == 0.1,
        "WSD stable start");
  CHECK(et_wsd_lr(&lin, 8u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_STABLE && lr == 0.1,
        "WSD stable end - 1");
  CHECK(et_wsd_lr(&lin, 9u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && lr == 0.1,
        "WSD linear decay start");
  CHECK(et_wsd_lr(&lin, 11u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY &&
            fabs(lr - (0.1 - 0.09 * (2.0 / 5.0))) < 1e-15,
        "WSD linear decay interior");
  CHECK(et_wsd_lr(&lin, 14u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && lr == 0.01,
        "WSD linear decay exact total_steps");
  CHECK(et_wsd_lr(&lin, 100u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && lr == 0.01,
        "WSD linear clamped past total_steps");

  /* Cosine WSD midpoint check. */
  const et_wsd_config cos_cfg = {2u, 2u, 8u, 0.2, 0.02, ET_WSD_DECAY_COSINE,
                                 0u};
  CHECK(et_wsd_lr(&cos_cfg, 6u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && fabs(lr - 0.11) < 1e-15,
        "WSD cosine midpoint equals average of max_lr and min_lr");
  CHECK(et_wsd_lr(&cos_cfg, 8u, &lr, NULL) == ET_MUON_OK && lr == 0.02,
        "WSD cosine allows NULL out_phase and hits exact min_lr");

  /* Edge cases: warmup=0, stable=0, decay=0. */
  const et_wsd_config no_warm = {0u, 3u, 6u, 0.1, 0.0, ET_WSD_DECAY_LINEAR, 0u};
  CHECK(et_wsd_lr(&no_warm, 0u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_STABLE && lr == 0.1,
        "WSD warmup=0 starts directly in STABLE");

  const et_wsd_config no_stable = {2u, 0u, 6u, 0.1, 0.0, ET_WSD_DECAY_COSINE,
                                   0u};
  CHECK(et_wsd_lr(&no_stable, 2u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && lr == 0.1,
        "WSD stable=0 transitions directly to DECAY");

  const et_wsd_config no_decay = {2u, 3u, 5u, 0.1, 0.02, ET_WSD_DECAY_LINEAR,
                                  0u};
  CHECK(et_wsd_lr(&no_decay, 5u, &lr, &phase) == ET_MUON_OK &&
            phase == ET_WSD_PHASE_DECAY && lr == 0.02,
        "WSD decay=0 clamps to min_lr at total_steps");
}

static void test_weight_decay_mask_and_aspect_scaling(void) {
  /* 1. 2D Muon weight decay mask: zero grad, nonzero weight, wd=0.2, lr=0.1. */
  float w_nod[4] = {1.0f, -2.0f, 3.0f, -4.0f};
  float w_wd[4] = {1.0f, -2.0f, 3.0f, -4.0f};
  const float g_zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float g_zero2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float b1[4] = {0.0f}, b2[4] = {0.0f};

  /* 2. 1D AdamW weight decay mask: zero grad, nonzero norm scale, wd=0.2, lr=0.1. */
  float gamma_nod[4] = {1.0f, 1.25f, 0.75f, 1.5f};
  float gamma_wd[4] = {1.0f, 1.25f, 0.75f, 1.5f};
  const float gg1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float gg2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float m1[4] = {0.0f}, v1[4] = {0.0f};
  float m2[4] = {0.0f}, v2[4] = {0.0f};

  const et_muon_param_group groups[4] = {
      {ET_MUON_GROUP_MUON_2D, 2u, 2u, 0u, w_nod, g_zero, b1, NULL},
      {ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, w_wd, g_zero2, b2, NULL},
      {ET_MUON_GROUP_ADAMW_1D, 4u, 1u, 0u, gamma_nod, gg1, m1, v1},
      {ET_MUON_GROUP_ADAMW_1D, 4u, 1u, 1u, gamma_wd, gg2, m2, v2},
  };
  const et_muon_step_config cfg = {
      0.1, 1u, 0.9, 1e-7, 0.2, 5u, 1u, 0.9, 0.999, 1e-8, 0.2, 1.0};
  CHECK(et_muon_hybrid_step(&cfg, 4u, groups) == ET_MUON_OK,
        "weight decay mask hybrid step succeeds");

  const float expected_w_orig[4] = {1.0f, -2.0f, 3.0f, -4.0f};
  const float expected_g_orig[4] = {1.0f, 1.25f, 0.75f, 1.5f};
  CHECK(memcmp(w_nod, expected_w_orig, sizeof(w_nod)) == 0,
        "2D Muon apply_weight_decay=0 leaves params bit-identical on zero grad");
  CHECK(memcmp(gamma_nod, expected_g_orig, sizeof(gamma_nod)) == 0,
        "1D AdamW apply_weight_decay=0 leaves norm scales bit-identical on zero grad");
  for (size_t i = 0; i < 4u; ++i) {
    CHECK(fabsf(w_wd[i] - expected_w_orig[i] * 0.98f) < 1e-6f,
          "2D Muon apply_weight_decay=1 applies decoupled weight decay");
    CHECK(fabsf(gamma_wd[i] - expected_g_orig[i] * 0.98f) < 1e-6f,
          "1D AdamW apply_weight_decay=1 applies decoupled weight decay");
  }

  /* 3. Aspect-ratio scaling: tall (8x2 -> scale=sqrt(4)=2.0) vs wide (2x8 -> scale=1.0). */
  float g_tall[16], g_wide[16];
  fill_matrix(g_tall, 8u, 2u, 61u);
  for (uint32_t r = 0; r < 8u; ++r)
    for (uint32_t c = 0; c < 2u; ++c)
      g_wide[(size_t)c * 8u + r] = g_tall[(size_t)r * 2u + c];
  float w_tall[16] = {0.0f}, b_tall[16] = {0.0f};
  float w_wide[16] = {0.0f}, b_wide[16] = {0.0f};
  const et_muon_param_group agroups[2] = {
      {ET_MUON_GROUP_MUON_2D, 8u, 2u, 0u, w_tall, g_tall, b_tall, NULL},
      {ET_MUON_GROUP_MUON_2D, 2u, 8u, 0u, w_wide, g_wide, b_wide, NULL},
  };
  const et_muon_step_config acfg = {
      0.05, 1u, 0.0, 1e-7, 0.0, 5u, 0u, 0.9, 0.999, 1e-8, 0.0, 1.0};
  CHECK(et_muon_hybrid_step(&acfg, 2u, agroups) == ET_MUON_OK,
        "aspect-ratio scaling test step succeeds");
  for (uint32_t r = 0; r < 8u; ++r) {
    for (uint32_t c = 0; c < 2u; ++c) {
      const float wt = w_tall[(size_t)r * 2u + c];
      const float ww = w_wide[(size_t)c * 8u + r];
      CHECK(fabsf(wt - 2.0f * ww) < 1e-5f,
            "tall 8x2 Muon update scales by sqrt(8/2)=2 relative to transposed 2x8");
    }
  }
}

static void test_failure_atomicity_and_negatives(void) {
  const et_muon_step_config good_cfg = {
      0.05, 1u, 0.9, 1e-7, 0.01, 5u, 1u, 0.9, 0.999, 1e-8, 0.01, 1.0};
  const et_wsd_config good_wsd = {
      4u, 5u, 14u, 0.05, 0.005, ET_WSD_DECAY_COSINE, 0u};

  /* Multi-group failure atomicity: group 0 valid, group 1 has NaN grad. */
  float w0[4] = {0.5f, -0.5f, 0.25f, -0.25f};
  const float g0[4] = {0.1f, 0.2f, -0.1f, 0.3f};
  float b0[4] = {0.01f, -0.02f, 0.03f, -0.04f};
  float w1[4] = {1.0f, 1.1f, 1.2f, 1.3f};
  float g1[4] = {0.1f, NAN, 0.2f, 0.3f};
  float m1[4] = {0.05f, 0.05f, 0.05f, 0.05f};
  float v1[4] = {0.01f, 0.01f, 0.01f, 0.01f};

  float w0_save[4], b0_save[4], w1_save[4], m1_save[4], v1_save[4];
  memcpy(w0_save, w0, sizeof(w0));
  memcpy(b0_save, b0, sizeof(b0));
  memcpy(w1_save, w1, sizeof(w1));
  memcpy(m1_save, m1, sizeof(m1));
  memcpy(v1_save, v1, sizeof(v1));

  const et_muon_param_group bad_groups[2] = {
      {ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, w0, g0, b0, NULL},
      {ET_MUON_GROUP_ADAMW_1D, 4u, 1u, 0u, w1, g1, m1, v1},
  };
  CHECK(et_muon_hybrid_step(&good_cfg, 2u, bad_groups) == ET_MUON_ERR_NONFINITE,
        "NaN grad in group 1 rejected");
  CHECK(memcmp(w0, w0_save, sizeof(w0)) == 0 &&
            memcmp(b0, b0_save, sizeof(b0)) == 0 &&
            memcmp(w1, w1_save, sizeof(w1)) == 0 &&
            memcmp(m1, m1_save, sizeof(m1)) == 0 &&
            memcmp(v1, v1_save, sizeof(v1)) == 0,
        "failure atomicity preserves group 0 and group 1 bitwise on NaN grad");

  /* Negative variance in group 1 preserves all buffers. */
  g1[1] = 0.1f;
  v1[2] = -0.001f;
  memcpy(v1_save, v1, sizeof(v1));
  CHECK(et_muon_hybrid_step(&good_cfg, 2u, bad_groups) == ET_MUON_ERR_ARGUMENT,
        "negative AdamW variance rejected");
  CHECK(memcmp(w0, w0_save, sizeof(w0)) == 0 &&
            memcmp(b0, b0_save, sizeof(b0)) == 0 &&
            memcmp(v1, v1_save, sizeof(v1)) == 0,
        "failure atomicity preserves buffers on negative variance");

  /* Staging-phase overflow in group 1 preserves group 0 bitwise! */
  v1[2] = 0.01f;
  w1[0] = FLT_MAX;
  g1[0] = -1.0f;
  memcpy(w1_save, w1, sizeof(w1));
  memcpy(v1_save, v1, sizeof(v1));
  et_muon_step_config big_lr = good_cfg;
  big_lr.lr = 1e35;
  CHECK(et_muon_hybrid_step(&big_lr, 2u, bad_groups) == ET_MUON_ERR_NONFINITE,
        "overflow during staging in group 1 rejected");
  CHECK(memcmp(w0, w0_save, sizeof(w0)) == 0 &&
            memcmp(b0, b0_save, sizeof(b0)) == 0 &&
            memcmp(w1, w1_save, sizeof(w1)) == 0 &&
            memcmp(m1, m1_save, sizeof(m1)) == 0 &&
            memcmp(v1, v1_save, sizeof(v1)) == 0,
        "failure atomicity preserves group 0 when group 1 overflows in staging");

  /* Restore finite w1. */
  w1[0] = 1.0f;

  /* Null pointers. */
  float u_sent[4] = {7.0f, 7.0f, 7.0f, 7.0f};
  double err_sent = 42.0;
  CHECK(et_muon_newton_schulz5_f32(2u, 2u, NULL, 5u, 1e-7, u_sent) ==
            ET_MUON_ERR_ARGUMENT,
        "NS5 rejects NULL input");
  CHECK(et_muon_newton_schulz5_f32(2u, 2u, g0, 5u, 1e-7, NULL) ==
            ET_MUON_ERR_ARGUMENT,
        "NS5 rejects NULL output");
  CHECK(et_muon_orthogonality_error_f32(2u, 2u, NULL, &err_sent) ==
            ET_MUON_ERR_ARGUMENT,
        "ortho error rejects NULL u");
  CHECK(et_muon_orthogonality_error_f32(2u, 2u, g0, NULL) ==
            ET_MUON_ERR_ARGUMENT,
        "ortho error rejects NULL out_error");
  CHECK(et_wsd_lr(NULL, 0u, &err_sent, NULL) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects NULL config");
  CHECK(et_wsd_lr(&good_wsd, 0u, NULL, NULL) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects NULL out_lr");
  CHECK(et_muon_hybrid_step(NULL, 1u, bad_groups) == ET_MUON_ERR_ARGUMENT,
        "hybrid step rejects NULL cfg");
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, NULL) == ET_MUON_ERR_ARGUMENT,
        "hybrid step rejects NULL groups");
  CHECK(et_muon_hybrid_step_wsd(NULL, 0u, &good_cfg, 1u, bad_groups, NULL,
                                NULL) == ET_MUON_ERR_ARGUMENT,
        "hybrid_step_wsd rejects NULL wsd");
  CHECK(et_muon_hybrid_step_wsd(&good_wsd, UINT64_MAX, &good_cfg, 1u,
                                bad_groups, NULL, NULL) == ET_MUON_ERR_OVERFLOW,
        "hybrid_step_wsd rejects step overflow");
  CHECK(u_sent[0] == 7.0f && err_sent == 42.0,
        "failed NS5/ortho/WSD calls leave sentinels untouched");

  /* Misaligned pointers (via void* raw interfaces, zero misaligned-ptr UB). */
  uint8_t raw_buf[64] = {0};
  CHECK(et_muon_newton_schulz5_raw(2u, 2u, raw_buf + 1u, 5u, 1e-7, u_sent) ==
            ET_MUON_ERR_ALIGNMENT,
        "NS5_raw rejects misaligned input");
  CHECK(et_muon_newton_schulz5_raw(2u, 2u, g0, 5u, 1e-7, raw_buf + 1u) ==
            ET_MUON_ERR_ALIGNMENT,
        "NS5_raw rejects misaligned output");
  CHECK(et_muon_orthogonality_error_raw(2u, 2u, raw_buf + 1u, &err_sent) ==
            ET_MUON_ERR_ALIGNMENT,
        "ortho_raw rejects misaligned input");
  et_muon_param_group mis_grp = {
      ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, raw_buf + 1u, g0, b0, NULL};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &mis_grp) == ET_MUON_ERR_ALIGNMENT,
        "hybrid step rejects misaligned params");

  /* Aliased buffers: NS5 self-alias and partial overlap, plus cross-group alias. */
  float overlap[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  CHECK(et_muon_newton_schulz5_f32(2u, 2u, overlap, 5u, 1e-7, overlap) ==
            ET_MUON_ERR_ALIAS,
        "NS5 rejects identical g==u");
  CHECK(et_muon_newton_schulz5_f32(2u, 2u, overlap, 5u, 1e-7, overlap + 2u) ==
            ET_MUON_ERR_ALIAS,
        "NS5 rejects partially overlapping g and u");
  et_muon_param_group alias_grp = {
      ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, w0, g0, w0, NULL};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &alias_grp) == ET_MUON_ERR_ALIAS,
        "hybrid step rejects params==momentum alias");
  et_muon_param_group cross_alias[2] = {
      {ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, w0, g0, b0, NULL},
      {ET_MUON_GROUP_ADAMW_1D, 4u, 1u, 0u, w1, g0, m1, v1},
  };
  CHECK(et_muon_hybrid_step(&good_cfg, 2u, cross_alias) == ET_MUON_ERR_ALIAS,
        "hybrid step rejects cross-group aliased buffer");

  /* Invalid shapes. */
  CHECK(et_muon_newton_schulz5_f32(0u, 4u, g0, 5u, 1e-7, u_sent) ==
            ET_MUON_ERR_SHAPE,
        "NS5 rejects rows=0");
  CHECK(et_muon_newton_schulz5_f32(65u, 4u, g0, 5u, 1e-7, u_sent) ==
            ET_MUON_ERR_SHAPE,
        "NS5 rejects rows=65");
  CHECK(et_muon_hybrid_step(&good_cfg, 0u, bad_groups) == ET_MUON_ERR_SHAPE,
        "hybrid step rejects num_groups=0");
  CHECK(et_muon_hybrid_step(&good_cfg, ET_MUON_MAX_GROUPS + 1u, bad_groups) ==
            ET_MUON_ERR_SHAPE,
        "hybrid step rejects num_groups>MAX");
  et_muon_param_group bad_1d_cols = {
      ET_MUON_GROUP_ADAMW_1D, 4u, 2u, 0u, w1, g1, m1, v1};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &bad_1d_cols) == ET_MUON_ERR_SHAPE,
        "1D AdamW rejects cols != 1");
  et_muon_param_group bad_muon_var = {
      ET_MUON_GROUP_MUON_2D, 2u, 2u, 1u, w0, g0, b0, v1};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &bad_muon_var) ==
            ET_MUON_ERR_ARGUMENT,
        "2D Muon rejects non-NULL variance_buf");
  et_muon_param_group bad_mask = {
      ET_MUON_GROUP_MUON_2D, 2u, 2u, 2u, w0, g0, b0, NULL};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &bad_mask) == ET_MUON_ERR_ARGUMENT,
        "hybrid step rejects apply_weight_decay=2");
  et_muon_param_group bad_kind = {9u, 2u, 2u, 1u, w0, g0, b0, NULL};
  CHECK(et_muon_hybrid_step(&good_cfg, 1u, &bad_kind) == ET_MUON_ERR_ARGUMENT,
        "hybrid step rejects unknown group kind");

  /* Invalid hyperparameters. */
  et_muon_step_config c = good_cfg;
  c.muon_beta = 1.0;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects muon_beta >= 1");
  c = good_cfg;
  c.muon_beta = -0.1;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects muon_beta < 0");
  c = good_cfg;
  c.muon_eps = 0.0;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects muon_eps <= 0");
  c = good_cfg;
  c.muon_ns_steps = 0u;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects ns_steps == 0");
  c = good_cfg;
  c.muon_ns_steps = ET_MUON_MAX_NS_STEPS + 1u;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects ns_steps > MAX");
  c = good_cfg;
  c.muon_nesterov = 2u;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects nesterov not in {0,1}");
  c = good_cfg;
  c.step_index = 0u;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects step_index == 0");
  c = good_cfg;
  c.adamw_beta2 = 1.0;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_ARGUMENT,
        "rejects adamw_beta2 >= 1");
  c = good_cfg;
  c.adamw_eps = NAN;
  CHECK(et_muon_validate_step_config(&c) == ET_MUON_ERR_NONFINITE,
        "rejects NaN adamw_eps");

  /* Invalid WSD configs. */
  et_wsd_config w = good_wsd;
  w.warmup_steps = 10u;
  w.stable_steps = 5u;
  w.total_steps = 14u;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects warmup + stable > total");
  w = good_wsd;
  w.warmup_steps = UINT64_MAX;
  w.stable_steps = 1u;
  w.total_steps = UINT64_MAX;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects uint64 warmup + stable overflow");
  w = good_wsd;
  w.total_steps = 0u;
  w.warmup_steps = 0u;
  w.stable_steps = 0u;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects total_steps == 0");
  w = good_wsd;
  w.min_lr = 0.1;
  w.max_lr = 0.05;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects min_lr > max_lr");
  w = good_wsd;
  w.decay_kind = 3u;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects invalid decay_kind");
  w = good_wsd;
  w.reserved = 1u;
  CHECK(et_wsd_validate_config(&w) == ET_MUON_ERR_ARGUMENT,
        "WSD rejects nonzero reserved");

  /* FP-environment tampering checks (rounding mode & MXCSR FTZ on x86-64). */
  const int saved_round = fegetround();
  if (fesetround(FE_UPWARD) == 0) {
    CHECK(et_wsd_validate_config(&good_wsd) == ET_MUON_ERR_FP_ENV,
          "WSD rejects non-FE_TONEAREST rounding");
    CHECK(et_muon_newton_schulz5_f32(2u, 2u, g0, 5u, 1e-7, u_sent) ==
              ET_MUON_ERR_FP_ENV,
          "NS5 rejects non-FE_TONEAREST rounding");
    CHECK(et_muon_hybrid_step(&good_cfg, 1u, &bad_groups[0]) ==
              ET_MUON_ERR_FP_ENV,
          "hybrid step rejects non-FE_TONEAREST rounding");
    CHECK(fesetround(saved_round) == 0, "restore rounding mode");
  }
#if defined(__x86_64__)
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* Enable FTZ. */
  CHECK(et_muon_newton_schulz5_f32(2u, 2u, g0, 5u, 1e-7, u_sent) ==
            ET_MUON_ERR_FP_ENV,
        "NS5 rejects MXCSR FTZ enabled");
  _mm_setcsr(saved_mxcsr);
#endif
}

static void test_bridge_lifecycle_and_negatives(void) {
  et_muon_aot_ctx *ctx = NULL;
  CHECK(et_muon_aot_create(0.9, 1e-7, 0.01, 5, 1, 0.9, 0.999, 1e-8, 0.01, 1.0,
                           10, 10, 15, 0.05, 0.005, ET_WSD_DECAY_COSINE,
                           &ctx) == ET_MUON_ERR_ARGUMENT &&
            ctx == NULL,
        "bridge create rejects warmup+stable>total");

  CHECK(et_muon_aot_create(0.9, 1e-7, 0.01, 5, 1, 0.9, 0.999, 1e-8, 0.01, 1.0,
                           4, 5, 14, 0.04, 0.004, ET_WSD_DECAY_COSINE,
                           &ctx) == ET_MUON_OK &&
            ctx != NULL,
        "bridge create succeeds");

  const int32_t g0 = et_muon_aot_add_muon_group(ctx, 4, 6, 1);
  const int32_t g1 = et_muon_aot_add_adamw_group(ctx, 4, 0);
  CHECK(g0 == 0 && g1 == 1 && et_muon_aot_group_count(ctx) == 2 &&
            et_muon_aot_group_size(ctx, 0) == 24 &&
            et_muon_aot_group_size(ctx, 1) == 4,
        "bridge group registration");
  CHECK(et_muon_aot_add_muon_group(ctx, 4, 6, 2) == ET_MUON_ERR_ARGUMENT,
        "bridge rejects invalid wd mask");
  CHECK(et_muon_aot_set_grad(ctx, 0, 0, NAN) == ET_MUON_ERR_NONFINITE,
        "bridge set_grad rejects NaN");

  for (int64_t i = 0; i < 24; ++i) {
    CHECK(et_muon_aot_set_param(ctx, 0, i, 0.1 * (double)(i - 12)) ==
              ET_MUON_OK,
          "bridge set_param 2D");
    CHECK(et_muon_aot_set_grad(ctx, 0, i,
                               (double)det_matrix_val((uint32_t)(i / 6),
                                                      (uint32_t)(i % 6), 6u,
                                                      7u)) == ET_MUON_OK,
          "bridge set_grad 2D");
  }
  for (int64_t i = 0; i < 4; ++i) {
    CHECK(et_muon_aot_set_param(ctx, 1, i, 1.0) == ET_MUON_OK,
          "bridge set_param 1D");
    CHECK(et_muon_aot_set_grad(ctx, 1, i, 0.0) == ET_MUON_OK,
          "bridge set_grad 1D zero");
  }

  CHECK(et_muon_aot_group_ns5_ortho_error(ctx, 0) == ET_MUON_OK &&
            isfinite(et_muon_aot_last_ortho_error(ctx)) &&
            et_muon_aot_last_ortho_error(ctx) < 1.3,
        "bridge group NS5 ortho error bounded");
  CHECK(et_muon_aot_group_ns5_ortho_error(ctx, 1) == ET_MUON_ERR_ARGUMENT,
        "bridge group NS5 rejects 1D group");
  CHECK(et_muon_aot_step_wsd(ctx, 0) == ET_MUON_OK &&
            et_muon_aot_last_phase(ctx) == ET_WSD_PHASE_WARMUP &&
            fabs(et_muon_aot_last_lr(ctx) - 0.01) < 1e-12,
        "bridge step_wsd step 0");
  for (int64_t i = 0; i < 4; ++i)
    CHECK(et_muon_aot_get_param(ctx, 1, i) == 1.0,
          "bridge 1D wd=0 zero-grad preserves norm scale");

  void *raw_g = et_muon_aot_alloc_f32(16);
  void *raw_u = et_muon_aot_alloc_f32(16);
  CHECK(raw_g != NULL && raw_u != NULL, "bridge raw alloc");
  for (int64_t i = 0; i < 16; ++i)
    CHECK(et_muon_aot_store_f32(raw_g, 16, i,
                                (double)det_matrix_val((uint32_t)(i / 4),
                                                       (uint32_t)(i % 4), 4u,
                                                       19u)) == ET_MUON_OK,
          "bridge raw store");
  CHECK(et_muon_aot_newton_schulz5(4, 4, raw_g, 5, 1e-7, raw_u) == ET_MUON_OK,
        "bridge raw NS5");
  double raw_err = -1.0;
  CHECK(et_muon_aot_orthogonality_error(4, 4, raw_u, &raw_err) == ET_MUON_OK &&
            isfinite(raw_err) && raw_err < 1.3,
        "bridge raw ortho error");
  CHECK(isfinite(et_muon_aot_load_f32(raw_u, 16, 0)), "bridge raw load");
  et_muon_aot_free_f32(raw_g);
  et_muon_aot_free_f32(raw_u);
  et_muon_aot_destroy(ctx);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return emit_oracle();

  test_newton_schulz5_shapes();
  test_zero_gradient_safety();
  test_wsd_boundaries();
  test_weight_decay_mask_and_aspect_scaling();
  test_failure_atomicity_and_negatives();
  test_bridge_lifecycle_and_negatives();

  if (g_failures != 0) {
    fprintf(stderr, "O2 MUON NATIVE TESTS FAILED (%d failures)\n", g_failures);
    return 1;
  }
  printf("O2 MUON NATIVE TESTS: PASS\n");
  return 0;
}
