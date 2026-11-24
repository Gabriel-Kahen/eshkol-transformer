#define _GNU_SOURCE
#include "eshkol_transformer/mtp_head_abi.h"
#include "mtp_head_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static uint64_t test_splitmix(uint64_t *s) {
  *s += UINT64_C(0x9e3779b97f4a7c15);
  uint64_t z = *s;
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static et_mtp_config_v1 make_cfg(uint32_t V, uint32_t H, uint32_t F, uint32_t D,
                                 uint32_t tie_weights, double lambda_mtp,
                                 double lr) {
  et_mtp_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.struct_size = sizeof(cfg);
  cfg.vocab_size = V;
  cfg.hidden_dim = H;
  cfg.ffn_dim = F;
  cfg.mtp_depth = D;
  cfg.tie_weights = tie_weights;
  cfg.rms_eps = 1e-6;
  cfg.lambda_mtp = lambda_mtp;
  cfg.lr = lr;
  cfg.beta1 = 0.9;
  cfg.beta2 = 0.999;
  cfg.eps = 1e-8;
  cfg.weight_decay = 0.01;
  cfg.max_grad_norm = 5.0;
  return cfg;
}

static void fill_deterministic_batch(size_t B, size_t T, size_t H, uint32_t V,
                                     uint64_t seed, float *h0,
                                     uint32_t *tokens) {
  uint64_t rng = seed ^ UINT64_C(0x9e3779b97f4a7c15);
  for (size_t b = 0u; b < B; ++b) {
    for (size_t t = 0u; t < T; ++t) {
      tokens[b * T + t] = (uint32_t)((t * 3u + b * 5u + 1u) % (size_t)V);
      for (size_t j = 0u; j < H; ++j) {
        const uint64_t z = test_splitmix(&rng);
        const double u = (double)(z >> 11u) * (1.0 / 9007199254740992.0);
        h0[(b * T + t) * H + j] = (float)((u * 2.0 - 1.0) * 0.9);
      }
    }
  }
}

static double slice_l1(const float *buf, size_t off, size_t len) {
  double s = 0.0;
  for (size_t i = 0u; i < len; ++i) {
    s += fabs((double)buf[off + i]);
  }
  return s;
}

static uint32_t argmax_f32(const float *v, size_t n) {
  uint32_t best = 0u;
  float bv = v[0];
  for (size_t i = 1u; i < n; ++i) {
    if (v[i] > bv) {
      bv = v[i];
      best = (uint32_t)i;
    }
  }
  return best;
}

/* Oracle dump mode: writes binary64/binary32 ground truth for Python oracle. */
static int run_oracle_dump(uint32_t D, uint32_t tie_weights,
                           const char *out_path) {
  const uint32_t V = 12u;
  const uint32_t H = 6u;
  const uint32_t F = 10u;
  const size_t B = 2u;
  const size_t T = 10u;
  const uint64_t w_seed = UINT64_C(20261124) + (uint64_t)D * 101u + tie_weights;
  const uint64_t d_seed = UINT64_C(77777) + (uint64_t)D * 17u;

  et_mtp_config_v1 cfg = make_cfg(V, H, F, D, tie_weights, 0.6, 0.08);
  et_mtp_model_t *m = NULL;
  CHECK(et_mtp_model_create_v1(&cfg, &m) == ET_MTP_OK, "oracle model create");
  CHECK(et_mtp_model_init_weights_v1(m, w_seed) == ET_MTP_OK, "oracle init");

  et_mtp_layout_v1 lay;
  CHECK(et_mtp_model_get_layout_v1(m, &lay) == ET_MTP_OK, "oracle layout");
  const size_t n = lay.total_params;

  float *init_params = (float *)malloc(n * sizeof(float));
  float *d_params = (float *)malloc(n * sizeof(float));
  float *h0 = (float *)malloc(B * T * H * sizeof(float));
  float *d_h0 = (float *)malloc(B * T * H * sizeof(float));
  uint32_t *tokens = (uint32_t *)malloc(B * T * sizeof(uint32_t));
  CHECK(init_params && d_params && h0 && d_h0 && tokens, "oracle alloc");

  CHECK(et_mtp_model_get_params_v1(m, init_params, n) == ET_MTP_OK,
        "get init params");
  fill_deterministic_batch(B, T, H, V, d_seed, h0, tokens);

  et_mtp_loss_stats_v1 st0;
  CHECK(et_mtp_forward_backward_v1(m, B, T, h0, tokens, &st0, d_h0, d_params,
                                   n) == ET_MTP_OK,
        "oracle fb0");

  FILE *fp = fopen(out_path, "wb");
  CHECK(fp != NULL, "fopen oracle dump");

  /* Header metadata */
  uint32_t meta[8] = {V, H, F, D, tie_weights, (uint32_t)B, (uint32_t)T,
                      (uint32_t)n};
  CHECK(fwrite(meta, sizeof(uint32_t), 8u, fp) == 8u, "write meta");
  double cfg_f64[8] = {cfg.rms_eps, cfg.lambda_mtp, cfg.lr,
                       cfg.beta1,   cfg.beta2,      cfg.eps,
                       cfg.weight_decay, cfg.max_grad_norm};
  CHECK(fwrite(cfg_f64, sizeof(double), 8u, fp) == 8u, "write cfg_f64");

  /* Inputs and step-0 gradients */
  CHECK(fwrite(init_params, sizeof(float), n, fp) == n, "write init_params");
  CHECK(fwrite(h0, sizeof(float), B * T * H, fp) == B * T * H, "write h0");
  CHECK(fwrite(tokens, sizeof(uint32_t), B * T, fp) == B * T, "write tokens");

  double losses0[7] = {st0.loss_total, st0.loss_main, st0.loss_mtp_avg,
                       st0.loss_depth[0], st0.loss_depth[1], st0.loss_depth[2],
                       st0.loss_depth[3]};
  CHECK(fwrite(losses0, sizeof(double), 7u, fp) == 7u, "write losses0");
  CHECK(fwrite(d_h0, sizeof(float), B * T * H, fp) == B * T * H, "write d_h0");
  CHECK(fwrite(d_params, sizeof(float), n, fp) == n, "write d_params");

  /* 12-step AdamW trajectory */
  for (int step = 0; step < 12; ++step) {
    et_mtp_loss_stats_v1 ls;
    et_mtp_step_stats_v1 ss;
    CHECK(et_mtp_forward_backward_v1(m, B, T, h0, tokens, &ls, d_h0, d_params,
                                     n) == ET_MTP_OK,
          "oracle step fb");
    CHECK(et_mtp_adamw_step_v1(m, d_params, n, &ss) == ET_MTP_OK,
          "oracle step adamw");
    double row[8] = {ls.loss_total, ls.loss_main, ls.loss_mtp_avg,
                     ls.loss_depth[0], ls.loss_depth[1], ls.loss_depth[2],
                     ls.loss_depth[3], ss.grad_norm};
    CHECK(fwrite(row, sizeof(double), 8u, fp) == 8u, "write traj row");
  }

  float *final_p = (float *)malloc(n * sizeof(float));
  float *final_m = (float *)malloc(n * sizeof(float));
  float *final_v = (float *)malloc(n * sizeof(float));
  CHECK(final_p && final_m && final_v, "alloc final");
  CHECK(et_mtp_model_get_params_v1(m, final_p, n) == ET_MTP_OK, "get final_p");
  CHECK(et_mtp_model_get_moments_v1(m, final_m, final_v, n) == ET_MTP_OK,
        "get final_mv");
  CHECK(fwrite(final_p, sizeof(float), n, fp) == n, "write final_p");
  CHECK(fwrite(final_m, sizeof(float), n, fp) == n, "write final_m");
  CHECK(fwrite(final_v, sizeof(float), n, fp) == n, "write final_v");

  /* Also dump one self-speculative greedy draft_verify call at prefix t=0. */
  float target_logits[(ET_MTP_MAX_DEPTH + 1u) * 12u];
  for (size_t j = 0u; j <= (size_t)D; ++j) {
    CHECK(et_mtp_main_logits_v1(m, h0 + (1u + j) * H,
                                target_logits + j * V) == ET_MTP_OK,
          "oracle main logits");
  }
  uint32_t drafts[ET_MTP_MAX_DEPTH] = {0u, 0u, 0u, 0u};
  uint32_t emitted[ET_MTP_MAX_DEPTH + 1u] = {0u, 0u, 0u, 0u, 0u};
  et_mtp_verify_result_v1 vres;
  CHECK(et_mtp_draft_verify_v1(m, h0, ET_MTP_ANCHOR_FROM_MAIN, target_logits,
                               0.0, ET_MTP_NO_EOS, NULL, drafts, emitted,
                               D + 1u, &vres) == ET_MTP_OK,
        "oracle draft verify");
  uint32_t spec_meta[4] = {vres.anchor_token_used, vres.accepted_count,
                           vres.emitted_count, vres.rejected};
  CHECK(fwrite(spec_meta, sizeof(uint32_t), 4u, fp) == 4u, "write spec meta");
  CHECK(fwrite(drafts, sizeof(uint32_t), D, fp) == D, "write drafts");
  CHECK(fwrite(emitted, sizeof(uint32_t), vres.emitted_count, fp) ==
            vres.emitted_count,
        "write emitted");

  fclose(fp);
  free(final_p);
  free(final_m);
  free(final_v);
  free(init_params);
  free(d_params);
  free(h0);
  free(d_h0);
  free(tokens);
  et_mtp_model_destroy_v1(m);
  return 0;
}

/* 1. Exhaustive central FD checks across D=1..4 (untied & tied). */
static void test_fd_all_depths(void) {
  const uint32_t V = 10u;
  const uint32_t H = 6u;
  const uint32_t F = 8u;
  const size_t B = 2u;
  const size_t T = 9u;
  const double fd_eps = 1.0 / 2048.0; /* exact power of 2 in f32 */

  for (uint32_t tie = 0u; tie <= 1u; ++tie) {
    for (uint32_t D = 1u; D <= 4u; ++D) {
      et_mtp_config_v1 cfg = make_cfg(V, H, F, D, tie, 0.7, 0.05);
      et_mtp_model_t *m = NULL;
      CHECK(et_mtp_model_create_v1(&cfg, &m) == ET_MTP_OK, "create model");
      CHECK(et_mtp_model_init_weights_v1(m, 1000u + D * 31u + tie) == ET_MTP_OK,
            "init weights");

      et_mtp_layout_v1 lay;
      CHECK(et_mtp_model_get_layout_v1(m, &lay) == ET_MTP_OK, "get layout");
      const size_t n = lay.total_params;
      const size_t h0_elems = B * T * H;

      float *h0 = (float *)malloc(h0_elems * sizeof(float));
      float *d_h0 = (float *)malloc(h0_elems * sizeof(float));
      uint32_t *tokens = (uint32_t *)malloc(B * T * sizeof(uint32_t));
      float *d_params = (float *)malloc(n * sizeof(float));
      float *orig_p = (float *)malloc(n * sizeof(float));
      float *work_p = (float *)malloc(n * sizeof(float));
      CHECK(h0 && d_h0 && tokens && d_params && orig_p && work_p, "alloc fd");
      fill_deterministic_batch(B, T, H, V, 555u + D, h0, tokens);

      et_mtp_loss_stats_v1 st_fwd, st_fb;
      CHECK(et_mtp_forward_v1(m, B, T, h0, tokens, &st_fwd) == ET_MTP_OK,
            "forward_v1");
      CHECK(et_mtp_forward_backward_v1(m, B, T, h0, tokens, &st_fb, d_h0,
                                       d_params, n) == ET_MTP_OK,
            "forward_backward_v1");

      /* Loss-only forward and forward_backward must match bitwise! */
      CHECK(st_fwd.loss_total == st_fb.loss_total, "loss_total exact match");
      CHECK(st_fwd.loss_main == st_fb.loss_main, "loss_main exact match");
      CHECK(st_fwd.loss_mtp_avg == st_fb.loss_mtp_avg, "loss_mtp_avg match");
      for (uint32_t k = 0u; k < ET_MTP_MAX_DEPTH; ++k) {
        CHECK(st_fwd.loss_depth[k] == st_fb.loss_depth[k], "loss_depth match");
      }

      /* Verify non-zero gradient on EVERY parameter group. */
      CHECK(slice_l1(d_params, lay.embed_offset, lay.embed_len) > 1e-5,
            "d_E active");
      if (!tie) {
        CHECK(slice_l1(d_params, lay.lm_head_offset, lay.lm_head_len) > 1e-5,
              "d_W_lm active");
      }
      CHECK(slice_l1(d_params, lay.gamma_main_offset, H) > 1e-5,
            "d_gamma_main active");
      for (uint32_t k = 0u; k < D; ++k) {
        const size_t base = lay.head_offset[k];
        CHECK(slice_l1(d_params, base + lay.rel_gamma_h, H) > 1e-6, "d_gh");
        CHECK(slice_l1(d_params, base + lay.rel_gamma_e, H) > 1e-6, "d_ge");
        CHECK(slice_l1(d_params, base + lay.rel_w_proj, H * 2u * H) > 1e-5,
              "d_w_proj");
        CHECK(slice_l1(d_params, base + lay.rel_gamma_f, H) > 1e-6, "d_gf");
        CHECK(slice_l1(d_params, base + lay.rel_w_gate, F * H) > 1e-5, "d_wg");
        CHECK(slice_l1(d_params, base + lay.rel_w_up, F * H) > 1e-5, "d_wu");
        CHECK(slice_l1(d_params, base + lay.rel_w_down, H * F) > 1e-5, "d_wd");
        CHECK(slice_l1(d_params, base + lay.rel_gamma_o, H) > 1e-6, "d_go");
      }

      /* Verify last position t = T - 1 has strictly zero d_h0. */
      for (size_t b = 0u; b < B; ++b) {
        for (size_t j = 0u; j < H; ++j) {
          CHECK(d_h0[(b * T + (T - 1u)) * H + j] == 0.0f,
                "d_h0 at t=T-1 must be 0");
        }
      }

      /* Exhaustive central FD over ALL n parameters. */
      CHECK(et_mtp_model_get_params_v1(m, orig_p, n) == ET_MTP_OK, "get p");
      memcpy(work_p, orig_p, n * sizeof(float));
      double max_p_err = 0.0;
      for (size_t i = 0u; i < n; ++i) {
        const float orig = orig_p[i];
        const float vp = (float)((double)orig + fd_eps);
        const float vm = (float)((double)orig - fd_eps);
        const double span = (double)vp - (double)vm;
        et_mtp_loss_stats_v1 sp, sm;
        work_p[i] = vp;
        CHECK(et_mtp_model_set_params_v1(m, work_p, n) == ET_MTP_OK, "set vp");
        CHECK(et_mtp_forward_v1(m, B, T, h0, tokens, &sp) == ET_MTP_OK, "fp");
        work_p[i] = vm;
        CHECK(et_mtp_model_set_params_v1(m, work_p, n) == ET_MTP_OK, "set vm");
        CHECK(et_mtp_forward_v1(m, B, T, h0, tokens, &sm) == ET_MTP_OK, "fm");
        work_p[i] = orig;
        const double num = (sp.loss_total - sm.loss_total) / span;
        const double err = fabs((double)d_params[i] - num);
        if (err > max_p_err) {
          max_p_err = err;
        }
      }
      CHECK(et_mtp_model_set_params_v1(m, orig_p, n) == ET_MTP_OK, "restore p");

      /* Exhaustive central FD over ALL h0 elements. */
      double max_h_err = 0.0;
      for (size_t i = 0u; i < h0_elems; ++i) {
        const float orig = h0[i];
        const float vp = (float)((double)orig + fd_eps);
        const float vm = (float)((double)orig - fd_eps);
        const double span = (double)vp - (double)vm;
        et_mtp_loss_stats_v1 sp, sm;
        h0[i] = vp;
        CHECK(et_mtp_forward_v1(m, B, T, h0, tokens, &sp) == ET_MTP_OK, "hp");
        h0[i] = vm;
        CHECK(et_mtp_forward_v1(m, B, T, h0, tokens, &sm) == ET_MTP_OK, "hm");
        h0[i] = orig;
        const double num = (sp.loss_total - sm.loss_total) / span;
        const double err = fabs((double)d_h0[i] - num);
        if (err > max_h_err) {
          max_h_err = err;
        }
      }

      printf("MTP NATIVE FD D=%u tie=%u params=%zu max_param_err=%.3e "
             "max_h0_err=%.3e\n",
             D, tie, n, max_p_err, max_h_err);
      CHECK(max_p_err < 1e-5, "param central FD max_err < 1e-5");
      CHECK(max_h_err < 1e-5, "h0 central FD max_err < 1e-5");

      free(h0);
      free(d_h0);
      free(tokens);
      free(d_params);
      free(orig_p);
      free(work_p);
      et_mtp_model_destroy_v1(m);
    }
  }
}

/* 2. Explicitly verify shared E / W_lm gradient summation across main + D. */
static void test_shared_gradient_accumulation(void) {
  const uint32_t V = 12u;
  const uint32_t H = 6u;
  const uint32_t F = 8u;
  const uint32_t D = 3u;
  const size_t B = 1u;
  const size_t T = 10u;

  /* Compare lambda=0.0 (main only), lambda=0.5, and lambda=1.0 -> exact linearity! */
  et_mtp_config_v1 c0 = make_cfg(V, H, F, D, 0u, 0.0, 0.05);
  et_mtp_config_v1 c1 = make_cfg(V, H, F, D, 0u, 0.5, 0.05);
  et_mtp_config_v1 c2 = make_cfg(V, H, F, D, 0u, 1.0, 0.05);

  et_mtp_model_t *m0 = NULL, *m1 = NULL, *m2 = NULL;
  CHECK(et_mtp_model_create_v1(&c0, &m0) == ET_MTP_OK, "c0");
  CHECK(et_mtp_model_create_v1(&c1, &m1) == ET_MTP_OK, "c1");
  CHECK(et_mtp_model_create_v1(&c2, &m2) == ET_MTP_OK, "c2");
  CHECK(et_mtp_model_init_weights_v1(m0, 999u) == ET_MTP_OK, "i0");
  CHECK(et_mtp_model_init_weights_v1(m1, 999u) == ET_MTP_OK, "i1");
  CHECK(et_mtp_model_init_weights_v1(m2, 999u) == ET_MTP_OK, "i2");

  et_mtp_layout_v1 lay;
  CHECK(et_mtp_model_get_layout_v1(m0, &lay) == ET_MTP_OK, "lay");
  const size_t n = lay.total_params;

  float *h0 = (float *)malloc(B * T * H * sizeof(float));
  uint32_t *tokens = (uint32_t *)malloc(B * T * sizeof(uint32_t));
  float *dh0 = (float *)malloc(B * T * H * sizeof(float));
  float *dp0 = (float *)malloc(n * sizeof(float));
  float *dp1 = (float *)malloc(n * sizeof(float));
  float *dp2 = (float *)malloc(n * sizeof(float));
  CHECK(h0 && tokens && dh0 && dp0 && dp1 && dp2, "alloc shared");
  fill_deterministic_batch(B, T, H, V, 888u, h0, tokens);

  et_mtp_loss_stats_v1 s0, s1, s2;
  CHECK(et_mtp_forward_backward_v1(m0, B, T, h0, tokens, &s0, dh0, dp0, n) ==
            ET_MTP_OK,
        "fb0");
  CHECK(et_mtp_forward_backward_v1(m1, B, T, h0, tokens, &s1, dh0, dp1, n) ==
            ET_MTP_OK,
        "fb1");
  CHECK(et_mtp_forward_backward_v1(m2, B, T, h0, tokens, &s2, dh0, dp2, n) ==
            ET_MTP_OK,
        "fb2");

  /* When lambda=0.0, d_E == 0 (untied) while d_W_lm != 0 (main head). */
  CHECK(slice_l1(dp0, lay.embed_offset, lay.embed_len) == 0.0,
        "lambda=0 untied d_E must be 0");
  CHECK(slice_l1(dp0, lay.lm_head_offset, lay.lm_head_len) > 1e-4,
        "lambda=0 d_W_lm must be active from main head");

  /* Linearity of shared W_lm gradient: dp1 - dp0 == 0.5 * (dp2 - dp0) */
  double max_lin_err = 0.0;
  double mtp_contrib_l1 = 0.0;
  for (size_t i = 0u; i < lay.lm_head_len; ++i) {
    const size_t idx = lay.lm_head_offset + i;
    const double g_main = (double)dp0[idx];
    const double g_half = (double)dp1[idx];
    const double g_full = (double)dp2[idx];
    const double expected_half = g_main + 0.5 * (g_full - g_main);
    const double err = fabs(g_half - expected_half);
    if (err > max_lin_err) {
      max_lin_err = err;
    }
    mtp_contrib_l1 += fabs(g_full - g_main);
  }
  CHECK(mtp_contrib_l1 > 1e-4, "MTP depths must contribute to shared d_W_lm");
  CHECK(max_lin_err < 1e-6, "shared d_W_lm linear combination mismatch");

  /* Linearity of shared E gradient: dp1[E] == 0.5 * dp2[E] */
  double max_e_err = 0.0;
  for (size_t i = 0u; i < lay.embed_len; ++i) {
    const size_t idx = lay.embed_offset + i;
    const double err = fabs((double)dp1[idx] - 0.5 * (double)dp2[idx]);
    if (err > max_e_err) {
      max_e_err = err;
    }
  }
  CHECK(slice_l1(dp2, lay.embed_offset, lay.embed_len) > 1e-4,
        "MTP depths must contribute to shared d_E");
  CHECK(max_e_err < 1e-6, "shared d_E linear combination mismatch");
  printf("MTP NATIVE SHARED_GRAD d_W_lm_mtp_l1=%.4f lin_err=%.3e e_err=%.3e "
         "PASS\n",
         mtp_contrib_l1, max_lin_err, max_e_err);

  free(h0);
  free(tokens);
  free(dh0);
  free(dp0);
  free(dp1);
  free(dp2);
  et_mtp_model_destroy_v1(m0);
  et_mtp_model_destroy_v1(m1);
  et_mtp_model_destroy_v1(m2);
}

/* 3. 12-step AdamW + 6+6 ETMT1000 resume parity + self-speculative verify. */
static void test_train_resume_and_spec_verify(void) {
  const uint32_t V = 16u;
  const uint32_t H = 8u;
  const uint32_t F = 16u;
  const uint32_t D = 3u;
  const size_t B = 1u;
  const size_t T = 16u;

  et_mtp_config_v1 cfg = make_cfg(V, H, F, D, 0u, 0.5, 0.08);
  et_mtp_model_t *m_full = NULL;
  et_mtp_model_t *m_res = NULL;
  CHECK(et_mtp_model_create_v1(&cfg, &m_full) == ET_MTP_OK, "create full");
  CHECK(et_mtp_model_create_v1(&cfg, &m_res) == ET_MTP_OK, "create res");
  CHECK(et_mtp_model_init_weights_v1(m_full, 20261124u) == ET_MTP_OK, "init f");
  CHECK(et_mtp_model_init_weights_v1(m_res, 20261124u) == ET_MTP_OK, "init r");

  et_mtp_layout_v1 lay;
  CHECK(et_mtp_model_get_layout_v1(m_full, &lay) == ET_MTP_OK, "lay");
  const size_t n = lay.total_params;

  float *h0 = (float *)malloc(B * T * H * sizeof(float));
  uint32_t *tokens = (uint32_t *)malloc(B * T * sizeof(uint32_t));
  float *d_h0 = (float *)malloc(B * T * H * sizeof(float));
  float *d_params = (float *)malloc(n * sizeof(float));
  CHECK(h0 && tokens && d_h0 && d_params, "alloc train");
  fill_deterministic_batch(B, T, H, V, 314159u, h0, tokens);

  char tmp_dir[] = "/tmp/mtp_native_XXXXXX";
  CHECK(mkdtemp(tmp_dir) != NULL, "mkdtemp");
  char ckpt_path[256];
  snprintf(ckpt_path, sizeof(ckpt_path), "%s/step6.etmt", tmp_dir);

  double prev_total = 1e9;
  double prev_main = 1e9;
  double prev_mtp = 1e9;
  for (int step = 0; step < 12; ++step) {
    et_mtp_loss_stats_v1 ls;
    et_mtp_step_stats_v1 ss;
    CHECK(et_mtp_forward_backward_v1(m_full, B, T, h0, tokens, &ls, d_h0,
                                     d_params, n) == ET_MTP_OK,
          "full fb");
    CHECK(ls.loss_total < prev_total, "L_total strictly decreasing");
    CHECK(ls.loss_main < prev_main, "L_main strictly decreasing");
    CHECK(ls.loss_mtp_avg < prev_mtp, "L_mtp_avg strictly decreasing");
    prev_total = ls.loss_total;
    prev_main = ls.loss_main;
    prev_mtp = ls.loss_mtp_avg;
    CHECK(et_mtp_adamw_step_v1(m_full, d_params, n, &ss) == ET_MTP_OK,
          "full adamw");
    CHECK(ss.step == (uint64_t)(step + 1), "step count");
  }

  /* 6 + 6 split with ETMT1000 checkpoint save & reload. */
  for (int step = 0; step < 6; ++step) {
    et_mtp_loss_stats_v1 ls;
    et_mtp_step_stats_v1 ss;
    CHECK(et_mtp_forward_backward_v1(m_res, B, T, h0, tokens, &ls, d_h0,
                                     d_params, n) == ET_MTP_OK,
          "res1 fb");
    CHECK(et_mtp_adamw_step_v1(m_res, d_params, n, &ss) == ET_MTP_OK,
          "res1 adamw");
  }
  CHECK(et_mtp_checkpoint_save_v1(m_res, ckpt_path) == ET_MTP_OK, "save ckpt");

  /* Verify strict 0600 file permissions. */
  struct stat st_mode;
  CHECK(stat(ckpt_path, &st_mode) == 0, "stat ckpt");
  CHECK((st_mode.st_mode & 0777) == 0600, "ckpt must be mode 0600");

  et_mtp_model_t *m_reloaded = NULL;
  CHECK(et_mtp_model_create_v1(&cfg, &m_reloaded) == ET_MTP_OK, "create rel");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, ckpt_path) == ET_MTP_OK,
        "load ckpt");

  uint64_t rel_step = 0u;
  CHECK(et_mtp_model_get_step_v1(m_reloaded, &rel_step) == ET_MTP_OK, "step");
  CHECK(rel_step == 6u, "reloaded step must be 6");

  for (int step = 6; step < 12; ++step) {
    et_mtp_loss_stats_v1 ls;
    et_mtp_step_stats_v1 ss;
    CHECK(et_mtp_forward_backward_v1(m_reloaded, B, T, h0, tokens, &ls, d_h0,
                                     d_params, n) == ET_MTP_OK,
          "res2 fb");
    CHECK(et_mtp_adamw_step_v1(m_reloaded, d_params, n, &ss) == ET_MTP_OK,
          "res2 adamw");
  }

  uint8_t dig_full[32], dig_rel[32];
  CHECK(et_mtp_model_state_sha256_v1(m_full, dig_full) == ET_MTP_OK, "dig f");
  CHECK(et_mtp_model_state_sha256_v1(m_reloaded, dig_rel) == ET_MTP_OK,
        "dig r");
  CHECK(memcmp(dig_full, dig_rel, 32u) == 0,
        "6+6 checkpoint resume must be bitwise identical to uninterrupted 12");

  /* Self-speculative greedy draft-verify equivalence on trained sequence. */
  float all_logits[16 * 16];
  uint32_t baseline[16];
  uint32_t spec_stream[24];
  float target_win[(ET_MTP_MAX_DEPTH + 1u) * 16u];
  for (size_t t = 0u; t + 1u < T; ++t) {
    CHECK(et_mtp_main_logits_v1(m_full, h0 + t * H, all_logits + t * V) ==
              ET_MTP_OK,
          "main logits");
    baseline[t] = argmax_f32(all_logits + t * V, V);
  }
  const size_t total_targets = T - 1u;
  spec_stream[0] = baseline[0];
  size_t produced = 1u;
  size_t verify_steps = 0u;
  size_t spec_emitted = 0u;
  while (produced < total_targets) {
    const size_t prefix = produced - 1u;
    for (size_t j = 0u; j <= D; ++j) {
      const size_t ti =
          (produced + j < total_targets) ? (produced + j) : (total_targets - 1u);
      memcpy(target_win + j * V, all_logits + ti * V, V * sizeof(float));
    }
    uint32_t drafts[ET_MTP_MAX_DEPTH];
    uint32_t emitted[ET_MTP_MAX_DEPTH + 1u];
    et_mtp_verify_result_v1 vr;
    CHECK(et_mtp_draft_verify_v1(m_full, h0 + prefix * H, spec_stream[prefix],
                                 target_win, 0.0, ET_MTP_NO_EOS, NULL, drafts,
                                 emitted, D + 1u, &vr) == ET_MTP_OK,
          "spec greedy");
    CHECK(vr.rng_draws == 0u, "greedy uses 0 rng draws");
    verify_steps += 1u;
    const size_t rem = total_targets - produced;
    const size_t take = (vr.emitted_count < rem) ? vr.emitted_count : rem;
    for (size_t j = 0u; j < take; ++j) {
      spec_stream[produced + j] = emitted[j];
    }
    produced += take;
    spec_emitted += take;
  }
  for (size_t t = 0u; t < total_targets; ++t) {
    CHECK(spec_stream[t] == baseline[t], "greedy stream bitwise identical");
  }
  const double tok_per_step = (double)spec_emitted / (double)verify_steps;
  CHECK(tok_per_step > 1.0, "trained MTP acceptance must exceed 1.0 tok/step");

  /* Also verify stochastic draft-verify determinism + EOS early stop. */
  et_mtp_rng_state_v1 rng = {UINT64_C(0x123456789abcdef0), 0u};
  uint32_t d1[ET_MTP_MAX_DEPTH], d2[ET_MTP_MAX_DEPTH];
  uint32_t e1[ET_MTP_MAX_DEPTH + 1u], e2[ET_MTP_MAX_DEPTH + 1u];
  et_mtp_verify_result_v1 r1, r2;
  for (size_t j = 0u; j <= D; ++j) {
    memcpy(target_win + j * V, all_logits + (1u + j) * V, V * sizeof(float));
  }
  CHECK(et_mtp_draft_verify_v1(m_full, h0, ET_MTP_ANCHOR_FROM_MAIN, target_win,
                               0.7, ET_MTP_NO_EOS, &rng, d1, e1, D + 1u,
                               &r1) == ET_MTP_OK,
        "stochastic run 1");
  CHECK(et_mtp_draft_verify_v1(m_full, h0, ET_MTP_ANCHOR_FROM_MAIN, target_win,
                               0.7, ET_MTP_NO_EOS, &rng, d2, e2, D + 1u,
                               &r2) == ET_MTP_OK,
        "stochastic run 2");
  CHECK(r1.rng_draws > 0u && r1.rng_draws == r2.rng_draws, "rng_draws match");
  CHECK(r1.emitted_count == r2.emitted_count &&
            r1.accepted_count == r2.accepted_count &&
            r1.anchor_token_used == r2.anchor_token_used,
        "stochastic determinism");
  CHECK(memcmp(d1, d2, D * sizeof(uint32_t)) == 0 &&
            memcmp(e1, e2, r1.emitted_count * sizeof(uint32_t)) == 0,
        "stochastic tokens match");

  /* EOS early termination check: set eos_token_id to baseline[1]. */
  et_mtp_verify_result_v1 reos;
  CHECK(et_mtp_draft_verify_v1(m_full, h0, baseline[0], target_win, 0.0,
                               baseline[1], NULL, d1, e1, D + 1u,
                               &reos) == ET_MTP_OK,
        "eos verify");
  CHECK(reos.eos_hit == 1u && reos.emitted_count == 1u &&
            e1[0] == baseline[1],
        "eos must terminate immediately on first emitted token");

  /* Negative checkpoint checks (tamper, magic, version, trunc, symlink, cfg). */
  char copy_path[256];
  snprintf(copy_path, sizeof(copy_path), "%s/copy.etmt", tmp_dir);
  uint8_t snap_before[32], snap_after[32];
  CHECK(et_mtp_model_state_sha256_v1(m_reloaded, snap_before) == ET_MTP_OK,
        "snap before");

  /* 1) Tampered payload byte -> CHECKSUM_MISMATCH */
  CHECK(et_mtp_aot_copy_file(ckpt_path, copy_path) == ET_MTP_OK, "copy");
  CHECK(et_mtp_aot_corrupt_file_byte(copy_path, 160) == ET_MTP_OK, "corrupt");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, copy_path) ==
            ET_MTP_ERR_CHECKSUM_MISMATCH,
        "tamper must fail checksum");

  /* 2) Corrupt magic -> BAD_MAGIC */
  CHECK(et_mtp_aot_copy_file(ckpt_path, copy_path) == ET_MTP_OK, "copy2");
  CHECK(et_mtp_aot_corrupt_file_byte(copy_path, 0) == ET_MTP_OK, "corrupt mag");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, copy_path) ==
            ET_MTP_ERR_BAD_MAGIC,
        "bad magic");

  /* 3) Corrupt version -> VERSION_MISMATCH */
  CHECK(et_mtp_aot_copy_file(ckpt_path, copy_path) == ET_MTP_OK, "copy3");
  CHECK(et_mtp_aot_corrupt_file_byte(copy_path, 8) == ET_MTP_OK, "corrupt ver");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, copy_path) ==
            ET_MTP_ERR_VERSION_MISMATCH,
        "version mismatch");

  /* 4) Truncated file -> TRUNCATED */
  CHECK(et_mtp_aot_copy_file(ckpt_path, copy_path) == ET_MTP_OK, "copy4");
  CHECK(et_mtp_aot_truncate_file(copy_path, 140) == ET_MTP_OK, "trunc");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, copy_path) ==
            ET_MTP_ERR_TRUNCATED,
        "truncated");

  /* 5) Symlink rejected on load AND save -> SYMLINK */
  char sym_path[256];
  snprintf(sym_path, sizeof(sym_path), "%s/link.etmt", tmp_dir);
  CHECK(et_mtp_aot_make_symlink(ckpt_path, sym_path) == ET_MTP_OK, "symlink");
  CHECK(et_mtp_checkpoint_load_v1(m_reloaded, sym_path) == ET_MTP_ERR_SYMLINK,
        "symlink load rejected");
  CHECK(et_mtp_checkpoint_save_v1(m_reloaded, sym_path) == ET_MTP_ERR_SYMLINK,
        "symlink save rejected");

  /* 6) Config mismatch (D=2 vs D=3) -> CONFIG_MISMATCH */
  et_mtp_config_v1 cfg_d2 = make_cfg(V, H, F, 2u, 0u, 0.5, 0.08);
  et_mtp_model_t *m_d2 = NULL;
  CHECK(et_mtp_model_create_v1(&cfg_d2, &m_d2) == ET_MTP_OK, "create d2");
  CHECK(et_mtp_checkpoint_load_v1(m_d2, ckpt_path) ==
            ET_MTP_ERR_CONFIG_MISMATCH,
        "config mismatch rejected");
  et_mtp_model_destroy_v1(m_d2);

  /* Confirm m_reloaded state remained byte-for-byte untouched across failures. */
  CHECK(et_mtp_model_state_sha256_v1(m_reloaded, snap_after) == ET_MTP_OK,
        "snap after");
  CHECK(memcmp(snap_before, snap_after, 32u) == 0,
        "failed loads must leave model untouched");

  printf("MTP NATIVE TRAIN12+RESUME6+6+SPEC final_L=%.6f tok_per_step=%.3f "
         "PASS\n",
         prev_total, tok_per_step);

  unlink(sym_path);
  unlink(copy_path);
  unlink(ckpt_path);
  rmdir(tmp_dir);
  free(h0);
  free(tokens);
  free(d_h0);
  free(d_params);
  et_mtp_model_destroy_v1(m_full);
  et_mtp_model_destroy_v1(m_res);
  et_mtp_model_destroy_v1(m_reloaded);
}

/* 4. Negative boundary, aliasing, non-finite, and FP-env trap checks. */
static void test_negative_boundaries(void) {
  /* Invalid depths 0 and 5. */
  et_mtp_config_v1 bad0 = make_cfg(16u, 8u, 16u, 0u, 0u, 0.5, 0.05);
  et_mtp_config_v1 bad5 = make_cfg(16u, 8u, 16u, 5u, 0u, 0.5, 0.05);
  CHECK(et_mtp_validate_config_v1(&bad0) == ET_MTP_ERR_UNSUPPORTED, "depth 0");
  CHECK(et_mtp_validate_config_v1(&bad5) == ET_MTP_ERR_UNSUPPORTED, "depth 5");

  et_mtp_config_v1 bad_neg0 = make_cfg(16u, 8u, 16u, 2u, 0u, -0.0, 0.05);
  CHECK(et_mtp_validate_config_v1(&bad_neg0) == ET_MTP_ERR_INVALID_ARG,
        "-0.0 lambda rejected");

  et_mtp_config_v1 cfg = make_cfg(16u, 8u, 16u, 3u, 0u, 0.5, 0.05);
  et_mtp_model_t *m = NULL;
  CHECK(et_mtp_model_create_v1(&cfg, &m) == ET_MTP_OK, "create neg");
  CHECK(et_mtp_model_init_weights_v1(m, 42u) == ET_MTP_OK, "init neg");
  et_mtp_layout_v1 lay;
  CHECK(et_mtp_model_get_layout_v1(m, &lay) == ET_MTP_OK, "lay neg");

  float h0[10 * 8];
  uint32_t tokens[10];
  float d_h0[10 * 8];
  float *d_params = (float *)malloc(lay.total_params * sizeof(float));
  CHECK(d_params != NULL, "alloc neg dp");
  fill_deterministic_batch(1u, 10u, 8u, 16u, 123u, h0, tokens);
  d_h0[0] = 13.5f;
  d_params[0] = -7.25f;

  et_mtp_loss_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  stats.loss_total = 99.0;

  /* Sequence too short: T <= D + 1 (D=3 => T=4 rejected, T=3 rejected). */
  CHECK(et_mtp_forward_v1(m, 1u, 4u, h0, tokens, &stats) ==
            ET_MTP_ERR_SHAPE_MISMATCH,
        "T=D+1 must fail");
  CHECK(et_mtp_forward_v1(m, 1u, 3u, h0, tokens, &stats) ==
            ET_MTP_ERR_SHAPE_MISMATCH,
        "T=D must fail");

  /* OOB token id. */
  tokens[9] = 16u;
  CHECK(et_mtp_forward_backward_v1(m, 1u, 10u, h0, tokens, &stats, d_h0,
                                   d_params, lay.total_params) ==
            ET_MTP_ERR_TOKEN_RANGE,
        "oob token");
  tokens[9] = 3u;

  /* Non-finite h0. */
  h0[79] = INFINITY;
  CHECK(et_mtp_forward_backward_v1(m, 1u, 10u, h0, tokens, &stats, d_h0,
                                   d_params, lay.total_params) ==
            ET_MTP_ERR_NONFINITE,
        "nonfinite h0");
  h0[79] = 0.25f;

  /* Aliased output (d_h0 == h0). */
  CHECK(et_mtp_forward_backward_v1(m, 1u, 10u, h0, tokens, &stats, h0, d_params,
                                   lay.total_params) == ET_MTP_ERR_INVALID_ARG,
        "aliased d_h0");

  /* FP-env trap: FE_UPWARD and MXCSR FTZ. */
  fenv_t saved;
  CHECK(fegetenv(&saved) == 0, "fegetenv");
  CHECK(fesetround(FE_UPWARD) == 0, "fesetround upward");
  CHECK(et_mtp_forward_backward_v1(m, 1u, 10u, h0, tokens, &stats, d_h0,
                                   d_params, lay.total_params) ==
            ET_MTP_ERR_FP_ENV,
        "FE_UPWARD trap");
  CHECK(fesetenv(&saved) == 0, "restore env 1");

  const unsigned int orig_mxcsr = _mm_getcsr();
  _mm_setcsr(orig_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_mtp_forward_v1(m, 1u, 10u, h0, tokens, &stats) == ET_MTP_ERR_FP_ENV,
        "MXCSR FTZ trap");
  _mm_setcsr(orig_mxcsr);

  /* Negative temperature and -0.0 temperature in draft_verify. */
  float tlog[4 * 16];
  memset(tlog, 0, sizeof(tlog));
  uint32_t drafts[3], emitted[4];
  et_mtp_verify_result_v1 vr;
  CHECK(et_mtp_draft_verify_v1(m, h0, 0u, tlog, -0.0, ET_MTP_NO_EOS, NULL,
                               drafts, emitted, 4u, &vr) ==
            ET_MTP_ERR_INVALID_ARG,
        "-0.0 temp rejected");
  CHECK(et_mtp_draft_verify_v1(m, h0, 0u, tlog, 0.5, ET_MTP_NO_EOS, NULL,
                               drafts, emitted, 4u, &vr) ==
            ET_MTP_ERR_INVALID_ARG,
        "NULL rng for temp > 0 rejected");

  /* Verify sentinels untouched across all failed calls! */
  CHECK(stats.loss_total == 99.0 && d_h0[0] == 13.5f && d_params[0] == -7.25f,
        "sentinels must remain untouched on error");

  printf("MTP NATIVE NEGATIVE+FP_ENV PASS\n");
  free(d_params);
  et_mtp_model_destroy_v1(m);
}

int main(int argc, char **argv) {
  if (argc == 5 && strcmp(argv[1], "--oracle-dump") == 0) {
    const uint32_t D = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t tie = (uint32_t)strtoul(argv[3], NULL, 10);
    return run_oracle_dump(D, tie, argv[4]);
  }
  test_fd_all_depths();
  test_shared_gradient_accumulation();
  test_train_resume_and_spec_verify();
  test_negative_boundaries();
  printf("MTP NATIVE ALL TESTS PASS\n");
  return 0;
}
