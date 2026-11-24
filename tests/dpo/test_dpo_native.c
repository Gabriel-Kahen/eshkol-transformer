#include "eshkol_transformer/dpo_alignment_abi.h"
#include "dpo_eshkol_bridge.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
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

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static float det_val(size_t index, int salt, float scale) {
  const int v =
      (int)((index * 31u + (size_t)salt * 17u + index / 5u) % 37u) - 18;
  return (float)v * scale;
}

static void fill_det(float *buf, size_t count, int salt, float scale) {
  for (size_t i = 0u; i < count; ++i)
    buf[i] = det_val(i, salt, scale);
}

static void fill_targets(int64_t *buf, size_t count, uint32_t v_dim, int salt) {
  for (size_t i = 0u; i < count; ++i)
    buf[i] = (int64_t)((i * 7u + (size_t)salt * 13u + 3u) % v_dim);
}

static void fill_masks(uint8_t *buf, uint32_t b_dim, uint32_t t_dim,
                       uint32_t prefix_zeros) {
  for (uint32_t b = 0u; b < b_dim; ++b) {
    for (uint32_t t = 0u; t < t_dim; ++t) {
      const uint32_t cut =
          (prefix_zeros < t_dim) ? prefix_zeros : (t_dim - 1u);
      buf[(size_t)b * t_dim + t] = (t < cut) ? 0u : 1u;
    }
  }
}

static void poison_bytes(void *buf, size_t bytes, uint8_t pattern) {
  memset(buf, (int)pattern, bytes);
}

static int check_poison_bytes(const void *buf, size_t bytes, uint8_t pattern) {
  const uint8_t *p = (const uint8_t *)buf;
  for (size_t i = 0u; i < bytes; ++i) {
    if (p[i] != pattern)
      return 0;
  }
  return 1;
}

static uint32_t f32_bits(float v) {
  uint32_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static void print_hex_f32(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    printf("%s%a", i == 0u ? "" : " ", (double)buf[i]);
  putchar('\n');
}

static void print_i64_vec(const int64_t *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    printf("%s%lld", i == 0u ? "" : " ", (long long)buf[i]);
  putchar('\n');
}

static void print_u8_vec(const uint8_t *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    printf("%s%u", i == 0u ? "" : " ", (unsigned)buf[i]);
  putchar('\n');
}

/* ---- Oracle vector emitter (--oracle) ---- */

static int run_oracle(void) {
  static const et_dpo_config cases[] = {
      {1u, 1u, 2u, 0u, 0.1, 0.0},
      {2u, 4u, 8u, 0u, 0.5, 0.0},
      {3u, 6u, 16u, 0u, 0.25, 0.1},
      {4u, 8u, 32u, 1u, 1.0, 0.0},
      {2u, 5u, 12u, 1u, 2.0, 0.15},
      {2u, 3u, 6u, 0u, 10.0, 0.0},
      {2u, 3u, 6u, 0u, 10.0, 0.1},
      {8u, 64u, 256u, 0u, 0.2, 0.05},
  };
  const size_t max_logits =
      (size_t)ET_DPO_MAX_B * ET_DPO_MAX_T * ET_DPO_MAX_V;
  const size_t max_tokens = (size_t)ET_DPO_MAX_B * ET_DPO_MAX_T;

  float *pi_w = (float *)calloc(max_logits, sizeof(float));
  float *pi_l = (float *)calloc(max_logits, sizeof(float));
  float *ref_w = (float *)calloc(max_logits, sizeof(float));
  float *ref_l = (float *)calloc(max_logits, sizeof(float));
  float *d_w = (float *)calloc(max_logits, sizeof(float));
  float *d_l = (float *)calloc(max_logits, sizeof(float));
  int64_t *tgt_w = (int64_t *)calloc(max_tokens, sizeof(int64_t));
  int64_t *tgt_l = (int64_t *)calloc(max_tokens, sizeof(int64_t));
  uint8_t *mask_w = (uint8_t *)calloc(max_tokens, sizeof(uint8_t));
  uint8_t *mask_l = (uint8_t *)calloc(max_tokens, sizeof(uint8_t));
  REQUIRE(pi_w && pi_l && ref_w && ref_l && d_w && d_l && tgt_w && tgt_l &&
          mask_w && mask_l);

  for (size_t c = 0u; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    const et_dpo_config cfg = cases[c];
    const size_t bt_n = (size_t)cfg.batch_size * cfg.seq_len;
    const size_t l_n = bt_n * cfg.vocab_size;

    fill_det(pi_w, l_n, (int)(c * 9u + 1u), 0.125f);
    fill_det(pi_l, l_n, (int)(c * 9u + 2u), 0.125f);
    fill_det(ref_w, l_n, (int)(c * 9u + 3u), 0.09375f);
    fill_det(ref_l, l_n, (int)(c * 9u + 4u), 0.09375f);
    fill_targets(tgt_w, bt_n, cfg.vocab_size, (int)(c * 5u + 1u));
    fill_targets(tgt_l, bt_n, cfg.vocab_size, (int)(c * 5u + 2u));
    fill_masks(mask_w, cfg.batch_size, cfg.seq_len, (uint32_t)(c % 3u));
    fill_masks(mask_l, cfg.batch_size, cfg.seq_len, (uint32_t)((c + 1u) % 3u));

    /* Case 5: force extreme positive margin >= +200. */
    if (c == 5u) {
      for (size_t bt = 0u; bt < bt_n; ++bt) {
        if (mask_w[bt] == 1u)
          pi_w[bt * cfg.vocab_size + (size_t)tgt_w[bt]] += 16.0f;
        if (mask_l[bt] == 1u)
          pi_l[bt * cfg.vocab_size + (size_t)tgt_l[bt]] -= 16.0f;
      }
    }
    /* Case 6: force extreme negative margin <= -200. */
    if (c == 6u) {
      for (size_t bt = 0u; bt < bt_n; ++bt) {
        if (mask_w[bt] == 1u)
          pi_w[bt * cfg.vocab_size + (size_t)tgt_w[bt]] -= 16.0f;
        if (mask_l[bt] == 1u)
          pi_l[bt * cfg.vocab_size + (size_t)tgt_l[bt]] += 16.0f;
      }
    }

    const float *rw_arg = (cfg.reference_free == 1u) ? NULL : ref_w;
    const float *rl_arg = (cfg.reference_free == 1u) ? NULL : ref_l;
    et_dpo_telemetry telem;
    REQUIRE(et_dpo_forward_backward(&cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w,
                                    tgt_l, mask_w, mask_l, &telem, d_w,
                                    d_l) == ET_DPO_OK);

    printf("CASE %u %u %u %u %a %a\n", cfg.batch_size, cfg.seq_len,
           cfg.vocab_size, cfg.reference_free, cfg.beta, cfg.label_smoothing);
    print_hex_f32(pi_w, l_n);
    print_hex_f32(pi_l, l_n);
    print_hex_f32(ref_w, l_n);
    print_hex_f32(ref_l, l_n);
    print_i64_vec(tgt_w, bt_n);
    print_i64_vec(tgt_l, bt_n);
    print_u8_vec(mask_w, bt_n);
    print_u8_vec(mask_l, bt_n);
    printf("%a %a %a %a %a\n", telem.mean_loss, telem.chosen_reward_mean,
           telem.rejected_reward_mean, telem.reward_margin_mean,
           telem.preference_accuracy);
    print_hex_f32(d_w, l_n);
    print_hex_f32(d_l, l_n);
  }

  free(pi_w);
  free(pi_l);
  free(ref_w);
  free(ref_l);
  free(d_w);
  free(d_l);
  free(tgt_w);
  free(tgt_l);
  free(mask_w);
  free(mask_l);
  return 0;
}

/* ---- Native unit, numerical, extreme, and negative tests ---- */

static void test_identity_policy_equals_ref(void) {
  const et_dpo_config cfg = {3u, 4u, 7u, 0u, 0.5, 0.0};
  const size_t bt_n = 12u;
  const size_t l_n = 84u;
  float pi_w[84], pi_l[84], ref_w[84], ref_l[84], d_w[84], d_l[84];
  int64_t tgt_w[12], tgt_l[12];
  uint8_t mask_w[12], mask_l[12];

  fill_det(ref_w, l_n, 11, 0.125f);
  fill_det(ref_l, l_n, 13, 0.125f);
  memcpy(pi_w, ref_w, sizeof(pi_w));
  memcpy(pi_l, ref_l, sizeof(pi_l));
  fill_targets(tgt_w, bt_n, cfg.vocab_size, 3);
  fill_targets(tgt_l, bt_n, cfg.vocab_size, 5);
  fill_masks(mask_w, cfg.batch_size, cfg.seq_len, 1u);
  fill_masks(mask_l, cfg.batch_size, cfg.seq_len, 2u);

  et_dpo_telemetry t_fwd, t_fb;
  REQUIRE(et_dpo_forward(&cfg, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l, mask_w,
                         mask_l, &t_fwd) == ET_DPO_OK);
  REQUIRE(et_dpo_forward_backward(&cfg, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l,
                                  mask_w, mask_l, &t_fb, d_w,
                                  d_l) == ET_DPO_OK);
  REQUIRE(memcmp(&t_fwd, &t_fb, sizeof(t_fwd)) == 0);
  REQUIRE(fabs(t_fb.mean_loss - log(2.0)) <= 1e-15);
  REQUIRE(t_fb.chosen_reward_mean == 0.0);
  REQUIRE(t_fb.rejected_reward_mean == 0.0);
  REQUIRE(t_fb.reward_margin_mean == 0.0);
  REQUIRE(t_fb.preference_accuracy == 0.0);
}

static void check_fd_config(const et_dpo_config *cfg, int salt,
                            uint32_t prefix_w, uint32_t prefix_l,
                            size_t *out_checks, double *out_max_err) {
  const size_t bt_n = (size_t)cfg->batch_size * cfg->seq_len;
  const size_t l_n = bt_n * cfg->vocab_size;
  float *pi_w = (float *)calloc(l_n, sizeof(float));
  float *pi_l = (float *)calloc(l_n, sizeof(float));
  float *ref_w = (float *)calloc(l_n, sizeof(float));
  float *ref_l = (float *)calloc(l_n, sizeof(float));
  float *d_w = (float *)calloc(l_n, sizeof(float));
  float *d_l = (float *)calloc(l_n, sizeof(float));
  int64_t *tgt_w = (int64_t *)calloc(bt_n, sizeof(int64_t));
  int64_t *tgt_l = (int64_t *)calloc(bt_n, sizeof(int64_t));
  uint8_t *mask_w = (uint8_t *)calloc(bt_n, sizeof(uint8_t));
  uint8_t *mask_l = (uint8_t *)calloc(bt_n, sizeof(uint8_t));
  REQUIRE(pi_w && pi_l && ref_w && ref_l && d_w && d_l && tgt_w && tgt_l &&
          mask_w && mask_l);

  fill_det(pi_w, l_n, salt + 1, 0.125f);
  fill_det(pi_l, l_n, salt + 2, 0.125f);
  fill_det(ref_w, l_n, salt + 3, 0.09375f);
  fill_det(ref_l, l_n, salt + 4, 0.09375f);
  fill_targets(tgt_w, bt_n, cfg->vocab_size, salt + 5);
  fill_targets(tgt_l, bt_n, cfg->vocab_size, salt + 6);
  fill_masks(mask_w, cfg->batch_size, cfg->seq_len, prefix_w);
  fill_masks(mask_l, cfg->batch_size, cfg->seq_len, prefix_l);

  const float *rw_arg = (cfg->reference_free == 1u) ? NULL : ref_w;
  const float *rl_arg = (cfg->reference_free == 1u) ? NULL : ref_l;

  uint8_t ref_w_sha_before[ET_DPO_SHA256_BYTES];
  uint8_t ref_l_sha_before[ET_DPO_SHA256_BYTES];
  REQUIRE(et_dpo_weight_sha256(l_n, ref_w, ref_w_sha_before) == ET_DPO_OK);
  REQUIRE(et_dpo_weight_sha256(l_n, ref_l, ref_l_sha_before) == ET_DPO_OK);

  et_dpo_telemetry telem;
  REQUIRE(et_dpo_forward_backward(cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_OK);

  /* Verify reference logits strictly untouched. */
  uint8_t ref_w_sha_after[ET_DPO_SHA256_BYTES];
  uint8_t ref_l_sha_after[ET_DPO_SHA256_BYTES];
  REQUIRE(et_dpo_weight_sha256(l_n, ref_w, ref_w_sha_after) == ET_DPO_OK);
  REQUIRE(et_dpo_weight_sha256(l_n, ref_l, ref_l_sha_after) == ET_DPO_OK);
  REQUIRE(memcmp(ref_w_sha_before, ref_w_sha_after, ET_DPO_SHA256_BYTES) == 0);
  REQUIRE(memcmp(ref_l_sha_before, ref_l_sha_after, ET_DPO_SHA256_BYTES) == 0);

  /* Verify exact +0.0f on masked positions and row-sum == 0 on active rows. */
  for (size_t bt = 0u; bt < bt_n; ++bt) {
    double sum_w = 0.0;
    double sum_l = 0.0;
    for (uint32_t v = 0u; v < cfg->vocab_size; ++v) {
      const size_t idx = bt * cfg->vocab_size + v;
      if (mask_w[bt] == 0u)
        REQUIRE(f32_bits(d_w[idx]) == 0u);
      else
        sum_w += (double)d_w[idx];
      if (mask_l[bt] == 0u)
        REQUIRE(f32_bits(d_l[idx]) == 0u);
      else
        sum_l += (double)d_l[idx];
    }
    REQUIRE(fabs(sum_w) <= 1e-6);
    REQUIRE(fabs(sum_l) <= 1e-6);
  }

  /* Central finite-difference check across ALL elements of pi_w and pi_l. */
  const float h = 1.0f / 512.0f;
  for (size_t idx = 0u; idx < l_n; ++idx) {
    et_dpo_telemetry tp, tm;
    const float orig_w = pi_w[idx];
    pi_w[idx] = orig_w + h;
    REQUIRE(et_dpo_forward(cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w, tgt_l,
                           mask_w, mask_l, &tp) == ET_DPO_OK);
    pi_w[idx] = orig_w - h;
    REQUIRE(et_dpo_forward(cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w, tgt_l,
                           mask_w, mask_l, &tm) == ET_DPO_OK);
    pi_w[idx] = orig_w;
    const double fd_w = (tp.mean_loss - tm.mean_loss) / (2.0 * (double)h);
    const double err_w = fabs((double)d_w[idx] - fd_w);
    REQUIRE(err_w <= 1e-5 + 1e-4 * fabs(fd_w));
    if (err_w > *out_max_err)
      *out_max_err = err_w;
    (*out_checks)++;

    const float orig_l = pi_l[idx];
    pi_l[idx] = orig_l + h;
    REQUIRE(et_dpo_forward(cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w, tgt_l,
                           mask_w, mask_l, &tp) == ET_DPO_OK);
    pi_l[idx] = orig_l - h;
    REQUIRE(et_dpo_forward(cfg, pi_w, pi_l, rw_arg, rl_arg, tgt_w, tgt_l,
                           mask_w, mask_l, &tm) == ET_DPO_OK);
    pi_l[idx] = orig_l;
    const double fd_l = (tp.mean_loss - tm.mean_loss) / (2.0 * (double)h);
    const double err_l = fabs((double)d_l[idx] - fd_l);
    REQUIRE(err_l <= 1e-5 + 1e-4 * fabs(fd_l));
    if (err_l > *out_max_err)
      *out_max_err = err_l;
    (*out_checks)++;
  }

  free(pi_w);
  free(pi_l);
  free(ref_w);
  free(ref_l);
  free(d_w);
  free(d_l);
  free(tgt_w);
  free(tgt_l);
  free(mask_w);
  free(mask_l);
}

static void test_central_fd_gradients(void) {
  static const et_dpo_config cfgs[] = {
      /* Standard DPO with prompt mask prefix */
      {2u, 4u, 6u, 0u, 0.5, 0.0},
      /* Conservative DPO (label_smoothing = 0.1) with prompt mask prefix */
      {2u, 4u, 6u, 0u, 0.35, 0.1},
      /* Reference-free mode (ref pointers NULL) */
      {2u, 3u, 5u, 1u, 0.75, 0.0},
      /* Reference-free + label smoothing */
      {2u, 3u, 5u, 1u, 0.4, 0.2},
  };
  size_t total_checks = 0u;
  double max_err = 0.0;
  for (size_t i = 0u; i < sizeof(cfgs) / sizeof(cfgs[0]); ++i)
    check_fd_config(&cfgs[i], (int)(i * 19u + 7u), 2u, 1u, &total_checks,
                    &max_err);
  printf("DPO native central FD gradients: PASS (%zu checks, max_abs_err=%.3e)\n",
         total_checks, max_err);
}

static void test_extreme_margins_and_shift_invariance(void) {
  const uint32_t b_dim = 2u, t_dim = 3u, v_dim = 5u;
  const size_t bt_n = 6u, l_n = 30u;
  float pi_w[30], pi_l[30], ref_w[30], ref_l[30], d_w[30], d_l[30];
  int64_t tgt_w[6] = {0, 1, 2, 3, 4, 0};
  int64_t tgt_l[6] = {4, 3, 2, 1, 0, 4};
  uint8_t mask_w[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  uint8_t mask_l[6] = {0u, 1u, 1u, 0u, 1u, 1u};

  /* 1. Extreme positive margin (+240) with eps=0 and eps=0.1 */
  memset(ref_w, 0, sizeof(ref_w));
  memset(ref_l, 0, sizeof(ref_l));
  memset(pi_w, 0, sizeof(pi_w));
  memset(pi_l, 0, sizeof(pi_l));
  for (size_t bt = 0u; bt < bt_n; ++bt) {
    if (mask_w[bt])
      pi_w[bt * v_dim + (size_t)tgt_w[bt]] = 12.0f;
    if (mask_l[bt])
      pi_l[bt * v_dim + (size_t)tgt_l[bt]] = -12.0f;
  }
  const et_dpo_config cfg_pos = {b_dim, t_dim, v_dim, 0u, 10.0, 0.0};
  et_dpo_telemetry t_pos;
  REQUIRE(et_dpo_forward_backward(&cfg_pos, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &t_pos, d_w,
                                  d_l) == ET_DPO_OK);
  REQUIRE(isfinite(t_pos.mean_loss) && t_pos.reward_margin_mean >= 200.0);
  REQUIRE(t_pos.mean_loss >= 0.0 && t_pos.mean_loss < 1e-80);
  REQUIRE(t_pos.preference_accuracy == 1.0);
  for (size_t i = 0u; i < l_n; ++i) {
    REQUIRE(isfinite(d_w[i]) && isfinite(d_l[i]));
  }
  for (size_t bt = 0u; bt < bt_n; ++bt) {
    if (mask_w[bt] == 0u) {
      for (uint32_t v = 0u; v < v_dim; ++v)
        REQUIRE(f32_bits(d_w[bt * v_dim + v]) == 0u);
    }
  }

  /* With cDPO label_smoothing=0.1 at margin >= +200, loss ~ 0.1 * margin. */
  const et_dpo_config cfg_pos_cdpo = {b_dim, t_dim, v_dim, 0u, 10.0, 0.1};
  et_dpo_telemetry t_pos_cdpo;
  REQUIRE(et_dpo_forward_backward(&cfg_pos_cdpo, pi_w, pi_l, ref_w, ref_l,
                                  tgt_w, tgt_l, mask_w, mask_l, &t_pos_cdpo,
                                  d_w, d_l) == ET_DPO_OK);
  REQUIRE(isfinite(t_pos_cdpo.mean_loss));
  REQUIRE(fabs(t_pos_cdpo.mean_loss - 0.1 * t_pos_cdpo.reward_margin_mean) <
          1e-12);
  REQUIRE(t_pos_cdpo.preference_accuracy == 1.0);

  /* 2. Extreme negative margin (-240) with eps=0 and eps=0.1 */
  const et_dpo_config cfg_neg = {b_dim, t_dim, v_dim, 0u, 10.0, 0.0};
  et_dpo_telemetry t_neg;
  REQUIRE(et_dpo_forward_backward(&cfg_neg, pi_l, pi_w, ref_l, ref_w, tgt_l,
                                  tgt_w, mask_l, mask_w, &t_neg, d_w,
                                  d_l) == ET_DPO_OK);
  REQUIRE(isfinite(t_neg.mean_loss) && t_neg.reward_margin_mean <= -200.0);
  REQUIRE(fabs(t_neg.mean_loss - (-t_neg.reward_margin_mean)) < 1e-12);
  REQUIRE(t_neg.preference_accuracy == 0.0);
  int any_nonzero = 0;
  for (size_t i = 0u; i < l_n; ++i) {
    REQUIRE(isfinite(d_w[i]) && isfinite(d_l[i]));
    if (d_w[i] != 0.0f)
      any_nonzero = 1;
  }
  REQUIRE(any_nonzero);

  const et_dpo_config cfg_neg_cdpo = {b_dim, t_dim, v_dim, 0u, 10.0, 0.1};
  et_dpo_telemetry t_neg_cdpo;
  REQUIRE(et_dpo_forward_backward(&cfg_neg_cdpo, pi_l, pi_w, ref_l, ref_w,
                                  tgt_l, tgt_w, mask_l, mask_w, &t_neg_cdpo,
                                  d_w, d_l) == ET_DPO_OK);
  REQUIRE(fabs(t_neg_cdpo.mean_loss - 0.9 * (-t_neg_cdpo.reward_margin_mean)) <
          1e-12);

  /* 3. Extreme constant logit shift (+1e6f) preserves log_softmax = -log(V). */
  float shifted[30];
  for (size_t i = 0u; i < l_n; ++i)
    shifted[i] = 1000000.0f;
  double logps[2] = {0.0, 0.0};
  REQUIRE(et_dpo_sequence_logps(b_dim, t_dim, v_dim, shifted, tgt_w, mask_w,
                                logps) == ET_DPO_OK);
  for (uint32_t b = 0u; b < b_dim; ++b)
    REQUIRE(fabs(logps[b] - (-2.0 * log((double)v_dim))) <= 1e-12);

  printf("DPO extreme margins (+/-200+) & shift invariance: PASS "
         "(pos_margin=%.2f neg_margin=%.2f)\n",
         t_pos.reward_margin_mean, t_neg.reward_margin_mean);
}

static void test_negative_and_failure_atomicity(void) {
  const et_dpo_config valid = {2u, 3u, 4u, 0u, 0.5, 0.1};
  const size_t bt_n = 6u;
  const size_t l_n = 24u;
  float pi_w[24], pi_l[24], ref_w[24], ref_l[24], d_w[24], d_l[24];
  int64_t tgt_w[6] = {0, 1, 2, 3, 1, 0};
  int64_t tgt_l[6] = {3, 2, 1, 0, 2, 3};
  uint8_t mask_w[6] = {0u, 1u, 1u, 1u, 0u, 1u};
  uint8_t mask_l[6] = {1u, 0u, 1u, 0u, 1u, 1u};
  et_dpo_telemetry telem;
  const uint8_t pat = 0x5Au;

  fill_det(pi_w, l_n, 1, 0.125f);
  fill_det(pi_l, l_n, 2, 0.125f);
  fill_det(ref_w, l_n, 3, 0.125f);
  fill_det(ref_l, l_n, 4, 0.125f);

#define RESET_POISON()                                                         \
  do {                                                                         \
    poison_bytes(&telem, sizeof(telem), pat);                                  \
    poison_bytes(d_w, sizeof(d_w), pat);                                       \
    poison_bytes(d_l, sizeof(d_l), pat);                                       \
  } while (0)

#define CHECK_POISON()                                                         \
  do {                                                                         \
    REQUIRE(check_poison_bytes(&telem, sizeof(telem), pat));                   \
    REQUIRE(check_poison_bytes(d_w, sizeof(d_w), pat));                        \
    REQUIRE(check_poison_bytes(d_l, sizeof(d_l), pat));                        \
  } while (0)

  /* 1. Null pointers */
  RESET_POISON();
  REQUIRE(et_dpo_validate_config(NULL) == ET_DPO_ERR_ARGUMENT);
  REQUIRE(et_dpo_forward_backward(NULL, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, NULL, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, NULL, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, NULL, ref_l, tgt_w, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, NULL, tgt_w, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, NULL, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w, NULL,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, NULL, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, NULL, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, NULL, d_w,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, NULL,
                                  d_l) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  NULL) == ET_DPO_ERR_ARGUMENT);
  CHECK_POISON();

  /* 2. Out-of-range shapes */
  const et_dpo_config bad_shapes[] = {
      {0u, 3u, 4u, 0u, 0.5, 0.0},
      {ET_DPO_MAX_B + 1u, 3u, 4u, 0u, 0.5, 0.0},
      {2u, 0u, 4u, 0u, 0.5, 0.0},
      {2u, ET_DPO_MAX_T + 1u, 4u, 0u, 0.5, 0.0},
      {2u, 3u, 0u, 0u, 0.5, 0.0},
      {2u, 3u, 1u, 0u, 0.5, 0.0},
      {2u, 3u, ET_DPO_MAX_V + 1u, 0u, 0.5, 0.0},
  };
  for (size_t i = 0u; i < sizeof(bad_shapes) / sizeof(bad_shapes[0]); ++i) {
    RESET_POISON();
    REQUIRE(et_dpo_forward_backward(&bad_shapes[i], pi_w, pi_l, ref_w, ref_l,
                                    tgt_w, tgt_l, mask_w, mask_l, &telem, d_w,
                                    d_l) == ET_DPO_ERR_SHAPE);
    CHECK_POISON();
  }

  /* 3. Invalid beta, label_smoothing, reference_free */
  const et_dpo_config bad_args[] = {
      {2u, 3u, 4u, 0u, 0.0, 0.0},   {2u, 3u, 4u, 0u, -0.0, 0.0},
      {2u, 3u, 4u, 0u, -0.5, 0.0},  {2u, 3u, 4u, 0u, 0.5, -0.001},
      {2u, 3u, 4u, 0u, 0.5, 0.5},   {2u, 3u, 4u, 0u, 0.5, 0.75},
      {2u, 3u, 4u, 2u, 0.5, 0.0},
  };
  for (size_t i = 0u; i < sizeof(bad_args) / sizeof(bad_args[0]); ++i) {
    RESET_POISON();
    REQUIRE(et_dpo_forward_backward(&bad_args[i], pi_w, pi_l, ref_w, ref_l,
                                    tgt_w, tgt_l, mask_w, mask_l, &telem, d_w,
                                    d_l) == ET_DPO_ERR_ARGUMENT);
    CHECK_POISON();
  }

  const et_dpo_config bad_nonfin[] = {
      {2u, 3u, 4u, 0u, NAN, 0.0},
      {2u, 3u, 4u, 0u, INFINITY, 0.0},
      {2u, 3u, 4u, 0u, -INFINITY, 0.0},
      {2u, 3u, 4u, 0u, 0.5, NAN},
      {2u, 3u, 4u, 0u, 0.5, INFINITY},
  };
  for (size_t i = 0u; i < sizeof(bad_nonfin) / sizeof(bad_nonfin[0]); ++i) {
    RESET_POISON();
    REQUIRE(et_dpo_forward_backward(&bad_nonfin[i], pi_w, pi_l, ref_w, ref_l,
                                    tgt_w, tgt_l, mask_w, mask_l, &telem, d_w,
                                    d_l) == ET_DPO_ERR_NONFINITE);
    CHECK_POISON();
  }

  /* 4. Aliased buffers */
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_w, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ALIAS);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, pi_w, ref_l, tgt_w, tgt_l,
                                  mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ALIAS);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_w) == ET_DPO_ERR_ALIAS);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, pi_w,
                                  d_l) == ET_DPO_ERR_ALIAS);
  CHECK_POISON();
  /* Partial overlap */
  float big[48];
  fill_det(big, 48u, 9, 0.1f);
  REQUIRE(et_dpo_forward_backward(&valid, big, big + 12, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_ALIAS);
  CHECK_POISON();

  /* 5. Misaligned buffers via _raw (zero misaligned typed pointer UB!) */
  uint8_t raw_bytes[256];
  memset(raw_bytes, 0, sizeof(raw_bytes));
  const void *misaligned = (const void *)(raw_bytes + 1);
  void *misaligned_rw = (void *)(raw_bytes + 1);
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward_raw(misaligned, pi_w, pi_l, ref_w, ref_l,
                                      tgt_w, tgt_l, mask_w, mask_l, &telem, d_w,
                                      d_l) == ET_DPO_ERR_ALIGNMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward_raw(&valid, misaligned, pi_l, ref_w, ref_l,
                                      tgt_w, tgt_l, mask_w, mask_l, &telem, d_w,
                                      d_l) == ET_DPO_ERR_ALIGNMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward_raw(&valid, pi_w, pi_l, ref_w, ref_l,
                                      misaligned, tgt_l, mask_w, mask_l, &telem,
                                      d_w, d_l) == ET_DPO_ERR_ALIGNMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward_raw(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                      tgt_l, mask_w, mask_l, misaligned_rw, d_w,
                                      d_l) == ET_DPO_ERR_ALIGNMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_forward_backward_raw(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                      tgt_l, mask_w, mask_l, &telem,
                                      misaligned_rw,
                                      d_l) == ET_DPO_ERR_ALIGNMENT);
  CHECK_POISON();
  REQUIRE(et_dpo_sequence_logps_raw(2u, 3u, 4u, misaligned, tgt_w, mask_w,
                                    raw_bytes + 128) == ET_DPO_ERR_ALIGNMENT);
  REQUIRE(et_dpo_weight_sha256_raw(4u, misaligned, raw_bytes + 128) ==
          ET_DPO_ERR_ALIGNMENT);

  /* 6. Invalid masks (> 1 or all-zero sequence) */
  uint8_t bad_mask[6];
  memcpy(bad_mask, mask_w, bt_n);
  bad_mask[2] = 2u;
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, bad_mask, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_MASK);
  CHECK_POISON();

  memcpy(bad_mask, mask_l, bt_n);
  bad_mask[3] = 0u;
  bad_mask[4] = 0u;
  bad_mask[5] = 0u; /* second batch row all-zero */
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, bad_mask, &telem, d_w,
                                  d_l) == ET_DPO_ERR_MASK);
  CHECK_POISON();

  /* 7. Out-of-range targets (< 0 or >= V, including on masked positions) */
  int64_t bad_tgt[6];
  memcpy(bad_tgt, tgt_w, sizeof(bad_tgt));
  bad_tgt[0] = -1; /* position 0 has mask_w[0] == 0, still must reject */
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, bad_tgt,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_TARGET);
  CHECK_POISON();

  memcpy(bad_tgt, tgt_l, sizeof(bad_tgt));
  bad_tgt[4] = (int64_t)valid.vocab_size;
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  bad_tgt, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_TARGET);
  CHECK_POISON();

  /* 8. Non-finite logits (including on masked position) & gradient overflow */
  float bad_logits[24];
  memcpy(bad_logits, pi_w, sizeof(bad_logits));
  bad_logits[0] = NAN;
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, bad_logits, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_NONFINITE);
  CHECK_POISON();

  memcpy(bad_logits, ref_l, sizeof(bad_logits));
  bad_logits[11] = -INFINITY;
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&valid, pi_w, pi_l, ref_w, bad_logits, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_NONFINITE);
  CHECK_POISON();

  /* Huge beta with cDPO (|dl_dm| >= 0.1) overflows f32 gradient -> preflight rejects cleanly */
  const et_dpo_config huge_beta = {2u, 3u, 4u, 0u, 1e100, 0.1};
  RESET_POISON();
  REQUIRE(et_dpo_forward_backward(&huge_beta, pi_w, pi_l, ref_w, ref_l, tgt_w,
                                  tgt_l, mask_w, mask_l, &telem, d_w,
                                  d_l) == ET_DPO_ERR_NONFINITE);
  CHECK_POISON();

  /* 9. FP environment checks (rounding mode & x86-64 MXCSR FTZ/DAZ) */
  RESET_POISON();
  REQUIRE(fesetround(FE_UPWARD) == 0);
  const et_dpo_status st_round = et_dpo_forward_backward(
      &valid, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l, mask_w, mask_l, &telem,
      d_w, d_l);
  REQUIRE(fesetround(FE_TONEAREST) == 0);
  REQUIRE(st_round == ET_DPO_ERR_FP_ENV);
  CHECK_POISON();

#if defined(__x86_64__)
  const unsigned int orig_mxcsr = _mm_getcsr();
  /* FTZ bit (0x8000) */
  RESET_POISON();
  _mm_setcsr(orig_mxcsr | 0x8000u);
  const et_dpo_status st_ftz = et_dpo_forward_backward(
      &valid, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l, mask_w, mask_l, &telem,
      d_w, d_l);
  _mm_setcsr(orig_mxcsr);
  REQUIRE(st_ftz == ET_DPO_ERR_FP_ENV);
  CHECK_POISON();

  /* DAZ bit (0x0040) */
  RESET_POISON();
  _mm_setcsr(orig_mxcsr | 0x0040u);
  const et_dpo_status st_daz = et_dpo_forward_backward(
      &valid, pi_w, pi_l, ref_w, ref_l, tgt_w, tgt_l, mask_w, mask_l, &telem,
      d_w, d_l);
  _mm_setcsr(orig_mxcsr);
  REQUIRE(st_daz == ET_DPO_ERR_FP_ENV);
  CHECK_POISON();
#endif

  /* 10. Bridge lifecycle & negative checks */
  et_dpo_aot_ctx *ctx = NULL;
  REQUIRE(et_dpo_aot_create(0, 2, 4, 2, 0.5, 0.0, 0, &ctx) == ET_DPO_ERR_SHAPE);
  REQUIRE(ctx == NULL);
  REQUIRE(et_dpo_aot_create(2, 3, 4, 2, -0.5, 0.0, 0, &ctx) ==
          ET_DPO_ERR_ARGUMENT);
  REQUIRE(ctx == NULL);
  REQUIRE(et_dpo_aot_create(2, 3, 4, 2, 0.5, 0.5, 0, &ctx) ==
          ET_DPO_ERR_ARGUMENT);
  REQUIRE(et_dpo_aot_create(2, 3, 4, 2, 0.5, 0.0, 0, &ctx) == ET_DPO_OK);
  REQUIRE(ctx != NULL);
  REQUIRE(et_dpo_aot_set_policy_chosen_logit(ctx, -1, 0.0) ==
          ET_DPO_ERR_ARGUMENT);
  REQUIRE(et_dpo_aot_set_chosen_mask(ctx, 999, 1) == ET_DPO_ERR_ARGUMENT);
  et_dpo_aot_destroy(ctx);
  et_dpo_aot_destroy(NULL);

#undef RESET_POISON
#undef CHECK_POISON
  printf("DPO negative & failure-atomicity suite: PASS\n");
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return run_oracle();
  test_identity_policy_equals_ref();
  test_central_fd_gradients();
  test_extreme_margins_and_shift_invariance();
  test_negative_and_failure_atomicity();
  printf("DPO native C11 test suite: PASS\n");
  return 0;
}
