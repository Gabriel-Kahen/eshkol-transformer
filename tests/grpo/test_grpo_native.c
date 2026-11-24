#include "eshkol_transformer/grpo_abi.h"
#include "grpo_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON

static int fail(const char *msg, int line) {
  fprintf(stderr, "GRPO native test FAIL at line %d: %s\n", line, msg);
  return 1;
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      return fail(#cond, __LINE__);                                            \
  } while (0)

static void fill_sentinel(void *buf, size_t bytes) {
  memset(buf, 0xA5, bytes);
}

static int is_sentinel(const void *buf, size_t bytes) {
  const uint8_t *p = (const uint8_t *)buf;
  for (size_t i = 0; i < bytes; ++i) {
    if (p[i] != 0xA5u)
      return 0;
  }
  return 1;
}

static uint64_t next_u64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static float rand_sym(uint64_t *state, float scale) {
  const uint32_t r = (uint32_t)(next_u64(state) >> 40u);
  const float u = ((float)r / (float)(1u << 24u)) * 2.0f - 1.0f;
  return u * scale;
}

static void print_f32_hex_vec(const float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

static void print_i32_vec(const int32_t *v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%d", v[i]);
  }
  putchar('\n');
}

static void print_u8_vec(const uint8_t *v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%u", (unsigned)v[i]);
  }
  putchar('\n');
}

/* ---- Oracle Emitter ---- */

static int run_oracle_dump(void) {
  typedef struct oracle_spec {
    uint32_t B;
    uint32_t G;
    uint32_t T;
    uint32_t V;
    uint32_t adv_mode;
    uint32_t norm_mode;
    float adv_eps;
    float clip_low;
    float clip_high;
    float kl_beta;
    int all_equal_rewards;
    int force_clips;
    int include_empty_seq;
    float logit_scale;
    uint64_t seed;
  } oracle_spec;

  static const oracle_spec specs[8] = {
      /* 0: Standard GRPO, global token mean, symmetric clip, unclipped/mixed */
      {2u, 4u, 4u, 6u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
       1e-6f, 0.2f, 0.2f, 0.04f, 0, 0, 0, 1.2f, 1001u},
      /* 1: Dr. GRPO, per-seq mean, asymmetric clip */
      {2u, 3u, 5u, 5u, ET_GRPO_ADV_DR_GRPO, ET_GRPO_NORM_PER_SEQ_MEAN, 1e-6f,
       0.15f, 0.28f, 0.10f, 0, 0, 0, 1.5f, 2002u},
      /* 2: kl_beta = 0.0 (zero KL penalty, non-zero mean_kl still measured) */
      {1u, 4u, 4u, 7u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
       1e-5f, 0.2f, 0.2f, 0.0f, 0, 0, 0, 1.0f, 3003u},
      /* 3: All-equal rewards edge case (advantages == 0) */
      {2u, 3u, 3u, 4u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_PER_SEQ_MEAN, 1e-6f,
       0.2f, 0.2f, 0.08f, 1, 0, 0, 1.0f, 4004u},
      /* 4: Forced upper + lower clipping regime */
      {2u, 2u, 4u, 5u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
       1e-6f, 0.15f, 0.20f, 0.05f, 0, 1, 0, 1.0f, 5005u},
      /* 5: Mixed sequence lengths including one completely masked sequence */
      {2u, 3u, 5u, 4u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_PER_SEQ_MEAN, 1e-6f,
       0.2f, 0.3f, 0.02f, 0, 0, 1, 1.2f, 6006u},
      /* 6: Large logit magnitudes (+/- 40.0) stability check */
      {1u, 4u, 3u, 6u, ET_GRPO_ADV_STANDARD, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
       1e-6f, 0.2f, 0.25f, 0.05f, 0, 0, 0, 40.0f, 7007u},
      /* 7: Dr. GRPO with forced clipping + kl_beta = 0.0 */
      {1u, 4u, 4u, 5u, ET_GRPO_ADV_DR_GRPO, ET_GRPO_NORM_PER_SEQ_MEAN, 0.0f,
       0.10f, 0.12f, 0.0f, 0, 1, 0, 1.1f, 8008u},
  };

  for (size_t c = 0; c < 8u; ++c) {
    const oracle_spec *s = &specs[c];
    const uint32_t N = s->B * s->G;
    const size_t nt = (size_t)N * (size_t)s->T;
    const size_t ntv = nt * (size_t)s->V;
    uint64_t rng = s->seed;

    float rewards[32];
    float adv[32];
    float logits[1024];
    int32_t targets[128];
    uint8_t mask[128];
    float tok_lp[128];
    float seq_lp[32];
    float log_old[128];
    float log_ref[128];
    float grad_logits[1024];
    float grad_log_pi[128];

    for (uint32_t n = 0; n < N; ++n) {
      if (s->all_equal_rewards) {
        rewards[n] = (n < s->G) ? 0.5f : -0.25f;
      } else {
        rewards[n] = rand_sym(&rng, 2.0f);
      }
    }
    CHECK(et_grpo_group_advantages_v1(s->B, s->G, rewards, s->adv_mode,
                                      s->adv_eps, adv) == ET_GRPO_OK);

    for (size_t i = 0; i < ntv; ++i)
      logits[i] = rand_sym(&rng, s->logit_scale);
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t t = 0; t < s->T; ++t) {
        const size_t idx = (size_t)n * (size_t)s->T + (size_t)t;
        targets[idx] = (int32_t)(next_u64(&rng) % s->V);
        if (s->include_empty_seq && n == 1u) {
          mask[idx] = 0u;
        } else if (t == 0u) {
          mask[idx] = 0u; /* prompt token */
        } else if (s->include_empty_seq && n == 2u && t > 1u) {
          mask[idx] = 0u; /* shorter sequence */
        } else {
          mask[idx] = 1u;
        }
      }
    }

    CHECK(et_grpo_token_logprobs_v1(N, s->T, s->V, logits, targets, mask,
                                    tok_lp, seq_lp) == ET_GRPO_OK);

    /* Construct log_old and log_ref away from clip kink points so Python f64 FD
     * can verify every single element unambiguously. */
    for (uint32_t n = 0; n < N; ++n) {
      for (uint32_t t = 0; t < s->T; ++t) {
        const size_t idx = (size_t)n * (size_t)s->T + (size_t)t;
        if (mask[idx] == 0u) {
          log_old[idx] = -1.0f;
          log_ref[idx] = -1.0f;
        } else {
          float shift_old;
          if (s->force_clips) {
            /* Push ratio strongly outside [1 - clip_low, 1 + clip_high] for
             * half the tokens and well inside for the other half. */
            if ((t & 1u) == 1u) {
              shift_old = (adv[n] >= 0.0f) ? -0.45f : 0.45f;
            } else {
              shift_old = 0.02f;
            }
          } else {
            /* Keep rho strictly inside (0.92, 1.08), safely away from kinks. */
            shift_old = rand_sym(&rng, 0.06f);
          }
          log_old[idx] = tok_lp[idx] + shift_old;
          log_ref[idx] = tok_lp[idx] + rand_sym(&rng, 0.25f);
        }
      }
    }

    const et_grpo_loss_config cfg = {N,         s->T,       s->V,
                                     s->norm_mode, s->clip_low, s->clip_high,
                                     s->kl_beta, 0u};
    et_grpo_loss_metrics fwd_m, bwd_m;
    CHECK(et_grpo_loss_forward_v1(&cfg, logits, targets, log_old, log_ref, mask,
                                  adv, rewards, &fwd_m) == ET_GRPO_OK);
    CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, log_old, log_ref,
                                   mask, adv, rewards, grad_logits, grad_log_pi,
                                   &bwd_m) == ET_GRPO_OK);
    CHECK(memcmp(&fwd_m, &bwd_m, sizeof(fwd_m)) == 0);

    printf("CASE %u %u %u %u %u %u %a %a %a %a\n", s->B, s->G, s->T, s->V,
           s->adv_mode, s->norm_mode, (double)s->adv_eps, (double)s->clip_low,
           (double)s->clip_high, (double)s->kl_beta);
    print_f32_hex_vec(rewards, N);
    print_f32_hex_vec(logits, ntv);
    print_i32_vec(targets, nt);
    print_u8_vec(mask, nt);
    print_f32_hex_vec(log_old, nt);
    print_f32_hex_vec(log_ref, nt);
    print_f32_hex_vec(adv, N);
    print_f32_hex_vec(tok_lp, nt);
    print_f32_hex_vec(seq_lp, N);
    printf("%a %a %a %a %a %u %u %u\n", (double)fwd_m.loss,
           (double)fwd_m.pg_loss, (double)fwd_m.mean_kl,
           (double)fwd_m.clip_fraction, (double)fwd_m.mean_reward,
           fwd_m.active_tokens, fwd_m.clipped_tokens, fwd_m.active_seqs);
    print_f32_hex_vec(grad_log_pi, nt);
    print_f32_hex_vec(grad_logits, ntv);
  }
  return 0;
}

/* ---- Native Unit Tests ---- */

static int test_group_advantages(void) {
  const float r[8] = {1.0f, 0.0f, 0.5f, 0.5f, 2.0f, 2.0f, 2.0f, 2.0f};
  float adv_std[8] = {0.0f};
  float adv_dr[8] = {0.0f};

  CHECK(et_grpo_group_advantages_v1(2u, 4u, r, ET_GRPO_ADV_STANDARD, 1e-6f,
                                    adv_std) == ET_GRPO_OK);
  /* Group 0: mean=0.5, var=(0.25+0.25+0+0)/4=0.125, std=sqrt(0.125)+1e-6 */
  const double expected_std = sqrt(0.125) + 1e-6;
  CHECK(fabs((double)adv_std[0] - (0.5 / expected_std)) < 1e-6);
  CHECK(fabs((double)adv_std[1] - (-0.5 / expected_std)) < 1e-6);
  CHECK(fabs((double)adv_std[2]) < 1e-7);
  CHECK(fabs((double)adv_std[3]) < 1e-7);
  /* Group 1: all equal -> exact zero */
  for (int i = 4; i < 8; ++i)
    CHECK(adv_std[i] == 0.0f);

  /* Dr. GRPO: mean-centered only (eps = 0.0f allowed) */
  CHECK(et_grpo_group_advantages_v1(2u, 4u, r, ET_GRPO_ADV_DR_GRPO, 0.0f,
                                    adv_dr) == ET_GRPO_OK);
  CHECK(fabs((double)adv_dr[0] - 0.5) < 1e-7);
  CHECK(fabs((double)adv_dr[1] - (-0.5)) < 1e-7);
  CHECK(adv_dr[2] == 0.0f && adv_dr[3] == 0.0f);
  for (int i = 4; i < 8; ++i)
    CHECK(adv_dr[i] == 0.0f);
  return 0;
}

static int test_token_logprobs_and_fd(double *out_max_rel) {
  const uint32_t N = 2u, T = 3u, V = 5u;
  const size_t nt = (size_t)N * (size_t)T;
  const size_t ntv = nt * (size_t)V;
  uint64_t rng = 12345u;

  float logits[30], shifted[30];
  int32_t targets[6] = {0, 3, 4, 2, 1, 0};
  uint8_t mask[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  float tok_lp[6], tok_lp_shift[6], seq_lp[2], seq_lp_shift[2];
  float grad_tok[6] = {99.0f, 0.7f, -1.2f, -55.0f, 0.4f, 0.9f};
  float grad_logits[30];

  for (size_t i = 0; i < ntv; ++i) {
    logits[i] = rand_sym(&rng, 1.5f);
    shifted[i] = logits[i] + 100.0f;
  }

  CHECK(et_grpo_token_logprobs_v1(N, T, V, logits, targets, mask, tok_lp,
                                  seq_lp) == ET_GRPO_OK);
  CHECK(et_grpo_token_logprobs_v1(N, T, V, shifted, targets, mask, tok_lp_shift,
                                  seq_lp_shift) == ET_GRPO_OK);
  CHECK(tok_lp[0] == 0.0f && tok_lp[3] == 0.0f);
  for (size_t i = 0; i < nt; ++i)
    CHECK(fabs((double)tok_lp[i] - (double)tok_lp_shift[i]) < 1e-5);
  for (uint32_t n = 0; n < N; ++n)
    CHECK(fabs((double)seq_lp[n] - (double)seq_lp_shift[n]) < 1e-5);

  CHECK(et_grpo_token_logprobs_backward_v1(N, T, V, logits, targets, mask,
                                           grad_tok, grad_logits) ==
        ET_GRPO_OK);

  /* Sum over vocab for each (n, t) must be 0 (log-softmax shift invariance). */
  for (size_t i = 0; i < nt; ++i) {
    double row_sum = 0.0;
    for (uint32_t v = 0; v < V; ++v) {
      const float g = grad_logits[i * V + v];
      if (mask[i] == 0u)
        CHECK(g == 0.0f);
      row_sum += (double)g;
    }
    CHECK(fabs(row_sum) < 1e-6);
  }

  /* Central finite differences on scalar objective sum_{n,t} grad_tok * tok_lp */
  const float h = 1.0f / 256.0f; /* optimal power-of-2 step for binary32 FD */
  double max_rel = 0.0;
  for (size_t i = 0; i < ntv; ++i) {
    float orig = logits[i];
    float lp_p[6], lp_m[6];
    logits[i] = orig + h;
    CHECK(et_grpo_token_logprobs_v1(N, T, V, logits, targets, mask, lp_p,
                                    NULL) == ET_GRPO_OK);
    logits[i] = orig - h;
    CHECK(et_grpo_token_logprobs_v1(N, T, V, logits, targets, mask, lp_m,
                                    NULL) == ET_GRPO_OK);
    logits[i] = orig;
    double obj_p = 0.0, obj_m = 0.0;
    for (size_t k = 0; k < nt; ++k) {
      obj_p += (double)grad_tok[k] * (double)lp_p[k];
      obj_m += (double)grad_tok[k] * (double)lp_m[k];
    }
    const double fd = (obj_p - obj_m) / (2.0 * (double)h);
    const double an = (double)grad_logits[i];
    const double err = fabs(an - fd);
    const double scale = fmax(fabs(an), fabs(fd));
    if (scale >= 1e-2) {
      const double rel = err / scale;
      if (rel > max_rel)
        max_rel = rel;
      CHECK(rel <= 1e-4);
    }
    CHECK(err <= 1e-5 + 1e-4 * fabs(fd));
  }
  *out_max_rel = max_rel;
  return 0;
}

static int check_grpo_fd_case(const et_grpo_loss_config *cfg,
                              const float *logits_in, const int32_t *targets,
                              const float *log_old, const float *log_ref,
                              const uint8_t *mask, const float *adv,
                              const float *rewards,
                              uint32_t expected_clipped_min,
                              double *out_max_rel_logits,
                              double *out_max_rel_logpi) {
  const size_t nt = (size_t)cfg->batch_size * (size_t)cfg->seq_len;
  const size_t ntv = nt * (size_t)cfg->vocab_size;

  float logits[128];
  float grad_logits[128];
  float grad_log_pi[32];
  float tok_lp[32];
  float composed_grad_lp[32];
  float composed_grad_logits[128];
  memcpy(logits, logits_in, ntv * sizeof(float));

  et_grpo_loss_metrics m_fwd, m_bwd, m_lp;
  CHECK(et_grpo_loss_forward_v1(cfg, logits, targets, log_old, log_ref, mask,
                                adv, rewards, &m_fwd) == ET_GRPO_OK);
  CHECK(et_grpo_loss_backward_v1(cfg, logits, targets, log_old, log_ref, mask,
                                 adv, rewards, grad_logits, grad_log_pi,
                                 &m_bwd) == ET_GRPO_OK);
  CHECK(memcmp(&m_fwd, &m_bwd, sizeof(m_fwd)) == 0);
  CHECK(m_fwd.mean_kl >= 0.0f);
  CHECK(m_fwd.clipped_tokens >= expected_clipped_min);
  CHECK(fabs((double)m_fwd.loss -
             ((double)m_fwd.pg_loss +
              (double)cfg->kl_beta * (double)m_fwd.mean_kl)) < 1e-6);

  /* Parity between fused logits ops and composed token_logprobs + logprob_loss */
  CHECK(et_grpo_token_logprobs_v1(cfg->batch_size, cfg->seq_len,
                                  cfg->vocab_size, logits, targets, mask,
                                  tok_lp, NULL) == ET_GRPO_OK);
  CHECK(et_grpo_logprob_loss_backward_v1(cfg, tok_lp, log_old, log_ref, mask,
                                         adv, rewards, composed_grad_lp,
                                         &m_lp) == ET_GRPO_OK);
  CHECK(et_grpo_token_logprobs_backward_v1(cfg->batch_size, cfg->seq_len,
                                           cfg->vocab_size, logits, targets,
                                           mask, composed_grad_lp,
                                           composed_grad_logits) == ET_GRPO_OK);
  CHECK(fabs((double)m_lp.loss - (double)m_fwd.loss) < 1e-6);
  for (size_t i = 0; i < nt; ++i)
    CHECK(fabs((double)composed_grad_lp[i] - (double)grad_log_pi[i]) < 1e-6);
  for (size_t i = 0; i < ntv; ++i)
    CHECK(fabs((double)composed_grad_logits[i] - (double)grad_logits[i]) <
          1e-6);

  /* Central FD on log_pi */
  const float h = 1.0f / 256.0f;
  double max_rel_lp = 0.0;
  for (size_t i = 0; i < nt; ++i) {
    float orig = tok_lp[i];
    et_grpo_loss_metrics mp, mm;
    tok_lp[i] = orig + h;
    CHECK(et_grpo_logprob_loss_forward_v1(cfg, tok_lp, log_old, log_ref, mask,
                                          adv, rewards, &mp) == ET_GRPO_OK);
    tok_lp[i] = orig - h;
    CHECK(et_grpo_logprob_loss_forward_v1(cfg, tok_lp, log_old, log_ref, mask,
                                          adv, rewards, &mm) == ET_GRPO_OK);
    tok_lp[i] = orig;
    const double fd = ((double)mp.loss - (double)mm.loss) / (2.0 * (double)h);
    const double an = (double)grad_log_pi[i];
    const double err = fabs(an - fd);
    const double scale = fmax(fabs(an), fabs(fd));
    if (scale >= 1e-2) {
      const double rel = err / scale;
      if (rel > max_rel_lp)
        max_rel_lp = rel;
      CHECK(rel <= 5e-4);
    }
    CHECK(err <= 2e-5 + 1e-4 * fabs(fd));
  }

  /* Central FD on fused pi_logits */
  double max_rel_lg = 0.0;
  for (size_t i = 0; i < ntv; ++i) {
    float orig = logits[i];
    et_grpo_loss_metrics mp, mm;
    logits[i] = orig + h;
    CHECK(et_grpo_loss_forward_v1(cfg, logits, targets, log_old, log_ref, mask,
                                  adv, rewards, &mp) == ET_GRPO_OK);
    logits[i] = orig - h;
    CHECK(et_grpo_loss_forward_v1(cfg, logits, targets, log_old, log_ref, mask,
                                  adv, rewards, &mm) == ET_GRPO_OK);
    logits[i] = orig;
    const double fd = ((double)mp.loss - (double)mm.loss) / (2.0 * (double)h);
    const double an = (double)grad_logits[i];
    const double err = fabs(an - fd);
    const double scale = fmax(fabs(an), fabs(fd));
    if (scale >= 1e-2) {
      const double rel = err / scale;
      if (rel > max_rel_lg)
        max_rel_lg = rel;
      CHECK(rel <= 5e-4);
    }
    CHECK(err <= 2e-5 + 1e-4 * fabs(fd));
  }

  if (max_rel_lg > *out_max_rel_logits)
    *out_max_rel_logits = max_rel_lg;
  if (max_rel_lp > *out_max_rel_logpi)
    *out_max_rel_logpi = max_rel_lp;
  return 0;
}

static int test_grpo_loss_regimes_and_fd(double *out_max_rel_logits,
                                         double *out_max_rel_logpi) {
  const uint32_t N = 4u, T = 3u, V = 4u;
  uint64_t rng = 98765u;
  float logits[48], tok_lp[12], log_old[12], log_ref[12];
  int32_t targets[12] = {0, 1, 2, 3, 2, 1, 1, 0, 3, 2, 3, 0};
  uint8_t mask[12] = {0u, 1u, 1u, 0u, 1u, 1u, 0u, 1u, 0u, 0u, 1u, 1u};
  float rewards[4] = {1.0f, 0.25f, 0.0f, 0.5f};
  float adv[4];

  CHECK(et_grpo_group_advantages_v1(1u, 4u, rewards, ET_GRPO_ADV_STANDARD,
                                    1e-6f, adv) == ET_GRPO_OK);
  for (size_t i = 0; i < 48u; ++i)
    logits[i] = rand_sym(&rng, 1.0f);
  CHECK(et_grpo_token_logprobs_v1(N, T, V, logits, targets, mask, tok_lp,
                                  NULL) == ET_GRPO_OK);

  /* Regime 1: Unclipped (rho well inside [0.85, 1.25]), GLOBAL_TOKEN_MEAN */
  for (size_t i = 0; i < 12u; ++i) {
    log_old[i] = tok_lp[i] + rand_sym(&rng, 0.04f);
    log_ref[i] = tok_lp[i] + rand_sym(&rng, 0.20f);
  }
  const et_grpo_loss_config cfg_unclipped = {
      N, T, V, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN, 0.15f, 0.25f, 0.05f, 0u};
  CHECK(check_grpo_fd_case(&cfg_unclipped, logits, targets, log_old, log_ref,
                           mask, adv, rewards, 0u, out_max_rel_logits,
                           out_max_rel_logpi) == 0);

  /* Regime 2: Both upper-clipped (adv > 0, rho > 1+eps_high) and lower-clipped
   * (adv < 0, rho < 1-eps_low), PER_SEQ_MEAN, kl_beta > 0 */
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t t = 0; t < T; ++t) {
      const size_t idx = (size_t)n * (size_t)T + (size_t)t;
      if (adv[n] > 0.0f) {
        log_old[idx] = tok_lp[idx] - 0.50f; /* rho = exp(+0.5) ~ 1.65 > 1.2 */
      } else {
        log_old[idx] = tok_lp[idx] + 0.50f; /* rho = exp(-0.5) ~ 0.61 < 0.85 */
      }
      log_ref[idx] = tok_lp[idx] + 0.15f;
    }
  }
  const et_grpo_loss_config cfg_clipped = {
      N, T, V, ET_GRPO_NORM_PER_SEQ_MEAN, 0.15f, 0.20f, 0.08f, 0u};
  CHECK(check_grpo_fd_case(&cfg_clipped, logits, targets, log_old, log_ref,
                           mask, adv, rewards, 7u, out_max_rel_logits,
                           out_max_rel_logpi) == 0);

  /* Regime 3: Fully clipped with kl_beta = 0 -> exact zero gradient! */
  const et_grpo_loss_config cfg_clipped_nokl = {
      N, T, V, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN, 0.15f, 0.20f, 0.0f, 0u};
  float zero_grads[48], zero_lp_grads[12];
  et_grpo_loss_metrics m_zero;
  CHECK(et_grpo_loss_backward_v1(&cfg_clipped_nokl, logits, targets, log_old,
                                 log_ref, mask, adv, rewards, zero_grads,
                                 zero_lp_grads, &m_zero) == ET_GRPO_OK);
  CHECK(m_zero.clip_fraction == 1.0f);
  CHECK(m_zero.mean_kl > 0.0f); /* KL still reported even when kl_beta == 0 */
  for (size_t i = 0; i < 48u; ++i)
    CHECK(zero_grads[i] == 0.0f);
  for (size_t i = 0; i < 12u; ++i)
    CHECK(zero_lp_grads[i] == 0.0f);

  /* Regime 4: Read-only input aliasing (log_old == log_ref == tok_lp) */
  et_grpo_loss_metrics m_alias, m_lp_alias;
  CHECK(et_grpo_loss_forward_v1(&cfg_unclipped, logits, targets, tok_lp, tok_lp,
                                mask, adv, NULL, &m_alias) == ET_GRPO_OK);
  CHECK(m_alias.mean_kl >= 0.0f && m_alias.mean_kl <= 1e-12f);
  CHECK(m_alias.clip_fraction == 0.0f);
  CHECK(m_alias.mean_reward == 0.0f);
  CHECK(et_grpo_logprob_loss_forward_v1(&cfg_unclipped, tok_lp, tok_lp, tok_lp,
                                        mask, adv, NULL,
                                        &m_lp_alias) == ET_GRPO_OK);
  CHECK(m_lp_alias.mean_kl == 0.0f);
  CHECK(m_lp_alias.clip_fraction == 0.0f);
  CHECK(m_lp_alias.mean_reward == 0.0f);
  return 0;
}

static int test_verifiers(void) {
  float r = -1.0f;
  CHECK(et_grpo_verify_exact_match_v1("42", 2u, "42", 2u,
                                      ET_GRPO_VERIFY_EXACT_BYTES,
                                      &r) == ET_GRPO_OK);
  CHECK(r == 1.0f);
  CHECK(et_grpo_verify_exact_match_v1(" 42\t\n", 5u, "42", 2u,
                                      ET_GRPO_VERIFY_EXACT_BYTES,
                                      &r) == ET_GRPO_OK);
  CHECK(r == 0.0f);
  CHECK(et_grpo_verify_exact_match_v1(" 42\t\n", 5u, " 42 ", 4u,
                                      ET_GRPO_VERIFY_TRIM_WS,
                                      &r) == ET_GRPO_OK);
  CHECK(r == 1.0f);
  CHECK(et_grpo_verify_exact_match_v1("", 0u, "42", 2u,
                                      ET_GRPO_VERIFY_TRIM_WS,
                                      &r) == ET_GRPO_OK);
  CHECK(r == 0.0f);

  const et_grpo_verify_format_config cfg = {0.25f, 0.75f,
                                            ET_GRPO_VERIFY_TRIM_WS, 0u};
  et_grpo_verify_detail det;

  const char *good = "  <think>1+1=2</think>\n<answer> 2 </answer> ";
  CHECK(et_grpo_verify_format_and_answer_v1(good, strlen(good), "2", 1u, &cfg,
                                            &r, &det) == ET_GRPO_OK);
  CHECK(r == 1.0f && det.format_matched == 1u && det.answer_matched == 1u &&
        det.answer_length == 1u && good[det.answer_offset] == '2');

  const char *wrong_ans = "<think>1+1=3</think><answer>3</answer>";
  CHECK(et_grpo_verify_format_and_answer_v1(wrong_ans, strlen(wrong_ans), "2",
                                            1u, &cfg, &r, &det) == ET_GRPO_OK);
  CHECK(r == 0.25f && det.format_matched == 1u && det.answer_matched == 0u);

  const char *bad_cases[6] = {
      "<think></think><answer>2</answer>",
      "<think>  </think><answer>2</answer>",
      "<think>a<think>b</think></think><answer>2</answer>",
      "<think>ok</think><answer>2</answer>junk",
      "<think>ok</think><answer>   </answer>",
      "2"};
  for (size_t i = 0; i < 6u; ++i) {
    CHECK(et_grpo_verify_format_and_answer_v1(bad_cases[i],
                                              strlen(bad_cases[i]), "2", 1u,
                                              &cfg, &r, &det) == ET_GRPO_OK);
    CHECK(r == 0.0f && det.format_matched == 0u && det.answer_matched == 0u);
  }
  return 0;
}

static int test_negative_and_output_preservation(void) {
  float rewards[4] = {1.0f, 0.0f, 0.5f, 0.25f};
  float adv[4];
  float logits[8] = {0.1f, -0.2f, 0.3f, 0.0f, -0.1f, 0.4f, 0.2f, -0.3f};
  int32_t targets[4] = {0, 1, 1, 0};
  uint8_t mask[4] = {0u, 1u, 1u, 1u};
  float log_old[4] = {-0.5f, -0.6f, -0.7f, -0.8f};
  float log_ref[4] = {-0.5f, -0.6f, -0.7f, -0.8f};
  float tok_lp[4], seq_lp[2], grad_logits[8], grad_lp[4];
  et_grpo_loss_metrics metrics;
  et_grpo_loss_config cfg = {2u, 2u, 2u, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
                             0.2f, 0.2f, 0.05f, 0u};

  /* 1. Group advantages negative tests + sentinel preservation */
  fill_sentinel(adv, sizeof(adv));
  CHECK(et_grpo_group_advantages_v1(1u, 1u, rewards, ET_GRPO_ADV_STANDARD,
                                    1e-6f, adv) == ET_GRPO_ERR_SHAPE);
  CHECK(is_sentinel(adv, sizeof(adv)));
  CHECK(et_grpo_group_advantages_v1(0u, 2u, rewards, ET_GRPO_ADV_STANDARD,
                                    1e-6f, adv) == ET_GRPO_ERR_SHAPE);
  CHECK(is_sentinel(adv, sizeof(adv)));
  CHECK(et_grpo_group_advantages_v1(1u, 4u, rewards, 99u, 1e-6f, adv) ==
        ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(adv, sizeof(adv)));
  CHECK(et_grpo_group_advantages_v1(1u, 4u, rewards, ET_GRPO_ADV_STANDARD, 0.0f,
                                    adv) == ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(adv, sizeof(adv)));
  CHECK(et_grpo_group_advantages_v1(1u, 4u, rewards, ET_GRPO_ADV_STANDARD, NAN,
                                    adv) == ET_GRPO_ERR_NONFINITE);
  CHECK(is_sentinel(adv, sizeof(adv)));
  CHECK(et_grpo_group_advantages_v1(1u, 4u, rewards, ET_GRPO_ADV_STANDARD,
                                    1e-6f, rewards) == ET_GRPO_ERR_ALIAS);

  /* Misaligned raw pointer */
  uint8_t raw_buf[32];
  fill_sentinel(raw_buf, sizeof(raw_buf));
  CHECK(et_grpo_group_advantages_v1_raw(1u, 4u, rewards, ET_GRPO_ADV_STANDARD,
                                        1e-6f,
                                        raw_buf + 1u) == ET_GRPO_ERR_ALIGNMENT);
  CHECK(is_sentinel(raw_buf, sizeof(raw_buf)));

  /* NaN reward */
  float bad_r[4] = {1.0f, NAN, 0.5f, 0.0f};
  CHECK(et_grpo_group_advantages_v1(1u, 4u, bad_r, ET_GRPO_ADV_STANDARD, 1e-6f,
                                    adv) == ET_GRPO_ERR_NONFINITE);
  CHECK(is_sentinel(adv, sizeof(adv)));

  /* 2. Token logprobs negative tests */
  fill_sentinel(tok_lp, sizeof(tok_lp));
  fill_sentinel(seq_lp, sizeof(seq_lp));
  int32_t bad_tgt[4] = {0, 2, 1, 0};
  CHECK(et_grpo_token_logprobs_v1(2u, 2u, 2u, logits, bad_tgt, mask, tok_lp,
                                  seq_lp) == ET_GRPO_ERR_INDEX);
  CHECK(is_sentinel(tok_lp, sizeof(tok_lp)) &&
        is_sentinel(seq_lp, sizeof(seq_lp)));

  int32_t neg_tgt[4] = {0, -1, 1, 0};
  CHECK(et_grpo_token_logprobs_v1(2u, 2u, 2u, logits, neg_tgt, mask, tok_lp,
                                  seq_lp) == ET_GRPO_ERR_INDEX);
  CHECK(is_sentinel(tok_lp, sizeof(tok_lp)));

  uint8_t bad_mask[4] = {0u, 2u, 1u, 1u};
  CHECK(et_grpo_token_logprobs_v1(2u, 2u, 2u, logits, targets, bad_mask, tok_lp,
                                  seq_lp) == ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(tok_lp, sizeof(tok_lp)));

  float bad_logits[8];
  memcpy(bad_logits, logits, sizeof(logits));
  bad_logits[5] = INFINITY;
  CHECK(et_grpo_token_logprobs_v1(2u, 2u, 2u, bad_logits, targets, mask, tok_lp,
                                  seq_lp) == ET_GRPO_ERR_NONFINITE);
  CHECK(is_sentinel(tok_lp, sizeof(tok_lp)));

  CHECK(et_grpo_token_logprobs_v1_raw(2u, 2u, 2u, raw_buf + 1u, targets, mask,
                                      tok_lp, seq_lp) == ET_GRPO_ERR_ALIGNMENT);
  CHECK(is_sentinel(tok_lp, sizeof(tok_lp)));

  /* 3. GRPO loss forward & backward negative tests */
  CHECK(et_grpo_group_advantages_v1(1u, 2u, rewards, ET_GRPO_ADV_STANDARD,
                                    1e-6f, adv) == ET_GRPO_OK);
  fill_sentinel(grad_logits, sizeof(grad_logits));
  fill_sentinel(grad_lp, sizeof(grad_lp));
  fill_sentinel(&metrics, sizeof(metrics));

  /* Empty mask */
  uint8_t zero_mask[4] = {0u, 0u, 0u, 0u};
  CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, log_old, log_ref,
                                 zero_mask, adv, rewards, grad_logits, grad_lp,
                                 &metrics) == ET_GRPO_ERR_EMPTY_MASK);
  CHECK(is_sentinel(grad_logits, sizeof(grad_logits)) &&
        is_sentinel(grad_lp, sizeof(grad_lp)) &&
        is_sentinel(&metrics, sizeof(metrics)));

  /* Ratio overflow (log_pi - log_old > 50) */
  float huge_old[4] = {-0.5f, -60.0f, -0.7f, -0.8f};
  CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, huge_old, log_ref, mask,
                                 adv, rewards, grad_logits, grad_lp,
                                 &metrics) == ET_GRPO_ERR_OVERFLOW);
  CHECK(is_sentinel(grad_logits, sizeof(grad_logits)) &&
        is_sentinel(grad_lp, sizeof(grad_lp)) &&
        is_sentinel(&metrics, sizeof(metrics)));

  /* KL overflow (log_ref - log_pi > 50) */
  float huge_ref[4] = {-0.5f, 60.0f, -0.7f, -0.8f};
  CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, log_old, huge_ref, mask,
                                 adv, rewards, grad_logits, grad_lp,
                                 &metrics) == ET_GRPO_ERR_OVERFLOW);
  CHECK(is_sentinel(grad_logits, sizeof(grad_logits)) &&
        is_sentinel(&metrics, sizeof(metrics)));

  /* Bad config fields */
  et_grpo_loss_config bad_cfg = cfg;
  bad_cfg.clip_eps_low = 0.0f;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.clip_eps_low = 1.5f;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.clip_eps_high = -0.1f;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.kl_beta = -0.01f;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.reserved = 1u;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.mask_norm_mode = 7u;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_ARGUMENT);
  bad_cfg = cfg;
  bad_cfg.vocab_size = 0u;
  CHECK(et_grpo_validate_loss_config(&bad_cfg) == ET_GRPO_ERR_SHAPE);

  /* Output aliases input or another output */
  CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, log_old, log_ref, mask,
                                 adv, rewards, logits, grad_lp,
                                 &metrics) == ET_GRPO_ERR_ALIAS);
  CHECK(is_sentinel(grad_lp, sizeof(grad_lp)) &&
        is_sentinel(&metrics, sizeof(metrics)));
  CHECK(et_grpo_loss_backward_v1(&cfg, logits, targets, log_old, log_ref, mask,
                                 adv, rewards, grad_logits, grad_logits,
                                 &metrics) == ET_GRPO_ERR_ALIAS);

  /* Verifier negative tests + output preservation */
  float vr;
  fill_sentinel(&vr, sizeof(vr));
  CHECK(et_grpo_verify_exact_match_v1("42", 2u, "", 0u, 0u, &vr) ==
        ET_GRPO_ERR_SHAPE);
  CHECK(is_sentinel(&vr, sizeof(vr)));
  CHECK(et_grpo_verify_exact_match_v1("42", 2u, "   ", 3u,
                                      ET_GRPO_VERIFY_TRIM_WS,
                                      &vr) == ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(&vr, sizeof(vr)));
  CHECK(et_grpo_verify_exact_match_v1("42", 2u, "42", 2u, 2u, &vr) ==
        ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(&vr, sizeof(vr)));

  const et_grpo_verify_format_config zero_vcfg = {0.0f, 0.0f, 0u, 0u};
  CHECK(et_grpo_verify_format_and_answer_v1("<think>a</think><answer>1</answer>",
                                            34u, "1", 1u, &zero_vcfg, &vr,
                                            NULL) == ET_GRPO_ERR_ARGUMENT);
  CHECK(is_sentinel(&vr, sizeof(vr)));

  /* FP rounding mode check (FE_UPWARD rejected, then restored to FE_TONEAREST) */
  if (fesetround(FE_UPWARD) == 0) {
    const et_grpo_status fp_rc = et_grpo_validate_loss_config(&cfg);
    CHECK(fesetround(FE_TONEAREST) == 0);
    CHECK(fp_rc == ET_GRPO_ERR_FP_ENV);
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return run_oracle_dump();
  if (argc != 1)
    return fail("unexpected arguments", __LINE__);

  double max_rel_tok_fd = 0.0;
  double max_rel_logits_fd = 0.0;
  double max_rel_logpi_fd = 0.0;

  if (test_group_advantages() != 0)
    return 1;
  if (test_token_logprobs_and_fd(&max_rel_tok_fd) != 0)
    return 1;
  if (test_grpo_loss_regimes_and_fd(&max_rel_logits_fd, &max_rel_logpi_fd) != 0)
    return 1;
  if (test_verifiers() != 0)
    return 1;
  if (test_negative_and_output_preservation() != 0)
    return 1;

  printf("GRPO native C11 tests: PASS "
         "(fd_max_rel_tok=%.3e fd_max_rel_logpi=%.3e fd_max_rel_logits=%.3e)\n",
         max_rel_tok_fd, max_rel_logpi_fd, max_rel_logits_fd);
  return 0;
}
