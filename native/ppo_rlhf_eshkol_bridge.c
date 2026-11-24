#define _POSIX_C_SOURCE 200809L

#include "ppo_rlhf_eshkol_bridge.h"

#include <fcntl.h>
#include <fenv.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define AOT_MAX_BT (ET_PPO_AOT_MAX_BATCH * ET_PPO_AOT_MAX_SEQ_LEN)
#define AOT_MAX_LOGITS (AOT_MAX_BT * ET_PPO_AOT_MAX_VOCAB)
#define AOT_MAX_FEATURES (AOT_MAX_BT * ET_PPO_AOT_MAX_FEAT)
#define AOT_MAX_PARAMS                                                         \
  (ET_PPO_AOT_MAX_VOCAB * ET_PPO_AOT_MAX_FEAT + ET_PPO_AOT_MAX_VOCAB +         \
   ET_PPO_AOT_MAX_FEAT + 1u)
#define AOT_MAX_REF_PARAMS                                                     \
  (ET_PPO_AOT_MAX_VOCAB * ET_PPO_AOT_MAX_FEAT + ET_PPO_AOT_MAX_VOCAB)

struct et_ppo_aot_ctx {
  uint32_t batch_size;
  uint32_t seq_len;
  uint32_t feat_dim;
  uint32_t vocab_size;
  size_t param_count;
  size_t ref_param_count;
  et_ppo_trainer_config cfg;
  uint64_t step;
  et_ppo_rollout_report last_report;
  uint8_t ref_sha256[ET_PPO_SHA256_BYTES];
  uint8_t param_sha256[ET_PPO_SHA256_BYTES];

  float features[AOT_MAX_FEATURES];
  int32_t actions[AOT_MAX_BT];
  float r_env[AOT_MAX_BT];
  uint8_t mask[AOT_MAX_BT];
  float ref_params[AOT_MAX_REF_PARAMS];
  float params[AOT_MAX_PARAMS];
  float adam_m[AOT_MAX_PARAMS];
  float adam_v[AOT_MAX_PARAMS];

  /* Isolated small FD check workspace (B=2, T=3, V=4 -> bt=6, btv=24). */
  et_ppo_loss_config fd_cfg;
  float fd_logits[24];
  float fd_values[6];
  int32_t fd_actions[6];
  float fd_old_logprobs[6];
  float fd_old_values[6];
  float fd_advantages[6];
  float fd_returns[6];
  uint8_t fd_mask[6];
  float fd_d_logits[24];
  float fd_d_values[6];
  et_ppo_loss_metrics fd_metrics;
};

static uint64_t splitmix64_next(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31u);
}

static double u64_to_unit_f64(uint64_t x) {
  return (double)(x >> 11u) * (1.0 / 9007199254740992.0);
}

static int64_t digest_word_be(const uint8_t digest[ET_PPO_SHA256_BYTES],
                              int64_t word_index) {
  if (word_index < 0 || word_index >= 4)
    return 0;
  const size_t off = (size_t)word_index * 8u;
  uint64_t u = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    u = (u << 8u) | (uint64_t)digest[off + i];
  int64_t s = 0;
  memcpy(&s, &u, sizeof(s));
  return s;
}

static float env_reward_for(uint32_t prompt_cls, uint32_t t, uint32_t action,
                            uint32_t vocab_size) {
  const uint32_t best = (prompt_cls + t) % vocab_size;
  const uint32_t second = (best + 1u) % vocab_size;
  if (action == best)
    return 1.0f;
  if (action == second)
    return 0.25f;
  return -0.5f;
}

static float det_small(uint32_t i, uint32_t j) {
  const int32_t bucket = (int32_t)((i * 19u + j * 29u + 7u) % 17u) - 8;
  return 0.015625f * (float)bucket;
}

int32_t et_ppo_aot_create(int64_t batch_size, int64_t seq_len, int64_t feat_dim,
                          int64_t vocab_size, et_ppo_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  *out_ctx = NULL;
  if (batch_size < 1 || batch_size > (int64_t)ET_PPO_AOT_MAX_BATCH ||
      seq_len < 2 || seq_len > (int64_t)ET_PPO_AOT_MAX_SEQ_LEN ||
      feat_dim < 4 || feat_dim > (int64_t)ET_PPO_AOT_MAX_FEAT ||
      vocab_size < 3 || vocab_size > (int64_t)ET_PPO_AOT_MAX_VOCAB)
    return ET_PPO_ERR_SHAPE;

  et_ppo_aot_ctx *ctx = (et_ppo_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_PPO_ERR_NOMEM;

  ctx->batch_size = (uint32_t)batch_size;
  ctx->seq_len = (uint32_t)seq_len;
  ctx->feat_dim = (uint32_t)feat_dim;
  ctx->vocab_size = (uint32_t)vocab_size;
  (void)et_ppo_model_param_count(ctx->feat_dim, ctx->vocab_size,
                                 &ctx->param_count);
  (void)et_ppo_ref_param_count(ctx->feat_dim, ctx->vocab_size,
                               &ctx->ref_param_count);

  /* Sensible defaults; overridden by et_ppo_aot_configure. */
  ctx->cfg.loss_cfg.batch_size = ctx->batch_size;
  ctx->cfg.loss_cfg.seq_len = ctx->seq_len;
  ctx->cfg.loss_cfg.vocab_size = ctx->vocab_size;
  ctx->cfg.loss_cfg.use_value_clip = 1u;
  ctx->cfg.loss_cfg.clip_eps_low = 0.2f;
  ctx->cfg.loss_cfg.clip_eps_high = 0.2f;
  ctx->cfg.loss_cfg.value_clip_eps = 0.2f;
  ctx->cfg.loss_cfg.vf_coef = 0.5f;
  ctx->cfg.loss_cfg.ent_coef = 0.01f;
  ctx->cfg.loss_cfg.reserved = 0u;

  ctx->cfg.optim_cfg.lr = 0.06f;
  ctx->cfg.optim_cfg.beta1 = 0.9f;
  ctx->cfg.optim_cfg.beta2 = 0.999f;
  ctx->cfg.optim_cfg.eps = 1e-8f;
  ctx->cfg.optim_cfg.weight_decay = 0.01f;
  ctx->cfg.optim_cfg.max_grad_norm = 1.0f;
  ctx->cfg.optim_cfg.reserved[0] = 0u;
  ctx->cfg.optim_cfg.reserved[1] = 0u;

  ctx->cfg.feat_dim = ctx->feat_dim;
  ctx->cfg.ppo_epochs = 4u;
  ctx->cfg.whiten_advantages = 1u;
  ctx->cfg.reserved = 0u;
  ctx->cfg.gamma = 0.95f;
  ctx->cfg.gae_lambda = 0.90f;
  ctx->cfg.beta_kl = 0.02f;
  ctx->cfg.adv_eps = 1e-6f;
  ctx->cfg.target_kl = 0.25f;
  ctx->cfg.reserved2 = 0u;

  *out_ctx = ctx;
  return ET_PPO_OK;
}

void et_ppo_aot_destroy(et_ppo_aot_ctx *ctx) { free(ctx); }

int32_t et_ppo_aot_configure(et_ppo_aot_ctx *ctx, int64_t ppo_epochs,
                             int64_t use_value_clip, int64_t whiten_advantages,
                             double clip_eps_low, double clip_eps_high,
                             double value_clip_eps, double vf_coef,
                             double ent_coef, double gamma, double gae_lambda,
                             double beta_kl, double adv_eps, double target_kl,
                             double lr, double weight_decay,
                             double max_grad_norm) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (ppo_epochs < 1 || ppo_epochs > (int64_t)ET_PPO_MAX_EPOCHS ||
      use_value_clip < 0 || use_value_clip > 1 || whiten_advantages < 0 ||
      whiten_advantages > 1)
    return ET_PPO_ERR_ARGUMENT;

  et_ppo_trainer_config next = ctx->cfg;
  next.loss_cfg.use_value_clip = (uint32_t)use_value_clip;
  next.loss_cfg.clip_eps_low = (float)clip_eps_low;
  next.loss_cfg.clip_eps_high = (float)clip_eps_high;
  next.loss_cfg.value_clip_eps = (float)value_clip_eps;
  next.loss_cfg.vf_coef = (float)vf_coef;
  next.loss_cfg.ent_coef = (float)ent_coef;

  next.optim_cfg.lr = (float)lr;
  next.optim_cfg.weight_decay = (float)weight_decay;
  next.optim_cfg.max_grad_norm = (float)max_grad_norm;

  next.ppo_epochs = (uint32_t)ppo_epochs;
  next.whiten_advantages = (uint32_t)whiten_advantages;
  next.gamma = (float)gamma;
  next.gae_lambda = (float)gae_lambda;
  next.beta_kl = (float)beta_kl;
  next.adv_eps = (float)adv_eps;
  next.target_kl = (float)target_kl;

  const et_ppo_status st = et_ppo_validate_trainer_config(&next);
  if (st != ET_PPO_OK)
    return st;
  ctx->cfg = next;
  return ET_PPO_OK;
}

int32_t et_ppo_aot_init_task(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;

  const uint32_t B = ctx->batch_size;
  const uint32_t T = ctx->seq_len;
  const uint32_t D = ctx->feat_dim;
  const uint32_t V = ctx->vocab_size;

  memset(ctx->adam_m, 0, sizeof(ctx->adam_m));
  memset(ctx->adam_v, 0, sizeof(ctx->adam_v));
  ctx->step = 0u;

  /* Deterministic features X[B, T, D] separating prompt class p = b % 4 and
   * step t, with t == 0 masked as prompt token (mask = 0) and t >= 1 active. */
  for (uint32_t b = 0u; b < B; ++b) {
    const uint32_t pcls = b % 4u;
    for (uint32_t t = 0u; t < T; ++t) {
      const size_t idx = (size_t)b * (size_t)T + (size_t)t;
      ctx->mask[idx] = t == 0u ? 0u : 1u;
      ctx->actions[idx] = 0;
      ctx->r_env[idx] = 0.0f;
      float *frow = ctx->features + idx * (size_t)D;
      for (uint32_t d = 0u; d < D; ++d) {
        float val = det_small(pcls * 13u + t, d);
        if (d == (pcls % D))
          val += 1.0f;
        if (D >= 8u && d == 4u + (t % 4u))
          val += 0.85f;
        else if (D < 8u && d == (t % D))
          val += 0.5f;
        frow[d] = val;
      }
    }
  }

  /* Initialize Actor (W_pi, b_pi) with small deterministic weights slightly
   * tilted away from optimal actions so alignment clearly lifts reward, and
   * snapshot frozen Reference Policy (W_ref, b_ref) = initial Actor. */
  const size_t b_pi_off = (size_t)V * (size_t)D;
  const size_t w_v_off = b_pi_off + (size_t)V;
  const size_t b_v_off = w_v_off + (size_t)D;
  for (uint32_t v = 0u; v < V; ++v) {
    for (uint32_t d = 0u; d < D; ++d) {
      const float w = det_small(v + 3u, d + 5u);
      ctx->params[(size_t)v * (size_t)D + (size_t)d] = w;
    }
    ctx->params[b_pi_off + (size_t)v] = (v % 2u == 0u) ? -0.15f : 0.15f;
  }
  for (uint32_t d = 0u; d < D; ++d)
    ctx->params[w_v_off + (size_t)d] = 0.0f;
  ctx->params[b_v_off] = 0.0f;

  memcpy(ctx->ref_params, ctx->params, ctx->ref_param_count * sizeof(float));
  return et_ppo_aot_refresh_digests(ctx);
}

/* ---- In-Eshkol FD Workspace ---- */

int32_t et_ppo_aot_setup_fd_case(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;

  ctx->fd_cfg.batch_size = 2u;
  ctx->fd_cfg.seq_len = 3u;
  ctx->fd_cfg.vocab_size = 4u;
  ctx->fd_cfg.use_value_clip = 1u;
  ctx->fd_cfg.clip_eps_low = 0.2f;
  ctx->fd_cfg.clip_eps_high = 0.2f;
  ctx->fd_cfg.value_clip_eps = 0.25f;
  ctx->fd_cfg.vf_coef = 0.5f;
  ctx->fd_cfg.ent_coef = 0.05f;
  ctx->fd_cfg.reserved = 0u;

  /* Chosen away from clip kinks, exercising both unclipped & clipped branches
   * for policy and value heads, plus masked prompt tokens at t=0. */
  static const float init_logits[24] = {
      0.20f,  -0.10f, 0.35f, -0.25f, 0.45f, -0.20f, 0.10f,  -0.30f,
      -0.15f, 0.50f,  0.05f, -0.10f, 0.10f, 0.25f,  -0.35f, 0.00f,
      -0.25f, 0.15f,  0.60f, -0.20f, 0.30f, -0.40f, 0.20f,  0.55f};
  static const int32_t init_actions[6] = {0, 0, 1, 2, 2, 3};
  static const uint8_t init_mask[6] = {0u, 1u, 1u, 0u, 1u, 1u};
  static const float init_values[6] = {0.10f, 0.42f, -0.18f,
                                       0.05f, 0.72f, -0.35f};
  static const float init_old_values[6] = {0.10f, 0.35f, -0.25f,
                                           0.05f, 0.20f, -0.30f};
  static const float init_advantages[6] = {0.0f,  0.65f, -0.45f,
                                           0.0f,  0.80f, -0.55f};
  static const float init_returns[6] = {0.0f,  0.90f, -0.60f,
                                        0.0f,  1.20f, -0.80f};

  memcpy(ctx->fd_logits, init_logits, sizeof(init_logits));
  memcpy(ctx->fd_actions, init_actions, sizeof(init_actions));
  memcpy(ctx->fd_mask, init_mask, sizeof(init_mask));
  memcpy(ctx->fd_values, init_values, sizeof(init_values));
  memcpy(ctx->fd_old_values, init_old_values, sizeof(init_old_values));
  memcpy(ctx->fd_advantages, init_advantages, sizeof(init_advantages));
  memcpy(ctx->fd_returns, init_returns, sizeof(init_returns));

  et_ppo_status st = et_ppo_token_logprobs_v1(
      2u, 3u, 4u, ctx->fd_logits, ctx->fd_actions, ctx->fd_mask,
      ctx->fd_old_logprobs);
  if (st != ET_PPO_OK)
    return st;

  /* Perturb old_logprobs cleanly so token 1 is unclipped (rho ~ 1.05),
   * token 2 is unclipped (rho ~ 0.92), token 4 is upper-clipped (A>0, rho ~ 1.49),
   * token 5 is lower-clipped (A<0, rho ~ 0.67). Value token 4 is value-clipped
   * (v=0.72, old_v=0.20, eps=0.25 -> v_clip=0.45, R=1.20). */
  ctx->fd_old_logprobs[1] -= 0.05f;
  ctx->fd_old_logprobs[2] += 0.08f;
  ctx->fd_old_logprobs[4] -= 0.40f;
  ctx->fd_old_logprobs[5] += 0.40f;
  return ET_PPO_OK;
}

int32_t et_ppo_aot_fd_loss_forward(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  return et_ppo_loss_forward_v1(&ctx->fd_cfg, ctx->fd_logits, ctx->fd_values,
                                ctx->fd_actions, ctx->fd_old_logprobs,
                                ctx->fd_old_values, ctx->fd_advantages,
                                ctx->fd_returns, ctx->fd_mask,
                                &ctx->fd_metrics);
}

int32_t et_ppo_aot_fd_loss_backward(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  return et_ppo_loss_backward_v1(
      &ctx->fd_cfg, ctx->fd_logits, ctx->fd_values, ctx->fd_actions,
      ctx->fd_old_logprobs, ctx->fd_old_values, ctx->fd_advantages,
      ctx->fd_returns, ctx->fd_mask, ctx->fd_d_logits, ctx->fd_d_values,
      &ctx->fd_metrics);
}

double et_ppo_aot_fd_total_loss(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->fd_metrics.total_loss : 0.0;
}

double et_ppo_aot_fd_get_logit(const et_ppo_aot_ctx *ctx, int64_t idx,
                               int64_t v) {
  if (ctx == NULL || idx < 0 || idx >= 6 || v < 0 || v >= 4)
    return 0.0;
  return (double)ctx->fd_logits[(size_t)idx * 4u + (size_t)v];
}

int32_t et_ppo_aot_fd_set_logit(et_ppo_aot_ctx *ctx, int64_t idx, int64_t v,
                                double value) {
  if (ctx == NULL || idx < 0 || idx >= 6 || v < 0 || v >= 4)
    return ET_PPO_ERR_INDEX;
  ctx->fd_logits[(size_t)idx * 4u + (size_t)v] = (float)value;
  return ET_PPO_OK;
}

double et_ppo_aot_fd_get_d_logit(const et_ppo_aot_ctx *ctx, int64_t idx,
                                 int64_t v) {
  if (ctx == NULL || idx < 0 || idx >= 6 || v < 0 || v >= 4)
    return 0.0;
  return (double)ctx->fd_d_logits[(size_t)idx * 4u + (size_t)v];
}

double et_ppo_aot_fd_get_value(const et_ppo_aot_ctx *ctx, int64_t idx) {
  if (ctx == NULL || idx < 0 || idx >= 6)
    return 0.0;
  return (double)ctx->fd_values[(size_t)idx];
}

int32_t et_ppo_aot_fd_set_value(et_ppo_aot_ctx *ctx, int64_t idx,
                                double value) {
  if (ctx == NULL || idx < 0 || idx >= 6)
    return ET_PPO_ERR_INDEX;
  ctx->fd_values[(size_t)idx] = (float)value;
  return ET_PPO_OK;
}

double et_ppo_aot_fd_get_d_value(const et_ppo_aot_ctx *ctx, int64_t idx) {
  if (ctx == NULL || idx < 0 || idx >= 6)
    return 0.0;
  return (double)ctx->fd_d_values[(size_t)idx];
}

int64_t et_ppo_aot_fd_bt(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? 6 : 0;
}

int64_t et_ppo_aot_fd_vocab(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? 4 : 0;
}

/* ---- On-Policy Rollout Sampling, Training & Metrics ---- */

int32_t et_ppo_aot_sample_rollout(et_ppo_aot_ctx *ctx, int64_t iter_idx) {
  if (ctx == NULL || iter_idx < 0)
    return ET_PPO_ERR_ARGUMENT;

  const uint32_t B = ctx->batch_size;
  const uint32_t T = ctx->seq_len;
  const uint32_t D = ctx->feat_dim;
  const uint32_t V = ctx->vocab_size;

  float logits[AOT_MAX_LOGITS];
  float values[AOT_MAX_BT];
  et_ppo_status st = et_ppo_actor_critic_forward_v1(B, T, D, V, ctx->features,
                                                    ctx->params, logits, values);
  if (st != ET_PPO_OK)
    return st;

  const uint64_t iter_key =
      UINT64_C(0x50504f524c484601) ^ ((uint64_t)iter_idx * UINT64_C(0x9e3779b97f4a7c15));

  for (uint32_t b = 0u; b < B; ++b) {
    const uint32_t pcls = b % 4u;
    for (uint32_t t = 0u; t < T; ++t) {
      const size_t idx = (size_t)b * (size_t)T + (size_t)t;
      if (ctx->mask[idx] == 0u) {
        ctx->actions[idx] = 0;
        ctx->r_env[idx] = 0.0f;
        continue;
      }
      const float *row = logits + idx * (size_t)V;
      double max_z = (double)row[0];
      for (uint32_t v = 1u; v < V; ++v) {
        if ((double)row[v] > max_z)
          max_z = (double)row[v];
      }
      double probs[ET_PPO_AOT_MAX_VOCAB];
      double sum = 0.0;
      for (uint32_t v = 0u; v < V; ++v) {
        probs[v] = exp((double)row[v] - max_z);
        sum += probs[v];
      }
      const uint64_t h = splitmix64_next(
          iter_key ^ ((uint64_t)b * UINT64_C(0x100000001b3) + (uint64_t)t));
      const double u = u64_to_unit_f64(h);
      double cdf = 0.0;
      uint32_t chosen = V - 1u;
      for (uint32_t v = 0u; v < V; ++v) {
        cdf += probs[v] / sum;
        if (u < cdf) {
          chosen = v;
          break;
        }
      }
      ctx->actions[idx] = (int32_t)chosen;
      ctx->r_env[idx] = env_reward_for(pcls, t, chosen, V);
    }
  }
  return ET_PPO_OK;
}

int32_t et_ppo_aot_train_rollout(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  et_ppo_status st = et_ppo_train_rollout_v1(
      &ctx->cfg, ctx->features, ctx->actions, ctx->r_env, ctx->mask,
      ctx->ref_params, ctx->params, ctx->adam_m, ctx->adam_v, &ctx->step,
      &ctx->last_report);
  if (st != ET_PPO_OK)
    return st;
  return et_ppo_aot_refresh_digests(ctx);
}

int32_t et_ppo_aot_set_target_kl(et_ppo_aot_ctx *ctx, double target_kl) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  et_ppo_trainer_config next = ctx->cfg;
  next.target_kl = (float)target_kl;
  const et_ppo_status st = et_ppo_validate_trainer_config(&next);
  if (st != ET_PPO_OK)
    return st;
  ctx->cfg = next;
  return ET_PPO_OK;
}

double et_ppo_aot_expected_env_reward(const et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return 0.0;
  const uint32_t B = ctx->batch_size;
  const uint32_t T = ctx->seq_len;
  const uint32_t D = ctx->feat_dim;
  const uint32_t V = ctx->vocab_size;

  float logits[AOT_MAX_LOGITS];
  float values[AOT_MAX_BT];
  if (et_ppo_actor_critic_forward_v1(B, T, D, V, ctx->features, ctx->params,
                                     logits, values) != ET_PPO_OK)
    return 0.0;

  double total = 0.0;
  uint32_t active = 0u;
  for (uint32_t b = 0u; b < B; ++b) {
    const uint32_t pcls = b % 4u;
    for (uint32_t t = 0u; t < T; ++t) {
      const size_t idx = (size_t)b * (size_t)T + (size_t)t;
      if (ctx->mask[idx] == 0u)
        continue;
      const float *row = logits + idx * (size_t)V;
      double max_z = (double)row[0];
      for (uint32_t v = 1u; v < V; ++v) {
        if ((double)row[v] > max_z)
          max_z = (double)row[v];
      }
      double sum = 0.0;
      double exps[ET_PPO_AOT_MAX_VOCAB];
      for (uint32_t v = 0u; v < V; ++v) {
        exps[v] = exp((double)row[v] - max_z);
        sum += exps[v];
      }
      double er = 0.0;
      for (uint32_t v = 0u; v < V; ++v)
        er += (exps[v] / sum) * (double)env_reward_for(pcls, t, v, V);
      total += er;
      ++active;
    }
  }
  return active > 0u ? (total / (double)active) : 0.0;
}

double et_ppo_aot_report_mean_env_reward(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.mean_env_reward : 0.0;
}

double et_ppo_aot_report_mean_shaped_reward(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.mean_shaped_reward : 0.0;
}

double et_ppo_aot_report_total_loss(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.total_loss : 0.0;
}

double et_ppo_aot_report_policy_loss(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.policy_loss : 0.0;
}

double et_ppo_aot_report_value_loss(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.value_loss : 0.0;
}

double et_ppo_aot_report_entropy(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.mean_entropy
                     : 0.0;
}

double et_ppo_aot_report_approx_kl(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.approx_kl : 0.0;
}

double et_ppo_aot_report_clip_fraction(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (double)ctx->last_report.final_metrics.clip_fraction
                     : 0.0;
}

double et_ppo_aot_report_explained_variance(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL
             ? (double)ctx->last_report.final_metrics.explained_variance
             : 0.0;
}

int64_t et_ppo_aot_report_epochs_completed(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->last_report.epochs_completed : 0;
}

int64_t et_ppo_aot_report_early_stopped(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->last_report.early_stopped : 0;
}

int64_t et_ppo_aot_step(const et_ppo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->step : 0;
}

int32_t et_ppo_aot_refresh_digests(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  et_ppo_status st = et_ppo_weight_sha256_v1(
      ctx->ref_param_count, ctx->ref_params, ctx->ref_sha256);
  if (st != ET_PPO_OK)
    return st;
  return et_ppo_weight_sha256_v1(ctx->param_count, ctx->params,
                                 ctx->param_sha256);
}

int64_t et_ppo_aot_ref_sha256_word(const et_ppo_aot_ctx *ctx,
                                   int64_t word_index) {
  return ctx != NULL ? digest_word_be(ctx->ref_sha256, word_index) : 0;
}

int64_t et_ppo_aot_param_sha256_word(const et_ppo_aot_ctx *ctx,
                                     int64_t word_index) {
  return ctx != NULL ? digest_word_be(ctx->param_sha256, word_index) : 0;
}

/* ---- Fresh-Process Checkpoint Env Helpers ---- */

int64_t et_ppo_aot_env_phase(void) {
  const char *p = getenv("PPO_AOT_PHASE");
  if (p != NULL && strcmp(p, "reload") == 0)
    return 1;
  return 0;
}

static int copy_file_modified(const char *src_path, const char *dst_path,
                              int flip_byte, int truncate_tail) {
  const int in_fd = open(src_path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (in_fd < 0)
    return 0;
  uint8_t buf[ET_PPO_CHECKPOINT_MAX_BYTES];
  size_t used = 0u;
  while (used < sizeof(buf)) {
    const ssize_t n = read(in_fd, buf + used, sizeof(buf) - used);
    if (n == 0)
      break;
    if (n < 0) {
      (void)close(in_fd);
      return 0;
    }
    used += (size_t)n;
  }
  (void)close(in_fd);
  if (used < 256u)
    return 0;
  if (flip_byte)
    buf[200] ^= 0x5au;
  if (truncate_tail)
    used -= 17u;

  const int out_fd =
      open(dst_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out_fd < 0)
    return 0;
  size_t off = 0u;
  while (off < used) {
    const ssize_t n = write(out_fd, buf + off, used - off);
    if (n <= 0) {
      (void)close(out_fd);
      return 0;
    }
    off += (size_t)n;
  }
  return close(out_fd) == 0;
}

int32_t et_ppo_aot_save_env_ckpt(const et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  const char *ckpt = getenv("PPO_AOT_CKPT");
  if (ckpt == NULL || ckpt[0] == '\0')
    return ET_PPO_ERR_ARGUMENT;

  et_ppo_status st = et_ppo_checkpoint_save_v1(
      ckpt, &ctx->cfg, ctx->step, ctx->ref_params, ctx->params, ctx->adam_m,
      ctx->adam_v);
  if (st != ET_PPO_OK)
    return st;

  const char *tamper = getenv("PPO_AOT_TAMPER");
  if (tamper != NULL && tamper[0] != '\0') {
    if (!copy_file_modified(ckpt, tamper, 1, 0))
      return ET_PPO_ERR_IO;
  }
  const char *trunc = getenv("PPO_AOT_TRUNC");
  if (trunc != NULL && trunc[0] != '\0') {
    if (!copy_file_modified(ckpt, trunc, 0, 1))
      return ET_PPO_ERR_IO;
  }
  const char *sym = getenv("PPO_AOT_SYMLINK");
  if (sym != NULL && sym[0] != '\0') {
    (void)unlink(sym);
    if (symlink(ckpt, sym) != 0)
      return ET_PPO_ERR_IO;
  }
  return ET_PPO_OK;
}

int32_t et_ppo_aot_load_env_ckpt(et_ppo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  const char *ckpt = getenv("PPO_AOT_CKPT");
  if (ckpt == NULL || ckpt[0] == '\0')
    return ET_PPO_ERR_ARGUMENT;

  et_ppo_checkpoint_meta meta;
  memset(&meta, 0, sizeof(meta));
  et_ppo_status st = et_ppo_checkpoint_load_v1(
      ckpt, ctx->feat_dim, ctx->vocab_size, ctx->ref_params, &meta, ctx->params,
      ctx->adam_m, ctx->adam_v);
  if (st != ET_PPO_OK)
    return st;
  ctx->step = meta.step;
  ctx->cfg = meta.cfg;
  return et_ppo_aot_refresh_digests(ctx);
}

int32_t et_ppo_aot_check_ckpt_negative(const et_ppo_aot_ctx *ctx,
                                       int64_t case_id) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;
  et_ppo_checkpoint_meta meta;
  float p[AOT_MAX_PARAMS];
  float m[AOT_MAX_PARAMS];
  float v[AOT_MAX_PARAMS];
  memset(&meta, 0, sizeof(meta));

  switch (case_id) {
  case 0: {
    const char *tamper = getenv("PPO_AOT_TAMPER");
    return et_ppo_checkpoint_load_v1(tamper, ctx->feat_dim, ctx->vocab_size,
                                     ctx->ref_params, &meta, p, m, v);
  }
  case 1: {
    const char *trunc = getenv("PPO_AOT_TRUNC");
    return et_ppo_checkpoint_load_v1(trunc, ctx->feat_dim, ctx->vocab_size,
                                     ctx->ref_params, &meta, p, m, v);
  }
  case 2: {
    const char *sym = getenv("PPO_AOT_SYMLINK");
    return et_ppo_checkpoint_load_v1(sym, ctx->feat_dim, ctx->vocab_size,
                                     ctx->ref_params, &meta, p, m, v);
  }
  case 3: {
    const char *ckpt = getenv("PPO_AOT_CKPT");
    float wrong_ref[AOT_MAX_REF_PARAMS];
    memcpy(wrong_ref, ctx->ref_params, ctx->ref_param_count * sizeof(float));
    wrong_ref[0] += 0.25f;
    return et_ppo_checkpoint_load_v1(ckpt, ctx->feat_dim, ctx->vocab_size,
                                     wrong_ref, &meta, p, m, v);
  }
  default:
    return ET_PPO_ERR_ARGUMENT;
  }
}

/* ---- Core Negative Boundary Probes ---- */

int32_t et_ppo_aot_check_negative_case(const et_ppo_aot_ctx *ctx,
                                       int64_t case_id) {
  if (ctx == NULL)
    return ET_PPO_ERR_ARGUMENT;

  et_ppo_loss_config cfg = {1u, 2u, 3u, 1u, 0.2f, 0.2f, 0.2f, 0.5f, 0.01f, 0u};
  float logits[6] = {0.1f, -0.2f, 0.3f, 0.0f, 0.4f, -0.1f};
  float values[2] = {0.2f, -0.1f};
  int32_t actions[2] = {0, 2};
  float old_lp[2] = {-1.0f, -1.1f};
  float old_v[2] = {0.15f, -0.05f};
  float adv[2] = {0.5f, -0.5f};
  float ret[2] = {0.7f, -0.6f};
  uint8_t mask[2] = {1u, 1u};
  float out_a[2] = {0.0f, 0.0f};
  float out_r[2] = {0.0f, 0.0f};
  et_ppo_loss_metrics metrics;
  memset(&metrics, 0, sizeof(metrics));

  switch (case_id) {
  case 0:
    logits[2] = NAN;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 1:
    values[1] = INFINITY;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 2:
    return et_ppo_gae_v1(1u, 2u, 1.1f, 0.95f, ret, values, mask, out_a, out_r);
  case 3:
    return et_ppo_gae_v1(1u, 2u, 0.99f, -0.05f, ret, values, mask, out_a, out_r);
  case 4:
    return et_ppo_whiten_advantages_v1(1u, 2u, 0.0f, adv, mask, out_a);
  case 5:
    cfg.clip_eps_low = 0.0f;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 6:
    mask[0] = 0u;
    mask[1] = 0u;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 7:
    mask[0] = 0u;
    mask[1] = 0u;
    return et_ppo_gae_v1(1u, 2u, 0.99f, 0.95f, ret, values, mask, out_a, out_r);
  case 8:
    actions[1] = 3;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 9:
    actions[0] = -1;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 10: {
    const int saved = fegetround();
    if (fesetround(FE_TOWARDZERO) != 0)
      return ET_PPO_ERR_ARGUMENT;
    const et_ppo_status st = et_ppo_loss_forward_v1(
        &cfg, logits, values, actions, old_lp, old_v, adv, ret, mask, &metrics);
    (void)fesetround(saved);
    return st;
  }
  case 11:
    old_lp[0] = -80.0f;
    return et_ppo_loss_forward_v1(&cfg, logits, values, actions, old_lp, old_v,
                                  adv, ret, mask, &metrics);
  case 12:
    return et_ppo_whiten_advantages_v1(1u, 2u, 1e-6f, adv, mask, adv);
  default:
    return ET_PPO_ERR_ARGUMENT;
  }
}
