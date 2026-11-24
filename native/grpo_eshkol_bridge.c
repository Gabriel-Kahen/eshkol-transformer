#include "grpo_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct et_grpo_aot_ctx {
  uint32_t num_groups;
  uint32_t group_size;
  uint32_t batch_size;
  uint32_t seq_len;
  uint32_t vocab_size;
  uint32_t feat_dim;
  et_grpo_loss_config loss_cfg;
  et_grpo_loss_metrics metrics;

  float features[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN *
                 ET_GRPO_AOT_MAX_DIM];
  float weights[ET_GRPO_AOT_MAX_VOCAB * ET_GRPO_AOT_MAX_DIM];
  float grad_weights[ET_GRPO_AOT_MAX_VOCAB * ET_GRPO_AOT_MAX_DIM];
  float bias[ET_GRPO_AOT_MAX_VOCAB];
  float grad_bias[ET_GRPO_AOT_MAX_VOCAB];

  float logits[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN *
               ET_GRPO_AOT_MAX_VOCAB];
  float grad_logits[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN *
                    ET_GRPO_AOT_MAX_VOCAB];
  int32_t targets[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];
  uint8_t mask[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];

  float log_pi[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];
  float seq_log_pi[ET_GRPO_AOT_MAX_BATCH];
  float log_old[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];
  float log_ref[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];
  float grad_log_pi[ET_GRPO_AOT_MAX_BATCH * ET_GRPO_AOT_MAX_SEQ_LEN];

  float rewards[ET_GRPO_AOT_MAX_BATCH];
  float advantages[ET_GRPO_AOT_MAX_BATCH];
};

static float f32_from_f64(double value) {
  if (!isfinite(value))
    return (float)value;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

int32_t et_grpo_aot_create(int64_t num_groups, int64_t group_size,
                           int64_t seq_len, int64_t vocab_size,
                           int64_t feat_dim, et_grpo_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if (num_groups < 1 || num_groups > (int64_t)ET_GRPO_AOT_MAX_GROUPS ||
      group_size < 2 || group_size > (int64_t)ET_GRPO_AOT_MAX_GROUP_SIZE ||
      seq_len < 1 || seq_len > (int64_t)ET_GRPO_AOT_MAX_SEQ_LEN ||
      vocab_size < 2 || vocab_size > (int64_t)ET_GRPO_AOT_MAX_VOCAB ||
      feat_dim < 1 || feat_dim > (int64_t)ET_GRPO_AOT_MAX_DIM)
    return ET_GRPO_ERR_SHAPE;

  const int64_t batch_size = num_groups * group_size;
  if (batch_size > (int64_t)ET_GRPO_AOT_MAX_BATCH)
    return ET_GRPO_ERR_SHAPE;

  et_grpo_loss_config cfg;
  cfg.batch_size = (uint32_t)batch_size;
  cfg.seq_len = (uint32_t)seq_len;
  cfg.vocab_size = (uint32_t)vocab_size;
  cfg.mask_norm_mode = ET_GRPO_NORM_PER_SEQ_MEAN;
  cfg.clip_eps_low = 0.2f;
  cfg.clip_eps_high = 0.2f;
  cfg.kl_beta = 0.04f;
  cfg.reserved = 0u;
  const et_grpo_status st = et_grpo_validate_loss_config(&cfg);
  if (st != ET_GRPO_OK)
    return st;

  et_grpo_aot_ctx *ctx = (et_grpo_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_GRPO_ERR_NOMEM;
  ctx->num_groups = (uint32_t)num_groups;
  ctx->group_size = (uint32_t)group_size;
  ctx->batch_size = (uint32_t)batch_size;
  ctx->seq_len = (uint32_t)seq_len;
  ctx->vocab_size = (uint32_t)vocab_size;
  ctx->feat_dim = (uint32_t)feat_dim;
  ctx->loss_cfg = cfg;
  *out_ctx = ctx;
  return ET_GRPO_OK;
}

void et_grpo_aot_destroy(et_grpo_aot_ctx *ctx) { free(ctx); }

int32_t et_grpo_aot_set_loss_params(et_grpo_aot_ctx *ctx,
                                    int64_t mask_norm_mode,
                                    double clip_eps_low, double clip_eps_high,
                                    double kl_beta) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if (mask_norm_mode < 0 || mask_norm_mode > 1)
    return ET_GRPO_ERR_ARGUMENT;
  et_grpo_loss_config cfg = ctx->loss_cfg;
  cfg.mask_norm_mode = (uint32_t)mask_norm_mode;
  cfg.clip_eps_low = f32_from_f64(clip_eps_low);
  cfg.clip_eps_high = f32_from_f64(clip_eps_high);
  cfg.kl_beta = f32_from_f64(kl_beta);
  cfg.reserved = 0u;
  const et_grpo_status st = et_grpo_validate_loss_config(&cfg);
  if (st != ET_GRPO_OK)
    return st;
  ctx->loss_cfg = cfg;
  return ET_GRPO_OK;
}

int64_t et_grpo_aot_batch_size(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->batch_size;
}

int64_t et_grpo_aot_vocab_size(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->vocab_size;
}

int64_t et_grpo_aot_feat_dim(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->feat_dim;
}

int32_t et_grpo_aot_set_feature(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                                int64_t d, double value) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size || t < 0 ||
      t >= (int64_t)ctx->seq_len || d < 0 || d >= (int64_t)ctx->feat_dim)
    return ET_GRPO_ERR_ARGUMENT;
  const size_t idx =
      (((size_t)n * (size_t)ctx->seq_len) + (size_t)t) * (size_t)ctx->feat_dim +
      (size_t)d;
  ctx->features[idx] = f32_from_f64(value);
  return ET_GRPO_OK;
}

int32_t et_grpo_aot_set_weight(et_grpo_aot_ctx *ctx, int64_t v, int64_t d,
                               double value) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size || d < 0 ||
      d >= (int64_t)ctx->feat_dim)
    return ET_GRPO_ERR_ARGUMENT;
  ctx->weights[(size_t)v * (size_t)ctx->feat_dim + (size_t)d] =
      f32_from_f64(value);
  return ET_GRPO_OK;
}

double et_grpo_aot_get_weight(const et_grpo_aot_ctx *ctx, int64_t v,
                              int64_t d) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size || d < 0 ||
      d >= (int64_t)ctx->feat_dim)
    return NAN;
  return (double)ctx->weights[(size_t)v * (size_t)ctx->feat_dim + (size_t)d];
}

double et_grpo_aot_get_grad_weight(const et_grpo_aot_ctx *ctx, int64_t v,
                                   int64_t d) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size || d < 0 ||
      d >= (int64_t)ctx->feat_dim)
    return NAN;
  return (double)ctx
      ->grad_weights[(size_t)v * (size_t)ctx->feat_dim + (size_t)d];
}

int32_t et_grpo_aot_set_bias(et_grpo_aot_ctx *ctx, int64_t v, double value) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size)
    return ET_GRPO_ERR_ARGUMENT;
  ctx->bias[v] = f32_from_f64(value);
  return ET_GRPO_OK;
}

double et_grpo_aot_get_bias(const et_grpo_aot_ctx *ctx, int64_t v) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size)
    return NAN;
  return (double)ctx->bias[v];
}

double et_grpo_aot_get_grad_bias(const et_grpo_aot_ctx *ctx, int64_t v) {
  if (ctx == NULL || v < 0 || v >= (int64_t)ctx->vocab_size)
    return NAN;
  return (double)ctx->grad_bias[v];
}

int32_t et_grpo_aot_set_target(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                               int64_t token_id) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size || t < 0 ||
      t >= (int64_t)ctx->seq_len || token_id < 0 ||
      token_id >= (int64_t)ctx->vocab_size)
    return ET_GRPO_ERR_ARGUMENT;
  ctx->targets[(size_t)n * (size_t)ctx->seq_len + (size_t)t] =
      (int32_t)token_id;
  return ET_GRPO_OK;
}

int32_t et_grpo_aot_set_mask(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                             int64_t mask_val) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size || t < 0 ||
      t >= (int64_t)ctx->seq_len || (mask_val != 0 && mask_val != 1))
    return ET_GRPO_ERR_ARGUMENT;
  ctx->mask[(size_t)n * (size_t)ctx->seq_len + (size_t)t] = (uint8_t)mask_val;
  return ET_GRPO_OK;
}

int32_t et_grpo_aot_set_reward(et_grpo_aot_ctx *ctx, int64_t n, double value) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size)
    return ET_GRPO_ERR_ARGUMENT;
  ctx->rewards[n] = f32_from_f64(value);
  return ET_GRPO_OK;
}

double et_grpo_aot_get_reward(const et_grpo_aot_ctx *ctx, int64_t n) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size)
    return NAN;
  return (double)ctx->rewards[n];
}

double et_grpo_aot_get_advantage(const et_grpo_aot_ctx *ctx, int64_t n) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size)
    return NAN;
  return (double)ctx->advantages[n];
}

double et_grpo_aot_get_seq_logprob(const et_grpo_aot_ctx *ctx, int64_t n) {
  if (ctx == NULL || n < 0 || n >= (int64_t)ctx->batch_size)
    return NAN;
  return (double)ctx->seq_log_pi[n];
}

/* Deterministic RLVR completion table for the Eshkol AOT driver:
 * Group 0 expected answer: "42"
 *   0: valid format + exact answer ("42") -> reward 1.00
 *   1: valid format + wrong answer ("41") -> reward 0.25
 *   2: missing <think> tag ("42")         -> reward 0.00
 *   3: empty <think> tag + right answer   -> reward 0.00
 * Group 1 expected answer: "17"
 *   4: valid format + exact answer (" 17 ") -> reward 1.00 (trimmed)
 *   5: valid format + wrong answer ("19")   -> reward 0.25
 *   6: trailing garbage after </answer>     -> reward 0.00
 *   7: unclosed <answer> tag                -> reward 0.00
 */
int32_t et_grpo_aot_score_rlvr_completion(et_grpo_aot_ctx *ctx,
                                          int64_t sample_idx,
                                          int64_t completion_case_id) {
  static const char *const kCompletions[8] = {
      "<think>6 * 7 = 42</think><answer>42</answer>",
      "<think>6 * 7 = 41</think><answer>41</answer>",
      "42",
      "<think>   </think><answer>42</answer>",
      " <think>8 + 9 = 17</think>\n<answer> 17 </answer> ",
      "<think>8 + 9 = 19</think><answer>19</answer>",
      "<think>8 + 9 = 17</think><answer>17</answer>extra",
      "<think>8 + 9 = 17</think><answer>17"};
  static const char *const kExpected[8] = {"42", "42", "42", "42",
                                           "17", "17", "17", "17"};
  if (ctx == NULL || sample_idx < 0 || sample_idx >= (int64_t)ctx->batch_size ||
      completion_case_id < 0 || completion_case_id >= 8)
    return ET_GRPO_ERR_ARGUMENT;

  const char *comp = kCompletions[completion_case_id];
  const char *exp_ans = kExpected[completion_case_id];
  const et_grpo_verify_format_config vcfg = {0.25f, 0.75f,
                                             ET_GRPO_VERIFY_TRIM_WS, 0u};
  float reward = 0.0f;
  et_grpo_verify_detail detail;
  const et_grpo_status st = et_grpo_verify_format_and_answer_v1(
      comp, strlen(comp), exp_ans, strlen(exp_ans), &vcfg, &reward, &detail);
  if (st != ET_GRPO_OK)
    return st;
  ctx->rewards[sample_idx] = reward;
  return ET_GRPO_OK;
}

double et_grpo_aot_verify_exact_case(int64_t case_id) {
  float reward = -1.0f;
  et_grpo_status st = ET_GRPO_ERR_ARGUMENT;
  if (case_id == 0) {
    st = et_grpo_verify_exact_match_v1("42", 2u, "42", 2u,
                                       ET_GRPO_VERIFY_EXACT_BYTES, &reward);
  } else if (case_id == 1) {
    st = et_grpo_verify_exact_match_v1(" 42\n", 4u, "42", 2u,
                                       ET_GRPO_VERIFY_TRIM_WS, &reward);
  } else if (case_id == 2) {
    st = et_grpo_verify_exact_match_v1(" 42", 3u, "42", 2u,
                                       ET_GRPO_VERIFY_EXACT_BYTES, &reward);
  } else if (case_id == 3) {
    st = et_grpo_verify_exact_match_v1("43", 2u, "42", 2u,
                                       ET_GRPO_VERIFY_TRIM_WS, &reward);
  } else {
    return -2.0;
  }
  return st == ET_GRPO_OK ? (double)reward : -1.0;
}

int32_t et_grpo_aot_policy_forward(et_grpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  const size_t nt = (size_t)ctx->batch_size * (size_t)ctx->seq_len;
  const size_t v_dim = (size_t)ctx->vocab_size;
  const size_t d_dim = (size_t)ctx->feat_dim;

  for (size_t i = 0; i < nt; ++i) {
    const float *feat = ctx->features + i * d_dim;
    float *lrow = ctx->logits + i * v_dim;
    for (size_t v = 0; v < v_dim; ++v) {
      double acc = (double)ctx->bias[v];
      const float *wrow = ctx->weights + v * d_dim;
      for (size_t d = 0; d < d_dim; ++d)
        acc += (double)feat[d] * (double)wrow[d];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_GRPO_ERR_NONFINITE;
      lrow[v] = val;
    }
  }
  return et_grpo_token_logprobs_v1(ctx->batch_size, ctx->seq_len,
                                   ctx->vocab_size, ctx->logits, ctx->targets,
                                   ctx->mask, ctx->log_pi, ctx->seq_log_pi);
}

int32_t et_grpo_aot_snapshot_ref(et_grpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  const size_t nt = (size_t)ctx->batch_size * (size_t)ctx->seq_len;
  memcpy(ctx->log_ref, ctx->log_pi, nt * sizeof(float));
  return ET_GRPO_OK;
}

int32_t et_grpo_aot_snapshot_old(et_grpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  const size_t nt = (size_t)ctx->batch_size * (size_t)ctx->seq_len;
  memcpy(ctx->log_old, ctx->log_pi, nt * sizeof(float));
  return ET_GRPO_OK;
}

int32_t et_grpo_aot_compute_advantages(et_grpo_aot_ctx *ctx, int64_t adv_mode,
                                       double eps) {
  if (ctx == NULL || adv_mode < 0 || adv_mode > 1)
    return ET_GRPO_ERR_ARGUMENT;
  return et_grpo_group_advantages_v1(ctx->num_groups, ctx->group_size,
                                     ctx->rewards, (uint32_t)adv_mode,
                                     f32_from_f64(eps), ctx->advantages);
}

int32_t et_grpo_aot_loss_forward(et_grpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  return et_grpo_loss_forward_v1(&ctx->loss_cfg, ctx->logits, ctx->targets,
                                 ctx->log_old, ctx->log_ref, ctx->mask,
                                 ctx->advantages, ctx->rewards, &ctx->metrics);
}

int32_t et_grpo_aot_loss_backward(et_grpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  const et_grpo_status st = et_grpo_loss_backward_v1(
      &ctx->loss_cfg, ctx->logits, ctx->targets, ctx->log_old, ctx->log_ref,
      ctx->mask, ctx->advantages, ctx->rewards, ctx->grad_logits,
      ctx->grad_log_pi, &ctx->metrics);
  if (st != ET_GRPO_OK)
    return st;

  const size_t nt = (size_t)ctx->batch_size * (size_t)ctx->seq_len;
  const size_t v_dim = (size_t)ctx->vocab_size;
  const size_t d_dim = (size_t)ctx->feat_dim;

  for (size_t v = 0; v < v_dim; ++v) {
    double gb = 0.0;
    for (size_t d = 0; d < d_dim; ++d) {
      double gw = 0.0;
      for (size_t i = 0; i < nt; ++i) {
        gw += (double)ctx->grad_logits[i * v_dim + v] *
              (double)ctx->features[i * d_dim + d];
      }
      ctx->grad_weights[v * d_dim + d] = (float)gw;
    }
    for (size_t i = 0; i < nt; ++i)
      gb += (double)ctx->grad_logits[i * v_dim + v];
    ctx->grad_bias[v] = (float)gb;
  }
  return ET_GRPO_OK;
}

double et_grpo_aot_expected_reward(const et_grpo_aot_ctx *ctx) {
  if (ctx == NULL || ctx->num_groups == 0u || ctx->group_size == 0u)
    return NAN;
  double total_exp_r = 0.0;
  for (uint32_t b = 0; b < ctx->num_groups; ++b) {
    const size_t base = (size_t)b * (size_t)ctx->group_size;
    double max_lp = (double)ctx->seq_log_pi[base];
    for (uint32_t g = 1u; g < ctx->group_size; ++g) {
      const double lp = (double)ctx->seq_log_pi[base + (size_t)g];
      if (lp > max_lp)
        max_lp = lp;
    }
    double denom = 0.0;
    double numer = 0.0;
    for (uint32_t g = 0u; g < ctx->group_size; ++g) {
      const double w = exp((double)ctx->seq_log_pi[base + (size_t)g] - max_lp);
      denom += w;
      numer += w * (double)ctx->rewards[base + (size_t)g];
    }
    if (!(denom > 0.0) || !isfinite(denom) || !isfinite(numer))
      return NAN;
    total_exp_r += numer / denom;
  }
  return total_exp_r / (double)ctx->num_groups;
}

double et_grpo_aot_greedy_reward(const et_grpo_aot_ctx *ctx) {
  if (ctx == NULL || ctx->num_groups == 0u || ctx->group_size == 0u)
    return NAN;
  double total_r = 0.0;
  for (uint32_t b = 0; b < ctx->num_groups; ++b) {
    const size_t base = (size_t)b * (size_t)ctx->group_size;
    uint32_t best_g = 0u;
    float best_lp = ctx->seq_log_pi[base];
    for (uint32_t g = 1u; g < ctx->group_size; ++g) {
      if (ctx->seq_log_pi[base + (size_t)g] > best_lp) {
        best_lp = ctx->seq_log_pi[base + (size_t)g];
        best_g = g;
      }
    }
    total_r += (double)ctx->rewards[base + (size_t)best_g];
  }
  return total_r / (double)ctx->num_groups;
}

double et_grpo_aot_metric_loss(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? NAN : (double)ctx->metrics.loss;
}

double et_grpo_aot_metric_pg_loss(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? NAN : (double)ctx->metrics.pg_loss;
}

double et_grpo_aot_metric_mean_kl(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? NAN : (double)ctx->metrics.mean_kl;
}

double et_grpo_aot_metric_clip_fraction(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? NAN : (double)ctx->metrics.clip_fraction;
}

double et_grpo_aot_metric_mean_reward(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? NAN : (double)ctx->metrics.mean_reward;
}

int64_t et_grpo_aot_metric_active_tokens(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->metrics.active_tokens;
}

int64_t et_grpo_aot_metric_clipped_tokens(const et_grpo_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->metrics.clipped_tokens;
}

/* Negative fail-closed probes for the AOT driver:
 *   1: group_size == 1                       -> ET_GRPO_ERR_SHAPE (-2)
 *   2: NaN reward in group_advantages        -> ET_GRPO_ERR_NONFINITE (-5)
 *   3: mask value == 2                       -> ET_GRPO_ERR_ARGUMENT (-1)
 *   4: out-of-range target token ID          -> ET_GRPO_ERR_INDEX (-8)
 *   5: all-zero response mask                -> ET_GRPO_ERR_EMPTY_MASK (-9)
 *   6: clip_eps_low <= 0.0f                  -> ET_GRPO_ERR_ARGUMENT (-1)
 *   7: kl_beta < 0.0f                        -> ET_GRPO_ERR_ARGUMENT (-1)
 *   8: log-ratio overflow (delta_old > 50)   -> ET_GRPO_ERR_OVERFLOW (-7)
 *   9: aliased output buffer                 -> ET_GRPO_ERR_ALIAS (-4)
 */
int32_t et_grpo_aot_check_negative_case(const et_grpo_aot_ctx *ctx,
                                        int64_t case_id) {
  if (ctx == NULL)
    return ET_GRPO_ERR_ARGUMENT;

  float tmp_adv[4] = {0.0f};
  float tmp_r[4] = {1.0f, 0.0f, 0.5f, 0.25f};
  float tmp_logits[4] = {0.1f, -0.2f, 0.3f, 0.0f};
  int32_t tmp_targets[2] = {0, 1};
  uint8_t tmp_mask[2] = {0u, 1u};
  float tmp_lp[2] = {-0.5f, -0.7f};
  float tmp_grad[4] = {0.0f};
  et_grpo_loss_metrics tmp_metrics;
  et_grpo_loss_config cfg = {1u, 2u, 2u, ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
                             0.2f, 0.2f, 0.05f, 0u};

  switch (case_id) {
    case 1:
      return et_grpo_group_advantages_v1(2u, 1u, tmp_r, ET_GRPO_ADV_STANDARD,
                                         1e-6f, tmp_adv);
    case 2:
      tmp_r[1] = NAN;
      return et_grpo_group_advantages_v1(1u, 4u, tmp_r, ET_GRPO_ADV_STANDARD,
                                         1e-6f, tmp_adv);
    case 3:
      tmp_mask[1] = 2u;
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    case 4:
      tmp_targets[1] = 2;
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    case 5:
      tmp_mask[0] = 0u;
      tmp_mask[1] = 0u;
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    case 6:
      cfg.clip_eps_low = 0.0f;
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    case 7:
      cfg.kl_beta = -0.01f;
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    case 8: {
      float extreme_old[2] = {-0.5f, -60.0f};
      return et_grpo_loss_forward_v1(&cfg, tmp_logits, tmp_targets, extreme_old,
                                     tmp_lp, tmp_mask, tmp_r, NULL,
                                     &tmp_metrics);
    }
    case 9:
      return et_grpo_loss_backward_v1(&cfg, tmp_logits, tmp_targets, tmp_lp,
                                      tmp_lp, tmp_mask, tmp_r, NULL, tmp_logits,
                                      NULL, &tmp_metrics);
    default:
      (void)tmp_grad;
      return ET_GRPO_ERR_ARGUMENT;
  }
}
