#include "dpo_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

struct et_dpo_aot_ctx {
  et_dpo_config cfg;
  uint32_t hidden_dim;
  size_t logit_count;
  size_t token_count;
  size_t hidden_count;
  size_t weight_count;
  size_t bias_count;
  size_t ref_param_count;

  float *policy_chosen_logits;
  float *policy_rejected_logits;
  float *ref_chosen_logits;
  float *ref_rejected_logits;
  int64_t *chosen_targets;
  int64_t *rejected_targets;
  uint8_t *chosen_mask;
  uint8_t *rejected_mask;
  float *d_logits_chosen;
  float *d_logits_rejected;

  float *hidden_chosen;
  float *hidden_rejected;
  /* Contiguous frozen reference parameters: [W_ref (V*D), b_ref (V)]. */
  float *ref_params;
  /* Trainable policy head parameters & gradients. */
  float *policy_weight;
  float *policy_bias;
  float *d_policy_weight;
  float *d_policy_bias;

  et_dpo_telemetry telemetry;
  uint8_t ref_sha256[ET_DPO_SHA256_BYTES];
};

static float f32_from_f64(double value) {
  if (isnan(value))
    return NAN;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int valid_index(int64_t index, size_t count) {
  return index >= 0 && (uint64_t)index < (uint64_t)count;
}

static int64_t load_f32_bits(const float *buf, int64_t index, size_t count) {
  if (buf == NULL || !valid_index(index, count))
    return -1;
  uint32_t bits = 0u;
  memcpy(&bits, &buf[(size_t)index], sizeof(bits));
  return (int64_t)(uint64_t)bits;
}

static int64_t digest_word_be(const uint8_t digest[ET_DPO_SHA256_BYTES],
                              int64_t word_index) {
  if (word_index < 0 || word_index >= 4)
    return 0;
  const size_t base = (size_t)word_index * 8u;
  uint64_t word = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    word = (word << 8u) | (uint64_t)digest[base + i];
  return (int64_t)word;
}

static void free_ctx(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  free(ctx->policy_chosen_logits);
  free(ctx->policy_rejected_logits);
  free(ctx->ref_chosen_logits);
  free(ctx->ref_rejected_logits);
  free(ctx->chosen_targets);
  free(ctx->rejected_targets);
  free(ctx->chosen_mask);
  free(ctx->rejected_mask);
  free(ctx->d_logits_chosen);
  free(ctx->d_logits_rejected);
  free(ctx->hidden_chosen);
  free(ctx->hidden_rejected);
  free(ctx->ref_params);
  free(ctx->policy_weight);
  free(ctx->policy_bias);
  free(ctx->d_policy_weight);
  free(ctx->d_policy_bias);
  free(ctx);
}

int32_t et_dpo_aot_create(int64_t batch_size, int64_t seq_len,
                          int64_t vocab_size, int64_t hidden_dim, double beta,
                          double label_smoothing, int64_t reference_free,
                          et_dpo_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  *out_ctx = NULL;
  if (batch_size < (int64_t)ET_DPO_MIN_B ||
      batch_size > (int64_t)ET_DPO_MAX_B || seq_len < (int64_t)ET_DPO_MIN_T ||
      seq_len > (int64_t)ET_DPO_MAX_T ||
      vocab_size < (int64_t)ET_DPO_MIN_V ||
      vocab_size > (int64_t)ET_DPO_MAX_V || hidden_dim < 1 ||
      hidden_dim > (int64_t)ET_DPO_AOT_MAX_HIDDEN)
    return ET_DPO_ERR_SHAPE;
  if (reference_free != 0 && reference_free != 1)
    return ET_DPO_ERR_ARGUMENT;

  et_dpo_config cfg;
  cfg.batch_size = (uint32_t)batch_size;
  cfg.seq_len = (uint32_t)seq_len;
  cfg.vocab_size = (uint32_t)vocab_size;
  cfg.reference_free = (uint32_t)reference_free;
  cfg.beta = beta;
  cfg.label_smoothing = label_smoothing;

  const et_dpo_status cfg_st = et_dpo_validate_config(&cfg);
  if (cfg_st != ET_DPO_OK)
    return cfg_st;

  et_dpo_aot_ctx *ctx = (et_dpo_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;

  ctx->cfg = cfg;
  ctx->hidden_dim = (uint32_t)hidden_dim;
  ctx->token_count = (size_t)cfg.batch_size * cfg.seq_len;
  ctx->logit_count = ctx->token_count * cfg.vocab_size;
  ctx->hidden_count = ctx->token_count * ctx->hidden_dim;
  ctx->weight_count = (size_t)cfg.vocab_size * ctx->hidden_dim;
  ctx->bias_count = (size_t)cfg.vocab_size;
  ctx->ref_param_count = ctx->weight_count + ctx->bias_count;

  ctx->policy_chosen_logits =
      (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->policy_rejected_logits =
      (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->ref_chosen_logits = (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->ref_rejected_logits = (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->chosen_targets = (int64_t *)calloc(ctx->token_count, sizeof(int64_t));
  ctx->rejected_targets = (int64_t *)calloc(ctx->token_count, sizeof(int64_t));
  ctx->chosen_mask = (uint8_t *)calloc(ctx->token_count, sizeof(uint8_t));
  ctx->rejected_mask = (uint8_t *)calloc(ctx->token_count, sizeof(uint8_t));
  ctx->d_logits_chosen = (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->d_logits_rejected = (float *)calloc(ctx->logit_count, sizeof(float));
  ctx->hidden_chosen = (float *)calloc(ctx->hidden_count, sizeof(float));
  ctx->hidden_rejected = (float *)calloc(ctx->hidden_count, sizeof(float));
  ctx->ref_params = (float *)calloc(ctx->ref_param_count, sizeof(float));
  ctx->policy_weight = (float *)calloc(ctx->weight_count, sizeof(float));
  ctx->policy_bias = (float *)calloc(ctx->bias_count, sizeof(float));
  ctx->d_policy_weight = (float *)calloc(ctx->weight_count, sizeof(float));
  ctx->d_policy_bias = (float *)calloc(ctx->bias_count, sizeof(float));

  if (ctx->policy_chosen_logits == NULL ||
      ctx->policy_rejected_logits == NULL || ctx->ref_chosen_logits == NULL ||
      ctx->ref_rejected_logits == NULL || ctx->chosen_targets == NULL ||
      ctx->rejected_targets == NULL || ctx->chosen_mask == NULL ||
      ctx->rejected_mask == NULL || ctx->d_logits_chosen == NULL ||
      ctx->d_logits_rejected == NULL || ctx->hidden_chosen == NULL ||
      ctx->hidden_rejected == NULL || ctx->ref_params == NULL ||
      ctx->policy_weight == NULL || ctx->policy_bias == NULL ||
      ctx->d_policy_weight == NULL || ctx->d_policy_bias == NULL) {
    free_ctx(ctx);
    return ET_DPO_ERR_ARGUMENT;
  }

  *out_ctx = ctx;
  return ET_DPO_OK;
}

void et_dpo_aot_destroy(et_dpo_aot_ctx *ctx) { free_ctx(ctx); }

int32_t et_dpo_aot_set_hyperparams(et_dpo_aot_ctx *ctx, double beta,
                                   double label_smoothing,
                                   int64_t reference_free) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  ctx->cfg.beta = beta;
  ctx->cfg.label_smoothing = label_smoothing;
  ctx->cfg.reference_free =
      (reference_free < 0 || reference_free > (int64_t)UINT32_MAX)
          ? 2u
          : (uint32_t)reference_free;
  return ET_DPO_OK;
}

int64_t et_dpo_aot_logit_count(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->logit_count : 0;
}

int64_t et_dpo_aot_token_count(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->token_count : 0;
}

int64_t et_dpo_aot_hidden_count(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->hidden_count : 0;
}

int64_t et_dpo_aot_weight_count(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->weight_count : 0;
}

int64_t et_dpo_aot_bias_count(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? (int64_t)ctx->bias_count : 0;
}

int32_t et_dpo_aot_set_policy_chosen_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                           double value) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->policy_chosen_logits[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_policy_rejected_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                             double value) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->policy_rejected_logits[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_ref_chosen_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                        double value) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->ref_chosen_logits[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_ref_rejected_logit(et_dpo_aot_ctx *ctx, int64_t index,
                                          double value) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->ref_rejected_logits[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

double et_dpo_aot_get_policy_chosen_logit(const et_dpo_aot_ctx *ctx,
                                          int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return 0.0;
  return (double)ctx->policy_chosen_logits[(size_t)index];
}

double et_dpo_aot_get_policy_rejected_logit(const et_dpo_aot_ctx *ctx,
                                            int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return 0.0;
  return (double)ctx->policy_rejected_logits[(size_t)index];
}

double et_dpo_aot_get_d_logit_chosen(const et_dpo_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return 0.0;
  return (double)ctx->d_logits_chosen[(size_t)index];
}

double et_dpo_aot_get_d_logit_rejected(const et_dpo_aot_ctx *ctx,
                                       int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->logit_count))
    return 0.0;
  return (double)ctx->d_logits_rejected[(size_t)index];
}

int64_t et_dpo_aot_d_logit_chosen_bits(const et_dpo_aot_ctx *ctx,
                                       int64_t index) {
  return ctx != NULL ? load_f32_bits(ctx->d_logits_chosen, index,
                                     ctx->logit_count)
                     : -1;
}

int64_t et_dpo_aot_d_logit_rejected_bits(const et_dpo_aot_ctx *ctx,
                                         int64_t index) {
  return ctx != NULL ? load_f32_bits(ctx->d_logits_rejected, index,
                                     ctx->logit_count)
                     : -1;
}

int32_t et_dpo_aot_poison_grads(et_dpo_aot_ctx *ctx, double sentinel) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  const float val = f32_from_f64(sentinel);
  for (size_t i = 0u; i < ctx->logit_count; ++i) {
    ctx->d_logits_chosen[i] = val;
    ctx->d_logits_rejected[i] = val;
  }
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_chosen_target(et_dpo_aot_ctx *ctx, int64_t index,
                                     int64_t token_id) {
  if (ctx == NULL || !valid_index(index, ctx->token_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->chosen_targets[(size_t)index] = token_id;
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_rejected_target(et_dpo_aot_ctx *ctx, int64_t index,
                                       int64_t token_id) {
  if (ctx == NULL || !valid_index(index, ctx->token_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->rejected_targets[(size_t)index] = token_id;
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_chosen_mask(et_dpo_aot_ctx *ctx, int64_t index,
                                   int64_t mask_val) {
  if (ctx == NULL || !valid_index(index, ctx->token_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->chosen_mask[(size_t)index] =
      (mask_val < 0 || mask_val > 255) ? (uint8_t)255u : (uint8_t)mask_val;
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_rejected_mask(et_dpo_aot_ctx *ctx, int64_t index,
                                     int64_t mask_val) {
  if (ctx == NULL || !valid_index(index, ctx->token_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->rejected_mask[(size_t)index] =
      (mask_val < 0 || mask_val > 255) ? (uint8_t)255u : (uint8_t)mask_val;
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_hidden_chosen(et_dpo_aot_ctx *ctx, int64_t index,
                                     double value) {
  if (ctx == NULL || !valid_index(index, ctx->hidden_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->hidden_chosen[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_hidden_rejected(et_dpo_aot_ctx *ctx, int64_t index,
                                       double value) {
  if (ctx == NULL || !valid_index(index, ctx->hidden_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->hidden_rejected[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_ref_weight(et_dpo_aot_ctx *ctx, int64_t index,
                                  double value) {
  if (ctx == NULL || !valid_index(index, ctx->weight_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->ref_params[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_ref_bias(et_dpo_aot_ctx *ctx, int64_t index,
                                double value) {
  if (ctx == NULL || !valid_index(index, ctx->bias_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->ref_params[ctx->weight_count + (size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_policy_weight(et_dpo_aot_ctx *ctx, int64_t index,
                                     double value) {
  if (ctx == NULL || !valid_index(index, ctx->weight_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->policy_weight[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

int32_t et_dpo_aot_set_policy_bias(et_dpo_aot_ctx *ctx, int64_t index,
                                   double value) {
  if (ctx == NULL || !valid_index(index, ctx->bias_count))
    return ET_DPO_ERR_ARGUMENT;
  ctx->policy_bias[(size_t)index] = f32_from_f64(value);
  return ET_DPO_OK;
}

double et_dpo_aot_get_policy_weight(const et_dpo_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->weight_count))
    return 0.0;
  return (double)ctx->policy_weight[(size_t)index];
}

double et_dpo_aot_get_policy_bias(const et_dpo_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->bias_count))
    return 0.0;
  return (double)ctx->policy_bias[(size_t)index];
}

double et_dpo_aot_get_d_policy_weight(const et_dpo_aot_ctx *ctx,
                                      int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->weight_count))
    return 0.0;
  return (double)ctx->d_policy_weight[(size_t)index];
}

double et_dpo_aot_get_d_policy_bias(const et_dpo_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->bias_count))
    return 0.0;
  return (double)ctx->d_policy_bias[(size_t)index];
}

int64_t et_dpo_aot_ref_weight_bits(const et_dpo_aot_ctx *ctx, int64_t index) {
  return ctx != NULL
             ? load_f32_bits(ctx->ref_params, index, ctx->weight_count)
             : -1;
}

int64_t et_dpo_aot_ref_bias_bits(const et_dpo_aot_ctx *ctx, int64_t index) {
  return ctx != NULL ? load_f32_bits(ctx->ref_params + ctx->weight_count, index,
                                     ctx->bias_count)
                     : -1;
}

static int32_t linear_project(size_t token_count, uint32_t hidden_dim,
                              uint32_t vocab_size, const float *hidden,
                              const float *weight, const float *bias,
                              float *out_logits) {
  for (size_t bt = 0u; bt < token_count; ++bt) {
    const float *h = hidden + bt * hidden_dim;
    float *y = out_logits + bt * vocab_size;
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      const float *w = weight + (size_t)v * hidden_dim;
      double acc = (double)bias[v];
      for (uint32_t d = 0u; d < hidden_dim; ++d)
        acc += (double)h[d] * (double)w[d];
      if (!isfinite(acc) || fabs(acc) > (double)FLT_MAX)
        return ET_DPO_ERR_NONFINITE;
      y[v] = (float)acc;
    }
  }
  return ET_DPO_OK;
}

int32_t et_dpo_aot_init_policy_from_ref(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  const float *w_ref = ctx->ref_params;
  const float *b_ref = ctx->ref_params + ctx->weight_count;
  memcpy(ctx->policy_weight, w_ref, ctx->weight_count * sizeof(float));
  memcpy(ctx->policy_bias, b_ref, ctx->bias_count * sizeof(float));

  int32_t st = linear_project(ctx->token_count, ctx->hidden_dim,
                              ctx->cfg.vocab_size, ctx->hidden_chosen, w_ref,
                              b_ref, ctx->ref_chosen_logits);
  if (st != ET_DPO_OK)
    return st;
  st = linear_project(ctx->token_count, ctx->hidden_dim, ctx->cfg.vocab_size,
                      ctx->hidden_rejected, w_ref, b_ref,
                      ctx->ref_rejected_logits);
  if (st != ET_DPO_OK)
    return st;
  return et_dpo_aot_project_policy_logits(ctx);
}

int32_t et_dpo_aot_project_policy_logits(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  int32_t st = linear_project(ctx->token_count, ctx->hidden_dim,
                              ctx->cfg.vocab_size, ctx->hidden_chosen,
                              ctx->policy_weight, ctx->policy_bias,
                              ctx->policy_chosen_logits);
  if (st != ET_DPO_OK)
    return st;
  return linear_project(ctx->token_count, ctx->hidden_dim, ctx->cfg.vocab_size,
                        ctx->hidden_rejected, ctx->policy_weight,
                        ctx->policy_bias, ctx->policy_rejected_logits);
}

int32_t et_dpo_aot_head_backward(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  const uint32_t v_dim = ctx->cfg.vocab_size;
  const uint32_t d_dim = ctx->hidden_dim;
  for (uint32_t v = 0u; v < v_dim; ++v) {
    double db = 0.0;
    for (size_t bt = 0u; bt < ctx->token_count; ++bt) {
      db += (double)ctx->d_logits_chosen[bt * v_dim + v];
      db += (double)ctx->d_logits_rejected[bt * v_dim + v];
    }
    if (!isfinite(db) || fabs(db) > (double)FLT_MAX)
      return ET_DPO_ERR_NONFINITE;
    ctx->d_policy_bias[v] = (float)db;

    for (uint32_t d = 0u; d < d_dim; ++d) {
      double dw = 0.0;
      for (size_t bt = 0u; bt < ctx->token_count; ++bt) {
        dw += (double)ctx->d_logits_chosen[bt * v_dim + v] *
              (double)ctx->hidden_chosen[bt * d_dim + d];
        dw += (double)ctx->d_logits_rejected[bt * v_dim + v] *
              (double)ctx->hidden_rejected[bt * d_dim + d];
      }
      if (!isfinite(dw) || fabs(dw) > (double)FLT_MAX)
        return ET_DPO_ERR_NONFINITE;
      ctx->d_policy_weight[(size_t)v * d_dim + d] = (float)dw;
    }
  }
  return ET_DPO_OK;
}

int32_t et_dpo_aot_forward(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  return et_dpo_forward(&ctx->cfg, ctx->policy_chosen_logits,
                        ctx->policy_rejected_logits, ctx->ref_chosen_logits,
                        ctx->ref_rejected_logits, ctx->chosen_targets,
                        ctx->rejected_targets, ctx->chosen_mask,
                        ctx->rejected_mask, &ctx->telemetry);
}

int32_t et_dpo_aot_forward_backward(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  return et_dpo_forward_backward(
      &ctx->cfg, ctx->policy_chosen_logits, ctx->policy_rejected_logits,
      ctx->ref_chosen_logits, ctx->ref_rejected_logits, ctx->chosen_targets,
      ctx->rejected_targets, ctx->chosen_mask, ctx->rejected_mask,
      &ctx->telemetry, ctx->d_logits_chosen, ctx->d_logits_rejected);
}

double et_dpo_aot_mean_loss(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? ctx->telemetry.mean_loss : 0.0;
}

double et_dpo_aot_chosen_reward_mean(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? ctx->telemetry.chosen_reward_mean : 0.0;
}

double et_dpo_aot_rejected_reward_mean(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? ctx->telemetry.rejected_reward_mean : 0.0;
}

double et_dpo_aot_reward_margin_mean(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? ctx->telemetry.reward_margin_mean : 0.0;
}

double et_dpo_aot_preference_accuracy(const et_dpo_aot_ctx *ctx) {
  return ctx != NULL ? ctx->telemetry.preference_accuracy : 0.0;
}

int32_t et_dpo_aot_compute_ref_sha256(et_dpo_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_DPO_ERR_ARGUMENT;
  return et_dpo_weight_sha256(ctx->ref_param_count, ctx->ref_params,
                              ctx->ref_sha256);
}

int64_t et_dpo_aot_ref_sha256_word(const et_dpo_aot_ctx *ctx,
                                   int64_t word_index) {
  return ctx != NULL ? digest_word_be(ctx->ref_sha256, word_index) : 0;
}
