#include "spec_decode_eshkol_bridge.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

struct et_spec_aot_context {
  uint32_t horizon;
  uint32_t vocab_size;
  uint32_t *draft_tokens;
  float *draft_logits;
  float *target_logits;
  uint32_t *out_tokens;
  double *probs;
  et_spec_verify_result_v1 last_result;
  int has_result;
  int has_probs;
  uint32_t last_sampled_token;
  et_spec_rng_state_v1 last_sample_rng;
  int has_sample;
};

static uint64_t f64_bits(double value) {
  uint64_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int is_nan_f64(double value) {
  const uint64_t bits = f64_bits(value);
  return (bits & UINT64_C(0x7ff0000000000000)) ==
             UINT64_C(0x7ff0000000000000) &&
         (bits & UINT64_C(0x000fffffffffffff)) != 0u;
}

static float f32_from_f64(double value) {
  if (is_nan_f64(value))
    return (float)NAN;
  if (value > (double)FLT_MAX)
    return (float)INFINITY;
  if (value < -(double)FLT_MAX)
    return -(float)INFINITY;
  return (float)value;
}

static int decode_eos(int64_t eos_token_id, uint32_t *out_eos) {
  if (eos_token_id == ET_SPEC_AOT_NO_EOS) {
    *out_eos = ET_SPEC_NO_EOS;
    return 1;
  }
  if (eos_token_id < 0 || eos_token_id > (int64_t)UINT32_MAX)
    return 0;
  *out_eos = (uint32_t)eos_token_id;
  return 1;
}

static int decode_top_k(int64_t top_k, uint32_t *out_top_k) {
  if (top_k < 0 || top_k > (int64_t)UINT32_MAX)
    return 0;
  *out_top_k = (uint32_t)top_k;
  return 1;
}

int32_t et_spec_aot_create(int64_t horizon, int64_t vocab_size,
                           et_spec_aot_context **out_ctx) {
  if (out_ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if ((((uintptr_t)out_ctx) % _Alignof(et_spec_aot_context *)) != 0u)
    return (int32_t)ET_SPEC_ERR_ALIGNMENT;
  if (horizon < (int64_t)ET_SPEC_MIN_HORIZON ||
      horizon > (int64_t)ET_SPEC_MAX_HORIZON ||
      vocab_size < (int64_t)ET_SPEC_MIN_VOCAB ||
      vocab_size > (int64_t)ET_SPEC_MAX_VOCAB)
    return (int32_t)ET_SPEC_ERR_BOUNDS;

  const uint32_t k = (uint32_t)horizon;
  const uint32_t v = (uint32_t)vocab_size;
  et_spec_aot_context *ctx =
      (et_spec_aot_context *)calloc(1u, sizeof(et_spec_aot_context));
  if (ctx == NULL)
    return (int32_t)ET_SPEC_AOT_ERR_NOMEM;

  ctx->horizon = k;
  ctx->vocab_size = v;
  ctx->draft_tokens = (uint32_t *)calloc((size_t)k, sizeof(uint32_t));
  ctx->draft_logits =
      (float *)calloc((size_t)k * (size_t)v, sizeof(float));
  ctx->target_logits =
      (float *)calloc(((size_t)k + 1u) * (size_t)v, sizeof(float));
  ctx->out_tokens = (uint32_t *)calloc((size_t)k + 1u, sizeof(uint32_t));
  ctx->probs = (double *)calloc((size_t)v, sizeof(double));

  if (ctx->draft_tokens == NULL || ctx->draft_logits == NULL ||
      ctx->target_logits == NULL || ctx->out_tokens == NULL ||
      ctx->probs == NULL) {
    et_spec_aot_destroy(ctx);
    return (int32_t)ET_SPEC_AOT_ERR_NOMEM;
  }

  *out_ctx = ctx;
  return (int32_t)ET_SPEC_OK;
}

void et_spec_aot_destroy(et_spec_aot_context *ctx) {
  if (ctx == NULL)
    return;
  free(ctx->draft_tokens);
  free(ctx->draft_logits);
  free(ctx->target_logits);
  free(ctx->out_tokens);
  free(ctx->probs);
  free(ctx);
}

int32_t et_spec_aot_set_draft_token(et_spec_aot_context *ctx, int64_t pos,
                                    int64_t token_id) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if (pos < 0 || (uint64_t)pos >= (uint64_t)ctx->horizon)
    return (int32_t)ET_SPEC_ERR_BOUNDS;
  if (token_id < 0 || token_id > (int64_t)UINT32_MAX)
    return (int32_t)ET_SPEC_ERR_TOKEN_RANGE;
  ctx->draft_tokens[(size_t)pos] = (uint32_t)token_id;
  return (int32_t)ET_SPEC_OK;
}

int32_t et_spec_aot_set_draft_logit(et_spec_aot_context *ctx, int64_t row,
                                    int64_t vocab_idx, double value) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if (row < 0 || (uint64_t)row >= (uint64_t)ctx->horizon || vocab_idx < 0 ||
      (uint64_t)vocab_idx >= (uint64_t)ctx->vocab_size)
    return (int32_t)ET_SPEC_ERR_BOUNDS;
  ctx->draft_logits[(size_t)row * (size_t)ctx->vocab_size + (size_t)vocab_idx] =
      f32_from_f64(value);
  return (int32_t)ET_SPEC_OK;
}

int32_t et_spec_aot_set_target_logit(et_spec_aot_context *ctx, int64_t row,
                                     int64_t vocab_idx, double value) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if (row < 0 || (uint64_t)row >= (uint64_t)ctx->horizon + 1u ||
      vocab_idx < 0 || (uint64_t)vocab_idx >= (uint64_t)ctx->vocab_size)
    return (int32_t)ET_SPEC_ERR_BOUNDS;
  ctx->target_logits[(size_t)row * (size_t)ctx->vocab_size +
                     (size_t)vocab_idx] = f32_from_f64(value);
  return (int32_t)ET_SPEC_OK;
}

int32_t et_spec_aot_verify_greedy(et_spec_aot_context *ctx,
                                  int64_t eos_token_id) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  uint32_t eos = ET_SPEC_NO_EOS;
  if (!decode_eos(eos_token_id, &eos))
    return (int32_t)ET_SPEC_ERR_TOKEN_RANGE;

  const et_spec_status st = et_spec_verify_greedy_v1(
      ctx->draft_tokens, ctx->target_logits, ctx->horizon, ctx->vocab_size, eos,
      ctx->out_tokens, ctx->horizon + 1u, &ctx->last_result);
  if (st == ET_SPEC_OK)
    ctx->has_result = 1;
  return (int32_t)st;
}

int32_t et_spec_aot_verify_sample(et_spec_aot_context *ctx, double temperature,
                                  int64_t top_k, double top_p,
                                  int64_t eos_token_id, int64_t seed,
                                  int64_t counter) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  uint32_t eos = ET_SPEC_NO_EOS;
  if (!decode_eos(eos_token_id, &eos))
    return (int32_t)ET_SPEC_ERR_TOKEN_RANGE;
  uint32_t k_param = 0u;
  if (!decode_top_k(top_k, &k_param))
    return (int32_t)ET_SPEC_ERR_PARAM;
  if (counter < 0)
    return (int32_t)ET_SPEC_ERR_BOUNDS;

  const et_spec_sampling_params_v1 params = {
      .temperature = temperature,
      .top_p = top_p,
      .top_k = k_param,
      .eos_token_id = eos,
  };
  const et_spec_rng_state_v1 rng = {
      .seed = (uint64_t)seed,
      .counter = (uint64_t)counter,
  };

  const et_spec_status st = et_spec_verify_sample_v1(
      ctx->draft_tokens, ctx->draft_logits, ctx->target_logits, ctx->horizon,
      ctx->vocab_size, &params, &rng, ctx->out_tokens, ctx->horizon + 1u,
      &ctx->last_result);
  if (st == ET_SPEC_OK)
    ctx->has_result = 1;
  return (int32_t)st;
}

int32_t et_spec_aot_logits_to_probs(et_spec_aot_context *ctx,
                                    int64_t source_is_target, int64_t row,
                                    double temperature, int64_t top_k,
                                    double top_p) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if (source_is_target != 0 && source_is_target != 1)
    return (int32_t)ET_SPEC_ERR_PARAM;
  const uint64_t max_rows =
      source_is_target ? ((uint64_t)ctx->horizon + 1u) : (uint64_t)ctx->horizon;
  if (row < 0 || (uint64_t)row >= max_rows)
    return (int32_t)ET_SPEC_ERR_BOUNDS;
  uint32_t k_param = 0u;
  if (!decode_top_k(top_k, &k_param))
    return (int32_t)ET_SPEC_ERR_PARAM;

  const float *base =
      source_is_target ? ctx->target_logits : ctx->draft_logits;
  const float *logits = base + ((size_t)row * (size_t)ctx->vocab_size);
  const et_spec_sampling_params_v1 params = {
      .temperature = temperature,
      .top_p = top_p,
      .top_k = k_param,
      .eos_token_id = ET_SPEC_NO_EOS,
  };

  const et_spec_status st =
      et_spec_logits_to_probs_v1(logits, ctx->vocab_size, &params, ctx->probs);
  if (st == ET_SPEC_OK)
    ctx->has_probs = 1;
  return (int32_t)st;
}

int32_t et_spec_aot_sample_row(et_spec_aot_context *ctx,
                               int64_t source_is_target, int64_t row,
                               double temperature, int64_t top_k, double top_p,
                               int64_t seed, int64_t counter) {
  if (ctx == NULL)
    return (int32_t)ET_SPEC_ERR_NULL;
  if (source_is_target != 0 && source_is_target != 1)
    return (int32_t)ET_SPEC_ERR_PARAM;
  const uint64_t max_rows =
      source_is_target ? ((uint64_t)ctx->horizon + 1u) : (uint64_t)ctx->horizon;
  if (row < 0 || (uint64_t)row >= max_rows)
    return (int32_t)ET_SPEC_ERR_BOUNDS;
  uint32_t k_param = 0u;
  if (!decode_top_k(top_k, &k_param))
    return (int32_t)ET_SPEC_ERR_PARAM;
  if (counter < 0)
    return (int32_t)ET_SPEC_ERR_BOUNDS;

  const float *base =
      source_is_target ? ctx->target_logits : ctx->draft_logits;
  const float *logits = base + ((size_t)row * (size_t)ctx->vocab_size);
  const et_spec_sampling_params_v1 params = {
      .temperature = temperature,
      .top_p = top_p,
      .top_k = k_param,
      .eos_token_id = ET_SPEC_NO_EOS,
  };
  const et_spec_rng_state_v1 rng = {
      .seed = (uint64_t)seed,
      .counter = (uint64_t)counter,
  };

  uint32_t token = 0u;
  et_spec_rng_state_v1 next_rng = {0u, 0u};
  const et_spec_status st = et_spec_sample_token_v1(
      logits, ctx->vocab_size, &params, &rng, &token, &next_rng);
  if (st == ET_SPEC_OK) {
    ctx->last_sampled_token = token;
    ctx->last_sample_rng = next_rng;
    ctx->has_sample = 1;
  }
  return (int32_t)st;
}

int64_t et_spec_aot_horizon(const et_spec_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->horizon : -1;
}

int64_t et_spec_aot_vocab_size(const et_spec_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->vocab_size : -1;
}

int64_t et_spec_aot_accepted_count(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_result)
             ? (int64_t)ctx->last_result.accepted_count
             : -1;
}

int64_t et_spec_aot_emitted_count(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_result)
             ? (int64_t)ctx->last_result.emitted_count
             : -1;
}

int64_t et_spec_aot_eos_hit(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_result) ? (int64_t)ctx->last_result.eos_hit
                                          : -1;
}

int64_t et_spec_aot_rejected(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_result) ? (int64_t)ctx->last_result.rejected
                                          : -1;
}

int64_t et_spec_aot_rng_draws(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_result) ? (int64_t)ctx->last_result.rng_draws
                                          : -1;
}

int64_t et_spec_aot_next_counter(const et_spec_aot_context *ctx) {
  if (ctx == NULL || !ctx->has_result ||
      ctx->last_result.next_rng.counter > (uint64_t)INT64_MAX)
    return -1;
  return (int64_t)ctx->last_result.next_rng.counter;
}

int64_t et_spec_aot_emitted_token(const et_spec_aot_context *ctx,
                                  int64_t index) {
  if (ctx == NULL || !ctx->has_result || index < 0 ||
      (uint64_t)index >= (uint64_t)ctx->last_result.emitted_count)
    return -1;
  return (int64_t)ctx->out_tokens[(size_t)index];
}

double et_spec_aot_prob(const et_spec_aot_context *ctx, int64_t vocab_idx) {
  if (ctx == NULL || !ctx->has_probs || vocab_idx < 0 ||
      (uint64_t)vocab_idx >= (uint64_t)ctx->vocab_size)
    return (double)NAN;
  return ctx->probs[(size_t)vocab_idx];
}

int64_t et_spec_aot_sampled_token(const et_spec_aot_context *ctx) {
  return (ctx != NULL && ctx->has_sample) ? (int64_t)ctx->last_sampled_token
                                          : -1;
}

int64_t et_spec_aot_sampled_next_counter(const et_spec_aot_context *ctx) {
  if (ctx == NULL || !ctx->has_sample ||
      ctx->last_sample_rng.counter > (uint64_t)INT64_MAX)
    return -1;
  return (int64_t)ctx->last_sample_rng.counter;
}
