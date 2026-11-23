#include "ac4_eshkol_bridge.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

struct et_ac4_aot_context {
  et_ac4_config_v1 config;
  et_ac4_memory_v1 plan;
  et_ac4_result_v1 last_result;
  et_ac4_io_v1 io;
  float *input;
  float *target;
  float *parameters;
  float *output;
  float *d_input;
  float *d_parameters;
  float *workspace;
  int has_result;
};

static uint64_t f64_bits(double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int is_nan_f64(double value) {
  const uint64_t bits = f64_bits(value);
  return (bits & UINT64_C(0x7ff0000000000000)) ==
             UINT64_C(0x7ff0000000000000) &&
         (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static int is_finite_f64(double value) {
  return (f64_bits(value) & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

static int is_finite_f32(float value) {
  return (f32_bits(value) & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int is_positive_finite_f32(float value) {
  const uint32_t bits = f32_bits(value);
  return bits > 0u && bits < UINT32_C(0x7f800000);
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

static int valid_index(int64_t index, size_t count) {
  return index >= 0 && (uint64_t)index < (uint64_t)count;
}

static int has_gradients(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result &&
         ctx->config.mode == ET_AC4_FORWARD_BACKWARD;
}

int32_t et_ac4_aot_create(int64_t layers, int64_t batch, int64_t tokens,
                          int64_t query_heads, int64_t kv_heads,
                          int64_t head_dim, int64_t mode, int64_t policy,
                          int64_t checkpoint_every, double epsilon,
                          double dropout_rate, int64_t seed,
                          int64_t activation_limit_bytes,
                          et_ac4_aot_context **out_ctx) {
  if (out_ctx == NULL)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  if (layers <= 0 || layers > (int64_t)ET_AC4_MAX_LAYERS || mode < 0 ||
      mode > (int64_t)ET_AC4_FORWARD_BACKWARD || policy < 0 ||
      policy > (int64_t)ET_AC4_POLICY_EVERY_K || checkpoint_every < 0 ||
      checkpoint_every > (int64_t)UINT32_MAX || activation_limit_bytes < 0 ||
      (uint64_t)activation_limit_bytes > (uint64_t)SIZE_MAX)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  if (!is_finite_f64(epsilon) || epsilon <= 0.0 ||
      !is_finite_f64(dropout_rate) || dropout_rate < 0.0 || dropout_rate >= 1.0)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  if (batch <= 0 || tokens <= 0 || query_heads <= 0 || kv_heads <= 0 ||
      head_dim <= 0)
    return (int32_t)ET_AC4_SHAPE_MISMATCH;
  if (batch > (int64_t)UINT32_MAX || tokens > (int64_t)UINT32_MAX ||
      query_heads > (int64_t)UINT32_MAX || kv_heads > (int64_t)UINT32_MAX ||
      head_dim > (int64_t)UINT32_MAX)
    return (int32_t)ET_AC4_UNSUPPORTED;

  const et_ac4_config_v1 cfg = {
      .shape = {.layers = (uint32_t)layers,
                .batch = (uint32_t)batch,
                .tokens = (uint32_t)tokens,
                .query_heads = (uint32_t)query_heads,
                .kv_heads = (uint32_t)kv_heads,
                .head_dim = (uint32_t)head_dim},
      .mode = (et_ac4_mode)mode,
      .policy = (et_ac4_policy)policy,
      .checkpoint_every = (uint32_t)checkpoint_every,
      .epsilon = f32_from_f64(epsilon),
      .dropout_rate = f32_from_f64(dropout_rate),
      .seed = (uint64_t)seed,
      .step = 0u,
      .activation_limit_bytes = (size_t)activation_limit_bytes};

  et_ac4_memory_v1 plan;
  const et_ac4_status status = et_ac4_plan_v1(&cfg, &plan);
  if (status != ET_AC4_OK)
    return (int32_t)status;
  if (plan.workspace_bytes == 0 || plan.workspace_bytes % sizeof(float) != 0)
    return (int32_t)ET_AC4_PROVIDER_ERROR;

  et_ac4_aot_context *ctx = (et_ac4_aot_context *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_AC4_AOT_OUT_OF_MEMORY;

  ctx->config = cfg;
  ctx->plan = plan;
  ctx->input = (float *)calloc(plan.activation_floats, sizeof(float));
  ctx->target = (float *)calloc(plan.activation_floats, sizeof(float));
  ctx->parameters = (float *)calloc(plan.parameter_floats, sizeof(float));
  ctx->output = (float *)calloc(plan.activation_floats, sizeof(float));
  ctx->workspace =
      (float *)calloc(plan.workspace_bytes / sizeof(float), sizeof(float));
  if (cfg.mode == ET_AC4_FORWARD_BACKWARD) {
    ctx->d_input = (float *)calloc(plan.activation_floats, sizeof(float));
    ctx->d_parameters = (float *)calloc(plan.parameter_floats, sizeof(float));
  }

  if (ctx->input == NULL || ctx->target == NULL || ctx->parameters == NULL ||
      ctx->output == NULL || ctx->workspace == NULL ||
      (cfg.mode == ET_AC4_FORWARD_BACKWARD &&
       (ctx->d_input == NULL || ctx->d_parameters == NULL))) {
    et_ac4_aot_destroy(ctx);
    return ET_AC4_AOT_OUT_OF_MEMORY;
  }

  ctx->io = (et_ac4_io_v1){
      .input = {ctx->input, plan.activation_floats},
      .target = {ctx->target, plan.activation_floats},
      .parameters = {ctx->parameters, plan.parameter_floats},
      .output = {ctx->output, plan.activation_floats},
      .d_input = {ctx->d_input, cfg.mode == ET_AC4_FORWARD_BACKWARD
                                    ? plan.activation_floats
                                    : 0u},
      .d_parameters = {ctx->d_parameters, cfg.mode == ET_AC4_FORWARD_BACKWARD
                                              ? plan.parameter_floats
                                              : 0u},
      .workspace = ctx->workspace,
      .workspace_bytes = plan.workspace_bytes};

  *out_ctx = ctx;
  return (int32_t)ET_AC4_OK;
}

void et_ac4_aot_destroy(et_ac4_aot_context *ctx) {
  if (ctx == NULL)
    return;
  free(ctx->input);
  free(ctx->target);
  free(ctx->parameters);
  free(ctx->output);
  free(ctx->d_input);
  free(ctx->d_parameters);
  free(ctx->workspace);
  free(ctx);
}

int32_t et_ac4_aot_set_activation_limit(et_ac4_aot_context *ctx,
                                        int64_t activation_limit_bytes) {
  if (ctx == NULL || activation_limit_bytes < 0 ||
      (uint64_t)activation_limit_bytes > (uint64_t)SIZE_MAX)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  ctx->config.activation_limit_bytes = (size_t)activation_limit_bytes;
  return (int32_t)ET_AC4_OK;
}

int64_t et_ac4_aot_activation_floats(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.activation_floats : -1;
}

int64_t et_ac4_aot_parameter_floats(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.parameter_floats : -1;
}

int64_t et_ac4_aot_parameter_floats_per_layer(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.parameter_floats_per_layer : -1;
}

int64_t et_ac4_aot_planned_peak_bytes(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.activation_peak_bytes : -1;
}

int64_t et_ac4_aot_planned_workspace_bytes(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.workspace_bytes : -1;
}

int64_t et_ac4_aot_planned_gradient_staging_bytes(
    const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.gradient_staging_bytes : -1;
}

int64_t et_ac4_aot_planned_retained_checkpoints(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.retained_checkpoints : -1;
}

int64_t et_ac4_aot_planned_retained_tapes(const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.retained_tapes : -1;
}

int64_t et_ac4_aot_planned_rematerialized_layers(
    const et_ac4_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.rematerialized_layers : -1;
}

int32_t et_ac4_aot_set_input(et_ac4_aot_context *ctx, int64_t index,
                             double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.activation_floats))
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  ctx->input[index] = f32_from_f64(value);
  return (int32_t)ET_AC4_OK;
}

int32_t et_ac4_aot_set_target(et_ac4_aot_context *ctx, int64_t index,
                              double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.activation_floats))
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  ctx->target[index] = f32_from_f64(value);
  return (int32_t)ET_AC4_OK;
}

int32_t et_ac4_aot_set_parameter(et_ac4_aot_context *ctx, int64_t index,
                                 double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.parameter_floats))
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  ctx->parameters[index] = f32_from_f64(value);
  return (int32_t)ET_AC4_OK;
}

int32_t et_ac4_aot_run(et_ac4_aot_context *ctx, int64_t step) {
  if (ctx == NULL || step < 0)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  et_ac4_config_v1 cfg = ctx->config;
  cfg.step = (uint64_t)step;
  et_ac4_result_v1 result;
  const et_ac4_status status = et_ac4_run_v1(&cfg, &ctx->io, &result);
  if (status != ET_AC4_OK)
    return (int32_t)status;
  ctx->config.step = cfg.step;
  ctx->last_result = result;
  ctx->has_result = 1;
  return (int32_t)ET_AC4_OK;
}

int32_t et_ac4_aot_sgd_step(et_ac4_aot_context *ctx, double learning_rate) {
  if (!has_gradients(ctx) || !is_finite_f64(learning_rate) ||
      learning_rate <= 0.0)
    return (int32_t)ET_AC4_INVALID_ARGUMENT;
  const float lr = f32_from_f64(learning_rate);
  if (!is_positive_finite_f32(lr))
    return (int32_t)ET_AC4_INVALID_ARGUMENT;

  const size_t count = ctx->plan.parameter_floats;
  for (size_t i = 0; i < count; ++i) {
    const float updated = ctx->parameters[i] - lr * ctx->d_parameters[i];
    if (!is_finite_f32(updated))
      return (int32_t)ET_AC4_NUMERIC_ERROR;
  }
  for (size_t i = 0; i < count; ++i)
    ctx->parameters[i] -= lr * ctx->d_parameters[i];
  return (int32_t)ET_AC4_OK;
}

int64_t et_ac4_aot_peak_activation_bytes(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.activation_peak_bytes
             : -1;
}

int64_t et_ac4_aot_recomputation_count(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.rematerialized_layers
             : -1;
}

int64_t et_ac4_aot_retained_checkpoints(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.retained_checkpoints
             : -1;
}

int64_t et_ac4_aot_retained_tapes(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.retained_tapes
             : -1;
}

int64_t et_ac4_aot_workspace_bytes(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.workspace_bytes
             : -1;
}

int64_t et_ac4_aot_gradient_staging_bytes(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result
             ? (int64_t)ctx->last_result.memory.gradient_staging_bytes
             : -1;
}

double et_ac4_aot_get_loss(const et_ac4_aot_context *ctx) {
  return ctx != NULL && ctx->has_result ? ctx->last_result.loss : (double)NAN;
}

double et_ac4_aot_get_output(const et_ac4_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !ctx->has_result ||
      !valid_index(index, ctx->plan.activation_floats))
    return (double)NAN;
  return (double)ctx->output[index];
}

double et_ac4_aot_get_parameter(const et_ac4_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->plan.parameter_floats))
    return (double)NAN;
  return (double)ctx->parameters[index];
}

double et_ac4_aot_get_d_input(const et_ac4_aot_context *ctx, int64_t index) {
  if (!has_gradients(ctx) || !valid_index(index, ctx->plan.activation_floats))
    return (double)NAN;
  return (double)ctx->d_input[index];
}

double et_ac4_aot_get_d_parameter(const et_ac4_aot_context *ctx,
                                  int64_t index) {
  if (!has_gradients(ctx) || !valid_index(index, ctx->plan.parameter_floats))
    return (double)NAN;
  return (double)ctx->d_parameters[index];
}

int64_t et_ac4_aot_loss_bits_lo(const et_ac4_aot_context *ctx) {
  if (ctx == NULL || !ctx->has_result)
    return -1;
  return (int64_t)(f64_bits(ctx->last_result.loss) & UINT64_C(0xffffffff));
}

int64_t et_ac4_aot_loss_bits_hi(const et_ac4_aot_context *ctx) {
  if (ctx == NULL || !ctx->has_result)
    return -1;
  return (int64_t)((f64_bits(ctx->last_result.loss) >> 32) &
                   UINT64_C(0xffffffff));
}

int64_t et_ac4_aot_output_bits(const et_ac4_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !ctx->has_result ||
      !valid_index(index, ctx->plan.activation_floats))
    return -1;
  return (int64_t)f32_bits(ctx->output[index]);
}

int64_t et_ac4_aot_parameter_bits(const et_ac4_aot_context *ctx,
                                  int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->plan.parameter_floats))
    return -1;
  return (int64_t)f32_bits(ctx->parameters[index]);
}

int64_t et_ac4_aot_d_input_bits(const et_ac4_aot_context *ctx, int64_t index) {
  if (!has_gradients(ctx) || !valid_index(index, ctx->plan.activation_floats))
    return -1;
  return (int64_t)f32_bits(ctx->d_input[index]);
}

int64_t et_ac4_aot_d_parameter_bits(const et_ac4_aot_context *ctx,
                                    int64_t index) {
  if (!has_gradients(ctx) || !valid_index(index, ctx->plan.parameter_floats))
    return -1;
  return (int64_t)f32_bits(ctx->d_parameters[index]);
}
