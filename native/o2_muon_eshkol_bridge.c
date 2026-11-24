#define _POSIX_C_SOURCE 200809L

#include "o2_muon_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ET_MUON_AOT_MAX_RAW_COUNT 65536

struct et_muon_aot_ctx {
  et_muon_step_config step_cfg;
  et_wsd_config wsd_cfg;
  size_t num_groups;
  et_muon_param_group groups[ET_MUON_MAX_GROUPS];
  size_t counts[ET_MUON_MAX_GROUPS];
  double last_lr;
  int64_t last_phase;
  double last_ortho_error;
};

static int narrow_f32_finite(double value, float *out) {
  if (!isfinite(value))
    return 0;
  if (value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  const float narrowed = (float)value;
  if (!isfinite(narrowed))
    return 0;
  *out = narrowed;
  return 1;
}

static float raw_f32_operand(double value) {
  if (isnan(value))
    return NAN;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int valid_group_index(const et_muon_aot_ctx *ctx, int64_t group_id,
                             size_t *out_gid) {
  if (ctx == NULL || group_id < 0 || (uint64_t)group_id >= ctx->num_groups)
    return 0;
  *out_gid = (size_t)group_id;
  return 1;
}

int32_t et_muon_aot_create(double muon_beta, double muon_eps, double muon_wd,
                           int64_t muon_ns_steps, int64_t muon_nesterov,
                           double adamw_beta1, double adamw_beta2,
                           double adamw_eps, double adamw_wd,
                           double adamw_lr_scale, int64_t wsd_warmup,
                           int64_t wsd_stable, int64_t wsd_total,
                           double wsd_max_lr, double wsd_min_lr,
                           int64_t wsd_decay_kind,
                           et_muon_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_MUON_ERR_ARGUMENT;
  *out_ctx = NULL;

  if (muon_ns_steps < 0 || muon_ns_steps > (int64_t)ET_MUON_MAX_NS_STEPS ||
      (muon_nesterov != 0 && muon_nesterov != 1) || wsd_warmup < 0 ||
      wsd_stable < 0 || wsd_total < 0 ||
      (wsd_decay_kind != (int64_t)ET_WSD_DECAY_LINEAR &&
       wsd_decay_kind != (int64_t)ET_WSD_DECAY_COSINE))
    return ET_MUON_ERR_ARGUMENT;

  et_wsd_config wsd = {(uint64_t)wsd_warmup,
                       (uint64_t)wsd_stable,
                       (uint64_t)wsd_total,
                       wsd_max_lr,
                       wsd_min_lr,
                       (uint32_t)wsd_decay_kind,
                       0u};
  et_muon_status st = et_wsd_validate_config(&wsd);
  if (st != ET_MUON_OK)
    return st;

  et_muon_step_config step_cfg = {wsd_max_lr,
                                  UINT64_C(1),
                                  muon_beta,
                                  muon_eps,
                                  muon_wd,
                                  (uint32_t)muon_ns_steps,
                                  (uint32_t)muon_nesterov,
                                  adamw_beta1,
                                  adamw_beta2,
                                  adamw_eps,
                                  adamw_wd,
                                  adamw_lr_scale};
  st = et_muon_validate_step_config(&step_cfg);
  if (st != ET_MUON_OK)
    return st;

  et_muon_aot_ctx *ctx = (et_muon_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_MUON_ERR_NOMEM;

  ctx->step_cfg = step_cfg;
  ctx->wsd_cfg = wsd;
  ctx->num_groups = 0u;
  ctx->last_lr = 0.0;
  ctx->last_phase = -1;
  ctx->last_ortho_error = -1.0;

  *out_ctx = ctx;
  return ET_MUON_OK;
}

void et_muon_aot_destroy(et_muon_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  for (size_t i = 0; i < ctx->num_groups; ++i) {
    free(ctx->groups[i].params);
    free((void *)(uintptr_t)ctx->groups[i].grads);
    free(ctx->groups[i].momentum_buf);
    free(ctx->groups[i].variance_buf);
  }
  free(ctx);
}

int32_t et_muon_aot_add_muon_group(et_muon_aot_ctx *ctx, int64_t rows,
                                   int64_t cols, int64_t apply_weight_decay) {
  if (ctx == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (apply_weight_decay != 0 && apply_weight_decay != 1)
    return ET_MUON_ERR_ARGUMENT;
  if (ctx->num_groups >= ET_MUON_MAX_GROUPS)
    return ET_MUON_ERR_SHAPE;
  if (rows < (int64_t)ET_MUON_MIN_DIM || rows > (int64_t)ET_MUON_MAX_DIM ||
      cols < (int64_t)ET_MUON_MIN_DIM || cols > (int64_t)ET_MUON_MAX_DIM)
    return ET_MUON_ERR_SHAPE;

  const size_t count = (size_t)rows * (size_t)cols;
  float *params = (float *)calloc(count, sizeof(float));
  float *grads = (float *)calloc(count, sizeof(float));
  float *mom = (float *)calloc(count, sizeof(float));
  if (params == NULL || grads == NULL || mom == NULL) {
    free(params);
    free(grads);
    free(mom);
    return ET_MUON_ERR_NOMEM;
  }

  const size_t gid = ctx->num_groups++;
  ctx->counts[gid] = count;
  ctx->groups[gid] = (et_muon_param_group){ET_MUON_GROUP_MUON_2D,
                                           (uint32_t)rows,
                                           (uint32_t)cols,
                                           (uint32_t)apply_weight_decay,
                                           params,
                                           grads,
                                           mom,
                                           NULL};
  return (int32_t)gid;
}

int32_t et_muon_aot_add_adamw_group(et_muon_aot_ctx *ctx, int64_t length,
                                    int64_t apply_weight_decay) {
  if (ctx == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (apply_weight_decay != 0 && apply_weight_decay != 1)
    return ET_MUON_ERR_ARGUMENT;
  if (ctx->num_groups >= ET_MUON_MAX_GROUPS)
    return ET_MUON_ERR_SHAPE;
  if (length < 1 || length > (int64_t)ET_MUON_MAX_VEC_LEN)
    return ET_MUON_ERR_SHAPE;

  const size_t count = (size_t)length;
  float *params = (float *)calloc(count, sizeof(float));
  float *grads = (float *)calloc(count, sizeof(float));
  float *mom = (float *)calloc(count, sizeof(float));
  float *var = (float *)calloc(count, sizeof(float));
  if (params == NULL || grads == NULL || mom == NULL || var == NULL) {
    free(params);
    free(grads);
    free(mom);
    free(var);
    return ET_MUON_ERR_NOMEM;
  }

  const size_t gid = ctx->num_groups++;
  ctx->counts[gid] = count;
  ctx->groups[gid] = (et_muon_param_group){ET_MUON_GROUP_ADAMW_1D,
                                           (uint32_t)length,
                                           1u,
                                           (uint32_t)apply_weight_decay,
                                           params,
                                           grads,
                                           mom,
                                           var};
  return (int32_t)gid;
}

int64_t et_muon_aot_group_count(const et_muon_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->num_groups;
}

int64_t et_muon_aot_group_size(const et_muon_aot_ctx *ctx, int64_t group_id) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid))
    return -1;
  return (int64_t)ctx->counts[gid];
}

int32_t et_muon_aot_set_param(et_muon_aot_ctx *ctx, int64_t group_id,
                              int64_t index, double value) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return ET_MUON_ERR_ARGUMENT;
  float narrowed = 0.0f;
  if (!narrow_f32_finite(value, &narrowed))
    return ET_MUON_ERR_NONFINITE;
  ((float *)ctx->groups[gid].params)[index] = narrowed;
  return ET_MUON_OK;
}

double et_muon_aot_get_param(const et_muon_aot_ctx *ctx, int64_t group_id,
                             int64_t index) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return NAN;
  return (double)((const float *)ctx->groups[gid].params)[index];
}

int32_t et_muon_aot_set_grad(et_muon_aot_ctx *ctx, int64_t group_id,
                             int64_t index, double value) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return ET_MUON_ERR_ARGUMENT;
  float narrowed = 0.0f;
  if (!narrow_f32_finite(value, &narrowed))
    return ET_MUON_ERR_NONFINITE;
  ((float *)(uintptr_t)ctx->groups[gid].grads)[index] = narrowed;
  return ET_MUON_OK;
}

double et_muon_aot_get_grad(const et_muon_aot_ctx *ctx, int64_t group_id,
                            int64_t index) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return NAN;
  return (double)((const float *)ctx->groups[gid].grads)[index];
}

double et_muon_aot_get_momentum(const et_muon_aot_ctx *ctx, int64_t group_id,
                                int64_t index) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return NAN;
  return (double)((const float *)ctx->groups[gid].momentum_buf)[index];
}

double et_muon_aot_get_variance(const et_muon_aot_ctx *ctx, int64_t group_id,
                                int64_t index) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid] ||
      ctx->groups[gid].variance_buf == NULL)
    return NAN;
  return (double)((const float *)ctx->groups[gid].variance_buf)[index];
}

int64_t et_muon_aot_param_bits(const et_muon_aot_ctx *ctx, int64_t group_id,
                               int64_t index) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid) || index < 0 ||
      (uint64_t)index >= ctx->counts[gid])
    return -1;
  uint32_t bits = 0u;
  const float val = ((const float *)ctx->groups[gid].params)[index];
  memcpy(&bits, &val, sizeof(bits));
  return (int64_t)bits;
}

int32_t et_muon_aot_query_wsd(et_muon_aot_ctx *ctx, int64_t step) {
  if (ctx == NULL || step < 0)
    return ET_MUON_ERR_ARGUMENT;
  double lr = 0.0;
  uint32_t phase = 0u;
  const et_muon_status st =
      et_wsd_lr(&ctx->wsd_cfg, (uint64_t)step, &lr, &phase);
  if (st != ET_MUON_OK)
    return st;
  ctx->last_lr = lr;
  ctx->last_phase = (int64_t)phase;
  return ET_MUON_OK;
}

double et_muon_aot_last_lr(const et_muon_aot_ctx *ctx) {
  return ctx == NULL ? NAN : ctx->last_lr;
}

int64_t et_muon_aot_last_phase(const et_muon_aot_ctx *ctx) {
  return ctx == NULL ? -1 : ctx->last_phase;
}

int32_t et_muon_aot_step_wsd(et_muon_aot_ctx *ctx, int64_t schedule_step) {
  if (ctx == NULL || schedule_step < 0)
    return ET_MUON_ERR_ARGUMENT;
  if (ctx->num_groups == 0u)
    return ET_MUON_ERR_SHAPE;
  double lr = 0.0;
  uint32_t phase = 0u;
  const et_muon_status st = et_muon_hybrid_step_wsd(
      &ctx->wsd_cfg, (uint64_t)schedule_step, &ctx->step_cfg, ctx->num_groups,
      ctx->groups, &lr, &phase);
  if (st != ET_MUON_OK)
    return st;
  ctx->last_lr = lr;
  ctx->last_phase = (int64_t)phase;
  return ET_MUON_OK;
}

int32_t et_muon_aot_group_ns5_ortho_error(et_muon_aot_ctx *ctx,
                                          int64_t group_id) {
  size_t gid = 0u;
  if (!valid_group_index(ctx, group_id, &gid))
    return ET_MUON_ERR_ARGUMENT;
  const et_muon_param_group *grp = &ctx->groups[gid];
  if (grp->kind != ET_MUON_GROUP_MUON_2D)
    return ET_MUON_ERR_ARGUMENT;

  const size_t count = ctx->counts[gid];
  float *u = (float *)malloc(count * sizeof(float));
  if (u == NULL)
    return ET_MUON_ERR_NOMEM;

  et_muon_status st = et_muon_newton_schulz5_raw(
      grp->rows, grp->cols, grp->grads, ctx->step_cfg.muon_ns_steps,
      ctx->step_cfg.muon_eps, u);
  if (st != ET_MUON_OK) {
    free(u);
    return st;
  }

  double err = 0.0;
  st = et_muon_orthogonality_error_f32(grp->rows, grp->cols, u, &err);
  free(u);
  if (st != ET_MUON_OK)
    return st;

  ctx->last_ortho_error = err;
  return ET_MUON_OK;
}

double et_muon_aot_last_ortho_error(const et_muon_aot_ctx *ctx) {
  return ctx == NULL ? NAN : ctx->last_ortho_error;
}

void *et_muon_aot_alloc_f32(int64_t count) {
  if (count <= 0 || count > ET_MUON_AOT_MAX_RAW_COUNT)
    return NULL;
  return calloc((size_t)count, sizeof(float));
}

void et_muon_aot_free_f32(void *buf) { free(buf); }

int32_t et_muon_aot_store_f32(void *buf, int64_t count, int64_t index,
                              double value) {
  if (buf == NULL || count <= 0 || count > ET_MUON_AOT_MAX_RAW_COUNT ||
      index < 0 || index >= count)
    return ET_MUON_ERR_ARGUMENT;
  if (((uintptr_t)buf % _Alignof(float)) != 0u)
    return ET_MUON_ERR_ALIGNMENT;
  ((float *)buf)[index] = raw_f32_operand(value);
  return ET_MUON_OK;
}

double et_muon_aot_load_f32(const void *buf, int64_t count, int64_t index) {
  if (buf == NULL || count <= 0 || count > ET_MUON_AOT_MAX_RAW_COUNT ||
      index < 0 || index >= count || ((uintptr_t)buf % _Alignof(float)) != 0u)
    return NAN;
  return (double)((const float *)buf)[index];
}

int32_t et_muon_aot_newton_schulz5(int64_t rows, int64_t cols, const void *g,
                                   int64_t ns_steps, double eps, void *u) {
  if (rows < 0 || cols < 0 || ns_steps < 0 ||
      rows > (int64_t)UINT32_MAX || cols > (int64_t)UINT32_MAX ||
      ns_steps > (int64_t)UINT32_MAX)
    return ET_MUON_ERR_ARGUMENT;
  return et_muon_newton_schulz5_raw((uint32_t)rows, (uint32_t)cols, g,
                                    (uint32_t)ns_steps, eps, u);
}

int32_t et_muon_aot_orthogonality_error(int64_t rows, int64_t cols,
                                        const void *u, double *out_error) {
  if (rows < 0 || cols < 0 || rows > (int64_t)UINT32_MAX ||
      cols > (int64_t)UINT32_MAX)
    return ET_MUON_ERR_ARGUMENT;
  return et_muon_orthogonality_error_raw((uint32_t)rows, (uint32_t)cols, u,
                                         out_error);
}
