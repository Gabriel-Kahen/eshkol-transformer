#ifndef ESHKOL_TRANSFORMER_O2_MUON_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_O2_MUON_ESHKOL_BRIDGE_H

#include "eshkol_transformer/o2_muon_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_muon_aot_ctx et_muon_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT hybrid Muon + AdamW + WSD. */
int32_t et_muon_aot_create(double muon_beta, double muon_eps, double muon_wd,
                           int64_t muon_ns_steps, int64_t muon_nesterov,
                           double adamw_beta1, double adamw_beta2,
                           double adamw_eps, double adamw_wd,
                           double adamw_lr_scale, int64_t wsd_warmup,
                           int64_t wsd_stable, int64_t wsd_total,
                           double wsd_max_lr, double wsd_min_lr,
                           int64_t wsd_decay_kind,
                           et_muon_aot_ctx **out_ctx);
void et_muon_aot_destroy(et_muon_aot_ctx *ctx);

/* Register 2D Muon or 1D AdamW parameter groups.
 * Returns non-negative group index on success, or negative et_muon_status. */
int32_t et_muon_aot_add_muon_group(et_muon_aot_ctx *ctx, int64_t rows,
                                   int64_t cols, int64_t apply_weight_decay);
int32_t et_muon_aot_add_adamw_group(et_muon_aot_ctx *ctx, int64_t length,
                                    int64_t apply_weight_decay);

int64_t et_muon_aot_group_count(const et_muon_aot_ctx *ctx);
int64_t et_muon_aot_group_size(const et_muon_aot_ctx *ctx, int64_t group_id);

/* Context parameter/gradient/state marshaling (binary32 <-> f64). */
int32_t et_muon_aot_set_param(et_muon_aot_ctx *ctx, int64_t group_id,
                              int64_t index, double value);
double et_muon_aot_get_param(const et_muon_aot_ctx *ctx, int64_t group_id,
                             int64_t index);
int32_t et_muon_aot_set_grad(et_muon_aot_ctx *ctx, int64_t group_id,
                             int64_t index, double value);
double et_muon_aot_get_grad(const et_muon_aot_ctx *ctx, int64_t group_id,
                            int64_t index);
double et_muon_aot_get_momentum(const et_muon_aot_ctx *ctx, int64_t group_id,
                                int64_t index);
double et_muon_aot_get_variance(const et_muon_aot_ctx *ctx, int64_t group_id,
                                int64_t index);
int64_t et_muon_aot_param_bits(const et_muon_aot_ctx *ctx, int64_t group_id,
                               int64_t index);

/* WSD LR query and cached accessors. */
int32_t et_muon_aot_query_wsd(et_muon_aot_ctx *ctx, int64_t step);
double et_muon_aot_last_lr(const et_muon_aot_ctx *ctx);
int64_t et_muon_aot_last_phase(const et_muon_aot_ctx *ctx);

/* Failure-atomic WSD-scheduled hybrid Muon + AdamW step. */
int32_t et_muon_aot_step_wsd(et_muon_aot_ctx *ctx, int64_t schedule_step);

/* Run Newton-Schulz5 on a registered 2D Muon group's current gradient buffer
 * and cache its orthogonality error ||U U^T - I||_F (or ||U^T U - I||_F). */
int32_t et_muon_aot_group_ns5_ortho_error(et_muon_aot_ctx *ctx,
                                          int64_t group_id);
double et_muon_aot_last_ortho_error(const et_muon_aot_ctx *ctx);

/* Direct raw f32 buffer marshaling and standalone NS5 / orthogonality calls. */
void *et_muon_aot_alloc_f32(int64_t count);
void et_muon_aot_free_f32(void *buf);
int32_t et_muon_aot_store_f32(void *buf, int64_t count, int64_t index,
                              double value);
double et_muon_aot_load_f32(const void *buf, int64_t count, int64_t index);
int32_t et_muon_aot_newton_schulz5(int64_t rows, int64_t cols, const void *g,
                                   int64_t ns_steps, double eps, void *u);
int32_t et_muon_aot_orthogonality_error(int64_t rows, int64_t cols,
                                        const void *u, double *out_error);

#ifdef __cplusplus
}
#endif

#endif
