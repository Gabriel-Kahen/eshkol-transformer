#ifndef ESHKOL_TRANSFORMER_GRPO_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_GRPO_ESHKOL_BRIDGE_H

#include "eshkol_transformer/grpo_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_GRPO_AOT_MAX_GROUPS 8u
#define ET_GRPO_AOT_MAX_GROUP_SIZE 8u
#define ET_GRPO_AOT_MAX_BATCH 32u
#define ET_GRPO_AOT_MAX_SEQ_LEN 16u
#define ET_GRPO_AOT_MAX_VOCAB 16u
#define ET_GRPO_AOT_MAX_DIM 16u

typedef struct et_grpo_aot_ctx et_grpo_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT GRPO / RLVR alignment driver. */
int32_t et_grpo_aot_create(int64_t num_groups, int64_t group_size,
                           int64_t seq_len, int64_t vocab_size,
                           int64_t feat_dim, et_grpo_aot_ctx **out_ctx);
void et_grpo_aot_destroy(et_grpo_aot_ctx *ctx);

/* Configure GRPO loss hyperparameters. */
int32_t et_grpo_aot_set_loss_params(et_grpo_aot_ctx *ctx,
                                    int64_t mask_norm_mode,
                                    double clip_eps_low, double clip_eps_high,
                                    double kl_beta);

/* Dimension queries. */
int64_t et_grpo_aot_batch_size(const et_grpo_aot_ctx *ctx);
int64_t et_grpo_aot_vocab_size(const et_grpo_aot_ctx *ctx);
int64_t et_grpo_aot_feat_dim(const et_grpo_aot_ctx *ctx);

/* Tensor and parameter accessors. */
int32_t et_grpo_aot_set_feature(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                                int64_t d, double value);
int32_t et_grpo_aot_set_weight(et_grpo_aot_ctx *ctx, int64_t v, int64_t d,
                               double value);
double et_grpo_aot_get_weight(const et_grpo_aot_ctx *ctx, int64_t v, int64_t d);
double et_grpo_aot_get_grad_weight(const et_grpo_aot_ctx *ctx, int64_t v,
                                   int64_t d);

int32_t et_grpo_aot_set_bias(et_grpo_aot_ctx *ctx, int64_t v, double value);
double et_grpo_aot_get_bias(const et_grpo_aot_ctx *ctx, int64_t v);
double et_grpo_aot_get_grad_bias(const et_grpo_aot_ctx *ctx, int64_t v);

int32_t et_grpo_aot_set_target(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                               int64_t token_id);
int32_t et_grpo_aot_set_mask(et_grpo_aot_ctx *ctx, int64_t n, int64_t t,
                             int64_t mask_val);
int32_t et_grpo_aot_set_reward(et_grpo_aot_ctx *ctx, int64_t n, double value);
double et_grpo_aot_get_reward(const et_grpo_aot_ctx *ctx, int64_t n);
double et_grpo_aot_get_advantage(const et_grpo_aot_ctx *ctx, int64_t n);
double et_grpo_aot_get_seq_logprob(const et_grpo_aot_ctx *ctx, int64_t n);

/* Deterministic RLVR verifier bridge helpers. */
int32_t et_grpo_aot_score_rlvr_completion(et_grpo_aot_ctx *ctx,
                                          int64_t sample_idx,
                                          int64_t completion_case_id);
double et_grpo_aot_verify_exact_case(int64_t case_id);

/* Core policy + GRPO operations. */
int32_t et_grpo_aot_policy_forward(et_grpo_aot_ctx *ctx);
int32_t et_grpo_aot_snapshot_ref(et_grpo_aot_ctx *ctx);
int32_t et_grpo_aot_snapshot_old(et_grpo_aot_ctx *ctx);
int32_t et_grpo_aot_compute_advantages(et_grpo_aot_ctx *ctx, int64_t adv_mode,
                                       double eps);
int32_t et_grpo_aot_loss_forward(et_grpo_aot_ctx *ctx);
int32_t et_grpo_aot_loss_backward(et_grpo_aot_ctx *ctx);

/* Policy evaluation metrics. */
double et_grpo_aot_expected_reward(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_greedy_reward(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_metric_loss(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_metric_pg_loss(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_metric_mean_kl(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_metric_clip_fraction(const et_grpo_aot_ctx *ctx);
double et_grpo_aot_metric_mean_reward(const et_grpo_aot_ctx *ctx);
int64_t et_grpo_aot_metric_active_tokens(const et_grpo_aot_ctx *ctx);
int64_t et_grpo_aot_metric_clipped_tokens(const et_grpo_aot_ctx *ctx);

/* Negative fail-closed validation probes returning exact et_grpo_status codes. */
int32_t et_grpo_aot_check_negative_case(const et_grpo_aot_ctx *ctx,
                                        int64_t case_id);

#ifdef __cplusplus
}
#endif

#endif
