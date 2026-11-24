#ifndef ESHKOL_TRANSFORMER_PPO_RLHF_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_PPO_RLHF_ESHKOL_BRIDGE_H

#include "eshkol_transformer/ppo_rlhf_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_PPO_AOT_MAX_BATCH 32u
#define ET_PPO_AOT_MAX_SEQ_LEN 16u
#define ET_PPO_AOT_MAX_FEAT 16u
#define ET_PPO_AOT_MAX_VOCAB 16u

typedef struct et_ppo_aot_ctx et_ppo_aot_ctx;

/* Lifecycle for compiled Eshkol AOT PPO-RLHF driver context. */
int32_t et_ppo_aot_create(int64_t batch_size, int64_t seq_len, int64_t feat_dim,
                          int64_t vocab_size, et_ppo_aot_ctx **out_ctx);
void et_ppo_aot_destroy(et_ppo_aot_ctx *ctx);

/* Configure PPO-RLHF trainer hyperparameters. */
int32_t et_ppo_aot_configure(et_ppo_aot_ctx *ctx, int64_t ppo_epochs,
                             int64_t use_value_clip, int64_t whiten_advantages,
                             double clip_eps_low, double clip_eps_high,
                             double value_clip_eps, double vf_coef,
                             double ent_coef, double gamma, double gae_lambda,
                             double beta_kl, double adv_eps, double target_kl,
                             double lr, double weight_decay,
                             double max_grad_norm);

/* Initialize deterministic contextual RLHF environment, actor-critic weights,
 * and frozen reference policy (snapshot of initial actor weights). */
int32_t et_ppo_aot_init_task(et_ppo_aot_ctx *ctx);

/* Direct PPO loss/backward FD workspace setup and accessors (for in-Eshkol
 * central finite-difference checks on d_policy_logits and d_values). */
int32_t et_ppo_aot_setup_fd_case(et_ppo_aot_ctx *ctx);
int32_t et_ppo_aot_fd_loss_forward(et_ppo_aot_ctx *ctx);
int32_t et_ppo_aot_fd_loss_backward(et_ppo_aot_ctx *ctx);
double et_ppo_aot_fd_total_loss(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_fd_get_logit(const et_ppo_aot_ctx *ctx, int64_t idx,
                               int64_t v);
int32_t et_ppo_aot_fd_set_logit(et_ppo_aot_ctx *ctx, int64_t idx, int64_t v,
                                double value);
double et_ppo_aot_fd_get_d_logit(const et_ppo_aot_ctx *ctx, int64_t idx,
                                 int64_t v);
double et_ppo_aot_fd_get_value(const et_ppo_aot_ctx *ctx, int64_t idx);
int32_t et_ppo_aot_fd_set_value(et_ppo_aot_ctx *ctx, int64_t idx, double value);
double et_ppo_aot_fd_get_d_value(const et_ppo_aot_ctx *ctx, int64_t idx);
int64_t et_ppo_aot_fd_bt(const et_ppo_aot_ctx *ctx);
int64_t et_ppo_aot_fd_vocab(const et_ppo_aot_ctx *ctx);

/* Deterministic on-policy rollout sampling keyed by iteration index, followed
 * by multi-epoch PPO update via et_ppo_train_rollout_v1. */
int32_t et_ppo_aot_sample_rollout(et_ppo_aot_ctx *ctx, int64_t iter_idx);
int32_t et_ppo_aot_train_rollout(et_ppo_aot_ctx *ctx);
int32_t et_ppo_aot_set_target_kl(et_ppo_aot_ctx *ctx, double target_kl);

/* Policy reward, diagnostic, and step queries. */
double et_ppo_aot_expected_env_reward(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_mean_env_reward(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_mean_shaped_reward(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_total_loss(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_policy_loss(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_value_loss(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_entropy(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_approx_kl(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_clip_fraction(const et_ppo_aot_ctx *ctx);
double et_ppo_aot_report_explained_variance(const et_ppo_aot_ctx *ctx);
int64_t et_ppo_aot_report_epochs_completed(const et_ppo_aot_ctx *ctx);
int64_t et_ppo_aot_report_early_stopped(const et_ppo_aot_ctx *ctx);
int64_t et_ppo_aot_step(const et_ppo_aot_ctx *ctx);

/* Canonical SHA-256 digests of frozen reference policy and trainable params,
 * exposed as four 64-bit big-endian words (word_index in 0..3). */
int32_t et_ppo_aot_refresh_digests(et_ppo_aot_ctx *ctx);
int64_t et_ppo_aot_ref_sha256_word(const et_ppo_aot_ctx *ctx,
                                   int64_t word_index);
int64_t et_ppo_aot_param_sha256_word(const et_ppo_aot_ctx *ctx,
                                     int64_t word_index);

/* Fresh-process ETPP1000 checkpoint env coordination (PPO_AOT_PHASE,
 * PPO_AOT_CKPT, PPO_AOT_TAMPER, PPO_AOT_TRUNC, PPO_AOT_SYMLINK). */
int64_t et_ppo_aot_env_phase(void);
int32_t et_ppo_aot_save_env_ckpt(const et_ppo_aot_ctx *ctx);
int32_t et_ppo_aot_load_env_ckpt(et_ppo_aot_ctx *ctx);
int32_t et_ppo_aot_check_ckpt_negative(const et_ppo_aot_ctx *ctx,
                                       int64_t case_id);

/* Core fail-closed negative boundary probes returning exact et_ppo_status. */
int32_t et_ppo_aot_check_negative_case(const et_ppo_aot_ctx *ctx,
                                       int64_t case_id);

#ifdef __cplusplus
}
#endif

#endif
