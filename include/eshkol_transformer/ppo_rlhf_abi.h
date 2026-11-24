#ifndef ESHKOL_TRANSFORMER_PPO_RLHF_ABI_H
#define ESHKOL_TRANSFORMER_PPO_RLHF_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_PPO_ABI_MAJOR 1u
#define ET_PPO_ABI_MINOR 0u
#define ET_PPO_SHA256_BYTES 32u

/* Bounded dimensions preventing size_t overflow and unbounded compute. */
#define ET_PPO_MAX_BATCH 256u
#define ET_PPO_MAX_SEQ_LEN 256u
#define ET_PPO_MAX_VOCAB 4096u
#define ET_PPO_MAX_LOGITS_ELEMS 4194304u

/* Bounded dimensions for the built-in linear Actor-Critic trainer & ETPP1000
 * checkpoint format (zero heap allocation in all core compute & IO paths). */
#define ET_PPO_MAX_MODEL_FEAT 64u
#define ET_PPO_MAX_MODEL_VOCAB 64u
#define ET_PPO_TRAINER_MAX_BT 1024u
#define ET_PPO_CHECKPOINT_MAX_BYTES 65536u
#define ET_PPO_MAX_EPOCHS 64u

/* Explicit overflow guard on log-probability differences before exp():
 *   delta_lp = logpi_new - old_logprob > ET_PPO_MAX_LOG_DIFF
 * Values above 50.0 nats (~5.18e21 ratio) fail closed with ET_PPO_ERR_OVERFLOW.
 */
#define ET_PPO_MAX_LOG_DIFF 50.0f

typedef int32_t et_ppo_status;

enum {
  ET_PPO_OK = 0,
  ET_PPO_ERR_ARGUMENT = -1,
  ET_PPO_ERR_SHAPE = -2,
  ET_PPO_ERR_ALIGNMENT = -3,
  ET_PPO_ERR_ALIAS = -4,
  ET_PPO_ERR_NONFINITE = -5,
  ET_PPO_ERR_FP_ENV = -6,
  ET_PPO_ERR_OVERFLOW = -7,
  ET_PPO_ERR_INDEX = -8,
  ET_PPO_ERR_EMPTY_MASK = -9,
  ET_PPO_ERR_IO = -10,
  ET_PPO_ERR_SYMLINK = -11,
  ET_PPO_ERR_FORMAT = -12,
  ET_PPO_ERR_CHECKSUM = -13,
  ET_PPO_ERR_TRUNCATED = -14,
  ET_PPO_ERR_REF_MISMATCH = -15,
  ET_PPO_ERR_NOMEM = -16
};

/* PPO-Clip policy + clipped value critic + entropy loss configuration.
 * Constraints:
 *   1 <= batch_size <= ET_PPO_MAX_BATCH
 *   1 <= seq_len <= ET_PPO_MAX_SEQ_LEN
 *   1 <= vocab_size <= ET_PPO_MAX_VOCAB
 *   batch_size * seq_len * vocab_size <= ET_PPO_MAX_LOGITS_ELEMS
 *   use_value_clip in {0u, 1u}
 *   clip_eps_low finite, 0.0f < clip_eps_low <= 1.0f
 *   clip_eps_high finite, clip_eps_high > 0.0f
 *   value_clip_eps finite, > 0.0f when use_value_clip == 1u (else >= 0.0f)
 *   vf_coef finite, >= 0.0f
 *   ent_coef finite, >= 0.0f
 *   reserved == 0u
 */
typedef struct et_ppo_loss_config {
  uint32_t batch_size;
  uint32_t seq_len;
  uint32_t vocab_size;
  uint32_t use_value_clip;
  float clip_eps_low;
  float clip_eps_high;
  float value_clip_eps;
  float vf_coef;
  float ent_coef;
  uint32_t reserved;
} et_ppo_loss_config;

/* Exact loss and diagnostic metrics produced by PPO forward/backward ops.
 * Diagnostics over active tokens (mask[b, t] == 1, count M >= 1):
 *   total_loss          = policy_loss + vf_coef * value_loss
 *                         - ent_coef * mean_entropy
 *   approx_kl           = mean_active(0.5 * (logpi_new - old_logprob)^2)
 *   clip_fraction       = clipped_tokens / active_tokens
 *   value_clip_fraction = value_clipped_tokens / active_tokens
 *   explained_variance  = 1 - Var(R - v) / Var(R)  (0.0f when Var(R) == 0)
 */
typedef struct et_ppo_loss_metrics {
  float total_loss;
  float policy_loss;
  float value_loss;
  float mean_entropy;
  float approx_kl;
  float clip_fraction;
  float value_clip_fraction;
  float explained_variance;
  uint32_t active_tokens;
  uint32_t clipped_tokens;
  uint32_t value_clipped_tokens;
  uint32_t reserved;
} et_ppo_loss_metrics;

/* Decoupled AdamW optimizer configuration.
 * Constraints:
 *   lr finite, > 0.0f
 *   beta1 finite, 0.0f <= beta1 < 1.0f
 *   beta2 finite, 0.0f <= beta2 < 1.0f
 *   eps finite, > 0.0f
 *   weight_decay finite, >= 0.0f
 *   max_grad_norm finite, >= 0.0f (0.0f disables global norm clipping)
 *   reserved[0] == 0u && reserved[1] == 0u
 */
typedef struct et_ppo_adamw_config {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float max_grad_norm;
  uint32_t reserved[2];
} et_ppo_adamw_config;

/* Multi-epoch RLHF PPO Actor-Critic rollout trainer configuration.
 * Parameter vector layout (length P = V*D + V + D + 1):
 *   w_pi[V * D], b_pi[V], w_v[D], b_v[1]
 * Frozen reference policy vector layout (length P_ref = V*D + V):
 *   w_ref[V * D], b_ref[V]
 * Constraints:
 *   1 <= feat_dim <= ET_PPO_MAX_MODEL_FEAT
 *   2 <= loss_cfg.vocab_size <= ET_PPO_MAX_MODEL_VOCAB
 *   loss_cfg.batch_size * loss_cfg.seq_len <= ET_PPO_TRAINER_MAX_BT
 *   1 <= ppo_epochs <= ET_PPO_MAX_EPOCHS
 *   whiten_advantages in {0u, 1u}
 *   gamma finite, 0.0f <= gamma <= 1.0f
 *   gae_lambda finite, 0.0f <= gae_lambda <= 1.0f
 *   beta_kl finite, >= 0.0f
 *   adv_eps finite, > 0.0f when whiten_advantages == 1u (else >= 0.0f)
 *   target_kl finite, >= 0.0f (0.0f disables KL early stopping)
 *   reserved == 0u && reserved2 == 0u
 */
typedef struct et_ppo_trainer_config {
  et_ppo_loss_config loss_cfg;
  et_ppo_adamw_config optim_cfg;
  uint32_t feat_dim;
  uint32_t ppo_epochs;
  uint32_t whiten_advantages;
  uint32_t reserved;
  float gamma;
  float gae_lambda;
  float beta_kl;
  float adv_eps;
  float target_kl;
  uint32_t reserved2;
} et_ppo_trainer_config;

/* Summary produced by one multi-epoch PPO rollout update. */
typedef struct et_ppo_rollout_report {
  et_ppo_loss_metrics final_metrics;
  uint32_t epochs_completed;
  uint32_t early_stopped;
  uint32_t stopped_epoch;
  uint32_t reserved;
  float mean_env_reward;
  float mean_shaped_reward;
} et_ppo_rollout_report;

/* Metadata restored by et_ppo_checkpoint_load_v1 (ETPP1000 v1.0). */
typedef struct et_ppo_checkpoint_meta {
  uint32_t version_major;
  uint32_t version_minor;
  uint32_t feat_dim;
  uint32_t vocab_size;
  uint32_t param_count;
  uint32_t reserved;
  uint64_t step;
  et_ppo_trainer_config cfg;
  uint8_t ref_sha256[ET_PPO_SHA256_BYTES];
} et_ppo_checkpoint_meta;

#if defined(__cplusplus)
#define ET_PPO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_PPO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_loss_config) == 40u &&
                         offsetof(et_ppo_loss_config, clip_eps_low) == 16u &&
                         offsetof(et_ppo_loss_config, reserved) == 36u,
                     "et_ppo_loss_config ABI layout changed");
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_loss_metrics) == 48u &&
                         offsetof(et_ppo_loss_metrics, approx_kl) == 16u &&
                         offsetof(et_ppo_loss_metrics, active_tokens) == 32u &&
                         offsetof(et_ppo_loss_metrics, reserved) == 44u,
                     "et_ppo_loss_metrics ABI layout changed");
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_adamw_config) == 32u &&
                         offsetof(et_ppo_adamw_config, weight_decay) == 16u &&
                         offsetof(et_ppo_adamw_config, reserved) == 24u,
                     "et_ppo_adamw_config ABI layout changed");
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_trainer_config) == 112u &&
                         offsetof(et_ppo_trainer_config, optim_cfg) == 40u &&
                         offsetof(et_ppo_trainer_config, feat_dim) == 72u &&
                         offsetof(et_ppo_trainer_config, gamma) == 88u &&
                         offsetof(et_ppo_trainer_config, reserved2) == 108u,
                     "et_ppo_trainer_config ABI layout changed");
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_rollout_report) == 72u &&
                         offsetof(et_ppo_rollout_report, epochs_completed) ==
                             48u &&
                         offsetof(et_ppo_rollout_report, mean_env_reward) ==
                             64u,
                     "et_ppo_rollout_report ABI layout changed");
ET_PPO_STATIC_ASSERT(sizeof(et_ppo_checkpoint_meta) == 176u &&
                         offsetof(et_ppo_checkpoint_meta, step) == 24u &&
                         offsetof(et_ppo_checkpoint_meta, cfg) == 32u &&
                         offsetof(et_ppo_checkpoint_meta, ref_sha256) == 144u,
                     "et_ppo_checkpoint_meta ABI layout changed");
#undef ET_PPO_STATIC_ASSERT

/* Configuration & dimension helpers. */
et_ppo_status et_ppo_validate_loss_config(const et_ppo_loss_config *cfg);
et_ppo_status et_ppo_validate_trainer_config(const et_ppo_trainer_config *cfg);
et_ppo_status et_ppo_model_param_count(uint32_t feat_dim, uint32_t vocab_size,
                                       size_t *out_count);
et_ppo_status et_ppo_ref_param_count(uint32_t feat_dim, uint32_t vocab_size,
                                     size_t *out_count);

/* Canonical IEEE-754 little-endian binary32 SHA-256 over finite float buffer.
 * Zero heap allocations; failure leaves out_digest untouched. */
et_ppo_status et_ppo_weight_sha256_v1(size_t float_count, const float *weights,
                                      uint8_t out_digest[ET_PPO_SHA256_BYTES]);

/* Per-token KL-shaped rewards:
 *   mask[b, t] == 1: r_shaped = r_env - beta_kl * (logpi_old - logpi_ref)
 *   mask[b, t] == 0: r_shaped = 0.0f
 * Zero heap allocations; failure leaves out_r_shaped untouched. */
et_ppo_status et_ppo_kl_shaped_rewards_v1(uint32_t batch_size, uint32_t seq_len,
                                          float beta_kl, const float *r_env,
                                          const float *logpi_old,
                                          const float *logpi_ref,
                                          const uint8_t *mask,
                                          float *out_r_shaped);

/* Generalized Advantage Estimation (GAE):
 * Reverse-time recurrence per sequence b in [0, B), t = T-1 .. 0:
 *   mask[b, t] == 0 -> out_adv = 0, out_ret = 0, next_v = 0, next_adv = 0
 *   mask[b, t] == 1 -> delta = r_shaped_t + gamma * next_v - v_t
 *                      A_t   = delta + gamma * gae_lambda * next_adv
 *                      R_t   = A_t + v_t
 *                      next_v = v_t, next_adv = A_t
 * Requires active_tokens >= 1. Zero heap allocations; failure-atomic. */
et_ppo_status et_ppo_gae_v1(uint32_t batch_size, uint32_t seq_len, float gamma,
                            float gae_lambda, const float *r_shaped,
                            const float *values, const uint8_t *mask,
                            float *out_advantages, float *out_returns);

/* Exact masked advantage whitening over active tokens (mask[b, t] == 1):
 *   mean = sum_active(A) / count            (direct binary64 division)
 *   var  = sum_active((A - mean)^2) / count (population variance in binary64)
 *   out  = (A - mean) / sqrt(var + eps)     (0.0f on masked positions)
 * Because direct binary64 division avoids reciprocal double-rounding, all-equal
 * finite binary32 advantages yield exact +0.0f for every count in [1, 65536].
 * Requires eps > 0.0f and active_tokens >= 1. Zero heap allocations. */
et_ppo_status et_ppo_whiten_advantages_v1(uint32_t batch_size, uint32_t seq_len,
                                          float eps, const float *advantages,
                                          const uint8_t *mask,
                                          float *out_whitened);

/* Numerically stable per-token action log-probabilities via log-softmax.
 * Masked positions write 0.0f. Zero heap allocations; failure-atomic. */
et_ppo_status et_ppo_token_logprobs_v1(uint32_t batch_size, uint32_t seq_len,
                                       uint32_t vocab_size,
                                       const float *policy_logits,
                                       const int32_t *actions,
                                       const uint8_t *mask,
                                       float *out_logprobs);

/* PPO-Clip Policy + Clipped Value Critic + Entropy Forward.
 * Tie conventions at non-differentiable kinks (strict inequality):
 *   policy clip active iff (A > 0 && ratio > 1 + eps_high) ||
 *                          (A < 0 && ratio < 1 - eps_low)
 *   value clip active  iff use_value_clip == 1 &&
 *                          (v_clipped - R)^2 > (v - R)^2
 * Zero heap allocations; failure leaves out_metrics untouched. */
et_ppo_status et_ppo_loss_forward_v1(
    const et_ppo_loss_config *cfg, const float *policy_logits,
    const float *values, const int32_t *actions, const float *old_logprobs,
    const float *old_values, const float *advantages, const float *returns,
    const uint8_t *mask, et_ppo_loss_metrics *out_metrics);

/* Exact analytic backward w.r.t. policy_logits[B, T, V] and values[B, T].
 * out_metrics is optional (may be NULL). Zero heap allocations; failure-atomic.
 */
et_ppo_status et_ppo_loss_backward_v1(
    const et_ppo_loss_config *cfg, const float *policy_logits,
    const float *values, const int32_t *actions, const float *old_logprobs,
    const float *old_values, const float *advantages, const float *returns,
    const uint8_t *mask, float *out_d_policy_logits, float *out_d_values,
    et_ppo_loss_metrics *out_metrics);

/* Linear Actor + Value Critic forward pass over features[B, T, D].
 * Zero heap allocations; failure-atomic. */
et_ppo_status et_ppo_actor_critic_forward_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t feat_dim,
    uint32_t vocab_size, const float *features, const float *params,
    float *out_logits, float *out_values);

/* Frozen reference policy per-token log-probabilities over features[B, T, D].
 * Zero heap allocations; failure-atomic. */
et_ppo_status et_ppo_ref_policy_logprobs_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t feat_dim,
    uint32_t vocab_size, const float *features, const float *ref_params,
    const int32_t *actions, const uint8_t *mask, float *out_ref_logprobs);

/* Decoupled AdamW optimizer step in IEEE-754 binary64 with optional global L2
 * gradient norm clipping. Zero heap allocations; failure-atomic. */
et_ppo_status et_ppo_adamw_step_v1(const et_ppo_adamw_config *cfg,
                                   size_t param_count, const float *grads,
                                   float *io_params, float *io_m, float *io_v,
                                   uint64_t *io_step);

/* Multi-epoch RLHF PPO rollout trainer with KL-shaped rewards, GAE, optional
 * advantage whitening, clipped surrogate + clipped value critic + entropy,
 * target_kl early stopping, and AdamW optimization.
 * Frozen ref_params is strictly read-only. Zero heap allocations; failure
 * leaves io_params, io_adam_m, io_adam_v, io_step, and out_report untouched. */
et_ppo_status et_ppo_train_rollout_v1(
    const et_ppo_trainer_config *cfg, const float *features,
    const int32_t *actions, const float *r_env, const uint8_t *mask,
    const float *ref_params, float *io_params, float *io_adam_m,
    float *io_adam_v, uint64_t *io_step, et_ppo_rollout_report *out_report);

/* Versioned SHA-256 checksummed ETPP1000 v1.0 checkpoint atomic 0600 save
 * (temp + fsync + rename) and O_NOFOLLOW checked load with frozen ref_sha256
 * binding. Zero heap allocations; load failure leaves all outputs untouched. */
et_ppo_status et_ppo_checkpoint_save_v1(const char *path,
                                        const et_ppo_trainer_config *cfg,
                                        uint64_t step, const float *ref_params,
                                        const float *params,
                                        const float *adam_m,
                                        const float *adam_v);

et_ppo_status et_ppo_checkpoint_load_v1(
    const char *path, uint32_t expected_feat_dim, uint32_t expected_vocab_size,
    const float *ref_params, et_ppo_checkpoint_meta *out_meta,
    float *out_params, float *out_adam_m, float *out_adam_v);

#ifdef __cplusplus
}
#endif

#endif
