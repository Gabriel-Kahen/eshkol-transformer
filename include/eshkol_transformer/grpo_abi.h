#ifndef ESHKOL_TRANSFORMER_GRPO_ABI_H
#define ESHKOL_TRANSFORMER_GRPO_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_GRPO_ABI_MAJOR 1u
#define ET_GRPO_ABI_MINOR 0u

/* Bounded dimensions preventing size_t overflow and unbounded compute. */
#define ET_GRPO_MAX_GROUPS 256u
#define ET_GRPO_MAX_GROUP_SIZE 64u
#define ET_GRPO_MAX_BATCH 1024u
#define ET_GRPO_MAX_SEQ_LEN 256u
#define ET_GRPO_MAX_VOCAB 4096u
#define ET_GRPO_MAX_LOGITS_ELEMS 4194304u
#define ET_GRPO_MAX_VERIFY_BYTES 16384u

/* Explicit overflow guard on log-probability differences before exp():
 *   delta_old = log_pi - log_old <= ET_GRPO_MAX_LOG_DIFF
 *   delta_ref = log_ref - log_pi <= ET_GRPO_MAX_LOG_DIFF
 * Values above 50.0 nats (~5.18e21 ratio) fail closed with ET_GRPO_ERR_OVERFLOW.
 */
#define ET_GRPO_MAX_LOG_DIFF 50.0f

typedef int32_t et_grpo_status;

enum {
  ET_GRPO_OK = 0,
  ET_GRPO_ERR_ARGUMENT = -1,
  ET_GRPO_ERR_SHAPE = -2,
  ET_GRPO_ERR_ALIGNMENT = -3,
  ET_GRPO_ERR_ALIAS = -4,
  ET_GRPO_ERR_NONFINITE = -5,
  ET_GRPO_ERR_FP_ENV = -6,
  ET_GRPO_ERR_OVERFLOW = -7,
  ET_GRPO_ERR_INDEX = -8,
  ET_GRPO_ERR_EMPTY_MASK = -9,
  ET_GRPO_ERR_NOMEM = -10
};

/* Group-relative advantage normalization modes.
 * Canonical population variance formula (accumulated in IEEE-754 binary64,
 * rounded to binary32 on output):
 *   mean_b = (1 / G) * sum_{g=0}^{G-1} r[b, g]
 *   var_b  = (1 / G) * sum_{g=0}^{G-1} (r[b, g] - mean_b)^2
 *   ET_GRPO_ADV_STANDARD: A[b, g] = (r[b, g] - mean_b) / (sqrt(var_b) + eps)
 *   ET_GRPO_ADV_DR_GRPO:  A[b, g] = r[b, g] - mean_b
 */
enum {
  ET_GRPO_ADV_STANDARD = 0u,
  ET_GRPO_ADV_DR_GRPO = 1u
};

/* Token mask reduction modes for GRPO objective and KL penalty.
 * Let M_n = sum_t mask[n, t], M_total = sum_n M_n, N_active = |{n : M_n > 0}|.
 *   ET_GRPO_NORM_GLOBAL_TOKEN_MEAN: w[n, t] = 1 / M_total
 *   ET_GRPO_NORM_PER_SEQ_MEAN:      w[n, t] = 1 / (N_active * M_n)
 * In both modes, sum_{mask[n, t] == 1} w[n, t] == 1.
 */
enum {
  ET_GRPO_NORM_GLOBAL_TOKEN_MEAN = 0u,
  ET_GRPO_NORM_PER_SEQ_MEAN = 1u
};

/* Deterministic RLVR verifier flags. */
enum {
  ET_GRPO_VERIFY_EXACT_BYTES = 0u,
  ET_GRPO_VERIFY_TRIM_WS = 1u
};

/* GRPO loss configuration.
 * Constraints:
 *   1 <= batch_size <= ET_GRPO_MAX_BATCH
 *   1 <= seq_len <= ET_GRPO_MAX_SEQ_LEN
 *   1 <= vocab_size <= ET_GRPO_MAX_VOCAB (for logits ops; may be 0 or valid
 *                                         for logprob-only ops)
 *   batch_size * seq_len * vocab_size <= ET_GRPO_MAX_LOGITS_ELEMS
 *   mask_norm_mode in {ET_GRPO_NORM_GLOBAL_TOKEN_MEAN,
 *                      ET_GRPO_NORM_PER_SEQ_MEAN}
 *   clip_eps_low finite, 0.0f < clip_eps_low <= 1.0f
 *   clip_eps_high finite, clip_eps_high > 0.0f
 *   kl_beta finite, kl_beta >= 0.0f
 *   reserved == 0u
 */
typedef struct et_grpo_loss_config {
  uint32_t batch_size;
  uint32_t seq_len;
  uint32_t vocab_size;
  uint32_t mask_norm_mode;
  float clip_eps_low;
  float clip_eps_high;
  float kl_beta;
  uint32_t reserved;
} et_grpo_loss_config;

/* Exact loss and diagnostic metrics produced by GRPO forward/backward ops. */
typedef struct et_grpo_loss_metrics {
  float loss;
  float pg_loss;
  float mean_kl;
  float clip_fraction;
  float mean_reward;
  uint32_t active_tokens;
  uint32_t clipped_tokens;
  uint32_t active_seqs;
} et_grpo_loss_metrics;

/* Configuration for structured <think>...</think><answer>...</answer> RLVR
 * verification. */
typedef struct et_grpo_verify_format_config {
  float format_weight;
  float answer_weight;
  uint32_t flags;
  uint32_t reserved;
} et_grpo_verify_format_config;

/* Detailed diagnostic output from et_grpo_verify_format_and_answer_v1. */
typedef struct et_grpo_verify_detail {
  uint32_t format_matched;
  uint32_t answer_matched;
  float total_reward;
  uint32_t reserved;
  uint64_t answer_offset;
  uint64_t answer_length;
} et_grpo_verify_detail;

#if defined(__cplusplus)
#define ET_GRPO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_GRPO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_GRPO_STATIC_ASSERT(sizeof(et_grpo_loss_config) == 32u &&
                          offsetof(et_grpo_loss_config, clip_eps_low) == 16u &&
                          offsetof(et_grpo_loss_config, reserved) == 28u,
                      "et_grpo_loss_config ABI layout changed");
ET_GRPO_STATIC_ASSERT(sizeof(et_grpo_loss_metrics) == 32u &&
                          offsetof(et_grpo_loss_metrics, mean_reward) == 16u &&
                          offsetof(et_grpo_loss_metrics, active_seqs) == 28u,
                      "et_grpo_loss_metrics ABI layout changed");
ET_GRPO_STATIC_ASSERT(sizeof(et_grpo_verify_format_config) == 16u &&
                          offsetof(et_grpo_verify_format_config, flags) == 8u &&
                          offsetof(et_grpo_verify_format_config, reserved) ==
                              12u,
                      "et_grpo_verify_format_config ABI layout changed");
ET_GRPO_STATIC_ASSERT(sizeof(et_grpo_verify_detail) == 32u &&
                          offsetof(et_grpo_verify_detail, answer_offset) ==
                              16u &&
                          offsetof(et_grpo_verify_detail, answer_length) == 24u,
                      "et_grpo_verify_detail ABI layout changed");
#undef ET_GRPO_STATIC_ASSERT

/* Validate GRPO logits-mode loss configuration and IEEE-754 FP environment. */
et_grpo_status et_grpo_validate_loss_config(const et_grpo_loss_config *cfg);

/* Group-relative advantage normalization over rewards[num_groups, group_size].
 * Requires 1 <= num_groups <= ET_GRPO_MAX_GROUPS,
 *          2 <= group_size <= ET_GRPO_MAX_GROUP_SIZE,
 *          num_groups * group_size <= ET_GRPO_MAX_BATCH.
 * Zero heap allocations; failure leaves out_advantages untouched. */
et_grpo_status et_grpo_group_advantages_v1(uint32_t num_groups,
                                           uint32_t group_size,
                                           const float *rewards,
                                           uint32_t adv_mode, float eps,
                                           float *out_advantages);
et_grpo_status et_grpo_group_advantages_v1_raw(uint32_t num_groups,
                                               uint32_t group_size,
                                               const void *rewards,
                                               uint32_t adv_mode, float eps,
                                               void *out_advantages);

/* Per-token and per-sequence log-probabilities via numerically stable
 * log-softmax over logits[N, T, V] with targets[N, T] in [0, V) and binary
 * response mask[N, T] in {0, 1}.
 * Masked tokens (mask == 0) write 0.0f to out_token_logprobs and contribute 0
 * to out_seq_logprobs. out_seq_logprobs may be NULL.
 * Zero heap allocations; failure leaves all outputs untouched. */
et_grpo_status et_grpo_token_logprobs_v1(uint32_t batch_size, uint32_t seq_len,
                                         uint32_t vocab_size,
                                         const float *logits,
                                         const int32_t *targets,
                                         const uint8_t *mask,
                                         float *out_token_logprobs,
                                         float *out_seq_logprobs);
et_grpo_status et_grpo_token_logprobs_v1_raw(uint32_t batch_size,
                                             uint32_t seq_len,
                                             uint32_t vocab_size,
                                             const void *logits,
                                             const void *targets,
                                             const uint8_t *mask,
                                             void *out_token_logprobs,
                                             void *out_seq_logprobs);

/* Exact analytic backward of et_grpo_token_logprobs_v1 w.r.t. logits[N, T, V]:
 *   dL/dz[n, t, v] = mask[n, t] * grad_token_logprobs[n, t]
 *                    * (1[v == targets[n, t]] - softmax(logits[n, t])[v]).
 * Zero heap allocations; failure leaves out_grad_logits untouched. */
et_grpo_status et_grpo_token_logprobs_backward_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const float *logits, const int32_t *targets, const uint8_t *mask,
    const float *grad_token_logprobs, float *out_grad_logits);
et_grpo_status et_grpo_token_logprobs_backward_v1_raw(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const void *logits, const void *targets, const uint8_t *mask,
    const void *grad_token_logprobs, void *out_grad_logits);

/* Log-probability-level GRPO clipped surrogate + Schulman k3 KL loss forward
 * and backward w.r.t. log_pi[N, T].
 *   rho     = exp(log_pi - log_old)
 *   surr1   = rho * A
 *   surr2   = clip(rho, 1 - clip_eps_low, 1 + clip_eps_high) * A
 *   pg_obj  = min(surr1, surr2)
 *   kl      = exp(log_ref - log_pi) - (log_ref - log_pi) - 1  (>= 0)
 *   L_t     = -pg_obj + kl_beta * kl
 * rewards[N] is optional (may be NULL -> mean_reward = 0.0f).
 * out_metrics in backward is optional (may be NULL).
 * Zero heap allocations; failure leaves all outputs untouched. */
et_grpo_status et_grpo_logprob_loss_forward_v1(
    const et_grpo_loss_config *cfg, const float *log_pi, const float *log_old,
    const float *log_ref, const uint8_t *mask, const float *advantages,
    const float *rewards, et_grpo_loss_metrics *out_metrics);

et_grpo_status et_grpo_logprob_loss_backward_v1(
    const et_grpo_loss_config *cfg, const float *log_pi, const float *log_old,
    const float *log_ref, const uint8_t *mask, const float *advantages,
    const float *rewards, float *out_grad_log_pi,
    et_grpo_loss_metrics *out_metrics);

/* Fused end-to-end GRPO loss forward over policy logits pi_logits[N, T, V].
 * rewards[N] is optional (may be NULL -> mean_reward = 0.0f).
 * Zero heap allocations; failure leaves out_metrics untouched. */
et_grpo_status et_grpo_loss_forward_v1(
    const et_grpo_loss_config *cfg, const float *pi_logits,
    const int32_t *targets, const float *log_old, const float *log_ref,
    const uint8_t *mask, const float *advantages, const float *rewards,
    et_grpo_loss_metrics *out_metrics);
et_grpo_status et_grpo_loss_forward_v1_raw(
    const et_grpo_loss_config *cfg, const void *pi_logits, const void *targets,
    const void *log_old, const void *log_ref, const uint8_t *mask,
    const void *advantages, const void *rewards, void *out_metrics);

/* Fused end-to-end GRPO analytic backward w.r.t. policy logits[N, T, V],
 * combining clipped surrogate subgradient, Schulman k3 KL derivative, and
 * log-softmax Jacobian.
 * rewards[N], out_grad_log_pi[N, T], and out_metrics are optional (nullable).
 * Zero heap allocations; failure leaves all outputs untouched. */
et_grpo_status et_grpo_loss_backward_v1(
    const et_grpo_loss_config *cfg, const float *pi_logits,
    const int32_t *targets, const float *log_old, const float *log_ref,
    const uint8_t *mask, const float *advantages, const float *rewards,
    float *out_grad_logits, float *out_grad_log_pi,
    et_grpo_loss_metrics *out_metrics);
et_grpo_status et_grpo_loss_backward_v1_raw(
    const et_grpo_loss_config *cfg, const void *pi_logits, const void *targets,
    const void *log_old, const void *log_ref, const uint8_t *mask,
    const void *advantages, const void *rewards, void *out_grad_logits,
    void *out_grad_log_pi, void *out_metrics);

/* Deterministic RLVR verifier: exact answer match (optional ASCII whitespace
 * trimming via ET_GRPO_VERIFY_TRIM_WS). Writes 1.0f on match, 0.0f otherwise.
 * Zero heap allocations; failure leaves out_reward untouched. */
et_grpo_status et_grpo_verify_exact_match_v1(const char *completion,
                                             size_t completion_len,
                                             const char *expected_answer,
                                             size_t expected_len,
                                             uint32_t flags, float *out_reward);

/* Deterministic RLVR verifier: strict <think>...</think><answer>...</answer>
 * format check + exact answer extraction and comparison.
 * out_detail may be NULL.
 * Zero heap allocations; failure leaves all outputs untouched. */
et_grpo_status et_grpo_verify_format_and_answer_v1(
    const char *completion, size_t completion_len, const char *expected_answer,
    size_t expected_len, const et_grpo_verify_format_config *cfg,
    float *out_reward, et_grpo_verify_detail *out_detail);

#ifdef __cplusplus
}
#endif

#endif
