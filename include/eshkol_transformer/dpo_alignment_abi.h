#ifndef ESHKOL_TRANSFORMER_DPO_ALIGNMENT_ABI_H
#define ESHKOL_TRANSFORMER_DPO_ALIGNMENT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_DPO_ABI_MAJOR 1u
#define ET_DPO_ABI_MINOR 0u

#define ET_DPO_MIN_B 1u
#define ET_DPO_MAX_B 8u
#define ET_DPO_MIN_T 1u
#define ET_DPO_MAX_T 64u
#define ET_DPO_MIN_V 2u
#define ET_DPO_MAX_V 256u

#define ET_DPO_SHA256_BYTES 32u

typedef int32_t et_dpo_status;

enum {
  ET_DPO_OK = 0,
  ET_DPO_ERR_ARGUMENT = -1,
  ET_DPO_ERR_SHAPE = -2,
  ET_DPO_ERR_ALIGNMENT = -3,
  ET_DPO_ERR_ALIAS = -4,
  ET_DPO_ERR_NONFINITE = -5,
  ET_DPO_ERR_FP_ENV = -6,
  ET_DPO_ERR_MASK = -7,
  ET_DPO_ERR_TARGET = -8,
  ET_DPO_ERR_OVERFLOW = -9
};

/* Direct Preference Optimization (DPO / conservative DPO / reference-free)
 * configuration.
 *
 * Domain constraints:
 *   batch_size     in [1, ET_DPO_MAX_B (8)]
 *   seq_len        in [1, ET_DPO_MAX_T (64)]
 *   vocab_size     in [ET_DPO_MIN_V (2), ET_DPO_MAX_V (256)]
 *   reference_free in {0, 1}
 *                    0 = standard DPO with strictly frozen reference logits
 *                    1 = reference-free / SimPO-style margin (ref logps = 0)
 *   beta           finite binary64, beta > 0.0
 *   label_smoothing finite binary64, 0.0 <= label_smoothing < 0.5
 *
 * Tensor schemas (dense row-major, zero offset, naturally aligned):
 *   policy_chosen_logits, policy_rejected_logits : const float[B, T, V]
 *   ref_chosen_logits, ref_rejected_logits       : const float[B, T, V]
 *     (required non-NULL & finite when reference_free == 0; may be NULL when
 *      reference_free == 1)
 *   chosen_targets, rejected_targets             : const int64_t[B, T]
 *     (every element in [0, V - 1])
 *   chosen_mask, rejected_mask                   : const uint8_t[B, T]
 *     (every element 0 or 1; at least one 1 per batch sequence)
 *   d_logits_chosen, d_logits_rejected           : float[B, T, V]
 *     (exact +0.0f wherever mask[b, t] == 0)
 */
typedef struct et_dpo_config {
  uint32_t batch_size;
  uint32_t seq_len;
  uint32_t vocab_size;
  uint32_t reference_free;
  double beta;
  double label_smoothing;
} et_dpo_config;

/* Batch-averaged DPO reward-margin and alignment telemetry (binary64). */
typedef struct et_dpo_telemetry {
  double mean_loss;
  double chosen_reward_mean;
  double rejected_reward_mean;
  double reward_margin_mean;
  double preference_accuracy;
} et_dpo_telemetry;

#if defined(__cplusplus)
#define ET_DPO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_DPO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_DPO_STATIC_ASSERT(sizeof(et_dpo_config) == 32u &&
                         offsetof(et_dpo_config, batch_size) == 0u &&
                         offsetof(et_dpo_config, seq_len) == 4u &&
                         offsetof(et_dpo_config, vocab_size) == 8u &&
                         offsetof(et_dpo_config, reference_free) == 12u &&
                         offsetof(et_dpo_config, beta) == 16u &&
                         offsetof(et_dpo_config, label_smoothing) == 24u,
                     "et_dpo_config ABI layout changed");
ET_DPO_STATIC_ASSERT(sizeof(et_dpo_telemetry) == 40u &&
                         offsetof(et_dpo_telemetry, mean_loss) == 0u &&
                         offsetof(et_dpo_telemetry, chosen_reward_mean) == 8u &&
                         offsetof(et_dpo_telemetry, rejected_reward_mean) ==
                             16u &&
                         offsetof(et_dpo_telemetry, reward_margin_mean) ==
                             24u &&
                         offsetof(et_dpo_telemetry, preference_accuracy) == 32u,
                     "et_dpo_telemetry ABI layout changed");
#undef ET_DPO_STATIC_ASSERT

/* Validate configuration bounds, hyperparameters, and FP environment. */
et_dpo_status et_dpo_validate_config(const et_dpo_config *cfg);

/* Compute masked sequence log-probabilities in binary64 via stable log-sum-exp:
 *   out_logps[b] = sum_{t: mask[b, t] == 1} log_softmax(logits[b, t])[target[b, t]]
 * Failure leaves out_logps untouched. */
et_dpo_status et_dpo_sequence_logps(uint32_t batch_size, uint32_t seq_len,
                                    uint32_t vocab_size, const float *logits,
                                    const int64_t *targets, const uint8_t *mask,
                                    double *out_logps);
et_dpo_status et_dpo_sequence_logps_raw(uint32_t batch_size, uint32_t seq_len,
                                        uint32_t vocab_size, const void *logits,
                                        const void *targets, const void *mask,
                                        void *out_logps);

/* Forward-only DPO / cDPO loss and reward-margin telemetry.
 * Failure leaves out_telemetry untouched. */
et_dpo_status et_dpo_forward(const et_dpo_config *cfg,
                             const float *policy_chosen_logits,
                             const float *policy_rejected_logits,
                             const float *ref_chosen_logits,
                             const float *ref_rejected_logits,
                             const int64_t *chosen_targets,
                             const int64_t *rejected_targets,
                             const uint8_t *chosen_mask,
                             const uint8_t *rejected_mask,
                             et_dpo_telemetry *out_telemetry);
et_dpo_status et_dpo_forward_raw(const void *cfg,
                                 const void *policy_chosen_logits,
                                 const void *policy_rejected_logits,
                                 const void *ref_chosen_logits,
                                 const void *ref_rejected_logits,
                                 const void *chosen_targets,
                                 const void *rejected_targets,
                                 const void *chosen_mask,
                                 const void *rejected_mask,
                                 void *out_telemetry);

/* Fused DPO / cDPO forward telemetry + exact analytic backward.
 * Computes d_logits_chosen [B, T, V] and d_logits_rejected [B, T, V] with exact
 * +0.0f on all masked positions (mask[b, t] == 0). Reference logits are strictly
 * read-only. All non-NULL buffers must be pairwise disjoint and naturally
 * aligned. Failure leaves out_telemetry, d_logits_chosen, and d_logits_rejected
 * byte-for-byte untouched. */
et_dpo_status et_dpo_forward_backward(
    const et_dpo_config *cfg, const float *policy_chosen_logits,
    const float *policy_rejected_logits, const float *ref_chosen_logits,
    const float *ref_rejected_logits, const int64_t *chosen_targets,
    const int64_t *rejected_targets, const uint8_t *chosen_mask,
    const uint8_t *rejected_mask, et_dpo_telemetry *out_telemetry,
    float *d_logits_chosen, float *d_logits_rejected);
et_dpo_status et_dpo_forward_backward_raw(
    const void *cfg, const void *policy_chosen_logits,
    const void *policy_rejected_logits, const void *ref_chosen_logits,
    const void *ref_rejected_logits, const void *chosen_targets,
    const void *rejected_targets, const void *chosen_mask,
    const void *rejected_mask, void *out_telemetry, void *d_logits_chosen,
    void *d_logits_rejected);

/* Canonical SHA-256 digest of a finite little-endian binary32 weight buffer.
 * Useful for proving frozen reference model parameters stay bit-identical. */
et_dpo_status et_dpo_weight_sha256(size_t float_count, const float *weights,
                                   uint8_t out_digest[ET_DPO_SHA256_BYTES]);
et_dpo_status et_dpo_weight_sha256_raw(size_t float_count, const void *weights,
                                       void *out_digest);

#ifdef __cplusplus
}
#endif

#endif
