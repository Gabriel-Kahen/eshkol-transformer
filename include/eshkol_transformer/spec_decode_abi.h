#ifndef ESHKOL_TRANSFORMER_SPEC_DECODE_ABI_H
#define ESHKOL_TRANSFORMER_SPEC_DECODE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_SPEC_ABI_VERSION_MAJOR 1u
#define ET_SPEC_ABI_VERSION_MINOR 0u

#define ET_SPEC_MIN_HORIZON 1u
#define ET_SPEC_MAX_HORIZON 16u
#define ET_SPEC_MIN_VOCAB 2u
#define ET_SPEC_MAX_VOCAB 4096u
#define ET_SPEC_NO_EOS UINT32_MAX

typedef int32_t et_spec_status;

enum {
  ET_SPEC_OK = 0,
  ET_SPEC_ERR_NULL = -1,
  ET_SPEC_ERR_ALIAS = -2,
  ET_SPEC_ERR_BOUNDS = -3,
  ET_SPEC_ERR_PARAM = -4,
  ET_SPEC_ERR_NONFINITE = -5,
  ET_SPEC_ERR_TOKEN_RANGE = -6,
  ET_SPEC_ERR_FP_ENV = -7,
  ET_SPEC_ERR_ALIGNMENT = -8
};

/* Sampling parameters for canonical binary64 logits-to-probability conversion
 * and exact speculative rejection-sampling verification.
 *
 * - temperature: finite >= 0.0 (positive zero selects greedy one-hot;
 *   strictly positive values scale logits in binary64 before log-sum-exp).
 *   Negative values and -0.0 are rejected.
 * - top_k: exact integer in [1, vocab_size].
 * - top_p: finite double in (0.0, 1.0].
 * - eos_token_id: token index in [0, vocab_size - 1], or ET_SPEC_NO_EOS to
 *   disable EOS early termination.
 */
typedef struct et_spec_sampling_params_v1 {
  double temperature;
  double top_p;
  uint32_t top_k;
  uint32_t eos_token_id;
} et_spec_sampling_params_v1;

/* Explicit deterministic counter-based RNG state (SplitMix64).
 * Draw at counter c maps (seed + c * 0x9e3779b97f4a7c15ULL) through SplitMix64
 * to a 53-bit IEEE-754 binary64 uniform variate in [0.0, 1.0).
 */
typedef struct et_spec_rng_state_v1 {
  uint64_t seed;
  uint64_t counter;
} et_spec_rng_state_v1;

/* Result metadata for greedy and stochastic speculative verification.
 * - accepted_count: number of accepted draft tokens in [0, horizon].
 * - emitted_count: total emitted tokens written to out_tokens in [1, horizon+1].
 * - eos_hit: 1 if an emitted token matched eos_token_id, else 0.
 * - rejected: 1 if a draft token was rejected, else 0.
 * - rng_draws: exact count of SplitMix64 uniform draws consumed (0 for greedy).
 * - next_rng: updated RNG state (zeroed for greedy).
 */
typedef struct et_spec_verify_result_v1 {
  uint32_t accepted_count;
  uint32_t emitted_count;
  uint32_t eos_hit;
  uint32_t rejected;
  uint64_t rng_draws;
  et_spec_rng_state_v1 next_rng;
} et_spec_verify_result_v1;

/* All pointer arguments in every ABI call must be non-NULL, properly aligned
 * for their element/struct type, and pairwise disjoint in memory.
 * On any error status (!= ET_SPEC_OK), all output buffers remain byte-for-byte
 * unchanged. Every entry point requires a standard round-to-nearest FP
 * environment with masked exceptions and gradual underflow (x87 + MXCSR on
 * x86-64).
 */

/* Canonical binary64 log-sum-exp logits-to-probability conversion with
 * temperature scaling, top-k renormalization, shortest top-p nucleus prefix,
 * and deterministic lowest-index tie-breaking.
 */
et_spec_status et_spec_logits_to_probs_v1(
    const float *logits, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params, double *probs_out);

/* Canonical single-token sampler using et_spec_logits_to_probs_v1 and
 * SplitMix64 counter RNG (0 draws when temperature == 0.0, 1 draw when > 0.0).
 */
et_spec_status et_spec_sample_token_v1(
    const float *logits, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params,
    const et_spec_rng_state_v1 *rng_in, uint32_t *out_token,
    et_spec_rng_state_v1 *out_rng);

/* Exact lossless greedy speculative verification.
 * - draft_tokens: length K (horizon in 1..16), each < vocab_size.
 * - target_logits: contiguous row-major [K + 1, V] finite binary32 logits.
 * - out_tokens: caller buffer with out_capacity >= K + 1.
 * Accepts the longest prefix matching target argmax (lowest-index tie-break)
 * and emits 1 correction or bonus token, stopping immediately on eos_token_id.
 */
et_spec_status et_spec_verify_greedy_v1(
    const uint32_t *draft_tokens, const float *target_logits, uint32_t horizon,
    uint32_t vocab_size, uint32_t eos_token_id, uint32_t *out_tokens,
    uint32_t out_capacity, et_spec_verify_result_v1 *out_result);

/* Exact lossless stochastic speculative rejection-sampling verification
 * (Leviathan et al. 2023 / Chen et al. 2023).
 * - draft_tokens: length K (horizon in 1..16), each < vocab_size.
 * - draft_logits: contiguous row-major [K, V] finite binary32 logits.
 * - target_logits: contiguous row-major [K + 1, V] finite binary32 logits.
 * - out_tokens: caller buffer with out_capacity >= K + 1.
 * Computes exact filtered q_i and p_i in binary64, accepts x_i with exact
 * probability min(1, p_i(x_i) / q_i(x_i)) without division by zero, samples
 * from normalized residual max(0, p_i - q_i) on first rejection, or samples a
 * bonus token from p_K when all K draft tokens are accepted, with exact EOS
 * early termination.
 */
et_spec_status et_spec_verify_sample_v1(
    const uint32_t *draft_tokens, const float *draft_logits,
    const float *target_logits, uint32_t horizon, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params,
    const et_spec_rng_state_v1 *rng_in, uint32_t *out_tokens,
    uint32_t out_capacity, et_spec_verify_result_v1 *out_result);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_SPEC_DECODE_ABI_H */
