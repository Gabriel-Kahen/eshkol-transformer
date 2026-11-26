#ifndef ESHKOL_TRANSFORMER_SPEC_DECODE_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_SPEC_DECODE_ESHKOL_BRIDGE_H

#include "eshkol_transformer/spec_decode_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bridge-specific constants beyond et_spec_status. */
enum {
  ET_SPEC_AOT_ERR_NOMEM = -100,
  ET_SPEC_AOT_NO_EOS = -1
};

typedef struct et_spec_aot_context et_spec_aot_context;

/* Allocate a speculative-decoding AOT context for horizon K in [1, 16] and
 * vocabulary size V in [2, 4096]. Leaves *out_ctx unchanged on any failure.
 */
int32_t et_spec_aot_create(int64_t horizon, int64_t vocab_size,
                           et_spec_aot_context **out_ctx);

/* Release a context created by et_spec_aot_create. Safe on NULL. */
void et_spec_aot_destroy(et_spec_aot_context *ctx);

/* Buffer marshaling from compiled Eshkol f64/i64 values into owned aligned
 * native buffers. Out-of-range binary64 logits saturate to +/-INFINITY (NaN
 * preserved) and are explicitly rejected with ET_SPEC_ERR_NONFINITE at verify
 * or probability-conversion time.
 */
int32_t et_spec_aot_set_draft_token(et_spec_aot_context *ctx, int64_t pos,
                                    int64_t token_id);
int32_t et_spec_aot_set_draft_logit(et_spec_aot_context *ctx, int64_t row,
                                    int64_t vocab_idx, double value);
int32_t et_spec_aot_set_target_logit(et_spec_aot_context *ctx, int64_t row,
                                     int64_t vocab_idx, double value);

/* Execution entry points (zero heap allocations; failure-atomic).
 * Pass eos_token_id == ET_SPEC_AOT_NO_EOS (-1) to disable EOS early stop.
 */
int32_t et_spec_aot_verify_greedy(et_spec_aot_context *ctx,
                                  int64_t eos_token_id);

int32_t et_spec_aot_verify_sample(et_spec_aot_context *ctx, double temperature,
                                  int64_t top_k, double top_p,
                                  int64_t eos_token_id, int64_t seed,
                                  int64_t counter);

/* Compute canonical binary64 probabilities for draft row (source_is_target==0,
 * row in [0, K-1]) or target row (source_is_target==1, row in [0, K]).
 */
int32_t et_spec_aot_logits_to_probs(et_spec_aot_context *ctx,
                                    int64_t source_is_target, int64_t row,
                                    double temperature, int64_t top_k,
                                    double top_p);

/* Sample one token from a draft or target logit row using canonical sampling
 * and SplitMix64 counter RNG.
 */
int32_t et_spec_aot_sample_row(et_spec_aot_context *ctx,
                               int64_t source_is_target, int64_t row,
                               double temperature, int64_t top_k, double top_p,
                               int64_t seed, int64_t counter);

/* Geometry and telemetry accessors (return -1 or NAN on NULL / missing / OOB). */
int64_t et_spec_aot_horizon(const et_spec_aot_context *ctx);
int64_t et_spec_aot_vocab_size(const et_spec_aot_context *ctx);
int64_t et_spec_aot_accepted_count(const et_spec_aot_context *ctx);
int64_t et_spec_aot_emitted_count(const et_spec_aot_context *ctx);
int64_t et_spec_aot_eos_hit(const et_spec_aot_context *ctx);
int64_t et_spec_aot_rejected(const et_spec_aot_context *ctx);
int64_t et_spec_aot_rng_draws(const et_spec_aot_context *ctx);
int64_t et_spec_aot_next_counter(const et_spec_aot_context *ctx);
int64_t et_spec_aot_emitted_token(const et_spec_aot_context *ctx,
                                  int64_t index);
double et_spec_aot_prob(const et_spec_aot_context *ctx, int64_t vocab_idx);
int64_t et_spec_aot_sampled_token(const et_spec_aot_context *ctx);
int64_t et_spec_aot_sampled_next_counter(const et_spec_aot_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_SPEC_DECODE_ESHKOL_BRIDGE_H */
