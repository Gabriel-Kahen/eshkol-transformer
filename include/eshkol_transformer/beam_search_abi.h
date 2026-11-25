#ifndef ESHKOL_TRANSFORMER_BEAM_SEARCH_ABI_H
#define ESHKOL_TRANSFORMER_BEAM_SEARCH_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_BEAM_ABI_MAJOR 1u
#define ET_BEAM_ABI_MINOR 0u

#define ET_BEAM_MIN_BEAMS 1u
#define ET_BEAM_MAX_BEAMS 8u
#define ET_BEAM_MIN_VOCAB 2u
#define ET_BEAM_MAX_VOCAB 4096u
#define ET_BEAM_MAX_SEQ_LEN 256u
#define ET_BEAM_MAX_FINISHED (2u * ET_BEAM_MAX_BEAMS)
#define ET_BEAM_MIN_NGRAM 2u

typedef int32_t et_beam_status;

enum {
  ET_BEAM_OK = 0,
  ET_BEAM_ERR_INVALID_ARG = -1,
  ET_BEAM_ERR_NONFINITE = -2,
  ET_BEAM_ERR_NO_VALID_TOKENS = -3,
  ET_BEAM_ERR_ALIASING = -4,
  ET_BEAM_ERR_FP_ENV = -5,
  ET_BEAM_ERR_STATE = -6,
  ET_BEAM_ERR_LIMIT = -7
};

/*
 * Configuration for deterministic beam search decoding.
 *
 * Invariants:
 *   - struct_size == sizeof(et_beam_config_v1)
 *   - 1 <= num_beams <= ET_BEAM_MAX_BEAMS (8)
 *   - 2 <= vocab_size <= ET_BEAM_MAX_VOCAB (4096)
 *   - eos_token_id < vocab_size
 *   - max_new_tokens >= 1, prompt_len >= 1,
 *     prompt_len + max_new_tokens <= ET_BEAM_MAX_SEQ_LEN (256)
 *   - min_new_tokens <= max_new_tokens; EOS is masked until this many
 *     non-EOS tokens have already been generated. Equality prevents an EOS
 *     completion within the budget, leaving active beams for finalization.
 *   - no_repeat_ngram_size == 0 (disabled) or
 *     ET_BEAM_MIN_NGRAM (2) <= no_repeat_ngram_size <= ET_BEAM_MAX_SEQ_LEN
 *   - early_stopping in {0, 1}
 *   - reserved == 0
 *   - isfinite(length_penalty) && length_penalty >= 0.0 (Wu et al. 2016 alpha)
 *   - isfinite(repetition_penalty) && repetition_penalty >= 1.0 (Keskar/CTRL)
 */
typedef struct et_beam_config_v1 {
  size_t struct_size;
  uint32_t num_beams;
  uint32_t vocab_size;
  uint32_t eos_token_id;
  uint32_t max_new_tokens;
  uint32_t min_new_tokens;
  uint32_t no_repeat_ngram_size;
  uint32_t early_stopping;
  uint32_t reserved;
  double length_penalty;
  double repetition_penalty;
} et_beam_config_v1;

/*
 * Single beam hypothesis (prompt + generated tokens).
 *
 * Wu et al. (2016) length penalty score:
 *   score = sum_log_prob / (((5.0 + generated_len) / 6.0) ^ length_penalty)
 * where length_penalty == 0.0 yields exact raw sum_log_prob, and
 * length_penalty == 1.0 yields sum_log_prob / ((5.0 + generated_len) / 6.0).
 */
typedef struct et_beam_hypothesis_v1 {
  uint32_t seq_len;
  uint32_t generated_len;
  uint32_t finished;
  uint32_t reserved;
  double sum_log_prob;
  double score;
  uint32_t tokens[ET_BEAM_MAX_SEQ_LEN];
} et_beam_hypothesis_v1;

/*
 * Caller-owned zero-allocation beam search engine state.
 * All per-step logit processing, log-softmax, expansion, and finished-pool
 * updates operate strictly within this struct and small fixed stack frames
 * with zero heap allocations.
 */
typedef struct et_beam_state_v1 {
  uint64_t magic;
  et_beam_config_v1 config;
  uint32_t prompt_len;
  uint32_t steps_done;
  uint32_t active_count;
  uint32_t finished_count;
  uint32_t done;
  uint32_t stopped_early;
  double active_log_probs[ET_BEAM_MAX_BEAMS];
  uint32_t active_lens[ET_BEAM_MAX_BEAMS];
  uint32_t active_tokens[ET_BEAM_MAX_BEAMS][ET_BEAM_MAX_SEQ_LEN];
  et_beam_hypothesis_v1 finished_pool[ET_BEAM_MAX_FINISHED];
  double scratch_logits[ET_BEAM_MAX_VOCAB];
  double scratch_log_probs[ET_BEAM_MAX_VOCAB];
  double next_log_probs[ET_BEAM_MAX_BEAMS];
  uint32_t next_lens[ET_BEAM_MAX_BEAMS];
  uint32_t next_tokens[ET_BEAM_MAX_BEAMS][ET_BEAM_MAX_SEQ_LEN];
} et_beam_state_v1;

/* ---- Config Validation, Wu Scoring, and Log-Softmax ---- */

et_beam_status et_beam_validate_config_v1(const et_beam_config_v1 *cfg,
                                          size_t prompt_len);

et_beam_status et_beam_wu_length_penalty_v1(size_t generated_len,
                                            double length_penalty,
                                            double *out_penalty);

et_beam_status et_beam_score_v1(double sum_log_prob, size_t generated_len,
                                double length_penalty, double *out_score);

et_beam_status et_beam_log_softmax_v1(size_t vocab_size,
                                      const double *logits_in,
                                      double *log_probs_out);

/* ---- Zero-Allocation Composable Logit Processors ---- */

/*
 * Keskar/CTRL repetition penalty (penalty >= 1.0).
 * For every unique token ID t in history_tokens[0..history_len-1]:
 *   if finite logit[t] > 0.0 -> logit[t] / penalty
 *   else if finite           -> logit[t] * penalty
 * penalty == 1.0 is a bit-exact identity copy.
 * Existing -INFINITY entries are preserved; NaN and +INFINITY fail closed with
 * ET_BEAM_ERR_NONFINITE; all -INFINITY fails with ET_BEAM_ERR_NO_VALID_TOKENS.
 * Supports disjoint buffers or exact in-place (logits_in == logits_out).
 */
et_beam_status et_beam_apply_repetition_penalty_v1(
    const uint32_t *history_tokens, size_t history_len, size_t vocab_size,
    double penalty, const double *logits_in, double *logits_out);

/*
 * Exact no-repeat N-gram blocking.
 * If ngram_size == 0, performs validated bit-exact copy.
 * If ngram_size >= 2, masks to -INFINITY every token v such that the N-gram
 * (seq[seq_len - (N - 1) .. seq_len - 1], v) already occurred earlier in
 * seq[0 .. seq_len - 1].
 * Fails closed with ET_BEAM_ERR_NO_VALID_TOKENS if all tokens become masked.
 * Supports disjoint buffers or exact in-place (logits_in == logits_out).
 */
et_beam_status et_beam_apply_no_repeat_ngram_v1(const uint32_t *seq_tokens,
                                                size_t seq_len,
                                                size_t vocab_size,
                                                uint32_t ngram_size,
                                                const double *logits_in,
                                                double *logits_out);

/*
 * Min-length EOS suppression.
 * Masks eos_token_id to -INFINITY whenever generated_len < min_new_tokens.
 * Fails closed with ET_BEAM_ERR_NO_VALID_TOKENS if all tokens become masked.
 * Supports disjoint buffers or exact in-place (logits_in == logits_out).
 */
et_beam_status et_beam_apply_min_length_v1(size_t generated_len,
                                           size_t min_new_tokens,
                                           uint32_t eos_token_id,
                                           size_t vocab_size,
                                           const double *logits_in,
                                           double *logits_out);

/* ---- Beam Search Engine Lifecycle ---- */

et_beam_status et_beam_init_v1(et_beam_state_v1 *state,
                               const et_beam_config_v1 *cfg,
                               const uint32_t *prompt_tokens,
                               size_t prompt_len);

/*
 * Advance beam search by one decoding step.
 * logits has row-major shape [logits_rows, vocab_size] where logits_rows must
 * equal et_beam_active_count_v1(state).
 * Failure-atomic: on any error, committed state fields are preserved untouched.
 */
et_beam_status et_beam_step_v1(et_beam_state_v1 *state, const double *logits,
                               size_t logits_rows);

/*
 * Finalize a completed beam search (et_beam_is_done_v1(state) == 1) and copy
 * the top-ranked hypotheses in strict total order into out_hyps[0..*out_count-1].
 * Failure-atomic: on any error, out_hyps and *out_count are preserved untouched.
 */
et_beam_status et_beam_finalize_v1(const et_beam_state_v1 *state,
                                   et_beam_hypothesis_v1 *out_hyps,
                                   size_t max_hyps, size_t *out_count);

/* ---- State Inspection Queries ---- */

et_beam_status et_beam_get_active_beam_v1(const et_beam_state_v1 *state,
                                          size_t beam_idx, uint32_t *out_tokens,
                                          size_t max_tokens, size_t *out_len,
                                          double *out_log_prob);

uint32_t et_beam_active_count_v1(const et_beam_state_v1 *state);
uint32_t et_beam_finished_count_v1(const et_beam_state_v1 *state);
uint32_t et_beam_steps_done_v1(const et_beam_state_v1 *state);
uint32_t et_beam_is_done_v1(const et_beam_state_v1 *state);
uint32_t et_beam_stopped_early_v1(const et_beam_state_v1 *state);

#ifdef __cplusplus
}
#endif

#endif
