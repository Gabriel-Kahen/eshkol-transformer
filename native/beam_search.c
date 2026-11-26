#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/beam_search_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Beam search engine requires 8-bit bytes and IEEE-754 binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "Beam search engine requires IEEE-754 binary32/binary64");

#define ET_BEAM_STATE_MAGIC UINT64_C(0x45544245414D5631) /* "ETBEAMV1" */
#define ET_BEAM_VOCAB_WORDS ((ET_BEAM_MAX_VOCAB + 63u) / 64u)

/* ---- Internal FP Environment, Alignment, Aliasing, and Bitset Helpers ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t la = (uintptr_t)a;
  const uintptr_t lb = (uintptr_t)b;
  if (la > UINTPTR_MAX - a_bytes || lb > UINTPTR_MAX - b_bytes)
    return 1;
  return la < lb + b_bytes && lb < la + a_bytes;
}

static int spans_partially_overlap(const void *a, const void *b, size_t bytes) {
  if (a == b || bytes == 0u)
    return 0;
  return spans_overlap(a, bytes, b, bytes);
}

static int bitset_get(const uint64_t *bits, uint32_t idx) {
  return (int)((bits[idx >> 6u] >> (idx & 63u)) & UINT64_C(1));
}

static void bitset_set(uint64_t *bits, uint32_t idx) {
  bits[idx >> 6u] |= (UINT64_C(1) << (idx & 63u));
}

static int is_pos_inf_or_nan(double x) {
  return isnan(x) || (isinf(x) && x > 0.0);
}

static et_beam_status validate_vocab_size(size_t vocab_size) {
  if (vocab_size > ET_BEAM_MAX_VOCAB)
    return ET_BEAM_ERR_LIMIT;
  if (vocab_size < ET_BEAM_MIN_VOCAB)
    return ET_BEAM_ERR_INVALID_ARG;
  return ET_BEAM_OK;
}

/* ---- Wu et al. (2016) Length Penalty and Scoring ---- */

static et_beam_status compute_wu_penalty_unchecked(size_t generated_len,
                                                   double length_penalty,
                                                   double *out_penalty) {
  if (!isfinite(length_penalty))
    return ET_BEAM_ERR_NONFINITE;
  if (length_penalty < 0.0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (generated_len > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;

  double denom = 1.0;
  if (length_penalty == 0.0) {
    denom = 1.0;
  } else if (length_penalty == 1.0) {
    denom = (5.0 + (double)generated_len) / 6.0;
  } else {
    const double base = (5.0 + (double)generated_len) / 6.0;
    denom = pow(base, length_penalty);
  }
  if (!isfinite(denom) || denom <= 0.0)
    return ET_BEAM_ERR_NONFINITE;
  *out_penalty = denom;
  return ET_BEAM_OK;
}

static et_beam_status compute_wu_score_unchecked(double sum_log_prob,
                                                 size_t generated_len,
                                                 double length_penalty,
                                                 double *out_score) {
  if (!isfinite(sum_log_prob) || !isfinite(length_penalty))
    return ET_BEAM_ERR_NONFINITE;
  if (length_penalty < 0.0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (generated_len > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;

  if (length_penalty == 0.0) {
    *out_score = sum_log_prob;
    return ET_BEAM_OK;
  }
  double denom = 1.0;
  const et_beam_status st =
      compute_wu_penalty_unchecked(generated_len, length_penalty, &denom);
  if (st != ET_BEAM_OK)
    return st;
  const double score = sum_log_prob / denom;
  if (!isfinite(score))
    return ET_BEAM_ERR_NONFINITE;
  *out_score = score;
  return ET_BEAM_OK;
}

/* ---- Config and State Validation ---- */

static et_beam_status validate_config_internal(const et_beam_config_v1 *cfg,
                                               size_t prompt_len) {
  if (!aligned_span(cfg, alignof(et_beam_config_v1), sizeof(et_beam_config_v1)))
    return ET_BEAM_ERR_INVALID_ARG;
  if (cfg->struct_size != sizeof(et_beam_config_v1))
    return ET_BEAM_ERR_INVALID_ARG;
  if (cfg->reserved != 0u)
    return ET_BEAM_ERR_INVALID_ARG;
  if (cfg->num_beams > ET_BEAM_MAX_BEAMS)
    return ET_BEAM_ERR_LIMIT;
  if (cfg->num_beams < ET_BEAM_MIN_BEAMS)
    return ET_BEAM_ERR_INVALID_ARG;

  const et_beam_status vst = validate_vocab_size((size_t)cfg->vocab_size);
  if (vst != ET_BEAM_OK)
    return vst;
  if (cfg->eos_token_id >= cfg->vocab_size)
    return ET_BEAM_ERR_INVALID_ARG;

  if (prompt_len == 0u || cfg->max_new_tokens == 0u)
    return ET_BEAM_ERR_INVALID_ARG;
  if (prompt_len > ET_BEAM_MAX_SEQ_LEN ||
      cfg->max_new_tokens > ET_BEAM_MAX_SEQ_LEN ||
      prompt_len > ET_BEAM_MAX_SEQ_LEN - (size_t)cfg->max_new_tokens)
    return ET_BEAM_ERR_LIMIT;

  if (cfg->min_new_tokens > cfg->max_new_tokens)
    return ET_BEAM_ERR_INVALID_ARG;

  if (cfg->no_repeat_ngram_size == 1u)
    return ET_BEAM_ERR_INVALID_ARG;
  if (cfg->no_repeat_ngram_size > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;

  if (cfg->early_stopping > 1u)
    return ET_BEAM_ERR_INVALID_ARG;

  if (!isfinite(cfg->length_penalty) || !isfinite(cfg->repetition_penalty))
    return ET_BEAM_ERR_NONFINITE;
  if (cfg->length_penalty < 0.0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (cfg->repetition_penalty < 1.0)
    return ET_BEAM_ERR_INVALID_ARG;

  /* Verify Wu denominator at max_new_tokens is finite and positive. */
  double max_denom = 1.0;
  const et_beam_status pst = compute_wu_penalty_unchecked(
      (size_t)cfg->max_new_tokens, cfg->length_penalty, &max_denom);
  if (pst != ET_BEAM_OK)
    return pst;

  return ET_BEAM_OK;
}

static int state_is_valid(const et_beam_state_v1 *state) {
  if (!aligned_span(state, alignof(et_beam_state_v1), sizeof(et_beam_state_v1)))
    return 0;
  if (state->magic != ET_BEAM_STATE_MAGIC)
    return 0;
  if (validate_config_internal(&state->config, (size_t)state->prompt_len) !=
      ET_BEAM_OK)
    return 0;
  if (state->steps_done > state->config.max_new_tokens)
    return 0;
  if (state->active_count > state->config.num_beams)
    return 0;
  if (state->finished_count > 2u * state->config.num_beams)
    return 0;
  if (state->done > 1u || state->stopped_early > 1u)
    return 0;
  if (state->stopped_early != 0u && state->done == 0u)
    return 0;
  if (state->done == 0u && (state->active_count == 0u ||
                            state->steps_done >= state->config.max_new_tokens))
    return 0;
  const uint32_t active_len = state->prompt_len + state->steps_done;
  for (uint32_t b = 0u; b < state->active_count; ++b) {
    if (state->active_lens[b] != active_len ||
        !isfinite(state->active_log_probs[b]) ||
        state->active_log_probs[b] > 0.0)
      return 0;
    for (uint32_t i = 0u; i < active_len; ++i) {
      if (state->active_tokens[b][i] >= state->config.vocab_size)
        return 0;
    }
  }
  for (uint32_t b = 0u; b < state->finished_count; ++b) {
    const et_beam_hypothesis_v1 *hyp = &state->finished_pool[b];
    if (hyp->generated_len == 0u || hyp->generated_len > state->steps_done ||
        hyp->seq_len != state->prompt_len + hyp->generated_len ||
        hyp->finished != 1u || hyp->reserved != 0u ||
        !isfinite(hyp->sum_log_prob) || hyp->sum_log_prob > 0.0 ||
        !isfinite(hyp->score) || hyp->score > 0.0)
      return 0;
    for (uint32_t i = 0u; i < hyp->seq_len; ++i) {
      if (hyp->tokens[i] >= state->config.vocab_size)
        return 0;
    }
  }
  return 1;
}

/* ---- Deterministic Total-Order Comparators ---- */

typedef struct beam_candidate {
  double log_prob;
  uint32_t parent_idx;
  uint32_t token_id;
} beam_candidate;

/* Strict total order: higher log_prob -> lower parent_idx -> lower token_id. */
static int candidate_is_better(const beam_candidate *a,
                               const beam_candidate *b) {
  if (a->log_prob != b->log_prob)
    return a->log_prob > b->log_prob;
  if (a->parent_idx != b->parent_idx)
    return a->parent_idx < b->parent_idx;
  return a->token_id < b->token_id;
}

/* Strict total order on hypotheses:
 *   1. higher Wu length-penalized score
 *   2. higher raw sum_log_prob
 *   3. shorter seq_len
 *   4. lexicographically smaller token sequence
 */
static int hypothesis_is_better_parts(double score_a, double logp_a,
                                      uint32_t len_a, const uint32_t *toks_a,
                                      double score_b, double logp_b,
                                      uint32_t len_b, const uint32_t *toks_b) {
  if (score_a != score_b)
    return score_a > score_b;
  if (logp_a != logp_b)
    return logp_a > logp_b;
  if (len_a != len_b)
    return len_a < len_b;
  for (uint32_t i = 0u; i < len_a; ++i) {
    if (toks_a[i] != toks_b[i])
      return toks_a[i] < toks_b[i];
  }
  return 0;
}

static int hypothesis_is_better(const et_beam_hypothesis_v1 *a,
                                const et_beam_hypothesis_v1 *b) {
  return hypothesis_is_better_parts(a->score, a->sum_log_prob, a->seq_len,
                                    a->tokens, b->score, b->sum_log_prob,
                                    b->seq_len, b->tokens);
}

/* Insert candidate into sorted top-K array (capacity K <= 2 * ET_BEAM_MAX_BEAMS). */
static void insert_top_candidate(beam_candidate *top, size_t *count,
                                 size_t capacity, const beam_candidate *cand) {
  size_t n = *count;
  if (n == capacity && !candidate_is_better(cand, &top[n - 1u]))
    return;
  size_t pos = n;
  if (pos == capacity)
    pos = capacity - 1u;
  else
    *count = n + 1u;

  while (pos > 0u && candidate_is_better(cand, &top[pos - 1u])) {
    top[pos] = top[pos - 1u];
    pos -= 1u;
  }
  top[pos] = *cand;
}

/* Insert finished hypothesis into sorted finished pool (capacity = 2 * B). */
static void insert_finished_hypothesis(et_beam_state_v1 *state,
                                       const et_beam_hypothesis_v1 *hyp) {
  const uint32_t cap = 2u * state->config.num_beams;
  uint32_t n = state->finished_count;
  if (n == cap && !hypothesis_is_better(hyp, &state->finished_pool[n - 1u]))
    return;

  uint32_t pos = n;
  if (pos == cap)
    pos = cap - 1u;
  else
    state->finished_count = n + 1u;

  while (pos > 0u &&
         hypothesis_is_better(hyp, &state->finished_pool[pos - 1u])) {
    state->finished_pool[pos] = state->finished_pool[pos - 1u];
    pos -= 1u;
  }
  state->finished_pool[pos] = *hyp;
}

/* ---- Internal Log-Softmax & Logit Processors (Shared by Public + Step) ---- */

static et_beam_status log_softmax_internal(size_t vocab_size,
                                           const double *logits_in,
                                           double *log_probs_out) {
  double max_val = -INFINITY;
  size_t valid_count = 0u;
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (is_pos_inf_or_nan(x))
      return ET_BEAM_ERR_NONFINITE;
    if (isfinite(x)) {
      if (valid_count == 0u || x > max_val)
        max_val = x;
      valid_count += 1u;
    }
  }
  if (valid_count == 0u)
    return ET_BEAM_ERR_NO_VALID_TOKENS;

  double sum_exp = 0.0;
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (isfinite(x)) {
      const double diff = x - max_val;
      if (!isfinite(diff))
        return ET_BEAM_ERR_NONFINITE;
      sum_exp += exp(diff);
    }
  }
  if (!isfinite(sum_exp) || sum_exp <= 0.0)
    return ET_BEAM_ERR_NONFINITE;

  const double log_sum_exp = log(sum_exp);
  if (!isfinite(log_sum_exp))
    return ET_BEAM_ERR_NONFINITE;

  /* Pre-check all resulting finite log-probs before mutating output. */
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (isfinite(x)) {
      const double lp = (x - max_val) - log_sum_exp;
      if (!isfinite(lp))
        return ET_BEAM_ERR_NONFINITE;
    }
  }

  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (isfinite(x)) {
      log_probs_out[v] = (x - max_val) - log_sum_exp;
    } else {
      log_probs_out[v] = -INFINITY;
    }
  }
  return ET_BEAM_OK;
}

static et_beam_status apply_rep_penalty_internal(const uint32_t *history_tokens,
                                                 size_t history_len,
                                                 size_t vocab_size,
                                                 double penalty,
                                                 const double *logits_in,
                                                 double *logits_out) {
  if (!isfinite(penalty))
    return ET_BEAM_ERR_NONFINITE;
  if (penalty < 1.0)
    return ET_BEAM_ERR_INVALID_ARG;

  uint64_t seen[ET_BEAM_VOCAB_WORDS];
  const size_t seen_words = (vocab_size + 63u) / 64u;
  memset(seen, 0, seen_words * sizeof(uint64_t));

  for (size_t i = 0u; i < history_len; ++i) {
    const uint32_t tok = history_tokens[i];
    if ((size_t)tok >= vocab_size)
      return ET_BEAM_ERR_INVALID_ARG;
    bitset_set(seen, tok);
  }

  /* Pass 1: validate finiteness, non-empty valid set, and no overflow. */
  size_t valid_count = 0u;
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (is_pos_inf_or_nan(x))
      return ET_BEAM_ERR_NONFINITE;
    if (isfinite(x)) {
      valid_count += 1u;
      if (penalty > 1.0 && bitset_get(seen, (uint32_t)v)) {
        const double scaled = (x > 0.0) ? (x / penalty) : (x * penalty);
        if (!isfinite(scaled))
          return ET_BEAM_ERR_NONFINITE;
      }
    }
  }
  if (valid_count == 0u)
    return ET_BEAM_ERR_NO_VALID_TOKENS;

  /* Pass 2: commit output. penalty == 1.0 is bit-exact identity. */
  if (penalty == 1.0) {
    if (logits_out != logits_in)
      memcpy(logits_out, logits_in, vocab_size * sizeof(double));
    return ET_BEAM_OK;
  }

  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (isfinite(x) && bitset_get(seen, (uint32_t)v)) {
      logits_out[v] = (x > 0.0) ? (x / penalty) : (x * penalty);
    } else {
      logits_out[v] = x;
    }
  }
  return ET_BEAM_OK;
}

static et_beam_status apply_no_repeat_ngram_internal(const uint32_t *seq_tokens,
                                                     size_t seq_len,
                                                     size_t vocab_size,
                                                     uint32_t ngram_size,
                                                     const double *logits_in,
                                                     double *logits_out) {
  if (ngram_size == 1u)
    return ET_BEAM_ERR_INVALID_ARG;
  if (ngram_size > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;

  for (size_t i = 0u; i < seq_len; ++i) {
    if ((size_t)seq_tokens[i] >= vocab_size)
      return ET_BEAM_ERR_INVALID_ARG;
  }

  uint64_t banned[ET_BEAM_VOCAB_WORDS];
  const size_t mask_words = (vocab_size + 63u) / 64u;
  memset(banned, 0, mask_words * sizeof(uint64_t));

  if (ngram_size >= ET_BEAM_MIN_NGRAM && seq_len >= (size_t)ngram_size) {
    const size_t n = (size_t)ngram_size;
    const size_t prefix_len = n - 1u;
    const uint32_t *target_prefix = seq_tokens + (seq_len - prefix_len);
    for (size_t start = 0u; start + n <= seq_len; ++start) {
      int match = 1;
      for (size_t k = 0u; k < prefix_len; ++k) {
        if (seq_tokens[start + k] != target_prefix[k]) {
          match = 0;
          break;
        }
      }
      if (match) {
        bitset_set(banned, seq_tokens[start + prefix_len]);
      }
    }
  }

  /* Pass 1: validate finiteness and ensure at least one unbanned finite token. */
  size_t valid_count = 0u;
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (is_pos_inf_or_nan(x))
      return ET_BEAM_ERR_NONFINITE;
    if (isfinite(x) && !bitset_get(banned, (uint32_t)v))
      valid_count += 1u;
  }
  if (valid_count == 0u)
    return ET_BEAM_ERR_NO_VALID_TOKENS;

  /* Pass 2: commit output bit-for-bit for unbanned, -INFINITY for banned. */
  for (size_t v = 0u; v < vocab_size; ++v) {
    if (bitset_get(banned, (uint32_t)v))
      logits_out[v] = -INFINITY;
    else
      logits_out[v] = logits_in[v];
  }
  return ET_BEAM_OK;
}

static et_beam_status apply_min_length_internal(size_t generated_len,
                                                size_t min_new_tokens,
                                                uint32_t eos_token_id,
                                                size_t vocab_size,
                                                const double *logits_in,
                                                double *logits_out) {
  if ((size_t)eos_token_id >= vocab_size)
    return ET_BEAM_ERR_INVALID_ARG;
  if (min_new_tokens > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;

  const int suppress_eos = (generated_len < min_new_tokens);

  /* Pass 1: validate finiteness and non-empty valid set after suppression. */
  size_t valid_count = 0u;
  for (size_t v = 0u; v < vocab_size; ++v) {
    const double x = logits_in[v];
    if (is_pos_inf_or_nan(x))
      return ET_BEAM_ERR_NONFINITE;
    if (isfinite(x) && !(suppress_eos && v == (size_t)eos_token_id))
      valid_count += 1u;
  }
  if (valid_count == 0u)
    return ET_BEAM_ERR_NO_VALID_TOKENS;

  /* Pass 2: commit output. */
  if (logits_out != logits_in)
    memcpy(logits_out, logits_in, vocab_size * sizeof(double));
  if (suppress_eos)
    logits_out[eos_token_id] = -INFINITY;
  return ET_BEAM_OK;
}

/* ---- Public API Implementation ---- */

et_beam_status et_beam_validate_config_v1(const et_beam_config_v1 *cfg,
                                          size_t prompt_len) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  return validate_config_internal(cfg, prompt_len);
}

et_beam_status et_beam_wu_length_penalty_v1(size_t generated_len,
                                            double length_penalty,
                                            double *out_penalty) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!aligned_span(out_penalty, alignof(double), sizeof(double)))
    return ET_BEAM_ERR_INVALID_ARG;

  double penalty = 1.0;
  const et_beam_status st =
      compute_wu_penalty_unchecked(generated_len, length_penalty, &penalty);
  if (st != ET_BEAM_OK)
    return st;
  *out_penalty = penalty;
  return ET_BEAM_OK;
}

et_beam_status et_beam_score_v1(double sum_log_prob, size_t generated_len,
                                double length_penalty, double *out_score) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!aligned_span(out_score, alignof(double), sizeof(double)))
    return ET_BEAM_ERR_INVALID_ARG;

  double score = 0.0;
  const et_beam_status st = compute_wu_score_unchecked(
      sum_log_prob, generated_len, length_penalty, &score);
  if (st != ET_BEAM_OK)
    return st;
  *out_score = score;
  return ET_BEAM_OK;
}

et_beam_status et_beam_log_softmax_v1(size_t vocab_size,
                                      const double *logits_in,
                                      double *log_probs_out) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  const et_beam_status vst = validate_vocab_size(vocab_size);
  if (vst != ET_BEAM_OK)
    return vst;
  const size_t bytes = vocab_size * sizeof(double);
  if (!aligned_span(logits_in, alignof(double), bytes) ||
      !aligned_span(log_probs_out, alignof(double), bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_partially_overlap(logits_in, log_probs_out, bytes))
    return ET_BEAM_ERR_ALIASING;
  return log_softmax_internal(vocab_size, logits_in, log_probs_out);
}

et_beam_status et_beam_apply_repetition_penalty_v1(
    const uint32_t *history_tokens, size_t history_len, size_t vocab_size,
    double penalty, const double *logits_in, double *logits_out) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  const et_beam_status vst = validate_vocab_size(vocab_size);
  if (vst != ET_BEAM_OK)
    return vst;
  if (history_len > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;
  const size_t hist_bytes = history_len * sizeof(uint32_t);
  const size_t logit_bytes = vocab_size * sizeof(double);
  if (history_len > 0u &&
      !aligned_span(history_tokens, alignof(uint32_t), hist_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (!aligned_span(logits_in, alignof(double), logit_bytes) ||
      !aligned_span(logits_out, alignof(double), logit_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_partially_overlap(logits_in, logits_out, logit_bytes) ||
      (history_len > 0u &&
       spans_overlap(history_tokens, hist_bytes, logits_out, logit_bytes)))
    return ET_BEAM_ERR_ALIASING;

  return apply_rep_penalty_internal(history_tokens, history_len, vocab_size,
                                    penalty, logits_in, logits_out);
}

et_beam_status et_beam_apply_no_repeat_ngram_v1(const uint32_t *seq_tokens,
                                                size_t seq_len,
                                                size_t vocab_size,
                                                uint32_t ngram_size,
                                                const double *logits_in,
                                                double *logits_out) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  const et_beam_status vst = validate_vocab_size(vocab_size);
  if (vst != ET_BEAM_OK)
    return vst;
  if (seq_len > ET_BEAM_MAX_SEQ_LEN)
    return ET_BEAM_ERR_LIMIT;
  const size_t seq_bytes = seq_len * sizeof(uint32_t);
  const size_t logit_bytes = vocab_size * sizeof(double);
  if (seq_len > 0u && !aligned_span(seq_tokens, alignof(uint32_t), seq_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (!aligned_span(logits_in, alignof(double), logit_bytes) ||
      !aligned_span(logits_out, alignof(double), logit_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_partially_overlap(logits_in, logits_out, logit_bytes) ||
      (seq_len > 0u &&
       spans_overlap(seq_tokens, seq_bytes, logits_out, logit_bytes)))
    return ET_BEAM_ERR_ALIASING;

  return apply_no_repeat_ngram_internal(seq_tokens, seq_len, vocab_size,
                                        ngram_size, logits_in, logits_out);
}

et_beam_status et_beam_apply_min_length_v1(size_t generated_len,
                                           size_t min_new_tokens,
                                           uint32_t eos_token_id,
                                           size_t vocab_size,
                                           const double *logits_in,
                                           double *logits_out) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  const et_beam_status vst = validate_vocab_size(vocab_size);
  if (vst != ET_BEAM_OK)
    return vst;
  const size_t logit_bytes = vocab_size * sizeof(double);
  if (!aligned_span(logits_in, alignof(double), logit_bytes) ||
      !aligned_span(logits_out, alignof(double), logit_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_partially_overlap(logits_in, logits_out, logit_bytes))
    return ET_BEAM_ERR_ALIASING;

  return apply_min_length_internal(generated_len, min_new_tokens, eos_token_id,
                                   vocab_size, logits_in, logits_out);
}

et_beam_status et_beam_init_v1(et_beam_state_v1 *state,
                               const et_beam_config_v1 *cfg,
                               const uint32_t *prompt_tokens,
                               size_t prompt_len) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!aligned_span(state, alignof(et_beam_state_v1), sizeof(et_beam_state_v1)))
    return ET_BEAM_ERR_INVALID_ARG;

  const et_beam_status cst = validate_config_internal(cfg, prompt_len);
  if (cst != ET_BEAM_OK)
    return cst;

  const size_t prompt_bytes = prompt_len * sizeof(uint32_t);
  if (!aligned_span(prompt_tokens, alignof(uint32_t), prompt_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_overlap(state, sizeof(et_beam_state_v1), cfg,
                    sizeof(et_beam_config_v1)) ||
      spans_overlap(state, sizeof(et_beam_state_v1), prompt_tokens,
                    prompt_bytes))
    return ET_BEAM_ERR_ALIASING;

  for (size_t i = 0u; i < prompt_len; ++i) {
    if (prompt_tokens[i] >= cfg->vocab_size)
      return ET_BEAM_ERR_INVALID_ARG;
  }

  /* Validation passed: commit initialized state cleanly. */
  memset(state, 0, sizeof(*state));
  state->magic = ET_BEAM_STATE_MAGIC;
  state->config = *cfg;
  state->prompt_len = (uint32_t)prompt_len;
  state->steps_done = 0u;
  state->active_count = 1u;
  state->finished_count = 0u;
  state->done = 0u;
  state->stopped_early = 0u;
  state->active_log_probs[0] = 0.0;
  state->active_lens[0] = (uint32_t)prompt_len;
  memcpy(state->active_tokens[0], prompt_tokens, prompt_bytes);
  return ET_BEAM_OK;
}

et_beam_status et_beam_step_v1(et_beam_state_v1 *state, const double *logits,
                               size_t logits_rows) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!state_is_valid(state))
    return ET_BEAM_ERR_STATE;
  if (state->done != 0u || state->active_count == 0u ||
      state->steps_done >= state->config.max_new_tokens)
    return ET_BEAM_ERR_STATE;
  if (logits_rows != (size_t)state->active_count)
    return ET_BEAM_ERR_INVALID_ARG;

  const size_t vocab_size = (size_t)state->config.vocab_size;
  const size_t total_elems = logits_rows * vocab_size;
  const size_t total_bytes = total_elems * sizeof(double);
  if (!aligned_span(logits, alignof(double), total_bytes))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_overlap(state, sizeof(et_beam_state_v1), logits, total_bytes))
    return ET_BEAM_ERR_ALIASING;

  const uint32_t num_beams = state->config.num_beams;
  const size_t top_capacity = 2u * (size_t)num_beams;
  beam_candidate top_cands[ET_BEAM_MAX_FINISHED];
  size_t top_count = 0u;

  /*
   * Stage 1 (failure-atomic): for every active beam, apply logit processors
   * in canonical order (repetition_penalty -> no_repeat_ngram -> min_length),
   * compute binary64 log-softmax, and collect top 2*B candidates under the
   * strict total order (log_prob DESC, parent_idx ASC, token_id ASC).
   */
  for (uint32_t b = 0u; b < state->active_count; ++b) {
    const double *row_logits = logits + (size_t)b * vocab_size;
    const uint32_t *seq = state->active_tokens[b];
    const size_t seq_len = (size_t)state->active_lens[b];

    et_beam_status st = apply_rep_penalty_internal(
        seq, seq_len, vocab_size, state->config.repetition_penalty, row_logits,
        state->scratch_logits);
    if (st != ET_BEAM_OK)
      return st;

    st = apply_no_repeat_ngram_internal(
        seq, seq_len, vocab_size, state->config.no_repeat_ngram_size,
        state->scratch_logits, state->scratch_logits);
    if (st != ET_BEAM_OK)
      return st;

    st = apply_min_length_internal(
        (size_t)state->steps_done, (size_t)state->config.min_new_tokens,
        state->config.eos_token_id, vocab_size, state->scratch_logits,
        state->scratch_logits);
    if (st != ET_BEAM_OK)
      return st;

    st = log_softmax_internal(vocab_size, state->scratch_logits,
                              state->scratch_log_probs);
    if (st != ET_BEAM_OK)
      return st;

    const double parent_lp = state->active_log_probs[b];
    for (size_t v = 0u; v < vocab_size; ++v) {
      const double step_lp = state->scratch_log_probs[v];
      if (!isfinite(step_lp))
        continue;
      const double cand_lp = parent_lp + step_lp;
      if (!isfinite(cand_lp))
        return ET_BEAM_ERR_NONFINITE;
      const beam_candidate cand = {cand_lp, b, (uint32_t)v};
      insert_top_candidate(top_cands, &top_count, top_capacity, &cand);
    }
  }

  if (top_count == 0u)
    return ET_BEAM_ERR_NO_VALID_TOKENS;

  /*
   * Stage 2 (pre-validate Wu scores before mutating committed state):
   * Identify EOS completions in rank 0..B-1 and next active beams (up to B).
   */
  const uint32_t new_gen_len = state->steps_done + 1u;
  double eos_scores[ET_BEAM_MAX_BEAMS];
  size_t eos_indices[ET_BEAM_MAX_BEAMS];
  size_t eos_count = 0u;
  size_t active_indices[ET_BEAM_MAX_BEAMS];
  uint32_t next_active_count = 0u;

  for (size_t r = 0u; r < top_count; ++r) {
    if (top_cands[r].token_id == state->config.eos_token_id) {
      if (r < (size_t)num_beams) {
        double score = 0.0;
        const et_beam_status sst = compute_wu_score_unchecked(
            top_cands[r].log_prob, (size_t)new_gen_len,
            state->config.length_penalty, &score);
        if (sst != ET_BEAM_OK)
          return sst;
        eos_scores[eos_count] = score;
        eos_indices[eos_count] = r;
        eos_count += 1u;
      }
    } else {
      if (next_active_count < num_beams) {
        active_indices[next_active_count] = r;
        next_active_count += 1u;
      }
    }
    if (next_active_count == num_beams && r + 1u >= (size_t)num_beams)
      break;
  }

  /* Pre-check early-stopping upper bound finiteness if applicable. */
  double best_possible_score = 0.0;
  if (state->config.early_stopping == 1u && next_active_count > 0u) {
    const double best_active_lp = top_cands[active_indices[0]].log_prob;
    const et_beam_status ust = compute_wu_score_unchecked(
        best_active_lp, (size_t)state->config.max_new_tokens,
        state->config.length_penalty, &best_possible_score);
    if (ust != ET_BEAM_OK)
      return ust;
  }

  /* Stage 3: atomic state commit (cannot fail from this point onward). */
  for (size_t i = 0u; i < eos_count; ++i) {
    const beam_candidate *c = &top_cands[eos_indices[i]];
    et_beam_hypothesis_v1 hyp;
    memset(&hyp, 0, sizeof(hyp));
    const uint32_t prev_len = state->active_lens[c->parent_idx];
    hyp.seq_len = prev_len + 1u;
    hyp.generated_len = new_gen_len;
    hyp.finished = 1u;
    hyp.reserved = 0u;
    hyp.sum_log_prob = c->log_prob;
    hyp.score = eos_scores[i];
    memcpy(hyp.tokens, state->active_tokens[c->parent_idx],
           (size_t)prev_len * sizeof(uint32_t));
    hyp.tokens[prev_len] = c->token_id;
    insert_finished_hypothesis(state, &hyp);
  }

  for (uint32_t i = 0u; i < next_active_count; ++i) {
    const beam_candidate *c = &top_cands[active_indices[i]];
    const uint32_t prev_len = state->active_lens[c->parent_idx];
    state->next_log_probs[i] = c->log_prob;
    state->next_lens[i] = prev_len + 1u;
    memcpy(state->next_tokens[i], state->active_tokens[c->parent_idx],
           (size_t)prev_len * sizeof(uint32_t));
    state->next_tokens[i][prev_len] = c->token_id;
  }

  for (uint32_t i = 0u; i < next_active_count; ++i) {
    state->active_log_probs[i] = state->next_log_probs[i];
    state->active_lens[i] = state->next_lens[i];
    memcpy(state->active_tokens[i], state->next_tokens[i],
           (size_t)state->next_lens[i] * sizeof(uint32_t));
  }
  state->active_count = next_active_count;
  state->steps_done = new_gen_len;

  if (state->steps_done >= state->config.max_new_tokens ||
      state->active_count == 0u) {
    state->done = 1u;
  } else if (state->config.early_stopping == 1u &&
             state->finished_count >= num_beams) {
    const double kth_finished_score = state->finished_pool[num_beams - 1u].score;
    if (best_possible_score <= kth_finished_score) {
      state->done = 1u;
      state->stopped_early = 1u;
    }
  }

  return ET_BEAM_OK;
}

typedef struct finalize_entry {
  uint32_t is_finished;
  uint32_t index;
  double score;
  double sum_log_prob;
} finalize_entry;

static void get_entry_seq(const et_beam_state_v1 *state,
                          const finalize_entry *e, uint32_t *out_len,
                          const uint32_t **out_toks) {
  if (e->is_finished) {
    *out_len = state->finished_pool[e->index].seq_len;
    *out_toks = state->finished_pool[e->index].tokens;
  } else {
    *out_len = state->active_lens[e->index];
    *out_toks = state->active_tokens[e->index];
  }
}

static int finalize_entry_is_better(const et_beam_state_v1 *state,
                                    const finalize_entry *a,
                                    const finalize_entry *b) {
  uint32_t len_a = 0u, len_b = 0u;
  const uint32_t *toks_a = NULL;
  const uint32_t *toks_b = NULL;
  get_entry_seq(state, a, &len_a, &toks_a);
  get_entry_seq(state, b, &len_b, &toks_b);
  return hypothesis_is_better_parts(a->score, a->sum_log_prob, len_a, toks_a,
                                    b->score, b->sum_log_prob, len_b, toks_b);
}

et_beam_status et_beam_finalize_v1(const et_beam_state_v1 *state,
                                   et_beam_hypothesis_v1 *out_hyps,
                                   size_t max_hyps, size_t *out_count) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!state_is_valid(state) || state->done == 0u || state->steps_done == 0u)
    return ET_BEAM_ERR_STATE;
  if (max_hyps == 0u)
    return ET_BEAM_ERR_INVALID_ARG;
  if (max_hyps > SIZE_MAX / sizeof(et_beam_hypothesis_v1))
    return ET_BEAM_ERR_LIMIT;

  const size_t hyps_bytes = max_hyps * sizeof(et_beam_hypothesis_v1);
  if (!aligned_span(out_hyps, alignof(et_beam_hypothesis_v1), hyps_bytes) ||
      !aligned_span(out_count, alignof(size_t), sizeof(size_t)))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_overlap(state, sizeof(et_beam_state_v1), out_hyps, hyps_bytes) ||
      spans_overlap(state, sizeof(et_beam_state_v1), out_count,
                    sizeof(size_t)) ||
      spans_overlap(out_hyps, hyps_bytes, out_count, sizeof(size_t)))
    return ET_BEAM_ERR_ALIASING;

  /* Build sorted candidate list (all finished, plus active beams if
   * finished_count < num_beams) before mutating out_hyps / *out_count. */
  finalize_entry entries[ET_BEAM_MAX_FINISHED + ET_BEAM_MAX_BEAMS];
  size_t entry_count = 0u;

  for (uint32_t i = 0u; i < state->finished_count; ++i) {
    const finalize_entry e = {1u, i, state->finished_pool[i].score,
                              state->finished_pool[i].sum_log_prob};
    size_t pos = entry_count;
    while (pos > 0u && finalize_entry_is_better(state, &e, &entries[pos - 1u])) {
      entries[pos] = entries[pos - 1u];
      pos -= 1u;
    }
    entries[pos] = e;
    entry_count += 1u;
  }

  if (state->finished_count < state->config.num_beams) {
    for (uint32_t i = 0u; i < state->active_count; ++i) {
      double score = 0.0;
      const et_beam_status sst = compute_wu_score_unchecked(
          state->active_log_probs[i], (size_t)state->steps_done,
          state->config.length_penalty, &score);
      if (sst != ET_BEAM_OK)
        return sst;
      const finalize_entry e = {0u, i, score, state->active_log_probs[i]};
      size_t pos = entry_count;
      while (pos > 0u &&
             finalize_entry_is_better(state, &e, &entries[pos - 1u])) {
        entries[pos] = entries[pos - 1u];
        pos -= 1u;
      }
      entries[pos] = e;
      entry_count += 1u;
    }
  }

  if (entry_count == 0u)
    return ET_BEAM_ERR_STATE;

  size_t emit = entry_count;
  if (emit > (size_t)state->config.num_beams)
    emit = (size_t)state->config.num_beams;
  if (emit > max_hyps)
    emit = max_hyps;

  for (size_t i = 0u; i < emit; ++i) {
    if (entries[i].is_finished) {
      out_hyps[i] = state->finished_pool[entries[i].index];
    } else {
      const uint32_t idx = entries[i].index;
      memset(&out_hyps[i], 0, sizeof(out_hyps[i]));
      out_hyps[i].seq_len = state->active_lens[idx];
      out_hyps[i].generated_len = state->steps_done;
      out_hyps[i].finished = 0u;
      out_hyps[i].reserved = 0u;
      out_hyps[i].sum_log_prob = state->active_log_probs[idx];
      out_hyps[i].score = entries[i].score;
      memcpy(out_hyps[i].tokens, state->active_tokens[idx],
             (size_t)state->active_lens[idx] * sizeof(uint32_t));
    }
  }
  *out_count = emit;
  return ET_BEAM_OK;
}

/* ---- Read-Only State Queries ---- */

et_beam_status et_beam_get_active_beam_v1(const et_beam_state_v1 *state,
                                          size_t beam_idx, uint32_t *out_tokens,
                                          size_t max_tokens, size_t *out_len,
                                          double *out_log_prob) {
  if (!fp_environment_supported())
    return ET_BEAM_ERR_FP_ENV;
  if (!state_is_valid(state))
    return ET_BEAM_ERR_STATE;
  if (beam_idx >= (size_t)state->active_count)
    return ET_BEAM_ERR_INVALID_ARG;

  const size_t len = (size_t)state->active_lens[beam_idx];
  if (max_tokens < len)
    return ET_BEAM_ERR_INVALID_ARG;
  if (max_tokens > SIZE_MAX / sizeof(uint32_t))
    return ET_BEAM_ERR_LIMIT;

  const size_t tok_bytes = max_tokens * sizeof(uint32_t);
  if (!aligned_span(out_tokens, alignof(uint32_t), tok_bytes) ||
      !aligned_span(out_len, alignof(size_t), sizeof(size_t)) ||
      !aligned_span(out_log_prob, alignof(double), sizeof(double)))
    return ET_BEAM_ERR_INVALID_ARG;
  if (spans_overlap(state, sizeof(et_beam_state_v1), out_tokens, tok_bytes) ||
      spans_overlap(state, sizeof(et_beam_state_v1), out_len, sizeof(size_t)) ||
      spans_overlap(state, sizeof(et_beam_state_v1), out_log_prob,
                    sizeof(double)) ||
      spans_overlap(out_tokens, tok_bytes, out_len, sizeof(size_t)) ||
      spans_overlap(out_tokens, tok_bytes, out_log_prob, sizeof(double)) ||
      spans_overlap(out_len, sizeof(size_t), out_log_prob, sizeof(double)))
    return ET_BEAM_ERR_ALIASING;

  memcpy(out_tokens, state->active_tokens[beam_idx], len * sizeof(uint32_t));
  *out_len = len;
  *out_log_prob = state->active_log_probs[beam_idx];
  return ET_BEAM_OK;
}

uint32_t et_beam_active_count_v1(const et_beam_state_v1 *state) {
  return state_is_valid(state) ? state->active_count : 0u;
}

uint32_t et_beam_finished_count_v1(const et_beam_state_v1 *state) {
  return state_is_valid(state) ? state->finished_count : 0u;
}

uint32_t et_beam_steps_done_v1(const et_beam_state_v1 *state) {
  return state_is_valid(state) ? state->steps_done : 0u;
}

uint32_t et_beam_is_done_v1(const et_beam_state_v1 *state) {
  return state_is_valid(state) ? state->done : 0u;
}

uint32_t et_beam_stopped_early_v1(const et_beam_state_v1 *state) {
  return state_is_valid(state) ? state->stopped_early : 0u;
}
