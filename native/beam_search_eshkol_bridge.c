#include "beam_search_eshkol_bridge.h"

#include <math.h>
#include <stdalign.h>
#include <string.h>

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
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

int64_t et_beam_aot_config_bytes(void) {
  return (int64_t)sizeof(et_beam_config_v1);
}

int64_t et_beam_aot_state_bytes(void) {
  return (int64_t)sizeof(et_beam_state_v1);
}

int64_t et_beam_aot_hyp_bytes(void) {
  return (int64_t)sizeof(et_beam_hypothesis_v1);
}

int32_t et_beam_aot_set_config(void *cfg_buf, int64_t num_beams,
                               int64_t vocab_size, int64_t eos_token_id,
                               int64_t max_new_tokens, int64_t min_new_tokens,
                               int64_t no_repeat_ngram_size,
                               int64_t early_stopping, double length_penalty,
                               double repetition_penalty) {
  if (!aligned_span(cfg_buf, alignof(et_beam_config_v1),
                    sizeof(et_beam_config_v1)))
    return ET_BEAM_ERR_INVALID_ARG;
  if (num_beams < 0 || vocab_size < 0 || eos_token_id < 0 ||
      max_new_tokens < 0 || min_new_tokens < 0 || no_repeat_ngram_size < 0 ||
      early_stopping < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (num_beams > UINT32_MAX || vocab_size > UINT32_MAX ||
      eos_token_id > UINT32_MAX || max_new_tokens > UINT32_MAX ||
      min_new_tokens > UINT32_MAX || no_repeat_ngram_size > UINT32_MAX ||
      early_stopping > UINT32_MAX)
    return ET_BEAM_ERR_LIMIT;

  const et_beam_config_v1 cfg = {
      sizeof(et_beam_config_v1),
      (uint32_t)num_beams,
      (uint32_t)vocab_size,
      (uint32_t)eos_token_id,
      (uint32_t)max_new_tokens,
      (uint32_t)min_new_tokens,
      (uint32_t)no_repeat_ngram_size,
      (uint32_t)early_stopping,
      0u,
      length_penalty,
      repetition_penalty};
  memcpy(cfg_buf, &cfg, sizeof(cfg));
  return ET_BEAM_OK;
}

int32_t et_beam_aot_init(void *state_buf, const void *cfg_buf,
                         const void *prompt_u32, int64_t prompt_len) {
  if (prompt_len < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  return et_beam_init_v1((et_beam_state_v1 *)state_buf,
                         (const et_beam_config_v1 *)cfg_buf,
                         (const uint32_t *)prompt_u32, (size_t)prompt_len);
}

int32_t et_beam_aot_step(void *state_buf, const void *logits_f64,
                         int64_t logits_rows) {
  if (logits_rows < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  return et_beam_step_v1((et_beam_state_v1 *)state_buf,
                         (const double *)logits_f64, (size_t)logits_rows);
}

int32_t et_beam_aot_finalize(const void *state_buf, void *hyps_buf,
                             int64_t max_hyps, void *out_count_i64) {
  if (max_hyps < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  if ((uint64_t)max_hyps > SIZE_MAX / sizeof(et_beam_hypothesis_v1))
    return ET_BEAM_ERR_LIMIT;
  if (!aligned_span(out_count_i64, alignof(int64_t), sizeof(int64_t)))
    return ET_BEAM_ERR_INVALID_ARG;
  const size_t hyps_bytes = (size_t)max_hyps * sizeof(et_beam_hypothesis_v1);
  if (spans_overlap(state_buf, sizeof(et_beam_state_v1), out_count_i64,
                    sizeof(int64_t)) ||
      spans_overlap(hyps_buf, hyps_bytes, out_count_i64, sizeof(int64_t)))
    return ET_BEAM_ERR_ALIASING;
  size_t count = 0u;
  const et_beam_status st = et_beam_finalize_v1(
      (const et_beam_state_v1 *)state_buf, (et_beam_hypothesis_v1 *)hyps_buf,
      (size_t)max_hyps, &count);
  if (st == ET_BEAM_OK) {
    const int64_t c64 = (int64_t)count;
    memcpy(out_count_i64, &c64, sizeof(c64));
  }
  return st;
}

int64_t et_beam_aot_active_count(const void *state_buf) {
  return (int64_t)et_beam_active_count_v1((const et_beam_state_v1 *)state_buf);
}

int64_t et_beam_aot_finished_count(const void *state_buf) {
  return (int64_t)et_beam_finished_count_v1((const et_beam_state_v1 *)state_buf);
}

int64_t et_beam_aot_steps_done(const void *state_buf) {
  return (int64_t)et_beam_steps_done_v1((const et_beam_state_v1 *)state_buf);
}

int64_t et_beam_aot_is_done(const void *state_buf) {
  return (int64_t)et_beam_is_done_v1((const et_beam_state_v1 *)state_buf);
}

int64_t et_beam_aot_stopped_early(const void *state_buf) {
  return (int64_t)et_beam_stopped_early_v1((const et_beam_state_v1 *)state_buf);
}

int64_t et_beam_aot_active_len(const void *state_buf, int64_t beam_idx) {
  if (beam_idx < 0)
    return -1;
  uint32_t toks[ET_BEAM_MAX_SEQ_LEN];
  size_t len = 0u;
  double lp = 0.0;
  if (et_beam_get_active_beam_v1((const et_beam_state_v1 *)state_buf,
                                 (size_t)beam_idx, toks, ET_BEAM_MAX_SEQ_LEN,
                                 &len, &lp) != ET_BEAM_OK)
    return -1;
  return (int64_t)len;
}

int64_t et_beam_aot_active_token(const void *state_buf, int64_t beam_idx,
                                 int64_t pos) {
  if (beam_idx < 0 || pos < 0)
    return -1;
  uint32_t toks[ET_BEAM_MAX_SEQ_LEN];
  size_t len = 0u;
  double lp = 0.0;
  if (et_beam_get_active_beam_v1((const et_beam_state_v1 *)state_buf,
                                 (size_t)beam_idx, toks, ET_BEAM_MAX_SEQ_LEN,
                                 &len, &lp) != ET_BEAM_OK)
    return -1;
  if ((size_t)pos >= len)
    return -1;
  return (int64_t)toks[pos];
}

double et_beam_aot_active_log_prob(const void *state_buf, int64_t beam_idx) {
  if (beam_idx < 0)
    return (double)NAN;
  uint32_t toks[ET_BEAM_MAX_SEQ_LEN];
  size_t len = 0u;
  double lp = 0.0;
  if (et_beam_get_active_beam_v1((const et_beam_state_v1 *)state_buf,
                                 (size_t)beam_idx, toks, ET_BEAM_MAX_SEQ_LEN,
                                 &len, &lp) != ET_BEAM_OK)
    return (double)NAN;
  return lp;
}

static int read_hyp(const void *hyps_buf, int64_t hyp_idx,
                    et_beam_hypothesis_v1 *out) {
  if (hyp_idx < 0 || hyp_idx >= (int64_t)ET_BEAM_MAX_FINISHED)
    return 0;
  const size_t offset = (size_t)hyp_idx * sizeof(et_beam_hypothesis_v1);
  if (!aligned_span(hyps_buf, alignof(et_beam_hypothesis_v1),
                    offset + sizeof(et_beam_hypothesis_v1)))
    return 0;
  memcpy(out, (const unsigned char *)hyps_buf + offset, sizeof(*out));
  return 1;
}

int64_t et_beam_aot_hyp_seq_len(const void *hyps_buf, int64_t hyp_idx) {
  et_beam_hypothesis_v1 h;
  return read_hyp(hyps_buf, hyp_idx, &h) ? (int64_t)h.seq_len : -1;
}

int64_t et_beam_aot_hyp_gen_len(const void *hyps_buf, int64_t hyp_idx) {
  et_beam_hypothesis_v1 h;
  return read_hyp(hyps_buf, hyp_idx, &h) ? (int64_t)h.generated_len : -1;
}

int64_t et_beam_aot_hyp_finished(const void *hyps_buf, int64_t hyp_idx) {
  et_beam_hypothesis_v1 h;
  return read_hyp(hyps_buf, hyp_idx, &h) ? (int64_t)h.finished : -1;
}

int64_t et_beam_aot_hyp_token(const void *hyps_buf, int64_t hyp_idx,
                              int64_t pos) {
  et_beam_hypothesis_v1 h;
  if (pos < 0 || !read_hyp(hyps_buf, hyp_idx, &h))
    return -1;
  if ((uint32_t)pos >= h.seq_len || (size_t)pos >= ET_BEAM_MAX_SEQ_LEN)
    return -1;
  return (int64_t)h.tokens[pos];
}

double et_beam_aot_hyp_sum_log_prob(const void *hyps_buf, int64_t hyp_idx) {
  et_beam_hypothesis_v1 h;
  return read_hyp(hyps_buf, hyp_idx, &h) ? h.sum_log_prob : (double)NAN;
}

double et_beam_aot_hyp_score(const void *hyps_buf, int64_t hyp_idx) {
  et_beam_hypothesis_v1 h;
  return read_hyp(hyps_buf, hyp_idx, &h) ? h.score : (double)NAN;
}

int32_t et_beam_aot_apply_rep_penalty(const void *hist_u32, int64_t hist_len,
                                      int64_t vocab_size, double penalty,
                                      const void *logits_in, void *logits_out) {
  if (hist_len < 0 || vocab_size < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  return et_beam_apply_repetition_penalty_v1(
      (const uint32_t *)hist_u32, (size_t)hist_len, (size_t)vocab_size, penalty,
      (const double *)logits_in, (double *)logits_out);
}

int32_t et_beam_aot_apply_no_repeat_ngram(const void *seq_u32, int64_t seq_len,
                                          int64_t vocab_size,
                                          int64_t ngram_size,
                                          const void *logits_in,
                                          void *logits_out) {
  if (seq_len < 0 || vocab_size < 0 || ngram_size < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (ngram_size > UINT32_MAX)
    return ET_BEAM_ERR_LIMIT;
  return et_beam_apply_no_repeat_ngram_v1(
      (const uint32_t *)seq_u32, (size_t)seq_len, (size_t)vocab_size,
      (uint32_t)ngram_size, (const double *)logits_in, (double *)logits_out);
}

int32_t et_beam_aot_apply_min_length(int64_t gen_len, int64_t min_new_tokens,
                                     int64_t eos_id, int64_t vocab_size,
                                     const void *logits_in, void *logits_out) {
  if (gen_len < 0 || min_new_tokens < 0 || eos_id < 0 || vocab_size < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  if (eos_id > UINT32_MAX)
    return ET_BEAM_ERR_INVALID_ARG;
  return et_beam_apply_min_length_v1(
      (size_t)gen_len, (size_t)min_new_tokens, (uint32_t)eos_id,
      (size_t)vocab_size, (const double *)logits_in, (double *)logits_out);
}

int32_t et_beam_aot_log_softmax(int64_t vocab_size, const void *logits_in,
                                void *log_probs_out) {
  if (vocab_size < 0)
    return ET_BEAM_ERR_INVALID_ARG;
  return et_beam_log_softmax_v1((size_t)vocab_size, (const double *)logits_in,
                                (double *)log_probs_out);
}

static int valid_element_span(const void *buf, int64_t buf_bytes,
                              int64_t index, size_t elem_size,
                              size_t alignment) {
  if (buf_bytes < 0 || index < 0 || (uint64_t)buf_bytes > SIZE_MAX ||
      (uint64_t)index > SIZE_MAX / elem_size)
    return 0;
  const size_t bytes = (size_t)buf_bytes;
  const size_t offset = (size_t)index * elem_size;
  return aligned_span(buf, alignment, bytes) && bytes >= elem_size &&
         offset <= bytes - elem_size;
}

int32_t et_beam_aot_store_u32(void *buf, int64_t buf_bytes, int64_t index,
                              int64_t value) {
  if (!valid_element_span(buf, buf_bytes, index, sizeof(uint32_t),
                          alignof(uint32_t)) || value < 0 ||
      (uint64_t)value > UINT32_MAX)
    return ET_BEAM_ERR_INVALID_ARG;
  const uint32_t v = (uint32_t)value;
  memcpy((unsigned char *)buf + (size_t)index * sizeof(v), &v, sizeof(v));
  return ET_BEAM_OK;
}

int64_t et_beam_aot_load_u32(const void *buf, int64_t buf_bytes, int64_t index) {
  if (!valid_element_span(buf, buf_bytes, index, sizeof(uint32_t),
                          alignof(uint32_t)))
    return -1;
  uint32_t v = 0u;
  memcpy(&v, (const unsigned char *)buf + (size_t)index * sizeof(v), sizeof(v));
  return (int64_t)v;
}

int64_t et_beam_aot_load_i64(const void *buf, int64_t buf_bytes, int64_t index) {
  if (!valid_element_span(buf, buf_bytes, index, sizeof(int64_t),
                          alignof(int64_t)))
    return -1;
  int64_t v = 0;
  memcpy(&v, (const unsigned char *)buf + (size_t)index * sizeof(v), sizeof(v));
  return v;
}

int32_t et_beam_aot_store_f64(void *buf, int64_t buf_bytes, int64_t index,
                              double value) {
  if (!valid_element_span(buf, buf_bytes, index, sizeof(double),
                          alignof(double)))
    return ET_BEAM_ERR_INVALID_ARG;
  memcpy((unsigned char *)buf + (size_t)index * sizeof(value), &value,
         sizeof(value));
  return ET_BEAM_OK;
}

double et_beam_aot_load_f64(const void *buf, int64_t buf_bytes, int64_t index) {
  if (!valid_element_span(buf, buf_bytes, index, sizeof(double),
                          alignof(double)))
    return (double)NAN;
  double v = 0.0;
  memcpy(&v, (const unsigned char *)buf + (size_t)index * sizeof(v), sizeof(v));
  return v;
}

double et_beam_aot_nan_f64(void) { return (double)NAN; }

double et_beam_aot_pos_inf_f64(void) { return (double)INFINITY; }

double et_beam_aot_neg_inf_f64(void) { return -(double)INFINITY; }
