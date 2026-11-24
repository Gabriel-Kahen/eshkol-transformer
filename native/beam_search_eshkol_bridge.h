#ifndef ESHKOL_TRANSFORMER_BEAM_SEARCH_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_BEAM_SEARCH_ESHKOL_BRIDGE_H

#include "eshkol_transformer/beam_search_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zero-allocation Eshkol AOT bridge for Deterministic Beam Search & Logit
 * Processors. All buffer pointers are accepted as void * so raw Eshkol FFI
 * addresses (including intentionally misaligned ptr-add addresses) are
 * validated cleanly without misaligned-pointer UB.
 */
int64_t et_beam_aot_config_bytes(void);
int64_t et_beam_aot_state_bytes(void);
int64_t et_beam_aot_hyp_bytes(void);

int32_t et_beam_aot_set_config(void *cfg_buf, int64_t num_beams,
                               int64_t vocab_size, int64_t eos_token_id,
                               int64_t max_new_tokens, int64_t min_new_tokens,
                               int64_t no_repeat_ngram_size,
                               int64_t early_stopping, double length_penalty,
                               double repetition_penalty);

int32_t et_beam_aot_init(void *state_buf, const void *cfg_buf,
                         const void *prompt_u32, int64_t prompt_len);

int32_t et_beam_aot_step(void *state_buf, const void *logits_f64,
                         int64_t logits_rows);

int32_t et_beam_aot_finalize(const void *state_buf, void *hyps_buf,
                             int64_t max_hyps, void *out_count_i64);

int64_t et_beam_aot_active_count(const void *state_buf);
int64_t et_beam_aot_finished_count(const void *state_buf);
int64_t et_beam_aot_steps_done(const void *state_buf);
int64_t et_beam_aot_is_done(const void *state_buf);
int64_t et_beam_aot_stopped_early(const void *state_buf);

int64_t et_beam_aot_active_len(const void *state_buf, int64_t beam_idx);
int64_t et_beam_aot_active_token(const void *state_buf, int64_t beam_idx,
                                 int64_t pos);
double et_beam_aot_active_log_prob(const void *state_buf, int64_t beam_idx);

int64_t et_beam_aot_hyp_seq_len(const void *hyps_buf, int64_t hyp_idx);
int64_t et_beam_aot_hyp_gen_len(const void *hyps_buf, int64_t hyp_idx);
int64_t et_beam_aot_hyp_finished(const void *hyps_buf, int64_t hyp_idx);
int64_t et_beam_aot_hyp_token(const void *hyps_buf, int64_t hyp_idx,
                              int64_t pos);
double et_beam_aot_hyp_sum_log_prob(const void *hyps_buf, int64_t hyp_idx);
double et_beam_aot_hyp_score(const void *hyps_buf, int64_t hyp_idx);

int32_t et_beam_aot_apply_rep_penalty(const void *hist_u32, int64_t hist_len,
                                      int64_t vocab_size, double penalty,
                                      const void *logits_in, void *logits_out);

int32_t et_beam_aot_apply_no_repeat_ngram(const void *seq_u32, int64_t seq_len,
                                          int64_t vocab_size,
                                          int64_t ngram_size,
                                          const void *logits_in,
                                          void *logits_out);

int32_t et_beam_aot_apply_min_length(int64_t gen_len, int64_t min_new_tokens,
                                     int64_t eos_id, int64_t vocab_size,
                                     const void *logits_in, void *logits_out);

int32_t et_beam_aot_log_softmax(int64_t vocab_size, const void *logits_in,
                                void *log_probs_out);

void et_beam_aot_store_u32(void *buf, int64_t index, int64_t value);
int64_t et_beam_aot_load_u32(const void *buf, int64_t index);
int64_t et_beam_aot_load_i64(const void *buf, int64_t index);
void et_beam_aot_store_f64(void *buf, int64_t index, double value);
double et_beam_aot_load_f64(const void *buf, int64_t index);
double et_beam_aot_nan_f64(void);
double et_beam_aot_pos_inf_f64(void);
double et_beam_aot_neg_inf_f64(void);

#ifdef __cplusplus
}
#endif

#endif
