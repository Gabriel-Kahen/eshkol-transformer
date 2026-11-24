#ifndef ESHKOL_TRANSFORMER_UNIGRAM_TOK_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_UNIGRAM_TOK_ESHKOL_BRIDGE_H

#include "eshkol_transformer/unigram_tok_abi.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_UNIGRAM_AOT_API __attribute__((visibility("default")))
#else
#define ET_UNIGRAM_AOT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque multi-document corpus builder for compiled Eshkol AOT callers. */
typedef struct et_unigram_aot_corpus et_unigram_aot_corpus;

/* Corpus lifecycle and document insertion (deep-copies input bytes). */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_create(
    et_unigram_aot_corpus **out_corpus);
ET_UNIGRAM_AOT_API void et_unigram_aot_corpus_destroy(
    et_unigram_aot_corpus *corpus);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_add_bytes(
    et_unigram_aot_corpus *corpus, const uint8_t *bytes, int64_t byte_count);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_add_cstr(
    et_unigram_aot_corpus *corpus, const char *text);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_doc_count(
    const et_unigram_aot_corpus *corpus, int64_t *out_count);

/* Deterministic Unigram LM training, lifecycle, and atomic persistence.
 * out_stats_f64_3 receives [initial_ll, final_em_start_ll, final_em_end_ll]
 * (may be NULL). out_stats_i64_4 receives [seed_active, final_active,
 * pruning_rounds, total_em_iters] (may be NULL). */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_train(
    const et_unigram_aot_corpus *corpus, int64_t target_vocab_size,
    int64_t seed_vocab_size, int64_t max_piece_bytes, int64_t min_frequency,
    int64_t num_em_sub_iters, int64_t final_em_iters, int64_t mstep_mode,
    int64_t special_mask, double shrinking_factor, double smoothing_alpha,
    et_unigram_model **out_model, int64_t *out_stats_i64_4,
    double *out_stats_f64_3);

ET_UNIGRAM_AOT_API void et_unigram_aot_destroy(et_unigram_model *model);

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_save(const et_unigram_model *model,
                                               const char *path,
                                               uint8_t *out_sha256);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_load(const char *path,
                                               et_unigram_model **out_model,
                                               uint8_t *out_sha256);

/* Model metadata inspection.
 * special_ids writes [bos, eos, pad, unk] with -1 for absent specials. */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_vocab_size(
    const et_unigram_model *model, int64_t *out_vocab_size);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_active_count(
    const et_unigram_model *model, int64_t *out_active_count);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_learned_count(
    const et_unigram_model *model, int64_t *out_learned_count);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_special_ids(
    const et_unigram_model *model, int64_t *out_ids_4);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_piece_log_prob(
    const et_unigram_model *model, int64_t token_id, double *out_log_prob);

/* Exact Viterbi encode, Kudo FFBS sample, Forward-Backward, and lossless decode. */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_encode_bytes(
    const et_unigram_model *model, const uint8_t *bytes, int64_t byte_count,
    uint32_t *out_tokens, int64_t capacity, int64_t *out_count,
    double *out_score);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_encode_cstr(
    const et_unigram_model *model, const char *text, uint32_t *out_tokens,
    int64_t capacity, int64_t *out_count, double *out_score);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_decode_bytes(
    const et_unigram_model *model, const uint32_t *tokens, int64_t token_count,
    uint8_t *out_bytes, int64_t capacity, int64_t *out_byte_count);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_forward_backward_bytes(
    const et_unigram_model *model, const uint8_t *bytes, int64_t byte_count,
    double *out_log_z, double *out_expected_counts, int64_t count_capacity);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_forward_backward_cstr(
    const et_unigram_model *model, const char *text, double *out_log_z,
    double *out_expected_counts, int64_t count_capacity);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_sample_cstr(
    const et_unigram_model *model, const char *text, double alpha_temp,
    int64_t seed, uint32_t *out_tokens, int64_t capacity, int64_t *out_count,
    double *out_score);

/* Perturb one piece's log_prob by delta in a temporary cloned model and return
 * its Forward-Backward marginal log Z (for central finite-difference checks
 * driven directly from compiled Eshkol AOT code). */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_perturbed_log_z_cstr(
    const et_unigram_model *model, const char *text, int64_t piece_id,
    double delta, double *out_log_z);

/* Explicit negative-probe helpers for AOT boundary verification:
 * - check_nonfinite_create: attempts et_unigram_create_v1 with NaN log_prob.
 * - check_fp_env_trap: temporarily sets FE_UPWARD, invokes et_unigram_encode_v1,
 *   immediately restores FE_TONEAREST, and returns the exact status code. */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_probe_nonfinite_create(void);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_probe_fp_env_trap(
    const et_unigram_model *model);

/* Checked buffer marshaling and comparison helpers for Eshkol AOT callers. */
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_store_u8(uint8_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index,
                                                   int64_t value);
ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_u8(const uint8_t *buffer,
                                                  int64_t capacity,
                                                  int64_t index);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_store_u32(uint32_t *buffer,
                                                    int64_t capacity,
                                                    int64_t index,
                                                    int64_t value);
ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_u32(const uint32_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index);
ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_i64(const int64_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index);
ET_UNIGRAM_AOT_API double et_unigram_aot_load_f64(const double *buffer,
                                                  int64_t capacity,
                                                  int64_t index);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_f64_bits_equal(double a, double b);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_bytes_equal(const uint8_t *a,
                                                      const uint8_t *b,
                                                      int64_t count,
                                                      int32_t *equal_out);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_u32_equal(const uint32_t *a,
                                                    const uint32_t *b,
                                                    int64_t count,
                                                    int32_t *equal_out);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_cstr_bytes_equal(const char *cstr,
                                                           const uint8_t *bytes,
                                                           int64_t byte_count,
                                                           int32_t *equal_out);
ET_UNIGRAM_AOT_API int32_t et_unigram_aot_cstr_len(const char *cstr,
                                                   int64_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_UNIGRAM_TOK_ESHKOL_BRIDGE_H */
