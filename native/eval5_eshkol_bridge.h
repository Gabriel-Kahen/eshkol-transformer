#ifndef ESHKOL_TRANSFORMER_EVAL5_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_EVAL5_ESHKOL_BRIDGE_H

#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_EVAL5_AOT_API __attribute__((visibility("default")))
#else
#define ET_EVAL5_AOT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles for compiled Eshkol AOT callers. */
typedef struct et_eval5_aot_corpus et_eval5_aot_corpus;
typedef struct et_eval5_aot_mc_batch et_eval5_aot_mc_batch;
typedef struct et_eval5_aot_manifest et_eval5_aot_manifest;

/* Flat summary_i64 indices for contamination audit results. */
enum {
  ET_EVAL5_AOT_AUDIT_EVAL_DOCS = 0,
  ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS = 1,
  ET_EVAL5_AOT_AUDIT_EXACT_MATCHES = 2,
  ET_EVAL5_AOT_AUDIT_NGRAM_DOCS = 3,
  ET_EVAL5_AOT_AUDIT_TOTAL_NGRAMS = 4,
  ET_EVAL5_AOT_AUDIT_MATCHED_NGRAMS = 5,
  ET_EVAL5_AOT_AUDIT_SUMMARY_LEN = 6
};

/* Flat metrics_f64 indices for LM scorer results. */
enum {
  ET_EVAL5_AOT_LM_TOTAL_NLL = 0,
  ET_EVAL5_AOT_LM_MEAN_NLL = 1,
  ET_EVAL5_AOT_LM_PERPLEXITY = 2,
  ET_EVAL5_AOT_LM_BITS_PER_BYTE = 3,
  ET_EVAL5_AOT_LM_METRICS_LEN = 4
};

/* Flat manifest readback layouts for et_eval5_aot_manifest_get_metrics:
 * counts_i64[12]:
 *   [0]  metric_mask
 *   [1]  mc_normalize_by_length
 *   [2]  lm.token_count
 *   [3]  lm.byte_count
 *   [4]  mc.correct_count
 *   [5]  mc.total_count
 *   [6]  audit.eval_documents
 *   [7]  audit.contaminated_documents
 *   [8]  audit.exact_doc_matches
 *   [9]  audit.ngram_contaminated_docs
 *   [10] audit.total_eval_ngrams
 *   [11] audit.matched_eval_ngrams
 * metrics_f64[6]:
 *   [0] lm.total_nll
 *   [1] lm.mean_nll
 *   [2] lm.perplexity
 *   [3] lm.bits_per_byte
 *   [4] mc.accuracy
 *   [5] audit.contamination_rate
 */
enum {
  ET_EVAL5_AOT_MANIFEST_COUNTS_LEN = 12,
  ET_EVAL5_AOT_MANIFEST_METRICS_LEN = 6
};

/* Document corpus lifecycle and insertion (deep-copies byte and token spans). */
ET_EVAL5_AOT_API int32_t et_eval5_aot_corpus_create(et_eval5_aot_corpus **out);
ET_EVAL5_AOT_API void et_eval5_aot_corpus_destroy(et_eval5_aot_corpus *corpus);
ET_EVAL5_AOT_API int32_t et_eval5_aot_corpus_add_doc(
    et_eval5_aot_corpus *corpus, const uint8_t *bytes, int64_t byte_count,
    const uint32_t *tokens, int64_t token_count);
ET_EVAL5_AOT_API int32_t et_eval5_aot_corpus_doc_count(
    const et_eval5_aot_corpus *corpus, int64_t *out_count);

/* Exact contamination audit delegating to et_eval5_audit_contamination.
 * Returns ET_EVAL5_OK on clean pass, ET_EVAL5_ERR_CONTAMINATED on policy
 * failure (populating summary_i64, contamination_rate_out, and
 * contaminated_indices), or another negative et_eval5_status leaving outputs
 * untouched. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_audit_contamination(
    const et_eval5_aot_corpus *train_corpus,
    const et_eval5_aot_corpus *eval_corpus, int32_t ngram_mode,
    int32_t ngram_order, int32_t forbid_exact_match, double min_ngram_overlap,
    double max_allowed_rate, int64_t *contaminated_indices,
    int64_t index_capacity, int64_t *summary_i64,
    double *contamination_rate_out);

/* Binary64 log-sum-exp LM scorer (total NLL, mean NLL, PPL, BPB). */
ET_EVAL5_AOT_API int32_t et_eval5_aot_score_lm(
    const float *logits, int64_t logit_count, const uint32_t *target_ids,
    int64_t token_count, int64_t vocab_size, int64_t byte_count,
    double *metrics_f64);

/* Multiple-choice batch lifecycle and scorer. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_mc_create(et_eval5_aot_mc_batch **out);
ET_EVAL5_AOT_API void et_eval5_aot_mc_destroy(et_eval5_aot_mc_batch *batch);
ET_EVAL5_AOT_API int32_t et_eval5_aot_mc_begin_item(
    et_eval5_aot_mc_batch *batch, int64_t correct_index);
ET_EVAL5_AOT_API int32_t et_eval5_aot_mc_add_choice(
    et_eval5_aot_mc_batch *batch, const float *logits, int64_t logit_count,
    const uint32_t *target_ids, int64_t token_count, int64_t vocab_size);
ET_EVAL5_AOT_API int32_t et_eval5_aot_score_mc(
    const et_eval5_aot_mc_batch *batch, int32_t normalize_by_length,
    int64_t *correct_count_out, int64_t *total_count_out, double *accuracy_out);

/* Versioned SHA-256 manifest lifecycle, population, fail-closed atomic
 * publication, O_NOFOLLOW read, and verification. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_create(
    et_eval5_aot_manifest **out);
ET_EVAL5_AOT_API void et_eval5_aot_manifest_destroy(
    et_eval5_aot_manifest *manifest);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_set_identity(
    et_eval5_aot_manifest *manifest, const char *suite, const char *task,
    const char *split, int64_t context_length, int64_t max_samples);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_set_digests(
    et_eval5_aot_manifest *manifest, const uint8_t *train_sha256,
    const uint8_t *eval_sha256, const uint8_t *tokenizer_sha256,
    const uint8_t *checkpoint_sha256);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_set_audit(
    et_eval5_aot_manifest *manifest, int32_t ngram_mode, int32_t ngram_order,
    int32_t forbid_exact_match, double min_ngram_overlap,
    double max_allowed_rate, const int64_t *summary_i64,
    double contamination_rate);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_set_lm(
    et_eval5_aot_manifest *manifest, int64_t token_count, int64_t byte_count,
    const double *metrics_f64);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_set_mc(
    et_eval5_aot_manifest *manifest, int32_t normalize_by_length,
    int64_t correct_count, int64_t total_count, double accuracy);

/* Verifies manifest schema, FP environment, and contamination pass gate.
 * Returns ET_EVAL5_OK only when the manifest is valid AND passes its
 * contamination policy; returns ET_EVAL5_ERR_CONTAMINATED on policy failure. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_verify_clean(
    const et_eval5_aot_manifest *manifest);

/* In-memory canonical format and verified parse. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_format(
    const et_eval5_aot_manifest *manifest, char *out_buf, int64_t capacity,
    int64_t *out_size);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_parse(
    const char *bytes, int64_t size, et_eval5_aot_manifest *out);

/* Fail-closed atomic manifest publication. Refuses contaminated manifests
 * (ET_EVAL5_ERR_CONTAMINATED) and refuses symlink or non-regular destination
 * entries (ET_EVAL5_ERR_IO) before delegating to et_eval5_manifest_write_atomic. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_write_atomic(
    const char *path, const et_eval5_aot_manifest *manifest);

/* O_NOFOLLOW regular-file manifest read and SHA-256 + schema verification. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_read_file(
    const char *path, et_eval5_aot_manifest *out);

/* Readback helpers for parsed or populated manifests. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_get_metrics(
    const et_eval5_aot_manifest *manifest, int64_t *counts_i64,
    double *metrics_f64, uint8_t *manifest_sha256_out);
ET_EVAL5_AOT_API int32_t et_eval5_aot_manifest_get_digests(
    const et_eval5_aot_manifest *manifest, uint8_t *train_sha256,
    uint8_t *eval_sha256, uint8_t *tokenizer_sha256,
    uint8_t *checkpoint_sha256);

/* Checked SHA-256 (O_NOFOLLOW regular-file for paths) and buffer marshaling. */
ET_EVAL5_AOT_API int32_t et_eval5_aot_sha256_bytes(const uint8_t *data,
                                                   int64_t size,
                                                   uint8_t *out_digest);
ET_EVAL5_AOT_API int32_t et_eval5_aot_sha256_file(const char *path,
                                                  uint8_t *out_digest);
ET_EVAL5_AOT_API int32_t et_eval5_aot_store_u32(uint32_t *buffer,
                                                int64_t capacity, int64_t index,
                                                int64_t value);
ET_EVAL5_AOT_API int32_t et_eval5_aot_store_f32(float *buffer, int64_t capacity,
                                                int64_t index, double value);
ET_EVAL5_AOT_API double et_eval5_aot_load_f64(const double *buffer,
                                              int64_t capacity, int64_t index);
ET_EVAL5_AOT_API int64_t et_eval5_aot_load_i64(const int64_t *buffer,
                                               int64_t capacity, int64_t index);
ET_EVAL5_AOT_API int32_t et_eval5_aot_bytes_equal(const uint8_t *a,
                                                  const uint8_t *b,
                                                  int64_t count,
                                                  int32_t *equal_out);

#ifdef __cplusplus
}
#endif

#endif
