#ifndef ESHKOL_TRANSFORMER_DATA4_PIPELINE_ABI_H
#define ESHKOL_TRANSFORMER_DATA4_PIPELINE_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_DATA4_REPORT_FORMAT_ID "eshkol-data4-corpus-pipeline-report"
#define ET_DATA4_REPORT_VERSION_MAJOR 1u
#define ET_DATA4_REPORT_VERSION_MINOR 0u
#define ET_DATA4_SHA256_BYTES 32u
#define ET_DATA4_SOURCE_ID_MAX 64u
#define ET_DATA4_MINHASH_MAX_PERMS 256u
#define ET_DATA4_MAX_NGRAM_SIZE 64u

typedef int32_t et_data4_status;
enum {
  ET_DATA4_OK = 0,
  ET_DATA4_ERR_ARGUMENT = -1,
  ET_DATA4_ERR_BOUNDS = -2,
  ET_DATA4_ERR_OVERFLOW = -3,
  ET_DATA4_ERR_NOMEM = -4,
  ET_DATA4_ERR_INVALID_SOURCE_ID = -5,
  ET_DATA4_ERR_DUPLICATE_SOURCE_ID = -6,
  ET_DATA4_ERR_FORMAT = -7,
  ET_DATA4_ERR_CHECKSUM = -8,
  ET_DATA4_ERR_VERSION = -9,
  ET_DATA4_ERR_ALIASING = -10,
  ET_DATA4_ERR_ALIGNMENT = -11,
  ET_DATA4_ERR_FP_ENV = -12
};

typedef uint32_t et_data4_disposition;
enum {
  ET_DATA4_DISP_KEPT = 0u,
  ET_DATA4_DISP_REJECTED_UTF8 = 1u,
  ET_DATA4_DISP_REJECTED_LENGTH = 2u,
  ET_DATA4_DISP_REJECTED_QUALITY = 3u,
  ET_DATA4_DISP_REJECTED_REPETITION = 4u,
  ET_DATA4_DISP_DUPLICATE_EXACT = 5u,
  ET_DATA4_DISP_DUPLICATE_NEAR = 6u
};

/* Input UTF-8 document with explicit unique ASCII source_id ([A-Za-z0-9._-],
 * length 1..64). When byte_count == 0, bytes may be NULL or non-NULL. */
typedef struct et_data4_document {
  const char *source_id;
  const uint8_t *bytes;
  size_t byte_count;
} et_data4_document;

/* Deterministic quality-filtering and deduplication configuration.
 * All ratios must be finite and in [0.0, 1.0] (non-negative, rejecting -0.0).
 * When near_dup_enabled == 1, dedup_ngram_size must be in [1, 64],
 * minhash_num_perms in [1, 256], minhash_bands in [1, minhash_num_perms]
 * dividing minhash_num_perms evenly, and near_dup_jaccard_threshold in
 * (0.0, 1.0]. reserved0 must be 0. */
typedef struct et_data4_pipeline_config {
  uint64_t min_bytes;
  uint64_t max_bytes;
  uint64_t min_codepoints;
  uint64_t max_codepoints;
  double min_printable_ratio;
  double min_alnum_ratio;
  double max_dup_line_ratio;
  double max_repeated_ngram_ratio;
  uint32_t rep_ngram_size;
  uint32_t near_dup_enabled;
  uint32_t dedup_ngram_size;
  uint32_t minhash_num_perms;
  uint32_t minhash_bands;
  uint32_t reserved0;
  uint64_t minhash_seed;
  double near_dup_jaccard_threshold;
} et_data4_pipeline_config;

/* Per-document disposition and provenance metadata.
 * For ET_DATA4_DISP_DUPLICATE_EXACT and ET_DATA4_DISP_DUPLICATE_NEAR,
 * has_matched_canonical == 1 and matched_canonical_ordinal / matched_source_id
 * identify the earliest kept canonical document.
 * For ET_DATA4_DISP_DUPLICATE_NEAR, jaccard_num / jaccard_den record the exact
 * byte n-gram set Jaccard similarity (intersection / union). For all other
 * dispositions, jaccard_num == 0 and jaccard_den == 0. */
typedef struct et_data4_doc_result {
  uint64_t ordinal;
  char source_id[ET_DATA4_SOURCE_ID_MAX + 1u];
  uint64_t byte_count;
  uint64_t codepoint_count;
  uint64_t printable_count;
  uint64_t alnum_count;
  uint64_t total_lines;
  uint64_t duplicate_lines;
  uint64_t total_rep_ngrams;
  uint64_t repeated_ngrams;
  uint8_t sha256[ET_DATA4_SHA256_BYTES];
  et_data4_disposition disposition;
  uint32_t has_matched_canonical;
  uint64_t matched_canonical_ordinal;
  char matched_source_id[ET_DATA4_SOURCE_ID_MAX + 1u];
  uint64_t jaccard_num;
  uint64_t jaccard_den;
} et_data4_doc_result;

/* Aggregate corpus pipeline statistics.
 * report_sha256 is zeroed by et_data4_pipeline_run. et_data4_report_format
 * writes the digest to its separate output; the caller may copy it here.
 * et_data4_report_verify populates it in the returned summary. */
typedef struct et_data4_pipeline_summary {
  uint64_t input_documents;
  uint64_t kept_documents;
  uint64_t rejected_utf8_documents;
  uint64_t rejected_length_documents;
  uint64_t rejected_quality_documents;
  uint64_t rejected_repetition_documents;
  uint64_t duplicate_exact_documents;
  uint64_t duplicate_near_documents;
  uint64_t input_bytes;
  uint64_t kept_bytes;
  uint8_t report_sha256[ET_DATA4_SHA256_BYTES];
} et_data4_pipeline_summary;

/* Populate deterministic default configuration values. */
et_data4_status et_data4_config_default(et_data4_pipeline_config *config);

/* Return canonical lowercase ASCII name for a disposition, or NULL if invalid. */
const char *et_data4_disposition_name(et_data4_disposition disposition);

/* Strict RFC 3629 UTF-8 validation. Rejects overlong encodings, UTF-16
 * surrogates (U+D800..U+DFFF), codepoints > U+10FFFF, stray continuations, and
 * truncated sequences. Returns ET_DATA4_OK on valid UTF-8, ET_DATA4_ERR_FORMAT
 * on invalid UTF-8. Output count pointers are optional (may be NULL). */
et_data4_status et_data4_validate_utf8(const uint8_t *bytes, size_t byte_count,
                                       uint64_t *codepoint_count,
                                       uint64_t *printable_count,
                                       uint64_t *alnum_count);

/* Compute canonical SHA-256 digest over raw bytes. */
et_data4_status et_data4_sha256(const uint8_t *bytes, size_t byte_count,
                                uint8_t digest_out[ET_DATA4_SHA256_BYTES]);

/* Compute deterministic seeded MinHash signature over unique byte n-grams.
 * If byte_count < ngram_size, unique_ngram_count_out is 0 and signature_out
 * is filled with UINT64_MAX. */
et_data4_status et_data4_compute_minhash(const uint8_t *bytes,
                                         size_t byte_count,
                                         uint32_t ngram_size,
                                         uint32_t num_perms, uint64_t seed,
                                         uint64_t *signature_out,
                                         size_t signature_capacity,
                                         uint64_t *unique_ngram_count_out);

/* Compute exact byte n-gram set Jaccard numerator (|A intersect B|) and
 * denominator (|A union B|). Both are 0 when neither input has any n-grams. */
et_data4_status et_data4_exact_jaccard(const uint8_t *a_bytes, size_t a_count,
                                       const uint8_t *b_bytes, size_t b_count,
                                       uint32_t ngram_size, uint64_t *num_out,
                                       uint64_t *den_out);

/* Run the deterministic DATA4 quality-filtering, exact deduplication, and
 * MinHash/LSH + exact-Jaccard near-deduplication pipeline.
 * Empty corpus (doc_count == 0) succeeds with zeroed counts.
 * Rejects aliased input/output buffers with ET_DATA4_ERR_ALIASING. */
et_data4_status et_data4_pipeline_run(
    const et_data4_document *documents, size_t doc_count,
    const et_data4_pipeline_config *config, et_data4_doc_result *results_out,
    size_t results_capacity, et_data4_pipeline_summary *summary_out);

/* Format canonical versioned, LF-terminated ASCII TSV provenance report with
 * trailing SHA-256 checksum record.
 * out == NULL with capacity == 0 queries required byte length (excluding NUL)
 * and computes report_sha256_out (if non-NULL). When out != NULL, capacity
 * must strictly exceed *size_out so the output is NUL-terminated. Input and
 * output spans must not overlap, including the two output spans. */
et_data4_status et_data4_report_format(
    const et_data4_pipeline_config *config,
    const et_data4_pipeline_summary *summary,
    const et_data4_doc_result *results, size_t result_count, char *out,
    size_t capacity, size_t *size_out,
    uint8_t report_sha256_out[ET_DATA4_SHA256_BYTES]);

/* Deterministically parse and verify a canonical DATA4 provenance report:
 * checks ASCII/LF framing, format/version, exact field order and grammar,
 * SHA-256 checksum integrity, source_id uniqueness, ordinal order, aggregate
 * counts/bytes, and canonical duplicate cross-references.
 * config_out, summary_out, and results_out are optional (NULL with
 * results_capacity == 0 validates in place). When results_out != NULL,
 * results_capacity must be >= input_documents. */
et_data4_status et_data4_report_verify(
    const char *report_bytes, size_t report_size,
    et_data4_pipeline_config *config_out,
    et_data4_pipeline_summary *summary_out, et_data4_doc_result *results_out,
    size_t results_capacity);

#ifdef __cplusplus
}
#endif

#endif
