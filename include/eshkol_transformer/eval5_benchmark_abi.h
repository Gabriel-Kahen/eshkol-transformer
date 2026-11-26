#ifndef ESHKOL_TRANSFORMER_EVAL5_BENCHMARK_ABI_H
#define ESHKOL_TRANSFORMER_EVAL5_BENCHMARK_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EVAL5_MANIFEST_V1 "EVAL5_MANIFEST_V1"
#define ET_EVAL5_VERSION_MAJOR 1u
#define ET_EVAL5_VERSION_MINOR 0u
#define ET_EVAL5_SHA256_BYTES 32u
#define ET_EVAL5_NAME_MAX 127u
#define ET_EVAL5_MANIFEST_MAX_BYTES 4096u

typedef int32_t et_eval5_status;
enum {
  ET_EVAL5_OK = 0,
  ET_EVAL5_ERR_ARGUMENT = -1,
  ET_EVAL5_ERR_BOUNDS = -2,
  ET_EVAL5_ERR_OVERFLOW = -3,
  ET_EVAL5_ERR_NOMEM = -4,
  ET_EVAL5_ERR_NONFINITE = -5,
  ET_EVAL5_ERR_TARGET = -6,
  ET_EVAL5_ERR_CONTAMINATED = -7,
  ET_EVAL5_ERR_FORMAT = -8,
  ET_EVAL5_ERR_CHECKSUM = -9,
  ET_EVAL5_ERR_VERSION = -10,
  ET_EVAL5_ERR_IO = -11,
  ET_EVAL5_ERR_FP_ENV = -12,
  ET_EVAL5_ERR_IDENTITY = -13
};

enum {
  ET_EVAL5_NGRAM_BYTES = 1,
  ET_EVAL5_NGRAM_TOKENS = 2
};

enum {
  ET_EVAL5_METRIC_LM = 1,
  ET_EVAL5_METRIC_MC = 2
};

typedef struct {
  const uint8_t *bytes;
  size_t byte_count;
  const uint32_t *tokens;
  size_t token_count;
} et_eval5_document;

typedef struct {
  uint32_t ngram_mode;
  uint32_t ngram_order;
  uint32_t forbid_exact_match;
  double min_ngram_overlap;
  double max_allowed_rate;
} et_eval5_contamination_config;

typedef struct {
  uint64_t eval_documents;
  uint64_t contaminated_documents;
  uint64_t exact_doc_matches;
  uint64_t ngram_contaminated_docs;
  uint64_t total_eval_ngrams;
  uint64_t matched_eval_ngrams;
  double contamination_rate;
} et_eval5_contamination_summary;

/* The index buffer has room for at least eval_count zero-based indices.
 * OK and ERR_CONTAMINATED both return the summary and its sorted union of
 * matching indices. All other errors leave both outputs untouched. */
et_eval5_status et_eval5_audit_contamination(
    const et_eval5_document *train, size_t train_count,
    const et_eval5_document *eval, size_t eval_count,
    const et_eval5_contamination_config *config,
    uint64_t *contaminated_indices, size_t index_capacity,
    et_eval5_contamination_summary *out);

typedef struct {
  /* Contiguous row-major logits [token_count, vocab_size] and token_count
   * target IDs. logit_count is the exact number of available float elements. */
  const float *logits;
  const uint32_t *target_ids;
  size_t token_count;
  size_t vocab_size;
  size_t logit_count;
} et_eval5_sequence;

typedef struct {
  uint64_t token_count;
  uint64_t byte_count;
  double total_nll;
  double mean_nll;
  double perplexity;
  double bits_per_byte;
} et_eval5_lm_metrics;

typedef struct {
  const et_eval5_sequence *choices;
  size_t choice_count;
  size_t correct_index;
} et_eval5_choice_item;

typedef struct {
  uint64_t correct_count;
  uint64_t total_count;
  double accuracy;
} et_eval5_mc_metrics;

/* Accept only canonical T1/T2 public fingerprints, require an exact match
 * with the authenticated D1 corpus summary, including vocabulary size, and
 * decode the identity digest into the manifest's tokenizer_sha256 field.
 * Both input spans have explicit
 * lengths; failure leaves out_sha256 untouched. The caller must obtain both
 * fingerprints from their validated owners in the same evaluation run. */
et_eval5_status et_eval5_bind_tokenizer_identity(
    const uint8_t *tokenizer_fingerprint, size_t tokenizer_length,
    uint32_t tokenizer_vocab_size,
    const uint8_t *corpus_fingerprint, size_t corpus_length,
    uint32_t corpus_vocab_size,
    uint8_t out_sha256[ET_EVAL5_SHA256_BYTES]);

/* byte_count is the number of original bytes represented by the target
 * tokens. Metrics and the audit require round-to-nearest floating point
 * with masked exceptions and gradual underflow. */
et_eval5_status et_eval5_score_lm(const et_eval5_sequence *sequence,
                                 uint64_t byte_count, et_eval5_lm_metrics *out);
et_eval5_status et_eval5_score_mc(const et_eval5_choice_item *items,
                                 size_t item_count, uint32_t normalize_by_length,
                                 et_eval5_mc_metrics *out);

typedef struct {
  uint32_t version_major;
  uint32_t version_minor;
  char suite[ET_EVAL5_NAME_MAX + 1u];
  char task[ET_EVAL5_NAME_MAX + 1u];
  char split[ET_EVAL5_NAME_MAX + 1u];
  uint8_t train_dataset_sha256[ET_EVAL5_SHA256_BYTES];
  uint8_t eval_dataset_sha256[ET_EVAL5_SHA256_BYTES];
  uint8_t tokenizer_sha256[ET_EVAL5_SHA256_BYTES];
  uint8_t checkpoint_sha256[ET_EVAL5_SHA256_BYTES];
  uint64_t context_length;
  uint64_t max_samples;
  et_eval5_contamination_config contamination;
  uint32_t metric_mask;
  uint32_t mc_normalize_by_length;
  et_eval5_lm_metrics lm;
  et_eval5_mc_metrics mc;
  et_eval5_contamination_summary audit;
  /* Populated by parse/read. Format/write recompute the checksum. */
  uint8_t manifest_sha256[ET_EVAL5_SHA256_BYTES];
} et_eval5_manifest;

/* out == NULL with capacity == 0 queries the required byte count. size
 * excludes the terminating NUL. For an actual output, capacity must exceed
 * size. ERR_BOUNDS returns the required size without writing the buffer. */
et_eval5_status et_eval5_manifest_format(const et_eval5_manifest *manifest,
                                        char *out, size_t capacity, size_t *size);
et_eval5_status et_eval5_manifest_parse(const char *bytes, size_t size,
                                       et_eval5_manifest *out);
et_eval5_status et_eval5_manifest_write_atomic(const char *path,
                                              const et_eval5_manifest *manifest);
et_eval5_status et_eval5_manifest_read_file(const char *path,
                                           et_eval5_manifest *out);

#ifdef __cplusplus
}
#endif
#endif
