#ifndef ESHKOL_TRANSFORMER_UNIGRAM_TOK_ABI_H
#define ESHKOL_TRANSFORMER_UNIGRAM_TOK_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_UNIGRAM_API __attribute__((visibility("default")))
#else
#define ET_UNIGRAM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Versioned, SHA-256 checksummed, non-executable binary artifact constants. */
#define ET_UNIGRAM_MAGIC "ETUG1000"
#define ET_UNIGRAM_MAGIC_LEN 8u
#define ET_UNIGRAM_VERSION_MAJOR 1u
#define ET_UNIGRAM_VERSION_MINOR 0u
#define ET_UNIGRAM_SHA256_BYTES 32u

#define ET_UNIGRAM_BASE_VOCAB_SIZE 256u
#define ET_UNIGRAM_MIN_VOCAB_SIZE 256u
#define ET_UNIGRAM_MAX_VOCAB_SIZE 4096u
#define ET_UNIGRAM_MAX_PIECE_BYTES 256u
#define ET_UNIGRAM_MAX_TRAIN_PIECE_BYTES 64u
#define ET_UNIGRAM_MAX_SEED_VOCAB 16384u
#define ET_UNIGRAM_MAX_INPUT_BYTES 1048576u
#define ET_UNIGRAM_MAX_CORPUS_BYTES 1048576u

#define ET_UNIGRAM_HEADER_BYTES 48u
#define ET_UNIGRAM_PIECE_OVERHEAD_BYTES 10u /* 8B f64 score + 2B u16 len */
#define ET_UNIGRAM_ARTIFACT_MIN_BYTES \
  (ET_UNIGRAM_HEADER_BYTES + (256u * (ET_UNIGRAM_PIECE_OVERHEAD_BYTES + 1u)) + \
   ET_UNIGRAM_SHA256_BYTES)
#define ET_UNIGRAM_ARTIFACT_MAX_BYTES                                          \
  (ET_UNIGRAM_HEADER_BYTES +                                                   \
   (ET_UNIGRAM_MAX_VOCAB_SIZE *                                                \
    (ET_UNIGRAM_PIECE_OVERHEAD_BYTES + ET_UNIGRAM_MAX_PIECE_BYTES)) +          \
   ET_UNIGRAM_SHA256_BYTES)

/* Sentinel ID for an unconfigured optional special token. */
#define ET_UNIGRAM_TOKEN_NONE UINT32_MAX

/* Optional special-token bitmask. Enabled specials receive contiguous IDs
 * immediately after the active vocabulary pieces (0 .. active_count - 1) in
 * canonical order: <bos>, <eos>, <pad>, <unk>. */
enum {
  ET_UNIGRAM_SPECIAL_NONE = 0u,
  ET_UNIGRAM_SPECIAL_BOS = 1u << 0,
  ET_UNIGRAM_SPECIAL_EOS = 1u << 1,
  ET_UNIGRAM_SPECIAL_PAD = 1u << 2,
  ET_UNIGRAM_SPECIAL_UNK = 1u << 3,
  ET_UNIGRAM_SPECIAL_ALL =
      (ET_UNIGRAM_SPECIAL_BOS | ET_UNIGRAM_SPECIAL_EOS |
       ET_UNIGRAM_SPECIAL_PAD | ET_UNIGRAM_SPECIAL_UNK)
};

/* EM M-step update modes. */
enum {
  /* Dirichlet-smoothed normalized MLE/MAP:
   *   log p_i = log(E[c_i] + alpha) - log(sum_j (E[c_j] + alpha))
   * Valid for smoothing_alpha in [1e-6, 10.0]. */
  ET_UNIGRAM_MSTEP_MLE = 0u,
  /* SentencePiece / Kudo (2018) Variational Bayes Digamma + LSE normalization:
   *   u_i = digamma(E[c_i] + alpha) - digamma(sum_j (E[c_j] + alpha))
   *   log p_i = u_i - logsumexp(u)
   * Valid for smoothing_alpha in [0.1, 10.0]. */
  ET_UNIGRAM_MSTEP_DIGAMMA_VB = 1u
};

typedef int32_t et_unigram_status;
enum {
  ET_UNIGRAM_OK = 0,
  ET_UNIGRAM_ERR_ARGUMENT = -1,
  ET_UNIGRAM_ERR_BOUNDS = -2,
  ET_UNIGRAM_ERR_OVERFLOW = -3,
  ET_UNIGRAM_ERR_NOMEM = -4,
  ET_UNIGRAM_ERR_MAGIC = -5,
  ET_UNIGRAM_ERR_VERSION = -6,
  ET_UNIGRAM_ERR_FORMAT = -7,
  ET_UNIGRAM_ERR_CHECKSUM = -8,
  ET_UNIGRAM_ERR_INVALID_VOCAB = -9,
  ET_UNIGRAM_ERR_TOKEN_ID = -10,
  ET_UNIGRAM_ERR_IO = -11,
  ET_UNIGRAM_ERR_NONFINITE = -12,
  ET_UNIGRAM_ERR_FP_ENV = -13
};

typedef struct {
  const uint8_t *bytes;
  size_t byte_count;
} et_unigram_document_v1;

typedef struct {
  const uint8_t *bytes;
  size_t byte_count;
  double log_prob;
} et_unigram_piece_spec_v1;

typedef struct {
  uint32_t struct_size;        /* sizeof(et_unigram_train_config_v1) */
  uint32_t target_vocab_size;  /* [256 + popcount(special_mask), 4096] */
  uint32_t seed_vocab_size;    /* [target_vocab_size - specials, 16384] */
  uint32_t max_piece_bytes;    /* [2, 64] */
  uint32_t min_frequency;      /* >= 1 */
  uint32_t num_em_sub_iters;   /* [1, 32] */
  uint32_t final_em_iters;     /* [1, 64] */
  uint32_t mstep_mode;         /* ET_UNIGRAM_MSTEP_MLE or _DIGAMMA_VB */
  uint32_t special_mask;       /* Bitmask of ET_UNIGRAM_SPECIAL_* */
  uint32_t reserved;           /* Must be 0 */
  double shrinking_factor;     /* (0.1, 0.99] */
  double smoothing_alpha;      /* [1e-6, 10.0] (MLE) or [0.1, 10.0] (VB) */
} et_unigram_train_config_v1;

typedef struct {
  uint32_t struct_size;         /* sizeof(et_unigram_train_stats_v1) */
  uint32_t seed_active_count;   /* 256 + extracted seed multi-byte pieces */
  uint32_t final_active_count;  /* 256 + final learned pieces */
  uint32_t pruning_rounds;      /* Pruning rounds executed */
  uint32_t total_em_iters;      /* Total EM iterations executed */
  uint32_t reserved;            /* 0 */
  double initial_log_likelihood;/* Marginal log-likelihood of initial seed */
  double final_em_start_ll;     /* Marginal LL at start of final EM polish */
  double final_em_end_ll;       /* Marginal LL of final canonical model */
} et_unigram_train_stats_v1;

typedef struct {
  uint32_t struct_size;     /* sizeof(et_unigram_info_v1) */
  uint32_t vocab_size;      /* active_count + popcount(special_mask) */
  uint32_t active_count;    /* 256 + learned_count */
  uint32_t learned_count;   /* Number of multi-byte pieces (len >= 2) */
  uint32_t special_mask;    /* Bitmask of ET_UNIGRAM_SPECIAL_* */
  uint32_t bos_id;          /* Token ID or ET_UNIGRAM_TOKEN_NONE */
  uint32_t eos_id;          /* Token ID or ET_UNIGRAM_TOKEN_NONE */
  uint32_t pad_id;          /* Token ID or ET_UNIGRAM_TOKEN_NONE */
  uint32_t unk_id;          /* Token ID or ET_UNIGRAM_TOKEN_NONE */
  uint32_t max_piece_bytes; /* Maximum byte length among active pieces */
} et_unigram_info_v1;

typedef struct et_unigram_model et_unigram_model;

/* Direct model construction from explicit pieces.
 * Pieces 0..255 must be exact single-byte fallbacks [0x00]..[0xFF].
 * Pieces 256..piece_count-1 must have length in [2, 256], strictly unique byte
 * strings, and finite log_prob <= 0.0. */
ET_UNIGRAM_API et_unigram_status et_unigram_create_v1(
    const et_unigram_piece_spec_v1 *pieces, size_t piece_count,
    uint32_t special_mask, et_unigram_model **out_model);

/* Train a deterministic byte-complete Unigram LM tokenizer via seed substring
 * extraction, Forward-Backward EM, per-piece pruning loss, and final EM polish.
 * out_stats is optional (may be NULL). */
ET_UNIGRAM_API et_unigram_status et_unigram_train_v1(
    const et_unigram_document_v1 *docs, size_t doc_count,
    const et_unigram_train_config_v1 *config, et_unigram_model **out_model,
    et_unigram_train_stats_v1 *out_stats);

ET_UNIGRAM_API void et_unigram_destroy_v1(et_unigram_model *model);

/* Model metadata and per-token inspection. */
ET_UNIGRAM_API et_unigram_status et_unigram_get_info_v1(
    const et_unigram_model *model, et_unigram_info_v1 *out_info);

/* Query/copy raw byte spelling and log-probability of token_id.
 * Passing out_bytes == NULL with capacity == 0 queries *out_len / *out_log_prob.
 * Special tokens have length 0 and log_prob 0.0. */
ET_UNIGRAM_API et_unigram_status et_unigram_get_piece_v1(
    const et_unigram_model *model, uint32_t token_id, uint8_t *out_bytes,
    size_t capacity, size_t *out_len, double *out_log_prob);

/* Exact Viterbi lattice DP encoding maximizing sum_j log P(p_j).
 * Deterministic total-order tie-break at each end position:
 *   1. higher cumulative log-prob score,
 *   2. longer piece ending at that position,
 *   3. smaller token ID.
 * Passing out_tokens == NULL with capacity == 0 queries *out_count / *out_score.
 * On insufficient capacity, *out_count receives required count and out_tokens
 * is left unmodified. out_score is optional (may be NULL). */
ET_UNIGRAM_API et_unigram_status et_unigram_encode_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t *out_tokens, size_t capacity, size_t *out_count,
    double *out_score);

ET_UNIGRAM_API et_unigram_status et_unigram_encode_ex_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t add_bos, uint32_t add_eos, uint32_t *out_tokens, size_t capacity,
    size_t *out_count, double *out_score);

/* Exact 100% lossless byte decode. Special tokens emit 0 bytes.
 * Any out-of-range token ID fails with ET_UNIGRAM_ERR_TOKEN_ID without
 * modifying out_bytes. */
ET_UNIGRAM_API et_unigram_status et_unigram_decode_v1(
    const et_unigram_model *model, const uint32_t *tokens, size_t token_count,
    uint8_t *out_bytes, size_t capacity, size_t *out_byte_count);

/* Exact log-sum-exp Forward-Backward lattice DP for one byte sequence.
 * Computes exact marginal log-likelihood log Z = log sum_{x} exp(sum_j log P(x_j))
 * and exact expected piece counts E[c_i] = d(log Z) / d(log P(i)) for
 * i in [0, active_count). out_expected_counts may be NULL when count_capacity == 0. */
ET_UNIGRAM_API et_unigram_status et_unigram_forward_backward_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    double *out_log_z, double *out_expected_counts, size_t count_capacity);

/* Exact Forward-Backward summed across a corpus of documents. */
ET_UNIGRAM_API et_unigram_status et_unigram_corpus_fb_v1(
    const et_unigram_model *model, const et_unigram_document_v1 *docs,
    size_t doc_count, double *out_log_z, double *out_expected_counts,
    size_t count_capacity);

/* Deterministic Kudo (2018) Forward-Filtering Backward-Sampling (FFBS) subword
 * regularization with temperature scaling alpha_temp > 0.0 and SplitMix64 seed.
 * Always yields a valid lossless segmentation of bytes. */
ET_UNIGRAM_API et_unigram_status et_unigram_sample_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    double alpha_temp, uint64_t seed, uint32_t *out_tokens, size_t capacity,
    size_t *out_count, double *out_score);

/* Versioned SHA-256 checksummed binary serialization (ETUG1000 v1.0).
 * out_sha256 is optional (may be NULL). */
ET_UNIGRAM_API et_unigram_status et_unigram_serialize_v1(
    const et_unigram_model *model, uint8_t *out_bytes, size_t capacity,
    size_t *out_size, uint8_t *out_sha256);

ET_UNIGRAM_API et_unigram_status et_unigram_deserialize_v1(
    const uint8_t *bytes, size_t size, et_unigram_model **out_model,
    uint8_t *out_sha256);

/* Failure-atomic 0600 temp + fsync + rename save (rejects symlinks/non-regular
 * destinations) and O_NOFOLLOW verified regular-file load. */
ET_UNIGRAM_API et_unigram_status et_unigram_save_file_v1(
    const et_unigram_model *model, const char *path, uint8_t *out_sha256);

ET_UNIGRAM_API et_unigram_status et_unigram_load_file_v1(
    const char *path, et_unigram_model **out_model, uint8_t *out_sha256);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_UNIGRAM_TOK_ABI_H */
