#ifndef ESHKOL_TRANSFORMER_T2_BPE_TRAINER_ABI_H
#define ESHKOL_TRANSFORMER_T2_BPE_TRAINER_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_T2_BPE_API __attribute__((visibility("default")))
#else
#define ET_T2_BPE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Versioned, SHA-256 checksummed, non-executable binary artifact constants. */
#define ET_T2_BPE_MAGIC "ETBPE100"
#define ET_T2_BPE_MAGIC_LEN 8u
#define ET_T2_BPE_VERSION_MAJOR 1u
#define ET_T2_BPE_VERSION_MINOR 0u
#define ET_T2_BPE_SHA256_BYTES 32u

#define ET_T2_BPE_BASE_VOCAB_SIZE 256u
#define ET_T2_BPE_MIN_VOCAB_SIZE 256u
#define ET_T2_BPE_MAX_VOCAB_SIZE 4096u
#define ET_T2_BPE_MAX_MERGES \
  (ET_T2_BPE_MAX_VOCAB_SIZE - ET_T2_BPE_BASE_VOCAB_SIZE)
#define ET_T2_BPE_MAX_TOKEN_BYTES 256u

#define ET_T2_BPE_HEADER_BYTES 44u
#define ET_T2_BPE_MERGE_RECORD_BYTES 12u
#define ET_T2_BPE_ARTIFACT_MIN_BYTES \
  (ET_T2_BPE_HEADER_BYTES + ET_T2_BPE_SHA256_BYTES)
#define ET_T2_BPE_ARTIFACT_MAX_BYTES                                    \
  (ET_T2_BPE_HEADER_BYTES +                                             \
   (ET_T2_BPE_MAX_MERGES * ET_T2_BPE_MERGE_RECORD_BYTES) +              \
   ET_T2_BPE_SHA256_BYTES)

/* Sentinel ID for an unconfigured optional special token. */
#define ET_T2_BPE_TOKEN_NONE UINT32_MAX

/* Optional special-token bitmask. Enabled specials are assigned contiguous IDs
 * immediately after learned merges in canonical order: BOS, EOS, PAD, UNK. */
enum {
  ET_T2_BPE_SPECIAL_NONE = 0u,
  ET_T2_BPE_SPECIAL_BOS = 1u << 0,
  ET_T2_BPE_SPECIAL_EOS = 1u << 1,
  ET_T2_BPE_SPECIAL_PAD = 1u << 2,
  ET_T2_BPE_SPECIAL_UNK = 1u << 3,
  ET_T2_BPE_SPECIAL_ALL =
      (ET_T2_BPE_SPECIAL_BOS | ET_T2_BPE_SPECIAL_EOS | ET_T2_BPE_SPECIAL_PAD |
       ET_T2_BPE_SPECIAL_UNK)
};

typedef int32_t et_t2_bpe_status;
enum {
  ET_T2_BPE_OK = 0,
  ET_T2_BPE_ERR_ARGUMENT = -1,
  ET_T2_BPE_ERR_BOUNDS = -2,
  ET_T2_BPE_ERR_OVERFLOW = -3,
  ET_T2_BPE_ERR_NOMEM = -4,
  ET_T2_BPE_ERR_MAGIC = -5,
  ET_T2_BPE_ERR_VERSION = -6,
  ET_T2_BPE_ERR_FORMAT = -7,
  ET_T2_BPE_ERR_CHECKSUM = -8,
  ET_T2_BPE_ERR_INVALID_MERGE = -9,
  ET_T2_BPE_ERR_TOKEN_ID = -10,
  ET_T2_BPE_ERR_IO = -11
};

typedef struct {
  const uint8_t *bytes;
  size_t byte_count;
} et_t2_bpe_document;

typedef struct {
  uint32_t target_vocab_size; /* Must be in [256 + popcount(special_mask), 4096]. */
  uint32_t min_frequency;     /* Must be >= 1. */
  uint32_t special_mask;      /* Bitmask of ET_T2_BPE_SPECIAL_*. */
} et_t2_bpe_train_config;

typedef struct {
  uint32_t left;
  uint32_t right;
  uint32_t new_id;
} et_t2_bpe_merge;

typedef struct {
  uint32_t vocab_size;
  uint32_t merge_count;
  uint32_t special_mask;
  uint32_t bos_id;
  uint32_t eos_id;
  uint32_t pad_id;
  uint32_t unk_id;
} et_t2_bpe_info;

typedef struct et_t2_bpe_model et_t2_bpe_model;

/* Train a deterministic byte-level BPE model from raw byte documents.
 * Document boundaries never contribute adjacent pairs. Tie-breaking picks the
 * lowest lexicographic (left_id, right_id) pair. Training stops cleanly when
 * target_vocab_size is reached or no eligible pair meets min_frequency. */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_train(
    const et_t2_bpe_document *docs, size_t doc_count,
    const et_t2_bpe_train_config *config, et_t2_bpe_model **out_model);

ET_T2_BPE_API void et_t2_bpe_destroy(et_t2_bpe_model *model);

/* Model metadata and merge inspection. */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_get_info(
    const et_t2_bpe_model *model, et_t2_bpe_info *out_info);

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_get_merge(
    const et_t2_bpe_model *model, uint32_t rank, et_t2_bpe_merge *out_merge);

/* Query or copy the exact raw byte spelling of token_id.
 * Passing out_bytes == NULL with capacity == 0 queries *out_len only.
 * Special tokens have length 0; out-of-range IDs return ET_T2_BPE_ERR_TOKEN_ID. */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_token_bytes(
    const et_t2_bpe_model *model, uint32_t token_id, uint8_t *out_bytes,
    size_t capacity, size_t *out_len);

/* Deterministic rank-order BPE encode and exact byte-for-byte lossless decode.
 * Passing out_buf == NULL with capacity == 0 queries the required element count.
 * If capacity is too small, *out_count receives the required count and
 * ET_T2_BPE_ERR_BOUNDS is returned without modifying out_buf. */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_encode(
    const et_t2_bpe_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t *out_tokens, size_t capacity, size_t *out_count);

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_encode_ex(
    const et_t2_bpe_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t add_bos, uint32_t add_eos, uint32_t *out_tokens, size_t capacity,
    size_t *out_count);

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_decode(
    const et_t2_bpe_model *model, const uint32_t *tokens, size_t token_count,
    uint8_t *out_bytes, size_t capacity, size_t *out_byte_count);

/* Versioned SHA-256 checksummed binary serialization (ETBPE100 v1.0).
 * out_sha256 is optional (may be NULL). */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_serialize(
    const et_t2_bpe_model *model, uint8_t *out_bytes, size_t capacity,
    size_t *out_size, uint8_t *out_sha256);

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_deserialize(
    const uint8_t *bytes, size_t size, et_t2_bpe_model **out_model,
    uint8_t *out_sha256);

/* Atomic fsync + rename non-executable artifact save (rejects existing
 * symlinks/non-regular targets) and O_NOFOLLOW verified regular-file load. */
ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_save_file(
    const et_t2_bpe_model *model, const char *path, uint8_t *out_sha256);

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_load_file(
    const char *path, et_t2_bpe_model **out_model, uint8_t *out_sha256);

#ifdef __cplusplus
}
#endif

#endif
