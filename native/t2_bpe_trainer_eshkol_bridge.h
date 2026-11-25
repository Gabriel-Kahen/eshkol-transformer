#ifndef ESHKOL_TRANSFORMER_T2_BPE_TRAINER_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_T2_BPE_TRAINER_ESHKOL_BRIDGE_H

#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_T2_BPE_AOT_API __attribute__((visibility("default")))
#else
#define ET_T2_BPE_AOT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque multi-document corpus builder for compiled Eshkol AOT callers. */
typedef struct et_t2_bpe_aot_corpus et_t2_bpe_aot_corpus;

/* Corpus lifecycle and document insertion (deep-copies input bytes). */
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_create(
    et_t2_bpe_aot_corpus **out_corpus);
ET_T2_BPE_AOT_API void et_t2_bpe_aot_corpus_destroy(
    et_t2_bpe_aot_corpus *corpus);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_add_bytes(
    et_t2_bpe_aot_corpus *corpus, const uint8_t *bytes, int64_t byte_count);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_add_cstr(
    et_t2_bpe_aot_corpus *corpus, const char *text);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_doc_count(
    const et_t2_bpe_aot_corpus *corpus, int64_t *out_count);

/* Deterministic byte-level BPE training, lifecycle, and atomic persistence. */
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_train(
    const et_t2_bpe_aot_corpus *corpus, int64_t target_vocab_size,
    int64_t min_frequency, int64_t special_mask, et_t2_bpe_model **out_model);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_train_bytes(
    const uint8_t *bytes, int64_t byte_count, int64_t target_vocab_size,
    int64_t min_frequency, int64_t special_mask, et_t2_bpe_model **out_model);
ET_T2_BPE_AOT_API void et_t2_bpe_aot_destroy(et_t2_bpe_model *model);

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_save(const et_t2_bpe_model *model,
                                             const char *path,
                                             uint8_t *out_sha256);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_load(const char *path,
                                             et_t2_bpe_model **out_model,
                                             uint8_t *out_sha256);

/* Model metadata inspection.
 * special_ids writes [bos, eos, pad, unk] with -1 for absent specials.
 * get_merge writes [left, right, new_id] into out_triple_3. */
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_vocab_size(
    const et_t2_bpe_model *model, int64_t *out_vocab_size);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_merge_count(
    const et_t2_bpe_model *model, int64_t *out_merge_count);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_special_ids(
    const et_t2_bpe_model *model, int64_t *out_ids_4);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_get_merge(
    const et_t2_bpe_model *model, int64_t rank, int64_t *out_triple_3);

/* Deterministic encode, token count, and 100% lossless byte decode. */
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_encode_bytes(
    const et_t2_bpe_model *model, const uint8_t *bytes, int64_t byte_count,
    uint32_t *out_tokens, int64_t capacity, int64_t *out_count);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_encode_cstr(
    const et_t2_bpe_model *model, const char *text, uint32_t *out_tokens,
    int64_t capacity, int64_t *out_count);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_count_tokens_cstr(
    const et_t2_bpe_model *model, const char *text, int64_t *out_count);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_decode_bytes(
    const et_t2_bpe_model *model, const uint32_t *tokens, int64_t token_count,
    uint8_t *out_bytes, int64_t capacity, int64_t *out_byte_count);

/* Checked buffer marshaling and comparison helpers for Eshkol AOT callers. */
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_store_u8(uint8_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index, int64_t value);
ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_u8(const uint8_t *buffer,
                                                int64_t capacity,
                                                int64_t index);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_store_u32(uint32_t *buffer,
                                                  int64_t capacity,
                                                  int64_t index,
                                                  int64_t value);
ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_u32(const uint32_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index);
ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_i64(const int64_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_bytes_equal(const uint8_t *a,
                                                    const uint8_t *b,
                                                    int64_t count,
                                                    int32_t *equal_out);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_cstr_bytes_equal(const char *cstr,
                                                         const uint8_t *bytes,
                                                         int64_t byte_count,
                                                         int32_t *equal_out);
ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_cstr_len(const char *cstr,
                                                 int64_t *out_len);

#ifdef __cplusplus
}
#endif

#endif
