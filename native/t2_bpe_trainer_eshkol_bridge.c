#define _POSIX_C_SOURCE 200809L

#include "t2_bpe_trainer_eshkol_bridge.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "T2 BPE AOT bridge requires 8-bit bytes and at most 64-bit size_t");

typedef struct {
  uint8_t *bytes;
  size_t byte_count;
} owned_doc;

struct et_t2_bpe_aot_corpus {
  owned_doc *docs;
  size_t count;
  size_t capacity;
};

static int i64_to_size(int64_t value, size_t *out) {
  if (value < 0) return 0;
  if ((uint64_t)value > (uint64_t)SIZE_MAX) return 0;
  *out = (size_t)value;
  return 1;
}

static int i64_to_u32(int64_t value, uint32_t *out) {
  if (value < 0 || value > (int64_t)UINT32_MAX) return 0;
  *out = (uint32_t)value;
  return 1;
}

static int size_to_i64(size_t value, int64_t *out) {
  if ((uint64_t)value > (uint64_t)INT64_MAX) return 0;
  *out = (int64_t)value;
  return 1;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_create(
    et_t2_bpe_aot_corpus **out_corpus) {
  if (out_corpus == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  et_t2_bpe_aot_corpus *corpus = calloc(1u, sizeof(*corpus));
  if (corpus == NULL) return ET_T2_BPE_ERR_NOMEM;
  *out_corpus = corpus;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API void et_t2_bpe_aot_corpus_destroy(
    et_t2_bpe_aot_corpus *corpus) {
  if (corpus == NULL) return;
  for (size_t i = 0u; i < corpus->count; ++i) {
    free(corpus->docs[i].bytes);
  }
  free(corpus->docs);
  free(corpus);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_add_bytes(
    et_t2_bpe_aot_corpus *corpus, const uint8_t *bytes, int64_t byte_count) {
  size_t n = 0u;
  if (corpus == NULL || !i64_to_size(byte_count, &n)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (n != 0u && bytes == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (n > (size_t)PTRDIFF_MAX) return ET_T2_BPE_ERR_OVERFLOW;

  if (corpus->count == corpus->capacity) {
    size_t new_cap = corpus->capacity == 0u ? 8u : corpus->capacity * 2u;
    if (new_cap < corpus->capacity ||
        new_cap > (size_t)PTRDIFF_MAX / sizeof(*corpus->docs)) {
      return ET_T2_BPE_ERR_OVERFLOW;
    }
    owned_doc *grown = realloc(corpus->docs, new_cap * sizeof(*grown));
    if (grown == NULL) return ET_T2_BPE_ERR_NOMEM;
    corpus->docs = grown;
    corpus->capacity = new_cap;
  }

  uint8_t *copy = NULL;
  if (n != 0u) {
    copy = malloc(n);
    if (copy == NULL) return ET_T2_BPE_ERR_NOMEM;
    memcpy(copy, bytes, n);
  }
  corpus->docs[corpus->count].bytes = copy;
  corpus->docs[corpus->count].byte_count = n;
  corpus->count += 1u;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_add_cstr(
    et_t2_bpe_aot_corpus *corpus, const char *text) {
  if (corpus == NULL || text == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  const size_t n = strlen(text);
  int64_t n64 = 0;
  if (!size_to_i64(n, &n64)) return ET_T2_BPE_ERR_OVERFLOW;
  return et_t2_bpe_aot_corpus_add_bytes(corpus, (const uint8_t *)text, n64);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_corpus_doc_count(
    const et_t2_bpe_aot_corpus *corpus, int64_t *out_count) {
  if (corpus == NULL || out_count == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (!size_to_i64(corpus->count, out_count)) return ET_T2_BPE_ERR_OVERFLOW;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_train(
    const et_t2_bpe_aot_corpus *corpus, int64_t target_vocab_size,
    int64_t min_frequency, int64_t special_mask, et_t2_bpe_model **out_model) {
  if (corpus == NULL || out_model == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  et_t2_bpe_train_config cfg;
  if (!i64_to_u32(target_vocab_size, &cfg.target_vocab_size) ||
      !i64_to_u32(min_frequency, &cfg.min_frequency) ||
      !i64_to_u32(special_mask, &cfg.special_mask)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }

  et_t2_bpe_document *views = NULL;
  if (corpus->count != 0u) {
    if (corpus->count > (size_t)PTRDIFF_MAX / sizeof(*views)) {
      return ET_T2_BPE_ERR_OVERFLOW;
    }
    views = malloc(corpus->count * sizeof(*views));
    if (views == NULL) return ET_T2_BPE_ERR_NOMEM;
    for (size_t i = 0u; i < corpus->count; ++i) {
      views[i].bytes = corpus->docs[i].bytes;
      views[i].byte_count = corpus->docs[i].byte_count;
    }
  }

  const int32_t st = et_t2_bpe_train(views, corpus->count, &cfg, out_model);
  free(views);
  return st;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_train_bytes(
    const uint8_t *bytes, int64_t byte_count, int64_t target_vocab_size,
    int64_t min_frequency, int64_t special_mask, et_t2_bpe_model **out_model) {
  size_t n = 0u;
  if (out_model == NULL || !i64_to_size(byte_count, &n)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (n != 0u && bytes == NULL) return ET_T2_BPE_ERR_ARGUMENT;

  et_t2_bpe_train_config cfg;
  if (!i64_to_u32(target_vocab_size, &cfg.target_vocab_size) ||
      !i64_to_u32(min_frequency, &cfg.min_frequency) ||
      !i64_to_u32(special_mask, &cfg.special_mask)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  const et_t2_bpe_document doc = {bytes, n};
  return et_t2_bpe_train(&doc, 1u, &cfg, out_model);
}

ET_T2_BPE_AOT_API void et_t2_bpe_aot_destroy(et_t2_bpe_model *model) {
  et_t2_bpe_destroy(model);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_save(const et_t2_bpe_model *model,
                                             const char *path,
                                             uint8_t *out_sha256) {
  return et_t2_bpe_save_file(model, path, out_sha256);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_load(const char *path,
                                             et_t2_bpe_model **out_model,
                                             uint8_t *out_sha256) {
  return et_t2_bpe_load_file(path, out_model, out_sha256);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_vocab_size(
    const et_t2_bpe_model *model, int64_t *out_vocab_size) {
  if (model == NULL || out_vocab_size == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  et_t2_bpe_info info;
  const int32_t st = et_t2_bpe_get_info(model, &info);
  if (st != ET_T2_BPE_OK) return st;
  *out_vocab_size = (int64_t)info.vocab_size;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_merge_count(
    const et_t2_bpe_model *model, int64_t *out_merge_count) {
  if (model == NULL || out_merge_count == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  et_t2_bpe_info info;
  const int32_t st = et_t2_bpe_get_info(model, &info);
  if (st != ET_T2_BPE_OK) return st;
  *out_merge_count = (int64_t)info.merge_count;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_special_ids(
    const et_t2_bpe_model *model, int64_t *out_ids_4) {
  if (model == NULL || out_ids_4 == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  et_t2_bpe_info info;
  const int32_t st = et_t2_bpe_get_info(model, &info);
  if (st != ET_T2_BPE_OK) return st;
  out_ids_4[0] =
      info.bos_id == ET_T2_BPE_TOKEN_NONE ? -1 : (int64_t)info.bos_id;
  out_ids_4[1] =
      info.eos_id == ET_T2_BPE_TOKEN_NONE ? -1 : (int64_t)info.eos_id;
  out_ids_4[2] =
      info.pad_id == ET_T2_BPE_TOKEN_NONE ? -1 : (int64_t)info.pad_id;
  out_ids_4[3] =
      info.unk_id == ET_T2_BPE_TOKEN_NONE ? -1 : (int64_t)info.unk_id;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_get_merge(
    const et_t2_bpe_model *model, int64_t rank, int64_t *out_triple_3) {
  uint32_t r = 0u;
  if (model == NULL || out_triple_3 == NULL || !i64_to_u32(rank, &r)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  et_t2_bpe_merge m;
  const int32_t st = et_t2_bpe_get_merge(model, r, &m);
  if (st != ET_T2_BPE_OK) return st;
  out_triple_3[0] = (int64_t)m.left;
  out_triple_3[1] = (int64_t)m.right;
  out_triple_3[2] = (int64_t)m.new_id;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_encode_bytes(
    const et_t2_bpe_model *model, const uint8_t *bytes, int64_t byte_count,
    uint32_t *out_tokens, int64_t capacity, int64_t *out_count) {
  size_t n = 0u;
  size_t cap = 0u;
  if (out_count == NULL || !i64_to_size(byte_count, &n) ||
      !i64_to_size(capacity, &cap)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  size_t written = 0u;
  const int32_t st =
      et_t2_bpe_encode(model, bytes, n, out_tokens, cap, &written);
  if (st == ET_T2_BPE_OK || st == ET_T2_BPE_ERR_BOUNDS) {
    if (!size_to_i64(written, out_count)) return ET_T2_BPE_ERR_OVERFLOW;
  }
  return st;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_encode_cstr(
    const et_t2_bpe_model *model, const char *text, uint32_t *out_tokens,
    int64_t capacity, int64_t *out_count) {
  if (text == NULL || out_count == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  const size_t n = strlen(text);
  int64_t n64 = 0;
  if (!size_to_i64(n, &n64)) return ET_T2_BPE_ERR_OVERFLOW;
  return et_t2_bpe_aot_encode_bytes(model, (const uint8_t *)text, n64,
                                    out_tokens, capacity, out_count);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_count_tokens_cstr(
    const et_t2_bpe_model *model, const char *text, int64_t *out_count) {
  return et_t2_bpe_aot_encode_cstr(model, text, NULL, 0, out_count);
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_decode_bytes(
    const et_t2_bpe_model *model, const uint32_t *tokens, int64_t token_count,
    uint8_t *out_bytes, int64_t capacity, int64_t *out_byte_count) {
  size_t n = 0u;
  size_t cap = 0u;
  if (out_byte_count == NULL || !i64_to_size(token_count, &n) ||
      !i64_to_size(capacity, &cap)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  size_t written = 0u;
  const int32_t st =
      et_t2_bpe_decode(model, tokens, n, out_bytes, cap, &written);
  if (st == ET_T2_BPE_OK || st == ET_T2_BPE_ERR_BOUNDS) {
    if (!size_to_i64(written, out_byte_count)) return ET_T2_BPE_ERR_OVERFLOW;
  }
  return st;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_store_u8(uint8_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index, int64_t value) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity ||
      value < 0 || value > 255) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  buffer[(size_t)index] = (uint8_t)value;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_u8(const uint8_t *buffer,
                                                int64_t capacity,
                                                int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return -1;
  }
  return (int64_t)buffer[(size_t)index];
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_store_u32(uint32_t *buffer,
                                                  int64_t capacity,
                                                  int64_t index,
                                                  int64_t value) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity ||
      value < 0 || value > (int64_t)UINT32_MAX) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  buffer[(size_t)index] = (uint32_t)value;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_u32(const uint32_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return -1;
  }
  return (int64_t)buffer[(size_t)index];
}

ET_T2_BPE_AOT_API int64_t et_t2_bpe_aot_load_i64(const int64_t *buffer,
                                                 int64_t capacity,
                                                 int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return 0;
  }
  return buffer[(size_t)index];
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_bytes_equal(const uint8_t *a,
                                                    const uint8_t *b,
                                                    int64_t count,
                                                    int32_t *equal_out) {
  size_t n = 0u;
  if (equal_out == NULL || !i64_to_size(count, &n)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (n != 0u && (a == NULL || b == NULL)) return ET_T2_BPE_ERR_ARGUMENT;
  *equal_out = (n == 0u || memcmp(a, b, n) == 0) ? 1 : 0;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_cstr_bytes_equal(const char *cstr,
                                                         const uint8_t *bytes,
                                                         int64_t byte_count,
                                                         int32_t *equal_out) {
  size_t n = 0u;
  if (cstr == NULL || equal_out == NULL || !i64_to_size(byte_count, &n)) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (n != 0u && bytes == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  const size_t slen = strlen(cstr);
  if (slen != n) {
    *equal_out = 0;
    return ET_T2_BPE_OK;
  }
  *equal_out = (n == 0u || memcmp(cstr, bytes, n) == 0) ? 1 : 0;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_AOT_API int32_t et_t2_bpe_aot_cstr_len(const char *cstr,
                                                 int64_t *out_len) {
  if (cstr == NULL || out_len == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (!size_to_i64(strlen(cstr), out_len)) return ET_T2_BPE_ERR_OVERFLOW;
  return ET_T2_BPE_OK;
}
