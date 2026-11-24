#define _POSIX_C_SOURCE 200809L

#include "unigram_tok_eshkol_bridge.h"

#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

struct et_unigram_aot_corpus {
  et_unigram_document_v1 *docs;
  uint8_t **owned_bufs;
  size_t count;
  size_t capacity;
};

ET_UNIGRAM_AOT_API int32_t
et_unigram_aot_corpus_create(et_unigram_aot_corpus **out_corpus) {
  if (out_corpus == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  *out_corpus = NULL;
  et_unigram_aot_corpus *c = calloc(1u, sizeof(et_unigram_aot_corpus));
  if (c == NULL) return ET_UNIGRAM_ERR_NOMEM;
  *out_corpus = c;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API void
et_unigram_aot_corpus_destroy(et_unigram_aot_corpus *corpus) {
  if (corpus == NULL) return;
  for (size_t i = 0u; i < corpus->count; ++i) {
    free(corpus->owned_bufs[i]);
  }
  free(corpus->owned_bufs);
  free(corpus->docs);
  free(corpus);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_add_bytes(
    et_unigram_aot_corpus *corpus, const uint8_t *bytes, int64_t byte_count) {
  if (corpus == NULL || byte_count < 0) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count > 0 && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;

  if (corpus->count == corpus->capacity) {
    const size_t new_cap = (corpus->capacity == 0u) ? 8u : corpus->capacity * 2u;
    if (new_cap < corpus->capacity) return ET_UNIGRAM_ERR_OVERFLOW;
    et_unigram_document_v1 *nd =
        realloc(corpus->docs, new_cap * sizeof(et_unigram_document_v1));
    if (nd == NULL) return ET_UNIGRAM_ERR_NOMEM;
    corpus->docs = nd;
    uint8_t **nb = realloc(corpus->owned_bufs, new_cap * sizeof(uint8_t *));
    if (nb == NULL) return ET_UNIGRAM_ERR_NOMEM;
    corpus->owned_bufs = nb;
    corpus->capacity = new_cap;
  }

  uint8_t *copy = NULL;
  if (byte_count > 0) {
    copy = malloc((size_t)byte_count);
    if (copy == NULL) return ET_UNIGRAM_ERR_NOMEM;
    memcpy(copy, bytes, (size_t)byte_count);
  }
  corpus->owned_bufs[corpus->count] = copy;
  corpus->docs[corpus->count].bytes = copy;
  corpus->docs[corpus->count].byte_count = (size_t)byte_count;
  corpus->count += 1u;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t
et_unigram_aot_corpus_add_cstr(et_unigram_aot_corpus *corpus,
                               const char *text) {
  if (corpus == NULL || text == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const size_t len = strlen(text);
  if (len > (size_t)INT64_MAX) return ET_UNIGRAM_ERR_OVERFLOW;
  return et_unigram_aot_corpus_add_bytes(corpus, (const uint8_t *)text,
                                         (int64_t)len);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_corpus_doc_count(
    const et_unigram_aot_corpus *corpus, int64_t *out_count) {
  if (corpus == NULL || out_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  *out_count = (int64_t)corpus->count;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_train(
    const et_unigram_aot_corpus *corpus, int64_t target_vocab_size,
    int64_t seed_vocab_size, int64_t max_piece_bytes, int64_t min_frequency,
    int64_t num_em_sub_iters, int64_t final_em_iters, int64_t mstep_mode,
    int64_t special_mask, double shrinking_factor, double smoothing_alpha,
    et_unigram_model **out_model, int64_t *out_stats_i64_4,
    double *out_stats_f64_3) {
  if (out_model != NULL) *out_model = NULL;
  if (corpus == NULL || out_model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (target_vocab_size < 0 || target_vocab_size > (int64_t)UINT32_MAX ||
      seed_vocab_size < 0 || seed_vocab_size > (int64_t)UINT32_MAX ||
      max_piece_bytes < 0 || max_piece_bytes > (int64_t)UINT32_MAX ||
      min_frequency < 0 || min_frequency > (int64_t)UINT32_MAX ||
      num_em_sub_iters < 0 || num_em_sub_iters > (int64_t)UINT32_MAX ||
      final_em_iters < 0 || final_em_iters > (int64_t)UINT32_MAX ||
      mstep_mode < 0 || mstep_mode > (int64_t)UINT32_MAX ||
      special_mask < 0 || special_mask > (int64_t)UINT32_MAX) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }

  et_unigram_train_config_v1 cfg;
  cfg.struct_size = (uint32_t)sizeof(et_unigram_train_config_v1);
  cfg.target_vocab_size = (uint32_t)target_vocab_size;
  cfg.seed_vocab_size = (uint32_t)seed_vocab_size;
  cfg.max_piece_bytes = (uint32_t)max_piece_bytes;
  cfg.min_frequency = (uint32_t)min_frequency;
  cfg.num_em_sub_iters = (uint32_t)num_em_sub_iters;
  cfg.final_em_iters = (uint32_t)final_em_iters;
  cfg.mstep_mode = (uint32_t)mstep_mode;
  cfg.special_mask = (uint32_t)special_mask;
  cfg.reserved = 0u;
  cfg.shrinking_factor = shrinking_factor;
  cfg.smoothing_alpha = smoothing_alpha;

  et_unigram_train_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  const et_unigram_status st = et_unigram_train_v1(
      corpus->docs, corpus->count, &cfg, out_model, &stats);
  if (st != ET_UNIGRAM_OK) return st;

  if (out_stats_i64_4 != NULL) {
    out_stats_i64_4[0] = (int64_t)stats.seed_active_count;
    out_stats_i64_4[1] = (int64_t)stats.final_active_count;
    out_stats_i64_4[2] = (int64_t)stats.pruning_rounds;
    out_stats_i64_4[3] = (int64_t)stats.total_em_iters;
  }
  if (out_stats_f64_3 != NULL) {
    out_stats_f64_3[0] = stats.initial_log_likelihood;
    out_stats_f64_3[1] = stats.final_em_start_ll;
    out_stats_f64_3[2] = stats.final_em_end_ll;
  }
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API void et_unigram_aot_destroy(et_unigram_model *model) {
  et_unigram_destroy_v1(model);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_save(const et_unigram_model *model,
                                               const char *path,
                                               uint8_t *out_sha256) {
  return et_unigram_save_file_v1(model, path, out_sha256);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_load(const char *path,
                                               et_unigram_model **out_model,
                                               uint8_t *out_sha256) {
  return et_unigram_load_file_v1(path, out_model, out_sha256);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_vocab_size(
    const et_unigram_model *model, int64_t *out_vocab_size) {
  if (model == NULL || out_vocab_size == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  et_unigram_info_v1 info;
  const et_unigram_status st = et_unigram_get_info_v1(model, &info);
  if (st != ET_UNIGRAM_OK) return st;
  *out_vocab_size = (int64_t)info.vocab_size;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_active_count(
    const et_unigram_model *model, int64_t *out_active_count) {
  if (model == NULL || out_active_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  et_unigram_info_v1 info;
  const et_unigram_status st = et_unigram_get_info_v1(model, &info);
  if (st != ET_UNIGRAM_OK) return st;
  *out_active_count = (int64_t)info.active_count;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_learned_count(
    const et_unigram_model *model, int64_t *out_learned_count) {
  if (model == NULL || out_learned_count == NULL) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  et_unigram_info_v1 info;
  const et_unigram_status st = et_unigram_get_info_v1(model, &info);
  if (st != ET_UNIGRAM_OK) return st;
  *out_learned_count = (int64_t)info.learned_count;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_special_ids(
    const et_unigram_model *model, int64_t *out_ids_4) {
  if (model == NULL || out_ids_4 == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  et_unigram_info_v1 info;
  const et_unigram_status st = et_unigram_get_info_v1(model, &info);
  if (st != ET_UNIGRAM_OK) return st;
  out_ids_4[0] = (info.bos_id == ET_UNIGRAM_TOKEN_NONE) ? -1 : (int64_t)info.bos_id;
  out_ids_4[1] = (info.eos_id == ET_UNIGRAM_TOKEN_NONE) ? -1 : (int64_t)info.eos_id;
  out_ids_4[2] = (info.pad_id == ET_UNIGRAM_TOKEN_NONE) ? -1 : (int64_t)info.pad_id;
  out_ids_4[3] = (info.unk_id == ET_UNIGRAM_TOKEN_NONE) ? -1 : (int64_t)info.unk_id;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_piece_log_prob(
    const et_unigram_model *model, int64_t token_id, double *out_log_prob) {
  if (model == NULL || out_log_prob == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (token_id < 0 || token_id > (int64_t)UINT32_MAX) {
    return ET_UNIGRAM_ERR_TOKEN_ID;
  }
  return et_unigram_get_piece_v1(model, (uint32_t)token_id, NULL, 0u, NULL,
                                 out_log_prob);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_encode_bytes(
    const et_unigram_model *model, const uint8_t *bytes, int64_t byte_count,
    uint32_t *out_tokens, int64_t capacity, int64_t *out_count,
    double *out_score) {
  if (model == NULL || out_count == NULL || byte_count < 0 || capacity < 0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  size_t cnt = 0u;
  const et_unigram_status st = et_unigram_encode_v1(
      model, bytes, (size_t)byte_count, out_tokens, (size_t)capacity, &cnt,
      out_score);
  *out_count = (int64_t)cnt;
  return st;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_encode_cstr(
    const et_unigram_model *model, const char *text, uint32_t *out_tokens,
    int64_t capacity, int64_t *out_count, double *out_score) {
  if (text == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const size_t len = strlen(text);
  if (len > (size_t)INT64_MAX) return ET_UNIGRAM_ERR_OVERFLOW;
  return et_unigram_aot_encode_bytes(model, (const uint8_t *)text, (int64_t)len,
                                     out_tokens, capacity, out_count,
                                     out_score);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_decode_bytes(
    const et_unigram_model *model, const uint32_t *tokens, int64_t token_count,
    uint8_t *out_bytes, int64_t capacity, int64_t *out_byte_count) {
  if (model == NULL || out_byte_count == NULL || token_count < 0 ||
      capacity < 0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  size_t cnt = 0u;
  const et_unigram_status st = et_unigram_decode_v1(
      model, tokens, (size_t)token_count, out_bytes, (size_t)capacity, &cnt);
  *out_byte_count = (int64_t)cnt;
  return st;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_forward_backward_bytes(
    const et_unigram_model *model, const uint8_t *bytes, int64_t byte_count,
    double *out_log_z, double *out_expected_counts, int64_t count_capacity) {
  if (model == NULL || out_log_z == NULL || byte_count < 0 ||
      count_capacity < 0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  return et_unigram_forward_backward_v1(model, bytes, (size_t)byte_count,
                                        out_log_z, out_expected_counts,
                                        (size_t)count_capacity);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_forward_backward_cstr(
    const et_unigram_model *model, const char *text, double *out_log_z,
    double *out_expected_counts, int64_t count_capacity) {
  if (text == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const size_t len = strlen(text);
  if (len > (size_t)INT64_MAX) return ET_UNIGRAM_ERR_OVERFLOW;
  return et_unigram_aot_forward_backward_bytes(model, (const uint8_t *)text,
                                               (int64_t)len, out_log_z,
                                               out_expected_counts,
                                               count_capacity);
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_sample_cstr(
    const et_unigram_model *model, const char *text, double alpha_temp,
    int64_t seed, uint32_t *out_tokens, int64_t capacity, int64_t *out_count,
    double *out_score) {
  if (model == NULL || text == NULL || out_count == NULL || capacity < 0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  const size_t len = strlen(text);
  size_t cnt = 0u;
  const et_unigram_status st = et_unigram_sample_v1(
      model, (const uint8_t *)text, len, alpha_temp, (uint64_t)seed, out_tokens,
      (size_t)capacity, &cnt, out_score);
  *out_count = (int64_t)cnt;
  return st;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_perturbed_log_z_cstr(
    const et_unigram_model *model, const char *text, int64_t piece_id,
    double delta, double *out_log_z) {
  if (model == NULL || text == NULL || out_log_z == NULL) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (!isfinite(delta)) return ET_UNIGRAM_ERR_NONFINITE;

  et_unigram_info_v1 info;
  et_unigram_status st = et_unigram_get_info_v1(model, &info);
  if (st != ET_UNIGRAM_OK) return st;
  if (piece_id < 0 || (uint64_t)piece_id >= (uint64_t)info.active_count) {
    return ET_UNIGRAM_ERR_TOKEN_ID;
  }

  et_unigram_piece_spec_v1 *specs =
      calloc((size_t)info.active_count, sizeof(et_unigram_piece_spec_v1));
  uint8_t *buf =
      malloc((size_t)info.active_count * ET_UNIGRAM_MAX_PIECE_BYTES);
  if (specs == NULL || buf == NULL) {
    free(specs);
    free(buf);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  for (uint32_t i = 0u; i < info.active_count; ++i) {
    uint8_t *dst = buf + ((size_t)i * ET_UNIGRAM_MAX_PIECE_BYTES);
    size_t plen = 0u;
    double lp = 0.0;
    st = et_unigram_get_piece_v1(model, i, dst, ET_UNIGRAM_MAX_PIECE_BYTES,
                                 &plen, &lp);
    if (st != ET_UNIGRAM_OK) {
      free(specs);
      free(buf);
      return st;
    }
    if (i == (uint32_t)piece_id) lp += delta;
    specs[i].bytes = dst;
    specs[i].byte_count = plen;
    specs[i].log_prob = lp;
  }

  et_unigram_model *tmp = NULL;
  st = et_unigram_create_v1(specs, (size_t)info.active_count, info.special_mask,
                            &tmp);
  free(specs);
  free(buf);
  if (st != ET_UNIGRAM_OK) return st;

  st = et_unigram_forward_backward_v1(tmp, (const uint8_t *)text, strlen(text),
                                      out_log_z, NULL, 0u);
  et_unigram_destroy_v1(tmp);
  return st;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_probe_nonfinite_create(void) {
  uint8_t b[256];
  et_unigram_piece_spec_v1 specs[256];
  for (uint32_t i = 0u; i < 256u; ++i) {
    b[i] = (uint8_t)i;
    specs[i].bytes = &b[i];
    specs[i].byte_count = 1u;
    specs[i].log_prob = -6.0;
  }
  specs[42].log_prob = (double)NAN;
  et_unigram_model *m = NULL;
  const et_unigram_status st = et_unigram_create_v1(specs, 256u, 0u, &m);
  if (m != NULL) et_unigram_destroy_v1(m);
  return st;
}

ET_UNIGRAM_AOT_API int32_t
et_unigram_aot_probe_fp_env_trap(const et_unigram_model *model) {
  if (model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const int saved = fegetround();
  if (fesetround(FE_UPWARD) != 0) return ET_UNIGRAM_ERR_FP_ENV;
  const uint8_t sample[4] = {'t', 'e', 's', 't'};
  size_t cnt = 0u;
  const et_unigram_status st =
      et_unigram_encode_v1(model, sample, 4u, NULL, 0u, &cnt, NULL);
  (void)fesetround(saved);
  return st;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_store_u8(uint8_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index,
                                                   int64_t value) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity ||
      value < 0 || value > 255) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }
  buffer[(size_t)index] = (uint8_t)value;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_u8(const uint8_t *buffer,
                                                  int64_t capacity,
                                                  int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return -1;
  }
  return (int64_t)buffer[(size_t)index];
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_store_u32(uint32_t *buffer,
                                                    int64_t capacity,
                                                    int64_t index,
                                                    int64_t value) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity ||
      value < 0 || value > (int64_t)UINT32_MAX) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }
  buffer[(size_t)index] = (uint32_t)value;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_u32(const uint32_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return -1;
  }
  return (int64_t)buffer[(size_t)index];
}

ET_UNIGRAM_AOT_API int64_t et_unigram_aot_load_i64(const int64_t *buffer,
                                                   int64_t capacity,
                                                   int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return -1;
  }
  return buffer[(size_t)index];
}

ET_UNIGRAM_AOT_API double et_unigram_aot_load_f64(const double *buffer,
                                                  int64_t capacity,
                                                  int64_t index) {
  if (buffer == NULL || capacity < 0 || index < 0 || index >= capacity) {
    return (double)NAN;
  }
  return buffer[(size_t)index];
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_f64_bits_equal(double a, double b) {
  uint64_t ba = 0u;
  uint64_t bb = 0u;
  memcpy(&ba, &a, sizeof(ba));
  memcpy(&bb, &b, sizeof(bb));
  return (ba == bb) ? 1 : 0;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_bytes_equal(const uint8_t *a,
                                                      const uint8_t *b,
                                                      int64_t count,
                                                      int32_t *equal_out) {
  if (equal_out == NULL || count < 0) return ET_UNIGRAM_ERR_ARGUMENT;
  if (count > 0 && (a == NULL || b == NULL)) return ET_UNIGRAM_ERR_ARGUMENT;
  *equal_out = (count == 0 || memcmp(a, b, (size_t)count) == 0) ? 1 : 0;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_u32_equal(const uint32_t *a,
                                                    const uint32_t *b,
                                                    int64_t count,
                                                    int32_t *equal_out) {
  if (equal_out == NULL || count < 0) return ET_UNIGRAM_ERR_ARGUMENT;
  if (count > 0 && (a == NULL || b == NULL)) return ET_UNIGRAM_ERR_ARGUMENT;
  *equal_out =
      (count == 0 || memcmp(a, b, (size_t)count * sizeof(uint32_t)) == 0) ? 1
                                                                          : 0;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_cstr_bytes_equal(const char *cstr,
                                                           const uint8_t *bytes,
                                                           int64_t byte_count,
                                                           int32_t *equal_out) {
  if (cstr == NULL || equal_out == NULL || byte_count < 0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (byte_count > 0 && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const size_t len = strlen(cstr);
  *equal_out = (len == (size_t)byte_count &&
                (len == 0u || memcmp(cstr, bytes, len) == 0))
                   ? 1
                   : 0;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_AOT_API int32_t et_unigram_aot_cstr_len(const char *cstr,
                                                   int64_t *out_len) {
  if (cstr == NULL || out_len == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  const size_t len = strlen(cstr);
  if (len > (size_t)INT64_MAX) return ET_UNIGRAM_ERR_OVERFLOW;
  *out_len = (int64_t)len;
  return ET_UNIGRAM_OK;
}
