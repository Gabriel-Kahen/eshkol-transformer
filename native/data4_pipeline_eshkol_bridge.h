#ifndef ESHKOL_TRANSFORMER_DATA4_PIPELINE_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_DATA4_PIPELINE_ESHKOL_BRIDGE_H

#include "eshkol_transformer/data4_pipeline_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ET_DATA4_AOT_STAT_INPUT_DOCS = 0,
  ET_DATA4_AOT_STAT_KEPT_DOCS = 1,
  ET_DATA4_AOT_STAT_REJECTED_UTF8 = 2,
  ET_DATA4_AOT_STAT_REJECTED_LENGTH = 3,
  ET_DATA4_AOT_STAT_REJECTED_QUALITY = 4,
  ET_DATA4_AOT_STAT_REJECTED_REPETITION = 5,
  ET_DATA4_AOT_STAT_DUPLICATE_EXACT = 6,
  ET_DATA4_AOT_STAT_DUPLICATE_NEAR = 7,
  ET_DATA4_AOT_STAT_INPUT_BYTES = 8,
  ET_DATA4_AOT_STAT_KEPT_BYTES = 9,
  ET_DATA4_AOT_STAT_REPORT_BYTES = 10
};

/* Opaque AOT session handle. Bridge functions accept void * so misaligned or
 * invalid pointers can be rejected cleanly without typed-pointer UB. */
void *et_data4_aot_session_create(void);
void et_data4_aot_session_destroy(void *session);

int32_t et_data4_aot_set_limits(void *session, int64_t min_bytes,
                                int64_t max_bytes, int64_t min_codepoints,
                                int64_t max_codepoints);

int32_t et_data4_aot_set_quality(void *session, double min_printable_ratio,
                                 double min_alnum_ratio,
                                 double max_dup_line_ratio,
                                 double max_repeated_ngram_ratio,
                                 int32_t rep_ngram_size);

int32_t et_data4_aot_set_near_dedup(void *session, int32_t enabled,
                                    int32_t dedup_ngram_size,
                                    int32_t num_perms, int32_t bands,
                                    int64_t seed, double jaccard_threshold);

int32_t et_data4_aot_add_doc(void *session, const void *source_id_bytes,
                             int64_t source_id_len, const void *doc_bytes,
                             int64_t doc_len);

int32_t et_data4_aot_run(void *session);

int64_t et_data4_aot_get_stat(const void *session, int32_t stat_id);
int32_t et_data4_aot_get_doc_disposition(const void *session, int64_t index);
int64_t et_data4_aot_get_doc_matched_ordinal(const void *session,
                                             int64_t index);
int64_t et_data4_aot_get_doc_jaccard_num(const void *session, int64_t index);
int64_t et_data4_aot_get_doc_jaccard_den(const void *session, int64_t index);

int32_t et_data4_aot_copy_report(const void *session, void *out_buf,
                                 int64_t capacity);
int32_t et_data4_aot_copy_report_sha256(const void *session, void *out_buf,
                                        int64_t capacity);
int32_t et_data4_aot_verify_report(const void *report_buf, int64_t report_size);

/* Safe byte-buffer marshal helpers for Eshkol AOT callers. */
int32_t et_data4_aot_buf_store_u8(void *buf, int64_t capacity, int64_t index,
                                  int64_t value);
int64_t et_data4_aot_buf_load_u8(const void *buf, int64_t capacity,
                                 int64_t index);

#ifdef __cplusplus
}
#endif

#endif
