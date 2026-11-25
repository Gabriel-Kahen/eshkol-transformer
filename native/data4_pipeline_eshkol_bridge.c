#include "data4_pipeline_eshkol_bridge.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define ET_DATA4_AOT_MAGIC UINT64_C(0x4441544134414f54)

typedef struct aot_owned_doc {
  char *source_id;
  uint8_t *bytes;
  size_t byte_count;
} aot_owned_doc;

typedef struct et_data4_aot_session {
  uint64_t magic;
  et_data4_pipeline_config config;
  aot_owned_doc *docs;
  size_t doc_count;
  size_t doc_capacity;
  int has_run;
  et_data4_doc_result *results;
  size_t result_count;
  et_data4_pipeline_summary summary;
  char *report_buf;
  size_t report_size;
} et_data4_aot_session;

static int valid_ptr_aligned(const void *p, size_t alignment, size_t bytes) {
  if (p == NULL) return 0;
  if (alignment > 1u && ((uintptr_t)p % alignment) != 0u) return 0;
  return (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static et_data4_aot_session *checked_session(void *p) {
  if (!valid_ptr_aligned(p, _Alignof(et_data4_aot_session),
                         sizeof(et_data4_aot_session)))
    return NULL;
  et_data4_aot_session *s = (et_data4_aot_session *)p;
  if (s->magic != ET_DATA4_AOT_MAGIC) return NULL;
  return s;
}

static const et_data4_aot_session *checked_session_const(const void *p) {
  if (!valid_ptr_aligned(p, _Alignof(et_data4_aot_session),
                         sizeof(et_data4_aot_session)))
    return NULL;
  const et_data4_aot_session *s = (const et_data4_aot_session *)p;
  if (s->magic != ET_DATA4_AOT_MAGIC) return NULL;
  return s;
}

static void clear_run_state(et_data4_aot_session *s) {
  free(s->results);
  s->results = NULL;
  s->result_count = 0u;
  free(s->report_buf);
  s->report_buf = NULL;
  s->report_size = 0u;
  memset(&s->summary, 0, sizeof(s->summary));
  s->has_run = 0;
}

void *et_data4_aot_session_create(void) {
  et_data4_aot_session *s =
      (et_data4_aot_session *)calloc(1u, sizeof(et_data4_aot_session));
  if (s == NULL) return NULL;
  if (et_data4_config_default(&s->config) != ET_DATA4_OK) {
    free(s);
    return NULL;
  }
  s->magic = ET_DATA4_AOT_MAGIC;
  return s;
}

void et_data4_aot_session_destroy(void *session) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL) return;
  clear_run_state(s);
  for (size_t i = 0u; i < s->doc_count; ++i) {
    free(s->docs[i].source_id);
    free(s->docs[i].bytes);
  }
  free(s->docs);
  s->magic = 0u;
  free(s);
}

int32_t et_data4_aot_set_limits(void *session, int64_t min_bytes,
                                int64_t max_bytes, int64_t min_codepoints,
                                int64_t max_codepoints) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (min_bytes < 0 || max_bytes < 0 || min_codepoints < 0 ||
      max_codepoints < 0 || min_bytes > max_bytes ||
      min_codepoints > max_codepoints)
    return ET_DATA4_ERR_ARGUMENT;
  s->config.min_bytes = (uint64_t)min_bytes;
  s->config.max_bytes = (uint64_t)max_bytes;
  s->config.min_codepoints = (uint64_t)min_codepoints;
  s->config.max_codepoints = (uint64_t)max_codepoints;
  clear_run_state(s);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_set_quality(void *session, double min_printable_ratio,
                                 double min_alnum_ratio,
                                 double max_dup_line_ratio,
                                 double max_repeated_ngram_ratio,
                                 int32_t rep_ngram_size) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL || rep_ngram_size <= 0 ||
      rep_ngram_size > (int32_t)ET_DATA4_MAX_NGRAM_SIZE)
    return ET_DATA4_ERR_ARGUMENT;
  et_data4_pipeline_config next = s->config;
  next.min_printable_ratio = min_printable_ratio;
  next.min_alnum_ratio = min_alnum_ratio;
  next.max_dup_line_ratio = max_dup_line_ratio;
  next.max_repeated_ngram_ratio = max_repeated_ngram_ratio;
  next.rep_ngram_size = (uint32_t)rep_ngram_size;
  et_data4_pipeline_summary dummy;
  const et_data4_status check =
      et_data4_pipeline_run(NULL, 0u, &next, NULL, 0u, &dummy);
  if (check != ET_DATA4_OK) return check;
  s->config = next;
  clear_run_state(s);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_set_near_dedup(void *session, int32_t enabled,
                                    int32_t dedup_ngram_size,
                                    int32_t num_perms, int32_t bands,
                                    int64_t seed, double jaccard_threshold) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL || (enabled != 0 && enabled != 1) || dedup_ngram_size < 0 ||
      num_perms < 0 || bands < 0)
    return ET_DATA4_ERR_ARGUMENT;
  et_data4_pipeline_config next = s->config;
  next.near_dup_enabled = (uint32_t)enabled;
  next.dedup_ngram_size = (uint32_t)dedup_ngram_size;
  next.minhash_num_perms = (uint32_t)num_perms;
  next.minhash_bands = (uint32_t)bands;
  next.minhash_seed = (uint64_t)seed;
  next.near_dup_jaccard_threshold = jaccard_threshold;
  et_data4_pipeline_summary dummy;
  const et_data4_status check =
      et_data4_pipeline_run(NULL, 0u, &next, NULL, 0u, &dummy);
  if (check != ET_DATA4_OK) return check;
  s->config = next;
  clear_run_state(s);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_add_doc(void *session, const void *source_id_bytes,
                             int64_t source_id_len, const void *doc_bytes,
                             int64_t doc_len) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL || source_id_bytes == NULL || source_id_len <= 0 ||
      source_id_len > (int64_t)ET_DATA4_SOURCE_ID_MAX || doc_len < 0)
    return ET_DATA4_ERR_ARGUMENT;
  if ((uint64_t)doc_len > (uint64_t)PTRDIFF_MAX ||
      (uint64_t)doc_len > UINT64_MAX / 8u)
    return ET_DATA4_ERR_OVERFLOW;
  if (doc_len > 0 && doc_bytes == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (memchr(source_id_bytes, '\0', (size_t)source_id_len) != NULL)
    return ET_DATA4_ERR_INVALID_SOURCE_ID;

  if (s->doc_count == s->doc_capacity) {
    const size_t next_cap = s->doc_capacity == 0u ? 16u : s->doc_capacity * 2u;
    if (next_cap < s->doc_capacity ||
        next_cap > PTRDIFF_MAX / sizeof(aot_owned_doc))
      return ET_DATA4_ERR_OVERFLOW;
    aot_owned_doc *grown =
        (aot_owned_doc *)realloc(s->docs, next_cap * sizeof(aot_owned_doc));
    if (grown == NULL) return ET_DATA4_ERR_NOMEM;
    s->docs = grown;
    s->doc_capacity = next_cap;
  }

  const size_t id_sz = (size_t)source_id_len;
  char *id_copy = (char *)malloc(id_sz + 1u);
  if (id_copy == NULL) return ET_DATA4_ERR_NOMEM;
  memcpy(id_copy, source_id_bytes, id_sz);
  id_copy[id_sz] = '\0';

  uint8_t *bytes_copy = NULL;
  if (doc_len > 0) {
    bytes_copy = (uint8_t *)malloc((size_t)doc_len);
    if (bytes_copy == NULL) {
      free(id_copy);
      return ET_DATA4_ERR_NOMEM;
    }
    memcpy(bytes_copy, doc_bytes, (size_t)doc_len);
  }

  s->docs[s->doc_count++] =
      (aot_owned_doc){id_copy, bytes_copy, (size_t)doc_len};
  clear_run_state(s);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_run(void *session) {
  et_data4_aot_session *s = checked_session(session);
  if (s == NULL) return ET_DATA4_ERR_ARGUMENT;
  clear_run_state(s);

  et_data4_document *views = NULL;
  et_data4_doc_result *results = NULL;
  if (s->doc_count != 0u) {
    views = (et_data4_document *)malloc(s->doc_count * sizeof(et_data4_document));
    results = (et_data4_doc_result *)calloc(s->doc_count,
                                            sizeof(et_data4_doc_result));
    if (views == NULL || results == NULL) {
      free(views);
      free(results);
      return ET_DATA4_ERR_NOMEM;
    }
    for (size_t i = 0u; i < s->doc_count; ++i) {
      views[i] = (et_data4_document){s->docs[i].source_id, s->docs[i].bytes,
                                     s->docs[i].byte_count};
    }
  }

  et_data4_pipeline_summary summary;
  memset(&summary, 0, sizeof(summary));
  et_data4_status st = et_data4_pipeline_run(
      views, s->doc_count, &s->config, results, s->doc_count, &summary);
  free(views);
  if (st != ET_DATA4_OK) {
    free(results);
    return st;
  }

  size_t report_len = 0u;
  uint8_t report_digest[ET_DATA4_SHA256_BYTES];
  st = et_data4_report_format(&s->config, &summary, results, s->doc_count, NULL,
                              0u, &report_len, report_digest);
  if (st != ET_DATA4_OK) {
    free(results);
    return st;
  }

  char *report_buf = (char *)malloc(report_len + 1u);
  if (report_buf == NULL) {
    free(results);
    return ET_DATA4_ERR_NOMEM;
  }
  size_t written = 0u;
  st = et_data4_report_format(&s->config, &summary, results, s->doc_count,
                              report_buf, report_len + 1u, &written,
                              report_digest);
  if (st != ET_DATA4_OK || written != report_len) {
    free(report_buf);
    free(results);
    return st != ET_DATA4_OK ? st : ET_DATA4_ERR_FORMAT;
  }

  s->results = results;
  s->result_count = s->doc_count;
  memcpy(summary.report_sha256, report_digest, ET_DATA4_SHA256_BYTES);
  s->summary = summary;
  s->report_buf = report_buf;
  s->report_size = report_len;
  s->has_run = 1;
  return ET_DATA4_OK;
}

int64_t et_data4_aot_get_stat(const void *session, int32_t stat_id) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run) return -1;
  switch (stat_id) {
    case ET_DATA4_AOT_STAT_INPUT_DOCS:
      return (int64_t)s->summary.input_documents;
    case ET_DATA4_AOT_STAT_KEPT_DOCS:
      return (int64_t)s->summary.kept_documents;
    case ET_DATA4_AOT_STAT_REJECTED_UTF8:
      return (int64_t)s->summary.rejected_utf8_documents;
    case ET_DATA4_AOT_STAT_REJECTED_LENGTH:
      return (int64_t)s->summary.rejected_length_documents;
    case ET_DATA4_AOT_STAT_REJECTED_QUALITY:
      return (int64_t)s->summary.rejected_quality_documents;
    case ET_DATA4_AOT_STAT_REJECTED_REPETITION:
      return (int64_t)s->summary.rejected_repetition_documents;
    case ET_DATA4_AOT_STAT_DUPLICATE_EXACT:
      return (int64_t)s->summary.duplicate_exact_documents;
    case ET_DATA4_AOT_STAT_DUPLICATE_NEAR:
      return (int64_t)s->summary.duplicate_near_documents;
    case ET_DATA4_AOT_STAT_INPUT_BYTES:
      return (int64_t)s->summary.input_bytes;
    case ET_DATA4_AOT_STAT_KEPT_BYTES:
      return (int64_t)s->summary.kept_bytes;
    case ET_DATA4_AOT_STAT_REPORT_BYTES:
      return (int64_t)s->report_size;
    default:
      return -1;
  }
}

int32_t et_data4_aot_get_doc_disposition(const void *session, int64_t index) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || index < 0 || (uint64_t)index >= s->result_count)
    return ET_DATA4_ERR_BOUNDS;
  return (int32_t)s->results[index].disposition;
}

int64_t et_data4_aot_get_doc_matched_ordinal(const void *session,
                                             int64_t index) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || index < 0 || (uint64_t)index >= s->result_count)
    return -1;
  if (!s->results[index].has_matched_canonical) return -1;
  return (int64_t)s->results[index].matched_canonical_ordinal;
}

int64_t et_data4_aot_get_doc_jaccard_num(const void *session, int64_t index) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || index < 0 || (uint64_t)index >= s->result_count)
    return -1;
  return (int64_t)s->results[index].jaccard_num;
}

int64_t et_data4_aot_get_doc_jaccard_den(const void *session, int64_t index) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || index < 0 || (uint64_t)index >= s->result_count)
    return -1;
  return (int64_t)s->results[index].jaccard_den;
}

int32_t et_data4_aot_copy_report(const void *session, void *out_buf,
                                 int64_t capacity) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || out_buf == NULL || capacity < 0)
    return ET_DATA4_ERR_ARGUMENT;
  if ((uint64_t)capacity < (uint64_t)s->report_size)
    return ET_DATA4_ERR_BOUNDS;
  memcpy(out_buf, s->report_buf, s->report_size);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_copy_report_sha256(const void *session, void *out_buf,
                                        int64_t capacity) {
  const et_data4_aot_session *s = checked_session_const(session);
  if (s == NULL || !s->has_run || out_buf == NULL || capacity < 0)
    return ET_DATA4_ERR_ARGUMENT;
  if (capacity < (int64_t)ET_DATA4_SHA256_BYTES) return ET_DATA4_ERR_BOUNDS;
  memcpy(out_buf, s->summary.report_sha256, ET_DATA4_SHA256_BYTES);
  return ET_DATA4_OK;
}

int32_t et_data4_aot_verify_report(const void *report_buf,
                                   int64_t report_size) {
  if (report_buf == NULL || report_size <= 0 ||
      (uint64_t)report_size > (uint64_t)PTRDIFF_MAX)
    return ET_DATA4_ERR_ARGUMENT;
  return et_data4_report_verify((const char *)report_buf, (size_t)report_size,
                                NULL, NULL, NULL, 0u);
}

int32_t et_data4_aot_buf_store_u8(void *buf, int64_t capacity, int64_t index,
                                  int64_t value) {
  if (buf == NULL || capacity <= 0 || index < 0 || index >= capacity ||
      value < 0 || value > 255)
    return ET_DATA4_ERR_ARGUMENT;
  if ((uint64_t)capacity > (uint64_t)PTRDIFF_MAX ||
      !valid_ptr_aligned(buf, 1u, (size_t)capacity))
    return ET_DATA4_ERR_BOUNDS;
  ((uint8_t *)buf)[index] = (uint8_t)value;
  return ET_DATA4_OK;
}

int64_t et_data4_aot_buf_load_u8(const void *buf, int64_t capacity,
                                 int64_t index) {
  if (buf == NULL || capacity <= 0 || index < 0 || index >= capacity)
    return -1;
  if ((uint64_t)capacity > (uint64_t)PTRDIFF_MAX ||
      !valid_ptr_aligned(buf, 1u, (size_t)capacity))
    return -1;
  return (int64_t)((const uint8_t *)buf)[index];
}
