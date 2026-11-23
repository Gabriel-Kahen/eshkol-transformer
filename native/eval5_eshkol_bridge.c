#define _POSIX_C_SOURCE 200809L

#include "eval5_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "EVAL5 AOT bridge requires 8-bit bytes and at most 64-bit size_t");

typedef struct {
  uint8_t *bytes;
  size_t byte_count;
  uint32_t *tokens;
  size_t token_count;
} owned_doc;

struct et_eval5_aot_corpus {
  owned_doc *docs;
  size_t count;
  size_t capacity;
};

typedef struct {
  float *logits;
  uint32_t *target_ids;
  size_t token_count;
  size_t vocab_size;
  size_t logit_count;
} owned_choice;

typedef struct {
  owned_choice *choices;
  size_t choice_count;
  size_t choice_capacity;
  size_t correct_index;
} owned_mc_item;

struct et_eval5_aot_mc_batch {
  owned_mc_item *items;
  size_t item_count;
  size_t item_capacity;
};

struct et_eval5_aot_manifest {
  et_eval5_manifest inner;
};

/* Standalone SHA-256 core for file and byte provenance digests. */
typedef struct {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} bridge_sha256_state;

static uint32_t bridge_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void bridge_sha256_block(bridge_sha256_state *state,
                                const uint8_t block[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  size_t i;
  for (i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = bridge_rotr32(w[i - 15u], 7u) ^
                        bridge_rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 = bridge_rotr32(w[i - 2u], 17u) ^
                        bridge_rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = state->h[0];
  uint32_t b = state->h[1];
  uint32_t c = state->h[2];
  uint32_t d = state->h[3];
  uint32_t e = state->h[4];
  uint32_t f = state->h[5];
  uint32_t g = state->h[6];
  uint32_t h = state->h[7];
  for (i = 0; i < 64u; ++i) {
    const uint32_t s1 =
        bridge_rotr32(e, 6u) ^ bridge_rotr32(e, 11u) ^ bridge_rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 =
        bridge_rotr32(a, 2u) ^ bridge_rotr32(a, 13u) ^ bridge_rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state->h[0] += a;
  state->h[1] += b;
  state->h[2] += c;
  state->h[3] += d;
  state->h[4] += e;
  state->h[5] += f;
  state->h[6] += g;
  state->h[7] += h;
}

static void bridge_sha256_init(bridge_sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(state->h, initial, sizeof(initial));
  state->length = 0u;
  state->used = 0u;
}

static void bridge_sha256_update(bridge_sha256_state *state,
                                 const uint8_t *bytes, size_t length) {
  state->length += (uint64_t)length;
  while (length != 0u) {
    size_t take = 64u - state->used;
    if (take > length) {
      take = length;
    }
    memcpy(state->block + state->used, bytes, take);
    state->used += take;
    bytes += take;
    length -= take;
    if (state->used == 64u) {
      bridge_sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void bridge_sha256_final(bridge_sha256_state *state,
                                uint8_t digest[ET_EVAL5_SHA256_BYTES]) {
  const uint64_t bits = state->length * UINT64_C(8);
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    bridge_sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0; i < 8u; ++i) {
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  }
  bridge_sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static int nonneg_i64_to_size(int64_t value, size_t *out) {
  if (value < 0) {
    return 0;
  }
  const uint64_t uval = (uint64_t)value;
  if (uval > (uint64_t)SIZE_MAX) {
    return 0;
  }
  *out = (size_t)uval;
  return 1;
}

static int u64_to_i64(uint64_t value, int64_t *out) {
  if (value > (uint64_t)INT64_MAX) {
    return 0;
  }
  *out = (int64_t)value;
  return 1;
}

int32_t et_eval5_aot_corpus_create(et_eval5_aot_corpus **out) {
  if (out == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  et_eval5_aot_corpus *corpus = (et_eval5_aot_corpus *)calloc(1u, sizeof(*corpus));
  if (corpus == NULL) {
    return ET_EVAL5_ERR_NOMEM;
  }
  *out = corpus;
  return ET_EVAL5_OK;
}

void et_eval5_aot_corpus_destroy(et_eval5_aot_corpus *corpus) {
  if (corpus == NULL) {
    return;
  }
  for (size_t i = 0; i < corpus->count; ++i) {
    free(corpus->docs[i].bytes);
    free(corpus->docs[i].tokens);
  }
  free(corpus->docs);
  free(corpus);
}

int32_t et_eval5_aot_corpus_add_doc(et_eval5_aot_corpus *corpus,
                                    const uint8_t *bytes, int64_t byte_count,
                                    const uint32_t *tokens,
                                    int64_t token_count) {
  size_t bcount = 0;
  size_t tcount = 0;
  if (corpus == NULL || !nonneg_i64_to_size(byte_count, &bcount) ||
      !nonneg_i64_to_size(token_count, &tcount)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if ((bcount != 0u && bytes == NULL) || (tcount != 0u && tokens == NULL)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (bcount > (size_t)PTRDIFF_MAX ||
      tcount > (size_t)PTRDIFF_MAX / sizeof(uint32_t)) {
    return ET_EVAL5_ERR_OVERFLOW;
  }
  if (corpus->count == corpus->capacity) {
    size_t next_cap = corpus->capacity == 0u ? 8u : corpus->capacity * 2u;
    if (next_cap < corpus->capacity ||
        next_cap > (size_t)PTRDIFF_MAX / sizeof(*corpus->docs)) {
      return ET_EVAL5_ERR_OVERFLOW;
    }
    owned_doc *grown =
        (owned_doc *)realloc(corpus->docs, next_cap * sizeof(*corpus->docs));
    if (grown == NULL) {
      return ET_EVAL5_ERR_NOMEM;
    }
    corpus->docs = grown;
    corpus->capacity = next_cap;
  }
  uint8_t *bcopy = NULL;
  uint32_t *tcopy = NULL;
  if (bcount != 0u) {
    bcopy = (uint8_t *)malloc(bcount);
    if (bcopy == NULL) {
      return ET_EVAL5_ERR_NOMEM;
    }
    memcpy(bcopy, bytes, bcount);
  }
  if (tcount != 0u) {
    tcopy = (uint32_t *)malloc(tcount * sizeof(uint32_t));
    if (tcopy == NULL) {
      free(bcopy);
      return ET_EVAL5_ERR_NOMEM;
    }
    memcpy(tcopy, tokens, tcount * sizeof(uint32_t));
  }
  corpus->docs[corpus->count++] =
      (owned_doc){bcopy, bcount, tcopy, tcount};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_corpus_doc_count(const et_eval5_aot_corpus *corpus,
                                      int64_t *out_count) {
  if (corpus == NULL || out_count == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (!u64_to_i64((uint64_t)corpus->count, out_count)) {
    return ET_EVAL5_ERR_OVERFLOW;
  }
  return ET_EVAL5_OK;
}

static et_eval5_document *build_doc_views(const et_eval5_aot_corpus *corpus) {
  if (corpus->count == 0u) {
    return NULL;
  }
  et_eval5_document *views =
      (et_eval5_document *)malloc(corpus->count * sizeof(*views));
  if (views == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < corpus->count; ++i) {
    views[i] = (et_eval5_document){
        corpus->docs[i].bytes, corpus->docs[i].byte_count,
        corpus->docs[i].tokens, corpus->docs[i].token_count};
  }
  return views;
}

int32_t et_eval5_aot_audit_contamination(
    const et_eval5_aot_corpus *train_corpus,
    const et_eval5_aot_corpus *eval_corpus, int32_t ngram_mode,
    int32_t ngram_order, int32_t forbid_exact_match, double min_ngram_overlap,
    double max_allowed_rate, int64_t *contaminated_indices,
    int64_t index_capacity, int64_t *summary_i64,
    double *contamination_rate_out) {
  size_t cap = 0;
  if (train_corpus == NULL || eval_corpus == NULL ||
      contaminated_indices == NULL || summary_i64 == NULL ||
      contamination_rate_out == NULL ||
      !nonneg_i64_to_size(index_capacity, &cap) || ngram_mode < 0 ||
      ngram_order < 0 || forbid_exact_match < 0) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (eval_corpus->count == 0u) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (cap < eval_corpus->count) {
    return ET_EVAL5_ERR_BOUNDS;
  }
  if (eval_corpus->count > (size_t)PTRDIFF_MAX / sizeof(uint64_t)) {
    return ET_EVAL5_ERR_OVERFLOW;
  }

  et_eval5_document *train_views = NULL;
  if (train_corpus->count != 0u) {
    train_views = build_doc_views(train_corpus);
    if (train_views == NULL) {
      return ET_EVAL5_ERR_NOMEM;
    }
  }
  et_eval5_document *eval_views = build_doc_views(eval_corpus);
  if (eval_views == NULL) {
    free(train_views);
    return ET_EVAL5_ERR_NOMEM;
  }
  uint64_t *raw_indices =
      (uint64_t *)malloc(eval_corpus->count * sizeof(*raw_indices));
  if (raw_indices == NULL) {
    free(eval_views);
    free(train_views);
    return ET_EVAL5_ERR_NOMEM;
  }

  const et_eval5_contamination_config config = {
      (uint32_t)ngram_mode, (uint32_t)ngram_order, (uint32_t)forbid_exact_match,
      min_ngram_overlap, max_allowed_rate};
  et_eval5_contamination_summary summary;
  memset(&summary, 0, sizeof(summary));

  const et_eval5_status status = et_eval5_audit_contamination(
      train_views, train_corpus->count, eval_views, eval_corpus->count, &config,
      raw_indices, eval_corpus->count, &summary);

  if (status == ET_EVAL5_OK || status == ET_EVAL5_ERR_CONTAMINATED) {
    int64_t converted[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN];
    int overflow =
        !u64_to_i64(summary.eval_documents,
                    &converted[ET_EVAL5_AOT_AUDIT_EVAL_DOCS]) ||
        !u64_to_i64(summary.contaminated_documents,
                    &converted[ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS]) ||
        !u64_to_i64(summary.exact_doc_matches,
                    &converted[ET_EVAL5_AOT_AUDIT_EXACT_MATCHES]) ||
        !u64_to_i64(summary.ngram_contaminated_docs,
                    &converted[ET_EVAL5_AOT_AUDIT_NGRAM_DOCS]) ||
        !u64_to_i64(summary.total_eval_ngrams,
                    &converted[ET_EVAL5_AOT_AUDIT_TOTAL_NGRAMS]) ||
        !u64_to_i64(summary.matched_eval_ngrams,
                    &converted[ET_EVAL5_AOT_AUDIT_MATCHED_NGRAMS]);
    for (uint64_t i = 0; !overflow && i < summary.contaminated_documents; ++i) {
      int64_t idx = 0;
      if (!u64_to_i64(raw_indices[i], &idx)) {
        overflow = 1;
      } else {
        contaminated_indices[i] = idx;
      }
    }
    if (overflow) {
      free(raw_indices);
      free(eval_views);
      free(train_views);
      return ET_EVAL5_ERR_OVERFLOW;
    }
    memcpy(summary_i64, converted, sizeof(converted));
    *contamination_rate_out = summary.contamination_rate;
  }

  free(raw_indices);
  free(eval_views);
  free(train_views);
  return status;
}

int32_t et_eval5_aot_score_lm(const float *logits, int64_t logit_count,
                              const uint32_t *target_ids, int64_t token_count,
                              int64_t vocab_size, int64_t byte_count,
                              double *metrics_f64) {
  size_t lcount = 0;
  size_t tcount = 0;
  size_t vcount = 0;
  if (logits == NULL || target_ids == NULL || metrics_f64 == NULL ||
      byte_count <= 0 || !nonneg_i64_to_size(logit_count, &lcount) ||
      !nonneg_i64_to_size(token_count, &tcount) ||
      !nonneg_i64_to_size(vocab_size, &vcount)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  const et_eval5_sequence seq = {logits, target_ids, tcount, vcount, lcount};
  et_eval5_lm_metrics lm;
  memset(&lm, 0, sizeof(lm));
  const et_eval5_status status =
      et_eval5_score_lm(&seq, (uint64_t)byte_count, &lm);
  if (status != ET_EVAL5_OK) {
    return status;
  }
  metrics_f64[ET_EVAL5_AOT_LM_TOTAL_NLL] = lm.total_nll;
  metrics_f64[ET_EVAL5_AOT_LM_MEAN_NLL] = lm.mean_nll;
  metrics_f64[ET_EVAL5_AOT_LM_PERPLEXITY] = lm.perplexity;
  metrics_f64[ET_EVAL5_AOT_LM_BITS_PER_BYTE] = lm.bits_per_byte;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_mc_create(et_eval5_aot_mc_batch **out) {
  if (out == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  et_eval5_aot_mc_batch *batch =
      (et_eval5_aot_mc_batch *)calloc(1u, sizeof(*batch));
  if (batch == NULL) {
    return ET_EVAL5_ERR_NOMEM;
  }
  *out = batch;
  return ET_EVAL5_OK;
}

void et_eval5_aot_mc_destroy(et_eval5_aot_mc_batch *batch) {
  if (batch == NULL) {
    return;
  }
  for (size_t i = 0; i < batch->item_count; ++i) {
    for (size_t j = 0; j < batch->items[i].choice_count; ++j) {
      free(batch->items[i].choices[j].logits);
      free(batch->items[i].choices[j].target_ids);
    }
    free(batch->items[i].choices);
  }
  free(batch->items);
  free(batch);
}

int32_t et_eval5_aot_mc_begin_item(et_eval5_aot_mc_batch *batch,
                                   int64_t correct_index) {
  size_t cidx = 0;
  if (batch == NULL || !nonneg_i64_to_size(correct_index, &cidx)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (batch->item_count == batch->item_capacity) {
    size_t next_cap = batch->item_capacity == 0u ? 8u : batch->item_capacity * 2u;
    if (next_cap < batch->item_capacity ||
        next_cap > (size_t)PTRDIFF_MAX / sizeof(*batch->items)) {
      return ET_EVAL5_ERR_OVERFLOW;
    }
    owned_mc_item *grown = (owned_mc_item *)realloc(
        batch->items, next_cap * sizeof(*batch->items));
    if (grown == NULL) {
      return ET_EVAL5_ERR_NOMEM;
    }
    batch->items = grown;
    batch->item_capacity = next_cap;
  }
  batch->items[batch->item_count++] = (owned_mc_item){NULL, 0u, 0u, cidx};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_mc_add_choice(et_eval5_aot_mc_batch *batch,
                                   const float *logits, int64_t logit_count,
                                   const uint32_t *target_ids,
                                   int64_t token_count, int64_t vocab_size) {
  size_t lcount = 0;
  size_t tcount = 0;
  size_t vcount = 0;
  if (batch == NULL || batch->item_count == 0u || logits == NULL ||
      target_ids == NULL || token_count <= 0 || vocab_size <= 0 ||
      !nonneg_i64_to_size(logit_count, &lcount) ||
      !nonneg_i64_to_size(token_count, &tcount) ||
      !nonneg_i64_to_size(vocab_size, &vcount)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (tcount > (size_t)PTRDIFF_MAX / sizeof(uint32_t) ||
      vcount > ((size_t)PTRDIFF_MAX / sizeof(float)) / tcount ||
      vcount - 1u > UINT32_MAX) {
    return ET_EVAL5_ERR_OVERFLOW;
  }
  if (lcount != tcount * vcount) {
    return ET_EVAL5_ERR_BOUNDS;
  }

  owned_mc_item *item = &batch->items[batch->item_count - 1u];
  if (item->choice_count == item->choice_capacity) {
    size_t next_cap = item->choice_capacity == 0u ? 4u : item->choice_capacity * 2u;
    if (next_cap < item->choice_capacity ||
        next_cap > (size_t)PTRDIFF_MAX / sizeof(*item->choices)) {
      return ET_EVAL5_ERR_OVERFLOW;
    }
    owned_choice *grown = (owned_choice *)realloc(
        item->choices, next_cap * sizeof(*item->choices));
    if (grown == NULL) {
      return ET_EVAL5_ERR_NOMEM;
    }
    item->choices = grown;
    item->choice_capacity = next_cap;
  }

  float *lcopy = (float *)malloc(lcount * sizeof(float));
  if (lcopy == NULL) {
    return ET_EVAL5_ERR_NOMEM;
  }
  uint32_t *tcopy = (uint32_t *)malloc(tcount * sizeof(uint32_t));
  if (tcopy == NULL) {
    free(lcopy);
    return ET_EVAL5_ERR_NOMEM;
  }
  memcpy(lcopy, logits, lcount * sizeof(float));
  memcpy(tcopy, target_ids, tcount * sizeof(uint32_t));
  item->choices[item->choice_count++] =
      (owned_choice){lcopy, tcopy, tcount, vcount, lcount};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_score_mc(const et_eval5_aot_mc_batch *batch,
                              int32_t normalize_by_length,
                              int64_t *correct_count_out,
                              int64_t *total_count_out, double *accuracy_out) {
  if (batch == NULL || normalize_by_length < 0 || normalize_by_length > 1 ||
      correct_count_out == NULL || total_count_out == NULL ||
      accuracy_out == NULL || batch->item_count == 0u) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  size_t total_choices = 0;
  for (size_t i = 0; i < batch->item_count; ++i) {
    if (batch->items[i].choice_count > SIZE_MAX - total_choices) {
      return ET_EVAL5_ERR_OVERFLOW;
    }
    total_choices += batch->items[i].choice_count;
  }
  if (batch->item_count > (size_t)PTRDIFF_MAX / sizeof(et_eval5_choice_item) ||
      total_choices > (size_t)PTRDIFF_MAX / sizeof(et_eval5_sequence)) {
    return ET_EVAL5_ERR_OVERFLOW;
  }

  et_eval5_choice_item *items =
      (et_eval5_choice_item *)malloc(batch->item_count * sizeof(*items));
  if (items == NULL) {
    return ET_EVAL5_ERR_NOMEM;
  }
  et_eval5_sequence *seqs = NULL;
  if (total_choices != 0u) {
    seqs = (et_eval5_sequence *)malloc(total_choices * sizeof(*seqs));
    if (seqs == NULL) {
      free(items);
      return ET_EVAL5_ERR_NOMEM;
    }
  }

  size_t offset = 0;
  for (size_t i = 0; i < batch->item_count; ++i) {
    const owned_mc_item *src = &batch->items[i];
    items[i] = (et_eval5_choice_item){
        src->choice_count != 0u ? (seqs + offset) : NULL, src->choice_count,
        src->correct_index};
    for (size_t j = 0; j < src->choice_count; ++j) {
      const owned_choice *c = &src->choices[j];
      seqs[offset++] = (et_eval5_sequence){
          c->logits, c->target_ids, c->token_count, c->vocab_size,
          c->logit_count};
    }
  }

  et_eval5_mc_metrics mc;
  memset(&mc, 0, sizeof(mc));
  const et_eval5_status status = et_eval5_score_mc(
      items, batch->item_count, (uint32_t)normalize_by_length, &mc);
  free(seqs);
  free(items);
  if (status != ET_EVAL5_OK) {
    return status;
  }
  int64_t correct_i64 = 0;
  int64_t total_i64 = 0;
  if (!u64_to_i64(mc.correct_count, &correct_i64) ||
      !u64_to_i64(mc.total_count, &total_i64)) {
    return ET_EVAL5_ERR_OVERFLOW;
  }
  *correct_count_out = correct_i64;
  *total_count_out = total_i64;
  *accuracy_out = mc.accuracy;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_create(et_eval5_aot_manifest **out) {
  if (out == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  et_eval5_aot_manifest *manifest =
      (et_eval5_aot_manifest *)calloc(1u, sizeof(*manifest));
  if (manifest == NULL) {
    return ET_EVAL5_ERR_NOMEM;
  }
  manifest->inner.version_major = ET_EVAL5_VERSION_MAJOR;
  manifest->inner.version_minor = ET_EVAL5_VERSION_MINOR;
  *out = manifest;
  return ET_EVAL5_OK;
}

void et_eval5_aot_manifest_destroy(et_eval5_aot_manifest *manifest) {
  free(manifest);
}

static int copy_name(char dst[ET_EVAL5_NAME_MAX + 1u], const char *src) {
  if (src == NULL) {
    return 0;
  }
  const size_t len = strlen(src);
  if (len == 0u || len > ET_EVAL5_NAME_MAX) {
    return 0;
  }
  memset(dst, 0, ET_EVAL5_NAME_MAX + 1u);
  memcpy(dst, src, len);
  return 1;
}

int32_t et_eval5_aot_manifest_set_identity(et_eval5_aot_manifest *manifest,
                                           const char *suite, const char *task,
                                           const char *split,
                                           int64_t context_length,
                                           int64_t max_samples) {
  if (manifest == NULL || suite == NULL || task == NULL || split == NULL ||
      context_length <= 0 || max_samples < 0) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  char suite_buf[ET_EVAL5_NAME_MAX + 1u];
  char task_buf[ET_EVAL5_NAME_MAX + 1u];
  char split_buf[ET_EVAL5_NAME_MAX + 1u];
  if (!copy_name(suite_buf, suite) || !copy_name(task_buf, task) ||
      !copy_name(split_buf, split)) {
    return ET_EVAL5_ERR_FORMAT;
  }
  memcpy(manifest->inner.suite, suite_buf, sizeof(suite_buf));
  memcpy(manifest->inner.task, task_buf, sizeof(task_buf));
  memcpy(manifest->inner.split, split_buf, sizeof(split_buf));
  manifest->inner.context_length = (uint64_t)context_length;
  manifest->inner.max_samples = (uint64_t)max_samples;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_set_digests(et_eval5_aot_manifest *manifest,
                                          const uint8_t *train_sha256,
                                          const uint8_t *eval_sha256,
                                          const uint8_t *tokenizer_sha256,
                                          const uint8_t *checkpoint_sha256) {
  if (manifest == NULL || train_sha256 == NULL || eval_sha256 == NULL ||
      tokenizer_sha256 == NULL || checkpoint_sha256 == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  memcpy(manifest->inner.train_dataset_sha256, train_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(manifest->inner.eval_dataset_sha256, eval_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(manifest->inner.tokenizer_sha256, tokenizer_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(manifest->inner.checkpoint_sha256, checkpoint_sha256,
         ET_EVAL5_SHA256_BYTES);
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_set_audit(
    et_eval5_aot_manifest *manifest, int32_t ngram_mode, int32_t ngram_order,
    int32_t forbid_exact_match, double min_ngram_overlap,
    double max_allowed_rate, const int64_t *summary_i64,
    double contamination_rate) {
  if (manifest == NULL || summary_i64 == NULL || ngram_mode < 0 ||
      ngram_order <= 0 || forbid_exact_match < 0 || forbid_exact_match > 1) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  for (int i = 0; i < ET_EVAL5_AOT_AUDIT_SUMMARY_LEN; ++i) {
    if (summary_i64[i] < 0) {
      return ET_EVAL5_ERR_ARGUMENT;
    }
  }
  manifest->inner.contamination = (et_eval5_contamination_config){
      (uint32_t)ngram_mode, (uint32_t)ngram_order, (uint32_t)forbid_exact_match,
      min_ngram_overlap, max_allowed_rate};
  manifest->inner.audit = (et_eval5_contamination_summary){
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_EVAL_DOCS],
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS],
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_EXACT_MATCHES],
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_NGRAM_DOCS],
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_TOTAL_NGRAMS],
      (uint64_t)summary_i64[ET_EVAL5_AOT_AUDIT_MATCHED_NGRAMS],
      contamination_rate};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_set_lm(et_eval5_aot_manifest *manifest,
                                     int64_t token_count, int64_t byte_count,
                                     const double *metrics_f64) {
  if (manifest == NULL || metrics_f64 == NULL || token_count <= 0 ||
      byte_count <= 0) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  manifest->inner.metric_mask |= (uint32_t)ET_EVAL5_METRIC_LM;
  manifest->inner.lm = (et_eval5_lm_metrics){
      (uint64_t)token_count, (uint64_t)byte_count,
      metrics_f64[ET_EVAL5_AOT_LM_TOTAL_NLL],
      metrics_f64[ET_EVAL5_AOT_LM_MEAN_NLL],
      metrics_f64[ET_EVAL5_AOT_LM_PERPLEXITY],
      metrics_f64[ET_EVAL5_AOT_LM_BITS_PER_BYTE]};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_set_mc(et_eval5_aot_manifest *manifest,
                                     int32_t normalize_by_length,
                                     int64_t correct_count, int64_t total_count,
                                     double accuracy) {
  if (manifest == NULL || normalize_by_length < 0 || normalize_by_length > 1 ||
      correct_count < 0 || total_count <= 0 || correct_count > total_count) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  manifest->inner.metric_mask |= (uint32_t)ET_EVAL5_METRIC_MC;
  manifest->inner.mc_normalize_by_length = (uint32_t)normalize_by_length;
  manifest->inner.mc = (et_eval5_mc_metrics){
      (uint64_t)correct_count, (uint64_t)total_count, accuracy};
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_verify_clean(
    const et_eval5_aot_manifest *manifest) {
  if (manifest == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  size_t required = 0;
  const et_eval5_status status =
      et_eval5_manifest_format(&manifest->inner, NULL, 0u, &required);
  if (status != ET_EVAL5_OK) {
    return status;
  }
  if (manifest->inner.audit.contamination_rate >
          manifest->inner.contamination.max_allowed_rate ||
      (manifest->inner.contamination.forbid_exact_match != 0u &&
       manifest->inner.audit.exact_doc_matches != 0u)) {
    return ET_EVAL5_ERR_CONTAMINATED;
  }
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_format(const et_eval5_aot_manifest *manifest,
                                     char *out_buf, int64_t capacity,
                                     int64_t *out_size) {
  size_t cap = 0;
  if (manifest == NULL || out_size == NULL ||
      !nonneg_i64_to_size(capacity, &cap)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  size_t used = 0;
  const et_eval5_status status =
      et_eval5_manifest_format(&manifest->inner, out_buf, cap, &used);
  if (status == ET_EVAL5_OK || status == ET_EVAL5_ERR_BOUNDS) {
    int64_t used_i64 = 0;
    if (!u64_to_i64((uint64_t)used, &used_i64)) {
      return ET_EVAL5_ERR_OVERFLOW;
    }
    *out_size = used_i64;
  }
  return status;
}

int32_t et_eval5_aot_manifest_parse(const char *bytes, int64_t size,
                                    et_eval5_aot_manifest *out) {
  size_t sz = 0;
  if (bytes == NULL || out == NULL || !nonneg_i64_to_size(size, &sz)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  et_eval5_manifest parsed;
  memset(&parsed, 0, sizeof(parsed));
  const et_eval5_status status = et_eval5_manifest_parse(bytes, sz, &parsed);
  if (status != ET_EVAL5_OK) {
    return status;
  }
  out->inner = parsed;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_write_atomic(
    const char *path, const et_eval5_aot_manifest *manifest) {
  if (path == NULL || path[0] == '\0' || manifest == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  /* Fail closed: refuse to publish contaminated or schema-invalid manifests. */
  const et_eval5_status gate = et_eval5_aot_manifest_verify_clean(manifest);
  if (gate != ET_EVAL5_OK) {
    return gate;
  }
  /* Refuse symlink or non-regular destination entries. */
  struct stat st;
  if (lstat(path, &st) == 0) {
    if (!S_ISREG(st.st_mode)) {
      return ET_EVAL5_ERR_IO;
    }
  } else if (errno != ENOENT) {
    return ET_EVAL5_ERR_IO;
  }
  return et_eval5_manifest_write_atomic(path, &manifest->inner);
}

int32_t et_eval5_aot_manifest_read_file(const char *path,
                                        et_eval5_aot_manifest *out) {
  if (path == NULL || path[0] == '\0' || out == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  et_eval5_manifest parsed;
  memset(&parsed, 0, sizeof(parsed));
  const et_eval5_status status = et_eval5_manifest_read_file(path, &parsed);
  if (status != ET_EVAL5_OK) {
    return status;
  }
  out->inner = parsed;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_get_metrics(const et_eval5_aot_manifest *manifest,
                                          int64_t *counts_i64,
                                          double *metrics_f64,
                                          uint8_t *manifest_sha256_out) {
  if (manifest == NULL || counts_i64 == NULL || metrics_f64 == NULL ||
      manifest_sha256_out == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  const et_eval5_manifest *m = &manifest->inner;
  int64_t c[ET_EVAL5_AOT_MANIFEST_COUNTS_LEN];
  c[0] = (int64_t)m->metric_mask;
  c[1] = (int64_t)m->mc_normalize_by_length;
  if (!u64_to_i64(m->lm.token_count, &c[2]) ||
      !u64_to_i64(m->lm.byte_count, &c[3]) ||
      !u64_to_i64(m->mc.correct_count, &c[4]) ||
      !u64_to_i64(m->mc.total_count, &c[5]) ||
      !u64_to_i64(m->audit.eval_documents, &c[6]) ||
      !u64_to_i64(m->audit.contaminated_documents, &c[7]) ||
      !u64_to_i64(m->audit.exact_doc_matches, &c[8]) ||
      !u64_to_i64(m->audit.ngram_contaminated_docs, &c[9]) ||
      !u64_to_i64(m->audit.total_eval_ngrams, &c[10]) ||
      !u64_to_i64(m->audit.matched_eval_ngrams, &c[11])) {
    return ET_EVAL5_ERR_OVERFLOW;
  }
  memcpy(counts_i64, c, sizeof(c));
  metrics_f64[0] = m->lm.total_nll;
  metrics_f64[1] = m->lm.mean_nll;
  metrics_f64[2] = m->lm.perplexity;
  metrics_f64[3] = m->lm.bits_per_byte;
  metrics_f64[4] = m->mc.accuracy;
  metrics_f64[5] = m->audit.contamination_rate;
  memcpy(manifest_sha256_out, m->manifest_sha256, ET_EVAL5_SHA256_BYTES);
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_manifest_get_digests(const et_eval5_aot_manifest *manifest,
                                          uint8_t *train_sha256,
                                          uint8_t *eval_sha256,
                                          uint8_t *tokenizer_sha256,
                                          uint8_t *checkpoint_sha256) {
  if (manifest == NULL || train_sha256 == NULL || eval_sha256 == NULL ||
      tokenizer_sha256 == NULL || checkpoint_sha256 == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  memcpy(train_sha256, manifest->inner.train_dataset_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(eval_sha256, manifest->inner.eval_dataset_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(tokenizer_sha256, manifest->inner.tokenizer_sha256,
         ET_EVAL5_SHA256_BYTES);
  memcpy(checkpoint_sha256, manifest->inner.checkpoint_sha256,
         ET_EVAL5_SHA256_BYTES);
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_sha256_bytes(const uint8_t *data, int64_t size,
                                  uint8_t *out_digest) {
  size_t sz = 0;
  if (out_digest == NULL || !nonneg_i64_to_size(size, &sz) ||
      (sz != 0u && data == NULL)) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  bridge_sha256_state state;
  bridge_sha256_init(&state);
  if (sz != 0u) {
    bridge_sha256_update(&state, data, sz);
  }
  bridge_sha256_final(&state, out_digest);
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_sha256_file(const char *path, uint8_t *out_digest) {
  if (path == NULL || path[0] == '\0' || out_digest == NULL) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    return ET_EVAL5_ERR_IO;
  }
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    (void)close(fd);
    return ET_EVAL5_ERR_IO;
  }
  bridge_sha256_state state;
  bridge_sha256_init(&state);
  uint8_t chunk[4096];
  int32_t status = ET_EVAL5_OK;
  for (;;) {
    const ssize_t n = read(fd, chunk, sizeof(chunk));
    if (n == 0) {
      break;
    }
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      status = ET_EVAL5_ERR_IO;
      break;
    }
    bridge_sha256_update(&state, chunk, (size_t)n);
  }
  if (close(fd) != 0 && status == ET_EVAL5_OK) {
    status = ET_EVAL5_ERR_IO;
  }
  if (status != ET_EVAL5_OK) {
    return status;
  }
  bridge_sha256_final(&state, out_digest);
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_store_u32(uint32_t *buffer, int64_t capacity,
                               int64_t index, int64_t value) {
  if (buffer == NULL || capacity <= 0 || index < 0 || index >= capacity ||
      value < 0 || (uint64_t)value > UINT32_MAX) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  buffer[(size_t)index] = (uint32_t)value;
  return ET_EVAL5_OK;
}

int32_t et_eval5_aot_store_f32(float *buffer, int64_t capacity, int64_t index,
                               double value) {
  if (buffer == NULL || capacity <= 0 || index < 0 || index >= capacity) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  if (isnan(value)) {
    buffer[(size_t)index] = NAN;
  } else if (value > (double)FLT_MAX) {
    buffer[(size_t)index] = INFINITY;
  } else if (value < -(double)FLT_MAX) {
    buffer[(size_t)index] = -INFINITY;
  } else {
    buffer[(size_t)index] = (float)value;
  }
  return ET_EVAL5_OK;
}

double et_eval5_aot_load_f64(const double *buffer, int64_t capacity,
                             int64_t index) {
  if (buffer == NULL || capacity <= 0 || index < 0 || index >= capacity) {
    return NAN;
  }
  return buffer[(size_t)index];
}

int64_t et_eval5_aot_load_i64(const int64_t *buffer, int64_t capacity,
                              int64_t index) {
  if (buffer == NULL || capacity <= 0 || index < 0 || index >= capacity) {
    return INT64_MIN;
  }
  return buffer[(size_t)index];
}

int32_t et_eval5_aot_bytes_equal(const uint8_t *a, const uint8_t *b,
                                 int64_t count, int32_t *equal_out) {
  size_t len = 0;
  if (equal_out == NULL || !nonneg_i64_to_size(count, &len) ||
      (len != 0u && (a == NULL || b == NULL))) {
    return ET_EVAL5_ERR_ARGUMENT;
  }
  *equal_out = (len == 0u || memcmp(a, b, len) == 0) ? 1 : 0;
  return ET_EVAL5_OK;
}
