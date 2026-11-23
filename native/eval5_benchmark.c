#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <errno.h>
#include <fenv.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "EVAL5 requires 8-bit bytes and at most 64-bit size_t");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   sizeof(float) == 4 && DBL_MANT_DIG == 53 &&
                   DBL_MAX_EXP == 1024 && sizeof(double) == sizeof(uint64_t),
               "EVAL5 requires binary32 and binary64");

/* Standalone copy of the SHA-256 core in c2_checkpoint_format.c. */
typedef struct sha256_state {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} sha256_state;

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_block(sha256_state *state, const uint8_t block[64]) {
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
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) |
           ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = state->h[0]; b = state->h[1]; c = state->h[2]; d = state->h[3];
  e = state->h[4]; f = state->h[5]; g = state->h[6]; h = state->h[7];
  for (i = 0; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
  state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

static void sha256_init(sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(state->h, initial, sizeof(initial));
  state->length = 0u;
  state->used = 0u;
}

static void sha256_update(sha256_state *state, const uint8_t *bytes,
                          size_t length) {
  state->length += (uint64_t)length;
  while (length != 0u) {
    size_t take = 64u - state->used;
    if (take > length) take = length;
    memcpy(state->block + state->used, bytes, take);
    state->used += take; bytes += take; length -= take;
    if (state->used == 64u) {
      sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void sha256_final(sha256_state *state, uint8_t digest[32]) {
  uint64_t bits = state->length * UINT64_C(8);
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0; i < 8u; ++i) state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256(const void *data, size_t size, uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  sha256_update(&state, data, size);
  sha256_final(&state, digest);
}

static int valid_rate(double value) {
  return isfinite(value) && !signbit(value) && value <= 1.0;
}

static int valid_policy(const et_eval5_contamination_config *config) {
  return (config->ngram_mode == ET_EVAL5_NGRAM_BYTES ||
          config->ngram_mode == ET_EVAL5_NGRAM_TOKENS) &&
         config->ngram_order != 0 && config->forbid_exact_match <= 1 &&
         valid_rate(config->min_ngram_overlap) &&
         valid_rate(config->max_allowed_rate);
}

static double fraction(uint64_t numerator, uint64_t denominator) {
  return (double)numerator / (double)denominator;
}

typedef struct {
  uint8_t digest[ET_EVAL5_SHA256_BYTES];
  const et_eval5_document *document;
} document_key;

static document_key make_document_key(const et_eval5_document *document) {
  document_key key = {.document = document};
  sha256(document->bytes, document->byte_count, key.digest);
  return key;
}

static int compare_documents(const void *left, const void *right) {
  const document_key *a = left;
  const document_key *b = right;
  int order = memcmp(a->digest, b->digest, sizeof(a->digest));
  if (order != 0) return order;

  size_t size = a->document->byte_count;
  size_t other_size = b->document->byte_count;
  if (size != other_size) return size < other_size ? -1 : 1;
  return size == 0 ? 0 : memcmp(a->document->bytes, b->document->bytes, size);
}

#define NGRAM_BASE UINT64_C(257)

typedef struct {
  uint64_t hash;
  const void *units;
  size_t order;
} ngram_key;

typedef int (*key_compare)(const void *, const void *);

static int compare_byte_ngrams(const void *left, const void *right) {
  const ngram_key *a = left;
  const ngram_key *b = right;
  if (a->hash != b->hash) return a->hash < b->hash ? -1 : 1;
  return memcmp(a->units, b->units, a->order);
}

static int compare_token_ngrams(const void *left, const void *right) {
  const ngram_key *a = left;
  const ngram_key *b = right;
  if (a->hash != b->hash) return a->hash < b->hash ? -1 : 1;
  const uint32_t *a_tokens = a->units;
  const uint32_t *b_tokens = b->units;
  for (size_t i = 0; i < a->order; ++i) {
    if (a_tokens[i] != b_tokens[i]) return a_tokens[i] < b_tokens[i] ? -1 : 1;
  }
  return 0;
}

static size_t window_count(const et_eval5_document *doc,
                           const et_eval5_contamination_config *config) {
  size_t size = config->ngram_mode == ET_EVAL5_NGRAM_BYTES
                    ? doc->byte_count : doc->token_count;
  return size < config->ngram_order ? 0 : size - config->ngram_order + 1;
}

static et_eval5_status count_windows(const et_eval5_document *documents,
                                     size_t count,
                                     const et_eval5_contamination_config *config,
                                     uint64_t *out) {
  uint64_t total = 0;
  for (size_t i = 0; i < count; ++i) {
    const et_eval5_document *doc = &documents[i];
    if (doc->byte_count > PTRDIFF_MAX || doc->byte_count > UINT64_MAX / 8 ||
        (config->ngram_mode == ET_EVAL5_NGRAM_TOKENS &&
         doc->token_count > PTRDIFF_MAX / sizeof(*doc->tokens)))
      return ET_EVAL5_ERR_OVERFLOW;
    if ((doc->byte_count != 0 && doc->bytes == NULL) ||
        (config->ngram_mode == ET_EVAL5_NGRAM_TOKENS &&
         doc->token_count != 0 && doc->tokens == NULL))
      return ET_EVAL5_ERR_ARGUMENT;
    size_t windows = window_count(doc, config);
    if (windows > UINT64_MAX - total) return ET_EVAL5_ERR_OVERFLOW;
    total += windows;
  }
  *out = total;
  return ET_EVAL5_OK;
}

typedef struct {
  const et_eval5_document *document;
  const et_eval5_contamination_config *config;
  uint64_t hash;
  uint64_t leading_weight;
  size_t position;
} ngram_cursor;

static uint64_t ngram_unit(const ngram_cursor *cursor, size_t position) {
  return cursor->config->ngram_mode == ET_EVAL5_NGRAM_BYTES
             ? (uint64_t)cursor->document->bytes[position] + 1
             : (uint64_t)cursor->document->tokens[position] + 1;
}

static ngram_key next_ngram(ngram_cursor *cursor) {
  size_t position = cursor->position++;
  size_t order = cursor->config->ngram_order;
  if (position == 0) {
    cursor->leading_weight = 1;
    for (size_t i = 0; i < order; ++i) {
      cursor->hash = cursor->hash * NGRAM_BASE + ngram_unit(cursor, i);
      if (i != 0) cursor->leading_weight *= NGRAM_BASE;
    }
  } else {
    cursor->hash -= ngram_unit(cursor, position - 1) * cursor->leading_weight;
    cursor->hash = cursor->hash * NGRAM_BASE + ngram_unit(cursor, position + order - 1);
  }
  const void *units = cursor->config->ngram_mode == ET_EVAL5_NGRAM_BYTES
      ? (const void *)(cursor->document->bytes + position)
      : (const void *)(cursor->document->tokens + position);
  return (ngram_key){cursor->hash, units, order};
}

static size_t index_ngrams(ngram_key *keys, size_t total,
                           const et_eval5_document *train, size_t train_count,
                           const et_eval5_contamination_config *config,
                           key_compare compare) {
  if (total == 0) return 0;
  size_t used = 0;
  for (size_t i = 0; i < train_count; ++i) {
    ngram_cursor cursor = {.document = &train[i], .config = config};
    size_t windows = window_count(&train[i], config);
    for (size_t j = 0; j < windows; ++j) keys[used++] = next_ngram(&cursor);
  }
  qsort(keys, used, sizeof(*keys), compare);

  size_t unique = 1;
  for (size_t i = 1; i < used; ++i) {
    if (compare(&keys[unique - 1], &keys[i]) != 0)
      keys[unique++] = keys[i];
  }
  return unique;
}

et_eval5_status et_eval5_audit_contamination(
    const et_eval5_document *train, size_t train_count,
    const et_eval5_document *eval, size_t eval_count,
    const et_eval5_contamination_config *config,
    uint64_t *contaminated_indices, size_t index_capacity,
    et_eval5_contamination_summary *out) {
  if (config == NULL || out == NULL || eval == NULL || eval_count == 0 ||
      contaminated_indices == NULL || (train == NULL && train_count != 0) ||
      !valid_policy(config))
    return ET_EVAL5_ERR_ARGUMENT;
  if (fegetround() != FE_TONEAREST) return ET_EVAL5_ERR_FP_ENV;
  if (train_count > PTRDIFF_MAX / sizeof(*train) ||
      train_count > PTRDIFF_MAX / sizeof(document_key) ||
      eval_count > PTRDIFF_MAX / sizeof(*eval) ||
      eval_count > PTRDIFF_MAX / sizeof(*contaminated_indices))
    return ET_EVAL5_ERR_OVERFLOW;
  if (index_capacity < eval_count) return ET_EVAL5_ERR_BOUNDS;

  uint64_t train_windows;
  et_eval5_contamination_summary summary = {.eval_documents = eval_count};
  et_eval5_status status = count_windows(train, train_count, config, &train_windows);
  if (status != ET_EVAL5_OK) return status;
  status = count_windows(eval, eval_count, config, &summary.total_eval_ngrams);
  if (status != ET_EVAL5_OK) return status;
  if (train_windows > PTRDIFF_MAX / sizeof(ngram_key)) return ET_EVAL5_ERR_OVERFLOW;

  document_key *documents = NULL;
  ngram_key *ngrams = NULL;
  if (train_count != 0) {
    documents = malloc(train_count * sizeof(*documents));
    if (documents == NULL) return ET_EVAL5_ERR_NOMEM;
  }
  if (train_windows != 0) {
    ngrams = malloc((size_t)train_windows * sizeof(*ngrams));
    if (ngrams == NULL) {
      free(documents);
      return ET_EVAL5_ERR_NOMEM;
    }
  }
  if (train_count != 0) {
    for (size_t i = 0; i < train_count; ++i) documents[i] = make_document_key(&train[i]);
    qsort(documents, train_count, sizeof(*documents), compare_documents);
  }
  key_compare compare = config->ngram_mode == ET_EVAL5_NGRAM_BYTES
                            ? compare_byte_ngrams : compare_token_ngrams;
  size_t unique = index_ngrams(ngrams, (size_t)train_windows, train, train_count,
                               config, compare);

  for (size_t i = 0; i < eval_count; ++i) {
    int exact = 0;
    if (train_count != 0) {
      document_key key = make_document_key(&eval[i]);
      exact = bsearch(&key, documents, train_count, sizeof(*documents),
                      compare_documents) != NULL;
    }
    size_t windows = window_count(&eval[i], config);
    uint64_t matched = 0;
    if (unique != 0) {
      ngram_cursor cursor = {.document = &eval[i], .config = config};
      for (size_t j = 0; j < windows; ++j) {
        ngram_key key = next_ngram(&cursor);
        matched += bsearch(&key, ngrams, unique, sizeof(*ngrams), compare) != NULL;
      }
    }
    int overlap = matched != 0 &&
                  fraction(matched, windows) >= config->min_ngram_overlap;
    summary.matched_eval_ngrams += matched;
    summary.exact_doc_matches += (uint64_t)exact;
    summary.ngram_contaminated_docs += (uint64_t)overlap;
    if (exact || overlap) contaminated_indices[summary.contaminated_documents++] = i;
  }
  free(ngrams);
  free(documents);
  summary.contamination_rate = fraction(summary.contaminated_documents, eval_count);
  *out = summary;
  if (summary.contamination_rate > config->max_allowed_rate ||
      (config->forbid_exact_match && summary.exact_doc_matches != 0))
    return ET_EVAL5_ERR_CONTAMINATED;
  return ET_EVAL5_OK;
}

static et_eval5_status sequence_nll(const et_eval5_sequence *sequence,
                                    double *out) {
  if (sequence == NULL || sequence->logits == NULL || sequence->target_ids == NULL ||
      sequence->token_count == 0 || sequence->vocab_size == 0)
    return ET_EVAL5_ERR_ARGUMENT;
  size_t rows = sequence->token_count;
  size_t cols = sequence->vocab_size;
  if (rows > PTRDIFF_MAX / sizeof(*sequence->target_ids) ||
      cols > (PTRDIFF_MAX / sizeof(float)) / rows ||
      cols - 1 > UINT32_MAX)
    return ET_EVAL5_ERR_OVERFLOW;
  if (sequence->logit_count != rows * cols) return ET_EVAL5_ERR_BOUNDS;

  double total = 0.0;
  for (size_t row = 0; row < rows; ++row) {
    uint32_t target = sequence->target_ids[row];
    if (target >= cols) return ET_EVAL5_ERR_TARGET;
    const float *logits = sequence->logits + row * cols;
    size_t max_index = 0;
    for (size_t col = 0; col < cols; ++col) {
      if (!isfinite(logits[col])) return ET_EVAL5_ERR_NONFINITE;
      if (logits[col] > logits[max_index]) max_index = col;
    }
    double maximum = logits[max_index];
    double tail = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      if (col != max_index) tail += exp((double)logits[col] - maximum);
    }
    double nll = (maximum - (double)logits[target]) + log1p(tail);
    total += nll;
  }
  *out = total;
  return ET_EVAL5_OK;
}

et_eval5_status et_eval5_score_lm(const et_eval5_sequence *sequence,
                                 uint64_t byte_count, et_eval5_lm_metrics *out) {
  if (out == NULL || byte_count == 0) return ET_EVAL5_ERR_ARGUMENT;
  if (fegetround() != FE_TONEAREST) return ET_EVAL5_ERR_FP_ENV;
  double total;
  et_eval5_status status = sequence_nll(sequence, &total);
  if (status != ET_EVAL5_OK) return status;

  double mean = total / (double)sequence->token_count;
  double perplexity = exp(mean);
  if (!isfinite(perplexity)) return ET_EVAL5_ERR_OVERFLOW;
  /* Correctly rounded binary64 ln(2). */
  const double ln2 = 0x1.62e42fefa39efp-1;
  double bytes_per_token = fraction(byte_count, sequence->token_count);
  *out = (et_eval5_lm_metrics){
      .token_count = sequence->token_count,
      .byte_count = byte_count,
      .total_nll = total,
      .mean_nll = mean,
      .perplexity = perplexity,
      .bits_per_byte = mean / (ln2 * bytes_per_token),
  };
  return ET_EVAL5_OK;
}

et_eval5_status et_eval5_score_mc(const et_eval5_choice_item *items,
                                 size_t item_count, uint32_t normalize_by_length,
                                 et_eval5_mc_metrics *out) {
  if (items == NULL || item_count == 0 || normalize_by_length > 1 || out == NULL)
    return ET_EVAL5_ERR_ARGUMENT;
  if (fegetround() != FE_TONEAREST) return ET_EVAL5_ERR_FP_ENV;
  if (item_count > PTRDIFF_MAX / sizeof(*items)) return ET_EVAL5_ERR_OVERFLOW;

  uint64_t correct = 0;
  for (size_t i = 0; i < item_count; ++i) {
    const et_eval5_choice_item *item = &items[i];
    if (item->choices == NULL || item->choice_count == 0)
      return ET_EVAL5_ERR_ARGUMENT;
    if (item->choice_count > PTRDIFF_MAX / sizeof(*item->choices))
      return ET_EVAL5_ERR_OVERFLOW;
    if (item->correct_index >= item->choice_count) return ET_EVAL5_ERR_TARGET;

    size_t best_index = 0;
    double best_cost = INFINITY;
    for (size_t j = 0; j < item->choice_count; ++j) {
      const et_eval5_sequence *choice = &item->choices[j];
      double cost;
      et_eval5_status status = sequence_nll(choice, &cost);
      if (status != ET_EVAL5_OK) return status;
      if (normalize_by_length) cost /= (double)choice->token_count;
      if (cost < best_cost) {
        best_cost = cost;
        best_index = j;
      }
    }
    correct += best_index == item->correct_index;
  }
  *out = (et_eval5_mc_metrics){correct, item_count, fraction(correct, item_count)};
  return ET_EVAL5_OK;
}

typedef enum { NAME, SHA256, U32, U64, F64 } field_kind;

typedef struct {
  const char *key;
  field_kind kind;
  size_t offset;
} manifest_field;

#define FIELD(member, kind) {#member, kind, offsetof(et_eval5_manifest, member)}
static const manifest_field fields[] = {
    FIELD(suite, NAME),
    FIELD(task, NAME),
    FIELD(split, NAME),
    FIELD(train_dataset_sha256, SHA256),
    FIELD(eval_dataset_sha256, SHA256),
    FIELD(tokenizer_sha256, SHA256),
    FIELD(checkpoint_sha256, SHA256),
    FIELD(context_length, U64),
    FIELD(max_samples, U64),
    FIELD(contamination.ngram_mode, U32),
    FIELD(contamination.ngram_order, U32),
    FIELD(contamination.forbid_exact_match, U32),
    FIELD(contamination.min_ngram_overlap, F64),
    FIELD(contamination.max_allowed_rate, F64),
    FIELD(metric_mask, U32),
    FIELD(mc_normalize_by_length, U32),
    FIELD(lm.token_count, U64),
    FIELD(lm.byte_count, U64),
    FIELD(lm.total_nll, F64),
    FIELD(lm.mean_nll, F64),
    FIELD(lm.perplexity, F64),
    FIELD(lm.bits_per_byte, F64),
    FIELD(mc.correct_count, U64),
    FIELD(mc.total_count, U64),
    FIELD(mc.accuracy, F64),
    FIELD(audit.eval_documents, U64),
    FIELD(audit.contaminated_documents, U64),
    FIELD(audit.exact_doc_matches, U64),
    FIELD(audit.ngram_contaminated_docs, U64),
    FIELD(audit.total_eval_ngrams, U64),
    FIELD(audit.matched_eval_ngrams, U64),
    FIELD(audit.contamination_rate, F64),
};
#undef FIELD

static const char manifest_prefix[] = EVAL5_MANIFEST_V1 "\nversion=1.0\n";
static const char checksum_key[] = "manifest_sha256";
static const char hex_digits[] = "0123456789abcdef";

static int ascii_alnum(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9');
}

static int valid_name(const char *name) {
  if (!ascii_alnum((unsigned char)name[0])) return 0;
  for (size_t i = 1; i <= ET_EVAL5_NAME_MAX; ++i) {
    unsigned char c = (unsigned char)name[i];
    if (c == '\0') return 1;
    if (!ascii_alnum(c) && c != '-' && c != '_' && c != '.') return 0;
  }
  return 0;
}

static int nonnegative(double value) {
  return isfinite(value) && !signbit(value);
}

static int valid_lm(const et_eval5_lm_metrics *lm, int enabled) {
  if (!nonnegative(lm->total_nll) || !nonnegative(lm->mean_nll) ||
      !nonnegative(lm->perplexity) || !nonnegative(lm->bits_per_byte))
    return 0;
  if (!enabled)
    return lm->token_count == 0 && lm->byte_count == 0 && lm->total_nll == 0.0 &&
           lm->mean_nll == 0.0 && lm->perplexity == 0.0 && lm->bits_per_byte == 0.0;
  return lm->token_count != 0 && lm->byte_count != 0 && lm->perplexity >= 1.0 &&
         lm->mean_nll == lm->total_nll / (double)lm->token_count;
}

static int valid_mc(const et_eval5_manifest *manifest) {
  const et_eval5_mc_metrics *mc = &manifest->mc;
  if (!valid_rate(mc->accuracy)) return 0;
  if (!(manifest->metric_mask & ET_EVAL5_METRIC_MC))
    return mc->total_count == 0 && mc->correct_count == 0 && mc->accuracy == 0.0 &&
           manifest->mc_normalize_by_length == 0;
  return manifest->mc_normalize_by_length <= 1 && mc->total_count != 0 &&
         mc->correct_count <= mc->total_count &&
         mc->accuracy == fraction(mc->correct_count, mc->total_count);
}

static int valid_audit(const et_eval5_contamination_summary *audit) {
  return audit->eval_documents != 0 &&
         audit->contaminated_documents <= audit->eval_documents &&
         audit->exact_doc_matches <= audit->contaminated_documents &&
         audit->ngram_contaminated_docs <= audit->contaminated_documents &&
         audit->contaminated_documents - audit->exact_doc_matches <=
             audit->ngram_contaminated_docs &&
         audit->matched_eval_ngrams <= audit->total_eval_ngrams &&
         audit->ngram_contaminated_docs <= audit->matched_eval_ngrams &&
         valid_rate(audit->contamination_rate) &&
         audit->contamination_rate ==
             fraction(audit->contaminated_documents, audit->eval_documents);
}

static et_eval5_status validate_manifest(const et_eval5_manifest *manifest) {
  if (manifest->version_major != ET_EVAL5_VERSION_MAJOR ||
      manifest->version_minor != ET_EVAL5_VERSION_MINOR)
    return ET_EVAL5_ERR_VERSION;
  if (fegetround() != FE_TONEAREST) return ET_EVAL5_ERR_FP_ENV;
  if (!valid_name(manifest->suite) || !valid_name(manifest->task) ||
      !valid_name(manifest->split) || manifest->context_length == 0 ||
      !valid_policy(&manifest->contamination) || manifest->metric_mask == 0 ||
      (manifest->metric_mask & ~(ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC)) != 0 ||
      !valid_lm(&manifest->lm, manifest->metric_mask & ET_EVAL5_METRIC_LM) ||
      !valid_mc(manifest) || !valid_audit(&manifest->audit))
    return ET_EVAL5_ERR_FORMAT;
  return ET_EVAL5_OK;
}

static void format_digest(const uint8_t digest[32], char out[65]) {
  for (size_t i = 0; i < ET_EVAL5_SHA256_BYTES; ++i) {
    out[2 * i] = hex_digits[digest[i] >> 4];
    out[2 * i + 1] = hex_digits[digest[i] & 15];
  }
  out[64] = '\0';
}

static int format_field(const et_eval5_manifest *manifest,
                        const manifest_field *field, char *out, size_t capacity) {
  const void *value = (const char *)manifest + field->offset;
  const char *key = field->key;
  switch (field->kind) {
    case NAME:
      return snprintf(out, capacity, "%s=%s\n", key, (const char *)value);
    case SHA256: {
      char hex[65];
      format_digest(value, hex);
      return snprintf(out, capacity, "%s=%s\n", key, hex);
    }
    case U32:
      return snprintf(out, capacity, "%s=%" PRIu32 "\n", key, *(const uint32_t *)value);
    case U64:
      return snprintf(out, capacity, "%s=%" PRIu64 "\n", key, *(const uint64_t *)value);
    case F64: {
      uint64_t bits;
      memcpy(&bits, value, sizeof(bits));
      return snprintf(out, capacity, "%s=%016" PRIx64 "\n", key, bits);
    }
  }
  return -1;
}

et_eval5_status et_eval5_manifest_format(const et_eval5_manifest *manifest,
                                        char *out, size_t capacity, size_t *size) {
  if (manifest == NULL || size == NULL || (out == NULL && capacity != 0))
    return ET_EVAL5_ERR_ARGUMENT;
  et_eval5_status status = validate_manifest(manifest);
  if (status != ET_EVAL5_OK) return status;

  char buffer[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t used = sizeof(manifest_prefix) - 1;
  memcpy(buffer, manifest_prefix, used);
  for (size_t i = 0; i < sizeof(fields) / sizeof(*fields); ++i) {
    int length = format_field(manifest, &fields[i], buffer + used, sizeof(buffer) - used);
    if (length < 0 || (size_t)length >= sizeof(buffer) - used)
      return ET_EVAL5_ERR_BOUNDS;
    used += (size_t)length;
  }
  uint8_t digest[32];
  char hex[65];
  sha256(buffer, used, digest);
  format_digest(digest, hex);
  int length = snprintf(buffer + used, sizeof(buffer) - used, "%s=%s\n", checksum_key, hex);
  if (length < 0 || (size_t)length >= sizeof(buffer) - used)
    return ET_EVAL5_ERR_BOUNDS;
  used += (size_t)length;
  *size = used;
  if (out == NULL) return ET_EVAL5_OK;
  if (capacity <= used) return ET_EVAL5_ERR_BOUNDS;
  memcpy(out, buffer, used + 1);
  return ET_EVAL5_OK;
}

typedef struct {
  const char *data;
  size_t size;
} text_span;

static int take_line(text_span *input, text_span *line) {
  const char *newline = memchr(input->data, '\n', input->size);
  if (newline == NULL) return 0;
  size_t length = (size_t)(newline - input->data);
  *line = (text_span){input->data, length};
  input->data = newline + 1;
  input->size -= length + 1;
  return 1;
}

static int take_value(text_span *input, const char *key, text_span *value) {
  text_span line;
  size_t key_size = strlen(key);
  if (!take_line(input, &line) || line.size <= key_size ||
      memcmp(line.data, key, key_size) != 0 || line.data[key_size] != '=')
    return 0;
  *value = (text_span){line.data + key_size + 1, line.size - key_size - 1};
  return 1;
}

static int parse_uint(text_span value, uint64_t *out) {
  if (value.size == 0 || value.size > 20 || (value.size > 1 && value.data[0] == '0'))
    return 0;
  uint64_t result = 0;
  for (size_t i = 0; i < value.size; ++i) {
    unsigned int digit = (unsigned int)((unsigned char)value.data[i] - '0');
    if (digit > 9 || result > (UINT64_MAX - digit) / 10) return 0;
    result = 10 * result + digit;
  }
  *out = result;
  return 1;
}

static int hex_value(unsigned char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

static int parse_digest(text_span value, uint8_t out[32]) {
  if (value.size != 64) return 0;
  for (size_t i = 0; i < ET_EVAL5_SHA256_BYTES; ++i) {
    int high = hex_value((unsigned char)value.data[2 * i]);
    int low = hex_value((unsigned char)value.data[2 * i + 1]);
    if (high < 0 || low < 0) return 0;
    out[i] = (uint8_t)(16 * high + low);
  }
  return 1;
}

static int parse_f64(text_span value, double *out) {
  if (value.size != 16) return 0;
  uint64_t bits = 0;
  for (size_t i = 0; i < value.size; ++i) {
    int digit = hex_value((unsigned char)value.data[i]);
    if (digit < 0) return 0;
    bits = (bits << 4) | (uint64_t)digit;
  }
  /* Reject infinities and NaNs before interpreting the bits as a double. */
  if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
    return 0;
  memcpy(out, &bits, sizeof(bits));
  return 1;
}

static int parse_field(text_span value, const manifest_field *field,
                       et_eval5_manifest *manifest) {
  void *out = (char *)manifest + field->offset;
  switch (field->kind) {
    case NAME:
      if (value.size > ET_EVAL5_NAME_MAX || memchr(value.data, '\0', value.size))
        return 0;
      memcpy(out, value.data, value.size);
      ((char *)out)[value.size] = '\0';
      return 1;
    case SHA256:
      return parse_digest(value, out);
    case U32: {
      uint64_t number;
      if (!parse_uint(value, &number) || number > UINT32_MAX) return 0;
      *(uint32_t *)out = (uint32_t)number;
      return 1;
    }
    case U64:
      return parse_uint(value, out);
    case F64:
      return parse_f64(value, out);
  }
  return 0;
}

et_eval5_status et_eval5_manifest_parse(const char *bytes, size_t size,
                                       et_eval5_manifest *out) {
  if (bytes == NULL || out == NULL) return ET_EVAL5_ERR_ARGUMENT;
  if (size > ET_EVAL5_MANIFEST_MAX_BYTES) return ET_EVAL5_ERR_BOUNDS;
  text_span input = {bytes, size};
  text_span value;
  if (!take_line(&input, &value) || value.size != sizeof(EVAL5_MANIFEST_V1) - 1 ||
      memcmp(value.data, EVAL5_MANIFEST_V1, value.size) != 0)
    return ET_EVAL5_ERR_FORMAT;
  if (!take_value(&input, "version", &value)) return ET_EVAL5_ERR_FORMAT;
  if (value.size != 3 || memcmp(value.data, "1.0", 3) != 0)
    return ET_EVAL5_ERR_VERSION;

  et_eval5_manifest manifest = {
      .version_major = ET_EVAL5_VERSION_MAJOR, .version_minor = ET_EVAL5_VERSION_MINOR,
  };
  for (size_t i = 0; i < sizeof(fields) / sizeof(*fields); ++i) {
    if (!take_value(&input, fields[i].key, &value) || !parse_field(value, &fields[i], &manifest))
      return ET_EVAL5_ERR_FORMAT;
  }

  size_t payload_size = (size_t)(input.data - bytes);
  if (!take_value(&input, checksum_key, &value) || input.size != 0 ||
      !parse_digest(value, manifest.manifest_sha256))
    return ET_EVAL5_ERR_FORMAT;
  uint8_t expected[32];
  sha256(bytes, payload_size, expected);
  if (memcmp(expected, manifest.manifest_sha256, sizeof(expected)) != 0)
    return ET_EVAL5_ERR_CHECKSUM;
  et_eval5_status status = validate_manifest(&manifest);
  if (status != ET_EVAL5_OK) return status;
  *out = manifest;
  return ET_EVAL5_OK;
}

static int write_all(int fd, const char *bytes, size_t size) {
  while (size != 0) {
    ssize_t written = write(fd, bytes, size);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return 0;
    bytes += written;
    size -= (size_t)written;
  }
  return 1;
}

static int sync_file(int fd) {
  int result;
  do {
    result = fsync(fd);
  } while (result < 0 && errno == EINTR);
  return result == 0;
}

et_eval5_status et_eval5_manifest_write_atomic(const char *path,
                                              const et_eval5_manifest *manifest) {
  if (path == NULL || path[0] == '\0') return ET_EVAL5_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0') return ET_EVAL5_ERR_ARGUMENT;
  char buffer[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t size;
  et_eval5_status status = et_eval5_manifest_format(manifest, buffer, sizeof(buffer), &size);
  if (status != ET_EVAL5_OK) return status;

  static const char suffix[] = ".eval5-manifest.XXXXXX";
  size_t prefix = slash == NULL ? 0 : (size_t)(slash - path) + 1;
  if (prefix > PTRDIFF_MAX - sizeof(suffix)) return ET_EVAL5_ERR_OVERFLOW;
  char *temporary = malloc(prefix + sizeof(suffix));
  if (temporary == NULL) return ET_EVAL5_ERR_NOMEM;
  memcpy(temporary, path, prefix);
  memcpy(temporary + prefix, suffix, sizeof(suffix));

  int fd = mkstemp(temporary);
  if (fd < 0) {
    free(temporary);
    return ET_EVAL5_ERR_IO;
  }
  int stored = write_all(fd, buffer, size) && sync_file(fd);
  if (close(fd) != 0) stored = 0;
  if (!stored || rename(temporary, path) != 0) {
    unlink(temporary);
    status = ET_EVAL5_ERR_IO;
  }
  free(temporary);
  return status;
}

et_eval5_status et_eval5_manifest_read_file(const char *path,
                                           et_eval5_manifest *out) {
  if (path == NULL || path[0] == '\0' || out == NULL) return ET_EVAL5_ERR_ARGUMENT;
  int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return ET_EVAL5_ERR_IO;

  struct stat metadata;
  et_eval5_status status = ET_EVAL5_OK;
  char buffer[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t used = 0;
  if (fstat(fd, &metadata) != 0 || !S_ISREG(metadata.st_mode)) {
    status = ET_EVAL5_ERR_IO;
  } else if (metadata.st_size < 0 || (uintmax_t)metadata.st_size >= sizeof(buffer)) {
    status = ET_EVAL5_ERR_BOUNDS;
  } else {
    while (used < sizeof(buffer)) {
      ssize_t count = read(fd, buffer + used, sizeof(buffer) - used);
      if (count == 0) break;
      if (count < 0) {
        if (errno == EINTR) continue;
        status = ET_EVAL5_ERR_IO;
        break;
      }
      used += (size_t)count;
    }
    if (status == ET_EVAL5_OK) {
      if (used == sizeof(buffer)) status = ET_EVAL5_ERR_BOUNDS;
      else if (used != (size_t)metadata.st_size) status = ET_EVAL5_ERR_FORMAT;
    }
  }
  if (close(fd) != 0 && status == ET_EVAL5_OK) status = ET_EVAL5_ERR_IO;
  if (status != ET_EVAL5_OK) return status;
  return et_eval5_manifest_parse(buffer, used, out);
}
