#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/constrained_decode_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Constrained decode engine requires 8-bit bytes and binary32");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "Constrained decode engine requires IEEE-754 binary32");
_Static_assert((ET_CD_MAX_STATES + 63u) / 64u == ET_CD_ACCEPT_WORDS_MAX,
               "ET_CD_ACCEPT_WORDS_MAX mismatch");
_Static_assert((ET_CD_MAX_VOCAB_SIZE + 63u) / 64u == ET_CD_VOCAB_WORDS_MAX,
               "ET_CD_VOCAB_WORDS_MAX mismatch");

/* ---- Internal Struct Definitions ---- */

struct et_cd_dfa {
  uint16_t num_states;
  uint16_t start_state;
  uint16_t live_state_count;
  uint16_t accepting_state_count;
  uint64_t accept_bitset[ET_CD_ACCEPT_WORDS_MAX];
  uint64_t live_bitset[ET_CD_ACCEPT_WORDS_MAX];
  uint16_t *next_state; /* num_states * 256 */
};

struct et_cd_vocab_index {
  uint16_t num_states;
  uint16_t start_state;
  uint16_t live_state_count;
  uint32_t vocab_size;
  uint32_t eos_token_id;
  uint32_t mask_words;
  uint64_t accept_bitset[ET_CD_ACCEPT_WORDS_MAX];
  uint64_t live_bitset[ET_CD_ACCEPT_WORDS_MAX];
  uint64_t *allowed_mask;  /* num_states * mask_words */
  uint32_t *allowed_count; /* num_states */
  uint16_t *dest_state;    /* num_states * vocab_size */
};

/* ---- Standalone SHA-256 (RFC 6234) ---- */

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
  for (i = 0u; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = state->h[0];
  b = state->h[1];
  c = state->h[2];
  d = state->h[3];
  e = state->h[4];
  f = state->h[5];
  g = state->h[6];
  h = state->h[7];
  for (i = 0u; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
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

static void sha256_init(sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
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
    if (take > length)
      take = length;
    memcpy(state->block + state->used, bytes, take);
    state->used += take;
    bytes += take;
    length -= take;
    if (state->used == 64u) {
      sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void sha256_final(sha256_state *state, uint8_t digest[32]) {
  const uint64_t bits = state->length * UINT64_C(8);
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0u; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256_bytes(const uint8_t *data, size_t size, uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  sha256_update(&state, data, size);
  sha256_final(&state, digest);
}

/* ---- Little-Endian Serialization Helpers ---- */

static void encode_u16_le(uint8_t out[2], uint16_t v) {
  out[0] = (uint8_t)(v & 0xffu);
  out[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static void encode_u32_le(uint8_t out[4], uint32_t v) {
  out[0] = (uint8_t)(v & 0xffu);
  out[1] = (uint8_t)((v >> 8u) & 0xffu);
  out[2] = (uint8_t)((v >> 16u) & 0xffu);
  out[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void encode_u64_le(uint8_t out[8], uint64_t v) {
  for (size_t i = 0u; i < 8u; ++i)
    out[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
}

static uint16_t decode_u16_le(const uint8_t in[2]) {
  return (uint16_t)((uint16_t)in[0] | ((uint16_t)in[1] << 8u));
}

static uint32_t decode_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) | ((uint32_t)in[2] << 16u) |
         ((uint32_t)in[3] << 24u);
}

static uint64_t decode_u64_le(const uint8_t in[8]) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    v |= ((uint64_t)in[i]) << (8u * i);
  return v;
}

/* ---- Bitset, FP Environment, and Span Helpers ---- */

static int bitset_get(const uint64_t *bits, uint32_t idx) {
  return (int)((bits[idx >> 6u] >> (idx & 63u)) & UINT64_C(1));
}

static void bitset_set(uint64_t *bits, uint32_t idx) {
  bits[idx >> 6u] |= (UINT64_C(1) << (idx & 63u));
}

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

static int spans_partially_overlap(const void *a, const void *b, size_t bytes) {
  if (a == b || bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - bytes || right > UINTPTR_MAX - bytes)
    return 1;
  return left < right + bytes && right < left + bytes;
}

/* ---- Co-Accessibility (Liveness) Precomputation & Pruning ---- */

/* Compute co-accessible states via reverse BFS from accept_bitset, prune all
 * non-live states and transitions into non-live states to ET_CD_DEAD_STATE,
 * and verify start_state is live. If out_pruned_any is non-NULL, sets *out_pruned_any
 * to 1 if any non-DEAD transition was rewritten to ET_CD_DEAD_STATE. */
static et_cd_status finalize_dfa_coaccessible(et_cd_dfa *dfa,
                                              int *out_pruned_any) {
  const uint16_t n = dfa->num_states;
  if (n == 0u || n > ET_CD_MAX_STATES || dfa->start_state >= n)
    return ET_CD_ERR_STATE;

  /* Count deduplicated reverse edges (u -> v) for CSR reverse graph. */
  uint32_t *in_deg = (uint32_t *)calloc((size_t)n, sizeof(uint32_t));
  if (in_deg == NULL)
    return ET_CD_ERR_NOMEM;

  size_t total_edges = 0u;
  const size_t state_words = ((size_t)n + 63u) / 64u;
  for (uint16_t u = 0u; u < n; ++u) {
    uint64_t seen[ET_CD_ACCEPT_WORDS_MAX];
    memset(seen, 0, state_words * sizeof(uint64_t));
    const uint16_t *row = dfa->next_state + (size_t)u * ET_CD_ALPHABET_SIZE;
    for (uint32_t b = 0u; b < ET_CD_ALPHABET_SIZE; ++b) {
      const uint16_t v = row[b];
      if (v == ET_CD_DEAD_STATE)
        continue;
      if (v >= n) {
        free(in_deg);
        return ET_CD_ERR_STATE;
      }
      if (!bitset_get(seen, v)) {
        bitset_set(seen, v);
        in_deg[v] += 1u;
        total_edges += 1u;
      }
    }
  }

  uint32_t *rev_off = (uint32_t *)malloc(((size_t)n + 1u) * sizeof(uint32_t));
  uint32_t *rev_cur = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
  uint16_t *rev_src =
      total_edges > 0u ? (uint16_t *)malloc(total_edges * sizeof(uint16_t))
                       : NULL;
  uint16_t *queue = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));

  if (rev_off == NULL || rev_cur == NULL ||
      (total_edges > 0u && rev_src == NULL) || queue == NULL) {
    free(in_deg);
    free(rev_off);
    free(rev_cur);
    free(rev_src);
    free(queue);
    return ET_CD_ERR_NOMEM;
  }

  rev_off[0] = 0u;
  for (uint16_t v = 0u; v < n; ++v) {
    rev_cur[v] = rev_off[v];
    rev_off[v + 1u] = rev_off[v] + in_deg[v];
  }

  for (uint16_t u = 0u; u < n; ++u) {
    uint64_t seen[ET_CD_ACCEPT_WORDS_MAX];
    memset(seen, 0, state_words * sizeof(uint64_t));
    const uint16_t *row = dfa->next_state + (size_t)u * ET_CD_ALPHABET_SIZE;
    for (uint32_t b = 0u; b < ET_CD_ALPHABET_SIZE; ++b) {
      const uint16_t v = row[b];
      if (v == ET_CD_DEAD_STATE)
        continue;
      if (!bitset_get(seen, v)) {
        bitset_set(seen, v);
        rev_src[rev_cur[v]++] = u;
      }
    }
  }

  /* Reverse BFS from all accepting states. */
  memset(dfa->live_bitset, 0, sizeof(dfa->live_bitset));
  uint32_t q_head = 0u;
  uint32_t q_tail = 0u;
  uint16_t accept_count = 0u;
  for (uint16_t s = 0u; s < n; ++s) {
    if (bitset_get(dfa->accept_bitset, s)) {
      bitset_set(dfa->live_bitset, s);
      queue[q_tail++] = s;
      accept_count += 1u;
    }
  }

  while (q_head < q_tail) {
    const uint16_t v = queue[q_head++];
    const uint32_t begin = rev_off[v];
    const uint32_t end = rev_off[v + 1u];
    for (uint32_t idx = begin; idx < end; ++idx) {
      const uint16_t u = rev_src[idx];
      if (!bitset_get(dfa->live_bitset, u)) {
        bitset_set(dfa->live_bitset, u);
        queue[q_tail++] = u;
      }
    }
  }

  free(in_deg);
  free(rev_off);
  free(rev_cur);
  free(rev_src);
  free(queue);

  dfa->live_state_count = (uint16_t)q_tail;
  dfa->accepting_state_count = accept_count;

  int pruned = 0;
  for (uint16_t s = 0u; s < n; ++s) {
    uint16_t *row = dfa->next_state + (size_t)s * ET_CD_ALPHABET_SIZE;
    if (!bitset_get(dfa->live_bitset, s)) {
      for (uint32_t b = 0u; b < ET_CD_ALPHABET_SIZE; ++b) {
        if (row[b] != ET_CD_DEAD_STATE) {
          row[b] = ET_CD_DEAD_STATE;
          pruned = 1;
        }
      }
    } else {
      for (uint32_t b = 0u; b < ET_CD_ALPHABET_SIZE; ++b) {
        const uint16_t t = row[b];
        if (t != ET_CD_DEAD_STATE && !bitset_get(dfa->live_bitset, t)) {
          row[b] = ET_CD_DEAD_STATE;
          pruned = 1;
        }
      }
    }
  }

  if (out_pruned_any != NULL)
    *out_pruned_any = pruned;

  if (!bitset_get(dfa->live_bitset, dfa->start_state))
    return ET_CD_ERR_EMPTY_LANGUAGE;

  return ET_CD_OK;
}

/* ---- Dynamic DFA Builder Helper ---- */

typedef struct dfa_builder {
  uint16_t num_states;
  uint16_t capacity;
  uint64_t accept_bitset[ET_CD_ACCEPT_WORDS_MAX];
  uint16_t *next_state;
} dfa_builder;

static et_cd_status builder_init(dfa_builder *b, uint16_t initial_cap) {
  memset(b, 0, sizeof(*b));
  if (initial_cap == 0u)
    initial_cap = 32u;
  if (initial_cap > ET_CD_MAX_STATES)
    initial_cap = ET_CD_MAX_STATES;
  const size_t count = (size_t)initial_cap * ET_CD_ALPHABET_SIZE;
  b->next_state = (uint16_t *)malloc(count * sizeof(uint16_t));
  if (b->next_state == NULL)
    return ET_CD_ERR_NOMEM;
  memset(b->next_state, 0xFF, count * sizeof(uint16_t));
  b->capacity = initial_cap;
  b->num_states = 0u;
  return ET_CD_OK;
}

static void builder_destroy(dfa_builder *b) {
  free(b->next_state);
  b->next_state = NULL;
  b->num_states = 0u;
  b->capacity = 0u;
}

static et_cd_status builder_new_state(dfa_builder *b, uint16_t *out_state) {
  if (b->num_states >= ET_CD_MAX_STATES)
    return ET_CD_ERR_LIMIT;
  if (b->num_states == b->capacity) {
    uint32_t new_cap = (uint32_t)b->capacity * 2u;
    if (new_cap > ET_CD_MAX_STATES)
      new_cap = ET_CD_MAX_STATES;
    const size_t old_count = (size_t)b->capacity * ET_CD_ALPHABET_SIZE;
    const size_t new_count = (size_t)new_cap * ET_CD_ALPHABET_SIZE;
    uint16_t *grown = (uint16_t *)malloc(new_count * sizeof(uint16_t));
    if (grown == NULL)
      return ET_CD_ERR_NOMEM;
    memcpy(grown, b->next_state, old_count * sizeof(uint16_t));
    memset(grown + old_count, 0xFF,
           (new_count - old_count) * sizeof(uint16_t));
    free(b->next_state);
    b->next_state = grown;
    b->capacity = (uint16_t)new_cap;
  }
  const uint16_t s = b->num_states++;
  *out_state = s;
  return ET_CD_OK;
}

static et_cd_status builder_set_edge(dfa_builder *b, uint16_t from, uint8_t byte,
                                     uint16_t to) {
  uint16_t *slot = &b->next_state[(size_t)from * ET_CD_ALPHABET_SIZE + byte];
  if (*slot != ET_CD_DEAD_STATE && *slot != to)
    return ET_CD_ERR_SCHEMA;
  *slot = to;
  return ET_CD_OK;
}

static et_cd_status builder_set_range(dfa_builder *b, uint16_t from, uint8_t lo,
                                      uint8_t hi, uint16_t to) {
  for (uint32_t c = lo; c <= hi; ++c) {
    const et_cd_status st = builder_set_edge(b, from, (uint8_t)c, to);
    if (st != ET_CD_OK)
      return st;
  }
  return ET_CD_OK;
}

static et_cd_status builder_emit_literal(dfa_builder *b, uint16_t from,
                                         const uint8_t *bytes, size_t len,
                                         uint16_t *out_end) {
  uint16_t cur = from;
  for (size_t i = 0u; i < len; ++i) {
    uint16_t nxt = 0u;
    et_cd_status st = builder_new_state(b, &nxt);
    if (st != ET_CD_OK)
      return st;
    st = builder_set_edge(b, cur, bytes[i], nxt);
    if (st != ET_CD_OK)
      return st;
    cur = nxt;
  }
  *out_end = cur;
  return ET_CD_OK;
}

static et_cd_status builder_finish(dfa_builder *b, uint16_t start_state,
                                   et_cd_dfa **out_dfa) {
  if (b->num_states == 0u || start_state >= b->num_states)
    return ET_CD_ERR_STATE;
  et_cd_dfa *dfa = (et_cd_dfa *)calloc(1u, sizeof(et_cd_dfa));
  if (dfa == NULL)
    return ET_CD_ERR_NOMEM;
  const size_t table_entries = (size_t)b->num_states * ET_CD_ALPHABET_SIZE;
  dfa->next_state = (uint16_t *)malloc(table_entries * sizeof(uint16_t));
  if (dfa->next_state == NULL) {
    free(dfa);
    return ET_CD_ERR_NOMEM;
  }
  dfa->num_states = b->num_states;
  dfa->start_state = start_state;
  memcpy(dfa->accept_bitset, b->accept_bitset, sizeof(dfa->accept_bitset));
  memcpy(dfa->next_state, b->next_state, table_entries * sizeof(uint16_t));

  const et_cd_status st = finalize_dfa_coaccessible(dfa, NULL);
  if (st != ET_CD_OK) {
    et_cd_dfa_destroy(dfa);
    return st;
  }
  *out_dfa = dfa;
  return ET_CD_OK;
}

/* ---- UTF-8 and JSON Escaping Validation Helpers ---- */

static int valid_utf8(const uint8_t *bytes, size_t len) {
  size_t i = 0u;
  while (i < len) {
    const uint8_t b0 = bytes[i];
    if (b0 <= 0x7Fu) {
      i += 1u;
    } else if (b0 >= 0xC2u && b0 <= 0xDFu) {
      if (i + 1u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      if ((b1 & 0xC0u) != 0x80u)
        return 0;
      i += 2u;
    } else if (b0 == 0xE0u) {
      if (i + 2u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      if (b1 < 0xA0u || b1 > 0xBFu || (b2 & 0xC0u) != 0x80u)
        return 0;
      i += 3u;
    } else if ((b0 >= 0xE1u && b0 <= 0xECu) || (b0 >= 0xEEu && b0 <= 0xEFu)) {
      if (i + 2u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u)
        return 0;
      i += 3u;
    } else if (b0 == 0xEDu) {
      if (i + 2u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      if (b1 < 0x80u || b1 > 0x9Fu || (b2 & 0xC0u) != 0x80u)
        return 0;
      i += 3u;
    } else if (b0 == 0xF0u) {
      if (i + 3u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      const uint8_t b3 = bytes[i + 3u];
      if (b1 < 0x90u || b1 > 0xBFu || (b2 & 0xC0u) != 0x80u ||
          (b3 & 0xC0u) != 0x80u)
        return 0;
      i += 4u;
    } else if (b0 >= 0xF1u && b0 <= 0xF3u) {
      if (i + 3u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      const uint8_t b3 = bytes[i + 3u];
      if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u ||
          (b3 & 0xC0u) != 0x80u)
        return 0;
      i += 4u;
    } else if (b0 == 0xF4u) {
      if (i + 3u >= len)
        return 0;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      const uint8_t b3 = bytes[i + 3u];
      if (b1 < 0x80u || b1 > 0x8Fu || (b2 & 0xC0u) != 0x80u ||
          (b3 & 0xC0u) != 0x80u)
        return 0;
      i += 4u;
    } else {
      return 0;
    }
  }
  return 1;
}

static size_t json_escape_bytes(const uint8_t *src, size_t len, uint8_t *dst) {
  static const char hex[17] = "0123456789abcdef";
  size_t out = 0u;
  for (size_t i = 0u; i < len; ++i) {
    const uint8_t c = src[i];
    if (c == '"') {
      dst[out++] = '\\';
      dst[out++] = '"';
    } else if (c == '\\') {
      dst[out++] = '\\';
      dst[out++] = '\\';
    } else if (c == '\b') {
      dst[out++] = '\\';
      dst[out++] = 'b';
    } else if (c == '\f') {
      dst[out++] = '\\';
      dst[out++] = 'f';
    } else if (c == '\n') {
      dst[out++] = '\\';
      dst[out++] = 'n';
    } else if (c == '\r') {
      dst[out++] = '\\';
      dst[out++] = 'r';
    } else if (c == '\t') {
      dst[out++] = '\\';
      dst[out++] = 't';
    } else if (c < 0x20u) {
      dst[out++] = '\\';
      dst[out++] = 'u';
      dst[out++] = '0';
      dst[out++] = '0';
      dst[out++] = (uint8_t)hex[(c >> 4u) & 0xFu];
      dst[out++] = (uint8_t)hex[c & 0xFu];
    } else {
      dst[out++] = c;
    }
  }
  return out;
}

/* ---- Public API: Raw DFA Construction ---- */

et_cd_status et_cd_dfa_build_raw(uint16_t num_states, uint16_t start_state,
                                 const uint16_t *accept_states,
                                 size_t num_accept_states,
                                 const uint16_t *transitions_row_major,
                                 et_cd_dfa **out_dfa) {
  if (out_dfa == NULL || transitions_row_major == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (num_accept_states > 0u && accept_states == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (num_states == 0u || num_states > ET_CD_MAX_STATES)
    return ET_CD_ERR_LIMIT;
  if (start_state >= num_states)
    return ET_CD_ERR_STATE;

  et_cd_dfa *dfa = (et_cd_dfa *)calloc(1u, sizeof(et_cd_dfa));
  if (dfa == NULL)
    return ET_CD_ERR_NOMEM;

  const size_t table_entries = (size_t)num_states * ET_CD_ALPHABET_SIZE;
  dfa->next_state = (uint16_t *)malloc(table_entries * sizeof(uint16_t));
  if (dfa->next_state == NULL) {
    free(dfa);
    return ET_CD_ERR_NOMEM;
  }

  dfa->num_states = num_states;
  dfa->start_state = start_state;

  for (size_t i = 0u; i < num_accept_states; ++i) {
    if (accept_states[i] >= num_states) {
      et_cd_dfa_destroy(dfa);
      return ET_CD_ERR_STATE;
    }
    bitset_set(dfa->accept_bitset, accept_states[i]);
  }

  for (size_t i = 0u; i < table_entries; ++i) {
    const uint16_t nxt = transitions_row_major[i];
    if (nxt != ET_CD_DEAD_STATE && nxt >= num_states) {
      et_cd_dfa_destroy(dfa);
      return ET_CD_ERR_STATE;
    }
    dfa->next_state[i] = nxt;
  }

  const et_cd_status st = finalize_dfa_coaccessible(dfa, NULL);
  if (st != ET_CD_OK) {
    et_cd_dfa_destroy(dfa);
    return st;
  }

  *out_dfa = dfa;
  return ET_CD_OK;
}

void et_cd_dfa_destroy(et_cd_dfa *dfa) {
  if (dfa == NULL)
    return;
  free(dfa->next_state);
  free(dfa);
}

/* ---- Public API: Built-in Compiler 1 — Literal Choices ---- */

et_cd_status et_cd_dfa_compile_literal_choices(const uint8_t *const *choices,
                                               const size_t *choice_lengths,
                                               size_t num_choices,
                                               et_cd_dfa **out_dfa) {
  if (out_dfa == NULL || choices == NULL || choice_lengths == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (num_choices == 0u)
    return ET_CD_ERR_SCHEMA;
  if (num_choices > ET_CD_MAX_CHOICES)
    return ET_CD_ERR_LIMIT;

  for (size_t i = 0u; i < num_choices; ++i) {
    if (choice_lengths[i] == 0u)
      return ET_CD_ERR_SCHEMA;
    if (choice_lengths[i] > ET_CD_MAX_CHOICE_BYTES)
      return ET_CD_ERR_LIMIT;
    if (choices[i] == NULL)
      return ET_CD_ERR_ARGUMENT;
    for (size_t j = 0u; j < i; ++j) {
      if (choice_lengths[i] == choice_lengths[j] &&
          memcmp(choices[i], choices[j], choice_lengths[i]) == 0)
        return ET_CD_ERR_SCHEMA;
    }
  }

  dfa_builder b;
  et_cd_status st = builder_init(&b, 64u);
  if (st != ET_CD_OK)
    return st;

  uint16_t start = 0u;
  st = builder_new_state(&b, &start);
  if (st != ET_CD_OK) {
    builder_destroy(&b);
    return st;
  }

  for (size_t i = 0u; i < num_choices; ++i) {
    uint16_t cur = start;
    const uint8_t *bytes = choices[i];
    const size_t len = choice_lengths[i];
    for (size_t k = 0u; k < len; ++k) {
      const uint8_t ch = bytes[k];
      uint16_t nxt = b.next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + ch];
      if (nxt == ET_CD_DEAD_STATE) {
        st = builder_new_state(&b, &nxt);
        if (st != ET_CD_OK) {
          builder_destroy(&b);
          return st;
        }
        b.next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + ch] = nxt;
      }
      cur = nxt;
    }
    bitset_set(b.accept_bitset, cur);
  }

  st = builder_finish(&b, start, out_dfa);
  builder_destroy(&b);
  return st;
}

/* ---- Public API: Built-in Compiler 2 — Flat JSON Object Schema ---- */

static et_cd_status emit_json_string_field(dfa_builder *b, uint16_t entry,
                                           uint8_t follow_byte,
                                           uint16_t *out_after_follow) {
  uint16_t s_body = 0u, s_esc = 0u, s_u1 = 0u, s_u2 = 0u, s_u3 = 0u, s_u4 = 0u;
  uint16_t u_c1 = 0u, u_c2 = 0u, u_c3 = 0u, u_e0 = 0u, u_ed = 0u, u_f0 = 0u,
           u_f4 = 0u;
  uint16_t s_end = 0u, s_after = 0u;
  et_cd_status st;

  if ((st = builder_new_state(b, &s_body)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_esc)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_u1)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_u2)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_u3)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_u4)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_c1)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_c2)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_c3)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_e0)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_ed)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_f0)) != ET_CD_OK ||
      (st = builder_new_state(b, &u_f4)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_end)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_after)) != ET_CD_OK)
    return st;

  /* Opening quote. */
  if ((st = builder_set_edge(b, entry, '"', s_body)) != ET_CD_OK)
    return st;

  /* Unescaped ASCII [0x20..0x7F] except '"' (0x22) and '\\' (0x5C). */
  if ((st = builder_set_range(b, s_body, 0x20u, 0x21u, s_body)) != ET_CD_OK ||
      (st = builder_set_range(b, s_body, 0x23u, 0x5Bu, s_body)) != ET_CD_OK ||
      (st = builder_set_range(b, s_body, 0x5Du, 0x7Fu, s_body)) != ET_CD_OK)
    return st;

  /* Closing quote and escape introducer. */
  if ((st = builder_set_edge(b, s_body, '"', s_end)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_body, '\\', s_esc)) != ET_CD_OK)
    return st;

  /* Valid RFC 3629 UTF-8 multi-byte sequences inside string body. */
  if ((st = builder_set_range(b, s_body, 0xC2u, 0xDFu, u_c1)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_body, 0xE0u, u_e0)) != ET_CD_OK ||
      (st = builder_set_range(b, s_body, 0xE1u, 0xECu, u_c2)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_body, 0xEDu, u_ed)) != ET_CD_OK ||
      (st = builder_set_range(b, s_body, 0xEEu, 0xEFu, u_c2)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_body, 0xF0u, u_f0)) != ET_CD_OK ||
      (st = builder_set_range(b, s_body, 0xF1u, 0xF3u, u_c3)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_body, 0xF4u, u_f4)) != ET_CD_OK)
    return st;

  if ((st = builder_set_range(b, u_c1, 0x80u, 0xBFu, s_body)) != ET_CD_OK ||
      (st = builder_set_range(b, u_e0, 0xA0u, 0xBFu, u_c1)) != ET_CD_OK ||
      (st = builder_set_range(b, u_c2, 0x80u, 0xBFu, u_c1)) != ET_CD_OK ||
      (st = builder_set_range(b, u_ed, 0x80u, 0x9Fu, u_c1)) != ET_CD_OK ||
      (st = builder_set_range(b, u_f0, 0x90u, 0xBFu, u_c2)) != ET_CD_OK ||
      (st = builder_set_range(b, u_c3, 0x80u, 0xBFu, u_c2)) != ET_CD_OK ||
      (st = builder_set_range(b, u_f4, 0x80u, 0x8Fu, u_c2)) != ET_CD_OK)
    return st;

  /* JSON escape sequences: \" \\ \/ \b \f \n \r \t and \uXXXX. */
  static const uint8_t simple_esc[8] = {'"', '\\', '/', 'b',
                                        'f', 'n',  'r', 't'};
  for (size_t i = 0u; i < 8u; ++i) {
    if ((st = builder_set_edge(b, s_esc, simple_esc[i], s_body)) != ET_CD_OK)
      return st;
  }
  if ((st = builder_set_edge(b, s_esc, 'u', s_u1)) != ET_CD_OK)
    return st;

  const uint16_t u_from[4] = {s_u1, s_u2, s_u3, s_u4};
  const uint16_t u_to[4] = {s_u2, s_u3, s_u4, s_body};
  for (size_t i = 0u; i < 4u; ++i) {
    if ((st = builder_set_range(b, u_from[i], '0', '9', u_to[i])) != ET_CD_OK ||
        (st = builder_set_range(b, u_from[i], 'A', 'F', u_to[i])) != ET_CD_OK ||
        (st = builder_set_range(b, u_from[i], 'a', 'f', u_to[i])) != ET_CD_OK)
      return st;
  }

  /* Follow delimiter (',' or '}'). */
  if ((st = builder_set_edge(b, s_end, follow_byte, s_after)) != ET_CD_OK)
    return st;

  *out_after_follow = s_after;
  return ET_CD_OK;
}

static et_cd_status emit_json_integer_field(dfa_builder *b, uint16_t entry,
                                            uint8_t follow_byte,
                                            uint16_t *out_after_follow) {
  uint16_t s_neg = 0u, s_zero = 0u, s_digits = 0u, s_after = 0u;
  et_cd_status st;

  if ((st = builder_new_state(b, &s_neg)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_zero)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_digits)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_after)) != ET_CD_OK)
    return st;

  /* Grammar: -?(0|[1-9][0-9]*) followed immediately by follow_byte. */
  if ((st = builder_set_edge(b, entry, '-', s_neg)) != ET_CD_OK ||
      (st = builder_set_edge(b, entry, '0', s_zero)) != ET_CD_OK ||
      (st = builder_set_range(b, entry, '1', '9', s_digits)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_neg, '0', s_zero)) != ET_CD_OK ||
      (st = builder_set_range(b, s_neg, '1', '9', s_digits)) != ET_CD_OK ||
      (st = builder_set_range(b, s_digits, '0', '9', s_digits)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_zero, follow_byte, s_after)) != ET_CD_OK ||
      (st = builder_set_edge(b, s_digits, follow_byte, s_after)) != ET_CD_OK)
    return st;

  *out_after_follow = s_after;
  return ET_CD_OK;
}

static et_cd_status emit_json_boolean_field(dfa_builder *b, uint16_t entry,
                                            uint8_t follow_byte,
                                            uint16_t *out_after_follow) {
  uint16_t s_end = 0u, s_after = 0u;
  et_cd_status st;
  if ((st = builder_new_state(b, &s_end)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_after)) != ET_CD_OK)
    return st;

  /* true */
  uint16_t t_pre = 0u;
  if ((st = builder_emit_literal(b, entry, (const uint8_t *)"tru", 3u,
                                 &t_pre)) != ET_CD_OK ||
      (st = builder_set_edge(b, t_pre, 'e', s_end)) != ET_CD_OK)
    return st;

  /* false */
  uint16_t f_pre = 0u;
  if ((st = builder_emit_literal(b, entry, (const uint8_t *)"fals", 4u,
                                 &f_pre)) != ET_CD_OK ||
      (st = builder_set_edge(b, f_pre, 'e', s_end)) != ET_CD_OK)
    return st;

  if ((st = builder_set_edge(b, s_end, follow_byte, s_after)) != ET_CD_OK)
    return st;

  *out_after_follow = s_after;
  return ET_CD_OK;
}

static et_cd_status emit_json_enum_field(dfa_builder *b, uint16_t entry,
                                         const char *const *values,
                                         size_t num_values, uint8_t follow_byte,
                                         uint16_t *out_after_follow) {
  uint16_t s_root = 0u, s_end = 0u, s_after = 0u;
  et_cd_status st;
  if ((st = builder_new_state(b, &s_root)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_end)) != ET_CD_OK ||
      (st = builder_new_state(b, &s_after)) != ET_CD_OK)
    return st;

  if ((st = builder_set_edge(b, entry, '"', s_root)) != ET_CD_OK)
    return st;

  /* Escape buffer: worst-case 6x expansion for \u00XX control bytes. */
  uint8_t escaped[ET_CD_MAX_CHOICE_BYTES * 6u];
  for (size_t i = 0u; i < num_values; ++i) {
    const uint8_t *raw = (const uint8_t *)values[i];
    const size_t raw_len = strlen(values[i]);
    const size_t esc_len = json_escape_bytes(raw, raw_len, escaped);
    uint16_t cur = s_root;
    for (size_t k = 0u; k < esc_len; ++k) {
      const uint8_t ch = escaped[k];
      uint16_t nxt = b->next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + ch];
      if (nxt == ET_CD_DEAD_STATE) {
        st = builder_new_state(b, &nxt);
        if (st != ET_CD_OK)
          return st;
        b->next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + ch] = nxt;
      }
      cur = nxt;
    }
    /* Raw '"' never occurs inside escaped bytes, so closing quote is conflict-free. */
    st = builder_set_edge(b, cur, '"', s_end);
    if (st != ET_CD_OK)
      return st;
  }

  if ((st = builder_set_edge(b, s_end, follow_byte, s_after)) != ET_CD_OK)
    return st;

  *out_after_follow = s_after;
  return ET_CD_OK;
}

et_cd_status
et_cd_dfa_compile_flat_json_schema(const et_cd_json_field_spec *fields,
                                   size_t num_fields, et_cd_dfa **out_dfa) {
  if (out_dfa == NULL || fields == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (num_fields == 0u)
    return ET_CD_ERR_SCHEMA;
  if (num_fields > ET_CD_MAX_SCHEMA_FIELDS)
    return ET_CD_ERR_LIMIT;

  /* Validate all field specifications before allocating automaton states. */
  for (size_t i = 0u; i < num_fields; ++i) {
    const et_cd_json_field_spec *f = &fields[i];
    if (f->key == NULL)
      return ET_CD_ERR_SCHEMA;
    const size_t klen = strlen(f->key);
    if (klen == 0u)
      return ET_CD_ERR_SCHEMA;
    if (klen > ET_CD_MAX_KEY_BYTES)
      return ET_CD_ERR_LIMIT;
    for (size_t c = 0u; c < klen; ++c) {
      const uint8_t b = (uint8_t)f->key[c];
      if (b < 0x20u || b > 0x7Eu || b == '"' || b == '\\')
        return ET_CD_ERR_SCHEMA;
    }
    for (size_t j = 0u; j < i; ++j) {
      if (strcmp(f->key, fields[j].key) == 0)
        return ET_CD_ERR_SCHEMA;
    }
    if (f->type == ET_CD_JSON_ENUM) {
      if (f->enum_values == NULL || f->num_enum_values == 0u)
        return ET_CD_ERR_SCHEMA;
      if (f->num_enum_values > ET_CD_MAX_CHOICES)
        return ET_CD_ERR_LIMIT;
      for (size_t e = 0u; e < f->num_enum_values; ++e) {
        if (f->enum_values[e] == NULL)
          return ET_CD_ERR_SCHEMA;
        const size_t vlen = strlen(f->enum_values[e]);
        if (vlen > ET_CD_MAX_CHOICE_BYTES)
          return ET_CD_ERR_LIMIT;
        if (!valid_utf8((const uint8_t *)f->enum_values[e], vlen))
          return ET_CD_ERR_SCHEMA;
        for (size_t prev = 0u; prev < e; ++prev) {
          if (strcmp(f->enum_values[e], f->enum_values[prev]) == 0)
            return ET_CD_ERR_SCHEMA;
        }
      }
    } else if (f->type == ET_CD_JSON_STRING || f->type == ET_CD_JSON_INTEGER ||
               f->type == ET_CD_JSON_BOOLEAN) {
      if (f->enum_values != NULL || f->num_enum_values != 0u)
        return ET_CD_ERR_SCHEMA;
    } else {
      return ET_CD_ERR_SCHEMA;
    }
  }

  dfa_builder b;
  et_cd_status st = builder_init(&b, 128u);
  if (st != ET_CD_OK)
    return st;

  uint16_t start = 0u;
  uint16_t cur = 0u;
  if ((st = builder_new_state(&b, &start)) != ET_CD_OK ||
      (st = builder_emit_literal(&b, start, (const uint8_t *)"{", 1u, &cur)) !=
          ET_CD_OK) {
    builder_destroy(&b);
    return st;
  }

  for (size_t i = 0u; i < num_fields; ++i) {
    const et_cd_json_field_spec *f = &fields[i];
    const size_t klen = strlen(f->key);
    const uint8_t follow = (i + 1u < num_fields) ? (uint8_t)',' : (uint8_t)'}';

    /* Emit "<key>": */
    uint16_t after_key = 0u;
    if ((st = builder_emit_literal(&b, cur, (const uint8_t *)"\"", 1u, &cur)) !=
            ET_CD_OK ||
        (st = builder_emit_literal(&b, cur, (const uint8_t *)f->key, klen,
                                   &cur)) != ET_CD_OK ||
        (st = builder_emit_literal(&b, cur, (const uint8_t *)"\":", 2u,
                                   &after_key)) != ET_CD_OK) {
      builder_destroy(&b);
      return st;
    }

    uint16_t after_follow = 0u;
    switch (f->type) {
    case ET_CD_JSON_STRING:
      st = emit_json_string_field(&b, after_key, follow, &after_follow);
      break;
    case ET_CD_JSON_INTEGER:
      st = emit_json_integer_field(&b, after_key, follow, &after_follow);
      break;
    case ET_CD_JSON_BOOLEAN:
      st = emit_json_boolean_field(&b, after_key, follow, &after_follow);
      break;
    case ET_CD_JSON_ENUM:
      st = emit_json_enum_field(&b, after_key, f->enum_values,
                                f->num_enum_values, follow, &after_follow);
      break;
    default:
      st = ET_CD_ERR_SCHEMA;
      break;
    }
    if (st != ET_CD_OK) {
      builder_destroy(&b);
      return st;
    }
    cur = after_follow;
  }

  /* cur is the state immediately after the closing '}'. */
  bitset_set(b.accept_bitset, cur);

  st = builder_finish(&b, start, out_dfa);
  builder_destroy(&b);
  return st;
}

/* ---- Public API: Zero-Allocation DFA Inspection ---- */

et_cd_status et_cd_dfa_info_get(const et_cd_dfa *dfa,
                                et_cd_dfa_info *out_info) {
  if (dfa == NULL || out_info == NULL)
    return ET_CD_ERR_ARGUMENT;
  out_info->num_states = dfa->num_states;
  out_info->start_state = dfa->start_state;
  out_info->live_state_count = dfa->live_state_count;
  out_info->accepting_state_count = dfa->accepting_state_count;
  return ET_CD_OK;
}

uint16_t et_cd_dfa_next_state(const et_cd_dfa *dfa, uint16_t state,
                              uint8_t byte) {
  if (dfa == NULL || state >= dfa->num_states)
    return ET_CD_DEAD_STATE;
  return dfa->next_state[(size_t)state * ET_CD_ALPHABET_SIZE + byte];
}

uint32_t et_cd_dfa_is_accepting(const et_cd_dfa *dfa, uint16_t state) {
  if (dfa == NULL || state >= dfa->num_states)
    return 0u;
  return (uint32_t)bitset_get(dfa->accept_bitset, state);
}

uint32_t et_cd_dfa_is_live(const et_cd_dfa *dfa, uint16_t state) {
  if (dfa == NULL || state >= dfa->num_states)
    return 0u;
  return (uint32_t)bitset_get(dfa->live_bitset, state);
}

et_cd_status et_cd_dfa_accepts_bytes(const et_cd_dfa *dfa, const uint8_t *bytes,
                                     size_t len, uint32_t *out_accepted) {
  if (dfa == NULL || out_accepted == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (len > 0u && bytes == NULL)
    return ET_CD_ERR_ARGUMENT;
  uint16_t cur = dfa->start_state;
  for (size_t i = 0u; i < len; ++i) {
    cur = dfa->next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + bytes[i]];
    if (cur == ET_CD_DEAD_STATE) {
      *out_accepted = 0u;
      return ET_CD_OK;
    }
  }
  *out_accepted = (uint32_t)bitset_get(dfa->accept_bitset, cur);
  return ET_CD_OK;
}

/* ---- Public API: Checksummed Binary Artifact (ETCD1000 v1.0) ---- */

static size_t artifact_payload_bytes(uint16_t num_states) {
  const size_t accept_words = ((size_t)num_states + 63u) / 64u;
  return accept_words * 8u + (size_t)num_states * ET_CD_ALPHABET_SIZE * 2u;
}

static size_t artifact_total_bytes(uint16_t num_states) {
  return ET_CD_ARTIFACT_HEADER_BYTES + artifact_payload_bytes(num_states) +
         ET_CD_SHA256_BYTES;
}

et_cd_status et_cd_dfa_artifact_size(const et_cd_dfa *dfa, size_t *out_size) {
  if (dfa == NULL || out_size == NULL)
    return ET_CD_ERR_ARGUMENT;
  *out_size = artifact_total_bytes(dfa->num_states);
  return ET_CD_OK;
}

et_cd_status et_cd_dfa_serialize(const et_cd_dfa *dfa, uint8_t *out_bytes,
                                 size_t capacity, size_t *out_size) {
  if (dfa == NULL || out_bytes == NULL || out_size == NULL)
    return ET_CD_ERR_ARGUMENT;
  const size_t total = artifact_total_bytes(dfa->num_states);
  if (capacity < total)
    return ET_CD_ERR_LIMIT;

  const uint32_t accept_words = ((uint32_t)dfa->num_states + 63u) / 64u;
  const uint64_t payload_bytes = (uint64_t)artifact_payload_bytes(dfa->num_states);

  memcpy(out_bytes, ET_CD_ARTIFACT_MAGIC, ET_CD_ARTIFACT_MAGIC_BYTES);
  encode_u16_le(out_bytes + 8u, ET_CD_ARTIFACT_VERSION_MAJOR);
  encode_u16_le(out_bytes + 10u, ET_CD_ARTIFACT_VERSION_MINOR);
  encode_u16_le(out_bytes + 12u, dfa->num_states);
  encode_u16_le(out_bytes + 14u, dfa->start_state);
  encode_u16_le(out_bytes + 16u, dfa->live_state_count);
  encode_u16_le(out_bytes + 18u, 0u);
  encode_u32_le(out_bytes + 20u, accept_words);
  encode_u64_le(out_bytes + 24u, payload_bytes);

  size_t off = ET_CD_ARTIFACT_HEADER_BYTES;
  for (uint32_t w = 0u; w < accept_words; ++w) {
    encode_u64_le(out_bytes + off, dfa->accept_bitset[w]);
    off += 8u;
  }
  const size_t table_entries = (size_t)dfa->num_states * ET_CD_ALPHABET_SIZE;
  for (size_t i = 0u; i < table_entries; ++i) {
    encode_u16_le(out_bytes + off, dfa->next_state[i]);
    off += 2u;
  }

  uint8_t digest[ET_CD_SHA256_BYTES];
  sha256_bytes(out_bytes, off, digest);
  memcpy(out_bytes + off, digest, ET_CD_SHA256_BYTES);
  *out_size = total;
  return ET_CD_OK;
}

et_cd_status et_cd_dfa_parse(const uint8_t *bytes, size_t size,
                             et_cd_dfa **out_dfa,
                             uint8_t out_sha256[ET_CD_SHA256_BYTES]) {
  if (bytes == NULL || out_dfa == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (size < ET_CD_ARTIFACT_MAGIC_BYTES)
    return ET_CD_ERR_TRUNCATED;
  if (memcmp(bytes, ET_CD_ARTIFACT_MAGIC, ET_CD_ARTIFACT_MAGIC_BYTES) != 0)
    return ET_CD_ERR_FORMAT;
  if (size < ET_CD_ARTIFACT_HEADER_BYTES + ET_CD_SHA256_BYTES)
    return ET_CD_ERR_TRUNCATED;

  const uint16_t ver_major = decode_u16_le(bytes + 8u);
  const uint16_t ver_minor = decode_u16_le(bytes + 10u);
  if (ver_major != ET_CD_ARTIFACT_VERSION_MAJOR ||
      ver_minor != ET_CD_ARTIFACT_VERSION_MINOR)
    return ET_CD_ERR_VERSION;

  const uint16_t num_states = decode_u16_le(bytes + 12u);
  const uint16_t start_state = decode_u16_le(bytes + 14u);
  const uint16_t file_live_count = decode_u16_le(bytes + 16u);
  const uint16_t flags = decode_u16_le(bytes + 18u);
  const uint32_t accept_words = decode_u32_le(bytes + 20u);
  const uint64_t payload_bytes = decode_u64_le(bytes + 24u);

  if (flags != 0u || num_states == 0u || num_states > ET_CD_MAX_STATES ||
      start_state >= num_states || file_live_count == 0u ||
      file_live_count > num_states)
    return ET_CD_ERR_FORMAT;

  const uint32_t expected_words = ((uint32_t)num_states + 63u) / 64u;
  const uint64_t expected_payload = (uint64_t)artifact_payload_bytes(num_states);
  if (accept_words != expected_words || payload_bytes != expected_payload)
    return ET_CD_ERR_FORMAT;

  const size_t expected_total = artifact_total_bytes(num_states);
  if (size < expected_total)
    return ET_CD_ERR_TRUNCATED;
  if (size > expected_total)
    return ET_CD_ERR_FORMAT;

  const size_t signed_len = size - ET_CD_SHA256_BYTES;
  uint8_t computed[ET_CD_SHA256_BYTES];
  sha256_bytes(bytes, signed_len, computed);
  if (memcmp(computed, bytes + signed_len, ET_CD_SHA256_BYTES) != 0)
    return ET_CD_ERR_CHECKSUM;

  et_cd_dfa *dfa = (et_cd_dfa *)calloc(1u, sizeof(et_cd_dfa));
  if (dfa == NULL)
    return ET_CD_ERR_NOMEM;

  const size_t table_entries = (size_t)num_states * ET_CD_ALPHABET_SIZE;
  dfa->next_state = (uint16_t *)malloc(table_entries * sizeof(uint16_t));
  if (dfa->next_state == NULL) {
    free(dfa);
    return ET_CD_ERR_NOMEM;
  }

  dfa->num_states = num_states;
  dfa->start_state = start_state;

  size_t off = ET_CD_ARTIFACT_HEADER_BYTES;
  for (uint32_t w = 0u; w < accept_words; ++w) {
    dfa->accept_bitset[w] = decode_u64_le(bytes + off);
    off += 8u;
  }
  /* Reject non-zero padding bits past num_states in the last accept word. */
  const uint32_t rem = (uint32_t)num_states & 63u;
  if (rem != 0u) {
    const uint64_t pad_mask = ~((UINT64_C(1) << rem) - UINT64_C(1));
    if ((dfa->accept_bitset[accept_words - 1u] & pad_mask) != 0u) {
      et_cd_dfa_destroy(dfa);
      return ET_CD_ERR_FORMAT;
    }
  }

  for (size_t i = 0u; i < table_entries; ++i) {
    const uint16_t nxt = decode_u16_le(bytes + off);
    off += 2u;
    if (nxt != ET_CD_DEAD_STATE && nxt >= num_states) {
      et_cd_dfa_destroy(dfa);
      return ET_CD_ERR_FORMAT;
    }
    dfa->next_state[i] = nxt;
  }

  int pruned = 0;
  const et_cd_status st = finalize_dfa_coaccessible(dfa, &pruned);
  if (st != ET_CD_OK) {
    et_cd_dfa_destroy(dfa);
    return st;
  }
  /* Artifact must match canonical co-accessible pruned form. */
  if (pruned || dfa->live_state_count != file_live_count) {
    et_cd_dfa_destroy(dfa);
    return ET_CD_ERR_FORMAT;
  }

  if (out_sha256 != NULL)
    memcpy(out_sha256, computed, ET_CD_SHA256_BYTES);
  *out_dfa = dfa;
  return ET_CD_OK;
}

static int write_all(int fd, const uint8_t *bytes, size_t size) {
  while (size != 0u) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return 0;
    bytes += (size_t)written;
    size -= (size_t)written;
  }
  return 1;
}

static int sync_fd(int fd) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc < 0 && errno == EINTR);
  return rc == 0;
}

et_cd_status et_cd_dfa_save(const char *path, const et_cd_dfa *dfa) {
  if (path == NULL || path[0] == '\0' || dfa == NULL)
    return ET_CD_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_CD_ERR_ARGUMENT;

  const size_t total = artifact_total_bytes(dfa->num_states);
  uint8_t *buffer = (uint8_t *)malloc(total);
  if (buffer == NULL)
    return ET_CD_ERR_NOMEM;

  size_t written_size = 0u;
  et_cd_status st = et_cd_dfa_serialize(dfa, buffer, total, &written_size);
  if (st != ET_CD_OK) {
    free(buffer);
    return st;
  }

  static const char suffix[] = ".etcd-tmp.XXXXXX";
  const size_t prefix = slash == NULL ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix)) {
    free(buffer);
    return ET_CD_ERR_OVERFLOW;
  }

  char *temp_path = (char *)malloc(prefix + sizeof(suffix));
  if (temp_path == NULL) {
    free(buffer);
    return ET_CD_ERR_NOMEM;
  }
  memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    free(buffer);
    return ET_CD_ERR_IO;
  }

  int ok = write_all(fd, buffer, written_size) && sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    st = ET_CD_ERR_IO;
  }

  free(temp_path);
  free(buffer);
  return st;
}

et_cd_status et_cd_dfa_load(const char *path, et_cd_dfa **out_dfa,
                            uint8_t out_sha256[ET_CD_SHA256_BYTES]) {
  if (path == NULL || path[0] == '\0' || out_dfa == NULL)
    return ET_CD_ERR_ARGUMENT;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_CD_ERR_SYMLINK : ET_CD_ERR_IO;

  const size_t max_bytes = artifact_total_bytes(ET_CD_MAX_STATES);
  struct stat meta;
  et_cd_status st = ET_CD_OK;
  uint8_t *buffer = NULL;
  size_t used = 0u;

  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    st = ET_CD_ERR_IO;
  } else if (meta.st_size < 0 || (uintmax_t)meta.st_size > max_bytes) {
    st = ET_CD_ERR_FORMAT;
  } else {
    const size_t expected_len = (size_t)meta.st_size;
    buffer = (uint8_t *)malloc(expected_len + 1u);
    if (buffer == NULL) {
      st = ET_CD_ERR_NOMEM;
    } else {
      while (used < expected_len + 1u) {
        const ssize_t nread =
            read(fd, buffer + used, (expected_len + 1u) - used);
        if (nread == 0)
          break;
        if (nread < 0) {
          if (errno == EINTR)
            continue;
          st = ET_CD_ERR_IO;
          break;
        }
        used += (size_t)nread;
      }
      if (st == ET_CD_OK) {
        if (used > expected_len)
          st = ET_CD_ERR_FORMAT;
        else if (used < expected_len)
          st = ET_CD_ERR_TRUNCATED;
      }
    }
  }

  if (close(fd) != 0 && st == ET_CD_OK)
    st = ET_CD_ERR_IO;

  if (st == ET_CD_OK)
    st = et_cd_dfa_parse(buffer, used, out_dfa, out_sha256);

  free(buffer);
  return st;
}

/* ---- Public API: Vocabulary Index Precomputation ---- */

et_cd_status et_cd_vocab_index_build(const et_cd_dfa *dfa,
                                     const uint8_t *const *token_bytes,
                                     const size_t *token_lengths,
                                     uint32_t vocab_size, uint32_t eos_token_id,
                                     et_cd_vocab_index **out_index) {
  if (dfa == NULL || token_bytes == NULL || token_lengths == NULL ||
      out_index == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (vocab_size == 0u || vocab_size > ET_CD_MAX_VOCAB_SIZE)
    return ET_CD_ERR_LIMIT;
  if (eos_token_id >= vocab_size)
    return ET_CD_ERR_ARGUMENT;

  for (uint32_t t = 0u; t < vocab_size; ++t) {
    if (token_lengths[t] > ET_CD_MAX_TOKEN_BYTES)
      return ET_CD_ERR_LIMIT;
    if (token_lengths[t] > 0u && token_bytes[t] == NULL)
      return ET_CD_ERR_ARGUMENT;
  }

  const uint16_t num_states = dfa->num_states;
  const uint32_t mask_words = (vocab_size + 63u) / 64u;

  if ((size_t)num_states > SIZE_MAX / (size_t)vocab_size ||
      (size_t)num_states * (size_t)vocab_size > SIZE_MAX / sizeof(uint16_t))
    return ET_CD_ERR_OVERFLOW;

  et_cd_vocab_index *idx =
      (et_cd_vocab_index *)calloc(1u, sizeof(et_cd_vocab_index));
  if (idx == NULL)
    return ET_CD_ERR_NOMEM;

  const size_t mask_elems = (size_t)num_states * (size_t)mask_words;
  const size_t dest_elems = (size_t)num_states * (size_t)vocab_size;

  idx->allowed_mask = (uint64_t *)calloc(mask_elems, sizeof(uint64_t));
  idx->allowed_count = (uint32_t *)calloc((size_t)num_states, sizeof(uint32_t));
  idx->dest_state = (uint16_t *)malloc(dest_elems * sizeof(uint16_t));
  if (idx->allowed_mask == NULL || idx->allowed_count == NULL ||
      idx->dest_state == NULL) {
    et_cd_vocab_index_destroy(idx);
    return ET_CD_ERR_NOMEM;
  }
  memset(idx->dest_state, 0xFF, dest_elems * sizeof(uint16_t));

  idx->num_states = num_states;
  idx->start_state = dfa->start_state;
  idx->live_state_count = dfa->live_state_count;
  idx->vocab_size = vocab_size;
  idx->eos_token_id = eos_token_id;
  idx->mask_words = mask_words;
  memcpy(idx->accept_bitset, dfa->accept_bitset, sizeof(idx->accept_bitset));
  memcpy(idx->live_bitset, dfa->live_bitset, sizeof(idx->live_bitset));

  for (uint16_t s = 0u; s < num_states; ++s) {
    if (!bitset_get(dfa->live_bitset, s))
      continue;
    uint64_t *mask_row = idx->allowed_mask + (size_t)s * (size_t)mask_words;
    uint16_t *dest_row = idx->dest_state + (size_t)s * (size_t)vocab_size;
    uint32_t count = 0u;
    const int s_accept = bitset_get(dfa->accept_bitset, s);

    for (uint32_t t = 0u; t < vocab_size; ++t) {
      if (t == eos_token_id) {
        if (s_accept) {
          bitset_set(mask_row, t);
          dest_row[t] = ET_CD_DEAD_STATE;
          count += 1u;
        }
        continue;
      }

      const size_t len = token_lengths[t];
      if (len == 0u)
        continue;

      const uint8_t *bytes = token_bytes[t];
      uint16_t cur = s;
      for (size_t k = 0u; k < len; ++k) {
        cur = dfa->next_state[(size_t)cur * ET_CD_ALPHABET_SIZE + bytes[k]];
        if (cur == ET_CD_DEAD_STATE)
          break;
      }
      if (cur != ET_CD_DEAD_STATE && bitset_get(dfa->live_bitset, cur)) {
        bitset_set(mask_row, t);
        dest_row[t] = cur;
        count += 1u;
      }
    }
    idx->allowed_count[s] = count;
  }

  *out_index = idx;
  return ET_CD_OK;
}

void et_cd_vocab_index_destroy(et_cd_vocab_index *index) {
  if (index == NULL)
    return;
  free(index->allowed_mask);
  free(index->allowed_count);
  free(index->dest_state);
  free(index);
}

et_cd_status et_cd_vocab_index_info_get(const et_cd_vocab_index *index,
                                        et_cd_vocab_index_info *out_info) {
  if (index == NULL || out_info == NULL)
    return ET_CD_ERR_ARGUMENT;
  out_info->num_states = index->num_states;
  out_info->start_state = index->start_state;
  out_info->live_state_count = index->live_state_count;
  out_info->reserved = 0u;
  out_info->vocab_size = index->vocab_size;
  out_info->eos_token_id = index->eos_token_id;
  out_info->mask_words = index->mask_words;
  return ET_CD_OK;
}

et_cd_status et_cd_vocab_index_allowed_count(const et_cd_vocab_index *index,
                                             uint16_t state,
                                             uint32_t *out_count) {
  if (index == NULL || out_count == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (state >= index->num_states || !bitset_get(index->live_bitset, state))
    return ET_CD_ERR_STATE;
  *out_count = index->allowed_count[state];
  return ET_CD_OK;
}

uint32_t et_cd_vocab_index_is_token_allowed(const et_cd_vocab_index *index,
                                            uint16_t state, uint32_t token_id) {
  if (index == NULL || state >= index->num_states ||
      !bitset_get(index->live_bitset, state) || token_id >= index->vocab_size)
    return 0u;
  const uint64_t *mask_row =
      index->allowed_mask + (size_t)state * (size_t)index->mask_words;
  return (uint32_t)bitset_get(mask_row, token_id);
}

et_cd_status et_cd_vocab_index_copy_mask_words(const et_cd_vocab_index *index,
                                               uint16_t state,
                                               uint64_t *out_words,
                                               size_t max_words) {
  if (index == NULL || out_words == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (state >= index->num_states || !bitset_get(index->live_bitset, state))
    return ET_CD_ERR_STATE;
  if (max_words < index->mask_words)
    return ET_CD_ERR_LIMIT;
  const uint64_t *mask_row =
      index->allowed_mask + (size_t)state * (size_t)index->mask_words;
  memcpy(out_words, mask_row, (size_t)index->mask_words * sizeof(uint64_t));
  return ET_CD_OK;
}

/* ---- Public API: Hot-Path Runtime Operations (ZERO Heap Allocations) ---- */

et_cd_status et_cd_apply_logit_mask_v1(const et_cd_vocab_index *index,
                                       uint16_t state, const float *logits,
                                       float *masked_logits) {
  if (!fp_environment_supported())
    return ET_CD_ERR_FP_ENV;
  if (index == NULL || logits == NULL || masked_logits == NULL)
    return ET_CD_ERR_ARGUMENT;

  const uint32_t v = index->vocab_size;
  const size_t bytes = (size_t)v * sizeof(float);
  if (!aligned_span(logits, _Alignof(float), bytes) ||
      !aligned_span(masked_logits, _Alignof(float), bytes))
    return ET_CD_ERR_ARGUMENT;
  if (spans_partially_overlap(logits, masked_logits, bytes))
    return ET_CD_ERR_ALIASING;

  if (state >= index->num_states || !bitset_get(index->live_bitset, state))
    return ET_CD_ERR_STATE;

  /* Validate all input logits are finite before modifying masked_logits. */
  for (uint32_t t = 0u; t < v; ++t) {
    if (!isfinite(logits[t]))
      return ET_CD_ERR_NONFINITE;
  }

  /* Fail closed if no token is allowed from this state (output preserved). */
  if (index->allowed_count[state] == 0u)
    return ET_CD_ERR_NO_VALID_TOKENS;

  const uint64_t *mask_row =
      index->allowed_mask + (size_t)state * (size_t)index->mask_words;
  const uint32_t neg_inf_bits = UINT32_C(0xFF800000);
  float neg_inf = 0.0f;
  memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));

  uint32_t t = 0u;
  for (uint32_t w = 0u; w < index->mask_words; ++w) {
    uint64_t word = mask_row[w];
    const uint32_t end = (t + 64u < v) ? (t + 64u) : v;
    for (; t < end; ++t) {
      if ((word & UINT64_C(1)) != 0u) {
        uint32_t bits = 0u;
        memcpy(&bits, &logits[t], sizeof(bits));
        memcpy(&masked_logits[t], &bits, sizeof(bits));
      } else {
        masked_logits[t] = neg_inf;
      }
      word >>= 1u;
    }
  }
  return ET_CD_OK;
}

et_cd_status et_cd_advance_token_v1(const et_cd_vocab_index *index,
                                    uint16_t state, uint32_t token_id,
                                    uint16_t *out_next_state,
                                    uint32_t *out_is_finished) {
  if (index == NULL || out_next_state == NULL || out_is_finished == NULL)
    return ET_CD_ERR_ARGUMENT;
  if (state >= index->num_states || !bitset_get(index->live_bitset, state))
    return ET_CD_ERR_STATE;
  if (token_id >= index->vocab_size)
    return ET_CD_ERR_ARGUMENT;

  const uint64_t *mask_row =
      index->allowed_mask + (size_t)state * (size_t)index->mask_words;
  if (!bitset_get(mask_row, token_id))
    return ET_CD_ERR_ILLEGAL_TOKEN;

  if (token_id == index->eos_token_id) {
    *out_next_state = ET_CD_DEAD_STATE;
    *out_is_finished = 1u;
    return ET_CD_OK;
  }

  const uint16_t next =
      index->dest_state[(size_t)state * (size_t)index->vocab_size + token_id];
  if (next >= index->num_states || !bitset_get(index->live_bitset, next))
    return ET_CD_ERR_STATE;

  *out_next_state = next;
  *out_is_finished = 0u;
  return ET_CD_OK;
}
