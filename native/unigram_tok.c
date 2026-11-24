#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/unigram_tok_abi.h"

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

_Static_assert(CHAR_BIT == 8 && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Unigram tokenizer requires 8-bit bytes and IEEE-754 binary64");
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "Unigram tokenizer requires IEEE-754 binary64 (FLT_EVAL_METHOD==0)");

/* ---- Runtime FP Environment Verification ---- */

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

/* ---- Standalone SHA-256 Implementation ---- */

typedef struct {
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
  size_t i;
  for (i = 0u; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
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
    if (take > length) take = length;
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
  for (i = 0u; i < 8u; ++i) {
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  }
  sha256_block(state, state->block);
  for (i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256_compute(const uint8_t *data, size_t size,
                           uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  if (size != 0u) {
    sha256_update(&state, data, size);
  }
  sha256_final(&state, digest);
}

static int sha256_ct_equal(const uint8_t a[32], const uint8_t b[32]) {
  uint8_t diff = 0u;
  size_t i;
  for (i = 0u; i < 32u; ++i) {
    diff |= (uint8_t)(a[i] ^ b[i]);
  }
  return diff == 0u;
}

/* ---- Alignment-Safe Little-Endian Encoding Helpers ---- */

static void put_le16(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static void put_le32(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void put_le64(uint8_t *dst, uint64_t v) {
  size_t i;
  for (i = 0u; i < 8u; ++i) {
    dst[i] = (uint8_t)((v >> (8u * i)) & UINT64_C(0xff));
  }
}

static void put_f64_le(uint8_t *dst, double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  put_le64(dst, bits);
}

static uint16_t get_le16(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static uint32_t get_le32(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint64_t get_le64(const uint8_t *src) {
  uint64_t v = 0u;
  size_t i;
  for (i = 0u; i < 8u; ++i) {
    v |= ((uint64_t)src[i]) << (8u * i);
  }
  return v;
}

static double get_f64_le(const uint8_t *src) {
  const uint64_t bits = get_le64(src);
  double v = 0.0;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

/* ---- Math Helpers (Binary64 Exact Order) ---- */

static double log_add_exp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  if (a < b) {
    const double tmp = a;
    a = b;
    b = tmp;
  }
  return a + log1p(exp(b - a));
}

static double log_sum_exp_array(const double *arr, size_t n) {
  double max_v = -INFINITY;
  size_t i;
  for (i = 0u; i < n; ++i) {
    if (arr[i] > max_v) max_v = arr[i];
  }
  if (!isfinite(max_v)) return max_v;
  double sum_exp = 0.0;
  for (i = 0u; i < n; ++i) {
    sum_exp += exp(arr[i] - max_v);
  }
  return max_v + log(sum_exp);
}

/* Exact binary64 Digamma psi(x) for x >= 0.1 via recurrence + Bernoulli series. */
static double digamma_f64(double x) {
  double result = 0.0;
  while (x < 8.0) {
    result -= 1.0 / x;
    x += 1.0;
  }
  const double inv = 1.0 / x;
  const double inv2 = inv * inv;
  result += log(x) - 0.5 * inv -
            inv2 * ((1.0 / 12.0) -
                    inv2 * ((1.0 / 120.0) -
                            inv2 * ((1.0 / 252.0) -
                                    inv2 * ((1.0 / 240.0) -
                                            inv2 * ((1.0 / 132.0) -
                                                    inv2 * (691.0 / 32760.0))))));
  return result;
}

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static uint32_t popcount_specials(uint32_t mask) {
  uint32_t count = 0u;
  if ((mask & ET_UNIGRAM_SPECIAL_BOS) != 0u) ++count;
  if ((mask & ET_UNIGRAM_SPECIAL_EOS) != 0u) ++count;
  if ((mask & ET_UNIGRAM_SPECIAL_PAD) != 0u) ++count;
  if ((mask & ET_UNIGRAM_SPECIAL_UNK) != 0u) ++count;
  return count;
}

static int compare_byte_slices(const uint8_t *a, uint32_t alen,
                               const uint8_t *b, uint32_t blen) {
  if (alen != blen) return (alen < blen) ? -1 : 1;
  if (alen == 0u) return 0;
  const int c = memcmp(a, b, (size_t)alen);
  if (c < 0) return -1;
  if (c > 0) return 1;
  return 0;
}

/* ---- Compact Deterministic Byte-Trie ---- */

typedef struct {
  uint32_t edge_offset;
  uint32_t edge_count;
  uint32_t token_id; /* UINT32_MAX if non-terminal */
} trie_node;

typedef struct {
  uint8_t byte;
  uint32_t child;
} trie_edge;

typedef struct {
  uint32_t root_child[256];
  trie_node *nodes;
  trie_edge *edges;
  uint32_t node_count;
  uint32_t edge_count;
} byte_trie;

static void byte_trie_free(byte_trie *trie) {
  if (trie == NULL) return;
  free(trie->nodes);
  free(trie->edges);
  trie->nodes = NULL;
  trie->edges = NULL;
  trie->node_count = 0u;
  trie->edge_count = 0u;
}

/* Builder node using sorted child linked list before flattening. */
typedef struct {
  uint32_t first_child;
  uint32_t next_sibling;
  uint32_t token_id;
  uint8_t byte;
} trie_build_node;

static et_unigram_status byte_trie_build(const uint8_t *piece_blob,
                                         const uint32_t *piece_offsets,
                                         const uint16_t *piece_lens,
                                         uint32_t active_count,
                                         byte_trie *out_trie) {
  memset(out_trie, 0, sizeof(*out_trie));
  size_t i;
  for (i = 0u; i < 256u; ++i) {
    out_trie->root_child[i] = UINT32_MAX;
  }

  /* Upper bound on nodes: 1 + sum(piece_lens). */
  size_t max_nodes = 1u;
  for (i = 0u; i < (size_t)active_count; ++i) {
    if (piece_lens[i] == 0u || piece_lens[i] > ET_UNIGRAM_MAX_PIECE_BYTES) {
      return ET_UNIGRAM_ERR_INVALID_VOCAB;
    }
    if (max_nodes > SIZE_MAX - (size_t)piece_lens[i]) {
      return ET_UNIGRAM_ERR_OVERFLOW;
    }
    max_nodes += (size_t)piece_lens[i];
  }

  trie_build_node *bnodes = malloc(max_nodes * sizeof(trie_build_node));
  if (bnodes == NULL) return ET_UNIGRAM_ERR_NOMEM;

  uint32_t used_nodes = 1u;
  bnodes[0].first_child = UINT32_MAX;
  bnodes[0].next_sibling = UINT32_MAX;
  bnodes[0].token_id = UINT32_MAX;
  bnodes[0].byte = 0u;

  for (uint32_t tok = 0u; tok < active_count; ++tok) {
    const uint8_t *p = piece_blob + piece_offsets[tok];
    const uint16_t plen = piece_lens[tok];
    uint32_t cur = 0u;
    for (uint16_t k = 0u; k < plen; ++k) {
      const uint8_t b = p[k];
      uint32_t prev = UINT32_MAX;
      uint32_t ch = bnodes[cur].first_child;
      while (ch != UINT32_MAX && bnodes[ch].byte < b) {
        prev = ch;
        ch = bnodes[ch].next_sibling;
      }
      if (ch != UINT32_MAX && bnodes[ch].byte == b) {
        cur = ch;
      } else {
        const uint32_t nid = used_nodes++;
        bnodes[nid].first_child = UINT32_MAX;
        bnodes[nid].next_sibling = ch;
        bnodes[nid].token_id = UINT32_MAX;
        bnodes[nid].byte = b;
        if (prev == UINT32_MAX) {
          bnodes[cur].first_child = nid;
        } else {
          bnodes[prev].next_sibling = nid;
        }
        cur = nid;
      }
    }
    if (bnodes[cur].token_id != UINT32_MAX) {
      /* Duplicate piece byte sequence! */
      free(bnodes);
      return ET_UNIGRAM_ERR_INVALID_VOCAB;
    }
    bnodes[cur].token_id = tok;
  }

  /* Flatten into compact sorted-edge arrays. */
  const uint32_t edge_total = (used_nodes > 0u) ? (used_nodes - 1u) : 0u;
  trie_node *nodes = calloc((size_t)used_nodes, sizeof(trie_node));
  trie_edge *edges =
      (edge_total > 0u) ? malloc((size_t)edge_total * sizeof(trie_edge)) : NULL;
  if (nodes == NULL || (edge_total > 0u && edges == NULL)) {
    free(nodes);
    free(edges);
    free(bnodes);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  uint32_t edge_cursor = 0u;
  for (uint32_t nid = 0u; nid < used_nodes; ++nid) {
    nodes[nid].token_id = bnodes[nid].token_id;
    nodes[nid].edge_offset = edge_cursor;
    uint32_t cnt = 0u;
    uint32_t ch = bnodes[nid].first_child;
    while (ch != UINT32_MAX) {
      edges[edge_cursor].byte = bnodes[ch].byte;
      edges[edge_cursor].child = ch;
      if (nid == 0u) {
        out_trie->root_child[bnodes[ch].byte] = ch;
      }
      ++edge_cursor;
      ++cnt;
      ch = bnodes[ch].next_sibling;
    }
    nodes[nid].edge_count = cnt;
  }

  free(bnodes);
  out_trie->nodes = nodes;
  out_trie->edges = edges;
  out_trie->node_count = used_nodes;
  out_trie->edge_count = edge_cursor;
  return ET_UNIGRAM_OK;
}

static uint32_t trie_step(const byte_trie *trie, uint32_t node, uint8_t b) {
  if (node == 0u) return trie->root_child[b];
  const uint32_t off = trie->nodes[node].edge_offset;
  const uint32_t cnt = trie->nodes[node].edge_count;
  if (cnt == 0u) return UINT32_MAX;
  uint32_t lo = 0u;
  uint32_t hi = cnt;
  while (lo < hi) {
    const uint32_t mid = lo + ((hi - lo) >> 1u);
    const uint8_t mb = trie->edges[off + mid].byte;
    if (mb == b) return trie->edges[off + mid].child;
    if (mb < b) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return UINT32_MAX;
}

/* ---- Model Struct ---- */

struct et_unigram_model {
  et_unigram_info_v1 info;
  uint8_t *piece_blob;
  size_t piece_blob_size;
  uint32_t *piece_offsets;
  uint16_t *piece_lens;
  double *log_probs;
  byte_trie trie;
};

static void model_destroy_internal(et_unigram_model *model) {
  if (model == NULL) return;
  byte_trie_free(&model->trie);
  free(model->piece_blob);
  free(model->piece_offsets);
  free(model->piece_lens);
  free(model->log_probs);
  free(model);
}

static et_unigram_status build_model_from_arrays(
    const uint8_t *blob, size_t blob_size, const uint32_t *offsets,
    const uint16_t *lens, const double *scores, uint32_t active_count,
    uint32_t special_mask, et_unigram_model **out_model) {
  *out_model = NULL;
  if (active_count < ET_UNIGRAM_BASE_VOCAB_SIZE ||
      active_count > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_INVALID_VOCAB;
  }
  if ((special_mask & ~(uint32_t)ET_UNIGRAM_SPECIAL_ALL) != 0u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  const uint32_t special_count = popcount_specials(special_mask);
  if (active_count + special_count > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }

  uint32_t max_len = 1u;
  for (uint32_t i = 0u; i < active_count; ++i) {
    if (!isfinite(scores[i])) return ET_UNIGRAM_ERR_NONFINITE;
    if (scores[i] > 0.0) return ET_UNIGRAM_ERR_INVALID_VOCAB;
    if (i < ET_UNIGRAM_BASE_VOCAB_SIZE) {
      if (lens[i] != 1u || blob[offsets[i]] != (uint8_t)i) {
        return ET_UNIGRAM_ERR_INVALID_VOCAB;
      }
    } else {
      if (lens[i] < 2u || lens[i] > ET_UNIGRAM_MAX_PIECE_BYTES) {
        return ET_UNIGRAM_ERR_INVALID_VOCAB;
      }
      if ((uint32_t)lens[i] > max_len) max_len = (uint32_t)lens[i];
    }
  }

  et_unigram_model *m = calloc(1u, sizeof(et_unigram_model));
  if (m == NULL) return ET_UNIGRAM_ERR_NOMEM;

  m->piece_blob = malloc(blob_size);
  m->piece_offsets = malloc((size_t)active_count * sizeof(uint32_t));
  m->piece_lens = malloc((size_t)active_count * sizeof(uint16_t));
  m->log_probs = malloc((size_t)active_count * sizeof(double));
  if (m->piece_blob == NULL || m->piece_offsets == NULL ||
      m->piece_lens == NULL || m->log_probs == NULL) {
    model_destroy_internal(m);
    return ET_UNIGRAM_ERR_NOMEM;
  }
  memcpy(m->piece_blob, blob, blob_size);
  memcpy(m->piece_offsets, offsets, (size_t)active_count * sizeof(uint32_t));
  memcpy(m->piece_lens, lens, (size_t)active_count * sizeof(uint16_t));
  memcpy(m->log_probs, scores, (size_t)active_count * sizeof(double));
  m->piece_blob_size = blob_size;

  const et_unigram_status tst = byte_trie_build(
      m->piece_blob, m->piece_offsets, m->piece_lens, active_count, &m->trie);
  if (tst != ET_UNIGRAM_OK) {
    model_destroy_internal(m);
    return tst;
  }

  uint32_t next_sp = active_count;
  m->info.struct_size = (uint32_t)sizeof(et_unigram_info_v1);
  m->info.vocab_size = active_count + special_count;
  m->info.active_count = active_count;
  m->info.learned_count = active_count - ET_UNIGRAM_BASE_VOCAB_SIZE;
  m->info.special_mask = special_mask;
  m->info.bos_id = ((special_mask & ET_UNIGRAM_SPECIAL_BOS) != 0u)
                       ? next_sp++
                       : ET_UNIGRAM_TOKEN_NONE;
  m->info.eos_id = ((special_mask & ET_UNIGRAM_SPECIAL_EOS) != 0u)
                       ? next_sp++
                       : ET_UNIGRAM_TOKEN_NONE;
  m->info.pad_id = ((special_mask & ET_UNIGRAM_SPECIAL_PAD) != 0u)
                       ? next_sp++
                       : ET_UNIGRAM_TOKEN_NONE;
  m->info.unk_id = ((special_mask & ET_UNIGRAM_SPECIAL_UNK) != 0u)
                       ? next_sp++
                       : ET_UNIGRAM_TOKEN_NONE;
  m->info.max_piece_bytes = max_len;

  *out_model = m;
  return ET_UNIGRAM_OK;
}

/* ---- Core Viterbi Lattice DP ---- */

/* Exact Viterbi DP over bytes[0..n-1].
 * exclude_id == UINT32_MAX enables all pieces; otherwise skips exclude_id.
 * Tie-break at each destination pos:
 *   1. higher cumulative score
 *   2. longer piece ending at pos
 *   3. smaller token_id
 */
static et_unigram_status viterbi_dp_internal(
    const byte_trie *trie, const double *scores, const uint8_t *bytes, size_t n,
    uint32_t exclude_id, uint32_t *out_tokens, size_t capacity,
    size_t *out_count, double *out_score) {
  if (n == 0u) {
    if (out_count != NULL) *out_count = 0u;
    if (out_score != NULL) *out_score = 0.0;
    return ET_UNIGRAM_OK;
  }
  if (n > ET_UNIGRAM_MAX_INPUT_BYTES) return ET_UNIGRAM_ERR_BOUNDS;

  double *best_score = malloc((n + 1u) * sizeof(double));
  uint32_t *best_tok = malloc((n + 1u) * sizeof(uint32_t));
  uint16_t *best_len = malloc((n + 1u) * sizeof(uint16_t));
  if (best_score == NULL || best_tok == NULL || best_len == NULL) {
    free(best_score);
    free(best_tok);
    free(best_len);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  best_score[0] = 0.0;
  best_tok[0] = UINT32_MAX;
  best_len[0] = 0u;
  for (size_t i = 1u; i <= n; ++i) {
    best_score[i] = -INFINITY;
    best_tok[i] = UINT32_MAX;
    best_len[i] = 0u;
  }

  for (size_t pos = 0u; pos < n; ++pos) {
    const double base = best_score[pos];
    if (base == -INFINITY) continue;
    uint32_t node = 0u;
    for (size_t k = 0u; pos + k < n; ++k) {
      node = trie_step(trie, node, bytes[pos + k]);
      if (node == UINT32_MAX) break;
      const uint32_t tok = trie->nodes[node].token_id;
      if (tok == UINT32_MAX || tok == exclude_id) continue;
      const uint16_t plen = (uint16_t)(k + 1u);
      const size_t next = pos + (size_t)plen;
      const double cand = base + scores[tok];
      if (!isfinite(cand)) {
        free(best_score);
        free(best_tok);
        free(best_len);
        return ET_UNIGRAM_ERR_NONFINITE;
      }
      int better = 0;
      if (best_tok[next] == UINT32_MAX) {
        better = 1;
      } else if (cand > best_score[next]) {
        better = 1;
      } else if (cand == best_score[next]) {
        if (plen > best_len[next] ||
            (plen == best_len[next] && tok < best_tok[next])) {
          better = 1;
        }
      }
      if (better) {
        best_score[next] = cand;
        best_tok[next] = tok;
        best_len[next] = plen;
      }
    }
  }

  if (best_tok[n] == UINT32_MAX || !isfinite(best_score[n])) {
    free(best_score);
    free(best_tok);
    free(best_len);
    return ET_UNIGRAM_ERR_NONFINITE;
  }

  /* Count path length first. */
  size_t count = 0u;
  size_t cur = n;
  while (cur > 0u) {
    const uint16_t plen = best_len[cur];
    if (plen == 0u || (size_t)plen > cur) {
      free(best_score);
      free(best_tok);
      free(best_len);
      return ET_UNIGRAM_ERR_INVALID_VOCAB;
    }
    cur -= (size_t)plen;
    ++count;
  }

  const double final_score = best_score[n];
  if (out_count != NULL) *out_count = count;
  if (out_score != NULL) *out_score = final_score;

  if (out_tokens == NULL) {
    free(best_score);
    free(best_tok);
    free(best_len);
    return (capacity == 0u) ? ET_UNIGRAM_OK : ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (capacity < count) {
    free(best_score);
    free(best_tok);
    free(best_len);
    return ET_UNIGRAM_ERR_BOUNDS;
  }

  cur = n;
  size_t idx = count;
  while (cur > 0u) {
    --idx;
    out_tokens[idx] = best_tok[cur];
    cur -= (size_t)best_len[cur];
  }

  free(best_score);
  free(best_tok);
  free(best_len);
  return ET_UNIGRAM_OK;
}

/* ---- Core Forward-Backward Lattice DP ---- */

/* Accumulates log Z into *acc_log_z and expected counts into acc_expected (if
 * non-NULL) for one byte sequence bytes[0..n-1]. Does NOT zero acc_expected. */
static et_unigram_status fb_dp_accumulate(const byte_trie *trie,
                                          const double *scores,
                                          const uint8_t *bytes, size_t n,
                                          double *acc_log_z,
                                          double *acc_expected) {
  if (n == 0u) return ET_UNIGRAM_OK;
  if (n > ET_UNIGRAM_MAX_INPUT_BYTES) return ET_UNIGRAM_ERR_BOUNDS;

  double *alpha = malloc((n + 1u) * sizeof(double));
  double *beta = (acc_expected != NULL) ? malloc((n + 1u) * sizeof(double)) : NULL;
  if (alpha == NULL || (acc_expected != NULL && beta == NULL)) {
    free(alpha);
    free(beta);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  alpha[0] = 0.0;
  for (size_t i = 1u; i <= n; ++i) {
    alpha[i] = -INFINITY;
  }

  for (size_t pos = 0u; pos < n; ++pos) {
    const double a_pos = alpha[pos];
    if (a_pos == -INFINITY) continue;
    uint32_t node = 0u;
    for (size_t k = 0u; pos + k < n; ++k) {
      node = trie_step(trie, node, bytes[pos + k]);
      if (node == UINT32_MAX) break;
      const uint32_t tok = trie->nodes[node].token_id;
      if (tok == UINT32_MAX) continue;
      const size_t next = pos + k + 1u;
      alpha[next] = log_add_exp(alpha[next], a_pos + scores[tok]);
    }
  }

  const double log_z = alpha[n];
  if (!isfinite(log_z)) {
    free(alpha);
    free(beta);
    return ET_UNIGRAM_ERR_NONFINITE;
  }

  if (acc_expected != NULL) {
    beta[n] = 0.0;
    for (size_t i = 0u; i < n; ++i) {
      beta[i] = -INFINITY;
    }
    for (size_t pos = n; pos > 0u; --pos) {
      const size_t p = pos - 1u;
      uint32_t node = 0u;
      for (size_t k = 0u; p + k < n; ++k) {
        node = trie_step(trie, node, bytes[p + k]);
        if (node == UINT32_MAX) break;
        const uint32_t tok = trie->nodes[node].token_id;
        if (tok == UINT32_MAX) continue;
        const size_t next = p + k + 1u;
        if (beta[next] == -INFINITY) continue;
        beta[p] = log_add_exp(beta[p], scores[tok] + beta[next]);
      }
    }
    if (!isfinite(beta[0])) {
      free(alpha);
      free(beta);
      return ET_UNIGRAM_ERR_NONFINITE;
    }
    for (size_t pos = 0u; pos < n; ++pos) {
      const double a_pos = alpha[pos];
      if (a_pos == -INFINITY) continue;
      uint32_t node = 0u;
      for (size_t k = 0u; pos + k < n; ++k) {
        node = trie_step(trie, node, bytes[pos + k]);
        if (node == UINT32_MAX) break;
        const uint32_t tok = trie->nodes[node].token_id;
        if (tok == UINT32_MAX) continue;
        const size_t next = pos + k + 1u;
        if (beta[next] == -INFINITY) continue;
        const double log_marg = a_pos + scores[tok] + beta[next] - log_z;
        const double contrib = exp(log_marg);
        if (!isfinite(contrib)) {
          free(alpha);
          free(beta);
          return ET_UNIGRAM_ERR_NONFINITE;
        }
        acc_expected[tok] += contrib;
      }
    }
  }

  *acc_log_z += log_z;
  if (!isfinite(*acc_log_z)) {
    free(alpha);
    free(beta);
    return ET_UNIGRAM_ERR_NONFINITE;
  }

  free(alpha);
  free(beta);
  return ET_UNIGRAM_OK;
}

/* ---- Seed Substring Counter Hash Table ---- */

typedef struct {
  uint32_t doc_idx;
  uint32_t start;
  uint32_t len;
  uint64_t count;
  uint64_t hash;
  uint8_t occupied;
} substr_slot;

typedef struct {
  substr_slot *slots;
  size_t capacity;
  size_t used;
} substr_table;

static uint64_t fnv1a_bytes(const uint8_t *p, uint32_t len) {
  uint64_t h = UINT64_C(14695981039346656037);
  for (uint32_t i = 0u; i < len; ++i) {
    h ^= (uint64_t)p[i];
    h *= UINT64_C(1099511628211);
  }
  /* Mix final bits. */
  h ^= h >> 33u;
  h *= UINT64_C(0xff51afd7ed558ccd);
  h ^= h >> 33u;
  return h;
}

static int substr_table_grow(substr_table *tbl,
                             const et_unigram_document_v1 *docs) {
  const size_t new_cap = (tbl->capacity == 0u) ? 4096u : (tbl->capacity * 2u);
  if (new_cap < tbl->capacity) return 0;
  substr_slot *ns = calloc(new_cap, sizeof(substr_slot));
  if (ns == NULL) return 0;

  const size_t mask = new_cap - 1u;
  for (size_t i = 0u; i < tbl->capacity; ++i) {
    if (!tbl->slots[i].occupied) continue;
    size_t idx = (size_t)(tbl->slots[i].hash & (uint64_t)mask);
    while (ns[idx].occupied) {
      idx = (idx + 1u) & mask;
    }
    ns[idx] = tbl->slots[i];
  }
  (void)docs;
  free(tbl->slots);
  tbl->slots = ns;
  tbl->capacity = new_cap;
  return 1;
}

static int substr_table_inc(substr_table *tbl,
                            const et_unigram_document_v1 *docs,
                            uint32_t doc_idx, uint32_t start, uint32_t len) {
  if (tbl->capacity == 0u || (tbl->used + 1u) * 10u >= tbl->capacity * 7u) {
    if (!substr_table_grow(tbl, docs)) return 0;
  }
  const uint8_t *p = docs[doc_idx].bytes + start;
  const uint64_t h = fnv1a_bytes(p, len);
  const size_t mask = tbl->capacity - 1u;
  size_t idx = (size_t)(h & (uint64_t)mask);
  while (tbl->slots[idx].occupied) {
    if (tbl->slots[idx].hash == h && tbl->slots[idx].len == len) {
      const uint8_t *q =
          docs[tbl->slots[idx].doc_idx].bytes + tbl->slots[idx].start;
      if (memcmp(p, q, (size_t)len) == 0) {
        tbl->slots[idx].count += 1u;
        return 1;
      }
    }
    idx = (idx + 1u) & mask;
  }
  tbl->slots[idx].doc_idx = doc_idx;
  tbl->slots[idx].start = start;
  tbl->slots[idx].len = len;
  tbl->slots[idx].count = 1u;
  tbl->slots[idx].hash = h;
  tbl->slots[idx].occupied = 1u;
  tbl->used += 1u;
  return 1;
}

/* ---- Deterministic Total-Order Comparators for qsort ---- */

typedef struct {
  const uint8_t *bytes;
  uint32_t len;
  uint64_t count;
  uint64_t score_cl; /* count * len */
} seed_candidate;

static int cmp_seed_candidate(const void *va, const void *vb) {
  const seed_candidate *a = (const seed_candidate *)va;
  const seed_candidate *b = (const seed_candidate *)vb;
  if (a->score_cl != b->score_cl) return (a->score_cl > b->score_cl) ? -1 : 1;
  if (a->count != b->count) return (a->count > b->count) ? -1 : 1;
  return compare_byte_slices(a->bytes, a->len, b->bytes, b->len);
}

typedef struct {
  const uint8_t *bytes;
  uint32_t len;
  uint32_t old_id;
  double delta_l;
  double expected;
  double theta;
} prune_entry;

static int cmp_prune_keep(const void *va, const void *vb) {
  const prune_entry *a = (const prune_entry *)va;
  const prune_entry *b = (const prune_entry *)vb;
  if (a->delta_l != b->delta_l) return (a->delta_l > b->delta_l) ? -1 : 1;
  if (a->expected != b->expected) return (a->expected > b->expected) ? -1 : 1;
  return compare_byte_slices(a->bytes, a->len, b->bytes, b->len);
}

typedef struct {
  const uint8_t *bytes;
  uint32_t len;
  double theta;
} canonical_entry;

static int cmp_canonical_piece(const void *va, const void *vb) {
  const canonical_entry *a = (const canonical_entry *)va;
  const canonical_entry *b = (const canonical_entry *)vb;
  if (a->theta != b->theta) return (a->theta > b->theta) ? -1 : 1;
  return compare_byte_slices(a->bytes, a->len, b->bytes, b->len);
}

/* ---- EM M-Step Update ---- */

static et_unigram_status run_mstep(const double *expected, uint32_t active_count,
                                   uint32_t mstep_mode, double smoothing_alpha,
                                   double *theta) {
  double sum_c = 0.0;
  for (uint32_t i = 0u; i < active_count; ++i) {
    sum_c += expected[i] + smoothing_alpha;
  }
  if (!isfinite(sum_c) || sum_c <= 0.0) return ET_UNIGRAM_ERR_NONFINITE;

  if (mstep_mode == ET_UNIGRAM_MSTEP_MLE) {
    const double log_sum = log(sum_c);
    for (uint32_t i = 0u; i < active_count; ++i) {
      theta[i] = log(expected[i] + smoothing_alpha) - log_sum;
      if (!isfinite(theta[i])) return ET_UNIGRAM_ERR_NONFINITE;
    }
  } else {
    const double psi_sum = digamma_f64(sum_c);
    for (uint32_t i = 0u; i < active_count; ++i) {
      theta[i] = digamma_f64(expected[i] + smoothing_alpha) - psi_sum;
      if (!isfinite(theta[i])) return ET_UNIGRAM_ERR_NONFINITE;
    }
  }

  /* Normalize via log-sum-exp so sum_i exp(theta[i]) == 1.0 exactly. */
  const double lse = log_sum_exp_array(theta, (size_t)active_count);
  if (!isfinite(lse)) return ET_UNIGRAM_ERR_NONFINITE;
  for (uint32_t i = 0u; i < active_count; ++i) {
    theta[i] -= lse;
    if (!isfinite(theta[i]) || theta[i] > 0.0) return ET_UNIGRAM_ERR_NONFINITE;
  }
  return ET_UNIGRAM_OK;
}

static et_unigram_status corpus_estep_internal(
    const byte_trie *trie, const double *theta, uint32_t active_count,
    const et_unigram_document_v1 *docs, size_t doc_count, double *out_ll,
    double *expected) {
  double ll = 0.0;
  if (expected != NULL) {
    for (uint32_t i = 0u; i < active_count; ++i) {
      expected[i] = 0.0;
    }
  }
  for (size_t d = 0u; d < doc_count; ++d) {
    if (docs[d].byte_count == 0u) continue;
    const et_unigram_status st = fb_dp_accumulate(
        trie, theta, docs[d].bytes, docs[d].byte_count, &ll, expected);
    if (st != ET_UNIGRAM_OK) return st;
  }
  *out_ll = ll;
  return ET_UNIGRAM_OK;
}

/* ---- Public API: Create & Train ---- */

ET_UNIGRAM_API et_unigram_status et_unigram_create_v1(
    const et_unigram_piece_spec_v1 *pieces, size_t piece_count,
    uint32_t special_mask, et_unigram_model **out_model) {
  if (out_model != NULL) *out_model = NULL;
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (pieces == NULL || out_model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (piece_count < ET_UNIGRAM_BASE_VOCAB_SIZE ||
      piece_count > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_INVALID_VOCAB;
  }
  if ((special_mask & ~(uint32_t)ET_UNIGRAM_SPECIAL_ALL) != 0u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }

  size_t total_blob = 0u;
  for (size_t i = 0u; i < piece_count; ++i) {
    if (pieces[i].bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
    if (!isfinite(pieces[i].log_prob)) return ET_UNIGRAM_ERR_NONFINITE;
    if (pieces[i].log_prob > 0.0) return ET_UNIGRAM_ERR_INVALID_VOCAB;
    if (i < ET_UNIGRAM_BASE_VOCAB_SIZE) {
      if (pieces[i].byte_count != 1u || pieces[i].bytes[0] != (uint8_t)i) {
        return ET_UNIGRAM_ERR_INVALID_VOCAB;
      }
    } else {
      if (pieces[i].byte_count < 2u ||
          pieces[i].byte_count > ET_UNIGRAM_MAX_PIECE_BYTES) {
        return ET_UNIGRAM_ERR_INVALID_VOCAB;
      }
    }
    if (total_blob > UINT32_MAX - pieces[i].byte_count) {
      return ET_UNIGRAM_ERR_OVERFLOW;
    }
    total_blob += pieces[i].byte_count;
  }

  uint8_t *blob = malloc(total_blob);
  uint32_t *offsets = malloc(piece_count * sizeof(uint32_t));
  uint16_t *lens = malloc(piece_count * sizeof(uint16_t));
  double *scores = malloc(piece_count * sizeof(double));
  if (blob == NULL || offsets == NULL || lens == NULL || scores == NULL) {
    free(blob);
    free(offsets);
    free(lens);
    free(scores);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  uint32_t cur = 0u;
  for (size_t i = 0u; i < piece_count; ++i) {
    offsets[i] = cur;
    lens[i] = (uint16_t)pieces[i].byte_count;
    memcpy(blob + cur, pieces[i].bytes, pieces[i].byte_count);
    cur += (uint32_t)pieces[i].byte_count;
    scores[i] = pieces[i].log_prob;
  }

  const et_unigram_status st = build_model_from_arrays(
      blob, total_blob, offsets, lens, scores, (uint32_t)piece_count,
      special_mask, out_model);
  free(blob);
  free(offsets);
  free(lens);
  free(scores);
  return st;
}

static et_unigram_status validate_train_config(
    const et_unigram_train_config_v1 *cfg) {
  if (cfg == NULL || cfg->struct_size != sizeof(et_unigram_train_config_v1)) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->reserved != 0u) return ET_UNIGRAM_ERR_ARGUMENT;
  if ((cfg->special_mask & ~(uint32_t)ET_UNIGRAM_SPECIAL_ALL) != 0u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  const uint32_t special_count = popcount_specials(cfg->special_mask);
  if (cfg->target_vocab_size < ET_UNIGRAM_BASE_VOCAB_SIZE + special_count) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->target_vocab_size > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }
  const uint32_t target_active = cfg->target_vocab_size - special_count;
  if (cfg->seed_vocab_size < target_active ||
      cfg->seed_vocab_size > ET_UNIGRAM_MAX_SEED_VOCAB) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->max_piece_bytes < 2u ||
      cfg->max_piece_bytes > ET_UNIGRAM_MAX_TRAIN_PIECE_BYTES) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->min_frequency < 1u) return ET_UNIGRAM_ERR_ARGUMENT;
  if (cfg->num_em_sub_iters < 1u || cfg->num_em_sub_iters > 32u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->final_em_iters < 1u || cfg->final_em_iters > 64u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (cfg->mstep_mode != ET_UNIGRAM_MSTEP_MLE &&
      cfg->mstep_mode != ET_UNIGRAM_MSTEP_DIGAMMA_VB) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (!isfinite(cfg->shrinking_factor) || !isfinite(cfg->smoothing_alpha)) {
    return ET_UNIGRAM_ERR_NONFINITE;
  }
  if (cfg->shrinking_factor <= 0.1 || cfg->shrinking_factor > 0.99) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  const double min_alpha =
      (cfg->mstep_mode == ET_UNIGRAM_MSTEP_DIGAMMA_VB) ? 0.1 : 1e-6;
  if (cfg->smoothing_alpha < min_alpha || cfg->smoothing_alpha > 10.0) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_train_v1(
    const et_unigram_document_v1 *docs, size_t doc_count,
    const et_unigram_train_config_v1 *config, et_unigram_model **out_model,
    et_unigram_train_stats_v1 *out_stats) {
  if (out_model != NULL) *out_model = NULL;
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (out_model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (doc_count > 0u && docs == NULL) return ET_UNIGRAM_ERR_ARGUMENT;

  const et_unigram_status cfg_st = validate_train_config(config);
  if (cfg_st != ET_UNIGRAM_OK) return cfg_st;

  size_t total_corpus_bytes = 0u;
  for (size_t d = 0u; d < doc_count; ++d) {
    if (docs[d].byte_count > 0u && docs[d].bytes == NULL) {
      return ET_UNIGRAM_ERR_ARGUMENT;
    }
    if (docs[d].byte_count > ET_UNIGRAM_MAX_CORPUS_BYTES - total_corpus_bytes) {
      return ET_UNIGRAM_ERR_BOUNDS;
    }
    total_corpus_bytes += docs[d].byte_count;
  }

  const uint32_t special_count = popcount_specials(config->special_mask);
  const uint32_t target_active = config->target_vocab_size - special_count;

  /* 1. Count raw byte frequencies and multi-byte substrings. */
  uint64_t byte_counts[256];
  memset(byte_counts, 0, sizeof(byte_counts));

  substr_table tbl;
  memset(&tbl, 0, sizeof(tbl));

  for (size_t d = 0u; d < doc_count; ++d) {
    const uint8_t *p = docs[d].bytes;
    const size_t n = docs[d].byte_count;
    for (size_t i = 0u; i < n; ++i) {
      byte_counts[p[i]] += 1u;
      if (target_active > ET_UNIGRAM_BASE_VOCAB_SIZE) {
        size_t max_l = n - i;
        if (max_l > (size_t)config->max_piece_bytes) {
          max_l = (size_t)config->max_piece_bytes;
        }
        for (size_t l = 2u; l <= max_l; ++l) {
          if (!substr_table_inc(&tbl, docs, (uint32_t)d, (uint32_t)i,
                                (uint32_t)l)) {
            free(tbl.slots);
            return ET_UNIGRAM_ERR_NOMEM;
          }
        }
      }
    }
  }

  /* Collect qualifying candidates (count >= min_frequency). */
  seed_candidate *cands = NULL;
  size_t cand_count = 0u;
  if (tbl.used > 0u) {
    cands = malloc(tbl.used * sizeof(seed_candidate));
    if (cands == NULL) {
      free(tbl.slots);
      return ET_UNIGRAM_ERR_NOMEM;
    }
    for (size_t i = 0u; i < tbl.capacity; ++i) {
      if (!tbl.slots[i].occupied) continue;
      if (tbl.slots[i].count < (uint64_t)config->min_frequency) continue;
      cands[cand_count].bytes =
          docs[tbl.slots[i].doc_idx].bytes + tbl.slots[i].start;
      cands[cand_count].len = tbl.slots[i].len;
      cands[cand_count].count = tbl.slots[i].count;
      cands[cand_count].score_cl =
          tbl.slots[i].count * (uint64_t)tbl.slots[i].len;
      ++cand_count;
    }
    if (cand_count > 1u) {
      qsort(cands, cand_count, sizeof(seed_candidate), cmp_seed_candidate);
    }
  }
  free(tbl.slots);

  const uint32_t max_seed_extra =
      config->seed_vocab_size - ET_UNIGRAM_BASE_VOCAB_SIZE;
  const uint32_t seed_extra = (cand_count > (size_t)max_seed_extra)
                                  ? max_seed_extra
                                  : (uint32_t)cand_count;
  uint32_t active_count = ET_UNIGRAM_BASE_VOCAB_SIZE + seed_extra;

  /* Build working vocabulary blob/arrays for seed pieces. */
  size_t blob_size = ET_UNIGRAM_BASE_VOCAB_SIZE;
  for (uint32_t i = 0u; i < seed_extra; ++i) {
    blob_size += (size_t)cands[i].len;
  }

  uint8_t *blob = malloc(blob_size);
  uint32_t *offsets = malloc((size_t)active_count * sizeof(uint32_t));
  uint16_t *lens = malloc((size_t)active_count * sizeof(uint16_t));
  double *theta = malloc((size_t)active_count * sizeof(double));
  double *expected = malloc((size_t)active_count * sizeof(double));
  if (blob == NULL || offsets == NULL || lens == NULL || theta == NULL ||
      expected == NULL) {
    free(cands);
    free(blob);
    free(offsets);
    free(lens);
    free(theta);
    free(expected);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  uint32_t cursor = 0u;
  double total_w = 0.0;
  for (uint32_t b = 0u; b < ET_UNIGRAM_BASE_VOCAB_SIZE; ++b) {
    blob[cursor] = (uint8_t)b;
    offsets[b] = cursor;
    lens[b] = 1u;
    cursor += 1u;
    const double w = (double)byte_counts[b] + config->smoothing_alpha;
    theta[b] = w;
    total_w += w;
  }
  for (uint32_t i = 0u; i < seed_extra; ++i) {
    const uint32_t id = ET_UNIGRAM_BASE_VOCAB_SIZE + i;
    offsets[id] = cursor;
    lens[id] = (uint16_t)cands[i].len;
    memcpy(blob + cursor, cands[i].bytes, (size_t)cands[i].len);
    cursor += cands[i].len;
    const double w = (double)cands[i].count + config->smoothing_alpha;
    theta[id] = w;
    total_w += w;
  }
  free(cands);

  const double log_total_w = log(total_w);
  for (uint32_t i = 0u; i < active_count; ++i) {
    theta[i] = log(theta[i]) - log_total_w;
  }
  {
    const double lse = log_sum_exp_array(theta, (size_t)active_count);
    for (uint32_t i = 0u; i < active_count; ++i) {
      theta[i] -= lse;
    }
  }

  byte_trie trie;
  memset(&trie, 0, sizeof(trie));
  et_unigram_status st =
      byte_trie_build(blob, offsets, lens, active_count, &trie);
  if (st != ET_UNIGRAM_OK) goto cleanup;

  const uint32_t seed_active_count = active_count;
  uint32_t pruning_rounds = 0u;
  uint32_t total_em_iters = 0u;
  double initial_ll = 0.0;
  st = corpus_estep_internal(&trie, theta, active_count, docs, doc_count,
                             &initial_ll, NULL);
  if (st != ET_UNIGRAM_OK) goto cleanup;

  /* 2. Iterative EM + Per-Piece Pruning Loop. */
  while (active_count > target_active) {
    ++pruning_rounds;
    for (uint32_t it = 0u; it < config->num_em_sub_iters; ++it) {
      double iter_ll = 0.0;
      st = corpus_estep_internal(&trie, theta, active_count, docs, doc_count,
                                 &iter_ll, expected);
      if (st != ET_UNIGRAM_OK) goto cleanup;
      st = run_mstep(expected, active_count, config->mstep_mode,
                     config->smoothing_alpha, theta);
      if (st != ET_UNIGRAM_OK) goto cleanup;
      ++total_em_iters;
    }

    /* Fresh E-step aligned with updated theta for pruning loss delta_L. */
    double post_ll = 0.0;
    st = corpus_estep_internal(&trie, theta, active_count, docs, doc_count,
                               &post_ll, expected);
    if (st != ET_UNIGRAM_OK) goto cleanup;

    const uint32_t cur_extra = active_count - ET_UNIGRAM_BASE_VOCAB_SIZE;
    const uint32_t target_extra = target_active - ET_UNIGRAM_BASE_VOCAB_SIZE;
    uint32_t desired_extra =
        (uint32_t)floor((double)cur_extra * config->shrinking_factor);
    if (desired_extra >= cur_extra) desired_extra = cur_extra - 1u;
    if (desired_extra < target_extra) desired_extra = target_extra;

    prune_entry *entries = malloc((size_t)cur_extra * sizeof(prune_entry));
    if (entries == NULL) {
      st = ET_UNIGRAM_ERR_NOMEM;
      goto cleanup;
    }

    for (uint32_t k = 0u; k < cur_extra; ++k) {
      const uint32_t id = ET_UNIGRAM_BASE_VOCAB_SIZE + k;
      double alt_score = 0.0;
      st = viterbi_dp_internal(&trie, theta, blob + offsets[id],
                               (size_t)lens[id], id, NULL, 0u, NULL,
                               &alt_score);
      if (st != ET_UNIGRAM_OK) {
        free(entries);
        goto cleanup;
      }
      entries[k].bytes = blob + offsets[id];
      entries[k].len = (uint32_t)lens[id];
      entries[k].old_id = id;
      entries[k].delta_l = expected[id] * (theta[id] - alt_score);
      entries[k].expected = expected[id];
      entries[k].theta = theta[id];
    }

    if (cur_extra > 1u) {
      qsort(entries, (size_t)cur_extra, sizeof(prune_entry), cmp_prune_keep);
    }

    const uint32_t new_active = ET_UNIGRAM_BASE_VOCAB_SIZE + desired_extra;
    size_t new_blob_size = ET_UNIGRAM_BASE_VOCAB_SIZE;
    for (uint32_t k = 0u; k < desired_extra; ++k) {
      new_blob_size += (size_t)entries[k].len;
    }
    uint8_t *new_blob = malloc(new_blob_size);
    uint32_t *new_offsets = malloc((size_t)new_active * sizeof(uint32_t));
    uint16_t *new_lens = malloc((size_t)new_active * sizeof(uint16_t));
    double *new_theta = malloc((size_t)new_active * sizeof(double));
    if (new_blob == NULL || new_offsets == NULL || new_lens == NULL ||
        new_theta == NULL) {
      free(entries);
      free(new_blob);
      free(new_offsets);
      free(new_lens);
      free(new_theta);
      st = ET_UNIGRAM_ERR_NOMEM;
      goto cleanup;
    }

    uint32_t ncur = 0u;
    for (uint32_t b = 0u; b < ET_UNIGRAM_BASE_VOCAB_SIZE; ++b) {
      new_blob[ncur] = (uint8_t)b;
      new_offsets[b] = ncur;
      new_lens[b] = 1u;
      new_theta[b] = theta[b];
      ncur += 1u;
    }
    for (uint32_t k = 0u; k < desired_extra; ++k) {
      const uint32_t nid = ET_UNIGRAM_BASE_VOCAB_SIZE + k;
      new_offsets[nid] = ncur;
      new_lens[nid] = (uint16_t)entries[k].len;
      memcpy(new_blob + ncur, entries[k].bytes, (size_t)entries[k].len);
      new_theta[nid] = entries[k].theta;
      ncur += entries[k].len;
    }
    free(entries);

    const double lse = log_sum_exp_array(new_theta, (size_t)new_active);
    for (uint32_t i = 0u; i < new_active; ++i) {
      new_theta[i] -= lse;
    }

    free(blob);
    free(offsets);
    free(lens);
    free(theta);
    blob = new_blob;
    blob_size = new_blob_size;
    offsets = new_offsets;
    lens = new_lens;
    theta = new_theta;
    active_count = new_active;

    byte_trie_free(&trie);
    st = byte_trie_build(blob, offsets, lens, active_count, &trie);
    if (st != ET_UNIGRAM_OK) goto cleanup;
  }

  /* 3. Final EM Polish on pruned vocabulary. */
  double final_em_start_ll = 0.0;
  for (uint32_t it = 0u; it < config->final_em_iters; ++it) {
    double iter_ll = 0.0;
    st = corpus_estep_internal(&trie, theta, active_count, docs, doc_count,
                               &iter_ll, expected);
    if (st != ET_UNIGRAM_OK) goto cleanup;
    if (it == 0u) final_em_start_ll = iter_ll;
    st = run_mstep(expected, active_count, config->mstep_mode,
                   config->smoothing_alpha, theta);
    if (st != ET_UNIGRAM_OK) goto cleanup;
    ++total_em_iters;
  }

  /* 4. Canonical ordering of learned pieces + final LSE normalization. */
  const uint32_t learned_count = active_count - ET_UNIGRAM_BASE_VOCAB_SIZE;
  if (learned_count > 0u) {
    canonical_entry *cent =
        malloc((size_t)learned_count * sizeof(canonical_entry));
    uint8_t *cblob = malloc(blob_size);
    uint32_t *coff = malloc((size_t)active_count * sizeof(uint32_t));
    uint16_t *clens = malloc((size_t)active_count * sizeof(uint16_t));
    double *ctheta = malloc((size_t)active_count * sizeof(double));
    if (cent == NULL || cblob == NULL || coff == NULL || clens == NULL ||
        ctheta == NULL) {
      free(cent);
      free(cblob);
      free(coff);
      free(clens);
      free(ctheta);
      st = ET_UNIGRAM_ERR_NOMEM;
      goto cleanup;
    }

    for (uint32_t k = 0u; k < learned_count; ++k) {
      const uint32_t id = ET_UNIGRAM_BASE_VOCAB_SIZE + k;
      cent[k].bytes = blob + offsets[id];
      cent[k].len = (uint32_t)lens[id];
      cent[k].theta = theta[id];
    }
    if (learned_count > 1u) {
      qsort(cent, (size_t)learned_count, sizeof(canonical_entry),
            cmp_canonical_piece);
    }

    uint32_t ccur = 0u;
    for (uint32_t b = 0u; b < ET_UNIGRAM_BASE_VOCAB_SIZE; ++b) {
      cblob[ccur] = (uint8_t)b;
      coff[b] = ccur;
      clens[b] = 1u;
      ctheta[b] = theta[b];
      ccur += 1u;
    }
    for (uint32_t k = 0u; k < learned_count; ++k) {
      const uint32_t id = ET_UNIGRAM_BASE_VOCAB_SIZE + k;
      coff[id] = ccur;
      clens[id] = (uint16_t)cent[k].len;
      memcpy(cblob + ccur, cent[k].bytes, (size_t)cent[k].len);
      ctheta[id] = cent[k].theta;
      ccur += cent[k].len;
    }
    free(cent);

    const double lse = log_sum_exp_array(ctheta, (size_t)active_count);
    for (uint32_t i = 0u; i < active_count; ++i) {
      ctheta[i] -= lse;
    }

    free(blob);
    free(offsets);
    free(lens);
    free(theta);
    blob = cblob;
    offsets = coff;
    lens = clens;
    theta = ctheta;
  } else {
    const double lse = log_sum_exp_array(theta, (size_t)active_count);
    for (uint32_t i = 0u; i < active_count; ++i) {
      theta[i] -= lse;
    }
  }

  et_unigram_model *final_model = NULL;
  st = build_model_from_arrays(blob, blob_size, offsets, lens, theta,
                               active_count, config->special_mask,
                               &final_model);
  if (st != ET_UNIGRAM_OK) goto cleanup;

  /* Final evaluation E-step on the exact emitted canonical model. */
  double final_em_end_ll = 0.0;
  st = corpus_estep_internal(&final_model->trie, final_model->log_probs,
                             final_model->info.active_count, docs, doc_count,
                             &final_em_end_ll, NULL);
  if (st != ET_UNIGRAM_OK) {
    model_destroy_internal(final_model);
    goto cleanup;
  }

  if (out_stats != NULL) {
    out_stats->struct_size = (uint32_t)sizeof(et_unigram_train_stats_v1);
    out_stats->seed_active_count = seed_active_count;
    out_stats->final_active_count = active_count;
    out_stats->pruning_rounds = pruning_rounds;
    out_stats->total_em_iters = total_em_iters;
    out_stats->reserved = 0u;
    out_stats->initial_log_likelihood = initial_ll;
    out_stats->final_em_start_ll = final_em_start_ll;
    out_stats->final_em_end_ll = final_em_end_ll;
  }

  *out_model = final_model;
  st = ET_UNIGRAM_OK;

cleanup:
  byte_trie_free(&trie);
  free(blob);
  free(offsets);
  free(lens);
  free(theta);
  free(expected);
  return st;
}

ET_UNIGRAM_API void et_unigram_destroy_v1(et_unigram_model *model) {
  model_destroy_internal(model);
}

/* ---- Public API: Metadata & Piece Inspection ---- */

ET_UNIGRAM_API et_unigram_status et_unigram_get_info_v1(
    const et_unigram_model *model, et_unigram_info_v1 *out_info) {
  if (model == NULL || out_info == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  *out_info = model->info;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_get_piece_v1(
    const et_unigram_model *model, uint32_t token_id, uint8_t *out_bytes,
    size_t capacity, size_t *out_len, double *out_log_prob) {
  if (model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (token_id >= model->info.vocab_size) return ET_UNIGRAM_ERR_TOKEN_ID;
  if (out_bytes == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;

  if (token_id >= model->info.active_count) {
    if (out_len != NULL) *out_len = 0u;
    if (out_log_prob != NULL) *out_log_prob = 0.0;
    return ET_UNIGRAM_OK;
  }

  const size_t plen = (size_t)model->piece_lens[token_id];
  if (out_len != NULL) *out_len = plen;
  if (out_log_prob != NULL) *out_log_prob = model->log_probs[token_id];
  if (out_bytes == NULL) return ET_UNIGRAM_OK;
  if (capacity < plen) return ET_UNIGRAM_ERR_BOUNDS;
  memcpy(out_bytes, model->piece_blob + model->piece_offsets[token_id], plen);
  return ET_UNIGRAM_OK;
}

/* ---- Public API: Viterbi Encode & Lossless Decode ---- */

ET_UNIGRAM_API et_unigram_status et_unigram_encode_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t *out_tokens, size_t capacity, size_t *out_count,
    double *out_score) {
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (model == NULL || out_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count > 0u && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_tokens == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;

  return viterbi_dp_internal(&model->trie, model->log_probs, bytes, byte_count,
                             UINT32_MAX, out_tokens, capacity, out_count,
                             out_score);
}

ET_UNIGRAM_API et_unigram_status et_unigram_encode_ex_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t add_bos, uint32_t add_eos, uint32_t *out_tokens, size_t capacity,
    size_t *out_count, double *out_score) {
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (model == NULL || out_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count > 0u && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_tokens == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;
  if (add_bos > 1u || add_eos > 1u) return ET_UNIGRAM_ERR_ARGUMENT;
  if (add_bos != 0u && model->info.bos_id == ET_UNIGRAM_TOKEN_NONE) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (add_eos != 0u && model->info.eos_id == ET_UNIGRAM_TOKEN_NONE) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }

  size_t core_count = 0u;
  double core_score = 0.0;
  et_unigram_status st = viterbi_dp_internal(
      &model->trie, model->log_probs, bytes, byte_count, UINT32_MAX, NULL, 0u,
      &core_count, &core_score);
  if (st != ET_UNIGRAM_OK) return st;

  const size_t extra = (size_t)add_bos + (size_t)add_eos;
  if (core_count > SIZE_MAX - extra) return ET_UNIGRAM_ERR_OVERFLOW;
  const size_t total = core_count + extra;
  *out_count = total;
  if (out_score != NULL) *out_score = core_score;
  if (out_tokens == NULL) return ET_UNIGRAM_OK;
  if (capacity < total) return ET_UNIGRAM_ERR_BOUNDS;

  size_t write_pos = 0u;
  if (add_bos != 0u) out_tokens[write_pos++] = model->info.bos_id;
  if (core_count > 0u) {
    size_t check_count = 0u;
    st = viterbi_dp_internal(&model->trie, model->log_probs, bytes, byte_count,
                             UINT32_MAX, out_tokens + write_pos, core_count,
                             &check_count, NULL);
    if (st != ET_UNIGRAM_OK) return st;
    write_pos += check_count;
  }
  if (add_eos != 0u) out_tokens[write_pos++] = model->info.eos_id;
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_decode_v1(
    const et_unigram_model *model, const uint32_t *tokens, size_t token_count,
    uint8_t *out_bytes, size_t capacity, size_t *out_byte_count) {
  if (model == NULL || out_byte_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (token_count > 0u && tokens == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_bytes == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;

  /* First pass: validate every token ID and compute exact output byte count. */
  size_t total = 0u;
  for (size_t i = 0u; i < token_count; ++i) {
    const uint32_t tid = tokens[i];
    if (tid >= model->info.vocab_size) return ET_UNIGRAM_ERR_TOKEN_ID;
    if (tid < model->info.active_count) {
      const size_t plen = (size_t)model->piece_lens[tid];
      if (total > SIZE_MAX - plen) return ET_UNIGRAM_ERR_OVERFLOW;
      total += plen;
    }
  }

  *out_byte_count = total;
  if (out_bytes == NULL) return ET_UNIGRAM_OK;
  if (capacity < total) return ET_UNIGRAM_ERR_BOUNDS;

  size_t cursor = 0u;
  for (size_t i = 0u; i < token_count; ++i) {
    const uint32_t tid = tokens[i];
    if (tid < model->info.active_count) {
      const size_t plen = (size_t)model->piece_lens[tid];
      memcpy(out_bytes + cursor,
             model->piece_blob + model->piece_offsets[tid], plen);
      cursor += plen;
    }
  }
  return ET_UNIGRAM_OK;
}

/* ---- Public API: Forward-Backward & Corpus Forward-Backward ---- */

ET_UNIGRAM_API et_unigram_status et_unigram_forward_backward_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    double *out_log_z, double *out_expected_counts, size_t count_capacity) {
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (model == NULL || out_log_z == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count > 0u && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_expected_counts == NULL && count_capacity != 0u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (out_expected_counts != NULL &&
      count_capacity < (size_t)model->info.active_count) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }

  double *tmp_exp = NULL;
  if (out_expected_counts != NULL) {
    tmp_exp = calloc((size_t)model->info.active_count, sizeof(double));
    if (tmp_exp == NULL) return ET_UNIGRAM_ERR_NOMEM;
  }

  double log_z = 0.0;
  const et_unigram_status st = fb_dp_accumulate(
      &model->trie, model->log_probs, bytes, byte_count, &log_z, tmp_exp);
  if (st != ET_UNIGRAM_OK) {
    free(tmp_exp);
    return st;
  }

  *out_log_z = log_z;
  if (out_expected_counts != NULL) {
    memcpy(out_expected_counts, tmp_exp,
           (size_t)model->info.active_count * sizeof(double));
    free(tmp_exp);
  }
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_corpus_fb_v1(
    const et_unigram_model *model, const et_unigram_document_v1 *docs,
    size_t doc_count, double *out_log_z, double *out_expected_counts,
    size_t count_capacity) {
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (model == NULL || out_log_z == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (doc_count > 0u && docs == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_expected_counts == NULL && count_capacity != 0u) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  if (out_expected_counts != NULL &&
      count_capacity < (size_t)model->info.active_count) {
    return ET_UNIGRAM_ERR_BOUNDS;
  }
  for (size_t d = 0u; d < doc_count; ++d) {
    if (docs[d].byte_count > 0u && docs[d].bytes == NULL) {
      return ET_UNIGRAM_ERR_ARGUMENT;
    }
  }

  double *tmp_exp = NULL;
  if (out_expected_counts != NULL) {
    tmp_exp = calloc((size_t)model->info.active_count, sizeof(double));
    if (tmp_exp == NULL) return ET_UNIGRAM_ERR_NOMEM;
  }

  double log_z = 0.0;
  const et_unigram_status st =
      corpus_estep_internal(&model->trie, model->log_probs,
                            model->info.active_count, docs, doc_count, &log_z,
                            tmp_exp);
  if (st != ET_UNIGRAM_OK) {
    free(tmp_exp);
    return st;
  }

  *out_log_z = log_z;
  if (out_expected_counts != NULL) {
    memcpy(out_expected_counts, tmp_exp,
           (size_t)model->info.active_count * sizeof(double));
    free(tmp_exp);
  }
  return ET_UNIGRAM_OK;
}

/* ---- Public API: Kudo (2018) FFBS Subword Regularization ---- */

typedef struct {
  uint32_t start;
  uint32_t tok;
} ffbs_in_edge;

ET_UNIGRAM_API et_unigram_status et_unigram_sample_v1(
    const et_unigram_model *model, const uint8_t *bytes, size_t byte_count,
    double alpha_temp, uint64_t seed, uint32_t *out_tokens, size_t capacity,
    size_t *out_count, double *out_score) {
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (model == NULL || out_count == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count > 0u && bytes == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_tokens == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;
  if (!isfinite(alpha_temp)) return ET_UNIGRAM_ERR_NONFINITE;
  if (alpha_temp <= 0.0 || alpha_temp > 1000.0) return ET_UNIGRAM_ERR_ARGUMENT;
  if (byte_count == 0u) {
    *out_count = 0u;
    if (out_score != NULL) *out_score = 0.0;
    return ET_UNIGRAM_OK;
  }
  if (byte_count > ET_UNIGRAM_MAX_INPUT_BYTES) return ET_UNIGRAM_ERR_BOUNDS;

  const size_t n = byte_count;
  double *alpha = malloc((n + 1u) * sizeof(double));
  uint32_t *in_deg = calloc(n + 1u, sizeof(uint32_t));
  if (alpha == NULL || in_deg == NULL) {
    free(alpha);
    free(in_deg);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  alpha[0] = 0.0;
  for (size_t i = 1u; i <= n; ++i) {
    alpha[i] = -INFINITY;
  }

  size_t total_edges = 0u;
  for (size_t pos = 0u; pos < n; ++pos) {
    const double a_pos = alpha[pos];
    if (a_pos == -INFINITY) continue;
    uint32_t node = 0u;
    for (size_t k = 0u; pos + k < n; ++k) {
      node = trie_step(&model->trie, node, bytes[pos + k]);
      if (node == UINT32_MAX) break;
      const uint32_t tok = model->trie.nodes[node].token_id;
      if (tok == UINT32_MAX) continue;
      const size_t next = pos + k + 1u;
      alpha[next] =
          log_add_exp(alpha[next], a_pos + alpha_temp * model->log_probs[tok]);
      in_deg[next] += 1u;
      ++total_edges;
    }
  }

  if (!isfinite(alpha[n])) {
    free(alpha);
    free(in_deg);
    return ET_UNIGRAM_ERR_NONFINITE;
  }

  /* Build CSR incoming-edge list in deterministic forward order. */
  uint32_t *in_off = malloc((n + 2u) * sizeof(uint32_t));
  uint32_t *in_cur = malloc((n + 1u) * sizeof(uint32_t));
  ffbs_in_edge *in_edges = malloc(total_edges * sizeof(ffbs_in_edge));
  uint32_t *rev_toks = malloc(n * sizeof(uint32_t));
  if (in_off == NULL || in_cur == NULL || in_edges == NULL ||
      rev_toks == NULL) {
    free(alpha);
    free(in_deg);
    free(in_off);
    free(in_cur);
    free(in_edges);
    free(rev_toks);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  in_off[0] = 0u;
  for (size_t i = 0u; i <= n; ++i) {
    in_off[i + 1u] = in_off[i] + in_deg[i];
    in_cur[i] = in_off[i];
  }

  for (size_t pos = 0u; pos < n; ++pos) {
    if (alpha[pos] == -INFINITY) continue;
    uint32_t node = 0u;
    for (size_t k = 0u; pos + k < n; ++k) {
      node = trie_step(&model->trie, node, bytes[pos + k]);
      if (node == UINT32_MAX) break;
      const uint32_t tok = model->trie.nodes[node].token_id;
      if (tok == UINT32_MAX) continue;
      const size_t next = pos + k + 1u;
      const uint32_t dst = in_cur[next]++;
      in_edges[dst].start = (uint32_t)pos;
      in_edges[dst].tok = tok;
    }
  }

  uint64_t rng = seed;
  size_t cur = n;
  size_t rev_len = 0u;
  double total_score = 0.0;
  while (cur > 0u) {
    const uint32_t lo = in_off[cur];
    const uint32_t hi = in_off[cur + 1u];
    const double u =
        (double)(splitmix64_next(&rng) >> 11u) * (1.0 / 9007199254740992.0);
    double cdf = 0.0;
    uint32_t chosen = hi - 1u;
    for (uint32_t e = lo; e < hi; ++e) {
      const double lw = alpha[in_edges[e].start] +
                        alpha_temp * model->log_probs[in_edges[e].tok] -
                        alpha[cur];
      cdf += exp(lw);
      if (u < cdf) {
        chosen = e;
        break;
      }
    }
    const uint32_t tok = in_edges[chosen].tok;
    rev_toks[rev_len++] = tok;
    total_score += model->log_probs[tok];
    cur = (size_t)in_edges[chosen].start;
  }

  *out_count = rev_len;
  if (out_score != NULL) *out_score = total_score;

  et_unigram_status st = ET_UNIGRAM_OK;
  if (out_tokens != NULL) {
    if (capacity < rev_len) {
      st = ET_UNIGRAM_ERR_BOUNDS;
    } else {
      for (size_t i = 0u; i < rev_len; ++i) {
        out_tokens[i] = rev_toks[rev_len - 1u - i];
      }
    }
  }

  free(alpha);
  free(in_deg);
  free(in_off);
  free(in_cur);
  free(in_edges);
  free(rev_toks);
  return st;
}

/* ---- Public API: Versioned SHA-256 Binary Serialization (ETUG1000 v1.0) ---- */

ET_UNIGRAM_API et_unigram_status et_unigram_serialize_v1(
    const et_unigram_model *model, uint8_t *out_bytes, size_t capacity,
    size_t *out_size, uint8_t *out_sha256) {
  if (model == NULL || out_size == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (out_bytes == NULL && capacity != 0u) return ET_UNIGRAM_ERR_ARGUMENT;

  const size_t pieces_blob_bytes =
      ((size_t)model->info.active_count * ET_UNIGRAM_PIECE_OVERHEAD_BYTES) +
      model->piece_blob_size;
  const size_t payload_bytes = ET_UNIGRAM_HEADER_BYTES + pieces_blob_bytes;
  const size_t total_bytes = payload_bytes + ET_UNIGRAM_SHA256_BYTES;
  *out_size = total_bytes;

  if (out_bytes == NULL) return ET_UNIGRAM_OK;
  if (capacity < total_bytes) return ET_UNIGRAM_ERR_BOUNDS;

  memcpy(out_bytes, ET_UNIGRAM_MAGIC, ET_UNIGRAM_MAGIC_LEN);
  put_le16(out_bytes + 8u, (uint16_t)ET_UNIGRAM_VERSION_MAJOR);
  put_le16(out_bytes + 10u, (uint16_t)ET_UNIGRAM_VERSION_MINOR);
  put_le32(out_bytes + 12u, 0u); /* flags */
  put_le32(out_bytes + 16u, model->info.vocab_size);
  put_le32(out_bytes + 20u, model->info.active_count);
  put_le32(out_bytes + 24u, model->info.special_mask);
  put_le32(out_bytes + 28u, model->info.bos_id);
  put_le32(out_bytes + 32u, model->info.eos_id);
  put_le32(out_bytes + 36u, model->info.pad_id);
  put_le32(out_bytes + 40u, model->info.unk_id);
  put_le32(out_bytes + 44u, (uint32_t)pieces_blob_bytes);

  size_t pos = ET_UNIGRAM_HEADER_BYTES;
  for (uint32_t i = 0u; i < model->info.active_count; ++i) {
    put_f64_le(out_bytes + pos, model->log_probs[i]);
    pos += 8u;
    put_le16(out_bytes + pos, model->piece_lens[i]);
    pos += 2u;
    const size_t plen = (size_t)model->piece_lens[i];
    memcpy(out_bytes + pos, model->piece_blob + model->piece_offsets[i], plen);
    pos += plen;
  }

  uint8_t digest[ET_UNIGRAM_SHA256_BYTES];
  sha256_compute(out_bytes, payload_bytes, digest);
  memcpy(out_bytes + payload_bytes, digest, ET_UNIGRAM_SHA256_BYTES);
  if (out_sha256 != NULL) {
    memcpy(out_sha256, digest, ET_UNIGRAM_SHA256_BYTES);
  }
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_deserialize_v1(
    const uint8_t *bytes, size_t size, et_unigram_model **out_model,
    uint8_t *out_sha256) {
  if (out_model != NULL) *out_model = NULL;
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (bytes == NULL || out_model == NULL) return ET_UNIGRAM_ERR_ARGUMENT;
  if (size < ET_UNIGRAM_HEADER_BYTES + ET_UNIGRAM_SHA256_BYTES) {
    return ET_UNIGRAM_ERR_FORMAT;
  }

  if (memcmp(bytes, ET_UNIGRAM_MAGIC, ET_UNIGRAM_MAGIC_LEN) != 0) {
    return ET_UNIGRAM_ERR_MAGIC;
  }
  const uint16_t major = get_le16(bytes + 8u);
  const uint16_t minor = get_le16(bytes + 10u);
  if (major != ET_UNIGRAM_VERSION_MAJOR || minor != ET_UNIGRAM_VERSION_MINOR) {
    return ET_UNIGRAM_ERR_VERSION;
  }

  const uint32_t flags = get_le32(bytes + 12u);
  const uint32_t vocab_size = get_le32(bytes + 16u);
  const uint32_t active_count = get_le32(bytes + 20u);
  const uint32_t special_mask = get_le32(bytes + 24u);
  const uint32_t bos_id = get_le32(bytes + 28u);
  const uint32_t eos_id = get_le32(bytes + 32u);
  const uint32_t pad_id = get_le32(bytes + 36u);
  const uint32_t unk_id = get_le32(bytes + 40u);
  const uint32_t pieces_blob_bytes = get_le32(bytes + 44u);

  if (flags != 0u) return ET_UNIGRAM_ERR_FORMAT;
  if ((size_t)pieces_blob_bytes >
      SIZE_MAX - ET_UNIGRAM_HEADER_BYTES - ET_UNIGRAM_SHA256_BYTES) {
    return ET_UNIGRAM_ERR_FORMAT;
  }
  const size_t payload_bytes =
      ET_UNIGRAM_HEADER_BYTES + (size_t)pieces_blob_bytes;
  if (size != payload_bytes + ET_UNIGRAM_SHA256_BYTES) {
    return ET_UNIGRAM_ERR_FORMAT;
  }

  uint8_t computed[ET_UNIGRAM_SHA256_BYTES];
  sha256_compute(bytes, payload_bytes, computed);
  if (!sha256_ct_equal(computed, bytes + payload_bytes)) {
    return ET_UNIGRAM_ERR_CHECKSUM;
  }

  if ((special_mask & ~(uint32_t)ET_UNIGRAM_SPECIAL_ALL) != 0u) {
    return ET_UNIGRAM_ERR_FORMAT;
  }
  if (active_count < ET_UNIGRAM_BASE_VOCAB_SIZE ||
      active_count > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_INVALID_VOCAB;
  }
  const uint32_t special_count = popcount_specials(special_mask);
  if (vocab_size != active_count + special_count ||
      vocab_size > ET_UNIGRAM_MAX_VOCAB_SIZE) {
    return ET_UNIGRAM_ERR_FORMAT;
  }

  uint32_t next_sp = active_count;
  const uint32_t exp_bos = ((special_mask & ET_UNIGRAM_SPECIAL_BOS) != 0u)
                               ? next_sp++
                               : ET_UNIGRAM_TOKEN_NONE;
  const uint32_t exp_eos = ((special_mask & ET_UNIGRAM_SPECIAL_EOS) != 0u)
                               ? next_sp++
                               : ET_UNIGRAM_TOKEN_NONE;
  const uint32_t exp_pad = ((special_mask & ET_UNIGRAM_SPECIAL_PAD) != 0u)
                               ? next_sp++
                               : ET_UNIGRAM_TOKEN_NONE;
  const uint32_t exp_unk = ((special_mask & ET_UNIGRAM_SPECIAL_UNK) != 0u)
                               ? next_sp++
                               : ET_UNIGRAM_TOKEN_NONE;
  if (bos_id != exp_bos || eos_id != exp_eos || pad_id != exp_pad ||
      unk_id != exp_unk) {
    return ET_UNIGRAM_ERR_FORMAT;
  }

  /* First pass over piece records: validate lengths, scores, and sum raw blob size. */
  size_t pos = ET_UNIGRAM_HEADER_BYTES;
  size_t raw_blob_size = 0u;
  for (uint32_t i = 0u; i < active_count; ++i) {
    if (pos + ET_UNIGRAM_PIECE_OVERHEAD_BYTES > payload_bytes) {
      return ET_UNIGRAM_ERR_FORMAT;
    }
    const double sc = get_f64_le(bytes + pos);
    const uint16_t plen = get_le16(bytes + pos + 8u);
    pos += ET_UNIGRAM_PIECE_OVERHEAD_BYTES;
    if (!isfinite(sc)) return ET_UNIGRAM_ERR_NONFINITE;
    if (sc > 0.0) return ET_UNIGRAM_ERR_INVALID_VOCAB;
    if (plen == 0u || plen > ET_UNIGRAM_MAX_PIECE_BYTES ||
        pos + (size_t)plen > payload_bytes) {
      return ET_UNIGRAM_ERR_FORMAT;
    }
    if (i < ET_UNIGRAM_BASE_VOCAB_SIZE) {
      if (plen != 1u || bytes[pos] != (uint8_t)i) {
        return ET_UNIGRAM_ERR_INVALID_VOCAB;
      }
    } else if (plen < 2u) {
      return ET_UNIGRAM_ERR_INVALID_VOCAB;
    }
    raw_blob_size += (size_t)plen;
    pos += (size_t)plen;
  }
  if (pos != payload_bytes) return ET_UNIGRAM_ERR_FORMAT;

  uint8_t *blob = malloc(raw_blob_size);
  uint32_t *offsets = malloc((size_t)active_count * sizeof(uint32_t));
  uint16_t *lens = malloc((size_t)active_count * sizeof(uint16_t));
  double *scores = malloc((size_t)active_count * sizeof(double));
  if (blob == NULL || offsets == NULL || lens == NULL || scores == NULL) {
    free(blob);
    free(offsets);
    free(lens);
    free(scores);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  pos = ET_UNIGRAM_HEADER_BYTES;
  uint32_t bcur = 0u;
  for (uint32_t i = 0u; i < active_count; ++i) {
    scores[i] = get_f64_le(bytes + pos);
    const uint16_t plen = get_le16(bytes + pos + 8u);
    pos += ET_UNIGRAM_PIECE_OVERHEAD_BYTES;
    offsets[i] = bcur;
    lens[i] = plen;
    memcpy(blob + bcur, bytes + pos, (size_t)plen);
    bcur += (uint32_t)plen;
    pos += (size_t)plen;
  }

  const et_unigram_status st = build_model_from_arrays(
      blob, raw_blob_size, offsets, lens, scores, active_count, special_mask,
      out_model);
  free(blob);
  free(offsets);
  free(lens);
  free(scores);
  if (st == ET_UNIGRAM_OK && out_sha256 != NULL) {
    memcpy(out_sha256, computed, ET_UNIGRAM_SHA256_BYTES);
  }
  return st;
}

/* ---- Atomic 0600 Save & O_NOFOLLOW Hardened Load ---- */

static int write_all(int fd, const uint8_t *bytes, size_t size) {
  while (size != 0u) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return 0;
    bytes += (size_t)written;
    size -= (size_t)written;
  }
  return 1;
}

static int sync_fd(int fd) {
  int result;
  do {
    result = fsync(fd);
  } while (result < 0 && errno == EINTR);
  return result == 0;
}

static int sync_parent_dir(const char *path, const char *slash) {
  char *dir = NULL;
  if (slash == NULL) {
    dir = malloc(2u);
    if (dir == NULL) return 0;
    dir[0] = '.';
    dir[1] = '\0';
  } else if (slash == path) {
    dir = malloc(2u);
    if (dir == NULL) return 0;
    dir[0] = '/';
    dir[1] = '\0';
  } else {
    const size_t dir_len = (size_t)(slash - path);
    dir = malloc(dir_len + 1u);
    if (dir == NULL) return 0;
    memcpy(dir, path, dir_len);
    dir[dir_len] = '\0';
  }

  const int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  free(dir);
  if (dfd < 0) return 0;
  int ok = 1;
  if (!sync_fd(dfd) && errno != EINVAL && errno != EROFS && errno != EBADF) {
    ok = 0;
  }
  if (close(dfd) != 0) ok = 0;
  return ok;
}

ET_UNIGRAM_API et_unigram_status et_unigram_save_file_v1(
    const et_unigram_model *model, const char *path, uint8_t *out_sha256) {
  if (model == NULL || path == NULL || path[0] == '\0') {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0') return ET_UNIGRAM_ERR_ARGUMENT;

  struct stat existing;
  if (lstat(path, &existing) == 0) {
    if (!S_ISREG(existing.st_mode)) return ET_UNIGRAM_ERR_IO;
  } else if (errno != ENOENT) {
    return ET_UNIGRAM_ERR_IO;
  }

  size_t artifact_size = 0u;
  et_unigram_status st =
      et_unigram_serialize_v1(model, NULL, 0u, &artifact_size, NULL);
  if (st != ET_UNIGRAM_OK) return st;

  uint8_t *buffer = malloc(artifact_size);
  if (buffer == NULL) return ET_UNIGRAM_ERR_NOMEM;

  uint8_t digest[ET_UNIGRAM_SHA256_BYTES];
  size_t written = 0u;
  st = et_unigram_serialize_v1(model, buffer, artifact_size, &written, digest);
  if (st != ET_UNIGRAM_OK) {
    free(buffer);
    return st;
  }

  static const char suffix[] = ".unigram-tok-tmp.XXXXXX";
  const size_t prefix = (slash == NULL) ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix)) {
    free(buffer);
    return ET_UNIGRAM_ERR_OVERFLOW;
  }
  char *temp_path = malloc(prefix + sizeof(suffix));
  if (temp_path == NULL) {
    free(buffer);
    return ET_UNIGRAM_ERR_NOMEM;
  }
  if (prefix != 0u) memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    free(buffer);
    return ET_UNIGRAM_ERR_IO;
  }

  int stored = write_all(fd, buffer, written) && sync_fd(fd);
  if (close(fd) != 0) stored = 0;

  if (stored && lstat(path, &existing) == 0 && !S_ISREG(existing.st_mode)) {
    stored = 0;
  }
  if (!stored || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    free(temp_path);
    free(buffer);
    return ET_UNIGRAM_ERR_IO;
  }
  if (!sync_parent_dir(path, slash)) {
    free(temp_path);
    free(buffer);
    return ET_UNIGRAM_ERR_IO;
  }

  free(temp_path);
  free(buffer);
  if (out_sha256 != NULL) {
    memcpy(out_sha256, digest, ET_UNIGRAM_SHA256_BYTES);
  }
  return ET_UNIGRAM_OK;
}

ET_UNIGRAM_API et_unigram_status et_unigram_load_file_v1(
    const char *path, et_unigram_model **out_model, uint8_t *out_sha256) {
  if (out_model != NULL) *out_model = NULL;
  if (!fp_environment_supported()) return ET_UNIGRAM_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0' || out_model == NULL) {
    return ET_UNIGRAM_ERR_ARGUMENT;
  }

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return ET_UNIGRAM_ERR_IO;

  struct stat meta;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    (void)close(fd);
    return ET_UNIGRAM_ERR_IO;
  }
  if (meta.st_size < 0 ||
      (uint64_t)meta.st_size > (uint64_t)ET_UNIGRAM_ARTIFACT_MAX_BYTES) {
    (void)close(fd);
    return ET_UNIGRAM_ERR_BOUNDS;
  }

  const size_t alloc_size = (size_t)meta.st_size + 1u;
  uint8_t *buffer = malloc(alloc_size);
  if (buffer == NULL) {
    (void)close(fd);
    return ET_UNIGRAM_ERR_NOMEM;
  }

  size_t used = 0u;
  et_unigram_status st = ET_UNIGRAM_OK;
  while (used < alloc_size) {
    const ssize_t n = read(fd, buffer + used, alloc_size - used);
    if (n == 0) break;
    if (n < 0) {
      if (errno == EINTR) continue;
      st = ET_UNIGRAM_ERR_IO;
      break;
    }
    used += (size_t)n;
  }
  if (close(fd) != 0 && st == ET_UNIGRAM_OK) {
    st = ET_UNIGRAM_ERR_IO;
  }
  if (st == ET_UNIGRAM_OK && used != (size_t)meta.st_size) {
    st = ET_UNIGRAM_ERR_FORMAT;
  }
  if (st == ET_UNIGRAM_OK) {
    st = et_unigram_deserialize_v1(buffer, used, out_model, out_sha256);
  }

  free(buffer);
  return st;
}
