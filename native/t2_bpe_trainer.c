#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "T2 BPE trainer requires 8-bit bytes and at most 64-bit size_t");

/* Standalone SHA-256 core matching native/eval5_benchmark.c. */
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

/* Alignment-safe little-endian serialization helpers (zero pointer casts). */
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

static uint16_t get_le16(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static uint32_t get_le32(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint32_t popcount_specials(uint32_t mask) {
  uint32_t count = 0u;
  if ((mask & ET_T2_BPE_SPECIAL_BOS) != 0u) ++count;
  if ((mask & ET_T2_BPE_SPECIAL_EOS) != 0u) ++count;
  if ((mask & ET_T2_BPE_SPECIAL_PAD) != 0u) ++count;
  if ((mask & ET_T2_BPE_SPECIAL_UNK) != 0u) ++count;
  return count;
}

static uint32_t pair_hash(uint32_t left, uint32_t right) {
  uint64_t x = ((uint64_t)left << 32u) | (uint64_t)right;
  x ^= x >> 33u;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33u;
  x *= UINT64_C(0xc4ceb9fe1a85ec53);
  x ^= x >> 33u;
  return (uint32_t)x;
}

typedef struct {
  uint32_t left;
  uint32_t right;
  uint32_t rank; /* UINT32_MAX when empty */
} merge_ht_slot;

struct et_t2_bpe_model {
  et_t2_bpe_info info;
  et_t2_bpe_merge *merges;
  merge_ht_slot *merge_ht;
  size_t merge_ht_mask;
  uint8_t *vocab_bytes;
  uint32_t *vocab_offsets;
  uint16_t *vocab_lengths;
};

static int merge_ht_lookup(const et_t2_bpe_model *model, uint32_t left,
                           uint32_t right, uint32_t *out_rank) {
  if (model->info.merge_count == 0u) return 0;
  size_t idx = (size_t)pair_hash(left, right) & model->merge_ht_mask;
  for (;;) {
    const merge_ht_slot *slot = &model->merge_ht[idx];
    if (slot->rank == UINT32_MAX) return 0;
    if (slot->left == left && slot->right == right) {
      *out_rank = slot->rank;
      return 1;
    }
    idx = (idx + 1u) & model->merge_ht_mask;
  }
}

/* Build a validated immutable model from special_mask and ordered merges. */
static et_t2_bpe_status build_model(uint32_t special_mask,
                                    const et_t2_bpe_merge *merges,
                                    uint32_t merge_count,
                                    et_t2_bpe_model **out_model) {
  if ((special_mask & ~(uint32_t)ET_T2_BPE_SPECIAL_ALL) != 0u) {
    return ET_T2_BPE_ERR_FORMAT;
  }
  const uint32_t special_count = popcount_specials(special_mask);
  if (merge_count > ET_T2_BPE_MAX_MERGES) {
    return ET_T2_BPE_ERR_FORMAT;
  }
  const uint32_t vocab_size =
      ET_T2_BPE_BASE_VOCAB_SIZE + merge_count + special_count;
  if (vocab_size < ET_T2_BPE_MIN_VOCAB_SIZE ||
      vocab_size > ET_T2_BPE_MAX_VOCAB_SIZE) {
    return ET_T2_BPE_ERR_FORMAT;
  }
  if (merge_count != 0u && merges == NULL) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }

  et_t2_bpe_model *model = calloc(1u, sizeof(*model));
  if (model == NULL) return ET_T2_BPE_ERR_NOMEM;

  model->info.vocab_size = vocab_size;
  model->info.merge_count = merge_count;
  model->info.special_mask = special_mask;
  model->info.bos_id = ET_T2_BPE_TOKEN_NONE;
  model->info.eos_id = ET_T2_BPE_TOKEN_NONE;
  model->info.pad_id = ET_T2_BPE_TOKEN_NONE;
  model->info.unk_id = ET_T2_BPE_TOKEN_NONE;

  uint32_t next_special = ET_T2_BPE_BASE_VOCAB_SIZE + merge_count;
  if ((special_mask & ET_T2_BPE_SPECIAL_BOS) != 0u) {
    model->info.bos_id = next_special++;
  }
  if ((special_mask & ET_T2_BPE_SPECIAL_EOS) != 0u) {
    model->info.eos_id = next_special++;
  }
  if ((special_mask & ET_T2_BPE_SPECIAL_PAD) != 0u) {
    model->info.pad_id = next_special++;
  }
  if ((special_mask & ET_T2_BPE_SPECIAL_UNK) != 0u) {
    model->info.unk_id = next_special++;
  }

  size_t ht_cap = 16u;
  while (ht_cap < (size_t)merge_count * 2u + 2u) {
    if (ht_cap > SIZE_MAX / 2u) {
      et_t2_bpe_destroy(model);
      return ET_T2_BPE_ERR_OVERFLOW;
    }
    ht_cap *= 2u;
  }

  model->vocab_offsets = calloc(vocab_size, sizeof(*model->vocab_offsets));
  model->vocab_lengths = calloc(vocab_size, sizeof(*model->vocab_lengths));
  if (model->vocab_offsets == NULL || model->vocab_lengths == NULL) {
    et_t2_bpe_destroy(model);
    return ET_T2_BPE_ERR_NOMEM;
  }

  if (merge_count != 0u) {
    model->merges = malloc((size_t)merge_count * sizeof(*model->merges));
    model->merge_ht = malloc(ht_cap * sizeof(*model->merge_ht));
    if (model->merges == NULL || model->merge_ht == NULL) {
      et_t2_bpe_destroy(model);
      return ET_T2_BPE_ERR_NOMEM;
    }
    model->merge_ht_mask = ht_cap - 1u;
    for (size_t i = 0u; i < ht_cap; ++i) {
      model->merge_ht[i].left = 0u;
      model->merge_ht[i].right = 0u;
      model->merge_ht[i].rank = UINT32_MAX;
    }
  }

  /* Pass 1: validate merge invariants, detect duplicates/cycles, size spellings. */
  size_t total_vocab_bytes = ET_T2_BPE_BASE_VOCAB_SIZE;
  for (uint32_t b = 0u; b < ET_T2_BPE_BASE_VOCAB_SIZE; ++b) {
    model->vocab_lengths[b] = 1u;
    model->vocab_offsets[b] = b;
  }

  for (uint32_t r = 0u; r < merge_count; ++r) {
    const uint32_t expected_id = ET_T2_BPE_BASE_VOCAB_SIZE + r;
    const et_t2_bpe_merge m = merges[r];
    if (m.new_id != expected_id || m.left >= expected_id ||
        m.right >= expected_id) {
      et_t2_bpe_destroy(model);
      return ET_T2_BPE_ERR_INVALID_MERGE;
    }
    const uint32_t combined_len =
        (uint32_t)model->vocab_lengths[m.left] +
        (uint32_t)model->vocab_lengths[m.right];
    if (combined_len == 0u || combined_len > ET_T2_BPE_MAX_TOKEN_BYTES) {
      et_t2_bpe_destroy(model);
      return ET_T2_BPE_ERR_INVALID_MERGE;
    }
    /* Insert into pair->rank hash table while rejecting duplicate pairs. */
    size_t idx = (size_t)pair_hash(m.left, m.right) & model->merge_ht_mask;
    for (;;) {
      merge_ht_slot *slot = &model->merge_ht[idx];
      if (slot->rank == UINT32_MAX) {
        slot->left = m.left;
        slot->right = m.right;
        slot->rank = r;
        break;
      }
      if (slot->left == m.left && slot->right == m.right) {
        et_t2_bpe_destroy(model);
        return ET_T2_BPE_ERR_INVALID_MERGE;
      }
      idx = (idx + 1u) & model->merge_ht_mask;
    }
    if ((size_t)combined_len > UINT32_MAX - total_vocab_bytes) {
      et_t2_bpe_destroy(model);
      return ET_T2_BPE_ERR_OVERFLOW;
    }
    model->merges[r] = m;
    model->vocab_offsets[expected_id] = (uint32_t)total_vocab_bytes;
    model->vocab_lengths[expected_id] = (uint16_t)combined_len;
    total_vocab_bytes += (size_t)combined_len;
  }

  /* Special tokens have 0 decoded raw bytes. */
  for (uint32_t s = ET_T2_BPE_BASE_VOCAB_SIZE + merge_count; s < vocab_size;
       ++s) {
    model->vocab_offsets[s] = (uint32_t)total_vocab_bytes;
    model->vocab_lengths[s] = 0u;
  }

  /* Pass 2: materialize flat non-recursive decoded byte spelling table. */
  model->vocab_bytes = malloc(total_vocab_bytes);
  if (model->vocab_bytes == NULL) {
    et_t2_bpe_destroy(model);
    return ET_T2_BPE_ERR_NOMEM;
  }
  for (uint32_t b = 0u; b < ET_T2_BPE_BASE_VOCAB_SIZE; ++b) {
    model->vocab_bytes[b] = (uint8_t)b;
  }
  for (uint32_t r = 0u; r < merge_count; ++r) {
    const et_t2_bpe_merge m = model->merges[r];
    const uint32_t dst_off = model->vocab_offsets[m.new_id];
    const uint16_t l_len = model->vocab_lengths[m.left];
    const uint16_t r_len = model->vocab_lengths[m.right];
    memcpy(model->vocab_bytes + dst_off,
           model->vocab_bytes + model->vocab_offsets[m.left], l_len);
    memcpy(model->vocab_bytes + dst_off + l_len,
           model->vocab_bytes + model->vocab_offsets[m.right], r_len);
  }

  *out_model = model;
  return ET_T2_BPE_OK;
}

/* Generation-stamped pair counter hash table for deterministic training. */
typedef struct {
  uint32_t left;
  uint32_t right;
  uint32_t gen;
  uint64_t count;
} pair_count_slot;

typedef struct {
  pair_count_slot *slots;
  size_t *active_indices;
  size_t capacity;
  size_t active_count;
  uint32_t gen;
} pair_counter;

static et_t2_bpe_status pair_counter_init(pair_counter *pc, size_t initial_cap) {
  pc->slots = calloc(initial_cap, sizeof(*pc->slots));
  pc->active_indices = malloc(initial_cap * sizeof(*pc->active_indices));
  if (pc->slots == NULL || pc->active_indices == NULL) {
    free(pc->slots);
    free(pc->active_indices);
    pc->slots = NULL;
    pc->active_indices = NULL;
    return ET_T2_BPE_ERR_NOMEM;
  }
  pc->capacity = initial_cap;
  pc->active_count = 0u;
  pc->gen = 0u;
  return ET_T2_BPE_OK;
}

static void pair_counter_destroy(pair_counter *pc) {
  free(pc->slots);
  free(pc->active_indices);
  pc->slots = NULL;
  pc->active_indices = NULL;
  pc->capacity = 0u;
  pc->active_count = 0u;
  pc->gen = 0u;
}

static void pair_counter_reset(pair_counter *pc) {
  pc->active_count = 0u;
  if (pc->gen == UINT32_MAX) {
    for (size_t i = 0u; i < pc->capacity; ++i) {
      pc->slots[i].gen = 0u;
    }
    pc->gen = 1u;
  } else {
    pc->gen += 1u;
  }
}

static et_t2_bpe_status pair_counter_grow(pair_counter *pc) {
  if (pc->capacity > SIZE_MAX / 2u) return ET_T2_BPE_ERR_OVERFLOW;
  const size_t new_cap = pc->capacity * 2u;
  if (new_cap > (size_t)PTRDIFF_MAX / sizeof(*pc->slots) ||
      new_cap > (size_t)PTRDIFF_MAX / sizeof(*pc->active_indices)) {
    return ET_T2_BPE_ERR_OVERFLOW;
  }
  pair_count_slot *new_slots = calloc(new_cap, sizeof(*new_slots));
  size_t *new_active = malloc(new_cap * sizeof(*new_active));
  if (new_slots == NULL || new_active == NULL) {
    free(new_slots);
    free(new_active);
    return ET_T2_BPE_ERR_NOMEM;
  }
  const size_t mask = new_cap - 1u;
  size_t new_active_count = 0u;
  for (size_t k = 0u; k < pc->active_count; ++k) {
    const pair_count_slot entry = pc->slots[pc->active_indices[k]];
    size_t idx = (size_t)pair_hash(entry.left, entry.right) & mask;
    while (new_slots[idx].gen == pc->gen) {
      idx = (idx + 1u) & mask;
    }
    new_slots[idx] = entry;
    new_active[new_active_count++] = idx;
  }
  free(pc->slots);
  free(pc->active_indices);
  pc->slots = new_slots;
  pc->active_indices = new_active;
  pc->capacity = new_cap;
  pc->active_count = new_active_count;
  return ET_T2_BPE_OK;
}

static et_t2_bpe_status pair_counter_add(pair_counter *pc, uint32_t left,
                                         uint32_t right) {
  if ((pc->active_count + 1u) * 2u > pc->capacity) {
    const et_t2_bpe_status st = pair_counter_grow(pc);
    if (st != ET_T2_BPE_OK) return st;
  }
  const size_t mask = pc->capacity - 1u;
  size_t idx = (size_t)pair_hash(left, right) & mask;
  for (;;) {
    pair_count_slot *slot = &pc->slots[idx];
    if (slot->gen != pc->gen) {
      slot->left = left;
      slot->right = right;
      slot->gen = pc->gen;
      slot->count = 1u;
      pc->active_indices[pc->active_count++] = idx;
      return ET_T2_BPE_OK;
    }
    if (slot->left == left && slot->right == right) {
      if (slot->count == UINT64_MAX) return ET_T2_BPE_ERR_OVERFLOW;
      slot->count += 1u;
      return ET_T2_BPE_OK;
    }
    idx = (idx + 1u) & mask;
  }
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_train(
    const et_t2_bpe_document *docs, size_t doc_count,
    const et_t2_bpe_train_config *config, et_t2_bpe_model **out_model) {
  if (config == NULL || out_model == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (doc_count != 0u && docs == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if ((config->special_mask & ~(uint32_t)ET_T2_BPE_SPECIAL_ALL) != 0u) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (config->min_frequency == 0u) return ET_T2_BPE_ERR_ARGUMENT;

  const uint32_t special_count = popcount_specials(config->special_mask);
  const uint32_t min_vocab = ET_T2_BPE_BASE_VOCAB_SIZE + special_count;
  if (config->target_vocab_size < min_vocab ||
      config->target_vocab_size > ET_T2_BPE_MAX_VOCAB_SIZE) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }

  size_t total_bytes = 0u;
  for (size_t d = 0u; d < doc_count; ++d) {
    if (docs[d].byte_count != 0u && docs[d].bytes == NULL) {
      return ET_T2_BPE_ERR_ARGUMENT;
    }
    if (docs[d].byte_count > (size_t)PTRDIFF_MAX / sizeof(uint32_t) ||
        docs[d].byte_count >
            ((size_t)PTRDIFF_MAX / sizeof(uint32_t)) - total_bytes) {
      return ET_T2_BPE_ERR_OVERFLOW;
    }
    total_bytes += docs[d].byte_count;
  }

  const uint32_t max_merges = config->target_vocab_size - min_vocab;
  if (max_merges == 0u || total_bytes < 2u) {
    return build_model(config->special_mask, NULL, 0u, out_model);
  }

  if (doc_count > (size_t)PTRDIFF_MAX / sizeof(size_t)) {
    return ET_T2_BPE_ERR_OVERFLOW;
  }

  uint32_t *seq = malloc(total_bytes * sizeof(*seq));
  size_t *doc_starts = malloc(doc_count * sizeof(*doc_starts));
  size_t *doc_lens = malloc(doc_count * sizeof(*doc_lens));
  et_t2_bpe_merge *merges = malloc((size_t)max_merges * sizeof(*merges));
  uint16_t *tok_lens =
      calloc(ET_T2_BPE_BASE_VOCAB_SIZE + max_merges, sizeof(*tok_lens));
  pair_counter pc = {0};

  if (seq == NULL || doc_starts == NULL || doc_lens == NULL ||
      merges == NULL || tok_lens == NULL ||
      pair_counter_init(&pc, 1024u) != ET_T2_BPE_OK) {
    pair_counter_destroy(&pc);
    free(tok_lens);
    free(merges);
    free(doc_lens);
    free(doc_starts);
    free(seq);
    return ET_T2_BPE_ERR_NOMEM;
  }

  for (uint32_t b = 0u; b < ET_T2_BPE_BASE_VOCAB_SIZE; ++b) {
    tok_lens[b] = 1u;
  }

  size_t cursor = 0u;
  for (size_t d = 0u; d < doc_count; ++d) {
    doc_starts[d] = cursor;
    doc_lens[d] = docs[d].byte_count;
    for (size_t i = 0u; i < docs[d].byte_count; ++i) {
      seq[cursor + i] = (uint32_t)docs[d].bytes[i];
    }
    cursor += docs[d].byte_count;
  }

  et_t2_bpe_status status = ET_T2_BPE_OK;
  uint32_t learned = 0u;

  while (learned < max_merges) {
    pair_counter_reset(&pc);
    for (size_t d = 0u; d < doc_count; ++d) {
      const size_t len = doc_lens[d];
      if (len < 2u) continue;
      const uint32_t *s = seq + doc_starts[d];
      for (size_t i = 0u; i + 1u < len; ++i) {
        status = pair_counter_add(&pc, s[i], s[i + 1u]);
        if (status != ET_T2_BPE_OK) break;
      }
      if (status != ET_T2_BPE_OK) break;
    }
    if (status != ET_T2_BPE_OK) break;

    int found = 0;
    uint64_t best_count = 0u;
    uint32_t best_left = 0u;
    uint32_t best_right = 0u;

    for (size_t k = 0u; k < pc.active_count; ++k) {
      const pair_count_slot entry = pc.slots[pc.active_indices[k]];
      if (entry.count < (uint64_t)config->min_frequency) continue;
      const uint32_t combined_len =
          (uint32_t)tok_lens[entry.left] + (uint32_t)tok_lens[entry.right];
      if (combined_len > ET_T2_BPE_MAX_TOKEN_BYTES) continue;

      if (!found || entry.count > best_count ||
          (entry.count == best_count &&
           (entry.left < best_left ||
            (entry.left == best_left && entry.right < best_right)))) {
        found = 1;
        best_count = entry.count;
        best_left = entry.left;
        best_right = entry.right;
      }
    }

    if (!found) break;

    const uint32_t new_id = ET_T2_BPE_BASE_VOCAB_SIZE + learned;
    merges[learned].left = best_left;
    merges[learned].right = best_right;
    merges[learned].new_id = new_id;
    tok_lens[new_id] =
        (uint16_t)(tok_lens[best_left] + tok_lens[best_right]);
    learned += 1u;

    /* Replace all non-overlapping occurrences left-to-right within each doc. */
    for (size_t d = 0u; d < doc_count; ++d) {
      const size_t len = doc_lens[d];
      if (len < 2u) continue;
      uint32_t *s = seq + doc_starts[d];
      size_t r = 0u;
      size_t w = 0u;
      while (r < len) {
        if (r + 1u < len && s[r] == best_left && s[r + 1u] == best_right) {
          s[w++] = new_id;
          r += 2u;
        } else {
          s[w++] = s[r++];
        }
      }
      doc_lens[d] = w;
    }
  }

  if (status == ET_T2_BPE_OK) {
    status = build_model(config->special_mask, merges, learned, out_model);
  }

  pair_counter_destroy(&pc);
  free(tok_lens);
  free(merges);
  free(doc_lens);
  free(doc_starts);
  free(seq);
  return status;
}

ET_T2_BPE_API void et_t2_bpe_destroy(et_t2_bpe_model *model) {
  if (model == NULL) return;
  free(model->merges);
  free(model->merge_ht);
  free(model->vocab_bytes);
  free(model->vocab_offsets);
  free(model->vocab_lengths);
  free(model);
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_get_info(
    const et_t2_bpe_model *model, et_t2_bpe_info *out_info) {
  if (model == NULL || out_info == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  *out_info = model->info;
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_get_merge(
    const et_t2_bpe_model *model, uint32_t rank, et_t2_bpe_merge *out_merge) {
  if (model == NULL || out_merge == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (rank >= model->info.merge_count) return ET_T2_BPE_ERR_BOUNDS;
  *out_merge = model->merges[rank];
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_token_bytes(
    const et_t2_bpe_model *model, uint32_t token_id, uint8_t *out_bytes,
    size_t capacity, size_t *out_len) {
  if (model == NULL || out_len == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (out_bytes == NULL && capacity != 0u) return ET_T2_BPE_ERR_ARGUMENT;
  if (token_id >= model->info.vocab_size) return ET_T2_BPE_ERR_TOKEN_ID;

  const size_t len = (size_t)model->vocab_lengths[token_id];
  *out_len = len;
  if (out_bytes == NULL) return ET_T2_BPE_OK;
  if (capacity < len) return ET_T2_BPE_ERR_BOUNDS;
  if (len != 0u) {
    memcpy(out_bytes, model->vocab_bytes + model->vocab_offsets[token_id], len);
  }
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_encode_ex(
    const et_t2_bpe_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t add_bos, uint32_t add_eos, uint32_t *out_tokens, size_t capacity,
    size_t *out_count) {
  if (model == NULL || out_count == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (byte_count != 0u && bytes == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (out_tokens == NULL && capacity != 0u) return ET_T2_BPE_ERR_ARGUMENT;
  if (add_bos > 1u || add_eos > 1u) return ET_T2_BPE_ERR_ARGUMENT;
  if (add_bos != 0u && model->info.bos_id == ET_T2_BPE_TOKEN_NONE) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (add_eos != 0u && model->info.eos_id == ET_T2_BPE_TOKEN_NONE) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (byte_count > (size_t)PTRDIFF_MAX / sizeof(uint32_t)) {
    return ET_T2_BPE_ERR_OVERFLOW;
  }

  uint32_t *work = NULL;
  size_t len = byte_count;
  if (byte_count != 0u) {
    work = malloc(byte_count * sizeof(*work));
    if (work == NULL) return ET_T2_BPE_ERR_NOMEM;
    for (size_t i = 0u; i < byte_count; ++i) {
      work[i] = (uint32_t)bytes[i];
    }

    /* Repeatedly apply the lowest-rank applicable merge left-to-right without
     * overlap. Because every merge produces a strictly larger ID than its
     * operands, this is mathematically identical to sweeping ranks 0..M-1. */
    if (model->info.merge_count != 0u) {
      while (len >= 2u) {
        uint32_t best_rank = UINT32_MAX;
        for (size_t i = 0u; i + 1u < len; ++i) {
          uint32_t rank = UINT32_MAX;
          if (merge_ht_lookup(model, work[i], work[i + 1u], &rank) &&
              rank < best_rank) {
            best_rank = rank;
          }
        }
        if (best_rank == UINT32_MAX) break;
        const et_t2_bpe_merge m = model->merges[best_rank];
        size_t r = 0u;
        size_t w = 0u;
        while (r < len) {
          if (r + 1u < len && work[r] == m.left && work[r + 1u] == m.right) {
            work[w++] = m.new_id;
            r += 2u;
          } else {
            work[w++] = work[r++];
          }
        }
        len = w;
      }
    }
  }

  const size_t extra = (size_t)add_bos + (size_t)add_eos;
  if (extra > SIZE_MAX - len) {
    free(work);
    return ET_T2_BPE_ERR_OVERFLOW;
  }
  const size_t total = len + extra;
  *out_count = total;

  if (out_tokens == NULL) {
    free(work);
    return ET_T2_BPE_OK;
  }
  if (capacity < total) {
    free(work);
    return ET_T2_BPE_ERR_BOUNDS;
  }

  size_t pos = 0u;
  if (add_bos != 0u) {
    out_tokens[pos++] = model->info.bos_id;
  }
  if (len != 0u) {
    memcpy(out_tokens + pos, work, len * sizeof(*work));
    pos += len;
  }
  if (add_eos != 0u) {
    out_tokens[pos++] = model->info.eos_id;
  }
  free(work);
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_encode(
    const et_t2_bpe_model *model, const uint8_t *bytes, size_t byte_count,
    uint32_t *out_tokens, size_t capacity, size_t *out_count) {
  return et_t2_bpe_encode_ex(model, bytes, byte_count, 0u, 0u, out_tokens,
                             capacity, out_count);
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_decode(
    const et_t2_bpe_model *model, const uint32_t *tokens, size_t token_count,
    uint8_t *out_bytes, size_t capacity, size_t *out_byte_count) {
  if (model == NULL || out_byte_count == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (token_count != 0u && tokens == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (out_bytes == NULL && capacity != 0u) return ET_T2_BPE_ERR_ARGUMENT;
  if (token_count > SIZE_MAX / sizeof(*tokens)) return ET_T2_BPE_ERR_OVERFLOW;

  size_t total = 0u;
  for (size_t i = 0u; i < token_count; ++i) {
    const uint32_t tid = tokens[i];
    if (tid >= model->info.vocab_size) return ET_T2_BPE_ERR_TOKEN_ID;
    const size_t tlen = (size_t)model->vocab_lengths[tid];
    if (tlen > SIZE_MAX - total) return ET_T2_BPE_ERR_OVERFLOW;
    total += tlen;
  }

  const size_t token_bytes = token_count * sizeof(*tokens);
  const uintptr_t token_addr = (uintptr_t)tokens;
  const uintptr_t count_addr = (uintptr_t)out_byte_count;
  if (token_bytes != 0u &&
      ((count_addr >= token_addr && count_addr - token_addr < token_bytes) ||
       (token_addr > count_addr && token_addr - count_addr < sizeof(*out_byte_count)))) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  if (out_bytes != NULL && capacity >= total && total != 0u) {
    const uintptr_t output_addr = (uintptr_t)out_bytes;
    if (token_bytes != 0u &&
        ((output_addr >= token_addr && output_addr - token_addr < token_bytes) ||
         (token_addr > output_addr && token_addr - output_addr < total))) {
      return ET_T2_BPE_ERR_ARGUMENT;
    }
    if ((count_addr >= output_addr && count_addr - output_addr < total) ||
        (output_addr > count_addr && output_addr - count_addr < sizeof(*out_byte_count))) {
      return ET_T2_BPE_ERR_ARGUMENT;
    }
  }

  *out_byte_count = total;
  if (out_bytes == NULL) return ET_T2_BPE_OK;
  if (capacity < total) return ET_T2_BPE_ERR_BOUNDS;

  size_t pos = 0u;
  for (size_t i = 0u; i < token_count; ++i) {
    const uint32_t tid = tokens[i];
    const size_t tlen = (size_t)model->vocab_lengths[tid];
    if (tlen != 0u) {
      memcpy(out_bytes + pos, model->vocab_bytes + model->vocab_offsets[tid],
             tlen);
      pos += tlen;
    }
  }
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_serialize(
    const et_t2_bpe_model *model, uint8_t *out_bytes, size_t capacity,
    size_t *out_size, uint8_t *out_sha256) {
  if (model == NULL || out_size == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (out_bytes == NULL && capacity != 0u) return ET_T2_BPE_ERR_ARGUMENT;

  const size_t payload_size =
      (size_t)ET_T2_BPE_HEADER_BYTES +
      ((size_t)model->info.merge_count * (size_t)ET_T2_BPE_MERGE_RECORD_BYTES);
  const size_t total_size = payload_size + (size_t)ET_T2_BPE_SHA256_BYTES;
  *out_size = total_size;

  if (out_bytes == NULL) {
    if (out_sha256 != NULL) {
      uint8_t *tmp = malloc(total_size);
      if (tmp == NULL) return ET_T2_BPE_ERR_NOMEM;
      size_t written = 0u;
      const et_t2_bpe_status st =
          et_t2_bpe_serialize(model, tmp, total_size, &written, out_sha256);
      free(tmp);
      return st;
    }
    return ET_T2_BPE_OK;
  }
  if (capacity < total_size) return ET_T2_BPE_ERR_BOUNDS;

  memcpy(out_bytes, ET_T2_BPE_MAGIC, ET_T2_BPE_MAGIC_LEN);
  put_le16(out_bytes + 8u, (uint16_t)ET_T2_BPE_VERSION_MAJOR);
  put_le16(out_bytes + 10u, (uint16_t)ET_T2_BPE_VERSION_MINOR);
  put_le32(out_bytes + 12u, 0u); /* flags */
  put_le32(out_bytes + 16u, model->info.vocab_size);
  put_le32(out_bytes + 20u, model->info.merge_count);
  put_le32(out_bytes + 24u, model->info.special_mask);
  put_le32(out_bytes + 28u, model->info.bos_id);
  put_le32(out_bytes + 32u, model->info.eos_id);
  put_le32(out_bytes + 36u, model->info.pad_id);
  put_le32(out_bytes + 40u, model->info.unk_id);

  size_t off = ET_T2_BPE_HEADER_BYTES;
  for (uint32_t r = 0u; r < model->info.merge_count; ++r) {
    put_le32(out_bytes + off, model->merges[r].left);
    put_le32(out_bytes + off + 4u, model->merges[r].right);
    put_le32(out_bytes + off + 8u, model->merges[r].new_id);
    off += ET_T2_BPE_MERGE_RECORD_BYTES;
  }

  uint8_t digest[ET_T2_BPE_SHA256_BYTES];
  sha256_compute(out_bytes, payload_size, digest);
  memcpy(out_bytes + payload_size, digest, ET_T2_BPE_SHA256_BYTES);
  if (out_sha256 != NULL) {
    memcpy(out_sha256, digest, ET_T2_BPE_SHA256_BYTES);
  }
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_deserialize(
    const uint8_t *bytes, size_t size, et_t2_bpe_model **out_model,
    uint8_t *out_sha256) {
  if (bytes == NULL || out_model == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  if (size > ET_T2_BPE_ARTIFACT_MAX_BYTES) return ET_T2_BPE_ERR_BOUNDS;
  if (size < 12u) return ET_T2_BPE_ERR_FORMAT;

  if (memcmp(bytes, ET_T2_BPE_MAGIC, ET_T2_BPE_MAGIC_LEN) != 0) {
    return ET_T2_BPE_ERR_MAGIC;
  }
  const uint16_t ver_major = get_le16(bytes + 8u);
  const uint16_t ver_minor = get_le16(bytes + 10u);
  if (ver_major != (uint16_t)ET_T2_BPE_VERSION_MAJOR ||
      ver_minor != (uint16_t)ET_T2_BPE_VERSION_MINOR) {
    return ET_T2_BPE_ERR_VERSION;
  }
  if (size < ET_T2_BPE_ARTIFACT_MIN_BYTES) return ET_T2_BPE_ERR_FORMAT;

  const uint32_t merge_count = get_le32(bytes + 20u);
  if (merge_count <= ET_T2_BPE_MAX_MERGES) {
    const size_t expected_size =
        (size_t)ET_T2_BPE_HEADER_BYTES +
        ((size_t)merge_count * (size_t)ET_T2_BPE_MERGE_RECORD_BYTES) +
        (size_t)ET_T2_BPE_SHA256_BYTES;
    if (size != expected_size) return ET_T2_BPE_ERR_FORMAT;
  }

  const size_t payload_size = size - ET_T2_BPE_SHA256_BYTES;
  uint8_t computed[ET_T2_BPE_SHA256_BYTES];
  sha256_compute(bytes, payload_size, computed);
  if (memcmp(computed, bytes + payload_size, ET_T2_BPE_SHA256_BYTES) != 0) {
    return ET_T2_BPE_ERR_CHECKSUM;
  }

  /* Checksum verified; now validate every header & semantic invariant. */
  const uint32_t flags = get_le32(bytes + 12u);
  const uint32_t vocab_size = get_le32(bytes + 16u);
  const uint32_t special_mask = get_le32(bytes + 24u);
  const uint32_t bos_id = get_le32(bytes + 28u);
  const uint32_t eos_id = get_le32(bytes + 32u);
  const uint32_t pad_id = get_le32(bytes + 36u);
  const uint32_t unk_id = get_le32(bytes + 40u);

  if (flags != 0u) return ET_T2_BPE_ERR_FORMAT;
  if (merge_count > ET_T2_BPE_MAX_MERGES) return ET_T2_BPE_ERR_FORMAT;
  const size_t exact_size =
      (size_t)ET_T2_BPE_HEADER_BYTES +
      ((size_t)merge_count * (size_t)ET_T2_BPE_MERGE_RECORD_BYTES) +
      (size_t)ET_T2_BPE_SHA256_BYTES;
  if (size != exact_size) return ET_T2_BPE_ERR_FORMAT;
  if ((special_mask & ~(uint32_t)ET_T2_BPE_SPECIAL_ALL) != 0u) {
    return ET_T2_BPE_ERR_FORMAT;
  }
  const uint32_t special_count = popcount_specials(special_mask);
  const uint32_t expected_vocab =
      ET_T2_BPE_BASE_VOCAB_SIZE + merge_count + special_count;
  if (vocab_size != expected_vocab || vocab_size < ET_T2_BPE_MIN_VOCAB_SIZE ||
      vocab_size > ET_T2_BPE_MAX_VOCAB_SIZE) {
    return ET_T2_BPE_ERR_FORMAT;
  }

  /* Verify canonical special token IDs match special_mask exactly. */
  uint32_t next_special = ET_T2_BPE_BASE_VOCAB_SIZE + merge_count;
  const uint32_t expected_bos =
      (special_mask & ET_T2_BPE_SPECIAL_BOS) != 0u ? next_special++ : ET_T2_BPE_TOKEN_NONE;
  const uint32_t expected_eos =
      (special_mask & ET_T2_BPE_SPECIAL_EOS) != 0u ? next_special++ : ET_T2_BPE_TOKEN_NONE;
  const uint32_t expected_pad =
      (special_mask & ET_T2_BPE_SPECIAL_PAD) != 0u ? next_special++ : ET_T2_BPE_TOKEN_NONE;
  const uint32_t expected_unk =
      (special_mask & ET_T2_BPE_SPECIAL_UNK) != 0u ? next_special++ : ET_T2_BPE_TOKEN_NONE;
  if (bos_id != expected_bos || eos_id != expected_eos ||
      pad_id != expected_pad || unk_id != expected_unk) {
    return ET_T2_BPE_ERR_FORMAT;
  }

  et_t2_bpe_merge *merges = NULL;
  if (merge_count != 0u) {
    merges = malloc((size_t)merge_count * sizeof(*merges));
    if (merges == NULL) return ET_T2_BPE_ERR_NOMEM;
    size_t off = ET_T2_BPE_HEADER_BYTES;
    for (uint32_t r = 0u; r < merge_count; ++r) {
      merges[r].left = get_le32(bytes + off);
      merges[r].right = get_le32(bytes + off + 4u);
      merges[r].new_id = get_le32(bytes + off + 8u);
      off += ET_T2_BPE_MERGE_RECORD_BYTES;
    }
  }

  const et_t2_bpe_status st =
      build_model(special_mask, merges, merge_count, out_model);
  free(merges);
  if (st != ET_T2_BPE_OK) return st;

  if (out_sha256 != NULL) {
    memcpy(out_sha256, computed, ET_T2_BPE_SHA256_BYTES);
  }
  return ET_T2_BPE_OK;
}

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
  if (!sync_fd(dfd)) {
    ok = 0;
  }
  if (close(dfd) != 0) ok = 0;
  return ok;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_save_file(
    const et_t2_bpe_model *model, const char *path, uint8_t *out_sha256) {
  if (model == NULL || path == NULL || path[0] == '\0') {
    return ET_T2_BPE_ERR_ARGUMENT;
  }
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0') return ET_T2_BPE_ERR_ARGUMENT;

  /* Refuse to overwrite an existing symlink or non-regular file. */
  struct stat existing;
  if (lstat(path, &existing) == 0) {
    if (!S_ISREG(existing.st_mode)) return ET_T2_BPE_ERR_IO;
  } else if (errno != ENOENT) {
    return ET_T2_BPE_ERR_IO;
  }

  size_t artifact_size = 0u;
  et_t2_bpe_status st =
      et_t2_bpe_serialize(model, NULL, 0u, &artifact_size, NULL);
  if (st != ET_T2_BPE_OK) return st;

  uint8_t *buffer = malloc(artifact_size);
  if (buffer == NULL) return ET_T2_BPE_ERR_NOMEM;

  uint8_t digest[ET_T2_BPE_SHA256_BYTES];
  size_t written = 0u;
  st = et_t2_bpe_serialize(model, buffer, artifact_size, &written, digest);
  if (st != ET_T2_BPE_OK) {
    free(buffer);
    return st;
  }

  static const char suffix[] = ".t2-bpe-tmp.XXXXXX";
  const size_t prefix = (slash == NULL) ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix)) {
    free(buffer);
    return ET_T2_BPE_ERR_OVERFLOW;
  }
  char *temp_path = malloc(prefix + sizeof(suffix));
  if (temp_path == NULL) {
    free(buffer);
    return ET_T2_BPE_ERR_NOMEM;
  }
  if (prefix != 0u) memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    free(buffer);
    return ET_T2_BPE_ERR_IO;
  }

  int stored = write_all(fd, buffer, written) && sync_fd(fd);
  if (close(fd) != 0) stored = 0;

  /* Re-check destination immediately before rename to narrow symlink race. */
  if (stored && lstat(path, &existing) == 0 && !S_ISREG(existing.st_mode)) {
    stored = 0;
  }
  if (!stored || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    free(temp_path);
    free(buffer);
    return ET_T2_BPE_ERR_IO;
  }
  if (!sync_parent_dir(path, slash)) {
    free(temp_path);
    free(buffer);
    return ET_T2_BPE_ERR_PUBLISHED_UNSYNCED;
  }

  free(temp_path);
  free(buffer);
  if (out_sha256 != NULL) {
    memcpy(out_sha256, digest, ET_T2_BPE_SHA256_BYTES);
  }
  return ET_T2_BPE_OK;
}

ET_T2_BPE_API et_t2_bpe_status et_t2_bpe_load_file(
    const char *path, et_t2_bpe_model **out_model, uint8_t *out_sha256) {
  if (path == NULL || path[0] == '\0' || out_model == NULL) {
    return ET_T2_BPE_ERR_ARGUMENT;
  }

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return ET_T2_BPE_ERR_IO;

  struct stat meta;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    (void)close(fd);
    return ET_T2_BPE_ERR_IO;
  }
  if (meta.st_size < 0 ||
      (uint64_t)meta.st_size > (uint64_t)ET_T2_BPE_ARTIFACT_MAX_BYTES) {
    (void)close(fd);
    return ET_T2_BPE_ERR_BOUNDS;
  }

  const size_t alloc_size = (size_t)meta.st_size + 1u;
  uint8_t *buffer = malloc(alloc_size);
  if (buffer == NULL) {
    (void)close(fd);
    return ET_T2_BPE_ERR_NOMEM;
  }

  size_t used = 0u;
  et_t2_bpe_status st = ET_T2_BPE_OK;
  while (used < alloc_size) {
    const ssize_t n = read(fd, buffer + used, alloc_size - used);
    if (n == 0) break;
    if (n < 0) {
      if (errno == EINTR) continue;
      st = ET_T2_BPE_ERR_IO;
      break;
    }
    used += (size_t)n;
  }
  if (close(fd) != 0 && st == ET_T2_BPE_OK) {
    st = ET_T2_BPE_ERR_IO;
  }
  if (st == ET_T2_BPE_OK && used != (size_t)meta.st_size) {
    st = ET_T2_BPE_ERR_FORMAT;
  }
  if (st == ET_T2_BPE_OK) {
    st = et_t2_bpe_deserialize(buffer, used, out_model, out_sha256);
  }

  free(buffer);
  return st;
}
