#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/paged_kv_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Paged KV manager requires 8-bit bytes and binary32 floats");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "Paged KV manager requires IEEE-754 binary32 evaluation");

/* ---- Standalone SHA-256 for canonical prefix-block hashing ---- */

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
  a = state->h[0];
  b = state->h[1];
  c = state->h[2];
  d = state->h[3];
  e = state->h[4];
  f = state->h[5];
  g = state->h[6];
  h = state->h[7];
  for (i = 0; i < 64u; ++i) {
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
  for (i = 0; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void encode_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

/* ---- FP environment, alignment, span, and validation helpers ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, mask all exceptions, and disable FTZ/DAZ. */
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

typedef struct span_entry {
  const void *ptr;
  size_t bytes;
} span_entry;

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int any_spans_overlap(const span_entry *entries, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(entries[i].ptr, entries[i].bytes, entries[j].ptr,
                        entries[j].bytes))
        return 1;
    }
  }
  return 0;
}

static int mul_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
  if (a != 0u && b > UINT64_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static int all_finite_f32(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

/* ---- Internal Pool & Sequence Structures ---- */

typedef struct paged_kv_seq {
  uint32_t alive;
  uint32_t seq_len;
  uint32_t num_logical_blocks;
  uint32_t *block_table; /* length: cfg.max_blocks_per_seq */
} paged_kv_seq;

typedef struct prefix_cache_entry {
  uint8_t occupied;
  uint32_t block_id;
  uint8_t hash[ET_PAGED_KV_HASH_BYTES];
} prefix_cache_entry;

struct et_paged_kv_pool {
  et_paged_kv_config cfg;
  size_t pool_floats;
  size_t pool_bytes;
  size_t floats_per_block;
  size_t floats_per_layer;
  size_t floats_per_kv_side;
  size_t floats_per_head;
  size_t floats_per_token_all_layers;

  int owns_kv_data;
  float *kv_data; /* [num_blocks, num_layers, 2, num_kv_heads, block_size,
                     head_dim] */

  uint32_t *ref_count;          /* [num_blocks] */
  uint8_t *block_prefix_active; /* [num_blocks]: 1 if indexed in prefix cache */
  uint8_t *block_prefix_hashes; /* [num_blocks * ET_PAGED_KV_HASH_BYTES] */

  uint32_t *free_stack; /* deterministic LIFO stack of free physical blocks */
  uint32_t free_count;

  paged_kv_seq *seqs;      /* [max_seqs] */
  uint32_t *block_tables;  /* contiguous backing [max_seqs * max_blocks_per_seq]
                            */
  uint32_t active_seqs;

  /* Open-addressing hash table with backward-shift deletion (no tombstones). */
  prefix_cache_entry *prefix_table;
  uint32_t prefix_table_cap; /* power of 2 >= 2 * num_blocks */
  uint32_t prefix_cached_count;
  int prefix_context_bound;
  et_paged_kv_prefix_context_v1 prefix_context;

  uint64_t cow_copies;
  uint64_t prefix_cache_hits;
};

/* Public outputs may not overwrite either caller-owned KV storage or the
 * manager's allocations, including its block tables and refcounts. */
static int output_overlaps_pool(const et_paged_kv_pool *pool, const void *out,
                                size_t bytes) {
  const span_entry owned[] = {
      {pool, sizeof(*pool)},
      {pool->kv_data, pool->pool_bytes},
      {pool->ref_count, (size_t)pool->cfg.num_blocks * sizeof(uint32_t)},
      {pool->block_prefix_active, (size_t)pool->cfg.num_blocks},
      {pool->block_prefix_hashes,
       (size_t)pool->cfg.num_blocks * ET_PAGED_KV_HASH_BYTES},
      {pool->free_stack, (size_t)pool->cfg.num_blocks * sizeof(uint32_t)},
      {pool->seqs, (size_t)pool->cfg.max_seqs * sizeof(paged_kv_seq)},
      {pool->block_tables,
       (size_t)pool->cfg.max_seqs * pool->cfg.max_blocks_per_seq *
           sizeof(uint32_t)},
      {pool->prefix_table,
       (size_t)pool->prefix_table_cap * sizeof(prefix_cache_entry)},
  };
  for (size_t i = 0; i < sizeof(owned) / sizeof(owned[0]); ++i) {
    if (spans_overlap(owned[i].ptr, owned[i].bytes, out, bytes))
      return 1;
  }
  return 0;
}

/* ---- Prefix Hash Table Helpers (Zero Heap Allocation on Hot Path) ---- */

static int prefix_context_matches(const et_paged_kv_prefix_context_v1 *a,
                                  const et_paged_kv_prefix_context_v1 *b) {
  return memcmp(a->model, b->model, sizeof(a->model)) == 0 &&
         memcmp(a->tokenizer, b->tokenizer, sizeof(a->tokenizer)) == 0 &&
         memcmp(a->producer_config, b->producer_config,
                sizeof(a->producer_config)) == 0;
}

static uint32_t hash_slot_index(const uint8_t hash[ET_PAGED_KV_HASH_BYTES],
                                uint32_t cap_mask) {
  uint32_t h = ((uint32_t)hash[0]) | ((uint32_t)hash[1] << 8u) |
               ((uint32_t)hash[2] << 16u) | ((uint32_t)hash[3] << 24u);
  return h & cap_mask;
}

static int prefix_table_find(const et_paged_kv_pool *pool,
                             const uint8_t hash[ET_PAGED_KV_HASH_BYTES],
                             uint32_t *out_slot) {
  const uint32_t mask = pool->prefix_table_cap - 1u;
  uint32_t slot = hash_slot_index(hash, mask);
  for (uint32_t step = 0; step < pool->prefix_table_cap; ++step) {
    const prefix_cache_entry *e = &pool->prefix_table[slot];
    if (!e->occupied)
      return 0;
    if (memcmp(e->hash, hash, ET_PAGED_KV_HASH_BYTES) == 0) {
      if (out_slot != NULL)
        *out_slot = slot;
      return 1;
    }
    slot = (slot + 1u) & mask;
  }
  return 0;
}

static void prefix_table_remove_by_block(et_paged_kv_pool *pool,
                                         uint32_t block_id) {
  if (!pool->block_prefix_active[block_id])
    return;
  const uint8_t *hash =
      pool->block_prefix_hashes + (size_t)block_id * ET_PAGED_KV_HASH_BYTES;
  uint32_t slot = 0u;
  if (prefix_table_find(pool, hash, &slot) &&
      pool->prefix_table[slot].block_id == block_id) {
    const uint32_t mask = pool->prefix_table_cap - 1u;
    /* Backward-shift deletion preserves open-addressing probe invariants. */
    uint32_t empty = slot;
    uint32_t cur = (empty + 1u) & mask;
    while (pool->prefix_table[cur].occupied) {
      const uint32_t ideal =
          hash_slot_index(pool->prefix_table[cur].hash, mask);
      const uint32_t dist_cur = (cur - ideal) & mask;
      const uint32_t dist_empty = (cur - empty) & mask;
      if (dist_cur >= dist_empty) {
        pool->prefix_table[empty] = pool->prefix_table[cur];
        empty = cur;
      }
      cur = (cur + 1u) & mask;
    }
    memset(&pool->prefix_table[empty], 0, sizeof(pool->prefix_table[empty]));
    if (pool->prefix_cached_count > 0u)
      pool->prefix_cached_count--;
  }
  pool->block_prefix_active[block_id] = 0u;
  memset(pool->block_prefix_hashes + (size_t)block_id * ET_PAGED_KV_HASH_BYTES,
         0, ET_PAGED_KV_HASH_BYTES);
}

static et_paged_kv_status prefix_table_insert(
    et_paged_kv_pool *pool, uint32_t block_id,
    const uint8_t hash[ET_PAGED_KV_HASH_BYTES]) {
  const uint32_t mask = pool->prefix_table_cap - 1u;
  uint32_t slot = hash_slot_index(hash, mask);
  for (uint32_t step = 0; step < pool->prefix_table_cap; ++step) {
    prefix_cache_entry *e = &pool->prefix_table[slot];
    if (!e->occupied) {
      e->occupied = 1u;
      e->block_id = block_id;
      memcpy(e->hash, hash, ET_PAGED_KV_HASH_BYTES);
      pool->block_prefix_active[block_id] = 1u;
      memcpy(pool->block_prefix_hashes +
                 (size_t)block_id * ET_PAGED_KV_HASH_BYTES,
             hash, ET_PAGED_KV_HASH_BYTES);
      pool->prefix_cached_count++;
      return ET_PAGED_KV_OK;
    }
    if (memcmp(e->hash, hash, ET_PAGED_KV_HASH_BYTES) == 0) {
      /* Already indexed under this exact hash; keep canonical resident entry. */
      return ET_PAGED_KV_OK;
    }
    slot = (slot + 1u) & mask;
  }
  return ET_PAGED_KV_ERR_STATE;
}

/* ---- Physical Block Indexing & Allocation Helpers ---- */

static float *block_base_ptr(const et_paged_kv_pool *pool, uint32_t phys_block) {
  return pool->kv_data + (size_t)phys_block * pool->floats_per_block;
}

static float *block_slot_ptr(const et_paged_kv_pool *pool, uint32_t phys_block,
                             uint32_t layer_idx, uint32_t is_v,
                             uint32_t kv_head_idx, uint32_t slot) {
  const size_t offset =
      (size_t)phys_block * pool->floats_per_block +
      (size_t)layer_idx * pool->floats_per_layer +
      (size_t)is_v * pool->floats_per_kv_side +
      (size_t)kv_head_idx * pool->floats_per_head +
      (size_t)slot * (size_t)pool->cfg.head_dim;
  return pool->kv_data + offset;
}

static uint32_t pop_free_block(et_paged_kv_pool *pool) {
  pool->free_count--;
  const uint32_t blk = pool->free_stack[pool->free_count];
  pool->ref_count[blk] = 1u;
  memset(block_base_ptr(pool, blk), 0, pool->floats_per_block * sizeof(float));
  return blk;
}

static void release_block(et_paged_kv_pool *pool, uint32_t phys_block) {
  if (pool->ref_count[phys_block] == 0u)
    return;
  pool->ref_count[phys_block]--;
  if (pool->ref_count[phys_block] == 0u) {
    prefix_table_remove_by_block(pool, phys_block);
    pool->free_stack[pool->free_count] = phys_block;
    pool->free_count++;
  }
}

/* Copy prefix tokens [0 .. valid_slots - 1] from src_blk to dst_blk across all
 * layers, both K and V, and all KV heads. Remaining slots are already zero. */
static void copy_block_prefix(et_paged_kv_pool *pool, uint32_t dst_blk,
                              uint32_t src_blk, uint32_t valid_slots) {
  const size_t copy_bytes =
      (size_t)valid_slots * (size_t)pool->cfg.head_dim * sizeof(float);
  if (copy_bytes == 0u)
    return;
  for (uint32_t layer = 0u; layer < pool->cfg.num_layers; ++layer) {
    for (uint32_t is_v = 0u; is_v < 2u; ++is_v) {
      for (uint32_t h = 0u; h < pool->cfg.num_kv_heads; ++h) {
        float *dst = block_slot_ptr(pool, dst_blk, layer, is_v, h, 0u);
        const float *src = block_slot_ptr(pool, src_blk, layer, is_v, h, 0u);
        memcpy(dst, src, copy_bytes);
      }
    }
  }
}

static void write_token_to_block(et_paged_kv_pool *pool, uint32_t phys_block,
                                 uint32_t slot, const float *k_token,
                                 const float *v_token) {
  const size_t hd = (size_t)pool->cfg.head_dim;
  const size_t hd_bytes = hd * sizeof(float);
  for (uint32_t layer = 0u; layer < pool->cfg.num_layers; ++layer) {
    for (uint32_t h = 0u; h < pool->cfg.num_kv_heads; ++h) {
      const size_t src_off =
          ((size_t)layer * (size_t)pool->cfg.num_kv_heads + (size_t)h) * hd;
      float *dst_k = block_slot_ptr(pool, phys_block, layer, 0u, h, slot);
      float *dst_v = block_slot_ptr(pool, phys_block, layer, 1u, h, slot);
      memcpy(dst_k, k_token + src_off, hd_bytes);
      memcpy(dst_v, v_token + src_off, hd_bytes);
    }
  }
}

static et_paged_kv_status validate_config_and_count(
    const et_paged_kv_config *cfg, size_t *out_pool_floats) {
  if (cfg == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_PAGED_KV_ERR_FP_ENV;
  if (cfg->reserved != 0u)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (cfg->num_blocks < 1u || cfg->num_blocks > ET_PAGED_KV_MAX_BLOCKS ||
      cfg->num_layers < 1u || cfg->num_layers > ET_PAGED_KV_MAX_LAYERS ||
      cfg->num_kv_heads < 1u || cfg->num_kv_heads > ET_PAGED_KV_MAX_KV_HEADS ||
      cfg->block_size < 1u || cfg->block_size > ET_PAGED_KV_MAX_BLOCK_SIZE ||
      cfg->head_dim < 1u || cfg->head_dim > ET_PAGED_KV_MAX_HEAD_DIM ||
      cfg->max_seqs < 1u || cfg->max_seqs > ET_PAGED_KV_MAX_SEQS ||
      cfg->max_blocks_per_seq < 1u ||
      cfg->max_blocks_per_seq > ET_PAGED_KV_MAX_BLOCKS_PER_SEQ ||
      cfg->max_blocks_per_seq > cfg->num_blocks)
    return ET_PAGED_KV_ERR_SHAPE;

  uint64_t total = (uint64_t)cfg->num_blocks;
  if (!mul_u64_checked(total, (uint64_t)cfg->num_layers, &total) ||
      !mul_u64_checked(total, UINT64_C(2), &total) ||
      !mul_u64_checked(total, (uint64_t)cfg->num_kv_heads, &total) ||
      !mul_u64_checked(total, (uint64_t)cfg->block_size, &total) ||
      !mul_u64_checked(total, (uint64_t)cfg->head_dim, &total))
    return ET_PAGED_KV_ERR_OVERFLOW;
  if (total > ET_PAGED_KV_MAX_POOL_FLOATS)
    return ET_PAGED_KV_ERR_OVERFLOW;
  if (total > (uint64_t)(SIZE_MAX / sizeof(float)))
    return ET_PAGED_KV_ERR_OVERFLOW;

  /* Verify max tokens per sequence does not overflow uint32_t. */
  if ((uint64_t)cfg->max_blocks_per_seq * (uint64_t)cfg->block_size >
      (uint64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_OVERFLOW;

  if (out_pool_floats != NULL)
    *out_pool_floats = (size_t)total;
  return ET_PAGED_KV_OK;
}

static uint32_t next_pow2_u32(uint32_t v) {
  uint32_t p = 1u;
  while (p < v && p <= (UINT32_MAX >> 1u))
    p <<= 1u;
  return p;
}

static void reset_pool_state(et_paged_kv_pool *pool) {
  memset(pool->kv_data, 0, pool->pool_bytes);
  memset(pool->ref_count, 0, (size_t)pool->cfg.num_blocks * sizeof(uint32_t));
  memset(pool->block_prefix_active, 0,
         (size_t)pool->cfg.num_blocks * sizeof(uint8_t));
  memset(pool->block_prefix_hashes, 0,
         (size_t)pool->cfg.num_blocks * ET_PAGED_KV_HASH_BYTES);
  memset(pool->block_tables, 0,
         (size_t)pool->cfg.max_seqs * (size_t)pool->cfg.max_blocks_per_seq *
             sizeof(uint32_t));
  memset(pool->prefix_table, 0,
         (size_t)pool->prefix_table_cap * sizeof(prefix_cache_entry));

  /* Push [num_blocks-1 .. 0] so pops return 0, 1, 2, ... deterministically. */
  for (uint32_t i = 0u; i < pool->cfg.num_blocks; ++i)
    pool->free_stack[i] = pool->cfg.num_blocks - 1u - i;
  pool->free_count = pool->cfg.num_blocks;

  for (uint32_t s = 0u; s < pool->cfg.max_seqs; ++s) {
    pool->seqs[s].alive = 0u;
    pool->seqs[s].seq_len = 0u;
    pool->seqs[s].num_logical_blocks = 0u;
    pool->seqs[s].block_table =
        pool->block_tables + (size_t)s * (size_t)pool->cfg.max_blocks_per_seq;
  }
  pool->active_seqs = 0u;
  pool->prefix_cached_count = 0u;
  pool->cow_copies = 0u;
  pool->prefix_cache_hits = 0u;
}

/* ---- Public API Implementation ---- */

et_paged_kv_status et_paged_kv_validate_config(const et_paged_kv_config *cfg) {
  return validate_config_and_count(cfg, NULL);
}

et_paged_kv_status et_paged_kv_pool_float_count(const et_paged_kv_config *cfg,
                                                size_t *out_count) {
  if (out_count == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (cfg != NULL &&
      spans_overlap(cfg, sizeof(*cfg), out_count, sizeof(*out_count)))
    return ET_PAGED_KV_ERR_ALIAS;
  return validate_config_and_count(cfg, out_count);
}

static et_paged_kv_status create_pool(
    const et_paged_kv_config *cfg, const et_paged_kv_prefix_context_v1 *context,
    void *external_pool, size_t external_pool_floats,
    et_paged_kv_pool **out_pool) {
  if (out_pool == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  size_t pool_floats = 0u;
  const et_paged_kv_status st = validate_config_and_count(cfg, &pool_floats);
  if (st != ET_PAGED_KV_OK)
    return st;
  if (spans_overlap(cfg, sizeof(*cfg), out_pool, sizeof(*out_pool)) ||
      (context != NULL &&
       spans_overlap(context, sizeof(*context), out_pool, sizeof(*out_pool))))
    return ET_PAGED_KV_ERR_ALIAS;

  const size_t pool_bytes = pool_floats * sizeof(float);
  if (external_pool == NULL) {
    if (external_pool_floats != 0u)
      return ET_PAGED_KV_ERR_ARGUMENT;
  } else {
    if (external_pool_floats < pool_floats)
      return ET_PAGED_KV_ERR_SHAPE;
    if (external_pool_floats > SIZE_MAX / sizeof(float))
      return ET_PAGED_KV_ERR_OVERFLOW;
    const size_t ext_bytes = external_pool_floats * sizeof(float);
    if (!aligned_span(external_pool, _Alignof(float), ext_bytes))
      return ET_PAGED_KV_ERR_ALIGNMENT;
    const span_entry spans[3] = {
        {cfg, sizeof(*cfg)},
        {out_pool, sizeof(*out_pool)},
        {external_pool, ext_bytes},
    };
    if (any_spans_overlap(spans, 3u) ||
        (context != NULL &&
         spans_overlap(context, sizeof(*context), external_pool, ext_bytes)))
      return ET_PAGED_KV_ERR_ALIAS;
  }

  et_paged_kv_pool *pool = (et_paged_kv_pool *)calloc(1u, sizeof(*pool));
  if (pool == NULL)
    return ET_PAGED_KV_ERR_NOMEM;

  pool->cfg = *cfg;
  if (context != NULL) {
    pool->prefix_context_bound = 1;
    pool->prefix_context = *context;
  }
  pool->pool_floats = pool_floats;
  pool->pool_bytes = pool_bytes;
  pool->floats_per_head = (size_t)cfg->block_size * (size_t)cfg->head_dim;
  pool->floats_per_kv_side = (size_t)cfg->num_kv_heads * pool->floats_per_head;
  pool->floats_per_layer = 2u * pool->floats_per_kv_side;
  pool->floats_per_block = (size_t)cfg->num_layers * pool->floats_per_layer;
  pool->floats_per_token_all_layers = (size_t)cfg->num_layers *
                                      (size_t)cfg->num_kv_heads *
                                      (size_t)cfg->head_dim;

  pool->prefix_table_cap = next_pow2_u32(cfg->num_blocks * 2u);
  if (pool->prefix_table_cap < 8u)
    pool->prefix_table_cap = 8u;

  if (external_pool == NULL) {
    pool->kv_data = (float *)calloc(pool_floats, sizeof(float));
    pool->owns_kv_data = 1;
  } else {
    pool->kv_data = (float *)external_pool;
    pool->owns_kv_data = 0;
  }

  pool->ref_count = (uint32_t *)calloc(cfg->num_blocks, sizeof(uint32_t));
  pool->block_prefix_active =
      (uint8_t *)calloc(cfg->num_blocks, sizeof(uint8_t));
  pool->block_prefix_hashes = (uint8_t *)calloc(
      (size_t)cfg->num_blocks * ET_PAGED_KV_HASH_BYTES, sizeof(uint8_t));
  pool->free_stack = (uint32_t *)calloc(cfg->num_blocks, sizeof(uint32_t));
  pool->seqs = (paged_kv_seq *)calloc(cfg->max_seqs, sizeof(paged_kv_seq));
  pool->block_tables = (uint32_t *)calloc(
      (size_t)cfg->max_seqs * (size_t)cfg->max_blocks_per_seq,
      sizeof(uint32_t));
  pool->prefix_table = (prefix_cache_entry *)calloc(
      pool->prefix_table_cap, sizeof(prefix_cache_entry));

  if (pool->kv_data == NULL || pool->ref_count == NULL ||
      pool->block_prefix_active == NULL || pool->block_prefix_hashes == NULL ||
      pool->free_stack == NULL || pool->seqs == NULL ||
      pool->block_tables == NULL || pool->prefix_table == NULL) {
    et_paged_kv_destroy(pool);
    return ET_PAGED_KV_ERR_NOMEM;
  }

  reset_pool_state(pool);
  *out_pool = pool;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_create_raw(const et_paged_kv_config *cfg,
                                         void *external_pool,
                                         size_t external_pool_floats,
                                         et_paged_kv_pool **out_pool) {
  return create_pool(cfg, NULL, external_pool, external_pool_floats, out_pool);
}

et_paged_kv_status et_paged_kv_create(const et_paged_kv_config *cfg,
                                      float *external_pool,
                                      size_t external_pool_floats,
                                      et_paged_kv_pool **out_pool) {
  return et_paged_kv_create_raw(cfg, external_pool, external_pool_floats,
                                out_pool);
}

et_paged_kv_status et_paged_kv_create_with_prefix_context_v1(
    const et_paged_kv_config *cfg, const et_paged_kv_prefix_context_v1 *context,
    float *external_pool, size_t external_pool_floats,
    et_paged_kv_pool **out_pool) {
  if (context == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  return create_pool(cfg, context, external_pool, external_pool_floats,
                     out_pool);
}

void et_paged_kv_destroy(et_paged_kv_pool *pool) {
  if (pool == NULL)
    return;
  if (pool->owns_kv_data)
    free(pool->kv_data);
  free(pool->ref_count);
  free(pool->block_prefix_active);
  free(pool->block_prefix_hashes);
  free(pool->free_stack);
  free(pool->seqs);
  free(pool->block_tables);
  free(pool->prefix_table);
  free(pool);
}

et_paged_kv_status et_paged_kv_reset(et_paged_kv_pool *pool) {
  if (pool == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_PAGED_KV_ERR_FP_ENV;
  reset_pool_state(pool);
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_stats_get(const et_paged_kv_pool *pool,
                                         et_paged_kv_stats *out_stats) {
  if (pool == NULL || out_stats == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (output_overlaps_pool(pool, out_stats, sizeof(*out_stats)))
    return ET_PAGED_KV_ERR_ALIAS;

  uint32_t shared = 0u;
  for (uint32_t b = 0u; b < pool->cfg.num_blocks; ++b) {
    if (pool->ref_count[b] > 1u)
      shared++;
  }

  et_paged_kv_stats s;
  s.total_blocks = pool->cfg.num_blocks;
  s.free_blocks = pool->free_count;
  s.used_blocks = pool->cfg.num_blocks - pool->free_count;
  s.active_seqs = pool->active_seqs;
  s.shared_blocks = shared;
  s.prefix_cached_blocks = pool->prefix_cached_count;
  s.cow_copies = pool->cow_copies;
  s.prefix_cache_hits = pool->prefix_cache_hits;
  *out_stats = s;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_config_get(const et_paged_kv_pool *pool,
                                          et_paged_kv_config *out_cfg) {
  if (pool == NULL || out_cfg == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (output_overlaps_pool(pool, out_cfg, sizeof(*out_cfg)))
    return ET_PAGED_KV_ERR_ALIAS;
  *out_cfg = pool->cfg;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_seq_alloc(et_paged_kv_pool *pool,
                                         uint32_t *out_seq_id) {
  if (pool == NULL || out_seq_id == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (output_overlaps_pool(pool, out_seq_id, sizeof(*out_seq_id)))
    return ET_PAGED_KV_ERR_ALIAS;
  for (uint32_t s = 0u; s < pool->cfg.max_seqs; ++s) {
    if (!pool->seqs[s].alive) {
      pool->seqs[s].alive = 1u;
      pool->seqs[s].seq_len = 0u;
      pool->seqs[s].num_logical_blocks = 0u;
      pool->active_seqs++;
      *out_seq_id = s;
      return ET_PAGED_KV_OK;
    }
  }
  return ET_PAGED_KV_ERR_OOM_SEQS;
}

et_paged_kv_status et_paged_kv_seq_free(et_paged_kv_pool *pool,
                                        uint32_t seq_id) {
  if (pool == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;

  paged_kv_seq *seq = &pool->seqs[seq_id];
  for (uint32_t i = seq->num_logical_blocks; i > 0u; --i) {
    const uint32_t idx = i - 1u;
    release_block(pool, seq->block_table[idx]);
    seq->block_table[idx] = 0u;
  }
  seq->alive = 0u;
  seq->seq_len = 0u;
  seq->num_logical_blocks = 0u;
  pool->active_seqs--;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_seq_fork(et_paged_kv_pool *pool,
                                        uint32_t parent_seq_id,
                                        uint32_t *out_child_seq_id) {
  if (pool == NULL || out_child_seq_id == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (parent_seq_id >= pool->cfg.max_seqs || !pool->seqs[parent_seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (output_overlaps_pool(pool, out_child_seq_id,
                           sizeof(*out_child_seq_id)))
    return ET_PAGED_KV_ERR_ALIAS;

  uint32_t child_id = pool->cfg.max_seqs;
  for (uint32_t s = 0u; s < pool->cfg.max_seqs; ++s) {
    if (!pool->seqs[s].alive) {
      child_id = s;
      break;
    }
  }
  if (child_id == pool->cfg.max_seqs)
    return ET_PAGED_KV_ERR_OOM_SEQS;

  const paged_kv_seq *parent = &pool->seqs[parent_seq_id];
  paged_kv_seq *child = &pool->seqs[child_id];
  child->alive = 1u;
  child->seq_len = parent->seq_len;
  child->num_logical_blocks = parent->num_logical_blocks;
  for (uint32_t i = 0u; i < parent->num_logical_blocks; ++i) {
    const uint32_t phys = parent->block_table[i];
    child->block_table[i] = phys;
    pool->ref_count[phys]++;
  }
  pool->active_seqs++;
  *out_child_seq_id = child_id;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_seq_truncate(et_paged_kv_pool *pool,
                                            uint32_t seq_id, uint32_t new_len) {
  if (pool == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;

  paged_kv_seq *seq = &pool->seqs[seq_id];
  if (new_len > seq->seq_len)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  if (new_len == seq->seq_len)
    return ET_PAGED_KV_OK;

  const uint32_t bs = pool->cfg.block_size;
  const uint32_t keep_blocks = (new_len == 0u) ? 0u : ((new_len + bs - 1u) / bs);

  for (uint32_t i = seq->num_logical_blocks; i > keep_blocks; --i) {
    const uint32_t idx = i - 1u;
    release_block(pool, seq->block_table[idx]);
    seq->block_table[idx] = 0u;
  }

  /* If the retained tail block is now partial and exclusively owned by this
   * sequence, remove any prefix-cache entry so future writes cannot alias a
   * cached full-block hash. If ref_count > 1, siblings still own the full
   * block intact and any future write on this sequence will trigger CoW. */
  if (keep_blocks > 0u && (new_len % bs) != 0u) {
    const uint32_t tail_phys = seq->block_table[keep_blocks - 1u];
    if (pool->ref_count[tail_phys] == 1u)
      prefix_table_remove_by_block(pool, tail_phys);
  }

  seq->seq_len = new_len;
  seq->num_logical_blocks = keep_blocks;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_seq_len(const et_paged_kv_pool *pool,
                                       uint32_t seq_id, uint32_t *out_len) {
  if (pool == NULL || out_len == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (output_overlaps_pool(pool, out_len, sizeof(*out_len)))
    return ET_PAGED_KV_ERR_ALIAS;
  *out_len = pool->seqs[seq_id].seq_len;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_seq_block_table(const et_paged_kv_pool *pool,
                                               uint32_t seq_id,
                                               uint32_t *out_blocks,
                                               uint32_t capacity,
                                               uint32_t *out_num_blocks) {
  if (pool == NULL || out_num_blocks == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;

  const paged_kv_seq *seq = &pool->seqs[seq_id];
  const size_t blocks_bytes =
      (size_t)seq->num_logical_blocks * sizeof(uint32_t);
  if (output_overlaps_pool(pool, out_num_blocks, sizeof(*out_num_blocks)) ||
      (out_blocks != NULL &&
       (output_overlaps_pool(pool, out_blocks, blocks_bytes) ||
        spans_overlap(out_blocks, blocks_bytes, out_num_blocks,
                      sizeof(*out_num_blocks)))))
    return ET_PAGED_KV_ERR_ALIAS;
  if (out_blocks == NULL) {
    if (capacity != 0u)
      return ET_PAGED_KV_ERR_ARGUMENT;
    *out_num_blocks = seq->num_logical_blocks;
    return ET_PAGED_KV_OK;
  }
  if (capacity < seq->num_logical_blocks)
    return ET_PAGED_KV_ERR_SHAPE;
  if (seq->num_logical_blocks > 0u) {
    memcpy(out_blocks, seq->block_table,
           (size_t)seq->num_logical_blocks * sizeof(uint32_t));
  }
  *out_num_blocks = seq->num_logical_blocks;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_block_refcount(const et_paged_kv_pool *pool,
                                              uint32_t block_id,
                                              uint32_t *out_refcount) {
  if (pool == NULL || out_refcount == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (block_id >= pool->cfg.num_blocks)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  if (output_overlaps_pool(pool, out_refcount, sizeof(*out_refcount)))
    return ET_PAGED_KV_ERR_ALIAS;
  *out_refcount = pool->ref_count[block_id];
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_append_tokens_raw(et_paged_kv_pool *pool,
                                                 uint32_t seq_id,
                                                 uint32_t num_tokens,
                                                 const void *k_tokens,
                                                 const void *v_tokens) {
  if (pool == NULL || k_tokens == NULL || v_tokens == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_PAGED_KV_ERR_FP_ENV;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (num_tokens == 0u)
    return ET_PAGED_KV_ERR_SHAPE;

  paged_kv_seq *seq = &pool->seqs[seq_id];
  const uint32_t bs = pool->cfg.block_size;
  const uint32_t cur_len = seq->seq_len;
  const uint64_t new_len_u64 = (uint64_t)cur_len + (uint64_t)num_tokens;
  const uint64_t max_tokens =
      (uint64_t)pool->cfg.max_blocks_per_seq * (uint64_t)bs;
  if (new_len_u64 > max_tokens)
    return ET_PAGED_KV_ERR_SEQ_FULL;

  uint64_t total_floats_u64 = 0u;
  if (!mul_u64_checked((uint64_t)num_tokens,
                       (uint64_t)pool->floats_per_token_all_layers,
                       &total_floats_u64) ||
      total_floats_u64 > (uint64_t)(SIZE_MAX / sizeof(float)))
    return ET_PAGED_KV_ERR_OVERFLOW;

  const size_t total_floats = (size_t)total_floats_u64;
  const size_t total_bytes = total_floats * sizeof(float);
  if (!aligned_span(k_tokens, _Alignof(float), total_bytes) ||
      !aligned_span(v_tokens, _Alignof(float), total_bytes))
    return ET_PAGED_KV_ERR_ALIGNMENT;

  const span_entry spans[4] = {
      {pool, sizeof(*pool)},
      {pool->kv_data, pool->pool_bytes},
      {k_tokens, total_bytes},
      {v_tokens, total_bytes},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_PAGED_KV_ERR_ALIAS;

  const float *kf = (const float *)k_tokens;
  const float *vf = (const float *)v_tokens;
  if (!all_finite_f32(kf, total_floats) || !all_finite_f32(vf, total_floats))
    return ET_PAGED_KV_ERR_NONFINITE;

  /* Exact pre-flight block availability check guarantees 100% rollback safety
   * before mutating any pool, refcount, free-list, or sequence state. */
  const uint32_t new_len = (uint32_t)new_len_u64;
  const uint32_t new_blocks = (new_len + bs - 1u) / bs;
  const uint32_t cur_blocks = seq->num_logical_blocks;
  const uint32_t extra_blocks = new_blocks - cur_blocks;
  const uint32_t needs_cow =
      ((cur_len % bs) != 0u &&
       pool->ref_count[seq->block_table[cur_blocks - 1u]] > 1u)
          ? 1u
          : 0u;
  const uint32_t total_needed_blocks = extra_blocks + needs_cow;
  if (pool->free_count < total_needed_blocks)
    return ET_PAGED_KV_ERR_OOM_BLOCKS;

  /* All validations passed; execute atomic CoW + writes. */
  const size_t tok_stride = pool->floats_per_token_all_layers;
  for (uint32_t t = 0u; t < num_tokens; ++t) {
    const uint32_t pos = seq->seq_len;
    const uint32_t lb = pos / bs;
    const uint32_t slot = pos % bs;
    uint32_t target_blk = 0u;

    if (slot == 0u) {
      target_blk = pop_free_block(pool);
      seq->block_table[lb] = target_blk;
      seq->num_logical_blocks = lb + 1u;
    } else {
      const uint32_t old_blk = seq->block_table[lb];
      if (pool->ref_count[old_blk] > 1u) {
        target_blk = pop_free_block(pool);
        copy_block_prefix(pool, target_blk, old_blk, slot);
        seq->block_table[lb] = target_blk;
        pool->ref_count[old_blk]--;
        pool->cow_copies++;
      } else {
        /* Exclusive ownership: ensure not prefix-indexed before mutating. */
        prefix_table_remove_by_block(pool, old_blk);
        target_blk = old_blk;
      }
    }

    write_token_to_block(pool, target_blk, slot, kf + (size_t)t * tok_stride,
                         vf + (size_t)t * tok_stride);
    seq->seq_len = pos + 1u;
  }

  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_append_tokens(et_paged_kv_pool *pool,
                                             uint32_t seq_id,
                                             uint32_t num_tokens,
                                             const float *k_tokens,
                                             const float *v_tokens) {
  return et_paged_kv_append_tokens_raw(pool, seq_id, num_tokens, k_tokens,
                                       v_tokens);
}

et_paged_kv_status et_paged_kv_append_token_raw(et_paged_kv_pool *pool,
                                                uint32_t seq_id,
                                                const void *k_token,
                                                const void *v_token) {
  return et_paged_kv_append_tokens_raw(pool, seq_id, 1u, k_token, v_token);
}

et_paged_kv_status et_paged_kv_append_token(et_paged_kv_pool *pool,
                                            uint32_t seq_id,
                                            const float *k_token,
                                            const float *v_token) {
  return et_paged_kv_append_tokens_raw(pool, seq_id, 1u, k_token, v_token);
}

et_paged_kv_status et_paged_kv_read_token_raw(const et_paged_kv_pool *pool,
                                              uint32_t seq_id, uint32_t pos,
                                              uint32_t layer_idx,
                                              uint32_t kv_head_idx, void *out_k,
                                              void *out_v) {
  if (pool == NULL || out_k == NULL || out_v == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  const paged_kv_seq *seq = &pool->seqs[seq_id];
  if (pos >= seq->seq_len)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  if (layer_idx >= pool->cfg.num_layers ||
      kv_head_idx >= pool->cfg.num_kv_heads)
    return ET_PAGED_KV_ERR_SHAPE;

  const size_t hd_bytes = (size_t)pool->cfg.head_dim * sizeof(float);
  if (!aligned_span(out_k, _Alignof(float), hd_bytes) ||
      !aligned_span(out_v, _Alignof(float), hd_bytes))
    return ET_PAGED_KV_ERR_ALIGNMENT;

  const span_entry spans[4] = {
      {pool, sizeof(*pool)},
      {pool->kv_data, pool->pool_bytes},
      {out_k, hd_bytes},
      {out_v, hd_bytes},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_PAGED_KV_ERR_ALIAS;

  const uint32_t lb = pos / pool->cfg.block_size;
  const uint32_t slot = pos % pool->cfg.block_size;
  const uint32_t phys = seq->block_table[lb];
  const float *src_k = block_slot_ptr(pool, phys, layer_idx, 0u, kv_head_idx,
                                      slot);
  const float *src_v = block_slot_ptr(pool, phys, layer_idx, 1u, kv_head_idx,
                                      slot);
  memcpy(out_k, src_k, hd_bytes);
  memcpy(out_v, src_v, hd_bytes);
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_read_token(const et_paged_kv_pool *pool,
                                          uint32_t seq_id, uint32_t pos,
                                          uint32_t layer_idx,
                                          uint32_t kv_head_idx, float *out_k,
                                          float *out_v) {
  return et_paged_kv_read_token_raw(pool, seq_id, pos, layer_idx, kv_head_idx,
                                    out_k, out_v);
}

et_paged_kv_status et_paged_kv_decode_attention_v1_raw(
    const et_paged_kv_pool *pool, uint32_t seq_id, uint32_t layer_idx,
    uint32_t num_q_heads, float scale, const void *q, void *out) {
  if (pool == NULL || q == NULL || out == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_PAGED_KV_ERR_FP_ENV;
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;

  const paged_kv_seq *seq = &pool->seqs[seq_id];
  if (seq->seq_len == 0u)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  if (layer_idx >= pool->cfg.num_layers || num_q_heads < 1u ||
      num_q_heads > ET_PAGED_KV_MAX_Q_HEADS ||
      (num_q_heads % pool->cfg.num_kv_heads) != 0u)
    return ET_PAGED_KV_ERR_SHAPE;
  if (!isfinite(scale))
    return ET_PAGED_KV_ERR_NONFINITE;
  if (!(scale > 0.0f))
    return ET_PAGED_KV_ERR_ARGUMENT;

  const uint32_t hd = pool->cfg.head_dim;
  const size_t q_floats = (size_t)num_q_heads * (size_t)hd;
  const size_t q_bytes = q_floats * sizeof(float);
  if (!aligned_span(q, _Alignof(float), q_bytes) ||
      !aligned_span(out, _Alignof(float), q_bytes))
    return ET_PAGED_KV_ERR_ALIGNMENT;

  const span_entry spans[4] = {
      {pool, sizeof(*pool)},
      {pool->kv_data, pool->pool_bytes},
      {q, q_bytes},
      {out, q_bytes},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_PAGED_KV_ERR_ALIAS;

  const float *qf = (const float *)q;
  if (!all_finite_f32(qf, q_floats))
    return ET_PAGED_KV_ERR_NONFINITE;

  /* Zero heap allocations on the hot path: bounded stack staging buffer
   * preserves strict failure atomicity (out is untouched on error). */
  float out_stage[ET_PAGED_KV_MAX_Q_HEADS * ET_PAGED_KV_MAX_HEAD_DIM];
  double acc[ET_PAGED_KV_MAX_HEAD_DIM];

  const uint32_t group_size = num_q_heads / pool->cfg.num_kv_heads;
  const uint32_t bs = pool->cfg.block_size;
  const uint32_t seq_len = seq->seq_len;
  const uint32_t nblocks = seq->num_logical_blocks;
  const double dscale = (double)scale;

  for (uint32_t qh = 0u; qh < num_q_heads; ++qh) {
    const uint32_t kv_h = qh / group_size;
    const float *q_head = qf + (size_t)qh * (size_t)hd;

    double running_max = -INFINITY;
    double running_sum = 0.0;
    for (uint32_t d = 0u; d < hd; ++d)
      acc[d] = 0.0;

    for (uint32_t lb = 0u; lb < nblocks; ++lb) {
      const uint32_t phys = seq->block_table[lb];
      const uint32_t rem = seq_len - lb * bs;
      const uint32_t tokens_in_blk = rem < bs ? rem : bs;
      const float *k_blk = block_slot_ptr(pool, phys, layer_idx, 0u, kv_h, 0u);
      const float *v_blk = block_slot_ptr(pool, phys, layer_idx, 1u, kv_h, 0u);

      for (uint32_t slot = 0u; slot < tokens_in_blk; ++slot) {
        const float *k_vec = k_blk + (size_t)slot * (size_t)hd;
        const float *v_vec = v_blk + (size_t)slot * (size_t)hd;

        double dot = 0.0;
        for (uint32_t d = 0u; d < hd; ++d) {
          if (!isfinite(k_vec[d]) || !isfinite(v_vec[d]))
            return ET_PAGED_KV_ERR_NONFINITE;
          dot += (double)q_head[d] * (double)k_vec[d];
        }
        const double score = dot * dscale;
        if (!isfinite(score))
          return ET_PAGED_KV_ERR_NONFINITE;

        if (score > running_max) {
          const double alpha =
              (running_sum == 0.0) ? 0.0 : exp(running_max - score);
          running_max = score;
          running_sum = running_sum * alpha + 1.0;
          for (uint32_t d = 0u; d < hd; ++d)
            acc[d] = acc[d] * alpha + (double)v_vec[d];
        } else {
          const double weight = exp(score - running_max);
          running_sum += weight;
          for (uint32_t d = 0u; d < hd; ++d)
            acc[d] += weight * (double)v_vec[d];
        }
      }
    }

    if (!isfinite(running_sum) || !(running_sum > 0.0))
      return ET_PAGED_KV_ERR_NONFINITE;

    float *dst = out_stage + (size_t)qh * (size_t)hd;
    for (uint32_t d = 0u; d < hd; ++d) {
      const double v = acc[d] / running_sum;
      const float fv = (float)v;
      if (!isfinite(fv))
        return ET_PAGED_KV_ERR_NONFINITE;
      dst[d] = fv;
    }
  }

  memcpy(out, out_stage, q_bytes);
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_decode_attention_v1(
    const et_paged_kv_pool *pool, uint32_t seq_id, uint32_t layer_idx,
    uint32_t num_q_heads, float scale, const float *q, float *out) {
  return et_paged_kv_decode_attention_v1_raw(pool, seq_id, layer_idx,
                                             num_q_heads, scale, q, out);
}

static void hash_token_block_finish(
    sha256_state *st, const uint8_t *prev_hash_or_null,
    const uint32_t *token_ids, uint32_t block_size,
    uint8_t out_hash[ET_PAGED_KV_HASH_BYTES]) {
  static const uint8_t zero_prev[ET_PAGED_KV_HASH_BYTES] = {0u};
  uint8_t le[4];

  sha256_update(st, prev_hash_or_null != NULL ? prev_hash_or_null : zero_prev,
                ET_PAGED_KV_HASH_BYTES);
  encode_u32_le(le, block_size);
  sha256_update(st, le, sizeof(le));
  for (uint32_t i = 0u; i < block_size; ++i) {
    encode_u32_le(le, token_ids[i]);
    sha256_update(st, le, sizeof(le));
  }
  sha256_final(st, out_hash);
}

et_paged_kv_status et_paged_kv_hash_token_block(
    const uint8_t *prev_hash_or_null, const uint32_t *token_ids,
    uint32_t block_size, uint8_t out_hash[ET_PAGED_KV_HASH_BYTES]) {
  if (token_ids == NULL || out_hash == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (block_size < 1u || block_size > ET_PAGED_KV_MAX_BLOCK_SIZE)
    return ET_PAGED_KV_ERR_SHAPE;

  static const uint8_t domain[8] = {'E', 'T', 'P', 'K', 'V', '1', 0u, 0u};
  sha256_state st;
  sha256_init(&st);
  sha256_update(&st, domain, sizeof(domain));
  hash_token_block_finish(&st, prev_hash_or_null, token_ids, block_size,
                          out_hash);
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_hash_token_block_with_context_v1(
    const et_paged_kv_prefix_context_v1 *context,
    const et_paged_kv_prefix_key_v1 *previous_or_null,
    const uint32_t *token_ids, uint32_t block_size,
    et_paged_kv_prefix_key_v1 *out_key) {
  if (context == NULL || token_ids == NULL || out_key == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (block_size < 1u || block_size > ET_PAGED_KV_MAX_BLOCK_SIZE)
    return ET_PAGED_KV_ERR_SHAPE;
  if (!aligned_span(context, _Alignof(et_paged_kv_prefix_context_v1),
                    sizeof(*context)) ||
      !aligned_span(token_ids, _Alignof(uint32_t),
                    (size_t)block_size * sizeof(*token_ids)) ||
      !aligned_span(out_key, _Alignof(et_paged_kv_prefix_key_v1),
                    sizeof(*out_key)))
    return ET_PAGED_KV_ERR_ALIGNMENT;
  if (previous_or_null != NULL) {
    if (!aligned_span(previous_or_null, _Alignof(et_paged_kv_prefix_key_v1),
                      sizeof(*previous_or_null)))
      return ET_PAGED_KV_ERR_ALIGNMENT;
    if (!prefix_context_matches(context, &previous_or_null->context))
      return ET_PAGED_KV_ERR_CONTEXT;
    if (previous_or_null->block_size != block_size)
      return ET_PAGED_KV_ERR_SHAPE;
  }

  static const uint8_t domain[8] = {'E', 'T', 'P', 'K', 'V', '2', 0u, 0u};
  const uint8_t has_previous = previous_or_null != NULL;
  et_paged_kv_prefix_key_v1 key;
  key.context = *context;
  key.block_size = block_size;
  sha256_state st;
  sha256_init(&st);
  sha256_update(&st, domain, sizeof(domain));
  sha256_update(&st, context->model, sizeof(context->model));
  sha256_update(&st, context->tokenizer, sizeof(context->tokenizer));
  sha256_update(&st, context->producer_config, sizeof(context->producer_config));
  sha256_update(&st, &has_previous, sizeof(has_previous));
  hash_token_block_finish(&st,
                          has_previous ? previous_or_null->digest : NULL,
                          token_ids, block_size, key.digest);
  *out_key = key;
  return ET_PAGED_KV_OK;
}

static et_paged_kv_status prefix_cache_register(
    et_paged_kv_pool *pool, uint32_t seq_id, uint32_t logical_block,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES]) {
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;

  const paged_kv_seq *seq = &pool->seqs[seq_id];
  const uint32_t full_blocks = seq->seq_len / pool->cfg.block_size;
  if (logical_block >= full_blocks)
    return ET_PAGED_KV_ERR_INVALID_RANGE;

  const uint32_t phys = seq->block_table[logical_block];
  if (pool->block_prefix_active[phys]) {
    const uint8_t *existing =
        pool->block_prefix_hashes + (size_t)phys * ET_PAGED_KV_HASH_BYTES;
    if (memcmp(existing, prefix_hash, ET_PAGED_KV_HASH_BYTES) == 0)
      return ET_PAGED_KV_OK;
    return ET_PAGED_KV_ERR_STATE;
  }

  return prefix_table_insert(pool, phys, prefix_hash);
}

static et_paged_kv_status prefix_cache_attach(
    et_paged_kv_pool *pool, uint32_t seq_id,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES], uint32_t *out_hit) {
  if (seq_id >= pool->cfg.max_seqs || !pool->seqs[seq_id].alive)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (output_overlaps_pool(pool, out_hit, sizeof(*out_hit)) ||
      spans_overlap(prefix_hash, ET_PAGED_KV_HASH_BYTES, out_hit,
                    sizeof(*out_hit)))
    return ET_PAGED_KV_ERR_ALIAS;

  paged_kv_seq *seq = &pool->seqs[seq_id];
  const uint32_t bs = pool->cfg.block_size;
  if ((seq->seq_len % bs) != 0u)
    return ET_PAGED_KV_ERR_STATE;
  if (seq->num_logical_blocks >= pool->cfg.max_blocks_per_seq)
    return ET_PAGED_KV_ERR_SEQ_FULL;

  uint32_t slot = 0u;
  if (!prefix_table_find(pool, prefix_hash, &slot)) {
    *out_hit = 0u;
    return ET_PAGED_KV_OK;
  }

  const uint32_t phys = pool->prefix_table[slot].block_id;
  seq->block_table[seq->num_logical_blocks] = phys;
  seq->num_logical_blocks++;
  seq->seq_len += bs;
  pool->ref_count[phys]++;
  pool->prefix_cache_hits++;
  *out_hit = 1u;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_prefix_cache_register(
    et_paged_kv_pool *pool, uint32_t seq_id, uint32_t logical_block,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES]) {
  if (pool == NULL || prefix_hash == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (pool->prefix_context_bound)
    return ET_PAGED_KV_ERR_STATE;
  return prefix_cache_register(pool, seq_id, logical_block, prefix_hash);
}

et_paged_kv_status et_paged_kv_prefix_cache_attach(
    et_paged_kv_pool *pool, uint32_t seq_id,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES], uint32_t *out_hit) {
  if (pool == NULL || prefix_hash == NULL || out_hit == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (pool->prefix_context_bound)
    return ET_PAGED_KV_ERR_STATE;
  return prefix_cache_attach(pool, seq_id, prefix_hash, out_hit);
}

static et_paged_kv_status validate_pool_prefix_key(
    const et_paged_kv_pool *pool, const et_paged_kv_prefix_key_v1 *key) {
  if (pool == NULL || key == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (!pool->prefix_context_bound)
    return ET_PAGED_KV_ERR_STATE;
  if (!aligned_span(key, _Alignof(et_paged_kv_prefix_key_v1), sizeof(*key)))
    return ET_PAGED_KV_ERR_ALIGNMENT;
  if (!prefix_context_matches(&pool->prefix_context, &key->context))
    return ET_PAGED_KV_ERR_CONTEXT;
  if (key->block_size != pool->cfg.block_size)
    return ET_PAGED_KV_ERR_SHAPE;
  return ET_PAGED_KV_OK;
}

et_paged_kv_status et_paged_kv_prefix_cache_register_with_context_v1(
    et_paged_kv_pool *pool, uint32_t seq_id, uint32_t logical_block,
    const et_paged_kv_prefix_key_v1 *key) {
  const et_paged_kv_status st = validate_pool_prefix_key(pool, key);
  if (st != ET_PAGED_KV_OK)
    return st;
  return prefix_cache_register(pool, seq_id, logical_block, key->digest);
}

et_paged_kv_status et_paged_kv_prefix_cache_attach_with_context_v1(
    et_paged_kv_pool *pool, uint32_t seq_id,
    const et_paged_kv_prefix_key_v1 *key, uint32_t *out_hit) {
  if (out_hit == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  const et_paged_kv_status st = validate_pool_prefix_key(pool, key);
  if (st != ET_PAGED_KV_OK)
    return st;
  if (!aligned_span(out_hit, _Alignof(uint32_t), sizeof(*out_hit)))
    return ET_PAGED_KV_ERR_ALIGNMENT;
  if (spans_overlap(key, sizeof(*key), out_hit, sizeof(*out_hit)))
    return ET_PAGED_KV_ERR_ALIAS;
  return prefix_cache_attach(pool, seq_id, key->digest, out_hit);
}
