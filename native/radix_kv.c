#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/radix_kv_abi.h"

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

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Radix KV engine requires 8-bit bytes and IEEE binary32/64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "Radix KV engine requires exact IEEE-754 binary32/binary64");

#define ET_RADIX_NONE UINT32_MAX
#define ET_RADIX_F32_ABS_MASK UINT32_C(0x7fffffff)
#define ET_RADIX_F32_INF_BITS UINT32_C(0x7f800000)

/* ---- Standalone streaming SHA-256 (zero heap allocations) ---- */

typedef struct sha256_state {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} sha256_state;

static uint32_t rotr32(uint32_t x, unsigned int n) {
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
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (size_t i = 0u; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (size_t i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

/* ---- Byte-wise little-endian encoding & numeric guards ---- */

static uint32_t load_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) |
         ((uint32_t)in[2] << 16u) | ((uint32_t)in[3] << 24u);
}

static void store_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static uint64_t load_u64_le(const uint8_t in[8]) {
  return (uint64_t)in[0] | ((uint64_t)in[1] << 8u) |
         ((uint64_t)in[2] << 16u) | ((uint64_t)in[3] << 24u) |
         ((uint64_t)in[4] << 32u) | ((uint64_t)in[5] << 40u) |
         ((uint64_t)in[6] << 48u) | ((uint64_t)in[7] << 56u);
}

static void store_u64_le(uint8_t out[8], uint64_t value) {
  for (size_t i = 0u; i < 8u; ++i)
    out[i] = (uint8_t)((value >> (8u * i)) & 0xffu);
}

static uint32_t f32_to_bits(float v) {
  uint32_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static float bits_to_f32(uint32_t bits) {
  float v = 0.0f;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static int f32_bits_finite(uint32_t bits) {
  return (bits & ET_RADIX_F32_ABS_MASK) < ET_RADIX_F32_INF_BITS;
}

static int f32_is_finite(float v) {
  return f32_bits_finite(f32_to_bits(v));
}

static int fp_environment_supported(void) {
  if (fegetround() != FE_TONEAREST)
    return 0;
#if defined(__x86_64__)
  if ((_mm_getcsr() & 0x8040u) != 0u)
    return 0;
#endif
  return 1;
}

static int mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static int add_size(size_t a, size_t b, size_t *out) {
  if (b > SIZE_MAX - a)
    return 0;
  *out = a + b;
  return 1;
}

static int is_aligned_f32(const void *ptr) {
  return (((uintptr_t)ptr) & (sizeof(float) - 1u)) == 0u;
}

static int ranges_overlap(const void *a, size_t a_bytes, const void *b,
                          size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t a0 = (uintptr_t)a;
  const uintptr_t b0 = (uintptr_t)b;
  if (a0 > UINTPTR_MAX - a_bytes || b0 > UINTPTR_MAX - b_bytes)
    return 1;
  const uintptr_t a1 = a0 + a_bytes;
  const uintptr_t b1 = b0 + b_bytes;
  return a0 < b1 && b0 < a1;
}

static int check_finite_f32_array(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!f32_is_finite(buf[i]))
      return 0;
  }
  return 1;
}

/* ---- Internal tree representation (zero hot-path allocations) ---- */

typedef struct radix_node {
  uint32_t parent;
  uint32_t first_child;
  uint32_t next_sibling;
  uint32_t prev_sibling;
  uint32_t num_children;
  uint32_t first_slot;
  uint32_t last_slot;
  uint32_t edge_len;
  uint32_t first_token;
  uint32_t pin_count;
  uint32_t self_pin_count;
  uint32_t generation;
  uint64_t last_access_tick;
  uint8_t alive;
} radix_node;

struct et_radix_kv_tree {
  et_radix_kv_config cfg;
  size_t floats_per_token_side;  /* n_layers * n_kv_heads * head_dim */
  size_t floats_per_token_total; /* 2 * floats_per_token_side */
  size_t floats_per_layer_kv;    /* n_kv_heads * head_dim */
  size_t pool_floats;

  float *kv_pool;            /* [max_kv_tokens, n_layers, 2, n_kv_heads, head_dim] */
  uint32_t *slot_token_id;   /* [max_kv_tokens] */
  uint32_t *slot_next;       /* [max_kv_tokens] */
  uint32_t *slot_owner_node; /* [max_kv_tokens] */
  uint32_t *free_slots;      /* [max_kv_tokens] stack */
  uint32_t free_slot_top;

  radix_node *nodes;    /* [max_nodes] */
  uint32_t *free_nodes; /* [max_nodes] stack */
  uint32_t free_node_top;

  /* Reusable scratch arrays (sized at create time) for non-const tree ops. */
  uint8_t *scratch_mark; /* [max_nodes] */

  uint32_t used_tokens;
  uint32_t pinned_tokens;
  uint32_t active_nodes;

  uint64_t lru_tick;
  uint64_t total_matches;
  uint64_t total_matched_tokens;
  uint64_t total_inserted_tokens;
  uint64_t total_evicted_tokens;
  uint64_t total_evicted_nodes;
  uint64_t total_edge_splits;
  uint64_t total_oom_rejections;
};

static et_radix_kv_handle make_handle(uint32_t idx, uint32_t gen) {
  return (((uint64_t)gen) << 32u) | (uint64_t)idx;
}

static et_radix_kv_status resolve_handle(const et_radix_kv_tree *tree,
                                         et_radix_kv_handle handle,
                                         uint32_t *out_idx) {
  if (tree == NULL || out_idx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (handle == ET_RADIX_KV_HANDLE_INVALID)
    return ET_RADIX_ERR_INVALID_HANDLE;
  const uint32_t idx = (uint32_t)(handle & UINT32_MAX);
  const uint32_t gen = (uint32_t)(handle >> 32u);
  if (idx >= tree->cfg.max_nodes || gen == 0u)
    return ET_RADIX_ERR_INVALID_HANDLE;
  if (!tree->nodes[idx].alive || tree->nodes[idx].generation != gen)
    return ET_RADIX_ERR_INVALID_HANDLE;
  *out_idx = idx;
  return ET_RADIX_OK;
}

static float *slot_layer_ptr(et_radix_kv_tree *tree, uint32_t slot,
                             uint32_t layer, uint32_t is_v) {
  const size_t offset =
      ((size_t)slot * tree->cfg.n_layers + (size_t)layer) * 2u +
      (size_t)is_v;
  return tree->kv_pool + offset * tree->floats_per_layer_kv;
}

static const float *slot_layer_ptr_const(const et_radix_kv_tree *tree,
                                         uint32_t slot, uint32_t layer,
                                         uint32_t is_v) {
  const size_t offset =
      ((size_t)slot * tree->cfg.n_layers + (size_t)layer) * 2u +
      (size_t)is_v;
  return tree->kv_pool + offset * tree->floats_per_layer_kv;
}

static uint32_t find_child_by_token(const et_radix_kv_tree *tree,
                                    uint32_t parent_idx, uint32_t token_id) {
  uint32_t child = tree->nodes[parent_idx].first_child;
  while (child != ET_RADIX_NONE) {
    if (tree->nodes[child].first_token == token_id)
      return child;
    child = tree->nodes[child].next_sibling;
  }
  return ET_RADIX_NONE;
}

static void attach_child(et_radix_kv_tree *tree, uint32_t parent_idx,
                         uint32_t child_idx) {
  radix_node *parent = &tree->nodes[parent_idx];
  radix_node *child = &tree->nodes[child_idx];
  const uint32_t old_first = parent->first_child;
  child->parent = parent_idx;
  child->prev_sibling = ET_RADIX_NONE;
  child->next_sibling = old_first;
  if (old_first != ET_RADIX_NONE)
    tree->nodes[old_first].prev_sibling = child_idx;
  parent->first_child = child_idx;
  parent->num_children++;
}

static void detach_child(et_radix_kv_tree *tree, uint32_t child_idx) {
  radix_node *child = &tree->nodes[child_idx];
  const uint32_t parent_idx = child->parent;
  const uint32_t prev = child->prev_sibling;
  const uint32_t next = child->next_sibling;
  if (prev != ET_RADIX_NONE)
    tree->nodes[prev].next_sibling = next;
  else
    tree->nodes[parent_idx].first_child = next;
  if (next != ET_RADIX_NONE)
    tree->nodes[next].prev_sibling = prev;
  tree->nodes[parent_idx].num_children--;
  child->parent = ET_RADIX_NONE;
  child->prev_sibling = ET_RADIX_NONE;
  child->next_sibling = ET_RADIX_NONE;
}

static uint32_t alloc_node(et_radix_kv_tree *tree) {
  const uint32_t idx = tree->free_nodes[--tree->free_node_top];
  radix_node *n = &tree->nodes[idx];
  const uint32_t gen = n->generation == 0u ? 1u : n->generation;
  memset(n, 0, sizeof(*n));
  n->parent = ET_RADIX_NONE;
  n->first_child = ET_RADIX_NONE;
  n->next_sibling = ET_RADIX_NONE;
  n->prev_sibling = ET_RADIX_NONE;
  n->first_slot = ET_RADIX_NONE;
  n->last_slot = ET_RADIX_NONE;
  n->generation = gen;
  n->alive = 1u;
  tree->active_nodes++;
  return idx;
}

static void free_node(et_radix_kv_tree *tree, uint32_t idx) {
  radix_node *n = &tree->nodes[idx];
  uint32_t next_gen = n->generation + 1u;
  if (next_gen == 0u)
    next_gen = 1u;
  memset(n, 0, sizeof(*n));
  n->parent = ET_RADIX_NONE;
  n->first_child = ET_RADIX_NONE;
  n->next_sibling = ET_RADIX_NONE;
  n->prev_sibling = ET_RADIX_NONE;
  n->first_slot = ET_RADIX_NONE;
  n->last_slot = ET_RADIX_NONE;
  n->generation = next_gen;
  n->alive = 0u;
  tree->active_nodes--;
  tree->free_nodes[tree->free_node_top++] = idx;
}

/* Split node target_idx at 0 < offset < target.edge_len.
 * Allocates a new parent node P holding tokens [0..offset-1] and trims
 * target_idx in place to hold [offset..edge_len-1], preserving target_idx's
 * handle, generation, children, self_pin_count, pin_count, and last_access_tick.
 * Caller must ensure tree->free_node_top >= 1. */
static uint32_t split_edge_at(et_radix_kv_tree *tree, uint32_t target_idx,
                              uint32_t offset) {
  const uint32_t p_idx = alloc_node(tree);
  radix_node *p = &tree->nodes[p_idx];
  radix_node *child = &tree->nodes[target_idx];

  /* Walk offset-1 steps from child->first_slot to locate split boundary. */
  uint32_t split_prev = child->first_slot;
  tree->slot_owner_node[split_prev] = p_idx;
  for (uint32_t i = 1u; i < offset; ++i) {
    split_prev = tree->slot_next[split_prev];
    tree->slot_owner_node[split_prev] = p_idx;
  }
  const uint32_t split_next = tree->slot_next[split_prev];
  tree->slot_next[split_prev] = ET_RADIX_NONE;

  /* Populate new upper parent node P. */
  p->first_slot = child->first_slot;
  p->last_slot = split_prev;
  p->edge_len = offset;
  p->first_token = tree->slot_token_id[p->first_slot];
  p->self_pin_count = 0u;
  p->pin_count = child->pin_count;
  p->last_access_tick = child->last_access_tick;

  /* Trim lower node in place. */
  child->first_slot = split_next;
  child->edge_len -= offset;
  child->first_token = tree->slot_token_id[split_next];

  /* Replace child with P inside old_parent's child list. */
  const uint32_t old_parent = child->parent;
  const uint32_t prev_sib = child->prev_sibling;
  const uint32_t next_sib = child->next_sibling;
  p->parent = old_parent;
  p->prev_sibling = prev_sib;
  p->next_sibling = next_sib;
  if (prev_sib != ET_RADIX_NONE)
    tree->nodes[prev_sib].next_sibling = p_idx;
  else
    tree->nodes[old_parent].first_child = p_idx;
  if (next_sib != ET_RADIX_NONE)
    tree->nodes[next_sib].prev_sibling = p_idx;

  /* Attach trimmed lower node as P's sole child. */
  child->parent = p_idx;
  child->prev_sibling = ET_RADIX_NONE;
  child->next_sibling = ET_RADIX_NONE;
  p->first_child = target_idx;
  p->num_children = 1u;

  tree->total_edge_splits++;
  return p_idx;
}

static void touch_path_lru(et_radix_kv_tree *tree, uint32_t node_idx) {
  const uint64_t tick = ++tree->lru_tick;
  uint32_t cur = node_idx;
  while (cur != ET_RADIX_NONE) {
    tree->nodes[cur].last_access_tick = tick;
    cur = tree->nodes[cur].parent;
  }
}

static void mark_ancestor_path(et_radix_kv_tree *tree, uint32_t node_idx) {
  memset(tree->scratch_mark, 0, tree->cfg.max_nodes);
  uint32_t cur = node_idx;
  while (cur != ET_RADIX_NONE) {
    tree->scratch_mark[cur] = 1u;
    cur = tree->nodes[cur].parent;
  }
}

/* Pick deterministic LRU unpinned leaf not marked in scratch_mark.
 * Tie-break strictly by (last_access_tick ASC, node_idx ASC). */
static uint32_t select_lru_eviction_victim(const et_radix_kv_tree *tree) {
  uint32_t best_idx = ET_RADIX_NONE;
  uint64_t best_tick = UINT64_MAX;
  for (uint32_t i = 1u; i < tree->cfg.max_nodes; ++i) {
    const radix_node *n = &tree->nodes[i];
    if (!n->alive || n->num_children != 0u || n->pin_count != 0u ||
        tree->scratch_mark[i] != 0u)
      continue;
    if (best_idx == ET_RADIX_NONE || n->last_access_tick < best_tick) {
      best_idx = i;
      best_tick = n->last_access_tick;
    }
  }
  return best_idx;
}

/* Evict one unpinned leaf node, releasing all of its physical KV slots. */
static uint32_t evict_leaf_node(et_radix_kv_tree *tree, uint32_t leaf_idx) {
  radix_node *leaf = &tree->nodes[leaf_idx];
  const uint32_t evicted_tokens = leaf->edge_len;

  uint32_t s = leaf->first_slot;
  while (s != ET_RADIX_NONE) {
    const uint32_t next_s = tree->slot_next[s];
    tree->slot_owner_node[s] = ET_RADIX_NONE;
    tree->slot_next[s] = ET_RADIX_NONE;
    tree->slot_token_id[s] = 0u;
    tree->free_slots[tree->free_slot_top++] = s;
    s = next_s;
  }
  tree->used_tokens -= evicted_tokens;
  tree->total_evicted_tokens += evicted_tokens;
  tree->total_evicted_nodes++;

  detach_child(tree, leaf_idx);
  free_node(tree, leaf_idx);
  return evicted_tokens;
}

/* Read-only longest prefix match starting at start_node_idx. */
static void walk_match_readonly(const et_radix_kv_tree *tree,
                                uint32_t start_node_idx,
                                const uint32_t *tokens, uint32_t seq_len,
                                uint32_t *out_matched_len,
                                uint32_t *out_node_idx,
                                uint32_t *out_edge_offset) {
  uint32_t cur = start_node_idx;
  uint32_t pos = 0u;
  uint32_t edge_offset = 0u;

  while (pos < seq_len) {
    const uint32_t child = find_child_by_token(tree, cur, tokens[pos]);
    if (child == ET_RADIX_NONE)
      break;
    const radix_node *cn = &tree->nodes[child];
    uint32_t k = 0u;
    uint32_t s = cn->first_slot;
    while (k < cn->edge_len && (pos + k) < seq_len && s != ET_RADIX_NONE) {
      if (tree->slot_token_id[s] != tokens[pos + k])
        break;
      s = tree->slot_next[s];
      k++;
    }
    pos += k;
    if (k == cn->edge_len) {
      cur = child;
      edge_offset = 0u;
    } else {
      cur = child;
      edge_offset = k;
      break;
    }
  }

  *out_matched_len = pos;
  *out_node_idx = cur;
  *out_edge_offset = edge_offset;
}

/* Gather all physical KV slot indices along root -> node_idx in forward order. */
static et_radix_kv_status gather_path_slots(const et_radix_kv_tree *tree,
                                            uint32_t node_idx,
                                            uint32_t *out_slots,
                                            uint32_t slots_cap,
                                            uint32_t *out_len) {
  uint32_t node_stack[ET_RADIX_KV_MAX_NODES];
  uint32_t depth = 0u;
  uint32_t total_tokens = 0u;
  uint32_t cur = node_idx;

  while (cur != 0u) {
    if (cur == ET_RADIX_NONE || depth >= tree->cfg.max_nodes)
      return ET_RADIX_ERR_INVARIANT;
    node_stack[depth++] = cur;
    total_tokens += tree->nodes[cur].edge_len;
    cur = tree->nodes[cur].parent;
  }

  if (out_slots != NULL) {
    if (slots_cap < total_tokens)
      return ET_RADIX_ERR_SHAPE;
    uint32_t pos = 0u;
    while (depth > 0u) {
      const uint32_t nidx = node_stack[--depth];
      uint32_t s = tree->nodes[nidx].first_slot;
      while (s != ET_RADIX_NONE) {
        out_slots[pos++] = s;
        s = tree->slot_next[s];
      }
    }
    if (pos != total_tokens)
      return ET_RADIX_ERR_INVARIANT;
  }

  if (out_len != NULL)
    *out_len = total_tokens;
  return ET_RADIX_OK;
}

/* ---- Public configuration & lifecycle APIs ---- */

et_radix_kv_status et_radix_kv_validate_config(const et_radix_kv_config *cfg) {
  if (!fp_environment_supported())
    return ET_RADIX_ERR_FP_ENV;
  if (cfg == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (cfg->reserved != 0u)
    return ET_RADIX_ERR_ARGUMENT;
  if (cfg->max_kv_tokens < 1u || cfg->max_kv_tokens > ET_RADIX_KV_MAX_TOKENS ||
      cfg->max_nodes < 2u || cfg->max_nodes > ET_RADIX_KV_MAX_NODES ||
      cfg->n_layers < 1u || cfg->n_layers > ET_RADIX_KV_MAX_LAYERS ||
      cfg->n_kv_heads < 1u || cfg->n_kv_heads > ET_RADIX_KV_MAX_KV_HEADS ||
      cfg->head_dim < 1u || cfg->head_dim > ET_RADIX_KV_MAX_HEAD_DIM)
    return ET_RADIX_ERR_SHAPE;

  size_t count = 0u;
  return et_radix_kv_pool_float_count(cfg, &count);
}

et_radix_kv_status et_radix_kv_pool_float_count(const et_radix_kv_config *cfg,
                                                size_t *out_count) {
  if (cfg == NULL || out_count == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (cfg->reserved != 0u || cfg->max_kv_tokens < 1u ||
      cfg->max_kv_tokens > ET_RADIX_KV_MAX_TOKENS || cfg->max_nodes < 2u ||
      cfg->max_nodes > ET_RADIX_KV_MAX_NODES || cfg->n_layers < 1u ||
      cfg->n_layers > ET_RADIX_KV_MAX_LAYERS || cfg->n_kv_heads < 1u ||
      cfg->n_kv_heads > ET_RADIX_KV_MAX_KV_HEADS || cfg->head_dim < 1u ||
      cfg->head_dim > ET_RADIX_KV_MAX_HEAD_DIM)
    return ET_RADIX_ERR_SHAPE;

  size_t total = (size_t)cfg->max_kv_tokens;
  if (!mul_size(total, (size_t)cfg->n_layers, &total) ||
      !mul_size(total, 2u, &total) ||
      !mul_size(total, (size_t)cfg->n_kv_heads, &total) ||
      !mul_size(total, (size_t)cfg->head_dim, &total))
    return ET_RADIX_ERR_OVERFLOW;

  if ((uint64_t)total > ET_RADIX_KV_MAX_POOL_FLOATS)
    return ET_RADIX_ERR_SHAPE;

  size_t bytes = 0u;
  if (!mul_size(total, sizeof(float), &bytes) || bytes > (size_t)PTRDIFF_MAX)
    return ET_RADIX_ERR_OVERFLOW;

  *out_count = total;
  return ET_RADIX_OK;
}

static void init_tree_state(et_radix_kv_tree *tree) {
  memset(tree->kv_pool, 0, tree->pool_floats * sizeof(float));
  for (uint32_t s = 0u; s < tree->cfg.max_kv_tokens; ++s) {
    tree->slot_token_id[s] = 0u;
    tree->slot_next[s] = ET_RADIX_NONE;
    tree->slot_owner_node[s] = ET_RADIX_NONE;
    tree->free_slots[s] = (tree->cfg.max_kv_tokens - 1u) - s;
  }
  tree->free_slot_top = tree->cfg.max_kv_tokens;

  for (uint32_t i = 0u; i < tree->cfg.max_nodes; ++i) {
    radix_node *n = &tree->nodes[i];
    memset(n, 0, sizeof(*n));
    n->parent = ET_RADIX_NONE;
    n->first_child = ET_RADIX_NONE;
    n->next_sibling = ET_RADIX_NONE;
    n->prev_sibling = ET_RADIX_NONE;
    n->first_slot = ET_RADIX_NONE;
    n->last_slot = ET_RADIX_NONE;
    n->generation = 1u;
    n->alive = 0u;
  }
  tree->nodes[0].alive = 1u;
  tree->nodes[0].generation = 1u;
  tree->nodes[0].last_access_tick = 0u;

  tree->free_node_top = 0u;
  for (uint32_t i = tree->cfg.max_nodes - 1u; i >= 1u; --i)
    tree->free_nodes[tree->free_node_top++] = i;

  memset(tree->scratch_mark, 0, tree->cfg.max_nodes);

  tree->used_tokens = 0u;
  tree->pinned_tokens = 0u;
  tree->active_nodes = 1u;
  tree->lru_tick = 0u;
  tree->total_matches = 0u;
  tree->total_matched_tokens = 0u;
  tree->total_inserted_tokens = 0u;
  tree->total_evicted_tokens = 0u;
  tree->total_evicted_nodes = 0u;
  tree->total_edge_splits = 0u;
  tree->total_oom_rejections = 0u;
}

et_radix_kv_status et_radix_kv_create(const et_radix_kv_config *cfg,
                                      et_radix_kv_tree **out_tree) {
  if (out_tree == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  *out_tree = NULL;

  et_radix_kv_status st = et_radix_kv_validate_config(cfg);
  if (st != ET_RADIX_OK)
    return st;

  size_t pool_floats = 0u;
  st = et_radix_kv_pool_float_count(cfg, &pool_floats);
  if (st != ET_RADIX_OK)
    return st;

  et_radix_kv_tree *tree =
      (et_radix_kv_tree *)calloc(1u, sizeof(et_radix_kv_tree));
  if (tree == NULL)
    return ET_RADIX_ERR_NOMEM;

  tree->cfg = *cfg;
  tree->floats_per_layer_kv = (size_t)cfg->n_kv_heads * (size_t)cfg->head_dim;
  tree->floats_per_token_side =
      (size_t)cfg->n_layers * tree->floats_per_layer_kv;
  tree->floats_per_token_total = 2u * tree->floats_per_token_side;
  tree->pool_floats = pool_floats;

  tree->kv_pool = (float *)calloc(pool_floats, sizeof(float));
  tree->slot_token_id =
      (uint32_t *)calloc(cfg->max_kv_tokens, sizeof(uint32_t));
  tree->slot_next = (uint32_t *)calloc(cfg->max_kv_tokens, sizeof(uint32_t));
  tree->slot_owner_node =
      (uint32_t *)calloc(cfg->max_kv_tokens, sizeof(uint32_t));
  tree->free_slots = (uint32_t *)calloc(cfg->max_kv_tokens, sizeof(uint32_t));
  tree->nodes = (radix_node *)calloc(cfg->max_nodes, sizeof(radix_node));
  tree->free_nodes = (uint32_t *)calloc(cfg->max_nodes, sizeof(uint32_t));
  tree->scratch_mark = (uint8_t *)calloc(cfg->max_nodes, sizeof(uint8_t));

  if (tree->kv_pool == NULL || tree->slot_token_id == NULL ||
      tree->slot_next == NULL || tree->slot_owner_node == NULL ||
      tree->free_slots == NULL || tree->nodes == NULL ||
      tree->free_nodes == NULL || tree->scratch_mark == NULL) {
    et_radix_kv_destroy(tree);
    return ET_RADIX_ERR_NOMEM;
  }

  init_tree_state(tree);
  *out_tree = tree;
  return ET_RADIX_OK;
}

void et_radix_kv_destroy(et_radix_kv_tree *tree) {
  if (tree == NULL)
    return;
  free(tree->kv_pool);
  free(tree->slot_token_id);
  free(tree->slot_next);
  free(tree->slot_owner_node);
  free(tree->free_slots);
  free(tree->nodes);
  free(tree->free_nodes);
  free(tree->scratch_mark);
  free(tree);
}

et_radix_kv_status et_radix_kv_reset(et_radix_kv_tree *tree) {
  if (tree == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  init_tree_state(tree);
  return ET_RADIX_OK;
}

et_radix_kv_handle et_radix_kv_root_handle(const et_radix_kv_tree *tree) {
  if (tree == NULL)
    return ET_RADIX_KV_HANDLE_INVALID;
  return make_handle(0u, tree->nodes[0].generation);
}

et_radix_kv_status et_radix_kv_stats_get(const et_radix_kv_tree *tree,
                                         et_radix_kv_stats *out_stats) {
  if (tree == NULL || out_stats == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  out_stats->max_kv_tokens = tree->cfg.max_kv_tokens;
  out_stats->used_tokens = tree->used_tokens;
  out_stats->free_tokens = tree->free_slot_top;
  out_stats->pinned_tokens = tree->pinned_tokens;
  out_stats->evictable_tokens = tree->used_tokens - tree->pinned_tokens;
  out_stats->max_nodes = tree->cfg.max_nodes;
  out_stats->active_nodes = tree->active_nodes;
  out_stats->free_nodes = tree->free_node_top;
  out_stats->lru_tick = tree->lru_tick;
  out_stats->total_matches = tree->total_matches;
  out_stats->total_matched_tokens = tree->total_matched_tokens;
  out_stats->total_inserted_tokens = tree->total_inserted_tokens;
  out_stats->total_evicted_tokens = tree->total_evicted_tokens;
  out_stats->total_evicted_nodes = tree->total_evicted_nodes;
  out_stats->total_edge_splits = tree->total_edge_splits;
  out_stats->total_oom_rejections = tree->total_oom_rejections;
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_config_get(const et_radix_kv_tree *tree,
                                          et_radix_kv_config *out_cfg) {
  if (tree == NULL || out_cfg == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  *out_cfg = tree->cfg;
  return ET_RADIX_OK;
}

/* ---- Prefix matching, pinning, insertion, and LRU eviction ---- */

et_radix_kv_status et_radix_kv_match_prefix(et_radix_kv_tree *tree,
                                            const uint32_t *tokens,
                                            uint32_t seq_len,
                                            et_radix_kv_match_result *out_match,
                                            uint32_t *out_kv_indices,
                                            uint32_t kv_indices_cap) {
  if (tree == NULL || out_match == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (seq_len > 0u && tokens == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (out_kv_indices == NULL && kv_indices_cap != 0u)
    return ET_RADIX_ERR_ARGUMENT;

  uint32_t matched_len = 0u;
  uint32_t cur = 0u;
  uint32_t edge_offset = 0u;
  walk_match_readonly(tree, 0u, tokens, seq_len, &matched_len, &cur,
                      &edge_offset);

  if (out_kv_indices != NULL && kv_indices_cap < matched_len)
    return ET_RADIX_ERR_SHAPE;

  uint32_t did_split = 0u;
  if (edge_offset > 0u) {
    /* Need 1 node to split cur at edge_offset; protect [root .. cur]. */
    mark_ancestor_path(tree, cur);
    if (tree->free_node_top == 0u) {
      const uint32_t victim = select_lru_eviction_victim(tree);
      if (victim == ET_RADIX_NONE) {
        tree->total_oom_rejections++;
        return ET_RADIX_ERR_OOM;
      }
      (void)evict_leaf_node(tree, victim);
    }
    cur = split_edge_at(tree, cur, edge_offset);
    did_split = 1u;
  }

  touch_path_lru(tree, cur);
  tree->total_matches++;
  tree->total_matched_tokens += matched_len;

  if (out_kv_indices != NULL) {
    uint32_t gathered = 0u;
    const et_radix_kv_status st =
        gather_path_slots(tree, cur, out_kv_indices, kv_indices_cap, &gathered);
    if (st != ET_RADIX_OK || gathered != matched_len)
      return ET_RADIX_ERR_INVARIANT;
  }

  out_match->handle = make_handle(cur, tree->nodes[cur].generation);
  out_match->matched_len = matched_len;
  out_match->did_split = did_split;
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_pin(et_radix_kv_tree *tree,
                                   et_radix_kv_handle handle) {
  uint32_t idx = 0u;
  const et_radix_kv_status st = resolve_handle(tree, handle, &idx);
  if (st != ET_RADIX_OK)
    return st;

  /* Overflow preflight before mutating any refcount. */
  if (tree->nodes[idx].self_pin_count == UINT32_MAX)
    return ET_RADIX_ERR_OVERFLOW;
  uint32_t cur = idx;
  while (cur != ET_RADIX_NONE) {
    if (tree->nodes[cur].pin_count == UINT32_MAX)
      return ET_RADIX_ERR_OVERFLOW;
    cur = tree->nodes[cur].parent;
  }

  tree->nodes[idx].self_pin_count++;
  cur = idx;
  while (cur != ET_RADIX_NONE) {
    if (cur != 0u && tree->nodes[cur].pin_count == 0u)
      tree->pinned_tokens += tree->nodes[cur].edge_len;
    tree->nodes[cur].pin_count++;
    cur = tree->nodes[cur].parent;
  }
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_unpin(et_radix_kv_tree *tree,
                                     et_radix_kv_handle handle) {
  uint32_t idx = 0u;
  const et_radix_kv_status st = resolve_handle(tree, handle, &idx);
  if (st != ET_RADIX_OK)
    return st;

  if (tree->nodes[idx].self_pin_count == 0u)
    return ET_RADIX_ERR_DOUBLE_UNPIN;

  /* Consistency preflight along ancestor path before mutation. */
  uint32_t cur = idx;
  while (cur != ET_RADIX_NONE) {
    if (tree->nodes[cur].pin_count == 0u)
      return ET_RADIX_ERR_INVARIANT;
    cur = tree->nodes[cur].parent;
  }

  tree->nodes[idx].self_pin_count--;
  cur = idx;
  while (cur != ET_RADIX_NONE) {
    tree->nodes[cur].pin_count--;
    if (cur != 0u && tree->nodes[cur].pin_count == 0u)
      tree->pinned_tokens -= tree->nodes[cur].edge_len;
    cur = tree->nodes[cur].parent;
  }
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_handle_info(const et_radix_kv_tree *tree,
                                           et_radix_kv_handle handle,
                                           uint32_t *out_depth_tokens,
                                           uint32_t *out_pin_count,
                                           uint32_t *out_self_pin_count,
                                           uint64_t *out_last_tick) {
  uint32_t idx = 0u;
  const et_radix_kv_status st = resolve_handle(tree, handle, &idx);
  if (st != ET_RADIX_OK)
    return st;

  if (out_depth_tokens != NULL) {
    uint32_t depth = 0u;
    const et_radix_kv_status gst =
        gather_path_slots(tree, idx, NULL, 0u, &depth);
    if (gst != ET_RADIX_OK)
      return gst;
    *out_depth_tokens = depth;
  }
  if (out_pin_count != NULL)
    *out_pin_count = tree->nodes[idx].pin_count;
  if (out_self_pin_count != NULL)
    *out_self_pin_count = tree->nodes[idx].self_pin_count;
  if (out_last_tick != NULL)
    *out_last_tick = tree->nodes[idx].last_access_tick;
  return ET_RADIX_OK;
}

static et_radix_kv_status insert_under_internal(
    et_radix_kv_tree *tree, uint32_t start_node_idx, const uint32_t *tokens,
    uint32_t seq_len, const float *k_data, const float *v_data,
    et_radix_kv_handle *out_handle, uint32_t *out_matched_len,
    uint32_t *out_inserted_len) {
  if (!fp_environment_supported())
    return ET_RADIX_ERR_FP_ENV;
  if (tree == NULL || tokens == NULL || k_data == NULL || v_data == NULL ||
      out_handle == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (seq_len == 0u || seq_len > tree->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;
  if (!is_aligned_f32(k_data) || !is_aligned_f32(v_data))
    return ET_RADIX_ERR_ALIGNMENT;

  size_t total_kv_floats = 0u;
  size_t total_kv_bytes = 0u;
  if (!mul_size((size_t)seq_len, tree->floats_per_token_side,
                &total_kv_floats) ||
      !mul_size(total_kv_floats, sizeof(float), &total_kv_bytes))
    return ET_RADIX_ERR_OVERFLOW;

  const size_t pool_bytes = tree->pool_floats * sizeof(float);
  if (ranges_overlap(k_data, total_kv_bytes, tree->kv_pool, pool_bytes) ||
      ranges_overlap(v_data, total_kv_bytes, tree->kv_pool, pool_bytes))
    return ET_RADIX_ERR_ALIAS;

  /* Validate finiteness of ALL input K/V floats before any mutation. */
  if (!check_finite_f32_array(k_data, total_kv_floats) ||
      !check_finite_f32_array(v_data, total_kv_floats))
    return ET_RADIX_ERR_NONFINITE;

  /* Read-only longest prefix match starting at start_node_idx. */
  uint32_t matched_len = 0u;
  uint32_t cur = start_node_idx;
  uint32_t edge_offset = 0u;
  walk_match_readonly(tree, start_node_idx, tokens, seq_len, &matched_len, &cur,
                      &edge_offset);

  const uint32_t suffix_len = seq_len - matched_len;
  const uint32_t need_split = edge_offset > 0u ? 1u : 0u;
  const uint32_t need_leaf = suffix_len > 0u ? 1u : 0u;
  const uint32_t needed_nodes = need_split + need_leaf;
  const uint32_t needed_tokens = suffix_len;

  /* Exact atomic preflight BEFORE mutating any tree state. */
  mark_ancestor_path(tree, cur);

  uint32_t outside_evictable_nodes = 0u;
  uint32_t outside_evictable_tokens = 0u;
  for (uint32_t i = 1u; i < tree->cfg.max_nodes; ++i) {
    const radix_node *n = &tree->nodes[i];
    if (!n->alive || n->pin_count != 0u || tree->scratch_mark[i] != 0u)
      continue;
    outside_evictable_nodes++;
    outside_evictable_tokens += n->edge_len;
  }

  /* If splitting an unpinned edge and inserting a suffix, the lower trimmed
   * half of cur becomes an unpinned non-protected node eligible for eviction
   * after the split creates the upper parent node P. */
  uint32_t tail_evictable_nodes = 0u;
  uint32_t tail_evictable_tokens = 0u;
  if (need_split && need_leaf && tree->nodes[cur].pin_count == 0u) {
    tail_evictable_nodes = 1u;
    tail_evictable_tokens = tree->nodes[cur].edge_len - edge_offset;
  }

  if ((need_split &&
       (tree->free_node_top + outside_evictable_nodes < 1u)) ||
      (tree->free_node_top + outside_evictable_nodes + tail_evictable_nodes <
       needed_nodes) ||
      (tree->free_slot_top + outside_evictable_tokens + tail_evictable_tokens <
       needed_tokens)) {
    tree->total_oom_rejections++;
    return ET_RADIX_ERR_OOM;
  }

  /* Mutation phase (mathematically guaranteed to succeed by preflight). */
  if (need_split) {
    while (tree->free_node_top < 1u) {
      const uint32_t victim = select_lru_eviction_victim(tree);
      if (victim == ET_RADIX_NONE)
        return ET_RADIX_ERR_INVARIANT;
      (void)evict_leaf_node(tree, victim);
    }
    cur = split_edge_at(tree, cur, edge_offset);
    /* Recompute protected set to [root .. new_parent_cur], exposing the
     * lower half of the split node to normal LRU eviction if unpinned. */
    mark_ancestor_path(tree, cur);
  }

  while (tree->free_slot_top < needed_tokens ||
         tree->free_node_top < need_leaf) {
    const uint32_t victim = select_lru_eviction_victim(tree);
    if (victim == ET_RADIX_NONE)
      return ET_RADIX_ERR_INVARIANT;
    (void)evict_leaf_node(tree, victim);
  }

  if (need_leaf) {
    const uint32_t leaf_idx = alloc_node(tree);
    radix_node *leaf = &tree->nodes[leaf_idx];
    uint32_t prev_slot = ET_RADIX_NONE;

    for (uint32_t i = 0u; i < suffix_len; ++i) {
      const uint32_t slot = tree->free_slots[--tree->free_slot_top];
      const uint32_t src_tok = matched_len + i;
      tree->slot_token_id[slot] = tokens[src_tok];
      tree->slot_next[slot] = ET_RADIX_NONE;
      tree->slot_owner_node[slot] = leaf_idx;

      if (prev_slot == ET_RADIX_NONE)
        leaf->first_slot = slot;
      else
        tree->slot_next[prev_slot] = slot;
      prev_slot = slot;

      const float *src_k = k_data + (size_t)src_tok * tree->floats_per_token_side;
      const float *src_v = v_data + (size_t)src_tok * tree->floats_per_token_side;
      for (uint32_t l = 0u; l < tree->cfg.n_layers; ++l) {
        const size_t layer_off = (size_t)l * tree->floats_per_layer_kv;
        memcpy(slot_layer_ptr(tree, slot, l, 0u), src_k + layer_off,
               tree->floats_per_layer_kv * sizeof(float));
        memcpy(slot_layer_ptr(tree, slot, l, 1u), src_v + layer_off,
               tree->floats_per_layer_kv * sizeof(float));
      }
    }

    leaf->last_slot = prev_slot;
    leaf->edge_len = suffix_len;
    leaf->first_token = tokens[matched_len];
    leaf->pin_count = 0u;
    leaf->self_pin_count = 0u;

    attach_child(tree, cur, leaf_idx);
    tree->used_tokens += suffix_len;
    tree->total_inserted_tokens += suffix_len;
    cur = leaf_idx;
  }

  touch_path_lru(tree, cur);

  *out_handle = make_handle(cur, tree->nodes[cur].generation);
  if (out_matched_len != NULL)
    *out_matched_len = matched_len;
  if (out_inserted_len != NULL)
    *out_inserted_len = suffix_len;
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_insert(et_radix_kv_tree *tree,
                                      const uint32_t *tokens, uint32_t seq_len,
                                      const float *k_data, const float *v_data,
                                      et_radix_kv_handle *out_handle,
                                      uint32_t *out_matched_prefix_len) {
  return insert_under_internal(tree, 0u, tokens, seq_len, k_data, v_data,
                               out_handle, out_matched_prefix_len, NULL);
}

et_radix_kv_status et_radix_kv_insert_suffix(
    et_radix_kv_tree *tree, et_radix_kv_handle prefix_handle,
    const uint32_t *suffix_tokens, uint32_t suffix_len, const float *suffix_k,
    const float *suffix_v, et_radix_kv_handle *out_handle,
    uint32_t *out_inserted_tokens) {
  uint32_t start_idx = 0u;
  const et_radix_kv_status st = resolve_handle(tree, prefix_handle, &start_idx);
  if (st != ET_RADIX_OK)
    return st;
  return insert_under_internal(tree, start_idx, suffix_tokens, suffix_len,
                               suffix_k, suffix_v, out_handle, NULL,
                               out_inserted_tokens);
}

et_radix_kv_status et_radix_kv_evict_lru(et_radix_kv_tree *tree,
                                         uint32_t target_free_tokens,
                                         uint32_t *out_evicted_tokens) {
  if (tree == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (target_free_tokens > tree->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;

  memset(tree->scratch_mark, 0, tree->cfg.max_nodes);
  uint32_t evicted = 0u;
  while (tree->free_slot_top < target_free_tokens) {
    const uint32_t victim = select_lru_eviction_victim(tree);
    if (victim == ET_RADIX_NONE)
      break;
    evicted += evict_leaf_node(tree, victim);
  }
  if (out_evicted_tokens != NULL)
    *out_evicted_tokens = evicted;
  return ET_RADIX_OK;
}

/* ---- KV gather and exact binary64 causal GQA attention ---- */

et_radix_kv_status et_radix_kv_read_kv(const et_radix_kv_tree *tree,
                                       et_radix_kv_handle handle,
                                       uint32_t layer, float *out_k,
                                       float *out_v, size_t out_floats,
                                       uint32_t *out_seq_len) {
  uint32_t idx = 0u;
  et_radix_kv_status st = resolve_handle(tree, handle, &idx);
  if (st != ET_RADIX_OK)
    return st;
  if (layer >= tree->cfg.n_layers)
    return ET_RADIX_ERR_SHAPE;

  uint32_t slots[ET_RADIX_KV_MAX_TOKENS];
  uint32_t path_len = 0u;
  st = gather_path_slots(tree, idx, slots, ET_RADIX_KV_MAX_TOKENS, &path_len);
  if (st != ET_RADIX_OK)
    return st;

  if (path_len == 0u) {
    if (out_seq_len != NULL)
      *out_seq_len = 0u;
    return ET_RADIX_OK;
  }

  if (out_k == NULL || out_v == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (!is_aligned_f32(out_k) || !is_aligned_f32(out_v))
    return ET_RADIX_ERR_ALIGNMENT;

  size_t needed_floats = 0u;
  size_t needed_bytes = 0u;
  if (!mul_size((size_t)path_len, tree->floats_per_layer_kv, &needed_floats) ||
      !mul_size(needed_floats, sizeof(float), &needed_bytes))
    return ET_RADIX_ERR_OVERFLOW;
  if (out_floats < needed_floats)
    return ET_RADIX_ERR_SHAPE;

  const size_t pool_bytes = tree->pool_floats * sizeof(float);
  if (ranges_overlap(out_k, needed_bytes, out_v, needed_bytes) ||
      ranges_overlap(out_k, needed_bytes, tree->kv_pool, pool_bytes) ||
      ranges_overlap(out_v, needed_bytes, tree->kv_pool, pool_bytes))
    return ET_RADIX_ERR_ALIAS;

  for (uint32_t t = 0u; t < path_len; ++t) {
    const size_t dst_off = (size_t)t * tree->floats_per_layer_kv;
    memcpy(out_k + dst_off, slot_layer_ptr_const(tree, slots[t], layer, 0u),
           tree->floats_per_layer_kv * sizeof(float));
    memcpy(out_v + dst_off, slot_layer_ptr_const(tree, slots[t], layer, 1u),
           tree->floats_per_layer_kv * sizeof(float));
  }

  if (out_seq_len != NULL)
    *out_seq_len = path_len;
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_attend_gqa_f32(
    const et_radix_kv_tree *tree, et_radix_kv_handle prefix_handle,
    uint32_t layer, uint32_t n_q_heads, uint32_t q_len, uint32_t suffix_len,
    float scale, const float *q, const float *suffix_k, const float *suffix_v,
    float *out) {
  if (!fp_environment_supported())
    return ET_RADIX_ERR_FP_ENV;

  uint32_t idx = 0u;
  et_radix_kv_status st = resolve_handle(tree, prefix_handle, &idx);
  if (st != ET_RADIX_OK)
    return st;
  if (layer >= tree->cfg.n_layers)
    return ET_RADIX_ERR_SHAPE;
  if (n_q_heads < 1u || n_q_heads > ET_RADIX_KV_MAX_Q_HEADS ||
      (n_q_heads % tree->cfg.n_kv_heads) != 0u)
    return ET_RADIX_ERR_SHAPE;
  if (!f32_is_finite(scale) || scale <= 0.0f)
    return ET_RADIX_ERR_NONFINITE;
  if (q_len == 0u || q == NULL || out == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (suffix_len == 0u) {
    if (suffix_k != NULL || suffix_v != NULL)
      return ET_RADIX_ERR_ARGUMENT;
  } else {
    if (suffix_k == NULL || suffix_v == NULL)
      return ET_RADIX_ERR_ARGUMENT;
  }

  uint32_t prefix_slots[ET_RADIX_KV_MAX_TOKENS];
  uint32_t prefix_len = 0u;
  st = gather_path_slots(tree, idx, prefix_slots, ET_RADIX_KV_MAX_TOKENS,
                         &prefix_len);
  if (st != ET_RADIX_OK)
    return st;

  if (suffix_len > UINT32_MAX - prefix_len)
    return ET_RADIX_ERR_OVERFLOW;
  const uint32_t total_len = prefix_len + suffix_len;
  if (total_len == 0u || q_len > total_len)
    return ET_RADIX_ERR_SHAPE;

  const uint32_t hd = tree->cfg.head_dim;
  const uint32_t hkv = tree->cfg.n_kv_heads;
  const uint32_t group_size = n_q_heads / hkv;

  size_t q_floats = 0u;
  size_t q_bytes = 0u;
  if (!mul_size((size_t)q_len, (size_t)n_q_heads, &q_floats) ||
      !mul_size(q_floats, (size_t)hd, &q_floats) ||
      !mul_size(q_floats, sizeof(float), &q_bytes))
    return ET_RADIX_ERR_OVERFLOW;

  size_t suf_floats = 0u;
  size_t suf_bytes = 0u;
  if (!mul_size((size_t)suffix_len, (size_t)hkv, &suf_floats) ||
      !mul_size(suf_floats, (size_t)hd, &suf_floats) ||
      !mul_size(suf_floats, sizeof(float), &suf_bytes))
    return ET_RADIX_ERR_OVERFLOW;

  if (!is_aligned_f32(q) || !is_aligned_f32(out) ||
      (suffix_len > 0u &&
       (!is_aligned_f32(suffix_k) || !is_aligned_f32(suffix_v))))
    return ET_RADIX_ERR_ALIGNMENT;

  const size_t pool_bytes = tree->pool_floats * sizeof(float);
  if (ranges_overlap(out, q_bytes, q, q_bytes) ||
      ranges_overlap(out, q_bytes, tree->kv_pool, pool_bytes) ||
      (suffix_len > 0u &&
       (ranges_overlap(out, q_bytes, suffix_k, suf_bytes) ||
        ranges_overlap(out, q_bytes, suffix_v, suf_bytes))))
    return ET_RADIX_ERR_ALIAS;

  if (!check_finite_f32_array(q, q_floats))
    return ET_RADIX_ERR_NONFINITE;
  if (suffix_len > 0u &&
      (!check_finite_f32_array(suffix_k, suf_floats) ||
       !check_finite_f32_array(suffix_v, suf_floats)))
    return ET_RADIX_ERR_NONFINITE;

  /* Check finiteness of cached prefix K/V before writing output. */
  for (uint32_t t = 0u; t < prefix_len; ++t) {
    const float *kp = slot_layer_ptr_const(tree, prefix_slots[t], layer, 0u);
    const float *vp = slot_layer_ptr_const(tree, prefix_slots[t], layer, 1u);
    if (!check_finite_f32_array(kp, tree->floats_per_layer_kv) ||
        !check_finite_f32_array(vp, tree->floats_per_layer_kv))
      return ET_RADIX_ERR_NONFINITE;
  }

  /* Compute exact single-pass online softmax in IEEE-754 binary64.
   * For each query qi, abs_pos = (total_len - q_len) + qi. */
  const double d_scale = (double)scale;
  const uint32_t q_base_pos = total_len - q_len;
  double acc[ET_RADIX_KV_MAX_HEAD_DIM];
  float staged_row[ET_RADIX_KV_MAX_HEAD_DIM];

  for (uint32_t qi = 0u; qi < q_len; ++qi) {
    const uint32_t abs_pos = q_base_pos + qi;
    for (uint32_t qh = 0u; qh < n_q_heads; ++qh) {
      const uint32_t kv_h = qh / group_size;
      const float *q_row = q + ((size_t)qi * n_q_heads + (size_t)qh) * hd;

      double running_max = 0.0;
      double running_sum = 0.0;
      for (uint32_t d = 0u; d < hd; ++d)
        acc[d] = 0.0;

      for (uint32_t t = 0u; t <= abs_pos; ++t) {
        const float *k_row = NULL;
        const float *v_row = NULL;
        if (t < prefix_len) {
          k_row = slot_layer_ptr_const(tree, prefix_slots[t], layer, 0u) +
                  (size_t)kv_h * hd;
          v_row = slot_layer_ptr_const(tree, prefix_slots[t], layer, 1u) +
                  (size_t)kv_h * hd;
        } else {
          const size_t st_idx = (size_t)(t - prefix_len);
          k_row = suffix_k + (st_idx * hkv + (size_t)kv_h) * hd;
          v_row = suffix_v + (st_idx * hkv + (size_t)kv_h) * hd;
        }

        double dot = 0.0;
        for (uint32_t d = 0u; d < hd; ++d)
          dot += (double)q_row[d] * (double)k_row[d];
        const double score = dot * d_scale;
        if (!isfinite(score))
          return ET_RADIX_ERR_NONFINITE;

        if (t == 0u) {
          running_max = score;
          running_sum = 1.0;
          for (uint32_t d = 0u; d < hd; ++d)
            acc[d] = (double)v_row[d];
        } else if (score > running_max) {
          const double rescale = exp(running_max - score);
          running_sum = running_sum * rescale + 1.0;
          for (uint32_t d = 0u; d < hd; ++d)
            acc[d] = acc[d] * rescale + (double)v_row[d];
          running_max = score;
        } else {
          const double w = exp(score - running_max);
          running_sum += w;
          for (uint32_t d = 0u; d < hd; ++d)
            acc[d] += w * (double)v_row[d];
        }
      }

      if (!isfinite(running_sum) || running_sum <= 0.0)
        return ET_RADIX_ERR_NONFINITE;
      for (uint32_t d = 0u; d < hd; ++d) {
        const float val = (float)(acc[d] / running_sum);
        if (!f32_is_finite(val))
          return ET_RADIX_ERR_NONFINITE;
        staged_row[d] = val;
      }
      float *out_row = out + ((size_t)qi * n_q_heads + (size_t)qh) * hd;
      memcpy(out_row, staged_row, (size_t)hd * sizeof(float));
    }
  }
  return ET_RADIX_OK;
}

/* ---- Full structural invariant verifier ---- */

et_radix_kv_status et_radix_kv_verify_invariants(const et_radix_kv_tree *tree) {
  if (tree == NULL)
    return ET_RADIX_ERR_ARGUMENT;

  const uint32_t max_nodes = tree->cfg.max_nodes;
  const uint32_t max_tokens = tree->cfg.max_kv_tokens;

  if (tree->active_nodes < 1u || tree->active_nodes > max_nodes ||
      tree->active_nodes + tree->free_node_top != max_nodes)
    return ET_RADIX_ERR_INVARIANT;
  if (tree->used_tokens > max_tokens ||
      tree->used_tokens + tree->free_slot_top != max_tokens ||
      tree->pinned_tokens > tree->used_tokens)
    return ET_RADIX_ERR_INVARIANT;

  /* Root node invariants. */
  const radix_node *root = &tree->nodes[0];
  if (!root->alive || root->parent != ET_RADIX_NONE || root->edge_len != 0u ||
      root->first_slot != ET_RADIX_NONE || root->last_slot != ET_RADIX_NONE ||
      root->prev_sibling != ET_RADIX_NONE ||
      root->next_sibling != ET_RADIX_NONE || root->generation == 0u ||
      root->last_access_tick > tree->lru_tick)
    return ET_RADIX_ERR_INVARIANT;

  /* Track node and slot visitation using bounded stack bitmaps. */
  uint8_t node_seen[ET_RADIX_KV_MAX_NODES];
  uint8_t slot_seen[ET_RADIX_KV_MAX_TOKENS];
  uint32_t dfs_stack[ET_RADIX_KV_MAX_NODES];
  memset(node_seen, 0, max_nodes);
  memset(slot_seen, 0, max_tokens);

  /* 1. Traverse tree from root verifying acyclicity, parent/sibling links,
   *    unique first-token per parent, slot chain integrity, pin_count
   *    monotonicity, and LRU timestamp monotonicity. */
  uint32_t sp = 0u;
  dfs_stack[sp++] = 0u;
  node_seen[0] = 1u;
  uint32_t visited_nodes = 0u;
  uint32_t counted_used_tokens = 0u;
  uint32_t counted_pinned_tokens = 0u;

  while (sp > 0u) {
    const uint32_t u = dfs_stack[--sp];
    const radix_node *un = &tree->nodes[u];
    visited_nodes++;

    if (!un->alive || un->generation == 0u ||
        un->last_access_tick > tree->lru_tick)
      return ET_RADIX_ERR_INVARIANT;

    if (u != 0u) {
      if (un->edge_len == 0u || un->first_slot == ET_RADIX_NONE ||
          un->last_slot == ET_RADIX_NONE)
        return ET_RADIX_ERR_INVARIANT;
      if (un->first_token != tree->slot_token_id[un->first_slot])
        return ET_RADIX_ERR_INVARIANT;

      uint32_t s = un->first_slot;
      uint32_t chain_len = 0u;
      uint32_t last = ET_RADIX_NONE;
      while (s != ET_RADIX_NONE) {
        if (s >= max_tokens || slot_seen[s] != 0u ||
            tree->slot_owner_node[s] != u || chain_len >= max_tokens)
          return ET_RADIX_ERR_INVARIANT;
        slot_seen[s] = 1u;
        last = s;
        s = tree->slot_next[s];
        chain_len++;
      }
      if (chain_len != un->edge_len || last != un->last_slot)
        return ET_RADIX_ERR_INVARIANT;

      counted_used_tokens += chain_len;
      if (un->pin_count > 0u)
        counted_pinned_tokens += chain_len;
    }

    /* Check child list links, unique first tokens, sum of child pin counts,
     * and parent/child LRU monotonicity. */
    uint32_t child_count = 0u;
    uint64_t child_pin_sum = 0u;
    uint32_t prev_c = ET_RADIX_NONE;
    uint32_t c = un->first_child;
    while (c != ET_RADIX_NONE) {
      if (c == 0u || c >= max_nodes || node_seen[c] != 0u ||
          child_count >= max_nodes)
        return ET_RADIX_ERR_INVARIANT;
      const radix_node *cn = &tree->nodes[c];
      if (!cn->alive || cn->parent != u || cn->prev_sibling != prev_c)
        return ET_RADIX_ERR_INVARIANT;
      if (u != 0u && cn->last_access_tick > un->last_access_tick)
        return ET_RADIX_ERR_INVARIANT;

      /* Verify unique first_token among siblings of u. */
      uint32_t sib = cn->next_sibling;
      while (sib != ET_RADIX_NONE) {
        if (sib >= max_nodes || tree->nodes[sib].first_token == cn->first_token)
          return ET_RADIX_ERR_INVARIANT;
        sib = tree->nodes[sib].next_sibling;
      }

      node_seen[c] = 1u;
      dfs_stack[sp++] = c;
      child_pin_sum += (uint64_t)cn->pin_count;
      child_count++;
      prev_c = c;
      c = cn->next_sibling;
    }

    if (child_count != un->num_children)
      return ET_RADIX_ERR_INVARIANT;
    if ((uint64_t)un->self_pin_count + child_pin_sum != (uint64_t)un->pin_count)
      return ET_RADIX_ERR_INVARIANT;
  }

  if (visited_nodes != tree->active_nodes ||
      counted_used_tokens != tree->used_tokens ||
      counted_pinned_tokens != tree->pinned_tokens)
    return ET_RADIX_ERR_INVARIANT;

  /* 2. Verify free_nodes stack contains exactly all non-visited dead nodes. */
  for (uint32_t i = 0u; i < tree->free_node_top; ++i) {
    const uint32_t idx = tree->free_nodes[i];
    if (idx == 0u || idx >= max_nodes || node_seen[idx] != 0u ||
        tree->nodes[idx].alive)
      return ET_RADIX_ERR_INVARIANT;
    node_seen[idx] = 2u;
  }
  for (uint32_t i = 0u; i < max_nodes; ++i) {
    if (node_seen[i] == 0u)
      return ET_RADIX_ERR_INVARIANT;
  }

  /* 3. Verify free_slots stack contains exactly all unowned free slots. */
  for (uint32_t i = 0u; i < tree->free_slot_top; ++i) {
    const uint32_t s = tree->free_slots[i];
    if (s >= max_tokens || slot_seen[s] != 0u ||
        tree->slot_owner_node[s] != ET_RADIX_NONE ||
        tree->slot_next[s] != ET_RADIX_NONE)
      return ET_RADIX_ERR_INVARIANT;
    slot_seen[s] = 2u;
  }
  for (uint32_t s = 0u; s < max_tokens; ++s) {
    if (slot_seen[s] == 0u)
      return ET_RADIX_ERR_INVARIANT;
  }

  return ET_RADIX_OK;
}

/* ---- ETRX1000 v1.0 SHA-256 Checksummed Snapshot Save / Load ---- */

static int write_all_fd(int fd, const uint8_t *bytes, size_t size) {
  while (size != 0u) {
    const ssize_t n = write(fd, bytes, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return 0;
    bytes += (size_t)n;
    size -= (size_t)n;
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

static int constant_time_eq32(const uint8_t a[32], const uint8_t b[32]) {
  uint8_t diff = 0u;
  for (size_t i = 0u; i < 32u; ++i)
    diff |= (uint8_t)(a[i] ^ b[i]);
  return diff == 0u;
}

/* Collect alive nodes in canonical pre-order DFS order (children sorted by
 * first_token ascending) so snapshots are 100% deterministic. */
static et_radix_kv_status collect_canonical_dfs(const et_radix_kv_tree *tree,
                                                uint32_t *dfs_nodes,
                                                uint32_t *dfs_parent_idx,
                                                uint32_t *out_count) {
  uint32_t stack_node[ET_RADIX_KV_MAX_NODES];
  uint32_t stack_parent_dfs[ET_RADIX_KV_MAX_NODES];
  uint32_t child_buf[ET_RADIX_KV_MAX_NODES];
  uint32_t sp = 0u;
  uint32_t count = 0u;

  stack_node[sp] = 0u;
  stack_parent_dfs[sp] = ET_RADIX_NONE;
  sp++;

  while (sp > 0u) {
    sp--;
    const uint32_t u = stack_node[sp];
    const uint32_t p_dfs = stack_parent_dfs[sp];
    const uint32_t my_dfs = count;
    dfs_nodes[count] = u;
    dfs_parent_idx[count] = p_dfs;
    count++;

    /* Gather children of u and insertion-sort by first_token ascending. */
    uint32_t nc = 0u;
    uint32_t c = tree->nodes[u].first_child;
    while (c != ET_RADIX_NONE) {
      child_buf[nc++] = c;
      c = tree->nodes[c].next_sibling;
    }
    for (uint32_t i = 1u; i < nc; ++i) {
      const uint32_t key = child_buf[i];
      const uint32_t key_tok = tree->nodes[key].first_token;
      uint32_t j = i;
      while (j > 0u && tree->nodes[child_buf[j - 1u]].first_token > key_tok) {
        child_buf[j] = child_buf[j - 1u];
        j--;
      }
      child_buf[j] = key;
    }
    /* Push in reverse order so smallest first_token is popped first. */
    for (uint32_t i = nc; i > 0u; --i) {
      stack_node[sp] = child_buf[i - 1u];
      stack_parent_dfs[sp] = my_dfs;
      sp++;
    }
  }

  if (count != tree->active_nodes)
    return ET_RADIX_ERR_INVARIANT;
  *out_count = count;
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_snapshot_save(const et_radix_kv_tree *tree,
                                             const char *path) {
  if (!fp_environment_supported())
    return ET_RADIX_ERR_FP_ENV;
  if (tree == NULL || path == NULL || path[0] == '\0')
    return ET_RADIX_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_RADIX_ERR_ARGUMENT;

  et_radix_kv_status st = et_radix_kv_verify_invariants(tree);
  if (st != ET_RADIX_OK)
    return st;

  uint32_t dfs_nodes[ET_RADIX_KV_MAX_NODES];
  uint32_t dfs_parent_idx[ET_RADIX_KV_MAX_NODES];
  uint32_t active_nodes = 0u;
  st = collect_canonical_dfs(tree, dfs_nodes, dfs_parent_idx, &active_nodes);
  if (st != ET_RADIX_OK)
    return st;

  /* Exact payload size:
   *   active_nodes * 24 (node header) +
   *   used_tokens * (4 + floats_per_token_total * 4). */
  size_t node_hdr_bytes = 0u;
  size_t tok_kv_bytes = 0u;
  size_t per_tok_bytes = 0u;
  size_t all_tok_bytes = 0u;
  size_t payload_bytes = 0u;
  size_t total_bytes = 0u;
  if (!mul_size((size_t)active_nodes, 24u, &node_hdr_bytes) ||
      !mul_size(tree->floats_per_token_total, sizeof(float), &tok_kv_bytes) ||
      !add_size(4u, tok_kv_bytes, &per_tok_bytes) ||
      !mul_size((size_t)tree->used_tokens, per_tok_bytes, &all_tok_bytes) ||
      !add_size(node_hdr_bytes, all_tok_bytes, &payload_bytes) ||
      !add_size(ET_RADIX_KV_SNAPSHOT_HEADER_BYTES, payload_bytes,
                &total_bytes) ||
      !add_size(total_bytes, ET_RADIX_KV_SNAPSHOT_DIGEST_BYTES, &total_bytes))
    return ET_RADIX_ERR_OVERFLOW;

  if ((uint64_t)total_bytes > ET_RADIX_KV_MAX_SNAPSHOT_BYTES)
    return ET_RADIX_ERR_OVERFLOW;

  uint8_t *buf = (uint8_t *)malloc(total_bytes);
  if (buf == NULL)
    return ET_RADIX_ERR_NOMEM;

  /* Encode 64-byte header. */
  memcpy(buf, ET_RADIX_KV_SNAPSHOT_MAGIC, 8u);
  store_u32_le(buf + 8u, ET_RADIX_KV_SNAPSHOT_VERSION_MAJOR);
  store_u32_le(buf + 12u, ET_RADIX_KV_SNAPSHOT_VERSION_MINOR);
  store_u32_le(buf + 16u, tree->cfg.max_kv_tokens);
  store_u32_le(buf + 20u, tree->cfg.max_nodes);
  store_u32_le(buf + 24u, tree->cfg.n_layers);
  store_u32_le(buf + 28u, tree->cfg.n_kv_heads);
  store_u32_le(buf + 32u, tree->cfg.head_dim);
  store_u32_le(buf + 36u, active_nodes);
  store_u32_le(buf + 40u, tree->used_tokens);
  store_u32_le(buf + 44u, 0u);
  store_u64_le(buf + 48u, tree->lru_tick);
  store_u64_le(buf + 56u, (uint64_t)payload_bytes);

  size_t off = ET_RADIX_KV_SNAPSHOT_HEADER_BYTES;
  for (uint32_t k = 0u; k < active_nodes; ++k) {
    const radix_node *n = &tree->nodes[dfs_nodes[k]];
    store_u32_le(buf + off, dfs_parent_idx[k]);
    store_u32_le(buf + off + 4u, n->edge_len);
    store_u32_le(buf + off + 8u, 0u); /* transient pins reset on load */
    store_u32_le(buf + off + 12u, 0u);
    store_u64_le(buf + off + 16u, n->last_access_tick);
    off += 24u;

    uint32_t s = n->first_slot;
    for (uint32_t i = 0u; i < n->edge_len; ++i) {
      store_u32_le(buf + off, tree->slot_token_id[s]);
      off += 4u;
      s = tree->slot_next[s];
    }

    s = n->first_slot;
    for (uint32_t i = 0u; i < n->edge_len; ++i) {
      for (uint32_t l = 0u; l < tree->cfg.n_layers; ++l) {
        for (uint32_t is_v = 0u; is_v < 2u; ++is_v) {
          const float *src = slot_layer_ptr_const(tree, s, l, is_v);
          for (size_t f = 0u; f < tree->floats_per_layer_kv; ++f) {
            const uint32_t bits = f32_to_bits(src[f]);
            if (!f32_bits_finite(bits)) {
              free(buf);
              return ET_RADIX_ERR_NONFINITE;
            }
            store_u32_le(buf + off, bits);
            off += 4u;
          }
        }
      }
      s = tree->slot_next[s];
    }
  }

  if (off != ET_RADIX_KV_SNAPSHOT_HEADER_BYTES + payload_bytes) {
    free(buf);
    return ET_RADIX_ERR_INVARIANT;
  }

  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, buf, off);
  sha256_final(&sha, buf + off);

  /* Atomic 0600 temp file + fsync + rename. */
  static const char suffix[] = ".etrx-tmp.XXXXXX";
  const size_t prefix = slash == NULL ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix)) {
    free(buf);
    return ET_RADIX_ERR_OVERFLOW;
  }
  char *temp_path = (char *)malloc(prefix + sizeof(suffix));
  if (temp_path == NULL) {
    free(buf);
    return ET_RADIX_ERR_NOMEM;
  }
  memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    free(buf);
    return ET_RADIX_ERR_IO;
  }

  int ok = (fchmod(fd, 0600) == 0) && write_all_fd(fd, buf, total_bytes) &&
           sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    free(temp_path);
    free(buf);
    return ET_RADIX_ERR_IO;
  }

  free(temp_path);
  free(buf);
  return ET_RADIX_OK;
}

et_radix_kv_status et_radix_kv_snapshot_load(const char *path,
                                             et_radix_kv_tree **out_tree) {
  if (!fp_environment_supported())
    return ET_RADIX_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0' || out_tree == NULL)
    return ET_RADIX_ERR_ARGUMENT;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_RADIX_ERR_SYMLINK : ET_RADIX_ERR_IO;

  struct stat meta;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    (void)close(fd);
    return ET_RADIX_ERR_IO;
  }
  const uint64_t min_bytes = (uint64_t)ET_RADIX_KV_SNAPSHOT_HEADER_BYTES +
                             (uint64_t)ET_RADIX_KV_SNAPSHOT_DIGEST_BYTES;
  if (meta.st_size < 0 || (uint64_t)meta.st_size < min_bytes) {
    (void)close(fd);
    return ET_RADIX_ERR_TRUNCATED;
  }
  if ((uint64_t)meta.st_size > ET_RADIX_KV_MAX_SNAPSHOT_BYTES) {
    (void)close(fd);
    return ET_RADIX_ERR_FORMAT;
  }

  const size_t file_size = (size_t)meta.st_size;
  uint8_t *buf = (uint8_t *)malloc(file_size + 1u);
  if (buf == NULL) {
    (void)close(fd);
    return ET_RADIX_ERR_NOMEM;
  }

  size_t used = 0u;
  et_radix_kv_status st = ET_RADIX_OK;
  while (used < file_size + 1u) {
    const ssize_t n = read(fd, buf + used, (file_size + 1u) - used);
    if (n == 0)
      break;
    if (n < 0) {
      if (errno == EINTR)
        continue;
      st = ET_RADIX_ERR_IO;
      break;
    }
    used += (size_t)n;
  }
  if (close(fd) != 0 && st == ET_RADIX_OK)
    st = ET_RADIX_ERR_IO;
  if (st != ET_RADIX_OK) {
    free(buf);
    return st;
  }
  if (used < file_size) {
    free(buf);
    return ET_RADIX_ERR_TRUNCATED;
  }
  if (used > file_size) {
    free(buf);
    return ET_RADIX_ERR_FORMAT;
  }

  /* Validate magic, version, and exact payload length before checksum. */
  if (memcmp(buf, ET_RADIX_KV_SNAPSHOT_MAGIC, 8u) != 0) {
    free(buf);
    return ET_RADIX_ERR_FORMAT;
  }
  const uint32_t ver_major = load_u32_le(buf + 8u);
  const uint32_t ver_minor = load_u32_le(buf + 12u);
  const uint32_t reserved = load_u32_le(buf + 44u);
  if (ver_major != ET_RADIX_KV_SNAPSHOT_VERSION_MAJOR ||
      ver_minor != ET_RADIX_KV_SNAPSHOT_VERSION_MINOR || reserved != 0u) {
    free(buf);
    return ET_RADIX_ERR_FORMAT;
  }

  const uint64_t payload_bytes_u64 = load_u64_le(buf + 56u);
  const uint64_t expected_total =
      (uint64_t)ET_RADIX_KV_SNAPSHOT_HEADER_BYTES + payload_bytes_u64 +
      (uint64_t)ET_RADIX_KV_SNAPSHOT_DIGEST_BYTES;
  if (payload_bytes_u64 > ET_RADIX_KV_MAX_SNAPSHOT_BYTES ||
      expected_total > (uint64_t)file_size) {
    free(buf);
    return ET_RADIX_ERR_TRUNCATED;
  }
  if (expected_total != (uint64_t)file_size) {
    free(buf);
    return ET_RADIX_ERR_FORMAT;
  }

  const size_t body_len =
      ET_RADIX_KV_SNAPSHOT_HEADER_BYTES + (size_t)payload_bytes_u64;
  uint8_t digest[32];
  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, buf, body_len);
  sha256_final(&sha, digest);
  if (!constant_time_eq32(digest, buf + body_len)) {
    free(buf);
    return ET_RADIX_ERR_CHECKSUM;
  }

  et_radix_kv_config cfg;
  cfg.max_kv_tokens = load_u32_le(buf + 16u);
  cfg.max_nodes = load_u32_le(buf + 20u);
  cfg.n_layers = load_u32_le(buf + 24u);
  cfg.n_kv_heads = load_u32_le(buf + 28u);
  cfg.head_dim = load_u32_le(buf + 32u);
  cfg.reserved = 0u;
  const uint32_t active_nodes = load_u32_le(buf + 36u);
  const uint32_t used_tokens = load_u32_le(buf + 40u);
  const uint64_t lru_tick = load_u64_le(buf + 48u);

  if (active_nodes < 1u || active_nodes > cfg.max_nodes ||
      used_tokens > cfg.max_kv_tokens) {
    free(buf);
    return ET_RADIX_ERR_FORMAT;
  }

  et_radix_kv_tree *tree = NULL;
  st = et_radix_kv_create(&cfg, &tree);
  if (st != ET_RADIX_OK) {
    free(buf);
    return st;
  }
  tree->lru_tick = lru_tick;

  uint32_t dfs_to_node[ET_RADIX_KV_MAX_NODES];
  size_t off = ET_RADIX_KV_SNAPSHOT_HEADER_BYTES;
  uint32_t loaded_tokens = 0u;

  for (uint32_t k = 0u; k < active_nodes; ++k) {
    if (body_len - off < 24u) {
      st = ET_RADIX_ERR_TRUNCATED;
      goto fail;
    }
    const uint32_t p_dfs = load_u32_le(buf + off);
    const uint32_t edge_len = load_u32_le(buf + off + 4u);
    const uint32_t flags = load_u32_le(buf + off + 8u);
    const uint32_t nres = load_u32_le(buf + off + 12u);
    const uint64_t node_tick = load_u64_le(buf + off + 16u);
    off += 24u;

    if (flags != 0u || nres != 0u || node_tick > lru_tick) {
      st = ET_RADIX_ERR_FORMAT;
      goto fail;
    }

    if (k == 0u) {
      if (p_dfs != ET_RADIX_NONE || edge_len != 0u) {
        st = ET_RADIX_ERR_FORMAT;
        goto fail;
      }
      tree->nodes[0].last_access_tick = node_tick;
      dfs_to_node[0] = 0u;
      continue;
    }

    if (p_dfs >= k || edge_len == 0u ||
        edge_len > used_tokens - loaded_tokens || tree->free_node_top == 0u ||
        tree->free_slot_top < edge_len) {
      st = ET_RADIX_ERR_FORMAT;
      goto fail;
    }

    const uint32_t parent_idx = dfs_to_node[p_dfs];
    if (parent_idx != 0u && node_tick > tree->nodes[parent_idx].last_access_tick) {
      st = ET_RADIX_ERR_FORMAT;
      goto fail;
    }

    size_t tok_bytes = 0u;
    size_t kv_bytes = 0u;
    size_t entry_bytes = 0u;
    if (!mul_size((size_t)edge_len, 4u, &tok_bytes) ||
        !mul_size((size_t)edge_len, tree->floats_per_token_total * sizeof(float),
                  &kv_bytes) ||
        !add_size(tok_bytes, kv_bytes, &entry_bytes) ||
        body_len - off < entry_bytes) {
      st = ET_RADIX_ERR_TRUNCATED;
      goto fail;
    }

    const uint32_t first_tok = load_u32_le(buf + off);
    if (find_child_by_token(tree, parent_idx, first_tok) != ET_RADIX_NONE) {
      st = ET_RADIX_ERR_FORMAT;
      goto fail;
    }

    const uint32_t n_idx = alloc_node(tree);
    dfs_to_node[k] = n_idx;
    radix_node *n = &tree->nodes[n_idx];
    n->edge_len = edge_len;
    n->first_token = first_tok;
    n->last_access_tick = node_tick;
    n->pin_count = 0u;
    n->self_pin_count = 0u;

    uint32_t prev_slot = ET_RADIX_NONE;
    for (uint32_t i = 0u; i < edge_len; ++i) {
      const uint32_t slot = tree->free_slots[--tree->free_slot_top];
      tree->slot_token_id[slot] = load_u32_le(buf + off);
      off += 4u;
      tree->slot_next[slot] = ET_RADIX_NONE;
      tree->slot_owner_node[slot] = n_idx;
      if (prev_slot == ET_RADIX_NONE)
        n->first_slot = slot;
      else
        tree->slot_next[prev_slot] = slot;
      prev_slot = slot;
    }
    n->last_slot = prev_slot;

    uint32_t s = n->first_slot;
    for (uint32_t i = 0u; i < edge_len; ++i) {
      for (uint32_t l = 0u; l < cfg.n_layers; ++l) {
        for (uint32_t is_v = 0u; is_v < 2u; ++is_v) {
          float *dst = slot_layer_ptr(tree, s, l, is_v);
          for (size_t f = 0u; f < tree->floats_per_layer_kv; ++f) {
            const uint32_t bits = load_u32_le(buf + off);
            off += 4u;
            if (!f32_bits_finite(bits)) {
              st = ET_RADIX_ERR_NONFINITE;
              goto fail;
            }
            dst[f] = bits_to_f32(bits);
          }
        }
      }
      s = tree->slot_next[s];
    }

    attach_child(tree, parent_idx, n_idx);
    loaded_tokens += edge_len;
    tree->used_tokens += edge_len;
  }

  if (off != body_len || loaded_tokens != used_tokens) {
    st = ET_RADIX_ERR_FORMAT;
    goto fail;
  }

  st = et_radix_kv_verify_invariants(tree);
  if (st != ET_RADIX_OK)
    goto fail;

  free(buf);
  *out_tree = tree;
  return ET_RADIX_OK;

fail:
  et_radix_kv_destroy(tree);
  free(buf);
  return st;
}
