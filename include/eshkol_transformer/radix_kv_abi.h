#ifndef ESHKOL_TRANSFORMER_RADIX_KV_ABI_H
#define ESHKOL_TRANSFORMER_RADIX_KV_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_RADIX_KV_ABI_MAJOR 1u
#define ET_RADIX_KV_ABI_MINOR 0u

/* Bounded pool and tensor dimension limits (explicit, zero silent fallback). */
#define ET_RADIX_KV_MAX_TOKENS 4096u
#define ET_RADIX_KV_MAX_NODES 4096u
#define ET_RADIX_KV_MAX_LAYERS 64u
#define ET_RADIX_KV_MAX_KV_HEADS 64u
#define ET_RADIX_KV_MAX_Q_HEADS 64u
#define ET_RADIX_KV_MAX_HEAD_DIM 256u

/* Maximum total binary32 elements in physical KV pool (64M floats = 256 MiB). */
#define ET_RADIX_KV_MAX_POOL_FLOATS UINT64_C(67108864)

/* Maximum serialized ETRX1000 snapshot size (64 MiB cap). */
#define ET_RADIX_KV_MAX_SNAPSHOT_BYTES UINT64_C(67108864)

/* Canonical ETRX1000 v1.0 snapshot format constants. */
#define ET_RADIX_KV_SNAPSHOT_MAGIC "ETRX1000"
#define ET_RADIX_KV_SNAPSHOT_VERSION_MAJOR 1u
#define ET_RADIX_KV_SNAPSHOT_VERSION_MINOR 0u
#define ET_RADIX_KV_SNAPSHOT_HEADER_BYTES 64u
#define ET_RADIX_KV_SNAPSHOT_DIGEST_BYTES 32u

typedef int32_t et_radix_kv_status;
typedef uint64_t et_radix_kv_handle;

#define ET_RADIX_KV_HANDLE_INVALID UINT64_C(0)

enum {
  ET_RADIX_OK = 0,
  ET_RADIX_ERR_ARGUMENT = -1,
  ET_RADIX_ERR_SHAPE = -2,
  ET_RADIX_ERR_ALIGNMENT = -3,
  ET_RADIX_ERR_ALIAS = -4,
  ET_RADIX_ERR_NONFINITE = -5,
  ET_RADIX_ERR_FP_ENV = -6,
  ET_RADIX_ERR_OOM = -7,
  ET_RADIX_ERR_INVALID_HANDLE = -8,
  ET_RADIX_ERR_DOUBLE_UNPIN = -9,
  ET_RADIX_ERR_OVERFLOW = -10,
  ET_RADIX_ERR_NOMEM = -11,
  ET_RADIX_ERR_INVARIANT = -12,
  ET_RADIX_ERR_IO = -13,
  ET_RADIX_ERR_SYMLINK = -14,
  ET_RADIX_ERR_FORMAT = -15,
  ET_RADIX_ERR_TRUNCATED = -16,
  ET_RADIX_ERR_CHECKSUM = -17
};

/* RadixAttention Prefix KV Cache Tree configuration.
 *
 * Physical KV pool tensor shape (row-major contiguous IEEE-754 binary32):
 *   [max_kv_tokens, n_layers, 2 (K=0, V=1), n_kv_heads, head_dim]
 *
 * Constraints:
 *   1 <= max_kv_tokens <= ET_RADIX_KV_MAX_TOKENS
 *   2 <= max_nodes     <= ET_RADIX_KV_MAX_NODES  (node 0 is always root)
 *   1 <= n_layers      <= ET_RADIX_KV_MAX_LAYERS
 *   1 <= n_kv_heads    <= ET_RADIX_KV_MAX_KV_HEADS
 *   1 <= head_dim      <= ET_RADIX_KV_MAX_HEAD_DIM
 *   reserved == 0
 *   total pool floats  <= ET_RADIX_KV_MAX_POOL_FLOATS
 */
typedef struct et_radix_kv_config {
  uint32_t max_kv_tokens;
  uint32_t max_nodes;
  uint32_t n_layers;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t reserved;
} et_radix_kv_config;

/* Exact accounting and LRU telemetry snapshot.
 *
 * Invariants guaranteed after every successful operation:
 *   used_tokens + free_tokens == max_kv_tokens
 *   evictable_tokens + pinned_tokens == used_tokens
 *   active_nodes + free_nodes == max_nodes
 *   active_nodes >= 1  (root node is always alive)
 */
typedef struct et_radix_kv_stats {
  uint32_t max_kv_tokens;
  uint32_t used_tokens;
  uint32_t free_tokens;
  uint32_t pinned_tokens;
  uint32_t evictable_tokens;
  uint32_t max_nodes;
  uint32_t active_nodes;
  uint32_t free_nodes;
  uint64_t lru_tick;
  uint64_t total_matches;
  uint64_t total_matched_tokens;
  uint64_t total_inserted_tokens;
  uint64_t total_evicted_tokens;
  uint64_t total_evicted_nodes;
  uint64_t total_edge_splits;
  uint64_t total_oom_rejections;
} et_radix_kv_stats;

/* Result of longest-prefix matching. */
typedef struct et_radix_kv_match_result {
  et_radix_kv_handle handle;
  uint32_t matched_len;
  uint32_t did_split;
} et_radix_kv_match_result;

#if defined(__cplusplus)
#define ET_RADIX_KV_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_RADIX_KV_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_RADIX_KV_STATIC_ASSERT(sizeof(et_radix_kv_config) == 24u &&
                              offsetof(et_radix_kv_config, reserved) == 20u,
                          "et_radix_kv_config ABI layout changed");
ET_RADIX_KV_STATIC_ASSERT(sizeof(et_radix_kv_stats) == 96u &&
                              offsetof(et_radix_kv_stats, lru_tick) == 32u &&
                              offsetof(et_radix_kv_stats,
                                       total_oom_rejections) == 88u,
                          "et_radix_kv_stats ABI layout changed");
ET_RADIX_KV_STATIC_ASSERT(sizeof(et_radix_kv_match_result) == 16u &&
                              offsetof(et_radix_kv_match_result,
                                       matched_len) == 8u &&
                              offsetof(et_radix_kv_match_result, did_split) ==
                                  12u,
                          "et_radix_kv_match_result ABI layout changed");
#undef ET_RADIX_KV_STATIC_ASSERT

typedef struct et_radix_kv_tree et_radix_kv_tree;

/* Validate configuration bounds and FP environment. */
et_radix_kv_status et_radix_kv_validate_config(const et_radix_kv_config *cfg);

/* Compute exact float element count of the contiguous physical KV pool:
 *   max_kv_tokens * n_layers * 2 * n_kv_heads * head_dim. */
et_radix_kv_status et_radix_kv_pool_float_count(const et_radix_kv_config *cfg,
                                                size_t *out_count);

/* Lifecycle management. All pools are pre-allocated at create time; zero heap
 * allocations occur during prefix matching, pinning, insertion, eviction,
 * invariant checking, or attention execution. */
et_radix_kv_status et_radix_kv_create(const et_radix_kv_config *cfg,
                                      et_radix_kv_tree **out_tree);
void et_radix_kv_destroy(et_radix_kv_tree *tree);
et_radix_kv_status et_radix_kv_reset(et_radix_kv_tree *tree);

/* Return generational handle of the always-alive root node (0 tokens), or
 * ET_RADIX_KV_HANDLE_INVALID if tree == NULL. */
et_radix_kv_handle et_radix_kv_root_handle(const et_radix_kv_tree *tree);

/* Query live telemetry and immutable configuration. */
et_radix_kv_status et_radix_kv_stats_get(const et_radix_kv_tree *tree,
                                         et_radix_kv_stats *out_stats);
et_radix_kv_status et_radix_kv_config_get(const et_radix_kv_tree *tree,
                                          et_radix_kv_config *out_cfg);

/* Longest-prefix match of tokens[0..seq_len-1].
 * - Automatically splits an interior compressed edge when the match terminates
 *   strictly mid-edge, preserving existing child handles, KV slot ownership,
 *   and pin refcounts.
 * - Advances monotonic logical lru_tick and updates last_access_tick along the
 *   matched ancestor path [root .. matched_node].
 * - If out_kv_indices != NULL, writes the matched physical KV slot indices in
 *   token order 0..matched_len-1 (requires kv_indices_cap >= matched_len).
 * - If out_kv_indices == NULL, kv_indices_cap must be 0.
 * - Fails atomically with ET_RADIX_ERR_OOM if an edge split requires a node
 *   and all non-path nodes are pinned. */
et_radix_kv_status et_radix_kv_match_prefix(et_radix_kv_tree *tree,
                                            const uint32_t *tokens,
                                            uint32_t seq_len,
                                            et_radix_kv_match_result *out_match,
                                            uint32_t *out_kv_indices,
                                            uint32_t kv_indices_cap);

/* Pin / unpin lock refcounts along the ancestor path [handle .. root].
 * - Pinned nodes (pin_count > 0) are never evicted by LRU eviction.
 * - Unpinning a handle with self_pin_count == 0 returns
 *   ET_RADIX_ERR_DOUBLE_UNPIN without mutating state.
 * - Stale/evicted handles return ET_RADIX_ERR_INVALID_HANDLE. */
et_radix_kv_status et_radix_kv_pin(et_radix_kv_tree *tree,
                                   et_radix_kv_handle handle);
et_radix_kv_status et_radix_kv_unpin(et_radix_kv_tree *tree,
                                     et_radix_kv_handle handle);

/* Inspect live handle metadata (any output pointer may be NULL). */
et_radix_kv_status et_radix_kv_handle_info(const et_radix_kv_tree *tree,
                                           et_radix_kv_handle handle,
                                           uint32_t *out_depth_tokens,
                                           uint32_t *out_pin_count,
                                           uint32_t *out_self_pin_count,
                                           uint64_t *out_last_tick);

/* Insert full sequence tokens[0..seq_len-1] with K/V tensors of shape
 * [seq_len, n_layers, n_kv_heads, head_dim].
 * - Shares any already-present prefix zero-copy without duplicating KV slots.
 * - Triggers deterministic LRU leaf eviction of unpinned (pin_count == 0)
 *   leaves (ordered by last_access_tick ASC, node_idx ASC) when space is needed.
 * - Atomic preflight: returns ET_RADIX_ERR_OOM with zero state mutation if
 *   pinned nodes prevent reclaiming enough KV slots or tree nodes.
 * - out_matched_prefix_len is optional (may be NULL). */
et_radix_kv_status et_radix_kv_insert(et_radix_kv_tree *tree,
                                      const uint32_t *tokens, uint32_t seq_len,
                                      const float *k_data, const float *v_data,
                                      et_radix_kv_handle *out_handle,
                                      uint32_t *out_matched_prefix_len);

/* Insert newly computed suffix_tokens[0..suffix_len-1] with suffix K/V of shape
 * [suffix_len, n_layers, n_kv_heads, head_dim] directly under an existing
 * prefix_handle (avoiding re-supplying cached prefix KV tensors).
 * - Shares any further matching prefix under prefix_handle zero-copy.
 * - Atomic preflight and deterministic LRU eviction identical to insert().
 * - out_inserted_tokens is optional (may be NULL). */
et_radix_kv_status et_radix_kv_insert_suffix(
    et_radix_kv_tree *tree, et_radix_kv_handle prefix_handle,
    const uint32_t *suffix_tokens, uint32_t suffix_len, const float *suffix_k,
    const float *suffix_v, et_radix_kv_handle *out_handle,
    uint32_t *out_inserted_tokens);

/* Explicitly evict unpinned LRU leaves until free_tokens >= target_free_tokens
 * or no evictable leaves remain. */
et_radix_kv_status et_radix_kv_evict_lru(et_radix_kv_tree *tree,
                                         uint32_t target_free_tokens,
                                         uint32_t *out_evicted_tokens);

/* Gather cached K and V tensors for one layer along the root-to-handle path.
 * Output shapes: out_k, out_v each [path_len, n_kv_heads, head_dim].
 * Requires out_floats >= path_len * n_kv_heads * head_dim. */
et_radix_kv_status et_radix_kv_read_kv(const et_radix_kv_tree *tree,
                                       et_radix_kv_handle handle,
                                       uint32_t layer, float *out_k,
                                       float *out_v, size_t out_floats,
                                       uint32_t *out_seq_len);

/* Exact binary64-accumulated causal GQA attention over cached prefix
 * [root .. prefix_handle] (length P) concatenated with new suffix (length S).
 *
 * Tensor shapes (row-major IEEE-754 binary32):
 *   q        : [q_len, n_q_heads, head_dim]
 *   suffix_k : [suffix_len, n_kv_heads, head_dim] (NULL iff suffix_len == 0)
 *   suffix_v : [suffix_len, n_kv_heads, head_dim] (NULL iff suffix_len == 0)
 *   out      : [q_len, n_q_heads, head_dim]
 *
 * Query qi in [0, q_len) sits at absolute position abs_pos = (P + S - q_len) +
 * qi and attends causally to t in [0, abs_pos] using identical single-pass
 * binary64 online softmax for both cached prefix and uncached prefill paths. */
et_radix_kv_status et_radix_kv_attend_gqa_f32(
    const et_radix_kv_tree *tree, et_radix_kv_handle prefix_handle,
    uint32_t layer, uint32_t n_q_heads, uint32_t q_len, uint32_t suffix_len,
    float scale, const float *q, const float *suffix_k, const float *suffix_v,
    float *out);

/* Full structural invariant verifier:
 * - Tree acyclicity and exact node reachability/free-list partition
 * - Unique first-token keys among children of every node
 * - Exact KV slot partition between alive edge chains and free-slot stack
 * - Exact token accounting (used + free == max, pinned + evictable == used)
 * - Refcount monotonicity (node.pin_count == self_pin + sum(child.pin_count))
 * - LRU monotonicity (parent.last_access_tick >= child.last_access_tick)
 */
et_radix_kv_status et_radix_kv_verify_invariants(const et_radix_kv_tree *tree);

/* Versioned SHA-256 checksummed ETRX1000 v1.0 snapshot persistence.
 * - Save: verifies invariants, serializes canonical pre-order DFS topology +
 *   token IDs + finite binary32 KV tensors + logical LRU ticks, writes via
 *   0600 temp file + fsync + atomic rename.
 * - Load: opens with O_NOFOLLOW (rejecting symlinks), validates exact size,
 *   magic, version, SHA-256 digest, finiteness, tree invariants, and returns a
 *   fresh tree with transient pin counts initialized to 0. */
et_radix_kv_status et_radix_kv_snapshot_save(const et_radix_kv_tree *tree,
                                             const char *path);
et_radix_kv_status et_radix_kv_snapshot_load(const char *path,
                                             et_radix_kv_tree **out_tree);

#ifdef __cplusplus
}
#endif

#endif
