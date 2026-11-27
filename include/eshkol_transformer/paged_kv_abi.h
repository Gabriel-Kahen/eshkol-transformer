#ifndef ESHKOL_TRANSFORMER_PAGED_KV_ABI_H
#define ESHKOL_TRANSFORMER_PAGED_KV_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_PAGED_KV_ABI_MAJOR 1u
#define ET_PAGED_KV_ABI_MINOR 1u

/* Bounded pool and tensor dimension limits (explicit, zero silent fallback). */
#define ET_PAGED_KV_MAX_BLOCKS 4096u
#define ET_PAGED_KV_MAX_LAYERS 64u
#define ET_PAGED_KV_MAX_KV_HEADS 64u
#define ET_PAGED_KV_MAX_Q_HEADS 64u
#define ET_PAGED_KV_MAX_BLOCK_SIZE 64u
#define ET_PAGED_KV_MAX_HEAD_DIM 256u
#define ET_PAGED_KV_MAX_SEQS 256u
#define ET_PAGED_KV_MAX_BLOCKS_PER_SEQ 1024u

/* Maximum total floats in a physical KV pool (64M floats = 256 MiB cap). */
#define ET_PAGED_KV_MAX_POOL_FLOATS UINT64_C(67108864)

/* Canonical SHA-256 prefix block hash width in bytes. */
#define ET_PAGED_KV_HASH_BYTES 32u

typedef int32_t et_paged_kv_status;

enum {
  ET_PAGED_KV_OK = 0,
  ET_PAGED_KV_ERR_ARGUMENT = -1,
  ET_PAGED_KV_ERR_SHAPE = -2,
  ET_PAGED_KV_ERR_ALIGNMENT = -3,
  ET_PAGED_KV_ERR_ALIAS = -4,
  ET_PAGED_KV_ERR_NONFINITE = -5,
  ET_PAGED_KV_ERR_FP_ENV = -6,
  ET_PAGED_KV_ERR_OOM_BLOCKS = -7,
  ET_PAGED_KV_ERR_OOM_SEQS = -8,
  ET_PAGED_KV_ERR_INVALID_SEQ = -9,
  ET_PAGED_KV_ERR_SEQ_FULL = -10,
  ET_PAGED_KV_ERR_INVALID_RANGE = -11,
  ET_PAGED_KV_ERR_NOMEM = -12,
  ET_PAGED_KV_ERR_OVERFLOW = -13,
  ET_PAGED_KV_ERR_STATE = -14,
  ET_PAGED_KV_ERR_CONTEXT = -15
};

/* Paged KV-cache pool configuration.
 *
 * Physical KV pool tensor shape (row-major contiguous binary32):
 *   [num_blocks, num_layers, 2 (K=0, V=1), num_kv_heads, block_size, head_dim]
 *
 * Constraints:
 *   1 <= num_blocks         <= ET_PAGED_KV_MAX_BLOCKS
 *   1 <= num_layers         <= ET_PAGED_KV_MAX_LAYERS
 *   1 <= num_kv_heads       <= ET_PAGED_KV_MAX_KV_HEADS
 *   1 <= block_size         <= ET_PAGED_KV_MAX_BLOCK_SIZE
 *   1 <= head_dim           <= ET_PAGED_KV_MAX_HEAD_DIM
 *   1 <= max_seqs           <= ET_PAGED_KV_MAX_SEQS
 *   1 <= max_blocks_per_seq <= ET_PAGED_KV_MAX_BLOCKS_PER_SEQ
 *   max_blocks_per_seq      <= num_blocks
 *   reserved == 0
 *   total pool floats       <= ET_PAGED_KV_MAX_POOL_FLOATS
 */
typedef struct et_paged_kv_config {
  uint32_t num_blocks;
  uint32_t num_layers;
  uint32_t num_kv_heads;
  uint32_t block_size;
  uint32_t head_dim;
  uint32_t max_seqs;
  uint32_t max_blocks_per_seq;
  uint32_t reserved;
} et_paged_kv_config;

/* Live pool telemetry and CoW/prefix-sharing statistics. */
typedef struct et_paged_kv_stats {
  uint32_t total_blocks;
  uint32_t free_blocks;
  uint32_t used_blocks;
  uint32_t active_seqs;
  uint32_t shared_blocks;        /* physical blocks with ref_count > 1 */
  uint32_t prefix_cached_blocks; /* physical blocks indexed in prefix cache */
  uint64_t cow_copies;           /* total atomic CoW block copies performed */
  uint64_t prefix_cache_hits;    /* total prefix-cache hits on attach */
} et_paged_kv_stats;

/* Caller-supplied fingerprints of the loaded model (including adapters),
 * tokenizer, and complete KV-producing configuration. Each field is 32 bytes.
 * The caller owns fingerprint provenance and its association with the KV. */
typedef struct et_paged_kv_prefix_context_v1 {
  uint8_t model[ET_PAGED_KV_HASH_BYTES];
  uint8_t tokenizer[ET_PAGED_KV_HASH_BYTES];
  uint8_t producer_config[ET_PAGED_KV_HASH_BYTES];
} et_paged_kv_prefix_context_v1;

typedef struct et_paged_kv_prefix_key_v1 {
  et_paged_kv_prefix_context_v1 context;
  uint8_t digest[ET_PAGED_KV_HASH_BYTES];
  uint32_t block_size;
} et_paged_kv_prefix_key_v1;

#if defined(__cplusplus)
#define ET_PAGED_KV_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_PAGED_KV_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_PAGED_KV_STATIC_ASSERT(sizeof(et_paged_kv_config) == 32u &&
                              offsetof(et_paged_kv_config, reserved) == 28u,
                          "et_paged_kv_config ABI layout changed");
ET_PAGED_KV_STATIC_ASSERT(sizeof(et_paged_kv_stats) == 40u &&
                              offsetof(et_paged_kv_stats, cow_copies) == 24u &&
                              offsetof(et_paged_kv_stats, prefix_cache_hits) ==
                                  32u,
                          "et_paged_kv_stats ABI layout changed");
ET_PAGED_KV_STATIC_ASSERT(
    sizeof(et_paged_kv_prefix_context_v1) == 96u &&
        offsetof(et_paged_kv_prefix_context_v1, tokenizer) == 32u &&
        offsetof(et_paged_kv_prefix_context_v1, producer_config) == 64u,
    "et_paged_kv_prefix_context_v1 ABI layout changed");
ET_PAGED_KV_STATIC_ASSERT(
    sizeof(et_paged_kv_prefix_key_v1) == 132u &&
        offsetof(et_paged_kv_prefix_key_v1, digest) == 96u &&
        offsetof(et_paged_kv_prefix_key_v1, block_size) == 128u,
    "et_paged_kv_prefix_key_v1 ABI layout changed");
#undef ET_PAGED_KV_STATIC_ASSERT

typedef struct et_paged_kv_pool et_paged_kv_pool;

/* Validate configuration bounds and FP environment. */
et_paged_kv_status et_paged_kv_validate_config(const et_paged_kv_config *cfg);

/* Compute exact float element count of the contiguous physical KV pool:
 *   num_blocks * num_layers * 2 * num_kv_heads * block_size * head_dim. */
et_paged_kv_status et_paged_kv_pool_float_count(const et_paged_kv_config *cfg,
                                                size_t *out_count);

/* Create a paged KV-cache pool.
 * If external_pool == NULL: external_pool_floats must be 0 and the pool owns
 * its heap-allocated contiguous float buffer.
 * If external_pool != NULL: must be 4-byte aligned, non-overlapping with cfg
 * and out_pool, and external_pool_floats >= required pool float count; caller
 * retains ownership and must keep external_pool alive until destroy(). */
et_paged_kv_status et_paged_kv_create(const et_paged_kv_config *cfg,
                                      float *external_pool,
                                      size_t external_pool_floats,
                                      et_paged_kv_pool **out_pool);
et_paged_kv_status et_paged_kv_create_raw(const et_paged_kv_config *cfg,
                                          void *external_pool,
                                          size_t external_pool_floats,
                                          et_paged_kv_pool **out_pool);

/* Destroy pool and release all pool-owned resources (safe on NULL). */
void et_paged_kv_destroy(et_paged_kv_pool *pool);

/* Reset all sequences, block refcounts, prefix cache, deterministic free list,
 * and counters to the freshly-created state without reallocating memory. */
et_paged_kv_status et_paged_kv_reset(et_paged_kv_pool *pool);

/* Query live statistics and immutable configuration. */
et_paged_kv_status et_paged_kv_stats_get(const et_paged_kv_pool *pool,
                                         et_paged_kv_stats *out_stats);
et_paged_kv_status et_paged_kv_config_get(const et_paged_kv_pool *pool,
                                          et_paged_kv_config *out_cfg);

/* Sequence lifecycle and Copy-on-Write prefix forking. */
et_paged_kv_status et_paged_kv_seq_alloc(et_paged_kv_pool *pool,
                                         uint32_t *out_seq_id);
et_paged_kv_status et_paged_kv_seq_free(et_paged_kv_pool *pool,
                                        uint32_t seq_id);

/* Fork child sequence from alive parent sequence sharing ALL physical blocks
 * via ref_count++ (zero physical block copies at fork time). */
et_paged_kv_status et_paged_kv_seq_fork(et_paged_kv_pool *pool,
                                        uint32_t parent_seq_id,
                                        uint32_t *out_child_seq_id);

/* Speculative decoding / beam rollback: truncate seq_id to new_len <= seq_len,
 * releasing unused tail blocks cleanly and preserving prefix-cache safety. */
et_paged_kv_status et_paged_kv_seq_truncate(et_paged_kv_pool *pool,
                                            uint32_t seq_id, uint32_t new_len);

/* Sequence and physical block inspection helpers. */
et_paged_kv_status et_paged_kv_seq_len(const et_paged_kv_pool *pool,
                                       uint32_t seq_id, uint32_t *out_len);
et_paged_kv_status et_paged_kv_seq_block_table(const et_paged_kv_pool *pool,
                                               uint32_t seq_id,
                                               uint32_t *out_blocks,
                                               uint32_t capacity,
                                               uint32_t *out_num_blocks);
et_paged_kv_status et_paged_kv_block_refcount(const et_paged_kv_pool *pool,
                                              uint32_t block_id,
                                              uint32_t *out_refcount);

/* Append one token's K and V tensors across all layers at pos = seq_len.
 * Tensor shapes:
 *   k_token: [num_layers, num_kv_heads, head_dim]
 *   v_token: [num_layers, num_kv_heads, head_dim]
 * If the target physical block has ref_count > 1, performs atomic CoW:
 * allocates a fresh physical block, copies existing prefix tokens 0..slot-1,
 * updates only seq_id's block table, decrements old block ref_count, and
 * writes the new token without mutating any sibling sequence.
 * Strictly rollback-safe: any error (including ET_PAGED_KV_ERR_OOM_BLOCKS or
 * ET_PAGED_KV_ERR_NONFINITE) leaves pool and sequence state unchanged. */
et_paged_kv_status et_paged_kv_append_token(et_paged_kv_pool *pool,
                                            uint32_t seq_id,
                                            const float *k_token,
                                            const float *v_token);
et_paged_kv_status et_paged_kv_append_token_raw(et_paged_kv_pool *pool,
                                                uint32_t seq_id,
                                                const void *k_token,
                                                const void *v_token);

/* Atomic batch append of num_tokens >= 1 tokens across all layers.
 * Tensor shapes:
 *   k_tokens: [num_tokens, num_layers, num_kv_heads, head_dim]
 *   v_tokens: [num_tokens, num_layers, num_kv_heads, head_dim]
 * Pre-validates all inputs and exact required free blocks before mutating any
 * pool state so ET_PAGED_KV_ERR_OOM_BLOCKS / ET_PAGED_KV_ERR_NONFINITE are
 * strictly rollback-safe. */
et_paged_kv_status et_paged_kv_append_tokens(et_paged_kv_pool *pool,
                                             uint32_t seq_id,
                                             uint32_t num_tokens,
                                             const float *k_tokens,
                                             const float *v_tokens);
et_paged_kv_status et_paged_kv_append_tokens_raw(et_paged_kv_pool *pool,
                                                 uint32_t seq_id,
                                                 uint32_t num_tokens,
                                                 const void *k_tokens,
                                                 const void *v_tokens);

/* Read back cached K and V vectors of length head_dim for a single
 * (seq_id, pos < seq_len, layer_idx, kv_head_idx). */
et_paged_kv_status et_paged_kv_read_token(const et_paged_kv_pool *pool,
                                          uint32_t seq_id, uint32_t pos,
                                          uint32_t layer_idx,
                                          uint32_t kv_head_idx, float *out_k,
                                          float *out_v);
et_paged_kv_status et_paged_kv_read_token_raw(const et_paged_kv_pool *pool,
                                              uint32_t seq_id, uint32_t pos,
                                              uint32_t layer_idx,
                                              uint32_t kv_head_idx, void *out_k,
                                              void *out_v);

/* Exact online-softmax Paged GQA Causal Decode Attention (zero heap allocation
 * on the hot path).
 *
 * Tensor shapes:
 *   q:   [num_q_heads, head_dim]  (const float *, finite, 4-byte aligned)
 *   out: [num_q_heads, head_dim]  (float *, 4-byte aligned, disjoint from q
 *                                  and physical KV pool)
 *
 * Constraints:
 *   seq_id alive with seq_len >= 1
 *   layer_idx < num_layers
 *   1 <= num_q_heads <= ET_PAGED_KV_MAX_Q_HEADS
 *   (num_q_heads % num_kv_heads) == 0
 *   scale finite and > 0.0f (typically 1.0f / sqrtf((float)head_dim))
 *
 * Failure leaves out completely untouched. */
et_paged_kv_status et_paged_kv_decode_attention_v1(
    const et_paged_kv_pool *pool, uint32_t seq_id, uint32_t layer_idx,
    uint32_t num_q_heads, float scale, const float *q, float *out);
et_paged_kv_status et_paged_kv_decode_attention_v1_raw(
    const et_paged_kv_pool *pool, uint32_t seq_id, uint32_t layer_idx,
    uint32_t num_q_heads, float scale, const void *q, void *out);

/* Canonical SHA-256 chain hash for a full token block:
 *   SHA256("ETPKV1\0\0" || (prev_hash ? prev_hash[32] : zeros[32]) ||
 *          LE32(block_size) || LE32(token_ids[0..block_size-1])).
 * Chaining prev_hash incorporates the token IDs in the preceding prefix. */
et_paged_kv_status et_paged_kv_hash_token_block(
    const uint8_t *prev_hash_or_null, const uint32_t *token_ids,
    uint32_t block_size, uint8_t out_hash[ET_PAGED_KV_HASH_BYTES]);

/* Register an already-full logical block (logical_block < seq_len / block_size)
 * in the pool's deterministic prefix cache under prefix_hash. */
et_paged_kv_status et_paged_kv_prefix_cache_register(
    et_paged_kv_pool *pool, uint32_t seq_id, uint32_t logical_block,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES]);

/* Look up prefix_hash in the pool's prefix cache and, on hit, attach the shared
 * physical block to seq_id at its current block-aligned position
 * (seq_len % block_size == 0) with ref_count++ and seq_len += block_size.
 * Sets *out_hit = 1 on cache hit, 0 on miss (leaving seq_id unchanged). */
et_paged_kv_status et_paged_kv_prefix_cache_attach(
    et_paged_kv_pool *pool, uint32_t seq_id,
    const uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES], uint32_t *out_hit);

/* Create a single-producer pool with an immutable copy of context. Reset
 * clears cache contents and preserves that binding. Same storage contract
 * as et_paged_kv_create; context must be non-null and disjoint from the
 * external storage and out_pool. Failure leaves out_pool untouched. */
et_paged_kv_status et_paged_kv_create_with_prefix_context_v1(
    const et_paged_kv_config *cfg, const et_paged_kv_prefix_context_v1 *context,
    float *external_pool, size_t external_pool_floats,
    et_paged_kv_pool **out_pool);

/* SHA256("ETPKV2\0\0" || model || tokenizer || producer_config ||
 *        predecessor_present_byte || predecessor_digest_or_zeros ||
 *        LE32(block_size) || LE32(token_ids[0..block_size-1])).
 * A NULL predecessor starts a chain. A supplied predecessor must have the
 * same context and block size. Failure leaves out_key untouched. out_key may
 * equal previous_or_null to advance a chain in place. */
et_paged_kv_status et_paged_kv_hash_token_block_with_context_v1(
    const et_paged_kv_prefix_context_v1 *context,
    const et_paged_kv_prefix_key_v1 *previous_or_null,
    const uint32_t *token_ids, uint32_t block_size,
    et_paged_kv_prefix_key_v1 *out_key);

/* Same registration and attachment ownership as the legacy APIs. Use the
 * matching API family for the pool mode (otherwise ET_PAGED_KV_ERR_STATE).
 * Key context and block size must match the pool (ET_PAGED_KV_ERR_CONTEXT
 * and ET_PAGED_KV_ERR_SHAPE respectively). Failures leave all pool state
 * and out_hit untouched. Keys are caller-controlled data. The caller must
 * supply the hash of the actual token chain and matching KV, with trustworthy
 * context fingerprints. The caller owns provenance for the stored KV. */
et_paged_kv_status et_paged_kv_prefix_cache_register_with_context_v1(
    et_paged_kv_pool *pool, uint32_t seq_id, uint32_t logical_block,
    const et_paged_kv_prefix_key_v1 *key);
et_paged_kv_status et_paged_kv_prefix_cache_attach_with_context_v1(
    et_paged_kv_pool *pool, uint32_t seq_id,
    const et_paged_kv_prefix_key_v1 *key, uint32_t *out_hit);

#ifdef __cplusplus
}
#endif

#endif
