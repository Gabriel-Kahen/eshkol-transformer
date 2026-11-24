#ifndef ESHKOL_TRANSFORMER_PAGED_KV_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_PAGED_KV_ESHKOL_BRIDGE_H

#include "eshkol_transformer/paged_kv_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_paged_kv_aot_ctx et_paged_kv_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT PagedAttention / CoW KV-cache
 * verification. All scalar arguments/returns use i32/i64/f64 for direct
 * Eshkol FFI compatibility. */
int32_t et_paged_kv_aot_create(int64_t num_blocks, int64_t num_layers,
                               int64_t num_kv_heads, int64_t block_size,
                               int64_t head_dim, int64_t max_seqs,
                               int64_t max_blocks_per_seq,
                               et_paged_kv_aot_ctx **out_ctx);
void et_paged_kv_aot_destroy(et_paged_kv_aot_ctx *ctx);
int32_t et_paged_kv_aot_reset(et_paged_kv_aot_ctx *ctx);

/* Sequence management (returns seq_id >= 0 on success, negative status on err). */
int64_t et_paged_kv_aot_seq_alloc(et_paged_kv_aot_ctx *ctx);
int32_t et_paged_kv_aot_seq_free(et_paged_kv_aot_ctx *ctx, int64_t seq_id);
int64_t et_paged_kv_aot_seq_fork(et_paged_kv_aot_ctx *ctx,
                                 int64_t parent_seq_id);
int32_t et_paged_kv_aot_seq_truncate(et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                                     int64_t new_len);

/* Sequence and physical block table inspection. */
int64_t et_paged_kv_aot_seq_len(const et_paged_kv_aot_ctx *ctx, int64_t seq_id);
int64_t et_paged_kv_aot_seq_num_blocks(const et_paged_kv_aot_ctx *ctx,
                                       int64_t seq_id);
int64_t et_paged_kv_aot_seq_phys_block(const et_paged_kv_aot_ctx *ctx,
                                       int64_t seq_id, int64_t logical_block);
int64_t et_paged_kv_aot_block_refcount(const et_paged_kv_aot_ctx *ctx,
                                       int64_t block_id);

/* Staging & appending K/V token tensors [num_layers, num_kv_heads, head_dim]. */
int32_t et_paged_kv_aot_set_k(et_paged_kv_aot_ctx *ctx, int64_t layer,
                              int64_t kv_head, int64_t dim, double value);
int32_t et_paged_kv_aot_set_v(et_paged_kv_aot_ctx *ctx, int64_t layer,
                              int64_t kv_head, int64_t dim, double value);
int32_t et_paged_kv_aot_append_token(et_paged_kv_aot_ctx *ctx, int64_t seq_id);

/* Staging Q[num_q_heads, head_dim], executing paged GQA causal decode
 * attention, and reading back out[num_q_heads, head_dim]. */
int32_t et_paged_kv_aot_set_q(et_paged_kv_aot_ctx *ctx, int64_t q_head,
                              int64_t dim, double value);
int32_t et_paged_kv_aot_decode_attention(et_paged_kv_aot_ctx *ctx,
                                         int64_t seq_id, int64_t layer_idx,
                                         int64_t num_q_heads, double scale);
double et_paged_kv_aot_get_out(const et_paged_kv_aot_ctx *ctx, int64_t q_head,
                               int64_t dim);
int64_t et_paged_kv_aot_get_out_bits(const et_paged_kv_aot_ctx *ctx,
                                     int64_t q_head, int64_t dim);

/* Read back cached K or V float element (is_v = 0 for K, 1 for V). */
double et_paged_kv_aot_read_kv(const et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                               int64_t pos, int64_t layer_idx,
                               int64_t kv_head, int64_t is_v, int64_t dim);

/* Prefix block hashing & deduplication helpers (hash slot 0..3). */
int32_t et_paged_kv_aot_set_token_id(et_paged_kv_aot_ctx *ctx, int64_t slot,
                                     int64_t token_id);
int32_t et_paged_kv_aot_hash_block(et_paged_kv_aot_ctx *ctx, int64_t prev_slot,
                                   int64_t out_slot);
int32_t et_paged_kv_aot_prefix_register(et_paged_kv_aot_ctx *ctx,
                                        int64_t seq_id, int64_t logical_block,
                                        int64_t hash_slot);
/* Returns 1 on hit, 0 on miss, or negative status on error. */
int64_t et_paged_kv_aot_prefix_attach(et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                                      int64_t hash_slot);

/* Pool telemetry getters. */
int64_t et_paged_kv_aot_stat_total_blocks(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_free_blocks(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_used_blocks(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_active_seqs(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_shared_blocks(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_prefix_cached(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_cow_copies(const et_paged_kv_aot_ctx *ctx);
int64_t et_paged_kv_aot_stat_prefix_hits(const et_paged_kv_aot_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
