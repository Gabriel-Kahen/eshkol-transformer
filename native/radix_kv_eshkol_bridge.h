#ifndef ESHKOL_TRANSFORMER_RADIX_KV_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_RADIX_KV_ESHKOL_BRIDGE_H

#include "eshkol_transformer/radix_kv_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_radix_kv_aot_ctx et_radix_kv_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT RadixAttention verification.
 * All scalar arguments/returns use i32/i64/f64 for direct Eshkol FFI. */
int32_t et_radix_kv_aot_create(int64_t max_kv_tokens, int64_t max_nodes,
                               int64_t n_layers, int64_t n_kv_heads,
                               int64_t head_dim,
                               et_radix_kv_aot_ctx **out_ctx);
void et_radix_kv_aot_destroy(et_radix_kv_aot_ctx *ctx);
int32_t et_radix_kv_aot_reset(et_radix_kv_aot_ctx *ctx);

/* Root handle getter. */
int64_t et_radix_kv_aot_root_handle(const et_radix_kv_aot_ctx *ctx);

/* Input staging buffers (tokens[pos], K/V[pos, layer, kv_head, dim],
 * Q[q_pos, q_head, dim]). */
int32_t et_radix_kv_aot_set_token(et_radix_kv_aot_ctx *ctx, int64_t pos,
                                  int64_t token_id);
int32_t et_radix_kv_aot_set_kv(et_radix_kv_aot_ctx *ctx, int64_t pos,
                               int64_t layer, int64_t kv_head, int64_t dim,
                               int64_t is_v, double value);
int32_t et_radix_kv_aot_set_q(et_radix_kv_aot_ctx *ctx, int64_t q_pos,
                              int64_t q_head, int64_t dim, double value);

/* Core radix tree operations. */
int32_t et_radix_kv_aot_match_prefix(et_radix_kv_aot_ctx *ctx, int64_t seq_len);
int32_t et_radix_kv_aot_insert(et_radix_kv_aot_ctx *ctx, int64_t seq_len);
int32_t et_radix_kv_aot_insert_suffix(et_radix_kv_aot_ctx *ctx,
                                      int64_t prefix_handle,
                                      int64_t suffix_len);
int32_t et_radix_kv_aot_pin(et_radix_kv_aot_ctx *ctx, int64_t handle);
int32_t et_radix_kv_aot_unpin(et_radix_kv_aot_ctx *ctx, int64_t handle);
int64_t et_radix_kv_aot_evict_lru(et_radix_kv_aot_ctx *ctx,
                                  int64_t target_free_tokens);
int32_t et_radix_kv_aot_verify_invariants(const et_radix_kv_aot_ctx *ctx);

/* Last-operation & handle inspection getters. */
int64_t et_radix_kv_aot_last_handle(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_last_matched_len(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_last_did_split(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_last_inserted_tokens(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_last_kv_slot(const et_radix_kv_aot_ctx *ctx,
                                     int64_t pos);
int64_t et_radix_kv_aot_handle_depth(const et_radix_kv_aot_ctx *ctx,
                                     int64_t handle);
int64_t et_radix_kv_aot_handle_pin_count(const et_radix_kv_aot_ctx *ctx,
                                         int64_t handle);

/* Cached KV gather & exact binary64 causal GQA attention. */
int32_t et_radix_kv_aot_read_kv(et_radix_kv_aot_ctx *ctx, int64_t handle,
                                int64_t layer);
double et_radix_kv_aot_get_read_kv(const et_radix_kv_aot_ctx *ctx, int64_t pos,
                                   int64_t kv_head, int64_t dim, int64_t is_v);
int32_t et_radix_kv_aot_attend_gqa(et_radix_kv_aot_ctx *ctx,
                                   int64_t prefix_handle, int64_t layer,
                                   int64_t n_q_heads, int64_t q_len,
                                   int64_t suffix_len, double scale);
double et_radix_kv_aot_get_out(const et_radix_kv_aot_ctx *ctx, int64_t q_pos,
                               int64_t q_head, int64_t dim);
int64_t et_radix_kv_aot_get_out_bits(const et_radix_kv_aot_ctx *ctx,
                                     int64_t q_pos, int64_t q_head,
                                     int64_t dim);

/* ETRX1000 snapshot save, reload into fresh tree, and tamper/truncation check
 * using path in ET_RADIX_KV_AOT_SNAPSHOT_PATH. */
int32_t et_radix_kv_aot_snapshot_save_env(const et_radix_kv_aot_ctx *ctx);
int32_t et_radix_kv_aot_snapshot_reload_env(et_radix_kv_aot_ctx *ctx);
int32_t et_radix_kv_aot_snapshot_tamper_check_env(
    const et_radix_kv_aot_ctx *ctx);

/* Live telemetry getters. */
int64_t et_radix_kv_aot_stat_used_tokens(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_free_tokens(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_pinned_tokens(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_evictable_tokens(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_active_nodes(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_lru_tick(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_total_matched_tokens(
    const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_total_inserted_tokens(
    const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_total_evicted_tokens(
    const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_total_edge_splits(const et_radix_kv_aot_ctx *ctx);
int64_t et_radix_kv_aot_stat_total_oom_rejections(
    const et_radix_kv_aot_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif
