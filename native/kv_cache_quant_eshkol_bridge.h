#ifndef ESHKOL_TRANSFORMER_KV_CACHE_QUANT_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_KV_CACHE_QUANT_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kv_cache_quant_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_kvq_aot_ctx et_kvq_aot_ctx;

/* Context lifecycle for compiled Eshkol AOT Quantized KV-Cache driver.
 * All scalar parameters and returns use i32/i64/f64 for direct Eshkol FFI
 * compatibility without struct-by-value or string marshaling fragility. */
int32_t et_kvq_aot_create(int64_t format, int64_t num_layers,
                          int64_t num_kv_heads, int64_t head_dim,
                          int64_t max_seq_len, et_kvq_aot_ctx **out_ctx);
void et_kvq_aot_destroy(et_kvq_aot_ctx *ctx);
int32_t et_kvq_aot_reset(et_kvq_aot_ctx *ctx);
int32_t et_kvq_aot_truncate(et_kvq_aot_ctx *ctx, int64_t new_seq_len);

/* State & memory accounting queries. */
int64_t et_kvq_aot_layer_seq_len(const et_kvq_aot_ctx *ctx, int64_t layer);
int64_t et_kvq_aot_stat_fp32_bytes(const et_kvq_aot_ctx *ctx);
int64_t et_kvq_aot_stat_quant_bytes(const et_kvq_aot_ctx *ctx);
double et_kvq_aot_stat_compression_ratio(const et_kvq_aot_ctx *ctx);

/* Staging & appending K/V token slices [num_kv_heads, head_dim] at layer. */
int32_t et_kvq_aot_set_k(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                         double value);
int32_t et_kvq_aot_set_v(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                         double value);
int32_t et_kvq_aot_set_k_bits(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                              int64_t f32_bits);
int32_t et_kvq_aot_append_token(et_kvq_aot_ctx *ctx, int64_t layer);

/* Dequantize stored (layer, pos) into context's deq_k/deq_v staging buffers,
 * and optionally append those dequantized slices into a destination FP32 ctx. */
int32_t et_kvq_aot_dequant_slice(et_kvq_aot_ctx *ctx, int64_t layer,
                                 int64_t pos);
double et_kvq_aot_get_deq_k(const et_kvq_aot_ctx *ctx, int64_t kv_head,
                            int64_t dim);
double et_kvq_aot_get_deq_v(const et_kvq_aot_ctx *ctx, int64_t kv_head,
                            int64_t dim);
int32_t et_kvq_aot_copy_dequant_to_fp32(et_kvq_aot_ctx *src_ctx,
                                        et_kvq_aot_ctx *dst_fp32_ctx,
                                        int64_t layer, int64_t pos);

/* Staging Q[num_q_heads, head_dim], executing fused quantized online-softmax
 * GQA decode, and inspecting binary32 outputs (float value or raw u32 bits). */
int32_t et_kvq_aot_set_q(et_kvq_aot_ctx *ctx, int64_t q_head, int64_t dim,
                         double value);
int32_t et_kvq_aot_set_q_bits(et_kvq_aot_ctx *ctx, int64_t q_head, int64_t dim,
                              int64_t f32_bits);
int32_t et_kvq_aot_decode_gqa(et_kvq_aot_ctx *ctx, int64_t layer,
                              int64_t num_q_heads, double scale,
                              int64_t window_size);
double et_kvq_aot_get_out(const et_kvq_aot_ctx *ctx, int64_t q_head,
                          int64_t dim);
int64_t et_kvq_aot_get_out_bits(const et_kvq_aot_ctx *ctx, int64_t q_head,
                                int64_t dim);

/* Snapshot save/load/tamper helpers using bounded slot indices [0..7] inside
 * ET_KVQ_SNAPSHOT_DIR (so compiled AOT binaries never embed host paths). */
int32_t et_kvq_aot_save_snapshot_slot(const et_kvq_aot_ctx *ctx, int64_t slot);
int32_t et_kvq_aot_load_snapshot_slot(et_kvq_aot_ctx *ctx, int64_t slot);
int32_t et_kvq_aot_tamper_snapshot_slot(int64_t slot, int64_t byte_offset);

/* Negative FP-environment trap check: temporarily sets FE_TOWARDZERO, invokes
 * et_kvq_append_token_v1 and et_kvq_decode_gqa_v1, restores FE_TONEAREST, and
 * returns 0 iff both returned ET_KVQ_ERR_FP_ENV without mutating state. */
int32_t et_kvq_aot_check_fp_env_trap(et_kvq_aot_ctx *ctx, int64_t layer,
                                     int64_t num_q_heads, double scale);

#ifdef __cplusplus
}
#endif

#endif
