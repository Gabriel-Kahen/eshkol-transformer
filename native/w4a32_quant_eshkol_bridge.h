#ifndef ESHKOL_TRANSFORMER_W4A32_QUANT_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_W4A32_QUANT_ESHKOL_BRIDGE_H

#include "eshkol_transformer/w4a32_quant_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_w4a32_aot_ctx et_w4a32_aot_ctx;

/* Create an AOT 2-layer MLP quantization & fused-GEMM verification context:
 *   Layer 1: W1[d_hidden, d_in], b1[d_hidden]
 *   Layer 2: W2[d_out, d_hidden], b2[d_out]
 *   Prefill batch M = max_m (e.g., 8), Decode M = 1 (uses row 0 of X).
 *   format: ET_W4A32_FORMAT_Q4_0 (1) or ET_W4A32_FORMAT_Q4_ASYM (2).
 *   group_size: 32 for Q4_0; 32, 64, or 128 for Q4_ASYM. */
int32_t et_w4a32_aot_create(int64_t max_m, int64_t d_in, int64_t d_hidden,
                            int64_t d_out, int64_t format, int64_t group_size,
                            et_w4a32_aot_ctx **out_ctx);
void et_w4a32_aot_destroy(et_w4a32_aot_ctx *ctx);

/* Element and byte counts for compression-ratio and shape checks. */
int64_t et_w4a32_aot_x_count(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_w1_count(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_w2_count(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_b1_count(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_b2_count(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_y_count(const et_w4a32_aot_ctx *ctx, int64_t m);
int64_t et_w4a32_aot_f32_weight_bytes(const et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_packed_weight_bytes(const et_w4a32_aot_ctx *ctx);

/* Tensor population accessors (driven directly from compiled Eshkol AOT). */
int32_t et_w4a32_aot_set_x(et_w4a32_aot_ctx *ctx, int64_t index, double value);
int32_t et_w4a32_aot_set_w1(et_w4a32_aot_ctx *ctx, int64_t index, double value);
int32_t et_w4a32_aot_set_b1(et_w4a32_aot_ctx *ctx, int64_t index, double value);
int32_t et_w4a32_aot_set_w2(et_w4a32_aot_ctx *ctx, int64_t index, double value);
int32_t et_w4a32_aot_set_b2(et_w4a32_aot_ctx *ctx, int64_t index, double value);

/* Quantize W1, W2 -> packed blocks, and dequantize packed blocks -> W1_deq, W2_deq. */
int32_t et_w4a32_aot_quantize_all(et_w4a32_aot_ctx *ctx);
int32_t et_w4a32_aot_dequantize_all(et_w4a32_aot_ctx *ctx);

/* Weight reconstruction inspection (orig vs dequantized). */
double et_w4a32_aot_get_w1(const et_w4a32_aot_ctx *ctx, int64_t index);
double et_w4a32_aot_get_w1_deq(const et_w4a32_aot_ctx *ctx, int64_t index);
double et_w4a32_aot_get_w2(const et_w4a32_aot_ctx *ctx, int64_t index);
double et_w4a32_aot_get_w2_deq(const et_w4a32_aot_ctx *ctx, int64_t index);

/* 2-layer MLP forward passes for batch size m in [1, max_m]:
 *   H = ReLU(X[m, d_in] @ W1^T + b1)
 *   Y = H @ W2^T + b2
 * _fused runs directly on packed 4-bit blocks via et_w4a32_*_gemm_v1.
 * _deq_ref runs dense f32 GEMM on dequant(W_q) via et_w4a32_f32_gemm_v1. */
int32_t et_w4a32_aot_mlp_forward_fused(et_w4a32_aot_ctx *ctx, int64_t m);
int32_t et_w4a32_aot_mlp_forward_deq_ref(et_w4a32_aot_ctx *ctx, int64_t m);

double et_w4a32_aot_get_y_fused(const et_w4a32_aot_ctx *ctx, int64_t index);
double et_w4a32_aot_get_y_ref(const et_w4a32_aot_ctx *ctx, int64_t index);
int64_t et_w4a32_aot_y_fused_bits(const et_w4a32_aot_ctx *ctx, int64_t index);

/* SHA-256 digests of fused output Y[m, d_out] (canonical LE binary32) and
 * packed weights W1_q || W2_q. word_index in [0, 3]. */
int32_t et_w4a32_aot_compute_y_sha256(et_w4a32_aot_ctx *ctx, int64_t m);
int64_t et_w4a32_aot_y_sha256_word(const et_w4a32_aot_ctx *ctx,
                                   int64_t word_index);
int32_t et_w4a32_aot_compute_packed_sha256(et_w4a32_aot_ctx *ctx);
int64_t et_w4a32_aot_packed_sha256_word(const et_w4a32_aot_ctx *ctx,
                                        int64_t word_index);

/* Multi-process ETQ41000 checkpoint save/load driven by environment variables:
 *   W4A32_AOT_PHASE: "both" (0), "save" (1), "reload" (2)
 *   W4A32_AOT_DIR:   directory for per-slot .etq4 checkpoint files
 * slot_id: 0 = Q4_0 layer 1, 1 = Q4_0 layer 2, 2 = Q4_ASYM layer 1,
 *          3 = Q4_ASYM layer 2. */
int64_t et_w4a32_aot_env_phase(void);
int32_t et_w4a32_aot_save_env_pair(const et_w4a32_aot_ctx *ctx,
                                   int64_t pair_id);
int32_t et_w4a32_aot_load_env_pair(et_w4a32_aot_ctx *ctx, int64_t pair_id);

/* Negative verification helpers returning exact et_w4a32_status codes:
 *   kind 1: tampered payload byte (ET_W4A32_ERR_CHECKSUM)
 *   kind 2: truncated container (ET_W4A32_ERR_TRUNCATED)
 *   kind 3: bad magic (ET_W4A32_ERR_FORMAT)
 *   kind 4: bad version (ET_W4A32_ERR_VERSION)
 *   kind 5: symlink path (ET_W4A32_ERR_SYMLINK)
 *   kind 6: wrong format on reload (ET_W4A32_ERR_FORMAT)
 *   kind 7: wrong shape on reload (ET_W4A32_ERR_SHAPE)
 *   kind 8: K % 32 != 0 quantize rejection (ET_W4A32_ERR_SHAPE)
 *   kind 9: f16 scale overflow on quantize (ET_W4A32_ERR_OVERFLOW)
 *   kind 10: non-finite f16 scale in GEMM + output preserved (ET_W4A32_ERR_NONFINITE)
 *   kind 11: non-finite X in GEMM + output preserved (ET_W4A32_ERR_NONFINITE)
 *   kind 12: aliased buffers in GEMM + output preserved (ET_W4A32_ERR_ALIAS) */
int32_t et_w4a32_aot_check_negative(const et_w4a32_aot_ctx *ctx,
                                    int64_t pair_id, int64_t kind);

#ifdef __cplusplus
}
#endif

#endif
