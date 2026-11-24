#ifndef ESHKOL_TRANSFORMER_FLASH_ATTN2_ABI_H
#define ESHKOL_TRANSFORMER_FLASH_ATTN2_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_FA2_ABI_MAJOR 1u
#define ET_FA2_ABI_MINOR 0u

#define ET_FA2_MIN_BATCH 1u
#define ET_FA2_MAX_BATCH 4u
#define ET_FA2_MIN_SEQ_LEN 1u
#define ET_FA2_MAX_SEQ_LEN 256u
#define ET_FA2_MIN_HEADS 1u
#define ET_FA2_MAX_HEADS 16u
#define ET_FA2_MIN_HEAD_DIM 1u
#define ET_FA2_MAX_HEAD_DIM 64u
#define ET_FA2_MIN_TILE 8u
#define ET_FA2_MAX_TILE 32u

typedef int32_t et_fa2_status_t;

enum {
  ET_FA2_OK = 0,
  ET_FA2_ERR_INVALID_ARG = 1,
  ET_FA2_ERR_SHAPE_MISMATCH = 2,
  ET_FA2_ERR_UNSUPPORTED = 3,
  ET_FA2_ERR_NONFINITE = 4,
  ET_FA2_ERR_FP_ENV = 5
};

/*
 * Configuration for FlashAttention-2 Exact Tiled Online-Softmax Attention.
 *
 * Tensor layout (dense row-major, BHTD / BNSD):
 *   Q, O, dO, dQ : float[B, H_q, T_q, D_h]
 *   K, V, dK, dV : float[B, H_kv, T_k, D_h]
 *   LSE          : double[B, H_q, T_q]
 *
 * GQA / MQA head mapping:
 *   H_q % H_kv == 0, group = H_q / H_kv, kv_head = q_head / group.
 *
 * Admitted domain:
 *   - batch (B)          in [1, 4]
 *   - num_heads (H_q)    in [1, 16]
 *   - num_kv_heads (H_kv) in [1, H_q] with H_q % H_kv == 0
 *   - seq_q (T_q), seq_k (T_k) in [1, 256]
 *   - head_dim (D_h)     in [1, 64]
 *   - tile_r (B_r), tile_c (B_c) in {8, 16, 32}
 *   - causal             in {0, 1}
 *   - sliding_window     0 (disabled) or >= 1:
 *       * causal == 1: keeps 0 <= (q_pos - k_pos) < sliding_window
 *                      (when sliding_window == 0, keeps k_pos <= q_pos)
 *       * causal == 0: keeps |q_pos - k_pos| < sliding_window
 *                      (when sliding_window == 0, keeps all 0 <= k_pos < T_k)
 *     Every query row q_pos in [0, T_q - 1] must have at least one unmasked
 *     key in [0, T_k - 1], which requires (sliding_window == 0) ||
 *     (T_q <= T_k) || (T_q - T_k < sliding_window).
 *   - reserved           must be 0
 *   - scale              0.0 selects 1.0 / sqrt((double)D_h);
 *                        otherwise finite and strictly positive
 *   - logit_softcap      0.0 (disabled); otherwise finite > 0.0 applying
 *                        S = cap * tanh(S_raw / cap)
 */
typedef struct et_fa2_config_v1 {
  size_t struct_size;
  size_t batch;
  size_t num_heads;
  size_t num_kv_heads;
  size_t seq_q;
  size_t seq_k;
  size_t head_dim;
  size_t tile_r;
  size_t tile_c;
  size_t sliding_window;
  uint32_t causal;
  uint32_t reserved;
  double scale;
  double logit_softcap;
} et_fa2_config_v1;

/*
 * Execution telemetry for a tiled FlashAttention-2 forward or backward pass.
 *
 *   total_tiles        = B * H_q * ceil(T_q / B_r) * ceil(T_k / B_c)
 *   evaluated_tiles    + skipped_tiles == total_tiles
 *   q_tiles_per_head   = ceil(T_q / B_r)
 *   k_tiles_per_head   = ceil(T_k / B_c)
 *   tile_scratch_bytes = et_fa2_tile_scratch_bytes_v1(B_r, B_c), strictly
 *                        O(B_r * B_c) and independent of T_q and T_k.
 */
typedef struct et_fa2_stats_v1 {
  uint64_t total_tiles;
  uint64_t evaluated_tiles;
  uint64_t skipped_tiles;
  uint64_t q_tiles_per_head;
  uint64_t k_tiles_per_head;
  uint64_t tile_scratch_bytes;
} et_fa2_stats_v1;

/*
 * Returns the exact O(B_r * B_c) binary64 tile scratch footprint in bytes
 * (2 * B_r * B_c * sizeof(double)) for valid tile sizes B_r, B_c in {8, 16, 32},
 * completely independent of sequence lengths T_q and T_k.
 * Returns 0 if either tile dimension is unsupported.
 */
size_t et_fa2_tile_scratch_bytes_v1(size_t tile_r, size_t tile_c);

/*
 * Validate configuration and optionally write the effective attention scale.
 * Returns ET_FA2_OK on success.
 */
et_fa2_status_t et_fa2_validate_config_v1(const void *cfg_ptr,
                                          double *effective_scale_out);

/*
 * FlashAttention-2 Exact Tiled Online-Softmax Forward pass.
 *
 * Iterates Q tiles (outer, B_r) and KV tiles (inner, B_c), skips completely
 * masked tiles, rescales (m, l, O_acc) online in IEEE-754 binary64, and writes
 * only O[B, H_q, T_q, D_h] (f32) and LSE[B, H_q, T_q] (f64) without ever
 * materializing a T_q x T_k attention matrix.
 *
 * Zero heap allocations. Failure-atomic: all outputs remain byte-for-byte
 * untouched on any error.
 */
et_fa2_status_t et_fa2_forward_v1(const void *cfg_ptr, const void *q,
                                  const void *k, const void *v, void *o_out,
                                  void *lse_out, et_fa2_stats_v1 *stats_out);

/*
 * FlashAttention-2 Exact Recomputation Backward pass.
 *
 * Precomputes D[b, h, i] = sum_d dO[b, h, i, d] * O[b, h, i, d] in binary64,
 * iterates KV tiles (outer, B_c) and Q tiles (inner, B_r), skips completely
 * masked tiles, recomputes S_tile and P_tile = exp(S_tile - LSE) on-the-fly,
 * applies the exact Gemma-2 softcap derivative (1 - tanh(S_raw / cap)^2) when
 * logit_softcap > 0, and accumulates dQ, dK, dV (with exact GQA head-group
 * reduction into H_kv) without saving or materializing a T_q x T_k matrix.
 *
 * Zero heap allocations. Failure-atomic: all outputs remain byte-for-byte
 * untouched on any error.
 */
et_fa2_status_t et_fa2_backward_v1(const void *cfg_ptr, const void *q,
                                   const void *k, const void *v, const void *o,
                                   const void *lse, const void *do_in,
                                   void *dq_out, void *dk_out, void *dv_out,
                                   et_fa2_stats_v1 *stats_out);

#ifdef __cplusplus
}
#endif

#endif
