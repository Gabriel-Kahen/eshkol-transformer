#ifndef ESHKOL_TRANSFORMER_KV_CACHE_QUANT_ABI_H
#define ESHKOL_TRANSFORMER_KV_CACHE_QUANT_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_KVQ_ABI_MAJOR 1u
#define ET_KVQ_ABI_MINOR 0u

/* Bounded explicit geometry limits (zero silent fallback). */
#define ET_KVQ_MAX_LAYERS 64u
#define ET_KVQ_MAX_KV_HEADS 64u
#define ET_KVQ_MAX_Q_HEADS 64u
#define ET_KVQ_MAX_HEAD_DIM 256u
#define ET_KVQ_MAX_SEQ_LEN 65536u
#define ET_KVQ_MAX_STORAGE_BYTES                                               \
  (UINT64_C(256) * UINT64_C(1024) * UINT64_C(1024))

/* OCP FP8 E4M3FN constants (v1.0: 1 sign, 4 exp bias 7, 3 mantissa, no Inf). */
#define ET_KVQ_FP8_E4M3_MAX_FINITE 448.0f
#define ET_KVQ_FP8_E4M3_MIN_SUBNORMAL 0.001953125f

/* Versioned SHA-256 checksummed KV-cache snapshot format (ETKQ1000 v1.0). */
#define ET_KVQ_SNAPSHOT_MAGIC "ETKQ1000"
#define ET_KVQ_SNAPSHOT_MAGIC_BYTES 8u
#define ET_KVQ_SNAPSHOT_VERSION_MAJOR 1u
#define ET_KVQ_SNAPSHOT_VERSION_MINOR 0u
#define ET_KVQ_SNAPSHOT_HEADER_BYTES 64u
#define ET_KVQ_SNAPSHOT_CHECKSUM_BYTES 32u

typedef uint32_t et_kvq_format;
enum {
  ET_KVQ_FMT_FP32 = 1u,
  ET_KVQ_FMT_INT8_SYM = 2u,
  ET_KVQ_FMT_FP8_E4M3FN = 3u
};

typedef int32_t et_kvq_status;
enum {
  ET_KVQ_OK = 0,
  ET_KVQ_ERR_INVALID_ARG = -1,
  ET_KVQ_ERR_SHAPE = -2,
  ET_KVQ_ERR_UNSUPPORTED_FORMAT = -3,
  ET_KVQ_ERR_NONFINITE = -4,
  ET_KVQ_ERR_CAPACITY = -5,
  ET_KVQ_ERR_OUT_OF_RANGE = -6,
  ET_KVQ_ERR_ALIAS = -7,
  ET_KVQ_ERR_ALIGNMENT = -8,
  ET_KVQ_ERR_FP_ENV = -9,
  ET_KVQ_ERR_ALLOC = -10,
  ET_KVQ_ERR_OVERFLOW = -11,
  ET_KVQ_ERR_IO = -12,
  ET_KVQ_ERR_CORRUPT = -13,
  ET_KVQ_ERR_CHECKSUM = -14,
  ET_KVQ_ERR_FORMAT_MISMATCH = -15,
  ET_KVQ_ERR_STATE = -16
};

/* Unified quantized KV-cache configuration.
 *
 * Logical layout per layer and KV side (0=K, 1=V):
 *   [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
 *
 * Per-vector storage bytes (per (layer, kv_side, pos, kv_head)):
 *   ET_KVQ_FMT_FP32       : 4 * head_dim bytes
 *   ET_KVQ_FMT_INT8_SYM   : head_dim int8 bytes + 4-byte f32 scale
 *   ET_KVQ_FMT_FP8_E4M3FN : head_dim uint8 bytes + 4-byte f32 scale
 *
 * Constraints:
 *   format in {ET_KVQ_FMT_FP32, ET_KVQ_FMT_INT8_SYM, ET_KVQ_FMT_FP8_E4M3FN}
 *   1 <= num_layers   <= ET_KVQ_MAX_LAYERS
 *   1 <= num_kv_heads <= ET_KVQ_MAX_KV_HEADS
 *   1 <= head_dim     <= ET_KVQ_MAX_HEAD_DIM
 *   1 <= max_seq_len  <= ET_KVQ_MAX_SEQ_LEN
 *   reserved == 0
 *   total backing storage <= ET_KVQ_MAX_STORAGE_BYTES
 */
typedef struct et_kvq_config {
  et_kvq_format format;
  uint32_t num_layers;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t max_seq_len;
  uint32_t reserved;
} et_kvq_config;

/* Exact memory accounting and compression statistics.
 *
 * fp32_baseline_bytes  = active_layer_tokens * 2 * num_kv_heads * (4 * head_dim)
 * quantized_bytes      = active_layer_tokens * 2 * num_kv_heads * bytes_per_vec
 * capacity_fp32_bytes  = max_seq_len * num_layers * 2 * num_kv_heads * (4 * head_dim)
 * capacity_quant_bytes = max_seq_len * num_layers * 2 * num_kv_heads * bytes_per_vec
 * compression_ratio    = (double)capacity_fp32_bytes / (double)capacity_quant_bytes
 *                        (identical to fp32_baseline_bytes / quantized_bytes
 *                        whenever active_layer_tokens > 0).
 */
typedef struct et_kvq_memory_stats {
  uint64_t fp32_baseline_bytes;
  uint64_t quantized_bytes;
  uint64_t capacity_fp32_bytes;
  uint64_t capacity_quant_bytes;
  double compression_ratio;
  uint32_t min_seq_len;
  uint32_t max_seq_len_used;
} et_kvq_memory_stats;

#if defined(__cplusplus)
#define ET_KVQ_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_KVQ_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_KVQ_STATIC_ASSERT(sizeof(et_kvq_config) == 24u &&
                         offsetof(et_kvq_config, reserved) == 20u,
                     "et_kvq_config ABI layout changed");
ET_KVQ_STATIC_ASSERT(sizeof(et_kvq_memory_stats) == 48u &&
                         offsetof(et_kvq_memory_stats, compression_ratio) ==
                             32u &&
                         offsetof(et_kvq_memory_stats, min_seq_len) == 40u &&
                         offsetof(et_kvq_memory_stats, max_seq_len_used) == 44u,
                     "et_kvq_memory_stats ABI layout changed");
#undef ET_KVQ_STATIC_ASSERT

typedef struct et_kvq_cache et_kvq_cache;

/* ---- Configuration & Backing Pool Sizing ---- */

et_kvq_status et_kvq_validate_config(const et_kvq_config *cfg);

et_kvq_status et_kvq_storage_bytes_required(const et_kvq_config *cfg,
                                            size_t *out_bytes);

/* ---- Cache Lifecycle & State Queries ----
 * If external_pool == NULL: external_pool_bytes must be 0 and the cache owns
 * its upfront heap-allocated backing storage.
 * If external_pool != NULL: must be 4-byte aligned, non-overlapping with cfg
 * and out_cache, and external_pool_bytes >= et_kvq_storage_bytes_required(cfg);
 * caller retains ownership and keeps external_pool alive until destroy().
 */
et_kvq_status et_kvq_cache_create(const et_kvq_config *cfg, void *external_pool,
                                  size_t external_pool_bytes,
                                  et_kvq_cache **out_cache);

void et_kvq_cache_destroy(et_kvq_cache *cache);

/* Zero heap allocations; resets all layer lengths to 0 and clears storage. */
et_kvq_status et_kvq_cache_reset(et_kvq_cache *cache);

/* Zero heap allocations; truncates every layer to new_seq_len <= layer_seq_len[l]
 * and deterministically clears truncated tail slots. Fails atomically with
 * ET_KVQ_ERR_OUT_OF_RANGE if new_seq_len exceeds any layer's current length. */
et_kvq_status et_kvq_cache_truncate(et_kvq_cache *cache, uint32_t new_seq_len);

/* Zero heap allocations; reports exact byte counts and compression ratio. */
et_kvq_status et_kvq_cache_memory_stats(const et_kvq_cache *cache,
                                        et_kvq_memory_stats *out_stats);

et_kvq_status et_kvq_cache_config_get(const et_kvq_cache *cache,
                                      et_kvq_config *out_cfg);

et_kvq_status et_kvq_cache_layer_seq_len(const et_kvq_cache *cache,
                                         uint32_t layer,
                                         uint32_t *out_seq_len);

/* ---- Hot-Path Quantized Append, Dequantize & Fused GQA Decode ----
 * All three kernels perform ZERO heap allocations, verify IEEE-754 FE_TONEAREST
 * + non-FTZ/DAZ FP environment, enforce finite-only inputs, and are strictly
 * failure-atomic (cache state and caller outputs remain untouched on error).
 */

/* Quantize and append one token slice (k_new[H_kv, D_h], v_new[H_kv, D_h]) at
 * layer. Fails atomically on non-finite input or capacity full. */
et_kvq_status et_kvq_append_token_v1(et_kvq_cache *cache, uint32_t layer,
                                     const float *k_new, const float *v_new);

/* Exact bit-for-bit dequantization of stored (layer, pos) K and V slices into
 * out_k[H_kv, D_h] and out_v[H_kv, D_h]. */
et_kvq_status et_kvq_dequant_slice_v1(const et_kvq_cache *cache, uint32_t layer,
                                      uint32_t pos, float *out_k, float *out_v);

/* Fused quantized online-softmax GQA decode over t in [t_start, seq_len - 1],
 * dequantizing K and V on the fly in registers/stack without ever materializing
 * a full FP32 KV cache.
 *
 * Tensor shapes:
 *   q   : [num_q_heads, head_dim]
 *   out : [num_q_heads, head_dim]
 *
 * window_size:
 *   0   : full causal context t in [0, seq_len - 1]
 *   > 0 : sliding window over the last min(window_size, seq_len) tokens
 *
 * Produces bitwise identical binary32 outputs (0.0 error, 0 ULP) compared to
 * running the exact same online-softmax GQA decode over explicitly dequantized
 * K/V slices.
 */
et_kvq_status et_kvq_decode_gqa_v1(const et_kvq_cache *cache, uint32_t layer,
                                   uint32_t num_q_heads, float scale,
                                   uint32_t window_size, const float *q,
                                   float *out);

/* Read raw stored bytes and scale for (layer, is_v in {0,1}, pos, kv_head).
 * For ET_KVQ_FMT_FP32, payload is 4 * head_dim bytes and *out_scale = 1.0f.
 * For INT8_SYM and FP8_E4M3FN, payload is head_dim bytes and *out_scale is
 * the exact stored per-vector f32 scale. Zero heap allocations. */
et_kvq_status et_kvq_read_raw_vector(const et_kvq_cache *cache, uint32_t layer,
                                     uint32_t is_v, uint32_t pos,
                                     uint32_t kv_head, uint8_t *out_payload,
                                     size_t payload_capacity, float *out_scale);

/* ---- Bit-Exact Scalar/Vector Codecs (Zero Heap Allocations) ---- */

et_kvq_status et_kvq_fp8_e4m3_encode_scalar(float value, uint8_t *out_code);

et_kvq_status et_kvq_fp8_e4m3_decode_scalar(uint8_t code, float *out_value);

et_kvq_status et_kvq_int8_sym_quantize_vec(const float *src, size_t count,
                                           int8_t *dst_q, float *out_scale);

et_kvq_status et_kvq_int8_sym_dequantize_vec(const int8_t *src_q, size_t count,
                                             float scale, float *dst);

et_kvq_status et_kvq_fp8_e4m3_quantize_vec(const float *src, size_t count,
                                           uint8_t *dst_q, float *out_scale);

et_kvq_status et_kvq_fp8_e4m3_dequantize_vec(const uint8_t *src_q, size_t count,
                                             float scale, float *dst);

/* ---- Versioned SHA-256 Checksummed Cache Snapshot (ETKQ1000 v1.0) ----
 * Save writes atomically via a 0600 temp file + fsync + renameat + dir fsync,
 * rejecting existing symlink or non-regular destinations.
 * Load opens with O_NOFOLLOW, verifies regular file, exact length, magic,
 * version, SHA-256 checksum, format, geometry, and payload finiteness before
 * mutating cache (strictly failure-atomic).
 */
et_kvq_status et_kvq_cache_save_snapshot(const et_kvq_cache *cache,
                                         const char *path);

et_kvq_status et_kvq_cache_load_snapshot(et_kvq_cache *cache, const char *path);

#ifdef __cplusplus
}
#endif

#endif
