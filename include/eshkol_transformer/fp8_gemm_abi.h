#ifndef ESHKOL_TRANSFORMER_FP8_GEMM_ABI_H
#define ESHKOL_TRANSFORMER_FP8_GEMM_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_FP8_GEMM_ABI_MAJOR 1u
#define ET_FP8_GEMM_ABI_MINOR 0u

/* OCP FP8 Format Specifications (v1.0):
 * - E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa. NO Infinity.
 *   Max finite = 448.0 (0x7E / 0xFE), min subnormal = 2^-9 = 0.001953125 (0x01),
 *   NaN = 0x7F / 0xFF, signed zero = 0x00 / 0x80.
 * - E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa.
 *   Max finite = 57344.0 (0x7B / 0xFB), min subnormal = 2^-16 (0x01),
 *   Inf = 0x7C / 0xFC, NaN = 0x7D..0x7F / 0xFD..0xFF, signed zero = 0x00 / 0x80.
 */
#define ET_FP8_E4M3_MAX_FINITE 448.0f
#define ET_FP8_E4M3_MIN_SUBNORMAL 0.001953125f
#define ET_FP8_E5M2_MAX_FINITE 57344.0f
#define ET_FP8_E5M2_MIN_SUBNORMAL 0.0000152587890625f
#define ET_FP8_DELAYED_SCALE_MAX_HISTORY 64u
#define ET_FP8_DELAYED_SCALE_MAX_MARGIN 16u

enum {
  ET_FP8_OK = 0,
  ET_FP8_ERR_INVALID_ARGUMENT = -1,
  ET_FP8_ERR_NONFINITE = -2,
  ET_FP8_ERR_OVERFLOW = -3,
  ET_FP8_ERR_UNSUPPORTED_FORMAT = -4,
  ET_FP8_ERR_UNSUPPORTED_BLOCK = -5,
  ET_FP8_ERR_SHAPE = -6,
  ET_FP8_ERR_ALIAS = -7,
  ET_FP8_ERR_ENVIRONMENT = -8
};
typedef int32_t et_fp8_status;

enum {
  ET_FP8_FORMAT_E4M3FN = 1,
  ET_FP8_FORMAT_E5M2 = 2
};
typedef uint32_t et_fp8_format;

enum {
  ET_FP8_OVERFLOW_SATURATE_TO_MAX = 0,
  ET_FP8_OVERFLOW_REJECT = 1
};
typedef uint32_t et_fp8_overflow_mode;

/* Delayed-scaling ring-buffer state for per-tensor amax history tracking.
 * Scale formula (evaluated in binary64, clamped to [min_scale, max_scale]):
 *   scale = (fp8_max / max(history_amax, eps)) * 2^(-margin)
 */
typedef struct et_fp8_delayed_scale_state_v1 {
  uint32_t abi_version;
  et_fp8_format format;
  uint32_t history_len;
  uint32_t cursor;
  uint32_t count;
  uint32_t margin;
  float eps;
  float min_scale;
  float max_scale;
  float current_scale;
  float history[ET_FP8_DELAYED_SCALE_MAX_HISTORY];
} et_fp8_delayed_scale_state_v1;

/* ---- Bit-Exact OCP FP8 Codecs (IEEE-754 Round-to-Nearest-Even) ----
 * All codec calls perform zero heap allocations, use byte-wise alignment-safe
 * loads/stores, reject non-finite f32 inputs on encode, reject NaN/Inf FP8
 * bit patterns on decode, and preserve caller output buffers untouched on error.
 */
et_fp8_status et_fp8_e4m3_encode_scalar_v1(float value,
                                           et_fp8_overflow_mode mode,
                                           uint8_t *out_code);

et_fp8_status et_fp8_e4m3_decode_scalar_v1(uint8_t code, float *out_value);

et_fp8_status et_fp8_e4m3_encode_v1(const float *src, size_t count,
                                    et_fp8_overflow_mode mode, uint8_t *dst);

et_fp8_status et_fp8_e4m3_decode_v1(const uint8_t *src, size_t count,
                                    float *dst);

et_fp8_status et_fp8_e5m2_encode_scalar_v1(float value,
                                           et_fp8_overflow_mode mode,
                                           uint8_t *out_code);

et_fp8_status et_fp8_e5m2_decode_scalar_v1(uint8_t code, float *out_value);

et_fp8_status et_fp8_e5m2_encode_v1(const float *src, size_t count,
                                    et_fp8_overflow_mode mode, uint8_t *dst);

et_fp8_status et_fp8_e5m2_decode_v1(const uint8_t *src, size_t count,
                                    float *dst);

/* ---- Delayed-Scaling Amax History Manager ----
 * Zero heap allocations; failure-atomic state updates.
 */
et_fp8_status et_fp8_delayed_scale_init(et_fp8_delayed_scale_state_v1 *state,
                                        et_fp8_format format,
                                        uint32_t history_len, uint32_t margin,
                                        float eps, float min_scale,
                                        float max_scale);

et_fp8_status et_fp8_delayed_scale_update(et_fp8_delayed_scale_state_v1 *state,
                                          float observed_amax);

et_fp8_status
et_fp8_delayed_scale_update_tensor(et_fp8_delayed_scale_state_v1 *state,
                                   const float *tensor, size_t count,
                                   float *out_observed_amax);

et_fp8_status
et_fp8_delayed_scale_compute(const et_fp8_delayed_scale_state_v1 *state,
                             float *out_scale, float *out_history_amax);

/* ---- Block-Scaled W8A8 E4M3FN Quantization, Dequantization, GEMM & STE ----
 * DeepSeek-V3 style fine-grained block scaling:
 * - Activation X[M, K]: 1 x block_k scales S_x[M, K / block_k].
 * - Weight W[N, K]: block_n x block_k scales S_w[N / block_n, K / block_k]
 *   where block_k in {16, 32}, block_n in {1, 16, 32}, K % block_k == 0,
 *   and N % block_n == 0.
 * - Dequantization convention: x_deq = decode_e4m3(q) * scale.
 * - Forward GEMM accumulates each block dot-product in binary64, multiplies by
 *   (scale_x * scale_w), sums across K / block_k blocks, adds optional bias[N],
 *   and writes Y[M, N] in binary32.
 * - STE backward computes exact gradients of Y = dequant(X_q, S_x) @
 *   dequant(W_q, S_w)^T + bias with respect to dequantized X, W, and bias.
 */
et_fp8_status et_fp8_block_quantize_e4m3_v1(const float *src, size_t rows,
                                            size_t cols, size_t block_rows,
                                            size_t block_cols, uint8_t *dst_q,
                                            float *dst_scales);

et_fp8_status et_fp8_block_dequantize_e4m3_v1(const uint8_t *src_q,
                                              const float *src_scales,
                                              size_t rows, size_t cols,
                                              size_t block_rows,
                                              size_t block_cols, float *dst);

et_fp8_status et_fp8_per_tensor_quantize_e4m3_v1(const float *src, size_t count,
                                                 uint8_t *dst_q,
                                                 float *out_scale);

et_fp8_status et_fp8_block_gemm_e4m3_v1(size_t m, size_t n, size_t k,
                                        size_t block_k, size_t block_n,
                                        const uint8_t *x_q,
                                        const float *x_scales,
                                        const uint8_t *w_q,
                                        const float *w_scales,
                                        const float *bias, float *y);

et_fp8_status et_fp8_block_gemm_e4m3_backward_v1(
    size_t m, size_t n, size_t k, size_t block_k, size_t block_n,
    const float *dy, const uint8_t *x_q, const float *x_scales,
    const uint8_t *w_q, const float *w_scales, float *dx, float *dw,
    float *dbias);

#ifdef __cplusplus
}
#endif

#endif
