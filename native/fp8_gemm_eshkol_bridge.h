#ifndef ESHKOL_TRANSFORMER_FP8_GEMM_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_FP8_GEMM_ESHKOL_BRIDGE_H

#include "eshkol_transformer/fp8_gemm_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zero-allocation Eshkol AOT transport for the OCP FP8 (E4M3FN & E5M2) codec,
 * delayed-scaling amax history manager, and DeepSeek-V3 block-scaled W8A8 GEMM.
 *
 * All buffer pointers are accepted as void * / const void * and accessed via
 * byte-wise memcpy so raw Eshkol FFI pointers never incur alignment UB.
 */

/* Buffer marshaling helpers (alignment-safe). */
void et_fp8_aot_store_f32(void *buf, int64_t index, double value);
void et_fp8_aot_store_f32_bits(void *buf, int64_t index, int64_t bits_u32);
double et_fp8_aot_load_f32(const void *buf, int64_t index);
int64_t et_fp8_aot_load_f32_bits(const void *buf, int64_t index);
void et_fp8_aot_store_u8(void *buf, int64_t index, int64_t value);
int64_t et_fp8_aot_load_u8(const void *buf, int64_t index);
int64_t et_fp8_aot_delayed_scale_state_bytes(void);

/* Scalar & bulk OCP FP8 codecs. */
int32_t et_fp8_aot_e4m3_encode_scalar(double value, int64_t mode,
                                      void *out_u8);
int32_t et_fp8_aot_e4m3_encode_scalar_bits(int64_t f32_bits_u32, int64_t mode,
                                           void *out_u8);
int32_t et_fp8_aot_e4m3_decode_scalar(int64_t code, void *out_f32);
int32_t et_fp8_aot_e4m3_encode(const void *src_f32, int64_t count, int64_t mode,
                               void *dst_u8);
int32_t et_fp8_aot_e4m3_decode(const void *src_u8, int64_t count,
                               void *dst_f32);

int32_t et_fp8_aot_e5m2_encode_scalar(double value, int64_t mode,
                                      void *out_u8);
int32_t et_fp8_aot_e5m2_encode_scalar_bits(int64_t f32_bits_u32, int64_t mode,
                                           void *out_u8);
int32_t et_fp8_aot_e5m2_decode_scalar(int64_t code, void *out_f32);
int32_t et_fp8_aot_e5m2_encode(const void *src_f32, int64_t count, int64_t mode,
                               void *dst_u8);
int32_t et_fp8_aot_e5m2_decode(const void *src_u8, int64_t count,
                               void *dst_f32);

/* Delayed-scaling amax history manager. */
int32_t et_fp8_aot_delayed_scale_init(void *state, int64_t format,
                                      int64_t history_len, int64_t margin,
                                      double eps, double min_scale,
                                      double max_scale);
int32_t et_fp8_aot_delayed_scale_update(void *state, double observed_amax);
int32_t et_fp8_aot_delayed_scale_update_tensor(void *state,
                                               const void *tensor_f32,
                                               int64_t count,
                                               void *out_amax_f32);
int32_t et_fp8_aot_delayed_scale_compute(const void *state, void *out_scale_f32,
                                         void *out_amax_f32);

/* Block-scaled W8A8 E4M3FN quantize, dequantize, per-tensor baseline, GEMM. */
int32_t et_fp8_aot_block_quantize_e4m3(const void *src_f32, int64_t rows,
                                       int64_t cols, int64_t block_rows,
                                       int64_t block_cols, void *dst_q,
                                       void *dst_scales);
int32_t et_fp8_aot_block_dequantize_e4m3(const void *src_q,
                                         const void *src_scales, int64_t rows,
                                         int64_t cols, int64_t block_rows,
                                         int64_t block_cols, void *dst_f32);
int32_t et_fp8_aot_per_tensor_quantize_e4m3(const void *src_f32, int64_t count,
                                            void *dst_q, void *out_scale_f32);
int32_t et_fp8_aot_block_gemm_e4m3(int64_t m, int64_t n, int64_t k,
                                   int64_t block_k, int64_t block_n,
                                   const void *x_q, const void *x_scales,
                                   const void *w_q, const void *w_scales,
                                   const void *bias_f32, void *y_f32);
int32_t et_fp8_aot_block_gemm_e4m3_backward(
    int64_t m, int64_t n, int64_t k, int64_t block_k, int64_t block_n,
    const void *dy_f32, const void *x_q, const void *x_scales, const void *w_q,
    const void *w_scales, void *dx_f32, void *dw_f32, void *dbias_f32);

#ifdef __cplusplus
}
#endif

#endif
