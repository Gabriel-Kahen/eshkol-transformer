#include "fp8_gemm_eshkol_bridge.h"

#include <stdint.h>
#include <string.h>

static int valid_u32_range(int64_t v) {
  return v >= 0 && v <= (int64_t)UINT32_MAX;
}

static int valid_u8_range(int64_t v) { return v >= 0 && v <= 255; }

static int valid_size_range(int64_t v) {
  return v >= 0 && (uint64_t)v <= (uint64_t)SIZE_MAX;
}

static int bridge_ranges_overlap(const void *a, size_t a_bytes, const void *b,
                                 size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t a0 = (uintptr_t)a;
  const uintptr_t b0 = (uintptr_t)b;
  if (a0 > UINTPTR_MAX - a_bytes || b0 > UINTPTR_MAX - b_bytes)
    return 1;
  return a0 < b0 + b_bytes && b0 < a0 + a_bytes;
}

void et_fp8_aot_store_f32(void *buf, int64_t index, double value) {
  if (buf == NULL || index < 0)
    return;
  const float f = (float)value;
  uint8_t *p = (uint8_t *)buf + (size_t)index * sizeof(float);
  memcpy(p, &f, sizeof(f));
}

void et_fp8_aot_store_f32_bits(void *buf, int64_t index, int64_t bits_u32) {
  if (buf == NULL || index < 0 || !valid_u32_range(bits_u32))
    return;
  const uint32_t u = (uint32_t)bits_u32;
  uint8_t *p = (uint8_t *)buf + (size_t)index * sizeof(uint32_t);
  memcpy(p, &u, sizeof(u));
}

double et_fp8_aot_load_f32(const void *buf, int64_t index) {
  if (buf == NULL || index < 0)
    return 0.0;
  float f = 0.0f;
  const uint8_t *p = (const uint8_t *)buf + (size_t)index * sizeof(float);
  memcpy(&f, p, sizeof(f));
  return (double)f;
}

int64_t et_fp8_aot_load_f32_bits(const void *buf, int64_t index) {
  if (buf == NULL || index < 0)
    return -1;
  uint32_t u = 0u;
  const uint8_t *p = (const uint8_t *)buf + (size_t)index * sizeof(uint32_t);
  memcpy(&u, p, sizeof(u));
  return (int64_t)u;
}

void et_fp8_aot_store_u8(void *buf, int64_t index, int64_t value) {
  if (buf == NULL || index < 0 || !valid_u8_range(value))
    return;
  ((uint8_t *)buf)[(size_t)index] = (uint8_t)value;
}

int64_t et_fp8_aot_load_u8(const void *buf, int64_t index) {
  if (buf == NULL || index < 0)
    return -1;
  return (int64_t)((const uint8_t *)buf)[(size_t)index];
}

int64_t et_fp8_aot_delayed_scale_state_bytes(void) {
  return (int64_t)sizeof(et_fp8_delayed_scale_state_v1);
}

int32_t et_fp8_aot_e4m3_encode_scalar(double value, int64_t mode,
                                      void *out_u8) {
  if (!valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e4m3_encode_scalar_v1((float)value, (et_fp8_overflow_mode)mode,
                                      (uint8_t *)out_u8);
}

int32_t et_fp8_aot_e4m3_encode_scalar_bits(int64_t f32_bits_u32, int64_t mode,
                                           void *out_u8) {
  if (!valid_u32_range(f32_bits_u32) || !valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  const uint32_t u = (uint32_t)f32_bits_u32;
  float v = 0.0f;
  memcpy(&v, &u, sizeof(v));
  return et_fp8_e4m3_encode_scalar_v1(v, (et_fp8_overflow_mode)mode,
                                      (uint8_t *)out_u8);
}

int32_t et_fp8_aot_e4m3_decode_scalar(int64_t code, void *out_f32) {
  if (out_f32 == NULL || !valid_u8_range(code))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  float v = 0.0f;
  const et_fp8_status st = et_fp8_e4m3_decode_scalar_v1((uint8_t)code, &v);
  if (st != ET_FP8_OK)
    return st;
  memcpy(out_f32, &v, sizeof(v));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_e4m3_encode(const void *src_f32, int64_t count, int64_t mode,
                               void *dst_u8) {
  if (!valid_size_range(count) || !valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e4m3_encode_v1((const float *)src_f32, (size_t)count,
                               (et_fp8_overflow_mode)mode, (uint8_t *)dst_u8);
}

int32_t et_fp8_aot_e4m3_decode(const void *src_u8, int64_t count,
                               void *dst_f32) {
  if (!valid_size_range(count))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e4m3_decode_v1((const uint8_t *)src_u8, (size_t)count,
                               (float *)dst_f32);
}

int32_t et_fp8_aot_e5m2_encode_scalar(double value, int64_t mode,
                                      void *out_u8) {
  if (!valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e5m2_encode_scalar_v1((float)value, (et_fp8_overflow_mode)mode,
                                      (uint8_t *)out_u8);
}

int32_t et_fp8_aot_e5m2_encode_scalar_bits(int64_t f32_bits_u32, int64_t mode,
                                           void *out_u8) {
  if (!valid_u32_range(f32_bits_u32) || !valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  const uint32_t u = (uint32_t)f32_bits_u32;
  float v = 0.0f;
  memcpy(&v, &u, sizeof(v));
  return et_fp8_e5m2_encode_scalar_v1(v, (et_fp8_overflow_mode)mode,
                                      (uint8_t *)out_u8);
}

int32_t et_fp8_aot_e5m2_decode_scalar(int64_t code, void *out_f32) {
  if (out_f32 == NULL || !valid_u8_range(code))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  float v = 0.0f;
  const et_fp8_status st = et_fp8_e5m2_decode_scalar_v1((uint8_t)code, &v);
  if (st != ET_FP8_OK)
    return st;
  memcpy(out_f32, &v, sizeof(v));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_e5m2_encode(const void *src_f32, int64_t count, int64_t mode,
                               void *dst_u8) {
  if (!valid_size_range(count) || !valid_u32_range(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e5m2_encode_v1((const float *)src_f32, (size_t)count,
                               (et_fp8_overflow_mode)mode, (uint8_t *)dst_u8);
}

int32_t et_fp8_aot_e5m2_decode(const void *src_u8, int64_t count,
                               void *dst_f32) {
  if (!valid_size_range(count))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_e5m2_decode_v1((const uint8_t *)src_u8, (size_t)count,
                               (float *)dst_f32);
}

int32_t et_fp8_aot_delayed_scale_init(void *state, int64_t format,
                                      int64_t history_len, int64_t margin,
                                      double eps, double min_scale,
                                      double max_scale) {
  if (state == NULL || !valid_u32_range(format) ||
      !valid_u32_range(history_len) || !valid_u32_range(margin))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  et_fp8_delayed_scale_state_v1 local;
  memcpy(&local, state, sizeof(local));
  const et_fp8_status st = et_fp8_delayed_scale_init(
      &local, (et_fp8_format)format, (uint32_t)history_len, (uint32_t)margin,
      (float)eps, (float)min_scale, (float)max_scale);
  if (st != ET_FP8_OK)
    return st;
  memcpy(state, &local, sizeof(local));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_delayed_scale_update(void *state, double observed_amax) {
  if (state == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  et_fp8_delayed_scale_state_v1 local;
  memcpy(&local, state, sizeof(local));
  const et_fp8_status st =
      et_fp8_delayed_scale_update(&local, (float)observed_amax);
  if (st != ET_FP8_OK)
    return st;
  memcpy(state, &local, sizeof(local));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_delayed_scale_update_tensor(void *state,
                                               const void *tensor_f32,
                                               int64_t count,
                                               void *out_amax_f32) {
  if (state == NULL || !valid_size_range(count) || count == 0)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  const size_t n = (size_t)count;
  if (n > SIZE_MAX / sizeof(float))
    return ET_FP8_ERR_SHAPE;
  const size_t tensor_bytes = n * sizeof(float);
  if (bridge_ranges_overlap(state, sizeof(et_fp8_delayed_scale_state_v1),
                            tensor_f32, tensor_bytes))
    return ET_FP8_ERR_ALIAS;
  if (out_amax_f32 != NULL &&
      (bridge_ranges_overlap(state, sizeof(et_fp8_delayed_scale_state_v1),
                             out_amax_f32, sizeof(float)) ||
       bridge_ranges_overlap(tensor_f32, tensor_bytes, out_amax_f32,
                             sizeof(float))))
    return ET_FP8_ERR_ALIAS;

  et_fp8_delayed_scale_state_v1 local;
  memcpy(&local, state, sizeof(local));
  float amax = 0.0f;
  const et_fp8_status st = et_fp8_delayed_scale_update_tensor(
      &local, (const float *)tensor_f32, n, &amax);
  if (st != ET_FP8_OK)
    return st;
  memcpy(state, &local, sizeof(local));
  if (out_amax_f32 != NULL)
    memcpy(out_amax_f32, &amax, sizeof(amax));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_delayed_scale_compute(const void *state, void *out_scale_f32,
                                         void *out_amax_f32) {
  if (state == NULL || out_scale_f32 == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (bridge_ranges_overlap(state, sizeof(et_fp8_delayed_scale_state_v1),
                            out_scale_f32, sizeof(float)))
    return ET_FP8_ERR_ALIAS;
  if (out_amax_f32 != NULL &&
      (bridge_ranges_overlap(state, sizeof(et_fp8_delayed_scale_state_v1),
                             out_amax_f32, sizeof(float)) ||
       bridge_ranges_overlap(out_scale_f32, sizeof(float), out_amax_f32,
                             sizeof(float))))
    return ET_FP8_ERR_ALIAS;

  et_fp8_delayed_scale_state_v1 local;
  memcpy(&local, state, sizeof(local));
  float scale = 0.0f;
  float amax = 0.0f;
  const et_fp8_status st =
      et_fp8_delayed_scale_compute(&local, &scale, &amax);
  if (st != ET_FP8_OK)
    return st;
  memcpy(out_scale_f32, &scale, sizeof(scale));
  if (out_amax_f32 != NULL)
    memcpy(out_amax_f32, &amax, sizeof(amax));
  return ET_FP8_OK;
}

int32_t et_fp8_aot_block_quantize_e4m3(const void *src_f32, int64_t rows,
                                       int64_t cols, int64_t block_rows,
                                       int64_t block_cols, void *dst_q,
                                       void *dst_scales) {
  if (!valid_size_range(rows) || !valid_size_range(cols) ||
      !valid_size_range(block_rows) || !valid_size_range(block_cols))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_block_quantize_e4m3_v1(
      (const float *)src_f32, (size_t)rows, (size_t)cols, (size_t)block_rows,
      (size_t)block_cols, (uint8_t *)dst_q, (float *)dst_scales);
}

int32_t et_fp8_aot_block_dequantize_e4m3(const void *src_q,
                                         const void *src_scales, int64_t rows,
                                         int64_t cols, int64_t block_rows,
                                         int64_t block_cols, void *dst_f32) {
  if (!valid_size_range(rows) || !valid_size_range(cols) ||
      !valid_size_range(block_rows) || !valid_size_range(block_cols))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_block_dequantize_e4m3_v1(
      (const uint8_t *)src_q, (const float *)src_scales, (size_t)rows,
      (size_t)cols, (size_t)block_rows, (size_t)block_cols, (float *)dst_f32);
}

int32_t et_fp8_aot_per_tensor_quantize_e4m3(const void *src_f32, int64_t count,
                                            void *dst_q, void *out_scale_f32) {
  if (!valid_size_range(count))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_per_tensor_quantize_e4m3_v1(
      (const float *)src_f32, (size_t)count, (uint8_t *)dst_q,
      (float *)out_scale_f32);
}

int32_t et_fp8_aot_block_gemm_e4m3(int64_t m, int64_t n, int64_t k,
                                   int64_t block_k, int64_t block_n,
                                   const void *x_q, const void *x_scales,
                                   const void *w_q, const void *w_scales,
                                   const void *bias_f32, void *y_f32) {
  if (!valid_size_range(m) || !valid_size_range(n) || !valid_size_range(k) ||
      !valid_size_range(block_k) || !valid_size_range(block_n))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_block_gemm_e4m3_v1(
      (size_t)m, (size_t)n, (size_t)k, (size_t)block_k, (size_t)block_n,
      (const uint8_t *)x_q, (const float *)x_scales, (const uint8_t *)w_q,
      (const float *)w_scales, (const float *)bias_f32, (float *)y_f32);
}

int32_t et_fp8_aot_block_gemm_e4m3_backward(
    int64_t m, int64_t n, int64_t k, int64_t block_k, int64_t block_n,
    const void *dy_f32, const void *x_q, const void *x_scales, const void *w_q,
    const void *w_scales, void *dx_f32, void *dw_f32, void *dbias_f32) {
  if (!valid_size_range(m) || !valid_size_range(n) || !valid_size_range(k) ||
      !valid_size_range(block_k) || !valid_size_range(block_n))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  return et_fp8_block_gemm_e4m3_backward_v1(
      (size_t)m, (size_t)n, (size_t)k, (size_t)block_k, (size_t)block_n,
      (const float *)dy_f32, (const uint8_t *)x_q, (const float *)x_scales,
      (const uint8_t *)w_q, (const float *)w_scales, (float *)dx_f32,
      (float *)dw_f32, (float *)dbias_f32);
}
