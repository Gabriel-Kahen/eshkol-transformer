#include "mp4_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_mp4(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mp4_kernel_provider_v1()
             : NULL;
}

static int valid_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int valid_buffer_index(const void *buffer, size_t elem_size,
                              int64_t capacity, int64_t index) {
  if (buffer == NULL || capacity <= 0 || index < 0 || index >= capacity)
    return 0;
  if ((uint64_t)capacity > (uint64_t)(SIZE_MAX / elem_size))
    return 0;
  return valid_span(buffer, elem_size, (size_t)capacity * elem_size);
}

static int valid_scaler_ptr(const void *scaler) {
  return valid_span(scaler, _Alignof(et_mp4_aot_scaler),
                    sizeof(et_mp4_aot_scaler));
}

static const et_mp4_aot_scaler *initialized_scaler(const void *scaler) {
  if (!valid_scaler_ptr(scaler))
    return NULL;
  const et_mp4_aot_scaler *s = (const et_mp4_aot_scaler *)scaler;
  if (s->initialized != 1u)
    return NULL;
  return s;
}

static float f32_from_double(double value) {
  if (isnan(value))
    return NAN;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int finite_f32_double(double value, float *out) {
  if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  *out = (float)value;
  return isfinite(*out);
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t elem_bytes, size_t count,
                                          size_t rank, const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                    data,
                                    count * elem_bytes,
                                    dtype,
                                    "cpu",
                                    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    0u,
                                    rank,
                                    shape};
}

static int32_t dispatch_1d(const et_kernel_runtime *runtime,
                           const char *capability, const char *operation,
                           int64_t count, size_t input_count,
                           const et_kernel_tensor_view_v1 *inputs,
                           size_t output_count,
                           et_kernel_tensor_view_v1 *outputs) {
  const uint64_t shape[1] = {(uint64_t)count};
  const et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", 1u, shape, 1u, {0}};
  const size_t stride = sizeof(et_kernel_tensor_view_v1);
  const et_kernel_call_v1 call = {sizeof(call),
                                  capability,
                                  &request,
                                  input_count,
                                  stride,
                                  input_count * stride,
                                  inputs,
                                  output_count,
                                  stride,
                                  output_count * stride,
                                  outputs};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

int32_t et_mp4_aot_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_mp4, NULL, runtime, &error);
}

void et_mp4_aot_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

int64_t et_mp4_aot_element_bytes(int32_t dtype) {
  switch (dtype) {
  case ET_MP4_AOT_DTYPE_F32:
    return 4;
  case ET_MP4_AOT_DTYPE_F16:
  case ET_MP4_AOT_DTYPE_BF16:
    return 2;
  case ET_MP4_AOT_DTYPE_BOOL:
    return 1;
  default:
    return -1;
  }
}

int64_t et_mp4_aot_tensor_bytes(int32_t dtype, int64_t count) {
  const int64_t width = et_mp4_aot_element_bytes(dtype);
  if (width < 0 || count <= 0 || count > INT64_MAX / width)
    return -1;
  return count * width;
}

int64_t et_mp4_aot_scaler_bytes(void) {
  return (int64_t)sizeof(et_mp4_aot_scaler);
}

static int32_t cast_1d(const et_kernel_runtime *runtime, const char *capability,
                       const char *operation, const char *in_dtype,
                       size_t in_width, const char *out_dtype, size_t out_width,
                       int64_t count, const void *src, void *dst) {
  if (count <= 0 || (uint64_t)count > (uint64_t)(SIZE_MAX / 4u))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const uint64_t shape[1] = {(uint64_t)count};
  const et_kernel_tensor_view_v1 inputs[1] = {
      make_view((void *)src, in_dtype, in_width, (size_t)count, 1u, shape)};
  et_kernel_tensor_view_v1 outputs[1] = {
      make_view(dst, out_dtype, out_width, (size_t)count, 1u, shape)};
  return dispatch_1d(runtime, capability, operation, count, 1u, inputs, 1u,
                     outputs);
}

int32_t et_mp4_aot_cast_f32_to_f16(const et_kernel_runtime *runtime,
                                   int64_t count, const void *src_f32,
                                   void *dst_f16) {
  return cast_1d(runtime, "kernel.cast-f16", "cast.f32-to-f16", "f32", 4u,
                 "f16", 2u, count, src_f32, dst_f16);
}

int32_t et_mp4_aot_cast_f16_to_f32(const et_kernel_runtime *runtime,
                                   int64_t count, const void *src_f16,
                                   void *dst_f32) {
  return cast_1d(runtime, "kernel.cast-f16", "cast.f16-to-f32", "f16", 2u,
                 "f32", 4u, count, src_f16, dst_f32);
}

int32_t et_mp4_aot_cast_f32_to_bf16(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_f32,
                                    void *dst_bf16) {
  return cast_1d(runtime, "kernel.cast-bf16", "cast.f32-to-bf16", "f32", 4u,
                 "bf16", 2u, count, src_f32, dst_bf16);
}

int32_t et_mp4_aot_cast_bf16_to_f32(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_bf16,
                                    void *dst_f32) {
  return cast_1d(runtime, "kernel.cast-bf16", "cast.bf16-to-f32", "bf16", 2u,
                 "f32", 4u, count, src_bf16, dst_f32);
}

static int32_t unscale_1d(const et_kernel_runtime *runtime,
                          const char *operation, const char *in_dtype,
                          int64_t count, const void *src_narrow,
                          const void *inv_scale_f32, void *dst_f32,
                          void *overflow_out) {
  if (count <= 0 || (uint64_t)count > (uint64_t)(SIZE_MAX / 4u))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const uint64_t shape[1] = {(uint64_t)count};
  const et_kernel_tensor_view_v1 inputs[2] = {
      make_view((void *)src_narrow, in_dtype, 2u, (size_t)count, 1u, shape),
      make_view((void *)inv_scale_f32, "f32", 4u, 1u, 0u, NULL)};
  et_kernel_tensor_view_v1 outputs[2] = {
      make_view(dst_f32, "f32", 4u, (size_t)count, 1u, shape),
      make_view(overflow_out, "bool", 1u, 1u, 0u, NULL)};
  return dispatch_1d(runtime, "kernel.grad-unscale", operation, count, 2u,
                     inputs, 2u, outputs);
}

int32_t et_mp4_aot_grad_unscale_f16(const et_kernel_runtime *runtime,
                                    int64_t count, const void *src_f16,
                                    const void *inv_scale_f32, void *dst_f32,
                                    void *overflow_out) {
  return unscale_1d(runtime, "grad.unscale-f16-to-f32", "f16", count, src_f16,
                    inv_scale_f32, dst_f32, overflow_out);
}

int32_t et_mp4_aot_grad_unscale_bf16(const et_kernel_runtime *runtime,
                                     int64_t count, const void *src_bf16,
                                     const void *inv_scale_f32, void *dst_f32,
                                     void *overflow_out) {
  return unscale_1d(runtime, "grad.unscale-bf16-to-f32", "bf16", count,
                    src_bf16, inv_scale_f32, dst_f32, overflow_out);
}

int32_t et_mp4_aot_scaler_init(void *scaler, double init_scale,
                               double growth_factor, double backoff_factor,
                               int64_t growth_interval, double min_scale,
                               double max_scale) {
  if (!valid_scaler_ptr(scaler) || growth_interval <= 0 ||
      growth_interval > (int64_t)UINT32_MAX)
    return (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;

  float f_scale = 0.0f;
  float f_growth = 0.0f;
  float f_backoff = 0.0f;
  float f_min = 0.0f;
  float f_max = 0.0f;
  if (!finite_f32_double(init_scale, &f_scale) ||
      !finite_f32_double(growth_factor, &f_growth) ||
      !finite_f32_double(backoff_factor, &f_backoff) ||
      !finite_f32_double(min_scale, &f_min) ||
      !finite_f32_double(max_scale, &f_max))
    return (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;

  et_mp4_loss_scaler_v1 next_state;
  const et_mp4_loss_scaler_status status = et_mp4_loss_scaler_init(
      &next_state, f_scale, f_growth, f_backoff, (uint32_t)growth_interval,
      f_min, f_max);
  if (status != ET_MP4_LOSS_SCALER_STEP_ACCEPTED)
    return (int32_t)status;

  et_mp4_aot_scaler *s = (et_mp4_aot_scaler *)scaler;
  s->state = next_state;
  s->initialized = 1u;
  s->overflow_count = 0u;
  return (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED;
}

double et_mp4_aot_scaler_scale(const void *scaler, double value) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL || !isfinite(value))
    return NAN;
  return value * (double)s->state.scale;
}

int32_t et_mp4_aot_scaler_scale_f32(const void *scaler, int64_t count,
                                    const void *src_f32, void *dst_f32) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL || count <= 0 ||
      (uint64_t)count > (uint64_t)(SIZE_MAX / sizeof(float)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const size_t n = (size_t)count;
  const size_t bytes = n * sizeof(float);
  if (!valid_span(src_f32, sizeof(float), bytes) ||
      !valid_span(dst_f32, sizeof(float), bytes) ||
      spans_overlap(src_f32, bytes, dst_f32, bytes) ||
      spans_overlap(scaler, sizeof(et_mp4_aot_scaler), dst_f32, bytes))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  const unsigned char *src = (const unsigned char *)src_f32;
  unsigned char *dst = (unsigned char *)dst_f32;
  const float scale = s->state.scale;
  if (!isfinite(scale) || scale <= 0.0f)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  for (size_t i = 0; i < n; ++i) {
    float v;
    memcpy(&v, src + i * sizeof(float), sizeof(float));
    const float prod = v * scale;
    if (!isfinite(v) || !isfinite(prod))
      return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  }
  for (size_t i = 0; i < n; ++i) {
    float v;
    memcpy(&v, src + i * sizeof(float), sizeof(float));
    const float prod = v * scale;
    memcpy(dst + i * sizeof(float), &prod, sizeof(float));
  }
  return 0;
}

static int32_t scaler_unscale(const et_kernel_runtime *runtime,
                              const void *scaler, const char *operation,
                              const char *in_dtype, int64_t count,
                              const void *src_narrow, void *dst_f32,
                              void *overflow_out) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL || count <= 0 || (uint64_t)count > (uint64_t)(SIZE_MAX / 4u))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (!isfinite(s->state.scale) || s->state.scale <= 0.0f)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const size_t dst_bytes = (size_t)count * sizeof(float);
  if (dst_f32 != NULL &&
      spans_overlap(scaler, sizeof(et_mp4_aot_scaler), dst_f32, dst_bytes))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (overflow_out != NULL &&
      spans_overlap(scaler, sizeof(et_mp4_aot_scaler), overflow_out, 1u))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const float inv_scale = 1.0f / s->state.scale;
  return unscale_1d(runtime, operation, in_dtype, count, src_narrow, &inv_scale,
                    dst_f32, overflow_out);
}

int32_t et_mp4_aot_scaler_unscale_f16(const et_kernel_runtime *runtime,
                                      const void *scaler, int64_t count,
                                      const void *src_f16, void *dst_f32,
                                      void *overflow_out) {
  return scaler_unscale(runtime, scaler, "grad.unscale-f16-to-f32", "f16",
                        count, src_f16, dst_f32, overflow_out);
}

int32_t et_mp4_aot_scaler_unscale_bf16(const et_kernel_runtime *runtime,
                                       const void *scaler, int64_t count,
                                       const void *src_bf16, void *dst_f32,
                                       void *overflow_out) {
  return scaler_unscale(runtime, scaler, "grad.unscale-bf16-to-f32", "bf16",
                        count, src_bf16, dst_f32, overflow_out);
}

int32_t et_mp4_aot_scaler_step(void *scaler, int32_t overflow_detected) {
  if (!valid_scaler_ptr(scaler))
    return (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;
  et_mp4_aot_scaler *s = (et_mp4_aot_scaler *)scaler;
  if (s->initialized != 1u ||
      (overflow_detected != 0 && overflow_detected != 1))
    return (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;
  const et_mp4_loss_scaler_status status =
      et_mp4_loss_scaler_update(&s->state, (uint8_t)overflow_detected);
  if (status == ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF)
    ++s->overflow_count;
  return (int32_t)status;
}

double et_mp4_aot_scaler_get_scale(const void *scaler) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL)
    return NAN;
  return (double)s->state.scale;
}

int64_t et_mp4_aot_scaler_get_growth_tracker(const void *scaler) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL)
    return -1;
  return (int64_t)s->state.clean_steps;
}

int64_t et_mp4_aot_scaler_get_overflow_count(const void *scaler) {
  const et_mp4_aot_scaler *s = initialized_scaler(scaler);
  if (s == NULL)
    return -1;
  return (int64_t)s->overflow_count;
}

int32_t et_mp4_aot_store_f32(void *buffer, int64_t capacity, int64_t index,
                             double value) {
  if (!valid_buffer_index(buffer, sizeof(float), capacity, index))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const float f = f32_from_double(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(float), &f,
         sizeof(float));
  return 0;
}

double et_mp4_aot_load_f32(const void *buffer, int64_t capacity,
                           int64_t index) {
  if (!valid_buffer_index(buffer, sizeof(float), capacity, index))
    return NAN;
  float f;
  memcpy(&f, (const unsigned char *)buffer + (size_t)index * sizeof(float),
         sizeof(float));
  return (double)f;
}

int32_t et_mp4_aot_store_f32_bits(void *buffer, int64_t capacity, int64_t index,
                                  int64_t bits) {
  if (!valid_buffer_index(buffer, sizeof(uint32_t), capacity, index) ||
      bits < 0 || bits > (int64_t)UINT32_MAX)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const uint32_t u = (uint32_t)bits;
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(uint32_t), &u,
         sizeof(uint32_t));
  return 0;
}

int64_t et_mp4_aot_load_f32_bits(const void *buffer, int64_t capacity,
                                 int64_t index) {
  if (!valid_buffer_index(buffer, sizeof(uint32_t), capacity, index))
    return -1;
  uint32_t u;
  memcpy(&u, (const unsigned char *)buffer + (size_t)index * sizeof(uint32_t),
         sizeof(uint32_t));
  return (int64_t)u;
}

int32_t et_mp4_aot_store_u16(void *buffer, int64_t capacity, int64_t index,
                             int64_t bits) {
  if (!valid_buffer_index(buffer, sizeof(uint16_t), capacity, index) ||
      bits < 0 || bits > (int64_t)UINT16_MAX)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const uint16_t u = (uint16_t)bits;
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(uint16_t), &u,
         sizeof(uint16_t));
  return 0;
}

int64_t et_mp4_aot_load_u16(const void *buffer, int64_t capacity,
                            int64_t index) {
  if (!valid_buffer_index(buffer, sizeof(uint16_t), capacity, index))
    return -1;
  uint16_t u;
  memcpy(&u, (const unsigned char *)buffer + (size_t)index * sizeof(uint16_t),
         sizeof(uint16_t));
  return (int64_t)u;
}

int32_t et_mp4_aot_store_u8(void *buffer, int64_t capacity, int64_t index,
                            int64_t value) {
  if (!valid_buffer_index(buffer, sizeof(uint8_t), capacity, index) ||
      value < 0 || value > (int64_t)UINT8_MAX)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const uint8_t u = (uint8_t)value;
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(uint8_t), &u,
         sizeof(uint8_t));
  return 0;
}

int64_t et_mp4_aot_load_u8(const void *buffer, int64_t capacity,
                           int64_t index) {
  if (!valid_buffer_index(buffer, sizeof(uint8_t), capacity, index))
    return -1;
  uint8_t u;
  memcpy(&u, (const unsigned char *)buffer + (size_t)index * sizeof(uint8_t),
         sizeof(uint8_t));
  return (int64_t)u;
}
