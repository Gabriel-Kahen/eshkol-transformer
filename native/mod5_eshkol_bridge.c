#include "mod5_eshkol_bridge.h"
#include "eshkol_transformer/mod5_primitives_abi.h"

#include <float.h>
#include <math.h>
#include <string.h>

enum { TOKENS = 8, WIDTH = 16, HIDDEN = 32, HEADS = 4, HEAD_WIDTH = 4 };

static const et_kernel_provider_v1 *resolve_mod5(void *context,
                                                 const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_kernel_provider_v1()
             : NULL;
}

int32_t et_mod5_aot_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_mod5, NULL, runtime, &error);
}

void et_mod5_aot_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

static et_kernel_tensor_view_v1 view(void *data, const char *dtype,
                                     size_t element_bytes, size_t rank,
                                     const uint64_t *shape) {
  size_t bytes = element_bytes;
  for (size_t i = 0; i < rank; ++i)
    bytes *= shape[i];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static et_kernel_tensor_view_v1 f32_view(float *data, size_t rank,
                                         const uint64_t *shape) {
  return view(data, "f32", sizeof(*data), rank, shape);
}

static float f32_operand(double value) {
  if (value > FLT_MAX)
    return INFINITY;
  if (value < -FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int32_t dispatch(const et_kernel_runtime *runtime, const char *cap,
                        const char *operation, size_t rank,
                        const uint64_t *shape, size_t ni,
                        const et_kernel_tensor_view_v1 *in, size_t no,
                        et_kernel_tensor_view_v1 *out) {
  const et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {sizeof(call), cap,         &request, ni,
                                  stride,       ni * stride, in,       no,
                                  stride,       no * stride, out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

static int32_t rms(const et_kernel_runtime *runtime, int backward, float *x,
                   float *gamma, float *epsilon, float *dy, float *out,
                   float *dgamma) {
  const uint64_t shape[] = {1, TOKENS, WIDTH};
  const et_kernel_tensor_view_v1 in[] = {
      f32_view(x, 3u, shape), f32_view(gamma, 1u, shape + 2u),
      f32_view(epsilon, 0u, NULL), f32_view(dy, 3u, shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 3u, shape),
                                        f32_view(dgamma, 1u, shape + 2u)};
  return dispatch(runtime, "kernel.rmsnorm",
                  backward ? "rmsnorm.backward" : "rmsnorm.forward", 3u, shape,
                  backward ? 4u : 3u, in, backward ? 2u : 1u, outputs);
}

int32_t et_mod5_aot_rms_forward(const et_kernel_runtime *runtime, float *x,
                                float *gamma, float *epsilon, float *y) {
  return rms(runtime, 0, x, gamma, epsilon, NULL, y, NULL);
}

int32_t et_mod5_aot_rms_backward(const et_kernel_runtime *runtime, float *x,
                                 float *gamma, float *epsilon, float *dy,
                                 float *dx, float *dgamma) {
  return rms(runtime, 1, x, gamma, epsilon, dy, dx, dgamma);
}

static int32_t swiglu(const et_kernel_runtime *runtime, int backward,
                      float *gate, float *up, float *dy, float *out,
                      float *dup) {
  const uint64_t shape[] = {1, TOKENS, HIDDEN};
  const et_kernel_tensor_view_v1 in[] = {f32_view(gate, 3u, shape),
                                         f32_view(up, 3u, shape),
                                         f32_view(dy, 3u, shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 3u, shape),
                                        f32_view(dup, 3u, shape)};
  return dispatch(runtime, "kernel.swiglu",
                  backward ? "swiglu.backward" : "swiglu.forward", 3u, shape,
                  backward ? 3u : 2u, in, backward ? 2u : 1u, outputs);
}

int32_t et_mod5_aot_swiglu_forward(const et_kernel_runtime *runtime,
                                   float *gate, float *up, float *y) {
  return swiglu(runtime, 0, gate, up, NULL, y, NULL);
}

int32_t et_mod5_aot_swiglu_backward(const et_kernel_runtime *runtime,
                                    float *gate, float *up, float *dy,
                                    float *dgate, float *dup) {
  return swiglu(runtime, 1, gate, up, dy, dgate, dup);
}

static int32_t attention(const et_kernel_runtime *runtime, int backward,
                         int64_t kv_heads, float *q, float *k, float *v,
                         int64_t *qpos, int64_t *kpos, uint8_t *mask,
                         float *slopes, int64_t window, float *dy, float *out,
                         float *dk, float *dv) {
  if (kv_heads != 1 && kv_heads != 2)
    return ET_KERNEL_ERROR_UNSUPPORTED;
  const uint64_t shape[] = {1,      HEADS,  (uint64_t)kv_heads,
                            TOKENS, TOKENS, HEAD_WIDTH};
  const uint64_t q_shape[] = {1, HEADS, TOKENS, HEAD_WIDTH};
  const uint64_t kv_shape[] = {1, (uint64_t)kv_heads, TOKENS, HEAD_WIDTH};
  const uint64_t pos_shape[] = {1, TOKENS};
  const uint64_t mask_shape[] = {1, TOKENS, TOKENS};
  const uint64_t slope_shape[] = {HEADS};
  const et_kernel_tensor_view_v1 in[] = {
      f32_view(q, 4u, q_shape),
      f32_view(k, 4u, kv_shape),
      f32_view(v, 4u, kv_shape),
      view(qpos, "i64", sizeof(*qpos), 2u, pos_shape),
      view(kpos, "i64", sizeof(*kpos), 2u, pos_shape),
      view(mask, "bool", sizeof(*mask), 3u, mask_shape),
      f32_view(slopes, 1u, slope_shape),
      view(&window, "i64", sizeof(window), 0u, NULL),
      f32_view(dy, 4u, q_shape)};
  et_kernel_tensor_view_v1 outputs[] = {f32_view(out, 4u, q_shape),
                                        f32_view(dk, 4u, kv_shape),
                                        f32_view(dv, 4u, kv_shape)};
  return dispatch(
      runtime, "kernel.variant-attention",
      backward ? "variant-attention.backward" : "variant-attention.forward", 6u,
      shape, backward ? 9u : 8u, in, backward ? 3u : 1u, outputs);
}

int32_t et_mod5_aot_attention_forward(const et_kernel_runtime *runtime,
                                      int64_t kv_heads, float *q, float *k,
                                      float *v, int64_t *qpos, int64_t *kpos,
                                      uint8_t *mask, float *slopes,
                                      int64_t window, float *y) {
  return attention(runtime, 0, kv_heads, q, k, v, qpos, kpos, mask, slopes,
                   window, NULL, y, NULL, NULL);
}

int32_t et_mod5_aot_attention_backward(const et_kernel_runtime *runtime,
                                       int64_t kv_heads, float *q, float *k,
                                       float *v, int64_t *qpos, int64_t *kpos,
                                       uint8_t *mask, float *slopes,
                                       int64_t window, float *dy, float *dq,
                                       float *dk, float *dv) {
  return attention(runtime, 1, kv_heads, q, k, v, qpos, kpos, mask, slopes,
                   window, dy, dq, dk, dv);
}

void et_mod5_aot_store_f32(float *buffer, int64_t index, double value) {
  buffer[index] = f32_operand(value);
}

double et_mod5_aot_load_f32(const float *buffer, int64_t index) {
  return buffer[index];
}
