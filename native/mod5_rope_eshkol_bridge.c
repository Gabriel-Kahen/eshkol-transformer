#include "mod5_rope_eshkol_bridge.h"
#include "eshkol_transformer/mod5_rope_abi.h"

#include <float.h>
#include <math.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_rope(void *context,
                                                 const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_rope_kernel_provider_v1()
             : NULL;
}

int32_t et_mod5_rope_aot_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_rope, NULL, runtime, &error);
}

void et_mod5_rope_aot_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

static float f32_operand(double value) {
  if (!isfinite(value))
    return (float)value;
  if (value > FLT_MAX)
    return INFINITY;
  if (value < -FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t element_bytes, size_t rank,
                                          const uint64_t *shape) {
  size_t bytes = element_bytes;
  for (size_t i = 0; i < rank; ++i)
    bytes *= (size_t)shape[i];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static int32_t dispatch_rope(const et_kernel_runtime *runtime, int backward,
                             int64_t batch, int64_t heads, int64_t tokens,
                             int64_t head_dim, void *in_buf, void *positions,
                             double theta, double scale, void *out_buf) {
  if (batch <= 0 || heads <= 0 || tokens <= 0 || head_dim <= 0)
    return ET_KERNEL_ERROR_UNSUPPORTED;
  if ((uint64_t)batch > ET_MOD5_ROPE_MAX_BATCH ||
      (uint64_t)heads > ET_MOD5_ROPE_MAX_HEADS ||
      (uint64_t)tokens > ET_MOD5_ROPE_MAX_TOKENS ||
      (uint64_t)head_dim > ET_MOD5_ROPE_MAX_HEAD_DIM)
    return ET_KERNEL_ERROR_UNSUPPORTED;

  const uint64_t shape[] = {(uint64_t)batch, (uint64_t)heads, (uint64_t)tokens,
                            (uint64_t)head_dim};
  const uint64_t pos_shape[] = {(uint64_t)batch, (uint64_t)tokens};
  float theta_f32 = f32_operand(theta);
  float scale_f32 = f32_operand(scale);
  const et_kernel_tensor_view_v1 in[] = {
      make_view(in_buf, "f32", sizeof(float), 4u, shape),
      make_view(positions, "i64", sizeof(int64_t), 2u, pos_shape),
      make_view(&theta_f32, "f32", sizeof(float), 0u, NULL),
      make_view(&scale_f32, "f32", sizeof(float), 0u, NULL)};
  et_kernel_tensor_view_v1 out[] = {
      make_view(out_buf, "f32", sizeof(float), 4u, shape)};
  const et_kernel_request_v1 request = {
      sizeof(request), backward ? "rope.backward" : "rope.forward",
      "f32",           "cpu",
      4u,              shape,
      1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.rope", &request, 4u, stride, 4u * stride, in,
      1u,           stride,        stride,   out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

int32_t et_mod5_rope_aot_forward(const et_kernel_runtime *runtime,
                                 int64_t batch, int64_t heads, int64_t tokens,
                                 int64_t head_dim, void *x, void *positions,
                                 double theta, double scale, void *y) {
  return dispatch_rope(runtime, 0, batch, heads, tokens, head_dim, x, positions,
                       theta, scale, y);
}

int32_t et_mod5_rope_aot_backward(const et_kernel_runtime *runtime,
                                  int64_t batch, int64_t heads, int64_t tokens,
                                  int64_t head_dim, void *dy, void *positions,
                                  double theta, double scale, void *dx) {
  return dispatch_rope(runtime, 1, batch, heads, tokens, head_dim, dy,
                       positions, theta, scale, dx);
}

void et_mod5_rope_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_mod5_rope_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}

void et_mod5_rope_aot_store_i64(void *buffer, int64_t index, int64_t value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

int64_t et_mod5_rope_aot_load_i64(const void *buffer, int64_t index) {
  int64_t value = 0;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}
