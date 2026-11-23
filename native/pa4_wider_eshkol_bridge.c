#include "pa4_wider_eshkol_bridge.h"
#include "eshkol_transformer/pa4_wider_attention_abi.h"

#include <float.h>
#include <math.h>
#include <string.h>

_Static_assert(sizeof(float) == sizeof(uint32_t) && FLT_RADIX == 2 &&
                   FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "PA4-Wider AOT requires IEEE binary32");

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  if (strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0)
    return NULL;
  return *(const int32_t *)context == 0 ? et_pa4_wider_ref_provider_v1()
                                        : et_pa4_wider_online_provider_v1();
}

int32_t et_pa4_wider_aot_open(int32_t variant, et_kernel_runtime **runtime) {
  if (runtime == NULL ||
      (uintptr_t)runtime % _Alignof(et_kernel_runtime *) != 0 ||
      (variant != 0 && variant != 1))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  et_kernel_runtime *discovered = NULL;
  et_kernel_error error;
  const int32_t status = et_kernel_runtime_discover(resolve_provider, &variant,
                                                    &discovered, &error);
  if (status == 0)
    *runtime = discovered;
  return status;
}

void et_pa4_wider_aot_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

static et_kernel_tensor_view_v1 view(void *data, const char *dtype,
                                     size_t bytes, size_t rank,
                                     const uint64_t *shape) {
  for (size_t i = 0; i < rank; ++i)
    bytes *= shape[i];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static et_kernel_tensor_view_v1 f32_view(float *data, const uint64_t *shape) {
  return view(data, "f32", sizeof(*data), 4u, shape);
}

static int32_t dispatch(const et_kernel_runtime *runtime, int backward,
                        const uint64_t shape[6], float *q, float *k, float *v,
                        int64_t *qpos, int64_t *kpos, uint8_t *mask,
                        float *dout, float *out, float *dk, float *dv) {
  if (runtime == NULL)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;

  const uint64_t n = shape[0], hq = shape[1], hkv = shape[2];
  const uint64_t tq = shape[3], tk = shape[4], dh = shape[5];
  /* Bound byte products before creating views. K1 checks the head pairs. */
  if (n < 1u || n > 2u || hq < 1u || hq > 4u || hkv < 1u || hkv > 4u ||
      tq < 1u || tq > 64u || tk < 1u || tk > ET_PA4_WIDER_MAX_KEYS || dh < 1u ||
      dh > 16u)
    return ET_KERNEL_ERROR_UNSUPPORTED;

  const uint64_t q_shape[] = {n, hq, tq, dh};
  const uint64_t kv_shape[] = {n, hkv, tk, dh};
  const uint64_t qp_shape[] = {n, tq};
  const uint64_t kp_shape[] = {n, tk};
  const uint64_t mask_shape[] = {n, tq, tk};
  const et_kernel_tensor_view_v1 inputs[] = {
      f32_view(q, q_shape),
      f32_view(k, kv_shape),
      f32_view(v, kv_shape),
      view(qpos, "i64", sizeof(*qpos), 2u, qp_shape),
      view(kpos, "i64", sizeof(*kpos), 2u, kp_shape),
      view(mask, "bool", sizeof(*mask), 3u, mask_shape),
      f32_view(dout, q_shape)};
  et_kernel_tensor_view_v1 outputs[] = {
      f32_view(out, q_shape), f32_view(dk, kv_shape), f32_view(dv, kv_shape)};
  const et_kernel_request_v1 request = {sizeof(request),
                                        backward ? "causal-attention.backward"
                                                 : "causal-attention.forward",
                                        "f32",
                                        "cpu",
                                        6u,
                                        shape,
                                        1u,
                                        {0}};
  const size_t ni = backward ? 7u : 6u;
  const size_t no = backward ? 3u : 1u;
  const et_kernel_call_v1 call = {
      sizeof(call),       "kernel.causal-attention", &request, ni,
      sizeof(inputs[0]),  ni * sizeof(inputs[0]),    inputs,   no,
      sizeof(outputs[0]), no * sizeof(outputs[0]),   outputs};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

int32_t et_pa4_wider_aot_forward(const et_kernel_runtime *runtime, int64_t n,
                                 int64_t hq, int64_t hkv, int64_t tq,
                                 int64_t tk, int64_t dh, float *q, float *k,
                                 float *v, int64_t *qpos, int64_t *kpos,
                                 uint8_t *mask, float *out) {
  const uint64_t shape[] = {(uint64_t)n,  (uint64_t)hq, (uint64_t)hkv,
                            (uint64_t)tq, (uint64_t)tk, (uint64_t)dh};
  return dispatch(runtime, 0, shape, q, k, v, qpos, kpos, mask, NULL, out, NULL,
                  NULL);
}

int32_t et_pa4_wider_aot_backward(const et_kernel_runtime *runtime, int64_t n,
                                  int64_t hq, int64_t hkv, int64_t tq,
                                  int64_t tk, int64_t dh, float *q, float *k,
                                  float *v, int64_t *qpos, int64_t *kpos,
                                  uint8_t *mask, float *dout, float *dq,
                                  float *dk, float *dv) {
  const uint64_t shape[] = {(uint64_t)n,  (uint64_t)hq, (uint64_t)hkv,
                            (uint64_t)tq, (uint64_t)tk, (uint64_t)dh};
  return dispatch(runtime, 1, shape, q, k, v, qpos, kpos, mask, dout, dq, dk,
                  dv);
}

void et_pa4_wider_aot_store_f32(float *buffer, int64_t index, double value) {
  if (value > FLT_MAX)
    buffer[index] = INFINITY;
  else if (value < -FLT_MAX)
    buffer[index] = -INFINITY;
  else
    buffer[index] = (float)value;
}

double et_pa4_wider_aot_load_f32(const float *buffer, int64_t index) {
  return buffer[index];
}

int64_t et_pa4_wider_aot_f32_bits(const float *buffer, int64_t index) {
  uint32_t bits;
  memcpy(&bits, &buffer[index], sizeof(bits));
  return bits;
}

void et_pa4_wider_aot_store_i64(int64_t *buffer, int64_t index, int64_t value) {
  buffer[index] = value;
}

void et_pa4_wider_aot_store_u8(uint8_t *buffer, int64_t index, int64_t value) {
  buffer[index] = (uint8_t)value;
}
