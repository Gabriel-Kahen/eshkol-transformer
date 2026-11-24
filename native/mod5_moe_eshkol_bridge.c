#include "mod5_moe_eshkol_bridge.h"
#include "eshkol_transformer/mod5_moe_abi.h"

#include <float.h>
#include <math.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_moe(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_moe_kernel_provider_v1()
             : NULL;
}

int32_t et_mod5_moe_aot_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_moe, NULL, runtime, &error);
}

void et_mod5_moe_aot_close(et_kernel_runtime *runtime) {
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

static int dims_bounded(int64_t batch, int64_t tokens, int64_t model_dim,
                        int64_t ffn_dim, int64_t num_experts, int64_t top_k) {
  return batch > 0 && tokens > 0 && model_dim > 0 && ffn_dim > 0 &&
         num_experts > 0 && top_k > 0 &&
         (uint64_t)batch <= ET_MOD5_MOE_MAX_BATCH &&
         (uint64_t)tokens <= ET_MOD5_MOE_MAX_TOKENS &&
         (uint64_t)model_dim <= ET_MOD5_MOE_MAX_MODEL_DIM &&
         (uint64_t)ffn_dim <= ET_MOD5_MOE_MAX_FFN_DIM &&
         (uint64_t)num_experts <= ET_MOD5_MOE_MAX_EXPERTS &&
         (uint64_t)top_k <= ET_MOD5_MOE_MAX_TOP_K;
}

int32_t et_mod5_moe_aot_forward(const et_kernel_runtime *runtime, int64_t batch,
                                int64_t tokens, int64_t model_dim,
                                int64_t ffn_dim, int64_t num_experts,
                                int64_t top_k, void *x, void *w_gate,
                                void *w_gate_exp, void *w_up_exp,
                                void *w_down_exp, double alpha_aux, void *y,
                                void *aux_loss, void *expert_counts,
                                void *active_expert_evals) {
  if (!dims_bounded(batch, tokens, model_dim, ffn_dim, num_experts, top_k))
    return ET_KERNEL_ERROR_UNSUPPORTED;

  const uint64_t shape[] = {(uint64_t)batch,       (uint64_t)tokens,
                            (uint64_t)model_dim,   (uint64_t)ffn_dim,
                            (uint64_t)num_experts, (uint64_t)top_k};
  const uint64_t x_shape[] = {shape[0], shape[1], shape[2]};
  const uint64_t wg_shape[] = {shape[4], shape[2]};
  const uint64_t wgu_shape[] = {shape[4], shape[3], shape[2]};
  const uint64_t wd_shape[] = {shape[4], shape[2], shape[3]};
  const uint64_t cnt_shape[] = {shape[4]};
  float alpha_f32 = f32_operand(alpha_aux);

  const et_kernel_tensor_view_v1 in[] = {
      make_view(x, "f32", sizeof(float), 3u, x_shape),
      make_view(w_gate, "f32", sizeof(float), 2u, wg_shape),
      make_view(w_gate_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(w_up_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(w_down_exp, "f32", sizeof(float), 3u, wd_shape),
      make_view(&alpha_f32, "f32", sizeof(float), 0u, NULL)};
  et_kernel_tensor_view_v1 out[] = {
      make_view(y, "f32", sizeof(float), 3u, x_shape),
      make_view(aux_loss, "f32", sizeof(float), 0u, NULL),
      make_view(expert_counts, "i64", sizeof(int64_t), 1u, cnt_shape),
      make_view(active_expert_evals, "i64", sizeof(int64_t), 0u, NULL)};

  const et_kernel_request_v1 request = {sizeof(request), "moe.forward",
                                        "f32",           "cpu",
                                        6u,              shape,
                                        1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.moe", &request, 6u, stride, 6u * stride, in,
      4u,           stride,       4u * stride, out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

int32_t et_mod5_moe_aot_backward(const et_kernel_runtime *runtime,
                                 int64_t batch, int64_t tokens,
                                 int64_t model_dim, int64_t ffn_dim,
                                 int64_t num_experts, int64_t top_k, void *x,
                                 void *w_gate, void *w_gate_exp, void *w_up_exp,
                                 void *w_down_exp, double alpha_aux, void *dy,
                                 void *dx, void *dw_gate, void *dw_gate_exp,
                                 void *dw_up_exp, void *dw_down_exp) {
  if (!dims_bounded(batch, tokens, model_dim, ffn_dim, num_experts, top_k))
    return ET_KERNEL_ERROR_UNSUPPORTED;

  const uint64_t shape[] = {(uint64_t)batch,       (uint64_t)tokens,
                            (uint64_t)model_dim,   (uint64_t)ffn_dim,
                            (uint64_t)num_experts, (uint64_t)top_k};
  const uint64_t x_shape[] = {shape[0], shape[1], shape[2]};
  const uint64_t wg_shape[] = {shape[4], shape[2]};
  const uint64_t wgu_shape[] = {shape[4], shape[3], shape[2]};
  const uint64_t wd_shape[] = {shape[4], shape[2], shape[3]};
  float alpha_f32 = f32_operand(alpha_aux);

  const et_kernel_tensor_view_v1 in[] = {
      make_view(x, "f32", sizeof(float), 3u, x_shape),
      make_view(w_gate, "f32", sizeof(float), 2u, wg_shape),
      make_view(w_gate_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(w_up_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(w_down_exp, "f32", sizeof(float), 3u, wd_shape),
      make_view(&alpha_f32, "f32", sizeof(float), 0u, NULL),
      make_view(dy, "f32", sizeof(float), 3u, x_shape)};
  et_kernel_tensor_view_v1 out[] = {
      make_view(dx, "f32", sizeof(float), 3u, x_shape),
      make_view(dw_gate, "f32", sizeof(float), 2u, wg_shape),
      make_view(dw_gate_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(dw_up_exp, "f32", sizeof(float), 3u, wgu_shape),
      make_view(dw_down_exp, "f32", sizeof(float), 3u, wd_shape)};

  const et_kernel_request_v1 request = {sizeof(request), "moe.backward",
                                        "f32",           "cpu",
                                        6u,              shape,
                                        1u,              {0}};
  const size_t stride = sizeof(*in);
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.moe", &request, 7u, stride, 7u * stride, in,
      5u,           stride,       5u * stride, out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

void et_mod5_moe_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_mod5_moe_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}

void et_mod5_moe_aot_store_i64(void *buffer, int64_t index, int64_t value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

int64_t et_mod5_moe_aot_load_i64(const void *buffer, int64_t index) {
  int64_t value = 0;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}
