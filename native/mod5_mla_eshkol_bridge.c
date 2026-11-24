#include "mod5_mla_eshkol_bridge.h"
#include "eshkol_transformer/mod5_mla_abi.h"

#include <float.h>
#include <math.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_mla(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_mla_kernel_provider_v1()
             : NULL;
}

int32_t et_mod5_mla_aot_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_mla, NULL, runtime, &error);
}

void et_mod5_mla_aot_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

int64_t et_mod5_mla_aot_dims_struct_bytes(void) {
  return (int64_t)sizeof(et_mod5_mla_dims);
}

int64_t et_mod5_mla_aot_cache_struct_bytes(void) {
  return (int64_t)sizeof(et_mod5_mla_kv_cache);
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

int32_t et_mod5_mla_aot_set_dims(void *dims_buf, int64_t batch, int64_t tokens,
                                 int64_t model_dim, int64_t heads,
                                 int64_t head_dim, int64_t latent_dim) {
  if (!aligned_span(dims_buf, _Alignof(et_mod5_mla_dims),
                    sizeof(et_mod5_mla_dims)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (batch < 0 || tokens < 0 || model_dim < 0 || heads < 0 || head_dim < 0 ||
      latent_dim < 0)
    return ET_KERNEL_ERROR_UNSUPPORTED;
  const et_mod5_mla_dims dims = {(size_t)batch,    (size_t)tokens,
                                 (size_t)model_dim, (size_t)heads,
                                 (size_t)head_dim, (size_t)latent_dim};
  memcpy(dims_buf, &dims, sizeof(dims));
  return 0;
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

static et_kernel_tensor_view_v1 make_f32_view(void *data, size_t rank,
                                              const uint64_t *shape) {
  size_t bytes = sizeof(float);
  for (size_t i = 0; i < rank; ++i)
    bytes *= (size_t)shape[i];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, "f32", "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static int32_t dispatch_mla(const et_kernel_runtime *runtime, int backward,
                            const void *dims_buf, void *x, void *w_q,
                            void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                            void *dy, void *out_or_dx, void *dw_q, void *dw_dkv,
                            void *dw_uk, void *dw_uv, void *dw_o) {
  if (!aligned_span(dims_buf, _Alignof(et_mod5_mla_dims),
                    sizeof(et_mod5_mla_dims)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  et_mod5_mla_dims d;
  memcpy(&d, dims_buf, sizeof(d));
  if (d.batch < ET_MOD5_MLA_MIN_BATCH || d.batch > ET_MOD5_MLA_MAX_BATCH ||
      d.tokens < ET_MOD5_MLA_MIN_TOKENS || d.tokens > ET_MOD5_MLA_MAX_TOKENS ||
      d.model_dim < ET_MOD5_MLA_MIN_MODEL_DIM ||
      d.model_dim > ET_MOD5_MLA_MAX_MODEL_DIM ||
      d.heads < ET_MOD5_MLA_MIN_HEADS || d.heads > ET_MOD5_MLA_MAX_HEADS ||
      d.head_dim < ET_MOD5_MLA_MIN_HEAD_DIM ||
      d.head_dim > ET_MOD5_MLA_MAX_HEAD_DIM ||
      d.latent_dim < ET_MOD5_MLA_MIN_LATENT_DIM ||
      d.latent_dim > ET_MOD5_MLA_MAX_LATENT_DIM ||
      d.latent_dim >= 2u * d.heads * d.head_dim)
    return ET_KERNEL_ERROR_UNSUPPORTED;

  const uint64_t req_shape[] = {(uint64_t)d.batch,      (uint64_t)d.tokens,
                                (uint64_t)d.model_dim,  (uint64_t)d.heads,
                                (uint64_t)d.head_dim,   (uint64_t)d.latent_dim};
  const uint64_t hd = (uint64_t)(d.heads * d.head_dim);
  const uint64_t x_shape[] = {(uint64_t)d.batch, (uint64_t)d.tokens,
                              (uint64_t)d.model_dim};
  const uint64_t wq_shape[] = {hd, (uint64_t)d.model_dim};
  const uint64_t wdkv_shape[] = {(uint64_t)d.latent_dim, (uint64_t)d.model_dim};
  const uint64_t wuk_shape[] = {hd, (uint64_t)d.latent_dim};
  const uint64_t wo_shape[] = {(uint64_t)d.model_dim, hd};

  const et_kernel_tensor_view_v1 in[] = {
      make_f32_view(x, 3u, x_shape),     make_f32_view(w_q, 2u, wq_shape),
      make_f32_view(w_dkv, 2u, wdkv_shape), make_f32_view(w_uk, 2u, wuk_shape),
      make_f32_view(w_uv, 2u, wuk_shape),   make_f32_view(w_o, 2u, wo_shape),
      make_f32_view(dy, 3u, x_shape)};
  et_kernel_tensor_view_v1 out[] = {
      make_f32_view(out_or_dx, 3u, x_shape),
      make_f32_view(dw_q, 2u, wq_shape),
      make_f32_view(dw_dkv, 2u, wdkv_shape),
      make_f32_view(dw_uk, 2u, wuk_shape),
      make_f32_view(dw_uv, 2u, wuk_shape),
      make_f32_view(dw_o, 2u, wo_shape)};

  const size_t ni = backward ? 7u : 6u;
  const size_t no = backward ? 6u : 1u;
  const size_t stride = sizeof(*in);
  const et_kernel_request_v1 request = {
      sizeof(request), backward ? "mla.backward" : "mla.forward",
      "f32",           "cpu",
      6u,              req_shape,
      1u,              {0}};
  const et_kernel_call_v1 call = {
      sizeof(call), "kernel.mla", &request, ni, stride, ni * stride, in,
      no,           stride,       no * stride, out};
  et_kernel_error error;
  return et_kernel_runtime_dispatch(runtime, &call, &error);
}

int32_t et_mod5_mla_aot_forward(const et_kernel_runtime *runtime,
                                const void *dims_buf, void *x, void *w_q,
                                void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                                void *out) {
  return dispatch_mla(runtime, 0, dims_buf, x, w_q, w_dkv, w_uk, w_uv, w_o,
                      NULL, out, NULL, NULL, NULL, NULL, NULL);
}

int32_t et_mod5_mla_aot_backward(const et_kernel_runtime *runtime,
                                 const void *dims_buf, void *x, void *w_q,
                                 void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                                 void *dy, void *dx, void *dw_q, void *dw_dkv,
                                 void *dw_uk, void *dw_uv, void *dw_o) {
  return dispatch_mla(runtime, 1, dims_buf, x, w_q, w_dkv, w_uk, w_uv, w_o, dy,
                      dx, dw_q, dw_dkv, dw_uk, dw_uv, dw_o);
}

int64_t et_mod5_mla_aot_latent_kv_bytes(int64_t batch, int64_t tokens,
                                        int64_t latent_dim) {
  if (batch <= 0 || tokens <= 0 || latent_dim <= 0)
    return 0;
  return (int64_t)et_mod5_mla_latent_kv_bytes((size_t)batch, (size_t)tokens,
                                              (size_t)latent_dim);
}

int64_t et_mod5_mla_aot_full_kv_bytes(int64_t batch, int64_t tokens,
                                      int64_t heads, int64_t head_dim) {
  if (batch <= 0 || tokens <= 0 || heads <= 0 || head_dim <= 0)
    return 0;
  return (int64_t)et_mod5_mla_full_kv_bytes((size_t)batch, (size_t)tokens,
                                            (size_t)heads, (size_t)head_dim);
}

double et_mod5_mla_aot_kv_compression_ratio(int64_t heads, int64_t head_dim,
                                            int64_t latent_dim) {
  if (heads <= 0 || head_dim <= 0 || latent_dim <= 0)
    return 0.0;
  return et_mod5_mla_kv_compression_ratio((size_t)heads, (size_t)head_dim,
                                          (size_t)latent_dim);
}

int32_t et_mod5_mla_aot_cache_init(void *cache, const void *dims_buf,
                                   void *c_kv_buf, int64_t c_kv_bytes) {
  if (c_kv_bytes < 0)
    return ET_KERNEL_ERROR_SHAPE_MISMATCH;
  et_kernel_error error;
  return et_mod5_mla_cache_init(cache, dims_buf, c_kv_buf, (size_t)c_kv_bytes,
                                &error);
}

int32_t et_mod5_mla_aot_cache_reset(void *cache) {
  et_kernel_error error;
  return et_mod5_mla_cache_reset(cache, &error);
}

int64_t et_mod5_mla_aot_cache_seq_len(const void *cache) {
  if (!aligned_span(cache, _Alignof(et_mod5_mla_kv_cache),
                    sizeof(et_mod5_mla_kv_cache)))
    return -1;
  et_mod5_mla_kv_cache state;
  memcpy(&state, cache, sizeof(state));
  if (state.struct_size != sizeof(et_mod5_mla_kv_cache))
    return -1;
  return (int64_t)state.seq_len;
}

int32_t et_mod5_mla_aot_cache_prefill(void *cache, int64_t prefill_tokens,
                                      void *x_prefill, void *w_q, void *w_dkv,
                                      void *w_uk, void *w_uv, void *w_o,
                                      void *out_prefill) {
  if (prefill_tokens < 0)
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  et_kernel_error error;
  return et_mod5_mla_cache_prefill_f32(cache, (size_t)prefill_tokens, x_prefill,
                                       w_q, w_dkv, w_uk, w_uv, w_o, out_prefill,
                                       &error);
}

int32_t et_mod5_mla_aot_cache_step(void *cache, void *x_step, void *w_q,
                                   void *w_dkv, void *w_uk, void *w_uv,
                                   void *w_o, void *out_step) {
  et_kernel_error error;
  return et_mod5_mla_cache_step_f32(cache, x_step, w_q, w_dkv, w_uk, w_uv, w_o,
                                    out_step, &error);
}

void et_mod5_mla_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_mod5_mla_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}
