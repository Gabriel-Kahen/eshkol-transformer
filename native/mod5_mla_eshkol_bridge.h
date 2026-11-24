#ifndef ESHKOL_TRANSFORMER_MOD5_MLA_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MOD5_MLA_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT transport for MOD5 Multi-Head Latent Attention (MLA).
 * Status is zero on success or an et_kernel_error_category on failure.
 * All tensor/struct pointers are accepted as void * so misaligned raw pointers
 * from Eshkol FFI are validated cleanly without misaligned-pointer UB.
 */
int32_t et_mod5_mla_aot_open(et_kernel_runtime **runtime);
void et_mod5_mla_aot_close(et_kernel_runtime *runtime);

int64_t et_mod5_mla_aot_dims_struct_bytes(void);
int64_t et_mod5_mla_aot_cache_struct_bytes(void);

int32_t et_mod5_mla_aot_set_dims(void *dims_buf, int64_t batch, int64_t tokens,
                                 int64_t model_dim, int64_t heads,
                                 int64_t head_dim, int64_t latent_dim);

int32_t et_mod5_mla_aot_forward(const et_kernel_runtime *runtime,
                                const void *dims_buf, void *x, void *w_q,
                                void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                                void *out);

int32_t et_mod5_mla_aot_backward(const et_kernel_runtime *runtime,
                                 const void *dims_buf, void *x, void *w_q,
                                 void *w_dkv, void *w_uk, void *w_uv, void *w_o,
                                 void *dy, void *dx, void *dw_q, void *dw_dkv,
                                 void *dw_uk, void *dw_uv, void *dw_o);

/* Exact KV-cache byte telemetry and compression ratio. */
int64_t et_mod5_mla_aot_latent_kv_bytes(int64_t batch, int64_t tokens,
                                        int64_t latent_dim);
int64_t et_mod5_mla_aot_full_kv_bytes(int64_t batch, int64_t tokens,
                                      int64_t heads, int64_t head_dim);
double et_mod5_mla_aot_kv_compression_ratio(int64_t heads, int64_t head_dim,
                                            int64_t latent_dim);

/* Compressed latent KV-cache lifecycle and incremental decode. */
int32_t et_mod5_mla_aot_cache_init(void *cache, const void *dims_buf,
                                   void *c_kv_buf, int64_t c_kv_bytes);
int32_t et_mod5_mla_aot_cache_reset(void *cache);
int64_t et_mod5_mla_aot_cache_seq_len(const void *cache);

int32_t et_mod5_mla_aot_cache_prefill(void *cache, int64_t prefill_tokens,
                                      void *x_prefill, void *w_q, void *w_dkv,
                                      void *w_uk, void *w_uv, void *w_o,
                                      void *out_prefill);

int32_t et_mod5_mla_aot_cache_step(void *cache, void *x_step, void *w_q,
                                   void *w_dkv, void *w_uk, void *w_uv,
                                   void *w_o, void *out_step);

/* Scalar marshaling between Eshkol f64 values and owned binary32 buffers.
 * Values outside finite binary32 range become +/-INFINITY and are rejected at
 * dispatch. Caller supplies a valid non-negative index within buffer bounds.
 */
void et_mod5_mla_aot_store_f32(void *buffer, int64_t index, double value);
double et_mod5_mla_aot_load_f32(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
