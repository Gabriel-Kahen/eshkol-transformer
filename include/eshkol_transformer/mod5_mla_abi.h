#ifndef ESHKOL_TRANSFORMER_MOD5_MLA_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_MLA_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_MLA_ABI_MAJOR 1u
#define ET_MOD5_MLA_ABI_MINOR 0u

#define ET_MOD5_MLA_MIN_BATCH 1u
#define ET_MOD5_MLA_MAX_BATCH 2u
#define ET_MOD5_MLA_MIN_TOKENS 1u
#define ET_MOD5_MLA_MAX_TOKENS 32u
#define ET_MOD5_MLA_MIN_MODEL_DIM 4u
#define ET_MOD5_MLA_MAX_MODEL_DIM 32u
#define ET_MOD5_MLA_MIN_HEADS 1u
#define ET_MOD5_MLA_MAX_HEADS 4u
#define ET_MOD5_MLA_MIN_HEAD_DIM 2u
#define ET_MOD5_MLA_MAX_HEAD_DIM 16u
#define ET_MOD5_MLA_MIN_LATENT_DIM 1u
#define ET_MOD5_MLA_MAX_LATENT_DIM 16u

/*
 * Shape descriptor for Multi-Head Latent Attention (MLA):
 *   batch      (N):  1..2
 *   tokens     (T):  1..32  (sequence length, or max_tokens for KV-cache)
 *   model_dim  (D):  4..32
 *   heads      (H):  1..4
 *   head_dim   (Dh): 2..16
 *   latent_dim (dc): 1..16  with strict compression dc < 2 * H * Dh
 */
typedef struct et_mod5_mla_dims {
  size_t batch;
  size_t tokens;
  size_t model_dim;
  size_t heads;
  size_t head_dim;
  size_t latent_dim;
} et_mod5_mla_dims;

/*
 * Compressed Latent KV-Cache.
 * Stores ONLY C_kv [N, max_tokens, dc] in caller-owned dense row-major f32
 * storage rather than full K, V [N, max_tokens, 2 * H * Dh], yielding exact
 * KV-cache memory compression ratio (2 * H * Dh) / dc.
 */
typedef struct et_mod5_mla_kv_cache {
  size_t struct_size;
  size_t batch;
  size_t max_tokens;
  size_t model_dim;
  size_t heads;
  size_t head_dim;
  size_t latent_dim;
  size_t seq_len;
  size_t c_kv_bytes;
  void *c_kv;
} et_mod5_mla_kv_cache;

/*
 * Carrier-neutral K1 ABI 1.0 provider for Multi-Head Latent Attention (MLA).
 * Capability: "kernel.mla"
 * Operations: "mla.forward", "mla.backward"
 * Request shape: [N, T, D, H, Dh, dc] with N in 1..2, T in 1..32, D in 4..32,
 * H in 1..4, Dh in 2..16, dc in 1..16, and dc < 2 * H * Dh.
 *
 * Let HD = H * Dh.
 * K1 input views for "mla.forward" (all dense row-major CPU f32):
 *   0: X     f32[N, T, D]
 *   1: W_q   f32[HD, D]
 *   2: W_dkv f32[dc, D]
 *   3: W_uk  f32[HD, dc]
 *   4: W_uv  f32[HD, dc]
 *   5: W_o   f32[D, HD]
 * K1 output view for "mla.forward":
 *   0: O     f32[N, T, D]
 *
 * K1 input views for "mla.backward":
 *   0..5: same as "mla.forward"
 *   6:    dO f32[N, T, D]
 * K1 output views for "mla.backward":
 *   0: dX     f32[N, T, D]
 *   1: dW_q   f32[HD, D]
 *   2: dW_dkv f32[dc, D]
 *   3: dW_uk  f32[HD, dc]
 *   4: dW_uv  f32[HD, dc]
 *   5: dW_o   f32[D, HD]
 *
 * Mathematical equations (evaluated in binary64, rounded to finite binary32):
 *   C_kv = X @ W_dkv^T                [N, T, dc]
 *   Q    = X @ W_q^T                  [N, T, H*Dh]
 *   K    = C_kv @ W_uk^T              [N, T, H*Dh]
 *   V    = C_kv @ W_uv^T              [N, T, H*Dh]
 *   Attn = CausalSDPA(Q, K, V; 1/sqrt(Dh)) via binary64 log-sum-exp
 *   O    = Attn @ W_o^T               [N, T, D]
 *
 * Borrows dense zero-offset CPU storage synchronously, allocates nothing,
 * validates alignment/aliasing/finiteness/FP-environment, and leaves all
 * output storage untouched on any error.
 */
const et_kernel_provider_v1 *et_mod5_mla_kernel_provider_v1(void);

/*
 * Direct C11 helpers accepting untyped pointers to avoid misaligned-pointer UB.
 * Return 0 on success or an et_kernel_error_category on failure; outputs and
 * cache state remain byte-for-byte untouched on any error.
 */
int32_t et_mod5_mla_forward_f32(const void *dims, const void *x,
                                const void *w_q, const void *w_dkv,
                                const void *w_uk, const void *w_uv,
                                const void *w_o, void *out,
                                et_kernel_error *error);

int32_t et_mod5_mla_backward_f32(const void *dims, const void *x,
                                 const void *w_q, const void *w_dkv,
                                 const void *w_uk, const void *w_uv,
                                 const void *w_o, const void *dy, void *dx,
                                 void *dw_q, void *dw_dkv, void *dw_uk,
                                 void *dw_uv, void *dw_o,
                                 et_kernel_error *error);

/*
 * Exact KV-cache byte telemetry helpers.
 * Return 0 (or 0.0) if dimensions are outside the admitted MLA domain.
 */
size_t et_mod5_mla_latent_kv_bytes(size_t batch, size_t tokens,
                                   size_t latent_dim);
size_t et_mod5_mla_full_kv_bytes(size_t batch, size_t tokens, size_t heads,
                                 size_t head_dim);
double et_mod5_mla_kv_compression_ratio(size_t heads, size_t head_dim,
                                        size_t latent_dim);

/*
 * Compressed latent KV-cache lifecycle and incremental decode.
 * All operations are zero-allocation and failure-atomic.
 */
int32_t et_mod5_mla_cache_init(void *cache, const void *dims, void *c_kv_buf,
                               size_t c_kv_bytes, et_kernel_error *error);

int32_t et_mod5_mla_cache_reset(void *cache, et_kernel_error *error);

int32_t et_mod5_mla_cache_prefill_f32(void *cache, size_t prefill_tokens,
                                      const void *x_prefill, const void *w_q,
                                      const void *w_dkv, const void *w_uk,
                                      const void *w_uv, const void *w_o,
                                      void *out_prefill,
                                      et_kernel_error *error);

int32_t et_mod5_mla_cache_step_f32(void *cache, const void *x_step,
                                   const void *w_q, const void *w_dkv,
                                   const void *w_uk, const void *w_uv,
                                   const void *w_o, void *out_step,
                                   et_kernel_error *error);

#ifdef __cplusplus
}
#endif

#endif
