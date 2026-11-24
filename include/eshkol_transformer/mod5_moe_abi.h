#ifndef ESHKOL_TRANSFORMER_MOD5_MOE_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_MOE_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_MOE_ABI_MAJOR 1u
#define ET_MOD5_MOE_ABI_MINOR 0u

#define ET_MOD5_MOE_MIN_BATCH 1u
#define ET_MOD5_MOE_MAX_BATCH 2u
#define ET_MOD5_MOE_MIN_TOKENS 1u
#define ET_MOD5_MOE_MAX_TOKENS 32u
#define ET_MOD5_MOE_MIN_MODEL_DIM 2u
#define ET_MOD5_MOE_MAX_MODEL_DIM 32u
#define ET_MOD5_MOE_MIN_FFN_DIM 2u
#define ET_MOD5_MOE_MAX_FFN_DIM 64u
#define ET_MOD5_MOE_MIN_EXPERTS 2u
#define ET_MOD5_MOE_MAX_EXPERTS 8u
#define ET_MOD5_MOE_MIN_TOP_K 1u
#define ET_MOD5_MOE_MAX_TOP_K 4u

/*
 * Carrier-neutral K1 ABI 1.0 provider for Sparse Top-K Mixture-of-Experts (MoE)
 * with SwiGLU experts and Switch load-balancing auxiliary loss.
 *
 * Capability: "kernel.moe"
 * Operations: "moe.forward", "moe.backward"
 * Request shape: [N, T, D, F, E, K] with:
 *   N in 1..2, T in 1..32, D in 2..32, F in 2..64, E in 2..8, K in 1..4, K <= E
 *
 * K1 forward ("moe.forward") input views (in order):
 *   0: x            f32[N, T, D]
 *   1: w_gate       f32[E, D]
 *   2: w_gate_exp   f32[E, F, D]
 *   3: w_up_exp     f32[E, F, D]
 *   4: w_down_exp   f32[E, D, F]
 *   5: alpha_aux    f32[]           (finite, >= 0.0)
 * K1 forward output views (in order):
 *   0: y                   f32[N, T, D]
 *   1: aux_loss            f32[]    (Switch auxiliary loss L_aux)
 *   2: expert_counts       i64[E]   (per-expert routed token count)
 *   3: active_expert_evals i64[]    (exact sparse SwiGLU evaluations == N*T*K)
 *
 * K1 backward ("moe.backward") input views (in order):
 *   0: x            f32[N, T, D]
 *   1: w_gate       f32[E, D]
 *   2: w_gate_exp   f32[E, F, D]
 *   3: w_up_exp     f32[E, F, D]
 *   4: w_down_exp   f32[E, D, F]
 *   5: alpha_aux    f32[]           (finite, >= 0.0)
 *   6: dy           f32[N, T, D]
 * K1 backward output views (in order):
 *   0: dx           f32[N, T, D]
 *   1: dw_gate      f32[E, D]
 *   2: dw_gate_exp  f32[E, F, D]
 *   3: dw_up_exp    f32[E, F, D]
 *   4: dw_down_exp  f32[E, D, F]
 *
 * Router logits R = X @ W_gate^T ([N*T, E]), binary64 softmax probabilities P,
 * deterministic top-K selection with lowest-expert-index tie-breaking, and
 * renormalized weights w_i = P[e_i] / sum_{j<K} P[e_j]. Only selected experts
 * execute SwiGLU down(silu(gate(x)) * up(x)) per token. Switch auxiliary loss:
 *   L_aux = alpha_aux * E * sum_{e=0..E-1} f_e * P_mean_e,
 *   f_e = expert_counts[e] / (N * T * K), P_mean_e = (1/(N*T)) * sum_s P[s, e].
 * Backward computes exact analytic gradients of L_task(Y) + L_aux.
 *
 * Borrows dense zero-offset CPU storage synchronously, allocates nothing,
 * validates alignment/aliasing/finiteness/FP-environment, and leaves all output
 * bytes untouched on any error.
 */
const et_kernel_provider_v1 *et_mod5_moe_kernel_provider_v1(void);

/*
 * Direct C11 helpers accepting untyped void * pointers so callers (including
 * raw FFI or negative alignment tests) avoid misaligned-pointer UB. Return 0 on
 * success or an et_kernel_error_category on failure; all output storage is
 * byte-for-byte untouched on failure.
 */
int32_t et_mod5_moe_forward_f32(size_t batch, size_t tokens, size_t model_dim,
                                size_t ffn_dim, size_t num_experts,
                                size_t top_k, const void *x,
                                const void *w_gate, const void *w_gate_exp,
                                const void *w_up_exp, const void *w_down_exp,
                                double alpha_aux, void *y, void *aux_loss,
                                void *expert_counts, void *active_expert_evals,
                                et_kernel_error *error);

int32_t et_mod5_moe_backward_f32(size_t batch, size_t tokens, size_t model_dim,
                                 size_t ffn_dim, size_t num_experts,
                                 size_t top_k, const void *x,
                                 const void *w_gate, const void *w_gate_exp,
                                 const void *w_up_exp, const void *w_down_exp,
                                 double alpha_aux, const void *dy, void *dx,
                                 void *dw_gate, void *dw_gate_exp,
                                 void *dw_up_exp, void *dw_down_exp,
                                 et_kernel_error *error);

#ifdef __cplusplus
}
#endif

#endif
