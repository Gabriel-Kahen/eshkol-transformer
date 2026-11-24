#ifndef ESHKOL_TRANSFORMER_MOD5_MOE_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MOD5_MOE_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT transport for MOD5 Sparse Top-K MoE K1 provider ("kernel.moe").
 * Status is zero on success or an et_kernel_error_category on failure.
 * All tensor pointers are accepted as void * so misaligned raw pointers from
 * Eshkol FFI are validated cleanly without misaligned-pointer UB.
 */
int32_t et_mod5_moe_aot_open(et_kernel_runtime **runtime);
void et_mod5_moe_aot_close(et_kernel_runtime *runtime);

int32_t et_mod5_moe_aot_forward(const et_kernel_runtime *runtime, int64_t batch,
                                int64_t tokens, int64_t model_dim,
                                int64_t ffn_dim, int64_t num_experts,
                                int64_t top_k, void *x, void *w_gate,
                                void *w_gate_exp, void *w_up_exp,
                                void *w_down_exp, double alpha_aux, void *y,
                                void *aux_loss, void *expert_counts,
                                void *active_expert_evals);

int32_t et_mod5_moe_aot_backward(const et_kernel_runtime *runtime,
                                 int64_t batch, int64_t tokens,
                                 int64_t model_dim, int64_t ffn_dim,
                                 int64_t num_experts, int64_t top_k, void *x,
                                 void *w_gate, void *w_gate_exp, void *w_up_exp,
                                 void *w_down_exp, double alpha_aux, void *dy,
                                 void *dx, void *dw_gate, void *dw_gate_exp,
                                 void *dw_up_exp, void *dw_down_exp);

/* Scalar marshaling between Eshkol f64/i64 values and owned f32/i64 buffers.
 * Values outside finite binary32 range become +/-INFINITY and are rejected at
 * dispatch. Caller supplies a valid non-negative index within buffer bounds.
 */
void et_mod5_moe_aot_store_f32(void *buffer, int64_t index, double value);
double et_mod5_moe_aot_load_f32(const void *buffer, int64_t index);
void et_mod5_moe_aot_store_i64(void *buffer, int64_t index, int64_t value);
int64_t et_mod5_moe_aot_load_i64(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
