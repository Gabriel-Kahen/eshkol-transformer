#ifndef ESHKOL_TRANSFORMER_MOD5_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MOD5_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private AOT transport. Status is zero on success or a K1 error category.
 * Keep the discovered runtime alive for all calls, then close it once.
 * All tensor spans are borrowed, correctly sized, aligned and disjoint.
 *
 * RMSNorm: x, dy, y, dx [1,8,16]; gamma, dgamma [16]; epsilon scalar.
 * SwiGLU: all views [1,8,32].
 * Attention: q, dy, y, dq [1,4,8,4]; k, v, dk, dv [1,Hkv,8,4];
 * qpos, kpos [1,8]; mask [1,8,8]; slopes [4]. Hkv is 1 or 2.
 */
int32_t et_mod5_aot_open(et_kernel_runtime **runtime);
void et_mod5_aot_close(et_kernel_runtime *runtime);

int32_t et_mod5_aot_rms_forward(const et_kernel_runtime *runtime, float *x,
                                float *gamma, float *epsilon, float *y);
int32_t et_mod5_aot_rms_backward(const et_kernel_runtime *runtime, float *x,
                                 float *gamma, float *epsilon, float *dy,
                                 float *dx, float *dgamma);
int32_t et_mod5_aot_swiglu_forward(const et_kernel_runtime *runtime,
                                   float *gate, float *up, float *y);
int32_t et_mod5_aot_swiglu_backward(const et_kernel_runtime *runtime,
                                    float *gate, float *up, float *dy,
                                    float *dgate, float *dup);
int32_t et_mod5_aot_attention_forward(const et_kernel_runtime *runtime,
                                      int64_t kv_heads, float *q, float *k,
                                      float *v, int64_t *qpos, int64_t *kpos,
                                      uint8_t *mask, float *slopes,
                                      int64_t window, float *y);
int32_t et_mod5_aot_attention_backward(const et_kernel_runtime *runtime,
                                       int64_t kv_heads, float *q, float *k,
                                       float *v, int64_t *qpos, int64_t *kpos,
                                       uint8_t *mask, float *slopes,
                                       int64_t window, float *dy, float *dq,
                                       float *dk, float *dv);

/* Scalar marshaling between Eshkol f64 tensors and owned binary32 buffers.
 * Values outside the finite f32 range become nonfinite and are rejected at
 * dispatch. The caller supplies a valid index into an aligned float buffer.
 */
void et_mod5_aot_store_f32(float *buffer, int64_t index, double value);
double et_mod5_aot_load_f32(const float *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
