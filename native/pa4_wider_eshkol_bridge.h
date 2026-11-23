#ifndef ESHKOL_TRANSFORMER_PA4_WIDER_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_PA4_WIDER_ESHKOL_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Private AOT transport. Status is zero or a K1 error category.
 * Variant 0 opens the reference provider and variant 1 opens the online
 * provider. Open publishes a runtime on success. Close it once after use.
 */
int32_t et_pa4_wider_aot_open(int32_t variant, et_kernel_runtime **runtime);
void et_pa4_wider_aot_close(et_kernel_runtime *runtime);

/* Borrowed, naturally aligned, dense CPU buffers, with extents in elements:
 *   Q, O, dO, dQ: [N,Hq,Tq,Dh] f32
 *   K, V, dK, dV: [N,Hkv,Tk,Dh] f32
 *   qpos: [N,Tq] i64; kpos: [N,Tk] i64; mask: [N,Tq,Tk] u8 (0 or 1).
 * N=1..2, Hq=1..4, Hkv divides Hq, Tq/Tk=1..64, Dh=1..16.
 * Input pointers borrow readable storage. K and V are disjoint, and each
 * output is disjoint from inputs and other outputs. The caller supplies
 * addressable spans of the specified sizes. Outputs are unchanged on error.
 */
int32_t et_pa4_wider_aot_forward(const et_kernel_runtime *runtime, int64_t n,
                                 int64_t hq, int64_t hkv, int64_t tq,
                                 int64_t tk, int64_t dh, float *q, float *k,
                                 float *v, int64_t *qpos, int64_t *kpos,
                                 uint8_t *mask, float *out);
int32_t et_pa4_wider_aot_backward(const et_kernel_runtime *runtime, int64_t n,
                                  int64_t hq, int64_t hkv, int64_t tq,
                                  int64_t tk, int64_t dh, float *q, float *k,
                                  float *v, int64_t *qpos, int64_t *kpos,
                                  uint8_t *mask, float *dout, float *dq,
                                  float *dk, float *dv);

/* The caller supplies a valid index into an aligned buffer. f32 storage
 * rounds f64 to binary32 and maps magnitudes beyond FLT_MAX to infinity.
 * Loading widens finite binary32 exactly. f32_bits returns the raw unsigned
 * binary32 representation in an Eshkol-compatible i64, including signed zero.
 * store_u8 accepts byte values 0..255.
 */
void et_pa4_wider_aot_store_f32(float *buffer, int64_t index, double value);
double et_pa4_wider_aot_load_f32(const float *buffer, int64_t index);
int64_t et_pa4_wider_aot_f32_bits(const float *buffer, int64_t index);
void et_pa4_wider_aot_store_i64(int64_t *buffer, int64_t index, int64_t value);
void et_pa4_wider_aot_store_u8(uint8_t *buffer, int64_t index, int64_t value);

#ifdef __cplusplus
}
#endif

#endif
