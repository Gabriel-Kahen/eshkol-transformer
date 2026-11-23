#ifndef ESHKOL_TRANSFORMER_MOD5_ROPE_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_ROPE_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_ROPE_ABI_MAJOR 1u
#define ET_MOD5_ROPE_ABI_MINOR 0u

#define ET_MOD5_ROPE_MIN_BATCH 1u
#define ET_MOD5_ROPE_MAX_BATCH 2u
#define ET_MOD5_ROPE_MIN_HEADS 1u
#define ET_MOD5_ROPE_MAX_HEADS 4u
#define ET_MOD5_ROPE_MIN_TOKENS 1u
#define ET_MOD5_ROPE_MAX_TOKENS 64u
#define ET_MOD5_ROPE_MIN_HEAD_DIM 2u
#define ET_MOD5_ROPE_MAX_HEAD_DIM 16u
#define ET_MOD5_ROPE_MAX_POSITION INT64_C(16777215)

#define ET_MOD5_ROPE_DEFAULT_THETA 10000.0
#define ET_MOD5_ROPE_DEFAULT_SCALE 1.0

/*
 * Carrier-neutral K1 ABI 1.0 provider for Rotary Position Embeddings (RoPE).
 * Capability: "kernel.rope"
 * Operations: "rope.forward", "rope.backward"
 * Request shape: [N, H, T, Dh] with N in 1..2, H in 1..4, T in 1..64,
 * even Dh in 2..16.
 *
 * K1 input views (in order):
 *   0: x (forward) or dy (backward)  f32[N, H, T, Dh]
 *   1: positions                     i64[N, T] in [0, ET_MOD5_ROPE_MAX_POSITION]
 *   2: theta                         f32[] finite, > 1.0
 *   3: scale                         f32[] finite, > 0.0
 * K1 output view:
 *   0: y (forward) or dx (backward)  f32[N, H, T, Dh]
 *
 * Pairing convention: adjacent pairs (x[2*k], x[2*k + 1]) for k = 0..Dh/2 - 1.
 * Angle phi = (pos / scale) * theta^(-2*k / Dh) evaluated in binary64, with
 * cos(phi) and sin(phi) rounded to binary32.
 *   forward:  y0  = x0 * c - x1 * s,   y1  = x0 * s + x1 * c
 *   backward: dx0 = dy0 * c + dy1 * s, dx1 = -dy0 * s + dy1 * c
 *
 * Borrows dense zero-offset CPU storage synchronously, allocates nothing,
 * validates alignment/aliasing/finiteness/FP-environment, and leaves output
 * bytes untouched on any error.
 */
const et_kernel_provider_v1 *et_mod5_rope_kernel_provider_v1(void);

/*
 * Direct C11 helpers accepting untyped pointers so callers (including raw FFI
 * or negative alignment tests) avoid misaligned-pointer UB. Return 0 on success
 * or an et_kernel_error_category on failure; output storage is untouched on
 * failure.
 */
int32_t et_mod5_rope_forward_f32(size_t batch, size_t heads, size_t tokens,
                                 size_t head_dim, const void *x,
                                 const void *positions, double theta,
                                 double scale, void *y,
                                 et_kernel_error *error);

int32_t et_mod5_rope_backward_f32(size_t batch, size_t heads, size_t tokens,
                                  size_t head_dim, const void *dy,
                                  const void *positions, double theta,
                                  double scale, void *dx,
                                  et_kernel_error *error);

#ifdef __cplusplus
}
#endif

#endif
