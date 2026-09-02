#ifndef ESHKOL_TRANSFORMER_A2_ATTENTION_ABI_H
#define ESHKOL_TRANSFORMER_A2_ATTENTION_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_A2_ATTENTION_ABI_MAJOR 1u
#define ET_A2_ATTENTION_ABI_MINOR 0u
#define ET_A2_MAX_EXACT_POSITION INT64_C(16777215)

/*
 * Carrier-neutral K1 provider.  It neither owns tensor storage nor exports the
 * canonical K1 provider symbol.
 *
 * kernel.causal-attention uses request shape [N,Hq,Hkv,Tq,Tk,Dh].
 * forward inputs:  q, k, v, query_positions, key_positions, keep_mask
 * forward output:  attention output
 * backward inputs: the six forward inputs, then upstream
 * backward outputs: dq, dk, dv
 *
 * kernel.rope uses request shape [N,H,T,Dh].
 * forward inputs:  x, positions, inv_freq; output: y
 * backward inputs: upstream, positions, inv_freq; output: dx
 *
 * All floating views are dense zero-offset CPU f32.  Positions are dense CPU
 * i64 and masks are dense CPU bool.  Outputs are caller-owned and disjoint as
 * required by K1.  Dispatch allocates no storage.
 */
const et_kernel_provider_v1 *et_a2_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
