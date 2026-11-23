#ifndef ESHKOL_TRANSFORMER_PA4_WIDER_ATTENTION_ABI_H
#define ESHKOL_TRANSFORMER_PA4_WIDER_ATTENTION_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#define ET_PA4_WIDER_ABI_MAJOR 1u
#define ET_PA4_WIDER_ABI_MINOR 0u
#define ET_PA4_WIDER_MAX_KEYS 64u
#define ET_PA4_WIDER_MAX_EXACT_POSITION 16777215

#ifdef __cplusplus
extern "C" {
#endif

/* Explicitly linked K1 providers for dense CPU-f32 causal attention. */
const et_kernel_provider_v1 *et_pa4_wider_ref_provider_v1(void);
const et_kernel_provider_v1 *et_pa4_wider_online_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
