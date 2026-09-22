#ifndef ESHKOL_TRANSFORMER_G3S_SAMPLING_ABI_H
#define ESHKOL_TRANSFORMER_G3S_SAMPLING_ABI_H
#include "eshkol_transformer/kernel_abi.h"
#define ET_G3S_SAMPLING_ABI_MAJOR 1u
#define ET_G3S_SAMPLING_ABI_MINOR 0u
#ifdef __cplusplus
extern "C" {
#endif
/* Registry-free synchronous CPU-f32 V256 sampling; see G3_S_ABI_PROPOSAL.md. */
const et_kernel_provider_v1 *et_g3s_kernel_provider_v1(void);
#ifdef __cplusplus
}
#endif
#endif
