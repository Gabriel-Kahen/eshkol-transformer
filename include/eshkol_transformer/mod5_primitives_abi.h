#ifndef ESHKOL_TRANSFORMER_MOD5_PRIMITIVES_ABI_H
#define ESHKOL_TRANSFORMER_MOD5_PRIMITIVES_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MOD5_PRIMITIVES_ABI_MAJOR 1u
#define ET_MOD5_PRIMITIVES_ABI_MINOR 0u
#define ET_MOD5_MAX_EXACT_POSITION INT64_C(16777215)

/*
 * Carrier-neutral K1 provider for RMSNorm, SwiGLU and variant attention.
 * Borrows dense, zero-offset CPU views synchronously and retains no storage.
 * Operation-specific tensor tables are documented in docs/MOD5_VARIANTS.md.
 */
const et_kernel_provider_v1 *et_mod5_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
