#ifndef ESHKOL_TRANSFORMER_N2_PRIMITIVES_ABI_H
#define ESHKOL_TRANSFORMER_N2_PRIMITIVES_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_N2_PRIMITIVES_ABI_MAJOR 1u
#define ET_N2_PRIMITIVES_ABI_MINOR 0u

/*
 * Carrier-neutral, registry-free K1 provider.  It borrows caller-owned dense,
 * zero-offset CPU views synchronously and retains no storage.  It neither owns
 * a tensor carrier nor exports K1's canonical provider symbol.
 */
const et_kernel_provider_v1 *et_n2_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
