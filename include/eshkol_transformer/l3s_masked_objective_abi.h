#ifndef ESHKOL_TRANSFORMER_L3S_MASKED_OBJECTIVE_ABI_H
#define ESHKOL_TRANSFORMER_L3S_MASKED_OBJECTIVE_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_L3S_MASKED_OBJECTIVE_ABI_MAJOR 1u
#define ET_L3S_MASKED_OBJECTIVE_ABI_MINOR 0u

/* Immutable metadata for explicit K1 discovery. Dispatch requires the exact
 * [1,2] schemas and caller-owned storage documented in L3S_MASKED_OBJECTIVE.md.
 * This accessor defines neither the canonical K1 resolver nor a storage owner. */
const et_kernel_provider_v1 *et_l3s_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif
#endif
