#ifndef ESHKOL_TRANSFORMER_N3K_PRIMITIVES_ABI_H
#define ESHKOL_TRANSFORMER_N3K_PRIMITIVES_ABI_H
#include "eshkol_transformer/kernel_abi.h"
#ifdef __cplusplus
extern "C" {
#endif
#define ET_N3K_PRIMITIVES_ABI_MAJOR 1u
#define ET_N3K_PRIMITIVES_ABI_MINOR 0u
/* Explicit immutable carrier-neutral provider. Synchronous borrowed inputs and
 * disjoint caller-owned outputs; no allocation, retained views or global registry.
 * Invoke requires complete K1 generic plus provider validation and stable storage. */
const et_kernel_provider_v1 *et_n3k_kernel_provider_v1(void);
#ifdef __cplusplus
}
#endif
#endif
