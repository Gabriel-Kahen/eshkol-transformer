#ifndef ESHKOL_TRANSFORMER_INDEXED_CROSS_ENTROPY_H
#define ESHKOL_TRANSFORMER_INDEXED_CROSS_ENTROPY_H

#include "eshkol_transformer/kernel_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_L2_INDEXED_CROSS_ENTROPY_ABI_MAJOR 1u
#define ET_L2_INDEXED_CROSS_ENTROPY_ABI_MINOR 0u
#define ET_L2_INDEXED_CROSS_ENTROPY_MAX_EXTENT UINT64_C(1664510)
#define ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY \
  "kernel.indexed-cross-entropy"
#define ET_L2_INDEXED_CROSS_ENTROPY_FORWARD \
  "indexed-cross-entropy.forward"
#define ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD \
  "indexed-cross-entropy.backward"

int32_t et_l2_indexed_cross_entropy_abi_major_v1(void);
int32_t et_l2_indexed_cross_entropy_abi_minor_v1(void);

/*
 * Returns immutable process-lifetime metadata for explicit K1 resolver use.
 * This library deliberately does not define ET_KERNEL_PROVIDER_SYMBOL_V1.
 */
const et_kernel_provider_v1 *et_l2_indexed_cross_entropy_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
