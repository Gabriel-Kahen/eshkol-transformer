#ifndef ESHKOL_TRANSFORMER_E3_EVALUATION_METRICS_ABI_H
#define ESHKOL_TRANSFORMER_E3_EVALUATION_METRICS_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_E3_EVALUATION_METRICS_ABI_MAJOR 1u
#define ET_E3_EVALUATION_METRICS_ABI_MINOR 0u

/* Immutable metadata for explicit K1 discovery only. CPU f32/BOOL/i64 views
 * follow docs/e3/E3_BOOL_METRICS_CONTRACT.md at request shape [1,2,256].
 * Dispatch borrows caller-owned storage and retains no pointers. */
const et_kernel_provider_v1 *et_e3_metrics_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif
#endif
