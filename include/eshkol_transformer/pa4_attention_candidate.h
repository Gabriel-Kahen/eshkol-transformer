#ifndef ESHKOL_TRANSFORMER_PA4_ATTENTION_CANDIDATE_H
#define ESHKOL_TRANSFORMER_PA4_ATTENTION_CANDIDATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Explicitly linked, experimental provider; it uses the A2 tensor contract. */
const et_kernel_provider_v1 *et_pa4_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
