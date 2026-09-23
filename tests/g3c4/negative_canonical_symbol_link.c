#include "eshkol_transformer/g3c4_primitives_abi.h"
extern const et_kernel_provider_v1 *eshkol_transformer_kernel_provider_v1(void);
int main(void) { return eshkol_transformer_kernel_provider_v1() ? 0 : 1; }
