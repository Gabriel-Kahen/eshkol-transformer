#include "eshkol_transformer/g3n_primitives_abi.h"
_Static_assert(ET_G3N_PRIMITIVES_ABI_MAJOR == 1u, "major");
_Static_assert(ET_G3N_PRIMITIVES_ABI_MINOR == 0u, "minor");
int main(void) {
  const et_kernel_provider_v1 *(*accessor)(void) = et_g3n_kernel_provider_v1;
  const et_kernel_provider_v1 *p = accessor();
  return p && p->struct_size == 96u && p->abi_major == 1u &&
      p->abi_minor == 0u && p->required_features == 0u ? 0 : 1;
}
