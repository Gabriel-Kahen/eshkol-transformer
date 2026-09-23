#include "eshkol_transformer/g3c4_primitives_abi.h"

_Static_assert(ET_G3C4_PRIMITIVES_ABI_MAJOR == 1u, "major");
_Static_assert(ET_G3C4_PRIMITIVES_ABI_MINOR == 0u, "minor");

int main(void) {
  const et_kernel_provider_v1 *(*accessor)(void) = et_g3c4_kernel_provider_v1;
  const et_kernel_provider_v1 *provider = accessor();
  return provider != 0 && provider->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE &&
                 provider->abi_major == ET_KERNEL_ABI_MAJOR &&
                 provider->abi_minor == ET_KERNEL_ABI_MINOR &&
                 provider->required_features == 0u
             ? 0
             : 1;
}
