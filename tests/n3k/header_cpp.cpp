#include "eshkol_transformer/n3k_primitives_abi.h"
#include <type_traits>

static_assert(ET_N3K_PRIMITIVES_ABI_MAJOR == 1u);
static_assert(ET_N3K_PRIMITIVES_ABI_MINOR == 0u);
static_assert(std::is_same_v<decltype(&et_n3k_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const et_kernel_provider_v1 *provider = et_n3k_kernel_provider_v1();
  return provider != nullptr && provider->abi_major == ET_KERNEL_ABI_MAJOR &&
                 provider->abi_minor == ET_KERNEL_ABI_MINOR &&
                 provider->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE
             ? 0 : 1;
}
