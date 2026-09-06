#include "eshkol_transformer/n2_primitives_abi.h"

#include <cstdint>
#include <type_traits>

static_assert(ET_N2_PRIMITIVES_ABI_MAJOR == 1u);
static_assert(ET_N2_PRIMITIVES_ABI_MINOR == 0u);
static_assert(std::is_same_v<decltype(&et_n2_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const et_kernel_provider_v1 *provider = et_n2_kernel_provider_v1();
  return provider != nullptr && provider->abi_major == ET_KERNEL_ABI_MAJOR ? 0
                                                                           : 1;
}
