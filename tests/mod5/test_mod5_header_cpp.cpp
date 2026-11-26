#include "eshkol_transformer/mod5_primitives_abi.h"

#include <type_traits>

static_assert(ET_MOD5_PRIMITIVES_ABI_MAJOR == 1u);
static_assert(ET_MOD5_PRIMITIVES_ABI_MINOR == 0u);
static_assert(ET_MOD5_MAX_EXACT_POSITION == 16777215);
static_assert(std::is_same_v<decltype(&et_mod5_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const auto *provider = et_mod5_kernel_provider_v1();
  return provider && provider->abi_major == ET_KERNEL_ABI_MAJOR ? 0 : 1;
}
