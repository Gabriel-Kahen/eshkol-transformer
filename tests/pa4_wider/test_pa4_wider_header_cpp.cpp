#include "eshkol_transformer/pa4_wider_attention_abi.h"

#include <type_traits>

using accessor = const et_kernel_provider_v1 *(*)();
static_assert(ET_PA4_WIDER_ABI_MAJOR == ET_KERNEL_ABI_MAJOR);
static_assert(ET_PA4_WIDER_ABI_MINOR == ET_KERNEL_ABI_MINOR);
static_assert(ET_PA4_WIDER_MAX_KEYS == 64u);
static_assert(ET_PA4_WIDER_MAX_EXACT_POSITION == 16777215);
static_assert(
    std::is_same_v<decltype(&et_pa4_wider_ref_provider_v1), accessor>);
static_assert(
    std::is_same_v<decltype(&et_pa4_wider_online_provider_v1), accessor>);

int main() {
  const auto *ref = et_pa4_wider_ref_provider_v1();
  const auto *online = et_pa4_wider_online_provider_v1();
  return ref != nullptr && online != nullptr && ref != online &&
                 ref->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE &&
                 online->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE &&
                 ref->abi_major == ET_PA4_WIDER_ABI_MAJOR &&
                 online->abi_major == ET_PA4_WIDER_ABI_MAJOR &&
                 ref->abi_minor == ET_PA4_WIDER_ABI_MINOR &&
                 online->abi_minor == ET_PA4_WIDER_ABI_MINOR
             ? 0
             : 1;
}
