#include "eshkol_transformer/kernel_abi.h"

#include <type_traits>

static_assert(std::is_standard_layout_v<et_kernel_provider_v1>);
static_assert(std::is_standard_layout_v<et_kernel_capability_v1>);
static_assert(std::is_standard_layout_v<et_kernel_tensor_view_v1>);

int main() {
  return et_kernel_abi_major() == 1 && et_kernel_abi_minor() == 0 ? 0 : 1;
}
