#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include <type_traits>

static_assert(ET_E3_EVALUATION_METRICS_ABI_MAJOR == 1u);
static_assert(ET_E3_EVALUATION_METRICS_ABI_MINOR == 0u);
static_assert(std::is_same_v<decltype(&et_e3_metrics_kernel_provider_v1),
                             const et_kernel_provider_v1 *(*)()>);

int main() {
  const auto *p = et_e3_metrics_kernel_provider_v1();
  return p && p->abi_major == 1u && p->abi_minor == 0u &&
                 p->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE ? 0 : 1;
}
