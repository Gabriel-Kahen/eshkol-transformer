#include "eshkol_transformer/amd4_hip_abi.h"

#include <type_traits>

static_assert(ET_AMD4_ABI_MAJOR == 1 && ET_AMD4_ABI_MINOR == 0);
static_assert(std::is_standard_layout_v<et_amd4_result_v1>);
static_assert(std::is_standard_layout_v<et_amd4_device_info_v1>);
static_assert(std::is_standard_layout_v<et_amd4_report_v1>);
static_assert(std::is_standard_layout_v<et_amd4_ops_v1>);

int main() {
  et_amd4_ops_v1 ops{};
  ops.struct_size = sizeof(ops);
  et_amd4_report_v1 report{};
  size_t required = 0;
  const auto *provider = et_amd4_kernel_provider_v1();
  return et_amd4_probe_v1(ET_AMD4_ALL_DEVICES, nullptr) ==
                     ET_AMD4_ERR_INVALID_ARG &&
                 et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES,
                                           &report) ==
                     ET_AMD4_ERR_UNAVAILABLE_RUNTIME &&
                 et_amd4_format_report_v1(&report, nullptr, 0, &required) ==
                     ET_AMD4_OK &&
                 required > 0 && provider->abi_major == 1 &&
                 provider->abi_minor == 0
             ? 0
             : 1;
}
