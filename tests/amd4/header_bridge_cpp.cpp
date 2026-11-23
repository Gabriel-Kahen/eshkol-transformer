#include "amd4_eshkol_bridge.h"

#include <cstdint>
#include <cstring>

int main() {
  if (et_amd4_aot_report_size() !=
      static_cast<int64_t>(sizeof(et_amd4_report_v1))) {
    return 1;
  }

  et_amd4_report_v1 report{};
  if (et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE,
                                 ET_AMD4_ALL_DEVICES, &report) != ET_AMD4_OK) {
    return 2;
  }
  if (et_amd4_aot_report_source(&report) != ET_AMD4_SOURCE_INJECTED ||
      et_amd4_aot_report_device_count(&report) != 2 ||
      et_amd4_aot_report_reported_device_count(&report) != 2 ||
      et_amd4_aot_report_device_compute_units(&report, 0) != 110 ||
      et_amd4_aot_report_device_wavefront_size(&report, 1) != 32) {
    return 3;
  }

  int64_t required = 0;
  if (et_amd4_aot_format_report(&report, nullptr, 0, &required) != ET_AMD4_OK ||
      required <= 1) {
    return 4;
  }

  et_kernel_runtime *runtime = nullptr;
  if (et_amd4_aot_k1_open(&runtime) != 0 || runtime == nullptr) {
    return 5;
  }
  if (et_amd4_aot_k1_capability_count(runtime) < 2 ||
      et_amd4_aot_k1_capability_status(runtime, ET_AMD4_AOT_CAP_GEMM) !=
          static_cast<int32_t>(ET_KERNEL_CAPABILITY_UNAVAILABLE) ||
      et_amd4_aot_k1_capability_status(runtime, ET_AMD4_AOT_CAP_ATTENTION) !=
          static_cast<int32_t>(ET_KERNEL_CAPABILITY_UNAVAILABLE)) {
    et_amd4_aot_k1_close(runtime);
    return 6;
  }

  uint32_t input = 0x3f800000u;
  uint32_t output = 0x7fc12345u;
  int32_t code = 0;
  const int32_t cat = et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM,
                                              &input, &output, &code);
  et_amd4_aot_k1_close(runtime);
  if (cat != ET_KERNEL_ERROR_UNSUPPORTED ||
      code != static_cast<int32_t>(ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED) ||
      output != 0x7fc12345u || input != 0x3f800000u) {
    return 7;
  }

  return 0;
}
