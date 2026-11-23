#ifndef ESHKOL_TRANSFORMER_AMD4_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_AMD4_ESHKOL_BRIDGE_H

#include "eshkol_transformer/amd4_hip_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Deterministic injected observability scenarios exposed for AOT verification.
 * All scenario reports carry ET_AMD4_SOURCE_INJECTED via
 * et_amd4_probe_with_ops_v1. */
enum {
  ET_AMD4_AOT_SCENARIO_TWO_DEVICE = 1,
  ET_AMD4_AOT_SCENARIO_OOM = 2,
  ET_AMD4_AOT_SCENARIO_DRIVER_ERROR = 3,
  ET_AMD4_AOT_SCENARIO_ROCBLAS_MISSING_SYMBOLS = 4
};

/* K1 AMD4 capability selectors. */
enum {
  ET_AMD4_AOT_CAP_GEMM = 0,
  ET_AMD4_AOT_CAP_ATTENTION = 1
};

/* Caller-owned report sizing and live/injected observability probes. */
int64_t et_amd4_aot_report_size(void);
int32_t et_amd4_aot_probe_live(int32_t ordinal, et_amd4_report_v1 *report);
int32_t et_amd4_aot_probe_scenario(int32_t scenario_id, int32_t ordinal,
                                   et_amd4_report_v1 *report);

/* Report field accessors. Return -1 (or ET_AMD4_ERR_INVALID_ARG for status)
 * on null report, out-of-range device index, or unobserved stage. */
int32_t et_amd4_aot_report_status(const et_amd4_report_v1 *report);
int32_t et_amd4_aot_report_source(const et_amd4_report_v1 *report);
int32_t et_amd4_aot_report_kfd(const et_amd4_report_v1 *report);
int32_t et_amd4_aot_report_hip_library(const et_amd4_report_v1 *report);
int32_t et_amd4_aot_report_rocblas_library(const et_amd4_report_v1 *report);
int64_t et_amd4_aot_report_device_count(const et_amd4_report_v1 *report);
int64_t et_amd4_aot_report_reported_device_count(
    const et_amd4_report_v1 *report);
int32_t et_amd4_aot_report_device_compute_units(const et_amd4_report_v1 *report,
                                                int32_t index);
int32_t et_amd4_aot_report_device_wavefront_size(
    const et_amd4_report_v1 *report, int32_t index);
int64_t et_amd4_aot_report_device_total_vram(const et_amd4_report_v1 *report,
                                             int32_t index);
int64_t et_amd4_aot_report_device_free_vram(const et_amd4_report_v1 *report,
                                            int32_t index);

/* Format a report into caller storage using et_amd4_format_report_v1.
 * Passing NULL with capacity 0 measures required_capacity (including NUL).
 * Short buffers return ET_AMD4_ERR_OUT_OF_RANGE and leave buffer intact. */
int32_t et_amd4_aot_format_report(const et_amd4_report_v1 *report, char *buffer,
                                  int64_t capacity, int64_t *required_capacity);
int32_t et_amd4_aot_read_byte(const char *buffer, int64_t index);

/* K1 runtime registration for et_amd4_kernel_provider_v1(), capability query,
 * JSON report, and explicit-rejection dispatch (zero silent CPU fallback). */
int32_t et_amd4_aot_k1_open(et_kernel_runtime **runtime);
void et_amd4_aot_k1_close(et_kernel_runtime *runtime);
int64_t et_amd4_aot_k1_capability_count(const et_kernel_runtime *runtime);
int32_t et_amd4_aot_k1_capability_status(const et_kernel_runtime *runtime,
                                         int32_t capability_id);
int32_t et_amd4_aot_k1_dispatch(const et_kernel_runtime *runtime,
                                int32_t capability_id, const uint32_t *input,
                                uint32_t *output, int32_t *out_error_code);
int32_t et_amd4_aot_k1_report_json(const et_kernel_runtime *runtime,
                                   char *buffer, int64_t capacity,
                                   int64_t *required_capacity);

/* 32-bit canary bit-pattern marshaling for Eshkol AOT tests. */
int32_t et_amd4_aot_store_u32(uint32_t *buffer, int64_t index, int64_t value);
int64_t et_amd4_aot_load_u32(const uint32_t *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
