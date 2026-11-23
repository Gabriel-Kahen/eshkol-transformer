#include "amd4_eshkol_bridge.h"

#include <limits.h>
#include <string.h>

typedef struct amd4_mock_state {
  int32_t scenario_id;
  int32_t device_count;
  int32_t active_ordinal;
  int32_t driver_code;
  int32_t memory_code;
  int32_t rocblas_create_code;
  int32_t handle_token;
  int32_t live_handles;
} amd4_mock_state;

static int span_overlaps(const void *a, size_t a_size, const void *b,
                         size_t b_size) {
  const uintptr_t a_start = (uintptr_t)a;
  const uintptr_t b_start = (uintptr_t)b;
  return a != NULL && b != NULL && a_size != 0u && b_size != 0u &&
         (a_start <= b_start ? b_start - a_start < a_size
                             : a_start - b_start < b_size);
}

static int32_t mock_hip_runtime_version(void *context, int32_t *version) {
  (void)context;
  *version = 60443482;
  return 0;
}

static int32_t mock_hip_driver_version(void *context, int32_t *version) {
  const amd4_mock_state *state = (const amd4_mock_state *)context;
  if (state->driver_code != 0) {
    return state->driver_code;
  }
  *version = 60443482;
  return 0;
}

static int32_t mock_hip_device_count(void *context, int32_t *count) {
  const amd4_mock_state *state = (const amd4_mock_state *)context;
  *count = state->device_count;
  return 0;
}

static int32_t mock_hip_device_properties(
    void *context, int32_t ordinal, et_amd4_device_properties_v1 *properties) {
  const amd4_mock_state *state = (const amd4_mock_state *)context;
  if (ordinal < 0 || ordinal >= state->device_count) {
    return 101;
  }
  memset(properties, 0, sizeof(*properties));
  if (ordinal == 0) {
    memcpy(properties->name, "AMD Instinct MI250X",
           sizeof("AMD Instinct MI250X"));
    memcpy(properties->gcn_arch_name, "gfx90a:sramecc+:xnack-",
           sizeof("gfx90a:sramecc+:xnack-"));
    properties->pci_domain = 0;
    properties->pci_bus = 10;
    properties->pci_device = 0;
    properties->compute_units = 110;
    properties->wavefront_size = 64;
    properties->total_vram_bytes = UINT64_C(68719476736);
  } else {
    memcpy(properties->name, "AMD Radeon RX 7900 XTX",
           sizeof("AMD Radeon RX 7900 XTX"));
    memcpy(properties->gcn_arch_name, "gfx1100", sizeof("gfx1100"));
    properties->pci_domain = 0;
    properties->pci_bus = 11;
    properties->pci_device = 0;
    properties->compute_units = 96;
    properties->wavefront_size = 32;
    properties->total_vram_bytes = UINT64_C(25769803776);
  }
  return 0;
}

static int32_t mock_hip_get_device(void *context, int32_t *ordinal) {
  const amd4_mock_state *state = (const amd4_mock_state *)context;
  *ordinal = state->active_ordinal;
  return 0;
}

static int32_t mock_hip_set_device(void *context, int32_t ordinal) {
  amd4_mock_state *state = (amd4_mock_state *)context;
  if (ordinal < 0 || ordinal >= state->device_count) {
    return 101;
  }
  state->active_ordinal = ordinal;
  return 0;
}

static int32_t mock_hip_memory_info(void *context, uint64_t *free_bytes,
                                    uint64_t *total_bytes) {
  const amd4_mock_state *state = (const amd4_mock_state *)context;
  if (state->memory_code != 0) {
    return state->memory_code;
  }
  if (state->active_ordinal == 0) {
    *free_bytes = UINT64_C(64424509440);
    *total_bytes = UINT64_C(68719476736);
  } else {
    *free_bytes = UINT64_C(21474836480);
    *total_bytes = UINT64_C(25769803776);
  }
  return 0;
}

static int32_t mock_rocblas_version(void *context, char *version,
                                    size_t capacity) {
  static const char text[] = "4.3.0.mock";
  (void)context;
  if (capacity < sizeof(text)) {
    return 3;
  }
  memcpy(version, text, sizeof(text));
  return 0;
}

static int32_t mock_rocblas_create_handle(void *context, void **handle) {
  amd4_mock_state *state = (amd4_mock_state *)context;
  if (state->rocblas_create_code != 0) {
    return state->rocblas_create_code;
  }
  if (state->live_handles != 0) {
    return 6;
  }
  state->handle_token = state->active_ordinal + 1;
  state->live_handles = 1;
  *handle = &state->handle_token;
  return 0;
}

static int32_t mock_rocblas_destroy_handle(void *context, void *handle) {
  amd4_mock_state *state = (amd4_mock_state *)context;
  if (handle != &state->handle_token || state->live_handles != 1) {
    return 6;
  }
  state->live_handles = 0;
  return 0;
}

static const et_kernel_provider_v1 *resolve_amd4_provider(void *context,
                                                          const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_amd4_kernel_provider_v1()
             : NULL;
}

static const char *capability_name(int32_t capability_id) {
  if (capability_id == ET_AMD4_AOT_CAP_GEMM) {
    return "kernel.amd-hip.gemm";
  }
  if (capability_id == ET_AMD4_AOT_CAP_ATTENTION) {
    return "kernel.amd-hip.attention";
  }
  return NULL;
}

static const char *capability_operation(int32_t capability_id) {
  if (capability_id == ET_AMD4_AOT_CAP_GEMM) {
    return "gemm";
  }
  if (capability_id == ET_AMD4_AOT_CAP_ATTENTION) {
    return "attention";
  }
  return NULL;
}

int64_t et_amd4_aot_report_size(void) {
  return (int64_t)sizeof(et_amd4_report_v1);
}

int32_t et_amd4_aot_probe_live(int32_t ordinal, et_amd4_report_v1 *report) {
  return (int32_t)et_amd4_probe_v1(ordinal, report);
}

int32_t et_amd4_aot_probe_scenario(int32_t scenario_id, int32_t ordinal,
                                   et_amd4_report_v1 *report) {
  amd4_mock_state state;
  memset(&state, 0, sizeof(state));
  state.scenario_id = scenario_id;

  et_amd4_ops_v1 ops;
  memset(&ops, 0, sizeof(ops));
  ops.struct_size = sizeof(ops);
  ops.context = &state;
  ops.kfd = ET_AMD4_KFD_ACCESSIBLE;
  ops.hip_library = ET_AMD4_LIBRARY_READY;
  ops.rocblas_library = ET_AMD4_LIBRARY_READY;
  ops.hip_runtime_version = mock_hip_runtime_version;
  ops.hip_driver_version = mock_hip_driver_version;
  ops.hip_device_count = mock_hip_device_count;
  ops.hip_device_properties = mock_hip_device_properties;
  ops.hip_get_device = mock_hip_get_device;
  ops.hip_set_device = mock_hip_set_device;
  ops.hip_memory_info = mock_hip_memory_info;
  ops.rocblas_version = mock_rocblas_version;
  ops.rocblas_create_handle = mock_rocblas_create_handle;
  ops.rocblas_destroy_handle = mock_rocblas_destroy_handle;

  switch (scenario_id) {
  case ET_AMD4_AOT_SCENARIO_TWO_DEVICE:
    state.device_count = 2;
    break;
  case ET_AMD4_AOT_SCENARIO_OOM:
    state.device_count = 1;
    state.memory_code = 2;
    break;
  case ET_AMD4_AOT_SCENARIO_DRIVER_ERROR:
    state.device_count = 1;
    state.driver_code = 35;
    state.rocblas_create_code = 6;
    break;
  case ET_AMD4_AOT_SCENARIO_ROCBLAS_MISSING_SYMBOLS:
    state.device_count = 1;
    ops.rocblas_library = ET_AMD4_LIBRARY_MISSING_SYMBOLS;
    ops.rocblas_version = NULL;
    ops.rocblas_create_handle = NULL;
    ops.rocblas_destroy_handle = NULL;
    break;
  default:
    return (int32_t)ET_AMD4_ERR_INVALID_ARG;
  }

  return (int32_t)et_amd4_probe_with_ops_v1(&ops, ordinal, report);
}

int32_t et_amd4_aot_report_status(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return (int32_t)ET_AMD4_ERR_INVALID_ARG;
  }
  return (int32_t)report->status;
}

int32_t et_amd4_aot_report_source(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int32_t)report->source;
}

int32_t et_amd4_aot_report_kfd(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int32_t)report->kfd;
}

int32_t et_amd4_aot_report_hip_library(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int32_t)report->hip_library;
}

int32_t et_amd4_aot_report_rocblas_library(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int32_t)report->rocblas_library;
}

int64_t et_amd4_aot_report_device_count(const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int64_t)report->device_count;
}

int64_t et_amd4_aot_report_reported_device_count(
    const et_amd4_report_v1 *report) {
  if (report == NULL) {
    return -1;
  }
  return (int64_t)report->reported_device_count;
}

static const et_amd4_device_info_v1 *device_at(const et_amd4_report_v1 *report,
                                               int32_t index) {
  if (report == NULL || index < 0 ||
      report->reported_device_count > ET_AMD4_MAX_DEVICES ||
      (uint32_t)index >= report->reported_device_count) {
    return NULL;
  }
  return &report->devices[index];
}

int32_t et_amd4_aot_report_device_compute_units(const et_amd4_report_v1 *report,
                                                int32_t index) {
  const et_amd4_device_info_v1 *device = device_at(report, index);
  if (device == NULL || device->properties_result.status != ET_AMD4_OK) {
    return -1;
  }
  return device->properties.compute_units;
}

int32_t et_amd4_aot_report_device_wavefront_size(
    const et_amd4_report_v1 *report, int32_t index) {
  const et_amd4_device_info_v1 *device = device_at(report, index);
  if (device == NULL || device->properties_result.status != ET_AMD4_OK) {
    return -1;
  }
  return device->properties.wavefront_size;
}

int64_t et_amd4_aot_report_device_total_vram(const et_amd4_report_v1 *report,
                                             int32_t index) {
  const et_amd4_device_info_v1 *device = device_at(report, index);
  if (device == NULL || device->properties_result.status != ET_AMD4_OK ||
      device->properties.total_vram_bytes > (uint64_t)INT64_MAX) {
    return -1;
  }
  return (int64_t)device->properties.total_vram_bytes;
}

int64_t et_amd4_aot_report_device_free_vram(const et_amd4_report_v1 *report,
                                            int32_t index) {
  const et_amd4_device_info_v1 *device = device_at(report, index);
  if (device == NULL || device->memory_result.status != ET_AMD4_OK ||
      device->free_vram_bytes > (uint64_t)INT64_MAX) {
    return -1;
  }
  return (int64_t)device->free_vram_bytes;
}

int32_t et_amd4_aot_format_report(const et_amd4_report_v1 *report, char *buffer,
                                  int64_t capacity,
                                  int64_t *required_capacity) {
  if (report == NULL || required_capacity == NULL || capacity < 0 ||
      ((uintptr_t)required_capacity % _Alignof(int64_t)) != 0u ||
      (buffer == NULL && capacity != 0) ||
      span_overlaps(report, sizeof(*report), required_capacity,
                    sizeof(*required_capacity)) ||
      (buffer != NULL && capacity > 0 &&
       span_overlaps(buffer, (size_t)capacity, required_capacity,
                     sizeof(*required_capacity)))) {
    return (int32_t)ET_AMD4_ERR_INVALID_ARG;
  }
  size_t required = 0u;
  const et_amd4_status status =
      et_amd4_format_report_v1(report, buffer, (size_t)capacity, &required);
  if (status == ET_AMD4_OK || status == ET_AMD4_ERR_OUT_OF_RANGE) {
    if (required > (size_t)INT64_MAX) {
      return (int32_t)ET_AMD4_ERR_OUT_OF_RANGE;
    }
    *required_capacity = (int64_t)required;
  }
  return (int32_t)status;
}

int32_t et_amd4_aot_read_byte(const char *buffer, int64_t index) {
  if (buffer == NULL || index < 0) {
    return -1;
  }
  return (int32_t)(unsigned char)buffer[index];
}

int32_t et_amd4_aot_k1_open(et_kernel_runtime **runtime) {
  et_kernel_error error;
  return et_kernel_runtime_discover(resolve_amd4_provider, NULL, runtime,
                                    &error);
}

void et_amd4_aot_k1_close(et_kernel_runtime *runtime) {
  et_kernel_runtime_destroy(runtime);
}

int64_t et_amd4_aot_k1_capability_count(const et_kernel_runtime *runtime) {
  if (runtime == NULL) {
    return -1;
  }
  return (int64_t)et_kernel_runtime_capability_count(runtime);
}

int32_t et_amd4_aot_k1_capability_status(const et_kernel_runtime *runtime,
                                         int32_t capability_id) {
  const char *name = capability_name(capability_id);
  if (runtime == NULL || name == NULL) {
    return -1;
  }
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, name);
  if (cap == NULL) {
    return -1;
  }
  return (int32_t)cap->status;
}

int32_t et_amd4_aot_k1_dispatch(const et_kernel_runtime *runtime,
                                int32_t capability_id, const uint32_t *input,
                                uint32_t *output, int32_t *out_error_code) {
  if (out_error_code != NULL &&
      (((uintptr_t)out_error_code % _Alignof(int32_t)) != 0u ||
       span_overlaps(out_error_code, sizeof(*out_error_code), input,
                     sizeof(*input)) ||
       span_overlaps(out_error_code, sizeof(*out_error_code), output,
                     sizeof(*output)))) {
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  }
  const char *name = capability_name(capability_id);
  const char *operation = capability_operation(capability_id);
  if (name == NULL || operation == NULL) {
    if (out_error_code != NULL) {
      *out_error_code = (int32_t)ET_KERNEL_CODE_INVALID_ENUM;
    }
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  }
  if (runtime == NULL || input == NULL || output == NULL) {
    if (out_error_code != NULL) {
      *out_error_code = (int32_t)ET_KERNEL_CODE_NULL_ARGUMENT;
    }
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  }
  const uint64_t shape[2] = {1u, 1u};
  /* K1 tensor views carry a non-const data pointer; inputs are never written
   * and AMD4 rejects every call before invocation. */
  void *input_ptr = NULL;
  memcpy(&input_ptr, &input, sizeof(input_ptr));
  const et_kernel_tensor_view_v1 in_view = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = input_ptr,
      .byte_length = sizeof(*input),
      .dtype = "f32",
      .device = "amd",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = 2u,
      .shape = shape,
  };
  et_kernel_tensor_view_v1 out_view = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = output,
      .byte_length = sizeof(*output),
      .dtype = "f32",
      .device = "amd",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = 2u,
      .shape = shape,
  };
  const et_kernel_request_v1 request = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = operation,
      .dtype = "f32",
      .device = "amd",
      .rank = 2u,
      .shape = shape,
      .deterministic = 0u,
      .reserved = {0},
  };
  const et_kernel_call_v1 call = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = name,
      .request = &request,
      .input_count = 1u,
      .input_stride = sizeof(in_view),
      .input_bytes = sizeof(in_view),
      .inputs = &in_view,
      .output_count = 1u,
      .output_stride = sizeof(out_view),
      .output_bytes = sizeof(out_view),
      .outputs = &out_view,
  };
  et_kernel_error error;
  const int32_t category = et_kernel_runtime_dispatch(runtime, &call, &error);
  if (out_error_code != NULL) {
    *out_error_code = (int32_t)error.code;
  }
  return category;
}

int32_t et_amd4_aot_k1_report_json(const et_kernel_runtime *runtime,
                                   char *buffer, int64_t capacity,
                                   int64_t *required_capacity) {
  if (runtime == NULL || required_capacity == NULL || capacity < 0 ||
      ((uintptr_t)required_capacity % _Alignof(int64_t)) != 0u ||
      (buffer == NULL && capacity != 0) ||
      (buffer != NULL && capacity > 0 &&
       span_overlaps(buffer, (size_t)capacity, required_capacity,
                     sizeof(*required_capacity)))) {
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  }
  size_t required = 0u;
  et_kernel_error error;
  const int32_t status = et_kernel_runtime_report_json(
      runtime, buffer, (size_t)capacity, &required, &error);
  if (status == ET_KERNEL_ERROR_NONE ||
      error.code == ET_KERNEL_CODE_BUFFER_TOO_SMALL) {
    if (required > (size_t)INT64_MAX) {
      return ET_KERNEL_ERROR_INVALID_ARGUMENT;
    }
    *required_capacity = (int64_t)required;
  }
  return status;
}

int32_t et_amd4_aot_store_u32(uint32_t *buffer, int64_t index, int64_t value) {
  if (buffer == NULL || index < 0 || value < 0 || value > (int64_t)UINT32_MAX) {
    return (int32_t)ET_AMD4_ERR_INVALID_ARG;
  }
  buffer[index] = (uint32_t)value;
  return (int32_t)ET_AMD4_OK;
}

int64_t et_amd4_aot_load_u32(const uint32_t *buffer, int64_t index) {
  if (buffer == NULL || index < 0) {
    return -1;
  }
  return (int64_t)buffer[index];
}
