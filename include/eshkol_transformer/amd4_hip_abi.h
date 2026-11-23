#ifndef ESHKOL_TRANSFORMER_AMD4_HIP_ABI_H
#define ESHKOL_TRANSFORMER_AMD4_HIP_ABI_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_AMD4_ABI_MAJOR 1u
#define ET_AMD4_ABI_MINOR 0u
#define ET_AMD4_ALL_DEVICES (-1)
#define ET_AMD4_MAX_DEVICES 64u
#define ET_AMD4_TEXT_CAPACITY 256u

/* K1 ABI 1.0 spells its explicit unavailable state "unsupported". */
#define ET_KERNEL_CAPABILITY_UNAVAILABLE ET_KERNEL_CAPABILITY_UNSUPPORTED

typedef uint32_t et_amd4_status;
enum {
  ET_AMD4_OK = 0,
  ET_AMD4_ERR_INVALID_ARG,
  ET_AMD4_ERR_UNAVAILABLE_RUNTIME,
  ET_AMD4_ERR_NO_DEVICE,
  ET_AMD4_ERR_DEVICE_UNSUPPORTED,
  ET_AMD4_ERR_ROCBLAS_UNAVAILABLE,
  ET_AMD4_ERR_DRIVER_FAILURE,
  ET_AMD4_ERR_OUT_OF_RANGE,
  ET_AMD4_ERR_ALLOC,
  ET_AMD4_NOT_RUN
};

typedef uint32_t et_amd4_kfd_state;
enum {
  ET_AMD4_KFD_ABSENT = 0,
  ET_AMD4_KFD_INACCESSIBLE,
  ET_AMD4_KFD_ACCESSIBLE
};

typedef uint32_t et_amd4_library_state;
enum {
  ET_AMD4_LIBRARY_UNAVAILABLE = 0,
  ET_AMD4_LIBRARY_MISSING_SYMBOLS,
  ET_AMD4_LIBRARY_READY
};

typedef uint32_t et_amd4_source;
enum { ET_AMD4_SOURCE_LIVE = 0, ET_AMD4_SOURCE_INJECTED };

typedef struct et_amd4_result_v1 {
  et_amd4_status status;
  int32_t native_code;
} et_amd4_result_v1;

typedef struct et_amd4_device_properties_v1 {
  char name[ET_AMD4_TEXT_CAPACITY];
  char gcn_arch_name[ET_AMD4_TEXT_CAPACITY];
  int32_t pci_domain;
  int32_t pci_bus;
  int32_t pci_device;
  int32_t compute_units;
  int32_t wavefront_size;
  uint64_t total_vram_bytes;
} et_amd4_device_properties_v1;

typedef struct et_amd4_device_info_v1 {
  int32_t ordinal;
  et_amd4_result_v1 properties_result;
  et_amd4_result_v1 selection_result;
  et_amd4_result_v1 memory_result;
  et_amd4_result_v1 rocblas_create_result;
  et_amd4_result_v1 rocblas_destroy_result;
  et_amd4_device_properties_v1 properties;
  uint64_t free_vram_bytes;
  uint64_t memory_query_total_bytes;
} et_amd4_device_info_v1;

typedef struct et_amd4_report_v1 {
  uint32_t abi_major;
  uint32_t abi_minor;
  et_amd4_source source;
  et_amd4_status status;
  et_amd4_kfd_state kfd;
  et_amd4_library_state hip_library;
  et_amd4_library_state rocblas_library;
  et_amd4_result_v1 hip_runtime_version_result;
  et_amd4_result_v1 hip_driver_version_result;
  et_amd4_result_v1 device_count_result;
  et_amd4_result_v1 rocblas_version_result;
  et_amd4_result_v1 current_device_result;
  et_amd4_result_v1 restore_device_result;
  int32_t hip_runtime_version;
  int32_t hip_driver_version;
  int32_t requested_ordinal;
  int32_t original_ordinal;
  uint32_t device_count;
  uint32_t reported_device_count;
  char rocblas_version[ET_AMD4_TEXT_CAPACITY];
  et_amd4_device_info_v1 devices[ET_AMD4_MAX_DEVICES];
} et_amd4_report_v1;

/* Callbacks return raw HIP or rocBLAS status codes and write outputs on
   success. Supply every callback for each READY library. A successfully created
   handle must be non-null and remains owned by the probe until destroy is
   called. */
typedef struct et_amd4_ops_v1 {
  size_t struct_size;
  void *context;
  et_amd4_kfd_state kfd;
  et_amd4_library_state hip_library;
  et_amd4_library_state rocblas_library;
  int32_t (*hip_runtime_version)(void *context, int32_t *version);
  int32_t (*hip_driver_version)(void *context, int32_t *version);
  int32_t (*hip_device_count)(void *context, int32_t *count);
  int32_t (*hip_device_properties)(void *context, int32_t ordinal,
                                   et_amd4_device_properties_v1 *properties);
  int32_t (*hip_get_device)(void *context, int32_t *ordinal);
  int32_t (*hip_set_device)(void *context, int32_t ordinal);
  int32_t (*hip_memory_info)(void *context, uint64_t *free_bytes,
                             uint64_t *total_bytes);
  int32_t (*rocblas_version)(void *context, char *version, size_t capacity);
  int32_t (*rocblas_create_handle)(void *context, void **handle);
  int32_t (*rocblas_destroy_handle)(void *context, void *handle);
} et_amd4_ops_v1;

#ifdef __cplusplus
#define ET_AMD4_STATIC_ASSERT(condition, message)                              \
  static_assert(condition, message)
#else
#define ET_AMD4_STATIC_ASSERT(condition, message)                              \
  _Static_assert(condition, message)
#endif
ET_AMD4_STATIC_ASSERT(sizeof(et_amd4_result_v1) == 8 &&
                          sizeof(et_amd4_device_properties_v1) == 544 &&
                          sizeof(et_amd4_device_info_v1) == 608 &&
                          offsetof(et_amd4_device_info_v1, properties) == 48,
                      "AMD4 v1 device layout changed");
ET_AMD4_STATIC_ASSERT(sizeof(et_amd4_report_v1) == 39272 &&
                          offsetof(et_amd4_report_v1, devices) == 360,
                      "AMD4 v1 report layout changed");
ET_AMD4_STATIC_ASSERT(sizeof(et_amd4_ops_v1) == 112 &&
                          offsetof(et_amd4_ops_v1, hip_runtime_version) == 32,
                      "AMD4 v1 ops layout changed");
#undef ET_AMD4_STATIC_ASSERT

/* Probe all visible ordinals or one explicit ordinal. Valid arguments always
   receive a report, including on probe failure. Storage must not overlap. */
et_amd4_status et_amd4_probe_v1(int32_t ordinal, et_amd4_report_v1 *report);
et_amd4_status et_amd4_probe_with_ops_v1(const et_amd4_ops_v1 *ops,
                                         int32_t ordinal,
                                         et_amd4_report_v1 *report);

/* NULL buffer with zero capacity measures the report. Required capacity
   includes the NUL. A short buffer is left intact and returns OUT_OF_RANGE.
   Report, buffer and required_capacity storage must not overlap. */
et_amd4_status et_amd4_format_report_v1(const et_amd4_report_v1 *report,
                                        char *buffer, size_t capacity,
                                        size_t *required_capacity);

const et_kernel_provider_v1 *et_amd4_kernel_provider_v1(void);

#ifdef __cplusplus
}
#endif

#endif
