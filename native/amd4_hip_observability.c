#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/amd4_hip_abi.h"

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

enum {
  HIP_ERROR_OUT_OF_MEMORY = 2,
  HIP_ERROR_NO_DEVICE = 100,
  HIP_ERROR_INVALID_DEVICE = 101,
  HIP_ERROR_NOT_SUPPORTED = 801,
  ROCBLAS_STATUS_MEMORY_ERROR = 5
};

static const et_amd4_result_v1 not_run = {ET_AMD4_NOT_RUN, 0};

static int valid_storage(const void *ptr, size_t size, size_t alignment) {
  const uintptr_t address = (uintptr_t)ptr;
  return ptr != NULL && address % alignment == 0 &&
         size <= UINTPTR_MAX - address;
}

static int overlaps(const void *a, size_t a_size, const void *b,
                    size_t b_size) {
  const uintptr_t a_start = (uintptr_t)a;
  const uintptr_t b_start = (uintptr_t)b;
  return a_start < b_start + b_size && b_start < a_start + a_size;
}

static int valid_target(int32_t ordinal, const et_amd4_report_v1 *report) {
  return ordinal >= ET_AMD4_ALL_DEVICES &&
         valid_storage(report, sizeof(*report), _Alignof(et_amd4_report_v1));
}

static int valid_ops(const et_amd4_ops_v1 *ops) {
  return ops->struct_size == sizeof(*ops) &&
         ops->kfd <= ET_AMD4_KFD_ACCESSIBLE &&
         ops->hip_library <= ET_AMD4_LIBRARY_READY &&
         ops->rocblas_library <= ET_AMD4_LIBRARY_READY &&
         (ops->hip_library != ET_AMD4_LIBRARY_READY ||
          (ops->hip_runtime_version && ops->hip_driver_version &&
           ops->hip_device_count && ops->hip_device_properties &&
           ops->hip_get_device && ops->hip_set_device &&
           ops->hip_memory_info)) &&
         (ops->rocblas_library != ET_AMD4_LIBRARY_READY ||
          (ops->rocblas_version && ops->rocblas_create_handle &&
           ops->rocblas_destroy_handle));
}

static et_amd4_result_v1 hip_result(int32_t code) {
  et_amd4_status status;
  switch (code) {
  case 0:
    status = ET_AMD4_OK;
    break;
  case HIP_ERROR_OUT_OF_MEMORY:
    status = ET_AMD4_ERR_ALLOC;
    break;
  case HIP_ERROR_NO_DEVICE:
    status = ET_AMD4_ERR_NO_DEVICE;
    break;
  case HIP_ERROR_INVALID_DEVICE:
    status = ET_AMD4_ERR_OUT_OF_RANGE;
    break;
  case HIP_ERROR_NOT_SUPPORTED:
    status = ET_AMD4_ERR_DEVICE_UNSUPPORTED;
    break;
  default:
    status = ET_AMD4_ERR_DRIVER_FAILURE;
    break;
  }
  return (et_amd4_result_v1){status, code};
}

static et_amd4_result_v1 rocblas_result(int32_t code) {
  if (code == 0) {
    return (et_amd4_result_v1){ET_AMD4_OK, code};
  }
  return (et_amd4_result_v1){code == ROCBLAS_STATUS_MEMORY_ERROR
                                 ? ET_AMD4_ERR_ALLOC
                                 : ET_AMD4_ERR_ROCBLAS_UNAVAILABLE,
                             code};
}

static void record_first_error(et_amd4_status *status,
                               et_amd4_result_v1 result) {
  if (*status == ET_AMD4_OK && result.status != ET_AMD4_NOT_RUN) {
    *status = result.status;
  }
}

static int nonempty_text(const char *text) {
  return text[0] != '\0' && memchr(text, '\0', ET_AMD4_TEXT_CAPACITY) != NULL;
}

static et_amd4_status probe_rocblas_version(const et_amd4_ops_v1 *ops,
                                            et_amd4_report_v1 *report) {
  if (ops->rocblas_library != ET_AMD4_LIBRARY_READY) {
    return ET_AMD4_ERR_ROCBLAS_UNAVAILABLE;
  }
  char version[ET_AMD4_TEXT_CAPACITY] = {0};
  et_amd4_result_v1 result = rocblas_result(
      ops->rocblas_version(ops->context, version, sizeof(version)));
  if (result.status == ET_AMD4_OK) {
    if (nonempty_text(version)) {
      memcpy(report->rocblas_version, version, sizeof(version));
    } else {
      result.status = ET_AMD4_ERR_ROCBLAS_UNAVAILABLE;
    }
  }
  report->rocblas_version_result = result;
  return result.status;
}

static et_amd4_result_v1 probe_properties(const et_amd4_ops_v1 *ops,
                                          et_amd4_device_info_v1 *device) {
  et_amd4_device_properties_v1 props = {0};
  et_amd4_result_v1 result = hip_result(
      ops->hip_device_properties(ops->context, device->ordinal, &props));
  if (result.status != ET_AMD4_OK) {
    return result;
  }
  if (!nonempty_text(props.gcn_arch_name) ||
      strncmp(props.gcn_arch_name, "gfx", 3) != 0 ||
      props.gcn_arch_name[3] == '\0') {
    return (et_amd4_result_v1){ET_AMD4_ERR_DEVICE_UNSUPPORTED, 0};
  }
  if (!nonempty_text(props.name) || props.pci_domain < 0 || props.pci_bus < 0 ||
      props.pci_device < 0 || props.compute_units <= 0 ||
      props.wavefront_size <= 0 || props.total_vram_bytes == 0) {
    return (et_amd4_result_v1){ET_AMD4_ERR_DRIVER_FAILURE, 0};
  }
  device->properties = props;
  return result;
}

static void probe_memory(const et_amd4_ops_v1 *ops,
                         et_amd4_device_info_v1 *device) {
  uint64_t free_bytes = 0;
  uint64_t total_bytes = 0;
  device->memory_result =
      hip_result(ops->hip_memory_info(ops->context, &free_bytes, &total_bytes));
  if (device->memory_result.status == ET_AMD4_OK) {
    if (free_bytes > total_bytes || total_bytes == 0) {
      device->memory_result.status = ET_AMD4_ERR_DRIVER_FAILURE;
      return;
    }
    device->free_vram_bytes = free_bytes;
    device->memory_query_total_bytes = total_bytes;
  }
}

static void probe_rocblas_handle(const et_amd4_ops_v1 *ops,
                                 et_amd4_device_info_v1 *device) {
  void *handle = NULL;
  device->rocblas_create_result =
      rocblas_result(ops->rocblas_create_handle(ops->context, &handle));
  if (device->rocblas_create_result.status != ET_AMD4_OK) {
    return;
  }
  if (handle == NULL) {
    device->rocblas_create_result.status = ET_AMD4_ERR_ROCBLAS_UNAVAILABLE;
    return;
  }
  device->rocblas_destroy_result =
      rocblas_result(ops->rocblas_destroy_handle(ops->context, handle));
}

static et_amd4_result_v1 probe_version(void *context,
                                       int32_t (*query)(void *, int32_t *),
                                       int32_t *output) {
  int32_t version = 0;
  et_amd4_result_v1 result = hip_result(query(context, &version));
  if (result.status == ET_AMD4_OK) {
    if (version <= 0) {
      result.status = ET_AMD4_ERR_DRIVER_FAILURE;
    } else {
      *output = version;
    }
  }
  return result;
}

static et_amd4_status probe_devices(const et_amd4_ops_v1 *ops,
                                    et_amd4_report_v1 *report,
                                    et_amd4_status blas_status) {
  int32_t count = 0;
  report->device_count_result =
      hip_result(ops->hip_device_count(ops->context, &count));
  if (report->device_count_result.status != ET_AMD4_OK) {
    return report->device_count_result.status;
  }
  if (count < 0) {
    report->device_count_result.status = ET_AMD4_ERR_DRIVER_FAILURE;
    return report->device_count_result.status;
  }
  report->device_count = (uint32_t)count;
  if (count == 0) {
    return ET_AMD4_ERR_NO_DEVICE;
  }
  const int32_t requested = report->requested_ordinal;
  if (report->device_count > ET_AMD4_MAX_DEVICES || requested >= count) {
    return ET_AMD4_ERR_OUT_OF_RANGE;
  }

  et_amd4_status hip_status = ET_AMD4_OK;
  record_first_error(&hip_status, report->hip_runtime_version_result);
  record_first_error(&hip_status, report->hip_driver_version_result);
  int32_t original = -1;
  report->current_device_result =
      hip_result(ops->hip_get_device(ops->context, &original));
  if (report->current_device_result.status == ET_AMD4_OK) {
    if (original < 0 || original >= count) {
      report->current_device_result.status = ET_AMD4_ERR_DRIVER_FAILURE;
    } else {
      report->original_ordinal = original;
    }
  }
  record_first_error(&hip_status, report->current_device_result);

  int32_t active = report->original_ordinal;
  int restore_needed = 0;
  const int32_t first = requested == ET_AMD4_ALL_DEVICES ? 0 : requested;
  const int32_t end = requested == ET_AMD4_ALL_DEVICES ? count : requested + 1;
  for (int32_t ordinal = first; ordinal < end; ++ordinal) {
    et_amd4_device_info_v1 *device =
        &report->devices[report->reported_device_count++];
    device->ordinal = ordinal;
    device->selection_result = not_run;
    device->memory_result = not_run;
    device->rocblas_create_result = not_run;
    device->rocblas_destroy_result = not_run;
    device->properties_result = probe_properties(ops, device);
    record_first_error(&hip_status, device->properties_result);
    if (device->properties_result.status != ET_AMD4_OK ||
        report->current_device_result.status != ET_AMD4_OK) {
      continue;
    }

    device->selection_result = hip_result(0);
    if (active != ordinal) {
      restore_needed = 1;
      device->selection_result =
          hip_result(ops->hip_set_device(ops->context, ordinal));
      active = device->selection_result.status == ET_AMD4_OK ? ordinal : -1;
    }
    record_first_error(&hip_status, device->selection_result);
    if (device->selection_result.status != ET_AMD4_OK) {
      continue;
    }
    probe_memory(ops, device);
    record_first_error(&hip_status, device->memory_result);
    if (ops->rocblas_library == ET_AMD4_LIBRARY_READY) {
      probe_rocblas_handle(ops, device);
      record_first_error(&blas_status, device->rocblas_create_result);
      record_first_error(&blas_status, device->rocblas_destroy_result);
    }
  }
  if (restore_needed && active != original) {
    report->restore_device_result =
        hip_result(ops->hip_set_device(ops->context, original));
    record_first_error(&hip_status, report->restore_device_result);
  }
  return hip_status == ET_AMD4_OK ? blas_status : hip_status;
}

static et_amd4_status probe(const et_amd4_ops_v1 *ops, int32_t ordinal,
                            et_amd4_source source, et_amd4_report_v1 *report) {
  memset(report, 0, sizeof(*report));
  report->abi_major = ET_AMD4_ABI_MAJOR;
  report->abi_minor = ET_AMD4_ABI_MINOR;
  report->source = source;
  report->kfd = ops->kfd;
  report->hip_library = ops->hip_library;
  report->rocblas_library = ops->rocblas_library;
  report->requested_ordinal = ordinal;
  report->original_ordinal = -1;
  report->hip_runtime_version_result = not_run;
  report->hip_driver_version_result = not_run;
  report->device_count_result = not_run;
  report->rocblas_version_result = not_run;
  report->current_device_result = not_run;
  report->restore_device_result = not_run;

  const et_amd4_status blas_status = probe_rocblas_version(ops, report);
  if (ops->hip_library != ET_AMD4_LIBRARY_READY) {
    report->status = ET_AMD4_ERR_UNAVAILABLE_RUNTIME;
    return report->status;
  }
  report->hip_runtime_version_result = probe_version(
      ops->context, ops->hip_runtime_version, &report->hip_runtime_version);
  report->hip_driver_version_result = probe_version(
      ops->context, ops->hip_driver_version, &report->hip_driver_version);
  report->status = probe_devices(ops, report, blas_status);
  return report->status;
}

et_amd4_status et_amd4_probe_with_ops_v1(const et_amd4_ops_v1 *ops,
                                         int32_t ordinal,
                                         et_amd4_report_v1 *report) {
  if (!valid_target(ordinal, report) ||
      !valid_storage(ops, sizeof(*ops), _Alignof(et_amd4_ops_v1)) ||
      overlaps(ops, sizeof(*ops), report, sizeof(*report)) || !valid_ops(ops) ||
      ((uintptr_t)ops->context >= (uintptr_t)report &&
       (uintptr_t)ops->context - (uintptr_t)report < sizeof(*report))) {
    return ET_AMD4_ERR_INVALID_ARG;
  }
  return probe(ops, ordinal, ET_AMD4_SOURCE_INJECTED, report);
}

/* LP64 HIP R0600 property ABI. The byte spans omit fields AMD4 never reads.
   Layout checked against AMD's HIP 6.0.0, 6.4.3 and 7.2.0 headers. */
typedef struct hip_properties_r0600 {
  char name[256];
  unsigned char before_total[32];
  size_t total;
  unsigned char before_wavefront[12];
  int32_t wavefront;
  unsigned char before_compute_units[76];
  int32_t compute_units;
  unsigned char before_pci[192];
  int32_t pci_bus;
  int32_t pci_device;
  int32_t pci_domain;
  unsigned char before_arch[564];
  char arch[256];
  unsigned char tail[56];
} hip_properties_r0600;

_Static_assert(sizeof(hip_properties_r0600) == 1472 &&
                   _Alignof(hip_properties_r0600) == 8 &&
                   offsetof(hip_properties_r0600, total) == 288 &&
                   offsetof(hip_properties_r0600, wavefront) == 308 &&
                   offsetof(hip_properties_r0600, compute_units) == 388 &&
                   offsetof(hip_properties_r0600, pci_bus) == 584 &&
                   offsetof(hip_properties_r0600, pci_device) == 588 &&
                   offsetof(hip_properties_r0600, pci_domain) == 592 &&
                   offsetof(hip_properties_r0600, arch) == 1160,
               "unexpected HIP R0600 property layout");

typedef struct _rocblas_handle *rocblas_handle;
typedef struct live_api {
  int32_t (*runtime_version)(int32_t *);
  int32_t (*driver_version)(int32_t *);
  int32_t (*device_count)(int32_t *);
  int32_t (*properties)(hip_properties_r0600 *, int32_t);
  int32_t (*get_device)(int32_t *);
  int32_t (*set_device)(int32_t);
  int32_t (*memory_info)(size_t *, size_t *);
  int32_t (*blas_version)(char *, size_t);
  int32_t (*blas_create)(rocblas_handle *);
  int32_t (*blas_destroy)(rocblas_handle);
} live_api;

static int32_t live_runtime_version(void *context, int32_t *version) {
  return ((live_api *)context)->runtime_version(version);
}

static int32_t live_driver_version(void *context, int32_t *version) {
  return ((live_api *)context)->driver_version(version);
}

static int32_t live_device_count(void *context, int32_t *count) {
  return ((live_api *)context)->device_count(count);
}

static int32_t live_device_properties(void *context, int32_t ordinal,
                                      et_amd4_device_properties_v1 *out) {
  hip_properties_r0600 props = {0};
  const int32_t code = ((live_api *)context)->properties(&props, ordinal);
  if (code == 0) {
    memcpy(out->name, props.name, sizeof(props.name));
    memcpy(out->gcn_arch_name, props.arch, sizeof(props.arch));
    out->pci_domain = props.pci_domain;
    out->pci_bus = props.pci_bus;
    out->pci_device = props.pci_device;
    out->compute_units = props.compute_units;
    out->wavefront_size = props.wavefront;
    out->total_vram_bytes = props.total;
  }
  return code;
}

static int32_t live_get_device(void *context, int32_t *ordinal) {
  return ((live_api *)context)->get_device(ordinal);
}

static int32_t live_set_device(void *context, int32_t ordinal) {
  return ((live_api *)context)->set_device(ordinal);
}

static int32_t live_memory_info(void *context, uint64_t *free_bytes,
                                uint64_t *total_bytes) {
  size_t free_size = 0;
  size_t total_size = 0;
  const int32_t code =
      ((live_api *)context)->memory_info(&free_size, &total_size);
  if (code == 0) {
    *free_bytes = free_size;
    *total_bytes = total_size;
  }
  return code;
}

static int32_t live_blas_version(void *context, char *version,
                                 size_t capacity) {
  return ((live_api *)context)->blas_version(version, capacity);
}

static int32_t live_blas_create(void *context, void **out) {
  rocblas_handle handle = NULL;
  const int32_t code = ((live_api *)context)->blas_create(&handle);
  if (code == 0) {
    *out = handle;
  }
  return code;
}

static int32_t live_blas_destroy(void *context, void *handle) {
  return ((live_api *)context)->blas_destroy(handle);
}

/* POSIX permits dlsym function addresses. Copy the representation to avoid
   an object-pointer-to-function-pointer cast in strict C11. */
#define LOAD_SYMBOL(library, field, name)                                      \
  do {                                                                         \
    void *symbol = dlsym((library), (name));                                   \
    _Static_assert(sizeof(field) == sizeof(symbol), "unexpected dlsym ABI");   \
    if (symbol == NULL) {                                                      \
      return ET_AMD4_LIBRARY_MISSING_SYMBOLS;                                  \
    }                                                                          \
    memcpy(&(field), &symbol, sizeof(field));                                  \
  } while (0)

static et_amd4_library_state load_hip(void *library, live_api *api) {
  if (library == NULL) {
    return ET_AMD4_LIBRARY_UNAVAILABLE;
  }
  LOAD_SYMBOL(library, api->runtime_version, "hipRuntimeGetVersion");
  LOAD_SYMBOL(library, api->driver_version, "hipDriverGetVersion");
  LOAD_SYMBOL(library, api->device_count, "hipGetDeviceCount");
  LOAD_SYMBOL(library, api->properties, "hipGetDevicePropertiesR0600");
  LOAD_SYMBOL(library, api->get_device, "hipGetDevice");
  LOAD_SYMBOL(library, api->set_device, "hipSetDevice");
  LOAD_SYMBOL(library, api->memory_info, "hipMemGetInfo");
  return ET_AMD4_LIBRARY_READY;
}

static et_amd4_library_state load_rocblas(void *library, live_api *api) {
  if (library == NULL) {
    return ET_AMD4_LIBRARY_UNAVAILABLE;
  }
  LOAD_SYMBOL(library, api->blas_version, "rocblas_get_version_string");
  LOAD_SYMBOL(library, api->blas_create, "rocblas_create_handle");
  LOAD_SYMBOL(library, api->blas_destroy, "rocblas_destroy_handle");
  return ET_AMD4_LIBRARY_READY;
}
#undef LOAD_SYMBOL

static et_amd4_kfd_state kfd_state(void) {
  if (access("/dev/kfd", R_OK | W_OK) == 0) {
    return ET_AMD4_KFD_ACCESSIBLE;
  }
  return errno == ENOENT || errno == ENOTDIR ? ET_AMD4_KFD_ABSENT
                                             : ET_AMD4_KFD_INACCESSIBLE;
}

et_amd4_status et_amd4_probe_v1(int32_t ordinal, et_amd4_report_v1 *report) {
  if (!valid_target(ordinal, report)) {
    return ET_AMD4_ERR_INVALID_ARG;
  }
  const et_amd4_kfd_state kfd = kfd_state();
  void *hip = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  void *rocblas = dlopen("librocblas.so", RTLD_LAZY | RTLD_LOCAL);
  live_api api = {0};
  const et_amd4_ops_v1 ops = {
      .struct_size = sizeof(ops),
      .context = &api,
      .kfd = kfd,
      .hip_library = load_hip(hip, &api),
      .rocblas_library = load_rocblas(rocblas, &api),
      .hip_runtime_version = live_runtime_version,
      .hip_driver_version = live_driver_version,
      .hip_device_count = live_device_count,
      .hip_device_properties = live_device_properties,
      .hip_get_device = live_get_device,
      .hip_set_device = live_set_device,
      .hip_memory_info = live_memory_info,
      .rocblas_version = live_blas_version,
      .rocblas_create_handle = live_blas_create,
      .rocblas_destroy_handle = live_blas_destroy,
  };
  const et_amd4_status status =
      probe(&ops, ordinal, ET_AMD4_SOURCE_LIVE, report);
  if (rocblas != NULL) {
    dlclose(rocblas);
  }
  if (hip != NULL) {
    dlclose(hip);
  }
  return status;
}

typedef struct report_writer {
  char *buffer;
  size_t capacity;
  size_t length;
  int failed;
} report_writer;

static void append(report_writer *writer, const char *format, ...) {
  if (writer->failed) {
    return;
  }
  char *target = NULL;
  size_t remaining = 0;
  if (writer->buffer != NULL) {
    target = writer->buffer + writer->length;
    remaining = writer->capacity - writer->length;
  }
  va_list args;
  va_start(args, format);
  const int count = vsnprintf(target, remaining, format, args);
  va_end(args);
  if (count < 0 || (writer->buffer != NULL && (size_t)count >= remaining)) {
    writer->failed = 1;
  } else {
    writer->length += (size_t)count;
  }
}

static void append_text(report_writer *writer, const char *key,
                        const char *text) {
  append(writer, "%s=\"", key);
  for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
    if (*p < 0x20u || *p > 0x7eu || *p == '"' || *p == '\\') {
      append(writer, "\\x%02x", (unsigned)*p);
    } else {
      append(writer, "%c", (int)*p);
    }
  }
  append(writer, "\"\n");
}

static const char *const status_names[] = {
    [ET_AMD4_OK] = "ok",
    [ET_AMD4_ERR_INVALID_ARG] = "invalid-arg",
    [ET_AMD4_ERR_UNAVAILABLE_RUNTIME] = "unavailable-runtime",
    [ET_AMD4_ERR_NO_DEVICE] = "no-device",
    [ET_AMD4_ERR_DEVICE_UNSUPPORTED] = "device-unsupported",
    [ET_AMD4_ERR_ROCBLAS_UNAVAILABLE] = "rocblas-unavailable",
    [ET_AMD4_ERR_DRIVER_FAILURE] = "driver-failure",
    [ET_AMD4_ERR_OUT_OF_RANGE] = "out-of-range",
    [ET_AMD4_ERR_ALLOC] = "alloc",
    [ET_AMD4_NOT_RUN] = "not-run",
};

static int valid_device_report(const et_amd4_device_info_v1 *device) {
  return device->properties_result.status <= ET_AMD4_NOT_RUN &&
         device->selection_result.status <= ET_AMD4_NOT_RUN &&
         device->memory_result.status <= ET_AMD4_NOT_RUN &&
         device->rocblas_create_result.status <= ET_AMD4_NOT_RUN &&
         device->rocblas_destroy_result.status <= ET_AMD4_NOT_RUN &&
         (device->properties_result.status != ET_AMD4_OK ||
          (nonempty_text(device->properties.name) &&
           nonempty_text(device->properties.gcn_arch_name)));
}

static int valid_report(const et_amd4_report_v1 *report) {
  if (report->abi_major != ET_AMD4_ABI_MAJOR ||
      report->abi_minor != ET_AMD4_ABI_MINOR ||
      report->source > ET_AMD4_SOURCE_INJECTED ||
      report->status >= ET_AMD4_NOT_RUN ||
      report->kfd > ET_AMD4_KFD_ACCESSIBLE ||
      report->hip_library > ET_AMD4_LIBRARY_READY ||
      report->rocblas_library > ET_AMD4_LIBRARY_READY ||
      report->hip_runtime_version_result.status > ET_AMD4_NOT_RUN ||
      report->hip_driver_version_result.status > ET_AMD4_NOT_RUN ||
      report->device_count_result.status > ET_AMD4_NOT_RUN ||
      report->rocblas_version_result.status > ET_AMD4_NOT_RUN ||
      report->current_device_result.status > ET_AMD4_NOT_RUN ||
      report->restore_device_result.status > ET_AMD4_NOT_RUN ||
      report->requested_ordinal < ET_AMD4_ALL_DEVICES ||
      report->reported_device_count > ET_AMD4_MAX_DEVICES ||
      report->reported_device_count > report->device_count ||
      (report->rocblas_version_result.status == ET_AMD4_OK &&
       !nonempty_text(report->rocblas_version))) {
    return 0;
  }
  for (uint32_t i = 0; i < report->reported_device_count; ++i) {
    const et_amd4_device_info_v1 *device = &report->devices[i];
    const int32_t ordinal = report->requested_ordinal == ET_AMD4_ALL_DEVICES
                                ? (int32_t)i
                                : report->requested_ordinal;
    if (device->ordinal != ordinal ||
        (uint32_t)ordinal >= report->device_count ||
        (report->requested_ordinal != ET_AMD4_ALL_DEVICES && i != 0) ||
        !valid_device_report(device)) {
      return 0;
    }
  }
  return 1;
}

static void append_result(report_writer *writer, const char *key,
                          et_amd4_result_v1 result) {
  append(writer, "%s=%s,%" PRId32 "\n", key, status_names[result.status],
         result.native_code);
}

static void format_device(report_writer *writer,
                          const et_amd4_device_info_v1 *device) {
  append(writer, "[device.%" PRId32 "]\n", device->ordinal);
  append_result(writer, "properties", device->properties_result);
  append_result(writer, "selection", device->selection_result);
  append_result(writer, "memory", device->memory_result);
  append_result(writer, "rocblas.create", device->rocblas_create_result);
  append_result(writer, "rocblas.destroy", device->rocblas_destroy_result);
  if (device->properties_result.status == ET_AMD4_OK) {
    const et_amd4_device_properties_v1 *props = &device->properties;
    append_text(writer, "name", props->name);
    append_text(writer, "gcn-arch", props->gcn_arch_name);
    append(writer,
           "pci.domain=%" PRId32 "\npci.bus=%" PRId32 "\npci.device=%" PRId32
           "\ncompute-units=%" PRId32 "\nwavefront-size=%" PRId32
           "\nvram.total-bytes=%" PRIu64 "\n",
           props->pci_domain, props->pci_bus, props->pci_device,
           props->compute_units, props->wavefront_size,
           props->total_vram_bytes);
  }
  if (device->memory_result.status == ET_AMD4_OK) {
    append(writer,
           "vram.free-bytes=%" PRIu64 "\nmemory.query-total-bytes=%" PRIu64
           "\n",
           device->free_vram_bytes, device->memory_query_total_bytes);
  }
}

static void format_report(report_writer *writer,
                          const et_amd4_report_v1 *report) {
  static const char *const kfd_names[] = {"absent", "inaccessible",
                                          "accessible"};
  static const char *const library_names[] = {"unavailable", "missing-symbols",
                                              "ready"};
  append(writer,
         "eshkol-transformer-amd4-observability:v1\n"
         "abi=%" PRIu32 ".%" PRIu32 "\nsource=%s\nstatus=%s\n"
         "kfd=%s\nhip.library=%s\nrocblas.library=%s\n",
         report->abi_major, report->abi_minor,
         report->source == ET_AMD4_SOURCE_LIVE ? "live" : "injected",
         status_names[report->status], kfd_names[report->kfd],
         library_names[report->hip_library],
         library_names[report->rocblas_library]);
  append_result(writer, "hip.runtime-version-query",
                report->hip_runtime_version_result);
  if (report->hip_runtime_version_result.status == ET_AMD4_OK) {
    append(writer, "hip.runtime-version=%" PRId32 "\n",
           report->hip_runtime_version);
  }
  append_result(writer, "hip.driver-version-query",
                report->hip_driver_version_result);
  if (report->hip_driver_version_result.status == ET_AMD4_OK) {
    append(writer, "hip.driver-version=%" PRId32 "\n",
           report->hip_driver_version);
  }
  append_result(writer, "rocblas.version-query",
                report->rocblas_version_result);
  if (report->rocblas_version_result.status == ET_AMD4_OK) {
    append_text(writer, "rocblas.version", report->rocblas_version);
  }
  append_result(writer, "hip.device-count-query", report->device_count_result);
  if (report->device_count_result.status == ET_AMD4_OK ||
      report->device_count_result.status == ET_AMD4_ERR_NO_DEVICE) {
    append(writer, "devices.visible=%" PRIu32 "\n", report->device_count);
  }
  append(writer, "devices.reported=%" PRIu32 "\n",
         report->reported_device_count);
  if (report->requested_ordinal == ET_AMD4_ALL_DEVICES) {
    append(writer, "device.requested=all\n");
  } else {
    append(writer, "device.requested=%" PRId32 "\n", report->requested_ordinal);
  }
  append_result(writer, "hip.current-device-query",
                report->current_device_result);
  if (report->current_device_result.status == ET_AMD4_OK) {
    append(writer, "device.original=%" PRId32 "\n", report->original_ordinal);
  }
  append_result(writer, "hip.restore-device", report->restore_device_result);
  append(writer, "kernel.amd-hip.gemm=unavailable\nkernel.amd-hip.attention="
                 "unavailable\n");
  for (uint32_t i = 0; i < report->reported_device_count; ++i) {
    format_device(writer, &report->devices[i]);
  }
}

et_amd4_status et_amd4_format_report_v1(const et_amd4_report_v1 *report,
                                        char *buffer, size_t capacity,
                                        size_t *required_capacity) {
  if (!valid_storage(report, sizeof(*report), _Alignof(et_amd4_report_v1)) ||
      !valid_storage(required_capacity, sizeof(*required_capacity),
                     _Alignof(size_t)) ||
      overlaps(report, sizeof(*report), required_capacity,
               sizeof(*required_capacity)) ||
      (buffer == NULL && capacity != 0) ||
      (buffer != NULL &&
       (!valid_storage(buffer, capacity, 1) ||
        (capacity != 0 &&
         (overlaps(report, sizeof(*report), buffer, capacity) ||
          overlaps(required_capacity, sizeof(*required_capacity), buffer,
                   capacity))))) ||
      !valid_report(report)) {
    return ET_AMD4_ERR_INVALID_ARG;
  }

  report_writer measure = {0};
  format_report(&measure, report);
  if (measure.failed) {
    return ET_AMD4_ERR_DRIVER_FAILURE;
  }
  *required_capacity = measure.length + 1;
  if (buffer == NULL) {
    return ET_AMD4_OK;
  }
  if (capacity < *required_capacity) {
    return ET_AMD4_ERR_OUT_OF_RANGE;
  }
  report_writer write = {.buffer = buffer, .capacity = capacity};
  format_report(&write, report);
  return write.failed ? ET_AMD4_ERR_DRIVER_FAILURE : ET_AMD4_OK;
}

static int32_t reject_call(const et_kernel_call_v1 *call,
                           et_kernel_error *error) {
  (void)call;
  if (error != NULL) {
    *error = (et_kernel_error){
        .category = ET_KERNEL_ERROR_UNSUPPORTED,
        .code = ET_KERNEL_CODE_PROVIDER_REJECTED,
        .operation = "amd4.hip",
        .message = "AMD HIP kernel execution is unavailable in AMD4 ABI 1.0",
    };
  }
  return ET_KERNEL_ERROR_UNSUPPORTED;
}

static void no_invoke(const et_kernel_call_v1 *call) { (void)call; }

static const char *const amd_devices[] = {"amd"};
static const char *const gemm_operations[] = {"gemm"};
static const char *const attention_operations[] = {"attention"};
#define AMD4_IMPLEMENTATION "amd4.hip.observability"
#define AMD4_EVIDENCE "Host observability only. AMD HIP execution unavailable."

static const et_kernel_capability_v1 capabilities[] = {
    {
        .struct_size = sizeof(et_kernel_capability_v1),
        .name = "kernel.amd-hip.gemm",
        .status = ET_KERNEL_CAPABILITY_UNAVAILABLE,
        .implementation = AMD4_IMPLEMENTATION,
        .version = "1.0",
        .evidence = AMD4_EVIDENCE,
        .operation_count = 1,
        .operations = gemm_operations,
        .device_count = 1,
        .devices = amd_devices,
    },
    {
        .struct_size = sizeof(et_kernel_capability_v1),
        .name = "kernel.amd-hip.attention",
        .status = ET_KERNEL_CAPABILITY_UNAVAILABLE,
        .implementation = AMD4_IMPLEMENTATION,
        .version = "1.0",
        .evidence = AMD4_EVIDENCE,
        .operation_count = 1,
        .operations = attention_operations,
        .device_count = 1,
        .devices = amd_devices,
    },
};

const et_kernel_provider_v1 *et_amd4_kernel_provider_v1(void) {
  static const et_kernel_provider_v1 provider = {
      .struct_size = sizeof(et_kernel_provider_v1),
      .abi_major = ET_KERNEL_ABI_MAJOR,
      .abi_minor = ET_KERNEL_ABI_MINOR,
      .name = AMD4_IMPLEMENTATION,
      .version = "1.0",
      .evidence = AMD4_EVIDENCE,
      .capability_count = sizeof(capabilities) / sizeof(capabilities[0]),
      .capability_stride = sizeof(capabilities[0]),
      .capability_bytes = sizeof(capabilities),
      .capabilities = capabilities,
      .validate_call = reject_call,
      .invoke_call = no_invoke,
  };
  return &provider;
}
