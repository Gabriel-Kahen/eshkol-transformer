#include "eshkol_transformer/amd4_hip_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(expr)) {                                                             \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef struct fixture {
  int32_t count;
  int32_t active;
  int32_t runtime_code;
  int32_t driver_code;
  int32_t count_code;
  int32_t properties_code;
  int32_t get_code;
  int32_t set_code;
  int32_t memory_code;
  int32_t version_code;
  int32_t create_code;
  int32_t destroy_code;
  unsigned bad_properties;
  unsigned bad_memory;
  unsigned bad_version;
  unsigned null_handle;
  unsigned properties_calls;
  unsigned get_calls;
  unsigned set_calls;
  int32_t selections[ET_AMD4_MAX_DEVICES];
  unsigned memory_calls;
  int32_t memory_devices[ET_AMD4_MAX_DEVICES];
  unsigned version_calls;
  unsigned create_calls;
  int32_t create_devices[ET_AMD4_MAX_DEVICES];
  unsigned destroy_calls;
  int32_t handle_device;
} fixture;

static int32_t fake_runtime(void *context, int32_t *version) {
  *version = 60412345;
  return ((fixture *)context)->runtime_code;
}

static int32_t fake_driver(void *context, int32_t *version) {
  *version = 60312345;
  return ((fixture *)context)->driver_code;
}

static int32_t fake_count(void *context, int32_t *count) {
  fixture *f = context;
  *count = f->count;
  return f->count_code;
}

static int32_t fake_properties(void *context, int32_t ordinal,
                               et_amd4_device_properties_v1 *props) {
  fixture *f = context;
  CHECK(ordinal >= 0 && ordinal < f->count &&
        (uint32_t)ordinal < ET_AMD4_MAX_DEVICES);
  ++f->properties_calls;
  if (f->properties_code != 0) {
    return f->properties_code;
  }
  *props = (et_amd4_device_properties_v1){
      .pci_domain = ordinal,
      .pci_bus = 10 + ordinal,
      .pci_device = 2 + ordinal,
      .compute_units = 80 + ordinal,
      .wavefront_size = ordinal == 0 ? 64 : 32,
      .total_vram_bytes = UINT64_C(8589934592) + (uint64_t)ordinal,
  };
  strcpy(props->name, ordinal == 0 ? "Probe GPU A" : "Probe GPU B\n\"\\\377");
  strcpy(props->gcn_arch_name, ordinal == 0 ? "gfx90a:xnack-" : "gfx1100");
  if (f->bad_properties == 1) {
    memset(props->gcn_arch_name, 'x', sizeof(props->gcn_arch_name));
  } else if (f->bad_properties == 2) {
    props->compute_units = 0;
  }
  return 0;
}

static int32_t fake_get(void *context, int32_t *ordinal) {
  fixture *f = context;
  ++f->get_calls;
  *ordinal = f->active;
  return f->get_code;
}

static int32_t fake_set(void *context, int32_t ordinal) {
  fixture *f = context;
  CHECK(f->set_calls < sizeof(f->selections) / sizeof(f->selections[0]));
  f->selections[f->set_calls++] = ordinal;
  f->active = ordinal;
  return f->set_code == 0 || ordinal == 1 ? 0 : f->set_code;
}

static int32_t fake_memory(void *context, uint64_t *free_bytes,
                           uint64_t *total_bytes) {
  fixture *f = context;
  CHECK(f->memory_calls <
        sizeof(f->memory_devices) / sizeof(f->memory_devices[0]));
  f->memory_devices[f->memory_calls++] = f->active;
  *free_bytes = UINT64_C(4294967296) + (uint64_t)f->active;
  *total_bytes = f->bad_memory ? *free_bytes - 1 : UINT64_C(7516192768);
  return f->memory_code;
}

static int32_t fake_blas_version(void *context, char *version,
                                 size_t capacity) {
  fixture *f = context;
  ++f->version_calls;
  CHECK(capacity >= sizeof("test-rocBLAS-4.5.6"));
  if (f->bad_version) {
    memset(version, 'x', capacity);
  } else {
    strcpy(version, "test-rocBLAS-4.5.6");
  }
  return f->version_code;
}

static int32_t fake_create(void *context, void **handle) {
  fixture *f = context;
  CHECK(f->create_calls <
        sizeof(f->create_devices) / sizeof(f->create_devices[0]));
  f->create_devices[f->create_calls++] = f->active;
  if (f->create_code == 0 && !f->null_handle) {
    f->handle_device = f->active;
    *handle = &f->handle_device;
  }
  return f->create_code;
}

static int32_t fake_destroy(void *context, void *handle) {
  fixture *f = context;
  CHECK(handle == &f->handle_device && f->handle_device == f->active);
  ++f->destroy_calls;
  return f->destroy_code;
}

static et_amd4_ops_v1 fixture_ops(fixture *f) {
  return (et_amd4_ops_v1){
      .struct_size = sizeof(et_amd4_ops_v1),
      .context = f,
      .kfd = ET_AMD4_KFD_ACCESSIBLE,
      .hip_library = ET_AMD4_LIBRARY_READY,
      .rocblas_library = ET_AMD4_LIBRARY_READY,
      .hip_runtime_version = fake_runtime,
      .hip_driver_version = fake_driver,
      .hip_device_count = fake_count,
      .hip_device_properties = fake_properties,
      .hip_get_device = fake_get,
      .hip_set_device = fake_set,
      .hip_memory_info = fake_memory,
      .rocblas_version = fake_blas_version,
      .rocblas_create_handle = fake_create,
      .rocblas_destroy_handle = fake_destroy,
  };
}

static void test_enumeration(void) {
  fixture f = {.count = 1};
  et_amd4_ops_v1 ops = fixture_ops(&f);
  et_amd4_report_v1 report;
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(report.source == ET_AMD4_SOURCE_INJECTED);
  CHECK(report.status == ET_AMD4_OK && report.kfd == ET_AMD4_KFD_ACCESSIBLE);
  CHECK(report.hip_runtime_version == 60412345);
  CHECK(report.hip_driver_version == 60312345);
  CHECK(strcmp(report.rocblas_version, "test-rocBLAS-4.5.6") == 0);
  CHECK(report.device_count == 1 && report.reported_device_count == 1);
  CHECK(report.original_ordinal == 0 && f.set_calls == 0);
  CHECK(report.restore_device_result.status == ET_AMD4_NOT_RUN);
  CHECK(f.properties_calls == 1 && f.memory_calls == 1);
  CHECK(f.memory_devices[0] == 0 && f.create_devices[0] == 0);
  CHECK(f.create_calls == 1 && f.destroy_calls == 1);
  const et_amd4_device_info_v1 *device = &report.devices[0];
  CHECK(device->ordinal == 0 && device->properties_result.status == ET_AMD4_OK);
  CHECK(device->selection_result.status == ET_AMD4_OK);
  CHECK(device->memory_result.status == ET_AMD4_OK);
  CHECK(device->rocblas_create_result.status == ET_AMD4_OK);
  CHECK(device->rocblas_destroy_result.status == ET_AMD4_OK);
  CHECK(device->properties.wavefront_size == 64);
  CHECK(device->properties.total_vram_bytes == UINT64_C(8589934592));
  CHECK(device->free_vram_bytes == UINT64_C(4294967296));

  f = (fixture){.count = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(report.device_count == 2 && report.reported_device_count == 2);
  CHECK(report.devices[0].ordinal == 0 && report.devices[1].ordinal == 1);
  CHECK(report.devices[1].properties.wavefront_size == 32);
  CHECK(strcmp(report.devices[1].properties.gcn_arch_name, "gfx1100") == 0);
  CHECK(f.memory_calls == 2 && f.memory_devices[0] == 0 &&
        f.memory_devices[1] == 1);
  CHECK(f.create_calls == 2 && f.create_devices[0] == 0 &&
        f.create_devices[1] == 1);
  CHECK(f.destroy_calls == 2 && f.active == 0 && f.set_calls == 2);
  CHECK(f.selections[0] == 1 && f.selections[1] == 0);
  CHECK(report.restore_device_result.status == ET_AMD4_OK);

  f = (fixture){.count = 2, .active = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(f.active == 1 && f.set_calls == 2);
  CHECK(f.selections[0] == 0 && f.selections[1] == 1);
  CHECK(report.restore_device_result.status == ET_AMD4_NOT_RUN);

  f = (fixture){.count = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 1, &report) == ET_AMD4_OK);
  CHECK(report.device_count == 2 && report.reported_device_count == 1);
  CHECK(report.requested_ordinal == 1 && report.devices[0].ordinal == 1);
  CHECK(f.properties_calls == 1 && f.memory_calls == 1 &&
        f.memory_devices[0] == 1);
  CHECK(f.create_calls == 1 && f.destroy_calls == 1 && f.active == 0);
  CHECK(f.set_calls == 2 && f.selections[0] == 1 && f.selections[1] == 0);

  f = (fixture){.count = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 2, &report) ==
        ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(report.device_count == 2 && report.reported_device_count == 0);
  CHECK(f.properties_calls == 0 && f.get_calls == 0 && f.set_calls == 0);
  CHECK(et_amd4_probe_with_ops_v1(&ops, INT32_MAX, &report) ==
        ET_AMD4_ERR_OUT_OF_RANGE);

  f = (fixture){.count = ET_AMD4_MAX_DEVICES};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(report.reported_device_count == ET_AMD4_MAX_DEVICES);
  CHECK(report.devices[ET_AMD4_MAX_DEVICES - 1].ordinal ==
        (int32_t)ET_AMD4_MAX_DEVICES - 1);
  CHECK(f.set_calls == ET_AMD4_MAX_DEVICES && f.active == 0);
  CHECK(f.memory_calls == ET_AMD4_MAX_DEVICES &&
        f.destroy_calls == ET_AMD4_MAX_DEVICES);
}

static void test_hip_failures(void) {
  fixture f = {.count_code = 100, .runtime_code = 3};
  et_amd4_ops_v1 ops = fixture_ops(&f);
  et_amd4_report_v1 report;
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_NO_DEVICE);
  CHECK(report.device_count_result.status == ET_AMD4_ERR_NO_DEVICE);
  CHECK(report.device_count_result.native_code == 100 &&
        report.device_count == 0);
  CHECK(report.hip_runtime_version_result.status == ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.hip_runtime_version == 0);
  CHECK(report.hip_driver_version == 60312345 && f.get_calls == 0);

  f = (fixture){0};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_NO_DEVICE);
  CHECK(report.device_count_result.status == ET_AMD4_OK &&
        report.device_count == 0);
  f.count_code = 999;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.device_count_result.native_code == 999 && f.get_calls == 0);
  f = (fixture){.count = -1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.device_count_result.status == ET_AMD4_ERR_DRIVER_FAILURE);
  f = (fixture){.count = INT32_MAX};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(report.device_count == INT32_MAX && report.reported_device_count == 0);
  CHECK(f.properties_calls == 0 && f.get_calls == 0);

  f = (fixture){.count = 1, .driver_code = 35};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.hip_driver_version_result.native_code == 35);
  CHECK(report.hip_driver_version == 0 &&
        report.hip_runtime_version == 60412345);
  CHECK(f.memory_calls == 1 && f.create_calls == 1 && f.destroy_calls == 1);

  f = (fixture){.count = 1, .properties_code = 101};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(report.devices[0].properties_result.native_code == 101);
  CHECK(report.devices[0].selection_result.status == ET_AMD4_NOT_RUN);
  CHECK(f.memory_calls == 0 && f.create_calls == 0);

  f = (fixture){.count = 1, .properties_code = 801};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DEVICE_UNSUPPORTED);
  CHECK(report.devices[0].properties_result.native_code == 801);
  f = (fixture){.count = 1, .bad_properties = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DEVICE_UNSUPPORTED);
  CHECK(report.devices[0].properties.gcn_arch_name[0] == '\0');
  f = (fixture){.count = 1, .bad_properties = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.devices[0].properties.compute_units == 0);

  f = (fixture){.count = 1, .get_code = 999};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.current_device_result.native_code == 999 &&
        report.original_ordinal == -1);
  CHECK(report.devices[0].properties_result.status == ET_AMD4_OK);
  CHECK(f.set_calls == 0 && f.memory_calls == 0);

  f = (fixture){.count = 1, .active = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.current_device_result.native_code == 0 &&
        report.original_ordinal == -1);

  f = (fixture){.count = 2, .active = 1, .set_code = 101};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(report.devices[0].selection_result.native_code == 101);
  CHECK(report.devices[0].memory_result.status == ET_AMD4_NOT_RUN);
  CHECK(report.devices[0].rocblas_create_result.status == ET_AMD4_NOT_RUN);
  CHECK(f.set_calls == 2 && f.selections[0] == 0 && f.selections[1] == 1);
  CHECK(report.restore_device_result.status == ET_AMD4_OK && f.active == 1);

  f = (fixture){.count = 2, .set_code = 999};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 1, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.restore_device_result.native_code == 999 && f.set_calls == 2);
  CHECK(report.devices[0].memory_result.status == ET_AMD4_OK &&
        f.destroy_calls == 1);

  f = (fixture){.count = 2, .memory_code = 999};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.devices[0].memory_result.native_code == 999);
  CHECK(report.devices[1].memory_result.status == ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.devices[1].free_vram_bytes == 0 &&
        report.devices[1].memory_query_total_bytes == 0);
  CHECK(f.properties_calls == 2 && f.memory_calls == 2 &&
        f.destroy_calls == 2 && f.active == 0);

  f = (fixture){.count = 1, .memory_code = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_ALLOC);
  CHECK(report.devices[0].memory_result.native_code == 2);
  f = (fixture){.count = 1, .bad_memory = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(report.devices[0].memory_result.native_code == 0 &&
        report.devices[0].free_vram_bytes == 0);
}

static void test_library_failures(void) {
  fixture f = {.count = 1};
  et_amd4_ops_v1 ops = fixture_ops(&f);
  et_amd4_report_v1 report;
  ops.hip_library = ET_AMD4_LIBRARY_UNAVAILABLE;
  ops.hip_runtime_version = NULL;
  ops.kfd = ET_AMD4_KFD_INACCESSIBLE;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_UNAVAILABLE_RUNTIME);
  CHECK(report.kfd == ET_AMD4_KFD_INACCESSIBLE &&
        report.hip_library == ET_AMD4_LIBRARY_UNAVAILABLE);
  CHECK(report.hip_runtime_version_result.status == ET_AMD4_NOT_RUN);
  CHECK(report.device_count_result.status == ET_AMD4_NOT_RUN);
  CHECK(report.rocblas_version_result.status == ET_AMD4_OK &&
        f.version_calls == 1);
  CHECK(f.properties_calls == 0 && f.create_calls == 0);

  ops.hip_library = ET_AMD4_LIBRARY_MISSING_SYMBOLS;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_UNAVAILABLE_RUNTIME);
  CHECK(report.hip_library == ET_AMD4_LIBRARY_MISSING_SYMBOLS);
  ops = fixture_ops(&f);
  ops.rocblas_library = ET_AMD4_LIBRARY_UNAVAILABLE;
  ops.rocblas_create_handle = NULL;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.rocblas_version_result.status == ET_AMD4_NOT_RUN);
  CHECK(report.devices[0].rocblas_create_result.status == ET_AMD4_NOT_RUN);
  CHECK(f.memory_calls == 1 && f.create_calls == 0 && f.destroy_calls == 0);

  f = (fixture){.count = 1};
  ops.rocblas_library = ET_AMD4_LIBRARY_MISSING_SYMBOLS;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.rocblas_library == ET_AMD4_LIBRARY_MISSING_SYMBOLS &&
        f.version_calls == 0);

  ops = fixture_ops(&f);
  f = (fixture){.count = 1, .version_code = 6};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.rocblas_version_result.native_code == 6 &&
        report.rocblas_version[0] == '\0');
  CHECK(f.create_calls == 1 && f.destroy_calls == 1);
  f = (fixture){.count = 1, .bad_version = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.rocblas_version_result.native_code == 0 &&
        report.rocblas_version[0] == '\0');

  f = (fixture){.count = 1, .create_code = 6};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.devices[0].rocblas_create_result.native_code == 6);
  CHECK(report.devices[0].rocblas_destroy_result.status == ET_AMD4_NOT_RUN);
  CHECK(f.create_calls == 1 && f.destroy_calls == 0);
  f = (fixture){.count = 1, .create_code = 5};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_ALLOC);
  CHECK(f.create_calls == 1 && f.destroy_calls == 0);
  f = (fixture){.count = 1, .null_handle = 1};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(f.create_calls == 1 && f.destroy_calls == 0);
  f = (fixture){.count = 2, .destroy_code = 6};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 1, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(report.devices[0].rocblas_destroy_result.native_code == 6);
  CHECK(f.create_calls == 1 && f.destroy_calls == 1 && f.active == 0);
}

static void test_format(void) {
  static const char expected[] =
      "eshkol-transformer-amd4-observability:v1\n"
      "abi=1.0\nsource=injected\nstatus=ok\n"
      "kfd=accessible\nhip.library=ready\nrocblas.library=ready\n"
      "hip.runtime-version-query=ok,0\nhip.runtime-version=60412345\n"
      "hip.driver-version-query=ok,0\nhip.driver-version=60312345\n"
      "rocblas.version-query=ok,0\nrocblas.version=\"test-rocBLAS-4.5.6\"\n"
      "hip.device-count-query=ok,0\ndevices.visible=2\ndevices.reported=2\n"
      "device.requested=all\nhip.current-device-query=ok,0\ndevice.original=0\n"
      "hip.restore-device=ok,0\n"
      "kernel.amd-hip.gemm=unavailable\nkernel.amd-hip.attention=unavailable\n"
      "[device.0]\nproperties=ok,0\nselection=ok,0\nmemory=ok,0\n"
      "rocblas.create=ok,0\nrocblas.destroy=ok,0\n"
      "name=\"Probe GPU A\"\ngcn-arch=\"gfx90a:xnack-\"\n"
      "pci.domain=0\npci.bus=10\npci.device=2\ncompute-units=80\n"
      "wavefront-size=64\nvram.total-bytes=8589934592\n"
      "vram.free-bytes=4294967296\nmemory.query-total-bytes=7516192768\n"
      "[device.1]\nproperties=ok,0\nselection=ok,0\nmemory=ok,0\n"
      "rocblas.create=ok,0\nrocblas.destroy=ok,0\n"
      "name=\"Probe GPU B\\x0a\\x22\\x5c\\xff\"\ngcn-arch=\"gfx1100\"\n"
      "pci.domain=1\npci.bus=11\npci.device=3\ncompute-units=81\n"
      "wavefront-size=32\nvram.total-bytes=8589934593\n"
      "vram.free-bytes=4294967297\nmemory.query-total-bytes=7516192768\n";
  fixture f = {.count = 2};
  const et_amd4_ops_v1 ops = fixture_ops(&f);
  et_amd4_report_v1 report;
  char text[sizeof(expected)];
  char sentinel[sizeof(text)];
  memset(text, 's', sizeof(text));
  memcpy(sentinel, text, sizeof(text));
  size_t required = 0;
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(et_amd4_format_report_v1(&report, NULL, 0, &required) == ET_AMD4_OK);
  CHECK(required == sizeof(expected));
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text) - 1, &required) ==
        ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(required == sizeof(expected) &&
        memcmp(text, sentinel, sizeof(text)) == 0);
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_OK);
  CHECK(memcmp(text, expected, sizeof(expected)) == 0);
  f = (fixture){.count = 2};
  CHECK(et_amd4_probe_with_ops_v1(&ops, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_OK);
  CHECK(et_amd4_format_report_v1(&report, sentinel, sizeof(sentinel),
                                 &required) == ET_AMD4_OK);
  CHECK(memcmp(text, sentinel, sizeof(text)) == 0);

  f = (fixture){.count = 1, .memory_code = 999};
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "memory=driver-failure,999\n") != NULL);
  CHECK(strstr(text, "vram.free-bytes=") == NULL);
  CHECK(strstr(text, "vram.total-bytes=8589934592\n") != NULL);
}

static void test_invalid_arguments(void) {
  fixture f = {.count = 1};
  et_amd4_ops_v1 ops = fixture_ops(&f);
  et_amd4_report_v1 report;
  et_amd4_report_v1 original;
  memset(&report, 0x5a, sizeof(report));
  memcpy(&original, &report, sizeof(original));
  CHECK(et_amd4_probe_v1(0, NULL) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_probe_v1(-2, &report) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, NULL) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_probe_with_ops_v1(NULL, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_probe_with_ops_v1(&ops, INT32_MIN, &report) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_probe_with_ops_v1((const et_amd4_ops_v1 *)&report, 0,
                                  &report) == ET_AMD4_ERR_INVALID_ARG);
  ops.context = &report.devices[0];
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  ops.context = &f;
  ops.struct_size = sizeof(ops) - 1;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  ops.struct_size = SIZE_MAX;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  ops.struct_size = sizeof(ops);
  ops.hip_memory_info = NULL;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  ops = fixture_ops(&f);
  ops.rocblas_destroy_handle = NULL;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  ops = fixture_ops(&f);
  ops.kfd = UINT32_MAX;
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(memcmp(&report, &original, sizeof(report)) == 0);
  CHECK(f.get_calls == 0 && f.version_calls == 0);

  ops = fixture_ops(&f);
  CHECK(et_amd4_probe_with_ops_v1(&ops, 0, &report) == ET_AMD4_OK);
  char text[] = "sentinel";
  size_t required = 37;
  CHECK(et_amd4_format_report_v1(NULL, text, sizeof(text), &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, NULL, 1, &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, NULL, 0, NULL) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, text, SIZE_MAX, &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, (char *)&report, sizeof(report),
                                 &required) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, (char *)&required, sizeof(required),
                                 &required) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_format_report_v1(&report, NULL, 0, (size_t *)&report) ==
        ET_AMD4_ERR_INVALID_ARG);
  report.reported_device_count = UINT32_MAX;
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  report.reported_device_count = 1;
  report.devices[0].ordinal = -1;
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  report.devices[0].ordinal = 0;
  report.abi_minor = 1;
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  report.abi_minor = 0;
  memset(report.devices[0].properties.name, 'x',
         sizeof(report.devices[0].properties.name));
  CHECK(et_amd4_format_report_v1(&report, text, sizeof(text), &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(required == 37 && strcmp(text, "sentinel") == 0);
}

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_amd4_kernel_provider_v1();
}

static void test_k1(void) {
  const et_kernel_provider_v1 *provider = et_amd4_kernel_provider_v1();
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(provider->abi_major == 1 && provider->abi_minor == 0);
  CHECK(provider->capability_count == 2);
  CHECK(et_kernel_runtime_discover(resolve_provider, NULL, &runtime, &error) ==
        0);
  const char *const names[] = {"kernel.amd-hip.gemm",
                               "kernel.amd-hip.attention"};
  const char *const operations[] = {"gemm", "attention"};
  const uint64_t shape[] = {1, 1};
  uint32_t input = UINT32_C(0x3f800000);
  uint32_t output = UINT32_C(0x7fc12345);
  et_kernel_tensor_view_v1 in_view = {
      .struct_size = sizeof(in_view),
      .data = &input,
      .byte_length = sizeof(input),
      .dtype = "f32",
      .device = "amd",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = 2,
      .shape = shape,
  };
  et_kernel_tensor_view_v1 out_view = in_view;
  out_view.data = &output;
  et_kernel_request_v1 request = {
      .struct_size = sizeof(request),
      .dtype = "f32",
      .device = "amd",
      .rank = 2,
      .shape = shape,
  };
  et_kernel_call_v1 call = {
      .struct_size = sizeof(call),
      .request = &request,
      .input_count = 1,
      .input_stride = sizeof(in_view),
      .input_bytes = sizeof(in_view),
      .inputs = &in_view,
      .output_count = 1,
      .output_stride = sizeof(out_view),
      .output_bytes = sizeof(out_view),
      .outputs = &out_view,
  };
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
    const et_kernel_capability_v1 *cap =
        et_kernel_runtime_capability_find(runtime, names[i]);
    CHECK(cap != NULL && cap->status == ET_KERNEL_CAPABILITY_UNAVAILABLE);
    CHECK(cap->device_count == 1 && strcmp(cap->devices[0], "amd") == 0);
    CHECK(cap->operation_count == 1 &&
          strcmp(cap->operations[0], operations[i]) == 0);
    CHECK(cap->deterministic == 0 && cap->dtype_count == 0 &&
          cap->shape_range_count == 0);
    request.operation = operations[i];
    call.capability = names[i];

    CHECK(et_kernel_runtime_capability_require(runtime, names[i], &request,
                                               &cap, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(cap == NULL && error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    CHECK(provider->validate_call(&call, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(error.category == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    provider->invoke_call(&call);
    CHECK(output == UINT32_C(0x7fc12345) && input == UINT32_C(0x3f800000));
  }

  CHECK(provider->validate_call(NULL, &error) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(provider->validate_call(NULL, NULL) == ET_KERNEL_ERROR_UNSUPPORTED);
  provider->invoke_call(NULL);
  CHECK(et_kernel_runtime_dispatch(runtime, NULL, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_NULL_ARGUMENT);
  out_view.data = &input;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  out_view.data = &output;
  call.output_count = SIZE_MAX;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_INTEGER_OVERFLOW);
  CHECK(output == UINT32_C(0x7fc12345) && input == UINT32_C(0x3f800000));

  size_t required = 0;
  CHECK(et_kernel_runtime_report_json(runtime, NULL, 0, &required, &error) ==
        0);
  char *json = malloc(required);
  CHECK(json != NULL);
  CHECK(et_kernel_runtime_report_json(runtime, json, required, &required,
                                      &error) == 0);
  CHECK(strstr(json,
               "\"name\":\"kernel.amd-hip.gemm\",\"status\":\"unsupported\"") !=
        NULL);
  CHECK(
      strstr(
          json,
          "\"name\":\"kernel.amd-hip.attention\",\"status\":\"unsupported\"") !=
      NULL);
  free(json);
  et_kernel_runtime_destroy(runtime);
}

static void test_live(void) {
  et_amd4_report_v1 report;
  const et_amd4_status status = et_amd4_probe_v1(ET_AMD4_ALL_DEVICES, &report);
  CHECK(report.source == ET_AMD4_SOURCE_LIVE && status == report.status);
  const char *expect_absent = getenv("AMD4_EXPECT_NO_HIP");
  if (expect_absent != NULL && strcmp(expect_absent, "1") == 0) {
    CHECK(report.kfd == ET_AMD4_KFD_ABSENT);
    CHECK(report.hip_library == ET_AMD4_LIBRARY_UNAVAILABLE);
    CHECK(report.rocblas_library == ET_AMD4_LIBRARY_UNAVAILABLE);
  }
  if (report.hip_library != ET_AMD4_LIBRARY_READY) {
    CHECK(status == ET_AMD4_ERR_UNAVAILABLE_RUNTIME);
    CHECK(report.device_count_result.status == ET_AMD4_NOT_RUN);
    CHECK(report.device_count == 0 && report.reported_device_count == 0);
  }
  size_t required = 0;
  CHECK(et_amd4_format_report_v1(&report, NULL, 0, &required) == ET_AMD4_OK);
  char *text = malloc(required);
  CHECK(text != NULL);
  CHECK(et_amd4_format_report_v1(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "kernel.amd-hip.gemm=unavailable\n") != NULL);
  CHECK(strstr(text, "kernel.amd-hip.attention=unavailable\n") != NULL);
  CHECK(strstr(text, "source=live\n") != NULL);
  fputs(text, stdout);
  free(text);
}

int main(void) {
  test_enumeration();
  test_hip_failures();
  test_library_failures();
  test_format();
  test_invalid_arguments();
  test_k1();
  test_live();
  printf("AMD4 PASS: %u checks\n", checks);
  return EXIT_SUCCESS;
}
