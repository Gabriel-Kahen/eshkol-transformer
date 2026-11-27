#define _GNU_SOURCE

#include "eshkol_transformer/amd4_hip_abi.h"
#include "loader_fixture.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

amd4_loader_fixture_state amd4_loader_fixture;

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static void expect_result(et_amd4_result_v1 result, et_amd4_status status,
                          int32_t code) {
  CHECK(result.status == status && result.native_code == code);
}

static void reset_fixture(int32_t count) {
  amd4_loader_fixture = (amd4_loader_fixture_state){.count = count, .active = 1};
}

static void expect_probe(et_amd4_report_v1 *report, int32_t ordinal,
                         et_amd4_status status, const char *events) {
  CHECK(et_amd4_probe_v1(ordinal, report) == status);
  CHECK(report->source == ET_AMD4_SOURCE_LIVE && report->status == status);
  CHECK(report->requested_ordinal == ordinal);
  if (strcmp(amd4_loader_fixture.events, events) != 0) {
    fprintf(stderr, "expected fixture events %s, got %s\n", events,
            amd4_loader_fixture.events);
    exit(EXIT_FAILURE);
  }
  CHECK(dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD) == NULL);
  CHECK(dlopen("librocblas.so", RTLD_NOW | RTLD_NOLOAD) == NULL);
}

static void expect_hip_versions(const et_amd4_report_v1 *report) {
  CHECK(report->hip_library == ET_AMD4_LIBRARY_READY);
  expect_result(report->hip_runtime_version_result, ET_AMD4_OK, 0);
  expect_result(report->hip_driver_version_result, ET_AMD4_OK, 0);
  CHECK(report->hip_runtime_version == 60431234);
  CHECK(report->hip_driver_version == 60430000);
}

static void expect_blas_version(const et_amd4_report_v1 *report) {
  CHECK(report->rocblas_library == ET_AMD4_LIBRARY_READY);
  expect_result(report->rocblas_version_result, ET_AMD4_OK, 0);
  CHECK(strcmp(report->rocblas_version, "fixture rocBLAS") == 0);
}

static void expect_device(const et_amd4_device_info_v1 *device,
                          int32_t ordinal) {
  CHECK(device->ordinal == ordinal);
  expect_result(device->properties_result, ET_AMD4_OK, 0);
  expect_result(device->selection_result, ET_AMD4_OK, 0);
  expect_result(device->memory_result, ET_AMD4_OK, 0);
  CHECK(strcmp(device->properties.name, "fixture GPU") == 0);
  CHECK(strcmp(device->properties.gcn_arch_name, "gfx90a:xnack-") == 0);
  CHECK(device->properties.pci_domain == 17);
  CHECK(device->properties.pci_bus == 83 + ordinal);
  CHECK(device->properties.pci_device == 22);
  CHECK(device->properties.compute_units == 110 + ordinal);
  CHECK(device->properties.wavefront_size == 64);
  CHECK(device->properties.total_vram_bytes == UINT64_C(0x601234567) +
                                                   (uint64_t)ordinal);
  CHECK(device->free_vram_bytes == (ordinal == 0 ? UINT64_C(0x401234567) : 0));
  CHECK(device->memory_query_total_bytes == UINT64_C(0x501234567) +
                                                (uint64_t)ordinal);
}

static void test_ready(void) {
  et_amd4_report_v1 report;
  reset_fixture(2);
  expect_probe(&report, 0, ET_AMD4_OK, "g1p0s0m0c0d0s1");
  expect_hip_versions(&report);
  expect_blas_version(&report);
  expect_result(report.current_device_result, ET_AMD4_OK, 0);
  expect_result(report.device_count_result, ET_AMD4_OK, 0);
  CHECK(report.original_ordinal == 1 && report.device_count == 2 &&
        report.reported_device_count == 1);
  expect_device(&report.devices[0], 0);
  expect_result(report.devices[0].rocblas_create_result, ET_AMD4_OK, 0);
  expect_result(report.devices[0].rocblas_destroy_result, ET_AMD4_OK, 0);
  expect_result(report.restore_device_result, ET_AMD4_OK, 0);
  CHECK(amd4_loader_fixture.active == 1);

  reset_fixture(2);
  expect_probe(&report, ET_AMD4_ALL_DEVICES, ET_AMD4_OK,
               "g1p0s0m0c0d0p1s1m1c1d1");
  CHECK(report.reported_device_count == 2);
  expect_device(&report.devices[0], 0);
  expect_device(&report.devices[1], 1);
  expect_result(report.restore_device_result, ET_AMD4_NOT_RUN, 0);
  CHECK(amd4_loader_fixture.active == 1);

  reset_fixture(0);
  expect_probe(&report, ET_AMD4_ALL_DEVICES, ET_AMD4_ERR_NO_DEVICE, "");
  expect_result(report.device_count_result, ET_AMD4_OK, 0);
  CHECK(report.device_count == 0 && report.reported_device_count == 0);
}

static void test_error_precedence(void) {
  et_amd4_report_v1 report;
  reset_fixture(2);
  amd4_loader_fixture.memory_code = 801;
  amd4_loader_fixture.create_code = 5;
  amd4_loader_fixture.destroy_code = 6;
  amd4_loader_fixture.restore_code = 101;
  expect_probe(&report, 0, ET_AMD4_ERR_DEVICE_UNSUPPORTED, "g1p0s0m0c0d0s1");
  const et_amd4_device_info_v1 *device = &report.devices[0];
  expect_result(device->memory_result, ET_AMD4_ERR_DEVICE_UNSUPPORTED, 801);
  CHECK(device->free_vram_bytes == 0 && device->memory_query_total_bytes == 0);
  expect_result(device->rocblas_create_result, ET_AMD4_ERR_ALLOC, 5);
  expect_result(device->rocblas_destroy_result, ET_AMD4_ERR_ROCBLAS_UNAVAILABLE,
                6);
  expect_result(report.restore_device_result, ET_AMD4_ERR_OUT_OF_RANGE, 101);
  CHECK(amd4_loader_fixture.active == 0);

  reset_fixture(2);
  amd4_loader_fixture.create_code = 5;
  amd4_loader_fixture.restore_code = 101;
  expect_probe(&report, 0, ET_AMD4_ERR_OUT_OF_RANGE, "g1p0s0m0c0d0s1");
  expect_result(report.restore_device_result, ET_AMD4_ERR_OUT_OF_RANGE, 101);
  CHECK(amd4_loader_fixture.active == 0);

  reset_fixture(2);
  amd4_loader_fixture.create_code = 5;
  amd4_loader_fixture.destroy_code = 6;
  expect_probe(&report, 0, ET_AMD4_ERR_ALLOC, "g1p0s0m0c0d0s1");
  expect_result(report.restore_device_result, ET_AMD4_OK, 0);
  CHECK(amd4_loader_fixture.active == 1);
}

static void test_hip_unavailable(et_amd4_library_state library_state) {
  et_amd4_report_v1 report;
  reset_fixture(2);
  expect_probe(&report, 0, ET_AMD4_ERR_UNAVAILABLE_RUNTIME, "");
  expect_blas_version(&report);
  CHECK(report.hip_library == library_state);
  expect_result(report.hip_runtime_version_result, ET_AMD4_NOT_RUN, 0);
  expect_result(report.device_count_result, ET_AMD4_NOT_RUN, 0);
  CHECK(report.device_count == 0 && report.reported_device_count == 0);
}

static void test_blas_unavailable(et_amd4_library_state library_state) {
  et_amd4_report_v1 report;
  reset_fixture(2);
  expect_probe(&report, 0, ET_AMD4_ERR_ROCBLAS_UNAVAILABLE, "g1p0s0m0s1");
  expect_hip_versions(&report);
  CHECK(report.rocblas_library == library_state);
  expect_result(report.rocblas_version_result, ET_AMD4_NOT_RUN, 0);
  CHECK(report.reported_device_count == 1);
  expect_device(&report.devices[0], 0);
  expect_result(report.devices[0].rocblas_create_result, ET_AMD4_NOT_RUN, 0);
  expect_result(report.devices[0].rocblas_destroy_result, ET_AMD4_NOT_RUN, 0);
  expect_result(report.restore_device_result, ET_AMD4_OK, 0);
  CHECK(amd4_loader_fixture.active == 1);
}

void test_amd4_loader_fixture(const char *scenario) {
  if (strcmp(scenario, "ready") == 0) {
    test_ready();
    test_error_precedence();
  } else if (strcmp(scenario, "hip-missing-export") == 0) {
    test_hip_unavailable(ET_AMD4_LIBRARY_MISSING_SYMBOLS);
  } else if (strcmp(scenario, "blas-missing-export") == 0) {
    test_blas_unavailable(ET_AMD4_LIBRARY_MISSING_SYMBOLS);
  } else if (strcmp(scenario, "hip-unavailable") == 0) {
    test_hip_unavailable(ET_AMD4_LIBRARY_UNAVAILABLE);
  } else if (strcmp(scenario, "blas-unavailable") == 0) {
    test_blas_unavailable(ET_AMD4_LIBRARY_UNAVAILABLE);
  } else {
    fprintf(stderr, "unknown AMD4 loader fixture scenario %s\n", scenario);
    exit(EXIT_FAILURE);
  }
}
