#include "amd4_eshkol_bridge.h"

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

static void test_live_probe(void) {
  et_amd4_report_v1 report;
  const int32_t status = et_amd4_aot_probe_live(ET_AMD4_ALL_DEVICES, &report);
  CHECK(status == et_amd4_aot_report_status(&report));
  CHECK(et_amd4_aot_report_source(&report) == ET_AMD4_SOURCE_LIVE);

  const char *expect_absent = getenv("AMD4_EXPECT_NO_HIP");
  if (expect_absent != NULL && strcmp(expect_absent, "1") == 0) {
    CHECK(status == ET_AMD4_ERR_UNAVAILABLE_RUNTIME);
    CHECK(et_amd4_aot_report_kfd(&report) == ET_AMD4_KFD_ABSENT);
    CHECK(et_amd4_aot_report_hip_library(&report) ==
          ET_AMD4_LIBRARY_UNAVAILABLE);
    CHECK(et_amd4_aot_report_rocblas_library(&report) ==
          ET_AMD4_LIBRARY_UNAVAILABLE);
    CHECK(et_amd4_aot_report_device_count(&report) == 0);
    CHECK(et_amd4_aot_report_reported_device_count(&report) == 0);
  }

  int64_t required = 0;
  CHECK(et_amd4_aot_format_report(&report, NULL, 0, &required) == ET_AMD4_OK);
  CHECK(required > 1);
  char *text = (char *)malloc((size_t)required);
  CHECK(text != NULL);
  CHECK(et_amd4_aot_format_report(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "eshkol-transformer-amd4-observability:v1\n") == text);
  CHECK(strstr(text, "source=live\n") != NULL);
  CHECK(strstr(text, "kernel.amd-hip.gemm=unavailable\n") != NULL);
  CHECK(strstr(text, "kernel.amd-hip.attention=unavailable\n") != NULL);
  CHECK(et_amd4_aot_read_byte(text, 0) == 'e');
  CHECK(et_amd4_aot_read_byte(text, required - 1) == 0);
  free(text);
}

static void test_injected_scenarios(void) {
  et_amd4_report_v1 report;
  int64_t required = 0;

  /* 1. Two-device MI250X + RX 7900 XTX mock. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE,
                                   ET_AMD4_ALL_DEVICES, &report) == ET_AMD4_OK);
  CHECK(et_amd4_aot_report_source(&report) == ET_AMD4_SOURCE_INJECTED);
  CHECK(et_amd4_aot_report_kfd(&report) == ET_AMD4_KFD_ACCESSIBLE);
  CHECK(et_amd4_aot_report_hip_library(&report) == ET_AMD4_LIBRARY_READY);
  CHECK(et_amd4_aot_report_rocblas_library(&report) == ET_AMD4_LIBRARY_READY);
  CHECK(et_amd4_aot_report_device_count(&report) == 2);
  CHECK(et_amd4_aot_report_reported_device_count(&report) == 2);
  CHECK(et_amd4_aot_report_device_compute_units(&report, 0) == 110);
  CHECK(et_amd4_aot_report_device_wavefront_size(&report, 0) == 64);
  CHECK(et_amd4_aot_report_device_total_vram(&report, 0) ==
        INT64_C(68719476736));
  CHECK(et_amd4_aot_report_device_free_vram(&report, 0) ==
        INT64_C(64424509440));
  CHECK(et_amd4_aot_report_device_compute_units(&report, 1) == 96);
  CHECK(et_amd4_aot_report_device_wavefront_size(&report, 1) == 32);
  CHECK(et_amd4_aot_report_device_total_vram(&report, 1) ==
        INT64_C(25769803776));
  CHECK(et_amd4_aot_report_device_free_vram(&report, 1) ==
        INT64_C(21474836480));
  CHECK(et_amd4_aot_report_device_compute_units(&report, 2) == -1);
  CHECK(et_amd4_aot_report_device_compute_units(&report, -1) == -1);

  CHECK(et_amd4_aot_format_report(&report, NULL, 0, &required) == ET_AMD4_OK);
  char *text = (char *)malloc((size_t)required);
  CHECK(text != NULL);
  CHECK(et_amd4_aot_format_report(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "eshkol-transformer-amd4-observability:v1\n") == text);
  CHECK(strstr(text, "source=injected\nstatus=ok\n") != NULL);
  CHECK(strstr(text, "name=\"AMD Instinct MI250X\"\n") != NULL);
  CHECK(strstr(text, "gcn-arch=\"gfx90a:sramecc+:xnack-\"\n") != NULL);
  CHECK(strstr(text, "name=\"AMD Radeon RX 7900 XTX\"\n") != NULL);
  CHECK(strstr(text, "gcn-arch=\"gfx1100\"\n") != NULL);
  CHECK(strstr(text, "hip.restore-device=ok,0\n") != NULL);
  free(text);

  /* Single ordinal selection and out-of-range ordinal. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE, 1,
                                   &report) == ET_AMD4_OK);
  CHECK(et_amd4_aot_report_device_count(&report) == 2);
  CHECK(et_amd4_aot_report_reported_device_count(&report) == 1);
  CHECK(et_amd4_aot_report_device_compute_units(&report, 0) == 96);
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE, 2,
                                   &report) == ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(et_amd4_aot_report_device_count(&report) == 2);
  CHECK(et_amd4_aot_report_reported_device_count(&report) == 0);

  /* 2. OOM scenario. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_OOM,
                                   ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_ALLOC);
  CHECK(et_amd4_aot_report_device_free_vram(&report, 0) == -1);
  CHECK(et_amd4_aot_format_report(&report, NULL, 0, &required) == ET_AMD4_OK);
  text = (char *)malloc((size_t)required);
  CHECK(text != NULL);
  CHECK(et_amd4_aot_format_report(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "source=injected\nstatus=alloc\n") != NULL);
  CHECK(strstr(text, "memory=alloc,2\n") != NULL);
  CHECK(strstr(text, "vram.free-bytes=") == NULL);
  free(text);

  /* 3. Driver error prioritized over rocBLAS create error. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_DRIVER_ERROR,
                                   ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_DRIVER_FAILURE);
  CHECK(et_amd4_aot_format_report(&report, NULL, 0, &required) == ET_AMD4_OK);
  text = (char *)malloc((size_t)required);
  CHECK(text != NULL);
  CHECK(et_amd4_aot_format_report(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "status=driver-failure\n") != NULL);
  CHECK(strstr(text, "hip.driver-version-query=driver-failure,35\n") != NULL);
  CHECK(strstr(text, "rocblas.create=rocblas-unavailable,6\n") != NULL);
  free(text);

  /* 4. rocBLAS missing symbols. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_ROCBLAS_MISSING_SYMBOLS,
                                   ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_ROCBLAS_UNAVAILABLE);
  CHECK(et_amd4_aot_report_rocblas_library(&report) ==
        ET_AMD4_LIBRARY_MISSING_SYMBOLS);
  CHECK(et_amd4_aot_format_report(&report, NULL, 0, &required) == ET_AMD4_OK);
  text = (char *)malloc((size_t)required);
  CHECK(text != NULL);
  CHECK(et_amd4_aot_format_report(&report, text, required, &required) ==
        ET_AMD4_OK);
  CHECK(strstr(text, "status=rocblas-unavailable\n") != NULL);
  CHECK(strstr(text, "rocblas.library=missing-symbols\n") != NULL);
  CHECK(strstr(text, "rocblas.version-query=not-run,0\n") != NULL);
  free(text);
}

static void test_negative_arguments(void) {
  et_amd4_report_v1 report;
  et_amd4_report_v1 before;
  memset(&report, 0x5a, sizeof(report));
  memcpy(&before, &report, sizeof(before));

  CHECK(et_amd4_aot_report_size() == (int64_t)sizeof(et_amd4_report_v1));
  CHECK(et_amd4_aot_probe_live(ET_AMD4_ALL_DEVICES, NULL) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_live(-2, &report) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_scenario(0, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_scenario(5, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_scenario(-1, ET_AMD4_ALL_DEVICES, &report) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE,
                                   ET_AMD4_ALL_DEVICES,
                                   NULL) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE, -2,
                                   &report) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(memcmp(&report, &before, sizeof(report)) == 0);

  CHECK(et_amd4_aot_report_status(NULL) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_report_source(NULL) == -1);
  CHECK(et_amd4_aot_report_kfd(NULL) == -1);
  CHECK(et_amd4_aot_report_hip_library(NULL) == -1);
  CHECK(et_amd4_aot_report_rocblas_library(NULL) == -1);
  CHECK(et_amd4_aot_report_device_count(NULL) == -1);
  CHECK(et_amd4_aot_report_reported_device_count(NULL) == -1);
  CHECK(et_amd4_aot_report_device_compute_units(NULL, 0) == -1);
  CHECK(et_amd4_aot_report_device_wavefront_size(NULL, 0) == -1);
  CHECK(et_amd4_aot_report_device_total_vram(NULL, 0) == -1);
  CHECK(et_amd4_aot_report_device_free_vram(NULL, 0) == -1);
  CHECK(et_amd4_aot_read_byte(NULL, 0) == -1);
  CHECK(et_amd4_aot_read_byte("abc", -1) == -1);

  /* Short report buffer and invalid format arguments leave storage intact. */
  CHECK(et_amd4_aot_probe_scenario(ET_AMD4_AOT_SCENARIO_TWO_DEVICE, 0,
                                   &report) == ET_AMD4_OK);
  char short_buf[16];
  char sentinel[16];
  memset(short_buf, 0x3c, sizeof(short_buf));
  memcpy(sentinel, short_buf, sizeof(sentinel));
  int64_t required = 99;
  CHECK(et_amd4_aot_format_report(NULL, short_buf, sizeof(short_buf),
                                  &required) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_format_report(&report, NULL, 1, &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_format_report(&report, short_buf, -1, &required) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_format_report(&report, short_buf, sizeof(short_buf),
                                  NULL) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_format_report(&report, (char *)&required, sizeof(required),
                                  &required) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(required == 99 && memcmp(short_buf, sentinel, sizeof(short_buf)) == 0);

  CHECK(et_amd4_aot_format_report(&report, short_buf, sizeof(short_buf),
                                  &required) == ET_AMD4_ERR_OUT_OF_RANGE);
  CHECK(required > (int64_t)sizeof(short_buf));
  CHECK(memcmp(short_buf, sentinel, sizeof(short_buf)) == 0);

  /* Canary marshaling bounds. */
  uint32_t word = 0x11223344u;
  CHECK(et_amd4_aot_store_u32(NULL, 0, 1) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_store_u32(&word, -1, 1) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_store_u32(&word, 0, -1) == ET_AMD4_ERR_INVALID_ARG);
  CHECK(et_amd4_aot_store_u32(&word, 0, INT64_C(0x100000000)) ==
        ET_AMD4_ERR_INVALID_ARG);
  CHECK(word == 0x11223344u);
  CHECK(et_amd4_aot_load_u32(NULL, 0) == -1);
  CHECK(et_amd4_aot_load_u32(&word, -1) == -1);
  CHECK(et_amd4_aot_store_u32(&word, 0, INT64_C(0xdeadbeef)) == ET_AMD4_OK);
  CHECK(et_amd4_aot_load_u32(&word, 0) == INT64_C(0xdeadbeef));
}

static void test_k1_capability_and_rejection(void) {
  et_kernel_runtime *runtime = NULL;
  CHECK(et_amd4_aot_k1_open(NULL) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_amd4_aot_k1_capability_count(NULL) == -1);
  CHECK(et_amd4_aot_k1_capability_status(NULL, ET_AMD4_AOT_CAP_GEMM) == -1);

  CHECK(et_amd4_aot_k1_open(&runtime) == ET_KERNEL_ERROR_NONE &&
        runtime != NULL);
  CHECK(et_amd4_aot_k1_capability_count(runtime) == 13);
  CHECK(et_amd4_aot_k1_capability_status(runtime, ET_AMD4_AOT_CAP_GEMM) ==
        (int32_t)ET_KERNEL_CAPABILITY_UNAVAILABLE);
  CHECK(et_amd4_aot_k1_capability_status(runtime, ET_AMD4_AOT_CAP_ATTENTION) ==
        (int32_t)ET_KERNEL_CAPABILITY_UNAVAILABLE);
  CHECK(et_amd4_aot_k1_capability_status(runtime, -1) == -1);
  CHECK(et_amd4_aot_k1_capability_status(runtime, 2) == -1);

  uint32_t input = 0x3f800000u;
  uint32_t output = 0xdeadbeefu;
  int32_t code = 0;

  /* Reject both AMD4 capabilities explicitly without touching output canary. */
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM, &input, &output,
                                &code) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(code == (int32_t)ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  CHECK(output == 0xdeadbeefu && input == 0x3f800000u);

  code = 0;
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_ATTENTION, &input,
                                &output, &code) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(code == (int32_t)ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  CHECK(output == 0xdeadbeefu && input == 0x3f800000u);

  /* Null, invalid capability, aliasing, and overlap error paths. */
  CHECK(et_amd4_aot_k1_dispatch(NULL, ET_AMD4_AOT_CAP_GEMM, &input, &output,
                                &code) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(code == (int32_t)ET_KERNEL_CODE_NULL_ARGUMENT);
  CHECK(et_amd4_aot_k1_dispatch(runtime, 99, &input, &output, &code) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(code == (int32_t)ET_KERNEL_CODE_INVALID_ENUM);
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM, NULL, &output,
                                &code) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(code == (int32_t)ET_KERNEL_CODE_NULL_ARGUMENT);
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM, &input, NULL,
                                &code) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(code == (int32_t)ET_KERNEL_CODE_NULL_ARGUMENT);
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM, &output, &output,
                                &code) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(code == (int32_t)ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(et_amd4_aot_k1_dispatch(runtime, ET_AMD4_AOT_CAP_GEMM, &input, &output,
                                (int32_t *)&output) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(output == 0xdeadbeefu && input == 0x3f800000u);

  /* K1 JSON report round-trip and short-buffer negative check. */
  int64_t required = 0;
  CHECK(et_amd4_aot_k1_report_json(NULL, NULL, 0, &required) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_amd4_aot_k1_report_json(runtime, NULL, 1, &required) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_amd4_aot_k1_report_json(runtime, NULL, 0, &required) ==
        ET_KERNEL_ERROR_NONE);
  CHECK(required > 16);
  char short_json[8];
  memset(short_json, 'x', sizeof(short_json));
  int64_t short_req = 0;
  CHECK(et_amd4_aot_k1_report_json(runtime, short_json, sizeof(short_json),
                                   &short_req) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(short_req == required);

  char *json = (char *)malloc((size_t)required);
  CHECK(json != NULL);
  CHECK(et_amd4_aot_k1_report_json(runtime, json, required, &required) ==
        ET_KERNEL_ERROR_NONE);
  CHECK(strstr(json,
               "\"name\":\"kernel.amd-hip.gemm\",\"status\":\"unsupported\"") !=
        NULL);
  CHECK(
      strstr(
          json,
          "\"name\":\"kernel.amd-hip.attention\",\"status\":\"unsupported\"") !=
      NULL);
  free(json);

  et_amd4_aot_k1_close(runtime);
}

int main(void) {
  test_live_probe();
  test_injected_scenarios();
  test_negative_arguments();
  test_k1_capability_and_rejection();
  printf("AMD4 AOT bridge C tests PASS: %u checks\n", checks);
  return EXIT_SUCCESS;
}
