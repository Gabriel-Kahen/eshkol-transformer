#include "eshkol_transformer/kernel_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? (const et_kernel_provider_v1 *)context
             : NULL;
}

int main(void) {
  static const et_kernel_dimension_range_v1 maximum[] = {
      {.minimum = 0u, .maximum = UINT64_MAX, .maximum_unbounded = 0u},
  };
  static const et_kernel_dimension_range_v1 ten[] = {
      {.minimum = 0u, .maximum = 10u, .maximum_unbounded = 0u},
  };
  static const et_kernel_dimension_range_v1 twenty[] = {
      {.minimum = 0u, .maximum = 20u, .maximum_unbounded = 0u},
  };
  static const et_kernel_dimension_range_v1 unbounded[] = {
      {.minimum = 0u, .maximum = 7u, .maximum_unbounded = 1u},
  };
  static const et_kernel_dimension_range_v1 later_minimum[] = {
      {.minimum = 1u, .maximum = 2u, .maximum_unbounded = 0u},
  };
  static const et_kernel_shape_range_v1 ranges[] = {
      {.rank = 1u, .dimensions = unbounded},
      {.rank = 1u, .dimensions = twenty},
      {.rank = 1u, .dimensions = later_minimum},
      {.rank = 1u, .dimensions = maximum},
      {.rank = 1u, .dimensions = ten},
  };
  static const char *const operations[] = {"shape-order"};
  static const char *const dtypes[] = {"f32"};
  static const char *const devices[] = {"test-cpu"};
  static const et_kernel_capability_v1 capability = {
      .struct_size = sizeof(et_kernel_capability_v1),
      .name = "test.shape-order",
      .status = ET_KERNEL_CAPABILITY_UNVERIFIED,
      .implementation = "test-only-json",
      .version = "1.1-test",
      .evidence = "TEST-ONLY:SHAPE-ORDER-GOLDEN",
      .operation_count = 1u,
      .operations = operations,
      .dtype_count = 1u,
      .dtypes = dtypes,
      .device_count = 1u,
      .devices = devices,
      .shape_range_count = 5u,
      .shape_ranges = ranges,
  };
  static const et_kernel_provider_v1 provider = {
      .struct_size = sizeof(et_kernel_provider_v1),
      .abi_major = ET_KERNEL_ABI_MAJOR,
      .abi_minor = 1u,
      .name = "test-shape-order-provider",
      .version = "1.1-test",
      .evidence = "TEST-ONLY:SHAPE-ORDER-GOLDEN",
      .capability_count = 1u,
      .capability_stride = sizeof(et_kernel_capability_v1),
      .capability_bytes = sizeof(et_kernel_capability_v1),
      .capabilities = &capability,
  };
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0u;
  char *report;

  if (et_kernel_runtime_discover(resolve_provider, (void *)&provider, &runtime,
                                 &error) != 0 ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) != 0) {
    fprintf(stderr, "error: %s\n", error.message);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  report = (char *)malloc(required);
  if (report == NULL ||
      et_kernel_runtime_report_json(runtime, report, required, &required,
                                    &error) != 0) {
    free(report);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  if (fwrite(report, 1u, required - 1u, stdout) != required - 1u) {
    free(report);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  free(report);
  et_kernel_runtime_destroy(runtime);
  return 0;
}
