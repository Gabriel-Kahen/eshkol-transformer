#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve(void *context, const char *name) {
  (void)context;
  return strcmp(name, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_e3_metrics_kernel_provider_v1() : NULL;
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0;
  if (et_kernel_runtime_discover(resolve, NULL, &runtime, &error) != 0 ||
      et_e3_metrics_kernel_provider_v1()->capability_count != 1u ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) != 0) {
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  char *report = malloc(required);
  int failed = report == NULL;
  if (!failed) failed = et_kernel_runtime_report_json(runtime, report, required,
                                                      &required, &error) != 0;
  if (!failed) failed = fwrite(report, 1u, required - 1u, stdout) != required - 1u;
  free(report);
  et_kernel_runtime_destroy(runtime);
  return failed;
}
