#include "eshkol_transformer/indexed_cross_entropy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve_l2(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_l2_indexed_cross_entropy_provider_v1()
             : NULL;
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0u;
  char *report;
  if (et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) != 0 ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) != 0) {
    return EXIT_FAILURE;
  }
  report = (char *)malloc(required);
  if (report == NULL ||
      et_kernel_runtime_report_json(runtime, report, required, &required,
                                    &error) != 0 ||
      fwrite(report, 1u, required - 1u, stdout) != required - 1u) {
    free(report);
    et_kernel_runtime_destroy(runtime);
    return EXIT_FAILURE;
  }
  free(report);
  et_kernel_runtime_destroy(runtime);
  return EXIT_SUCCESS;
}
