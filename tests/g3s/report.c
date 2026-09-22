#include "eshkol_transformer/g3s_sampling_abi.h"
#include <string.h>

#include <stdio.h>
#include <stdlib.h>

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
      ? et_g3s_kernel_provider_v1() : NULL;
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0;
  char *report;
  if (et_kernel_runtime_discover(resolve, NULL, &runtime, &error) != 0 ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) != 0) {
    fprintf(stderr, "error: %s\n", error.message);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  if (resolve(NULL, "et_g3s_kernel_provider_v0") != NULL ||
      resolve(NULL, "et_n3k_kernel_provider_v1") != NULL) return 1;
  report = (char *)malloc(required);
  if (report == NULL ||
      et_kernel_runtime_report_json(runtime, report, required, &required,
                                    &error) != 0) {
    fprintf(stderr, "error: cannot create baseline report\n");
    free(report);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  if (fwrite(report, 1u, required - 1u, stdout) != required - 1u) {
    fprintf(stderr, "error: cannot write baseline report\n");
    free(report);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
  free(report);
  et_kernel_runtime_destroy(runtime);
  return 0;
}
