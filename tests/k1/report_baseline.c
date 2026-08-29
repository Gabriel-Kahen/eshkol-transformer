#include "eshkol_transformer/kernel_abi.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0;
  char *report;
  if (et_kernel_runtime_baseline(&runtime, &error) != 0 ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) != 0) {
    fprintf(stderr, "error: %s\n", error.message);
    et_kernel_runtime_destroy(runtime);
    return 1;
  }
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
