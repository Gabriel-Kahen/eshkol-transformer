#include "eshkol_transformer/g3n_primitives_abi.h"

#ifndef G3N_REPORT_ACCESSOR
#define G3N_REPORT_ACCESSOR et_g3n_kernel_provider_v1
#endif
extern const et_kernel_provider_v1 *G3N_REPORT_ACCESSOR(void);
static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context; (void)symbol; return G3N_REPORT_ACCESSOR();
}

#include <stdio.h>
#include <stdlib.h>

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
