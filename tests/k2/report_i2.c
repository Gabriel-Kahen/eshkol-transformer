#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/kernel_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int resolver_sentinel;

static const et_kernel_provider_v1 *resolve_i2(void *context,
                                               const char *symbol_name) {
  if (context != (void *)&resolver_sentinel || symbol_name == NULL ||
      strcmp(symbol_name, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0) {
    return NULL;
  }
  return et_f32_tensor_provider_v1();
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  size_t required = 0u;
  char *report = NULL;
  int status = 1;

  if (et_kernel_runtime_discover(resolve_i2, (void *)&resolver_sentinel,
                                 &runtime, &error) != 0 ||
      et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) !=
          0) {
    fprintf(stderr, "error: %s\n", error.message);
    goto cleanup;
  }
  report = (char *)malloc(required);
  if (report == NULL || et_kernel_runtime_report_json(runtime, report, required,
                                                      &required, &error) != 0) {
    fprintf(stderr, "error: cannot create I2 report\n");
    goto cleanup;
  }
  if (required == 0u ||
      fwrite(report, 1u, required - 1u, stdout) != required - 1u) {
    fprintf(stderr, "error: cannot write I2 report\n");
    goto cleanup;
  }
  status = 0;

cleanup:
  free(report);
  et_kernel_runtime_destroy(runtime);
  return status;
}
