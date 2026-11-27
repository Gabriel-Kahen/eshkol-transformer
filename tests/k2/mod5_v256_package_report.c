/* Serialize the runtime that the actual K2 resolver successfully admitted. */
#include "k2_capabilities_internal.h"
#include "eshkol_transformer/kernel_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static et_kernel_runtime *discovered_runtime;
static unsigned discovery_count;

int32_t __real_et_kernel_runtime_discover(
    et_kernel_provider_resolver_v1 resolver, void *context,
    et_kernel_runtime **output, et_kernel_error *error);

int32_t __wrap_et_kernel_runtime_discover(
    et_kernel_provider_resolver_v1 resolver, void *context,
    et_kernel_runtime **output, et_kernel_error *error) {
  int32_t status = __real_et_kernel_runtime_discover(
      resolver, context, output, error);
  if (status == 0) {
    discovered_runtime = *output;
    ++discovery_count;
  }
  return status;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const int opt_in = strcmp(argv[1], "v256") == 0;
  if (!opt_in && strcmp(argv[1], "default") != 0) return 2;

  struct diagnostic_carrier {
    int64_t length;
    et_kernel_error payload;
  } diagnostic = {264, {0}};
  _Static_assert(offsetof(struct diagnostic_carrier, payload) == 8u,
                 "K2 diagnostic carrier header must be one i64");
  int64_t status = opt_in
      ? et_k2_private_mod5_v256_ensure_v1(&diagnostic, diagnostic.length)
      : et_k2_private_runtime_ensure_v1(&diagnostic, diagnostic.length);
  int64_t generation = opt_in ? et_k2_private_mod5_v256_generation_v1()
                              : et_k2_private_runtime_generation_v1();
  if (status != 0 || generation <= 0 || discovery_count != 1u ||
      discovered_runtime == NULL ||
      et_kernel_runtime_capability_count(discovered_runtime) != 11u) {
    fputs("K2 runtime did not admit exactly one eleven-row report\n", stderr);
    return 1;
  }

  et_kernel_error error = {0};
  size_t required = 0;
  if (et_kernel_runtime_report_json(discovered_runtime, NULL, 0u, &required,
                                    &error) != 0 || required == 0u)
    return 1;
  char *report = malloc(required);
  if (report == NULL) return 1;
  status = et_kernel_runtime_report_json(discovered_runtime, report, required,
                                         &required, &error);
  int published = status == 0 && required != 0u &&
                  fwrite(report, 1u, required - 1u, stdout) == required - 1u;
  free(report);
  return published ? 0 : 1;
}
