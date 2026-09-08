/* Diagnostic only: separate resolver contexts are not a production composer. */
#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/n2_primitives_abi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #test); return 1; \
} } while (0)

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0 ? context : NULL;
}

int main(void) {
  const et_kernel_provider_v1 *providers[] = {
    et_i64_tensor_provider_v1(), et_f32_tensor_provider_v1(),
    et_n2_kernel_provider_v1(), et_a2_kernel_provider_v1(),
    et_l2_indexed_cross_entropy_provider_v1()
  };
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  const char *owners[64];
  size_t owner_count = 0;
  size_t baseline_count, verified_total = 0;
  CHECK(et_kernel_runtime_baseline(&runtime, &error) == 0);
  baseline_count = et_kernel_runtime_capability_count(runtime);
  for (size_t i = 0; i < baseline_count; ++i)
    CHECK(et_kernel_runtime_capability_at(runtime, i)->status ==
          ET_KERNEL_CAPABILITY_UNVERIFIED);
  printf("linked-all baseline: %zu unverified, 0 verified\n", baseline_count);
  et_kernel_runtime_destroy(runtime);
  for (size_t p = 0; p < sizeof(providers) / sizeof(providers[0]); ++p) {
    size_t verified = 0;
    CHECK(et_kernel_runtime_discover(resolve, (void *)providers[p],
                                     &runtime, &error) == 0);
    for (size_t i = 0; i < providers[p]->capability_count; ++i) {
      const et_kernel_capability_v1 *cap = (const void *)
          ((const unsigned char *)providers[p]->capabilities +
           i * providers[p]->capability_stride);
      for (size_t previous = 0; previous < owner_count; ++previous)
        CHECK(strcmp(owners[previous], cap->name) != 0);
      CHECK(owner_count < sizeof(owners) / sizeof(owners[0]));
      owners[owner_count++] = cap->name;
    }
    for (size_t i = 0; i < et_kernel_runtime_capability_count(runtime); ++i) {
      const et_kernel_capability_v1 *cap =
          et_kernel_runtime_capability_at(runtime, i);
      if (cap->status != ET_KERNEL_CAPABILITY_VERIFIED) continue;
      ++verified;
      printf("%s: %s [", providers[p]->name, cap->name);
      for (size_t op = 0; op < cap->operation_count; ++op)
        printf("%s%s", op ? "," : "", cap->operations[op]);
      puts("]");
    }
    CHECK(verified == providers[p]->capability_count);
    verified_total += verified;
    et_kernel_runtime_destroy(runtime);
  }
  printf("PASS: 5 independent descriptors, %zu distinct verified entries; "
         "no composition or execution claim\n", verified_total);
  return 0;
}
