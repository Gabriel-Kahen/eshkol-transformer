/* Development-only M3 preflight: inspect accepted providers, never widen them. */
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include <stdio.h>
#include <string.h>

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0 ? context : NULL;
}

static int row(et_kernel_runtime *runtime, const char *capability,
               const char *operation, size_t rank, const uint64_t *shape,
               int expected) {
  et_kernel_request_v1 request = {
      .struct_size = sizeof(request), .operation = operation, .dtype = "f32",
      .device = "cpu", .rank = rank, .shape = shape, .deterministic = 1};
  et_kernel_error error;
  const et_kernel_capability_v1 *entry = NULL;
  int result = et_kernel_runtime_capability_require(runtime, capability,
                                                   &request, &entry, &error);
  printf("%s [", operation);
  for (size_t index = 0; index < rank; index++)
    printf("%s%llu", index ? "," : "", (unsigned long long)shape[index]);
  printf("] result=%d code=%u\n", result, error.code);
  return result != expected ||
         (expected == 0 ? entry == NULL || error.code != ET_KERNEL_CODE_OK
                        : error.code != ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
}

int main(void) {
  et_kernel_runtime *n2 = NULL, *a2 = NULL, *l2 = NULL;
  et_kernel_error error;
  int failures = 0;
  if (et_kernel_runtime_discover(resolve, (void *)et_n2_kernel_provider_v1(),
                                &n2, &error) ||
      et_kernel_runtime_discover(resolve, (void *)et_a2_kernel_provider_v1(),
                                &a2, &error) ||
      et_kernel_runtime_discover(resolve,
          (void *)et_l2_indexed_cross_entropy_provider_v1(), &l2, &error))
    return 2;
#define ROW(runtime, capability, operation, expected, ...) do { \
  const uint64_t shape[] = {__VA_ARGS__}; \
  failures += row(runtime, capability, operation, sizeof(shape)/sizeof(*shape), \
                  shape, expected); \
} while (0)
  for (int backward = 0; backward < 2; backward++) {
    ROW(n2, backward ? "kernel.embedding-backward" : "kernel.embedding-forward",
        backward ? "embedding.backward" : "embedding.forward", 6, 1,2,256,4);
    ROW(n2, backward ? "kernel.embedding-backward" : "kernel.embedding-forward",
        backward ? "embedding.backward" : "embedding.forward", 6, 1,2,2,4);
    ROW(n2, "kernel.matmul", backward ? "linear.backward" : "linear.forward",
        6, 1,2,4,4);
    ROW(n2, "kernel.matmul", backward ? "linear.backward" : "linear.forward",
        6, 1,2,4,8);
    ROW(n2, "kernel.matmul", backward ? "linear.backward" : "linear.forward",
        6, 1,2,8,4);
    ROW(n2, "kernel.matmul", backward ? "linear.backward-no-bias" : "linear.forward-no-bias",
        6, 1,2,4,4);
    ROW(n2, "kernel.matmul", backward ? "linear.backward-no-bias" : "linear.forward-no-bias",
        6, 1,2,4,8);
    ROW(n2, "kernel.matmul", backward ? "linear.backward-no-bias" : "linear.forward-no-bias",
        6, 1,2,8,4);
    ROW(n2, "kernel.matmul", backward ? "linear.backward-no-bias" : "linear.forward-no-bias",
        6, 1,2,4,256);
    ROW(n2, "kernel.norm", backward ? "layer-norm.backward" : "layer-norm.forward",
        0, 1,2,4);
    ROW(n2, "kernel.activation", backward ? "gelu.backward" : "gelu.forward",
        6, 1,2,8);
    ROW(n2, "kernel.residual", backward ? "residual.backward" : "residual.forward",
        6, 1,2,4);
    ROW(a2, "kernel.causal-attention", backward ? "causal-attention.backward" : "causal-attention.forward",
        0, 1,2,2,2,2,2);
    ROW(a2, "kernel.rope", backward ? "rope.backward" : "rope.forward",
        6, 1,2,2,2);
    ROW(l2, "kernel.indexed-cross-entropy", backward ? "indexed-cross-entropy.backward" : "indexed-cross-entropy.forward",
        0, 1,2,256);
  }
  /* A possible rank-three representation of the tied [256,4] sum is absent. */
  ROW(n2, "kernel.residual", "residual.forward", 6, 1,256,4);
  /* Distinct token/head markers show reshape cannot perform the permutation. */
  const float token_major[8] = {0,1,10,11,100,101,110,111};
  const float expected_head_major[8] = {0,1,100,101,10,11,110,111};
  if (memcmp(token_major, expected_head_major, sizeof(token_major)) == 0)
    failures++;
  puts("layout [1,2,4] -> [1,2,2,2]: token-major index 2=10; required head-major index 2=100");
  et_kernel_runtime_destroy(n2);
  et_kernel_runtime_destroy(a2);
  et_kernel_runtime_destroy(l2);
  return failures != 0;
}
