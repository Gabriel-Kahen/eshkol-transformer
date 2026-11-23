#ifndef ESHKOL_TRANSFORMER_MOD5_TEST_H
#define ESHKOL_TRANSFORMER_MOD5_TEST_H

#include "eshkol_transformer/mod5_primitives_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static inline const et_kernel_provider_v1 *resolve_mod5(void *context,
                                                        const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mod5_kernel_provider_v1()
             : NULL;
}

static inline et_kernel_runtime *discover_mod5(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mod5, NULL, &runtime, &error) == 0);
  return runtime;
}

static inline et_kernel_tensor_view_v1
test_view(void *data, const char *dtype, size_t rank, const uint64_t *shape) {
  size_t bytes = strcmp(dtype, "i64") == 0    ? sizeof(int64_t)
                 : strcmp(dtype, "bool") == 0 ? sizeof(uint8_t)
                                              : sizeof(float);
  for (size_t d = 0; d < rank; ++d)
    bytes *= shape[d];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static inline et_kernel_call_v1 test_call(const char *cap,
                                          const et_kernel_request_v1 *request,
                                          size_t ni, const void *in, size_t no,
                                          void *out) {
  const size_t stride = sizeof(et_kernel_tensor_view_v1);
  return (et_kernel_call_v1){sizeof(et_kernel_call_v1),
                             cap,
                             request,
                             ni,
                             stride,
                             ni * stride,
                             in,
                             no,
                             stride,
                             no * stride,
                             out};
}

static inline void dispatch(const et_kernel_runtime *runtime,
                            const et_kernel_call_v1 *call) {
  et_kernel_error error;
  if (et_kernel_runtime_dispatch(runtime, call, &error) != 0) {
    fprintf(stderr, "%s (%u/%u): %s\n", error.operation, error.category,
            error.code, error.message);
    exit(EXIT_FAILURE);
  }
}

static inline void check_close(double actual, double expected, double atol,
                               double rtol) {
  if (!(fabs(actual - expected) <= atol + rtol * fabs(expected))) {
    fprintf(stderr, "observed %.9g, expected %.9g (atol %g, rtol %g)\n", actual,
            expected, atol, rtol);
    exit(EXIT_FAILURE);
  }
}

#endif
