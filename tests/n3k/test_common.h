#ifndef ET_N3K_TEST_COMMON_H
#define ET_N3K_TEST_COMMON_H

#include "eshkol_transformer/n3k_primitives_abi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
static size_t checks;
#define CHECK(condition) do { \
  checks++; \
  if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    exit(1); \
  } \
} while (0)

static inline uint32_t fbits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static inline int64_t signed_bits(uint64_t value) {
  int64_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static inline int nearf(float a, float b, float tolerance) {
  return isfinite(a) && isfinite(b) && fabsf(a - b) <= tolerance;
}

static inline et_kernel_tensor_view_v1 tv(void *data, size_t bytes, const char *dtype,
                                         size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 view = {sizeof(view), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, rank, shape};
  return view;
}

static inline et_kernel_request_v1 rq(const char *operation, size_t rank,
                                     const uint64_t *shape) {
  et_kernel_request_v1 request = {sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  return request;
}

static inline et_kernel_call_v1 kc(const char *capability, const et_kernel_request_v1 *request,
                                  et_kernel_tensor_view_v1 *inputs, size_t input_count,
                                  et_kernel_tensor_view_v1 *outputs, size_t output_count) {
  et_kernel_call_v1 call = {sizeof(call), capability, request,
      input_count, sizeof(*inputs), input_count * sizeof(*inputs), inputs,
      output_count, sizeof(*outputs), output_count * sizeof(*outputs), outputs};
  return call;
}

static inline const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_n3k_kernel_provider_v1();
}

static inline et_kernel_runtime *runtime_new(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static inline void run(et_kernel_runtime *runtime, et_kernel_call_v1 *call) {
  et_kernel_error error;
  if (et_kernel_runtime_dispatch(runtime, call, &error) != 0) {
    fprintf(stderr, "unexpected dispatch error: category=%u code=%u op=%s msg=%s\n",
            error.category, error.code, error.operation, error.message);
    CHECK(0);
  }
}

static inline void reject(et_kernel_runtime *runtime, et_kernel_call_v1 *call,
                          uint32_t category, uint32_t code) {
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, call, &error) != 0);
  CHECK(error.category == category);
  if (code != UINT32_MAX) CHECK(error.code == code);
}

static inline void expect_floats(const float *actual, const float *expected,
                                 size_t count, float tolerance) {
  for (size_t index = 0; index < count; index++) {
    if (!nearf(actual[index], expected[index], tolerance))
      fprintf(stderr, "float mismatch[%zu]: actual %.9g expected %.9g tolerance %.9g\n",
              index, actual[index], expected[index], tolerance);
    CHECK(nearf(actual[index], expected[index], tolerance));
  }
}

static inline double dot(const float *a, const float *b, size_t count) {
  double result = 0.0;
  for (size_t index = 0; index < count; index++) result += (double)a[index] * b[index];
  return result;
}

#endif
