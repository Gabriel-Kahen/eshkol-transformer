#ifndef ET_G3N_TEST_COMMON_H
#define ET_G3N_TEST_COMMON_H

#include "eshkol_transformer/g3n_primitives_abi.h"
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
  return et_g3n_kernel_provider_v1();
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

/* Independently enumerated accepted rows; never inferred from provider metadata. */
#define G3N_ROWS 11u
typedef struct g3n_fixture {
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 inputs[6], outputs[1];
  uint64_t shape[6], input_shapes[6][6], output_shape[6];
  float data[6][1024], output[256];
  int64_t positions[2];
  uint8_t keep;
} g3n_fixture;

static inline void fixture_input(g3n_fixture *f, size_t i, void *data,
                                  const char *dtype, size_t rank,
                                  const uint64_t *shape) {
  size_t elements = 1u;
  memcpy(f->input_shapes[i], shape, rank * sizeof(*shape));
  for (size_t d = 0; d < rank; ++d) elements *= (size_t)shape[d];
  f->inputs[i] = tv(data, elements * (strcmp(dtype, "i64") == 0 ? 8u :
                    strcmp(dtype, "bool") == 0 ? 1u : 4u), dtype,
                    rank, f->input_shapes[i]);
}

static inline void fixture_init(g3n_fixture *f, size_t row) {
  static const char *const caps[G3N_ROWS] = {
    "g3n.embedding-forward", "g3n.embedding-forward",
    "g3n.linear-forward", "g3n.linear-forward", "g3n.linear-forward", "g3n.linear-forward",
    "g3n.layer-norm-forward", "g3n.residual-forward",
    "g3n.head-layout-forward", "g3n.head-layout-forward", "g3n.causal-attention-forward"};
  static const char *const ops[G3N_ROWS] = {
    "g3n.embedding.forward", "g3n.embedding.forward",
    "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias",
    "g3n.layer-norm.forward", "g3n.residual.forward",
    "g3n.heads.split.forward", "g3n.heads.merge.forward", "g3n.causal-attention.forward"};
  static const uint64_t shapes[G3N_ROWS][6] = {
    {1,1,256,4}, {1,1,2,4}, {1,1,4,4}, {1,1,4,8}, {1,1,8,4}, {1,1,4,256},
    {1,1,4}, {1,1,4}, {1,1,2,2}, {1,1,2,2}, {1,2,2,1,1,2}};
  const uint64_t token[] = {1,1,4}, head[] = {1,2,1,2}, pos[] = {1,1};
  size_t rank = row == 10u ? 6u : (row == 6u || row == 7u ? 3u : 4u);
  size_t count = 2u, out_rank = 3u, out_n = 4u;
  CHECK(row < G3N_ROWS);
  memset(f, 0, sizeof(*f));
  memcpy(f->shape, shapes[row], sizeof(f->shape));
  memcpy(f->output_shape, token, sizeof(token));
  for (size_t i = 0; i < 6u; ++i)
    for (size_t j = 0; j < 1024u; ++j)
      f->data[i][j] = (float)((int)((j * 7u + i * 3u) % 23u) - 11) / 16.0f;
  if (row < 2u) {
    uint64_t weight[] = {f->shape[2],4};
    fixture_input(f, 0, &f->positions[0], "i64", 2, pos);
    fixture_input(f, 1, f->data[1], "f32", 2, weight);
  } else if (row < 6u) {
    uint64_t x[] = {1,1,f->shape[2]}, w[] = {f->shape[3],f->shape[2]};
    fixture_input(f, 0, f->data[0], "f32", 3, x);
    fixture_input(f, 1, f->data[1], "f32", 2, w);
    out_n = (size_t)f->shape[3]; f->output_shape[2] = out_n;
  } else if (row == 6u) {
    uint64_t channel[] = {4};
    fixture_input(f, 0, f->data[0], "f32", 3, token);
    fixture_input(f, 1, f->data[1], "f32", 1, channel);
    fixture_input(f, 2, f->data[2], "f32", 1, channel);
    fixture_input(f, 3, f->data[3], "f32", 0, channel);
    f->data[3][0] = 1e-5f; count = 4u;
  } else if (row == 7u) {
    fixture_input(f, 0, f->data[0], "f32", 3, token);
    fixture_input(f, 1, f->data[1], "f32", 3, token);
  } else if (row < 10u) {
    fixture_input(f, 0, f->data[0], "f32", row == 8u ? 3u : 4u, row == 8u ? token : head);
    if (row == 8u) { out_rank = 4u; memcpy(f->output_shape, head, sizeof(head)); }
    count = 1u;
  } else {
    for (size_t i = 0; i < 3u; ++i) fixture_input(f, i, f->data[i], "f32", 4, head);
    fixture_input(f, 3, &f->positions[0], "i64", 2, pos);
    fixture_input(f, 4, &f->positions[1], "i64", 2, pos);
    uint64_t mask[] = {1,1,1};
    fixture_input(f, 5, &f->keep, "bool", 3, mask);
    f->keep = 1u; count = 6u; out_rank = 4u;
    memcpy(f->output_shape, head, sizeof(head));
  }
  f->outputs[0] = tv(f->output, out_n * sizeof(float), "f32", out_rank, f->output_shape);
  f->request = rq(ops[row], rank, f->shape);
  f->call = kc(caps[row], &f->request, f->inputs, count, f->outputs, 1u);
}

#endif
