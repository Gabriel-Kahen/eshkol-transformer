#ifndef ET_G3C4_TEST_COMMON_H
#define ET_G3C4_TEST_COMMON_H

#include "eshkol_transformer/g3c4_primitives_abi.h"
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
  return et_g3c4_kernel_provider_v1();
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
  for (size_t index = 0; index < count; ++index) {
    if (!nearf(actual[index], expected[index], tolerance))
      fprintf(stderr, "float mismatch[%zu]: actual %.9g expected %.9g tolerance %.9g\n",
              index, actual[index], expected[index], tolerance);
    CHECK(nearf(actual[index], expected[index], tolerance));
  }
}

/* Independent literal enumeration: fixture construction never reads provider rows. */
#define G3C4_ROWS 28u
typedef struct g3c4_fixture {
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 inputs[6], outputs[1];
  uint64_t shape[6], input_shapes[6][6], output_shape[6];
  float data[6][2048], output[2048];
  int64_t positions[2][4];
  uint8_t keep[16];
} g3c4_fixture;

static inline void fixture_input(g3c4_fixture *f, size_t i, void *data,
                                  const char *dtype, size_t rank,
                                  const uint64_t *shape) {
  size_t elements = 1u;
  memcpy(f->input_shapes[i], shape, rank * sizeof(*shape));
  for (size_t d = 0; d < rank; ++d) elements *= (size_t)shape[d];
  f->inputs[i] = tv(data, elements * (strcmp(dtype, "i64") == 0 ? 8u :
                    strcmp(dtype, "bool") == 0 ? 1u : 4u), dtype,
                    rank, f->input_shapes[i]);
}

static inline void fixture_init(g3c4_fixture *f, size_t row) {
  static const char *const caps[G3C4_ROWS] = {
    "g3c4.embedding-forward","g3c4.embedding-forward","g3c4.embedding-forward",
    "g3c4.embedding-forward","g3c4.embedding-forward","g3c4.embedding-forward",
    "g3c4.linear","g3c4.linear","g3c4.linear","g3c4.linear",
    "g3c4.linear","g3c4.linear","g3c4.linear","g3c4.linear",
    "g3c4.layer-norm","g3c4.layer-norm","g3c4.gelu","g3c4.gelu",
    "g3c4.residual","g3c4.head-layout","g3c4.head-layout",
    "g3c4.head-layout","g3c4.head-layout",
    "g3c4.causal-attention","g3c4.causal-attention","g3c4.causal-attention",
    "g3c4.causal-attention","g3c4.causal-attention"};
  static const char *const ops[G3C4_ROWS] = {
    "g3c4.embedding.forward","g3c4.embedding.forward","g3c4.embedding.forward",
    "g3c4.embedding.forward","g3c4.embedding.forward","g3c4.embedding.forward",
    "g3c4.linear.forward-no-bias","g3c4.linear.forward-no-bias",
    "g3c4.linear.forward-no-bias","g3c4.linear.forward-no-bias",
    "g3c4.linear.forward-no-bias","g3c4.linear.forward-no-bias",
    "g3c4.linear.forward-no-bias","g3c4.linear.forward-no-bias",
    "g3c4.layer-norm.forward","g3c4.layer-norm.forward",
    "g3c4.gelu.forward","g3c4.gelu.forward","g3c4.residual.forward",
    "g3c4.heads.split.forward","g3c4.heads.split.forward",
    "g3c4.heads.merge.forward","g3c4.heads.merge.forward",
    "g3c4.causal-attention.forward","g3c4.causal-attention.forward",
    "g3c4.causal-attention.forward","g3c4.causal-attention.forward",
    "g3c4.causal-attention.forward"};
  static const uint64_t shapes[G3C4_ROWS][6] = {
    {1,1,4,4},{1,2,4,4},{1,3,4,4},{1,4,4,4},{1,3,256,4},{1,4,256,4},
    {1,3,4,4},{1,3,4,8},{1,3,8,4},{1,3,4,256},
    {1,4,4,4},{1,4,4,8},{1,4,8,4},{1,4,4,256},
    {1,3,4},{1,4,4},{1,3,8},{1,4,8},{1,4,4},
    {1,3,2,2},{1,4,2,2},{1,3,2,2},{1,4,2,2},
    {1,2,2,1,4,2},{1,2,2,2,4,2},{1,2,2,3,4,2},{1,2,2,4,4,2},
    {1,2,2,3,3,2}};
  size_t count = 2u, output_rank = 3u, output_count = 0u;
  CHECK(row < G3C4_ROWS);
  memset(f, 0, sizeof(*f));
  memcpy(f->shape, shapes[row], sizeof(f->shape));
  for (size_t i = 0; i < 6u; ++i)
    for (size_t j = 0; j < 2048u; ++j)
      f->data[i][j] = (float)((int)((j * 7u + i * 3u) % 23u) - 11) / 16.0f;
  if (row < 6u) {
    const uint64_t ids[] = {1,f->shape[1]}, weight[] = {f->shape[2],f->shape[3]};
    for (size_t t = 0; t < (size_t)f->shape[1]; ++t) f->positions[0][t] = (int64_t)(t % f->shape[2]);
    fixture_input(f,0,f->positions[0],"i64",2,ids);
    fixture_input(f,1,f->data[1],"f32",2,weight);
    memcpy(f->output_shape,ids,2*sizeof(uint64_t));f->output_shape[2]=f->shape[3];
    output_count=(size_t)(f->shape[1]*f->shape[3]);
  } else if (row < 14u) {
    const uint64_t x[] = {1,f->shape[1],f->shape[2]}, weight[] = {f->shape[3],f->shape[2]};
    fixture_input(f,0,f->data[0],"f32",3,x);fixture_input(f,1,f->data[1],"f32",2,weight);
    f->output_shape[0]=1;f->output_shape[1]=f->shape[1];f->output_shape[2]=f->shape[3];
    output_count=(size_t)(f->shape[1]*f->shape[3]);
  } else if (row < 16u) {
    const uint64_t x[] = {1,f->shape[1],f->shape[2]}, channel[] = {f->shape[2]};
    fixture_input(f,0,f->data[0],"f32",3,x);fixture_input(f,1,f->data[1],"f32",1,channel);
    fixture_input(f,2,f->data[2],"f32",1,channel);fixture_input(f,3,f->data[3],"f32",0,channel);
    f->data[3][0]=1e-5f;count=4u;memcpy(f->output_shape,x,sizeof(x));output_count=(size_t)(f->shape[1]*f->shape[2]);
  } else if (row < 18u) {
    const uint64_t x[] = {1,f->shape[1],f->shape[2]};
    fixture_input(f,0,f->data[0],"f32",3,x);count=1u;memcpy(f->output_shape,x,sizeof(x));output_count=(size_t)(f->shape[1]*f->shape[2]);
  } else if (row == 18u) {
    const uint64_t x[] = {1,4,4};
    fixture_input(f,0,f->data[0],"f32",3,x);fixture_input(f,1,f->data[1],"f32",3,x);
    memcpy(f->output_shape,x,sizeof(x));output_count=16u;
  } else if (row < 23u) {
    const uint64_t token[] = {1,f->shape[1],4}, head[] = {1,2,f->shape[1],2};
    const int split = row < 21u;
    fixture_input(f,0,f->data[0],"f32",split ? 3u:4u,split ? token:head);
    memcpy(f->output_shape,split ? head:token,(split ? 4u:3u)*sizeof(uint64_t));
    output_rank=split ? 4u:3u;output_count=(size_t)(f->shape[1]*4u);count=1u;
  } else {
    const uint64_t q[] = {1,2,f->shape[3],2}, kv[] = {1,2,f->shape[4],2};
    const uint64_t qp[] = {1,f->shape[3]}, kp[] = {1,f->shape[4]}, mask[] = {1,f->shape[3],f->shape[4]};
    for (size_t t=0;t<(size_t)f->shape[3];++t) f->positions[0][t]=(int64_t)t;
    for (size_t t=0;t<(size_t)f->shape[4];++t) f->positions[1][t]=(int64_t)t;
    memset(f->keep,1,(size_t)(f->shape[3]*f->shape[4]));
    fixture_input(f,0,f->data[0],"f32",4,q);fixture_input(f,1,f->data[1],"f32",4,kv);
    fixture_input(f,2,f->data[2],"f32",4,kv);fixture_input(f,3,f->positions[0],"i64",2,qp);
    fixture_input(f,4,f->positions[1],"i64",2,kp);fixture_input(f,5,f->keep,"bool",3,mask);
    memcpy(f->output_shape,q,sizeof(q));output_rank=4u;output_count=(size_t)(2u*f->shape[3]*2u);count=6u;
  }
  f->outputs[0]=tv(f->output,output_count*sizeof(float),"f32",output_rank,f->output_shape);
  f->request=rq(ops[row],row>=23u ? 6u:(row>=14u && row<=18u ? 3u:4u),f->shape);
  f->call=kc(caps[row],&f->request,f->inputs,count,f->outputs,1u);
}

#endif
