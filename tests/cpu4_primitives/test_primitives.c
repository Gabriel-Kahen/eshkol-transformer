#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "eshkol_transformer/n3k_primitives_abi.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct tensor {
  void *data;
  size_t bytes;
  const char *dtype;
  size_t rank;
  uint64_t shape[4];
  const double *expected;
  size_t count;
} tensor;
typedef struct test_case {
  size_t provider;
  const char *capability, *operation;
  size_t rank;
  uint64_t shape[6];
  size_t ni, no;
  tensor input[7], output[3];
} test_case;
#include "reference.h"

static size_t checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)
static const et_kernel_provider_v1 *resolver(void *context, const char *symbol) {
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return context;
}
static et_kernel_tensor_view_v1 view(tensor *t) {
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1), t->data,
    t->bytes, t->dtype, "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0,
    t->rank, t->rank ? t->shape : NULL};
}
static void fill_outputs(test_case *c) {
  for (size_t i = 0; i < c->no; ++i) memset(c->output[i].data, 0x5a, c->output[i].bytes);
}
static void reject_unchanged(et_kernel_runtime *runtime, test_case *c,
                             et_kernel_call_v1 *call, int unsupported) {
  fill_outputs(c);
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, call, &error) != 0);
  if (unsupported) CHECK(error.category == ET_KERNEL_ERROR_UNSUPPORTED);
  for (size_t i = 0; i < c->no; ++i)
    for (size_t j = 0; j < c->output[i].bytes; ++j)
      CHECK(((unsigned char *)c->output[i].data)[j] == 0x5a);
}
static void test(et_kernel_runtime *runtime, test_case *c) {
  et_kernel_tensor_view_v1 inputs[7], outputs[3];
  for (size_t i = 0; i < c->ni; ++i) inputs[i] = view(&c->input[i]);
  for (size_t i = 0; i < c->no; ++i) outputs[i] = view(&c->output[i]);
  et_kernel_request_v1 request = {sizeof(request), c->operation, "f32", "cpu",
    c->rank, c->shape, 1, {0}};
  et_kernel_call_v1 call = {sizeof(call), c->capability, &request,
    c->ni, sizeof(*inputs), c->ni * sizeof(*inputs), inputs,
    c->no, sizeof(*outputs), c->no * sizeof(*outputs), outputs};
  et_kernel_error error;
  if (et_kernel_runtime_dispatch(runtime, &call, &error)) {
    fprintf(stderr, "%s: %s\n", c->operation, error.message);
    CHECK(0);
  }
  for (size_t i = 0; i < c->no; ++i)
    for (size_t j = 0; j < c->output[i].count; ++j) {
      double actual = ((float *)c->output[i].data)[j], expected = c->output[i].expected[j];
      /* Fixed before execution: binary32 primitive vs independent binary64 oracle. */
      if (!(isfinite(actual) && fabs(actual - expected) <= 3e-5 + 3e-4 * fabs(expected)))
        fprintf(stderr, "%s out%zu[%zu]: actual %.9g expected %.12g\n",
                c->operation, i, j, actual, expected);
      CHECK(isfinite(actual) && fabs(actual - expected) <= 3e-5 + 3e-4 * fabs(expected));
    }

  uint64_t wrong[6];
  memcpy(wrong, c->shape, sizeof(wrong));
  request.shape = wrong;
  for (uint64_t t = 3; t <= 5; t += 2) {
    if (c->provider == 2) wrong[3] = wrong[4] = t;
    else wrong[1] = t;
    reject_unchanged(runtime, c, &call, 1);
  }
  request.shape = c->shape;
  --outputs[0].byte_length;
  reject_unchanged(runtime, c, &call, 0);
  ++outputs[0].byte_length;
  const char *dtype = inputs[0].dtype;
  inputs[0].dtype = "f64";
  reject_unchanged(runtime, c, &call, 0);
  inputs[0].dtype = dtype;
  request.device = "cuda";
  reject_unchanged(runtime, c, &call, 1);
  request.device = "cpu";

  for (size_t i = 0; i < c->ni; ++i) if (!strcmp(inputs[i].dtype, "f32")) {
    float *value = inputs[i].data, saved = value[0];
    value[0] = NAN;
    reject_unchanged(runtime, c, &call, 0);
    value[0] = saved;
    break;
  }
  if (strstr(c->operation, "embedding")) {
    int64_t *ids = inputs[0].data, saved = ids[3];
    ids[3] = -1;
    reject_unchanged(runtime, c, &call, 0);
    ids[3] = saved;
  }
  if (c->provider == 2) {
    unsigned char *mask = inputs[5].data, saved = mask[15];
    mask[15] = 2;
    reject_unchanged(runtime, c, &call, 0);
    mask[15] = saved;
  }
  if (!strcmp(c->capability, "n3k.head-layout")) {
    /* Same bytes but transposed metadata must fail, not silently transpose twice. */
    et_kernel_tensor_view_v1 *head = inputs[0].rank == 4 ? &inputs[0] : &outputs[0];
    const uint64_t transposed[] = {1, 4, 2, 2};
    const uint64_t *saved = head->shape;
    head->shape = transposed;
    reject_unchanged(runtime, c, &call, 0);
    head->shape = saved;
    const uint64_t tensor_as_request[] = {1, 2, 4, 2};
    request.shape = tensor_as_request;
    reject_unchanged(runtime, c, &call, 1);
    request.shape = c->shape;
  }
}
int main(void) {
  const et_kernel_provider_v1 *providers[] = {et_n3k_kernel_provider_v1(),
    et_n2_kernel_provider_v1(), et_a2_kernel_provider_v1()};
  et_kernel_runtime *runtime[3] = {0};
  et_kernel_error error;
  for (size_t i = 0; i < 3; ++i)
    CHECK(et_kernel_runtime_discover(resolver, (void *)providers[i], &runtime[i], &error) == 0);
  for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i) test(runtime[cases[i].provider], &cases[i]);
  for (size_t i = 0; i < 3; ++i) et_kernel_runtime_destroy(runtime[i]);
  printf("CPU4 primitives PASS: %zu cases, %zu numerical/negative checks\n",
         sizeof(cases) / sizeof(*cases), checks);
  return 0;
}
