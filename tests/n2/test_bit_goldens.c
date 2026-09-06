#include "bit_goldens.h"
#include "eshkol_transformer/n2_primitives_abi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static size_t checks;

#define CHECK(x)                                                               \
  do {                                                                         \
    checks++;                                                                  \
    if (!(x)) {                                                                \
      fprintf(stderr, "BIT-GOLDEN FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                     const char *dtype, size_t rank,
                                     const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {sizeof(result),
                                     data,
                                     bytes,
                                     dtype,
                                     "cpu",
                                     ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                     0u,
                                     rank,
                                     shape};
  return result;
}

static et_kernel_request_v1 request(const char *operation, size_t rank,
                                    const uint64_t *shape) {
  et_kernel_request_v1 result = {sizeof(result), operation, "f32", "cpu",
                                 rank,           shape,     1u,    {0}};
  return result;
}

static et_kernel_call_v1 call(const char *capability,
                              const et_kernel_request_v1 *req,
                              et_kernel_tensor_view_v1 *inputs, size_t ni,
                              et_kernel_tensor_view_v1 *outputs, size_t no) {
  et_kernel_call_v1 result = {sizeof(result),
                              capability,
                              req,
                              ni,
                              sizeof(*inputs),
                              ni * sizeof(*inputs),
                              inputs,
                              no,
                              sizeof(*outputs),
                              no * sizeof(*outputs),
                              outputs};
  return result;
}

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_n2_kernel_provider_v1();
}

static void dispatch(et_kernel_runtime *runtime,
                     et_kernel_call_v1 *invocation) {
  et_kernel_error error;
  if (et_kernel_runtime_dispatch(runtime, invocation, &error) != 0) {
    fprintf(stderr, "BIT-GOLDEN dispatch failure %u/%u: %s\n", error.category,
            error.code, error.message);
    CHECK(0);
  }
}

static void exact_f32(const char *name, const float *actual,
                      const uint32_t *expected, size_t count) {
  if (memcmp(actual, expected, count * sizeof(*actual)) != 0) {
    for (size_t i = 0; i < count; i++) {
      uint32_t bits;
      memcpy(&bits, &actual[i], sizeof(bits));
      if (bits != expected[i]) {
        fprintf(stderr,
                "BIT-GOLDEN mismatch %s[%zu]: got 0x%08x expected 0x%08x\n",
                name, i, bits, expected[i]);
        break;
      }
    }
  }
  CHECK(memcmp(actual, expected, count * sizeof(*actual)) == 0);
}

static uint64_t hash_bytes(const void *data, size_t bytes) {
  const unsigned char *cursor = (const unsigned char *)data;
  uint64_t hash = UINT64_C(14695981039346656037);
  for (size_t i = 0; i < bytes; i++) {
    hash ^= cursor[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static void exact_hash(const char *name, const void *data, size_t bytes,
                       uint64_t expected) {
  const uint64_t actual = hash_bytes(data, bytes);
#if defined(ET_N2_EMIT_BIT_GOLDENS)
  printf("%s 0x%016llx\n", name, (unsigned long long)actual);
#else
  if (actual != expected)
    fprintf(stderr,
            "BIT-GOLDEN hash mismatch %s: got 0x%016llx expected "
            "0x%016llx\n",
            name, (unsigned long long)actual, (unsigned long long)expected);
  CHECK(actual == expected);
#endif
}

static void test_layer_norm_boundaries(et_kernel_runtime *runtime) {
  {
    uint64_t shape[] = {1, 1, 1}, d_shape[] = {1};
    float x[] = {7}, gamma[] = {1.25f}, beta[] = {-.375f};
    float upstream[] = {1.5f}, epsilon = 1e-5f;
    float y[1], dx[1], dgamma[1], dbeta[1];
    et_kernel_tensor_view_v1 inputs[] = {
        view(x, sizeof(x), "f32", 3, shape),
        view(gamma, sizeof(gamma), "f32", 1, d_shape),
        view(beta, sizeof(beta), "f32", 1, d_shape),
        view(&epsilon, sizeof(epsilon), "f32", 0, NULL)};
    et_kernel_tensor_view_v1 outputs[] = {
        view(y, sizeof(y), "f32", 3, shape),
        view(dgamma, sizeof(dgamma), "f32", 1, d_shape),
        view(dbeta, sizeof(dbeta), "f32", 1, d_shape)};
    et_kernel_request_v1 req = request("layer-norm.forward", 3, shape);
    et_kernel_call_v1 invocation =
        call("kernel.norm", &req, inputs, 4, outputs, 1);
    dispatch(runtime, &invocation);
    req.operation = "layer-norm.backward";
    inputs[2] = view(&epsilon, sizeof(epsilon), "f32", 0, NULL);
    inputs[3] = view(upstream, sizeof(upstream), "f32", 3, shape);
    outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
    invocation.output_count = 3;
    invocation.output_bytes = sizeof(outputs);
    dispatch(runtime, &invocation);
    exact_hash("et_n2_hash_layer_norm_d1_y", y, sizeof(y),
               et_n2_hash_layer_norm_d1_y);
    exact_hash("et_n2_hash_layer_norm_d1_dx", dx, sizeof(dx),
               et_n2_hash_layer_norm_d1_dx);
    exact_hash("et_n2_hash_layer_norm_d1_dgamma", dgamma, sizeof(dgamma),
               et_n2_hash_layer_norm_d1_dgamma);
    exact_hash("et_n2_hash_layer_norm_d1_dbeta", dbeta, sizeof(dbeta),
               et_n2_hash_layer_norm_d1_dbeta);
  }
  {
    uint64_t shape[] = {1, 1, 3}, d_shape[] = {3};
    float x[] = {3, 3, 3}, gamma[] = {.75f, -1.25f, 2};
    float beta[] = {-.2f, .1f, .3f}, upstream[] = {.5f, -1, 2};
    float epsilon = 1e-5f, y[3], dx[3], dgamma[3], dbeta[3];
    et_kernel_tensor_view_v1 inputs[] = {
        view(x, sizeof(x), "f32", 3, shape),
        view(gamma, sizeof(gamma), "f32", 1, d_shape),
        view(beta, sizeof(beta), "f32", 1, d_shape),
        view(&epsilon, sizeof(epsilon), "f32", 0, NULL)};
    et_kernel_tensor_view_v1 outputs[] = {
        view(y, sizeof(y), "f32", 3, shape),
        view(dgamma, sizeof(dgamma), "f32", 1, d_shape),
        view(dbeta, sizeof(dbeta), "f32", 1, d_shape)};
    et_kernel_request_v1 req = request("layer-norm.forward", 3, shape);
    et_kernel_call_v1 invocation =
        call("kernel.norm", &req, inputs, 4, outputs, 1);
    dispatch(runtime, &invocation);
    req.operation = "layer-norm.backward";
    inputs[2] = view(&epsilon, sizeof(epsilon), "f32", 0, NULL);
    inputs[3] = view(upstream, sizeof(upstream), "f32", 3, shape);
    outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
    invocation.output_count = 3;
    invocation.output_bytes = sizeof(outputs);
    dispatch(runtime, &invocation);
    exact_hash("et_n2_hash_layer_norm_constant_y", y, sizeof(y),
               et_n2_hash_layer_norm_constant_y);
    exact_hash("et_n2_hash_layer_norm_constant_dx", dx, sizeof(dx),
               et_n2_hash_layer_norm_constant_dx);
    exact_hash("et_n2_hash_layer_norm_constant_dgamma", dgamma, sizeof(dgamma),
               et_n2_hash_layer_norm_constant_dgamma);
    exact_hash("et_n2_hash_layer_norm_constant_dbeta", dbeta, sizeof(dbeta),
               et_n2_hash_layer_norm_constant_dbeta);
  }
}

#if defined(ET_N2_EMIT_BIT_GOLDENS)
static void emit_f32(const char *name, const float *values, size_t count) {
  printf("static const uint32_t %s[%zu] = {\n", name, count);
  for (size_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, &values[i], sizeof(bits));
    printf("UINT32_C(0x%08x)%s", bits, i + 1u == count ? "" : ",");
    printf("%s", i % 6u == 5u || i + 1u == count ? "\n" : " ");
  }
  printf("};\n");
}

static void emit_i64(const char *name, const int64_t *values, size_t count) {
  printf("static const int64_t %s[%zu] = {\n", name, count);
  for (size_t i = 0; i < count; i++)
    printf("INT64_C(%lld)%s%s", (long long)values[i],
           i + 1u == count ? "" : ",", i + 1u == count ? "\n" : " ");
  printf("};\n");
}
#endif

static void test_embedding(et_kernel_runtime *runtime) {
  uint64_t semantic[] = {2, 3, 5, 6}, ids_shape[] = {2, 3};
  uint64_t weight_shape[] = {5, 6}, y_shape[] = {2, 3, 6};
  int64_t ids[] = {1, 1, 1, 1, 1, 1};
  float weight[30], upstream[36], y[36], dweight[30];
  for (size_t i = 0; i < COUNT(weight); i++)
    weight[i] = (float)((int)(i % 11u) - 5) * 0.125f;
  for (size_t i = 0; i < COUNT(upstream); i++)
    upstream[i] = (float)((int)(i % 7u) - 3) * 0.2f;
  upstream[0] = 0x1.5af1d8p+66f;
  upstream[6] = -0x1.5af1d8p+66f;
  upstream[12] = 3.0f;
  upstream[18] = 1.0f;
  upstream[24] = 2.0f;
  upstream[30] = 4.0f;
  et_kernel_tensor_view_v1 inputs[] = {
      view(ids, sizeof(ids), "i64", 2, ids_shape),
      view(weight, sizeof(weight), "f32", 2, weight_shape)};
  et_kernel_tensor_view_v1 outputs[] = {view(y, sizeof(y), "f32", 3, y_shape)};
  et_kernel_request_v1 req = request("embedding.forward", 4, semantic);
  et_kernel_call_v1 invocation =
      call("kernel.embedding-forward", &req, inputs, 2, outputs, 1);
  dispatch(runtime, &invocation);
  inputs[1] = view(upstream, sizeof(upstream), "f32", 3, y_shape);
  outputs[0] = view(dweight, sizeof(dweight), "f32", 2, weight_shape);
  req.operation = "embedding.backward";
  invocation.capability = "kernel.embedding-backward";
  dispatch(runtime, &invocation);
  exact_hash("et_n2_hash_embedding_y", y, sizeof(y), et_n2_hash_embedding_y);
  exact_hash("et_n2_hash_embedding_dweight", dweight, sizeof(dweight),
             et_n2_hash_embedding_dweight);
  for (size_t row = 0; row < 5; row++)
    if (row != 1u)
      for (size_t d = 0; d < 6; d++)
        CHECK(dweight[row * 6u + d] == 0.0f);
}

static void test_linear(et_kernel_runtime *runtime) {
  uint64_t semantic[] = {2, 3, 4, 5}, x_shape[] = {2, 3, 4};
  uint64_t weight_shape[] = {5, 4}, y_shape[] = {2, 3, 5};
  float x[24], weight[20], bias[5], upstream[30], y[30], bias_y[30], dx[24];
  float dweight[20], dbias[5], no_bias_dx[24], no_bias_dweight[20];
  for (size_t i = 0; i < COUNT(x); i++)
    x[i] = (float)((int)(i % 9u) - 4) * 0.25f;
  x[0] = 0x1.5af1d8p+66f;
  x[4] = -0x1.5af1d8p+66f;
  x[8] = 3.0f;
  x[12] = 1.0f;
  x[16] = 2.0f;
  x[20] = 4.0f;
  for (size_t i = 0; i < COUNT(weight); i++)
    weight[i] = (float)((int)(i % 7u) - 3) * 0.125f;
  for (size_t i = 0; i < COUNT(upstream); i++)
    upstream[i] = i % 5u == 0u ? 1.0f : (float)((int)(i % 5u) - 2) * 0.25f;
  for (size_t i = 0; i < COUNT(bias); i++)
    bias[i] = (float)((int)i - 2) * 0.0625f;
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 3, x_shape),
      view(weight, sizeof(weight), "f32", 2, weight_shape),
      view(upstream, sizeof(upstream), "f32", 3, y_shape)};
  et_kernel_tensor_view_v1 outputs[] = {
      view(y, sizeof(y), "f32", 3, y_shape),
      view(dweight, sizeof(dweight), "f32", 2, weight_shape),
      view(dbias, sizeof(dbias), "f32", 1, semantic + 3)};
  et_kernel_request_v1 req = request("linear.forward-no-bias", 4, semantic);
  et_kernel_call_v1 invocation =
      call("kernel.matmul", &req, inputs, 2, outputs, 1);
  dispatch(runtime, &invocation);
  inputs[2] = view(bias, sizeof(bias), "f32", 1, semantic + 3);
  outputs[0] = view(bias_y, sizeof(bias_y), "f32", 3, y_shape);
  req.operation = "linear.forward";
  invocation.input_count = 3;
  invocation.input_bytes = sizeof(inputs);
  dispatch(runtime, &invocation);
  inputs[2] = view(upstream, sizeof(upstream), "f32", 3, y_shape);
  req.operation = "linear.backward";
  outputs[0] = view(dx, sizeof(dx), "f32", 3, x_shape);
  invocation.input_count = 3;
  invocation.input_bytes = sizeof(inputs);
  invocation.output_count = 3;
  invocation.output_bytes = sizeof(outputs);
  dispatch(runtime, &invocation);
  req.operation = "linear.backward-no-bias";
  outputs[0] = view(no_bias_dx, sizeof(no_bias_dx), "f32", 3, x_shape);
  outputs[1] =
      view(no_bias_dweight, sizeof(no_bias_dweight), "f32", 2, weight_shape);
  invocation.output_count = 2;
  invocation.output_bytes = 2u * sizeof(*outputs);
  dispatch(runtime, &invocation);
  CHECK(memcmp(no_bias_dx, dx, sizeof(dx)) == 0);
  CHECK(memcmp(no_bias_dweight, dweight, sizeof(dweight)) == 0);
#if defined(ET_N2_EMIT_BIT_GOLDENS)
  emit_f32("et_n2_bits_linear_y", y, COUNT(y));
  emit_f32("et_n2_bits_linear_dweight", dweight, COUNT(dweight));
#else
  exact_f32("linear.y", y, et_n2_bits_linear_y, COUNT(y));
  exact_f32("linear.dweight", dweight, et_n2_bits_linear_dweight,
            COUNT(dweight));
#endif
  exact_hash("et_n2_hash_linear_bias_y", bias_y, sizeof(bias_y),
             et_n2_hash_linear_bias_y);
  exact_hash("et_n2_hash_linear_dx", dx, sizeof(dx), et_n2_hash_linear_dx);
  exact_hash("et_n2_hash_linear_dbias", dbias, sizeof(dbias),
             et_n2_hash_linear_dbias);
}

static void test_layer_norm(et_kernel_runtime *runtime) {
  uint64_t shape[] = {2, 2, 4}, d_shape[] = {4};
  float x[] = {1.0f,     1.0001f, .9999f, .0002f + 1.0f, 0x1p24f, 1,
               -0x1p24f, 1,       .25f,   -.75f,         1.25f,   2.5f,
               4,        4.001f,  3.999f, 4.002f};
  float gamma[] = {.75f, -1.25f, .5f, 1.5f};
  float beta[] = {-.2f, .1f, .3f, -.4f};
  float upstream[] = {-.35f, -.175f, 0,    .175f, .35f,  .525f,  -.35f, -.175f,
                      0,     .175f,  .35f, .525f, -.35f, -.175f, 0,     .175f};
  float epsilon = 1e-5f, y[16], dx[16], dgamma[4], dbeta[4];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 3, shape),
      view(gamma, sizeof(gamma), "f32", 1, d_shape),
      view(beta, sizeof(beta), "f32", 1, d_shape),
      view(&epsilon, sizeof(epsilon), "f32", 0, NULL)};
  et_kernel_tensor_view_v1 outputs[] = {
      view(y, sizeof(y), "f32", 3, shape),
      view(dgamma, sizeof(dgamma), "f32", 1, d_shape),
      view(dbeta, sizeof(dbeta), "f32", 1, d_shape)};
  et_kernel_request_v1 req = request("layer-norm.forward", 3, shape);
  et_kernel_call_v1 invocation =
      call("kernel.norm", &req, inputs, 4, outputs, 1);
  dispatch(runtime, &invocation);
  req.operation = "layer-norm.backward";
  inputs[2] = view(&epsilon, sizeof(epsilon), "f32", 0, NULL);
  inputs[3] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.output_count = 3;
  invocation.output_bytes = sizeof(outputs);
  dispatch(runtime, &invocation);
#if defined(ET_N2_EMIT_BIT_GOLDENS)
  emit_f32("et_n2_bits_layer_norm_y", y, COUNT(y));
  emit_f32("et_n2_bits_layer_norm_dx", dx, COUNT(dx));
#else
  exact_f32("layer-norm.y", y, et_n2_bits_layer_norm_y, COUNT(y));
  exact_f32("layer-norm.dx", dx, et_n2_bits_layer_norm_dx, COUNT(dx));
#endif
  exact_hash("et_n2_hash_layer_norm_dgamma", dgamma, sizeof(dgamma),
             et_n2_hash_layer_norm_dgamma);
  exact_hash("et_n2_hash_layer_norm_dbeta", dbeta, sizeof(dbeta),
             et_n2_hash_layer_norm_dbeta);
}

static void test_gelu(et_kernel_runtime *runtime) {
  uint64_t shape[] = {1, 1, 8};
  float x[] = {-10, -3, -1, -0.0f, 0.0f, .5f, 2, 10};
  float upstream[] = {.25f, .5f, -.75f, 1, -1.25f, 1.5f, -2, 2.5f};
  float y[8], dx[8];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 3, shape),
      view(upstream, sizeof(upstream), "f32", 3, shape)};
  et_kernel_tensor_view_v1 outputs[] = {view(y, sizeof(y), "f32", 3, shape)};
  et_kernel_request_v1 req = request("gelu.forward", 3, shape);
  et_kernel_call_v1 invocation =
      call("kernel.activation", &req, inputs, 1, outputs, 1);
  dispatch(runtime, &invocation);
  req.operation = "gelu.backward";
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.input_count = 2;
  invocation.input_bytes = sizeof(inputs);
  dispatch(runtime, &invocation);
#if defined(ET_N2_EMIT_BIT_GOLDENS)
  emit_f32("et_n2_bits_gelu_y", y, COUNT(y));
  emit_f32("et_n2_bits_gelu_dx", dx, COUNT(dx));
#else
  exact_f32("gelu.y", y, et_n2_bits_gelu_y, COUNT(y));
  exact_f32("gelu.dx", dx, et_n2_bits_gelu_dx, COUNT(dx));
#endif
  req.operation = "relu.forward";
  outputs[0] = view(y, sizeof(y), "f32", 3, shape);
  invocation.input_count = 1;
  invocation.input_bytes = sizeof(*inputs);
  dispatch(runtime, &invocation);
  req.operation = "relu.backward";
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.input_count = 2;
  invocation.input_bytes = sizeof(inputs);
  dispatch(runtime, &invocation);
  exact_hash("et_n2_hash_relu_y", y, sizeof(y), et_n2_hash_relu_y);
  exact_hash("et_n2_hash_relu_dx", dx, sizeof(dx), et_n2_hash_relu_dx);
}

static void test_residual(et_kernel_runtime *runtime) {
  uint64_t shape[] = {1, 3, 4};
  float x[12], branch[12], upstream[12], y[12], dx[12], dbranch[12];
  for (size_t i = 0; i < 12; i++) {
    x[i] = (float)((int)i - 5) * .125f;
    branch[i] = (float)((int)(i % 5u) - 2) * .25f;
    upstream[i] = (float)((int)(i % 7u) - 3) * .1f;
  }
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 3, shape),
      view(branch, sizeof(branch), "f32", 3, shape)};
  et_kernel_tensor_view_v1 outputs[] = {
      view(y, sizeof(y), "f32", 3, shape),
      view(dbranch, sizeof(dbranch), "f32", 3, shape)};
  et_kernel_request_v1 req = request("residual.forward", 3, shape);
  et_kernel_call_v1 invocation =
      call("kernel.residual", &req, inputs, 2, outputs, 1);
  dispatch(runtime, &invocation);
  req.operation = "residual.backward";
  inputs[0] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.input_count = 1;
  invocation.input_bytes = sizeof(*inputs);
  invocation.output_count = 2;
  invocation.output_bytes = sizeof(outputs);
  dispatch(runtime, &invocation);
  exact_hash("et_n2_hash_residual_y", y, sizeof(y), et_n2_hash_residual_y);
  exact_hash("et_n2_hash_residual_dx", dx, sizeof(dx), et_n2_hash_residual_dx);
  exact_hash("et_n2_hash_residual_dbranch", dbranch, sizeof(dbranch),
             et_n2_hash_residual_dbranch);
}

static void test_dropout(et_kernel_runtime *runtime) {
  uint64_t shape[] = {1, 2, 5}, state_shape[] = {4};
  float x[] = {-.6f, -.4f, -.2f, 0, .2f, .4f, .6f, -.6f, -.4f, -.2f};
  float upstream[] = {.125f, .25f,  .375f, .5f,   .125f,
                      .25f,  .375f, .5f,   .125f, .25f};
  float probability = .25f, y[10], dx[10];
  int64_t state[] = {1, 2999170649027065890LL, -8817193942522041720LL,
                     247824715720788526LL};
  int64_t next[4];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 3, shape),
      view(&probability, sizeof(probability), "f32", 0, NULL),
      view(state, sizeof(state), "i64", 1, state_shape)};
  et_kernel_tensor_view_v1 outputs[] = {
      view(y, sizeof(y), "f32", 3, shape),
      view(next, sizeof(next), "i64", 1, state_shape)};
  et_kernel_request_v1 req = request("dropout.train.forward", 3, shape);
  et_kernel_call_v1 invocation =
      call("kernel.dropout", &req, inputs, 3, outputs, 2);
  dispatch(runtime, &invocation);
  req.operation = "dropout.train.backward";
  inputs[0] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.output_count = 1;
  invocation.output_bytes = sizeof(*outputs);
  dispatch(runtime, &invocation);
#if defined(ET_N2_EMIT_BIT_GOLDENS)
  emit_f32("et_n2_bits_dropout_y", y, COUNT(y));
  emit_i64("et_n2_bits_dropout_next", next, COUNT(next));
  emit_f32("et_n2_bits_dropout_dx", dx, COUNT(dx));
#else
  exact_f32("dropout.y", y, et_n2_bits_dropout_y, COUNT(y));
  CHECK(memcmp(next, et_n2_bits_dropout_next, sizeof(next)) == 0);
  exact_f32("dropout.dx", dx, et_n2_bits_dropout_dx, COUNT(dx));
#endif
  req.operation = "dropout.eval.forward";
  inputs[0] = view(x, sizeof(x), "f32", 3, shape);
  outputs[0] = view(y, sizeof(y), "f32", 3, shape);
  invocation.input_count = 1;
  invocation.input_bytes = sizeof(*inputs);
  invocation.output_count = 1;
  invocation.output_bytes = sizeof(*outputs);
  dispatch(runtime, &invocation);
  CHECK(memcmp(y, x, sizeof(y)) == 0);
  exact_hash("et_n2_hash_dropout_eval_y", y, sizeof(y),
             et_n2_hash_dropout_eval_y);
  req.operation = "dropout.eval.backward";
  inputs[0] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  dispatch(runtime, &invocation);
  CHECK(memcmp(dx, upstream, sizeof(dx)) == 0);
  exact_hash("et_n2_hash_dropout_eval_dx", dx, sizeof(dx),
             et_n2_hash_dropout_eval_dx);
  req.operation = "dropout.train.forward";
  inputs[0] = view(x, sizeof(x), "f32", 3, shape);
  inputs[1] = view(&probability, sizeof(probability), "f32", 0, NULL);
  inputs[2] = view(state, sizeof(state), "i64", 1, state_shape);
  outputs[0] = view(y, sizeof(y), "f32", 3, shape);
  outputs[1] = view(next, sizeof(next), "i64", 1, state_shape);
  invocation.input_count = 3;
  invocation.input_bytes = sizeof(inputs);
  invocation.output_count = 2;
  invocation.output_bytes = sizeof(outputs);
  probability = 0.0f;
  dispatch(runtime, &invocation);
  CHECK(memcmp(y, x, sizeof(y)) == 0 && memcmp(next, state, sizeof(next)) == 0);
  exact_hash("et_n2_hash_dropout_pos_zero_y", y, sizeof(y),
             et_n2_hash_dropout_pos_zero_y);
  exact_hash("et_n2_hash_dropout_pos_zero_state", next, sizeof(next),
             et_n2_hash_dropout_pos_zero_state);
  req.operation = "dropout.train.backward";
  inputs[0] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.output_count = 1;
  invocation.output_bytes = sizeof(*outputs);
  dispatch(runtime, &invocation);
  CHECK(memcmp(dx, upstream, sizeof(dx)) == 0);
  exact_hash("et_n2_hash_dropout_pos_zero_dx", dx, sizeof(dx),
             et_n2_hash_dropout_pos_zero_dx);
  probability = -0.0f;
  req.operation = "dropout.train.forward";
  inputs[0] = view(x, sizeof(x), "f32", 3, shape);
  outputs[0] = view(y, sizeof(y), "f32", 3, shape);
  invocation.output_count = 2;
  invocation.output_bytes = sizeof(outputs);
  dispatch(runtime, &invocation);
  CHECK(memcmp(y, x, sizeof(y)) == 0 && memcmp(next, state, sizeof(next)) == 0);
  exact_hash("et_n2_hash_dropout_neg_zero_y", y, sizeof(y),
             et_n2_hash_dropout_neg_zero_y);
  exact_hash("et_n2_hash_dropout_neg_zero_state", next, sizeof(next),
             et_n2_hash_dropout_neg_zero_state);
  req.operation = "dropout.train.backward";
  inputs[0] = view(upstream, sizeof(upstream), "f32", 3, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 3, shape);
  invocation.output_count = 1;
  invocation.output_bytes = sizeof(*outputs);
  dispatch(runtime, &invocation);
  CHECK(memcmp(dx, upstream, sizeof(dx)) == 0);
  exact_hash("et_n2_hash_dropout_neg_zero_dx", dx, sizeof(dx),
             et_n2_hash_dropout_neg_zero_dx);
  probability = .25f;
  req.operation = "dropout.train.forward";
  inputs[0] = view(x, sizeof(x), "f32", 3, shape);
  outputs[0] = view(y, sizeof(y), "f32", 3, shape);
  invocation.output_count = 2;
  invocation.output_bytes = sizeof(outputs);
  state[2] = -1;
  state[3] = -1;
  memset(y, 0xa5, sizeof(y));
  memset(next, 0xa5, sizeof(next));
  {
    float y_snapshot[10];
    int64_t state_snapshot[4];
    et_kernel_error error;
    memcpy(y_snapshot, y, sizeof(y));
    memcpy(state_snapshot, next, sizeof(next));
    CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
    CHECK(error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(memcmp(y, y_snapshot, sizeof(y)) == 0);
    CHECK(memcmp(next, state_snapshot, sizeof(next)) == 0);
  }
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve, NULL, &runtime, &error) == 0);
  test_embedding(runtime);
  test_linear(runtime);
  test_layer_norm(runtime);
  test_layer_norm_boundaries(runtime);
  test_gelu(runtime);
  test_residual(runtime);
  test_dropout(runtime);
  et_kernel_runtime_destroy(runtime);
#if !defined(ET_N2_EMIT_BIT_GOLDENS)
  printf("N2 bit goldens PASS (%zu checks; %s)\n", checks,
         ET_N2_BIT_GOLDEN_PLATFORM);
#endif
  return 0;
}
