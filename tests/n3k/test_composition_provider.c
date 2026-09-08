#include "test_common.h"
#include "composition_reference.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <xmmintrin.h>

/* This suite exercises the real explicit provider. Reference arrays are generated
 * from the independent integer/IEEE-bit Python oracle solely at test build time. */
static const uint64_t token_shape[] = {1, 2, 4};
static const uint64_t head_shape[] = {1, 2, 2, 2};
static const uint64_t tied_shape[] = {256, 4};
static const uint64_t state_shape[] = {4};

static void exact_words(const float *actual, const uint32_t *expected, size_t n) {
  for (size_t i = 0; i < n; ++i) CHECK(fbits(actual[i]) == expected[i]);
}

static void preserved_reject(et_kernel_runtime *rt, et_kernel_call_v1 *call,
                             uint32_t category, uint32_t code) {
  unsigned char input[3][4096], output[3][4096];
  const et_kernel_tensor_view_v1 *in = call->inputs;
  const et_kernel_tensor_view_v1 *out = call->outputs;
  CHECK(call->input_count <= 3 && call->output_count <= 3);
  for (size_t i = 0; i < call->input_count; ++i) {
    CHECK(in[i].byte_length <= sizeof(input[i]));
    memcpy(input[i], in[i].data, in[i].byte_length);
  }
  for (size_t i = 0; i < call->output_count; ++i) {
    CHECK(out[i].byte_length <= sizeof(output[i]));
    memcpy(output[i], out[i].data, out[i].byte_length);
  }
  reject(rt, call, category, code);
  for (size_t i = 0; i < call->input_count; ++i)
    CHECK(memcmp(input[i], in[i].data, in[i].byte_length) == 0);
  for (size_t i = 0; i < call->output_count; ++i)
    CHECK(memcmp(output[i], out[i].data, out[i].byte_length) == 0);
}

static void layout_call(et_kernel_runtime *rt, const char *operation,
                        int token_input, float *x, float *y) {
  et_kernel_request_v1 request = rq(operation, 4, head_shape);
  et_kernel_tensor_view_v1 input = tv(x, 32, "f32", token_input ? 3 : 4,
                                     token_input ? token_shape : head_shape);
  et_kernel_tensor_view_v1 output = tv(y, 32, "f32", token_input ? 4 : 3,
                                      token_input ? head_shape : token_shape);
  et_kernel_call_v1 call = kc("n3k.head-layout", &request, &input, 1, &output, 1);
  run(rt, &call);
}

static void test_layout(et_kernel_runtime *rt) {
  static const char *operations[] = {
      "n3k.heads.split.forward", "n3k.heads.merge.forward",
      "n3k.heads.split.backward", "n3k.heads.merge.backward"};
  static const int token_inputs[] = {1, 0, 0, 1};
  static const size_t permutation[] = {0, 1, 4, 5, 2, 3, 6, 7};
  float x[] = {0, 1, 10, 11, 100, 101, 110, 111};
  float y[8], original[8];
  memcpy(original, x, sizeof(x));
  for (size_t op = 0; op < COUNT(operations); ++op) {
    memset(y, 0xa5, sizeof(y));
    layout_call(rt, operations[op], token_inputs[op], x, y);
    for (size_t i = 0; i < 8; ++i) CHECK(y[i] == x[permutation[i]]);
    CHECK(memcmp(x, original, sizeof(x)) == 0);
    CHECK(memcmp(y, x, sizeof(y)) != 0);
    /* T=H=2 makes the flat permutation self-inverse; ranks distinguish direction. */
    et_kernel_request_v1 request = rq(operations[op], 4, head_shape);
    et_kernel_tensor_view_v1 input = tv(x, 32, "f32", token_inputs[op] ? 4 : 3,
                                       token_inputs[op] ? head_shape : token_shape);
    et_kernel_tensor_view_v1 output = tv(y, 32, "f32", token_inputs[op] ? 4 : 3,
                                        token_inputs[op] ? head_shape : token_shape);
    et_kernel_call_v1 call = kc("n3k.head-layout", &request, &input, 1, &output, 1);
    preserved_reject(rt, &call, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE);
    input = tv(x, 32, "f32", token_inputs[op] ? 3 : 4,
               token_inputs[op] ? token_shape : head_shape);
    output.data = x;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    output.data = y;
    for (size_t invalid = 0; invalid < 3; ++invalid) {
      const uint32_t word = (uint32_t[]){0x7f800000u, 0xff800000u, 0x7f800001u}[invalid];
      memcpy(x + 7, &word, 4);
      preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    }
    memcpy(x, original, sizeof(x));
    for (size_t dimension = 0; dimension < 4; ++dimension) {
      for (int delta = -1; delta <= 1; delta += 2) {
        uint64_t neighbor[] = {1, 2, 2, 2};
        neighbor[dimension] = (uint64_t)((int64_t)neighbor[dimension] + delta);
        request.shape = neighbor;
        preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
      }
    }
  }
  for (size_t direction = 0; direction < 2; ++direction) {
    float upstream[8], gradient[8], plus[8], minus[8];
    for (size_t i = 0; i < 8; ++i) {
      x[i] = (float)((int)i - 3) * 0.125f;
      upstream[i] = (float)(3 * (int)i - 7) * 0.0625f;
    }
    layout_call(rt, operations[direction + 2], !token_inputs[direction],
                upstream, gradient);
    for (size_t i = 0; i < 8; ++i) {
      const float saved = x[i];
      x[i] = saved + 0.03125f;
      layout_call(rt, operations[direction], token_inputs[direction], x, plus);
      x[i] = saved - 0.03125f;
      layout_call(rt, operations[direction], token_inputs[direction], x, minus);
      x[i] = saved;
      CHECK((dot(plus, upstream, 8) - dot(minus, upstream, 8)) / 0.0625 == gradient[i]);
    }
  }
  const uint32_t finite[] = {0, 0x80000000u, 1, 0x80000001u, 0x00800000u,
                              0x80800000u, 0x7f7fffffu, 0xff7fffffu};
  memcpy(x, finite, sizeof(x));
  layout_call(rt, operations[0], 1, x, y);
  layout_call(rt, operations[1], 0, y, original);
  exact_words(original, finite, 8);
}

static void sum_call(et_kernel_runtime *rt, const char *cap, size_t rank,
                     const uint64_t *shape, size_t count, size_t edges,
                     int backward, float *a, float *b, float *c,
                     float *y, float *dy2, float *dy3) {
  const char *operation = edges == 2
      ? (backward ? "n3k.sum2.backward" : "n3k.sum2.forward")
      : (backward ? "n3k.sum3.backward" : "n3k.sum3.forward");
  et_kernel_request_v1 request = rq(operation, rank, shape);
  et_kernel_tensor_view_v1 input[] = {
      tv(a, count * 4, "f32", rank, shape), tv(b, count * 4, "f32", rank, shape),
      tv(c, count * 4, "f32", rank, shape)};
  et_kernel_tensor_view_v1 output[] = {
      tv(y, count * 4, "f32", rank, shape), tv(dy2, count * 4, "f32", rank, shape),
      tv(dy3, count * 4, "f32", rank, shape)};
  et_kernel_call_v1 call = kc(cap, &request, input, backward ? 1 : edges,
                             output, backward ? edges : 1);
  run(rt, &call);
}

static void test_sums(et_kernel_runtime *rt) {
  float a[1024], b[1024], c[1024], y[1024], gradients[3][1024];
  float upstream[1024], plus[1024], minus[1024];
  for (size_t which = 0; which < 3; ++which) {
    const size_t count = which == 1 ? 1024 : 8;
    const size_t edges = which == 2 ? 3 : 2;
    const size_t rank = which == 1 ? 2 : 3;
    const uint64_t *shape = which == 1 ? tied_shape : token_shape;
    const char *cap = which == 1 ? "n3k.tied-sum" : "n3k.activation-sum";
    for (size_t i = 0; i < count; ++i) {
      a[i] = (float)((int)(i % 23) - 11) * 0.25f;
      b[i] = (float)((int)(i % 17) - 7) * 0.125f;
      c[i] = (float)((int)(i % 13) - 3) * 0.0625f;
      upstream[i] = (float)((int)(i % 7) - 3) * 0.125f;
    }
    sum_call(rt, cap, rank, shape, count, edges, 0, a, b, c, y, NULL, NULL);
    for (size_t i = 0; i < count; ++i) {
      const float first = a[i] + b[i];
      CHECK(y[i] == (edges == 3 ? first + c[i] : first));
    }
    CHECK(memcmp(y, a, count * 4) != 0 && memcmp(y, b, count * 4) != 0);
    sum_call(rt, cap, rank, shape, count, edges, 1, upstream, NULL, NULL,
             gradients[0], gradients[1], gradients[2]);
    for (size_t edge = 0; edge < edges; ++edge)
      CHECK(memcmp(gradients[edge], upstream, count * 4) == 0);
    /* Every coordinate of every analytic edge, including the full tied matrix. */
    float *primals[] = {a, b, c};
    for (size_t edge = 0; edge < edges; ++edge) {
      for (size_t i = 0; i < count; ++i) {
        const float original = primals[edge][i];
        primals[edge][i] = original + 0.03125f;
        sum_call(rt, cap, rank, shape, count, edges, 0, a, b, c, plus, NULL, NULL);
        primals[edge][i] = original - 0.03125f;
        sum_call(rt, cap, rank, shape, count, edges, 0, a, b, c, minus, NULL, NULL);
        primals[edge][i] = original;
        CHECK((dot(plus, upstream, count) - dot(minus, upstream, count)) / 0.0625 ==
              gradients[edge][i]);
      }
    }
    /* Exact repeated semantic views are accepted and remain separate VJP edges. */
    sum_call(rt, cap, rank, shape, count, edges, 0, a, a, a, y, NULL, NULL);
    for (size_t i = 0; i < count; ++i) CHECK(y[i] == (float)edges * a[i]);
    const uint32_t signed_zero = 0x80000000u;
    memcpy(upstream, &signed_zero, 4);
    sum_call(rt, cap, rank, shape, count, edges, 1, upstream, NULL, NULL,
             gradients[0], gradients[1], gradients[2]);
    for (size_t edge = 0; edge < edges; ++edge) CHECK(fbits(gradients[edge][0]) == signed_zero);
  }
  a[0] = 0x1p24f; b[0] = -0x1p24f; c[0] = 1;
  a[1] = 1; b[1] = 0x1p24f; c[1] = -0x1p24f;
  sum_call(rt, "n3k.activation-sum", 3, token_shape, 8, 3, 0, a, b, c, y, NULL, NULL);
  CHECK(y[0] == 1 && y[1] == 0); /* Right association gives 1 at index 1. */
  sum_call(rt, "n3k.activation-sum", 3, token_shape, 8, 3, 0, a, c, b, y, NULL, NULL);
  CHECK(y[0] == 0); /* Detect a reassociated/order-changed implementation. */
}

static et_kernel_call_v1 initializer_call(et_kernel_request_v1 *request,
                                         et_kernel_tensor_view_v1 *input,
                                         et_kernel_tensor_view_v1 output[2],
                                         const uint64_t shape[2], int64_t state[4],
                                         float *matrix, int64_t successor[4]) {
  *request = rq("n3k.matrix-init.uniform", 2, shape);
  *input = tv(state, 32, "i64", 1, state_shape);
  output[0] = tv(matrix, (size_t)(shape[0] * shape[1]) * 4, "f32", 2, shape);
  output[1] = tv(successor, 32, "i64", 1, state_shape);
  return kc("n3k.matrix-init", request, input, 1, output, 2);
}

static void test_initializer(et_kernel_runtime *rt) {
  float matrix[1024], repeated[1024];
  int64_t state[4], successor[4], repeated_state[4];
  for (size_t k = 0; k < COUNT(n3k_init_references); ++k) {
    const n3k_init_reference_case *ref = &n3k_init_references[k];
    et_kernel_request_v1 request;
    et_kernel_tensor_view_v1 input, output[2];
    memcpy(state, ref->state, sizeof(state));
    memset(matrix, 0xa5, sizeof(matrix));
    memset(successor, 0xa5, sizeof(successor));
    et_kernel_call_v1 call = initializer_call(&request, &input, output, ref->shape,
                                              state, matrix, successor);
    run(rt, &call);
    exact_words(matrix, ref->words, ref->count);
    CHECK(memcmp(successor, ref->successor, sizeof(successor)) == 0);
    CHECK(memcmp(state, ref->state, sizeof(state)) == 0);
    output[0].data = repeated;
    output[1].data = repeated_state;
    memset(repeated, 0x5a, sizeof(repeated));
    run(rt, &call);
    CHECK(memcmp(repeated, matrix, ref->count * 4) == 0);
    CHECK(memcmp(repeated_state, successor, sizeof(successor)) == 0);
  }
  /* Native continuation across an independently specified eight-unique schedule.
   * Paths/identities are test-only; the provider owns no registry or tie policy. */
  static const size_t starts[] = {0, 256, 258, 262, 266, 270, 274, 282};
  int64_t threaded[] = {1, 1729, 0, 0};
  size_t draws = 0;
  for (size_t k = 0; k < COUNT(n3k_init_references); ++k) {
    const n3k_init_reference_case *ref = &n3k_init_references[k];
    if (strncmp(ref->name, "sequence-", 9) != 0) continue;
    CHECK(draws < COUNT(starts) && threaded[2] == (int64_t)starts[draws]);
    et_kernel_request_v1 request;
    et_kernel_tensor_view_v1 input, output[2];
    et_kernel_call_v1 call = initializer_call(&request, &input, output, ref->shape,
                                              threaded, matrix, successor);
    run(rt, &call);
    exact_words(matrix, ref->words, ref->count);
    memcpy(threaded, successor, sizeof(threaded));
    ++draws;
  }
  CHECK(draws == 8 && threaded[2] == 290 && threaded[3] == 0);
  /* Repeat using reversed registration, minimum-path tied representative, and
   * byte-sorted representatives. Only this test scheduler knows these identities. */
  const struct { const char *path; size_t identity; } registration[] = {
    {"z.token", 0}, {"h.contract", 7}, {"g.expand", 6}, {"f.attention", 5},
    {"e.value", 4}, {"d.key", 3}, {"c.query", 2}, {"b.position", 1}, {"a.head", 0}
  };
  const char *representative[8] = {0};
  size_t order[8];
  for (size_t k = 0; k < COUNT(registration); ++k) {
    const size_t identity = registration[k].identity;
    if (!representative[identity] || strcmp(registration[k].path, representative[identity]) < 0)
      representative[identity] = registration[k].path;
  }
  for (size_t k = 0; k < 8; ++k) {
    CHECK(representative[k] != NULL);
    order[k] = k;
    for (size_t j = k; j > 0 && strcmp(representative[order[j]], representative[order[j-1]]) < 0; --j) {
      const size_t tmp = order[j]; order[j] = order[j-1]; order[j-1] = tmp;
    }
  }
  memset(threaded, 0, sizeof(threaded)); threaded[0] = 1; threaded[1] = 1729;
  for (size_t k = 0; k < 8; ++k) {
    CHECK(order[k] == k);
    char name[32]; (void)snprintf(name, sizeof(name), "sequence-%zu", order[k]);
    const n3k_init_reference_case *ref = NULL;
    for (size_t j = 0; j < COUNT(n3k_init_references); ++j)
      if (strcmp(n3k_init_references[j].name, name) == 0) ref = &n3k_init_references[j];
    CHECK(ref != NULL && threaded[2] == (int64_t)starts[k]);
    et_kernel_request_v1 request;
    et_kernel_tensor_view_v1 input, output[2];
    et_kernel_call_v1 call = initializer_call(&request, &input, output, ref->shape,
                                              threaded, matrix, successor);
    run(rt, &call);
    exact_words(matrix, ref->words, ref->count);
    memcpy(threaded, successor, sizeof(threaded));
  }
  CHECK(threaded[2] == 290 && threaded[3] == 0);
  for (size_t row = 0; row < 5; ++row) {
    const n3k_init_reference_case *ref = &n3k_init_references[row];
    const int64_t blocks = (int64_t)(ref->count / 4);
    et_kernel_request_v1 request;
    et_kernel_tensor_view_v1 input, output[2];
    int64_t exhausted[] = {1, -1, -blocks, -1};
    et_kernel_call_v1 call = initializer_call(&request, &input, output, ref->shape,
                                              exhausted, matrix, successor);
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    exhausted[2] = -1;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
}

static void test_initializer_errors(et_kernel_runtime *rt) {
  const uint64_t shape[] = {2, 4};
  int64_t state[] = {1, 1729, 0, 0}, successor[4];
  float matrix[9];
  memset(matrix, 0xa5, sizeof(matrix));
  memset(successor, 0x5a, sizeof(successor));
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 input, output[2];
  et_kernel_call_v1 call = initializer_call(&request, &input, output, shape,
                                            state, matrix, successor);
  for (size_t i = 0; i < 3; ++i) {
    state[0] = (int64_t[]){0, 2, INT64_MIN}[i];
    preserved_reject(rt, &call, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  state[0] = 1;
  for (size_t i = 0; i < 2; ++i) {
    state[2] = -(int64_t)(i + 1); state[3] = -1;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  state[2] = state[3] = 0;
  const et_kernel_tensor_view_v1 valid_input = input;
  input.dtype = "f32";
  input.byte_length = 16;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT);
  input = valid_input;
  const uint64_t wrong_rank[] = {1, 4};
  input.rank = 2; input.shape = wrong_rank;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
  input = valid_input;
  const uint64_t short_state[] = {3};
  input.shape = short_state; input.byte_length = 24;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
  input = valid_input;
  input.device = "cuda";
  preserved_reject(rt, &call, ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
  input = valid_input;
  int64_t unaligned_backing[5] = {1, 1729, 0, 0, 0};
  input.data = (unsigned char *)unaligned_backing + 1;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  input = valid_input;
  output[0].data = (unsigned char *)matrix + 1;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  output[0].data = matrix;
  output[1].dtype = "f32"; output[1].byte_length = 16;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT);
  output[1].dtype = "i64"; output[1].byte_length = 32;
  output[1].shape = short_state; output[1].byte_length = 24;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
  output[1].shape = state_shape; output[1].byte_length = 32;
  output[1].data = state;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
  output[1].data = successor;
  output[0].data = successor;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
  output[0].data = matrix;
  /* Corrected accepted K1 precedence: shape multiplication vs address overflow. */
  const uint64_t product_overflow[] = {UINT64_MAX, 2};
  output[0].shape = product_overflow;
  preserved_reject(rt, &call, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INTEGER_OVERFLOW);
  output[0].shape = shape;
  output[0].data = (void *)(uintptr_t)(UINTPTR_MAX - 15);
  reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
  et_kernel_error error;
  CHECK(et_n3k_kernel_provider_v1()->validate_call(&call, &error) != 0);
  CHECK(error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
        error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  output[0].data = matrix;
  for (size_t i = 0; i < sizeof(matrix); ++i) CHECK(((unsigned char *)matrix)[i] == 0xa5);
  for (size_t i = 0; i < sizeof(successor); ++i) CHECK(((unsigned char *)successor)[i] == 0x5a);
  /* Every neighboring extent of every exact initializer row must reject. */
  for (size_t row = 0; row < 5; ++row) {
    for (size_t d = 0; d < 2; ++d) {
      for (int delta = -1; delta <= 1; delta += 2) {
        uint64_t neighbor[] = {n3k_init_references[row].shape[0],
                               n3k_init_references[row].shape[1]};
        neighbor[d] = (uint64_t)((int64_t)neighbor[d] + delta);
        request.shape = neighbor;
        preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED,
                         ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
      }
    }
  }
  request.shape = shape;
  request.operation = "n3k.matrix-init.backward";
  preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
}

static void test_sum_errors(et_kernel_runtime *rt) {
  float a[1025], b[1024], c[1024], y[1024], z[1024], w[1024];
  for (size_t i = 0; i < 1025; ++i) a[i] = 1;
  for (size_t i = 0; i < 1024; ++i) { b[i] = 2; c[i] = 3; }
  memset(y, 0xa5, sizeof(y)); memset(z, 0x5a, sizeof(z)); memset(w, 0x33, sizeof(w));
  for (size_t which = 0; which < 3; ++which) {
    const size_t n = which == 1 ? 1024 : 8, rank = which == 1 ? 2 : 3;
    const size_t edges = which == 2 ? 3 : 2;
    const uint64_t *shape = which == 1 ? tied_shape : token_shape;
    et_kernel_request_v1 request = rq(edges == 3 ? "n3k.sum3.forward" : "n3k.sum2.forward", rank, shape);
    et_kernel_tensor_view_v1 input[] = {tv(a, n*4, "f32", rank, shape),
        tv(b, n*4, "f32", rank, shape), tv(c, n*4, "f32", rank, shape)};
    et_kernel_tensor_view_v1 output[] = {tv(y, n*4, "f32", rank, shape),
        tv(z, n*4, "f32", rank, shape), tv(w, n*4, "f32", rank, shape)};
    et_kernel_call_v1 call = kc(which == 1 ? "n3k.tied-sum" : "n3k.activation-sum",
                               &request, input, edges, output, 1);
    input[1].data = a + 1;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    input[1].data = b;
    output[0].data = a;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    output[0].data = y;
    for (size_t edge = 0; edge < edges; ++edge) {
      float *values[] = {a, b, c};
      const float saved = values[edge][n-1];
      for (size_t bad = 0; bad < 3; ++bad) {
        const uint32_t word = (uint32_t[]){0x7f800000u, 0xff800000u, 0x7f800001u}[bad];
        memcpy(&values[edge][n-1], &word, 4);
        preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
      }
      values[edge][n-1] = saved;
    }
    a[n-1] = FLT_MAX; b[n-1] = FLT_MAX; c[n-1] = -FLT_MAX;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    a[n-1] = 1; b[n-1] = 2; c[n-1] = 3;
    request.operation = edges == 3 ? "n3k.sum3.backward" : "n3k.sum2.backward";
    call.input_count = 1; call.input_bytes = sizeof(input[0]);
    call.output_count = edges; call.output_bytes = edges * sizeof(output[0]);
    output[1].data = y;
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    output[1].data = z;
    const uint32_t nan = 0x7fc00000;
    memcpy(a + n-1, &nan, 4);
    preserved_reject(rt, &call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    a[n-1] = 1;
  }
}

static unsigned short x87_control(void) {
  unsigned short value; __asm__ volatile("fnstcw %0" : "=m"(value)); return value;
}
static unsigned short x87_status(void) {
  unsigned short value; __asm__ volatile("fnstsw %0" : "=m"(value)); return value;
}
static void set_x87_control(unsigned short value) {
  __asm__ volatile("fldcw %0" : : "m"(value));
}

static void test_fenv(et_kernel_runtime *rt) {
  fenv_t saved;
  CHECK(fegetenv(&saved) == 0);
  const unsigned original_mxcsr = _mm_getcsr();
  float a[8] = {1}, b[8] = {2}, y[8];
  memset(y, 0xa5, sizeof(y));
  et_kernel_request_v1 request = rq("n3k.sum2.forward", 3, token_shape);
  et_kernel_tensor_view_v1 input[] = {tv(a, 32, "f32", 3, token_shape), tv(b, 32, "f32", 3, token_shape)};
  et_kernel_tensor_view_v1 output = tv(y, 32, "f32", 3, token_shape);
  et_kernel_call_v1 call = kc("n3k.activation-sum", &request, input, 2, &output, 1);
  for (size_t failed = 0; failed < 2; ++failed) {
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    CHECK(feraiseexcept(FE_INEXACT | FE_DIVBYZERO) == 0);
    _mm_setcsr((_mm_getcsr() & ~0x3fu) | 0x2au);
    if (failed) { const uint32_t snan = 0x7f800001; memcpy(a + 7, &snan, 4); }
    const unsigned short before_x87 = x87_status();
    const unsigned before_mxcsr = _mm_getcsr();
    et_kernel_error error;
    const int32_t result = et_n3k_kernel_provider_v1()->validate_call(&call, &error);
    const unsigned short after_x87 = x87_status();
    const unsigned after_mxcsr = _mm_getcsr();
    CHECK((result != 0) == (failed != 0));
    CHECK(before_x87 == after_x87 && before_mxcsr == after_mxcsr);
    for (size_t i = 0; i < sizeof(y); ++i) CHECK(((unsigned char *)y)[i] == 0xa5);
  }
  a[7] = 0;
  CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
  for (size_t mode = 0; mode < 3; ++mode) {
    CHECK(fesetround((int[]){FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}[mode]) == 0);
    preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK((fegetround() == (int[]){FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}[mode]));
    CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
  }
  for (size_t bit = 0; bit < 8; ++bit) {
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    const unsigned base = _mm_getcsr();
    const unsigned altered = bit < 2 ? base | (bit == 0 ? 0x40u : 0x8000u)
                                    : base & ~(1u << (bit + 5));
    _mm_setcsr(altered);
    et_kernel_error error;
    const int32_t result = et_kernel_runtime_dispatch(rt, &call, &error);
    const unsigned after = _mm_getcsr();
    _mm_setcsr(base);
    CHECK(result != 0 && error.category == ET_KERNEL_ERROR_UNSUPPORTED &&
          error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(after == altered);
  }
  for (size_t bit = 0; bit < 6; ++bit) {
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    const unsigned short base = x87_control();
    const unsigned short altered = (unsigned short)(base & ~(1u << bit));
    set_x87_control(altered);
    et_kernel_error error;
    const int32_t result = et_kernel_runtime_dispatch(rt, &call, &error);
    const unsigned short after = x87_control();
    set_x87_control(base);
    CHECK(result != 0 && error.category == ET_KERNEL_ERROR_UNSUPPORTED &&
          error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(after == altered);
  }
  for (size_t i = 0; i < sizeof(y); ++i) CHECK(((unsigned char *)y)[i] == 0xa5);
  CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
}

static void test_initializer_fenv(et_kernel_runtime *rt) {
  fenv_t saved;
  CHECK(fegetenv(&saved) == 0);
  const unsigned original_mxcsr = _mm_getcsr();
  const uint64_t shape[] = {2, 4};
  int64_t state[] = {1, -1, 0, 0}, successor[4];
  float matrix[8];
  memset(matrix, 0xa5, sizeof(matrix)); memset(successor, 0x5a, sizeof(successor));
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 input, output[2];
  et_kernel_call_v1 call = initializer_call(&request, &input, output, shape,
                                            state, matrix, successor);
  for (size_t invalid = 0; invalid < 3; ++invalid) {
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    CHECK(feraiseexcept(FE_INEXACT | FE_DIVBYZERO) == 0);
    _mm_setcsr((_mm_getcsr() & ~0x3fu) | 0x2au);
    state[0] = invalid == 1 ? 2 : 1;
    state[2] = state[3] = invalid == 2 ? -1 : 0;
    const unsigned short before_x87 = x87_status();
    const unsigned before_mxcsr = _mm_getcsr();
    et_kernel_error error;
    const int32_t result = et_n3k_kernel_provider_v1()->validate_call(&call, &error);
    const unsigned short after_x87 = x87_status();
    const unsigned after_mxcsr = _mm_getcsr();
    CHECK((result != 0) == (invalid != 0));
    CHECK(before_x87 == after_x87 && before_mxcsr == after_mxcsr);
    if (invalid) {
      CHECK(error.category == (invalid == 1 ? ET_KERNEL_ERROR_VERSION_MISMATCH
                                            : ET_KERNEL_ERROR_INVALID_ARGUMENT));
      CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    }
  }
  CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
  state[0] = 1; state[2] = state[3] = 0;
  CHECK(fesetround(FE_UPWARD) == 0);
  preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
  for (size_t bit = 0; bit < 2; ++bit) {
    const unsigned changed = original_mxcsr | (bit == 0 ? 0x40u : 0x8000u);
    _mm_setcsr(changed);
    preserved_reject(rt, &call, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(_mm_getcsr() == changed);
    _mm_setcsr(original_mxcsr);
  }
  for (size_t i = 0; i < sizeof(matrix); ++i) CHECK(((unsigned char *)matrix)[i] == 0xa5);
  for (size_t i = 0; i < sizeof(successor); ++i) CHECK(((unsigned char *)successor)[i] == 0x5a);
  CHECK(fesetenv(&saved) == 0); _mm_setcsr(original_mxcsr);
}

int main(void) {
  et_kernel_runtime *rt = runtime_new();
  test_layout(rt);
  test_sums(rt);
  test_initializer(rt);
  test_initializer_errors(rt);
  test_sum_errors(rt);
  test_fenv(rt);
  test_initializer_fenv(rt);
  et_kernel_runtime_destroy(rt);
  printf("N3K composition/initializer PASS (%zu checks)\n", checks);
  return 0;
}
