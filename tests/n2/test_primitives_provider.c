#include "eshkol_transformer/n2_primitives_abi.h"
#include "reference_vectors.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

static size_t checks;

#define CHECK(x)                                                               \
  do {                                                                         \
    checks++;                                                                  \
    if (!(x)) {                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int nearf(float a, float b, float atol) {
  return isfinite(a) && isfinite(b) && fabsf(a - b) <= atol;
}

static uint32_t fbits(float x) {
  uint32_t u;
  memcpy(&u, &x, 4u);
  return u;
}
static int64_t signed_bits(uint64_t u) {
  int64_t s;
  memcpy(&s, &u, 8u);
  return s;
}

static et_kernel_tensor_view_v1 tv(void *data, size_t bytes, const char *dtype,
                                   size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 v = {
      sizeof(v), data, bytes, dtype, "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      0u,        rank, shape};
  return v;
}

static et_kernel_request_v1 rq(const char *op, size_t rank,
                               const uint64_t *shape) {
  et_kernel_request_v1 r = {sizeof(r), op, "f32", "cpu", rank, shape, 1u, {0}};
  return r;
}

static et_kernel_call_v1 kc(const char *cap, const et_kernel_request_v1 *r,
                            et_kernel_tensor_view_v1 *in, size_t ni,
                            et_kernel_tensor_view_v1 *out, size_t no) {
  et_kernel_call_v1 c = {sizeof(c),
                         cap,
                         r,
                         ni,
                         sizeof(*in),
                         ni * sizeof(*in),
                         in,
                         no,
                         sizeof(*out),
                         no * sizeof(*out),
                         out};
  return c;
}

static const et_kernel_provider_v1 *resolve(void *ctx, const char *symbol) {
  (void)ctx;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_n2_kernel_provider_v1();
}

static et_kernel_runtime *runtime_new(void) {
  et_kernel_runtime *rt = NULL;
  et_kernel_error e;
  CHECK(et_kernel_runtime_discover(resolve, NULL, &rt, &e) == 0);
  CHECK(rt != NULL);
  return rt;
}

static void run(et_kernel_runtime *rt, et_kernel_call_v1 *c) {
  et_kernel_error e;
  if (et_kernel_runtime_dispatch(rt, c, &e) != 0) {
    fprintf(stderr,
            "unexpected dispatch error: category=%u code=%u op=%s msg=%s\n",
            e.category, e.code, e.operation, e.message);
    CHECK(0);
  }
}

static void reject(et_kernel_runtime *rt, et_kernel_call_v1 *c,
                   uint32_t category, uint32_t code) {
  et_kernel_error e;
  CHECK(et_kernel_runtime_dispatch(rt, c, &e) != 0);
  CHECK(e.category == category);
  if (code != UINT32_MAX)
    CHECK(e.code == code);
}

static void expect_floats(const float *a, const float *b, size_t n, float tol) {
  for (size_t i = 0; i < n; i++) {
    if (!nearf(a[i], b[i], tol))
      fprintf(stderr,
              "float mismatch[%zu]: actual %.9g expected %.9g tol %.9g\n", i,
              a[i], b[i], tol);
    CHECK(nearf(a[i], b[i], tol));
  }
}

static double dot(const float *a, const float *b, size_t n) {
  double result = 0.0;
  for (size_t i = 0; i < n; i++)
    result += (double)a[i] * (double)b[i];
  return result;
}

static float finite_difference_step(float value, float epsilon) {
  return epsilon * fmaxf(1.0f, fabsf(value));
}

static void test_metadata(et_kernel_runtime *rt) {
  static const char *const embedding_forward_ops[] = {"embedding.forward"};
  static const char *const embedding_backward_ops[] = {"embedding.backward"};
  static const char *const matmul_ops[] = {
      "linear.backward", "linear.backward-no-bias", "linear.forward",
      "linear.forward-no-bias"};
  static const char *const norm_ops[] = {"layer-norm.backward",
                                         "layer-norm.forward"};
  static const char *const activation_ops[] = {"gelu.backward", "gelu.forward",
                                               "relu.backward", "relu.forward"};
  static const char *const dropout_ops[] = {
      "dropout.eval.backward", "dropout.eval.forward", "dropout.train.backward",
      "dropout.train.forward"};
  static const char *const residual_ops[] = {"residual.backward",
                                             "residual.forward"};
  static const uint64_t embedding_rows[] = {1, 1, 1, 1, 1, 4, 3, 2, 2, 3, 5, 6};
  static const uint64_t linear_rows[] = {1, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4, 5};
  static const uint64_t norm_rows[] = {1, 1, 1, 1, 1, 3, 1, 2, 4, 2, 2, 4};
  static const uint64_t activation_rows[] = {1, 1, 1, 1, 1, 8,
                                             1, 2, 5, 2, 2, 4};
  static const uint64_t dropout_rows[] = {1, 1, 1, 1, 1, 2, 1, 1,
                                          3, 1, 1, 4, 1, 2, 5};
  static const uint64_t residual_rows[] = {1, 1, 1, 1, 3, 4, 2, 2, 4};
  static const struct {
    const char *name;
    const char *const *ops;
    size_t nops;
    const uint64_t *rows;
    size_t nrows;
    size_t rank;
  } expected[] = {
      {"kernel.embedding-forward", embedding_forward_ops, 1, embedding_rows, 3,
       4},
      {"kernel.embedding-backward", embedding_backward_ops, 1, embedding_rows,
       3, 4},
      {"kernel.matmul", matmul_ops, 4, linear_rows, 3, 4},
      {"kernel.norm", norm_ops, 2, norm_rows, 4, 3},
      {"kernel.activation", activation_ops, 4, activation_rows, 4, 3},
      {"kernel.dropout", dropout_ops, 4, dropout_rows, 5, 3},
      {"kernel.residual", residual_ops, 2, residual_rows, 3, 3},
  };
  const et_kernel_provider_v1 *p = et_n2_kernel_provider_v1();
  CHECK(ET_N2_PRIMITIVES_ABI_MAJOR == 1u && ET_N2_PRIMITIVES_ABI_MINOR == 0u);
  CHECK(p && p == et_n2_kernel_provider_v1());
  CHECK(p->abi_major == 1u && p->abi_minor == 0u && p->required_features == 0u);
  CHECK(strcmp(p->name, "n2.cpu-f32.serial") == 0);
  CHECK(strcmp(p->version, "1.0") == 0);
  CHECK(strcmp(p->evidence, "N2:cpu-f32-primitives-v1") == 0);
  CHECK(p->capability_count == COUNT(expected));
  for (size_t k = 0; k < COUNT(expected); k++) {
    const et_kernel_capability_v1 *cap =
        et_kernel_runtime_capability_find(rt, expected[k].name);
    CHECK(cap && cap->status == ET_KERNEL_CAPABILITY_VERIFIED &&
          cap->deterministic == 1u);
    CHECK(strcmp(cap->implementation, "n2.cpu-f32.serial") == 0);
    CHECK(strcmp(cap->version, "1.0") == 0 &&
          strcmp(cap->evidence, "N2:cpu-f32-primitives-v1") == 0);
    CHECK(cap->operation_count == expected[k].nops && cap->dtype_count == 1u &&
          cap->device_count == 1u);
    CHECK(strcmp(cap->dtypes[0], "f32") == 0 &&
          strcmp(cap->devices[0], "cpu") == 0);
    CHECK(cap->shape_range_count == expected[k].nrows);
    for (size_t o = 0; o < cap->operation_count; o++)
      CHECK(strcmp(cap->operations[o], expected[k].ops[o]) == 0);
    for (size_t r = 0; r < expected[k].nrows; r++) {
      CHECK(cap->shape_ranges[r].rank == expected[k].rank);
      for (size_t d = 0; d < expected[k].rank; d++) {
        const et_kernel_dimension_range_v1 *dr =
            &cap->shape_ranges[r].dimensions[d];
        uint64_t want = expected[k].rows[r * expected[k].rank + d];
        CHECK(dr->minimum == want && dr->maximum == want &&
              dr->maximum_unbounded == 0u);
      }
      for (size_t o = 0; o < expected[k].nops; o++) {
        et_kernel_request_v1 req = rq(expected[k].ops[o], expected[k].rank,
                                      expected[k].rows + r * expected[k].rank);
        const et_kernel_capability_v1 *found = NULL;
        et_kernel_error e;
        CHECK(et_kernel_runtime_capability_require(rt, expected[k].name, &req,
                                                   &found, &e) == 0);
        CHECK(found == cap);
      }
    }
  }
  {
    uint64_t zero[] = {1, 1, 0};
    et_kernel_request_v1 r = rq("relu.forward", 3, zero);
    const et_kernel_capability_v1 *cap = NULL;
    et_kernel_error e;
    CHECK(et_kernel_runtime_capability_require(rt, "kernel.activation", &r,
                                               &cap, &e) != 0);
    CHECK(e.category == ET_KERNEL_ERROR_UNSUPPORTED);
  }
}

static void test_embedding(et_kernel_runtime *rt) {
  uint64_t sem[] = {1, 4, 3, 2}, ids_shape[] = {1, 4}, w_shape[] = {3, 2},
           y_shape[] = {1, 4, 2};
  int64_t ids[] = {2, 0, 2, 1};
  float w[] = {1, 2, 3, 4, 5, 6};
  float y[8];
  float up[] = {1, 2, 3, 4, 5, 6, 7, 8}, dw[6];
  et_kernel_tensor_view_v1 in[] = {tv(ids, sizeof(ids), "i64", 2, ids_shape),
                                   tv(w, sizeof(w), "f32", 2, w_shape)};
  et_kernel_tensor_view_v1 out[] = {tv(y, sizeof(y), "f32", 3, y_shape)};
  et_kernel_request_v1 req = rq("embedding.forward", 4, sem);
  et_kernel_call_v1 c = kc("kernel.embedding-forward", &req, in, 2, out, 1);
  const float ey[] = {5, 6, 1, 2, 5, 6, 3, 4};
  run(rt, &c);
  expect_floats(y, ey, 8, 0);
  in[1] = tv(up, sizeof(up), "f32", 3, y_shape);
  out[0] = tv(dw, sizeof(dw), "f32", 2, w_shape);
  req.operation = "embedding.backward";
  c.capability = "kernel.embedding-backward";
  run(rt, &c);
  {
    const float edw[] = {3, 4, 7, 8, 6, 8};
    expect_floats(dw, edw, 6, 0);
  }
  /* Direct numerical VJP for the differentiable embedding weight on an
   * admitted row. */
  {
    const float epsilon = 1.0e-3f;
    const float tolerance = 2.0e-3f;
    req.operation = "embedding.forward";
    c.capability = "kernel.embedding-forward";
    in[1] = tv(w, sizeof(w), "f32", 2, w_shape);
    out[0] = tv(y, sizeof(y), "f32", 3, y_shape);
    for (size_t i = 0; i < COUNT(w); i++) {
      const float original = w[i];
      const float step = finite_difference_step(original, epsilon);
      double positive;
      double negative;
      w[i] = original + step;
      run(rt, &c);
      positive = dot(y, up, COUNT(y));
      w[i] = original - step;
      run(rt, &c);
      negative = dot(y, up, COUNT(y));
      w[i] = original;
      CHECK(nearf(dw[i], (positive - negative) / (2.0f * step), tolerance));
    }
    req.operation = "embedding.backward";
    c.capability = "kernel.embedding-backward";
    in[1] = tv(up, sizeof(up), "f32", 3, y_shape);
    out[0] = tv(dw, sizeof(dw), "f32", 2, w_shape);
  }
  /* Repeated IDs preserve fixed reduction order; unused rows are +0. */
  ids[0] = 0;
  ids[1] = 0;
  ids[2] = 0;
  ids[3] = 0;
  run(rt, &c);
  CHECK(fbits(dw[2]) == 0u && fbits(dw[3]) == 0u && fbits(dw[4]) == 0u &&
        fbits(dw[5]) == 0u);
  for (size_t q = 0; q < 2; q++) {
    ids[0] = q ? 3 : -1;
    float snap[6];
    memcpy(snap, dw, sizeof snap);
    reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(memcmp(snap, dw, sizeof snap) == 0);
  }
}

static void test_linear(et_kernel_runtime *rt) {
  uint64_t sem[] = {1, 2, 3, 4}, xs[] = {1, 2, 3}, ws[] = {4, 3}, bs[] = {4},
           ys[] = {1, 2, 4};
  float x[] = {1, 2, 3, -1, 0.5f, 2},
        w[] = {1, 0, -1, 2, 1, 0, 0, -2, 1, .5f, .25f, -.5f};
  float b[] = {.5f, -1, 2, 0}, up[] = {1, 2, 3, 4, -1, .5f, 2, -2}, y[8], dx[6],
        dw[12], db[4];
  et_kernel_tensor_view_v1 in[] = {tv(x, sizeof x, "f32", 3, xs),
                                   tv(w, sizeof w, "f32", 2, ws),
                                   tv(b, sizeof b, "f32", 1, bs)};
  et_kernel_tensor_view_v1 out[3] = {tv(y, sizeof y, "f32", 3, ys)};
  et_kernel_request_v1 req = rq("linear.forward", 4, sem);
  et_kernel_call_v1 c = kc("kernel.matmul", &req, in, 3, out, 1);
  run(rt, &c);
  {
    const float e[] = {-1.5f, 3, 1, -.5f, -2.5f, -2.5f, 3, -1.375f};
    expect_floats(y, e, 8, 1e-6f);
  }
  req.operation = "linear.forward-no-bias";
  c.input_count = 2;
  c.input_bytes = 2 * sizeof(*in);
  run(rt, &c);
  {
    const float e[] = {-2, 4, -1, -.5f, -3, -1.5f, 1, -1.375f};
    expect_floats(y, e, 8, 1e-6f);
  }
  in[2] = tv(up, sizeof up, "f32", 3, ys);
  out[0] = tv(dx, sizeof dx, "f32", 3, xs);
  out[1] = tv(dw, sizeof dw, "f32", 2, ws);
  out[2] = tv(db, sizeof db, "f32", 1, bs);
  req.operation = "linear.backward";
  c.input_count = 3;
  c.input_bytes = sizeof(in);
  c.output_count = 3;
  c.output_bytes = sizeof(out);
  run(rt, &c);
  {
    const float edx[] = {7, -3, 0, -1, -4, 4}, edb[] = {0, 2.5f, 5, 2};
    expect_floats(dx, edx, 6, 1e-6f);
    expect_floats(db, edb, 4, 1e-6f);
  }
  req.operation = "linear.backward-no-bias";
  c.output_count = 2;
  c.output_bytes = 2 * sizeof(*out);
  run(rt, &c);
  /* Direct scaled numerical VJPs on the admitted no-bias row. The bias form
   * has the same x/weight VJPs and is checked separately for dBias below. */
  req.operation = "linear.forward-no-bias";
  c.input_count = 2;
  c.input_bytes = 2 * sizeof(*in);
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  out[0] = tv(y, sizeof y, "f32", 3, ys);
  const float epsilon = 1.0e-3f;
  const float tolerance = 2.0e-3f;
  for (size_t i = 0; i < COUNT(x); i++) {
    const float original = x[i];
    const float step = finite_difference_step(original, epsilon);
    double positive;
    double negative;
    x[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    x[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    x[i] = original;
    CHECK(nearf(dx[i], (positive - negative) / (2.0f * step), tolerance));
  }
  for (size_t i = 0; i < COUNT(w); i++) {
    const float original = w[i];
    const float step = finite_difference_step(original, epsilon);
    double positive;
    double negative;
    w[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    w[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    w[i] = original;
    CHECK(nearf(dw[i], (positive - negative) / (2.0f * step), tolerance));
  }
  req.operation = "linear.forward";
  in[2] = tv(b, sizeof(b), "f32", 1, bs);
  c.input_count = 3u;
  c.input_bytes = sizeof(in);
  for (size_t i = 0; i < COUNT(b); i++) {
    const float original = b[i];
    const float step = finite_difference_step(original, epsilon);
    double positive;
    double negative;
    b[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    b[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    b[i] = original;
    CHECK(nearf(db[i], (positive - negative) / (2.0f * step), tolerance));
  }
}

static void test_norm(et_kernel_runtime *rt) {
  uint64_t sem[] = {1, 1, 3}, xs[] = {1, 1, 3}, ds[] = {3};
  float x[] = {1, 2, 4}, g[] = {1, .5f, -2}, b[] = {.25f, -1, 2},
        epsilon = 1e-5f, up[] = {.5f, -1, 2}, y[3], dx[3], dg[3], db[3];
  et_kernel_tensor_view_v1 in[] = {
      tv(x, sizeof x, "f32", 3, xs), tv(g, sizeof g, "f32", 1, ds),
      tv(b, sizeof b, "f32", 1, ds),
      tv(&epsilon, sizeof epsilon, "f32", 0, NULL)};
  et_kernel_tensor_view_v1 out[3] = {tv(y, sizeof y, "f32", 3, xs)};
  et_kernel_request_v1 req = rq("layer-norm.forward", 3, sem);
  et_kernel_call_v1 c = kc("kernel.norm", &req, in, 4, out, 1);
  run(rt, &c);
  in[2] = tv(&epsilon, sizeof epsilon, "f32", 0, NULL);
  in[3] = tv(up, sizeof up, "f32", 3, xs);
  out[0] = tv(dx, sizeof dx, "f32", 3, xs);
  out[1] = tv(dg, sizeof dg, "f32", 1, ds);
  out[2] = tv(db, sizeof db, "f32", 1, ds);
  req.operation = "layer-norm.backward";
  c.output_count = 3;
  c.output_bytes = sizeof(out);
  run(rt, &c);
  req.operation = "layer-norm.forward";
  in[2] = tv(b, sizeof b, "f32", 1, ds);
  in[3] = tv(&epsilon, sizeof epsilon, "f32", 0, NULL);
  c.input_count = 4;
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  out[0] = tv(y, sizeof y, "f32", 3, xs);
  const float finite_difference_epsilon = 1.0e-3f;
  const float finite_difference_tolerance = 3.0e-3f;
  for (size_t i = 0; i < COUNT(x); i++) {
    const float original = x[i];
    const float step =
        finite_difference_step(original, finite_difference_epsilon);
    double positive;
    double negative;
    x[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    x[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    x[i] = original;
    CHECK(nearf(dx[i], (positive - negative) / (2.0f * step),
                finite_difference_tolerance));
  }
  for (size_t i = 0; i < COUNT(g); i++) {
    const float original = g[i];
    const float step =
        finite_difference_step(original, finite_difference_epsilon);
    double positive;
    double negative;
    g[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    g[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    g[i] = original;
    CHECK(nearf(dg[i], (positive - negative) / (2.0f * step),
                finite_difference_tolerance));
  }
  for (size_t i = 0; i < COUNT(b); i++) {
    const float original = b[i];
    const float step =
        finite_difference_step(original, finite_difference_epsilon);
    double positive;
    double negative;
    b[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    b[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    b[i] = original;
    CHECK(nearf(db[i], (positive - negative) / (2.0f * step),
                finite_difference_tolerance));
  }
  /* D=1 and constant rows are defined; epsilon rejects every
   * non-positive/nonfinite value atomically. */
  {
    uint64_t s[] = {1, 1, 1};
    float one = 7, gg = 2, bb = -3, oo = 99;
    in[0] = tv(&one, 4, "f32", 3, s);
    in[1] = tv(&gg, 4, "f32", 1, s + 2);
    in[2] = tv(&bb, 4, "f32", 1, s + 2);
    in[3] = tv(&epsilon, 4, "f32", 0, NULL);
    out[0] = tv(&oo, 4, "f32", 3, s);
    req.shape = s;
    run(rt, &c);
    CHECK(oo == bb);
  }
  req.shape = sem;
  in[0] = tv(x, sizeof x, "f32", 3, xs);
  in[1] = tv(g, sizeof g, "f32", 1, ds);
  in[2] = tv(b, sizeof b, "f32", 1, ds);
  out[0] = tv(y, sizeof y, "f32", 3, xs);
  const float bad[] = {0.0f, -0.0f, -1.0f, INFINITY, -INFINITY, NAN};
  for (size_t i = 0; i < COUNT(bad); i++) {
    epsilon = bad[i];
    float snap[3] = {91, 92, 93};
    memcpy(y, snap, sizeof y);
    reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(memcmp(y, snap, sizeof y) == 0);
  }
  epsilon = 1e-5f;
  x[0] = 1;
  x[1] = nextafterf(1, 2);
  x[2] = nextafterf(x[1], 2);
  run(rt, &c);
  for (size_t i = 0; i < 3; i++)
    CHECK(isfinite(y[i]));
}

static void test_activations(et_kernel_runtime *rt) {
  uint64_t s[] = {1, 1, 8};
  float x[] = {-8, -2, -0.0f, 0.0f, .5f, 1, 2, 8},
        up[] = {1, -2, 3, 4, -1, .5f, 2, -3}, y[8], dx[8];
  et_kernel_tensor_view_v1 in[] = {tv(x, sizeof x, "f32", 3, s),
                                   tv(up, sizeof up, "f32", 3, s)};
  et_kernel_tensor_view_v1 out[] = {tv(y, sizeof y, "f32", 3, s)};
  et_kernel_request_v1 req = rq("relu.forward", 3, s);
  et_kernel_call_v1 c = kc("kernel.activation", &req, in, 1, out, 1);
  run(rt, &c);
  CHECK(fbits(y[2]) == 0u && fbits(y[3]) == 0u && fbits(y[0]) == 0u &&
        y[7] == 8);
  req.operation = "relu.backward";
  c.input_count = 2;
  c.input_bytes = sizeof(in);
  out[0] = tv(dx, sizeof dx, "f32", 3, s);
  run(rt, &c);
  CHECK(fbits(dx[2]) == 0u && fbits(dx[3]) == 0u && dx[7] == up[7]);
  {
    const float finite_difference_epsilon = 1.0e-3f;
    const float finite_difference_tolerance = 2.0e-3f;
    const size_t samples[] = {0, 1, 4, 5, 6, 7};
    req.operation = "relu.forward";
    c.input_count = 1;
    c.input_bytes = sizeof(*in);
    out[0] = tv(y, sizeof y, "f32", 3, s);
    for (size_t j = 0; j < COUNT(samples); j++) {
      const size_t i = samples[j];
      const float original = x[i];
      const float step =
          finite_difference_step(original, finite_difference_epsilon);
      double positive;
      double negative;
      x[i] = original + step;
      run(rt, &c);
      positive = dot(y, up, COUNT(y));
      x[i] = original - step;
      run(rt, &c);
      negative = dot(y, up, COUNT(y));
      x[i] = original;
      CHECK(nearf(dx[i], (positive - negative) / (2.0f * step),
                  finite_difference_tolerance));
    }
  }
  req.operation = "gelu.forward";
  c.input_count = 1;
  c.input_bytes = sizeof(*in);
  out[0] = tv(y, sizeof y, "f32", 3, s);
  run(rt, &c);
  for (size_t i = 0; i < 8; i++)
    CHECK(isfinite(y[i]));
  req.operation = "gelu.backward";
  c.input_count = 2;
  c.input_bytes = sizeof(in);
  out[0] = tv(dx, sizeof dx, "f32", 3, s);
  run(rt, &c);
  req.operation = "gelu.forward";
  c.input_count = 1;
  c.input_bytes = sizeof(*in);
  out[0] = tv(y, sizeof y, "f32", 3, s);
  const size_t samples[] = {1, 4, 5, 6};
  const float finite_difference_epsilon = 1.0e-3f;
  const float finite_difference_tolerance = 3.0e-3f;
  for (size_t j = 0; j < COUNT(samples); j++) {
    const size_t i = samples[j];
    const float original = x[i];
    const float step =
        finite_difference_step(original, finite_difference_epsilon);
    double positive;
    double negative;
    x[i] = original + step;
    run(rt, &c);
    positive = dot(y, up, COUNT(y));
    x[i] = original - step;
    run(rt, &c);
    negative = dot(y, up, COUNT(y));
    x[i] = original;
    CHECK(nearf(dx[i], (positive - negative) / (2.0f * step),
                finite_difference_tolerance));
  }
}

static void test_residual(et_kernel_runtime *rt) {
  uint64_t s[] = {1, 3, 4};
  float x[12], b[12], y[12], up[12], dx[12], db[12];
  for (size_t i = 0; i < 12; i++) {
    x[i] = (float)i;
    b[i] = (float)(12 - i);
    up[i] = (float)i - .5f;
  }
  et_kernel_tensor_view_v1 in[] = {tv(x, sizeof x, "f32", 3, s),
                                   tv(b, sizeof b, "f32", 3, s)};
  et_kernel_tensor_view_v1 out[2] = {tv(y, sizeof y, "f32", 3, s)};
  et_kernel_request_v1 req = rq("residual.forward", 3, s);
  et_kernel_call_v1 c = kc("kernel.residual", &req, in, 2, out, 1);
  run(rt, &c);
  for (size_t i = 0; i < 12; i++)
    CHECK(y[i] == 12);
  in[1] = in[0];
  run(rt, &c);
  for (size_t i = 0; i < 12; i++)
    CHECK(y[i] == x[i] + x[i]);
  in[1] = tv((unsigned char *)x + 4, sizeof(x), "f32", 3, s);
  float snap[12];
  memcpy(snap, y, sizeof y);
  reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
         ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(snap, y, sizeof y) == 0);
  req.operation = "residual.backward";
  in[0] = tv(up, sizeof up, "f32", 3, s);
  c.input_count = 1;
  c.input_bytes = sizeof(*in);
  out[0] = tv(dx, sizeof dx, "f32", 3, s);
  out[1] = tv(db, sizeof db, "f32", 3, s);
  c.output_count = 2;
  c.output_bytes = sizeof(out);
  run(rt, &c);
  CHECK(memcmp(dx, up, sizeof up) == 0 && memcmp(db, up, sizeof up) == 0);
  req.operation = "residual.forward";
  in[0] = tv(x, sizeof x, "f32", 3, s);
  in[1] = tv(b, sizeof b, "f32", 3, s);
  out[0] = tv(y, sizeof y, "f32", 3, s);
  c.input_count = 2;
  c.input_bytes = sizeof(in);
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  {
    const float finite_difference_epsilon = 1.0e-3f;
    const float finite_difference_tolerance = 1.0e-2f;
    float *operands[] = {x, b};
    const float *gradients[] = {dx, db};
    for (size_t operand = 0; operand < COUNT(operands); operand++) {
      for (size_t i = 0; i < COUNT(x); i++) {
        const float original = operands[operand][i];
        const float step =
            finite_difference_step(original, finite_difference_epsilon);
        double positive;
        double negative;
        operands[operand][i] = original + step;
        run(rt, &c);
        positive = dot(y, up, COUNT(y));
        operands[operand][i] = original - step;
        run(rt, &c);
        negative = dot(y, up, COUNT(y));
        operands[operand][i] = original;
        CHECK(nearf(gradients[operand][i],
                    (positive - negative) / (2.0f * step),
                    finite_difference_tolerance));
      }
    }
  }
}

static void reference_philox(uint32_t c[4], uint32_t k[2]) {
  for (size_t round = 0; round < 10u; round++) {
    uint64_t p0 = UINT64_C(0xd2511f53) * c[0];
    uint64_t p1 = UINT64_C(0xcd9e8d57) * c[2];
    uint32_t n0 = (uint32_t)(p1 >> 32) ^ c[1] ^ k[0];
    uint32_t n1 = (uint32_t)p1;
    uint32_t n2 = (uint32_t)(p0 >> 32) ^ c[3] ^ k[1];
    uint32_t n3 = (uint32_t)p0;
    c[0] = n0;
    c[1] = n1;
    c[2] = n2;
    c[3] = n3;
    if (round != 9u) {
      k[0] += UINT32_C(0x9e3779b9);
      k[1] += UINT32_C(0xbb67ae85);
    }
  }
}

static void test_official_philox_vectors(void) {
  static const uint32_t inputs[][6] = {
      {0, 0, 0, 0, 0, 0},
      {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
      {UINT32_C(0x243f6a88), UINT32_C(0x85a308d3), UINT32_C(0x13198a2e),
       UINT32_C(0x03707344), UINT32_C(0xa4093822), UINT32_C(0x299f31d0)}};
  static const uint32_t outputs[][4] = {
      {UINT32_C(0x6627e8d5), UINT32_C(0xe169c58d), UINT32_C(0xbc57ac4c),
       UINT32_C(0x9b00dbd8)},
      {UINT32_C(0x408f276d), UINT32_C(0x41c83b0e), UINT32_C(0xa20bc7c6),
       UINT32_C(0x6d5451fd)},
      {UINT32_C(0xd16cfe09), UINT32_C(0x94fdcceb), UINT32_C(0x5001e420),
       UINT32_C(0x24126ea1)}};
  for (size_t i = 0; i < COUNT(inputs); i++) {
    uint32_t c[4] = {inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]};
    uint32_t k[2] = {inputs[i][4], inputs[i][5]};
    reference_philox(c, k);
    CHECK(memcmp(c, outputs[i], sizeof c) == 0);
  }
}

static void test_dropout(et_kernel_runtime *rt) {
  uint64_t s4[] = {1, 1, 4}, state_shape[] = {4};
  float x[] = {1, 1, 1, 1}, up[] = {2, 2, 2, 2}, y[4], dx[4], p = .5f;
  int64_t state[] = {1, 0, 0, 0}, next[4];
  et_kernel_tensor_view_v1 in[] = {
      tv(x, sizeof x, "f32", 3, s4), tv(&p, 4, "f32", 0, NULL),
      tv(state, sizeof state, "i64", 1, state_shape)};
  et_kernel_tensor_view_v1 out[] = {
      tv(y, sizeof y, "f32", 3, s4),
      tv(next, sizeof next, "i64", 1, state_shape)};
  et_kernel_request_v1 req = rq("dropout.train.forward", 3, s4);
  et_kernel_call_v1 c = kc("kernel.dropout", &req, in, 3, out, 2);
  run(rt, &c);
  /* Random123 zero vector: high-byte uniforms 0x66,0xe1,0xbc,0x9b. */
  {
    const float e[] = {0, 2, 2, 2};
    expect_floats(y, e, 4, 0);
  }
  CHECK(next[0] == 1 && next[1] == 0 && next[2] == 1 && next[3] == 0);
  /* The nontrivial official vector's lane order is visible through the mask. */
  state[1] = signed_bits(UINT64_C(0x299f31d0a4093822));
  state[2] = signed_bits(UINT64_C(0x85a308d3243f6a88));
  state[3] = signed_bits(UINT64_C(0x0370734413198a2e));
  run(rt, &c);
  {
    const float e[] = {2, 2, 0, 0};
    expect_floats(y, e, 4, 0);
  }
  state[1] = 0;
  state[2] = 0;
  state[3] = 0;
  in[0] = tv(up, sizeof up, "f32", 3, s4);
  out[0] = tv(dx, sizeof dx, "f32", 3, s4);
  req.operation = "dropout.train.backward";
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  run(rt, &c);
  {
    const float e[] = {0, 4, 4, 4};
    expect_floats(dx, e, 4, 0);
  }
  /* Training is differentiable with respect to x for a fixed original RNG
   * state. Re-dispatch both sides from that same state so the mask is fixed. */
  req.operation = "dropout.train.forward";
  in[0] = tv(x, sizeof x, "f32", 3, s4);
  out[0] = tv(y, sizeof y, "f32", 3, s4);
  out[1] = tv(next, sizeof next, "i64", 1, state_shape);
  c.input_count = 3;
  c.input_bytes = sizeof(in);
  c.output_count = 2;
  c.output_bytes = sizeof(out);
  {
    const float finite_difference_epsilon = 1.0e-3f;
    const float finite_difference_tolerance = 2.0e-3f;
    for (size_t i = 0; i < COUNT(x); i++) {
      const float original = x[i];
      const float step =
          finite_difference_step(original, finite_difference_epsilon);
      double positive;
      double negative;
      x[i] = original + step;
      run(rt, &c);
      positive = dot(y, up, COUNT(y));
      x[i] = original - step;
      run(rt, &c);
      negative = dot(y, up, COUNT(y));
      x[i] = original;
      CHECK(nearf(dx[i], (positive - negative) / (2.0f * step),
                  finite_difference_tolerance));
    }
  }
  /* All lane remainders and tail discard: each call advances ceil(E/4). */
  for (uint64_t e = 1; e <= 4; e++) {
    uint64_t ss[] = {1, 1, e};
    float xx[4] = {1, 1, 1, 1}, yy[4];
    int64_t nn[4];
    in[0] = tv(xx, e * 4, "f32", 3, ss);
    out[0] = tv(yy, e * 4, "f32", 3, ss);
    out[1] = tv(nn, sizeof nn, "i64", 1, state_shape);
    req.operation = "dropout.train.forward";
    req.shape = ss;
    c.output_count = 2;
    c.output_bytes = sizeof(out);
    run(rt, &c);
    CHECK(nn[2] == 1);
  }
  /* A split 2+2 consumes two blocks and therefore differs from one length-4
   * call. */
  {
    uint64_t s2[] = {1, 1, 2};
    float a[2], bb[2];
    int64_t n1[4], n2[4];
    in[0] = tv(x, 8, "f32", 3, s2);
    out[0] = tv(a, 8, "f32", 3, s2);
    out[1] = tv(n1, sizeof n1, "i64", 1, state_shape);
    req.shape = s2;
    run(rt, &c);
    in[0] = tv(x + 2, 8, "f32", 3, s2);
    in[2] = tv(n1, sizeof n1, "i64", 1, state_shape);
    out[0] = tv(bb, 8, "f32", 3, s2);
    out[1] = tv(n2, sizeof n2, "i64", 1, state_shape);
    run(rt, &c);
    CHECK(n2[2] == 2);
    in[2] = tv(state, sizeof state, "i64", 1, state_shape);
  }
  /* Signed i64 encoding, 64-bit carry, and max-1 to max / exhausted sentinel.
   */
  state[1] = signed_bits(UINT64_MAX);
  state[2] = signed_bits(UINT64_MAX);
  state[3] = 0;
  req.shape = s4;
  in[0] = tv(x, sizeof x, "f32", 3, s4);
  out[0] = tv(y, sizeof y, "f32", 3, s4);
  out[1] = tv(next, sizeof next, "i64", 1, state_shape);
  run(rt, &c);
  CHECK((uint64_t)next[2] == 0u && next[3] == 1);
  state[2] = signed_bits(UINT64_MAX - 1);
  state[3] = signed_bits(UINT64_MAX);
  run(rt, &c);
  CHECK((uint64_t)next[2] == UINT64_MAX && (uint64_t)next[3] == UINT64_MAX);
  state[2] = signed_bits(UINT64_MAX);
  state[3] = signed_bits(UINT64_MAX);
  float snap[4] = {81, 82, 83, 84};
  memcpy(y, snap, sizeof y);
  reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
         ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(y, snap, sizeof y) == 0);
  /* All signed-encoding boundary words decode and the key is copied exactly. */
  {
    const int64_t words[] = {0, INT64_MAX, INT64_MIN, -1};
    state[2] = 0;
    state[3] = 0;
    for (size_t i = 0; i < COUNT(words); i++) {
      state[1] = words[i];
      run(rt, &c);
      CHECK(next[1] == words[i] && next[2] == 1 && next[3] == 0);
    }
  }
  /* Both zeros and eval are exact byte-copy paths, consume no RNG, and accept
   * max. */
  float zeros[] = {0.0f, -0.0f, 1, -1};
  for (size_t z = 0; z < 2; z++) {
    p = z ? -0.0f : 0.0f;
    in[0] = tv(zeros, sizeof zeros, "f32", 3, s4);
    run(rt, &c);
    CHECK(memcmp(y, zeros, sizeof y) == 0 &&
          memcmp(next, state, sizeof next) == 0);
  }
  req.operation = "dropout.eval.forward";
  c.input_count = 1;
  c.input_bytes = sizeof(*in);
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  run(rt, &c);
  CHECK(memcmp(y, zeros, sizeof y) == 0);
  in[0] = tv(up, sizeof up, "f32", 3, s4);
  out[0] = tv(dx, sizeof dx, "f32", 3, s4);
  req.operation = "dropout.eval.backward";
  run(rt, &c);
  CHECK(memcmp(dx, up, sizeof dx) == 0);
  req.operation = "dropout.eval.forward";
  in[0] = tv(x, sizeof x, "f32", 3, s4);
  out[0] = tv(y, sizeof y, "f32", 3, s4);
  {
    const float finite_difference_epsilon = 1.0e-3f;
    const float finite_difference_tolerance = 2.0e-3f;
    for (size_t i = 0; i < COUNT(x); i++) {
      const float original = x[i];
      const float step =
          finite_difference_step(original, finite_difference_epsilon);
      double positive;
      double negative;
      x[i] = original + step;
      run(rt, &c);
      positive = dot(y, up, COUNT(y));
      x[i] = original - step;
      run(rt, &c);
      negative = dot(y, up, COUNT(y));
      x[i] = original;
      CHECK(nearf(dx[i], (positive - negative) / (2.0f * step),
                  finite_difference_tolerance));
    }
  }
  /* Unknown version and malformed probabilities preserve sentinel outputs. */
  req.operation = "dropout.train.forward";
  c.input_count = 3;
  c.input_bytes = sizeof(in);
  c.output_count = 2;
  c.output_bytes = sizeof(out);
  in[0] = tv(x, sizeof x, "f32", 3, s4);
  in[2] = tv(state, sizeof state, "i64", 1, state_shape);
  out[0] = tv(y, sizeof y, "f32", 3, s4);
  out[1] = tv(next, sizeof next, "i64", 1, state_shape);
  state[0] = 2;
  p = .5f;
  reject(rt, &c, ET_KERNEL_ERROR_VERSION_MISMATCH, UINT32_MAX);
  state[0] = 1;
  const float bad[] = {-1, -0.1f, 1, INFINITY, NAN};
  for (size_t i = 0; i < COUNT(bad); i++) {
    p = bad[i];
    memcpy(y, snap, sizeof y);
    reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(memcmp(y, snap, sizeof y) == 0);
  }
  p = .5f;

  /* Checkpoint replay is byte-identical and neither backward nor a rejection
   * mutates state. */
  state[0] = 1;
  state[1] = signed_bits(UINT64_C(0x8000000000000000));
  state[2] = 7;
  state[3] = 9;
  int64_t state_copy[4];
  float first[4];
  memcpy(state_copy, state, sizeof state);
  run(rt, &c);
  memcpy(first, y, sizeof y);
  CHECK(memcmp(state, state_copy, sizeof state) == 0);
  memset(y, 0xa5, sizeof y);
  run(rt, &c);
  CHECK(memcmp(y, first, sizeof y) == 0 &&
        memcmp(state, state_copy, sizeof state) == 0);
  req.operation = "dropout.train.backward";
  c.output_count = 1;
  c.output_bytes = sizeof(*out);
  in[0] = tv(up, sizeof up, "f32", 3, s4);
  out[0] = tv(dx, sizeof dx, "f32", 3, s4);
  run(rt, &c);
  CHECK(memcmp(state, state_copy, sizeof state) == 0);
}

/* Execute every operation on every advertised row, independent of richer cases.
 */
static void test_every_row(et_kernel_runtime *rt) {
  static const uint64_t e[][4] = {{1, 1, 1, 1}, {1, 4, 3, 2}, {2, 3, 5, 6}};
  static const uint64_t l[][4] = {{1, 1, 1, 1}, {1, 2, 3, 4}, {2, 3, 4, 5}};
  static const uint64_t n[][3] = {{1, 1, 1}, {1, 1, 3}, {1, 2, 4}, {2, 2, 4}};
  static const uint64_t arows[][3] = {
      {1, 1, 1}, {1, 1, 8}, {1, 2, 5}, {2, 2, 4}};
  static const uint64_t d[][3] = {
      {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 1, 4}, {1, 2, 5}};
  static const uint64_t r[][3] = {{1, 1, 1}, {1, 3, 4}, {2, 2, 4}};
  for (size_t q = 0; q < COUNT(e); q++) {
    size_t nt = e[q][0] * e[q][1], vd = e[q][2] * e[q][3], ny = nt * e[q][3];
    int64_t *ids = (int64_t *)calloc(nt, 8);
    float *w = (float *)calloc(vd, 4), *up = (float *)calloc(ny, 4),
          *outp = (float *)calloc(vd > ny ? vd : ny, 4);
    CHECK(ids && w && up && outp);
    uint64_t is[] = {e[q][0], e[q][1]}, ws[] = {e[q][2], e[q][3]},
             ys[] = {e[q][0], e[q][1], e[q][3]};
    et_kernel_tensor_view_v1 in[] = {tv(ids, nt * 8, "i64", 2, is),
                                     tv(w, vd * 4, "f32", 2, ws)},
                             out[] = {tv(outp, ny * 4, "f32", 3, ys)};
    et_kernel_request_v1 req = rq("embedding.forward", 4, e[q]);
    et_kernel_call_v1 c = kc("kernel.embedding-forward", &req, in, 2, out, 1);
    run(rt, &c);
    in[1] = tv(up, ny * 4, "f32", 3, ys);
    out[0] = tv(outp, vd * 4, "f32", 2, ws);
    req.operation = "embedding.backward";
    c.capability = "kernel.embedding-backward";
    run(rt, &c);
    free(ids);
    free(w);
    free(up);
    free(outp);
  }
  for (size_t q = 0; q < COUNT(l); q++) {
    size_t nt = l[q][0] * l[q][1], nx = nt * l[q][2], nw = l[q][3] * l[q][2],
           ny = nt * l[q][3];
    float *x = (float *)calloc(nx, 4), *w = (float *)calloc(nw, 4),
          *b = (float *)calloc(l[q][3], 4), *up = (float *)calloc(ny, 4),
          *y = (float *)calloc(ny, 4), *dx = (float *)calloc(nx, 4),
          *dw = (float *)calloc(nw, 4), *db = (float *)calloc(l[q][3], 4);
    CHECK(x && w && b && up && y && dx && dw && db);
    uint64_t xs[] = {l[q][0], l[q][1], l[q][2]}, ws[] = {l[q][3], l[q][2]},
             bs[] = {l[q][3]}, ys[] = {l[q][0], l[q][1], l[q][3]};
    et_kernel_tensor_view_v1 in[] = {tv(x, nx * 4, "f32", 3, xs),
                                     tv(w, nw * 4, "f32", 2, ws),
                                     tv(b, l[q][3] * 4, "f32", 1, bs)},
                             out[] = {tv(y, ny * 4, "f32", 3, ys),
                                      tv(dw, nw * 4, "f32", 2, ws),
                                      tv(db, l[q][3] * 4, "f32", 1, bs)};
    et_kernel_request_v1 req = rq("linear.forward", 4, l[q]);
    et_kernel_call_v1 c = kc("kernel.matmul", &req, in, 3, out, 1);
    run(rt, &c);
    req.operation = "linear.forward-no-bias";
    c.input_count = 2;
    c.input_bytes = 2 * sizeof(*in);
    run(rt, &c);
    in[2] = tv(up, ny * 4, "f32", 3, ys);
    out[0] = tv(dx, nx * 4, "f32", 3, xs);
    c.input_count = 3;
    c.input_bytes = sizeof(in);
    c.output_count = 3;
    c.output_bytes = sizeof(out);
    req.operation = "linear.backward";
    run(rt, &c);
    req.operation = "linear.backward-no-bias";
    c.output_count = 2;
    c.output_bytes = 2 * sizeof(*out);
    run(rt, &c);
    free(x);
    free(w);
    free(b);
    free(up);
    free(y);
    free(dx);
    free(dw);
    free(db);
  }
  for (size_t q = 0; q < COUNT(n); q++) {
    size_t z = n[q][0] * n[q][1] * n[q][2];
    float *x = (float *)calloc(z, 4), *g = (float *)calloc(n[q][2], 4),
          *b = (float *)calloc(n[q][2], 4), *up = (float *)calloc(z, 4),
          *a = (float *)calloc(z, 4),
          *b1 = (float *)calloc(z > n[q][2] ? z : n[q][2], 4),
          *b2 = (float *)calloc(z > n[q][2] ? z : n[q][2], 4);
    CHECK(x && g && b && up && a && b1 && b2);
    for (size_t i = 0; i < n[q][2]; i++)
      g[i] = 1;
    float eps = 1e-5f;
    uint64_t ds[] = {n[q][2]};
    et_kernel_tensor_view_v1 in[] = {tv(x, z * 4, "f32", 3, n[q]),
                                     tv(g, n[q][2] * 4, "f32", 1, ds),
                                     tv(b, n[q][2] * 4, "f32", 1, ds),
                                     tv(&eps, 4, "f32", 0, NULL)},
                             out[] = {tv(a, z * 4, "f32", 3, n[q]),
                                      tv(b1, n[q][2] * 4, "f32", 1, ds),
                                      tv(b2, n[q][2] * 4, "f32", 1, ds)};
    et_kernel_request_v1 req = rq("layer-norm.forward", 3, n[q]);
    et_kernel_call_v1 c = kc("kernel.norm", &req, in, 4, out, 1);
    run(rt, &c);
    in[2] = tv(&eps, 4, "f32", 0, NULL);
    in[3] = tv(up, z * 4, "f32", 3, n[q]);
    out[0] = tv(a, z * 4, "f32", 3, n[q]);
    c.output_count = 3;
    c.output_bytes = sizeof(out);
    req.operation = "layer-norm.backward";
    run(rt, &c);
    free(x);
    free(g);
    free(b);
    free(up);
    free(a);
    free(b1);
    free(b2);
  }
  for (size_t q = 0; q < COUNT(arows); q++) {
    size_t z = arows[q][0] * arows[q][1] * arows[q][2];
    float *x = (float *)calloc(z, 4), *up = (float *)calloc(z, 4),
          *out_data = (float *)calloc(z, 4);
    CHECK(x && up && out_data);
    et_kernel_tensor_view_v1 in[] = {tv(x, z * 4, "f32", 3, arows[q]),
                                     tv(up, z * 4, "f32", 3, arows[q])};
    et_kernel_tensor_view_v1 out[] = {tv(out_data, z * 4, "f32", 3, arows[q])};
    et_kernel_request_v1 req = rq("relu.forward", 3, arows[q]);
    et_kernel_call_v1 c = kc("kernel.activation", &req, in, 1, out, 1);
    const char *ops[] = {"relu.forward", "relu.backward", "gelu.forward",
                         "gelu.backward"};
    for (size_t i = 0; i < COUNT(ops); i++) {
      req.operation = ops[i];
      c.input_count = strstr(ops[i], "backward") ? 2u : 1u;
      c.input_bytes = c.input_count * sizeof(*in);
      run(rt, &c);
    }
    free(x);
    free(up);
    free(out_data);
  }
  for (size_t q = 0; q < COUNT(d); q++) {
    size_t z = d[q][0] * d[q][1] * d[q][2];
    float *x = (float *)calloc(z, 4), *y = (float *)calloc(z, 4);
    CHECK(x && y);
    float p = .25f;
    int64_t st[] = {1, 0, 0, 0}, ns[4];
    uint64_t ss[] = {4};
    et_kernel_tensor_view_v1 in[] = {tv(x, z * 4, "f32", 3, d[q]),
                                     tv(&p, 4, "f32", 0, NULL),
                                     tv(st, 32, "i64", 1, ss)},
                             out[] = {tv(y, z * 4, "f32", 3, d[q]),
                                      tv(ns, 32, "i64", 1, ss)};
    et_kernel_request_v1 req = rq("dropout.train.forward", 3, d[q]);
    et_kernel_call_v1 c = kc("kernel.dropout", &req, in, 3, out, 2);
    run(rt, &c);
    req.operation = "dropout.train.backward";
    c.output_count = 1;
    c.output_bytes = sizeof(*out);
    run(rt, &c);
    req.operation = "dropout.eval.forward";
    c.input_count = 1;
    c.input_bytes = sizeof(*in);
    run(rt, &c);
    req.operation = "dropout.eval.backward";
    run(rt, &c);
    free(x);
    free(y);
  }
  for (size_t q = 0; q < COUNT(r); q++) {
    size_t z = r[q][0] * r[q][1] * r[q][2];
    float *x = (float *)calloc(z, 4), *b = (float *)calloc(z, 4),
          *y = (float *)calloc(z, 4), *o = (float *)calloc(z, 4);
    CHECK(x && b && y && o);
    et_kernel_tensor_view_v1 in[] = {tv(x, z * 4, "f32", 3, r[q]),
                                     tv(b, z * 4, "f32", 3, r[q])},
                             out[] = {tv(y, z * 4, "f32", 3, r[q]),
                                      tv(o, z * 4, "f32", 3, r[q])};
    et_kernel_request_v1 req = rq("residual.forward", 3, r[q]);
    et_kernel_call_v1 c = kc("kernel.residual", &req, in, 2, out, 1);
    run(rt, &c);
    req.operation = "residual.backward";
    c.input_count = 1;
    c.input_bytes = sizeof(*in);
    c.output_count = 2;
    c.output_bytes = sizeof(out);
    run(rt, &c);
    free(x);
    free(b);
    free(y);
    free(o);
  }
}

static void test_generic_failures(et_kernel_runtime *rt) {
  uint64_t s[] = {1, 1, 1};
  float x = 1, y = 91, z = 2;
  et_kernel_tensor_view_v1 in[] = {tv(&x, 4, "f32", 3, s)},
                           out[] = {tv(&y, 4, "f32", 3, s)};
  et_kernel_request_v1 req = rq("relu.forward", 3, s);
  et_kernel_call_v1 c = kc("kernel.activation", &req, in, 1, out, 1);
  float sent = y;
  in[0].dtype = "f64";
  reject(rt, &c, ET_KERNEL_ERROR_DTYPE_MISMATCH, UINT32_MAX);
  CHECK(y == sent);
  in[0].dtype = "f32";
  in[0].device = "cuda";
  reject(rt, &c, ET_KERNEL_ERROR_DEVICE_MISMATCH, UINT32_MAX);
  CHECK(y == sent);
  in[0].device = "cpu";
  {
    et_kernel_error error;
    req.device = "cuda";
    in[0].device = "cuda";
    out[0].device = "cuda";
    reject(rt, &c, ET_KERNEL_ERROR_UNSUPPORTED,
           ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    CHECK(y == sent);
    CHECK(et_n2_kernel_provider_v1()->validate_call(&c, &error) != 0);
    CHECK(error.category == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(y == sent);
    req.device = "cpu";
    in[0].device = "cpu";
    out[0].device = "cpu";
  }
  in[0].layout = 99;
  reject(rt, &c, ET_KERNEL_ERROR_NONCONTIGUOUS, UINT32_MAX);
  CHECK(y == sent);
  in[0].layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  in[0].offset_bytes = 4;
  reject(rt, &c, ET_KERNEL_ERROR_NONCONTIGUOUS, UINT32_MAX);
  CHECK(y == sent);
  in[0].offset_bytes = 0;
  in[0].byte_length = 3;
  reject(rt, &c, ET_KERNEL_ERROR_SHAPE_MISMATCH, UINT32_MAX);
  CHECK(y == sent);
  in[0].byte_length = 4;
  in[0].rank = 2;
  reject(rt, &c, ET_KERNEL_ERROR_SHAPE_MISMATCH, UINT32_MAX);
  CHECK(y == sent);
  in[0].rank = 3;
  {
    uint64_t wrong_shape[] = {1, 1, 2};
    float two[] = {1, 2};
    in[0] = tv(two, sizeof two, "f32", 3, wrong_shape);
    reject(rt, &c, ET_KERNEL_ERROR_SHAPE_MISMATCH, UINT32_MAX);
    CHECK(y == sent);
  }
  {
    _Alignas(float) unsigned char raw[sizeof(float) + 1u] = {0};
    in[0] = tv(raw + 1, sizeof(float), "f32", 3, s);
    reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT, UINT32_MAX);
    CHECK(y == sent);
  }
  in[0] = tv(&x, sizeof x, "f32", 3, s);
  out[0].data = &x;
  reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
         ET_KERNEL_CODE_ALIASING_OUTPUT);
  out[0].data = &y;
  x = NAN;
  reject(rt, &c, ET_KERNEL_ERROR_INVALID_ARGUMENT,
         ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(y == sent);
  x = 1;
  req.operation = "relu.forward";
  /* Initial output sNaN bytes are ignored and fully overwritten on success. */
  {
    uint32_t sn = UINT32_C(0x7f800001);
    memcpy(&y, &sn, 4);
    run(rt, &c);
    CHECK(y == 1);
  }
  /* Non-default rounding and x86 FTZ/DAZ are unsupported without environment
   * repair. */
  sent = y;
  CHECK(fesetround(FE_DOWNWARD) == 0);
  reject(rt, &c, ET_KERNEL_ERROR_UNSUPPORTED, UINT32_MAX);
  CHECK(y == sent);
  CHECK(fesetround(FE_TONEAREST) == 0);
#if defined(__x86_64__)
  {
    unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | UINT32_C(0x8000));
    reject(rt, &c, ET_KERNEL_ERROR_UNSUPPORTED, UINT32_MAX);
    CHECK(y == sent);
    _mm_setcsr(saved);
  }
  {
    unsigned saved = _mm_getcsr();
    _mm_setcsr(saved | UINT32_C(0x0040));
    reject(rt, &c, ET_KERNEL_ERROR_UNSUPPORTED, UINT32_MAX);
    CHECK(y == sent);
    _mm_setcsr(saved);
  }
  {
    static const unsigned exception_masks[] = {
        UINT32_C(0x0080), UINT32_C(0x0100), UINT32_C(0x0200),
        UINT32_C(0x0400), UINT32_C(0x0800), UINT32_C(0x1000)};
    const unsigned saved = _mm_getcsr();
    for (size_t i = 0; i < COUNT(exception_masks); i++) {
      const unsigned unsupported = saved & ~exception_masks[i];
      _mm_setcsr(unsupported);
      reject(rt, &c, ET_KERNEL_ERROR_UNSUPPORTED,
             ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(y == sent);
      CHECK(_mm_getcsr() == unsupported);
    }
    _mm_setcsr(saved);
  }
#endif
  /* Provider-level input/input overlap uses provider-rejected, never
   * aliasing-output. */
  {
    uint64_t rs[] = {1, 1, 1};
    et_kernel_tensor_view_v1 rin[] = {tv(&x, 4, "f32", 3, rs),
                                      tv(&x, 4, "f32", 3, rs)},
                             rout[] = {tv(&z, 4, "f32", 3, rs)};
    et_kernel_request_v1 rr = rq("residual.forward", 3, rs);
    et_kernel_call_v1 cc = kc("kernel.residual", &rr, rin, 2, rout,
                              1); /* exact residual alias is allowed */
    run(rt, &cc);
    rr.operation = "gelu.backward";
    cc.capability = "kernel.activation";
    reject(rt, &cc, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  /* Provider validation restores the caller's complete floating exception set.
   */
  {
    et_kernel_error e;
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
    CHECK(feraiseexcept(FE_DIVBYZERO | FE_INEXACT) == 0);
    int before = fetestexcept(FE_ALL_EXCEPT);
    CHECK(et_n2_kernel_provider_v1()->validate_call(&c, &e) == 0);
    CHECK(fetestexcept(FE_ALL_EXCEPT) == before);
    CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  }
  /* A finite-input overflow is discovered by dry evaluation before mutation. */
  {
    uint64_t rs[] = {1, 1, 1};
    float a = FLT_MAX, b = FLT_MAX, o = 73;
    et_kernel_tensor_view_v1 rin[] = {tv(&a, 4, "f32", 3, rs),
                                      tv(&b, 4, "f32", 3, rs)},
                             rout[] = {tv(&o, 4, "f32", 3, rs)};
    et_kernel_request_v1 rr = rq("residual.forward", 3, rs);
    et_kernel_call_v1 cc = kc("kernel.residual", &rr, rin, 2, rout, 1);
    reject(rt, &cc, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(o == 73);
  }
}

typedef struct fixture_storage {
  void *data;
  size_t bytes;
  uint64_t shape[4];
} fixture_storage;

static fixture_storage fixture_copy(const struct et_n2_reference_tensor *tensor,
                                    int output) {
  fixture_storage storage = {0};
  const size_t word_size =
      tensor->dtype == ET_N2_REFERENCE_F32 ? sizeof(uint32_t) : sizeof(int64_t);
  CHECK(tensor->rank <= COUNT(storage.shape));
  CHECK(tensor->element_count <= SIZE_MAX / word_size);
  storage.bytes = (size_t)tensor->element_count * word_size;
  storage.data = malloc(storage.bytes);
  CHECK(storage.data != NULL);
  if (output)
    memset(storage.data, 0xa5, storage.bytes);
  else
    memcpy(storage.data, tensor->words, storage.bytes);
  for (size_t d = 0; d < tensor->rank; d++) {
    CHECK(tensor->shape[d] > 0);
    storage.shape[d] = (uint64_t)tensor->shape[d];
  }
  return storage;
}

static const char *fixture_capability(const char *operation) {
  if (strncmp(operation, "embedding.forward", 18) == 0)
    return "kernel.embedding-forward";
  if (strncmp(operation, "embedding.backward", 19) == 0)
    return "kernel.embedding-backward";
  if (strncmp(operation, "linear.", 7) == 0)
    return "kernel.matmul";
  if (strncmp(operation, "layer-norm.", 11) == 0)
    return "kernel.norm";
  if (strncmp(operation, "gelu.", 5) == 0 ||
      strncmp(operation, "relu.", 5) == 0)
    return "kernel.activation";
  if (strncmp(operation, "dropout.", 8) == 0)
    return "kernel.dropout";
  if (strncmp(operation, "residual.", 9) == 0)
    return "kernel.residual";
  CHECK(0);
  return NULL;
}

static size_t
fixture_request_shape(const struct et_n2_reference_case *reference_case,
                      const struct et_n2_reference_tensor *const *inputs,
                      const struct et_n2_reference_tensor *const *outputs,
                      uint64_t shape[4]) {
  if (strncmp(reference_case->operation, "embedding.", 10) == 0) {
    const struct et_n2_reference_tensor *ids = inputs[0];
    const struct et_n2_reference_tensor *matrix =
        strcmp(reference_case->operation, "embedding.forward") == 0
            ? inputs[1]
            : outputs[0];
    shape[0] = (uint64_t)ids->shape[0];
    shape[1] = (uint64_t)ids->shape[1];
    shape[2] = (uint64_t)matrix->shape[0];
    shape[3] = (uint64_t)matrix->shape[1];
    return 4u;
  }
  if (strncmp(reference_case->operation, "linear.", 7) == 0) {
    shape[0] = (uint64_t)inputs[0]->shape[0];
    shape[1] = (uint64_t)inputs[0]->shape[1];
    shape[2] = (uint64_t)inputs[0]->shape[2];
    shape[3] = (uint64_t)inputs[1]->shape[0];
    return 4u;
  }
  CHECK(inputs[0]->rank == 3u);
  for (size_t d = 0; d < 3u; d++)
    shape[d] = (uint64_t)inputs[0]->shape[d];
  return 3u;
}

static void
expect_fixture_output(const struct et_n2_reference_case *reference_case,
                      const struct et_n2_reference_tensor *expected,
                      const void *actual) {
  const size_t bytes =
      (size_t)expected->element_count * (expected->dtype == ET_N2_REFERENCE_F32
                                             ? sizeof(uint32_t)
                                             : sizeof(int64_t));
  const int exact = expected->dtype == ET_N2_REFERENCE_I64 ||
                    strncmp(reference_case->operation, "dropout.", 8) == 0 ||
                    strncmp(reference_case->operation, "relu.", 5) == 0 ||
                    strncmp(reference_case->operation, "residual.", 9) == 0;
  if (exact) {
    if (memcmp(actual, expected->words, bytes) != 0)
      fprintf(stderr, "frozen exact mismatch: %s / %s\n", reference_case->name,
              expected->name);
    CHECK(memcmp(actual, expected->words, bytes) == 0);
    return;
  }
  {
    const float absolute =
        strcmp(reference_case->kind, "gradient") == 0 ? 5.0e-5f : 2.0e-5f;
    const float relative = absolute;
    const float *values = (const float *)actual;
    const uint32_t *bits = (const uint32_t *)expected->words;
    for (size_t i = 0; i < expected->element_count; i++) {
      float want;
      memcpy(&want, &bits[i], sizeof(want));
      const float tolerance = absolute + relative * fabsf(want);
      if (!nearf(values[i], want, tolerance))
        fprintf(stderr,
                "frozen mismatch: %s / %s[%zu] got %.9g want %.9g tol %.9g\n",
                reference_case->name, expected->name, i, values[i], want,
                tolerance);
      CHECK(nearf(values[i], want, tolerance));
    }
  }
}

static void test_frozen_reference_cases(et_kernel_runtime *rt) {
  CHECK(COUNT(et_n2_reference_tensors) == ET_N2_REFERENCE_TENSOR_COUNT);
  CHECK(COUNT(et_n2_reference_cases) == ET_N2_REFERENCE_CASE_COUNT);
  CHECK(strlen(ET_N2_REFERENCE_FIXTURE_SHA256) == 64u);
  for (size_t ci = 0; ci < COUNT(et_n2_reference_cases); ci++) {
    const struct et_n2_reference_case *reference_case =
        &et_n2_reference_cases[ci];
    fixture_storage input_storage[4] = {{0}}, output_storage[3] = {{0}};
    const struct et_n2_reference_tensor *input_tensors[4] = {0};
    const struct et_n2_reference_tensor *output_tensors[3] = {0};
    et_kernel_tensor_view_v1 inputs[4] = {{0}}, outputs[3] = {{0}};
    uint64_t semantic_shape[4] = {0};
    CHECK(reference_case->input_count <= COUNT(inputs));
    CHECK(reference_case->output_count <= COUNT(outputs));
    for (size_t i = 0; i < reference_case->input_count; i++) {
      CHECK(reference_case->inputs[i] < ET_N2_REFERENCE_TENSOR_COUNT);
      input_tensors[i] = &et_n2_reference_tensors[reference_case->inputs[i]];
      CHECK(input_tensors[i]->role == ET_N2_REFERENCE_INPUT);
      input_storage[i] = fixture_copy(input_tensors[i], 0);
      inputs[i] =
          tv(input_storage[i].data, input_storage[i].bytes,
             input_tensors[i]->dtype == ET_N2_REFERENCE_F32 ? "f32" : "i64",
             input_tensors[i]->rank, input_storage[i].shape);
    }
    for (size_t i = 0; i < reference_case->output_count; i++) {
      CHECK(reference_case->outputs[i] < ET_N2_REFERENCE_TENSOR_COUNT);
      output_tensors[i] = &et_n2_reference_tensors[reference_case->outputs[i]];
      CHECK(output_tensors[i]->role == ET_N2_REFERENCE_EXPECTED ||
            output_tensors[i]->role == ET_N2_REFERENCE_ANALYTIC_GRADIENT);
      output_storage[i] = fixture_copy(output_tensors[i], 1);
      outputs[i] =
          tv(output_storage[i].data, output_storage[i].bytes,
             output_tensors[i]->dtype == ET_N2_REFERENCE_F32 ? "f32" : "i64",
             output_tensors[i]->rank, output_storage[i].shape);
    }
    {
      const size_t rank = fixture_request_shape(reference_case, input_tensors,
                                                output_tensors, semantic_shape);
      et_kernel_request_v1 req =
          rq(reference_case->operation, rank, semantic_shape);
      et_kernel_call_v1 c = kc(fixture_capability(reference_case->operation),
                               &req, inputs, reference_case->input_count,
                               outputs, reference_case->output_count);
      run(rt, &c);
    }
    for (size_t i = 0; i < reference_case->output_count; i++)
      expect_fixture_output(reference_case, output_tensors[i],
                            output_storage[i].data);
    for (size_t i = 0; i < reference_case->input_count; i++) {
      CHECK(memcmp(input_storage[i].data, input_tensors[i]->words,
                   input_storage[i].bytes) == 0);
      free(input_storage[i].data);
    }
    for (size_t i = 0; i < reference_case->output_count; i++)
      free(output_storage[i].data);
  }
}

int main(void) {
  et_kernel_runtime *rt = runtime_new();
  test_metadata(rt);
  test_embedding(rt);
  test_linear(rt);
  test_norm(rt);
  test_activations(rt);
  test_residual(rt);
  test_official_philox_vectors();
  test_dropout(rt);
  test_every_row(rt);
  test_generic_failures(rt);
  test_frozen_reference_cases(rt);
  et_kernel_runtime_destroy(rt);
  printf("N2 provider PASS (%zu checks)\n", checks);
  return 0;
}
