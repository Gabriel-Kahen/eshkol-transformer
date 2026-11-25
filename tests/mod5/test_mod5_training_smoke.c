#include "mod5_test.h"

#pragma STDC FP_CONTRACT OFF

enum {
  BATCHES = 2,
  TOKENS = 3,
  WIDTH = 8,
  HEADS = 4,
  HEAD_WIDTH = 2,
  ROWS = BATCHES * TOKENS,
  ELEMENTS = ROWS * WIDTH,
  STEPS = 24
};
enum parameter {
  WQ,
  WK,
  WV,
  WO,
  WGATE,
  WUP,
  WDOWN,
  GAMMA1,
  GAMMA2,
  PARAMETERS
};

typedef struct weights {
  float data[PARAMETERS][WIDTH * WIDTH];
} weights;

typedef struct activations {
  float norm1[ELEMENTS], q[ELEMENTS], k[ELEMENTS], v[ELEMENTS];
  float attention[ELEMENTS], projected[ELEMENTS], residual[ELEMENTS];
  float norm2[ELEMENTS], gate[ELEMENTS], up[ELEMENTS], hidden[ELEMENTS];
  float feedforward[ELEMENTS], output[ELEMENTS];
} activations;

static const uint64_t ntd[] = {BATCHES, TOKENS, WIDTH};
static const uint64_t gamma_shape[] = {WIDTH};

static void rms(const et_kernel_runtime *runtime, float *x, float *gamma,
                float *upstream, float *y, float *dgamma) {
  float epsilon = 0.125f;
  const et_kernel_request_v1 request = {sizeof(request),
                                        upstream ? "rmsnorm.backward"
                                                 : "rmsnorm.forward",
                                        "f32",
                                        "cpu",
                                        3u,
                                        ntd,
                                        1u,
                                        {0}};
  const et_kernel_tensor_view_v1 in[] = {
      test_view(x, "f32", 3u, ntd), test_view(gamma, "f32", 1u, gamma_shape),
      test_view(&epsilon, "f32", 0u, NULL),
      test_view(upstream, "f32", 3u, ntd)};
  et_kernel_tensor_view_v1 out[] = {test_view(y, "f32", 3u, ntd),
                                    test_view(dgamma, "f32", 1u, gamma_shape)};
  const et_kernel_call_v1 call =
      test_call("kernel.rmsnorm", &request, upstream ? 4u : 3u, in,
                upstream ? 2u : 1u, out);
  dispatch(runtime, &call);
}

static void glu(const et_kernel_runtime *runtime, float *gate, float *up,
                float *upstream, float *y, float *dup) {
  const et_kernel_request_v1 request = {sizeof(request),
                                        upstream ? "swiglu.backward"
                                                 : "swiglu.forward",
                                        "f32",
                                        "cpu",
                                        3u,
                                        ntd,
                                        1u,
                                        {0}};
  const et_kernel_tensor_view_v1 in[] = {test_view(gate, "f32", 3u, ntd),
                                         test_view(up, "f32", 3u, ntd),
                                         test_view(upstream, "f32", 3u, ntd)};
  et_kernel_tensor_view_v1 out[] = {test_view(y, "f32", 3u, ntd),
                                    test_view(dup, "f32", 3u, ntd)};
  const et_kernel_call_v1 call =
      test_call("kernel.swiglu", &request, upstream ? 3u : 2u, in,
                upstream ? 2u : 1u, out);
  dispatch(runtime, &call);
}

static void pack_heads(const float *tokens, size_t heads, float *packed) {
  for (size_t t = 0; t < TOKENS; ++t) {
    for (size_t h = 0; h < heads; ++h) {
      for (size_t d = 0; d < HEAD_WIDTH; ++d)
        packed[(h * TOKENS + t) * HEAD_WIDTH + d] =
            tokens[(t * heads + h) * HEAD_WIDTH + d];
    }
  }
}

static void unpack_heads(const float *packed, size_t heads, float *tokens) {
  for (size_t t = 0; t < TOKENS; ++t) {
    for (size_t h = 0; h < heads; ++h) {
      for (size_t d = 0; d < HEAD_WIDTH; ++d)
        tokens[(t * heads + h) * HEAD_WIDTH + d] =
            packed[(h * TOKENS + t) * HEAD_WIDTH + d];
    }
  }
}

static void attend(const et_kernel_runtime *runtime, size_t kv_heads,
                   activations *a, activations *grad) {
  const uint64_t shape[] = {1, HEADS, kv_heads, TOKENS, TOKENS, HEAD_WIDTH};
  const uint64_t q_shape[] = {1, HEADS, TOKENS, HEAD_WIDTH};
  const uint64_t kv_shape[] = {1, kv_heads, TOKENS, HEAD_WIDTH};
  const uint64_t pos_shape[] = {1, TOKENS};
  const uint64_t mask_shape[] = {1, TOKENS, TOKENS};
  const uint64_t slopes_shape[] = {HEADS};
  const int backward = grad != NULL;
  const et_kernel_request_v1 request = {sizeof(request),
                                        backward ? "variant-attention.backward"
                                                 : "variant-attention.forward",
                                        "f32",
                                        "cpu",
                                        6u,
                                        shape,
                                        1u,
                                        {0}};
  float slopes[] = {0.125f, 0.25f, 0.375f, 0.5f};
  int64_t window = 3;
  const size_t kv_stride = TOKENS * kv_heads * HEAD_WIDTH;
  for (size_t batch = 0; batch < BATCHES; ++batch) {
    const size_t offset = batch * TOKENS * WIDTH;
    const size_t kv_offset = batch * kv_stride;
    float q[TOKENS * WIDTH], k[TOKENS * WIDTH], v[TOKENS * WIDTH];
    float upstream[TOKENS * WIDTH], y[TOKENS * WIDTH];
    float dk[TOKENS * WIDTH], dv[TOKENS * WIDTH];
    int64_t key_pos[] = {0, 1, 3};
    int64_t query_pos[] = {0, 1, 3 + (int64_t)batch};
    uint8_t mask[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    pack_heads(a->q + offset, HEADS, q);
    pack_heads(a->k + kv_offset, kv_heads, k);
    pack_heads(a->v + kv_offset, kv_heads, v);
    if (backward)
      pack_heads(grad->attention + offset, HEADS, upstream);
    const et_kernel_tensor_view_v1 in[] = {
        test_view(q, "f32", 4u, q_shape),
        test_view(k, "f32", 4u, kv_shape),
        test_view(v, "f32", 4u, kv_shape),
        test_view(query_pos, "i64", 2u, pos_shape),
        test_view(key_pos, "i64", 2u, pos_shape),
        test_view(mask, "bool", 3u, mask_shape),
        test_view(slopes, "f32", 1u, slopes_shape),
        test_view(&window, "i64", 0u, NULL),
        test_view(upstream, "f32", 4u, q_shape)};
    et_kernel_tensor_view_v1 out[] = {test_view(y, "f32", 4u, q_shape),
                                      test_view(dk, "f32", 4u, kv_shape),
                                      test_view(dv, "f32", 4u, kv_shape)};
    const et_kernel_call_v1 call =
        test_call("kernel.variant-attention", &request, backward ? 9u : 8u, in,
                  backward ? 3u : 1u, out);
    dispatch(runtime, &call);
    if (backward) {
      unpack_heads(y, HEADS, grad->q + offset);
      unpack_heads(dk, kv_heads, grad->k + kv_offset);
      unpack_heads(dv, kv_heads, grad->v + kv_offset);
    } else {
      unpack_heads(y, HEADS, a->attention + offset);
    }
  }
}

/* Test-only dense maps bridge the token-major residual stream to projected
 * heads. */
static void project(const float *x, const float *w, size_t width, float *y) {
  for (size_t t = 0; t < ROWS; ++t) {
    for (size_t o = 0; o < width; ++o) {
      float sum = 0.0f;
      for (size_t i = 0; i < WIDTH; ++i)
        sum += x[t * WIDTH + i] * w[o * WIDTH + i];
      y[t * width + o] = sum;
    }
  }
}

static void project_backward(const float *x, const float *w, size_t width,
                             const float *dy, float *dx, float *dw) {
  for (size_t t = 0; t < ROWS; ++t) {
    for (size_t o = 0; o < width; ++o) {
      const float up = dy[t * width + o];
      for (size_t i = 0; i < WIDTH; ++i) {
        dx[t * WIDTH + i] += up * w[o * WIDTH + i];
        dw[o * WIDTH + i] += up * x[t * WIDTH + i];
      }
    }
  }
}

static void block_forward(const et_kernel_runtime *runtime, size_t kv_heads,
                          weights *w, float *x, activations *a) {
  const size_t kv_width = kv_heads * HEAD_WIDTH;
  rms(runtime, x, w->data[GAMMA1], NULL, a->norm1, NULL);
  project(a->norm1, w->data[WQ], WIDTH, a->q);
  project(a->norm1, w->data[WK], kv_width, a->k);
  project(a->norm1, w->data[WV], kv_width, a->v);
  attend(runtime, kv_heads, a, NULL);
  project(a->attention, w->data[WO], WIDTH, a->projected);
  for (size_t i = 0; i < ELEMENTS; ++i)
    a->residual[i] = x[i] + a->projected[i];
  rms(runtime, a->residual, w->data[GAMMA2], NULL, a->norm2, NULL);
  project(a->norm2, w->data[WGATE], WIDTH, a->gate);
  project(a->norm2, w->data[WUP], WIDTH, a->up);
  glu(runtime, a->gate, a->up, NULL, a->hidden, NULL);
  project(a->hidden, w->data[WDOWN], WIDTH, a->feedforward);
  for (size_t i = 0; i < ELEMENTS; ++i)
    a->output[i] = a->residual[i] + a->feedforward[i];
}

static double block_loss(const et_kernel_runtime *runtime, size_t kv_heads,
                         weights *w, float *x, const float *target, weights *dw,
                         float *dx) {
  activations a;
  block_forward(runtime, kv_heads, w, x, &a);
  double sum = 0.0;
  for (size_t i = 0; i < ELEMENTS; ++i) {
    const double delta = (double)a.output[i] - target[i];
    sum += delta * delta;
  }
  if (dw == NULL)
    return 0.5 * sum / ELEMENTS;

  const size_t kv_width = kv_heads * HEAD_WIDTH;
  activations g = {0};
  memset(dw, 0, sizeof(*dw));
  for (size_t i = 0; i < ELEMENTS; ++i)
    g.output[i] = (a.output[i] - target[i]) / (float)ELEMENTS;
  project_backward(a.hidden, w->data[WDOWN], WIDTH, g.output, g.hidden,
                   dw->data[WDOWN]);
  glu(runtime, a.gate, a.up, g.hidden, g.gate, g.up);
  project_backward(a.norm2, w->data[WGATE], WIDTH, g.gate, g.norm2,
                   dw->data[WGATE]);
  project_backward(a.norm2, w->data[WUP], WIDTH, g.up, g.norm2, dw->data[WUP]);
  rms(runtime, a.residual, w->data[GAMMA2], g.norm2, g.residual,
      dw->data[GAMMA2]);
  for (size_t i = 0; i < ELEMENTS; ++i)
    g.residual[i] += g.output[i];
  project_backward(a.attention, w->data[WO], WIDTH, g.residual, g.attention,
                   dw->data[WO]);
  attend(runtime, kv_heads, &a, &g);
  project_backward(a.norm1, w->data[WQ], WIDTH, g.q, g.norm1, dw->data[WQ]);
  project_backward(a.norm1, w->data[WK], kv_width, g.k, g.norm1, dw->data[WK]);
  project_backward(a.norm1, w->data[WV], kv_width, g.v, g.norm1, dw->data[WV]);
  rms(runtime, x, w->data[GAMMA1], g.norm1, dx, dw->data[GAMMA1]);
  for (size_t i = 0; i < ELEMENTS; ++i)
    dx[i] += g.residual[i];
  return 0.5 * sum / ELEMENTS;
}

static size_t param_size(size_t tensor, size_t kv_heads) {
  if (tensor >= GAMMA1)
    return WIDTH;
  if (tensor == WK || tensor == WV)
    return kv_heads * HEAD_WIDTH * WIDTH;
  return WIDTH * WIDTH;
}

static void check_gradient(const et_kernel_runtime *runtime, size_t kv_heads,
                           weights *w, float *x, const float *target,
                           float *value, float gradient) {
  const float original = *value;
  const float step = 0x1p-9f;
  *value = original + step;
  const double plus = block_loss(runtime, kv_heads, w, x, target, NULL, NULL);
  *value = original - step;
  const double minus = block_loss(runtime, kv_heads, w, x, target, NULL, NULL);
  *value = original;
  check_close(gradient, (plus - minus) / (2.0 * step), 0.0000002, 0.003);
}

static void train(const et_kernel_runtime *runtime, size_t kv_heads, weights *w,
                  float *x, const float *target, double loss[STEPS + 1u]) {
  weights dw;
  float dx[ELEMENTS];
  for (size_t step = 0; step < STEPS; ++step) {
    loss[step] = block_loss(runtime, kv_heads, w, x, target, &dw, dx);
    for (size_t i = 0; i < ELEMENTS; ++i)
      CHECK(isfinite(dx[i]));
    CHECK(isfinite(loss[step]));
    for (size_t p = 0; p < PARAMETERS; ++p) {
      for (size_t i = 0; i < param_size(p, kv_heads); ++i) {
        CHECK(isfinite(dw.data[p][i]));
        w->data[p][i] -= 2.0f * dw.data[p][i];
      }
    }
  }
  loss[STEPS] = block_loss(runtime, kv_heads, w, x, target, NULL, NULL);
  for (size_t step = 1; step <= STEPS; ++step)
    CHECK(isfinite(loss[step]) && loss[step] < loss[step - 1u]);
  CHECK(loss[STEPS] < 0.9 * loss[0]);
}

static void train_variant(const et_kernel_runtime *runtime, size_t kv_heads) {
  weights initial = {0};
  weights dw;
  float x[ELEMENTS], target[ELEMENTS], dx[ELEMENTS];
  for (size_t p = 0; p < PARAMETERS; ++p) {
    for (size_t i = 0; i < param_size(p, kv_heads); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      initial.data[p][i] = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
    }
  }
  for (size_t i = 0; i < ELEMENTS; ++i) {
    x[i] = (float)((int)((i * 7u + i / 5u) % 31u) - 15) * 0x1p-5f;
    target[i] = 0.65f * x[i] + (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f;
  }
  (void)block_loss(runtime, kv_heads, &initial, x, target, &dw, dx);
  for (size_t i = 0; i < ELEMENTS; ++i)
    check_gradient(runtime, kv_heads, &initial, x, target, &x[i], dx[i]);
  size_t checked = ELEMENTS;
  for (size_t p = 0; p < PARAMETERS; ++p) {
    double activity = 0.0;
    for (size_t i = 0; i < param_size(p, kv_heads); ++i) {
      check_gradient(runtime, kv_heads, &initial, x, target,
                     &initial.data[p][i], dw.data[p][i]);
      activity += fabs(dw.data[p][i]);
      ++checked;
    }
    if (!(activity > 0.00001))
      fprintf(stderr, "parameter %zu gradient activity %.9g\n", p, activity);
    CHECK(activity > 0.00001);
  }

  double first[STEPS + 1u], repeat[STEPS + 1u];
  weights a = initial, b = initial;
  train(runtime, kv_heads, &a, x, target, first);
  train(runtime, kv_heads, &b, x, target, repeat);
  CHECK(memcmp(first, repeat, sizeof(first)) == 0);
  CHECK(memcmp(&a, &b, sizeof(a)) == 0);
  printf("MOD5 %s training smoke: PASS (%zu finite differences, %u steps, loss "
         "%.9g -> %.9g)\n",
         kv_heads == 1u ? "MQA" : "GQA", checked, (unsigned)STEPS, first[0],
         first[STEPS]);
}

int main(void) {
  et_kernel_runtime *runtime = discover_mod5();
  train_variant(runtime, 1u);
  train_variant(runtime, 2u);
  et_kernel_runtime_destroy(runtime);
  return 0;
}
