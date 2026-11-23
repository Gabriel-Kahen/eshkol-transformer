#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include "eshkol_transformer/mod5_primitives_abi.h"

#include <math.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))

enum parameter { GAMMA1, WQ, WK, WV, WO, GAMMA2, WGATE, WUP, WDOWN, PARAMS };
enum scratch { TMP0, TMP1, TMP2, TMP3, TMP4, SCRATCH_COUNT };
enum branch { ATTENTION_BRANCH, FFN_BRANCH };

typedef struct span {
  const void *data;
  size_t bytes;
} span;

typedef struct layout {
  et_ac4_memory_v1 memory;
  uint64_t ntd[3];
  size_t rows, width, kv_width, elements, kv_elements, tape_floats;
  size_t parameter_offset[PARAMS];
  uint32_t stride;
} layout;

typedef struct arena {
  float *data;
  size_t capacity, used, peak;
} arena;

typedef struct tape {
  float *norm1, *q, *k, *v, *attention, *residual, *norm2, *gate, *up, *hidden;
} tape;

typedef struct stack {
  const et_ac4_config_v1 *config;
  const layout *dims;
  const et_kernel_provider_v1 *provider;
} stack;

static int valid_span(span range, size_t alignment) {
  const uintptr_t address = (uintptr_t)range.data;
  return range.data != NULL && address % alignment == 0 &&
         range.bytes <= UINTPTR_MAX - address;
}

static int overlaps(span a, span b) {
  if (a.bytes == 0 || b.bytes == 0)
    return 0;
  const uintptr_t x = (uintptr_t)a.data;
  const uintptr_t y = (uintptr_t)b.data;
  return x <= y ? y - x < a.bytes : x - y < b.bytes;
}

static int finite_values(const float *values, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (!isfinite(values[i]))
      return 0;
  return 1;
}

static et_ac4_status make_layout(const et_ac4_config_v1 *config, layout *dims) {
  const et_ac4_shape_v1 *shape = &config->shape;
  const uint32_t layers = shape->layers;
  if (layers == 0 || layers > ET_AC4_MAX_LAYERS ||
      config->mode > ET_AC4_FORWARD_BACKWARD ||
      config->policy > ET_AC4_POLICY_EVERY_K || !isfinite(config->epsilon) ||
      config->epsilon <= 0.0f || !isfinite(config->dropout_rate) ||
      config->dropout_rate < 0.0f || config->dropout_rate >= 1.0f)
    return ET_AC4_INVALID_ARGUMENT;

  uint32_t stride = 1;
  if (config->policy == ET_AC4_POLICY_EVERY_K) {
    stride = config->checkpoint_every;
    if (stride == 0 || stride > layers)
      return ET_AC4_INVALID_ARGUMENT;
  } else if (config->checkpoint_every != 0) {
    return ET_AC4_INVALID_ARGUMENT;
  } else if (config->policy == ET_AC4_POLICY_SQRT_SEGMENTS) {
    while (stride * stride < layers)
      ++stride;
  }

  /* Intersect the MOD5 RMSNorm/SwiGLU rows with the attention rows.
   * Dispatch self-attention one batch at a time to the exact N=1 row. */
  if (shape->batch == 0 || shape->tokens == 0 || shape->head_dim == 0 ||
      shape->query_heads == 0 || shape->kv_heads == 0 ||
      shape->query_heads % shape->kv_heads != 0)
    return ET_AC4_SHAPE_MISMATCH;
  const int short_mqa = shape->batch == 1 && shape->tokens == 2 &&
                        shape->query_heads == 2 && shape->kv_heads == 1;
  const int long_mqa_gqa =
      shape->query_heads == 4 && (shape->kv_heads == 1 || shape->kv_heads == 2);
  const int long_mha = shape->query_heads == 2 && shape->kv_heads == 2;
  const int long_batch = shape->batch == 2 && shape->tokens == 3;
  if (shape->head_dim != 2 ||
      !(short_mqa || (long_batch && (long_mqa_gqa || long_mha))))
    return ET_AC4_UNSUPPORTED;

  const int backward = config->mode == ET_AC4_FORWARD_BACKWARD;
  if (!backward)
    stride = layers;
  const size_t width = shape->query_heads * shape->head_dim;
  const size_t kv_width = shape->kv_heads * shape->head_dim;
  const size_t rows = shape->batch * shape->tokens;
  const size_t x = rows * width;
  const size_t kv = rows * kv_width;
  const size_t tape_floats = 8 * x + 2 * kv;
  const size_t matrix = width * width;
  const size_t kv_matrix = kv_width * width;
  const size_t per_layer = 2 * width + 5 * matrix + 2 * kv_matrix;
  const uint32_t checkpoints = (layers - 1) / stride;
  const uint32_t tapes =
      backward && config->policy == ET_AC4_POLICY_NONE ? layers : 1;
  size_t live_streams =
      2 + (stride > 1); /* Final output, temporary, rolling stream. */
  uint32_t replays = 0;

  if (backward) {
    live_streams =
        2 + SCRATCH_COUNT; /* Final output, carry, backward temporaries. */
    if (config->policy == ET_AC4_POLICY_SQRT_SEGMENTS) {
      live_streams += stride - 1;
      replays = 2 * layers - (checkpoints + 1);
    } else if (config->policy == ET_AC4_POLICY_EVERY_K) {
      live_streams +=
          stride > 1; /* Rolling replay must survive the backward call. */
      for (uint32_t begin = 0; begin < layers; begin += stride) {
        const uint32_t left = layers - begin;
        const uint32_t length = left < stride ? left : stride;
        replays += length * (length + 1) / 2;
      }
    } else if (config->policy == ET_AC4_POLICY_EVERY_LAYER) {
      replays = layers;
    }
  }

  const size_t peak =
      ((live_streams + checkpoints) * x + tapes * tape_floats) * sizeof(float);
  const size_t staging = backward ? layers * per_layer * sizeof(float) : 0;
  *dims = (layout){.memory = {x, per_layer, layers * per_layer, peak, staging,
                              peak + staging, checkpoints, tapes, replays},
                   .ntd = {shape->batch, shape->tokens, width},
                   .rows = rows,
                   .width = width,
                   .kv_width = kv_width,
                   .elements = x,
                   .kv_elements = kv,
                   .tape_floats = tape_floats,
                   .stride = stride};
  const size_t counts[PARAMS] = {width, matrix, kv_matrix, kv_matrix, matrix,
                                 width, matrix, matrix,    matrix};
  for (size_t i = 1; i < PARAMS; ++i)
    dims->parameter_offset[i] = dims->parameter_offset[i - 1] + counts[i - 1];
  return ET_AC4_OK;
}

et_ac4_status et_ac4_plan_v1(const et_ac4_config_v1 *config,
                             et_ac4_memory_v1 *memory) {
  const span config_span = {config, sizeof(*config)};
  const span memory_span = {memory, sizeof(*memory)};
  if (!valid_span(config_span, _Alignof(et_ac4_config_v1)) ||
      !valid_span(memory_span, _Alignof(et_ac4_memory_v1)))
    return ET_AC4_INVALID_ARGUMENT;
  if (overlaps(config_span, memory_span))
    return ET_AC4_ALIASING;
  layout dims;
  const et_ac4_status status = make_layout(config, &dims);
  if (status == ET_AC4_OK)
    *memory = dims.memory;
  return status;
}

static et_ac4_status validate_io(const et_ac4_config_v1 *config,
                                 const et_ac4_io_v1 *io,
                                 et_ac4_result_v1 *result,
                                 const et_ac4_memory_v1 *memory) {
  const size_t x = memory->activation_floats;
  const size_t p = memory->parameter_floats;
  const int backward = config->mode == ET_AC4_FORWARD_BACKWARD;
  if (io->input.count != x || io->target.count != x || io->output.count != x ||
      io->parameters.count != p || io->d_input.count != (backward ? x : 0) ||
      io->d_parameters.count != (backward ? p : 0))
    return ET_AC4_SHAPE_MISMATCH;
  if (!backward && (io->d_input.data != NULL || io->d_parameters.data != NULL))
    return ET_AC4_INVALID_ARGUMENT;

  const span payloads[] = {
      {io->input.data, x * sizeof(float)},
      {io->target.data, x * sizeof(float)},
      {io->output.data, x * sizeof(float)},
      {io->parameters.data, p * sizeof(float)},
      {io->d_input.data, io->d_input.count * sizeof(float)},
      {io->d_parameters.data, io->d_parameters.count * sizeof(float)},
      {io->workspace, io->workspace_bytes}};
  for (size_t i = 0; i < COUNT(payloads); ++i) {
    const span payload = payloads[i];
    if (payload.data == NULL && payload.bytes == 0 && (i == 4 || i == 5))
      continue;
    if (!valid_span(payload, _Alignof(float)))
      return ET_AC4_INVALID_ARGUMENT;
  }
  const span all[] = {{config, sizeof(*config)},
                      {io, sizeof(*io)},
                      {result, sizeof(*result)},
                      payloads[0],
                      payloads[1],
                      payloads[2],
                      payloads[3],
                      payloads[4],
                      payloads[5],
                      payloads[6]};
  for (size_t i = 0; i < COUNT(all); ++i)
    for (size_t j = 0; j < i; ++j)
      if (overlaps(all[i], all[j]))
        return ET_AC4_ALIASING;

  if (config->activation_limit_bytes != 0 &&
      config->activation_limit_bytes < memory->activation_peak_bytes)
    return ET_AC4_ACTIVATION_LIMIT;
  if (io->workspace_bytes < memory->workspace_bytes)
    return ET_AC4_WORKSPACE_TOO_SMALL;
  if (!finite_values(io->input.data, x) || !finite_values(io->target.data, x) ||
      !finite_values(io->parameters.data, p))
    return ET_AC4_NUMERIC_ERROR;
  return ET_AC4_OK;
}

static float *take(arena *storage, size_t count) {
  if (count > storage->capacity - storage->used)
    return NULL;
  float *data = storage->data + storage->used;
  storage->used += count;
  if (storage->used > storage->peak)
    storage->peak = storage->used;
  return data;
}

static tape tape_at(const layout *dims, float *data) {
  const size_t x = dims->elements;
  const size_t kv = dims->kv_elements;
  tape a = {.norm1 = data};
  a.q = a.norm1 + x;
  a.k = a.q + x;
  a.v = a.k + kv;
  a.attention = a.v + kv;
  a.residual = a.attention + x;
  a.norm2 = a.residual + x;
  a.gate = a.norm2 + x;
  a.up = a.gate + x;
  a.hidden = a.up + x;
  return a;
}

static et_kernel_tensor_view_v1 view(const void *data, const char *dtype,
                                     size_t item_size, size_t rank,
                                     const uint64_t *shape) {
  size_t bytes = item_size;
  for (size_t i = 0; i < rank; ++i)
    bytes *= shape[i];
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                    (void *)data,
                                    bytes,
                                    dtype,
                                    "cpu",
                                    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    0,
                                    rank,
                                    shape};
}

static et_kernel_tensor_view_v1 f32(const void *data, size_t rank,
                                    const uint64_t *shape) {
  return view(data, "f32", sizeof(float), rank, shape);
}

static et_ac4_status dispatch(const stack *model, const char *capability,
                              const char *operation, size_t rank,
                              const uint64_t *shape,
                              const et_kernel_tensor_view_v1 *in, size_t n_in,
                              et_kernel_tensor_view_v1 *out, size_t n_out) {
  const et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", rank, shape, 1, {0}};
  const et_kernel_call_v1 call = {
      sizeof(call), capability,           &request, n_in,
      sizeof(*in),  n_in * sizeof(*in),   in,       n_out,
      sizeof(*out), n_out * sizeof(*out), out};
  et_kernel_error error;
  const et_kernel_provider_v1 *provider = model->provider;
  /* Statically linked, pinned K1 provider. Validation precedes every invoke;
   * both phases borrow the same local descriptors and payloads. */
  if (provider->validate_call(&call, &error) == 0) {
    provider->invoke_call(&call);
    return ET_AC4_OK;
  }
  if (error.category == ET_KERNEL_ERROR_UNSUPPORTED)
    return ET_AC4_UNSUPPORTED;
  if (error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
      error.code == ET_KERNEL_CODE_PROVIDER_REJECTED)
    return ET_AC4_NUMERIC_ERROR;
  return ET_AC4_PROVIDER_ERROR;
}

static et_ac4_status rms(const stack *model, const float *x, const float *gamma,
                         const float *upstream, float *out, float *dgamma) {
  const uint64_t *ntd = model->dims->ntd;
  const et_kernel_tensor_view_v1 in[] = {f32(x, 3, ntd), f32(gamma, 1, &ntd[2]),
                                         f32(&model->config->epsilon, 0, NULL),
                                         f32(upstream, 3, ntd)};
  et_kernel_tensor_view_v1 outputs[] = {f32(out, 3, ntd),
                                        f32(dgamma, 1, &ntd[2])};
  const int backward = upstream != NULL;
  return dispatch(model, "kernel.rmsnorm",
                  backward ? "rmsnorm.backward" : "rmsnorm.forward", 3, ntd, in,
                  backward ? 4 : 3, outputs, backward ? 2 : 1);
}

static et_ac4_status swiglu(const stack *model, const tape *a,
                            const float *upstream, float *out, float *dup) {
  const uint64_t *ntd = model->dims->ntd;
  const et_kernel_tensor_view_v1 in[] = {
      f32(a->gate, 3, ntd), f32(a->up, 3, ntd), f32(upstream, 3, ntd)};
  et_kernel_tensor_view_v1 outputs[] = {f32(out, 3, ntd), f32(dup, 3, ntd)};
  const int backward = upstream != NULL;
  return dispatch(model, "kernel.swiglu",
                  backward ? "swiglu.backward" : "swiglu.forward", 3, ntd, in,
                  backward ? 3 : 2, outputs, backward ? 2 : 1);
}

static et_ac4_status attend(const stack *model, const tape *a,
                            const float *upstream, float *out, float *dk,
                            float *dv) {
  const et_ac4_shape_v1 *s = &model->config->shape;
  const uint64_t shape[] = {1,         s->query_heads, s->kv_heads,
                            s->tokens, s->tokens,      s->head_dim};
  const uint64_t q_shape[] = {1, s->query_heads, s->tokens, s->head_dim};
  const uint64_t kv_shape[] = {1, s->kv_heads, s->tokens, s->head_dim};
  const uint64_t pos_shape[] = {1, s->tokens};
  const uint64_t mask_shape[] = {1, s->tokens, s->tokens};
  const int64_t q_pos[] = {0, 1, 2}, k_pos[] = {0, 1, 2};
  const uint8_t keep[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  const float slopes[4] = {0};
  const int64_t window = s->tokens;
  const size_t q_size = s->tokens * model->dims->width;
  const size_t kv_size = s->tokens * model->dims->kv_width;
  const int backward = upstream != NULL;
  for (size_t n = 0; n < s->batch; ++n) {
    const size_t q = n * q_size;
    const size_t kv = n * kv_size;
    const et_kernel_tensor_view_v1 in[] = {
        f32(a->q + q, 4, q_shape),
        f32(a->k + kv, 4, kv_shape),
        f32(a->v + kv, 4, kv_shape),
        view(q_pos, "i64", sizeof(*q_pos), 2, pos_shape),
        view(k_pos, "i64", sizeof(*k_pos), 2, pos_shape),
        view(keep, "bool", sizeof(*keep), 3, mask_shape),
        f32(slopes, 1, &q_shape[1]),
        view(&window, "i64", sizeof(window), 0, NULL),
        f32(backward ? upstream + q : NULL, 4, q_shape)};
    et_kernel_tensor_view_v1 outputs[] = {
        f32(out + q, 4, q_shape), f32(backward ? dk + kv : NULL, 4, kv_shape),
        f32(backward ? dv + kv : NULL, 4, kv_shape)};
    const et_ac4_status status = dispatch(
        model, "kernel.variant-attention",
        backward ? "variant-attention.backward" : "variant-attention.forward",
        6, shape, in, backward ? 9 : 8, outputs, backward ? 3 : 1);
    if (status != ET_AC4_OK)
      return status;
  }
  return ET_AC4_OK;
}

/* Converts between dense [N,T,H,Dh] and [N,H,T,Dh]. */
static void reorder_heads(const et_ac4_shape_v1 *s, size_t heads,
                          const float *in, float *out, int pack) {
  const size_t tokens = s->tokens;
  const size_t head_dim = s->head_dim;
  for (size_t n = 0; n < s->batch; ++n) {
    for (size_t t = 0; t < tokens; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        const size_t th = ((n * tokens + t) * heads + h) * head_dim;
        const size_t ht = ((n * heads + h) * tokens + t) * head_dim;
        memcpy(out + (pack ? ht : th), in + (pack ? th : ht),
               head_dim * sizeof(float));
      }
    }
  }
}

static int project(const layout *dims, const float *x, const float *w,
                   size_t out_width, float *out) {
  const size_t width = dims->width;
  for (size_t t = 0; t < dims->rows; ++t) {
    for (size_t o = 0; o < out_width; ++o) {
      float sum = 0.0f;
      for (size_t i = 0; i < width; ++i)
        sum += x[t * width + i] * w[o * width + i];
      out[t * out_width + o] = sum;
    }
  }
  return finite_values(out, dims->rows * out_width);
}

static void project_backward(const layout *dims, const float *x, const float *w,
                             size_t out_width, const float *dy, float *dx,
                             float *dw) {
  const size_t width = dims->width;
  for (size_t t = 0; t < dims->rows; ++t) {
    for (size_t o = 0; o < out_width; ++o) {
      const float delta = dy[t * out_width + o];
      for (size_t i = 0; i < width; ++i) {
        dx[t * width + i] += delta * w[o * width + i];
        dw[o * width + i] += delta * x[t * width + i];
      }
    }
  }
}

static uint64_t mix64(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static uint64_t layer_key(const et_ac4_config_v1 *config, uint32_t layer) {
  const uint64_t step = mix64(config->step + UINT64_C(0x9e3779b97f4a7c15));
  return mix64(config->seed ^ step ^
               mix64((uint64_t)layer + UINT64_C(0xd1b54a32d192ed03)));
}

static void dropout(const stack *model, uint64_t key, enum branch branch,
                    const float *input, const float *residual, float *output) {
  const float rate = model->config->dropout_rate;
  const float scale = 1.0f / (1.0f - rate);
  for (size_t i = 0; i < model->dims->elements; ++i) {
    float value = input[i];
    if (rate != 0.0f) {
      const uint64_t counter = 2 * (uint64_t)i + (uint64_t)branch + 1;
      const uint64_t bits = mix64(key + counter * UINT64_C(0x9e3779b97f4a7c15));
      const double uniform = (double)(bits >> 11) * 0x1p-53;
      value = uniform < (double)rate ? 0.0f : value * scale;
    }
    output[i] = residual != NULL ? residual[i] + value : value;
  }
}

static et_ac4_status layer_forward(const stack *model, uint32_t layer,
                                   const float *x, const float *params, tape a,
                                   float *tmp, float *out) {
  const layout *dims = model->dims;
  const et_ac4_shape_v1 *shape = &model->config->shape;
  const size_t *offset = dims->parameter_offset;
  const size_t width = dims->width;
  const size_t kv_width = dims->kv_width;
  et_ac4_status status =
      rms(model, x, params + offset[GAMMA1], NULL, a.norm1, NULL);
  if (status != ET_AC4_OK)
    return status;
  if (!project(dims, a.norm1, params + offset[WQ], width, tmp))
    return ET_AC4_NUMERIC_ERROR;
  reorder_heads(shape, shape->query_heads, tmp, a.q, 1);
  if (!project(dims, a.norm1, params + offset[WK], kv_width, tmp))
    return ET_AC4_NUMERIC_ERROR;
  reorder_heads(shape, shape->kv_heads, tmp, a.k, 1);
  if (!project(dims, a.norm1, params + offset[WV], kv_width, tmp))
    return ET_AC4_NUMERIC_ERROR;
  reorder_heads(shape, shape->kv_heads, tmp, a.v, 1);
  status = attend(model, &a, NULL, tmp, NULL, NULL);
  if (status != ET_AC4_OK)
    return status;
  reorder_heads(shape, shape->query_heads, tmp, a.attention, 0);
  if (!project(dims, a.attention, params + offset[WO], width, tmp))
    return ET_AC4_NUMERIC_ERROR;
  const uint64_t key = layer_key(model->config, layer);
  dropout(model, key, ATTENTION_BRANCH, tmp, x, a.residual);

  status = rms(model, a.residual, params + offset[GAMMA2], NULL, a.norm2, NULL);
  if (status != ET_AC4_OK)
    return status;
  if (!project(dims, a.norm2, params + offset[WGATE], width, a.gate) ||
      !project(dims, a.norm2, params + offset[WUP], width, a.up))
    return ET_AC4_NUMERIC_ERROR;
  status = swiglu(model, &a, NULL, a.hidden, NULL);
  if (status != ET_AC4_OK)
    return status;
  if (!project(dims, a.hidden, params + offset[WDOWN], width, tmp))
    return ET_AC4_NUMERIC_ERROR;
  dropout(model, key, FFN_BRANCH, tmp, a.residual, out);
  return finite_values(out, dims->elements) ? ET_AC4_OK : ET_AC4_NUMERIC_ERROR;
}

static et_ac4_status layer_backward(const stack *model, uint32_t layer,
                                    const float *x, const float *params, tape a,
                                    float *carry, float *grad,
                                    float *scratch[SCRATCH_COUNT]) {
  const layout *dims = model->dims;
  const et_ac4_shape_v1 *shape = &model->config->shape;
  const size_t *offset = dims->parameter_offset;
  const size_t width = dims->width;
  const size_t bytes = dims->elements * sizeof(float);
  float *s0 = scratch[TMP0], *s1 = scratch[TMP1], *s2 = scratch[TMP2];
  float *s3 = scratch[TMP3], *s4 = scratch[TMP4];
  const uint64_t key = layer_key(model->config, layer);

  dropout(model, key, FFN_BRANCH, carry, NULL, s0);
  memset(s1, 0, bytes);
  project_backward(dims, a.hidden, params + offset[WDOWN], width, s0, s1,
                   grad + offset[WDOWN]);
  et_ac4_status status = swiglu(model, &a, s1, s2, s3);
  if (status != ET_AC4_OK)
    return status;
  memset(s0, 0, bytes);
  project_backward(dims, a.norm2, params + offset[WGATE], width, s2, s0,
                   grad + offset[WGATE]);
  project_backward(dims, a.norm2, params + offset[WUP], width, s3, s0,
                   grad + offset[WUP]);
  status = rms(model, a.residual, params + offset[GAMMA2], s0, s1,
               grad + offset[GAMMA2]);
  if (status != ET_AC4_OK)
    return status;
  for (size_t i = 0; i < dims->elements; ++i)
    carry[i] += s1[i];

  dropout(model, key, ATTENTION_BRANCH, carry, NULL, s0);
  memset(s1, 0, bytes);
  project_backward(dims, a.attention, params + offset[WO], width, s0, s1,
                   grad + offset[WO]);
  reorder_heads(shape, shape->query_heads, s1, s0, 1);
  status = attend(model, &a, s0, s1, s2, s3);
  if (status != ET_AC4_OK)
    return status;
  memset(s4, 0, bytes);
  reorder_heads(shape, shape->query_heads, s1, s0, 0);
  project_backward(dims, a.norm1, params + offset[WQ], width, s0, s4,
                   grad + offset[WQ]);
  reorder_heads(shape, shape->kv_heads, s2, s0, 0);
  project_backward(dims, a.norm1, params + offset[WK], dims->kv_width, s0, s4,
                   grad + offset[WK]);
  reorder_heads(shape, shape->kv_heads, s3, s0, 0);
  project_backward(dims, a.norm1, params + offset[WV], dims->kv_width, s0, s4,
                   grad + offset[WV]);
  status =
      rms(model, x, params + offset[GAMMA1], s4, s0, grad + offset[GAMMA1]);
  if (status != ET_AC4_OK)
    return status;
  for (size_t i = 0; i < dims->elements; ++i)
    carry[i] += s0[i];
  return finite_values(carry, dims->elements) &&
                 finite_values(grad, dims->memory.parameter_floats_per_layer)
             ? ET_AC4_OK
             : ET_AC4_NUMERIC_ERROR;
}

static const float *layer_parameters(const stack *model, const et_ac4_io_v1 *io,
                                     uint32_t layer) {
  return io->parameters.data +
         layer * model->dims->memory.parameter_floats_per_layer;
}

static et_ac4_status forward(const stack *model, const et_ac4_io_v1 *io,
                             arena *storage, float *checkpoints, float *tapes,
                             float *output) {
  const layout *dims = model->dims;
  const uint32_t layers = model->config->shape.layers;
  const uint32_t stride = dims->stride;
  const size_t x = dims->elements;
  const size_t mark = storage->used;
  float *tmp = take(storage, x);
  float *rolling = take(storage, stride > 1 ? x : 0);
  if (tmp == NULL || rolling == NULL)
    return ET_AC4_WORKSPACE_TOO_SMALL;

  const float *input = io->input.data;
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const uint32_t next = layer + 1;
    float *out = rolling;
    if (next == layers)
      out = output;
    else if (next % stride == 0)
      out = checkpoints + (next / stride - 1) * x;
    const size_t slot = dims->memory.retained_tapes == 1 ? 0 : layer;
    const tape a = tape_at(dims, tapes + slot * dims->tape_floats);
    const et_ac4_status status = layer_forward(
        model, layer, input, layer_parameters(model, io, layer), a, tmp, out);
    if (status != ET_AC4_OK)
      return status;
    input = out;
  }
  storage->used = mark;
  return ET_AC4_OK;
}

static et_ac4_status replay(const stack *model, const et_ac4_io_v1 *io,
                            uint32_t layer, const float *input, tape a,
                            float *tmp, float *out, uint32_t *replayed) {
  ++*replayed;
  return layer_forward(model, layer, input, layer_parameters(model, io, layer),
                       a, tmp, out);
}

static et_ac4_status
backward_segment(const stack *model, const et_ac4_io_v1 *io, uint32_t begin,
                 uint32_t end, const float *initial_input, float *tapes,
                 float *segment_cache, float *rolling, float *carry,
                 float *staged, float *scratch[SCRATCH_COUNT],
                 uint32_t *replayed) {
  const layout *dims = model->dims;
  const et_ac4_policy policy = model->config->policy;
  const size_t x = dims->elements;
  float *tmp = scratch[TMP0];
  const tape working_tape = tape_at(dims, tapes);

  /* SQRT keeps just the inter-layer stream inside this segment. */
  if (policy == ET_AC4_POLICY_SQRT_SEGMENTS) {
    const float *input = initial_input;
    for (uint32_t layer = begin; layer + 1 < end; ++layer) {
      float *next = segment_cache + (layer - begin) * x;
      const et_ac4_status status =
          replay(model, io, layer, input, working_tape, tmp, next, replayed);
      if (status != ET_AC4_OK)
        return status;
      input = next;
    }
  }

  for (uint32_t next = end; next > begin; --next) {
    const uint32_t layer = next - 1;
    const float *input = initial_input;
    tape a = working_tape;
    if (policy == ET_AC4_POLICY_NONE) {
      a = tape_at(dims, tapes + layer * dims->tape_floats);
    } else {
      if (policy == ET_AC4_POLICY_SQRT_SEGMENTS && layer > begin) {
        input = segment_cache + (layer - begin - 1) * x;
      } else if (policy == ET_AC4_POLICY_EVERY_K) {
        /* Stream the prefix anew, retaining only the current layer input. */
        for (uint32_t i = begin; i < layer; ++i) {
          const et_ac4_status status =
              replay(model, io, i, input, a, tmp, rolling, replayed);
          if (status != ET_AC4_OK)
            return status;
          input = rolling;
        }
      }
      const et_ac4_status status =
          replay(model, io, layer, input, a, tmp, tmp, replayed);
      if (status != ET_AC4_OK)
        return status;
    }
    float *grad = staged + layer * dims->memory.parameter_floats_per_layer;
    const et_ac4_status status =
        layer_backward(model, layer, input, layer_parameters(model, io, layer),
                       a, carry, grad, scratch);
    if (status != ET_AC4_OK)
      return status;
  }
  return ET_AC4_OK;
}

static et_ac4_status backward(const stack *model, const et_ac4_io_v1 *io,
                              arena *storage, const float *checkpoints,
                              float *tapes, const float *output, float *carry,
                              float *staged, uint32_t *replayed) {
  const layout *dims = model->dims;
  const et_ac4_policy policy = model->config->policy;
  const uint32_t stride = dims->stride;
  const size_t x = dims->elements;
  float *scratch[SCRATCH_COUNT];
  for (size_t i = 0; i < COUNT(scratch); ++i) {
    scratch[i] = take(storage, x);
    if (scratch[i] == NULL)
      return ET_AC4_WORKSPACE_TOO_SMALL;
  }
  float *rolling =
      take(storage, policy == ET_AC4_POLICY_EVERY_K && stride > 1 ? x : 0);
  if (rolling == NULL)
    return ET_AC4_WORKSPACE_TOO_SMALL;
  const size_t mark = storage->used;
  for (size_t i = 0; i < x; ++i)
    carry[i] = (float)(((double)output[i] - io->target.data[i]) / (double)x);

  uint32_t end = model->config->shape.layers;
  while (end != 0) {
    const uint32_t begin = ((end - 1) / stride) * stride;
    const float *input =
        begin == 0 ? io->input.data : checkpoints + (begin / stride - 1) * x;
    const size_t cache =
        policy == ET_AC4_POLICY_SQRT_SEGMENTS ? (end - begin - 1) * x : 0;
    float *segment_cache = take(storage, cache);
    if (segment_cache == NULL)
      return ET_AC4_WORKSPACE_TOO_SMALL;
    const et_ac4_status status =
        backward_segment(model, io, begin, end, input, tapes, segment_cache,
                         rolling, carry, staged, scratch, replayed);
    if (status != ET_AC4_OK)
      return status;
    storage->used = mark;
    end = begin;
  }
  return ET_AC4_OK;
}

et_ac4_status et_ac4_run_v1(const et_ac4_config_v1 *config,
                            const et_ac4_io_v1 *io, et_ac4_result_v1 *result) {
  const span config_span = {config, sizeof(*config)};
  const span io_span = {io, sizeof(*io)};
  const span result_span = {result, sizeof(*result)};
  if (!valid_span(config_span, _Alignof(et_ac4_config_v1)) ||
      !valid_span(io_span, _Alignof(et_ac4_io_v1)) ||
      !valid_span(result_span, _Alignof(et_ac4_result_v1)))
    return ET_AC4_INVALID_ARGUMENT;
  if (overlaps(config_span, io_span) || overlaps(config_span, result_span) ||
      overlaps(io_span, result_span))
    return ET_AC4_ALIASING;
  layout dims;
  et_ac4_status status = make_layout(config, &dims);
  if (status != ET_AC4_OK)
    return status;
  status = validate_io(config, io, result, &dims.memory);
  if (status != ET_AC4_OK)
    return status;

  /* The plan's upper bound caps the activation arena even when the caller
   * supplies a larger buffer. The staging region begins after that bound. */
  arena storage = {io->workspace,
                   dims.memory.activation_peak_bytes / sizeof(float), 0, 0};
  float *checkpoints =
      take(&storage, dims.memory.retained_checkpoints * dims.elements);
  float *tapes = take(&storage, dims.memory.retained_tapes * dims.tape_floats);
  float *output = take(&storage, dims.elements);
  if (checkpoints == NULL || tapes == NULL || output == NULL)
    return ET_AC4_WORKSPACE_TOO_SMALL;
  const stack model = {config, &dims, et_mod5_kernel_provider_v1()};
  status = forward(&model, io, &storage, checkpoints, tapes, output);
  if (status != ET_AC4_OK)
    return status;

  double sum = 0.0;
  for (size_t i = 0; i < dims.elements; ++i) {
    const double delta = (double)output[i] - io->target.data[i];
    sum += delta * delta;
  }
  et_ac4_result_v1 completed = {.loss = 0.5 * sum / (double)dims.elements,
                                .memory = dims.memory};
  completed.memory.rematerialized_layers = 0;
  if (config->mode == ET_AC4_FORWARD_BACKWARD) {
    float *carry = take(&storage, dims.elements);
    if (carry == NULL)
      return ET_AC4_WORKSPACE_TOO_SMALL;
    float *staged = storage.data + storage.capacity;
    memset(staged, 0, dims.memory.gradient_staging_bytes);
    status = backward(&model, io, &storage, checkpoints, tapes, output, carry,
                      staged, &completed.memory.rematerialized_layers);
    if (status != ET_AC4_OK)
      return status;
    memcpy(io->d_parameters.data, staged, dims.memory.gradient_staging_bytes);
    memcpy(io->d_input.data, carry, dims.elements * sizeof(float));
  }

  memcpy(io->output.data, output, dims.elements * sizeof(float));
  completed.memory.activation_peak_bytes = storage.peak * sizeof(float);
  *result = completed;
  return ET_AC4_OK;
}
