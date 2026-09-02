#include "eshkol_transformer/a2_attention_abi.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#define ET_ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

typedef struct attention_shape {
  size_t n;
  size_t hq;
  size_t hkv;
  size_t tq;
  size_t tk;
  size_t dh;
} attention_shape;

typedef struct rope_shape {
  size_t n;
  size_t h;
  size_t t;
  size_t dh;
} rope_shape;

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int pointer_span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int aligned_pointer(const void *pointer, size_t alignment) {
  return pointer != NULL && (uintptr_t)pointer % alignment == 0u;
}

static int ranges_overlap(const void *left, size_t left_bytes,
                          const void *right, size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u ||
      left_start > UINTPTR_MAX - left_bytes ||
      right_start > UINTPTR_MAX - right_bytes) {
    return left_bytes != 0u && right_bytes != 0u;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static const et_kernel_tensor_view_v1 *input_at(
    const et_kernel_call_v1 *call, size_t index) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)call->inputs +
                                             index * call->input_stride);
}

static et_kernel_tensor_view_v1 *output_at(const et_kernel_call_v1 *call,
                                           size_t index) {
  return (et_kernel_tensor_view_v1 *)((unsigned char *)call->outputs +
                                      index * call->output_stride);
}

static int32_t set_error(et_kernel_error *error,
                         et_kernel_error_category category,
                         et_kernel_error_code code, const char *operation,
                         const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int checked_product(size_t *result, size_t left, size_t right) {
  if (right != 0u && left > SIZE_MAX / right) {
    return 0;
  }
  *result = left * right;
  return 1;
}

static int shape_to_size(size_t rank, const uint64_t *shape, size_t *result) {
  for (size_t index = 0; index < rank; index++) {
    if (shape[index] == 0u || shape[index] > SIZE_MAX) {
      return 0;
    }
    result[index] = (size_t)shape[index];
  }
  return 1;
}

static int32_t table_views(const et_kernel_call_v1 *call, int output,
                           size_t expected, const char *operation,
                           const et_kernel_tensor_view_v1 **views,
                           et_kernel_error *error) {
  const size_t count = output ? call->output_count : call->input_count;
  const size_t stride = output ? call->output_stride : call->input_stride;
  const size_t bytes = output ? call->output_bytes : call->input_bytes;
  const void *base = output ? call->outputs : call->inputs;

  if (count != expected || base == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     output ? "output tensor count does not match the operation"
                            : "input tensor count does not match the operation");
  }
  if (stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                     "tensor table stride truncates the K1 v1 prefix");
  }
  if (stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      !aligned_pointer(base, _Alignof(et_kernel_tensor_view_v1)) ||
      expected > SIZE_MAX / stride || bytes != expected * stride ||
      !pointer_span_fits(base, bytes)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor table span or alignment is invalid");
  }
  for (size_t index = 0; index < expected; index++) {
    const et_kernel_tensor_view_v1 *view =
        (const et_kernel_tensor_view_v1 *)((const unsigned char *)base +
                                            index * stride);
    if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
        view->struct_size > stride) {
      return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                       ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                       "tensor descriptor does not fit its table stride");
    }
    views[index] = view;
  }
  return 0;
}

static int32_t validate_view(const et_kernel_tensor_view_v1 *view,
                             const char *dtype, size_t element_size,
                             size_t alignment, size_t rank,
                             const size_t *shape, const char *operation,
                             et_kernel_error *error) {
  size_t elements = 1u;
  size_t expected_bytes;
  if (!exact_text(view->dtype, dtype)) {
    return set_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, operation,
                     "tensor dtype does not match the operation schema");
  }
  if (!exact_text(view->device, "cpu")) {
    return set_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, operation,
                     "A2 numerical kernels accept only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "A2 numerical kernels require dense zero-offset views");
  }
  if (view->rank != rank || (rank != 0u && view->shape == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, operation,
                     "tensor rank does not match the operation schema");
  }
  if (rank != 0u &&
      (!aligned_pointer(view->shape, _Alignof(uint64_t)) ||
       !pointer_span_fits(view->shape, rank * sizeof(*view->shape)))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor shape storage is invalid");
  }
  for (size_t index = 0; index < rank; index++) {
    if (view->shape[index] != (uint64_t)shape[index] ||
        !checked_product(&elements, elements, shape[index])) {
      return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                       ET_KERNEL_CODE_INVALID_SHAPE, operation,
                       "tensor extent does not match the operation schema");
    }
  }
  if (!checked_product(&expected_bytes, elements, element_size) ||
      view->byte_length != expected_bytes ||
      !aligned_pointer(view->data, alignment) ||
      !pointer_span_fits(view->data, expected_bytes)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor data alignment or byte span is invalid");
  }
  return 0;
}

static int validate_finite(const float *values, size_t count) {
  for (size_t index = 0; index < count; index++) {
    if (!isfinite(values[index])) {
      return 0;
    }
  }
  return 1;
}

static int validate_positions(const int64_t *positions, size_t n, size_t t) {
  for (size_t batch = 0; batch < n; batch++) {
    int64_t previous = -1;
    for (size_t token = 0; token < t; token++) {
      const int64_t position = positions[batch * t + token];
      if (position < 0 || position > ET_A2_MAX_EXACT_POSITION ||
          position <= previous) {
        return 0;
      }
      previous = position;
    }
  }
  return 1;
}

static float attention_scale(size_t head_dimension) {
  const float root = sqrtf((float)head_dimension);
  return 1.0f / root;
}

static size_t q_index(const attention_shape *s, size_t n, size_t h,
                      size_t t, size_t d) {
  return (((n * s->hq + h) * s->tq + t) * s->dh) + d;
}

static size_t kv_index(const attention_shape *s, size_t n, size_t h,
                       size_t t, size_t d) {
  return (((n * s->hkv + h) * s->tk + t) * s->dh) + d;
}

static size_t mask_index(const attention_shape *s, size_t n, size_t tq,
                         size_t tk) {
  return (n * s->tq + tq) * s->tk + tk;
}

static int admitted(const attention_shape *s, const int64_t *query_positions,
                    const int64_t *key_positions, const uint8_t *mask, size_t n,
                    size_t tq, size_t tk) {
  return mask[mask_index(s, n, tq, tk)] != 0u &&
         key_positions[n * s->tk + tk] <=
             query_positions[n * s->tq + tq];
}

static int attention_score(const attention_shape *s, const float *q,
                           const float *k, float scale, size_t n, size_t hq,
                           size_t tq, size_t tk, float *result) {
  const size_t group = s->hq / s->hkv;
  const size_t hkv = hq / group;
  float sum = 0.0f;
  for (size_t d = 0; d < s->dh; d++) {
    const float product = q[q_index(s, n, hq, tq, d)] *
                          k[kv_index(s, n, hkv, tk, d)];
    if (!isfinite(product)) {
      return 0;
    }
    sum = sum + product;
    if (!isfinite(sum)) {
      return 0;
    }
  }
  sum = sum * scale;
  if (!isfinite(sum)) {
    return 0;
  }
  *result = sum;
  return 1;
}

static int attention_row(const attention_shape *s, const float *q,
                         const float *k, const int64_t *query_positions,
                         const int64_t *key_positions, const uint8_t *mask,
                         float scale, size_t n, size_t hq, size_t tq,
                         float *maximum, float *denominator) {
  int any = 0;
  float row_max = 0.0f;
  float sum = 0.0f;
  for (size_t tk = 0; tk < s->tk; tk++) {
    float score;
    if (!admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
      continue;
    }
    if (!attention_score(s, q, k, scale, n, hq, tq, tk, &score)) {
      return -1;
    }
    if (!any || score > row_max) {
      row_max = score;
    }
    any = 1;
  }
  if (!any) {
    *maximum = 0.0f;
    *denominator = 1.0f;
    return 0;
  }
  for (size_t tk = 0; tk < s->tk; tk++) {
    float score;
    float weight;
    if (!admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
      continue;
    }
    if (!attention_score(s, q, k, scale, n, hq, tq, tk, &score)) {
      return -1;
    }
    weight = expf(score - row_max);
    sum = sum + weight;
    if (!isfinite(weight) || !isfinite(sum)) {
      return -1;
    }
  }
  *maximum = row_max;
  *denominator = sum;
  return 1;
}

static int attention_probability(const attention_shape *s, const float *q,
                                 const float *k, float scale, size_t n,
                                 size_t hq, size_t tq, size_t tk, float maximum,
                                 float denominator, float *result) {
  float score;
  if (!attention_score(s, q, k, scale, n, hq, tq, tk, &score)) {
    return 0;
  }
  *result = expf(score - maximum) / denominator;
  return isfinite(*result);
}

static int attention_mean_dp(
    const attention_shape *s, const float *q, const float *k, const float *v,
    const float *upstream, const int64_t *query_positions,
    const int64_t *key_positions, const uint8_t *mask, float scale, size_t n,
    size_t hq, size_t tq, float maximum, float denominator, float *result) {
  const size_t group = s->hq / s->hkv;
  const size_t hkv = hq / group;
  float mean = 0.0f;
  for (size_t tk = 0; tk < s->tk; tk++) {
    float probability;
    float dp = 0.0f;
    if (!admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
      continue;
    }
    if (!attention_probability(s, q, k, scale, n, hq, tq, tk, maximum,
                               denominator, &probability)) {
      return 0;
    }
    for (size_t d = 0; d < s->dh; d++) {
      const float product = upstream[q_index(s, n, hq, tq, d)] *
                            v[kv_index(s, n, hkv, tk, d)];
      if (!isfinite(product)) {
        return 0;
      }
      dp = dp + product;
      if (!isfinite(dp)) {
        return 0;
      }
    }
    mean = mean + probability * dp;
    if (!isfinite(mean)) {
      return 0;
    }
  }
  *result = mean;
  return 1;
}

static int attention_forward_preflight(
    const attention_shape *s, const float *q, const float *k, const float *v,
    const int64_t *query_positions, const int64_t *key_positions,
    const uint8_t *mask) {
  const float scale = attention_scale(s->dh);
  for (size_t n = 0; n < s->n; n++) {
    for (size_t hq = 0; hq < s->hq; hq++) {
      const size_t hkv = hq / (s->hq / s->hkv);
      for (size_t tq = 0; tq < s->tq; tq++) {
        float maximum;
        float denominator;
        const int state = attention_row(s, q, k, query_positions, key_positions,
                                        mask, scale, n, hq, tq, &maximum,
                                        &denominator);
        if (state < 0) {
          return 0;
        }
        if (state == 0) {
          continue;
        }
        for (size_t d = 0; d < s->dh; d++) {
          float output = 0.0f;
          for (size_t tk = 0; tk < s->tk; tk++) {
            float probability;
            if (!admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
              continue;
            }
            if (!attention_probability(s, q, k, scale, n, hq, tq, tk,
                                       maximum, denominator, &probability)) {
              return 0;
            }
            output = output + probability * v[kv_index(s, n, hkv, tk, d)];
            if (!isfinite(output)) {
              return 0;
            }
          }
        }
      }
    }
  }
  return 1;
}

static int attention_backward_preflight(
    const attention_shape *s, const float *q, const float *k, const float *v,
    const int64_t *query_positions, const int64_t *key_positions,
    const uint8_t *mask, const float *upstream) {
  const float scale = attention_scale(s->dh);
  /* dQ: accumulation order is key position order. */
  for (size_t n = 0; n < s->n; n++) {
    for (size_t hq = 0; hq < s->hq; hq++) {
      const size_t hkv = hq / (s->hq / s->hkv);
      for (size_t tq = 0; tq < s->tq; tq++) {
        float maximum;
        float denominator;
        float mean_dp;
        const int state = attention_row(s, q, k, query_positions, key_positions,
                                        mask, scale, n, hq, tq, &maximum,
                                        &denominator);
        if (state < 0) {
          return 0;
        }
        if (state == 0) {
          continue;
        }
        if (!attention_mean_dp(s, q, k, v, upstream, query_positions,
                               key_positions, mask, scale, n, hq, tq, maximum,
                               denominator, &mean_dp)) {
          return 0;
        }
        for (size_t d = 0; d < s->dh; d++) {
          float gradient = 0.0f;
          for (size_t tk = 0; tk < s->tk; tk++) {
            float probability;
            float dp = 0.0f;
            float ds;
            if (!admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
              continue;
            }
            if (!attention_probability(s, q, k, scale, n, hq, tq, tk,
                                       maximum, denominator, &probability)) {
              return 0;
            }
            for (size_t inner = 0; inner < s->dh; inner++) {
              dp = dp + upstream[q_index(s, n, hq, tq, inner)] *
                            v[kv_index(s, n, hkv, tk, inner)];
              if (!isfinite(dp)) {
                return 0;
              }
            }
            ds = probability * (dp - mean_dp);
            gradient = gradient + ds * scale * k[kv_index(s, n, hkv, tk, d)];
            if (!isfinite(ds) || !isfinite(gradient)) {
              return 0;
            }
          }
        }
      }
    }
  }
  /* dK and dV: query-head-major, then query-position accumulation order. */
  for (size_t n = 0; n < s->n; n++) {
    for (size_t hkv = 0; hkv < s->hkv; hkv++) {
      const size_t first_hq = hkv * (s->hq / s->hkv);
      const size_t last_hq = first_hq + (s->hq / s->hkv);
      for (size_t tk = 0; tk < s->tk; tk++) {
        for (size_t d = 0; d < s->dh; d++) {
          float dk = 0.0f;
          float dv = 0.0f;
          for (size_t hq = first_hq; hq < last_hq; hq++) {
            for (size_t tq = 0; tq < s->tq; tq++) {
              float maximum;
              float denominator;
              float mean_dp;
              float probability;
              float dp = 0.0f;
              float ds;
              const int state = attention_row(
                  s, q, k, query_positions, key_positions, mask, scale, n, hq,
                  tq, &maximum, &denominator);
              if (state < 0) {
                return 0;
              }
              if (state == 0 ||
                  !admitted(s, query_positions, key_positions, mask, n, tq,
                            tk)) {
                continue;
              }
              if (!attention_mean_dp(s, q, k, v, upstream, query_positions,
                                     key_positions, mask, scale, n, hq, tq,
                                     maximum, denominator, &mean_dp) ||
                  !attention_probability(s, q, k, scale, n, hq, tq, tk,
                                         maximum, denominator, &probability)) {
                return 0;
              }
              for (size_t inner = 0; inner < s->dh; inner++) {
                dp = dp + upstream[q_index(s, n, hq, tq, inner)] *
                              v[kv_index(s, n, hkv, tk, inner)];
                if (!isfinite(dp)) {
                  return 0;
                }
              }
              ds = probability * (dp - mean_dp);
              dk = dk + ds * scale * q[q_index(s, n, hq, tq, d)];
              dv = dv + probability * upstream[q_index(s, n, hq, tq, d)];
              if (!isfinite(ds) || !isfinite(dk) || !isfinite(dv)) {
                return 0;
              }
            }
          }
        }
      }
    }
  }
  return 1;
}

static int32_t validate_attention_call(const et_kernel_call_v1 *call,
                                       int backward,
                                       et_kernel_error *error) {
  const char *operation = backward ? "causal-attention.backward"
                                   : "causal-attention.forward";
  const et_kernel_tensor_view_v1 *inputs[7];
  const et_kernel_tensor_view_v1 *outputs[3];
  size_t dimensions[6];
  attention_shape s;
  size_t q_shape[4];
  size_t kv_shape[4];
  size_t qp_shape[2];
  size_t kp_shape[2];
  size_t mask_shape[3];
  size_t q_elements;
  size_t kv_elements;
  size_t mask_elements;
  int32_t result;

  if (call == NULL || call->request == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                     "call and request descriptors are required");
  }
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      call->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                     "call or request truncates the K1 v1 prefix");
  }
  if (!exact_text(call->capability, "kernel.causal-attention") ||
      !exact_text(call->request->operation, operation) ||
      !exact_text(call->request->dtype, "f32") ||
      !exact_text(call->request->device, "cpu") ||
      call->request->rank != 6u || call->request->shape == NULL ||
      !shape_to_size(6u, call->request->shape, dimensions)) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "attention request is outside A2 v1 evidence");
  }
  s = (attention_shape){dimensions[0], dimensions[1], dimensions[2],
                        dimensions[3], dimensions[4], dimensions[5]};
  if (s.hkv < 2u || s.hq < s.hkv || s.hq % s.hkv != 0u) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, operation,
                     "attention requires Hkv>=2 and Hq divisible by Hkv");
  }
  result = table_views(call, 0, backward ? 7u : 6u, operation, inputs, error);
  if (result != 0) {
    return result;
  }
  if (ranges_overlap(inputs[1]->data, inputs[1]->byte_length, inputs[2]->data,
                     inputs[2]->byte_length)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                     "attention key and value storage must be distinct");
  }
  result = table_views(call, 1, backward ? 3u : 1u, operation, outputs, error);
  if (result != 0) {
    return result;
  }
  q_shape[0] = s.n;
  q_shape[1] = s.hq;
  q_shape[2] = s.tq;
  q_shape[3] = s.dh;
  kv_shape[0] = s.n;
  kv_shape[1] = s.hkv;
  kv_shape[2] = s.tk;
  kv_shape[3] = s.dh;
  qp_shape[0] = s.n;
  qp_shape[1] = s.tq;
  kp_shape[0] = s.n;
  kp_shape[1] = s.tk;
  mask_shape[0] = s.n;
  mask_shape[1] = s.tq;
  mask_shape[2] = s.tk;
  result = validate_view(inputs[0], "f32", sizeof(float), _Alignof(float), 4u,
                         q_shape, operation, error);
  if (result == 0) result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float), 4u, kv_shape, operation, error);
  if (result == 0) result = validate_view(inputs[2], "f32", sizeof(float), _Alignof(float), 4u, kv_shape, operation, error);
  if (result == 0) result = validate_view(inputs[3], "i64", sizeof(int64_t), _Alignof(int64_t), 2u, qp_shape, operation, error);
  if (result == 0) result = validate_view(inputs[4], "i64", sizeof(int64_t), _Alignof(int64_t), 2u, kp_shape, operation, error);
  if (result == 0) result = validate_view(inputs[5], "bool", sizeof(uint8_t), _Alignof(uint8_t), 3u, mask_shape, operation, error);
  if (result == 0 && backward) result = validate_view(inputs[6], "f32", sizeof(float), _Alignof(float), 4u, q_shape, operation, error);
  if (result == 0) result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float), 4u, q_shape, operation, error);
  if (result == 0 && backward) result = validate_view(outputs[1], "f32", sizeof(float), _Alignof(float), 4u, kv_shape, operation, error);
  if (result == 0 && backward) result = validate_view(outputs[2], "f32", sizeof(float), _Alignof(float), 4u, kv_shape, operation, error);
  if (result != 0) {
    return result;
  }
  q_elements = inputs[0]->byte_length / sizeof(float);
  kv_elements = inputs[1]->byte_length / sizeof(float);
  mask_elements = inputs[5]->byte_length;
  if (!validate_finite((const float *)inputs[0]->data, q_elements) ||
      !validate_finite((const float *)inputs[1]->data, kv_elements) ||
      !validate_finite((const float *)inputs[2]->data, kv_elements) ||
      (backward && !validate_finite((const float *)inputs[6]->data,
                                    q_elements))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "attention floating inputs must be finite");
  }
  if (!validate_positions((const int64_t *)inputs[3]->data, s.n, s.tq) ||
      !validate_positions((const int64_t *)inputs[4]->data, s.n, s.tk)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "attention positions must be strictly increasing and exact");
  }
  for (size_t index = 0; index < mask_elements; index++) {
    if (((const uint8_t *)inputs[5]->data)[index] > 1u) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_BUFFER, operation,
                       "attention bool mask contains a noncanonical value");
    }
  }
  if (backward) {
    if (!attention_backward_preflight(
            &s, (const float *)inputs[0]->data,
            (const float *)inputs[1]->data, (const float *)inputs[2]->data,
            (const int64_t *)inputs[3]->data,
            (const int64_t *)inputs[4]->data,
            (const uint8_t *)inputs[5]->data,
            (const float *)inputs[6]->data)) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "attention backward would produce a nonfinite result");
    }
  } else if (!attention_forward_preflight(
                 &s, (const float *)inputs[0]->data,
                 (const float *)inputs[1]->data,
                 (const float *)inputs[2]->data,
                 (const int64_t *)inputs[3]->data,
                 (const int64_t *)inputs[4]->data,
                 (const uint8_t *)inputs[5]->data)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "attention forward would produce a nonfinite result");
  }
  et_kernel_error_clear(error);
  return 0;
}

static void invoke_attention_forward(const et_kernel_call_v1 *call) {
  const et_kernel_tensor_view_v1 *q_view = input_at(call, 0u);
  const et_kernel_tensor_view_v1 *k_view = input_at(call, 1u);
  const et_kernel_tensor_view_v1 *v_view = input_at(call, 2u);
  const et_kernel_tensor_view_v1 *query_position_view = input_at(call, 3u);
  const et_kernel_tensor_view_v1 *key_position_view = input_at(call, 4u);
  const et_kernel_tensor_view_v1 *mask_view = input_at(call, 5u);
  et_kernel_tensor_view_v1 *output_view = output_at(call, 0u);
  const uint64_t *shape = call->request->shape;
  const attention_shape s = {(size_t)shape[0], (size_t)shape[1],
                             (size_t)shape[2], (size_t)shape[3],
                             (size_t)shape[4], (size_t)shape[5]};
  const float *q = (const float *)q_view->data;
  const float *k = (const float *)k_view->data;
  const float *v = (const float *)v_view->data;
  const int64_t *query_positions = (const int64_t *)query_position_view->data;
  const int64_t *key_positions = (const int64_t *)key_position_view->data;
  const uint8_t *mask = (const uint8_t *)mask_view->data;
  float *output = (float *)output_view->data;
  const float scale = attention_scale(s.dh);
  memset(output, 0, output_view->byte_length);
  for (size_t n = 0; n < s.n; n++) {
    for (size_t hq = 0; hq < s.hq; hq++) {
      const size_t hkv = hq / (s.hq / s.hkv);
      for (size_t tq = 0; tq < s.tq; tq++) {
        float maximum;
        float denominator;
        if (attention_row(&s, q, k, query_positions, key_positions, mask,
                          scale, n, hq, tq, &maximum, &denominator) <= 0) {
          continue;
        }
        for (size_t tk = 0; tk < s.tk; tk++) {
          float probability = 0.0f;
          if (!admitted(&s, query_positions, key_positions, mask, n, tq, tk)) {
            continue;
          }
          (void)attention_probability(&s, q, k, scale, n, hq, tq, tk,
                                      maximum, denominator, &probability);
          for (size_t d = 0; d < s.dh; d++) {
            output[q_index(&s, n, hq, tq, d)] =
                output[q_index(&s, n, hq, tq, d)] +
                probability * v[kv_index(&s, n, hkv, tk, d)];
          }
        }
      }
    }
  }
}

static void invoke_attention_backward(const et_kernel_call_v1 *call) {
  const et_kernel_tensor_view_v1 *q_view = input_at(call, 0u);
  const et_kernel_tensor_view_v1 *k_view = input_at(call, 1u);
  const et_kernel_tensor_view_v1 *v_view = input_at(call, 2u);
  const et_kernel_tensor_view_v1 *query_position_view = input_at(call, 3u);
  const et_kernel_tensor_view_v1 *key_position_view = input_at(call, 4u);
  const et_kernel_tensor_view_v1 *mask_view = input_at(call, 5u);
  const et_kernel_tensor_view_v1 *upstream_view = input_at(call, 6u);
  et_kernel_tensor_view_v1 *dq_view = output_at(call, 0u);
  et_kernel_tensor_view_v1 *dk_view = output_at(call, 1u);
  et_kernel_tensor_view_v1 *dv_view = output_at(call, 2u);
  const uint64_t *shape = call->request->shape;
  const attention_shape s = {(size_t)shape[0], (size_t)shape[1],
                             (size_t)shape[2], (size_t)shape[3],
                             (size_t)shape[4], (size_t)shape[5]};
  const float *q = (const float *)q_view->data;
  const float *k = (const float *)k_view->data;
  const float *v = (const float *)v_view->data;
  const int64_t *query_positions = (const int64_t *)query_position_view->data;
  const int64_t *key_positions = (const int64_t *)key_position_view->data;
  const uint8_t *mask = (const uint8_t *)mask_view->data;
  const float *upstream = (const float *)upstream_view->data;
  float *dq = (float *)dq_view->data;
  float *dk = (float *)dk_view->data;
  float *dv = (float *)dv_view->data;
  const float scale = attention_scale(s.dh);
  memset(dq, 0, dq_view->byte_length);
  memset(dk, 0, dk_view->byte_length);
  memset(dv, 0, dv_view->byte_length);
  for (size_t n = 0; n < s.n; n++) {
    for (size_t hq = 0; hq < s.hq; hq++) {
      const size_t hkv = hq / (s.hq / s.hkv);
      for (size_t tq = 0; tq < s.tq; tq++) {
        float maximum;
        float denominator;
        float mean_dp = 0.0f;
        if (attention_row(&s, q, k, query_positions, key_positions, mask,
                          scale, n, hq, tq, &maximum, &denominator) <= 0) {
          continue;
        }
        (void)attention_mean_dp(&s, q, k, v, upstream, query_positions,
                                key_positions, mask, scale, n, hq, tq, maximum,
                                denominator, &mean_dp);
        for (size_t tk = 0; tk < s.tk; tk++) {
          float probability = 0.0f;
          float dp = 0.0f;
          float ds;
          if (!admitted(&s, query_positions, key_positions, mask, n, tq, tk)) {
            continue;
          }
          (void)attention_probability(&s, q, k, scale, n, hq, tq, tk,
                                      maximum, denominator, &probability);
          for (size_t d = 0; d < s.dh; d++) {
            dp = dp + upstream[q_index(&s, n, hq, tq, d)] *
                          v[kv_index(&s, n, hkv, tk, d)];
          }
          ds = probability * (dp - mean_dp);
          for (size_t d = 0; d < s.dh; d++) {
            dq[q_index(&s, n, hq, tq, d)] =
                dq[q_index(&s, n, hq, tq, d)] +
                ds * scale * k[kv_index(&s, n, hkv, tk, d)];
            dk[kv_index(&s, n, hkv, tk, d)] =
                dk[kv_index(&s, n, hkv, tk, d)] +
                ds * scale * q[q_index(&s, n, hq, tq, d)];
            dv[kv_index(&s, n, hkv, tk, d)] =
                dv[kv_index(&s, n, hkv, tk, d)] +
                probability * upstream[q_index(&s, n, hq, tq, d)];
          }
        }
      }
    }
  }
}

static int rope_pair(float even, float odd, float cosine, float sine,
                     int backward, float *first, float *second) {
  const float a = even * cosine;
  const float b = odd * sine;
  const float c = even * sine;
  const float d = odd * cosine;
  if (!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d)) {
    return 0;
  }
  *first = backward ? a + b : a - b;
  *second = backward ? -c + d : c + d;
  return isfinite(*first) && isfinite(*second);
}

static int32_t validate_rope_call(const et_kernel_call_v1 *call, int backward,
                                  et_kernel_error *error) {
  const char *operation = backward ? "rope.backward" : "rope.forward";
  const et_kernel_tensor_view_v1 *inputs[3];
  const et_kernel_tensor_view_v1 *outputs[1];
  size_t dimensions[4];
  rope_shape s;
  size_t x_shape[4];
  size_t position_shape[2];
  size_t frequency_shape[1];
  size_t x_elements;
  int32_t result;
  if (call == NULL || call->request == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                     "call and request descriptors are required");
  }
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      call->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                     "call or request truncates the K1 v1 prefix");
  }
  if (!exact_text(call->capability, "kernel.rope") ||
      !exact_text(call->request->operation, operation) ||
      !exact_text(call->request->dtype, "f32") ||
      !exact_text(call->request->device, "cpu") ||
      call->request->rank != 4u || call->request->shape == NULL ||
      !shape_to_size(4u, call->request->shape, dimensions)) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "RoPE request is outside A2 v1 evidence");
  }
  s = (rope_shape){dimensions[0], dimensions[1], dimensions[2], dimensions[3]};
  if (s.dh < 2u || s.dh % 2u != 0u) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, operation,
                     "RoPE requires an even head dimension of at least two");
  }
  result = table_views(call, 0, 3u, operation, inputs, error);
  if (result != 0) return result;
  result = table_views(call, 1, 1u, operation, outputs, error);
  if (result != 0) return result;
  x_shape[0] = s.n;
  x_shape[1] = s.h;
  x_shape[2] = s.t;
  x_shape[3] = s.dh;
  position_shape[0] = s.n;
  position_shape[1] = s.t;
  frequency_shape[0] = s.dh / 2u;
  result = validate_view(inputs[0], "f32", sizeof(float), _Alignof(float), 4u,
                         x_shape, operation, error);
  if (result == 0) result = validate_view(inputs[1], "i64", sizeof(int64_t), _Alignof(int64_t), 2u, position_shape, operation, error);
  if (result == 0) result = validate_view(inputs[2], "f32", sizeof(float), _Alignof(float), 1u, frequency_shape, operation, error);
  if (result == 0) result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float), 4u, x_shape, operation, error);
  if (result != 0) return result;
  x_elements = inputs[0]->byte_length / sizeof(float);
  if (!validate_finite((const float *)inputs[0]->data, x_elements) ||
      !validate_positions((const int64_t *)inputs[1]->data, s.n, s.t)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "RoPE inputs or positions are outside the finite exact domain");
  }
  for (size_t pair = 0; pair < s.dh / 2u; pair++) {
    const float frequency = ((const float *)inputs[2]->data)[pair];
    if (!isfinite(frequency) || frequency <= 0.0f) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_BUFFER, operation,
                       "RoPE inverse frequencies must be positive and finite");
    }
  }
  for (size_t n = 0; n < s.n; n++) {
    for (size_t t = 0; t < s.t; t++) {
      const int64_t position = ((const int64_t *)inputs[1]->data)[n * s.t + t];
      for (size_t pair = 0; pair < s.dh / 2u; pair++) {
        const float angle = (float)position * ((const float *)inputs[2]->data)[pair];
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        if (!isfinite(angle) || !isfinite(cosine) || !isfinite(sine)) {
          return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                           ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                           "RoPE angle is outside the finite numerical domain");
        }
        for (size_t h = 0; h < s.h; h++) {
          const size_t base = (((n * s.h + h) * s.t + t) * s.dh) + pair * 2u;
          float first;
          float second;
          if (!rope_pair(((const float *)inputs[0]->data)[base],
                         ((const float *)inputs[0]->data)[base + 1u], cosine,
                         sine, backward, &first, &second)) {
            return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                             ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                             "RoPE output would be nonfinite");
          }
        }
      }
    }
  }
  et_kernel_error_clear(error);
  return 0;
}

static void invoke_rope(const et_kernel_call_v1 *call, int backward) {
  const et_kernel_tensor_view_v1 *x_view = input_at(call, 0u);
  const et_kernel_tensor_view_v1 *position_view = input_at(call, 1u);
  const et_kernel_tensor_view_v1 *frequency_view = input_at(call, 2u);
  et_kernel_tensor_view_v1 *output_view = output_at(call, 0u);
  const uint64_t *shape = call->request->shape;
  const rope_shape s = {(size_t)shape[0], (size_t)shape[1], (size_t)shape[2],
                        (size_t)shape[3]};
  const float *x = (const float *)x_view->data;
  const int64_t *positions = (const int64_t *)position_view->data;
  const float *inv_freq = (const float *)frequency_view->data;
  float *output = (float *)output_view->data;
  for (size_t n = 0; n < s.n; n++) {
    for (size_t h = 0; h < s.h; h++) {
      for (size_t t = 0; t < s.t; t++) {
        const int64_t position = positions[n * s.t + t];
        for (size_t pair = 0; pair < s.dh / 2u; pair++) {
          const size_t base = (((n * s.h + h) * s.t + t) * s.dh) + pair * 2u;
          const float angle = (float)position * inv_freq[pair];
          const float cosine = cosf(angle);
          const float sine = sinf(angle);
          (void)rope_pair(x[base], x[base + 1u], cosine, sine, backward,
                          &output[base], &output[base + 1u]);
        }
      }
    }
  }
}

static int32_t provider_validate_call(const et_kernel_call_v1 *call,
                                      et_kernel_error *error) {
  if (call == NULL || call->request == NULL || call->request->operation == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "a2.provider",
                     "call, request, and operation are required");
  }
  if (exact_text(call->request->operation, "causal-attention.forward")) {
    return validate_attention_call(call, 0, error);
  }
  if (exact_text(call->request->operation, "causal-attention.backward")) {
    return validate_attention_call(call, 1, error);
  }
  if (exact_text(call->request->operation, "rope.forward")) {
    return validate_rope_call(call, 0, error);
  }
  if (exact_text(call->request->operation, "rope.backward")) {
    return validate_rope_call(call, 1, error);
  }
  return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                   ET_KERNEL_CODE_PROVIDER_REJECTED, "a2.provider",
                   "operation is outside A2 v1 evidence");
}

static void provider_invoke_call(const et_kernel_call_v1 *call) {
  if (exact_text(call->request->operation, "causal-attention.forward")) {
    invoke_attention_forward(call);
  } else if (exact_text(call->request->operation,
                        "causal-attention.backward")) {
    invoke_attention_backward(call);
  } else if (exact_text(call->request->operation, "rope.forward")) {
    invoke_rope(call, 0);
  } else {
    invoke_rope(call, 1);
  }
}

#define EXACT_DIMENSION(value)                                                   \
  {.minimum = (value), .maximum = (value), .maximum_unbounded = 0u,             \
   .reserved = {0}}

static const et_kernel_dimension_range_v1 attention_mha_t1_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(1u), EXACT_DIMENSION(1u), EXACT_DIMENSION(1u)};
static const et_kernel_dimension_range_v1 attention_mha_square_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(2u), EXACT_DIMENSION(2u), EXACT_DIMENSION(1u)};
static const et_kernel_dimension_range_v1 attention_aot_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(1u)};
static const et_kernel_dimension_range_v1 attention_scale_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 attention_mha_d2_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(2u), EXACT_DIMENSION(2u), EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 attention_gqa_t1_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(4u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(1u), EXACT_DIMENSION(1u), EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 attention_gqa_t3_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(4u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(3u), EXACT_DIMENSION(3u), EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 attention_n2_rectangular_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(4u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(2u), EXACT_DIMENSION(3u), EXACT_DIMENSION(4u)};
static const et_kernel_dimension_range_v1 attention_n2_cache_full_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(4u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(3u), EXACT_DIMENSION(3u), EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 attention_n2_cache_step_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(4u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(1u), EXACT_DIMENSION(3u), EXACT_DIMENSION(2u)};

static const et_kernel_dimension_range_v1 rope_t1_d2_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(1u), EXACT_DIMENSION(1u),
    EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 rope_t2_d2_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(1u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 rope_t2_d4_dimensions[] = {
    EXACT_DIMENSION(1u), EXACT_DIMENSION(1u), EXACT_DIMENSION(2u),
    EXACT_DIMENSION(4u)};
static const et_kernel_dimension_range_v1 rope_n2_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(2u), EXACT_DIMENSION(3u),
    EXACT_DIMENSION(4u)};
static const et_kernel_dimension_range_v1 rope_n2_q_cache_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(4u), EXACT_DIMENSION(3u),
    EXACT_DIMENSION(2u)};
static const et_kernel_dimension_range_v1 rope_n2_k_cache_dimensions[] = {
    EXACT_DIMENSION(2u), EXACT_DIMENSION(2u), EXACT_DIMENSION(3u),
    EXACT_DIMENSION(2u)};

static const et_kernel_shape_range_v1 attention_ranges[] = {
    {.rank = 6u, .dimensions = attention_mha_t1_dimensions},
    {.rank = 6u, .dimensions = attention_mha_square_dimensions},
    {.rank = 6u, .dimensions = attention_aot_dimensions},
    {.rank = 6u, .dimensions = attention_scale_dimensions},
    {.rank = 6u, .dimensions = attention_mha_d2_dimensions},
    {.rank = 6u, .dimensions = attention_gqa_t1_dimensions},
    {.rank = 6u, .dimensions = attention_gqa_t3_dimensions},
    {.rank = 6u, .dimensions = attention_n2_rectangular_dimensions},
    {.rank = 6u, .dimensions = attention_n2_cache_full_dimensions},
    {.rank = 6u, .dimensions = attention_n2_cache_step_dimensions},
};
static const et_kernel_shape_range_v1 rope_ranges[] = {
    {.rank = 4u, .dimensions = rope_t1_d2_dimensions},
    {.rank = 4u, .dimensions = rope_t2_d2_dimensions},
    {.rank = 4u, .dimensions = rope_t2_d4_dimensions},
    {.rank = 4u, .dimensions = rope_n2_dimensions},
    {.rank = 4u, .dimensions = rope_n2_q_cache_dimensions},
    {.rank = 4u, .dimensions = rope_n2_k_cache_dimensions},
};

#undef EXACT_DIMENSION
static const char *const attention_operations[] = {
    "causal-attention.forward", "causal-attention.backward"};
static const char *const rope_operations[] = {"rope.forward", "rope.backward"};
static const char *const provider_dtypes[] = {"f32"};
static const char *const provider_devices[] = {"cpu"};
static const et_kernel_capability_v1 provider_capabilities[] = {
    {.struct_size = sizeof(et_kernel_capability_v1),
     .name = "kernel.causal-attention",
     .status = ET_KERNEL_CAPABILITY_VERIFIED,
     .implementation = "a2.cpu-f32.serial",
     .version = "1.0",
     .evidence = "A2:cpu-f32-attention-rope-v1",
     .deterministic = 1u,
     .reserved = {0},
     .operation_count = ET_ARRAY_COUNT(attention_operations),
     .operations = attention_operations,
     .dtype_count = ET_ARRAY_COUNT(provider_dtypes),
     .dtypes = provider_dtypes,
     .device_count = ET_ARRAY_COUNT(provider_devices),
     .devices = provider_devices,
     .shape_range_count = ET_ARRAY_COUNT(attention_ranges),
     .shape_ranges = attention_ranges},
    {.struct_size = sizeof(et_kernel_capability_v1),
     .name = "kernel.rope",
     .status = ET_KERNEL_CAPABILITY_VERIFIED,
     .implementation = "a2.cpu-f32.serial",
     .version = "1.0",
     .evidence = "A2:cpu-f32-attention-rope-v1",
     .deterministic = 1u,
     .reserved = {0},
     .operation_count = ET_ARRAY_COUNT(rope_operations),
     .operations = rope_operations,
     .dtype_count = ET_ARRAY_COUNT(provider_dtypes),
     .dtypes = provider_dtypes,
     .device_count = ET_ARRAY_COUNT(provider_devices),
     .devices = provider_devices,
     .shape_range_count = ET_ARRAY_COUNT(rope_ranges),
     .shape_ranges = rope_ranges},
};
static const et_kernel_provider_v1 provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = ET_KERNEL_ABI_MINOR,
    .required_features = 0u,
    .name = "a2.cpu-f32.serial",
    .version = "1.0",
    .evidence = "A2:cpu-f32-attention-rope-v1",
    .capability_count = ET_ARRAY_COUNT(provider_capabilities),
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(provider_capabilities),
    .capabilities = provider_capabilities,
    .validate_call = provider_validate_call,
    .invoke_call = provider_invoke_call,
};

const et_kernel_provider_v1 *et_a2_kernel_provider_v1(void) {
  return &provider;
}
