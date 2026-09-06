#include "eshkol_transformer/n2_primitives_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if !defined(__x86_64__)
#error "N2 v1 supports only the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ET_ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

_Static_assert(CHAR_BIT == 8, "N2 v1 requires eight-bit bytes");
_Static_assert(sizeof(float) == 4u, "N2 v1 requires binary32 float");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "N2 v1 requires IEEE-754 binary32 float");
_Static_assert(FLT_EVAL_METHOD == 0,
               "N2 v1 requires standard binary32 evaluation");

typedef struct shape4 {
  size_t n;
  size_t t;
  size_t a;
  size_t b;
} shape4;

typedef struct shape3 {
  size_t n;
  size_t t;
  size_t d;
} shape3;

typedef struct rng_state {
  uint64_t key;
  uint64_t counter_low;
  uint64_t counter_high;
} rng_state;

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
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
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

static int request_shape(const et_kernel_request_v1 *request, size_t rank,
                         size_t *dimensions) {
  if (request->rank != rank || request->shape == NULL ||
      !aligned_pointer(request->shape, _Alignof(uint64_t)) ||
      rank > SIZE_MAX / sizeof(*request->shape) ||
      !pointer_span_fits(request->shape, rank * sizeof(*request->shape))) {
    return 0;
  }
  for (size_t index = 0; index < rank; index++) {
    if (request->shape[index] == 0u || request->shape[index] > SIZE_MAX) {
      return 0;
    }
    dimensions[index] = (size_t)request->shape[index];
  }
  return 1;
}

static int row4_admitted(const size_t *d, const size_t rows[][4],
                         size_t count) {
  for (size_t row = 0; row < count; row++) {
    if (d[0] == rows[row][0] && d[1] == rows[row][1] &&
        d[2] == rows[row][2] && d[3] == rows[row][3]) {
      return 1;
    }
  }
  return 0;
}

static int row3_admitted(const size_t *d, const size_t rows[][3],
                         size_t count) {
  for (size_t row = 0; row < count; row++) {
    if (d[0] == rows[row][0] && d[1] == rows[row][1] &&
        d[2] == rows[row][2]) {
      return 1;
    }
  }
  return 0;
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
                     output ? "output count does not match operation schema"
                            : "input count does not match operation schema");
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
                     "tensor dtype does not match operation schema");
  }
  if (!exact_text(view->device, "cpu")) {
    return set_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, operation,
                     "N2 v1 accepts only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "N2 v1 requires dense zero-offset tensor views");
  }
  if (view->rank != rank || (rank != 0u && view->shape == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, operation,
                     "tensor rank does not match operation schema");
  }
  if (rank != 0u &&
      (!aligned_pointer(view->shape, _Alignof(uint64_t)) ||
       rank > SIZE_MAX / sizeof(*view->shape) ||
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
                       "tensor extent does not match operation schema");
    }
  }
  if (!checked_product(&expected_bytes, elements, element_size) ||
      view->byte_length != expected_bytes ||
      !aligned_pointer(view->data, alignment) ||
      !pointer_span_fits(view->data, expected_bytes)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "tensor data alignment or byte span is invalid");
  }
  return 0;
}

static int same_view(const et_kernel_tensor_view_v1 *left,
                     const et_kernel_tensor_view_v1 *right) {
  if (left->data != right->data || left->byte_length != right->byte_length ||
      left->rank != right->rank || left->layout != right->layout ||
      left->offset_bytes != right->offset_bytes ||
      !exact_text(left->dtype, right->dtype) ||
      !exact_text(left->device, right->device)) {
    return 0;
  }
  for (size_t index = 0; index < left->rank; index++) {
    if (left->shape[index] != right->shape[index]) {
      return 0;
    }
  }
  return 1;
}

static int32_t validate_input_aliases(
    const et_kernel_tensor_view_v1 *const *inputs, size_t count,
    int residual_forward, const char *operation, et_kernel_error *error) {
  for (size_t left = 0; left < count; left++) {
    for (size_t right = left + 1u; right < count; right++) {
      if (!ranges_overlap(inputs[left]->data, inputs[left]->byte_length,
                          inputs[right]->data, inputs[right]->byte_length)) {
        continue;
      }
      if (residual_forward && left == 0u && right == 1u &&
          same_view(inputs[left], inputs[right])) {
        continue;
      }
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "input tensor storage overlap is not permitted");
    }
  }
  return 0;
}

static int finite_values(const float *values, size_t count) {
  for (size_t index = 0; index < count; index++) {
    if (!isfinite(values[index])) return 0;
  }
  return 1;
}

static int fp_environment_supported(void) {
  const unsigned int mxcsr = _mm_getcsr();
  const unsigned int exception_masks = UINT32_C(0x00001f80);
  const unsigned int rounding = UINT32_C(0x00006000);
  const unsigned int daz = UINT32_C(0x00000040);
  const unsigned int ftz = UINT32_C(0x00008000);
  return fegetround() == FE_TONEAREST &&
         (mxcsr & exception_masks) == exception_masks &&
         (mxcsr & rounding) == 0u && (mxcsr & daz) == 0u &&
         (mxcsr & ftz) == 0u;
}

static size_t index3(const shape3 *s, size_t n, size_t t, size_t d) {
  return (n * s->t + t) * s->d + d;
}

static size_t index_linear_x(const shape4 *s, size_t n, size_t t, size_t i) {
  return (n * s->t + t) * s->a + i;
}

static size_t index_linear_y(const shape4 *s, size_t n, size_t t, size_t o) {
  return (n * s->t + t) * s->b + o;
}

static int embedding_forward_run(const shape4 *s, const int64_t *ids,
                                 const float *weight, float *output) {
  for (size_t n = 0; n < s->n; n++) {
    for (size_t t = 0; t < s->t; t++) {
      const size_t row = (size_t)ids[n * s->t + t];
      for (size_t d = 0; d < s->b; d++) {
        const float value = weight[row * s->b + d];
        if (!isfinite(value)) return 0;
        if (output != NULL) output[(n * s->t + t) * s->b + d] = value;
      }
    }
  }
  return 1;
}

static int embedding_backward_run(const shape4 *s, const int64_t *ids,
                                  const float *upstream, float *dweight) {
  for (size_t v = 0; v < s->a; v++) {
    for (size_t d = 0; d < s->b; d++) {
      float sum = 0.0f;
      for (size_t n = 0; n < s->n; n++) {
        for (size_t t = 0; t < s->t; t++) {
          if ((size_t)ids[n * s->t + t] == v) {
            sum = sum + upstream[(n * s->t + t) * s->b + d];
            if (!isfinite(sum)) return 0;
          }
        }
      }
      if (dweight != NULL) dweight[v * s->b + d] = sum;
    }
  }
  return 1;
}

static int linear_forward_run(const shape4 *s, const float *x,
                              const float *weight, const float *bias,
                              float *output) {
  for (size_t n = 0; n < s->n; n++) {
    for (size_t t = 0; t < s->t; t++) {
      for (size_t o = 0; o < s->b; o++) {
        float sum = 0.0f;
        for (size_t i = 0; i < s->a; i++) {
          const float product = x[index_linear_x(s, n, t, i)] *
                                weight[o * s->a + i];
          if (!isfinite(product)) return 0;
          sum = sum + product;
          if (!isfinite(sum)) return 0;
        }
        if (bias != NULL) {
          sum = sum + bias[o];
          if (!isfinite(sum)) return 0;
        }
        if (output != NULL) output[index_linear_y(s, n, t, o)] = sum;
      }
    }
  }
  return 1;
}

static int linear_backward_run(const shape4 *s, const float *x,
                               const float *weight, const float *upstream,
                               int include_dbias, float *dx, float *dweight,
                               float *dbias) {
  for (size_t n = 0; n < s->n; n++) {
    for (size_t t = 0; t < s->t; t++) {
      for (size_t i = 0; i < s->a; i++) {
        float sum = 0.0f;
        for (size_t o = 0; o < s->b; o++) {
          const float product = upstream[index_linear_y(s, n, t, o)] *
                                weight[o * s->a + i];
          if (!isfinite(product)) return 0;
          sum = sum + product;
          if (!isfinite(sum)) return 0;
        }
        if (dx != NULL) dx[index_linear_x(s, n, t, i)] = sum;
      }
    }
  }
  for (size_t o = 0; o < s->b; o++) {
    for (size_t i = 0; i < s->a; i++) {
      float sum = 0.0f;
      for (size_t n = 0; n < s->n; n++) {
        for (size_t t = 0; t < s->t; t++) {
          const float product = upstream[index_linear_y(s, n, t, o)] *
                                x[index_linear_x(s, n, t, i)];
          if (!isfinite(product)) return 0;
          sum = sum + product;
          if (!isfinite(sum)) return 0;
        }
      }
      if (dweight != NULL) dweight[o * s->a + i] = sum;
    }
  }
  if (include_dbias) {
    for (size_t o = 0; o < s->b; o++) {
      float sum = 0.0f;
      for (size_t n = 0; n < s->n; n++) {
        for (size_t t = 0; t < s->t; t++) {
          sum = sum + upstream[index_linear_y(s, n, t, o)];
          if (!isfinite(sum)) return 0;
        }
      }
      if (dbias != NULL) dbias[o] = sum;
    }
  }
  return 1;
}

static int layer_norm_statistics(const shape3 *s, const float *x, size_t n,
                                 size_t t, float *mean, float *rstd) {
  float sum = 0.0f;
  float variance_sum = 0.0f;
  const float dimension = (float)s->d;
  for (size_t d = 0; d < s->d; d++) {
    sum = sum + x[index3(s, n, t, d)];
    if (!isfinite(sum)) return 0;
  }
  *mean = sum / dimension;
  if (!isfinite(*mean)) return 0;
  for (size_t d = 0; d < s->d; d++) {
    const float delta = x[index3(s, n, t, d)] - *mean;
    const float square = delta * delta;
    if (!isfinite(delta) || !isfinite(square)) return 0;
    variance_sum = variance_sum + square;
    if (!isfinite(variance_sum)) return 0;
  }
  *rstd = variance_sum / dimension;
  return isfinite(*rstd);
}

static int layer_norm_complete_statistics(float variance, float epsilon,
                                          float *rstd) {
  const float adjusted = variance + epsilon;
  const float root = sqrtf(adjusted);
  if (!isfinite(adjusted) || !isfinite(root)) return 0;
  *rstd = 1.0f / root;
  return isfinite(*rstd);
}

static int layer_norm_forward_run(const shape3 *s, const float *x,
                                  const float *gamma, const float *beta,
                                  float epsilon, float *output) {
  for (size_t n = 0; n < s->n; n++) {
    for (size_t t = 0; t < s->t; t++) {
      float mean;
      float variance;
      float rstd;
      if (!layer_norm_statistics(s, x, n, t, &mean, &variance) ||
          !layer_norm_complete_statistics(variance, epsilon, &rstd)) return 0;
      for (size_t d = 0; d < s->d; d++) {
        const float delta = x[index3(s, n, t, d)] - mean;
        const float normalized = delta * rstd;
        const float scaled = normalized * gamma[d];
        const float value = scaled + beta[d];
        if (!isfinite(delta) || !isfinite(normalized) || !isfinite(scaled) ||
            !isfinite(value)) return 0;
        if (output != NULL) output[index3(s, n, t, d)] = value;
      }
    }
  }
  return 1;
}

static int layer_norm_backward_run(const shape3 *s, const float *x,
                                   const float *gamma, float epsilon,
                                   const float *upstream, float *dx,
                                   float *dgamma, float *dbeta) {
  const float dimension = (float)s->d;
  for (size_t n = 0; n < s->n; n++) {
    for (size_t t = 0; t < s->t; t++) {
      float mean;
      float variance;
      float rstd;
      float sum_h = 0.0f;
      float sum_h_normalized = 0.0f;
      if (!layer_norm_statistics(s, x, n, t, &mean, &variance) ||
          !layer_norm_complete_statistics(variance, epsilon, &rstd)) return 0;
      for (size_t d = 0; d < s->d; d++) {
        const float delta = x[index3(s, n, t, d)] - mean;
        const float normalized = delta * rstd;
        const float h = upstream[index3(s, n, t, d)] * gamma[d];
        const float product = h * normalized;
        if (!isfinite(delta) || !isfinite(normalized) || !isfinite(h) ||
            !isfinite(product)) return 0;
        sum_h = sum_h + h;
        if (!isfinite(sum_h)) return 0;
        sum_h_normalized = sum_h_normalized + product;
        if (!isfinite(sum_h_normalized)) return 0;
      }
      {
        const float reciprocal_dimension = 1.0f / dimension;
        const float factor = rstd * reciprocal_dimension;
        if (!isfinite(reciprocal_dimension) || !isfinite(factor)) return 0;
        for (size_t d = 0; d < s->d; d++) {
          const float delta = x[index3(s, n, t, d)] - mean;
          const float normalized = delta * rstd;
          const float h = upstream[index3(s, n, t, d)] * gamma[d];
          const float first = dimension * h;
          const float second = first - sum_h;
          const float third = normalized * sum_h_normalized;
          const float fourth = second - third;
          const float value = factor * fourth;
          if (!isfinite(delta) || !isfinite(normalized) || !isfinite(h) ||
              !isfinite(first) || !isfinite(second) || !isfinite(third) ||
              !isfinite(fourth) || !isfinite(value)) return 0;
          if (dx != NULL) dx[index3(s, n, t, d)] = value;
        }
      }
    }
  }
  for (size_t d = 0; d < s->d; d++) {
    float sum = 0.0f;
    for (size_t n = 0; n < s->n; n++) {
      for (size_t t = 0; t < s->t; t++) {
        float mean;
        float variance;
        float rstd;
        float delta;
        float normalized;
        float product;
        if (!layer_norm_statistics(s, x, n, t, &mean, &variance) ||
            !layer_norm_complete_statistics(variance, epsilon, &rstd)) return 0;
        delta = x[index3(s, n, t, d)] - mean;
        normalized = delta * rstd;
        product = upstream[index3(s, n, t, d)] * normalized;
        if (!isfinite(delta) || !isfinite(normalized) || !isfinite(product))
          return 0;
        sum = sum + product;
        if (!isfinite(sum)) return 0;
      }
    }
    if (dgamma != NULL) dgamma[d] = sum;
  }
  for (size_t d = 0; d < s->d; d++) {
    float sum = 0.0f;
    for (size_t n = 0; n < s->n; n++) {
      for (size_t t = 0; t < s->t; t++) {
        sum = sum + upstream[index3(s, n, t, d)];
        if (!isfinite(sum)) return 0;
      }
    }
    if (dbeta != NULL) dbeta[d] = sum;
  }
  return 1;
}

static int gelu_one(float x, float *result) {
  const float half = 0x1p-1f;
  const float one = 0x1p+0f;
  const float inv_sqrt_2 = 0x1.6a09e6p-1f;
  const float scaled = x * inv_sqrt_2;
  const float erf_value = erff(scaled);
  const float sum = one + erf_value;
  const float half_x = half * x;
  *result = half_x * sum;
  return isfinite(scaled) && isfinite(erf_value) && isfinite(sum) &&
         isfinite(half_x) && isfinite(*result);
}

static int gelu_backward_one(float x, float upstream, float *result) {
  const float half = 0x1p-1f;
  const float one = 0x1p+0f;
  const float inv_sqrt_2 = 0x1.6a09e6p-1f;
  const float inv_sqrt_2pi = 0x1.988454p-2f;
  const float scaled = x * inv_sqrt_2;
  const float erf_value = erff(scaled);
  const float sum = one + erf_value;
  const float cdf = half * sum;
  const float square = x * x;
  const float exponent = -(half * square);
  const float density = expf(exponent) * inv_sqrt_2pi;
  const float slope = cdf + x * density;
  *result = upstream * slope;
  return isfinite(scaled) && isfinite(erf_value) && isfinite(sum) &&
         isfinite(cdf) && isfinite(square) && isfinite(exponent) &&
         isfinite(density) && isfinite(slope) && isfinite(*result);
}

static int activation_run(const shape3 *s, const float *x,
                          const float *upstream, int gelu, float *output) {
  const size_t count = s->n * s->t * s->d;
  for (size_t index = 0; index < count; index++) {
    float value;
    if (gelu) {
      const int valid = upstream == NULL
                            ? gelu_one(x[index], &value)
                            : gelu_backward_one(x[index], upstream[index], &value);
      if (!valid) return 0;
    } else {
      value = x[index] > 0.0f
                  ? (upstream == NULL ? x[index] : upstream[index])
                  : 0.0f;
      if (!isfinite(value)) return 0;
    }
    if (output != NULL) output[index] = value;
  }
  return 1;
}

static int residual_forward_run(const shape3 *s, const float *x,
                                const float *branch, float *output) {
  const size_t count = s->n * s->t * s->d;
  for (size_t index = 0; index < count; index++) {
    const float value = x[index] + branch[index];
    if (!isfinite(value)) return 0;
    if (output != NULL) output[index] = value;
  }
  return 1;
}

static uint64_t decode_signed_word(int64_t value) {
  if (value >= 0) return (uint64_t)value;
  return UINT64_MAX - (uint64_t)(-(value + INT64_C(1)));
}

static int64_t encode_signed_word(uint64_t value) {
  if (value <= (uint64_t)INT64_MAX) return (int64_t)value;
  return -(int64_t)(UINT64_MAX - value) - INT64_C(1);
}

static int decode_rng_state(const int64_t *words, rng_state *state) {
  if (words[0] != INT64_C(1)) return 0;
  state->key = decode_signed_word(words[1]);
  state->counter_low = decode_signed_word(words[2]);
  state->counter_high = decode_signed_word(words[3]);
  return 1;
}

static void encode_rng_state(const rng_state *state, int64_t *words) {
  words[0] = INT64_C(1);
  words[1] = encode_signed_word(state->key);
  words[2] = encode_signed_word(state->counter_low);
  words[3] = encode_signed_word(state->counter_high);
}

static int advance_counter(const rng_state *state, uint64_t blocks,
                           rng_state *next) {
  next->key = state->key;
  next->counter_low = state->counter_low + blocks;
  next->counter_high = state->counter_high;
  if (next->counter_low < state->counter_low) {
    if (state->counter_high == UINT64_MAX) return 0;
    next->counter_high = state->counter_high + UINT64_C(1);
  }
  return 1;
}

static void philox4x32_10(uint64_t key, uint64_t counter_low,
                          uint64_t counter_high, uint32_t output[4]) {
  uint32_t c0 = (uint32_t)counter_low;
  uint32_t c1 = (uint32_t)(counter_low >> 32);
  uint32_t c2 = (uint32_t)counter_high;
  uint32_t c3 = (uint32_t)(counter_high >> 32);
  uint32_t k0 = (uint32_t)key;
  uint32_t k1 = (uint32_t)(key >> 32);
  for (size_t round = 0; round < 10u; round++) {
    const uint64_t product0 = UINT64_C(0xd2511f53) * (uint64_t)c0;
    const uint64_t product1 = UINT64_C(0xcd9e8d57) * (uint64_t)c2;
    const uint32_t next0 = (uint32_t)(product1 >> 32) ^ c1 ^ k0;
    const uint32_t next1 = (uint32_t)product1;
    const uint32_t next2 = (uint32_t)(product0 >> 32) ^ c3 ^ k1;
    const uint32_t next3 = (uint32_t)product0;
    c0 = next0;
    c1 = next1;
    c2 = next2;
    c3 = next3;
    if (round != 9u) {
      k0 += UINT32_C(0x9e3779b9);
      k1 += UINT32_C(0xbb67ae85);
    }
  }
  output[0] = c0;
  output[1] = c1;
  output[2] = c2;
  output[3] = c3;
}

#if defined(ET_N2_TESTING)
/* Test-only access to the production Philox implementation and signed state
 * decoding.  This symbol is absent from every normal N2 build. */
int32_t et_n2_test_philox4x32_10_v1(const int64_t state_words[4],
                                    uint32_t output[4]) {
  rng_state state = {0u, 0u, 0u};
  if (state_words == NULL || output == NULL ||
      !decode_rng_state(state_words, &state)) {
    return 0;
  }
  philox4x32_10(state.key, state.counter_low, state.counter_high, output);
  return 1;
}
#endif

static int dropout_run(const shape3 *s, const float *input, float probability,
                       const rng_state *state, float *output) {
  const size_t count = s->n * s->t * s->d;
  const float denominator = 1.0f - probability;
  const float scale = 1.0f / denominator;
  uint64_t block_low = state->counter_low;
  uint64_t block_high = state->counter_high;
  uint32_t lanes[4] = {0u, 0u, 0u, 0u};
  if (!isfinite(denominator) || !isfinite(scale)) return 0;
  for (size_t index = 0; index < count; index++) {
    const size_t lane = index % 4u;
    if (lane == 0u) {
      if (index != 0u) {
        block_low += UINT64_C(1);
        if (block_low == 0u) block_high += UINT64_C(1);
      }
      philox4x32_10(state->key, block_low, block_high, lanes);
    }
    {
      const float converted = (float)(lanes[lane] >> 8);
      const float uniform = converted * 0x1p-24f;
      float value = 0.0f;
      if (!isfinite(converted) || !isfinite(uniform)) return 0;
      if (uniform >= probability) {
        value = input[index] * scale;
        if (!isfinite(value)) return 0;
      }
      if (output != NULL) output[index] = value;
    }
  }
  return 1;
}

enum operation_kind {
  OP_EMBEDDING_FORWARD,
  OP_EMBEDDING_BACKWARD,
  OP_LINEAR_FORWARD,
  OP_LINEAR_FORWARD_NO_BIAS,
  OP_LINEAR_BACKWARD,
  OP_LINEAR_BACKWARD_NO_BIAS,
  OP_LAYER_NORM_FORWARD,
  OP_LAYER_NORM_BACKWARD,
  OP_GELU_FORWARD,
  OP_GELU_BACKWARD,
  OP_RELU_FORWARD,
  OP_RELU_BACKWARD,
  OP_RESIDUAL_FORWARD,
  OP_RESIDUAL_BACKWARD,
  OP_DROPOUT_TRAIN_FORWARD,
  OP_DROPOUT_TRAIN_BACKWARD,
  OP_DROPOUT_EVAL_FORWARD,
  OP_DROPOUT_EVAL_BACKWARD,
  OP_UNKNOWN
};

static enum operation_kind operation_kind(const char *operation) {
  if (exact_text(operation, "embedding.forward")) return OP_EMBEDDING_FORWARD;
  if (exact_text(operation, "embedding.backward")) return OP_EMBEDDING_BACKWARD;
  if (exact_text(operation, "linear.forward")) return OP_LINEAR_FORWARD;
  if (exact_text(operation, "linear.forward-no-bias"))
    return OP_LINEAR_FORWARD_NO_BIAS;
  if (exact_text(operation, "linear.backward")) return OP_LINEAR_BACKWARD;
  if (exact_text(operation, "linear.backward-no-bias"))
    return OP_LINEAR_BACKWARD_NO_BIAS;
  if (exact_text(operation, "layer-norm.forward")) return OP_LAYER_NORM_FORWARD;
  if (exact_text(operation, "layer-norm.backward")) return OP_LAYER_NORM_BACKWARD;
  if (exact_text(operation, "gelu.forward")) return OP_GELU_FORWARD;
  if (exact_text(operation, "gelu.backward")) return OP_GELU_BACKWARD;
  if (exact_text(operation, "relu.forward")) return OP_RELU_FORWARD;
  if (exact_text(operation, "relu.backward")) return OP_RELU_BACKWARD;
  if (exact_text(operation, "residual.forward")) return OP_RESIDUAL_FORWARD;
  if (exact_text(operation, "residual.backward")) return OP_RESIDUAL_BACKWARD;
  if (exact_text(operation, "dropout.train.forward"))
    return OP_DROPOUT_TRAIN_FORWARD;
  if (exact_text(operation, "dropout.train.backward"))
    return OP_DROPOUT_TRAIN_BACKWARD;
  if (exact_text(operation, "dropout.eval.forward"))
    return OP_DROPOUT_EVAL_FORWARD;
  if (exact_text(operation, "dropout.eval.backward"))
    return OP_DROPOUT_EVAL_BACKWARD;
  return OP_UNKNOWN;
}

static const char *operation_capability(enum operation_kind kind) {
  switch (kind) {
    case OP_EMBEDDING_FORWARD: return "kernel.embedding-forward";
    case OP_EMBEDDING_BACKWARD: return "kernel.embedding-backward";
    case OP_LINEAR_FORWARD:
    case OP_LINEAR_FORWARD_NO_BIAS:
    case OP_LINEAR_BACKWARD:
    case OP_LINEAR_BACKWARD_NO_BIAS: return "kernel.matmul";
    case OP_LAYER_NORM_FORWARD:
    case OP_LAYER_NORM_BACKWARD: return "kernel.norm";
    case OP_GELU_FORWARD:
    case OP_GELU_BACKWARD:
    case OP_RELU_FORWARD:
    case OP_RELU_BACKWARD: return "kernel.activation";
    case OP_RESIDUAL_FORWARD:
    case OP_RESIDUAL_BACKWARD: return "kernel.residual";
    case OP_DROPOUT_TRAIN_FORWARD:
    case OP_DROPOUT_TRAIN_BACKWARD:
    case OP_DROPOUT_EVAL_FORWARD:
    case OP_DROPOUT_EVAL_BACKWARD: return "kernel.dropout";
    default: return NULL;
  }
}

static int request_row(enum operation_kind kind,
                       const et_kernel_request_v1 *request, size_t *dimensions) {
  static const size_t embedding_rows[][4] = {
      {1u, 1u, 1u, 1u}, {1u, 4u, 3u, 2u}, {2u, 3u, 5u, 6u}};
  static const size_t linear_rows[][4] = {
      {1u, 1u, 1u, 1u}, {1u, 2u, 3u, 4u}, {2u, 3u, 4u, 5u}};
  static const size_t norm_rows[][3] = {
      {1u, 1u, 1u}, {1u, 1u, 3u}, {1u, 2u, 4u}, {2u, 2u, 4u}};
  static const size_t activation_rows[][3] = {
      {1u, 1u, 1u}, {1u, 1u, 8u}, {1u, 2u, 5u}, {2u, 2u, 4u}};
  static const size_t residual_rows[][3] = {
      {1u, 1u, 1u}, {1u, 3u, 4u}, {2u, 2u, 4u}};
  static const size_t dropout_rows[][3] = {
      {1u, 1u, 1u}, {1u, 1u, 2u}, {1u, 1u, 3u},
      {1u, 1u, 4u}, {1u, 2u, 5u}};
  switch (kind) {
    case OP_EMBEDDING_FORWARD:
    case OP_EMBEDDING_BACKWARD:
      return request_shape(request, 4u, dimensions) &&
             row4_admitted(dimensions, embedding_rows,
                           ET_ARRAY_COUNT(embedding_rows));
    case OP_LINEAR_FORWARD:
    case OP_LINEAR_FORWARD_NO_BIAS:
    case OP_LINEAR_BACKWARD:
    case OP_LINEAR_BACKWARD_NO_BIAS:
      return request_shape(request, 4u, dimensions) &&
             row4_admitted(dimensions, linear_rows,
                           ET_ARRAY_COUNT(linear_rows));
    case OP_LAYER_NORM_FORWARD:
    case OP_LAYER_NORM_BACKWARD:
      return request_shape(request, 3u, dimensions) &&
             row3_admitted(dimensions, norm_rows, ET_ARRAY_COUNT(norm_rows));
    case OP_GELU_FORWARD:
    case OP_GELU_BACKWARD:
    case OP_RELU_FORWARD:
    case OP_RELU_BACKWARD:
      return request_shape(request, 3u, dimensions) &&
             row3_admitted(dimensions, activation_rows,
                           ET_ARRAY_COUNT(activation_rows));
    case OP_RESIDUAL_FORWARD:
    case OP_RESIDUAL_BACKWARD:
      return request_shape(request, 3u, dimensions) &&
             row3_admitted(dimensions, residual_rows,
                           ET_ARRAY_COUNT(residual_rows));
    case OP_DROPOUT_TRAIN_FORWARD:
    case OP_DROPOUT_TRAIN_BACKWARD:
    case OP_DROPOUT_EVAL_FORWARD:
    case OP_DROPOUT_EVAL_BACKWARD:
      return request_shape(request, 3u, dimensions) &&
             row3_admitted(dimensions, dropout_rows,
                           ET_ARRAY_COUNT(dropout_rows));
    default: return 0;
  }
}

static int32_t validate_common_request(const et_kernel_call_v1 *call,
                                       enum operation_kind kind,
                                       size_t *dimensions,
                                       et_kernel_error *error) {
  const char *operation = call != NULL && call->request != NULL &&
                                  call->request->operation != NULL
                              ? call->request->operation
                              : "n2.provider";
  const char *capability = operation_capability(kind);
  if (call == NULL || call->request == NULL ||
      call->request->operation == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "n2.provider",
                     "call, request, and operation are required");
  }
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      call->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                     "call or request truncates the K1 v1 prefix");
  }
  if (kind == OP_UNKNOWN || !exact_text(call->capability, capability) ||
      !exact_text(call->request->dtype, "f32") ||
      !exact_text(call->request->device, "cpu") ||
      !request_row(kind, call->request, dimensions)) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "request is outside N2 v1 evidence");
  }
  if (!fp_environment_supported()) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "floating-point control environment is unsupported");
  }
  return 0;
}

static int32_t restore_preflight_environment(const fenv_t *environment,
                                             const char *operation,
                                             et_kernel_error *error) {
  if (fesetenv(environment) != 0) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "floating-point environment could not be restored");
  }
  return 0;
}

static int32_t begin_preflight(fenv_t *environment, const char *operation,
                               et_kernel_error *error) {
  if (fegetenv(environment) != 0) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "floating-point environment cannot be observed");
  }
  return 0;
}

static int32_t preflight_result(int valid, const fenv_t *environment,
                                const char *operation, et_kernel_error *error) {
  int32_t result =
      restore_preflight_environment(environment, operation, error);
  if (result != 0) return result;
  if (!valid) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "operation would produce a nonfinite result");
  }
  et_kernel_error_clear(error);
  return 0;
}

static int32_t validate_embedding(const et_kernel_call_v1 *call,
                                  enum operation_kind kind, const size_t *d,
                                  et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int backward = kind == OP_EMBEDDING_BACKWARD;
  const et_kernel_tensor_view_v1 *inputs[2];
  const et_kernel_tensor_view_v1 *outputs[1];
  const size_t ids_shape[2] = {d[0], d[1]};
  const size_t weight_shape[2] = {d[2], d[3]};
  const size_t value_shape[3] = {d[0], d[1], d[3]};
  const shape4 s = {d[0], d[1], d[2], d[3]};
  fenv_t status;
  int32_t result = table_views(call, 0, 2u, operation, inputs, error);
  if (result == 0) result = table_views(call, 1, 1u, operation, outputs, error);
  if (result == 0)
    result = validate_view(inputs[0], "i64", sizeof(int64_t),
                           _Alignof(int64_t), 2u, ids_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0 && !backward)
    result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float),
                           2u, weight_shape, operation, error);
  if (result == 0)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           2u + (backward ? 0u : 1u),
                           backward ? weight_shape : value_shape, operation,
                           error);
  if (result == 0)
    result = validate_input_aliases(inputs, 2u, 0, operation, error);
  if (result != 0) return result;
  for (size_t index = 0; index < d[0] * d[1]; index++) {
    const int64_t id = ((const int64_t *)inputs[0]->data)[index];
    if (id < 0 || (uint64_t)id >= (uint64_t)d[2]) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "embedding ID is outside the signed exact vocabulary range");
    }
  }
  if (!finite_values((const float *)inputs[1]->data,
                     inputs[1]->byte_length / sizeof(float))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "embedding floating input is nonfinite");
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      backward ? embedding_backward_run(&s, (const int64_t *)inputs[0]->data,
                                        (const float *)inputs[1]->data, NULL)
               : embedding_forward_run(&s, (const int64_t *)inputs[0]->data,
                                       (const float *)inputs[1]->data, NULL),
      &status, operation, error);
}

static int32_t validate_linear(const et_kernel_call_v1 *call,
                               enum operation_kind kind, const size_t *d,
                               et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int backward = kind == OP_LINEAR_BACKWARD ||
                       kind == OP_LINEAR_BACKWARD_NO_BIAS;
  const int bias = kind == OP_LINEAR_FORWARD || kind == OP_LINEAR_BACKWARD;
  const size_t input_count = backward ? 3u : (bias ? 3u : 2u);
  const size_t output_count = backward ? (bias ? 3u : 2u) : 1u;
  const et_kernel_tensor_view_v1 *inputs[3];
  const et_kernel_tensor_view_v1 *outputs[3];
  const size_t x_shape[3] = {d[0], d[1], d[2]};
  const size_t weight_shape[2] = {d[3], d[2]};
  const size_t bias_shape[1] = {d[3]};
  const size_t y_shape[3] = {d[0], d[1], d[3]};
  const shape4 s = {d[0], d[1], d[2], d[3]};
  fenv_t status;
  int32_t result = table_views(call, 0, input_count, operation, inputs, error);
  if (result == 0)
    result = table_views(call, 1, output_count, operation, outputs, error);
  if (result == 0)
    result = validate_view(inputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, x_shape, operation, error);
  if (result == 0)
    result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float),
                           2u, weight_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(inputs[2], "f32", sizeof(float), _Alignof(float),
                           3u, y_shape, operation, error);
  if (result == 0 && !backward && bias)
    result = validate_view(inputs[2], "f32", sizeof(float), _Alignof(float),
                           1u, bias_shape, operation, error);
  if (result == 0 && !backward)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, y_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, x_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(outputs[1], "f32", sizeof(float), _Alignof(float),
                           2u, weight_shape, operation, error);
  if (result == 0 && backward && bias)
    result = validate_view(outputs[2], "f32", sizeof(float), _Alignof(float),
                           1u, bias_shape, operation, error);
  if (result == 0)
    result = validate_input_aliases(inputs, input_count, 0, operation, error);
  if (result != 0) return result;
  for (size_t index = 0; index < input_count; index++) {
    if (!finite_values((const float *)inputs[index]->data,
                       inputs[index]->byte_length / sizeof(float))) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "linear floating input is nonfinite");
    }
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      backward
          ? linear_backward_run(&s, (const float *)inputs[0]->data,
                                (const float *)inputs[1]->data,
                                (const float *)inputs[2]->data, bias, NULL,
                                NULL, NULL)
          : linear_forward_run(&s, (const float *)inputs[0]->data,
                               (const float *)inputs[1]->data,
                               bias ? (const float *)inputs[2]->data : NULL,
                               NULL),
      &status, operation, error);
}

static int32_t validate_layer_norm(const et_kernel_call_v1 *call,
                                   enum operation_kind kind, const size_t *d,
                                   et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int backward = kind == OP_LAYER_NORM_BACKWARD;
  const et_kernel_tensor_view_v1 *inputs[4];
  const et_kernel_tensor_view_v1 *outputs[3];
  const size_t value_shape[3] = {d[0], d[1], d[2]};
  const size_t affine_shape[1] = {d[2]};
  const shape3 s = {d[0], d[1], d[2]};
  const size_t input_count = 4u;
  const size_t output_count = backward ? 3u : 1u;
  fenv_t status;
  float epsilon;
  int32_t result = table_views(call, 0, input_count, operation, inputs, error);
  if (result == 0)
    result = table_views(call, 1, output_count, operation, outputs, error);
  if (result == 0)
    result = validate_view(inputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0)
    result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float),
                           1u, affine_shape, operation, error);
  if (result == 0 && !backward)
    result = validate_view(inputs[2], "f32", sizeof(float), _Alignof(float),
                           1u, affine_shape, operation, error);
  if (result == 0)
    result = validate_view(inputs[backward ? 2u : 3u], "f32", sizeof(float),
                           _Alignof(float), 0u, NULL, operation, error);
  if (result == 0 && backward)
    result = validate_view(inputs[3], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(outputs[1], "f32", sizeof(float), _Alignof(float),
                           1u, affine_shape, operation, error);
  if (result == 0 && backward)
    result = validate_view(outputs[2], "f32", sizeof(float), _Alignof(float),
                           1u, affine_shape, operation, error);
  if (result == 0)
    result = validate_input_aliases(inputs, input_count, 0, operation, error);
  if (result != 0) return result;
  for (size_t index = 0; index < input_count; index++) {
    if (!finite_values((const float *)inputs[index]->data,
                       inputs[index]->byte_length / sizeof(float))) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "LayerNorm floating input is nonfinite");
    }
  }
  epsilon = *(const float *)inputs[backward ? 2u : 3u]->data;
  if (!(epsilon > 0.0f)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "LayerNorm epsilon must be finite and greater than zero");
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      backward
          ? layer_norm_backward_run(
                &s, (const float *)inputs[0]->data,
                (const float *)inputs[1]->data, epsilon,
                (const float *)inputs[3]->data, NULL, NULL, NULL)
          : layer_norm_forward_run(
                &s, (const float *)inputs[0]->data,
                (const float *)inputs[1]->data,
                (const float *)inputs[2]->data, epsilon, NULL),
      &status, operation, error);
}

static int32_t validate_activation(const et_kernel_call_v1 *call,
                                   enum operation_kind kind, const size_t *d,
                                   et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int backward = kind == OP_GELU_BACKWARD || kind == OP_RELU_BACKWARD;
  const int gelu = kind == OP_GELU_FORWARD || kind == OP_GELU_BACKWARD;
  const size_t count = backward ? 2u : 1u;
  const et_kernel_tensor_view_v1 *inputs[2];
  const et_kernel_tensor_view_v1 *outputs[1];
  const size_t value_shape[3] = {d[0], d[1], d[2]};
  const shape3 s = {d[0], d[1], d[2]};
  fenv_t status;
  int32_t result = table_views(call, 0, count, operation, inputs, error);
  if (result == 0) result = table_views(call, 1, 1u, operation, outputs, error);
  for (size_t index = 0; result == 0 && index < count; index++) {
    result = validate_view(inputs[index], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  }
  if (result == 0)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0)
    result = validate_input_aliases(inputs, count, 0, operation, error);
  if (result != 0) return result;
  for (size_t index = 0; index < count; index++) {
    if (!finite_values((const float *)inputs[index]->data,
                       inputs[index]->byte_length / sizeof(float))) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "activation floating input is nonfinite");
    }
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      activation_run(&s, (const float *)inputs[0]->data,
                     backward ? (const float *)inputs[1]->data : NULL, gelu,
                     NULL),
      &status, operation, error);
}

static int32_t validate_residual(const et_kernel_call_v1 *call,
                                 enum operation_kind kind, const size_t *d,
                                 et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int backward = kind == OP_RESIDUAL_BACKWARD;
  const size_t input_count = backward ? 1u : 2u;
  const size_t output_count = backward ? 2u : 1u;
  const et_kernel_tensor_view_v1 *inputs[2];
  const et_kernel_tensor_view_v1 *outputs[2];
  const size_t value_shape[3] = {d[0], d[1], d[2]};
  const shape3 s = {d[0], d[1], d[2]};
  fenv_t status;
  int32_t result = table_views(call, 0, input_count, operation, inputs, error);
  if (result == 0)
    result = table_views(call, 1, output_count, operation, outputs, error);
  for (size_t index = 0; result == 0 && index < input_count; index++) {
    result = validate_view(inputs[index], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  }
  for (size_t index = 0; result == 0 && index < output_count; index++) {
    result = validate_view(outputs[index], "f32", sizeof(float),
                           _Alignof(float), 3u, value_shape, operation, error);
  }
  if (result == 0)
    result = validate_input_aliases(inputs, input_count, !backward, operation,
                                    error);
  if (result != 0) return result;
  for (size_t index = 0; index < input_count; index++) {
    if (!finite_values((const float *)inputs[index]->data,
                       inputs[index]->byte_length / sizeof(float))) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "residual floating input is nonfinite");
    }
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      backward || residual_forward_run(&s, (const float *)inputs[0]->data,
                                       (const float *)inputs[1]->data, NULL),
      &status, operation, error);
}

static int32_t validate_dropout(const et_kernel_call_v1 *call,
                                enum operation_kind kind, const size_t *d,
                                et_kernel_error *error) {
  const char *operation = call->request->operation;
  const int train = kind == OP_DROPOUT_TRAIN_FORWARD ||
                    kind == OP_DROPOUT_TRAIN_BACKWARD;
  const size_t input_count = train ? 3u : 1u;
  const size_t output_count = kind == OP_DROPOUT_TRAIN_FORWARD ? 2u : 1u;
  const et_kernel_tensor_view_v1 *inputs[3];
  const et_kernel_tensor_view_v1 *outputs[2];
  const size_t value_shape[3] = {d[0], d[1], d[2]};
  const size_t state_shape[1] = {4u};
  const shape3 s = {d[0], d[1], d[2]};
  const size_t elements = d[0] * d[1] * d[2];
  fenv_t status;
  rng_state state = {0u, 0u, 0u};
  rng_state next = {0u, 0u, 0u};
  float probability = 0.0f;
  int32_t result = table_views(call, 0, input_count, operation, inputs, error);
  if (result == 0)
    result = table_views(call, 1, output_count, operation, outputs, error);
  if (result == 0)
    result = validate_view(inputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0 && train)
    result = validate_view(inputs[1], "f32", sizeof(float), _Alignof(float),
                           0u, NULL, operation, error);
  if (result == 0 && train)
    result = validate_view(inputs[2], "i64", sizeof(int64_t),
                           _Alignof(int64_t), 1u, state_shape, operation, error);
  if (result == 0)
    result = validate_view(outputs[0], "f32", sizeof(float), _Alignof(float),
                           3u, value_shape, operation, error);
  if (result == 0 && kind == OP_DROPOUT_TRAIN_FORWARD)
    result = validate_view(outputs[1], "i64", sizeof(int64_t),
                           _Alignof(int64_t), 1u, state_shape, operation, error);
  if (result == 0)
    result = validate_input_aliases(inputs, input_count, 0, operation, error);
  if (result != 0) return result;
  if (!finite_values((const float *)inputs[0]->data, elements)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                     "dropout floating input is nonfinite");
  }
  if (train) {
    const int64_t *words = (const int64_t *)inputs[2]->data;
    if (words[0] != INT64_C(1)) {
      return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "dropout RNG state version is unsupported");
    }
    (void)decode_rng_state(words, &state);
    probability = *(const float *)inputs[1]->data;
    if (!isfinite(probability) || probability < 0.0f ||
        !(probability < 1.0f)) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                       "dropout probability must be finite and in [0,1)");
    }
    if (probability != 0.0f) {
      const uint64_t blocks = (uint64_t)(elements / 4u) +
                              (uint64_t)(elements % 4u != 0u);
      if ((state.counter_low == UINT64_MAX &&
           state.counter_high == UINT64_MAX) ||
          !advance_counter(&state, blocks, &next)) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                         "dropout RNG counter is exhausted");
      }
    }
  }
  result = begin_preflight(&status, operation, error);
  if (result != 0) return result;
  return preflight_result(
      !train || probability == 0.0f ||
          dropout_run(&s, (const float *)inputs[0]->data, probability, &state,
                      NULL),
      &status, operation, error);
}

static void invoke_embedding(const et_kernel_call_v1 *call,
                             enum operation_kind kind) {
  const uint64_t *d = call->request->shape;
  const shape4 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2], (size_t)d[3]};
  if (kind == OP_EMBEDDING_FORWARD) {
    (void)embedding_forward_run(
        &s, (const int64_t *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        (float *)output_at(call, 0u)->data);
  } else {
    (void)embedding_backward_run(
        &s, (const int64_t *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        (float *)output_at(call, 0u)->data);
  }
}

static void invoke_linear(const et_kernel_call_v1 *call,
                          enum operation_kind kind) {
  const uint64_t *d = call->request->shape;
  const shape4 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2], (size_t)d[3]};
  const int backward = kind == OP_LINEAR_BACKWARD ||
                       kind == OP_LINEAR_BACKWARD_NO_BIAS;
  const int bias = kind == OP_LINEAR_FORWARD || kind == OP_LINEAR_BACKWARD;
  if (backward) {
    (void)linear_backward_run(
        &s, (const float *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        (const float *)input_at(call, 2u)->data, bias,
        (float *)output_at(call, 0u)->data,
        (float *)output_at(call, 1u)->data,
        bias ? (float *)output_at(call, 2u)->data : NULL);
  } else {
    (void)linear_forward_run(
        &s, (const float *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        bias ? (const float *)input_at(call, 2u)->data : NULL,
        (float *)output_at(call, 0u)->data);
  }
}

static void invoke_layer_norm(const et_kernel_call_v1 *call,
                              enum operation_kind kind) {
  const uint64_t *d = call->request->shape;
  const shape3 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2]};
  if (kind == OP_LAYER_NORM_FORWARD) {
    (void)layer_norm_forward_run(
        &s, (const float *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        (const float *)input_at(call, 2u)->data,
        *(const float *)input_at(call, 3u)->data,
        (float *)output_at(call, 0u)->data);
  } else {
    (void)layer_norm_backward_run(
        &s, (const float *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        *(const float *)input_at(call, 2u)->data,
        (const float *)input_at(call, 3u)->data,
        (float *)output_at(call, 0u)->data,
        (float *)output_at(call, 1u)->data,
        (float *)output_at(call, 2u)->data);
  }
}

static void invoke_activation(const et_kernel_call_v1 *call,
                              enum operation_kind kind) {
  const uint64_t *d = call->request->shape;
  const shape3 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2]};
  const int backward = kind == OP_GELU_BACKWARD || kind == OP_RELU_BACKWARD;
  const int gelu = kind == OP_GELU_FORWARD || kind == OP_GELU_BACKWARD;
  (void)activation_run(
      &s, (const float *)input_at(call, 0u)->data,
      backward ? (const float *)input_at(call, 1u)->data : NULL, gelu,
      (float *)output_at(call, 0u)->data);
}

static void invoke_residual(const et_kernel_call_v1 *call,
                            enum operation_kind kind) {
  if (kind == OP_RESIDUAL_FORWARD) {
    const uint64_t *d = call->request->shape;
    const shape3 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2]};
    (void)residual_forward_run(
        &s, (const float *)input_at(call, 0u)->data,
        (const float *)input_at(call, 1u)->data,
        (float *)output_at(call, 0u)->data);
  } else {
    const size_t bytes = input_at(call, 0u)->byte_length;
    memcpy(output_at(call, 0u)->data, input_at(call, 0u)->data, bytes);
    memcpy(output_at(call, 1u)->data, input_at(call, 0u)->data, bytes);
  }
}

static void invoke_dropout(const et_kernel_call_v1 *call,
                           enum operation_kind kind) {
  const size_t bytes = input_at(call, 0u)->byte_length;
  if (kind == OP_DROPOUT_EVAL_FORWARD ||
      kind == OP_DROPOUT_EVAL_BACKWARD) {
    memcpy(output_at(call, 0u)->data, input_at(call, 0u)->data, bytes);
    return;
  }
  {
    const uint64_t *d = call->request->shape;
    const shape3 s = {(size_t)d[0], (size_t)d[1], (size_t)d[2]};
    const float probability = *(const float *)input_at(call, 1u)->data;
    const int64_t *state_words = (const int64_t *)input_at(call, 2u)->data;
    rng_state state = {0u, 0u, 0u};
    (void)decode_rng_state(state_words, &state);
    if (probability == 0.0f) {
      memcpy(output_at(call, 0u)->data, input_at(call, 0u)->data, bytes);
      if (kind == OP_DROPOUT_TRAIN_FORWARD) {
        memcpy(output_at(call, 1u)->data, state_words, 4u * sizeof(int64_t));
      }
      return;
    }
    (void)dropout_run(&s, (const float *)input_at(call, 0u)->data,
                      probability, &state,
                      (float *)output_at(call, 0u)->data);
    if (kind == OP_DROPOUT_TRAIN_FORWARD) {
      const size_t elements = s.n * s.t * s.d;
      const uint64_t blocks = (uint64_t)(elements / 4u) +
                              (uint64_t)(elements % 4u != 0u);
      rng_state next = {0u, 0u, 0u};
      (void)advance_counter(&state, blocks, &next);
      encode_rng_state(&next, (int64_t *)output_at(call, 1u)->data);
    }
  }
}

static int32_t provider_validate_call_inner(const et_kernel_call_v1 *call,
                                            et_kernel_error *error) {
  size_t dimensions[4] = {0u, 0u, 0u, 0u};
  enum operation_kind kind = OP_UNKNOWN;
  int32_t result;
  if (call != NULL && call->request != NULL)
    kind = operation_kind(call->request->operation);
  result = validate_common_request(call, kind, dimensions, error);
  if (result != 0) return result;
  switch (kind) {
    case OP_EMBEDDING_FORWARD:
    case OP_EMBEDDING_BACKWARD:
      return validate_embedding(call, kind, dimensions, error);
    case OP_LINEAR_FORWARD:
    case OP_LINEAR_FORWARD_NO_BIAS:
    case OP_LINEAR_BACKWARD:
    case OP_LINEAR_BACKWARD_NO_BIAS:
      return validate_linear(call, kind, dimensions, error);
    case OP_LAYER_NORM_FORWARD:
    case OP_LAYER_NORM_BACKWARD:
      return validate_layer_norm(call, kind, dimensions, error);
    case OP_GELU_FORWARD:
    case OP_GELU_BACKWARD:
    case OP_RELU_FORWARD:
    case OP_RELU_BACKWARD:
      return validate_activation(call, kind, dimensions, error);
    case OP_RESIDUAL_FORWARD:
    case OP_RESIDUAL_BACKWARD:
      return validate_residual(call, kind, dimensions, error);
    case OP_DROPOUT_TRAIN_FORWARD:
    case OP_DROPOUT_TRAIN_BACKWARD:
    case OP_DROPOUT_EVAL_FORWARD:
    case OP_DROPOUT_EVAL_BACKWARD:
      return validate_dropout(call, kind, dimensions, error);
    default:
      return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, "n2.provider",
                       "operation is outside N2 v1 evidence");
  }
}

static int32_t provider_validate_call(const et_kernel_call_v1 *call,
                                      et_kernel_error *error) {
  fenv_t environment;
  int32_t result;
  if (fegetenv(&environment) != 0) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "n2.provider",
                     "floating-point environment cannot be observed");
  }
  result = provider_validate_call_inner(call, error);
  if (fesetenv(&environment) != 0) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_PROVIDER_REJECTED, "n2.provider",
                     "floating-point environment could not be restored");
  }
  return result;
}

static void provider_invoke_call(const et_kernel_call_v1 *call) {
  const enum operation_kind kind = operation_kind(call->request->operation);
  switch (kind) {
    case OP_EMBEDDING_FORWARD:
    case OP_EMBEDDING_BACKWARD: invoke_embedding(call, kind); break;
    case OP_LINEAR_FORWARD:
    case OP_LINEAR_FORWARD_NO_BIAS:
    case OP_LINEAR_BACKWARD:
    case OP_LINEAR_BACKWARD_NO_BIAS: invoke_linear(call, kind); break;
    case OP_LAYER_NORM_FORWARD:
    case OP_LAYER_NORM_BACKWARD: invoke_layer_norm(call, kind); break;
    case OP_GELU_FORWARD:
    case OP_GELU_BACKWARD:
    case OP_RELU_FORWARD:
    case OP_RELU_BACKWARD: invoke_activation(call, kind); break;
    case OP_RESIDUAL_FORWARD:
    case OP_RESIDUAL_BACKWARD: invoke_residual(call, kind); break;
    case OP_DROPOUT_TRAIN_FORWARD:
    case OP_DROPOUT_TRAIN_BACKWARD:
    case OP_DROPOUT_EVAL_FORWARD:
    case OP_DROPOUT_EVAL_BACKWARD: invoke_dropout(call, kind); break;
    default: break;
  }
}

#define EXACT_DIMENSION(value)                                                  \
  {.minimum = (value), .maximum = (value), .maximum_unbounded = 0u,            \
   .reserved = {0}}

#define DEFINE_ROW4(name, a, b, c, d)                                          \
  static const et_kernel_dimension_range_v1 name##_dimensions[] = {            \
      EXACT_DIMENSION(a), EXACT_DIMENSION(b), EXACT_DIMENSION(c),               \
      EXACT_DIMENSION(d)}

#define DEFINE_ROW3(name, a, b, c)                                             \
  static const et_kernel_dimension_range_v1 name##_dimensions[] = {            \
      EXACT_DIMENSION(a), EXACT_DIMENSION(b), EXACT_DIMENSION(c)}

DEFINE_ROW4(embedding_1111, 1u, 1u, 1u, 1u);
DEFINE_ROW4(embedding_1432, 1u, 4u, 3u, 2u);
DEFINE_ROW4(embedding_2356, 2u, 3u, 5u, 6u);
DEFINE_ROW4(linear_1111, 1u, 1u, 1u, 1u);
DEFINE_ROW4(linear_1234, 1u, 2u, 3u, 4u);
DEFINE_ROW4(linear_2345, 2u, 3u, 4u, 5u);
DEFINE_ROW3(norm_111, 1u, 1u, 1u);
DEFINE_ROW3(norm_113, 1u, 1u, 3u);
DEFINE_ROW3(norm_124, 1u, 2u, 4u);
DEFINE_ROW3(norm_224, 2u, 2u, 4u);
DEFINE_ROW3(activation_111, 1u, 1u, 1u);
DEFINE_ROW3(activation_118, 1u, 1u, 8u);
DEFINE_ROW3(activation_125, 1u, 2u, 5u);
DEFINE_ROW3(activation_224, 2u, 2u, 4u);
DEFINE_ROW3(dropout_111, 1u, 1u, 1u);
DEFINE_ROW3(dropout_112, 1u, 1u, 2u);
DEFINE_ROW3(dropout_113, 1u, 1u, 3u);
DEFINE_ROW3(dropout_114, 1u, 1u, 4u);
DEFINE_ROW3(dropout_125, 1u, 2u, 5u);
DEFINE_ROW3(residual_111, 1u, 1u, 1u);
DEFINE_ROW3(residual_134, 1u, 3u, 4u);
DEFINE_ROW3(residual_224, 2u, 2u, 4u);

#undef DEFINE_ROW3
#undef DEFINE_ROW4
#undef EXACT_DIMENSION

static const et_kernel_shape_range_v1 embedding_ranges[] = {
    {.rank = 4u, .dimensions = embedding_1111_dimensions},
    {.rank = 4u, .dimensions = embedding_1432_dimensions},
    {.rank = 4u, .dimensions = embedding_2356_dimensions}};
static const et_kernel_shape_range_v1 linear_ranges[] = {
    {.rank = 4u, .dimensions = linear_1111_dimensions},
    {.rank = 4u, .dimensions = linear_1234_dimensions},
    {.rank = 4u, .dimensions = linear_2345_dimensions}};
static const et_kernel_shape_range_v1 norm_ranges[] = {
    {.rank = 3u, .dimensions = norm_111_dimensions},
    {.rank = 3u, .dimensions = norm_113_dimensions},
    {.rank = 3u, .dimensions = norm_124_dimensions},
    {.rank = 3u, .dimensions = norm_224_dimensions}};
static const et_kernel_shape_range_v1 activation_ranges[] = {
    {.rank = 3u, .dimensions = activation_111_dimensions},
    {.rank = 3u, .dimensions = activation_118_dimensions},
    {.rank = 3u, .dimensions = activation_125_dimensions},
    {.rank = 3u, .dimensions = activation_224_dimensions}};
static const et_kernel_shape_range_v1 dropout_ranges[] = {
    {.rank = 3u, .dimensions = dropout_111_dimensions},
    {.rank = 3u, .dimensions = dropout_112_dimensions},
    {.rank = 3u, .dimensions = dropout_113_dimensions},
    {.rank = 3u, .dimensions = dropout_114_dimensions},
    {.rank = 3u, .dimensions = dropout_125_dimensions}};
static const et_kernel_shape_range_v1 residual_ranges[] = {
    {.rank = 3u, .dimensions = residual_111_dimensions},
    {.rank = 3u, .dimensions = residual_134_dimensions},
    {.rank = 3u, .dimensions = residual_224_dimensions}};

static const char *const embedding_forward_operations[] = {
    "embedding.forward"};
static const char *const embedding_backward_operations[] = {
    "embedding.backward"};
static const char *const linear_operations[] = {
    "linear.backward", "linear.backward-no-bias", "linear.forward",
    "linear.forward-no-bias"};
static const char *const norm_operations[] = {"layer-norm.backward",
                                               "layer-norm.forward"};
static const char *const activation_operations[] = {
    "gelu.backward", "gelu.forward", "relu.backward", "relu.forward"};
static const char *const dropout_operations[] = {
    "dropout.eval.backward", "dropout.eval.forward", "dropout.train.backward",
    "dropout.train.forward"};
static const char *const residual_operations[] = {"residual.backward",
                                                   "residual.forward"};
static const char *const provider_dtypes[] = {"f32"};
static const char *const provider_devices[] = {"cpu"};

#define CAPABILITY(name_value, operations_value, ranges_value)                  \
  {.struct_size = sizeof(et_kernel_capability_v1),                             \
   .name = (name_value),                                                        \
   .status = ET_KERNEL_CAPABILITY_VERIFIED,                                     \
   .implementation = "n2.cpu-f32.serial",                                     \
   .version = "1.0",                                                           \
   .evidence = "N2:cpu-f32-primitives-v1",                                    \
   .deterministic = 1u,                                                         \
   .reserved = {0},                                                             \
   .operation_count = ET_ARRAY_COUNT(operations_value),                         \
   .operations = (operations_value),                                            \
   .dtype_count = ET_ARRAY_COUNT(provider_dtypes),                              \
   .dtypes = provider_dtypes,                                                    \
   .device_count = ET_ARRAY_COUNT(provider_devices),                            \
   .devices = provider_devices,                                                  \
   .shape_range_count = ET_ARRAY_COUNT(ranges_value),                           \
   .shape_ranges = (ranges_value)}

static const et_kernel_capability_v1 provider_capabilities[] = {
    CAPABILITY("kernel.activation", activation_operations, activation_ranges),
    CAPABILITY("kernel.dropout", dropout_operations, dropout_ranges),
    CAPABILITY("kernel.embedding-backward", embedding_backward_operations,
               embedding_ranges),
    CAPABILITY("kernel.embedding-forward", embedding_forward_operations,
               embedding_ranges),
    CAPABILITY("kernel.matmul", linear_operations, linear_ranges),
    CAPABILITY("kernel.norm", norm_operations, norm_ranges),
    CAPABILITY("kernel.residual", residual_operations, residual_ranges)};

#undef CAPABILITY

static const et_kernel_provider_v1 provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = ET_KERNEL_ABI_MINOR,
    .required_features = 0u,
    .name = "n2.cpu-f32.serial",
    .version = "1.0",
    .evidence = "N2:cpu-f32-primitives-v1",
    .capability_count = ET_ARRAY_COUNT(provider_capabilities),
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(provider_capabilities),
    .capabilities = provider_capabilities,
    .validate_call = provider_validate_call,
    .invoke_call = provider_invoke_call};

const et_kernel_provider_v1 *et_n2_kernel_provider_v1(void) {
  return &provider;
}
