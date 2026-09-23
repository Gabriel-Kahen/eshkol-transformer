#include "eshkol_transformer/g3c4_primitives_abi.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if !defined(__x86_64__)
#error "G3-C4 v1 requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4, "G3-C4 requires eight-bit bytes/binary32");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
               FLT_EVAL_METHOD == 0, "G3-C4 requires standard IEEE binary32 evaluation");
typedef struct shape4 { size_t n, t, a, b; } shape4;
typedef struct shape3 { size_t n, t, d; } shape3;
typedef struct attention_shape { size_t n, hq, hkv, tq, tk, dh; } attention_shape;

/* Forward helpers copied from the accepted source baseline ba0e37d:
 * g3n_primitives_provider.c sha256 cdc7a0416ea8f519c135aeb79b4c74f4660b2c116684dc24ffb7a975c4b7d195
 * supplies the checked envelope and forward helpers; n2_primitives_provider.c
 * sha256 81a213545f2e9256eedf57455b0ea33f56b05d908f1165813da9a2f5077137c1
 * supplies gelu_one. Adaptations are the accepted G3-C4 names, metadata,
 * operand schemas and rows; dynamic T/Tq/Tk; complete multirow ID, position
 * and mask checks; GELU dispatch; residual count; and two explicit finite
 * score-minus-maximum checks. No predecessor source or accessor is linked. */
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

static int32_t table_views(const et_kernel_call_v1 *call, int output,
                           size_t expected, const char *operation,
                           const et_kernel_tensor_view_v1 **views,
                           et_kernel_error *error) {
  const size_t count = output ? call->output_count : call->input_count;
  const size_t stride = output ? call->output_stride : call->input_stride;
  const size_t bytes = output ? call->output_bytes : call->input_bytes;
  const void *base = output ? call->outputs : call->inputs;
  if (count != expected) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     output ? "output count does not match operation schema"
                            : "input count does not match operation schema");
  }
  if (base == NULL) return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT, operation, "tensor table is required");
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
                     "G3-C4 v1 accepts only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "G3-C4 v1 requires dense zero-offset tensor views");
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
      !aligned_pointer(view->data, alignment)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor data alignment or byte span is invalid");
  }
  if (!pointer_span_fits(view->data, expected_bytes))
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_PROVIDER_REJECTED, operation, "tensor data address wraps");
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

static int finite_values(const float *values, size_t count) {
  for (size_t index = 0; index < count; index++) {
    if (!isfinite(values[index])) return 0;
  }
  return 1;
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

static size_t index3(const shape3 *s, size_t n, size_t t, size_t d) {
  return (n * s->t + t) * s->d + d;
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

static int gelu_forward_run(const shape3 *s, const float *x, float *output) {
  const size_t count = s->n * s->t * s->d;
  for (size_t index = 0; index < count; index++) {
    float value;
    if (!gelu_one(x[index], &value)) return 0;
    if (output != NULL) output[index] = value;
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

static int attention_admitted(const attention_shape *s, const int64_t *query_positions,
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
    if (!attention_admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
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
    if (!attention_admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
      continue;
    }
    if (!attention_score(s, q, k, scale, n, hq, tq, tk, &score)) {
      return -1;
    }
    const float shifted = score - row_max;
    if (!isfinite(shifted)) return -1;
    weight = expf(shifted);
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
  const float shifted = score - maximum;
  if (!isfinite(shifted)) return 0;
  *result = expf(shifted) / denominator;
  return isfinite(*result);
}

static int attention_forward_run(
    const attention_shape *s, const float *q, const float *k, const float *v,
    const int64_t *query_positions, const int64_t *key_positions,
    const uint8_t *mask, float *result) {
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
          if (result != NULL) for (size_t d = 0; d < s->dh; d++)
            result[q_index(s, n, hq, tq, d)] = 0.0f;
          continue;
        }
        for (size_t d = 0; d < s->dh; d++) {
          float output = 0.0f;
          for (size_t tk = 0; tk < s->tk; tk++) {
            float probability;
            if (!attention_admitted(s, query_positions, key_positions, mask, n, tq, tk)) {
              continue;
            }
            if (!attention_probability(s, q, k, scale, n, hq, tq, tk,
                                       maximum, denominator, &probability)) {
              return 0;
            }
            const float product = probability * v[kv_index(s, n, hkv, tk, d)];
            if (!isfinite(product)) return 0;
            output = output + product;
            if (!isfinite(output)) {
              return 0;
            }
          }
          if (result != NULL) result[q_index(s, n, hq, tq, d)] = output;
        }
      }
    }
  }
  return 1;
}

/* Controls are checked before any floating value (including sNaN) is read. */
static int fp_environment_supported(void) {
  unsigned short control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  const unsigned int mxcsr = _mm_getcsr();
  return (control & 0x003fu) == 0x003fu && (control & 0x0c00u) == 0u &&
         (mxcsr & 0x00001f80u) == 0x00001f80u &&
         (mxcsr & 0x00006040u) == 0u && (mxcsr & 0x00008000u) == 0u;
}

enum kind { ATTN_F, EMB_F, GELU_F, MERGE_F, SPLIT_F, LN_F, LIN_F, RES_F, UNKNOWN };
static enum kind operation_kind(const char *name) {
  static const char *const names[] = {
    "g3c4.causal-attention.forward", "g3c4.embedding.forward",
    "g3c4.gelu.forward", "g3c4.heads.merge.forward",
    "g3c4.heads.split.forward", "g3c4.layer-norm.forward",
    "g3c4.linear.forward-no-bias", "g3c4.residual.forward"
  };
  for (size_t i=0; i<COUNT(names); ++i)
    if (exact_text(name,names[i])) return (enum kind)i;
  return UNKNOWN;
}

/* Descriptor metadata is the sole authority for exact request rows. */
static const et_kernel_capability_v1 *find_capability(const char *name);
static int admitted(const et_kernel_call_v1 *call) {
  const et_kernel_request_v1 *r=call->request;
  const et_kernel_capability_v1 *cap=find_capability(call->capability);
  if (!cap || !exact_text(r->dtype,"f32") || !exact_text(r->device,"cpu") ||
      r->rank>6u || r->shape==NULL || r->deterministic>1u ||
      !aligned_pointer(r->shape,_Alignof(uint64_t)) ||
      !pointer_span_fits(r->shape,r->rank*sizeof(uint64_t))) return 0;
  size_t op;
  for (op=0; op<cap->operation_count; ++op)
    if (exact_text(r->operation,cap->operations[op])) break;
  if (op==cap->operation_count) return 0;
  for (size_t row=0; row<cap->shape_range_count; ++row) {
    const et_kernel_shape_range_v1 *range=&cap->shape_ranges[row];
    if (r->rank!=range->rank) continue;
    size_t dim;
    for (dim=0; dim<r->rank; ++dim)
      if (r->shape[dim]!=range->dimensions[dim].minimum) break;
    if (dim==r->rank) return 1;
  }
  return 0;
}

typedef struct operand { const char *dtype; size_t rank, d[4]; } operand;
typedef struct schema { size_t ni; operand in[6], out; } schema;
static operand fshape(size_t rank,size_t a,size_t b,size_t c,size_t d) {
  const operand o={"f32",rank,{a,b,c,d}}; return o;
}
static schema operation_schema(enum kind k,const et_kernel_request_v1 *r) {
  schema s={0};
  switch (k) {
    case EMB_F:
      s.ni=2; s.in[0]=(operand){"i64",2,{(size_t)r->shape[0],
        (size_t)r->shape[1],0,0}};
      s.in[1]=fshape(2,(size_t)r->shape[2],(size_t)r->shape[3],0,0);
      s.out=fshape(3,(size_t)r->shape[0],(size_t)r->shape[1],
        (size_t)r->shape[3],0); break;
    case LIN_F:
      s.ni=2; s.in[0]=fshape(3,(size_t)r->shape[0],
        (size_t)r->shape[1],(size_t)r->shape[2],0);
      s.in[1]=fshape(2,(size_t)r->shape[3],(size_t)r->shape[2],0,0);
      s.out=fshape(3,(size_t)r->shape[0],(size_t)r->shape[1],
        (size_t)r->shape[3],0); break;
    case LN_F:
      s.ni=4; s.in[0]=s.out=fshape(3,(size_t)r->shape[0],
        (size_t)r->shape[1],(size_t)r->shape[2],0);
      s.in[1]=s.in[2]=fshape(1,(size_t)r->shape[2],0,0,0);
      s.in[3]=fshape(0,0,0,0,0); break;
    case GELU_F:
      s.ni=1; s.in[0]=s.out=fshape(3,(size_t)r->shape[0],
        (size_t)r->shape[1],(size_t)r->shape[2],0); break;
    case RES_F:
      s.ni=2; s.in[0]=s.in[1]=s.out=fshape(3,(size_t)r->shape[0],
        (size_t)r->shape[1],(size_t)r->shape[2],0); break;
    case SPLIT_F: case MERGE_F:
      s.ni=1;
      s.in[0]=k==SPLIT_F
        ? fshape(3,(size_t)r->shape[0],(size_t)r->shape[1],
                 (size_t)r->shape[2]*(size_t)r->shape[3],0)
        : fshape(4,(size_t)r->shape[0],(size_t)r->shape[2],
                 (size_t)r->shape[1],(size_t)r->shape[3]);
      s.out=k==SPLIT_F
        ? fshape(4,(size_t)r->shape[0],(size_t)r->shape[2],
                 (size_t)r->shape[1],(size_t)r->shape[3])
        : fshape(3,(size_t)r->shape[0],(size_t)r->shape[1],
                 (size_t)r->shape[2]*(size_t)r->shape[3],0); break;
    case ATTN_F:
      s.ni=6;
      s.in[0]=s.out=fshape(4,(size_t)r->shape[0],(size_t)r->shape[1],
        (size_t)r->shape[3],(size_t)r->shape[5]);
      s.in[1]=s.in[2]=fshape(4,(size_t)r->shape[0],
        (size_t)r->shape[2],(size_t)r->shape[4],(size_t)r->shape[5]);
      s.in[3]=(operand){"i64",2,{(size_t)r->shape[0],
        (size_t)r->shape[3],0,0}};
      s.in[4]=(operand){"i64",2,{(size_t)r->shape[0],
        (size_t)r->shape[4],0,0}};
      s.in[5]=(operand){"bool",3,{(size_t)r->shape[0],
        (size_t)r->shape[3],(size_t)r->shape[4],0}}; break;
    case UNKNOWN: break;
  }
  return s;
}

/* This identical calculation serves dry validation and synchronous publication.
 * Successful complete preflight plus stable storage makes invoke nonfailing. */
static int run_operation(const et_kernel_call_v1 *call,enum kind k,int write) {
  const uint64_t *d=call->request->shape;
  const void *in[6]={0};
  for (size_t i=0;i<call->input_count;++i) in[i]=input_at(call,i)->data;
  float *out=write ? output_at(call,0)->data:NULL;
  if (k==EMB_F || k==LIN_F) {
    const shape4 s={(size_t)d[0],(size_t)d[1],(size_t)d[2],(size_t)d[3]};
    return k==EMB_F ? embedding_forward_run(&s,in[0],in[1],out):
      linear_forward_run(&s,in[0],in[1],NULL,out);
  }
  if (k==LN_F) {
    const shape3 s={(size_t)d[0],(size_t)d[1],(size_t)d[2]};
    return layer_norm_forward_run(&s,in[0],in[1],in[2],*(const float *)in[3],out);
  }
  if (k==GELU_F) {
    const shape3 s={(size_t)d[0],(size_t)d[1],(size_t)d[2]};
    return gelu_forward_run(&s,in[0],out);
  }
  if (k==ATTN_F) {
    const attention_shape s={(size_t)d[0],(size_t)d[1],(size_t)d[2],
      (size_t)d[3],(size_t)d[4],(size_t)d[5]};
    return attention_forward_run(&s,in[0],in[1],in[2],in[3],in[4],in[5],out);
  }
  if (k==SPLIT_F || k==MERGE_F) {
    if (!write) return 1;
    for (size_t n=0;n<d[0];++n) for (size_t t=0;t<d[1];++t)
      for (size_t h=0;h<d[2];++h) for (size_t x=0;x<d[3];++x) {
        const size_t token=((n*(size_t)d[1]+t)*(size_t)d[2]+h)*(size_t)d[3]+x;
        const size_t head=((n*(size_t)d[2]+h)*(size_t)d[1]+t)*(size_t)d[3]+x;
        memcpy(out+(k==SPLIT_F ? head:token),
          (const float *)in[0]+(k==SPLIT_F ? token:head),sizeof(float));
      }
    return 1;
  }
  const size_t count=(size_t)d[0]*(size_t)d[1]*(size_t)d[2];
  for (size_t i=0;i<count;++i) {
    const float value=((const float *)in[0])[i]+((const float *)in[1])[i];
    if (!isfinite(value)) return 0;
    if (write) out[i]=value;
  }
  return 1;
}

static int32_t validate_inner(const et_kernel_call_v1 *call,et_kernel_error *error) {
  const char *op="g3c4.provider";
  if (!call) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"call is required");
  if (!aligned_pointer(call,_Alignof(et_kernel_call_v1)))
    return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"call is misaligned");
  if (call->struct_size<ET_KERNEL_CALL_V1_0_SIZE)
    return set_error(error,ET_KERNEL_ERROR_VERSION_MISMATCH,
      ET_KERNEL_CODE_INVALID_STRUCT_SIZE,op,"call truncates K1 prefix");
  if (!call->request) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"request is required");
  if (!aligned_pointer(call->request,_Alignof(et_kernel_request_v1)))
    return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"request is misaligned");
  if (call->request->struct_size<ET_KERNEL_REQUEST_V1_0_SIZE)
    return set_error(error,ET_KERNEL_ERROR_VERSION_MISMATCH,
      ET_KERNEL_CODE_INVALID_STRUCT_SIZE,op,"request truncates K1 prefix");
  if (!call->request->operation) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"operation is required");
  op=call->request->operation;
  if (!admitted(call)) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"request is outside exact G3-C4 evidence");
  const enum kind k=operation_kind(op);
  const schema s=operation_schema(k,call->request);
  const et_kernel_tensor_view_v1 *in[6],*out[1];
  int32_t rc=table_views(call,0,s.ni,op,in,error);
  if (!rc) rc=table_views(call,1,1,op,out,error);
  for (size_t i=0;!rc && i<s.ni+1u;++i) {
    const operand *o=i<s.ni ? &s.in[i]:&s.out;
    const et_kernel_tensor_view_v1 *v=i<s.ni ? in[i]:out[0];
    const int f=exact_text(o->dtype,"f32"), b=exact_text(o->dtype,"bool");
    rc=validate_view(v,o->dtype,f ? 4u:(b ? 1u:8u),
      f ? _Alignof(float):(b ? 1u:_Alignof(int64_t)),o->rank,o->d,op,error);
  }
  if (rc) return rc;
  for (size_t i=0;i<s.ni;++i) for (size_t j=i+1;j<s.ni;++j) {
    const int allowed=k==RES_F || (k==ATTN_F && i==3u && j==4u);
    if (ranges_overlap(in[i]->data,in[i]->byte_length,in[j]->data,in[j]->byte_length) &&
        !(allowed && same_view(in[i],in[j])))
      return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"input storage overlap is forbidden");
  }
  if (!fp_environment_supported()) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"unsupported x87 or MXCSR controls");
  for (size_t i=0;i<s.ni;++i)
    if (exact_text(s.in[i].dtype,"f32") &&
        !finite_values(in[i]->data,in[i]->byte_length/sizeof(float)))
      return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"floating input is nonfinite");
  if (k==EMB_F) {
    const size_t count=(size_t)call->request->shape[0]*
      (size_t)call->request->shape[1];
    const int64_t *ids=in[0]->data;
    for (size_t i=0;i<count;++i)
      if (ids[i]<0 || (uint64_t)ids[i]>=call->request->shape[2])
        return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
          ET_KERNEL_CODE_PROVIDER_REJECTED,op,
          "embedding ID is outside vocabulary");
  }
  if (k==LN_F && !(*(const float *)in[3]->data>0.0f))
    return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_PROVIDER_REJECTED,op,"epsilon must be positive and finite");
  if (k==ATTN_F) {
    const size_t n=(size_t)call->request->shape[0];
    const size_t tq=(size_t)call->request->shape[3];
    const size_t tk=(size_t)call->request->shape[4];
    const int64_t *q=in[3]->data,*kp=in[4]->data;
    const uint8_t *mask=in[5]->data;
    for (size_t batch=0;batch<n;++batch) {
      for (size_t t=0;t<tq;++t) {
        const int64_t position=q[batch*tq+t];
        if (position<0 || position>16777215 ||
            (t>0u && position<=q[batch*tq+t-1u]))
          return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
            ET_KERNEL_CODE_PROVIDER_REJECTED,op,
            "query positions must be in range and strictly increasing");
      }
      for (size_t t=0;t<tk;++t) {
        const int64_t position=kp[batch*tk+t];
        if (position<0 || position>16777215 ||
            (t>0u && position<=kp[batch*tk+t-1u]))
          return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
            ET_KERNEL_CODE_PROVIDER_REJECTED,op,
            "key positions must be in range and strictly increasing");
      }
    }
    for (size_t i=0;i<n*tq*tk;++i)
      if (mask[i]>1u)
        return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
          ET_KERNEL_CODE_PROVIDER_REJECTED,op,
          "attention keep bytes must be zero or one");
  }
  if (!run_operation(call,k,0)) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"prospective intermediate or result is nonfinite");
  et_kernel_error_clear(error);
  return 0;
}
static const char *error_operation(const et_kernel_call_v1 *call) {
  if (aligned_pointer(call,_Alignof(et_kernel_call_v1)) &&
      call->struct_size>=ET_KERNEL_CALL_V1_0_SIZE &&
      aligned_pointer(call->request,_Alignof(et_kernel_request_v1)) &&
      call->request->struct_size>=ET_KERNEL_REQUEST_V1_0_SIZE && call->request->operation)
    return call->request->operation;
  return "g3c4.provider";
}
static int32_t provider_validate(const et_kernel_call_v1 *call,et_kernel_error *error) {
  fenv_t environment;
  if (fegetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,error_operation(call),"cannot observe floating environment");
  const int32_t rc=validate_inner(call,error);
  if (fesetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_INTERNAL,
    ET_KERNEL_CODE_PROVIDER_REJECTED,error_operation(call),"cannot restore floating environment");
  return rc;
}
static void provider_invoke(const et_kernel_call_v1 *call) {
  (void)run_operation(call,operation_kind(call->request->operation),1);
}

/* Exact alternatives, in the accepted table order. */
#define D(v) {(v),(v),0,{0}}
static const et_kernel_dimension_range_v1 dims_0_0[] = {D(1),D(2),D(2),D(1),D(4),D(2)};
static const et_kernel_dimension_range_v1 dims_0_1[] = {D(1),D(2),D(2),D(2),D(4),D(2)};
static const et_kernel_dimension_range_v1 dims_0_2[] = {D(1),D(2),D(2),D(3),D(4),D(2)};
static const et_kernel_dimension_range_v1 dims_0_3[] = {D(1),D(2),D(2),D(4),D(4),D(2)};
static const et_kernel_dimension_range_v1 dims_0_4[] = {D(1),D(2),D(2),D(3),D(3),D(2)};
static const et_kernel_shape_range_v1 rows_0[] = {
  {6,dims_0_0},{6,dims_0_1},{6,dims_0_2},{6,dims_0_3},{6,dims_0_4}};
static const char *const ops_0[] = {"g3c4.causal-attention.forward"};

static const et_kernel_dimension_range_v1 dims_1_0[] = {D(1),D(1),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_1_1[] = {D(1),D(2),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_1_2[] = {D(1),D(3),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_1_3[] = {D(1),D(4),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_1_4[] = {D(1),D(3),D(256),D(4)};
static const et_kernel_dimension_range_v1 dims_1_5[] = {D(1),D(4),D(256),D(4)};
static const et_kernel_shape_range_v1 rows_1[] = {
  {4,dims_1_0},{4,dims_1_1},{4,dims_1_2},{4,dims_1_3},
  {4,dims_1_4},{4,dims_1_5}};
static const char *const ops_1[] = {"g3c4.embedding.forward"};

static const et_kernel_dimension_range_v1 dims_2_0[] = {D(1),D(3),D(8)};
static const et_kernel_dimension_range_v1 dims_2_1[] = {D(1),D(4),D(8)};
static const et_kernel_shape_range_v1 rows_2[] = {{3,dims_2_0},{3,dims_2_1}};
static const char *const ops_2[] = {"g3c4.gelu.forward"};

static const et_kernel_dimension_range_v1 dims_3_0[] = {D(1),D(3),D(2),D(2)};
static const et_kernel_dimension_range_v1 dims_3_1[] = {D(1),D(4),D(2),D(2)};
static const et_kernel_shape_range_v1 rows_3[] = {{4,dims_3_0},{4,dims_3_1}};
static const char *const ops_3[] = {
  "g3c4.heads.merge.forward","g3c4.heads.split.forward"};

static const et_kernel_dimension_range_v1 dims_4_0[] = {D(1),D(3),D(4)};
static const et_kernel_dimension_range_v1 dims_4_1[] = {D(1),D(4),D(4)};
static const et_kernel_shape_range_v1 rows_4[] = {{3,dims_4_0},{3,dims_4_1}};
static const char *const ops_4[] = {"g3c4.layer-norm.forward"};

static const et_kernel_dimension_range_v1 dims_5_0[] = {D(1),D(3),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_5_1[] = {D(1),D(3),D(4),D(8)};
static const et_kernel_dimension_range_v1 dims_5_2[] = {D(1),D(3),D(8),D(4)};
static const et_kernel_dimension_range_v1 dims_5_3[] = {D(1),D(3),D(4),D(256)};
static const et_kernel_dimension_range_v1 dims_5_4[] = {D(1),D(4),D(4),D(4)};
static const et_kernel_dimension_range_v1 dims_5_5[] = {D(1),D(4),D(4),D(8)};
static const et_kernel_dimension_range_v1 dims_5_6[] = {D(1),D(4),D(8),D(4)};
static const et_kernel_dimension_range_v1 dims_5_7[] = {D(1),D(4),D(4),D(256)};
static const et_kernel_shape_range_v1 rows_5[] = {
  {4,dims_5_0},{4,dims_5_1},{4,dims_5_2},{4,dims_5_3},
  {4,dims_5_4},{4,dims_5_5},{4,dims_5_6},{4,dims_5_7}};
static const char *const ops_5[] = {"g3c4.linear.forward-no-bias"};

static const et_kernel_dimension_range_v1 dims_6_0[] = {D(1),D(4),D(4)};
static const et_kernel_shape_range_v1 rows_6[] = {{3,dims_6_0}};
static const char *const ops_6[] = {"g3c4.residual.forward"};
#undef D
static const char *const dtypes[]={"f32"};
static const char *const devices[]={"cpu"};
static const et_kernel_capability_v1 capabilities[]={
  {sizeof(et_kernel_capability_v1),"g3c4.causal-attention",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_0),ops_0,1,dtypes,1,devices,COUNT(rows_0),rows_0},
  {sizeof(et_kernel_capability_v1),"g3c4.embedding-forward",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_1),ops_1,1,dtypes,1,devices,COUNT(rows_1),rows_1},
  {sizeof(et_kernel_capability_v1),"g3c4.gelu",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_2),ops_2,1,dtypes,1,devices,COUNT(rows_2),rows_2},
  {sizeof(et_kernel_capability_v1),"g3c4.head-layout",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_3),ops_3,1,dtypes,1,devices,COUNT(rows_3),rows_3},
  {sizeof(et_kernel_capability_v1),"g3c4.layer-norm",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_4),ops_4,1,dtypes,1,devices,COUNT(rows_4),rows_4},
  {sizeof(et_kernel_capability_v1),"g3c4.linear",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_5),ops_5,1,dtypes,1,devices,COUNT(rows_5),rows_5},
  {sizeof(et_kernel_capability_v1),"g3c4.residual",ET_KERNEL_CAPABILITY_VERIFIED,
    "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",1,{0},
    COUNT(ops_6),ops_6,1,dtypes,1,devices,COUNT(rows_6),rows_6},
};
static const et_kernel_capability_v1 *find_capability(const char *name) {
  for (size_t i=0;i<COUNT(capabilities);++i)
    if (exact_text(name,capabilities[i].name)) return &capabilities[i];
  return NULL;
}
static const et_kernel_provider_v1 provider={
  sizeof(et_kernel_provider_v1),ET_KERNEL_ABI_MAJOR,ET_KERNEL_ABI_MINOR,0,
  "g3c4.cpu-f32.serial","1.0","G3-C4:cpu-f32-c4-forward-v1",COUNT(capabilities),
  sizeof(et_kernel_capability_v1),sizeof(capabilities),capabilities,
  provider_validate,provider_invoke
};
const et_kernel_provider_v1 *et_g3c4_kernel_provider_v1(void) { return &provider; }
