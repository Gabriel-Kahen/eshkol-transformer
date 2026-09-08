#include "eshkol_transformer/n3k_primitives_abi.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if !defined(__x86_64__)
#error "N3K v1 requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4, "N3K requires eight-bit bytes/binary32");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
               FLT_EVAL_METHOD == 0, "N3K requires standard IEEE binary32 evaluation");
typedef struct shape4 { size_t n, t, a, b; } shape4;
typedef struct rng_state { uint64_t key, counter_low, counter_high; } rng_state;

/* Required private helpers copied from frozen N2 at 231f935. Numerical helper
 * statements retain the reviewed order. N2 source/artifacts are not modified;
 * this provider has independent row admission, schemas, controls and identity. */
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
                     "N3K v1 accepts only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "N3K v1 requires dense zero-offset tensor views");
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

/* Controls are checked before any floating value (including sNaN) is read. */
static int fp_environment_supported(void) {
  unsigned short control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  const unsigned int mxcsr = _mm_getcsr();
  return (control & 0x003fu) == 0x003fu && (control & 0x0c00u) == 0u &&
         (mxcsr & 0x00001f80u) == 0x00001f80u &&
         (mxcsr & 0x00006040u) == 0u && (mxcsr & 0x00008000u) == 0u;
}

enum kind { EMB_F, EMB_B, LIN_F, LIN_B, GELU_F, GELU_B, RES_F, RES_B,
            SPLIT_F, SPLIT_B, MERGE_F, MERGE_B, SUM2_F, SUM2_B,
            SUM3_F, SUM3_B, INIT, UNKNOWN };
static enum kind operation_kind(const char *name) {
  static const char *const names[] = {
    "n3k.embedding.forward", "n3k.embedding.backward",
    "n3k.linear.forward-no-bias", "n3k.linear.backward-no-bias",
    "n3k.gelu.forward", "n3k.gelu.backward",
    "n3k.residual.forward", "n3k.residual.backward",
    "n3k.heads.split.forward", "n3k.heads.split.backward",
    "n3k.heads.merge.forward", "n3k.heads.merge.backward",
    "n3k.sum2.forward", "n3k.sum2.backward",
    "n3k.sum3.forward", "n3k.sum3.backward", "n3k.matrix-init.uniform"
  };
  for (size_t i = 0; i < COUNT(names); ++i)
    if (exact_text(name, names[i])) return (enum kind)i;
  return UNKNOWN;
}

/* Metadata is the single exact-row admission source, also for direct validation. */
static const et_kernel_capability_v1 *find_capability(const char *name);
static int admitted(const et_kernel_call_v1 *call) {
  const et_kernel_request_v1 *r = call->request;
  const et_kernel_capability_v1 *cap = find_capability(call->capability);
  if (!cap || !exact_text(r->dtype, "f32") || !exact_text(r->device, "cpu") ||
      r->rank > 4u || r->shape == NULL ||
      !aligned_pointer(r->shape, _Alignof(uint64_t)) ||
      !pointer_span_fits(r->shape, r->rank * sizeof(uint64_t))) return 0;
  size_t op;
  for (op = 0; op < cap->operation_count; ++op)
    if (exact_text(r->operation, cap->operations[op])) break;
  if (op == cap->operation_count) return 0;
  for (size_t row = 0; row < cap->shape_range_count; ++row) {
    const et_kernel_shape_range_v1 *range = &cap->shape_ranges[row];
    if (r->rank != range->rank) continue;
    size_t dim;
    for (dim = 0; dim < r->rank; ++dim)
      if (r->shape[dim] != range->dimensions[dim].minimum) break;
    if (dim == r->rank) return 1;
  }
  return 0;
}

typedef struct operand { const char *dtype; size_t rank, d[4]; } operand;
typedef struct schema { size_t ni, no; operand in[3], out[3]; } schema;
static operand fshape(size_t rank, size_t a, size_t b, size_t c, size_t d) {
  const operand o = {"f32", rank, {a,b,c,d}}; return o;
}
static schema operation_schema(enum kind k, const et_kernel_request_v1 *r) {
  const size_t a=(size_t)r->shape[0], b=(size_t)r->shape[1];
  const size_t c=r->rank > 2u ? (size_t)r->shape[2] : 0u;
  const size_t d=r->rank > 3u ? (size_t)r->shape[3] : 0u;
  schema s = {0};
  operand value=fshape(r->rank,a,b,c,d);
  switch (k) {
    case EMB_F: case EMB_B:
      s.ni=2; s.no=1;
      s.in[0]=(operand){"i64",2,{a,b,0,0}};
      s.in[1]=fshape(k==EMB_F ? 2u:3u,k==EMB_F ? c:a,k==EMB_F ? d:b,d,0);
      s.out[0]=k==EMB_F ? fshape(3,a,b,d,0):fshape(2,c,d,0,0);
      break;
    case LIN_F: case LIN_B:
      s.ni=k==LIN_F ? 2u:3u; s.no=k==LIN_F ? 1u:2u;
      s.in[0]=fshape(3,a,b,c,0); s.in[1]=fshape(2,d,c,0,0);
      s.in[2]=fshape(3,a,b,d,0);
      s.out[0]=k==LIN_F ? s.in[2]:s.in[0]; s.out[1]=s.in[1];
      break;
    case GELU_F: case GELU_B:
      s.ni=k==GELU_F ? 1u:2u; s.no=1;
      s.in[0]=s.in[1]=s.out[0]=value; break;
    case SPLIT_F: case SPLIT_B: case MERGE_F: case MERGE_B:
      s.ni=s.no=1;
      s.in[0]=(k==SPLIT_F || k==MERGE_B) ? fshape(3,a,b,c*d,0):fshape(4,a,c,b,d);
      s.out[0]=(k==SPLIT_F || k==MERGE_B) ? fshape(4,a,c,b,d):fshape(3,a,b,c*d,0);
      break;
    case INIT:
      s.ni=1; s.no=2; s.in[0]=(operand){"i64",1,{4,0,0,0}};
      s.out[0]=value; s.out[1]=s.in[0]; break;
    default: {
      const int backward=k==RES_B || k==SUM2_B || k==SUM3_B;
      const size_t edges=(k==SUM3_F || k==SUM3_B) ? 3u:2u;
      s.ni=backward ? 1u:edges; s.no=backward ? edges:1u;
      for (size_t i=0;i<s.ni;++i) s.in[i]=value;
      for (size_t i=0;i<s.no;++i) s.out[i]=value;
      break;
    }
  }
  return s;
}

/* A NULL output means dry evaluation. Every prospective intermediate follows
 * exactly the invoke order; no output, counter or other state is mutated. */
static int run_operation(const et_kernel_call_v1 *call, enum kind k, int write) {
  const uint64_t *d=call->request->shape;
  const void *in[3]={NULL,NULL,NULL};
  void *out[3]={NULL,NULL,NULL};
  for (size_t i=0;i<call->input_count;++i) in[i]=input_at(call,i)->data;
  if (write) for (size_t i=0;i<call->output_count;++i) out[i]=output_at(call,i)->data;
  if (k==EMB_F || k==EMB_B || k==LIN_F || k==LIN_B) {
    const shape4 s={(size_t)d[0],(size_t)d[1],(size_t)d[2],(size_t)d[3]};
    if (k==EMB_F) return embedding_forward_run(&s,in[0],in[1],out[0]);
    if (k==EMB_B) return embedding_backward_run(&s,in[0],in[1],out[0]);
    if (k==LIN_F) return linear_forward_run(&s,in[0],in[1],NULL,out[0]);
    return linear_backward_run(&s,in[0],in[1],in[2],0,out[0],out[1],NULL);
  }
  if (k==INIT) {
    rng_state state, next;
    (void)decode_rng_state(in[0],&state);
    const size_t count=(size_t)d[0]*(size_t)d[1];
    const uint64_t blocks=(uint64_t)(count/4u)+(uint64_t)(count%4u!=0);
    if (!advance_counter(&state,blocks,&next)) return 0;
    uint32_t lanes[4]={0};
    uint64_t low=state.counter_low, high=state.counter_high;
    for (size_t i=0;i<count;++i) {
      if (i%4u==0) {
        if (i!=0u && ++low==0u) ++high;
        philox4x32_10(state.key,low,high,lanes);
      }
      const float u=(float)(lanes[i%4u]>>8)*0x1p-24f;
      const float centered=u-0.5f;
      const float value=centered*0x1p-4f;
      if (!isfinite(value)) return 0;
      if (write) ((float *)out[0])[i]=value;
    }
    if (write) encode_rng_state(&next,out[1]);
    return 1;
  }
  const size_t count=input_at(call,0)->byte_length/sizeof(float);
  if (k==SPLIT_F || k==SPLIT_B || k==MERGE_F || k==MERGE_B) {
    if (!write) return 1;
    for (size_t n=0;n<d[0];++n) for (size_t t=0;t<d[1];++t)
      for (size_t h=0;h<d[2];++h) for (size_t x=0;x<d[3];++x) {
        const size_t token=((n*(size_t)d[1]+t)*(size_t)d[2]+h)*(size_t)d[3]+x;
        const size_t head=((n*(size_t)d[2]+h)*(size_t)d[1]+t)*(size_t)d[3]+x;
        const int split=k==SPLIT_F || k==MERGE_B;
        memcpy((float *)out[0]+(split ? head:token),
               (const float *)in[0]+(split ? token:head),sizeof(float));
      }
    return 1;
  }
  if (k==RES_B || k==SUM2_B || k==SUM3_B) {
    if (write) for (size_t i=0;i<call->output_count;++i)
      memcpy(out[i],in[0],count*sizeof(float));
    return 1;
  }
  for (size_t i=0;i<count;++i) {
    float value;
    const float a=((const float *)in[0])[i];
    if (k==GELU_F || k==GELU_B) {
      const int valid=k==GELU_F ? gelu_one(a,&value):
        gelu_backward_one(a,((const float *)in[1])[i],&value);
      if (!valid) return 0;
    } else {
      value=a+((const float *)in[1])[i];
      if (!isfinite(value)) return 0;
      if (k==SUM3_F) value=value+((const float *)in[2])[i];
      if (!isfinite(value)) return 0;
    }
    if (write) ((float *)out[0])[i]=value;
  }
  return 1;
}

static int32_t validate_inner(const et_kernel_call_v1 *call, et_kernel_error *error) {
  const char *op="n3k.provider";
  if (!call) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"call is required");
  if (call->struct_size<ET_KERNEL_CALL_V1_0_SIZE)
    return set_error(error,ET_KERNEL_ERROR_VERSION_MISMATCH,
      ET_KERNEL_CODE_INVALID_STRUCT_SIZE,op,"call truncates K1 prefix");
  if (!call->request) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"request is required");
  if (call->request->struct_size<ET_KERNEL_REQUEST_V1_0_SIZE)
    return set_error(error,ET_KERNEL_ERROR_VERSION_MISMATCH,
      ET_KERNEL_CODE_INVALID_STRUCT_SIZE,op,"request truncates K1 prefix");
  if (!call->request->operation) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT,op,"operation is required");
  op=call->request->operation;
  if (!admitted(call)) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"request is outside exact N3K evidence");
  if (!fp_environment_supported()) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"unsupported x87 or MXCSR controls");
  const enum kind k=operation_kind(op);
  const schema s=operation_schema(k,call->request);
  const et_kernel_tensor_view_v1 *in[3], *out[3];
  int32_t rc=table_views(call,0,s.ni,op,in,error);
  if (!rc) rc=table_views(call,1,s.no,op,out,error);
  for (size_t i=0;!rc && i<s.ni+s.no;++i) {
    const operand *o=i<s.ni ? &s.in[i]:&s.out[i-s.ni];
    const et_kernel_tensor_view_v1 *v=i<s.ni ? in[i]:out[i-s.ni];
    const int f=exact_text(o->dtype,"f32");
    rc=validate_view(v,o->dtype,f ? sizeof(float):sizeof(int64_t),
      f ? _Alignof(float):_Alignof(int64_t),o->rank,o->d,op,error);
  }
  if (rc) return rc;
  for (size_t i=0;i<s.ni;++i) for (size_t j=i+1;j<s.ni;++j) {
    if (ranges_overlap(in[i]->data,in[i]->byte_length,in[j]->data,in[j]->byte_length) &&
        !((k==RES_F || k==SUM2_F || k==SUM3_F) && same_view(in[i],in[j])))
      return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"input storage overlap is forbidden");
  }
  for (size_t i=0;i<s.ni;++i)
    if (exact_text(s.in[i].dtype,"f32") &&
        !finite_values(in[i]->data,in[i]->byte_length/sizeof(float)))
      return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"floating input is nonfinite");
  if (k==EMB_F || k==EMB_B) {
    const int64_t *ids=in[0]->data;
    for (size_t i=0;i<in[0]->byte_length/sizeof(int64_t);++i)
      if (ids[i]<0 || (uint64_t)ids[i]>=call->request->shape[2])
        return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
          ET_KERNEL_CODE_PROVIDER_REJECTED,op,"embedding ID is outside vocabulary");
  }
  if (k==INIT) {
    rng_state state,next;
    if (!decode_rng_state(in[0]->data,&state))
      return set_error(error,ET_KERNEL_ERROR_VERSION_MISMATCH,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"unknown initializer state version");
    const size_t count=out[0]->byte_length/sizeof(float);
    const uint64_t blocks=(uint64_t)(count/4u)+(uint64_t)(count%4u!=0);
    if ((state.counter_low==UINT64_MAX && state.counter_high==UINT64_MAX) ||
        !advance_counter(&state,blocks,&next))
      return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"initializer counter exhausted");
  }
  if (!run_operation(call,k,0)) return set_error(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_PROVIDER_REJECTED,op,"prospective intermediate or result is nonfinite");
  et_kernel_error_clear(error);
  return 0;
}
static int32_t provider_validate(const et_kernel_call_v1 *call, et_kernel_error *error) {
  fenv_t environment;
  if (fegetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,"n3k.provider","cannot observe floating environment");
  const int32_t rc=validate_inner(call,error);
  if (fesetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_INTERNAL,
    ET_KERNEL_CODE_PROVIDER_REJECTED,"n3k.provider","cannot restore floating environment");
  return rc;
}
static void provider_invoke(const et_kernel_call_v1 *call) {
  (void)run_operation(call,operation_kind(call->request->operation),1);
}

/* Exact alternatives: no rectangular widening or operation cross-product gaps. */
static const et_kernel_dimension_range_v1 dims_0_0[] = {{1,1,0,{0}},{2,2,0,{0}},{4,4,0,{0}}};
static const et_kernel_shape_range_v1 rows_0[] = {{3,dims_0_0}};
static const char *const ops_0[] = {"n3k.sum2.backward","n3k.sum2.forward","n3k.sum3.backward","n3k.sum3.forward"};
static const et_kernel_dimension_range_v1 dims_1_0[] = {{1,1,0,{0}},{2,2,0,{0}},{256,256,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_1_1[] = {{1,1,0,{0}},{2,2,0,{0}},{2,2,0,{0}},{4,4,0,{0}}};
static const et_kernel_shape_range_v1 rows_1[] = {{4,dims_1_0},{4,dims_1_1}};
static const char *const ops_1[] = {"n3k.embedding.backward"};
static const et_kernel_dimension_range_v1 dims_2_0[] = {{1,1,0,{0}},{2,2,0,{0}},{256,256,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_2_1[] = {{1,1,0,{0}},{2,2,0,{0}},{2,2,0,{0}},{4,4,0,{0}}};
static const et_kernel_shape_range_v1 rows_2[] = {{4,dims_2_0},{4,dims_2_1}};
static const char *const ops_2[] = {"n3k.embedding.forward"};
static const et_kernel_dimension_range_v1 dims_3_0[] = {{1,1,0,{0}},{2,2,0,{0}},{8,8,0,{0}}};
static const et_kernel_shape_range_v1 rows_3[] = {{3,dims_3_0}};
static const char *const ops_3[] = {"n3k.gelu.backward","n3k.gelu.forward"};
static const et_kernel_dimension_range_v1 dims_4_0[] = {{1,1,0,{0}},{2,2,0,{0}},{2,2,0,{0}},{2,2,0,{0}}};
static const et_kernel_shape_range_v1 rows_4[] = {{4,dims_4_0}};
static const char *const ops_4[] = {"n3k.heads.merge.backward","n3k.heads.merge.forward","n3k.heads.split.backward","n3k.heads.split.forward"};
static const et_kernel_dimension_range_v1 dims_5_0[] = {{1,1,0,{0}},{2,2,0,{0}},{4,4,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_5_1[] = {{1,1,0,{0}},{2,2,0,{0}},{4,4,0,{0}},{8,8,0,{0}}};
static const et_kernel_dimension_range_v1 dims_5_2[] = {{1,1,0,{0}},{2,2,0,{0}},{8,8,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_5_3[] = {{1,1,0,{0}},{2,2,0,{0}},{4,4,0,{0}},{256,256,0,{0}}};
static const et_kernel_shape_range_v1 rows_5[] = {{4,dims_5_0},{4,dims_5_1},{4,dims_5_2},{4,dims_5_3}};
static const char *const ops_5[] = {"n3k.linear.backward-no-bias","n3k.linear.forward-no-bias"};
static const et_kernel_dimension_range_v1 dims_6_0[] = {{256,256,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_6_1[] = {{2,2,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_6_2[] = {{4,4,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_6_3[] = {{8,8,0,{0}},{4,4,0,{0}}};
static const et_kernel_dimension_range_v1 dims_6_4[] = {{4,4,0,{0}},{8,8,0,{0}}};
static const et_kernel_shape_range_v1 rows_6[] = {{2,dims_6_0},{2,dims_6_1},{2,dims_6_2},{2,dims_6_3},{2,dims_6_4}};
static const char *const ops_6[] = {"n3k.matrix-init.uniform"};
static const et_kernel_dimension_range_v1 dims_7_0[] = {{1,1,0,{0}},{2,2,0,{0}},{4,4,0,{0}}};
static const et_kernel_shape_range_v1 rows_7[] = {{3,dims_7_0}};
static const char *const ops_7[] = {"n3k.residual.backward","n3k.residual.forward"};
static const et_kernel_dimension_range_v1 dims_8_0[] = {{256,256,0,{0}},{4,4,0,{0}}};
static const et_kernel_shape_range_v1 rows_8[] = {{2,dims_8_0}};
static const char *const ops_8[] = {"n3k.sum2.backward","n3k.sum2.forward"};
static const char *const dtypes[]={"f32"};
static const char *const devices[]={"cpu"};
static const et_kernel_capability_v1 capabilities[]={
  {sizeof(et_kernel_capability_v1),"n3k.activation-sum",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_0),ops_0,1,dtypes,1,devices,COUNT(rows_0),rows_0},
  {sizeof(et_kernel_capability_v1),"n3k.embedding-backward",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_1),ops_1,1,dtypes,1,devices,COUNT(rows_1),rows_1},
  {sizeof(et_kernel_capability_v1),"n3k.embedding-forward",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_2),ops_2,1,dtypes,1,devices,COUNT(rows_2),rows_2},
  {sizeof(et_kernel_capability_v1),"n3k.gelu",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_3),ops_3,1,dtypes,1,devices,COUNT(rows_3),rows_3},
  {sizeof(et_kernel_capability_v1),"n3k.head-layout",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_4),ops_4,1,dtypes,1,devices,COUNT(rows_4),rows_4},
  {sizeof(et_kernel_capability_v1),"n3k.linear",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_5),ops_5,1,dtypes,1,devices,COUNT(rows_5),rows_5},
  {sizeof(et_kernel_capability_v1),"n3k.matrix-init",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_6),ops_6,1,dtypes,1,devices,COUNT(rows_6),rows_6},
  {sizeof(et_kernel_capability_v1),"n3k.residual",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_7),ops_7,1,dtypes,1,devices,COUNT(rows_7),rows_7},
  {sizeof(et_kernel_capability_v1),"n3k.tied-sum",ET_KERNEL_CAPABILITY_VERIFIED,
    "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",1,{0},
    COUNT(ops_8),ops_8,1,dtypes,1,devices,COUNT(rows_8),rows_8},
};
static const et_kernel_capability_v1 *find_capability(const char *name) {
  for (size_t i=0;i<COUNT(capabilities);++i)
    if (exact_text(name,capabilities[i].name)) return &capabilities[i];
  return NULL;
}
static const et_kernel_provider_v1 provider={
  sizeof(et_kernel_provider_v1),ET_KERNEL_ABI_MAJOR,ET_KERNEL_ABI_MINOR,0,
  "n3k.cpu-f32.serial","1.0","N3K:cpu-f32-diagnostic-v1",COUNT(capabilities),
  sizeof(et_kernel_capability_v1),sizeof(capabilities),capabilities,
  provider_validate,provider_invoke
};
const et_kernel_provider_v1 *et_n3k_kernel_provider_v1(void) { return &provider; }
