#include "eshkol_transformer/mod5_primitives_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u, "MOD5 requires binary32");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 requires standard IEEE-754 binary32 evaluation");
_Static_assert(ET_MOD5_MAX_EXACT_POSITION < (INT64_C(1) << FLT_MANT_DIG),
               "every admitted position difference must be exact in f32");

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define DIM(value) {value, value, 0u, {0}}
#define ROW3(n, t, d)                                                          \
  {                                                                            \
    3u, (const et_kernel_dimension_range_v1[]) { DIM(n), DIM(t), DIM(d) }      \
  }
#define ROW6(n, hq, hkv, tq, tk, dh)                                           \
  {                                                                            \
    6u, (const et_kernel_dimension_range_v1[]) {                               \
      DIM(n), DIM(hq), DIM(hkv), DIM(tq), DIM(tk), DIM(dh)                     \
    }                                                                          \
  }
#define IMPLEMENTATION "mod5.cpu-f32.serial"
#define EVIDENCE "MOD5:cpu-f32-variants-v1"

enum primitive { RMSNORM, SWIGLU, ATTENTION };
enum { MAX_KEYS = 3, MAX_OUTPUT_ELEMENTS = 64, MAX_OUTPUTS = 3 };

static const et_kernel_shape_range_v1 rmsnorm_rows[] = {
    ROW3(1, 1, 1), ROW3(1, 1, 4), ROW3(1, 2, 4), ROW3(2, 3, 4), ROW3(2, 3, 8)};
static const et_kernel_shape_range_v1 swiglu_rows[] = {
    ROW3(1, 1, 1), ROW3(1, 1, 8), ROW3(1, 2, 4), ROW3(2, 3, 4), ROW3(2, 3, 8)};
static const et_kernel_shape_range_v1 attention_rows[] = {
    ROW6(1, 2, 1, 2, 2, 2), ROW6(1, 4, 1, 3, 3, 2), ROW6(2, 4, 1, 2, 3, 4),
    ROW6(1, 4, 2, 3, 3, 2), ROW6(2, 4, 2, 2, 3, 4), ROW6(1, 2, 2, 3, 3, 2)};

static const char *const rmsnorm_operations[] = {"rmsnorm.forward",
                                                 "rmsnorm.backward"};
static const char *const swiglu_operations[] = {"swiglu.forward",
                                                "swiglu.backward"};
static const char *const attention_operations[] = {
    "variant-attention.forward", "variant-attention.backward"};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};

#define CAPABILITY(name, operations, rows)                                     \
  {sizeof(et_kernel_capability_v1),                                            \
   name,                                                                       \
   ET_KERNEL_CAPABILITY_VERIFIED,                                              \
   IMPLEMENTATION,                                                             \
   "1.0",                                                                      \
   EVIDENCE,                                                                   \
   1u,                                                                         \
   {0},                                                                        \
   COUNT(operations),                                                          \
   operations,                                                                 \
   COUNT(dtypes),                                                              \
   dtypes,                                                                     \
   COUNT(devices),                                                             \
   devices,                                                                    \
   COUNT(rows),                                                                \
   rows}

static const et_kernel_capability_v1 capabilities[] = {
    [RMSNORM] = CAPABILITY("kernel.rmsnorm", rmsnorm_operations, rmsnorm_rows),
    [SWIGLU] = CAPABILITY("kernel.swiglu", swiglu_operations, swiglu_rows),
    [ATTENTION] = CAPABILITY("kernel.variant-attention", attention_operations,
                             attention_rows)};

typedef struct tensor_table {
  size_t count;
  size_t stride;
  size_t bytes;
  const void *base;
} tensor_table;

enum dtype { F32, I64, BOOL };
typedef struct tensor_schema {
  enum dtype dtype;
  size_t rank;
  const uint64_t *shape;
} tensor_schema;

typedef float result_buffers[MAX_OUTPUTS][MAX_OUTPUT_ELEMENTS];

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int overlap(const et_kernel_tensor_view_v1 *a,
                   const et_kernel_tensor_view_v1 *b) {
  return spans_overlap(a->data, a->byte_length, b->data, b->byte_length);
}

static int32_t fail(et_kernel_error *error, et_kernel_error_category category,
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

static int32_t rejected(et_kernel_error *error, const char *operation,
                        const char *message) {
  return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
              ET_KERNEL_CODE_PROVIDER_REJECTED, operation, message);
}

static tensor_table inputs(const et_kernel_call_v1 *call) {
  return (tensor_table){call->input_count, call->input_stride,
                        call->input_bytes, call->inputs};
}

static tensor_table outputs(const et_kernel_call_v1 *call) {
  return (tensor_table){call->output_count, call->output_stride,
                        call->output_bytes, call->outputs};
}

static const et_kernel_tensor_view_v1 *view_at(tensor_table table,
                                               size_t index) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)table.base +
                                            index * table.stride);
}

static size_t bounded_text_bytes(const char *text) {
  size_t len = 0u;
  while (len <= ET_KERNEL_MAX_SYMBOL_BYTES && text[len] != '\0')
    ++len;
  return len + 1u;
}

static int output_aliases_metadata(const et_kernel_tensor_view_v1 *view,
                                   const et_kernel_call_v1 *call,
                                   tensor_table in, tensor_table out,
                                   const et_kernel_error *error) {
  const void *data = view->data;
  const size_t bytes = view->byte_length;
  const et_kernel_request_v1 *request = call->request;
  if (spans_overlap(data, bytes, call, call->struct_size) ||
      spans_overlap(data, bytes, request, request->struct_size) ||
      spans_overlap(data, bytes, request->shape,
                    request->rank * sizeof(*request->shape)) ||
      spans_overlap(data, bytes, call->capability,
                    bounded_text_bytes(call->capability)) ||
      spans_overlap(data, bytes, request->operation,
                    bounded_text_bytes(request->operation)) ||
      spans_overlap(data, bytes, request->dtype,
                    bounded_text_bytes(request->dtype)) ||
      spans_overlap(data, bytes, request->device,
                    bounded_text_bytes(request->device)) ||
      spans_overlap(data, bytes, in.base, in.bytes) ||
      spans_overlap(data, bytes, out.base, out.bytes) ||
      (error != NULL && spans_overlap(data, bytes, error, sizeof(*error))))
    return 1;
  for (size_t j = 0; j < in.count; ++j) {
    const et_kernel_tensor_view_v1 *entry = view_at(in, j);
    if ((entry->rank != 0u &&
         spans_overlap(data, bytes, entry->shape,
                       entry->rank * sizeof(*entry->shape))) ||
        spans_overlap(data, bytes, entry->dtype,
                      bounded_text_bytes(entry->dtype)) ||
        spans_overlap(data, bytes, entry->device,
                      bounded_text_bytes(entry->device)))
      return 1;
  }
  for (size_t j = 0; j < out.count; ++j) {
    const et_kernel_tensor_view_v1 *entry = view_at(out, j);
    if ((entry->rank != 0u &&
         spans_overlap(data, bytes, entry->shape,
                       entry->rank * sizeof(*entry->shape))) ||
        spans_overlap(data, bytes, entry->dtype,
                      bounded_text_bytes(entry->dtype)) ||
        spans_overlap(data, bytes, entry->device,
                      bounded_text_bytes(entry->device)))
      return 1;
  }
  return 0;
}

static int32_t validate_view(const et_kernel_tensor_view_v1 *view,
                             const tensor_schema *schema, size_t stride,
                             const char *operation, et_kernel_error *error) {
  static const char *const names[] = {
      [F32] = "f32", [I64] = "i64", [BOOL] = "bool"};
  static const size_t sizes[] = {
      [F32] = sizeof(float), [I64] = sizeof(int64_t), [BOOL] = sizeof(uint8_t)};
  static const size_t alignments[] = {[F32] = _Alignof(float),
                                      [I64] = _Alignof(int64_t),
                                      [BOOL] = _Alignof(uint8_t)};
  size_t bytes = sizes[schema->dtype];
  if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      view->struct_size > stride) {
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "invalid view prefix size");
  }
  if (!exact_text(view->dtype, names[schema->dtype])) {
    return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "view dtype does not match the operation");
  }
  if (!exact_text(view->device, "cpu")) {
    return fail(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "view must be on CPU");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return fail(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "view must be dense and zero-offset");
  }
  if (view->rank != schema->rank) {
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, operation,
                "view rank does not match the operation");
  }
  if (schema->rank != 0u &&
      !aligned_span(view->shape, _Alignof(uint64_t),
                    schema->rank * sizeof(*view->shape))) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "invalid view shape storage");
  }
  for (size_t d = 0; d < schema->rank; ++d) {
    if (view->shape[d] != schema->shape[d]) {
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, operation,
                  "view extent does not match the operation");
    }
    /* The schema uses only extents from the admitted exact rows. */
    bytes *= schema->shape[d];
  }
  if (view->byte_length != bytes) {
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "incorrect view byte length");
  }
  if (!aligned_span(view->data, alignments[schema->dtype], bytes)) {
    return rejected(error, operation, "invalid view data span or alignment");
  }
  return 0;
}

static int32_t validate_table(tensor_table table, size_t count,
                              const tensor_schema *schema,
                              const char *operation, et_kernel_error *error) {
  if (table.count != count || table.base == NULL) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "incorrect tensor table count or null table");
  }
  if (table.stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "truncated tensor table stride");
  }
  if (table.stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      !aligned_span(table.base, _Alignof(et_kernel_tensor_view_v1), 0u)) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "misaligned tensor table");
  }
  if (count > SIZE_MAX / table.stride || table.bytes != count * table.stride ||
      !aligned_span(table.base, _Alignof(et_kernel_tensor_view_v1),
                    table.bytes)) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                "invalid tensor table span");
  }
  for (size_t i = 0; i < count; ++i) {
    const int32_t status = validate_view(view_at(table, i), &schema[i],
                                         table.stride, operation, error);
    if (status != 0)
      return status;
  }
  return 0;
}

static int32_t validate_tables(const et_kernel_call_v1 *call, enum primitive op,
                               int backward, et_kernel_error *error) {
  const char *operation = call->request->operation;
  const uint64_t *s = call->request->shape;
  const uint64_t q[] = {s[0], s[1], op == ATTENTION ? s[3] : 0u,
                        op == ATTENTION ? s[5] : 0u};
  const uint64_t kv[] = {s[0], s[2], op == ATTENTION ? s[4] : 0u, q[3]};
  const uint64_t query_pos[] = {s[0], q[2]};
  const uint64_t key_pos[] = {s[0], kv[2]};
  const uint64_t mask[] = {s[0], q[2], kv[2]};
  const tensor_table in = inputs(call);
  const tensor_table out = outputs(call);
  const tensor_schema ntd = {F32, 3u, s};
  const tensor_schema gamma = {F32, 1u, s + 2u};
  const tensor_schema query = {F32, 4u, q};
  const tensor_schema key_value = {F32, 4u, kv};
  const tensor_schema rms_in[] = {ntd, gamma, {F32, 0u, NULL}, ntd};
  const tensor_schema rms_out[] = {ntd, gamma};
  const tensor_schema swi_in[] = {ntd, ntd, ntd};
  const tensor_schema attention_in[] = {query,
                                        key_value,
                                        key_value,
                                        {I64, 2u, query_pos},
                                        {I64, 2u, key_pos},
                                        {BOOL, 3u, mask},
                                        {F32, 1u, s + 1u},
                                        {I64, 0u, NULL},
                                        query};
  const tensor_schema attention_out[] = {query, key_value, key_value};
  int32_t status;

  switch (op) {
  case RMSNORM:
    status = validate_table(in, backward ? 4u : 3u, rms_in, operation, error);
    if (status == 0)
      status =
          validate_table(out, backward ? 2u : 1u, rms_out, operation, error);
    break;
  case SWIGLU:
    status = validate_table(in, backward ? 3u : 2u, swi_in, operation, error);
    if (status == 0)
      status =
          validate_table(out, backward ? 2u : 1u, swi_in, operation, error);
    break;
  case ATTENTION:
    status =
        validate_table(in, backward ? 9u : 8u, attention_in, operation, error);
    if (status == 0)
      status = validate_table(out, backward ? 3u : 1u, attention_out, operation,
                              error);
    break;
  default:
    return rejected(error, operation, "unknown primitive");
  }
  if (status != 0)
    return status;

  for (size_t i = 0; i < in.count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (overlap(view_at(in, i), view_at(in, j)))
        return rejected(error, operation, "input storage overlaps");
    }
  }
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(out, i);
    if (output_aliases_metadata(view, call, in, out, error))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases live K1 metadata");
    for (size_t j = 0; j < in.count; ++j) {
      if (overlap(view, view_at(in, j)))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases input");
    }
    for (size_t j = 0; j < i; ++j) {
      if (overlap(view, view_at(out, j)))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases another output");
    }
  }
  return 0;
}

static int fp_environment_supported(void) {
  uint16_t x87;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round to nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int finite_inputs(tensor_table in) {
  for (size_t i = 0; i < in.count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(in, i);
    if (!exact_text(view->dtype, "f32"))
      continue;
    const float *data = view->data;
    for (size_t j = 0; j < view->byte_length / sizeof(*data); ++j) {
      if (!isfinite(data[j]))
        return 0;
    }
  }
  return 1;
}

static int rmsnorm(tensor_table in, const uint64_t *s, int backward,
                   result_buffers result) {
  const float *x = view_at(in, 0u)->data;
  const float *gamma = view_at(in, 1u)->data;
  const float epsilon = *(const float *)view_at(in, 2u)->data;
  const float *upstream = backward ? view_at(in, 3u)->data : NULL;
  const size_t width = s[2];
  if (!(epsilon > 0.0f))
    return 0;
  for (size_t row = 0; row < s[0] * s[1]; ++row) {
    const size_t start = row * width;
    float sum = 0.0f;
    float dot = 0.0f;
    for (size_t d = 0; d < width; ++d) {
      const float value = x[start + d];
      sum = sum + value * value;
      if (!isfinite(sum))
        return 0;
    }
    const float mean_sq = sum / (float)width;
    const float stabilized = mean_sq + epsilon;
    if (!isfinite(stabilized))
      return 0;
    const float rstd = 1.0f / sqrtf(stabilized);
    if (!isfinite(rstd))
      return 0;
    if (backward) {
      for (size_t d = 0; d < width; ++d) {
        const size_t i = start + d;
        const float norm = x[i] * rstd;
        const float h = upstream[i] * gamma[d];
        dot = dot + h * norm;
        result[1][d] = result[1][d] + upstream[i] * norm;
        if (!isfinite(dot) || !isfinite(result[1][d]))
          return 0;
      }
    }
    const float mean = dot / (float)width;
    for (size_t d = 0; d < width; ++d) {
      const size_t i = start + d;
      const float norm = x[i] * rstd;
      result[0][i] = backward ? (upstream[i] * gamma[d] - norm * mean) * rstd
                              : norm * gamma[d];
      if (!isfinite(result[0][i]))
        return 0;
    }
  }
  return 1;
}

static int swiglu(tensor_table in, const uint64_t *s, int backward,
                  result_buffers result) {
  const float *gate = view_at(in, 0u)->data;
  const float *up = view_at(in, 1u)->data;
  const float *upstream = backward ? view_at(in, 2u)->data : NULL;
  for (size_t i = 0; i < s[0] * s[1] * s[2]; ++i) {
    const float exp_value = expf(-gate[i]);
    const float denominator = 1.0f + exp_value;
    if (!isfinite(exp_value) || !isfinite(denominator))
      return 0;
    const float sig = 1.0f / denominator;
    const float silu = gate[i] * sig;
    if (!isfinite(sig) || !isfinite(silu))
      return 0;
    if (backward) {
      const float complement = 1.0f - sig;
      const float scaled_gate = gate[i] * complement;
      const float correction = 1.0f + scaled_gate;
      const float slope = sig * correction;
      const float weighted = upstream[i] * up[i];
      if (!isfinite(complement) || !isfinite(scaled_gate) ||
          !isfinite(correction) || !isfinite(slope) || !isfinite(weighted))
        return 0;
      result[0][i] = weighted * slope;
      result[1][i] = upstream[i] * silu;
      if (!isfinite(result[1][i]))
        return 0;
    } else {
      result[0][i] = silu * up[i];
    }
    if (!isfinite(result[0][i]))
      return 0;
  }
  return 1;
}

typedef struct attention {
  size_t n, hq, hkv, tq, tk, dh;
  const float *q, *k, *v, *slopes, *upstream;
  const int64_t *query_pos, *key_pos;
  const uint8_t *mask;
  int64_t window;
  float scale;
} attention;

static int valid_positions(const int64_t *positions, size_t n, size_t t) {
  for (size_t batch = 0; batch < n; ++batch) {
    int64_t previous = -1;
    for (size_t token = 0; token < t; ++token) {
      const int64_t position = positions[batch * t + token];
      if (position <= previous || position > ET_MOD5_MAX_EXACT_POSITION)
        return 0;
      previous = position;
    }
  }
  return 1;
}

static int finite_dot(const float *a, const float *b, size_t count,
                      float *dot) {
  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    sum = sum + a[i] * b[i];
    if (!isfinite(sum))
      return 0;
  }
  *dot = sum;
  return 1;
}

static int attention_row(const attention *a, size_t n, size_t h, size_t q,
                         result_buffers result) {
  const size_t head = h / (a->hq / a->hkv);
  const size_t qi = ((n * a->hq + h) * a->tq + q) * a->dh;
  const int64_t query_pos = a->query_pos[n * a->tq + q];
  size_t indices[MAX_KEYS];
  float probabilities[MAX_KEYS];
  float dp[MAX_KEYS];
  size_t count = 0u;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float mean_dp = 0.0f;

  for (size_t k = 0; k < a->tk; ++k) {
    /* Position validation proves that subtraction is safe and the cast exact.
     */
    const int64_t delta = a->key_pos[n * a->tk + k] - query_pos;
    if (a->mask[(n * a->tq + q) * a->tk + k] == 0u || delta > 0 ||
        -delta >= a->window)
      continue;

    const size_t ki = ((n * a->hkv + head) * a->tk + k) * a->dh;
    float dot;
    if (!finite_dot(a->q + qi, a->k + ki, a->dh, &dot))
      return 0;
    const float scaled = dot * a->scale;
    const float bias = a->slopes[h] * (float)delta;
    const float score = scaled + bias;
    if (!isfinite(score))
      return 0;
    if (score > maximum)
      maximum = score;
    indices[count] = ki;
    probabilities[count] = score;
    ++count;
  }

  for (size_t k = 0; k < count; ++k) {
    const float shifted = probabilities[k] - maximum;
    if (!isfinite(shifted))
      return 0;
    const float weight = expf(shifted);
    denominator = denominator + weight;
    if (!isfinite(denominator))
      return 0;
    probabilities[k] = weight;
  }

  for (size_t k = 0; k < count; ++k) {
    const size_t ki = indices[k];
    const float probability = probabilities[k] / denominator;
    if (!isfinite(probability))
      return 0;
    probabilities[k] = probability;
    if (a->upstream != NULL) {
      if (!finite_dot(a->upstream + qi, a->v + ki, a->dh, &dp[k]))
        return 0;
      mean_dp = mean_dp + probability * dp[k];
      if (!isfinite(mean_dp))
        return 0;
    } else {
      for (size_t d = 0; d < a->dh; ++d) {
        result[0][qi + d] = result[0][qi + d] + probability * a->v[ki + d];
        if (!isfinite(result[0][qi + d]))
          return 0;
      }
    }
  }

  if (a->upstream != NULL) {
    for (size_t k = 0; k < count; ++k) {
      const size_t ki = indices[k];
      const float difference = dp[k] - mean_dp;
      const float ds = probabilities[k] * difference;
      const float scaled_ds = ds * a->scale;
      if (!isfinite(scaled_ds))
        return 0;
      for (size_t d = 0; d < a->dh; ++d) {
        result[0][qi + d] = result[0][qi + d] + scaled_ds * a->k[ki + d];
        result[1][ki + d] = result[1][ki + d] + scaled_ds * a->q[qi + d];
        result[2][ki + d] =
            result[2][ki + d] + probabilities[k] * a->upstream[qi + d];
        if (!isfinite(result[0][qi + d]) || !isfinite(result[1][ki + d]) ||
            !isfinite(result[2][ki + d]))
          return 0;
      }
    }
  }
  return 1;
}

static int variant_attention(tensor_table in, const uint64_t *s, int backward,
                             result_buffers result) {
  const attention a = {.n = s[0],
                       .hq = s[1],
                       .hkv = s[2],
                       .tq = s[3],
                       .tk = s[4],
                       .dh = s[5],
                       .q = view_at(in, 0u)->data,
                       .k = view_at(in, 1u)->data,
                       .v = view_at(in, 2u)->data,
                       .query_pos = view_at(in, 3u)->data,
                       .key_pos = view_at(in, 4u)->data,
                       .mask = view_at(in, 5u)->data,
                       .slopes = view_at(in, 6u)->data,
                       .window = *(const int64_t *)view_at(in, 7u)->data,
                       .upstream = backward ? view_at(in, 8u)->data : NULL,
                       .scale = 1.0f / sqrtf((float)s[5])};
  if (a.window < 1 || !valid_positions(a.query_pos, a.n, a.tq) ||
      !valid_positions(a.key_pos, a.n, a.tk))
    return 0;
  for (size_t i = 0; i < a.n * a.tq * a.tk; ++i) {
    if (a.mask[i] > 1u)
      return 0;
  }
  for (size_t n = 0; n < a.n; ++n) {
    for (size_t h = 0; h < a.hq; ++h) {
      for (size_t q = 0; q < a.tq; ++q) {
        if (!attention_row(&a, n, h, q, result))
          return 0;
      }
    }
  }
  return 1;
}

static int find_operation(const et_kernel_call_v1 *call, enum primitive *op,
                          int *backward) {
  for (size_t i = 0; i < COUNT(capabilities); ++i) {
    const et_kernel_capability_v1 *cap = &capabilities[i];
    if (!exact_text(call->capability, cap->name))
      continue;
    for (size_t j = 0; j < cap->operation_count; ++j) {
      if (exact_text(call->request->operation, cap->operations[j])) {
        *op = (enum primitive)i;
        *backward = j == 1u;
        return 1;
      }
    }
  }
  return 0;
}

static int request_admitted(const et_kernel_request_v1 *request,
                            const et_kernel_capability_v1 *capability) {
  if (!exact_text(request->dtype, "f32") ||
      !exact_text(request->device, "cpu") ||
      request->rank != capability->shape_ranges[0].rank ||
      !aligned_span(request->shape, _Alignof(uint64_t),
                    request->rank * sizeof(*request->shape)))
    return 0;
  for (size_t i = 0; i < capability->shape_range_count; ++i) {
    const et_kernel_shape_range_v1 *row = &capability->shape_ranges[i];
    size_t d = 0;
    while (d < row->rank && row->dimensions[d].minimum == request->shape[d])
      ++d;
    if (d == row->rank)
      return 1;
  }
  return 0;
}

static int evaluate(const et_kernel_call_v1 *call, enum primitive op,
                    int backward, result_buffers result) {
  const tensor_table in = inputs(call);
  const uint64_t *shape = call->request->shape;
  if (!finite_inputs(in))
    return 0;
  switch (op) {
  case RMSNORM:
    return rmsnorm(in, shape, backward, result);
  case SWIGLU:
    return swiglu(in, shape, backward, result);
  case ATTENTION:
    return variant_attention(in, shape, backward, result);
  }
  return 0;
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  enum primitive op;
  int backward;
  if (call == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.provider",
                "call is required");
  if (!aligned_span(call, _Alignof(et_kernel_call_v1),
                    ET_KERNEL_CALL_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.provider",
                "invalid call descriptor span or alignment");
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.provider",
                "truncated call descriptor");
  const et_kernel_request_v1 *request = call->request;
  if (request == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.provider",
                "request is required");
  if (!aligned_span(request, _Alignof(et_kernel_request_v1),
                    ET_KERNEL_REQUEST_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.provider",
                "invalid request descriptor span or alignment");
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.provider",
                "truncated request descriptor");
  if (!find_operation(call, &op, &backward) ||
      !request_admitted(request, &capabilities[op]))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, "mod5.provider",
                "request is outside the verified MOD5 rows");

  const char *operation = request->operation;
  if (request->deterministic > 1u)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "invalid deterministic request flag");

  const int32_t status = validate_tables(call, op, backward, error);
  if (status != 0)
    return status;

  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0)
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");

  result_buffers result = {{0.0f}};
  const int saved_errno = errno;
  const int valid = evaluate(call, op, backward, result);
  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  if (!valid)
    return rejected(error, operation,
                    "invalid numeric input or nonfinite evaluation");
  et_kernel_error_clear(error);
  return 0;
}

static void invoke_call(const et_kernel_call_v1 *call) {
  enum primitive op;
  int backward;
  result_buffers result = {{0.0f}};
  /* K1 keeps descriptors, data and FP controls stable between both phases. */
  if (!find_operation(call, &op, &backward) ||
      !evaluate(call, op, backward, result))
    return;
  const tensor_table out = outputs(call);
  void *destinations[MAX_OUTPUTS];
  size_t lengths[MAX_OUTPUTS];
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(out, i);
    destinations[i] = view->data;
    lengths[i] = view->byte_length;
  }
  for (size_t i = 0; i < out.count; ++i)
    memcpy(destinations[i], result[i], lengths[i]);
}

static const et_kernel_provider_v1 provider = {
    sizeof(et_kernel_provider_v1),
    ET_KERNEL_ABI_MAJOR,
    ET_KERNEL_ABI_MINOR,
    ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES,
    IMPLEMENTATION,
    "1.0",
    EVIDENCE,
    COUNT(capabilities),
    sizeof(et_kernel_capability_v1),
    sizeof(capabilities),
    capabilities,
    validate_call,
    invoke_call};

const et_kernel_provider_v1 *et_mod5_kernel_provider_v1(void) {
  return &provider;
}
