#include "eshkol_transformer/mod5_rope_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 RoPE v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MOD5 RoPE requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 RoPE requires standard IEEE-754 binary32/binary64 eval");
_Static_assert(ET_MOD5_ROPE_MAX_POSITION < (INT64_C(1) << FLT_MANT_DIG),
               "bounded positions must fit exactly in binary32 and binary64");

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define ROPE_ROW(dh)                                                           \
  {                                                                            \
    4u, (const et_kernel_dimension_range_v1[]) {                               \
      {ET_MOD5_ROPE_MIN_BATCH, ET_MOD5_ROPE_MAX_BATCH, 0u, {0}},               \
          {ET_MOD5_ROPE_MIN_HEADS, ET_MOD5_ROPE_MAX_HEADS, 0u, {0}},           \
          {ET_MOD5_ROPE_MIN_TOKENS, ET_MOD5_ROPE_MAX_TOKENS, 0u, {0}},         \
          {(dh), (dh), 0u, {0}}                                                \
    }                                                                          \
  }

#define IMPLEMENTATION "mod5.rope.cpu-f32.serial"
#define EVIDENCE "MOD5:cpu-f32-rope-v1"

static const et_kernel_shape_range_v1 rope_rows[] = {
    ROPE_ROW(2u),  ROPE_ROW(4u),  ROPE_ROW(6u),  ROPE_ROW(8u),
    ROPE_ROW(10u), ROPE_ROW(12u), ROPE_ROW(14u), ROPE_ROW(16u)};

static const char *const rope_operations[] = {"rope.forward", "rope.backward"};
static const char *const rope_dtypes[] = {"f32"};
static const char *const rope_devices[] = {"cpu"};

static const et_kernel_capability_v1 capabilities[] = {
    {sizeof(et_kernel_capability_v1), "kernel.rope",
     ET_KERNEL_CAPABILITY_VERIFIED, IMPLEMENTATION, "1.0", EVIDENCE, 1u, {0},
     COUNT(rope_operations), rope_operations, COUNT(rope_dtypes), rope_dtypes,
     COUNT(rope_devices), rope_devices, COUNT(rope_rows), rope_rows}};

typedef struct tensor_table {
  size_t count;
  size_t stride;
  size_t bytes;
  const void *base;
} tensor_table;

enum dtype { F32, I64 };
typedef struct tensor_schema {
  enum dtype dtype;
  size_t rank;
  const uint64_t *shape;
} tensor_schema;

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
    const int saved_errno = errno;
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
    errno = saved_errno;
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

static int shape_admitted(uint64_t n, uint64_t h, uint64_t t, uint64_t dh) {
  return n >= ET_MOD5_ROPE_MIN_BATCH && n <= ET_MOD5_ROPE_MAX_BATCH &&
         h >= ET_MOD5_ROPE_MIN_HEADS && h <= ET_MOD5_ROPE_MAX_HEADS &&
         t >= ET_MOD5_ROPE_MIN_TOKENS && t <= ET_MOD5_ROPE_MAX_TOKENS &&
         dh >= ET_MOD5_ROPE_MIN_HEAD_DIM && dh <= ET_MOD5_ROPE_MAX_HEAD_DIM &&
         (dh & 1u) == 0u;
}

static int fp_environment_supported(void) {
  uint16_t x87;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round to nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int rope_evaluate(size_t batch, size_t heads, size_t tokens,
                         size_t head_dim, const float *in,
                         const int64_t *positions, double theta, double scale,
                         int backward, float *out) {
  if (!isfinite(theta) || !(theta > 1.0) || !isfinite(scale) ||
      !(scale > 0.0))
    return 0;

  const size_t half = head_dim / 2u;
  double inv_freq[ET_MOD5_ROPE_MAX_HEAD_DIM / 2u];
  for (size_t k = 0; k < half; ++k) {
    const double exponent = (-2.0 * (double)k) / (double)head_dim;
    const double freq = pow(theta, exponent);
    if (!isfinite(freq))
      return 0;
    inv_freq[k] = freq;
  }

  for (size_t n = 0; n < batch; ++n) {
    for (size_t t = 0; t < tokens; ++t) {
      const int64_t pos = positions[n * tokens + t];
      if (pos < 0 || pos > ET_MOD5_ROPE_MAX_POSITION)
        return 0;
      const double scaled_pos = (double)pos / scale;
      if (!isfinite(scaled_pos))
        return 0;

      float cos_table[ET_MOD5_ROPE_MAX_HEAD_DIM / 2u];
      float sin_table[ET_MOD5_ROPE_MAX_HEAD_DIM / 2u];
      for (size_t k = 0; k < half; ++k) {
        const double phi = scaled_pos * inv_freq[k];
        if (!isfinite(phi))
          return 0;
        const double cd = cos(phi);
        const double sd = sin(phi);
        if (!isfinite(cd) || !isfinite(sd))
          return 0;
        const float cf = (float)cd;
        const float sf = (float)sd;
        if (!isfinite(cf) || !isfinite(sf))
          return 0;
        cos_table[k] = cf;
        sin_table[k] = sf;
      }

      for (size_t h = 0; h < heads; ++h) {
        const size_t base = ((n * heads + h) * tokens + t) * head_dim;
        for (size_t k = 0; k < half; ++k) {
          const size_t i0 = base + 2u * k;
          const size_t i1 = i0 + 1u;
          const float v0 = in[i0];
          const float v1 = in[i1];
          if (!isfinite(v0) || !isfinite(v1))
            return 0;
          const float c = cos_table[k];
          const float s = sin_table[k];
          const float p00 = v0 * c;
          const float p11 = v1 * s;
          const float p01 = v0 * s;
          const float p10 = v1 * c;
          if (!isfinite(p00) || !isfinite(p11) || !isfinite(p01) ||
              !isfinite(p10))
            return 0;
          const float r0 = backward ? (p00 + p11) : (p00 - p11);
          const float r1 = backward ? (-p01 + p10) : (p01 + p10);
          if (!isfinite(r0) || !isfinite(r1))
            return 0;
          if (out != NULL) {
            out[i0] = r0;
            out[i1] = r1;
          }
        }
      }
    }
  }
  return 1;
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
  static const char *const names[] = {[F32] = "f32", [I64] = "i64"};
  static const size_t sizes[] = {[F32] = sizeof(float),
                                 [I64] = sizeof(int64_t)};
  static const size_t alignments[] = {[F32] = _Alignof(float),
                                      [I64] = _Alignof(int64_t)};
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
                "view dtype does not match rope schema");
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
                "view rank does not match rope schema");
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
                  "view extent does not match rope schema");
    }
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

static int32_t validate_tables(const et_kernel_call_v1 *call,
                               et_kernel_error *error) {
  const char *operation = call->request->operation;
  const uint64_t *s = call->request->shape;
  const uint64_t pos_shape[] = {s[0], s[2]};
  const tensor_table in = inputs(call);
  const tensor_table out = outputs(call);
  const tensor_schema in_schema[] = {{F32, 4u, s},
                                     {I64, 2u, pos_shape},
                                     {F32, 0u, NULL},
                                     {F32, 0u, NULL}};
  const tensor_schema out_schema[] = {{F32, 4u, s}};

  int32_t status = validate_table(in, 4u, in_schema, operation, error);
  if (status != 0)
    return status;
  status = validate_table(out, 1u, out_schema, operation, error);
  if (status != 0)
    return status;

  for (size_t i = 0; i < in.count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (overlap(view_at(in, i), view_at(in, j)))
        return rejected(error, operation, "input storage overlaps");
    }
  }
  const et_kernel_tensor_view_v1 *out_view = view_at(out, 0u);
  if (output_aliases_metadata(out_view, call, in, out, error)) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "output aliases live K1 metadata");
  }
  for (size_t j = 0; j < in.count; ++j) {
    if (overlap(out_view, view_at(in, j))) {
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases input");
    }
  }
  return 0;
}

static int find_operation(const et_kernel_call_v1 *call, int *backward) {
  if (!exact_text(call->capability, "kernel.rope"))
    return 0;
  if (exact_text(call->request->operation, "rope.forward")) {
    *backward = 0;
    return 1;
  }
  if (exact_text(call->request->operation, "rope.backward")) {
    *backward = 1;
    return 1;
  }
  return 0;
}

static int request_admitted(const et_kernel_request_v1 *request) {
  if (!exact_text(request->dtype, "f32") ||
      !exact_text(request->device, "cpu") || request->rank != 4u ||
      !aligned_span(request->shape, _Alignof(uint64_t),
                    4u * sizeof(*request->shape)))
    return 0;
  return shape_admitted(request->shape[0], request->shape[1], request->shape[2],
                        request->shape[3]);
}

static int evaluate_call(const et_kernel_call_v1 *call, int backward,
                         float *out) {
  const tensor_table in = inputs(call);
  const uint64_t *s = call->request->shape;
  const float *x = (const float *)view_at(in, 0u)->data;
  const int64_t *positions = (const int64_t *)view_at(in, 1u)->data;
  const float theta_f32 = *(const float *)view_at(in, 2u)->data;
  const float scale_f32 = *(const float *)view_at(in, 3u)->data;
  if (!isfinite(theta_f32) || !isfinite(scale_f32))
    return 0;
  return rope_evaluate((size_t)s[0], (size_t)s[1], (size_t)s[2], (size_t)s[3],
                       x, positions, (double)theta_f32, (double)scale_f32,
                       backward, out);
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  int backward = 0;
  if (call == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.rope.provider",
                "call is required");
  if (!aligned_span(call, _Alignof(et_kernel_call_v1),
                    ET_KERNEL_CALL_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.rope.provider",
                "invalid call descriptor span or alignment");
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.rope.provider",
                "truncated call descriptor");

  const et_kernel_request_v1 *request = call->request;
  if (request == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.rope.provider",
                "request is required");
  if (!aligned_span(request, _Alignof(et_kernel_request_v1),
                    ET_KERNEL_REQUEST_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.rope.provider",
                "invalid request descriptor span or alignment");
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.rope.provider",
                "truncated request descriptor");
  if (!find_operation(call, &backward) || !request_admitted(request))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, "mod5.rope.provider",
                "request is outside the verified MOD5 RoPE rows");

  const char *operation = request->operation;
  if (request->deterministic > 1u)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "invalid deterministic request flag");

  const int32_t status = validate_tables(call, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = evaluate_call(call, backward, NULL);
  const int restored = fesetenv(&environment);
  int32_t numeric_status = 0;
  if (restored != 0)
    numeric_status = fail(error, ET_KERNEL_ERROR_INTERNAL,
                          ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                          "floating-point environment could not be restored");
  else if (!valid)
    numeric_status = rejected(error, operation,
                              "invalid rope numeric input or nonfinite result");
  else
    et_kernel_error_clear(error);
  errno = saved_errno;
  return numeric_status;
}

static void invoke_call(const et_kernel_call_v1 *call) {
  int backward = 0;
  if (!find_operation(call, &backward))
    return;
  const tensor_table out = outputs(call);
  float *destination = (float *)view_at(out, 0u)->data;
  (void)evaluate_call(call, backward, destination);
}

static int32_t direct_rope(size_t batch, size_t heads, size_t tokens,
                           size_t head_dim, const void *in,
                           const void *positions, double theta, double scale,
                           int backward, void *out, et_kernel_error *error) {
  const char *operation = backward ? "rope.backward" : "rope.forward";
  if (!shape_admitted((uint64_t)batch, (uint64_t)heads, (uint64_t)tokens,
                      (uint64_t)head_dim)) {
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "shape is outside verified MOD5 RoPE domain");
  }
  if (in == NULL || positions == NULL || out == NULL) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                "input, positions, and output pointers are required");
  }
  const size_t data_bytes = batch * heads * tokens * head_dim * sizeof(float);
  const size_t pos_bytes = batch * tokens * sizeof(int64_t);
  if (!aligned_span(in, _Alignof(float), data_bytes) ||
      !aligned_span(positions, _Alignof(int64_t), pos_bytes) ||
      !aligned_span(out, _Alignof(float), data_bytes)) {
    return rejected(error, operation, "invalid buffer alignment or span");
  }
  if (spans_overlap(in, data_bytes, positions, pos_bytes)) {
    return rejected(error, operation, "input and positions storage overlap");
  }
  if (spans_overlap(out, data_bytes, in, data_bytes) ||
      spans_overlap(out, data_bytes, positions, pos_bytes) ||
      (error != NULL && spans_overlap(out, data_bytes, error, sizeof(*error)))) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "output aliases input or error storage");
  }

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const float *in_f32 = (const float *)in;
  const int64_t *pos_i64 = (const int64_t *)positions;
  const int valid = rope_evaluate(batch, heads, tokens, head_dim, in_f32,
                                  pos_i64, theta, scale, backward, NULL);
  const int restored = fesetenv(&environment);
  if (restored != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  }
  if (!valid) {
    errno = saved_errno;
    return rejected(error, operation,
                    "invalid rope numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)rope_evaluate(batch, heads, tokens, head_dim, in_f32, pos_i64, theta,
                      scale, backward, (float *)out);
  et_kernel_error_clear(error);
  return 0;
}

int32_t et_mod5_rope_forward_f32(size_t batch, size_t heads, size_t tokens,
                                 size_t head_dim, const void *x,
                                 const void *positions, double theta,
                                 double scale, void *y,
                                 et_kernel_error *error) {
  return direct_rope(batch, heads, tokens, head_dim, x, positions, theta, scale,
                     0, y, error);
}

int32_t et_mod5_rope_backward_f32(size_t batch, size_t heads, size_t tokens,
                                  size_t head_dim, const void *dy,
                                  const void *positions, double theta,
                                  double scale, void *dx,
                                  et_kernel_error *error) {
  return direct_rope(batch, heads, tokens, head_dim, dy, positions, theta,
                     scale, 1, dx, error);
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

const et_kernel_provider_v1 *et_mod5_rope_kernel_provider_v1(void) {
  return &provider;
}
