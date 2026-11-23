#include "eshkol_transformer/pa4_wider_attention_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#if !defined(__x86_64__)
#error "PA4-Wider requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && FLT_RADIX == 2 &&
                   FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "PA4-Wider requires IEEE binary32 evaluation");

enum { MAX_BATCH = 2, MAX_HEADS = 4, MAX_QUERIES = 64, MAX_WIDTH = 16 };
enum { Q, K, V, QUERY_POSITION, KEY_POSITION, MASK, UPSTREAM };
enum { FORWARD, BACKWARD };

static const char capability_name[] = "kernel.causal-attention";
static const char *const operations[] = {"causal-attention.forward",
                                         "causal-attention.backward"};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define DIM(low, high) {low, high, 0u, {0}}
#define HEAD_PAIR(hq, hkv)                                                     \
  {                                                                            \
    6u, (const et_kernel_dimension_range_v1[]) {                               \
      DIM(1, MAX_BATCH), DIM(hq, hq), DIM(hkv, hkv), DIM(1, MAX_QUERIES),      \
          DIM(1, ET_PA4_WIDER_MAX_KEYS), DIM(1, MAX_WIDTH)                     \
    }                                                                          \
  }
static const et_kernel_shape_range_v1 ranges[] = {
    HEAD_PAIR(1, 1), HEAD_PAIR(2, 1), HEAD_PAIR(2, 2), HEAD_PAIR(3, 1),
    HEAD_PAIR(3, 3), HEAD_PAIR(4, 1), HEAD_PAIR(4, 2), HEAD_PAIR(4, 4)};
#undef HEAD_PAIR
#undef DIM

typedef struct {
  const void *start;
  size_t bytes;
} span;

typedef struct {
  span storage;
  size_t count;
  size_t stride;
} table;

typedef struct {
  size_t n, hq, hkv, tq, tk, dh;
} shape;

typedef struct {
  const float *q, *k, *v, *upstream;
  const int64_t *qp, *kp;
  const uint8_t *mask;
} data;

typedef struct {
  const float *q, *k, *v, *upstream;
  const int64_t *kp;
  const uint8_t *mask;
  int64_t qp;
  size_t tk, dh;
  float scale;
} row;

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int overlaps(span a, span b) {
  const uintptr_t x = (uintptr_t)a.start;
  const uintptr_t y = (uintptr_t)b.start;
  return x != 0u && y != 0u && a.bytes != 0u && b.bytes != 0u &&
         (x <= y ? y - x < a.bytes : x - y < b.bytes);
}

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int overlaps_text(span target, const char *text) {
  const size_t maximum = ET_KERNEL_MAX_SYMBOL_BYTES + 1u;
  if (!overlaps(target, (span){text, maximum}))
    return 0;
  size_t length = 0u;
  while (length < maximum && text[length] != '\0')
    ++length;
  return overlaps(target,
                  (span){text, length < maximum ? length + 1u : length});
}

static int32_t fail(et_kernel_error *error, et_kernel_error_category category,
                    et_kernel_error_code code, const char *operation,
                    const char *message) {
  if (error != NULL) {
    *error = (et_kernel_error){.category = category, .code = code};
    memcpy(error->operation, operation, strlen(operation) + 1u);
    memcpy(error->message, message, strlen(message) + 1u);
  }
  return (int32_t)category;
}

static table input_table(const et_kernel_call_v1 *call) {
  return (table){
      {call->inputs, call->input_bytes}, call->input_count, call->input_stride};
}

static table output_table(const et_kernel_call_v1 *call) {
  return (table){{call->outputs, call->output_bytes},
                 call->output_count,
                 call->output_stride};
}

static const et_kernel_tensor_view_v1 *view_at(table views, size_t index) {
  return (const void *)((const unsigned char *)views.storage.start +
                        index * views.stride);
}

static span view_data(const et_kernel_tensor_view_v1 *view) {
  return (span){view->data, view->byte_length};
}

static int table_fits(table views) {
  const size_t alignment = _Alignof(et_kernel_tensor_view_v1);
  return views.stride >= ET_KERNEL_TENSOR_VIEW_V1_0_SIZE &&
         views.stride % alignment == 0u &&
         views.count <= ET_KERNEL_MAX_TENSORS &&
         views.count <= SIZE_MAX / views.stride &&
         views.storage.bytes == views.count * views.stride &&
         aligned_span(views.storage.start, alignment, views.storage.bytes);
}

static span shape_span(const uint64_t *extents, size_t rank) {
  return (span){extents,
                rank <= ET_KERNEL_MAX_RANK ? rank * sizeof(*extents) : 0u};
}

static int aliases_table(span target, table views, size_t max_count,
                         int include_data) {
  if (views.storage.start != NULL && views.storage.bytes != 0u &&
      overlaps(target, views.storage))
    return 1;
  const size_t alignment = _Alignof(et_kernel_tensor_view_v1);
  if (views.stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      views.stride % alignment != 0u ||
      !aligned_span(views.storage.start, alignment, views.storage.bytes))
    return 0;
  const size_t readable =
      views.storage.bytes >= ET_KERNEL_TENSOR_VIEW_V1_0_SIZE
          ? 1u + (views.storage.bytes - ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) /
                     views.stride
          : 0u;
  size_t count = views.count < readable ? views.count : readable;
  if (count > max_count)
    count = max_count;
  for (size_t i = 0; i < count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(views, i);
    if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
        view->struct_size > views.stride)
      continue;
    if (overlaps(target, shape_span(view->shape, view->rank)) ||
        overlaps_text(target, view->dtype) ||
        overlaps_text(target, view->device) ||
        (include_data && overlaps(target, view_data(view))))
      return 1;
  }
  return 0;
}

/* The optional diagnostic is also writable. When it aliases a borrowed span,
   direct validation rejects without writing a diagnostic to that address. */
static int aliases_call(span target, const et_kernel_call_v1 *call,
                        int include_data) {
  if (overlaps(target, (span){call, sizeof(size_t)}))
    return 1;
  if (!aligned_span(call, _Alignof(et_kernel_call_v1), sizeof(size_t)))
    return 0;
  if (call->struct_size >= sizeof(size_t) &&
      aligned_span(call, _Alignof(et_kernel_call_v1), call->struct_size) &&
      overlaps(target, (span){call, call->struct_size}))
    return 1;
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      !aligned_span(call, _Alignof(et_kernel_call_v1), call->struct_size))
    return 0;
  const et_kernel_request_v1 *request = call->request;
  size_t input_limit = 7u;
  if (request != NULL &&
      aligned_span(request, _Alignof(et_kernel_request_v1), sizeof(size_t)) &&
      request->struct_size >= ET_KERNEL_REQUEST_V1_0_SIZE &&
      aligned_span(request, _Alignof(et_kernel_request_v1),
                   request->struct_size) &&
      exact_text(request->operation, operations[FORWARD]))
    input_limit = 6u;
  if (overlaps_text(target, call->capability) ||
      aliases_table(target, input_table(call), input_limit, include_data) ||
      aliases_table(target, output_table(call), ET_KERNEL_MAX_TENSORS,
                    include_data))
    return 1;

  if (overlaps(target, (span){request, sizeof(size_t)}))
    return 1;
  if (!aligned_span(request, _Alignof(et_kernel_request_v1), sizeof(size_t)))
    return 0;
  if (request->struct_size >= sizeof(size_t) &&
      aligned_span(request, _Alignof(et_kernel_request_v1),
                   request->struct_size) &&
      overlaps(target, (span){request, request->struct_size}))
    return 1;
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE ||
      !aligned_span(request, _Alignof(et_kernel_request_v1),
                    request->struct_size))
    return 0;
  return overlaps(target, shape_span(request->shape, request->rank)) ||
         overlaps_text(target, request->operation) ||
         overlaps_text(target, request->dtype) ||
         overlaps_text(target, request->device);
}

static shape read_shape(const uint64_t *s) {
  return (shape){s[0], s[1], s[2], s[3], s[4], s[5]};
}

static int shape_supported(shape s) {
  return s.n >= 1u && s.n <= MAX_BATCH && s.hq >= 1u && s.hq <= MAX_HEADS &&
         s.hkv >= 1u && s.hkv <= s.hq && s.hq % s.hkv == 0u && s.tq >= 1u &&
         s.tq <= MAX_QUERIES && s.tk >= 1u && s.tk <= ET_PA4_WIDER_MAX_KEYS &&
         s.dh >= 1u && s.dh <= MAX_WIDTH;
}

enum dtype { F32, I64, BOOL };
typedef struct {
  enum dtype dtype;
  size_t rank;
  const uint64_t *shape;
} schema;

static int32_t validate_table(table views, size_t count, const schema *schemas,
                              const char *op, et_kernel_error *error) {
  static const char *const names[] = {"f32", "i64", "bool"};
  static const size_t sizes[] = {sizeof(float), sizeof(int64_t),
                                 sizeof(uint8_t)};
  static const size_t alignments[] = {_Alignof(float), _Alignof(int64_t),
                                      _Alignof(uint8_t)};

  if (views.count != count || views.storage.start == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op,
                "invalid tensor table count or base");
  if (views.stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, op,
                "truncated tensor table stride");
  if (!table_fits(views))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op,
                "invalid tensor table span or alignment");

  for (size_t i = 0; i < count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(views, i);
    const schema *expected = schemas + i;
    const enum dtype type = expected->dtype;
    if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
        view->struct_size > views.stride)
      return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                  ET_KERNEL_CODE_INVALID_STRUCT_SIZE, op,
                  "invalid tensor view prefix");
    if (!exact_text(view->dtype, names[type]))
      return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_TEXT, op,
                  "tensor dtype differs from schema");
    if (!exact_text(view->device, "cpu"))
      return fail(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_TEXT, op, "tensor device must be CPU");
    if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
        view->offset_bytes != 0u)
      return fail(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                  ET_KERNEL_CODE_INVALID_BUFFER, op,
                  "tensor must be dense and zero-offset");
    if (view->rank != expected->rank ||
        !aligned_span(view->shape, _Alignof(uint64_t),
                      expected->rank * sizeof(uint64_t)))
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, op,
                  "invalid tensor rank or shape span");
    size_t bytes = sizes[type];
    for (size_t dim = 0; dim < expected->rank; ++dim) {
      if (view->shape[dim] != expected->shape[dim])
        return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                    ET_KERNEL_CODE_INVALID_SHAPE, op,
                    "tensor extents differ from schema");
      /* Checked request limits bound all products. */
      bytes *= expected->shape[dim];
    }
    if (view->byte_length != bytes ||
        !aligned_span(view->data, alignments[type], bytes))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER, op,
                  "invalid tensor data span or alignment");
  }
  return 0;
}

static int32_t validate_storage(const et_kernel_call_v1 *call, shape s,
                                int direction, et_kernel_error *error) {
  const char *op = operations[direction];
  const int backward = direction == BACKWARD;
  const table in = input_table(call);
  const table out = output_table(call);
  const uint64_t q[] = {s.n, s.hq, s.tq, s.dh};
  const uint64_t kv[] = {s.n, s.hkv, s.tk, s.dh};
  const uint64_t qp[] = {s.n, s.tq};
  const uint64_t kp[] = {s.n, s.tk};
  const uint64_t mask[] = {s.n, s.tq, s.tk};
  const schema inputs[] = {{F32, 4u, q},  {F32, 4u, kv}, {F32, 4u, kv},
                           {I64, 2u, qp}, {I64, 2u, kp}, {BOOL, 3u, mask},
                           {F32, 4u, q}};
  const schema outputs[] = {{F32, 4u, q}, {F32, 4u, kv}, {F32, 4u, kv}};
  int32_t status = validate_table(in, backward ? 7u : 6u, inputs, op, error);
  if (status == 0)
    status = validate_table(out, backward ? 3u : 1u, outputs, op, error);
  if (status != 0)
    return status;

  if (overlaps(view_data(view_at(in, K)), view_data(view_at(in, V))))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, op,
                "key and value storage overlap");
  for (size_t i = 0; i < out.count; ++i) {
    const span output = view_data(view_at(out, i));
    if (aliases_call(output, call, 0))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, op,
                  "output overlaps K1 metadata");
    for (size_t j = 0; j < in.count; ++j)
      if (overlaps(output, view_data(view_at(in, j))))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, op,
                    "output overlaps input");
    for (size_t j = 0; j < i; ++j)
      if (overlaps(output, view_data(view_at(out, j))))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, op,
                    "output overlaps another output");
  }
  return 0;
}

static data read_data(table in, int direction) {
  return (data){view_at(in, Q)->data,
                view_at(in, K)->data,
                view_at(in, V)->data,
                direction == BACKWARD ? view_at(in, UPSTREAM)->data : NULL,
                view_at(in, QUERY_POSITION)->data,
                view_at(in, KEY_POSITION)->data,
                view_at(in, MASK)->data};
}

static int valid_positions(const int64_t *positions, size_t batches,
                           size_t count) {
  for (size_t n = 0; n < batches; ++n) {
    int64_t previous = -1;
    for (size_t i = 0; i < count; ++i) {
      const int64_t current = positions[n * count + i];
      if (current <= previous || current > ET_PA4_WIDER_MAX_EXACT_POSITION)
        return 0;
      previous = current;
    }
  }
  return 1;
}

static uint32_t max_magnitude_bits(const float *values, size_t count) {
  uint32_t maximum = 0u;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits;
    memcpy(&bits, values + i, sizeof(bits));
    bits &= UINT32_C(0x7fffffff);
    if (bits > maximum)
      maximum = bits;
  }
  return maximum;
}

static int supported_fenv(void) {
  uint16_t x87;
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, all traps masked, and gradual underflow on both units. */
  return (x87 & 0x0c3fu) == 0x003fu && (_mm_getcsr() & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static size_t q_base(shape s, size_t n, size_t hq, size_t tq) {
  return ((n * s.hq + hq) * s.tq + tq) * s.dh;
}

static size_t kv_base(shape s, size_t n, size_t hkv) {
  return (n * s.hkv + hkv) * s.tk * s.dh;
}

static row make_row(shape s, data x, float scale, size_t n, size_t hq,
                    size_t tq) {
  const size_t qi = q_base(s, n, hq, tq);
  const size_t ki = kv_base(s, n, hq / (s.hq / s.hkv));
  return (row){x.q + qi,
               x.k + ki,
               x.v + ki,
               x.upstream != NULL ? x.upstream + qi : NULL,
               x.kp + n * s.tk,
               x.mask + (n * s.tq + tq) * s.tk,
               x.qp[n * s.tq + tq],
               s.tk,
               s.dh,
               scale};
}

static int admitted(const row *r, size_t tk) {
  return r->mask[tk] != 0u && r->kp[tk] <= r->qp;
}

static float dot(const float *a, const float *b, size_t width) {
  float sum = 0.0f;
  for (size_t d = 0; d < width; ++d)
    sum += a[d] * b[d];
  return sum;
}

static float score(const row *r, size_t tk) {
  return dot(r->q, r->k + tk * r->dh, r->dh) * r->scale;
}

/* Three-pass reference uses these first two passes in both directions. */
static int ref_stats(const row *r, float *maximum, float *denominator,
                     int check) {
  float m = -INFINITY;
  float l = 0.0f;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (!admitted(r, tk))
      continue;
    const float s = score(r, tk);
    if (check && !isfinite(s))
      return 0;
    if (s > m)
      m = s;
  }
  if (m != -INFINITY)
    for (size_t tk = 0; tk < r->tk; ++tk)
      if (admitted(r, tk))
        l += expf(score(r, tk) - m);
  *maximum = m;
  *denominator = l;
  return 1;
}

static int ref_forward(const row *r, float *acc, int check) {
  float maximum, denominator;
  if (!ref_stats(r, &maximum, &denominator, check))
    return 0;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (!admitted(r, tk))
      continue;
    const float p = expf(score(r, tk) - maximum) / denominator;
    const float *v = r->v + tk * r->dh;
    for (size_t d = 0; d < r->dh; ++d)
      acc[d] += p * v[d];
  }
  return 1;
}

static int online_forward(const row *r, float *acc, int check) {
  float maximum = -INFINITY;
  float denominator = 0.0f;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (!admitted(r, tk))
      continue;
    const float s = score(r, tk);
    if (check && !isfinite(s))
      return 0;
    const float *v = r->v + tk * r->dh;
    if (s > maximum) {
      const float rescale = maximum == -INFINITY ? 0.0f : expf(maximum - s);
      denominator = denominator * rescale + 1.0f;
      for (size_t d = 0; d < r->dh; ++d)
        acc[d] = acc[d] * rescale + v[d];
      maximum = s;
    } else {
      const float weight = expf(s - maximum);
      denominator += weight;
      for (size_t d = 0; d < r->dh; ++d)
        acc[d] += weight * v[d];
    }
  }
  if (denominator != 0.0f)
    for (size_t d = 0; d < r->dh; ++d)
      acc[d] /= denominator;
  return 1;
}

static int finite_values(const float *values, size_t count) {
  for (size_t i = 0; i < count; ++i)
    if (!isfinite(values[i]))
      return 0;
  return 1;
}

static int run_forward(shape s, data x, float *output, int online) {
  const float scale = 1.0f / sqrtf((float)s.dh);
  const int check = output == NULL;
  for (size_t n = 0; n < s.n; ++n)
    for (size_t hq = 0; hq < s.hq; ++hq)
      for (size_t tq = 0; tq < s.tq; ++tq) {
        const row r = make_row(s, x, scale, n, hq, tq);
        float acc[MAX_WIDTH] = {0.0f};
        if (!(online ? online_forward(&r, acc, check)
                     : ref_forward(&r, acc, check)))
          return 0;
        if (check) {
          if (!finite_values(acc, r.dh))
            return 0;
        } else {
          memcpy(output + q_base(s, n, hq, tq), acc, r.dh * sizeof(*acc));
        }
      }
  return 1;
}

static void accumulate_gradient(const row *r, size_t tk, float p, float dp,
                                float mean, float *dq, float *dk, float *dv) {
  const float ds = (p * (dp - mean)) * r->scale;
  const size_t start = tk * r->dh;
  for (size_t d = 0; d < r->dh; ++d) {
    dq[d] += ds * r->k[start + d];
    dk[start + d] += ds * r->q[d];
    dv[start + d] += p * r->upstream[d];
  }
}

static int ref_backward(const row *r, float *dq, float *dk, float *dv,
                        int check) {
  float maximum, denominator;
  if (!ref_stats(r, &maximum, &denominator, check))
    return 0;
  float mean = 0.0f;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (!admitted(r, tk))
      continue;
    const float p = expf(score(r, tk) - maximum) / denominator;
    const float dp = dot(r->upstream, r->v + tk * r->dh, r->dh);
    mean += p * dp;
  }
  if (check && !isfinite(mean))
    return 0;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (!admitted(r, tk))
      continue;
    const float p = expf(score(r, tk) - maximum) / denominator;
    const float dp = dot(r->upstream, r->v + tk * r->dh, r->dh);
    accumulate_gradient(r, tk, p, dp, mean, dq, dk, dv);
  }
  return 1;
}

static int online_backward(const row *r, float *dq, float *dk, float *dv,
                           int check) {
  /* Reuse the score slots as weights, then probabilities. */
  float probabilities[ET_PA4_WIDER_MAX_KEYS];
  float dp[ET_PA4_WIDER_MAX_KEYS];
  float maximum = -INFINITY;
  float denominator = 0.0f;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    probabilities[tk] = -INFINITY;
    if (!admitted(r, tk))
      continue;
    const float s = score(r, tk);
    if (check && !isfinite(s))
      return 0;
    probabilities[tk] = s;
    if (s > maximum)
      maximum = s;
  }
  if (maximum == -INFINITY)
    return 1;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (probabilities[tk] == -INFINITY)
      continue;
    probabilities[tk] = expf(probabilities[tk] - maximum);
    denominator += probabilities[tk];
  }
  float mean = 0.0f;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (probabilities[tk] == -INFINITY)
      continue;
    probabilities[tk] /= denominator;
    dp[tk] = dot(r->upstream, r->v + tk * r->dh, r->dh);
    mean += probabilities[tk] * dp[tk];
  }
  if (check && !isfinite(mean))
    return 0;
  for (size_t tk = 0; tk < r->tk; ++tk) {
    if (probabilities[tk] == -INFINITY)
      continue;
    accumulate_gradient(r, tk, probabilities[tk], dp[tk], mean, dq, dk, dv);
  }
  return 1;
}

static int backward_head(shape s, data x, float scale, size_t n, size_t hkv,
                         float *dq, float *dk, float *dv, int online) {
  const size_t group = s.hq / s.hkv;
  const int check = dq == NULL;
  for (size_t hq = hkv * group; hq < (hkv + 1u) * group; ++hq)
    for (size_t tq = 0; tq < s.tq; ++tq) {
      const row r = make_row(s, x, scale, n, hq, tq);
      float scratch[MAX_WIDTH] = {0.0f};
      float *dq_row = check ? scratch : dq + q_base(s, n, hq, tq);
      if (!(online ? online_backward(&r, dq_row, dk, dv, check)
                   : ref_backward(&r, dq_row, dk, dv, check)))
        return 0;
      if (check && !finite_values(dq_row, r.dh))
        return 0;
    }
  /* A nonfinite sum cannot become finite through further additions. */
  return !check ||
         (finite_values(dk, s.tk * s.dh) && finite_values(dv, s.tk * s.dh));
}

/* Keep the KV tile out of the fast and forward validation stack frames. */
__attribute__((noinline)) static int preflight_backward(shape s, data x,
                                                        int online) {
  const float scale = 1.0f / sqrtf((float)s.dh);
  for (size_t n = 0; n < s.n; ++n)
    for (size_t hkv = 0; hkv < s.hkv; ++hkv) {
      /* One KV head tile shared by its query heads, never caller outputs. */
      float dk[ET_PA4_WIDER_MAX_KEYS * MAX_WIDTH] = {0.0f};
      float dv[ET_PA4_WIDER_MAX_KEYS * MAX_WIDTH] = {0.0f};
      if (!backward_head(s, x, scale, n, hkv, NULL, dk, dv, online))
        return 0;
    }
  return 1;
}

static int32_t validate_numeric(shape s, data x, int direction, int online,
                                et_kernel_error *error) {
  const char *op = operations[direction];
  if (!supported_fenv())
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op,
                "unsupported floating-point environment");
  if (!valid_positions(x.qp, s.n, s.tq) || !valid_positions(x.kp, s.n, s.tk))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op,
                "positions must be nonnegative, increasing and exact");
  for (size_t i = 0; i < s.n * s.tq * s.tk; ++i)
    if (x.mask[i] > 1u)
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER, op,
                  "mask bytes must be zero or one");

  const size_t q_count = s.n * s.hq * s.tq * s.dh;
  const size_t kv_count = s.n * s.hkv * s.tk * s.dh;
  const uint32_t q = max_magnitude_bits(x.q, q_count);
  const uint32_t k = max_magnitude_bits(x.k, kv_count);
  const uint32_t v = max_magnitude_bits(x.v, kv_count);
  const uint32_t g =
      x.upstream != NULL ? max_magnitude_bits(x.upstream, q_count) : 0u;
  const uint32_t infinity = UINT32_C(0x7f800000);
  if (q >= infinity || k >= infinity || v >= infinity || g >= infinity)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op,
                "all floating inputs must be finite");

  /* For magnitudes <= 2^20, dot products are <= 2^44, weights are <= 1,
     and even the worst gradient sum (256 contributions) is below 2^81
     in real arithmetic. The remaining binary32 headroom covers rounding.
     Only the exact dry run executes floating-point operations in validation. */
  const uint32_t bound = UINT32_C(0x49800000);
  if (q <= bound && k <= bound && v <= bound && g <= bound)
    return 0;

  fenv_t saved;
  const int saved_errno = errno;
  if (fegetenv(&saved) != 0)
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op,
                "cannot snapshot the floating-point environment");
  const int finite = direction == BACKWARD ? preflight_backward(s, x, online)
                                           : run_forward(s, x, NULL, online);
  const int restored = fesetenv(&saved);
  errno = saved_errno;
  if (restored != 0)
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op,
                "cannot restore the floating-point environment");
  if (!finite)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op,
                "attention arithmetic would become nonfinite");
  return 0;
}

static int32_t validate(const et_kernel_call_v1 *call, et_kernel_error *error,
                        int online) {
  const char *op = "pa4-wider.provider";
  if (error != NULL && aliases_call((span){error, sizeof(*error)}, call, 1))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  if (call == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, op, "call is required");
  if (!aligned_span(call, _Alignof(et_kernel_call_v1), sizeof(size_t)) ||
      call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      !aligned_span(call, _Alignof(et_kernel_call_v1), call->struct_size))
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, op,
                "invalid call descriptor prefix or span");
  const et_kernel_request_v1 *request = call->request;
  if (request == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, op, "request is required");
  if (!aligned_span(request, _Alignof(et_kernel_request_v1), sizeof(size_t)) ||
      request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE ||
      !aligned_span(request, _Alignof(et_kernel_request_v1),
                    request->struct_size))
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, op,
                "invalid request descriptor prefix or span");

  const int direction =
      exact_text(request->operation, operations[BACKWARD]) ? BACKWARD : FORWARD;
  if (!exact_text(request->operation, operations[direction]) ||
      !exact_text(call->capability, capability_name) ||
      !exact_text(request->dtype, "f32") || !exact_text(request->device, "cpu"))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op,
                "unsupported capability, operation, dtype or device");
  op = operations[direction];
  if (request->deterministic > 1u || request->rank != 6u ||
      !aligned_span(request->shape, _Alignof(uint64_t), 6u * sizeof(uint64_t)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_SHAPE, op,
                "invalid request rank, shape span or deterministic flag");
  const shape s = read_shape(request->shape);
  if (!shape_supported(s))
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, op,
                "request is outside the PA4-Wider shape ranges");
  const int32_t storage = validate_storage(call, s, direction, error);
  if (storage != 0)
    return storage;
  const int32_t numeric = validate_numeric(
      s, read_data(input_table(call), direction), direction, online, error);
  if (numeric == 0 && error != NULL)
    *error = (et_kernel_error){0};
  return numeric;
}

static void invoke(const et_kernel_call_v1 *call, int online) {
  const int direction =
      exact_text(call->request->operation, operations[BACKWARD]) ? BACKWARD
                                                                 : FORWARD;
  const shape s = read_shape(call->request->shape);
  const data x = read_data(input_table(call), direction);
  const table out = output_table(call);
  float *first = view_at(out, 0u)->data;
  if (direction == FORWARD) {
    (void)run_forward(s, x, first, online);
    return;
  }
  float *dk = view_at(out, 1u)->data;
  float *dv = view_at(out, 2u)->data;
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *view = view_at(out, i);
    memset(view->data, 0, view->byte_length);
  }
  const float scale = 1.0f / sqrtf((float)s.dh);
  for (size_t n = 0; n < s.n; ++n)
    for (size_t hkv = 0; hkv < s.hkv; ++hkv) {
      const size_t offset = kv_base(s, n, hkv);
      (void)backward_head(s, x, scale, n, hkv, first, dk + offset, dv + offset,
                          online);
    }
}

static int32_t ref_validate(const et_kernel_call_v1 *call,
                            et_kernel_error *error) {
  return validate(call, error, 0);
}

static int32_t online_validate(const et_kernel_call_v1 *call,
                               et_kernel_error *error) {
  return validate(call, error, 1);
}

static void ref_invoke(const et_kernel_call_v1 *call) { invoke(call, 0); }
static void online_invoke(const et_kernel_call_v1 *call) { invoke(call, 1); }

#define DECLARE_PROVIDER(tag, implementation_label)                            \
  static const char tag##_name[] = "pa4-wider.cpu-f32." implementation_label;  \
  static const char tag##_evidence[] = "PA4-Wider:cpu-f32-component-v1";       \
  static const et_kernel_capability_v1 tag##_capability = {                    \
      .struct_size = ET_KERNEL_CAPABILITY_V1_0_SIZE,                           \
      .name = capability_name,                                                 \
      .status = ET_KERNEL_CAPABILITY_VERIFIED,                                 \
      .implementation = tag##_name,                                            \
      .version = "1.0",                                                        \
      .evidence = tag##_evidence,                                              \
      .deterministic = 1u,                                                     \
      .operation_count = COUNT(operations),                                    \
      .operations = operations,                                                \
      .dtype_count = COUNT(dtypes),                                            \
      .dtypes = dtypes,                                                        \
      .device_count = COUNT(devices),                                          \
      .devices = devices,                                                      \
      .shape_range_count = COUNT(ranges),                                      \
      .shape_ranges = ranges};                                                 \
  static const et_kernel_provider_v1 tag##_provider = {                        \
      .struct_size = ET_KERNEL_PROVIDER_V1_0_SIZE,                             \
      .abi_major = ET_KERNEL_ABI_MAJOR,                                        \
      .abi_minor = ET_KERNEL_ABI_MINOR,                                        \
      .name = tag##_name,                                                      \
      .version = "1.0",                                                        \
      .evidence = tag##_evidence,                                              \
      .capability_count = 1u,                                                  \
      .capability_stride = ET_KERNEL_CAPABILITY_V1_0_SIZE,                     \
      .capability_bytes = ET_KERNEL_CAPABILITY_V1_0_SIZE,                      \
      .capabilities = &tag##_capability,                                       \
      .validate_call = tag##_validate,                                         \
      .invoke_call = tag##_invoke}

DECLARE_PROVIDER(ref, "reference");
DECLARE_PROVIDER(online, "online");

const et_kernel_provider_v1 *et_pa4_wider_ref_provider_v1(void) {
  return &ref_provider;
}

const et_kernel_provider_v1 *et_pa4_wider_online_provider_v1(void) {
  return &online_provider;
}
