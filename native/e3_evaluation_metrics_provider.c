#include "eshkol_transformer/e3_evaluation_metrics_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if !defined(__x86_64__)
#error "E3 metrics v1 requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
  FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
  "E3 metrics requires IEEE binary32 evaluation");

_Static_assert(sizeof(int64_t) == 8 && INT64_MAX == INT64_C(9223372036854775807),
  "E3 metrics requires exact signed i64 storage");

enum operation { ACCUMULATE, CORRECT_BOOL, FINALIZE, UNKNOWN };
static const char *const operations[] = {
  "e3.evaluation-metrics.accumulate",
  "e3.evaluation-metrics.correct.bool",
  "e3.evaluation-metrics.finalize"
};
static int same(const char *a, const char *b) {
  return a != NULL && strcmp(a, b) == 0;
}
static enum operation identify(const char *name) {
  for (size_t i = 0; i < UNKNOWN; ++i)
    if (same(name, operations[i])) return (enum operation)i;
  return UNKNOWN;
}
static int span(const void *p, size_t n) {
  return n == 0 || (p != NULL && (uintptr_t)p <= UINTPTR_MAX - n);
}
static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
  if (!na || !nb) return 0;
  if (!span(a, na) || !span(b, nb)) return 1;
  return (uintptr_t)a < (uintptr_t)b + nb &&
         (uintptr_t)b < (uintptr_t)a + na;
}
static const et_kernel_tensor_view_v1 *view_at(const void *base,
                                               size_t stride, size_t i) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)base + stride*i);
}
static const et_kernel_tensor_view_v1 *input(const et_kernel_call_v1 *c, size_t i) {
  return view_at(c->inputs, c->input_stride, i);
}
static const et_kernel_tensor_view_v1 *output(const et_kernel_call_v1 *c, size_t i) {
  return view_at(c->outputs, c->output_stride, i);
}
static int32_t fail(et_kernel_error *e, uint32_t category, uint32_t code,
                    const char *operation, const char *message) {
  if (e != NULL) {
    et_kernel_error_clear(e);
    e->category = category; e->code = code;
    (void)snprintf(e->operation, sizeof(e->operation), "%s", operation);
    (void)snprintf(e->message, sizeof(e->message), "%s", message);
  }
  return (int32_t)category;
}
static int fp_supported(void) {
  unsigned short control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  const unsigned mxcsr = _mm_getcsr();
  return (control & 0x003fu) == 0x003fu && (control & 0x0c00u) == 0 &&
    (mxcsr & 0x1f80u) == 0x1f80u && (mxcsr & 0xe040u) == 0;
}
static int32_t validate_view(const et_kernel_tensor_view_v1 *v, const char *dtype,
                             size_t rank, const char *op, et_kernel_error *e) {
  if (!same(v->dtype, dtype))
    return fail(e, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
                op, "operand dtype differs from schema");
  if (!same(v->device, "cpu"))
    return fail(e, ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
                op, "operand must be CPU");
  if (v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || v->offset_bytes != 0)
    return fail(e, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand must be dense and zero-offset");
  if (v->rank != rank || (rank == 0 ? v->shape != NULL :
      v->shape == NULL || v->shape[0] != 1 || v->shape[1] != 2 ||
      (rank == 3 && v->shape[2] != 256)))
    return fail(e, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
                op, "operand shape differs from exact schema");
  const size_t width = same(dtype, "bool") ? 1u : same(dtype, "i64") ? 8u : 4u;
  const size_t bytes = width * (rank == 0 ? 1u : rank == 2 ? 2u : 512u);
  if (v->byte_length != bytes || !span(v->data, bytes) ||
      (uintptr_t)v->data % width != 0)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand alignment or span is invalid");
  return 0;
}

/* Complete live metadata spans, including compatible-minor table extensions.
 * K1 already validated readable strings and shape/table prefixes. */
static int hits_text(const et_kernel_tensor_view_v1 *out, const char *text) {
  return overlaps(out->data, out->byte_length, text, strlen(text) + 1u);
}
static int hits_metadata(const et_kernel_call_v1 *c,
                         const et_kernel_tensor_view_v1 *out) {
  const et_kernel_request_v1 *r = c->request;
  if (overlaps(out->data, out->byte_length, c, c->struct_size) ||
      overlaps(out->data, out->byte_length, r, r->struct_size) ||
      overlaps(out->data, out->byte_length, c->inputs, c->input_bytes) ||
      overlaps(out->data, out->byte_length, c->outputs, c->output_bytes) ||
      overlaps(out->data, out->byte_length, r->shape, r->rank*sizeof(uint64_t)))
    return 1;
  for (size_t i = 0; i < c->input_count + c->output_count; ++i) {
    const et_kernel_tensor_view_v1 *v = i < c->input_count ? input(c,i) :
      output(c,i-c->input_count);
    if (overlaps(out->data, out->byte_length, v->shape, v->rank*sizeof(uint64_t)))
      return 1;
  }
  if (hits_text(out, c->capability) || hits_text(out, r->operation) ||
      hits_text(out, r->dtype) || hits_text(out, r->device)) return 1;
  for (size_t i = 0; i < c->input_count + c->output_count; ++i) {
    const et_kernel_tensor_view_v1 *v = i < c->input_count ? input(c,i) :
      output(c,i-c->input_count);
    if (hits_text(out, v->dtype) || hits_text(out, v->device)) return 1;
  }
  return 0;
}
static float positive_zero(float x) { return x == 0.0f ? 0.0f : x; }
static int nonnegative(float x) { return isfinite(x) && x >= 0.0f; }
static float scalar(const et_kernel_call_v1 *c, size_t i) {
  return *(const float *)input(c,i)->data;
}
static int64_t counter(const et_kernel_call_v1 *c, size_t i) {
  return *(const int64_t *)input(c,i)->data;
}
static int integral_bounded(float x) { return x == (float)(int64_t)x; }
static int32_t domain(et_kernel_error *e, const char *op) {
  return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED,
              op, "invalid operand, intermediate or result");
}
static int32_t capped(et_kernel_error *e, const char *op) {
  return fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
              op, "state exceeds fixed evaluation profile");
}

/* The result union occupies 32 bytes at most; each row uses only its specified
 * 1f+1i, 3f+2i, or 4f result. All additional scratch is fixed scalar storage.
 * Strict FP compilation preserves each separate binary32 rounding point. */
union results {
  struct { float correct; int64_t active; } correct;
  struct { float n, w, correct; int64_t tokens, batches; } accumulate;
  struct { float loss, weight, perplexity, accuracy; } finalize;
};
static int64_t argmax(const float *row) {
  int64_t winner = 0;
  float maximum = row[0];
  for (int64_t j = 1; j < 256; ++j) {
    if (row[j] > maximum) {
      maximum = row[j];
      winner = j;
    }
  }
  return winner;
}
static int32_t evaluate_correct(const et_kernel_call_v1 *c, union results *result,
                                et_kernel_error *e) {
  const char *op = operations[CORRECT_BOOL];
  const float *logits = input(c,0)->data;
  const int64_t *targets = input(c,1)->data;
  const uint8_t *mask = input(c,2)->data;
  for (size_t i = 0; i < 2; ++i)
    if (targets[i] < 0 || targets[i] >= 256)
      return fail(e, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
                  op, "target index is outside vocabulary");
  for (size_t i = 0; i < 2; ++i)
    for (size_t j = 0; j < 256; ++j)
      if (!isfinite(logits[i*256+j])) return domain(e,op);
  for (size_t i = 0; i < 2; ++i)
    if (mask[i] > 1u) return domain(e,op);
  if (mask[0] == 0u && mask[1] == 0u) return domain(e,op);
  const int64_t winner0 = argmax(logits);
  const int64_t winner1 = argmax(logits + 256);
  const float weight0 = mask[0] == 0u ? 0.0f : 1.0f;
  const float weight1 = mask[1] == 0u ? 0.0f : 1.0f;
  const float hit0 = winner0 == targets[0] ? 1.0f : 0.0f;
  const float hit1 = winner1 == targets[1] ? 1.0f : 0.0f;
  const float p0 = weight0 * hit0;
  const float p1 = weight1 * hit1;
  float correct = 0.0f;
  correct = correct + p0;
  correct = correct + p1;
  result->correct.correct = correct;
  result->correct.active = (int64_t)mask[0] + (int64_t)mask[1];
  return 0;
}
static int32_t evaluate_accumulate(const et_kernel_call_v1 *c, union results *result,
                                   et_kernel_error *e) {
  const char *op = operations[ACCUMULATE];
  /* Check floats in input-table order before examining any integer domain. */
  for (size_t i = 0; i < 9; ++i)
    if (i != 3 && i != 4 && i != 8 && !nonnegative(scalar(c,i)))
      return domain(e,op);
  const float n = positive_zero(scalar(c,0)), w = positive_zero(scalar(c,1));
  const float correct = positive_zero(scalar(c,2));
  const float nb = positive_zero(scalar(c,5)), wb = positive_zero(scalar(c,6));
  const float cb = positive_zero(scalar(c,7));
  const int64_t tokens = counter(c,3), batches = counter(c,4), active = counter(c,8);
  if (tokens < 0 || batches < 0 || active < 0) return domain(e,op);
  if (active != 1 && active != 2) return domain(e,op);
  if (batches > 8192 || tokens > 16384) return capped(e,op);
  if (batches > tokens) return domain(e,op);
  if (tokens-batches > batches) return domain(e,op);
  if (w != (float)tokens) return domain(e,op);
  if (correct > w || !integral_bounded(correct)) return domain(e,op);
  if (batches == 0 && n != 0.0f) return domain(e,op);
  if (wb != (float)active) return domain(e,op);
  if (cb > wb || !integral_bounded(cb)) return domain(e,op);
  if (tokens > INT64_MAX-active || batches == INT64_MAX)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
                op, "integer counter addition overflow");
  const int64_t next_tokens = tokens + active;
  const int64_t next_batches = batches + 1;
  if (next_batches > 8192 || next_tokens > 16384) return capped(e,op);
  const float next_n = n + nb;
  if (!isfinite(next_n)) return domain(e,op);
  const float next_w = w + wb;
  if (!isfinite(next_w)) return domain(e,op);
  const float next_c = correct + cb;
  if (!isfinite(next_c)) return domain(e,op);
  result->accumulate.n = positive_zero(next_n);
  result->accumulate.w = positive_zero(next_w);
  result->accumulate.correct = positive_zero(next_c);
  result->accumulate.tokens = next_tokens;
  result->accumulate.batches = next_batches;
  return 0;
}
static int32_t evaluate_finalize(const et_kernel_call_v1 *c, union results *result,
                                 et_kernel_error *e) {
  const char *op = operations[FINALIZE];
  for (size_t i = 0; i < 3; ++i)
    if (!nonnegative(scalar(c,i))) return domain(e,op);
  const float n = positive_zero(scalar(c,0)), w = positive_zero(scalar(c,1));
  const float correct = positive_zero(scalar(c,2));
  if (w == 0.0f) return domain(e,op);
  if (w > 16384.0f) return capped(e,op);
  if (!integral_bounded(w)) return domain(e,op);
  if (correct > w || !integral_bounded(correct)) return domain(e,op);
  const float loss = n / w;
  if (!isfinite(loss)) return domain(e,op);
  const float accuracy = correct / w;
  if (!isfinite(accuracy)) return domain(e,op);
  const float perplexity = expf(loss);
  if (!isfinite(perplexity)) return domain(e,op);
  result->finalize.loss = positive_zero(loss);
  result->finalize.weight = w;
  result->finalize.perplexity = perplexity;
  result->finalize.accuracy = positive_zero(accuracy);
  return 0;
}
static int32_t evaluate(const et_kernel_call_v1 *c, enum operation op,
                         union results *result, et_kernel_error *e) {
  if (op == CORRECT_BOOL) return evaluate_correct(c,result,e);
  if (op == ACCUMULATE) return evaluate_accumulate(c,result,e);
  return evaluate_finalize(c,result,e);
}

/* Direct callbacks retain K1's readable-metadata and disjoint-error control
 * preconditions. Supported dispatch always performs generic K1 validation first. */
static int32_t validate_inner(const et_kernel_call_v1 *c, et_kernel_error *e) {
  const char *op = "e3.metrics.provider";
  if (c == NULL)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_NULL_ARGUMENT,
                op, "call is required");
  if (c->struct_size < ET_KERNEL_CALL_V1_0_SIZE)
    return fail(e, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                op, "call truncates K1 prefix");
  if (c->request == NULL)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_NULL_ARGUMENT,
                op, "request is required");
  if (c->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(e, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                op, "request truncates K1 prefix");
  const et_kernel_request_v1 *r = c->request;
  const enum operation kind = identify(r->operation);
  if (kind != UNKNOWN) op = operations[kind];
  if (kind == UNKNOWN || !same(c->capability,"e3.evaluation-metrics") ||
      !same(r->dtype,"f32") || !same(r->device,"cpu") || r->rank != 3u ||
      r->shape == NULL || r->shape[0] != 1u || r->shape[1] != 2u ||
      r->shape[2] != 256u)
    return fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
                op, "request is outside exact E3 metrics profile");
  const size_t ni = kind == ACCUMULATE ? 9u : 3u;
  const size_t no = kind == ACCUMULATE ? 5u : kind == CORRECT_BOOL ? 2u : 4u;
  if (c->input_count != ni || c->output_count != no)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand counts differ from schema");
  for (size_t i = 0; i < ni; ++i) {
    const char *dtype = kind == CORRECT_BOOL ?
      (i == 0 ? "f32" : i == 1 ? "i64" : "bool") :
      kind == ACCUMULATE && (i == 3 || i == 4 || i == 8) ? "i64" : "f32";
    const size_t rank = kind == CORRECT_BOOL ? (i == 0 ? 3u : 2u) : 0u;
    const int32_t rc = validate_view(input(c,i),dtype,rank,op,e);
    if (rc) return rc;
  }
  for (size_t i = 0; i < no; ++i) {
    const char *dtype = (kind == ACCUMULATE && i >= 3) ||
      (kind == CORRECT_BOOL && i == 1) ? "i64" : "f32";
    const int32_t rc = validate_view(output(c,i),dtype,0u,op,e);
    if (rc) return rc;
  }
  for (size_t i = 0; i < ni; ++i) for (size_t j = i+1; j < ni; ++j)
    if (overlaps(input(c,i)->data, input(c,i)->byte_length,
                 input(c,j)->data, input(c,j)->byte_length))
      return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT,
                  op, "input storage must be pairwise disjoint");
  for (size_t i = 0; i < no; ++i)
    if (hits_metadata(c,output(c,i)))
      return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT,
                  op, "output overlaps live call metadata");
  if (!fp_supported())
    return fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
                op, "unsupported x87 or MXCSR controls");
  union results result;
  const int32_t rc = evaluate(c,kind,&result,e);
  if (rc) return rc;
  et_kernel_error_clear(e);
  return 0;
}
static int32_t provider_validate(const et_kernel_call_v1 *c, et_kernel_error *e) {
  const int saved_errno = errno;
  fenv_t environment;
  int32_t rc;
  if (fegetenv(&environment) != 0) {
    rc = fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
              "e3.metrics.provider", "cannot observe floating environment");
  } else {
    rc = validate_inner(c,e);
    if (fesetenv(&environment) != 0)
      rc = fail(e, ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_PROVIDER_REJECTED,
                "e3.metrics.provider", "cannot restore floating environment");
  }
  errno = saved_errno;
  return rc;
}
static void provider_invoke(const et_kernel_call_v1 *c) {
  const int saved_errno = errno;
  const enum operation op = identify(c->request->operation);
  union results result;
  /* Mutation after successful validation violates K1's invocation precondition. */
  if (evaluate(c,op,&result,NULL) != 0) __builtin_trap();
  if (op == CORRECT_BOOL) {
    *(float *)output(c,0)->data = result.correct.correct;
    *(int64_t *)output(c,1)->data = result.correct.active;
  } else if (op == ACCUMULATE) {
    *(float *)output(c,0)->data = result.accumulate.n;
    *(float *)output(c,1)->data = result.accumulate.w;
    *(float *)output(c,2)->data = result.accumulate.correct;
    *(int64_t *)output(c,3)->data = result.accumulate.tokens;
    *(int64_t *)output(c,4)->data = result.accumulate.batches;
  } else {
    *(float *)output(c,0)->data = result.finalize.loss;
    *(float *)output(c,1)->data = result.finalize.weight;
    *(float *)output(c,2)->data = result.finalize.perplexity;
    *(float *)output(c,3)->data = result.finalize.accuracy;
  }
  errno = saved_errno;
}

static const et_kernel_dimension_range_v1 dimensions[] = {
  {1,1,0,{0}}, {2,2,0,{0}}, {256,256,0,{0}}
};
static const et_kernel_shape_range_v1 rows[] = {{3,dimensions}};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};
static const et_kernel_capability_v1 capability = {
  sizeof(et_kernel_capability_v1), "e3.evaluation-metrics", ET_KERNEL_CAPABILITY_VERIFIED,
  "e3.cpu-f32-bool.serial", "1.0", "E3:cpu-f32-bool-metrics-v1", 1, {0},
  3, operations, 1, dtypes, 1, devices, 1, rows
};
static const et_kernel_provider_v1 provider = {
  sizeof(et_kernel_provider_v1), ET_KERNEL_ABI_MAJOR, ET_KERNEL_ABI_MINOR, 0,
  "e3.cpu-f32-bool.serial", "1.0", "E3:cpu-f32-bool-metrics-v1",
  1, sizeof(capability), sizeof(capability), &capability,
  provider_validate, provider_invoke
};
const et_kernel_provider_v1 *et_e3_metrics_kernel_provider_v1(void) { return &provider; }
