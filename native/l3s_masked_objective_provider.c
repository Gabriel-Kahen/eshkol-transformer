#include "eshkol_transformer/l3s_masked_objective_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if !defined(__x86_64__)
#error "L3S v1 requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
  FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 && FLT_EVAL_METHOD == 0,
  "L3S requires IEEE binary32 evaluation");

enum operation { REDUCE_BOOL, REDUCE_F32, NUMERATOR_BOOL, NUMERATOR_F32,
                 MEAN_BOOL, MEAN_F32, UNKNOWN };
static const char *const operations[] = {
  "l3s.masked-objective.reduce.bool", "l3s.masked-objective.reduce.f32",
  "l3s.masked-objective.numerator-seed.bool",
  "l3s.masked-objective.numerator-seed.f32",
  "l3s.masked-objective.mean-seed.bool", "l3s.masked-objective.mean-seed.f32"
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
                             int scalar, const char *op, et_kernel_error *e) {
  if (!same(v->dtype, dtype))
    return fail(e, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
                op, "operand dtype differs from schema");
  if (!same(v->device, "cpu"))
    return fail(e, ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
                op, "operand must be CPU");
  if (v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || v->offset_bytes != 0)
    return fail(e, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand must be dense and zero-offset");
  if (v->rank != (scalar ? 0u : 2u) || (scalar ? v->shape != NULL :
      v->shape == NULL || v->shape[0] != 1 || v->shape[1] != 2))
    return fail(e, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
                op, "operand must have exact scalar or [1,2] shape");
  const size_t width = same(dtype, "bool") ? 1u : sizeof(float);
  const size_t bytes = width * (scalar ? 1u : 2u);
  if (v->byte_length != bytes || !span(v->data, bytes) ||
      (uintptr_t)v->data % width != 0)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand alignment or span is invalid");
  return 0;
}

/* Exact live metadata spans, including compatible-minor table extensions.
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
      overlaps(out->data, out->byte_length, r->shape, r->rank*sizeof(uint64_t)) ||
      hits_text(out, c->capability) || hits_text(out, r->operation) ||
      hits_text(out, r->dtype) || hits_text(out, r->device)) return 1;
  for (size_t i = 0; i < c->input_count + c->output_count; ++i) {
    const et_kernel_tensor_view_v1 *v = i < c->input_count ? input(c,i) :
      output(c,i-c->input_count);
    if (overlaps(out->data, out->byte_length, v->shape, v->rank*sizeof(uint64_t)) ||
        hits_text(out, v->dtype) || hits_text(out, v->device)) return 1;
  }
  return 0;
}
static float positive_zero(float x) { return x == 0.0f ? 0.0f : x; }
static int nonnegative(float x) { return isfinite(x) && x >= 0.0f; }

/* Same fixed stack evaluation is used for preflight and commit. Separate
 * statements and strict FP compilation preserve every binary32 rounding point. */
static int evaluate(const et_kernel_call_v1 *c, enum operation op, float result[3]) {
  const int reduce = op == REDUCE_BOOL || op == REDUCE_F32;
  const int boolean = op % 2 == 0;
  const void *mask = input(c, reduce ? 1u : 0u)->data;
  float m[2], ce[2] = {0.0f, 0.0f};
  for (size_t i = 0; i < 2; ++i) {
    if (boolean) {
      const uint8_t byte = ((const uint8_t *)mask)[i];
      if (byte > 1u) return 0;
      m[i] = byte == 0u ? 0.0f : 1.0f;
    } else {
      const float value = ((const float *)mask)[i];
      if (!nonnegative(value)) return 0;
      m[i] = positive_zero(value);
    }
    if (reduce) {
      const float value = ((const float *)input(c,0)->data)[i];
      if (!nonnegative(value)) return 0;
      ce[i] = positive_zero(value);
    }
  }
  float w = 0.0f;
  w = w + m[0];
  if (!isfinite(w)) return 0;
  w = w + m[1];
  if (!isfinite(w) || w <= 0.0f) return 0;
  if (reduce) {
    const float p0 = m[0] * ce[0];
    const float p1 = m[1] * ce[1];
    if (!isfinite(p0) || !isfinite(p1)) return 0;
    float n = 0.0f;
    n = n + p0;
    if (!isfinite(n)) return 0;
    n = n + p1;
    if (!isfinite(n)) return 0;
    const float mean = n / w;
    if (!isfinite(mean)) return 0;
    result[0] = positive_zero(n);
    result[1] = w;
    result[2] = positive_zero(mean);
  } else {
    for (size_t i = 0; i < 2; ++i) {
      const float seed = op == MEAN_BOOL || op == MEAN_F32 ? m[i] / w : m[i];
      if (!isfinite(seed)) return 0;
      result[i] = positive_zero(seed);
    }
  }
  return 1;
}

/* Direct callbacks retain K1's generic descriptor/readability/disjoint-error
 * preconditions; the complete supported public call goes through K1 dispatch. */
static int32_t validate_inner(const et_kernel_call_v1 *c, et_kernel_error *e) {
  const char *op = "l3s.provider";
  if (c == NULL || c->request == NULL)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_NULL_ARGUMENT,
                op, "call and request are required");
  if (c->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      c->request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(e, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                op, "call or request truncates K1 prefix");
  const et_kernel_request_v1 *r = c->request;
  const enum operation kind = identify(r->operation);
  if (kind == UNKNOWN || !same(c->capability, "l3s.masked-objective") ||
      !same(r->dtype,"f32") || !same(r->device,"cpu") || r->rank != 2u ||
      r->shape == NULL || r->shape[0] != 1u || r->shape[1] != 2u)
    return fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
                op, "request is outside the exact L3S profile");
  op = r->operation;
  const int reduce = kind == REDUCE_BOOL || kind == REDUCE_F32;
  const size_t ni = reduce ? 2u : 1u, no = reduce ? 3u : 1u;
  if (c->input_count != ni || c->output_count != no)
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
                op, "operand counts differ from schema");
  int32_t rc;
  for (size_t i = 0; i < ni; ++i) {
    const char *dtype = i == ni-1 && kind % 2 == 0 ? "bool" : "f32";
    rc = validate_view(input(c,i), dtype, 0, op, e);
    if (rc) return rc;
  }
  for (size_t i = 0; i < no; ++i) {
    rc = validate_view(output(c,i), "f32", reduce, op, e);
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
  float result[3];
  if (!evaluate(c,kind,result))
    return fail(e, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED,
                op, "invalid operand, weight, intermediate or result");
  et_kernel_error_clear(e);
  return 0;
}
static int32_t provider_validate(const et_kernel_call_v1 *c, et_kernel_error *e) {
  fenv_t environment;
  if (fegetenv(&environment) != 0)
    return fail(e, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED,
                "l3s.provider", "cannot observe floating environment");
  const int32_t rc = validate_inner(c,e);
  if (fesetenv(&environment) != 0)
    return fail(e, ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_PROVIDER_REJECTED,
                "l3s.provider", "cannot restore floating environment");
  return rc;
}
static void provider_invoke(const et_kernel_call_v1 *c) {
  const enum operation op = identify(c->request->operation);
  float result[3];
  (void)evaluate(c,op,result);
  if (op == REDUCE_BOOL || op == REDUCE_F32) {
    for (size_t i = 0; i < 3; ++i) *(float *)output(c,i)->data = result[i];
  } else {
    memcpy(output(c,0)->data, result, 2u*sizeof(float));
  }
}

static const et_kernel_dimension_range_v1 dimensions[] = {
  {1,1,0,{0}}, {2,2,0,{0}}
};
static const et_kernel_shape_range_v1 rows[] = {{2,dimensions}};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};
static const et_kernel_capability_v1 capability = {
  sizeof(et_kernel_capability_v1), "l3s.masked-objective", ET_KERNEL_CAPABILITY_VERIFIED,
  "l3s.cpu-f32.serial", "1.0", "L3S:cpu-f32-masked-objective-v1", 1, {0},
  6, operations, 1, dtypes, 1, devices, 1, rows
};
static const et_kernel_provider_v1 provider = {
  sizeof(et_kernel_provider_v1), ET_KERNEL_ABI_MAJOR, ET_KERNEL_ABI_MINOR, 0,
  "l3s.cpu-f32.serial", "1.0", "L3S:cpu-f32-masked-objective-v1",
  1, sizeof(capability), sizeof(capability), &capability,
  provider_validate, provider_invoke
};
const et_kernel_provider_v1 *et_l3s_kernel_provider_v1(void) { return &provider; }
