#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MP4 v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && FLT_RADIX == 2 &&
                   FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "MP4 requires IEEE binary32 and binary64 evaluation");
_Static_assert(sizeof(et_mp4_loss_scaler_v1) == 28u,
               "MP4 loss scaler layout changed");

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define DIM(n) {n, n, 0u, {0}}
#define ROW(rank, ...)                                                         \
  {                                                                            \
    rank, (const et_kernel_dimension_range_v1[]) { __VA_ARGS__ }               \
  }
#define IMPLEMENTATION "mp4.cpu-f32.serial"
#define EVIDENCE "MP4:cpu-mixed-precision-foundation-v1"
#define F32_ABS UINT32_C(0x7fffffff)
#define F32_INF UINT32_C(0x7f800000)
#define F32_ONE UINT32_C(0x3f800000)
#define F16_MAX_AS_F32 UINT32_C(0x477fe000)

static const et_kernel_shape_range_v1 rows[] = {
    ROW(1u, DIM(1)),
    ROW(1u, DIM(8)),
    ROW(1u, DIM(16)),
    ROW(1u, DIM(64)),
    ROW(1u, DIM(256)),
    ROW(2u, DIM(256), DIM(4)),
    ROW(3u, DIM(1), DIM(4), DIM(4)),
    ROW(3u, DIM(2), DIM(8), DIM(16))};

static const char *const f16_ops[] = {"cast.f32-to-f16", "cast.f16-to-f32"};
static const char *const bf16_ops[] = {"cast.f32-to-bf16", "cast.bf16-to-f32"};
static const char *const grad_ops[] = {"grad.unscale-f16-to-f32",
                                       "grad.unscale-bf16-to-f32"};
static const char *const dtypes[] = {"f32"};
static const char *const devices[] = {"cpu"};

enum capability { CAST_F16, CAST_BF16, UNSCALE };
enum dtype { F16, BF16, F32, BOOL };
enum { MAX_ELEMENTS = 1024 };

#define CAPABILITY(name, ops)                                                  \
  {sizeof(et_kernel_capability_v1),                                            \
   name,                                                                       \
   ET_KERNEL_CAPABILITY_VERIFIED,                                              \
   IMPLEMENTATION,                                                             \
   "1.0",                                                                      \
   EVIDENCE,                                                                   \
   1u,                                                                         \
   {0},                                                                        \
   COUNT(ops),                                                                 \
   ops,                                                                        \
   COUNT(dtypes),                                                              \
   dtypes,                                                                     \
   COUNT(devices),                                                             \
   devices,                                                                    \
   COUNT(rows),                                                                \
   rows}

static const et_kernel_capability_v1 capabilities[] = {
    [CAST_F16] = CAPABILITY("kernel.cast-f16", f16_ops),
    [CAST_BF16] = CAPABILITY("kernel.cast-bf16", bf16_ops),
    [UNSCALE] = CAPABILITY("kernel.grad-unscale", grad_ops)};

typedef struct operation_schema {
  enum dtype input;
  enum dtype output;
  size_t arity;
} operation_schema;

static const operation_schema operations[][2] = {
    [CAST_F16] = {{F32, F16, 1u}, {F16, F32, 1u}},
    [CAST_BF16] = {{F32, BF16, 1u}, {BF16, F32, 1u}},
    [UNSCALE] = {{F16, F32, 2u}, {BF16, F32, 2u}}};

typedef struct tensor_table {
  size_t count;
  size_t stride;
  size_t bytes;
  const void *base;
} tensor_table;

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (a_bytes == 0u || b_bytes == 0u)
    return 0;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int text_overlap(const void *data, size_t bytes, const char *text) {
  return spans_overlap(data, bytes, text, strlen(text) + 1u);
}

static tensor_table inputs(const et_kernel_call_v1 *call) {
  return (tensor_table){call->input_count, call->input_stride,
                        call->input_bytes, call->inputs};
}

static tensor_table outputs(const et_kernel_call_v1 *call) {
  return (tensor_table){call->output_count, call->output_stride,
                        call->output_bytes, call->outputs};
}

static const et_kernel_tensor_view_v1 *view_at(tensor_table table, size_t i) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)table.base +
                                            i * table.stride);
}

static int aliases_metadata(const void *data, size_t bytes,
                            const et_kernel_call_v1 *call,
                            const tensor_table tables[2]) {
  const et_kernel_request_v1 *r = call->request;
  if (spans_overlap(data, bytes, call, call->struct_size) ||
      spans_overlap(data, bytes, r, r->struct_size) ||
      spans_overlap(data, bytes, r->shape, r->rank * sizeof(*r->shape)) ||
      text_overlap(data, bytes, call->capability) ||
      text_overlap(data, bytes, r->operation) ||
      text_overlap(data, bytes, r->dtype) ||
      text_overlap(data, bytes, r->device))
    return 1;
  for (size_t t = 0; t < 2u; ++t) {
    if (spans_overlap(data, bytes, tables[t].base, tables[t].bytes))
      return 1;
    for (size_t i = 0; i < tables[t].count; ++i) {
      const et_kernel_tensor_view_v1 *v = view_at(tables[t], i);
      if (spans_overlap(data, bytes, v->shape, v->rank * sizeof(*v->shape)) ||
          text_overlap(data, bytes, v->dtype) ||
          text_overlap(data, bytes, v->device))
        return 1;
    }
  }
  return 0;
}

static int32_t fail(et_kernel_error *error, et_kernel_error_category category,
                    et_kernel_error_code code, const char *operation,
                    const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    /* Every caller passes a matched MP4 operation or a fixed literal. */
    memcpy(error->operation, operation, strlen(operation) + 1u);
    memcpy(error->message, message, strlen(message) + 1u);
  }
  return (int32_t)category;
}

static const operation_schema *find_operation(const et_kernel_call_v1 *call) {
  for (size_t c = 0; c < COUNT(capabilities); ++c) {
    if (!exact_text(call->capability, capabilities[c].name))
      continue;
    for (size_t i = 0; i < capabilities[c].operation_count; ++i) {
      if (exact_text(call->request->operation, capabilities[c].operations[i]))
        return &operations[c][i];
    }
  }
  return NULL;
}

static size_t request_elements(const et_kernel_request_v1 *r) {
  if (!exact_text(r->dtype, "f32") || !exact_text(r->device, "cpu") ||
      r->deterministic > 1u || r->rank == 0u || r->rank > 3u ||
      !aligned_span(r->shape, _Alignof(uint64_t), r->rank * sizeof(*r->shape)))
    return 0u;
  for (size_t i = 0; i < COUNT(rows); ++i) {
    if (r->rank != rows[i].rank)
      continue;
    size_t count = 1u;
    for (size_t d = 0; d < r->rank; ++d) {
      if (r->shape[d] != rows[i].dimensions[d].minimum) {
        count = 0u;
        break;
      }
      count *= (size_t)r->shape[d];
    }
    if (count != 0u && count <= MAX_ELEMENTS)
      return count;
  }
  return 0u;
}

static int32_t validate_view(const et_kernel_tensor_view_v1 *v, size_t stride,
                             enum dtype dtype, size_t elements, size_t rank,
                             const et_kernel_request_v1 *r,
                             et_kernel_error *error) {
  static const char *const names[] = {
      [F16] = "f16", [BF16] = "bf16", [F32] = "f32", [BOOL] = "bool"};
  static const size_t widths[] = {
      [F16] = 2u, [BF16] = 2u, [F32] = 4u, [BOOL] = 1u};
  if (v->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      v->struct_size > stride)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, r->operation,
                "tensor descriptor does not fit the table stride");
  if (!exact_text(v->dtype, names[dtype]))
    return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_TEXT, r->operation,
                "tensor dtype does not match MP4 schema");
  if (!exact_text(v->device, "cpu"))
    return fail(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                ET_KERNEL_CODE_INVALID_TEXT, r->operation,
                "tensor must be on CPU");
  if (v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || v->offset_bytes != 0u)
    return fail(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                ET_KERNEL_CODE_INVALID_BUFFER, r->operation,
                "tensor must be dense and zero-offset");
  if (v->rank != rank ||
      (rank != 0u &&
       (!aligned_span(v->shape, _Alignof(uint64_t), rank * sizeof(*v->shape)) ||
        memcmp(v->shape, r->shape, rank * sizeof(*v->shape)) != 0)))
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, r->operation,
                "tensor shape does not match MP4 schema");
  if (v->byte_length != elements * widths[dtype] ||
      !aligned_span(v->data, widths[dtype], v->byte_length))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, r->operation,
                "incorrect tensor byte length, alignment or span");
  return 0;
}

static int32_t validate_table(tensor_table table, size_t arity, enum dtype data,
                              enum dtype scalar, size_t elements,
                              const et_kernel_request_v1 *r,
                              et_kernel_error *error) {
  if (table.count != arity || table.stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      table.stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      table.count > SIZE_MAX / table.stride ||
      table.bytes != table.count * table.stride ||
      !aligned_span(table.base, _Alignof(et_kernel_tensor_view_v1),
                    table.bytes))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, r->operation,
                "incorrect tensor table count, stride or span");
  const int32_t status = validate_view(view_at(table, 0u), table.stride, data,
                                       elements, r->rank, r, error);
  if (status == 0 && arity == 2u)
    return validate_view(view_at(table, 1u), table.stride, scalar, 1u, 0u, r,
                         error);
  return status;
}

static int overlaps_tensor(const void *data, size_t bytes, tensor_table table) {
  for (size_t i = 0; i < table.count; ++i) {
    const et_kernel_tensor_view_v1 *v = view_at(table, i);
    if (spans_overlap(data, bytes, v->data, v->byte_length))
      return 1;
  }
  return 0;
}

static int32_t validate_aliases(const et_kernel_call_v1 *call,
                                const tensor_table tables[2],
                                et_kernel_error *error) {
  const tensor_table in = tables[0];
  const tensor_table out = tables[1];
  /* Direct callback callers with valid schemas get a non-writing rejection.
   * K1 itself requires the error record to be disjoint before dispatch. */
  if (error != NULL && (aliases_metadata(error, sizeof(*error), call, tables) ||
                        overlaps_tensor(error, sizeof(*error), in) ||
                        overlaps_tensor(error, sizeof(*error), out)))
    return ET_KERNEL_ERROR_INVALID_ARGUMENT;
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *v = view_at(out, i);
    if (aliases_metadata(v->data, v->byte_length, call, tables) ||
        overlaps_tensor(v->data, v->byte_length, in) ||
        (i != 0u &&
         spans_overlap(v->data, v->byte_length, view_at(out, 0u)->data,
                       view_at(out, 0u)->byte_length)))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, call->request->operation,
                  "output aliases K1 metadata or tensor storage");
  }
  return 0;
}

static int fp_environment_supported(void) {
  uint16_t x87;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Mask all exceptions, use nearest on both units, and preserve subnormals. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static uint32_t load32(const void *p) {
  uint32_t bits;
  memcpy(&bits, p, sizeof(bits));
  return bits;
}

static uint16_t load16(const void *p) {
  uint16_t bits;
  memcpy(&bits, p, sizeof(bits));
  return bits;
}

static float f32_from_bits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int positive_f32(uint32_t bits) { return bits != 0u && bits < F32_INF; }

static uint32_t round_shift_even(uint32_t value, unsigned int shift) {
  const uint32_t retained = value >> shift;
  const uint32_t half = UINT32_C(1) << (shift - 1u);
  const uint32_t discarded = value & ((half << 1u) - 1u);
  return retained +
         (discarded > half || (discarded == half && (retained & 1u)));
}

static uint16_t narrow_f16(uint32_t bits) {
  const uint32_t sign = (bits >> 16u) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23u) & 0xffu;
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  /* Exponent 102 contains the halfway point to the first f16 subnormal. */
  if (exponent < 102u)
    return (uint16_t)sign;
  if (exponent < 113u)
    return (uint16_t)(sign | round_shift_even(UINT32_C(0x800000) | fraction,
                                              126u - exponent));
  return (uint16_t)(sign | (((exponent - 112u) << 10u) +
                            round_shift_even(fraction, 13u)));
}

static uint16_t narrow_bf16(uint32_t bits) {
  return (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) >> 16u);
}

static int finite_narrow(enum dtype dtype, uint16_t bits) {
  const uint16_t exponent_mask = dtype == F16 ? 0x7c00u : 0x7f80u;
  return (bits & exponent_mask) != exponent_mask;
}

static uint32_t widen(enum dtype dtype, uint16_t bits) {
  if (dtype == BF16)
    return (uint32_t)bits << 16u;
  const uint32_t sign = ((uint32_t)bits & 0x8000u) << 16u;
  uint32_t fraction = (uint32_t)bits & 0x03ffu;
  uint32_t exponent = ((uint32_t)bits >> 10u) & 0x1fu;
  if (exponent == 0u) {
    if (fraction == 0u)
      return sign;
    exponent = 113u;
    while ((fraction & 0x0400u) == 0u) {
      fraction <<= 1u;
      --exponent;
    }
    fraction &= 0x03ffu;
  } else {
    exponent += 112u;
  }
  return sign | (exponent << 23u) | (fraction << 13u);
}

static int valid_inputs(const operation_schema *op, tensor_table in) {
  if (op->arity == 2u)
    return positive_f32(load32(view_at(in, 1u)->data));
  const et_kernel_tensor_view_v1 *v = view_at(in, 0u);
  const unsigned char *src = v->data;
  if (op->input == F32) {
    for (size_t i = 0; i < v->byte_length; i += sizeof(uint32_t)) {
      const uint32_t bits = load32(src + i);
      const uint32_t abs_bits = bits & F32_ABS;
      if (op->output == F16
              ? abs_bits > F16_MAX_AS_F32
              : abs_bits >= F32_INF || !finite_narrow(BF16, narrow_bf16(bits)))
        return 0;
    }
  } else {
    for (size_t i = 0; i < v->byte_length; i += sizeof(uint16_t)) {
      if (!finite_narrow(op->input, load16(src + i)))
        return 0;
    }
  }
  return 1;
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  if (!aligned_span(call, _Alignof(et_kernel_call_v1),
                    ET_KERNEL_CALL_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mp4.provider",
                "invalid call span");
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE ||
      !aligned_span(call, _Alignof(et_kernel_call_v1), call->struct_size))
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mp4.provider",
                "invalid call descriptor size");
  const et_kernel_request_v1 *r = call->request;
  if (!aligned_span(r, _Alignof(et_kernel_request_v1),
                    ET_KERNEL_REQUEST_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mp4.provider",
                "invalid request span");
  if (r->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE ||
      !aligned_span(r, _Alignof(et_kernel_request_v1), r->struct_size))
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mp4.provider",
                "invalid request descriptor size");
  const operation_schema *op = find_operation(call);
  const size_t elements = request_elements(r);
  if (op == NULL || elements == 0u)
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, "mp4.provider",
                "request is outside the verified MP4 rows");

  const tensor_table tables[] = {inputs(call), outputs(call)};
  int32_t status =
      validate_table(tables[0], op->arity, op->input, F32, elements, r, error);
  if (status == 0)
    status = validate_table(tables[1], op->arity, op->output, BOOL, elements, r,
                            error);
  if (status == 0)
    status = validate_aliases(call, tables, error);
  if (status != 0)
    return status;

  fenv_t environment;
  const int saved_errno = errno;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, r->operation,
                "unsupported or unreadable floating-point environment");
  }
  const int valid = valid_inputs(op, tables[0]);
  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, r->operation,
                "floating-point environment could not be restored");
  if (!valid)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, r->operation,
                "invalid inverse scale or out-of-range cast input");
  et_kernel_error_clear(error);
  return 0;
}

static void unscale(enum dtype dtype, tensor_table in, tensor_table out) {
  const float inverse = f32_from_bits(load32(view_at(in, 1u)->data));
  const unsigned char *src = view_at(in, 0u)->data;
  const et_kernel_tensor_view_v1 *v = view_at(out, 0u);
  uint32_t products[MAX_ELEMENTS];
  const size_t count = v->byte_length / sizeof(uint32_t);
  const unsigned int mxcsr = _mm_getcsr();
  uint8_t overflow = 0u;
  for (size_t i = 0; i < count; ++i) {
    const uint16_t bits = load16(src + i * sizeof(bits));
    if (!finite_narrow(dtype, bits)) {
      overflow = 1u;
      break;
    }
    /* Scalar binary32 SSE multiply, with specials handled as integer bits. */
    const __m128 product = _mm_mul_ss(
        _mm_set_ss(f32_from_bits(widen(dtype, bits))), _mm_set_ss(inverse));
    const float value = _mm_cvtss_f32(product);
    if ((load32(&value) & F32_ABS) >= F32_INF) {
      overflow = 1u;
      break;
    }
    products[i] = load32(&value);
  }
  _mm_setcsr(mxcsr);
  if (overflow)
    memset(v->data, 0, v->byte_length);
  else
    memcpy(v->data, products, v->byte_length);
  memcpy(view_at(out, 1u)->data, &overflow, sizeof(overflow));
}

static void invoke_call(const et_kernel_call_v1 *call) {
  /* K1 guarantees the validated request, tables, inputs and FP controls hold.
   */
  const operation_schema *op = find_operation(call);
  const tensor_table in = inputs(call);
  const tensor_table out = outputs(call);
  if (op->arity == 2u) {
    unscale(op->input, in, out);
    return;
  }
  const et_kernel_tensor_view_v1 *v = view_at(in, 0u);
  const unsigned char *src = v->data;
  unsigned char *dst = view_at(out, 0u)->data;
  if (op->input == F32) {
    for (size_t i = 0; i < v->byte_length / sizeof(uint32_t); ++i) {
      const uint32_t bits = load32(src + i * sizeof(bits));
      const uint16_t converted =
          op->output == F16 ? narrow_f16(bits) : narrow_bf16(bits);
      memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
    }
  } else {
    for (size_t i = 0; i < v->byte_length / sizeof(uint16_t); ++i) {
      const uint32_t converted =
          widen(op->input, load16(src + i * sizeof(uint16_t)));
      memcpy(dst + i * sizeof(converted), &converted, sizeof(converted));
    }
  }
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

const et_kernel_provider_v1 *et_mp4_kernel_provider_v1(void) {
  return &provider;
}

static int valid_state(const et_mp4_loss_scaler_v1 *state) {
  const uint32_t scale = load32(&state->scale);
  const uint32_t min = load32(&state->min_scale);
  const uint32_t max = load32(&state->max_scale);
  const uint32_t growth = load32(&state->growth_factor);
  const uint32_t backoff = load32(&state->backoff_factor);
  return positive_f32(min) && max < F32_INF && min <= scale && scale <= max &&
         F32_ONE < growth && growth < F32_INF && 0u < backoff &&
         backoff < F32_ONE && state->clean_steps < state->growth_interval;
}

et_mp4_loss_scaler_status
et_mp4_loss_scaler_init(et_mp4_loss_scaler_v1 *state, float scale,
                        float growth_factor, float backoff_factor,
                        uint32_t growth_interval, float min_scale,
                        float max_scale) {
  const et_mp4_loss_scaler_v1 initial = {
      scale, growth_factor, backoff_factor, growth_interval,
      0u,    min_scale,     max_scale};
  if (state == NULL || !valid_state(&initial))
    return ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;
  *state = initial;
  return ET_MP4_LOSS_SCALER_STEP_ACCEPTED;
}

et_mp4_loss_scaler_status
et_mp4_loss_scaler_update(et_mp4_loss_scaler_v1 *state,
                          uint8_t overflow_detected) {
  if (state == NULL || overflow_detected > 1u || !valid_state(state))
    return ET_MP4_LOSS_SCALER_INVALID_ARGUMENT;
  fenv_t environment;
  const int saved_errno = errno;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT;
  }

  et_mp4_loss_scaler_v1 next = *state;
  et_mp4_loss_scaler_status status = ET_MP4_LOSS_SCALER_STEP_ACCEPTED;
  if (overflow_detected) {
    /* Every product of two finite binary32 values is exact in binary64. */
    const double backed_off = (double)next.scale * (double)next.backoff_factor;
    const float rounded = (float)backed_off;
    if (backed_off < (double)next.min_scale || rounded >= next.scale) {
      status = ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT;
    } else {
      next.scale = rounded;
      next.clean_steps = 0u;
      status = ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF;
    }
  } else if (++next.clean_steps == next.growth_interval) {
    const double grown = (double)next.scale * (double)next.growth_factor;
    next.scale =
        grown >= (double)next.max_scale ? next.max_scale : (float)grown;
    next.clean_steps = 0u;
  }

  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT;
  if (status != ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT)
    *state = next;
  return status;
}
