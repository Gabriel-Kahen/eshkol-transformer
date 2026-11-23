#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

typedef struct format {
  const char *dtype;
  const char *capability;
  const char *narrow;
  const char *widen;
  const char *unscale;
  uint16_t infinity;
} format;

static const format formats[] = {
    {"f16", "kernel.cast-f16", "cast.f32-to-f16", "cast.f16-to-f32",
     "grad.unscale-f16-to-f32", 0x7c00u},
    {"bf16", "kernel.cast-bf16", "cast.f32-to-bf16", "cast.bf16-to-f32",
     "grad.unscale-bf16-to-f32", 0x7f80u},
};

static const uint64_t scalar_batch[] = {1u};
static const uint64_t vector_batch[] = {8u};
static const uint64_t exhaustive_batch[] = {256u, 4u};

static uint32_t bits_of(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float float_of(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static const et_kernel_provider_v1 *resolve_mp4(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mp4_kernel_provider_v1()
             : NULL;
}

static et_kernel_request_v1 request(const char *operation, size_t rank,
                                    const uint64_t *shape) {
  return (et_kernel_request_v1){sizeof(et_kernel_request_v1),
                                operation,
                                "f32",
                                "cpu",
                                rank,
                                shape,
                                1u,
                                {0}};
}

static et_kernel_tensor_view_v1 view(void *data, const char *dtype, size_t rank,
                                     const uint64_t *shape) {
  size_t bytes = strcmp(dtype, "f32") == 0    ? 4u
                 : strcmp(dtype, "bool") == 0 ? 1u
                                              : 2u;
  for (size_t d = 0; d < rank; ++d)
    bytes *= (size_t)shape[d];
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u,   rank,  shape};
}

static et_kernel_call_v1 make_call(const char *cap,
                                   const et_kernel_request_v1 *r, size_t arity,
                                   const void *in, void *out) {
  const size_t stride = sizeof(et_kernel_tensor_view_v1);
  return (et_kernel_call_v1){sizeof(et_kernel_call_v1),
                             cap,
                             r,
                             arity,
                             stride,
                             arity * stride,
                             in,
                             arity,
                             stride,
                             arity * stride,
                             out};
}

static void dispatch(const et_kernel_runtime *runtime,
                     const et_kernel_call_v1 *call) {
  et_kernel_error error;
  if (et_kernel_runtime_dispatch(runtime, call, &error) != 0) {
    fprintf(stderr, "%s (%u/%u): %s\n", error.operation, error.category,
            error.code, error.message);
    exit(EXIT_FAILURE);
  }
  CHECK(error.code == ET_KERNEL_CODE_OK);
}

static int32_t cast(const et_kernel_runtime *runtime, const format *fmt,
                    int narrow, size_t rank, const uint64_t *shape, void *src,
                    void *dst, et_kernel_error *error) {
  const et_kernel_request_v1 r =
      request(narrow ? fmt->narrow : fmt->widen, rank, shape);
  const et_kernel_tensor_view_v1 in =
      view(src, narrow ? "f32" : fmt->dtype, rank, shape);
  et_kernel_tensor_view_v1 out =
      view(dst, narrow ? fmt->dtype : "f32", rank, shape);
  CHECK(narrow ? in.byte_length == 2u * out.byte_length
               : out.byte_length == 2u * in.byte_length);
  const et_kernel_call_v1 call = make_call(fmt->capability, &r, 1u, &in, &out);
  return et_kernel_runtime_dispatch(runtime, &call, error);
}

static void unscale(const et_kernel_runtime *runtime, const format *fmt,
                    size_t rank, const uint64_t *shape, uint16_t *scaled,
                    float inverse, void *master, uint8_t *overflow) {
  const et_kernel_request_v1 r = request(fmt->unscale, rank, shape);
  const et_kernel_tensor_view_v1 in[] = {view(scaled, fmt->dtype, rank, shape),
                                         view(&inverse, "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 out[] = {view(master, "f32", rank, shape),
                                    view(overflow, "bool", 0u, NULL)};
  CHECK(out[0].byte_length == 2u * in[0].byte_length);
  const et_kernel_call_v1 call =
      make_call("kernel.grad-unscale", &r, 2u, in, out);
  dispatch(runtime, &call);
}

/* Independent numerical widening for every finite storage word. */
static uint32_t expected_widen(uint16_t bits, int bf16) {
  const unsigned int fraction_bits = bf16 ? 7u : 10u;
  const unsigned int fraction_mask = (1u << fraction_bits) - 1u;
  const int bias = bf16 ? 127 : 15;
  const int exponent = (bits & 0x7fffu) >> fraction_bits;
  const unsigned int fraction = bits & fraction_mask;
  const unsigned int significand =
      exponent == 0 ? fraction : fraction + fraction_mask + 1u;
  const float magnitude =
      ldexpf((float)significand,
             (exponent == 0 ? 1 : exponent) - bias - (int)fraction_bits);
  return bits_of(bits & 0x8000u ? -magnitude : magnitude);
}

static void test_exhaustive(const et_kernel_runtime *runtime, const format *fmt,
                            int bf16) {
  enum { BATCH = 1024 };
  uint16_t source[BATCH];
  uint16_t roundtrip[BATCH];
  uint32_t widened[BATCH];
  et_kernel_error error;
  size_t count = 0u;
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    uint16_t word = (uint16_t)bits;
    if ((word & 0x7fffu) >= fmt->infinity) {
      uint32_t untouched = UINT32_C(0xabcdef12);
      CHECK(cast(runtime, fmt, 0, 1u, scalar_batch, &word, &untouched,
                 &error) != 0);
      CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(untouched == UINT32_C(0xabcdef12) && word == bits);
      continue;
    }
    source[count++] = word;
    if (count != BATCH)
      continue;
    CHECK(cast(runtime, fmt, 0, 2u, exhaustive_batch, source, widened,
               &error) == 0);
    CHECK(cast(runtime, fmt, 1, 2u, exhaustive_batch, widened, roundtrip,
               &error) == 0);
    CHECK(memcmp(source, roundtrip, sizeof(source)) == 0);
    for (size_t i = 0; i < count; ++i)
      CHECK(widened[i] == expected_widen(source[i], bf16));
    count = 0u;
  }
  for (size_t i = 0; i < count; ++i) {
    CHECK(cast(runtime, fmt, 0, 1u, scalar_batch, &source[i], &widened[i],
               &error) == 0);
    CHECK(cast(runtime, fmt, 1, 1u, scalar_batch, &widened[i], &roundtrip[i],
               &error) == 0);
    CHECK(widened[i] == expected_widen(source[i], bf16) &&
          roundtrip[i] == source[i]);
  }
}

typedef struct rounding_case {
  uint32_t source;
  int expected; /* -1 means rejection */
} rounding_case;

static void test_boundaries(const et_kernel_runtime *runtime,
                            size_t format_index) {
  static const rounding_case f16[] = {
      {0x00000000u, 0x0000}, {0x00000001u, 0x0000}, {0x33000000u, 0x0000},
      {0x33000001u, 0x0001}, {0x33800000u, 0x0001}, {0x33c00000u, 0x0002},
      {0x387fc000u, 0x03ff}, {0x387fdfffu, 0x03ff}, {0x387fe000u, 0x0400},
      {0x38800000u, 0x0400}, {0x3f801000u, 0x3c00}, {0x3f801001u, 0x3c01},
      {0x3f803000u, 0x3c02}, {0x3ffff000u, 0x4000}, {0x477fe000u, 0x7bff},
      {0x477fe001u, -1},     {0x477ff000u, -1},     {0x7f800000u, -1},
      {0x7f800001u, -1},     {0x7fc00001u, -1},
  };
  static const rounding_case bf16[] = {
      {0x00000000u, 0x0000}, {0x00008000u, 0x0000}, {0x00008001u, 0x0001},
      {0x00010000u, 0x0001}, {0x00018000u, 0x0002}, {0x007f0000u, 0x007f},
      {0x007f7fffu, 0x007f}, {0x007f8000u, 0x0080}, {0x00800000u, 0x0080},
      {0x3f808000u, 0x3f80}, {0x3f808001u, 0x3f81}, {0x3f818000u, 0x3f82},
      {0x3fff8000u, 0x4000}, {0x7f7f0000u, 0x7f7f}, {0x7f7f7fffu, 0x7f7f},
      {0x7f7f8000u, -1},     {0x7f7fffffu, -1},     {0x7f800000u, -1},
      {0x7f800001u, -1},     {0x7fc00001u, -1},
  };
  const rounding_case *cases = format_index == 0u ? f16 : bf16;
  const size_t count = format_index == 0u ? COUNT(f16) : COUNT(bf16);
  const format *fmt = &formats[format_index];
  et_kernel_error error;
  for (uint32_t sign = 0u; sign <= 1u; ++sign) {
    for (size_t i = 0; i < count; ++i) {
      uint32_t source = cases[i].source | (sign << 31u);
      const uint32_t before = source;
      uint16_t result = 0xdead;
      const int32_t status =
          cast(runtime, fmt, 1, 1u, scalar_batch, &source, &result, &error);
      if (cases[i].expected < 0) {
        CHECK(status == ET_KERNEL_ERROR_INVALID_ARGUMENT && result == 0xdead);
        CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
      } else {
        CHECK(status == 0 &&
              result == ((uint32_t)cases[i].expected | (sign << 15u)));
      }
      CHECK(source == before);
    }
  }
}

static void test_rows(const et_kernel_runtime *runtime, const format *fmt,
                      int bf16) {
  static const struct {
    size_t rank;
    uint64_t shape[3];
  } rows[] = {{1u, {1}},   {1u, {8}},      {1u, {16}},      {1u, {64}},
              {1u, {256}}, {2u, {256, 4}}, {3u, {1, 4, 4}}, {3u, {2, 8, 16}}};
  et_kernel_error error;
  const et_kernel_capability_v1 *cap =
      et_kernel_runtime_capability_find(runtime, fmt->capability);
  const et_kernel_capability_v1 *grad =
      et_kernel_runtime_capability_find(runtime, "kernel.grad-unscale");
  CHECK(cap && grad && cap->shape_range_count == COUNT(rows) &&
        grad->shape_range_count == COUNT(rows) && cap->operation_count == 2u &&
        grad->operation_count == 2u && cap->deterministic &&
        grad->deterministic && cap->status == ET_KERNEL_CAPABILITY_VERIFIED &&
        grad->status == ET_KERNEL_CAPABILITY_VERIFIED);
  for (size_t j = 0; j < COUNT(rows); ++j) {
    const size_t rank = rows[j].rank;
    const uint64_t *shape = rows[j].shape;
    const et_kernel_request_v1 r = request(fmt->narrow, rank, shape);
    CHECK(et_kernel_runtime_capability_require(runtime, fmt->capability, &r,
                                               NULL, &error) == 0);
    uint32_t source[1024];
    uint32_t result[1024];
    uint16_t narrow[1024];
    uint8_t overflow = UINT8_MAX;
    size_t elements = 1u;
    for (size_t i = 0; i < rank; ++i)
      elements *= shape[i];
    for (size_t i = 0; i < elements; ++i)
      source[i] = bits_of(i % 2u == 0u ? 1.0f : -2.0f);
    CHECK(cast(runtime, fmt, 1, rank, shape, source, narrow, &error) == 0);
    for (size_t i = 0; i < elements; ++i) {
      CHECK(narrow[i] == (i % 2u == 0u ? (bf16 ? 0x3f80u : 0x3c00u) : 0xc000u));
    }
    CHECK(cast(runtime, fmt, 0, rank, shape, narrow, result, &error) == 0);
    CHECK(memcmp(source, result, elements * sizeof(source[0])) == 0);
    unscale(runtime, fmt, rank, shape, narrow, 0.5f, result, &overflow);
    CHECK(overflow == 0u);
    for (size_t i = 0; i < elements; ++i)
      CHECK(result[i] == bits_of(i % 2u == 0u ? 0.5f : -1.0f));
  }
}

static void test_unscale(const et_kernel_runtime *runtime, const format *fmt) {
  float source[] = {8.0f, -4.0f, 0.0f, -0.0f, 2.0f, -2.0f, 1.0f, -1.0f};
  uint16_t scaled[COUNT(source)];
  uint32_t master[COUNT(source)];
  uint8_t overflow = UINT8_MAX;
  et_kernel_error error;
  CHECK(cast(runtime, fmt, 1, 1u, vector_batch, source, scaled, &error) == 0);
  unscale(runtime, fmt, 1u, vector_batch, scaled, 0.25f, master, &overflow);
  CHECK(overflow == 0u);
  for (size_t i = 0; i < COUNT(source); ++i)
    CHECK(master[i] == bits_of(source[i] * 0.25f));

  const uint16_t last = scaled[COUNT(scaled) - 1u];
  const uint16_t invalid[] = {
      fmt->infinity, (uint16_t)(fmt->infinity | 0x8000u),
      (uint16_t)(fmt->infinity | 1u), (uint16_t)(fmt->infinity | 0x8040u)};
  for (size_t j = 0; j <= COUNT(invalid); ++j) {
    scaled[COUNT(scaled) - 1u] = j < COUNT(invalid) ? invalid[j] : last;
    const float inverse = j < COUNT(invalid) ? 0.25f : FLT_MAX;
    memset(master, 0xa5, sizeof(master));
    overflow = UINT8_MAX;
    const int flags = fetestexcept(FE_ALL_EXCEPT);
    const unsigned int mxcsr = _mm_getcsr();
    unscale(runtime, fmt, 1u, vector_batch, scaled, inverse, master, &overflow);
    CHECK(overflow == 1u && fetestexcept(FE_ALL_EXCEPT) == flags &&
          _mm_getcsr() == mxcsr);
    for (size_t i = 0; i < COUNT(master); ++i)
      CHECK(master[i] == 0u);
  }
  unscale(runtime, fmt, 1u, vector_batch, scaled, 0.125f, master, &overflow);
  CHECK(overflow == 0u && master[0] == bits_of(1.0f));
  unscale(runtime, fmt, 1u, vector_batch, scaled, float_of(1u), master,
          &overflow);
  CHECK(overflow == 0u && master[0] == 8u && master[1] == UINT32_C(0x80000004));

  /* The largest bf16 stays finite for a small inverse that removes its
   * exponent. */
  if (fmt->infinity == 0x7f80u) {
    uint16_t largest = 0x7f7fu;
    unscale(runtime, fmt, 1u, scalar_batch, &largest, 0x1p-127f, master,
            &overflow);
    CHECK(overflow == 0u && master[0] == UINT32_C(0x3fff0000));
    uint16_t tiniest = 0x8001u;
    unscale(runtime, fmt, 1u, scalar_batch, &tiniest, 0x1p-17f, master,
            &overflow);
    CHECK(overflow == 0u && master[0] == UINT32_C(0x80000000));
  }
}

static void test_failures(const et_kernel_runtime *runtime) {
  const format *fmt = &formats[0];
  et_kernel_error error;
  uint64_t shape[] = {8u};
  et_kernel_request_v1 r = request(fmt->narrow, 1u, shape);
  uint32_t source[] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0x477fe001u};
  uint32_t source_before[COUNT(source)];
  uint16_t result[COUNT(source)];
  uint16_t before[COUNT(result)];
  memcpy(source_before, source, sizeof(source));
  memset(result, 0xa5, sizeof(result));
  memcpy(before, result, sizeof(before));
  const et_kernel_tensor_view_v1 in = view(source, "f32", 1u, shape);
  et_kernel_tensor_view_v1 out = view(result, "f16", 1u, shape);
  et_kernel_call_v1 call = make_call(fmt->capability, &r, 1u, &in, &out);
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(result, before, sizeof(result)) == 0);
  CHECK(memcmp(source, source_before, sizeof(source)) == 0);
  memset(source, 0, sizeof(source));

  const et_kernel_tensor_view_v1 correct = out;
  const uint64_t other_shape[] = {16u};
  const et_kernel_tensor_view_v1 malformed[] = {
      view(result, "bf16", 1u, shape),
      view(result, "f16", 1u, other_shape),
      view(result, "f16", 0u, NULL),
      view(NULL, "f16", 1u, shape),
      view((unsigned char *)result + 1u, "f16", 1u, shape),
  };
  for (size_t i = 0; i < COUNT(malformed); ++i) {
    out = malformed[i];
    CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) != 0);
  }
  out = correct;
  out.byte_length += 2u;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) != 0);
  out = correct;
  out.offset_bytes = 2u;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_NONCONTIGUOUS);
  out = correct;
  const uint64_t unsupported_shape[] = {2u, 4u};
  const et_kernel_tensor_view_v1 unsupported_in =
      view(source, "f32", 2u, unsupported_shape);
  out = view(result, "f16", 2u, unsupported_shape);
  r = request(fmt->narrow, 2u, unsupported_shape);
  call = make_call(fmt->capability, &r, 1u, &unsupported_in, &out);
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  r = request(fmt->narrow, 1u, shape);
  out = correct;
  call = make_call(fmt->capability, &r, 1u, &in, &out);
  call.capability = formats[1].capability;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  call.capability = fmt->capability;
  r.dtype = "f16";
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  r.dtype = "f32";
  r.device = "cuda";
  CHECK(et_kernel_runtime_capability_require(runtime, fmt->capability, &r, NULL,
                                             &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  r.device = "cpu";
  r.deterministic = 2u;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  r.deterministic = 1u;
  call.input_bytes -= 1u;
  CHECK(et_mp4_kernel_provider_v1()->validate_call(&call, &error) != 0);
  CHECK(memcmp(result, before, sizeof(result)) == 0);

  uint16_t storage[COUNT(source)] = {0u};
  float inverse = 1.0f;
  uint8_t flag = 0xbau;
  uint32_t master[COUNT(source)];
  uint32_t master_before[COUNT(master)];
  memset(master, 0x5a, sizeof(master));
  memcpy(master_before, master, sizeof(master));
  r = request(fmt->unscale, 1u, shape);
  et_kernel_tensor_view_v1 grad_in[] = {view(storage, "f16", 1u, shape),
                                        view(&inverse, "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 grad_out[] = {view(master, "f32", 1u, shape),
                                         view(&flag, "bool", 0u, NULL)};
  call = make_call("kernel.grad-unscale", &r, 2u, grad_in, grad_out);
  const uint32_t invalid_inverse[] = {0u,          0x80000000u, 0xbf800000u,
                                      0x7f800000u, 0xff800000u, 0x7f800001u,
                                      0x7fc00001u};
  for (size_t i = 0; i < COUNT(invalid_inverse); ++i) {
    memcpy(&inverse, &invalid_inverse[i], sizeof(inverse));
    CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED && flag == 0xbau);
    CHECK(memcmp(master, master_before, sizeof(master)) == 0);
    CHECK(bits_of(inverse) == invalid_inverse[i]);
  }
  inverse = 1.0f;
  grad_in[1] = view(&inverse, "f32", 1u, scalar_batch);
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_SHAPE_MISMATCH);
  grad_in[1] = view(&inverse, "f32", 0u, NULL);
  grad_out[1].dtype = "f16";
  grad_out[1].byte_length = 2u;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
        ET_KERNEL_ERROR_DTYPE_MISMATCH);
  CHECK(memcmp(master, master_before, sizeof(master)) == 0 && flag == 0xbau);
}

static void test_aliases(const et_kernel_runtime *runtime) {
  et_kernel_error error;
  _Alignas(8) char capability[128] = "kernel.cast-f16";
  _Alignas(8) char operation[128] = "cast.f32-to-f16";
  _Alignas(8) char dtype[128] = "f32";
  _Alignas(8) char device[128] = "cpu";
  _Alignas(8) char in_dtype[128] = "f32";
  _Alignas(8) char out_dtype[128] = "f16";
  _Alignas(8) char in_device[128] = "cpu";
  _Alignas(8) char out_device[128] = "cpu";
  uint64_t request_shape[] = {1u}, in_shape[] = {1u}, out_shape[] = {1u};
  struct {
    et_kernel_request_v1 prefix;
    uint64_t extension;
  } r = {request(operation, 1u, request_shape), UINT64_C(0x123456789)};
  r.prefix.struct_size = sizeof(r);
  r.prefix.dtype = dtype;
  r.prefix.device = device;
  uint32_t source = 0x3f800000u;
  uint16_t result = 0xbeefu;
  typedef struct {
    et_kernel_tensor_view_v1 prefix;
    uint64_t extension;
  } entry;
  entry in = {view(&source, in_dtype, 1u, in_shape), UINT64_C(0x987654321)};
  entry out = {view(&result, out_dtype, 1u, out_shape), UINT64_C(0x135792468)};
  in.prefix.device = in_device;
  out.prefix.device = out_device;
  in.prefix.struct_size = sizeof(in);
  out.prefix.struct_size = sizeof(out);
  struct {
    et_kernel_call_v1 prefix;
    uint64_t extension;
  } call = {make_call(capability, &r.prefix, 1u, &in, &out),
            UINT64_C(0x246813579)};
  call.prefix.struct_size = sizeof(call);
  call.prefix.input_stride = call.prefix.input_bytes = sizeof(in);
  call.prefix.output_stride = call.prefix.output_bytes = sizeof(out);

  const et_kernel_provider_v1 *provider = et_mp4_kernel_provider_v1();
  void *aliased[] = {
      &call.prefix,   &call.extension, &r.prefix,
      &r.extension,   request_shape,   capability,
      operation,      dtype,           device,
      &in.prefix,     &in.extension,   in_shape,
      in_dtype,       in_device,       &out.prefix,
      &out.extension, out_shape,       out_dtype,
      out_device,     &source,         operation + strlen(operation) - 1u};
  for (size_t i = 0; i < COUNT(aliased); ++i) {
    out.prefix.data = aliased[i];
    unsigned char before[sizeof(result)];
    memcpy(before, aliased[i], sizeof(before));
    CHECK(provider->validate_call(&call.prefix, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
    CHECK(et_kernel_runtime_dispatch(runtime, &call.prefix, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
    CHECK(memcmp(aliased[i], before, sizeof(before)) == 0);
  }
  out.prefix.data = &result;
  CHECK(result == 0xbeefu &&
        provider->validate_call(&call.prefix, &error) == 0);
  CHECK(result == 0xbeefu);
  dispatch(runtime, &call.prefix);
  CHECK(result == 0x3c00u);

  memset(&error, 0x5a, sizeof(error));
  unsigned char error_snapshot[sizeof(error)];
  memcpy(error_snapshot, &error, sizeof(error_snapshot));
  out.prefix.data = &error;
  CHECK(provider->validate_call(&call.prefix, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&error, error_snapshot, sizeof(error_snapshot)) == 0);

  uint16_t scaled = 0x3c00u;
  float inverse = 0.5f;
  uint8_t flag = 0xabu;
  uint32_t master = 0x87654321u;
  r.prefix = request(formats[0].unscale, 1u, request_shape);
  et_kernel_tensor_view_v1 grad_in[] = {view(&scaled, "f16", 1u, in_shape),
                                        view(&inverse, "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 grad_out[] = {view(&master, "f32", 1u, out_shape),
                                         view(&flag, "bool", 0u, NULL)};
  call.prefix =
      make_call("kernel.grad-unscale", &r.prefix, 2u, grad_in, grad_out);
  void *flag_aliases[] = {
      &master,      &inverse,
      &scaled,      &call.prefix,
      &grad_out[0], (char *)r.prefix.operation + strlen(r.prefix.operation)};
  for (size_t i = 0; i < COUNT(flag_aliases); ++i) {
    grad_out[1].data = flag_aliases[i];
    unsigned char byte;
    memcpy(&byte, flag_aliases[i], sizeof(byte));
    CHECK(provider->validate_call(&call.prefix, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
    CHECK(et_kernel_runtime_dispatch(runtime, &call.prefix, &error) ==
          ET_KERNEL_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
    CHECK(memcmp(flag_aliases[i], &byte, sizeof(byte)) == 0);
    CHECK(master == 0x87654321u && flag == 0xabu);
  }
}

static uint16_t x87_control(void) {
  uint16_t control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  return control;
}

static void set_x87_control(uint16_t control) {
  __asm__ volatile("fldcw %0" : : "m"(control));
}

static void test_fenv(const et_kernel_runtime *runtime) {
  fenv_t original;
  CHECK(fegetenv(&original) == 0);
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  const uint16_t x87 = x87_control();
  const unsigned int mxcsr = _mm_getcsr();
  uint32_t input = 0x3f801001u;
  uint16_t output = 0xabcdu;
  et_kernel_error error;
  const et_kernel_request_v1 r = request(formats[0].narrow, 1u, scalar_batch);
  const et_kernel_tensor_view_v1 in = view(&input, "f32", 1u, scalar_batch);
  et_kernel_tensor_view_v1 out = view(&output, "f16", 1u, scalar_batch);
  const et_kernel_call_v1 call =
      make_call(formats[0].capability, &r, 1u, &in, &out);
  const et_kernel_provider_v1 *provider = et_mp4_kernel_provider_v1();

  CHECK(feraiseexcept(FE_INEXACT | FE_DIVBYZERO) == 0);
  const int flags = fetestexcept(FE_ALL_EXCEPT);
  const unsigned int mxcsr_flags = _mm_getcsr();
  errno = EDOM;
  CHECK(provider->validate_call(&call, &error) == 0 && output == 0xabcdu);
  CHECK(fetestexcept(FE_ALL_EXCEPT) == flags && errno == EDOM &&
        _mm_getcsr() == mxcsr_flags && x87_control() == x87);
  input = 0x7f800001u;
  CHECK(provider->validate_call(&call, &error) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(output == 0xabcdu && fetestexcept(FE_ALL_EXCEPT) == flags &&
        errno == EDOM);
  input = 0x3f801001u;
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);

  const unsigned int bad_mxcsr[] = {mxcsr | 0x8000u, mxcsr | 0x40u,
                                    mxcsr | 0x4000u, mxcsr & ~0x400u};
  for (size_t i = 0; i < COUNT(bad_mxcsr); ++i) {
    _mm_setcsr(bad_mxcsr[i]);
    CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(output == 0xabcdu && _mm_getcsr() == bad_mxcsr[i]);
  }
  _mm_setcsr(mxcsr);
  const uint16_t bad_x87[] = {(uint16_t)(x87 | 0x0400u),
                              (uint16_t)(x87 & ~0x08u)};
  for (size_t i = 0; i < COUNT(bad_x87); ++i) {
    set_x87_control(bad_x87[i]);
    CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) ==
          ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(output == 0xabcdu && x87_control() == bad_x87[i]);
  }
  set_x87_control(x87);

  et_mp4_loss_scaler_v1 scale;
  CHECK(et_mp4_loss_scaler_init(&scale, 3.0f, 1.1f, 0.5f, 1u, 1.0f, 16.0f) ==
        0);
  CHECK(feraiseexcept(FE_DIVBYZERO) == 0);
  const int before_growth = fetestexcept(FE_ALL_EXCEPT);
  CHECK(et_mp4_loss_scaler_update(&scale, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(fetestexcept(FE_ALL_EXCEPT) == before_growth);
  const et_mp4_loss_scaler_v1 snapshot = scale;
  CHECK(fesetround(FE_DOWNWARD) == 0);
  CHECK(et_mp4_loss_scaler_update(&scale, 1u) ==
        ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT);
  CHECK(memcmp(&scale, &snapshot, sizeof(scale)) == 0 &&
        fegetround() == FE_DOWNWARD);
  CHECK(fesetenv(&original) == 0);
}

static void test_scaler(void) {
  et_mp4_loss_scaler_v1 state;
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 3.0f, 0.5f, 2u, 2.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(state.scale == 8.0f && state.clean_steps == 0u);
  CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(state.scale == 8.0f && state.clean_steps == 1u);
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
  CHECK(state.scale == 4.0f && state.clean_steps == 0u);
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
  CHECK(state.scale == 2.0f && state.clean_steps == 0u);
  CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  const et_mp4_loss_scaler_v1 before_fault = state;
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT);
  CHECK(memcmp(&state, &before_fault, sizeof(state)) == 0);

  CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(state.scale == 6.0f && state.clean_steps == 0u);
  const float expected[] = {18.0f, 32.0f, 32.0f};
  for (size_t i = 0; i < COUNT(expected); ++i) {
    CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
          ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
    CHECK(state.clean_steps == 1u);
    CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
          ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
    CHECK(state.clean_steps == 0u && state.scale == expected[i]);
  }

  const et_mp4_loss_scaler_v1 initial = state;
  CHECK(et_mp4_loss_scaler_init(NULL, 8.0f, 2.0f, 0.5f, 2u, 1.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 2.0f, 0.5f, 0u, 1.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_init(&state, 0.5f, 2.0f, 0.5f, 1u, 1.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 1.0f, 0.5f, 2u, 1.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 2.0f, 1.0f, 2u, 1.0f, 32.0f) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_init(&state, 8.0f, 2.0f, 0.5f, 2u, 1.0f, INFINITY) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_update(NULL, 0u) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_loss_scaler_update(&state, 2u) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(memcmp(&state, &initial, sizeof(state)) == 0);
  state.clean_steps = state.growth_interval;
  const et_mp4_loss_scaler_v1 invalid = state;
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(memcmp(&state, &invalid, sizeof(state)) == 0);

  CHECK(et_mp4_loss_scaler_init(&state, FLT_MAX, FLT_MAX, 0.5f, UINT32_MAX,
                                1.0f, FLT_MAX) == 0);
  state.clean_steps = UINT32_MAX - 1u;
  CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(state.clean_steps == 0u && state.scale == FLT_MAX);
  CHECK(et_mp4_loss_scaler_init(&state, float_of(2u), 2.0f, 0.75f, 1u,
                                float_of(2u), 1.0f) == 0);
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT);
  CHECK(bits_of(state.scale) == 2u && state.clean_steps == 0u);

  /* Backoff whose exact product exceeds min_scale but rounds back to scale
   * must fault rather than stalling at an unchanged scale. */
  CHECK(et_mp4_loss_scaler_init(&state, 0x1p-127f, 2.0f, 0x1.fffffep-1f, 2u,
                                0x1p-128f, 1.0f) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(et_mp4_loss_scaler_update(&state, 0u) ==
        ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  const et_mp4_loss_scaler_v1 before_stall = state;
  CHECK(et_mp4_loss_scaler_update(&state, 1u) ==
        ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT);
  CHECK(memcmp(&state, &before_stall, sizeof(state)) == 0);
}

typedef struct train_result {
  float weights[8];
  float velocity[8];
  et_mp4_loss_scaler_v1 scaler;
  float loss;
  unsigned int steps;
} train_result;

static float quadratic_loss(const float *weights, const float *target,
                            size_t count) {
  float loss = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float d = weights[i] - target[i];
    loss += 0.5f * d * d;
  }
  return loss;
}

static train_result train(const et_kernel_runtime *runtime, const format *fmt,
                          int inject_overflow) {
  static const float target[] = {0.5f,  -1.0f, 1.5f,  -2.0f,
                                 0.25f, -0.5f, 0.75f, -1.0f};
  train_result state = {0};
  et_kernel_error error;
  CHECK(et_mp4_loss_scaler_init(&state.scaler, 32.0f, 2.0f, 0.5f, 2u, 1.0f,
                                256.0f) == 0);
  const float initial_loss =
      quadratic_loss(state.weights, target, COUNT(target));
  state.loss = initial_loss;
  for (size_t i = 0; i < 13u; ++i) {
    if (i == 2u && !inject_overflow)
      continue;
    uint16_t storage[COUNT(target)];
    float low_precision[COUNT(target)], scaled[COUNT(target)],
        grads[COUNT(target)];
    CHECK(cast(runtime, fmt, 1, 1u, vector_batch, state.weights, storage,
               &error) == 0);
    CHECK(cast(runtime, fmt, 0, 1u, vector_batch, storage, low_precision,
               &error) == 0);
    for (size_t j = 0; j < COUNT(target); ++j)
      scaled[j] = (low_precision[j] - target[j]) * state.scaler.scale;
    CHECK(cast(runtime, fmt, 1, 1u, vector_batch, scaled, storage, &error) ==
          0);
    if (i == 2u)
      storage[COUNT(target) - 1u] = (uint16_t)(fmt->infinity | 1u);
    uint8_t overflow = 0xffu;
    unscale(runtime, fmt, 1u, vector_batch, storage, 1.0f / state.scaler.scale,
            grads, &overflow);
    const train_result before = state;
    const et_mp4_loss_scaler_status step =
        et_mp4_loss_scaler_update(&state.scaler, overflow);
    if (step == ET_MP4_LOSS_SCALER_STEP_ACCEPTED) {
      CHECK(i != 2u && overflow == 0u);
      for (size_t j = 0; j < COUNT(target); ++j) {
        state.velocity[j] = 0.25f * state.velocity[j] + grads[j];
        state.weights[j] -= 0.25f * state.velocity[j];
      }
      state.loss = quadratic_loss(state.weights, target, COUNT(target));
      CHECK(state.loss < before.loss);
      ++state.steps;
    } else {
      CHECK(i == 2u && overflow == 1u &&
            step == ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
      CHECK(memcmp(state.weights, before.weights, sizeof(state.weights)) == 0 &&
            memcmp(state.velocity, before.velocity, sizeof(state.velocity)) ==
                0);
      CHECK(state.loss == before.loss && state.steps == before.steps);
      CHECK(before.scaler.scale == 64.0f && state.scaler.scale == 32.0f &&
            state.scaler.clean_steps == 0u);
      for (size_t j = 0; j < COUNT(target); ++j)
        CHECK(bits_of(grads[j]) == 0u);
    }
    if (i == 0u) {
      for (size_t j = 0; j < COUNT(target); ++j)
        CHECK(state.weights[j] == target[j] * 0.25f);
    }
    if (i == 3u)
      CHECK(state.steps == 3u && state.loss < before.loss);
  }
  CHECK(state.steps == 12u && state.loss < initial_loss * 0.001f);
  return state;
}

static void test_training(const et_kernel_runtime *runtime, const format *fmt) {
  const train_result first = train(runtime, fmt, 1);
  const train_result repeat = train(runtime, fmt, 1);
  const train_result skipped = train(runtime, fmt, 0);
  CHECK(memcmp(first.weights, repeat.weights, sizeof(first.weights)) == 0 &&
        memcmp(first.velocity, repeat.velocity, sizeof(first.velocity)) == 0 &&
        memcmp(&first.scaler, &repeat.scaler, sizeof(first.scaler)) == 0 &&
        bits_of(first.loss) == bits_of(repeat.loss));
  CHECK(memcmp(first.weights, skipped.weights, sizeof(first.weights)) == 0 &&
        memcmp(first.velocity, skipped.velocity, sizeof(first.velocity)) == 0 &&
        bits_of(first.loss) == bits_of(skipped.loss));
  printf("MP4 %s training: 12 accepted steps, one injected overflow skipped, "
         "final loss %.9g\n",
         fmt->dtype, (double)first.loss);
}

static void oracle_stream(const et_kernel_runtime *runtime) {
  char kind;
  uint32_t word;
  et_kernel_error error;
  int scanned;
  while ((scanned = scanf(" %c %" SCNx32, &kind, &word)) == 2) {
    CHECK(kind == 'h' || kind == 'b' || kind == 'H' || kind == 'B');
    const format *fmt = &formats[kind == 'b' || kind == 'B'];
    const int narrow = kind == 'h' || kind == 'b';
    uint32_t result32 = UINT32_MAX;
    uint16_t word16 = (uint16_t)word;
    uint16_t result16 = UINT16_MAX;
    CHECK(narrow || word <= UINT16_MAX);
    const int32_t status =
        cast(runtime, fmt, narrow, 1u, scalar_batch,
             narrow ? (void *)&word : (void *)&word16,
             narrow ? (void *)&result16 : (void *)&result32, &error);
    if (status != 0) {
      CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(result32 == UINT32_MAX && result16 == UINT16_MAX);
      puts("reject");
    } else {
      printf("%08" PRIx32 "\n", narrow ? (uint32_t)result16 : result32);
    }
  }
  CHECK(scanned == EOF && !ferror(stdin));
}

int main(int argc, char **argv) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mp4, NULL, &runtime, &error) == 0);
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    oracle_stream(runtime);
  } else {
    CHECK(argc == 1);
    for (size_t i = 0; i < COUNT(formats); ++i) {
      test_exhaustive(runtime, &formats[i], i == 1u);
      test_boundaries(runtime, i);
      test_rows(runtime, &formats[i], i == 1u);
      test_unscale(runtime, &formats[i]);
      test_training(runtime, &formats[i]);
    }
    test_aliases(runtime);
    test_failures(runtime);
    test_fenv(runtime);
    test_scaler();
    puts("MP4 K1 storage, exhaustive round trips, unscale, scaler and "
         "atomicity: PASS");
  }
  et_kernel_runtime_destroy(runtime);
  return 0;
}
