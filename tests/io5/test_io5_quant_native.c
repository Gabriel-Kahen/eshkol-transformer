#include "eshkol_transformer/io5_quant_abi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ET_IO5_QUANT_WRAP_ALLOC
static int g_forbid_alloc = 0;
void *__real_malloc(size_t size);
void *__real_calloc(size_t nmemb, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);

void *__wrap_malloc(size_t size) {
  if (g_forbid_alloc) {
    fprintf(stderr, "FAIL: unexpected malloc(%zu) inside IO5 quant codec\n",
            size);
    abort();
  }
  return __real_malloc(size);
}

void *__wrap_calloc(size_t nmemb, size_t size) {
  if (g_forbid_alloc) {
    fprintf(stderr, "FAIL: unexpected calloc(%zu, %zu) inside IO5 quant codec\n",
            nmemb, size);
    abort();
  }
  return __real_calloc(nmemb, size);
}

void *__wrap_realloc(void *ptr, size_t size) {
  if (g_forbid_alloc) {
    fprintf(stderr, "FAIL: unexpected realloc(%p, %zu) inside IO5 quant codec\n",
            ptr, size);
    abort();
  }
  return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr) {
  if (g_forbid_alloc) {
    fprintf(stderr, "FAIL: unexpected free(%p) inside IO5 quant codec\n", ptr);
    abort();
  }
  __real_free(ptr);
}
#endif

static void check_cond(int cond, const char *msg, int line) {
  if (!cond) {
    fprintf(stderr, "FAIL line %d: %s\n", line, msg);
    exit(1);
  }
}
#define CHECK(c, m) check_cond((c), (m), __LINE__)

static uint32_t f32_to_bits(float v) {
  uint32_t b;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static float bits_to_f32(uint32_t b) {
  float v;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static void test_exhaustive_f16(void) {
  size_t finite_count = 0u;
  size_t nonfinite_count = 0u;
  for (uint32_t raw = 0u; raw < 65536u; ++raw) {
    const uint16_t h = (uint16_t)raw;
    const int is_nonfinite = ((h & UINT16_C(0x7c00)) == UINT16_C(0x7c00));
    uint32_t f32_bits = UINT32_C(0xdeadbeef);
    const et_io5_quant_status st = et_io5_f16_bits_to_f32_bits(h, &f32_bits);
    if (is_nonfinite) {
      CHECK(st == ET_IO5_QUANT_ERR_NONFINITE, "f16 nonfinite must reject");
      CHECK(f32_bits == UINT32_C(0xdeadbeef),
            "f16 nonfinite must not overwrite output");
      ++nonfinite_count;
    } else {
      CHECK(st == ET_IO5_QUANT_OK, "finite f16 widen must succeed");
      /* Round-trip f32 -> f16 must reproduce identical 16-bit pattern. */
      uint16_t rt = UINT16_C(0xbeef);
      CHECK(et_io5_f32_bits_to_f16_bits(f32_bits, &rt) == ET_IO5_QUANT_OK,
            "finite f16 round-trip narrow must succeed");
      CHECK(rt == h, "finite f16 bit-exact round-trip failed");
      /* Sign bit preservation including signed zero. */
      CHECK(((f32_bits >> 31u) & 1u) == ((uint32_t)(h >> 15u) & 1u),
            "f16 sign bit preservation");
      ++finite_count;
    }
  }
  CHECK(finite_count == 63488u && nonfinite_count == 2048u,
        "exhaustive f16 partition count");

  /* Signed zero exact checks. */
  uint16_t h_out = 0u;
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x00000000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x0000),
        "+0.0f -> +0.0 f16");
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x80000000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x8000),
        "-0.0f -> -0.0 f16");

  /* Underflow to signed zero & subnormal tie-to-even boundaries. */
  /* 2^-25 = 0x33000000 -> exact tie between 0 (even) and 2^-24 (odd) -> 0. */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x33000000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x0000),
        "f16 tie-to-even at 2^-25 -> +0");
  /* Just above 2^-25 -> rounds up to smallest subnormal 0x0001. */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x33000001), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x0001),
        "f16 above 2^-25 -> 0x0001");
  /* Negative tiny value preserves -0.0. */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x80000001), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x8000),
        "negative tiny f32 -> -0.0 f16");

  /* Normal RNE tie-to-even check: 1.0 + 2^-11 = 0x3f801000 (tie, LSB even -> 0x3c00). */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x3f801000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x3c00),
        "f16 RNE tie to even down");
  /* 1.0 + 3*2^-11 = 0x3f803000 (tie, LSB odd -> rounds up to 0x3c02). */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x3f803000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x3c02),
        "f16 RNE tie to even up");

  /* F16 max boundary (65504.0f = 0x477fe000) and overflow rejections. */
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x477fe000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0x7bff),
        "+65504.0f -> 0x7bff");
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0xc77fe000), &h_out) ==
            ET_IO5_QUANT_OK &&
        h_out == UINT16_C(0xfbff),
        "-65504.0f -> 0xfbff");
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0x477fe001), &h_out) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "nextafter(65504.0f) rejects with OVERFLOW");
  CHECK(et_io5_f32_bits_to_f16_bits(UINT32_C(0xc77fe001), &h_out) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "-nextafter(65504.0f) rejects with OVERFLOW");
  CHECK(et_io5_f32_bits_to_f16_bits(f32_to_bits(65520.0f), &h_out) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "65520.0f rejects with OVERFLOW");

  /* Non-finite f32 rejections. */
  const uint32_t bad_f32[] = {UINT32_C(0x7f800000), UINT32_C(0xff800000),
                              UINT32_C(0x7fc00000), UINT32_C(0x7f800001),
                              UINT32_C(0xffffffff)};
  for (size_t i = 0u; i < sizeof(bad_f32) / sizeof(bad_f32[0]); ++i) {
    CHECK(et_io5_f32_bits_to_f16_bits(bad_f32[i], &h_out) ==
              ET_IO5_QUANT_ERR_NONFINITE,
          "nonfinite f32 -> f16 must reject");
  }
}

static void test_exhaustive_bf16(void) {
  size_t finite_count = 0u;
  size_t nonfinite_count = 0u;
  for (uint32_t raw = 0u; raw < 65536u; ++raw) {
    const uint16_t b = (uint16_t)raw;
    const int is_nonfinite = ((b & UINT16_C(0x7f80)) == UINT16_C(0x7f80));
    uint32_t f32_bits = UINT32_C(0xdeadbeef);
    const et_io5_quant_status st = et_io5_bf16_bits_to_f32_bits(b, &f32_bits);
    if (is_nonfinite) {
      CHECK(st == ET_IO5_QUANT_ERR_NONFINITE, "bf16 nonfinite must reject");
      CHECK(f32_bits == UINT32_C(0xdeadbeef),
            "bf16 nonfinite must not overwrite output");
      ++nonfinite_count;
    } else {
      CHECK(st == ET_IO5_QUANT_OK, "finite bf16 widen must succeed");
      CHECK(f32_bits == ((uint32_t)b << 16u), "bf16 widen exact shift");
      uint16_t rt = UINT16_C(0xbeef);
      CHECK(et_io5_f32_bits_to_bf16_bits(f32_bits, &rt) == ET_IO5_QUANT_OK,
            "finite bf16 round-trip narrow must succeed");
      CHECK(rt == b, "finite bf16 bit-exact round-trip failed");
      ++finite_count;
    }
  }
  CHECK(finite_count == 65280u && nonfinite_count == 256u,
        "exhaustive bf16 partition count");

  /* BF16 RNE ties-to-even & overflow boundary checks. */
  uint16_t b_out = 0u;
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0x3f808000), &b_out) ==
            ET_IO5_QUANT_OK &&
        b_out == UINT16_C(0x3f80),
        "bf16 RNE tie to even down");
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0x3f818000), &b_out) ==
            ET_IO5_QUANT_OK &&
        b_out == UINT16_C(0x3f82),
        "bf16 RNE tie to even up");
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0x7f7f7fff), &b_out) ==
            ET_IO5_QUANT_OK &&
        b_out == UINT16_C(0x7f7f),
        "bf16 max finite boundary accepted");
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0x7f7f8000), &b_out) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "bf16 rounding to +Inf rejected");
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0xff7f8000), &b_out) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "bf16 rounding to -Inf rejected");
  CHECK(et_io5_f32_bits_to_bf16_bits(UINT32_C(0x80000000), &b_out) ==
            ET_IO5_QUANT_OK &&
        b_out == UINT16_C(0x8000),
        "-0.0f -> -0.0 bf16");
}

static void test_bulk_narrow_and_unaligned(void) {
  /* Deliberately misalign buffers by +1 and +3 bytes to prove zero UB. */
  uint8_t src_backing[64 * 4 + 8];
  uint8_t mid_backing[64 * 2 + 8];
  uint8_t dst_backing[64 * 4 + 8];
  uint8_t *src = src_backing + 1;
  uint8_t *mid = mid_backing + 3;
  uint8_t *dst = dst_backing + 1;

  for (size_t i = 0u; i < 64u; ++i) {
    const float v = (float)((int)i - 32) * 0.125f;
    const uint32_t bits = f32_to_bits(v);
    memcpy(src + i * 4u, &bits, 4u);
  }
  CHECK(et_io5_encode_f16(src, 64u * 4u, mid, 64u * 2u, 64u) == ET_IO5_QUANT_OK,
        "unaligned bulk f16 encode");
  CHECK(et_io5_decode_f16(mid, 64u * 2u, dst, 64u * 4u, 64u) == ET_IO5_QUANT_OK,
        "unaligned bulk f16 decode");
  CHECK(memcmp(src, dst, 64u * 4u) == 0, "exact dyadic f16 round-trip");

  CHECK(et_io5_encode_bf16(src, 64u * 4u, mid, 64u * 2u, 64u) == ET_IO5_QUANT_OK,
        "unaligned bulk bf16 encode");
  CHECK(et_io5_decode_bf16(mid, 64u * 2u, dst, 64u * 4u, 64u) == ET_IO5_QUANT_OK,
        "unaligned bulk bf16 decode");
  CHECK(memcmp(src, dst, 64u * 4u) == 0, "exact dyadic bf16 round-trip");

  /* Empty (0-element) bulk narrow conversion succeeds cleanly with non-NULL. */
  CHECK(et_io5_encode_f16(src, 0u, mid, 0u, 0u) == ET_IO5_QUANT_OK,
        "empty f16 encode");
  CHECK(et_io5_decode_f16(mid, 0u, dst, 0u, 0u) == ET_IO5_QUANT_OK,
        "empty f16 decode");
  CHECK(et_io5_encode_bf16(src, 0u, mid, 0u, 0u) == ET_IO5_QUANT_OK,
        "empty bf16 encode");
  CHECK(et_io5_decode_bf16(mid, 0u, dst, 0u, 0u) == ET_IO5_QUANT_OK,
        "empty bf16 decode");

  /* Destination must remain untouched when last element fails validation. */
  memset(mid, 0xA5, 64u * 2u);
  const uint32_t bad = f32_to_bits(70000.0f);
  memcpy(src + 63u * 4u, &bad, 4u);
  CHECK(et_io5_encode_f16(src, 64u * 4u, mid, 64u * 2u, 64u) ==
            ET_IO5_QUANT_ERR_OVERFLOW,
        "last-element overflow must reject");
  for (size_t i = 0u; i < 64u * 2u; ++i) {
    CHECK(mid[i] == 0xA5u, "failed f16 encode must not mutate dst");
  }
}

static void test_q8_0_roundtrip_and_edges(void) {
  /* 1. Multi-block Q8_0 round-trip with varied dynamic ranges and zero block. */
  float x[128];
  /* Block 0: linear ramp in [-2.0, 2.0]. */
  for (size_t i = 0u; i < 32u; ++i) {
    x[i] = -2.0f + (4.0f * (float)i) / 31.0f;
  }
  /* Block 1: all zeros including signed -0.0f. */
  for (size_t i = 0u; i < 32u; ++i) {
    x[32u + i] = (i & 1u) ? -0.0f : +0.0f;
  }
  /* Block 2: large range near [-1000.0, 1000.0]. */
  for (size_t i = 0u; i < 32u; ++i) {
    x[64u + i] = ((float)((int)i - 15) / 16.0f) * 1000.0f;
  }
  /* Block 3: tiny subnormal-scale values around 1e-7 (scale underflows f16 to 0). */
  for (size_t i = 0u; i < 32u; ++i) {
    x[96u + i] = ((float)((int)i - 16)) * 1.0e-8f;
  }

  /* Use deliberately unaligned Q8_0 and output buffers. */
  uint8_t q8_backing[4 * 34 + 5];
  uint8_t out_backing[128 * 4 + 5];
  uint8_t *q8 = q8_backing + 1;
  uint8_t *out_bytes = out_backing + 3;

  CHECK(et_io5_quantize_q8_0(x, sizeof(x), q8, 4u * 34u, 128u) ==
            ET_IO5_QUANT_OK,
        "Q8_0 quantize 4 blocks");

  /* Verify zero block (block 1) encodes as d=0x0000 and all qs=0. */
  CHECK(q8[34u] == 0u && q8[35u] == 0u, "zero block scale is +0.0 f16");
  for (size_t i = 0u; i < 32u; ++i) {
    CHECK(q8[34u + 2u + i] == 0u, "zero block quants are 0");
  }
  /* Verify all emitted quants are within [-127, 127]. */
  for (size_t b = 0u; b < 4u; ++b) {
    for (size_t i = 0u; i < 32u; ++i) {
      const int8_t q = (int8_t)q8[b * 34u + 2u + i];
      CHECK(q >= -127 && q <= 127, "Q8_0 quant clamped to [-127, 127]");
    }
  }

  CHECK(et_io5_dequantize_q8_0(q8, 4u * 34u, out_bytes, 128u * 4u, 128u) ==
            ET_IO5_QUANT_OK,
        "Q8_0 dequantize 4 blocks");

  float x_hat[128];
  memcpy(x_hat, out_bytes, sizeof(x_hat));

  /* Block 0: max_abs <= d_stored * 0.5 + tiny rounding (amax=2 -> d ~= 0.01575). */
  for (size_t i = 0u; i < 32u; ++i) {
    const float err = fabsf(x[i] - x_hat[i]);
    CHECK(err <= 0.0085f, "Q8_0 block 0 error bound");
  }
  /* Block 1: exact zero reconstruction. */
  for (size_t i = 0u; i < 32u; ++i) {
    CHECK(x_hat[32u + i] == 0.0f, "Q8_0 zero block exact zero");
  }
  /* Block 2: max_abs <= amax / 254 + f16 scale rounding (~4.5 for amax=1000). */
  for (size_t i = 0u; i < 32u; ++i) {
    const float err = fabsf(x[64u + i] - x_hat[64u + i]);
    CHECK(err <= 4.5f, "Q8_0 block 2 error bound");
  }
  /* Block 3: underflowed scale -> reconstructed 0.0, abs error <= 1.6e-7. */
  for (size_t i = 0u; i < 32u; ++i) {
    CHECK(x_hat[96u + i] == 0.0f, "Q8_0 underflowed scale block -> 0.0");
  }

  /* 2. Exact maximum representable Q8_0 scale boundary (127 * 65504 = 8319008). */
  float max_blk[32];
  memset(max_blk, 0, sizeof(max_blk));
  max_blk[0] = 8319008.0f;
  max_blk[1] = -8319008.0f;
  uint8_t q8_max[34];
  CHECK(et_io5_quantize_q8_0(max_blk, sizeof(max_blk), q8_max, sizeof(q8_max),
                             32u) == ET_IO5_QUANT_OK,
        "Q8_0 exact max scale 65504 accepted");
  CHECK(q8_max[0] == 0xffu && q8_max[1] == 0x7bu, "Q8_0 max scale is 0x7bff");
  CHECK((int8_t)q8_max[2] == 127 && (int8_t)q8_max[3] == -127,
        "Q8_0 max scale endpoints are +-127");

  /* Above max scale -> OVERFLOW, dst untouched. */
  max_blk[0] = 8319010.0f;
  memset(q8_max, 0x5Au, sizeof(q8_max));
  CHECK(et_io5_quantize_q8_0(max_blk, sizeof(max_blk), q8_max, sizeof(q8_max),
                             32u) == ET_IO5_QUANT_ERR_OVERFLOW,
        "Q8_0 above max scale rejects with OVERFLOW");
  for (size_t i = 0u; i < sizeof(q8_max); ++i) {
    CHECK(q8_max[i] == 0x5Au, "Q8_0 overflow must not mutate dst");
  }
}

static void test_negatives_and_aliasing(void) {
  float f32[64];
  uint8_t q8[68];
  uint8_t narrow[128];
  memset(f32, 0, sizeof(f32));
  memset(q8, 0, sizeof(q8));
  memset(narrow, 0, sizeof(narrow));

  /* NULL buffer rejections. */
  CHECK(et_io5_q8_0_byte_size(32u, NULL) == ET_IO5_QUANT_ERR_INVALID_ARG,
        "q8_0_byte_size NULL");
  CHECK(et_io5_f32_bits_to_f16_bits(0u, NULL) == ET_IO5_QUANT_ERR_INVALID_ARG,
        "f32->f16 NULL");
  CHECK(et_io5_f16_bits_to_f32_bits(0u, NULL) == ET_IO5_QUANT_ERR_INVALID_ARG,
        "f16->f32 NULL");
  CHECK(et_io5_f32_bits_to_bf16_bits(0u, NULL) == ET_IO5_QUANT_ERR_INVALID_ARG,
        "f32->bf16 NULL");
  CHECK(et_io5_bf16_bits_to_f32_bits(0u, NULL) == ET_IO5_QUANT_ERR_INVALID_ARG,
        "bf16->f32 NULL");
  CHECK(et_io5_validate_f32_finite(NULL, 4u, 1u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "validate_f32 NULL");
  CHECK(et_io5_encode_f16(NULL, 4u, narrow, 2u, 1u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "encode_f16 NULL src");
  CHECK(et_io5_encode_f16(f32, 4u, NULL, 2u, 1u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "encode_f16 NULL dst");
  CHECK(et_io5_quantize_q8_0(NULL, 128u, q8, 34u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "quantize_q8_0 NULL src");
  CHECK(et_io5_quantize_q8_0(f32, 128u, NULL, 34u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "quantize_q8_0 NULL dst");
  CHECK(et_io5_dequantize_q8_0(NULL, 34u, f32, 128u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "dequantize_q8_0 NULL src");
  CHECK(et_io5_dequantize_q8_0(q8, 34u, NULL, 128u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_ARG,
        "dequantize_q8_0 NULL dst");

  /* Non-multiple-of-32 and zero length rejections for Q8_0. */
  const size_t bad_counts[] = {0u, 1u, 15u, 31u, 33u, 63u, 65u};
  for (size_t i = 0u; i < sizeof(bad_counts) / sizeof(bad_counts[0]); ++i) {
    size_t bytes = 0u;
    CHECK(et_io5_q8_0_byte_size(bad_counts[i], &bytes) ==
              ET_IO5_QUANT_ERR_INVALID_LENGTH,
          "q8_0_byte_size rejects non-multiple-of-32");
    CHECK(et_io5_quantize_q8_0(f32, bad_counts[i] * 4u, q8, sizeof(q8),
                               bad_counts[i]) ==
              ET_IO5_QUANT_ERR_INVALID_LENGTH,
          "quantize_q8_0 rejects non-multiple-of-32");
    CHECK(et_io5_dequantize_q8_0(q8, sizeof(q8), f32, bad_counts[i] * 4u,
                                 bad_counts[i]) ==
              ET_IO5_QUANT_ERR_INVALID_LENGTH,
          "dequantize_q8_0 rejects non-multiple-of-32");
  }

  /* Mismatched byte sizes (truncated / oversize). */
  CHECK(et_io5_quantize_q8_0(f32, 124u, q8, 34u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_LENGTH,
        "quantize_q8_0 truncated src");
  CHECK(et_io5_quantize_q8_0(f32, 128u, q8, 33u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_LENGTH,
        "quantize_q8_0 truncated dst");
  CHECK(et_io5_dequantize_q8_0(q8, 33u, f32, 128u, 32u) ==
            ET_IO5_QUANT_ERR_INVALID_LENGTH,
        "dequantize_q8_0 truncated block");
  CHECK(et_io5_encode_f16(f32, 12u, narrow, 8u, 4u) ==
            ET_IO5_QUANT_ERR_INVALID_LENGTH,
        "encode_f16 wrong src bytes");
  CHECK(et_io5_decode_bf16(narrow, 8u, f32, 12u, 4u) ==
            ET_IO5_QUANT_ERR_INVALID_LENGTH,
        "decode_bf16 wrong dst bytes");

  /* Aliased and partially overlapping buffers must reject. */
  uint8_t shared[256];
  memset(shared, 0, sizeof(shared));
  CHECK(et_io5_encode_f16(shared, 64u, shared, 32u, 16u) ==
            ET_IO5_QUANT_ERR_ALIAS,
        "encode_f16 exact alias");
  CHECK(et_io5_encode_f16(shared, 64u, shared + 63u, 32u, 16u) ==
            ET_IO5_QUANT_ERR_ALIAS,
        "encode_f16 1-byte overlap");
  CHECK(et_io5_decode_f16(shared + 30u, 32u, shared, 64u, 16u) ==
            ET_IO5_QUANT_ERR_ALIAS,
        "decode_f16 partial overlap");
  CHECK(et_io5_quantize_q8_0(shared, 128u, shared + 127u, 34u, 32u) ==
            ET_IO5_QUANT_ERR_ALIAS,
        "quantize_q8_0 partial overlap");
  CHECK(et_io5_dequantize_q8_0(shared + 100u, 34u, shared, 128u, 32u) ==
            ET_IO5_QUANT_ERR_ALIAS,
        "dequantize_q8_0 partial overlap");
  /* Adjacent non-overlapping buffers must succeed! */
  CHECK(et_io5_quantize_q8_0(shared, 128u, shared + 128u, 34u, 32u) ==
            ET_IO5_QUANT_OK,
        "quantize_q8_0 adjacent non-overlapping succeeds");

  /* Non-finite f32 inputs to Q8_0 quantize & validate_f32_finite. */
  const uint32_t nonfin_f32[] = {UINT32_C(0x7f800000), UINT32_C(0xff800000),
                                 UINT32_C(0x7fc00000), UINT32_C(0x7f800001)};
  for (size_t i = 0u; i < sizeof(nonfin_f32) / sizeof(nonfin_f32[0]); ++i) {
    memset(f32, 0, sizeof(f32));
    f32[63] = bits_to_f32(nonfin_f32[i]);
    memset(q8, 0xCC, sizeof(q8));
    CHECK(et_io5_quantize_q8_0(f32, sizeof(f32), q8, sizeof(q8), 64u) ==
              ET_IO5_QUANT_ERR_NONFINITE,
          "Q8_0 quantize rejects non-finite f32 in second block");
    for (size_t k = 0u; k < sizeof(q8); ++k) {
      CHECK(q8[k] == 0xCCu, "failed Q8_0 quantize must not mutate dst");
    }
    CHECK(et_io5_validate_f32_finite(f32, sizeof(f32), 64u) ==
              ET_IO5_QUANT_ERR_NONFINITE,
          "validate_f32_finite rejects non-finite");
  }

  /* Non-finite f16 block scales in Q8_0 dequantize (+Inf, -Inf, qNaN, sNaN). */
  const uint16_t bad_scales[] = {UINT16_C(0x7c00), UINT16_C(0xfc00),
                                 UINT16_C(0x7e00), UINT16_C(0x7c01),
                                 UINT16_C(0xffff)};
  for (size_t i = 0u; i < sizeof(bad_scales) / sizeof(bad_scales[0]); ++i) {
    memset(q8, 0, sizeof(q8));
    /* Put valid scale in block 0, bad scale in block 1. */
    q8[0] = 0x00u;
    q8[1] = 0x3cu;
    q8[34] = (uint8_t)(bad_scales[i] & 0xffu);
    q8[35] = (uint8_t)((bad_scales[i] >> 8u) & 0xffu);
    uint8_t sentinel[64 * 4];
    memset(sentinel, 0x77, sizeof(sentinel));
    CHECK(et_io5_dequantize_q8_0(q8, sizeof(q8), sentinel, sizeof(sentinel),
                                 64u) == ET_IO5_QUANT_ERR_NONFINITE,
          "Q8_0 dequantize rejects non-finite f16 scale");
    for (size_t k = 0u; k < sizeof(sentinel); ++k) {
      CHECK(sentinel[k] == 0x77u, "failed Q8_0 dequantize must not mutate dst");
    }
  }
}

int main(void) {
#ifdef ET_IO5_QUANT_WRAP_ALLOC
  g_forbid_alloc = 1;
#endif
  CHECK(et_io5_quant_abi_version() == ((1u << 16u) | 0u), "ABI version");
  test_exhaustive_f16();
  test_exhaustive_bf16();
  test_bulk_narrow_and_unaligned();
  test_q8_0_roundtrip_and_edges();
  test_negatives_and_aliasing();
#ifdef ET_IO5_QUANT_WRAP_ALLOC
  g_forbid_alloc = 0;
#endif
  printf("IO5 quant native C11 codec tests: PASS\n");
  return 0;
}
