#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/w4a32_quant_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "W4A32 engine requires 8-bit bytes and IEEE binary32/binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "W4A32 engine requires exact IEEE-754 binary32/binary64");

#define F32_ABS_MASK UINT32_C(0x7fffffff)
#define F32_INF_BITS UINT32_C(0x7f800000)
/* Exact round-to-nearest-even binary16 overflow boundary:
 * 65520.0f = 0x477ff000 ties halfway between 65504 (0x7bff, odd mantissa)
 * and 65536 (0x7c00 = Inf, even), so >= 65520.0f overflows binary16. */
#define F16_OVERFLOW_F32_BITS UINT32_C(0x477ff000)
#define F16_EXP_MASK UINT16_C(0x7c00)

/* ---- Standalone SHA-256 (streaming, zero heap allocations) ---- */

typedef struct sha256_state {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} sha256_state;

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_block(sha256_state *state, const uint8_t block[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  size_t i;
  for (i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = state->h[0];
  uint32_t b = state->h[1];
  uint32_t c = state->h[2];
  uint32_t d = state->h[3];
  uint32_t e = state->h[4];
  uint32_t f = state->h[5];
  uint32_t g = state->h[6];
  uint32_t h = state->h[7];
  for (i = 0; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state->h[0] += a;
  state->h[1] += b;
  state->h[2] += c;
  state->h[3] += d;
  state->h[4] += e;
  state->h[5] += f;
  state->h[6] += g;
  state->h[7] += h;
}

static void sha256_init(sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(state->h, initial, sizeof(initial));
  state->length = 0u;
  state->used = 0u;
}

static void sha256_update(sha256_state *state, const uint8_t *bytes,
                          size_t length) {
  state->length += (uint64_t)length;
  while (length != 0u) {
    size_t take = 64u - state->used;
    if (take > length)
      take = length;
    memcpy(state->block + state->used, bytes, take);
    state->used += take;
    bytes += take;
    length -= take;
    if (state->used == 64u) {
      sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void sha256_final(sha256_state *state, uint8_t digest[32]) {
  const uint64_t bits = state->length * UINT64_C(8);
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (size_t i = 0; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (size_t i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

/* ---- Byte-wise little-endian loads & stores (zero misaligned-pointer UB) ---- */

static uint16_t load_u16_le(const uint8_t in[2]) {
  return (uint16_t)((uint16_t)in[0] | ((uint16_t)in[1] << 8u));
}

static void store_u16_le(uint8_t out[2], uint16_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
}

static uint32_t load_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) |
         ((uint32_t)in[2] << 16u) | ((uint32_t)in[3] << 24u);
}

static void store_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static uint64_t load_u64_le(const uint8_t in[8]) {
  return (uint64_t)in[0] | ((uint64_t)in[1] << 8u) |
         ((uint64_t)in[2] << 16u) | ((uint64_t)in[3] << 24u) |
         ((uint64_t)in[4] << 32u) | ((uint64_t)in[5] << 40u) |
         ((uint64_t)in[6] << 48u) | ((uint64_t)in[7] << 56u);
}

static void store_u64_le(uint8_t out[8], uint64_t value) {
  for (size_t i = 0; i < 8u; ++i)
    out[i] = (uint8_t)((value >> (8u * i)) & 0xffu);
}

static uint32_t f32_to_bits(float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static float bits_to_f32(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

/* ---- Exact IEEE-754 binary32 <-> binary16 (round-to-nearest-even) ---- */

static uint32_t round_shift_even(uint32_t value, unsigned int shift) {
  const uint32_t retained = value >> shift;
  const uint32_t half = UINT32_C(1) << (shift - 1u);
  const uint32_t discarded = value & ((half << 1u) - 1u);
  return retained +
         (uint32_t)(discarded > half || (discarded == half && (retained & 1u)));
}

static int f16_is_finite(uint16_t bits) {
  return (bits & F16_EXP_MASK) != F16_EXP_MASK;
}

static et_w4a32_status narrow_f32_to_f16_checked(float value,
                                                 uint16_t *out_bits) {
  const uint32_t bits = f32_to_bits(value);
  const uint32_t abs_bits = bits & F32_ABS_MASK;
  if (abs_bits >= F32_INF_BITS)
    return ET_W4A32_ERR_NONFINITE;
  if (abs_bits >= F16_OVERFLOW_F32_BITS)
    return ET_W4A32_ERR_OVERFLOW;

  const uint32_t sign = (bits >> 16u) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23u) & 0xffu;
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  uint16_t encoded = 0u;
  if (exponent < 102u) {
    encoded = (uint16_t)sign;
  } else if (exponent < 113u) {
    encoded = (uint16_t)(sign | round_shift_even(UINT32_C(0x800000) | fraction,
                                                 126u - exponent));
  } else {
    encoded = (uint16_t)(sign | (((exponent - 112u) << 10u) +
                                 round_shift_even(fraction, 13u)));
  }
  *out_bits = encoded;
  return ET_W4A32_OK;
}

static float widen_f16_to_f32_unchecked(uint16_t bits) {
  const uint32_t sign = ((uint32_t)bits & 0x8000u) << 16u;
  uint32_t fraction = (uint32_t)bits & 0x03ffu;
  uint32_t exponent = ((uint32_t)bits >> 10u) & 0x1fu;
  if (exponent == 0u) {
    if (fraction == 0u)
      return bits_to_f32(sign);
    exponent = 113u;
    while ((fraction & 0x0400u) == 0u) {
      fraction <<= 1u;
      --exponent;
    }
    fraction &= 0x03ffu;
  } else {
    exponent += 112u;
  }
  return bits_to_f32(sign | (exponent << 23u) | (fraction << 13u));
}

/* ---- FP environment, alignment, span, and shape validation helpers ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, mask all exceptions, and disable FTZ/DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int valid_asym_group_size(uint32_t group_size) {
  return group_size == ET_W4A32_GROUP_SIZE_32 ||
         group_size == ET_W4A32_GROUP_SIZE_64 ||
         group_size == ET_W4A32_GROUP_SIZE_128;
}

static et_w4a32_status validate_format_and_shape(et_w4a32_format format,
                                                 uint32_t group_size,
                                                 uint32_t rows_n,
                                                 uint32_t cols_k,
                                                 size_t *out_block_bytes,
                                                 size_t *out_packed_bytes) {
  size_t block_bytes = 0u;
  if (format == ET_W4A32_FORMAT_Q4_0) {
    if (group_size != ET_W4A32_Q4_0_GROUP_SIZE)
      return ET_W4A32_ERR_SHAPE;
    block_bytes = ET_W4A32_Q4_0_BLOCK_BYTES;
  } else if (format == ET_W4A32_FORMAT_Q4_ASYM) {
    if (!valid_asym_group_size(group_size))
      return ET_W4A32_ERR_SHAPE;
    block_bytes = ET_W4A32_ASYM_BLOCK_BYTES(group_size);
  } else {
    return ET_W4A32_ERR_FORMAT;
  }

  if (rows_n < 1u || rows_n > ET_W4A32_MAX_N || cols_k < group_size ||
      cols_k > ET_W4A32_MAX_K || (cols_k % group_size) != 0u)
    return ET_W4A32_ERR_SHAPE;

  const size_t groups_per_row = (size_t)(cols_k / group_size);
  const size_t packed_bytes =
      (size_t)rows_n * groups_per_row * block_bytes;
  if (out_block_bytes != NULL)
    *out_block_bytes = block_bytes;
  if (out_packed_bytes != NULL)
    *out_packed_bytes = packed_bytes;
  return ET_W4A32_OK;
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

static et_w4a32_status check_f32_buffer(const void *ptr, size_t count) {
  if (ptr == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(ptr, _Alignof(float), bytes))
    return ET_W4A32_ERR_ALIGNMENT;
  return ET_W4A32_OK;
}

static et_w4a32_status check_byte_buffer(const void *ptr, size_t bytes) {
  if (ptr == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  if (!aligned_span(ptr, 1u, bytes))
    return ET_W4A32_ERR_ALIGNMENT;
  return ET_W4A32_OK;
}

typedef struct span_entry {
  const void *ptr;
  size_t bytes;
} span_entry;

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

static int any_spans_overlap(const span_entry *entries, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(entries[i].ptr, entries[i].bytes, entries[j].ptr,
                        entries[j].bytes))
        return 1;
    }
  }
  return 0;
}

static uint8_t round_clamp_nibble(double scaled) {
  if (scaled <= 0.0)
    return 0u;
  if (scaled >= 15.0)
    return 15u;
  const double rounded = floor(scaled + 0.5);
  if (rounded <= 0.0)
    return 0u;
  if (rounded >= 15.0)
    return 15u;
  return (uint8_t)rounded;
}

static et_w4a32_status validate_packed_blocks(et_w4a32_format format,
                                              uint32_t group_size,
                                              uint32_t rows_n, uint32_t cols_k,
                                              const uint8_t *packed,
                                              double *out_max_w_abs) {
  const size_t groups_per_row = (size_t)(cols_k / group_size);
  const size_t block_bytes = (format == ET_W4A32_FORMAT_Q4_0)
                                 ? ET_W4A32_Q4_0_BLOCK_BYTES
                                 : ET_W4A32_ASYM_BLOCK_BYTES(group_size);
  const size_t total_blocks = (size_t)rows_n * groups_per_row;
  double max_w = 0.0;

  if (format == ET_W4A32_FORMAT_Q4_0) {
    for (size_t b = 0; b < total_blocks; ++b) {
      const uint16_t d_bits = load_u16_le(packed + b * block_bytes);
      if (!f16_is_finite(d_bits))
        return ET_W4A32_ERR_NONFINITE;
      if (out_max_w_abs != NULL) {
        const double bound = fabs((double)widen_f16_to_f32_unchecked(d_bits)) * 8.0;
        if (bound > max_w)
          max_w = bound;
      }
    }
  } else {
    for (size_t b = 0; b < total_blocks; ++b) {
      const uint8_t *blk = packed + b * block_bytes;
      const uint16_t d_bits = load_u16_le(blk);
      const uint16_t m_bits = load_u16_le(blk + 2u);
      if (!f16_is_finite(d_bits) || !f16_is_finite(m_bits))
        return ET_W4A32_ERR_NONFINITE;
      if (out_max_w_abs != NULL) {
        const double d_abs = fabs((double)widen_f16_to_f32_unchecked(d_bits));
        const double m_abs = fabs((double)widen_f16_to_f32_unchecked(m_bits));
        const double bound = d_abs * 15.0 + m_abs;
        if (bound > max_w)
          max_w = bound;
      }
    }
  }

  if (out_max_w_abs != NULL)
    *out_max_w_abs = max_w;
  return ET_W4A32_OK;
}

/* ---- Public Helpers: Sizes, F16 Conversion, SHA-256 ---- */

et_w4a32_status et_w4a32_packed_bytes(et_w4a32_format format,
                                      uint32_t group_size, uint32_t rows_n,
                                      uint32_t cols_k, size_t *out_bytes) {
  if (out_bytes == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  size_t packed = 0u;
  const et_w4a32_status status = validate_format_and_shape(
      format, group_size, rows_n, cols_k, NULL, &packed);
  if (status != ET_W4A32_OK)
    return status;
  *out_bytes = packed;
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_container_size(et_w4a32_format format,
                                        uint32_t group_size, uint32_t rows_n,
                                        uint32_t cols_k, size_t *out_bytes) {
  if (out_bytes == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  size_t packed = 0u;
  const et_w4a32_status status = validate_format_and_shape(
      format, group_size, rows_n, cols_k, NULL, &packed);
  if (status != ET_W4A32_OK)
    return status;
  *out_bytes = ET_W4A32_CONTAINER_MIN_BYTES + packed;
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_f32_to_f16(float value, uint16_t *out_bits) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (out_bits == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  uint16_t encoded = 0u;
  const et_w4a32_status status = narrow_f32_to_f16_checked(value, &encoded);
  if (status != ET_W4A32_OK)
    return status;
  *out_bits = encoded;
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_f16_to_f32(uint16_t bits, float *out_value) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (out_value == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  if (!f16_is_finite(bits))
    return ET_W4A32_ERR_NONFINITE;
  *out_value = widen_f16_to_f32_unchecked(bits);
  return ET_W4A32_OK;
}

et_w4a32_status
et_w4a32_sha256_bytes(const uint8_t *data, size_t size,
                      uint8_t out_digest[ET_W4A32_SHA256_BYTES]) {
  if (out_digest == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  if (size != 0u && data == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  if (size != 0u && !aligned_span(data, 1u, size))
    return ET_W4A32_ERR_ALIGNMENT;
  if (!aligned_span(out_digest, 1u, ET_W4A32_SHA256_BYTES))
    return ET_W4A32_ERR_ALIGNMENT;
  if (size != 0u &&
      spans_overlap(data, size, out_digest, ET_W4A32_SHA256_BYTES))
    return ET_W4A32_ERR_ALIAS;

  sha256_state state;
  sha256_init(&state);
  if (size != 0u)
    sha256_update(&state, data, size);
  uint8_t digest[ET_W4A32_SHA256_BYTES];
  sha256_final(&state, digest);
  memcpy(out_digest, digest, sizeof(digest));
  return ET_W4A32_OK;
}

/* ---- GGML Q4_0 Quantize, Dequantize, and Fused GEMM ---- */

static et_w4a32_status compute_q4_0_block_scale(const float *w_blk,
                                                uint16_t *out_d_bits) {
  float amax = 0.0f;
  float signed_max = 0.0f;
  for (uint32_t j = 0; j < ET_W4A32_Q4_0_GROUP_SIZE; ++j) {
    const float v = w_blk[j];
    if (!isfinite(v))
      return ET_W4A32_ERR_NONFINITE;
    const float av = fabsf(v);
    if (av > amax) {
      amax = av;
      signed_max = v;
    }
  }
  if (amax == 0.0f) {
    *out_d_bits = 0u;
    return ET_W4A32_OK;
  }
  const double d_exact = (double)signed_max / -8.0;
  if (!isfinite(d_exact) || fabs(d_exact) >= 65520.0)
    return ET_W4A32_ERR_OVERFLOW;
  return narrow_f32_to_f16_checked((float)d_exact, out_d_bits);
}

et_w4a32_status et_w4a32_q4_0_quantize_v1(uint32_t rows_n, uint32_t cols_k,
                                          const float *weights_f32,
                                          uint8_t *out_packed,
                                          size_t out_packed_bytes) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_0, ET_W4A32_Q4_0_GROUP_SIZE, rows_n, cols_k, NULL,
      &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (out_packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t weight_count = (size_t)rows_n * (size_t)cols_k;
  if ((status = check_f32_buffer(weights_f32, weight_count)) != ET_W4A32_OK ||
      (status = check_byte_buffer(out_packed, expected_packed)) != ET_W4A32_OK)
    return status;
  if (spans_overlap(weights_f32, weight_count * sizeof(float), out_packed,
                    expected_packed))
    return ET_W4A32_ERR_ALIAS;

  const uint32_t groups_per_row = cols_k / ET_W4A32_Q4_0_GROUP_SIZE;

  /* Pass 1: validate every input weight and f16 scale before writing anything. */
  for (uint32_t n = 0; n < rows_n; ++n) {
    const float *w_row = weights_f32 + (size_t)n * (size_t)cols_k;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      uint16_t d_bits = 0u;
      status = compute_q4_0_block_scale(w_row + (size_t)g * 32u, &d_bits);
      if (status != ET_W4A32_OK)
        return status;
    }
  }

  /* Pass 2: emit exact GGML Q4_0 blocks directly (row-aligned, zero alloc). */
  const size_t row_bytes = (size_t)groups_per_row * ET_W4A32_Q4_0_BLOCK_BYTES;
  for (uint32_t n = 0; n < rows_n; ++n) {
    const float *w_row = weights_f32 + (size_t)n * (size_t)cols_k;
    uint8_t *out_row = out_packed + (size_t)n * row_bytes;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      const float *w_blk = w_row + (size_t)g * 32u;
      uint8_t *blk = out_row + (size_t)g * ET_W4A32_Q4_0_BLOCK_BYTES;
      uint16_t d_bits = 0u;
      (void)compute_q4_0_block_scale(w_blk, &d_bits);
      store_u16_le(blk, d_bits);

      const float d_q = widen_f16_to_f32_unchecked(d_bits);
      const double inv_d = (d_q != 0.0f) ? (1.0 / (double)d_q) : 0.0;
      uint8_t *qs = blk + 2u;
      for (uint32_t j = 0; j < 16u; ++j) {
        const uint8_t q0 = round_clamp_nibble((double)w_blk[j] * inv_d + 8.0);
        const uint8_t q1 =
            round_clamp_nibble((double)w_blk[j + 16u] * inv_d + 8.0);
        qs[j] = (uint8_t)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
      }
    }
  }
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_q4_0_dequantize_v1(uint32_t rows_n, uint32_t cols_k,
                                            const uint8_t *packed,
                                            size_t packed_bytes,
                                            float *out_weights_f32) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_0, ET_W4A32_Q4_0_GROUP_SIZE, rows_n, cols_k, NULL,
      &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t weight_count = (size_t)rows_n * (size_t)cols_k;
  if ((status = check_byte_buffer(packed, expected_packed)) != ET_W4A32_OK ||
      (status = check_f32_buffer(out_weights_f32, weight_count)) != ET_W4A32_OK)
    return status;
  if (spans_overlap(packed, expected_packed, out_weights_f32,
                    weight_count * sizeof(float)))
    return ET_W4A32_ERR_ALIAS;

  /* Pass 1: validate every block's f16 scale before writing any output. */
  status = validate_packed_blocks(ET_W4A32_FORMAT_Q4_0,
                                  ET_W4A32_Q4_0_GROUP_SIZE, rows_n, cols_k,
                                  packed, NULL);
  if (status != ET_W4A32_OK)
    return status;

  /* Pass 2: dequantize directly into out_weights_f32. */
  const uint32_t groups_per_row = cols_k / ET_W4A32_Q4_0_GROUP_SIZE;
  const size_t row_bytes = (size_t)groups_per_row * ET_W4A32_Q4_0_BLOCK_BYTES;
  for (uint32_t n = 0; n < rows_n; ++n) {
    const uint8_t *in_row = packed + (size_t)n * row_bytes;
    float *out_row = out_weights_f32 + (size_t)n * (size_t)cols_k;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      const uint8_t *blk = in_row + (size_t)g * ET_W4A32_Q4_0_BLOCK_BYTES;
      const float d = widen_f16_to_f32_unchecked(load_u16_le(blk));
      const uint8_t *qs = blk + 2u;
      float *dst = out_row + (size_t)g * 32u;
      for (uint32_t j = 0; j < 16u; ++j) {
        const uint8_t byte = qs[j];
        const int32_t q0 = (int32_t)(byte & 0x0Fu) - 8;
        const int32_t q1 = (int32_t)((byte >> 4u) & 0x0Fu) - 8;
        dst[j] = d * (float)q0;
        dst[j + 16u] = d * (float)q1;
      }
    }
  }
  return ET_W4A32_OK;
}

static et_w4a32_status q4_0_gemm_pass(uint32_t m, uint32_t n, uint32_t k,
                                      const float *x, const uint8_t *w_q4_0,
                                      const float *bias, float *y,
                                      int dry_run) {
  const uint32_t groups_per_row = k / ET_W4A32_Q4_0_GROUP_SIZE;
  const size_t row_bytes = (size_t)groups_per_row * ET_W4A32_Q4_0_BLOCK_BYTES;

  for (uint32_t row = 0; row < m; ++row) {
    const float *x_row = x + (size_t)row * (size_t)k;
    float *y_row = (y != NULL) ? (y + (size_t)row * (size_t)n) : NULL;
    for (uint32_t col = 0; col < n; ++col) {
      const uint8_t *w_row = w_q4_0 + (size_t)col * row_bytes;
      double acc = (bias != NULL) ? (double)bias[col] : 0.0;
      for (uint32_t g = 0; g < groups_per_row; ++g) {
        const uint8_t *blk = w_row + (size_t)g * ET_W4A32_Q4_0_BLOCK_BYTES;
        const double d = (double)widen_f16_to_f32_unchecked(load_u16_le(blk));
        const uint8_t *qs = blk + 2u;
        const float *xg0 = x_row + (size_t)g * 32u;
        const float *xg1 = xg0 + 16u;
        double blk_sum = 0.0;
        for (uint32_t j = 0; j < 16u; ++j) {
          const uint8_t packed = qs[j];
          const int32_t q0 = (int32_t)(packed & 0x0Fu) - 8;
          const int32_t q1 = (int32_t)((packed >> 4u) & 0x0Fu) - 8;
          blk_sum += (double)xg0[j] * (double)q0 + (double)xg1[j] * (double)q1;
        }
        acc += d * blk_sum;
      }
      if (!isfinite(acc))
        return ET_W4A32_ERR_OVERFLOW;
      const float out_val = (float)acc;
      if (!isfinite(out_val))
        return ET_W4A32_ERR_OVERFLOW;
      if (!dry_run)
        y_row[col] = out_val;
    }
  }
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_q4_0_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                      const float *x, const uint8_t *w_q4_0,
                                      size_t w_q4_0_bytes, const float *bias,
                                      float *y) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (m < 1u || m > ET_W4A32_MAX_M)
    return ET_W4A32_ERR_SHAPE;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_0, ET_W4A32_Q4_0_GROUP_SIZE, n, k, NULL,
      &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (w_q4_0_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t x_count = (size_t)m * (size_t)k;
  const size_t y_count = (size_t)m * (size_t)n;
  if ((status = check_f32_buffer(x, x_count)) != ET_W4A32_OK ||
      (status = check_byte_buffer(w_q4_0, expected_packed)) != ET_W4A32_OK ||
      (status = check_f32_buffer(y, y_count)) != ET_W4A32_OK)
    return status;
  if (bias != NULL && (status = check_f32_buffer(bias, n)) != ET_W4A32_OK)
    return status;

  span_entry spans[4];
  size_t span_count = 0u;
  spans[span_count++] = (span_entry){x, x_count * sizeof(float)};
  spans[span_count++] = (span_entry){w_q4_0, expected_packed};
  spans[span_count++] = (span_entry){y, y_count * sizeof(float)};
  if (bias != NULL)
    spans[span_count++] = (span_entry){bias, (size_t)n * sizeof(float)};
  if (any_spans_overlap(spans, span_count))
    return ET_W4A32_ERR_ALIAS;

  /* Pass 1: validate finite X, finite bias, finite f16 scales, and track
   * upper bound to determine whether f32 output overflow is possible. */
  double max_x = 0.0;
  for (size_t i = 0; i < x_count; ++i) {
    if (!isfinite(x[i]))
      return ET_W4A32_ERR_NONFINITE;
    const double ax = fabs((double)x[i]);
    if (ax > max_x)
      max_x = ax;
  }
  double max_bias = 0.0;
  if (bias != NULL) {
    for (uint32_t col = 0; col < n; ++col) {
      if (!isfinite(bias[col]))
        return ET_W4A32_ERR_NONFINITE;
      const double ab = fabs((double)bias[col]);
      if (ab > max_bias)
        max_bias = ab;
    }
  }
  double max_w = 0.0;
  status = validate_packed_blocks(ET_W4A32_FORMAT_Q4_0,
                                  ET_W4A32_Q4_0_GROUP_SIZE, n, k, w_q4_0,
                                  &max_w);
  if (status != ET_W4A32_OK)
    return status;

  const double worst_case = (double)k * max_x * max_w + max_bias;
  if (!isfinite(worst_case) || worst_case > (double)FLT_MAX) {
    /* Extreme magnitudes: dry-run check guarantees Y is never corrupted. */
    status = q4_0_gemm_pass(m, n, k, x, w_q4_0, bias, NULL, 1);
    if (status != ET_W4A32_OK)
      return status;
  }

  return q4_0_gemm_pass(m, n, k, x, w_q4_0, bias, y, 0);
}

/* ---- Asymmetric Group-Wise Q4_ASYM Quantize, Dequantize, Fused GEMM ---- */

static et_w4a32_status compute_asym_block_params(const float *w_blk,
                                                 uint32_t group_size,
                                                 uint16_t *out_d_bits,
                                                 uint16_t *out_m_bits) {
  float w_min = w_blk[0];
  float w_max = w_blk[0];
  if (!isfinite(w_min))
    return ET_W4A32_ERR_NONFINITE;
  for (uint32_t j = 1u; j < group_size; ++j) {
    const float v = w_blk[j];
    if (!isfinite(v))
      return ET_W4A32_ERR_NONFINITE;
    if (v < w_min)
      w_min = v;
    if (v > w_max)
      w_max = v;
  }

  uint16_t m_bits = 0u;
  et_w4a32_status status = narrow_f32_to_f16_checked(w_min, &m_bits);
  if (status != ET_W4A32_OK)
    return status;
  const float m_q = widen_f16_to_f32_unchecked(m_bits);
  const double span = (double)w_max - (double)m_q;
  uint16_t d_bits = 0u;
  if (span > 0.0) {
    const double d_exact = span / 15.0;
    if (!isfinite(d_exact) || d_exact >= 65520.0)
      return ET_W4A32_ERR_OVERFLOW;
    status = narrow_f32_to_f16_checked((float)d_exact, &d_bits);
    if (status != ET_W4A32_OK)
      return status;
  }

  *out_d_bits = d_bits;
  *out_m_bits = m_bits;
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_asym_quantize_v1(uint32_t rows_n, uint32_t cols_k,
                                          uint32_t group_size,
                                          const float *weights_f32,
                                          uint8_t *out_packed,
                                          size_t out_packed_bytes) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t block_bytes = 0u;
  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_ASYM, group_size, rows_n, cols_k, &block_bytes,
      &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (out_packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t weight_count = (size_t)rows_n * (size_t)cols_k;
  if ((status = check_f32_buffer(weights_f32, weight_count)) != ET_W4A32_OK ||
      (status = check_byte_buffer(out_packed, expected_packed)) != ET_W4A32_OK)
    return status;
  if (spans_overlap(weights_f32, weight_count * sizeof(float), out_packed,
                    expected_packed))
    return ET_W4A32_ERR_ALIAS;

  const uint32_t groups_per_row = cols_k / group_size;

  /* Pass 1: validate all input elements and f16 scale/min bounds. */
  for (uint32_t n = 0; n < rows_n; ++n) {
    const float *w_row = weights_f32 + (size_t)n * (size_t)cols_k;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      uint16_t d_bits = 0u;
      uint16_t m_bits = 0u;
      status = compute_asym_block_params(w_row + (size_t)g * (size_t)group_size,
                                         group_size, &d_bits, &m_bits);
      if (status != ET_W4A32_OK)
        return status;
    }
  }

  /* Pass 2: emit packed asymmetric blocks directly. */
  const uint32_t half = group_size / 2u;
  const size_t row_bytes = (size_t)groups_per_row * block_bytes;
  for (uint32_t n = 0; n < rows_n; ++n) {
    const float *w_row = weights_f32 + (size_t)n * (size_t)cols_k;
    uint8_t *out_row = out_packed + (size_t)n * row_bytes;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      const float *w_blk = w_row + (size_t)g * (size_t)group_size;
      uint8_t *blk = out_row + (size_t)g * block_bytes;
      uint16_t d_bits = 0u;
      uint16_t m_bits = 0u;
      (void)compute_asym_block_params(w_blk, group_size, &d_bits, &m_bits);
      store_u16_le(blk, d_bits);
      store_u16_le(blk + 2u, m_bits);

      const double d_q = (double)widen_f16_to_f32_unchecked(d_bits);
      const double m_q = (double)widen_f16_to_f32_unchecked(m_bits);
      const double inv_d = (d_q != 0.0) ? (1.0 / d_q) : 0.0;
      uint8_t *qs = blk + ET_W4A32_ASYM_HEADER_BYTES;
      for (uint32_t j = 0; j < half; ++j) {
        const uint8_t q0 = round_clamp_nibble(((double)w_blk[j] - m_q) * inv_d);
        const uint8_t q1 =
            round_clamp_nibble(((double)w_blk[j + half] - m_q) * inv_d);
        qs[j] = (uint8_t)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
      }
    }
  }
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_asym_dequantize_v1(uint32_t rows_n, uint32_t cols_k,
                                            uint32_t group_size,
                                            const uint8_t *packed,
                                            size_t packed_bytes,
                                            float *out_weights_f32) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t block_bytes = 0u;
  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_ASYM, group_size, rows_n, cols_k, &block_bytes,
      &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t weight_count = (size_t)rows_n * (size_t)cols_k;
  if ((status = check_byte_buffer(packed, expected_packed)) != ET_W4A32_OK ||
      (status = check_f32_buffer(out_weights_f32, weight_count)) != ET_W4A32_OK)
    return status;
  if (spans_overlap(packed, expected_packed, out_weights_f32,
                    weight_count * sizeof(float)))
    return ET_W4A32_ERR_ALIAS;

  status = validate_packed_blocks(ET_W4A32_FORMAT_Q4_ASYM, group_size, rows_n,
                                  cols_k, packed, NULL);
  if (status != ET_W4A32_OK)
    return status;

  const uint32_t groups_per_row = cols_k / group_size;
  const uint32_t half = group_size / 2u;
  const size_t row_bytes = (size_t)groups_per_row * block_bytes;
  for (uint32_t n = 0; n < rows_n; ++n) {
    const uint8_t *in_row = packed + (size_t)n * row_bytes;
    float *out_row = out_weights_f32 + (size_t)n * (size_t)cols_k;
    for (uint32_t g = 0; g < groups_per_row; ++g) {
      const uint8_t *blk = in_row + (size_t)g * block_bytes;
      const double d = (double)widen_f16_to_f32_unchecked(load_u16_le(blk));
      const double m_off =
          (double)widen_f16_to_f32_unchecked(load_u16_le(blk + 2u));
      const uint8_t *qs = blk + ET_W4A32_ASYM_HEADER_BYTES;
      float *dst = out_row + (size_t)g * (size_t)group_size;
      for (uint32_t j = 0; j < half; ++j) {
        const uint8_t byte = qs[j];
        const uint32_t q0 = (uint32_t)(byte & 0x0Fu);
        const uint32_t q1 = (uint32_t)((byte >> 4u) & 0x0Fu);
        dst[j] = (float)(d * (double)q0 + m_off);
        dst[j + half] = (float)(d * (double)q1 + m_off);
      }
    }
  }
  return ET_W4A32_OK;
}

static et_w4a32_status asym_gemm_pass(uint32_t m, uint32_t n, uint32_t k,
                                      uint32_t group_size, const float *x,
                                      const uint8_t *w_asym, const float *bias,
                                      float *y, int dry_run) {
  const uint32_t groups_per_row = k / group_size;
  const uint32_t half = group_size / 2u;
  const size_t block_bytes = ET_W4A32_ASYM_BLOCK_BYTES(group_size);
  const size_t row_bytes = (size_t)groups_per_row * block_bytes;

  for (uint32_t row = 0; row < m; ++row) {
    const float *x_row = x + (size_t)row * (size_t)k;
    float *y_row = (y != NULL) ? (y + (size_t)row * (size_t)n) : NULL;
    for (uint32_t col = 0; col < n; ++col) {
      const uint8_t *w_row = w_asym + (size_t)col * row_bytes;
      double acc = (bias != NULL) ? (double)bias[col] : 0.0;
      for (uint32_t g = 0; g < groups_per_row; ++g) {
        const uint8_t *blk = w_row + (size_t)g * block_bytes;
        const double d = (double)widen_f16_to_f32_unchecked(load_u16_le(blk));
        const double m_off =
            (double)widen_f16_to_f32_unchecked(load_u16_le(blk + 2u));
        const uint8_t *qs = blk + ET_W4A32_ASYM_HEADER_BYTES;
        const float *xg0 = x_row + (size_t)g * (size_t)group_size;
        const float *xg1 = xg0 + half;
        double sum_xq = 0.0;
        double sum_x = 0.0;
        for (uint32_t j = 0; j < half; ++j) {
          const uint8_t packed = qs[j];
          const uint32_t q0 = (uint32_t)(packed & 0x0Fu);
          const uint32_t q1 = (uint32_t)((packed >> 4u) & 0x0Fu);
          const double x0 = (double)xg0[j];
          const double x1 = (double)xg1[j];
          sum_xq += x0 * (double)q0 + x1 * (double)q1;
          sum_x += x0 + x1;
        }
        acc += d * sum_xq + m_off * sum_x;
      }
      if (!isfinite(acc))
        return ET_W4A32_ERR_OVERFLOW;
      const float out_val = (float)acc;
      if (!isfinite(out_val))
        return ET_W4A32_ERR_OVERFLOW;
      if (!dry_run)
        y_row[col] = out_val;
    }
  }
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_asym_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                      uint32_t group_size, const float *x,
                                      const uint8_t *w_asym,
                                      size_t w_asym_bytes, const float *bias,
                                      float *y) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (m < 1u || m > ET_W4A32_MAX_M)
    return ET_W4A32_ERR_SHAPE;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      ET_W4A32_FORMAT_Q4_ASYM, group_size, n, k, NULL, &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (w_asym_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t x_count = (size_t)m * (size_t)k;
  const size_t y_count = (size_t)m * (size_t)n;
  if ((status = check_f32_buffer(x, x_count)) != ET_W4A32_OK ||
      (status = check_byte_buffer(w_asym, expected_packed)) != ET_W4A32_OK ||
      (status = check_f32_buffer(y, y_count)) != ET_W4A32_OK)
    return status;
  if (bias != NULL && (status = check_f32_buffer(bias, n)) != ET_W4A32_OK)
    return status;

  span_entry spans[4];
  size_t span_count = 0u;
  spans[span_count++] = (span_entry){x, x_count * sizeof(float)};
  spans[span_count++] = (span_entry){w_asym, expected_packed};
  spans[span_count++] = (span_entry){y, y_count * sizeof(float)};
  if (bias != NULL)
    spans[span_count++] = (span_entry){bias, (size_t)n * sizeof(float)};
  if (any_spans_overlap(spans, span_count))
    return ET_W4A32_ERR_ALIAS;

  double max_x = 0.0;
  for (size_t i = 0; i < x_count; ++i) {
    if (!isfinite(x[i]))
      return ET_W4A32_ERR_NONFINITE;
    const double ax = fabs((double)x[i]);
    if (ax > max_x)
      max_x = ax;
  }
  double max_bias = 0.0;
  if (bias != NULL) {
    for (uint32_t col = 0; col < n; ++col) {
      if (!isfinite(bias[col]))
        return ET_W4A32_ERR_NONFINITE;
      const double ab = fabs((double)bias[col]);
      if (ab > max_bias)
        max_bias = ab;
    }
  }
  double max_w = 0.0;
  status = validate_packed_blocks(ET_W4A32_FORMAT_Q4_ASYM, group_size, n, k,
                                  w_asym, &max_w);
  if (status != ET_W4A32_OK)
    return status;

  const double worst_case = (double)k * max_x * max_w + max_bias;
  if (!isfinite(worst_case) || worst_case > (double)FLT_MAX) {
    status = asym_gemm_pass(m, n, k, group_size, x, w_asym, bias, NULL, 1);
    if (status != ET_W4A32_OK)
      return status;
  }

  return asym_gemm_pass(m, n, k, group_size, x, w_asym, bias, y, 0);
}

/* ---- Dense F32 Reference GEMM (zero alloc, two-pass validate-before-write) ---- */

static et_w4a32_status f32_gemm_pass(uint32_t m, uint32_t n, uint32_t k,
                                     const float *x, const float *w_f32,
                                     const float *bias, float *y,
                                     int dry_run) {
  for (uint32_t row = 0; row < m; ++row) {
    const float *x_row = x + (size_t)row * (size_t)k;
    float *y_row = (y != NULL) ? (y + (size_t)row * (size_t)n) : NULL;
    for (uint32_t col = 0; col < n; ++col) {
      const float *w_row = w_f32 + (size_t)col * (size_t)k;
      double acc = (bias != NULL) ? (double)bias[col] : 0.0;
      for (uint32_t idx = 0; idx < k; ++idx)
        acc += (double)x_row[idx] * (double)w_row[idx];
      if (!isfinite(acc))
        return ET_W4A32_ERR_OVERFLOW;
      const float out_val = (float)acc;
      if (!isfinite(out_val))
        return ET_W4A32_ERR_OVERFLOW;
      if (!dry_run)
        y_row[col] = out_val;
    }
  }
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_f32_gemm_v1(uint32_t m, uint32_t n, uint32_t k,
                                     const float *x, const float *w_f32,
                                     const float *bias, float *y) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (m < 1u || m > ET_W4A32_MAX_M || n < 1u || n > ET_W4A32_MAX_N || k < 1u ||
      k > ET_W4A32_MAX_K)
    return ET_W4A32_ERR_SHAPE;

  const size_t x_count = (size_t)m * (size_t)k;
  const size_t w_count = (size_t)n * (size_t)k;
  const size_t y_count = (size_t)m * (size_t)n;
  et_w4a32_status status;
  if ((status = check_f32_buffer(x, x_count)) != ET_W4A32_OK ||
      (status = check_f32_buffer(w_f32, w_count)) != ET_W4A32_OK ||
      (status = check_f32_buffer(y, y_count)) != ET_W4A32_OK)
    return status;
  if (bias != NULL && (status = check_f32_buffer(bias, n)) != ET_W4A32_OK)
    return status;

  span_entry spans[4];
  size_t span_count = 0u;
  spans[span_count++] = (span_entry){x, x_count * sizeof(float)};
  spans[span_count++] = (span_entry){w_f32, w_count * sizeof(float)};
  spans[span_count++] = (span_entry){y, y_count * sizeof(float)};
  if (bias != NULL)
    spans[span_count++] = (span_entry){bias, (size_t)n * sizeof(float)};
  if (any_spans_overlap(spans, span_count))
    return ET_W4A32_ERR_ALIAS;

  double max_x = 0.0;
  for (size_t i = 0; i < x_count; ++i) {
    if (!isfinite(x[i]))
      return ET_W4A32_ERR_NONFINITE;
    const double ax = fabs((double)x[i]);
    if (ax > max_x)
      max_x = ax;
  }
  double max_w = 0.0;
  for (size_t i = 0; i < w_count; ++i) {
    if (!isfinite(w_f32[i]))
      return ET_W4A32_ERR_NONFINITE;
    const double aw = fabs((double)w_f32[i]);
    if (aw > max_w)
      max_w = aw;
  }
  double max_bias = 0.0;
  if (bias != NULL) {
    for (uint32_t col = 0; col < n; ++col) {
      if (!isfinite(bias[col]))
        return ET_W4A32_ERR_NONFINITE;
      const double ab = fabs((double)bias[col]);
      if (ab > max_bias)
        max_bias = ab;
    }
  }

  const double worst_case = (double)k * max_x * max_w + max_bias;
  if (!isfinite(worst_case) || worst_case > (double)FLT_MAX) {
    status = f32_gemm_pass(m, n, k, x, w_f32, bias, NULL, 1);
    if (status != ET_W4A32_OK)
      return status;
  }

  return f32_gemm_pass(m, n, k, x, w_f32, bias, y, 0);
}

/* ---- Checksummed Container (ETQ41000 v1.0) Serialize & Parse ---- */

static void build_container_header(uint8_t hdr[ET_W4A32_CONTAINER_HEADER_BYTES],
                                   et_w4a32_format format, uint32_t group_size,
                                   uint32_t rows_n, uint32_t cols_k,
                                   uint64_t payload_bytes) {
  memcpy(hdr, ET_W4A32_CONTAINER_MAGIC, ET_W4A32_CONTAINER_MAGIC_BYTES);
  store_u32_le(hdr + 8u, ET_W4A32_CONTAINER_VERSION_MAJOR);
  store_u32_le(hdr + 12u, ET_W4A32_CONTAINER_VERSION_MINOR);
  store_u32_le(hdr + 16u, (uint32_t)format);
  store_u32_le(hdr + 20u, group_size);
  store_u32_le(hdr + 24u, rows_n);
  store_u32_le(hdr + 28u, cols_k);
  store_u64_le(hdr + 32u, payload_bytes);
  store_u64_le(hdr + 40u, UINT64_C(0));
}

et_w4a32_status et_w4a32_container_serialize(
    et_w4a32_format format, uint32_t group_size, uint32_t rows_n,
    uint32_t cols_k, const uint8_t *packed, size_t packed_bytes,
    uint8_t *out_bytes, size_t capacity, size_t *out_size) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      format, group_size, rows_n, cols_k, NULL, &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;

  const size_t total = ET_W4A32_CONTAINER_MIN_BYTES + expected_packed;
  if (out_bytes == NULL || out_size == NULL || capacity < total)
    return ET_W4A32_ERR_ARGUMENT;
  if ((status = check_byte_buffer(packed, expected_packed)) != ET_W4A32_OK ||
      (status = check_byte_buffer(out_bytes, total)) != ET_W4A32_OK)
    return status;

  const span_entry spans[] = {
      {packed, expected_packed}, {out_bytes, total}, {out_size, sizeof(*out_size)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])))
    return ET_W4A32_ERR_ALIAS;

  status = validate_packed_blocks(format, group_size, rows_n, cols_k, packed,
                                  NULL);
  if (status != ET_W4A32_OK)
    return status;

  uint8_t hdr[ET_W4A32_CONTAINER_HEADER_BYTES];
  build_container_header(hdr, format, group_size, rows_n, cols_k,
                         (uint64_t)expected_packed);

  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, hdr, sizeof(hdr));
  sha256_update(&sha, packed, expected_packed);
  uint8_t digest[ET_W4A32_SHA256_BYTES];
  sha256_final(&sha, digest);

  memcpy(out_bytes, hdr, sizeof(hdr));
  memcpy(out_bytes + ET_W4A32_CONTAINER_HEADER_BYTES, packed, expected_packed);
  memcpy(out_bytes + ET_W4A32_CONTAINER_HEADER_BYTES + expected_packed, digest,
         sizeof(digest));
  *out_size = total;
  return ET_W4A32_OK;
}

et_w4a32_status et_w4a32_container_parse(
    const uint8_t *bytes, size_t size, et_w4a32_format expected_format,
    uint32_t expected_group_size, uint32_t expected_rows_n,
    uint32_t expected_cols_k, et_w4a32_container_meta *out_meta,
    uint8_t *out_packed, size_t out_packed_capacity) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      expected_format, expected_group_size, expected_rows_n, expected_cols_k,
      NULL, &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (bytes == NULL || out_packed == NULL ||
      out_packed_capacity < expected_packed)
    return ET_W4A32_ERR_ARGUMENT;
  if (!aligned_span(bytes, 1u, size) ||
      !aligned_span(out_packed, 1u, expected_packed))
    return ET_W4A32_ERR_ALIGNMENT;
  if (spans_overlap(bytes, size, out_packed, expected_packed))
    return ET_W4A32_ERR_ALIAS;
  if (out_meta != NULL) {
    if (!aligned_span(out_meta, _Alignof(et_w4a32_container_meta),
                      sizeof(*out_meta)))
      return ET_W4A32_ERR_ALIGNMENT;
    if (spans_overlap(out_meta, sizeof(*out_meta), bytes, size) ||
        spans_overlap(out_meta, sizeof(*out_meta), out_packed, expected_packed))
      return ET_W4A32_ERR_ALIAS;
  }

  /* 1. Magic and minimum/maximum bounds. */
  if (size >= ET_W4A32_CONTAINER_MAGIC_BYTES &&
      memcmp(bytes, ET_W4A32_CONTAINER_MAGIC,
             ET_W4A32_CONTAINER_MAGIC_BYTES) != 0)
    return ET_W4A32_ERR_FORMAT;
  if (size < ET_W4A32_CONTAINER_MIN_BYTES)
    return ET_W4A32_ERR_TRUNCATED;
  if (size > ET_W4A32_CONTAINER_MAX_BYTES)
    return ET_W4A32_ERR_FORMAT;

  /* 2. Version check. */
  const uint32_t ver_major = load_u32_le(bytes + 8u);
  const uint32_t ver_minor = load_u32_le(bytes + 12u);
  if (ver_major != ET_W4A32_CONTAINER_VERSION_MAJOR ||
      ver_minor != ET_W4A32_CONTAINER_VERSION_MINOR)
    return ET_W4A32_ERR_VERSION;

  /* 3. Header fields & length check before checksum. */
  const uint32_t file_format = load_u32_le(bytes + 16u);
  const uint32_t file_group = load_u32_le(bytes + 20u);
  const uint32_t file_n = load_u32_le(bytes + 24u);
  const uint32_t file_k = load_u32_le(bytes + 28u);
  const uint64_t file_payload = load_u64_le(bytes + 32u);
  const uint64_t file_flags = load_u64_le(bytes + 40u);

  size_t file_expected_packed = 0u;
  const et_w4a32_status header_shape_status = validate_format_and_shape(
      file_format, file_group, file_n, file_k, NULL, &file_expected_packed);
  if (header_shape_status == ET_W4A32_OK &&
      file_payload == (uint64_t)file_expected_packed) {
    const size_t total_expected =
        ET_W4A32_CONTAINER_MIN_BYTES + file_expected_packed;
    if (size < total_expected)
      return ET_W4A32_ERR_TRUNCATED;
    if (size > total_expected)
      return ET_W4A32_ERR_FORMAT;
  }

  /* 4. SHA-256 verification over header + payload. */
  const size_t signed_bytes = size - ET_W4A32_CONTAINER_CHECKSUM_BYTES;
  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, bytes, signed_bytes);
  uint8_t computed[ET_W4A32_SHA256_BYTES];
  sha256_final(&sha, computed);
  if (memcmp(computed, bytes + signed_bytes, ET_W4A32_SHA256_BYTES) != 0)
    return ET_W4A32_ERR_CHECKSUM;

  /* 5. Structural validation of signed header. */
  if (file_flags != UINT64_C(0))
    return ET_W4A32_ERR_FORMAT;
  if (header_shape_status != ET_W4A32_OK)
    return header_shape_status;
  if (file_payload != (uint64_t)file_expected_packed ||
      size != ET_W4A32_CONTAINER_MIN_BYTES + file_expected_packed)
    return ET_W4A32_ERR_FORMAT;

  /* 6. Validate all packed block scales/mins are finite. */
  const uint8_t *payload_ptr = bytes + ET_W4A32_CONTAINER_HEADER_BYTES;
  status = validate_packed_blocks(file_format, file_group, file_n, file_k,
                                  payload_ptr, NULL);
  if (status != ET_W4A32_OK)
    return status;

  /* 7. Caller format & shape binding checks. */
  if (file_format != expected_format)
    return ET_W4A32_ERR_FORMAT;
  if (file_group != expected_group_size || file_n != expected_rows_n ||
      file_k != expected_cols_k)
    return ET_W4A32_ERR_SHAPE;

  /* 8. Commit validated payload and metadata atomically. */
  memcpy(out_packed, payload_ptr, expected_packed);
  if (out_meta != NULL) {
    out_meta->version_major = ver_major;
    out_meta->version_minor = ver_minor;
    out_meta->format = file_format;
    out_meta->group_size = file_group;
    out_meta->rows_n = file_n;
    out_meta->cols_k = file_k;
    out_meta->payload_bytes = file_payload;
    memcpy(out_meta->sha256, computed, sizeof(computed));
  }
  return ET_W4A32_OK;
}

/* ---- Atomic File Save & O_NOFOLLOW Checked File Load ---- */

static int write_all(int fd, const uint8_t *bytes, size_t size) {
  while (size != 0u) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return 0;
    bytes += (size_t)written;
    size -= (size_t)written;
  }
  return 1;
}

static int sync_fd(int fd) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc < 0 && errno == EINTR);
  return rc == 0;
}

et_w4a32_status et_w4a32_container_save(const char *path,
                                        et_w4a32_format format,
                                        uint32_t group_size, uint32_t rows_n,
                                        uint32_t cols_k, const uint8_t *packed,
                                        size_t packed_bytes) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0')
    return ET_W4A32_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_W4A32_ERR_ARGUMENT;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      format, group_size, rows_n, cols_k, NULL, &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (packed_bytes != expected_packed)
    return ET_W4A32_ERR_ARGUMENT;
  if ((status = check_byte_buffer(packed, expected_packed)) != ET_W4A32_OK)
    return status;

  status = validate_packed_blocks(format, group_size, rows_n, cols_k, packed,
                                  NULL);
  if (status != ET_W4A32_OK)
    return status;

  uint8_t hdr[ET_W4A32_CONTAINER_HEADER_BYTES];
  build_container_header(hdr, format, group_size, rows_n, cols_k,
                         (uint64_t)expected_packed);

  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, hdr, sizeof(hdr));
  sha256_update(&sha, packed, expected_packed);
  uint8_t digest[ET_W4A32_SHA256_BYTES];
  sha256_final(&sha, digest);

  static const char suffix[] = ".w4a32-ckpt.XXXXXX";
  const size_t prefix = (slash == NULL) ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix))
    return ET_W4A32_ERR_OVERFLOW;

  char *temp_path = (char *)malloc(prefix + sizeof(suffix));
  if (temp_path == NULL)
    return ET_W4A32_ERR_NOMEM;
  memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    return ET_W4A32_ERR_IO;
  }

  int ok = write_all(fd, hdr, sizeof(hdr)) &&
           write_all(fd, packed, expected_packed) &&
           write_all(fd, digest, sizeof(digest)) && sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    status = ET_W4A32_ERR_IO;
  }
  free(temp_path);
  return status;
}

et_w4a32_status et_w4a32_container_load(
    const char *path, et_w4a32_format expected_format,
    uint32_t expected_group_size, uint32_t expected_rows_n,
    uint32_t expected_cols_k, et_w4a32_container_meta *out_meta,
    uint8_t *out_packed, size_t out_packed_capacity) {
  if (!fp_environment_supported())
    return ET_W4A32_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0')
    return ET_W4A32_ERR_ARGUMENT;

  size_t expected_packed = 0u;
  et_w4a32_status status = validate_format_and_shape(
      expected_format, expected_group_size, expected_rows_n, expected_cols_k,
      NULL, &expected_packed);
  if (status != ET_W4A32_OK)
    return status;
  if (out_packed == NULL || out_packed_capacity < expected_packed)
    return ET_W4A32_ERR_ARGUMENT;
  if (!aligned_span(out_packed, 1u, expected_packed))
    return ET_W4A32_ERR_ALIGNMENT;
  if (out_meta != NULL) {
    if (!aligned_span(out_meta, _Alignof(et_w4a32_container_meta),
                      sizeof(*out_meta)))
      return ET_W4A32_ERR_ALIGNMENT;
    if (spans_overlap(out_meta, sizeof(*out_meta), out_packed, expected_packed))
      return ET_W4A32_ERR_ALIAS;
  }

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_W4A32_ERR_SYMLINK : ET_W4A32_ERR_IO;

  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    (void)close(fd);
    return ET_W4A32_ERR_IO;
  }
  if (st.st_size < 0 || (uintmax_t)st.st_size > ET_W4A32_CONTAINER_MAX_BYTES) {
    (void)close(fd);
    return ET_W4A32_ERR_FORMAT;
  }

  const size_t file_size = (size_t)st.st_size;
  const size_t alloc_size = file_size + 1u;
  uint8_t *buf = (uint8_t *)malloc(alloc_size);
  if (buf == NULL) {
    (void)close(fd);
    return ET_W4A32_ERR_NOMEM;
  }

  size_t used = 0u;
  status = ET_W4A32_OK;
  while (used < alloc_size) {
    const ssize_t n = read(fd, buf + used, alloc_size - used);
    if (n == 0)
      break;
    if (n < 0) {
      if (errno == EINTR)
        continue;
      status = ET_W4A32_ERR_IO;
      break;
    }
    used += (size_t)n;
  }
  if (close(fd) != 0 && status == ET_W4A32_OK)
    status = ET_W4A32_ERR_IO;

  if (status == ET_W4A32_OK) {
    if (used > ET_W4A32_CONTAINER_MAX_BYTES)
      status = ET_W4A32_ERR_FORMAT;
    else if (used != file_size)
      status = ET_W4A32_ERR_TRUNCATED;
    else
      status = et_w4a32_container_parse(
          buf, used, expected_format, expected_group_size, expected_rows_n,
          expected_cols_k, out_meta, out_packed, out_packed_capacity);
  }
  free(buf);
  return status;
}
