#include "eshkol_transformer/fp8_gemm_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "FP8 GEMM engine requires 8-bit bytes and IEEE-754 f32/f64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "FP8 GEMM engine requires IEEE-754 binary32 and binary64");

/* ---- Alignment-safe byte helpers and IEEE-754 bit predicates ---- */

static uint32_t f32_to_bits(float v) {
  uint32_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static float bits_to_f32(uint32_t bits) {
  float v = 0.0f;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static uint64_t f64_to_bits(double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static int f32_is_finite(float v) {
  return (f32_to_bits(v) & 0x7F800000u) != 0x7F800000u;
}

static int f64_is_finite(double v) {
  return (f64_to_bits(v) & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

static float f32_abs(float v) {
  return bits_to_f32(f32_to_bits(v) & 0x7FFFFFFFu);
}

static float load_f32_at(const void *base, size_t idx) {
  float v = 0.0f;
  const uint8_t *p = (const uint8_t *)base + idx * sizeof(float);
  memcpy(&v, p, sizeof(v));
  return v;
}

static void store_f32_at(void *base, size_t idx, float v) {
  uint8_t *p = (uint8_t *)base + idx * sizeof(float);
  memcpy(p, &v, sizeof(v));
}

static int mul_size_overflow(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a)
    return 1;
  *out = a * b;
  return 0;
}

static int ranges_overlap(const void *a, size_t a_bytes, const void *b,
                          size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t a0 = (uintptr_t)a;
  const uintptr_t b0 = (uintptr_t)b;
  if (a0 > UINTPTR_MAX - a_bytes || b0 > UINTPTR_MAX - b_bytes)
    return 1;
  const uintptr_t a1 = a0 + a_bytes;
  const uintptr_t b1 = b0 + b_bytes;
  return a0 < b1 && b0 < a1;
}

static int fp_env_supported(void) {
  if (fegetround() != FE_TONEAREST)
    return 0;
#if defined(__x86_64__)
  /* Reject Flush-To-Zero (bit 15 = 0x8000) or Denormals-Are-Zero (bit 6 = 0x40). */
  const unsigned int mxcsr = _mm_getcsr();
  if ((mxcsr & 0x8040u) != 0u)
    return 0;
#endif
  return 1;
}

static int valid_overflow_mode(et_fp8_overflow_mode mode) {
  return mode == ET_FP8_OVERFLOW_SATURATE_TO_MAX ||
         mode == ET_FP8_OVERFLOW_REJECT;
}

static int valid_block_k(size_t block_k) {
  return block_k == 16u || block_k == 32u;
}

static int valid_block_n(size_t block_n) {
  return block_n == 1u || block_n == 16u || block_n == 32u;
}

/* ---- Internal bit-exact OCP FP8 codec primitives (pure integer RNE) ---- */

static et_fp8_status e4m3_encode_bits(uint32_t f32_bits,
                                      et_fp8_overflow_mode mode,
                                      uint8_t *out_code) {
  const uint8_t sign = (uint8_t)((f32_bits >> 31u) << 7u);
  const uint32_t abs_u = f32_bits & 0x7FFFFFFFu;
  const int32_t f32_exp = (int32_t)((abs_u >> 23u) & 0xFFu);
  const uint32_t f32_mant = abs_u & 0x7FFFFFu;

  if (f32_exp == 0xFF)
    return ET_FP8_ERR_NONFINITE;

  if (abs_u == 0u) {
    *out_code = sign;
    return ET_FP8_OK;
  }

  /* Subnormal f32 has |x| < 2^-126 << 2^-10 (half of E4M3FN min subnormal). */
  if (f32_exp == 0) {
    *out_code = sign;
    return ET_FP8_OK;
  }

  const int32_t e = f32_exp - 127;
  const int32_t fp8_exp = e + 7;

  if (fp8_exp >= 1) {
    if (fp8_exp > 15) {
      if (mode == ET_FP8_OVERFLOW_REJECT)
        return ET_FP8_ERR_OVERFLOW;
      *out_code = (uint8_t)(sign | 0x7Eu);
      return ET_FP8_OK;
    }

    const uint32_t q = f32_mant >> 20u;
    const uint32_t rem = f32_mant & 0xFFFFFu;
    const uint32_t half = 0x80000u;
    const uint32_t round_up =
        (rem > half || (rem == half && (q & 1u) != 0u)) ? 1u : 0u;
    const uint32_t res = (((uint32_t)fp8_exp << 3u) | q) + round_up;

    /* E4M3FN has no Inf: 0x7F is NaN, so >= 0x7F overflows max finite 448.0. */
    if (res >= 0x7Fu) {
      if (mode == ET_FP8_OVERFLOW_REJECT)
        return ET_FP8_ERR_OVERFLOW;
      *out_code = (uint8_t)(sign | 0x7Eu);
      return ET_FP8_OK;
    }

    *out_code = (uint8_t)(sign | (uint8_t)res);
    return ET_FP8_OK;
  }

  /* Subnormal / underflow range: e <= -7. Half of min subnormal is 2^-10. */
  if (e < -10) {
    *out_code = sign;
    return ET_FP8_OK;
  }

  /* e in {-10, -9, -8, -7} => shift in {24, 23, 22, 21}. */
  const uint32_t shift = (uint32_t)(14 - e);
  const uint32_t sig = (1u << 23u) | f32_mant;
  const uint32_t q = sig >> shift;
  const uint32_t mask = (1u << shift) - 1u;
  const uint32_t rem = sig & mask;
  const uint32_t half = 1u << (shift - 1u);
  const uint32_t round_up =
      (rem > half || (rem == half && (q & 1u) != 0u)) ? 1u : 0u;
  const uint32_t res = q + round_up; /* 0..8; 8 == 0x08 (min normal 2^-6). */

  *out_code = (uint8_t)(sign | (uint8_t)res);
  return ET_FP8_OK;
}

static et_fp8_status e4m3_decode_bits(uint8_t code, uint32_t *out_f32_bits) {
  const uint32_t sign = (uint32_t)(code >> 7u);
  const uint32_t exp = (uint32_t)((code >> 3u) & 0xFu);
  const uint32_t mant = (uint32_t)(code & 0x7u);

  if (exp == 15u && mant == 7u)
    return ET_FP8_ERR_NONFINITE;

  if (exp == 0u) {
    if (mant == 0u) {
      *out_f32_bits = sign << 31u;
      return ET_FP8_OK;
    }
    uint32_t m = mant;
    uint32_t f32_exp = 121u; /* 127 - 6 */
    while ((m & 0x8u) == 0u) {
      m <<= 1u;
      f32_exp -= 1u;
    }
    const uint32_t f32_mant = (m & 0x7u) << 20u;
    *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
    return ET_FP8_OK;
  }

  const uint32_t f32_exp = exp + 120u; /* exp - 7 + 127 */
  const uint32_t f32_mant = mant << 20u;
  *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
  return ET_FP8_OK;
}

static float e4m3_decode_valid_fast(uint8_t code) {
  uint32_t bits = 0u;
  (void)e4m3_decode_bits(code, &bits);
  return bits_to_f32(bits);
}

static et_fp8_status e5m2_encode_bits(uint32_t f32_bits,
                                      et_fp8_overflow_mode mode,
                                      uint8_t *out_code) {
  const uint8_t sign = (uint8_t)((f32_bits >> 31u) << 7u);
  const uint32_t abs_u = f32_bits & 0x7FFFFFFFu;
  const int32_t f32_exp = (int32_t)((abs_u >> 23u) & 0xFFu);
  const uint32_t f32_mant = abs_u & 0x7FFFFFu;

  if (f32_exp == 0xFF)
    return ET_FP8_ERR_NONFINITE;

  if (abs_u == 0u || f32_exp == 0) {
    *out_code = sign;
    return ET_FP8_OK;
  }

  const int32_t e = f32_exp - 127;
  const int32_t fp8_exp = e + 15;

  if (fp8_exp >= 1) {
    if (fp8_exp > 30) {
      if (mode == ET_FP8_OVERFLOW_REJECT)
        return ET_FP8_ERR_OVERFLOW;
      *out_code = (uint8_t)(sign | 0x7Bu);
      return ET_FP8_OK;
    }

    const uint32_t q = f32_mant >> 21u;
    const uint32_t rem = f32_mant & 0x1FFFFFu;
    const uint32_t half = 0x100000u;
    const uint32_t round_up =
        (rem > half || (rem == half && (q & 1u) != 0u)) ? 1u : 0u;
    const uint32_t res = (((uint32_t)fp8_exp << 2u) | q) + round_up;

    /* 0x7C is Inf in E5M2; >= 0x7C overflows max finite 57344.0 (0x7B). */
    if (res >= 0x7Cu) {
      if (mode == ET_FP8_OVERFLOW_REJECT)
        return ET_FP8_ERR_OVERFLOW;
      *out_code = (uint8_t)(sign | 0x7Bu);
      return ET_FP8_OK;
    }

    *out_code = (uint8_t)(sign | (uint8_t)res);
    return ET_FP8_OK;
  }

  /* Subnormal / underflow range: e <= -15. Half of min subnormal is 2^-17. */
  if (e < -17) {
    *out_code = sign;
    return ET_FP8_OK;
  }

  /* e in {-17, -16, -15} => shift in {24, 23, 22}. */
  const uint32_t shift = (uint32_t)(7 - e);
  const uint32_t sig = (1u << 23u) | f32_mant;
  const uint32_t q = sig >> shift;
  const uint32_t mask = (1u << shift) - 1u;
  const uint32_t rem = sig & mask;
  const uint32_t half = 1u << (shift - 1u);
  const uint32_t round_up =
      (rem > half || (rem == half && (q & 1u) != 0u)) ? 1u : 0u;
  const uint32_t res = q + round_up; /* 0..4; 4 == 0x04 (min normal 2^-14). */

  *out_code = (uint8_t)(sign | (uint8_t)res);
  return ET_FP8_OK;
}

static et_fp8_status e5m2_decode_bits(uint8_t code, uint32_t *out_f32_bits) {
  const uint32_t sign = (uint32_t)(code >> 7u);
  const uint32_t exp = (uint32_t)((code >> 2u) & 0x1Fu);
  const uint32_t mant = (uint32_t)(code & 0x3u);

  if (exp == 31u)
    return ET_FP8_ERR_NONFINITE;

  if (exp == 0u) {
    if (mant == 0u) {
      *out_f32_bits = sign << 31u;
      return ET_FP8_OK;
    }
    uint32_t m = mant;
    uint32_t f32_exp = 113u; /* 127 - 14 */
    while ((m & 0x4u) == 0u) {
      m <<= 1u;
      f32_exp -= 1u;
    }
    const uint32_t f32_mant = (m & 0x3u) << 21u;
    *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
    return ET_FP8_OK;
  }

  const uint32_t f32_exp = exp + 112u; /* exp - 15 + 127 */
  const uint32_t f32_mant = mant << 21u;
  *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
  return ET_FP8_OK;
}

/* ---- Public Codec Functions ---- */

et_fp8_status et_fp8_e4m3_encode_scalar_v1(float value,
                                           et_fp8_overflow_mode mode,
                                           uint8_t *out_code) {
  if (out_code == NULL || !valid_overflow_mode(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  uint8_t code = 0u;
  const et_fp8_status st = e4m3_encode_bits(f32_to_bits(value), mode, &code);
  if (st != ET_FP8_OK)
    return st;
  *out_code = code;
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e4m3_decode_scalar_v1(uint8_t code, float *out_value) {
  if (out_value == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  uint32_t bits = 0u;
  const et_fp8_status st = e4m3_decode_bits(code, &bits);
  if (st != ET_FP8_OK)
    return st;
  store_f32_at(out_value, 0u, bits_to_f32(bits));
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e4m3_encode_v1(const float *src, size_t count,
                                    et_fp8_overflow_mode mode, uint8_t *dst) {
  size_t src_bytes = 0u;
  if (src == NULL || dst == NULL || count == 0u || !valid_overflow_mode(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (mul_size_overflow(count, sizeof(float), &src_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(src, src_bytes, dst, count))
    return ET_FP8_ERR_ALIAS;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;

  /* Pass 1: validate every input before writing a single byte of dst. */
  for (size_t i = 0u; i < count; ++i) {
    uint8_t tmp = 0u;
    const float v = load_f32_at(src, i);
    const et_fp8_status st = e4m3_encode_bits(f32_to_bits(v), mode, &tmp);
    if (st != ET_FP8_OK)
      return st;
  }

  /* Pass 2: commit encoded bytes. */
  for (size_t i = 0u; i < count; ++i) {
    uint8_t code = 0u;
    const float v = load_f32_at(src, i);
    (void)e4m3_encode_bits(f32_to_bits(v), mode, &code);
    dst[i] = code;
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e4m3_decode_v1(const uint8_t *src, size_t count,
                                    float *dst) {
  size_t dst_bytes = 0u;
  if (src == NULL || dst == NULL || count == 0u)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (mul_size_overflow(count, sizeof(float), &dst_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(src, count, dst, dst_bytes))
    return ET_FP8_ERR_ALIAS;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;

  /* Pass 1: validate every FP8 code before mutating dst. */
  for (size_t i = 0u; i < count; ++i) {
    if ((src[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }

  /* Pass 2: decode and write. */
  for (size_t i = 0u; i < count; ++i) {
    uint32_t bits = 0u;
    (void)e4m3_decode_bits(src[i], &bits);
    store_f32_at(dst, i, bits_to_f32(bits));
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e5m2_encode_scalar_v1(float value,
                                           et_fp8_overflow_mode mode,
                                           uint8_t *out_code) {
  if (out_code == NULL || !valid_overflow_mode(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  uint8_t code = 0u;
  const et_fp8_status st = e5m2_encode_bits(f32_to_bits(value), mode, &code);
  if (st != ET_FP8_OK)
    return st;
  *out_code = code;
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e5m2_decode_scalar_v1(uint8_t code, float *out_value) {
  if (out_value == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  uint32_t bits = 0u;
  const et_fp8_status st = e5m2_decode_bits(code, &bits);
  if (st != ET_FP8_OK)
    return st;
  store_f32_at(out_value, 0u, bits_to_f32(bits));
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e5m2_encode_v1(const float *src, size_t count,
                                    et_fp8_overflow_mode mode, uint8_t *dst) {
  size_t src_bytes = 0u;
  if (src == NULL || dst == NULL || count == 0u || !valid_overflow_mode(mode))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (mul_size_overflow(count, sizeof(float), &src_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(src, src_bytes, dst, count))
    return ET_FP8_ERR_ALIAS;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;

  for (size_t i = 0u; i < count; ++i) {
    uint8_t tmp = 0u;
    const float v = load_f32_at(src, i);
    const et_fp8_status st = e5m2_encode_bits(f32_to_bits(v), mode, &tmp);
    if (st != ET_FP8_OK)
      return st;
  }

  for (size_t i = 0u; i < count; ++i) {
    uint8_t code = 0u;
    const float v = load_f32_at(src, i);
    (void)e5m2_encode_bits(f32_to_bits(v), mode, &code);
    dst[i] = code;
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_e5m2_decode_v1(const uint8_t *src, size_t count,
                                    float *dst) {
  size_t dst_bytes = 0u;
  if (src == NULL || dst == NULL || count == 0u)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (mul_size_overflow(count, sizeof(float), &dst_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(src, count, dst, dst_bytes))
    return ET_FP8_ERR_ALIAS;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;

  for (size_t i = 0u; i < count; ++i) {
    if ((src[i] & 0x7Cu) == 0x7Cu)
      return ET_FP8_ERR_NONFINITE;
  }

  for (size_t i = 0u; i < count; ++i) {
    uint32_t bits = 0u;
    (void)e5m2_decode_bits(src[i], &bits);
    store_f32_at(dst, i, bits_to_f32(bits));
  }
  return ET_FP8_OK;
}

/* ---- Delayed-Scaling Amax History Manager ---- */

static et_fp8_status
validate_delayed_scale_state(const et_fp8_delayed_scale_state_v1 *state) {
  if (state == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (state->abi_version != ET_FP8_GEMM_ABI_MAJOR)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (state->format != ET_FP8_FORMAT_E4M3FN &&
      state->format != ET_FP8_FORMAT_E5M2)
    return ET_FP8_ERR_UNSUPPORTED_FORMAT;
  if (state->history_len == 0u ||
      state->history_len > ET_FP8_DELAYED_SCALE_MAX_HISTORY)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (state->cursor >= state->history_len ||
      state->count > state->history_len)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (state->margin > ET_FP8_DELAYED_SCALE_MAX_MARGIN)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(state->eps) || state->eps <= 0.0f)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(state->min_scale) || state->min_scale <= 0.0f)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(state->max_scale) ||
      state->max_scale < state->min_scale)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  for (uint32_t i = 0u; i < state->count; ++i) {
    const float v = state->history[i];
    if (!f32_is_finite(v) || v < 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
  }
  return ET_FP8_OK;
}

static void compute_delayed_scale_internal(
    const et_fp8_delayed_scale_state_v1 *state, float *out_scale,
    float *out_history_amax) {
  double hist_amax = 0.0;
  for (uint32_t i = 0u; i < state->count; ++i) {
    const double v = (double)state->history[i];
    if (v > hist_amax)
      hist_amax = v;
  }
  const double denom =
      hist_amax > (double)state->eps ? hist_amax : (double)state->eps;
  const double fp8_max = (state->format == ET_FP8_FORMAT_E4M3FN)
                             ? (double)ET_FP8_E4M3_MAX_FINITE
                             : (double)ET_FP8_E5M2_MAX_FINITE;
  /* 2^(-margin) is exact in IEEE-754 binary64 for margin in 0..16. */
  const double margin_factor = 1.0 / (double)(1u << state->margin);
  double raw = (fp8_max / denom) * margin_factor;

  if (!f64_is_finite(raw) || raw > (double)state->max_scale)
    raw = (double)state->max_scale;
  if (raw < (double)state->min_scale)
    raw = (double)state->min_scale;

  float scale_f32 = (float)raw;
  if (!f32_is_finite(scale_f32) || scale_f32 > state->max_scale)
    scale_f32 = state->max_scale;
  if (scale_f32 < state->min_scale)
    scale_f32 = state->min_scale;

  *out_scale = scale_f32;
  if (out_history_amax != NULL)
    *out_history_amax = (float)hist_amax;
}

et_fp8_status et_fp8_delayed_scale_init(et_fp8_delayed_scale_state_v1 *state,
                                        et_fp8_format format,
                                        uint32_t history_len, uint32_t margin,
                                        float eps, float min_scale,
                                        float max_scale) {
  if (state == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (format != ET_FP8_FORMAT_E4M3FN && format != ET_FP8_FORMAT_E5M2)
    return ET_FP8_ERR_UNSUPPORTED_FORMAT;
  if (history_len == 0u || history_len > ET_FP8_DELAYED_SCALE_MAX_HISTORY)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (margin > ET_FP8_DELAYED_SCALE_MAX_MARGIN)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(eps) || eps <= 0.0f)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(min_scale) || min_scale <= 0.0f)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!f32_is_finite(max_scale) || max_scale < min_scale)
    return ET_FP8_ERR_INVALID_ARGUMENT;

  et_fp8_delayed_scale_state_v1 next;
  memset(&next, 0, sizeof(next));
  next.abi_version = ET_FP8_GEMM_ABI_MAJOR;
  next.format = format;
  next.history_len = history_len;
  next.cursor = 0u;
  next.count = 0u;
  next.margin = margin;
  next.eps = eps;
  next.min_scale = min_scale;
  next.max_scale = max_scale;

  float init_scale = 1.0f;
  compute_delayed_scale_internal(&next, &init_scale, NULL);
  next.current_scale = init_scale;

  *state = next;
  return ET_FP8_OK;
}

et_fp8_status et_fp8_delayed_scale_update(et_fp8_delayed_scale_state_v1 *state,
                                          float observed_amax) {
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  const et_fp8_status st = validate_delayed_scale_state(state);
  if (st != ET_FP8_OK)
    return st;
  if (!f32_is_finite(observed_amax))
    return ET_FP8_ERR_NONFINITE;
  if (observed_amax < 0.0f)
    return ET_FP8_ERR_INVALID_ARGUMENT;

  /* Canonicalize -0.0f to +0.0f. */
  const float clean_amax = (observed_amax == 0.0f) ? 0.0f : observed_amax;

  state->history[state->cursor] = clean_amax;
  state->cursor = (state->cursor + 1u) % state->history_len;
  if (state->count < state->history_len)
    state->count += 1u;

  float next_scale = 1.0f;
  compute_delayed_scale_internal(state, &next_scale, NULL);
  state->current_scale = next_scale;
  return ET_FP8_OK;
}

et_fp8_status
et_fp8_delayed_scale_update_tensor(et_fp8_delayed_scale_state_v1 *state,
                                   const float *tensor, size_t count,
                                   float *out_observed_amax) {
  size_t tensor_bytes = 0u;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  const et_fp8_status st = validate_delayed_scale_state(state);
  if (st != ET_FP8_OK)
    return st;
  if (tensor == NULL || count == 0u)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (mul_size_overflow(count, sizeof(float), &tensor_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(state, sizeof(*state), tensor, tensor_bytes))
    return ET_FP8_ERR_ALIAS;
  if (out_observed_amax != NULL) {
    if (ranges_overlap(state, sizeof(*state), out_observed_amax,
                       sizeof(float)) ||
        ranges_overlap(tensor, tensor_bytes, out_observed_amax, sizeof(float)))
      return ET_FP8_ERR_ALIAS;
  }

  /* Pass 1: validate all tensor entries and compute exact amax. */
  float amax = 0.0f;
  for (size_t i = 0u; i < count; ++i) {
    const float v = load_f32_at(tensor, i);
    if (!f32_is_finite(v))
      return ET_FP8_ERR_NONFINITE;
    const float a = f32_abs(v);
    if (a > amax)
      amax = a;
  }

  /* Pass 2: commit ring buffer update and optional output. */
  const et_fp8_status ust = et_fp8_delayed_scale_update(state, amax);
  if (ust != ET_FP8_OK)
    return ust;
  if (out_observed_amax != NULL)
    store_f32_at(out_observed_amax, 0u, amax);
  return ET_FP8_OK;
}

et_fp8_status
et_fp8_delayed_scale_compute(const et_fp8_delayed_scale_state_v1 *state,
                             float *out_scale, float *out_history_amax) {
  if (out_scale == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  const et_fp8_status st = validate_delayed_scale_state(state);
  if (st != ET_FP8_OK)
    return st;
  if (ranges_overlap(state, sizeof(*state), out_scale, sizeof(float)))
    return ET_FP8_ERR_ALIAS;
  if (out_history_amax != NULL) {
    if (ranges_overlap(state, sizeof(*state), out_history_amax,
                       sizeof(float)) ||
        ranges_overlap(out_scale, sizeof(float), out_history_amax,
                       sizeof(float)))
      return ET_FP8_ERR_ALIAS;
  }

  float scale = 1.0f;
  float hist_amax = 0.0f;
  compute_delayed_scale_internal(state, &scale, &hist_amax);
  store_f32_at(out_scale, 0u, scale);
  if (out_history_amax != NULL)
    store_f32_at(out_history_amax, 0u, hist_amax);
  return ET_FP8_OK;
}

/* ---- Block-Scaled W8A8 E4M3FN Quantization, Dequantization, GEMM & STE ---- */

static float compute_dequant_scale_from_amax(float amax) {
  if (amax == 0.0f)
    return 1.0f;
  float s = (float)((double)amax / (double)ET_FP8_E4M3_MAX_FINITE);
  if (s < FLT_MIN)
    s = FLT_MIN;
  return s;
}

et_fp8_status et_fp8_block_quantize_e4m3_v1(const float *src, size_t rows,
                                            size_t cols, size_t block_rows,
                                            size_t block_cols, uint8_t *dst_q,
                                            float *dst_scales) {
  size_t total_elems = 0u;
  size_t src_bytes = 0u;
  size_t num_scales = 0u;
  size_t scale_bytes = 0u;

  if (src == NULL || dst_q == NULL || dst_scales == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (!valid_block_n(block_rows) || !valid_block_k(block_cols))
    return ET_FP8_ERR_UNSUPPORTED_BLOCK;
  if (rows == 0u || cols == 0u || (rows % block_rows) != 0u ||
      (cols % block_cols) != 0u)
    return ET_FP8_ERR_SHAPE;

  const size_t r_blocks = rows / block_rows;
  const size_t c_blocks = cols / block_cols;
  if (mul_size_overflow(rows, cols, &total_elems) ||
      mul_size_overflow(total_elems, sizeof(float), &src_bytes) ||
      mul_size_overflow(r_blocks, c_blocks, &num_scales) ||
      mul_size_overflow(num_scales, sizeof(float), &scale_bytes))
    return ET_FP8_ERR_SHAPE;

  if (ranges_overlap(src, src_bytes, dst_q, total_elems) ||
      ranges_overlap(src, src_bytes, dst_scales, scale_bytes) ||
      ranges_overlap(dst_q, total_elems, dst_scales, scale_bytes))
    return ET_FP8_ERR_ALIAS;

  /* Pass 1: validate every input value is finite before touching outputs. */
  for (size_t i = 0u; i < total_elems; ++i) {
    if (!f32_is_finite(load_f32_at(src, i)))
      return ET_FP8_ERR_NONFINITE;
  }

  /* Pass 2: compute per-block scale and quantize every block. */
  for (size_t rb = 0u; rb < r_blocks; ++rb) {
    for (size_t cb = 0u; cb < c_blocks; ++cb) {
      float amax = 0.0f;
      for (size_t rr = 0u; rr < block_rows; ++rr) {
        const size_t r = rb * block_rows + rr;
        const size_t row_off = r * cols + cb * block_cols;
        for (size_t cc = 0u; cc < block_cols; ++cc) {
          const float a = f32_abs(load_f32_at(src, row_off + cc));
          if (a > amax)
            amax = a;
        }
      }

      const float scale = compute_dequant_scale_from_amax(amax);
      const double inv_scale = 1.0 / (double)scale;
      store_f32_at(dst_scales, rb * c_blocks + cb, scale);

      for (size_t rr = 0u; rr < block_rows; ++rr) {
        const size_t r = rb * block_rows + rr;
        const size_t row_off = r * cols + cb * block_cols;
        for (size_t cc = 0u; cc < block_cols; ++cc) {
          const float v = load_f32_at(src, row_off + cc);
          const float scaled = (float)((double)v * inv_scale);
          uint8_t q = 0u;
          (void)e4m3_encode_bits(f32_to_bits(scaled),
                                 ET_FP8_OVERFLOW_SATURATE_TO_MAX, &q);
          dst_q[row_off + cc] = q;
        }
      }
    }
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_block_dequantize_e4m3_v1(const uint8_t *src_q,
                                              const float *src_scales,
                                              size_t rows, size_t cols,
                                              size_t block_rows,
                                              size_t block_cols, float *dst) {
  size_t total_elems = 0u;
  size_t dst_bytes = 0u;
  size_t num_scales = 0u;
  size_t scale_bytes = 0u;

  if (src_q == NULL || src_scales == NULL || dst == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (!valid_block_n(block_rows) || !valid_block_k(block_cols))
    return ET_FP8_ERR_UNSUPPORTED_BLOCK;
  if (rows == 0u || cols == 0u || (rows % block_rows) != 0u ||
      (cols % block_cols) != 0u)
    return ET_FP8_ERR_SHAPE;

  const size_t r_blocks = rows / block_rows;
  const size_t c_blocks = cols / block_cols;
  if (mul_size_overflow(rows, cols, &total_elems) ||
      mul_size_overflow(total_elems, sizeof(float), &dst_bytes) ||
      mul_size_overflow(r_blocks, c_blocks, &num_scales) ||
      mul_size_overflow(num_scales, sizeof(float), &scale_bytes))
    return ET_FP8_ERR_SHAPE;

  if (ranges_overlap(src_q, total_elems, dst, dst_bytes) ||
      ranges_overlap(src_scales, scale_bytes, dst, dst_bytes))
    return ET_FP8_ERR_ALIAS;

  /* Pass 1: validate FP8 codes, positive finite scales, and overflow safety. */
  for (size_t i = 0u; i < total_elems; ++i) {
    if ((src_q[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }
  double max_scale = 0.0;
  for (size_t i = 0u; i < num_scales; ++i) {
    const float s = load_f32_at(src_scales, i);
    if (!f32_is_finite(s))
      return ET_FP8_ERR_NONFINITE;
    if (s <= 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
    if ((double)s > max_scale)
      max_scale = (double)s;
  }

  /* Dry-run overflow check only if worst-case product could exceed FLT_MAX. */
  if (max_scale * (double)ET_FP8_E4M3_MAX_FINITE >= 3.0e38) {
    for (size_t r = 0u; r < rows; ++r) {
      const size_t rb = r / block_rows;
      for (size_t c = 0u; c < cols; ++c) {
        const size_t cb = c / block_cols;
        const double qv = (double)e4m3_decode_valid_fast(src_q[r * cols + c]);
        const double sv = (double)load_f32_at(src_scales, rb * c_blocks + cb);
        const float out_v = (float)(qv * sv);
        if (!f32_is_finite(out_v))
          return ET_FP8_ERR_OVERFLOW;
      }
    }
  }

  /* Pass 2: commit dequantized values. */
  for (size_t r = 0u; r < rows; ++r) {
    const size_t rb = r / block_rows;
    for (size_t c = 0u; c < cols; ++c) {
      const size_t cb = c / block_cols;
      const double qv = (double)e4m3_decode_valid_fast(src_q[r * cols + c]);
      const double sv = (double)load_f32_at(src_scales, rb * c_blocks + cb);
      store_f32_at(dst, r * cols + c, (float)(qv * sv));
    }
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_per_tensor_quantize_e4m3_v1(const float *src, size_t count,
                                                 uint8_t *dst_q,
                                                 float *out_scale) {
  size_t src_bytes = 0u;
  if (src == NULL || dst_q == NULL || out_scale == NULL || count == 0u)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (mul_size_overflow(count, sizeof(float), &src_bytes))
    return ET_FP8_ERR_SHAPE;
  if (ranges_overlap(src, src_bytes, dst_q, count) ||
      ranges_overlap(src, src_bytes, out_scale, sizeof(float)) ||
      ranges_overlap(dst_q, count, out_scale, sizeof(float)))
    return ET_FP8_ERR_ALIAS;

  float amax = 0.0f;
  for (size_t i = 0u; i < count; ++i) {
    const float v = load_f32_at(src, i);
    if (!f32_is_finite(v))
      return ET_FP8_ERR_NONFINITE;
    const float a = f32_abs(v);
    if (a > amax)
      amax = a;
  }

  const float scale = compute_dequant_scale_from_amax(amax);
  const double inv_scale = 1.0 / (double)scale;
  store_f32_at(out_scale, 0u, scale);

  for (size_t i = 0u; i < count; ++i) {
    const float v = load_f32_at(src, i);
    const float scaled = (float)((double)v * inv_scale);
    uint8_t q = 0u;
    (void)e4m3_encode_bits(f32_to_bits(scaled),
                           ET_FP8_OVERFLOW_SATURATE_TO_MAX, &q);
    dst_q[i] = q;
  }
  return ET_FP8_OK;
}

static double block_gemm_dot_f64(size_t m_idx, size_t n_idx, size_t k,
                                 size_t block_k, size_t block_n,
                                 size_t k_blocks, const uint8_t *x_q,
                                 const float *x_scales, const uint8_t *w_q,
                                 const float *w_scales, const float *bias) {
  const uint8_t *x_row = x_q + m_idx * k;
  const uint8_t *w_row = w_q + n_idx * k;
  const size_t x_scale_base = m_idx * k_blocks;
  const size_t w_scale_base = (n_idx / block_n) * k_blocks;
  double total_acc = 0.0;

  for (size_t kb = 0u; kb < k_blocks; ++kb) {
    const size_t k0 = kb * block_k;
    double block_acc = 0.0;
    for (size_t kk = 0u; kk < block_k; ++kk) {
      const double xv = (double)e4m3_decode_valid_fast(x_row[k0 + kk]);
      const double wv = (double)e4m3_decode_valid_fast(w_row[k0 + kk]);
      block_acc += xv * wv;
    }
    const double sx = (double)load_f32_at(x_scales, x_scale_base + kb);
    const double sw = (double)load_f32_at(w_scales, w_scale_base + kb);
    total_acc += block_acc * (sx * sw);
  }
  if (bias != NULL)
    total_acc += (double)load_f32_at(bias, n_idx);
  return total_acc;
}

et_fp8_status et_fp8_block_gemm_e4m3_v1(size_t m, size_t n, size_t k,
                                        size_t block_k, size_t block_n,
                                        const uint8_t *x_q,
                                        const float *x_scales,
                                        const uint8_t *w_q,
                                        const float *w_scales,
                                        const float *bias, float *y) {
  size_t x_elems = 0u, w_elems = 0u, y_elems = 0u;
  size_t sx_elems = 0u, sw_elems = 0u;
  size_t sx_bytes = 0u, sw_bytes = 0u, y_bytes = 0u, bias_bytes = 0u;

  if (x_q == NULL || x_scales == NULL || w_q == NULL || w_scales == NULL ||
      y == NULL)
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (!valid_block_k(block_k) || !valid_block_n(block_n))
    return ET_FP8_ERR_UNSUPPORTED_BLOCK;
  if (m == 0u || n == 0u || k == 0u || (k % block_k) != 0u ||
      (n % block_n) != 0u)
    return ET_FP8_ERR_SHAPE;

  const size_t k_blocks = k / block_k;
  const size_t n_blocks = n / block_n;
  if (mul_size_overflow(m, k, &x_elems) || mul_size_overflow(n, k, &w_elems) ||
      mul_size_overflow(m, n, &y_elems) ||
      mul_size_overflow(m, k_blocks, &sx_elems) ||
      mul_size_overflow(n_blocks, k_blocks, &sw_elems) ||
      mul_size_overflow(sx_elems, sizeof(float), &sx_bytes) ||
      mul_size_overflow(sw_elems, sizeof(float), &sw_bytes) ||
      mul_size_overflow(y_elems, sizeof(float), &y_bytes) ||
      mul_size_overflow(n, sizeof(float), &bias_bytes))
    return ET_FP8_ERR_SHAPE;

  if (ranges_overlap(y, y_bytes, x_q, x_elems) ||
      ranges_overlap(y, y_bytes, x_scales, sx_bytes) ||
      ranges_overlap(y, y_bytes, w_q, w_elems) ||
      ranges_overlap(y, y_bytes, w_scales, sw_bytes) ||
      (bias != NULL && ranges_overlap(y, y_bytes, bias, bias_bytes)))
    return ET_FP8_ERR_ALIAS;

  /* Pass 1: validate all FP8 codes, positive finite scales, and finite bias. */
  for (size_t i = 0u; i < x_elems; ++i) {
    if ((x_q[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < w_elems; ++i) {
    if ((w_q[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }
  double max_sx = 0.0;
  for (size_t i = 0u; i < sx_elems; ++i) {
    const float s = load_f32_at(x_scales, i);
    if (!f32_is_finite(s))
      return ET_FP8_ERR_NONFINITE;
    if (s <= 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
    if ((double)s > max_sx)
      max_sx = (double)s;
  }
  double max_sw = 0.0;
  for (size_t i = 0u; i < sw_elems; ++i) {
    const float s = load_f32_at(w_scales, i);
    if (!f32_is_finite(s))
      return ET_FP8_ERR_NONFINITE;
    if (s <= 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
    if ((double)s > max_sw)
      max_sw = (double)s;
  }
  double max_bias = 0.0;
  if (bias != NULL) {
    for (size_t i = 0u; i < n; ++i) {
      const float b = load_f32_at(bias, i);
      if (!f32_is_finite(b))
        return ET_FP8_ERR_NONFINITE;
      const double ab = (double)f32_abs(b);
      if (ab > max_bias)
        max_bias = ab;
    }
  }

  /* Mathematical upper bound for f32 overflow safety before touching y. */
  const double bound =
      (double)k * (double)ET_FP8_E4M3_MAX_FINITE *
          (double)ET_FP8_E4M3_MAX_FINITE * max_sx * max_sw +
      max_bias;
  if (!f64_is_finite(bound) || bound >= 3.0e38) {
    for (size_t mi = 0u; mi < m; ++mi) {
      for (size_t ni = 0u; ni < n; ++ni) {
        const double acc =
            block_gemm_dot_f64(mi, ni, k, block_k, block_n, k_blocks, x_q,
                               x_scales, w_q, w_scales, bias);
        if (!f64_is_finite(acc) || !f32_is_finite((float)acc))
          return ET_FP8_ERR_OVERFLOW;
      }
    }
  }

  /* Pass 2: compute and store Y[m, n]. */
  for (size_t mi = 0u; mi < m; ++mi) {
    for (size_t ni = 0u; ni < n; ++ni) {
      const double acc =
          block_gemm_dot_f64(mi, ni, k, block_k, block_n, k_blocks, x_q,
                             x_scales, w_q, w_scales, bias);
      store_f32_at(y, mi * n + ni, (float)acc);
    }
  }
  return ET_FP8_OK;
}

et_fp8_status et_fp8_block_gemm_e4m3_backward_v1(
    size_t m, size_t n, size_t k, size_t block_k, size_t block_n,
    const float *dy, const uint8_t *x_q, const float *x_scales,
    const uint8_t *w_q, const float *w_scales, float *dx, float *dw,
    float *dbias) {
  size_t x_elems = 0u, w_elems = 0u, dy_elems = 0u;
  size_t sx_elems = 0u, sw_elems = 0u;
  size_t sx_bytes = 0u, sw_bytes = 0u, dy_bytes = 0u;
  size_t dx_bytes = 0u, dw_bytes = 0u, dbias_bytes = 0u;

  if (dy == NULL || x_q == NULL || x_scales == NULL || w_q == NULL ||
      w_scales == NULL || (dx == NULL && dw == NULL && dbias == NULL))
    return ET_FP8_ERR_INVALID_ARGUMENT;
  if (!fp_env_supported())
    return ET_FP8_ERR_ENVIRONMENT;
  if (!valid_block_k(block_k) || !valid_block_n(block_n))
    return ET_FP8_ERR_UNSUPPORTED_BLOCK;
  if (m == 0u || n == 0u || k == 0u || (k % block_k) != 0u ||
      (n % block_n) != 0u)
    return ET_FP8_ERR_SHAPE;

  const size_t k_blocks = k / block_k;
  const size_t n_blocks = n / block_n;
  if (mul_size_overflow(m, k, &x_elems) || mul_size_overflow(n, k, &w_elems) ||
      mul_size_overflow(m, n, &dy_elems) ||
      mul_size_overflow(m, k_blocks, &sx_elems) ||
      mul_size_overflow(n_blocks, k_blocks, &sw_elems) ||
      mul_size_overflow(sx_elems, sizeof(float), &sx_bytes) ||
      mul_size_overflow(sw_elems, sizeof(float), &sw_bytes) ||
      mul_size_overflow(dy_elems, sizeof(float), &dy_bytes) ||
      mul_size_overflow(x_elems, sizeof(float), &dx_bytes) ||
      mul_size_overflow(w_elems, sizeof(float), &dw_bytes) ||
      mul_size_overflow(n, sizeof(float), &dbias_bytes))
    return ET_FP8_ERR_SHAPE;

  /* Check output-vs-input and output-vs-output aliasing. */
  if (dx != NULL) {
    if (ranges_overlap(dx, dx_bytes, dy, dy_bytes) ||
        ranges_overlap(dx, dx_bytes, x_q, x_elems) ||
        ranges_overlap(dx, dx_bytes, x_scales, sx_bytes) ||
        ranges_overlap(dx, dx_bytes, w_q, w_elems) ||
        ranges_overlap(dx, dx_bytes, w_scales, sw_bytes))
      return ET_FP8_ERR_ALIAS;
  }
  if (dw != NULL) {
    if (ranges_overlap(dw, dw_bytes, dy, dy_bytes) ||
        ranges_overlap(dw, dw_bytes, x_q, x_elems) ||
        ranges_overlap(dw, dw_bytes, x_scales, sx_bytes) ||
        ranges_overlap(dw, dw_bytes, w_q, w_elems) ||
        ranges_overlap(dw, dw_bytes, w_scales, sw_bytes) ||
        (dx != NULL && ranges_overlap(dw, dw_bytes, dx, dx_bytes)))
      return ET_FP8_ERR_ALIAS;
  }
  if (dbias != NULL) {
    if (ranges_overlap(dbias, dbias_bytes, dy, dy_bytes) ||
        ranges_overlap(dbias, dbias_bytes, x_q, x_elems) ||
        ranges_overlap(dbias, dbias_bytes, x_scales, sx_bytes) ||
        ranges_overlap(dbias, dbias_bytes, w_q, w_elems) ||
        ranges_overlap(dbias, dbias_bytes, w_scales, sw_bytes) ||
        (dx != NULL && ranges_overlap(dbias, dbias_bytes, dx, dx_bytes)) ||
        (dw != NULL && ranges_overlap(dbias, dbias_bytes, dw, dw_bytes)))
      return ET_FP8_ERR_ALIAS;
  }

  /* Pass 1: validate all inputs. */
  double max_dy = 0.0;
  for (size_t i = 0u; i < dy_elems; ++i) {
    const float v = load_f32_at(dy, i);
    if (!f32_is_finite(v))
      return ET_FP8_ERR_NONFINITE;
    const double av = (double)f32_abs(v);
    if (av > max_dy)
      max_dy = av;
  }
  for (size_t i = 0u; i < x_elems; ++i) {
    if ((x_q[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < w_elems; ++i) {
    if ((w_q[i] & 0x7Fu) == 0x7Fu)
      return ET_FP8_ERR_NONFINITE;
  }
  double max_sx = 0.0;
  for (size_t i = 0u; i < sx_elems; ++i) {
    const float s = load_f32_at(x_scales, i);
    if (!f32_is_finite(s))
      return ET_FP8_ERR_NONFINITE;
    if (s <= 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
    if ((double)s > max_sx)
      max_sx = (double)s;
  }
  double max_sw = 0.0;
  for (size_t i = 0u; i < sw_elems; ++i) {
    const float s = load_f32_at(w_scales, i);
    if (!f32_is_finite(s))
      return ET_FP8_ERR_NONFINITE;
    if (s <= 0.0f)
      return ET_FP8_ERR_INVALID_ARGUMENT;
    if ((double)s > max_sw)
      max_sw = (double)s;
  }

  /* Dry-run overflow check only if mathematical upper bound risks FLT_MAX. */
  const double dx_bound =
      (double)n * max_dy * (double)ET_FP8_E4M3_MAX_FINITE * max_sw;
  const double dw_bound =
      (double)m * max_dy * (double)ET_FP8_E4M3_MAX_FINITE * max_sx;
  const double db_bound = (double)m * max_dy;
  const int need_dry =
      (dx != NULL && (!f64_is_finite(dx_bound) || dx_bound >= 3.0e38)) ||
      (dw != NULL && (!f64_is_finite(dw_bound) || dw_bound >= 3.0e38)) ||
      (dbias != NULL && (!f64_is_finite(db_bound) || db_bound >= 3.0e38));

  if (need_dry) {
    if (dx != NULL) {
      for (size_t mi = 0u; mi < m; ++mi) {
        for (size_t ki = 0u; ki < k; ++ki) {
          const size_t kb = ki / block_k;
          double acc = 0.0;
          for (size_t ni = 0u; ni < n; ++ni) {
            const double dy_v = (double)load_f32_at(dy, mi * n + ni);
            const double w_v =
                (double)e4m3_decode_valid_fast(w_q[ni * k + ki]) *
                (double)load_f32_at(w_scales, (ni / block_n) * k_blocks + kb);
            acc += dy_v * w_v;
          }
          if (!f64_is_finite(acc) || !f32_is_finite((float)acc))
            return ET_FP8_ERR_OVERFLOW;
        }
      }
    }
    if (dw != NULL) {
      for (size_t ni = 0u; ni < n; ++ni) {
        for (size_t ki = 0u; ki < k; ++ki) {
          const size_t kb = ki / block_k;
          double acc = 0.0;
          for (size_t mi = 0u; mi < m; ++mi) {
            const double dy_v = (double)load_f32_at(dy, mi * n + ni);
            const double x_v =
                (double)e4m3_decode_valid_fast(x_q[mi * k + ki]) *
                (double)load_f32_at(x_scales, mi * k_blocks + kb);
            acc += dy_v * x_v;
          }
          if (!f64_is_finite(acc) || !f32_is_finite((float)acc))
            return ET_FP8_ERR_OVERFLOW;
        }
      }
    }
    if (dbias != NULL) {
      for (size_t ni = 0u; ni < n; ++ni) {
        double acc = 0.0;
        for (size_t mi = 0u; mi < m; ++mi)
          acc += (double)load_f32_at(dy, mi * n + ni);
        if (!f64_is_finite(acc) || !f32_is_finite((float)acc))
          return ET_FP8_ERR_OVERFLOW;
      }
    }
  }

  /* Pass 2: commit gradients. */
  if (dx != NULL) {
    for (size_t mi = 0u; mi < m; ++mi) {
      for (size_t ki = 0u; ki < k; ++ki) {
        const size_t kb = ki / block_k;
        double acc = 0.0;
        for (size_t ni = 0u; ni < n; ++ni) {
          const double dy_v = (double)load_f32_at(dy, mi * n + ni);
          const double w_v =
              (double)e4m3_decode_valid_fast(w_q[ni * k + ki]) *
              (double)load_f32_at(w_scales, (ni / block_n) * k_blocks + kb);
          acc += dy_v * w_v;
        }
        store_f32_at(dx, mi * k + ki, (float)acc);
      }
    }
  }
  if (dw != NULL) {
    for (size_t ni = 0u; ni < n; ++ni) {
      for (size_t ki = 0u; ki < k; ++ki) {
        const size_t kb = ki / block_k;
        double acc = 0.0;
        for (size_t mi = 0u; mi < m; ++mi) {
          const double dy_v = (double)load_f32_at(dy, mi * n + ni);
          const double x_v =
              (double)e4m3_decode_valid_fast(x_q[mi * k + ki]) *
              (double)load_f32_at(x_scales, mi * k_blocks + kb);
          acc += dy_v * x_v;
        }
        store_f32_at(dw, ni * k + ki, (float)acc);
      }
    }
  }
  if (dbias != NULL) {
    for (size_t ni = 0u; ni < n; ++ni) {
      double acc = 0.0;
      for (size_t mi = 0u; mi < m; ++mi)
        acc += (double)load_f32_at(dy, mi * n + ni);
      store_f32_at(dbias, ni, (float)acc);
    }
  }
  return ET_FP8_OK;
}
