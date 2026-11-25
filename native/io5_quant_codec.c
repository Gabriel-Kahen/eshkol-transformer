#include "eshkol_transformer/io5_quant_abi.h"

#include <float.h>
#include <limits.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024,
               "IO5 quant codec requires IEEE-754 binary32 and binary64");

#define F32_ABS_MASK UINT32_C(0x7fffffff)
#define F32_INF_BITS UINT32_C(0x7f800000)
#define F16_MAX_AS_F32_BITS UINT32_C(0x477fe000)
#define F16_EXP_MASK UINT16_C(0x7c00)
#define BF16_EXP_MASK UINT16_C(0x7f80)

static uint16_t load_le16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8u));
}

static void store_le16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static uint32_t load_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) |
         ((uint32_t)p[3] << 24u);
}

static void store_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)((v >> 8u) & 0xffu);
  p[2] = (uint8_t)((v >> 16u) & 0xffu);
  p[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static float f32_from_bits(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t bits_from_f32(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int span_valid(const void *p, size_t bytes) {
  if (p == NULL) {
    return 0;
  }
  return (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (a_bytes == 0u || b_bytes == 0u) {
    return 0;
  }
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes) {
    return 1;
  }
  return left < right + b_bytes && right < left + a_bytes;
}

static uint32_t round_shift_even(uint32_t value, unsigned int shift) {
  const uint32_t retained = value >> shift;
  const uint32_t half = UINT32_C(1) << (shift - 1u);
  const uint32_t discarded = value & ((half << 1u) - 1u);
  return retained +
         (discarded > half || (discarded == half && (retained & 1u)));
}

static et_io5_quant_status f32_to_f16_internal(uint32_t bits, uint16_t *out) {
  const uint32_t abs_bits = bits & F32_ABS_MASK;
  if (abs_bits >= F32_INF_BITS) {
    return ET_IO5_QUANT_ERR_NONFINITE;
  }
  if (abs_bits > F16_MAX_AS_F32_BITS) {
    return ET_IO5_QUANT_ERR_OVERFLOW;
  }
  const uint32_t sign = (bits >> 16u) & UINT32_C(0x8000);
  const uint32_t exponent = (bits >> 23u) & UINT32_C(0xff);
  const uint32_t fraction = bits & UINT32_C(0x7fffff);
  if (exponent < 102u) {
    *out = (uint16_t)sign;
  } else if (exponent < 113u) {
    *out = (uint16_t)(sign | round_shift_even(UINT32_C(0x800000) | fraction,
                                              126u - exponent));
  } else {
    *out = (uint16_t)(sign | (((exponent - 112u) << 10u) +
                              round_shift_even(fraction, 13u)));
  }
  return ET_IO5_QUANT_OK;
}

static et_io5_quant_status f16_to_f32_internal(uint16_t bits, uint32_t *out) {
  if ((bits & F16_EXP_MASK) == F16_EXP_MASK) {
    return ET_IO5_QUANT_ERR_NONFINITE;
  }
  const uint32_t sign = ((uint32_t)bits & UINT32_C(0x8000)) << 16u;
  uint32_t fraction = (uint32_t)bits & UINT32_C(0x03ff);
  uint32_t exponent = ((uint32_t)bits >> 10u) & UINT32_C(0x1f);
  if (exponent == 0u) {
    if (fraction == 0u) {
      *out = sign;
      return ET_IO5_QUANT_OK;
    }
    exponent = 113u;
    while ((fraction & UINT32_C(0x0400)) == 0u) {
      fraction <<= 1u;
      --exponent;
    }
    fraction &= UINT32_C(0x03ff);
  } else {
    exponent += 112u;
  }
  *out = sign | (exponent << 23u) | (fraction << 13u);
  return ET_IO5_QUANT_OK;
}

static et_io5_quant_status f32_to_bf16_internal(uint32_t bits, uint16_t *out) {
  const uint32_t abs_bits = bits & F32_ABS_MASK;
  if (abs_bits >= F32_INF_BITS) {
    return ET_IO5_QUANT_ERR_NONFINITE;
  }
  const uint16_t narrowed =
      (uint16_t)((bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) >> 16u);
  if ((narrowed & BF16_EXP_MASK) == BF16_EXP_MASK) {
    return ET_IO5_QUANT_ERR_OVERFLOW;
  }
  *out = narrowed;
  return ET_IO5_QUANT_OK;
}

static et_io5_quant_status bf16_to_f32_internal(uint16_t bits, uint32_t *out) {
  if ((bits & BF16_EXP_MASK) == BF16_EXP_MASK) {
    return ET_IO5_QUANT_ERR_NONFINITE;
  }
  *out = (uint32_t)bits << 16u;
  return ET_IO5_QUANT_OK;
}

static int8_t round_nearest_even_clamp_i8(double value) {
  const int neg = value < 0.0;
  double mag = neg ? -value : value;
  if (mag >= 127.0) {
    return neg ? (int8_t)-127 : (int8_t)127;
  }
  int32_t k = (int32_t)mag;
  const double rem = mag - (double)k;
  if (rem > 0.5 || (rem == 0.5 && (k & 1) != 0)) {
    ++k;
  }
  if (k > 127) {
    k = 127;
  }
  return neg ? (int8_t)(-k) : (int8_t)k;
}

uint32_t et_io5_quant_abi_version(void) {
  return (ET_IO5_QUANT_ABI_MAJOR << 16u) | ET_IO5_QUANT_ABI_MINOR;
}

et_io5_quant_status et_io5_q8_0_byte_size(size_t element_count,
                                          size_t *out_bytes) {
  if (out_bytes == NULL) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  if (element_count == 0u || (element_count % ET_IO5_QK8_0) != 0u) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  const size_t blocks = element_count / ET_IO5_QK8_0;
  if (blocks > SIZE_MAX / ET_IO5_Q8_0_BLOCK_BYTES) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  *out_bytes = blocks * ET_IO5_Q8_0_BLOCK_BYTES;
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_f32_bits_to_f16_bits(uint32_t f32_bits,
                                                uint16_t *out_f16_bits) {
  if (out_f16_bits == NULL) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  return f32_to_f16_internal(f32_bits, out_f16_bits);
}

et_io5_quant_status et_io5_f16_bits_to_f32_bits(uint16_t f16_bits,
                                                uint32_t *out_f32_bits) {
  if (out_f32_bits == NULL) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  return f16_to_f32_internal(f16_bits, out_f32_bits);
}

et_io5_quant_status et_io5_f32_bits_to_bf16_bits(uint32_t f32_bits,
                                                 uint16_t *out_bf16_bits) {
  if (out_bf16_bits == NULL) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  return f32_to_bf16_internal(f32_bits, out_bf16_bits);
}

et_io5_quant_status et_io5_bf16_bits_to_f32_bits(uint16_t bf16_bits,
                                                 uint32_t *out_f32_bits) {
  if (out_f32_bits == NULL) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  return bf16_to_f32_internal(bf16_bits, out_f32_bits);
}

et_io5_quant_status et_io5_validate_f32_finite(const void *src_f32_le,
                                               size_t src_bytes,
                                               size_t element_count) {
  if (element_count > SIZE_MAX / 4u || src_bytes != element_count * 4u) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  if (!span_valid(src_f32_le, src_bytes)) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  const uint8_t *src = (const uint8_t *)src_f32_le;
  for (size_t i = 0u; i < element_count; ++i) {
    const uint32_t bits = load_le32(src + i * 4u);
    if ((bits & F32_ABS_MASK) >= F32_INF_BITS) {
      return ET_IO5_QUANT_ERR_NONFINITE;
    }
  }
  return ET_IO5_QUANT_OK;
}

static et_io5_quant_status validate_narrow_buffers(const void *src,
                                                   size_t src_bytes,
                                                   size_t src_width,
                                                   void *dst, size_t dst_bytes,
                                                   size_t dst_width,
                                                   size_t element_count) {
  if (element_count > SIZE_MAX / src_width ||
      element_count > SIZE_MAX / dst_width ||
      src_bytes != element_count * src_width ||
      dst_bytes != element_count * dst_width) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  if (!span_valid(src, src_bytes) || !span_valid(dst, dst_bytes)) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  if (spans_overlap(src, src_bytes, dst, dst_bytes)) {
    return ET_IO5_QUANT_ERR_ALIAS;
  }
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_encode_f16(const void *src_f32_le, size_t src_bytes,
                                      void *dst_f16_le, size_t dst_bytes,
                                      size_t element_count) {
  et_io5_quant_status st = validate_narrow_buffers(
      src_f32_le, src_bytes, 4u, dst_f16_le, dst_bytes, 2u, element_count);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  const uint8_t *src = (const uint8_t *)src_f32_le;
  uint8_t *dst = (uint8_t *)dst_f16_le;
  for (size_t i = 0u; i < element_count; ++i) {
    uint16_t dummy = 0u;
    st = f32_to_f16_internal(load_le32(src + i * 4u), &dummy);
    if (st != ET_IO5_QUANT_OK) {
      return st;
    }
  }
  for (size_t i = 0u; i < element_count; ++i) {
    uint16_t out = 0u;
    (void)f32_to_f16_internal(load_le32(src + i * 4u), &out);
    store_le16(dst + i * 2u, out);
  }
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_decode_f16(const void *src_f16_le, size_t src_bytes,
                                      void *dst_f32_le, size_t dst_bytes,
                                      size_t element_count) {
  et_io5_quant_status st = validate_narrow_buffers(
      src_f16_le, src_bytes, 2u, dst_f32_le, dst_bytes, 4u, element_count);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  const uint8_t *src = (const uint8_t *)src_f16_le;
  uint8_t *dst = (uint8_t *)dst_f32_le;
  for (size_t i = 0u; i < element_count; ++i) {
    uint32_t dummy = 0u;
    st = f16_to_f32_internal(load_le16(src + i * 2u), &dummy);
    if (st != ET_IO5_QUANT_OK) {
      return st;
    }
  }
  for (size_t i = 0u; i < element_count; ++i) {
    uint32_t out = 0u;
    (void)f16_to_f32_internal(load_le16(src + i * 2u), &out);
    store_le32(dst + i * 4u, out);
  }
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_encode_bf16(const void *src_f32_le, size_t src_bytes,
                                       void *dst_bf16_le, size_t dst_bytes,
                                       size_t element_count) {
  et_io5_quant_status st = validate_narrow_buffers(
      src_f32_le, src_bytes, 4u, dst_bf16_le, dst_bytes, 2u, element_count);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  const uint8_t *src = (const uint8_t *)src_f32_le;
  uint8_t *dst = (uint8_t *)dst_bf16_le;
  for (size_t i = 0u; i < element_count; ++i) {
    uint16_t dummy = 0u;
    st = f32_to_bf16_internal(load_le32(src + i * 4u), &dummy);
    if (st != ET_IO5_QUANT_OK) {
      return st;
    }
  }
  for (size_t i = 0u; i < element_count; ++i) {
    uint16_t out = 0u;
    (void)f32_to_bf16_internal(load_le32(src + i * 4u), &out);
    store_le16(dst + i * 2u, out);
  }
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_decode_bf16(const void *src_bf16_le,
                                       size_t src_bytes, void *dst_f32_le,
                                       size_t dst_bytes, size_t element_count) {
  et_io5_quant_status st = validate_narrow_buffers(
      src_bf16_le, src_bytes, 2u, dst_f32_le, dst_bytes, 4u, element_count);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  const uint8_t *src = (const uint8_t *)src_bf16_le;
  uint8_t *dst = (uint8_t *)dst_f32_le;
  for (size_t i = 0u; i < element_count; ++i) {
    uint32_t dummy = 0u;
    st = bf16_to_f32_internal(load_le16(src + i * 2u), &dummy);
    if (st != ET_IO5_QUANT_OK) {
      return st;
    }
  }
  for (size_t i = 0u; i < element_count; ++i) {
    uint32_t out = 0u;
    (void)bf16_to_f32_internal(load_le16(src + i * 2u), &out);
    store_le32(dst + i * 4u, out);
  }
  return ET_IO5_QUANT_OK;
}

static et_io5_quant_status compute_q8_block_scale(const uint8_t *in_blk,
                                                  uint16_t *out_d_f16) {
  uint32_t max_abs_bits = 0u;
  for (size_t i = 0u; i < ET_IO5_QK8_0; ++i) {
    const uint32_t abs_bits = load_le32(in_blk + i * 4u) & F32_ABS_MASK;
    if (abs_bits >= F32_INF_BITS) {
      return ET_IO5_QUANT_ERR_NONFINITE;
    }
    if (abs_bits > max_abs_bits) {
      max_abs_bits = abs_bits;
    }
  }
  if (max_abs_bits == 0u) {
    *out_d_f16 = UINT16_C(0x0000);
    return ET_IO5_QUANT_OK;
  }
  const double amax = (double)f32_from_bits(max_abs_bits);
  const float d_f32 = (float)(amax / 127.0);
  return f32_to_f16_internal(bits_from_f32(d_f32), out_d_f16);
}

et_io5_quant_status et_io5_quantize_q8_0(const void *src_f32_le,
                                         size_t src_bytes, void *dst_q8_0,
                                         size_t dst_bytes,
                                         size_t element_count) {
  size_t expected_q8_bytes = 0u;
  et_io5_quant_status st =
      et_io5_q8_0_byte_size(element_count, &expected_q8_bytes);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  if (element_count > SIZE_MAX / 4u || src_bytes != element_count * 4u ||
      dst_bytes != expected_q8_bytes) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  if (!span_valid(src_f32_le, src_bytes) || !span_valid(dst_q8_0, dst_bytes)) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  if (spans_overlap(src_f32_le, src_bytes, dst_q8_0, dst_bytes)) {
    return ET_IO5_QUANT_ERR_ALIAS;
  }

  const uint8_t *src = (const uint8_t *)src_f32_le;
  uint8_t *dst = (uint8_t *)dst_q8_0;
  const size_t n_blocks = element_count / ET_IO5_QK8_0;

  /* Pass 1: validate every block before mutating dst_q8_0. */
  for (size_t b = 0u; b < n_blocks; ++b) {
    uint16_t d_f16 = 0u;
    st = compute_q8_block_scale(src + b * ET_IO5_QK8_0 * 4u, &d_f16);
    if (st != ET_IO5_QUANT_OK) {
      return st;
    }
  }

  /* Pass 2: quantize validated blocks. */
  for (size_t b = 0u; b < n_blocks; ++b) {
    const uint8_t *in_blk = src + b * ET_IO5_QK8_0 * 4u;
    uint8_t *out_blk = dst + b * ET_IO5_Q8_0_BLOCK_BYTES;
    uint16_t d_f16 = 0u;
    (void)compute_q8_block_scale(in_blk, &d_f16);
    store_le16(out_blk, d_f16);
    if ((d_f16 & UINT16_C(0x7fff)) == 0u) {
      memset(out_blk + 2u, 0, ET_IO5_QK8_0);
    } else {
      uint32_t d_f32_bits = 0u;
      (void)f16_to_f32_internal(d_f16, &d_f32_bits);
      const double d_stored = (double)f32_from_bits(d_f32_bits);
      for (size_t i = 0u; i < ET_IO5_QK8_0; ++i) {
        const double x = (double)f32_from_bits(load_le32(in_blk + i * 4u));
        const int8_t q = round_nearest_even_clamp_i8(x / d_stored);
        out_blk[2u + i] = (uint8_t)q;
      }
    }
  }
  return ET_IO5_QUANT_OK;
}

et_io5_quant_status et_io5_dequantize_q8_0(const void *src_q8_0,
                                           size_t src_bytes, void *dst_f32_le,
                                           size_t dst_bytes,
                                           size_t element_count) {
  size_t expected_q8_bytes = 0u;
  et_io5_quant_status st =
      et_io5_q8_0_byte_size(element_count, &expected_q8_bytes);
  if (st != ET_IO5_QUANT_OK) {
    return st;
  }
  if (element_count > SIZE_MAX / 4u || src_bytes != expected_q8_bytes ||
      dst_bytes != element_count * 4u) {
    return ET_IO5_QUANT_ERR_INVALID_LENGTH;
  }
  if (!span_valid(src_q8_0, src_bytes) || !span_valid(dst_f32_le, dst_bytes)) {
    return ET_IO5_QUANT_ERR_INVALID_ARG;
  }
  if (spans_overlap(src_q8_0, src_bytes, dst_f32_le, dst_bytes)) {
    return ET_IO5_QUANT_ERR_ALIAS;
  }

  const uint8_t *src = (const uint8_t *)src_q8_0;
  uint8_t *dst = (uint8_t *)dst_f32_le;
  const size_t n_blocks = element_count / ET_IO5_QK8_0;

  /* Pass 1: reject any non-finite f16 block scale before mutating dst_f32_le. */
  for (size_t b = 0u; b < n_blocks; ++b) {
    const uint16_t d_f16 = load_le16(src + b * ET_IO5_Q8_0_BLOCK_BYTES);
    if ((d_f16 & F16_EXP_MASK) == F16_EXP_MASK) {
      return ET_IO5_QUANT_ERR_NONFINITE;
    }
  }

  /* Pass 2: dequantize validated blocks. */
  for (size_t b = 0u; b < n_blocks; ++b) {
    const uint8_t *in_blk = src + b * ET_IO5_Q8_0_BLOCK_BYTES;
    uint8_t *out_blk = dst + b * ET_IO5_QK8_0 * 4u;
    const uint16_t d_f16 = load_le16(in_blk);
    uint32_t d_f32_bits = 0u;
    (void)f16_to_f32_internal(d_f16, &d_f32_bits);
    const float d = f32_from_bits(d_f32_bits);
    for (size_t i = 0u; i < ET_IO5_QK8_0; ++i) {
      const int8_t q = (int8_t)in_blk[2u + i];
      const float x_hat = d * (float)q;
      store_le32(out_blk + i * 4u, bits_from_f32(x_hat));
    }
  }
  return ET_IO5_QUANT_OK;
}
