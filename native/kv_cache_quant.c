#define _GNU_SOURCE

#include "eshkol_transformer/kv_cache_quant_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "Quantized KV-cache requires 8-bit bytes and IEEE-754 f32/f64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0 && DBL_MANT_DIG == 53 &&
                   DBL_MAX_EXP == 1024,
               "Quantized KV-cache requires IEEE-754 binary32/binary64");

/* ---- Alignment-safe IEEE-754 bit helpers and overflow checks ---- */

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

static int mul_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
  if (a != 0u && b > UINT64_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static int add_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
  if (b > UINT64_MAX - a)
    return 0;
  *out = a + b;
  return 1;
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

typedef struct span_entry {
  const void *ptr;
  size_t bytes;
} span_entry;

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

static int any_spans_overlap(const span_entry *entries, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    for (size_t j = 0u; j < i; ++j) {
      if (spans_overlap(entries[i].ptr, entries[i].bytes, entries[j].ptr,
                        entries[j].bytes))
        return 1;
    }
  }
  return 0;
}

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, mask all exceptions, disable FTZ (bit 15) & DAZ (bit 6). */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

/* ---- Standalone SHA-256 (FIPS 180-4) for ETKQ1000 snapshots ---- */

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
  for (i = 0u; i < 16u; ++i) {
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
  for (i = 0u; i < 64u; ++i) {
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
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0u; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256_bytes(const uint8_t *data, size_t len, uint8_t digest[32]) {
  sha256_state st;
  sha256_init(&st);
  if (len > 0u)
    sha256_update(&st, data, len);
  sha256_final(&st, digest);
}

static int ct_memeq_32(const uint8_t a[32], const uint8_t b[32]) {
  uint8_t diff = 0u;
  for (size_t i = 0u; i < 32u; ++i)
    diff |= (uint8_t)(a[i] ^ b[i]);
  return diff == 0u;
}

static void put_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static uint16_t get_u16_le(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static uint32_t get_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static void put_u64_le(uint8_t *dst, uint64_t v) {
  for (size_t i = 0u; i < 8u; ++i)
    dst[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
}

static uint64_t get_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    v |= ((uint64_t)src[i]) << (8u * i);
  return v;
}

/* ---- Bit-Exact OCP FP8 E4M3FN & Symmetric INT8 RNE Primitives ---- */

static et_kvq_status e4m3_encode_bits(uint32_t f32_bits, uint8_t *out_code) {
  const uint8_t sign = (uint8_t)((f32_bits >> 31u) << 7u);
  const uint32_t abs_u = f32_bits & 0x7FFFFFFFu;
  const int32_t f32_exp = (int32_t)((abs_u >> 23u) & 0xFFu);
  const uint32_t f32_mant = abs_u & 0x7FFFFFu;

  if (f32_exp == 0xFF)
    return ET_KVQ_ERR_NONFINITE;

  if (abs_u == 0u || f32_exp == 0) {
    *out_code = sign;
    return ET_KVQ_OK;
  }

  const int32_t e = f32_exp - 127;
  const int32_t fp8_exp = e + 7;

  if (fp8_exp >= 1) {
    if (fp8_exp > 15) {
      *out_code = (uint8_t)(sign | 0x7Eu);
      return ET_KVQ_OK;
    }

    const uint32_t q = f32_mant >> 20u;
    const uint32_t rem = f32_mant & 0xFFFFFu;
    const uint32_t half = 0x80000u;
    const uint32_t round_up =
        (rem > half || (rem == half && (q & 1u) != 0u)) ? 1u : 0u;
    const uint32_t res = (((uint32_t)fp8_exp << 3u) | q) + round_up;

    /* E4M3FN has no Inf: 0x7F is NaN, so >= 0x7F saturates to 448.0 (0x7E). */
    if (res >= 0x7Fu) {
      *out_code = (uint8_t)(sign | 0x7Eu);
      return ET_KVQ_OK;
    }

    *out_code = (uint8_t)(sign | (uint8_t)res);
    return ET_KVQ_OK;
  }

  /* Subnormal / underflow range: e <= -7. Half of min subnormal is 2^-10. */
  if (e < -10) {
    *out_code = sign;
    return ET_KVQ_OK;
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
  return ET_KVQ_OK;
}

static et_kvq_status e4m3_decode_bits(uint8_t code, uint32_t *out_f32_bits) {
  const uint32_t sign = (uint32_t)(code >> 7u);
  const uint32_t exp = (uint32_t)((code >> 3u) & 0xFu);
  const uint32_t mant = (uint32_t)(code & 0x7u);

  if (exp == 15u && mant == 7u)
    return ET_KVQ_ERR_NONFINITE;

  if (exp == 0u) {
    if (mant == 0u) {
      *out_f32_bits = sign << 31u;
      return ET_KVQ_OK;
    }
    uint32_t m = mant;
    uint32_t f32_exp = 121u; /* 127 - 6 */
    while ((m & 0x8u) == 0u) {
      m <<= 1u;
      f32_exp -= 1u;
    }
    const uint32_t f32_mant = (m & 0x7u) << 20u;
    *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
    return ET_KVQ_OK;
  }

  const uint32_t f32_exp = exp + 120u; /* exp - 7 + 127 */
  const uint32_t f32_mant = mant << 20u;
  *out_f32_bits = (sign << 31u) | (f32_exp << 23u) | f32_mant;
  return ET_KVQ_OK;
}

/* Exact IEEE-754 Round-to-Nearest-Even from finite double into [-127, 127]. */
static int8_t rne_round_clamp_i8(double v) {
  if (v >= 127.0)
    return (int8_t)127;
  if (v <= -127.0)
    return (int8_t)-127;
  const int neg = (v < 0.0);
  const double av = neg ? -v : v;
  int32_t fl = (int32_t)av;
  const double frac = av - (double)fl;
  if (frac > 0.5 || (frac == 0.5 && (fl & 1) != 0))
    fl += 1;
  if (fl > 127)
    fl = 127;
  return (int8_t)(neg ? -fl : fl);
}

static void quantize_int8_vec_unchecked(const float *src, size_t count,
                                        int8_t *dst_q, float *out_scale) {
  float amax = 0.0f;
  for (size_t i = 0u; i < count; ++i) {
    const float a = f32_abs(load_f32_at(src, i));
    if (a > amax)
      amax = a;
  }
  if (amax == 0.0f) {
    *out_scale = 0.0f;
    memset(dst_q, 0, count);
    return;
  }
  float scale = (float)((double)amax / 127.0);
  if (scale == 0.0f) {
    /* Subnormal amax < 63.5 * 2^-149: use smallest positive binary32 subnormal
     * so every x[i] / scale is an exact integer in [-63, 63]. */
    scale = bits_to_f32(1u);
  }
  const double dscale = (double)scale;
  for (size_t i = 0u; i < count; ++i) {
    const double x = (double)load_f32_at(src, i);
    dst_q[i] = rne_round_clamp_i8(x / dscale);
  }
  *out_scale = scale;
}

static void quantize_fp8_vec_unchecked(const float *src, size_t count,
                                       uint8_t *dst_q, float *out_scale) {
  float amax = 0.0f;
  for (size_t i = 0u; i < count; ++i) {
    const float a = f32_abs(load_f32_at(src, i));
    if (a > amax)
      amax = a;
  }
  if (amax == 0.0f) {
    *out_scale = 0.0f;
    memset(dst_q, 0, count);
    return;
  }
  float scale = (float)((double)amax / 448.0);
  if (scale == 0.0f) {
    scale = bits_to_f32(1u);
  }
  const double dscale = (double)scale;
  for (size_t i = 0u; i < count; ++i) {
    const float x = load_f32_at(src, i);
    const float scaled = (float)((double)x / dscale);
    uint8_t code = 0u;
    (void)e4m3_encode_bits(f32_to_bits(scaled), &code);
    dst_q[i] = code;
  }
  *out_scale = scale;
}

static float dequant_int8_elem_f32(int8_t q, double dscale) {
  return (float)((double)q * dscale);
}

static et_kvq_status dequant_fp8_elem_f32(uint8_t code, double dscale,
                                          float *out_val) {
  uint32_t bits = 0u;
  const et_kvq_status st = e4m3_decode_bits(code, &bits);
  if (st != ET_KVQ_OK)
    return st;
  const float fv = (float)((double)bits_to_f32(bits) * dscale);
  if (!f32_is_finite(fv))
    return ET_KVQ_ERR_NONFINITE;
  *out_val = fv;
  return ET_KVQ_OK;
}

/* ---- Cache Struct & Storage Indexing ----
 *
 * Contiguous backing buffer layout (4-byte aligned, zero wasted padding):
 *
 *   ET_KVQ_FMT_FP32:
 *     payload_f32: [num_layers, 2 (K=0,V=1), max_seq_len, num_kv_heads, head_dim]
 *                  (total = num_vecs * head_dim * 4 bytes)
 *
 *   ET_KVQ_FMT_INT8_SYM / ET_KVQ_FMT_FP8_E4M3FN:
 *     Region 0 (offset 0, 4-byte aligned):
 *       scales_f32: [num_layers, 2, max_seq_len, num_kv_heads]
 *                   (total = num_vecs * 4 bytes)
 *     Region 1 (offset = num_vecs * 4):
 *       quant_u8:   [num_layers, 2, max_seq_len, num_kv_heads, head_dim]
 *                   (total = num_vecs * head_dim bytes)
 *     Combined total = num_vecs * (head_dim + 4) bytes!
 */

struct et_kvq_cache {
  et_kvq_config cfg;
  size_t storage_bytes;
  size_t total_vectors; /* num_layers * 2 * max_seq_len * num_kv_heads */
  size_t scales_offset; /* 0 for quantized; 0/unused for FP32 */
  size_t payload_offset;
  size_t vecs_per_layer; /* 2 * max_seq_len * num_kv_heads */
  size_t vecs_per_side;  /* max_seq_len * num_kv_heads */
  int owns_storage;
  uint8_t *storage;
  uint32_t layer_seq_len[ET_KVQ_MAX_LAYERS];
};

static size_t bytes_per_vector_for_format(et_kvq_format fmt,
                                          uint32_t head_dim) {
  if (fmt == ET_KVQ_FMT_FP32)
    return (size_t)head_dim * sizeof(float);
  return (size_t)head_dim + sizeof(float);
}

static size_t vector_index(const et_kvq_cache *cache, uint32_t layer,
                           uint32_t side, uint32_t pos, uint32_t kv_head) {
  return (size_t)layer * cache->vecs_per_layer +
         (size_t)side * cache->vecs_per_side +
         (size_t)pos * (size_t)cache->cfg.num_kv_heads + (size_t)kv_head;
}

static float *cache_scale_ptr(et_kvq_cache *cache, size_t vidx) {
  return (float *)(cache->storage + cache->scales_offset +
                   vidx * sizeof(float));
}

static const float *cache_scale_cptr(const et_kvq_cache *cache, size_t vidx) {
  return (const float *)(cache->storage + cache->scales_offset +
                         vidx * sizeof(float));
}

static uint8_t *cache_quant_ptr(et_kvq_cache *cache, size_t vidx) {
  return cache->storage + cache->payload_offset +
         vidx * (size_t)cache->cfg.head_dim;
}

static const uint8_t *cache_quant_cptr(const et_kvq_cache *cache, size_t vidx) {
  return cache->storage + cache->payload_offset +
         vidx * (size_t)cache->cfg.head_dim;
}

static float *cache_fp32_ptr(et_kvq_cache *cache, size_t vidx) {
  return (float *)(cache->storage +
                   vidx * (size_t)cache->cfg.head_dim * sizeof(float));
}

static const float *cache_fp32_cptr(const et_kvq_cache *cache, size_t vidx) {
  return (const float *)(cache->storage +
                         vidx * (size_t)cache->cfg.head_dim * sizeof(float));
}

/* ---- Public Configuration & Sizing APIs ---- */

et_kvq_status et_kvq_validate_config(const et_kvq_config *cfg) {
  if (cfg == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (cfg->reserved != 0u)
    return ET_KVQ_ERR_INVALID_ARG;
  if (cfg->format != ET_KVQ_FMT_FP32 && cfg->format != ET_KVQ_FMT_INT8_SYM &&
      cfg->format != ET_KVQ_FMT_FP8_E4M3FN)
    return ET_KVQ_ERR_UNSUPPORTED_FORMAT;
  if (cfg->num_layers < 1u || cfg->num_layers > ET_KVQ_MAX_LAYERS ||
      cfg->num_kv_heads < 1u || cfg->num_kv_heads > ET_KVQ_MAX_KV_HEADS ||
      cfg->head_dim < 1u || cfg->head_dim > ET_KVQ_MAX_HEAD_DIM ||
      cfg->max_seq_len < 1u || cfg->max_seq_len > ET_KVQ_MAX_SEQ_LEN)
    return ET_KVQ_ERR_SHAPE;

  uint64_t vecs = 0u;
  if (!mul_u64_checked((uint64_t)cfg->num_layers, 2u, &vecs) ||
      !mul_u64_checked(vecs, (uint64_t)cfg->max_seq_len, &vecs) ||
      !mul_u64_checked(vecs, (uint64_t)cfg->num_kv_heads, &vecs))
    return ET_KVQ_ERR_OVERFLOW;

  const uint64_t bpv =
      (uint64_t)bytes_per_vector_for_format(cfg->format, cfg->head_dim);
  uint64_t total_bytes = 0u;
  if (!mul_u64_checked(vecs, bpv, &total_bytes))
    return ET_KVQ_ERR_OVERFLOW;
  if (total_bytes > ET_KVQ_MAX_STORAGE_BYTES ||
      total_bytes > (uint64_t)SIZE_MAX)
    return ET_KVQ_ERR_OVERFLOW;

  /* Also ensure FP32 baseline capacity never overflows uint64_t. */
  uint64_t fp32_bytes = 0u;
  if (!mul_u64_checked(vecs, (uint64_t)cfg->head_dim * 4u, &fp32_bytes))
    return ET_KVQ_ERR_OVERFLOW;

  return ET_KVQ_OK;
}

et_kvq_status et_kvq_storage_bytes_required(const et_kvq_config *cfg,
                                            size_t *out_bytes) {
  if (cfg == NULL || out_bytes == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  const et_kvq_status st = et_kvq_validate_config(cfg);
  if (st != ET_KVQ_OK)
    return st;

  const uint64_t vecs = (uint64_t)cfg->num_layers * 2u *
                        (uint64_t)cfg->max_seq_len *
                        (uint64_t)cfg->num_kv_heads;
  const uint64_t bpv =
      (uint64_t)bytes_per_vector_for_format(cfg->format, cfg->head_dim);
  *out_bytes = (size_t)(vecs * bpv);
  return ET_KVQ_OK;
}

/* ---- Cache Lifecycle & State Queries ---- */

et_kvq_status et_kvq_cache_create(const et_kvq_config *cfg, void *external_pool,
                                  size_t external_pool_bytes,
                                  et_kvq_cache **out_cache) {
  if (cfg == NULL || out_cache == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;

  size_t req_bytes = 0u;
  const et_kvq_status st = et_kvq_storage_bytes_required(cfg, &req_bytes);
  if (st != ET_KVQ_OK)
    return st;

  if (external_pool == NULL) {
    if (external_pool_bytes != 0u)
      return ET_KVQ_ERR_INVALID_ARG;
  } else {
    if (external_pool_bytes < req_bytes)
      return ET_KVQ_ERR_CAPACITY;
    if (!aligned_span(external_pool, _Alignof(float), external_pool_bytes))
      return ET_KVQ_ERR_ALIGNMENT;
    const span_entry spans[3] = {
        {cfg, sizeof(*cfg)},
        {out_cache, sizeof(*out_cache)},
        {external_pool, external_pool_bytes},
    };
    if (any_spans_overlap(spans, 3u))
      return ET_KVQ_ERR_ALIAS;
  }

  et_kvq_cache *cache = (et_kvq_cache *)calloc(1u, sizeof(et_kvq_cache));
  if (cache == NULL)
    return ET_KVQ_ERR_ALLOC;

  uint8_t *storage = NULL;
  int owns_storage = 0;
  if (external_pool == NULL) {
    storage = (uint8_t *)calloc(1u, req_bytes);
    if (storage == NULL) {
      free(cache);
      return ET_KVQ_ERR_ALLOC;
    }
    owns_storage = 1;
  } else {
    storage = (uint8_t *)external_pool;
    memset(storage, 0, req_bytes);
    owns_storage = 0;
  }

  cache->cfg = *cfg;
  cache->storage_bytes = req_bytes;
  cache->vecs_per_side = (size_t)cfg->max_seq_len * (size_t)cfg->num_kv_heads;
  cache->vecs_per_layer = 2u * cache->vecs_per_side;
  cache->total_vectors = (size_t)cfg->num_layers * cache->vecs_per_layer;
  cache->scales_offset = 0u;
  cache->payload_offset = (cfg->format == ET_KVQ_FMT_FP32)
                              ? 0u
                              : (cache->total_vectors * sizeof(float));
  cache->owns_storage = owns_storage;
  cache->storage = storage;

  *out_cache = cache;
  return ET_KVQ_OK;
}

void et_kvq_cache_destroy(et_kvq_cache *cache) {
  if (cache == NULL)
    return;
  if (cache->owns_storage)
    free(cache->storage);
  free(cache);
}

et_kvq_status et_kvq_cache_reset(et_kvq_cache *cache) {
  if (cache == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  memset(cache->layer_seq_len, 0, sizeof(cache->layer_seq_len));
  memset(cache->storage, 0, cache->storage_bytes);
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_cache_truncate(et_kvq_cache *cache, uint32_t new_seq_len) {
  if (cache == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  /* Two-pass validate before mutating any layer so truncate is atomic. */
  for (uint32_t l = 0u; l < cache->cfg.num_layers; ++l) {
    if (new_seq_len > cache->layer_seq_len[l])
      return ET_KVQ_ERR_OUT_OF_RANGE;
  }

  const uint32_t hkv = cache->cfg.num_kv_heads;
  const uint32_t hd = cache->cfg.head_dim;
  for (uint32_t l = 0u; l < cache->cfg.num_layers; ++l) {
    const uint32_t old_len = cache->layer_seq_len[l];
    if (new_seq_len < old_len) {
      const size_t tail_vecs = (size_t)(old_len - new_seq_len) * (size_t)hkv;
      for (uint32_t side = 0u; side < 2u; ++side) {
        const size_t start_vidx = vector_index(cache, l, side, new_seq_len, 0u);
        if (cache->cfg.format == ET_KVQ_FMT_FP32) {
          memset(cache_fp32_ptr(cache, start_vidx), 0,
                 tail_vecs * (size_t)hd * sizeof(float));
        } else {
          memset(cache_scale_ptr(cache, start_vidx), 0,
                 tail_vecs * sizeof(float));
          memset(cache_quant_ptr(cache, start_vidx), 0,
                 tail_vecs * (size_t)hd);
        }
      }
      cache->layer_seq_len[l] = new_seq_len;
    }
  }
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_cache_memory_stats(const et_kvq_cache *cache,
                                        et_kvq_memory_stats *out_stats) {
  if (cache == NULL || out_stats == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;

  uint64_t total_slices = 0u;
  uint32_t min_len = cache->layer_seq_len[0];
  uint32_t max_len = cache->layer_seq_len[0];
  for (uint32_t l = 0u; l < cache->cfg.num_layers; ++l) {
    const uint32_t sl = cache->layer_seq_len[l];
    total_slices += (uint64_t)sl;
    if (sl < min_len)
      min_len = sl;
    if (sl > max_len)
      max_len = sl;
  }

  const uint64_t vecs_per_slice = 2u * (uint64_t)cache->cfg.num_kv_heads;
  const uint64_t active_vecs = total_slices * vecs_per_slice;
  const uint64_t fp32_bpv = (uint64_t)cache->cfg.head_dim * 4u;
  const uint64_t quant_bpv = (uint64_t)bytes_per_vector_for_format(
      cache->cfg.format, cache->cfg.head_dim);

  const uint64_t cap_vecs = (uint64_t)cache->total_vectors;
  const uint64_t cap_fp32 = cap_vecs * fp32_bpv;
  const uint64_t cap_quant = cap_vecs * quant_bpv;

  et_kvq_memory_stats st;
  st.fp32_baseline_bytes = active_vecs * fp32_bpv;
  st.quantized_bytes = active_vecs * quant_bpv;
  st.capacity_fp32_bytes = cap_fp32;
  st.capacity_quant_bytes = cap_quant;
  st.compression_ratio = (double)cap_fp32 / (double)cap_quant;
  st.min_seq_len = min_len;
  st.max_seq_len_used = max_len;
  *out_stats = st;
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_cache_config_get(const et_kvq_cache *cache,
                                      et_kvq_config *out_cfg) {
  if (cache == NULL || out_cfg == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  *out_cfg = cache->cfg;
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_cache_layer_seq_len(const et_kvq_cache *cache,
                                         uint32_t layer,
                                         uint32_t *out_seq_len) {
  if (cache == NULL || out_seq_len == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (layer >= cache->cfg.num_layers)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  *out_seq_len = cache->layer_seq_len[layer];
  return ET_KVQ_OK;
}

/* ---- Hot-Path Append, Dequantize & Fused Quantized Online-Softmax Decode ---- */

et_kvq_status et_kvq_append_token_v1(et_kvq_cache *cache, uint32_t layer,
                                     const float *k_new, const float *v_new) {
  if (cache == NULL || k_new == NULL || v_new == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (layer >= cache->cfg.num_layers)
    return ET_KVQ_ERR_OUT_OF_RANGE;

  const uint32_t pos = cache->layer_seq_len[layer];
  if (pos >= cache->cfg.max_seq_len)
    return ET_KVQ_ERR_CAPACITY;

  const uint32_t hkv = cache->cfg.num_kv_heads;
  const uint32_t hd = cache->cfg.head_dim;
  const size_t slice_floats = (size_t)hkv * (size_t)hd;
  const size_t slice_bytes = slice_floats * sizeof(float);

  const span_entry spans[4] = {
      {cache, sizeof(*cache)},
      {cache->storage, cache->storage_bytes},
      {k_new, slice_bytes},
      {v_new, slice_bytes},
  };
  /* Allow k_new == v_new? No, treat overlapping inputs with cache storage as alias. */
  if (spans_overlap(k_new, slice_bytes, cache, sizeof(*cache)) ||
      spans_overlap(v_new, slice_bytes, cache, sizeof(*cache)) ||
      spans_overlap(k_new, slice_bytes, cache->storage, cache->storage_bytes) ||
      spans_overlap(v_new, slice_bytes, cache->storage, cache->storage_bytes))
    return ET_KVQ_ERR_ALIAS;
  (void)spans;

  /* Pass 1: validate ALL inputs are finite before writing a single byte. */
  for (size_t i = 0u; i < slice_floats; ++i) {
    if (!f32_is_finite(load_f32_at(k_new, i)) ||
        !f32_is_finite(load_f32_at(v_new, i)))
      return ET_KVQ_ERR_NONFINITE;
  }

  /* Pass 2: quantize and store K (side=0) and V (side=1) at (layer, pos). */
  for (uint32_t h = 0u; h < hkv; ++h) {
    const uint8_t *k_src_bytes =
        (const uint8_t *)k_new + (size_t)h * (size_t)hd * sizeof(float);
    const uint8_t *v_src_bytes =
        (const uint8_t *)v_new + (size_t)h * (size_t)hd * sizeof(float);
    const size_t k_vidx = vector_index(cache, layer, 0u, pos, h);
    const size_t v_vidx = vector_index(cache, layer, 1u, pos, h);

    if (cache->cfg.format == ET_KVQ_FMT_FP32) {
      memcpy(cache_fp32_ptr(cache, k_vidx), k_src_bytes,
             (size_t)hd * sizeof(float));
      memcpy(cache_fp32_ptr(cache, v_vidx), v_src_bytes,
             (size_t)hd * sizeof(float));
    } else if (cache->cfg.format == ET_KVQ_FMT_INT8_SYM) {
      float k_scale = 0.0f;
      float v_scale = 0.0f;
      quantize_int8_vec_unchecked((const float *)k_src_bytes, (size_t)hd,
                                  (int8_t *)cache_quant_ptr(cache, k_vidx),
                                  &k_scale);
      quantize_int8_vec_unchecked((const float *)v_src_bytes, (size_t)hd,
                                  (int8_t *)cache_quant_ptr(cache, v_vidx),
                                  &v_scale);
      store_f32_at(cache_scale_ptr(cache, k_vidx), 0u, k_scale);
      store_f32_at(cache_scale_ptr(cache, v_vidx), 0u, v_scale);
    } else {
      float k_scale = 0.0f;
      float v_scale = 0.0f;
      quantize_fp8_vec_unchecked((const float *)k_src_bytes, (size_t)hd,
                                 cache_quant_ptr(cache, k_vidx), &k_scale);
      quantize_fp8_vec_unchecked((const float *)v_src_bytes, (size_t)hd,
                                 cache_quant_ptr(cache, v_vidx), &v_scale);
      store_f32_at(cache_scale_ptr(cache, k_vidx), 0u, k_scale);
      store_f32_at(cache_scale_ptr(cache, v_vidx), 0u, v_scale);
    }
  }

  cache->layer_seq_len[layer] = pos + 1u;
  return ET_KVQ_OK;
}

static et_kvq_status validate_stored_vector(const et_kvq_cache *cache,
                                            size_t vidx) {
  const uint32_t hd = cache->cfg.head_dim;
  if (cache->cfg.format == ET_KVQ_FMT_FP32) {
    const float *src = cache_fp32_cptr(cache, vidx);
    for (uint32_t d = 0u; d < hd; ++d) {
      if (!f32_is_finite(load_f32_at(src, d)))
        return ET_KVQ_ERR_NONFINITE;
    }
    return ET_KVQ_OK;
  }

  const float scale = load_f32_at(cache_scale_cptr(cache, vidx), 0u);
  if (!f32_is_finite(scale) || scale < 0.0f)
    return ET_KVQ_ERR_NONFINITE;
  const double dscale = (double)scale;

  if (cache->cfg.format == ET_KVQ_FMT_INT8_SYM) {
    const int8_t *q = (const int8_t *)cache_quant_cptr(cache, vidx);
    for (uint32_t d = 0u; d < hd; ++d) {
      if (q[d] == (int8_t)-128)
        return ET_KVQ_ERR_CORRUPT;
      if (scale == 0.0f && q[d] != 0)
        return ET_KVQ_ERR_CORRUPT;
      const float fv = dequant_int8_elem_f32(q[d], dscale);
      if (!f32_is_finite(fv))
        return ET_KVQ_ERR_NONFINITE;
    }
    return ET_KVQ_OK;
  }

  const uint8_t *q = cache_quant_cptr(cache, vidx);
  for (uint32_t d = 0u; d < hd; ++d) {
    float fv = 0.0f;
    const et_kvq_status st = dequant_fp8_elem_f32(q[d], dscale, &fv);
    if (st != ET_KVQ_OK)
      return st;
  }
  return ET_KVQ_OK;
}

static void dequant_stored_vector_unchecked(const et_kvq_cache *cache,
                                            size_t vidx, float *dst) {
  const uint32_t hd = cache->cfg.head_dim;
  if (cache->cfg.format == ET_KVQ_FMT_FP32) {
    memcpy(dst, cache_fp32_cptr(cache, vidx), (size_t)hd * sizeof(float));
    return;
  }
  const double dscale = (double)load_f32_at(cache_scale_cptr(cache, vidx), 0u);
  if (cache->cfg.format == ET_KVQ_FMT_INT8_SYM) {
    const int8_t *q = (const int8_t *)cache_quant_cptr(cache, vidx);
    for (uint32_t d = 0u; d < hd; ++d)
      store_f32_at(dst, d, dequant_int8_elem_f32(q[d], dscale));
    return;
  }
  const uint8_t *q = cache_quant_cptr(cache, vidx);
  for (uint32_t d = 0u; d < hd; ++d) {
    float fv = 0.0f;
    (void)dequant_fp8_elem_f32(q[d], dscale, &fv);
    store_f32_at(dst, d, fv);
  }
}

et_kvq_status et_kvq_dequant_slice_v1(const et_kvq_cache *cache, uint32_t layer,
                                      uint32_t pos, float *out_k,
                                      float *out_v) {
  if (cache == NULL || out_k == NULL || out_v == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (layer >= cache->cfg.num_layers)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  if (pos >= cache->layer_seq_len[layer])
    return ET_KVQ_ERR_OUT_OF_RANGE;

  const uint32_t hkv = cache->cfg.num_kv_heads;
  const uint32_t hd = cache->cfg.head_dim;
  const size_t slice_bytes = (size_t)hkv * (size_t)hd * sizeof(float);
  const span_entry spans[4] = {
      {cache, sizeof(*cache)},
      {cache->storage, cache->storage_bytes},
      {out_k, slice_bytes},
      {out_v, slice_bytes},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_KVQ_ERR_ALIAS;

  /* Pass 1: validate all K/V vectors before writing caller outputs. */
  for (uint32_t h = 0u; h < hkv; ++h) {
    et_kvq_status st =
        validate_stored_vector(cache, vector_index(cache, layer, 0u, pos, h));
    if (st != ET_KVQ_OK)
      return st;
    st = validate_stored_vector(cache, vector_index(cache, layer, 1u, pos, h));
    if (st != ET_KVQ_OK)
      return st;
  }

  /* Pass 2: dequantize directly into caller buffers. */
  for (uint32_t h = 0u; h < hkv; ++h) {
    uint8_t *dk = (uint8_t *)out_k + (size_t)h * (size_t)hd * sizeof(float);
    uint8_t *dv = (uint8_t *)out_v + (size_t)h * (size_t)hd * sizeof(float);
    dequant_stored_vector_unchecked(
        cache, vector_index(cache, layer, 0u, pos, h), (float *)dk);
    dequant_stored_vector_unchecked(
        cache, vector_index(cache, layer, 1u, pos, h), (float *)dv);
  }
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_read_raw_vector(const et_kvq_cache *cache, uint32_t layer,
                                     uint32_t is_v, uint32_t pos,
                                     uint32_t kv_head, uint8_t *out_payload,
                                     size_t payload_capacity,
                                     float *out_scale) {
  if (cache == NULL || out_payload == NULL || out_scale == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (is_v > 1u || layer >= cache->cfg.num_layers ||
      pos >= cache->layer_seq_len[layer] ||
      kv_head >= cache->cfg.num_kv_heads)
    return ET_KVQ_ERR_OUT_OF_RANGE;

  const size_t needed = (cache->cfg.format == ET_KVQ_FMT_FP32)
                            ? ((size_t)cache->cfg.head_dim * sizeof(float))
                            : (size_t)cache->cfg.head_dim;
  if (payload_capacity < needed)
    return ET_KVQ_ERR_CAPACITY;

  const span_entry spans[4] = {
      {cache, sizeof(*cache)},
      {cache->storage, cache->storage_bytes},
      {out_payload, needed},
      {out_scale, sizeof(*out_scale)},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_KVQ_ERR_ALIAS;

  const size_t vidx = vector_index(cache, layer, is_v, pos, kv_head);
  if (cache->cfg.format == ET_KVQ_FMT_FP32) {
    memcpy(out_payload, cache_fp32_cptr(cache, vidx), needed);
    *out_scale = 1.0f;
  } else {
    memcpy(out_payload, cache_quant_cptr(cache, vidx), needed);
    *out_scale = load_f32_at(cache_scale_cptr(cache, vidx), 0u);
  }
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_decode_gqa_v1(const et_kvq_cache *cache, uint32_t layer,
                                   uint32_t num_q_heads, float scale,
                                   uint32_t window_size, const float *q,
                                   float *out) {
  if (cache == NULL || q == NULL || out == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (layer >= cache->cfg.num_layers)
    return ET_KVQ_ERR_OUT_OF_RANGE;

  const uint32_t seq_len = cache->layer_seq_len[layer];
  if (seq_len == 0u)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  if (num_q_heads < 1u || num_q_heads > ET_KVQ_MAX_Q_HEADS ||
      (num_q_heads % cache->cfg.num_kv_heads) != 0u)
    return ET_KVQ_ERR_SHAPE;
  if (!f32_is_finite(scale))
    return ET_KVQ_ERR_NONFINITE;
  if (!(scale > 0.0f))
    return ET_KVQ_ERR_INVALID_ARG;

  const uint32_t hd = cache->cfg.head_dim;
  const size_t q_floats = (size_t)num_q_heads * (size_t)hd;
  const size_t q_bytes = q_floats * sizeof(float);

  const span_entry spans[4] = {
      {cache, sizeof(*cache)},
      {cache->storage, cache->storage_bytes},
      {q, q_bytes},
      {out, q_bytes},
  };
  if (any_spans_overlap(spans, 4u))
    return ET_KVQ_ERR_ALIAS;

  for (size_t i = 0u; i < q_floats; ++i) {
    if (!f32_is_finite(load_f32_at(q, i)))
      return ET_KVQ_ERR_NONFINITE;
  }

  /* Zero heap allocations: bounded stack staging guarantees failure atomicity
   * (out remains byte-for-byte untouched on any error). */
  float out_stage[ET_KVQ_MAX_Q_HEADS * ET_KVQ_MAX_HEAD_DIM];
  double acc[ET_KVQ_MAX_HEAD_DIM];

  const uint32_t group_size = num_q_heads / cache->cfg.num_kv_heads;
  const uint32_t t_start =
      (window_size == 0u || window_size >= seq_len) ? 0u
                                                    : (seq_len - window_size);
  const double dscale = (double)scale;
  const et_kvq_format fmt = cache->cfg.format;

  for (uint32_t qh = 0u; qh < num_q_heads; ++qh) {
    const uint32_t kv_h = qh / group_size;
    const uint8_t *q_head_bytes =
        (const uint8_t *)q + (size_t)qh * (size_t)hd * sizeof(float);

    double running_max = -INFINITY;
    double running_sum = 0.0;
    for (uint32_t d = 0u; d < hd; ++d)
      acc[d] = 0.0;

    for (uint32_t t = t_start; t < seq_len; ++t) {
      const size_t k_vidx = vector_index(cache, layer, 0u, t, kv_h);
      const size_t v_vidx = vector_index(cache, layer, 1u, t, kv_h);

      double dot = 0.0;
      if (fmt == ET_KVQ_FMT_FP32) {
        const float *k_vec = cache_fp32_cptr(cache, k_vidx);
        for (uint32_t d = 0u; d < hd; ++d) {
          const float kd = load_f32_at(k_vec, d);
          if (!f32_is_finite(kd))
            return ET_KVQ_ERR_NONFINITE;
          dot += (double)load_f32_at(q_head_bytes, d) * (double)kd;
        }
      } else if (fmt == ET_KVQ_FMT_INT8_SYM) {
        const float k_scale = load_f32_at(cache_scale_cptr(cache, k_vidx), 0u);
        if (!f32_is_finite(k_scale) || k_scale < 0.0f)
          return ET_KVQ_ERR_NONFINITE;
        const double kdscale = (double)k_scale;
        const int8_t *k_q = (const int8_t *)cache_quant_cptr(cache, k_vidx);
        for (uint32_t d = 0u; d < hd; ++d) {
          if (k_q[d] == (int8_t)-128 || (k_scale == 0.0f && k_q[d] != 0))
            return ET_KVQ_ERR_CORRUPT;
          const float kd = dequant_int8_elem_f32(k_q[d], kdscale);
          if (!f32_is_finite(kd))
            return ET_KVQ_ERR_NONFINITE;
          dot += (double)load_f32_at(q_head_bytes, d) * (double)kd;
        }
      } else {
        const float k_scale = load_f32_at(cache_scale_cptr(cache, k_vidx), 0u);
        if (!f32_is_finite(k_scale) || k_scale < 0.0f)
          return ET_KVQ_ERR_NONFINITE;
        const double kdscale = (double)k_scale;
        const uint8_t *k_q = cache_quant_cptr(cache, k_vidx);
        for (uint32_t d = 0u; d < hd; ++d) {
          float kd = 0.0f;
          const et_kvq_status st = dequant_fp8_elem_f32(k_q[d], kdscale, &kd);
          if (st != ET_KVQ_OK)
            return st;
          dot += (double)load_f32_at(q_head_bytes, d) * (double)kd;
        }
      }

      const double score = dot * dscale;
      if (!f64_is_finite(score))
        return ET_KVQ_ERR_NONFINITE;

      double alpha = 1.0;
      double weight = 1.0;
      if (score > running_max) {
        alpha = (running_sum == 0.0) ? 0.0 : exp(running_max - score);
        running_max = score;
        running_sum = running_sum * alpha + 1.0;
        weight = 1.0;
      } else {
        alpha = 1.0;
        weight = exp(score - running_max);
        running_sum += weight;
      }

      /* On-the-fly V dequantization and weighted online-softmax accumulation. */
      if (fmt == ET_KVQ_FMT_FP32) {
        const float *v_vec = cache_fp32_cptr(cache, v_vidx);
        if (alpha != 1.0) {
          for (uint32_t d = 0u; d < hd; ++d) {
            const float vd = load_f32_at(v_vec, d);
            if (!f32_is_finite(vd))
              return ET_KVQ_ERR_NONFINITE;
            acc[d] = acc[d] * alpha + (double)vd;
          }
        } else {
          for (uint32_t d = 0u; d < hd; ++d) {
            const float vd = load_f32_at(v_vec, d);
            if (!f32_is_finite(vd))
              return ET_KVQ_ERR_NONFINITE;
            acc[d] += weight * (double)vd;
          }
        }
      } else if (fmt == ET_KVQ_FMT_INT8_SYM) {
        const float v_scale = load_f32_at(cache_scale_cptr(cache, v_vidx), 0u);
        if (!f32_is_finite(v_scale) || v_scale < 0.0f)
          return ET_KVQ_ERR_NONFINITE;
        const double vdscale = (double)v_scale;
        const int8_t *v_q = (const int8_t *)cache_quant_cptr(cache, v_vidx);
        if (alpha != 1.0) {
          for (uint32_t d = 0u; d < hd; ++d) {
            if (v_q[d] == (int8_t)-128 || (v_scale == 0.0f && v_q[d] != 0))
              return ET_KVQ_ERR_CORRUPT;
            const float vd = dequant_int8_elem_f32(v_q[d], vdscale);
            if (!f32_is_finite(vd))
              return ET_KVQ_ERR_NONFINITE;
            acc[d] = acc[d] * alpha + (double)vd;
          }
        } else {
          for (uint32_t d = 0u; d < hd; ++d) {
            if (v_q[d] == (int8_t)-128 || (v_scale == 0.0f && v_q[d] != 0))
              return ET_KVQ_ERR_CORRUPT;
            const float vd = dequant_int8_elem_f32(v_q[d], vdscale);
            if (!f32_is_finite(vd))
              return ET_KVQ_ERR_NONFINITE;
            acc[d] += weight * (double)vd;
          }
        }
      } else {
        const float v_scale = load_f32_at(cache_scale_cptr(cache, v_vidx), 0u);
        if (!f32_is_finite(v_scale) || v_scale < 0.0f)
          return ET_KVQ_ERR_NONFINITE;
        const double vdscale = (double)v_scale;
        const uint8_t *v_q = cache_quant_cptr(cache, v_vidx);
        if (alpha != 1.0) {
          for (uint32_t d = 0u; d < hd; ++d) {
            float vd = 0.0f;
            const et_kvq_status st = dequant_fp8_elem_f32(v_q[d], vdscale, &vd);
            if (st != ET_KVQ_OK)
              return st;
            acc[d] = acc[d] * alpha + (double)vd;
          }
        } else {
          for (uint32_t d = 0u; d < hd; ++d) {
            float vd = 0.0f;
            const et_kvq_status st = dequant_fp8_elem_f32(v_q[d], vdscale, &vd);
            if (st != ET_KVQ_OK)
              return st;
            acc[d] += weight * (double)vd;
          }
        }
      }
    }

    if (!f64_is_finite(running_sum) || !(running_sum > 0.0))
      return ET_KVQ_ERR_NONFINITE;

    float *dst = out_stage + (size_t)qh * (size_t)hd;
    for (uint32_t d = 0u; d < hd; ++d) {
      const float fv = (float)(acc[d] / running_sum);
      if (!f32_is_finite(fv))
        return ET_KVQ_ERR_NONFINITE;
      dst[d] = fv;
    }
  }

  memcpy(out, out_stage, q_bytes);
  return ET_KVQ_OK;
}

/* ---- Public Bit-Exact Scalar & Vector Codec APIs ---- */

et_kvq_status et_kvq_fp8_e4m3_encode_scalar(float value, uint8_t *out_code) {
  if (out_code == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  return e4m3_encode_bits(f32_to_bits(value), out_code);
}

et_kvq_status et_kvq_fp8_e4m3_decode_scalar(uint8_t code, float *out_value) {
  if (out_value == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  uint32_t bits = 0u;
  const et_kvq_status st = e4m3_decode_bits(code, &bits);
  if (st != ET_KVQ_OK)
    return st;
  *out_value = bits_to_f32(bits);
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_int8_sym_quantize_vec(const float *src, size_t count,
                                           int8_t *dst_q, float *out_scale) {
  if (src == NULL || dst_q == NULL || out_scale == NULL || count == 0u)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  const size_t src_bytes = count * sizeof(float);
  if (count > SIZE_MAX / sizeof(float))
    return ET_KVQ_ERR_OVERFLOW;
  const span_entry spans[3] = {
      {src, src_bytes},
      {dst_q, count},
      {out_scale, sizeof(*out_scale)},
  };
  if (any_spans_overlap(spans, 3u))
    return ET_KVQ_ERR_ALIAS;
  for (size_t i = 0u; i < count; ++i) {
    if (!f32_is_finite(load_f32_at(src, i)))
      return ET_KVQ_ERR_NONFINITE;
  }
  quantize_int8_vec_unchecked(src, count, dst_q, out_scale);
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_int8_sym_dequantize_vec(const int8_t *src_q, size_t count,
                                             float scale, float *dst) {
  if (src_q == NULL || dst == NULL || count == 0u)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (!f32_is_finite(scale) || scale < 0.0f)
    return ET_KVQ_ERR_NONFINITE;
  if (count > SIZE_MAX / sizeof(float))
    return ET_KVQ_ERR_OVERFLOW;
  if (spans_overlap(src_q, count, dst, count * sizeof(float)))
    return ET_KVQ_ERR_ALIAS;
  const double dscale = (double)scale;
  for (size_t i = 0u; i < count; ++i) {
    if (src_q[i] == (int8_t)-128 || (scale == 0.0f && src_q[i] != 0))
      return ET_KVQ_ERR_CORRUPT;
    if (!f32_is_finite(dequant_int8_elem_f32(src_q[i], dscale)))
      return ET_KVQ_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < count; ++i)
    store_f32_at(dst, i, dequant_int8_elem_f32(src_q[i], dscale));
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_fp8_e4m3_quantize_vec(const float *src, size_t count,
                                           uint8_t *dst_q, float *out_scale) {
  if (src == NULL || dst_q == NULL || out_scale == NULL || count == 0u)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (count > SIZE_MAX / sizeof(float))
    return ET_KVQ_ERR_OVERFLOW;
  const span_entry spans[3] = {
      {src, count * sizeof(float)},
      {dst_q, count},
      {out_scale, sizeof(*out_scale)},
  };
  if (any_spans_overlap(spans, 3u))
    return ET_KVQ_ERR_ALIAS;
  for (size_t i = 0u; i < count; ++i) {
    if (!f32_is_finite(load_f32_at(src, i)))
      return ET_KVQ_ERR_NONFINITE;
  }
  quantize_fp8_vec_unchecked(src, count, dst_q, out_scale);
  return ET_KVQ_OK;
}

et_kvq_status et_kvq_fp8_e4m3_dequantize_vec(const uint8_t *src_q, size_t count,
                                             float scale, float *dst) {
  if (src_q == NULL || dst == NULL || count == 0u)
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;
  if (!f32_is_finite(scale) || scale < 0.0f)
    return ET_KVQ_ERR_NONFINITE;
  if (count > SIZE_MAX / sizeof(float))
    return ET_KVQ_ERR_OVERFLOW;
  if (spans_overlap(src_q, count, dst, count * sizeof(float)))
    return ET_KVQ_ERR_ALIAS;
  const double dscale = (double)scale;
  for (size_t i = 0u; i < count; ++i) {
    float fv = 0.0f;
    const et_kvq_status st = dequant_fp8_elem_f32(src_q[i], dscale, &fv);
    if (st != ET_KVQ_OK)
      return st;
  }
  for (size_t i = 0u; i < count; ++i) {
    float fv = 0.0f;
    (void)dequant_fp8_elem_f32(src_q[i], dscale, &fv);
    store_f32_at(dst, i, fv);
  }
  return ET_KVQ_OK;
}

/* ---- Versioned SHA-256 Checksummed Snapshot (ETKQ1000 v1.0) ---- */

struct path_parts {
  char *parent;
  char *base;
};

static void free_path_parts(struct path_parts *p) {
  free(p->parent);
  free(p->base);
  p->parent = NULL;
  p->base = NULL;
}

static int split_path(const char *path, struct path_parts *p) {
  memset(p, 0, sizeof(*p));
  if (path == NULL || path[0] == '\0')
    return -1;
  const char *slash = strrchr(path, '/');
  const char *base = (slash == NULL) ? path : (slash + 1);
  const size_t blen = strlen(base);
  if (blen == 0u || (blen == 1u && base[0] == '.') ||
      (blen == 2u && base[0] == '.' && base[1] == '.'))
    return -1;
  p->base = (char *)malloc(blen + 1u);
  if (p->base == NULL)
    return -1;
  memcpy(p->base, base, blen + 1u);
  if (slash == NULL) {
    p->parent = (char *)malloc(2u);
    if (p->parent != NULL)
      memcpy(p->parent, ".", 2u);
  } else if (slash == path) {
    p->parent = (char *)malloc(2u);
    if (p->parent != NULL)
      memcpy(p->parent, "/", 2u);
  } else {
    size_t plen = (size_t)(slash - path);
    while (plen > 1u && path[plen - 1u] == '/')
      --plen;
    p->parent = (char *)malloc(plen + 1u);
    if (p->parent != NULL) {
      memcpy(p->parent, path, plen);
      p->parent[plen] = '\0';
    }
  }
  if (p->parent == NULL) {
    free_path_parts(p);
    return -1;
  }
  return 0;
}

static int write_all_fd(int fd, const uint8_t *data, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, data + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int read_all_fd(int fd, uint8_t *dst, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, dst + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = ENODATA;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int fsync_fd_retry(int fd) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc < 0 && errno == EINTR);
  return rc;
}

static int fill_random_16(uint8_t out[16]) {
  size_t off = 0u;
  while (off < 16u) {
    const ssize_t n = getrandom(out + off, 16u - off, 0u);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static et_kvq_status atomic_write_0600(const char *path, const uint8_t *data,
                                       size_t len) {
  struct path_parts parts;
  if (split_path(path, &parts) != 0)
    return ET_KVQ_ERR_INVALID_ARG;

  int dir_fd;
  do {
    dir_fd = open(parts.parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  } while (dir_fd < 0 && errno == EINTR);
  if (dir_fd < 0) {
    free_path_parts(&parts);
    return ET_KVQ_ERR_IO;
  }

  /* Reject existing symlink or non-regular destination before writing. */
  struct stat dst_st;
  if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0) {
    if (!S_ISREG(dst_st.st_mode)) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_KVQ_ERR_IO;
    }
  } else if (errno != ENOENT) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_KVQ_ERR_IO;
  }

  int tmp_fd = -1;
  char tmp_name[48];
  static const char hex[] = "0123456789abcdef";
  for (int attempt = 0; attempt < 16; ++attempt) {
    uint8_t rnd[16];
    if (fill_random_16(rnd) != 0) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_KVQ_ERR_IO;
    }
    memcpy(tmp_name, ".etkq-", 6u);
    for (size_t i = 0u; i < 16u; ++i) {
      tmp_name[6u + i * 2u] = hex[rnd[i] >> 4u];
      tmp_name[7u + i * 2u] = hex[rnd[i] & 0x0fu];
    }
    memcpy(tmp_name + 38u, ".tmp", 5u);
    do {
      tmp_fd = openat(dir_fd, tmp_name,
                      O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                      (mode_t)0600);
    } while (tmp_fd < 0 && errno == EINTR);
    if (tmp_fd >= 0)
      break;
    if (errno != EEXIST) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_KVQ_ERR_IO;
    }
  }
  if (tmp_fd < 0) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_KVQ_ERR_IO;
  }

  et_kvq_status st = ET_KVQ_OK;
  if (fchmod(tmp_fd, (mode_t)0600) != 0 ||
      write_all_fd(tmp_fd, data, len) != 0 || fsync_fd_retry(tmp_fd) != 0) {
    st = ET_KVQ_ERR_IO;
  }
  if (close(tmp_fd) != 0 && st == ET_KVQ_OK)
    st = ET_KVQ_ERR_IO;

  if (st == ET_KVQ_OK) {
    if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0 &&
        !S_ISREG(dst_st.st_mode)) {
      st = ET_KVQ_ERR_IO;
    } else if (renameat(dir_fd, tmp_name, dir_fd, parts.base) != 0 ||
               fsync_fd_retry(dir_fd) != 0) {
      st = ET_KVQ_ERR_IO;
    }
  }
  if (st != ET_KVQ_OK)
    (void)unlinkat(dir_fd, tmp_name, 0);

  close(dir_fd);
  free_path_parts(&parts);
  return st;
}

et_kvq_status et_kvq_cache_save_snapshot(const et_kvq_cache *cache,
                                         const char *path) {
  if (cache == NULL || path == NULL || path[0] == '\0')
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;

  const uint32_t nl = cache->cfg.num_layers;
  const uint32_t hkv = cache->cfg.num_kv_heads;
  const uint32_t hd = cache->cfg.head_dim;

  uint64_t total_slices = 0u;
  uint32_t min_len = cache->layer_seq_len[0];
  for (uint32_t l = 0u; l < nl; ++l) {
    const uint32_t sl = cache->layer_seq_len[l];
    if (sl > cache->cfg.max_seq_len)
      return ET_KVQ_ERR_STATE;
    total_slices += (uint64_t)sl;
    if (sl < min_len)
      min_len = sl;
  }

  const uint64_t bpv =
      (uint64_t)bytes_per_vector_for_format(cache->cfg.format, hd);
  const uint64_t active_vecs = total_slices * 2u * (uint64_t)hkv;
  const uint64_t payload_bytes = (uint64_t)nl * 4u + active_vecs * bpv;
  const uint64_t total_bytes = (uint64_t)ET_KVQ_SNAPSHOT_HEADER_BYTES +
                               payload_bytes +
                               (uint64_t)ET_KVQ_SNAPSHOT_CHECKSUM_BYTES;
  if (total_bytes > (uint64_t)SIZE_MAX)
    return ET_KVQ_ERR_OVERFLOW;

  uint8_t *buf = (uint8_t *)calloc(1u, (size_t)total_bytes);
  if (buf == NULL)
    return ET_KVQ_ERR_ALLOC;

  /* 64-byte canonical little-endian header. */
  memcpy(buf + 0u, ET_KVQ_SNAPSHOT_MAGIC, ET_KVQ_SNAPSHOT_MAGIC_BYTES);
  put_u16_le(buf + 8u, (uint16_t)ET_KVQ_SNAPSHOT_VERSION_MAJOR);
  put_u16_le(buf + 10u, (uint16_t)ET_KVQ_SNAPSHOT_VERSION_MINOR);
  put_u32_le(buf + 12u, cache->cfg.format);
  put_u32_le(buf + 16u, nl);
  put_u32_le(buf + 20u, hkv);
  put_u32_le(buf + 24u, hd);
  put_u32_le(buf + 28u, cache->cfg.max_seq_len);
  put_u32_le(buf + 32u, min_len);
  put_u32_le(buf + 36u, 0u);
  put_u64_le(buf + 40u, payload_bytes);
  put_u64_le(buf + 48u, total_slices);
  put_u64_le(buf + 56u, 0u);

  size_t off = ET_KVQ_SNAPSHOT_HEADER_BYTES;
  for (uint32_t l = 0u; l < nl; ++l) {
    put_u32_le(buf + off, cache->layer_seq_len[l]);
    off += 4u;
  }

  for (uint32_t l = 0u; l < nl; ++l) {
    const uint32_t sl = cache->layer_seq_len[l];
    for (uint32_t side = 0u; side < 2u; ++side) {
      for (uint32_t pos = 0u; pos < sl; ++pos) {
        for (uint32_t h = 0u; h < hkv; ++h) {
          const size_t vidx = vector_index(cache, l, side, pos, h);
          const et_kvq_status vst = validate_stored_vector(cache, vidx);
          if (vst != ET_KVQ_OK) {
            free(buf);
            return vst;
          }
          if (cache->cfg.format == ET_KVQ_FMT_FP32) {
            const float *src = cache_fp32_cptr(cache, vidx);
            for (uint32_t d = 0u; d < hd; ++d) {
              put_u32_le(buf + off, f32_to_bits(load_f32_at(src, d)));
              off += 4u;
            }
          } else {
            const float sc = load_f32_at(cache_scale_cptr(cache, vidx), 0u);
            put_u32_le(buf + off, f32_to_bits(sc));
            off += 4u;
            memcpy(buf + off, cache_quant_cptr(cache, vidx), (size_t)hd);
            off += (size_t)hd;
          }
        }
      }
    }
  }

  uint8_t digest[ET_KVQ_SNAPSHOT_CHECKSUM_BYTES];
  sha256_bytes(buf, off, digest);
  memcpy(buf + off, digest, ET_KVQ_SNAPSHOT_CHECKSUM_BYTES);

  const et_kvq_status st = atomic_write_0600(path, buf, (size_t)total_bytes);
  free(buf);
  return st;
}

et_kvq_status et_kvq_cache_load_snapshot(et_kvq_cache *cache,
                                         const char *path) {
  if (cache == NULL || path == NULL || path[0] == '\0')
    return ET_KVQ_ERR_INVALID_ARG;
  if (!fp_environment_supported())
    return ET_KVQ_ERR_FP_ENV;

  int fd;
  do {
    fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0)
    return ET_KVQ_ERR_IO;

  struct stat st_buf;
  if (fstat(fd, &st_buf) != 0 || !S_ISREG(st_buf.st_mode)) {
    close(fd);
    return ET_KVQ_ERR_IO;
  }

  const uint64_t min_bytes = (uint64_t)ET_KVQ_SNAPSHOT_HEADER_BYTES +
                             (uint64_t)ET_KVQ_SNAPSHOT_CHECKSUM_BYTES;
  const uint64_t max_bytes = ET_KVQ_MAX_STORAGE_BYTES + UINT64_C(65536);
  if (st_buf.st_size < 0 || (uint64_t)st_buf.st_size < min_bytes) {
    close(fd);
    return ET_KVQ_ERR_CORRUPT;
  }
  if ((uint64_t)st_buf.st_size > max_bytes ||
      (uint64_t)st_buf.st_size > (uint64_t)SIZE_MAX) {
    close(fd);
    return ET_KVQ_ERR_OVERFLOW;
  }

  const size_t file_len = (size_t)st_buf.st_size;
  uint8_t *buf = (uint8_t *)malloc(file_len);
  if (buf == NULL) {
    close(fd);
    return ET_KVQ_ERR_ALLOC;
  }
  if (read_all_fd(fd, buf, file_len) != 0) {
    free(buf);
    close(fd);
    return ET_KVQ_ERR_IO;
  }
  uint8_t extra = 0u;
  const ssize_t tail_rc = read(fd, &extra, 1u);
  close(fd);
  if (tail_rc != 0) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  /* Validate magic, version, reserved fields, and exact framed length. */
  if (memcmp(buf, ET_KVQ_SNAPSHOT_MAGIC, ET_KVQ_SNAPSHOT_MAGIC_BYTES) != 0) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }
  const uint16_t ver_major = get_u16_le(buf + 8u);
  const uint16_t ver_minor = get_u16_le(buf + 10u);
  if (ver_major != (uint16_t)ET_KVQ_SNAPSHOT_VERSION_MAJOR ||
      ver_minor != (uint16_t)ET_KVQ_SNAPSHOT_VERSION_MINOR) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }
  const uint32_t hdr_fmt = get_u32_le(buf + 12u);
  const uint32_t hdr_nl = get_u32_le(buf + 16u);
  const uint32_t hdr_hkv = get_u32_le(buf + 20u);
  const uint32_t hdr_hd = get_u32_le(buf + 24u);
  const uint32_t hdr_max_seq = get_u32_le(buf + 28u);
  const uint32_t hdr_min_seq = get_u32_le(buf + 32u);
  const uint32_t hdr_res1 = get_u32_le(buf + 36u);
  const uint64_t hdr_payload_bytes = get_u64_le(buf + 40u);
  const uint64_t hdr_total_slices = get_u64_le(buf + 48u);
  const uint64_t hdr_res2 = get_u64_le(buf + 56u);

  if (hdr_res1 != 0u || hdr_res2 != 0u) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  uint64_t expected_file_len = 0u;
  if (!add_u64_checked((uint64_t)ET_KVQ_SNAPSHOT_HEADER_BYTES,
                       hdr_payload_bytes, &expected_file_len) ||
      !add_u64_checked(expected_file_len,
                       (uint64_t)ET_KVQ_SNAPSHOT_CHECKSUM_BYTES,
                       &expected_file_len) ||
      expected_file_len != (uint64_t)file_len) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  /* Verify SHA-256 digest over header + payload before accepting any metadata. */
  const size_t signed_len = file_len - ET_KVQ_SNAPSHOT_CHECKSUM_BYTES;
  uint8_t digest[ET_KVQ_SNAPSHOT_CHECKSUM_BYTES];
  sha256_bytes(buf, signed_len, digest);
  if (!ct_memeq_32(digest, buf + signed_len)) {
    free(buf);
    return ET_KVQ_ERR_CHECKSUM;
  }

  /* Check format and geometry compatibility against destination cache. */
  if (hdr_fmt != ET_KVQ_FMT_FP32 && hdr_fmt != ET_KVQ_FMT_INT8_SYM &&
      hdr_fmt != ET_KVQ_FMT_FP8_E4M3FN) {
    free(buf);
    return ET_KVQ_ERR_UNSUPPORTED_FORMAT;
  }
  if (hdr_fmt != cache->cfg.format) {
    free(buf);
    return ET_KVQ_ERR_FORMAT_MISMATCH;
  }
  if (hdr_nl != cache->cfg.num_layers || hdr_hkv != cache->cfg.num_kv_heads ||
      hdr_hd != cache->cfg.head_dim) {
    free(buf);
    return ET_KVQ_ERR_SHAPE;
  }
  if (hdr_max_seq < 1u || hdr_max_seq > ET_KVQ_MAX_SEQ_LEN) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  /* Pass 1: validate layer_seq_len array and EVERY payload value before mutating. */
  if (hdr_payload_bytes < (uint64_t)hdr_nl * 4u) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  uint32_t parsed_lens[ET_KVQ_MAX_LAYERS];
  memset(parsed_lens, 0, sizeof(parsed_lens));
  size_t off = ET_KVQ_SNAPSHOT_HEADER_BYTES;
  uint64_t sum_slices = 0u;
  uint32_t min_len = UINT32_MAX;
  for (uint32_t l = 0u; l < hdr_nl; ++l) {
    const uint32_t sl = get_u32_le(buf + off);
    off += 4u;
    if (sl > hdr_max_seq) {
      free(buf);
      return ET_KVQ_ERR_CORRUPT;
    }
    if (sl > cache->cfg.max_seq_len) {
      free(buf);
      return ET_KVQ_ERR_CAPACITY;
    }
    parsed_lens[l] = sl;
    sum_slices += (uint64_t)sl;
    if (sl < min_len)
      min_len = sl;
  }
  if (sum_slices != hdr_total_slices || min_len != hdr_min_seq) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  const uint64_t bpv =
      (uint64_t)bytes_per_vector_for_format(hdr_fmt, hdr_hd);
  uint64_t expected_payload = 0u;
  uint64_t vec_bytes = 0u;
  if (!mul_u64_checked(sum_slices * 2u * (uint64_t)hdr_hkv, bpv, &vec_bytes) ||
      !add_u64_checked((uint64_t)hdr_nl * 4u, vec_bytes, &expected_payload) ||
      expected_payload != hdr_payload_bytes) {
    free(buf);
    return ET_KVQ_ERR_CORRUPT;
  }

  size_t scan = off;
  for (uint32_t l = 0u; l < hdr_nl; ++l) {
    const uint32_t sl = parsed_lens[l];
    for (uint32_t side = 0u; side < 2u; ++side) {
      for (uint32_t pos = 0u; pos < sl; ++pos) {
        for (uint32_t h = 0u; h < hdr_hkv; ++h) {
          if (hdr_fmt == ET_KVQ_FMT_FP32) {
            for (uint32_t d = 0u; d < hdr_hd; ++d) {
              const float fv = bits_to_f32(get_u32_le(buf + scan));
              scan += 4u;
              if (!f32_is_finite(fv)) {
                free(buf);
                return ET_KVQ_ERR_NONFINITE;
              }
            }
          } else if (hdr_fmt == ET_KVQ_FMT_INT8_SYM) {
            const float sc = bits_to_f32(get_u32_le(buf + scan));
            scan += 4u;
            if (!f32_is_finite(sc) || sc < 0.0f) {
              free(buf);
              return ET_KVQ_ERR_NONFINITE;
            }
            const double dsc = (double)sc;
            for (uint32_t d = 0u; d < hdr_hd; ++d) {
              const int8_t qv = (int8_t)buf[scan++];
              if (qv == (int8_t)-128 || (sc == 0.0f && qv != 0)) {
                free(buf);
                return ET_KVQ_ERR_CORRUPT;
              }
              if (!f32_is_finite(dequant_int8_elem_f32(qv, dsc))) {
                free(buf);
                return ET_KVQ_ERR_NONFINITE;
              }
            }
          } else {
            const float sc = bits_to_f32(get_u32_le(buf + scan));
            scan += 4u;
            if (!f32_is_finite(sc) || sc < 0.0f) {
              free(buf);
              return ET_KVQ_ERR_NONFINITE;
            }
            const double dsc = (double)sc;
            for (uint32_t d = 0u; d < hdr_hd; ++d) {
              const uint8_t code = buf[scan++];
              float fv = 0.0f;
              const et_kvq_status dst = dequant_fp8_elem_f32(code, dsc, &fv);
              if (dst != ET_KVQ_OK) {
                free(buf);
                return dst;
              }
            }
          }
        }
      }
    }
  }

  /* Pass 2: all validation passed; commit atomically into destination cache. */
  memset(cache->storage, 0, cache->storage_bytes);
  memset(cache->layer_seq_len, 0, sizeof(cache->layer_seq_len));

  size_t cur = off;
  for (uint32_t l = 0u; l < hdr_nl; ++l) {
    const uint32_t sl = parsed_lens[l];
    for (uint32_t side = 0u; side < 2u; ++side) {
      for (uint32_t pos = 0u; pos < sl; ++pos) {
        for (uint32_t h = 0u; h < hdr_hkv; ++h) {
          const size_t vidx = vector_index(cache, l, side, pos, h);
          if (hdr_fmt == ET_KVQ_FMT_FP32) {
            float *dst = cache_fp32_ptr(cache, vidx);
            for (uint32_t d = 0u; d < hdr_hd; ++d) {
              store_f32_at(dst, d, bits_to_f32(get_u32_le(buf + cur)));
              cur += 4u;
            }
          } else {
            const float sc = bits_to_f32(get_u32_le(buf + cur));
            cur += 4u;
            store_f32_at(cache_scale_ptr(cache, vidx), 0u, sc);
            memcpy(cache_quant_ptr(cache, vidx), buf + cur, (size_t)hdr_hd);
            cur += (size_t)hdr_hd;
          }
        }
      }
    }
    cache->layer_seq_len[l] = sl;
  }

  free(buf);
  return ET_KVQ_OK;
}
