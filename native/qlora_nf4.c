#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/qlora_nf4_abi.h"

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

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "QLoRA NF4 engine requires 8-bit bytes and binary32 floats");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "QLoRA NF4 engine requires IEEE-754 binary32 evaluation");

/* Exact Dettmers et al. (2023) 16-entry NormalFloat4 (NF4) quantile codebook. */
const float ET_NF4_CODEBOOK[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f};

/* ---- Standalone SHA-256 (zero heap allocations) ---- */

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
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0u; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) |
           ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = state->h[0];
  b = state->h[1];
  c = state->h[2];
  d = state->h[3];
  e = state->h[4];
  f = state->h[5];
  g = state->h[6];
  h = state->h[7];
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
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
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

/* ---- Little-endian serialization helpers ---- */

static void store_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static uint32_t load_u32_le(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static void store_u64_le(uint8_t *dst, uint64_t v) {
  for (unsigned i = 0u; i < 8u; ++i)
    dst[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
}

static uint64_t load_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (unsigned i = 0u; i < 8u; ++i)
    v |= ((uint64_t)src[i]) << (8u * i);
  return v;
}

static uint32_t f32_to_bits(float x) {
  uint32_t bits = 0u;
  memcpy(&bits, &x, sizeof(bits));
  return bits;
}

static float bits_to_f32(uint32_t bits) {
  float x = 0.0f;
  memcpy(&x, &bits, sizeof(x));
  return x;
}

static void store_f32_le(uint8_t *dst, float x) {
  store_u32_le(dst, f32_to_bits(x));
}

static float load_f32_le(const uint8_t *src) {
  return bits_to_f32(load_u32_le(src));
}

static int f32_is_finite(float x) {
  return (f32_to_bits(x) & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int f64_is_finite(double x) {
  uint64_t bits = 0u;
  memcpy(&bits, &x, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

/* ---- FP environment, pointer alignment & disjointness checks ---- */

static et_qlora_status check_fp_env(void) {
  if (fegetround() != FE_TONEAREST)
    return ET_QLORA_ERR_FP_ENV;
#if defined(__x86_64__)
  uint16_t cw = 0u;
  __asm__ __volatile__("fnstcw %0" : "=m"(cw));
  if ((cw & 0x0c3fu) != 0x003fu)
    return ET_QLORA_ERR_FP_ENV;
  const unsigned int mxcsr = _mm_getcsr();
  if ((mxcsr & 0xffc0u) != 0x1f80u)
    return ET_QLORA_ERR_FP_ENV;
#endif
  return ET_QLORA_OK;
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

static et_qlora_status check_f32_buffer(const void *ptr, size_t count) {
  if (ptr == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (count > SIZE_MAX / sizeof(float))
    return ET_QLORA_ERR_OVERFLOW;
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(ptr, _Alignof(float), bytes))
    return ET_QLORA_ERR_ALIGNMENT;
  return ET_QLORA_OK;
}

static et_qlora_status check_byte_buffer(const void *ptr, size_t bytes) {
  if (ptr == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (!aligned_span(ptr, 1u, bytes))
    return ET_QLORA_ERR_ALIGNMENT;
  return ET_QLORA_OK;
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

static et_qlora_status check_finite_f32_array(const float *arr, size_t count,
                                              double *out_max_abs) {
  double max_a = 0.0;
  for (size_t i = 0u; i < count; ++i) {
    const float v = arr[i];
    if (!f32_is_finite(v))
      return ET_QLORA_ERR_NONFINITE;
    const double av = fabs((double)v);
    if (av > max_a)
      max_a = av;
  }
  if (out_max_abs != NULL)
    *out_max_abs = max_a;
  return ET_QLORA_OK;
}

/* ---- Config & footprint validation ---- */

static int valid_block_size(uint32_t bs) {
  return bs == ET_QLORA_BLOCK_SIZE_32 || bs == ET_QLORA_BLOCK_SIZE_64 ||
         bs == ET_QLORA_BLOCK_SIZE_128;
}

static int valid_dq_block_size(uint32_t dqs) {
  return dqs == ET_QLORA_DQ_BLOCK_SIZE_16 || dqs == ET_QLORA_DQ_BLOCK_SIZE_32 ||
         dqs == ET_QLORA_DQ_BLOCK_SIZE_64 || dqs == ET_QLORA_DQ_BLOCK_SIZE_256;
}

et_qlora_status et_qlora_validate_quant_spec(const et_qlora_quant_spec *spec) {
  if (spec == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  const et_qlora_status fp = check_fp_env();
  if (fp != ET_QLORA_OK)
    return fp;
  if (spec->reserved != 0u)
    return ET_QLORA_ERR_ARGUMENT;
  if (spec->rows_out == 0u || spec->rows_out > ET_QLORA_MAX_OUT)
    return ET_QLORA_ERR_SHAPE;
  if (!valid_block_size(spec->block_size))
    return ET_QLORA_ERR_SHAPE;
  if (spec->cols_in == 0u || spec->cols_in > ET_QLORA_MAX_IN ||
      (spec->cols_in % spec->block_size) != 0u)
    return ET_QLORA_ERR_SHAPE;
  if (spec->mode == ET_QLORA_QUANT_NF4_SQ) {
    if (spec->dq_block_size != 0u)
      return ET_QLORA_ERR_SHAPE;
  } else if (spec->mode == ET_QLORA_QUANT_NF4_DQ) {
    if (!valid_dq_block_size(spec->dq_block_size))
      return ET_QLORA_ERR_SHAPE;
  } else {
    return ET_QLORA_ERR_FORMAT;
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_validate_linear_config(
    const et_qlora_linear_config *cfg) {
  if (cfg == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  const et_qlora_status qst = et_qlora_validate_quant_spec(&cfg->quant);
  if (qst != ET_QLORA_OK)
    return qst;
  if (cfg->reserved != 0u)
    return ET_QLORA_ERR_ARGUMENT;
  if (cfg->m == 0u || cfg->m > ET_QLORA_MAX_M)
    return ET_QLORA_ERR_SHAPE;
  const uint32_t min_dim = (cfg->quant.cols_in < cfg->quant.rows_out)
                               ? cfg->quant.cols_in
                               : cfg->quant.rows_out;
  if (cfg->r == 0u || cfg->r > ET_QLORA_MAX_R || cfg->r > min_dim)
    return ET_QLORA_ERR_SHAPE;
  if (!f32_is_finite(cfg->alpha) || cfg->alpha <= 0.0f)
    return ET_QLORA_ERR_NONFINITE;
  const double scale = (double)cfg->alpha / (double)cfg->r;
  if (!f64_is_finite(scale) || !f32_is_finite((float)scale))
    return ET_QLORA_ERR_NONFINITE;
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_validate_mlp_config(const et_qlora_mlp_config *cfg) {
  if (cfg == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  const et_qlora_status fp = check_fp_env();
  if (fp != ET_QLORA_OK)
    return fp;
  if (cfg->reserved != 0u)
    return ET_QLORA_ERR_ARGUMENT;
  if (cfg->m == 0u || cfg->m > ET_QLORA_MAX_M)
    return ET_QLORA_ERR_SHAPE;
  if (!valid_block_size(cfg->block_size))
    return ET_QLORA_ERR_SHAPE;
  if (cfg->d_in == 0u || cfg->d_in > ET_QLORA_MAX_IN ||
      (cfg->d_in % cfg->block_size) != 0u)
    return ET_QLORA_ERR_SHAPE;
  if (cfg->d_hidden == 0u || cfg->d_hidden > ET_QLORA_MAX_OUT ||
      cfg->d_hidden > ET_QLORA_MAX_IN ||
      (cfg->d_hidden % cfg->block_size) != 0u)
    return ET_QLORA_ERR_SHAPE;
  if (cfg->d_out == 0u || cfg->d_out > ET_QLORA_MAX_OUT)
    return ET_QLORA_ERR_SHAPE;
  uint32_t min_dim = cfg->d_in;
  if (cfg->d_hidden < min_dim)
    min_dim = cfg->d_hidden;
  if (cfg->d_out < min_dim)
    min_dim = cfg->d_out;
  if (cfg->r == 0u || cfg->r > ET_QLORA_MAX_R || cfg->r > min_dim)
    return ET_QLORA_ERR_SHAPE;
  if (cfg->quant_mode == ET_QLORA_QUANT_NF4_SQ) {
    if (cfg->dq_block_size != 0u)
      return ET_QLORA_ERR_SHAPE;
  } else if (cfg->quant_mode == ET_QLORA_QUANT_NF4_DQ) {
    if (!valid_dq_block_size(cfg->dq_block_size))
      return ET_QLORA_ERR_SHAPE;
  } else {
    return ET_QLORA_ERR_FORMAT;
  }
  if (!f32_is_finite(cfg->alpha) || cfg->alpha <= 0.0f)
    return ET_QLORA_ERR_NONFINITE;
  const double scale = (double)cfg->alpha / (double)cfg->r;
  if (!f64_is_finite(scale) || !f32_is_finite((float)scale))
    return ET_QLORA_ERR_NONFINITE;
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_validate_adamw_config(
    const et_qlora_adamw_config *opt) {
  if (opt == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  const et_qlora_status fp = check_fp_env();
  if (fp != ET_QLORA_OK)
    return fp;
  if (opt->reserved != 0u)
    return ET_QLORA_ERR_ARGUMENT;
  if (!f32_is_finite(opt->lr) || opt->lr <= 0.0f)
    return ET_QLORA_ERR_NONFINITE;
  if (!f32_is_finite(opt->beta1) || opt->beta1 < 0.0f || opt->beta1 >= 1.0f)
    return ET_QLORA_ERR_NONFINITE;
  if (!f32_is_finite(opt->beta2) || opt->beta2 < 0.0f || opt->beta2 >= 1.0f)
    return ET_QLORA_ERR_NONFINITE;
  if (!f32_is_finite(opt->eps) || opt->eps <= 0.0f)
    return ET_QLORA_ERR_NONFINITE;
  if (!f32_is_finite(opt->weight_decay) || opt->weight_decay < 0.0f)
    return ET_QLORA_ERR_NONFINITE;
  if (opt->step == UINT64_MAX)
    return ET_QLORA_ERR_OVERFLOW;
  return ET_QLORA_OK;
}

static void compute_quant_layout(const et_qlora_quant_spec *spec,
                                 size_t *out_num_params,
                                 size_t *out_num_blocks,
                                 size_t *out_num_dq_blocks,
                                 size_t *out_nibble_bytes,
                                 size_t *out_scale_bytes,
                                 size_t *out_packed_bytes) {
  const size_t num_params = (size_t)spec->rows_out * (size_t)spec->cols_in;
  const size_t num_blocks = num_params / (size_t)spec->block_size;
  const size_t nibble_bytes = num_params / 2u;
  size_t num_dq_blocks = 0u;
  size_t scale_bytes = 0u;
  if (spec->mode == ET_QLORA_QUANT_NF4_SQ) {
    scale_bytes = num_blocks * 4u;
  } else {
    num_dq_blocks =
        (num_blocks + (size_t)spec->dq_block_size - 1u) /
        (size_t)spec->dq_block_size;
    scale_bytes = num_blocks + num_dq_blocks * 8u;
  }
  if (out_num_params != NULL)
    *out_num_params = num_params;
  if (out_num_blocks != NULL)
    *out_num_blocks = num_blocks;
  if (out_num_dq_blocks != NULL)
    *out_num_dq_blocks = num_dq_blocks;
  if (out_nibble_bytes != NULL)
    *out_nibble_bytes = nibble_bytes;
  if (out_scale_bytes != NULL)
    *out_scale_bytes = scale_bytes;
  if (out_packed_bytes != NULL)
    *out_packed_bytes = nibble_bytes + scale_bytes;
}

et_qlora_status et_qlora_packed_footprint(
    const et_qlora_quant_spec *spec, et_qlora_quant_footprint *out_footprint) {
  const et_qlora_status st = et_qlora_validate_quant_spec(spec);
  if (st != ET_QLORA_OK)
    return st;
  if (out_footprint == NULL)
    return ET_QLORA_ERR_ARGUMENT;

  size_t num_params = 0u, nibble_bytes = 0u, scale_bytes = 0u,
         packed_bytes = 0u;
  compute_quant_layout(spec, &num_params, NULL, NULL, &nibble_bytes,
                       &scale_bytes, &packed_bytes);
  out_footprint->f32_bytes = (uint64_t)num_params * UINT64_C(4);
  out_footprint->packed_bytes = (uint64_t)packed_bytes;
  out_footprint->nibble_bytes = (uint64_t)nibble_bytes;
  out_footprint->scale_bytes = (uint64_t)scale_bytes;
  out_footprint->bits_per_param =
      ((double)packed_bytes * 8.0) / (double)num_params;
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_sha256_bytes(
    const uint8_t *data, size_t size,
    uint8_t out_digest[ET_QLORA_SHA256_BYTES]) {
  if (out_digest == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (size > 0u && data == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (size > 0u && !aligned_span(data, 1u, size))
    return ET_QLORA_ERR_ALIGNMENT;
  if (!aligned_span(out_digest, 1u, ET_QLORA_SHA256_BYTES))
    return ET_QLORA_ERR_ALIGNMENT;

  sha256_state state;
  sha256_init(&state);
  if (size > 0u)
    sha256_update(&state, data, size);
  sha256_final(&state, out_digest);
  return ET_QLORA_OK;
}

/* ---- NF4 Quantization & Dequantization (zero heap allocations) ---- */

static uint8_t nf4_quantize_normalized(double v) {
  uint8_t q = 0u;
  for (uint8_t i = 0u; i < 15u; ++i) {
    const double mid =
        0.5 * ((double)ET_NF4_CODEBOOK[i] + (double)ET_NF4_CODEBOOK[i + 1u]);
    if (v >= mid) {
      q = (uint8_t)(i + 1u);
    } else {
      break;
    }
  }
  return q;
}

static uint8_t round_clamp_u8(double scaled) {
  if (scaled <= 0.0)
    return 0u;
  if (scaled >= 255.0)
    return 255u;
  const double r = floor(scaled + 0.5);
  if (r <= 0.0)
    return 0u;
  if (r >= 255.0)
    return 255u;
  return (uint8_t)r;
}

static float nf4_block_absmax_unchecked(const et_qlora_quant_spec *spec,
                                        const uint8_t *packed,
                                        size_t nibble_bytes, size_t num_blocks,
                                        size_t block_idx) {
  if (spec->mode == ET_QLORA_QUANT_NF4_SQ) {
    return load_f32_le(packed + nibble_bytes + block_idx * 4u);
  }
  const uint8_t q_s = packed[nibble_bytes + block_idx];
  const size_t dq_idx = block_idx / (size_t)spec->dq_block_size;
  const uint8_t *dq_meta =
      packed + nibble_bytes + num_blocks + dq_idx * 8u;
  const float offset_f32 = load_f32_le(dq_meta);
  const float scale_f32 = load_f32_le(dq_meta + 4u);
  return (float)((double)offset_f32 + (double)scale_f32 * (double)q_s);
}

static et_qlora_status validate_packed_nf4(const et_qlora_quant_spec *spec,
                                           const uint8_t *packed,
                                           size_t packed_bytes,
                                           double *out_max_absmax) {
  size_t num_blocks = 0u, num_dq_blocks = 0u, nibble_bytes = 0u,
         expected_bytes = 0u;
  compute_quant_layout(spec, NULL, &num_blocks, &num_dq_blocks, &nibble_bytes,
                       NULL, &expected_bytes);
  if (packed_bytes != expected_bytes)
    return ET_QLORA_ERR_CAPACITY;
  const et_qlora_status bst = check_byte_buffer(packed, packed_bytes);
  if (bst != ET_QLORA_OK)
    return bst;

  double max_s = 0.0;
  if (spec->mode == ET_QLORA_QUANT_NF4_SQ) {
    for (size_t b = 0u; b < num_blocks; ++b) {
      const float s = load_f32_le(packed + nibble_bytes + b * 4u);
      if (!f32_is_finite(s) || s < 0.0f)
        return ET_QLORA_ERR_NONFINITE;
      if ((double)s > max_s)
        max_s = (double)s;
    }
  } else {
    const uint8_t *dq_base = packed + nibble_bytes + num_blocks;
    for (size_t g = 0u; g < num_dq_blocks; ++g) {
      const float offset_f32 = load_f32_le(dq_base + g * 8u);
      const float scale_f32 = load_f32_le(dq_base + g * 8u + 4u);
      if (!f32_is_finite(offset_f32) || offset_f32 < 0.0f ||
          !f32_is_finite(scale_f32) || scale_f32 < 0.0f)
        return ET_QLORA_ERR_NONFINITE;
      const double hi_d = (double)offset_f32 + (double)scale_f32 * 255.0;
      if (!f64_is_finite(hi_d) || !f32_is_finite((float)hi_d))
        return ET_QLORA_ERR_NONFINITE;
      if (hi_d > max_s)
        max_s = hi_d;
    }
  }
  if (out_max_absmax != NULL)
    *out_max_absmax = max_s;
  return ET_QLORA_OK;
}

static float quantize_one_nf4_block(const float *src, uint32_t block_size,
                                    uint8_t *dst_nibbles) {
  float absmax = 0.0f;
  for (uint32_t j = 0u; j < block_size; ++j) {
    const float a = fabsf(src[j]);
    if (a > absmax)
      absmax = a;
  }
  const uint32_t half = block_size / 2u;
  if (absmax == 0.0f) {
    /* NF4 index 7 is exact 0.0f -> 0x77 packs two zeros. */
    memset(dst_nibbles, 0x77, half);
    return 0.0f;
  }
  const double inv_absmax_denom = (double)absmax;
  for (uint32_t p = 0u; p < half; ++p) {
    const double v0 = (double)src[2u * p] / inv_absmax_denom;
    const double v1 = (double)src[2u * p + 1u] / inv_absmax_denom;
    const uint8_t q0 = nf4_quantize_normalized(v0);
    const uint8_t q1 = nf4_quantize_normalized(v1);
    dst_nibbles[p] = (uint8_t)((q0 & 0x0Fu) | ((q1 & 0x0Fu) << 4u));
  }
  return absmax;
}

et_qlora_status et_qlora_nf4_quantize(const et_qlora_quant_spec *spec,
                                      const float *weights_f32,
                                      uint8_t *out_packed,
                                      size_t out_packed_bytes) {
  const et_qlora_status st = et_qlora_validate_quant_spec(spec);
  if (st != ET_QLORA_OK)
    return st;

  size_t num_params = 0u, num_blocks = 0u, num_dq_blocks = 0u,
         nibble_bytes = 0u, expected_bytes = 0u;
  compute_quant_layout(spec, &num_params, &num_blocks, &num_dq_blocks,
                       &nibble_bytes, NULL, &expected_bytes);
  if (out_packed_bytes != expected_bytes)
    return ET_QLORA_ERR_CAPACITY;

  const et_qlora_status wst = check_f32_buffer(weights_f32, num_params);
  if (wst != ET_QLORA_OK)
    return wst;
  const et_qlora_status ost = check_byte_buffer(out_packed, out_packed_bytes);
  if (ost != ET_QLORA_OK)
    return ost;

  const span_entry spans[3] = {
      {spec, sizeof(*spec)},
      {weights_f32, num_params * sizeof(float)},
      {out_packed, out_packed_bytes}};
  if (any_spans_overlap(spans, 3u))
    return ET_QLORA_ERR_ALIAS;

  /* Pass 1: validate all input weights are finite before writing anything. */
  const et_qlora_status fst =
      check_finite_f32_array(weights_f32, num_params, NULL);
  if (fst != ET_QLORA_OK)
    return fst;

  /* Pass 2: quantize blocks and scales (zero heap allocations). */
  const size_t bs = (size_t)spec->block_size;
  const size_t half = bs / 2u;
  if (spec->mode == ET_QLORA_QUANT_NF4_SQ) {
    uint8_t *scale_dst = out_packed + nibble_bytes;
    for (size_t b = 0u; b < num_blocks; ++b) {
      const float absmax = quantize_one_nf4_block(
          weights_f32 + b * bs, spec->block_size, out_packed + b * half);
      store_f32_le(scale_dst + b * 4u, absmax);
    }
  } else {
    uint8_t *qs_dst = out_packed + nibble_bytes;
    uint8_t *dq_dst = out_packed + nibble_bytes + num_blocks;
    const size_t dqs = (size_t)spec->dq_block_size;
    float group_scales[ET_QLORA_DQ_BLOCK_SIZE_256];

    for (size_t g = 0u; g < num_dq_blocks; ++g) {
      const size_t b_start = g * dqs;
      size_t b_end = b_start + dqs;
      if (b_end > num_blocks)
        b_end = num_blocks;
      const size_t count = b_end - b_start;

      float s_min = FLT_MAX;
      float s_max = 0.0f;
      for (size_t j = 0u; j < count; ++j) {
        const size_t b = b_start + j;
        const float absmax = quantize_one_nf4_block(
            weights_f32 + b * bs, spec->block_size, out_packed + b * half);
        group_scales[j] = absmax;
        if (absmax < s_min)
          s_min = absmax;
        if (absmax > s_max)
          s_max = absmax;
      }

      const float offset_f32 = s_min;
      const double range = (double)s_max - (double)s_min;
      const float scale_f32 = (range > 0.0) ? (float)(range / 255.0) : 0.0f;
      store_f32_le(dq_dst + g * 8u, offset_f32);
      store_f32_le(dq_dst + g * 8u + 4u, scale_f32);

      if (scale_f32 == 0.0f) {
        memset(qs_dst + b_start, 0, count);
      } else {
        const double inv_den = (double)scale_f32;
        for (size_t j = 0u; j < count; ++j) {
          const double scaled =
              ((double)group_scales[j] - (double)offset_f32) / inv_den;
          qs_dst[b_start + j] = round_clamp_u8(scaled);
        }
      }
    }
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_nf4_dequantize(const et_qlora_quant_spec *spec,
                                        const uint8_t *packed,
                                        size_t packed_bytes,
                                        float *out_weights_f32) {
  const et_qlora_status st = et_qlora_validate_quant_spec(spec);
  if (st != ET_QLORA_OK)
    return st;

  size_t num_params = 0u, num_blocks = 0u, nibble_bytes = 0u;
  compute_quant_layout(spec, &num_params, &num_blocks, NULL, &nibble_bytes,
                       NULL, NULL);

  const et_qlora_status pst =
      validate_packed_nf4(spec, packed, packed_bytes, NULL);
  if (pst != ET_QLORA_OK)
    return pst;
  const et_qlora_status ost = check_f32_buffer(out_weights_f32, num_params);
  if (ost != ET_QLORA_OK)
    return ost;

  const span_entry spans[3] = {
      {spec, sizeof(*spec)},
      {packed, packed_bytes},
      {out_weights_f32, num_params * sizeof(float)}};
  if (any_spans_overlap(spans, 3u))
    return ET_QLORA_ERR_ALIAS;

  const size_t bs = (size_t)spec->block_size;
  const size_t half = bs / 2u;
  for (size_t b = 0u; b < num_blocks; ++b) {
    const double absmax = (double)nf4_block_absmax_unchecked(
        spec, packed, nibble_bytes, num_blocks, b);
    const uint8_t *src = packed + b * half;
    float *dst = out_weights_f32 + b * bs;
    for (size_t p = 0u; p < half; ++p) {
      const uint8_t byte = src[p];
      dst[2u * p] = (float)(absmax * (double)ET_NF4_CODEBOOK[byte & 0x0Fu]);
      dst[2u * p + 1u] =
          (float)(absmax * (double)ET_NF4_CODEBOOK[(byte >> 4u) & 0x0Fu]);
    }
  }
  return ET_QLORA_OK;
}

/* ---- Deterministic LoRA adapter initialization ---- */

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

et_qlora_status et_qlora_init_lora(uint32_t cols_in, uint32_t rows_out,
                                   uint32_t r, uint64_t seed, float *a,
                                   float *b) {
  const et_qlora_status fp = check_fp_env();
  if (fp != ET_QLORA_OK)
    return fp;
  if (cols_in == 0u || cols_in > ET_QLORA_MAX_IN || rows_out == 0u ||
      rows_out > ET_QLORA_MAX_OUT)
    return ET_QLORA_ERR_SHAPE;
  const uint32_t min_dim = (cols_in < rows_out) ? cols_in : rows_out;
  if (r == 0u || r > ET_QLORA_MAX_R || r > min_dim)
    return ET_QLORA_ERR_SHAPE;

  const size_t a_count = (size_t)r * (size_t)cols_in;
  const size_t b_count = (size_t)rows_out * (size_t)r;
  const et_qlora_status ast = check_f32_buffer(a, a_count);
  if (ast != ET_QLORA_OK)
    return ast;
  const et_qlora_status bst = check_f32_buffer(b, b_count);
  if (bst != ET_QLORA_OK)
    return bst;

  const span_entry spans[2] = {
      {a, a_count * sizeof(float)}, {b, b_count * sizeof(float)}};
  if (any_spans_overlap(spans, 2u))
    return ET_QLORA_ERR_ALIAS;

  uint64_t rng = seed ^ UINT64_C(0x51a7c29f4d8e3b11);
  const double bound = 1.0 / sqrt((double)cols_in);
  for (size_t i = 0u; i < a_count; ++i) {
    const uint64_t w = splitmix64(&rng);
    /* Map to [-bound, +bound], avoiding exact 0. */
    const double u = ((double)(w >> 11u) + 0.5) * (1.0 / 9007199254740992.0);
    a[i] = (float)((2.0 * u - 1.0) * bound);
  }
  for (size_t i = 0u; i < b_count; ++i)
    b[i] = 0.0f;
  return ET_QLORA_OK;
}

/* ---- Fused Frozen-NF4 + LoRA Linear Forward, Backward & Merge ---- */

et_qlora_status et_qlora_linear_forward(const et_qlora_linear_config *cfg,
                                        const float *x, const uint8_t *w_nf4,
                                        size_t w_nf4_bytes, const float *bias,
                                        const float *a, const float *b,
                                        float *y) {
  const et_qlora_status cst = et_qlora_validate_linear_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;
  if ((a == NULL) != (b == NULL))
    return ET_QLORA_ERR_ARGUMENT;

  const size_t m = (size_t)cfg->m;
  const size_t k = (size_t)cfg->quant.cols_in;
  const size_t n = (size_t)cfg->quant.rows_out;
  const size_t r = (size_t)cfg->r;
  const int has_lora = (a != NULL);

  double max_w = 0.0;
  const et_qlora_status pst =
      validate_packed_nf4(&cfg->quant, w_nf4, w_nf4_bytes, &max_w);
  if (pst != ET_QLORA_OK)
    return pst;

  et_qlora_status st = check_f32_buffer(x, m * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(y, m * n);
  if (st != ET_QLORA_OK)
    return st;
  if (bias != NULL) {
    st = check_f32_buffer(bias, n);
    if (st != ET_QLORA_OK)
      return st;
  }
  if (has_lora) {
    st = check_f32_buffer(a, r * k);
    if (st != ET_QLORA_OK)
      return st;
    st = check_f32_buffer(b, n * r);
    if (st != ET_QLORA_OK)
      return st;
  }

  const span_entry spans[7] = {
      {cfg, sizeof(*cfg)},
      {x, m * k * sizeof(float)},
      {w_nf4, w_nf4_bytes},
      {bias, bias != NULL ? n * sizeof(float) : 0u},
      {a, has_lora ? r * k * sizeof(float) : 0u},
      {b, has_lora ? n * r * sizeof(float) : 0u},
      {y, m * n * sizeof(float)}};
  if (any_spans_overlap(spans, 7u))
    return ET_QLORA_ERR_ALIAS;

  double max_x = 0.0, max_bias = 0.0, max_a = 0.0, max_b = 0.0;
  st = check_finite_f32_array(x, m * k, &max_x);
  if (st != ET_QLORA_OK)
    return st;
  if (bias != NULL) {
    st = check_finite_f32_array(bias, n, &max_bias);
    if (st != ET_QLORA_OK)
      return st;
  }
  if (has_lora) {
    st = check_finite_f32_array(a, r * k, &max_a);
    if (st != ET_QLORA_OK)
      return st;
    st = check_finite_f32_array(b, n * r, &max_b);
    if (st != ET_QLORA_OK)
      return st;
  }

  /* Compute H = X @ A^T in binary64 (bounded <= 32 KiB stack). */
  double h_buf[ET_QLORA_MAX_M * ET_QLORA_MAX_R];
  double max_h = 0.0;
  const double scale = (double)cfg->alpha / (double)cfg->r;
  if (has_lora) {
    for (size_t mi = 0u; mi < m; ++mi) {
      const float *x_row = x + mi * k;
      for (size_t ri = 0u; ri < r; ++ri) {
        const float *a_row = a + ri * k;
        double acc = 0.0;
        for (size_t ki = 0u; ki < k; ++ki)
          acc += (double)x_row[ki] * (double)a_row[ki];
        if (!f64_is_finite(acc))
          return ET_QLORA_ERR_NONFINITE;
        h_buf[mi * r + ri] = acc;
        if (fabs(acc) > max_h)
          max_h = fabs(acc);
      }
    }
  }

  /* Conservative bound check: skip dry-run pass when overflow is impossible. */
  const double upper_bound =
      (double)k * max_x * max_w + max_bias +
      (has_lora ? scale * (double)r * max_h * max_b : 0.0);
  const int need_dry_run =
      (!f64_is_finite(upper_bound) || upper_bound >= 0.5 * (double)FLT_MAX);

  size_t num_blocks = 0u, nibble_bytes = 0u;
  compute_quant_layout(&cfg->quant, NULL, &num_blocks, NULL, &nibble_bytes,
                       NULL, NULL);
  const size_t bs = (size_t)cfg->quant.block_size;
  const size_t half = bs / 2u;
  const size_t blocks_per_row = k / bs;

  for (int pass = need_dry_run ? 0 : 1; pass < 2; ++pass) {
    for (size_t mi = 0u; mi < m; ++mi) {
      const float *x_row = x + mi * k;
      const double *h_row = h_buf + mi * r;
      for (size_t ni = 0u; ni < n; ++ni) {
        double acc = 0.0;
        const size_t row_block_base = ni * blocks_per_row;
        for (size_t bi = 0u; bi < blocks_per_row; ++bi) {
          const size_t blk = row_block_base + bi;
          const double absmax = (double)nf4_block_absmax_unchecked(
              &cfg->quant, w_nf4, nibble_bytes, num_blocks, blk);
          const uint8_t *qsrc = w_nf4 + blk * half;
          const float *xblk = x_row + bi * bs;
          for (size_t p = 0u; p < half; ++p) {
            const uint8_t byte = qsrc[p];
            const float w0 =
                (float)(absmax * (double)ET_NF4_CODEBOOK[byte & 0x0Fu]);
            const float w1 = (float)(absmax *
                                     (double)ET_NF4_CODEBOOK[(byte >> 4u) &
                                                             0x0Fu]);
            acc += (double)w0 * (double)xblk[2u * p];
            acc += (double)w1 * (double)xblk[2u * p + 1u];
          }
        }
        if (bias != NULL)
          acc += (double)bias[ni];
        if (has_lora) {
          const float *b_row = b + ni * r;
          double lora_acc = 0.0;
          for (size_t ri = 0u; ri < r; ++ri)
            lora_acc += h_row[ri] * (double)b_row[ri];
          acc += scale * lora_acc;
        }
        const float yf = (float)acc;
        if (!f64_is_finite(acc) || !f32_is_finite(yf))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          y[mi * n + ni] = yf;
      }
    }
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_linear_backward(const et_qlora_linear_config *cfg,
                                         const float *x, const uint8_t *w_nf4,
                                         size_t w_nf4_bytes, const float *a,
                                         const float *b, const float *dy,
                                         float *dx, float *da, float *db,
                                         float *dbias) {
  const et_qlora_status cst = et_qlora_validate_linear_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;

  const size_t m = (size_t)cfg->m;
  const size_t k = (size_t)cfg->quant.cols_in;
  const size_t n = (size_t)cfg->quant.rows_out;
  const size_t r = (size_t)cfg->r;

  double max_w = 0.0;
  et_qlora_status st =
      validate_packed_nf4(&cfg->quant, w_nf4, w_nf4_bytes, &max_w);
  if (st != ET_QLORA_OK)
    return st;

  st = check_f32_buffer(x, m * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(a, r * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(b, n * r);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(dy, m * n);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(dx, m * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(da, r * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(db, n * r);
  if (st != ET_QLORA_OK)
    return st;
  if (dbias != NULL) {
    st = check_f32_buffer(dbias, n);
    if (st != ET_QLORA_OK)
      return st;
  }

  const span_entry spans[10] = {
      {cfg, sizeof(*cfg)},
      {x, m * k * sizeof(float)},
      {w_nf4, w_nf4_bytes},
      {a, r * k * sizeof(float)},
      {b, n * r * sizeof(float)},
      {dy, m * n * sizeof(float)},
      {dx, m * k * sizeof(float)},
      {da, r * k * sizeof(float)},
      {db, n * r * sizeof(float)},
      {dbias, dbias != NULL ? n * sizeof(float) : 0u}};
  if (any_spans_overlap(spans, 10u))
    return ET_QLORA_ERR_ALIAS;

  double max_x = 0.0, max_a = 0.0, max_b = 0.0, max_dy = 0.0;
  st = check_finite_f32_array(x, m * k, &max_x);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(a, r * k, &max_a);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(b, n * r, &max_b);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(dy, m * n, &max_dy);
  if (st != ET_QLORA_OK)
    return st;

  const double scale = (double)cfg->alpha / (double)cfg->r;

  /* H = X @ A^T [m, r] and dH = scale * dY @ B [m, r] in binary64. */
  double h_buf[ET_QLORA_MAX_M * ET_QLORA_MAX_R];
  double dh_buf[ET_QLORA_MAX_M * ET_QLORA_MAX_R];
  double max_h = 0.0, max_dh = 0.0;

  for (size_t mi = 0u; mi < m; ++mi) {
    const float *x_row = x + mi * k;
    const float *dy_row = dy + mi * n;
    for (size_t ri = 0u; ri < r; ++ri) {
      const float *a_row = a + ri * k;
      double h_acc = 0.0;
      for (size_t ki = 0u; ki < k; ++ki)
        h_acc += (double)x_row[ki] * (double)a_row[ki];
      if (!f64_is_finite(h_acc))
        return ET_QLORA_ERR_NONFINITE;
      h_buf[mi * r + ri] = h_acc;
      if (fabs(h_acc) > max_h)
        max_h = fabs(h_acc);

      double dh_acc = 0.0;
      for (size_t ni = 0u; ni < n; ++ni)
        dh_acc += (double)dy_row[ni] * (double)b[ni * r + ri];
      dh_acc *= scale;
      if (!f64_is_finite(dh_acc))
        return ET_QLORA_ERR_NONFINITE;
      dh_buf[mi * r + ri] = dh_acc;
      if (fabs(dh_acc) > max_dh)
        max_dh = fabs(dh_acc);
    }
  }

  /* Conservative upper bounds for all outputs (dB, dA, dX, dbias). */
  const double b_db = scale * (double)m * max_dy * max_h;
  const double b_da = (double)m * max_dh * max_x;
  const double b_dx = (double)n * max_dy * max_w + (double)r * max_dh * max_a;
  const double b_dbias = (double)m * max_dy;
  double max_out_bound = b_db;
  if (b_da > max_out_bound)
    max_out_bound = b_da;
  if (b_dx > max_out_bound)
    max_out_bound = b_dx;
  if (b_dbias > max_out_bound)
    max_out_bound = b_dbias;
  const int need_dry_run =
      (!f64_is_finite(max_out_bound) || max_out_bound >= 0.5 * (double)FLT_MAX);

  size_t num_blocks = 0u, nibble_bytes = 0u;
  compute_quant_layout(&cfg->quant, NULL, &num_blocks, NULL, &nibble_bytes,
                       NULL, NULL);
  const size_t bs = (size_t)cfg->quant.block_size;
  const size_t half = bs / 2u;
  const size_t blocks_per_row = k / bs;

  double dx_row[ET_QLORA_MAX_IN];

  for (int pass = need_dry_run ? 0 : 1; pass < 2; ++pass) {
    /* 1. dB = scale * dY^T @ H  [n, r] */
    for (size_t ni = 0u; ni < n; ++ni) {
      for (size_t ri = 0u; ri < r; ++ri) {
        double acc = 0.0;
        for (size_t mi = 0u; mi < m; ++mi)
          acc += (double)dy[mi * n + ni] * h_buf[mi * r + ri];
        acc *= scale;
        const float v = (float)acc;
        if (!f64_is_finite(acc) || !f32_is_finite(v))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          db[ni * r + ri] = v;
      }
    }

    /* 2. dA = dH^T @ X  [r, k] */
    for (size_t ri = 0u; ri < r; ++ri) {
      for (size_t ki = 0u; ki < k; ++ki) {
        double acc = 0.0;
        for (size_t mi = 0u; mi < m; ++mi)
          acc += dh_buf[mi * r + ri] * (double)x[mi * k + ki];
        const float v = (float)acc;
        if (!f64_is_finite(acc) || !f32_is_finite(v))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          da[ri * k + ki] = v;
      }
    }

    /* 3. dX = dY @ dequant(W_nf4) + dH @ A  [m, k] */
    for (size_t mi = 0u; mi < m; ++mi) {
      for (size_t ki = 0u; ki < k; ++ki)
        dx_row[ki] = 0.0;

      const float *dy_row = dy + mi * n;
      for (size_t ni = 0u; ni < n; ++ni) {
        const double dy_mn = (double)dy_row[ni];
        const size_t row_block_base = ni * blocks_per_row;
        for (size_t bi = 0u; bi < blocks_per_row; ++bi) {
          const size_t blk = row_block_base + bi;
          const double absmax = (double)nf4_block_absmax_unchecked(
              &cfg->quant, w_nf4, nibble_bytes, num_blocks, blk);
          const uint8_t *qsrc = w_nf4 + blk * half;
          double *dxb = dx_row + bi * bs;
          for (size_t p = 0u; p < half; ++p) {
            const uint8_t byte = qsrc[p];
            const float w0 =
                (float)(absmax * (double)ET_NF4_CODEBOOK[byte & 0x0Fu]);
            const float w1 = (float)(absmax *
                                     (double)ET_NF4_CODEBOOK[(byte >> 4u) &
                                                             0x0Fu]);
            dxb[2u * p] += dy_mn * (double)w0;
            dxb[2u * p + 1u] += dy_mn * (double)w1;
          }
        }
      }

      const double *dh_row = dh_buf + mi * r;
      for (size_t ki = 0u; ki < k; ++ki) {
        double lora_g = 0.0;
        for (size_t ri = 0u; ri < r; ++ri)
          lora_g += dh_row[ri] * (double)a[ri * k + ki];
        const double total = dx_row[ki] + lora_g;
        const float v = (float)total;
        if (!f64_is_finite(total) || !f32_is_finite(v))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          dx[mi * k + ki] = v;
      }
    }

    /* 4. Optional dbias = sum_m dY[m, :]  [n] */
    if (dbias != NULL) {
      for (size_t ni = 0u; ni < n; ++ni) {
        double acc = 0.0;
        for (size_t mi = 0u; mi < m; ++mi)
          acc += (double)dy[mi * n + ni];
        const float v = (float)acc;
        if (!f64_is_finite(acc) || !f32_is_finite(v))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          dbias[ni] = v;
      }
    }
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_merge_to_f32(const et_qlora_linear_config *cfg,
                                      const uint8_t *w_nf4, size_t w_nf4_bytes,
                                      const float *a, const float *b,
                                      float *out_w_f32) {
  const et_qlora_status cst = et_qlora_validate_linear_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;

  const size_t k = (size_t)cfg->quant.cols_in;
  const size_t n = (size_t)cfg->quant.rows_out;
  const size_t r = (size_t)cfg->r;

  double max_w = 0.0;
  et_qlora_status st =
      validate_packed_nf4(&cfg->quant, w_nf4, w_nf4_bytes, &max_w);
  if (st != ET_QLORA_OK)
    return st;

  st = check_f32_buffer(a, r * k);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(b, n * r);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(out_w_f32, n * k);
  if (st != ET_QLORA_OK)
    return st;

  const span_entry spans[5] = {
      {cfg, sizeof(*cfg)},
      {w_nf4, w_nf4_bytes},
      {a, r * k * sizeof(float)},
      {b, n * r * sizeof(float)},
      {out_w_f32, n * k * sizeof(float)}};
  if (any_spans_overlap(spans, 5u))
    return ET_QLORA_ERR_ALIAS;

  double max_a = 0.0, max_b = 0.0;
  st = check_finite_f32_array(a, r * k, &max_a);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(b, n * r, &max_b);
  if (st != ET_QLORA_OK)
    return st;

  const double scale = (double)cfg->alpha / (double)cfg->r;
  const double bound = max_w + scale * (double)r * max_b * max_a;
  const int need_dry_run =
      (!f64_is_finite(bound) || bound >= 0.5 * (double)FLT_MAX);

  size_t num_blocks = 0u, nibble_bytes = 0u;
  compute_quant_layout(&cfg->quant, NULL, &num_blocks, NULL, &nibble_bytes,
                       NULL, NULL);
  const size_t bs = (size_t)cfg->quant.block_size;
  const size_t half = bs / 2u;
  const size_t blocks_per_row = k / bs;

  for (int pass = need_dry_run ? 0 : 1; pass < 2; ++pass) {
    for (size_t ni = 0u; ni < n; ++ni) {
      const float *b_row = b + ni * r;
      const size_t row_block_base = ni * blocks_per_row;
      for (size_t bi = 0u; bi < blocks_per_row; ++bi) {
        const size_t blk = row_block_base + bi;
        const double absmax = (double)nf4_block_absmax_unchecked(
            &cfg->quant, w_nf4, nibble_bytes, num_blocks, blk);
        const uint8_t *qsrc = w_nf4 + blk * half;
        for (size_t p = 0u; p < half; ++p) {
          const uint8_t byte = qsrc[p];
          const size_t col0 = bi * bs + 2u * p;
          const size_t col1 = col0 + 1u;
          const float w0 =
              (float)(absmax * (double)ET_NF4_CODEBOOK[byte & 0x0Fu]);
          const float w1 = (float)(absmax *
                                   (double)ET_NF4_CODEBOOK[(byte >> 4u) &
                                                           0x0Fu]);

          double d0 = 0.0, d1 = 0.0;
          for (size_t ri = 0u; ri < r; ++ri) {
            const double br = (double)b_row[ri];
            d0 += br * (double)a[ri * k + col0];
            d1 += br * (double)a[ri * k + col1];
          }
          const double m0 = (double)w0 + scale * d0;
          const double m1 = (double)w1 + scale * d1;
          const float f0 = (float)m0;
          const float f1 = (float)m1;
          if (!f64_is_finite(m0) || !f32_is_finite(f0) || !f64_is_finite(m1) ||
              !f32_is_finite(f1))
            return ET_QLORA_ERR_NONFINITE;
          if (pass == 1) {
            out_w_f32[ni * k + col0] = f0;
            out_w_f32[ni * k + col1] = f1;
          }
        }
      }
    }
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_f32_linear_forward(uint32_t m, uint32_t rows_out,
                                            uint32_t cols_in, const float *x,
                                            const float *w_f32,
                                            const float *bias, float *y) {
  const et_qlora_status fp = check_fp_env();
  if (fp != ET_QLORA_OK)
    return fp;
  if (m == 0u || m > ET_QLORA_MAX_M || rows_out == 0u ||
      rows_out > ET_QLORA_MAX_OUT || cols_in == 0u ||
      cols_in > ET_QLORA_MAX_IN)
    return ET_QLORA_ERR_SHAPE;

  const size_t mm = (size_t)m;
  const size_t nn = (size_t)rows_out;
  const size_t kk = (size_t)cols_in;

  et_qlora_status st = check_f32_buffer(x, mm * kk);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(w_f32, nn * kk);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(y, mm * nn);
  if (st != ET_QLORA_OK)
    return st;
  if (bias != NULL) {
    st = check_f32_buffer(bias, nn);
    if (st != ET_QLORA_OK)
      return st;
  }

  const span_entry spans[4] = {
      {x, mm * kk * sizeof(float)},
      {w_f32, nn * kk * sizeof(float)},
      {bias, bias != NULL ? nn * sizeof(float) : 0u},
      {y, mm * nn * sizeof(float)}};
  if (any_spans_overlap(spans, 4u))
    return ET_QLORA_ERR_ALIAS;

  double max_x = 0.0, max_w = 0.0, max_b = 0.0;
  st = check_finite_f32_array(x, mm * kk, &max_x);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(w_f32, nn * kk, &max_w);
  if (st != ET_QLORA_OK)
    return st;
  if (bias != NULL) {
    st = check_finite_f32_array(bias, nn, &max_b);
    if (st != ET_QLORA_OK)
      return st;
  }

  const double bound = (double)kk * max_x * max_w + max_b;
  const int need_dry_run =
      (!f64_is_finite(bound) || bound >= 0.5 * (double)FLT_MAX);

  for (int pass = need_dry_run ? 0 : 1; pass < 2; ++pass) {
    for (size_t mi = 0u; mi < mm; ++mi) {
      const float *x_row = x + mi * kk;
      for (size_t ni = 0u; ni < nn; ++ni) {
        const float *w_row = w_f32 + ni * kk;
        double acc = 0.0;
        for (size_t ki = 0u; ki < kk; ++ki)
          acc += (double)w_row[ki] * (double)x_row[ki];
        if (bias != NULL)
          acc += (double)bias[ni];
        const float yf = (float)acc;
        if (!f64_is_finite(acc) || !f32_is_finite(yf))
          return ET_QLORA_ERR_NONFINITE;
        if (pass == 1)
          y[mi * nn + ni] = yf;
      }
    }
  }
  return ET_QLORA_OK;
}

/* ---- Decoupled AdamW Optimizer (zero heap allocations) ---- */

static double pow_u64_exact(double base, uint64_t exp) {
  double result = 1.0;
  double b = base;
  uint64_t e = exp;
  while (e > 0u) {
    if ((e & 1u) != 0u)
      result *= b;
    e >>= 1u;
    if (e > 0u)
      b *= b;
  }
  return result;
}

static et_qlora_status adamw_step_internal(size_t count,
                                           const et_qlora_adamw_config *opt,
                                           float *params, const float *grads,
                                           float *m_state, float *v_state,
                                           int dry_run) {
  const double lr = (double)opt->lr;
  const double b1 = (double)opt->beta1;
  const double b2 = (double)opt->beta2;
  const double eps = (double)opt->eps;
  const double wd = (double)opt->weight_decay;
  const uint64_t t = opt->step + 1u;
  const double bc1 = 1.0 - pow_u64_exact(b1, t);
  const double bc2 = 1.0 - pow_u64_exact(b2, t);
  if (!f64_is_finite(bc1) || bc1 <= 0.0 || !f64_is_finite(bc2) || bc2 <= 0.0)
    return ET_QLORA_ERR_NONFINITE;
  const double decay_factor = 1.0 - lr * wd;

  for (size_t i = 0u; i < count; ++i) {
    const float pf = params[i];
    const float gf = grads[i];
    const float mf = m_state[i];
    const float vf = v_state[i];
    if (!f32_is_finite(pf) || !f32_is_finite(gf) || !f32_is_finite(mf) ||
        !f32_is_finite(vf) || vf < 0.0f)
      return ET_QLORA_ERR_NONFINITE;

    const double g = (double)gf;
    const double m_new = b1 * (double)mf + (1.0 - b1) * g;
    const double v_new = b2 * (double)vf + (1.0 - b2) * g * g;
    const double m_hat = m_new / bc1;
    const double v_hat = v_new / bc2;
    const double denom = sqrt(v_hat) + eps;
    const double p_new = (double)pf * decay_factor - lr * (m_hat / denom);

    const float m_out = (float)m_new;
    const float v_out = (float)v_new;
    const float p_out = (float)p_new;
    if (!f64_is_finite(m_new) || !f64_is_finite(v_new) ||
        !f64_is_finite(p_new) || !f32_is_finite(m_out) ||
        !f32_is_finite(v_out) || v_out < 0.0f || !f32_is_finite(p_out))
      return ET_QLORA_ERR_NONFINITE;
    if (!dry_run) {
      m_state[i] = m_out;
      v_state[i] = v_out;
      params[i] = p_out;
    }
  }
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_adamw_step(size_t count,
                                    const et_qlora_adamw_config *opt,
                                    float *params, const float *grads,
                                    float *m_state, float *v_state) {
  const et_qlora_status ost = et_qlora_validate_adamw_config(opt);
  if (ost != ET_QLORA_OK)
    return ost;
  if (count == 0u)
    return ET_QLORA_ERR_SHAPE;

  et_qlora_status st = check_f32_buffer(params, count);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(grads, count);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(m_state, count);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(v_state, count);
  if (st != ET_QLORA_OK)
    return st;

  const span_entry spans[5] = {
      {opt, sizeof(*opt)},
      {params, count * sizeof(float)},
      {grads, count * sizeof(float)},
      {m_state, count * sizeof(float)},
      {v_state, count * sizeof(float)}};
  if (any_spans_overlap(spans, 5u))
    return ET_QLORA_ERR_ALIAS;

  st = adamw_step_internal(count, opt, params, grads, m_state, v_state, 1);
  if (st != ET_QLORA_OK)
    return st;
  return adamw_step_internal(count, opt, params, grads, m_state, v_state, 0);
}

/* ---- 2-Layer QLoRA MLP/Projection Trainer ---- */

static void mlp_layer_configs(const et_qlora_mlp_config *cfg,
                              et_qlora_linear_config *l1,
                              et_qlora_linear_config *l2) {
  memset(l1, 0, sizeof(*l1));
  l1->quant.rows_out = cfg->d_hidden;
  l1->quant.cols_in = cfg->d_in;
  l1->quant.block_size = cfg->block_size;
  l1->quant.dq_block_size = cfg->dq_block_size;
  l1->quant.mode = cfg->quant_mode;
  l1->m = cfg->m;
  l1->r = cfg->r;
  l1->alpha = cfg->alpha;

  memset(l2, 0, sizeof(*l2));
  l2->quant.rows_out = cfg->d_out;
  l2->quant.cols_in = cfg->d_hidden;
  l2->quant.block_size = cfg->block_size;
  l2->quant.dq_block_size = cfg->dq_block_size;
  l2->quant.mode = cfg->quant_mode;
  l2->m = cfg->m;
  l2->r = cfg->r;
  l2->alpha = cfg->alpha;
}

et_qlora_status et_qlora_mlp_base_sha256(
    const et_qlora_mlp_config *cfg, const uint8_t *w1_nf4, size_t w1_bytes,
    const uint8_t *w2_nf4, size_t w2_bytes,
    uint8_t out_digest[ET_QLORA_SHA256_BYTES]) {
  const et_qlora_status cst = et_qlora_validate_mlp_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;
  if (out_digest == NULL)
    return ET_QLORA_ERR_ARGUMENT;

  et_qlora_linear_config l1, l2;
  mlp_layer_configs(cfg, &l1, &l2);
  et_qlora_status st = validate_packed_nf4(&l1.quant, w1_nf4, w1_bytes, NULL);
  if (st != ET_QLORA_OK)
    return st;
  st = validate_packed_nf4(&l2.quant, w2_nf4, w2_bytes, NULL);
  if (st != ET_QLORA_OK)
    return st;

  uint8_t len_hdr[16];
  store_u64_le(len_hdr, (uint64_t)w1_bytes);
  store_u64_le(len_hdr + 8u, (uint64_t)w2_bytes);

  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, len_hdr, sizeof(len_hdr));
  sha256_update(&sha, w1_nf4, w1_bytes);
  sha256_update(&sha, w2_nf4, w2_bytes);
  sha256_final(&sha, out_digest);
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_mlp_scratch_floats(const et_qlora_mlp_config *cfg,
                                            size_t *out_floats) {
  const et_qlora_status cst = et_qlora_validate_mlp_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;
  if (out_floats == NULL)
    return ET_QLORA_ERR_ARGUMENT;

  const size_t m = (size_t)cfg->m;
  const size_t din = (size_t)cfg->d_in;
  const size_t dh = (size_t)cfg->d_hidden;
  const size_t dout = (size_t)cfg->d_out;
  const size_t r = (size_t)cfg->r;

  *out_floats = (m * dh) * 4u + /* z1, h1, dh1, dz1 */
                (m * dout) * 2u + /* y, dy */
                (m * din) + /* dx */
                (r * din) + /* da1 */
                (dh * r) + /* db1 */
                (r * dh) + /* da2 */
                (dout * r); /* db2 */
  return ET_QLORA_OK;
}

static double silu_f64(double z) {
  if (z >= 0.0) {
    const double e = exp(-z);
    return z / (1.0 + e);
  }
  const double e = exp(z);
  return (z * e) / (1.0 + e);
}

static double silu_grad_f64(double z) {
  double s;
  if (z >= 0.0) {
    const double e = exp(-z);
    s = 1.0 / (1.0 + e);
  } else {
    const double e = exp(z);
    s = e / (1.0 + e);
  }
  return s * (1.0 + z * (1.0 - s));
}

et_qlora_status et_qlora_mlp_forward(
    const et_qlora_mlp_config *cfg, const float *x, const uint8_t *w1_nf4,
    size_t w1_bytes, const float *b1, const float *a1, const float *b1_lora,
    const uint8_t *w2_nf4, size_t w2_bytes, const float *b2, const float *a2,
    const float *b2_lora, float *out_z1, float *out_h1, float *out_y) {
  const et_qlora_status cst = et_qlora_validate_mlp_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;

  et_qlora_linear_config l1, l2;
  mlp_layer_configs(cfg, &l1, &l2);

  const size_t mh = (size_t)cfg->m * (size_t)cfg->d_hidden;
  et_qlora_status st = check_f32_buffer(out_z1, mh);
  if (st != ET_QLORA_OK)
    return st;
  st = check_f32_buffer(out_h1, mh);
  if (st != ET_QLORA_OK)
    return st;
  if (spans_overlap(out_z1, mh * sizeof(float), out_h1, mh * sizeof(float)))
    return ET_QLORA_ERR_ALIAS;

  st = et_qlora_linear_forward(&l1, x, w1_nf4, w1_bytes, b1, a1, b1_lora,
                               out_z1);
  if (st != ET_QLORA_OK)
    return st;

  for (size_t i = 0u; i < mh; ++i) {
    const double hv = silu_f64((double)out_z1[i]);
    const float hf = (float)hv;
    if (!f64_is_finite(hv) || !f32_is_finite(hf))
      return ET_QLORA_ERR_NONFINITE;
    out_h1[i] = hf;
  }

  return et_qlora_linear_forward(&l2, out_h1, w2_nf4, w2_bytes, b2, a2,
                                 b2_lora, out_y);
}

et_qlora_status et_qlora_mlp_train_step(
    const et_qlora_mlp_config *cfg, et_qlora_adamw_config *opt, const float *x,
    const float *target, const uint8_t *w1_nf4, size_t w1_bytes,
    const float *b1, const uint8_t *w2_nf4, size_t w2_bytes, const float *b2,
    float *a1, float *b1_lora, float *a2, float *b2_lora, float *m_a1,
    float *v_a1, float *m_b1, float *v_b1, float *m_a2, float *v_a2,
    float *m_b2, float *v_b2, float *scratch_f32, size_t scratch_floats,
    double *out_loss) {
  et_qlora_status st = et_qlora_validate_mlp_config(cfg);
  if (st != ET_QLORA_OK)
    return st;
  st = et_qlora_validate_adamw_config(opt);
  if (st != ET_QLORA_OK)
    return st;
  if (out_loss == NULL)
    return ET_QLORA_ERR_ARGUMENT;

  size_t need_floats = 0u;
  st = et_qlora_mlp_scratch_floats(cfg, &need_floats);
  if (st != ET_QLORA_OK)
    return st;
  if (scratch_floats < need_floats)
    return ET_QLORA_ERR_CAPACITY;
  st = check_f32_buffer(scratch_f32, need_floats);
  if (st != ET_QLORA_OK)
    return st;

  const size_t m = (size_t)cfg->m;
  const size_t din = (size_t)cfg->d_in;
  const size_t dh = (size_t)cfg->d_hidden;
  const size_t dout = (size_t)cfg->d_out;
  const size_t r = (size_t)cfg->r;

  const size_t a1_n = r * din;
  const size_t b1_n = dh * r;
  const size_t a2_n = r * dh;
  const size_t b2_n = dout * r;
  const size_t y_n = m * dout;

  st = check_f32_buffer(target, y_n);
  if (st != ET_QLORA_OK)
    return st;
  st = check_finite_f32_array(target, y_n, NULL);
  if (st != ET_QLORA_OK)
    return st;

  /* Partition scratch workspace (zero heap allocations). */
  float *cursor = scratch_f32;
  float *z1 = cursor;
  cursor += m * dh;
  float *h1 = cursor;
  cursor += m * dh;
  float *y = cursor;
  cursor += y_n;
  float *dy = cursor;
  cursor += y_n;
  float *dh1 = cursor;
  cursor += m * dh;
  float *dz1 = cursor;
  cursor += m * dh;
  float *dx = cursor;
  cursor += m * din;
  float *da1 = cursor;
  cursor += a1_n;
  float *db1 = cursor;
  cursor += b1_n;
  float *da2 = cursor;
  cursor += a2_n;
  float *db2 = cursor;

  /* Check mutable parameter / AdamW state buffers and disjointness. */
  const span_entry opt_spans[14] = {
      {a1, a1_n * sizeof(float)},
      {b1_lora, b1_n * sizeof(float)},
      {a2, a2_n * sizeof(float)},
      {b2_lora, b2_n * sizeof(float)},
      {m_a1, a1_n * sizeof(float)},
      {v_a1, a1_n * sizeof(float)},
      {m_b1, b1_n * sizeof(float)},
      {v_b1, b1_n * sizeof(float)},
      {m_a2, a2_n * sizeof(float)},
      {v_a2, a2_n * sizeof(float)},
      {m_b2, b2_n * sizeof(float)},
      {v_b2, b2_n * sizeof(float)},
      {scratch_f32, need_floats * sizeof(float)},
      {target, y_n * sizeof(float)}};
  for (size_t i = 0u; i < 12u; ++i) {
    st = check_f32_buffer(opt_spans[i].ptr, opt_spans[i].bytes / sizeof(float));
    if (st != ET_QLORA_OK)
      return st;
  }
  if (any_spans_overlap(opt_spans, 14u))
    return ET_QLORA_ERR_ALIAS;

  /* 1. Forward pass across both frozen NF4 + LoRA layers. */
  st = et_qlora_mlp_forward(cfg, x, w1_nf4, w1_bytes, b1, a1, b1_lora, w2_nf4,
                            w2_bytes, b2, a2, b2_lora, z1, h1, y);
  if (st != ET_QLORA_OK)
    return st;

  /* 2. MSE loss and upstream gradient dY = (Y - target) / (m * d_out). */
  const double inv_n = 1.0 / (double)y_n;
  double sum_sq = 0.0;
  for (size_t i = 0u; i < y_n; ++i) {
    const double diff = (double)y[i] - (double)target[i];
    sum_sq += diff * diff;
    const double g = diff * inv_n;
    const float gf = (float)g;
    if (!f64_is_finite(sum_sq) || !f32_is_finite(gf))
      return ET_QLORA_ERR_NONFINITE;
    dy[i] = gf;
  }
  const double loss = 0.5 * inv_n * sum_sq;
  if (!f64_is_finite(loss))
    return ET_QLORA_ERR_NONFINITE;

  /* 3. Analytic backward: Layer 2 -> SiLU -> Layer 1 (W1_nf4, W2_nf4 frozen). */
  et_qlora_linear_config l1, l2;
  mlp_layer_configs(cfg, &l1, &l2);

  st = et_qlora_linear_backward(&l2, h1, w2_nf4, w2_bytes, a2, b2_lora, dy,
                                dh1, da2, db2, NULL);
  if (st != ET_QLORA_OK)
    return st;

  for (size_t i = 0u; i < m * dh; ++i) {
    const double g = (double)dh1[i] * silu_grad_f64((double)z1[i]);
    const float gf = (float)g;
    if (!f64_is_finite(g) || !f32_is_finite(gf))
      return ET_QLORA_ERR_NONFINITE;
    dz1[i] = gf;
  }

  st = et_qlora_linear_backward(&l1, x, w1_nf4, w1_bytes, a1, b1_lora, dz1, dx,
                                da1, db1, NULL);
  if (st != ET_QLORA_OK)
    return st;

  /* 4. Dry-run AdamW across all 4 LoRA tensors for strict failure atomicity. */
  st = adamw_step_internal(a1_n, opt, a1, da1, m_a1, v_a1, 1);
  if (st != ET_QLORA_OK)
    return st;
  st = adamw_step_internal(b1_n, opt, b1_lora, db1, m_b1, v_b1, 1);
  if (st != ET_QLORA_OK)
    return st;
  st = adamw_step_internal(a2_n, opt, a2, da2, m_a2, v_a2, 1);
  if (st != ET_QLORA_OK)
    return st;
  st = adamw_step_internal(b2_n, opt, b2_lora, db2, m_b2, v_b2, 1);
  if (st != ET_QLORA_OK)
    return st;

  /* 5. Commit AdamW updates on LoRA (A1, B1, A2, B2) ONLY and increment step. */
  (void)adamw_step_internal(a1_n, opt, a1, da1, m_a1, v_a1, 0);
  (void)adamw_step_internal(b1_n, opt, b1_lora, db1, m_b1, v_b1, 0);
  (void)adamw_step_internal(a2_n, opt, a2, da2, m_a2, v_a2, 0);
  (void)adamw_step_internal(b2_n, opt, b2_lora, db2, m_b2, v_b2, 0);
  opt->step += 1u;
  *out_loss = loss;
  return ET_QLORA_OK;
}

/* ---- Versioned SHA-256 Checksummed Checkpoint (ETQL1000 v1.0) ---- */

static size_t mlp_lora_param_count(const et_qlora_mlp_config *cfg) {
  const size_t din = (size_t)cfg->d_in;
  const size_t dh = (size_t)cfg->d_hidden;
  const size_t dout = (size_t)cfg->d_out;
  const size_t r = (size_t)cfg->r;
  return r * din + dh * r + r * dh + dout * r;
}

et_qlora_status et_qlora_checkpoint_size(const et_qlora_mlp_config *cfg,
                                         size_t *out_bytes) {
  const et_qlora_status cst = et_qlora_validate_mlp_config(cfg);
  if (cst != ET_QLORA_OK)
    return cst;
  if (out_bytes == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  /* 3 tensors per LoRA weight: params, AdamW m, AdamW v (4 bytes each). */
  const size_t total_floats = mlp_lora_param_count(cfg) * 3u;
  *out_bytes = (size_t)ET_QLORA_CHECKPOINT_HEADER_BYTES +
               total_floats * sizeof(float) +
               (size_t)ET_QLORA_CHECKPOINT_CHECKSUM_BYTES;
  return ET_QLORA_OK;
}

static void write_f32_array_le(uint8_t *dst, const float *src, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    store_f32_le(dst + i * 4u, src[i]);
}

static et_qlora_status validate_f32_array_le(const uint8_t *src, size_t count,
                                             int require_nonneg) {
  for (size_t i = 0u; i < count; ++i) {
    const float v = load_f32_le(src + i * 4u);
    if (!f32_is_finite(v) || (require_nonneg && v < 0.0f))
      return ET_QLORA_ERR_NONFINITE;
  }
  return ET_QLORA_OK;
}

static void read_f32_array_le(const uint8_t *src, float *dst, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    dst[i] = load_f32_le(src + i * 4u);
}

et_qlora_status et_qlora_checkpoint_serialize(
    const et_qlora_mlp_config *cfg, const et_qlora_adamw_config *opt,
    const uint8_t base_nf4_sha256[ET_QLORA_SHA256_BYTES], const float *a1,
    const float *b1_lora, const float *a2, const float *b2_lora,
    const float *m_a1, const float *v_a1, const float *m_b1, const float *v_b1,
    const float *m_a2, const float *v_a2, const float *m_b2, const float *v_b2,
    uint8_t *out_bytes, size_t capacity, size_t *out_size) {
  size_t expected_size = 0u;
  et_qlora_status st = et_qlora_checkpoint_size(cfg, &expected_size);
  if (st != ET_QLORA_OK)
    return st;
  st = et_qlora_validate_adamw_config(opt);
  if (st != ET_QLORA_OK)
    return st;
  if (base_nf4_sha256 == NULL || out_bytes == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (capacity < expected_size)
    return ET_QLORA_ERR_CAPACITY;
  st = check_byte_buffer(out_bytes, expected_size);
  if (st != ET_QLORA_OK)
    return st;

  const size_t a1_n = (size_t)cfg->r * (size_t)cfg->d_in;
  const size_t b1_n = (size_t)cfg->d_hidden * (size_t)cfg->r;
  const size_t a2_n = (size_t)cfg->r * (size_t)cfg->d_hidden;
  const size_t b2_n = (size_t)cfg->d_out * (size_t)cfg->r;

  const float *arrays[12] = {a1,   b1_lora, a2,   b2_lora, m_a1, m_b1,
                             m_a2, m_b2,    v_a1, v_b1,    v_a2, v_b2};
  const size_t counts[12] = {a1_n, b1_n, a2_n, b2_n, a1_n, b1_n,
                             a2_n, b2_n, a1_n, b1_n, a2_n, b2_n};

  for (size_t i = 0u; i < 12u; ++i) {
    st = check_f32_buffer(arrays[i], counts[i]);
    if (st != ET_QLORA_OK)
      return st;
    if (spans_overlap(out_bytes, expected_size, arrays[i],
                      counts[i] * sizeof(float)))
      return ET_QLORA_ERR_ALIAS;
    for (size_t j = 0u; j < counts[i]; ++j) {
      const float v = arrays[i][j];
      if (!f32_is_finite(v) || (i >= 8u && v < 0.0f))
        return ET_QLORA_ERR_NONFINITE;
    }
  }
  if (spans_overlap(out_bytes, expected_size, base_nf4_sha256,
                    ET_QLORA_SHA256_BYTES))
    return ET_QLORA_ERR_ALIAS;

  /* Write 128-byte header. */
  memset(out_bytes, 0, ET_QLORA_CHECKPOINT_HEADER_BYTES);
  memcpy(out_bytes, ET_QLORA_CHECKPOINT_MAGIC, ET_QLORA_CHECKPOINT_MAGIC_BYTES);
  store_u32_le(out_bytes + 8u, ET_QLORA_CHECKPOINT_VERSION_MAJOR);
  store_u32_le(out_bytes + 12u, ET_QLORA_CHECKPOINT_VERSION_MINOR);
  store_u32_le(out_bytes + 16u, cfg->d_in);
  store_u32_le(out_bytes + 20u, cfg->d_hidden);
  store_u32_le(out_bytes + 24u, cfg->d_out);
  store_u32_le(out_bytes + 28u, cfg->r);
  store_u32_le(out_bytes + 32u, cfg->block_size);
  store_u32_le(out_bytes + 36u, cfg->dq_block_size);
  store_u32_le(out_bytes + 40u, cfg->quant_mode);
  store_f32_le(out_bytes + 44u, cfg->alpha);
  store_u64_le(out_bytes + 48u, opt->step);
  store_f32_le(out_bytes + 56u, opt->lr);
  store_f32_le(out_bytes + 60u, opt->beta1);
  store_f32_le(out_bytes + 64u, opt->beta2);
  store_f32_le(out_bytes + 68u, opt->eps);
  store_f32_le(out_bytes + 72u, opt->weight_decay);
  store_u32_le(out_bytes + 76u, 0u); /* flags */
  memcpy(out_bytes + 80u, base_nf4_sha256, ET_QLORA_SHA256_BYTES);
  /* bytes [112..127] remain 0 */

  /* Write payload. */
  uint8_t *cur = out_bytes + ET_QLORA_CHECKPOINT_HEADER_BYTES;
  for (size_t i = 0u; i < 12u; ++i) {
    write_f32_array_le(cur, arrays[i], counts[i]);
    cur += counts[i] * sizeof(float);
  }

  /* Compute trailing SHA-256 over [0, expected_size - 32). */
  uint8_t digest[ET_QLORA_SHA256_BYTES];
  st = et_qlora_sha256_bytes(
      out_bytes, expected_size - ET_QLORA_CHECKPOINT_CHECKSUM_BYTES, digest);
  if (st != ET_QLORA_OK)
    return st;
  memcpy(out_bytes + expected_size - ET_QLORA_CHECKPOINT_CHECKSUM_BYTES, digest,
         ET_QLORA_SHA256_BYTES);

  if (out_size != NULL)
    *out_size = expected_size;
  return ET_QLORA_OK;
}

et_qlora_status et_qlora_checkpoint_parse(
    const uint8_t *bytes, size_t size, const et_qlora_mlp_config *expected_cfg,
    const uint8_t expected_base_sha256[ET_QLORA_SHA256_BYTES],
    et_qlora_checkpoint_meta *out_meta, et_qlora_adamw_config *out_opt,
    float *out_a1, float *out_b1_lora, float *out_a2, float *out_b2_lora,
    float *out_m_a1, float *out_v_a1, float *out_m_b1, float *out_v_b1,
    float *out_m_a2, float *out_v_a2, float *out_m_b2, float *out_v_b2) {
  size_t expected_size = 0u;
  et_qlora_status st = et_qlora_checkpoint_size(expected_cfg, &expected_size);
  if (st != ET_QLORA_OK)
    return st;
  if (bytes == NULL || expected_base_sha256 == NULL)
    return ET_QLORA_ERR_ARGUMENT;
  if (size < ET_QLORA_CHECKPOINT_MIN_BYTES)
    return ET_QLORA_ERR_TRUNCATED;
  st = check_byte_buffer(bytes, size);
  if (st != ET_QLORA_OK)
    return st;

  /* 1. Magic and version check first. */
  if (memcmp(bytes, ET_QLORA_CHECKPOINT_MAGIC,
             ET_QLORA_CHECKPOINT_MAGIC_BYTES) != 0)
    return ET_QLORA_ERR_FORMAT;
  const uint32_t ver_maj = load_u32_le(bytes + 8u);
  const uint32_t ver_min = load_u32_le(bytes + 12u);
  if (ver_maj != ET_QLORA_CHECKPOINT_VERSION_MAJOR ||
      ver_min != ET_QLORA_CHECKPOINT_VERSION_MINOR)
    return ET_QLORA_ERR_VERSION;

  /* 2. Exact size check. */
  if (size < expected_size)
    return ET_QLORA_ERR_TRUNCATED;
  if (size > expected_size)
    return ET_QLORA_ERR_FORMAT;

  /* 3. SHA-256 integrity check over header + payload. */
  uint8_t computed_sha[ET_QLORA_SHA256_BYTES];
  st = et_qlora_sha256_bytes(
      bytes, size - ET_QLORA_CHECKPOINT_CHECKSUM_BYTES, computed_sha);
  if (st != ET_QLORA_OK)
    return st;
  const uint8_t *stored_sha =
      bytes + size - ET_QLORA_CHECKPOINT_CHECKSUM_BYTES;
  if (memcmp(computed_sha, stored_sha, ET_QLORA_SHA256_BYTES) != 0)
    return ET_QLORA_ERR_CHECKSUM;

  /* 4. Header fields, reserved bytes, config binding, and base NF4 hash. */
  const uint32_t flags = load_u32_le(bytes + 76u);
  if (flags != 0u)
    return ET_QLORA_ERR_FORMAT;
  for (size_t i = 112u; i < ET_QLORA_CHECKPOINT_HEADER_BYTES; ++i) {
    if (bytes[i] != 0u)
      return ET_QLORA_ERR_FORMAT;
  }

  et_qlora_checkpoint_meta meta;
  memset(&meta, 0, sizeof(meta));
  meta.version_major = ver_maj;
  meta.version_minor = ver_min;
  meta.d_in = load_u32_le(bytes + 16u);
  meta.d_hidden = load_u32_le(bytes + 20u);
  meta.d_out = load_u32_le(bytes + 24u);
  meta.r = load_u32_le(bytes + 28u);
  meta.block_size = load_u32_le(bytes + 32u);
  meta.dq_block_size = load_u32_le(bytes + 36u);
  meta.quant_mode = load_u32_le(bytes + 40u);
  meta.alpha = load_f32_le(bytes + 44u);
  meta.adamw_step = load_u64_le(bytes + 48u);
  meta.lr = load_f32_le(bytes + 56u);
  meta.beta1 = load_f32_le(bytes + 60u);
  meta.beta2 = load_f32_le(bytes + 64u);
  meta.eps = load_f32_le(bytes + 68u);
  meta.weight_decay = load_f32_le(bytes + 72u);
  meta.flags = flags;
  memcpy(meta.base_nf4_sha256, bytes + 80u, ET_QLORA_SHA256_BYTES);
  memcpy(meta.checkpoint_sha256, stored_sha, ET_QLORA_SHA256_BYTES);

  if (meta.d_in != expected_cfg->d_in ||
      meta.d_hidden != expected_cfg->d_hidden ||
      meta.d_out != expected_cfg->d_out || meta.r != expected_cfg->r ||
      meta.block_size != expected_cfg->block_size ||
      meta.dq_block_size != expected_cfg->dq_block_size ||
      meta.quant_mode != expected_cfg->quant_mode ||
      f32_to_bits(meta.alpha) != f32_to_bits(expected_cfg->alpha))
    return ET_QLORA_ERR_SHAPE;

  et_qlora_adamw_config parsed_opt;
  memset(&parsed_opt, 0, sizeof(parsed_opt));
  parsed_opt.lr = meta.lr;
  parsed_opt.beta1 = meta.beta1;
  parsed_opt.beta2 = meta.beta2;
  parsed_opt.eps = meta.eps;
  parsed_opt.weight_decay = meta.weight_decay;
  parsed_opt.step = meta.adamw_step;
  st = et_qlora_validate_adamw_config(&parsed_opt);
  if (st != ET_QLORA_OK)
    return st;

  if (memcmp(meta.base_nf4_sha256, expected_base_sha256,
             ET_QLORA_SHA256_BYTES) != 0)
    return ET_QLORA_ERR_BASE_HASH;

  /* 5. Validate output buffers and every payload float before committing. */
  const size_t a1_n = (size_t)expected_cfg->r * (size_t)expected_cfg->d_in;
  const size_t b1_n = (size_t)expected_cfg->d_hidden * (size_t)expected_cfg->r;
  const size_t a2_n = (size_t)expected_cfg->r * (size_t)expected_cfg->d_hidden;
  const size_t b2_n = (size_t)expected_cfg->d_out * (size_t)expected_cfg->r;

  float *outs[12] = {out_a1,   out_b1_lora, out_a2,   out_b2_lora,
                     out_m_a1, out_m_b1,    out_m_a2, out_m_b2,
                     out_v_a1, out_v_b1,    out_v_a2, out_v_b2};
  const size_t counts[12] = {a1_n, b1_n, a2_n, b2_n, a1_n, b1_n,
                             a2_n, b2_n, a1_n, b1_n, a2_n, b2_n};

  span_entry spans[13];
  spans[0].ptr = bytes;
  spans[0].bytes = size;
  for (size_t i = 0u; i < 12u; ++i) {
    st = check_f32_buffer(outs[i], counts[i]);
    if (st != ET_QLORA_OK)
      return st;
    spans[i + 1u].ptr = outs[i];
    spans[i + 1u].bytes = counts[i] * sizeof(float);
  }
  if (any_spans_overlap(spans, 13u))
    return ET_QLORA_ERR_ALIAS;

  const uint8_t *cur = bytes + ET_QLORA_CHECKPOINT_HEADER_BYTES;
  for (size_t i = 0u; i < 12u; ++i) {
    st = validate_f32_array_le(cur, counts[i], i >= 8u);
    if (st != ET_QLORA_OK)
      return st;
    cur += counts[i] * sizeof(float);
  }

  /* 6. Commit outputs only after every check passed. */
  cur = bytes + ET_QLORA_CHECKPOINT_HEADER_BYTES;
  for (size_t i = 0u; i < 12u; ++i) {
    read_f32_array_le(cur, outs[i], counts[i]);
    cur += counts[i] * sizeof(float);
  }
  if (out_meta != NULL)
    *out_meta = meta;
  if (out_opt != NULL)
    *out_opt = parsed_opt;
  return ET_QLORA_OK;
}

/* ---- Atomic File I/O (0600 mkstemp + fsync + rename / O_NOFOLLOW) ---- */

static int write_all_fd(int fd, const uint8_t *buf, size_t len) {
  size_t written = 0u;
  while (written < len) {
    const ssize_t n = write(fd, buf + written, len - written);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    if (n == 0)
      return -1;
    written += (size_t)n;
  }
  return 0;
}

et_qlora_status et_qlora_checkpoint_save(
    const char *path, const et_qlora_mlp_config *cfg,
    const et_qlora_adamw_config *opt,
    const uint8_t base_nf4_sha256[ET_QLORA_SHA256_BYTES], const float *a1,
    const float *b1_lora, const float *a2, const float *b2_lora,
    const float *m_a1, const float *v_a1, const float *m_b1, const float *v_b1,
    const float *m_a2, const float *v_a2, const float *m_b2, const float *v_b2) {
  if (path == NULL || path[0] == '\0')
    return ET_QLORA_ERR_ARGUMENT;

  size_t ckpt_bytes = 0u;
  et_qlora_status st = et_qlora_checkpoint_size(cfg, &ckpt_bytes);
  if (st != ET_QLORA_OK)
    return st;

  uint8_t *buf = (uint8_t *)malloc(ckpt_bytes);
  if (buf == NULL)
    return ET_QLORA_ERR_NOMEM;

  size_t actual_size = 0u;
  st = et_qlora_checkpoint_serialize(
      cfg, opt, base_nf4_sha256, a1, b1_lora, a2, b2_lora, m_a1, v_a1, m_b1,
      v_b1, m_a2, v_a2, m_b2, v_b2, buf, ckpt_bytes, &actual_size);
  if (st != ET_QLORA_OK) {
    free(buf);
    return st;
  }

  /* Build sibling temp template "<dir>/.etql.tmp.XXXXXX" (atomic rename). */
  const char *slash = strrchr(path, '/');
  const size_t dir_len = (slash != NULL) ? (size_t)(slash - path + 1) : 0u;
  static const char tmpl_suffix[] = ".etql.tmp.XXXXXX";
  const size_t tmpl_len = dir_len + sizeof(tmpl_suffix);
  char *tmpl = (char *)malloc(tmpl_len);
  if (tmpl == NULL) {
    free(buf);
    return ET_QLORA_ERR_NOMEM;
  }
  if (dir_len > 0u)
    memcpy(tmpl, path, dir_len);
  memcpy(tmpl + dir_len, tmpl_suffix, sizeof(tmpl_suffix));

  const int fd = mkstemp(tmpl);
  if (fd < 0) {
    free(tmpl);
    free(buf);
    return ET_QLORA_ERR_IO;
  }

  int io_ok = 1;
  if (fchmod(fd, S_IRUSR | S_IWUSR) != 0)
    io_ok = 0;
  if (io_ok && write_all_fd(fd, buf, actual_size) != 0)
    io_ok = 0;
  if (io_ok && fsync(fd) != 0)
    io_ok = 0;
  if (close(fd) != 0)
    io_ok = 0;
  if (io_ok && rename(tmpl, path) != 0)
    io_ok = 0;
  if (!io_ok)
    (void)unlink(tmpl);

  free(tmpl);
  free(buf);
  return io_ok ? ET_QLORA_OK : ET_QLORA_ERR_IO;
}

et_qlora_status et_qlora_checkpoint_load(
    const char *path, const et_qlora_mlp_config *expected_cfg,
    const uint8_t expected_base_sha256[ET_QLORA_SHA256_BYTES],
    et_qlora_checkpoint_meta *out_meta, et_qlora_adamw_config *out_opt,
    float *out_a1, float *out_b1_lora, float *out_a2, float *out_b2_lora,
    float *out_m_a1, float *out_v_a1, float *out_m_b1, float *out_v_b1,
    float *out_m_a2, float *out_v_a2, float *out_m_b2, float *out_v_b2) {
  if (path == NULL || path[0] == '\0')
    return ET_QLORA_ERR_ARGUMENT;

  size_t expected_size = 0u;
  et_qlora_status st = et_qlora_checkpoint_size(expected_cfg, &expected_size);
  if (st != ET_QLORA_OK)
    return st;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ELOOP)
      return ET_QLORA_ERR_SYMLINK;
    return ET_QLORA_ERR_IO;
  }

  struct stat sb;
  if (fstat(fd, &sb) != 0) {
    (void)close(fd);
    return ET_QLORA_ERR_IO;
  }
  if (!S_ISREG(sb.st_mode)) {
    (void)close(fd);
    return ET_QLORA_ERR_FORMAT;
  }
  if (sb.st_size < 0) {
    (void)close(fd);
    return ET_QLORA_ERR_IO;
  }
  const size_t file_size = (size_t)sb.st_size;
  if (file_size < ET_QLORA_CHECKPOINT_MIN_BYTES) {
    (void)close(fd);
    return ET_QLORA_ERR_TRUNCATED;
  }
  /* Cap allocation at expected_size + 1 to detect oversized files safely. */
  const size_t alloc_size =
      (file_size > expected_size) ? (expected_size + 1u) : file_size;
  uint8_t *buf = (uint8_t *)malloc(alloc_size);
  if (buf == NULL) {
    (void)close(fd);
    return ET_QLORA_ERR_NOMEM;
  }

  size_t total_read = 0u;
  while (total_read < alloc_size) {
    const ssize_t n = read(fd, buf + total_read, alloc_size - total_read);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      free(buf);
      (void)close(fd);
      return ET_QLORA_ERR_IO;
    }
    if (n == 0)
      break;
    total_read += (size_t)n;
  }
  (void)close(fd);

  st = et_qlora_checkpoint_parse(
      buf, total_read, expected_cfg, expected_base_sha256, out_meta, out_opt,
      out_a1, out_b1_lora, out_a2, out_b2_lora, out_m_a1, out_v_a1, out_m_b1,
      out_v_b1, out_m_a2, out_v_a2, out_m_b2, out_v_b2);
  free(buf);
  return st;
}
