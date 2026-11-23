#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/lora_adapter_abi.h"

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
               "LoRA engine requires 8-bit bytes and binary32 floats");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "LoRA engine requires IEEE-754 binary32 evaluation");

/* ---- Standalone SHA-256 (matching eval5_benchmark.c / c2_checkpoint) ---- */

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
  for (i = 0; i < 16u; ++i) {
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
  for (i = 0; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256_bytes(const uint8_t *data, size_t size, uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  sha256_update(&state, data, size);
  sha256_final(&state, digest);
}

/* ---- Little-endian encoding helpers ---- */

static void encode_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static uint32_t decode_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) |
         ((uint32_t)in[2] << 16u) | ((uint32_t)in[3] << 24u);
}

static uint32_t f32_to_bits(float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int bits_to_finite_f32(uint32_t bits, float *out) {
  if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000))
    return 0;
  memcpy(out, &bits, sizeof(*out));
  return isfinite(*out);
}

/* ---- FP environment, alignment, span, and validation helpers ---- */

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

static int valid_knr(uint32_t k, uint32_t n, uint32_t r) {
  const uint32_t min_kn = k < n ? k : n;
  return k >= 1u && k <= ET_LORA_MAX_K && n >= 1u && n <= ET_LORA_MAX_N &&
         r >= 1u && r <= ET_LORA_MAX_R && r <= min_kn;
}

static et_lora_status validate_config_internal(const et_lora_config *cfg,
                                               float *out_scale) {
  if (cfg == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_LORA_ERR_FP_ENV;
  if (cfg->m < 1u || cfg->m > ET_LORA_MAX_M ||
      !valid_knr(cfg->k, cfg->n, cfg->r))
    return ET_LORA_ERR_SHAPE;
  if (!isfinite(cfg->alpha))
    return ET_LORA_ERR_NONFINITE;
  if (!(cfg->alpha > 0.0f))
    return ET_LORA_ERR_ARGUMENT;
  const float scale = cfg->alpha / (float)cfg->r;
  if (!isfinite(scale))
    return ET_LORA_ERR_NONFINITE;
  if (out_scale != NULL)
    *out_scale = scale;
  return ET_LORA_OK;
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

static et_lora_status check_f32_buffer(const void *ptr, size_t count) {
  const size_t bytes = count * sizeof(float);
  if (ptr == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (!aligned_span(ptr, _Alignof(float), bytes))
    return ET_LORA_ERR_ALIGNMENT;
  return ET_LORA_OK;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static void hash_f32_le(const float *data, size_t count,
                        uint8_t digest[ET_LORA_SHA256_BYTES]) {
  sha256_state state;
  uint8_t le_buf[4];
  sha256_init(&state);
  for (size_t i = 0; i < count; ++i) {
    encode_u32_le(le_buf, f32_to_bits(data[i]));
    sha256_update(&state, le_buf, sizeof(le_buf));
  }
  sha256_final(&state, digest);
}

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

/* ---- Public API: Config and Adapter Initialization ---- */

et_lora_status et_lora_validate_config(const et_lora_config *cfg) {
  return validate_config_internal(cfg, NULL);
}

et_lora_status et_lora_init_adapter_raw(const et_lora_config *cfg,
                                        uint64_t seed, void *a_raw,
                                        void *b_raw) {
  et_lora_status status = validate_config_internal(cfg, NULL);
  if (status != ET_LORA_OK)
    return status;

  const size_t a_count = (size_t)cfg->r * (size_t)cfg->k;
  const size_t b_count = (size_t)cfg->n * (size_t)cfg->r;
  const size_t a_bytes = a_count * sizeof(float);
  const size_t b_bytes = b_count * sizeof(float);

  status = check_f32_buffer(a_raw, a_count);
  if (status != ET_LORA_OK)
    return status;
  status = check_f32_buffer(b_raw, b_count);
  if (status != ET_LORA_OK)
    return status;

  const span_entry spans[] = {
      {cfg, sizeof(*cfg)}, {a_raw, a_bytes}, {b_raw, b_bytes}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])))
    return ET_LORA_ERR_ALIAS;

  float a_tmp[ET_LORA_MAX_R * ET_LORA_MAX_K];
  const float bound = 0.25f / sqrtf((float)cfg->k);
  if (!isfinite(bound) || !(bound > 0.0f))
    return ET_LORA_ERR_NONFINITE;

  uint64_t rng = seed ^ UINT64_C(0x4c6f524131303030);
  for (size_t i = 0; i < a_count; ++i) {
    /* Deterministic non-zero signed value in [-bound, +bound]. */
    const uint64_t word = splitmix64_next(&rng);
    const int32_t bucket = (int32_t)(word % UINT64_C(254)) - 127;
    const float signed_unit =
        (bucket == 0 ? 1.0f : (float)bucket) / 127.0f;
    const float val = signed_unit * bound;
    if (!isfinite(val))
      return ET_LORA_ERR_NONFINITE;
    a_tmp[i] = val;
  }

  memcpy(a_raw, a_tmp, a_bytes);
  memset(b_raw, 0, b_bytes);
  return ET_LORA_OK;
}

et_lora_status et_lora_init_adapter(const et_lora_config *cfg, uint64_t seed,
                                    float *a, float *b) {
  return et_lora_init_adapter_raw(cfg, seed, (void *)a, (void *)b);
}

/* ---- Public API: Base Forward & LoRA Forward ---- */

et_lora_status et_lora_base_forward_raw(const et_lora_config *cfg,
                                        const void *x_raw, const void *w_raw,
                                        void *y_raw) {
  et_lora_status status = validate_config_internal(cfg, NULL);
  if (status != ET_LORA_OK)
    return status;

  const size_t x_count = (size_t)cfg->m * (size_t)cfg->k;
  const size_t w_count = (size_t)cfg->n * (size_t)cfg->k;
  const size_t y_count = (size_t)cfg->m * (size_t)cfg->n;

  status = check_f32_buffer(x_raw, x_count);
  if (status != ET_LORA_OK)
    return status;
  status = check_f32_buffer(w_raw, w_count);
  if (status != ET_LORA_OK)
    return status;
  status = check_f32_buffer(y_raw, y_count);
  if (status != ET_LORA_OK)
    return status;

  const span_entry spans[] = {{x_raw, x_count * sizeof(float)},
                              {w_raw, w_count * sizeof(float)},
                              {y_raw, y_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])) ||
      spans_overlap(y_raw, y_count * sizeof(float), cfg, sizeof(*cfg)))
    return ET_LORA_ERR_ALIAS;

  const float *x = (const float *)x_raw;
  const float *w = (const float *)w_raw;
  if (!all_finite_f32(x, x_count) || !all_finite_f32(w, w_count))
    return ET_LORA_ERR_NONFINITE;

  float y_tmp[ET_LORA_MAX_M * ET_LORA_MAX_N];
  const uint32_t M = cfg->m;
  const uint32_t K = cfg->k;
  const uint32_t N = cfg->n;

  for (uint32_t m = 0; m < M; ++m) {
    const float *x_row = x + (size_t)m * K;
    for (uint32_t n = 0; n < N; ++n) {
      const float *w_row = w + (size_t)n * K;
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc = acc + x_row[k] * w_row[k];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      y_tmp[(size_t)m * N + n] = acc;
    }
  }

  memcpy(y_raw, y_tmp, y_count * sizeof(float));
  return ET_LORA_OK;
}

et_lora_status et_lora_base_forward(const et_lora_config *cfg, const float *x,
                                    const float *w, float *y) {
  return et_lora_base_forward_raw(cfg, (const void *)x, (const void *)w,
                                  (void *)y);
}

et_lora_status et_lora_forward_raw(const et_lora_config *cfg, const void *x_raw,
                                   const void *w_raw, const void *a_raw,
                                   const void *b_raw, void *y_raw) {
  float scale = 0.0f;
  et_lora_status status = validate_config_internal(cfg, &scale);
  if (status != ET_LORA_OK)
    return status;

  const size_t x_count = (size_t)cfg->m * (size_t)cfg->k;
  const size_t w_count = (size_t)cfg->n * (size_t)cfg->k;
  const size_t a_count = (size_t)cfg->r * (size_t)cfg->k;
  const size_t b_count = (size_t)cfg->n * (size_t)cfg->r;
  const size_t y_count = (size_t)cfg->m * (size_t)cfg->n;

  if ((status = check_f32_buffer(x_raw, x_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(w_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(b_raw, b_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(y_raw, y_count)) != ET_LORA_OK)
    return status;

  const span_entry spans[] = {{x_raw, x_count * sizeof(float)},
                              {w_raw, w_count * sizeof(float)},
                              {a_raw, a_count * sizeof(float)},
                              {b_raw, b_count * sizeof(float)},
                              {y_raw, y_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])) ||
      spans_overlap(y_raw, y_count * sizeof(float), cfg, sizeof(*cfg)))
    return ET_LORA_ERR_ALIAS;

  const float *x = (const float *)x_raw;
  const float *w = (const float *)w_raw;
  const float *a = (const float *)a_raw;
  const float *b = (const float *)b_raw;
  if (!all_finite_f32(x, x_count) || !all_finite_f32(w, w_count) ||
      !all_finite_f32(a, a_count) || !all_finite_f32(b, b_count))
    return ET_LORA_ERR_NONFINITE;

  float h_buf[ET_LORA_MAX_M * ET_LORA_MAX_R];
  float y_tmp[ET_LORA_MAX_M * ET_LORA_MAX_N];
  const uint32_t M = cfg->m;
  const uint32_t K = cfg->k;
  const uint32_t N = cfg->n;
  const uint32_t R = cfg->r;

  /* H[m, i] = sum_k X[m, k] * A[i, k] */
  for (uint32_t m = 0; m < M; ++m) {
    const float *x_row = x + (size_t)m * K;
    for (uint32_t i = 0; i < R; ++i) {
      const float *a_row = a + (size_t)i * K;
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc = acc + x_row[k] * a_row[k];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      h_buf[(size_t)m * R + i] = acc;
    }
  }

  /* Y[m, n] = sum_k X[m, k] * W[n, k] + scale * sum_i H[m, i] * B[n, i] */
  for (uint32_t m = 0; m < M; ++m) {
    const float *x_row = x + (size_t)m * K;
    const float *h_row = h_buf + (size_t)m * R;
    for (uint32_t n = 0; n < N; ++n) {
      const float *w_row = w + (size_t)n * K;
      const float *b_row = b + (size_t)n * R;
      float base_acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        base_acc = base_acc + x_row[k] * w_row[k];
        if (!isfinite(base_acc))
          return ET_LORA_ERR_NONFINITE;
      }
      float lora_acc = 0.0f;
      for (uint32_t i = 0; i < R; ++i) {
        lora_acc = lora_acc + h_row[i] * b_row[i];
        if (!isfinite(lora_acc))
          return ET_LORA_ERR_NONFINITE;
      }
      const float scaled = scale * lora_acc;
      const float total = base_acc + scaled;
      if (!isfinite(scaled) || !isfinite(total))
        return ET_LORA_ERR_NONFINITE;
      y_tmp[(size_t)m * N + n] = total;
    }
  }

  memcpy(y_raw, y_tmp, y_count * sizeof(float));
  return ET_LORA_OK;
}

et_lora_status et_lora_forward(const et_lora_config *cfg, const float *x,
                               const float *w, const float *a, const float *b,
                               float *y) {
  return et_lora_forward_raw(cfg, (const void *)x, (const void *)w,
                             (const void *)a, (const void *)b, (void *)y);
}

/* ---- Public API: Analytic Backward ---- */

et_lora_status et_lora_backward_raw(const et_lora_config *cfg,
                                    const void *x_raw, const void *w_raw,
                                    const void *a_raw, const void *b_raw,
                                    const void *dy_raw, void *dx_raw,
                                    void *da_raw, void *db_raw) {
  float scale = 0.0f;
  et_lora_status status = validate_config_internal(cfg, &scale);
  if (status != ET_LORA_OK)
    return status;

  const size_t x_count = (size_t)cfg->m * (size_t)cfg->k;
  const size_t w_count = (size_t)cfg->n * (size_t)cfg->k;
  const size_t a_count = (size_t)cfg->r * (size_t)cfg->k;
  const size_t b_count = (size_t)cfg->n * (size_t)cfg->r;
  const size_t y_count = (size_t)cfg->m * (size_t)cfg->n;

  if ((status = check_f32_buffer(x_raw, x_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(w_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(b_raw, b_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(dy_raw, y_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(dx_raw, x_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(da_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(db_raw, b_count)) != ET_LORA_OK)
    return status;

  const span_entry spans[] = {
      {x_raw, x_count * sizeof(float)},  {w_raw, w_count * sizeof(float)},
      {a_raw, a_count * sizeof(float)},  {b_raw, b_count * sizeof(float)},
      {dy_raw, y_count * sizeof(float)}, {dx_raw, x_count * sizeof(float)},
      {da_raw, a_count * sizeof(float)}, {db_raw, b_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])) ||
      spans_overlap(dx_raw, x_count * sizeof(float), cfg, sizeof(*cfg)) ||
      spans_overlap(da_raw, a_count * sizeof(float), cfg, sizeof(*cfg)) ||
      spans_overlap(db_raw, b_count * sizeof(float), cfg, sizeof(*cfg)))
    return ET_LORA_ERR_ALIAS;

  const float *x = (const float *)x_raw;
  const float *w = (const float *)w_raw;
  const float *a = (const float *)a_raw;
  const float *b = (const float *)b_raw;
  const float *dy = (const float *)dy_raw;
  if (!all_finite_f32(x, x_count) || !all_finite_f32(w, w_count) ||
      !all_finite_f32(a, a_count) || !all_finite_f32(b, b_count) ||
      !all_finite_f32(dy, y_count))
    return ET_LORA_ERR_NONFINITE;

  float h_buf[ET_LORA_MAX_M * ET_LORA_MAX_R];
  float dh_buf[ET_LORA_MAX_M * ET_LORA_MAX_R];
  float dx_tmp[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float da_tmp[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float db_tmp[ET_LORA_MAX_N * ET_LORA_MAX_R];

  const uint32_t M = cfg->m;
  const uint32_t K = cfg->k;
  const uint32_t N = cfg->n;
  const uint32_t R = cfg->r;

  /* 1. Recompute H = X @ A^T  [M, R] */
  for (uint32_t m = 0; m < M; ++m) {
    const float *x_row = x + (size_t)m * K;
    for (uint32_t i = 0; i < R; ++i) {
      const float *a_row = a + (size_t)i * K;
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        acc = acc + x_row[k] * a_row[k];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      h_buf[(size_t)m * R + i] = acc;
    }
  }

  /* 2. dB[n, i] = scale * sum_m dY[m, n] * H[m, i]  [N, R] */
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t i = 0; i < R; ++i) {
      float acc = 0.0f;
      for (uint32_t m = 0; m < M; ++m) {
        acc = acc + dy[(size_t)m * N + n] * h_buf[(size_t)m * R + i];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      const float scaled = scale * acc;
      if (!isfinite(scaled))
        return ET_LORA_ERR_NONFINITE;
      db_tmp[(size_t)n * R + i] = scaled;
    }
  }

  /* 3. dH[m, i] = scale * sum_n dY[m, n] * B[n, i]  [M, R] */
  for (uint32_t m = 0; m < M; ++m) {
    const float *dy_row = dy + (size_t)m * N;
    for (uint32_t i = 0; i < R; ++i) {
      float acc = 0.0f;
      for (uint32_t n = 0; n < N; ++n) {
        acc = acc + dy_row[n] * b[(size_t)n * R + i];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      const float scaled = scale * acc;
      if (!isfinite(scaled))
        return ET_LORA_ERR_NONFINITE;
      dh_buf[(size_t)m * R + i] = scaled;
    }
  }

  /* 4. dA[i, k] = sum_m dH[m, i] * X[m, k]  [R, K] */
  for (uint32_t i = 0; i < R; ++i) {
    for (uint32_t k = 0; k < K; ++k) {
      float acc = 0.0f;
      for (uint32_t m = 0; m < M; ++m) {
        acc = acc + dh_buf[(size_t)m * R + i] * x[(size_t)m * K + k];
        if (!isfinite(acc))
          return ET_LORA_ERR_NONFINITE;
      }
      da_tmp[(size_t)i * K + k] = acc;
    }
  }

  /* 5. dX[m, k] = sum_n dY[m, n] * W[n, k] + sum_i dH[m, i] * A[i, k]  [M, K] */
  for (uint32_t m = 0; m < M; ++m) {
    const float *dy_row = dy + (size_t)m * N;
    const float *dh_row = dh_buf + (size_t)m * R;
    for (uint32_t k = 0; k < K; ++k) {
      float base_acc = 0.0f;
      for (uint32_t n = 0; n < N; ++n) {
        base_acc = base_acc + dy_row[n] * w[(size_t)n * K + k];
        if (!isfinite(base_acc))
          return ET_LORA_ERR_NONFINITE;
      }
      float lora_acc = 0.0f;
      for (uint32_t i = 0; i < R; ++i) {
        lora_acc = lora_acc + dh_row[i] * a[(size_t)i * K + k];
        if (!isfinite(lora_acc))
          return ET_LORA_ERR_NONFINITE;
      }
      const float total = base_acc + lora_acc;
      if (!isfinite(total))
        return ET_LORA_ERR_NONFINITE;
      dx_tmp[(size_t)m * K + k] = total;
    }
  }

  memcpy(dx_raw, dx_tmp, x_count * sizeof(float));
  memcpy(da_raw, da_tmp, a_count * sizeof(float));
  memcpy(db_raw, db_tmp, b_count * sizeof(float));
  return ET_LORA_OK;
}

et_lora_status et_lora_backward(const et_lora_config *cfg, const float *x,
                                const float *w, const float *a, const float *b,
                                const float *dy, float *dx, float *da,
                                float *db) {
  return et_lora_backward_raw(cfg, (const void *)x, (const void *)w,
                              (const void *)a, (const void *)b,
                              (const void *)dy, (void *)dx, (void *)da,
                              (void *)db);
}

/* ---- Public API: Merge and Unmerge ---- */

static et_lora_status merge_impl(const et_lora_config *cfg, const void *w_in_raw,
                                 const void *a_raw, const void *b_raw,
                                 void *w_out_raw, int subtract) {
  float scale = 0.0f;
  et_lora_status status = validate_config_internal(cfg, &scale);
  if (status != ET_LORA_OK)
    return status;

  const size_t w_count = (size_t)cfg->n * (size_t)cfg->k;
  const size_t a_count = (size_t)cfg->r * (size_t)cfg->k;
  const size_t b_count = (size_t)cfg->n * (size_t)cfg->r;

  if ((status = check_f32_buffer(w_in_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(b_raw, b_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(w_out_raw, w_count)) != ET_LORA_OK)
    return status;

  const span_entry spans[] = {{w_in_raw, w_count * sizeof(float)},
                              {a_raw, a_count * sizeof(float)},
                              {b_raw, b_count * sizeof(float)},
                              {w_out_raw, w_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])) ||
      spans_overlap(w_out_raw, w_count * sizeof(float), cfg, sizeof(*cfg)))
    return ET_LORA_ERR_ALIAS;

  const float *w_in = (const float *)w_in_raw;
  const float *a = (const float *)a_raw;
  const float *b = (const float *)b_raw;
  if (!all_finite_f32(w_in, w_count) || !all_finite_f32(a, a_count) ||
      !all_finite_f32(b, b_count))
    return ET_LORA_ERR_NONFINITE;

  float w_tmp[ET_LORA_MAX_N * ET_LORA_MAX_K];
  const uint32_t K = cfg->k;
  const uint32_t N = cfg->n;
  const uint32_t R = cfg->r;

  for (uint32_t n = 0; n < N; ++n) {
    const float *b_row = b + (size_t)n * R;
    for (uint32_t k = 0; k < K; ++k) {
      float delta = 0.0f;
      for (uint32_t i = 0; i < R; ++i) {
        delta = delta + b_row[i] * a[(size_t)i * K + k];
        if (!isfinite(delta))
          return ET_LORA_ERR_NONFINITE;
      }
      const float scaled = scale * delta;
      const float base = w_in[(size_t)n * K + k];
      const float out_val = subtract ? (base - scaled) : (base + scaled);
      if (!isfinite(scaled) || !isfinite(out_val))
        return ET_LORA_ERR_NONFINITE;
      w_tmp[(size_t)n * K + k] = out_val;
    }
  }

  memcpy(w_out_raw, w_tmp, w_count * sizeof(float));
  return ET_LORA_OK;
}

et_lora_status et_lora_merge_raw(const et_lora_config *cfg, const void *w_base,
                                 const void *a, const void *b, void *w_merged) {
  return merge_impl(cfg, w_base, a, b, w_merged, 0);
}

et_lora_status et_lora_merge(const et_lora_config *cfg, const float *w_base,
                             const float *a, const float *b, float *w_merged) {
  return merge_impl(cfg, (const void *)w_base, (const void *)a, (const void *)b,
                    (void *)w_merged, 0);
}

et_lora_status et_lora_unmerge_raw(const et_lora_config *cfg,
                                   const void *w_merged, const void *a,
                                   const void *b, void *w_restored) {
  return merge_impl(cfg, w_merged, a, b, w_restored, 1);
}

et_lora_status et_lora_unmerge(const et_lora_config *cfg, const float *w_merged,
                               const float *a, const float *b,
                               float *w_restored) {
  return merge_impl(cfg, (const void *)w_merged, (const void *)a,
                    (const void *)b, (void *)w_restored, 1);
}

/* ---- Public API: Base Weight SHA-256 ---- */

et_lora_status et_lora_base_weight_sha256_raw(uint32_t k, uint32_t n,
                                              const void *w_base_raw,
                                              void *out_digest_raw) {
  if (!fp_environment_supported())
    return ET_LORA_ERR_FP_ENV;
  if (k < 1u || k > ET_LORA_MAX_K || n < 1u || n > ET_LORA_MAX_N)
    return ET_LORA_ERR_SHAPE;
  const size_t w_count = (size_t)n * (size_t)k;
  et_lora_status status = check_f32_buffer(w_base_raw, w_count);
  if (status != ET_LORA_OK)
    return status;
  if (out_digest_raw == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (!aligned_span(out_digest_raw, 1u, ET_LORA_SHA256_BYTES))
    return ET_LORA_ERR_ALIGNMENT;
  if (spans_overlap(w_base_raw, w_count * sizeof(float), out_digest_raw,
                    ET_LORA_SHA256_BYTES))
    return ET_LORA_ERR_ALIAS;

  const float *w_base = (const float *)w_base_raw;
  if (!all_finite_f32(w_base, w_count))
    return ET_LORA_ERR_NONFINITE;

  uint8_t digest[ET_LORA_SHA256_BYTES];
  hash_f32_le(w_base, w_count, digest);
  memcpy(out_digest_raw, digest, sizeof(digest));
  return ET_LORA_OK;
}

et_lora_status et_lora_base_weight_sha256(
    uint32_t k, uint32_t n, const float *w_base,
    uint8_t out_digest[ET_LORA_SHA256_BYTES]) {
  return et_lora_base_weight_sha256_raw(k, n, (const void *)w_base,
                                        (void *)out_digest);
}

/* ---- Public API: Versioned Binary Adapter Checkpoint (ETLR1000 v1.0) ---- */

static size_t checkpoint_bytes_unchecked(uint32_t k, uint32_t n, uint32_t r) {
  const size_t floats = (size_t)r * (size_t)k + (size_t)n * (size_t)r;
  return ET_LORA_CHECKPOINT_HEADER_BYTES + floats * 4u +
         ET_LORA_CHECKPOINT_CHECKSUM_BYTES;
}

et_lora_status et_lora_checkpoint_size(uint32_t k, uint32_t n, uint32_t r,
                                       size_t *out_bytes) {
  if (out_bytes == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (!valid_knr(k, n, r))
    return ET_LORA_ERR_SHAPE;
  *out_bytes = checkpoint_bytes_unchecked(k, n, r);
  return ET_LORA_OK;
}

static et_lora_status serialize_internal(const et_lora_config *cfg,
                                         const void *w_base_raw,
                                         const void *a_raw, const void *b_raw,
                                         uint8_t *out_bytes, size_t capacity,
                                         size_t *out_size) {
  et_lora_status status = validate_config_internal(cfg, NULL);
  if (status != ET_LORA_OK)
    return status;

  const size_t w_count = (size_t)cfg->n * (size_t)cfg->k;
  const size_t a_count = (size_t)cfg->r * (size_t)cfg->k;
  const size_t b_count = (size_t)cfg->n * (size_t)cfg->r;
  const size_t total = checkpoint_bytes_unchecked(cfg->k, cfg->n, cfg->r);

  if ((status = check_f32_buffer(w_base_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(b_raw, b_count)) != ET_LORA_OK)
    return status;
  if (out_bytes == NULL || out_size == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (capacity < total)
    return ET_LORA_ERR_ARGUMENT;
  if (!aligned_span(out_bytes, 1u, total))
    return ET_LORA_ERR_ALIGNMENT;

  const span_entry spans[] = {{w_base_raw, w_count * sizeof(float)},
                              {a_raw, a_count * sizeof(float)},
                              {b_raw, b_count * sizeof(float)},
                              {out_bytes, total}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])) ||
      spans_overlap(out_bytes, total, cfg, sizeof(*cfg)) ||
      spans_overlap(out_bytes, total, out_size, sizeof(*out_size)))
    return ET_LORA_ERR_ALIAS;

  const float *w_base = (const float *)w_base_raw;
  const float *a = (const float *)a_raw;
  const float *b = (const float *)b_raw;
  if (!all_finite_f32(w_base, w_count) || !all_finite_f32(a, a_count) ||
      !all_finite_f32(b, b_count))
    return ET_LORA_ERR_NONFINITE;

  uint8_t base_digest[ET_LORA_SHA256_BYTES];
  hash_f32_le(w_base, w_count, base_digest);

  uint8_t buf[ET_LORA_CHECKPOINT_MAX_BYTES];
  memcpy(buf, ET_LORA_CHECKPOINT_MAGIC, ET_LORA_CHECKPOINT_MAGIC_BYTES);
  encode_u32_le(buf + 8u, ET_LORA_CHECKPOINT_VERSION_MAJOR);
  encode_u32_le(buf + 12u, ET_LORA_CHECKPOINT_VERSION_MINOR);
  encode_u32_le(buf + 16u, cfg->k);
  encode_u32_le(buf + 20u, cfg->n);
  encode_u32_le(buf + 24u, cfg->r);
  encode_u32_le(buf + 28u, f32_to_bits(cfg->alpha));
  memcpy(buf + 32u, base_digest, sizeof(base_digest));

  size_t offset = ET_LORA_CHECKPOINT_HEADER_BYTES;
  for (size_t i = 0; i < a_count; ++i) {
    encode_u32_le(buf + offset, f32_to_bits(a[i]));
    offset += 4u;
  }
  for (size_t i = 0; i < b_count; ++i) {
    encode_u32_le(buf + offset, f32_to_bits(b[i]));
    offset += 4u;
  }

  uint8_t ckpt_digest[ET_LORA_SHA256_BYTES];
  sha256_bytes(buf, offset, ckpt_digest);
  memcpy(buf + offset, ckpt_digest, sizeof(ckpt_digest));
  offset += sizeof(ckpt_digest);

  memcpy(out_bytes, buf, offset);
  *out_size = offset;
  return ET_LORA_OK;
}

et_lora_status et_lora_checkpoint_serialize(const et_lora_config *cfg,
                                            const float *w_base, const float *a,
                                            const float *b, uint8_t *out_bytes,
                                            size_t capacity, size_t *out_size) {
  return serialize_internal(cfg, (const void *)w_base, (const void *)a,
                            (const void *)b, out_bytes, capacity, out_size);
}

static et_lora_status parse_internal(const uint8_t *bytes, size_t size,
                                     uint32_t expected_k, uint32_t expected_n,
                                     uint32_t expected_r,
                                     const void *w_base_raw,
                                     et_lora_checkpoint_meta *out_meta,
                                     void *out_a_raw, void *out_b_raw) {
  if (!fp_environment_supported())
    return ET_LORA_ERR_FP_ENV;
  if (bytes == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (!valid_knr(expected_k, expected_n, expected_r))
    return ET_LORA_ERR_SHAPE;

  const size_t w_count = (size_t)expected_n * (size_t)expected_k;
  const size_t a_count = (size_t)expected_r * (size_t)expected_k;
  const size_t b_count = (size_t)expected_n * (size_t)expected_r;
  et_lora_status status;
  if ((status = check_f32_buffer(w_base_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(out_a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(out_b_raw, b_count)) != ET_LORA_OK)
    return status;

  const span_entry spans[] = {{bytes, size},
                              {w_base_raw, w_count * sizeof(float)},
                              {out_a_raw, a_count * sizeof(float)},
                              {out_b_raw, b_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])))
    return ET_LORA_ERR_ALIAS;
  if (out_meta != NULL) {
    if (!aligned_span(out_meta, _Alignof(et_lora_checkpoint_meta),
                      sizeof(*out_meta)))
      return ET_LORA_ERR_ALIGNMENT;
    if (spans_overlap(out_meta, sizeof(*out_meta), bytes, size) ||
        spans_overlap(out_meta, sizeof(*out_meta), w_base_raw,
                      w_count * sizeof(float)) ||
        spans_overlap(out_meta, sizeof(*out_meta), out_a_raw,
                      a_count * sizeof(float)) ||
        spans_overlap(out_meta, sizeof(*out_meta), out_b_raw,
                      b_count * sizeof(float)))
      return ET_LORA_ERR_ALIAS;
  }

  const float *w_base = (const float *)w_base_raw;
  if (!all_finite_f32(w_base, w_count))
    return ET_LORA_ERR_NONFINITE;

  /* 1. Magic and minimum size check. */
  if (size >= ET_LORA_CHECKPOINT_MAGIC_BYTES &&
      memcmp(bytes, ET_LORA_CHECKPOINT_MAGIC, ET_LORA_CHECKPOINT_MAGIC_BYTES) !=
          0)
    return ET_LORA_ERR_FORMAT;
  if (size < ET_LORA_CHECKPOINT_MIN_BYTES)
    return ET_LORA_ERR_TRUNCATED;
  if (size > ET_LORA_CHECKPOINT_MAX_BYTES)
    return ET_LORA_ERR_FORMAT;

  /* 2. Format version check. */
  const uint32_t ver_major = decode_u32_le(bytes + 8u);
  const uint32_t ver_minor = decode_u32_le(bytes + 12u);
  if (ver_major != ET_LORA_CHECKPOINT_VERSION_MAJOR ||
      ver_minor != ET_LORA_CHECKPOINT_VERSION_MINOR)
    return ET_LORA_ERR_VERSION;

  /* 3. Truncation / trailing-byte check when header shape is in-range. */
  const uint32_t file_k = decode_u32_le(bytes + 16u);
  const uint32_t file_n = decode_u32_le(bytes + 20u);
  const uint32_t file_r = decode_u32_le(bytes + 24u);
  if (valid_knr(file_k, file_n, file_r)) {
    const size_t expected_size =
        checkpoint_bytes_unchecked(file_k, file_n, file_r);
    if (size < expected_size)
      return ET_LORA_ERR_TRUNCATED;
    if (size > expected_size)
      return ET_LORA_ERR_FORMAT;
  }

  /* 4. SHA-256 integrity verification over header + payload. */
  const size_t signed_bytes = size - ET_LORA_CHECKPOINT_CHECKSUM_BYTES;
  uint8_t computed_digest[ET_LORA_SHA256_BYTES];
  sha256_bytes(bytes, signed_bytes, computed_digest);
  if (memcmp(computed_digest, bytes + signed_bytes, ET_LORA_SHA256_BYTES) != 0)
    return ET_LORA_ERR_CHECKSUM;

  /* 5. Shape validation (header validity and caller shape match). */
  if (!valid_knr(file_k, file_n, file_r))
    return ET_LORA_ERR_SHAPE;
  if (size != checkpoint_bytes_unchecked(file_k, file_n, file_r))
    return ET_LORA_ERR_FORMAT;

  /* 6. Alpha validation (finite, > 0, finite scale). */
  float alpha = 0.0f;
  if (!bits_to_finite_f32(decode_u32_le(bytes + 28u), &alpha))
    return ET_LORA_ERR_NONFINITE;
  if (!(alpha > 0.0f))
    return ET_LORA_ERR_FORMAT;
  if (!isfinite(alpha / (float)file_r))
    return ET_LORA_ERR_NONFINITE;

  /* 7. Decode and finite-validate all A and B floats from file payload. */
  const size_t file_a_count = (size_t)file_r * (size_t)file_k;
  const size_t file_b_count = (size_t)file_n * (size_t)file_r;
  float a_tmp[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b_tmp[ET_LORA_MAX_N * ET_LORA_MAX_R];
  size_t offset = ET_LORA_CHECKPOINT_HEADER_BYTES;
  for (size_t i = 0; i < file_a_count; ++i) {
    if (!bits_to_finite_f32(decode_u32_le(bytes + offset), &a_tmp[i]))
      return ET_LORA_ERR_NONFINITE;
    offset += 4u;
  }
  for (size_t i = 0; i < file_b_count; ++i) {
    if (!bits_to_finite_f32(decode_u32_le(bytes + offset), &b_tmp[i]))
      return ET_LORA_ERR_NONFINITE;
    offset += 4u;
  }

  /* 8. Caller shape binding check. */
  if (file_k != expected_k || file_n != expected_n || file_r != expected_r)
    return ET_LORA_ERR_SHAPE;

  /* 9. Base-weight SHA-256 binding check. */
  uint8_t expected_base_digest[ET_LORA_SHA256_BYTES];
  hash_f32_le(w_base, w_count, expected_base_digest);
  if (memcmp(bytes + 32u, expected_base_digest, ET_LORA_SHA256_BYTES) != 0)
    return ET_LORA_ERR_BASE_HASH;

  /* 10. Commit validated outputs atomically. */
  memcpy(out_a_raw, a_tmp, a_count * sizeof(float));
  memcpy(out_b_raw, b_tmp, b_count * sizeof(float));
  if (out_meta != NULL) {
    out_meta->version_major = ver_major;
    out_meta->version_minor = ver_minor;
    out_meta->k = file_k;
    out_meta->n = file_n;
    out_meta->r = file_r;
    out_meta->alpha = alpha;
    memcpy(out_meta->base_weight_sha256, bytes + 32u, ET_LORA_SHA256_BYTES);
    memcpy(out_meta->checkpoint_sha256, computed_digest, ET_LORA_SHA256_BYTES);
  }
  return ET_LORA_OK;
}

et_lora_status et_lora_checkpoint_parse(
    const uint8_t *bytes, size_t size, uint32_t expected_k, uint32_t expected_n,
    uint32_t expected_r, const float *w_base, et_lora_checkpoint_meta *out_meta,
    float *out_a, float *out_b) {
  return parse_internal(bytes, size, expected_k, expected_n, expected_r,
                        (const void *)w_base, out_meta, (void *)out_a,
                        (void *)out_b);
}

/* ---- Atomic File Save and O_NOFOLLOW Checked File Load ---- */

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

et_lora_status et_lora_checkpoint_save_raw(const char *path,
                                           const et_lora_config *cfg,
                                           const void *w_base_raw,
                                           const void *a_raw,
                                           const void *b_raw) {
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_LORA_ERR_ARGUMENT;

  uint8_t buffer[ET_LORA_CHECKPOINT_MAX_BYTES];
  size_t size = 0u;
  et_lora_status status = serialize_internal(cfg, w_base_raw, a_raw, b_raw,
                                             buffer, sizeof(buffer), &size);
  if (status != ET_LORA_OK)
    return status;

  static const char suffix[] = ".lora-ckpt.XXXXXX";
  const size_t prefix = slash == NULL ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix))
    return ET_LORA_ERR_OVERFLOW;

  char *temp_path = (char *)malloc(prefix + sizeof(suffix));
  if (temp_path == NULL)
    return ET_LORA_ERR_NOMEM;
  memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    return ET_LORA_ERR_IO;
  }

  int ok = write_all(fd, buffer, size) && sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    status = ET_LORA_ERR_IO;
  }
  free(temp_path);
  return status;
}

et_lora_status et_lora_checkpoint_save(const char *path,
                                       const et_lora_config *cfg,
                                       const float *w_base, const float *a,
                                       const float *b) {
  return et_lora_checkpoint_save_raw(path, cfg, (const void *)w_base,
                                     (const void *)a, (const void *)b);
}

et_lora_status et_lora_checkpoint_load_raw(const char *path,
                                           uint32_t expected_k,
                                           uint32_t expected_n,
                                           uint32_t expected_r,
                                           const void *w_base_raw,
                                           et_lora_checkpoint_meta *out_meta,
                                           void *out_a_raw, void *out_b_raw) {
  if (!fp_environment_supported())
    return ET_LORA_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  if (!valid_knr(expected_k, expected_n, expected_r))
    return ET_LORA_ERR_SHAPE;

  const size_t w_count = (size_t)expected_n * (size_t)expected_k;
  const size_t a_count = (size_t)expected_r * (size_t)expected_k;
  const size_t b_count = (size_t)expected_n * (size_t)expected_r;
  et_lora_status status;
  if ((status = check_f32_buffer(w_base_raw, w_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(out_a_raw, a_count)) != ET_LORA_OK ||
      (status = check_f32_buffer(out_b_raw, b_count)) != ET_LORA_OK)
    return status;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_LORA_ERR_SYMLINK : ET_LORA_ERR_IO;

  struct stat meta;
  uint8_t buffer[ET_LORA_CHECKPOINT_MAX_BYTES + 1u];
  size_t used = 0u;
  status = ET_LORA_OK;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    status = ET_LORA_ERR_IO;
  } else if (meta.st_size < 0 ||
             (uintmax_t)meta.st_size > ET_LORA_CHECKPOINT_MAX_BYTES) {
    status = ET_LORA_ERR_FORMAT;
  } else {
    while (used < sizeof(buffer)) {
      const ssize_t count = read(fd, buffer + used, sizeof(buffer) - used);
      if (count == 0)
        break;
      if (count < 0) {
        if (errno == EINTR)
          continue;
        status = ET_LORA_ERR_IO;
        break;
      }
      used += (size_t)count;
    }
    if (status == ET_LORA_OK) {
      if (used > ET_LORA_CHECKPOINT_MAX_BYTES)
        status = ET_LORA_ERR_FORMAT;
      else if (used != (size_t)meta.st_size)
        status = ET_LORA_ERR_TRUNCATED;
    }
  }
  if (close(fd) != 0 && status == ET_LORA_OK)
    status = ET_LORA_ERR_IO;
  if (status != ET_LORA_OK)
    return status;

  return parse_internal(buffer, used, expected_k, expected_n, expected_r,
                        w_base_raw, out_meta, out_a_raw, out_b_raw);
}

et_lora_status et_lora_checkpoint_load(const char *path, uint32_t expected_k,
                                       uint32_t expected_n, uint32_t expected_r,
                                       const float *w_base,
                                       et_lora_checkpoint_meta *out_meta,
                                       float *out_a, float *out_b) {
  return et_lora_checkpoint_load_raw(path, expected_k, expected_n, expected_r,
                                     (const void *)w_base, out_meta,
                                     (void *)out_a, (void *)out_b);
}
