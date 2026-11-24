#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/multi_lora_sgmv_abi.h"

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
               "Multi-LoRA SGMV requires 8-bit bytes and IEEE-754 binary32/64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "Multi-LoRA SGMV requires IEEE-754 binary32/64 evaluation");

/* ---- Standalone SHA-256 ---- */

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

static void sha256_bytes(const uint8_t *data, size_t size, uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  sha256_update(&state, data, size);
  sha256_final(&state, digest);
}

/* ---- Little-endian encoding & IEEE-754 helpers ---- */

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

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

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

typedef struct span_entry {
  const void *ptr;
  size_t bytes;
} span_entry;

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

static et_mlora_status check_f32_buffer(const void *ptr, size_t count) {
  const size_t bytes = count * sizeof(float);
  if (ptr == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!aligned_span(ptr, _Alignof(float), bytes))
    return ET_MLORA_ERR_ALIGNMENT;
  return ET_MLORA_OK;
}

static et_mlora_status check_i32_buffer(const void *ptr, size_t count) {
  const size_t bytes = count * sizeof(int32_t);
  if (ptr == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!aligned_span(ptr, _Alignof(int32_t), bytes))
    return ET_MLORA_ERR_ALIGNMENT;
  return ET_MLORA_OK;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static double pow_u32_f64(double base, uint32_t exp) {
  double acc = 1.0;
  for (uint32_t i = 0u; i < exp; ++i)
    acc *= base;
  return acc;
}

/* ---- Internal Fixed-Slot Registry Structure ---- */

#define ET_MLORA_MAX_A_ELEMS (ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM)
#define ET_MLORA_MAX_B_ELEMS (ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK)
#define ET_MLORA_MAX_W_ELEMS (ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_IN_DIM)

typedef struct mlora_adapter_slot {
  uint32_t active;
  int32_t adapter_id;
  uint32_t generation;
  uint32_t rank;
  float alpha;
  uint32_t pin_count;
  uint32_t adamw_step;
  float a[ET_MLORA_MAX_A_ELEMS];
  float b[ET_MLORA_MAX_B_ELEMS];
  float da[ET_MLORA_MAX_A_ELEMS];
  float db[ET_MLORA_MAX_B_ELEMS];
  float m_a[ET_MLORA_MAX_A_ELEMS];
  float v_a[ET_MLORA_MAX_A_ELEMS];
  float m_b[ET_MLORA_MAX_B_ELEMS];
  float v_b[ET_MLORA_MAX_B_ELEMS];
} mlora_adapter_slot;

struct et_mlora_registry {
  et_mlora_registry_config cfg;
  float w_base[ET_MLORA_MAX_W_ELEMS];
  float bias[ET_MLORA_MAX_OUT_DIM];
  mlora_adapter_slot slots[ET_MLORA_MAX_ADAPTERS];
  /* Preallocated hot-path scratch/staging buffers (ZERO heap allocs in SGMV
   * backward and AdamW!). */
  double dx_acc[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
  double h_seg[ET_MLORA_MAX_M * ET_MLORA_MAX_RANK];
  double dh_seg[ET_MLORA_MAX_M * ET_MLORA_MAX_RANK];
  float dx_stage[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
  float da_stage[ET_MLORA_MAX_ADAPTERS][ET_MLORA_MAX_A_ELEMS];
  float db_stage[ET_MLORA_MAX_ADAPTERS][ET_MLORA_MAX_B_ELEMS];
  float adamw_a_stage[ET_MLORA_MAX_A_ELEMS];
  float adamw_ma_stage[ET_MLORA_MAX_A_ELEMS];
  float adamw_va_stage[ET_MLORA_MAX_A_ELEMS];
  float adamw_b_stage[ET_MLORA_MAX_B_ELEMS];
  float adamw_mb_stage[ET_MLORA_MAX_B_ELEMS];
  float adamw_vb_stage[ET_MLORA_MAX_B_ELEMS];
};

static void compute_base_digest_internal(const et_mlora_registry_config *cfg,
                                         const float *w_base, const float *bias,
                                         uint8_t digest[ET_MLORA_SHA256_BYTES]) {
  sha256_state state;
  uint8_t le[4];
  sha256_init(&state);
  encode_u32_le(le, cfg->in_dim);
  sha256_update(&state, le, 4u);
  encode_u32_le(le, cfg->out_dim);
  sha256_update(&state, le, 4u);
  encode_u32_le(le, cfg->has_bias);
  sha256_update(&state, le, 4u);
  const size_t w_count = (size_t)cfg->out_dim * (size_t)cfg->in_dim;
  for (size_t i = 0u; i < w_count; ++i) {
    encode_u32_le(le, f32_to_bits(w_base[i]));
    sha256_update(&state, le, 4u);
  }
  if (cfg->has_bias != 0u) {
    for (size_t i = 0u; i < (size_t)cfg->out_dim; ++i) {
      encode_u32_le(le, f32_to_bits(bias[i]));
      sha256_update(&state, le, 4u);
    }
  }
  sha256_final(&state, digest);
}

static int32_t find_active_slot(const et_mlora_registry *reg,
                                int32_t adapter_id) {
  if (adapter_id < 0)
    return -1;
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    if (reg->slots[s].active != 0u && reg->slots[s].adapter_id == adapter_id)
      return (int32_t)s;
  }
  return -1;
}

static int32_t find_free_slot(const et_mlora_registry *reg) {
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    if (reg->slots[s].active == 0u)
      return (int32_t)s;
  }
  return -1;
}

static void fill_adapter_info(const mlora_adapter_slot *slot, uint32_t idx,
                              et_mlora_adapter_info *out) {
  out->adapter_id = slot->adapter_id;
  out->slot_index = idx;
  out->generation = slot->generation;
  out->rank = slot->rank;
  out->alpha = slot->alpha;
  out->pin_count = slot->pin_count;
  out->active = slot->active;
  out->adamw_step = slot->adamw_step;
}

static et_mlora_status validate_rank_alpha(const et_mlora_registry *reg,
                                           uint32_t rank, float alpha) {
  if (rank < 1u || rank > reg->cfg.max_rank)
    return ET_MLORA_ERR_SHAPE;
  if (!isfinite(alpha))
    return ET_MLORA_ERR_NONFINITE;
  if (!(alpha > 0.0f))
    return ET_MLORA_ERR_ARGUMENT;
  const double scale = (double)alpha / (double)rank;
  if (!isfinite(scale) || !(scale > 0.0))
    return ET_MLORA_ERR_NONFINITE;
  return ET_MLORA_OK;
}

static uint32_t collect_sorted_active_slots(const et_mlora_registry *reg,
                                            uint32_t out_slots[ET_MLORA_MAX_ADAPTERS]) {
  uint32_t count = 0u;
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    if (reg->slots[s].active != 0u)
      out_slots[count++] = s;
  }
  /* Insertion sort by ascending adapter_id (deterministic canonical order). */
  for (uint32_t i = 1u; i < count; ++i) {
    const uint32_t key = out_slots[i];
    const int32_t key_id = reg->slots[key].adapter_id;
    uint32_t j = i;
    while (j > 0u && reg->slots[out_slots[j - 1u]].adapter_id > key_id) {
      out_slots[j] = out_slots[j - 1u];
      --j;
    }
    out_slots[j] = key;
  }
  return count;
}

/* ---- Public API: Registry Lifecycle ---- */

et_mlora_status et_mlora_validate_registry_config_v1(
    const et_mlora_registry_config *cfg) {
  if (cfg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (cfg->reserved != 0u)
    return ET_MLORA_ERR_ARGUMENT;
  if (cfg->has_bias > 1u)
    return ET_MLORA_ERR_ARGUMENT;
  if (cfg->in_dim < 1u || cfg->in_dim > ET_MLORA_MAX_IN_DIM ||
      cfg->out_dim < 1u || cfg->out_dim > ET_MLORA_MAX_OUT_DIM ||
      cfg->max_adapters < 1u || cfg->max_adapters > ET_MLORA_MAX_ADAPTERS)
    return ET_MLORA_ERR_SHAPE;
  const uint32_t min_dim =
      cfg->in_dim < cfg->out_dim ? cfg->in_dim : cfg->out_dim;
  const uint32_t max_r_cap =
      min_dim < ET_MLORA_MAX_RANK ? min_dim : ET_MLORA_MAX_RANK;
  if (cfg->max_rank < 1u || cfg->max_rank > max_r_cap)
    return ET_MLORA_ERR_SHAPE;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_registry_create_v1(
    const et_mlora_registry_config *cfg, const float *w_base, const float *bias,
    et_mlora_registry **out_reg) {
  et_mlora_status status = et_mlora_validate_registry_config_v1(cfg);
  if (status != ET_MLORA_OK)
    return status;
  if (out_reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;

  const size_t w_count = (size_t)cfg->out_dim * (size_t)cfg->in_dim;
  status = check_f32_buffer(w_base, w_count);
  if (status != ET_MLORA_OK)
    return status;
  if (cfg->has_bias != 0u) {
    status = check_f32_buffer(bias, (size_t)cfg->out_dim);
    if (status != ET_MLORA_OK)
      return status;
  } else if (bias != NULL) {
    return ET_MLORA_ERR_ARGUMENT;
  }

  if (!all_finite_f32(w_base, w_count))
    return ET_MLORA_ERR_NONFINITE;
  if (cfg->has_bias != 0u && !all_finite_f32(bias, (size_t)cfg->out_dim))
    return ET_MLORA_ERR_NONFINITE;

  et_mlora_registry *reg =
      (et_mlora_registry *)calloc(1u, sizeof(et_mlora_registry));
  if (reg == NULL)
    return ET_MLORA_ERR_NOMEM;

  reg->cfg = *cfg;
  memcpy(reg->w_base, w_base, w_count * sizeof(float));
  if (cfg->has_bias != 0u)
    memcpy(reg->bias, bias, (size_t)cfg->out_dim * sizeof(float));

  *out_reg = reg;
  return ET_MLORA_OK;
}

void et_mlora_registry_destroy_v1(et_mlora_registry *reg) {
  free(reg);
}

et_mlora_status et_mlora_registry_config_get_v1(
    const et_mlora_registry *reg, et_mlora_registry_config *out_cfg) {
  if (reg == NULL || out_cfg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (spans_overlap(reg, sizeof(*reg), out_cfg, sizeof(*out_cfg)))
    return ET_MLORA_ERR_ALIAS;
  *out_cfg = reg->cfg;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_get_base_sha256_v1(
    const et_mlora_registry *reg, uint8_t out_digest[ET_MLORA_SHA256_BYTES]) {
  if (reg == NULL || out_digest == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (!aligned_span(out_digest, 1u, ET_MLORA_SHA256_BYTES))
    return ET_MLORA_ERR_ALIGNMENT;
  if (spans_overlap(reg, sizeof(*reg), out_digest, ET_MLORA_SHA256_BYTES))
    return ET_MLORA_ERR_ALIAS;

  uint8_t digest[ET_MLORA_SHA256_BYTES];
  compute_base_digest_internal(&reg->cfg, reg->w_base, reg->bias, digest);
  memcpy(out_digest, digest, sizeof(digest));
  return ET_MLORA_OK;
}

/* ---- Public API: Heterogeneous-Rank Adapter Pool (Zero-Alloc) ---- */

et_mlora_status et_mlora_register_adapter_v1(et_mlora_registry *reg,
                                             int32_t adapter_id, uint32_t rank,
                                             float alpha, const float *a,
                                             const float *b,
                                             et_mlora_adapter_info *out_info) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  et_mlora_status status = validate_rank_alpha(reg, rank, alpha);
  if (status != ET_MLORA_OK)
    return status;

  const size_t a_count = (size_t)rank * (size_t)reg->cfg.in_dim;
  const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)rank;
  if ((status = check_f32_buffer(a, a_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(b, b_count)) != ET_MLORA_OK)
    return status;

  if (spans_overlap(reg, sizeof(*reg), a, a_count * sizeof(float)) ||
      spans_overlap(reg, sizeof(*reg), b, b_count * sizeof(float)))
    return ET_MLORA_ERR_ALIAS;
  if (out_info != NULL) {
    if (!aligned_span(out_info, _Alignof(et_mlora_adapter_info),
                      sizeof(*out_info)))
      return ET_MLORA_ERR_ALIGNMENT;
    if (spans_overlap(out_info, sizeof(*out_info), reg, sizeof(*reg)) ||
        spans_overlap(out_info, sizeof(*out_info), a,
                      a_count * sizeof(float)) ||
        spans_overlap(out_info, sizeof(*out_info), b, b_count * sizeof(float)))
      return ET_MLORA_ERR_ALIAS;
  }

  if (!all_finite_f32(a, a_count) || !all_finite_f32(b, b_count))
    return ET_MLORA_ERR_NONFINITE;

  if (find_active_slot(reg, adapter_id) >= 0)
    return ET_MLORA_ERR_ALREADY_EXISTS;

  const int32_t slot_idx = find_free_slot(reg);
  if (slot_idx < 0)
    return ET_MLORA_ERR_FULL;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  if (slot->generation == UINT32_MAX)
    return ET_MLORA_ERR_OVERFLOW;

  const uint32_t next_gen = slot->generation + 1u;
  memset(slot, 0, sizeof(*slot));
  slot->active = 1u;
  slot->adapter_id = adapter_id;
  slot->generation = next_gen;
  slot->rank = rank;
  slot->alpha = alpha;
  slot->pin_count = 0u;
  slot->adamw_step = 0u;
  memcpy(slot->a, a, a_count * sizeof(float));
  memcpy(slot->b, b, b_count * sizeof(float));

  if (out_info != NULL)
    fill_adapter_info(slot, (uint32_t)slot_idx, out_info);
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_register_seeded_adapter_v1(
    et_mlora_registry *reg, int32_t adapter_id, uint32_t rank, float alpha,
    uint64_t seed, et_mlora_adapter_info *out_info) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  et_mlora_status status = validate_rank_alpha(reg, rank, alpha);
  if (status != ET_MLORA_OK)
    return status;

  if (out_info != NULL) {
    if (!aligned_span(out_info, _Alignof(et_mlora_adapter_info),
                      sizeof(*out_info)))
      return ET_MLORA_ERR_ALIGNMENT;
    if (spans_overlap(out_info, sizeof(*out_info), reg, sizeof(*reg)))
      return ET_MLORA_ERR_ALIAS;
  }

  if (find_active_slot(reg, adapter_id) >= 0)
    return ET_MLORA_ERR_ALREADY_EXISTS;
  const int32_t slot_idx = find_free_slot(reg);
  if (slot_idx < 0)
    return ET_MLORA_ERR_FULL;
  if (reg->slots[slot_idx].generation == UINT32_MAX)
    return ET_MLORA_ERR_OVERFLOW;

  const size_t a_count = (size_t)rank * (size_t)reg->cfg.in_dim;
  const double bound = 0.25 / sqrt((double)reg->cfg.in_dim);
  if (!isfinite(bound) || !(bound > 0.0))
    return ET_MLORA_ERR_NONFINITE;

  float a_tmp[ET_MLORA_MAX_A_ELEMS];
  uint64_t rng = seed ^ UINT64_C(0x4d4c6f5241313030) ^
                 ((uint64_t)(uint32_t)adapter_id * UINT64_C(0x9e3779b97f4a7c15));
  for (size_t i = 0u; i < a_count; ++i) {
    const uint64_t word = splitmix64_next(&rng);
    const int32_t bucket = (int32_t)(word % UINT64_C(254)) - 127;
    const double signed_unit = (bucket == 0 ? 1.0 : (double)bucket) / 127.0;
    const float val = (float)(signed_unit * bound);
    if (!isfinite(val))
      return ET_MLORA_ERR_NONFINITE;
    a_tmp[i] = val;
  }

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  const uint32_t next_gen = slot->generation + 1u;
  memset(slot, 0, sizeof(*slot));
  slot->active = 1u;
  slot->adapter_id = adapter_id;
  slot->generation = next_gen;
  slot->rank = rank;
  slot->alpha = alpha;
  slot->pin_count = 0u;
  slot->adamw_step = 0u;
  memcpy(slot->a, a_tmp, a_count * sizeof(float));
  /* slot->b is zero-filled (+0.0f) by memset above. */

  if (out_info != NULL)
    fill_adapter_info(slot, (uint32_t)slot_idx, out_info);
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_unregister_adapter_v1(et_mlora_registry *reg,
                                               int32_t adapter_id) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  if (slot->pin_count > 0u)
    return ET_MLORA_ERR_BUSY;

  const uint32_t saved_gen = slot->generation;
  memset(slot, 0, sizeof(*slot));
  slot->generation = saved_gen;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_pin_adapter_v1(et_mlora_registry *reg,
                                        int32_t adapter_id,
                                        uint32_t *out_pin_count) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (out_pin_count != NULL &&
      spans_overlap(reg, sizeof(*reg), out_pin_count, sizeof(*out_pin_count)))
    return ET_MLORA_ERR_ALIAS;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  if (slot->pin_count == UINT32_MAX)
    return ET_MLORA_ERR_OVERFLOW;
  slot->pin_count += 1u;
  if (out_pin_count != NULL)
    *out_pin_count = slot->pin_count;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_unpin_adapter_v1(et_mlora_registry *reg,
                                          int32_t adapter_id,
                                          uint32_t *out_pin_count) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (out_pin_count != NULL &&
      spans_overlap(reg, sizeof(*reg), out_pin_count, sizeof(*out_pin_count)))
    return ET_MLORA_ERR_ALIAS;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  if (slot->pin_count == 0u)
    return ET_MLORA_ERR_UNDERFLOW;
  slot->pin_count -= 1u;
  if (out_pin_count != NULL)
    *out_pin_count = slot->pin_count;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_get_adapter_info_v1(const et_mlora_registry *reg,
                                             int32_t adapter_id,
                                             et_mlora_adapter_info *out_info) {
  if (reg == NULL || out_info == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (!aligned_span(out_info, _Alignof(et_mlora_adapter_info),
                    sizeof(*out_info)))
    return ET_MLORA_ERR_ALIGNMENT;
  if (spans_overlap(reg, sizeof(*reg), out_info, sizeof(*out_info)))
    return ET_MLORA_ERR_ALIAS;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  fill_adapter_info(&reg->slots[slot_idx], (uint32_t)slot_idx, out_info);
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_list_adapters_v1(const et_mlora_registry *reg,
                                          et_mlora_adapter_info *out_infos,
                                          uint32_t capacity,
                                          uint32_t *out_count) {
  if (reg == NULL || out_count == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (spans_overlap(reg, sizeof(*reg), out_count, sizeof(*out_count)))
    return ET_MLORA_ERR_ALIAS;

  uint32_t sorted[ET_MLORA_MAX_ADAPTERS];
  const uint32_t count = collect_sorted_active_slots(reg, sorted);
  if (out_infos != NULL) {
    if (capacity < count)
      return ET_MLORA_ERR_ARGUMENT;
    if (count > 0u) {
      const size_t bytes = (size_t)count * sizeof(et_mlora_adapter_info);
      if (!aligned_span(out_infos, _Alignof(et_mlora_adapter_info), bytes))
        return ET_MLORA_ERR_ALIGNMENT;
      if (spans_overlap(reg, sizeof(*reg), out_infos, bytes) ||
          spans_overlap(out_count, sizeof(*out_count), out_infos, bytes))
        return ET_MLORA_ERR_ALIAS;
      for (uint32_t i = 0u; i < count; ++i)
        fill_adapter_info(&reg->slots[sorted[i]], sorted[i], &out_infos[i]);
    }
  }
  *out_count = count;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_get_adapter_weights_v1(const et_mlora_registry *reg,
                                                int32_t adapter_id,
                                                float *out_a, float *out_b) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  const mlora_adapter_slot *slot = &reg->slots[slot_idx];
  const size_t a_count = (size_t)slot->rank * (size_t)reg->cfg.in_dim;
  const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)slot->rank;
  et_mlora_status status;
  if ((status = check_f32_buffer(out_a, a_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(out_b, b_count)) != ET_MLORA_OK)
    return status;

  const span_entry spans[] = {{reg, sizeof(*reg)},
                              {out_a, a_count * sizeof(float)},
                              {out_b, b_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])))
    return ET_MLORA_ERR_ALIAS;

  memcpy(out_a, slot->a, a_count * sizeof(float));
  memcpy(out_b, slot->b, b_count * sizeof(float));
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_set_adapter_weights_v1(et_mlora_registry *reg,
                                                int32_t adapter_id,
                                                const float *a,
                                                const float *b) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  const size_t a_count = (size_t)slot->rank * (size_t)reg->cfg.in_dim;
  const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)slot->rank;
  et_mlora_status status;
  if ((status = check_f32_buffer(a, a_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(b, b_count)) != ET_MLORA_OK)
    return status;
  if (spans_overlap(reg, sizeof(*reg), a, a_count * sizeof(float)) ||
      spans_overlap(reg, sizeof(*reg), b, b_count * sizeof(float)))
    return ET_MLORA_ERR_ALIAS;

  if (!all_finite_f32(a, a_count) || !all_finite_f32(b, b_count))
    return ET_MLORA_ERR_NONFINITE;

  memcpy(slot->a, a, a_count * sizeof(float));
  memcpy(slot->b, b, b_count * sizeof(float));
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_get_adapter_grads_v1(const et_mlora_registry *reg,
                                              int32_t adapter_id, float *out_da,
                                              float *out_db) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  const mlora_adapter_slot *slot = &reg->slots[slot_idx];
  const size_t a_count = (size_t)slot->rank * (size_t)reg->cfg.in_dim;
  const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)slot->rank;
  et_mlora_status status;
  if ((status = check_f32_buffer(out_da, a_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(out_db, b_count)) != ET_MLORA_OK)
    return status;

  const span_entry spans[] = {{reg, sizeof(*reg)},
                              {out_da, a_count * sizeof(float)},
                              {out_db, b_count * sizeof(float)}};
  if (any_spans_overlap(spans, sizeof(spans) / sizeof(spans[0])))
    return ET_MLORA_ERR_ALIAS;

  memcpy(out_da, slot->da, a_count * sizeof(float));
  memcpy(out_db, slot->db, b_count * sizeof(float));
  return ET_MLORA_OK;
}

/* ---- Segmented Dispatch Helper (Zero-Alloc) ---- */

static et_mlora_status build_segments(
    const et_mlora_registry *reg, uint32_t m, const int32_t *row_adapter_ids,
    int32_t row_slots[ET_MLORA_MAX_M],
    uint32_t seg_counts[ET_MLORA_MAX_ADAPTERS],
    uint32_t seg_offsets[ET_MLORA_MAX_ADAPTERS + 1u],
    uint32_t seg_rows[ET_MLORA_MAX_M], et_mlora_sgmv_stats *out_stats) {
  const uint32_t max_adapters = reg->cfg.max_adapters;
  uint32_t seg_cursor[ET_MLORA_MAX_ADAPTERS];
  for (uint32_t s = 0u; s < max_adapters; ++s) {
    seg_counts[s] = 0u;
    seg_cursor[s] = 0u;
  }

  uint32_t base_only = 0u;
  uint32_t adapted = 0u;
  for (uint32_t i = 0u; i < m; ++i) {
    const int32_t aid = row_adapter_ids[i];
    if (aid == ET_MLORA_BASE_ONLY_ID) {
      row_slots[i] = -1;
      ++base_only;
    } else if (aid < ET_MLORA_BASE_ONLY_ID) {
      return ET_MLORA_ERR_ARGUMENT;
    } else {
      const int32_t s = find_active_slot(reg, aid);
      if (s < 0)
        return ET_MLORA_ERR_NOT_FOUND;
      row_slots[i] = s;
      seg_counts[s] += 1u;
      ++adapted;
    }
  }

  uint32_t active_segments = 0u;
  uint32_t running = 0u;
  for (uint32_t s = 0u; s < max_adapters; ++s) {
    seg_offsets[s] = running;
    seg_cursor[s] = running;
    running += seg_counts[s];
    if (seg_counts[s] > 0u)
      ++active_segments;
  }
  seg_offsets[max_adapters] = running;

  /* Preserve ascending row index order within every adapter segment. */
  for (uint32_t i = 0u; i < m; ++i) {
    const int32_t s = row_slots[i];
    if (s >= 0)
      seg_rows[seg_cursor[s]++] = i;
  }

  out_stats->batch_rows = m;
  out_stats->base_only_rows = base_only;
  out_stats->adapted_rows = adapted;
  out_stats->active_segments = active_segments;
  return ET_MLORA_OK;
}

/* ---- Public API: Batched Heterogeneous SGMV Forward (Zero-Alloc) ---- */

et_mlora_status et_mlora_sgmv_forward_v1(const et_mlora_registry *reg,
                                         uint32_t m, const float *x,
                                         const int32_t *row_adapter_ids,
                                         float *y,
                                         et_mlora_sgmv_stats *out_stats) {
  if (reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (m < 1u || m > ET_MLORA_MAX_M)
    return ET_MLORA_ERR_SHAPE;

  const uint32_t in_dim = reg->cfg.in_dim;
  const uint32_t out_dim = reg->cfg.out_dim;
  const uint32_t has_bias = reg->cfg.has_bias;
  const size_t x_count = (size_t)m * (size_t)in_dim;
  const size_t y_count = (size_t)m * (size_t)out_dim;

  et_mlora_status status;
  if ((status = check_f32_buffer(x, x_count)) != ET_MLORA_OK ||
      (status = check_i32_buffer(row_adapter_ids, (size_t)m)) != ET_MLORA_OK ||
      (status = check_f32_buffer(y, y_count)) != ET_MLORA_OK)
    return status;

  if (spans_overlap(y, y_count * sizeof(float), x, x_count * sizeof(float)) ||
      spans_overlap(y, y_count * sizeof(float), row_adapter_ids,
                    (size_t)m * sizeof(int32_t)) ||
      spans_overlap(y, y_count * sizeof(float), reg, sizeof(*reg)))
    return ET_MLORA_ERR_ALIAS;
  if (out_stats != NULL) {
    if (!aligned_span(out_stats, _Alignof(et_mlora_sgmv_stats),
                      sizeof(*out_stats)))
      return ET_MLORA_ERR_ALIGNMENT;
    if (spans_overlap(out_stats, sizeof(*out_stats), y,
                      y_count * sizeof(float)) ||
        spans_overlap(out_stats, sizeof(*out_stats), x,
                      x_count * sizeof(float)) ||
        spans_overlap(out_stats, sizeof(*out_stats), row_adapter_ids,
                      (size_t)m * sizeof(int32_t)) ||
        spans_overlap(out_stats, sizeof(*out_stats), reg, sizeof(*reg)))
      return ET_MLORA_ERR_ALIAS;
  }

  if (!all_finite_f32(x, x_count))
    return ET_MLORA_ERR_NONFINITE;

  int32_t row_slots[ET_MLORA_MAX_M];
  uint32_t seg_counts[ET_MLORA_MAX_ADAPTERS];
  uint32_t seg_offsets[ET_MLORA_MAX_ADAPTERS + 1u];
  uint32_t seg_rows[ET_MLORA_MAX_M];
  et_mlora_sgmv_stats stats;
  status = build_segments(reg, m, row_adapter_ids, row_slots, seg_counts,
                          seg_offsets, seg_rows, &stats);
  if (status != ET_MLORA_OK)
    return status;

  float y_stage[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  double h_seg[ET_MLORA_MAX_M * ET_MLORA_MAX_RANK];

  /* 1. Base-only rows (adapter_id == -1). */
  for (uint32_t i = 0u; i < m; ++i) {
    if (row_slots[i] >= 0)
      continue;
    const float *x_row = &x[(size_t)i * in_dim];
    float *y_row = &y_stage[(size_t)i * out_dim];
    for (uint32_t o = 0u; o < out_dim; ++o) {
      const float *w_row = &reg->w_base[(size_t)o * in_dim];
      double acc = 0.0;
      for (uint32_t k = 0u; k < in_dim; ++k)
        acc += (double)w_row[k] * (double)x_row[k];
      if (has_bias != 0u)
        acc += (double)reg->bias[o];
      const float val = (float)acc;
      if (!isfinite(val))
        return ET_MLORA_ERR_NONFINITE;
      y_row[o] = val;
    }
  }

  /* 2. Segmented Gather-Matrix-Multiply (SGMV) for each active adapter segment. */
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    const uint32_t seg_n = seg_counts[s];
    if (seg_n == 0u)
      continue;
    const uint32_t start = seg_offsets[s];
    const mlora_adapter_slot *slot = &reg->slots[s];
    const uint32_t r_a = slot->rank;
    const double scale_a = (double)slot->alpha / (double)r_a;

    /* SGMV Shrink: H_seg[p, r] = sum_k A_a[r, k] * X[seg_rows[p], k]. */
    for (uint32_t p = 0u; p < seg_n; ++p) {
      const uint32_t row = seg_rows[start + p];
      const float *x_row = &x[(size_t)row * in_dim];
      double *h_row = &h_seg[(size_t)p * r_a];
      for (uint32_t r = 0u; r < r_a; ++r) {
        const float *a_row = &slot->a[(size_t)r * in_dim];
        double h_acc = 0.0;
        for (uint32_t k = 0u; k < in_dim; ++k)
          h_acc += (double)a_row[k] * (double)x_row[k];
        if (!isfinite(h_acc))
          return ET_MLORA_ERR_NONFINITE;
        h_row[r] = h_acc;
      }
    }

    /* SGMV Expand + Base Add: Y[row, o] = base + scale_a * sum_r B_a[o, r] * H[p, r]. */
    for (uint32_t p = 0u; p < seg_n; ++p) {
      const uint32_t row = seg_rows[start + p];
      const float *x_row = &x[(size_t)row * in_dim];
      const double *h_row = &h_seg[(size_t)p * r_a];
      float *y_row = &y_stage[(size_t)row * out_dim];
      for (uint32_t o = 0u; o < out_dim; ++o) {
        const float *w_row = &reg->w_base[(size_t)o * in_dim];
        const float *b_row = &slot->b[(size_t)o * r_a];
        double base_acc = 0.0;
        for (uint32_t k = 0u; k < in_dim; ++k)
          base_acc += (double)w_row[k] * (double)x_row[k];
        if (has_bias != 0u)
          base_acc += (double)reg->bias[o];
        double lora_acc = 0.0;
        for (uint32_t r = 0u; r < r_a; ++r)
          lora_acc += (double)b_row[r] * h_row[r];
        const double total = base_acc + scale_a * lora_acc;
        const float val = (float)total;
        if (!isfinite(val))
          return ET_MLORA_ERR_NONFINITE;
        y_row[o] = val;
      }
    }
  }

  /* Atomic staged commit. */
  memcpy(y, y_stage, y_count * sizeof(float));
  if (out_stats != NULL)
    *out_stats = stats;
  return ET_MLORA_OK;
}

et_mlora_status et_mlora_single_adapter_forward_v1(
    const et_mlora_registry *reg, int32_t adapter_id, uint32_t m,
    const float *x, float *y) {
  if (reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (m < 1u || m > ET_MLORA_MAX_M)
    return ET_MLORA_ERR_SHAPE;

  const uint32_t in_dim = reg->cfg.in_dim;
  const uint32_t out_dim = reg->cfg.out_dim;
  const uint32_t has_bias = reg->cfg.has_bias;
  const size_t x_count = (size_t)m * (size_t)in_dim;
  const size_t y_count = (size_t)m * (size_t)out_dim;

  et_mlora_status status;
  if ((status = check_f32_buffer(x, x_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(y, y_count)) != ET_MLORA_OK)
    return status;
  if (spans_overlap(y, y_count * sizeof(float), x, x_count * sizeof(float)) ||
      spans_overlap(y, y_count * sizeof(float), reg, sizeof(*reg)))
    return ET_MLORA_ERR_ALIAS;
  if (!all_finite_f32(x, x_count))
    return ET_MLORA_ERR_NONFINITE;

  const mlora_adapter_slot *slot = NULL;
  if (adapter_id == ET_MLORA_BASE_ONLY_ID) {
    slot = NULL;
  } else if (adapter_id < ET_MLORA_BASE_ONLY_ID) {
    return ET_MLORA_ERR_ARGUMENT;
  } else {
    const int32_t s = find_active_slot(reg, adapter_id);
    if (s < 0)
      return ET_MLORA_ERR_NOT_FOUND;
    slot = &reg->slots[s];
  }

  float y_stage[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  double h_row[ET_MLORA_MAX_RANK];

  for (uint32_t i = 0u; i < m; ++i) {
    const float *x_row = &x[(size_t)i * in_dim];
    float *y_row = &y_stage[(size_t)i * out_dim];
    uint32_t r_a = 0u;
    double scale_a = 0.0;
    if (slot != NULL) {
      r_a = slot->rank;
      scale_a = (double)slot->alpha / (double)r_a;
      for (uint32_t r = 0u; r < r_a; ++r) {
        const float *a_row = &slot->a[(size_t)r * in_dim];
        double h_acc = 0.0;
        for (uint32_t k = 0u; k < in_dim; ++k)
          h_acc += (double)a_row[k] * (double)x_row[k];
        if (!isfinite(h_acc))
          return ET_MLORA_ERR_NONFINITE;
        h_row[r] = h_acc;
      }
    }
    for (uint32_t o = 0u; o < out_dim; ++o) {
      const float *w_row = &reg->w_base[(size_t)o * in_dim];
      double base_acc = 0.0;
      for (uint32_t k = 0u; k < in_dim; ++k)
        base_acc += (double)w_row[k] * (double)x_row[k];
      if (has_bias != 0u)
        base_acc += (double)reg->bias[o];
      double total = base_acc;
      if (slot != NULL) {
        const float *b_row = &slot->b[(size_t)o * r_a];
        double lora_acc = 0.0;
        for (uint32_t r = 0u; r < r_a; ++r)
          lora_acc += (double)b_row[r] * h_row[r];
        total = base_acc + scale_a * lora_acc;
      }
      const float val = (float)total;
      if (!isfinite(val))
        return ET_MLORA_ERR_NONFINITE;
      y_row[o] = val;
    }
  }

  memcpy(y, y_stage, y_count * sizeof(float));
  return ET_MLORA_OK;
}

/* ---- Public API: Multi-Adapter Analytic Backward (Zero-Alloc) ---- */

et_mlora_status et_mlora_sgmv_backward_v1(et_mlora_registry *reg, uint32_t m,
                                          const float *x,
                                          const int32_t *row_adapter_ids,
                                          const float *dy, float *dx,
                                          et_mlora_sgmv_stats *out_stats) {
  if (reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (m < 1u || m > ET_MLORA_MAX_M)
    return ET_MLORA_ERR_SHAPE;

  const uint32_t in_dim = reg->cfg.in_dim;
  const uint32_t out_dim = reg->cfg.out_dim;
  const size_t x_count = (size_t)m * (size_t)in_dim;
  const size_t dy_count = (size_t)m * (size_t)out_dim;

  et_mlora_status status;
  if ((status = check_f32_buffer(x, x_count)) != ET_MLORA_OK ||
      (status = check_i32_buffer(row_adapter_ids, (size_t)m)) != ET_MLORA_OK ||
      (status = check_f32_buffer(dy, dy_count)) != ET_MLORA_OK ||
      (status = check_f32_buffer(dx, x_count)) != ET_MLORA_OK)
    return status;

  if (spans_overlap(dx, x_count * sizeof(float), x, x_count * sizeof(float)) ||
      spans_overlap(dx, x_count * sizeof(float), dy, dy_count * sizeof(float)) ||
      spans_overlap(dx, x_count * sizeof(float), row_adapter_ids,
                    (size_t)m * sizeof(int32_t)) ||
      spans_overlap(dx, x_count * sizeof(float), reg, sizeof(*reg)) ||
      spans_overlap(reg, sizeof(*reg), x, x_count * sizeof(float)) ||
      spans_overlap(reg, sizeof(*reg), dy, dy_count * sizeof(float)) ||
      spans_overlap(reg, sizeof(*reg), row_adapter_ids,
                    (size_t)m * sizeof(int32_t)))
    return ET_MLORA_ERR_ALIAS;

  if (out_stats != NULL) {
    if (!aligned_span(out_stats, _Alignof(et_mlora_sgmv_stats),
                      sizeof(*out_stats)))
      return ET_MLORA_ERR_ALIGNMENT;
    if (spans_overlap(out_stats, sizeof(*out_stats), dx,
                      x_count * sizeof(float)) ||
        spans_overlap(out_stats, sizeof(*out_stats), x,
                      x_count * sizeof(float)) ||
        spans_overlap(out_stats, sizeof(*out_stats), dy,
                      dy_count * sizeof(float)) ||
        spans_overlap(out_stats, sizeof(*out_stats), row_adapter_ids,
                      (size_t)m * sizeof(int32_t)) ||
        spans_overlap(out_stats, sizeof(*out_stats), reg, sizeof(*reg)))
      return ET_MLORA_ERR_ALIAS;
  }

  if (!all_finite_f32(x, x_count) || !all_finite_f32(dy, dy_count))
    return ET_MLORA_ERR_NONFINITE;

  int32_t row_slots[ET_MLORA_MAX_M];
  uint32_t seg_counts[ET_MLORA_MAX_ADAPTERS];
  uint32_t seg_offsets[ET_MLORA_MAX_ADAPTERS + 1u];
  uint32_t seg_rows[ET_MLORA_MAX_M];
  et_mlora_sgmv_stats stats;
  status = build_segments(reg, m, row_adapter_ids, row_slots, seg_counts,
                          seg_offsets, seg_rows, &stats);
  if (status != ET_MLORA_OK)
    return status;

  /* 1. Base backward for all rows: dx_acc[i, k] = sum_o dY[i, o] * W_base[o, k]. */
  for (uint32_t i = 0u; i < m; ++i) {
    const float *dy_row = &dy[(size_t)i * out_dim];
    double *dx_acc_row = &reg->dx_acc[(size_t)i * in_dim];
    float *dx_out_row = &reg->dx_stage[(size_t)i * in_dim];
    for (uint32_t k = 0u; k < in_dim; ++k) {
      double acc = 0.0;
      for (uint32_t o = 0u; o < out_dim; ++o)
        acc += (double)dy_row[o] * (double)reg->w_base[(size_t)o * in_dim + k];
      if (!isfinite(acc))
        return ET_MLORA_ERR_NONFINITE;
      dx_acc_row[k] = acc;
      if (row_slots[i] < 0) {
        const float val = (float)acc;
        if (!isfinite(val))
          return ET_MLORA_ERR_NONFINITE;
        dx_out_row[k] = val;
      }
    }
  }

  /* 2. Segmented per-adapter backward; idle adapters explicitly get +0.0f. */
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    const mlora_adapter_slot *slot = &reg->slots[s];
    if (slot->active == 0u)
      continue;

    const uint32_t r_a = slot->rank;
    const size_t a_count = (size_t)r_a * (size_t)in_dim;
    const size_t b_count = (size_t)out_dim * (size_t)r_a;
    const uint32_t seg_n = seg_counts[s];

    if (seg_n == 0u) {
      /* Zero cross-adapter gradient bleed: idle adapter receives exact +0.0f. */
      memset(reg->da_stage[s], 0, a_count * sizeof(float));
      memset(reg->db_stage[s], 0, b_count * sizeof(float));
      continue;
    }

    const uint32_t start = seg_offsets[s];
    const double scale_a = (double)slot->alpha / (double)r_a;

    /* 2a. Recompute H_seg[p, r] and dH_seg[p, r] in binary64. */
    for (uint32_t p = 0u; p < seg_n; ++p) {
      const uint32_t row = seg_rows[start + p];
      const float *x_row = &x[(size_t)row * in_dim];
      const float *dy_row = &dy[(size_t)row * out_dim];
      double *h_row = &reg->h_seg[(size_t)p * r_a];
      double *dh_row = &reg->dh_seg[(size_t)p * r_a];
      for (uint32_t r = 0u; r < r_a; ++r) {
        const float *a_row = &slot->a[(size_t)r * in_dim];
        double h_acc = 0.0;
        for (uint32_t k = 0u; k < in_dim; ++k)
          h_acc += (double)a_row[k] * (double)x_row[k];
        double dh_acc = 0.0;
        for (uint32_t o = 0u; o < out_dim; ++o)
          dh_acc += (double)dy_row[o] * (double)slot->b[(size_t)o * r_a + r];
        dh_acc *= scale_a;
        if (!isfinite(h_acc) || !isfinite(dh_acc))
          return ET_MLORA_ERR_NONFINITE;
        h_row[r] = h_acc;
        dh_row[r] = dh_acc;
      }
    }

    /* 2b. dB_a[o, r] = scale_a * sum_{p in seg} dY[row_p, o] * H_seg[p, r]. */
    for (uint32_t o = 0u; o < out_dim; ++o) {
      for (uint32_t r = 0u; r < r_a; ++r) {
        double acc = 0.0;
        for (uint32_t p = 0u; p < seg_n; ++p) {
          const uint32_t row = seg_rows[start + p];
          acc += (double)dy[(size_t)row * out_dim + o] *
                 reg->h_seg[(size_t)p * r_a + r];
        }
        const float val = (float)(scale_a * acc);
        if (!isfinite(val))
          return ET_MLORA_ERR_NONFINITE;
        reg->db_stage[s][(size_t)o * r_a + r] = val;
      }
    }

    /* 2c. dA_a[r, k] = sum_{p in seg} dH_seg[p, r] * X[row_p, k]. */
    for (uint32_t r = 0u; r < r_a; ++r) {
      for (uint32_t k = 0u; k < in_dim; ++k) {
        double acc = 0.0;
        for (uint32_t p = 0u; p < seg_n; ++p) {
          const uint32_t row = seg_rows[start + p];
          acc += reg->dh_seg[(size_t)p * r_a + r] *
                 (double)x[(size_t)row * in_dim + k];
        }
        const float val = (float)acc;
        if (!isfinite(val))
          return ET_MLORA_ERR_NONFINITE;
        reg->da_stage[s][(size_t)r * in_dim + k] = val;
      }
    }

    /* 2d. dX[row_p, k] = dx_acc[row_p, k] + sum_r dH_seg[p, r] * A_a[r, k]. */
    for (uint32_t p = 0u; p < seg_n; ++p) {
      const uint32_t row = seg_rows[start + p];
      const double *dh_row = &reg->dh_seg[(size_t)p * r_a];
      const double *dx_acc_row = &reg->dx_acc[(size_t)row * in_dim];
      float *dx_out_row = &reg->dx_stage[(size_t)row * in_dim];
      for (uint32_t k = 0u; k < in_dim; ++k) {
        double lora_g = 0.0;
        for (uint32_t r = 0u; r < r_a; ++r)
          lora_g += dh_row[r] * (double)slot->a[(size_t)r * in_dim + k];
        const float val = (float)(dx_acc_row[k] + lora_g);
        if (!isfinite(val))
          return ET_MLORA_ERR_NONFINITE;
        dx_out_row[k] = val;
      }
    }
  }

  /* 3. Atomic commit to caller dX and all active adapter gradient buffers. */
  memcpy(dx, reg->dx_stage, x_count * sizeof(float));
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    mlora_adapter_slot *slot = &reg->slots[s];
    if (slot->active == 0u)
      continue;
    const size_t a_count = (size_t)slot->rank * (size_t)in_dim;
    const size_t b_count = (size_t)out_dim * (size_t)slot->rank;
    memcpy(slot->da, reg->da_stage[s], a_count * sizeof(float));
    memcpy(slot->db, reg->db_stage[s], b_count * sizeof(float));
  }
  if (out_stats != NULL)
    *out_stats = stats;
  return ET_MLORA_OK;
}

/* ---- Public API: Per-Adapter AdamW (Zero-Alloc) ---- */

et_mlora_status et_mlora_validate_adamw_config_v1(
    const et_mlora_adamw_config *cfg) {
  if (cfg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (!isfinite(cfg->lr) || !isfinite(cfg->beta1) || !isfinite(cfg->beta2) ||
      !isfinite(cfg->eps) || !isfinite(cfg->weight_decay))
    return ET_MLORA_ERR_NONFINITE;
  if (!(cfg->lr > 0.0f) || !(cfg->eps > 0.0f) ||
      !(cfg->beta1 >= 0.0f && cfg->beta1 < 1.0f) ||
      !(cfg->beta2 >= 0.0f && cfg->beta2 < 1.0f) ||
      !(cfg->weight_decay >= 0.0f))
    return ET_MLORA_ERR_ARGUMENT;
  return ET_MLORA_OK;
}

static int adamw_update_array(size_t count, const float *param,
                              const float *grad, const float *m_in,
                              const float *v_in, double lr, double beta1,
                              double beta2, double eps, double wd, double bc1,
                              double bc2, float *param_out, float *m_out,
                              float *v_out) {
  for (size_t i = 0u; i < count; ++i) {
    const double p = (double)param[i];
    const double g = (double)grad[i];
    const double m = beta1 * (double)m_in[i] + (1.0 - beta1) * g;
    const double v = beta2 * (double)v_in[i] + (1.0 - beta2) * g * g;
    if (!isfinite(m) || !isfinite(v) || v < 0.0)
      return 0;
    const double m_hat = m / bc1;
    const double v_hat = v / bc2;
    const double denom = sqrt(v_hat) + eps;
    const double p_new = p - lr * wd * p - lr * (m_hat / denom);
    const float pf = (float)p_new;
    const float mf = (float)m;
    const float vf = (float)v;
    if (!isfinite(pf) || !isfinite(mf) || !isfinite(vf) || vf < 0.0f)
      return 0;
    param_out[i] = pf;
    m_out[i] = mf;
    v_out[i] = vf;
  }
  return 1;
}

et_mlora_status et_mlora_adamw_step_adapter_v1(
    et_mlora_registry *reg, int32_t adapter_id,
    const et_mlora_adamw_config *cfg) {
  if (reg == NULL || adapter_id < 0)
    return ET_MLORA_ERR_ARGUMENT;
  et_mlora_status status = et_mlora_validate_adamw_config_v1(cfg);
  if (status != ET_MLORA_OK)
    return status;

  const int32_t slot_idx = find_active_slot(reg, adapter_id);
  if (slot_idx < 0)
    return ET_MLORA_ERR_NOT_FOUND;

  mlora_adapter_slot *slot = &reg->slots[slot_idx];
  if (slot->adamw_step == UINT32_MAX)
    return ET_MLORA_ERR_OVERFLOW;

  const uint32_t next_step = slot->adamw_step + 1u;
  const double lr = (double)cfg->lr;
  const double beta1 = (double)cfg->beta1;
  const double beta2 = (double)cfg->beta2;
  const double eps = (double)cfg->eps;
  const double wd = (double)cfg->weight_decay;
  const double bc1 = 1.0 - pow_u32_f64(beta1, next_step);
  const double bc2 = 1.0 - pow_u32_f64(beta2, next_step);
  if (!isfinite(bc1) || !isfinite(bc2) || !(bc1 > 0.0) || !(bc2 > 0.0))
    return ET_MLORA_ERR_NONFINITE;

  const size_t a_count = (size_t)slot->rank * (size_t)reg->cfg.in_dim;
  const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)slot->rank;

  if (!adamw_update_array(a_count, slot->a, slot->da, slot->m_a, slot->v_a, lr,
                          beta1, beta2, eps, wd, bc1, bc2, reg->adamw_a_stage,
                          reg->adamw_ma_stage, reg->adamw_va_stage))
    return ET_MLORA_ERR_NONFINITE;

  if (!adamw_update_array(b_count, slot->b, slot->db, slot->m_b, slot->v_b, lr,
                          beta1, beta2, eps, wd, bc1, bc2, reg->adamw_b_stage,
                          reg->adamw_mb_stage, reg->adamw_vb_stage))
    return ET_MLORA_ERR_NONFINITE;

  /* Atomic commit. */
  memcpy(slot->a, reg->adamw_a_stage, a_count * sizeof(float));
  memcpy(slot->m_a, reg->adamw_ma_stage, a_count * sizeof(float));
  memcpy(slot->v_a, reg->adamw_va_stage, a_count * sizeof(float));
  memcpy(slot->b, reg->adamw_b_stage, b_count * sizeof(float));
  memcpy(slot->m_b, reg->adamw_mb_stage, b_count * sizeof(float));
  memcpy(slot->v_b, reg->adamw_vb_stage, b_count * sizeof(float));
  slot->adamw_step = next_step;
  return ET_MLORA_OK;
}

/* ---- Public API: Versioned SHA-256 Multi-Adapter Bundle (ETML1000 v1.0) ---- */

static size_t compute_bundle_size_internal(const et_mlora_registry *reg,
                                           uint32_t include_adamw) {
  const size_t mult = include_adamw != 0u ? 3u : 1u;
  size_t total = ET_MLORA_BUNDLE_HEADER_BYTES + ET_MLORA_BUNDLE_CHECKSUM_BYTES;
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    const mlora_adapter_slot *slot = &reg->slots[s];
    if (slot->active == 0u)
      continue;
    const size_t floats =
        ((size_t)slot->rank * (size_t)reg->cfg.in_dim +
         (size_t)reg->cfg.out_dim * (size_t)slot->rank) *
        mult;
    total += ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES + floats * 4u;
  }
  return total;
}

et_mlora_status et_mlora_bundle_size_v1(const et_mlora_registry *reg,
                                        uint32_t include_adamw,
                                        size_t *out_bytes) {
  if (reg == NULL || out_bytes == NULL || include_adamw > 1u)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  *out_bytes = compute_bundle_size_internal(reg, include_adamw);
  return ET_MLORA_OK;
}

static void encode_f32_array(uint8_t *dst, const float *src, size_t count) {
  for (size_t i = 0u; i < count; ++i)
    encode_u32_le(dst + i * 4u, f32_to_bits(src[i]));
}

et_mlora_status et_mlora_bundle_serialize_v1(const et_mlora_registry *reg,
                                             uint32_t include_adamw,
                                             uint8_t *out_bytes,
                                             size_t capacity,
                                             size_t *out_size) {
  if (reg == NULL || out_bytes == NULL || out_size == NULL ||
      include_adamw > 1u)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;

  const size_t total = compute_bundle_size_internal(reg, include_adamw);
  if (total > ET_MLORA_BUNDLE_MAX_BYTES || total > UINT32_MAX)
    return ET_MLORA_ERR_OVERFLOW;
  if (capacity < total)
    return ET_MLORA_ERR_ARGUMENT;
  if (!aligned_span(out_bytes, 1u, total))
    return ET_MLORA_ERR_ALIGNMENT;
  if (spans_overlap(out_bytes, total, reg, sizeof(*reg)) ||
      spans_overlap(out_bytes, total, out_size, sizeof(*out_size)))
    return ET_MLORA_ERR_ALIAS;

  uint8_t *stage = (uint8_t *)malloc(total);
  if (stage == NULL)
    return ET_MLORA_ERR_NOMEM;

  uint32_t sorted[ET_MLORA_MAX_ADAPTERS];
  const uint32_t adapter_count = collect_sorted_active_slots(reg, sorted);

  uint8_t base_digest[ET_MLORA_SHA256_BYTES];
  compute_base_digest_internal(&reg->cfg, reg->w_base, reg->bias, base_digest);

  const uint32_t flags =
      (reg->cfg.has_bias != 0u ? ET_MLORA_BUNDLE_FLAG_HAS_BIAS : 0u) |
      (include_adamw != 0u ? ET_MLORA_BUNDLE_FLAG_HAS_ADAMW : 0u);

  memcpy(stage, ET_MLORA_BUNDLE_MAGIC, ET_MLORA_BUNDLE_MAGIC_BYTES);
  encode_u32_le(stage + 8u, ET_MLORA_BUNDLE_VERSION_MAJOR);
  encode_u32_le(stage + 12u, ET_MLORA_BUNDLE_VERSION_MINOR);
  encode_u32_le(stage + 16u, reg->cfg.in_dim);
  encode_u32_le(stage + 20u, reg->cfg.out_dim);
  encode_u32_le(stage + 24u, reg->cfg.max_rank);
  encode_u32_le(stage + 28u, adapter_count);
  encode_u32_le(stage + 32u, flags);
  encode_u32_le(stage + 36u, (uint32_t)total);
  memcpy(stage + 40u, base_digest, sizeof(base_digest));

  size_t offset = ET_MLORA_BUNDLE_HEADER_BYTES;
  for (uint32_t idx = 0u; idx < adapter_count; ++idx) {
    const mlora_adapter_slot *slot = &reg->slots[sorted[idx]];
    const size_t a_count = (size_t)slot->rank * (size_t)reg->cfg.in_dim;
    const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)slot->rank;

    encode_u32_le(stage + offset + 0u, (uint32_t)slot->adapter_id);
    encode_u32_le(stage + offset + 4u, slot->rank);
    encode_u32_le(stage + offset + 8u, f32_to_bits(slot->alpha));
    encode_u32_le(stage + offset + 12u,
                  include_adamw != 0u ? slot->adamw_step : 0u);
    encode_u32_le(stage + offset + 16u, slot->generation);
    encode_u32_le(stage + offset + 20u, 0u);
    offset += ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES;

    encode_f32_array(stage + offset, slot->a, a_count);
    offset += a_count * 4u;
    encode_f32_array(stage + offset, slot->b, b_count);
    offset += b_count * 4u;

    if (include_adamw != 0u) {
      encode_f32_array(stage + offset, slot->m_a, a_count);
      offset += a_count * 4u;
      encode_f32_array(stage + offset, slot->v_a, a_count);
      offset += a_count * 4u;
      encode_f32_array(stage + offset, slot->m_b, b_count);
      offset += b_count * 4u;
      encode_f32_array(stage + offset, slot->v_b, b_count);
      offset += b_count * 4u;
    }
  }

  uint8_t bundle_digest[ET_MLORA_SHA256_BYTES];
  sha256_bytes(stage, offset, bundle_digest);
  memcpy(stage + offset, bundle_digest, sizeof(bundle_digest));
  offset += sizeof(bundle_digest);

  memcpy(out_bytes, stage, offset);
  *out_size = offset;
  free(stage);
  return ET_MLORA_OK;
}

static int decode_f32_array(const uint8_t *src, size_t count, int require_nonneg,
                            float *dst) {
  for (size_t i = 0u; i < count; ++i) {
    float val = 0.0f;
    if (!bits_to_finite_f32(decode_u32_le(src + i * 4u), &val))
      return 0;
    if (require_nonneg && val < 0.0f)
      return 0;
    dst[i] = val;
  }
  return 1;
}

et_mlora_status et_mlora_bundle_parse_v1(const uint8_t *bytes, size_t size,
                                         et_mlora_registry *reg,
                                         et_mlora_bundle_meta *out_meta) {
  if (bytes == NULL || reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (spans_overlap(bytes, size, reg, sizeof(*reg)))
    return ET_MLORA_ERR_ALIAS;
  if (out_meta != NULL) {
    if (!aligned_span(out_meta, _Alignof(et_mlora_bundle_meta),
                      sizeof(*out_meta)))
      return ET_MLORA_ERR_ALIGNMENT;
    if (spans_overlap(out_meta, sizeof(*out_meta), bytes, size) ||
        spans_overlap(out_meta, sizeof(*out_meta), reg, sizeof(*reg)))
      return ET_MLORA_ERR_ALIAS;
  }

  /* Reject loading over any currently pinned adapter. */
  for (uint32_t s = 0u; s < reg->cfg.max_adapters; ++s) {
    if (reg->slots[s].active != 0u && reg->slots[s].pin_count > 0u)
      return ET_MLORA_ERR_BUSY;
  }

  /* 1. Magic and minimum/maximum bounds. */
  if (size >= ET_MLORA_BUNDLE_MAGIC_BYTES &&
      memcmp(bytes, ET_MLORA_BUNDLE_MAGIC, ET_MLORA_BUNDLE_MAGIC_BYTES) != 0)
    return ET_MLORA_ERR_FORMAT;
  if (size < ET_MLORA_BUNDLE_MIN_BYTES)
    return ET_MLORA_ERR_TRUNCATED;
  if (size > ET_MLORA_BUNDLE_MAX_BYTES)
    return ET_MLORA_ERR_FORMAT;

  /* 2. Version check. */
  const uint32_t ver_major = decode_u32_le(bytes + 8u);
  const uint32_t ver_minor = decode_u32_le(bytes + 12u);
  if (ver_major != ET_MLORA_BUNDLE_VERSION_MAJOR ||
      ver_minor != ET_MLORA_BUNDLE_VERSION_MINOR)
    return ET_MLORA_ERR_VERSION;

  /* 3. Truncation / trailing-bytes check against header total_bytes. */
  const uint32_t file_total_bytes = decode_u32_le(bytes + 36u);
  if (file_total_bytes >= ET_MLORA_BUNDLE_MIN_BYTES &&
      file_total_bytes <= ET_MLORA_BUNDLE_MAX_BYTES) {
    if (size < (size_t)file_total_bytes)
      return ET_MLORA_ERR_TRUNCATED;
    if (size > (size_t)file_total_bytes)
      return ET_MLORA_ERR_FORMAT;
  }

  /* 4. SHA-256 integrity verification over [0 .. size - 32). */
  const size_t signed_bytes = size - ET_MLORA_BUNDLE_CHECKSUM_BYTES;
  uint8_t computed_digest[ET_MLORA_SHA256_BYTES];
  sha256_bytes(bytes, signed_bytes, computed_digest);
  if (memcmp(computed_digest, bytes + signed_bytes, ET_MLORA_SHA256_BYTES) != 0)
    return ET_MLORA_ERR_CHECKSUM;

  if ((size_t)file_total_bytes != size)
    return ET_MLORA_ERR_FORMAT;

  /* 5. Header fields & shape/flag binding. */
  const uint32_t file_in_dim = decode_u32_le(bytes + 16u);
  const uint32_t file_out_dim = decode_u32_le(bytes + 20u);
  const uint32_t file_max_rank = decode_u32_le(bytes + 24u);
  const uint32_t adapter_count = decode_u32_le(bytes + 28u);
  const uint32_t flags = decode_u32_le(bytes + 32u);

  if ((flags & ~(ET_MLORA_BUNDLE_FLAG_HAS_BIAS |
                 ET_MLORA_BUNDLE_FLAG_HAS_ADAMW)) != 0u)
    return ET_MLORA_ERR_FORMAT;
  const uint32_t file_has_bias =
      (flags & ET_MLORA_BUNDLE_FLAG_HAS_BIAS) != 0u ? 1u : 0u;
  const uint32_t file_has_adamw =
      (flags & ET_MLORA_BUNDLE_FLAG_HAS_ADAMW) != 0u ? 1u : 0u;

  if (file_in_dim != reg->cfg.in_dim || file_out_dim != reg->cfg.out_dim ||
      file_has_bias != reg->cfg.has_bias)
    return ET_MLORA_ERR_SHAPE;
  if (adapter_count > reg->cfg.max_adapters)
    return ET_MLORA_ERR_FULL;

  /* 6. Base SHA-256 binding check. */
  uint8_t expected_base[ET_MLORA_SHA256_BYTES];
  compute_base_digest_internal(&reg->cfg, reg->w_base, reg->bias,
                               expected_base);
  if (memcmp(bytes + 40u, expected_base, ET_MLORA_SHA256_BYTES) != 0)
    return ET_MLORA_ERR_BASE_HASH;

  /* 7. Decode all heterogeneous adapter records into staged slot pool. */
  mlora_adapter_slot *staged_slots =
      (mlora_adapter_slot *)calloc(ET_MLORA_MAX_ADAPTERS,
                                   sizeof(mlora_adapter_slot));
  if (staged_slots == NULL)
    return ET_MLORA_ERR_NOMEM;

  size_t offset = ET_MLORA_BUNDLE_HEADER_BYTES;
  int32_t prev_id = -1;
  et_mlora_status status = ET_MLORA_OK;

  for (uint32_t i = 0u; i < adapter_count; ++i) {
    if (signed_bytes - offset < ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES) {
      status = ET_MLORA_ERR_TRUNCATED;
      break;
    }
    const uint32_t raw_id = decode_u32_le(bytes + offset + 0u);
    const uint32_t rank = decode_u32_le(bytes + offset + 4u);
    const uint32_t alpha_bits = decode_u32_le(bytes + offset + 8u);
    const uint32_t adamw_step = decode_u32_le(bytes + offset + 12u);
    const uint32_t generation = decode_u32_le(bytes + offset + 16u);
    const uint32_t reserved = decode_u32_le(bytes + offset + 20u);
    offset += ET_MLORA_BUNDLE_ADAPTER_HEADER_BYTES;

    if (raw_id > (uint32_t)INT32_MAX || reserved != 0u || generation == 0u) {
      status = ET_MLORA_ERR_FORMAT;
      break;
    }
    const int32_t aid = (int32_t)raw_id;
    if (aid <= prev_id) {
      status = ET_MLORA_ERR_FORMAT;
      break;
    }
    prev_id = aid;

    if (file_has_adamw == 0u && adamw_step != 0u) {
      status = ET_MLORA_ERR_FORMAT;
      break;
    }

    float alpha = 0.0f;
    if (!bits_to_finite_f32(alpha_bits, &alpha)) {
      status = ET_MLORA_ERR_NONFINITE;
      break;
    }
    status = validate_rank_alpha(reg, rank, alpha);
    if (status != ET_MLORA_OK)
      break;

    const size_t a_count = (size_t)rank * (size_t)reg->cfg.in_dim;
    const size_t b_count = (size_t)reg->cfg.out_dim * (size_t)rank;
    const size_t mult = file_has_adamw != 0u ? 3u : 1u;
    const size_t payload_bytes = (a_count + b_count) * mult * 4u;
    if (signed_bytes - offset < payload_bytes) {
      status = ET_MLORA_ERR_TRUNCATED;
      break;
    }

    mlora_adapter_slot *dst = &staged_slots[i];
    dst->active = 1u;
    dst->adapter_id = aid;
    dst->generation = generation;
    dst->rank = rank;
    dst->alpha = alpha;
    dst->pin_count = 0u;
    dst->adamw_step = adamw_step;

    if (!decode_f32_array(bytes + offset, a_count, 0, dst->a)) {
      status = ET_MLORA_ERR_NONFINITE;
      break;
    }
    offset += a_count * 4u;
    if (!decode_f32_array(bytes + offset, b_count, 0, dst->b)) {
      status = ET_MLORA_ERR_NONFINITE;
      break;
    }
    offset += b_count * 4u;

    if (file_has_adamw != 0u) {
      if (!decode_f32_array(bytes + offset, a_count, 0, dst->m_a)) {
        status = ET_MLORA_ERR_NONFINITE;
        break;
      }
      offset += a_count * 4u;
      if (!decode_f32_array(bytes + offset, a_count, 1, dst->v_a)) {
        status = ET_MLORA_ERR_NONFINITE;
        break;
      }
      offset += a_count * 4u;
      if (!decode_f32_array(bytes + offset, b_count, 0, dst->m_b)) {
        status = ET_MLORA_ERR_NONFINITE;
        break;
      }
      offset += b_count * 4u;
      if (!decode_f32_array(bytes + offset, b_count, 1, dst->v_b)) {
        status = ET_MLORA_ERR_NONFINITE;
        break;
      }
      offset += b_count * 4u;
    }
  }

  if (status == ET_MLORA_OK && offset != signed_bytes)
    status = ET_MLORA_ERR_FORMAT;

  if (status != ET_MLORA_OK) {
    free(staged_slots);
    return status;
  }

  /* 8. Atomic commit to registry. */
  memcpy(reg->slots, staged_slots,
         ET_MLORA_MAX_ADAPTERS * sizeof(mlora_adapter_slot));
  free(staged_slots);

  if (out_meta != NULL) {
    out_meta->version_major = ver_major;
    out_meta->version_minor = ver_minor;
    out_meta->in_dim = file_in_dim;
    out_meta->out_dim = file_out_dim;
    out_meta->max_rank = file_max_rank;
    out_meta->adapter_count = adapter_count;
    out_meta->flags = flags;
    out_meta->total_bytes = file_total_bytes;
    memcpy(out_meta->base_sha256, bytes + 40u, ET_MLORA_SHA256_BYTES);
    memcpy(out_meta->bundle_sha256, computed_digest, ET_MLORA_SHA256_BYTES);
  }
  return ET_MLORA_OK;
}

/* ---- Atomic File Save (0600 temp + fsync + rename) & O_NOFOLLOW Load ---- */

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

et_mlora_status et_mlora_bundle_save_v1(const char *path,
                                        const et_mlora_registry *reg,
                                        uint32_t include_adamw) {
  if (path == NULL || path[0] == '\0' || reg == NULL || include_adamw > 1u)
    return ET_MLORA_ERR_ARGUMENT;
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_MLORA_ERR_ARGUMENT;

  size_t needed = 0u;
  et_mlora_status status = et_mlora_bundle_size_v1(reg, include_adamw, &needed);
  if (status != ET_MLORA_OK)
    return status;

  uint8_t *buffer = (uint8_t *)malloc(needed);
  if (buffer == NULL)
    return ET_MLORA_ERR_NOMEM;

  size_t written_size = 0u;
  status = et_mlora_bundle_serialize_v1(reg, include_adamw, buffer, needed,
                                        &written_size);
  if (status != ET_MLORA_OK) {
    free(buffer);
    return status;
  }

  static const char suffix[] = ".mlora-bundle.XXXXXX";
  const size_t prefix = slash == NULL ? 0u : (size_t)(slash - path) + 1u;
  if (prefix > (size_t)PTRDIFF_MAX - sizeof(suffix)) {
    free(buffer);
    return ET_MLORA_ERR_OVERFLOW;
  }

  char *temp_path = (char *)malloc(prefix + sizeof(suffix));
  if (temp_path == NULL) {
    free(buffer);
    return ET_MLORA_ERR_NOMEM;
  }
  memcpy(temp_path, path, prefix);
  memcpy(temp_path + prefix, suffix, sizeof(suffix));

  const int fd = mkstemp(temp_path);
  if (fd < 0) {
    free(temp_path);
    free(buffer);
    return ET_MLORA_ERR_IO;
  }

  int ok = (fchmod(fd, S_IRUSR | S_IWUSR) == 0) &&
           write_all(fd, buffer, written_size) && sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    status = ET_MLORA_ERR_IO;
  }

  free(temp_path);
  free(buffer);
  return status;
}

et_mlora_status et_mlora_bundle_load_v1(const char *path,
                                        et_mlora_registry *reg,
                                        et_mlora_bundle_meta *out_meta) {
  if (!fp_environment_supported())
    return ET_MLORA_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0' || reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_MLORA_ERR_SYMLINK : ET_MLORA_ERR_IO;

  struct stat meta;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    (void)close(fd);
    return ET_MLORA_ERR_IO;
  }
  if (meta.st_size < 0 ||
      (uintmax_t)meta.st_size > ET_MLORA_BUNDLE_MAX_BYTES) {
    (void)close(fd);
    return ET_MLORA_ERR_FORMAT;
  }

  const size_t file_size = (size_t)meta.st_size;
  const size_t alloc_size = file_size + 1u;
  uint8_t *buffer = (uint8_t *)malloc(alloc_size);
  if (buffer == NULL) {
    (void)close(fd);
    return ET_MLORA_ERR_NOMEM;
  }

  size_t used = 0u;
  et_mlora_status status = ET_MLORA_OK;
  while (used < alloc_size) {
    const ssize_t count = read(fd, buffer + used, alloc_size - used);
    if (count == 0)
      break;
    if (count < 0) {
      if (errno == EINTR)
        continue;
      status = ET_MLORA_ERR_IO;
      break;
    }
    used += (size_t)count;
  }
  if (close(fd) != 0 && status == ET_MLORA_OK)
    status = ET_MLORA_ERR_IO;

  if (status == ET_MLORA_OK) {
    if (used > ET_MLORA_BUNDLE_MAX_BYTES)
      status = ET_MLORA_ERR_FORMAT;
    else if (used != file_size)
      status = ET_MLORA_ERR_TRUNCATED;
    else
      status = et_mlora_bundle_parse_v1(buffer, used, reg, out_meta);
  }

  free(buffer);
  return status;
}
