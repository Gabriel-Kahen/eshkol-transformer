#define _GNU_SOURCE
#include "eshkol_transformer/mtp_head_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(__x86_64__)
#error "DeepSeek-V3 MTP engine v1 requires the reviewed x86-64 FP lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MTP engine requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "MTP engine requires standard IEEE-754 binary32/binary64 eval");

struct et_mtp_model {
  et_mtp_config_v1 cfg;
  et_mtp_layout_v1 layout;
  uint64_t step;
  float *params;
  float *m;
  float *v;
};

/* --- Low-level FP / Bit / Endian / Alignment Helpers --- */

static int fp_environment_supported(void) {
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round-to-nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u) {
    return 0;
  }
  const uintptr_t la = (uintptr_t)a;
  const uintptr_t lb = (uintptr_t)b;
  if (la > UINTPTR_MAX - a_bytes || lb > UINTPTR_MAX - b_bytes) {
    return 1;
  }
  return la < lb + b_bytes && lb < la + a_bytes;
}

static uint32_t f32_to_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static float bits_to_f32(uint32_t b) {
  float v = 0.0f;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static uint64_t f64_to_bits(double v) {
  uint64_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static double bits_to_f64(uint64_t b) {
  double v = 0.0;
  memcpy(&v, &b, sizeof(v));
  return v;
}

static float canonical_f32(float v) { return v == 0.0f ? 0.0f : v; }

static int finite_f64_to_f32(double value, float *out) {
  if (!isfinite(value)) {
    return 0;
  }
  const float f = (float)value;
  if (!isfinite(f)) {
    return 0;
  }
  *out = canonical_f32(f);
  return 1;
}

static int all_f32_finite(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i])) {
      return 0;
    }
  }
  return 1;
}

static void put_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void put_u64_le(uint8_t *dst, uint64_t v) {
  for (size_t i = 0u; i < 8u; ++i) {
    dst[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
  }
}

static uint16_t get_u16_le(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static uint32_t get_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint64_t get_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i) {
    v |= ((uint64_t)src[i] << (8u * i));
  }
  return v;
}

static double ipow_f64(double base, uint64_t exp) {
  double result = 1.0;
  double factor = base;
  while (exp > 0u) {
    if ((exp & UINT64_C(1)) != 0u) {
      result *= factor;
    }
    exp >>= 1u;
    if (exp > 0u) {
      factor *= factor;
    }
  }
  return result;
}

/* --- Standalone FIPS 180-4 SHA-256 --- */

typedef struct mtp_sha256_ctx {
  uint32_t state[8];
  uint64_t bitlen;
  uint8_t buf[64];
  size_t buflen;
} mtp_sha256_ctx;

static uint32_t rotr32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_transform(mtp_sha256_ctx *ctx, const uint8_t block[64]) {
  static const uint32_t K[64] = {
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
  for (size_t i = 0u; i < 16u; ++i) {
    w[i] = ((uint32_t)block[i * 4u] << 24u) |
           ((uint32_t)block[i * 4u + 1u] << 16u) |
           ((uint32_t)block[i * 4u + 2u] << 8u) |
           ((uint32_t)block[i * 4u + 3u]);
  }
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }

  uint32_t a = ctx->state[0];
  uint32_t b = ctx->state[1];
  uint32_t c = ctx->state[2];
  uint32_t d = ctx->state[3];
  uint32_t e = ctx->state[4];
  uint32_t f = ctx->state[5];
  uint32_t g = ctx->state[6];
  uint32_t h = ctx->state[7];

  for (size_t i = 0u; i < 64u; ++i) {
    const uint32_t S1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 = h + S1 + ch + K[i] + w[i];
    const uint32_t S0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = S0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  ctx->state[0] += a;
  ctx->state[1] += b;
  ctx->state[2] += c;
  ctx->state[3] += d;
  ctx->state[4] += e;
  ctx->state[5] += f;
  ctx->state[6] += g;
  ctx->state[7] += h;
}

static void sha256_init(mtp_sha256_ctx *ctx) {
  ctx->state[0] = 0x6a09e667u;
  ctx->state[1] = 0xbb67ae85u;
  ctx->state[2] = 0x3c6ef372u;
  ctx->state[3] = 0xa54ff53au;
  ctx->state[4] = 0x510e527fu;
  ctx->state[5] = 0x9b05688cu;
  ctx->state[6] = 0x1f83d9abu;
  ctx->state[7] = 0x5be0cd19u;
  ctx->bitlen = 0u;
  ctx->buflen = 0u;
}

static void sha256_update(mtp_sha256_ctx *ctx, const uint8_t *data,
                          size_t len) {
  for (size_t i = 0u; i < len; ++i) {
    ctx->buf[ctx->buflen++] = data[i];
    if (ctx->buflen == 64u) {
      sha256_transform(ctx, ctx->buf);
      ctx->bitlen += 512u;
      ctx->buflen = 0u;
    }
  }
}

static void sha256_final(mtp_sha256_ctx *ctx, uint8_t out[32]) {
  const uint64_t total_bits = ctx->bitlen + ((uint64_t)ctx->buflen * 8u);
  ctx->buf[ctx->buflen++] = 0x80u;
  if (ctx->buflen > 56u) {
    while (ctx->buflen < 64u) {
      ctx->buf[ctx->buflen++] = 0u;
    }
    sha256_transform(ctx, ctx->buf);
    ctx->buflen = 0u;
  }
  while (ctx->buflen < 56u) {
    ctx->buf[ctx->buflen++] = 0u;
  }
  for (size_t i = 0u; i < 8u; ++i) {
    ctx->buf[56u + i] = (uint8_t)((total_bits >> (56u - 8u * i)) & 0xffu);
  }
  sha256_transform(ctx, ctx->buf);
  for (size_t i = 0u; i < 8u; ++i) {
    out[i * 4u] = (uint8_t)((ctx->state[i] >> 24u) & 0xffu);
    out[i * 4u + 1u] = (uint8_t)((ctx->state[i] >> 16u) & 0xffu);
    out[i * 4u + 2u] = (uint8_t)((ctx->state[i] >> 8u) & 0xffu);
    out[i * 4u + 3u] = (uint8_t)(ctx->state[i] & 0xffu);
  }
}

et_mtp_status_t et_mtp_sha256_bytes_v1(const void *data, size_t len,
                                       uint8_t out_digest[32]) {
  if (out_digest == NULL || (data == NULL && len > 0u)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  mtp_sha256_ctx ctx;
  sha256_init(&ctx);
  if (len > 0u) {
    sha256_update(&ctx, (const uint8_t *)data, len);
  }
  sha256_final(&ctx, out_digest);
  return ET_MTP_OK;
}

/* --- SplitMix64 Deterministic Counter RNG --- */

static uint64_t splitmix64_mix(uint64_t x) {
  x = (x ^ (x >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31u);
}

static double rng_draw_uniform(et_mtp_rng_state_v1 *rng, uint64_t *draws) {
  const uint64_t z =
      splitmix64_mix(rng->seed + rng->counter * UINT64_C(0x9e3779b97f4a7c15));
  rng->counter += 1u;
  if (draws != NULL) {
    *draws += 1u;
  }
  /* Exact 53-bit uniform variate in [0.0, 1.0). */
  return (double)(z >> 11u) * (1.0 / 9007199254740992.0);
}

static uint64_t init_splitmix_next(uint64_t *state) {
  *state += UINT64_C(0x9e3779b97f4a7c15);
  return splitmix64_mix(*state);
}

/* --- Configuration & Layout Validation --- */

et_mtp_status_t et_mtp_validate_config_v1(const et_mtp_config_v1 *cfg) {
  if (cfg == NULL ||
      !aligned_span(cfg, _Alignof(et_mtp_config_v1), sizeof(*cfg))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (cfg->struct_size != sizeof(et_mtp_config_v1) || cfg->reserved != 0u) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (cfg->mtp_depth < ET_MTP_MIN_DEPTH || cfg->mtp_depth > ET_MTP_MAX_DEPTH ||
      cfg->vocab_size < ET_MTP_MIN_VOCAB ||
      cfg->vocab_size > ET_MTP_MAX_VOCAB ||
      cfg->hidden_dim < ET_MTP_MIN_HIDDEN ||
      cfg->hidden_dim > ET_MTP_MAX_HIDDEN || cfg->ffn_dim < ET_MTP_MIN_FFN ||
      cfg->ffn_dim > ET_MTP_MAX_FFN || cfg->tie_weights > 1u) {
    return ET_MTP_ERR_UNSUPPORTED;
  }
  if (!isfinite(cfg->rms_eps) || cfg->rms_eps <= 0.0 || cfg->rms_eps > 1.0 ||
      !isfinite(cfg->lambda_mtp) || cfg->lambda_mtp < 0.0 ||
      signbit(cfg->lambda_mtp) || !isfinite(cfg->lr) || cfg->lr < 0.0 ||
      signbit(cfg->lr) || !isfinite(cfg->beta1) || cfg->beta1 < 0.0 ||
      cfg->beta1 >= 1.0 || !isfinite(cfg->beta2) || cfg->beta2 < 0.0 ||
      cfg->beta2 >= 1.0 || !isfinite(cfg->eps) || cfg->eps <= 0.0 ||
      !isfinite(cfg->weight_decay) || cfg->weight_decay < 0.0 ||
      signbit(cfg->weight_decay) || !isfinite(cfg->max_grad_norm) ||
      cfg->max_grad_norm < 0.0 || signbit(cfg->max_grad_norm)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_compute_layout_v1(const et_mtp_config_v1 *cfg,
                                         et_mtp_layout_v1 *out_layout) {
  const et_mtp_status_t vst = et_mtp_validate_config_v1(cfg);
  if (vst != ET_MTP_OK) {
    return vst;
  }
  if (out_layout == NULL || !aligned_span(out_layout, _Alignof(et_mtp_layout_v1),
                                          sizeof(*out_layout))) {
    return ET_MTP_ERR_INVALID_ARG;
  }

  const size_t V = (size_t)cfg->vocab_size;
  const size_t H = (size_t)cfg->hidden_dim;
  const size_t F = (size_t)cfg->ffn_dim;
  const size_t D = (size_t)cfg->mtp_depth;

  et_mtp_layout_v1 lay;
  memset(&lay, 0, sizeof(lay));
  lay.struct_size = sizeof(et_mtp_layout_v1);

  size_t off = 0u;
  lay.embed_offset = off;
  lay.embed_len = V * H;
  off += lay.embed_len;

  if (cfg->tie_weights == 1u) {
    lay.lm_head_offset = lay.embed_offset;
    lay.lm_head_len = 0u;
  } else {
    lay.lm_head_offset = off;
    lay.lm_head_len = V * H;
    off += lay.lm_head_len;
  }

  lay.gamma_main_offset = off;
  lay.gamma_main_len = H;
  off += lay.gamma_main_len;

  size_t h_rel = 0u;
  lay.rel_gamma_h = h_rel;
  h_rel += H;
  lay.rel_gamma_e = h_rel;
  h_rel += H;
  lay.rel_w_proj = h_rel;
  h_rel += H * (2u * H);
  lay.rel_gamma_f = h_rel;
  h_rel += H;
  lay.rel_w_gate = h_rel;
  h_rel += F * H;
  lay.rel_w_up = h_rel;
  h_rel += F * H;
  lay.rel_w_down = h_rel;
  h_rel += H * F;
  lay.rel_gamma_o = h_rel;
  h_rel += H;
  lay.head_stride = h_rel;

  for (size_t k = 0u; k < D; ++k) {
    lay.head_offset[k] = off;
    off += lay.head_stride;
  }
  lay.total_params = off;

  *out_layout = lay;
  return ET_MTP_OK;
}

/* --- Model Lifecycle & Accessors --- */

et_mtp_status_t et_mtp_model_create_v1(const et_mtp_config_v1 *cfg,
                                       et_mtp_model_t **out_model) {
  if (out_model == NULL ||
      !aligned_span(out_model, _Alignof(et_mtp_model_t *),
                    sizeof(*out_model))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  et_mtp_layout_v1 lay;
  const et_mtp_status_t lst = et_mtp_compute_layout_v1(cfg, &lay);
  if (lst != ET_MTP_OK) {
    return lst;
  }

  et_mtp_model_t *model = (et_mtp_model_t *)calloc(1u, sizeof(*model));
  if (model == NULL) {
    return ET_MTP_ERR_ALLOC;
  }
  model->params = (float *)calloc(lay.total_params, sizeof(float));
  model->m = (float *)calloc(lay.total_params, sizeof(float));
  model->v = (float *)calloc(lay.total_params, sizeof(float));
  if (model->params == NULL || model->m == NULL || model->v == NULL) {
    free(model->params);
    free(model->m);
    free(model->v);
    free(model);
    return ET_MTP_ERR_ALLOC;
  }
  model->cfg = *cfg;
  model->layout = lay;
  model->step = 0u;

  /* Initialize RMSNorm gammas to 1.0f so freshly created model is valid. */
  const size_t H = (size_t)cfg->hidden_dim;
  for (size_t j = 0u; j < H; ++j) {
    model->params[lay.gamma_main_offset + j] = 1.0f;
  }
  for (size_t k = 0u; k < (size_t)cfg->mtp_depth; ++k) {
    const size_t base = lay.head_offset[k];
    for (size_t j = 0u; j < H; ++j) {
      model->params[base + lay.rel_gamma_h + j] = 1.0f;
      model->params[base + lay.rel_gamma_e + j] = 1.0f;
      model->params[base + lay.rel_gamma_f + j] = 1.0f;
      model->params[base + lay.rel_gamma_o + j] = 1.0f;
    }
  }

  *out_model = model;
  return ET_MTP_OK;
}

void et_mtp_model_destroy_v1(et_mtp_model_t *model) {
  if (model == NULL) {
    return;
  }
  free(model->params);
  free(model->m);
  free(model->v);
  free(model);
}

et_mtp_status_t et_mtp_model_init_weights_v1(et_mtp_model_t *model,
                                             uint64_t seed) {
  if (model == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    return ET_MTP_ERR_FP_ENV;
  }

  const et_mtp_layout_v1 *lay = &model->layout;
  const size_t H = (size_t)model->cfg.hidden_dim;
  uint64_t state = seed ^ UINT64_C(0x517cc1b727220a95);

  /* Uniform [-0.25, +0.25] for all weights, then 1.0f for RMSNorm gammas. */
  for (size_t i = 0u; i < lay->total_params; ++i) {
    const uint64_t z = init_splitmix_next(&state);
    const double u = (double)(z >> 11u) * (1.0 / 9007199254740992.0);
    const double val = (u * 2.0 - 1.0) * 0.25;
    model->params[i] = canonical_f32((float)val);
  }
  for (size_t j = 0u; j < H; ++j) {
    model->params[lay->gamma_main_offset + j] = 1.0f;
  }
  for (size_t k = 0u; k < (size_t)model->cfg.mtp_depth; ++k) {
    const size_t base = lay->head_offset[k];
    for (size_t j = 0u; j < H; ++j) {
      model->params[base + lay->rel_gamma_h + j] = 1.0f;
      model->params[base + lay->rel_gamma_e + j] = 1.0f;
      model->params[base + lay->rel_gamma_f + j] = 1.0f;
      model->params[base + lay->rel_gamma_o + j] = 1.0f;
    }
  }
  memset(model->m, 0, lay->total_params * sizeof(float));
  memset(model->v, 0, lay->total_params * sizeof(float));
  model->step = 0u;

  (void)fesetenv(&saved_env);
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_get_config_v1(const et_mtp_model_t *model,
                                           et_mtp_config_v1 *out_cfg) {
  if (model == NULL || out_cfg == NULL ||
      !aligned_span(out_cfg, _Alignof(et_mtp_config_v1), sizeof(*out_cfg))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  *out_cfg = model->cfg;
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_get_layout_v1(const et_mtp_model_t *model,
                                           et_mtp_layout_v1 *out_layout) {
  if (model == NULL || out_layout == NULL ||
      !aligned_span(out_layout, _Alignof(et_mtp_layout_v1),
                    sizeof(*out_layout))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  *out_layout = model->layout;
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_get_step_v1(const et_mtp_model_t *model,
                                         uint64_t *out_step) {
  if (model == NULL || out_step == NULL ||
      !aligned_span(out_step, _Alignof(uint64_t), sizeof(*out_step))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  *out_step = model->step;
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_get_params_v1(const et_mtp_model_t *model,
                                           float *out_params, size_t count) {
  if (model == NULL || out_params == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (count != model->layout.total_params) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(out_params, _Alignof(float), bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  memcpy(out_params, model->params, bytes);
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_set_params_v1(et_mtp_model_t *model,
                                           const float *params, size_t count) {
  if (model == NULL || params == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (count != model->layout.total_params) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(params, _Alignof(float), bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!all_f32_finite(params, count)) {
    return ET_MTP_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < count; ++i) {
    model->params[i] = canonical_f32(params[i]);
  }
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_get_moments_v1(const et_mtp_model_t *model,
                                            float *out_m, float *out_v,
                                            size_t count) {
  if (model == NULL || out_m == NULL || out_v == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (count != model->layout.total_params) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(out_m, _Alignof(float), bytes) ||
      !aligned_span(out_v, _Alignof(float), bytes) ||
      spans_overlap(out_m, bytes, out_v, bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  memcpy(out_m, model->m, bytes);
  memcpy(out_v, model->v, bytes);
  return ET_MTP_OK;
}

et_mtp_status_t et_mtp_model_state_sha256_v1(const et_mtp_model_t *model,
                                             uint8_t out_digest[32]) {
  if (model == NULL || out_digest == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  mtp_sha256_ctx ctx;
  sha256_init(&ctx);

  uint8_t u64_buf[8];
  put_u64_le(u64_buf, model->step);
  sha256_update(&ctx, u64_buf, sizeof(u64_buf));

  const size_t n = model->layout.total_params;
  uint8_t u32_buf[4];
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(u32_buf, f32_to_bits(canonical_f32(model->params[i])));
    sha256_update(&ctx, u32_buf, sizeof(u32_buf));
  }
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(u32_buf, f32_to_bits(canonical_f32(model->m[i])));
    sha256_update(&ctx, u32_buf, sizeof(u32_buf));
  }
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(u32_buf, f32_to_bits(canonical_f32(model->v[i])));
    sha256_update(&ctx, u32_buf, sizeof(u32_buf));
  }
  sha256_final(&ctx, out_digest);
  return ET_MTP_OK;
}

/* --- Core Binary64 Math Primitives (RMSNorm, SwiGLU, Cross-Entropy) --- */

static double sigmoid_f64(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  }
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

static int rmsnorm_forward_f64(const double *x, const float *gamma, size_t H,
                               double eps, double *y_out, double *inv_rms_out) {
  double sum_sq = 0.0;
  for (size_t j = 0u; j < H; ++j) {
    sum_sq += x[j] * x[j];
  }
  const double ms = sum_sq / (double)H;
  const double inv_rms = 1.0 / sqrt(ms + eps);
  if (!isfinite(inv_rms)) {
    return 0;
  }
  for (size_t j = 0u; j < H; ++j) {
    const double v = x[j] * inv_rms * (double)gamma[j];
    if (!isfinite(v)) {
      return 0;
    }
    y_out[j] = v;
  }
  *inv_rms_out = inv_rms;
  return 1;
}

/* Accumulates into dx_acc[H] and dgamma_acc[H]. */
static int rmsnorm_backward_add_f64(const double *x, const float *gamma,
                                    double inv_rms, const double *dy, size_t H,
                                    double *dx_acc, double *dgamma_acc) {
  double g_dot = 0.0;
  for (size_t j = 0u; j < H; ++j) {
    const double dy_g = dy[j] * (double)gamma[j];
    g_dot += dy_g * x[j];
    dgamma_acc[j] += dy[j] * x[j] * inv_rms;
  }
  const double coeff = (inv_rms * inv_rms / (double)H) * g_dot;
  if (!isfinite(coeff)) {
    return 0;
  }
  for (size_t j = 0u; j < H; ++j) {
    const double dx = inv_rms * (dy[j] * (double)gamma[j] - x[j] * coeff);
    if (!isfinite(dx)) {
      return 0;
    }
    dx_acc[j] += dx;
  }
  return 1;
}

static int lm_logits_f64(const float *W_lm, const double *o, size_t V, size_t H,
                         double *logits_out) {
  for (size_t v = 0u; v < V; ++v) {
    const float *row = W_lm + v * H;
    double acc = 0.0;
    for (size_t j = 0u; j < H; ++j) {
      acc += (double)row[j] * o[j];
    }
    if (!isfinite(acc)) {
      return 0;
    }
    logits_out[v] = acc;
  }
  return 1;
}

static int log_softmax_ce_f64(const double *logits, size_t V, uint32_t target,
                              double *loss_out, double *lse_out) {
  double max_l = logits[0];
  for (size_t v = 1u; v < V; ++v) {
    if (logits[v] > max_l) {
      max_l = logits[v];
    }
  }
  double sum_exp = 0.0;
  for (size_t v = 0u; v < V; ++v) {
    sum_exp += exp(logits[v] - max_l);
  }
  const double lse = max_l + log(sum_exp);
  const double loss = lse - logits[target];
  if (!isfinite(lse) || !isfinite(loss)) {
    return 0;
  }
  *lse_out = lse;
  *loss_out = loss;
  return 1;
}

/* Single MTP block step k (0-indexed: k in 0..D-1) forward for one position. */
typedef struct mtp_block_cache {
  double h_in[ET_MTP_MAX_HIDDEN];
  double e_in[ET_MTP_MAX_HIDDEN];
  double u_h[ET_MTP_MAX_HIDDEN];
  double u_e[ET_MTP_MAX_HIDDEN];
  double inv_rms_h;
  double inv_rms_e;
  double z[ET_MTP_MAX_HIDDEN];
  double r[ET_MTP_MAX_HIDDEN];
  double inv_rms_f;
  double g[ET_MTP_MAX_FFN];
  double u[ET_MTP_MAX_FFN];
  double a[ET_MTP_MAX_FFN];
  double h_out[ET_MTP_MAX_HIDDEN];
  double o[ET_MTP_MAX_HIDDEN];
  double inv_rms_o;
} mtp_block_cache;

static int mtp_block_forward_one(const et_mtp_model_t *model, size_t k,
                                 const double *h_prev, uint32_t tok_embed,
                                 mtp_block_cache *c) {
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t F = (size_t)model->cfg.ffn_dim;
  const double eps = model->cfg.rms_eps;
  const et_mtp_layout_v1 *lay = &model->layout;
  const float *p = model->params;
  const size_t base = lay->head_offset[k];

  const float *E = p + lay->embed_offset;
  const float *gamma_h = p + base + lay->rel_gamma_h;
  const float *gamma_e = p + base + lay->rel_gamma_e;
  const float *W_proj = p + base + lay->rel_w_proj;
  const float *gamma_f = p + base + lay->rel_gamma_f;
  const float *W_gate = p + base + lay->rel_w_gate;
  const float *W_up = p + base + lay->rel_w_up;
  const float *W_down = p + base + lay->rel_w_down;
  const float *gamma_o = p + base + lay->rel_gamma_o;

  for (size_t j = 0u; j < H; ++j) {
    c->h_in[j] = h_prev[j];
    c->e_in[j] = (double)E[(size_t)tok_embed * H + j];
  }

  if (!rmsnorm_forward_f64(c->h_in, gamma_h, H, eps, c->u_h, &c->inv_rms_h) ||
      !rmsnorm_forward_f64(c->e_in, gamma_e, H, eps, c->u_e, &c->inv_rms_e)) {
    return 0;
  }

  /* z_k = W_proj_k @ concat(u_h, u_e), W_proj_k shape [H, 2*H]. */
  const size_t two_H = 2u * H;
  for (size_t i = 0u; i < H; ++i) {
    const float *row = W_proj + i * two_H;
    double acc = 0.0;
    for (size_t j = 0u; j < H; ++j) {
      acc += (double)row[j] * c->u_h[j];
    }
    for (size_t j = 0u; j < H; ++j) {
      acc += (double)row[H + j] * c->u_e[j];
    }
    if (!isfinite(acc)) {
      return 0;
    }
    c->z[i] = acc;
  }

  /* r = RMSNorm(z_k, gamma_f_k) */
  if (!rmsnorm_forward_f64(c->z, gamma_f, H, eps, c->r, &c->inv_rms_f)) {
    return 0;
  }

  /* SwiGLU_k(r) = W_down_k @ (SiLU(W_gate_k @ r) * (W_up_k @ r)) */
  for (size_t m = 0u; m < F; ++m) {
    const float *g_row = W_gate + m * H;
    const float *u_row = W_up + m * H;
    double gm = 0.0;
    double um = 0.0;
    for (size_t j = 0u; j < H; ++j) {
      gm += (double)g_row[j] * c->r[j];
      um += (double)u_row[j] * c->r[j];
    }
    const double sig = sigmoid_f64(gm);
    const double am = (gm * sig) * um;
    if (!isfinite(gm) || !isfinite(um) || !isfinite(am)) {
      return 0;
    }
    c->g[m] = gm;
    c->u[m] = um;
    c->a[m] = am;
  }

  for (size_t i = 0u; i < H; ++i) {
    const float *d_row = W_down + i * F;
    double acc = 0.0;
    for (size_t m = 0u; m < F; ++m) {
      acc += (double)d_row[m] * c->a[m];
    }
    const double hk = c->z[i] + acc;
    if (!isfinite(hk)) {
      return 0;
    }
    c->h_out[i] = hk;
  }

  /* o = RMSNorm(h^k, gamma_o_k) */
  if (!rmsnorm_forward_f64(c->h_out, gamma_o, H, eps, c->o, &c->inv_rms_o)) {
    return 0;
  }
  return 1;
}

/* --- Main Head Logits Helper --- */

et_mtp_status_t et_mtp_main_logits_v1(const et_mtp_model_t *model,
                                      const float *h0_vec, float *logits_out) {
  if (model == NULL || h0_vec == NULL || logits_out == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t V = (size_t)model->cfg.vocab_size;
  if (!aligned_span(h0_vec, _Alignof(float), H * sizeof(float)) ||
      !aligned_span(logits_out, _Alignof(float), V * sizeof(float)) ||
      spans_overlap(h0_vec, H * sizeof(float), logits_out, V * sizeof(float))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!all_f32_finite(h0_vec, H) ||
      !all_f32_finite(model->params, model->layout.total_params)) {
    return ET_MTP_ERR_NONFINITE;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    return ET_MTP_ERR_FP_ENV;
  }

  double h_f64[ET_MTP_MAX_HIDDEN];
  double o_f64[ET_MTP_MAX_HIDDEN];
  double logits_f64[ET_MTP_MAX_VOCAB];
  float staged[ET_MTP_MAX_VOCAB];
  double inv_rms = 0.0;

  for (size_t j = 0u; j < H; ++j) {
    h_f64[j] = (double)h0_vec[j];
  }
  const float *gamma_main = model->params + model->layout.gamma_main_offset;
  const float *W_lm = model->params + model->layout.lm_head_offset;

  int ok = rmsnorm_forward_f64(h_f64, gamma_main, H, model->cfg.rms_eps, o_f64,
                               &inv_rms) &&
           lm_logits_f64(W_lm, o_f64, V, H, logits_f64);
  for (size_t v = 0u; ok && v < V; ++v) {
    ok = finite_f64_to_f32(logits_f64[v], &staged[v]);
  }
  (void)fesetenv(&saved_env);
  if (!ok) {
    return ET_MTP_ERR_NONFINITE;
  }
  memcpy(logits_out, staged, V * sizeof(float));
  return ET_MTP_OK;
}

/* --- Shared Batch/Sequence Validation --- */

static et_mtp_status_t validate_seq_inputs(const et_mtp_model_t *model,
                                           size_t batch, size_t seq_len,
                                           const float *h0,
                                           const uint32_t *tokens) {
  if (model == NULL || h0 == NULL || tokens == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (batch < ET_MTP_MIN_BATCH || batch > ET_MTP_MAX_BATCH ||
      seq_len > ET_MTP_MAX_SEQ_LEN) {
    return ET_MTP_ERR_UNSUPPORTED;
  }
  /* Require seq_len >= mtp_depth + 2 so depth D has >= 1 valid target. */
  if (seq_len < (size_t)model->cfg.mtp_depth + 2u) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t bt = batch * seq_len;
  const size_t h0_elems = bt * H;
  if (!aligned_span(h0, _Alignof(float), h0_elems * sizeof(float)) ||
      !aligned_span(tokens, _Alignof(uint32_t), bt * sizeof(uint32_t))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  for (size_t i = 0u; i < bt; ++i) {
    if (tokens[i] >= model->cfg.vocab_size) {
      return ET_MTP_ERR_TOKEN_RANGE;
    }
  }
  if (!all_f32_finite(h0, h0_elems) ||
      !all_f32_finite(model->params, model->layout.total_params)) {
    return ET_MTP_ERR_NONFINITE;
  }
  return ET_MTP_OK;
}

/* --- Loss-Only Forward Pass --- */

et_mtp_status_t et_mtp_forward_v1(const et_mtp_model_t *model, size_t batch,
                                  size_t seq_len, const float *h0,
                                  const uint32_t *tokens,
                                  et_mtp_loss_stats_v1 *out_stats) {
  if (out_stats == NULL ||
      !aligned_span(out_stats, _Alignof(et_mtp_loss_stats_v1),
                    sizeof(*out_stats))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  const et_mtp_status_t vst =
      validate_seq_inputs(model, batch, seq_len, h0, tokens);
  if (vst != ET_MTP_OK) {
    return vst;
  }
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t V = (size_t)model->cfg.vocab_size;
  const size_t D = (size_t)model->cfg.mtp_depth;
  if (spans_overlap(out_stats, sizeof(*out_stats), h0,
                    batch * seq_len * H * sizeof(float)) ||
      spans_overlap(out_stats, sizeof(*out_stats), tokens,
                    batch * seq_len * sizeof(uint32_t))) {
    return ET_MTP_ERR_INVALID_ARG;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    return ET_MTP_ERR_FP_ENV;
  }

  mtp_block_cache cache;
  double h_cur[ET_MTP_MAX_HIDDEN];
  double o_main[ET_MTP_MAX_HIDDEN];
  double logits[ET_MTP_MAX_VOCAB];

  const float *W_lm = model->params + model->layout.lm_head_offset;
  const float *gamma_main = model->params + model->layout.gamma_main_offset;

  double sum_main = 0.0;
  double sum_depth[ET_MTP_MAX_DEPTH] = {0.0, 0.0, 0.0, 0.0};
  int ok = 1;

  for (size_t b = 0u; ok && b < batch; ++b) {
    const uint32_t *tok_row = tokens + b * seq_len;
    for (size_t t = 0u; ok && t + 1u < seq_len; ++t) {
      const float *h0_bt = h0 + (b * seq_len + t) * H;
      for (size_t j = 0u; j < H; ++j) {
        h_cur[j] = (double)h0_bt[j];
      }

      /* Main head (k = 0): predicts tokens[b, t + 1]. */
      double inv_rms_main = 0.0;
      double loss_s = 0.0;
      double lse = 0.0;
      if (!rmsnorm_forward_f64(h_cur, gamma_main, H, model->cfg.rms_eps, o_main,
                               &inv_rms_main) ||
          !lm_logits_f64(W_lm, o_main, V, H, logits) ||
          !log_softmax_ce_f64(logits, V, tok_row[t + 1u], &loss_s, &lse)) {
        ok = 0;
        break;
      }
      sum_main += loss_s;

      /* Sequential MTP heads k = 1 .. D (0-indexed 0 .. D - 1). */
      const size_t max_rem = seq_len - 2u - t;
      const size_t k_limit = (D < max_rem) ? D : max_rem;
      for (size_t k = 0u; k < k_limit; ++k) {
        const uint32_t embed_tok = tok_row[t + k + 1u];
        const uint32_t target_tok = tok_row[t + k + 2u];
        if (!mtp_block_forward_one(model, k, h_cur, embed_tok, &cache) ||
            !lm_logits_f64(W_lm, cache.o, V, H, logits) ||
            !log_softmax_ce_f64(logits, V, target_tok, &loss_s, &lse)) {
          ok = 0;
          break;
        }
        sum_depth[k] += loss_s;
        memcpy(h_cur, cache.h_out, H * sizeof(double));
      }
    }
  }

  et_mtp_loss_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  if (ok) {
    stats.main_token_count = (uint32_t)(batch * (seq_len - 1u));
    stats.loss_main = sum_main / (double)stats.main_token_count;
    double depth_sum = 0.0;
    for (size_t k = 0u; k < D; ++k) {
      stats.depth_token_count[k] = (uint32_t)(batch * (seq_len - 2u - k));
      stats.loss_depth[k] = sum_depth[k] / (double)stats.depth_token_count[k];
      depth_sum += stats.loss_depth[k];
    }
    stats.loss_mtp_avg = depth_sum / (double)D;
    stats.loss_total =
        stats.loss_main + model->cfg.lambda_mtp * stats.loss_mtp_avg;
    if (!isfinite(stats.loss_main) || !isfinite(stats.loss_mtp_avg) ||
        !isfinite(stats.loss_total)) {
      ok = 0;
    }
  }

  (void)fesetenv(&saved_env);
  if (!ok) {
    return ET_MTP_ERR_NONFINITE;
  }
  *out_stats = stats;
  return ET_MTP_OK;
}

/* --- Exact Analytic Forward + Reverse-Depth Backward Pass --- */

et_mtp_status_t et_mtp_forward_backward_v1(
    const et_mtp_model_t *model, size_t batch, size_t seq_len, const float *h0,
    const uint32_t *tokens, et_mtp_loss_stats_v1 *out_stats, float *d_h0_out,
    float *d_params_out, size_t d_params_count) {
  if (out_stats == NULL || d_h0_out == NULL || d_params_out == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  const et_mtp_status_t vst =
      validate_seq_inputs(model, batch, seq_len, h0, tokens);
  if (vst != ET_MTP_OK) {
    return vst;
  }
  const size_t total_params = model->layout.total_params;
  if (d_params_count != total_params) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t F = (size_t)model->cfg.ffn_dim;
  const size_t V = (size_t)model->cfg.vocab_size;
  const size_t D = (size_t)model->cfg.mtp_depth;
  const size_t h0_elems = batch * seq_len * H;
  const size_t h0_bytes = h0_elems * sizeof(float);
  const size_t tok_bytes = batch * seq_len * sizeof(uint32_t);
  const size_t dp_bytes = total_params * sizeof(float);

  if (!aligned_span(out_stats, _Alignof(et_mtp_loss_stats_v1),
                    sizeof(*out_stats)) ||
      !aligned_span(d_h0_out, _Alignof(float), h0_bytes) ||
      !aligned_span(d_params_out, _Alignof(float), dp_bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (spans_overlap(d_h0_out, h0_bytes, d_params_out, dp_bytes) ||
      spans_overlap(d_h0_out, h0_bytes, out_stats, sizeof(*out_stats)) ||
      spans_overlap(d_params_out, dp_bytes, out_stats, sizeof(*out_stats)) ||
      spans_overlap(d_h0_out, h0_bytes, h0, h0_bytes) ||
      spans_overlap(d_h0_out, h0_bytes, tokens, tok_bytes) ||
      spans_overlap(d_params_out, dp_bytes, h0, h0_bytes) ||
      spans_overlap(d_params_out, dp_bytes, tokens, tok_bytes) ||
      spans_overlap(out_stats, sizeof(*out_stats), h0, h0_bytes) ||
      spans_overlap(out_stats, sizeof(*out_stats), tokens, tok_bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }

  /* Allocate binary64 accumulators and staged outputs for failure atomicity. */
  double *d_params_f64 = (double *)calloc(total_params, sizeof(double));
  float *staged_d_h0 = (float *)calloc(h0_elems, sizeof(float));
  float *staged_d_params = (float *)calloc(total_params, sizeof(float));
  if (d_params_f64 == NULL || staged_d_h0 == NULL || staged_d_params == NULL) {
    free(d_params_f64);
    free(staged_d_h0);
    free(staged_d_params);
    return ET_MTP_ERR_ALLOC;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(d_params_f64);
    free(staged_d_h0);
    free(staged_d_params);
    return ET_MTP_ERR_FP_ENV;
  }

  const et_mtp_layout_v1 *lay = &model->layout;
  const float *p = model->params;
  const float *W_lm = p + lay->lm_head_offset;
  const float *gamma_main = p + lay->gamma_main_offset;
  double *d_E = d_params_f64 + lay->embed_offset;
  double *d_W_lm = d_params_f64 + lay->lm_head_offset;
  double *d_gamma_main = d_params_f64 + lay->gamma_main_offset;

  const uint32_t main_cnt = (uint32_t)(batch * (seq_len - 1u));
  const double scale_main = 1.0 / (double)main_cnt;
  double scale_depth[ET_MTP_MAX_DEPTH] = {0.0, 0.0, 0.0, 0.0};
  for (size_t k = 0u; k < D; ++k) {
    const uint32_t cnt_k = (uint32_t)(batch * (seq_len - 2u - k));
    scale_depth[k] =
        (model->cfg.lambda_mtp / (double)D) / (double)cnt_k;
  }

  mtp_block_cache caches[ET_MTP_MAX_DEPTH];
  double h0_f64[ET_MTP_MAX_HIDDEN];
  double o_main[ET_MTP_MAX_HIDDEN];
  double logits[ET_MTP_MAX_VOCAB];
  double dlogits[ET_MTP_MAX_VOCAB];
  double do_vec[ET_MTP_MAX_HIDDEN];
  double dh_carry[ET_MTP_MAX_HIDDEN];
  double dh_total[ET_MTP_MAX_HIDDEN];
  double dz[ET_MTP_MAX_HIDDEN];
  double dr[ET_MTP_MAX_HIDDEN];
  double dg[ET_MTP_MAX_FFN];
  double du[ET_MTP_MAX_FFN];
  double du_h[ET_MTP_MAX_HIDDEN];
  double du_e[ET_MTP_MAX_HIDDEN];
  double dh_prev[ET_MTP_MAX_HIDDEN];

  double sum_main = 0.0;
  double sum_depth[ET_MTP_MAX_DEPTH] = {0.0, 0.0, 0.0, 0.0};
  int ok = 1;

  for (size_t b = 0u; ok && b < batch; ++b) {
    const uint32_t *tok_row = tokens + b * seq_len;
    for (size_t t = 0u; ok && t + 1u < seq_len; ++t) {
      const float *h0_bt = h0 + (b * seq_len + t) * H;
      for (size_t j = 0u; j < H; ++j) {
        h0_f64[j] = (double)h0_bt[j];
      }

      /* 1. Forward replay vertical chain k = 0 .. k_limit - 1 for (b, t). */
      const size_t max_rem = seq_len - 2u - t;
      const size_t k_limit = (D < max_rem) ? D : max_rem;
      const double *prev_ptr = h0_f64;
      for (size_t k = 0u; k < k_limit; ++k) {
        const uint32_t embed_tok = tok_row[t + k + 1u];
        if (!mtp_block_forward_one(model, k, prev_ptr, embed_tok, &caches[k])) {
          ok = 0;
          break;
        }
        prev_ptr = caches[k].h_out;
      }
      if (!ok) {
        break;
      }

      /* 2. Reverse-depth backward sweep: k = k_limit - 1 down to 0. */
      memset(dh_carry, 0, H * sizeof(double));
      for (size_t rev = k_limit; rev > 0u; --rev) {
        const size_t k = rev - 1u;
        const mtp_block_cache *c = &caches[k];
        const size_t base = lay->head_offset[k];
        const uint32_t embed_tok = tok_row[t + k + 1u];
        const uint32_t target_tok = tok_row[t + k + 2u];

        const float *gamma_h = p + base + lay->rel_gamma_h;
        const float *gamma_e = p + base + lay->rel_gamma_e;
        const float *W_proj = p + base + lay->rel_w_proj;
        const float *gamma_f = p + base + lay->rel_gamma_f;
        const float *W_gate = p + base + lay->rel_w_gate;
        const float *W_up = p + base + lay->rel_w_up;
        const float *W_down = p + base + lay->rel_w_down;
        const float *gamma_o = p + base + lay->rel_gamma_o;

        double *d_gamma_h = d_params_f64 + base + lay->rel_gamma_h;
        double *d_gamma_e = d_params_f64 + base + lay->rel_gamma_e;
        double *d_W_proj = d_params_f64 + base + lay->rel_w_proj;
        double *d_gamma_f = d_params_f64 + base + lay->rel_gamma_f;
        double *d_W_gate = d_params_f64 + base + lay->rel_w_gate;
        double *d_W_up = d_params_f64 + base + lay->rel_w_up;
        double *d_W_down = d_params_f64 + base + lay->rel_w_down;
        double *d_gamma_o = d_params_f64 + base + lay->rel_gamma_o;

        /* 2a. Output head loss + backward for depth k. */
        double loss_s = 0.0;
        double lse = 0.0;
        if (!lm_logits_f64(W_lm, c->o, V, H, logits) ||
            !log_softmax_ce_f64(logits, V, target_tok, &loss_s, &lse)) {
          ok = 0;
          break;
        }
        sum_depth[k] += loss_s;

        const double sc = scale_depth[k];
        memset(do_vec, 0, H * sizeof(double));
        for (size_t v = 0u; v < V; ++v) {
          const double prob = exp(logits[v] - lse);
          const double dl = sc * (prob - ((v == (size_t)target_tok) ? 1.0 : 0.0));
          dlogits[v] = dl;
          double *d_row = d_W_lm + v * H;
          const float *w_row = W_lm + v * H;
          for (size_t j = 0u; j < H; ++j) {
            d_row[j] += dl * c->o[j];
            do_vec[j] += dl * (double)w_row[j];
          }
        }

        /* dh_total = dh_carry + d(RMSNorm(h^k, gamma_o_k))/dh^k */
        memcpy(dh_total, dh_carry, H * sizeof(double));
        if (!rmsnorm_backward_add_f64(c->h_out, gamma_o, c->inv_rms_o, do_vec,
                                      H, dh_total, d_gamma_o)) {
          ok = 0;
          break;
        }

        /* 2b. Residual SwiGLU backward: h^k = z_k + W_down @ (SiLU(g) * u). */
        memcpy(dz, dh_total, H * sizeof(double));
        for (size_t m = 0u; m < F; ++m) {
          double da = 0.0;
          for (size_t i = 0u; i < H; ++i) {
            da += dh_total[i] * (double)W_down[i * F + m];
            d_W_down[i * F + m] += dh_total[i] * c->a[m];
          }
          const double gm = c->g[m];
          const double sig = sigmoid_f64(gm);
          const double silu = gm * sig;
          const double silu_prime = sig * (1.0 + gm * (1.0 - sig));
          du[m] = da * silu;
          dg[m] = da * c->u[m] * silu_prime;
        }

        memset(dr, 0, H * sizeof(double));
        for (size_t m = 0u; m < F; ++m) {
          const float *wg_row = W_gate + m * H;
          const float *wu_row = W_up + m * H;
          double *dwg_row = d_W_gate + m * H;
          double *dwu_row = d_W_up + m * H;
          const double dgm = dg[m];
          const double dum = du[m];
          for (size_t j = 0u; j < H; ++j) {
            dr[j] += dgm * (double)wg_row[j] + dum * (double)wu_row[j];
            dwg_row[j] += dgm * c->r[j];
            dwu_row[j] += dum * c->r[j];
          }
        }

        if (!rmsnorm_backward_add_f64(c->z, gamma_f, c->inv_rms_f, dr, H, dz,
                                      d_gamma_f)) {
          ok = 0;
          break;
        }

        /* 2c. Projection backward: z_k = W_proj_k @ concat(u_h, u_e). */
        memset(du_h, 0, H * sizeof(double));
        memset(du_e, 0, H * sizeof(double));
        const size_t two_H = 2u * H;
        for (size_t i = 0u; i < H; ++i) {
          const double dzi = dz[i];
          const float *wp_row = W_proj + i * two_H;
          double *dwp_row = d_W_proj + i * two_H;
          for (size_t j = 0u; j < H; ++j) {
            dwp_row[j] += dzi * c->u_h[j];
            du_h[j] += dzi * (double)wp_row[j];
          }
          for (size_t j = 0u; j < H; ++j) {
            dwp_row[H + j] += dzi * c->u_e[j];
            du_e[j] += dzi * (double)wp_row[H + j];
          }
        }

        /* 2d. Backprop through RMSNorm(E[embed_tok]) and RMSNorm(h^{k-1}). */
        double *d_e_row = d_E + (size_t)embed_tok * H;
        if (!rmsnorm_backward_add_f64(c->e_in, gamma_e, c->inv_rms_e, du_e, H,
                                      d_e_row, d_gamma_e)) {
          ok = 0;
          break;
        }
        memset(dh_prev, 0, H * sizeof(double));
        if (!rmsnorm_backward_add_f64(c->h_in, gamma_h, c->inv_rms_h, du_h, H,
                                      dh_prev, d_gamma_h)) {
          ok = 0;
          break;
        }
        memcpy(dh_carry, dh_prev, H * sizeof(double));
      }
      if (!ok) {
        break;
      }

      /* 3. Main head (k = 0) forward + backward, combined with dh_carry. */
      double inv_rms_main = 0.0;
      double loss_s = 0.0;
      double lse = 0.0;
      if (!rmsnorm_forward_f64(h0_f64, gamma_main, H, model->cfg.rms_eps,
                               o_main, &inv_rms_main) ||
          !lm_logits_f64(W_lm, o_main, V, H, logits) ||
          !log_softmax_ce_f64(logits, V, tok_row[t + 1u], &loss_s, &lse)) {
        ok = 0;
        break;
      }
      sum_main += loss_s;

      memset(do_vec, 0, H * sizeof(double));
      for (size_t v = 0u; v < V; ++v) {
        const double prob = exp(logits[v] - lse);
        const double dl =
            scale_main * (prob - ((v == (size_t)tok_row[t + 1u]) ? 1.0 : 0.0));
        double *d_row = d_W_lm + v * H;
        const float *w_row = W_lm + v * H;
        for (size_t j = 0u; j < H; ++j) {
          d_row[j] += dl * o_main[j];
          do_vec[j] += dl * (double)w_row[j];
        }
      }

      if (!rmsnorm_backward_add_f64(h0_f64, gamma_main, inv_rms_main, do_vec, H,
                                    dh_carry, d_gamma_main)) {
        ok = 0;
        break;
      }

      float *dh0_bt = staged_d_h0 + (b * seq_len + t) * H;
      for (size_t j = 0u; j < H; ++j) {
        if (!finite_f64_to_f32(dh_carry[j], &dh0_bt[j])) {
          ok = 0;
          break;
        }
      }
    }
  }

  et_mtp_loss_stats_v1 stats;
  memset(&stats, 0, sizeof(stats));
  if (ok) {
    stats.main_token_count = main_cnt;
    stats.loss_main = sum_main / (double)main_cnt;
    double depth_sum = 0.0;
    for (size_t k = 0u; k < D; ++k) {
      stats.depth_token_count[k] = (uint32_t)(batch * (seq_len - 2u - k));
      stats.loss_depth[k] = sum_depth[k] / (double)stats.depth_token_count[k];
      depth_sum += stats.loss_depth[k];
    }
    stats.loss_mtp_avg = depth_sum / (double)D;
    stats.loss_total =
        stats.loss_main + model->cfg.lambda_mtp * stats.loss_mtp_avg;
    if (!isfinite(stats.loss_main) || !isfinite(stats.loss_mtp_avg) ||
        !isfinite(stats.loss_total)) {
      ok = 0;
    }
  }

  for (size_t i = 0u; ok && i < total_params; ++i) {
    if (!finite_f64_to_f32(d_params_f64[i], &staged_d_params[i])) {
      ok = 0;
    }
  }

  (void)fesetenv(&saved_env);
  if (ok) {
    *out_stats = stats;
    memcpy(d_h0_out, staged_d_h0, h0_bytes);
    memcpy(d_params_out, staged_d_params, dp_bytes);
  }

  free(d_params_f64);
  free(staged_d_h0);
  free(staged_d_params);
  return ok ? ET_MTP_OK : ET_MTP_ERR_NONFINITE;
}

/* --- Failure-Atomic Binary64 Decoupled AdamW Step --- */

et_mtp_status_t et_mtp_adamw_step_v1(et_mtp_model_t *model,
                                     const float *d_params,
                                     size_t d_params_count,
                                     et_mtp_step_stats_v1 *out_stats) {
  if (model == NULL || d_params == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  const size_t n = model->layout.total_params;
  if (d_params_count != n) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }
  if (!aligned_span(d_params, _Alignof(float), n * sizeof(float))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (out_stats != NULL &&
      (!aligned_span(out_stats, _Alignof(et_mtp_step_stats_v1),
                     sizeof(*out_stats)) ||
       spans_overlap(out_stats, sizeof(*out_stats), d_params,
                     n * sizeof(float)))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!all_f32_finite(d_params, n) || !all_f32_finite(model->params, n) ||
      !all_f32_finite(model->m, n) || !all_f32_finite(model->v, n)) {
    return ET_MTP_ERR_NONFINITE;
  }

  float *tmp_p = (float *)malloc(n * sizeof(float));
  float *tmp_m = (float *)malloc(n * sizeof(float));
  float *tmp_v = (float *)malloc(n * sizeof(float));
  if (tmp_p == NULL || tmp_m == NULL || tmp_v == NULL) {
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_MTP_ERR_ALLOC;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_MTP_ERR_FP_ENV;
  }

  double sq_sum = 0.0;
  for (size_t i = 0u; i < n; ++i) {
    const double g = (double)d_params[i];
    sq_sum += g * g;
  }
  const double grad_norm = sqrt(sq_sum);
  double clip_scale = 1.0;
  uint32_t clipped = 0u;
  if (model->cfg.max_grad_norm > 0.0 && grad_norm > model->cfg.max_grad_norm) {
    clip_scale = model->cfg.max_grad_norm / grad_norm;
    clipped = 1u;
  }

  const uint64_t next_step = model->step + 1u;
  const double b1 = model->cfg.beta1;
  const double b2 = model->cfg.beta2;
  const double bc1 = 1.0 - ipow_f64(b1, next_step);
  const double bc2 = 1.0 - ipow_f64(b2, next_step);
  const double lr = model->cfg.lr;
  const double eps = model->cfg.eps;
  const double wd = model->cfg.weight_decay;

  int ok = isfinite(grad_norm) && isfinite(clip_scale) && bc1 > 0.0 && bc2 > 0.0;
  for (size_t i = 0u; ok && i < n; ++i) {
    const double g = (double)d_params[i] * clip_scale;
    const double m_new = b1 * (double)model->m[i] + (1.0 - b1) * g;
    const double v_new = b2 * (double)model->v[i] + (1.0 - b2) * g * g;
    const double m_hat = m_new / bc1;
    const double v_hat = v_new / bc2;
    const double p_old = (double)model->params[i];
    const double p_new = p_old - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p_old);
    if (!finite_f64_to_f32(p_new, &tmp_p[i]) ||
        !finite_f64_to_f32(m_new, &tmp_m[i]) ||
        !finite_f64_to_f32(v_new, &tmp_v[i]) || tmp_v[i] < 0.0f) {
      ok = 0;
    }
  }

  (void)fesetenv(&saved_env);
  if (ok) {
    memcpy(model->params, tmp_p, n * sizeof(float));
    memcpy(model->m, tmp_m, n * sizeof(float));
    memcpy(model->v, tmp_v, n * sizeof(float));
    model->step = next_step;
    if (out_stats != NULL) {
      out_stats->step = next_step;
      out_stats->grad_norm = grad_norm;
      out_stats->clip_scale = clip_scale;
      out_stats->clipped = clipped;
      out_stats->reserved = 0u;
    }
  }

  free(tmp_p);
  free(tmp_m);
  free(tmp_v);
  return ok ? ET_MTP_OK : ET_MTP_ERR_NONFINITE;
}

/* --- Self-Speculative MTP Draft + Exact Verification --- */

static uint32_t argmax_f64(const double *vals, size_t n) {
  uint32_t best_idx = 0u;
  double best_val = vals[0];
  for (size_t i = 1u; i < n; ++i) {
    if (vals[i] > best_val) {
      best_val = vals[i];
      best_idx = (uint32_t)i;
    }
  }
  return best_idx;
}

static int softmax_temp_f64(const double *logits, size_t V, double temperature,
                            double *probs_out) {
  double max_scaled = logits[0] / temperature;
  for (size_t v = 1u; v < V; ++v) {
    const double s = logits[v] / temperature;
    if (!isfinite(s)) {
      return 0;
    }
    if (s > max_scaled) {
      max_scaled = s;
    }
  }
  double sum_exp = 0.0;
  for (size_t v = 0u; v < V; ++v) {
    const double e = exp((logits[v] / temperature) - max_scaled);
    probs_out[v] = e;
    sum_exp += e;
  }
  if (!isfinite(sum_exp) || sum_exp <= 0.0) {
    return 0;
  }
  for (size_t v = 0u; v < V; ++v) {
    probs_out[v] /= sum_exp;
    if (!isfinite(probs_out[v])) {
      return 0;
    }
  }
  return 1;
}

static uint32_t sample_categorical_f64(const double *probs, size_t V,
                                       double u) {
  double cdf = 0.0;
  uint32_t last_nonzero = 0u;
  for (size_t v = 0u; v < V; ++v) {
    if (probs[v] > 0.0) {
      last_nonzero = (uint32_t)v;
    }
    cdf += probs[v];
    if (u < cdf) {
      return (uint32_t)v;
    }
  }
  return last_nonzero;
}

et_mtp_status_t et_mtp_draft_verify_v1(
    const et_mtp_model_t *model, const float *h0_prefix, uint32_t anchor_token,
    const float *target_logits, double temperature, uint32_t eos_token_id,
    const et_mtp_rng_state_v1 *rng_in, uint32_t *draft_tokens_out,
    uint32_t *out_tokens, uint32_t out_capacity,
    et_mtp_verify_result_v1 *out_result) {
  if (model == NULL || h0_prefix == NULL || target_logits == NULL ||
      draft_tokens_out == NULL || out_tokens == NULL || out_result == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  if (!isfinite(temperature) || temperature < 0.0 || signbit(temperature)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const int greedy = (temperature == 0.0);
  if (!greedy && rng_in == NULL) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  const size_t H = (size_t)model->cfg.hidden_dim;
  const size_t V = (size_t)model->cfg.vocab_size;
  const size_t D = (size_t)model->cfg.mtp_depth;

  if (anchor_token != ET_MTP_ANCHOR_FROM_MAIN &&
      anchor_token >= model->cfg.vocab_size) {
    return ET_MTP_ERR_TOKEN_RANGE;
  }
  if (eos_token_id != ET_MTP_NO_EOS && eos_token_id >= model->cfg.vocab_size) {
    return ET_MTP_ERR_TOKEN_RANGE;
  }
  if (out_capacity < (uint32_t)(D + 1u)) {
    return ET_MTP_ERR_SHAPE_MISMATCH;
  }

  const size_t h0_bytes = H * sizeof(float);
  const size_t tlog_elems = (D + 1u) * V;
  const size_t tlog_bytes = tlog_elems * sizeof(float);
  const size_t draft_bytes = D * sizeof(uint32_t);
  const size_t out_bytes = (size_t)out_capacity * sizeof(uint32_t);
  if (!aligned_span(h0_prefix, _Alignof(float), h0_bytes) ||
      !aligned_span(target_logits, _Alignof(float), tlog_bytes) ||
      !aligned_span(draft_tokens_out, _Alignof(uint32_t), draft_bytes) ||
      !aligned_span(out_tokens, _Alignof(uint32_t), out_bytes) ||
      !aligned_span(out_result, _Alignof(et_mtp_verify_result_v1),
                    sizeof(*out_result)) ||
      (rng_in != NULL &&
       !aligned_span(rng_in, _Alignof(et_mtp_rng_state_v1), sizeof(*rng_in)))) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (spans_overlap(draft_tokens_out, draft_bytes, out_tokens, out_bytes) ||
      spans_overlap(draft_tokens_out, draft_bytes, out_result,
                    sizeof(*out_result)) ||
      spans_overlap(out_tokens, out_bytes, out_result, sizeof(*out_result)) ||
      spans_overlap(draft_tokens_out, draft_bytes, h0_prefix, h0_bytes) ||
      spans_overlap(draft_tokens_out, draft_bytes, target_logits, tlog_bytes) ||
      spans_overlap(out_tokens, out_bytes, h0_prefix, h0_bytes) ||
      spans_overlap(out_tokens, out_bytes, target_logits, tlog_bytes) ||
      spans_overlap(out_result, sizeof(*out_result), h0_prefix, h0_bytes) ||
      spans_overlap(out_result, sizeof(*out_result), target_logits,
                    tlog_bytes)) {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!all_f32_finite(h0_prefix, H) ||
      !all_f32_finite(target_logits, tlog_elems) ||
      !all_f32_finite(model->params, model->layout.total_params)) {
    return ET_MTP_ERR_NONFINITE;
  }

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0) {
    return ET_MTP_ERR_FP_ENV;
  }

  et_mtp_rng_state_v1 rng = {0u, 0u};
  if (!greedy) {
    rng = *rng_in;
  }
  uint64_t rng_draws = 0u;

  double h_cur[ET_MTP_MAX_HIDDEN];
  double o_main[ET_MTP_MAX_HIDDEN];
  double logits[ET_MTP_MAX_VOCAB];
  double q_probs[ET_MTP_MAX_DEPTH][ET_MTP_MAX_VOCAB];
  double p_probs[ET_MTP_MAX_VOCAB];
  double residual[ET_MTP_MAX_VOCAB];
  mtp_block_cache cache;

  for (size_t j = 0u; j < H; ++j) {
    h_cur[j] = (double)h0_prefix[j];
  }
  const float *W_lm = model->params + model->layout.lm_head_offset;
  const float *gamma_main = model->params + model->layout.gamma_main_offset;

  int ok = 1;
  uint32_t anchor_used = anchor_token;
  if (anchor_token == ET_MTP_ANCHOR_FROM_MAIN) {
    double inv_rms_main = 0.0;
    if (!rmsnorm_forward_f64(h_cur, gamma_main, H, model->cfg.rms_eps, o_main,
                             &inv_rms_main) ||
        !lm_logits_f64(W_lm, o_main, V, H, logits)) {
      ok = 0;
    } else if (greedy) {
      anchor_used = argmax_f64(logits, V);
    } else {
      if (!softmax_temp_f64(logits, V, temperature, p_probs)) {
        ok = 0;
      } else {
        const double u = rng_draw_uniform(&rng, &rng_draws);
        anchor_used = sample_categorical_f64(p_probs, V, u);
      }
    }
  }

  /* 1. Sequential MTP draft proposal across depths k = 0 .. D - 1. */
  uint32_t staged_drafts[ET_MTP_MAX_DEPTH] = {0u, 0u, 0u, 0u};
  uint32_t prev_tok = anchor_used;
  for (size_t k = 0u; ok && k < D; ++k) {
    if (!mtp_block_forward_one(model, k, h_cur, prev_tok, &cache) ||
        !lm_logits_f64(W_lm, cache.o, V, H, logits)) {
      ok = 0;
      break;
    }
    uint32_t draft_tok = 0u;
    if (greedy) {
      draft_tok = argmax_f64(logits, V);
    } else {
      if (!softmax_temp_f64(logits, V, temperature, q_probs[k])) {
        ok = 0;
        break;
      }
      const double u = rng_draw_uniform(&rng, &rng_draws);
      draft_tok = sample_categorical_f64(q_probs[k], V, u);
    }
    staged_drafts[k] = draft_tok;
    prev_tok = draft_tok;
    memcpy(h_cur, cache.h_out, H * sizeof(double));
  }

  /* 2. Exact verification against target_logits[0 .. D, V]. */
  uint32_t staged_out[ET_MTP_MAX_DEPTH + 1u] = {0u, 0u, 0u, 0u, 0u};
  et_mtp_verify_result_v1 res;
  memset(&res, 0, sizeof(res));
  res.anchor_token_used = anchor_used;

  for (size_t i = 0u; ok && i < D; ++i) {
    const float *t_row = target_logits + i * V;
    for (size_t v = 0u; v < V; ++v) {
      logits[v] = (double)t_row[v];
    }
    const uint32_t draft_tok = staged_drafts[i];
    if (greedy) {
      const uint32_t target_tok = argmax_f64(logits, V);
      if (draft_tok == target_tok) {
        staged_out[res.emitted_count++] = draft_tok;
        res.accepted_count += 1u;
        if (eos_token_id != ET_MTP_NO_EOS && draft_tok == eos_token_id) {
          res.eos_hit = 1u;
          break;
        }
      } else {
        staged_out[res.emitted_count++] = target_tok;
        res.rejected = 1u;
        if (eos_token_id != ET_MTP_NO_EOS && target_tok == eos_token_id) {
          res.eos_hit = 1u;
        }
        break;
      }
    } else {
      if (!softmax_temp_f64(logits, V, temperature, p_probs)) {
        ok = 0;
        break;
      }
      const double px = p_probs[draft_tok];
      const double qx = q_probs[i][draft_tok];
      const double u_acc = rng_draw_uniform(&rng, &rng_draws);
      const int accepted = (px >= qx) || (qx > 0.0 && (u_acc * qx) < px);
      if (accepted) {
        staged_out[res.emitted_count++] = draft_tok;
        res.accepted_count += 1u;
        if (eos_token_id != ET_MTP_NO_EOS && draft_tok == eos_token_id) {
          res.eos_hit = 1u;
          break;
        }
      } else {
        double res_sum = 0.0;
        for (size_t v = 0u; v < V; ++v) {
          const double d = p_probs[v] - q_probs[i][v];
          residual[v] = (d > 0.0) ? d : 0.0;
          res_sum += residual[v];
        }
        uint32_t corr_tok = 0u;
        const double u_res = rng_draw_uniform(&rng, &rng_draws);
        if (res_sum > 0.0) {
          for (size_t v = 0u; v < V; ++v) {
            residual[v] /= res_sum;
          }
          corr_tok = sample_categorical_f64(residual, V, u_res);
        } else {
          corr_tok = sample_categorical_f64(p_probs, V, u_res);
        }
        staged_out[res.emitted_count++] = corr_tok;
        res.rejected = 1u;
        if (eos_token_id != ET_MTP_NO_EOS && corr_tok == eos_token_id) {
          res.eos_hit = 1u;
        }
        break;
      }
    }
  }

  /* 3. Bonus token from target_logits[D, :] if all D drafts accepted. */
  if (ok && !res.rejected && !res.eos_hit) {
    const float *b_row = target_logits + D * V;
    for (size_t v = 0u; v < V; ++v) {
      logits[v] = (double)b_row[v];
    }
    uint32_t bonus_tok = 0u;
    if (greedy) {
      bonus_tok = argmax_f64(logits, V);
    } else {
      if (!softmax_temp_f64(logits, V, temperature, p_probs)) {
        ok = 0;
      } else {
        const double u_bonus = rng_draw_uniform(&rng, &rng_draws);
        bonus_tok = sample_categorical_f64(p_probs, V, u_bonus);
      }
    }
    if (ok) {
      staged_out[res.emitted_count++] = bonus_tok;
      if (eos_token_id != ET_MTP_NO_EOS && bonus_tok == eos_token_id) {
        res.eos_hit = 1u;
      }
    }
  }

  (void)fesetenv(&saved_env);
  if (!ok) {
    return ET_MTP_ERR_NONFINITE;
  }

  res.rng_draws = rng_draws;
  if (greedy) {
    res.next_rng.seed = 0u;
    res.next_rng.counter = 0u;
  } else {
    res.next_rng = rng;
  }

  memcpy(draft_tokens_out, staged_drafts, D * sizeof(uint32_t));
  memcpy(out_tokens, staged_out, (size_t)res.emitted_count * sizeof(uint32_t));
  *out_result = res;
  return ET_MTP_OK;
}

/* --- Atomic 0600 Checkpoint I/O ("ETMT1000" v1.0) --- */

static int write_all_fd(int fd, const uint8_t *data, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, data + off, len - off);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    off += (size_t)n;
  }
  return 0;
}

static int read_all_fd(int fd, uint8_t *dst, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, dst + off, len - off);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    off += (size_t)n;
  }
  return 0;
}

static int fsync_fd_retry(int fd) {
  while (fsync(fd) != 0) {
    if (errno == EINTR) {
      continue;
    }
    return -1;
  }
  return 0;
}

typedef struct path_parts {
  char *dir;
  char *base;
} path_parts;

static int split_path(const char *path, path_parts *out) {
  out->dir = NULL;
  out->base = NULL;
  if (path == NULL || path[0] == '\0') {
    return -1;
  }
  const size_t len = strlen(path);
  if (path[len - 1u] == '/') {
    return -1;
  }
  const char *slash = strrchr(path, '/');
  if (slash == NULL) {
    out->dir = strdup(".");
    out->base = strdup(path);
  } else if (slash == path) {
    out->dir = strdup("/");
    out->base = strdup(slash + 1);
  } else {
    const size_t dlen = (size_t)(slash - path);
    out->dir = (char *)malloc(dlen + 1u);
    if (out->dir != NULL) {
      memcpy(out->dir, path, dlen);
      out->dir[dlen] = '\0';
    }
    out->base = strdup(slash + 1);
  }
  if (out->dir == NULL || out->base == NULL || out->base[0] == '\0' ||
      strcmp(out->base, ".") == 0 || strcmp(out->base, "..") == 0) {
    free(out->dir);
    free(out->base);
    out->dir = NULL;
    out->base = NULL;
    return -1;
  }
  return 0;
}

static void free_path_parts(path_parts *p) {
  free(p->dir);
  free(p->base);
  p->dir = NULL;
  p->base = NULL;
}

static et_mtp_status_t atomic_write_0600(const char *path, const uint8_t *data,
                                         size_t len) {
  path_parts parts;
  if (split_path(path, &parts) != 0) {
    return ET_MTP_ERR_INVALID_ARG;
  }

  int dir_fd;
  do {
    dir_fd = open(parts.dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  } while (dir_fd < 0 && errno == EINTR);
  if (dir_fd < 0) {
    free_path_parts(&parts);
    return ET_MTP_ERR_IO;
  }

  /* Reject existing symlink or non-regular destination before staging. */
  struct stat dst_st;
  if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0) {
    if (S_ISLNK(dst_st.st_mode)) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_MTP_ERR_SYMLINK;
    }
    if (!S_ISREG(dst_st.st_mode)) {
      close(dir_fd);
      free_path_parts(&parts);
      return ET_MTP_ERR_IO;
    }
  } else if (errno != ENOENT) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_MTP_ERR_IO;
  }

  char tmp_name[128];
  int tmp_fd = -1;
  const unsigned long pid = (unsigned long)getpid();
  for (unsigned int attempt = 0u; attempt < 64u; ++attempt) {
    const int n = snprintf(tmp_name, sizeof(tmp_name),
                           ".etmt.tmp.%lu.%u", pid, attempt);
    if (n <= 0 || (size_t)n >= sizeof(tmp_name)) {
      break;
    }
    tmp_fd = openat(dir_fd, tmp_name,
                    O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                    0600);
    if (tmp_fd >= 0) {
      break;
    }
    if (errno != EEXIST) {
      break;
    }
  }
  if (tmp_fd < 0) {
    close(dir_fd);
    free_path_parts(&parts);
    return ET_MTP_ERR_IO;
  }

  et_mtp_status_t st = ET_MTP_OK;
  if (fchmod(tmp_fd, 0600) != 0 || write_all_fd(tmp_fd, data, len) != 0 ||
      fsync_fd_retry(tmp_fd) != 0) {
    st = ET_MTP_ERR_IO;
  }
  if (close(tmp_fd) != 0 && st == ET_MTP_OK) {
    st = ET_MTP_ERR_IO;
  }

  if (st == ET_MTP_OK) {
    if (fstatat(dir_fd, parts.base, &dst_st, AT_SYMLINK_NOFOLLOW) == 0) {
      if (S_ISLNK(dst_st.st_mode)) {
        st = ET_MTP_ERR_SYMLINK;
      } else if (!S_ISREG(dst_st.st_mode)) {
        st = ET_MTP_ERR_IO;
      }
    }
    if (st == ET_MTP_OK &&
        (renameat(dir_fd, tmp_name, dir_fd, parts.base) != 0 ||
         fsync_fd_retry(dir_fd) != 0)) {
      st = ET_MTP_ERR_IO;
    }
  }
  if (st != ET_MTP_OK) {
    (void)unlinkat(dir_fd, tmp_name, 0);
  }

  close(dir_fd);
  free_path_parts(&parts);
  return st;
}

static uint64_t expected_ckpt_payload_bytes(const et_mtp_model_t *model) {
  return 3u * (uint64_t)model->layout.total_params * (uint64_t)sizeof(float);
}

et_mtp_status_t et_mtp_checkpoint_save_v1(const et_mtp_model_t *model,
                                          const char *path) {
  if (model == NULL || path == NULL || path[0] == '\0') {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }
  const size_t n = model->layout.total_params;
  if (!all_f32_finite(model->params, n) || !all_f32_finite(model->m, n) ||
      !all_f32_finite(model->v, n)) {
    return ET_MTP_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < n; ++i) {
    if (model->v[i] < 0.0f) {
      return ET_MTP_ERR_NONFINITE;
    }
  }

  const uint64_t payload_bytes = expected_ckpt_payload_bytes(model);
  const uint64_t total_bytes = (uint64_t)ET_MTP_CKPT_HEADER_BYTES +
                               payload_bytes +
                               (uint64_t)ET_MTP_CKPT_CHECKSUM_BYTES;
  if (total_bytes > ET_MTP_MAX_CKPT_BYTES) {
    return ET_MTP_ERR_UNSUPPORTED;
  }

  uint8_t *buf = (uint8_t *)calloc(1u, (size_t)total_bytes);
  if (buf == NULL) {
    return ET_MTP_ERR_ALLOC;
  }

  /* Fixed 128-byte little-endian header. */
  memcpy(buf + 0u, ET_MTP_CKPT_MAGIC, ET_MTP_CKPT_MAGIC_BYTES);
  put_u16_le(buf + 8u, (uint16_t)ET_MTP_CKPT_VERSION_MAJOR);
  put_u16_le(buf + 10u, (uint16_t)ET_MTP_CKPT_VERSION_MINOR);
  put_u32_le(buf + 12u, model->cfg.vocab_size);
  put_u32_le(buf + 16u, model->cfg.hidden_dim);
  put_u32_le(buf + 20u, model->cfg.ffn_dim);
  put_u32_le(buf + 24u, model->cfg.mtp_depth);
  put_u32_le(buf + 28u, model->cfg.tie_weights);
  put_u64_le(buf + 32u, (uint64_t)n);
  put_u64_le(buf + 40u, model->step);
  put_u64_le(buf + 48u, f64_to_bits(model->cfg.rms_eps));
  put_u64_le(buf + 56u, f64_to_bits(model->cfg.lambda_mtp));
  put_u64_le(buf + 64u, f64_to_bits(model->cfg.lr));
  put_u64_le(buf + 72u, f64_to_bits(model->cfg.beta1));
  put_u64_le(buf + 80u, f64_to_bits(model->cfg.beta2));
  put_u64_le(buf + 88u, f64_to_bits(model->cfg.eps));
  put_u64_le(buf + 96u, f64_to_bits(model->cfg.weight_decay));
  put_u64_le(buf + 104u, f64_to_bits(model->cfg.max_grad_norm));
  put_u64_le(buf + 112u, payload_bytes);
  put_u64_le(buf + 120u, 0u);

  size_t off = ET_MTP_CKPT_HEADER_BYTES;
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(model->params[i])));
    off += 4u;
  }
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(model->m[i])));
    off += 4u;
  }
  for (size_t i = 0u; i < n; ++i) {
    put_u32_le(buf + off, f32_to_bits(canonical_f32(model->v[i])));
    off += 4u;
  }

  uint8_t digest[32];
  (void)et_mtp_sha256_bytes_v1(buf, off, digest);
  memcpy(buf + off, digest, ET_MTP_CKPT_CHECKSUM_BYTES);

  const et_mtp_status_t st = atomic_write_0600(path, buf, (size_t)total_bytes);
  free(buf);
  return st;
}

et_mtp_status_t et_mtp_checkpoint_load_v1(et_mtp_model_t *model,
                                          const char *path) {
  if (model == NULL || path == NULL || path[0] == '\0') {
    return ET_MTP_ERR_INVALID_ARG;
  }
  if (!fp_environment_supported()) {
    return ET_MTP_ERR_FP_ENV;
  }

  int fd;
  do {
    fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return (errno == ELOOP) ? ET_MTP_ERR_SYMLINK : ET_MTP_ERR_IO;
  }

  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    return ET_MTP_ERR_IO;
  }

  const uint64_t min_bytes = (uint64_t)ET_MTP_CKPT_HEADER_BYTES +
                             (uint64_t)ET_MTP_CKPT_CHECKSUM_BYTES;
  if (st.st_size < 0 || (uint64_t)st.st_size < min_bytes ||
      (uint64_t)st.st_size > ET_MTP_MAX_CKPT_BYTES) {
    close(fd);
    return (st.st_size >= 0 && (uint64_t)st.st_size < min_bytes)
               ? ET_MTP_ERR_TRUNCATED
               : ET_MTP_ERR_IO;
  }

  const size_t file_len = (size_t)st.st_size;
  uint8_t *buf = (uint8_t *)malloc(file_len);
  if (buf == NULL) {
    close(fd);
    return ET_MTP_ERR_ALLOC;
  }
  if (read_all_fd(fd, buf, file_len) != 0) {
    free(buf);
    close(fd);
    return ET_MTP_ERR_IO;
  }
  uint8_t extra = 0u;
  const ssize_t tail_rc = read(fd, &extra, 1u);
  close(fd);
  if (tail_rc != 0) {
    free(buf);
    return ET_MTP_ERR_TRUNCATED;
  }

  /* 1. Magic and version checks. */
  if (memcmp(buf, ET_MTP_CKPT_MAGIC, ET_MTP_CKPT_MAGIC_BYTES) != 0) {
    free(buf);
    return ET_MTP_ERR_BAD_MAGIC;
  }
  const uint16_t vmaj = get_u16_le(buf + 8u);
  const uint16_t vmin = get_u16_le(buf + 10u);
  if (vmaj != (uint16_t)ET_MTP_CKPT_VERSION_MAJOR ||
      vmin != (uint16_t)ET_MTP_CKPT_VERSION_MINOR) {
    free(buf);
    return ET_MTP_ERR_VERSION_MISMATCH;
  }

  /* 2. Exact length & SHA-256 verification before trusting payload fields. */
  const uint64_t hdr_payload = get_u64_le(buf + 112u);
  const uint64_t expected_file_len = (uint64_t)ET_MTP_CKPT_HEADER_BYTES +
                                     hdr_payload +
                                     (uint64_t)ET_MTP_CKPT_CHECKSUM_BYTES;
  if (hdr_payload > ET_MTP_MAX_CKPT_BYTES ||
      expected_file_len != (uint64_t)file_len) {
    free(buf);
    return ET_MTP_ERR_TRUNCATED;
  }

  uint8_t actual_digest[32];
  (void)et_mtp_sha256_bytes_v1(buf, file_len - ET_MTP_CKPT_CHECKSUM_BYTES,
                               actual_digest);
  if (memcmp(actual_digest, buf + file_len - ET_MTP_CKPT_CHECKSUM_BYTES,
             ET_MTP_CKPT_CHECKSUM_BYTES) != 0) {
    free(buf);
    return ET_MTP_ERR_CHECKSUM_MISMATCH;
  }

  /* 3. Exact model & AdamW configuration match. */
  const uint32_t vocab_size = get_u32_le(buf + 12u);
  const uint32_t hidden_dim = get_u32_le(buf + 16u);
  const uint32_t ffn_dim = get_u32_le(buf + 20u);
  const uint32_t mtp_depth = get_u32_le(buf + 24u);
  const uint32_t tie_weights = get_u32_le(buf + 28u);
  const uint64_t total_params = get_u64_le(buf + 32u);
  const uint64_t step = get_u64_le(buf + 40u);
  const double rms_eps = bits_to_f64(get_u64_le(buf + 48u));
  const double lambda_mtp = bits_to_f64(get_u64_le(buf + 56u));
  const double lr = bits_to_f64(get_u64_le(buf + 64u));
  const double beta1 = bits_to_f64(get_u64_le(buf + 72u));
  const double beta2 = bits_to_f64(get_u64_le(buf + 80u));
  const double eps = bits_to_f64(get_u64_le(buf + 88u));
  const double wd = bits_to_f64(get_u64_le(buf + 96u));
  const double max_gn = bits_to_f64(get_u64_le(buf + 104u));
  const uint64_t reserved = get_u64_le(buf + 120u);

  const size_t n = model->layout.total_params;
  if (reserved != 0u || vocab_size != model->cfg.vocab_size ||
      hidden_dim != model->cfg.hidden_dim || ffn_dim != model->cfg.ffn_dim ||
      mtp_depth != model->cfg.mtp_depth ||
      tie_weights != model->cfg.tie_weights ||
      total_params != (uint64_t)n ||
      hdr_payload != expected_ckpt_payload_bytes(model) ||
      f64_to_bits(rms_eps) != f64_to_bits(model->cfg.rms_eps) ||
      f64_to_bits(lambda_mtp) != f64_to_bits(model->cfg.lambda_mtp) ||
      f64_to_bits(lr) != f64_to_bits(model->cfg.lr) ||
      f64_to_bits(beta1) != f64_to_bits(model->cfg.beta1) ||
      f64_to_bits(beta2) != f64_to_bits(model->cfg.beta2) ||
      f64_to_bits(eps) != f64_to_bits(model->cfg.eps) ||
      f64_to_bits(wd) != f64_to_bits(model->cfg.weight_decay) ||
      f64_to_bits(max_gn) != f64_to_bits(model->cfg.max_grad_norm)) {
    free(buf);
    return ET_MTP_ERR_CONFIG_MISMATCH;
  }

  /* 4. Stage and validate all float payloads before atomic commit. */
  float *tmp_p = (float *)malloc(n * sizeof(float));
  float *tmp_m = (float *)malloc(n * sizeof(float));
  float *tmp_v = (float *)malloc(n * sizeof(float));
  if (tmp_p == NULL || tmp_m == NULL || tmp_v == NULL) {
    free(buf);
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return ET_MTP_ERR_ALLOC;
  }

  size_t off = ET_MTP_CKPT_HEADER_BYTES;
  et_mtp_status_t val_st = ET_MTP_OK;
  for (size_t i = 0u; i < n; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    if (!isfinite(v)) {
      val_st = ET_MTP_ERR_NONFINITE;
      break;
    }
    tmp_p[i] = canonical_f32(v);
  }
  for (size_t i = 0u; val_st == ET_MTP_OK && i < n; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    if (!isfinite(v)) {
      val_st = ET_MTP_ERR_NONFINITE;
      break;
    }
    tmp_m[i] = canonical_f32(v);
  }
  for (size_t i = 0u; val_st == ET_MTP_OK && i < n; ++i) {
    const float v = bits_to_f32(get_u32_le(buf + off));
    off += 4u;
    if (!isfinite(v) || v < 0.0f) {
      val_st = ET_MTP_ERR_NONFINITE;
      break;
    }
    tmp_v[i] = canonical_f32(v);
  }

  free(buf);
  if (val_st != ET_MTP_OK) {
    free(tmp_p);
    free(tmp_m);
    free(tmp_v);
    return val_st;
  }

  memcpy(model->params, tmp_p, n * sizeof(float));
  memcpy(model->m, tmp_m, n * sizeof(float));
  memcpy(model->v, tmp_v, n * sizeof(float));
  model->step = step;

  free(tmp_p);
  free(tmp_m);
  free(tmp_v);
  return ET_MTP_OK;
}
