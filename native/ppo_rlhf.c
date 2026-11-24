#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/ppo_rlhf_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
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
               "PPO RLHF engine requires 8-bit bytes and IEEE-754 binary32/64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "PPO RLHF engine requires IEEE-754 binary32/64 evaluation");

/* Max trainable parameter count for built-in Actor-Critic:
 *   V*D + V + D + 1 <= 64*64 + 64 + 64 + 1 = 4225. */
#define PPO_MAX_PARAMS                                                         \
  ((ET_PPO_MAX_MODEL_VOCAB * ET_PPO_MAX_MODEL_FEAT) + ET_PPO_MAX_MODEL_VOCAB + \
   ET_PPO_MAX_MODEL_FEAT + 1u)
#define PPO_CKPT_HEADER_BYTES 184u
#define PPO_CKPT_TRAILER_BYTES ET_PPO_SHA256_BYTES
#define PPO_CKPT_MAX_PATH 3968u

_Static_assert(PPO_CKPT_HEADER_BYTES + 12u * PPO_MAX_PARAMS +
                       PPO_CKPT_TRAILER_BYTES <=
                   ET_PPO_CHECKPOINT_MAX_BYTES,
               "ETPP1000 max checkpoint buffer too small");

/* ---- Standalone SHA-256 (canonical digests & ETPP1000 integrity) ---- */

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

static int ct_equal_32(const uint8_t a[32], const uint8_t b[32]) {
  uint8_t diff = 0u;
  for (size_t i = 0u; i < 32u; ++i)
    diff |= (uint8_t)(a[i] ^ b[i]);
  return diff == 0u;
}

static void encode_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static uint32_t decode_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) | ((uint32_t)in[2] << 16u) |
         ((uint32_t)in[3] << 24u);
}

static void encode_u64_le(uint8_t out[8], uint64_t value) {
  for (size_t i = 0u; i < 8u; ++i)
    out[i] = (uint8_t)((value >> (8u * i)) & 0xffu);
}

static uint64_t decode_u64_le(const uint8_t in[8]) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    v |= ((uint64_t)in[i]) << (8u * i);
  return v;
}

static void encode_f32_le(uint8_t out[4], float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  encode_u32_le(out, bits);
}

static float decode_f32_le(const uint8_t in[4]) {
  const uint32_t bits = decode_u32_le(in);
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
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

static int outputs_alias_inputs_or_each_other(const span_entry *inputs,
                                              size_t num_inputs,
                                              const span_entry *outputs,
                                              size_t num_outputs) {
  for (size_t i = 0u; i < num_outputs; ++i) {
    if (outputs[i].ptr == NULL || outputs[i].bytes == 0u)
      continue;
    for (size_t j = 0u; j < i; ++j) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, outputs[j].ptr,
                        outputs[j].bytes))
        return 1;
    }
    for (size_t j = 0u; j < num_inputs; ++j) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, inputs[j].ptr,
                        inputs[j].bytes))
        return 1;
    }
  }
  return 0;
}

static et_ppo_status check_buf(const void *ptr, size_t alignment,
                               size_t bytes) {
  if (ptr == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(ptr, alignment, bytes))
    return ET_PPO_ERR_ALIGNMENT;
  return ET_PPO_OK;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static int all_nonneg_finite_f32(const float *data, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(data[i]) || data[i] < 0.0f)
      return 0;
  }
  return 1;
}

static int validate_mask_count(size_t bt, const uint8_t *mask,
                               uint32_t *out_active) {
  uint32_t active = 0u;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] > 1u)
      return 0;
    active += (uint32_t)mask[i];
  }
  *out_active = active;
  return 1;
}

static int validate_actions(size_t bt, uint32_t vocab_size,
                            const int32_t *actions) {
  for (size_t i = 0u; i < bt; ++i) {
    if (actions[i] < 0 || (uint32_t)actions[i] >= vocab_size)
      return 0;
  }
  return 1;
}

static double pow_u64_f64(double base, uint64_t exp) {
  double result = 1.0;
  double factor = base;
  uint64_t e = exp;
  while (e != 0u) {
    if ((e & UINT64_C(1)) != 0u)
      result *= factor;
    factor *= factor;
    e >>= 1u;
  }
  return result;
}

static et_ppo_status validate_bt_dims(uint32_t batch_size, uint32_t seq_len,
                                      size_t *out_bt) {
  if (batch_size < 1u || batch_size > ET_PPO_MAX_BATCH || seq_len < 1u ||
      seq_len > ET_PPO_MAX_SEQ_LEN)
    return ET_PPO_ERR_SHAPE;
  if (out_bt != NULL)
    *out_bt = (size_t)batch_size * (size_t)seq_len;
  return ET_PPO_OK;
}

static et_ppo_status validate_btv_dims(uint32_t batch_size, uint32_t seq_len,
                                       uint32_t vocab_size, size_t *out_bt,
                                       size_t *out_btv) {
  size_t bt = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if (vocab_size < 1u || vocab_size > ET_PPO_MAX_VOCAB)
    return ET_PPO_ERR_SHAPE;
  if (bt > (size_t)ET_PPO_MAX_LOGITS_ELEMS / (size_t)vocab_size)
    return ET_PPO_ERR_SHAPE;
  if (out_bt != NULL)
    *out_bt = bt;
  if (out_btv != NULL)
    *out_btv = bt * (size_t)vocab_size;
  return ET_PPO_OK;
}

static et_ppo_status validate_adamw_config_internal(
    const et_ppo_adamw_config *cfg) {
  if (cfg == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(cfg, alignof(et_ppo_adamw_config), sizeof(*cfg)))
    return ET_PPO_ERR_ALIGNMENT;
  if (cfg->reserved[0] != 0u || cfg->reserved[1] != 0u)
    return ET_PPO_ERR_ARGUMENT;
  if (!isfinite(cfg->lr) || !isfinite(cfg->beta1) || !isfinite(cfg->beta2) ||
      !isfinite(cfg->eps) || !isfinite(cfg->weight_decay) ||
      !isfinite(cfg->max_grad_norm))
    return ET_PPO_ERR_NONFINITE;
  if (!(cfg->lr > 0.0f) || !(cfg->beta1 >= 0.0f && cfg->beta1 < 1.0f) ||
      !(cfg->beta2 >= 0.0f && cfg->beta2 < 1.0f) || !(cfg->eps > 0.0f) ||
      !(cfg->weight_decay >= 0.0f) || !(cfg->max_grad_norm >= 0.0f))
    return ET_PPO_ERR_ARGUMENT;
  return ET_PPO_OK;
}

static et_ppo_status validate_loss_config_internal(
    const et_ppo_loss_config *cfg, size_t *out_bt, size_t *out_btv) {
  if (cfg == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(cfg, alignof(et_ppo_loss_config), sizeof(*cfg)))
    return ET_PPO_ERR_ALIGNMENT;
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  if (cfg->reserved != 0u)
    return ET_PPO_ERR_ARGUMENT;
  et_ppo_status st = validate_btv_dims(cfg->batch_size, cfg->seq_len,
                                       cfg->vocab_size, out_bt, out_btv);
  if (st != ET_PPO_OK)
    return st;
  if (cfg->use_value_clip > 1u)
    return ET_PPO_ERR_ARGUMENT;
  if (!isfinite(cfg->clip_eps_low) || !isfinite(cfg->clip_eps_high) ||
      !isfinite(cfg->value_clip_eps) || !isfinite(cfg->vf_coef) ||
      !isfinite(cfg->ent_coef))
    return ET_PPO_ERR_NONFINITE;
  if (!(cfg->clip_eps_low > 0.0f && cfg->clip_eps_low <= 1.0f) ||
      !(cfg->clip_eps_high > 0.0f) || !(cfg->vf_coef >= 0.0f) ||
      !(cfg->ent_coef >= 0.0f))
    return ET_PPO_ERR_ARGUMENT;
  if (cfg->use_value_clip == 1u) {
    if (!(cfg->value_clip_eps > 0.0f))
      return ET_PPO_ERR_ARGUMENT;
  } else {
    if (!(cfg->value_clip_eps >= 0.0f))
      return ET_PPO_ERR_ARGUMENT;
  }
  return ET_PPO_OK;
}

et_ppo_status et_ppo_validate_loss_config(const et_ppo_loss_config *cfg) {
  return validate_loss_config_internal(cfg, NULL, NULL);
}

et_ppo_status et_ppo_model_param_count(uint32_t feat_dim, uint32_t vocab_size,
                                       size_t *out_count) {
  if (out_count == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(out_count, alignof(size_t), sizeof(*out_count)))
    return ET_PPO_ERR_ALIGNMENT;
  if (feat_dim < 1u || feat_dim > ET_PPO_MAX_MODEL_FEAT || vocab_size < 2u ||
      vocab_size > ET_PPO_MAX_MODEL_VOCAB)
    return ET_PPO_ERR_SHAPE;
  *out_count = (size_t)vocab_size * (size_t)feat_dim + (size_t)vocab_size +
               (size_t)feat_dim + 1u;
  return ET_PPO_OK;
}

et_ppo_status et_ppo_ref_param_count(uint32_t feat_dim, uint32_t vocab_size,
                                     size_t *out_count) {
  if (out_count == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(out_count, alignof(size_t), sizeof(*out_count)))
    return ET_PPO_ERR_ALIGNMENT;
  if (feat_dim < 1u || feat_dim > ET_PPO_MAX_MODEL_FEAT || vocab_size < 2u ||
      vocab_size > ET_PPO_MAX_MODEL_VOCAB)
    return ET_PPO_ERR_SHAPE;
  *out_count = (size_t)vocab_size * (size_t)feat_dim + (size_t)vocab_size;
  return ET_PPO_OK;
}

et_ppo_status et_ppo_validate_trainer_config(const et_ppo_trainer_config *cfg) {
  if (cfg == NULL)
    return ET_PPO_ERR_ARGUMENT;
  if (!aligned_span(cfg, alignof(et_ppo_trainer_config), sizeof(*cfg)))
    return ET_PPO_ERR_ALIGNMENT;
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u;
  et_ppo_status st = validate_loss_config_internal(&cfg->loss_cfg, &bt, NULL);
  if (st != ET_PPO_OK)
    return st;
  st = validate_adamw_config_internal(&cfg->optim_cfg);
  if (st != ET_PPO_OK)
    return st;
  if (cfg->reserved != 0u || cfg->reserved2 != 0u)
    return ET_PPO_ERR_ARGUMENT;
  if (cfg->feat_dim < 1u || cfg->feat_dim > ET_PPO_MAX_MODEL_FEAT ||
      cfg->loss_cfg.vocab_size < 2u ||
      cfg->loss_cfg.vocab_size > ET_PPO_MAX_MODEL_VOCAB ||
      bt > ET_PPO_TRAINER_MAX_BT)
    return ET_PPO_ERR_SHAPE;
  if (cfg->ppo_epochs < 1u || cfg->ppo_epochs > ET_PPO_MAX_EPOCHS)
    return ET_PPO_ERR_ARGUMENT;
  if (cfg->whiten_advantages > 1u)
    return ET_PPO_ERR_ARGUMENT;
  if (!isfinite(cfg->gamma) || !isfinite(cfg->gae_lambda) ||
      !isfinite(cfg->beta_kl) || !isfinite(cfg->adv_eps) ||
      !isfinite(cfg->target_kl))
    return ET_PPO_ERR_NONFINITE;
  if (!(cfg->gamma >= 0.0f && cfg->gamma <= 1.0f) ||
      !(cfg->gae_lambda >= 0.0f && cfg->gae_lambda <= 1.0f) ||
      !(cfg->beta_kl >= 0.0f) || !(cfg->target_kl >= 0.0f))
    return ET_PPO_ERR_ARGUMENT;
  if (cfg->whiten_advantages == 1u) {
    if (!(cfg->adv_eps > 0.0f))
      return ET_PPO_ERR_ARGUMENT;
  } else {
    if (!(cfg->adv_eps >= 0.0f))
      return ET_PPO_ERR_ARGUMENT;
  }
  return ET_PPO_OK;
}

/* ---- Canonical Weight SHA-256 ---- */

et_ppo_status et_ppo_weight_sha256_v1(size_t float_count, const float *weights,
                                      uint8_t out_digest[ET_PPO_SHA256_BYTES]) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  if (float_count == 0u || float_count > ET_PPO_MAX_LOGITS_ELEMS)
    return ET_PPO_ERR_SHAPE;
  const size_t bytes = float_count * sizeof(float);
  et_ppo_status st = check_buf(weights, alignof(float), bytes);
  if (st != ET_PPO_OK)
    return st;
  st = check_buf(out_digest, 1u, ET_PPO_SHA256_BYTES);
  if (st != ET_PPO_OK)
    return st;
  if (spans_overlap(weights, bytes, out_digest, ET_PPO_SHA256_BYTES))
    return ET_PPO_ERR_ALIAS;
  if (!all_finite_f32(weights, float_count))
    return ET_PPO_ERR_NONFINITE;

  sha256_state state;
  sha256_init(&state);
  uint8_t chunk[64];
  size_t i = 0u;
  while (i < float_count) {
    size_t take = float_count - i;
    if (take > 16u)
      take = 16u;
    for (size_t k = 0u; k < take; ++k)
      encode_f32_le(chunk + k * 4u, weights[i + k]);
    sha256_update(&state, chunk, take * 4u);
    i += take;
  }
  sha256_final(&state, out_digest);
  return ET_PPO_OK;
}

/* ---- Per-token KL-shaped rewards ---- */

et_ppo_status et_ppo_kl_shaped_rewards_v1(uint32_t batch_size, uint32_t seq_len,
                                          float beta_kl, const float *r_env,
                                          const float *logpi_old,
                                          const float *logpi_ref,
                                          const uint8_t *mask,
                                          float *out_r_shaped) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if (!isfinite(beta_kl))
    return ET_PPO_ERR_NONFINITE;
  if (!(beta_kl >= 0.0f))
    return ET_PPO_ERR_ARGUMENT;

  const size_t fbytes = bt * sizeof(float);
  if ((st = check_buf(r_env, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(logpi_old, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(logpi_ref, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(out_r_shaped, alignof(float), fbytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[4] = {
      {r_env, fbytes}, {logpi_old, fbytes}, {logpi_ref, fbytes}, {mask, bt}};
  const span_entry outputs[1] = {{out_r_shaped, fbytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 4u, outputs, 1u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (!all_finite_f32(r_env, bt) || !all_finite_f32(logpi_old, bt) ||
      !all_finite_f32(logpi_ref, bt))
    return ET_PPO_ERR_NONFINITE;

  /* Pass 1: validate finiteness before mutating out_r_shaped. */
  const double beta_d = (double)beta_kl;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      const double v =
          (double)r_env[i] - beta_d * ((double)logpi_old[i] - (double)logpi_ref[i]);
      if (!isfinite((float)v))
        return ET_PPO_ERR_OVERFLOW;
    }
  }
  /* Pass 2: commit. */
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] == 0u) {
      out_r_shaped[i] = 0.0f;
    } else {
      const double v =
          (double)r_env[i] - beta_d * ((double)logpi_old[i] - (double)logpi_ref[i]);
      out_r_shaped[i] = (float)v;
    }
  }
  return ET_PPO_OK;
}

/* ---- Generalized Advantage Estimation (GAE) ---- */

et_ppo_status et_ppo_gae_v1(uint32_t batch_size, uint32_t seq_len, float gamma,
                            float gae_lambda, const float *r_shaped,
                            const float *values, const uint8_t *mask,
                            float *out_advantages, float *out_returns) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if (!isfinite(gamma) || !isfinite(gae_lambda))
    return ET_PPO_ERR_NONFINITE;
  if (!(gamma >= 0.0f && gamma <= 1.0f) ||
      !(gae_lambda >= 0.0f && gae_lambda <= 1.0f))
    return ET_PPO_ERR_ARGUMENT;

  const size_t fbytes = bt * sizeof(float);
  if ((st = check_buf(r_shaped, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(values, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(out_advantages, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(out_returns, alignof(float), fbytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[3] = {
      {r_shaped, fbytes}, {values, fbytes}, {mask, bt}};
  const span_entry outputs[2] = {
      {out_advantages, fbytes}, {out_returns, fbytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 3u, outputs, 2u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (active == 0u)
    return ET_PPO_ERR_EMPTY_MASK;
  if (!all_finite_f32(r_shaped, bt) || !all_finite_f32(values, bt))
    return ET_PPO_ERR_NONFINITE;

  const double gamma_d = (double)gamma;
  const double lam_d = (double)gae_lambda;

  /* Pass 1: dry run verifying no binary32 overflow before mutating outputs. */
  for (uint32_t b = 0u; b < batch_size; ++b) {
    double next_v = 0.0;
    double next_adv = 0.0;
    const size_t row = (size_t)b * (size_t)seq_len;
    for (uint32_t ti = seq_len; ti > 0u; --ti) {
      const size_t idx = row + (size_t)(ti - 1u);
      if (mask[idx] == 0u) {
        next_v = 0.0;
        next_adv = 0.0;
      } else {
        const double v_t = (double)values[idx];
        const double r_t = (double)r_shaped[idx];
        const double delta = r_t + gamma_d * next_v - v_t;
        const double adv = delta + gamma_d * lam_d * next_adv;
        const double ret = adv + v_t;
        if (!isfinite((float)adv) || !isfinite((float)ret))
          return ET_PPO_ERR_OVERFLOW;
        next_v = v_t;
        next_adv = adv;
      }
    }
  }

  /* Pass 2: commit exact reverse-time GAE and returns. */
  for (uint32_t b = 0u; b < batch_size; ++b) {
    double next_v = 0.0;
    double next_adv = 0.0;
    const size_t row = (size_t)b * (size_t)seq_len;
    for (uint32_t ti = seq_len; ti > 0u; --ti) {
      const size_t idx = row + (size_t)(ti - 1u);
      if (mask[idx] == 0u) {
        out_advantages[idx] = 0.0f;
        out_returns[idx] = 0.0f;
        next_v = 0.0;
        next_adv = 0.0;
      } else {
        const double v_t = (double)values[idx];
        const double r_t = (double)r_shaped[idx];
        const double delta = r_t + gamma_d * next_v - v_t;
        const double adv = delta + gamma_d * lam_d * next_adv;
        const double ret = adv + v_t;
        out_advantages[idx] = (float)adv;
        out_returns[idx] = (float)ret;
        next_v = v_t;
        next_adv = adv;
      }
    }
  }
  return ET_PPO_OK;
}

/* ---- Exact Masked Advantage Whitening ---- */

et_ppo_status et_ppo_whiten_advantages_v1(uint32_t batch_size, uint32_t seq_len,
                                          float eps, const float *advantages,
                                          const uint8_t *mask,
                                          float *out_whitened) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if (!isfinite(eps))
    return ET_PPO_ERR_NONFINITE;
  if (!(eps > 0.0f))
    return ET_PPO_ERR_ARGUMENT;

  const size_t fbytes = bt * sizeof(float);
  if ((st = check_buf(advantages, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(out_whitened, alignof(float), fbytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[2] = {{advantages, fbytes}, {mask, bt}};
  const span_entry outputs[1] = {{out_whitened, fbytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 2u, outputs, 1u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (active == 0u)
    return ET_PPO_ERR_EMPTY_MASK;
  if (!all_finite_f32(advantages, bt))
    return ET_PPO_ERR_NONFINITE;

  /* Direct binary64 division sum / count guarantees all-equal binary32
   * advantages (for any active count <= 65536) have zero rounding error in
   * sum and exact mean == advantages[i], producing exact +0.0f output. */
  double sum = 0.0;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u)
      sum += (double)advantages[i];
  }
  const double count_d = (double)active;
  const double mean = sum / count_d;
  if (!isfinite(mean))
    return ET_PPO_ERR_OVERFLOW;

  double var_sum = 0.0;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      const double diff = (double)advantages[i] - mean;
      var_sum += diff * diff;
    }
  }
  const double var = var_sum / count_d;
  const double denom = sqrt(var + (double)eps);
  if (!isfinite(var) || !isfinite(denom) || !(denom > 0.0))
    return ET_PPO_ERR_OVERFLOW;

  /* Pass 1: verify finiteness before writing output. */
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      const double diff = (double)advantages[i] - mean;
      const double norm = diff / denom;
      if (!isfinite((float)norm))
        return ET_PPO_ERR_OVERFLOW;
    }
  }
  /* Pass 2: commit. */
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] == 0u) {
      out_whitened[i] = 0.0f;
    } else {
      const double diff = (double)advantages[i] - mean;
      const double norm = diff == 0.0 ? 0.0 : (diff / denom);
      out_whitened[i] = (float)norm;
    }
  }
  return ET_PPO_OK;
}

/* ---- Numerically Stable Log-Softmax & Entropy Helpers ---- */

static int row_lse_and_entropy(const float *row, uint32_t vocab_size,
                               int32_t action, double *out_lse,
                               double *out_logpi, double *out_entropy) {
  double max_z = (double)row[0];
  for (uint32_t v = 1u; v < vocab_size; ++v) {
    const double zv = (double)row[v];
    if (zv > max_z)
      max_z = zv;
  }
  double sum_exp = 0.0;
  for (uint32_t v = 0u; v < vocab_size; ++v)
    sum_exp += exp((double)row[v] - max_z);
  const double lse = max_z + log(sum_exp);
  if (!isfinite(lse))
    return 0;

  const double logpi = (double)row[(uint32_t)action] - lse;
  if (!isfinite(logpi))
    return 0;

  double entropy = 0.0;
  if (out_entropy != NULL) {
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      const double lp = (double)row[v] - lse;
      const double p = exp(lp);
      if (p > 0.0)
        entropy -= p * lp;
    }
    if (!isfinite(entropy))
      return 0;
    *out_entropy = entropy;
  }
  *out_lse = lse;
  *out_logpi = logpi;
  return 1;
}

et_ppo_status et_ppo_token_logprobs_v1(uint32_t batch_size, uint32_t seq_len,
                                       uint32_t vocab_size,
                                       const float *policy_logits,
                                       const int32_t *actions,
                                       const uint8_t *mask,
                                       float *out_logprobs) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u, btv = 0u;
  et_ppo_status st =
      validate_btv_dims(batch_size, seq_len, vocab_size, &bt, &btv);
  if (st != ET_PPO_OK)
    return st;

  const size_t lbytes = btv * sizeof(float);
  const size_t abytes = bt * sizeof(int32_t);
  const size_t obytes = bt * sizeof(float);
  if ((st = check_buf(policy_logits, alignof(float), lbytes)) != ET_PPO_OK ||
      (st = check_buf(actions, alignof(int32_t), abytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(out_logprobs, alignof(float), obytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[3] = {
      {policy_logits, lbytes}, {actions, abytes}, {mask, bt}};
  const span_entry outputs[1] = {{out_logprobs, obytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 3u, outputs, 1u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (!validate_actions(bt, vocab_size, actions))
    return ET_PPO_ERR_INDEX;
  if (!all_finite_f32(policy_logits, btv))
    return ET_PPO_ERR_NONFINITE;

  /* Pass 1: verify finiteness. */
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      double lse = 0.0, lp = 0.0;
      if (!row_lse_and_entropy(policy_logits + i * (size_t)vocab_size,
                               vocab_size, actions[i], &lse, &lp, NULL) ||
          !isfinite((float)lp))
        return ET_PPO_ERR_OVERFLOW;
    }
  }
  /* Pass 2: commit. */
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] == 0u) {
      out_logprobs[i] = 0.0f;
    } else {
      double lse = 0.0, lp = 0.0;
      (void)row_lse_and_entropy(policy_logits + i * (size_t)vocab_size,
                                vocab_size, actions[i], &lse, &lp, NULL);
      out_logprobs[i] = (float)lp;
    }
  }
  return ET_PPO_OK;
}

/* ---- Core PPO-Clip + Clipped Value + Entropy Token Kernel ---- */

typedef struct token_ppo_eval {
  double lse;
  double logpi;
  double entropy;
  double pg_obj;
  double d_pg_d_logpi;
  double v_loss;
  double d_v;
  double approx_kl;
  uint32_t clipped;
  uint32_t v_clipped;
} token_ppo_eval;

static et_ppo_status eval_ppo_token(const et_ppo_loss_config *cfg,
                                    const float *row_logits, float value,
                                    int32_t action, float old_logprob,
                                    float old_value, float advantage,
                                    float ret_val, token_ppo_eval *out) {
  double lse = 0.0, logpi = 0.0, entropy = 0.0;
  if (!row_lse_and_entropy(row_logits, cfg->vocab_size, action, &lse, &logpi,
                           &entropy))
    return ET_PPO_ERR_OVERFLOW;

  const double delta_lp = logpi - (double)old_logprob;
  if (!isfinite(delta_lp) || delta_lp > (double)ET_PPO_MAX_LOG_DIFF)
    return ET_PPO_ERR_OVERFLOW;

  const double ratio = exp(delta_lp);
  const double c_low = 1.0 - (double)cfg->clip_eps_low;
  const double c_high = 1.0 + (double)cfg->clip_eps_high;
  const double a_d = (double)advantage;
  const double ratio_clip =
      ratio < c_low ? c_low : (ratio > c_high ? c_high : ratio);
  const uint32_t clipped =
      ((a_d > 0.0 && ratio > c_high) || (a_d < 0.0 && ratio < c_low)) ? 1u : 0u;
  const double pg_obj = clipped ? (ratio_clip * a_d) : (ratio * a_d);
  const double d_pg_d_logpi = clipped ? 0.0 : (ratio * a_d);
  const double approx_kl = 0.5 * delta_lp * delta_lp;

  const double v_d = (double)value;
  const double r_d = (double)ret_val;
  const double diff_u = v_d - r_d;
  const double sq_u = diff_u * diff_u;
  double v_loss = 0.5 * sq_u;
  double d_v = diff_u;
  uint32_t v_clipped = 0u;
  if (cfg->use_value_clip == 1u) {
    const double vo_d = (double)old_value;
    const double eps_v = (double)cfg->value_clip_eps;
    const double dv = v_d - vo_d;
    if (dv < -eps_v || dv > eps_v) {
      const double v_c = dv < -eps_v ? (vo_d - eps_v) : (vo_d + eps_v);
      const double diff_c = v_c - r_d;
      const double sq_c = diff_c * diff_c;
      if (sq_c > sq_u) {
        v_loss = 0.5 * sq_c;
        d_v = 0.0;
        v_clipped = 1u;
      }
    }
  }

  if (!isfinite(pg_obj) || !isfinite(d_pg_d_logpi) || !isfinite(v_loss) ||
      !isfinite(d_v) || !isfinite(approx_kl))
    return ET_PPO_ERR_OVERFLOW;

  out->lse = lse;
  out->logpi = logpi;
  out->entropy = entropy;
  out->pg_obj = pg_obj;
  out->d_pg_d_logpi = d_pg_d_logpi;
  out->v_loss = v_loss;
  out->d_v = d_v;
  out->approx_kl = approx_kl;
  out->clipped = clipped;
  out->v_clipped = v_clipped;
  return ET_PPO_OK;
}

static double compute_explained_variance(size_t bt, uint32_t active,
                                         const float *values,
                                         const float *returns,
                                         const uint8_t *mask) {
  const double count_d = (double)active;
  double sum_r = 0.0;
  double sum_e = 0.0;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      const double r = (double)returns[i];
      const double e = r - (double)values[i];
      sum_r += r;
      sum_e += e;
    }
  }
  const double mean_r = sum_r / count_d;
  const double mean_e = sum_e / count_d;
  double var_r = 0.0;
  double var_e = 0.0;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      const double r = (double)returns[i];
      const double dr = r - mean_r;
      const double de = (r - (double)values[i]) - mean_e;
      var_r += dr * dr;
      var_e += de * de;
    }
  }
  var_r /= count_d;
  var_e /= count_d;
  if (!(var_r > 0.0) || !isfinite(var_r) || !isfinite(var_e))
    return 0.0;
  const double ev = 1.0 - (var_e / var_r);
  return isfinite(ev) ? ev : 0.0;
}

static et_ppo_status ppo_loss_internal(
    const et_ppo_loss_config *cfg, const float *policy_logits,
    const float *values, const int32_t *actions, const float *old_logprobs,
    const float *old_values, const float *advantages, const float *returns,
    const uint8_t *mask, float *out_d_policy_logits, float *out_d_values,
    et_ppo_loss_metrics *out_metrics, int want_backward) {
  size_t bt = 0u, btv = 0u;
  et_ppo_status st = validate_loss_config_internal(cfg, &bt, &btv);
  if (st != ET_PPO_OK)
    return st;

  const size_t lbytes = btv * sizeof(float);
  const size_t fbytes = bt * sizeof(float);
  const size_t abytes = bt * sizeof(int32_t);
  if ((st = check_buf(policy_logits, alignof(float), lbytes)) != ET_PPO_OK ||
      (st = check_buf(values, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(actions, alignof(int32_t), abytes)) != ET_PPO_OK ||
      (st = check_buf(old_logprobs, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(old_values, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(advantages, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(returns, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK)
    return st;

  if (want_backward) {
    if ((st = check_buf(out_d_policy_logits, alignof(float), lbytes)) !=
            ET_PPO_OK ||
        (st = check_buf(out_d_values, alignof(float), fbytes)) != ET_PPO_OK)
      return st;
    if (out_metrics != NULL &&
        (st = check_buf(out_metrics, alignof(et_ppo_loss_metrics),
                        sizeof(*out_metrics))) != ET_PPO_OK)
      return st;
  } else {
    if ((st = check_buf(out_metrics, alignof(et_ppo_loss_metrics),
                        sizeof(*out_metrics))) != ET_PPO_OK)
      return st;
  }

  const span_entry inputs[8] = {{policy_logits, lbytes},
                                {values, fbytes},
                                {actions, abytes},
                                {old_logprobs, fbytes},
                                {old_values, fbytes},
                                {advantages, fbytes},
                                {returns, fbytes},
                                {mask, bt}};
  const span_entry outputs[3] = {
      {out_d_policy_logits, want_backward ? lbytes : 0u},
      {out_d_values, want_backward ? fbytes : 0u},
      {out_metrics, out_metrics != NULL ? sizeof(*out_metrics) : 0u}};
  if (outputs_alias_inputs_or_each_other(inputs, 8u, outputs, 3u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (active == 0u)
    return ET_PPO_ERR_EMPTY_MASK;
  if (!validate_actions(bt, cfg->vocab_size, actions))
    return ET_PPO_ERR_INDEX;
  if (!all_finite_f32(policy_logits, btv) || !all_finite_f32(values, bt) ||
      !all_finite_f32(old_logprobs, bt) || !all_finite_f32(old_values, bt) ||
      !all_finite_f32(advantages, bt) || !all_finite_f32(returns, bt))
    return ET_PPO_ERR_NONFINITE;

  /* Pass 1: evaluate all active tokens, accumulate metrics, verify finiteness.
   */
  const double m_d = (double)active;
  const double vf_d = (double)cfg->vf_coef;
  const double ent_d = (double)cfg->ent_coef;
  double sum_pg = 0.0;
  double sum_vl = 0.0;
  double sum_ent = 0.0;
  double sum_kl = 0.0;
  uint32_t clipped_tokens = 0u;
  uint32_t v_clipped_tokens = 0u;

  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] == 0u)
      continue;
    const float *row = policy_logits + i * (size_t)cfg->vocab_size;
    token_ppo_eval ev;
    st = eval_ppo_token(cfg, row, values[i], actions[i], old_logprobs[i],
                        old_values[i], advantages[i], returns[i], &ev);
    if (st != ET_PPO_OK)
      return st;
    sum_pg += ev.pg_obj;
    sum_vl += ev.v_loss;
    sum_ent += ev.entropy;
    sum_kl += ev.approx_kl;
    clipped_tokens += ev.clipped;
    v_clipped_tokens += ev.v_clipped;

    if (want_backward) {
      const double dv_out = (vf_d * ev.d_v) / m_d;
      if (!isfinite((float)dv_out))
        return ET_PPO_ERR_OVERFLOW;
      const double g_lp = -ev.d_pg_d_logpi / m_d;
      const double g_ent = ent_d / m_d;
      const uint32_t act = (uint32_t)actions[i];
      for (uint32_t v = 0u; v < cfg->vocab_size; ++v) {
        const double lp = (double)row[v] - ev.lse;
        const double p = exp(lp);
        const double g_z = g_lp * ((v == act ? 1.0 : 0.0) - p) +
                           g_ent * p * (lp + ev.entropy);
        if (!isfinite((float)g_z))
          return ET_PPO_ERR_OVERFLOW;
      }
    }
  }

  const double policy_loss = -sum_pg / m_d;
  const double value_loss = sum_vl / m_d;
  const double mean_entropy = sum_ent / m_d;
  const double approx_kl = sum_kl / m_d;
  const double total_loss =
      policy_loss + vf_d * value_loss - ent_d * mean_entropy;
  const double clip_fraction = (double)clipped_tokens / m_d;
  const double v_clip_fraction = (double)v_clipped_tokens / m_d;
  const double explained_var =
      compute_explained_variance(bt, active, values, returns, mask);

  if (!isfinite((float)total_loss) || !isfinite((float)policy_loss) ||
      !isfinite((float)value_loss) || !isfinite((float)mean_entropy) ||
      !isfinite((float)approx_kl) || !isfinite((float)explained_var))
    return ET_PPO_ERR_OVERFLOW;

  /* Pass 2: commit outputs atomically. */
  if (want_backward) {
    for (size_t i = 0u; i < bt; ++i) {
      float *drow = out_d_policy_logits + i * (size_t)cfg->vocab_size;
      if (mask[i] == 0u) {
        out_d_values[i] = 0.0f;
        for (uint32_t v = 0u; v < cfg->vocab_size; ++v)
          drow[v] = 0.0f;
        continue;
      }
      const float *row = policy_logits + i * (size_t)cfg->vocab_size;
      token_ppo_eval ev;
      (void)eval_ppo_token(cfg, row, values[i], actions[i], old_logprobs[i],
                           old_values[i], advantages[i], returns[i], &ev);
      out_d_values[i] = (float)((vf_d * ev.d_v) / m_d);
      const double g_lp = -ev.d_pg_d_logpi / m_d;
      const double g_ent = ent_d / m_d;
      const uint32_t act = (uint32_t)actions[i];
      for (uint32_t v = 0u; v < cfg->vocab_size; ++v) {
        const double lp = (double)row[v] - ev.lse;
        const double p = exp(lp);
        const double g_z = g_lp * ((v == act ? 1.0 : 0.0) - p) +
                           g_ent * p * (lp + ev.entropy);
        drow[v] = (float)g_z;
      }
    }
  }

  if (out_metrics != NULL) {
    out_metrics->total_loss = (float)total_loss;
    out_metrics->policy_loss = (float)policy_loss;
    out_metrics->value_loss = (float)value_loss;
    out_metrics->mean_entropy = (float)mean_entropy;
    out_metrics->approx_kl = (float)approx_kl;
    out_metrics->clip_fraction = (float)clip_fraction;
    out_metrics->value_clip_fraction = (float)v_clip_fraction;
    out_metrics->explained_variance = (float)explained_var;
    out_metrics->active_tokens = active;
    out_metrics->clipped_tokens = clipped_tokens;
    out_metrics->value_clipped_tokens = v_clipped_tokens;
    out_metrics->reserved = 0u;
  }
  return ET_PPO_OK;
}

et_ppo_status et_ppo_loss_forward_v1(
    const et_ppo_loss_config *cfg, const float *policy_logits,
    const float *values, const int32_t *actions, const float *old_logprobs,
    const float *old_values, const float *advantages, const float *returns,
    const uint8_t *mask, et_ppo_loss_metrics *out_metrics) {
  return ppo_loss_internal(cfg, policy_logits, values, actions, old_logprobs,
                           old_values, advantages, returns, mask, NULL, NULL,
                           out_metrics, 0);
}

et_ppo_status et_ppo_loss_backward_v1(
    const et_ppo_loss_config *cfg, const float *policy_logits,
    const float *values, const int32_t *actions, const float *old_logprobs,
    const float *old_values, const float *advantages, const float *returns,
    const uint8_t *mask, float *out_d_policy_logits, float *out_d_values,
    et_ppo_loss_metrics *out_metrics) {
  return ppo_loss_internal(cfg, policy_logits, values, actions, old_logprobs,
                           old_values, advantages, returns, mask,
                           out_d_policy_logits, out_d_values, out_metrics, 1);
}

/* ---- Linear Actor + Value Critic Forward & Frozen Ref Logprobs ---- */

static int linear_actor_token_logits(uint32_t feat_dim, uint32_t vocab_size,
                                     const float *feat_row,
                                     const float *actor_params,
                                     float *out_row_logits) {
  const float *w_pi = actor_params;
  const float *b_pi = actor_params + (size_t)vocab_size * (size_t)feat_dim;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    const float *wv = w_pi + (size_t)v * (size_t)feat_dim;
    double acc = (double)b_pi[v];
    for (uint32_t d = 0u; d < feat_dim; ++d)
      acc += (double)wv[d] * (double)feat_row[d];
    if (!isfinite((float)acc))
      return 0;
    out_row_logits[v] = (float)acc;
  }
  return 1;
}

static int linear_critic_token_value(uint32_t feat_dim, uint32_t vocab_size,
                                     const float *feat_row,
                                     const float *params, float *out_value) {
  const size_t critic_off =
      (size_t)vocab_size * (size_t)feat_dim + (size_t)vocab_size;
  const float *w_v = params + critic_off;
  const float b_v = params[critic_off + (size_t)feat_dim];
  double acc = (double)b_v;
  for (uint32_t d = 0u; d < feat_dim; ++d)
    acc += (double)w_v[d] * (double)feat_row[d];
  if (!isfinite((float)acc))
    return 0;
  *out_value = (float)acc;
  return 1;
}

et_ppo_status et_ppo_actor_critic_forward_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t feat_dim,
    uint32_t vocab_size, const float *features, const float *params,
    float *out_logits, float *out_values) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u, pcount = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if ((st = et_ppo_model_param_count(feat_dim, vocab_size, &pcount)) !=
      ET_PPO_OK)
    return st;
  if (bt > ET_PPO_TRAINER_MAX_BT)
    return ET_PPO_ERR_SHAPE;

  const size_t xbytes = bt * (size_t)feat_dim * sizeof(float);
  const size_t pbytes = pcount * sizeof(float);
  const size_t lbytes = bt * (size_t)vocab_size * sizeof(float);
  const size_t vbytes = bt * sizeof(float);
  if ((st = check_buf(features, alignof(float), xbytes)) != ET_PPO_OK ||
      (st = check_buf(params, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(out_logits, alignof(float), lbytes)) != ET_PPO_OK ||
      (st = check_buf(out_values, alignof(float), vbytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[2] = {{features, xbytes}, {params, pbytes}};
  const span_entry outputs[2] = {{out_logits, lbytes}, {out_values, vbytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 2u, outputs, 2u))
    return ET_PPO_ERR_ALIAS;
  if (!all_finite_f32(features, bt * (size_t)feat_dim) ||
      !all_finite_f32(params, pcount))
    return ET_PPO_ERR_NONFINITE;

  /* Pass 1: validate finiteness. */
  float tmp_logits[ET_PPO_MAX_MODEL_VOCAB];
  for (size_t i = 0u; i < bt; ++i) {
    const float *frow = features + i * (size_t)feat_dim;
    float val = 0.0f;
    if (!linear_actor_token_logits(feat_dim, vocab_size, frow, params,
                                   tmp_logits) ||
        !linear_critic_token_value(feat_dim, vocab_size, frow, params, &val))
      return ET_PPO_ERR_OVERFLOW;
  }
  /* Pass 2: commit. */
  for (size_t i = 0u; i < bt; ++i) {
    const float *frow = features + i * (size_t)feat_dim;
    (void)linear_actor_token_logits(feat_dim, vocab_size, frow, params,
                                    out_logits + i * (size_t)vocab_size);
    (void)linear_critic_token_value(feat_dim, vocab_size, frow, params,
                                    &out_values[i]);
  }
  return ET_PPO_OK;
}

et_ppo_status et_ppo_ref_policy_logprobs_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t feat_dim,
    uint32_t vocab_size, const float *features, const float *ref_params,
    const int32_t *actions, const uint8_t *mask, float *out_ref_logprobs) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  size_t bt = 0u, rcount = 0u;
  et_ppo_status st = validate_bt_dims(batch_size, seq_len, &bt);
  if (st != ET_PPO_OK)
    return st;
  if ((st = et_ppo_ref_param_count(feat_dim, vocab_size, &rcount)) != ET_PPO_OK)
    return st;
  if (bt > ET_PPO_TRAINER_MAX_BT)
    return ET_PPO_ERR_SHAPE;

  const size_t xbytes = bt * (size_t)feat_dim * sizeof(float);
  const size_t rbytes = rcount * sizeof(float);
  const size_t abytes = bt * sizeof(int32_t);
  const size_t obytes = bt * sizeof(float);
  if ((st = check_buf(features, alignof(float), xbytes)) != ET_PPO_OK ||
      (st = check_buf(ref_params, alignof(float), rbytes)) != ET_PPO_OK ||
      (st = check_buf(actions, alignof(int32_t), abytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(out_ref_logprobs, alignof(float), obytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[4] = {{features, xbytes},
                                {ref_params, rbytes},
                                {actions, abytes},
                                {mask, bt}};
  const span_entry outputs[1] = {{out_ref_logprobs, obytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 4u, outputs, 1u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (!validate_actions(bt, vocab_size, actions))
    return ET_PPO_ERR_INDEX;
  if (!all_finite_f32(features, bt * (size_t)feat_dim) ||
      !all_finite_f32(ref_params, rcount))
    return ET_PPO_ERR_NONFINITE;

  float tmp_logits[ET_PPO_MAX_MODEL_VOCAB];
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      double lse = 0.0, lp = 0.0;
      if (!linear_actor_token_logits(feat_dim, vocab_size,
                                     features + i * (size_t)feat_dim,
                                     ref_params, tmp_logits) ||
          !row_lse_and_entropy(tmp_logits, vocab_size, actions[i], &lse, &lp,
                               NULL) ||
          !isfinite((float)lp))
        return ET_PPO_ERR_OVERFLOW;
    }
  }
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] == 0u) {
      out_ref_logprobs[i] = 0.0f;
    } else {
      double lse = 0.0, lp = 0.0;
      (void)linear_actor_token_logits(feat_dim, vocab_size,
                                      features + i * (size_t)feat_dim,
                                      ref_params, tmp_logits);
      (void)row_lse_and_entropy(tmp_logits, vocab_size, actions[i], &lse, &lp,
                                NULL);
      out_ref_logprobs[i] = (float)lp;
    }
  }
  return ET_PPO_OK;
}

/* ---- Decoupled AdamW Optimizer Step ---- */

et_ppo_status et_ppo_adamw_step_v1(const et_ppo_adamw_config *cfg,
                                   size_t param_count, const float *grads,
                                   float *io_params, float *io_m, float *io_v,
                                   uint64_t *io_step) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  et_ppo_status st = validate_adamw_config_internal(cfg);
  if (st != ET_PPO_OK)
    return st;
  if (param_count == 0u || param_count > PPO_MAX_PARAMS)
    return ET_PPO_ERR_SHAPE;

  const size_t pbytes = param_count * sizeof(float);
  if ((st = check_buf(grads, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_params, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_m, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_v, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_step, alignof(uint64_t), sizeof(uint64_t))) !=
          ET_PPO_OK)
    return st;

  const span_entry inputs[1] = {{grads, pbytes}};
  const span_entry outputs[4] = {{io_params, pbytes},
                                 {io_m, pbytes},
                                 {io_v, pbytes},
                                 {io_step, sizeof(uint64_t)}};
  if (outputs_alias_inputs_or_each_other(inputs, 1u, outputs, 4u))
    return ET_PPO_ERR_ALIAS;

  if (*io_step == UINT64_MAX)
    return ET_PPO_ERR_OVERFLOW;
  if (!all_finite_f32(grads, param_count) ||
      !all_finite_f32(io_params, param_count) ||
      !all_finite_f32(io_m, param_count) ||
      !all_nonneg_finite_f32(io_v, param_count))
    return ET_PPO_ERR_NONFINITE;

  const uint64_t next_step = *io_step + 1u;
  const double lr = (double)cfg->lr;
  const double b1 = (double)cfg->beta1;
  const double b2 = (double)cfg->beta2;
  const double eps = (double)cfg->eps;
  const double wd = (double)cfg->weight_decay;
  const double bc1 = 1.0 - pow_u64_f64(b1, next_step);
  const double bc2 = 1.0 - pow_u64_f64(b2, next_step);
  if (!(bc1 > 0.0) || !(bc2 > 0.0) || !isfinite(bc1) || !isfinite(bc2))
    return ET_PPO_ERR_OVERFLOW;

  double grad_scale = 1.0;
  if (cfg->max_grad_norm > 0.0f) {
    double sq_norm = 0.0;
    for (size_t i = 0u; i < param_count; ++i) {
      const double g = (double)grads[i];
      sq_norm += g * g;
    }
    const double gnorm = sqrt(sq_norm);
    if (!isfinite(gnorm))
      return ET_PPO_ERR_OVERFLOW;
    if (gnorm > (double)cfg->max_grad_norm)
      grad_scale = (double)cfg->max_grad_norm / gnorm;
  }

  /* Pass 1: verify all updated parameters and moments remain finite. */
  for (size_t i = 0u; i < param_count; ++i) {
    const double g = (double)grads[i] * grad_scale;
    const double m = b1 * (double)io_m[i] + (1.0 - b1) * g;
    const double v = b2 * (double)io_v[i] + (1.0 - b2) * g * g;
    const double m_hat = m / bc1;
    const double v_hat = v / bc2;
    const double denom = sqrt(v_hat) + eps;
    const double p = (double)io_params[i] * (1.0 - lr * wd) - lr * m_hat / denom;
    if (!isfinite((float)m) || !isfinite((float)v) || !isfinite((float)p) ||
        (float)v < 0.0f)
      return ET_PPO_ERR_OVERFLOW;
  }

  /* Pass 2: commit atomically. */
  for (size_t i = 0u; i < param_count; ++i) {
    const double g = (double)grads[i] * grad_scale;
    const double m = b1 * (double)io_m[i] + (1.0 - b1) * g;
    const double v = b2 * (double)io_v[i] + (1.0 - b2) * g * g;
    const double m_hat = m / bc1;
    const double v_hat = v / bc2;
    const double denom = sqrt(v_hat) + eps;
    const double p = (double)io_params[i] * (1.0 - lr * wd) - lr * m_hat / denom;
    io_m[i] = (float)m;
    io_v[i] = (float)v;
    io_params[i] = (float)p;
  }
  *io_step = next_step;
  return ET_PPO_OK;
}

/* ---- Multi-Epoch RLHF PPO Actor-Critic Rollout Trainer ---- */

et_ppo_status et_ppo_train_rollout_v1(
    const et_ppo_trainer_config *cfg, const float *features,
    const int32_t *actions, const float *r_env, const uint8_t *mask,
    const float *ref_params, float *io_params, float *io_adam_m,
    float *io_adam_v, uint64_t *io_step, et_ppo_rollout_report *out_report) {
  et_ppo_status st = et_ppo_validate_trainer_config(cfg);
  if (st != ET_PPO_OK)
    return st;

  const uint32_t B = cfg->loss_cfg.batch_size;
  const uint32_t T = cfg->loss_cfg.seq_len;
  const uint32_t V = cfg->loss_cfg.vocab_size;
  const uint32_t D = cfg->feat_dim;
  const size_t bt = (size_t)B * (size_t)T;
  size_t pcount = 0u, rcount = 0u;
  (void)et_ppo_model_param_count(D, V, &pcount);
  (void)et_ppo_ref_param_count(D, V, &rcount);

  const size_t xbytes = bt * (size_t)D * sizeof(float);
  const size_t abytes = bt * sizeof(int32_t);
  const size_t fbytes = bt * sizeof(float);
  const size_t rbytes = rcount * sizeof(float);
  const size_t pbytes = pcount * sizeof(float);

  if ((st = check_buf(features, alignof(float), xbytes)) != ET_PPO_OK ||
      (st = check_buf(actions, alignof(int32_t), abytes)) != ET_PPO_OK ||
      (st = check_buf(r_env, alignof(float), fbytes)) != ET_PPO_OK ||
      (st = check_buf(mask, 1u, bt)) != ET_PPO_OK ||
      (st = check_buf(ref_params, alignof(float), rbytes)) != ET_PPO_OK ||
      (st = check_buf(io_params, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_adam_m, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_adam_v, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(io_step, alignof(uint64_t), sizeof(uint64_t))) !=
          ET_PPO_OK ||
      (st = check_buf(out_report, alignof(et_ppo_rollout_report),
                      sizeof(*out_report))) != ET_PPO_OK)
    return st;

  const span_entry inputs[5] = {{features, xbytes},
                                {actions, abytes},
                                {r_env, fbytes},
                                {mask, bt},
                                {ref_params, rbytes}};
  const span_entry outputs[5] = {{io_params, pbytes},
                                 {io_adam_m, pbytes},
                                 {io_adam_v, pbytes},
                                 {io_step, sizeof(uint64_t)},
                                 {out_report, sizeof(*out_report)}};
  if (outputs_alias_inputs_or_each_other(inputs, 5u, outputs, 5u))
    return ET_PPO_ERR_ALIAS;

  uint32_t active = 0u;
  if (!validate_mask_count(bt, mask, &active))
    return ET_PPO_ERR_ARGUMENT;
  if (active == 0u)
    return ET_PPO_ERR_EMPTY_MASK;
  if (!validate_actions(bt, V, actions))
    return ET_PPO_ERR_INDEX;
  if (!all_finite_f32(features, bt * (size_t)D) || !all_finite_f32(r_env, bt) ||
      !all_finite_f32(ref_params, rcount) || !all_finite_f32(io_params, pcount) ||
      !all_finite_f32(io_adam_m, pcount) ||
      !all_nonneg_finite_f32(io_adam_v, pcount))
    return ET_PPO_ERR_NONFINITE;
  if (*io_step > UINT64_MAX - (uint64_t)cfg->ppo_epochs)
    return ET_PPO_ERR_OVERFLOW;

  /* Bounded stack workspace: zero heap allocations, failure-atomic staging. */
  float old_logprobs[ET_PPO_TRAINER_MAX_BT];
  float ref_logprobs[ET_PPO_TRAINER_MAX_BT];
  float old_values[ET_PPO_TRAINER_MAX_BT];
  float cur_values[ET_PPO_TRAINER_MAX_BT];
  float r_shaped[ET_PPO_TRAINER_MAX_BT];
  float adv_raw[ET_PPO_TRAINER_MAX_BT];
  float adv_use[ET_PPO_TRAINER_MAX_BT];
  float returns[ET_PPO_TRAINER_MAX_BT];
  float work_params[PPO_MAX_PARAMS];
  float work_m[PPO_MAX_PARAMS];
  float work_v[PPO_MAX_PARAMS];
  float grads[PPO_MAX_PARAMS];
  double grad_acc[PPO_MAX_PARAMS];

  /* Snapshot old policy logprobs & critic values and frozen reference logprobs.
   */
  float row_logits[ET_PPO_MAX_MODEL_VOCAB];
  for (size_t i = 0u; i < bt; ++i) {
    const float *frow = features + i * (size_t)D;
    if (!linear_critic_token_value(D, V, frow, io_params, &old_values[i]))
      return ET_PPO_ERR_OVERFLOW;
    if (mask[i] == 0u) {
      old_logprobs[i] = 0.0f;
    } else {
      double lse = 0.0, lp = 0.0;
      if (!linear_actor_token_logits(D, V, frow, io_params, row_logits) ||
          !row_lse_and_entropy(row_logits, V, actions[i], &lse, &lp, NULL) ||
          !isfinite((float)lp))
        return ET_PPO_ERR_OVERFLOW;
      old_logprobs[i] = (float)lp;
    }
  }
  st = et_ppo_ref_policy_logprobs_v1(B, T, D, V, features, ref_params, actions,
                                     mask, ref_logprobs);
  if (st != ET_PPO_OK)
    return st;

  st = et_ppo_kl_shaped_rewards_v1(B, T, cfg->beta_kl, r_env, old_logprobs,
                                   ref_logprobs, mask, r_shaped);
  if (st != ET_PPO_OK)
    return st;

  st = et_ppo_gae_v1(B, T, cfg->gamma, cfg->gae_lambda, r_shaped, old_values,
                     mask, adv_raw, returns);
  if (st != ET_PPO_OK)
    return st;

  if (cfg->whiten_advantages == 1u) {
    st = et_ppo_whiten_advantages_v1(B, T, cfg->adv_eps, adv_raw, mask, adv_use);
    if (st != ET_PPO_OK)
      return st;
  } else {
    memcpy(adv_use, adv_raw, bt * sizeof(float));
  }

  double sum_env_r = 0.0;
  double sum_shaped_r = 0.0;
  for (size_t i = 0u; i < bt; ++i) {
    if (mask[i] != 0u) {
      sum_env_r += (double)r_env[i];
      sum_shaped_r += (double)r_shaped[i];
    }
  }

  memcpy(work_params, io_params, pbytes);
  memcpy(work_m, io_adam_m, pbytes);
  memcpy(work_v, io_adam_v, pbytes);
  uint64_t work_step = *io_step;

  et_ppo_rollout_report report;
  memset(&report, 0, sizeof(report));
  report.mean_env_reward = (float)(sum_env_r / (double)active);
  report.mean_shaped_reward = (float)(sum_shaped_r / (double)active);

  const double m_d = (double)active;
  const double vf_d = (double)cfg->loss_cfg.vf_coef;
  const double ent_d = (double)cfg->loss_cfg.ent_coef;
  const size_t b_pi_off = (size_t)V * (size_t)D;
  const size_t w_v_off = b_pi_off + (size_t)V;
  const size_t b_v_off = w_v_off + (size_t)D;

  for (uint32_t epoch = 0u; epoch < cfg->ppo_epochs; ++epoch) {
    double sum_pg = 0.0;
    double sum_vl = 0.0;
    double sum_ent = 0.0;
    double sum_kl = 0.0;
    uint32_t clipped_tokens = 0u;
    uint32_t v_clipped_tokens = 0u;

    for (size_t p = 0u; p < pcount; ++p)
      grad_acc[p] = 0.0;

    for (size_t i = 0u; i < bt; ++i) {
      const float *frow = features + i * (size_t)D;
      if (!linear_critic_token_value(D, V, frow, work_params, &cur_values[i]))
        return ET_PPO_ERR_OVERFLOW;
      if (mask[i] == 0u)
        continue;
      if (!linear_actor_token_logits(D, V, frow, work_params, row_logits))
        return ET_PPO_ERR_OVERFLOW;

      token_ppo_eval ev;
      st = eval_ppo_token(&cfg->loss_cfg, row_logits, cur_values[i], actions[i],
                          old_logprobs[i], old_values[i], adv_use[i],
                          returns[i], &ev);
      if (st != ET_PPO_OK)
        return st;

      sum_pg += ev.pg_obj;
      sum_vl += ev.v_loss;
      sum_ent += ev.entropy;
      sum_kl += ev.approx_kl;
      clipped_tokens += ev.clipped;
      v_clipped_tokens += ev.v_clipped;

      const float d_val_f = (float)((vf_d * ev.d_v) / m_d);
      if (!isfinite(d_val_f))
        return ET_PPO_ERR_OVERFLOW;
      const double d_val = (double)d_val_f;
      grad_acc[b_v_off] += d_val;
      for (uint32_t d = 0u; d < D; ++d)
        grad_acc[w_v_off + (size_t)d] += d_val * (double)frow[d];

      const double g_lp = -ev.d_pg_d_logpi / m_d;
      const double g_ent = ent_d / m_d;
      const uint32_t act = (uint32_t)actions[i];
      for (uint32_t v = 0u; v < V; ++v) {
        const double lp = (double)row_logits[v] - ev.lse;
        const double p = exp(lp);
        const float gz_f = (float)(g_lp * ((v == act ? 1.0 : 0.0) - p) +
                                   g_ent * p * (lp + ev.entropy));
        if (!isfinite(gz_f))
          return ET_PPO_ERR_OVERFLOW;
        const double gz = (double)gz_f;
        grad_acc[b_pi_off + (size_t)v] += gz;
        double *gw = grad_acc + (size_t)v * (size_t)D;
        for (uint32_t d = 0u; d < D; ++d)
          gw[d] += gz * (double)frow[d];
      }
    }

    const double policy_loss = -sum_pg / m_d;
    const double value_loss = sum_vl / m_d;
    const double mean_entropy = sum_ent / m_d;
    const double approx_kl = sum_kl / m_d;
    const double total_loss =
        policy_loss + vf_d * value_loss - ent_d * mean_entropy;
    const double explained_var =
        compute_explained_variance(bt, active, cur_values, returns, mask);
    if (!isfinite((float)total_loss) || !isfinite((float)policy_loss) ||
        !isfinite((float)value_loss) || !isfinite((float)mean_entropy) ||
        !isfinite((float)approx_kl) || !isfinite((float)explained_var))
      return ET_PPO_ERR_OVERFLOW;

    et_ppo_loss_metrics metrics;
    metrics.total_loss = (float)total_loss;
    metrics.policy_loss = (float)policy_loss;
    metrics.value_loss = (float)value_loss;
    metrics.mean_entropy = (float)mean_entropy;
    metrics.approx_kl = (float)approx_kl;
    metrics.clip_fraction = (float)((double)clipped_tokens / m_d);
    metrics.value_clip_fraction = (float)((double)v_clipped_tokens / m_d);
    metrics.explained_variance = (float)explained_var;
    metrics.active_tokens = active;
    metrics.clipped_tokens = clipped_tokens;
    metrics.value_clipped_tokens = v_clipped_tokens;
    metrics.reserved = 0u;

    /* KL early stopping: abort epoch before stepping if approx_kl > target_kl.
     */
    if (cfg->target_kl > 0.0f && metrics.approx_kl > cfg->target_kl) {
      report.early_stopped = 1u;
      report.stopped_epoch = epoch;
      report.final_metrics = metrics;
      break;
    }

    for (size_t p = 0u; p < pcount; ++p) {
      const float gf = (float)grad_acc[p];
      if (!isfinite(gf))
        return ET_PPO_ERR_OVERFLOW;
      grads[p] = gf;
    }

    st = et_ppo_adamw_step_v1(&cfg->optim_cfg, pcount, grads, work_params,
                              work_m, work_v, &work_step);
    if (st != ET_PPO_OK)
      return st;

    report.epochs_completed = epoch + 1u;
    report.final_metrics = metrics;
  }

  /* Atomic commit after all epochs succeed. */
  memcpy(io_params, work_params, pbytes);
  memcpy(io_adam_m, work_m, pbytes);
  memcpy(io_adam_v, work_v, pbytes);
  *io_step = work_step;
  *out_report = report;
  return ET_PPO_OK;
}

/* ---- Versioned SHA-256 Checksummed ETPP1000 v1.0 Checkpoint ---- */

static const uint8_t PPO_CKPT_MAGIC[8] = {'E', 'T', 'P', 'P',
                                          '1', '0', '0', '0'};

static int write_all(int fd, const uint8_t *bytes, size_t size) {
  while (size != 0u) {
    const ssize_t n = write(fd, bytes, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      return 0;
    bytes += (size_t)n;
    size -= (size_t)n;
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

static void sync_parent_dir(const char *path) {
  char dir_buf[PPO_CKPT_MAX_PATH + 16u];
  const char *slash = strrchr(path, '/');
  const char *dir = ".";
  if (slash != NULL) {
    const size_t len = slash == path ? 1u : (size_t)(slash - path);
    if (len < sizeof(dir_buf)) {
      memcpy(dir_buf, path, len);
      dir_buf[len] = '\0';
      dir = dir_buf;
    }
  }
  const int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd >= 0) {
    (void)sync_fd(dfd);
    (void)close(dfd);
  }
}

static et_ppo_status serialize_checkpoint(
    const et_ppo_trainer_config *cfg, uint64_t step, const float *ref_params,
    const float *params, const float *adam_m, const float *adam_v,
    uint8_t *buf, size_t cap, size_t *out_size) {
  et_ppo_status st = et_ppo_validate_trainer_config(cfg);
  if (st != ET_PPO_OK)
    return st;

  const uint32_t D = cfg->feat_dim;
  const uint32_t V = cfg->loss_cfg.vocab_size;
  size_t pcount = 0u, rcount = 0u;
  (void)et_ppo_model_param_count(D, V, &pcount);
  (void)et_ppo_ref_param_count(D, V, &rcount);

  const size_t rbytes = rcount * sizeof(float);
  const size_t pbytes = pcount * sizeof(float);
  if ((st = check_buf(ref_params, alignof(float), rbytes)) != ET_PPO_OK ||
      (st = check_buf(params, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(adam_m, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(adam_v, alignof(float), pbytes)) != ET_PPO_OK)
    return st;

  if (!all_finite_f32(ref_params, rcount) || !all_finite_f32(params, pcount) ||
      !all_finite_f32(adam_m, pcount) ||
      !all_nonneg_finite_f32(adam_v, pcount))
    return ET_PPO_ERR_NONFINITE;

  uint8_t ref_sha[ET_PPO_SHA256_BYTES];
  st = et_ppo_weight_sha256_v1(rcount, ref_params, ref_sha);
  if (st != ET_PPO_OK)
    return st;

  const uint32_t payload_bytes = (uint32_t)(3u * pbytes);
  const size_t total =
      PPO_CKPT_HEADER_BYTES + (size_t)payload_bytes + PPO_CKPT_TRAILER_BYTES;
  if (total > cap)
    return ET_PPO_ERR_OVERFLOW;

  size_t off = 0u;
  memcpy(buf + off, PPO_CKPT_MAGIC, 8u);
  off += 8u;
  encode_u32_le(buf + off, ET_PPO_ABI_MAJOR);
  off += 4u;
  encode_u32_le(buf + off, ET_PPO_ABI_MINOR);
  off += 4u;
  encode_u32_le(buf + off, D);
  off += 4u;
  encode_u32_le(buf + off, V);
  off += 4u;
  encode_u32_le(buf + off, (uint32_t)pcount);
  off += 4u;
  encode_u32_le(buf + off, 0u);
  off += 4u;
  encode_u64_le(buf + off, step);
  off += 8u;

  /* loss_cfg (40 bytes) */
  encode_u32_le(buf + off, cfg->loss_cfg.batch_size);
  off += 4u;
  encode_u32_le(buf + off, cfg->loss_cfg.seq_len);
  off += 4u;
  encode_u32_le(buf + off, cfg->loss_cfg.vocab_size);
  off += 4u;
  encode_u32_le(buf + off, cfg->loss_cfg.use_value_clip);
  off += 4u;
  encode_f32_le(buf + off, cfg->loss_cfg.clip_eps_low);
  off += 4u;
  encode_f32_le(buf + off, cfg->loss_cfg.clip_eps_high);
  off += 4u;
  encode_f32_le(buf + off, cfg->loss_cfg.value_clip_eps);
  off += 4u;
  encode_f32_le(buf + off, cfg->loss_cfg.vf_coef);
  off += 4u;
  encode_f32_le(buf + off, cfg->loss_cfg.ent_coef);
  off += 4u;
  encode_u32_le(buf + off, cfg->loss_cfg.reserved);
  off += 4u;

  /* optim_cfg (32 bytes) */
  encode_f32_le(buf + off, cfg->optim_cfg.lr);
  off += 4u;
  encode_f32_le(buf + off, cfg->optim_cfg.beta1);
  off += 4u;
  encode_f32_le(buf + off, cfg->optim_cfg.beta2);
  off += 4u;
  encode_f32_le(buf + off, cfg->optim_cfg.eps);
  off += 4u;
  encode_f32_le(buf + off, cfg->optim_cfg.weight_decay);
  off += 4u;
  encode_f32_le(buf + off, cfg->optim_cfg.max_grad_norm);
  off += 4u;
  encode_u32_le(buf + off, cfg->optim_cfg.reserved[0]);
  off += 4u;
  encode_u32_le(buf + off, cfg->optim_cfg.reserved[1]);
  off += 4u;

  /* trainer hyperparams (36 bytes) */
  encode_u32_le(buf + off, cfg->ppo_epochs);
  off += 4u;
  encode_u32_le(buf + off, cfg->whiten_advantages);
  off += 4u;
  encode_f32_le(buf + off, cfg->gamma);
  off += 4u;
  encode_f32_le(buf + off, cfg->gae_lambda);
  off += 4u;
  encode_f32_le(buf + off, cfg->beta_kl);
  off += 4u;
  encode_f32_le(buf + off, cfg->adv_eps);
  off += 4u;
  encode_f32_le(buf + off, cfg->target_kl);
  off += 4u;
  encode_u32_le(buf + off, cfg->reserved);
  off += 4u;
  encode_u32_le(buf + off, cfg->reserved2);
  off += 4u;

  memcpy(buf + off, ref_sha, ET_PPO_SHA256_BYTES);
  off += ET_PPO_SHA256_BYTES;
  encode_u32_le(buf + off, payload_bytes);
  off += 4u;

  /* Exact header offset invariant: off == 184. */
  if (off != PPO_CKPT_HEADER_BYTES)
    return ET_PPO_ERR_FORMAT;

  for (size_t i = 0u; i < pcount; ++i) {
    encode_f32_le(buf + off, params[i]);
    off += 4u;
  }
  for (size_t i = 0u; i < pcount; ++i) {
    encode_f32_le(buf + off, adam_m[i]);
    off += 4u;
  }
  for (size_t i = 0u; i < pcount; ++i) {
    encode_f32_le(buf + off, adam_v[i]);
    off += 4u;
  }

  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, buf, off);
  sha256_final(&sha, buf + off);
  off += PPO_CKPT_TRAILER_BYTES;

  *out_size = off;
  return ET_PPO_OK;
}

et_ppo_status et_ppo_checkpoint_save_v1(const char *path,
                                        const et_ppo_trainer_config *cfg,
                                        uint64_t step, const float *ref_params,
                                        const float *params,
                                        const float *adam_m,
                                        const float *adam_v) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0')
    return ET_PPO_ERR_ARGUMENT;
  const size_t path_len = strlen(path);
  if (path_len > PPO_CKPT_MAX_PATH)
    return ET_PPO_ERR_ARGUMENT;
  const char *slash = strrchr(path, '/');
  if (slash != NULL && slash[1] == '\0')
    return ET_PPO_ERR_ARGUMENT;

  uint8_t buffer[ET_PPO_CHECKPOINT_MAX_BYTES];
  size_t size = 0u;
  et_ppo_status st = serialize_checkpoint(cfg, step, ref_params, params, adam_m,
                                          adam_v, buffer, sizeof(buffer), &size);
  if (st != ET_PPO_OK)
    return st;

  const size_t prefix = slash == NULL ? 0u : (size_t)(slash - path) + 1u;
  char temp_path[PPO_CKPT_MAX_PATH + 64u];
  int fd = -1;
  const unsigned long pid = (unsigned long)getpid();
  for (unsigned attempt = 0u; attempt < 256u; ++attempt) {
    memcpy(temp_path, path, prefix);
    const int n = snprintf(temp_path + prefix, sizeof(temp_path) - prefix,
                           ".etpp1000.%lu.%u.tmp", pid, attempt);
    if (n <= 0 || (size_t)n >= sizeof(temp_path) - prefix)
      return ET_PPO_ERR_OVERFLOW;
    fd = open(temp_path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
              0600);
    if (fd >= 0)
      break;
    if (errno != EEXIST)
      return ET_PPO_ERR_IO;
  }
  if (fd < 0)
    return ET_PPO_ERR_IO;

  int ok = (fchmod(fd, 0600) == 0) && write_all(fd, buffer, size) &&
           sync_fd(fd);
  if (close(fd) != 0)
    ok = 0;
  if (!ok || rename(temp_path, path) != 0) {
    (void)unlink(temp_path);
    return ET_PPO_ERR_IO;
  }
  sync_parent_dir(path);
  return ET_PPO_OK;
}

static et_ppo_status parse_checkpoint(
    const uint8_t *buf, size_t size, uint32_t expected_feat_dim,
    uint32_t expected_vocab_size, const float *ref_params,
    et_ppo_checkpoint_meta *out_meta, float *out_params, float *out_adam_m,
    float *out_adam_v) {
  if (size < PPO_CKPT_HEADER_BYTES + PPO_CKPT_TRAILER_BYTES)
    return ET_PPO_ERR_TRUNCATED;

  if (memcmp(buf, PPO_CKPT_MAGIC, 8u) != 0)
    return ET_PPO_ERR_FORMAT;

  size_t off = 8u;
  const uint32_t ver_major = decode_u32_le(buf + off);
  off += 4u;
  const uint32_t ver_minor = decode_u32_le(buf + off);
  off += 4u;
  if (ver_major != ET_PPO_ABI_MAJOR || ver_minor != ET_PPO_ABI_MINOR)
    return ET_PPO_ERR_FORMAT;

  const uint32_t feat_dim = decode_u32_le(buf + off);
  off += 4u;
  const uint32_t vocab_size = decode_u32_le(buf + off);
  off += 4u;
  const uint32_t param_count = decode_u32_le(buf + off);
  off += 4u;
  const uint32_t flags = decode_u32_le(buf + off);
  off += 4u;
  const uint64_t step = decode_u64_le(buf + off);
  off += 8u;
  if (flags != 0u)
    return ET_PPO_ERR_FORMAT;

  et_ppo_trainer_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.loss_cfg.batch_size = decode_u32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.seq_len = decode_u32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.vocab_size = decode_u32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.use_value_clip = decode_u32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.clip_eps_low = decode_f32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.clip_eps_high = decode_f32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.value_clip_eps = decode_f32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.vf_coef = decode_f32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.ent_coef = decode_f32_le(buf + off);
  off += 4u;
  cfg.loss_cfg.reserved = decode_u32_le(buf + off);
  off += 4u;

  cfg.optim_cfg.lr = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.beta1 = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.beta2 = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.eps = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.weight_decay = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.max_grad_norm = decode_f32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.reserved[0] = decode_u32_le(buf + off);
  off += 4u;
  cfg.optim_cfg.reserved[1] = decode_u32_le(buf + off);
  off += 4u;

  cfg.feat_dim = feat_dim;
  cfg.ppo_epochs = decode_u32_le(buf + off);
  off += 4u;
  cfg.whiten_advantages = decode_u32_le(buf + off);
  off += 4u;
  cfg.gamma = decode_f32_le(buf + off);
  off += 4u;
  cfg.gae_lambda = decode_f32_le(buf + off);
  off += 4u;
  cfg.beta_kl = decode_f32_le(buf + off);
  off += 4u;
  cfg.adv_eps = decode_f32_le(buf + off);
  off += 4u;
  cfg.target_kl = decode_f32_le(buf + off);
  off += 4u;
  cfg.reserved = decode_u32_le(buf + off);
  off += 4u;
  cfg.reserved2 = decode_u32_le(buf + off);
  off += 4u;

  uint8_t stored_ref_sha[ET_PPO_SHA256_BYTES];
  memcpy(stored_ref_sha, buf + off, ET_PPO_SHA256_BYTES);
  off += ET_PPO_SHA256_BYTES;
  const uint32_t payload_bytes = decode_u32_le(buf + off);
  off += 4u;

  if (off != PPO_CKPT_HEADER_BYTES)
    return ET_PPO_ERR_FORMAT;

  size_t expected_pcount = 0u, expected_rcount = 0u;
  et_ppo_status st = et_ppo_model_param_count(
      expected_feat_dim, expected_vocab_size, &expected_pcount);
  if (st != ET_PPO_OK)
    return st;
  (void)et_ppo_ref_param_count(expected_feat_dim, expected_vocab_size,
                               &expected_rcount);

  /* Validate payload length against file size first so truncated files are
   * cleanly reported as ET_PPO_ERR_TRUNCATED. */
  const size_t expected_total_from_hdr =
      PPO_CKPT_HEADER_BYTES + (size_t)payload_bytes + PPO_CKPT_TRAILER_BYTES;
  if (size < expected_total_from_hdr)
    return ET_PPO_ERR_TRUNCATED;
  if (size != expected_total_from_hdr)
    return ET_PPO_ERR_FORMAT;

  /* Verify SHA-256 trailer over header + payload before trusting decoded data.
   */
  uint8_t computed_sha[ET_PPO_SHA256_BYTES];
  sha256_state sha;
  sha256_init(&sha);
  sha256_update(&sha, buf, size - PPO_CKPT_TRAILER_BYTES);
  sha256_final(&sha, computed_sha);
  if (!ct_equal_32(computed_sha, buf + (size - PPO_CKPT_TRAILER_BYTES)))
    return ET_PPO_ERR_CHECKSUM;

  if (feat_dim != expected_feat_dim || vocab_size != expected_vocab_size ||
      cfg.loss_cfg.vocab_size != expected_vocab_size ||
      (size_t)param_count != expected_pcount)
    return ET_PPO_ERR_SHAPE;
  if ((size_t)payload_bytes != 3u * expected_pcount * sizeof(float))
    return ET_PPO_ERR_FORMAT;

  st = et_ppo_validate_trainer_config(&cfg);
  if (st != ET_PPO_OK)
    return ET_PPO_ERR_FORMAT;

  uint8_t actual_ref_sha[ET_PPO_SHA256_BYTES];
  st = et_ppo_weight_sha256_v1(expected_rcount, ref_params, actual_ref_sha);
  if (st != ET_PPO_OK)
    return st;
  if (!ct_equal_32(actual_ref_sha, stored_ref_sha))
    return ET_PPO_ERR_REF_MISMATCH;

  /* Stage decoded payload and validate finiteness before mutating outputs. */
  float staged_p[PPO_MAX_PARAMS];
  float staged_m[PPO_MAX_PARAMS];
  float staged_v[PPO_MAX_PARAMS];
  for (size_t i = 0u; i < expected_pcount; ++i) {
    staged_p[i] = decode_f32_le(buf + off);
    off += 4u;
    if (!isfinite(staged_p[i]))
      return ET_PPO_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < expected_pcount; ++i) {
    staged_m[i] = decode_f32_le(buf + off);
    off += 4u;
    if (!isfinite(staged_m[i]))
      return ET_PPO_ERR_NONFINITE;
  }
  for (size_t i = 0u; i < expected_pcount; ++i) {
    staged_v[i] = decode_f32_le(buf + off);
    off += 4u;
    if (!isfinite(staged_v[i]) || staged_v[i] < 0.0f)
      return ET_PPO_ERR_NONFINITE;
  }

  /* Commit atomically. */
  memcpy(out_params, staged_p, expected_pcount * sizeof(float));
  memcpy(out_adam_m, staged_m, expected_pcount * sizeof(float));
  memcpy(out_adam_v, staged_v, expected_pcount * sizeof(float));
  out_meta->version_major = ver_major;
  out_meta->version_minor = ver_minor;
  out_meta->feat_dim = feat_dim;
  out_meta->vocab_size = vocab_size;
  out_meta->param_count = param_count;
  out_meta->reserved = 0u;
  out_meta->step = step;
  out_meta->cfg = cfg;
  memcpy(out_meta->ref_sha256, stored_ref_sha, ET_PPO_SHA256_BYTES);
  return ET_PPO_OK;
}

et_ppo_status et_ppo_checkpoint_load_v1(
    const char *path, uint32_t expected_feat_dim, uint32_t expected_vocab_size,
    const float *ref_params, et_ppo_checkpoint_meta *out_meta,
    float *out_params, float *out_adam_m, float *out_adam_v) {
  if (!fp_environment_supported())
    return ET_PPO_ERR_FP_ENV;
  if (path == NULL || path[0] == '\0')
    return ET_PPO_ERR_ARGUMENT;

  size_t pcount = 0u, rcount = 0u;
  et_ppo_status st = et_ppo_model_param_count(
      expected_feat_dim, expected_vocab_size, &pcount);
  if (st != ET_PPO_OK)
    return st;
  (void)et_ppo_ref_param_count(expected_feat_dim, expected_vocab_size, &rcount);

  const size_t rbytes = rcount * sizeof(float);
  const size_t pbytes = pcount * sizeof(float);
  if ((st = check_buf(ref_params, alignof(float), rbytes)) != ET_PPO_OK ||
      (st = check_buf(out_meta, alignof(et_ppo_checkpoint_meta),
                      sizeof(*out_meta))) != ET_PPO_OK ||
      (st = check_buf(out_params, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(out_adam_m, alignof(float), pbytes)) != ET_PPO_OK ||
      (st = check_buf(out_adam_v, alignof(float), pbytes)) != ET_PPO_OK)
    return st;

  const span_entry inputs[1] = {{ref_params, rbytes}};
  const span_entry outputs[4] = {{out_meta, sizeof(*out_meta)},
                                 {out_params, pbytes},
                                 {out_adam_m, pbytes},
                                 {out_adam_v, pbytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 1u, outputs, 4u))
    return ET_PPO_ERR_ALIAS;
  if (!all_finite_f32(ref_params, rcount))
    return ET_PPO_ERR_NONFINITE;

  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0)
    return errno == ELOOP ? ET_PPO_ERR_SYMLINK : ET_PPO_ERR_IO;

  struct stat meta;
  uint8_t buffer[ET_PPO_CHECKPOINT_MAX_BYTES + 1u];
  size_t used = 0u;
  st = ET_PPO_OK;
  if (fstat(fd, &meta) != 0 || !S_ISREG(meta.st_mode)) {
    st = ET_PPO_ERR_IO;
  } else if (meta.st_size < 0 ||
             (uintmax_t)meta.st_size > ET_PPO_CHECKPOINT_MAX_BYTES) {
    st = ET_PPO_ERR_FORMAT;
  } else {
    while (used < sizeof(buffer)) {
      const ssize_t n = read(fd, buffer + used, sizeof(buffer) - used);
      if (n == 0)
        break;
      if (n < 0) {
        if (errno == EINTR)
          continue;
        st = ET_PPO_ERR_IO;
        break;
      }
      used += (size_t)n;
    }
    if (st == ET_PPO_OK) {
      if (used > ET_PPO_CHECKPOINT_MAX_BYTES)
        st = ET_PPO_ERR_FORMAT;
      else if (used != (size_t)meta.st_size)
        st = ET_PPO_ERR_TRUNCATED;
    }
  }
  if (close(fd) != 0 && st == ET_PPO_OK)
    st = ET_PPO_ERR_IO;
  if (st != ET_PPO_OK)
    return st;

  return parse_checkpoint(buffer, used, expected_feat_dim, expected_vocab_size,
                          ref_params, out_meta, out_params, out_adam_m,
                          out_adam_v);
}
