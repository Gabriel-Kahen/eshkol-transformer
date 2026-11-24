#include "eshkol_transformer/dpo_alignment_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "DPO engine requires 8-bit bytes and IEEE-754 binary32/64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "DPO engine requires IEEE-754 binary32/64 evaluation");

/* ---- Standalone SHA-256 for canonical weight immutability proofs ---- */

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

static void encode_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
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

static et_dpo_status check_span(const void *ptr, size_t alignment,
                                size_t bytes) {
  if (ptr == NULL)
    return ET_DPO_ERR_ARGUMENT;
  const uintptr_t addr = (uintptr_t)ptr;
  if ((addr % alignment) != 0u)
    return ET_DPO_ERR_ALIGNMENT;
  if (addr > UINTPTR_MAX - bytes)
    return ET_DPO_ERR_OVERFLOW;
  return ET_DPO_OK;
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
  return left < right + b_bytes && right < left + a_bytes;
}

static int any_spans_overlap(const span_entry *spans, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    for (size_t j = i + 1u; j < count; ++j) {
      if (spans_overlap(spans[i].ptr, spans[i].bytes, spans[j].ptr,
                        spans[j].bytes))
        return 1;
    }
  }
  return 0;
}

static int valid_btv(uint32_t b, uint32_t t, uint32_t v) {
  return b >= ET_DPO_MIN_B && b <= ET_DPO_MAX_B && t >= ET_DPO_MIN_T &&
         t <= ET_DPO_MAX_T && v >= ET_DPO_MIN_V && v <= ET_DPO_MAX_V;
}

static et_dpo_status validate_config_raw(const void *cfg_raw,
                                         et_dpo_config *out_cfg) {
  if (!fp_environment_supported())
    return ET_DPO_ERR_FP_ENV;
  const et_dpo_status span_st =
      check_span(cfg_raw, alignof(et_dpo_config), sizeof(et_dpo_config));
  if (span_st != ET_DPO_OK)
    return span_st;
  et_dpo_config cfg;
  memcpy(&cfg, cfg_raw, sizeof(cfg));
  if (!valid_btv(cfg.batch_size, cfg.seq_len, cfg.vocab_size))
    return ET_DPO_ERR_SHAPE;
  if (cfg.reference_free > 1u)
    return ET_DPO_ERR_ARGUMENT;
  if (!isfinite(cfg.beta))
    return ET_DPO_ERR_NONFINITE;
  if (!(cfg.beta > 0.0))
    return ET_DPO_ERR_ARGUMENT;
  if (!isfinite(cfg.label_smoothing))
    return ET_DPO_ERR_NONFINITE;
  if (!(cfg.label_smoothing >= 0.0 && cfg.label_smoothing < 0.5))
    return ET_DPO_ERR_ARGUMENT;
  if (out_cfg != NULL)
    *out_cfg = cfg;
  return ET_DPO_OK;
}

static int all_f32_finite(const float *buf, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static et_dpo_status validate_targets_and_mask(uint32_t b_dim, uint32_t t_dim,
                                               uint32_t v_dim,
                                               const int64_t *targets,
                                               const uint8_t *mask) {
  for (uint32_t b = 0u; b < b_dim; ++b) {
    uint32_t active = 0u;
    for (uint32_t t = 0u; t < t_dim; ++t) {
      const size_t idx = (size_t)b * t_dim + t;
      const uint8_t m = mask[idx];
      if (m > 1u)
        return ET_DPO_ERR_MASK;
      active += (uint32_t)m;
      const int64_t tok = targets[idx];
      if (tok < 0 || tok >= (int64_t)v_dim)
        return ET_DPO_ERR_TARGET;
    }
    if (active == 0u)
      return ET_DPO_ERR_MASK;
  }
  return ET_DPO_OK;
}

/* ---- Numerically stable binary64 math primitives ---- */

static double stable_softplus(double x) {
  if (x > 0.0)
    return x + log1p(exp(-x));
  return log1p(exp(x));
}

static double stable_sigmoid(double x) {
  if (x >= 0.0) {
    const double ez = exp(-x);
    return 1.0 / (1.0 + ez);
  }
  const double ez = exp(x);
  return ez / (1.0 + ez);
}

/* Compute exact masked sequence log-probabilities in binary64 using stable
 * log-sum-exp:
 *   m = max_v logits[v]
 *   s = sum_v exp(logits[v] - m)
 *   tok_logp = (logits[target] - m) - log(s)
 */
static et_dpo_status compute_seq_logps_checked(uint32_t b_dim, uint32_t t_dim,
                                               uint32_t v_dim,
                                               const float *logits,
                                               const int64_t *targets,
                                               const uint8_t *mask,
                                               double out_logps[ET_DPO_MAX_B]) {
  for (uint32_t b = 0u; b < b_dim; ++b) {
    double seq_sum = 0.0;
    for (uint32_t t = 0u; t < t_dim; ++t) {
      const size_t bt = (size_t)b * t_dim + t;
      if (mask[bt] == 0u)
        continue;
      const float *row = logits + bt * v_dim;
      double max_val = (double)row[0];
      for (uint32_t v = 1u; v < v_dim; ++v) {
        const double z = (double)row[v];
        if (z > max_val)
          max_val = z;
      }
      double sum_exp = 0.0;
      for (uint32_t v = 0u; v < v_dim; ++v)
        sum_exp += exp((double)row[v] - max_val);
      const double log_sum_exp_shift = log(sum_exp);
      const uint32_t target = (uint32_t)targets[bt];
      const double tok_logp = ((double)row[target] - max_val) - log_sum_exp_shift;
      seq_sum += tok_logp;
      if (!isfinite(tok_logp) || !isfinite(seq_sum))
        return ET_DPO_ERR_NONFINITE;
    }
    out_logps[b] = seq_sum;
  }
  return ET_DPO_OK;
}

/* Preflight a logit-gradient tensor for finite binary32 representability
 * without mutating any caller storage. */
static et_dpo_status preflight_seq_grads(uint32_t b_dim, uint32_t t_dim,
                                         uint32_t v_dim, const float *logits,
                                         const int64_t *targets,
                                         const uint8_t *mask,
                                         const double seq_scale[ET_DPO_MAX_B]) {
  double probs[ET_DPO_MAX_V];
  for (uint32_t b = 0u; b < b_dim; ++b) {
    const double scale = seq_scale[b];
    if (!isfinite(scale))
      return ET_DPO_ERR_NONFINITE;
    for (uint32_t t = 0u; t < t_dim; ++t) {
      const size_t bt = (size_t)b * t_dim + t;
      if (mask[bt] == 0u)
        continue;
      const float *row = logits + bt * v_dim;
      double max_val = (double)row[0];
      for (uint32_t v = 1u; v < v_dim; ++v) {
        const double z = (double)row[v];
        if (z > max_val)
          max_val = z;
      }
      double sum_exp = 0.0;
      for (uint32_t v = 0u; v < v_dim; ++v) {
        probs[v] = exp((double)row[v] - max_val);
        sum_exp += probs[v];
      }
      if (!isfinite(sum_exp) || !(sum_exp > 0.0))
        return ET_DPO_ERR_NONFINITE;
      const uint32_t target = (uint32_t)targets[bt];
      for (uint32_t v = 0u; v < v_dim; ++v) {
        const double p = probs[v] / sum_exp;
        const double ind = (v == target) ? 1.0 : 0.0;
        const double grad_d = scale * (ind - p);
        if (!isfinite(grad_d) || fabs(grad_d) > (double)FLT_MAX)
          return ET_DPO_ERR_NONFINITE;
        const float grad_f = (float)grad_d;
        if (!isfinite(grad_f))
          return ET_DPO_ERR_NONFINITE;
      }
    }
  }
  return ET_DPO_OK;
}

/* Commit preflighted logit gradients into caller-owned storage.
 * Masked positions (mask[b, t] == 0) always receive exact +0.0f. */
static void commit_seq_grads(uint32_t b_dim, uint32_t t_dim, uint32_t v_dim,
                             const float *logits, const int64_t *targets,
                             const uint8_t *mask,
                             const double seq_scale[ET_DPO_MAX_B],
                             float *d_logits) {
  double probs[ET_DPO_MAX_V];
  for (uint32_t b = 0u; b < b_dim; ++b) {
    const double scale = seq_scale[b];
    for (uint32_t t = 0u; t < t_dim; ++t) {
      const size_t bt = (size_t)b * t_dim + t;
      float *dst = d_logits + bt * v_dim;
      if (mask[bt] == 0u) {
        for (uint32_t v = 0u; v < v_dim; ++v)
          dst[v] = 0.0f;
        continue;
      }
      const float *row = logits + bt * v_dim;
      double max_val = (double)row[0];
      for (uint32_t v = 1u; v < v_dim; ++v) {
        const double z = (double)row[v];
        if (z > max_val)
          max_val = z;
      }
      double sum_exp = 0.0;
      for (uint32_t v = 0u; v < v_dim; ++v) {
        probs[v] = exp((double)row[v] - max_val);
        sum_exp += probs[v];
      }
      const uint32_t target = (uint32_t)targets[bt];
      for (uint32_t v = 0u; v < v_dim; ++v) {
        const double p = probs[v] / sum_exp;
        const double ind = (v == target) ? 1.0 : 0.0;
        dst[v] = (float)(scale * (ind - p));
      }
    }
  }
}

/* Shared core for forward and fused forward+backward. */
static et_dpo_status run_dpo_raw(
    const void *cfg_raw, const void *pi_w_raw, const void *pi_l_raw,
    const void *ref_w_raw, const void *ref_l_raw, const void *tgt_w_raw,
    const void *tgt_l_raw, const void *mask_w_raw, const void *mask_l_raw,
    void *telemetry_raw, int with_backward, void *d_w_raw, void *d_l_raw) {
  const int saved_errno = errno;
  et_dpo_config cfg;
  et_dpo_status st = validate_config_raw(cfg_raw, &cfg);
  if (st != ET_DPO_OK) {
    errno = saved_errno;
    return st;
  }

  const size_t bt_count = (size_t)cfg.batch_size * cfg.seq_len;
  const size_t logit_count = bt_count * cfg.vocab_size;
  const size_t logit_bytes = logit_count * sizeof(float);
  const size_t target_bytes = bt_count * sizeof(int64_t);
  const size_t mask_bytes = bt_count * sizeof(uint8_t);

  st = check_span(pi_w_raw, alignof(float), logit_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(pi_l_raw, alignof(float), logit_bytes);
  if (st != ET_DPO_OK)
    goto fail;

  if (cfg.reference_free == 0u) {
    st = check_span(ref_w_raw, alignof(float), logit_bytes);
    if (st != ET_DPO_OK)
      goto fail;
    st = check_span(ref_l_raw, alignof(float), logit_bytes);
    if (st != ET_DPO_OK)
      goto fail;
  } else {
    if (ref_w_raw != NULL) {
      st = check_span(ref_w_raw, alignof(float), logit_bytes);
      if (st != ET_DPO_OK)
        goto fail;
    }
    if (ref_l_raw != NULL) {
      st = check_span(ref_l_raw, alignof(float), logit_bytes);
      if (st != ET_DPO_OK)
        goto fail;
    }
  }

  st = check_span(tgt_w_raw, alignof(int64_t), target_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(tgt_l_raw, alignof(int64_t), target_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(mask_w_raw, alignof(uint8_t), mask_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(mask_l_raw, alignof(uint8_t), mask_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(telemetry_raw, alignof(et_dpo_telemetry),
                  sizeof(et_dpo_telemetry));
  if (st != ET_DPO_OK)
    goto fail;

  if (with_backward) {
    st = check_span(d_w_raw, alignof(float), logit_bytes);
    if (st != ET_DPO_OK)
      goto fail;
    st = check_span(d_l_raw, alignof(float), logit_bytes);
    if (st != ET_DPO_OK)
      goto fail;
  }

  span_entry spans[12];
  size_t span_count = 0u;
  spans[span_count++] = (span_entry){cfg_raw, sizeof(et_dpo_config)};
  spans[span_count++] = (span_entry){pi_w_raw, logit_bytes};
  spans[span_count++] = (span_entry){pi_l_raw, logit_bytes};
  if (ref_w_raw != NULL)
    spans[span_count++] = (span_entry){ref_w_raw, logit_bytes};
  if (ref_l_raw != NULL)
    spans[span_count++] = (span_entry){ref_l_raw, logit_bytes};
  spans[span_count++] = (span_entry){tgt_w_raw, target_bytes};
  spans[span_count++] = (span_entry){tgt_l_raw, target_bytes};
  spans[span_count++] = (span_entry){mask_w_raw, mask_bytes};
  spans[span_count++] = (span_entry){mask_l_raw, mask_bytes};
  spans[span_count++] = (span_entry){telemetry_raw, sizeof(et_dpo_telemetry)};
  if (with_backward) {
    spans[span_count++] = (span_entry){d_w_raw, logit_bytes};
    spans[span_count++] = (span_entry){d_l_raw, logit_bytes};
  }
  if (any_spans_overlap(spans, span_count)) {
    st = ET_DPO_ERR_ALIAS;
    goto fail;
  }

  const float *pi_w = (const float *)pi_w_raw;
  const float *pi_l = (const float *)pi_l_raw;
  const float *ref_w = (const float *)ref_w_raw;
  const float *ref_l = (const float *)ref_l_raw;
  const int64_t *tgt_w = (const int64_t *)tgt_w_raw;
  const int64_t *tgt_l = (const int64_t *)tgt_l_raw;
  const uint8_t *mask_w = (const uint8_t *)mask_w_raw;
  const uint8_t *mask_l = (const uint8_t *)mask_l_raw;

  st = validate_targets_and_mask(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                 tgt_w, mask_w);
  if (st != ET_DPO_OK)
    goto fail;
  st = validate_targets_and_mask(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                 tgt_l, mask_l);
  if (st != ET_DPO_OK)
    goto fail;

  if (!all_f32_finite(pi_w, logit_count) ||
      !all_f32_finite(pi_l, logit_count)) {
    st = ET_DPO_ERR_NONFINITE;
    goto fail;
  }
  if (cfg.reference_free == 0u) {
    if (!all_f32_finite(ref_w, logit_count) ||
        !all_f32_finite(ref_l, logit_count)) {
      st = ET_DPO_ERR_NONFINITE;
      goto fail;
    }
  }

  double logp_pi_w[ET_DPO_MAX_B];
  double logp_pi_l[ET_DPO_MAX_B];
  double logp_ref_w[ET_DPO_MAX_B];
  double logp_ref_l[ET_DPO_MAX_B];

  st = compute_seq_logps_checked(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                 pi_w, tgt_w, mask_w, logp_pi_w);
  if (st != ET_DPO_OK)
    goto fail;
  st = compute_seq_logps_checked(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                 pi_l, tgt_l, mask_l, logp_pi_l);
  if (st != ET_DPO_OK)
    goto fail;

  if (cfg.reference_free == 0u) {
    st = compute_seq_logps_checked(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                   ref_w, tgt_w, mask_w, logp_ref_w);
    if (st != ET_DPO_OK)
      goto fail;
    st = compute_seq_logps_checked(cfg.batch_size, cfg.seq_len, cfg.vocab_size,
                                   ref_l, tgt_l, mask_l, logp_ref_l);
    if (st != ET_DPO_OK)
      goto fail;
  } else {
    for (uint32_t b = 0u; b < cfg.batch_size; ++b) {
      logp_ref_w[b] = 0.0;
      logp_ref_l[b] = 0.0;
    }
  }

  const double inv_b = 1.0 / (double)cfg.batch_size;
  const double eps = cfg.label_smoothing;
  double loss_sum = 0.0;
  double r_w_sum = 0.0;
  double r_l_sum = 0.0;
  double margin_sum = 0.0;
  double acc_sum = 0.0;
  double g_w[ET_DPO_MAX_B];
  double g_l[ET_DPO_MAX_B];

  for (uint32_t b = 0u; b < cfg.batch_size; ++b) {
    const double delta_w = logp_pi_w[b] - logp_ref_w[b];
    const double delta_l = logp_pi_l[b] - logp_ref_l[b];
    const double r_w = cfg.beta * delta_w;
    const double r_l = cfg.beta * delta_l;
    const double margin = cfg.beta * (delta_w - delta_l);
    if (!isfinite(r_w) || !isfinite(r_l) || !isfinite(margin)) {
      st = ET_DPO_ERR_NONFINITE;
      goto fail;
    }

    const double loss_b =
        (eps == 0.0)
            ? stable_softplus(-margin)
            : ((1.0 - eps) * stable_softplus(-margin) +
               eps * stable_softplus(margin));
    if (!isfinite(loss_b)) {
      st = ET_DPO_ERR_NONFINITE;
      goto fail;
    }

    /* Exact derivative d(loss_b)/d(margin):
     *   -(1 - eps) * sigmoid(-margin) + eps * sigmoid(margin)
     *   = eps - sigmoid(-margin)        (used when margin >= 0)
     *   = sigmoid(margin) - (1 - eps)   (used when margin < 0)
     */
    const double dl_dm = (margin >= 0.0)
                             ? (eps - stable_sigmoid(-margin))
                             : (stable_sigmoid(margin) - (1.0 - eps));
    const double gw_b = (cfg.beta * inv_b) * dl_dm;
    const double gl_b = -gw_b;
    if (!isfinite(dl_dm) || !isfinite(gw_b) || !isfinite(gl_b)) {
      st = ET_DPO_ERR_NONFINITE;
      goto fail;
    }

    g_w[b] = gw_b;
    g_l[b] = gl_b;
    loss_sum += loss_b;
    r_w_sum += r_w;
    r_l_sum += r_l;
    margin_sum += margin;
    acc_sum += (margin > 0.0) ? 1.0 : 0.0;
  }

  et_dpo_telemetry telem;
  telem.mean_loss = loss_sum * inv_b;
  telem.chosen_reward_mean = r_w_sum * inv_b;
  telem.rejected_reward_mean = r_l_sum * inv_b;
  telem.reward_margin_mean = margin_sum * inv_b;
  telem.preference_accuracy = acc_sum * inv_b;

  if (!isfinite(telem.mean_loss) || !isfinite(telem.chosen_reward_mean) ||
      !isfinite(telem.rejected_reward_mean) ||
      !isfinite(telem.reward_margin_mean) ||
      !isfinite(telem.preference_accuracy)) {
    st = ET_DPO_ERR_NONFINITE;
    goto fail;
  }

  if (with_backward) {
    st = preflight_seq_grads(cfg.batch_size, cfg.seq_len, cfg.vocab_size, pi_w,
                             tgt_w, mask_w, g_w);
    if (st != ET_DPO_OK)
      goto fail;
    st = preflight_seq_grads(cfg.batch_size, cfg.seq_len, cfg.vocab_size, pi_l,
                             tgt_l, mask_l, g_l);
    if (st != ET_DPO_OK)
      goto fail;
  }

  /* Commit phase (non-failing, all validation and preflights passed). */
  *(et_dpo_telemetry *)telemetry_raw = telem;
  if (with_backward) {
    commit_seq_grads(cfg.batch_size, cfg.seq_len, cfg.vocab_size, pi_w, tgt_w,
                     mask_w, g_w, (float *)d_w_raw);
    commit_seq_grads(cfg.batch_size, cfg.seq_len, cfg.vocab_size, pi_l, tgt_l,
                     mask_l, g_l, (float *)d_l_raw);
  }
  errno = saved_errno;
  return ET_DPO_OK;

fail:
  errno = saved_errno;
  return st;
}

/* ---- Public ABI entry points ---- */

et_dpo_status et_dpo_validate_config(const et_dpo_config *cfg) {
  return validate_config_raw(cfg, NULL);
}

et_dpo_status et_dpo_sequence_logps_raw(uint32_t batch_size, uint32_t seq_len,
                                        uint32_t vocab_size, const void *logits,
                                        const void *targets, const void *mask,
                                        void *out_logps) {
  const int saved_errno = errno;
  if (!fp_environment_supported()) {
    errno = saved_errno;
    return ET_DPO_ERR_FP_ENV;
  }
  if (!valid_btv(batch_size, seq_len, vocab_size)) {
    errno = saved_errno;
    return ET_DPO_ERR_SHAPE;
  }
  const size_t bt_count = (size_t)batch_size * seq_len;
  const size_t logit_count = bt_count * vocab_size;
  const size_t logit_bytes = logit_count * sizeof(float);
  const size_t target_bytes = bt_count * sizeof(int64_t);
  const size_t mask_bytes = bt_count * sizeof(uint8_t);
  const size_t out_bytes = (size_t)batch_size * sizeof(double);

  et_dpo_status st = check_span(logits, alignof(float), logit_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(targets, alignof(int64_t), target_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(mask, alignof(uint8_t), mask_bytes);
  if (st != ET_DPO_OK)
    goto fail;
  st = check_span(out_logps, alignof(double), out_bytes);
  if (st != ET_DPO_OK)
    goto fail;

  const span_entry spans[4] = {
      {logits, logit_bytes},
      {targets, target_bytes},
      {mask, mask_bytes},
      {out_logps, out_bytes},
  };
  if (any_spans_overlap(spans, 4u)) {
    st = ET_DPO_ERR_ALIAS;
    goto fail;
  }

  const float *l_ptr = (const float *)logits;
  const int64_t *t_ptr = (const int64_t *)targets;
  const uint8_t *m_ptr = (const uint8_t *)mask;

  st = validate_targets_and_mask(batch_size, seq_len, vocab_size, t_ptr, m_ptr);
  if (st != ET_DPO_OK)
    goto fail;
  if (!all_f32_finite(l_ptr, logit_count)) {
    st = ET_DPO_ERR_NONFINITE;
    goto fail;
  }

  double staged[ET_DPO_MAX_B];
  st = compute_seq_logps_checked(batch_size, seq_len, vocab_size, l_ptr, t_ptr,
                                 m_ptr, staged);
  if (st != ET_DPO_OK)
    goto fail;

  memcpy(out_logps, staged, out_bytes);
  errno = saved_errno;
  return ET_DPO_OK;

fail:
  errno = saved_errno;
  return st;
}

et_dpo_status et_dpo_sequence_logps(uint32_t batch_size, uint32_t seq_len,
                                    uint32_t vocab_size, const float *logits,
                                    const int64_t *targets, const uint8_t *mask,
                                    double *out_logps) {
  return et_dpo_sequence_logps_raw(batch_size, seq_len, vocab_size, logits,
                                   targets, mask, out_logps);
}

et_dpo_status et_dpo_forward_raw(const void *cfg,
                                 const void *policy_chosen_logits,
                                 const void *policy_rejected_logits,
                                 const void *ref_chosen_logits,
                                 const void *ref_rejected_logits,
                                 const void *chosen_targets,
                                 const void *rejected_targets,
                                 const void *chosen_mask,
                                 const void *rejected_mask,
                                 void *out_telemetry) {
  return run_dpo_raw(cfg, policy_chosen_logits, policy_rejected_logits,
                     ref_chosen_logits, ref_rejected_logits, chosen_targets,
                     rejected_targets, chosen_mask, rejected_mask,
                     out_telemetry, 0, NULL, NULL);
}

et_dpo_status et_dpo_forward(const et_dpo_config *cfg,
                             const float *policy_chosen_logits,
                             const float *policy_rejected_logits,
                             const float *ref_chosen_logits,
                             const float *ref_rejected_logits,
                             const int64_t *chosen_targets,
                             const int64_t *rejected_targets,
                             const uint8_t *chosen_mask,
                             const uint8_t *rejected_mask,
                             et_dpo_telemetry *out_telemetry) {
  return et_dpo_forward_raw(cfg, policy_chosen_logits, policy_rejected_logits,
                            ref_chosen_logits, ref_rejected_logits,
                            chosen_targets, rejected_targets, chosen_mask,
                            rejected_mask, out_telemetry);
}

et_dpo_status et_dpo_forward_backward_raw(
    const void *cfg, const void *policy_chosen_logits,
    const void *policy_rejected_logits, const void *ref_chosen_logits,
    const void *ref_rejected_logits, const void *chosen_targets,
    const void *rejected_targets, const void *chosen_mask,
    const void *rejected_mask, void *out_telemetry, void *d_logits_chosen,
    void *d_logits_rejected) {
  return run_dpo_raw(cfg, policy_chosen_logits, policy_rejected_logits,
                     ref_chosen_logits, ref_rejected_logits, chosen_targets,
                     rejected_targets, chosen_mask, rejected_mask,
                     out_telemetry, 1, d_logits_chosen, d_logits_rejected);
}

et_dpo_status et_dpo_forward_backward(
    const et_dpo_config *cfg, const float *policy_chosen_logits,
    const float *policy_rejected_logits, const float *ref_chosen_logits,
    const float *ref_rejected_logits, const int64_t *chosen_targets,
    const int64_t *rejected_targets, const uint8_t *chosen_mask,
    const uint8_t *rejected_mask, et_dpo_telemetry *out_telemetry,
    float *d_logits_chosen, float *d_logits_rejected) {
  return et_dpo_forward_backward_raw(
      cfg, policy_chosen_logits, policy_rejected_logits, ref_chosen_logits,
      ref_rejected_logits, chosen_targets, rejected_targets, chosen_mask,
      rejected_mask, out_telemetry, d_logits_chosen, d_logits_rejected);
}

et_dpo_status et_dpo_weight_sha256_raw(size_t float_count, const void *weights,
                                       void *out_digest) {
  if (!fp_environment_supported())
    return ET_DPO_ERR_FP_ENV;
  if (float_count == 0u)
    return ET_DPO_ERR_SHAPE;
  if (float_count > SIZE_MAX / sizeof(float))
    return ET_DPO_ERR_OVERFLOW;
  const size_t weight_bytes = float_count * sizeof(float);
  et_dpo_status st = check_span(weights, alignof(float), weight_bytes);
  if (st != ET_DPO_OK)
    return st;
  st = check_span(out_digest, 1u, ET_DPO_SHA256_BYTES);
  if (st != ET_DPO_OK)
    return st;
  if (spans_overlap(weights, weight_bytes, out_digest, ET_DPO_SHA256_BYTES))
    return ET_DPO_ERR_ALIAS;

  const float *w = (const float *)weights;
  if (!all_f32_finite(w, float_count))
    return ET_DPO_ERR_NONFINITE;

  sha256_state state;
  sha256_init(&state);
  uint8_t chunk[256];
  size_t pos = 0u;
  while (pos < float_count) {
    size_t take = float_count - pos;
    if (take > 64u)
      take = 64u;
    for (size_t i = 0u; i < take; ++i) {
      uint32_t bits = 0u;
      memcpy(&bits, &w[pos + i], sizeof(bits));
      encode_u32_le(chunk + i * 4u, bits);
    }
    sha256_update(&state, chunk, take * 4u);
    pos += take;
  }
  uint8_t digest[ET_DPO_SHA256_BYTES];
  sha256_final(&state, digest);
  memcpy(out_digest, digest, ET_DPO_SHA256_BYTES);
  return ET_DPO_OK;
}

et_dpo_status et_dpo_weight_sha256(size_t float_count, const float *weights,
                                   uint8_t out_digest[ET_DPO_SHA256_BYTES]) {
  return et_dpo_weight_sha256_raw(float_count, weights, out_digest);
}
