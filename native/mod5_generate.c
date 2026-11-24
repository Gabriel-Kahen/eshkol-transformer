#include "eshkol_transformer/mod5_generate_abi.h"
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/mod5_primitives_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MOD5 generation requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 generation requires standard IEEE-754 binary32 evaluation");
_Static_assert(ET_MOD5_GEN_MAX_SEQ <= (uint32_t)ET_MOD5_MAX_EXACT_POSITION,
               "max_seq must fit within exact MOD5 position range");

/* Internal engine and KV-cache structures. */
struct et_mod5_gen_engine {
  et_mod5_gen_config cfg;
  uint32_t head_dim;
  uint32_t kv_dim;
  uint32_t global_loaded;
  uint32_t layer_loaded_mask;
  et_kernel_runtime *k1_runtime;

  /* Owned finite f32 weights in canonical MOD5 row-major [out, in] layout. */
  float *token_embedding; /* [V, D] */
  float *norm_final;      /* [D] */
  float *head;            /* [V, D] */

  float *norm1; /* [L, D] */
  float *wq;    /* [L, D, D] */
  float *wk;    /* [L, kv_dim, D] */
  float *wv;    /* [L, kv_dim, D] */
  float *wo;    /* [L, D, D] */
  float *norm2; /* [L, D] */
  float *wgate; /* [L, F, D] */
  float *wup;   /* [L, F, D] */
  float *wdown; /* [L, D, F] */
};

struct et_mod5_gen_kv_cache {
  uint32_t num_layers;
  uint32_t num_kv_heads;
  uint32_t max_seq;
  uint32_t head_dim;
  uint32_t seq_len;
  /* Physical layout: [L, Hkv, max_seq, Dh] dense row-major binary32. */
  float *k_data;
  float *v_data;
};

static int ptr_aligned(const void *p, size_t alignment) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u;
}

static int fp_env_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int checked_mul_size(size_t a, size_t b, size_t *out) {
  if (b != 0u && a > SIZE_MAX / b)
    return 0;
  *out = a * b;
  return 1;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static int f64_to_finite_f32(double value, float *out) {
  if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  const float converted = (float)value;
  if (!isfinite(converted))
    return 0;
  *out = converted;
  return 1;
}

/* ---------------- Counter-based SplitMix64 RNG ---------------- */

static uint64_t splitmix64_mix(uint64_t z) {
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static uint64_t splitmix64_block(uint64_t key, uint64_t counter) {
  const uint64_t key_mix = splitmix64_mix(key ^ UINT64_C(0x243f6a8885a308d3));
  const uint64_t step = (counter + UINT64_C(1)) * UINT64_C(0x9e3779b97f4a7c15);
  return splitmix64_mix(key_mix + step);
}

et_mod5_gen_status et_mod5_gen_rng_init(et_mod5_gen_rng *rng, uint64_t seed) {
  if (!ptr_aligned(rng, _Alignof(et_mod5_gen_rng)))
    return rng == NULL ? ET_MOD5_GEN_ERR_ARGUMENT : ET_MOD5_GEN_ERR_UNALIGNED;
  rng->key = seed;
  rng->counter = 0u;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_rng_next_u64(et_mod5_gen_rng *rng,
                                            uint64_t *out_value) {
  if (rng == NULL || out_value == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(rng, _Alignof(et_mod5_gen_rng)) ||
      !ptr_aligned(out_value, _Alignof(uint64_t)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (rng->counter == UINT64_MAX)
    return ET_MOD5_GEN_ERR_OVERFLOW;
  const uint64_t val = splitmix64_block(rng->key, rng->counter);
  rng->counter += UINT64_C(1);
  *out_value = val;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_rng_next_f64(et_mod5_gen_rng *rng,
                                            double *out_unit) {
  if (rng == NULL || out_unit == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(rng, _Alignof(et_mod5_gen_rng)) ||
      !ptr_aligned(out_unit, _Alignof(double)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;
  uint64_t raw = 0u;
  const et_mod5_gen_status st = et_mod5_gen_rng_next_u64(rng, &raw);
  if (st != ET_MOD5_GEN_OK)
    return st;
  /* Exact 53-bit IEEE-754 binary64 uniform sample in [0.0, 1.0). */
  *out_unit = (double)(raw >> 11u) * (1.0 / 9007199254740992.0);
  return ET_MOD5_GEN_OK;
}

/* ---------------- Config Validation ---------------- */

et_mod5_gen_status et_mod5_gen_validate_config(
    const et_mod5_gen_config *config) {
  if (config == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(config, _Alignof(et_mod5_gen_config)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (config->vocab_size == 0u || config->vocab_size > ET_MOD5_GEN_MAX_VOCAB)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->max_seq == 0u || config->max_seq > ET_MOD5_GEN_MAX_SEQ)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->num_layers == 0u || config->num_layers > ET_MOD5_GEN_MAX_LAYERS)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->hidden_dim == 0u || config->hidden_dim > ET_MOD5_GEN_MAX_DIM)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->ffn_dim == 0u || config->ffn_dim > ET_MOD5_GEN_MAX_FFN)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->num_heads == 0u || config->num_heads > ET_MOD5_GEN_MAX_HEADS)
    return ET_MOD5_GEN_ERR_CONFIG;
  if ((config->hidden_dim % config->num_heads) != 0u)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->num_kv_heads == 0u || config->num_kv_heads > config->num_heads ||
      (config->num_heads % config->num_kv_heads) != 0u)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->use_alibi > 1u)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (!isfinite(config->rms_epsilon) || !(config->rms_epsilon > 0.0f))
    return ET_MOD5_GEN_ERR_CONFIG;
  if (config->use_alibi == 1u) {
    for (uint32_t h = 0u; h < config->num_heads; ++h) {
      if (!isfinite(config->alibi_slopes[h]))
        return ET_MOD5_GEN_ERR_NONFINITE;
    }
  }
  return ET_MOD5_GEN_OK;
}

/* ---------------- K1 MOD5 Provider Helpers ---------------- */

static const et_kernel_provider_v1 *resolve_mod5_provider(void *context,
                                                          const char *symbol) {
  (void)context;
  return (symbol != NULL && strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0)
             ? et_mod5_kernel_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 make_view(void *data, const char *dtype,
                                          size_t elem_bytes, size_t rank,
                                          const uint64_t *shape) {
  size_t bytes = elem_bytes;
  for (size_t i = 0u; i < rank; ++i)
    bytes *= (size_t)shape[i];
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                    data,
                                    bytes,
                                    dtype,
                                    "cpu",
                                    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    0u,
                                    rank,
                                    shape};
}

static int is_mod5_rmsnorm_row(size_t t, size_t d) {
  return (t == 1u && d == 1u) || (t == 1u && d == 4u) ||
         (t == 2u && d == 4u) || (t == 8u && d == 16u) ||
         (t == 16u && d == 32u);
}

static int is_mod5_swiglu_row(size_t t, size_t f) {
  return (t == 1u && f == 1u) || (t == 1u && f == 8u) ||
         (t == 2u && f == 4u) || (t == 8u && f == 16u) ||
         (t == 8u && f == 32u) || (t == 16u && f == 64u);
}

static int is_mod5_attention_row(size_t hq, size_t hkv, size_t t, size_t dh) {
  return (hq == 2u && hkv == 1u && t == 2u && dh == 2u) ||
         (hq == 4u && hkv == 1u && t == 3u && dh == 2u) ||
         (hq == 4u && hkv == 2u && t == 3u && dh == 2u) ||
         (hq == 2u && hkv == 2u && t == 3u && dh == 2u) ||
         (hq == 4u && hkv == 1u && t == 8u && dh == 4u) ||
         (hq == 4u && hkv == 2u && t == 8u && dh == 4u) ||
         (hq == 4u && hkv == 1u && t == 16u && dh == 8u) ||
         (hq == 4u && hkv == 2u && t == 16u && dh == 8u);
}

static int dispatch_mod5_rmsnorm(const et_kernel_runtime *rt, size_t t,
                                 size_t d, float *x, float *gamma_w,
                                 float *eps_ptr, float *out) {
  const uint64_t shape[3] = {1u, (uint64_t)t, (uint64_t)d};
  const et_kernel_tensor_view_v1 ins[3] = {
      make_view(x, "f32", sizeof(float), 3u, shape),
      make_view(gamma_w, "f32", sizeof(float), 1u, shape + 2u),
      make_view(eps_ptr, "f32", sizeof(float), 0u, NULL)};
  et_kernel_tensor_view_v1 outs[1] = {
      make_view(out, "f32", sizeof(float), 3u, shape)};
  const et_kernel_request_v1 req = {
      sizeof(req), "rmsnorm.forward", "f32", "cpu", 3u, shape, 1u, {0}};
  const et_kernel_call_v1 call = {sizeof(call),    "kernel.rmsnorm",
                                  &req,            3u,
                                  sizeof(ins[0]),  sizeof(ins),
                                  ins,             1u,
                                  sizeof(outs[0]), sizeof(outs),
                                  outs};
  et_kernel_error err;
  return et_kernel_runtime_dispatch(rt, &call, &err) == 0;
}

static int dispatch_mod5_swiglu(const et_kernel_runtime *rt, size_t t, size_t f,
                                float *gate, float *up, float *out) {
  const uint64_t shape[3] = {1u, (uint64_t)t, (uint64_t)f};
  const et_kernel_tensor_view_v1 ins[2] = {
      make_view(gate, "f32", sizeof(float), 3u, shape),
      make_view(up, "f32", sizeof(float), 3u, shape)};
  et_kernel_tensor_view_v1 outs[1] = {
      make_view(out, "f32", sizeof(float), 3u, shape)};
  const et_kernel_request_v1 req = {
      sizeof(req), "swiglu.forward", "f32", "cpu", 3u, shape, 1u, {0}};
  const et_kernel_call_v1 call = {sizeof(call),    "kernel.swiglu",
                                  &req,            2u,
                                  sizeof(ins[0]),  sizeof(ins),
                                  ins,             1u,
                                  sizeof(outs[0]), sizeof(outs),
                                  outs};
  et_kernel_error err;
  return et_kernel_runtime_dispatch(rt, &call, &err) == 0;
}

static int dispatch_mod5_attention(const et_kernel_runtime *rt, size_t hq,
                                   size_t hkv, size_t t, size_t dh, float *q,
                                   float *k, float *v, int64_t *qpos,
                                   int64_t *kpos, uint8_t *mask, float *slopes,
                                   int64_t *window_ptr, float *out) {
  const uint64_t shape[6] = {1u,          (uint64_t)hq, (uint64_t)hkv,
                             (uint64_t)t, (uint64_t)t,  (uint64_t)dh};
  const uint64_t q_shape[4] = {1u, (uint64_t)hq, (uint64_t)t, (uint64_t)dh};
  const uint64_t kv_shape[4] = {1u, (uint64_t)hkv, (uint64_t)t, (uint64_t)dh};
  const uint64_t pos_shape[2] = {1u, (uint64_t)t};
  const uint64_t mask_shape[3] = {1u, (uint64_t)t, (uint64_t)t};
  const uint64_t slope_shape[1] = {(uint64_t)hq};

  const et_kernel_tensor_view_v1 ins[8] = {
      make_view(q, "f32", sizeof(float), 4u, q_shape),
      make_view(k, "f32", sizeof(float), 4u, kv_shape),
      make_view(v, "f32", sizeof(float), 4u, kv_shape),
      make_view(qpos, "i64", sizeof(int64_t), 2u, pos_shape),
      make_view(kpos, "i64", sizeof(int64_t), 2u, pos_shape),
      make_view(mask, "bool", sizeof(uint8_t), 3u, mask_shape),
      make_view(slopes, "f32", sizeof(float), 1u, slope_shape),
      make_view(window_ptr, "i64", sizeof(int64_t), 0u, NULL)};
  et_kernel_tensor_view_v1 outs[1] = {
      make_view(out, "f32", sizeof(float), 4u, q_shape)};
  const et_kernel_request_v1 req = {sizeof(req),
                                    "variant-attention.forward",
                                    "f32",
                                    "cpu",
                                    6u,
                                    shape,
                                    1u,
                                    {0}};
  const et_kernel_call_v1 call = {sizeof(call),    "kernel.variant-attention",
                                  &req,            8u,
                                  sizeof(ins[0]),  sizeof(ins),
                                  ins,             1u,
                                  sizeof(outs[0]), sizeof(outs),
                                  outs};
  et_kernel_error err;
  return et_kernel_runtime_dispatch(rt, &call, &err) == 0;
}

/* ---------------- Serial Binary32 Primitives (MOD5 exact) ---------------- */

static int serial_rmsnorm_f32(const float *x, const float *gamma_w, size_t t,
                              size_t d, float eps, float *out) {
  if (!(eps > 0.0f) || !isfinite(eps))
    return 0;
  for (size_t row = 0u; row < t; ++row) {
    const size_t base = row * d;
    float sum = 0.0f;
    for (size_t i = 0u; i < d; ++i) {
      const float val = x[base + i];
      sum = sum + val * val;
      if (!isfinite(sum))
        return 0;
    }
    const float mean_sq = sum / (float)d;
    const float stabilized = mean_sq + eps;
    if (!isfinite(stabilized))
      return 0;
    const float rstd = 1.0f / sqrtf(stabilized);
    if (!isfinite(rstd))
      return 0;
    for (size_t i = 0u; i < d; ++i) {
      const float norm = x[base + i] * rstd;
      const float y = norm * gamma_w[i];
      if (!isfinite(y))
        return 0;
      out[base + i] = y;
    }
  }
  return 1;
}

static int serial_swiglu_f32(const float *gate, const float *up, size_t count,
                             float *out) {
  for (size_t i = 0u; i < count; ++i) {
    const float exp_val = expf(-gate[i]);
    const float denom = 1.0f + exp_val;
    if (!isfinite(exp_val) || !isfinite(denom))
      return 0;
    const float sig = 1.0f / denom;
    const float silu = gate[i] * sig;
    if (!isfinite(sig) || !isfinite(silu))
      return 0;
    const float y = silu * up[i];
    if (!isfinite(y))
      return 0;
    out[i] = y;
  }
  return 1;
}

static int serial_linear_f32(const float *x, const float *w, size_t t,
                             size_t in_dim, size_t out_dim, float *out) {
  /* x: [t, in_dim], w: [out_dim, in_dim], out: [t, out_dim] */
  for (size_t row = 0u; row < t; ++row) {
    const float *x_row = x + row * in_dim;
    float *y_row = out + row * out_dim;
    for (size_t o = 0u; o < out_dim; ++o) {
      const float *w_row = w + o * in_dim;
      float sum = 0.0f;
      for (size_t i = 0u; i < in_dim; ++i) {
        sum = sum + x_row[i] * w_row[i];
        if (!isfinite(sum))
          return 0;
      }
      y_row[o] = sum;
    }
  }
  return 1;
}

static int serial_add_f32(const float *a, const float *b, size_t count,
                          float *out) {
  for (size_t i = 0u; i < count; ++i) {
    const float s = a[i] + b[i];
    if (!isfinite(s))
      return 0;
    out[i] = s;
  }
  return 1;
}

/*
 * Serial causal variant attention matching MOD5 binary32 evaluation order:
 * - q_tokens: [tq, Hq * Dh] (for positions start_pos .. start_pos + tq - 1)
 * - past_k, past_v: cached prefix [Hkv, max_seq, Dh] valid for [0 .. start_pos - 1]
 * - new_k, new_v:   new slice    [Hkv, tq, Dh]      valid for [start_pos .. start_pos + tq - 1]
 * - out_tokens:     [tq, Hq * Dh]
 */
static int serial_kv_attention_f32(const float *q_tokens, const float *past_k,
                                   const float *past_v, uint32_t past_stride,
                                   const float *new_k, const float *new_v,
                                   size_t start_pos, size_t tq, size_t hq,
                                   size_t hkv, size_t dh, uint32_t window_size,
                                   uint32_t use_alibi, const float *slopes,
                                   float *out_tokens) {
  const size_t group_size = hq / hkv;
  const size_t total_keys = start_pos + tq;
  const float inv_sqrt_dh = 1.0f / sqrtf((float)dh);
  if (!isfinite(inv_sqrt_dh))
    return 0;

  for (size_t q_idx = 0u; q_idx < tq; ++q_idx) {
    const int64_t q_pos = (int64_t)(start_pos + q_idx);
    for (size_t h = 0u; h < hq; ++h) {
      const size_t kv_h = h / group_size;
      const float *q_vec = q_tokens + (q_idx * hq + h) * dh;
      float *o_vec = out_tokens + (q_idx * hq + h) * dh;
      const float slope = use_alibi ? slopes[h] : 0.0f;

      size_t admitted_pos[ET_MOD5_GEN_MAX_SEQ];
      float scores[ET_MOD5_GEN_MAX_SEQ];
      size_t count = 0u;
      float max_score = -INFINITY;

      for (size_t k_pos_u = 0u; k_pos_u < total_keys; ++k_pos_u) {
        const int64_t k_pos = (int64_t)k_pos_u;
        const int64_t delta = k_pos - q_pos;
        if (delta > 0)
          continue;
        if (window_size > 0u && (-delta) >= (int64_t)window_size)
          continue;

        const float *k_vec =
            (k_pos_u < start_pos)
                ? (past_k + (kv_h * (size_t)past_stride + k_pos_u) * dh)
                : (new_k + (kv_h * tq + (k_pos_u - start_pos)) * dh);

        float dot = 0.0f;
        for (size_t d = 0u; d < dh; ++d) {
          dot = dot + q_vec[d] * k_vec[d];
          if (!isfinite(dot))
            return 0;
        }
        const float scaled = dot * inv_sqrt_dh;
        const float bias = slope * (float)delta;
        const float score = scaled + bias;
        if (!isfinite(score))
          return 0;
        if (score > max_score)
          max_score = score;
        admitted_pos[count] = k_pos_u;
        scores[count] = score;
        ++count;
      }

      for (size_t d = 0u; d < dh; ++d)
        o_vec[d] = 0.0f;
      if (count == 0u)
        continue;

      float denom = 0.0f;
      for (size_t idx = 0u; idx < count; ++idx) {
        const float shifted = scores[idx] - max_score;
        if (!isfinite(shifted))
          return 0;
        const float weight = expf(shifted);
        denom = denom + weight;
        if (!isfinite(denom))
          return 0;
        scores[idx] = weight;
      }

      for (size_t idx = 0u; idx < count; ++idx) {
        const float prob = scores[idx] / denom;
        if (!isfinite(prob))
          return 0;
        const size_t k_pos_u = admitted_pos[idx];
        const float *v_vec =
            (k_pos_u < start_pos)
                ? (past_v + (kv_h * (size_t)past_stride + k_pos_u) * dh)
                : (new_v + (kv_h * tq + (k_pos_u - start_pos)) * dh);
        for (size_t d = 0u; d < dh; ++d) {
          o_vec[d] = o_vec[d] + prob * v_vec[d];
          if (!isfinite(o_vec[d]))
            return 0;
        }
      }
    }
  }
  return 1;
}

/* ---------------- Provider Parity Verification ---------------- */

et_mod5_gen_status et_mod5_gen_verify_provider(double *out_max_abs_diff) {
  if (out_max_abs_diff == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(out_max_abs_diff, _Alignof(double)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;

  et_kernel_runtime *rt = NULL;
  et_kernel_error err;
  if (et_kernel_runtime_discover(resolve_mod5_provider, NULL, &rt, &err) != 0 ||
      rt == NULL)
    return ET_MOD5_GEN_ERR_PROVIDER;

  double worst = 0.0;
  int ok = 1;

  /* 1. Verify RMSNorm [1, 8, 16] and [1, 16, 32] */
  {
    const size_t shapes[2][2] = {{8u, 16u}, {16u, 32u}};
    for (size_t c = 0u; c < 2u && ok; ++c) {
      const size_t t = shapes[c][0];
      const size_t d = shapes[c][1];
      float x[16 * 32];
      float g[32];
      float eps = 0.125f;
      float out_k1[16 * 32];
      float out_ref[16 * 32];
      for (size_t i = 0u; i < t * d; ++i)
        x[i] = (float)((int)(i % 19u) - 9) * 0.125f;
      for (size_t i = 0u; i < d; ++i)
        g[i] = 1.0f + (float)((int)(i % 7u) - 3) * 0.0625f;
      if (!dispatch_mod5_rmsnorm(rt, t, d, x, g, &eps, out_k1) ||
          !serial_rmsnorm_f32(x, g, t, d, eps, out_ref)) {
        ok = 0;
        break;
      }
      for (size_t i = 0u; i < t * d; ++i) {
        const double diff = fabs((double)out_k1[i] - (double)out_ref[i]);
        if (diff > worst)
          worst = diff;
      }
    }
  }

  /* 2. Verify SwiGLU [1, 8, 32] and [1, 16, 64] */
  {
    const size_t shapes[2][2] = {{8u, 32u}, {16u, 64u}};
    for (size_t c = 0u; c < 2u && ok; ++c) {
      const size_t t = shapes[c][0];
      const size_t f = shapes[c][1];
      float gate[16 * 64];
      float up[16 * 64];
      float out_k1[16 * 64];
      float out_ref[16 * 64];
      for (size_t i = 0u; i < t * f; ++i) {
        gate[i] = (float)((int)(i % 23u) - 11) * 0.125f;
        up[i] = (float)((int)((i * 3u) % 17u) - 8) * 0.125f;
      }
      if (!dispatch_mod5_swiglu(rt, t, f, gate, up, out_k1) ||
          !serial_swiglu_f32(gate, up, t * f, out_ref)) {
        ok = 0;
        break;
      }
      for (size_t i = 0u; i < t * f; ++i) {
        const double diff = fabs((double)out_k1[i] - (double)out_ref[i]);
        if (diff > worst)
          worst = diff;
      }
    }
  }

  /* 3. Verify Variant Attention: MQA [1,4,1,8,8,4], GQA [1,4,2,8,8,4], GQA [1,4,2,16,16,8] */
  {
    const size_t cases[3][4] = {
        {4u, 1u, 8u, 4u}, {4u, 2u, 8u, 4u}, {4u, 2u, 16u, 8u}};
    const int64_t windows[3] = {4, 5, 6};
    for (size_t c = 0u; c < 3u && ok; ++c) {
      const size_t hq = cases[c][0];
      const size_t hkv = cases[c][1];
      const size_t t = cases[c][2];
      const size_t dh = cases[c][3];
      float q_htd[4 * 16 * 8];
      float k_htd[2 * 16 * 8];
      float v_htd[2 * 16 * 8];
      float q_thd[16 * 4 * 8];
      float out_k1_htd[4 * 16 * 8];
      float out_ref_thd[16 * 4 * 8];
      int64_t qpos[16];
      int64_t kpos[16];
      uint8_t mask[16 * 16];
      float slopes[4] = {0.5f, 0.25f, 0.125f, 0.0625f};
      int64_t win = windows[c];

      for (size_t i = 0u; i < t; ++i) {
        qpos[i] = (int64_t)i;
        kpos[i] = (int64_t)i;
      }
      memset(mask, 1, t * t);
      for (size_t h = 0u; h < hq; ++h) {
        for (size_t ti = 0u; ti < t; ++ti) {
          for (size_t d = 0u; d < dh; ++d) {
            const float val =
                (float)((int)((h * 13u + ti * 7u + d * 5u) % 21u) - 10) *
                0.09375f;
            q_htd[(h * t + ti) * dh + d] = val;
            q_thd[(ti * hq + h) * dh + d] = val;
          }
        }
      }
      for (size_t h = 0u; h < hkv; ++h) {
        for (size_t ti = 0u; ti < t; ++ti) {
          for (size_t d = 0u; d < dh; ++d) {
            k_htd[(h * t + ti) * dh + d] =
                (float)((int)((h * 17u + ti * 11u + d * 3u) % 19u) - 9) *
                0.09375f;
            v_htd[(h * t + ti) * dh + d] =
                (float)((int)((h * 19u + ti * 5u + d * 7u) % 23u) - 11) *
                0.0625f;
          }
        }
      }

      if (!dispatch_mod5_attention(rt, hq, hkv, t, dh, q_htd, k_htd, v_htd,
                                   qpos, kpos, mask, slopes, &win,
                                   out_k1_htd) ||
          !serial_kv_attention_f32(q_thd, NULL, NULL, 0u, k_htd, v_htd, 0u, t,
                                   hq, hkv, dh, (uint32_t)win, 1u, slopes,
                                   out_ref_thd)) {
        ok = 0;
        break;
      }

      for (size_t h = 0u; h < hq; ++h) {
        for (size_t ti = 0u; ti < t; ++ti) {
          for (size_t d = 0u; d < dh; ++d) {
            const float a = out_k1_htd[(h * t + ti) * dh + d];
            const float b = out_ref_thd[(ti * hq + h) * dh + d];
            const double diff = fabs((double)a - (double)b);
            if (diff > worst)
              worst = diff;
          }
        }
      }
    }
  }

  et_kernel_runtime_destroy(rt);
  if (!ok)
    return ET_MOD5_GEN_ERR_PROVIDER;
  *out_max_abs_diff = worst;
  return worst == 0.0 ? ET_MOD5_GEN_OK : ET_MOD5_GEN_ERR_PROVIDER;
}

/* ---------------- Engine Lifecycle & Weight Loading ---------------- */

et_mod5_gen_status et_mod5_gen_engine_create(const et_mod5_gen_config *config,
                                             et_mod5_gen_engine **out_engine) {
  if (config == NULL || out_engine == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(out_engine, _Alignof(et_mod5_gen_engine *)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  const et_mod5_gen_status cfg_st = et_mod5_gen_validate_config(config);
  if (cfg_st != ET_MOD5_GEN_OK)
    return cfg_st;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;

  et_mod5_gen_engine *eng =
      (et_mod5_gen_engine *)calloc(1u, sizeof(et_mod5_gen_engine));
  if (eng == NULL)
    return ET_MOD5_GEN_ERR_NOMEM;

  eng->cfg = *config;
  if (eng->cfg.use_alibi == 0u) {
    for (uint32_t h = 0u; h < ET_MOD5_GEN_MAX_HEADS; ++h)
      eng->cfg.alibi_slopes[h] = 0.0f;
  }
  eng->head_dim = config->hidden_dim / config->num_heads;
  eng->kv_dim = config->num_kv_heads * eng->head_dim;

  et_kernel_error err;
  if (et_kernel_runtime_discover(resolve_mod5_provider, NULL, &eng->k1_runtime,
                                 &err) != 0 ||
      eng->k1_runtime == NULL) {
    free(eng);
    return ET_MOD5_GEN_ERR_PROVIDER;
  }

  const size_t V = config->vocab_size;
  const size_t D = config->hidden_dim;
  const size_t F = config->ffn_dim;
  const size_t L = config->num_layers;
  const size_t KVD = eng->kv_dim;

  eng->token_embedding = (float *)calloc(V * D, sizeof(float));
  eng->norm_final = (float *)calloc(D, sizeof(float));
  eng->head = (float *)calloc(V * D, sizeof(float));
  eng->norm1 = (float *)calloc(L * D, sizeof(float));
  eng->wq = (float *)calloc(L * D * D, sizeof(float));
  eng->wk = (float *)calloc(L * KVD * D, sizeof(float));
  eng->wv = (float *)calloc(L * KVD * D, sizeof(float));
  eng->wo = (float *)calloc(L * D * D, sizeof(float));
  eng->norm2 = (float *)calloc(L * D, sizeof(float));
  eng->wgate = (float *)calloc(L * F * D, sizeof(float));
  eng->wup = (float *)calloc(L * F * D, sizeof(float));
  eng->wdown = (float *)calloc(L * D * F, sizeof(float));

  if (!eng->token_embedding || !eng->norm_final || !eng->head || !eng->norm1 ||
      !eng->wq || !eng->wk || !eng->wv || !eng->wo || !eng->norm2 ||
      !eng->wgate || !eng->wup || !eng->wdown) {
    et_mod5_gen_engine_destroy(eng);
    return ET_MOD5_GEN_ERR_NOMEM;
  }

  *out_engine = eng;
  return ET_MOD5_GEN_OK;
}

void et_mod5_gen_engine_destroy(et_mod5_gen_engine *engine) {
  if (engine == NULL)
    return;
  if (engine->k1_runtime != NULL)
    et_kernel_runtime_destroy(engine->k1_runtime);
  free(engine->token_embedding);
  free(engine->norm_final);
  free(engine->head);
  free(engine->norm1);
  free(engine->wq);
  free(engine->wk);
  free(engine->wv);
  free(engine->wo);
  free(engine->norm2);
  free(engine->wgate);
  free(engine->wup);
  free(engine->wdown);
  free(engine);
}

static int engine_weights_ready(const et_mod5_gen_engine *engine) {
  const uint32_t expected_mask = (1u << engine->cfg.num_layers) - 1u;
  return engine->global_loaded == 1u &&
         engine->layer_loaded_mask == expected_mask;
}

et_mod5_gen_status et_mod5_gen_engine_set_global_weights(
    et_mod5_gen_engine *engine, const float *token_embedding,
    const float *norm_final, const float *head) {
  if (engine == NULL || token_embedding == NULL || norm_final == NULL ||
      head == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(token_embedding, _Alignof(float)) ||
      !ptr_aligned(norm_final, _Alignof(float)) ||
      !ptr_aligned(head, _Alignof(float)))
    return ET_MOD5_GEN_ERR_UNALIGNED;

  const size_t vd = (size_t)engine->cfg.vocab_size * engine->cfg.hidden_dim;
  const size_t d = engine->cfg.hidden_dim;
  if (!all_finite_f32(token_embedding, vd) || !all_finite_f32(norm_final, d) ||
      !all_finite_f32(head, vd))
    return ET_MOD5_GEN_ERR_NONFINITE;

  memcpy(engine->token_embedding, token_embedding, vd * sizeof(float));
  memcpy(engine->norm_final, norm_final, d * sizeof(float));
  memcpy(engine->head, head, vd * sizeof(float));
  engine->global_loaded = 1u;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_engine_set_layer_weights(
    et_mod5_gen_engine *engine, uint32_t layer_index, const float *norm1,
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *norm2, const float *wgate, const float *wup,
    const float *wdown) {
  if (engine == NULL || norm1 == NULL || wq == NULL || wk == NULL ||
      wv == NULL || wo == NULL || norm2 == NULL || wgate == NULL ||
      wup == NULL || wdown == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(norm1, _Alignof(float)) ||
      !ptr_aligned(wq, _Alignof(float)) || !ptr_aligned(wk, _Alignof(float)) ||
      !ptr_aligned(wv, _Alignof(float)) || !ptr_aligned(wo, _Alignof(float)) ||
      !ptr_aligned(norm2, _Alignof(float)) ||
      !ptr_aligned(wgate, _Alignof(float)) ||
      !ptr_aligned(wup, _Alignof(float)) ||
      !ptr_aligned(wdown, _Alignof(float)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (layer_index >= engine->cfg.num_layers)
    return ET_MOD5_GEN_ERR_BOUNDS;

  const size_t d = engine->cfg.hidden_dim;
  const size_t f = engine->cfg.ffn_dim;
  const size_t kvd = engine->kv_dim;
  if (!all_finite_f32(norm1, d) || !all_finite_f32(wq, d * d) ||
      !all_finite_f32(wk, kvd * d) || !all_finite_f32(wv, kvd * d) ||
      !all_finite_f32(wo, d * d) || !all_finite_f32(norm2, d) ||
      !all_finite_f32(wgate, f * d) || !all_finite_f32(wup, f * d) ||
      !all_finite_f32(wdown, d * f))
    return ET_MOD5_GEN_ERR_NONFINITE;

  const size_t l = layer_index;
  memcpy(engine->norm1 + l * d, norm1, d * sizeof(float));
  memcpy(engine->wq + l * d * d, wq, d * d * sizeof(float));
  memcpy(engine->wk + l * kvd * d, wk, kvd * d * sizeof(float));
  memcpy(engine->wv + l * kvd * d, wv, kvd * d * sizeof(float));
  memcpy(engine->wo + l * d * d, wo, d * d * sizeof(float));
  memcpy(engine->norm2 + l * d, norm2, d * sizeof(float));
  memcpy(engine->wgate + l * f * d, wgate, f * d * sizeof(float));
  memcpy(engine->wup + l * f * d, wup, f * d * sizeof(float));
  memcpy(engine->wdown + l * d * f, wdown, d * f * sizeof(float));
  engine->layer_loaded_mask |= (1u << layer_index);
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_engine_load_weights(
    et_mod5_gen_engine *engine, const et_mod5_gen_model_weights *weights) {
  if (engine == NULL || weights == NULL || weights->layers == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(weights, _Alignof(et_mod5_gen_model_weights)) ||
      !ptr_aligned(weights->layers, _Alignof(et_mod5_gen_layer_weights)))
    return ET_MOD5_GEN_ERR_UNALIGNED;

  /* Validate all tensors before mutating engine state (atomic load). */
  const size_t d = engine->cfg.hidden_dim;
  const size_t f = engine->cfg.ffn_dim;
  const size_t kvd = engine->kv_dim;
  const size_t vd = (size_t)engine->cfg.vocab_size * d;
  if (weights->token_embedding == NULL || weights->norm_final == NULL ||
      weights->head == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(weights->token_embedding, _Alignof(float)) ||
      !ptr_aligned(weights->norm_final, _Alignof(float)) ||
      !ptr_aligned(weights->head, _Alignof(float)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!all_finite_f32(weights->token_embedding, vd) ||
      !all_finite_f32(weights->norm_final, d) ||
      !all_finite_f32(weights->head, vd))
    return ET_MOD5_GEN_ERR_NONFINITE;

  for (uint32_t l = 0u; l < engine->cfg.num_layers; ++l) {
    const et_mod5_gen_layer_weights *lw = &weights->layers[l];
    if (lw->norm1 == NULL || lw->wq == NULL || lw->wk == NULL ||
        lw->wv == NULL || lw->wo == NULL || lw->norm2 == NULL ||
        lw->wgate == NULL || lw->wup == NULL || lw->wdown == NULL)
      return ET_MOD5_GEN_ERR_ARGUMENT;
    if (!ptr_aligned(lw->norm1, _Alignof(float)) ||
        !ptr_aligned(lw->wq, _Alignof(float)) ||
        !ptr_aligned(lw->wk, _Alignof(float)) ||
        !ptr_aligned(lw->wv, _Alignof(float)) ||
        !ptr_aligned(lw->wo, _Alignof(float)) ||
        !ptr_aligned(lw->norm2, _Alignof(float)) ||
        !ptr_aligned(lw->wgate, _Alignof(float)) ||
        !ptr_aligned(lw->wup, _Alignof(float)) ||
        !ptr_aligned(lw->wdown, _Alignof(float)))
      return ET_MOD5_GEN_ERR_UNALIGNED;
    if (!all_finite_f32(lw->norm1, d) || !all_finite_f32(lw->wq, d * d) ||
        !all_finite_f32(lw->wk, kvd * d) || !all_finite_f32(lw->wv, kvd * d) ||
        !all_finite_f32(lw->wo, d * d) || !all_finite_f32(lw->norm2, d) ||
        !all_finite_f32(lw->wgate, f * d) || !all_finite_f32(lw->wup, f * d) ||
        !all_finite_f32(lw->wdown, d * f))
      return ET_MOD5_GEN_ERR_NONFINITE;
  }

  /* All validated; commit atomically. */
  et_mod5_gen_status st = et_mod5_gen_engine_set_global_weights(
      engine, weights->token_embedding, weights->norm_final, weights->head);
  if (st != ET_MOD5_GEN_OK)
    return st;
  for (uint32_t l = 0u; l < engine->cfg.num_layers; ++l) {
    const et_mod5_gen_layer_weights *lw = &weights->layers[l];
    st = et_mod5_gen_engine_set_layer_weights(
        engine, l, lw->norm1, lw->wq, lw->wk, lw->wv, lw->wo, lw->norm2,
        lw->wgate, lw->wup, lw->wdown);
    if (st != ET_MOD5_GEN_OK)
      return st;
  }
  return ET_MOD5_GEN_OK;
}

/* ---------------- KV-Cache Lifecycle & Inspection ---------------- */

et_mod5_gen_status et_mod5_gen_kv_cache_create(
    const et_mod5_gen_config *config, et_mod5_gen_kv_cache **out_cache) {
  if (config == NULL || out_cache == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(out_cache, _Alignof(et_mod5_gen_kv_cache *)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  const et_mod5_gen_status cfg_st = et_mod5_gen_validate_config(config);
  if (cfg_st != ET_MOD5_GEN_OK)
    return cfg_st;

  et_mod5_gen_kv_cache *cache =
      (et_mod5_gen_kv_cache *)calloc(1u, sizeof(et_mod5_gen_kv_cache));
  if (cache == NULL)
    return ET_MOD5_GEN_ERR_NOMEM;

  const uint32_t dh = config->hidden_dim / config->num_heads;
  cache->num_layers = config->num_layers;
  cache->num_kv_heads = config->num_kv_heads;
  cache->max_seq = config->max_seq;
  cache->head_dim = dh;
  cache->seq_len = 0u;

  const size_t total_elems = (size_t)cache->num_layers * cache->num_kv_heads *
                             cache->max_seq * cache->head_dim;
  cache->k_data = (float *)calloc(total_elems, sizeof(float));
  cache->v_data = (float *)calloc(total_elems, sizeof(float));
  if (cache->k_data == NULL || cache->v_data == NULL) {
    et_mod5_gen_kv_cache_destroy(cache);
    return ET_MOD5_GEN_ERR_NOMEM;
  }

  *out_cache = cache;
  return ET_MOD5_GEN_OK;
}

void et_mod5_gen_kv_cache_destroy(et_mod5_gen_kv_cache *cache) {
  if (cache == NULL)
    return;
  free(cache->k_data);
  free(cache->v_data);
  free(cache);
}

et_mod5_gen_status et_mod5_gen_kv_cache_reset(et_mod5_gen_kv_cache *cache) {
  if (cache == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  const size_t total_elems = (size_t)cache->num_layers * cache->num_kv_heads *
                             cache->max_seq * cache->head_dim;
  memset(cache->k_data, 0, total_elems * sizeof(float));
  memset(cache->v_data, 0, total_elems * sizeof(float));
  cache->seq_len = 0u;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_kv_cache_len(const et_mod5_gen_kv_cache *cache,
                                            uint32_t *out_len) {
  if (cache == NULL || out_len == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)) ||
      !ptr_aligned(out_len, _Alignof(uint32_t)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  *out_len = cache->seq_len;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_kv_cache_capacity(
    const et_mod5_gen_kv_cache *cache, uint32_t *out_max_seq) {
  if (cache == NULL || out_max_seq == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)) ||
      !ptr_aligned(out_max_seq, _Alignof(uint32_t)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  *out_max_seq = cache->max_seq;
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_kv_cache_read_slot(
    const et_mod5_gen_kv_cache *cache, uint32_t layer, uint32_t kv_head,
    uint32_t pos, float *k_out, float *v_out) {
  if (cache == NULL || k_out == NULL || v_out == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)) ||
      !ptr_aligned(k_out, _Alignof(float)) ||
      !ptr_aligned(v_out, _Alignof(float)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (layer >= cache->num_layers || kv_head >= cache->num_kv_heads ||
      pos >= cache->seq_len)
    return ET_MOD5_GEN_ERR_BOUNDS;

  const size_t offset =
      (((size_t)layer * cache->num_kv_heads + kv_head) * cache->max_seq + pos) *
      cache->head_dim;
  memcpy(k_out, cache->k_data + offset, cache->head_dim * sizeof(float));
  memcpy(v_out, cache->v_data + offset, cache->head_dim * sizeof(float));
  return ET_MOD5_GEN_OK;
}

static int cache_matches_engine(const et_mod5_gen_engine *engine,
                                const et_mod5_gen_kv_cache *cache) {
  return cache->num_layers == engine->cfg.num_layers &&
         cache->num_kv_heads == engine->cfg.num_kv_heads &&
         cache->max_seq == engine->cfg.max_seq &&
         cache->head_dim == engine->head_dim;
}

/* ---------------- Core Forward Path (Prefill / Decode / Recompute) ---------------- */

static int run_rmsnorm(const et_mod5_gen_engine *engine, const float *x,
                       const float *gamma_w, size_t t, size_t d, float *out) {
  if (engine->k1_runtime != NULL && is_mod5_rmsnorm_row(t, d)) {
    /* Copy into disjoint aligned buffers to satisfy strict K1 disjointness. */
    float x_buf[16 * 32];
    float g_buf[32];
    float eps_buf = engine->cfg.rms_epsilon;
    memcpy(x_buf, x, t * d * sizeof(float));
    memcpy(g_buf, gamma_w, d * sizeof(float));
    return dispatch_mod5_rmsnorm(engine->k1_runtime, t, d, x_buf, g_buf,
                                 &eps_buf, out);
  }
  return serial_rmsnorm_f32(x, gamma_w, t, d, engine->cfg.rms_epsilon, out);
}

static int run_swiglu(const et_mod5_gen_engine *engine, const float *gate,
                      const float *up, size_t t, size_t f, float *out) {
  if (engine->k1_runtime != NULL && is_mod5_swiglu_row(t, f)) {
    float gate_buf[16 * 64];
    float up_buf[16 * 64];
    memcpy(gate_buf, gate, t * f * sizeof(float));
    memcpy(up_buf, up, t * f * sizeof(float));
    return dispatch_mod5_swiglu(engine->k1_runtime, t, f, gate_buf, up_buf,
                                out);
  }
  return serial_swiglu_f32(gate, up, t * f, out);
}

/*
 * Execute forward pass on tokens[0..tq-1] starting at position start_pos.
 * If cache != NULL: reads past KV [0..start_pos-1] from cache and atomically
 * commits new KV [start_pos..start_pos+tq-1] into cache on success.
 * If cache == NULL: requires start_pos == 0 (full-context recompute).
 */
static et_mod5_gen_status forward_internal(const et_mod5_gen_engine *engine,
                                           et_mod5_gen_kv_cache *cache,
                                           const int64_t *tokens, size_t tq,
                                           float *out_last_logits,
                                           float *out_all_logits) {
  if (engine == NULL || tokens == NULL ||
      (out_last_logits == NULL && out_all_logits == NULL))
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(tokens, _Alignof(int64_t)) ||
      (out_last_logits != NULL &&
       !ptr_aligned(out_last_logits, _Alignof(float))) ||
      (out_all_logits != NULL &&
       !ptr_aligned(out_all_logits, _Alignof(float))))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;
  if (!engine_weights_ready(engine))
    return ET_MOD5_GEN_ERR_STATE;

  size_t start_pos = 0u;
  if (cache != NULL) {
    if (!ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)))
      return ET_MOD5_GEN_ERR_UNALIGNED;
    if (!cache_matches_engine(engine, cache))
      return ET_MOD5_GEN_ERR_CONFIG;
    start_pos = cache->seq_len;
  }

  if (tq == 0u)
    return ET_MOD5_GEN_ERR_BOUNDS;
  if (tq > engine->cfg.max_seq || start_pos > engine->cfg.max_seq - tq)
    return cache != NULL ? ET_MOD5_GEN_ERR_CACHE_FULL : ET_MOD5_GEN_ERR_BOUNDS;

  for (size_t i = 0u; i < tq; ++i) {
    if (tokens[i] < 0 || (uint64_t)tokens[i] >= engine->cfg.vocab_size)
      return ET_MOD5_GEN_ERR_BOUNDS;
  }

  const size_t V = engine->cfg.vocab_size;
  const size_t D = engine->cfg.hidden_dim;
  const size_t F = engine->cfg.ffn_dim;
  const size_t L = engine->cfg.num_layers;
  const size_t Hq = engine->cfg.num_heads;
  const size_t Hkv = engine->cfg.num_kv_heads;
  const size_t Dh = engine->head_dim;
  const size_t KVD = engine->kv_dim;

  /* Heap-allocate temporary buffers so stack usage stays tiny and
   * failure leaves cache and caller outputs atomically untouched. */
  float *x = (float *)calloc(tq * D, sizeof(float));
  float *norm_buf = (float *)calloc(tq * D, sizeof(float));
  float *q_buf = (float *)calloc(tq * D, sizeof(float));
  float *k_tok = (float *)calloc(tq * KVD, sizeof(float));
  float *v_tok = (float *)calloc(tq * KVD, sizeof(float));
  float *attn_out = (float *)calloc(tq * D, sizeof(float));
  float *proj_buf = (float *)calloc(tq * D, sizeof(float));
  float *res_buf = (float *)calloc(tq * D, sizeof(float));
  float *gate_buf = (float *)calloc(tq * F, sizeof(float));
  float *up_buf = (float *)calloc(tq * F, sizeof(float));
  float *glu_buf = (float *)calloc(tq * F, sizeof(float));
  float *logits_buf = (float *)calloc(tq * V, sizeof(float));
  float *stage_k = (float *)calloc(L * Hkv * tq * Dh, sizeof(float));
  float *stage_v = (float *)calloc(L * Hkv * tq * Dh, sizeof(float));

  et_mod5_gen_status status = ET_MOD5_GEN_OK;
  if (!x || !norm_buf || !q_buf || !k_tok || !v_tok || !attn_out || !proj_buf ||
      !res_buf || !gate_buf || !up_buf || !glu_buf || !logits_buf || !stage_k ||
      !stage_v) {
    status = ET_MOD5_GEN_ERR_NOMEM;
    goto cleanup;
  }

  /* Token embedding lookup: x[t, :] = token_embedding[tokens[t], :] */
  for (size_t ti = 0u; ti < tq; ++ti) {
    const size_t tok = (size_t)tokens[ti];
    memcpy(x + ti * D, engine->token_embedding + tok * D, D * sizeof(float));
  }

  for (size_t l = 0u; l < L; ++l) {
    const float *n1 = engine->norm1 + l * D;
    const float *wq = engine->wq + l * D * D;
    const float *wk = engine->wk + l * KVD * D;
    const float *wv = engine->wv + l * KVD * D;
    const float *wo = engine->wo + l * D * D;
    const float *n2 = engine->norm2 + l * D;
    const float *wg = engine->wgate + l * F * D;
    const float *wu = engine->wup + l * F * D;
    const float *wd = engine->wdown + l * D * F;

    if (!run_rmsnorm(engine, x, n1, tq, D, norm_buf) ||
        !serial_linear_f32(norm_buf, wq, tq, D, D, q_buf) ||
        !serial_linear_f32(norm_buf, wk, tq, D, KVD, k_tok) ||
        !serial_linear_f32(norm_buf, wv, tq, D, KVD, v_tok)) {
      status = ET_MOD5_GEN_ERR_NONFINITE;
      goto cleanup;
    }

    /* Pack new K, V from [tq, Hkv * Dh] -> stage [Hkv, tq, Dh]. */
    float *layer_stage_k = stage_k + l * Hkv * tq * Dh;
    float *layer_stage_v = stage_v + l * Hkv * tq * Dh;
    for (size_t h = 0u; h < Hkv; ++h) {
      for (size_t ti = 0u; ti < tq; ++ti) {
        memcpy(layer_stage_k + (h * tq + ti) * Dh,
               k_tok + (ti * Hkv + h) * Dh, Dh * sizeof(float));
        memcpy(layer_stage_v + (h * tq + ti) * Dh,
               v_tok + (ti * Hkv + h) * Dh, Dh * sizeof(float));
      }
    }

    /* Dispatch through K1 MOD5 provider when full-prefix exact shape matches. */
    if (start_pos == 0u && engine->k1_runtime != NULL &&
        is_mod5_attention_row(Hq, Hkv, tq, Dh)) {
      float q_htd[4 * 16 * 8];
      float out_htd[4 * 16 * 8];
      int64_t qpos[16];
      int64_t kpos[16];
      uint8_t mask[16 * 16];
      float slopes_buf[ET_MOD5_GEN_MAX_HEADS];
      int64_t win = engine->cfg.window_size > 0u
                        ? (int64_t)engine->cfg.window_size
                        : (int64_t)engine->cfg.max_seq;
      for (size_t h = 0u; h < Hq; ++h) {
        slopes_buf[h] =
            engine->cfg.use_alibi ? engine->cfg.alibi_slopes[h] : 0.0f;
        for (size_t ti = 0u; ti < tq; ++ti) {
          memcpy(q_htd + (h * tq + ti) * Dh, q_buf + (ti * Hq + h) * Dh,
                 Dh * sizeof(float));
        }
      }
      for (size_t ti = 0u; ti < tq; ++ti) {
        qpos[ti] = (int64_t)ti;
        kpos[ti] = (int64_t)ti;
      }
      memset(mask, 1, tq * tq);
      if (!dispatch_mod5_attention(engine->k1_runtime, Hq, Hkv, tq, Dh, q_htd,
                                   layer_stage_k, layer_stage_v, qpos, kpos,
                                   mask, slopes_buf, &win, out_htd)) {
        status = ET_MOD5_GEN_ERR_NONFINITE;
        goto cleanup;
      }
      for (size_t h = 0u; h < Hq; ++h) {
        for (size_t ti = 0u; ti < tq; ++ti) {
          memcpy(attn_out + (ti * Hq + h) * Dh, out_htd + (h * tq + ti) * Dh,
                 Dh * sizeof(float));
        }
      }
    } else {
      const float *past_k =
          cache != NULL ? cache->k_data + l * Hkv * cache->max_seq * Dh : NULL;
      const float *past_v =
          cache != NULL ? cache->v_data + l * Hkv * cache->max_seq * Dh : NULL;
      const uint32_t past_stride = cache != NULL ? cache->max_seq : 0u;
      if (!serial_kv_attention_f32(
              q_buf, past_k, past_v, past_stride, layer_stage_k, layer_stage_v,
              start_pos, tq, Hq, Hkv, Dh, engine->cfg.window_size,
              engine->cfg.use_alibi, engine->cfg.alibi_slopes, attn_out)) {
        status = ET_MOD5_GEN_ERR_NONFINITE;
        goto cleanup;
      }
    }

    if (!serial_linear_f32(attn_out, wo, tq, D, D, proj_buf) ||
        !serial_add_f32(x, proj_buf, tq * D, res_buf) ||
        !run_rmsnorm(engine, res_buf, n2, tq, D, norm_buf) ||
        !serial_linear_f32(norm_buf, wg, tq, D, F, gate_buf) ||
        !serial_linear_f32(norm_buf, wu, tq, D, F, up_buf) ||
        !run_swiglu(engine, gate_buf, up_buf, tq, F, glu_buf) ||
        !serial_linear_f32(glu_buf, wd, tq, F, D, proj_buf) ||
        !serial_add_f32(res_buf, proj_buf, tq * D, x)) {
      status = ET_MOD5_GEN_ERR_NONFINITE;
      goto cleanup;
    }
  }

  if (!run_rmsnorm(engine, x, engine->norm_final, tq, D, norm_buf) ||
      !serial_linear_f32(norm_buf, engine->head, tq, D, V, logits_buf)) {
    status = ET_MOD5_GEN_ERR_NONFINITE;
    goto cleanup;
  }

  /* All layers and final logits succeeded and are finite; commit atomically. */
  if (cache != NULL) {
    for (size_t l = 0u; l < L; ++l) {
      const float *src_k = stage_k + l * Hkv * tq * Dh;
      const float *src_v = stage_v + l * Hkv * tq * Dh;
      for (size_t h = 0u; h < Hkv; ++h) {
        const size_t dst_off =
            ((l * Hkv + h) * cache->max_seq + start_pos) * Dh;
        memcpy(cache->k_data + dst_off, src_k + h * tq * Dh,
               tq * Dh * sizeof(float));
        memcpy(cache->v_data + dst_off, src_v + h * tq * Dh,
               tq * Dh * sizeof(float));
      }
    }
    cache->seq_len = (uint32_t)(start_pos + tq);
  }

  if (out_all_logits != NULL)
    memcpy(out_all_logits, logits_buf, tq * V * sizeof(float));
  if (out_last_logits != NULL)
    memcpy(out_last_logits, logits_buf + (tq - 1u) * V, V * sizeof(float));

cleanup:
  free(x);
  free(norm_buf);
  free(q_buf);
  free(k_tok);
  free(v_tok);
  free(attn_out);
  free(proj_buf);
  free(res_buf);
  free(gate_buf);
  free(up_buf);
  free(glu_buf);
  free(logits_buf);
  free(stage_k);
  free(stage_v);
  return status;
}

et_mod5_gen_status et_mod5_gen_prefill(const et_mod5_gen_engine *engine,
                                       et_mod5_gen_kv_cache *cache,
                                       const int64_t *prompt_tokens,
                                       size_t prompt_len,
                                       float *out_last_logits,
                                       float *out_all_logits) {
  if (cache == NULL || out_last_logits == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  return forward_internal(engine, cache, prompt_tokens, prompt_len,
                          out_last_logits, out_all_logits);
}

et_mod5_gen_status et_mod5_gen_decode_step(const et_mod5_gen_engine *engine,
                                           et_mod5_gen_kv_cache *cache,
                                           int64_t token_id,
                                           float *out_logits) {
  if (cache == NULL || out_logits == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  return forward_internal(engine, cache, &token_id, 1u, out_logits, NULL);
}

et_mod5_gen_status et_mod5_gen_recompute_forward(
    const et_mod5_gen_engine *engine, const int64_t *tokens, size_t seq_len,
    float *out_all_logits) {
  if (out_all_logits == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  return forward_internal(engine, NULL, tokens, seq_len, NULL, out_all_logits);
}

/* ---------------- Deterministic Sampling ---------------- */

typedef struct {
  double prob;
  uint32_t id;
} sample_candidate;

/* Strict total order: higher prob is better; tie breaks to lower token ID. */
static int candidate_worse(sample_candidate a, sample_candidate b) {
  return (a.prob < b.prob) || (a.prob == b.prob && a.id > b.id);
}

/* Deterministic in-place heapsort placing best candidate at index 0. */
static void sort_candidates_desc(sample_candidate *arr, size_t n) {
  if (n <= 1u)
    return;
  /* Build max-heap under candidate_worse so worst item sits at root. */
  for (size_t i = n / 2u; i > 0u; --i) {
    size_t root = i - 1u;
    for (;;) {
      size_t left = root * 2u + 1u;
      if (left >= n)
        break;
      size_t worst_child = left;
      size_t right = left + 1u;
      if (right < n && candidate_worse(arr[right], arr[left]))
        worst_child = right;
      if (candidate_worse(arr[worst_child], arr[root])) {
        const sample_candidate tmp = arr[root];
        arr[root] = arr[worst_child];
        arr[worst_child] = tmp;
        root = worst_child;
      } else {
        break;
      }
    }
  }
  /* Extract worst items to the end -> ascending by worst == descending by best. */
  for (size_t end = n - 1u; end > 0u; --end) {
    const sample_candidate tmp = arr[0];
    arr[0] = arr[end];
    arr[end] = tmp;
    size_t root = 0u;
    for (;;) {
      size_t left = root * 2u + 1u;
      if (left >= end)
        break;
      size_t worst_child = left;
      size_t right = left + 1u;
      if (right < end && candidate_worse(arr[right], arr[left]))
        worst_child = right;
      if (candidate_worse(arr[worst_child], arr[root])) {
        const sample_candidate swap_item = arr[root];
        arr[root] = arr[worst_child];
        arr[worst_child] = swap_item;
        root = worst_child;
      } else {
        break;
      }
    }
  }
}

static et_mod5_gen_status validate_sampling_config(
    const et_mod5_gen_sampling_config *sampling, size_t vocab_size) {
  if (sampling == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(sampling, _Alignof(et_mod5_gen_sampling_config)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (vocab_size == 0u || vocab_size > ET_MOD5_GEN_MAX_VOCAB)
    return ET_MOD5_GEN_ERR_BOUNDS;
  if (sampling->mode != ET_MOD5_GEN_SAMPLE_GREEDY &&
      sampling->mode != ET_MOD5_GEN_SAMPLE_CATEGORICAL)
    return ET_MOD5_GEN_ERR_CONFIG;
  if (sampling->eos_token_id < -1 ||
      (sampling->eos_token_id >= 0 &&
       (uint64_t)sampling->eos_token_id >= vocab_size))
    return ET_MOD5_GEN_ERR_CONFIG;
  if (sampling->mode == ET_MOD5_GEN_SAMPLE_CATEGORICAL) {
    if (!isfinite(sampling->temperature) || !(sampling->temperature > 0.0f))
      return ET_MOD5_GEN_ERR_CONFIG;
    if (sampling->top_k > vocab_size)
      return ET_MOD5_GEN_ERR_CONFIG;
    if (!isfinite(sampling->top_p) || !(sampling->top_p > 0.0f) ||
        sampling->top_p > 1.0f)
      return ET_MOD5_GEN_ERR_CONFIG;
  }
  return ET_MOD5_GEN_OK;
}

et_mod5_gen_status et_mod5_gen_sample(
    const float *logits, size_t vocab_size,
    const et_mod5_gen_sampling_config *sampling, et_mod5_gen_rng *rng,
    int64_t *out_token) {
  if (logits == NULL || sampling == NULL || out_token == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(logits, _Alignof(float)) ||
      !ptr_aligned(out_token, _Alignof(int64_t)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;

  const et_mod5_gen_status cfg_st =
      validate_sampling_config(sampling, vocab_size);
  if (cfg_st != ET_MOD5_GEN_OK)
    return cfg_st;

  if (!all_finite_f32(logits, vocab_size))
    return ET_MOD5_GEN_ERR_NONFINITE;

  if (sampling->mode == ET_MOD5_GEN_SAMPLE_GREEDY) {
    if (rng != NULL && !ptr_aligned(rng, _Alignof(et_mod5_gen_rng)))
      return ET_MOD5_GEN_ERR_UNALIGNED;
    float best_val = logits[0];
    size_t best_id = 0u;
    for (size_t i = 1u; i < vocab_size; ++i) {
      if (logits[i] > best_val) {
        best_val = logits[i];
        best_id = i;
      }
    }
    *out_token = (int64_t)best_id;
    return ET_MOD5_GEN_OK;
  }

  /* Categorical sampling with temperature, top-k, and top-p (nucleus). */
  if (rng == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(rng, _Alignof(et_mod5_gen_rng)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (rng->counter == UINT64_MAX)
    return ET_MOD5_GEN_ERR_OVERFLOW;

  float max_logit = logits[0];
  for (size_t i = 1u; i < vocab_size; ++i) {
    if (logits[i] > max_logit)
      max_logit = logits[i];
  }

  sample_candidate cands[ET_MOD5_GEN_MAX_VOCAB];
  const double inv_temp_denom = (double)sampling->temperature;
  double sum_exp = 0.0;
  for (size_t i = 0u; i < vocab_size; ++i) {
    const double scaled = ((double)logits[i] - (double)max_logit) / inv_temp_denom;
    if (!isfinite(scaled))
      return ET_MOD5_GEN_ERR_NONFINITE;
    const double w = exp(scaled);
    if (!isfinite(w))
      return ET_MOD5_GEN_ERR_NONFINITE;
    sum_exp += w;
    if (!isfinite(sum_exp))
      return ET_MOD5_GEN_ERR_NONFINITE;
    cands[i].prob = w;
    cands[i].id = (uint32_t)i;
  }
  if (!(sum_exp > 0.0) || !isfinite(sum_exp))
    return ET_MOD5_GEN_ERR_NONFINITE;

  for (size_t i = 0u; i < vocab_size; ++i) {
    cands[i].prob /= sum_exp;
    if (!isfinite(cands[i].prob))
      return ET_MOD5_GEN_ERR_NONFINITE;
  }

  sort_candidates_desc(cands, vocab_size);

  /* Top-k truncation and renormalization. */
  const size_t k_limit =
      (sampling->top_k == 0u) ? vocab_size : (size_t)sampling->top_k;
  double sum_k = 0.0;
  for (size_t i = 0u; i < k_limit; ++i)
    sum_k += cands[i].prob;
  if (!(sum_k > 0.0) || !isfinite(sum_k))
    return ET_MOD5_GEN_ERR_NONFINITE;
  for (size_t i = 0u; i < k_limit; ++i)
    cands[i].prob /= sum_k;

  /* Top-p (nucleus) shortest prefix keeping at least 1 candidate. */
  const double p_thresh = (double)sampling->top_p;
  size_t keep = 0u;
  double cum_p = 0.0;
  while (keep < k_limit) {
    cum_p += cands[keep].prob;
    ++keep;
    if (cum_p >= p_thresh)
      break;
  }
  if (!(cum_p > 0.0) || !isfinite(cum_p))
    return ET_MOD5_GEN_ERR_NONFINITE;

  /* Consume one deterministic RNG draw and sample from the nucleus. */
  double unit = 0.0;
  const et_mod5_gen_status rng_st = et_mod5_gen_rng_next_f64(rng, &unit);
  if (rng_st != ET_MOD5_GEN_OK)
    return rng_st;

  const double target = unit * cum_p;
  double running = 0.0;
  size_t chosen = keep - 1u;
  for (size_t i = 0u; i < keep; ++i) {
    running += cands[i].prob;
    if (target < running) {
      chosen = i;
      break;
    }
  }
  *out_token = (int64_t)cands[chosen].id;
  return ET_MOD5_GEN_OK;
}

/* ---------------- End-to-End Autoregressive Generation ---------------- */

et_mod5_gen_status et_mod5_gen_generate(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, size_t prompt_len,
    const et_mod5_gen_sampling_config *sampling, et_mod5_gen_rng *rng,
    int64_t *out_generated_tokens, size_t out_token_capacity,
    float *out_step_logits, size_t out_logits_capacity,
    et_mod5_gen_result *out_result) {
  if (engine == NULL || cache == NULL || prompt_tokens == NULL ||
      sampling == NULL || out_result == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(cache, _Alignof(et_mod5_gen_kv_cache)) ||
      !ptr_aligned(prompt_tokens, _Alignof(int64_t)) ||
      !ptr_aligned(out_result, _Alignof(et_mod5_gen_result)) ||
      (out_generated_tokens != NULL &&
       !ptr_aligned(out_generated_tokens, _Alignof(int64_t))) ||
      (out_step_logits != NULL &&
       !ptr_aligned(out_step_logits, _Alignof(float))))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;
  if (!engine_weights_ready(engine))
    return ET_MOD5_GEN_ERR_STATE;
  if (!cache_matches_engine(engine, cache))
    return ET_MOD5_GEN_ERR_CONFIG;

  const size_t V = engine->cfg.vocab_size;
  const et_mod5_gen_status s_st = validate_sampling_config(sampling, V);
  if (s_st != ET_MOD5_GEN_OK)
    return s_st;

  if (sampling->mode == ET_MOD5_GEN_SAMPLE_CATEGORICAL) {
    if (rng == NULL)
      return ET_MOD5_GEN_ERR_ARGUMENT;
    if (!ptr_aligned(rng, _Alignof(et_mod5_gen_rng)))
      return ET_MOD5_GEN_ERR_UNALIGNED;
    if (sampling->max_new_tokens > 0u &&
        rng->counter > UINT64_MAX - (uint64_t)sampling->max_new_tokens)
      return ET_MOD5_GEN_ERR_OVERFLOW;
  } else if (rng != NULL && !ptr_aligned(rng, _Alignof(et_mod5_gen_rng))) {
    return ET_MOD5_GEN_ERR_UNALIGNED;
  }

  if (prompt_len == 0u)
    return ET_MOD5_GEN_ERR_BOUNDS;
  if (prompt_len > engine->cfg.max_seq ||
      sampling->max_new_tokens > engine->cfg.max_seq - prompt_len)
    return ET_MOD5_GEN_ERR_CACHE_FULL;

  for (size_t i = 0u; i < prompt_len; ++i) {
    if (prompt_tokens[i] < 0 || (uint64_t)prompt_tokens[i] >= V)
      return ET_MOD5_GEN_ERR_BOUNDS;
  }

  if (sampling->max_new_tokens > 0u) {
    if (out_generated_tokens == NULL ||
        out_token_capacity < sampling->max_new_tokens)
      return ET_MOD5_GEN_ERR_BOUNDS;
    if (out_step_logits != NULL) {
      size_t needed_logits = 0u;
      if (!checked_mul_size(sampling->max_new_tokens, V, &needed_logits) ||
          out_logits_capacity < needed_logits)
        return ET_MOD5_GEN_ERR_BOUNDS;
    }
  }

  et_mod5_gen_status st = et_mod5_gen_kv_cache_reset(cache);
  if (st != ET_MOD5_GEN_OK)
    return st;

  float step_logits[ET_MOD5_GEN_MAX_VOCAB];
  st = et_mod5_gen_prefill(engine, cache, prompt_tokens, prompt_len,
                           step_logits, NULL);
  if (st != ET_MOD5_GEN_OK)
    return st;

  et_mod5_gen_rng local_rng = {0u, 0u};
  if (rng != NULL)
    local_rng = *rng;

  uint32_t generated = 0u;
  uint32_t stopped_eos = 0u;

  for (uint32_t step = 0u; step < sampling->max_new_tokens; ++step) {
    if (out_step_logits != NULL)
      memcpy(out_step_logits + (size_t)step * V, step_logits,
             V * sizeof(float));

    int64_t next_tok = -1;
    st = et_mod5_gen_sample(
        step_logits, V, sampling,
        sampling->mode == ET_MOD5_GEN_SAMPLE_CATEGORICAL ? &local_rng : NULL,
        &next_tok);
    if (st != ET_MOD5_GEN_OK)
      return st;

    out_generated_tokens[step] = next_tok;
    ++generated;

    if (sampling->eos_token_id >= 0 && next_tok == sampling->eos_token_id) {
      stopped_eos = 1u;
      break;
    }

    if (step + 1u < sampling->max_new_tokens) {
      st = et_mod5_gen_decode_step(engine, cache, next_tok, step_logits);
      if (st != ET_MOD5_GEN_OK)
        return st;
    }
  }

  if (rng != NULL)
    *rng = local_rng;

  out_result->prompt_len = (uint32_t)prompt_len;
  out_result->generated_len = generated;
  out_result->total_len = (uint32_t)prompt_len + generated;
  out_result->stopped_by_eos = stopped_eos;
  out_result->final_rng = local_rng;
  return ET_MOD5_GEN_OK;
}

/* ---------------- Flat-Scalar AOT Bridge Layer ---------------- */

static int i64_in_u32_range(int64_t v, uint32_t *out) {
  if (v < 0 || v > (int64_t)UINT32_MAX)
    return 0;
  *out = (uint32_t)v;
  return 1;
}

int32_t et_mod5_gen_aot_engine_create(
    int64_t vocab_size, int64_t max_seq, int64_t num_layers,
    int64_t hidden_dim, int64_t ffn_dim, int64_t num_heads,
    int64_t num_kv_heads, int64_t window_size, int64_t use_alibi,
    double rms_epsilon, const float *alibi_slopes,
    et_mod5_gen_engine **out_engine) {
  if (out_engine == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(out_engine, _Alignof(et_mod5_gen_engine *)))
    return ET_MOD5_GEN_ERR_UNALIGNED;

  et_mod5_gen_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  if (!i64_in_u32_range(vocab_size, &cfg.vocab_size) ||
      !i64_in_u32_range(max_seq, &cfg.max_seq) ||
      !i64_in_u32_range(num_layers, &cfg.num_layers) ||
      !i64_in_u32_range(hidden_dim, &cfg.hidden_dim) ||
      !i64_in_u32_range(ffn_dim, &cfg.ffn_dim) ||
      !i64_in_u32_range(num_heads, &cfg.num_heads) ||
      !i64_in_u32_range(num_kv_heads, &cfg.num_kv_heads) ||
      !i64_in_u32_range(window_size, &cfg.window_size) ||
      !i64_in_u32_range(use_alibi, &cfg.use_alibi))
    return ET_MOD5_GEN_ERR_CONFIG;

  if (!f64_to_finite_f32(rms_epsilon, &cfg.rms_epsilon))
    return ET_MOD5_GEN_ERR_CONFIG;

  if (cfg.use_alibi == 1u) {
    if (alibi_slopes == NULL)
      return ET_MOD5_GEN_ERR_ARGUMENT;
    if (!ptr_aligned(alibi_slopes, _Alignof(float)))
      return ET_MOD5_GEN_ERR_UNALIGNED;
    if (cfg.num_heads == 0u || cfg.num_heads > ET_MOD5_GEN_MAX_HEADS)
      return ET_MOD5_GEN_ERR_CONFIG;
    for (uint32_t h = 0u; h < cfg.num_heads; ++h)
      cfg.alibi_slopes[h] = alibi_slopes[h];
  }

  return et_mod5_gen_engine_create(&cfg, out_engine);
}

void et_mod5_gen_aot_engine_destroy(et_mod5_gen_engine *engine) {
  et_mod5_gen_engine_destroy(engine);
}

int32_t et_mod5_gen_aot_set_global_weights(et_mod5_gen_engine *engine,
                                           const float *token_embedding,
                                           const float *norm_final,
                                           const float *head) {
  return et_mod5_gen_engine_set_global_weights(engine, token_embedding,
                                               norm_final, head);
}

int32_t et_mod5_gen_aot_set_layer_weights(
    et_mod5_gen_engine *engine, int64_t layer_index, const float *norm1,
    const float *wq, const float *wk, const float *wv, const float *wo,
    const float *norm2, const float *wgate, const float *wup,
    const float *wdown) {
  uint32_t l_u32 = 0u;
  if (!i64_in_u32_range(layer_index, &l_u32))
    return ET_MOD5_GEN_ERR_BOUNDS;
  return et_mod5_gen_engine_set_layer_weights(
      engine, l_u32, norm1, wq, wk, wv, wo, norm2, wgate, wup, wdown);
}

int32_t et_mod5_gen_aot_kv_cache_create(const et_mod5_gen_engine *engine,
                                        et_mod5_gen_kv_cache **out_cache) {
  if (engine == NULL || out_cache == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(out_cache, _Alignof(et_mod5_gen_kv_cache *)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  return et_mod5_gen_kv_cache_create(&engine->cfg, out_cache);
}

void et_mod5_gen_aot_kv_cache_destroy(et_mod5_gen_kv_cache *cache) {
  et_mod5_gen_kv_cache_destroy(cache);
}

int32_t et_mod5_gen_aot_kv_cache_reset(et_mod5_gen_kv_cache *cache) {
  return et_mod5_gen_kv_cache_reset(cache);
}

int64_t et_mod5_gen_aot_kv_cache_len(const et_mod5_gen_kv_cache *cache) {
  uint32_t len = 0u;
  if (et_mod5_gen_kv_cache_len(cache, &len) != ET_MOD5_GEN_OK)
    return -1;
  return (int64_t)len;
}

int32_t et_mod5_gen_aot_prefill(const et_mod5_gen_engine *engine,
                                et_mod5_gen_kv_cache *cache,
                                const int64_t *prompt_tokens,
                                int64_t prompt_len, float *out_last_logits,
                                float *out_all_logits) {
  if (prompt_len <= 0)
    return ET_MOD5_GEN_ERR_BOUNDS;
  return et_mod5_gen_prefill(engine, cache, prompt_tokens, (size_t)prompt_len,
                             out_last_logits, out_all_logits);
}

int32_t et_mod5_gen_aot_decode_step(const et_mod5_gen_engine *engine,
                                    et_mod5_gen_kv_cache *cache,
                                    int64_t token_id, float *out_logits) {
  return et_mod5_gen_decode_step(engine, cache, token_id, out_logits);
}

int32_t et_mod5_gen_aot_recompute_forward(const et_mod5_gen_engine *engine,
                                          const int64_t *tokens,
                                          int64_t seq_len,
                                          float *out_all_logits) {
  if (seq_len <= 0)
    return ET_MOD5_GEN_ERR_BOUNDS;
  return et_mod5_gen_recompute_forward(engine, tokens, (size_t)seq_len,
                                       out_all_logits);
}

int32_t et_mod5_gen_aot_generate(
    const et_mod5_gen_engine *engine, et_mod5_gen_kv_cache *cache,
    const int64_t *prompt_tokens, int64_t prompt_len, int64_t mode,
    double temperature, int64_t top_k, double top_p, int64_t eos_token_id,
    int64_t max_new_tokens, int64_t rng_seed, int64_t *out_tokens,
    float *out_step_logits, int64_t *out_summary) {
  if (engine == NULL || out_summary == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(engine, _Alignof(et_mod5_gen_engine)) ||
      !ptr_aligned(out_summary, _Alignof(int64_t)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (prompt_len <= 0 || max_new_tokens < 0)
    return ET_MOD5_GEN_ERR_BOUNDS;

  et_mod5_gen_sampling_config sc;
  memset(&sc, 0, sizeof(sc));
  if (!i64_in_u32_range(mode, &sc.mode) ||
      !i64_in_u32_range(top_k, &sc.top_k) ||
      !i64_in_u32_range(max_new_tokens, &sc.max_new_tokens))
    return ET_MOD5_GEN_ERR_CONFIG;

  if (sc.mode == ET_MOD5_GEN_SAMPLE_CATEGORICAL) {
    if (!f64_to_finite_f32(temperature, &sc.temperature) ||
        !f64_to_finite_f32(top_p, &sc.top_p))
      return ET_MOD5_GEN_ERR_CONFIG;
  } else {
    sc.temperature = 1.0f;
    sc.top_p = 1.0f;
  }
  sc.eos_token_id = eos_token_id;

  et_mod5_gen_rng rng;
  const et_mod5_gen_status rng_st =
      et_mod5_gen_rng_init(&rng, (uint64_t)rng_seed);
  if (rng_st != ET_MOD5_GEN_OK)
    return rng_st;

  size_t logits_cap = 0u;
  if (out_step_logits != NULL &&
      !checked_mul_size((size_t)sc.max_new_tokens, engine->cfg.vocab_size,
                        &logits_cap))
    return ET_MOD5_GEN_ERR_OVERFLOW;

  et_mod5_gen_result res;
  memset(&res, 0, sizeof(res));
  const et_mod5_gen_status st = et_mod5_gen_generate(
      engine, cache, prompt_tokens, (size_t)prompt_len, &sc, &rng, out_tokens,
      (size_t)sc.max_new_tokens, out_step_logits, logits_cap, &res);
  if (st != ET_MOD5_GEN_OK)
    return st;

  out_summary[ET_MOD5_GEN_AOT_SUMMARY_GENERATED_LEN] =
      (int64_t)res.generated_len;
  out_summary[ET_MOD5_GEN_AOT_SUMMARY_STOPPED_EOS] =
      (int64_t)res.stopped_by_eos;
  out_summary[ET_MOD5_GEN_AOT_SUMMARY_RNG_KEY] = (int64_t)res.final_rng.key;
  out_summary[ET_MOD5_GEN_AOT_SUMMARY_RNG_COUNTER] =
      (int64_t)res.final_rng.counter;
  out_summary[ET_MOD5_GEN_AOT_SUMMARY_CACHE_LEN] = (int64_t)cache->seq_len;
  return ET_MOD5_GEN_OK;
}

int32_t et_mod5_gen_aot_verify_provider(double *out_max_abs_diff) {
  return et_mod5_gen_verify_provider(out_max_abs_diff);
}

int32_t et_mod5_gen_aot_max_abs_diff_f32(const float *a, const float *b,
                                         int64_t count, double *out_max_diff) {
  if (a == NULL || b == NULL || out_max_diff == NULL)
    return ET_MOD5_GEN_ERR_ARGUMENT;
  if (!ptr_aligned(a, _Alignof(float)) || !ptr_aligned(b, _Alignof(float)) ||
      !ptr_aligned(out_max_diff, _Alignof(double)))
    return ET_MOD5_GEN_ERR_UNALIGNED;
  if (count <= 0)
    return ET_MOD5_GEN_ERR_BOUNDS;
  if (!fp_env_supported())
    return ET_MOD5_GEN_ERR_FP_ENV;

  double worst = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    if (!isfinite(a[i]) || !isfinite(b[i]))
      return ET_MOD5_GEN_ERR_NONFINITE;
    const double diff = fabs((double)a[i] - (double)b[i]);
    if (diff > worst)
      worst = diff;
  }
  *out_max_diff = worst;
  return ET_MOD5_GEN_OK;
}

double et_mod5_gen_aot_load_f64(const double *buffer, int64_t index) {
  if (!ptr_aligned(buffer, _Alignof(double)) || index < 0)
    return (double)NAN;
  return buffer[index];
}
