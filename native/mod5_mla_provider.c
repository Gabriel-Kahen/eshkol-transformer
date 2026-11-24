#include "eshkol_transformer/mod5_mla_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 MLA v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MOD5 MLA requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 MLA requires standard IEEE-754 binary32/binary64 eval");

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define MLA_MAX_HD (ET_MOD5_MLA_MAX_HEADS * ET_MOD5_MLA_MAX_HEAD_DIM)

#define MLA_ROW(h_val, dh_min, dh_max, dc_max)                                 \
  {                                                                            \
    6u, (const et_kernel_dimension_range_v1[]) {                               \
      {ET_MOD5_MLA_MIN_BATCH, ET_MOD5_MLA_MAX_BATCH, 0u, {0}},                 \
          {ET_MOD5_MLA_MIN_TOKENS, ET_MOD5_MLA_MAX_TOKENS, 0u, {0}},           \
          {ET_MOD5_MLA_MIN_MODEL_DIM, ET_MOD5_MLA_MAX_MODEL_DIM, 0u, {0}},     \
          {(h_val), (h_val), 0u, {0}}, {(dh_min), (dh_max), 0u, {0}}, {        \
        ET_MOD5_MLA_MIN_LATENT_DIM, (dc_max), 0u, { 0 }                        \
      }                                                                        \
    }                                                                          \
  }

#define IMPLEMENTATION "mod5.mla.cpu-f32.serial"
#define EVIDENCE "MOD5:cpu-f32-mla-v1"

/*
 * 16 pairwise-disjoint K1 shape range rows covering the exact admitted domain:
 *   N in 1..2, T in 1..32, D in 4..32, H in 1..4, Dh in 2..16,
 *   dc in 1..16 with dc < 2 * H * Dh.
 */
static const et_kernel_shape_range_v1 mla_rows[] = {
    /* H = 1 */
    MLA_ROW(1u, 2u, 2u, 3u),   MLA_ROW(1u, 3u, 3u, 5u),
    MLA_ROW(1u, 4u, 4u, 7u),   MLA_ROW(1u, 5u, 5u, 9u),
    MLA_ROW(1u, 6u, 6u, 11u),  MLA_ROW(1u, 7u, 7u, 13u),
    MLA_ROW(1u, 8u, 8u, 15u),  MLA_ROW(1u, 9u, 16u, 16u),
    /* H = 2 */
    MLA_ROW(2u, 2u, 2u, 7u),   MLA_ROW(2u, 3u, 3u, 11u),
    MLA_ROW(2u, 4u, 4u, 15u),  MLA_ROW(2u, 5u, 16u, 16u),
    /* H = 3 */
    MLA_ROW(3u, 2u, 2u, 11u),  MLA_ROW(3u, 3u, 16u, 16u),
    /* H = 4 */
    MLA_ROW(4u, 2u, 2u, 15u),  MLA_ROW(4u, 3u, 16u, 16u)};

static const char *const mla_operations[] = {"mla.forward", "mla.backward"};
static const char *const mla_dtypes[] = {"f32"};
static const char *const mla_devices[] = {"cpu"};

static const et_kernel_capability_v1 capabilities[] = {
    {sizeof(et_kernel_capability_v1), "kernel.mla",
     ET_KERNEL_CAPABILITY_VERIFIED, IMPLEMENTATION, "1.0", EVIDENCE, 1u, {0},
     COUNT(mla_operations), mla_operations, COUNT(mla_dtypes), mla_dtypes,
     COUNT(mla_devices), mla_devices, COUNT(mla_rows), mla_rows}};

typedef struct tensor_table {
  size_t count;
  size_t stride;
  size_t bytes;
  const void *base;
} tensor_table;

typedef struct tensor_schema {
  size_t rank;
  const uint64_t *shape;
} tensor_schema;

typedef struct byte_span {
  const void *data;
  size_t bytes;
} byte_span;

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
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

static int overlap_views(const et_kernel_tensor_view_v1 *a,
                         const et_kernel_tensor_view_v1 *b) {
  return spans_overlap(a->data, a->byte_length, b->data, b->byte_length);
}

static int32_t fail(et_kernel_error *error, et_kernel_error_category category,
                    et_kernel_error_code code, const char *operation,
                    const char *message) {
  if (error != NULL) {
    const int saved_errno = errno;
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
    errno = saved_errno;
  }
  return (int32_t)category;
}

static int32_t rejected(et_kernel_error *error, const char *operation,
                        const char *message) {
  return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
              ET_KERNEL_CODE_PROVIDER_REJECTED, operation, message);
}

static tensor_table inputs_table(const et_kernel_call_v1 *call) {
  return (tensor_table){call->input_count, call->input_stride,
                        call->input_bytes, call->inputs};
}

static tensor_table outputs_table(const et_kernel_call_v1 *call) {
  return (tensor_table){call->output_count, call->output_stride,
                        call->output_bytes, call->outputs};
}

static const et_kernel_tensor_view_v1 *view_at(tensor_table table,
                                               size_t index) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)table.base +
                                            index * table.stride);
}

static size_t bounded_text_bytes(const char *text) {
  size_t len = 0u;
  while (len <= ET_KERNEL_MAX_SYMBOL_BYTES && text[len] != '\0')
    ++len;
  return len + 1u;
}

static int shape_admitted(uint64_t n, uint64_t t, uint64_t d, uint64_t h,
                          uint64_t dh, uint64_t dc) {
  return n >= ET_MOD5_MLA_MIN_BATCH && n <= ET_MOD5_MLA_MAX_BATCH &&
         t >= ET_MOD5_MLA_MIN_TOKENS && t <= ET_MOD5_MLA_MAX_TOKENS &&
         d >= ET_MOD5_MLA_MIN_MODEL_DIM && d <= ET_MOD5_MLA_MAX_MODEL_DIM &&
         h >= ET_MOD5_MLA_MIN_HEADS && h <= ET_MOD5_MLA_MAX_HEADS &&
         dh >= ET_MOD5_MLA_MIN_HEAD_DIM && dh <= ET_MOD5_MLA_MAX_HEAD_DIM &&
         dc >= ET_MOD5_MLA_MIN_LATENT_DIM && dc <= ET_MOD5_MLA_MAX_LATENT_DIM &&
         dc < 2u * h * dh;
}

static int fp_environment_supported(void) {
  uint16_t x87;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round-to-nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int all_f32_finite(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static int finite_f64_to_f32(double value, float *out) {
  if (!isfinite(value))
    return 0;
  const float f = (float)value;
  if (!isfinite(f))
    return 0;
  if (out != NULL)
    *out = f;
  return 1;
}

static int causal_lse_softmax(const double *scores, size_t len, double *probs) {
  double max_score = scores[0];
  if (!isfinite(max_score))
    return 0;
  for (size_t tk = 1; tk < len; ++tk) {
    if (!isfinite(scores[tk]))
      return 0;
    if (scores[tk] > max_score)
      max_score = scores[tk];
  }
  double sum_exp = 0.0;
  for (size_t tk = 0; tk < len; ++tk) {
    const double shifted = scores[tk] - max_score;
    if (!isfinite(shifted))
      return 0;
    const double e = exp(shifted);
    if (!isfinite(e))
      return 0;
    sum_exp += e;
    if (!isfinite(sum_exp))
      return 0;
  }
  if (!(sum_exp > 0.0))
    return 0;
  const double lse = max_score + log(sum_exp);
  if (!isfinite(lse))
    return 0;
  for (size_t tk = 0; tk < len; ++tk) {
    const double shifted = scores[tk] - lse;
    if (!isfinite(shifted))
      return 0;
    const double p = exp(shifted);
    if (!isfinite(p))
      return 0;
    probs[tk] = p;
  }
  return 1;
}

/*
 * Forward workspace for one batch slice (allocated on stack, zero heap usage).
 */
typedef struct mla_fwd_ws {
  double c_kv[ET_MOD5_MLA_MAX_TOKENS * ET_MOD5_MLA_MAX_LATENT_DIM];
  double q[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double k[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double v[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double attn[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double scores[ET_MOD5_MLA_MAX_TOKENS];
  double probs[ET_MOD5_MLA_MAX_TOKENS];
} mla_fwd_ws;

/*
 * Backward workspace (allocated on stack, zero heap usage).
 */
typedef struct mla_bwd_ws {
  mla_fwd_ws fwd;
  double d_attn[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double d_q[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double d_k[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double d_v[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double d_c_kv[ET_MOD5_MLA_MAX_TOKENS * ET_MOD5_MLA_MAX_LATENT_DIM];
  double dp[ET_MOD5_MLA_MAX_TOKENS];
  double dw_q[MLA_MAX_HD * ET_MOD5_MLA_MAX_MODEL_DIM];
  double dw_dkv[ET_MOD5_MLA_MAX_LATENT_DIM * ET_MOD5_MLA_MAX_MODEL_DIM];
  double dw_uk[MLA_MAX_HD * ET_MOD5_MLA_MAX_LATENT_DIM];
  double dw_uv[MLA_MAX_HD * ET_MOD5_MLA_MAX_LATENT_DIM];
  double dw_o[ET_MOD5_MLA_MAX_MODEL_DIM * MLA_MAX_HD];
} mla_bwd_ws;

static int mla_forward_batch(const et_mod5_mla_dims *dims, size_t n,
                             const float *x, const float *w_q,
                             const float *w_dkv, const float *w_uk,
                             const float *w_uv, const float *w_o,
                             int quantize_c_kv, mla_fwd_ws *ws,
                             float *c_kv_dst_batch, float *out_batch) {
  const size_t T = dims->tokens;
  const size_t D = dims->model_dim;
  const size_t H = dims->heads;
  const size_t Dh = dims->head_dim;
  const size_t dc = dims->latent_dim;
  const size_t HD = H * Dh;
  const double scale = 1.0 / sqrt((double)Dh);
  if (!isfinite(scale))
    return 0;

  const float *xn = x + n * T * D;

  /* 1. Down-project X -> C_kv [T, dc] and project X -> Q [T, HD]. */
  for (size_t t = 0; t < T; ++t) {
    const float *xt = xn + t * D;
    for (size_t r = 0; r < dc; ++r) {
      const float *wr = w_dkv + r * D;
      double acc = 0.0;
      for (size_t d = 0; d < D; ++d) {
        acc += (double)xt[d] * (double)wr[d];
      }
      if (!isfinite(acc))
        return 0;
      if (quantize_c_kv) {
        float cf = 0.0f;
        if (!finite_f64_to_f32(acc, &cf))
          return 0;
        ws->c_kv[t * dc + r] = (double)cf;
        if (c_kv_dst_batch != NULL)
          c_kv_dst_batch[t * dc + r] = cf;
      } else {
        ws->c_kv[t * dc + r] = acc;
      }
    }
    for (size_t i = 0; i < HD; ++i) {
      const float *wi = w_q + i * D;
      double acc = 0.0;
      for (size_t d = 0; d < D; ++d) {
        acc += (double)xt[d] * (double)wi[d];
      }
      if (!isfinite(acc))
        return 0;
      ws->q[t * HD + i] = acc;
    }
  }

  /* 2. Up-project C_kv -> K [T, HD] and V [T, HD]. */
  for (size_t t = 0; t < T; ++t) {
    const double *ct = ws->c_kv + t * dc;
    for (size_t i = 0; i < HD; ++i) {
      const float *wuk_i = w_uk + i * dc;
      const float *wuv_i = w_uv + i * dc;
      double ak = 0.0;
      double av = 0.0;
      for (size_t r = 0; r < dc; ++r) {
        ak += ct[r] * (double)wuk_i[r];
        av += ct[r] * (double)wuv_i[r];
      }
      if (!isfinite(ak) || !isfinite(av))
        return 0;
      ws->k[t * HD + i] = ak;
      ws->v[t * HD + i] = av;
    }
  }

  /* 3. Exact binary64 log-sum-exp causal scaled dot-product attention. */
  for (size_t h = 0; h < H; ++h) {
    const size_t base_h = h * Dh;
    for (size_t tq = 0; tq < T; ++tq) {
      const double *q_row = ws->q + tq * HD + base_h;
      for (size_t tk = 0; tk <= tq; ++tk) {
        const double *k_row = ws->k + tk * HD + base_h;
        double dot = 0.0;
        for (size_t k = 0; k < Dh; ++k) {
          dot += q_row[k] * k_row[k];
        }
        const double s = dot * scale;
        if (!isfinite(s))
          return 0;
        ws->scores[tk] = s;
      }
      if (!causal_lse_softmax(ws->scores, tq + 1u, ws->probs))
        return 0;
      double *attn_row = ws->attn + tq * HD + base_h;
      for (size_t k = 0; k < Dh; ++k) {
        double acc = 0.0;
        for (size_t tk = 0; tk <= tq; ++tk) {
          acc += ws->probs[tk] * ws->v[tk * HD + base_h + k];
        }
        if (!isfinite(acc))
          return 0;
        attn_row[k] = acc;
      }
    }
  }

  /* 4. Output projection O = Attn @ W_o^T [T, D]. */
  if (w_o != NULL) {
    for (size_t t = 0; t < T; ++t) {
      const double *at = ws->attn + t * HD;
      for (size_t d = 0; d < D; ++d) {
        const float *wd = w_o + d * HD;
        double acc = 0.0;
        for (size_t i = 0; i < HD; ++i) {
          acc += at[i] * (double)wd[i];
        }
        float *dst = out_batch != NULL ? &out_batch[t * D + d] : NULL;
        if (!finite_f64_to_f32(acc, dst))
          return 0;
      }
    }
  }
  return 1;
}

static int mla_evaluate_forward(const et_mod5_mla_dims *dims, const float *x,
                                const float *w_q, const float *w_dkv,
                                const float *w_uk, const float *w_uv,
                                const float *w_o, float *out) {
  const size_t N = dims->batch;
  const size_t T = dims->tokens;
  const size_t D = dims->model_dim;
  const size_t HD = dims->heads * dims->head_dim;
  const size_t dc = dims->latent_dim;

  if (!all_f32_finite(x, N * T * D) || !all_f32_finite(w_q, HD * D) ||
      !all_f32_finite(w_dkv, dc * D) || !all_f32_finite(w_uk, HD * dc) ||
      !all_f32_finite(w_uv, HD * dc) || !all_f32_finite(w_o, D * HD))
    return 0;

  mla_fwd_ws ws;
  for (size_t n = 0; n < N; ++n) {
    float *out_n = out != NULL ? out + n * T * D : NULL;
    if (!mla_forward_batch(dims, n, x, w_q, w_dkv, w_uk, w_uv, w_o, 0, &ws,
                           NULL, out_n))
      return 0;
  }
  return 1;
}

static int mla_evaluate_backward(const et_mod5_mla_dims *dims, const float *x,
                                 const float *w_q, const float *w_dkv,
                                 const float *w_uk, const float *w_uv,
                                 const float *w_o, const float *dy, float *dx,
                                 float *dw_q, float *dw_dkv, float *dw_uk,
                                 float *dw_uv, float *dw_o) {
  const size_t N = dims->batch;
  const size_t T = dims->tokens;
  const size_t D = dims->model_dim;
  const size_t H = dims->heads;
  const size_t Dh = dims->head_dim;
  const size_t dc = dims->latent_dim;
  const size_t HD = H * Dh;
  const double scale = 1.0 / sqrt((double)Dh);
  if (!isfinite(scale))
    return 0;

  if (!all_f32_finite(x, N * T * D) || !all_f32_finite(w_q, HD * D) ||
      !all_f32_finite(w_dkv, dc * D) || !all_f32_finite(w_uk, HD * dc) ||
      !all_f32_finite(w_uv, HD * dc) || !all_f32_finite(w_o, D * HD) ||
      !all_f32_finite(dy, N * T * D))
    return 0;

  mla_bwd_ws ws;
  memset(ws.dw_q, 0, HD * D * sizeof(double));
  memset(ws.dw_dkv, 0, dc * D * sizeof(double));
  memset(ws.dw_uk, 0, HD * dc * sizeof(double));
  memset(ws.dw_uv, 0, HD * dc * sizeof(double));
  memset(ws.dw_o, 0, D * HD * sizeof(double));

  for (size_t n = 0; n < N; ++n) {
    if (!mla_forward_batch(dims, n, x, w_q, w_dkv, w_uk, w_uv, NULL, 0,
                           &ws.fwd, NULL, NULL))
      return 0;

    const float *xn = x + n * T * D;
    const float *dyn = dy + n * T * D;

    /* B1: Backprop through O = Attn @ W_o^T. */
    for (size_t t = 0; t < T; ++t) {
      const float *dyt = dyn + t * D;
      const double *at = ws.fwd.attn + t * HD;
      for (size_t d = 0; d < D; ++d) {
        const double dy_td = (double)dyt[d];
        for (size_t i = 0; i < HD; ++i) {
          ws.dw_o[d * HD + i] += dy_td * at[i];
          if (!isfinite(ws.dw_o[d * HD + i]))
            return 0;
        }
      }
      for (size_t i = 0; i < HD; ++i) {
        double acc = 0.0;
        for (size_t d = 0; d < D; ++d) {
          acc += (double)dyt[d] * (double)w_o[d * HD + i];
        }
        if (!isfinite(acc))
          return 0;
        ws.d_attn[t * HD + i] = acc;
      }
    }

    /* B2: Backprop through causal binary64 log-sum-exp attention. */
    memset(ws.d_q, 0, T * HD * sizeof(double));
    memset(ws.d_k, 0, T * HD * sizeof(double));
    memset(ws.d_v, 0, T * HD * sizeof(double));

    for (size_t h = 0; h < H; ++h) {
      const size_t base_h = h * Dh;
      for (size_t tq = 0; tq < T; ++tq) {
        const double *q_row = ws.fwd.q + tq * HD + base_h;
        const double *da_row = ws.d_attn + tq * HD + base_h;
        for (size_t tk = 0; tk <= tq; ++tk) {
          const double *k_row = ws.fwd.k + tk * HD + base_h;
          double dot = 0.0;
          for (size_t k = 0; k < Dh; ++k) {
            dot += q_row[k] * k_row[k];
          }
          const double s = dot * scale;
          if (!isfinite(s))
            return 0;
          ws.fwd.scores[tk] = s;
        }
        if (!causal_lse_softmax(ws.fwd.scores, tq + 1u, ws.fwd.probs))
          return 0;

        double mean_dp = 0.0;
        for (size_t tk = 0; tk <= tq; ++tk) {
          const double *v_row = ws.fwd.v + tk * HD + base_h;
          double dp_tk = 0.0;
          for (size_t k = 0; k < Dh; ++k) {
            dp_tk += da_row[k] * v_row[k];
          }
          if (!isfinite(dp_tk))
            return 0;
          ws.dp[tk] = dp_tk;
          mean_dp += ws.fwd.probs[tk] * dp_tk;
          if (!isfinite(mean_dp))
            return 0;
        }

        double *dq_row = ws.d_q + tq * HD + base_h;
        for (size_t tk = 0; tk <= tq; ++tk) {
          const double p = ws.fwd.probs[tk];
          const double ds = p * (ws.dp[tk] - mean_dp);
          const double ddot = ds * scale;
          if (!isfinite(ds) || !isfinite(ddot))
            return 0;
          const double *k_row = ws.fwd.k + tk * HD + base_h;
          double *dk_row = ws.d_k + tk * HD + base_h;
          double *dv_row = ws.d_v + tk * HD + base_h;
          for (size_t k = 0; k < Dh; ++k) {
            dq_row[k] += ddot * k_row[k];
            dk_row[k] += ddot * q_row[k];
            dv_row[k] += p * da_row[k];
            if (!isfinite(dq_row[k]) || !isfinite(dk_row[k]) ||
                !isfinite(dv_row[k]))
              return 0;
          }
        }
      }
    }

    /* B3: Backprop through K = C_kv @ W_uk^T and V = C_kv @ W_uv^T. */
    for (size_t t = 0; t < T; ++t) {
      const double *ct = ws.fwd.c_kv + t * dc;
      const double *dkt = ws.d_k + t * HD;
      const double *dvt = ws.d_v + t * HD;
      for (size_t i = 0; i < HD; ++i) {
        const double dki = dkt[i];
        const double dvi = dvt[i];
        for (size_t r = 0; r < dc; ++r) {
          ws.dw_uk[i * dc + r] += dki * ct[r];
          ws.dw_uv[i * dc + r] += dvi * ct[r];
          if (!isfinite(ws.dw_uk[i * dc + r]) ||
              !isfinite(ws.dw_uv[i * dc + r]))
            return 0;
        }
      }
      for (size_t r = 0; r < dc; ++r) {
        double acc = 0.0;
        for (size_t i = 0; i < HD; ++i) {
          acc += dkt[i] * (double)w_uk[i * dc + r] +
                 dvt[i] * (double)w_uv[i * dc + r];
        }
        if (!isfinite(acc))
          return 0;
        ws.d_c_kv[t * dc + r] = acc;
      }
    }

    /* B4: Backprop through Q = X @ W_q^T and C_kv = X @ W_dkv^T. */
    float *dxn = dx != NULL ? dx + n * T * D : NULL;
    for (size_t t = 0; t < T; ++t) {
      const float *xt = xn + t * D;
      const double *dqt = ws.d_q + t * HD;
      const double *dct = ws.d_c_kv + t * dc;
      for (size_t i = 0; i < HD; ++i) {
        const double dqi = dqt[i];
        for (size_t d = 0; d < D; ++d) {
          ws.dw_q[i * D + d] += dqi * (double)xt[d];
          if (!isfinite(ws.dw_q[i * D + d]))
            return 0;
        }
      }
      for (size_t r = 0; r < dc; ++r) {
        const double dcr = dct[r];
        for (size_t d = 0; d < D; ++d) {
          ws.dw_dkv[r * D + d] += dcr * (double)xt[d];
          if (!isfinite(ws.dw_dkv[r * D + d]))
            return 0;
        }
      }
      for (size_t d = 0; d < D; ++d) {
        double acc_q = 0.0;
        for (size_t i = 0; i < HD; ++i) {
          acc_q += dqt[i] * (double)w_q[i * D + d];
        }
        double acc_c = 0.0;
        for (size_t r = 0; r < dc; ++r) {
          acc_c += dct[r] * (double)w_dkv[r * D + d];
        }
        const double total = acc_q + acc_c;
        float *dst = dxn != NULL ? &dxn[t * D + d] : NULL;
        if (!finite_f64_to_f32(total, dst))
          return 0;
      }
    }
  }

  /* Finalize and verify finiteness of all 5 f32 weight gradients. */
  for (size_t idx = 0; idx < HD * D; ++idx) {
    if (!finite_f64_to_f32(ws.dw_q[idx], dw_q != NULL ? &dw_q[idx] : NULL))
      return 0;
  }
  for (size_t idx = 0; idx < dc * D; ++idx) {
    if (!finite_f64_to_f32(ws.dw_dkv[idx],
                           dw_dkv != NULL ? &dw_dkv[idx] : NULL))
      return 0;
  }
  for (size_t idx = 0; idx < HD * dc; ++idx) {
    if (!finite_f64_to_f32(ws.dw_uk[idx], dw_uk != NULL ? &dw_uk[idx] : NULL))
      return 0;
  }
  for (size_t idx = 0; idx < HD * dc; ++idx) {
    if (!finite_f64_to_f32(ws.dw_uv[idx], dw_uv != NULL ? &dw_uv[idx] : NULL))
      return 0;
  }
  for (size_t idx = 0; idx < D * HD; ++idx) {
    if (!finite_f64_to_f32(ws.dw_o[idx], dw_o != NULL ? &dw_o[idx] : NULL))
      return 0;
  }
  return 1;
}

/* ==================== K1 Provider Validation & Dispatch ==================== */

static int output_aliases_metadata(const et_kernel_tensor_view_v1 *view,
                                   const et_kernel_call_v1 *call,
                                   tensor_table in, tensor_table out,
                                   const et_kernel_error *error) {
  const void *data = view->data;
  const size_t bytes = view->byte_length;
  const et_kernel_request_v1 *request = call->request;
  if (spans_overlap(data, bytes, call, call->struct_size) ||
      spans_overlap(data, bytes, request, request->struct_size) ||
      spans_overlap(data, bytes, request->shape,
                    request->rank * sizeof(*request->shape)) ||
      spans_overlap(data, bytes, call->capability,
                    bounded_text_bytes(call->capability)) ||
      spans_overlap(data, bytes, request->operation,
                    bounded_text_bytes(request->operation)) ||
      spans_overlap(data, bytes, request->dtype,
                    bounded_text_bytes(request->dtype)) ||
      spans_overlap(data, bytes, request->device,
                    bounded_text_bytes(request->device)) ||
      spans_overlap(data, bytes, in.base, in.bytes) ||
      spans_overlap(data, bytes, out.base, out.bytes) ||
      (error != NULL && spans_overlap(data, bytes, error, sizeof(*error))))
    return 1;
  for (size_t j = 0; j < in.count; ++j) {
    const et_kernel_tensor_view_v1 *entry = view_at(in, j);
    if ((entry->rank != 0u &&
         spans_overlap(data, bytes, entry->shape,
                       entry->rank * sizeof(*entry->shape))) ||
        spans_overlap(data, bytes, entry->dtype,
                      bounded_text_bytes(entry->dtype)) ||
        spans_overlap(data, bytes, entry->device,
                      bounded_text_bytes(entry->device)))
      return 1;
  }
  for (size_t j = 0; j < out.count; ++j) {
    const et_kernel_tensor_view_v1 *entry = view_at(out, j);
    if ((entry->rank != 0u &&
         spans_overlap(data, bytes, entry->shape,
                       entry->rank * sizeof(*entry->shape))) ||
        spans_overlap(data, bytes, entry->dtype,
                      bounded_text_bytes(entry->dtype)) ||
        spans_overlap(data, bytes, entry->device,
                      bounded_text_bytes(entry->device)))
      return 1;
  }
  return 0;
}

static int32_t validate_f32_view(const et_kernel_tensor_view_v1 *view,
                                 const tensor_schema *schema, size_t stride,
                                 const char *operation,
                                 et_kernel_error *error) {
  size_t bytes = sizeof(float);
  if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      view->struct_size > stride) {
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "invalid view prefix size");
  }
  if (!exact_text(view->dtype, "f32")) {
    return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "view dtype does not match mla schema");
  }
  if (!exact_text(view->device, "cpu")) {
    return fail(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "view must be on CPU");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return fail(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "view must be dense and zero-offset");
  }
  if (view->rank != schema->rank) {
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, operation,
                "view rank does not match mla schema");
  }
  if (schema->rank != 0u &&
      !aligned_span(view->shape, _Alignof(uint64_t),
                    schema->rank * sizeof(*view->shape))) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "invalid view shape storage");
  }
  for (size_t d = 0; d < schema->rank; ++d) {
    if (view->shape[d] != schema->shape[d]) {
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, operation,
                  "view extent does not match mla schema");
    }
    bytes *= (size_t)schema->shape[d];
  }
  if (view->byte_length != bytes) {
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "incorrect view byte length");
  }
  if (!aligned_span(view->data, _Alignof(float), bytes)) {
    return rejected(error, operation, "invalid view data span or alignment");
  }
  return 0;
}

static int32_t validate_table(tensor_table table, size_t count,
                              const tensor_schema *schema,
                              const char *operation, et_kernel_error *error) {
  if (table.count != count || table.base == NULL) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "incorrect tensor table count or null table");
  }
  if (table.stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "truncated tensor table stride");
  }
  if (table.stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      !aligned_span(table.base, _Alignof(et_kernel_tensor_view_v1), 0u)) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "misaligned tensor table");
  }
  if (count > SIZE_MAX / table.stride || table.bytes != count * table.stride ||
      !aligned_span(table.base, _Alignof(et_kernel_tensor_view_v1),
                    table.bytes)) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                "invalid tensor table span");
  }
  for (size_t i = 0; i < count; ++i) {
    const int32_t status = validate_f32_view(view_at(table, i), &schema[i],
                                             table.stride, operation, error);
    if (status != 0)
      return status;
  }
  return 0;
}

static int find_operation(const et_kernel_call_v1 *call, int *backward) {
  if (!exact_text(call->capability, "kernel.mla"))
    return 0;
  if (exact_text(call->request->operation, "mla.forward")) {
    *backward = 0;
    return 1;
  }
  if (exact_text(call->request->operation, "mla.backward")) {
    *backward = 1;
    return 1;
  }
  return 0;
}

static int request_admitted(const et_kernel_request_v1 *request) {
  if (!exact_text(request->dtype, "f32") ||
      !exact_text(request->device, "cpu") || request->rank != 6u ||
      !aligned_span(request->shape, _Alignof(uint64_t),
                    6u * sizeof(*request->shape)))
    return 0;
  return shape_admitted(request->shape[0], request->shape[1], request->shape[2],
                        request->shape[3], request->shape[4],
                        request->shape[5]);
}

static int32_t validate_tables(const et_kernel_call_v1 *call, int backward,
                               et_kernel_error *error) {
  const char *operation = call->request->operation;
  const uint64_t *s = call->request->shape;
  const uint64_t hd = s[3] * s[4];
  const uint64_t x_shape[] = {s[0], s[1], s[2]};
  const uint64_t wq_shape[] = {hd, s[2]};
  const uint64_t wdkv_shape[] = {s[5], s[2]};
  const uint64_t wuk_shape[] = {hd, s[5]};
  const uint64_t wo_shape[] = {s[2], hd};

  const tensor_schema in_schema[] = {{3u, x_shape},    {2u, wq_shape},
                                     {2u, wdkv_shape}, {2u, wuk_shape},
                                     {2u, wuk_shape},  {2u, wo_shape},
                                     {3u, x_shape}};
  const tensor_schema out_schema[] = {{3u, x_shape},    {2u, wq_shape},
                                      {2u, wdkv_shape}, {2u, wuk_shape},
                                      {2u, wuk_shape},  {2u, wo_shape}};

  const tensor_table in = inputs_table(call);
  const tensor_table out = outputs_table(call);
  const size_t expected_in = backward ? 7u : 6u;
  const size_t expected_out = backward ? 6u : 1u;

  int32_t status = validate_table(in, expected_in, in_schema, operation, error);
  if (status != 0)
    return status;
  status = validate_table(out, expected_out, out_schema, operation, error);
  if (status != 0)
    return status;

  /* Pairwise input disjointness. */
  for (size_t i = 0; i < in.count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (overlap_views(view_at(in, i), view_at(in, j)))
        return rejected(error, operation, "input storage overlaps");
    }
  }

  /* Output disjointness against metadata, error, other outputs, and inputs. */
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *ov = view_at(out, i);
    if (output_aliases_metadata(ov, call, in, out, error)) {
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases live K1 metadata");
    }
    for (size_t j = 0; j < i; ++j) {
      if (overlap_views(ov, view_at(out, j))) {
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "outputs overlap each other");
      }
    }
    for (size_t j = 0; j < in.count; ++j) {
      if (overlap_views(ov, view_at(in, j))) {
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases input");
      }
    }
  }
  return 0;
}

static et_mod5_mla_dims dims_from_shape(const uint64_t *s) {
  return (et_mod5_mla_dims){(size_t)s[0], (size_t)s[1], (size_t)s[2],
                            (size_t)s[3], (size_t)s[4], (size_t)s[5]};
}

static int evaluate_k1_call(const et_kernel_call_v1 *call, int backward,
                            int write_outputs) {
  const tensor_table in = inputs_table(call);
  const tensor_table out = outputs_table(call);
  const et_mod5_mla_dims dims = dims_from_shape(call->request->shape);

  const float *x = (const float *)view_at(in, 0u)->data;
  const float *w_q = (const float *)view_at(in, 1u)->data;
  const float *w_dkv = (const float *)view_at(in, 2u)->data;
  const float *w_uk = (const float *)view_at(in, 3u)->data;
  const float *w_uv = (const float *)view_at(in, 4u)->data;
  const float *w_o = (const float *)view_at(in, 5u)->data;

  if (!backward) {
    float *o = write_outputs ? (float *)view_at(out, 0u)->data : NULL;
    return mla_evaluate_forward(&dims, x, w_q, w_dkv, w_uk, w_uv, w_o, o);
  }
  const float *dy = (const float *)view_at(in, 6u)->data;
  float *dx = write_outputs ? (float *)view_at(out, 0u)->data : NULL;
  float *dw_q = write_outputs ? (float *)view_at(out, 1u)->data : NULL;
  float *dw_dkv = write_outputs ? (float *)view_at(out, 2u)->data : NULL;
  float *dw_uk = write_outputs ? (float *)view_at(out, 3u)->data : NULL;
  float *dw_uv = write_outputs ? (float *)view_at(out, 4u)->data : NULL;
  float *dw_o = write_outputs ? (float *)view_at(out, 5u)->data : NULL;
  return mla_evaluate_backward(&dims, x, w_q, w_dkv, w_uk, w_uv, w_o, dy, dx,
                               dw_q, dw_dkv, dw_uk, dw_uv, dw_o);
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  int backward = 0;
  if (call == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.mla.provider",
                "call is required");
  if (!aligned_span(call, _Alignof(et_kernel_call_v1),
                    ET_KERNEL_CALL_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.mla.provider",
                "invalid call descriptor span or alignment");
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.mla.provider",
                "truncated call descriptor");

  const et_kernel_request_v1 *request = call->request;
  if (request == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.mla.provider",
                "request is required");
  if (!aligned_span(request, _Alignof(et_kernel_request_v1),
                    ET_KERNEL_REQUEST_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.mla.provider",
                "invalid request descriptor span or alignment");
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.mla.provider",
                "truncated request descriptor");
  if (!find_operation(call, &backward) || !request_admitted(request))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, "mod5.mla.provider",
                "request is outside the verified MOD5 MLA rows");

  const char *operation = request->operation;
  if (request->deterministic > 1u)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "invalid deterministic request flag");

  const int32_t status = validate_tables(call, backward, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = evaluate_k1_call(call, backward, 0);
  const int restored = fesetenv(&environment);
  int32_t numeric_status = 0;
  if (restored != 0)
    numeric_status = fail(error, ET_KERNEL_ERROR_INTERNAL,
                          ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                          "floating-point environment could not be restored");
  else if (!valid)
    numeric_status = rejected(error, operation,
                              "invalid mla numeric input or nonfinite result");
  else
    et_kernel_error_clear(error);
  errno = saved_errno;
  return numeric_status;
}

static void invoke_call(const et_kernel_call_v1 *call) {
  int backward = 0;
  if (!find_operation(call, &backward))
    return;
  (void)evaluate_k1_call(call, backward, 1);
}

/* ==================== Direct C11 Helpers ==================== */

static int32_t load_and_validate_dims(const void *dims_ptr,
                                      const char *operation,
                                      et_mod5_mla_dims *out_dims,
                                      et_kernel_error *error) {
  if (dims_ptr == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                "dims descriptor is required");
  if (!aligned_span(dims_ptr, _Alignof(et_mod5_mla_dims),
                    sizeof(et_mod5_mla_dims)))
    return rejected(error, operation, "invalid dims alignment or span");
  memcpy(out_dims, dims_ptr, sizeof(*out_dims));
  if (!shape_admitted((uint64_t)out_dims->batch, (uint64_t)out_dims->tokens,
                      (uint64_t)out_dims->model_dim, (uint64_t)out_dims->heads,
                      (uint64_t)out_dims->head_dim,
                      (uint64_t)out_dims->latent_dim)) {
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "dimensions are outside verified MOD5 MLA domain");
  }
  return 0;
}

static int32_t check_direct_spans(const byte_span *in_spans, size_t in_count,
                                  const byte_span *out_spans, size_t out_count,
                                  const void *meta_ptr, size_t meta_bytes,
                                  const et_kernel_error *error,
                                  const char *operation,
                                  et_kernel_error *err_out) {
  for (size_t i = 0; i < in_count; ++i) {
    if (in_spans[i].data == NULL)
      return fail(err_out, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                  "required input buffer is null");
    if (!aligned_span(in_spans[i].data, _Alignof(float), in_spans[i].bytes))
      return rejected(err_out, operation,
                      "invalid input buffer alignment or span");
  }
  for (size_t i = 0; i < out_count; ++i) {
    if (out_spans[i].data == NULL)
      return fail(err_out, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                  "required output buffer is null");
    if (!aligned_span(out_spans[i].data, _Alignof(float), out_spans[i].bytes))
      return rejected(err_out, operation,
                      "invalid output buffer alignment or span");
  }
  for (size_t i = 0; i < in_count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(in_spans[i].data, in_spans[i].bytes, in_spans[j].data,
                        in_spans[j].bytes))
        return rejected(err_out, operation, "input buffers overlap");
    }
  }
  for (size_t i = 0; i < out_count; ++i) {
    if ((meta_ptr != NULL &&
         spans_overlap(out_spans[i].data, out_spans[i].bytes, meta_ptr,
                       meta_bytes)) ||
        (error != NULL &&
         spans_overlap(out_spans[i].data, out_spans[i].bytes, error,
                       sizeof(*error)))) {
      return fail(err_out, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases metadata or error storage");
    }
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(out_spans[i].data, out_spans[i].bytes,
                        out_spans[j].data, out_spans[j].bytes))
        return fail(err_out, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output buffers overlap");
    }
    for (size_t j = 0; j < in_count; ++j) {
      if (spans_overlap(out_spans[i].data, out_spans[i].bytes,
                        in_spans[j].data, in_spans[j].bytes))
        return fail(err_out, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases input buffer");
    }
  }
  return 0;
}

int32_t et_mod5_mla_forward_f32(const void *dims_ptr, const void *x,
                                const void *w_q, const void *w_dkv,
                                const void *w_uk, const void *w_uv,
                                const void *w_o, void *out,
                                et_kernel_error *error) {
  const char *operation = "mla.forward";
  et_mod5_mla_dims dims;
  int32_t status = load_and_validate_dims(dims_ptr, operation, &dims, error);
  if (status != 0)
    return status;

  const size_t hd = dims.heads * dims.head_dim;
  const size_t x_bytes = dims.batch * dims.tokens * dims.model_dim * sizeof(float);
  const size_t wq_bytes = hd * dims.model_dim * sizeof(float);
  const size_t wdkv_bytes = dims.latent_dim * dims.model_dim * sizeof(float);
  const size_t wuk_bytes = hd * dims.latent_dim * sizeof(float);
  const size_t wo_bytes = dims.model_dim * hd * sizeof(float);

  const byte_span ins[] = {{x, x_bytes},      {w_q, wq_bytes},
                           {w_dkv, wdkv_bytes}, {w_uk, wuk_bytes},
                           {w_uv, wuk_bytes}, {w_o, wo_bytes}};
  const byte_span outs[] = {{out, x_bytes}};
  status = check_direct_spans(ins, COUNT(ins), outs, COUNT(outs), dims_ptr,
                              sizeof(dims), error, operation, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = mla_evaluate_forward(
      &dims, (const float *)x, (const float *)w_q, (const float *)w_dkv,
      (const float *)w_uk, (const float *)w_uv, (const float *)w_o, NULL);
  const int restored = fesetenv(&environment);
  if (restored != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  }
  if (!valid) {
    errno = saved_errno;
    return rejected(error, operation,
                    "invalid mla numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)mla_evaluate_forward(&dims, (const float *)x, (const float *)w_q,
                             (const float *)w_dkv, (const float *)w_uk,
                             (const float *)w_uv, (const float *)w_o,
                             (float *)out);
  et_kernel_error_clear(error);
  return 0;
}

int32_t et_mod5_mla_backward_f32(const void *dims_ptr, const void *x,
                                 const void *w_q, const void *w_dkv,
                                 const void *w_uk, const void *w_uv,
                                 const void *w_o, const void *dy, void *dx,
                                 void *dw_q, void *dw_dkv, void *dw_uk,
                                 void *dw_uv, void *dw_o,
                                 et_kernel_error *error) {
  const char *operation = "mla.backward";
  et_mod5_mla_dims dims;
  int32_t status = load_and_validate_dims(dims_ptr, operation, &dims, error);
  if (status != 0)
    return status;

  const size_t hd = dims.heads * dims.head_dim;
  const size_t x_bytes = dims.batch * dims.tokens * dims.model_dim * sizeof(float);
  const size_t wq_bytes = hd * dims.model_dim * sizeof(float);
  const size_t wdkv_bytes = dims.latent_dim * dims.model_dim * sizeof(float);
  const size_t wuk_bytes = hd * dims.latent_dim * sizeof(float);
  const size_t wo_bytes = dims.model_dim * hd * sizeof(float);

  const byte_span ins[] = {{x, x_bytes},      {w_q, wq_bytes},
                           {w_dkv, wdkv_bytes}, {w_uk, wuk_bytes},
                           {w_uv, wuk_bytes}, {w_o, wo_bytes},
                           {dy, x_bytes}};
  const byte_span outs[] = {{dx, x_bytes},      {dw_q, wq_bytes},
                            {dw_dkv, wdkv_bytes}, {dw_uk, wuk_bytes},
                            {dw_uv, wuk_bytes}, {dw_o, wo_bytes}};
  status = check_direct_spans(ins, COUNT(ins), outs, COUNT(outs), dims_ptr,
                              sizeof(dims), error, operation, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = mla_evaluate_backward(
      &dims, (const float *)x, (const float *)w_q, (const float *)w_dkv,
      (const float *)w_uk, (const float *)w_uv, (const float *)w_o,
      (const float *)dy, NULL, NULL, NULL, NULL, NULL, NULL);
  const int restored = fesetenv(&environment);
  if (restored != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  }
  if (!valid) {
    errno = saved_errno;
    return rejected(error, operation,
                    "invalid mla numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)mla_evaluate_backward(
      &dims, (const float *)x, (const float *)w_q, (const float *)w_dkv,
      (const float *)w_uk, (const float *)w_uv, (const float *)w_o,
      (const float *)dy, (float *)dx, (float *)dw_q, (float *)dw_dkv,
      (float *)dw_uk, (float *)dw_uv, (float *)dw_o);
  et_kernel_error_clear(error);
  return 0;
}

/* ==================== Compressed Latent KV-Cache ==================== */

size_t et_mod5_mla_latent_kv_bytes(size_t batch, size_t tokens,
                                   size_t latent_dim) {
  if (batch < ET_MOD5_MLA_MIN_BATCH || batch > ET_MOD5_MLA_MAX_BATCH ||
      tokens < ET_MOD5_MLA_MIN_TOKENS || tokens > ET_MOD5_MLA_MAX_TOKENS ||
      latent_dim < ET_MOD5_MLA_MIN_LATENT_DIM ||
      latent_dim > ET_MOD5_MLA_MAX_LATENT_DIM)
    return 0u;
  return batch * tokens * latent_dim * sizeof(float);
}

size_t et_mod5_mla_full_kv_bytes(size_t batch, size_t tokens, size_t heads,
                                 size_t head_dim) {
  if (batch < ET_MOD5_MLA_MIN_BATCH || batch > ET_MOD5_MLA_MAX_BATCH ||
      tokens < ET_MOD5_MLA_MIN_TOKENS || tokens > ET_MOD5_MLA_MAX_TOKENS ||
      heads < ET_MOD5_MLA_MIN_HEADS || heads > ET_MOD5_MLA_MAX_HEADS ||
      head_dim < ET_MOD5_MLA_MIN_HEAD_DIM ||
      head_dim > ET_MOD5_MLA_MAX_HEAD_DIM)
    return 0u;
  return batch * tokens * 2u * heads * head_dim * sizeof(float);
}

double et_mod5_mla_kv_compression_ratio(size_t heads, size_t head_dim,
                                        size_t latent_dim) {
  if (heads < ET_MOD5_MLA_MIN_HEADS || heads > ET_MOD5_MLA_MAX_HEADS ||
      head_dim < ET_MOD5_MLA_MIN_HEAD_DIM ||
      head_dim > ET_MOD5_MLA_MAX_HEAD_DIM ||
      latent_dim < ET_MOD5_MLA_MIN_LATENT_DIM ||
      latent_dim > ET_MOD5_MLA_MAX_LATENT_DIM ||
      latent_dim >= 2u * heads * head_dim)
    return 0.0;
  return (double)(2u * heads * head_dim) / (double)latent_dim;
}

static int32_t validate_cache_state(const void *cache_ptr,
                                    const char *operation,
                                    et_mod5_mla_kv_cache *out_cache,
                                    et_kernel_error *error) {
  if (cache_ptr == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                "cache pointer is required");
  if (!aligned_span(cache_ptr, _Alignof(et_mod5_mla_kv_cache),
                    sizeof(et_mod5_mla_kv_cache)))
    return rejected(error, operation, "invalid cache alignment or span");
  memcpy(out_cache, cache_ptr, sizeof(*out_cache));
  if (out_cache->struct_size != sizeof(et_mod5_mla_kv_cache))
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "invalid cache struct_size");
  if (!shape_admitted((uint64_t)out_cache->batch,
                      (uint64_t)out_cache->max_tokens,
                      (uint64_t)out_cache->model_dim,
                      (uint64_t)out_cache->heads,
                      (uint64_t)out_cache->head_dim,
                      (uint64_t)out_cache->latent_dim))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "cache dimensions are outside verified MOD5 MLA domain");
  if (out_cache->seq_len > out_cache->max_tokens)
    return rejected(error, operation, "corrupted cache seq_len");
  const size_t expected_bytes = et_mod5_mla_latent_kv_bytes(
      out_cache->batch, out_cache->max_tokens, out_cache->latent_dim);
  if (out_cache->c_kv_bytes != expected_bytes ||
      !aligned_span(out_cache->c_kv, _Alignof(float), expected_bytes))
    return rejected(error, operation, "invalid cache c_kv buffer or byte size");
  if (spans_overlap(cache_ptr, sizeof(*out_cache), out_cache->c_kv,
                    expected_bytes))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "cache descriptor aliases c_kv buffer");
  if (error != NULL &&
      (spans_overlap(cache_ptr, sizeof(*out_cache), error, sizeof(*error)) ||
       spans_overlap(out_cache->c_kv, expected_bytes, error, sizeof(*error))))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "cache aliases error storage");
  return 0;
}

int32_t et_mod5_mla_cache_init(void *cache_ptr, const void *dims_ptr,
                               void *c_kv_buf, size_t c_kv_bytes,
                               et_kernel_error *error) {
  const char *operation = "mla.cache.init";
  if (cache_ptr == NULL || dims_ptr == NULL || c_kv_buf == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                "cache, dims, and c_kv_buf are required");
  if (!aligned_span(cache_ptr, _Alignof(et_mod5_mla_kv_cache),
                    sizeof(et_mod5_mla_kv_cache)))
    return rejected(error, operation, "invalid cache alignment or span");
  et_mod5_mla_dims dims;
  const int32_t status =
      load_and_validate_dims(dims_ptr, operation, &dims, error);
  if (status != 0)
    return status;
  const size_t expected_bytes =
      et_mod5_mla_latent_kv_bytes(dims.batch, dims.tokens, dims.latent_dim);
  if (c_kv_bytes != expected_bytes)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "c_kv_bytes does not match latent KV cache size");
  if (!aligned_span(c_kv_buf, _Alignof(float), expected_bytes))
    return rejected(error, operation, "invalid c_kv buffer alignment or span");
  if (spans_overlap(cache_ptr, sizeof(et_mod5_mla_kv_cache), c_kv_buf,
                    expected_bytes) ||
      spans_overlap(cache_ptr, sizeof(et_mod5_mla_kv_cache), dims_ptr,
                    sizeof(dims)) ||
      spans_overlap(c_kv_buf, expected_bytes, dims_ptr, sizeof(dims)) ||
      (error != NULL &&
       (spans_overlap(cache_ptr, sizeof(et_mod5_mla_kv_cache), error,
                      sizeof(*error)) ||
        spans_overlap(c_kv_buf, expected_bytes, error, sizeof(*error))))) {
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "cache init buffers overlap");
  }

  memset(c_kv_buf, 0, expected_bytes);
  const et_mod5_mla_kv_cache init = {sizeof(et_mod5_mla_kv_cache),
                                     dims.batch,
                                     dims.tokens,
                                     dims.model_dim,
                                     dims.heads,
                                     dims.head_dim,
                                     dims.latent_dim,
                                     0u,
                                     expected_bytes,
                                     c_kv_buf};
  memcpy(cache_ptr, &init, sizeof(init));
  et_kernel_error_clear(error);
  return 0;
}

int32_t et_mod5_mla_cache_reset(void *cache_ptr, et_kernel_error *error) {
  const char *operation = "mla.cache.reset";
  et_mod5_mla_kv_cache state;
  const int32_t status =
      validate_cache_state(cache_ptr, operation, &state, error);
  if (status != 0)
    return status;
  memset(state.c_kv, 0, state.c_kv_bytes);
  state.seq_len = 0u;
  memcpy(cache_ptr, &state, sizeof(state));
  et_kernel_error_clear(error);
  return 0;
}

static int mla_evaluate_prefill(const et_mod5_mla_kv_cache *state,
                                size_t prefill_tokens, const float *x_prefill,
                                const float *w_q, const float *w_dkv,
                                const float *w_uk, const float *w_uv,
                                const float *w_o, float *c_kv_out,
                                float *out_prefill) {
  const et_mod5_mla_dims dims = {state->batch, prefill_tokens, state->model_dim,
                                 state->heads, state->head_dim,
                                 state->latent_dim};
  const size_t N = dims.batch;
  const size_t D = dims.model_dim;
  const size_t HD = dims.heads * dims.head_dim;
  const size_t dc = dims.latent_dim;

  if (!all_f32_finite(x_prefill, N * prefill_tokens * D) ||
      !all_f32_finite(w_q, HD * D) || !all_f32_finite(w_dkv, dc * D) ||
      !all_f32_finite(w_uk, HD * dc) || !all_f32_finite(w_uv, HD * dc) ||
      !all_f32_finite(w_o, D * HD))
    return 0;

  mla_fwd_ws ws;
  for (size_t n = 0; n < N; ++n) {
    float *c_dst =
        c_kv_out != NULL ? c_kv_out + n * state->max_tokens * dc : NULL;
    float *o_dst =
        out_prefill != NULL ? out_prefill + n * prefill_tokens * D : NULL;
    if (!mla_forward_batch(&dims, n, x_prefill, w_q, w_dkv, w_uk, w_uv, w_o, 1,
                           &ws, c_dst, o_dst))
      return 0;
  }
  return 1;
}

int32_t et_mod5_mla_cache_prefill_f32(void *cache_ptr, size_t prefill_tokens,
                                      const void *x_prefill, const void *w_q,
                                      const void *w_dkv, const void *w_uk,
                                      const void *w_uv, const void *w_o,
                                      void *out_prefill,
                                      et_kernel_error *error) {
  const char *operation = "mla.cache.prefill";
  et_mod5_mla_kv_cache state;
  int32_t status = validate_cache_state(cache_ptr, operation, &state, error);
  if (status != 0)
    return status;

  if (state.seq_len != 0u)
    return rejected(error, operation,
                    "cache prefill requires an empty cache (seq_len == 0)");
  if (prefill_tokens == 0u || prefill_tokens > state.max_tokens)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_BUFFER_TOO_SMALL, operation,
                "prefill_tokens must be in [1, max_tokens]");

  const size_t hd = state.heads * state.head_dim;
  const size_t x_bytes =
      state.batch * prefill_tokens * state.model_dim * sizeof(float);
  const size_t wq_bytes = hd * state.model_dim * sizeof(float);
  const size_t wdkv_bytes = state.latent_dim * state.model_dim * sizeof(float);
  const size_t wuk_bytes = hd * state.latent_dim * sizeof(float);
  const size_t wo_bytes = state.model_dim * hd * sizeof(float);

  const byte_span ins[] = {{x_prefill, x_bytes}, {w_q, wq_bytes},
                           {w_dkv, wdkv_bytes},  {w_uk, wuk_bytes},
                           {w_uv, wuk_bytes},    {w_o, wo_bytes}};
  const byte_span outs[] = {{out_prefill, x_bytes},
                            {state.c_kv, state.c_kv_bytes}};
  status = check_direct_spans(ins, COUNT(ins), outs, COUNT(outs), cache_ptr,
                              sizeof(state), error, operation, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = mla_evaluate_prefill(
      &state, prefill_tokens, (const float *)x_prefill, (const float *)w_q,
      (const float *)w_dkv, (const float *)w_uk, (const float *)w_uv,
      (const float *)w_o, NULL, NULL);
  const int restored = fesetenv(&environment);
  if (restored != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  }
  if (!valid) {
    errno = saved_errno;
    return rejected(error, operation,
                    "invalid mla prefill numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)mla_evaluate_prefill(
      &state, prefill_tokens, (const float *)x_prefill, (const float *)w_q,
      (const float *)w_dkv, (const float *)w_uk, (const float *)w_uv,
      (const float *)w_o, (float *)state.c_kv, (float *)out_prefill);
  state.seq_len = prefill_tokens;
  memcpy(cache_ptr, &state, sizeof(state));
  et_kernel_error_clear(error);
  return 0;
}

typedef struct mla_step_ws {
  double q[MLA_MAX_HD];
  double k[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double v[ET_MOD5_MLA_MAX_TOKENS * MLA_MAX_HD];
  double attn[MLA_MAX_HD];
  double scores[ET_MOD5_MLA_MAX_TOKENS];
  double probs[ET_MOD5_MLA_MAX_TOKENS];
  float c_new[ET_MOD5_MLA_MAX_BATCH * ET_MOD5_MLA_MAX_LATENT_DIM];
} mla_step_ws;

static int mla_evaluate_step(const et_mod5_mla_kv_cache *state,
                             const float *x_step, const float *w_q,
                             const float *w_dkv, const float *w_uk,
                             const float *w_uv, const float *w_o,
                             float *c_kv_buf, float *out_step) {
  const size_t N = state->batch;
  const size_t D = state->model_dim;
  const size_t H = state->heads;
  const size_t Dh = state->head_dim;
  const size_t dc = state->latent_dim;
  const size_t HD = H * Dh;
  const size_t cur_len = state->seq_len;
  const size_t new_len = cur_len + 1u;
  const double scale = 1.0 / sqrt((double)Dh);
  if (!isfinite(scale))
    return 0;

  if (!all_f32_finite(x_step, N * D) || !all_f32_finite(w_q, HD * D) ||
      !all_f32_finite(w_dkv, dc * D) || !all_f32_finite(w_uk, HD * dc) ||
      !all_f32_finite(w_uv, HD * dc) || !all_f32_finite(w_o, D * HD))
    return 0;

  const float *c_kv_ro = (const float *)state->c_kv;
  mla_step_ws ws;

  for (size_t n = 0; n < N; ++n) {
    const float *cn_ro = c_kv_ro + n * state->max_tokens * dc;
    if (cur_len > 0u && !all_f32_finite(cn_ro, cur_len * dc))
      return 0;

    const float *xn = x_step + n * D;
    float *c_new_n = ws.c_new + n * dc;

    /* 1. Compress new token X_step -> c_new [dc] and project Q_new [HD]. */
    for (size_t r = 0; r < dc; ++r) {
      const float *wr = w_dkv + r * D;
      double acc = 0.0;
      for (size_t d = 0; d < D; ++d) {
        acc += (double)xn[d] * (double)wr[d];
      }
      if (!finite_f64_to_f32(acc, &c_new_n[r]))
        return 0;
    }
    for (size_t i = 0; i < HD; ++i) {
      const float *wi = w_q + i * D;
      double acc = 0.0;
      for (size_t d = 0; d < D; ++d) {
        acc += (double)xn[d] * (double)wi[d];
      }
      if (!isfinite(acc))
        return 0;
      ws.q[i] = acc;
    }

    /* 2. Reconstruct K, V [0..cur_len] on the fly from compressed C_kv only. */
    for (size_t tk = 0; tk < new_len; ++tk) {
      const float *c_row = (tk < cur_len) ? (cn_ro + tk * dc) : c_new_n;
      for (size_t i = 0; i < HD; ++i) {
        const float *wuk_i = w_uk + i * dc;
        const float *wuv_i = w_uv + i * dc;
        double ak = 0.0;
        double av = 0.0;
        for (size_t r = 0; r < dc; ++r) {
          const double cr = (double)c_row[r];
          ak += cr * (double)wuk_i[r];
          av += cr * (double)wuv_i[r];
        }
        if (!isfinite(ak) || !isfinite(av))
          return 0;
        ws.k[tk * HD + i] = ak;
        ws.v[tk * HD + i] = av;
      }
    }

    /* 3. Causal binary64 log-sum-exp attention for new query over 0..cur_len. */
    for (size_t h = 0; h < H; ++h) {
      const size_t base_h = h * Dh;
      const double *q_row = ws.q + base_h;
      for (size_t tk = 0; tk < new_len; ++tk) {
        const double *k_row = ws.k + tk * HD + base_h;
        double dot = 0.0;
        for (size_t k = 0; k < Dh; ++k) {
          dot += q_row[k] * k_row[k];
        }
        const double s = dot * scale;
        if (!isfinite(s))
          return 0;
        ws.scores[tk] = s;
      }
      if (!causal_lse_softmax(ws.scores, new_len, ws.probs))
        return 0;
      for (size_t k = 0; k < Dh; ++k) {
        double acc = 0.0;
        for (size_t tk = 0; tk < new_len; ++tk) {
          acc += ws.probs[tk] * ws.v[tk * HD + base_h + k];
        }
        if (!isfinite(acc))
          return 0;
        ws.attn[base_h + k] = acc;
      }
    }

    /* 4. Output projection O_step = Attn @ W_o^T [D]. */
    float *out_n = out_step != NULL ? out_step + n * D : NULL;
    for (size_t d = 0; d < D; ++d) {
      const float *wd = w_o + d * HD;
      double acc = 0.0;
      for (size_t i = 0; i < HD; ++i) {
        acc += ws.attn[i] * (double)wd[i];
      }
      float *dst = out_n != NULL ? &out_n[d] : NULL;
      if (!finite_f64_to_f32(acc, dst))
        return 0;
    }
  }

  /* Commit compressed latent slot only on write pass. */
  if (c_kv_buf != NULL) {
    for (size_t n = 0; n < N; ++n) {
      float *dst = c_kv_buf + (n * state->max_tokens + cur_len) * dc;
      memcpy(dst, ws.c_new + n * dc, dc * sizeof(float));
    }
  }
  return 1;
}

int32_t et_mod5_mla_cache_step_f32(void *cache_ptr, const void *x_step,
                                   const void *w_q, const void *w_dkv,
                                   const void *w_uk, const void *w_uv,
                                   const void *w_o, void *out_step,
                                   et_kernel_error *error) {
  const char *operation = "mla.cache.step";
  et_mod5_mla_kv_cache state;
  int32_t status = validate_cache_state(cache_ptr, operation, &state, error);
  if (status != 0)
    return status;

  if (state.seq_len >= state.max_tokens)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_BUFFER_TOO_SMALL, operation,
                "compressed latent KV-cache overflow (seq_len == max_tokens)");

  const size_t hd = state.heads * state.head_dim;
  const size_t x_bytes = state.batch * state.model_dim * sizeof(float);
  const size_t wq_bytes = hd * state.model_dim * sizeof(float);
  const size_t wdkv_bytes = state.latent_dim * state.model_dim * sizeof(float);
  const size_t wuk_bytes = hd * state.latent_dim * sizeof(float);
  const size_t wo_bytes = state.model_dim * hd * sizeof(float);

  const byte_span ins[] = {{x_step, x_bytes},   {w_q, wq_bytes},
                           {w_dkv, wdkv_bytes}, {w_uk, wuk_bytes},
                           {w_uv, wuk_bytes},   {w_o, wo_bytes}};
  const byte_span outs[] = {{out_step, x_bytes},
                            {state.c_kv, state.c_kv_bytes}};
  status = check_direct_spans(ins, COUNT(ins), outs, COUNT(outs), cache_ptr,
                              sizeof(state), error, operation, error);
  if (status != 0)
    return status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = mla_evaluate_step(
      &state, (const float *)x_step, (const float *)w_q, (const float *)w_dkv,
      (const float *)w_uk, (const float *)w_uv, (const float *)w_o, NULL, NULL);
  const int restored = fesetenv(&environment);
  if (restored != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_INTERNAL,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "floating-point environment could not be restored");
  }
  if (!valid) {
    errno = saved_errno;
    return rejected(error, operation,
                    "invalid mla cache step numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)mla_evaluate_step(&state, (const float *)x_step, (const float *)w_q,
                          (const float *)w_dkv, (const float *)w_uk,
                          (const float *)w_uv, (const float *)w_o,
                          (float *)state.c_kv, (float *)out_step);
  state.seq_len += 1u;
  memcpy(cache_ptr, &state, sizeof(state));
  et_kernel_error_clear(error);
  return 0;
}

static const et_kernel_provider_v1 provider = {
    sizeof(et_kernel_provider_v1),
    ET_KERNEL_ABI_MAJOR,
    ET_KERNEL_ABI_MINOR,
    ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES,
    IMPLEMENTATION,
    "1.0",
    EVIDENCE,
    COUNT(capabilities),
    sizeof(et_kernel_capability_v1),
    sizeof(capabilities),
    capabilities,
    validate_call,
    invoke_call};

const et_kernel_provider_v1 *et_mod5_mla_kernel_provider_v1(void) {
  return &provider;
}
