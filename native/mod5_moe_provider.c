#include "eshkol_transformer/mod5_moe_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#if !defined(__x86_64__)
#error "MOD5 MoE v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MOD5 MoE requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 MoE requires standard IEEE-754 binary32/binary64 eval");

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define MAX_TOTAL_TOKENS (ET_MOD5_MOE_MAX_BATCH * ET_MOD5_MOE_MAX_TOKENS)

#define MOE_ROW(min_e, k)                                                      \
  {                                                                            \
    6u, (const et_kernel_dimension_range_v1[]) {                               \
      {ET_MOD5_MOE_MIN_BATCH, ET_MOD5_MOE_MAX_BATCH, 0u, {0}},                 \
          {ET_MOD5_MOE_MIN_TOKENS, ET_MOD5_MOE_MAX_TOKENS, 0u, {0}},           \
          {ET_MOD5_MOE_MIN_MODEL_DIM, ET_MOD5_MOE_MAX_MODEL_DIM, 0u, {0}},     \
          {ET_MOD5_MOE_MIN_FFN_DIM, ET_MOD5_MOE_MAX_FFN_DIM, 0u, {0}},         \
          {(min_e), ET_MOD5_MOE_MAX_EXPERTS, 0u, {0}}, {                       \
        (k), (k), 0u, { 0 }                                                    \
      }                                                                        \
    }                                                                          \
  }

#define IMPLEMENTATION "mod5.moe.cpu-f32.serial"
#define EVIDENCE "MOD5:cpu-f32-moe-v1"

static const et_kernel_shape_range_v1 moe_rows[] = {
    MOE_ROW(2u, 1u), MOE_ROW(2u, 2u), MOE_ROW(3u, 3u), MOE_ROW(4u, 4u)};

static const char *const moe_operations[] = {"moe.forward", "moe.backward"};
static const char *const moe_dtypes[] = {"f32"};
static const char *const moe_devices[] = {"cpu"};

static const et_kernel_capability_v1 capabilities[] = {
    {sizeof(et_kernel_capability_v1), "kernel.moe",
     ET_KERNEL_CAPABILITY_VERIFIED, IMPLEMENTATION, "1.0", EVIDENCE, 1u, {0},
     COUNT(moe_operations), moe_operations, COUNT(moe_dtypes), moe_dtypes,
     COUNT(moe_devices), moe_devices, COUNT(moe_rows), moe_rows}};

typedef struct tensor_table {
  size_t count;
  size_t stride;
  size_t bytes;
  const void *base;
} tensor_table;

enum dtype { F32, I64 };
typedef struct tensor_schema {
  enum dtype dtype;
  size_t rank;
  const uint64_t *shape;
} tensor_schema;

typedef struct span_desc {
  const void *ptr;
  size_t bytes;
  size_t alignment;
} span_desc;

typedef struct routing_state {
  double probs[MAX_TOTAL_TOKENS][ET_MOD5_MOE_MAX_EXPERTS];
  double top_sum_p[MAX_TOTAL_TOKENS];
  double top_w[MAX_TOTAL_TOKENS][ET_MOD5_MOE_MAX_TOP_K];
  uint8_t top_idx[MAX_TOTAL_TOKENS][ET_MOD5_MOE_MAX_TOP_K];
  int64_t counts[ET_MOD5_MOE_MAX_EXPERTS];
  float aux_loss_f32;
} routing_state;

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

static int overlap(const et_kernel_tensor_view_v1 *a,
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

static tensor_table inputs(const et_kernel_call_v1 *call) {
  return (tensor_table){call->input_count, call->input_stride,
                        call->input_bytes, call->inputs};
}

static tensor_table outputs(const et_kernel_call_v1 *call) {
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

static int shape_admitted(uint64_t n, uint64_t t, uint64_t d, uint64_t f,
                          uint64_t e, uint64_t k) {
  return n >= ET_MOD5_MOE_MIN_BATCH && n <= ET_MOD5_MOE_MAX_BATCH &&
         t >= ET_MOD5_MOE_MIN_TOKENS && t <= ET_MOD5_MOE_MAX_TOKENS &&
         d >= ET_MOD5_MOE_MIN_MODEL_DIM && d <= ET_MOD5_MOE_MAX_MODEL_DIM &&
         f >= ET_MOD5_MOE_MIN_FFN_DIM && f <= ET_MOD5_MOE_MAX_FFN_DIM &&
         e >= ET_MOD5_MOE_MIN_EXPERTS && e <= ET_MOD5_MOE_MAX_EXPERTS &&
         k >= ET_MOD5_MOE_MIN_TOP_K && k <= ET_MOD5_MOE_MAX_TOP_K && k <= e;
}

static int fp_environment_supported(void) {
  uint16_t x87;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round to nearest on both units, no FTZ or DAZ. */
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

static int f64_to_finite_f32(double value, float *out) {
  if (!isfinite(value))
    return 0;
  const float f = (float)value;
  if (!isfinite(f))
    return 0;
  if (out != NULL)
    *out = f;
  return 1;
}

static double stable_sigmoid(double g) {
  if (g >= 0.0) {
    const double eg = exp(-g);
    return 1.0 / (1.0 + eg);
  }
  const double eg = exp(g);
  return eg / (1.0 + eg);
}

static int compute_routing(size_t total_tokens, size_t model_dim,
                           size_t num_experts, size_t top_k, const float *x,
                           const float *w_gate, double alpha_aux,
                           routing_state *state) {
  double p_sum[ET_MOD5_MOE_MAX_EXPERTS];
  for (size_t e = 0; e < num_experts; ++e) {
    state->counts[e] = 0;
    p_sum[e] = 0.0;
  }

  for (size_t s = 0; s < total_tokens; ++s) {
    const float *x_row = x + s * model_dim;
    double logits[ET_MOD5_MOE_MAX_EXPERTS];
    double max_logit = -INFINITY;
    for (size_t e = 0; e < num_experts; ++e) {
      const float *w_row = w_gate + e * model_dim;
      double acc = 0.0;
      for (size_t d = 0; d < model_dim; ++d) {
        const double prod = (double)x_row[d] * (double)w_row[d];
        if (!isfinite(prod))
          return 0;
        acc += prod;
        if (!isfinite(acc))
          return 0;
      }
      logits[e] = acc;
      if (e == 0u || acc > max_logit)
        max_logit = acc;
    }

    double exp_vals[ET_MOD5_MOE_MAX_EXPERTS];
    double denom = 0.0;
    for (size_t e = 0; e < num_experts; ++e) {
      const double shifted = logits[e] - max_logit;
      if (!isfinite(shifted))
        return 0;
      const double ex = exp(shifted);
      if (!isfinite(ex))
        return 0;
      exp_vals[e] = ex;
      denom += ex;
      if (!isfinite(denom))
        return 0;
    }
    if (!(denom > 0.0))
      return 0;

    for (size_t e = 0; e < num_experts; ++e) {
      const double p = exp_vals[e] / denom;
      if (!isfinite(p))
        return 0;
      state->probs[s][e] = p;
      p_sum[e] += p;
      if (!isfinite(p_sum[e]))
        return 0;
    }

    /* Deterministic top-K selection with lowest-expert-index tie-breaking. */
    uint8_t picked[ET_MOD5_MOE_MAX_EXPERTS] = {0};
    double sum_p = 0.0;
    for (size_t j = 0; j < top_k; ++j) {
      size_t best_e = 0u;
      double best_p = -1.0;
      int found = 0;
      for (size_t e = 0; e < num_experts; ++e) {
        if (picked[e])
          continue;
        if (!found || state->probs[s][e] > best_p) {
          best_e = e;
          best_p = state->probs[s][e];
          found = 1;
        }
      }
      picked[best_e] = 1u;
      state->top_idx[s][j] = (uint8_t)best_e;
      state->counts[best_e] += 1;
      sum_p += best_p;
      if (!isfinite(sum_p))
        return 0;
    }
    if (!(sum_p > 0.0))
      return 0;
    state->top_sum_p[s] = sum_p;

    for (size_t j = 0; j < top_k; ++j) {
      const double w = state->probs[s][state->top_idx[s][j]] / sum_p;
      if (!isfinite(w))
        return 0;
      state->top_w[s][j] = w;
    }
  }

  /* Switch load-balancing auxiliary loss: alpha_aux * E * sum_e f_e * P_mean_e */
  const double total_slots = (double)(total_tokens * top_k);
  const double total_tok_d = (double)total_tokens;
  double aux_acc = 0.0;
  for (size_t e = 0; e < num_experts; ++e) {
    const double f_e = (double)state->counts[e] / total_slots;
    const double p_mean_e = p_sum[e] / total_tok_d;
    const double term = f_e * p_mean_e;
    if (!isfinite(term))
      return 0;
    aux_acc += term;
    if (!isfinite(aux_acc))
      return 0;
  }
  const double aux_loss = alpha_aux * (double)num_experts * aux_acc;
  if (!f64_to_finite_f32(aux_loss, &state->aux_loss_f32))
    return 0;
  return 1;
}

static int expert_swiglu_forward(size_t model_dim, size_t ffn_dim,
                                 const float *x_row, const float *wg_e,
                                 const float *wu_e, const float *wd_e,
                                 double *gate_out, double *up_out,
                                 double *sig_out, double *silu_out,
                                 double *hidden_out, double *exp_out) {
  for (size_t f = 0; f < ffn_dim; ++f) {
    const float *wg_row = wg_e + f * model_dim;
    const float *wu_row = wu_e + f * model_dim;
    double g = 0.0;
    double u = 0.0;
    for (size_t d = 0; d < model_dim; ++d) {
      const double xd = (double)x_row[d];
      const double gp = (double)wg_row[d] * xd;
      const double up = (double)wu_row[d] * xd;
      if (!isfinite(gp) || !isfinite(up))
        return 0;
      g += gp;
      u += up;
      if (!isfinite(g) || !isfinite(u))
        return 0;
    }
    const double sig = stable_sigmoid(g);
    const double silu = g * sig;
    const double h = silu * u;
    if (!isfinite(sig) || !isfinite(silu) || !isfinite(h))
      return 0;
    if (gate_out != NULL)
      gate_out[f] = g;
    if (up_out != NULL)
      up_out[f] = u;
    if (sig_out != NULL)
      sig_out[f] = sig;
    if (silu_out != NULL)
      silu_out[f] = silu;
    hidden_out[f] = h;
  }

  for (size_t d = 0; d < model_dim; ++d) {
    const float *wd_row = wd_e + d * ffn_dim;
    double acc = 0.0;
    for (size_t f = 0; f < ffn_dim; ++f) {
      const double prod = (double)wd_row[f] * hidden_out[f];
      if (!isfinite(prod))
        return 0;
      acc += prod;
      if (!isfinite(acc))
        return 0;
    }
    exp_out[d] = acc;
  }
  return 1;
}

static int moe_forward_eval(size_t batch, size_t tokens, size_t model_dim,
                            size_t ffn_dim, size_t num_experts, size_t top_k,
                            const float *x, const float *w_gate,
                            const float *w_gate_exp, const float *w_up_exp,
                            const float *w_down_exp, double alpha_aux,
                            int write_outputs, float *y_out,
                            float *aux_loss_out, int64_t *expert_counts_out,
                            int64_t *active_evals_out) {
  if (!isfinite(alpha_aux) || !(alpha_aux >= 0.0))
    return 0;

  const size_t total_tokens = batch * tokens;
  const size_t exp_stride = ffn_dim * model_dim;
  if (!all_f32_finite(x, total_tokens * model_dim) ||
      !all_f32_finite(w_gate, num_experts * model_dim) ||
      !all_f32_finite(w_gate_exp, num_experts * exp_stride) ||
      !all_f32_finite(w_up_exp, num_experts * exp_stride) ||
      !all_f32_finite(w_down_exp, num_experts * exp_stride))
    return 0;

  routing_state route;
  if (!compute_routing(total_tokens, model_dim, num_experts, top_k, x, w_gate,
                       alpha_aux, &route))
    return 0;

  int64_t active_evals = 0;
  double hidden[ET_MOD5_MOE_MAX_FFN_DIM];
  double exp_out[ET_MOD5_MOE_MAX_MODEL_DIM];
  double y_acc[ET_MOD5_MOE_MAX_MODEL_DIM];

  for (size_t s = 0; s < total_tokens; ++s) {
    const float *x_row = x + s * model_dim;
    for (size_t d = 0; d < model_dim; ++d)
      y_acc[d] = 0.0;

    /* Sparse dispatch: execute SwiGLU ONLY for the selected top-K experts. */
    for (size_t j = 0; j < top_k; ++j) {
      const size_t e = (size_t)route.top_idx[s][j];
      const double w = route.top_w[s][j];
      const float *wg_e = w_gate_exp + e * exp_stride;
      const float *wu_e = w_up_exp + e * exp_stride;
      const float *wd_e = w_down_exp + e * exp_stride;
      if (!expert_swiglu_forward(model_dim, ffn_dim, x_row, wg_e, wu_e, wd_e,
                                 NULL, NULL, NULL, NULL, hidden, exp_out))
        return 0;
      active_evals += 1;
      for (size_t d = 0; d < model_dim; ++d) {
        const double contrib = w * exp_out[d];
        if (!isfinite(contrib))
          return 0;
        y_acc[d] += contrib;
        if (!isfinite(y_acc[d]))
          return 0;
      }
    }

    for (size_t d = 0; d < model_dim; ++d) {
      float *dst = write_outputs ? (y_out + s * model_dim + d) : NULL;
      if (!f64_to_finite_f32(y_acc[d], dst))
        return 0;
    }
  }

  if (write_outputs) {
    *aux_loss_out = route.aux_loss_f32;
    for (size_t e = 0; e < num_experts; ++e)
      expert_counts_out[e] = route.counts[e];
    *active_evals_out = active_evals;
  }
  return 1;
}

static int moe_backward_eval(size_t batch, size_t tokens, size_t model_dim,
                             size_t ffn_dim, size_t num_experts, size_t top_k,
                             const float *x, const float *w_gate,
                             const float *w_gate_exp, const float *w_up_exp,
                             const float *w_down_exp, double alpha_aux,
                             const float *dy, int write_outputs, float *dx_out,
                             float *dw_gate_out, float *dw_gate_exp_out,
                             float *dw_up_exp_out, float *dw_down_exp_out) {
  if (!isfinite(alpha_aux) || !(alpha_aux >= 0.0))
    return 0;

  const size_t total_tokens = batch * tokens;
  const size_t exp_stride = ffn_dim * model_dim;
  if (!all_f32_finite(x, total_tokens * model_dim) ||
      !all_f32_finite(w_gate, num_experts * model_dim) ||
      !all_f32_finite(w_gate_exp, num_experts * exp_stride) ||
      !all_f32_finite(w_up_exp, num_experts * exp_stride) ||
      !all_f32_finite(w_down_exp, num_experts * exp_stride) ||
      !all_f32_finite(dy, total_tokens * model_dim))
    return 0;

  routing_state route;
  if (!compute_routing(total_tokens, model_dim, num_experts, top_k, x, w_gate,
                       alpha_aux, &route))
    return 0;

  double dx_acc[MAX_TOTAL_TOKENS][ET_MOD5_MOE_MAX_MODEL_DIM];
  double dw_gate_acc[ET_MOD5_MOE_MAX_EXPERTS][ET_MOD5_MOE_MAX_MODEL_DIM];
  double dw_tok[MAX_TOTAL_TOKENS][ET_MOD5_MOE_MAX_TOP_K];
  for (size_t s = 0; s < total_tokens; ++s) {
    for (size_t d = 0; d < model_dim; ++d)
      dx_acc[s][d] = 0.0;
    for (size_t j = 0; j < top_k; ++j)
      dw_tok[s][j] = 0.0;
  }
  for (size_t e = 0; e < num_experts; ++e) {
    for (size_t d = 0; d < model_dim; ++d)
      dw_gate_acc[e][d] = 0.0;
  }

  /* Per-expert scratch buffers keep stack frame bounded (< 80 KB) and zero-heap. */
  double dw_ge[ET_MOD5_MOE_MAX_FFN_DIM * ET_MOD5_MOE_MAX_MODEL_DIM];
  double dw_ue[ET_MOD5_MOE_MAX_FFN_DIM * ET_MOD5_MOE_MAX_MODEL_DIM];
  double dw_de[ET_MOD5_MOE_MAX_MODEL_DIM * ET_MOD5_MOE_MAX_FFN_DIM];

  double g_buf[ET_MOD5_MOE_MAX_FFN_DIM];
  double u_buf[ET_MOD5_MOE_MAX_FFN_DIM];
  double sig_buf[ET_MOD5_MOE_MAX_FFN_DIM];
  double silu_buf[ET_MOD5_MOE_MAX_FFN_DIM];
  double h_buf[ET_MOD5_MOE_MAX_FFN_DIM];
  double exp_out[ET_MOD5_MOE_MAX_MODEL_DIM];
  double do_buf[ET_MOD5_MOE_MAX_MODEL_DIM];

  for (size_t e = 0; e < num_experts; ++e) {
    for (size_t i = 0; i < exp_stride; ++i) {
      dw_ge[i] = 0.0;
      dw_ue[i] = 0.0;
      dw_de[i] = 0.0;
    }

    const float *wg_e = w_gate_exp + e * exp_stride;
    const float *wu_e = w_up_exp + e * exp_stride;
    const float *wd_e = w_down_exp + e * exp_stride;

    /* Sparse backward: skip experts with zero routed tokens completely. */
    if (route.counts[e] > 0) {
      for (size_t s = 0; s < total_tokens; ++s) {
        size_t slot = top_k;
        for (size_t j = 0; j < top_k; ++j) {
          if ((size_t)route.top_idx[s][j] == e) {
            slot = j;
            break;
          }
        }
        if (slot == top_k)
          continue;

        const float *x_row = x + s * model_dim;
        const float *dy_row = dy + s * model_dim;
        const double w = route.top_w[s][slot];

        if (!expert_swiglu_forward(model_dim, ffn_dim, x_row, wg_e, wu_e, wd_e,
                                   g_buf, u_buf, sig_buf, silu_buf, h_buf,
                                   exp_out))
          return 0;

        /* Routing weight gradient dw = <dY_s, expert_e(x_s)>. */
        double dw_val = 0.0;
        for (size_t d = 0; d < model_dim; ++d) {
          const double dy_d = (double)dy_row[d];
          const double prod = dy_d * exp_out[d];
          const double do_d = w * dy_d;
          if (!isfinite(prod) || !isfinite(do_d))
            return 0;
          dw_val += prod;
          if (!isfinite(dw_val))
            return 0;
          do_buf[d] = do_d;
        }
        dw_tok[s][slot] = dw_val;

        /* Down projection backward: dW_down += do * h^T, dh = W_down^T * do. */
        for (size_t d = 0; d < model_dim; ++d) {
          const double do_d = do_buf[d];
          double *dw_de_row = dw_de + d * ffn_dim;
          for (size_t f = 0; f < ffn_dim; ++f) {
            const double prod = do_d * h_buf[f];
            if (!isfinite(prod))
              return 0;
            dw_de_row[f] += prod;
            if (!isfinite(dw_de_row[f]))
              return 0;
          }
        }

        for (size_t f = 0; f < ffn_dim; ++f) {
          double dh = 0.0;
          for (size_t d = 0; d < model_dim; ++d) {
            const double prod = (double)wd_e[d * ffn_dim + f] * do_buf[d];
            if (!isfinite(prod))
              return 0;
            dh += prod;
            if (!isfinite(dh))
              return 0;
          }

          /* SwiGLU backward: h = silu(g) * u. */
          const double du = dh * silu_buf[f];
          const double sig = sig_buf[f];
          const double dsilu_dg = sig * (1.0 + g_buf[f] * (1.0 - sig));
          const double dg = (dh * u_buf[f]) * dsilu_dg;
          if (!isfinite(du) || !isfinite(dsilu_dg) || !isfinite(dg))
            return 0;

          const float *wg_row = wg_e + f * model_dim;
          const float *wu_row = wu_e + f * model_dim;
          double *dw_ge_row = dw_ge + f * model_dim;
          double *dw_ue_row = dw_ue + f * model_dim;
          for (size_t d = 0; d < model_dim; ++d) {
            const double xd = (double)x_row[d];
            const double p_ge = dg * xd;
            const double p_ue = du * xd;
            const double p_xg = (double)wg_row[d] * dg;
            const double p_xu = (double)wu_row[d] * du;
            if (!isfinite(p_ge) || !isfinite(p_ue) || !isfinite(p_xg) ||
                !isfinite(p_xu))
              return 0;
            dw_ge_row[d] += p_ge;
            dw_ue_row[d] += p_ue;
            dx_acc[s][d] += p_xg;
            dx_acc[s][d] += p_xu;
            if (!isfinite(dw_ge_row[d]) || !isfinite(dw_ue_row[d]) ||
                !isfinite(dx_acc[s][d]))
              return 0;
          }
        }
      }
    }

    for (size_t i = 0; i < exp_stride; ++i) {
      float *dst_ge =
          write_outputs ? (dw_gate_exp_out + e * exp_stride + i) : NULL;
      float *dst_ue =
          write_outputs ? (dw_up_exp_out + e * exp_stride + i) : NULL;
      float *dst_de =
          write_outputs ? (dw_down_exp_out + e * exp_stride + i) : NULL;
      if (!f64_to_finite_f32(dw_ge[i], dst_ge) ||
          !f64_to_finite_f32(dw_ue[i], dst_ue) ||
          !f64_to_finite_f32(dw_de[i], dst_de))
        return 0;
    }
  }

  /* Router backward: combine sparse top-K renormalization & Switch aux loss. */
  const double total_slots = (double)(total_tokens * top_k);
  const double total_tok_d = (double)total_tokens;
  double dp_aux[ET_MOD5_MOE_MAX_EXPERTS];
  for (size_t e = 0; e < num_experts; ++e) {
    const double f_e = (double)route.counts[e] / total_slots;
    const double grad = (alpha_aux * (double)num_experts * f_e) / total_tok_d;
    if (!isfinite(grad))
      return 0;
    dp_aux[e] = grad;
  }

  for (size_t s = 0; s < total_tokens; ++s) {
    double dp[ET_MOD5_MOE_MAX_EXPERTS];
    for (size_t e = 0; e < num_experts; ++e)
      dp[e] = dp_aux[e];

    /* Top-K renormalization backward: w_j = P[e_j] / S_p. */
    double w_dot_dw = 0.0;
    for (size_t j = 0; j < top_k; ++j) {
      const double prod = route.top_w[s][j] * dw_tok[s][j];
      if (!isfinite(prod))
        return 0;
      w_dot_dw += prod;
      if (!isfinite(w_dot_dw))
        return 0;
    }
    const double sum_p = route.top_sum_p[s];
    for (size_t j = 0; j < top_k; ++j) {
      const size_t e = (size_t)route.top_idx[s][j];
      const double dp_route = (dw_tok[s][j] - w_dot_dw) / sum_p;
      if (!isfinite(dp_route))
        return 0;
      dp[e] += dp_route;
      if (!isfinite(dp[e]))
        return 0;
    }

    /* Softmax backward: dR_e = P_e * (dP_e - sum_m P_m * dP_m). */
    double p_dot_dp = 0.0;
    for (size_t e = 0; e < num_experts; ++e) {
      const double prod = route.probs[s][e] * dp[e];
      if (!isfinite(prod))
        return 0;
      p_dot_dp += prod;
      if (!isfinite(p_dot_dp))
        return 0;
    }

    const float *x_row = x + s * model_dim;
    for (size_t e = 0; e < num_experts; ++e) {
      const double dr = route.probs[s][e] * (dp[e] - p_dot_dp);
      if (!isfinite(dr))
        return 0;
      const float *w_row = w_gate + e * model_dim;
      for (size_t d = 0; d < model_dim; ++d) {
        const double dw_term = dr * (double)x_row[d];
        const double dx_term = dr * (double)w_row[d];
        if (!isfinite(dw_term) || !isfinite(dx_term))
          return 0;
        dw_gate_acc[e][d] += dw_term;
        dx_acc[s][d] += dx_term;
        if (!isfinite(dw_gate_acc[e][d]) || !isfinite(dx_acc[s][d]))
          return 0;
      }
    }
  }

  for (size_t s = 0; s < total_tokens; ++s) {
    for (size_t d = 0; d < model_dim; ++d) {
      float *dst = write_outputs ? (dx_out + s * model_dim + d) : NULL;
      if (!f64_to_finite_f32(dx_acc[s][d], dst))
        return 0;
    }
  }
  for (size_t e = 0; e < num_experts; ++e) {
    for (size_t d = 0; d < model_dim; ++d) {
      float *dst = write_outputs ? (dw_gate_out + e * model_dim + d) : NULL;
      if (!f64_to_finite_f32(dw_gate_acc[e][d], dst))
        return 0;
    }
  }
  return 1;
}

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

static int32_t validate_view(const et_kernel_tensor_view_v1 *view,
                             const tensor_schema *schema, size_t stride,
                             const char *operation, et_kernel_error *error) {
  static const char *const names[] = {[F32] = "f32", [I64] = "i64"};
  static const size_t sizes[] = {[F32] = sizeof(float),
                                 [I64] = sizeof(int64_t)};
  static const size_t alignments[] = {[F32] = _Alignof(float),
                                      [I64] = _Alignof(int64_t)};
  size_t bytes = sizes[schema->dtype];
  if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      view->struct_size > stride) {
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "invalid view prefix size");
  }
  if (!exact_text(view->dtype, names[schema->dtype])) {
    return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_TEXT, operation,
                "view dtype does not match moe schema");
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
                "view rank does not match moe schema");
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
                  "view extent does not match moe schema");
    }
    bytes *= (size_t)schema->shape[d];
  }
  if (view->byte_length != bytes) {
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "incorrect view byte length");
  }
  if (!aligned_span(view->data, alignments[schema->dtype], bytes)) {
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
    const int32_t status = validate_view(view_at(table, i), &schema[i],
                                         table.stride, operation, error);
    if (status != 0)
      return status;
  }
  return 0;
}

static int32_t validate_tables(const et_kernel_call_v1 *call, int backward,
                               et_kernel_error *error) {
  const char *operation = call->request->operation;
  const uint64_t *s = call->request->shape;
  const uint64_t x_shape[] = {s[0], s[1], s[2]};
  const uint64_t wg_shape[] = {s[4], s[2]};
  const uint64_t wgu_shape[] = {s[4], s[3], s[2]};
  const uint64_t wd_shape[] = {s[4], s[2], s[3]};
  const uint64_t cnt_shape[] = {s[4]};

  const tensor_table in = inputs(call);
  const tensor_table out = outputs(call);

  const tensor_schema in_schema[] = {
      {F32, 3u, x_shape},   {F32, 2u, wg_shape}, {F32, 3u, wgu_shape},
      {F32, 3u, wgu_shape}, {F32, 3u, wd_shape}, {F32, 0u, NULL},
      {F32, 3u, x_shape}};
  const tensor_schema fwd_out_schema[] = {
      {F32, 3u, x_shape}, {F32, 0u, NULL}, {I64, 1u, cnt_shape}, {I64, 0u, NULL}};
  const tensor_schema bwd_out_schema[] = {
      {F32, 3u, x_shape},   {F32, 2u, wg_shape}, {F32, 3u, wgu_shape},
      {F32, 3u, wgu_shape}, {F32, 3u, wd_shape}};

  int32_t status = validate_table(in, backward ? 7u : 6u, in_schema, operation,
                                  error);
  if (status != 0)
    return status;
  status = validate_table(out, backward ? 5u : 4u,
                          backward ? bwd_out_schema : fwd_out_schema, operation,
                          error);
  if (status != 0)
    return status;

  for (size_t i = 0; i < in.count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (overlap(view_at(in, i), view_at(in, j)))
        return rejected(error, operation, "input storage overlaps");
    }
  }
  for (size_t i = 0; i < out.count; ++i) {
    const et_kernel_tensor_view_v1 *out_i = view_at(out, i);
    if (output_aliases_metadata(out_i, call, in, out, error)) {
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases live K1 metadata");
    }
    for (size_t j = 0; j < i; ++j) {
      if (overlap(out_i, view_at(out, j))) {
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output storage overlaps");
      }
    }
    for (size_t j = 0; j < in.count; ++j) {
      if (overlap(out_i, view_at(in, j))) {
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases input");
      }
    }
  }
  return 0;
}

static int find_operation(const et_kernel_call_v1 *call, int *backward) {
  if (!exact_text(call->capability, "kernel.moe"))
    return 0;
  if (exact_text(call->request->operation, "moe.forward")) {
    *backward = 0;
    return 1;
  }
  if (exact_text(call->request->operation, "moe.backward")) {
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

static int evaluate_call(const et_kernel_call_v1 *call, int backward,
                         int write_outputs) {
  const tensor_table in = inputs(call);
  const tensor_table out = outputs(call);
  const uint64_t *s = call->request->shape;
  const float *x = (const float *)view_at(in, 0u)->data;
  const float *w_gate = (const float *)view_at(in, 1u)->data;
  const float *w_gate_exp = (const float *)view_at(in, 2u)->data;
  const float *w_up_exp = (const float *)view_at(in, 3u)->data;
  const float *w_down_exp = (const float *)view_at(in, 4u)->data;
  const float alpha_f32 = *(const float *)view_at(in, 5u)->data;
  if (!isfinite(alpha_f32))
    return 0;

  if (!backward) {
    float *y = write_outputs ? (float *)view_at(out, 0u)->data : NULL;
    float *aux = write_outputs ? (float *)view_at(out, 1u)->data : NULL;
    int64_t *counts = write_outputs ? (int64_t *)view_at(out, 2u)->data : NULL;
    int64_t *evals = write_outputs ? (int64_t *)view_at(out, 3u)->data : NULL;
    return moe_forward_eval((size_t)s[0], (size_t)s[1], (size_t)s[2],
                            (size_t)s[3], (size_t)s[4], (size_t)s[5], x, w_gate,
                            w_gate_exp, w_up_exp, w_down_exp, (double)alpha_f32,
                            write_outputs, y, aux, counts, evals);
  }

  const float *dy = (const float *)view_at(in, 6u)->data;
  float *dx = write_outputs ? (float *)view_at(out, 0u)->data : NULL;
  float *dw_gate = write_outputs ? (float *)view_at(out, 1u)->data : NULL;
  float *dw_gate_exp = write_outputs ? (float *)view_at(out, 2u)->data : NULL;
  float *dw_up_exp = write_outputs ? (float *)view_at(out, 3u)->data : NULL;
  float *dw_down_exp = write_outputs ? (float *)view_at(out, 4u)->data : NULL;
  return moe_backward_eval((size_t)s[0], (size_t)s[1], (size_t)s[2],
                           (size_t)s[3], (size_t)s[4], (size_t)s[5], x, w_gate,
                           w_gate_exp, w_up_exp, w_down_exp, (double)alpha_f32,
                           dy, write_outputs, dx, dw_gate, dw_gate_exp,
                           dw_up_exp, dw_down_exp);
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  int backward = 0;
  if (call == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.moe.provider",
                "call is required");
  if (!aligned_span(call, _Alignof(et_kernel_call_v1),
                    ET_KERNEL_CALL_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.moe.provider",
                "invalid call descriptor span or alignment");
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.moe.provider",
                "truncated call descriptor");

  const et_kernel_request_v1 *request = call->request;
  if (request == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, "mod5.moe.provider",
                "request is required");
  if (!aligned_span(request, _Alignof(et_kernel_request_v1),
                    ET_KERNEL_REQUEST_V1_0_SIZE))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, "mod5.moe.provider",
                "invalid request descriptor span or alignment");
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "mod5.moe.provider",
                "truncated request descriptor");
  if (!find_operation(call, &backward) || !request_admitted(request))
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, "mod5.moe.provider",
                "request is outside the verified MOD5 MoE rows");

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

  const int valid = evaluate_call(call, backward, 0);
  const int restored = fesetenv(&environment);
  int32_t numeric_status = 0;
  if (restored != 0)
    numeric_status = fail(error, ET_KERNEL_ERROR_INTERNAL,
                          ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                          "floating-point environment could not be restored");
  else if (!valid)
    numeric_status = rejected(error, operation,
                              "invalid moe numeric input or nonfinite result");
  else
    et_kernel_error_clear(error);
  errno = saved_errno;
  return numeric_status;
}

static void invoke_call(const et_kernel_call_v1 *call) {
  int backward = 0;
  if (!find_operation(call, &backward))
    return;
  (void)evaluate_call(call, backward, 1);
}

static int32_t validate_direct_spans(const span_desc *in_spans, size_t n_in,
                                     const span_desc *out_spans, size_t n_out,
                                     const char *operation,
                                     et_kernel_error *error) {
  for (size_t i = 0; i < n_in; ++i) {
    if (in_spans[i].ptr == NULL)
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                  "input buffer pointer is required");
    if (!aligned_span(in_spans[i].ptr, in_spans[i].alignment,
                      in_spans[i].bytes))
      return rejected(error, operation, "invalid input alignment or span");
  }
  for (size_t i = 0; i < n_out; ++i) {
    if (out_spans[i].ptr == NULL)
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                  "output buffer pointer is required");
    if (!aligned_span(out_spans[i].ptr, out_spans[i].alignment,
                      out_spans[i].bytes))
      return rejected(error, operation, "invalid output alignment or span");
  }
  for (size_t i = 0; i < n_in; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(in_spans[i].ptr, in_spans[i].bytes, in_spans[j].ptr,
                        in_spans[j].bytes))
        return rejected(error, operation, "input buffers overlap");
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    if (error != NULL && spans_overlap(out_spans[i].ptr, out_spans[i].bytes,
                                       error, sizeof(*error)))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                  "output aliases error storage");
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(out_spans[i].ptr, out_spans[i].bytes, out_spans[j].ptr,
                        out_spans[j].bytes))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output buffers overlap");
    }
    for (size_t j = 0; j < n_in; ++j) {
      if (spans_overlap(out_spans[i].ptr, out_spans[i].bytes, in_spans[j].ptr,
                        in_spans[j].bytes))
        return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                    "output aliases input");
    }
  }
  return 0;
}

int32_t et_mod5_moe_forward_f32(size_t batch, size_t tokens, size_t model_dim,
                                size_t ffn_dim, size_t num_experts,
                                size_t top_k, const void *x,
                                const void *w_gate, const void *w_gate_exp,
                                const void *w_up_exp, const void *w_down_exp,
                                double alpha_aux, void *y, void *aux_loss,
                                void *expert_counts, void *active_expert_evals,
                                et_kernel_error *error) {
  const char *operation = "moe.forward";
  if (!shape_admitted((uint64_t)batch, (uint64_t)tokens, (uint64_t)model_dim,
                      (uint64_t)ffn_dim, (uint64_t)num_experts,
                      (uint64_t)top_k)) {
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "shape is outside verified MOD5 MoE domain");
  }

  const size_t x_bytes = batch * tokens * model_dim * sizeof(float);
  const size_t wg_bytes = num_experts * model_dim * sizeof(float);
  const size_t exp_bytes = num_experts * ffn_dim * model_dim * sizeof(float);
  const size_t cnt_bytes = num_experts * sizeof(int64_t);

  const span_desc in_spans[] = {
      {x, x_bytes, _Alignof(float)},
      {w_gate, wg_bytes, _Alignof(float)},
      {w_gate_exp, exp_bytes, _Alignof(float)},
      {w_up_exp, exp_bytes, _Alignof(float)},
      {w_down_exp, exp_bytes, _Alignof(float)}};
  const span_desc out_spans[] = {
      {y, x_bytes, _Alignof(float)},
      {aux_loss, sizeof(float), _Alignof(float)},
      {expert_counts, cnt_bytes, _Alignof(int64_t)},
      {active_expert_evals, sizeof(int64_t), _Alignof(int64_t)}};

  const int32_t span_status = validate_direct_spans(
      in_spans, COUNT(in_spans), out_spans, COUNT(out_spans), operation, error);
  if (span_status != 0)
    return span_status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = moe_forward_eval(
      batch, tokens, model_dim, ffn_dim, num_experts, top_k, (const float *)x,
      (const float *)w_gate, (const float *)w_gate_exp, (const float *)w_up_exp,
      (const float *)w_down_exp, alpha_aux, 0, NULL, NULL, NULL, NULL);
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
                    "invalid moe numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)moe_forward_eval(batch, tokens, model_dim, ffn_dim, num_experts, top_k,
                         (const float *)x, (const float *)w_gate,
                         (const float *)w_gate_exp, (const float *)w_up_exp,
                         (const float *)w_down_exp, alpha_aux, 1, (float *)y,
                         (float *)aux_loss, (int64_t *)expert_counts,
                         (int64_t *)active_expert_evals);
  et_kernel_error_clear(error);
  return 0;
}

int32_t et_mod5_moe_backward_f32(size_t batch, size_t tokens, size_t model_dim,
                                 size_t ffn_dim, size_t num_experts,
                                 size_t top_k, const void *x,
                                 const void *w_gate, const void *w_gate_exp,
                                 const void *w_up_exp, const void *w_down_exp,
                                 double alpha_aux, const void *dy, void *dx,
                                 void *dw_gate, void *dw_gate_exp,
                                 void *dw_up_exp, void *dw_down_exp,
                                 et_kernel_error *error) {
  const char *operation = "moe.backward";
  if (!shape_admitted((uint64_t)batch, (uint64_t)tokens, (uint64_t)model_dim,
                      (uint64_t)ffn_dim, (uint64_t)num_experts,
                      (uint64_t)top_k)) {
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "shape is outside verified MOD5 MoE domain");
  }

  const size_t x_bytes = batch * tokens * model_dim * sizeof(float);
  const size_t wg_bytes = num_experts * model_dim * sizeof(float);
  const size_t exp_bytes = num_experts * ffn_dim * model_dim * sizeof(float);

  const span_desc in_spans[] = {
      {x, x_bytes, _Alignof(float)},
      {w_gate, wg_bytes, _Alignof(float)},
      {w_gate_exp, exp_bytes, _Alignof(float)},
      {w_up_exp, exp_bytes, _Alignof(float)},
      {w_down_exp, exp_bytes, _Alignof(float)},
      {dy, x_bytes, _Alignof(float)}};
  const span_desc out_spans[] = {
      {dx, x_bytes, _Alignof(float)},
      {dw_gate, wg_bytes, _Alignof(float)},
      {dw_gate_exp, exp_bytes, _Alignof(float)},
      {dw_up_exp, exp_bytes, _Alignof(float)},
      {dw_down_exp, exp_bytes, _Alignof(float)}};

  const int32_t span_status = validate_direct_spans(
      in_spans, COUNT(in_spans), out_spans, COUNT(out_spans), operation, error);
  if (span_status != 0)
    return span_status;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return fail(error, ET_KERNEL_ERROR_UNSUPPORTED,
                ET_KERNEL_CODE_PROVIDER_REJECTED, operation,
                "unsupported or unreadable floating-point environment");
  }

  const int valid = moe_backward_eval(
      batch, tokens, model_dim, ffn_dim, num_experts, top_k, (const float *)x,
      (const float *)w_gate, (const float *)w_gate_exp, (const float *)w_up_exp,
      (const float *)w_down_exp, alpha_aux, (const float *)dy, 0, NULL, NULL,
      NULL, NULL, NULL);
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
                    "invalid moe numeric input or nonfinite result");
  }
  errno = saved_errno;

  (void)moe_backward_eval(
      batch, tokens, model_dim, ffn_dim, num_experts, top_k, (const float *)x,
      (const float *)w_gate, (const float *)w_gate_exp, (const float *)w_up_exp,
      (const float *)w_down_exp, alpha_aux, (const float *)dy, 1, (float *)dx,
      (float *)dw_gate, (float *)dw_gate_exp, (float *)dw_up_exp,
      (float *)dw_down_exp);
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

const et_kernel_provider_v1 *et_mod5_moe_kernel_provider_v1(void) {
  return &provider;
}
