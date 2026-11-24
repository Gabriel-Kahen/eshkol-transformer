#include "eshkol_transformer/mamba_ssm_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(__x86_64__)
#error "Mamba S6 SSM v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "Mamba S6 SSM requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "Mamba S6 SSM requires standard IEEE-754 binary32/binary64 eval");

#define MAX_N ET_MAMBA_MAX_BATCH
#define MAX_T ET_MAMBA_MAX_SEQ_LEN
#define MAX_D ET_MAMBA_MAX_D_INNER
#define MAX_S ET_MAMBA_MAX_D_STATE
#define MAX_K ET_MAMBA_MAX_K_CONV

#define MAX_NTD (MAX_N * MAX_T * MAX_D)
#define MAX_NTS (MAX_N * MAX_T * MAX_S)
#define MAX_DS (MAX_D * MAX_S)
#define MAX_DK (MAX_D * MAX_K)
#define MAX_TS (MAX_T * MAX_S)
#define MAX_ND (MAX_N * MAX_D)
#define MAX_NDS (MAX_N * MAX_D * MAX_S)
#define MAX_N_KM1_D (MAX_N * (MAX_K - 1u) * MAX_D)

typedef struct byte_span {
  const void *data;
  size_t bytes;
} byte_span;

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

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

static int outputs_disjoint_from_all(const byte_span *outputs, size_t out_count,
                                     const byte_span *inputs, size_t in_count) {
  for (size_t i = 0; i < out_count; ++i) {
    if (outputs[i].data == NULL)
      continue;
    for (size_t j = i + 1u; j < out_count; ++j) {
      if (outputs[j].data == NULL)
        continue;
      if (spans_overlap(outputs[i].data, outputs[i].bytes, outputs[j].data,
                        outputs[j].bytes))
        return 0;
    }
    for (size_t j = 0; j < in_count; ++j) {
      if (inputs[j].data == NULL)
        continue;
      if (spans_overlap(outputs[i].data, outputs[i].bytes, inputs[j].data,
                        inputs[j].bytes))
        return 0;
    }
  }
  return 1;
}

static int fp_environment_supported(void) {
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round-to-nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int dims_valid(size_t batch, size_t seq_len, size_t d_inner,
                      size_t d_state, size_t k_conv) {
  return batch >= ET_MAMBA_MIN_BATCH && batch <= ET_MAMBA_MAX_BATCH &&
         seq_len >= ET_MAMBA_MIN_SEQ_LEN && seq_len <= ET_MAMBA_MAX_SEQ_LEN &&
         d_inner >= ET_MAMBA_MIN_D_INNER && d_inner <= ET_MAMBA_MAX_D_INNER &&
         d_state >= ET_MAMBA_MIN_D_STATE && d_state <= ET_MAMBA_MAX_D_STATE &&
         k_conv >= ET_MAMBA_MIN_K_CONV && k_conv <= ET_MAMBA_MAX_K_CONV;
}

static int cache_dims_valid(size_t batch, size_t d_inner, size_t d_state,
                            size_t k_conv) {
  return batch >= ET_MAMBA_MIN_BATCH && batch <= ET_MAMBA_MAX_BATCH &&
         d_inner >= ET_MAMBA_MIN_D_INNER && d_inner <= ET_MAMBA_MAX_D_INNER &&
         d_state >= ET_MAMBA_MIN_D_STATE && d_state <= ET_MAMBA_MAX_D_STATE &&
         k_conv >= ET_MAMBA_MIN_K_CONV && k_conv <= ET_MAMBA_MAX_K_CONV;
}

static et_mamba_status_t read_config(const void *cfg_ptr,
                                     et_mamba_config_v1 *out) {
  if (!aligned_span(cfg_ptr, _Alignof(et_mamba_config_v1),
                    sizeof(et_mamba_config_v1)))
    return ET_MAMBA_ERR_INVALID_ARG;
  et_mamba_config_v1 cfg;
  memcpy(&cfg, cfg_ptr, sizeof(cfg));
  if (cfg.struct_size != sizeof(et_mamba_config_v1))
    return ET_MAMBA_ERR_INVALID_ARG;
  if (!dims_valid(cfg.batch, cfg.seq_len, cfg.d_inner, cfg.d_state, cfg.k_conv))
    return ET_MAMBA_ERR_UNSUPPORTED;
  *out = cfg;
  return ET_MAMBA_OK;
}

static int all_f32_finite(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(buf[i]))
      return 0;
  }
  return 1;
}

static int all_f64_finite(const double *buf, size_t count) {
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
  *out = f;
  return 1;
}

static double sigmoid_f64(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  }
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

static double silu_f64(double x) { return x * sigmoid_f64(x); }

static double silu_grad_f64(double x) {
  const double s = sigmoid_f64(x);
  return s * (1.0 + x * (1.0 - s));
}

static double softplus_f64(double x) {
  if (x > 0.0) {
    return x + log1p(exp(-x));
  }
  return log1p(exp(x));
}

static et_mamba_status_t compute_neg_exp_a(const float *a_log, size_t count,
                                           double *a_out) {
  for (size_t i = 0; i < count; ++i) {
    const double al = (double)a_log[i];
    if (!isfinite(al))
      return ET_MAMBA_ERR_NONFINITE;
    if (al > ET_MAMBA_MAX_A_LOG)
      return ET_MAMBA_ERR_OVERFLOW;
    const double pos = exp(al);
    if (!isfinite(pos) || pos > (double)FLT_MAX)
      return ET_MAMBA_ERR_OVERFLOW;
    a_out[i] = -pos;
  }
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_validate_config_v1(const void *cfg_ptr) {
  et_mamba_config_v1 cfg;
  return read_config(cfg_ptr, &cfg);
}

/* ============================================================================
 * Stage A: Depthwise Causal Conv1D + SiLU
 * ============================================================================ */

et_mamba_status_t et_mamba_causal_conv1d_silu_forward_v1(
    const void *cfg_ptr, const void *u, const void *conv_w, const void *conv_b,
    void *x_out) {
  if (!fp_environment_supported())
    return ET_MAMBA_ERR_FP_ENV;

  et_mamba_config_v1 cfg;
  et_mamba_status_t status = read_config(cfg_ptr, &cfg);
  if (status != ET_MAMBA_OK)
    return status;

  const size_t n = cfg.batch;
  const size_t t = cfg.seq_len;
  const size_t d = cfg.d_inner;
  const size_t k = cfg.k_conv;

  const size_t ntd = n * t * d;
  const size_t dk = d * k;
  const size_t ntd_bytes = ntd * sizeof(float);
  const size_t dk_bytes = dk * sizeof(float);
  const size_t d_bytes = d * sizeof(float);

  if (!aligned_span(u, _Alignof(float), ntd_bytes) ||
      !aligned_span(conv_w, _Alignof(float), dk_bytes) ||
      !aligned_span(conv_b, _Alignof(float), d_bytes) ||
      !aligned_span(x_out, _Alignof(float), ntd_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {{x_out, ntd_bytes}};
  const byte_span ins[] = {
      {cfg_ptr, sizeof(et_mamba_config_v1)},
      {u, ntd_bytes},
      {conv_w, dk_bytes},
      {conv_b, d_bytes},
  };
  if (!outputs_disjoint_from_all(outs, 1u, ins, 4u))
    return ET_MAMBA_ERR_INVALID_ARG;

  const float *uf = (const float *)u;
  const float *wf = (const float *)conv_w;
  const float *bf = (const float *)conv_b;

  if (!all_f32_finite(uf, ntd) || !all_f32_finite(wf, dk) ||
      !all_f32_finite(bf, d))
    return ET_MAMBA_ERR_NONFINITE;

  fenv_t saved_env;
  fegetenv(&saved_env);

  float x_stage[MAX_NTD];
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ti = 0; ti < t; ++ti) {
      for (size_t di = 0; di < d; ++di) {
        double acc = (double)bf[di];
        for (size_t ki = 0; ki < k; ++ki) {
          const size_t lag = (k - 1u) - ki;
          if (ti >= lag) {
            const size_t tau = ti - lag;
            acc += (double)wf[di * k + ki] *
                   (double)uf[(ni * t + tau) * d + di];
          }
        }
        const double val = silu_f64(acc);
        if (!finite_f64_to_f32(val, &x_stage[(ni * t + ti) * d + di])) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
      }
    }
  }

  fesetenv(&saved_env);
  memcpy(x_out, x_stage, ntd_bytes);
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_causal_conv1d_silu_backward_v1(
    const void *cfg_ptr, const void *u, const void *conv_w, const void *conv_b,
    const void *dx, void *du_out, void *dconv_w_out, void *dconv_b_out) {
  if (!fp_environment_supported())
    return ET_MAMBA_ERR_FP_ENV;

  et_mamba_config_v1 cfg;
  et_mamba_status_t status = read_config(cfg_ptr, &cfg);
  if (status != ET_MAMBA_OK)
    return status;

  const size_t n = cfg.batch;
  const size_t t = cfg.seq_len;
  const size_t d = cfg.d_inner;
  const size_t k = cfg.k_conv;

  const size_t ntd = n * t * d;
  const size_t dk = d * k;
  const size_t ntd_bytes = ntd * sizeof(float);
  const size_t dk_bytes = dk * sizeof(float);
  const size_t d_bytes = d * sizeof(float);

  if (!aligned_span(u, _Alignof(float), ntd_bytes) ||
      !aligned_span(conv_w, _Alignof(float), dk_bytes) ||
      !aligned_span(conv_b, _Alignof(float), d_bytes) ||
      !aligned_span(dx, _Alignof(float), ntd_bytes) ||
      !aligned_span(du_out, _Alignof(float), ntd_bytes) ||
      !aligned_span(dconv_w_out, _Alignof(float), dk_bytes) ||
      !aligned_span(dconv_b_out, _Alignof(float), d_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {
      {du_out, ntd_bytes},
      {dconv_w_out, dk_bytes},
      {dconv_b_out, d_bytes},
  };
  const byte_span ins[] = {
      {cfg_ptr, sizeof(et_mamba_config_v1)},
      {u, ntd_bytes},
      {conv_w, dk_bytes},
      {conv_b, d_bytes},
      {dx, ntd_bytes},
  };
  if (!outputs_disjoint_from_all(outs, 3u, ins, 5u))
    return ET_MAMBA_ERR_INVALID_ARG;

  const float *uf = (const float *)u;
  const float *wf = (const float *)conv_w;
  const float *bf = (const float *)conv_b;
  const float *dxf = (const float *)dx;

  if (!all_f32_finite(uf, ntd) || !all_f32_finite(wf, dk) ||
      !all_f32_finite(bf, d) || !all_f32_finite(dxf, ntd))
    return ET_MAMBA_ERR_NONFINITE;

  fenv_t saved_env;
  fegetenv(&saved_env);

  double du_acc[MAX_NTD];
  double dw_acc[MAX_DK];
  double db_acc[MAX_D];
  for (size_t i = 0; i < ntd; ++i)
    du_acc[i] = 0.0;
  for (size_t i = 0; i < dk; ++i)
    dw_acc[i] = 0.0;
  for (size_t i = 0; i < d; ++i)
    db_acc[i] = 0.0;

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t ti = 0; ti < t; ++ti) {
      for (size_t di = 0; di < d; ++di) {
        double c_val = (double)bf[di];
        for (size_t ki = 0; ki < k; ++ki) {
          const size_t lag = (k - 1u) - ki;
          if (ti >= lag) {
            const size_t tau = ti - lag;
            c_val += (double)wf[di * k + ki] *
                     (double)uf[(ni * t + tau) * d + di];
          }
        }
        const double dc = (double)dxf[(ni * t + ti) * d + di] *
                          silu_grad_f64(c_val);
        if (!isfinite(dc)) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
        db_acc[di] += dc;
        for (size_t ki = 0; ki < k; ++ki) {
          const size_t lag = (k - 1u) - ki;
          if (ti >= lag) {
            const size_t tau = ti - lag;
            const size_t u_idx = (ni * t + tau) * d + di;
            dw_acc[di * k + ki] += dc * (double)uf[u_idx];
            du_acc[u_idx] += dc * (double)wf[di * k + ki];
          }
        }
      }
    }
  }

  float du_stage[MAX_NTD];
  float dw_stage[MAX_DK];
  float db_stage[MAX_D];
  for (size_t i = 0; i < ntd; ++i) {
    if (!finite_f64_to_f32(du_acc[i], &du_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }
  for (size_t i = 0; i < dk; ++i) {
    if (!finite_f64_to_f32(dw_acc[i], &dw_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }
  for (size_t i = 0; i < d; ++i) {
    if (!finite_f64_to_f32(db_acc[i], &db_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }

  fesetenv(&saved_env);
  memcpy(du_out, du_stage, ntd_bytes);
  memcpy(dconv_w_out, dw_stage, dk_bytes);
  memcpy(dconv_b_out, db_stage, d_bytes);
  return ET_MAMBA_OK;
}

/* ============================================================================
 * Stage B: Selective Scan (S6) Forward & Exact Reverse-Time Adjoint Backward
 * ============================================================================ */

et_mamba_status_t et_mamba_selective_scan_forward_v1(
    const void *cfg_ptr, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    void *out) {
  if (!fp_environment_supported())
    return ET_MAMBA_ERR_FP_ENV;

  et_mamba_config_v1 cfg;
  et_mamba_status_t status = read_config(cfg_ptr, &cfg);
  if (status != ET_MAMBA_OK)
    return status;

  const size_t n = cfg.batch;
  const size_t t = cfg.seq_len;
  const size_t d = cfg.d_inner;
  const size_t s = cfg.d_state;

  const size_t ntd = n * t * d;
  const size_t nts = n * t * s;
  const size_t ds = d * s;

  const size_t ntd_bytes = ntd * sizeof(float);
  const size_t nts_bytes = nts * sizeof(float);
  const size_t ds_bytes = ds * sizeof(float);
  const size_t d_bytes = d * sizeof(float);

  if (!aligned_span(x, _Alignof(float), ntd_bytes) ||
      !aligned_span(dt_raw, _Alignof(float), ntd_bytes) ||
      !aligned_span(a_log, _Alignof(float), ds_bytes) ||
      !aligned_span(b_in, _Alignof(float), nts_bytes) ||
      !aligned_span(c_in, _Alignof(float), nts_bytes) ||
      !aligned_span(d_skip, _Alignof(float), d_bytes) ||
      !aligned_span(out, _Alignof(float), ntd_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  if (z_opt != NULL && !aligned_span(z_opt, _Alignof(float), ntd_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {{out, ntd_bytes}};
  const byte_span ins[] = {
      {cfg_ptr, sizeof(et_mamba_config_v1)},
      {x, ntd_bytes},
      {dt_raw, ntd_bytes},
      {a_log, ds_bytes},
      {b_in, nts_bytes},
      {c_in, nts_bytes},
      {d_skip, d_bytes},
      {z_opt, z_opt != NULL ? ntd_bytes : 0u},
  };
  if (!outputs_disjoint_from_all(outs, 1u, ins, 8u))
    return ET_MAMBA_ERR_INVALID_ARG;

  const float *xf = (const float *)x;
  const float *dtf = (const float *)dt_raw;
  const float *alf = (const float *)a_log;
  const float *bf = (const float *)b_in;
  const float *cf = (const float *)c_in;
  const float *dskf = (const float *)d_skip;
  const float *zf = (const float *)z_opt;

  if (!all_f32_finite(xf, ntd) || !all_f32_finite(dtf, ntd) ||
      !all_f32_finite(alf, ds) || !all_f32_finite(bf, nts) ||
      !all_f32_finite(cf, nts) || !all_f32_finite(dskf, d) ||
      (zf != NULL && !all_f32_finite(zf, ntd)))
    return ET_MAMBA_ERR_NONFINITE;

  fenv_t saved_env;
  fegetenv(&saved_env);

  double a_neg[MAX_DS];
  status = compute_neg_exp_a(alf, ds, a_neg);
  if (status != ET_MAMBA_OK) {
    fesetenv(&saved_env);
    return status;
  }

  float out_stage[MAX_NTD];
  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t di = 0; di < d; ++di) {
      double h[MAX_S];
      for (size_t si = 0; si < s; ++si)
        h[si] = 0.0;

      const double d_val = (double)dskf[di];
      for (size_t ti = 0; ti < t; ++ti) {
        const size_t ntd_idx = (ni * t + ti) * d + di;
        const size_t nts_base = (ni * t + ti) * s;
        const double x_val = (double)xf[ntd_idx];
        const double dt_val = softplus_f64((double)dtf[ntd_idx]);
        if (!isfinite(dt_val)) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }

        double y_val = d_val * x_val;
        for (size_t si = 0; si < s; ++si) {
          const double da_val = exp(dt_val * a_neg[di * s + si]);
          const double db_x = dt_val * (double)bf[nts_base + si] * x_val;
          h[si] = da_val * h[si] + db_x;
          if (!isfinite(h[si])) {
            fesetenv(&saved_env);
            return ET_MAMBA_ERR_OVERFLOW;
          }
          y_val += h[si] * (double)cf[nts_base + si];
        }

        const double out_val =
            (zf != NULL) ? (y_val * silu_f64((double)zf[ntd_idx])) : y_val;
        if (!finite_f64_to_f32(out_val, &out_stage[ntd_idx])) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
      }
    }
  }

  fesetenv(&saved_env);
  memcpy(out, out_stage, ntd_bytes);
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_selective_scan_backward_v1(
    const void *cfg_ptr, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    const void *dout, void *dx_out, void *ddt_raw_out, void *da_log_out,
    void *db_out, void *dc_out, void *dd_skip_out, void *dz_out_opt) {
  if (!fp_environment_supported())
    return ET_MAMBA_ERR_FP_ENV;

  et_mamba_config_v1 cfg;
  et_mamba_status_t status = read_config(cfg_ptr, &cfg);
  if (status != ET_MAMBA_OK)
    return status;

  if ((z_opt == NULL) != (dz_out_opt == NULL))
    return ET_MAMBA_ERR_INVALID_ARG;

  const size_t n = cfg.batch;
  const size_t t = cfg.seq_len;
  const size_t d = cfg.d_inner;
  const size_t s = cfg.d_state;

  const size_t ntd = n * t * d;
  const size_t nts = n * t * s;
  const size_t ds = d * s;

  const size_t ntd_bytes = ntd * sizeof(float);
  const size_t nts_bytes = nts * sizeof(float);
  const size_t ds_bytes = ds * sizeof(float);
  const size_t d_bytes = d * sizeof(float);

  if (!aligned_span(x, _Alignof(float), ntd_bytes) ||
      !aligned_span(dt_raw, _Alignof(float), ntd_bytes) ||
      !aligned_span(a_log, _Alignof(float), ds_bytes) ||
      !aligned_span(b_in, _Alignof(float), nts_bytes) ||
      !aligned_span(c_in, _Alignof(float), nts_bytes) ||
      !aligned_span(d_skip, _Alignof(float), d_bytes) ||
      !aligned_span(dout, _Alignof(float), ntd_bytes) ||
      !aligned_span(dx_out, _Alignof(float), ntd_bytes) ||
      !aligned_span(ddt_raw_out, _Alignof(float), ntd_bytes) ||
      !aligned_span(da_log_out, _Alignof(float), ds_bytes) ||
      !aligned_span(db_out, _Alignof(float), nts_bytes) ||
      !aligned_span(dc_out, _Alignof(float), nts_bytes) ||
      !aligned_span(dd_skip_out, _Alignof(float), d_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  if (z_opt != NULL &&
      (!aligned_span(z_opt, _Alignof(float), ntd_bytes) ||
       !aligned_span(dz_out_opt, _Alignof(float), ntd_bytes)))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {
      {dx_out, ntd_bytes},
      {ddt_raw_out, ntd_bytes},
      {da_log_out, ds_bytes},
      {db_out, nts_bytes},
      {dc_out, nts_bytes},
      {dd_skip_out, d_bytes},
      {dz_out_opt, dz_out_opt != NULL ? ntd_bytes : 0u},
  };
  const byte_span ins[] = {
      {cfg_ptr, sizeof(et_mamba_config_v1)},
      {x, ntd_bytes},
      {dt_raw, ntd_bytes},
      {a_log, ds_bytes},
      {b_in, nts_bytes},
      {c_in, nts_bytes},
      {d_skip, d_bytes},
      {z_opt, z_opt != NULL ? ntd_bytes : 0u},
      {dout, ntd_bytes},
  };
  if (!outputs_disjoint_from_all(outs, 7u, ins, 9u))
    return ET_MAMBA_ERR_INVALID_ARG;

  const float *xf = (const float *)x;
  const float *dtf = (const float *)dt_raw;
  const float *alf = (const float *)a_log;
  const float *bf = (const float *)b_in;
  const float *cf = (const float *)c_in;
  const float *dskf = (const float *)d_skip;
  const float *zf = (const float *)z_opt;
  const float *doutf = (const float *)dout;

  if (!all_f32_finite(xf, ntd) || !all_f32_finite(dtf, ntd) ||
      !all_f32_finite(alf, ds) || !all_f32_finite(bf, nts) ||
      !all_f32_finite(cf, nts) || !all_f32_finite(dskf, d) ||
      !all_f32_finite(doutf, ntd) ||
      (zf != NULL && !all_f32_finite(zf, ntd)))
    return ET_MAMBA_ERR_NONFINITE;

  fenv_t saved_env;
  fegetenv(&saved_env);

  double a_neg[MAX_DS];
  status = compute_neg_exp_a(alf, ds, a_neg);
  if (status != ET_MAMBA_OK) {
    fesetenv(&saved_env);
    return status;
  }

  float dx_stage[MAX_NTD];
  float ddt_stage[MAX_NTD];
  float dz_stage[MAX_NTD];
  double da_log_acc[MAX_DS];
  double db_acc[MAX_NTS];
  double dc_acc[MAX_NTS];
  double dd_skip_acc[MAX_D];

  for (size_t i = 0; i < ds; ++i)
    da_log_acc[i] = 0.0;
  for (size_t i = 0; i < nts; ++i) {
    db_acc[i] = 0.0;
    dc_acc[i] = 0.0;
  }
  for (size_t i = 0; i < d; ++i)
    dd_skip_acc[i] = 0.0;

  double h_hist[MAX_TS];
  double da_hist[MAX_TS];
  double dt_hist[MAX_T];
  double y_hist[MAX_T];

  for (size_t ni = 0; ni < n; ++ni) {
    for (size_t di = 0; di < d; ++di) {
      const double d_val = (double)dskf[di];

      /* Forward trajectory for (ni, di) in binary64. */
      for (size_t ti = 0; ti < t; ++ti) {
        const size_t ntd_idx = (ni * t + ti) * d + di;
        const size_t nts_base = (ni * t + ti) * s;
        const double x_val = (double)xf[ntd_idx];
        const double dt_val = softplus_f64((double)dtf[ntd_idx]);
        if (!isfinite(dt_val)) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
        dt_hist[ti] = dt_val;

        double y_val = d_val * x_val;
        for (size_t si = 0; si < s; ++si) {
          const double h_prev = (ti > 0u) ? h_hist[(ti - 1u) * s + si] : 0.0;
          const double da_val = exp(dt_val * a_neg[di * s + si]);
          const double h_cur =
              da_val * h_prev + dt_val * (double)bf[nts_base + si] * x_val;
          if (!isfinite(h_cur)) {
            fesetenv(&saved_env);
            return ET_MAMBA_ERR_OVERFLOW;
          }
          da_hist[ti * s + si] = da_val;
          h_hist[ti * s + si] = h_cur;
          y_val += h_cur * (double)cf[nts_base + si];
        }
        if (!isfinite(y_val)) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
        y_hist[ti] = y_val;
      }

      /* Exact reverse-time adjoint sweep for (ni, di). */
      double dh_next[MAX_S];
      for (size_t si = 0; si < s; ++si)
        dh_next[si] = 0.0;

      for (size_t ti = t; ti-- > 0u;) {
        const size_t ntd_idx = (ni * t + ti) * d + di;
        const size_t nts_base = (ni * t + ti) * s;
        const double x_val = (double)xf[ntd_idx];
        const double dt_raw_val = (double)dtf[ntd_idx];
        const double dt_val = dt_hist[ti];
        const double dout_val = (double)doutf[ntd_idx];

        double dy = dout_val;
        if (zf != NULL) {
          const double z_val = (double)zf[ntd_idx];
          dy = dout_val * silu_f64(z_val);
          const double dz_val = dout_val * y_hist[ti] * silu_grad_f64(z_val);
          if (!finite_f64_to_f32(dz_val, &dz_stage[ntd_idx])) {
            fesetenv(&saved_env);
            return ET_MAMBA_ERR_OVERFLOW;
          }
        }

        dd_skip_acc[di] += dy * x_val;
        double dx_val = dy * d_val;
        double ddt_val = 0.0;

        for (size_t si = 0; si < s; ++si) {
          const double h_cur = h_hist[ti * s + si];
          const double h_prev = (ti > 0u) ? h_hist[(ti - 1u) * s + si] : 0.0;
          const double da_val = da_hist[ti * s + si];
          const double a_val = a_neg[di * s + si];
          const double b_val = (double)bf[nts_base + si];
          const double c_val = (double)cf[nts_base + si];

          const double dh = dh_next[si] + dy * c_val;
          dc_acc[nts_base + si] += dy * h_cur;

          const double d_dt_a = dh * h_prev * da_val;
          ddt_val += d_dt_a * a_val + dh * b_val * x_val;
          da_log_acc[di * s + si] += d_dt_a * dt_val * a_val;
          db_acc[nts_base + si] += dh * dt_val * x_val;
          dx_val += dh * dt_val * b_val;

          dh_next[si] = dh * da_val;
          if (!isfinite(dh_next[si])) {
            fesetenv(&saved_env);
            return ET_MAMBA_ERR_OVERFLOW;
          }
        }

        const double ddt_raw_val = ddt_val * sigmoid_f64(dt_raw_val);
        if (!finite_f64_to_f32(dx_val, &dx_stage[ntd_idx]) ||
            !finite_f64_to_f32(ddt_raw_val, &ddt_stage[ntd_idx])) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
      }
    }
  }

  float da_log_stage[MAX_DS];
  float db_stage[MAX_NTS];
  float dc_stage[MAX_NTS];
  float dd_skip_stage[MAX_D];

  for (size_t i = 0; i < ds; ++i) {
    if (!finite_f64_to_f32(da_log_acc[i], &da_log_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }
  for (size_t i = 0; i < nts; ++i) {
    if (!finite_f64_to_f32(db_acc[i], &db_stage[i]) ||
        !finite_f64_to_f32(dc_acc[i], &dc_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }
  for (size_t i = 0; i < d; ++i) {
    if (!finite_f64_to_f32(dd_skip_acc[i], &dd_skip_stage[i])) {
      fesetenv(&saved_env);
      return ET_MAMBA_ERR_OVERFLOW;
    }
  }

  fesetenv(&saved_env);
  memcpy(dx_out, dx_stage, ntd_bytes);
  memcpy(ddt_raw_out, ddt_stage, ntd_bytes);
  memcpy(da_log_out, da_log_stage, ds_bytes);
  memcpy(db_out, db_stage, nts_bytes);
  memcpy(dc_out, dc_stage, nts_bytes);
  memcpy(dd_skip_out, dd_skip_stage, d_bytes);
  if (dz_out_opt != NULL)
    memcpy(dz_out_opt, dz_stage, ntd_bytes);
  return ET_MAMBA_OK;
}

/* ============================================================================
 * Stage C: O(1) Constant-Memory Recurrent Inference Cache
 * ============================================================================ */

size_t et_mamba_cache_conv_state_bytes_v1(const void *cfg_ptr) {
  et_mamba_config_v1 cfg;
  if (read_config(cfg_ptr, &cfg) != ET_MAMBA_OK)
    return 0u;
  return cfg.batch * (cfg.k_conv - 1u) * cfg.d_inner * sizeof(float);
}

size_t et_mamba_cache_ssm_state_bytes_v1(const void *cfg_ptr) {
  et_mamba_config_v1 cfg;
  if (read_config(cfg_ptr, &cfg) != ET_MAMBA_OK)
    return 0u;
  return cfg.batch * cfg.d_inner * cfg.d_state * sizeof(double);
}

size_t et_mamba_cache_total_state_bytes_v1(const void *cfg_ptr) {
  et_mamba_config_v1 cfg;
  if (read_config(cfg_ptr, &cfg) != ET_MAMBA_OK)
    return 0u;
  return cfg.batch * (cfg.k_conv - 1u) * cfg.d_inner * sizeof(float) +
         cfg.batch * cfg.d_inner * cfg.d_state * sizeof(double);
}

static et_mamba_status_t read_cache(const void *cache_ptr,
                                    et_mamba_cache_v1 *out) {
  if (!aligned_span(cache_ptr, _Alignof(et_mamba_cache_v1),
                    sizeof(et_mamba_cache_v1)))
    return ET_MAMBA_ERR_INVALID_ARG;
  et_mamba_cache_v1 st;
  memcpy(&st, cache_ptr, sizeof(st));
  if (st.struct_size != sizeof(et_mamba_cache_v1))
    return ET_MAMBA_ERR_INVALID_ARG;
  if (!cache_dims_valid(st.batch, st.d_inner, st.d_state, st.k_conv))
    return ET_MAMBA_ERR_UNSUPPORTED;

  const size_t expected_conv =
      st.batch * (st.k_conv - 1u) * st.d_inner * sizeof(float);
  const size_t expected_ssm =
      st.batch * st.d_inner * st.d_state * sizeof(double);
  if (st.conv_state_bytes != expected_conv ||
      st.ssm_state_bytes != expected_ssm)
    return ET_MAMBA_ERR_SHAPE_MISMATCH;
  if (!aligned_span(st.conv_state, _Alignof(float), expected_conv) ||
      !aligned_span(st.ssm_state, _Alignof(double), expected_ssm))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span cache_spans[] = {
      {cache_ptr, sizeof(et_mamba_cache_v1)},
      {st.conv_state, expected_conv},
      {st.ssm_state, expected_ssm},
  };
  if (!outputs_disjoint_from_all(cache_spans, 3u, NULL, 0u))
    return ET_MAMBA_ERR_INVALID_ARG;

  *out = st;
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_cache_init_v1(void *cache_ptr, const void *cfg_ptr,
                                         void *conv_state_buf,
                                         size_t conv_state_bytes,
                                         void *ssm_state_buf,
                                         size_t ssm_state_bytes) {
  if (!aligned_span(cache_ptr, _Alignof(et_mamba_cache_v1),
                    sizeof(et_mamba_cache_v1)))
    return ET_MAMBA_ERR_INVALID_ARG;

  et_mamba_config_v1 cfg;
  et_mamba_status_t status = read_config(cfg_ptr, &cfg);
  if (status != ET_MAMBA_OK)
    return status;

  const size_t expected_conv =
      cfg.batch * (cfg.k_conv - 1u) * cfg.d_inner * sizeof(float);
  const size_t expected_ssm =
      cfg.batch * cfg.d_inner * cfg.d_state * sizeof(double);
  if (conv_state_bytes != expected_conv || ssm_state_bytes != expected_ssm)
    return ET_MAMBA_ERR_SHAPE_MISMATCH;

  if (!aligned_span(conv_state_buf, _Alignof(float), expected_conv) ||
      !aligned_span(ssm_state_buf, _Alignof(double), expected_ssm))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {
      {cache_ptr, sizeof(et_mamba_cache_v1)},
      {conv_state_buf, expected_conv},
      {ssm_state_buf, expected_ssm},
  };
  const byte_span ins[] = {{cfg_ptr, sizeof(et_mamba_config_v1)}};
  if (!outputs_disjoint_from_all(outs, 3u, ins, 1u))
    return ET_MAMBA_ERR_INVALID_ARG;

  memset(conv_state_buf, 0, expected_conv);
  memset(ssm_state_buf, 0, expected_ssm);

  const et_mamba_cache_v1 state = {
      sizeof(et_mamba_cache_v1),
      cfg.batch,
      cfg.d_inner,
      cfg.d_state,
      cfg.k_conv,
      0u,
      (float *)conv_state_buf,
      expected_conv,
      (double *)ssm_state_buf,
      expected_ssm,
  };
  memcpy(cache_ptr, &state, sizeof(state));
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_cache_reset_v1(void *cache_ptr) {
  et_mamba_cache_v1 st;
  et_mamba_status_t status = read_cache(cache_ptr, &st);
  if (status != ET_MAMBA_OK)
    return status;

  memset(st.conv_state, 0, st.conv_state_bytes);
  memset(st.ssm_state, 0, st.ssm_state_bytes);
  st.step_count = 0u;
  memcpy(cache_ptr, &st, sizeof(st));
  return ET_MAMBA_OK;
}

et_mamba_status_t et_mamba_cache_step_v1(
    void *cache_ptr, const void *u_step, const void *conv_w, const void *conv_b,
    const void *dt_raw_step, const void *a_log, const void *b_step,
    const void *c_step, const void *d_skip, const void *z_step_opt,
    void *out_step) {
  if (!fp_environment_supported())
    return ET_MAMBA_ERR_FP_ENV;

  et_mamba_cache_v1 st;
  et_mamba_status_t status = read_cache(cache_ptr, &st);
  if (status != ET_MAMBA_OK)
    return status;

  const size_t n = st.batch;
  const size_t d = st.d_inner;
  const size_t s = st.d_state;
  const size_t k = st.k_conv;
  const size_t km1 = k - 1u;

  const size_t nd = n * d;
  const size_t ns = n * s;
  const size_t dk = d * k;
  const size_t ds = d * s;
  const size_t conv_elems = n * km1 * d;
  const size_t ssm_elems = n * d * s;

  const size_t nd_bytes = nd * sizeof(float);
  const size_t ns_bytes = ns * sizeof(float);
  const size_t dk_bytes = dk * sizeof(float);
  const size_t ds_bytes = ds * sizeof(float);
  const size_t d_bytes = d * sizeof(float);

  if (!aligned_span(u_step, _Alignof(float), nd_bytes) ||
      !aligned_span(conv_w, _Alignof(float), dk_bytes) ||
      !aligned_span(conv_b, _Alignof(float), d_bytes) ||
      !aligned_span(dt_raw_step, _Alignof(float), nd_bytes) ||
      !aligned_span(a_log, _Alignof(float), ds_bytes) ||
      !aligned_span(b_step, _Alignof(float), ns_bytes) ||
      !aligned_span(c_step, _Alignof(float), ns_bytes) ||
      !aligned_span(d_skip, _Alignof(float), d_bytes) ||
      !aligned_span(out_step, _Alignof(float), nd_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  if (z_step_opt != NULL && !aligned_span(z_step_opt, _Alignof(float), nd_bytes))
    return ET_MAMBA_ERR_INVALID_ARG;

  const byte_span outs[] = {
      {out_step, nd_bytes},
      {cache_ptr, sizeof(et_mamba_cache_v1)},
      {st.conv_state, st.conv_state_bytes},
      {st.ssm_state, st.ssm_state_bytes},
  };
  const byte_span ins[] = {
      {u_step, nd_bytes},
      {conv_w, dk_bytes},
      {conv_b, d_bytes},
      {dt_raw_step, nd_bytes},
      {a_log, ds_bytes},
      {b_step, ns_bytes},
      {c_step, ns_bytes},
      {d_skip, d_bytes},
      {z_step_opt, z_step_opt != NULL ? nd_bytes : 0u},
  };
  if (!outputs_disjoint_from_all(outs, 4u, ins, 9u))
    return ET_MAMBA_ERR_INVALID_ARG;

  const float *uf = (const float *)u_step;
  const float *wf = (const float *)conv_w;
  const float *bf_conv = (const float *)conv_b;
  const float *dtf = (const float *)dt_raw_step;
  const float *alf = (const float *)a_log;
  const float *bf = (const float *)b_step;
  const float *cf = (const float *)c_step;
  const float *dskf = (const float *)d_skip;
  const float *zf = (const float *)z_step_opt;

  if (!all_f32_finite(uf, nd) || !all_f32_finite(wf, dk) ||
      !all_f32_finite(bf_conv, d) || !all_f32_finite(dtf, nd) ||
      !all_f32_finite(alf, ds) || !all_f32_finite(bf, ns) ||
      !all_f32_finite(cf, ns) || !all_f32_finite(dskf, d) ||
      (zf != NULL && !all_f32_finite(zf, nd)) ||
      !all_f32_finite(st.conv_state, conv_elems) ||
      !all_f64_finite(st.ssm_state, ssm_elems))
    return ET_MAMBA_ERR_NONFINITE;

  fenv_t saved_env;
  fegetenv(&saved_env);

  double a_neg[MAX_DS];
  status = compute_neg_exp_a(alf, ds, a_neg);
  if (status != ET_MAMBA_OK) {
    fesetenv(&saved_env);
    return status;
  }

  float next_conv[MAX_N_KM1_D];
  double next_ssm[MAX_NDS];
  float out_stage[MAX_ND];

  for (size_t ni = 0; ni < n; ++ni) {
    const size_t conv_batch_off = ni * km1 * d;
    const size_t ssm_batch_off = ni * d * s;
    const size_t ns_base = ni * s;

    for (size_t di = 0; di < d; ++di) {
      /* Stage A single-step causal conv1d + SiLU matching full-sequence order. */
      double c_acc = (double)bf_conv[di];
      for (size_t ki = 0; ki < km1; ++ki) {
        c_acc += (double)wf[di * k + ki] *
                 (double)st.conv_state[conv_batch_off + ki * d + di];
      }
      const float u_cur = uf[ni * d + di];
      c_acc += (double)wf[di * k + km1] * (double)u_cur;

      float x_f32 = 0.0f;
      if (!finite_f64_to_f32(silu_f64(c_acc), &x_f32)) {
        fesetenv(&saved_env);
        return ET_MAMBA_ERR_OVERFLOW;
      }

      /* Shift conv_state window left by 1 and append u_cur. */
      for (size_t ki = 0; ki + 1u < km1; ++ki) {
        next_conv[conv_batch_off + ki * d + di] =
            st.conv_state[conv_batch_off + (ki + 1u) * d + di];
      }
      next_conv[conv_batch_off + (km1 - 1u) * d + di] = u_cur;

      /* Stage B single-step S6 update. */
      const double x_val = (double)x_f32;
      const double dt_val = softplus_f64((double)dtf[ni * d + di]);
      if (!isfinite(dt_val)) {
        fesetenv(&saved_env);
        return ET_MAMBA_ERR_OVERFLOW;
      }

      double y_val = (double)dskf[di] * x_val;
      for (size_t si = 0; si < s; ++si) {
        const size_t h_idx = ssm_batch_off + di * s + si;
        const double da_val = exp(dt_val * a_neg[di * s + si]);
        const double h_new = da_val * st.ssm_state[h_idx] +
                             dt_val * (double)bf[ns_base + si] * x_val;
        if (!isfinite(h_new)) {
          fesetenv(&saved_env);
          return ET_MAMBA_ERR_OVERFLOW;
        }
        next_ssm[h_idx] = h_new;
        y_val += h_new * (double)cf[ns_base + si];
      }

      const double out_val =
          (zf != NULL) ? (y_val * silu_f64((double)zf[ni * d + di])) : y_val;
      if (!finite_f64_to_f32(out_val, &out_stage[ni * d + di])) {
        fesetenv(&saved_env);
        return ET_MAMBA_ERR_OVERFLOW;
      }
    }
  }

  fesetenv(&saved_env);
  memcpy(st.conv_state, next_conv, st.conv_state_bytes);
  memcpy(st.ssm_state, next_ssm, st.ssm_state_bytes);
  st.step_count += 1u;
  memcpy(cache_ptr, &st, sizeof(st));
  memcpy(out_step, out_stage, nd_bytes);
  return ET_MAMBA_OK;
}
