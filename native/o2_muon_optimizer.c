#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/o2_muon_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u &&
                   sizeof(size_t) <= sizeof(uint64_t),
               "O2 Muon optimizer requires 8-bit bytes and binary32/binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "O2 Muon optimizer requires IEEE-754 binary32/binary64");

static const double ET_MUON_PI = 3.14159265358979323846264338327950288;

/* ---- FP environment, alignment, aliasing, and finiteness helpers ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest-even, mask all exceptions, and disable FTZ/DAZ. */
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

static et_muon_status check_f32_buffer(const void *ptr, size_t count) {
  if (ptr == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (count > SIZE_MAX / sizeof(float))
    return ET_MUON_ERR_OVERFLOW;
  const size_t bytes = count * sizeof(float);
  if (!aligned_span(ptr, _Alignof(float), bytes))
    return ET_MUON_ERR_ALIGNMENT;
  return ET_MUON_OK;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static int all_finite_nonneg_f32(const float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(data[i]))
      return -1;
    if (data[i] < 0.0f)
      return 0;
  }
  return 1;
}

static int narrow_finite_f32(double value, float *out) {
  if (!isfinite(value))
    return 0;
  if (value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  const float narrowed = (float)value;
  if (!isfinite(narrowed))
    return 0;
  *out = narrowed;
  return 1;
}

static double pow_u64_f64(double base, uint64_t exp) {
  double result = 1.0;
  double factor = base;
  uint64_t e = exp;
  while (e > 0u) {
    if ((e & UINT64_C(1)) != 0u)
      result *= factor;
    e >>= 1u;
    if (e > 0u)
      factor *= factor;
  }
  return result;
}

/* ---- Core Newton-Schulz quintic orthogonalization in binary64 ---- */

static et_muon_status newton_schulz5_core_f64(uint32_t rows, uint32_t cols,
                                              const double *g_f64,
                                              uint32_t ns_steps, double eps,
                                              float *out_u) {
  const size_t n = (size_t)rows * (size_t)cols;
  double norm_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double v = g_f64[i];
    if (!isfinite(v))
      return ET_MUON_ERR_NONFINITE;
    norm_sq += v * v;
    if (!isfinite(norm_sq))
      return ET_MUON_ERR_NONFINITE;
  }
  const double norm = sqrt(norm_sq);
  const double denom = norm + eps;
  if (!isfinite(norm) || !isfinite(denom) || !(denom > 0.0))
    return ET_MUON_ERR_NONFINITE;

  const int transposed = rows > cols;
  const uint32_t r = transposed ? cols : rows;
  const uint32_t c = transposed ? rows : cols;
  const size_t rc = (size_t)r * (size_t)c;
  const size_t rr = (size_t)r * (size_t)r;

  /* Allocate workspace for X [r, c], X_next [r, c], A [r, r], B [r, r]. */
  const size_t total_doubles = (2u * rc) + (2u * rr);
  double *work = (double *)malloc(total_doubles * sizeof(double));
  if (work == NULL)
    return ET_MUON_ERR_NOMEM;

  double *x = work;
  double *x_next = x + rc;
  double *a_mat = x_next + rc;
  double *b_mat = a_mat + rr;

  /* Initialize X_0 = (G or G^T) / (||G||_F + eps). */
  if (transposed) {
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        const double val = g_f64[(size_t)i * cols + j] / denom;
        if (!isfinite(val)) {
          free(work);
          return ET_MUON_ERR_NONFINITE;
        }
        x[(size_t)j * c + i] = val;
      }
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      const double val = g_f64[i] / denom;
      if (!isfinite(val)) {
        free(work);
        return ET_MUON_ERR_NONFINITE;
      }
      x[i] = val;
    }
  }

  /* Run ns_steps quintic iterations:
   *   A_k = X_k X_k^T                        [r, r]
   *   B_k = b * A_k + c * (A_k A_k)          [r, r]
   *   X_{k+1} = a * X_k + B_k X_k            [r, c]
   */
  for (uint32_t step = 0; step < ns_steps; ++step) {
    for (uint32_t i = 0; i < r; ++i) {
      for (uint32_t j = 0; j < r; ++j) {
        double sum = 0.0;
        const double *xi = x + (size_t)i * c;
        const double *xj = x + (size_t)j * c;
        for (uint32_t k = 0; k < c; ++k)
          sum += xi[k] * xj[k];
        if (!isfinite(sum)) {
          free(work);
          return ET_MUON_ERR_NONFINITE;
        }
        a_mat[(size_t)i * r + j] = sum;
      }
    }

    for (uint32_t i = 0; i < r; ++i) {
      for (uint32_t j = 0; j < r; ++j) {
        double a2_ij = 0.0;
        const double *ai = a_mat + (size_t)i * r;
        for (uint32_t k = 0; k < r; ++k)
          a2_ij += ai[k] * a_mat[(size_t)k * r + j];
        const double bij =
            ET_MUON_NS_B * a_mat[(size_t)i * r + j] + ET_MUON_NS_C * a2_ij;
        if (!isfinite(a2_ij) || !isfinite(bij)) {
          free(work);
          return ET_MUON_ERR_NONFINITE;
        }
        b_mat[(size_t)i * r + j] = bij;
      }
    }

    for (uint32_t i = 0; i < r; ++i) {
      const double *bi = b_mat + (size_t)i * r;
      const double *xi = x + (size_t)i * c;
      double *xni = x_next + (size_t)i * c;
      for (uint32_t j = 0; j < c; ++j) {
        double bx = 0.0;
        for (uint32_t k = 0; k < r; ++k)
          bx += bi[k] * x[(size_t)k * c + j];
        const double next_val = ET_MUON_NS_A * xi[j] + bx;
        if (!isfinite(bx) || !isfinite(next_val)) {
          free(work);
          return ET_MUON_ERR_NONFINITE;
        }
        xni[j] = next_val;
      }
    }

    memcpy(x, x_next, rc * sizeof(double));
  }

  /* Stage narrowed f32 result before touching out_u (failure-atomic). */
  float *staged = (float *)malloc(n * sizeof(float));
  if (staged == NULL) {
    free(work);
    return ET_MUON_ERR_NOMEM;
  }

  if (transposed) {
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        if (!narrow_finite_f32(x[(size_t)j * c + i],
                               &staged[(size_t)i * cols + j])) {
          free(staged);
          free(work);
          return ET_MUON_ERR_NONFINITE;
        }
      }
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      if (!narrow_finite_f32(x[i], &staged[i])) {
        free(staged);
        free(work);
        return ET_MUON_ERR_NONFINITE;
      }
    }
  }

  memcpy(out_u, staged, n * sizeof(float));
  free(staged);
  free(work);
  return ET_MUON_OK;
}

/* ---- Public API: Newton-Schulz5 and Orthogonality Error ---- */

et_muon_status et_muon_newton_schulz5_raw(uint32_t rows, uint32_t cols,
                                          const void *g, uint32_t ns_steps,
                                          double eps, void *u) {
  if (!fp_environment_supported())
    return ET_MUON_ERR_FP_ENV;
  if (rows < ET_MUON_MIN_DIM || rows > ET_MUON_MAX_DIM ||
      cols < ET_MUON_MIN_DIM || cols > ET_MUON_MAX_DIM)
    return ET_MUON_ERR_SHAPE;
  if (ns_steps < ET_MUON_MIN_NS_STEPS || ns_steps > ET_MUON_MAX_NS_STEPS)
    return ET_MUON_ERR_ARGUMENT;
  if (!isfinite(eps))
    return ET_MUON_ERR_NONFINITE;
  if (!(eps > 0.0))
    return ET_MUON_ERR_ARGUMENT;

  const size_t n = (size_t)rows * (size_t)cols;
  et_muon_status st = check_f32_buffer(g, n);
  if (st != ET_MUON_OK)
    return st;
  st = check_f32_buffer(u, n);
  if (st != ET_MUON_OK)
    return st;
  if (spans_overlap(g, n * sizeof(float), u, n * sizeof(float)))
    return ET_MUON_ERR_ALIAS;

  const float *gf = (const float *)g;
  float *uf = (float *)u;
  if (!all_finite_f32(gf, n))
    return ET_MUON_ERR_NONFINITE;

  double *g_f64 = (double *)malloc(n * sizeof(double));
  if (g_f64 == NULL)
    return ET_MUON_ERR_NOMEM;
  for (size_t i = 0; i < n; ++i)
    g_f64[i] = (double)gf[i];

  st = newton_schulz5_core_f64(rows, cols, g_f64, ns_steps, eps, uf);
  free(g_f64);
  return st;
}

et_muon_status et_muon_newton_schulz5_f32(uint32_t rows, uint32_t cols,
                                          const float *g, uint32_t ns_steps,
                                          double eps, float *u) {
  return et_muon_newton_schulz5_raw(rows, cols, g, ns_steps, eps, u);
}

et_muon_status et_muon_orthogonality_error_raw(uint32_t rows, uint32_t cols,
                                               const void *u,
                                               double *out_error) {
  if (!fp_environment_supported())
    return ET_MUON_ERR_FP_ENV;
  if (out_error == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (rows < ET_MUON_MIN_DIM || rows > ET_MUON_MAX_DIM ||
      cols < ET_MUON_MIN_DIM || cols > ET_MUON_MAX_DIM)
    return ET_MUON_ERR_SHAPE;

  const size_t n = (size_t)rows * (size_t)cols;
  const et_muon_status st = check_f32_buffer(u, n);
  if (st != ET_MUON_OK)
    return st;
  if (spans_overlap(u, n * sizeof(float), out_error, sizeof(*out_error)))
    return ET_MUON_ERR_ALIAS;

  const float *uf = (const float *)u;
  if (!all_finite_f32(uf, n))
    return ET_MUON_ERR_NONFINITE;

  double err_sq = 0.0;
  if (rows <= cols) {
    /* ||U U^T - I_{rows}||_F */
    for (uint32_t i = 0; i < rows; ++i) {
      for (uint32_t j = 0; j < rows; ++j) {
        double dot = 0.0;
        for (uint32_t k = 0; k < cols; ++k)
          dot += (double)uf[(size_t)i * cols + k] *
                 (double)uf[(size_t)j * cols + k];
        const double diff = dot - (i == j ? 1.0 : 0.0);
        err_sq += diff * diff;
        if (!isfinite(err_sq))
          return ET_MUON_ERR_NONFINITE;
      }
    }
  } else {
    /* ||U^T U - I_{cols}||_F */
    for (uint32_t i = 0; i < cols; ++i) {
      for (uint32_t j = 0; j < cols; ++j) {
        double dot = 0.0;
        for (uint32_t k = 0; k < rows; ++k)
          dot += (double)uf[(size_t)k * cols + i] *
                 (double)uf[(size_t)k * cols + j];
        const double diff = dot - (i == j ? 1.0 : 0.0);
        err_sq += diff * diff;
        if (!isfinite(err_sq))
          return ET_MUON_ERR_NONFINITE;
      }
    }
  }

  const double err = sqrt(err_sq);
  if (!isfinite(err))
    return ET_MUON_ERR_NONFINITE;
  *out_error = err;
  return ET_MUON_OK;
}

et_muon_status et_muon_orthogonality_error_f32(uint32_t rows, uint32_t cols,
                                               const float *u,
                                               double *out_error) {
  return et_muon_orthogonality_error_raw(rows, cols, u, out_error);
}

/* ---- Public API: WSD (Warmup-Stable-Decay) LR Scheduler ---- */

et_muon_status et_wsd_validate_config(const et_wsd_config *cfg) {
  if (!fp_environment_supported())
    return ET_MUON_ERR_FP_ENV;
  if (cfg == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->reserved != 0u)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->decay_kind != ET_WSD_DECAY_LINEAR &&
      cfg->decay_kind != ET_WSD_DECAY_COSINE)
    return ET_MUON_ERR_ARGUMENT;
  if (!isfinite(cfg->max_lr) || !isfinite(cfg->min_lr))
    return ET_MUON_ERR_NONFINITE;
  if (!(cfg->max_lr > 0.0) || cfg->min_lr < 0.0 || cfg->min_lr > cfg->max_lr)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->total_steps == 0u)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->warmup_steps > cfg->total_steps ||
      cfg->stable_steps > cfg->total_steps - cfg->warmup_steps)
    return ET_MUON_ERR_ARGUMENT;
  return ET_MUON_OK;
}

et_muon_status et_wsd_lr(const et_wsd_config *cfg, uint64_t step,
                         double *out_lr, uint32_t *out_phase) {
  const et_muon_status st = et_wsd_validate_config(cfg);
  if (st != ET_MUON_OK)
    return st;
  if (out_lr == NULL)
    return ET_MUON_ERR_ARGUMENT;

  const uint64_t stable_end = cfg->warmup_steps + cfg->stable_steps;
  const uint64_t decay_steps = cfg->total_steps - stable_end;
  double lr = 0.0;
  uint32_t phase = ET_WSD_PHASE_DECAY;

  if (step < cfg->warmup_steps) {
    phase = ET_WSD_PHASE_WARMUP;
    /* step < warmup_steps <= total_steps <= UINT64_MAX; step + 1 cannot wrap. */
    const double ratio = (double)(step + UINT64_C(1)) / (double)cfg->warmup_steps;
    lr = cfg->max_lr * ratio;
  } else if (step < stable_end) {
    phase = ET_WSD_PHASE_STABLE;
    lr = cfg->max_lr;
  } else {
    phase = ET_WSD_PHASE_DECAY;
    if (decay_steps == 0u || step >= cfg->total_steps) {
      lr = cfg->min_lr;
    } else if (step == stable_end) {
      lr = cfg->max_lr;
    } else {
      const double t = (double)(step - stable_end) / (double)decay_steps;
      if (cfg->decay_kind == ET_WSD_DECAY_LINEAR) {
        lr = cfg->max_lr - (cfg->max_lr - cfg->min_lr) * t;
      } else {
        const double c = cos(ET_MUON_PI * t);
        lr = cfg->min_lr + 0.5 * (cfg->max_lr - cfg->min_lr) * (1.0 + c);
      }
    }
  }

  if (!isfinite(lr) || lr < 0.0)
    return ET_MUON_ERR_NONFINITE;
  *out_lr = lr;
  if (out_phase != NULL)
    *out_phase = phase;
  return ET_MUON_OK;
}

/* ---- Public API: Hybrid Muon + AdamW Step ---- */

et_muon_status et_muon_validate_step_config(const et_muon_step_config *cfg) {
  if (!fp_environment_supported())
    return ET_MUON_ERR_FP_ENV;
  if (cfg == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (!isfinite(cfg->lr) || !isfinite(cfg->muon_beta) ||
      !isfinite(cfg->muon_eps) || !isfinite(cfg->muon_weight_decay) ||
      !isfinite(cfg->adamw_beta1) || !isfinite(cfg->adamw_beta2) ||
      !isfinite(cfg->adamw_eps) || !isfinite(cfg->adamw_weight_decay) ||
      !isfinite(cfg->adamw_lr_scale))
    return ET_MUON_ERR_NONFINITE;

  if (cfg->lr < 0.0)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->step_index == 0u)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->muon_beta < 0.0 || cfg->muon_beta >= 1.0)
    return ET_MUON_ERR_ARGUMENT;
  if (!(cfg->muon_eps > 0.0))
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->muon_weight_decay < 0.0)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->muon_ns_steps < ET_MUON_MIN_NS_STEPS ||
      cfg->muon_ns_steps > ET_MUON_MAX_NS_STEPS)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->muon_nesterov != 0u && cfg->muon_nesterov != 1u)
    return ET_MUON_ERR_ARGUMENT;

  if (cfg->adamw_beta1 < 0.0 || cfg->adamw_beta1 >= 1.0)
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->adamw_beta2 < 0.0 || cfg->adamw_beta2 >= 1.0)
    return ET_MUON_ERR_ARGUMENT;
  if (!(cfg->adamw_eps > 0.0))
    return ET_MUON_ERR_ARGUMENT;
  if (cfg->adamw_weight_decay < 0.0)
    return ET_MUON_ERR_ARGUMENT;
  if (!(cfg->adamw_lr_scale > 0.0))
    return ET_MUON_ERR_ARGUMENT;

  const double eff_adamw_lr = cfg->lr * cfg->adamw_lr_scale;
  if (!isfinite(eff_adamw_lr))
    return ET_MUON_ERR_NONFINITE;

  return ET_MUON_OK;
}

typedef struct staged_group_slice {
  size_t offset;
  size_t count;
} staged_group_slice;

et_muon_status et_muon_hybrid_step(const et_muon_step_config *cfg,
                                   size_t num_groups,
                                   const et_muon_param_group *groups) {
  et_muon_status st = et_muon_validate_step_config(cfg);
  if (st != ET_MUON_OK)
    return st;
  if (num_groups == 0u || num_groups > ET_MUON_MAX_GROUPS)
    return ET_MUON_ERR_SHAPE;
  if (groups == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (!aligned_span(groups, _Alignof(et_muon_param_group),
                    num_groups * sizeof(et_muon_param_group)))
    return ET_MUON_ERR_ALIGNMENT;

  span_entry spans[(ET_MUON_MAX_GROUPS * 4u) + 2u];
  size_t span_count = 0u;
  spans[span_count++] = (span_entry){cfg, sizeof(*cfg)};
  spans[span_count++] =
      (span_entry){groups, num_groups * sizeof(et_muon_param_group)};

  staged_group_slice slices[ET_MUON_MAX_GROUPS];
  size_t total_elems = 0u;
  int has_adamw = 0;

  /* Pass 1: validate every group's shape, mask, alignment, aliasing, and inputs. */
  for (size_t g = 0; g < num_groups; ++g) {
    const et_muon_param_group *grp = &groups[g];
    if (grp->apply_weight_decay != 0u && grp->apply_weight_decay != 1u)
      return ET_MUON_ERR_ARGUMENT;

    size_t count = 0u;
    if (grp->kind == ET_MUON_GROUP_MUON_2D) {
      if (grp->rows < ET_MUON_MIN_DIM || grp->rows > ET_MUON_MAX_DIM ||
          grp->cols < ET_MUON_MIN_DIM || grp->cols > ET_MUON_MAX_DIM)
        return ET_MUON_ERR_SHAPE;
      if (grp->variance_buf != NULL)
        return ET_MUON_ERR_ARGUMENT;
      count = (size_t)grp->rows * (size_t)grp->cols;
    } else if (grp->kind == ET_MUON_GROUP_ADAMW_1D) {
      if (grp->rows < 1u || grp->rows > ET_MUON_MAX_VEC_LEN || grp->cols != 1u)
        return ET_MUON_ERR_SHAPE;
      count = (size_t)grp->rows;
      has_adamw = 1;
    } else {
      return ET_MUON_ERR_ARGUMENT;
    }

    st = check_f32_buffer(grp->params, count);
    if (st != ET_MUON_OK)
      return st;
    st = check_f32_buffer(grp->grads, count);
    if (st != ET_MUON_OK)
      return st;
    st = check_f32_buffer(grp->momentum_buf, count);
    if (st != ET_MUON_OK)
      return st;
    if (grp->kind == ET_MUON_GROUP_ADAMW_1D) {
      st = check_f32_buffer(grp->variance_buf, count);
      if (st != ET_MUON_OK)
        return st;
    }

    const size_t bytes = count * sizeof(float);
    spans[span_count++] = (span_entry){grp->params, bytes};
    spans[span_count++] = (span_entry){grp->grads, bytes};
    spans[span_count++] = (span_entry){grp->momentum_buf, bytes};
    if (grp->kind == ET_MUON_GROUP_ADAMW_1D)
      spans[span_count++] = (span_entry){grp->variance_buf, bytes};

    if (count > SIZE_MAX - total_elems)
      return ET_MUON_ERR_OVERFLOW;
    slices[g].offset = total_elems;
    slices[g].count = count;
    total_elems += count;
  }

  if (any_spans_overlap(spans, span_count))
    return ET_MUON_ERR_ALIAS;

  /* Validate all input values are finite (and AdamW variance_buf >= 0). */
  for (size_t g = 0; g < num_groups; ++g) {
    const et_muon_param_group *grp = &groups[g];
    const size_t count = slices[g].count;
    if (!all_finite_f32((const float *)grp->params, count) ||
        !all_finite_f32((const float *)grp->grads, count) ||
        !all_finite_f32((const float *)grp->momentum_buf, count))
      return ET_MUON_ERR_NONFINITE;
    if (grp->kind == ET_MUON_GROUP_ADAMW_1D) {
      const int v_ok =
          all_finite_nonneg_f32((const float *)grp->variance_buf, count);
      if (v_ok < 0)
        return ET_MUON_ERR_NONFINITE;
      if (v_ok == 0)
        return ET_MUON_ERR_ARGUMENT;
    }
  }

  /* Validate AdamW bias-correction denominators before allocating/computing. */
  double bc1 = 1.0;
  double bc2 = 1.0;
  if (has_adamw) {
    const double b1_t = pow_u64_f64(cfg->adamw_beta1, cfg->step_index);
    const double b2_t = pow_u64_f64(cfg->adamw_beta2, cfg->step_index);
    bc1 = 1.0 - b1_t;
    bc2 = 1.0 - b2_t;
    if (!isfinite(bc1) || !isfinite(bc2) || !(bc1 > 0.0) || !(bc2 > 0.0))
      return ET_MUON_ERR_NONFINITE;
  }

  /* Pass 2: stage ALL updated parameters and optimizer states off-buffer. */
  if (total_elems > SIZE_MAX / (3u * sizeof(float)))
    return ET_MUON_ERR_OVERFLOW;
  float *staging = (float *)malloc(3u * total_elems * sizeof(float));
  if (staging == NULL)
    return ET_MUON_ERR_NOMEM;

  float *staged_w = staging;
  float *staged_m = staged_w + total_elems;
  float *staged_v = staged_m + total_elems;

  double *m_in_f64 = NULL;
  float *u_tmp = NULL;
  const size_t max_2d = (size_t)ET_MUON_MAX_DIM * (size_t)ET_MUON_MAX_DIM;

  for (size_t g = 0; g < num_groups; ++g) {
    const et_muon_param_group *grp = &groups[g];
    const size_t offset = slices[g].offset;
    const size_t count = slices[g].count;
    const float *w = (const float *)grp->params;
    const float *grad = (const float *)grp->grads;
    const float *mbuf = (const float *)grp->momentum_buf;
    float *out_w = staged_w + offset;
    float *out_m = staged_m + offset;
    float *out_v = staged_v + offset;

    if (grp->kind == ET_MUON_GROUP_MUON_2D) {
      if (m_in_f64 == NULL) {
        m_in_f64 = (double *)malloc(max_2d * sizeof(double));
        u_tmp = (float *)malloc(max_2d * sizeof(float));
        if (m_in_f64 == NULL || u_tmp == NULL) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NOMEM;
        }
      }

      /* B_next = beta * B + G
       * M_in   = (nesterov ? (G + beta * B_next) : B_next)
       */
      for (size_t i = 0; i < count; ++i) {
        const double gi = (double)grad[i];
        const double b_next = cfg->muon_beta * (double)mbuf[i] + gi;
        if (!narrow_finite_f32(b_next, &out_m[i])) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
        const double m_val =
            cfg->muon_nesterov ? (gi + cfg->muon_beta * b_next) : b_next;
        if (!isfinite(m_val)) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
        m_in_f64[i] = m_val;
      }

      st = newton_schulz5_core_f64(grp->rows, grp->cols, m_in_f64,
                                   cfg->muon_ns_steps, cfg->muon_eps, u_tmp);
      if (st != ET_MUON_OK) {
        free(m_in_f64);
        free(u_tmp);
        free(staging);
        return st;
      }

      /* Aspect-ratio scaling: sqrt(max(1, rows / cols)). */
      const double aspect = (double)grp->rows / (double)grp->cols;
      const double scale = sqrt(aspect > 1.0 ? aspect : 1.0);
      if (!isfinite(scale)) {
        free(m_in_f64);
        free(u_tmp);
        free(staging);
        return ET_MUON_ERR_NONFINITE;
      }

      for (size_t i = 0; i < count; ++i) {
        const double wi = (double)w[i];
        const double ui = (double)u_tmp[i];
        double w_next = 0.0;
        if (grp->apply_weight_decay == 1u) {
          w_next = wi - cfg->lr * (cfg->muon_weight_decay * wi + scale * ui);
        } else {
          w_next = wi - cfg->lr * (scale * ui);
        }
        if (!narrow_finite_f32(w_next, &out_w[i])) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
      }
    } else {
      /* 1D deterministic AdamW with explicit apply_weight_decay mask. */
      const float *vbuf = (const float *)grp->variance_buf;
      const double eff_lr = cfg->lr * cfg->adamw_lr_scale;
      const double one_minus_b1 = 1.0 - cfg->adamw_beta1;
      const double one_minus_b2 = 1.0 - cfg->adamw_beta2;
      const double decay_factor =
          grp->apply_weight_decay == 1u
              ? (1.0 - eff_lr * cfg->adamw_weight_decay)
              : 1.0;
      if (!isfinite(decay_factor)) {
        free(m_in_f64);
        free(u_tmp);
        free(staging);
        return ET_MUON_ERR_NONFINITE;
      }

      for (size_t i = 0; i < count; ++i) {
        const double wi = (double)w[i];
        const double gi = (double)grad[i];
        const double m_next = cfg->adamw_beta1 * (double)mbuf[i] +
                              one_minus_b1 * gi;
        const double v_next = cfg->adamw_beta2 * (double)vbuf[i] +
                              one_minus_b2 * (gi * gi);
        if (!narrow_finite_f32(m_next, &out_m[i]) ||
            !narrow_finite_f32(v_next, &out_v[i]) || !(out_v[i] >= 0.0f)) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
        const double m_hat = m_next / bc1;
        const double v_hat = v_next / bc2;
        const double denom = sqrt(v_hat) + cfg->adamw_eps;
        if (!isfinite(m_hat) || !isfinite(v_hat) || !isfinite(denom) ||
            !(denom > 0.0)) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
        const double w_base =
            grp->apply_weight_decay == 1u ? (wi * decay_factor) : wi;
        const double w_next = w_base - eff_lr * (m_hat / denom);
        if (!narrow_finite_f32(w_next, &out_w[i])) {
          free(m_in_f64);
          free(u_tmp);
          free(staging);
          return ET_MUON_ERR_NONFINITE;
        }
      }
    }
  }

  /* Pass 3: non-failing, non-allocating atomic commit of all groups. */
  for (size_t g = 0; g < num_groups; ++g) {
    const et_muon_param_group *grp = &groups[g];
    const size_t offset = slices[g].offset;
    const size_t bytes = slices[g].count * sizeof(float);
    memcpy(grp->params, staged_w + offset, bytes);
    memcpy(grp->momentum_buf, staged_m + offset, bytes);
    if (grp->kind == ET_MUON_GROUP_ADAMW_1D)
      memcpy(grp->variance_buf, staged_v + offset, bytes);
  }

  free(m_in_f64);
  free(u_tmp);
  free(staging);
  return ET_MUON_OK;
}

et_muon_status et_muon_hybrid_step_wsd(const et_wsd_config *wsd,
                                       uint64_t schedule_step,
                                       const et_muon_step_config *base_cfg,
                                       size_t num_groups,
                                       const et_muon_param_group *groups,
                                       double *out_lr, uint32_t *out_phase) {
  if (wsd == NULL || base_cfg == NULL)
    return ET_MUON_ERR_ARGUMENT;
  if (schedule_step == UINT64_MAX)
    return ET_MUON_ERR_OVERFLOW;

  double scheduled_lr = 0.0;
  uint32_t phase = 0u;
  et_muon_status st = et_wsd_lr(wsd, schedule_step, &scheduled_lr, &phase);
  if (st != ET_MUON_OK)
    return st;

  et_muon_step_config step_cfg = *base_cfg;
  step_cfg.lr = scheduled_lr;
  step_cfg.step_index = schedule_step + UINT64_C(1);

  st = et_muon_hybrid_step(&step_cfg, num_groups, groups);
  if (st != ET_MUON_OK)
    return st;

  if (out_lr != NULL)
    *out_lr = scheduled_lr;
  if (out_phase != NULL)
    *out_phase = phase;
  return ET_MUON_OK;
}
