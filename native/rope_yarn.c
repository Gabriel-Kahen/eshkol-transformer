#include "eshkol_transformer/rope_yarn_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#if !defined(__x86_64__)
#error "RoPE YaRN engine requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "RoPE YaRN requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "RoPE YaRN requires standard IEEE-754 binary32/binary64 eval");
_Static_assert(ET_ROPE_YARN_MAX_HEAD_DIM <= 256u &&
                   (ET_ROPE_YARN_MAX_HEAD_DIM & 1u) == 0u,
               "RoPE YaRN max head_dim must be even and bounded for stack buffers");

#define ET_ROPE_YARN_PI 3.141592653589793238462643383279502884
#define ET_ROPE_YARN_TWO_PI (2.0 * ET_ROPE_YARN_PI)
#define ET_ROPE_YARN_MAX_HALF (ET_ROPE_YARN_MAX_HEAD_DIM / 2u)

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
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

static int fp_environment_supported(void) {
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* All exception masks set, round-to-nearest on both units, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int checked_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}

static et_rope_yarn_status validate_config_internal(
    const et_rope_yarn_config_t *cfg) {
  if (cfg == NULL)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (!aligned_span(cfg, _Alignof(et_rope_yarn_config_t), sizeof(*cfg)))
    return ET_ROPE_YARN_ERR_ALIGNMENT;
  if (cfg->mode > (uint32_t)ET_ROPE_SCALE_YARN)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (cfg->layout > (uint32_t)ET_ROPE_LAYOUT_INTERLEAVED)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (cfg->head_dim < ET_ROPE_YARN_MIN_HEAD_DIM ||
      cfg->head_dim > ET_ROPE_YARN_MAX_HEAD_DIM)
    return ET_ROPE_YARN_ERR_SHAPE;
  if (cfg->rotary_dim < ET_ROPE_YARN_MIN_HEAD_DIM ||
      cfg->rotary_dim > cfg->head_dim || (cfg->rotary_dim & 1u) != 0u)
    return ET_ROPE_YARN_ERR_SHAPE;
  if (cfg->mode == (uint32_t)ET_ROPE_SCALE_NTK_AWARE && cfg->rotary_dim < 4u)
    return ET_ROPE_YARN_ERR_SHAPE;
  if (cfg->orig_max_pos == 0u ||
      cfg->orig_max_pos > (uint64_t)ET_ROPE_YARN_MAX_POSITION)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (!isfinite(cfg->base) || !(cfg->base > 1.0))
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (!isfinite(cfg->scale_factor) || !(cfg->scale_factor >= 1.0) ||
      cfg->scale_factor > ET_ROPE_YARN_MAX_SCALE)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (cfg->mode == (uint32_t)ET_ROPE_SCALE_STANDARD &&
      cfg->scale_factor != 1.0)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (!isfinite(cfg->beta_fast) || !isfinite(cfg->beta_slow))
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (cfg->mode == (uint32_t)ET_ROPE_SCALE_YARN) {
    if (!(cfg->beta_slow > 0.0) || !(cfg->beta_fast > cfg->beta_slow))
      return ET_ROPE_YARN_ERR_ARGUMENT;
  }
  return ET_ROPE_YARN_OK;
}

static int64_t compute_max_allowed_pos(const et_rope_yarn_config_t *cfg) {
  const double prod = (double)cfg->orig_max_pos * cfg->scale_factor;
  if (!isfinite(prod) || prod >= (double)ET_ROPE_YARN_MAX_POSITION)
    return ET_ROPE_YARN_MAX_POSITION;
  const double floored = floor(prod);
  if (floored < 0.0)
    return 0;
  return (int64_t)floored;
}

/*
 * Computes binary64 inv_freq[0..rotary_dim/2 - 1] and *out_mscale into caller
 * stack storage. Assumes cfg has already passed validate_config_internal().
 */
static et_rope_yarn_status compute_freqs_internal(
    const et_rope_yarn_config_t *cfg, double *inv_freq, double *out_mscale) {
  const size_t d_rot = (size_t)cfg->rotary_dim;
  const size_t half = d_rot / 2u;
  const double base = cfg->base;
  const double s = cfg->scale_factor;

  double mscale = 1.0;
  double effective_base = base;

  if (cfg->mode == (uint32_t)ET_ROPE_SCALE_NTK_AWARE) {
    const double ntk_exp = (double)d_rot / (double)(d_rot - 2u);
    const double factor = pow(s, ntk_exp);
    effective_base = base * factor;
    if (!isfinite(factor) || !isfinite(effective_base) ||
        !(effective_base > 1.0))
      return ET_ROPE_YARN_ERR_NONFINITE;
  } else if (cfg->mode == (uint32_t)ET_ROPE_SCALE_YARN) {
    mscale = (s <= 1.0) ? 1.0 : (0.1 * log(s) + 1.0);
    if (!isfinite(mscale) || !(mscale > 0.0))
      return ET_ROPE_YARN_ERR_NONFINITE;
  }

  for (size_t i = 0u; i < half; ++i) {
    const double exponent = (-2.0 * (double)i) / (double)d_rot;
    double freq = 0.0;

    if (cfg->mode == (uint32_t)ET_ROPE_SCALE_STANDARD) {
      freq = pow(base, exponent);
    } else if (cfg->mode == (uint32_t)ET_ROPE_SCALE_LINEAR_PI) {
      const double orig = pow(base, exponent);
      if (!isfinite(orig) || !(orig > 0.0))
        return ET_ROPE_YARN_ERR_NONFINITE;
      freq = orig / s;
    } else if (cfg->mode == (uint32_t)ET_ROPE_SCALE_NTK_AWARE) {
      freq = pow(effective_base, exponent);
    } else {
      /* ET_ROPE_SCALE_YARN (Peng et al., 2023) */
      const double orig = pow(base, exponent);
      if (!isfinite(orig) || !(orig > 0.0))
        return ET_ROPE_YARN_ERR_NONFINITE;
      const double lambda = ET_ROPE_YARN_TWO_PI / orig;
      if (!isfinite(lambda) || !(lambda > 0.0))
        return ET_ROPE_YARN_ERR_NONFINITE;
      const double rotations = (double)cfg->orig_max_pos / lambda;
      const double denom = cfg->beta_fast - cfg->beta_slow;
      const double raw_gamma = (rotations - cfg->beta_slow) / denom;
      if (!isfinite(rotations) || !isfinite(raw_gamma))
        return ET_ROPE_YARN_ERR_NONFINITE;
      double gamma = raw_gamma;
      if (gamma < 0.0)
        gamma = 0.0;
      else if (gamma > 1.0)
        gamma = 1.0;
      const double interp = orig / s;
      freq = (1.0 - gamma) * interp + gamma * orig;
    }

    if (!isfinite(freq) || !(freq > 0.0))
      return ET_ROPE_YARN_ERR_NONFINITE;
    inv_freq[i] = freq;
  }

  *out_mscale = mscale;
  return ET_ROPE_YARN_OK;
}

static int round_finite_f32(double value, float *out) {
  if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  const float f = (float)value;
  if (!isfinite(f))
    return 0;
  *out = f;
  return 1;
}

/*
 * Evaluates RoPE forward (backward == 0) or backward (backward == 1) over
 * [batch, tokens, heads, head_dim]. When out == NULL, runs a strict dry-run
 * validation pass touching zero output bytes.
 */
static et_rope_yarn_status evaluate_rope_pass(
    const et_rope_yarn_config_t *cfg, const double *inv_freq, double mscale,
    int64_t max_pos, size_t batch, size_t tokens, size_t heads, const float *in,
    const int64_t *pos_ids, int backward, float *out) {
  const size_t dh = (size_t)cfg->head_dim;
  const size_t d_rot = (size_t)cfg->rotary_dim;
  const size_t half = d_rot / 2u;
  const int interleaved =
      (cfg->layout == (uint32_t)ET_ROPE_LAYOUT_INTERLEAVED);

  double cos_m[ET_ROPE_YARN_MAX_HALF];
  double sin_m[ET_ROPE_YARN_MAX_HALF];

  for (size_t n = 0u; n < batch; ++n) {
    for (size_t t = 0u; t < tokens; ++t) {
      const int64_t pos = pos_ids[n * tokens + t];
      if (pos < 0 || pos > max_pos)
        return ET_ROPE_YARN_ERR_POSITION;

      const double pos_d = (double)pos;
      for (size_t k = 0u; k < half; ++k) {
        const double phi = pos_d * inv_freq[k];
        if (!isfinite(phi))
          return ET_ROPE_YARN_ERR_NONFINITE;
        const double cm = mscale * cos(phi);
        const double sm = mscale * sin(phi);
        if (!isfinite(cm) || !isfinite(sm))
          return ET_ROPE_YARN_ERR_NONFINITE;
        cos_m[k] = cm;
        sin_m[k] = sm;
      }

      const size_t nt_base = (n * tokens + t) * heads;
      for (size_t h = 0u; h < heads; ++h) {
        const size_t base = (nt_base + h) * dh;
        for (size_t k = 0u; k < half; ++k) {
          const size_t i0 = interleaved ? (base + 2u * k) : (base + k);
          const size_t i1 =
              interleaved ? (base + 2u * k + 1u) : (base + half + k);
          const float f0 = in[i0];
          const float f1 = in[i1];
          if (!isfinite(f0) || !isfinite(f1))
            return ET_ROPE_YARN_ERR_NONFINITE;
          const double v0 = (double)f0;
          const double v1 = (double)f1;
          const double cm = cos_m[k];
          const double sm = sin_m[k];
          const double r0 = backward ? (v0 * cm + v1 * sm)
                                     : (v0 * cm - v1 * sm);
          const double r1 = backward ? (-v0 * sm + v1 * cm)
                                     : (v0 * sm + v1 * cm);
          float o0 = 0.0f;
          float o1 = 0.0f;
          if (!round_finite_f32(r0, &o0) || !round_finite_f32(r1, &o1))
            return ET_ROPE_YARN_ERR_NONFINITE;
          if (out != NULL) {
            out[i0] = o0;
            out[i1] = o1;
          }
        }
        /* Partial rotary: pass trailing [d_rot .. dh - 1] untouched. */
        for (size_t d = d_rot; d < dh; ++d) {
          const float v = in[base + d];
          if (!isfinite(v))
            return ET_ROPE_YARN_ERR_NONFINITE;
          if (out != NULL)
            out[base + d] = v;
        }
      }
    }
  }
  return ET_ROPE_YARN_OK;
}

static et_rope_yarn_status run_rope_raw(const et_rope_yarn_config_t *cfg,
                                        size_t batch, size_t tokens,
                                        size_t heads, const void *in,
                                        const void *pos_ids, int backward,
                                        void *out) {
  const et_rope_yarn_status cfg_status = validate_config_internal(cfg);
  if (cfg_status != ET_ROPE_YARN_OK)
    return cfg_status;

  if (batch < ET_ROPE_YARN_MIN_BATCH || batch > ET_ROPE_YARN_MAX_BATCH ||
      tokens < ET_ROPE_YARN_MIN_TOKENS || tokens > ET_ROPE_YARN_MAX_TOKENS ||
      heads < ET_ROPE_YARN_MIN_HEADS || heads > ET_ROPE_YARN_MAX_HEADS)
    return ET_ROPE_YARN_ERR_SHAPE;

  if (in == NULL || pos_ids == NULL || out == NULL)
    return ET_ROPE_YARN_ERR_ARGUMENT;

  size_t nt = 0u;
  size_t nth = 0u;
  size_t elems = 0u;
  size_t data_bytes = 0u;
  size_t pos_bytes = 0u;
  if (!checked_mul_size(batch, tokens, &nt) ||
      !checked_mul_size(nt, heads, &nth) ||
      !checked_mul_size(nth, (size_t)cfg->head_dim, &elems) ||
      !checked_mul_size(elems, sizeof(float), &data_bytes) ||
      !checked_mul_size(nt, sizeof(int64_t), &pos_bytes))
    return ET_ROPE_YARN_ERR_OVERFLOW;

  if (!aligned_span(in, _Alignof(float), data_bytes) ||
      !aligned_span(pos_ids, _Alignof(int64_t), pos_bytes) ||
      !aligned_span(out, _Alignof(float), data_bytes))
    return ET_ROPE_YARN_ERR_ALIGNMENT;

  /* Pairwise disjointness across all live spans. */
  if (spans_overlap(in, data_bytes, pos_ids, pos_bytes) ||
      spans_overlap(out, data_bytes, in, data_bytes) ||
      spans_overlap(out, data_bytes, pos_ids, pos_bytes) ||
      spans_overlap(out, data_bytes, cfg, sizeof(*cfg)) ||
      spans_overlap(in, data_bytes, cfg, sizeof(*cfg)) ||
      spans_overlap(pos_ids, pos_bytes, cfg, sizeof(*cfg)))
    return ET_ROPE_YARN_ERR_ALIAS;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return ET_ROPE_YARN_ERR_FP_ENV;
  }

  double inv_freq[ET_ROPE_YARN_MAX_HALF];
  double mscale = 1.0;
  et_rope_yarn_status status = compute_freqs_internal(cfg, inv_freq, &mscale);
  if (status == ET_ROPE_YARN_OK) {
    const int64_t max_pos = compute_max_allowed_pos(cfg);
    /* Failure-atomic preflight pass: validates every input & output before write. */
    status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, batch, tokens,
                                heads, (const float *)in,
                                (const int64_t *)pos_ids, backward, NULL);
    if (status == ET_ROPE_YARN_OK) {
      status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, batch, tokens,
                                  heads, (const float *)in,
                                  (const int64_t *)pos_ids, backward,
                                  (float *)out);
    }
  }

  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return ET_ROPE_YARN_ERR_FP_ENV;
  return status;
}

et_rope_yarn_status et_rope_yarn_validate_config_v1(
    const et_rope_yarn_config_t *cfg) {
  const et_rope_yarn_status status = validate_config_internal(cfg);
  if (status != ET_ROPE_YARN_OK)
    return status;
  if (!fp_environment_supported())
    return ET_ROPE_YARN_ERR_FP_ENV;
  return ET_ROPE_YARN_OK;
}

et_rope_yarn_status et_rope_yarn_compute_freqs_raw_v1(
    const et_rope_yarn_config_t *cfg, void *out_inv_freq, size_t inv_freq_len,
    void *out_mscale) {
  const et_rope_yarn_status cfg_status = validate_config_internal(cfg);
  if (cfg_status != ET_ROPE_YARN_OK)
    return cfg_status;

  if (out_inv_freq == NULL || out_mscale == NULL)
    return ET_ROPE_YARN_ERR_ARGUMENT;

  const size_t half = (size_t)cfg->rotary_dim / 2u;
  if (inv_freq_len != half)
    return ET_ROPE_YARN_ERR_SHAPE;

  const size_t freq_bytes = half * sizeof(double);
  if (!aligned_span(out_inv_freq, _Alignof(double), freq_bytes) ||
      !aligned_span(out_mscale, _Alignof(double), sizeof(double)))
    return ET_ROPE_YARN_ERR_ALIGNMENT;

  if (spans_overlap(out_inv_freq, freq_bytes, out_mscale, sizeof(double)) ||
      spans_overlap(out_inv_freq, freq_bytes, cfg, sizeof(*cfg)) ||
      spans_overlap(out_mscale, sizeof(double), cfg, sizeof(*cfg)))
    return ET_ROPE_YARN_ERR_ALIAS;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return ET_ROPE_YARN_ERR_FP_ENV;
  }

  double staged_freq[ET_ROPE_YARN_MAX_HALF];
  double staged_mscale = 1.0;
  const et_rope_yarn_status status =
      compute_freqs_internal(cfg, staged_freq, &staged_mscale);
  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return ET_ROPE_YARN_ERR_FP_ENV;
  if (status != ET_ROPE_YARN_OK)
    return status;

  memcpy(out_inv_freq, staged_freq, freq_bytes);
  memcpy(out_mscale, &staged_mscale, sizeof(staged_mscale));
  return ET_ROPE_YARN_OK;
}

et_rope_yarn_status et_rope_yarn_compute_freqs_v1(
    const et_rope_yarn_config_t *cfg, double *out_inv_freq, size_t inv_freq_len,
    double *out_mscale) {
  return et_rope_yarn_compute_freqs_raw_v1(cfg, out_inv_freq, inv_freq_len,
                                           out_mscale);
}

et_rope_yarn_status et_rope_yarn_forward_raw_v1(
    const et_rope_yarn_config_t *cfg, size_t batch, size_t tokens, size_t heads,
    const void *x, const void *pos_ids, void *y) {
  return run_rope_raw(cfg, batch, tokens, heads, x, pos_ids, 0, y);
}

et_rope_yarn_status et_rope_yarn_forward_v1(const et_rope_yarn_config_t *cfg,
                                            size_t batch, size_t tokens,
                                            size_t heads, const float *x,
                                            const int64_t *pos_ids, float *y) {
  return run_rope_raw(cfg, batch, tokens, heads, x, pos_ids, 0, y);
}

et_rope_yarn_status et_rope_yarn_backward_raw_v1(
    const et_rope_yarn_config_t *cfg, size_t batch, size_t tokens, size_t heads,
    const void *dy, const void *pos_ids, void *dx) {
  return run_rope_raw(cfg, batch, tokens, heads, dy, pos_ids, 1, dx);
}

et_rope_yarn_status et_rope_yarn_backward_v1(const et_rope_yarn_config_t *cfg,
                                             size_t batch, size_t tokens,
                                             size_t heads, const float *dy,
                                             const int64_t *pos_ids,
                                             float *dx) {
  return run_rope_raw(cfg, batch, tokens, heads, dy, pos_ids, 1, dx);
}

et_rope_yarn_status et_rope_yarn_qk_dot_invariance_check_raw_v1(
    const et_rope_yarn_config_t *cfg, const void *q, const void *k,
    int64_t pos_m, int64_t pos_n, int64_t shift, void *out_result) {
  const et_rope_yarn_status cfg_status = validate_config_internal(cfg);
  if (cfg_status != ET_ROPE_YARN_OK)
    return cfg_status;

  if (q == NULL || k == NULL || out_result == NULL)
    return ET_ROPE_YARN_ERR_ARGUMENT;

  const size_t dh = (size_t)cfg->head_dim;
  const size_t d_rot = (size_t)cfg->rotary_dim;
  const size_t half = d_rot / 2u;
  const size_t vec_bytes = dh * sizeof(float);

  if (!aligned_span(q, _Alignof(float), vec_bytes) ||
      !aligned_span(k, _Alignof(float), vec_bytes) ||
      !aligned_span(out_result, _Alignof(et_rope_yarn_invariance_result_t),
                    sizeof(et_rope_yarn_invariance_result_t)))
    return ET_ROPE_YARN_ERR_ALIGNMENT;

  if (spans_overlap(out_result, sizeof(et_rope_yarn_invariance_result_t), q,
                    vec_bytes) ||
      spans_overlap(out_result, sizeof(et_rope_yarn_invariance_result_t), k,
                    vec_bytes) ||
      spans_overlap(out_result, sizeof(et_rope_yarn_invariance_result_t), cfg,
                    sizeof(*cfg)))
    return ET_ROPE_YARN_ERR_ALIAS;

  const int64_t max_pos = compute_max_allowed_pos(cfg);
  if (pos_m < 0 || pos_n < 0 || shift < 0 || pos_m > max_pos ||
      pos_n > max_pos || shift > max_pos - pos_m || shift > max_pos - pos_n)
    return ET_ROPE_YARN_ERR_POSITION;

  const int saved_errno = errno;
  fenv_t environment;
  if (!fp_environment_supported() || fegetenv(&environment) != 0) {
    errno = saved_errno;
    return ET_ROPE_YARN_ERR_FP_ENV;
  }

  double inv_freq[ET_ROPE_YARN_MAX_HALF];
  double mscale = 1.0;
  et_rope_yarn_status status = compute_freqs_internal(cfg, inv_freq, &mscale);
  et_rope_yarn_invariance_result_t res;
  memset(&res, 0, sizeof(res));

  if (status == ET_ROPE_YARN_OK) {
    const float *qf = (const float *)q;
    const float *kf = (const float *)k;
    float q_rot_mn[ET_ROPE_YARN_MAX_HEAD_DIM];
    float k_rot_mn[ET_ROPE_YARN_MAX_HEAD_DIM];
    float q_rot_sh[ET_ROPE_YARN_MAX_HEAD_DIM];
    float k_rot_sh[ET_ROPE_YARN_MAX_HEAD_DIM];

    const int64_t m_sh = pos_m + shift;
    const int64_t n_sh = pos_n + shift;

    status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, 1u, 1u, 1u, qf,
                                &pos_m, 0, q_rot_mn);
    if (status == ET_ROPE_YARN_OK)
      status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, 1u, 1u, 1u,
                                  kf, &pos_n, 0, k_rot_mn);
    if (status == ET_ROPE_YARN_OK)
      status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, 1u, 1u, 1u,
                                  qf, &m_sh, 0, q_rot_sh);
    if (status == ET_ROPE_YARN_OK)
      status = evaluate_rope_pass(cfg, inv_freq, mscale, max_pos, 1u, 1u, 1u,
                                  kf, &n_sh, 0, k_rot_sh);

    if (status == ET_ROPE_YARN_OK) {
      double dot_mn = 0.0;
      double dot_sh = 0.0;
      for (size_t d = 0u; d < dh; ++d) {
        dot_mn += (double)q_rot_mn[d] * (double)k_rot_mn[d];
        dot_sh += (double)q_rot_sh[d] * (double)k_rot_sh[d];
      }

      /* Exact analytical relative-position dot product in binary64:
       *   mscale^2 * <R(m - n) q_rot, k_rot> + <q_tail, k_tail> */
      const int interleaved =
          (cfg->layout == (uint32_t)ET_ROPE_LAYOUT_INTERLEAVED);
      const double rel = (double)(pos_m - pos_n);
      double rot_dot = 0.0;
      for (size_t i = 0u; i < half; ++i) {
        const size_t i0 = interleaved ? (2u * i) : i;
        const size_t i1 = interleaved ? (2u * i + 1u) : (half + i);
        const double phi = rel * inv_freq[i];
        const double c = cos(phi);
        const double s = sin(phi);
        const double q0 = (double)qf[i0];
        const double q1 = (double)qf[i1];
        const double k0 = (double)kf[i0];
        const double k1 = (double)kf[i1];
        const double rq0 = q0 * c - q1 * s;
        const double rq1 = q0 * s + q1 * c;
        rot_dot += rq0 * k0 + rq1 * k1;
      }
      double tail_dot = 0.0;
      for (size_t d = d_rot; d < dh; ++d)
        tail_dot += (double)qf[d] * (double)kf[d];

      const double dot_rel = (mscale * mscale) * rot_dot + tail_dot;
      const double shift_err = fabs(dot_mn - dot_sh);
      const double rel_err = fabs(dot_mn - dot_rel);

      if (!isfinite(dot_mn) || !isfinite(dot_sh) || !isfinite(dot_rel) ||
          !isfinite(shift_err) || !isfinite(rel_err)) {
        status = ET_ROPE_YARN_ERR_NONFINITE;
      } else {
        res.dot_mn = dot_mn;
        res.dot_shifted = dot_sh;
        res.dot_relative_ref = dot_rel;
        res.shift_abs_err = shift_err;
        res.relative_abs_err = rel_err;
        res.mscale = mscale;
      }
    }
  }

  const int restored = fesetenv(&environment);
  errno = saved_errno;
  if (restored != 0)
    return ET_ROPE_YARN_ERR_FP_ENV;
  if (status != ET_ROPE_YARN_OK)
    return status;

  memcpy(out_result, &res, sizeof(res));
  return ET_ROPE_YARN_OK;
}

et_rope_yarn_status et_rope_yarn_qk_dot_invariance_check_v1(
    const et_rope_yarn_config_t *cfg, const float *q, const float *k,
    int64_t pos_m, int64_t pos_n, int64_t shift,
    et_rope_yarn_invariance_result_t *out_result) {
  return et_rope_yarn_qk_dot_invariance_check_raw_v1(cfg, q, k, pos_m, pos_n,
                                                     shift, out_result);
}
