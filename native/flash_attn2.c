#include "eshkol_transformer/flash_attn2_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#if !defined(__x86_64__)
#error "FlashAttention-2 v1 requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "FlashAttention-2 requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   FLT_EVAL_METHOD == 0,
               "FlashAttention-2 requires standard IEEE-754 eval");

#define FA2_MAX_T ET_FA2_MAX_SEQ_LEN
#define FA2_MAX_DH ET_FA2_MAX_HEAD_DIM
#define FA2_MAX_TILE ET_FA2_MAX_TILE

typedef struct byte_span {
  const void *data;
  size_t bytes;
} byte_span;

static int tile_size_supported(size_t tile) {
  return tile == 8u || tile == 16u || tile == 32u;
}

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
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
  for (size_t i = 0u; i < out_count; ++i) {
    if (outputs[i].data == NULL)
      continue;
    for (size_t j = i + 1u; j < out_count; ++j) {
      if (outputs[j].data == NULL)
        continue;
      if (spans_overlap(outputs[i].data, outputs[i].bytes, outputs[j].data,
                        outputs[j].bytes))
        return 0;
    }
    for (size_t j = 0u; j < in_count; ++j) {
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
  /* All exception masks set, round-to-nearest on x87 & SSE, no FTZ or DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
}

static int scan_f32_finite_max(const float *buf, size_t count,
                               double *max_abs_out) {
  double max_abs = 0.0;
  for (size_t i = 0u; i < count; ++i) {
    const float v = buf[i];
    if (!isfinite(v))
      return 0;
    const double av = v < 0.0f ? -(double)v : (double)v;
    if (av > max_abs)
      max_abs = av;
  }
  *max_abs_out = max_abs;
  return 1;
}

static int scan_f64_finite_minmax(const double *buf, size_t count,
                                  double *min_out, double *max_out) {
  double lo = INFINITY;
  double hi = -INFINITY;
  for (size_t i = 0u; i < count; ++i) {
    const double v = buf[i];
    if (!isfinite(v))
      return 0;
    if (v < lo)
      lo = v;
    if (v > hi)
      hi = v;
  }
  *min_out = lo;
  *max_out = hi;
  return 1;
}

static int finite_f64_to_f32(double value, float *out) {
  if (!isfinite(value))
    return 0;
  if (value > (double)FLT_MAX || value < -(double)FLT_MAX)
    return 0;
  const float cast_v = (float)value;
  if (!isfinite(cast_v))
    return 0;
  *out = cast_v;
  return 1;
}

static et_fa2_status_t read_config(const void *cfg_ptr, et_fa2_config_v1 *out,
                                   double *effective_scale_out) {
  if (!aligned_span(cfg_ptr, _Alignof(et_fa2_config_v1),
                    sizeof(et_fa2_config_v1)))
    return ET_FA2_ERR_INVALID_ARG;

  et_fa2_config_v1 cfg;
  memcpy(&cfg, cfg_ptr, sizeof(cfg));

  if (cfg.struct_size != sizeof(et_fa2_config_v1))
    return ET_FA2_ERR_INVALID_ARG;
  if (cfg.reserved != 0u)
    return ET_FA2_ERR_INVALID_ARG;
  if (cfg.causal > 1u)
    return ET_FA2_ERR_INVALID_ARG;
  if (!isfinite(cfg.scale) || cfg.scale < 0.0)
    return ET_FA2_ERR_INVALID_ARG;
  if (!isfinite(cfg.logit_softcap) || cfg.logit_softcap < 0.0)
    return ET_FA2_ERR_INVALID_ARG;

  if (cfg.batch < ET_FA2_MIN_BATCH || cfg.batch > ET_FA2_MAX_BATCH ||
      cfg.num_heads < ET_FA2_MIN_HEADS || cfg.num_heads > ET_FA2_MAX_HEADS ||
      cfg.num_kv_heads < ET_FA2_MIN_HEADS ||
      cfg.num_kv_heads > ET_FA2_MAX_HEADS ||
      cfg.seq_q < ET_FA2_MIN_SEQ_LEN || cfg.seq_q > ET_FA2_MAX_SEQ_LEN ||
      cfg.seq_k < ET_FA2_MIN_SEQ_LEN || cfg.seq_k > ET_FA2_MAX_SEQ_LEN ||
      cfg.head_dim < ET_FA2_MIN_HEAD_DIM ||
      cfg.head_dim > ET_FA2_MAX_HEAD_DIM ||
      !tile_size_supported(cfg.tile_r) || !tile_size_supported(cfg.tile_c))
    return ET_FA2_ERR_UNSUPPORTED;

  if (cfg.num_kv_heads > cfg.num_heads ||
      (cfg.num_heads % cfg.num_kv_heads) != 0u)
    return ET_FA2_ERR_SHAPE_MISMATCH;

  /*
   * Every query row q_pos in [0, seq_q - 1] must have at least one unmasked
   * key k_pos in [0, seq_k - 1]. For sliding_window >= 1, the last query row
   * q_pos = seq_q - 1 requires (seq_q - 1) - (seq_k - 1) < sliding_window.
   */
  if (cfg.sliding_window > 0u && cfg.seq_q > cfg.seq_k &&
      (cfg.seq_q - cfg.seq_k) >= cfg.sliding_window)
    return ET_FA2_ERR_SHAPE_MISMATCH;

  const double eff_scale =
      (cfg.scale == 0.0) ? (1.0 / sqrt((double)cfg.head_dim)) : cfg.scale;
  if (!isfinite(eff_scale) || eff_scale <= 0.0)
    return ET_FA2_ERR_INVALID_ARG;

  *out = cfg;
  if (effective_scale_out != NULL)
    *effective_scale_out = eff_scale;
  return ET_FA2_OK;
}

static int pair_allowed(size_t q_pos, size_t k_pos, uint32_t causal,
                        size_t sliding_window) {
  if (causal != 0u) {
    if (k_pos > q_pos)
      return 0;
    if (sliding_window > 0u && (q_pos - k_pos) >= sliding_window)
      return 0;
    return 1;
  }
  if (sliding_window > 0u) {
    const size_t dist = (q_pos >= k_pos) ? (q_pos - k_pos) : (k_pos - q_pos);
    if (dist >= sliding_window)
      return 0;
  }
  return 1;
}

/*
 * Exact necessary-and-sufficient tile masking predicate for query interval
 * [q_start, q_end) and key interval [k_start, k_end). Returns 1 iff EVERY
 * (q_pos, k_pos) pair in the tile is masked out.
 */
static int tile_is_masked(size_t q_start, size_t q_end, size_t k_start,
                          size_t k_end, uint32_t causal,
                          size_t sliding_window) {
  const size_t q_last = q_end - 1u;
  const size_t k_last = k_end - 1u;
  if (causal != 0u) {
    if (k_start > q_last)
      return 1;
    if (sliding_window > 0u && q_start > k_last &&
        (q_start - k_last) >= sliding_window)
      return 1;
    return 0;
  }
  if (sliding_window > 0u) {
    if (k_start > q_last && (k_start - q_last) >= sliding_window)
      return 1;
    if (q_start > k_last && (q_start - k_last) >= sliding_window)
      return 1;
  }
  return 0;
}

size_t et_fa2_tile_scratch_bytes_v1(size_t tile_r, size_t tile_c) {
  if (!tile_size_supported(tile_r) || !tile_size_supported(tile_c))
    return 0u;
  /*
   * Two binary64 tile matrices of shape [B_r, B_c]: score/prob tile and dS
   * tile. Strictly O(B_r * B_c) and completely independent of T_q and T_k.
   */
  return 2u * tile_r * tile_c * sizeof(double);
}

et_fa2_status_t et_fa2_validate_config_v1(const void *cfg_ptr,
                                          double *effective_scale_out) {
  if (!fp_environment_supported())
    return ET_FA2_ERR_FP_ENV;
  if (effective_scale_out != NULL) {
    if (!aligned_span(effective_scale_out, _Alignof(double), sizeof(double)))
      return ET_FA2_ERR_INVALID_ARG;
    if (spans_overlap(effective_scale_out, sizeof(double), cfg_ptr,
                      sizeof(et_fa2_config_v1)))
      return ET_FA2_ERR_INVALID_ARG;
  }
  et_fa2_config_v1 cfg;
  double eff_scale = 0.0;
  const et_fa2_status_t st = read_config(cfg_ptr, &cfg, &eff_scale);
  if (st != ET_FA2_OK)
    return st;
  if (effective_scale_out != NULL)
    *effective_scale_out = eff_scale;
  return ET_FA2_OK;
}

/*
 * Core tiled FlashAttention-2 forward kernel.
 * When commit == 0, performs an identical numeric dry-run verifying every
 * intermediate score, LSE, and f32 output without modifying o_out or lse_out.
 */
static et_fa2_status_t fa2_forward_exec(const et_fa2_config_v1 *cfg,
                                        double scale, const float *qf,
                                        const float *kf, const float *vf,
                                        float *of, double *lsef, int commit,
                                        et_fa2_stats_v1 *stats_out) {
  const size_t B = cfg->batch;
  const size_t Hq = cfg->num_heads;
  const size_t Hkv = cfg->num_kv_heads;
  const size_t group = Hq / Hkv;
  const size_t Tq = cfg->seq_q;
  const size_t Tk = cfg->seq_k;
  const size_t Dh = cfg->head_dim;
  const size_t Br = cfg->tile_r;
  const size_t Bc = cfg->tile_c;
  const uint32_t causal = cfg->causal;
  const size_t window = cfg->sliding_window;
  const double softcap = cfg->logit_softcap;

  const uint64_t q_tiles = (uint64_t)((Tq + Br - 1u) / Br);
  const uint64_t k_tiles = (uint64_t)((Tk + Bc - 1u) / Bc);
  uint64_t evaluated = 0u;
  uint64_t skipped = 0u;

  /* Strictly O(B_r * B_c + B_r * D_h) stack scratch, independent of T_q, T_k. */
  double m_row[FA2_MAX_TILE];
  double l_row[FA2_MAX_TILE];
  double o_acc[FA2_MAX_TILE * FA2_MAX_DH];
  double s_tile[FA2_MAX_TILE * FA2_MAX_TILE];

  for (size_t b = 0u; b < B; ++b) {
    for (size_t h = 0u; h < Hq; ++h) {
      const size_t hkv = h / group;
      const float *q_head = qf + ((b * Hq + h) * Tq) * Dh;
      const float *k_head = kf + ((b * Hkv + hkv) * Tk) * Dh;
      const float *v_head = vf + ((b * Hkv + hkv) * Tk) * Dh;
      float *o_head = of + ((b * Hq + h) * Tq) * Dh;
      double *lse_head = lsef + (b * Hq + h) * Tq;

      for (size_t q_start = 0u; q_start < Tq; q_start += Br) {
        const size_t q_len = (Tq - q_start < Br) ? (Tq - q_start) : Br;
        const size_t q_end = q_start + q_len;

        for (size_t r = 0u; r < q_len; ++r) {
          m_row[r] = -INFINITY;
          l_row[r] = 0.0;
        }
        memset(o_acc, 0, q_len * Dh * sizeof(double));

        for (size_t k_start = 0u; k_start < Tk; k_start += Bc) {
          const size_t k_len = (Tk - k_start < Bc) ? (Tk - k_start) : Bc;
          const size_t k_end = k_start + k_len;

          if (tile_is_masked(q_start, q_end, k_start, k_end, causal, window)) {
            ++skipped;
            continue;
          }
          ++evaluated;

          for (size_t r = 0u; r < q_len; ++r) {
            const size_t q_pos = q_start + r;
            const float *q_vec = q_head + q_pos * Dh;
            double m_tile = -INFINITY;
            int has_valid = 0;

            for (size_t c = 0u; c < k_len; ++c) {
              const size_t k_pos = k_start + c;
              if (!pair_allowed(q_pos, k_pos, causal, window)) {
                s_tile[r * Bc + c] = -INFINITY;
                continue;
              }
              const float *k_vec = k_head + k_pos * Dh;
              double dot = 0.0;
              for (size_t d = 0u; d < Dh; ++d)
                dot += (double)q_vec[d] * (double)k_vec[d];
              const double s_raw = dot * scale;
              if (!isfinite(s_raw))
                return ET_FA2_ERR_NONFINITE;
              const double s_eff =
                  (softcap > 0.0) ? (softcap * tanh(s_raw / softcap)) : s_raw;
              if (!isfinite(s_eff))
                return ET_FA2_ERR_NONFINITE;
              s_tile[r * Bc + c] = s_eff;
              if (!has_valid || s_eff > m_tile) {
                m_tile = s_eff;
                has_valid = 1;
              }
            }

            if (!has_valid)
              continue;

            const double m_old = m_row[r];
            const double l_old = l_row[r];
            const double m_new =
                (l_old == 0.0) ? m_tile : (m_old > m_tile ? m_old : m_tile);
            const double alpha = (l_old == 0.0) ? 0.0 : exp(m_old - m_new);
            double l_new = alpha * l_old;
            double *o_row = o_acc + r * Dh;
            if (l_old != 0.0) {
              for (size_t d = 0u; d < Dh; ++d)
                o_row[d] *= alpha;
            }

            for (size_t c = 0u; c < k_len; ++c) {
              const double s_eff = s_tile[r * Bc + c];
              if (s_eff == -INFINITY)
                continue;
              const double p = exp(s_eff - m_new);
              l_new += p;
              const float *v_vec = v_head + (k_start + c) * Dh;
              for (size_t d = 0u; d < Dh; ++d)
                o_row[d] += p * (double)v_vec[d];
            }

            if (!isfinite(l_new) || l_new <= 0.0)
              return ET_FA2_ERR_NONFINITE;
            m_row[r] = m_new;
            l_row[r] = l_new;
          }
        }

        for (size_t r = 0u; r < q_len; ++r) {
          const double l_val = l_row[r];
          if (!isfinite(l_val) || l_val <= 0.0 || !isfinite(m_row[r]))
            return ET_FA2_ERR_NONFINITE;
          const double lse_val = m_row[r] + log(l_val);
          if (!isfinite(lse_val))
            return ET_FA2_ERR_NONFINITE;
          const double *o_row = o_acc + r * Dh;
          float *o_dst = o_head + (q_start + r) * Dh;
          for (size_t d = 0u; d < Dh; ++d) {
            float o_f32 = 0.0f;
            if (!finite_f64_to_f32(o_row[d] / l_val, &o_f32))
              return ET_FA2_ERR_NONFINITE;
            if (commit)
              o_dst[d] = o_f32;
          }
          if (commit)
            lse_head[q_start + r] = lse_val;
        }
      }
    }
  }

  if (stats_out != NULL) {
    stats_out->total_tiles = (uint64_t)(B * Hq) * q_tiles * k_tiles;
    stats_out->evaluated_tiles = evaluated;
    stats_out->skipped_tiles = skipped;
    stats_out->q_tiles_per_head = q_tiles;
    stats_out->k_tiles_per_head = k_tiles;
    stats_out->tile_scratch_bytes =
        (uint64_t)et_fa2_tile_scratch_bytes_v1(Br, Bc);
  }
  return ET_FA2_OK;
}

et_fa2_status_t et_fa2_forward_v1(const void *cfg_ptr, const void *q,
                                  const void *k, const void *v, void *o_out,
                                  void *lse_out, et_fa2_stats_v1 *stats_out) {
  if (!fp_environment_supported())
    return ET_FA2_ERR_FP_ENV;

  et_fa2_config_v1 cfg;
  double scale = 0.0;
  const et_fa2_status_t cfg_st = read_config(cfg_ptr, &cfg, &scale);
  if (cfg_st != ET_FA2_OK)
    return cfg_st;

  const size_t q_elems = cfg.batch * cfg.num_heads * cfg.seq_q * cfg.head_dim;
  const size_t kv_elems =
      cfg.batch * cfg.num_kv_heads * cfg.seq_k * cfg.head_dim;
  const size_t lse_elems = cfg.batch * cfg.num_heads * cfg.seq_q;

  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q, _Alignof(float), q_bytes) ||
      !aligned_span(k, _Alignof(float), kv_bytes) ||
      !aligned_span(v, _Alignof(float), kv_bytes) ||
      !aligned_span(o_out, _Alignof(float), q_bytes) ||
      !aligned_span(lse_out, _Alignof(double), lse_bytes))
    return ET_FA2_ERR_INVALID_ARG;
  if (stats_out != NULL &&
      !aligned_span(stats_out, _Alignof(et_fa2_stats_v1),
                    sizeof(et_fa2_stats_v1)))
    return ET_FA2_ERR_INVALID_ARG;

  const byte_span outputs[3] = {
      {o_out, q_bytes},
      {lse_out, lse_bytes},
      {stats_out, stats_out != NULL ? sizeof(et_fa2_stats_v1) : 0u}};
  const byte_span inputs[4] = {{cfg_ptr, sizeof(et_fa2_config_v1)},
                               {q, q_bytes},
                               {k, kv_bytes},
                               {v, kv_bytes}};
  if (!outputs_disjoint_from_all(outputs, 3u, inputs, 4u))
    return ET_FA2_ERR_INVALID_ARG;

  const float *qf = (const float *)q;
  const float *kf = (const float *)k;
  const float *vf = (const float *)v;
  double max_q = 0.0, max_k = 0.0, max_v = 0.0;
  if (!scan_f32_finite_max(qf, q_elems, &max_q) ||
      !scan_f32_finite_max(kf, kv_elems, &max_k) ||
      !scan_f32_finite_max(vf, kv_elems, &max_v))
    return ET_FA2_ERR_NONFINITE;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0)
    return ET_FA2_ERR_FP_ENV;

  /*
   * Analytically safe range where no intermediate S_raw, LSE, or f32 output O
   * can overflow: single-pass commit. Outside this range, run a non-committing
   * preflight pass first to guarantee failure-atomicity with zero heap allocs.
   */
  const int safe_direct =
      (max_q <= 1.0e4 && max_k <= 1.0e4 && max_v <= 1.0e30 && scale <= 1.0e4);

  et_fa2_stats_v1 local_stats;
  memset(&local_stats, 0, sizeof(local_stats));

  et_fa2_status_t st = ET_FA2_OK;
  if (!safe_direct) {
    st = fa2_forward_exec(&cfg, scale, qf, kf, vf, (float *)o_out,
                          (double *)lse_out, 0, NULL);
  }
  if (st == ET_FA2_OK) {
    st = fa2_forward_exec(&cfg, scale, qf, kf, vf, (float *)o_out,
                          (double *)lse_out, 1, &local_stats);
  }

  if (fesetenv(&saved_env) != 0)
    return ET_FA2_ERR_FP_ENV;

  if (st == ET_FA2_OK && stats_out != NULL)
    memcpy(stats_out, &local_stats, sizeof(local_stats));
  return st;
}

/*
 * Core tiled FlashAttention-2 recomputation backward kernel.
 * Uses strictly O(B_r * B_c) tile scratch (p_tile, ds_tile) and never saves or
 * materializes a T_q x T_k attention matrix.
 * When commit == 0, performs an identical numeric dry-run verifying every
 * intermediate and f32 output without modifying dq_out, dk_out, or dv_out.
 */
static et_fa2_status_t fa2_backward_exec(
    const et_fa2_config_v1 *cfg, double scale, const float *qf,
    const float *kf, const float *vf, const float *of, const double *lsef,
    const float *dof, float *dqf, float *dkf, float *dvf, int commit,
    et_fa2_stats_v1 *stats_out) {
  const size_t B = cfg->batch;
  const size_t Hq = cfg->num_heads;
  const size_t Hkv = cfg->num_kv_heads;
  const size_t group = Hq / Hkv;
  const size_t Tq = cfg->seq_q;
  const size_t Tk = cfg->seq_k;
  const size_t Dh = cfg->head_dim;
  const size_t Br = cfg->tile_r;
  const size_t Bc = cfg->tile_c;
  const uint32_t causal = cfg->causal;
  const size_t window = cfg->sliding_window;
  const double softcap = cfg->logit_softcap;

  const uint64_t q_tiles = (uint64_t)((Tq + Br - 1u) / Br);
  const uint64_t k_tiles = (uint64_t)((Tk + Bc - 1u) / Bc);
  uint64_t evaluated = 0u;
  uint64_t skipped = 0u;

  /*
   * Per-KV-head binary64 accumulators (O(T * D_h)) and strictly O(B_r * B_c)
   * tile scratch. Total stack usage is bounded (~434 KiB), zero heap allocs.
   */
  double dk_kv[FA2_MAX_T * FA2_MAX_DH];
  double dv_kv[FA2_MAX_T * FA2_MAX_DH];
  double dq_h[FA2_MAX_T * FA2_MAX_DH];
  double d_vec[FA2_MAX_T];
  double dk_tile[FA2_MAX_TILE * FA2_MAX_DH];
  double dv_tile[FA2_MAX_TILE * FA2_MAX_DH];
  double p_tile[FA2_MAX_TILE * FA2_MAX_TILE];
  double ds_tile[FA2_MAX_TILE * FA2_MAX_TILE];

  for (size_t b = 0u; b < B; ++b) {
    for (size_t hkv = 0u; hkv < Hkv; ++hkv) {
      const float *k_head = kf + ((b * Hkv + hkv) * Tk) * Dh;
      const float *v_head = vf + ((b * Hkv + hkv) * Tk) * Dh;
      float *dk_dst_head = dkf + ((b * Hkv + hkv) * Tk) * Dh;
      float *dv_dst_head = dvf + ((b * Hkv + hkv) * Tk) * Dh;

      memset(dk_kv, 0, Tk * Dh * sizeof(double));
      memset(dv_kv, 0, Tk * Dh * sizeof(double));

      for (size_t g = 0u; g < group; ++g) {
        const size_t h = hkv * group + g;
        const float *q_head = qf + ((b * Hq + h) * Tq) * Dh;
        const float *o_head = of + ((b * Hq + h) * Tq) * Dh;
        const float *do_head = dof + ((b * Hq + h) * Tq) * Dh;
        const double *lse_head = lsef + (b * Hq + h) * Tq;
        float *dq_dst_head = dqf + ((b * Hq + h) * Tq) * Dh;

        /* Precompute D[i] = sum_d dO[i, d] * O[i, d] in binary64. */
        for (size_t i = 0u; i < Tq; ++i) {
          const float *do_vec = do_head + i * Dh;
          const float *o_vec = o_head + i * Dh;
          double acc = 0.0;
          for (size_t d = 0u; d < Dh; ++d)
            acc += (double)do_vec[d] * (double)o_vec[d];
          if (!isfinite(acc))
            return ET_FA2_ERR_NONFINITE;
          d_vec[i] = acc;
        }

        memset(dq_h, 0, Tq * Dh * sizeof(double));

        /* FlashAttention-2 backward: KV tiles outer, Q tiles inner. */
        for (size_t k_start = 0u; k_start < Tk; k_start += Bc) {
          const size_t k_len = (Tk - k_start < Bc) ? (Tk - k_start) : Bc;
          const size_t k_end = k_start + k_len;

          memset(dk_tile, 0, k_len * Dh * sizeof(double));
          memset(dv_tile, 0, k_len * Dh * sizeof(double));

          for (size_t q_start = 0u; q_start < Tq; q_start += Br) {
            const size_t q_len = (Tq - q_start < Br) ? (Tq - q_start) : Br;
            const size_t q_end = q_start + q_len;

            if (tile_is_masked(q_start, q_end, k_start, k_end, causal,
                               window)) {
              ++skipped;
              continue;
            }
            ++evaluated;

            /* Recompute S_tile, P_tile, dP_tile, dS_tile on the fly. */
            for (size_t r = 0u; r < q_len; ++r) {
              const size_t q_pos = q_start + r;
              const float *q_vec = q_head + q_pos * Dh;
              const float *do_vec = do_head + q_pos * Dh;
              const double lse_i = lse_head[q_pos];
              const double d_i = d_vec[q_pos];

              for (size_t c = 0u; c < k_len; ++c) {
                const size_t k_pos = k_start + c;
                if (!pair_allowed(q_pos, k_pos, causal, window)) {
                  p_tile[r * Bc + c] = 0.0;
                  ds_tile[r * Bc + c] = 0.0;
                  continue;
                }
                const float *k_vec = k_head + k_pos * Dh;
                const float *v_vec = v_head + k_pos * Dh;
                double dot = 0.0;
                double dp = 0.0;
                for (size_t d = 0u; d < Dh; ++d) {
                  dot += (double)q_vec[d] * (double)k_vec[d];
                  dp += (double)do_vec[d] * (double)v_vec[d];
                }
                const double s_raw = dot * scale;
                if (!isfinite(s_raw) || !isfinite(dp))
                  return ET_FA2_ERR_NONFINITE;

                double s_eff = s_raw;
                double softcap_deriv = 1.0;
                if (softcap > 0.0) {
                  const double t = tanh(s_raw / softcap);
                  s_eff = softcap * t;
                  softcap_deriv = 1.0 - t * t;
                }
                const double p = exp(s_eff - lse_i);
                const double ds_capped = p * (dp - d_i);
                const double ds_raw = ds_capped * softcap_deriv;
                const double ddot = ds_raw * scale;
                if (!isfinite(p) || !isfinite(ddot))
                  return ET_FA2_ERR_NONFINITE;

                p_tile[r * Bc + c] = p;
                ds_tile[r * Bc + c] = ddot;
              }
            }

            /* Accumulate dQ_h, dK_tile, dV_tile from O(B_r * B_c) tiles. */
            for (size_t r = 0u; r < q_len; ++r) {
              const size_t q_pos = q_start + r;
              const float *q_vec = q_head + q_pos * Dh;
              const float *do_vec = do_head + q_pos * Dh;
              double *dq_row = dq_h + q_pos * Dh;

              for (size_t c = 0u; c < k_len; ++c) {
                const double p = p_tile[r * Bc + c];
                const double ddot = ds_tile[r * Bc + c];
                if (p == 0.0 && ddot == 0.0)
                  continue;
                const float *k_vec = k_head + (k_start + c) * Dh;
                double *dk_row = dk_tile + c * Dh;
                double *dv_row = dv_tile + c * Dh;
                for (size_t d = 0u; d < Dh; ++d) {
                  dq_row[d] += ddot * (double)k_vec[d];
                  dk_row[d] += ddot * (double)q_vec[d];
                  dv_row[d] += p * (double)do_vec[d];
                }
              }
            }
          }

          /* Fold tile dK, dV into GQA KV-head accumulator. */
          for (size_t c = 0u; c < k_len; ++c) {
            double *dk_dst = dk_kv + (k_start + c) * Dh;
            double *dv_dst = dv_kv + (k_start + c) * Dh;
            const double *dk_src = dk_tile + c * Dh;
            const double *dv_src = dv_tile + c * Dh;
            for (size_t d = 0u; d < Dh; ++d) {
              dk_dst[d] += dk_src[d];
              dv_dst[d] += dv_src[d];
            }
          }
        }

        /* Validate and optionally commit dQ for query head h. */
        const size_t head_q_elems = Tq * Dh;
        for (size_t idx = 0u; idx < head_q_elems; ++idx) {
          float dq_v = 0.0f;
          if (!finite_f64_to_f32(dq_h[idx], &dq_v))
            return ET_FA2_ERR_NONFINITE;
          if (commit)
            dq_dst_head[idx] = dq_v;
        }
      }

      /* Validate and optionally commit GQA-reduced dK and dV for hkv. */
      const size_t head_kv_elems = Tk * Dh;
      for (size_t idx = 0u; idx < head_kv_elems; ++idx) {
        float dk_v = 0.0f;
        float dv_v = 0.0f;
        if (!finite_f64_to_f32(dk_kv[idx], &dk_v) ||
            !finite_f64_to_f32(dv_kv[idx], &dv_v))
          return ET_FA2_ERR_NONFINITE;
        if (commit) {
          dk_dst_head[idx] = dk_v;
          dv_dst_head[idx] = dv_v;
        }
      }
    }
  }

  if (stats_out != NULL) {
    stats_out->total_tiles = (uint64_t)(B * Hq) * q_tiles * k_tiles;
    stats_out->evaluated_tiles = evaluated;
    stats_out->skipped_tiles = skipped;
    stats_out->q_tiles_per_head = q_tiles;
    stats_out->k_tiles_per_head = k_tiles;
    stats_out->tile_scratch_bytes =
        (uint64_t)et_fa2_tile_scratch_bytes_v1(Br, Bc);
  }
  return ET_FA2_OK;
}

et_fa2_status_t et_fa2_backward_v1(const void *cfg_ptr, const void *q,
                                   const void *k, const void *v, const void *o,
                                   const void *lse, const void *do_in,
                                   void *dq_out, void *dk_out, void *dv_out,
                                   et_fa2_stats_v1 *stats_out) {
  if (!fp_environment_supported())
    return ET_FA2_ERR_FP_ENV;

  et_fa2_config_v1 cfg;
  double scale = 0.0;
  const et_fa2_status_t cfg_st = read_config(cfg_ptr, &cfg, &scale);
  if (cfg_st != ET_FA2_OK)
    return cfg_st;

  const size_t q_elems = cfg.batch * cfg.num_heads * cfg.seq_q * cfg.head_dim;
  const size_t kv_elems =
      cfg.batch * cfg.num_kv_heads * cfg.seq_k * cfg.head_dim;
  const size_t lse_elems = cfg.batch * cfg.num_heads * cfg.seq_q;

  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t lse_bytes = lse_elems * sizeof(double);

  if (!aligned_span(q, _Alignof(float), q_bytes) ||
      !aligned_span(k, _Alignof(float), kv_bytes) ||
      !aligned_span(v, _Alignof(float), kv_bytes) ||
      !aligned_span(o, _Alignof(float), q_bytes) ||
      !aligned_span(lse, _Alignof(double), lse_bytes) ||
      !aligned_span(do_in, _Alignof(float), q_bytes) ||
      !aligned_span(dq_out, _Alignof(float), q_bytes) ||
      !aligned_span(dk_out, _Alignof(float), kv_bytes) ||
      !aligned_span(dv_out, _Alignof(float), kv_bytes))
    return ET_FA2_ERR_INVALID_ARG;
  if (stats_out != NULL &&
      !aligned_span(stats_out, _Alignof(et_fa2_stats_v1),
                    sizeof(et_fa2_stats_v1)))
    return ET_FA2_ERR_INVALID_ARG;

  const byte_span outputs[4] = {
      {dq_out, q_bytes},
      {dk_out, kv_bytes},
      {dv_out, kv_bytes},
      {stats_out, stats_out != NULL ? sizeof(et_fa2_stats_v1) : 0u}};
  const byte_span inputs[7] = {{cfg_ptr, sizeof(et_fa2_config_v1)},
                               {q, q_bytes},
                               {k, kv_bytes},
                               {v, kv_bytes},
                               {o, q_bytes},
                               {lse, lse_bytes},
                               {do_in, q_bytes}};
  if (!outputs_disjoint_from_all(outputs, 4u, inputs, 7u))
    return ET_FA2_ERR_INVALID_ARG;

  const float *qf = (const float *)q;
  const float *kf = (const float *)k;
  const float *vf = (const float *)v;
  const float *of = (const float *)o;
  const double *lsef = (const double *)lse;
  const float *dof = (const float *)do_in;

  double max_q = 0.0, max_k = 0.0, max_v = 0.0, max_o = 0.0, max_do = 0.0;
  double min_lse = 0.0, max_lse = 0.0;
  if (!scan_f32_finite_max(qf, q_elems, &max_q) ||
      !scan_f32_finite_max(kf, kv_elems, &max_k) ||
      !scan_f32_finite_max(vf, kv_elems, &max_v) ||
      !scan_f32_finite_max(of, q_elems, &max_o) ||
      !scan_f32_finite_max(dof, q_elems, &max_do) ||
      !scan_f64_finite_minmax(lsef, lse_elems, &min_lse, &max_lse))
    return ET_FA2_ERR_NONFINITE;

  fenv_t saved_env;
  if (fegetenv(&saved_env) != 0)
    return ET_FA2_ERR_FP_ENV;

  /*
   * Conservative analytic bound guaranteeing zero overflow in exp(S - LSE) and
   * dQ/dK/dV f32 conversion; outside this box, run a non-committing preflight
   * pass first to preserve strict failure-atomicity.
   */
  const double max_s_raw =
      (double)cfg.head_dim * max_q * max_k * scale;
  const double max_s = (cfg.logit_softcap > 0.0) ? cfg.logit_softcap : max_s_raw;
  const int safe_direct =
      (max_q <= 1.0e4 && max_k <= 1.0e4 && max_v <= 1.0e4 && max_o <= 1.0e4 &&
       max_do <= 1.0e4 && scale <= 1.0e4 && max_s <= 1.0e6 &&
       min_lse >= -1.0e6 && max_lse <= 1.0e6 && (max_s - min_lse) <= 20.0);

  et_fa2_stats_v1 local_stats;
  memset(&local_stats, 0, sizeof(local_stats));

  et_fa2_status_t st = ET_FA2_OK;
  if (!safe_direct) {
    st = fa2_backward_exec(&cfg, scale, qf, kf, vf, of, lsef, dof,
                           (float *)dq_out, (float *)dk_out, (float *)dv_out, 0,
                           NULL);
  }
  if (st == ET_FA2_OK) {
    st = fa2_backward_exec(&cfg, scale, qf, kf, vf, of, lsef, dof,
                           (float *)dq_out, (float *)dk_out, (float *)dv_out, 1,
                           &local_stats);
  }

  if (fesetenv(&saved_env) != 0)
    return ET_FA2_ERR_FP_ENV;

  if (st == ET_FA2_OK && stats_out != NULL)
    memcpy(stats_out, &local_stats, sizeof(local_stats));
  return st;
}
