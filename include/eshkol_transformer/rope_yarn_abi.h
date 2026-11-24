#ifndef ESHKOL_TRANSFORMER_ROPE_YARN_ABI_H
#define ESHKOL_TRANSFORMER_ROPE_YARN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_ROPE_YARN_ABI_MAJOR 1u
#define ET_ROPE_YARN_ABI_MINOR 0u

/* Bounded dimensions for deterministic, zero-heap-allocation CPU execution. */
#define ET_ROPE_YARN_MIN_BATCH 1u
#define ET_ROPE_YARN_MAX_BATCH 64u
#define ET_ROPE_YARN_MIN_TOKENS 1u
#define ET_ROPE_YARN_MAX_TOKENS 16384u
#define ET_ROPE_YARN_MIN_HEADS 1u
#define ET_ROPE_YARN_MAX_HEADS 128u
#define ET_ROPE_YARN_MIN_HEAD_DIM 2u
#define ET_ROPE_YARN_MAX_HEAD_DIM 256u
#define ET_ROPE_YARN_MAX_POSITION INT64_C(16777216)

/* Default RoPE and YaRN parameters (Peng et al., 2023). */
#define ET_ROPE_YARN_DEFAULT_BASE 10000.0
#define ET_ROPE_YARN_DEFAULT_SCALE 1.0
#define ET_ROPE_YARN_DEFAULT_BETA_FAST 32.0
#define ET_ROPE_YARN_DEFAULT_BETA_SLOW 1.0
#define ET_ROPE_YARN_MAX_SCALE 1048576.0

typedef int32_t et_rope_yarn_status;

enum {
  ET_ROPE_YARN_OK = 0,
  ET_ROPE_YARN_ERR_ARGUMENT = -1,
  ET_ROPE_YARN_ERR_SHAPE = -2,
  ET_ROPE_YARN_ERR_ALIGNMENT = -3,
  ET_ROPE_YARN_ERR_ALIAS = -4,
  ET_ROPE_YARN_ERR_NONFINITE = -5,
  ET_ROPE_YARN_ERR_FP_ENV = -6,
  ET_ROPE_YARN_ERR_OVERFLOW = -7,
  ET_ROPE_YARN_ERR_POSITION = -8
};

/*
 * Four exact RoPE long-context frequency scaling modes:
 *   0: STANDARD   - inv_freq[i] = base^(-2i / d_rot), mscale = 1.0 (requires s == 1.0)
 *   1: LINEAR_PI  - inv_freq[i] = base^(-2i / d_rot) / s, mscale = 1.0 (s >= 1.0)
 *   2: NTK_AWARE  - base' = base * s^(d_rot / (d_rot - 2)),
 *                   inv_freq[i] = (base')^(-2i / d_rot), mscale = 1.0
 *                   (requires s >= 1.0 and even d_rot >= 4)
 *   3: YARN       - Peng et al. (2023) NTK-by-parts + attention temperature:
 *                   lambda[i] = 2 * pi / inv_freq_orig[i]
 *                   r[i]      = orig_max_pos / lambda[i]
 *                   gamma[i]  = clamp((r[i] - beta_slow) / (beta_fast - beta_slow), 0, 1)
 *                   inv_freq[i] = (1 - gamma[i]) * (inv_freq_orig[i] / s)
 *                               + gamma[i] * inv_freq_orig[i]
 *                   mscale = (s <= 1.0) ? 1.0 : (0.1 * ln(s) + 1.0)
 */
typedef enum et_rope_scaling_mode_t {
  ET_ROPE_SCALE_STANDARD = 0,
  ET_ROPE_SCALE_LINEAR_PI = 1,
  ET_ROPE_SCALE_NTK_AWARE = 2,
  ET_ROPE_SCALE_YARN = 3
} et_rope_scaling_mode_t;

/*
 * Rotary channel-pair layouts over the leading rotary prefix [0 .. rotary_dim - 1]
 * (with trailing [rotary_dim .. head_dim - 1] passed untouched for partial rotary):
 *   0: NEOX_SPLIT_HALF - pair k is (x[k], x[d_rot/2 + k]) for k = 0 .. d_rot/2 - 1
 *   1: INTERLEAVED     - pair k is (x[2*k], x[2*k + 1])   for k = 0 .. d_rot/2 - 1
 */
typedef enum et_rope_pair_layout_t {
  ET_ROPE_LAYOUT_NEOX_SPLIT_HALF = 0,
  ET_ROPE_LAYOUT_INTERLEAVED = 1
} et_rope_pair_layout_t;

/*
 * RoPE long-context configuration descriptor.
 * Constraints:
 *   mode in {0, 1, 2, 3}
 *   layout in {0, 1}
 *   head_dim in [2, 256]
 *   rotary_dim even, in [2, head_dim]; >= 4 when mode == ET_ROPE_SCALE_NTK_AWARE
 *   orig_max_pos in [1, ET_ROPE_YARN_MAX_POSITION]
 *   base finite binary64, > 1.0
 *   scale_factor finite binary64, in [1.0, ET_ROPE_YARN_MAX_SCALE];
 *     must equal 1.0 when mode == ET_ROPE_SCALE_STANDARD
 *   beta_fast, beta_slow finite binary64;
 *     when mode == ET_ROPE_SCALE_YARN: 0.0 < beta_slow < beta_fast
 */
typedef struct et_rope_yarn_config_t {
  uint32_t mode;
  uint32_t layout;
  uint32_t head_dim;
  uint32_t rotary_dim;
  uint64_t orig_max_pos;
  double base;
  double scale_factor;
  double beta_fast;
  double beta_slow;
} et_rope_yarn_config_t;

/*
 * Result of relative-position QK dot-product invariance verification:
 *   dot_mn           = <RoPE(q, m), RoPE(k, n)>
 *   dot_shifted      = <RoPE(q, m + shift), RoPE(k, n + shift)>
 *   dot_relative_ref = mscale^2 * <R(m - n) q_rot, k_rot> + <q_tail, k_tail>
 *   shift_abs_err    = |dot_mn - dot_shifted|
 *   relative_abs_err = |dot_mn - dot_relative_ref|
 *   mscale           = attention temperature scaling factor
 */
typedef struct et_rope_yarn_invariance_result_t {
  double dot_mn;
  double dot_shifted;
  double dot_relative_ref;
  double shift_abs_err;
  double relative_abs_err;
  double mscale;
} et_rope_yarn_invariance_result_t;

#if defined(__cplusplus)
#define ET_ROPE_YARN_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_ROPE_YARN_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_ROPE_YARN_STATIC_ASSERT(sizeof(et_rope_yarn_config_t) == 56u &&
                               offsetof(et_rope_yarn_config_t, orig_max_pos) ==
                                   16u &&
                               offsetof(et_rope_yarn_config_t, base) == 24u &&
                               offsetof(et_rope_yarn_config_t, beta_slow) ==
                                   48u,
                           "et_rope_yarn_config_t ABI layout changed");
ET_ROPE_YARN_STATIC_ASSERT(
    sizeof(et_rope_yarn_invariance_result_t) == 48u &&
        offsetof(et_rope_yarn_invariance_result_t, mscale) == 40u,
    "et_rope_yarn_invariance_result_t ABI layout changed");
#undef ET_ROPE_YARN_STATIC_ASSERT

/*
 * Validate configuration and host IEEE-754 x87/MXCSR FP environment.
 * Zero heap allocations.
 */
et_rope_yarn_status et_rope_yarn_validate_config_v1(
    const et_rope_yarn_config_t *cfg);

/*
 * Compute binary64 inverse frequencies inv_freq[rotary_dim / 2] and attention
 * temperature scalar *out_mscale.
 * Zero heap allocations; failure leaves caller outputs untouched bit-for-bit.
 */
et_rope_yarn_status et_rope_yarn_compute_freqs_v1(
    const et_rope_yarn_config_t *cfg, double *out_inv_freq, size_t inv_freq_len,
    double *out_mscale);
et_rope_yarn_status et_rope_yarn_compute_freqs_raw_v1(
    const et_rope_yarn_config_t *cfg, void *out_inv_freq, size_t inv_freq_len,
    void *out_mscale);

/*
 * Forward RoPE over dense row-major X[N, T, H, Dh] (f32) with explicit integer
 * positions pos_ids[N, T] (i64) in [0, min(ET_ROPE_YARN_MAX_POSITION,
 * floor(orig_max_pos * scale_factor))].
 *
 * For each rotary pair (x0, x1) at frequency k and position pos:
 *   phi = (double)pos * inv_freq[k]
 *   c_m = mscale * cos(phi), s_m = mscale * sin(phi)  (binary64)
 *   y0  = (float)((double)x0 * c_m - (double)x1 * s_m)
 *   y1  = (float)((double)x0 * s_m + (double)x1 * c_m)
 * Trailing partial-rotary channels d in [rotary_dim .. head_dim - 1] pass
 * through unchanged (y[d] = x[d]).
 *
 * Zero heap allocations; pairwise disjoint buffers; failure-atomic (output y
 * is untouched bit-for-bit on any error).
 */
et_rope_yarn_status et_rope_yarn_forward_v1(const et_rope_yarn_config_t *cfg,
                                            size_t batch, size_t tokens,
                                            size_t heads, const float *x,
                                            const int64_t *pos_ids, float *y);
et_rope_yarn_status et_rope_yarn_forward_raw_v1(
    const et_rope_yarn_config_t *cfg, size_t batch, size_t tokens, size_t heads,
    const void *x, const void *pos_ids, void *y);

/*
 * Exact analytic backward RoPE (VJP) over dense row-major dY[N, T, H, Dh]
 * (f32) -> dX[N, T, H, Dh] (f32):
 *   dx0 = (float)( (double)dy0 * c_m + (double)dy1 * s_m)
 *   dx1 = (float)(-(double)dy0 * s_m + (double)dy1 * c_m)
 * Trailing partial-rotary channels d in [rotary_dim .. head_dim - 1] pass
 * through unchanged (dx[d] = dy[d]).
 *
 * Zero heap allocations; pairwise disjoint buffers; failure-atomic.
 */
et_rope_yarn_status et_rope_yarn_backward_v1(const et_rope_yarn_config_t *cfg,
                                             size_t batch, size_t tokens,
                                             size_t heads, const float *dy,
                                             const int64_t *pos_ids, float *dx);
et_rope_yarn_status et_rope_yarn_backward_raw_v1(
    const et_rope_yarn_config_t *cfg, size_t batch, size_t tokens, size_t heads,
    const void *dy, const void *pos_ids, void *dx);

/*
 * Verify relative-position QK dot-product invariance for head vectors q[Dh]
 * and k[Dh] at positions (pos_m, pos_n) and shifted positions
 * (pos_m + shift, pos_n + shift), comparing against exact binary64
 * mscale^2 * <R(pos_m - pos_n) q_rot, k_rot> + <q_tail, k_tail>.
 *
 * Zero heap allocations; failure leaves *out_result untouched.
 */
et_rope_yarn_status et_rope_yarn_qk_dot_invariance_check_v1(
    const et_rope_yarn_config_t *cfg, const float *q, const float *k,
    int64_t pos_m, int64_t pos_n, int64_t shift,
    et_rope_yarn_invariance_result_t *out_result);
et_rope_yarn_status et_rope_yarn_qk_dot_invariance_check_raw_v1(
    const et_rope_yarn_config_t *cfg, const void *q, const void *k,
    int64_t pos_m, int64_t pos_n, int64_t shift, void *out_result);

#ifdef __cplusplus
}
#endif

#endif
