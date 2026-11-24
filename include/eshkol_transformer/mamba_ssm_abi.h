#ifndef ESHKOL_TRANSFORMER_MAMBA_SSM_ABI_H
#define ESHKOL_TRANSFORMER_MAMBA_SSM_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MAMBA_SSM_ABI_MAJOR 1u
#define ET_MAMBA_SSM_ABI_MINOR 0u

/* Bounded admitted dimensions for deterministic zero-allocation execution. */
#define ET_MAMBA_MIN_BATCH 1u
#define ET_MAMBA_MAX_BATCH 4u
#define ET_MAMBA_MIN_SEQ_LEN 1u
#define ET_MAMBA_MAX_SEQ_LEN 64u
#define ET_MAMBA_MIN_D_INNER 2u
#define ET_MAMBA_MAX_D_INNER 32u
#define ET_MAMBA_MIN_D_STATE 2u
#define ET_MAMBA_MAX_D_STATE 16u
#define ET_MAMBA_MIN_K_CONV 2u
#define ET_MAMBA_MAX_K_CONV 4u

/* Upper bound on A_log so A = -exp(A_log) remains finite in binary32/binary64. */
#define ET_MAMBA_MAX_A_LOG 80.0

typedef int32_t et_mamba_status_t;

enum {
  ET_MAMBA_OK = 0,
  ET_MAMBA_ERR_INVALID_ARG = 1,
  ET_MAMBA_ERR_SHAPE_MISMATCH = 2,
  ET_MAMBA_ERR_UNSUPPORTED = 3,
  ET_MAMBA_ERR_NONFINITE = 4,
  ET_MAMBA_ERR_OVERFLOW = 5,
  ET_MAMBA_ERR_FP_ENV = 6
};

/*
 * Configuration for Mamba / S6 Selective State-Space Model block.
 *
 * Tensor layouts (dense row-major binary32 float unless stated otherwise):
 *   Stage A (Causal Conv1D + SiLU):
 *     u, x_out, dx, du_out : float[N, T, D]
 *     conv_w, dconv_w_out  : float[D, K]
 *     conv_b, dconv_b_out  : float[D]
 *
 *   Stage B (Selective Scan S6):
 *     x, dt_raw, z_opt, out, dout,
 *     dx_out, ddt_raw_out, dz_out_opt : float[N, T, D]
 *     a_log, da_log_out               : float[D, S]
 *     b_in, c_in, db_out, dc_out      : float[N, T, S]
 *     d_skip, dd_skip_out             : float[D]
 *
 *   Stage C (O(1) Recurrent Inference Cache):
 *     conv_state : float[N, K - 1, D]   (constant w.r.t. sequence length T)
 *     ssm_state  : double[N, D, S]      (constant w.r.t. sequence length T)
 *     u_step, dt_raw_step, z_step_opt, out_step : float[N, D]
 *     b_step, c_step                            : float[N, S]
 *
 * All functions perform internal accumulation in IEEE-754 binary64 under
 * FE_TONEAREST with FP contraction disabled, allocate zero heap memory, and
 * are failure-atomic (caller outputs and cache state remain byte-for-byte
 * untouched on any error).
 */
typedef struct et_mamba_config_v1 {
  size_t struct_size;
  size_t batch;   /* N in [1, 4] */
  size_t seq_len; /* T in [1, 64] */
  size_t d_inner; /* D in [2, 32] */
  size_t d_state; /* S in [2, 16] */
  size_t k_conv;  /* K in [2, 4] */
} et_mamba_config_v1;

/*
 * Caller-owned O(1) constant-memory recurrent inference state for Mamba S6.
 * State byte footprint depends ONLY on (N, D, S, K) and is independent of T.
 */
typedef struct et_mamba_cache_v1 {
  size_t struct_size;
  size_t batch;
  size_t d_inner;
  size_t d_state;
  size_t k_conv;
  uint64_t step_count;
  float *conv_state;
  size_t conv_state_bytes;
  double *ssm_state;
  size_t ssm_state_bytes;
} et_mamba_cache_v1;

/* Validate a Mamba S6 configuration descriptor. */
et_mamba_status_t et_mamba_validate_config_v1(const void *cfg_ptr);

/*
 * Stage A: Depthwise causal 1D convolution with left zero-padding + SiLU.
 *   c[n, t, d] = conv_b[d] + sum_{k=0..K-1} conv_w[d, k] * u[n, t-(K-1-k), d]
 *   x[n, t, d] = SiLU(c[n, t, d]) = c * sigmoid(c)
 */
et_mamba_status_t et_mamba_causal_conv1d_silu_forward_v1(
    const void *cfg_ptr, const void *u, const void *conv_w, const void *conv_b,
    void *x_out);

et_mamba_status_t et_mamba_causal_conv1d_silu_backward_v1(
    const void *cfg_ptr, const void *u, const void *conv_w, const void *conv_b,
    const void *dx, void *du_out, void *dconv_w_out, void *dconv_b_out);

/*
 * Stage B: Input-dependent Selective Scan (S6) with optional SiLU output gate.
 *   A[d, s]     = -exp(A_log[d, s]) < 0
 *   dt[n, t, d] = softplus(dt_raw[n, t, d])
 *   dA          = exp(dt * A)
 *   h_t[d, s]   = dA * h_{t-1}[d, s] + dt * B[n, t, s] * x[n, t, d]  (h_{-1}=0)
 *   y_t[d]      = sum_s(h_t[d, s] * C[n, t, s]) + D_skip[d] * x[n, t, d]
 *   out_t[d]    = (z_opt != NULL) ? (y_t[d] * SiLU(z[n, t, d])) : y_t[d]
 *
 * In backward, z_opt and dz_out_opt must either both be NULL (ungated) or
 * both be non-NULL (gated).
 */
et_mamba_status_t et_mamba_selective_scan_forward_v1(
    const void *cfg_ptr, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    void *out);

et_mamba_status_t et_mamba_selective_scan_backward_v1(
    const void *cfg_ptr, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    const void *dout, void *dx_out, void *ddt_raw_out, void *da_log_out,
    void *db_out, void *dc_out, void *dd_skip_out, void *dz_out_opt);

/*
 * Stage C: O(1) constant-memory recurrent cache sizing, lifecycle, and step.
 * Byte helpers return 0 if cfg_ptr is invalid.
 */
size_t et_mamba_cache_conv_state_bytes_v1(const void *cfg_ptr);
size_t et_mamba_cache_ssm_state_bytes_v1(const void *cfg_ptr);
size_t et_mamba_cache_total_state_bytes_v1(const void *cfg_ptr);

et_mamba_status_t et_mamba_cache_init_v1(void *cache_ptr, const void *cfg_ptr,
                                         void *conv_state_buf,
                                         size_t conv_state_bytes,
                                         void *ssm_state_buf,
                                         size_t ssm_state_bytes);

et_mamba_status_t et_mamba_cache_reset_v1(void *cache_ptr);

et_mamba_status_t et_mamba_cache_step_v1(
    void *cache_ptr, const void *u_step, const void *conv_w, const void *conv_b,
    const void *dt_raw_step, const void *a_log, const void *b_step,
    const void *c_step, const void *d_skip, const void *z_step_opt,
    void *out_step);

#ifdef __cplusplus
}
#endif

#endif
