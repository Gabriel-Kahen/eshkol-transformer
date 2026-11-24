#ifndef ESHKOL_TRANSFORMER_ROPE_YARN_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_ROPE_YARN_ESHKOL_BRIDGE_H

#include "eshkol_transformer/rope_yarn_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zero-allocation Eshkol AOT FFI bridge for the RoPE Long-Context Extension
 * Engine (STANDARD, LINEAR_PI, NTK_AWARE, YARN + mscale).
 *
 * All tensor/output pointers are accepted as void * so intentionally
 * misaligned raw pointers from Eshkol FFI are rejected cleanly without UB.
 * Returns 0 on success or a negative et_rope_yarn_status code on failure.
 */
int32_t et_rope_yarn_aot_compute_freqs(int64_t mode, int64_t layout,
                                       int64_t head_dim, int64_t rotary_dim,
                                       int64_t orig_max_pos, double base,
                                       double scale_factor, double beta_fast,
                                       double beta_slow, void *out_inv_freq,
                                       int64_t inv_freq_len, void *out_mscale);

int32_t et_rope_yarn_aot_forward(int64_t mode, int64_t layout,
                                 int64_t head_dim, int64_t rotary_dim,
                                 int64_t orig_max_pos, double base,
                                 double scale_factor, double beta_fast,
                                 double beta_slow, int64_t batch,
                                 int64_t tokens, int64_t heads, const void *x,
                                 const void *pos_ids, void *y);

int32_t et_rope_yarn_aot_backward(int64_t mode, int64_t layout,
                                  int64_t head_dim, int64_t rotary_dim,
                                  int64_t orig_max_pos, double base,
                                  double scale_factor, double beta_fast,
                                  double beta_slow, int64_t batch,
                                  int64_t tokens, int64_t heads, const void *dy,
                                  const void *pos_ids, void *dx);

/*
 * Writes 6 binary64 values into out_metrics_f64:
 *   [0] dot_mn, [1] dot_shifted, [2] dot_relative_ref,
 *   [3] shift_abs_err, [4] relative_abs_err, [5] mscale.
 */
int32_t et_rope_yarn_aot_qk_invariance(int64_t mode, int64_t layout,
                                       int64_t head_dim, int64_t rotary_dim,
                                       int64_t orig_max_pos, double base,
                                       double scale_factor, double beta_fast,
                                       double beta_slow, const void *q,
                                       const void *k, int64_t pos_m,
                                       int64_t pos_n, int64_t shift,
                                       void *out_metrics_f64);

/* Alignment-safe scalar marshaling for Eshkol AOT buffers.
 * Values outside finite binary32 range encode as +/-INFINITY so non-finite
 * rejection paths can be tested cleanly from Eshkol. */
void et_rope_yarn_aot_store_f32(void *buffer, int64_t index, double value);
double et_rope_yarn_aot_load_f32(const void *buffer, int64_t index);
void et_rope_yarn_aot_store_i64(void *buffer, int64_t index, int64_t value);
int64_t et_rope_yarn_aot_load_i64(const void *buffer, int64_t index);
void et_rope_yarn_aot_store_f64(void *buffer, int64_t index, double value);
double et_rope_yarn_aot_load_f64(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
