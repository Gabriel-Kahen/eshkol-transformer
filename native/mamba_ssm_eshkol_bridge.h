#ifndef ESHKOL_TRANSFORMER_MAMBA_SSM_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MAMBA_SSM_ESHKOL_BRIDGE_H

#include "eshkol_transformer/mamba_ssm_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT transport bridge for Mamba / S6 Selective State-Space Model.
 * All struct and tensor pointers are accepted as untyped void * so raw Eshkol
 * FFI addresses (including intentionally misaligned ptr-add pointers) are
 * validated cleanly without misaligned-pointer UB. Zero heap allocations.
 */
int64_t et_mamba_aot_config_bytes(void);
int64_t et_mamba_aot_cache_bytes(void);

int32_t et_mamba_aot_set_config(void *cfg_buf, int64_t batch, int64_t seq_len,
                                int64_t d_inner, int64_t d_state,
                                int64_t k_conv);

int64_t et_mamba_aot_conv_state_bytes(const void *cfg_buf);
int64_t et_mamba_aot_ssm_state_bytes(const void *cfg_buf);
int64_t et_mamba_aot_total_state_bytes(const void *cfg_buf);

int32_t et_mamba_aot_conv1d_silu_forward(const void *cfg_buf, const void *u,
                                         const void *conv_w, const void *conv_b,
                                         void *x_out);

int32_t et_mamba_aot_conv1d_silu_backward(const void *cfg_buf, const void *u,
                                          const void *conv_w,
                                          const void *conv_b, const void *dx,
                                          void *du_out, void *dconv_w_out,
                                          void *dconv_b_out);

int32_t et_mamba_aot_selective_scan_forward(const void *cfg_buf, const void *x,
                                            const void *dt_raw,
                                            const void *a_log, const void *b_in,
                                            const void *c_in,
                                            const void *d_skip,
                                            const void *z_opt, void *out);

int32_t et_mamba_aot_selective_scan_backward(
    const void *cfg_buf, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    const void *dout, void *dx_out, void *ddt_raw_out, void *da_log_out,
    void *db_out, void *dc_out, void *dd_skip_out, void *dz_out_opt);

int32_t et_mamba_aot_cache_init(void *cache_buf, const void *cfg_buf,
                                void *conv_state_buf, int64_t conv_state_bytes,
                                void *ssm_state_buf, int64_t ssm_state_bytes);

int32_t et_mamba_aot_cache_reset(void *cache_buf);
int64_t et_mamba_aot_cache_step_count(const void *cache_buf);

int32_t et_mamba_aot_cache_step(void *cache_buf, const void *u_step,
                                const void *conv_w, const void *conv_b,
                                const void *dt_raw_step, const void *a_log,
                                const void *b_step, const void *c_step,
                                const void *d_skip, const void *z_step_opt,
                                void *out_step);

void *et_mamba_aot_null_ptr(void);

/* Scalar marshaling between Eshkol f64 values and owned binary32 buffers. */
void et_mamba_aot_store_f32(void *buffer, int64_t index, double value);
double et_mamba_aot_load_f32(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
