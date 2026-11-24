#include "mamba_ssm_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && (uintptr_t)p % alignment == 0u &&
         (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static float f32_operand(double value) {
  if (!isfinite(value))
    return (float)value;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

int64_t et_mamba_aot_config_bytes(void) {
  return (int64_t)sizeof(et_mamba_config_v1);
}

int64_t et_mamba_aot_cache_bytes(void) {
  return (int64_t)sizeof(et_mamba_cache_v1);
}

int32_t et_mamba_aot_set_config(void *cfg_buf, int64_t batch, int64_t seq_len,
                                int64_t d_inner, int64_t d_state,
                                int64_t k_conv) {
  if (!aligned_span(cfg_buf, _Alignof(et_mamba_config_v1),
                    sizeof(et_mamba_config_v1)))
    return ET_MAMBA_ERR_INVALID_ARG;
  if (batch < 0 || seq_len < 0 || d_inner < 0 || d_state < 0 || k_conv < 0)
    return ET_MAMBA_ERR_UNSUPPORTED;

  const et_mamba_config_v1 cfg = {
      sizeof(et_mamba_config_v1),
      (size_t)batch,
      (size_t)seq_len,
      (size_t)d_inner,
      (size_t)d_state,
      (size_t)k_conv,
  };
  memcpy(cfg_buf, &cfg, sizeof(cfg));
  return ET_MAMBA_OK;
}

int64_t et_mamba_aot_conv_state_bytes(const void *cfg_buf) {
  return (int64_t)et_mamba_cache_conv_state_bytes_v1(cfg_buf);
}

int64_t et_mamba_aot_ssm_state_bytes(const void *cfg_buf) {
  return (int64_t)et_mamba_cache_ssm_state_bytes_v1(cfg_buf);
}

int64_t et_mamba_aot_total_state_bytes(const void *cfg_buf) {
  return (int64_t)et_mamba_cache_total_state_bytes_v1(cfg_buf);
}

int32_t et_mamba_aot_conv1d_silu_forward(const void *cfg_buf, const void *u,
                                         const void *conv_w, const void *conv_b,
                                         void *x_out) {
  return et_mamba_causal_conv1d_silu_forward_v1(cfg_buf, u, conv_w, conv_b,
                                                x_out);
}

int32_t et_mamba_aot_conv1d_silu_backward(const void *cfg_buf, const void *u,
                                          const void *conv_w,
                                          const void *conv_b, const void *dx,
                                          void *du_out, void *dconv_w_out,
                                          void *dconv_b_out) {
  return et_mamba_causal_conv1d_silu_backward_v1(cfg_buf, u, conv_w, conv_b, dx,
                                                 du_out, dconv_w_out,
                                                 dconv_b_out);
}

int32_t et_mamba_aot_selective_scan_forward(const void *cfg_buf, const void *x,
                                            const void *dt_raw,
                                            const void *a_log, const void *b_in,
                                            const void *c_in,
                                            const void *d_skip,
                                            const void *z_opt, void *out) {
  return et_mamba_selective_scan_forward_v1(cfg_buf, x, dt_raw, a_log, b_in,
                                            c_in, d_skip, z_opt, out);
}

int32_t et_mamba_aot_selective_scan_backward(
    const void *cfg_buf, const void *x, const void *dt_raw, const void *a_log,
    const void *b_in, const void *c_in, const void *d_skip, const void *z_opt,
    const void *dout, void *dx_out, void *ddt_raw_out, void *da_log_out,
    void *db_out, void *dc_out, void *dd_skip_out, void *dz_out_opt) {
  return et_mamba_selective_scan_backward_v1(
      cfg_buf, x, dt_raw, a_log, b_in, c_in, d_skip, z_opt, dout, dx_out,
      ddt_raw_out, da_log_out, db_out, dc_out, dd_skip_out, dz_out_opt);
}

int32_t et_mamba_aot_cache_init(void *cache_buf, const void *cfg_buf,
                                void *conv_state_buf, int64_t conv_state_bytes,
                                void *ssm_state_buf, int64_t ssm_state_bytes) {
  if (conv_state_bytes < 0 || ssm_state_bytes < 0)
    return ET_MAMBA_ERR_SHAPE_MISMATCH;
  return et_mamba_cache_init_v1(cache_buf, cfg_buf, conv_state_buf,
                                (size_t)conv_state_bytes, ssm_state_buf,
                                (size_t)ssm_state_bytes);
}

int32_t et_mamba_aot_cache_reset(void *cache_buf) {
  return et_mamba_cache_reset_v1(cache_buf);
}

int64_t et_mamba_aot_cache_step_count(const void *cache_buf) {
  if (!aligned_span(cache_buf, _Alignof(et_mamba_cache_v1),
                    sizeof(et_mamba_cache_v1)))
    return -1;
  et_mamba_cache_v1 st;
  memcpy(&st, cache_buf, sizeof(st));
  if (st.struct_size != sizeof(et_mamba_cache_v1))
    return -1;
  return (int64_t)st.step_count;
}

int32_t et_mamba_aot_cache_step(void *cache_buf, const void *u_step,
                                const void *conv_w, const void *conv_b,
                                const void *dt_raw_step, const void *a_log,
                                const void *b_step, const void *c_step,
                                const void *d_skip, const void *z_step_opt,
                                void *out_step) {
  return et_mamba_cache_step_v1(cache_buf, u_step, conv_w, conv_b, dt_raw_step,
                                a_log, b_step, c_step, d_skip, z_step_opt,
                                out_step);
}

void *et_mamba_aot_null_ptr(void) { return NULL; }

void et_mamba_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_mamba_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}
