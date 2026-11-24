#include "flash_attn2_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

static int aligned_span(const void *p, size_t alignment, size_t bytes) {
  return p != NULL && ((uintptr_t)p % alignment) == 0u &&
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

int64_t et_fa2_aot_config_bytes(void) {
  return (int64_t)sizeof(et_fa2_config_v1);
}

int64_t et_fa2_aot_stats_bytes(void) {
  return (int64_t)sizeof(et_fa2_stats_v1);
}

int64_t et_fa2_aot_tile_scratch_bytes(int64_t tile_r, int64_t tile_c) {
  if (tile_r < 0 || tile_c < 0)
    return 0;
  return (int64_t)et_fa2_tile_scratch_bytes_v1((size_t)tile_r, (size_t)tile_c);
}

int32_t et_fa2_aot_set_config(void *cfg_buf, int64_t batch, int64_t num_heads,
                              int64_t num_kv_heads, int64_t seq_q,
                              int64_t seq_k, int64_t head_dim, int64_t tile_r,
                              int64_t tile_c, int64_t causal,
                              int64_t sliding_window, double scale,
                              double logit_softcap) {
  if (!aligned_span(cfg_buf, _Alignof(et_fa2_config_v1),
                    sizeof(et_fa2_config_v1)))
    return ET_FA2_ERR_INVALID_ARG;
  if (batch < 0 || num_heads < 0 || num_kv_heads < 0 || seq_q < 0 ||
      seq_k < 0 || head_dim < 0 || tile_r < 0 || tile_c < 0 || causal < 0 ||
      causal > 255 || sliding_window < 0)
    return ET_FA2_ERR_INVALID_ARG;

  const et_fa2_config_v1 cfg = {sizeof(et_fa2_config_v1),
                                (size_t)batch,
                                (size_t)num_heads,
                                (size_t)num_kv_heads,
                                (size_t)seq_q,
                                (size_t)seq_k,
                                (size_t)head_dim,
                                (size_t)tile_r,
                                (size_t)tile_c,
                                (size_t)sliding_window,
                                (uint32_t)causal,
                                0u,
                                scale,
                                logit_softcap};
  memcpy(cfg_buf, &cfg, sizeof(cfg));
  return ET_FA2_OK;
}

int32_t et_fa2_aot_forward(const void *cfg_buf, const void *q, const void *k,
                           const void *v, void *o_out, void *lse_out,
                           void *stats_buf) {
  et_fa2_stats_v1 *stats_ptr = NULL;
  if (stats_buf != NULL) {
    if (!aligned_span(stats_buf, _Alignof(et_fa2_stats_v1),
                      sizeof(et_fa2_stats_v1)))
      return ET_FA2_ERR_INVALID_ARG;
    stats_ptr = (et_fa2_stats_v1 *)stats_buf;
  }
  return et_fa2_forward_v1(cfg_buf, q, k, v, o_out, lse_out, stats_ptr);
}

int32_t et_fa2_aot_backward(const void *cfg_buf, const void *q, const void *k,
                            const void *v, const void *o, const void *lse,
                            const void *do_in, void *dq_out, void *dk_out,
                            void *dv_out, void *stats_buf) {
  et_fa2_stats_v1 *stats_ptr = NULL;
  if (stats_buf != NULL) {
    if (!aligned_span(stats_buf, _Alignof(et_fa2_stats_v1),
                      sizeof(et_fa2_stats_v1)))
      return ET_FA2_ERR_INVALID_ARG;
    stats_ptr = (et_fa2_stats_v1 *)stats_buf;
  }
  return et_fa2_backward_v1(cfg_buf, q, k, v, o, lse, do_in, dq_out, dk_out,
                            dv_out, stats_ptr);
}

int32_t et_fa2_aot_check_bad_fpenv_forward(const void *cfg_buf, const void *q,
                                           const void *k, const void *v,
                                           void *o_out, void *lse_out,
                                           void *stats_buf) {
  fenv_t saved;
  if (fegetenv(&saved) != 0)
    return ET_FA2_ERR_FP_ENV;
  if (fesetround(FE_TOWARDZERO) != 0) {
    (void)fesetenv(&saved);
    return ET_FA2_ERR_FP_ENV;
  }
  const int32_t st =
      et_fa2_aot_forward(cfg_buf, q, k, v, o_out, lse_out, stats_buf);
  (void)fesetenv(&saved);
  return st;
}

static int read_stats(const void *stats_buf, et_fa2_stats_v1 *out) {
  if (!aligned_span(stats_buf, _Alignof(et_fa2_stats_v1),
                    sizeof(et_fa2_stats_v1)))
    return 0;
  memcpy(out, stats_buf, sizeof(*out));
  return 1;
}

int64_t et_fa2_aot_stats_total_tiles(const void *stats_buf) {
  et_fa2_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.total_tiles : -1;
}

int64_t et_fa2_aot_stats_evaluated_tiles(const void *stats_buf) {
  et_fa2_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.evaluated_tiles : -1;
}

int64_t et_fa2_aot_stats_skipped_tiles(const void *stats_buf) {
  et_fa2_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.skipped_tiles : -1;
}

int64_t et_fa2_aot_stats_tile_scratch_bytes(const void *stats_buf) {
  et_fa2_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.tile_scratch_bytes : -1;
}

void et_fa2_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_fa2_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value,
         (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}

void et_fa2_aot_store_f64(void *buffer, int64_t index, double value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

double et_fa2_aot_load_f64(const void *buffer, int64_t index) {
  double value = 0.0;
  memcpy(&value,
         (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}
