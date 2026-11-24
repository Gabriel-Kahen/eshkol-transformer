#include "rope_yarn_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <string.h>

static int build_config(int64_t mode, int64_t layout, int64_t head_dim,
                        int64_t rotary_dim, int64_t orig_max_pos, double base,
                        double scale_factor, double beta_fast, double beta_slow,
                        et_rope_yarn_config_t *cfg) {
  if (mode < 0 || mode > (int64_t)UINT32_MAX || layout < 0 ||
      layout > (int64_t)UINT32_MAX || orig_max_pos <= 0)
    return ET_ROPE_YARN_ERR_ARGUMENT;
  if (head_dim < 0 || head_dim > (int64_t)UINT32_MAX || rotary_dim < 0 ||
      rotary_dim > (int64_t)UINT32_MAX)
    return ET_ROPE_YARN_ERR_SHAPE;
  cfg->mode = (uint32_t)mode;
  cfg->layout = (uint32_t)layout;
  cfg->head_dim = (uint32_t)head_dim;
  cfg->rotary_dim = (uint32_t)rotary_dim;
  cfg->orig_max_pos = (uint64_t)orig_max_pos;
  cfg->base = base;
  cfg->scale_factor = scale_factor;
  cfg->beta_fast = beta_fast;
  cfg->beta_slow = beta_slow;
  return ET_ROPE_YARN_OK;
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

int32_t et_rope_yarn_aot_compute_freqs(int64_t mode, int64_t layout,
                                       int64_t head_dim, int64_t rotary_dim,
                                       int64_t orig_max_pos, double base,
                                       double scale_factor, double beta_fast,
                                       double beta_slow, void *out_inv_freq,
                                       int64_t inv_freq_len, void *out_mscale) {
  et_rope_yarn_config_t cfg;
  const int32_t status =
      build_config(mode, layout, head_dim, rotary_dim, orig_max_pos, base,
                   scale_factor, beta_fast, beta_slow, &cfg);
  if (status != ET_ROPE_YARN_OK)
    return status;
  if (inv_freq_len < 0)
    return ET_ROPE_YARN_ERR_SHAPE;
  return et_rope_yarn_compute_freqs_raw_v1(&cfg, out_inv_freq,
                                           (size_t)inv_freq_len, out_mscale);
}

int32_t et_rope_yarn_aot_forward(int64_t mode, int64_t layout,
                                 int64_t head_dim, int64_t rotary_dim,
                                 int64_t orig_max_pos, double base,
                                 double scale_factor, double beta_fast,
                                 double beta_slow, int64_t batch,
                                 int64_t tokens, int64_t heads, const void *x,
                                 const void *pos_ids, void *y) {
  et_rope_yarn_config_t cfg;
  const int32_t status =
      build_config(mode, layout, head_dim, rotary_dim, orig_max_pos, base,
                   scale_factor, beta_fast, beta_slow, &cfg);
  if (status != ET_ROPE_YARN_OK)
    return status;
  if (batch <= 0 || tokens <= 0 || heads <= 0)
    return ET_ROPE_YARN_ERR_SHAPE;
  return et_rope_yarn_forward_raw_v1(&cfg, (size_t)batch, (size_t)tokens,
                                     (size_t)heads, x, pos_ids, y);
}

int32_t et_rope_yarn_aot_backward(int64_t mode, int64_t layout,
                                  int64_t head_dim, int64_t rotary_dim,
                                  int64_t orig_max_pos, double base,
                                  double scale_factor, double beta_fast,
                                  double beta_slow, int64_t batch,
                                  int64_t tokens, int64_t heads, const void *dy,
                                  const void *pos_ids, void *dx) {
  et_rope_yarn_config_t cfg;
  const int32_t status =
      build_config(mode, layout, head_dim, rotary_dim, orig_max_pos, base,
                   scale_factor, beta_fast, beta_slow, &cfg);
  if (status != ET_ROPE_YARN_OK)
    return status;
  if (batch <= 0 || tokens <= 0 || heads <= 0)
    return ET_ROPE_YARN_ERR_SHAPE;
  return et_rope_yarn_backward_raw_v1(&cfg, (size_t)batch, (size_t)tokens,
                                      (size_t)heads, dy, pos_ids, dx);
}

int32_t et_rope_yarn_aot_qk_invariance(int64_t mode, int64_t layout,
                                       int64_t head_dim, int64_t rotary_dim,
                                       int64_t orig_max_pos, double base,
                                       double scale_factor, double beta_fast,
                                       double beta_slow, const void *q,
                                       const void *k, int64_t pos_m,
                                       int64_t pos_n, int64_t shift,
                                       void *out_metrics_f64) {
  et_rope_yarn_config_t cfg;
  const int32_t status =
      build_config(mode, layout, head_dim, rotary_dim, orig_max_pos, base,
                   scale_factor, beta_fast, beta_slow, &cfg);
  if (status != ET_ROPE_YARN_OK)
    return status;
  return et_rope_yarn_qk_dot_invariance_check_raw_v1(&cfg, q, k, pos_m, pos_n,
                                                     shift, out_metrics_f64);
}

void et_rope_yarn_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_rope_yarn_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}

void et_rope_yarn_aot_store_i64(void *buffer, int64_t index, int64_t value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

int64_t et_rope_yarn_aot_load_i64(const void *buffer, int64_t index) {
  int64_t value = 0;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}

void et_rope_yarn_aot_store_f64(void *buffer, int64_t index, double value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

double et_rope_yarn_aot_load_f64(const void *buffer, int64_t index) {
  double value = 0.0;
  memcpy(&value, (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}
