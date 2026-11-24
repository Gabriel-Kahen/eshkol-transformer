#include "ring_attn_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <string.h>

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

int64_t et_ring_attn_aot_config_bytes(void) {
  return (int64_t)sizeof(et_ring_attn_config_v1);
}

int64_t et_ring_attn_aot_stats_bytes(void) {
  return (int64_t)sizeof(et_ring_attn_stats_v1);
}

int32_t et_ring_attn_aot_set_config(void *cfg_buf, int64_t batch,
                                    int64_t seq_len, int64_t num_heads,
                                    int64_t num_kv_heads, int64_t head_dim,
                                    int64_t world_size, int64_t causal,
                                    double scale, int64_t timeout_ms) {
  if (!aligned_span(cfg_buf, _Alignof(et_ring_attn_config_v1),
                    sizeof(et_ring_attn_config_v1)))
    return ET_RING_ATTN_ERR_INVALID_ARG;
  if (batch < 0 || seq_len < 0 || num_heads < 0 || num_kv_heads < 0 ||
      head_dim < 0 || world_size < 0 || causal < 0 || timeout_ms < 0 ||
      timeout_ms > ET_RING_ATTN_MAX_TIMEOUT_MS)
    return ET_RING_ATTN_ERR_INVALID_ARG;

  const et_ring_attn_config_v1 cfg = {
      sizeof(et_ring_attn_config_v1),
      (size_t)batch,
      (size_t)seq_len,
      (size_t)num_heads,
      (size_t)num_kv_heads,
      (size_t)head_dim,
      (uint32_t)world_size,
      (uint32_t)causal,
      scale,
      (int32_t)timeout_ms};
  memcpy(cfg_buf, &cfg, sizeof(cfg));
  return ET_RING_ATTN_OK;
}

int32_t et_ring_attn_aot_dense_forward(const void *cfg_buf, const void *q,
                                       const void *k, const void *v, void *o,
                                       void *lse) {
  return et_ring_attn_dense_ref_forward_v1(cfg_buf, q, k, v, o, lse);
}

int32_t et_ring_attn_aot_dense_backward(const void *cfg_buf, const void *q,
                                        const void *k, const void *v,
                                        const void *o, const void *lse,
                                        const void *do_buf, void *dq, void *dk,
                                        void *dv) {
  return et_ring_attn_dense_ref_backward_v1(cfg_buf, q, k, v, o, lse, do_buf,
                                            dq, dk, dv);
}

int32_t et_ring_attn_aot_mp_forward(const void *cfg_buf, const void *q,
                                    const void *k, const void *v, void *o,
                                    void *lse, void *stats_buf) {
  et_ring_attn_stats_v1 *stats_ptr = NULL;
  et_ring_attn_stats_v1 local_stats;
  if (stats_buf != NULL) {
    if (!aligned_span(stats_buf, _Alignof(et_ring_attn_stats_v1),
                      sizeof(et_ring_attn_stats_v1)))
      return ET_RING_ATTN_ERR_INVALID_ARG;
    memset(&local_stats, 0, sizeof(local_stats));
    stats_ptr = &local_stats;
  }
  const int32_t st = et_ring_attn_multiprocess_forward_v1(
      cfg_buf, q, k, v, o, lse, stats_ptr);
  if (st == ET_RING_ATTN_OK && stats_buf != NULL)
    memcpy(stats_buf, &local_stats, sizeof(local_stats));
  return st;
}

int32_t et_ring_attn_aot_mp_backward(const void *cfg_buf, const void *q,
                                     const void *k, const void *v,
                                     const void *o, const void *lse,
                                     const void *do_buf, void *dq, void *dk,
                                     void *dv, void *stats_buf) {
  et_ring_attn_stats_v1 *stats_ptr = NULL;
  et_ring_attn_stats_v1 local_stats;
  if (stats_buf != NULL) {
    if (!aligned_span(stats_buf, _Alignof(et_ring_attn_stats_v1),
                      sizeof(et_ring_attn_stats_v1)))
      return ET_RING_ATTN_ERR_INVALID_ARG;
    memset(&local_stats, 0, sizeof(local_stats));
    stats_ptr = &local_stats;
  }
  const int32_t st = et_ring_attn_multiprocess_backward_v1(
      cfg_buf, q, k, v, o, lse, do_buf, dq, dk, dv, stats_ptr);
  if (st == ET_RING_ATTN_OK && stats_buf != NULL)
    memcpy(stats_buf, &local_stats, sizeof(local_stats));
  return st;
}

static int read_stats(const void *stats_buf, et_ring_attn_stats_v1 *out) {
  if (!aligned_span(stats_buf, _Alignof(et_ring_attn_stats_v1),
                    sizeof(et_ring_attn_stats_v1)))
    return 0;
  memcpy(out, stats_buf, sizeof(*out));
  return 1;
}

int64_t et_ring_attn_aot_stats_total_blocks(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.total_block_steps : -1;
}

int64_t et_ring_attn_aot_stats_executed_blocks(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.executed_blocks : -1;
}

int64_t et_ring_attn_aot_stats_skipped_causal(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.skipped_causal_blocks : -1;
}

int64_t et_ring_attn_aot_stats_ring_rounds(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.ring_rounds : -1;
}

int64_t et_ring_attn_aot_stats_bytes_sent(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.bytes_sent : -1;
}

int64_t et_ring_attn_aot_stats_bytes_received(const void *stats_buf) {
  et_ring_attn_stats_v1 s;
  return read_stats(stats_buf, &s) ? (int64_t)s.bytes_received : -1;
}

void et_ring_attn_aot_store_f32(void *buffer, int64_t index, double value) {
  const float encoded = f32_operand(value);
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(encoded), &encoded,
         sizeof(encoded));
}

double et_ring_attn_aot_load_f32(const void *buffer, int64_t index) {
  float value = 0.0f;
  memcpy(&value,
         (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return (double)value;
}

void et_ring_attn_aot_store_f64(void *buffer, int64_t index, double value) {
  memcpy((unsigned char *)buffer + (size_t)index * sizeof(value), &value,
         sizeof(value));
}

double et_ring_attn_aot_load_f64(const void *buffer, int64_t index) {
  double value = 0.0;
  memcpy(&value,
         (const unsigned char *)buffer + (size_t)index * sizeof(value),
         sizeof(value));
  return value;
}
