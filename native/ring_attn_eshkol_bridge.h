#ifndef ESHKOL_TRANSFORMER_RING_ATTN_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_RING_ATTN_ESHKOL_BRIDGE_H

#include "eshkol_transformer/ring_attn_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT bridge for Ring Attention / Sequence-Parallel Causal Attention.
 * All struct and tensor pointers are accepted as void * so raw Eshkol FFI
 * addresses (including intentionally misaligned ptr-add addresses) are
 * validated cleanly without misaligned-pointer UB.
 */
int64_t et_ring_attn_aot_config_bytes(void);
int64_t et_ring_attn_aot_stats_bytes(void);

int32_t et_ring_attn_aot_set_config(void *cfg_buf, int64_t batch,
                                    int64_t seq_len, int64_t num_heads,
                                    int64_t num_kv_heads, int64_t head_dim,
                                    int64_t world_size, int64_t causal,
                                    double scale, int64_t timeout_ms);

int32_t et_ring_attn_aot_dense_forward(const void *cfg_buf, const void *q,
                                       const void *k, const void *v, void *o,
                                       void *lse);

int32_t et_ring_attn_aot_dense_backward(const void *cfg_buf, const void *q,
                                        const void *k, const void *v,
                                        const void *o, const void *lse,
                                        const void *do_buf, void *dq, void *dk,
                                        void *dv);

int32_t et_ring_attn_aot_mp_forward(const void *cfg_buf, const void *q,
                                    const void *k, const void *v, void *o,
                                    void *lse, void *stats_buf);

int32_t et_ring_attn_aot_mp_backward(const void *cfg_buf, const void *q,
                                     const void *k, const void *v,
                                     const void *o, const void *lse,
                                     const void *do_buf, void *dq, void *dk,
                                     void *dv, void *stats_buf);

/* Telemetry accessors for et_ring_attn_stats_v1 buffers. */
int64_t et_ring_attn_aot_stats_total_blocks(const void *stats_buf);
int64_t et_ring_attn_aot_stats_executed_blocks(const void *stats_buf);
int64_t et_ring_attn_aot_stats_skipped_causal(const void *stats_buf);
int64_t et_ring_attn_aot_stats_ring_rounds(const void *stats_buf);
int64_t et_ring_attn_aot_stats_bytes_sent(const void *stats_buf);
int64_t et_ring_attn_aot_stats_bytes_received(const void *stats_buf);

/* Scalar marshaling between Eshkol f64 values and owned f32 / f64 buffers. */
void et_ring_attn_aot_store_f32(void *buffer, int64_t index, double value);
double et_ring_attn_aot_load_f32(const void *buffer, int64_t index);
void et_ring_attn_aot_store_f64(void *buffer, int64_t index, double value);
double et_ring_attn_aot_load_f64(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
