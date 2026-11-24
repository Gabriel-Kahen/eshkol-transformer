#ifndef ESHKOL_TRANSFORMER_FLASH_ATTN2_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_FLASH_ATTN2_ESHKOL_BRIDGE_H

#include "eshkol_transformer/flash_attn2_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT bridge for FlashAttention-2 Exact Tiled Online-Softmax Attention.
 * All struct and tensor pointers are accepted as void * so raw Eshkol FFI
 * addresses (including intentionally misaligned ptr-add addresses) are
 * validated cleanly without misaligned-pointer UB. Zero heap allocations.
 */
int64_t et_fa2_aot_config_bytes(void);
int64_t et_fa2_aot_stats_bytes(void);
int64_t et_fa2_aot_tile_scratch_bytes(int64_t tile_r, int64_t tile_c);

int32_t et_fa2_aot_set_config(void *cfg_buf, int64_t batch, int64_t num_heads,
                              int64_t num_kv_heads, int64_t seq_q,
                              int64_t seq_k, int64_t head_dim, int64_t tile_r,
                              int64_t tile_c, int64_t causal,
                              int64_t sliding_window, double scale,
                              double logit_softcap);

int32_t et_fa2_aot_forward(const void *cfg_buf, const void *q, const void *k,
                           const void *v, void *o_out, void *lse_out,
                           void *stats_buf);

int32_t et_fa2_aot_backward(const void *cfg_buf, const void *q, const void *k,
                            const void *v, const void *o, const void *lse,
                            const void *do_in, void *dq_out, void *dk_out,
                            void *dv_out, void *stats_buf);

/*
 * Negative FP-environment test helper: switches rounding mode to FE_TOWARDZERO,
 * invokes et_fa2_forward_v1, restores the caller's FP environment, and returns
 * the status (expected ET_FA2_ERR_FP_ENV).
 */
int32_t et_fa2_aot_check_bad_fpenv_forward(const void *cfg_buf, const void *q,
                                           const void *k, const void *v,
                                           void *o_out, void *lse_out,
                                           void *stats_buf);

/* Telemetry accessors for et_fa2_stats_v1 buffers. */
int64_t et_fa2_aot_stats_total_tiles(const void *stats_buf);
int64_t et_fa2_aot_stats_evaluated_tiles(const void *stats_buf);
int64_t et_fa2_aot_stats_skipped_tiles(const void *stats_buf);
int64_t et_fa2_aot_stats_tile_scratch_bytes(const void *stats_buf);

/* Scalar marshaling between Eshkol f64 values and owned f32 / f64 buffers. */
void et_fa2_aot_store_f32(void *buffer, int64_t index, double value);
double et_fa2_aot_load_f32(const void *buffer, int64_t index);
void et_fa2_aot_store_f64(void *buffer, int64_t index, double value);
double et_fa2_aot_load_f64(const void *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
