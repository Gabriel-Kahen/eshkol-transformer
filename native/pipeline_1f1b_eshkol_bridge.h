#ifndef ESHKOL_TRANSFORMER_PIPELINE_1F1B_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_PIPELINE_1F1B_ESHKOL_BRIDGE_H

#include "eshkol_transformer/pipeline_1f1b_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Eshkol AOT FFI bridge for the standalone 1F1B Pipeline Parallelism Engine.
 * All tensor/metric pointers are accepted as void * and validated before use.
 * Returns 0 (ET_PP_OK) on success or a negative et_pp_status_t code on error.
 */

int64_t et_pp_aot_param_count(int64_t total_layers, int64_t d_model,
                              int64_t d_ff, int64_t out_dim);

/*
 * Writes 5 int64_t values to out_summary_i64:
 *   [0] num_warmup_steps, [1] num_steady_pairs, [2] num_cooldown_steps,
 *   [3] step_count, [4] peak_live_activations
 * Writes 1 double to out_bubble_f64: bubble_fraction.
 */
int32_t et_pp_aot_build_schedule(int64_t num_stages, int64_t stage_id,
                                 int64_t num_microbatches,
                                 int64_t schedule_kind, void *out_summary_i64,
                                 void *out_bubble_f64);

int32_t et_pp_aot_layer_forward(int64_t tokens, int64_t d_model, int64_t d_ff,
                                double rms_eps, const void *x,
                                const void *layer_params, void *y);

int32_t et_pp_aot_layer_backward(int64_t tokens, int64_t d_model, int64_t d_ff,
                                 double rms_eps, const void *x,
                                 const void *layer_params, const void *dy,
                                 void *dx, void *dlayer_params);

int32_t et_pp_aot_run_reference_step(
    int64_t total_layers, int64_t num_microbatches, int64_t tokens_per_mb,
    int64_t d_model, int64_t d_ff, int64_t out_dim, int64_t loss_type,
    double rms_eps, const void *inputs, const void *targets, const void *params,
    void *out_grads, void *out_loss_f64);

/*
 * out_metrics_f64 (2 doubles): [0] total_loss, [1] bubble_fraction
 * out_stage_metrics_i64 (17 int64_t):
 *   [0..3]   stage_peak_live[0..3]
 *   [4..7]   stage_final_live[0..3]
 *   [8..11]  stage_bytes_sent[0..3]
 *   [12..15] stage_bytes_recv[0..3]
 *   [16]     distinct_pid_count
 */
int32_t et_pp_aot_run_multiprocess_step(
    int64_t num_stages, int64_t layers_per_stage, int64_t num_microbatches,
    int64_t tokens_per_mb, int64_t d_model, int64_t d_ff, int64_t out_dim,
    int64_t schedule_kind, int64_t loss_type, int64_t timeout_ms,
    double rms_eps, const void *inputs, const void *targets, const void *params,
    void *out_grads, void *out_metrics_f64, void *out_stage_metrics_i64);

int32_t et_pp_aot_train_reference_sgd(
    int64_t total_layers, int64_t num_microbatches, int64_t tokens_per_mb,
    int64_t d_model, int64_t d_ff, int64_t out_dim, int64_t loss_type,
    int64_t num_steps, double lr, double rms_eps, const void *inputs,
    const void *targets, const void *init_params, void *out_final_params,
    void *out_step_losses);

/*
 * out_peak_live_i64 (5 int64_t):
 *   [0..3] stage_peak_live[0..3]
 *   [4]    distinct_pid_count
 */
int32_t et_pp_aot_train_multiprocess_sgd(
    int64_t num_stages, int64_t layers_per_stage, int64_t num_microbatches,
    int64_t tokens_per_mb, int64_t d_model, int64_t d_ff, int64_t out_dim,
    int64_t schedule_kind, int64_t loss_type, int64_t num_steps,
    int64_t timeout_ms, double lr, double rms_eps, const void *inputs,
    const void *targets, const void *init_params, void *out_final_params,
    void *out_step_losses, void *out_peak_live_i64);

/*
 * Real AF_UNIX socketpair negative transport probes (with FD-neutral cleanup):
 *   scenario 0: mismatched microbatch_id -> ET_PP_ERR_PROTOCOL (-6)
 *   scenario 1: peer disconnect          -> ET_PP_ERR_PEER_DISCONNECTED (-8)
 *   scenario 2: monotonic timeout        -> ET_PP_ERR_TIMEOUT (-7)
 *   scenario 3: non-finite wire payload  -> ET_PP_ERR_NONFINITE (-5)
 */
int32_t et_pp_aot_probe_transport_error(int64_t scenario, int64_t timeout_ms);

/* Alignment-safe scalar marshaling for Eshkol AOT buffers. */
void et_pp_aot_store_f64(void *buf, int64_t index, double value);
double et_pp_aot_load_f64(const void *buf, int64_t index);
void et_pp_aot_store_i32(void *buf, int64_t index, int64_t value);
int64_t et_pp_aot_load_i64(const void *buf, int64_t index);
double et_pp_aot_make_nonfinite(int64_t kind);

#ifdef __cplusplus
}
#endif

#endif
