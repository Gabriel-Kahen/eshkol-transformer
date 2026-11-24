#ifndef ESHKOL_TRANSFORMER_PIPELINE_1F1B_ABI_H
#define ESHKOL_TRANSFORMER_PIPELINE_1F1B_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_PP_ABI_MAJOR 1u
#define ET_PP_ABI_MINOR 0u

#define ET_PP_MIN_STAGES 2u
#define ET_PP_MAX_STAGES 4u
#define ET_PP_MAX_MICROBATCHES 32u
#define ET_PP_MAX_SCHEDULE_STEPS (2u * ET_PP_MAX_MICROBATCHES)
#define ET_PP_MAX_LAYERS_PER_STAGE 8u
#define ET_PP_MAX_TOTAL_LAYERS 32u
#define ET_PP_MAX_D_MODEL 64u
#define ET_PP_MAX_D_FF 128u
#define ET_PP_MAX_TOKENS_PER_MB 32u
#define ET_PP_MAX_OUT_DIM 64u
#define ET_PP_MAX_TRAIN_STEPS 128u
#define ET_PP_MAX_TIMEOUT_MS 60000u

#define ET_PP_WIRE_MAGIC UINT64_C(0x4554505031463142)

typedef int32_t et_pp_status_t;

enum et_pp_status {
  ET_PP_OK = 0,
  ET_PP_ERR_INVALID_ARG = -1,
  ET_PP_ERR_INVALID_STAGES = -2,
  ET_PP_ERR_INVALID_MICROBATCHES = -3,
  ET_PP_ERR_SHAPE = -4,
  ET_PP_ERR_NONFINITE = -5,
  ET_PP_ERR_PROTOCOL = -6,
  ET_PP_ERR_TIMEOUT = -7,
  ET_PP_ERR_PEER_DISCONNECTED = -8,
  ET_PP_ERR_IO = -9,
  ET_PP_ERR_ALLOC = -10,
  ET_PP_ERR_SCHEDULE = -11,
  ET_PP_ERR_TARGET_RANGE = -12
};

enum et_pp_schedule_kind {
  ET_PP_SCHEDULE_1F1B = 0,
  ET_PP_SCHEDULE_GPIPE = 1
};

enum et_pp_step_op {
  ET_PP_OP_FORWARD = 1,
  ET_PP_OP_BACKWARD = 2
};

enum et_pp_phase {
  ET_PP_PHASE_WARMUP = 0,
  ET_PP_PHASE_STEADY = 1,
  ET_PP_PHASE_COOLDOWN = 2
};

enum et_pp_loss_type {
  ET_PP_LOSS_CROSS_ENTROPY = 0,
  ET_PP_LOSS_MSE = 1
};

enum et_pp_msg_type {
  ET_PP_MSG_FORWARD_ACT = 1,
  ET_PP_MSG_BACKWARD_GRAD = 2,
  ET_PP_MSG_STAGE_RESULT = 3,
  ET_PP_MSG_ABORT = 4
};

/*
 * Standard-layout wire frame header sent over AF_UNIX SOCK_STREAM links
 * between adjacent pipeline stages (s -> s+1 forward, s+1 -> s backward).
 */
typedef struct et_pp_frame_header {
  uint64_t magic;
  uint32_t msg_type;
  uint32_t microbatch_id;
  uint32_t src_stage;
  uint32_t dst_stage;
  uint64_t payload_bytes;
} et_pp_frame_header_t;

typedef struct et_pp_schedule_step {
  int32_t op;             /* ET_PP_OP_FORWARD or ET_PP_OP_BACKWARD */
  int32_t phase;          /* ET_PP_PHASE_WARMUP / STEADY / COOLDOWN */
  uint32_t microbatch_id; /* 0 .. num_microbatches - 1 */
  uint32_t live_after;    /* live in-flight activation slots after this step */
} et_pp_schedule_step_t;

typedef struct et_pp_schedule {
  uint32_t num_stages;
  uint32_t stage_id;
  uint32_t num_microbatches;
  int32_t schedule_kind;
  uint32_t num_warmup_steps;
  uint32_t num_steady_pairs;
  uint32_t num_cooldown_steps;
  uint32_t step_count;
  uint32_t peak_live_activations;
  double bubble_fraction;
  et_pp_schedule_step_t steps[ET_PP_MAX_SCHEDULE_STEPS];
} et_pp_schedule_t;

typedef struct et_pp_config {
  uint32_t abi_major;
  uint32_t abi_minor;
  uint32_t num_stages;       /* 2 or 4 for multi-process; 1, 2, or 4 for reference */
  uint32_t layers_per_stage; /* 1 .. ET_PP_MAX_LAYERS_PER_STAGE */
  uint32_t num_microbatches; /* num_stages .. ET_PP_MAX_MICROBATCHES */
  uint32_t tokens_per_mb;    /* 1 .. ET_PP_MAX_TOKENS_PER_MB */
  uint32_t d_model;          /* 2 .. ET_PP_MAX_D_MODEL */
  uint32_t d_ff;             /* 2 .. ET_PP_MAX_D_FF */
  uint32_t out_dim;          /* 2 .. ET_PP_MAX_OUT_DIM (vocab_size for CE) */
  int32_t schedule_kind;     /* ET_PP_SCHEDULE_1F1B or ET_PP_SCHEDULE_GPIPE */
  int32_t loss_type;         /* ET_PP_LOSS_CROSS_ENTROPY or ET_PP_LOSS_MSE */
  uint32_t timeout_ms;       /* 1 .. ET_PP_MAX_TIMEOUT_MS */
  double rms_eps;            /* > 0.0, finite */
} et_pp_config_t;

typedef struct et_pp_run_telemetry {
  double total_loss;
  double bubble_fraction;
  uint32_t num_stages;
  uint32_t num_microbatches;
  int32_t schedule_kind;
  uint32_t distinct_pid_count;
  uint32_t stage_peak_live[ET_PP_MAX_STAGES];
  uint32_t stage_final_live[ET_PP_MAX_STAGES];
  uint32_t stage_fwd_count[ET_PP_MAX_STAGES];
  uint32_t stage_bwd_count[ET_PP_MAX_STAGES];
  uint64_t stage_bytes_sent[ET_PP_MAX_STAGES];
  uint64_t stage_bytes_recv[ET_PP_MAX_STAGES];
  int32_t stage_pids[ET_PP_MAX_STAGES];
} et_pp_run_telemetry_t;

/*
 * Build a deterministic 1F1B or GPipe execution schedule for stage_id in
 * 0 .. num_stages - 1 with num_stages in {2, 4} and num_microbatches in
 * [num_stages, ET_PP_MAX_MICROBATCHES].
 *
 * For 1F1B:
 *   - num_warmup = P - 1 - s forward steps
 *   - num_steady_pairs = M - num_warmup interleaved (F, B) pairs
 *   - num_cooldown = num_warmup backward steps
 *   - peak_live_activations = P - s <= P
 * For GPipe:
 *   - M forward steps followed by M backward steps
 *   - peak_live_activations = M
 * Bubble fraction for both: (P - 1) / (M + P - 1).
 */
et_pp_status_t et_pp_build_schedule_v1(uint32_t num_stages, uint32_t stage_id,
                                       uint32_t num_microbatches,
                                       int32_t schedule_kind,
                                       et_pp_schedule_t *out_schedule);

/* Validate pipeline configuration (requires num_stages in {2, 4}). */
et_pp_status_t et_pp_validate_config_v1(const et_pp_config_t *cfg);

/* Parameter count for one Pre-RMSNorm + SwiGLU residual layer. */
size_t et_pp_layer_param_count_v1(uint32_t d_model, uint32_t d_ff);

/*
 * Total flat parameter count for all total_layers = num_stages * layers_per_stage
 * plus the final linear output head (w_head[out_dim * d_model], b_head[out_dim]).
 */
et_pp_status_t et_pp_param_count_v1(const et_pp_config_t *cfg,
                                    size_t *out_count);

/*
 * Single differentiable Pre-RMSNorm + SwiGLU residual layer forward/backward.
 * Tensor shapes:
 *   x, dy, y, dx: [tokens, d_model] binary64
 *   layer_params, dlayer_params: [d_model + 3 * d_ff * d_model] binary64
 *     layout: rms_weight[d_model], w_gate[d_ff * d_model],
 *             w_up[d_ff * d_model], w_down[d_model * d_ff]
 * dlayer_params is zeroed and written with exact analytical gradients.
 */
et_pp_status_t et_pp_layer_forward_v1(uint32_t tokens, uint32_t d_model,
                                      uint32_t d_ff, double rms_eps,
                                      const double *x,
                                      const double *layer_params, double *y);

et_pp_status_t et_pp_layer_backward_v1(uint32_t tokens, uint32_t d_model,
                                       uint32_t d_ff, double rms_eps,
                                       const double *x,
                                       const double *layer_params,
                                       const double *dy, double *dx,
                                       double *dlayer_params);

/*
 * Framed AF_UNIX SOCK_STREAM point-to-point send/recv with monotonic timeout.
 * recv_frame verifies magic, msg_type, microbatch_id, src_stage, dst_stage,
 * payload_bytes, and finite binary64 payloads for activation/gradient frames.
 */
et_pp_status_t et_pp_p2p_send_frame_v1(int fd, uint32_t msg_type,
                                       uint32_t microbatch_id,
                                       uint32_t src_stage, uint32_t dst_stage,
                                       const void *payload,
                                       uint64_t payload_bytes,
                                       uint32_t timeout_ms);

et_pp_status_t et_pp_p2p_recv_frame_v1(int fd, uint32_t expected_msg_type,
                                       uint32_t expected_microbatch_id,
                                       uint32_t expected_src_stage,
                                       uint32_t expected_dst_stage,
                                       void *payload_buf,
                                       uint64_t expected_payload_bytes,
                                       uint32_t timeout_ms);

/*
 * Single-process unpipelined full-model forward + backward reference.
 * Accepts cfg->num_stages in {1, 2, 4} and runs all
 * total_layers = num_stages * layers_per_stage sequentially over microbatches
 * 0 .. num_microbatches - 1, averaging loss by 1 / M.
 *
 * targets_i32: [M * tokens_per_mb] when loss_type == ET_PP_LOSS_CROSS_ENTROPY
 * targets_f64: [M * tokens_per_mb * out_dim] when loss_type == ET_PP_LOSS_MSE
 */
et_pp_status_t et_pp_run_reference_step_v1(const et_pp_config_t *cfg,
                                           const double *inputs,
                                           const int32_t *targets_i32,
                                           const double *targets_f64,
                                           const double *params,
                                           double *out_grads,
                                           double *out_loss);

/*
 * Real multi-process pipeline parallel forward + backward across P in {2, 4}
 * OS child processes connected exclusively by AF_UNIX SOCK_STREAM socketpairs.
 * Stage s executes the deterministic 1F1B (or GPipe) schedule using at most
 * sched.peak_live_activations live activation slots.
 */
et_pp_status_t et_pp_run_multiprocess_step_v1(
    const et_pp_config_t *cfg, const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *params, double *out_grads,
    et_pp_run_telemetry_t *out_telemetry);

/*
 * Single-process unpipelined multi-step SGD reference.
 */
et_pp_status_t et_pp_train_reference_sgd_v1(
    const et_pp_config_t *cfg, uint32_t num_steps, double lr,
    const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *init_params,
    double *out_final_params, double *out_step_losses);

/*
 * Real multi-process P in {2, 4} multi-step SGD over persistent AF_UNIX stage
 * workers. Each stage updates its own parameter shard in-place after every
 * 1F1B step and returns final weights and per-step losses over control sockets.
 */
et_pp_status_t et_pp_train_multiprocess_sgd_v1(
    const et_pp_config_t *cfg, uint32_t num_steps, double lr,
    const double *inputs, const int32_t *targets_i32,
    const double *targets_f64, const double *init_params,
    double *out_final_params, double *out_step_losses,
    et_pp_run_telemetry_t *out_last_telemetry);

#ifdef __cplusplus
}
#endif

#endif
