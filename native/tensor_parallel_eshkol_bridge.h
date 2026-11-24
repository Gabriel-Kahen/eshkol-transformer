#ifndef ESHKOL_TRANSFORMER_TENSOR_PARALLEL_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_TENSOR_PARALLEL_ESHKOL_BRIDGE_H

#include "eshkol_transformer/tensor_parallel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_tp_aot_session et_tp_aot_session_t;

/*
 * Parameter IDs for full/gathered block weights and gradients:
 *   0: gamma1 [d_model]
 *   1: w_q    [num_q_heads * head_dim, d_model]
 *   2: w_k    [num_kv_heads * head_dim, d_model]
 *   3: w_v    [num_kv_heads * head_dim, d_model]
 *   4: w_o    [d_model, num_q_heads * head_dim]
 *   5: gamma2 [d_model]
 *   6: w_gate [d_ff, d_model]
 *   7: w_up   [d_ff, d_model]
 *   8: w_down [d_model, d_ff]
 */
enum {
  ET_TP_AOT_PARAM_GAMMA1 = 0,
  ET_TP_AOT_PARAM_WQ = 1,
  ET_TP_AOT_PARAM_WK = 2,
  ET_TP_AOT_PARAM_WV = 3,
  ET_TP_AOT_PARAM_WO = 4,
  ET_TP_AOT_PARAM_GAMMA2 = 5,
  ET_TP_AOT_PARAM_WGATE = 6,
  ET_TP_AOT_PARAM_WUP = 7,
  ET_TP_AOT_PARAM_WDOWN = 8,
  ET_TP_AOT_PARAM_COUNT = 9
};

/*
 * Create a TP Transformer block training session with world_size in {1, 2, 4}.
 * For world_size > 1, spawns (world_size - 1) real OS worker processes
 * connected over AF_UNIX SOCK_STREAM sockets (zero shared memory).
 */
int32_t et_tp_aot_session_create(int64_t world_size, int64_t batch,
                                 int64_t seq_len, int64_t d_model, int64_t d_ff,
                                 int64_t num_q_heads, int64_t num_kv_heads,
                                 int64_t head_dim, double rms_eps,
                                 et_tp_aot_session_t **out_session);

void et_tp_aot_session_destroy(et_tp_aot_session_t *session);

/* Forward, MSE loss, backward, SGD update, and AllGather across all ranks. */
int32_t et_tp_aot_forward(et_tp_aot_session_t *session);
double et_tp_aot_mse_loss(et_tp_aot_session_t *session);
int32_t et_tp_aot_backward(et_tp_aot_session_t *session);
int32_t et_tp_aot_sgd_step(et_tp_aot_session_t *session, double lr);
int32_t et_tp_aot_gather_weights(et_tp_aot_session_t *session);
int32_t et_tp_aot_gather_grads(et_tp_aot_session_t *session);
int32_t et_tp_aot_reset_stats(et_tp_aot_session_t *session);

/* Collective and memory statistics. */
int64_t et_tp_aot_allreduce_calls(const et_tp_aot_session_t *session);
int64_t et_tp_aot_ipc_allreduce_calls(const et_tp_aot_session_t *session);
int64_t et_tp_aot_allgather_calls(const et_tp_aot_session_t *session);
int64_t et_tp_aot_ipc_allgather_calls(const et_tp_aot_session_t *session);
int64_t et_tp_aot_full_sharded_weight_bytes(const et_tp_aot_session_t *session);
int64_t et_tp_aot_rank_sharded_weight_bytes(const et_tp_aot_session_t *session);
int64_t et_tp_aot_replicated_weight_bytes(const et_tp_aot_session_t *session);

/* Element accessors & single-rank perturbation for Eshkol-driven FD checks. */
int64_t et_tp_aot_output_count(const et_tp_aot_session_t *session);
int64_t et_tp_aot_param_count(const et_tp_aot_session_t *session,
                              int64_t param_id);
double et_tp_aot_get_input(const et_tp_aot_session_t *session, int64_t idx);
int32_t et_tp_aot_set_input(et_tp_aot_session_t *session, int64_t idx,
                            double value);
double et_tp_aot_get_output(const et_tp_aot_session_t *session, int64_t idx);
double et_tp_aot_get_dx(const et_tp_aot_session_t *session, int64_t idx);
double et_tp_aot_get_weight(const et_tp_aot_session_t *session,
                            int64_t param_id, int64_t idx);
int32_t et_tp_aot_set_weight(et_tp_aot_session_t *session, int64_t param_id,
                             int64_t idx, double value);
double et_tp_aot_get_gathered_weight(const et_tp_aot_session_t *session,
                                     int64_t param_id, int64_t idx);
double et_tp_aot_get_gathered_grad(const et_tp_aot_session_t *session,
                                   int64_t param_id, int64_t idx);

/* Cross-world parity comparison helpers (max absolute error). */
double et_tp_aot_max_output_diff(const et_tp_aot_session_t *a,
                                 const et_tp_aot_session_t *b);
double et_tp_aot_max_dx_diff(const et_tp_aot_session_t *a,
                             const et_tp_aot_session_t *b);
double et_tp_aot_max_grad_diff(const et_tp_aot_session_t *a,
                               const et_tp_aot_session_t *b);
double et_tp_aot_max_weight_diff(const et_tp_aot_session_t *a,
                                 const et_tp_aot_session_t *b);

/* Fail-closed negative validation cases returning exact et_tp_status_t. */
int32_t et_tp_aot_check_negative_case(int64_t case_id);

#ifdef __cplusplus
}
#endif

#endif
