#ifndef ESHKOL_TRANSFORMER_TENSOR_PARALLEL_ABI_H
#define ESHKOL_TRANSFORMER_TENSOR_PARALLEL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_TP_ABI_MAJOR 1u
#define ET_TP_ABI_MINOR 0u
#define ET_TP_MAX_WORLD_SIZE 4u
#define ET_TP_MAX_COLLECTIVE_BYTES (64u * 1024u * 1024u)

typedef struct et_tp_group et_tp_group_t;
typedef int32_t et_tp_status_t;

enum {
  ET_TP_OK = 0,
  ET_TP_ERR_INVALID_ARG = -1,
  ET_TP_ERR_SHAPE = -2,
  ET_TP_ERR_DIVISIBILITY = -3,
  ET_TP_ERR_ALIAS = -4,
  ET_TP_ERR_ALIGNMENT = -5,
  ET_TP_ERR_NONFINITE = -6,
  ET_TP_ERR_OVERFLOW = -7,
  ET_TP_ERR_MISMATCH = -8,
  ET_TP_ERR_TIMEOUT = -9,
  ET_TP_ERR_PEER_DISCONNECTED = -10,
  ET_TP_ERR_IO = -11,
  ET_TP_ERR_ALLOC = -12
};

typedef struct et_tp_collective_stats {
  uint64_t allreduce_calls;
  uint64_t allgather_calls;
  uint64_t ipc_allreduce_calls;
  uint64_t ipc_allgather_calls;
  uint64_t allreduce_elements;
  uint64_t allgather_elements;
  uint64_t allreduce_bytes;
  uint64_t allgather_bytes;
} et_tp_collective_stats_t;

typedef struct et_tp_swiglu_config {
  uint32_t tokens;
  uint32_t d_model;
  uint32_t d_ff;
} et_tp_swiglu_config_t;

typedef struct et_tp_gqa_config {
  uint32_t batch;
  uint32_t seq_len;
  uint32_t d_model;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
} et_tp_gqa_config_t;

typedef struct et_tp_block_config {
  uint32_t batch;
  uint32_t seq_len;
  uint32_t d_model;
  uint32_t d_ff;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  uint32_t head_dim;
  float rms_eps;
} et_tp_block_config_t;

typedef struct et_tp_block_memory_stats {
  uint64_t full_sharded_weight_bytes;
  uint64_t rank_sharded_weight_bytes;
  uint64_t replicated_weight_bytes;
  uint64_t rank_total_weight_bytes;
} et_tp_block_memory_stats_t;

/*
 * Per-rank Transformer block weights (or full dense weights when world_size==1).
 * Sharded ColumnParallel: w_q [q_dim_local, d_model], w_k [kv_dim_local, d_model],
 *                         w_v [kv_dim_local, d_model], w_gate [d_ff_local, d_model],
 *                         w_up [d_ff_local, d_model].
 * Sharded RowParallel:    w_o [d_model, q_dim_local], w_down [d_model, d_ff_local].
 * Replicated:             gamma1 [d_model], gamma2 [d_model].
 */
typedef struct et_tp_block_weights {
  float *gamma1;
  float *w_q;
  float *w_k;
  float *w_v;
  float *w_o;
  float *gamma2;
  float *w_gate;
  float *w_up;
  float *w_down;
} et_tp_block_weights_t;

typedef struct et_tp_block_grads {
  float *dgamma1;
  float *dw_q;
  float *dw_k;
  float *dw_v;
  float *dw_o;
  float *dgamma2;
  float *dw_gate;
  float *dw_up;
  float *dw_down;
} et_tp_block_grads_t;

/*
 * Forward activation cache required by et_tp_block_backward_v1.
 * All buffers are rank-local except replicated activations (x, norm1, h1, norm2).
 *   x        [B*T, d_model]
 *   norm1    [B*T, d_model]
 *   q_r      [B*T, q_dim_local]
 *   k_r      [B*T, kv_dim_local]
 *   v_r      [B*T, kv_dim_local]
 *   attn_r   [B*T, q_dim_local]
 *   h1       [B*T, d_model]
 *   norm2    [B*T, d_model]
 *   gate_r   [B*T, d_ff_local]
 *   up_r     [B*T, d_ff_local]
 */
typedef struct et_tp_block_cache {
  float *x;
  float *norm1;
  float *q_r;
  float *k_r;
  float *v_r;
  float *attn_r;
  float *h1;
  float *norm2;
  float *gate_r;
  float *up_r;
} et_tp_block_cache_t;

/* -------------------------------------------------------------------------
 * Multi-process AF_UNIX Tensor Parallel Collective Group (W in {1, 2, 4})
 * ------------------------------------------------------------------------- */

/*
 * Create a TP collective group over pre-connected AF_UNIX SOCK_STREAM sockets.
 * world_size must be 1, 2, or 4.
 *   W == 1: fd_count == 0, fds may be NULL.
 *   W > 1, rank == 0: fd_count == W - 1, fds[i] connected to peer rank (i + 1).
 *   W > 1, rank > 0:  fd_count == 1, fds[0] connected to rank 0.
 * Sockets are duplicated with F_DUPFD_CLOEXEC; caller closes originals.
 * Zero FD leaks on any error.
 */
et_tp_status_t et_tp_group_create(uint32_t abi_major, uint32_t abi_minor,
                                  uint32_t rank, uint32_t world_size,
                                  const int *fds, size_t fd_count,
                                  int32_t default_timeout_ms,
                                  et_tp_group_t **out_group);

void et_tp_group_destroy(et_tp_group_t *group);

et_tp_status_t et_tp_group_rank(const et_tp_group_t *group, uint32_t *out_rank);
et_tp_status_t et_tp_group_world_size(const et_tp_group_t *group,
                                      uint32_t *out_world_size);
et_tp_status_t et_tp_group_get_stats(const et_tp_group_t *group,
                                     et_tp_collective_stats_t *out_stats);
et_tp_status_t et_tp_group_reset_stats(et_tp_group_t *group);
et_tp_status_t et_tp_group_set_timeout_ms(et_tp_group_t *group,
                                          int32_t timeout_ms);

/*
 * Canonical ascending rank-order binary64-accumulated AllReduce SUM over
 * AF_UNIX sockets. Produces bitwise-identical binary32 outputs on all ranks.
 * Exact in-place (sendbuf == recvbuf) is supported; partial overlap fails with
 * ET_TP_ERR_ALIAS. Non-finite inputs fail atomically with ET_TP_ERR_NONFINITE
 * without modifying recvbuf.
 */
et_tp_status_t et_tp_allreduce_sum_f32(et_tp_group_t *group,
                                       const float *sendbuf, float *recvbuf,
                                       size_t count, uint64_t tag,
                                       int32_t timeout_ms);

/*
 * AllGather over AF_UNIX sockets in ascending rank order [r=0 .. W-1].
 * sendbuf [send_count] and recvbuf [W * send_count] must be disjoint.
 */
et_tp_status_t et_tp_allgather_f32(et_tp_group_t *group, const float *sendbuf,
                                   size_t send_count, float *recvbuf,
                                   uint64_t tag, int32_t timeout_ms);

/* -------------------------------------------------------------------------
 * 1. ColumnParallelLinear
 *    Weight W_full [out_features, in_features] sharded by output rows:
 *    W_r [out_features / W, in_features], optional b_r [out_features / W].
 *    Forward:  0 AllReduces.
 *    Backward: 1 AllReduce on dX.
 * ------------------------------------------------------------------------- */

et_tp_status_t et_tp_column_linear_forward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x, const float *w_r, const float *b_r,
    float *y_r);

et_tp_status_t et_tp_column_linear_backward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x, const float *w_r, const float *dy_r,
    float *dx, float *dw_r, float *db_r, uint64_t tag, int32_t timeout_ms);

/* -------------------------------------------------------------------------
 * 2. RowParallelLinear
 *    Weight W_full [out_features, in_features] sharded by input columns:
 *    W_r [out_features, in_features / W], optional unsharded b [out_features].
 *    Forward:  1 AllReduce on Y_partial_r, then add b once.
 *    Backward: 0 AllReduces.
 * ------------------------------------------------------------------------- */

et_tp_status_t et_tp_row_linear_forward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x_r, const float *w_r, const float *b,
    float *y, uint64_t tag, int32_t timeout_ms);

et_tp_status_t et_tp_row_linear_backward_v1(
    et_tp_group_t *group, uint32_t rows, uint32_t in_features,
    uint32_t out_features, const float *x_r, const float *w_r, const float *dy,
    float *dx_r, float *dw_r, float *db);

/* -------------------------------------------------------------------------
 * 3. TP-Sharded SwiGLU MLP Block
 *    ColumnParallel(w_gate_r, w_up_r) -> SiLU(gate_r) * up_r ->
 *    RowParallel(w_down_r).
 *    Forward:  1 AllReduce.
 *    Backward: 1 AllReduce (fused dX_local from gate + up).
 * ------------------------------------------------------------------------- */

et_tp_status_t et_tp_swiglu_validate_config(const et_tp_swiglu_config_t *cfg,
                                            uint32_t world_size);

et_tp_status_t et_tp_swiglu_forward_v1(
    et_tp_group_t *group, const et_tp_swiglu_config_t *cfg, const float *x,
    const float *w_gate_r, const float *w_up_r, const float *w_down_r,
    float *gate_cache_r, float *up_cache_r, float *y, uint64_t tag,
    int32_t timeout_ms);

et_tp_status_t et_tp_swiglu_backward_v1(
    et_tp_group_t *group, const et_tp_swiglu_config_t *cfg, const float *x,
    const float *w_gate_r, const float *w_up_r, const float *w_down_r,
    const float *gate_cache_r, const float *up_cache_r, const float *dy,
    float *dx, float *dw_gate_r, float *dw_up_r, float *dw_down_r, uint64_t tag,
    int32_t timeout_ms);

/* -------------------------------------------------------------------------
 * 4. TP-Sharded Causal GQA Attention Block
 *    Head-sharded Q_r, K_r, V_r -> local causal GQA SDPA -> RowParallel(w_o_r).
 *    Requires H_q % W == 0, H_kv % W == 0, H_q % H_kv == 0, d_model == H_q*D_h.
 *    Forward:  1 AllReduce.
 *    Backward: 1 AllReduce (fused dX_local from Q + K + V).
 * ------------------------------------------------------------------------- */

et_tp_status_t et_tp_gqa_validate_config(const et_tp_gqa_config_t *cfg,
                                         uint32_t world_size);

et_tp_status_t et_tp_gqa_forward_v1(
    et_tp_group_t *group, const et_tp_gqa_config_t *cfg, const float *x,
    const float *w_q_r, const float *w_k_r, const float *w_v_r,
    const float *w_o_r, float *q_cache_r, float *k_cache_r, float *v_cache_r,
    float *attn_cache_r, float *y, uint64_t tag, int32_t timeout_ms);

et_tp_status_t et_tp_gqa_backward_v1(
    et_tp_group_t *group, const et_tp_gqa_config_t *cfg, const float *x,
    const float *w_q_r, const float *w_k_r, const float *w_v_r,
    const float *w_o_r, const float *q_cache_r, const float *k_cache_r,
    const float *v_cache_r, const float *attn_cache_r, const float *dy,
    float *dx, float *dw_q_r, float *dw_k_r, float *dw_v_r, float *dw_o_r,
    uint64_t tag, int32_t timeout_ms);

/* -------------------------------------------------------------------------
 * 5. Full TP Transformer Block (RMSNorm + TP-GQA + Res + RMSNorm + TP-SwiGLU + Res)
 *    Forward:  2 AllReduces (1 GQA + 1 SwiGLU).
 *    Backward: 2 AllReduces (1 SwiGLU + 1 GQA).
 * ------------------------------------------------------------------------- */

et_tp_status_t et_tp_rmsnorm_forward_v1(uint32_t rows, uint32_t d_model,
                                        float eps, const float *x,
                                        const float *gamma, float *y);

et_tp_status_t et_tp_rmsnorm_backward_v1(uint32_t rows, uint32_t d_model,
                                         float eps, const float *x,
                                         const float *gamma, const float *dy,
                                         float *dx, float *dgamma);

et_tp_status_t et_tp_block_validate_config(const et_tp_block_config_t *cfg,
                                           uint32_t world_size);

et_tp_status_t et_tp_block_memory_stats_v1(
    const et_tp_block_config_t *cfg, uint32_t world_size,
    et_tp_block_memory_stats_t *out_stats);

et_tp_status_t et_tp_block_shard_weights_v1(
    const et_tp_block_config_t *cfg, uint32_t rank, uint32_t world_size,
    const et_tp_block_weights_t *full_weights,
    et_tp_block_weights_t *rank_weights);

et_tp_status_t et_tp_block_gather_weights_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *rank_weights,
    et_tp_block_weights_t *full_weights, uint64_t tag_base, int32_t timeout_ms);

et_tp_status_t et_tp_block_forward_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *weights, const float *x, float *y,
    et_tp_block_cache_t *cache, uint64_t tag_base, int32_t timeout_ms);

et_tp_status_t et_tp_block_backward_v1(
    et_tp_group_t *group, const et_tp_block_config_t *cfg,
    const et_tp_block_weights_t *weights, const et_tp_block_cache_t *cache,
    const float *dy, float *dx, et_tp_block_grads_t *grads, uint64_t tag_base,
    int32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif
