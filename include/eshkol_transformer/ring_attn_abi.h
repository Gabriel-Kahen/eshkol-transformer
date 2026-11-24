#ifndef ESHKOL_TRANSFORMER_RING_ATTN_ABI_H
#define ESHKOL_TRANSFORMER_RING_ATTN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_RING_ATTN_ABI_MAJOR 1u
#define ET_RING_ATTN_ABI_MINOR 0u

#define ET_RING_ATTN_MIN_WORLD_SIZE 1u
#define ET_RING_ATTN_MAX_WORLD_SIZE 4u
#define ET_RING_ATTN_MIN_BATCH 1u
#define ET_RING_ATTN_MAX_BATCH 4u
#define ET_RING_ATTN_MIN_SEQ_LEN 1u
#define ET_RING_ATTN_MAX_SEQ_LEN 1024u
#define ET_RING_ATTN_MIN_HEADS 1u
#define ET_RING_ATTN_MAX_HEADS 16u
#define ET_RING_ATTN_MIN_HEAD_DIM 1u
#define ET_RING_ATTN_MAX_HEAD_DIM 128u
#define ET_RING_ATTN_MAX_TIMEOUT_MS 600000

typedef struct et_ring_attn_comm et_ring_attn_comm_t;
typedef int32_t et_ring_attn_status_t;

enum {
  ET_RING_ATTN_OK = 0,
  ET_RING_ATTN_ERR_INVALID_ARG = 1,
  ET_RING_ATTN_ERR_SHAPE_MISMATCH = 2,
  ET_RING_ATTN_ERR_UNSUPPORTED = 3,
  ET_RING_ATTN_ERR_NONFINITE = 4,
  ET_RING_ATTN_ERR_PEER_DISCONNECTED = 5,
  ET_RING_ATTN_ERR_TIMEOUT = 6,
  ET_RING_ATTN_ERR_IO = 7,
  ET_RING_ATTN_ERR_ALLOC = 8,
  ET_RING_ATTN_ERR_PROTOCOL = 9
};

enum {
  ET_RING_ATTN_PHASE_FORWARD = 1u,
  ET_RING_ATTN_PHASE_BACKWARD = 2u,
  ET_RING_ATTN_PHASE_BACKWARD_FINAL = 3u,
  ET_RING_ATTN_PHASE_TEST = 4u
};

/*
 * Configuration for Ring Attention / Sequence-Parallel Causal Attention.
 *
 * Tensor layout (dense row-major):
 *   Global full tensors (seq_len = T):
 *     Q, O, dO, dQ : float[N, T, H, Dh]
 *     K, V, dK, dV : float[N, T, Hkv, Dh]
 *     LSE          : double[N, T, H]
 *
 *   Local rank-r shard tensors (T_local = T / W):
 *     Q_r, O_r, dO_r, dQ_r : float[N, T_local, H, Dh]
 *     K_r, V_r, dK_r, dV_r : float[N, T_local, Hkv, Dh]
 *     LSE_r                : double[N, T_local, H]
 *
 * Constraints:
 *   - world_size (W) in {1, 2, 4}
 *   - seq_len (T) % world_size == 0
 *   - num_heads (H) % num_kv_heads (Hkv) == 0 (supports MHA and GQA)
 *   - causal in {0, 1}
 *   - scale == 0.0 selects default 1.0 / sqrt((double)head_dim);
 *     otherwise scale must be finite and strictly positive.
 *   - timeout_ms in [1, ET_RING_ATTN_MAX_TIMEOUT_MS].
 */
typedef struct et_ring_attn_config_v1 {
  size_t struct_size;
  size_t batch;
  size_t seq_len;
  size_t num_heads;
  size_t num_kv_heads;
  size_t head_dim;
  uint32_t world_size;
  uint32_t causal;
  double scale;
  int32_t timeout_ms;
} et_ring_attn_config_v1;

/*
 * Execution telemetry for a ring pass.
 *
 * For per-rank calls (et_ring_attn_forward_v1 / et_ring_attn_backward_v1):
 *   - total_block_steps     == W
 *   - skipped_causal_blocks == (causal ? (W - 1 - rank) : 0)
 *   - executed_blocks       == total_block_steps - skipped_causal_blocks
 *   - ring_rounds           == (W == 1 ? 0 : (forward ? W - 1 : W))
 *
 * For aggregate multiprocess calls (summed across all W ranks):
 *   - total_block_steps     == W * W
 *   - skipped_causal_blocks == (causal ? W * (W - 1) / 2 : 0)
 *                              (e.g., W=4 causal skips 6 / 16 rank-blocks)
 *   - executed_blocks       == W * W - skipped_causal_blocks
 *   - ring_rounds           == per-rank rounds (W - 1 forward, W backward for W > 1)
 *   - bytes_sent            == total payload bytes sent across all ranks
 *   - bytes_received        == total payload bytes received across all ranks
 */
typedef struct et_ring_attn_stats_v1 {
  uint64_t total_block_steps;
  uint64_t executed_blocks;
  uint64_t skipped_causal_blocks;
  uint64_t bytes_sent;
  uint64_t bytes_received;
  uint32_t ring_rounds;
} et_ring_attn_stats_v1;

/*
 * Validate configuration and compute effective attention scale.
 * Returns ET_RING_ATTN_OK on success.
 */
et_ring_attn_status_t et_ring_attn_validate_config_v1(
    const void *cfg_ptr, double *effective_scale_out);

/*
 * Create an AF_UNIX ring communicator for rank `rank` of `world_size` in {1, 2, 4}.
 *   - For world_size == 1: send_fd must be -1 and recv_fd must be -1.
 *   - For world_size in {2, 4}: send_fd connects to (rank + 1) % W and recv_fd
 *     connects to (rank + W - 1) % W; both must be connected AF_UNIX SOCK_STREAM
 *     sockets. Descriptors are duplicated with FD_CLOEXEC on success.
 */
et_ring_attn_status_t et_ring_attn_comm_create_v1(
    uint32_t abi_major, uint32_t abi_minor, uint32_t rank, uint32_t world_size,
    int send_fd, int recv_fd, et_ring_attn_comm_t **out);

void et_ring_attn_comm_destroy_v1(et_ring_attn_comm_t *comm);

/*
 * Deadlock-free simultaneous non-blocking poll() ring exchange over AF_UNIX
 * sockets with a shared monotonic timeout deadline and framed header check.
 */
et_ring_attn_status_t et_ring_attn_ring_exchange_v1(
    et_ring_attn_comm_t *comm, const void *send_buf, void *recv_buf,
    size_t payload_bytes, uint32_t phase, uint32_t round,
    uint32_t send_kv_rank, uint32_t expected_recv_kv_rank, int32_t timeout_ms);

/*
 * Single-process block-step helpers for deterministic unit-level verification
 * of online softmax accumulation, causal block skipping, and analytic block
 * backward accumulation.
 *
 * Caller initializes:
 *   m_state[N * T_local * H] = -INFINITY
 *   l_state[N * T_local * H] = 0.0
 *   o_acc[N * T_local * H * Dh] = 0.0
 */
et_ring_attn_status_t et_ring_attn_fwd_block_step_v1(
    const void *cfg_ptr, uint32_t q_rank, uint32_t kv_rank,
    const void *q_shard, const void *k_block, const void *v_block,
    void *m_state, void *l_state, void *o_acc,
    et_ring_attn_stats_v1 *stats_accum);

et_ring_attn_status_t et_ring_attn_fwd_finalize_v1(
    const void *cfg_ptr, const void *m_state, const void *l_state,
    const void *o_acc, void *o_shard, void *lse_shard);

et_ring_attn_status_t et_ring_attn_bwd_block_step_v1(
    const void *cfg_ptr, uint32_t q_rank, uint32_t kv_rank,
    const void *q_shard, const void *do_shard, const void *lse_shard,
    const void *d_vec, const void *k_block, const void *v_block,
    void *dq_acc, void *dk_acc, void *dv_acc,
    et_ring_attn_stats_v1 *stats_accum);

/*
 * Per-rank distributed Ring Attention forward and backward over AF_UNIX ring.
 * All outputs remain byte-for-byte untouched on any error.
 */
et_ring_attn_status_t et_ring_attn_forward_v1(
    et_ring_attn_comm_t *comm, const void *cfg_ptr, const void *q_shard,
    const void *k_shard, const void *v_shard, void *o_shard, void *lse_shard,
    et_ring_attn_stats_v1 *stats_out);

et_ring_attn_status_t et_ring_attn_backward_v1(
    et_ring_attn_comm_t *comm, const void *cfg_ptr, const void *q_shard,
    const void *k_shard, const void *v_shard, const void *o_shard,
    const void *lse_shard, const void *do_shard, void *dq_shard, void *dk_shard,
    void *dv_shard, et_ring_attn_stats_v1 *stats_out);

/*
 * Multi-process fork + AF_UNIX socket ring execution across W in {1, 2, 4}
 * OS processes. No shared memory is used: each rank process holds only its
 * local shard and communicates K/V (forward) and K/V/dK/dV (backward) strictly
 * around the AF_UNIX socket ring.
 * All outputs remain byte-for-byte untouched on any error.
 */
et_ring_attn_status_t et_ring_attn_multiprocess_forward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, void *o_full, void *lse_full,
    et_ring_attn_stats_v1 *agg_stats_out);

et_ring_attn_status_t et_ring_attn_multiprocess_backward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, const void *o_full, const void *lse_full,
    const void *do_full, void *dq_full, void *dk_full, void *dv_full,
    et_ring_attn_stats_v1 *agg_stats_out);

/*
 * Exact single-rank dense binary64 attention reference on full [N, T, H, Dh].
 * Used to prove W = 2 and W = 4 multi-process Ring Attention match single-rank
 * full attention for O, LSE, dQ, dK, and dV.
 */
et_ring_attn_status_t et_ring_attn_dense_ref_forward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, void *o_full, void *lse_full);

et_ring_attn_status_t et_ring_attn_dense_ref_backward_v1(
    const void *cfg_ptr, const void *q_full, const void *k_full,
    const void *v_full, const void *o_full, const void *lse_full,
    const void *do_full, void *dq_full, void *dk_full, void *dv_full);

#ifdef __cplusplus
}
#endif

#endif
