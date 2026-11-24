#ifndef ESHKOL_TRANSFORMER_HYBRID_MESH_ABI_H
#define ESHKOL_TRANSFORMER_HYBRID_MESH_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_MESH_API __attribute__((visibility("default")))
#else
#define ET_MESH_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MESH_ABI_MAJOR 1u
#define ET_MESH_ABI_MINOR 0u
#define ET_MESH_MAX_WORLD_SIZE 4u
#define ET_MESH_MAX_BUFFER_BYTES (64u * 1024u * 1024u)

/* Versioned, SHA-256 checksummed, non-executable per-shard checkpoint. */
#define ET_MESH_CKPT_MAGIC "ET2D1000"
#define ET_MESH_CKPT_MAGIC_BYTES 8u
#define ET_MESH_CKPT_VERSION_MAJOR 1u
#define ET_MESH_CKPT_VERSION_MINOR 0u
#define ET_MESH_CKPT_HEADER_BYTES 160u
#define ET_MESH_CKPT_CHECKSUM_BYTES 32u

typedef int32_t et_mesh_status_t;
enum {
  ET_MESH_OK = 0,
  ET_MESH_ERR_INVALID_ARG = -1,
  ET_MESH_ERR_DIVISIBILITY = -2,
  ET_MESH_ERR_ALIAS = -3,
  ET_MESH_ERR_MISMATCH = -4,
  ET_MESH_ERR_PEER_DISCONNECTED = -5,
  ET_MESH_ERR_TIMEOUT = -6,
  ET_MESH_ERR_NONFINITE = -7,
  ET_MESH_ERR_IO = -8,
  ET_MESH_ERR_ALLOC = -9,
  ET_MESH_ERR_BAD_MAGIC = -10,
  ET_MESH_ERR_VERSION_MISMATCH = -11,
  ET_MESH_ERR_CHECKSUM_MISMATCH = -12,
  ET_MESH_ERR_TRUNCATED = -13,
  ET_MESH_ERR_MESH_MISMATCH = -14,
  ET_MESH_ERR_CONFIG_MISMATCH = -15,
  ET_MESH_ERR_STATE = -16
};

/*
 * Deterministic 2D device mesh coordinates:
 *   world_size = dp_size * tp_size <= 4, with dp_size, tp_size in {1, 2, 4}.
 *   dp_rank = global_rank / tp_size
 *   tp_rank = global_rank % tp_size
 */
typedef struct et_mesh_coords_v1 {
  uint32_t global_rank;
  uint32_t world_size;
  uint32_t dp_size;
  uint32_t tp_size;
  uint32_t dp_rank;
  uint32_t tp_rank;
} et_mesh_coords_v1;

/* Separate orthogonal TP and DP subgroup collective counters. */
typedef struct et_mesh_comm_stats_v1 {
  uint64_t tp_allreduce_calls;
  uint64_t dp_allreduce_calls;
  uint64_t tp_allgather_calls;
  uint64_t tp_consensus_calls;
  uint64_t dp_consensus_calls;
  uint64_t tp_bytes_sent;
  uint64_t tp_bytes_recv;
  uint64_t dp_bytes_sent;
  uint64_t dp_bytes_recv;
} et_mesh_comm_stats_v1;

/*
 * 2D Hybrid-Parallel SwiGLU FFN + Linear Head model configuration:
 *   RMSNorm(gamma) -> ColumnParallel(W_gate_r, W_up_r) -> SiLU(gate)*up ->
 *   RowParallel(W_down_r) + TP AllReduceSum + b_down (once) + residual ->
 *   LinearHead(W_head, b_head) -> Cross-Entropy loss.
 * Constraints:
 *   d_model > 0, d_ff > 0, num_classes > 0, global_batch > 0,
 *   d_ff % tp_size == 0, global_batch % dp_size == 0,
 *   rms_eps finite and > 0.0f, reserved == 0.
 */
typedef struct et_mesh_model_config_v1 {
  uint32_t d_model;
  uint32_t d_ff;
  uint32_t num_classes;
  uint32_t global_batch;
  float rms_eps;
  uint32_t reserved;
} et_mesh_model_config_v1;

/* Decoupled AdamW + global L2 gradient clipping configuration. */
typedef struct et_mesh_optim_config_v1 {
  double lr;
  double beta1;
  double beta2;
  double eps;
  double weight_decay;
  double max_grad_norm;
} et_mesh_optim_config_v1;

/* Exact per-rank sharded & replicated weight/optimizer memory report. */
typedef struct et_mesh_memory_report_v1 {
  uint32_t dp_size;
  uint32_t tp_size;
  uint32_t dp_rank;
  uint32_t tp_rank;
  uint64_t full_sharded_params;
  uint64_t rank_sharded_params;
  uint64_t replicated_params;
  uint64_t full_total_params;
  uint64_t rank_total_params;
  uint64_t full_sharded_weight_bytes;
  uint64_t rank_sharded_weight_bytes;
  uint64_t replicated_weight_bytes;
  uint64_t rank_total_weight_bytes;
  uint64_t rank_optimizer_state_bytes;
  double tp_sharding_ratio;
} et_mesh_memory_report_v1;

/* Per-step telemetry emitted by et_mesh_engine_train_step. */
typedef struct et_mesh_step_info_v1 {
  uint64_t step;
  double local_loss;
  double global_loss;
  double global_grad_norm;
  double clip_scale;
  uint32_t clipped;
  uint32_t reserved;
} et_mesh_step_info_v1;

/* Opaque 2D device mesh owning orthogonal AF_UNIX tp_group and dp_group. */
typedef struct et_mesh et_mesh_t;

/* Opaque 2D hybrid-parallel SwiGLU FFN + Linear Head training engine. */
typedef struct et_mesh_engine et_mesh_engine_t;

/* --- 1. 2D Device Mesh Topology & Orthogonal AF_UNIX Rendezvous --- */

ET_MESH_API et_mesh_status_t et_mesh_validate_topology(
    uint32_t dp_size, uint32_t tp_size, uint32_t global_rank,
    et_mesh_coords_v1 *out_coords);

/*
 * Bootstrap independent OS processes over AF_UNIX SOCK_STREAM rendezvous
 * sockets inside sock_dir with a shared CLOCK_MONOTONIC deadline:
 *   - tp_group: "<sock_dir>/mesh_tp_dp<dp_rank>.sock" (shared dp_rank)
 *   - dp_group: "<sock_dir>/mesh_dp_tp<tp_rank>.sock" (shared tp_rank)
 * Supported topologies: (1,1), (2,1), (1,2), (2,2), (4,1), (1,4).
 * Subgroup size 1 opens zero sockets. Zero FD leaks on any error.
 */
ET_MESH_API et_mesh_status_t et_mesh_open_unix(
    uint32_t abi_major, uint32_t abi_minor, const char *sock_dir,
    uint32_t global_rank, uint32_t dp_size, uint32_t tp_size, int timeout_ms,
    et_mesh_t **out_mesh);

/*
 * Create a 2D mesh from pre-connected AF_UNIX SOCK_STREAM descriptors.
 * For each subgroup (tp_group of size S_tp, dp_group of size S_dp):
 *   - S == 1: fd_count == 0, fds may be NULL.
 *   - S > 1, sub_rank == 0: fd_count == S - 1, fds[i] connected to sub_rank i+1.
 *   - S > 1, sub_rank > 0:  fd_count == 1, fds[0] connected to sub_rank 0.
 * Descriptors are duplicated with F_DUPFD_CLOEXEC; caller closes originals.
 */
ET_MESH_API et_mesh_status_t et_mesh_create_fds(
    uint32_t abi_major, uint32_t abi_minor, uint32_t global_rank,
    uint32_t dp_size, uint32_t tp_size, const int *tp_fds, size_t tp_fd_count,
    const int *dp_fds, size_t dp_fd_count, et_mesh_t **out_mesh);

ET_MESH_API void et_mesh_destroy(et_mesh_t *mesh);

ET_MESH_API et_mesh_status_t et_mesh_get_coords(const et_mesh_t *mesh,
                                                et_mesh_coords_v1 *out_coords);

ET_MESH_API et_mesh_status_t et_mesh_get_stats(
    const et_mesh_t *mesh, et_mesh_comm_stats_v1 *out_stats);

ET_MESH_API et_mesh_status_t et_mesh_reset_stats(et_mesh_t *mesh);

/* --- 2. Orthogonal Subgroup Collectives (Canonical Ascending binary64) --- */

/*
 * TP AllReduceSum (binary32 via canonical ascending tp_rank binary64 sum).
 * Supports exact in-place (sendbuf == recvbuf); partial overlap fails with
 * ET_MESH_ERR_ALIAS. Non-finite inputs trigger recoverable subgroup-wide
 * ET_MESH_ERR_NONFINITE without modifying recvbuf.
 */
ET_MESH_API et_mesh_status_t et_mesh_tp_allreduce_sum_f32(
    et_mesh_t *mesh, const float *sendbuf, float *recvbuf, size_t count,
    uint64_t tag, int timeout_ms);

/* TP AllReduceSum (binary64, canonical ascending tp_rank order). */
ET_MESH_API et_mesh_status_t et_mesh_tp_allreduce_sum_f64(
    et_mesh_t *mesh, const double *sendbuf, double *recvbuf, size_t count,
    uint64_t tag, int timeout_ms);

/* TP AllGather (binary32, canonical ascending tp_rank order). Disjoint only. */
ET_MESH_API et_mesh_status_t et_mesh_tp_allgather_f32(
    et_mesh_t *mesh, const float *sendbuf, size_t send_count, float *recvbuf,
    uint64_t tag, int timeout_ms);

/*
 * DP AllReduceAvg (binary32 via canonical ascending dp_rank binary64 sum / dp_size).
 * Matching tp_rank replicas receive bitwise-identical averaged outputs.
 */
ET_MESH_API et_mesh_status_t et_mesh_dp_allreduce_avg_f32(
    et_mesh_t *mesh, const float *sendbuf, float *recvbuf, size_t count,
    uint64_t tag, int timeout_ms);

/* DP AllReduceAvg (binary64 via canonical ascending dp_rank sum / dp_size). */
ET_MESH_API et_mesh_status_t et_mesh_dp_allreduce_avg_f64(
    et_mesh_t *mesh, const double *sendbuf, double *recvbuf, size_t count,
    uint64_t tag, int timeout_ms);

/*
 * Full 2D mesh-wide bitwise-OR consensus across BOTH tp_group then dp_group.
 * Guarantees every rank in the dp_size x tp_size mesh observes any flag raised
 * by any single rank (used for zero-split-brain atomic non-finite aborts).
 */
ET_MESH_API et_mesh_status_t et_mesh_consensus_or_u64(
    et_mesh_t *mesh, uint64_t local_flags, uint64_t *out_global_flags,
    uint64_t tag, int timeout_ms);

/* --- 3. Model Config, Parameter Layout & Deterministic Data --- */

ET_MESH_API et_mesh_status_t et_mesh_validate_configs(
    const et_mesh_model_config_v1 *mcfg, const et_mesh_optim_config_v1 *ocfg,
    uint32_t dp_size, uint32_t tp_size);

/*
 * Exact full and per-rank parameter counts.
 * Full dense layout:
 *   [gamma(d_model), b_down(d_model), W_head(num_classes*d_model), b_head(num_classes),
 *    W_gate(d_ff*d_model), W_up(d_ff*d_model), W_down(d_model*d_ff)]
 * Per-rank local layout (d_ff_local = d_ff / tp_size):
 *   [gamma(d_model), b_down(d_model), W_head(num_classes*d_model), b_head(num_classes),
 *    W_gate_r(d_ff_local*d_model), W_up_r(d_ff_local*d_model), W_down_r(d_model*d_ff_local)]
 */
ET_MESH_API et_mesh_status_t et_mesh_param_counts(
    const et_mesh_model_config_v1 *mcfg, uint32_t tp_size,
    uint64_t *out_full_params, uint64_t *out_rank_params,
    uint64_t *out_replicated_params, uint64_t *out_rank_sharded_params);

ET_MESH_API et_mesh_status_t et_mesh_init_full_params(
    const et_mesh_model_config_v1 *mcfg, uint64_t seed, float *out_full_params,
    size_t full_param_count);

/* Generate deterministic microbatch slice [local_batch, d_model] + [local_batch]
 * for dp_rank (independent of tp_rank, matching full batch partitioning). */
ET_MESH_API et_mesh_status_t et_mesh_make_microbatch(
    const et_mesh_model_config_v1 *mcfg, uint32_t dp_rank, uint32_t dp_size,
    uint64_t step_idx, float *out_x, int32_t *out_targets);

/* --- 4. 2D Hybrid-Parallel Training Engine --- */

/* Create engine borrowing mesh (caller keeps mesh alive until engine_destroy). */
ET_MESH_API et_mesh_status_t et_mesh_engine_create(
    et_mesh_t *mesh, const et_mesh_model_config_v1 *mcfg,
    const et_mesh_optim_config_v1 *ocfg, et_mesh_engine_t **out_engine);

ET_MESH_API void et_mesh_engine_destroy(et_mesh_engine_t *engine);

/* Shard full dense parameters onto this rank's tp_rank slice; resets m, v, step=0. */
ET_MESH_API et_mesh_status_t et_mesh_engine_load_full_params(
    et_mesh_engine_t *engine, const float *full_params,
    size_t full_param_count);

ET_MESH_API et_mesh_status_t et_mesh_engine_get_memory_report(
    const et_mesh_engine_t *engine, et_mesh_memory_report_v1 *out_report);

ET_MESH_API uint64_t et_mesh_engine_get_step(const et_mesh_engine_t *engine);

/*
 * Forward pass on this rank's local microbatch:
 *   - 1 TP AllReduceSum across tp_group (when tp_size > 1)
 *   - ZERO DP communication!
 * Caches activations for et_mesh_engine_backward.
 */
ET_MESH_API et_mesh_status_t et_mesh_engine_forward(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint64_t tag_base, int timeout_ms,
    double *out_local_loss);

/*
 * Analytic local backward pass using activations cached by et_mesh_engine_forward:
 *   - 1 TP AllReduceSum across tp_group for dU (when tp_size > 1)
 *   - ZERO DP communication (call et_mesh_engine_train_step for full DP sync + AdamW).
 * out_dx_local [local_batch * d_model] may be NULL if input gradient is not needed.
 */
ET_MESH_API et_mesh_status_t et_mesh_engine_backward(
    et_mesh_engine_t *engine, uint64_t tag_base, int timeout_ms,
    float *out_dx_local);

/*
 * Full synchronized 2D hybrid training step:
 *  1. Forward on local microbatch (1 TP AllReduceSum, 0 DP comm)
 *  2. Analytic backward (1 TP AllReduceSum for dU)
 *  3. 2D mesh-wide non-finite consensus (tp_group then dp_group); any non-finite
 *     on any single rank aborts ALL dp_size*tp_size ranks atomically with
 *     ET_MESH_ERR_NONFINITE, leaving weights, m, v, and step bitwise untouched.
 *  4. DP AllReduceAvg across dp_group on all local parameter gradients (matching
 *     tp_rank replicas receive bitwise-identical gradients) + DP AllReduceAvg_f64
 *     for exact global mean loss.
 *  5. Global L2 norm clipping across the 2D mesh (TP AllReduceSum_f64 of sharded
 *     squared norms + replicated squared norm, verified finite across 2D mesh).
 *  6. Decoupled AdamW update in binary64, advancing step += 1.
 */
ET_MESH_API et_mesh_status_t et_mesh_engine_train_step(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint64_t tag_base, int timeout_ms,
    et_mesh_step_info_v1 *out_info);

/*
 * Diagnostic fault-injection variant of et_mesh_engine_train_step:
 * injects bad_f32_bits into local gradient slot grad_index ONLY on
 * global_rank == target_global_rank after local backward, verifying atomic
 * 2D mesh-wide abort across all dp_size * tp_size ranks.
 */
ET_MESH_API et_mesh_status_t et_mesh_engine_train_step_inject_nonfinite(
    et_mesh_engine_t *engine, const float *x_local,
    const int32_t *targets_local, uint32_t target_global_rank,
    size_t grad_index, uint32_t bad_f32_bits, uint64_t tag_base, int timeout_ms,
    et_mesh_step_info_v1 *out_info);

/* Gather full dense weights across tp_group via TP AllGather. */
ET_MESH_API et_mesh_status_t et_mesh_engine_gather_full_params(
    et_mesh_engine_t *engine, float *out_full_params, size_t full_param_count,
    uint64_t tag, int timeout_ms);

/* Inspect per-rank local parameter, gradient, and AdamW moment shards. */
ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_params(
    const et_mesh_engine_t *engine, float *out_params,
    size_t local_param_count);

ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_grads(
    const et_mesh_engine_t *engine, float *out_grads, size_t local_param_count);

ET_MESH_API et_mesh_status_t et_mesh_engine_get_local_moments(
    const et_mesh_engine_t *engine, float *out_m, float *out_v,
    size_t local_param_count);

/* SHA-256 over this rank's persistent (step, local_params, m, v) bytes. */
ET_MESH_API et_mesh_status_t et_mesh_engine_local_state_sha256(
    const et_mesh_engine_t *engine, uint8_t out_digest[32]);

/* Gather full dense weights across tp_group and SHA-256 raw f32 bytes. */
ET_MESH_API et_mesh_status_t et_mesh_engine_gather_full_params_sha256(
    et_mesh_engine_t *engine, uint64_t tag, int timeout_ms,
    uint8_t out_digest[32]);

/* --- 5. Versioned SHA-256 Checksummed Per-Shard Checkpoint (ET2D1000 v1.0) --- */

/* Atomic 0600 temp+fsync+rename save; refuses symlink or non-regular path. */
ET_MESH_API et_mesh_status_t et_mesh_engine_save_checkpoint(
    const et_mesh_engine_t *engine, const char *path);

/* O_NOFOLLOW checked load rejecting symlinks, non-regular files, bad magic,
 * version mismatch, truncation/trailing bytes, SHA-256 mismatch, mesh coord
 * mismatch, config mismatch, non-finite values, or negative v. Atomic on error. */
ET_MESH_API et_mesh_status_t et_mesh_engine_load_checkpoint(
    et_mesh_engine_t *engine, const char *path);

/* Standalone FIPS 180-4 SHA-256. */
ET_MESH_API et_mesh_status_t et_mesh_sha256_bytes(const uint8_t *data,
                                                  size_t len,
                                                  uint8_t out_digest[32]);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_HYBRID_MESH_ABI_H */
