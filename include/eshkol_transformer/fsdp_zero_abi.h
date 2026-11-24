#ifndef ESHKOL_TRANSFORMER_FSDP_ZERO_ABI_H
#define ESHKOL_TRANSFORMER_FSDP_ZERO_ABI_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_FSDP_API __attribute__((visibility("default")))
#else
#define ET_FSDP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ET_FSDP_ABI_MAJOR 1u
#define ET_FSDP_ABI_MINOR 0u
#define ET_FSDP_MAX_WORLD_SIZE 4u
#define ET_FSDP_MAX_PARAMS UINT64_C(4194304)
#define ET_FSDP_MAX_BUFFER_BYTES (64u * 1024u * 1024u)

/* Versioned, SHA-256 checksummed, non-executable per-rank shard checkpoint. */
#define ET_FSDP_CKPT_MAGIC "ETFS1000"
#define ET_FSDP_CKPT_MAGIC_BYTES 8u
#define ET_FSDP_CKPT_VERSION_MAJOR 1u
#define ET_FSDP_CKPT_VERSION_MINOR 0u
#define ET_FSDP_CKPT_HEADER_BYTES 128u
#define ET_FSDP_CKPT_CHECKSUM_BYTES 32u

/* Quiet NaN bit pattern written to poisoned ephemeral FSDP parameter buffers. */
#define ET_FSDP_POISON_F32_BITS UINT32_C(0x7fc0dead)

typedef int32_t et_fsdp_status_t;
enum {
  ET_FSDP_OK = 0,
  ET_FSDP_ERR_INVALID_ARG = -1,
  ET_FSDP_ERR_MISMATCH = -2,
  ET_FSDP_ERR_PEER_DISCONNECTED = -3,
  ET_FSDP_ERR_TIMEOUT = -4,
  ET_FSDP_ERR_NONFINITE = -5,
  ET_FSDP_ERR_IO = -6,
  ET_FSDP_ERR_ALLOC = -7,
  ET_FSDP_ERR_BAD_MAGIC = -8,
  ET_FSDP_ERR_VERSION_MISMATCH = -9,
  ET_FSDP_ERR_CHECKSUM_MISMATCH = -10,
  ET_FSDP_ERR_TRUNCATED = -11,
  ET_FSDP_ERR_CONFIG_MISMATCH = -12,
  ET_FSDP_ERR_STATE = -13
};

/* Explicit ZeRO sharding stage identifiers. */
typedef uint32_t et_zero_stage_t;
enum {
  ET_ZERO_STAGE_1 = 1u,
  ET_ZERO_STAGE_2 = 2u,
  ET_ZERO_STAGE_3_FSDP = 3u
};

/* Standard-layout ZeRO/FSDP training configuration.
 * - stage: ET_ZERO_STAGE_1, ET_ZERO_STAGE_2, or ET_ZERO_STAGE_3_FSDP
 * - reserved: must be 0
 * - num_params: 1 .. ET_FSDP_MAX_PARAMS
 * - lr: finite, >= 0.0
 * - beta1, beta2: finite, in [0.0, 1.0)
 * - eps: finite, > 0.0
 * - weight_decay: finite, >= 0.0
 * - max_grad_norm: finite, >= 0.0 (0.0 disables global L2 clipping)
 */
typedef struct et_fsdp_config_v1 {
  uint32_t stage;
  uint32_t reserved;
  uint64_t num_params;
  double lr;
  double beta1;
  double beta2;
  double eps;
  double weight_decay;
  double max_grad_norm;
} et_fsdp_config_v1;

/* Exact persistent memory and sharding reduction accounting for one rank. */
typedef struct et_fsdp_memory_report_v1 {
  uint32_t stage;
  uint32_t rank;
  uint32_t world_size;
  uint32_t reserved;
  uint64_t num_params;
  uint64_t padded_params;
  uint64_t shard_len;
  uint64_t persistent_param_bytes;
  uint64_t persistent_grad_bytes;
  uint64_t persistent_opt_bytes;
  uint64_t persistent_total_bytes;
  uint64_t unsharded_baseline_bytes;
  uint64_t ephemeral_peak_bytes;
  double optimizer_reduction_ratio;
  double total_persistent_reduction_ratio;
} et_fsdp_memory_report_v1;

/* Cumulative communicator payload accounting for one rank. */
typedef struct et_fsdp_comm_stats_v1 {
  uint64_t allgather_calls;
  uint64_t reduce_scatter_calls;
  uint64_t allreduce_f32_calls;
  uint64_t allreduce_f64_calls;
  uint64_t payload_bytes_sent;
  uint64_t payload_bytes_recv;
} et_fsdp_comm_stats_v1;

/* Per-step telemetry emitted by et_fsdp_engine_step. */
typedef struct et_fsdp_step_info_v1 {
  uint64_t step;
  double global_grad_norm;
  double clip_scale;
  uint32_t clipped;
  uint32_t reserved;
} et_fsdp_step_info_v1;

/* Opaque AF_UNIX multi-process collective communicator (W in {1, 2, 4}). */
typedef struct et_fsdp_comm et_fsdp_comm_t;

/* Opaque per-rank ZeRO-1/2/3 (FSDP) sharded training engine. */
typedef struct et_fsdp_engine et_fsdp_engine_t;

/* --- Communicator lifecycle & real AF_UNIX collectives --- */

/*
 * Create a communicator from pre-connected AF_UNIX SOCK_STREAM endpoints.
 * Supported world sizes: W in {1, 2, 4}.
 * Rank 0 supplies W - 1 descriptors in ascending peer rank order (1..W-1).
 * Other ranks supply 1 descriptor connected to rank 0.
 * W == 1 supplies 0 descriptors.
 * Descriptors are duplicated with FD_CLOEXEC; caller closes originals.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_comm_create_fds(
    uint32_t abi_major, uint32_t abi_minor, uint32_t rank, uint32_t world_size,
    const int *fds, size_t fd_count, et_fsdp_comm_t **out_comm);

/*
 * Bootstrap independent OS processes over an AF_UNIX SOCK_STREAM rendezvous
 * socket at "<sock_dir>/fsdp_r0.sock" with a shared monotonic timeout.
 * Supported world sizes: W in {1, 2, 4}. W == 1 opens zero sockets.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_comm_open_unix(
    uint32_t abi_major, uint32_t abi_minor, const char *sock_dir, uint32_t rank,
    uint32_t world_size, int timeout_ms, et_fsdp_comm_t **out_comm);

ET_FSDP_API void et_fsdp_comm_destroy(et_fsdp_comm_t *comm);
ET_FSDP_API uint32_t et_fsdp_comm_rank(const et_fsdp_comm_t *comm);
ET_FSDP_API uint32_t et_fsdp_comm_world_size(const et_fsdp_comm_t *comm);
ET_FSDP_API et_fsdp_status_t et_fsdp_comm_get_stats(
    const et_fsdp_comm_t *comm, et_fsdp_comm_stats_v1 *out_stats);

/*
 * AllGather binary32 shards: each rank supplies send_shard[shard_len] and
 * receives recv_full[W * shard_len] concatenated in canonical ascending rank
 * order 0..W-1. Input and output must be disjoint.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_allgather_f32(
    et_fsdp_comm_t *comm, const float *send_shard, size_t shard_len,
    float *recv_full, uint64_t tag, int timeout_ms);

/*
 * ReduceScatter binary32 mean: each rank supplies send_full[W * shard_len].
 * All inputs are validated finite in preflight (any non-finite element on any
 * rank triggers coordinated ET_FSDP_ERR_NONFINITE leaving all outputs
 * untouched). Rank 0 sums in canonical ascending rank order 0..W-1 in binary64,
 * divides by W, converts to binary32, and scatters ONLY slice
 * [rank * shard_len, (rank + 1) * shard_len) into recv_shard[shard_len].
 * Input and output must be disjoint.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_reduce_scatter_avg_f32(
    et_fsdp_comm_t *comm, const float *send_full, float *recv_shard,
    size_t shard_len, uint64_t tag, int timeout_ms);

/*
 * AllReduce binary32 mean: canonical ascending rank order binary64 sum divided
 * by W, delivered to all ranks. Used by ZeRO-1 full-gradient synchronization.
 * Coordinated non-finite preflight; supports exact in-place or disjoint buffers.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_allreduce_avg_f32(
    et_fsdp_comm_t *comm, const float *send_full, float *recv_full,
    size_t count, uint64_t tag, int timeout_ms);

/*
 * AllReduce binary64 sum: canonical ascending rank order binary64 sum across
 * all ranks, used for exact global L2 gradient norm clipping across sharded
 * gradients. Coordinated non-finite preflight; supports exact in-place or
 * disjoint buffers.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_allreduce_sum_f64(
    et_fsdp_comm_t *comm, const double *send_vals, double *recv_vals,
    size_t count, uint64_t tag, int timeout_ms);

/* --- Deterministic padding and ZeRO-1/2/3 (FSDP) sharded engine --- */

ET_FSDP_API et_fsdp_status_t et_fsdp_compute_padded_len(
    uint64_t num_params, uint32_t world_size, uint64_t *out_padded_len,
    uint64_t *out_shard_len);

/*
 * Create a per-rank sharded engine borrowing comm (caller keeps comm alive
 * until after et_fsdp_engine_destroy).
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_create(
    et_fsdp_comm_t *comm, const et_fsdp_config_v1 *cfg,
    et_fsdp_engine_t **out_engine);

ET_FSDP_API void et_fsdp_engine_destroy(et_fsdp_engine_t *engine);

/*
 * Initialize sharded parameters from a full finite parameter vector
 * full_params[num_params]. Stage 3 retains ONLY rank r's owned shard
 * [r * shard_len .. (r + 1) * shard_len); trailing pad slots are strictly
 * zeroed. Resets m_shard, v_shard, grad_shard, and step to 0. Atomic on error.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_load_initial_params(
    et_fsdp_engine_t *engine, const float *full_params, size_t num_params);

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_memory_report(
    const et_fsdp_engine_t *engine, et_fsdp_memory_report_v1 *out_report);

ET_FSDP_API uint64_t et_fsdp_engine_get_step(const et_fsdp_engine_t *engine);

/*
 * Materialize full parameters for forward+backward into out_full_params[out_count],
 * where out_count must be num_params or padded_params.
 * - Stage 1 & 2: copies persistent replicated params_full (no collective).
 * - Stage 3 FSDP: runs AllGather(params_shard) into an ephemeral buffer, copies
 *   out_count floats, then poisons and frees its internal ephemeral buffer.
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_gather_params(
    et_fsdp_engine_t *engine, float *out_full_params, size_t out_count,
    uint64_t tag, int timeout_ms);

/* Poison caller's ephemeral full parameter buffer with ET_FSDP_POISON_F32_BITS. */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_poison_ephemeral(float *buf,
                                                             size_t count);

/*
 * Execute one synchronized ZeRO-1 / ZeRO-2 / ZeRO-3 (FSDP) optimization step:
 *  1. Zero-pad raw_full_grads[num_params] to padded_params (pad slots == 0.0f).
 *  2. Stage 1: AllReduceAvg(grads_full); Stage 2 & 3: ReduceScatterAvg(grads -> grad_shard).
 *     Any non-finite gradient on any rank triggers atomic ET_FSDP_ERR_NONFINITE
 *     across ALL ranks leaving params/m/v/step bitwise untouched.
 *  3. Exact global L2 norm via AllReduceSum_f64 of shard squared norms.
 *  4. Shard-local decoupled AdamW update on owned slice [r*S .. (r+1)*S),
 *     strictly preserving trailing pad slots at 0.0f.
 *  5. Stage 1 & 2: AllGather(params_shard) to update persistent params_full.
 *     Stage 3 FSDP: keeps ONLY updated params_shard[shard_len] persistent!
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_step(
    et_fsdp_engine_t *engine, const float *raw_full_grads, size_t num_params,
    uint64_t tag, int timeout_ms, et_fsdp_step_info_v1 *out_info);

/* Inspect per-rank persistent shards (out_len must equal shard_len). */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_param_shard(
    const et_fsdp_engine_t *engine, float *out_shard, size_t shard_len);

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_moments_shard(
    const et_fsdp_engine_t *engine, float *out_m, float *out_v,
    size_t shard_len);

ET_FSDP_API et_fsdp_status_t et_fsdp_engine_get_grad_shard(
    const et_fsdp_engine_t *engine, float *out_grad, size_t shard_len);

/* Gather full unpadded parameters [0..num_params) and hash raw f32 bytes. */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_gather_full_params_sha256(
    et_fsdp_engine_t *engine, uint64_t tag, int timeout_ms,
    uint8_t out_digest[32]);

/* --- Versioned, SHA-256 checksummed per-rank sharded checkpoint I/O --- */

/* Atomic 0600 fsync+rename save; refuses symlink or non-regular destination. */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_save_checkpoint(
    const et_fsdp_engine_t *engine, const char *path);

/* O_NOFOLLOW checked load rejecting symlinks, non-regular files, bad magic,
 * version mismatch, truncation/trailing bytes, bad SHA-256, config/stage/rank/
 * world mismatch, non-finite values, negative v, or nonzero pad slots. Atomic:
 * live state is untouched on any error. */
ET_FSDP_API et_fsdp_status_t et_fsdp_engine_load_checkpoint(
    et_fsdp_engine_t *engine, const char *path);

/* --- Deterministic 2-layer tanh MLP + cross-entropy workload --- */

/*
 * Parameter layout:
 *   W1 [in_dim, hidden_dim]  offset 0
 *   b1 [hidden_dim]          offset in_dim * hidden_dim
 *   W2 [hidden_dim, classes] offset in_dim * hidden_dim + hidden_dim
 *   b2 [classes]             offset in_dim * hidden_dim + hidden_dim + hidden_dim * classes
 * Input X: [batch_size, in_dim] (binary32), targets: [batch_size] (int32 in 0..classes-1).
 */
ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_num_params(size_t in_dim,
                                                    size_t hidden_dim,
                                                    size_t num_classes,
                                                    uint64_t *out_num_params);

ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_init_params(
    size_t in_dim, size_t hidden_dim, size_t num_classes, uint64_t seed,
    float *out_params, size_t num_params);

/* Generate global_batch deterministic samples and emit rank r's contiguous
 * microbatch of local_batch = global_batch / world_size samples. */
ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_make_batch(
    size_t in_dim, size_t num_classes, size_t global_batch, uint32_t rank,
    uint32_t world_size, uint64_t step_idx, float *out_x, int32_t *out_targets);

/* Exact forward + analytical backward (binary64 internal accumulation).
 * Pass out_grads == NULL for forward-only loss evaluation. */
ET_FSDP_API et_fsdp_status_t et_fsdp_mlp_forward_backward(
    size_t in_dim, size_t hidden_dim, size_t num_classes, size_t batch_size,
    const float *params, const float *x, const int32_t *targets,
    double *out_loss, float *out_grads);

/* Standalone SHA-256 over arbitrary byte spans. */
ET_FSDP_API et_fsdp_status_t et_fsdp_sha256_bytes(const uint8_t *data,
                                                  size_t len,
                                                  uint8_t out_digest[32]);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_FSDP_ZERO_ABI_H */
