#ifndef ESHKOL_TRANSFORMER_FSDP_ZERO_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_FSDP_ZERO_ESHKOL_BRIDGE_H

#include "eshkol_transformer/fsdp_zero_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Eshkol AOT per-rank session owning the AF_UNIX communicator,
 * ZeRO-1/2/3 (FSDP) engine, and reusable MLP training buffers. */
typedef struct et_fsdp_aot_session et_fsdp_aot_session_t;

/* Open a multi-process AF_UNIX FSDP session (W in {1, 2, 4}, stage in {1, 2, 3})
 * for a 2-layer tanh MLP (in_dim, hidden_dim, num_classes, global_batch). */
ET_FSDP_API int32_t et_fsdp_aot_open(
    const char *sock_dir, int64_t rank, int64_t world_size, int64_t stage,
    int64_t in_dim, int64_t hidden_dim, int64_t num_classes,
    int64_t global_batch, double lr, double beta1, double beta2, double eps,
    double weight_decay, double max_grad_norm, int64_t timeout_ms,
    et_fsdp_aot_session_t **out_session);

ET_FSDP_API void et_fsdp_aot_close(et_fsdp_aot_session_t *session);

/* Initialize sharded parameters from deterministic seed. */
ET_FSDP_API int32_t et_fsdp_aot_init_params(et_fsdp_aot_session_t *session,
                                            int64_t seed);

/* Run one complete synchronized FSDP/ZeRO MLP training step:
 *  1. AllGather ephemeral params (Stage 3) / copy params (Stage 1/2).
 *  2. Deterministic microbatch forward + backward on rank r's slice.
 *  3. Poison ephemeral full params buffer (Stage 3).
 *  4. AllReduceSum_f64 global mean loss across ranks.
 *  5. Synchronized ZeRO step (ReduceScatter/AllReduce + global L2 clip + AdamW).
 */
ET_FSDP_API int32_t et_fsdp_aot_train_step(et_fsdp_aot_session_t *session,
                                           int64_t step_idx, int64_t tag,
                                           int64_t timeout_ms);

/* Run a step with injected non-finite bit pattern (e.g. NaN or +Inf) on a
 * chosen target_rank at grad_index, proving atomic abort across all ranks. */
ET_FSDP_API int32_t et_fsdp_aot_step_inject_nonfinite(
    et_fsdp_aot_session_t *session, int64_t step_idx, int64_t target_rank,
    int64_t grad_index, int64_t bad_bits_u32, int64_t tag, int64_t timeout_ms);

/* Run a deliberate collective tag mismatch probe (uses tag on rank 0,
 * tag + alt_delta on rank > 0) to verify sticky ET_FSDP_ERR_MISMATCH. */
ET_FSDP_API int32_t et_fsdp_aot_probe_tag_mismatch(
    et_fsdp_aot_session_t *session, int64_t tag, int64_t alt_delta,
    int64_t timeout_ms);

/* Run central finite-difference gradient check (W=1) exhaustively over all
 * num_params parameters spanning W1, b1, W2, b2. Returns max absolute error via
 * et_fsdp_aot_last_fd_max_err(). */
ET_FSDP_API int32_t et_fsdp_aot_fd_grad_check(et_fsdp_aot_session_t *session,
                                              int64_t step_idx, double fd_eps);

/* Save/load versioned SHA-256 checksummed per-rank sharded checkpoint. */
ET_FSDP_API int32_t et_fsdp_aot_save_checkpoint(
    const et_fsdp_aot_session_t *session, const char *path);

ET_FSDP_API int32_t et_fsdp_aot_load_checkpoint(et_fsdp_aot_session_t *session,
                                                const char *path);

/* Compute full gathered parameters SHA-256 hex (64 chars + NUL) and optionally
 * dump full gathered binary32 weights to a file for cross-W parity checking. */
ET_FSDP_API int32_t et_fsdp_aot_gather_digest(et_fsdp_aot_session_t *session,
                                              int64_t tag, int64_t timeout_ms,
                                              const char *dump_path);

/* Compute SHA-256 hex over rank r's persistent shard state (params_shard +
 * m_shard + v_shard) into session's shard digest buffer (no collective). */
ET_FSDP_API int32_t et_fsdp_aot_hash_local_shard(
    et_fsdp_aot_session_t *session);

/* Compare two 64-byte hex digests stored in session: returns 1 if equal, 0 else. */
ET_FSDP_API int32_t et_fsdp_aot_save_shard_digest_mark(
    et_fsdp_aot_session_t *session);
ET_FSDP_API int32_t et_fsdp_aot_shard_digest_matches_mark(
    const et_fsdp_aot_session_t *session);

/* Read byte i (0..63) of the last full gathered params SHA-256 hex digest. */
ET_FSDP_API int32_t et_fsdp_aot_full_digest_byte(
    const et_fsdp_aot_session_t *session, int64_t index);

/* Negative test helper: tamper one byte or truncate a file at path, or create
 * a symlink at link_path pointing to target_path. */
ET_FSDP_API int32_t et_fsdp_aot_corrupt_file_byte(const char *path,
                                                  int64_t byte_offset);
ET_FSDP_API int32_t et_fsdp_aot_truncate_file(const char *path,
                                              int64_t new_size);
ET_FSDP_API int32_t et_fsdp_aot_make_symlink(const char *target_path,
                                             const char *link_path);

/* Scalar accessors for Eshkol AOT driver (zero heap allocations). */
ET_FSDP_API int64_t et_fsdp_aot_rank(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_world_size(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_stage(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_num_params(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_padded_params(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_shard_len(const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_step(const et_fsdp_aot_session_t *session);

ET_FSDP_API int64_t et_fsdp_aot_persistent_param_bytes(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_persistent_grad_bytes(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_persistent_opt_bytes(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_persistent_total_bytes(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_unsharded_baseline_bytes(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API double et_fsdp_aot_opt_reduction_ratio(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API double et_fsdp_aot_total_reduction_ratio(
    const et_fsdp_aot_session_t *session);

ET_FSDP_API double et_fsdp_aot_last_loss(const et_fsdp_aot_session_t *session);
ET_FSDP_API double et_fsdp_aot_last_grad_norm(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API double et_fsdp_aot_last_clip_scale(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_last_clipped(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API double et_fsdp_aot_last_fd_max_err(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_ephemeral_poison_verified(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_comm_bytes_sent(
    const et_fsdp_aot_session_t *session);
ET_FSDP_API int64_t et_fsdp_aot_comm_bytes_recv(
    const et_fsdp_aot_session_t *session);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_FSDP_ZERO_ESHKOL_BRIDGE_H */
