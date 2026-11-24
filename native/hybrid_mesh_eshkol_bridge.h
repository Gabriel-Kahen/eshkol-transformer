#ifndef ESHKOL_TRANSFORMER_HYBRID_MESH_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_HYBRID_MESH_ESHKOL_BRIDGE_H

#include "eshkol_transformer/hybrid_mesh_abi.h"

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_MESH_AOT_API __attribute__((visibility("default")))
#else
#define ET_MESH_AOT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque Eshkol AOT session wrapping a 2D mesh + hybrid-parallel engine. */
typedef struct et_mesh_aot_session et_mesh_aot_session_t;

ET_MESH_AOT_API int32_t et_mesh_aot_open(
    const char *sock_dir, int64_t global_rank, int64_t dp_size, int64_t tp_size,
    int64_t d_model, int64_t d_ff, int64_t num_classes, int64_t global_batch,
    double rms_eps, double lr, double beta1, double beta2, double eps,
    double weight_decay, double max_grad_norm, int64_t timeout_ms,
    et_mesh_aot_session_t **out_sess);

ET_MESH_AOT_API void et_mesh_aot_close(et_mesh_aot_session_t *sess);

ET_MESH_AOT_API int32_t et_mesh_aot_init_params(et_mesh_aot_session_t *sess,
                                                int64_t seed);

ET_MESH_AOT_API int32_t et_mesh_aot_train_step(et_mesh_aot_session_t *sess,
                                               int64_t step_idx,
                                               int64_t tag_base,
                                               int64_t timeout_ms);

ET_MESH_AOT_API int32_t et_mesh_aot_step_inject_nonfinite(
    et_mesh_aot_session_t *sess, int64_t step_idx, int64_t target_global_rank,
    int64_t grad_index, int64_t bad_f32_bits, int64_t tag_base,
    int64_t timeout_ms);

ET_MESH_AOT_API int32_t et_mesh_aot_fd_grad_check(et_mesh_aot_session_t *sess,
                                                  int64_t step_idx,
                                                  double fd_eps,
                                                  int64_t timeout_ms);

ET_MESH_AOT_API int32_t et_mesh_aot_probe_tag_mismatch(
    et_mesh_aot_session_t *sess, int64_t base_tag, int64_t diverge_on_rank,
    int64_t timeout_ms);

ET_MESH_AOT_API int32_t et_mesh_aot_save_checkpoint(
    et_mesh_aot_session_t *sess, const char *path);

ET_MESH_AOT_API int32_t et_mesh_aot_load_checkpoint(
    et_mesh_aot_session_t *sess, const char *path);

ET_MESH_AOT_API int32_t et_mesh_aot_gather_digest(et_mesh_aot_session_t *sess,
                                                  int64_t tag,
                                                  int64_t timeout_ms,
                                                  const char *dump_weights_path);

ET_MESH_AOT_API int32_t et_mesh_aot_hash_local_shard(
    et_mesh_aot_session_t *sess);

ET_MESH_AOT_API int32_t et_mesh_aot_save_shard_mark(
    et_mesh_aot_session_t *sess);

ET_MESH_AOT_API int32_t et_mesh_aot_shard_matches_mark(
    const et_mesh_aot_session_t *sess);

/* ASCII lowercase hex byte [0..63] of last gathered full weights SHA-256. */
ET_MESH_AOT_API int32_t et_mesh_aot_full_digest_byte(
    const et_mesh_aot_session_t *sess, int64_t index);

/* ASCII lowercase hex byte [0..63] of last local shard state SHA-256. */
ET_MESH_AOT_API int32_t et_mesh_aot_local_digest_byte(
    const et_mesh_aot_session_t *sess, int64_t index);

/* File fault-injection helpers for negative checkpoint tests. */
ET_MESH_AOT_API int32_t et_mesh_aot_corrupt_file_byte(const char *path,
                                                      int64_t offset);
ET_MESH_AOT_API int32_t et_mesh_aot_truncate_file(const char *path,
                                                  int64_t new_size);
ET_MESH_AOT_API int32_t et_mesh_aot_make_symlink(const char *target,
                                                 const char *link_path);

/* Telemetry & accounting getters. */
ET_MESH_AOT_API int64_t et_mesh_aot_global_rank(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_dp_size(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_tp_size(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_dp_rank(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_tp_rank(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_step(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_full_params(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_rank_params(const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_full_sharded_bytes(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_rank_sharded_bytes(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_replicated_bytes(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API double et_mesh_aot_tp_sharding_ratio(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API double et_mesh_aot_last_local_loss(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API double et_mesh_aot_last_global_loss(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API double et_mesh_aot_last_grad_norm(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_last_clipped(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API double et_mesh_aot_last_fd_max_err(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_tp_allreduce_calls(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_dp_allreduce_calls(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_tp_allgather_calls(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_tp_bytes_sent(
    const et_mesh_aot_session_t *s);
ET_MESH_AOT_API int64_t et_mesh_aot_dp_bytes_sent(
    const et_mesh_aot_session_t *s);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_HYBRID_MESH_ESHKOL_BRIDGE_H */
