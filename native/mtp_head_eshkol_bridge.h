#ifndef ESHKOL_TRANSFORMER_MTP_HEAD_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_MTP_HEAD_ESHKOL_BRIDGE_H

#include "eshkol_transformer/mtp_head_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque Eshkol AOT session owning an et_mtp_model_t, deterministic sequence
 * fixtures (h0[B, T, H], tokens[B, T]), reusable backward buffers, and last
 * step/FD/speculative-verification telemetry.
 */
typedef struct et_mtp_aot_session et_mtp_aot_session_t;

int32_t et_mtp_aot_open(int64_t vocab_size, int64_t hidden_dim, int64_t ffn_dim,
                        int64_t mtp_depth, int64_t tie_weights, int64_t batch,
                        int64_t seq_len, double rms_eps, double lambda_mtp,
                        double lr, double beta1, double beta2, double eps,
                        double weight_decay, double max_grad_norm,
                        et_mtp_aot_session_t **out_session);

void et_mtp_aot_close(et_mtp_aot_session_t *session);

int32_t et_mtp_aot_init_weights(et_mtp_aot_session_t *session, int64_t seed);
int32_t et_mtp_aot_init_sequence(et_mtp_aot_session_t *session, int64_t seed);

/* Run one forward + backward + AdamW step on the session sequence. */
int32_t et_mtp_aot_train_step(et_mtp_aot_session_t *session);

/* Run central finite-difference gradient check across ALL parameters and d_h0. */
int32_t et_mtp_aot_fd_grad_check(et_mtp_aot_session_t *session, double fd_eps);

/*
 * Run self-speculative greedy decoding over the trained sequence using the
 * model's main head as target logits, comparing bitwise against step-by-step
 * greedy decode and computing mean emitted tokens per verification step.
 */
int32_t et_mtp_aot_spec_greedy_verify(et_mtp_aot_session_t *session);

/* Checkpoint save/load ("ETMT1000" v1.0) and state SHA-256 digest accessors. */
int32_t et_mtp_aot_save_checkpoint(const et_mtp_aot_session_t *session,
                                   const char *path);
int32_t et_mtp_aot_load_checkpoint(et_mtp_aot_session_t *session,
                                   const char *path);
int32_t et_mtp_aot_compute_state_digest(et_mtp_aot_session_t *session);
int32_t et_mtp_aot_digest_byte(const et_mtp_aot_session_t *session,
                               int64_t index);

/* Negative boundary & tamper probes for compiled Eshkol AOT driver. */
int32_t et_mtp_aot_probe_invalid_depth(int64_t bad_depth);
int32_t et_mtp_aot_probe_short_seq(const et_mtp_aot_session_t *session,
                                   int64_t short_len);
int32_t et_mtp_aot_probe_oob_token(const et_mtp_aot_session_t *session);
int32_t et_mtp_aot_probe_nonfinite_h0(const et_mtp_aot_session_t *session);
int32_t et_mtp_aot_probe_fp_env_trap(const et_mtp_aot_session_t *session);
int32_t et_mtp_aot_probe_ckpt_config_mismatch(const char *path,
                                              int64_t alt_depth);

int32_t et_mtp_aot_corrupt_file_byte(const char *path, int64_t byte_offset);
int32_t et_mtp_aot_truncate_file(const char *path, int64_t new_size);
int32_t et_mtp_aot_copy_file(const char *src_path, const char *dst_path);
int32_t et_mtp_aot_make_symlink(const char *target_path, const char *link_path);

/* Scalar telemetry accessors for Eshkol AOT driver. */
int64_t et_mtp_aot_total_params(const et_mtp_aot_session_t *session);
int64_t et_mtp_aot_step(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_loss_total(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_loss_main(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_loss_mtp_avg(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_loss_depth(const et_mtp_aot_session_t *session,
                                  int64_t depth_idx);
double et_mtp_aot_last_grad_norm(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_fd_param_max_err(const et_mtp_aot_session_t *session);
double et_mtp_aot_last_fd_h0_max_err(const et_mtp_aot_session_t *session);
int64_t et_mtp_aot_last_fd_checked(const et_mtp_aot_session_t *session);
int64_t et_mtp_aot_spec_exact_match(const et_mtp_aot_session_t *session);
int64_t et_mtp_aot_spec_total_emitted(const et_mtp_aot_session_t *session);
int64_t et_mtp_aot_spec_verify_steps(const et_mtp_aot_session_t *session);
double et_mtp_aot_spec_tokens_per_step(const et_mtp_aot_session_t *session);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_TRANSFORMER_MTP_HEAD_ESHKOL_BRIDGE_H */
