#ifndef ESHKOL_TRANSFORMER_WAVE4_COMPOSED_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_WAVE4_COMPOSED_ESHKOL_BRIDGE_H

#include "eshkol_transformer/wave4_composed_trainer_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bridge-specific status codes outside et_wave4_composed_status. */
enum {
  ET_WAVE4_COMPOSED_AOT_OUT_OF_MEMORY = 400,
  ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED = 401,
  ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL = 402
};

enum {
  ET_WAVE4_COMPOSED_AOT_SHA256_BYTES = ET_EVAL5_SHA256_BYTES,
  ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES = ET_EVAL5_SHA256_BYTES * 2u + 1u
};

typedef struct et_wave4_composed_aot_context et_wave4_composed_aot_context;

/*
 * Allocate, validate, and connect a Wave 4/5 composed trainer context for
 * compiled Eshkol AOT callers (MP4 + AC4 + DIST5 + MOD5 + EVAL5).
 *
 * Uses the reviewed CPU shape [N=1, T=2, Hq=2, Hkv=1, Dh=2] with L in {4, 8},
 * narrow storage dtype in {ET_WAVE4_COMPOSED_F16, ET_WAVE4_COMPOSED_BF16},
 * and AC4 policy in {NONE, EVERY_LAYER, SQRT_SEGMENTS, EVERY_K}.
 *
 * For world > 1, establishes real AF_UNIX SOCK_STREAM peer connections inside
 * `sock_dir` bounded by `timeout_ms` before creating the DIST5 communicator.
 * For world == 1, no sockets are created and `sock_dir` may be NULL or empty.
 *
 * Returns ET_WAVE4_COMPOSED_OK (0) on success or an explicit non-zero error
 * code. Leaves *out_ctx unchanged on any failure. Zero silent fallbacks.
 */
int32_t et_wave4_composed_aot_create(
    const char *sock_dir, int64_t rank, int64_t world, int64_t timeout_ms,
    int64_t layers, int64_t dtype, int64_t policy, int64_t checkpoint_every,
    double epsilon, double dropout_rate, int64_t seed, double learning_rate,
    double momentum, double init_scale, double growth_factor,
    double backoff_factor, double min_scale, double max_scale,
    int64_t growth_interval, et_wave4_composed_aot_context **out_ctx);

/* Release communicator, buffers, and context. Safe on NULL. */
void et_wave4_composed_aot_destroy(et_wave4_composed_aot_context *ctx);

/* Buffer mutators (return OK or INVALID_ARGUMENT). */
int32_t et_wave4_composed_aot_set_master(et_wave4_composed_aot_context *ctx,
                                         int64_t index, double value);
int32_t et_wave4_composed_aot_set_input(et_wave4_composed_aot_context *ctx,
                                        int64_t index, double value);
int32_t et_wave4_composed_aot_set_target(et_wave4_composed_aot_context *ctx,
                                         int64_t index, double value);

/* Toggle deterministic single-rank narrow-overflow target synthesis (0 or 1). */
int32_t et_wave4_composed_aot_set_overflow_inject(
    et_wave4_composed_aot_context *ctx, int64_t enabled);

/* Update the AC4 activation cap (0 disables). Rejects NULL or negative values. */
int32_t et_wave4_composed_aot_set_activation_limit(
    et_wave4_composed_aot_context *ctx, int64_t limit_bytes);

/* Broadcast rank 0 master, zero momentum, and initialize the MP4 loss scaler. */
int32_t et_wave4_composed_aot_bootstrap(et_wave4_composed_aot_context *ctx);

/* Execute one composed distributed training step at AC4 counter `step`. */
int32_t et_wave4_composed_aot_step(et_wave4_composed_aot_context *ctx,
                                   int64_t step);

/* Run local EVAL5 evaluation, emit canonical manifest atomically to eval_path,
 * reload and verify SHA-256 + canonical bytes, and store results on ctx. */
int32_t et_wave4_composed_aot_evaluate(et_wave4_composed_aot_context *ctx,
                                       const char *eval_path);

/* Reload an EVAL5 manifest from eval_path and verify its embedded SHA-256 and
 * agreement with ctx's recorded evaluation manifest. */
int32_t et_wave4_composed_aot_verify_manifest(
    const et_wave4_composed_aot_context *ctx, const char *eval_path);

/* Planned geometry and memory accessors (-1 on NULL). */
int64_t et_wave4_composed_aot_parameter_count(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_activation_count(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_planned_activation_peak_bytes(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_planned_workspace_bytes(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_narrow_parameter_bytes(
    const et_wave4_composed_aot_context *ctx);

/* Step, scaler, and weight accessors. */
int32_t et_wave4_composed_aot_last_status(
    const et_wave4_composed_aot_context *ctx);
double et_wave4_composed_aot_last_loss(const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_last_local_overflow(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_live_activation_peak_bytes(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_workspace_bytes(
    const et_wave4_composed_aot_context *ctx);
double et_wave4_composed_aot_scaler_scale(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_scaler_clean_steps(
    const et_wave4_composed_aot_context *ctx);
double et_wave4_composed_aot_get_master(const et_wave4_composed_aot_context *ctx,
                                        int64_t index);
int64_t et_wave4_composed_aot_master_bits(
    const et_wave4_composed_aot_context *ctx, int64_t index);

/* Exact SHA-256 digest of current f32 master parameter bytes. */
int32_t et_wave4_composed_aot_param_digest_hex(
    const et_wave4_composed_aot_context *ctx, char *out_buf, int64_t capacity);
int64_t et_wave4_composed_aot_param_digest_byte(
    const et_wave4_composed_aot_context *ctx, int64_t index);

/* EVAL5 metric and manifest digest accessors. */
double et_wave4_composed_aot_eval_mean_nll(
    const et_wave4_composed_aot_context *ctx);
double et_wave4_composed_aot_eval_perplexity(
    const et_wave4_composed_aot_context *ctx);
int64_t et_wave4_composed_aot_eval_contaminated_documents(
    const et_wave4_composed_aot_context *ctx);
int32_t et_wave4_composed_aot_manifest_digest_hex(
    const et_wave4_composed_aot_context *ctx, char *out_buf, int64_t capacity);
int64_t et_wave4_composed_aot_manifest_digest_byte(
    const et_wave4_composed_aot_context *ctx, int64_t index);

/* Safe unsigned byte reader for caller-owned C string buffers (-1 on error). */
int64_t et_wave4_composed_aot_read_byte(const char *buffer, int64_t index);

#ifdef __cplusplus
}
#endif

#endif
