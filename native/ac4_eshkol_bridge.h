#ifndef ESHKOL_TRANSFORMER_AC4_ESHKOL_BRIDGE_H
#define ESHKOL_TRANSFORMER_AC4_ESHKOL_BRIDGE_H

#include "eshkol_transformer/ac4_checkpointing_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bridge-specific error code beyond et_ac4_status (0..8). */
enum { ET_AC4_AOT_OUT_OF_MEMORY = 100 };

typedef struct et_ac4_aot_context et_ac4_aot_context;

/*
 * Allocate and validate an AC4 checkpointed training/evaluation context for
 * compiled Eshkol AOT callers.
 *
 * Returns et_ac4_status (ET_AC4_OK == 0 on success) or
 * ET_AC4_AOT_OUT_OF_MEMORY on allocation failure. Leaves *out_ctx unchanged
 * on any failure. Zero silent fallbacks.
 */
int32_t et_ac4_aot_create(int64_t layers, int64_t batch, int64_t tokens,
                          int64_t query_heads, int64_t kv_heads,
                          int64_t head_dim, int64_t mode, int64_t policy,
                          int64_t checkpoint_every, double epsilon,
                          double dropout_rate, int64_t seed,
                          int64_t activation_limit_bytes,
                          et_ac4_aot_context **out_ctx);

/* Release a context created by et_ac4_aot_create. Safe on NULL. */
void et_ac4_aot_destroy(et_ac4_aot_context *ctx);

/* Update the activation cap (0 disables). Rejects NULL or negative values. */
int32_t et_ac4_aot_set_activation_limit(et_ac4_aot_context *ctx,
                                        int64_t activation_limit_bytes);

/* Planned geometry and memory accessors. Return -1 if ctx is NULL. */
int64_t et_ac4_aot_activation_floats(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_parameter_floats(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_parameter_floats_per_layer(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_peak_bytes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_workspace_bytes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_gradient_staging_bytes(
    const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_retained_checkpoints(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_retained_tapes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_planned_rematerialized_layers(const et_ac4_aot_context *ctx);

/*
 * Element mutators for input, target, and packed parameter buffers.
 * Values outside finite binary32 range become +/-INFINITY (NaN stays NaN)
 * and are explicitly rejected with ET_AC4_NUMERIC_ERROR at run time.
 * Returns ET_AC4_OK or ET_AC4_INVALID_ARGUMENT (NULL handle or OOB index).
 */
int32_t et_ac4_aot_set_input(et_ac4_aot_context *ctx, int64_t index,
                             double value);
int32_t et_ac4_aot_set_target(et_ac4_aot_context *ctx, int64_t index,
                              double value);
int32_t et_ac4_aot_set_parameter(et_ac4_aot_context *ctx, int64_t index,
                                 double value);

/*
 * Execute one forward or checkpointed forward+backward step at counter `step`.
 * Failure-atomic: outputs, gradients, and last result are unchanged on error.
 * Performs zero heap allocations.
 */
int32_t et_ac4_aot_run(et_ac4_aot_context *ctx, int64_t step);

/*
 * Apply one failure-atomic binary32 SGD step:
 *   parameters[i] -= (float)learning_rate * d_parameters[i].
 * Requires a succeeded ET_AC4_FORWARD_BACKWARD run and finite positive lr.
 * Performs zero heap allocations.
 */
int32_t et_ac4_aot_sgd_step(et_ac4_aot_context *ctx, double learning_rate);

/* Observed telemetry from the last successful run (-1 if NULL or no run). */
int64_t et_ac4_aot_peak_activation_bytes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_recomputation_count(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_retained_checkpoints(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_retained_tapes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_workspace_bytes(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_gradient_staging_bytes(const et_ac4_aot_context *ctx);

/* Scalar value accessors (return NAN on NULL, missing result, or OOB). */
double et_ac4_aot_get_loss(const et_ac4_aot_context *ctx);
double et_ac4_aot_get_output(const et_ac4_aot_context *ctx, int64_t index);
double et_ac4_aot_get_parameter(const et_ac4_aot_context *ctx, int64_t index);
double et_ac4_aot_get_d_input(const et_ac4_aot_context *ctx, int64_t index);
double et_ac4_aot_get_d_parameter(const et_ac4_aot_context *ctx, int64_t index);

/*
 * Exact IEEE-754 bit-pattern accessors for bitwise parity verification.
 * Return an unsigned 32-bit value in [0, 0xFFFFFFFF] or -1 on invalid call.
 */
int64_t et_ac4_aot_loss_bits_lo(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_loss_bits_hi(const et_ac4_aot_context *ctx);
int64_t et_ac4_aot_output_bits(const et_ac4_aot_context *ctx, int64_t index);
int64_t et_ac4_aot_parameter_bits(const et_ac4_aot_context *ctx, int64_t index);
int64_t et_ac4_aot_d_input_bits(const et_ac4_aot_context *ctx, int64_t index);
int64_t et_ac4_aot_d_parameter_bits(const et_ac4_aot_context *ctx,
                                    int64_t index);

#ifdef __cplusplus
}
#endif

#endif
