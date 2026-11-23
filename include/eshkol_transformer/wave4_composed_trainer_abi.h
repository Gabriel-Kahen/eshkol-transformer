#ifndef ESHKOL_TRANSFORMER_WAVE4_COMPOSED_TRAINER_ABI_H
#define ESHKOL_TRANSFORMER_WAVE4_COMPOSED_TRAINER_ABI_H

#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include "eshkol_transformer/dist5_collectives_abi.h"
#include "eshkol_transformer/eval5_benchmark_abi.h"
#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ET_WAVE4_COMPOSED_ABI_MAJOR 1u
#define ET_WAVE4_COMPOSED_ABI_MINOR 0u

typedef uint32_t et_wave4_composed_status;
enum {
  ET_WAVE4_COMPOSED_OK = 0,
  ET_WAVE4_COMPOSED_OVERFLOW_BACKOFF = 1,
  ET_WAVE4_COMPOSED_UNDERFLOW_FAULT = 2,
  ET_WAVE4_COMPOSED_INVALID_ARGUMENT = 3,
  ET_WAVE4_COMPOSED_ALIASING = 4,
  ET_WAVE4_COMPOSED_WORKSPACE_TOO_SMALL = 5,
  ET_WAVE4_COMPOSED_UNSUPPORTED_ENVIRONMENT = 6,
  ET_WAVE4_COMPOSED_ACTIVATION_LIMIT = 7,
  ET_WAVE4_COMPOSED_NUMERIC_ERROR = 8,
  ET_WAVE4_COMPOSED_PROVIDER_ERROR = 9,
  ET_WAVE4_COMPOSED_DIST5_ERROR_BASE = 100, /* Add the DIST5 error. */
  ET_WAVE4_COMPOSED_EVAL5_ERROR_BASE = 200  /* Subtract the EVAL5 error. */
};

typedef uint32_t et_wave4_composed_dtype;
enum { ET_WAVE4_COMPOSED_F16 = 1, ET_WAVE4_COMPOSED_BF16 = 2 };

typedef struct {
  et_ac4_config_v1 ac4; /* Layers 4 or 8, mode FORWARD_BACKWARD. */
  et_wave4_composed_dtype dtype;
  float learning_rate; /* Finite and strictly positive. */
  float momentum;      /* Finite in [0, 1). */
} et_wave4_composed_config_v1;

typedef struct {
  /* Finite, positive bounds matching MP4; 1.0f / min_scale must also be finite
   * and positive so composed unscaling stays representable in binary32. */
  float scale, growth_factor, backoff_factor, min_scale, max_scale;
  uint32_t growth_interval;
} et_wave4_composed_scaler_config_v1;

typedef struct {
  size_t parameter_count;
  float *master;
  float *momentum;
  et_mp4_loss_scaler_v1 scaler;
} et_wave4_composed_state_v1;

typedef struct {
  et_ac4_const_span_v1 input;
  et_ac4_const_span_v1 target;
} et_wave4_composed_batch_v1;

typedef struct {
  void *data; /* Caller-owned allocated scratch, aligned for float. */
  size_t bytes;
} et_wave4_composed_workspace_v1;

typedef struct {
  et_ac4_memory_v1 ac4;
  size_t narrow_parameter_bytes; /* Same buffer reused for scaled gradients. */
  size_t workspace_bytes;        /* Includes AC4 and composition scratch. */
} et_wave4_composed_memory_v1;

typedef struct {
  double loss; /* Local pre-update half mean squared error. */
  et_wave4_composed_memory_v1 memory;
  uint32_t local_overflow;
} et_wave4_composed_result_v1;

typedef struct {
  et_wave4_composed_batch_v1 batch;
  et_eval5_document train_document;
  et_eval5_document eval_document;
  const uint32_t *target_ids; /* One per output row, [N,T]. */
  size_t target_count;
  const char *path;
  const et_eval5_manifest *identity;
} et_wave4_composed_eval_v1;

/*
 * CPU x86-64, reviewed AC4 shapes, MP4 narrow storage and f32 compute.
 * Descriptors and their full payload spans must be naturally aligned and
 * disjoint. Counts must exactly match the plan. Scratch may change on failure.
 */
et_wave4_composed_status
et_wave4_composed_plan_v1(const et_wave4_composed_config_v1 *config,
                          et_wave4_composed_memory_v1 *out);

/*
 * All ranks supply matching configuration, scaler settings, and collective
 * call order on a dedicated DIST5 communicator. Each rank supplies a
 * readable master buffer. Bootstrap broadcasts rank 0's master, zeros
 * momentum and initializes the scaler. A rank may supply its own AC4 seed.
 * A transport failure requires discarding the distributed state and
 * communicator. The caller controls the AC4 step counter and retry policy.
 */
et_wave4_composed_status
et_wave4_composed_bootstrap_v1(const et_wave4_composed_config_v1 *config,
                               et_dist5_comm_t *comm, int timeout_ms,
                               const et_wave4_composed_scaler_config_v1 *scaler,
                               et_wave4_composed_state_v1 *state,
                               et_wave4_composed_workspace_v1 workspace);

/*
 * One equal-size shard per rank. Accept commits momentum SGD and the scaler.
 * Backoff commits only the scaler. Both publish result. Other outcomes leave
 * the local master, momentum, scaler, and result unchanged.
 */
et_wave4_composed_status
et_wave4_composed_step_v1(const et_wave4_composed_config_v1 *config,
                          et_dist5_comm_t *comm, int timeout_ms,
                          et_wave4_composed_state_v1 *state,
                          const et_wave4_composed_batch_v1 *batch,
                          et_wave4_composed_workspace_v1 workspace,
                          et_wave4_composed_result_v1 *result);

/*
 * Local evaluation on the caller-selected rank. The held-out output [N,T,D]
 * is scored as [N*T,D] logits, with dropout disabled. The two documents use
 * byte n-grams and cover the audit for this fixture. Identity supplies
 * version, names, hashes, max_samples, and contamination settings. The
 * implementation computes context length, LM metrics and audit, writes the
 * manifest atomically, then reloads and verifies it before publishing out.
 */
et_wave4_composed_status
et_wave4_composed_evaluate_v1(const et_wave4_composed_config_v1 *config,
                              const et_wave4_composed_state_v1 *state,
                              const et_wave4_composed_eval_v1 *eval,
                              et_wave4_composed_workspace_v1 workspace,
                              et_eval5_manifest *out);

#ifdef __cplusplus
}
#endif
#endif
