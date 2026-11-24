#ifndef ESHKOL_TRANSFORMER_O2_MUON_ABI_H
#define ESHKOL_TRANSFORMER_O2_MUON_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_MUON_ABI_MAJOR 1u
#define ET_MUON_ABI_MINOR 0u

/* Bounded dimensions for deterministic, failure-atomic CPU execution. */
#define ET_MUON_MIN_DIM 1u
#define ET_MUON_MAX_DIM 64u
#define ET_MUON_MAX_VEC_LEN 4096u
#define ET_MUON_MAX_GROUPS 64u

#define ET_MUON_MIN_NS_STEPS 1u
#define ET_MUON_DEFAULT_NS_STEPS 5u
#define ET_MUON_MAX_NS_STEPS 16u

/* Quintic Newton-Schulz orthogonalization coefficients:
 *   X_{k+1} = a * X_k + (b * A_k + c * A_k^2) * X_k,  A_k = X_k X_k^T
 */
#define ET_MUON_NS_A 3.4445
#define ET_MUON_NS_B (-4.7750)
#define ET_MUON_NS_C 2.0315

typedef int32_t et_muon_status;

enum {
  ET_MUON_OK = 0,
  ET_MUON_ERR_ARGUMENT = -1,
  ET_MUON_ERR_SHAPE = -2,
  ET_MUON_ERR_ALIGNMENT = -3,
  ET_MUON_ERR_ALIAS = -4,
  ET_MUON_ERR_NONFINITE = -5,
  ET_MUON_ERR_FP_ENV = -6,
  ET_MUON_ERR_NOMEM = -7,
  ET_MUON_ERR_OVERFLOW = -8
};

/* Parameter group optimizer kind. */
enum {
  ET_MUON_GROUP_MUON_2D = 0u,
  ET_MUON_GROUP_ADAMW_1D = 1u
};

/* WSD (Warmup-Stable-Decay) decay curve kind. */
enum {
  ET_WSD_DECAY_LINEAR = 0u,
  ET_WSD_DECAY_COSINE = 1u
};

/* WSD schedule phase reported at a given 0-based step. */
enum {
  ET_WSD_PHASE_WARMUP = 0u,
  ET_WSD_PHASE_STABLE = 1u,
  ET_WSD_PHASE_DECAY = 2u
};

/* WSD (Warmup-Stable-Decay) LR scheduler configuration.
 * Constraints:
 *   total_steps >= 1
 *   warmup_steps + stable_steps <= total_steps (no uint64 overflow)
 *   max_lr, min_lr finite binary64 with 0.0 <= min_lr <= max_lr and max_lr > 0.0
 *   decay_kind in {ET_WSD_DECAY_LINEAR, ET_WSD_DECAY_COSINE}
 *   reserved == 0
 *
 * Schedule semantics for 0-based `step`:
 *   stable_end  = warmup_steps + stable_steps
 *   decay_steps = total_steps - stable_end
 *   - Warmup [0, warmup_steps):
 *       phase = ET_WSD_PHASE_WARMUP
 *       lr    = max_lr * ((double)(step + 1) / (double)warmup_steps)
 *   - Stable [warmup_steps, stable_end):
 *       phase = ET_WSD_PHASE_STABLE
 *       lr    = max_lr
 *   - Decay  [stable_end, total_steps] (clamped at min_lr for step > total_steps):
 *       phase = ET_WSD_PHASE_DECAY
 *       if decay_steps == 0 or step >= total_steps: lr = min_lr
 *       else if step == stable_end:                 lr = max_lr
 *       else with t = (double)(step - stable_end) / (double)decay_steps:
 *         LINEAR: lr = max_lr - (max_lr - min_lr) * t
 *         COSINE: lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + cos(pi * t))
 */
typedef struct et_wsd_config {
  uint64_t warmup_steps;
  uint64_t stable_steps;
  uint64_t total_steps;
  double max_lr;
  double min_lr;
  uint32_t decay_kind;
  uint32_t reserved;
} et_wsd_config;

/* Parameter group descriptor for a hybrid Muon + AdamW step.
 * - For ET_MUON_GROUP_MUON_2D:
 *     rows in [1, 64], cols in [1, 64], count = rows * cols
 *     params:       float[count], 4-byte aligned (in/out)
 *     grads:        const float[count], 4-byte aligned (in)
 *     momentum_buf: float[count], 4-byte aligned (in/out, buffer B)
 *     variance_buf: must be NULL
 *     apply_weight_decay: 0 (no decay) or 1 (decoupled weight decay)
 * - For ET_MUON_GROUP_ADAMW_1D:
 *     rows in [1, 4096] (vector length), cols == 1, count = rows
 *     params:       float[count], 4-byte aligned (in/out)
 *     grads:        const float[count], 4-byte aligned (in)
 *     momentum_buf: float[count], 4-byte aligned (in/out, first moment m)
 *     variance_buf: float[count], 4-byte aligned (in/out, second moment v >= 0)
 *     apply_weight_decay: 0 (no decay) or 1 (decoupled weight decay)
 *
 * All non-NULL data spans across all groups in a step must be pairwise disjoint.
 */
typedef struct et_muon_param_group {
  uint32_t kind;
  uint32_t rows;
  uint32_t cols;
  uint32_t apply_weight_decay;
  void *params;
  const void *grads;
  void *momentum_buf;
  void *variance_buf;
} et_muon_param_group;

/* Shared hyperparameters for one hybrid Muon + AdamW step.
 * Constraints:
 *   lr finite >= 0.0
 *   step_index >= 1 (1-based update count for AdamW bias correction)
 *   muon_beta finite in [0.0, 1.0)
 *   muon_eps finite > 0.0
 *   muon_weight_decay finite >= 0.0
 *   muon_ns_steps in [1, 16]
 *   muon_nesterov in {0, 1}
 *   adamw_beta1, adamw_beta2 finite in [0.0, 1.0)
 *   adamw_eps finite > 0.0
 *   adamw_weight_decay finite >= 0.0
 *   adamw_lr_scale finite > 0.0
 */
typedef struct et_muon_step_config {
  double lr;
  uint64_t step_index;
  double muon_beta;
  double muon_eps;
  double muon_weight_decay;
  uint32_t muon_ns_steps;
  uint32_t muon_nesterov;
  double adamw_beta1;
  double adamw_beta2;
  double adamw_eps;
  double adamw_weight_decay;
  double adamw_lr_scale;
} et_muon_step_config;

#if defined(__cplusplus)
#define ET_MUON_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define ET_MUON_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
ET_MUON_STATIC_ASSERT(sizeof(et_wsd_config) == 48u &&
                          offsetof(et_wsd_config, max_lr) == 24u &&
                          offsetof(et_wsd_config, decay_kind) == 40u,
                      "et_wsd_config ABI layout changed");
ET_MUON_STATIC_ASSERT(sizeof(et_muon_param_group) == 16u + 4u * sizeof(void *) &&
                          offsetof(et_muon_param_group, params) == 16u,
                      "et_muon_param_group ABI layout changed");
ET_MUON_STATIC_ASSERT(sizeof(et_muon_step_config) == 88u &&
                          offsetof(et_muon_step_config, muon_ns_steps) == 40u &&
                          offsetof(et_muon_step_config, adamw_lr_scale) == 80u,
                      "et_muon_step_config ABI layout changed");
#undef ET_MUON_STATIC_ASSERT

/* Newton-Schulz quintic orthogonalization of dense row-major binary32
 * G [rows, cols] into disjoint U [rows, cols].
 * Normalizes by (||G||_F + eps) in binary64, transposes when rows > cols,
 * runs ns_steps quintic iterations in binary64, and rounds finite output to f32.
 * Failure leaves U untouched. */
et_muon_status et_muon_newton_schulz5_f32(uint32_t rows, uint32_t cols,
                                          const float *g, uint32_t ns_steps,
                                          double eps, float *u);
et_muon_status et_muon_newton_schulz5_raw(uint32_t rows, uint32_t cols,
                                          const void *g, uint32_t ns_steps,
                                          double eps, void *u);

/* Orthogonality error diagnostic in binary64:
 *   rows <= cols: ||U U^T - I_{rows}||_F
 *   rows >  cols: ||U^T U - I_{cols}||_F
 * Failure leaves *out_error untouched. */
et_muon_status et_muon_orthogonality_error_f32(uint32_t rows, uint32_t cols,
                                               const float *u,
                                               double *out_error);
et_muon_status et_muon_orthogonality_error_raw(uint32_t rows, uint32_t cols,
                                               const void *u,
                                               double *out_error);

/* Validate WSD scheduler configuration and FP environment. */
et_muon_status et_wsd_validate_config(const et_wsd_config *cfg);

/* Evaluate WSD learning rate (and optional phase) at 0-based `step`.
 * Failure leaves *out_lr and *out_phase untouched. */
et_muon_status et_wsd_lr(const et_wsd_config *cfg, uint64_t step,
                         double *out_lr, uint32_t *out_phase);

/* Validate hybrid Muon + AdamW step configuration and FP environment. */
et_muon_status et_muon_validate_step_config(const et_muon_step_config *cfg);

/* Failure-atomic hybrid Muon (2D) + AdamW (1D) step over `num_groups` groups.
 * Stages every updated parameter and optimizer state buffer before committing
 * any write. Any validation or non-finite error preserves every caller buffer
 * bit-for-bit. */
et_muon_status et_muon_hybrid_step(const et_muon_step_config *cfg,
                                   size_t num_groups,
                                   const et_muon_param_group *groups);

/* Failure-atomic WSD-scheduled hybrid step at 0-based `schedule_step`.
 * Evaluates WSD LR at `schedule_step`, sets lr = scheduled_lr and
 * step_index = schedule_step + 1, and commits all groups atomically.
 * Writes optional *out_lr and *out_phase only on success. */
et_muon_status et_muon_hybrid_step_wsd(const et_wsd_config *wsd,
                                       uint64_t schedule_step,
                                       const et_muon_step_config *base_cfg,
                                       size_t num_groups,
                                       const et_muon_param_group *groups,
                                       double *out_lr, uint32_t *out_phase);

#ifdef __cplusplus
}
#endif

#endif
