#ifndef ESHKOL_TRANSFORMER_TR3_C_O2_RESTORE_INTERNAL_H
#define ESHKOL_TRANSFORMER_TR3_C_O2_RESTORE_INTERNAL_H

#include "o2_optimizer_internal.h"
#include "tr3_c_i2_restore_internal.h"

#include <stddef.h>
#include <stdint.h>

#if !defined(ET_TR3_C_O2_RESTORE_NATIVE)
#error "TR3-C O2 restore declarations require the private restore feature"
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_o2_tr3_c_restore_entry_v1 {
  size_t struct_size;
  uint32_t learning_rate_bits;
  uint32_t beta1_bits;
  uint32_t beta2_bits;
  uint32_t epsilon_bits;
  uint32_t weight_decay_bits;
  uint32_t reserved;
  const et_f32_tensor *exp_avg_owned;
  const et_f32_tensor *exp_avg_sq_owned;
} et_o2_tr3_c_restore_entry_v1;

typedef struct et_o2_tr3_c_restore_request_v1 {
  size_t struct_size;
  uint64_t target_completed_updates;
  uint64_t expected_receiver_completed_updates;
  uint32_t clip_kind;
  uint32_t clip_max_bits;
  uint32_t schedule_kind;
  uint32_t minimum_ratio_bits;
  uint64_t warmup_updates;
  uint64_t total_updates;
  et_o2_tr3_c_restore_entry_v1 entries[ET_TR3_C_MODEL_DESTINATIONS];
} et_o2_tr3_c_restore_request_v1;

#define ET_O2_TR3_C_RESTORE_ENTRY_V1_0_SIZE                                  \
  ((size_t)sizeof(et_o2_tr3_c_restore_entry_v1))
#define ET_O2_TR3_C_RESTORE_REQUEST_V1_0_SIZE                                \
  ((size_t)sizeof(et_o2_tr3_c_restore_request_v1))

#if defined(__cplusplus)
#define ET_O2_TR3_C_STATIC_ASSERT(condition, message)                         \
  static_assert(condition, message)
#else
#define ET_O2_TR3_C_STATIC_ASSERT(condition, message)                         \
  _Static_assert(condition, message)
#endif

ET_O2_TR3_C_STATIC_ASSERT(sizeof(size_t) == 8u && sizeof(void *) == 8u,
                          "TR3-C O2 restore requires LP64 carriers");
ET_O2_TR3_C_STATIC_ASSERT(sizeof(et_o2_tr3_c_restore_entry_v1) == 48u,
                          "TR3-C O2 restore entry size changed");
ET_O2_TR3_C_STATIC_ASSERT(
    offsetof(et_o2_tr3_c_restore_entry_v1, learning_rate_bits) == 8u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, beta1_bits) == 12u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, beta2_bits) == 16u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, epsilon_bits) == 20u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, weight_decay_bits) == 24u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, reserved) == 28u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, exp_avg_owned) == 32u &&
        offsetof(et_o2_tr3_c_restore_entry_v1, exp_avg_sq_owned) == 40u,
    "TR3-C O2 restore entry offsets changed");
ET_O2_TR3_C_STATIC_ASSERT(sizeof(et_o2_tr3_c_restore_request_v1) == 728u,
                          "TR3-C O2 restore request size changed");
ET_O2_TR3_C_STATIC_ASSERT(
    offsetof(et_o2_tr3_c_restore_request_v1, target_completed_updates) == 8u &&
        offsetof(et_o2_tr3_c_restore_request_v1,
                 expected_receiver_completed_updates) == 16u &&
        offsetof(et_o2_tr3_c_restore_request_v1, clip_kind) == 24u &&
        offsetof(et_o2_tr3_c_restore_request_v1, clip_max_bits) == 28u &&
        offsetof(et_o2_tr3_c_restore_request_v1, schedule_kind) == 32u &&
        offsetof(et_o2_tr3_c_restore_request_v1, minimum_ratio_bits) == 36u &&
        offsetof(et_o2_tr3_c_restore_request_v1, warmup_updates) == 40u &&
        offsetof(et_o2_tr3_c_restore_request_v1, total_updates) == 48u &&
        offsetof(et_o2_tr3_c_restore_request_v1, entries) == 56u,
    "TR3-C O2 restore request offsets changed");

#undef ET_O2_TR3_C_STATIC_ASSERT

typedef struct et_o2_tr3_c_restore_plan et_o2_tr3_c_restore_plan;

int32_t et_o2_tr3_c_restore_prepare_v1(
    et_o2_optimizer *optimizer,
    const et_o2_tr3_c_restore_request_v1 *request,
    et_o2_tr3_c_restore_plan **plan, et_o2_error_v1 *error);
int32_t et_o2_tr3_c_restore_append_v1(et_o2_tr3_c_restore_plan *plan,
                                      void *exact_i2_copy_builder,
                                      size_t first_index,
                                      et_o2_error_v1 *error);
int32_t et_o2_tr3_c_restore_check_v1(et_o2_tr3_c_restore_plan *plan,
                                     et_o2_error_v1 *error);
void et_o2_tr3_c_restore_commit_native_v1(et_o2_tr3_c_restore_plan *plan);
void et_o2_tr3_c_restore_finalize_v1(et_o2_tr3_c_restore_plan *plan);
int32_t et_o2_tr3_c_restore_abort_v1(et_o2_tr3_c_restore_plan *plan,
                                     et_o2_error_v1 *error);

/* Existing checked I2 tail consumed by this private native participant. */
void et_tr3_c_i2_owned_release_checked_v1(void *owned);

#ifdef ET_O2_TESTING
typedef struct et_o2_tr3_c_restore_test_counts_v1 {
  size_t struct_size;
  size_t live_plans;
  size_t committed_plans;
  size_t aborted_plans;
  size_t live_stages;
  size_t retained_control_bytes;
} et_o2_tr3_c_restore_test_counts_v1;

void et_o2_tr3_c_restore_test_counts_snapshot_v1(
    et_o2_tr3_c_restore_test_counts_v1 *counts);
#endif

#ifdef __cplusplus
}
#endif

#endif
