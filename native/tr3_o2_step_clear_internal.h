#ifndef ESHKOL_TRANSFORMER_TR3_O2_STEP_CLEAR_INTERNAL_H
#define ESHKOL_TRANSFORMER_TR3_O2_STEP_CLEAR_INTERNAL_H

#include "o2_optimizer_internal.h"

#include <stddef.h>
#include <stdint.h>

#if defined(ET_TR3_O2_STEP_CLEAR_NATIVE) ||                                  \
    defined(ET_TR3_O2_STEP_CLEAR_BRIDGE)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_o2_trainer_step_clear_plan
    et_o2_trainer_step_clear_plan;

int32_t et_o2_trainer_step_clear_prepare_v1(
    et_o2_optimizer *optimizer, uint64_t expected_completed_updates,
    uint64_t expected_contribution_count,
    uint32_t expected_normalization_weight_bits,
    et_o2_trainer_step_clear_plan **plan, et_o2_error_v1 *error);

int32_t et_o2_trainer_step_clear_commit_v1(
    et_o2_trainer_step_clear_plan *plan, et_o2_error_v1 *error);

int32_t et_o2_trainer_step_clear_abort_v1(
    et_o2_trainer_step_clear_plan *plan, et_o2_error_v1 *error);

#ifdef ET_O2_TESTING
enum {
  ET_O2_TR3_TEST_TERMINAL_NONE = 0,
  ET_O2_TR3_TEST_TERMINAL_COPY_COMMIT = 1,
  ET_O2_TR3_TEST_TERMINAL_RESET_COMMIT = 2,
  ET_O2_TR3_TEST_TERMINAL_RESET_RELEASE = 3,
  ET_O2_TR3_TEST_TERMINAL_COPY_RELEASE = 4,
  ET_O2_TR3_TEST_TERMINAL_STAGE_DESTROY = 5
};

typedef struct et_o2_trainer_step_clear_test_counts_v1 {
  size_t struct_size;
  size_t live_plans;
  size_t committed_plans;
  size_t aborted_plans;
  size_t live_stages;
  size_t retained_control_bytes;
} et_o2_trainer_step_clear_test_counts_v1;

void et_o2_trainer_step_clear_test_counts_snapshot_v1(
    et_o2_trainer_step_clear_test_counts_v1 *counts);
void et_o2_trainer_step_clear_test_fail_alloc_after_v1(
    size_t successful_allocations);
void et_o2_trainer_step_clear_test_fail_publication_v1(int enabled);
void et_o2_trainer_step_clear_test_fail_terminal_v1(uint32_t site,
                                                     size_t stage_index);
void et_o2_trainer_step_clear_test_reset_failpoints_v1(void);
const et_f32_tensor *et_o2_trainer_step_clear_test_moment_storage_v1(
    const et_o2_optimizer *optimizer, size_t index, uint32_t moment_kind);
const et_f32_tensor *et_o2_trainer_step_clear_test_stage_v1(
    const et_o2_trainer_step_clear_plan *plan, size_t stage_index);
#endif

#ifdef __cplusplus
}
#endif

#endif
#endif
