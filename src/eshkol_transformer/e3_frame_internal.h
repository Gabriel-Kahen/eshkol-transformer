#ifndef ET_E3_FRAME_INTERNAL_H
#define ET_E3_FRAME_INTERNAL_H

#include <stdint.h>

#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/kernel_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct et_e3_frame_internal et_e3_frame_internal;

void *et_e3_private_frame_create_v1(
    void *model, et_f32_tensor *loss, et_f32_tensor *weight,
    et_f32_tensor *perplexity, et_f32_tensor *accuracy);
int64_t et_e3_private_frame_check_v1(void *frame);
int64_t et_e3_private_acquire_v1(void *frame);
int64_t et_e3_private_stage_inputs_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
int64_t et_e3_private_stage_targets_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
int64_t et_e3_private_stage_mask_v1(
    void *frame, const et_kernel_tensor_view_v1 *view);
int64_t et_e3_private_forward_role_v1(void *frame, int64_t role);
int64_t et_e3_private_ce_v1(void *frame);
int64_t et_e3_private_reduce_v1(void *frame);
int64_t et_e3_private_correct_v1(void *frame);
int64_t et_e3_private_accumulate_v1(void *frame);
int64_t et_e3_private_finalize_v1(void *frame);
int64_t et_e3_private_cleanup_preflight_v1(void *frame);
void et_e3_private_drain_v1(void *frame);
int64_t et_e3_private_publish_v1(void *frame);
void et_e3_private_finish_v1(void *frame);
int64_t et_e3_private_selected_metric_bits_ref_v1(
    void *frame, int64_t selector);
int64_t et_e3_private_counter_ref_v1(void *frame, int64_t selector);
int64_t et_e3_private_frame_destroy_preflight_v1(void *frame);
void et_e3_private_frame_destroy_v1(void *frame);

int64_t et_e3_private_last_error_domain_v1(void);
int64_t et_e3_private_last_error_category_v1(void);
int64_t et_e3_private_last_error_code_v1(void);

#ifdef ET_E3_TESTING
void et_e3_test_frame_fail_alloc_after_v1(size_t count);
#endif

#ifdef ET_E3_FAULT_TESTING
/* Native-only test seam. Status selectors: state, role, occurrence, reached,
 * fired (0..4). State: clear=0, armed=1, entered=2, fired=3, missed=4. */
int64_t et_e3_test_forward_fault_arm_v1(
    void *frame, int64_t role, int64_t occurrence);
/* The test driver brackets exactly one e3-evaluate-into! attempt with begin
 * and end, including rejected attempts. Neither frame check is an entry hook. */
int64_t et_e3_test_forward_fault_begin_v1(void *frame);
int64_t et_e3_test_forward_fault_end_v1(void *frame);
int64_t et_e3_test_forward_fault_status_v1(void *frame, int64_t selector);
int64_t et_e3_test_forward_fault_reset_v1(void *frame);
/* Read retained published counts even after a failed acquired invocation
 * clears the public committed bit. Idle authenticated frame only. */
int64_t et_e3_test_fault_counter_raw_v1(void *frame, int64_t selector);
#endif

#ifdef __cplusplus
}
#endif

#endif
