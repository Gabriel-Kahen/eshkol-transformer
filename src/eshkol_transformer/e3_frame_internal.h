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
int64_t et_e3_private_counter_ref_v1(void *frame, int64_t selector);
int64_t et_e3_private_frame_destroy_preflight_v1(void *frame);
void et_e3_private_frame_destroy_v1(void *frame);

int64_t et_e3_private_last_error_domain_v1(void);
int64_t et_e3_private_last_error_category_v1(void);
int64_t et_e3_private_last_error_code_v1(void);

#ifdef __cplusplus
}
#endif

#endif
