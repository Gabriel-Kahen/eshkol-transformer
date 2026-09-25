#ifndef ESHKOL_TRANSFORMER_TR3_C_RESTORE_BINDINGS_H
#define ESHKOL_TRANSFORMER_TR3_C_RESTORE_BINDINGS_H

#include <stdint.h>

#if !defined(ET_TR3_C_RESTORE_BINDINGS)
#error "TR3-C source bindings require the private binding feature"
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { ET_TR3_C_RESTORE_REQUEST_BYTES = 728 };

int64_t et_tr3_c_private_restore_last_category_v1(void);
int64_t et_tr3_c_private_restore_request_initialize_v1(
    void *request_bytes, int64_t target_completed_updates,
    int64_t expected_receiver_completed_updates, int64_t clip_kind,
    int64_t clip_max_bits, int64_t schedule_kind,
    int64_t minimum_ratio_bits, int64_t warmup_updates,
    int64_t total_updates);
int64_t et_tr3_c_private_restore_request_append14_v1(
    void *request_bytes, int64_t learning_rate_bits,
    int64_t beta1_bits, int64_t beta2_bits, int64_t epsilon_bits,
    int64_t weight_decay_bits, void *exp_avg_owned,
    void *exp_avg_sq_owned);

void *et_tr3_c_private_o2_restore_prepare_v1(void *optimizer,
                                             void *request_bytes);
int64_t et_tr3_c_private_o2_restore_append_v1(void *plan, void *builder);
int64_t et_tr3_c_private_o2_restore_check_v1(void *plan);
void et_tr3_c_private_o2_restore_commit_native_v1(void *plan);
void et_tr3_c_private_o2_restore_finalize_v1(void *plan);
int64_t et_tr3_c_private_o2_restore_abort_v1(void *plan);

void *et_tr3_c_private_i2_restore_create_v1(void *exact_o2_plan);
int64_t et_tr3_c_private_i2_restore_append_parameter_v1(
    void *builder, void *exact_o2_plan, void *parameter, void *p1_handle,
    void *owned_source);
int64_t et_tr3_c_private_i2_restore_prepare_v1(void *builder,
                                               void *exact_o2_plan);
int64_t et_tr3_c_private_i2_restore_commit_checked_v1(
    void *builder, void *exact_o2_plan);
int64_t et_tr3_c_private_i2_restore_abort_checked_v1(
    void *builder, void *exact_o2_plan);
void et_tr3_c_private_i2_owned_release_checked_v1(void *owned);

#ifdef __cplusplus
}
#endif

#endif
