/* Standalone native state-machine probe. Compile with function sections and
 * link garbage collection; no E3 package or provider is linked by this test. */
#define ET_E3_FAULT_TESTING 1
#include "../../src/eshkol_transformer/e3_frame.c"

#include <assert.h>

/* The fake frame must fail storage admission at its first null tensor. If
 * admission advances into any provider/tensor dependency, this probe fails. */
const et_f32_tensor *et_f32_tensor_canonical_owner_v1(const et_f32_tensor *p) {
  (void)p;
  abort();
}
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  (void)pins; (void)error;
  abort();
}
int32_t et_f32_tensor_scoped_begin_internal(
    et_f32_tensor *tensor, et_f32_scoped_guard_internal *guard,
    et_f32_tensor_error *error) {
  (void)tensor; (void)guard; (void)error;
  abort();
}
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *guard) {
  (void)guard;
  abort();
}

int main(void) {
  owner model = {0};
  et_e3_frame_internal frame = {0};
  enroll(&model.r, OWNER);
  frame.lifecycle = 1;
  frame.phase = E3_IDLE;
  frame.model = &model;
  e3_registry = &frame;

  assert(et_e3_test_forward_fault_begin_v1(NULL) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_end_v1(NULL) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_end_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_arm_v1(NULL, 1, 1) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_arm_v1(&frame, -1, 1) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 21, 1) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 0) == E3_INVALID_ARGUMENT);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 8193) == E3_INVALID_ARGUMENT);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  assert(et_e3_test_forward_fault_status_v1(&frame, 3) == 0);

  assert(et_e3_test_forward_fault_arm_v1(&frame, 1, 1) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 1, 1) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_ARMED);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == E3_INVALID_STATE);
  /* The real wrapper checks at stage 0, then D2 prepare records the frame
   * and checks it again at stage 2. This fake frame fails storage admission;
   * the probe verifies both checks leave the selector scoped to one attempt. */
  assert(et_e3_private_frame_check_v1(&frame) == E3_INTERNAL);
  assert(et_e3_private_frame_check_v1(&frame) == E3_INTERNAL);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_ENTERED);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  assert(e3_fault_forward_preinvoke(&frame, 1) == E3_INVALID_STATE);
  assert(et_e3_private_last_error_domain_v1() == 0);
  assert(et_e3_private_last_error_category_v1() == E3_INVALID_STATE);
  assert(et_e3_private_last_error_code_v1() == E3_CODE_PHASE);
  assert(e3_fault_forward_preinvoke(&frame, 1) == 0);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_FIRED);
  assert(et_e3_test_forward_fault_status_v1(&frame, 1) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 2) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 3) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 4) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 5) == -1);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_end_v1(&frame) == 0);

  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 2) == 0);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == 0);
  assert(et_e3_private_frame_check_v1(&frame) == E3_INTERNAL);
  assert(et_e3_private_frame_check_v1(&frame) == E3_INTERNAL);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  assert(et_e3_test_forward_fault_end_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_MISSED);
  assert(et_e3_test_forward_fault_status_v1(&frame, 3) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 4) == 0);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);

  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 3, 1) == 0);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == 0);
  assert(et_e3_private_frame_check_v1(&frame) == E3_INTERNAL);
  /* An early P1 rejection has no second frame check. The test driver must
   * close the attempt before a same-frame retry. */
  assert(et_e3_test_forward_fault_end_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_MISSED);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == E3_INVALID_STATE);
  assert(e3_fault_forward_preinvoke(&frame, 3) == 0);

  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 20, 8192) == 0);
  frame.phase = E3_FORWARD;
  model.active = &frame;
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_end_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_reset_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == -1);
  frame.phase = E3_IDLE;
  model.active = NULL;
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_ARMED);
  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  frame.lifecycle = 2;
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_begin_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_end_v1(&frame) == E3_INVALID_STATE);
  return 0;
}
