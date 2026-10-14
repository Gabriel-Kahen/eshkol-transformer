/* Standalone native state-machine probe. Compile with function sections and
 * link garbage collection; no E3 package or provider is linked by this test. */
#define ET_E3_FAULT_TESTING 1
#include "../../src/eshkol_transformer/e3_frame.c"

#include <assert.h>

int main(void) {
  owner model = {0};
  et_e3_frame_internal frame = {0};
  enroll(&model.r, OWNER);
  frame.lifecycle = 1;
  frame.phase = E3_IDLE;
  frame.model = &model;
  e3_registry = &frame;

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
  e3_fault_enter(&frame);
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

  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 2) == 0);
  e3_fault_enter(&frame);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  e3_fault_enter(&frame);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_MISSED);
  assert(et_e3_test_forward_fault_status_v1(&frame, 3) == 1);
  assert(et_e3_test_forward_fault_status_v1(&frame, 4) == 0);
  assert(e3_fault_forward_preinvoke(&frame, 0) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);

  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  assert(et_e3_test_forward_fault_arm_v1(&frame, 20, 8192) == 0);
  frame.phase = E3_FORWARD;
  model.active = &frame;
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_reset_v1(&frame) == E3_INVALID_STATE);
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == -1);
  frame.phase = E3_IDLE;
  model.active = NULL;
  assert(et_e3_test_forward_fault_status_v1(&frame, 0) == E3_FAULT_ARMED);
  assert(et_e3_test_forward_fault_reset_v1(&frame) == 0);
  frame.lifecycle = 2;
  assert(et_e3_test_forward_fault_arm_v1(&frame, 0, 1) == E3_INVALID_STATE);
  return 0;
}
