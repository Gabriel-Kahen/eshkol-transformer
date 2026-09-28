#include "e1b_error_consumer_bridge.h"
#include <stdint.h>

extern eshkol_tagged_value_t tr3_step_begin(void)
    __asm__("tr3-metrics-step-begin-internal!");
extern eshkol_tagged_value_t tr3_step_fill(
    eshkol_tagged_value_t, eshkol_tagged_value_t, eshkol_tagged_value_t,
    eshkol_tagged_value_t, eshkol_tagged_value_t)
    __asm__("tr3-metrics-step-fill-internal!");
extern eshkol_tagged_value_t tr3_step_precommit(void)
    __asm__("tr3-metrics-step-precommit-internal!");
extern eshkol_tagged_value_t tr3_step_publish(void)
    __asm__("tr3-metrics-step-publish-internal!");
extern eshkol_tagged_value_t tr3_step_test_inject(eshkol_tagged_value_t)
    __asm__("tr3-step-composer-test-inject-internal!");
extern eshkol_tagged_value_t et_e1b_private_tr3_stop_policy_validate_cabi_v1(
    eshkol_tagged_value_t policy);
extern eshkol_tagged_value_t tr3_train_test_arm_after_one(
    eshkol_tagged_value_t enabled)
    __asm__("tr3-train-test-arm-after-one-internal!");
extern eshkol_tagged_value_t tr3_stop_policy_root
    __asm__("tr3-stop-policy-root");
extern eshkol_tagged_value_t tr3_metrics_root
    __asm__("tr3-metrics");
extern eshkol_tagged_value_t tr3_step_test_hook
    __asm__("tr3-step-composer-test-hook");

/* Test-only producer, linked with the unlocalized copy of the same aggregate. */
void et_tr3_test_publish_step_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  (void)tr3_step_begin();
  (void)tr3_step_fill(eshkol_make_int64(INT64_C(0x3eaaaaab), true),
                      eshkol_make_int64(INT64_C(0x3f800000), true),
                      eshkol_make_int64(7, true),
                      eshkol_make_int64(2, true),
                      eshkol_make_int64(3, true));
  (void)tr3_step_precommit();
  *et_e1b_box_value_v1(output) = tr3_step_publish();
}

int64_t et_tr3_test_f32_bits_v1(void *input) {
  uint32_t bits = 0;
  eshkol_tagged_value_t value = *et_e1b_box_value_v1(input);
  if (eshkol_value_is_f32_v1(&value) != 1 ||
      eshkol_value_f32_to_bits_v1(&value, &bits) != ESHKOL_VALUE_F32_OK)
    return -1;
  return (int64_t)bits;
}

void et_tr3_test_set_step_fault_v1(void *cut, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      tr3_step_test_inject(*et_e1b_box_value_v1(cut));
}

void et_tr3_test_validate_stop_policy_v1(void *policy, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_tr3_stop_policy_validate_cabi_v1(
          *et_e1b_box_value_v1(policy));
}

void et_tr3_test_stop_policy_root_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = tr3_stop_policy_root;
}

void et_tr3_test_metrics_root_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = tr3_metrics_root;
}

void et_tr3_test_set_observation_hook_v1(void *callback) {
  et_e1b_ensure_private_initialized_v1();
  tr3_step_test_hook = *et_e1b_box_value_v1(callback);
}

void et_tr3_test_arm_train_after_one_v1(void *enabled, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      tr3_train_test_arm_after_one(*et_e1b_box_value_v1(enabled));
}
