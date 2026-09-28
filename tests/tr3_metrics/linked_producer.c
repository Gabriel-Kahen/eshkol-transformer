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
