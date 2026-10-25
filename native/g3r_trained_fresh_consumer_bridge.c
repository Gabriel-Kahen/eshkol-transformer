#include <stdint.h>
#include "e1b_error_consumer_bridge.h"

extern eshkol_tagged_value_t et_e1b_private_g3r_fresh_test_run_cabi_v1(
    eshkol_tagged_value_t arguments);

void et_g3r_private_test_fresh_consume_v1(void *arguments, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_g3r_fresh_test_run_cabi_v1(
          *et_e1b_box_value_v1(arguments));
}

/* Physical I2 parameter copies are compared to little-endian V1 words. */
int64_t et_g3r_private_fresh_host_le_v1(void) {
  const uint32_t one = 1u;
  return *(const uint8_t *)&one == 1u ? 1 : 0;
}
