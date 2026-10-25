#include "e1b_error_consumer_bridge.h"
#include <stdint.h>

extern eshkol_tagged_value_t et_e1b_private_g3r_trained_transcript_cabi_v1(
    eshkol_tagged_value_t arguments);
extern eshkol_tagged_value_t et_e1b_private_g3r_transcript_decode_check_cabi_v1(
    eshkol_tagged_value_t arguments);

void et_g3r_private_test_trained_transcript_v1(void *arguments,
                                                void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_g3r_trained_transcript_cabi_v1(
          *et_e1b_box_value_v1(arguments));
}

int64_t et_g3r_private_transcript_host_le_v1(void) {
  const uint32_t one = 1u;
  return *(const uint8_t *)&one == 1u ? 1 : 0;
}

void et_g3r_private_test_transcript_decode_check_v1(void *arguments,
                                                     void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_g3r_transcript_decode_check_cabi_v1(
          *et_e1b_box_value_v1(arguments));
}
