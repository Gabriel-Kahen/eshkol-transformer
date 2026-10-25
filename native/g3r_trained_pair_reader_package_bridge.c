#include "e1b_error_consumer_bridge.h"

extern eshkol_tagged_value_t et_e1b_private_g3r_trained_pair_read_cabi_v1(
    eshkol_tagged_value_t arguments);

/* Reader and its boxes execute in the one private CLI3/G3-R package. */
void et_g3r_private_test_trained_pair_read_v1(void *arguments, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e1b_private_g3r_trained_pair_read_cabi_v1(
          *et_e1b_box_value_v1(arguments));
}
