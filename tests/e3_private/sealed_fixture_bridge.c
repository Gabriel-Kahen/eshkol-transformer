/* Boxed test-only entry. This tuple is never installed or public. */
#include "../../native/e3_private_bridge.c"

extern eshkol_tagged_value_t et_e3_private_g3r_fixture_run_cabi_v1(void);

void et_e3_sealed_fixture_run_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_g3r_fixture_run_cabi_v1();
}
