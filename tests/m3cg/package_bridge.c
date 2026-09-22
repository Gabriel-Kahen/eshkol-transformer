/* Development-only runner; all other test seams are localized. */
#include "../../native/m3_package_bridge.c"
extern eshkol_tagged_value_t et_m3cg_test_run_cabi_v1(void);
int64_t et_m3cg_test_run_v1(void) {
  et_e1b_ensure_private_initialized_v1();
  eshkol_tagged_value_t result = et_m3cg_test_run_cabi_v1();
  (void)result;
  return 1;
}
