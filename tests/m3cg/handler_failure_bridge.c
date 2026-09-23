/* One inherited bridge/init lineage; this runner exists only in this fixture. */
#include "package_bridge.c"
extern eshkol_tagged_value_t et_m3cg_handler_run_cabi_v1(void);
int64_t et_m3cg_handler_run_v1(void) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_m3cg_handler_run_cabi_v1();
  return 1;
}
