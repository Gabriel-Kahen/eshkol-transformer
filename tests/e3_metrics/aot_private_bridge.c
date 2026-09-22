/* Inventoried test-only reachability; never part of the production archive. */
#define main et_test_e3_metrics_native
#include "native_test.c"
#undef main

int et_test_e3_metrics_composition(void);

int64_t et_e3_metrics_test_aot_bridge_v1(void) {
  if (et_test_e3_metrics_native() != 0) return 0;
  return et_test_e3_metrics_composition() == 0 ? 1 : 0;
}

#ifdef ET_E3_METRICS_AOT_BRIDGE_STANDALONE
int main(void) { return et_e3_metrics_test_aot_bridge_v1() == 1 ? 0 : 1; }
#endif
