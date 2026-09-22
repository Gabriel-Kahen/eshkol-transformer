/* Private test translation unit: no installed owner, graph or model transport. */
#define ET_L3S_NO_MAIN 1
#include "test_i2_integration.c"
int64_t et_l3s_test_aot_bridge_v1(void) {
  return et_l3s_test_owned_composition_v1();
}
#ifdef ET_L3S_AOT_BRIDGE_TESTING
int main(void) { return et_l3s_test_aot_bridge_v1()==1?0:1; }
#endif
