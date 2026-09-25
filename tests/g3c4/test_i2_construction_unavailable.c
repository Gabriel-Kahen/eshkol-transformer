#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_I2_CONSTRUCTION_PRIVATE 1
#define ET_I2_NATIVE_HELPERS_ONLY 1
#include "../../src/eshkol_transformer/m3t_f32_integration.c"
#include "../../native/i2_wave2_package_bridge.c"
#include <stdio.h>
#define CHECK(x) do { if(!(x)) return 1; } while(0)
int main(void) {
  memset(&et_i2_last_error,0x5a,sizeof(et_i2_last_error));
  et_f32_tensor_error before=et_i2_last_error;
  CHECK(et_i2_private_g3c4_construction_available_v1()==-1);
  CHECK(et_i2_private_g3c4_construction_parameter_preflight_v1(
      (void *)(uintptr_t)1,(void *)(uintptr_t)2,(void *)(uintptr_t)3)==-1);
  CHECK(!memcmp(&before,&et_i2_last_error,sizeof(before)));
  puts("G3-C4 I2 construction bridge unavailable: error unchanged");
  return 0;
}
