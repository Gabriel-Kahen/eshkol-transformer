#include "cpuv1_resource_private.h"
#ifdef ET_CPUV1_RESOURCE_TESTING
#include "eshkol_transformer/kernel_abi.h"
#include <string.h>

#define DECLARE(N) void et_cpuv1_test_census_##N##_v1(et_cpuv1_resource_packet_v1 *)
DECLARE(f32);DECLARE(i2);DECLARE(p1);DECLARE(c4);DECLARE(training);DECLARE(m3);
DECLARE(e3);DECLARE(inspector);DECLARE(o2);DECLARE(i64);DECLARE(t1);
DECLARE(generation);DECLARE(a2);DECLARE(d2);DECLARE(c2);DECLARE(c2_reader);DECLARE(io);
#undef DECLARE
void et_cpuv1_test_census_k1_v1(et_cpuv1_resource_packet_v1 *,et_kernel_runtime *const *,size_t);
et_kernel_runtime *et_cpuv1_test_k2_runtime_slot_v1(void);
et_kernel_runtime *et_cpuv1_test_c4_runtime_slot_v1(size_t);
et_kernel_runtime *et_cpuv1_test_training_runtime_slot_v1(size_t);

/* No caller/sidecar/training policy here. Native owner TUs alone walk their
 * actual registries. No test double can assert Scheme/runtime coverage. */
int et_cpuv1_test_native_resource_snapshot_v1(et_cpuv1_resource_packet_v1 *p) {
  if(!p || p->status!=ET_R_OK) return p?(int)p->status:ET_R_ARGUMENT;
#define OBSERVE(N) do {et_cpuv1_test_census_##N##_v1(p);if(p->status)return(int)p->status;}while(0)
  OBSERVE(f32);OBSERVE(i2);
  et_cpuv1_test_resource_domain_v1(p,ET_R_F32);
  OBSERVE(p1);et_cpuv1_test_resource_domain_v1(p,ET_R_P1);
  OBSERVE(c4);et_cpuv1_test_resource_domain_v1(p,ET_R_C4);
  OBSERVE(training);OBSERVE(m3);et_cpuv1_test_resource_domain_v1(p,ET_R_TRAIN);
  OBSERVE(e3);OBSERVE(inspector);et_cpuv1_test_resource_domain_v1(p,ET_R_E3);
  OBSERVE(o2);et_cpuv1_test_resource_domain_v1(p,ET_R_O2);
  OBSERVE(i64);OBSERVE(t1);et_cpuv1_test_resource_domain_v1(p,ET_R_I64_T1);
  OBSERVE(generation);OBSERVE(a2);et_cpuv1_test_resource_domain_v1(p,ET_R_GENERATION);
  OBSERVE(d2);et_cpuv1_test_resource_domain_v1(p,ET_R_D2);
  OBSERVE(c2);OBSERVE(c2_reader);OBSERVE(io);
  et_cpuv1_test_resource_domain_v1(p,ET_R_C2_IO);
#undef OBSERVE
  et_kernel_runtime *slots[7];
  slots[0]=et_cpuv1_test_k2_runtime_slot_v1();
  for(size_t i=0;i<3;i++) {
    slots[1+i]=et_cpuv1_test_c4_runtime_slot_v1(i);
    slots[4+i]=et_cpuv1_test_training_runtime_slot_v1(i);
  }
  et_cpuv1_test_census_k1_v1(p,slots,7u);
  return (int)p->status;
}
#endif
