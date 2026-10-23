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
int64_t et_cpuv1_test_resource_open_v1(void *raw) {
  et_cpuv1_resource_packet_v1 *p=raw;
  int rc=et_cpuv1_test_resource_begin_v1(p);
  return rc?rc:et_cpuv1_test_native_resource_snapshot_v1(p);
}
int64_t et_cpuv1_test_resource_close_v1(void *raw) {
  et_cpuv1_resource_packet_v1 *p=raw;
  et_cpuv1_test_runtime_resource_snapshot_v1(p);
  return et_cpuv1_test_resource_finish_v1(p);
}
int64_t et_cpuv1_test_resource_warm_v1(void) {
  return et_cpuv1_test_runtime_resource_warm_v1();
}
int64_t et_cpuv1_test_resource_pair_v1(void *left,void *right) {
  const et_cpuv1_resource_packet_v1 *a=left,*b=right;
  if(!a || !b || a==b || a->status || b->status ||
     a->coverage!=(UINT64_C(1)<<ET_R_DOMAINS)-1 || a->coverage!=b->coverage)
    return 0;
  return memcmp(a,b,sizeof(*a))==0;
}
int64_t et_cpuv1_test_resource_word_v1(void *raw,int64_t group,int64_t index) {
  const et_cpuv1_resource_packet_v1 *p=raw;
  if(!p || index<0 || index>=64)return -1;
  if(group==0)return (int64_t)p->runtime[index];
  if(group==1)return (int64_t)p->details[ET_R_SCHEME][index];
  if(group==2)return (int64_t)p->rows[ET_R_SCHEME][index].live;
  if(group==3)return (int64_t)p->details[ET_R_C4][index];
  if(group==4 && index<24)return (int64_t)p->scheme_p1[index];
  if(group>=5 && group<13)return (int64_t)p->scheme_states[index][group-5];
  if(group==13)return (int64_t)p->scheme_scalar[index];
  if(group==14 && index<8)return (int64_t)p->scheme_candidate[index];
  if(group>=15 && group<31)return (int64_t)p->scheme_source[index][group-15];
  if(group==31 && index<16)return (int64_t)p->scheme_source[64][index];
  return -1;
}
#endif
