#ifdef ET_CPUV1_RESOURCE_TESTING
#define I_BASE(K,B,N,S,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_I64_T1,K,B,N,S,ET_R_LIVE,A)) return; \
} while (0)
void et_cpuv1_test_census_i64_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_I64_T1,0,sizeof(et_i64_tensor),_Alignof(et_i64_tensor));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_I64_T1,1,sizeof(et_i64_tensor_borrow),_Alignof(et_i64_tensor_borrow));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_I64_T1,2,sizeof(int64_t),_Alignof(int64_t));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_I64_T1,3,sizeof(uint64_t),_Alignof(uint64_t));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_I64_T1,4,sizeof(size_t),_Alignof(size_t));

  for (et_i64_tensor *p=live_tensors;p;p=p->registry_next) {
    if (p->magic!=ET_I64_TENSOR_MAGIC || p->rank>ET_KERNEL_MAX_RANK ||
        p->element_count>SIZE_MAX/sizeof(int64_t) ||
        p->byte_length!=p->element_count*sizeof(int64_t)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    I_BASE(0,p,1u,sizeof(*p),_Alignof(et_i64_tensor));
    I_BASE(2,p->data,p->element_count,sizeof(*p->data),_Alignof(int64_t));
    I_BASE(3,p->shape,p->rank,sizeof(*p->shape),_Alignof(uint64_t));
    I_BASE(4,p->strides,p->rank,sizeof(*p->strides),_Alignof(size_t));
  }
  for (et_i64_tensor_borrow *p=live_borrows;p;p=p->registry_next)
    I_BASE(1,p,1u,sizeof(*p),_Alignof(et_i64_tensor_borrow));
}
#undef I_BASE
#endif
