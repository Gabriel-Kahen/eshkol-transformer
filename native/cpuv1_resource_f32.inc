/* Included only inside the F32 owning TU, after its canonical registries. */
#ifdef ET_CPUV1_RESOURCE_TESTING
#define R_BASE(K,B,N,S,STATE,A) do { \
  if (et_cpuv1_test_resource_add_v1(packet,ET_R_F32,K,B,N,S,STATE,A)) return; \
} while (0)
#define R_CONTROL(K,B,STATE) R_BASE(K,B,1u,sizeof(*(B)),STATE,_Alignof(__typeof__(*(B))))
void et_cpuv1_test_census_f32_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,0,sizeof(et_f32_tensor),_Alignof(et_f32_tensor));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,1,sizeof(et_f32_tensor_borrow),_Alignof(et_f32_tensor_borrow));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,2,sizeof(et_f32_tensor_copy_plan),_Alignof(et_f32_tensor_copy_plan));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,3,sizeof(et_f32_parameter),_Alignof(et_f32_parameter));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,4,sizeof(et_f32_gradient_plan),_Alignof(et_f32_gradient_plan));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,5,sizeof(et_f32_gradient_reset_plan),_Alignof(et_f32_gradient_reset_plan));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,6,sizeof(float),_Alignof(float));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,7,sizeof(uint64_t),_Alignof(uint64_t));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,8,sizeof(size_t),_Alignof(size_t));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,9,sizeof(et_f32_tensor_copy_assignment_v1),_Alignof(et_f32_tensor_copy_assignment_v1));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,10,sizeof(et_f32_gradient_plan_entry),_Alignof(et_f32_gradient_plan_entry));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,11,sizeof(float),_Alignof(float));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,12,sizeof(et_f32_parameter *),_Alignof(et_f32_parameter *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,13,sizeof(f32_destroy_provenance),_Alignof(f32_destroy_provenance));

#ifndef ET_I2_PRIVATE_OWNED_CLONE_MATCH
#error "CPUv1 resource F32 census requires the original canonical-layout check"
#endif
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
  /* Frozen producer is history/capture/save-witness OFF. */
  et_cpuv1_test_resource_fail_v1(packet,ET_R_UNSUPPORTED); return;
#endif
  for (et_f32_tensor *t=live_tensors;t;t=t->registry_next) {
    if (t->magic!=ET_F32_TENSOR_MAGIC || t->rank>ET_KERNEL_MAX_RANK ||
        t->element_count>SIZE_MAX/sizeof(float) ||
        t->byte_length!=t->element_count*sizeof(float) ||
        !et_i2_private_tensor_is_canonical(t)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    R_CONTROL(0,t,ET_R_LIVE);
    R_BASE(6,t->data,t->element_count,sizeof(float),ET_R_LIVE,_Alignof(float));
    R_BASE(7,t->shape,t->rank,sizeof(*t->shape),ET_R_LIVE,_Alignof(uint64_t));
    R_BASE(8,t->strides,t->rank,sizeof(*t->strides),ET_R_LIVE,_Alignof(size_t));
  }
  for (et_f32_parameter *p=live_parameters;p;p=p->registry_next)
    R_CONTROL(3,p,ET_R_LIVE);
  for (et_f32_tensor_borrow *b=live_borrows;b;b=b->registry_next)
    R_CONTROL(1,b,ET_R_LIVE);
  for (et_f32_tensor_copy_plan *p=live_copy_plans;p;p=p->registry_next) {
    R_CONTROL(2,p,ET_R_PREPARED);
    R_BASE(9,p->assignments,p->count,sizeof(*p->assignments),ET_R_PREPARED,
           _Alignof(et_f32_tensor_copy_assignment_v1));
  }
  for (et_f32_gradient_plan *p=live_gradient_plans;p;p=p->registry_next) {
    R_CONTROL(4,p,ET_R_PREPARED);
    R_BASE(10,p->entries,p->count,sizeof(*p->entries),ET_R_PREPARED,
           _Alignof(et_f32_gradient_plan_entry));
    for (size_t i=0;i<p->count;i++) {
      if (!p->entries[i].parameter || !p->entries[i].parameter->gradient) {
        et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
      }
      R_BASE(11,p->entries[i].prepared,
        p->entries[i].parameter->gradient->element_count,sizeof(float),
        ET_R_PREPARED,_Alignof(float));
    }
  }
  for (et_f32_gradient_reset_plan *p=live_reset_plans;p;p=p->registry_next) {
    R_CONTROL(5,p,ET_R_PREPARED);
    R_BASE(12,p->parameters,p->count,sizeof(*p->parameters),ET_R_PREPARED,
           _Alignof(et_f32_parameter *));
  }
#define R_RETIRED(TYPE,HEAD,K) \
  for (TYPE *p=HEAD;p;p=p->registry_next) R_CONTROL(K,p,ET_R_DEAD)
  R_RETIRED(et_f32_tensor,retired_tensors,0);
  R_RETIRED(et_f32_tensor_borrow,retired_borrows,1);
  R_RETIRED(et_f32_tensor_copy_plan,retired_copy_plans,2);
  R_RETIRED(et_f32_parameter,retired_parameters,3);
  R_RETIRED(et_f32_gradient_plan,retired_gradient_plans,4);
  R_RETIRED(et_f32_gradient_reset_plan,retired_reset_plans,5);
#undef R_RETIRED
#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE
  for (f32_destroy_provenance *p=f32_destroy_records;p;p=p->next)
    R_CONTROL(13,p,ET_R_LIVE);
#endif
}
#undef R_BASE
#undef R_CONTROL
#endif
