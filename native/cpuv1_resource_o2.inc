#ifdef ET_CPUV1_RESOURCE_TESTING
#if !defined(ET_TR3_O2_STEP_CLEAR_NATIVE) || !defined(ET_TR3_C_O2_RESTORE_NATIVE)
#error "CPUv1 O2 census needs both original private native adapters"
#endif
#define O_BASE(K,B,N,S,STATE,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_O2,K,B,N,S,STATE,A)) return; \
} while (0)
#define O_CONTROL(K,B,STATE) O_BASE(K,B,1u,sizeof(*(B)),STATE,_Alignof(__typeof__(*(B))))
#define O_ARRAY(K,B,N,STATE) O_BASE(K,B,N,sizeof(*(B)),STATE,_Alignof(__typeof__(*(B))))
void et_cpuv1_test_census_o2_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,0,sizeof(et_o2_optimizer_builder),_Alignof(et_o2_optimizer_builder));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,1,sizeof(et_o2_optimizer),_Alignof(et_o2_optimizer));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,2,sizeof(et_o2_optimizer_state),_Alignof(et_o2_optimizer_state));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,3,sizeof(et_o2_optimizer_state_handle),_Alignof(et_o2_optimizer_state_handle));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,4,sizeof(et_o2_optimizer_state_borrow),_Alignof(et_o2_optimizer_state_borrow));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,5,sizeof(et_o2_state_reconstruct_builder),_Alignof(et_o2_state_reconstruct_builder));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,7,sizeof(et_o2_entry),_Alignof(et_o2_entry));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,8,sizeof(et_o2_state_entry),_Alignof(et_o2_state_entry));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,9,sizeof(et_o2_optimizer_state_handle *),_Alignof(et_o2_optimizer_state_handle *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,10,sizeof(et_f32_tensor *),_Alignof(et_f32_tensor *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,11,sizeof(et_f32_tensor *),_Alignof(et_f32_tensor *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,12,sizeof(et_f32_tensor_copy_assignment_v1),_Alignof(et_f32_tensor_copy_assignment_v1));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,13,sizeof(et_f32_parameter *),_Alignof(et_f32_parameter *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,14,sizeof(et_o2_trainer_step_clear_plan),_Alignof(et_o2_trainer_step_clear_plan));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,15,sizeof(et_f32_tensor *),_Alignof(et_f32_tensor *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,16,sizeof(et_f32_tensor_copy_assignment_v1),_Alignof(et_f32_tensor_copy_assignment_v1));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,17,sizeof(et_f32_parameter *),_Alignof(et_f32_parameter *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,18,sizeof(et_o2_state_entry),_Alignof(et_o2_state_entry));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,19,sizeof(et_o2_tr3_c_restore_plan),_Alignof(et_o2_tr3_c_restore_plan));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,20,sizeof(et_f32_tensor *),_Alignof(et_f32_tensor *));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,21,sizeof(unsigned char),_Alignof(unsigned char));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_O2,6,1u,_Alignof(et_o2_retire_provenance));
  packet->details[ET_R_O2][32]=sizeof(et_o2_retire_provenance);

  for (et_o2_optimizer_builder *p=et_o2_builders;p;p=p->registry_next) {
    O_CONTROL(0,p,ET_R_PREPARED);
    O_ARRAY(7,p->entries,p->count,ET_R_PREPARED);
    et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
  }
  for (et_o2_optimizer *p=et_o2_optimizers;p;p=p->registry_next) {
    int retired=p->magic==ET_O2_RETIRED_OPTIMIZER_MAGIC;
    O_CONTROL(1,p,retired?ET_R_DEAD:ET_R_LIVE);
    if ((!retired && p->magic!=ET_O2_OPTIMIZER_MAGIC) ||
        p->count>ET_O2_MAX_PARAMETERS) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    O_ARRAY(7,p->entries,p->count,ET_R_LIVE);
    if (p->busy || p->active_operation) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
    packet->details[ET_R_O2][retired?1:0]++;
  }
  for (et_o2_optimizer_state *p=et_o2_states;p;p=p->registry_next) {
    int dead=p->lifecycle==ET_O2_OPTIMIZER_STATE_DEAD;
    O_CONTROL(2,p,dead?ET_R_DEAD:ET_R_LIVE);
    if (p->count>ET_O2_MAX_PARAMETERS ||
        (p->lifecycle!=ET_O2_OPTIMIZER_STATE_LIVE && !dead)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
    O_ARRAY(8,p->entries,p->count,ET_R_LIVE);
    O_ARRAY(9,p->handles,p->count*2u,ET_R_LIVE);
    O_ARRAY(10,p->owned_clones,p->count*2u,ET_R_LIVE);
    packet->details[ET_R_O2][dead?3:2]++;
    if (p->active_borrows) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
  for (et_o2_optimizer_state_handle *p=et_o2_handles;p;p=p->registry_next)
    O_CONTROL(3,p,p->live?ET_R_LIVE:ET_R_DEAD);
  for (et_o2_optimizer_state_borrow *p=et_o2_borrows;p;p=p->registry_next) {
    O_CONTROL(4,p,ET_R_BUSY);
    et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
  }
  for (et_o2_optimizer_state_borrow *p=et_o2_retired_borrows;p;p=p->registry_next)
    O_CONTROL(4,p,ET_R_DEAD);
  for (et_o2_state_reconstruct_builder *p=et_o2_reconstruct_builders;p;p=p->registry_next) {
    O_CONTROL(5,p,ET_R_PREPARED);
    O_ARRAY(18,p->entries,p->count,ET_R_PREPARED);
    O_ARRAY(20,p->owned_clones,p->count*2u,ET_R_PREPARED);
    O_ARRAY(21,p->entry_set,p->count,ET_R_PREPARED);
    /* Prepared unpublished state/handles are intentionally not accepted at
     * a resource snapshot. The constructor must return/abort before observe. */
    et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
  }
  for (et_o2_retire_provenance *p=et_o2_retire_records;p;p=p->next) {
    if (p->count>ET_O2_MAX_PARAMETERS ||
        p->count>(SIZE_MAX-sizeof(*p))/sizeof(et_o2_entry)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_OVERFLOW); return;
    }
    size_t bytes=sizeof(*p)+p->count*sizeof(et_o2_entry);
    O_BASE(6,p,bytes,1u,p->live?ET_R_LIVE:ET_R_DEAD,
           _Alignof(et_o2_retire_provenance));
    packet->details[ET_R_O2][32]=sizeof(*p);
  }
  for (et_o2_trainer_step_clear_plan *p=et_o2_tr3_step_clear_plans;p;p=p->registry_next) {
    unsigned phase=p->lifecycle;
    if (phase<ET_O2_TR3_STEP_CLEAR_PREPARED || phase>ET_O2_TR3_STEP_CLEAR_ABORTED ||
        p->count>ET_O2_MAX_PARAMETERS) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    int terminal=phase==ET_O2_TR3_STEP_CLEAR_COMMITTED || phase==ET_O2_TR3_STEP_CLEAR_ABORTED;
    O_CONTROL(14,p,terminal?ET_R_DEAD:ET_R_PREPARED);
    O_ARRAY(15,p->stages,p->count*3u,ET_R_PREPARED);
    O_ARRAY(16,p->assignments,p->count*3u,ET_R_PREPARED);
    O_ARRAY(17,p->parameters,p->count,ET_R_PREPARED);
    packet->details[ET_R_O2][8+phase]++;
    if (!terminal) { et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return; }
    if (p->stages || p->assignments || p->parameters || p->copy_plan || p->reset_plan) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return;
    }
  }
  for (et_o2_tr3_c_restore_plan *p=et_o2_tr3_c_restore_plans;p;p=p->registry_next) {
    unsigned phase=p->lifecycle;
    if (phase<ET_O2_TR3_C_RESTORE_PREPARED || phase>ET_O2_TR3_C_RESTORE_ABORTED) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return;
    }
    int terminal=phase==ET_O2_TR3_C_RESTORE_COMMITTED || phase==ET_O2_TR3_C_RESTORE_ABORTED;
    O_CONTROL(19,p,terminal?ET_R_DEAD:ET_R_PREPARED);
    packet->details[ET_R_O2][16+phase]++;
    if (!terminal) { et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return; }
  }
}
#undef O_BASE
#undef O_CONTROL
#undef O_ARRAY
#endif
