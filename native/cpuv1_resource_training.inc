#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_training_v1(et_cpuv1_resource_packet_v1 *packet) {
  for (c4t_record *p=c4t_registry;p;p=p->next) {
    size_t size,alignment; unsigned kind;
    if (p->kind==C4T_INPUT) { kind=0;size=sizeof(c4t_input);alignment=_Alignof(c4t_input); }
    else if (p->kind==C4T_WORKSPACE) { kind=1;size=sizeof(c4t_workspace);alignment=_Alignof(c4t_workspace); }
    else if (p->kind==C4T_LOGITS) { kind=2;size=sizeof(c4t_logits);alignment=_Alignof(c4t_logits); }
    else { et_cpuv1_test_resource_fail_v1(packet,ET_R_COVERAGE); return; }
    if (p->extent!=size || p->state<C4T_FORWARD || p->state>C4T_DEAD) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_TRAIN,kind,p,1u,size,
           p->state==C4T_DEAD?ET_R_DEAD:ET_R_BUSY,alignment)) return;
    if (p->state!=C4T_DEAD) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
}
et_kernel_runtime *et_cpuv1_test_training_runtime_slot_v1(size_t index) {
  return index<3?c4t_runtimes[index]:NULL;
}
#endif
