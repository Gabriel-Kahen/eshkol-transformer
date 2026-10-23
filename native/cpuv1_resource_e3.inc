#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_e3_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_E3,0,sizeof(et_e3_frame_internal),_Alignof(et_e3_frame_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_E3,1,sizeof(e3_cpu4_workspace),_Alignof(e3_cpu4_workspace));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_E3,2,sizeof(e3_cpu4_payload),_Alignof(e3_cpu4_payload));

  for (et_e3_frame_internal *p=e3_registry;p;p=p->next) {
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_E3,0,p,1u,sizeof(*p),
         p->lifecycle==2?ET_R_DEAD:ET_R_LIVE,_Alignof(et_e3_frame_internal))) return;
    if (p->phase!=E3_IDLE) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
  for (e3_cpu4_workspace *p=e3_workspaces;p;p=p->next) {
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_E3,1,p,1u,sizeof(*p),
          p->live?ET_R_BUSY:ET_R_DEAD,_Alignof(e3_cpu4_workspace))) return;
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_E3,2,p->data,1u,
          sizeof(*p->data),ET_R_BUSY,_Alignof(e3_cpu4_payload))) return;
    if (p->live || p->data || p->workspace_child || p->pin_child || !e3_pins_idle(&p->pins)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
}
#endif
