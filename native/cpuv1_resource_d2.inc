#ifdef ET_CPUV1_RESOURCE_TESTING
#define D_BASE(K,B,N,S,STATE,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_D2,K,B,N,S,STATE,A)) return; \
} while (0)
void et_cpuv1_test_census_d2_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_D2,0,sizeof(et_d2_dataset_control),_Alignof(et_d2_dataset_control));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_D2,1,sizeof(int64_t),_Alignof(int64_t));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_D2,2,sizeof(et_d2_batch_control),_Alignof(et_d2_batch_control));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_D2,3,sizeof(uint8_t),_Alignof(uint8_t));

  for (et_d2_dataset_control *p=live_datasets;p;p=p->registry_next) {
    D_BASE(0,p,1u,sizeof(*p),ET_R_LIVE,_Alignof(et_d2_dataset_control));
    D_BASE(1,p->shuffle_window,p->shuffle_slots,sizeof(*p->shuffle_window),
           ET_R_LIVE,_Alignof(int64_t));
    et_d2_batch_control *b=p->current_batch;
    if (b) {
      D_BASE(2,b,1u,sizeof(*b),ET_R_BUSY,_Alignof(et_d2_batch_control));
      D_BASE(3,b->mask_data,b->elements,sizeof(*b->mask_data),ET_R_BUSY,_Alignof(uint8_t));
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
}
#undef D_BASE
#endif
