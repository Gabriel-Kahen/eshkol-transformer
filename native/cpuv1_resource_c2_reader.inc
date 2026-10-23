#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_c2_reader_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,1,sizeof(et_c2_checkpoint_reader),_Alignof(et_c2_checkpoint_reader));

  for (et_c2_checkpoint_reader *p=live_readers;p;p=p->registry_next) {
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_C2_IO,1,p,1u,sizeof(*p),
          ET_R_LIVE,_Alignof(et_c2_checkpoint_reader))) return;
    if (p->descriptor<0) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return;
    }
    packet->details[ET_R_C2_IO][0]++;
  }
}
#endif
