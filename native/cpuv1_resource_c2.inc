#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_c2_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,0,1u,_Alignof(et_c2_private_checkpoint_image));
  packet->details[ET_R_C2_IO][1]=offsetof(et_c2_private_checkpoint_image,bytes);
  for (et_c2_private_checkpoint_image *p=c2_images;p;p=p->registry_next) {
    if (p->bytevector_length<0 ||
        (uint64_t)p->bytevector_length>SIZE_MAX-offsetof(et_c2_private_checkpoint_image,bytes)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_OVERFLOW);return;
    }
    size_t bytes=offsetof(et_c2_private_checkpoint_image,bytes)+(size_t)p->bytevector_length;
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_C2_IO,0,p,bytes,1u,
          ET_R_LIVE,_Alignof(et_c2_private_checkpoint_image))) return;
  }
}
#endif
