#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_inspector_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_E3,3,sizeof(image),_Alignof(image));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_E3,4,sizeof(output),_Alignof(output));

  for (image *p=images;p;p=p->next)
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_E3,3,p,1u,sizeof(*p),
          ET_R_LIVE,_Alignof(image))) return;
  for (output *p=outputs;p;p=p->next)
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_E3,4,p,1u,sizeof(*p),
          ET_R_LIVE,_Alignof(output))) return;
}
#endif
