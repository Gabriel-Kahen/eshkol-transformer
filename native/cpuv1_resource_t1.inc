#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_t1_v1(et_cpuv1_resource_packet_v1 *packet) {
  for (et_t1_i64_shell *p=et_t1_i64_shell_registry;p;p=p->registry_next) {
    if (et_cpuv1_test_resource_add_v1(packet,ET_R_I64_T1,5,p,1u,sizeof(*p),
          p->sealed ? ET_R_LIVE : ET_R_PENDING,_Alignof(et_t1_i64_shell))) return;
  }
}
#endif
