#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_m3_v1(et_cpuv1_resource_packet_v1 *packet) {
#define M_LAYOUT(K,T) et_cpuv1_test_resource_layout_v1(packet,ET_R_TRAIN,K,sizeof(T),_Alignof(T))
  M_LAYOUT(3,initializer);M_LAYOUT(4,owner);M_LAYOUT(5,input);M_LAYOUT(6,logits);
  M_LAYOUT(7,workspace);M_LAYOUT(8,m3_workspace);M_LAYOUT(9,m3_graph);M_LAYOUT(10,m3_logits);
#undef M_LAYOUT
  /* Source closure includes this legacy owner; the frozen CPU4 one-pass path
   * invokes none of it. Any legacy authority/cache is an explicit profile
   * rejection, never an uncharged substitute for canonical C4 authority. */
  if(registry || staged_owner || m3_workspaces || runtimes[0] || runtimes[1] || runtimes[2])
    et_cpuv1_test_resource_fail_v1(packet,ET_R_UNSUPPORTED);
}
#endif
