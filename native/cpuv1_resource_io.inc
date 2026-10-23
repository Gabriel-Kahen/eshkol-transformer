#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_io_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,2,1u,1u);
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,3,1u,1u);
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,4,1u,1u);
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C2_IO,5,1u,1u);
  /* The serialized observer is called only after synchronous IO returns.
   * Paths and read/save staging allocations have no surviving owner. Their
   * independently expected zero census must reconcile with all event hooks;
   * counters alone never manufacture an owned base. */
  (void)packet;
}
#endif
