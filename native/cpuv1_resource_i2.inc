#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_i2_v1(et_cpuv1_resource_packet_v1 *packet) {
#define I2_LAYOUT(K,T) et_cpuv1_test_resource_layout_v1(packet,ET_R_F32,K,sizeof(T),_Alignof(T))
  I2_LAYOUT(15,et_i2_copy_builder);I2_LAYOUT(16,et_i2_reset_builder);
  I2_LAYOUT(17,et_i2_decode_builder);I2_LAYOUT(18,uint64_t);
  I2_LAYOUT(19,unsigned char);I2_LAYOUT(20,et_f32_tensor_copy_assignment_v1);
  I2_LAYOUT(21,et_f32_parameter *);I2_LAYOUT(22,uint32_t);
#undef I2_LAYOUT
  if(et_i2_live_copy_builders || et_i2_live_reset_builders || et_i2_live_decode_builders) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return;
  }
#define I2_DEAD(T,H,K,EXTRA) \
  for(T *p=H;p;p=p->registry_next) { \
    if(et_cpuv1_test_resource_add_v1(packet,ET_R_F32,K,p,1u,sizeof(*p),ET_R_DEAD,_Alignof(T)))return; \
    if(p->magic || (EXTRA)) {et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return;} \
  }
  I2_DEAD(et_i2_copy_builder,et_i2_retired_copy_builders,15,p->count || p->assignments || p->plan);
  I2_DEAD(et_i2_reset_builder,et_i2_retired_reset_builders,16,p->count || p->parameters || p->plan);
  I2_DEAD(et_i2_decode_builder,et_i2_retired_decode_builders,17,p->rank || p->shape || p->payload || p->payload_size);
#undef I2_DEAD
  packet->details[ET_R_F32][32]=et_i2_owned_clone_count;
}
#endif
