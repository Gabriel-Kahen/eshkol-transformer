#ifdef ET_CPUV1_RESOURCE_TESTING
/* Cache payload belongs solely to A2. F32/C4 references never charge it. */
void et_cpuv1_test_census_a2_v1(et_cpuv1_resource_packet_v1 *p) {
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,16,sizeof(et_a2_kv_cache),_Alignof(et_a2_kv_cache));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,17,sizeof(et_a2_kv_cache_transaction),_Alignof(et_a2_kv_cache_transaction));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,18,sizeof(et_a2_kv_cache_transaction_view),_Alignof(et_a2_kv_cache_transaction_view));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,19,sizeof(et_a2_kv_cache_read_borrow),_Alignof(et_a2_kv_cache_read_borrow));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,20,sizeof(float),_Alignof(float));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,21,sizeof(float),_Alignof(float));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,22,sizeof(int64_t),_Alignof(int64_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,23,sizeof(uint8_t),_Alignof(uint8_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,24,sizeof(int64_t),_Alignof(int64_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,25,sizeof(int64_t),_Alignof(int64_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,26,sizeof(uint8_t),_Alignof(uint8_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,27,sizeof(uint8_t),_Alignof(uint8_t));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,28,sizeof(et_kernel_tensor_view_v1),_Alignof(et_kernel_tensor_view_v1));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,29,sizeof(et_kernel_tensor_view_v1),_Alignof(et_kernel_tensor_view_v1));

#define A_ADD(K,B,N,S,T) do { \
  if(et_cpuv1_test_resource_add_v1(p,ET_R_GENERATION,K,B,N,S,ET_R_LIVE,_Alignof(T)))return; \
} while(0)
  for(et_a2_kv_cache *c=live_caches;c;c=c->registry_next) {
    A_ADD(16,c,1u,sizeof(*c),et_a2_kv_cache);
    if(c->magic!=CACHE_MAGIC || !c->layers || !c->batch || !c->capacity ||
       c->active_transaction || c->active_borrow) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_NOT_QUIESCENT);return;
    }
    size_t elements=c->layers,mask;
    if(!mul_size(elements,c->batch,&elements) ||
       !mul_size(elements,c->kv_heads,&elements) ||
       !mul_size(elements,c->capacity,&elements) ||
       !mul_size(elements,c->head_dimension,&elements) ||
       !mul_size(c->batch,c->capacity,&mask) || elements!=c->total_elements) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_OVERFLOW);return;
    }
    A_ADD(20,c->keys,elements,sizeof(float),float);
    A_ADD(21,c->values,elements,sizeof(float),float);
    A_ADD(22,c->lengths,c->batch,sizeof(int64_t),int64_t);
    A_ADD(23,c->key_keep_mask,mask,1u,uint8_t);
  }
  /* Active transactions/views/borrows are prohibited at quiescent observe.
   * Their independent events must be fully freed before this boundary. */
  if(live_transactions || live_views || live_borrows)
    et_cpuv1_test_resource_fail_v1(p,ET_R_NOT_QUIESCENT);
#undef A_ADD
}
#endif
