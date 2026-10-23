#ifdef ET_CPUV1_RESOURCE_TESTING
void et_cpuv1_test_census_generation_v1(et_cpuv1_resource_packet_v1 *p) {
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,0,sizeof(g3t_context),_Alignof(g3t_context));
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,1,sizeof(g3t_input),_Alignof(g3t_input));
#ifdef ET_G3T_MANUAL_LOGITS_PRIVATE
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,2,sizeof(g3t_logits),_Alignof(g3t_logits));
#endif
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,3,sizeof(g3t_output),_Alignof(g3t_output));
#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,4,sizeof(g3t_ids_clone),_Alignof(g3t_ids_clone));
#endif
#ifdef ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,5,sizeof(g3t_lengths_clone),_Alignof(g3t_lengths_clone));
#endif
#ifdef ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,6,sizeof(g3t_cache_lengths_clone),_Alignof(g3t_cache_lengths_clone));
#endif
#ifdef ET_G3T_OUTPUT_RNG_CLONE_PRIVATE
  et_cpuv1_test_resource_layout_v1(p,ET_R_GENERATION,7,sizeof(g3t_rng_clone),_Alignof(g3t_rng_clone));
#endif

  for(g3t_record *h=g3t_registry;h;h=h->next) {
    size_t size,alignment;unsigned kind;
#define G_KIND(T,K) do{size=sizeof(T);alignment=_Alignof(T);kind=K;}while(0)
    switch(h->kind) {
      case G3T_GENERATOR:G_KIND(g3t_context,0);break;
      case G3T_INPUT:G_KIND(g3t_input,1);break;
#ifdef ET_G3T_MANUAL_LOGITS_PRIVATE
      case G3T_LOGITS:G_KIND(g3t_logits,2);break;
#endif
      case G3T_OUTPUT:G_KIND(g3t_output,3);break;
#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
      case G3T_IDS_CLONE:G_KIND(g3t_ids_clone,4);break;
#endif
#ifdef ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
      case G3T_LENGTHS_CLONE:G_KIND(g3t_lengths_clone,5);break;
#endif
#ifdef ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
      case G3T_CACHE_LENGTHS_CLONE:G_KIND(g3t_cache_lengths_clone,6);break;
#endif
#ifdef ET_G3T_OUTPUT_RNG_CLONE_PRIVATE
      case G3T_RNG_CLONE:G_KIND(g3t_rng_clone,7);break;
#endif
      default:et_cpuv1_test_resource_fail_v1(p,ET_R_COVERAGE);return;
    }
#undef G_KIND
    if(et_cpuv1_test_resource_add_v1(p,ET_R_GENERATION,kind,h,1u,size,
        h->state==G3T_DEAD?ET_R_DEAD:ET_R_BUSY,alignment))return;
    if(h->state!=G3T_DEAD || h->busy) {
      et_cpuv1_test_resource_fail_v1(p,ET_R_NOT_QUIESCENT);return;
    }
    switch(kind) {
      case 0:{g3t_context *c=(void *)h;
        if(c->cache || c->frame.active || c->frame.transaction ||
           c->frame.candidate_cache || c->pins.held_mask || c->pending_output) {
          et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
        }
        break;}
      case 1:
#if defined(ET_G3T_INPUT_FROM_T1_PRIVATE) || defined(ET_G3T_OWNED_TOKEN_INPUT_PRIVATE)
        if(((g3t_input *)(void *)h)->tensor) {
          et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;
        }
#endif
        break;
#ifdef ET_G3T_MANUAL_LOGITS_PRIVATE
      case 2:if(((g3t_logits *)(void *)h)->tensor) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;}break;
#endif
      case 3:if(((g3t_output *)(void *)h)->ids) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;}break;
#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
      case 4:if(((g3t_ids_clone *)(void *)h)->ids) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;}break;
#endif
#ifdef ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
      case 5:if(((g3t_lengths_clone *)(void *)h)->value) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;}break;
#endif
#ifdef ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
      case 6:if(((g3t_cache_lengths_clone *)(void *)h)->value) {
        et_cpuv1_test_resource_fail_v1(p,ET_R_MISMATCH);return;}break;
#endif
      default:break;
    }
  }
}
#endif
