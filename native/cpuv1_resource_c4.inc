#ifdef ET_CPUV1_RESOURCE_TESTING
#if !defined(ET_G3C4_GENERATOR_PRIVATE) || !defined(ET_CPU4_TRAINING_PRIVATE) || !defined(ET_CPU4_CANDIDATE_PRIVATE)
#error "CPUv1 census requires the exact candidate-ON common owner profile"
#endif
#define C_BASE(K,B,N,S,STATE,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_C4,K,B,N,S,STATE,A)) return; \
} while (0)
#define C_CONTROL(K,B,STATE) C_BASE(K,B,1u,sizeof(*(B)),STATE,_Alignof(__typeof__(*(B))))
void et_cpuv1_test_census_c4_v1(et_cpuv1_resource_packet_v1 *packet) {
  for (et_g3c4_model_owner_internal *p=et_g3c4_owner_registry;p;p=p->registry_next) {
    unsigned state=p->state==ET_G3C4_OWNER_ABORTED?ET_R_DEAD:
        p->state==ET_G3C4_OWNER_SEALED?ET_R_LIVE:ET_R_PREPARED;
    C_CONTROL(0,p,state);
    if (p->magic!=ET_G3C4_OWNER_MAGIC || p->state<ET_G3C4_OWNER_OPEN ||
        p->state>ET_G3C4_OWNER_ABORTED || p->initialized>14) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    packet->details[ET_R_C4][p->state==ET_G3C4_OWNER_ABORTED?1:0]++;
    /* A live idle generator is allowed; an active native call is not. */
    if (p->active) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
  for (et_cpu4_candidate_internal *p=cpu4_candidates;p;p=p->next) {
    unsigned state=p->state==CPU4_CANDIDATE_DEAD?ET_R_DEAD:
                   p->state==CPU4_CANDIDATE_PREPARED?ET_R_PREPARED:ET_R_LIVE;
    C_CONTROL(1,p,state);
    if (p->state<CPU4_CANDIDATE_CONSTRUCTING || p->state>CPU4_CANDIDATE_DEAD ||
        !p->owner || p->owner->candidate!=p) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    if (p->state==CPU4_CANDIDATE_ADOPTED) packet->details[ET_R_C4][2]++;
  }
  for (et_g3c4_training_lease_internal *p=cpu4_training_leases;p;p=p->next) {
    unsigned state=p->phase==CPU4_LEASE_DEAD?ET_R_DEAD:
                   p->phase==CPU4_LEASE_IDLE?ET_R_LIVE:ET_R_BUSY;
    C_CONTROL(2,p,state);
    if (p->magic!=CPU4_LEASE_MAGIC || !p->owner ||
        p->phase<CPU4_LEASE_IDLE || p->phase>CPU4_LEASE_DEAD) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    if (p->phase!=CPU4_LEASE_DEAD) packet->details[ET_R_C4][3]++;
    if (p->phase==CPU4_LEASE_READ) packet->details[ET_R_C4][4]++;
    if (p->phase==CPU4_LEASE_WRITE) packet->details[ET_R_C4][5]++;
    for (cpu4_training_child *c=p->children;c;c=c->next) {
      C_CONTROL(3,c,c->live?ET_R_BUSY:ET_R_DEAD);
      if (c->kind<1 || c->kind>5) {
        et_cpuv1_test_resource_fail_v1(packet,ET_R_COVERAGE); return;
      }
      packet->details[ET_R_C4][7]++;
      packet->details[ET_R_C4][8+(unsigned)c->kind]++;
      if (c->live) packet->details[ET_R_C4][6]++;
    }
    if (p->phase!=CPU4_LEASE_IDLE && p->phase!=CPU4_LEASE_DEAD) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
  }
  for (et_g3c4_transport_header_internal *h=et_g3c4_transport_registry;
       h;h=h->registry_next) {
    size_t size,alignment; unsigned kind;
    unsigned state=h->state==ET_G3C4_CONTEXT_DEAD?ET_R_DEAD:
                   h->busy?ET_R_BUSY:ET_R_LIVE;
#define C_KIND(TYPE,K) do {kind=K;size=sizeof(TYPE);alignment=_Alignof(TYPE);} while (0)
    switch (h->kind) {
      case ET_G3C4_CONTEXT_KIND:C_KIND(et_g3c4_context_internal,4);break;
      case ET_G3C4_RNG_KIND:C_KIND(et_g3c4_rng_internal,5);break;
      case ET_G3C4_INPUT_KIND:C_KIND(et_g3c4_input_internal,6);break;
      case ET_G3C4_OUTPUT_KIND:C_KIND(et_g3c4_output_internal,7);break;
      case ET_G3C4_LOGITS_KIND:C_KIND(et_g3c4_logits_internal,8);break;
      case ET_G3C4_TYPED_IDS_KIND:
      case ET_G3C4_TYPED_LENGTHS_KIND:
      case ET_G3C4_TYPED_CACHE_KIND:C_KIND(et_g3c4_typed_result_internal,9);break;
      case ET_G3C4_TERMINAL_IDS_KIND:C_KIND(et_g3c4_terminal_ids_clone_internal,10);break;
      case ET_G3C4_IDS_KIND:
      case ET_G3C4_LENGTHS_KIND:
      case ET_G3C4_CACHE_LENGTHS_KIND:C_KIND(et_g3c4_result_tensor_internal,11);break;
      default:et_cpuv1_test_resource_fail_v1(packet,ET_R_COVERAGE);return;
    }
#undef C_KIND
    C_BASE(kind,h,1u,size,state,alignment);
    if (h->busy) { et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return; }
    if (h->kind==ET_G3C4_CONTEXT_KIND) {
      et_g3c4_context_internal *c=(void *)h;
      packet->details[ET_R_C4][h->state==ET_G3C4_CONTEXT_DEAD?17:16]++;
#ifdef ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE
      C_CONTROL(12,c->manual_frame,ET_R_BUSY);
      if (c->manual_frame) {
        et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return;
      }
#endif
      if (c->pins.held_mask || c->call_kind || c->acquired_mask) {
        et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);return;
      }
    }
  }
  /* The first walks already rejected duplicate/cyclic allocation bases. */
  for (et_g3c4_model_owner_internal *o=et_g3c4_owner_registry;o;o=o->registry_next) {
    size_t contexts=0;
    for (et_g3c4_transport_header_internal *h=et_g3c4_transport_registry;
         h;h=h->registry_next)
      if (h->kind==ET_G3C4_CONTEXT_KIND && h->state!=ET_G3C4_CONTEXT_DEAD &&
          ((et_g3c4_context_internal *)(void *)h)->owner==o) contexts++;
    if (contexts!=o->live_contexts) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
  }
}
et_kernel_runtime *et_cpuv1_test_c4_runtime_slot_v1(size_t index) {
  switch(index) {
    case 0:return et_g3c4_initializer_runtime;
    case 1:return et_g3c4_forward_runtime;
    case 2:return et_g3c4_sampler_runtime;
    default:return NULL;
  }
}
#undef C_BASE
#undef C_CONTROL
#endif
