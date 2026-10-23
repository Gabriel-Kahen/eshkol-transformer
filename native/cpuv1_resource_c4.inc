#ifdef ET_CPUV1_RESOURCE_TESTING
#if !defined(ET_G3C4_GENERATOR_PRIVATE) || !defined(ET_CPU4_TRAINING_PRIVATE) || !defined(ET_CPU4_CANDIDATE_PRIVATE)
#error "CPUv1 census requires the exact candidate-ON common owner profile"
#endif
#define C_BASE(K,B,N,S,STATE,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_C4,K,B,N,S,STATE,A)) return; \
} while (0)
#define C_CONTROL(K,B,STATE) C_BASE(K,B,1u,sizeof(*(B)),STATE,_Alignof(__typeof__(*(B))))
void et_cpuv1_test_census_c4_v1(et_cpuv1_resource_packet_v1 *packet) {
  packet->details[ET_R_C4][ET_R_C4_COMPILED_KINDS]=(UINT64_C(1)<<13)-1u;
  packet->details[ET_R_C4][ET_R_C4_NOT_APPLICABLE_KINDS]=0;
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,0,sizeof(et_g3c4_model_owner_internal),_Alignof(et_g3c4_model_owner_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,1,sizeof(et_cpu4_candidate_internal),_Alignof(et_cpu4_candidate_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,2,sizeof(et_g3c4_training_lease_internal),_Alignof(et_g3c4_training_lease_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,3,sizeof(cpu4_training_child),_Alignof(cpu4_training_child));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,4,sizeof(et_g3c4_context_internal),_Alignof(et_g3c4_context_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,5,sizeof(et_g3c4_rng_internal),_Alignof(et_g3c4_rng_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,6,sizeof(et_g3c4_input_internal),_Alignof(et_g3c4_input_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,7,sizeof(et_g3c4_output_internal),_Alignof(et_g3c4_output_internal));
#ifdef ET_G3C4_LOGITS_RESERVATION_PRIVATE
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,8,sizeof(et_g3c4_logits_internal),_Alignof(et_g3c4_logits_internal));
#else
  /* The owning type/allocator is compiled out: N/A, never a measured size0. */
  packet->details[ET_R_C4][ET_R_C4_COMPILED_KINDS]&=~(UINT64_C(1)<<8);
  packet->details[ET_R_C4][ET_R_C4_NOT_APPLICABLE_KINDS]|=UINT64_C(1)<<8;
#endif
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,9,sizeof(et_g3c4_typed_result_internal),_Alignof(et_g3c4_typed_result_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,10,sizeof(et_g3c4_terminal_ids_clone_internal),_Alignof(et_g3c4_terminal_ids_clone_internal));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,11,sizeof(et_g3c4_result_tensor_internal),_Alignof(et_g3c4_result_tensor_internal));
#ifdef ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE
  et_cpuv1_test_resource_layout_v1(packet,ET_R_C4,12,sizeof(et_g3c4_manual_frame_internal),_Alignof(et_g3c4_manual_frame_internal));
#else
  /* The owning type/allocator is compiled out: N/A, never a measured size0. */
  packet->details[ET_R_C4][ET_R_C4_COMPILED_KINDS]&=~(UINT64_C(1)<<12);
  packet->details[ET_R_C4][ET_R_C4_NOT_APPLICABLE_KINDS]|=UINT64_C(1)<<12;
#endif

  for (et_g3c4_model_owner_internal *p=et_g3c4_owner_registry;p;p=p->registry_next) {
    unsigned state=p->state==ET_G3C4_OWNER_ABORTED?ET_R_DEAD:
        p->state==ET_G3C4_OWNER_SEALED?ET_R_LIVE:ET_R_PREPARED;
    C_CONTROL(0,p,state);
    if (p->magic!=ET_G3C4_OWNER_MAGIC || p->state<ET_G3C4_OWNER_OPEN ||
        p->state>ET_G3C4_OWNER_ABORTED || p->initialized>14) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    packet->details[ET_R_C4][p->state==ET_G3C4_OWNER_ABORTED?1:0]++;
    if (p->state!=ET_G3C4_OWNER_SEALED && p->state!=ET_G3C4_OWNER_ABORTED) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
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
    packet->details[ET_R_C4][24+(unsigned)p->state]++;
    if (p->state==CPU4_CANDIDATE_CONSTRUCTING ||
        p->state==CPU4_CANDIDATE_PREPARED || p->state==CPU4_CANDIDATE_REVOKED) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
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
      if (c->live) {
        packet->details[ET_R_C4][6]++;
        et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
      }
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
#ifdef ET_G3C4_LOGITS_RESERVATION_PRIVATE
      case ET_G3C4_LOGITS_KIND:C_KIND(et_g3c4_logits_internal,8);break;
#endif
      case ET_G3C4_TYPED_IDS_KIND:
      case ET_G3C4_TYPED_LENGTHS_KIND:
      case ET_G3C4_TYPED_CACHE_KIND:C_KIND(et_g3c4_typed_result_internal,9);break;
      case ET_G3C4_TERMINAL_IDS_KIND:C_KIND(et_g3c4_terminal_ids_clone_internal,10);break;
      case ET_G3C4_IDS_KIND:
      case ET_G3C4_LENGTHS_KIND:
      case ET_G3C4_CACHE_LENGTHS_KIND:C_KIND(et_g3c4_result_tensor_internal,11);break;
      /* Includes any forged member of a compiled-out kind. */
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
/* Source/native bijection for the generated candidate owning-scope census.
 * Scan real registries before any source-supplied pointer is dereferenced.
 * Same-object and copied-row duplicate capability references are separately
 * rejected by bounded owning-Scheme occurrence counts before this call. */
int64_t et_cpuv1_test_candidate_link_v1(void *raw,void *capability,
 void *owner,void *scope,int64_t source_phase) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  et_cpu4_candidate_internal *p=cpu4_candidates;
  et_g3c4_model_owner_internal *o=et_g3c4_owner_registry;
  while(p && p!=capability)p=p->next;
  while(o && o!=owner)o=o->registry_next;
  int expected=source_phase==1?CPU4_CANDIDATE_UNPUBLISHED:
               source_phase==2?CPU4_CANDIDATE_ADOPTED:
               source_phase==3?CPU4_CANDIDATE_DEAD:-1;
  if(!packet || !p || !o || expected<0 || p->owner!=o ||
     o->candidate!=p || o->magic!=ET_G3C4_OWNER_MAGIC ||
     p->origin_pid!=(int64_t)getpid() || p->state!=expected ||
     !packet->scheme_p1[0] ||
     p->p1_context!=(void *)(uintptr_t)packet->scheme_p1[0] ||
     !scope || p->p1_scope!=scope || o->active) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  if(expected==CPU4_CANDIDATE_DEAD) {
    if(!cpu4_candidate_dead_exact(p)) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
    }
  } else if(o->state!=ET_G3C4_OWNER_SEALED || o->initialized!=14 ||
            memcmp(p->parameters,o->parameters,sizeof(p->parameters)) ||
            memcmp(p->handles,o->handles,sizeof(p->handles))) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  packet->details[ET_R_C4][40+(unsigned)p->state]++;
  return (int64_t)packet->status;
}
int64_t et_cpuv1_test_candidate_links_finish_v1(void *raw) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  if(!packet)return ET_R_ARGUMENT;
  const unsigned phases[]={CPU4_CANDIDATE_UNPUBLISHED,
                           CPU4_CANDIDATE_ADOPTED,CPU4_CANDIDATE_DEAD};
  for(unsigned i=0;i<3;i++) {
    unsigned phase=phases[i];
    if(packet->details[ET_R_C4][24+phase]!=packet->details[ET_R_C4][40+phase] ||
       packet->scheme_source[64][i+1]!=packet->details[ET_R_C4][40+phase])
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);
  }
  if(packet->scheme_candidate[0]!=packet->rows[ET_R_C4][1].bases)
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);
  return (int64_t)packet->status;
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
