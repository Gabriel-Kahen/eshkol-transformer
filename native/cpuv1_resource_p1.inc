#ifdef ET_CPUV1_RESOURCE_TESTING
#ifndef ET_P1_TRUSTED_BUILD
#error "CPUv1 resource producer requires the original trusted P1 profile"
#endif
#define P_BASE(K,B,N,S,STATE,A) do { \
 if (et_cpuv1_test_resource_add_v1(packet,ET_R_P1,K,B,N,S,STATE,A)) return; \
} while (0)
void et_cpuv1_test_census_p1_v1(et_cpuv1_resource_packet_v1 *packet) {
  et_cpuv1_test_resource_layout_v1(packet,ET_R_P1,0,sizeof(et_p1_token),_Alignof(et_p1_token));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_P1,1,sizeof(et_p1_record),_Alignof(et_p1_record));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_P1,2,sizeof(et_p1_context),_Alignof(et_p1_context));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_P1,3,sizeof(et_p1_construction),_Alignof(et_p1_construction));
  et_cpuv1_test_resource_layout_v1(packet,ET_R_P1,4,sizeof(et_p1_record *),_Alignof(et_p1_record *));

  for (et_p1_context *p=contexts;p;p=p->next)
    P_BASE(2,p,1u,sizeof(*p),p->live?ET_R_LIVE:ET_R_DEAD,_Alignof(et_p1_context));
  for (et_p1_record *p=records;p;p=p->next) {
    unsigned state=p->live?ET_R_LIVE:ET_R_DEAD;
    if (!p->token || p->kind<1 || p->kind>ET_P1_TOKEN_STATE_TENSOR) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_COVERAGE); return;
    }
    P_BASE(1,p,1u,sizeof(*p),state,_Alignof(et_p1_record));
    P_BASE(0,p->token,1u,sizeof(*p->token),state,_Alignof(et_p1_token));
    packet->details[ET_R_P1][(p->live?0u:16u)+(unsigned)p->kind]++;
    et_p1_context *owner=contexts;
    while (owner && owner!=p->owner) owner=owner->next;
    if (!owner) { et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return; }
  }
  for (et_p1_construction *p=constructions;p;p=p->next) {
    unsigned state=p->state==0?ET_R_PENDING:p->state==3?ET_R_PREPARED:
                   (p->state==2 || p->state==4)?ET_R_DEAD:ET_R_LIVE;
    if (p->state>4 || p->count>ET_P1_CONSTRUCTION_CAPACITY) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
    P_BASE(3,p,1u,sizeof(*p),state,_Alignof(et_p1_construction));
    packet->details[ET_R_P1][40+p->state]++;
    if (p->state==0 || p->state==3) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT); return;
    }
    P_BASE(4,p->entries,ET_P1_CONSTRUCTION_CAPACITY,sizeof(*p->entries),state,
           _Alignof(et_p1_record *));
#ifdef ET_G3R_CANDIDATE_RETIRE_PRIVATE
    if (p->candidate_roster_base && p->candidate_roster_base!=p->entries) {
      et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH); return;
    }
#endif
  }
  packet->details[ET_R_P1][32]=active_construction!=NULL;
  if (active_construction) et_cpuv1_test_resource_fail_v1(packet,ET_R_NOT_QUIESCENT);
}
int64_t et_cpuv1_test_p1_context_v1(void *raw,void *candidate) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  et_p1_context *p=contexts;
  while(p && p!=candidate)p=p->next;
  if(!packet || !p || !p->live) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  packet->scheme_p1[0]=(uint64_t)(uintptr_t)p;
  for(et_p1_record *r=records;r;r=r->next) {
    if(r->owner!=p)continue;
    if(r->kind==ET_P1_TOKEN_CALLBACK_IDENTITY)packet->scheme_p1[15]++;
    else packet->scheme_p1[14]++;
    if(r->kind==ET_P1_TOKEN_PROVIDER && r->live && r->sealed)
      packet->scheme_p1[16]++;
  }
  return (int64_t)packet->status;
}
/* Read the immutable seven callback slots plus release-owned binding slot.
 * Native callback records are not Scheme shell/index entries. */
int64_t et_cpuv1_test_p1_callback_v1(void *raw,void *context,void *provider,
 void *callback,int64_t slot) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  et_p1_record *p=find_record(provider),*c=find_record(callback);
  if(!packet || !p || !c || slot<0 || slot>7 ||
     !token_integrity(p) || !token_integrity(c) || !p->live || !c->live ||
     p->owner!=context || c->owner!=context || p->kind!=ET_P1_TOKEN_PROVIDER ||
     c->kind!=ET_P1_TOKEN_CALLBACK_IDENTITY || !p->sealed || c->binding!=p ||
     (slot<7?p->callbacks[slot]:p->binding)!=c) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  packet->scheme_p1[13]++;
  return (int64_t)packet->status;
}
int64_t et_cpuv1_test_p1_shell_v1(void *raw,void *context,void *shell,
 int64_t kind,int64_t known_dead) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  et_p1_record *record=find_record(shell);
  if(!packet || !record || !token_identity_integrity(record) ||
     record->origin_pid!=(int64_t)getpid() || record->owner!=context ||
     !context_integrity(record->owner) ||
     kind<1 || kind>8 || record->kind!=kind || known_dead<0 || known_dead>1 ||
     (known_dead?record->live:!record->live)) {
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);return ET_R_MISMATCH;
  }
  packet->scheme_p1[4+(unsigned)kind]++;
  et_cpuv1_test_scheme_state_v1(packet,3,record->live?1:2,2,record->live?1:2);
  return (int64_t)packet->status;
}
int64_t et_cpuv1_test_p1_links_finish_v1(void *raw) {
  et_cpuv1_resource_packet_v1 *packet=raw;
  if(!packet)return ET_R_ARGUMENT;
  if(packet->scheme_p1[1]!=packet->scheme_states[3][0] ||
     packet->scheme_states[3][0]!=packet->scheme_p1[14] ||
     packet->scheme_p1[13]!=packet->scheme_p1[15] ||
     packet->scheme_p1[16]>UINT64_MAX/8u ||
     packet->scheme_p1[13]!=packet->scheme_p1[16]*8u)
    et_cpuv1_test_resource_fail_v1(packet,ET_R_MISMATCH);
  return (int64_t)packet->status;
}
/* Genuine native-unit negatives use actual owning fields, never guessed ABI
 * offsets. All mutation state is stack-local and restored on the same call. */
int64_t et_cpuv1_test_p1_shell_mutation_v1(void *raw,void *context,void *shell,
 int64_t kind,int64_t known_dead,unsigned mutation) {
  et_p1_record *record=find_record(shell);
  if(!record || !record->token || (mutation!=1 && mutation!=2))return ET_R_ARGUMENT;
  uint64_t nonce=record->token->nonce_lo;
  int64_t pid=record->origin_pid;
  if(mutation==1)record->token->nonce_lo^=UINT64_C(1);
  else record->origin_pid=-1;
  int64_t result=et_cpuv1_test_p1_shell_v1(raw,context,shell,kind,known_dead);
  record->token->nonce_lo=nonce;record->origin_pid=pid;
  return result;
}
#undef P_BASE
#endif
