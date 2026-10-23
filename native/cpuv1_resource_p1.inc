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
#undef P_BASE
#endif
