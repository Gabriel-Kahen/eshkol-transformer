/* Future native control; SOURCE STAGED ONLY. Reuse the actual manual gate roster.
 * Add --wrap=et_a2_kv_cache_transaction_stage_layer_v1 and
 * --wrap=et_a2_kv_cache_transaction_commit_v1 for the prefix failure controls. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE 1
#define ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN cpu4_history_manual_predecessor_main
#include "test_manual_role_step.c"
#undef ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN

static et_g3c4_context_internal *prefix_cut_context;
static int prefix_cut_phase;
static size_t prefix_custody_witnesses;
int32_t __real_et_a2_kv_cache_transaction_stage_layer_v1(
    et_a2_kv_cache_transaction *,uint64_t,const et_kernel_tensor_view_v1 *,
    const et_kernel_tensor_view_v1 *,et_kernel_error *);
int32_t __real_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **,et_kernel_error *);
static int prefix_failure(et_kernel_error *e) {
  et_kernel_error_clear(e);
  e->category=ET_KERNEL_ERROR_INTERNAL;e->code=ET_KERNEL_CODE_PROVIDER_REJECTED;
  return ET_KERNEL_ERROR_INTERNAL;
}
static void prefix_custody(et_a2_kv_cache_transaction *tx) {
  cpu4_history_node *h=cpu4_history_auth(prefix_cut_context);CHECK(h!=NULL);
  CHECK(h->pending.operation==4u && h->pending.transaction==tx &&
        prefix_cut_context->manual_frame->a2_transaction==tx);
  ++prefix_custody_witnesses;
}
int32_t __wrap_et_a2_kv_cache_transaction_stage_layer_v1(
    et_a2_kv_cache_transaction *tx,uint64_t layer,
    const et_kernel_tensor_view_v1 *keys,const et_kernel_tensor_view_v1 *values,
    et_kernel_error *e) {
  if(prefix_cut_context!=NULL && prefix_cut_phase==1) {
    prefix_custody(tx);return prefix_failure(e);
  }
  return __real_et_a2_kv_cache_transaction_stage_layer_v1(tx,layer,keys,values,e);
}
int32_t __wrap_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **tx,et_kernel_error *e) {
  if(prefix_cut_context!=NULL && prefix_cut_phase==2) {
    prefix_custody(*tx);return prefix_failure(e);
  }
  return __real_et_a2_kv_cache_transaction_commit_v1(tx,e);
}
static size_t manual_capture_count(void) {
  size_t n=0;for(cpu4_capture_provenance *p=cpu4_captures;p;p=p->next) ++n;
  return n;
}
/* The source is a detached capture; frame/context below are real writable
 * native allocations. No fabricated destination is dereferenced. */
static void manual_alias_control(et_g3c4_context_internal *c,
                                  et_cpu4_c4_capture *capture) {
  cpu4_history_node *h=cpu4_history_auth(c);CHECK(h!=NULL);
  et_g3c4_manual_frame_internal *frame=c->manual_frame;
  CHECK(frame!=NULL && h->pending.frame==frame);
  _Static_assert(sizeof(*frame)>=4768u,"manual alias destination extent");
  et_g3c4_manual_frame_internal before=*frame;
  cpu4_history_pending pending=h->pending;
  size_t census=manual_capture_count();
  for(int phase=0;phase<3;++phase) {
    uint32_t kind=c->transport.kind;
    if(phase!=1) c->manual_frame=NULL;
    if(phase!=0) c->transport.kind=ET_G3C4_RNG_KIND;
    et_g3c4_context_internal corrupted=*c;
    CHECK(et_cpu4_private_c4_capture_copy_record_v1(capture,
        (et_cpu4_c4_capture_record_v1 *)frame,224u)==ET_G3C4_INVALID_ARGUMENT);
    CHECK(et_cpu4_private_c4_capture_copy_parameters_v1(capture,
        frame,4768u)==ET_G3C4_INVALID_ARGUMENT);
    CHECK(et_cpu4_private_c4_capture_copy_record_v1(capture,
        (et_cpu4_c4_capture_record_v1 *)((unsigned char *)c+sizeof(*c)-224u),
        224u)==ET_G3C4_INVALID_ARGUMENT);
    CHECK(memcmp(frame,&before,sizeof(before))==0 &&
          memcmp(c,&corrupted,sizeof(corrupted))==0 &&
          memcmp(&h->pending,&pending,sizeof(pending))==0 &&
          manual_capture_count()==census);
    CHECK(et_g3c4_private_call_finish_v1(c)!=0);
    CHECK(et_g3c4_private_call_abort_v1(c)!=0);
    c->transport.kind=kind;c->manual_frame=frame;
    CHECK(cpu4_history_auth(c)==h);
  }
  et_g3c4_manual_frame_internal forged=*frame;
  c->manual_frame=&forged;
  CHECK(et_g3c4_private_call_finish_v1(c)!=0 && et_g3c4_private_call_abort_v1(c)!=0);
  c->manual_frame=frame;CHECK(cpu4_history_auth(c)==h);
}
static void manual_history_case(et_g3c4_model_owner_internal *owner,
                                 int kind,int length,int cut) {
  et_g3c4_context_internal *source=create_generator(owner);
  et_cpu4_c4_capture *detached=et_cpu4_private_c4_capture_create_v1(source);
  CHECK(detached!=NULL);OK(et_g3c4_private_generator_close_v1(source));
  et_g3c4_context_internal *c=et_g3c4_private_generator_seed_v1(owner,7,0,
      INT64_C(0x3f800000),256,INT64_C(0x3f800000),1,41);
  CHECK(c!=NULL);
  if(kind==2) {
    float logits[256];OK(et_g3c4_private_call_acquire_v1(c,2,0));
    OK(et_g3c4_private_prefill1_v1(c,47,logits));
    OK(et_g3c4_private_call_abort_v1(c));
  }
  cpu4_history_record before=c->cpu4_history;
  int64_t rng[4];memcpy(rng,c->generator_rng_words,sizeof(rng));
  et_g3c4_input_internal *input=create_prompt((const int64_t[2]){41,43},length);
  OK(et_g3c4_private_call_acquire_v1(c,kind-1,0));
  et_g3c4_logits_internal *pending=et_g3c4_private_logits_reserve_v1(c);
  CHECK(pending!=NULL);OK(et_g3c4_private_frame_begin_v1(c,input,kind));
  manual_alias_control(c,detached); /* ACTIVE */
  for(int64_t ordinal=0;ordinal<=20;++ordinal) {
    if(ordinal==10 && cut!=0) {
      prefix_cut_context=c;prefix_cut_phase=cut;
      size_t witnessed=prefix_custody_witnesses;
      CHECK(et_g3c4_private_role_step_v1(c,ordinal)!=0);
      prefix_cut_context=NULL;prefix_cut_phase=0;
      cpu4_history_node *h=cpu4_history_auth(c);CHECK(h!=NULL);
      CHECK(prefix_custody_witnesses==witnessed+1u && h->pending.frame==c->manual_frame &&
            h->pending.transaction==NULL && h->pending.candidate==NULL &&
            c->manual_frame->a2_transaction==NULL && c->manual_frame->a2_candidate==NULL &&
            c->manual_frame->next_ordinal==10);
      CHECK(memcmp(&before,&c->cpu4_history,sizeof(before))==0 &&
            memcmp(rng,c->generator_rng_words,sizeof(rng))==0);
    }
    OK(et_g3c4_private_role_step_v1(c,ordinal));
    CHECK(memcmp(&before,&c->cpu4_history,sizeof(before))==0);
  }
  OK(et_g3c4_private_frame_prepare_v1(c,pending));
  OK(et_g3c4_private_call_prepare_end_v1(c));
  OK(et_g3c4_private_frame_commit_v1(c));
  CHECK(c->cpu4_history.count==(uint32_t)(kind==1?length:2) &&
        c->cpu4_history.commit_epoch==before.commit_epoch+1u &&
        c->cpu4_history.generated_count==0u && c->cpu4_history.generated_eos_mask==0u &&
        memcmp(rng,c->generator_rng_words,sizeof(rng))==0);
  if(kind==2) CHECK(c->cpu4_history.ids[0]==47 && c->cpu4_history.origin[0]==1u &&
                    c->cpu4_history.ids[1]==41 && c->cpu4_history.origin[1]==3u);
  else CHECK(c->cpu4_history.ids[0]==41 && c->cpu4_history.origin[0]==1u);
  manual_alias_control(c,detached); /* COMMITTED allocation still owned */
  cpu4_history_node *h=cpu4_history_auth(c);CHECK(h!=NULL);
  CHECK(h->pending.frame==c->manual_frame && h->pending.epoch==h->history.commit_epoch);
  /* Artificial trusted test setup proves drain is independent of future room. */
  h->history.commit_epoch=c->cpu4_history.commit_epoch=h->pending.epoch=UINT64_MAX;
  cpu4_history_record committed=c->cpu4_history;
  CHECK(et_g3c4_private_frame_commit_v1(c)!=0 && et_g3c4_private_role_step_v1(c,0)!=0 &&
        et_g3c4_private_call_abort_v1(c)!=0);
  CHECK(memcmp(&committed,&c->cpu4_history,sizeof(committed))==0);
  OK(et_g3c4_private_call_finish_v1(c));
  CHECK(cpu4_history_zero(&h->pending,sizeof(h->pending)) && c->manual_frame==NULL);
  CHECK(memcmp(&committed,&c->cpu4_history,sizeof(committed))==0);
  et_cpu4_c4_capture *capture=et_cpu4_private_c4_capture_create_v1(c);
  CHECK(capture!=NULL);et_cpu4_c4_capture_record_v1 record;
  OK(et_cpu4_private_c4_capture_copy_record_v1(capture,&record,sizeof(record)));
  CHECK(record.cache_capacity==2u && record.count==c->cpu4_history.count);
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  OK(et_cpu4_private_c4_capture_release_v1(detached));
  OK(et_g3c4_private_tensor_release_v1(input));OK(et_g3c4_private_tensor_release_v1(pending));
  OK(et_g3c4_private_generator_close_v1(c));
}
int main(void) {
  et_g3c4_model_owner_internal *owner=create_owner();
  manual_history_case(owner,1,1,0);manual_history_case(owner,1,2,0);
  manual_history_case(owner,2,1,0);manual_history_case(owner,2,1,1);manual_history_case(owner,2,1,2);
  printf("CPU4 C4 manual history future native controls PASS: checks=%zu\n",checks);
  return 0;
}
