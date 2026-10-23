/* Future native control; SOURCE STAGED ONLY. Reuse the actual manual gate roster. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE 1
#define ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN cpu4_history_manual_predecessor_main
#include "test_manual_role_step.c"
#undef ET_G3C4_MANUAL_ROLE_STEP_TEST_MAIN

static void manual_history_case(et_g3c4_model_owner_internal *owner,int kind,int length) {
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
  for(int64_t ordinal=0;ordinal<=20;++ordinal) {
    OK(et_g3c4_private_role_step_v1(c,ordinal));
    CHECK(memcmp(&before,&c->cpu4_history,sizeof(before))==0);
    /* Includes role10's real candidate-only prefix/append: no context event. */
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
  OK(et_g3c4_private_call_finish_v1(c));
  et_cpu4_c4_capture *capture=et_cpu4_private_c4_capture_create_v1(c);
  CHECK(capture!=NULL);et_cpu4_c4_capture_record_v1 record;
  OK(et_cpu4_private_c4_capture_copy_record_v1(capture,&record,sizeof(record)));
  CHECK(record.cache_capacity==2u && record.count==c->cpu4_history.count);
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  OK(et_g3c4_private_tensor_release_v1(input));OK(et_g3c4_private_tensor_release_v1(pending));
  OK(et_g3c4_private_generator_close_v1(c));
}
int main(void) {
  et_g3c4_model_owner_internal *owner=create_owner();
  manual_history_case(owner,1,1);manual_history_case(owner,1,2);manual_history_case(owner,2,1);
  printf("CPU4 C4 manual history future native controls PASS: checks=%zu\n",checks);
  return 0;
}
