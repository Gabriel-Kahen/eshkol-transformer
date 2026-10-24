/* Genuine future native controls. SOURCE STAGED ONLY: compiler/runtime NOT RUN.
 * Add --wrap=et_kernel_runtime_discover, --wrap=et_a2_kv_cache_create_v1 and
 * --wrap=et_a2_kv_cache_transaction_begin_v1 to the existing terminal roster. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_CPU4_C4_HISTORY_TESTING 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_P2_G2_EOS_TEST_DISPATCH cpu4_history_lower_dispatch
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN cpu4_history_terminal_predecessor_main
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

static size_t history_dispatches, history_discoveries, history_creates, history_begins, history_samples;
int32_t __wrap_et_kernel_runtime_dispatch(const et_kernel_runtime *runtime,
    const et_kernel_call_v1 *call, et_kernel_error *error) {
  ++history_dispatches;
  if (strcmp(call->capability,"g3s.greedy")==0 ||
      strcmp(call->capability,"g3s.categorical")==0) ++history_samples;
  return cpu4_history_lower_dispatch(runtime,call,error);
}
int32_t __real_et_kernel_runtime_discover(et_kernel_provider_resolver_v1,
    void *,et_kernel_runtime **,et_kernel_error *);
int32_t __wrap_et_kernel_runtime_discover(et_kernel_provider_resolver_v1 resolver,
    void *context,et_kernel_runtime **runtime,et_kernel_error *error) {
  ++history_discoveries;
  return __real_et_kernel_runtime_discover(resolver,context,runtime,error);
}
int32_t __real_et_a2_kv_cache_create_v1(uint64_t,uint64_t,uint64_t,uint64_t,
    uint64_t,et_a2_kv_cache **,et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_create_v1(uint64_t l,uint64_t b,uint64_t h,
    uint64_t c,uint64_t d,et_a2_kv_cache **out,et_kernel_error *error) {
  ++history_creates;
  return __real_et_a2_kv_cache_create_v1(l,b,h,c,d,out,error);
}
int32_t __real_et_a2_kv_cache_transaction_begin_v1(et_a2_kv_cache *,uint64_t,
    const et_kernel_tensor_view_v1 *,et_a2_kv_cache_transaction **,et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_begin_v1(et_a2_kv_cache *cache,
    uint64_t width,const et_kernel_tensor_view_v1 *counts,
    et_a2_kv_cache_transaction **out,et_kernel_error *error) {
  ++history_begins;
  return __real_et_a2_kv_cache_transaction_begin_v1(cache,width,counts,out,error);
}

typedef struct history_effects { size_t dispatches,discoveries,creates,begins; } history_effects;
static history_effects history_now(void) {
  return (history_effects){history_dispatches,history_discoveries,history_creates,history_begins};
}
static void history_no_effects(history_effects before) {
  history_effects after=history_now(); CHECK(memcmp(&before,&after,sizeof(before))==0);
}
/* Artificial epoch setup only: configure BOTH trusted mirror and mutable record.
 * This is not a claim that hostile writes to private provenance are admitted. */
static void history_set_epoch(et_g3c4_context_internal *c,uint64_t epoch) {
  cpu4_history_node *h=cpu4_history_auth(c); CHECK(h!=NULL);
  h->history.commit_epoch=c->cpu4_history.commit_epoch=epoch;
}
static void history_epoch_and_tamper(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *c=create_generator(owner);
  cpu4_history_node *h=cpu4_history_auth(c); CHECK(h!=NULL);
  CHECK(c->cpu4_history.state==1u && c->cpu4_history.count==0u);
  int64_t original=c->generator_policy[5]; c->generator_policy[5]=42;
  CHECK(et_g3c4_private_call_acquire_v1(c,2,1)!=0);
  c->generator_policy[5]=original;
  cpu4_history_record saved=c->cpu4_history;
  c->cpu4_history.count=1u;c->cpu4_history.prompt_count=1u;
  c->cpu4_history.ids[0]=7;c->cpu4_history.origin[0]=1u;
  CHECK(et_g3c4_private_call_acquire_v1(c,2,1)!=0);
  c->cpu4_history=saved;
  history_set_epoch(c,UINT64_MAX);
  OK(et_g3c4_private_call_acquire_v1(c,2,1)); /* query acquire still legal */
  float full[1024]={0},logits[256];int64_t selected=-73,successor[4];
  OK(et_g3c4_private_sample_last_v1(c,full,&selected,successor));
  history_effects before=history_now();
  memset(logits,0xa5,sizeof(logits));
  CHECK(et_g3c4_private_prefill3_v1(c,(const int64_t[3]){3,7,11},logits)!=0);
  history_no_effects(before);
  CHECK(((unsigned char *)logits)[0]==0xa5u && c->cpu4_history.commit_epoch==UINT64_MAX);
  OK(et_g3c4_private_call_abort_v1(c));
  OK(et_g3c4_private_generator_close_v1(c));
  CHECK(cpu4_history_auth(c)!=NULL && cpu4_history_zero(&c->cpu4_history,80u));
}
static void history_prefix_overflow(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2]={7,11};
  et_g3c4_context_internal *c;et_g3c4_input_internal *in;et_g3c4_output_internal *out;
  start_prefix(owner,prompt,1,-1,&c,&in,&out);
  CHECK(c->cpu4_history.count==3u && c->cpu4_history.prompt_count==2u &&
        c->cpu4_history.generated_count==1u && c->cpu4_history.commit_epoch==2u);
  CHECK(c->cpu4_history.ids[2]==c->p2g2_prefix_token && c->cpu4_history.origin[2]==2u);
  history_set_epoch(c,UINT64_MAX);
  cpu4_history_record saved=c->cpu4_history;int64_t rng[4];
  memcpy(rng,c->generator_rng_words,sizeof(rng));
  bridge_carrier carrier=new_carrier();history_effects before=history_now();
  CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(c,out,&carrier,16)!=0);
  history_no_effects(before);
  CHECK(memcmp(&saved,&c->cpu4_history,sizeof(saved))==0);
  OK(et_g3c4_private_call_abort_v1(c));
  CHECK(memcmp(&saved,&c->cpu4_history,sizeof(saved))==0 &&
        memcmp(rng,c->generator_rng_words,sizeof(rng))==0);
  et_cpu4_c4_capture *capture=et_cpu4_private_c4_capture_create_v1(c);
  CHECK(capture!=NULL);OK(et_cpu4_private_c4_capture_release_v1(capture));
  OK(et_g3c4_private_tensor_release_v1(in));OK(et_g3c4_private_output_release_v1(out));
  OK(et_g3c4_private_generator_close_v1(c));
}
static void history_full_and_replace(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2]={7,11};
  et_g3c4_context_internal *c;et_g3c4_input_internal *in;et_g3c4_output_internal *out;
  terminal_raw first,second;
  terminal_ready(owner,prompt,0,&c,&in,&out,&first,&second);
  OK(et_g3c4_private_p2g2_terminal_commit_v1(c,out,&first,9,&second,9));
  CHECK(c->cpu4_history.count==4u && c->cpu4_history.generated_count==2u &&
        c->cpu4_history.commit_epoch==3u);
  OK(et_g3c4_private_call_acquire_v1(c,2,1));
  float full[1024]={0},logits[256];int64_t token=-73,rng[4];
  OK(et_g3c4_private_sample_last_v1(c,full,&token,rng));
  history_effects before=history_now();token=-73;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(c,logits,&token)!=0);
  history_no_effects(before);CHECK(token==-73);
  OK(et_g3c4_private_prefill3_v1(c,(const int64_t[3]){3,7,11},logits));
  CHECK(c->cpu4_history.count==3u && c->cpu4_history.prompt_count==3u &&
        c->cpu4_history.generated_count==0u && c->cpu4_history.commit_epoch==4u);
  OK(et_g3c4_private_call_abort_v1(c));
  OK(et_g3c4_private_tensor_release_v1(in));OK(et_g3c4_private_output_release_v1(out));
  OK(et_g3c4_private_generator_close_v1(c));
}
/* Bitwise computation witness ONLY; not a resumed-import/history mutation API.
 * Same original model/context/policy2/RNG, remaining budget1. Replay never samples. */
static void history_replay3_witness(et_g3c4_model_owner_internal *owner,int categorical) {
  const int64_t prompt[2]={7,11};
  et_g3c4_context_internal *c;et_g3c4_input_internal *in;et_g3c4_output_internal *out;
  start_prefix(owner,prompt,categorical,-1,&c,&in,&out);
  CHECK(c->cpu4_history.count==3u && c->cpu4_history.generated_count==1u &&
        c->generator_policy[4]==2);
  int64_t ids[3],rng[4],policy[6];float prefix_logits[256],replay_logits[256];
  memcpy(ids,c->cpu4_history.ids,sizeof(ids));
  memcpy(rng,c->generator_rng_words,sizeof(rng));
  memcpy(policy,c->generator_policy,sizeof(policy));
  memcpy(prefix_logits,c->p2g2_prefix_next_logits,sizeof(prefix_logits));
  cache_snapshot prefix,replay,after_abort;
  snapshot_cache(c->cache,&prefix);
  bridge_carrier rejected=new_carrier();rejected.length=9;
  size_t samples=history_samples;
  CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(c,out,&rejected,16)!=0);
  CHECK(history_samples==samples);
  OK(et_g3c4_private_call_abort_v1(c));snapshot_cache(c->cache,&after_abort);
  CHECK(memcmp(prefix.keys,after_abort.keys,sizeof(prefix.keys))==0 &&
        memcmp(prefix.values,after_abort.values,sizeof(prefix.values))==0 &&
        memcmp(prefix.keep,after_abort.keep,sizeof(prefix.keep))==0 && prefix.length==after_abort.length);
  CHECK(memcmp(rng,c->generator_rng_words,sizeof(rng))==0 &&
        memcmp(policy,c->generator_policy,sizeof(policy))==0);
  OK(et_g3c4_private_call_acquire_v1(c,2,1));
  OK(et_g3c4_private_prefill3_v1(c,ids,replay_logits));
  snapshot_cache(c->cache,&replay);
  /* Full physical K/V arrays include unused cells; no tolerance or scalar oracle. */
  CHECK(memcmp(prefix.keys,replay.keys,sizeof(prefix.keys))==0);
  CHECK(memcmp(prefix.values,replay.values,sizeof(prefix.values))==0);
  CHECK(memcmp(prefix.keep,replay.keep,sizeof(prefix.keep))==0 && prefix.length==replay.length);
  CHECK(memcmp(prefix_logits,replay_logits,sizeof(prefix_logits))==0);
  CHECK(memcmp(rng,c->generator_rng_words,sizeof(rng))==0 && history_samples==samples);
  CHECK(memcmp(policy,c->generator_policy,sizeof(policy))==0);
  OK(et_g3c4_private_call_abort_v1(c));
  OK(et_g3c4_private_tensor_release_v1(in));OK(et_g3c4_private_output_release_v1(out));
  OK(et_g3c4_private_generator_close_v1(c));
}

#ifndef ET_CPU4_C4_HISTORY_TEST_MAIN
#define ET_CPU4_C4_HISTORY_TEST_MAIN main
#endif
int ET_CPU4_C4_HISTORY_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner=create_owner();
  history_effects before=history_now();
  cpu4_history_node *head=cpu4_history_registry;
  cpu4_history_allocation_limit=cpu4_history_successful_allocations;
  CHECK(et_g3c4_private_generator_seed_v1(owner,7,0,INT64_C(0x3f800000),
      256,INT64_C(0x3f800000),1,-1)==NULL);
  cpu4_history_allocation_limit=SIZE_MAX;
  CHECK(cpu4_history_registry==head);history_no_effects(before);
  history_replay3_witness(owner,0);history_replay3_witness(owner,1);
  history_epoch_and_tamper(owner);history_prefix_overflow(owner);history_full_and_replace(owner);
  printf("CPU4 C4 history future native controls PASS: checks=%zu\n",checks);
  return 0;
}
