/* Native-only authentic C4 lease fixture. No P1/Eshkol/D2 acceptance claim. */
#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_CPU4_TRAINING_PRIVATE 1
#define ET_E3_CPU4_TESTING 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "../../src/eshkol_transformer/e3_cpu4_frame.c"
#include <math.h>
#include <stdio.h>
#include "reference.h"

void et_m3t_test_k1_fail_after(size_t);
void et_m3t_test_k1_allocator_reset(void);
size_t et_e3_cpu4_test_k1_live_allocations(void);
static size_t k1_baseline;
static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr,"FAIL %d: %s E3=%lld/%lld/%lld raw=%lld/%lld/%lld\n",__LINE__,#x, \
    (long long)e3_last_domain,(long long)e3_last_category,(long long)e3_last_code, \
    (long long)e3_raw_domain,(long long)e3_raw_category,(long long)e3_raw_code); exit(1); \
} } while (0)
#define OK(x) CHECK((x)==0)
#define NO(x) CHECK((x)!=0)
static unsigned char identities[2][14], foreign;
typedef struct image {
  et_f32_parameter *parameters[14]; const void *handles[14];
  uint32_t values[14][1024], gradients[14][1024];
  et_f32_gradient_metadata_v1 metadata[14]; int64_t successor[4];
} image;
static void capture(et_g3c4_model_owner_internal *owner,image *out) {
  memset(out,0,sizeof(*out)); et_f32_tensor_error error;
  memcpy(out->parameters,owner->parameters,sizeof(out->parameters));
  memcpy(out->handles,owner->handles,sizeof(out->handles));
  memcpy(out->successor,owner->successor,sizeof(out->successor));
  for (size_t i=0;i<14;++i) {
    OK(et_f32_tensor_copy_bits_to_v1(et_g3c4_parameter_value(owner,i),out->values[i],parameter_counts[i],&error));
    OK(et_f32_tensor_copy_bits_to_v1(owner->parameters[i]->gradient,out->gradients[i],parameter_counts[i],&error));
    out->metadata[i].struct_size=sizeof(out->metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(owner->parameters[i],&out->metadata[i],&error));
  }
}
static void unchanged(et_g3c4_model_owner_internal *owner,const image *before) {
  image after; capture(owner,&after); CHECK(memcmp(before,&after,sizeof(after))==0);
}
static et_g3c4_model_owner_internal *make_model(size_t index,int present) {
  et_g3c4_model_owner_internal *owner=et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner!=NULL); et_f32_tensor_error error;
  for (int64_t i=0;i<14;++i) OK(et_g3c4_private_model_owner_bind_v1(owner,i,&identities[index][i]));
  for (int64_t i=0;i<14;++i) {
    OK(et_g3c4_private_model_owner_initialize_v1(owner,i));
    OK(et_f32_tensor_copy_bits_from_v1(et_g3c4_parameter_value(owner,(size_t)i),parameter_bits[i],parameter_counts[i],&error));
    if (present) {
      uint32_t gradient[1024];
      for (size_t j=0;j<parameter_counts[i];++j) gradient[j]=UINT32_C(0x3e800000)+(uint32_t)(j+17*(size_t)i);
      et_f32_parameter_test_set_gradient_bits_v1(owner->parameters[i],gradient,parameter_counts[i]);
      et_f32_parameter_test_set_metadata_v1(owner->parameters[i],ET_F32_GRADIENT_PRESENT,3,UINT32_C(0x40400000));
    }
  }
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner)); return owner;
}
static void destinations_bits(et_f32_tensor *dest[4],uint32_t words[4]) {
  et_f32_tensor_error error;
  for (size_t i=0;i<4;++i) OK(et_f32_tensor_copy_bits_to_v1(dest[i],&words[i],1,&error));
}
static void assert_idle(void *lease,et_e3_frame_internal *frame) {
  CHECK(frame->phase==E3_IDLE && frame->work==NULL);
  CHECK(et_e3_cpu4_test_k1_live_allocations()==k1_baseline);
  OK(et_g3c4_private_training_lease_check_v1(lease));
  OK(et_g3c4_private_training_lease_enter_v1(lease,1));
  OK(et_g3c4_private_training_lease_leave_v1(lease));
  for (e3_cpu4_workspace *w=e3_workspaces;w;w=w->next) {
    CHECK(!w->live && !w->data && !w->frame && !w->lease);
    CHECK(!w->workspace_child && !w->pin_child && e3_pins_idle(&w->pins));
  }
}
static void tensor_close(et_f32_tensor *value,const double *expected,size_t count) {
  uint32_t words[1024];et_f32_tensor_error error;
  OK(et_f32_tensor_copy_bits_to_v1(value,words,count,&error));
  for (size_t i=0;i<count;++i) {
    float actual; memcpy(&actual,&words[i],4);
    CHECK(isfinite(actual) && fabs((double)actual-expected[i]) <= 4e-5+5e-4*fabs(expected[i]));
  }
}
static void stage(et_e3_frame_internal *frame) {
  int64_t ids[4]={3,197,3,41},targets[4]={197,3,41,7};
  unsigned char mask[4]={1,1,1,0};
  et_kernel_tensor_view_v1 view=e3_native_view(ids,sizeof(ids),"i64",2,e3_shape_pair);
  /* Invalid token rejects before any input copy and remains retryable. */
  ids[3]=256;NO(et_e3_private_stage_inputs_v1(frame,&view));
  CHECK(frame->phase==E3_ACQUIRED && frame->stage_plane==0);
  ids[3]=41;OK(et_e3_private_stage_inputs_v1(frame,&view));
  view=e3_native_view(targets,sizeof(targets),"i64",2,e3_shape_pair);
  OK(et_e3_private_stage_targets_v1(frame,&view));
  view=e3_native_view(mask,sizeof(mask),"bool",2,e3_shape_pair);
  OK(et_e3_private_stage_mask_v1(frame,&view));
}
static void forward(et_e3_frame_internal *frame) {
  stage(frame);
  NO(et_e3_private_forward_role_v1(frame,1));
  for (int64_t role=0;role<21;++role) {
    OK(et_e3_private_forward_role_v1(frame,role));
    tensor_close(frame->work->data->forward[role],expected_roles[role],role_counts[role]);
  }
  OK(et_e3_private_ce_v1(frame)); tensor_close(frame->work->data->ce,expected_ce,4);
  OK(et_e3_private_reduce_v1(frame));OK(et_e3_private_correct_v1(frame));
  OK(et_e3_private_accumulate_v1(frame));OK(et_e3_private_finalize_v1(frame));
  for (size_t i=0;i<4;++i) tensor_close(frame->work->data->result[i],&expected_metrics[i],1);
}
static void abort_call(et_e3_frame_internal *frame) {
  OK(et_e3_private_cleanup_preflight_v1(frame));et_e3_private_drain_v1(frame);et_e3_private_finish_v1(frame);
}
static void run(int present) {
  et_g3c4_model_owner_internal *owner=make_model((size_t)present,present);
  image before;capture(owner,&before);void *lease=et_g3c4_private_training_lease_create_v1(owner);CHECK(lease);
  et_f32_tensor *dest[4];et_f32_tensor_error error;
  for (size_t i=0;i<4;++i) {
    OK(et_f32_tensor_create_v1(0,NULL,&dest[i],&error));uint32_t sentinel=UINT32_C(0x3f000000)+(uint32_t)i;
    OK(et_f32_tensor_copy_bits_from_v1(dest[i],&sentinel,1,&error));
  }
  CHECK(!et_e3_private_cpu4_frame_create_v1(&foreign,dest[0],dest[1],dest[2],dest[3]));
  CHECK(!et_e3_private_cpu4_frame_create_v1(owner,dest[0],dest[1],dest[2],dest[3]));
  CHECK(!et_e3_private_cpu4_frame_create_v1(lease,dest[0],dest[0],dest[2],dest[3]));
  CHECK(!et_e3_private_cpu4_frame_create_v1(lease,et_g3c4_parameter_value(owner,10),dest[1],dest[2],dest[3]));
  et_e3_frame_internal *frame=et_e3_private_cpu4_frame_create_v1(lease,dest[0],dest[1],dest[2],dest[3]);CHECK(frame);
  et_f32_test_live_counts_v1 baseline={.struct_size=sizeof(baseline)};
  et_f32_test_live_counts_snapshot_v1(&baseline);
  k1_baseline=et_e3_cpu4_test_k1_live_allocations();
  uint32_t published[4],after[4];destinations_bits(dest,published);
  NO(et_e3_private_acquire_v1(&foreign));
  /* Busy existing READ rejects without allocating another workspace. */
  OK(et_g3c4_private_training_lease_enter_v1(lease,1));NO(et_e3_private_acquire_v1(frame));
  CHECK(frame->work==NULL);OK(et_g3c4_private_training_lease_leave_v1(lease));
  OK(et_e3_private_acquire_v1(frame));NO(et_e3_private_acquire_v1(frame));
  NO(et_g3c4_private_training_lease_leave_v1(lease));
  forward(frame);
  /* Cleanup refusal preserves enrollment; ending the borrow permits retry. */
  et_f32_tensor_borrow *borrow=NULL;
  OK(et_f32_tensor_borrow_begin_v1(frame->work->data->forward[0],&borrow,&error));
  NO(et_e3_private_cleanup_preflight_v1(frame));CHECK(!frame->cleanup_ready);
  NO(et_g3c4_private_training_lease_leave_v1(lease));
  OK(et_f32_tensor_borrow_end_v1(&borrow,&error));
  abort_call(frame);assert_idle(lease,frame);unchanged(owner,&before);
  destinations_bits(dest,after);CHECK(!memcmp(after,published,sizeof(after)));
  OK(et_e3_private_acquire_v1(frame));forward(frame);
  OK(et_e3_private_cleanup_preflight_v1(frame));et_e3_private_drain_v1(frame);
  NO(et_g3c4_private_training_lease_leave_v1(lease));
  CHECK(e3_raw_category==0);
  /* Late scalar rejection is byte-atomic across all four destinations. */
  OK(et_f32_tensor_borrow_begin_v1(dest[3],&borrow,&error));
  NO(et_e3_private_publish_v1(frame));
  CHECK(e3_last_domain==0 && e3_last_category==E3_INVALID_STATE && e3_last_code==E3_CODE_DESTINATION);
  CHECK(e3_raw_domain==3 && e3_raw_category==ET_F32_TENSOR_ERROR_INVALID_STATE && e3_raw_code==ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  destinations_bits(dest,after);CHECK(!memcmp(after,published,sizeof(after)));
  OK(et_f32_tensor_borrow_end_v1(&borrow,&error));
  OK(et_e3_private_publish_v1(frame));et_e3_private_finish_v1(frame);
  CHECK(e3_raw_domain==3 && e3_raw_category==ET_F32_TENSOR_ERROR_INVALID_STATE && e3_raw_code==ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  assert_idle(lease,frame);unchanged(owner,&before);
  destinations_bits(dest,published);CHECK(frame->published_counts[0]==3 && frame->published_counts[1]==1);
  for (size_t i=0;i<4;++i) tensor_close(dest[i],&expected_metrics[i],1);
  /* Exhaust local control/payload/37 tensor-create denial boundaries. */
  for (size_t budget=0;budget<39;++budget) {
    et_e3_cpu4_test_fail_after(budget);NO(et_e3_private_acquire_v1(frame));
    CHECK(e3_last_category==E3_INTERNAL && e3_last_code==E3_CODE_ALLOCATION);
    et_e3_cpu4_test_fail_after(SIZE_MAX);assert_idle(lease,frame);
    destinations_bits(dest,after);CHECK(!memcmp(after,published,sizeof(after)));
    CHECK(et_e3_private_selected_metric_bits_ref_v1(frame,0)==published[0]);
    CHECK(et_e3_private_selected_metric_bits_ref_v1(frame,1)==published[1]);
    CHECK(et_e3_private_counter_ref_v1(frame,0)==3 && et_e3_private_counter_ref_v1(frame,1)==1);
    unchanged(owner,&before);
  }
  /* Both real child-enrollment allocation failures unwind their own prefixes. */
  for (size_t child=0;child<2;++child) {
    et_g3c4_allocation_limit=et_g3c4_successful_allocations+child;
    NO(et_e3_private_acquire_v1(frame));
    CHECK(e3_last_domain==0 && e3_last_category==E3_INTERNAL && e3_last_code==E3_CODE_ALLOCATION);
    CHECK(e3_raw_domain==4 && e3_raw_category==ET_G3C4_INTERNAL && e3_raw_code==ET_G3C4_CODE_ALLOCATION);
    et_g3c4_allocation_limit=SIZE_MAX;assert_idle(lease,frame);
    destinations_bits(dest,after);CHECK(!memcmp(after,published,sizeof(after)));unchanged(owner,&before);
  }
  et_f32_tensor_test_fail_alloc_after_v1(0);NO(et_e3_private_acquire_v1(frame));
  CHECK(e3_raw_domain==3 && e3_raw_code==ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  et_f32_tensor_test_reset_allocator_v1();assert_idle(lease,frame);
  et_m3t_test_k1_fail_after(0);NO(et_e3_private_acquire_v1(frame));
  CHECK(e3_raw_domain==1 && e3_raw_category==ET_KERNEL_ERROR_INTERNAL && e3_raw_code==ET_KERNEL_CODE_ALLOCATION_FAILED);
  et_m3t_test_k1_allocator_reset();assert_idle(lease,frame);
  /* A failure after a complete earlier invocation preserves the prior report. */
  OK(et_e3_private_acquire_v1(frame));stage(frame);
  for (int64_t role=0;role<21;++role) OK(et_e3_private_forward_role_v1(frame,role));
  frame->work->data->targets[3]=256;NO(et_e3_private_ce_v1(frame));
  int64_t raw[3]={e3_raw_domain,e3_raw_category,e3_raw_code};
  abort_call(frame);CHECK(e3_raw_domain==raw[0] && e3_raw_category==raw[1] && e3_raw_code==raw[2]);
  destinations_bits(dest,after);CHECK(!memcmp(after,published,sizeof(after)));assert_idle(lease,frame);
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame,0)==published[0]);
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame,1)==published[1]);
  CHECK(et_e3_private_counter_ref_v1(frame,0)==3 && et_e3_private_counter_ref_v1(frame,1)==1);
  OK(et_e3_private_acquire_v1(frame));forward(frame);OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);OK(et_e3_private_publish_v1(frame));et_e3_private_finish_v1(frame);
  assert_idle(lease,frame);unchanged(owner,&before);
  et_f32_test_live_counts_v1 live={.struct_size=sizeof(live)};et_f32_test_live_counts_snapshot_v1(&live);
  CHECK(!memcmp(&live,&baseline,sizeof(live)));
  /* Explicit safe stale-frame destruction never dereferences/releases dead lease. */
  OK(et_g3c4_private_training_lease_prepare_release_v1(lease));OK(et_g3c4_private_training_lease_commit_release_v1(lease));
  NO(et_e3_private_frame_check_v1(frame));OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);et_e3_private_frame_destroy_v1(frame);
  for (size_t i=0;i<4;++i) OK(et_f32_tensor_destroy_v1(&dest[i],&error));
}
int main(void) {
  /* Reject valid category paired with unrelated known code, retaining raw. */
  e3_raw_category=0;CHECK(e3_dependency(3,ET_F32_TENSOR_ERROR_INVALID_STATE,ET_F32_TENSOR_CODE_ALLOCATION_FAILED)==E3_INTERNAL);
  CHECK(e3_last_domain==0 && e3_last_code==E3_CODE_INVARIANT && e3_raw_category==ET_F32_TENSOR_ERROR_INVALID_STATE);
  et_g3c4_error_set_internal(0,ET_G3C4_INTERNAL,ET_G3C4_CODE_ALLOCATION);
  e3_raw_category=0;CHECK(e3_c4_error()==E3_INTERNAL);CHECK(e3_last_code==E3_CODE_ALLOCATION && e3_raw_domain==4);
  et_g3c4_error_set_internal(0,ET_G3C4_INVALID_STATE,ET_G3C4_CODE_ALLOCATION);
  e3_raw_category=0;CHECK(e3_c4_error()==E3_INTERNAL && e3_last_code==E3_CODE_INVARIANT);
  run(0);run(1);
  size_t workspace_controls=0,pin_controls=0,child_records=0,live_children=0,workspace_children=0,pin_children=0;
  for (e3_cpu4_workspace *w=e3_workspaces;w;w=w->next) {
    ++workspace_controls; ++pin_controls;
    CHECK(!w->live && !w->data && e3_pins_idle(&w->pins));
  }
  for (et_g3c4_training_lease_internal *lease=cpu4_training_leases;lease;lease=lease->next)
    for (cpu4_training_child *child=lease->children;child;child=child->next) {
      ++child_records;live_children+=(size_t)child->live;
      workspace_children+=(size_t)(child->kind==2);pin_children+=(size_t)(child->kind==5);
    }
  CHECK(live_children==0);
  printf("CPU4-E3-NATIVE-PASS checks=%zu roles=21 logits=1024 gradients=absent,present scratch=per-call "
         "workspace-controls=%zu pin-subobjects=%zu workspace-control-bytes=%zu pin-bytes-included=%zu "
         "lease-child-records=%zu lease-child-bytes=%zu registered-workspaces=%zu registered-pins=%zu live-children=%zu live-payloads=0 "
         "payload-tensors-per-call=37 k1-live-baseline=%zu\n",checks,
         workspace_controls,pin_controls,workspace_controls*sizeof(e3_cpu4_workspace),
         pin_controls*sizeof(et_g3c4_model_pins_internal),child_records,
         child_records*sizeof(cpu4_training_child),workspace_children,pin_children,live_children,k1_baseline);
  return 0;
}
