#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_I2_CONSTRUCTION_PRIVATE 1
#define ET_I2_NATIVE_HELPERS_ONLY 1
#include "../../src/eshkol_transformer/m3t_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "../../native/i2_wave2_package_bridge.c"

#include <stdio.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s i2=%lld/%lld c4=%lld/%lld\n", \
  __FILE__,__LINE__,#x,(long long)et_i2_private_last_error_category_v1(), \
  (long long)et_i2_private_last_error_code_v1(), \
  (long long)et_g3c4_private_last_error_category_v1(), \
  (long long)et_g3c4_private_last_error_code_v1()); exit(1); } } while(0)
#define OK(x) CHECK((x)==0)
static unsigned char identities[8][14];

static et_g3c4_model_owner_internal *create_bound(size_t row,int64_t seed) {
  et_g3c4_model_owner_internal *owner=
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner!=NULL);
  for(size_t i=0;i<14;++i)
    OK(et_g3c4_private_model_owner_bind_v1(owner,(int64_t)i,&identities[row][i]));
  return owner;
}
static et_g3c4_model_owner_internal *create_initialized(size_t row,int64_t seed) {
  et_g3c4_model_owner_internal *owner=create_bound(row,seed);
  for(size_t i=0;i<14;++i)
    OK(et_g3c4_private_model_owner_initialize_v1(owner,(int64_t)i));
  return owner;
}
static void expect_bad(void *owner,void *parameter,void *handle) {
  et_g3c4_error_set_internal(ET_G3C4_DOMAIN_C4,ET_G3C4_INTERNAL,ET_G3C4_CODE_INVARIANT);
  CHECK(et_i2_private_g3c4_construction_parameter_preflight_v1(
      owner,parameter,handle)==ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_i2_private_last_error_category_v1()==ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_i2_private_last_error_code_v1()==ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_g3c4_private_last_error_domain_v1()==ET_G3C4_DOMAIN_I2);
  CHECK(et_g3c4_private_last_error_category_v1()==ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1()==ET_F32_TENSOR_CODE_INVALID_HANDLE);
}
static void expect_ok(et_g3c4_model_owner_internal *owner,size_t index) {
  memset(&et_i2_last_error,0xa5,sizeof(et_i2_last_error));
  et_g3c4_error_set_internal(ET_G3C4_DOMAIN_C4,ET_G3C4_INTERNAL,ET_G3C4_CODE_INVARIANT);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  OK(et_i2_private_g3c4_construction_parameter_preflight_v1(
      owner,owner->parameters[index],(void *)owner->handles[index]));
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(!et_i2_private_last_error_category_v1() && !et_i2_private_last_error_code_v1());
  CHECK(!et_g3c4_private_last_error_category_v1() && !et_g3c4_private_last_error_code_v1());
}
typedef struct held_control {
  et_f32_tensor_borrow *borrow;
  et_f32_scoped_guard_internal scoped;
  et_f32_tensor_copy_plan *copy;
  et_f32_gradient_reset_plan *reset;
  et_f32_gradient_plan *gradient;
  et_f32_tensor *numerator;
} held_control;
static void hold_control(et_g3c4_model_owner_internal *owner,size_t i,int kind,
    held_control *held) {
  et_f32_parameter *parameter=owner->parameters[i];
  et_f32_tensor *value=parameter->value,*gradient=parameter->gradient;
  et_f32_tensor_error error;
  if(kind==0) OK(et_f32_tensor_borrow_begin_v1(value,&held->borrow,&error));
  else if(kind==1) OK(et_f32_tensor_borrow_begin_v1(gradient,&held->borrow,&error));
  else if(kind==2) OK(et_f32_tensor_scoped_begin_internal(value,&held->scoped,&error));
  else if(kind==3) OK(et_f32_tensor_scoped_begin_internal(gradient,&held->scoped,&error));
  else if(kind==4) {
    et_f32_tensor_copy_assignment_v1 row={sizeof(row),value,gradient};
    OK(et_f32_tensor_copy_plan_prepare_v1(1,&row,&held->copy,&error));
  } else if(kind==5) {
    OK(et_f32_gradient_reset_plan_prepare_v1(1,&parameter,&held->reset,&error));
  } else {
    et_f32_gradient_contribution_v1 row={sizeof(row),parameter,NULL,0};
    OK(et_f32_tensor_clone_v1(value,&held->numerator,&error));
    row.weighted_numerator=held->numerator;
    OK(et_f32_gradient_plan_prepare_v1(
        1,&row,UINT32_C(0x3f800000),&held->gradient,&error));
  }
}
static void release_control(held_control *held) {
  et_f32_tensor_error error;
  OK(et_f32_tensor_borrow_end_v1(&held->borrow,&error));
  OK(et_f32_tensor_scoped_end_internal(&held->scoped));
  OK(et_f32_tensor_copy_plan_release_v1(&held->copy,&error));
  OK(et_f32_gradient_reset_plan_release_v1(&held->reset,&error));
  OK(et_f32_gradient_plan_release_v1(&held->gradient,&error));
  OK(et_f32_tensor_destroy_v1(&held->numerator,&error));
}
static void admission(void) {
  CHECK(et_i2_private_g3c4_construction_available_v1()==0);
  et_g3c4_model_owner_internal *owner=
      et_g3c4_private_model_owner_create_seeded_v1(101);
  CHECK(owner!=NULL);
  expect_ok(owner,0); /* Genuinely unbound member accepts NULL. */
  expect_bad(owner,owner->parameters[0],&identities[0][0]);
  OK(et_g3c4_private_model_owner_bind_v1(owner,0,&identities[0][0]));
  expect_bad(owner,owner->parameters[0],NULL);
  expect_ok(owner,0);
  for(size_t i=1;i<14;++i)
    OK(et_g3c4_private_model_owner_bind_v1(owner,(int64_t)i,&identities[0][i]));
  expect_bad(NULL,owner->parameters[0],(void *)owner->handles[0]);
  expect_bad(&et_i2_last_error,owner->parameters[0],(void *)owner->handles[0]);
  expect_bad(owner,&et_i2_last_error,(void *)owner->handles[0]);
  expect_bad(owner,owner->parameters[0],&identities[0][1]);
  et_f32_tensor *raw=NULL; et_f32_parameter *nonmember=NULL; et_f32_tensor_error error;
  const uint64_t shape[2]={4,4};
  OK(et_f32_tensor_create_v1(2,shape,&raw,&error));
  OK(et_f32_parameter_create_v1(raw,&nonmember,&error));
  OK(et_f32_tensor_destroy_v1(&raw,&error));
  expect_bad(owner,nonmember,NULL);
  OK(et_f32_parameter_destroy_v1(&nonmember,&error));
  et_f32_parameter *saved_parameter=owner->parameters[1];
  owner->parameters[1]=owner->parameters[0];
  expect_bad(owner,owner->parameters[0],(void *)owner->handles[0]);
  owner->parameters[1]=saved_parameter;
  const void *saved_handle=owner->handles[1];
  owner->handles[1]=owner->handles[0];
  expect_bad(owner,owner->parameters[0],(void *)owner->handles[0]);
  owner->handles[1]=saved_handle;
  OK(et_g3c4_private_model_owner_abort_v1(owner));
  expect_bad(owner,NULL,NULL); /* Authenticated ABORTED tombstone. */
}
static void lifecycle_and_controls(void) {
  et_g3c4_model_owner_internal *owner=create_initialized(1,103);
  expect_ok(owner,13);
  for(size_t index=0;index<14;++index) for(int kind=0;kind<7;++kind) {
    held_control held={0}; hold_control(owner,index,kind,&held);
    et_g3c4_model_owner_internal before=*owner;
    expect_bad(owner,owner->parameters[0],(void *)owner->handles[0]);
    CHECK(!memcmp(&before,owner,sizeof(before)));
    release_control(&held); expect_ok(owner,index);
  }
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  CHECK(owner->state==ET_G3C4_OWNER_PREPARED); expect_ok(owner,0);
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  CHECK(owner->state==ET_G3C4_OWNER_SEALED);
  expect_bad(owner,owner->parameters[0],(void *)owner->handles[0]);

  et_g3c4_model_owner_internal *staged=create_initialized(2,107);
  expect_bad(owner,owner->parameters[0],(void *)owner->handles[0]);
  expect_ok(staged,0);
  OK(et_g3c4_private_model_owner_abort_v1(staged));
}
int main(void) {
  admission(); lifecycle_and_controls();
  puts("G3-C4 I2 construction bridge: availability, lifecycle, 98 busy controls, exact diagnostics passed");
  return 0;
}
