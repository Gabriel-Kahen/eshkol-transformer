/* Development-only intrusive tests: genuine I2 owners, test-local identity anchors.
 * Public P1 identity and source-composed admission are exercised by the AOT lane.
 * Including the single replacement TU exposes controls for adversarial checks;
 * no test observer or corruption authority exists in the normal artifact. */
#include <stdlib.h>
static size_t allocation_calls, release_calls;
static int allocation_disabled;
static void *tracked_calloc(size_t n,size_t bytes) {
  ++allocation_calls; return allocation_disabled ? NULL : calloc(n,bytes);
}
static void *tracked_malloc(size_t bytes) {
  ++allocation_calls; return allocation_disabled ? NULL : malloc(bytes);
}
static void *tracked_realloc(void *p,size_t bytes) {
  ++allocation_calls; return allocation_disabled ? NULL : realloc(p,bytes);
}
static void tracked_free(void *p) { ++release_calls; free(p); }
#define malloc tracked_malloc
#define realloc tracked_realloc
#define calloc tracked_calloc
#define free tracked_free
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#undef malloc
#undef realloc
#undef calloc
#undef free
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", \
  __FILE__, __LINE__, #x); exit(1); } } while (0)
#define OK(x) CHECK((x) == 0)
#define FULL UINT16_C(0x3fff)
static const uint64_t shapes[14][2] = {
  {4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},{4,0},{4,0},
  {256,4},{4,0},{4,0},{4,4}
};
typedef struct fixture {
  et_g3c4_model_owner_internal *owner;
  et_f32_parameter *p[14];
  const void *identities[14];
  unsigned char anchors[14];
  et_g3c4_model_pins_internal pins;
  uint32_t value[14][1024], gradient[14][1024];
  et_f32_gradient_metadata_v1 metadata[14];
} fixture;
static fixture f;
static et_f32_tensor_error error;

static size_t elements(size_t i) { return shapes[i][0] * (shapes[i][1] ? shapes[i][1] : 1); }
static void expect(int rc, int category, int code, const char *operation) {
  if(rc!=category || error.code!=(uint32_t)code) fprintf(stderr,"got %d/%u %s; wanted %d/%d %s\n",rc,error.code,error.operation,category,code,operation);
  CHECK(rc == category); CHECK(error.category == (uint32_t)category);
  CHECK(error.code == (uint32_t)code); CHECK(!strcmp(error.operation, operation));
  CHECK(error.message[0]);
}
static void empty(void) {
  CHECK(!f.pins.self && !f.pins.held_mask);
  for (size_t i=0; i<14; ++i) {
    const et_kernel_tensor_view_v1 *v=&f.pins.views[i];
    CHECK(!f.pins.parameters[i] && !f.pins.identities[i] && !f.pins.values[i]);
    CHECK(!v->struct_size && !v->data && !v->byte_length && !v->dtype && !v->device &&
          !v->layout && !v->offset_bytes && !v->rank && !v->shape);
    CHECK(!f.p[i]->plan_pins && !f.p[i]->value->active_borrow);
    CHECK(!f.p[i]->gradient->active_borrow && !f.p[i]->value->plan_pins &&
          !f.p[i]->gradient->plan_pins);
  }
}
static void snapshot(void) {
  for (size_t i=0; i<14; ++i) {
    memcpy(f.value[i],f.p[i]->value->data,elements(i)*4);
    memcpy(f.gradient[i],f.p[i]->gradient->data,elements(i)*4);
    f.metadata[i].struct_size=sizeof(f.metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(f.p[i],&f.metadata[i],&error));
  }
}
static void preserved(void) {
  for (size_t i=0; i<14; ++i) {
    et_f32_gradient_metadata_v1 m={.struct_size=sizeof(m)};
    CHECK(!memcmp(f.value[i],f.p[i]->value->data,elements(i)*4));
    CHECK(!memcmp(f.gradient[i],f.p[i]->gradient->data,elements(i)*4));
    OK(et_f32_parameter_gradient_metadata_v1(f.p[i],&m,&error));
    CHECK(m.state==f.metadata[i].state && m.contribution_count==f.metadata[i].contribution_count &&
          m.normalization_weight_bits==f.metadata[i].normalization_weight_bits);
  }
}
static void setup(void) {
  allocation_disabled=1; CHECK(!tracked_malloc(1)); CHECK(!tracked_realloc(NULL,1));
  allocation_disabled=0;
  f.owner = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(f.owner != NULL);
  for (size_t i=0; i<14; ++i) {
    f.p[i] = et_g3c4_private_model_owner_parameter_v1(f.owner, (int64_t)i);
    CHECK(f.p[i] != NULL);
    f.identities[i]=&f.anchors[i];
    OK(et_g3c4_private_model_owner_bind_v1(
        f.owner, (int64_t)i, (void *)f.identities[i]));
  }
  for (size_t i=0; i<14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(f.owner, (int64_t)i));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(f.owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(f.owner));
  snapshot(); empty();
}
static void begin(void) {
  CHECK(f.owner->active==NULL);
  memset(&error,0xa5,sizeof(error));
  OK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error));
  CHECK(!error.category && !error.code && !error.operation[0] && !error.message[0]);
  CHECK(f.pins.self==&f.pins && f.pins.held_mask==FULL);
  size_t bytes=0;
  for(size_t i=0;i<14;++i) {
    CHECK(f.pins.parameters[i]==f.p[i] && f.pins.identities[i]==f.identities[i]);
    CHECK(f.pins.values[i]==f.p[i]->value && f.p[i]->plan_pins==1);
    CHECK(f.p[i]->value->active_borrow==(void *)&f.pins.views[i]);
    CHECK(f.pins.views[i].byte_length==elements(i)*4); bytes+=f.pins.views[i].byte_length;
  }
  CHECK(bytes==4768); OK(et_g3c4_model_pins_check_internal(&f.pins,&error));
  CHECK(f.owner->active==NULL);
}
static void end(void) {
  m3_call_test_release_count=0;
  et_g3c4_model_pins_end_internal(&f.pins);
  CHECK(m3_call_test_release_count==14);
  for(size_t i=0;i<14;++i) CHECK(m3_call_test_release_order[i]==13-i);
  empty(); et_g3c4_model_pins_end_internal(&f.pins); empty();
  CHECK(f.owner->active==NULL);
}
static void admission(void) {
  et_f32_parameter *saved=f.p[0]; const void *identity=f.identities[0];
#define BAD_BEGIN(c,k) do { expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),c,k,"g3c4-model-pins-begin"); } while(0)
  expect(et_g3c4_model_pins_begin_internal(NULL,f.identities,&f.pins,&error),1,1,"g3c4-model-pins-begin");
  expect(et_g3c4_model_pins_begin_internal(f.p,NULL,&f.pins,&error),1,1,"g3c4-model-pins-begin");
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,NULL,&error),1,1,"g3c4-model-pins-begin");
  f.p[0]=NULL; BAD_BEGIN(1,1); f.p[0]=saved;
  f.p[0]=(et_f32_parameter *)(uintptr_t)1; BAD_BEGIN(4,7); f.p[0]=saved;
  f.identities[0]=NULL; BAD_BEGIN(1,1); f.identities[0]=identity;
  f.identities[0]=f.identities[1]; BAD_BEGIN(1,7); f.identities[0]=identity;
  f.p[0]=f.p[1]; f.identities[0]=f.identities[1]; BAD_BEGIN(1,7);
  f.p[0]=saved; f.identities[0]=identity;
  et_f32_tensor *shared_value=f.p[1]->value;
  f.p[1]->value=f.p[0]->value; BAD_BEGIN(1,7); f.p[1]->value=shared_value;
  et_f32_tensor *wrong=NULL; et_f32_parameter *other=NULL;
  unsigned char other_identity=0; const uint64_t shape[]={2,8};
  OK(et_f32_tensor_create_v1(2,shape,&wrong,&error));
  OK(et_f32_parameter_create_v1(wrong,&other,&error));
  OK(et_f32_parameter_bind_identity_v1(other,&other_identity,&error));
  f.p[0]=other; f.identities[0]=&other_identity; BAD_BEGIN(2,3);
  f.p[0]=saved; f.identities[0]=identity;
  et_f32_parameter *stale=other; OK(et_f32_parameter_destroy_v1(&other,&error));
  f.p[0]=stale; BAD_BEGIN(4,7); f.p[0]=saved;
  OK(et_f32_tensor_destroy_v1(&wrong,&error));
  f.pins.held_mask=1; BAD_BEGIN(4,8); f.pins.held_mask=0;
  et_f32_tensor_borrow *borrow=NULL;
  OK(et_f32_parameter_value_borrow_begin_v1(f.p[13],&borrow,&error));
  BAD_BEGIN(4,8); CHECK(!f.pins.self && !f.pins.held_mask);
  OK(et_f32_tensor_borrow_end_v1(&borrow,&error));
  et_f32_gradient_reset_plan *reset=NULL;
  OK(et_f32_gradient_reset_plan_prepare_v1(14,f.p,&reset,&error));
  BAD_BEGIN(4,8); OK(et_f32_gradient_reset_plan_release_v1(&reset,&error));
  empty(); preserved();
#undef BAD_BEGIN
}
static void geometry_admission(void) {
#define BAD_SHAPE() do { \
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error), \
         2,3,"g3c4-model-pins-begin"); \
  empty(); \
} while(0)
  for(size_t i=0;i<14;++i) for(size_t carrier=0;carrier<2;++carrier) {
    et_f32_tensor *tensor=carrier?f.p[i]->gradient:f.p[i]->value;
    const size_t rank=shapes[i][1]?2:1;
    size_t saved=tensor->rank; tensor->rank=0; BAD_SHAPE(); tensor->rank=saved;
    saved=tensor->element_count; ++tensor->element_count;
    BAD_SHAPE(); tensor->element_count=saved;
    saved=tensor->byte_length; tensor->byte_length-=sizeof(float);
    BAD_SHAPE(); tensor->byte_length=saved;
    for(size_t dimension=0;dimension<rank;++dimension) {
      uint64_t extent=tensor->shape[dimension]; ++tensor->shape[dimension];
      BAD_SHAPE(); tensor->shape[dimension]=extent;
    }
    for(size_t dimension=0;dimension<rank;++dimension) {
      size_t stride=tensor->strides[dimension]; ++tensor->strides[dimension];
      BAD_SHAPE(); tensor->strides[dimension]=stride;
    }
  }
  empty(); preserved();
#undef BAD_SHAPE
}
static void geometry_separation(void) {
  et_g3t_model_pins_internal c2_pins={0};
  expect(et_g3t_model_pins_begin_internal(
      f.p,f.identities,&c2_pins,&error),2,3,"m3-call-pins-begin");
  et_f32_tensor *initial=NULL;
  et_f32_parameter *c2_position=NULL;
  const uint64_t c2_shape[2]={2,4};
  unsigned char c2_identity=0;
  OK(et_f32_tensor_create_v1(2,c2_shape,&initial,&error));
  OK(et_f32_parameter_create_v1(initial,&c2_position,&error));
  OK(et_f32_parameter_bind_identity_v1(
      c2_position,&c2_identity,&error));
  OK(et_f32_tensor_destroy_v1(&initial,&error));
  et_f32_parameter *saved_parameter=f.p[13];
  const void *saved_identity=f.identities[13];
  f.p[13]=c2_position; f.identities[13]=&c2_identity;
  OK(et_g3t_model_pins_begin_internal(
      f.p,f.identities,&c2_pins,&error));
  et_g3t_model_pins_end_internal(&c2_pins);
  expect(et_g3c4_model_pins_begin_internal(
      f.p,f.identities,&f.pins,&error),2,3,"g3c4-model-pins-begin");
  f.p[13]=saved_parameter; f.identities[13]=saved_identity;
  OK(et_f32_parameter_destroy_v1(&c2_position,&error));
  empty(); preserved();
}
typedef struct held_control {
  et_f32_tensor_borrow *borrow;
  et_f32_scoped_guard_internal scoped;
  et_f32_tensor_copy_plan *copy;
  et_f32_gradient_reset_plan *reset;
  et_f32_gradient_plan *gradient;
  et_f32_tensor *numerator;
} held_control;
static void hold_control(size_t index,int kind,held_control *held) {
  et_f32_parameter *parameter=f.p[index];
  et_f32_tensor *value=parameter->value,*gradient=parameter->gradient;
  if(kind==0) OK(et_f32_tensor_borrow_begin_v1(value,&held->borrow,&error));
  else if(kind==1) OK(et_f32_tensor_borrow_begin_v1(gradient,&held->borrow,&error));
  else if(kind==2) OK(et_f32_tensor_scoped_begin_internal(value,&held->scoped,&error));
  else if(kind==3) OK(et_f32_tensor_scoped_begin_internal(gradient,&held->scoped,&error));
  else if(kind==4) {
    et_f32_tensor_copy_assignment_v1 assignment={sizeof(assignment),value,gradient};
    OK(et_f32_tensor_copy_plan_prepare_v1(1,&assignment,&held->copy,&error));
  } else if(kind==5) {
    OK(et_f32_gradient_reset_plan_prepare_v1(1,&parameter,&held->reset,&error));
  } else {
    et_f32_gradient_contribution_v1 contribution={sizeof(contribution),parameter,NULL,0};
    OK(et_f32_tensor_clone_v1(value,&held->numerator,&error));
    contribution.weighted_numerator=held->numerator;
    OK(et_f32_gradient_plan_prepare_v1(
        1,&contribution,UINT32_C(0x3f800000),&held->gradient,&error));
  }
}
static void release_control(int kind,held_control *held) {
  if(kind<=1) OK(et_f32_tensor_borrow_end_v1(&held->borrow,&error));
  else if(kind<=3) OK(et_f32_tensor_scoped_end_internal(&held->scoped));
  else if(kind==4) OK(et_f32_tensor_copy_plan_release_v1(&held->copy,&error));
  else if(kind==5) OK(et_f32_gradient_reset_plan_release_v1(&held->reset,&error));
  else {
    OK(et_f32_gradient_plan_release_v1(&held->gradient,&error));
    OK(et_f32_tensor_destroy_v1(&held->numerator,&error));
  }
}
static void busy_admission(void) {
  for(size_t index=0;index<14;++index) for(int kind=0;kind<7;++kind) {
    held_control held={0}; hold_control(index,kind,&held);
    expect(et_g3c4_model_pins_begin_internal(
        f.p,f.identities,&f.pins,&error),4,8,"g3c4-model-pins-begin");
    CHECK(!f.pins.self && !f.pins.held_mask);
    release_control(kind,&held); empty(); preserved();
  }
}
static void spans(void) {
  et_g3c4_model_pins_internal before=f.pins;
  void *overflow=(void *)(uintptr_t)(UINTPTR_MAX-15);
  memset(&error,0xa5,sizeof(error)); et_f32_tensor_error overflow_error=error;
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,overflow,&error)==1);
  CHECK(!memcmp(&overflow_error,&error,sizeof(error)));
  CHECK(et_g3c4_model_pins_begin_internal(overflow,f.identities,&f.pins,&error)==1);
  CHECK(!memcmp(&overflow_error,&error,sizeof(error)));
  CHECK(et_g3c4_model_pins_begin_internal(f.p,overflow,&f.pins,&error)==1);
  CHECK(!memcmp(&overflow_error,&error,sizeof(error)));
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,overflow)==1);
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,NULL)==1);
  CHECK(et_g3c4_model_pins_check_internal(&f.pins,overflow)==1);
  CHECK(et_g3c4_model_pins_check_internal(&f.pins,NULL)==1);
  CHECK(!memcmp(&before,&f.pins,sizeof(before)));
  expect(et_g3c4_model_pins_begin_internal(f.p,(void *)f.p,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,(void *)&f.pins)==1);
  CHECK(!memcmp(&before,&f.pins,sizeof(before)));
  et_f32_parameter *saved[14]; memcpy(saved,f.p,sizeof(saved));
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,(void *)f.p)==1);
  CHECK(!memcmp(saved,f.p,sizeof(saved)));
  expect(et_g3c4_model_pins_begin_internal(f.pins.parameters,f.identities,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  expect(et_g3c4_model_pins_begin_internal((void *)((char *)f.p+1),f.identities,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,(void *)f.p[10]->value->data,&error),1,4,"g3c4-model-pins-begin");
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,(void *)f.p[10]->value->data)==1);
  et_f32_tensor *foreign=NULL;
  OK(et_f32_tensor_create_v1(2,shapes[0],&foreign,&error));
  float *data=f.p[0]->gradient->data;
  f.p[0]->gradient->data=foreign->data;
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  f.p[0]->gradient->data=data;
  uint64_t *shape=f.p[0]->gradient->shape;
  f.p[0]->gradient->shape=foreign->shape;
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  f.p[0]->gradient->shape=shape;
  size_t *strides=f.p[0]->gradient->strides;
  f.p[0]->gradient->strides=foreign->strides;
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),1,4,"g3c4-model-pins-begin");
  f.p[0]->gradient->strides=strides;
  memset(&error,0xa5,sizeof(error)); et_f32_tensor_error saved_error=error;
  f.p[0]->gradient->data=(void *)&error;
  CHECK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error)==1);
  CHECK(!memcmp(&saved_error,&error,sizeof(error))); f.p[0]->gradient->data=data;
  OK(et_f32_tensor_destroy_v1(&foreign,&error));
  empty(); preserved();
  begin(); before=f.pins;
  CHECK(et_g3c4_model_pins_check_internal(&f.pins,(void *)&f.pins)==1);
  CHECK(!memcmp(&before,&f.pins,sizeof(before))); end();
}
static void acquisition_failures(void) {
  for(size_t n=0;n<14;++n) {
    m3_call_test_fail_after=n; m3_call_test_release_count=0;
    et_f32_tensor_test_fail_alloc_after_v1(0);
    size_t allocations=allocation_calls, releases=release_calls; allocation_disabled=1;
    expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),6,6,"g3c4-model-pins-begin");
    CHECK(m3_call_test_release_count==n);
    for(size_t i=0;i<n;++i) CHECK(m3_call_test_release_order[i]==n-1-i);
    empty(); preserved();
    CHECK(allocation_calls==allocations && release_calls==releases); allocation_disabled=0;
    et_f32_tensor_test_reset_allocator_v1();
  }
  m3_call_test_fail_after=SIZE_MAX;
}
static void exclusions(int present) {
  et_f32_tensor *source=NULL;
  OK(et_f32_tensor_clone_v1(f.p[0]->value,&source,&error));
  begin();
  expect(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error),4,8,"g3c4-model-pins-begin");
  et_f32_parameter *slot=f.p[0];
  CHECK(et_f32_parameter_destroy_v1(&slot,&error)==4 && slot==f.p[0]);
  et_f32_tensor_borrow *borrow=NULL;
  CHECK(et_f32_parameter_value_borrow_begin_v1(f.p[0],&borrow,&error)==4 && !borrow);
  et_f32_gradient_reset_plan *reset=NULL;
  CHECK(et_f32_gradient_reset_plan_prepare_v1(14,f.p,&reset,&error)==4 && !reset);
  et_f32_gradient_contribution_v1 row={sizeof(row),f.p[0],source,present?1:0};
  et_f32_gradient_plan *plan=NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(1,&row,UINT32_C(0x3f800000),&plan,&error)==4 && !plan);
  et_f32_tensor_borrow *sentinel=(void *)&f.pins.views[0];
  CHECK(et_f32_tensor_borrow_end_v1(&sentinel,&error)==4 && sentinel==(void *)&f.pins.views[0]);
  if(present) {
    /* This succeeds: plan_pins is deliberately NOT a universal gradient-read lock. */
    OK(et_f32_parameter_gradient_borrow_begin_v1(f.p[0],&borrow,&error));
    expect(et_g3c4_model_pins_check_internal(&f.pins,&error),4,7,"g3c4-model-pins-check");
    OK(et_f32_tensor_borrow_end_v1(&borrow,&error));
  }
  OK(et_g3c4_model_pins_check_internal(&f.pins,&error)); end(); preserved();
  OK(et_f32_tensor_destroy_v1(&source,&error));
}
/* Each child owns its corruption. Verify abort happens before any control drain,
 * including a defect at index0 which a release-as-you-validate loop finds last. */
static int abort_pipe;
static et_f32_parameter *abort_parameters[14];
static size_t abort_counts[14];
static et_f32_tensor *abort_values[14], *abort_gradients[14];
static size_t abort_allocations, abort_releases;
static et_f32_tensor_borrow *abort_sentinels[14];
static void on_abort(int sig) {
  (void)sig; unsigned char intact=allocation_calls==abort_allocations && release_calls==abort_releases;
  for(size_t i=0;i<14;++i)
    if(abort_parameters[i]->plan_pins!=abort_counts[i] ||
       abort_values[i]->active_borrow!=abort_sentinels[i]) intact=0;
  if(write(abort_pipe,&intact,1)!=1) _exit(92);
}
static void corrupt(size_t kind) {
  et_kernel_tensor_view_v1 *v=&f.pins.views[0];
  switch(kind) {
    case 0:f.pins.self=(void *)1;break;
    case 1:f.pins.held_mask=FULL^1;break;
    case 2:f.p[0]->value->active_borrow=NULL;break;
    case 3:f.p[0]->plan_pins=2;break;
    case 4:f.p[0]->value->plan_pins=1;break;
    case 5:f.p[0]->gradient->plan_pins=1;break;
    case 6:f.p[0]->gradient->active_borrow=(void *)1;break;
    case 7:v->struct_size=0;break;
    case 8:v->data=(void *)1;break;
    case 9:v->byte_length--;break;
    case 10:v->dtype=(void *)1;break;
    case 11:v->device=(void *)1;break;
    case 12:v->layout=0;break;
    case 13:v->offset_bytes=4;break;
    case 14:v->rank=64;break;
    case 15:v->shape=(void *)1;break;
    case 16:f.pins.identities[0]=f.identities[1];break;
    case 17:f.pins.parameters[0]=(void *)1;break;
    case 18:f.pins.values[0]=(void *)1;break;
    case 19:f.p[0]->gradient=(void *)1;break;
    case 20:f.p[0]->value->shape[0]++;break;
    case 21:f.p[0]->value->strides[0]++;break;
    case 22:f.pins.held_mask=UINT16_C(0xffff);break;
    default:CHECK(0);
  }
}
static void failstop(void) {
  for(size_t kind=0;kind<25;++kind) {
    int pipefd[2]; CHECK(!pipe(pipefd)); pid_t child=fork(); CHECK(child>=0);
    if(!child) {
      close(pipefd[0]); abort_pipe=pipefd[1];
      struct rlimit no_core={0,0}; CHECK(!setrlimit(RLIMIT_CORE,&no_core));
      begin();
      for(size_t i=0;i<14;++i) {
        abort_parameters[i]=f.p[i]; abort_values[i]=f.p[i]->value; abort_gradients[i]=f.p[i]->gradient;
      }
      if(kind<23) {
        corrupt(kind);
        et_g3c4_model_pins_internal pin_before;
        et_f32_parameter parameters_before[14];
        et_f32_tensor values_before[14], gradients_before[14];
        uint64_t shape_before[14][2][2]; size_t stride_before[14][2][2];
        memcpy(&pin_before,&f.pins,sizeof(pin_before));
        for(size_t i=0;i<14;++i) {
          memcpy(&parameters_before[i],abort_parameters[i],sizeof(parameters_before[i]));
          memcpy(&values_before[i],abort_values[i],sizeof(values_before[i]));
          memcpy(&gradients_before[i],abort_gradients[i],sizeof(gradients_before[i]));
          et_f32_tensor *tensors[]={abort_values[i],abort_gradients[i]};
          for(size_t k=0;k<2;++k) {
            memcpy(shape_before[i][k],tensors[k]->shape,tensors[k]->rank*sizeof(uint64_t));
            memcpy(stride_before[i][k],tensors[k]->strides,tensors[k]->rank*sizeof(size_t));
          }
        }
        size_t allocations=allocation_calls, releases=release_calls;
        int rc=et_g3c4_model_pins_check_internal(&f.pins,&error);
        CHECK(rc==4 && error.code==7);
        CHECK(allocation_calls==allocations && release_calls==releases);
        CHECK(!memcmp(&pin_before,&f.pins,sizeof(pin_before)));
        for(size_t i=0;i<14;++i) {
          CHECK(!memcmp(&parameters_before[i],abort_parameters[i],sizeof(parameters_before[i])));
          CHECK(!memcmp(&values_before[i],abort_values[i],sizeof(values_before[i])));
          CHECK(!memcmp(&gradients_before[i],abort_gradients[i],sizeof(gradients_before[i])));
          CHECK(!memcmp(f.value[i],abort_values[i]->data,elements(i)*4));
          CHECK(!memcmp(f.gradient[i],abort_gradients[i]->data,elements(i)*4));
          et_f32_tensor *tensors[]={abort_values[i],abort_gradients[i]};
          for(size_t k=0;k<2;++k) {
            CHECK(!memcmp(shape_before[i][k],tensors[k]->shape,tensors[k]->rank*sizeof(uint64_t)));
            CHECK(!memcmp(stride_before[i][k],tensors[k]->strides,tensors[k]->rank*sizeof(size_t)));
          }
        }
      }
      for(size_t i=0;i<14;++i) {
        abort_counts[i]=abort_parameters[i]->plan_pins;
        abort_sentinels[i]=abort_values[i]->active_borrow;
      }
      CHECK(signal(SIGABRT,on_abort)!=SIG_ERR);
      et_f32_tensor_test_fail_alloc_after_v1(0); allocation_disabled=1;
      abort_allocations=allocation_calls; abort_releases=release_calls;
      if(kind==23) {et_g3c4_model_pins_internal copy=f.pins;et_g3c4_model_pins_end_internal(&copy);}
      else et_g3c4_model_pins_end_internal(kind==24?NULL:&f.pins);
      _exit(93);
    }
    close(pipefd[1]); unsigned char intact=0; CHECK(read(pipefd[0],&intact,1)==1);
    close(pipefd[0]); int status; CHECK(waitpid(child,&status,0)==child);
    CHECK(intact && WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
    empty(); preserved();
  }
}
static void present_gradients(void) {
  et_f32_gradient_contribution_v1 rows[14]; et_f32_gradient_plan *plan=NULL;
  et_f32_tensor *sources[14]={0};
  for(size_t i=0;i<14;++i) {
    OK(et_f32_tensor_clone_v1(f.p[i]->value,&sources[i],&error));
    rows[i]=(et_f32_gradient_contribution_v1){sizeof(rows[i]),f.p[i],sources[i],0};
  }
  OK(et_f32_gradient_plan_prepare_v1(14,rows,UINT32_C(0x40000000),&plan,&error));
  OK(et_f32_gradient_plan_commit_v1(plan,&error));
  OK(et_f32_gradient_plan_release_v1(&plan,&error));
  for(size_t i=0;i<14;++i) OK(et_f32_tensor_destroy_v1(&sources[i],&error));
  snapshot();
  for(size_t i=0;i<14;++i) CHECK(f.metadata[i].state==1 &&
    f.metadata[i].contribution_count==1 && f.metadata[i].normalization_weight_bits==UINT32_C(0x40000000));
}
static void retention(size_t cycles) {
  et_f32_test_live_counts_v1 live={.struct_size=sizeof(live)},after_live=live;
  et_f32_test_retired_counts_v1 retired={.struct_size=sizeof(retired)},after_retired=retired;
  et_f32_test_borrow_event_counts_v1 events={.struct_size=sizeof(events)},after_events=events;
  et_f32_test_live_counts_snapshot_v1(&live); et_f32_test_retired_counts_snapshot_v1(&retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&events);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  size_t allocations=allocation_calls, releases=release_calls; allocation_disabled=1;
  for(size_t n=0;n<cycles;++n) {
    OK(et_g3c4_model_pins_begin_internal(f.p,f.identities,&f.pins,&error));
    OK(et_g3c4_model_pins_check_internal(&f.pins,&error));
    m3_call_test_release_count=0; et_g3c4_model_pins_end_internal(&f.pins);
  }
  CHECK(!successful_allocations && allocation_calls==allocations && release_calls==releases);
  allocation_disabled=0; et_f32_tensor_test_reset_allocator_v1();
  et_f32_test_live_counts_snapshot_v1(&after_live); et_f32_test_retired_counts_snapshot_v1(&after_retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&after_events);
  CHECK(!memcmp(&live,&after_live,sizeof(live)));
  CHECK(!memcmp(&retired,&after_retired,sizeof(retired)));
  CHECK(!memcmp(&events,&after_events,sizeof(events))); empty(); preserved();
  printf("g3c4 pins: %zu cycles, live=%zu tensors/%zu parameters, retired=%zu bytes unchanged, no borrow events\n",
         cycles,live.tensors,live.parameters,retired.retained_control_bytes);
}
int main(void) {
  setup();
  expect(et_g3c4_model_pins_check_internal(&f.pins,&error),4,7,"g3c4-model-pins-check");
  et_g3c4_model_pins_end_internal(&f.pins);
  admission(); geometry_admission(); geometry_separation(); busy_admission(); spans();
  acquisition_failures(); exclusions(0); failstop();
  retention(1024); retention(8192);
  present_gradients(); acquisition_failures(); exclusions(1); retention(1024); retention(8192);
  et_f32_test_live_counts_v1 live={.struct_size=sizeof(live)};
  et_f32_test_live_counts_snapshot_v1(&live);
  CHECK(live.tensors==28 && live.parameters==14 && !live.borrows &&
        !live.copy_plans && !live.gradient_plans && !live.reset_plans);
  puts("g3c4 pins: genuine owner, geometry split, 98 busy controls, shared-value rejection, 172 geometry defects, 14 failure prefixes, 25 fail-stop cases, absent/present gradient preservation passed");
  return 0;
}
