#define ET_M3T_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#include "../../src/eshkol_transformer/m3t_transport.c"
#include <stdio.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); \
} } while (0)

static owner *sealed_owner(int64_t seed, unsigned char handles[14]) {
  initializer *original=et_m3t_private_initializer_create_v1(seed);
  CHECK(original!=NULL);
  CHECK(et_m3t_private_initializer_seal_v1(original)==0);
  owner *o=et_m3t_private_owner_create_v1(original);
  CHECK(o!=NULL);
  for(size_t i=0;i<14;i++)
    CHECK(et_m3t_private_owner_bind_v1(o,(int64_t)i,&handles[i])==0);
  for(size_t i=0;i<14;i++)
    CHECK(et_m3t_private_owner_initialize_v1(o,(int64_t)i)==0);
  CHECK(et_m3t_private_owner_seal_v1(o)==0);
  CHECK(et_m3t_private_initializer_seal_v1(o->successor)==0);
  return o;
}

static void expect_preflight(owner *o, void *original, void *successor,
                             int64_t expected) {
  owner before=*o;
  et_f32_test_live_counts_v1 live_before={.struct_size=sizeof(live_before)};
  et_f32_test_live_counts_v1 live_after=live_before;
  et_f32_test_retired_counts_v1 retired_before={.struct_size=sizeof(retired_before)};
  et_f32_test_retired_counts_v1 retired_after=retired_before;
  et_f32_test_live_counts_snapshot_v1(&live_before);
  et_f32_test_retired_counts_snapshot_v1(&retired_before);
  CHECK(et_m3t_private_candidate_retire_preflight_v1(o,original,successor)==expected);
  et_f32_test_live_counts_snapshot_v1(&live_after);
  et_f32_test_retired_counts_snapshot_v1(&retired_after);
  CHECK(memcmp(&before,o,sizeof(before))==0);
  CHECK(memcmp(&live_before,&live_after,sizeof(live_before))==0);
  CHECK(memcmp(&retired_before,&retired_after,sizeof(retired_before))==0);
}

int main(void) {
  unsigned char handles_a[14]={0},handles_b[14]={0};
  owner *a=sealed_owner(1729,handles_a);
  owner *b=sealed_owner(1730,handles_b);
  initializer *original=a->original,*successor=a->successor;
  expect_preflight(a,original,successor,0);
  expect_preflight(a,original,successor,0);
  expect_preflight(a,b->original,successor,ET_M3T_INVALID_STATE);
  expect_preflight(a,original,b->successor,ET_M3T_INVALID_STATE);
  expect_preflight(a,successor,original,ET_M3T_INVALID_STATE);
  expect_preflight(a,original,(void *)(uintptr_t)1,ET_M3T_INVALID_ARGUMENT);
  CHECK(et_m3t_private_candidate_retire_preflight_v1((void *)(uintptr_t)1,
        original,successor)==ET_M3T_INVALID_ARGUMENT);
  expect_preflight(b,original,successor,ET_M3T_INVALID_STATE);
  initializer *stale=et_m3t_private_initializer_create_v1(1731);
  CHECK(stale!=NULL);
  CHECK(et_m3t_private_initializer_abort_v1(stale)==0);
  expect_preflight(a,stale,successor,ET_M3T_INVALID_STATE);
  initializer *construction_seed=et_m3t_private_initializer_create_v1(1732);
  CHECK(construction_seed!=NULL);
  CHECK(et_m3t_private_initializer_seal_v1(construction_seed)==0);
  owner *retired=et_m3t_private_owner_create_v1(construction_seed);
  CHECK(retired!=NULL);
  CHECK(et_m3t_private_owner_abort_v1(retired)==0);
  CHECK(et_m3t_private_candidate_retire_preflight_v1(retired,
        construction_seed,retired->successor)==ET_M3T_INVALID_STATE);

  a->original=b->original;
  expect_preflight(a,b->original,successor,ET_M3T_INVALID_STATE);
  a->original=original;
  a->successor=b->successor;
  expect_preflight(a,original,b->successor,ET_M3T_INVALID_STATE);
  a->successor=successor;
  et_f32_parameter *p=a->p[0];void *h=a->handles[0];
  a->p[0]=b->p[0];a->handles[0]=b->handles[0];
  expect_preflight(a,original,successor,ET_M3T_INVALID_STATE);
  a->p[0]=p;a->handles[0]=h;
  a->handles[0]=b->handles[0];
  expect_preflight(a,original,successor,ET_M3T_INVALID_STATE);
  a->handles[0]=h;

  workspace *w=et_m3t_private_workspace_create_v1(a);
  CHECK(w!=NULL);
  expect_preflight(a,original,successor,ET_M3T_INVALID_STATE);
  input *in=et_m3t_private_input_create_v1();CHECK(in!=NULL);
  CHECK(et_m3t_private_input_copy_v1(in,7,8)==0);
  CHECK(et_m3t_private_workspace_begin_v1(w,in,0)==0);
  expect_preflight(a,original,successor,ET_M3T_INVALID_STATE);
  CHECK(et_m3t_private_workspace_reset_v1(w)==0);
  expect_preflight(a,original,successor,ET_M3T_INVALID_STATE);
  CHECK(et_m3t_private_workspace_release_v1(w)==0);
  CHECK(et_m3t_private_input_release_v1(in)==0);
  expect_preflight(a,original,successor,0);

  et_f32_tensor_error e;
  et_f32_tensor_borrow *borrow=NULL;
  CHECK(et_f32_parameter_value_borrow_begin_v1(p,&borrow,&e)==0);
  expect_preflight(a,original,successor,ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow,&e)==0);
  CHECK(et_f32_tensor_borrow_begin_v1(
        (et_f32_tensor *)et_f32_parameter_test_gradient_tensor_v1(p),
        &borrow,&e)==0);
  expect_preflight(a,original,successor,ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow,&e)==0);

  const et_f32_tensor *v=NULL;
  CHECK(et_f32_parameter_value_tensor_v1(p,&v,&e)==0);
  et_f32_tensor *dst=NULL;
  CHECK(et_f32_tensor_create_v1(2,parameter_shape[0],&dst,&e)==0);
  et_f32_tensor_copy_assignment_v1 assignment={sizeof(assignment),dst,v};
  et_f32_tensor_copy_plan *plan=NULL;
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1,&assignment,&plan,&e)==0);
  expect_preflight(a,original,successor,ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan,&e)==0);
  CHECK(et_f32_tensor_destroy_v1(&dst,&e)==0);

  et_f32_parameter *foreign=b->p[1];
  CHECK(et_f32_parameter_test_replace_child_v1(p,0,
        (et_f32_tensor *)et_f32_parameter_test_gradient_tensor_v1(foreign))==0);
  expect_preflight(a,original,successor,ET_F32_TENSOR_ERROR_INTERNAL);
  CHECK(et_f32_parameter_test_replace_child_v1(p,0,(et_f32_tensor *)v)==0);
  expect_preflight(a,original,successor,0);

  et_m3t_test_fail_alloc_after(0);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  expect_preflight(a,original,successor,0);
  et_f32_tensor_test_reset_allocator_v1();
  et_m3t_test_fail_alloc_after(SIZE_MAX);
  printf("G3-R M3T parameter preflight PASS: %u checks\n",checks);
  return 0;
}
