#include "../../src/eshkol_transformer/m3t_f32_scoped.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1);}}while(0)
int main(void) {
 const uint64_t shape[]={1,2,4};et_f32_tensor *a=NULL,*b=NULL;et_f32_tensor_error e;
 CHECK(!et_f32_tensor_create_v1(3,shape,&a,&e));CHECK(!et_f32_tensor_create_v1(3,shape,&b,&e));
 et_f32_scoped_guard_internal g={0},copy={0},other={0};
 CHECK(!et_f32_tensor_scoped_begin_internal(a,&g,&e));copy=g;
 CHECK(et_f32_tensor_scoped_end_internal(&copy)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(et_f32_tensor_scoped_begin_internal(b,&copy,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(et_f32_tensor_scoped_begin_internal(a,&g,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(et_f32_tensor_scoped_begin_internal(a,&other,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 et_f32_tensor_borrow *lease=NULL;CHECK(et_f32_tensor_borrow_begin_v1(a,&lease,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(!lease);
 const et_kernel_tensor_view_v1 *v=NULL;CHECK(et_f32_tensor_borrow_view_v1((void*)&g,&v,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(!v);
 lease=(void*)&g;CHECK(et_f32_tensor_borrow_end_v1(&lease,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(lease==(void*)&g);lease=NULL;
 et_f32_tensor *slot=a;CHECK(et_f32_tensor_destroy_v1(&slot,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(slot==a);
 uint32_t bits[8]={0};CHECK(et_f32_tensor_copy_bits_from_v1(a,bits,8,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 et_f32_tensor_copy_assignment_v1 assignment={sizeof(assignment),a,b};et_f32_tensor_copy_plan *plan=NULL;
 CHECK(et_f32_tensor_copy_plan_prepare_v1(1,&assignment,&plan,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(!plan);
 CHECK(!et_f32_tensor_scoped_end_internal(&g));CHECK(!et_f32_tensor_scoped_end_internal(&g));
 CHECK(!et_f32_tensor_copy_plan_prepare_v1(1,&assignment,&plan,&e));
 CHECK(et_f32_tensor_scoped_begin_internal(a,&g,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(et_f32_tensor_scoped_begin_internal(b,&g,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(!et_f32_tensor_copy_plan_release_v1(&plan,&e));
 CHECK(!et_f32_tensor_borrow_begin_v1(a,&lease,&e));
 CHECK(et_f32_tensor_scoped_begin_internal(a,&g,&e)==ET_F32_TENSOR_ERROR_INVALID_STATE);
 CHECK(!et_f32_tensor_borrow_end_v1(&lease,&e));
 et_f32_test_retired_counts_v1 before={.struct_size=sizeof(before)},after=before;
 et_f32_test_retired_counts_snapshot_v1(&before);
 et_f32_tensor_test_fail_alloc_after_v1(0);
 for(size_t i=0;i<30000;i++) {CHECK(!et_f32_tensor_scoped_begin_internal(a,&g,&e));CHECK(!et_f32_tensor_scoped_end_internal(&g));}
 et_f32_tensor_test_reset_allocator_v1();et_f32_test_retired_counts_snapshot_v1(&after);
 CHECK(before.borrows==after.borrows);CHECK(before.retained_control_bytes==after.retained_control_bytes);
 et_f32_test_live_counts_v1 live={.struct_size=sizeof(live)};et_f32_test_live_counts_snapshot_v1(&live);CHECK(live.borrows==0);
 CHECK(!et_f32_tensor_destroy_v1(&a,&e));CHECK(!et_f32_tensor_destroy_v1(&b,&e));
 puts("m3t scoped guards: 30000 allocation-disabled cycles; zero retained-shell growth");return 0;
}
