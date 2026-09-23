#include "test_cases.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include "f32_parameter_internal.h"
#include <errno.h>

static int deny_allocation;static size_t forbidden_calls;
void *__real_malloc(size_t);void *__real_calloc(size_t,size_t);void *__real_realloc(void *,size_t);void *__real_aligned_alloc(size_t,size_t);int __real_posix_memalign(void **,size_t,size_t);void __real_free(void *);
void *__wrap_malloc(size_t n){if(deny_allocation){++forbidden_calls;return NULL;}return __real_malloc(n);}
void *__wrap_calloc(size_t n,size_t s){if(deny_allocation){++forbidden_calls;return NULL;}return __real_calloc(n,s);}
void *__wrap_realloc(void *p,size_t n){if(deny_allocation){++forbidden_calls;return NULL;}return __real_realloc(p,n);}
void *__wrap_aligned_alloc(size_t a,size_t n){if(deny_allocation){++forbidden_calls;return NULL;}return __real_aligned_alloc(a,n);}
int __wrap_posix_memalign(void **p,size_t a,size_t n){if(deny_allocation){++forbidden_calls;return ENOMEM;}return __real_posix_memalign(p,a,n);}
void __wrap_free(void *p){if(deny_allocation){++forbidden_calls;return;}__real_free(p);}

typedef struct owner{et_f32_tensor *f;et_i64_tensor *i;et_f32_tensor_borrow *fb;et_i64_tensor_borrow *ib;const et_kernel_tensor_view_v1 *view;et_kernel_tensor_view_v1 local;unsigned char bytes[16];}owner;

static void create(owner *o,const et_kernel_tensor_view_v1 *v){
  et_f32_tensor_error fe;et_i64_tensor_error ie;memset(o,0,sizeof(*o));
  if(strcmp(v->dtype,"f32")==0){
    CHECK(et_f32_tensor_create_v1(v->rank,v->shape,&o->f,&fe)==0);uint32_t bits[2048];CHECK(v->byte_length<=sizeof(bits));memcpy(bits,v->data,v->byte_length);CHECK(et_f32_tensor_copy_bits_from_v1(o->f,bits,v->byte_length/4u,&fe)==0);CHECK(et_f32_tensor_borrow_begin_v1(o->f,&o->fb,&fe)==0);CHECK(et_f32_tensor_borrow_view_v1(o->fb,&o->view,&fe)==0);
    et_f32_tensor *attempt=o->f;CHECK(et_f32_tensor_destroy_v1(&attempt,&fe)==ET_F32_TENSOR_ERROR_INVALID_STATE);CHECK(fe.code==ET_F32_TENSOR_CODE_ACTIVE_BORROW);CHECK(attempt==o->f);CHECK(et_f32_tensor_copy_bits_from_v1(o->f,bits,v->byte_length/4u,&fe)==ET_F32_TENSOR_ERROR_INVALID_STATE);
  }else if(strcmp(v->dtype,"i64")==0){
    CHECK(et_i64_tensor_create_v1(v->rank,v->shape,&o->i,&ie)==0);CHECK(et_i64_tensor_copy_from_v1(o->i,v->data,v->byte_length/8u,&ie)==0);CHECK(et_i64_tensor_borrow_begin_v1(o->i,&o->ib,&ie)==0);CHECK(et_i64_tensor_borrow_view_v1(o->ib,&o->view,&ie)==0);
    et_i64_tensor *attempt=o->i;CHECK(et_i64_tensor_destroy_v1(&attempt,&ie)==ET_I64_TENSOR_ERROR_INVALID_STATE);CHECK(ie.code==ET_I64_TENSOR_CODE_ACTIVE_BORROW);CHECK(attempt==o->i);CHECK(et_i64_tensor_copy_from_v1(o->i,v->data,v->byte_length/8u,&ie)==ET_I64_TENSOR_ERROR_INVALID_STATE);
  }else{
    CHECK(strcmp(v->dtype,"bool")==0);CHECK(v->byte_length<=sizeof(o->bytes));memcpy(o->bytes,v->data,v->byte_length);o->local=*v;o->local.data=o->bytes;o->view=&o->local;
  }
  CHECK(o->view!=NULL);CHECK(o->view->data!=v->data);CHECK(o->view->rank==v->rank);CHECK(o->view->byte_length==v->byte_length);
}

static void destroy(owner *o){
  et_f32_tensor_error fe;et_i64_tensor_error ie;
  if(o->f){CHECK(et_f32_tensor_borrow_end_v1(&o->fb,&fe)==0);CHECK(o->fb==NULL);CHECK(et_f32_tensor_destroy_v1(&o->f,&fe)==0);CHECK(o->f==NULL);CHECK(et_f32_tensor_destroy_v1(&o->f,&fe)==0);}
  else if(o->i){CHECK(et_i64_tensor_borrow_end_v1(&o->ib,&ie)==0);CHECK(o->ib==NULL);CHECK(et_i64_tensor_destroy_v1(&o->i,&ie)==0);CHECK(o->i==NULL);CHECK(et_i64_tensor_destroy_v1(&o->i,&ie)==0);}memset(o,0,sizeof(*o));
}

static void rows(et_kernel_runtime *rt){
  const et_kernel_provider_v1 *p=et_g3c4_kernel_provider_v1(),*again;const et_kernel_provider_v1 descriptor=*p;const et_kernel_capability_v1 *caps=p->capabilities;et_kernel_capability_v1 metadata[7];CHECK(p->capability_count==COUNT(metadata));memcpy(metadata,caps,sizeof(metadata));
  char before_report[32768],after_report[32768];size_t required;et_kernel_error e;CHECK(et_kernel_runtime_report_json(rt,before_report,sizeof(before_report),&required,&e)==0);size_t cases=0;
  for(size_t repetition=0;repetition<3u;++repetition)for(size_t ci=0;ci<p->capability_count;++ci)for(size_t oi=0;oi<caps[ci].operation_count;++oi)for(size_t ri=0;ri<caps[ci].shape_range_count;++ri){
    fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);setup(f,&caps[ci],oi,ri);run(rt,&f->call);owner inputs[6],output;et_kernel_tensor_view_v1 iv[6],ov[1];const size_t ni=f->call.input_count;
    for(size_t i=0;i<ni;++i){create(&inputs[i],&f->in[i]);iv[i]=*inputs[i].view;}storage sentinel;memset(&sentinel,0x5a,sizeof(sentinel));et_kernel_tensor_view_v1 v=f->out[0];v.data=&sentinel;create(&output,&v);ov[0]=*output.view;et_kernel_call_v1 call=kc(caps[ci].name,&f->r,iv,ni,ov,1u);
    ++f->row[0];deny_allocation=1;int32_t absent=et_kernel_runtime_dispatch(rt,&call,&e);deny_allocation=0;CHECK(absent==ET_KERNEL_ERROR_UNSUPPORTED);CHECK(e.code==ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);--f->row[0];for(size_t i=0;i<ov[0].byte_length;++i)CHECK(((unsigned char *)ov[0].data)[i]==0x5a);
    deny_allocation=1;int32_t admitted=et_kernel_runtime_dispatch(rt,&call,&e);deny_allocation=0;CHECK(admitted==0);CHECK(memcmp(ov[0].data,f->out[0].data,ov[0].byte_length)==0);
    memset(ov[0].data,0x5a,ov[0].byte_length);deny_allocation=1;again=et_g3c4_kernel_provider_v1();int32_t valid=again->validate_call(&call,&e);deny_allocation=0;CHECK(again==p);CHECK(valid==0);for(size_t i=0;i<ov[0].byte_length;++i)CHECK(((unsigned char *)ov[0].data)[i]==0x5a);
    deny_allocation=1;again->invoke_call(&call);deny_allocation=0;CHECK(memcmp(ov[0].data,f->out[0].data,ov[0].byte_length)==0);
    memset(ov[0].data,0x5a,ov[0].byte_length);deny_allocation=1;CHECK(et_kernel_runtime_dispatch(rt,&call,&e)==0);deny_allocation=0;CHECK(forbidden_calls==0);CHECK(memcmp(ov[0].data,f->out[0].data,ov[0].byte_length)==0);
    for(size_t i=0;i<ni;++i)CHECK(memcmp(iv[i].data,f->in[i].data,iv[i].byte_length)==0);destroy(&output);for(size_t i=0;i<ni;++i)destroy(&inputs[i]);free(f);++cases;
  }
  CHECK(cases==84u);CHECK(memcmp(&descriptor,p,sizeof(descriptor))==0);CHECK(memcmp(metadata,caps,sizeof(metadata))==0);CHECK(et_kernel_runtime_report_json(rt,after_report,sizeof(after_report),&required,&e)==0);CHECK(strcmp(before_report,after_report)==0);CHECK(forbidden_calls==0);
}

int main(void){
  deny_allocation=1;const et_kernel_provider_v1 *p=et_g3c4_kernel_provider_v1();deny_allocation=0;CHECK(p!=NULL);CHECK(forbidden_calls==0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_v1 before={.struct_size=sizeof(before)},after={.struct_size=sizeof(after)};et_f32_test_live_counts_snapshot_v1(&before);
#endif
  et_kernel_runtime *rt=runtime_new();rows(rt);et_kernel_runtime_destroy(rt);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_snapshot_v1(&after);CHECK(memcmp(&before,&after,sizeof(before))==0);
#endif
  printf("G3-C4-N I1/I2 integration: %zu checks\n",checks);return 0;
}
