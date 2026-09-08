#include "test_cases.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include "f32_parameter_internal.h"

typedef struct owner {
  et_f32_tensor *f;
  et_i64_tensor *i;
  et_f32_tensor_borrow *fb;
  et_i64_tensor_borrow *ib;
  const et_kernel_tensor_view_v1 *view;
} owner;
static void create(owner *o,const et_kernel_tensor_view_v1 *v) {
  et_f32_tensor_error fe;et_i64_tensor_error ie;memset(o,0,sizeof(*o));
  if(strcmp(v->dtype,"f32")==0) {
    CHECK(et_f32_tensor_create_v1(v->rank,v->shape,&o->f,&fe)==0);
    uint32_t bits[1024];CHECK(v->byte_length<=sizeof(bits));memcpy(bits,v->data,v->byte_length);
    CHECK(et_f32_tensor_copy_bits_from_v1(o->f,bits,v->byte_length/4u,&fe)==0);
    CHECK(et_f32_tensor_borrow_begin_v1(o->f,&o->fb,&fe)==0);
    CHECK(et_f32_tensor_borrow_view_v1(o->fb,&o->view,&fe)==0);
    et_f32_tensor *attempt=o->f;
    CHECK(et_f32_tensor_destroy_v1(&attempt,&fe)!=0);CHECK(attempt==o->f);
    CHECK(et_f32_tensor_copy_bits_from_v1(o->f,bits,v->byte_length/4u,&fe)!=0);
  } else {
    CHECK(et_i64_tensor_create_v1(v->rank,v->shape,&o->i,&ie)==0);
    CHECK(et_i64_tensor_copy_from_v1(o->i,v->data,v->byte_length/8u,&ie)==0);
    CHECK(et_i64_tensor_borrow_begin_v1(o->i,&o->ib,&ie)==0);
    CHECK(et_i64_tensor_borrow_view_v1(o->ib,&o->view,&ie)==0);
    et_i64_tensor *attempt=o->i;
    CHECK(et_i64_tensor_destroy_v1(&attempt,&ie)!=0);CHECK(attempt==o->i);
    CHECK(et_i64_tensor_copy_from_v1(o->i,v->data,v->byte_length/8u,&ie)!=0);
  }
  CHECK(o->view!=NULL);CHECK(o->view->data!=v->data);
  CHECK(o->view->rank==v->rank);CHECK(o->view->byte_length==v->byte_length);
}
static void destroy(owner *o) {
  et_f32_tensor_error fe;et_i64_tensor_error ie;
  if(o->f) {
    CHECK(et_f32_tensor_borrow_end_v1(&o->fb,&fe)==0);CHECK(o->fb==NULL);
    CHECK(et_f32_tensor_destroy_v1(&o->f,&fe)==0);CHECK(o->f==NULL);
    CHECK(et_f32_tensor_destroy_v1(&o->f,&fe)==0);
  } else {
    CHECK(et_i64_tensor_borrow_end_v1(&o->ib,&ie)==0);CHECK(o->ib==NULL);
    CHECK(et_i64_tensor_destroy_v1(&o->i,&ie)==0);CHECK(o->i==NULL);
    CHECK(et_i64_tensor_destroy_v1(&o->i,&ie)==0);
  }
  o->view=NULL;
}
static void carrier_rows(et_kernel_runtime *rt) {
  const et_kernel_provider_v1 *p=et_n3k_kernel_provider_v1();
  const et_kernel_capability_v1 *caps=p->capabilities;
  fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);size_t cases=0;
  for(size_t ci=0;ci<p->capability_count;++ci) for(size_t oi=0;oi<caps[ci].operation_count;++oi)
    for(size_t ri=0;ri<caps[ci].shape_range_count;++ri) {
      setup(f,&caps[ci],oi,ri);run(rt,&f->call);
      owner inputs[3],outputs[3];et_kernel_tensor_view_v1 iv[3],ov[3];
      const size_t ni=f->call.input_count,no=f->call.output_count;
      for(size_t i=0;i<ni;++i) {create(&inputs[i],&f->in[i]);iv[i]=*inputs[i].view;}
      for(size_t i=0;i<no;++i) {
        storage sentinels;memset(&sentinels,0x5a,sizeof(sentinels));
        et_kernel_tensor_view_v1 v=f->out[i];v.data=&sentinels;
        create(&outputs[i],&v);ov[i]=*outputs[i].view;
      }
      et_kernel_call_v1 call=kc(caps[ci].name,&f->r,iv,ni,ov,no);
      /* A malformed request fails before writes with every carrier borrow live. */
      ++f->row[0];et_kernel_error e;CHECK(et_kernel_runtime_dispatch(rt,&call,&e)!=0);--f->row[0];
      for(size_t i=0;i<no;++i) {
        const unsigned char *b=ov[i].data;
        for(size_t j=0;j<ov[i].byte_length;++j) CHECK(b[j]==0x5a);
      }
      run(rt,&call);
      for(size_t i=0;i<ni;++i) {
        CHECK(iv[i].data==inputs[i].view->data);
        CHECK(memcmp(iv[i].data,f->in[i].data,iv[i].byte_length)==0);
      }
      for(size_t i=0;i<no;++i) {
        CHECK(ov[i].data==outputs[i].view->data);
        CHECK(memcmp(ov[i].data,f->out[i].data,ov[i].byte_length)==0);
      }
      for(size_t i=0;i<no;++i) destroy(&outputs[i]);
      for(size_t i=0;i<ni;++i) destroy(&inputs[i]);++cases;
    }
  CHECK(cases==31u);free(f);
}
static et_f32_tensor *filled(const uint64_t shape[2],float value) {
  et_f32_tensor *t=NULL;et_f32_tensor_error e;uint32_t bits[1024];
  for(size_t i=0;i<1024;++i) bits[i]=fbits(value);
  CHECK(et_f32_tensor_create_v1(2,shape,&t,&e)==0);
  CHECK(et_f32_tensor_copy_bits_from_v1(t,bits,1024,&e)==0);return t;
}
static void tied_contribution(et_kernel_runtime *rt) {
  const uint64_t shape[]={256,4};et_f32_tensor_error e;
  et_f32_tensor *weight=filled(shape,0.125f),*head=filled(shape,0.25f),*embedding=filled(shape,0.5f),*sum=filled(shape,-1.0f);
  et_f32_parameter *parameter=NULL;
  CHECK(et_f32_parameter_create_v1(weight,&parameter,&e)==0);
  static const int identity=1;CHECK(et_f32_parameter_bind_identity_v1(parameter,&identity,&e)==0);
  et_f32_tensor_borrow *hb=NULL,*eb=NULL,*sb=NULL;const et_kernel_tensor_view_v1 *hv,*ev,*sv;
  CHECK(et_f32_tensor_borrow_begin_v1(head,&hb,&e)==0);CHECK(et_f32_tensor_borrow_view_v1(hb,&hv,&e)==0);
  CHECK(et_f32_tensor_borrow_begin_v1(embedding,&eb,&e)==0);CHECK(et_f32_tensor_borrow_view_v1(eb,&ev,&e)==0);
  CHECK(et_f32_tensor_borrow_begin_v1(sum,&sb,&e)==0);CHECK(et_f32_tensor_borrow_view_v1(sb,&sv,&e)==0);
  et_kernel_tensor_view_v1 in[]={*hv,*ev},out[]={*sv};
  et_kernel_request_v1 r=rq("n3k.sum2.forward",2,shape);et_kernel_call_v1 c=kc("n3k.tied-sum",&r,in,2,out,1);run(rt,&c);
  CHECK(et_f32_tensor_borrow_end_v1(&hb,&e)==0);CHECK(et_f32_tensor_borrow_end_v1(&eb,&e)==0);CHECK(et_f32_tensor_borrow_end_v1(&sb,&e)==0);
  et_f32_gradient_contribution_v1 contributions[]={{sizeof(contributions[0]),parameter,sum,0},{sizeof(contributions[0]),parameter,sum,0}};
  et_f32_gradient_plan *plan=NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(2,contributions,fbits(1.0f),&plan,&e)!=0);CHECK(plan==NULL);
  CHECK(et_f32_gradient_plan_prepare_v1(1,contributions,fbits(1.0f),&plan,&e)==0);
  CHECK(et_f32_gradient_plan_commit_v1(plan,&e)==0);
  CHECK(et_f32_gradient_plan_release_v1(&plan,&e)==0);CHECK(plan==NULL);
  et_f32_gradient_metadata_v1 meta={sizeof(meta),0,0,0};CHECK(et_f32_parameter_gradient_metadata_v1(parameter,&meta,&e)==0);
  CHECK(meta.contribution_count==1);CHECK(meta.normalization_weight_bits==fbits(1.0f));
  et_f32_tensor *gradient=NULL;CHECK(et_f32_parameter_gradient_snapshot_v1(parameter,&gradient,&e)==0);
  uint32_t bits[1024];CHECK(et_f32_tensor_copy_bits_to_v1(gradient,bits,1024,&e)==0);
  for(size_t i=0;i<1024;++i) CHECK(bits[i]==fbits(0.75f));
  CHECK(et_f32_tensor_destroy_v1(&gradient,&e)==0);
  CHECK(et_f32_parameter_destroy_v1(&parameter,&e)==0);
  CHECK(et_f32_tensor_destroy_v1(&sum,&e)==0);CHECK(et_f32_tensor_destroy_v1(&embedding,&e)==0);
  CHECK(et_f32_tensor_destroy_v1(&head,&e)==0);CHECK(et_f32_tensor_destroy_v1(&weight,&e)==0);
}
int main(void) {
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_v1 before,after;et_f32_test_live_counts_snapshot_v1(&before);
#endif
  et_kernel_runtime *rt=runtime_new();carrier_rows(rt);tied_contribution(rt);et_kernel_runtime_destroy(rt);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_snapshot_v1(&after);
  if(memcmp(&before,&after,sizeof(before))!=0) {fprintf(stderr,"N3K carrier resource baseline mismatch\n");return 1;}
#endif
  printf("N3K I1/I2 integration: %zu checks\n",checks);return 0;
}
