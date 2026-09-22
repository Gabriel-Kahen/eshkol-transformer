#include "test_common.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
typedef struct owned {
  et_f32_tensor *tensor;
  et_f32_tensor_borrow *borrow;
  et_kernel_tensor_view_v1 view;
} owned;
static owned create_f32(size_t rank,const uint64_t *shape,const float *initial,size_t count) {
  owned o={0};et_f32_tensor_error e;uint32_t words[512];CHECK(count<=512);
  for(size_t i=0;i<count;i++)words[i]=initial?bits(initial[i]):0xa5a5a5a5u;
  CHECK(et_f32_tensor_create_v1(rank,shape,&o.tensor,&e)==0);
  CHECK(et_f32_tensor_copy_bits_from_v1(o.tensor,words,count,&e)==0);
  CHECK(et_f32_tensor_borrow_begin_v1(o.tensor,&o.borrow,&e)==0);
  const et_kernel_tensor_view_v1 *v=NULL;
  CHECK(et_f32_tensor_borrow_view_v1(o.borrow,&v,&e)==0);o.view=*v;
  et_f32_tensor *attempt=o.tensor;
  CHECK(et_f32_tensor_destroy_v1(&attempt,&e)!=0 && attempt==o.tensor);
  CHECK(et_f32_tensor_copy_bits_from_v1(o.tensor,words,count,&e)!=0);
  CHECK(o.view.data!=(const void *)initial);return o;
}
static void release_f32(owned *o) {
  et_f32_tensor_error e;
  CHECK(et_f32_tensor_borrow_end_v1(&o->borrow,&e)==0 && o->borrow==NULL);
  CHECK(et_f32_tensor_destroy_v1(&o->tensor,&e)==0 && o->tensor==NULL);
}
static int64_t et_l3s_test_owned_composition_v1(void) {
  environment();const uint64_t ls[]={1,2,256},ts[]={1,2};
  float logits[512],mask[]={0.5f,2};logits_init(logits);
  owned x=create_f32(3,ls,logits,512),ce=create_f32(2,ts,NULL,2),m=create_f32(2,ts,mask,2);
  owned n=create_f32(0,NULL,NULL,1),w=create_f32(0,NULL,NULL,1),mean=create_f32(0,NULL,NULL,1);
  owned ns=create_f32(2,ts,NULL,2),ms=create_f32(2,ts,NULL,2);
  owned ng=create_f32(3,ls,NULL,512),mg=create_f32(3,ls,NULL,512);
  et_i64_tensor *target=NULL;et_i64_tensor_borrow *tb=NULL;et_i64_tensor_error ie;
  int64_t ids[]={3,201};const et_kernel_tensor_view_v1 *tv=NULL;
  CHECK(et_i64_tensor_create_v1(2,ts,&target,&ie)==0);
  CHECK(et_i64_tensor_copy_from_v1(target,ids,2,&ie)==0);
  CHECK(et_i64_tensor_borrow_begin_v1(target,&tb,&ie)==0);
  CHECK(et_i64_tensor_borrow_view_v1(tb,&tv,&ie)==0);
  et_i64_tensor *attempt=target;
  CHECK(et_i64_tensor_destroy_v1(&attempt,&ie)!=0 && attempt==target);
  CHECK(et_i64_tensor_copy_from_v1(target,ids,2,&ie)!=0);
  et_kernel_runtime *r=runtime(et_l3s_kernel_provider_v1()),*l2=runtime(et_l2_indexed_cross_entropy_provider_v1());
  et_kernel_tensor_view_v1 fi[]={x.view,*tv},fo[]={ce.view};
  dispatch(l2,"kernel.indexed-cross-entropy","indexed-cross-entropy.forward",3,ls,fi,2,fo,1);
  et_kernel_tensor_view_v1 ri[]={ce.view,m.view},ro[]={n.view,w.view,mean.view};
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.reduce.f32",2,ts,ri,2,ro,3);
  et_kernel_tensor_view_v1 si[]={m.view},so[]={ns.view};
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.numerator-seed.f32",2,ts,si,1,so,1);
  so[0]=ms.view;
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.mean-seed.f32",2,ts,si,1,so,1);
  et_kernel_tensor_view_v1 bi[]={x.view,*tv,ns.view},bo[]={ng.view};
  dispatch(l2,"kernel.indexed-cross-entropy","indexed-cross-entropy.backward",3,ls,bi,3,bo,1);
  bi[2]=ms.view;bo[0]=mg.view;
  dispatch(l2,"kernel.indexed-cross-entropy","indexed-cross-entropy.backward",3,ls,bi,3,bo,1);
  CHECK(*(float *)w.view.data==2.5f);
  CHECK(memcmp(ns.view.data,mask,8)==0);
  const float *mv=ms.view.data;CHECK(bits(mv[0])==0x3e4ccccdu && bits(mv[1])==0x3f4ccccdu);
  float loss[2],result[3],numseed[2],meanseed[2],numgrad[512],meangrad[512];
  l2_forward(l2,logits,loss);reduce(r,loss,mask,0,result);
  seed(r,mask,0,0,numseed);seed(r,mask,0,1,meanseed);
  l2_backward(l2,logits,numseed,numgrad);l2_backward(l2,logits,meanseed,meangrad);
  CHECK(memcmp(ce.view.data,loss,8)==0 && memcmp(ng.view.data,numgrad,2048)==0 && memcmp(mg.view.data,meangrad,2048)==0);
  CHECK(bits(*(float *)n.view.data)==bits(result[0]) && bits(*(float *)mean.view.data)==bits(result[2]));
  CHECK(memcmp(x.view.data,logits,2048)==0 && memcmp(tv->data,ids,16)==0 && memcmp(m.view.data,mask,8)==0);
  /* bool masks remain plain private bytes; scalar and seed outputs are I2-owned. */
  unsigned char bm[]={0,1};ri[1]=view(bm,2,"bool",2,ts);
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.reduce.bool",2,ts,ri,2,ro,3);
  CHECK(*(float *)mean.view.data==loss[1] && *(float *)w.view.data==1);
  si[0]=ri[1];so[0]=ns.view;
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.numerator-seed.bool",2,ts,si,1,so,1);
  so[0]=ms.view;
  dispatch(r,"l3s.masked-objective","l3s.masked-objective.mean-seed.bool",2,ts,si,1,so,1);
  CHECK(memcmp(ns.view.data,ms.view.data,8)==0 && ((float *)ms.view.data)[0]==0 && ((float *)ms.view.data)[1]==1);
  {
    float badmask[]={1,-1};uint32_t saved[]={bits(*(float *)n.view.data),bits(*(float *)w.view.data),bits(*(float *)mean.view.data)};
    et_kernel_tensor_view_v1 badin[]={ce.view,view(badmask,8,"f32",2,ts)};
    et_kernel_request_v1 q={sizeof(q),"l3s.masked-objective.reduce.f32","f32","cpu",2,ts,1,{0}};
    et_kernel_call_v1 call={sizeof(call),"l3s.masked-objective",&q,2,sizeof(*badin),sizeof(badin),badin,3,sizeof(*ro),sizeof(ro),ro};
    et_kernel_error error;CHECK(et_kernel_runtime_dispatch(r,&call,&error)!=0);
    CHECK(error.category==ET_KERNEL_ERROR_INVALID_ARGUMENT && error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(bits(*(float *)n.view.data)==saved[0] && bits(*(float *)w.view.data)==saved[1] && bits(*(float *)mean.view.data)==saved[2]);
    CHECK(memcmp(ce.view.data,loss,8)==0 && badmask[0]==1 && badmask[1]==-1);
  }
  et_kernel_runtime_destroy(l2);et_kernel_runtime_destroy(r);
  CHECK(et_i64_tensor_borrow_end_v1(&tb,&ie)==0 && tb==NULL);
  CHECK(et_i64_tensor_destroy_v1(&target,&ie)==0 && target==NULL);
  release_f32(&mg);release_f32(&ng);release_f32(&ms);release_f32(&ns);
  release_f32(&mean);release_f32(&w);release_f32(&n);release_f32(&m);release_f32(&ce);release_f32(&x);
  return 1;
}
#ifndef ET_L3S_NO_MAIN
int main(void) {
  for(size_t i=0;i<4;i++)CHECK(et_l3s_test_owned_composition_v1()==1);
  puts("L3S I1/I2 PASS: scoped carriers, independent scalar borrows and real L2 composition");return 0;
}
#endif
