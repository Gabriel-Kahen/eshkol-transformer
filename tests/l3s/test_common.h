#ifndef ET_L3S_TEST_COMMON_H
#define ET_L3S_TEST_COMMON_H
#include "eshkol_transformer/l3s_masked_objective_abi.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c); exit(1); } } while (0)
static inline uint32_t bits(float x) { uint32_t b; memcpy(&b,&x,4); return b; }
static inline float unbits(uint32_t b) { float x; memcpy(&x,&b,4); return x; }
static inline void environment(void) {
  CHECK(fesetenv(FE_DFL_ENV)==0); CHECK(fesetround(FE_TONEAREST)==0);
  _mm_setcsr((_mm_getcsr() & ~0xe040u) | 0x1f80u);
}
static inline et_kernel_tensor_view_v1 view(void *p,size_t bytes,const char *dtype,size_t rank,const uint64_t *shape) {
  et_kernel_tensor_view_v1 v={sizeof(v),p,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};return v;
}
static inline const et_kernel_provider_v1 *resolver(void *context,const char *symbol) {
  CHECK(strcmp(symbol,ET_KERNEL_PROVIDER_SYMBOL_V1)==0);return context;
}
static inline et_kernel_runtime *runtime(const et_kernel_provider_v1 *p) {
  et_kernel_runtime *r=NULL;et_kernel_error e;
  CHECK(et_kernel_runtime_discover(resolver,(void *)p,&r,&e)==0);return r;
}
static inline void dispatch(et_kernel_runtime *r,const char *cap,const char *op,size_t rank,const uint64_t *shape,
    et_kernel_tensor_view_v1 *in,size_t ni,et_kernel_tensor_view_v1 *out,size_t no) {
  et_kernel_request_v1 q={sizeof(q),op,"f32","cpu",rank,shape,1,{0}};
  et_kernel_call_v1 c={sizeof(c),cap,&q,ni,sizeof(*in),ni*sizeof(*in),in,no,sizeof(*out),no*sizeof(*out),out};
  et_kernel_error e;
  if(et_kernel_runtime_dispatch(r,&c,&e)!=0) {fprintf(stderr,"dispatch %s: %u/%u %s\n",op,e.category,e.code,e.message);CHECK(0);}
}
static inline void reduce(et_kernel_runtime *r,float loss[2],void *mask,int boolean,float result[3]) {
  const uint64_t shape[]={1,2};
  et_kernel_tensor_view_v1 in[]={view(loss,8,"f32",2,shape),view(mask,boolean?2:8,boolean?"bool":"f32",2,shape)};
  et_kernel_tensor_view_v1 out[]={view(result,4,"f32",0,NULL),view(result+1,4,"f32",0,NULL),view(result+2,4,"f32",0,NULL)};
  dispatch(r,"l3s.masked-objective",boolean?"l3s.masked-objective.reduce.bool":"l3s.masked-objective.reduce.f32",2,shape,in,2,out,3);
}
static inline void seed(et_kernel_runtime *r,void *mask,int boolean,int mean,float result[2]) {
  const uint64_t shape[]={1,2};
  const char *op=boolean?(mean?"l3s.masked-objective.mean-seed.bool":"l3s.masked-objective.numerator-seed.bool"):
      (mean?"l3s.masked-objective.mean-seed.f32":"l3s.masked-objective.numerator-seed.f32");
  et_kernel_tensor_view_v1 in[]={view(mask,boolean?2:8,boolean?"bool":"f32",2,shape)},out[]={view(result,8,"f32",2,shape)};
  dispatch(r,"l3s.masked-objective",op,2,shape,in,1,out,1);
}
static inline void logits_init(float logits[512]) {
  for(size_t i=0;i<512;i++) logits[i]=(float)((int)((i%256)%17)-8)*0.125f+(i>=256?0.03125f:0.0f);
  logits[3]=1.75f;logits[256+201]=-1.5f;
}
static inline void l2_forward(et_kernel_runtime *r,float logits[512],float loss[2]) {
  const uint64_t ls[]={1,2,256},ts[]={1,2};int64_t targets[]={3,201};
  et_kernel_tensor_view_v1 in[]={view(logits,2048,"f32",3,ls),view(targets,16,"i64",2,ts)},out[]={view(loss,8,"f32",2,ts)};
  dispatch(r,"kernel.indexed-cross-entropy","indexed-cross-entropy.forward",3,ls,in,2,out,1);
}
static inline void l2_backward(et_kernel_runtime *r,float logits[512],float upstream[2],float gradient[512]) {
  const uint64_t ls[]={1,2,256},ts[]={1,2};int64_t targets[]={3,201};
  et_kernel_tensor_view_v1 in[]={view(logits,2048,"f32",3,ls),view(targets,16,"i64",2,ts),view(upstream,8,"f32",2,ts)},out[]={view(gradient,2048,"f32",3,ls)};
  dispatch(r,"kernel.indexed-cross-entropy","indexed-cross-entropy.backward",3,ls,in,3,out,1);
}
#endif
