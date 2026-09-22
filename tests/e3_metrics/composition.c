/* Private numerical evidence only: no production carrier or evaluator facade. */
#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"
#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#define REQUIRE(c) do { if (!(c)) { fprintf(stderr,"NUMERIC ASSERTION %s:%d: %s\n",__FILE__,__LINE__,#c); exit(1); } } while (0)
#define CAP "e3.evaluation-metrics"
#define OP(s) CAP "." s
static const uint64_t ls[] = {1,2,256}, ts[] = {1,2};
static uint32_t word(float x) { uint32_t result; memcpy(&result,&x,4); return result; }
#ifdef ET_E3_COMPOSITION_STANDALONE
static float from_word(uint32_t x) { float result; memcpy(&result,&x,4); return result; }
#endif
static et_kernel_tensor_view_v1 view(void *p,size_t bytes,const char *dtype,size_t rank,const uint64_t *shape) {
  et_kernel_tensor_view_v1 v={sizeof(v),p,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};return v;
}
static const et_kernel_provider_v1 *resolver(void *context,const char *symbol) {
  REQUIRE(strcmp(symbol,ET_KERNEL_PROVIDER_SYMBOL_V1)==0);return context;
}
static et_kernel_runtime *runtime(const et_kernel_provider_v1 *provider) {
  et_kernel_runtime *r=NULL;et_kernel_error error;
  REQUIRE(et_kernel_runtime_discover(resolver,(void *)provider,&r,&error)==0);return r;
}
static int call(et_kernel_runtime *r,const char *cap,const char *op,size_t rank,const uint64_t *shape,
                et_kernel_tensor_view_v1 *in,size_t ni,et_kernel_tensor_view_v1 *out,size_t no,et_kernel_error *e) {
  et_kernel_request_v1 request={sizeof(request),op,"f32","cpu",rank,shape,1,{0}};
  et_kernel_call_v1 c={sizeof(c),cap,&request,ni,sizeof(*in),ni*sizeof(*in),in,no,sizeof(*out),no*sizeof(*out),out};
  return et_kernel_runtime_dispatch(r,&c,e);
}
static void dispatch(et_kernel_runtime *r,const char *cap,const char *op,size_t rank,const uint64_t *shape,
                     et_kernel_tensor_view_v1 *in,size_t ni,et_kernel_tensor_view_v1 *out,size_t no) {
  et_kernel_error e;int status=call(r,cap,op,rank,shape,in,ni,out,no,&e);
  if(status) fprintf(stderr,"UNEXPECTED DISPATCH %s: %u/%u %s\n",op,e.category,e.code,e.message);
  REQUIRE(status==0);
}
typedef struct owned { et_f32_tensor *tensor; et_f32_tensor_borrow *borrow; et_kernel_tensor_view_v1 view; } owned;
static owned own(size_t rank,const uint64_t *shape,const float *initial,size_t count) {
  owned o={0};et_f32_tensor_error e;uint32_t words[512];const et_kernel_tensor_view_v1 *v=NULL;
  REQUIRE(count<=512);
  for(size_t i=0;i<count;i++) words[i]=initial?word(initial[i]):0;
  REQUIRE(et_f32_tensor_create_v1(rank,shape,&o.tensor,&e)==0);
  REQUIRE(et_f32_tensor_copy_bits_from_v1(o.tensor,words,count,&e)==0);
  REQUIRE(et_f32_tensor_borrow_begin_v1(o.tensor,&o.borrow,&e)==0);
  REQUIRE(et_f32_tensor_borrow_view_v1(o.borrow,&v,&e)==0);o.view=*v;
  REQUIRE(o.view.data!=(const void *)initial);return o;
}
static void release(owned *o) {
  et_f32_tensor_error e;
  REQUIRE(et_f32_tensor_borrow_end_v1(&o->borrow,&e)==0 && o->borrow==NULL);
  REQUIRE(et_f32_tensor_destroy_v1(&o->tensor,&e)==0 && o->tensor==NULL);
}
static float scalar(const owned *o) { return *(float *)o->view.data; }
static void print_values(const char *name,const float *values,size_t count) {
  printf("%s",name);for(size_t i=0;i<count;i++)printf(" %08" PRIx32,word(values[i]));putchar('\n');
}
static void batch(size_t index,float *logits,int64_t *ids,unsigned char *mask) {
  for(size_t i=0;i<512;i++)logits[i]=(float)((int)((i%256)%17)-8)*0.125f+(float)index*0.0625f+(i>=256?0.03125f:0.0f);
  if(index==0) { logits[3]=logits[7]=logits[256]=logits[511]=4;ids[0]=3;ids[1]=254;mask[0]=mask[1]=1; }
  else if(index==1) { for(size_t i=0;i<256;i++)logits[i]=i%2?0.0f:-0.0f;logits[511]=4;ids[0]=0;ids[1]=255;mask[0]=0;mask[1]=1; }
  else { logits[0]=logits[258]=4;ids[0]=0;ids[1]=2;mask[0]=1;mask[1]=0; }
}
static void l3s_owned_overflow(et_kernel_runtime *l3) {
  float huge[2]={FLT_MAX,FLT_MAX},sentinels[]={123.0f,456.0f,789.0f};
  unsigned char guards[]={0xa5,0x5a,1,1,0x17,0xe3};
  owned ce=own(2,ts,huge,2),outputs[3];et_kernel_tensor_view_v1 out[3];
  for(size_t i=0;i<3;i++) { outputs[i]=own(0,NULL,&sentinels[i],1);out[i]=outputs[i].view; }
  et_kernel_tensor_view_v1 in[]={ce.view,view(guards+2,2,"bool",2,ts)},saved_in[2],saved_out[3];
  memcpy(saved_in,in,sizeof(in));memcpy(saved_out,out,sizeof(out));et_kernel_error error;
  REQUIRE(call(l3,"l3s.masked-objective","l3s.masked-objective.reduce.bool",2,ts,in,2,out,3,&error)!=0);
  REQUIRE(error.category==ET_KERNEL_ERROR_INVALID_ARGUMENT && error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  REQUIRE(memcmp(saved_in,in,sizeof(in))==0 && memcmp(saved_out,out,sizeof(out))==0);
  REQUIRE(memcmp(ce.view.data,huge,sizeof(huge))==0);
  const unsigned char expected[]={0xa5,0x5a,1,1,0x17,0xe3};REQUIRE(memcmp(guards,expected,sizeof(guards))==0);
  for(size_t i=0;i<3;i++) { REQUIRE(word(scalar(&outputs[i]))==word(sentinels[i]));release(&outputs[i]); }
  release(&ce);
}
static void run_composition(int emit) {
  et_kernel_runtime *e3=runtime(et_e3_metrics_kernel_provider_v1());
  et_kernel_runtime *l2=runtime(et_l2_indexed_cross_entropy_provider_v1());
  et_kernel_runtime *l3=runtime(et_l3s_kernel_provider_v1());
  l3s_owned_overflow(l3);
  owned state[2][3];int64_t counters[2][2]={{0}};
  for(size_t s=0;s<2;s++)for(size_t j=0;j<3;j++)state[s][j]=own(0,NULL,NULL,1);
  size_t current=0;
  for(size_t b=0;b<3;b++) {
    float logits[512];int64_t ids[2],active=0;unsigned char mask[2];batch(b,logits,ids,mask);
    owned x=own(3,ls,logits,512),ce=own(2,ts,NULL,2),n=own(0,NULL,NULL,1),w=own(0,NULL,NULL,1),mean=own(0,NULL,NULL,1),correct=own(0,NULL,NULL,1);
    et_i64_tensor *target=NULL;et_i64_tensor_borrow *tb=NULL;et_i64_tensor_error ie;const et_kernel_tensor_view_v1 *tv=NULL;
    REQUIRE(et_i64_tensor_create_v1(2,ts,&target,&ie)==0);
    REQUIRE(et_i64_tensor_copy_from_v1(target,ids,2,&ie)==0);
    REQUIRE(et_i64_tensor_borrow_begin_v1(target,&tb,&ie)==0);
    REQUIRE(et_i64_tensor_borrow_view_v1(tb,&tv,&ie)==0);
    et_kernel_tensor_view_v1 ci[]={x.view,*tv},co[]={ce.view};
    dispatch(l2,"kernel.indexed-cross-entropy","indexed-cross-entropy.forward",3,ls,ci,2,co,1);
    et_kernel_tensor_view_v1 ri[]={ce.view,view(mask,2,"bool",2,ts)},ro[]={n.view,w.view,mean.view};
    dispatch(l3,"l3s.masked-objective","l3s.masked-objective.reduce.bool",2,ts,ri,2,ro,3);
    et_kernel_tensor_view_v1 ai[]={x.view,*tv,ri[1]},ao[]={correct.view,view(&active,8,"i64",0,NULL)};
    dispatch(e3,CAP,OP("correct.bool"),3,ls,ai,3,ao,2);
    REQUIRE(scalar(&correct)==1.0f && active==(b==0?2:1));
    size_t next=1-current;
    et_kernel_tensor_view_v1 si[]={state[current][0].view,state[current][1].view,state[current][2].view,
      view(&counters[current][0],8,"i64",0,NULL),view(&counters[current][1],8,"i64",0,NULL),n.view,w.view,correct.view,ao[1]};
    et_kernel_tensor_view_v1 so[]={state[next][0].view,state[next][1].view,state[next][2].view,
      view(&counters[next][0],8,"i64",0,NULL),view(&counters[next][1],8,"i64",0,NULL)};
    dispatch(e3,CAP,OP("accumulate"),3,ls,si,9,so,5);current=next;
    REQUIRE(counters[current][0]==(int64_t)(b+2) && counters[current][1]==(int64_t)(b+1));
    if(emit) {
      char name[40];float reduction[]={scalar(&n),scalar(&w),scalar(&mean)};
      snprintf(name,sizeof(name),"batch%zu.ce",b);print_values(name,ce.view.data,2);
      snprintf(name,sizeof(name),"batch%zu.reduction",b);print_values(name,reduction,3);
      snprintf(name,sizeof(name),"batch%zu.correct",b);print_values(name,correct.view.data,1);
      float values[]={scalar(&state[current][0]),scalar(&state[current][1]),scalar(&state[current][2])};
      snprintf(name,sizeof(name),"batch%zu.state",b);print_values(name,values,3);
      printf("batch%zu.counters %" PRId64 " %" PRId64 "\n",b,counters[current][0],counters[current][1]);
    }
    /* Genuine I1 target/I2 outputs: a masked invalid target must reject with
       every live owned output/input unchanged. No counterfeit owner is used. */
    if(b==2) {
      uint32_t cb=word(scalar(&correct));int64_t saved_active=active;
      unsigned char saved_logits[2048],saved_ce[8];memcpy(saved_logits,x.view.data,2048);memcpy(saved_ce,ce.view.data,8);
      ((int64_t *)tv->data)[1]=256;
      et_kernel_tensor_view_v1 before_in[3],before_out[2];memcpy(before_in,ai,sizeof(ai));memcpy(before_out,ao,sizeof(ao));
      et_kernel_error error;REQUIRE(call(e3,CAP,OP("correct.bool"),3,ls,ai,3,ao,2,&error)!=0);
      REQUIRE(error.category==ET_KERNEL_ERROR_SHAPE_MISMATCH && error.code==ET_KERNEL_CODE_INVALID_SHAPE);
      REQUIRE(word(scalar(&correct))==cb && active==saved_active);
      REQUIRE(memcmp(before_in,ai,sizeof(ai))==0 && memcmp(before_out,ao,sizeof(ao))==0);
      REQUIRE(memcmp(saved_logits,x.view.data,2048)==0 && memcmp(saved_ce,ce.view.data,8)==0);
      REQUIRE(((int64_t *)tv->data)[1]==256);((int64_t *)tv->data)[1]=ids[1];
    }
    REQUIRE(memcmp(x.view.data,logits,sizeof(logits))==0 && memcmp(tv->data,ids,sizeof(ids))==0);
    REQUIRE(et_i64_tensor_borrow_end_v1(&tb,&ie)==0 && tb==NULL);
    REQUIRE(et_i64_tensor_destroy_v1(&target,&ie)==0 && target==NULL);
    release(&correct);release(&mean);release(&w);release(&n);release(&ce);release(&x);
  }
  owned metrics[4];et_kernel_tensor_view_v1 in[3],out[4];
  for(size_t i=0;i<3;i++)in[i]=state[current][i].view;
  for(size_t i=0;i<4;i++) { metrics[i]=own(0,NULL,NULL,1);out[i]=metrics[i].view; }
  dispatch(e3,CAP,OP("finalize"),3,ls,in,3,out,4);
  REQUIRE(scalar(&metrics[1])==4.0f && scalar(&metrics[3])==0.75f);
  if(emit) { float values[4];for(size_t i=0;i<4;i++)values[i]=scalar(&metrics[i]);print_values("global.metrics",values,4); }
  for(size_t i=0;i<4;i++)release(&metrics[i]);
  for(size_t s=0;s<2;s++)for(size_t i=0;i<3;i++)release(&state[s][i]);
  et_kernel_runtime_destroy(l3);et_kernel_runtime_destroy(l2);et_kernel_runtime_destroy(e3);
}
static void environment(void) {
  REQUIRE(fesetenv(FE_DFL_ENV)==0);REQUIRE(fesetround(FE_TONEAREST)==0);
  _mm_setcsr((_mm_getcsr() & ~0xe040u) | 0x1f80u);
}
int et_test_e3_metrics_composition(void) {
  fenv_t saved;REQUIRE(fegetenv(&saved)==0);environment();run_composition(0);REQUIRE(fesetenv(&saved)==0);return 0;
}
#ifdef ET_E3_COMPOSITION_STANDALONE
static void final_cli(char **args) {
  et_kernel_runtime *r=runtime(et_e3_metrics_kernel_provider_v1());float input[3],output[4];
  et_kernel_tensor_view_v1 in[3],out[4];et_kernel_error error;
  for(size_t i=0;i<3;i++) { input[i]=from_word((uint32_t)strtoul(args[i],NULL,16));in[i]=view(input+i,4,"f32",0,NULL); }
  for(size_t i=0;i<4;i++) { output[i]=from_word(0x4abc1234u);out[i]=view(output+i,4,"f32",0,NULL); }
  int status=call(r,CAP,OP("finalize"),3,ls,in,3,out,4,&error);
  printf("status %d %u %u\n",status,error.category,error.code);print_values("final",output,4);
  et_kernel_runtime_destroy(r);
}
static void witness(int mean_case) {
  et_kernel_runtime *r=runtime(et_e3_metrics_kernel_provider_v1());
  float state[2][3]={{0}};int64_t counters[2][2]={{0}};size_t current=0;
  const float numerators[]={64.0f,0x1p-18f,0x1p-18f};
  const float means_n[]={1.0f,6.0f},means_w[]={1.0f,2.0f};
  for(size_t b=0;b<(mean_case?2u:3u);b++) {
    float bn=mean_case?means_n[b]:numerators[b],bw=mean_case?means_w[b]:1.0f,bc=0.0f;int64_t active=(int64_t)bw;
    size_t next=1-current;et_kernel_tensor_view_v1 in[9],out[5];
    for(size_t i=0;i<3;i++) { in[i]=view(&state[current][i],4,"f32",0,NULL);out[i]=view(&state[next][i],4,"f32",0,NULL); }
    for(size_t i=0;i<2;i++) { in[3+i]=view(&counters[current][i],8,"i64",0,NULL);out[3+i]=view(&counters[next][i],8,"i64",0,NULL); }
    in[5]=view(&bn,4,"f32",0,NULL);in[6]=view(&bw,4,"f32",0,NULL);in[7]=view(&bc,4,"f32",0,NULL);in[8]=view(&active,8,"i64",0,NULL);
    dispatch(r,CAP,OP("accumulate"),3,ls,in,9,out,5);current=next;
  }
  print_values("witness.state",state[current],3);
  float output[4];et_kernel_tensor_view_v1 in[3],out[4];
  for(size_t i=0;i<3;i++)in[i]=view(&state[current][i],4,"f32",0,NULL);
  for(size_t i=0;i<4;i++)out[i]=view(&output[i],4,"f32",0,NULL);
  dispatch(r,CAP,OP("finalize"),3,ls,in,3,out,4);print_values("witness.final",output,4);
  et_kernel_runtime_destroy(r);
}
int main(int argc,char **argv) {
  environment();
  if(argc==2 && strcmp(argv[1],"--emit")==0)run_composition(1);
  else if(argc==5 && strcmp(argv[1],"--finalize")==0)final_cli(argv+2);
  else if(argc==2 && strcmp(argv[1],"--witness")==0)witness(0);
  else if(argc==2 && strcmp(argv[1],"--mean-witness")==0)witness(1);
  else { REQUIRE(argc==1);REQUIRE(et_test_e3_metrics_composition()==0);puts("E3-METRICS NUMERICAL PASS: three real L2/L3S/I1/I2 BOOL batches and owned failure atomicity"); }
  return 0;
}
#endif
