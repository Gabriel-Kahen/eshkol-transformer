/* Development-only real-dispatch parity and literal intermediate evidence. */
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "eshkol_transformer/g3s_sampling_abi.h"
#include "g3s_reference.h"

/* This TU alone sees static helpers. The archive exports no debug interface. */
#ifndef G3S_PROVIDER_SOURCE
#define G3S_PROVIDER_SOURCE "../../native/g3s_sampling_provider.c"
#endif
#include G3S_PROVIDER_SOURCE

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { fprintf(stderr,"numerical:%d: %s\n",__LINE__,#c); exit(1); } } while (0)

static uint32_t fbits(float value) { uint32_t out; memcpy(&out,&value,4); return out; }
static float frombits(uint32_t value) { float out; memcpy(&out,&value,4); return out; }
static const et_kernel_provider_v1 *resolve(void *unused, const char *symbol) {
  (void)unused;
  return strcmp(symbol,ET_KERNEL_PROVIDER_SYMBOL_V1)==0 ? et_g3s_kernel_provider_v1():NULL;
}

static const uint64_t logits_shape[2]={1,256}, rng_shape[1]={4}, token_shape[1]={1};
typedef struct {
  float logits[256], temperature, p;
  int64_t k, rng[4], token, successor[4];
  et_kernel_tensor_view_v1 inputs[5], outputs[2];
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
} numeric_fixture;

static et_kernel_tensor_view_v1 nview(void *data,size_t bytes,const char *dtype,size_t rank,const uint64_t *shape) {
  et_kernel_tensor_view_v1 out={sizeof(out),data,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};
  return out;
}
static void setup(numeric_fixture *f,const g3s_reference_case *ref,int greedy,unsigned deterministic) {
  memset(f,0,sizeof(*f));
  for (unsigned i=0;i<256;++i) f->logits[i]=frombits(ref->logits[i]);
  f->temperature=frombits(ref->temperature); f->p=frombits(ref->p); f->k=ref->k;
  f->rng[0]=1; memcpy(&f->rng[1],&ref->seed,8); memcpy(&f->rng[2],&ref->low,8); memcpy(&f->rng[3],&ref->high,8);
  f->token=-17; memset(f->successor,0xa5,sizeof(f->successor));
  f->inputs[0]=nview(f->logits,sizeof(f->logits),"f32",2,logits_shape);
  f->inputs[1]=nview(&f->temperature,4,"f32",0,NULL);
  f->inputs[2]=nview(&f->k,8,"i64",0,NULL);
  f->inputs[3]=nview(&f->p,4,"f32",0,NULL);
  f->inputs[4]=nview(f->rng,32,"i64",1,rng_shape);
  if (greedy) f->inputs[1]=f->inputs[4];
  f->outputs[0]=nview(&f->token,8,"i64",1,token_shape);
  f->outputs[1]=nview(f->successor,32,"i64",1,rng_shape);
  f->request=(et_kernel_request_v1){sizeof(f->request),greedy?"g3s.greedy.forward":"g3s.categorical.forward","f32","cpu",2,logits_shape,(uint8_t)deterministic,{0}};
  f->call=(et_kernel_call_v1){sizeof(f->call),greedy?"g3s.greedy":"g3s.categorical",&f->request,greedy?2u:5u,sizeof(f->inputs[0]),(greedy?2u:5u)*sizeof(f->inputs[0]),f->inputs,2,sizeof(f->outputs[0]),sizeof(f->outputs),f->outputs};
}

static void dispatched_cases(et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *provider=et_g3s_kernel_provider_v1();
  for (size_t n=0;n<G3S_REFERENCE_COUNT;++n) {
    const g3s_reference_case *ref=&g3s_cases[n];
    for (unsigned deterministic=0;deterministic<2;++deterministic) {
      numeric_fixture f;
      setup(&f,ref,0,deterministic);
      numeric_fixture saved=f;
      et_kernel_error error;
      CHECK(provider->validate_call(&f.call,&error)==0);
      CHECK(memcmp(&f,&saved,sizeof(f))==0);
      int32_t result=et_kernel_runtime_dispatch(runtime,&f.call,&error);
      if (result!=0) fprintf(stderr,"fixture %s: %u/%u %s\n",ref->name,error.category,error.code,error.message);
      CHECK(result==0);
      CHECK(f.token==(int64_t)ref->token);
      if (strcmp(ref->name,"strict_cdf_equality")==0) CHECK(f.token==143);
      if (strcmp(ref->name,"uniform_zero_uniform")==0) CHECK(f.token==0);
      if (strcmp(ref->name,"uniform_max_uniform")==0) CHECK(f.token==255);
      if (strcmp(ref->name,"p1_early")==0) CHECK(f.token==0);
      CHECK(memcmp(f.logits,saved.logits,sizeof(f.logits))==0);
      CHECK(memcmp(f.rng,saved.rng,sizeof(f.rng))==0);
      uint64_t lo=ref->low+UINT64_C(1),hi=ref->high+(lo==0);
      CHECK(f.successor[0]==1 && f.successor[1]==f.rng[1]);
      CHECK(memcmp(&f.successor[2],&lo,8)==0 && memcmp(&f.successor[3],&hi,8)==0);
      /* Repeat from the same independent numeric state, irrespective of address. */
      numeric_fixture again;
      setup(&again,ref,0,deterministic);
      CHECK(et_kernel_runtime_dispatch(runtime,&again.call,&error)==0);
      CHECK(again.token==f.token && memcmp(again.successor,f.successor,32)==0);
      setup(&f,ref,1,deterministic);
      unsigned maximum=0;
      for (unsigned i=1;i<256;++i) if (f.logits[i]>f.logits[maximum]) maximum=i;
      /* Exhausted numeric state is valid for every greedy fixture. */
      f.rng[2]=-1; f.rng[3]=-1;
      CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&error)==0);
      CHECK(f.token==(int64_t)maximum);
      CHECK(memcmp(f.rng,f.successor,32)==0);
    }
  }
}

static void intermediate_cases(void) {
  for (size_t n=0;n<G3S_REFERENCE_COUNT;++n) {
    const g3s_reference_case *ref=&g3s_cases[n];
    numeric_fixture f;
    setup(&f,ref,0,1);
    sampling_scratch scratch;
    CHECK(distribution(f.logits,f.temperature,(size_t)f.k,f.p,&scratch)==1);
    CHECK(scratch.count==ref->prefix);
    CHECK(fbits(scratch.weight)==ref->weight);
    uint32_t lanes[4];
    philox4x32_10(ref->seed^UINT64_C(0x4733434154454731),ref->low,ref->high,lanes);
    CHECK(memcmp(lanes,ref->lanes,sizeof(lanes))==0);
    const float uniform=(float)(lanes[0]>>8)*0x1p-24f;
    CHECK(fbits(uniform*scratch.weight)==ref->threshold);
    CHECK(select_token(&scratch,uniform)==(int64_t)ref->token);
    for (unsigned i=0;i<256;++i) {
      CHECK(fbits(scratch.a[i])==ref->a[i]);
      CHECK(scratch.order[i]==ref->order[i]);
      CHECK(fbits(scratch.b[i])==ref->b[i]);
      CHECK(fabs((double)scratch.a[i]-ref->mathematical[i])<=2e-5);
      if (i<(unsigned)f.k) CHECK(fabs((double)scratch.b[i]-ref->mathematical_b[i])<=2e-5);
    }
  }
  /* No transcendental dependence in these fixed literal probability/CDF cases. */
  sampling_scratch equal;
  float zeros[256]={0};
  CHECK(distribution(zeros,1,256,1,&equal)==1);
  for (unsigned i=0;i<256;++i) {
    CHECK(fbits(equal.a[i])==UINT32_C(0x3b800000));
    CHECK(fbits(equal.b[i])==UINT32_C(0x3b800000));
    CHECK(equal.order[i]==i);
  }
  CHECK(equal.count==256 && fbits(equal.weight)==UINT32_C(0x3f800000));
  CHECK(select_token(&equal,0)==0);
  CHECK(select_token(&equal,0x1.fffffep-1f)==255);
  CHECK(select_token(&equal,0.5f)==128);
  CHECK(distribution(zeros,1,256,0.5f,&equal)==1 && equal.count==128);
  CHECK(select_token(&equal,0.5f)==64);
  CHECK(distribution(zeros,1,256,frombits(UINT32_C(0x3effffff)),&equal)==1 && equal.count==128);
  CHECK(distribution(zeros,1,256,frombits(UINT32_C(0x3f000001)),&equal)==1 && equal.count==129);
  /* A defensive endpoint helper test, deliberately outside RNG u<1. This is
     not a claim that u=1 is produced by the public sampler. Zero tails cannot win. */
  sampling_scratch endpoint={0};
  endpoint.b[0]=0.25f; endpoint.b[1]=0.75f;
  endpoint.order[0]=17; endpoint.order[1]=29; endpoint.order[2]=255;
  endpoint.count=3; endpoint.weight=1;
  CHECK(select_token(&endpoint,1)==29);
  CHECK(select_token(&endpoint,0)==17);
  CHECK(select_token(&endpoint,0.25f)==29);
  CHECK(select_token(&endpoint,frombits(UINT32_C(0x3e7fffff)))==17);
  CHECK(select_token(&endpoint,frombits(UINT32_C(0x3e800001)))==29);
  endpoint.b[0]=0; endpoint.b[1]=1;
  CHECK(select_token(&endpoint,0)==29);
}

static void continuation(et_kernel_runtime *runtime) {
  numeric_fixture f;
  setup(&f,&g3s_cases[0],0,1);
  f.rng[1]=7; f.rng[2]=-3; f.rng[3]=4;
  for (unsigned step=0;step<16;++step) {
    int64_t before[4]; memcpy(before,f.rng,32);
    et_kernel_error error;
    CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&error)==0);
    /* Equal weights turn uniform lane0 high8 directly into a token ID. */
    CHECK(f.token==(int64_t)(continuation_lanes[step][0]>>24));
    uint64_t lo,hi; memcpy(&lo,&before[2],8); memcpy(&hi,&before[3],8);
    ++lo; if (lo==0) ++hi;
    CHECK(memcmp(&f.successor[2],&lo,8)==0);
    CHECK(memcmp(&f.successor[3],&hi,8)==0);
    CHECK(memcmp(f.rng,before,32)==0);
    memcpy(f.rng,f.successor,32);
  }
}

int main(void) {
  et_kernel_runtime *runtime=NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve,NULL,&runtime,&error)==0);
  intermediate_cases();
  dispatched_cases(runtime);
  continuation(runtime);
  et_kernel_runtime_destroy(runtime);
  printf("G3-S numerical PASS: %u checks; %zu independent literal/Decimal80 fixtures\n",checks,G3S_REFERENCE_COUNT);
  return 0;
}
