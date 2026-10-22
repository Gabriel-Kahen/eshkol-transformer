/* Provider-only T4 K1 dispatch, atomicity and real L2 seed composition. */
#include "../l3s/test_common.h"
#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include "reference.h"
#include <float.h>

enum { REDUCE, NUMERATOR_SEED, MEAN_SEED, CORRECT };
typedef struct {
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 inputs[3], outputs[3];
  uint64_t request_shape[3], shapes[3][3], seed_shape[2];
  float losses[4], weights[4], logits[1024], result[4];
  int64_t targets[4], active;
  uint8_t mask[4];
} fixture;

static const char *const l3s_ops[3][2] = {
  {"l3s.masked-objective.reduce.f32", "l3s.masked-objective.reduce.bool"},
  {"l3s.masked-objective.numerator-seed.f32", "l3s.masked-objective.numerator-seed.bool"},
  {"l3s.masked-objective.mean-seed.f32", "l3s.masked-objective.mean-seed.bool"}
};
static void setup(fixture *f, int op, int boolean) {
  memset(f,0,sizeof(*f));
  f->request_shape[0]=1; f->request_shape[1]=4; f->request_shape[2]=256;
  f->seed_shape[0]=1; f->seed_shape[1]=4;
  for(size_t i=0;i<3;++i) memcpy(f->shapes[i],f->request_shape,sizeof(f->request_shape));
  for(size_t i=0;i<4;++i) {
    f->losses[i]=(float)(2+i); f->weights[i]=1; f->mask[i]=1;
    f->result[i]=unbits(0x4a010203u); f->targets[i]=(int64_t)i;
    f->logits[i*256+i]=2;
  }
  f->active=INT64_C(1234567);
  const char *name=op==CORRECT?"e3.evaluation-metrics.correct.bool":l3s_ops[op][boolean];
  const char *cap=op==CORRECT?"e3.evaluation-metrics":"l3s.masked-objective";
  const size_t rank=op==CORRECT?3u:2u;
  f->request=(et_kernel_request_v1){sizeof(f->request),name,"f32","cpu",rank,f->request_shape,1,{0}};
  size_t ni, no;
  if(op==CORRECT) {
    f->inputs[0]=view(f->logits,sizeof(f->logits),"f32",3,f->shapes[0]);
    f->inputs[1]=view(f->targets,sizeof(f->targets),"i64",2,f->shapes[1]);
    f->inputs[2]=view(f->mask,sizeof(f->mask),"bool",2,f->shapes[2]);
    f->outputs[0]=view(f->result,4,"f32",0,NULL);
    f->outputs[1]=view(&f->active,8,"i64",0,NULL);
    ni=3; no=2;
  } else {
    const size_t m=op==REDUCE?1u:0u;
    if(op==REDUCE) f->inputs[0]=view(f->losses,16,"f32",2,f->shapes[0]);
    f->inputs[m]=view(boolean?(void *)f->mask:(void *)f->weights,
                      boolean?4u:16u,boolean?"bool":"f32",2,f->shapes[m]);
    if(op==REDUCE) for(size_t i=0;i<3;++i) f->outputs[i]=view(f->result+i,4,"f32",0,NULL);
    else f->outputs[0]=view(f->result,16,"f32",2,f->seed_shape);
    ni=op==REDUCE?2u:1u; no=op==REDUCE?3u:1u;
  }
  f->call=(et_kernel_call_v1){sizeof(f->call),cap,&f->request,ni,sizeof(f->inputs[0]),
    ni*sizeof(f->inputs[0]),f->inputs,no,sizeof(f->outputs[0]),no*sizeof(f->outputs[0]),f->outputs};
}
static void perform(et_kernel_runtime *r, fixture *f) {
  et_kernel_error e;
  int32_t status=et_kernel_runtime_dispatch(r,&f->call,&e);
  if(status) fprintf(stderr,"unexpected dispatch %u/%u %s\n",e.category,e.code,e.message);
  CHECK(status==0);
}
static void reject(et_kernel_runtime *r, fixture *f) {
  fixture before; memcpy(&before,f,sizeof(before));
  et_kernel_error e; fenv_t env; CHECK(fegetenv(&env)==0);
  unsigned csr=_mm_getcsr(); int rounding=fegetround();
  CHECK(et_kernel_runtime_dispatch(r,&f->call,&e)!=0);
  CHECK(memcmp(&before,f,sizeof(before))==0);
  CHECK(_mm_getcsr()==csr); CHECK(fegetround()==rounding);
  CHECK(fesetenv(&env)==0);
}
static void exact_cases(et_kernel_runtime *l3s) {
  for(size_t c=0;c<sizeof(cpu4_cases)/sizeof(cpu4_cases[0]);++c) {
    const cpu4_case *ref=cpu4_cases+c;
    for(int op=REDUCE;op<=MEAN_SEED;++op) {
      fixture f; setup(&f,op,ref->boolean);
      for(size_t i=0;i<4;++i) {
        f.losses[i]=unbits(ref->losses[i]); f.weights[i]=unbits(ref->weights[i]);
        f.mask[i]=(uint8_t)f.weights[i];
      }
      float old_losses[4],old_weights[4]; uint8_t old_mask[4];
      memcpy(old_losses,f.losses,16);memcpy(old_weights,f.weights,16);memcpy(old_mask,f.mask,4);
      perform(l3s,&f);
      const uint32_t *expected=op==REDUCE?ref->reduction:op==NUMERATOR_SEED?ref->numerator_seed:ref->mean_seed;
      for(size_t i=0;i<(op==REDUCE?3u:4u);++i) CHECK(bits(f.result[i])==expected[i]);
      CHECK(memcmp(old_losses,f.losses,16)==0);CHECK(memcmp(old_weights,f.weights,16)==0);
      CHECK(memcmp(old_mask,f.mask,4)==0);
    }
  }
  /* Provider arithmetic witness only: never claim actual model reachability. */
  fixture f; setup(&f,REDUCE,1);
  f.losses[0]=64;f.losses[1]=0x1p-18f;f.losses[2]=0x1p-18f;f.losses[3]=0;
  perform(l3s,&f); CHECK(bits(f.result[0])==0x42800000u);
  CHECK(bits(64.0f+(0x1p-18f+0x1p-18f))==0x42800001u);
}
static void correct_cases(et_kernel_runtime *metrics) {
  fixture all;setup(&all,CORRECT,1);perform(metrics,&all);
  CHECK(all.result[0]==4 && all.active==4);
  setup(&all,CORRECT,1);for(size_t i=0;i<4;++i) all.targets[i]=(int64_t)(i+1);
  perform(metrics,&all);CHECK(all.result[0]==0 && all.active==4);
  for(unsigned mask=1;mask<16;++mask) {
    fixture f;setup(&f,CORRECT,1);int64_t active=0;float correct=0;
    /* All-zero row ties at0; another row ties at2/9; two right and two wrong. */
    memset(f.logits,0,sizeof(f.logits));
    f.logits[256+2]=3;f.logits[256+9]=3;
    f.logits[512+17]=4;f.logits[768+255]=5;
    f.targets[0]=0;f.targets[1]=9;f.targets[2]=17;f.targets[3]=0;
    for(size_t i=0;i<4;++i) {
      f.mask[i]=(uint8_t)((mask>>i)&1u);active+=f.mask[i];
      if(i==0 || i==2) correct+=(float)f.mask[i];
    }
    perform(metrics,&f);CHECK(f.result[0]==correct);CHECK(f.active==active);
  }
  fixture tiny;setup(&tiny,CORRECT,1);memset(tiny.logits,0,sizeof(tiny.logits));
  memset(tiny.mask,0,4);tiny.mask[3]=1;tiny.targets[3]=255;tiny.logits[1023]=unbits(1);
  perform(metrics,&tiny);CHECK(tiny.result[0]==1 && tiny.active==1);
  for(size_t i=0;i<1024;++i) tiny.logits[i]=-0.0f;
  tiny.targets[3]=0;perform(metrics,&tiny);CHECK(tiny.result[0]==1 && tiny.active==1);
}
static void negative_cases(et_kernel_runtime *l3s, et_kernel_runtime *metrics) {
  for(int op=REDUCE;op<=CORRECT;++op) for(int boolean=0;boolean<2;++boolean) {
    if(op==CORRECT && !boolean) continue;
    et_kernel_runtime *r=op==CORRECT?metrics:l3s;fixture f;
    for(size_t t=3;t<=5;t+=2) { setup(&f,op,boolean);f.request_shape[1]=t;reject(r,&f); }
    setup(&f,op,boolean);f.request_shape[1]=2;reject(r,&f);
    for(size_t i=0;i<(op==CORRECT?3u:op==REDUCE?2u:1u);++i) {
      setup(&f,op,boolean);f.shapes[i][1]=2;
      f.inputs[i].byte_length/=2;reject(r,&f);
      setup(&f,op,boolean);f.inputs[i].dtype="f64";reject(r,&f);
      setup(&f,op,boolean);f.inputs[i].device="cuda";reject(r,&f);
      setup(&f,op,boolean);f.inputs[i].offset_bytes=1;reject(r,&f);
      setup(&f,op,boolean);f.inputs[i].rank=1;reject(r,&f);
      setup(&f,op,boolean);f.inputs[i].byte_length-=1;reject(r,&f);
    }
    setup(&f,op,boolean);f.outputs[0].data=f.inputs[0].data;reject(r,&f);
    if(op==REDUCE) {
      setup(&f,op,boolean);f.inputs[1].data=f.inputs[0].data;reject(r,&f);
      setup(&f,op,boolean);f.outputs[1].data=f.outputs[0].data;reject(r,&f);
    }
    setup(&f,op,boolean);f.outputs[0].data=f.request_shape;reject(r,&f);
    setup(&f,op,boolean);f.outputs[0].device="cuda";reject(r,&f);
    setup(&f,op,boolean);f.outputs[0].byte_length-=1;reject(r,&f);
    if(op==NUMERATOR_SEED || op==MEAN_SEED) {
      setup(&f,op,boolean);f.seed_shape[1]=2;f.outputs[0].byte_length=8;reject(r,&f);
    }
    setup(&f,op,boolean);memset(f.mask,0,4);memset(f.weights,0,16);reject(r,&f);
    if(boolean) {setup(&f,op,boolean);f.mask[3]=2;reject(r,&f);}
    else {setup(&f,op,boolean);f.weights[3]=NAN;reject(r,&f);}
    if(op==REDUCE) {
      setup(&f,op,boolean);f.mask[3]=0;f.weights[3]=0;f.losses[3]=NAN;reject(r,&f);
      setup(&f,op,boolean);f.losses[3]=-1;reject(r,&f);
      setup(&f,op,boolean);f.losses[2]=FLT_MAX;f.losses[3]=FLT_MAX;reject(r,&f);
      if(!boolean) {
        setup(&f,op,boolean);f.weights[3]=2;f.losses[3]=FLT_MAX;reject(r,&f);
        setup(&f,op,boolean);f.weights[2]=FLT_MAX;f.weights[3]=FLT_MAX;reject(r,&f);
      }
    }
    if(op==CORRECT) {
      setup(&f,op,boolean);f.mask[3]=0;f.targets[3]=256;reject(r,&f);
      setup(&f,op,boolean);f.targets[3]=-1;reject(r,&f);
      setup(&f,op,boolean);f.mask[3]=0;f.logits[1023]=NAN;reject(r,&f);
      setup(&f,op,boolean);f.logits[1023]=INFINITY;reject(r,&f);
    }
    setup(&f,op,boolean);CHECK(fesetround(FE_UPWARD)==0);reject(r,&f);environment();
    setup(&f,op,boolean);_mm_setcsr(_mm_getcsr()|0x8040u);reject(r,&f);environment();
  }
}

typedef struct {
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 in[9], out[5];
  uint64_t shape[3];
  float values[9], results[5];
  int64_t counts[3], next[2];
} scalar_fixture;
static void scalar_setup(scalar_fixture *f, size_t width, int finalize) {
  memset(f,0,sizeof(*f));
  f->shape[0]=1;f->shape[1]=width;f->shape[2]=256;
  f->request=(et_kernel_request_v1){sizeof(f->request),finalize?
    "e3.evaluation-metrics.finalize":"e3.evaluation-metrics.accumulate",
    "f32","cpu",3,f->shape,1,{0}};
  const size_t ni=finalize?3u:9u,no=finalize?4u:5u;
  for(size_t i=0;i<ni;++i) f->in[i]=view(f->values+i,4,"f32",0,NULL);
  for(size_t i=0;i<no;++i) {f->results[i]=unbits(0x4a010203u);f->out[i]=view(f->results+i,4,"f32",0,NULL);}
  if(finalize) f->values[1]=1;
  else {
    f->counts[2]=(int64_t)width;f->values[6]=(float)width;
    f->in[3]=view(f->counts,8,"i64",0,NULL);
    f->in[4]=view(f->counts+1,8,"i64",0,NULL);
    f->in[8]=view(f->counts+2,8,"i64",0,NULL);
    f->next[0]=f->next[1]=1234567;
    f->out[3]=view(f->next,8,"i64",0,NULL);f->out[4]=view(f->next+1,8,"i64",0,NULL);
  }
  f->call=(et_kernel_call_v1){sizeof(f->call),"e3.evaluation-metrics",&f->request,
    ni,sizeof(f->in[0]),ni*sizeof(f->in[0]),f->in,
    no,sizeof(f->out[0]),no*sizeof(f->out[0]),f->out};
}
static void scalar_check(et_kernel_runtime *r,scalar_fixture *f,int expected) {
  scalar_fixture before;memcpy(&before,f,sizeof(before));et_kernel_error e;
  CHECK(et_kernel_runtime_dispatch(r,&f->call,&e)==expected);
  if(expected) CHECK(memcmp(&before,f,sizeof(before))==0);
}
static void scalar_cases(et_kernel_runtime *metrics) {
  scalar_fixture f;
  scalar_setup(&f,2,0);f.counts[2]=3;f.values[6]=3;
  scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  scalar_setup(&f,4,0);scalar_check(metrics,&f,0);
  CHECK(f.next[0]==4 && f.next[1]==1 && f.results[1]==4);
  for(size_t width=2;width<=4;width+=2) {
    int64_t cap=8192*(int64_t)width;
    scalar_setup(&f,width,0);f.counts[0]=cap-(int64_t)width;f.counts[1]=8191;
    f.values[1]=(float)f.counts[0];scalar_check(metrics,&f,0);
    CHECK(f.next[0]==cap && f.next[1]==8192 && f.results[1]==(float)cap);
    scalar_setup(&f,width,0);f.counts[0]=cap;f.counts[1]=8192;f.values[1]=(float)cap;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_UNSUPPORTED);
    scalar_setup(&f,width,0);f.counts[0]=cap+1;f.counts[1]=8192;f.values[1]=(float)(cap+1);
    scalar_check(metrics,&f,ET_KERNEL_ERROR_UNSUPPORTED);
    scalar_setup(&f,width,0);f.counts[0]=INT64_MAX;f.counts[1]=INT64_MAX;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_UNSUPPORTED);
    /* Float domain errors precede over-cap counter errors. */
    f.values[0]=NAN;scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,0);f.counts[0]=-1;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,0);f.counts[0]=1;f.counts[1]=2;f.values[1]=1;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,0);f.counts[0]=(int64_t)width+1;f.counts[1]=1;f.values[1]=(float)f.counts[0];
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,0);f.values[2]=FLT_MAX;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,1);f.values[1]=(float)cap;f.values[2]=(float)cap;
    scalar_check(metrics,&f,0);CHECK(f.results[0]==0 && f.results[1]==(float)cap);
    CHECK(f.results[2]==1 && f.results[3]==1);
    scalar_setup(&f,width,1);f.values[1]=(float)(cap+1);
    scalar_check(metrics,&f,ET_KERNEL_ERROR_UNSUPPORTED);
    f.values[0]=NAN;scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,1);f.values[1]=1.5f;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,1);f.values[2]=FLT_MAX;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
    scalar_setup(&f,width,1);f.values[0]=FLT_MAX;
    scalar_check(metrics,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT);
  }
}
static double loss_reference(const float logits[1024], size_t row, int64_t target) {
  double maximum=logits[row*256], sum=0;
  for(size_t j=1;j<256;++j) if(logits[row*256+j]>maximum) maximum=logits[row*256+j];
  for(size_t j=0;j<256;++j) sum+=exp((double)logits[row*256+j]-maximum);
  return log(sum)+maximum-logits[row*256+(size_t)target];
}
static double objective_reference(const float logits[1024], const int64_t targets[4],
                                   const float weights[4], int mean) {
  double n=0,w=0;
  for(size_t i=0;i<4;++i) {n+=(double)weights[i]*loss_reference(logits,i,targets[i]);w+=weights[i];}
  return mean?n/w:n;
}
static void l2_call(et_kernel_runtime *l2, fixture *f, float losses[4],
                     float upstream[4], float gradient[1024]) {
  const uint64_t ts[]={1,4},ls[]={1,4,256};
  et_kernel_tensor_view_v1 in[]={view(f->logits,4096,"f32",3,ls),
    view(f->targets,32,"i64",2,ts),view(upstream,16,"f32",2,ts)};
  et_kernel_tensor_view_v1 out[]={view(gradient?gradient:losses,gradient?4096u:16u,
    "f32",gradient?3u:2u,gradient?ls:ts)};
  dispatch(l2,"kernel.indexed-cross-entropy",gradient?"indexed-cross-entropy.backward":
    "indexed-cross-entropy.forward",3,ls,in,gradient?3u:2u,out,1);
}
static void composition(et_kernel_runtime *l3s, et_kernel_runtime *l2) {
  for(size_t c=0;c+1<sizeof(cpu4_cases)/sizeof(cpu4_cases[0]);++c) {
    const cpu4_case *ref=cpu4_cases+c;fixture f;setup(&f,REDUCE,ref->boolean);
    for(size_t i=0;i<1024;++i) f.logits[i]=(float)((int)(i%31)-15)*0.0625f+(float)(i/256)*0.03125f;
    const int64_t targets[]={3,197,0,255};memcpy(f.targets,targets,sizeof(targets));
    for(size_t i=0;i<4;++i) {f.weights[i]=unbits(ref->weights[i]);f.mask[i]=(uint8_t)f.weights[i];}
    l2_call(l2,&f,f.losses,NULL,NULL);
    for(size_t i=0;i<4;++i) CHECK(fabs((double)f.losses[i]-loss_reference(f.logits,i,f.targets[i]))<0.000002);
    perform(l3s,&f);
    CHECK(fabs((double)f.result[0]-objective_reference(f.logits,f.targets,f.weights,0))<0.00002);
    CHECK(fabs((double)f.result[2]-objective_reference(f.logits,f.targets,f.weights,1))<0.000003);
    for(int mean=0;mean<2;++mean) {
      fixture seed;setup(&seed,mean?MEAN_SEED:NUMERATOR_SEED,ref->boolean);
      memcpy(seed.weights,f.weights,16);memcpy(seed.mask,f.mask,4);perform(l3s,&seed);
      float gradient[1024];l2_call(l2,&f,NULL,seed.result,gradient);
      double w=0;for(size_t i=0;i<4;++i) w+=f.weights[i];
      for(size_t i=0;i<1024;++i) {
        size_t row=i/256;double sum=0,maximum=f.logits[row*256];
        for(size_t j=1;j<256;++j) if(f.logits[row*256+j]>maximum) maximum=f.logits[row*256+j];
        for(size_t j=0;j<256;++j) sum+=exp((double)f.logits[row*256+j]-maximum);
        double probability=exp((double)f.logits[i]-maximum)/sum;
        double expected=(probability-(i%256==(size_t)f.targets[row]?1.0:0.0))*f.weights[row]/(mean?w:1.0);
        CHECK(fabs((double)gradient[i]-expected)<0.0000005);
      }
      for(size_t row=0;row<4;++row) for(size_t probe=0;probe<4;++probe) {
        size_t index=row*256+(probe==0?(size_t)f.targets[row]:probe==1?0u:probe==2?127u:255u);
        float original=f.logits[index];
        f.logits[index]=original+0.015625f;
        double high=objective_reference(f.logits,f.targets,f.weights,mean);
        f.logits[index]=original-0.015625f;
        double low=objective_reference(f.logits,f.targets,f.weights,mean);
        f.logits[index]=original;
        CHECK(fabs((high-low)/0.03125-(double)gradient[index])<0.000003);
      }
    }
  }
}
static void zeros_and_subnormals(et_kernel_runtime *l3s) {
  fixture f;setup(&f,REDUCE,0);
  for(size_t i=0;i<4;++i) {f.losses[i]=-0.0f;f.weights[i]=i==3?unbits(1):-0.0f;}
  perform(l3s,&f);CHECK(bits(f.result[0])==0);CHECK(bits(f.result[1])==1);CHECK(bits(f.result[2])==0);
  setup(&f,MEAN_SEED,0);for(size_t i=0;i<4;++i) f.weights[i]=i==3?unbits(1):-0.0f;
  perform(l3s,&f);for(size_t i=0;i<3;++i) CHECK(bits(f.result[i])==0);CHECK(f.result[3]==1);
}
int main(void) {
  environment();
  et_kernel_runtime *l3s=runtime(et_l3s_kernel_provider_v1());
  et_kernel_runtime *metrics=runtime(et_e3_metrics_kernel_provider_v1());
  et_kernel_runtime *l2=runtime(et_l2_indexed_cross_entropy_provider_v1());
  exact_cases(l3s);correct_cases(metrics);negative_cases(l3s,metrics);
  zeros_and_subnormals(l3s);scalar_cases(metrics);composition(l3s,l2);
  et_kernel_runtime_destroy(l2);et_kernel_runtime_destroy(metrics);et_kernel_runtime_destroy(l3s);
  puts("E3 CPU4 PROVIDER PASS: T4 masks, exact words, atomicity, L2 forward/VJP and finite differences");
  return 0;
}
