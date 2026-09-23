/* Independent scalar oracles; no production source or helper is included. */
#include "test_common.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>

static float add32(float a,float b){volatile float v=a+b;return v;}
static float sub32(float a,float b){volatile float v=a-b;return v;}
static float mul32(float a,float b){volatile float v=a*b;return v;}
static float div32(float a,float b){volatile float v=a/b;return v;}

static void reference(const g3c4_fixture *f,size_t row,float *out) {
  const size_t t=(size_t)f->shape[1];
  if(row<6u) {
    const size_t d=(size_t)f->shape[3];
    for(size_t j=0;j<t;++j) memcpy(out+j*d,f->data[1]+(size_t)f->positions[0][j]*d,d*sizeof(float));
  } else if(row<14u) {
    const size_t di=(size_t)f->shape[2],dout=(size_t)f->shape[3];
    for(size_t j=0;j<t;++j) for(size_t o=0;o<dout;++o) {
      float sum=0.0f;
      for(size_t i=0;i<di;++i) sum=add32(sum,mul32(f->data[0][j*di+i],f->data[1][o*di+i]));
      out[j*dout+o]=sum;
    }
  } else if(row<16u) {
    const size_t d=(size_t)f->shape[2];
    for(size_t j=0;j<t;++j) {
      float mean=0.0f,var=0.0f;
      for(size_t i=0;i<d;++i) mean=add32(mean,f->data[0][j*d+i]);
      mean=div32(mean,(float)d);
      for(size_t i=0;i<d;++i){float delta=sub32(f->data[0][j*d+i],mean);var=add32(var,mul32(delta,delta));}
      var=div32(var,(float)d);float root=sqrtf(add32(var,f->data[3][0]));float rstd=div32(1.0f,root);
      for(size_t i=0;i<d;++i) out[j*d+i]=add32(mul32(mul32(sub32(f->data[0][j*d+i],mean),rstd),f->data[1][i]),f->data[2][i]);
    }
  } else if(row<18u) {
    const size_t n=t*(size_t)f->shape[2];
    for(size_t i=0;i<n;++i) {
      float scaled=mul32(f->data[0][i],0x1.6a09e6p-1f);
      float sum=add32(1.0f,erff(scaled));
      out[i]=mul32(mul32(0.5f,f->data[0][i]),sum);
    }
  } else if(row==18u) {
    for(size_t i=0;i<16u;++i) out[i]=add32(f->data[0][i],f->data[1][i]);
  } else if(row<23u) {
    const size_t tokens=(size_t)f->shape[1];
    if(row<21u) for(size_t j=0;j<tokens;++j) for(size_t h=0;h<2u;++h) for(size_t d=0;d<2u;++d)
      out[(h*tokens+j)*2u+d]=f->data[0][j*4u+h*2u+d];
    else for(size_t j=0;j<tokens;++j) for(size_t h=0;h<2u;++h) for(size_t d=0;d<2u;++d)
      out[j*4u+h*2u+d]=f->data[0][(h*tokens+j)*2u+d];
  } else {
    const size_t tq=(size_t)f->shape[3],tk=(size_t)f->shape[4];
    for(size_t h=0;h<2u;++h) for(size_t q=0;q<tq;++q) {
      float scores[4],maximum=-INFINITY,denom=0.0f;unsigned selected[4]={0};
      for(size_t k=0;k<tk;++k) if(f->keep[q*tk+k] && f->positions[1][k]<=f->positions[0][q]) {
        float dot=0.0f;
        for(size_t d=0;d<2u;++d) dot=add32(dot,mul32(f->data[0][(h*tq+q)*2u+d],f->data[1][(h*tk+k)*2u+d]));
        scores[k]=mul32(dot,div32(1.0f,sqrtf(2.0f)));selected[k]=1u;if(scores[k]>maximum) maximum=scores[k];
      }
      for(size_t k=0;k<tk;++k) if(selected[k]) denom=add32(denom,expf(sub32(scores[k],maximum)));
      for(size_t d=0;d<2u;++d) {
        float sum=0.0f;
        for(size_t k=0;k<tk;++k) if(selected[k]) {
          float probability=div32(expf(sub32(scores[k],maximum)),denom);
          sum=add32(sum,mul32(probability,f->data[2][(h*tk+k)*2u+d]));
        }
        out[(h*tq+q)*2u+d]=sum;
      }
    }
  }
}

/* Algebraic long-double oracle is deliberately separate from the statement-
 * rounded reference above. It checks the mathematical operation, while the
 * first oracle checks the accepted binary32 evaluation order. */
static void math_reference(const g3c4_fixture *f,size_t row,float *out) {
  const size_t t=(size_t)f->shape[1];
  if(row<6u) {
    const size_t d=(size_t)f->shape[3];
    for(size_t j=0;j<t;++j)memcpy(out+j*d,f->data[1]+(size_t)f->positions[0][j]*d,d*4u);
  } else if(row<14u) {
    const size_t di=(size_t)f->shape[2],dout=(size_t)f->shape[3];
    for(size_t j=0;j<t;++j)for(size_t o=0;o<dout;++o){long double sum=0.0L;for(size_t i=0;i<di;++i)sum+=(long double)f->data[0][j*di+i]*f->data[1][o*di+i];out[j*dout+o]=(float)sum;}
  } else if(row<16u) {
    const size_t d=(size_t)f->shape[2];
    for(size_t j=0;j<t;++j){long double mean=0.0L,var=0.0L;for(size_t i=0;i<d;++i)mean+=f->data[0][j*d+i];mean/=(long double)d;for(size_t i=0;i<d;++i){long double delta=(long double)f->data[0][j*d+i]-mean;var+=delta*delta;}var/=(long double)d;long double rstd=1.0L/sqrtl(var+(long double)f->data[3][0]);for(size_t i=0;i<d;++i)out[j*d+i]=(float)(((long double)f->data[0][j*d+i]-mean)*rstd*f->data[1][i]+f->data[2][i]);}
  } else if(row<18u) {
    const size_t n=t*(size_t)f->shape[2];for(size_t i=0;i<n;++i){long double x=f->data[0][i];out[i]=(float)(0.5L*x*(1.0L+erfl(x/sqrtl(2.0L))));}
  } else if(row==18u) {
    for(size_t i=0;i<16u;++i)out[i]=(float)((long double)f->data[0][i]+f->data[1][i]);
  } else if(row<23u) {
    reference(f,row,out);
  } else {
    const size_t tq=(size_t)f->shape[3],tk=(size_t)f->shape[4];
    for(size_t h=0;h<2u;++h)for(size_t q=0;q<tq;++q){long double score[4],maximum=-INFINITY,denom=0.0L;unsigned selected[4]={0};for(size_t k=0;k<tk;++k)if(f->keep[q*tk+k]&&f->positions[1][k]<=f->positions[0][q]){long double dot=0.0L;for(size_t d=0;d<2u;++d)dot+=(long double)f->data[0][(h*tq+q)*2u+d]*f->data[1][(h*tk+k)*2u+d];score[k]=dot/sqrtl(2.0L);selected[k]=1u;if(score[k]>maximum)maximum=score[k];}for(size_t k=0;k<tk;++k)if(selected[k])denom+=expl(score[k]-maximum);for(size_t d=0;d<2u;++d){long double sum=0.0L;for(size_t k=0;k<tk;++k)if(selected[k])sum+=expl(score[k]-maximum)/denom*f->data[2][(h*tk+k)*2u+d];out[(h*tq+q)*2u+d]=(float)sum;}}
  }
}

static void exact_metadata(void) {
  static const char *const names[]={"g3c4.causal-attention","g3c4.embedding-forward","g3c4.gelu","g3c4.head-layout","g3c4.layer-norm","g3c4.linear","g3c4.residual"};
  static const size_t counts[]={5,6,2,2,2,8,1};
  static const size_t first[]={23,0,16,21,14,6,18};
  const et_kernel_provider_v1 *p=et_g3c4_kernel_provider_v1();
  CHECK(p!=NULL);CHECK(p==et_g3c4_kernel_provider_v1());CHECK(p->struct_size==ET_KERNEL_PROVIDER_V1_0_SIZE);
  CHECK(p->abi_major==1u);CHECK(p->abi_minor==0u);CHECK(p->required_features==0u);
  CHECK(strcmp(p->name,"g3c4.cpu-f32.serial")==0);CHECK(strcmp(p->version,"1.0")==0);
  CHECK(strcmp(p->evidence,"G3-C4:cpu-f32-c4-forward-v1")==0);CHECK(p->capability_count==7u);
  CHECK(p->capability_stride==ET_KERNEL_CAPABILITY_V1_0_SIZE);CHECK(p->capability_bytes==7u*ET_KERNEL_CAPABILITY_V1_0_SIZE);
  size_t pairs=0;
  const et_kernel_capability_v1 *capabilities=p->capabilities;
  for(size_t ci=0;ci<7u;++ci) {
    const et_kernel_capability_v1 *c=&capabilities[ci];
    CHECK(c->struct_size==ET_KERNEL_CAPABILITY_V1_0_SIZE);CHECK(strcmp(c->name,names[ci])==0);
    CHECK(c->status==ET_KERNEL_CAPABILITY_VERIFIED);CHECK(strcmp(c->implementation,p->name)==0);
    CHECK(strcmp(c->version,"1.0")==0);CHECK(strcmp(c->evidence,p->evidence)==0);CHECK(c->deterministic==1u);
    CHECK(c->dtype_count==1u);CHECK(strcmp(c->dtypes[0],"f32")==0);CHECK(c->device_count==1u);CHECK(strcmp(c->devices[0],"cpu")==0);
    CHECK(c->shape_range_count==counts[ci]);CHECK(c->operation_count==(ci==3u ? 2u:1u));
    for(size_t z=0;z<sizeof(c->reserved);++z) CHECK(c->reserved[z]==0u);
    for(size_t oi=0;oi<c->operation_count;++oi) for(size_t ri=0;ri<c->shape_range_count;++ri) {
      size_t fixture=first[ci]+ri;
      if(ci==3u) fixture=(oi==0u ? 21u:19u)+ri;
      g3c4_fixture f;fixture_init(&f,fixture);
      CHECK(strcmp(c->operations[oi],f.request.operation)==0);const et_kernel_shape_range_v1 *r=&c->shape_ranges[ri];
      CHECK(r->rank==f.request.rank);
      for(size_t d=0;d<r->rank;++d){CHECK(r->dimensions[d].minimum==f.shape[d]);CHECK(r->dimensions[d].maximum==f.shape[d]);CHECK(r->dimensions[d].maximum_unbounded==0u);for(size_t z=0;z<sizeof(r->dimensions[d].reserved);++z)CHECK(r->dimensions[d].reserved[z]==0u);}
      ++pairs;
    }
  }
  CHECK(pairs==28u);
}

static void all_rows(et_kernel_runtime *rt) {
  for(size_t row=0;row<G3C4_ROWS;++row) {
    g3c4_fixture f;float expected[2048]={0},mathematical[2048]={0};fixture_init(&f,row);reference(&f,row,expected);math_reference(&f,row,mathematical);run(rt,&f.call);
    const size_t n=f.outputs[0].byte_length/4u;CHECK(memcmp(f.output,expected,n*4u)==0);
    expect_floats(f.output,mathematical,n,5e-5f);
    float saved[2048];memcpy(saved,f.output,n*4u);run(rt,&f.call);CHECK(memcmp(saved,f.output,n*4u)==0);
    f.request.deterministic=0u;run(rt,&f.call);CHECK(memcmp(saved,f.output,n*4u)==0);
  }
}

static void rejected(et_kernel_runtime *rt,g3c4_fixture *f) {
  g3c4_fixture before;memcpy(&before,f,sizeof(before));
  et_kernel_error error;int32_t rc=et_kernel_runtime_dispatch(rt,&f->call,&error);
  if(rc==0) fprintf(stderr,"unexpected admission: %s shape[1]=%llu\n",f->request.operation,(unsigned long long)f->shape[1]);
  CHECK(rc==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(error.category==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(&before,f,sizeof(before))==0);
}

static void embedding_and_linear(et_kernel_runtime *rt) {
  for(size_t row=0;row<6u;++row) {
    g3c4_fixture f;fixture_init(&f,row);const size_t v=(size_t)f.shape[2],t=(size_t)f.shape[1];
    for(size_t i=0;i<t;++i) f.positions[0][i]=(int64_t)(v-1u-i%v);run(rt,&f.call);
    for(size_t i=0;i<t;++i) CHECK(memcmp(f.output+4u*i,f.data[1]+4u*(size_t)f.positions[0][i],16u)==0);
    for(size_t slot=0;slot<t;++slot)for(size_t id=0;id<v;++id){fixture_init(&f,row);f.positions[0][slot]=(int64_t)id;run(rt,&f.call);CHECK(memcmp(f.output+4u*slot,f.data[1]+4u*id,16u)==0);}
    f.positions[0][t-1]=-1;rejected(rt,&f);f.positions[0][t-1]=(int64_t)v;rejected(rt,&f);
    fixture_init(&f,row);f.data[1][v*4u-1u]=NAN;rejected(rt,&f);
  }
  for(size_t row=6u;row<14u;++row) {
    g3c4_fixture f;fixture_init(&f,row);const size_t t=(size_t)f.shape[1],di=(size_t)f.shape[2],dout=(size_t)f.shape[3];
    memset(f.data[0],0,t*di*4u);for(size_t i=0;i<di*dout;++i)f.data[1][i]=(float)(i+1u)/2048.0f;
    for(size_t token=0;token<t;++token)for(size_t selected=0;selected<di;++selected){memset(f.data[0],0,t*di*4u);f.data[0][token*di+selected]=1.0f;run(rt,&f.call);for(size_t o=0;o<dout;++o)CHECK(f.output[token*dout+o]==f.data[1][o*di+selected]);}
    fixture_init(&f,row);memset(f.data[0],0,t*di*4u);memset(f.data[1],0,di*dout*4u);
    f.data[0][0]=FLT_MAX;f.data[1][0]=2.0f;rejected(rt,&f);
    fixture_init(&f,row);memset(f.data[0],0,t*di*4u);memset(f.data[1],0,di*dout*4u);
    f.data[0][di-2u]=FLT_MAX;f.data[0][di-1u]=FLT_MAX;f.data[1][di-2u]=1.0f;f.data[1][di-1u]=1.0f;rejected(rt,&f);
    fixture_init(&f,row);memset(f.data[0],0,t*di*4u);for(size_t i=0;i<di*dout;++i)f.data[1][i]=1.0f;
    f.data[0][0]=0x1p24f;f.data[0][1]=1.0f;f.data[0][2]=-0x1p24f;f.data[0][3]=1.0f;run(rt,&f.call);CHECK(fbits(f.output[0])==fbits(1.0f));
    float reverse=0.0f;for(size_t i=di;i>0u;--i)reverse=add32(reverse,f.data[0][i-1u]);CHECK(reverse!=f.output[0]);
    fixture_init(&f,row);memset(f.data[0],0,t*di*4u);memset(f.data[1],0,di*dout*4u);f.data[0][0]=1.0f;f.data[0][1]=0x1.000002p0f;f.data[1][0]=-1.0f;f.data[1][1]=0x1.fffffep-1f;run(rt,&f.call);CHECK(fbits(f.output[0])==0u);CHECK(fmaf(f.data[0][1],f.data[1][1],-1.0f)!=f.output[0]);
  }
}

static void norm_gelu_residual_layout(et_kernel_runtime *rt) {
  for(size_t row=14u;row<16u;++row) {
    g3c4_fixture f;fixture_init(&f,row);const size_t n=(size_t)(f.shape[1]*4u);
    for(size_t i=0;i<n;++i)f.data[0][i]=2.0f;run(rt,&f.call);
    for(size_t j=0;j<(size_t)f.shape[1];++j)CHECK(memcmp(f.output+j*4u,f.data[2],16u)==0);
    fixture_init(&f,row);f.data[3][0]=FLT_TRUE_MIN;run(rt,&f.call);
    static const float bad[]={0.0f,-0.0f,-1.0f,NAN,INFINITY};
    for(size_t i=0;i<COUNT(bad);++i){fixture_init(&f,row);f.data[3][0]=bad[i];rejected(rt,&f);}
    fixture_init(&f,row);f.data[0][0]=FLT_MAX;f.data[0][1]=FLT_MAX;rejected(rt,&f);
    fixture_init(&f,row);f.data[0][0]=1e20f;rejected(rt,&f);
    fixture_init(&f,row);f.data[0][0]=1.0f;f.data[1][0]=FLT_MAX;rejected(rt,&f);
    uint32_t state=UINT32_C(0x1739ac51);
    for(size_t trial=0;trial<64u;++trial){fixture_init(&f,row);for(size_t operand=0;operand<3u;++operand)for(size_t i=0;i<f.inputs[operand].byte_length/4u;++i){state=state*UINT32_C(1664525)+UINT32_C(1013904223);uint32_t word=UINT32_C(0x3f000000)|(state&UINT32_C(0x007fffff));if(state&UINT32_C(0x80000000))word|=UINT32_C(0x80000000);memcpy(&f.data[operand][i],&word,4u);}f.data[3][0]=trial%2u?0.125f:FLT_TRUE_MIN;float literal[2048]={0};reference(&f,row,literal);run(rt,&f.call);CHECK(memcmp(literal,f.output,f.outputs[0].byte_length)==0);}
  }
  for(size_t row=16u;row<18u;++row) {
    g3c4_fixture f;fixture_init(&f,row);const float values[]={-8.0f,-1.0f,-FLT_TRUE_MIN,-0.0f,0.0f,FLT_TRUE_MIN,1.0f,8.0f};
    for(size_t i=0;i<8u;++i)f.data[0][i]=values[i];float expected[2048]={0};reference(&f,row,expected);run(rt,&f.call);
    CHECK(memcmp(f.output,expected,(size_t)(f.shape[1]*8u)*4u)==0);CHECK(signbit(f.output[3]));CHECK(fbits(f.output[4])==0u);
    fixture_init(&f,row);f.data[0][0]=FLT_MAX;run(rt,&f.call);CHECK(isfinite(f.output[0]));
  }
  g3c4_fixture f;fixture_init(&f,18u);const float z[]={-0.0f,0.0f,FLT_TRUE_MIN,-FLT_TRUE_MIN};
  for(size_t i=0;i<16u;++i){f.data[0][i]=z[i%4u];f.data[1][i]=z[i%4u];}float expected[2048]={0};reference(&f,18u,expected);run(rt,&f.call);CHECK(memcmp(f.output,expected,64u)==0);
  fixture_init(&f,18u);f.data[0][15]=FLT_MAX;f.data[1][15]=FLT_MAX;rejected(rt,&f);
  for(size_t row=19u;row<23u;++row){fixture_init(&f,row);const size_t n=(size_t)(f.shape[1]*4u);for(size_t i=0;i<n;++i)f.data[0][i]=(float)(1000u+i);reference(&f,row,expected);run(rt,&f.call);CHECK(memcmp(f.output,expected,n*4u)==0);int differs=memcmp(f.output,f.data[0],n*4u)!=0;CHECK(differs);}
}

static void attention_case(et_kernel_runtime *rt,size_t row) {
  g3c4_fixture f;fixture_init(&f,row);const size_t tq=(size_t)f.shape[3],tk=(size_t)f.shape[4];float expected[2048]={0};
  /* Mark every head/key/value distinctly and exercise causal, explicit mask,
   * multikey normalization and an empty query in one call. */
  for(size_t h=0;h<2u;++h)for(size_t q=0;q<tq;++q)for(size_t d=0;d<2u;++d)f.data[0][(h*tq+q)*2u+d]=(float)(1u+h*3u+q+d)/8.0f;
  for(size_t h=0;h<2u;++h)for(size_t k=0;k<tk;++k)for(size_t d=0;d<2u;++d){f.data[1][(h*tk+k)*2u+d]=(float)((int)(h*5u+k*2u+d)-3)/7.0f;f.data[2][(h*tk+k)*2u+d]=(float)(100u*h+10u*k+d);}
  for(size_t q=0;q<tq;++q)f.positions[0][q]=(int64_t)(2u*q+1u);
  for(size_t k=0;k<tk;++k)f.positions[1][k]=(int64_t)(2u*k);
  memset(f.keep,1,tq*tk);for(size_t q=0;q<tq;++q)if(tk>1u)f.keep[q*tk+(q+1u)%tk]=0u;
  if(tq>1u)memset(f.keep+(tq-1u)*tk,0,tk);
  reference(&f,row,expected);run(rt,&f.call);CHECK(memcmp(f.output,expected,tq*4u*4u)==0);
  if(tq>1u)for(size_t h=0;h<2u;++h)for(size_t d=0;d<2u;++d)CHECK(fbits(f.output[(h*tq+tq-1u)*2u+d])==0u);
  /* Permuting two admitted K/V slots changes this nonuniform softmax oracle. */
  if(tk>1u){float first=f.output[0];for(size_t d=0;d<2u;++d){float x=f.data[1][d];f.data[1][d]=f.data[1][2u+d];f.data[1][2u+d]=x;x=f.data[2][d];f.data[2][d]=f.data[2][2u+d];f.data[2][2u+d]=x;}run(rt,&f.call);CHECK(f.output[0]!=first);}
}

static void attention(et_kernel_runtime *rt) {
  for(size_t row=23u;row<28u;++row) attention_case(rt,row);
  g3c4_fixture f;
  for(size_t row=23u;row<28u;++row) {
    fixture_init(&f,row);const size_t tq=(size_t)f.shape[3],tk=(size_t)f.shape[4];
    /* Every mask cell is independently decisive in both directions, alongside
     * all-empty and all-kept rows. */
    for(size_t pattern=0;pattern<2u+2u*tq*tk;++pattern){fixture_init(&f,row);if(pattern==0u)memset(f.keep,0,tq*tk);else if(pattern==1u)memset(f.keep,1,tq*tk);else{const size_t cell=(pattern-2u)/2u;memset(f.keep,(pattern-2u)%2u?1:0,tq*tk);f.keep[cell]=(uint8_t)(((pattern-2u)%2u)?0u:1u);}float expected[2048]={0};reference(&f,row,expected);run(rt,&f.call);CHECK(memcmp(f.output,expected,f.outputs[0].byte_length)==0);}
    fixture_init(&f,row);for(size_t q=0;q<tq;++q)f.positions[0][q]=16777215-(int64_t)(tq-1u-q);for(size_t k=0;k<tk;++k)f.positions[1][k]=16777215-(int64_t)(tk-1u-k);float maximum_positions[2048]={0};reference(&f,row,maximum_positions);run(rt,&f.call);CHECK(memcmp(f.output,maximum_positions,f.outputs[0].byte_length)==0);
    for(size_t q=0;q<tq;++q)for(size_t k=0;k<tk;++k)f.keep[q*tk+k]=0u;
    f.data[0][0]=FLT_MAX;f.data[1][0]=FLT_MAX;run(rt,&f.call);for(size_t i=0;i<tq*4u;++i)CHECK(fbits(f.output[i])==0u);
    fixture_init(&f,row);f.keep[tk-1u]=2u;rejected(rt,&f);
    fixture_init(&f,row);f.positions[0][0]=-1;rejected(rt,&f);fixture_init(&f,row);f.positions[1][tk-1u]=16777216;rejected(rt,&f);
    if(tq>1u){fixture_init(&f,row);f.positions[0][1]=f.positions[0][0];rejected(rt,&f);}
    if(tk>1u){fixture_init(&f,row);f.positions[1][1]=f.positions[1][0];rejected(rt,&f);}
    fixture_init(&f,row);f.data[2][2u*tk-1u]=NAN;rejected(rt,&f);
  }
  /* Two finite scores, approximately +/-FLT_MAX/sqrt(2), make score-max overflow to -inf.
   * Both denominator and recomputation pass are separately mutation-tested. */
  fixture_init(&f,27u);memset(f.data[0],0,sizeof(f.data[0]));memset(f.data[1],0,sizeof(f.data[1]));
  f.data[0][0]=1.0f;f.data[1][0]=FLT_MAX;f.data[1][2]=-FLT_MAX;f.keep[0]=f.keep[1]=1u;f.positions[0][0]=2;f.positions[0][1]=3;f.positions[0][2]=4;f.positions[1][0]=0;f.positions[1][1]=1;rejected(rt,&f);
  /* Finite negative shifted values may underflow expf to zero. */
  fixture_init(&f,27u);memset(f.data[0],0,sizeof(f.data[0]));memset(f.data[1],0,sizeof(f.data[1]));
  f.data[0][0]=1.0f;f.data[1][0]=0.0f;f.data[1][2]=-200.0f;f.positions[0][0]=2;f.positions[0][1]=3;f.positions[0][2]=4;run(rt,&f.call);CHECK(isfinite(f.output[0]));
  /* All products are finite, but binary32 probability rounding makes the
   * increasing-key value reduction overflow. */
  fixture_init(&f,23u);memset(f.data[0],0,sizeof(f.data[0]));memset(f.data[1],0,sizeof(f.data[1]));memset(f.data[2],0,sizeof(f.data[2]));
  f.data[0][0]=1.0f;f.data[1][0]=0x0p+0f;f.data[1][2]=-0x1.e2993p+1f;f.data[1][4]=-0x1.db81d8p+2f;f.data[1][6]=-0x1.1a793ep+3f;
  f.data[2][0]=f.data[2][2]=f.data[2][4]=f.data[2][6]=FLT_MAX;f.positions[0][0]=3;
  for(size_t k=0;k<4u;++k){f.positions[1][k]=(int64_t)k;f.keep[k]=1u;}
  float scores[4],maximum=-INFINITY,denominator=0.0f,sum=0.0f;
  for(size_t k=0;k<4u;++k){scores[k]=mul32(f.data[1][2u*k],div32(1.0f,sqrtf(2.0f)));CHECK(isfinite(scores[k]));if(scores[k]>maximum)maximum=scores[k];}
  for(size_t k=0;k<4u;++k)denominator=add32(denominator,expf(sub32(scores[k],maximum)));
  for(size_t k=0;k<4u;++k){float probability=div32(expf(sub32(scores[k],maximum)),denominator);float product=mul32(probability,f.data[2][2u*k]);CHECK(isfinite(probability));CHECK(isfinite(product));sum=add32(sum,product);}
  CHECK(!isfinite(sum));rejected(rt,&f);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST)==0);exact_metadata();et_kernel_runtime *rt=runtime_new();
  all_rows(rt);embedding_and_linear(rt);norm_gelu_residual_layout(rt);attention(rt);
  et_kernel_runtime_destroy(rt);printf("G3-C4-N numerical: %zu checks\n",checks);return 0;
}
