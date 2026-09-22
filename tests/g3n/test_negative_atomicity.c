#include "test_cases.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <xmmintrin.h>

static int capture_failure,restore_failure;
int __real_fegetenv(fenv_t *);
int __real_fesetenv(const fenv_t *);
int __wrap_fegetenv(fenv_t *e) {return capture_failure ? -1:__real_fegetenv(e);}
int __wrap_fesetenv(const fenv_t *e) {int rc=__real_fesetenv(e);return restore_failure ? -1:rc;}

static unsigned short cw_read(void) { unsigned short x; __asm__ volatile("fnstcw %0":"=m"(x)); return x; }
static unsigned short sw_read(void) { unsigned short x; __asm__ volatile("fnstsw %0":"=am"(x)); return x; }

/* Include descriptor/shape storage, unused input rows and output guard bytes. */
static void rejected(et_kernel_runtime *rt,fixture *f,int direct,uint32_t category,uint32_t code) {
  fixture before;memcpy(&before,f,sizeof(before));
  const unsigned short cw=cw_read(),sw=sw_read();const unsigned mx=_mm_getcsr();
  et_kernel_error e;
  const int32_t rc=direct ? et_g3n_kernel_provider_v1()->validate_call(&f->call,&e):et_kernel_runtime_dispatch(rt,&f->call,&e);
  if(rc!=(int32_t)category || e.category!=category || e.code!=code)
    fprintf(stderr,"%s %s got %d/%u/%u expected %u/%u: %s\n",direct ? "direct":"K1",f->r.operation,rc,e.category,e.code,category,code,e.message);
  CHECK(rc==(int32_t)category);CHECK(e.category==category);CHECK(e.code==code);
  if(direct) CHECK(strcmp(e.operation,(f->call.struct_size<88 || f->r.struct_size<56) ? "g3n.provider":f->r.operation)==0);
  CHECK(memcmp(&before,f,sizeof(before))==0);
  CHECK(cw_read()==cw);CHECK(sw_read()==sw);CHECK(_mm_getcsr()==mx);
}
#define BOTH(cat,code) do { rejected(rt,f,0,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##code); rejected(rt,f,1,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##code); } while(0)
#define DIFFER(cat,kcode,dcode) do { rejected(rt,f,0,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##kcode); rejected(rt,f,1,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##dcode); } while(0)

static void descriptors(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  setup(f,cap,op,row);const size_t ni=f->call.input_count;
  for(size_t side=0;side<2;++side) for(size_t i=0;i<(side ? 1:ni);++i) {
    setup(f,cap,op,row);et_kernel_tensor_view_v1 *v=side ? &f->out[i]:&f->in[i];
    const et_kernel_tensor_view_v1 original=*v;
    v->struct_size=71;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
    v->struct_size=73;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
    v->dtype="i32";BOTH(DTYPE_MISMATCH,INVALID_TEXT);*v=original;
    v->dtype=NULL;BOTH(DTYPE_MISMATCH,INVALID_TEXT);*v=original;
    v->device="bad device";rejected(rt,f,0,ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_INVALID_TEXT);*v=original;
    v->device=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_INVALID_TEXT);*v=original;
    v->data=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);
    rejected(rt,f,1,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
    v->device="cuda:0";DIFFER(DEVICE_MISMATCH,INVALID_BUFFER,INVALID_TEXT);*v=original;
    v->layout=0;BOTH(NONCONTIGUOUS,INVALID_BUFFER);*v=original;
    v->offset_bytes=4;BOTH(NONCONTIGUOUS,INVALID_BUFFER);*v=original;
    --v->byte_length;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);
    rejected(rt,f,1,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
    if(strcmp(v->dtype,"bool")!=0) {v->data=(unsigned char *)v->data+1;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*v=original;}
    v->data=(void *)(UINTPTR_MAX-((strcmp(v->dtype,"i64")==0) ? 7u:3u));
    if(original.byte_length>4) {DIFFER(INVALID_ARGUMENT,ALIASING_OUTPUT,PROVIDER_REJECTED);}*v=original;
    if(v->rank) {
      v->shape=(const uint64_t *)((const unsigned char *)original.shape+1);BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
      v->shape=NULL;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
      uint64_t *shape=side ? f->osh[i]:f->ish[i];const uint64_t old=shape[0];
      shape[0]=UINT64_MAX;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,strcmp(v->dtype,"bool")==0 ? ET_KERNEL_CODE_INVALID_BUFFER:ET_KERNEL_CODE_INTEGER_OVERFLOW);shape[0]=old;
      /* Same element count, wrong rank isolates provider schema admission. */
      uint64_t flat=original.byte_length/(strcmp(original.dtype,"i64")==0 ? 8u:strcmp(original.dtype,"bool")==0 ? 1u:4u);
      v->rank=original.rank==1 ? 0:1;v->shape=&flat;
      if(v->rank==0) v->byte_length=4;
      BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
      shape[0]=0;v->byte_length=0;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);shape[0]=old;*v=original;
      for(size_t d=0;d<original.rank;++d) for(size_t delta=0;delta<3;++delta) {
        const uint64_t saved=shape[d];shape[d]=delta==0 ? 0:delta==1 ? saved-1:saved+1;
        size_t elements=1;for(size_t k=0;k<original.rank;++k) elements*=(size_t)shape[k];
        v->byte_length=elements*(strcmp(v->dtype,"i64")==0 ? 8u:strcmp(v->dtype,"bool")==0 ? 1u:4u);
        BOTH(SHAPE_MISMATCH,INVALID_SHAPE);shape[d]=saved;*v=original;
      }
    } else {
      uint64_t one=1;v->rank=1;v->shape=&one;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
    }
    if(!side && strcmp(v->dtype,"f32")==0) {
      const uint32_t patterns[]={0x7fc00001u,0x7f800000u,0xff800000u,0x7f800001u};
      for(size_t p=0;p<COUNT(patterns);++p) for(size_t end=0;end<2;++end) {
        const size_t j=end ? v->byte_length/4-1:0;uint32_t saved;
        memcpy(&saved,&f->ib[i].f[j],4);memcpy(&f->ib[i].f[j],&patterns[p],4);
        BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);memcpy(&f->ib[i].f[j],&saved,4);
      }
    }
  }
  for(size_t side=0;side<2;++side) {
    setup(f,cap,op,row);
    size_t *stride=side ? &f->call.output_stride:&f->call.input_stride;
    size_t *bytes=side ? &f->call.output_bytes:&f->call.input_bytes;
    size_t *count=side ? &f->call.output_count:&f->call.input_count;
    const size_t old_stride=*stride,old_bytes=*bytes,old_count=*count;
    *stride=71;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*stride=old_stride;
    *stride=73;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*stride=old_stride;
    --*bytes;DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);*bytes=old_bytes;
    *count=SIZE_MAX;DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);*count=old_count;
    *count=0;*bytes=0;*stride=0;
    if(side) f->call.outputs=NULL;else f->call.inputs=NULL;
    BOTH(INVALID_ARGUMENT,INVALID_BUFFER);
    setup(f,cap,op,row);
    if(side) f->call.outputs=(unsigned char *)f->out+1;else f->call.inputs=(unsigned char *)f->in+1;
    BOTH(INVALID_ARGUMENT,INVALID_BUFFER);
    setup(f,cap,op,row);
    if(side) f->call.outputs=NULL;else f->call.inputs=NULL;
    BOTH(INVALID_ARGUMENT,NULL_ARGUMENT);
    setup(f,cap,op,row);
    if(side) f->call.outputs=(void *)(UINTPTR_MAX-7u);else f->call.inputs=(void *)(UINTPTR_MAX-7u);
    DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);
    setup(f,cap,op,row);
    if(side) {f->call.output_stride=SIZE_MAX-7u;f->call.output_bytes=SIZE_MAX-7u;}
    else {f->call.input_stride=SIZE_MAX-7u;f->call.input_bytes=SIZE_MAX-7u;}
    DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);
  }
  setup(f,cap,op,row);f->call.struct_size=87;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(f,cap,op,row);f->r.struct_size=55;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(f,cap,op,row);f->r.operation="g3n.absent";DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  setup(f,cap,op,row);f->r.dtype="f64";DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  setup(f,cap,op,row);f->r.device="cuda:0";
  /* Generic tensor validation precedes capability matching. */
  for(size_t i=0;i<ni;++i) f->in[i].device="cuda:0";f->out[0].device="cuda:0";
  DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
}

static void requests(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  setup(f,cap,op,row);const size_t rank=f->r.rank;
  for(size_t d=0;d<rank;++d) for(size_t delta=0;delta<3;++delta) {
    setup(f,cap,op,row);const uint64_t old=f->row[d];f->row[d]=delta==0 ? 0:delta==1 ? old-1:old+1;
    DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
    et_kernel_error e;const et_kernel_capability_v1 *found=NULL;
    CHECK(et_kernel_runtime_capability_require(rt,cap->name,&f->r,&found,&e)==ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(e.code==ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);CHECK(found==NULL);
  }
  const size_t ranks[]={0,rank-1,rank+1};
  for(size_t i=0;i<COUNT(ranks);++i) {setup(f,cap,op,row);f->r.rank=ranks[i];DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);}
  static const char *const absent[]={"g3n.embedding.backward","g3n.gelu.forward","g3n.matrix-init.forward","g3n.cache.append","g3n.sample.forward"};
  for(size_t i=0;i<COUNT(absent);++i) {setup(f,cap,op,row);f->r.operation=absent[i];DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);}
  setup(f,cap,op,row);f->call.capability=strcmp(cap->name,"g3n.residual-forward")==0 ? "g3n.embedding-forward":"g3n.residual-forward";
  DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  for(size_t which=0;which<10;++which) {
    setup(f,cap,op,row);uint32_t code=ET_KERNEL_CODE_INVALID_TEXT;
    switch(which) {
      case 0:f->r.deterministic=2;break;
      case 1:f->r.rank=65;break;
      case 2:f->r.shape=NULL;break;
      case 3:f->r.shape=(const uint64_t *)((const uint8_t *)f->row+1);code=ET_KERNEL_CODE_INVALID_BUFFER;break;
      case 4:f->r.shape=(const uint64_t *)(UINTPTR_MAX-7u);code=ET_KERNEL_CODE_INVALID_BUFFER;break;
      case 5:f->call.request=NULL;code=ET_KERNEL_CODE_NULL_ARGUMENT;break;
      case 6:f->call.capability=NULL;code=ET_KERNEL_CODE_NULL_ARGUMENT;break;
      case 7:f->r.operation=NULL;break;
      case 8:f->r.dtype=NULL;break;
      case 9:f->r.device=NULL;break;
    }
    rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,code);
  }
  setup(f,cap,op,row);
  _Alignas(et_kernel_call_v1) unsigned char call_bytes[sizeof(f->call)+8];
  _Alignas(et_kernel_request_v1) unsigned char request_bytes[sizeof(f->r)+8];
  memcpy(call_bytes+1,&f->call,sizeof(f->call));memcpy(request_bytes+1,&f->r,sizeof(f->r));
  et_kernel_error e;const et_kernel_provider_v1 *p=et_g3n_kernel_provider_v1();
  fixture before;memcpy(&before,f,sizeof(before));
  for(size_t failure=0;failure<3;++failure) {
    capture_failure=failure==1;restore_failure=failure==2;
    const int32_t expected=failure==1 ? ET_KERNEL_ERROR_UNSUPPORTED:failure==2 ? ET_KERNEL_ERROR_INTERNAL:ET_KERNEL_ERROR_INVALID_ARGUMENT;
    CHECK(p->validate_call((const et_kernel_call_v1 *)(call_bytes+1),&e)==expected);
    CHECK(e.code==(failure ? ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_INVALID_BUFFER));CHECK(strcmp(e.operation,"g3n.provider")==0);
    f->call.request=(const et_kernel_request_v1 *)(request_bytes+1);
    CHECK(p->validate_call(&f->call,&e)==expected);
    CHECK(e.code==(failure ? ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_INVALID_BUFFER));CHECK(strcmp(e.operation,"g3n.provider")==0);
    f->call.request=&f->r;capture_failure=restore_failure=0;
  }
  CHECK(memcmp(&before,f,sizeof(before))==0);
}

static void aliases(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  setup(f,cap,op,row);const size_t ni=f->call.input_count;
  for(size_t i=0;i<ni;++i) for(size_t j=i+1;j<ni;++j) {
    setup(f,cap,op,row);const int allowed=strstr(f->r.operation,"residual")!=NULL || (strstr(f->r.operation,"attention")!=NULL && i==3 && j==4);
    f->in[j].data=f->in[i].data;
    if(allowed) {CHECK(&f->in[i]!=&f->in[j]);run(rt,&f->call);}else {BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
    setup(f,cap,op,row);
    /* Align to both operand types; one-element i64 positions cannot have a
     * naturally aligned partial overlap, so test the required alignment error. */
    const size_t align=strcmp(f->in[i].dtype,"i64")==0 || strcmp(f->in[j].dtype,"i64")==0 ? 8u:strcmp(f->in[i].dtype,"f32")==0 || strcmp(f->in[j].dtype,"f32")==0 ? 4u:1u;
    if(f->in[i].byte_length>align) {
      f->in[j].data=(unsigned char *)f->in[i].data+align;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    } else if(f->in[j].byte_length>align) {
      f->in[i].data=(unsigned char *)f->in[j].data+align;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    } else if(allowed) {
      f->in[j].data=(unsigned char *)f->in[i].data+1;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);
    }
  }
  for(size_t i=0;i<ni;++i) {
    setup(f,cap,op,row);f->out[0].data=f->in[i].data;
    rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
    setup(f,cap,op,row);f->out[0].data=(unsigned char *)f->in[i].data+1;
    if(f->in[i].byte_length>1) rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
  }
}

static void invalid_values(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  setup(f,cap,op,row);const char *name=f->r.operation;
  if(strstr(name,"embedding")) {
    const int64_t ids[]={-1,(int64_t)f->row[2],INT64_MIN,INT64_MAX};
    for(size_t i=0;i<COUNT(ids);++i) {f->ib[0].i[0]=ids[i];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  } else if(strstr(name,"linear")) {
    const size_t di=(size_t)f->row[2],dout=(size_t)f->row[3];
    for(size_t stage=0;stage<2;++stage) {
      setup(f,cap,op,row);memset(f->ib[1].f,0,f->in[1].byte_length);
      f->ib[0].f[di-2]=FLT_MAX;f->ib[0].f[di-1]=FLT_MAX;
      f->ib[1].f[(dout-1)*di+di-2]=1.0f;f->ib[1].f[(dout-1)*di+di-1]=stage ? 1.0f:2.0f;
      BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    }
  } else if(strstr(name,"residual")) {
    f->ib[0].f[3]=FLT_MAX;f->ib[1].f[3]=FLT_MAX;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
  } else if(strstr(name,"layer-norm")) {
    const uint32_t eps[]={0u,0x80000000u,0xbf800000u,0x7fc00000u,0x7f800000u};
    for(size_t i=0;i<COUNT(eps);++i) {memcpy(&f->ib[3].f[0],&eps[i],4);BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
    /* Ordered sum, deviation, square, square sum, variance+epsilon,
     * affine product and final addition each fail before any output write. */
    for(size_t stage=0;stage<7;++stage) {
      setup(f,cap,op,row);
      for(size_t i=0;i<4;++i) {f->ib[0].f[i]=0;f->ib[1].f[i]=1;f->ib[2].f[i]=0;}
      if(stage==0) f->ib[0].f[0]=f->ib[0].f[1]=FLT_MAX;
      if(stage==1) {f->ib[0].f[0]=FLT_MAX;f->ib[0].f[1]=-FLT_MAX;f->ib[0].f[2]=f->ib[0].f[3]=-FLT_MAX/2;}
      if(stage==2) {f->ib[0].f[0]=1e30f;f->ib[0].f[1]=-1e30f;}
      if(stage==3) {f->ib[0].f[0]=f->ib[0].f[2]=1e19f;f->ib[0].f[1]=f->ib[0].f[3]=-1e19f;}
      if(stage==4) {f->ib[0].f[0]=1e19f;f->ib[0].f[1]=-1e19f;f->ib[3].f[0]=FLT_MAX;}
      if(stage==5) {f->ib[0].f[3]=1;f->ib[1].f[3]=FLT_MAX;}
      if(stage==6) {f->ib[0].f[3]=1;f->ib[1].f[3]=FLT_MAX/2;f->ib[2].f[3]=FLT_MAX;}
      BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    }
  } else if(strstr(name,"attention")) {
    const int64_t bad[]={-1,16777216,INT64_MIN,INT64_MAX};
    for(size_t pos=3;pos<5;++pos) for(size_t i=0;i<COUNT(bad);++i) {
      setup(f,cap,op,row);f->ib[pos].i[0]=bad[i];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    }
    const uint8_t bad_mask[]={2,127,255};
    for(size_t i=0;i<COUNT(bad_mask);++i) {setup(f,cap,op,row);f->ib[5].b[0]=bad_mask[i];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
    for(size_t head=0;head<2;++head) for(size_t stage=0;stage<2;++stage) {
      setup(f,cap,op,row);f->ib[0].f[head*2]=f->ib[0].f[head*2+1]=FLT_MAX;
      f->ib[1].f[head*2]=1;f->ib[1].f[head*2+1]=stage ? 1.0f:2.0f;
      BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    }
  }
}

static void controls(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  fenv_t saved;CHECK(fegetenv(&saved)==0);CHECK(feclearexcept(FE_ALL_EXCEPT)==0);
  const unsigned short cw=cw_read();const unsigned mx=_mm_getcsr();
  const unsigned changes[]={0x40u,0x8000u,0x2000u,0x4000u,0x6000u};
  setup(f,cap,op,row);
  for(size_t i=0;i<COUNT(changes);++i) {_mm_setcsr(mx|changes[i]);BOTH(UNSUPPORTED,PROVIDER_REJECTED);_mm_setcsr(mx);}
  for(size_t i=0;i<6;++i) {
    _mm_setcsr(mx&~(1u<<(7u+i)));BOTH(UNSUPPORTED,PROVIDER_REJECTED);_mm_setcsr(mx);
    unsigned short bad=(unsigned short)(cw&~(1u<<i));__asm__ volatile("fldcw %0"::"m"(bad));
    BOTH(UNSUPPORTED,PROVIDER_REJECTED);__asm__ volatile("fldcw %0"::"m"(cw));
  }
  for(unsigned i=0x400;i<=0xc00;i+=0x400) {
    unsigned short bad=(unsigned short)(cw|i);__asm__ volatile("fldcw %0"::"m"(bad));
    BOTH(UNSUPPORTED,PROVIDER_REJECTED);__asm__ volatile("fldcw %0"::"m"(cw));
  }
  for(size_t fail=0;fail<2;++fail) {
    setup(f,cap,op,row);
    if(fail) for(size_t i=0;i<f->call.input_count;++i) if(strcmp(f->in[i].dtype,"f32")==0) {uint32_t sn=0x7f800001u;memcpy(f->in[i].data,&sn,4);break;}
    CHECK(feclearexcept(FE_ALL_EXCEPT)==0);CHECK(feraiseexcept(FE_DIVBYZERO)==0);
    _mm_setcsr((_mm_getcsr()&~0x3fu)|0x20u);
    const unsigned short sw=sw_read();const unsigned before_mx=_mm_getcsr();fixture before;memcpy(&before,f,sizeof(before));
    et_kernel_error e;const int32_t rc=et_g3n_kernel_provider_v1()->validate_call(&f->call,&e);
    CHECK((rc!=0)==(fail!=0));CHECK(sw_read()==sw);CHECK(cw_read()==cw);CHECK(_mm_getcsr()==before_mx);CHECK(memcmp(&before,f,sizeof(before))==0);
  }
  CHECK(fesetenv(&saved)==0);
  setup(f,cap,op,row);capture_failure=1;BOTH(UNSUPPORTED,PROVIDER_REJECTED);capture_failure=0;
  setup(f,cap,op,row);restore_failure=1;BOTH(INTERNAL,PROVIDER_REJECTED);restore_failure=0;
}

static void future_tails(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  struct extended_view {et_kernel_tensor_view_v1 v;uint64_t tail[2];} in[6],out[1];
  struct extended_request {et_kernel_request_v1 r;uint64_t tail[2];} request;
  struct extended_call {et_kernel_call_v1 c;uint64_t tail[2];} call;
  setup(f,cap,op,row);memset(in,0xa5,sizeof(in));memset(out,0xa5,sizeof(out));
  memset(&request,0xa5,sizeof(request));memset(&call,0xa5,sizeof(call));
  for(size_t i=0;i<f->call.input_count;++i) {in[i].v=f->in[i];in[i].v.struct_size=sizeof(in[i]);}
  out[0].v=f->out[0];out[0].v.struct_size=sizeof(out[0]);
  request.r=f->r;request.r.struct_size=sizeof(request);memset(request.r.reserved,0xa5,sizeof(request.r.reserved));
  call.c=kc(cap->name,&request.r,&in[0].v,f->call.input_count,&out[0].v,1);
  call.c.struct_size=sizeof(call);call.c.input_stride=sizeof(in[0]);call.c.input_bytes=call.c.input_count*sizeof(in[0]);
  call.c.output_stride=sizeof(out[0]);call.c.output_bytes=sizeof(out);
  et_kernel_error e;CHECK(et_g3n_kernel_provider_v1()->validate_call(&call.c,&e)==0);run(rt,&call.c);
  for(size_t i=0;i<call.c.input_count;++i) CHECK(in[i].tail[0]==UINT64_C(0xa5a5a5a5a5a5a5a5));
  CHECK(out[0].tail[1]==UINT64_C(0xa5a5a5a5a5a5a5a5));CHECK(request.tail[0]==UINT64_C(0xa5a5a5a5a5a5a5a5));CHECK(call.tail[1]==UINT64_C(0xa5a5a5a5a5a5a5a5));
}

static void short_prefixes(void) {
  et_kernel_error e;const et_kernel_provider_v1 *p=et_g3n_kernel_provider_v1();
  size_t *tiny=malloc(sizeof(*tiny));CHECK(tiny!=NULL);*tiny=sizeof(*tiny);
  CHECK(p->validate_call((const et_kernel_call_v1 *)tiny,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);
  CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);CHECK(strcmp(e.operation,"g3n.provider")==0);
  et_kernel_call_v1 c={0};c.struct_size=sizeof(c);c.request=(const et_kernel_request_v1 *)tiny;
  CHECK(p->validate_call(&c,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  CHECK(p->validate_call(NULL,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_NULL_ARGUMENT);
  free(tiny);
}

int main(void) {
  et_kernel_runtime *rt=runtime_new();fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);
  const et_kernel_provider_v1 *p=et_g3n_kernel_provider_v1();const et_kernel_capability_v1 *caps=p->capabilities;size_t rows=0;
  for(size_t ci=0;ci<p->capability_count;++ci) for(size_t oi=0;oi<caps[ci].operation_count;++oi) for(size_t ri=0;ri<caps[ci].shape_range_count;++ri) {
    descriptors(rt,f,&caps[ci],oi,ri);requests(rt,f,&caps[ci],oi,ri);aliases(rt,f,&caps[ci],oi,ri);invalid_values(rt,f,&caps[ci],oi,ri);controls(rt,f,&caps[ci],oi,ri);future_tails(rt,f,&caps[ci],oi,ri);++rows;
  }
  CHECK(rows==11);short_prefixes();free(f);et_kernel_runtime_destroy(rt);
  printf("G3-N negative atomicity: %zu checks\n",checks);return 0;
}
