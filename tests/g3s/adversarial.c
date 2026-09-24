/* Independent contract/adversarial caller. No production implementation inclusion. */
#include "eshkol_transformer/g3s_sampling_abi.h"
#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include <fenv.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr,"G3S adversarial:%d: %s\n",__LINE__,#x); exit(1); } } while(0)
static int allocation_disabled;
static size_t blocked_allocations;
static size_t live_allocations;
static int fail_observe,fail_restore;
static size_t exponential_calls;
int __real_fegetenv(fenv_t *);
int __real_fesetenv(const fenv_t *);
float __real_expf(float);
int __wrap_fegetenv(fenv_t *e) {return fail_observe?-1:__real_fegetenv(e);}
int __wrap_fesetenv(const fenv_t *e) {return fail_restore?-1:__real_fesetenv(e);}
float __wrap_expf(float x) {++exponential_calls;return __real_expf(x);}
void *__real_malloc(size_t);
void *__real_calloc(size_t,size_t);
void *__real_realloc(void *,size_t);
void __real_free(void *);
void *__wrap_malloc(size_t n) { if(allocation_disabled) { ++blocked_allocations; return NULL; } void *p=__real_malloc(n);if(p) ++live_allocations;return p; }
void *__wrap_calloc(size_t n,size_t s) { if(allocation_disabled) { ++blocked_allocations; return NULL; } void *p=__real_calloc(n,s);if(p) ++live_allocations;return p; }
void *__wrap_realloc(void *p,size_t n) { if(allocation_disabled) { ++blocked_allocations; return NULL; } void *q=__real_realloc(p,n);if(q&&!p) ++live_allocations;if(!q&&p&&n==0) --live_allocations;return q; }
void __wrap_free(void *p) { if(allocation_disabled) ++blocked_allocations; if(p) --live_allocations;__real_free(p); }
typedef union { uint64_t align; unsigned char bytes[1032]; float f[258]; int64_t i[129]; } payload;
typedef struct { payload b[7]; uint64_t row[3],sh[7][3]; et_kernel_tensor_view_v1 in[5],out[2]; et_kernel_request_v1 req; et_kernel_call_v1 call; } fixture;
static et_kernel_runtime *runtime;
static const et_kernel_provider_v1 *provider;
static const et_kernel_provider_v1 *resolve(void *ctx,const char *name) { (void)ctx; return strcmp(name,ET_KERNEL_PROVIDER_SYMBOL_V1)==0 ? provider:NULL; }
static et_kernel_tensor_view_v1 view(void *data,size_t n,const char *dtype,size_t rank,const uint64_t *shape) {
 et_kernel_tensor_view_v1 v={sizeof(v),data,n,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape}; return v;
}
static void setup(fixture *f,int cat) {
 memset(f,0,sizeof(*f)); f->row[0]=1;f->row[1]=256;
 for(size_t i=0;i<256;++i) f->b[0].f[i]=(float)(i%13)*0.125f;
 f->sh[0][0]=1;f->sh[0][1]=256;f->in[0]=view(f->b[0].bytes,1024,"f32",2,f->sh[0]);
 size_t rng=cat?4:1;f->sh[rng][0]=4; f->b[rng].i[0]=1; f->b[rng].i[1]=7;
 f->in[rng]=view(f->b[rng].bytes,32,"i64",1,f->sh[rng]);
 if(cat) { f->b[1].f[0]=1;f->b[2].i[0]=256;f->b[3].f[0]=1; f->in[1]=view(f->b[1].bytes,4,"f32",0,NULL);f->in[2]=view(f->b[2].bytes,8,"i64",0,NULL);f->in[3]=view(f->b[3].bytes,4,"f32",0,NULL); }
 f->sh[5][0]=1;f->sh[6][0]=4; memset(f->b[5].bytes,0xa5,sizeof(payload));memset(f->b[6].bytes,0x5a,sizeof(payload));
 f->out[0]=view(f->b[5].bytes,8,"i64",1,f->sh[5]); f->out[1]=view(f->b[6].bytes,32,"i64",1,f->sh[6]);
 f->req=(et_kernel_request_v1){sizeof(f->req),cat?"g3s.categorical.forward":"g3s.greedy.forward","f32","cpu",2,f->row,1,{0}};
 size_t n=cat?5:2;f->call=(et_kernel_call_v1){sizeof(f->call),cat?"g3s.categorical":"g3s.greedy",&f->req,n,sizeof(f->in[0]),n*sizeof(f->in[0]),f->in,2,sizeof(f->out[0]),sizeof(f->out),f->out};
}
static unsigned short cw(void) { unsigned short v;__asm__ volatile("fnstcw %0":"=m"(v));return v; }
static unsigned short sw(void) { unsigned short v;__asm__ volatile("fnstsw %0":"=am"(v));return v; }
static void putcw(unsigned short v) { __asm__ volatile("fldcw %0"::"m"(v)); }
static void reject(fixture *f,int direct,uint32_t category,uint32_t code) {
 fixture before=*f;et_kernel_error e;unsigned mx=_mm_getcsr();unsigned short c=cw(),s=sw();
 int32_t rc=direct?provider->validate_call(&f->call,&e):et_kernel_runtime_dispatch(runtime,&f->call,&e);
 if(rc==0 || e.category!=category || (code!=UINT32_MAX && e.code!=code)) fprintf(stderr,"negative direct=%d actual %d/%u/%u expected %u/%u %s\n",direct,rc,e.category,e.code,category,code,e.message);
 CHECK(rc==(int32_t)category);CHECK(e.category==category);if(code!=UINT32_MAX) CHECK(e.code==code);
 CHECK(memcmp(&before,f,sizeof(*f))==0);CHECK(_mm_getcsr()==mx);CHECK(cw()==c);CHECK(sw()==s);
}
#define REJ(d,cat,code) reject(&f,d,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##code)
static void schema(void) {
 fixture f;et_kernel_error e;
 for(int cat=0;cat<2;++cat) for(int d=0;d<2;++d) {
  setup(&f,cat);CHECK(provider->validate_call(&f.call,&e)==0);
  for(unsigned det=0;det<2;++det) {f.req.deterministic=(uint8_t)det;CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&e)==0);}
  for(size_t side=0;side<2;++side) for(size_t i=0;i<(side?2:(cat?5:2));++i) {
   setup(&f,cat);et_kernel_tensor_view_v1 *v=side?f.out+i:f.in+i;et_kernel_tensor_view_v1 original=*v;
   v->struct_size=71;REJ(d,VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
   v->struct_size=80;REJ(d,VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
   v->dtype=strcmp(v->dtype,"f32")==0?"i32":"f64";REJ(d,DTYPE_MISMATCH,INVALID_TEXT);*v=original;
   v->device="cuda:0";reject(&f,d,ET_KERNEL_ERROR_DEVICE_MISMATCH,d?ET_KERNEL_CODE_INVALID_TEXT:ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
   v->layout=0;REJ(d,NONCONTIGUOUS,INVALID_BUFFER);*v=original;
   v->offset_bytes=4;REJ(d,NONCONTIGUOUS,INVALID_BUFFER);*v=original;
   v->byte_length--;reject(&f,d,d?ET_KERNEL_ERROR_INVALID_ARGUMENT:ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
   v->byte_length++;reject(&f,d,d?ET_KERNEL_ERROR_INVALID_ARGUMENT:ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
   v->data=(unsigned char *)v->data+1;REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
   v->data=NULL;reject(&f,d,d?ET_KERNEL_ERROR_INVALID_ARGUMENT:ET_KERNEL_ERROR_SHAPE_MISMATCH,UINT32_MAX);*v=original;
   v->data=(void *)(UINTPTR_MAX-(strcmp(v->dtype,"f32")==0?3u:7u));reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_ALIASING_OUTPUT);*v=original;
   uint64_t shape[3]={1,1,1};v->shape=shape;v->rank=original.rank==0?1:0;v->byte_length=strcmp(v->dtype,"f32")==0?4:8;REJ(d,SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
   if(v->rank) {
    size_t slot=side?5+i:i;
    for(size_t dimension=0;dimension<v->rank;++dimension) for(int delta=-1;delta<=1;delta+=2) {
     uint64_t old=f.sh[slot][dimension];f.sh[slot][dimension]=(uint64_t)((int64_t)old+delta);
     size_t bytes=strcmp(v->dtype,"f32")==0?4:8;for(size_t dim=0;dim<v->rank;++dim) bytes*=f.sh[slot][dim];v->byte_length=bytes;
     REJ(d,SHAPE_MISMATCH,INVALID_SHAPE);f.sh[slot][dimension]=old;*v=original;
    }
    uint64_t old=f.sh[slot][0];f.sh[slot][0]=UINT64_MAX;reject(&f,d,ET_KERNEL_ERROR_SHAPE_MISMATCH,d?ET_KERNEL_CODE_INVALID_SHAPE:ET_KERNEL_CODE_INTEGER_OVERFLOW);f.sh[slot][0]=old;
    v->shape=(const uint64_t *)((const unsigned char *)original.shape+1);REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
    v->shape=(const uint64_t *)(UINTPTR_MAX-(v->rank*sizeof(uint64_t)-1u));REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
   }
   if(v->rank) {v->shape=NULL;reject(&f,d,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE);*v=original;}
  }
  if(cat) {
   setup(&f,cat);f.b[1].f[0]=2;f.b[3].f[0]=0.75f;et_kernel_tensor_view_v1 swap=f.in[1];f.in[1]=f.in[3];f.in[3]=swap;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);
   setup(&f,cat);swap=f.in[2];f.in[2]=f.in[4];f.in[4]=swap;REJ(d,SHAPE_MISMATCH,INVALID_SHAPE);
  }
  for(size_t i=0;i<(cat?5u:2u);++i) for(size_t j=i+1;j<(cat?5u:2u);++j) {setup(&f,cat);f.in[j].data=f.in[i].data;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);}
  for(size_t i=0;i<2;++i) for(size_t j=0;j<(cat?5u:2u);++j) {setup(&f,cat);f.out[i].data=f.in[j].data;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);}
  /* Every legal partial overlap in both directions; scalar extents equal to
   * their alignment have no partially overlapping aligned placement. */
  for(size_t i=0;i<(cat?5u:2u);++i) for(size_t j=0;j<(cat?5u:2u);++j) if(i!=j) {
   setup(&f,cat);size_t align=strcmp(f.in[i].dtype,"f32")==0?4:8;
   if(align<f.in[j].byte_length) {f.in[i].data=(unsigned char *)f.in[j].data+align;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);}
  }
  for(size_t i=0;i<2;++i) for(size_t j=0;j<(cat?5u:2u);++j) {
   setup(&f,cat);if(f.in[j].byte_length>8) {f.out[i].data=(unsigned char *)f.in[j].data+8;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);}
   setup(&f,cat);size_t align=strcmp(f.in[j].dtype,"f32")==0?4:8;
   if(align<f.out[i].byte_length) {f.in[j].data=(unsigned char *)f.out[i].data+align;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);}
  }
  setup(&f,cat);f.out[0].data=(unsigned char *)f.out[1].data+8;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);

  setup(&f,cat);f.out[1].data=f.out[0].data;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);
  setup(&f,cat);f.in[0].data=(unsigned char *)f.out[1].data+8;REJ(d,INVALID_ARGUMENT,ALIASING_OUTPUT);
  for(size_t n=0;n<8;++n) if(n!=(cat?5u:2u)) {setup(&f,cat);f.call.input_count=n;reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);}
  for(size_t n=0;n<5;++n) if(n!=2) {setup(&f,cat);f.call.output_count=n;reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);}
  for(int side=0;side<2;++side) {
   setup(&f,cat);if(side) {f.call.output_count=0;f.call.outputs=NULL;f.call.output_stride=0;f.call.output_bytes=0;}else {f.call.input_count=0;f.call.inputs=NULL;f.call.input_stride=0;f.call.input_bytes=0;}reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
   setup(&f,cat);if(side) f.call.output_count=SIZE_MAX;else f.call.input_count=SIZE_MAX;reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
   setup(&f,cat);if(side) f.call.output_stride=71;else f.call.input_stride=71;REJ(d,VERSION_MISMATCH,INVALID_STRUCT_SIZE);
   setup(&f,cat);if(side) f.call.output_bytes--;else f.call.input_bytes--;reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
   setup(&f,cat);if(side) f.call.output_stride=73;else f.call.input_stride=73;REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);
   setup(&f,cat);if(side) f.call.output_bytes++;else f.call.input_bytes++;reject(&f,d,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
   setup(&f,cat);if(side) f.call.output_bytes=SIZE_MAX;else f.call.input_bytes=SIZE_MAX;REJ(d,INVALID_ARGUMENT,INTEGER_OVERFLOW);
   setup(&f,cat);if(side) f.call.outputs=NULL;else f.call.inputs=NULL;REJ(d,INVALID_ARGUMENT,NULL_ARGUMENT);
   setup(&f,cat);if(side) f.call.outputs=(unsigned char *)f.out+1;else f.call.inputs=(unsigned char *)f.in+1;REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);
   setup(&f,cat);if(side) f.call.output_stride=SIZE_MAX-7;else f.call.input_stride=SIZE_MAX-7;REJ(d,INVALID_ARGUMENT,INTEGER_OVERFLOW);
   setup(&f,cat);if(side) f.call.outputs=(void *)(UINTPTR_MAX-7);else f.call.inputs=(void *)(UINTPTR_MAX-7);REJ(d,INVALID_ARGUMENT,INTEGER_OVERFLOW);
  }
  setup(&f,cat);f.call.struct_size=87;REJ(d,VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(&f,cat);f.req.struct_size=55;REJ(d,VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(&f,cat);f.call.request=NULL;REJ(d,INVALID_ARGUMENT,NULL_ARGUMENT);
  setup(&f,cat);f.req.operation="g3s.absent.forward";reject(&f,d,ET_KERNEL_ERROR_UNSUPPORTED,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  setup(&f,cat);f.call.capability=cat?"g3s.greedy":"g3s.categorical";reject(&f,d,ET_KERNEL_ERROR_UNSUPPORTED,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  const uint64_t rows[][2]={{0,256},{2,256},{1,255},{1,257},{256,1}};
  for(size_t i=0;i<sizeof(rows)/sizeof(rows[0]);++i) {setup(&f,cat);memcpy(f.row,rows[i],sizeof(rows[i]));reject(&f,d,ET_KERNEL_ERROR_UNSUPPORTED,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);}
  for(size_t rank=0;rank<4;++rank) if(rank!=2) {setup(&f,cat);f.req.rank=rank;reject(&f,d,ET_KERNEL_ERROR_UNSUPPORTED,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);}
 }
 /* Future tails are opaque; request/call may advertise prefix-compatible growth. */
 for(int cat=0;cat<2;++cat) {
  setup(&f,cat);struct wide {et_kernel_tensor_view_v1 v;uint64_t tail[2];} in[5],out[2];
  for(size_t i=0;i<f.call.input_count;++i) {memset(in+i,0xa5,sizeof(*in));in[i].v=f.in[i];in[i].v.struct_size=sizeof(*in);}
  for(size_t i=0;i<2;++i) {memset(out+i,0x5a,sizeof(*out));out[i].v=f.out[i];out[i].v.struct_size=sizeof(*out);}
  f.call.inputs=in;f.call.input_stride=sizeof(*in);f.call.input_bytes=f.call.input_count*sizeof(*in);f.call.outputs=out;f.call.output_stride=sizeof(*out);f.call.output_bytes=sizeof(out);
  struct {et_kernel_request_v1 prefix;uint64_t tail[2];} request={f.req,{UINT64_MAX,UINT64_MAX}};
  struct {et_kernel_call_v1 prefix;uint64_t tail[2];} call={f.call,{UINT64_MAX,UINT64_MAX}};
  request.prefix.struct_size=sizeof(request);call.prefix.struct_size=sizeof(call);call.prefix.request=&request.prefix;
  CHECK(provider->validate_call(&call.prefix,&e)==0);CHECK(et_kernel_runtime_dispatch(runtime,&call.prefix,&e)==0);
  for(size_t i=0;i<2;++i) {CHECK(request.tail[i]==UINT64_MAX);CHECK(call.tail[i]==UINT64_MAX);}
  for(size_t i=0;i<f.call.input_count;++i) for(size_t j=0;j<2;++j) CHECK(in[i].tail[j]==UINT64_C(0xa5a5a5a5a5a5a5a5));
  for(size_t i=0;i<2;++i) for(size_t j=0;j<2;++j) CHECK(out[i].tail[j]==UINT64_C(0x5a5a5a5a5a5a5a5a));
 }
}
static void prefixes(void) {
 fixture f;et_kernel_error e;size_t *tiny=malloc(sizeof(*tiny));CHECK(tiny!=NULL);*tiny=sizeof(*tiny);
 CHECK(provider->validate_call((const et_kernel_call_v1 *)tiny,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);CHECK(strcmp(e.operation,"g3s.provider")==0);
 CHECK(et_kernel_runtime_dispatch(runtime,(const et_kernel_call_v1 *)tiny,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
 setup(&f,1);f.call.request=(const et_kernel_request_v1 *)tiny;CHECK(provider->validate_call(&f.call,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);free(tiny);
 CHECK(provider->validate_call(NULL,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_NULL_ARGUMENT);CHECK(strcmp(e.operation,"g3s.provider")==0);
 CHECK(et_kernel_runtime_dispatch(runtime,NULL,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_NULL_ARGUMENT);
 setup(&f,1);const et_kernel_call_v1 *unaligned=(const et_kernel_call_v1 *)((const unsigned char *)&f.call+1);
 CHECK(provider->validate_call(unaligned,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_INVALID_BUFFER);
 CHECK(et_kernel_runtime_dispatch(runtime,unaligned,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_INVALID_BUFFER);
 setup(&f,1);CHECK(provider->validate_call(&f.call,NULL)==0);
 const char *names[]={"foreign.operation","g","g3","g3s","g3s.","g3s.absent.forward"};
 for(size_t i=0;i<sizeof(names)/sizeof(names[0]);++i) {
  setup(&f,1);f.req.operation=names[i];fixture before=f;
  CHECK(provider->validate_call(&f.call,&e)==ET_KERNEL_ERROR_UNSUPPORTED);CHECK(e.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(strcmp(e.operation,i>=4?names[i]:"g3s.provider")==0);CHECK(memcmp(&before,&f,sizeof(f))==0);
 }

 for(int d=0;d<2;++d) {
  for(size_t i=0;i<sizeof(f.req.reserved);++i) {setup(&f,1);f.req.reserved[i]=1;REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);}
  setup(&f,1);f.req.deterministic=2;reject(&f,d,d?ET_KERNEL_ERROR_UNSUPPORTED:ET_KERNEL_ERROR_INVALID_ARGUMENT,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_INVALID_TEXT);
  setup(&f,1);f.req.dtype="f16";reject(&f,d,ET_KERNEL_ERROR_UNSUPPORTED,d?ET_KERNEL_CODE_PROVIDER_REJECTED:ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  setup(&f,1);f.req.device="cuda:0";/* Generic tensor/device agreement precedes K1 row admission. */
  if(d) REJ(d,UNSUPPORTED,PROVIDER_REJECTED);else REJ(d,DEVICE_MISMATCH,INVALID_BUFFER);
  setup(&f,1);f.req.shape=NULL;reject(&f,d,d?ET_KERNEL_ERROR_SHAPE_MISMATCH:ET_KERNEL_ERROR_INVALID_ARGUMENT,d?ET_KERNEL_CODE_INVALID_SHAPE:ET_KERNEL_CODE_INVALID_TEXT);
  setup(&f,1);f.req.shape=(const uint64_t *)((const unsigned char *)f.row+1);REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);
  setup(&f,1);f.req.shape=(const uint64_t *)(UINTPTR_MAX-15u);REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);
  setup(&f,1);f.call.request=(const et_kernel_request_v1 *)((const unsigned char *)&f.req+1);REJ(d,INVALID_ARGUMENT,INVALID_BUFFER);
 }
}
static void adjacent_storage(void) {
 fixture f;et_kernel_error e;
 union {uint64_t align;unsigned char bytes[1112];} packed;
 const size_t input_offsets[5]={88,0,8,4,16};
 for(int cat=0;cat<2;++cat) {
  setup(&f,cat);memset(packed.bytes,0xcc,sizeof(packed.bytes));
  et_kernel_tensor_view_v1 inputs[5],outputs[2];
  for(size_t i=0;i<f.call.input_count;++i) {inputs[i]=f.in[i];inputs[i].data=packed.bytes+(cat?input_offsets[i]:(i?16:88));memcpy(inputs[i].data,f.in[i].data,inputs[i].byte_length);}
  outputs[0]=f.out[0];outputs[0].data=packed.bytes+48;outputs[1]=f.out[1];outputs[1].data=packed.bytes+56;
  et_kernel_call_v1 call=f.call;call.inputs=inputs;call.outputs=outputs;
  unsigned char saved[sizeof(packed.bytes)];memcpy(saved,packed.bytes,sizeof(saved));
  CHECK(provider->validate_call(&call,&e)==0);CHECK(memcmp(saved,packed.bytes,sizeof(saved))==0);
  CHECK(et_kernel_runtime_dispatch(runtime,&call,&e)==0);CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&e)==0);
  for(size_t i=0;i<f.call.input_count;++i) CHECK(memcmp(inputs[i].data,f.in[i].data,inputs[i].byte_length)==0);
  for(size_t i=0;i<2;++i) CHECK(memcmp(outputs[i].data,f.out[i].data,outputs[i].byte_length)==0);
 }
}
static void numerical_rejects(void) {
 fixture f;
 const uint32_t bad[]={0x7f800000,0xff800000,0x7fc00001,0x7f800001};
 for(int cat=0;cat<2;++cat) for(int d=0;d<2;++d) {
  for(size_t j=0;j<256;++j) for(size_t k=0;k<4;++k) {setup(&f,cat);memcpy(f.b[0].f+j,bad+k,4);REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);}
  for(size_t k=0;k<2;++k) {setup(&f,cat);f.b[cat?4:1].i[k]=k?-1:2;reject(&f,d,k?ET_KERNEL_ERROR_INVALID_ARGUMENT:ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_PROVIDER_REJECTED);}
  if(!cat) {setup(&f,cat);f.b[1].i[2]=-1;f.b[1].i[3]=-1;et_kernel_error e;CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&e)==0);CHECK(memcmp(f.b[1].i,f.b[6].i,32)==0);continue;}
  const uint32_t policy[]={0,0x80000000,0xbf800000,0x7f800000,0xff800000,0x7fc00001,0x7f800001};
  for(size_t j=1;j<=3;j+=2) for(size_t k=0;k<sizeof(policy)/sizeof(policy[0]);++k) {setup(&f,cat);memcpy(f.b[j].f,policy+k,4);REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);}
  const int64_t ks[]={INT64_MIN,-1,0,257,INT64_MAX};for(size_t j=0;j<5;++j) {setup(&f,cat);f.b[2].i[0]=ks[j];REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);}
  setup(&f,cat);f.b[3].f[0]=1.00000011920928955078125f;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);
  setup(&f,cat);f.b[0].f[0]=FLT_MAX;f.b[0].f[255]=-FLT_MAX;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);
  setup(&f,cat);f.b[1].f[0]=FLT_TRUE_MIN;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);
  setup(&f,cat);f.b[4].i[2]=-1;f.b[4].i[3]=-1;REJ(d,INVALID_ARGUMENT,PROVIDER_REJECTED);
 }
}
static void fenv_failures(void) {
 fixture f;et_kernel_error e;setup(&f,1);fixture before=f;
 fail_observe=1;CHECK(provider->validate_call(&f.call,&e)==ET_KERNEL_ERROR_UNSUPPORTED);fail_observe=0;CHECK(e.code==ET_KERNEL_CODE_PROVIDER_REJECTED);CHECK(memcmp(&before,&f,sizeof(f))==0);
 fenv_t saved;CHECK(fegetenv(&saved)==0);fail_restore=1;CHECK(provider->validate_call(&f.call,&e)==ET_KERNEL_ERROR_INTERNAL);fail_restore=0;CHECK(fesetenv(&saved)==0);CHECK(e.code==ET_KERNEL_CODE_PROVIDER_REJECTED);CHECK(memcmp(&before,&f,sizeof(f))==0);
 setup(&f,1);f.b[4].i[2]=f.b[4].i[3]=-1;exponential_calls=0;REJ(1,INVALID_ARGUMENT,PROVIDER_REJECTED);CHECK(exponential_calls==0);
 setup(&f,0);CHECK(provider->validate_call(&f.call,&e)==0);provider->invoke_call(&f.call);CHECK(exponential_calls==0);
}
static void fenv_cases(void) {
 fenv_t saved;CHECK(fegetenv(&saved)==0);CHECK(feclearexcept(FE_ALL_EXCEPT)==0);unsigned mx=_mm_getcsr();unsigned short c=cw();fixture f;et_kernel_error e;
 for(int cat=0;cat<2;++cat) for(int direct=0;direct<2;++direct) {
  setup(&f,cat);uint32_t sn=0x7f800001;memcpy(f.b[0].f,&sn,4);
  const unsigned changes[]={0x40,0x8000,0x2000,0x4000,0x6000};
  for(size_t i=0;i<5;++i) {_mm_setcsr(mx|changes[i]);reject(&f,direct,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);_mm_setcsr(mx);}
  for(unsigned i=0;i<6;++i) {_mm_setcsr(mx&~(1u<<(7+i)));reject(&f,direct,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);_mm_setcsr(mx);putcw((unsigned short)(c&~(1u<<i)));reject(&f,direct,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);putcw(c);}
  for(unsigned i=1;i<4;++i) {putcw((unsigned short)(c|(i<<10)));reject(&f,direct,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);putcw(c);}
 }
 for(unsigned flags=0;flags<64;++flags) {
  CHECK(feclearexcept(FE_ALL_EXCEPT)==0);CHECK(feraiseexcept(FE_DIVBYZERO)==0);_mm_setcsr((mx&~63u)|flags);
  for(int mode=0;mode<4;++mode) {setup(&f,1);if(mode==1) f.call.struct_size=0;if(mode==2) f.b[0].f[255]=-FLT_MAX;if(mode==3) f.b[4].i[2]=f.b[4].i[3]=-1;
   unsigned m=_mm_getcsr();unsigned short st=sw(),ct=cw();int rc=provider->validate_call(&f.call,&e);CHECK((rc==0)==(mode==0||mode==2));CHECK(_mm_getcsr()==m);CHECK(sw()==st);CHECK(cw()==ct);
  }
 }
 CHECK(fesetenv(&saved)==0);
 /* Trap-enabled signaling payload must be rejected before any classification. */
 for(int x87=0;x87<2;++x87) {pid_t pid=fork();CHECK(pid>=0);if(pid==0) {setup(&f,1);uint32_t sn=0x7f800001;memcpy(f.b[0].f,&sn,4);feclearexcept(FE_ALL_EXCEPT);if(x87) putcw((unsigned short)(cw()&~1u));else _mm_setcsr(_mm_getcsr()&~0x80u);int rc=provider->validate_call(&f.call,&e);_exit(rc==ET_KERNEL_ERROR_UNSUPPORTED?0:2);}int status;CHECK(waitpid(pid,&status,0)==pid);CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);}
 setup(&f,1);unsigned before=_mm_getcsr()&~63u;unsigned short cb=cw();CHECK(provider->validate_call(&f.call,&e)==0);provider->invoke_call(&f.call);CHECK((_mm_getcsr()&~63u)==before);CHECK(cw()==cb);
 CHECK(fesetenv(&saved)==0);
}
typedef struct { et_f32_tensor *f;et_i64_tensor *i;et_f32_tensor_borrow *fb;et_i64_tensor_borrow *ib;const et_kernel_tensor_view_v1 *v; } owner;
static void acquire(owner *o,const et_kernel_tensor_view_v1 *v) {
 memset(o,0,sizeof(*o));et_f32_tensor_error fe;et_i64_tensor_error ie;
 if(strcmp(v->dtype,"f32")==0) {uint32_t bits[256];memcpy(bits,v->data,v->byte_length);CHECK(et_f32_tensor_create_v1(v->rank,v->shape,&o->f,&fe)==0);CHECK(et_f32_tensor_copy_bits_from_v1(o->f,bits,v->byte_length/4,&fe)==0);CHECK(et_f32_tensor_borrow_begin_v1(o->f,&o->fb,&fe)==0);CHECK(et_f32_tensor_borrow_view_v1(o->fb,&o->v,&fe)==0);et_f32_tensor *copy=o->f;CHECK(et_f32_tensor_destroy_v1(&copy,&fe)!=0);CHECK(copy==o->f);}
 else {CHECK(et_i64_tensor_create_v1(v->rank,v->shape,&o->i,&ie)==0);CHECK(et_i64_tensor_copy_from_v1(o->i,v->data,v->byte_length/8,&ie)==0);CHECK(et_i64_tensor_borrow_begin_v1(o->i,&o->ib,&ie)==0);CHECK(et_i64_tensor_borrow_view_v1(o->ib,&o->v,&ie)==0);et_i64_tensor *copy=o->i;CHECK(et_i64_tensor_destroy_v1(&copy,&ie)!=0);CHECK(copy==o->i);}
}
static void release(owner *o) {et_f32_tensor_error fe;et_i64_tensor_error ie;if(o->f) {CHECK(et_f32_tensor_borrow_end_v1(&o->fb,&fe)==0);CHECK(et_f32_tensor_destroy_v1(&o->f,&fe)==0);}else {CHECK(et_i64_tensor_borrow_end_v1(&o->ib,&ie)==0);CHECK(et_i64_tensor_destroy_v1(&o->i,&ie)==0);}CHECK(!o->f&&!o->i&&!o->fb&&!o->ib);}
static void carrier(void) {
 size_t baseline=live_allocations,retired_controls=0;fixture f;et_kernel_error e;
 for(int cat=0;cat<2;++cat) for(size_t repetition=0;repetition<16;++repetition) {
  setup(&f,cat);owner in[5],out[2];et_kernel_tensor_view_v1 iv[5],ov[2];
  for(size_t i=0;i<f.call.input_count;++i) {acquire(in+i,f.in+i);iv[i]=*in[i].v;}
  for(size_t i=0;i<2;++i) {acquire(out+i,f.out+i);ov[i]=*out[i].v;}
  et_kernel_call_v1 call=f.call;call.inputs=iv;call.outputs=ov;
  allocation_disabled=1;CHECK(et_kernel_runtime_dispatch(runtime,&call,&e)==0);CHECK(blocked_allocations==0);allocation_disabled=0;
  CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&e)==0);
  for(size_t i=0;i<f.call.input_count;++i) CHECK(memcmp(iv[i].data,f.in[i].data,iv[i].byte_length)==0);
  for(size_t i=0;i<2;++i) CHECK(memcmp(ov[i].data,f.out[i].data,ov[i].byte_length)==0);
  f.row[1]=257;allocation_disabled=1;CHECK(et_kernel_runtime_dispatch(runtime,&call,&e)!=0);CHECK(blocked_allocations==0);allocation_disabled=0;
  for(size_t i=0;i<2;++i) {CHECK(memcmp(ov[i].data,f.out[i].data,ov[i].byte_length)==0);release(out+i);}
  for(size_t i=0;i<f.call.input_count;++i) release(in+i);
  /* Accepted I2 keeps the tensor and borrow control tombstones; its payload and
   * shape/stride allocations are released. I1 retains no controls. */
  retired_controls+=2u*(cat?3u:1u);CHECK(live_allocations==baseline+retired_controls);
 }
}
int main(void) { provider=et_g3s_kernel_provider_v1();et_kernel_error e;CHECK(et_kernel_runtime_discover(resolve,NULL,&runtime,&e)==0);CHECK(feraiseexcept(FE_ALL_EXCEPT)==0);_mm_setcsr(_mm_getcsr()|63u);schema();prefixes();adjacent_storage();numerical_rejects();fenv_cases();fenv_failures();carrier();et_kernel_runtime_destroy(runtime);printf("G3S adversarial/carrier/fenv PASS: %zu checks; disabled allocation attempts=%zu\n",checks,blocked_allocations);return 0; }
