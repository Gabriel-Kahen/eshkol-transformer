#include "test_cases.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <xmmintrin.h>

static int capture_failure,restore_failure;
int __real_fegetenv(fenv_t *);int __real_fesetenv(const fenv_t *);
int __wrap_fegetenv(fenv_t *e){return capture_failure ? -1:__real_fegetenv(e);}
int __wrap_fesetenv(const fenv_t *e){int rc=__real_fesetenv(e);return restore_failure ? -1:rc;}
static unsigned short cw_read(void){unsigned short x;__asm__ volatile("fnstcw %0":"=m"(x));return x;}
static unsigned short sw_read(void){unsigned short x;__asm__ volatile("fnstsw %0":"=am"(x));return x;}

static void rejected(et_kernel_runtime *rt,fixture *f,int direct,uint32_t category,uint32_t code){
  fixture before;memcpy(&before,f,sizeof(before));const unsigned short cw=cw_read(),sw=sw_read();const unsigned mx=_mm_getcsr();et_kernel_error e;
  int32_t rc=direct ? et_g3c4_kernel_provider_v1()->validate_call(&f->call,&e):et_kernel_runtime_dispatch(rt,&f->call,&e);
  if(rc!=(int32_t)category||e.category!=category||e.code!=code)fprintf(stderr,"%s %s got %d/%u/%u expected %u/%u: %s\n",direct?"direct":"K1",f->r.operation,rc,e.category,e.code,category,code,e.message);
  CHECK(rc==(int32_t)category);CHECK(e.category==category);CHECK(e.code==code);
  if(direct)CHECK(strcmp(e.operation,(f->call.struct_size<88u||f->r.struct_size<56u)?"g3c4.provider":f->r.operation)==0);
  CHECK(memcmp(&before,f,sizeof(before))==0);CHECK(cw_read()==cw);CHECK(sw_read()==sw);CHECK(_mm_getcsr()==mx);
}
#define BOTH(cat,code) do{rejected(rt,f,0,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##code);rejected(rt,f,1,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##code);}while(0)
#define DIFFER(cat,kcode,dcode) do{rejected(rt,f,0,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##kcode);rejected(rt,f,1,ET_KERNEL_ERROR_##cat,ET_KERNEL_CODE_##dcode);}while(0)

static void descriptors(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  setup(f,cap,op,row);const size_t ni=f->call.input_count;
  for(size_t side=0;side<2u;++side)for(size_t i=0;i<(side?1u:ni);++i){
    setup(f,cap,op,row);et_kernel_tensor_view_v1 *v=side?&f->out[i]:&f->in[i];const et_kernel_tensor_view_v1 original=*v;
    v->struct_size=71u;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
    v->struct_size=73u;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*v=original;
    v->dtype="i32";BOTH(DTYPE_MISMATCH,INVALID_TEXT);*v=original;
    v->dtype=NULL;BOTH(DTYPE_MISMATCH,INVALID_TEXT);*v=original;
    v->device=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_INVALID_TEXT);*v=original;
    v->device="cuda:0";DIFFER(DEVICE_MISMATCH,INVALID_BUFFER,INVALID_TEXT);*v=original;
    v->data=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);rejected(rt,f,1,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
    v->layout=0u;BOTH(NONCONTIGUOUS,INVALID_BUFFER);*v=original;v->offset_bytes=4u;BOTH(NONCONTIGUOUS,INVALID_BUFFER);*v=original;
    --v->byte_length;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);rejected(rt,f,1,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
    if(strcmp(v->dtype,"bool")!=0){v->data=(unsigned char *)v->data+1;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*v=original;}
    v->data=(void *)(UINTPTR_MAX-(strcmp(v->dtype,"i64")==0?7u:3u));if(original.byte_length>4u)DIFFER(INVALID_ARGUMENT,ALIASING_OUTPUT,PROVIDER_REJECTED);*v=original;
    if(v->rank){
      uint64_t *shape=side?f->osh[i]:f->ish[i];const uint64_t old=shape[0];
      v->shape=NULL;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
      v->shape=(const uint64_t *)((const unsigned char *)original.shape+1);BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
      v->shape=(const uint64_t *)(UINTPTR_MAX-7u);BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*v=original;
      shape[0]=old+1u;size_t elements=1u;for(size_t d=0;d<v->rank;++d)elements*=(size_t)shape[d];v->byte_length=elements*(strcmp(v->dtype,"i64")==0?8u:strcmp(v->dtype,"bool")==0?1u:4u);BOTH(SHAPE_MISMATCH,INVALID_SHAPE);shape[0]=old;*v=original;
      shape[0]=UINT64_MAX;rejected(rt,f,0,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INTEGER_OVERFLOW);shape[0]=old;*v=original;
      uint64_t flat=original.byte_length/(strcmp(original.dtype,"i64")==0?8u:strcmp(original.dtype,"bool")==0?1u:4u);v->rank=original.rank==1u?0u:1u;v->shape=&flat;if(v->rank==0u)v->byte_length=strcmp(original.dtype,"i64")==0?8u:strcmp(original.dtype,"bool")==0?1u:4u;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;
      shape[0]=0u;v->byte_length=0u;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);shape[0]=old;*v=original;
    }else{uint64_t one=1u;v->rank=1u;v->shape=&one;BOTH(SHAPE_MISMATCH,INVALID_SHAPE);*v=original;}
    if(!side&&strcmp(v->dtype,"f32")==0){const uint32_t bad[]={0x7fc00001u,0x7f800000u,0xff800000u,0x7f800001u};for(size_t p=0;p<COUNT(bad);++p){uint32_t saved;memcpy(&saved,v->data,4u);memcpy(v->data,&bad[p],4u);BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);memcpy(v->data,&saved,4u);}}
  }
  for(size_t side=0;side<2u;++side){
    setup(f,cap,op,row);size_t *stride=side?&f->call.output_stride:&f->call.input_stride,*bytes=side?&f->call.output_bytes:&f->call.input_bytes,*count=side?&f->call.output_count:&f->call.input_count;const size_t os=*stride,ob=*bytes,oc=*count;
    *stride=71u;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);*stride=os;*stride=73u;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);*stride=os;
    --*bytes;DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);*bytes=ob;*count=SIZE_MAX;DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);*count=oc;
    *count=0u;*bytes=0u;*stride=0u;if(side)f->call.outputs=NULL;else f->call.inputs=NULL;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);
    setup(f,cap,op,row);if(side)f->call.outputs=(unsigned char *)f->out+1;else f->call.inputs=(unsigned char *)f->in+1;BOTH(INVALID_ARGUMENT,INVALID_BUFFER);
    setup(f,cap,op,row);if(side)f->call.outputs=NULL;else f->call.inputs=NULL;BOTH(INVALID_ARGUMENT,NULL_ARGUMENT);
    setup(f,cap,op,row);if(side)f->call.outputs=(void *)(UINTPTR_MAX-7u);else f->call.inputs=(void *)(UINTPTR_MAX-7u);DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);
    setup(f,cap,op,row);if(side){f->call.output_stride=SIZE_MAX-7u;f->call.output_bytes=SIZE_MAX-7u;}else{f->call.input_stride=SIZE_MAX-7u;f->call.input_bytes=SIZE_MAX-7u;}DIFFER(INVALID_ARGUMENT,INTEGER_OVERFLOW,INVALID_BUFFER);
  }
  setup(f,cap,op,row);f->call.struct_size=87u;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(f,cap,op,row);f->r.struct_size=55u;BOTH(VERSION_MISMATCH,INVALID_STRUCT_SIZE);
  setup(f,cap,op,row);f->r.operation="g3c4.absent";DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  setup(f,cap,op,row);f->r.dtype="f64";DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
}

static void requests(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  setup(f,cap,op,row);const size_t rank=f->r.rank;
  for(size_t d=0;d<rank;++d)for(int delta=-1;delta<=1;delta+=2){
    setup(f,cap,op,row);f->row[d]=(uint64_t)((int64_t)f->row[d]+delta);int advertised=0;
    for(size_t ri=0;ri<cap->shape_range_count;++ri){const et_kernel_shape_range_v1 *candidate=&cap->shape_ranges[ri];if(candidate->rank!=rank)continue;size_t k=0;for(;k<rank;++k)if(candidate->dimensions[k].minimum!=f->row[k])break;if(k==rank){advertised=1;break;}}
    et_kernel_error e;const et_kernel_capability_v1 *found=NULL;int32_t rc=et_kernel_runtime_capability_require(rt,cap->name,&f->r,&found,&e);if(advertised){CHECK(rc==0);CHECK(found!=NULL);CHECK(strcmp(found->name,cap->name)==0);}else{CHECK(rc==ET_KERNEL_ERROR_UNSUPPORTED);CHECK(e.code==ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);CHECK(found==NULL);}
  }
  for(size_t d=0;d<rank;++d){setup(f,cap,op,row);f->row[d]=0u;DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);}
  setup(f,cap,op,row);f->r.rank=rank-1u;DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  setup(f,cap,op,row);f->r.deterministic=2u;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT);
  setup(f,cap,op,row);f->r.shape=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT);
  setup(f,cap,op,row);f->r.shape=(const uint64_t *)((const unsigned char *)f->row+1);rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);
  setup(f,cap,op,row);f->r.shape=(const uint64_t *)(UINTPTR_MAX-7u);rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_BUFFER);
  setup(f,cap,op,row);f->call.request=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_NULL_ARGUMENT);
  setup(f,cap,op,row);f->call.capability=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_NULL_ARGUMENT);
  setup(f,cap,op,row);f->r.operation=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT);
  setup(f,cap,op,row);f->r.dtype=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT);
  setup(f,cap,op,row);f->r.device=NULL;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_INVALID_TEXT);
  setup(f,cap,op,row);f->call.capability="g3c4.absent";DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);
  static const char *const absent[]={"g3c4.embedding.backward","g3c4.matrix-init.forward","g3c4.cache.append","g3c4.sample.forward"};
  for(size_t i=0;i<COUNT(absent);++i){setup(f,cap,op,row);f->r.operation=absent[i];DIFFER(UNSUPPORTED,CAPABILITY_NOT_VERIFIED,PROVIDER_REJECTED);}
}

static void aliases(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  setup(f,cap,op,row);const size_t ni=f->call.input_count;
  for(size_t i=0;i<ni;++i)for(size_t j=i+1u;j<ni;++j){
    setup(f,cap,op,row);const int allowed=strstr(f->r.operation,"residual")!=NULL||(strstr(f->r.operation,"attention")!=NULL&&i==3u&&j==4u&&f->in[i].byte_length==f->in[j].byte_length);
    f->in[j].data=f->in[i].data;if(allowed)run(rt,&f->call);else BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);
    setup(f,cap,op,row);const size_t align=strcmp(f->in[i].dtype,"i64")==0||strcmp(f->in[j].dtype,"i64")==0?8u:strcmp(f->in[i].dtype,"f32")==0||strcmp(f->in[j].dtype,"f32")==0?4u:1u;
    if(f->in[i].byte_length>align){f->in[j].data=(unsigned char *)f->in[i].data+align;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  }
  for(size_t i=0;i<ni;++i){setup(f,cap,op,row);f->out[0].data=f->in[i].data;rejected(rt,f,0,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);}
}

static void domain(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  setup(f,cap,op,row);const char *name=f->r.operation;
  if(strstr(name,"embedding")){const int64_t bad[]={-1,(int64_t)f->row[2],INT64_MIN,INT64_MAX};for(size_t j=0;j<f->in[0].byte_length/8u;++j)for(size_t i=0;i<COUNT(bad);++i){setup(f,cap,op,row);f->ib[0].i[j]=bad[i];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}}
  else if(strstr(name,"gelu")){setup(f,cap,op,row);f->ib[0].f[f->in[0].byte_length/4u-1u]=NAN;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  else if(strstr(name,"residual")){f->ib[0].f[15]=FLT_MAX;f->ib[1].f[15]=FLT_MAX;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  else if(strstr(name,"layer-norm")){const uint32_t eps[]={0u,0x80000000u,0xbf800000u,0x7fc00000u,0x7f800000u};for(size_t i=0;i<COUNT(eps);++i){setup(f,cap,op,row);memcpy(&f->ib[3].f[0],&eps[i],4u);BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}setup(f,cap,op,row);f->ib[0].f[0]=f->ib[0].f[1]=FLT_MAX;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  else if(strstr(name,"linear")){const size_t di=(size_t)f->row[2];setup(f,cap,op,row);memset(f->ib[1].f,0,f->in[1].byte_length);f->ib[0].f[di-2u]=f->ib[0].f[di-1u]=FLT_MAX;f->ib[1].f[di-2u]=f->ib[1].f[di-1u]=1.0f;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}
  else if(strstr(name,"attention")){const size_t tq=(size_t)f->row[3],tk=(size_t)f->row[4];const size_t lengths[]={tq,tk};for(size_t pos=3;pos<5;++pos)for(size_t j=0;j<lengths[pos-3u];++j){setup(f,cap,op,row);f->ib[pos].i[j]=-1;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);setup(f,cap,op,row);f->ib[pos].i[j]=16777216;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}for(size_t q=0;q<tq;++q)for(size_t k=0;k<tk;++k){setup(f,cap,op,row);f->ib[5].b[q*tk+k]=2u;BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}for(size_t j=1;j<tq;++j){setup(f,cap,op,row);f->ib[3].i[j]=f->ib[3].i[j-1u];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}for(size_t j=1;j<tk;++j){setup(f,cap,op,row);f->ib[4].i[j]=f->ib[4].i[j-1u];BOTH(INVALID_ARGUMENT,PROVIDER_REJECTED);}}
}

static void controls(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  fenv_t saved;CHECK(fegetenv(&saved)==0);const unsigned short cw=cw_read();const unsigned mx=_mm_getcsr();setup(f,cap,op,row);
  const unsigned changes[]={0x40u,0x8000u,0x2000u,0x4000u,0x6000u};for(size_t i=0;i<COUNT(changes);++i){_mm_setcsr(mx|changes[i]);BOTH(UNSUPPORTED,PROVIDER_REJECTED);_mm_setcsr(mx);}
  for(size_t i=0;i<6u;++i){_mm_setcsr(mx&~(1u<<(7u+i)));BOTH(UNSUPPORTED,PROVIDER_REJECTED);_mm_setcsr(mx);unsigned short bad=(unsigned short)(cw&~(1u<<i));__asm__ volatile("fldcw %0"::"m"(bad));BOTH(UNSUPPORTED,PROVIDER_REJECTED);__asm__ volatile("fldcw %0"::"m"(cw));}
  for(unsigned mode=0x400u;mode<=0xc00u;mode+=0x400u){unsigned short bad=(unsigned short)(cw|mode);__asm__ volatile("fldcw %0"::"m"(bad));BOTH(UNSUPPORTED,PROVIDER_REJECTED);__asm__ volatile("fldcw %0"::"m"(cw));}
  setup(f,cap,op,row);for(size_t i=0;i<f->call.input_count;++i)if(strcmp(f->in[i].dtype,"f32")==0){uint32_t snan=0x7f800001u;memcpy(f->in[i].data,&snan,4u);break;}_mm_setcsr(mx|0x40u);BOTH(UNSUPPORTED,PROVIDER_REJECTED);_mm_setcsr(mx);
  setup(f,cap,op,row);
  CHECK(feclearexcept(FE_ALL_EXCEPT)==0);CHECK(feraiseexcept(FE_DIVBYZERO)==0);_mm_setcsr((_mm_getcsr()&~0x3fu)|0x20u);const unsigned short sw=sw_read();const unsigned flags=_mm_getcsr();fixture before;memcpy(&before,f,sizeof(before));et_kernel_error e;CHECK(et_g3c4_kernel_provider_v1()->validate_call(&f->call,&e)==0);CHECK(sw_read()==sw);CHECK(_mm_getcsr()==flags);CHECK(memcmp(&before,f,sizeof(before))==0);CHECK(fesetenv(&saved)==0);CHECK(cw_read()==cw);
  setup(f,cap,op,row);capture_failure=1;BOTH(UNSUPPORTED,PROVIDER_REJECTED);capture_failure=0;setup(f,cap,op,row);restore_failure=1;BOTH(INTERNAL,PROVIDER_REJECTED);restore_failure=0;
}

static void future_tails(et_kernel_runtime *rt,fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row){
  struct ev{et_kernel_tensor_view_v1 v;uint64_t tail;}in[6],out;struct er{et_kernel_request_v1 r;uint64_t tail;}request;struct ec{et_kernel_call_v1 c;uint64_t tail;}call;
  setup(f,cap,op,row);memset(in,0xa5,sizeof(in));memset(&out,0xa5,sizeof(out));memset(&request,0xa5,sizeof(request));memset(&call,0xa5,sizeof(call));
  for(size_t i=0;i<f->call.input_count;++i){in[i].v=f->in[i];in[i].v.struct_size=sizeof(in[i]);}out.v=f->out[0];out.v.struct_size=sizeof(out);request.r=f->r;request.r.struct_size=sizeof(request);memset(request.r.reserved,0xa5,sizeof(request.r.reserved));call.c=kc(cap->name,&request.r,&in[0].v,f->call.input_count,&out.v,1u);call.c.struct_size=sizeof(call);call.c.input_stride=sizeof(in[0]);call.c.input_bytes=call.c.input_count*sizeof(in[0]);call.c.output_stride=sizeof(out);call.c.output_bytes=sizeof(out);run(rt,&call.c);CHECK(call.tail==UINT64_C(0xa5a5a5a5a5a5a5a5));CHECK(request.tail==UINT64_C(0xa5a5a5a5a5a5a5a5));CHECK(out.tail==UINT64_C(0xa5a5a5a5a5a5a5a5));
}

int main(void){
  et_kernel_runtime *rt=runtime_new();fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);const et_kernel_provider_v1 *p=et_g3c4_kernel_provider_v1();const et_kernel_capability_v1 *caps=p->capabilities;size_t rows=0;
  for(size_t ci=0;ci<p->capability_count;++ci)for(size_t oi=0;oi<caps[ci].operation_count;++oi)for(size_t ri=0;ri<caps[ci].shape_range_count;++ri){descriptors(rt,f,&caps[ci],oi,ri);requests(rt,f,&caps[ci],oi,ri);aliases(rt,f,&caps[ci],oi,ri);domain(rt,f,&caps[ci],oi,ri);controls(rt,f,&caps[ci],oi,ri);future_tails(rt,f,&caps[ci],oi,ri);++rows;}
  CHECK(rows==28u);et_kernel_error e;CHECK(p->validate_call(NULL,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_NULL_ARGUMENT);free(f);et_kernel_runtime_destroy(rt);printf("G3-C4-N adversarial: %zu checks\n",checks);return 0;
}
