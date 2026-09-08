#include "test_common.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <xmmintrin.h>

#include "test_cases.h"
static void unchanged_reject(et_kernel_runtime *rt,fixture *f,uint32_t category,uint32_t code) {
  storage in[3],out[3];memcpy(in,f->ib,sizeof(in));memcpy(out,f->ob,sizeof(out));
  et_kernel_error e;CHECK(et_kernel_runtime_dispatch(rt,&f->call,&e)!=0);
  if(e.category!=category || (code!=UINT32_MAX && e.code!=code))
    fprintf(stderr,"negative mismatch %s category=%u code=%u expected=%u/%u: %s\n",f->r.operation,e.category,e.code,category,code,e.message);
  CHECK(e.category==category);if(code!=UINT32_MAX) CHECK(e.code==code);
  CHECK(memcmp(in,f->ib,sizeof(in))==0);CHECK(memcmp(out,f->ob,sizeof(out))==0);
}
static void all_rows(et_kernel_runtime *rt) {
  const et_kernel_provider_v1 *p=et_n3k_kernel_provider_v1();
  const et_kernel_capability_v1 *caps=p->capabilities;
  fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);
  size_t cases=0;
  for(size_t ci=0;ci<p->capability_count;++ci) for(size_t oi=0;oi<caps[ci].operation_count;++oi)
    for(size_t ri=0;ri<caps[ci].shape_range_count;++ri) {
      setup(f,&caps[ci],oi,ri);run(rt,&f->call);++cases;
      const size_t ni=f->call.input_count,no=f->call.output_count;
      for(size_t side=0;side<2;++side) for(size_t index=0;index<(side ? no:ni);++index) {
        setup(f,&caps[ci],oi,ri);
        et_kernel_tensor_view_v1 *v=side ? &f->out[index]:&f->in[index];
        const et_kernel_tensor_view_v1 original=*v;
        v->rank=0;v->byte_length=strcmp(v->dtype,"f32")==0 ? 4u:8u;unchanged_reject(rt,f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE);*v=original;
        v->dtype=strcmp(v->dtype,"f32")==0 ? "i32":"f32";
        /* K1 sees same-sized i32 before provider dtype validation; i64->f32
         * changes byte-span first, as required by generic precedence. */
        unchanged_reject(rt,f,strcmp(original.dtype,"f32")==0 ? ET_KERNEL_ERROR_DTYPE_MISMATCH:ET_KERNEL_ERROR_SHAPE_MISMATCH,UINT32_MAX);*v=original;
        v->device="cuda:0";unchanged_reject(rt,f,ET_KERNEL_ERROR_DEVICE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
        v->offset_bytes=4;unchanged_reject(rt,f,ET_KERNEL_ERROR_NONCONTIGUOUS,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
        v->byte_length--;unchanged_reject(rt,f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);*v=original;
        v->data=(unsigned char *)original.data+1;unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);*v=original;
        v->struct_size=sizeof(*v)-1;unchanged_reject(rt,f,ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_INVALID_STRUCT_SIZE);*v=original;
        uint64_t *shape=side ? f->osh[index]:f->ish[index];
        const uint64_t old=shape[0];shape[0]=UINT64_MAX;
        unchanged_reject(rt,f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INTEGER_OVERFLOW);shape[0]=old;
        v->data=(void *)(UINTPTR_MAX-7u);unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);*v=original;
        if(!side && strcmp(v->dtype,"f32")==0) {
          const uint32_t patterns[]={0x7fc00001u,0x7f800000u,0xff800000u,0x7f800001u};
          for(size_t pat=0;pat<COUNT(patterns);++pat) for(size_t end=0;end<2;++end) {
            const size_t j=end ? v->byte_length/4u-1u:0;
            const float oldf=f->ib[index].f[j];memcpy(&f->ib[index].f[j],&patterns[pat],4);
            unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
            f->ib[index].f[j]=oldf;
          }
        }
      }
      setup(f,&caps[ci],oi,ri);
      if(strstr(f->r.operation,"embedding")) {
        const int64_t bad[]={-1,(int64_t)f->row[2],INT64_MIN,INT64_MAX};
        for(size_t bi=0;bi<COUNT(bad);++bi) for(size_t token=0;token<2;++token) {
          f->ib[0].i[token]=bad[bi];
          unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
          f->ib[0].i[token]=0;
        }
        if(strstr(f->r.operation,"backward")) {
          f->ib[1].f[3]=FLT_MAX;f->ib[1].f[7]=FLT_MAX;
          unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
          f->ib[1].f[3]=f->ib[1].f[7]=0.125f;
        }
      }
      const int ordinary=strstr(f->r.operation,"embedding") ||
        strstr(f->r.operation,"linear") || strstr(f->r.operation,"gelu");
      for(size_t i=0;i<ni;++i) for(size_t j=i+1;j<ni;++j) {
        void *saved=f->in[j].data;
        if(ordinary) {
          f->in[j].data=f->in[i].data;
          unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
        }
        f->in[j].data=(unsigned char *)f->in[i].data+8;
        unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
        f->in[j].data=saved;
      }
      for(size_t o=0;o<no;++o) for(size_t i=0;i<ni;++i) {
        void *saved=f->out[o].data;f->out[o].data=f->in[i].data;
        unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);f->out[o].data=saved;
      }
      if(no>1) {f->out[1].data=f->out[0].data;unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);}
      setup(f,&caps[ci],oi,ri);f->call.input_stride=sizeof(f->in[0])-1;
      unchanged_reject(rt,f,ET_KERNEL_ERROR_VERSION_MISMATCH,ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      setup(f,&caps[ci],oi,ri);f->call.input_bytes=SIZE_MAX;
      unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
      setup(f,&caps[ci],oi,ri);f->call.input_count=SIZE_MAX;
      unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,UINT32_MAX);
    }
  CHECK(cases==31u);
  free(f);
}
static unsigned short control_word(void) {unsigned short x;__asm__ volatile("fnstcw %0":"=m"(x));return x;}
static unsigned short status_word(void) {unsigned short x;__asm__ volatile("fnstsw %0":"=am"(x));return x;}
static void controls_and_flags(et_kernel_runtime *rt) {
  const et_kernel_provider_v1 *p=et_n3k_kernel_provider_v1();
  const et_kernel_capability_v1 *cap=et_kernel_runtime_capability_find(rt,"n3k.gelu");
  fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);
  fenv_t saved;CHECK(fegetenv(&saved)==0);
  setup(f,cap,1,0);
  const unsigned short original_cw=control_word();const unsigned original_mx=_mm_getcsr();
  const unsigned mx_changes[]={0x0040u,0x8000u,0x2000u,0x4000u,0x6000u};
  for(size_t i=0;i<COUNT(mx_changes);++i) {
    _mm_setcsr(original_mx|mx_changes[i]);
    unchanged_reject(rt,f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(_mm_getcsr()==(original_mx|mx_changes[i]));_mm_setcsr(original_mx);
  }
  for(size_t i=0;i<6;++i) {
    const unsigned mx=original_mx&~(1u<<(7u+i));_mm_setcsr(mx);
    unchanged_reject(rt,f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(_mm_getcsr()==mx);_mm_setcsr(original_mx);
    unsigned short cw=(unsigned short)(original_cw&~(1u<<i));
    __asm__ volatile("fldcw %0"::"m"(cw));
    unchanged_reject(rt,f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(control_word()==cw);__asm__ volatile("fldcw %0"::"m"(original_cw));
  }
  for(unsigned bits=0x400;bits<=0xc00;bits+=0x400) {
    const unsigned short cw=(unsigned short)(original_cw|bits);__asm__ volatile("fldcw %0"::"m"(cw));
    unchanged_reject(rt,f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(control_word()==cw);__asm__ volatile("fldcw %0"::"m"(original_cw));
  }
  for(size_t fail=0;fail<2;++fail) {
    setup(f,cap,1,0);if(fail) {uint32_t sn=0x7f800001;memcpy(&f->ib[0].f[15],&sn,4);}
    CHECK(feclearexcept(FE_ALL_EXCEPT)==0);CHECK(feraiseexcept(FE_DIVBYZERO)==0);
    _mm_setcsr((_mm_getcsr()&~0x3fu)|0x20u);
    const unsigned short sw=status_word(),cw=control_word();const unsigned mx=_mm_getcsr();
    et_kernel_error e;const int32_t rc=p->validate_call(&f->call,&e);
    CHECK((rc!=0)==(fail!=0));CHECK(status_word()==sw);CHECK(control_word()==cw);CHECK(_mm_getcsr()==mx);
  }
  CHECK(fesetenv(&saved)==0);free(f);
}
static void prefixes_and_overflow(et_kernel_runtime *rt) {
  const et_kernel_provider_v1 *p=et_n3k_kernel_provider_v1();et_kernel_error e;
  size_t *short_prefix=malloc(sizeof(size_t));CHECK(short_prefix!=NULL);*short_prefix=sizeof(size_t);
  CHECK(p->validate_call((const et_kernel_call_v1 *)short_prefix,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);
  CHECK(e.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  et_kernel_call_v1 call={0};call.struct_size=sizeof(call);call.request=(const et_kernel_request_v1 *)short_prefix;
  CHECK(p->validate_call(&call,&e)==ET_KERNEL_ERROR_VERSION_MISMATCH);free(short_prefix);
  const et_kernel_capability_v1 *cap=et_kernel_runtime_capability_find(rt,"n3k.linear");
  fixture *f=malloc(sizeof(*f));CHECK(f!=NULL);setup(f,cap,0,0);
  /* All early dX may be finite while a late dWeight product overflows. */
  f->ib[0].f[7]=FLT_MAX;f->ib[2].f[7]=2.0f;
  unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
  setup(f,cap,1,0);f->ib[0].f[7]=FLT_MAX;f->ib[1].f[15]=2.0f;
  unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
  cap=et_kernel_runtime_capability_find(rt,"n3k.gelu");setup(f,cap,0,0);f->ib[0].f[15]=FLT_MAX;
  unchanged_reject(rt,f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED);
  /* Direct address overflow retains provider category rather than K1 alias code. */
  setup(f,cap,1,0);f->in[0].data=(void *)(UINTPTR_MAX-7u);
  CHECK(p->validate_call(&f->call,&e)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(e.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  free(f);
}
int main(void) {
  et_kernel_runtime *rt=runtime_new();all_rows(rt);controls_and_flags(rt);prefixes_and_overflow(rt);
  et_kernel_runtime_destroy(rt);printf("N3K negative atomicity: %zu checks\n",checks);return 0;
}
