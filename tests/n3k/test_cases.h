#ifndef N3K_TEST_CASES_H
#define N3K_TEST_CASES_H
#include "test_common.h"
typedef union storage { float f[2048]; int64_t i[1024]; unsigned char b[8192]; } storage;
typedef struct fixture {
  et_kernel_request_v1 r;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 in[3],out[3];
  uint64_t row[4],ish[3][4],osh[3][4];
  storage ib[3],ob[3];
} fixture;
static void view(fixture *f,int output,size_t i,const char *dtype,size_t rank,
                 uint64_t a,uint64_t b,uint64_t c,uint64_t d) {
  uint64_t *shape=output ? f->osh[i]:f->ish[i];
  shape[0]=a;shape[1]=b;shape[2]=c;shape[3]=d;
  size_t elements=1;
  for (size_t j=0;j<rank;++j) elements*=(size_t)shape[j];
  CHECK(elements<=1024);
  storage *data=output ? &f->ob[i]:&f->ib[i];
  const int integer=strcmp(dtype,"i64")==0;
  if (!output) {
    for(size_t j=0;j<elements;++j) {
      if (integer) data->i[j]=0;
      else data->f[j]=0.125f;
    }
  }
  et_kernel_tensor_view_v1 v=tv(data,elements*(integer ? 8u:4u),dtype,rank,shape);
  if(output) f->out[i]=v; else f->in[i]=v;
}
static void setup(fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  memset(f,0,sizeof(*f));memset(f->ob,0xa5,sizeof(f->ob));
  const et_kernel_shape_range_v1 *range=&cap->shape_ranges[row];
  for(size_t i=0;i<range->rank;++i) f->row[i]=range->dimensions[i].minimum;
  const uint64_t a=f->row[0],b=f->row[1],c=f->row[2],d=f->row[3];
  const char *name=cap->operations[op];
  const int backward=strstr(name,"backward")!=NULL;
  size_t ni=1,no=1;
  if(strstr(name,"embedding")) {
    ni=2;view(f,0,0,"i64",2,a,b,0,0);
    if(!backward) {view(f,0,1,"f32",2,c,d,0,0);view(f,1,0,"f32",3,a,b,d,0);}
    else {view(f,0,1,"f32",3,a,b,d,0);view(f,1,0,"f32",2,c,d,0,0);}
  } else if(strstr(name,"linear")) {
    ni=backward ? 3u:2u;no=backward ? 2u:1u;
    view(f,0,0,"f32",3,a,b,c,0);view(f,0,1,"f32",2,d,c,0,0);
    if(backward) {view(f,0,2,"f32",3,a,b,d,0);view(f,1,0,"f32",3,a,b,c,0);view(f,1,1,"f32",2,d,c,0,0);}
    else view(f,1,0,"f32",3,a,b,d,0);
  } else if(strstr(name,"matrix-init")) {
    no=2;view(f,0,0,"i64",1,4,0,0,0);f->ib[0].i[0]=1;
    view(f,1,0,"f32",2,a,b,0,0);view(f,1,1,"i64",1,4,0,0,0);
  } else if(strstr(name,"heads")) {
    const int split=(strstr(name,"split")!=NULL)!=backward;
    view(f,0,0,"f32",split ? 3u:4u,a,split ? b:c,split ? c*d:b,d);
    view(f,1,0,"f32",split ? 4u:3u,a,split ? c:b,split ? b:c*d,d);
  } else {
    if(strstr(name,"gelu")) ni=backward ? 2u:1u;
    else {
      const size_t edges=strstr(name,"sum3") ? 3u:2u;
      ni=backward ? 1u:edges;no=backward ? edges:1u;
    }
    for(size_t i=0;i<ni;++i) view(f,0,i,"f32",range->rank,a,b,c,d);
    for(size_t i=0;i<no;++i) view(f,1,i,"f32",range->rank,a,b,c,d);
  }
  f->r=rq(name,range->rank,f->row);f->call=kc(cap->name,&f->r,f->in,ni,f->out,no);
}
#endif
