#ifndef ET_G3C4_ADVERSARIAL_CASES_H
#define ET_G3C4_ADVERSARIAL_CASES_H
#include "test_common.h"
typedef union storage { float f[2048]; int64_t i[1024]; unsigned char b[8192]; } storage;
typedef struct fixture {
  et_kernel_request_v1 r;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 in[6],out[1];
  uint64_t row[6],ish[6][6],osh[1][6];
  storage ib[6],ob[1];
} fixture;
static inline void setup(fixture *f,const et_kernel_capability_v1 *cap,size_t op,size_t row) {
  g3c4_fixture src;size_t found=G3C4_ROWS;
  for(size_t i=0;i<G3C4_ROWS;++i) {
    fixture_init(&src,i);
    if(strcmp(src.call.capability,cap->name)!=0 || strcmp(src.request.operation,cap->operations[op])!=0 || src.request.rank!=cap->shape_ranges[row].rank) continue;
    size_t d=0;for(;d<src.request.rank;++d) if(src.shape[d]!=cap->shape_ranges[row].dimensions[d].minimum) break;
    if(d==src.request.rank) {found=i;break;}
  }
  CHECK(found<G3C4_ROWS);fixture_init(&src,found);memset(f,0,sizeof(*f));memset(f->ob,0xa5,sizeof(f->ob));
  memcpy(f->row,src.shape,sizeof(f->row));f->r=src.request;f->r.shape=f->row;
  for(size_t i=0;i<src.call.input_count;++i) {
    memcpy(f->ish[i],src.input_shapes[i],sizeof(f->ish[i]));f->in[i]=src.inputs[i];
    memcpy(&f->ib[i],src.inputs[i].data,src.inputs[i].byte_length);f->in[i].data=&f->ib[i];f->in[i].shape=f->ish[i];
  }
  memcpy(f->osh[0],src.output_shape,sizeof(f->osh[0]));f->out[0]=src.outputs[0];
  f->out[0].data=&f->ob[0];f->out[0].shape=f->osh[0];
  f->call=kc(cap->name,&f->r,f->in,src.call.input_count,f->out,1);
}
#endif
