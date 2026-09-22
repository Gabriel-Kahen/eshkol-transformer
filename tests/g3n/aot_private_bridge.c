#include "eshkol_transformer/g3n_primitives_abi.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Independent fixed-buffer development bridge, never in production closure. */
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"G3-N AOT line %d: %s\n",__LINE__,#x); abort(); } } while (0)
static et_kernel_tensor_view_v1 view(void *p, size_t bytes, const char *dtype,
                                    size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 v = {sizeof(v),p,bytes,dtype,"cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,rank,shape};
  return v;
}
static const et_kernel_provider_v1 *resolve(void *context,const char *name) {
  (void)context; REQUIRE(strcmp(name,ET_KERNEL_PROVIDER_SYMBOL_V1)==0);
  return et_g3n_kernel_provider_v1();
}
static void exercise(et_kernel_runtime *runtime, const char *cap, const char *op,
                     size_t rank, const uint64_t *shape,
                     et_kernel_tensor_view_v1 *inputs, size_t count,
                     et_kernel_tensor_view_v1 output) {
  et_kernel_request_v1 r={sizeof(r),op,"f32","cpu",rank,shape,1,{0}};
  et_kernel_call_v1 c={sizeof(c),cap,&r,count,sizeof(*inputs),count*sizeof(*inputs),inputs,1,sizeof(output),sizeof(output),&output};
  et_kernel_error e;
  REQUIRE(et_kernel_runtime_dispatch(runtime,&c,&e)==0);
  /* Adjacent request and malformed descriptor fail without touching output. */
  unsigned char before[1024]; REQUIRE(output.byte_length<=sizeof(before));
  memcpy(before,output.data,output.byte_length);
  uint64_t bad[6]; memcpy(bad,shape,rank*sizeof(*shape)); bad[rank-1]++;
  r.shape=bad;
  REQUIRE(et_kernel_runtime_dispatch(runtime,&c,&e)!=0);
  REQUIRE(e.category==ET_KERNEL_ERROR_UNSUPPORTED);
  REQUIRE(memcmp(before,output.data,output.byte_length)==0);
  r.shape=shape; c.output_count=0;
  REQUIRE(et_kernel_runtime_dispatch(runtime,&c,&e)!=0);
  REQUIRE(memcmp(before,output.data,output.byte_length)==0);
}
int64_t et_g3n_test_provider_transport_v1(void) {
  et_kernel_runtime *runtime=NULL; et_kernel_error error;
  REQUIRE(et_kernel_runtime_discover(resolve,NULL,&runtime,&error)==0);
  const uint64_t ids_shape[]={1,1}, token[]={1,1,4}, head[]={1,2,1,2};
  const uint64_t vocabularies[]={256,2};
  for(size_t v=0;v<2;v++) {
    uint64_t ws[]={vocabularies[v],4}, row[]={1,1,vocabularies[v],4};
    float weight[1024],out[4]; int64_t id=(int64_t)vocabularies[v]-1;
    for(size_t i=0;i<vocabularies[v]*4;i++) weight[i]=(float)i;
    et_kernel_tensor_view_v1 in[]={view(&id,8,"i64",2,ids_shape),view(weight,(size_t)vocabularies[v]*16,"f32",2,ws)};
    exercise(runtime,"g3n.embedding-forward","g3n.embedding.forward",4,row,in,2,view(out,16,"f32",3,token));
    REQUIRE(memcmp(out,weight+(size_t)id*4,16)==0);
  }
  const uint64_t widths[][2]={{4,4},{4,8},{8,4},{4,256}};
  for(size_t p=0;p<4;p++) {
    size_t di=(size_t)widths[p][0],dout=(size_t)widths[p][1];
    uint64_t row[]={1,1,di,dout},xs[]={1,1,di},ys[]={1,1,dout},ws[]={dout,di};
    float x[8],w[1024],y[256];
    for(size_t i=0;i<di;i++) x[i]=(float)(i+1);
    for(size_t o=0;o<dout;o++) for(size_t i=0;i<di;i++) w[o*di+i]=(float)(o+1);
    et_kernel_tensor_view_v1 in[]={view(x,di*4,"f32",3,xs),view(w,di*dout*4,"f32",2,ws)};
    exercise(runtime,"g3n.linear-forward","g3n.linear.forward-no-bias",4,row,in,2,view(y,dout*4,"f32",3,ys));
    for(size_t o=0;o<dout;o++) REQUIRE(y[o]==(float)((o+1)*di*(di+1)/2));
  }
  {
    const uint64_t parameter[]={4}; float x[]={2,2,2,2},gamma[]={1,2,3,4},beta[]={-1,0,1,2},eps=0.01f,out[4];
    et_kernel_tensor_view_v1 in[]={view(x,16,"f32",3,token),view(gamma,16,"f32",1,parameter),view(beta,16,"f32",1,parameter),view(&eps,4,"f32",0,NULL)};
    exercise(runtime,"g3n.layer-norm-forward","g3n.layer-norm.forward",3,token,in,4,view(out,16,"f32",3,token));
    REQUIRE(memcmp(out,beta,16)==0);
  }
  {
    float x[]={1,2,3,4},b[]={2,4,6,8},out[4];
    et_kernel_tensor_view_v1 in[]={view(x,16,"f32",3,token),view(b,16,"f32",3,token)};
    exercise(runtime,"g3n.residual-forward","g3n.residual.forward",3,token,in,2,view(out,16,"f32",3,token));
    for(size_t i=0;i<4;i++) REQUIRE(out[i]==3*x[i]);
  }
  {
    const uint64_t row[]={1,1,2,2}; float x[]={-0.0f,1,-2,3},split[4],merged[4];
    et_kernel_tensor_view_v1 in=view(x,16,"f32",3,token);
    exercise(runtime,"g3n.head-layout-forward","g3n.heads.split.forward",4,row,&in,1,view(split,16,"f32",4,head));
    REQUIRE(memcmp(split,x,16)==0); in=view(split,16,"f32",4,head);
    exercise(runtime,"g3n.head-layout-forward","g3n.heads.merge.forward",4,row,&in,1,view(merged,16,"f32",3,token));
    REQUIRE(memcmp(merged,x,16)==0);
  }
  {
    const uint64_t row[]={1,2,2,1,1,2},ms[]={1,1,1};
    float q[]={1,2,3,4},k[]={2,1,4,3},v[]={-2,4,6,-8},out[4];
    int64_t qp=16777215,kp=0; uint8_t keep=1;
    et_kernel_tensor_view_v1 in[]={view(q,16,"f32",4,head),view(k,16,"f32",4,head),view(v,16,"f32",4,head),view(&qp,8,"i64",2,ids_shape),view(&kp,8,"i64",2,ids_shape),view(&keep,1,"bool",3,ms)};
    exercise(runtime,"g3n.causal-attention-forward","g3n.causal-attention.forward",6,row,in,6,view(out,16,"f32",4,head));
    REQUIRE(memcmp(out,v,16)==0);
    keep=0;
    exercise(runtime,"g3n.causal-attention-forward","g3n.causal-attention.forward",6,row,in,6,view(out,16,"f32",4,head));
    const float zeros[4]={0}; REQUIRE(memcmp(out,zeros,16)==0);
    keep=1; q[0]=FLT_MAX; k[0]=2;
    et_kernel_request_v1 r={sizeof(r),"g3n.causal-attention.forward","f32","cpu",6,row,1,{0}};
    et_kernel_tensor_view_v1 o=view(out,16,"f32",4,head);
    et_kernel_call_v1 c={sizeof(c),"g3n.causal-attention-forward",&r,6,sizeof(*in),sizeof(in),in,1,sizeof(o),sizeof(o),&o};
    REQUIRE(et_kernel_runtime_dispatch(runtime,&c,&error)!=0);
    REQUIRE(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
    REQUIRE(memcmp(out,zeros,16)==0);
  }
  et_kernel_runtime_destroy(runtime);
  return 11;
}
#if defined(ET_G3N_AOT_BRIDGE_TESTING)
int main(void) { return et_g3n_test_provider_transport_v1()==11 ? 0 : 1; }
#endif
