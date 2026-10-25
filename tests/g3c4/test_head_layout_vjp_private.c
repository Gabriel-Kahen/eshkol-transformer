#include "test_common.h"
#include "../../native/m3t_c4_head_layout_vjp_private.h"
#include <fenv.h>

static float x[16], u[16], dx[16], output[16];
static const uint64_t token_shape[3] = {1u,4u,4u};
static const uint64_t head_shape[4] = {1u,2u,4u,2u};
static const uint64_t request_shape[4] = {1u,4u,2u,2u};
static et_kernel_runtime *runtime;

static et_kernel_tensor_view_v1 xv(int split) {
  return tv(x,sizeof(x),"f32",split ? 3u:4u,
            split ? token_shape:head_shape);
}
static et_kernel_tensor_view_v1 uv(int split) {
  return tv(u,sizeof(u),"f32",split ? 4u:3u,
            split ? head_shape:token_shape);
}
static et_kernel_tensor_view_v1 dxv(int split) {
  return tv(dx,sizeof(dx),"f32",split ? 3u:4u,
            split ? token_shape:head_shape);
}

static double loss(int split) {
  et_kernel_tensor_view_v1 input=xv(split);
  et_kernel_tensor_view_v1 result=tv(output,sizeof(output),"f32",
      split ? 4u:3u,split ? head_shape:token_shape);
  et_kernel_request_v1 request=rq(split ? "g3c4.heads.split.forward" :
      "g3c4.heads.merge.forward",4u,request_shape);
  et_kernel_call_v1 call=kc("g3c4.head-layout",&request,&input,1u,&result,1u);
  run(runtime,&call);
  double sum=0.0;
  for (size_t i=0;i<16u;++i) sum+=(double)output[i]*(double)u[i];
  return sum;
}

static void reject_unchanged(int operation,et_kernel_tensor_view_v1 *a,
                             et_kernel_tensor_view_v1 *up,
                             et_kernel_tensor_view_v1 *da,int expected) {
  unsigned char before_x[sizeof(x)],before_u[sizeof(u)],before_dx[sizeof(dx)];
  memcpy(before_x,x,sizeof(x));
  memcpy(before_u,u,sizeof(u));
  memcpy(before_dx,dx,sizeof(dx));
  CHECK(et_m3t_c4_head_layout_vjp_private_v1(operation,a,up,da)==expected);
  CHECK(memcmp(before_x,x,sizeof(x))==0);
  CHECK(memcmp(before_u,u,sizeof(u))==0);
  CHECK(memcmp(before_dx,dx,sizeof(dx))==0);
}

static void positive(int split) {
  const int operation=split ? ET_M3T_C4_HEAD_LAYOUT_SPLIT:
                              ET_M3T_C4_HEAD_LAYOUT_MERGE;
  for (size_t i=0;i<16u;++i) {
    x[i]=(float)((int)i-7)/16.0f;
    u[i]=(float)((int)((i*5u)%17u)-8)/32.0f;
  }
  u[5]=-0.0f;
  memset(dx,0xa5,sizeof(dx));
  et_kernel_tensor_view_v1 a=xv(split),up=uv(split),da=dxv(split);
  CHECK(et_m3t_c4_head_layout_vjp_private_v1(operation,&a,&up,&da)==
        ET_M3T_C4_HEAD_LAYOUT_VJP_OK);
  for (size_t t=0;t<4u;++t)
    for (size_t h=0;h<2u;++h)
      for (size_t d=0;d<2u;++d) {
        const size_t token=(t*2u+h)*2u+d;
        const size_t head=(h*4u+t)*2u+d;
        CHECK(fbits(dx[split ? token:head])==fbits(u[split ? head:token]));
      }
  const float step=0.0078125f;
  for (size_t i=0;i<16u;++i) {
    const float original=x[i];
    x[i]=original+step;const double high=loss(split);
    x[i]=original-step;const double low=loss(split);
    x[i]=original;
    CHECK(fabs((high-low)/(2.0*step)-(double)dx[i])<0.0001);
  }
  CHECK(isfinite(loss(split)));
}

static void negatives(int split) {
  const int operation=split ? ET_M3T_C4_HEAD_LAYOUT_SPLIT:
                              ET_M3T_C4_HEAD_LAYOUT_MERGE;
  et_kernel_tensor_view_v1 a=xv(split),up=uv(split),da=dxv(split);
  reject_unchanged(0,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT);
  a.struct_size=0u;
  reject_unchanged(operation,&a,&up,&da,
                   ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT);
  a=xv(split);a.rank=2u;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_SHAPE);
  a=xv(split);a.byte_length=60u;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_SHAPE);
  a=xv(split);uint64_t wrong[4]={1u,3u,4u,2u};a.shape=wrong;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_SHAPE);
  a=xv(split);up.dtype="i64";
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_DTYPE);
  up=uv(split);da.device="gpu";
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_DEVICE);
  da=dxv(split);a.layout=0u;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_LAYOUT);
  a=xv(split);da.offset_bytes=4u;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_LAYOUT);
  da=dxv(split);da.data=x;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_OVERLAP);
  da=dxv(split);da.data=u;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_OVERLAP);
  da=dxv(split);x[0]=NAN;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_VALUE);
  x[0]=0.0f;u[0]=INFINITY;
  reject_unchanged(operation,&a,&up,&da,ET_M3T_C4_HEAD_LAYOUT_VJP_VALUE);
  u[0]=0.0f;
  CHECK(et_m3t_c4_head_layout_vjp_private_v1(operation,NULL,&up,&da)==
        ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_head_layout_vjp_private_v1(operation,&a,NULL,&da)==
        ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT);
  CHECK(et_m3t_c4_head_layout_vjp_private_v1(operation,&a,&up,NULL)==
        ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST)==0);
  runtime=runtime_new();
  positive(1);negatives(1);
  positive(0);negatives(0);
  et_kernel_runtime_destroy(runtime);
  printf("PASS: C4 head-layout VJP checks=%zu\n",checks);
  return 0;
}
