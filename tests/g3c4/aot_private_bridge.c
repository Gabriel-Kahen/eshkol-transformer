#include "eshkol_transformer/g3c4_primitives_abi.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "G3-C4-N AOT line %d: %s\n", __LINE__, #condition);     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

typedef struct row {
  const char *capability;
  const char *operation;
  size_t rank;
  uint64_t shape[6];
} row;

static const row rows[] = {
    {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6, {1,2,2,1,4,2}},
    {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6, {1,2,2,2,4,2}},
    {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6, {1,2,2,3,4,2}},
    {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6, {1,2,2,4,4,2}},
    {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6, {1,2,2,3,3,2}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,1,4,4}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,2,4,4}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,3,4,4}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,4,4,4}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,3,256,4}},
    {"g3c4.embedding-forward", "g3c4.embedding.forward", 4, {1,4,256,4}},
    {"g3c4.gelu", "g3c4.gelu.forward", 3, {1,3,8}},
    {"g3c4.gelu", "g3c4.gelu.forward", 3, {1,4,8}},
    {"g3c4.head-layout", "g3c4.heads.merge.forward", 4, {1,3,2,2}},
    {"g3c4.head-layout", "g3c4.heads.merge.forward", 4, {1,4,2,2}},
    {"g3c4.head-layout", "g3c4.heads.split.forward", 4, {1,3,2,2}},
    {"g3c4.head-layout", "g3c4.heads.split.forward", 4, {1,4,2,2}},
    {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3, {1,3,4}},
    {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3, {1,4,4}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,3,4,4}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,3,4,8}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,3,8,4}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,3,4,256}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,4,4,4}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,4,4,8}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,4,8,4}},
    {"g3c4.linear", "g3c4.linear.forward-no-bias", 4, {1,4,4,256}},
    {"g3c4.residual", "g3c4.residual.forward", 3, {1,4,4}},
};

static et_kernel_tensor_view_v1 view(void *data, size_t bytes, const char *dtype,
                                     size_t rank, const uint64_t *shape) {
  const et_kernel_tensor_view_v1 result = {
      sizeof(result), data, bytes, dtype, "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      0u, rank, shape};
  return result;
}

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  REQUIRE(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_g3c4_kernel_provider_v1();
}

static size_t elements(size_t rank, const uint64_t *shape) {
  size_t result = 1u;
  for (size_t i = 0u; i < rank; ++i) result *= (size_t)shape[i];
  return result;
}

static void exercise(et_kernel_runtime *runtime, const row *candidate) {
  float f0[2048], f1[2048], f2[2048], f3[2048], output[2048], before[2048];
  int64_t i0[4] = {0, 1, 2, 3}, i1[4] = {0, 1, 2, 3};
  uint8_t keep[16];
  float epsilon = 1.0f;
  uint64_t shapes[7][6] = {{0}};
  et_kernel_tensor_view_v1 inputs[6];
  et_kernel_tensor_view_v1 outputs[1];
  size_t input_count = 0u;
  size_t output_elements = 0u;
  for (size_t i = 0u; i < 2048u; ++i) {
    f0[i] = (float)((int)(i % 7u) - 3) * 0.125f;
    f1[i] = (float)((int)(i % 5u) - 2) * 0.0625f;
    f2[i] = (float)((int)(i % 3u) - 1) * 0.25f;
    f3[i] = 1.0f;
    output[i] = -123.0f;
  }
  memset(keep, 1, sizeof(keep));

  if (strcmp(candidate->operation, "g3c4.embedding.forward") == 0) {
    const size_t t = (size_t)candidate->shape[1], v = (size_t)candidate->shape[2];
    shapes[0][0] = 1u; shapes[0][1] = t;
    shapes[1][0] = v; shapes[1][1] = 4u;
    shapes[6][0] = 1u; shapes[6][1] = t; shapes[6][2] = 4u;
    for (size_t i = 0u; i < t; ++i) i0[i] = (int64_t)(i % v);
    inputs[0] = view(i0, t * sizeof(*i0), "i64", 2u, shapes[0]);
    inputs[1] = view(f1, v * 4u * sizeof(float), "f32", 2u, shapes[1]);
    input_count = 2u; output_elements = t * 4u;
    outputs[0] = view(output, output_elements * sizeof(float), "f32", 3u, shapes[6]);
  } else if (strcmp(candidate->operation, "g3c4.linear.forward-no-bias") == 0) {
    const size_t t = (size_t)candidate->shape[1], di = (size_t)candidate->shape[2];
    const size_t dout = (size_t)candidate->shape[3];
    shapes[0][0]=1u; shapes[0][1]=t; shapes[0][2]=di;
    shapes[1][0]=dout; shapes[1][1]=di;
    shapes[6][0]=1u; shapes[6][1]=t; shapes[6][2]=dout;
    inputs[0]=view(f0,t*di*sizeof(float),"f32",3u,shapes[0]);
    inputs[1]=view(f1,dout*di*sizeof(float),"f32",2u,shapes[1]);
    input_count=2u; output_elements=t*dout;
    outputs[0]=view(output,output_elements*sizeof(float),"f32",3u,shapes[6]);
  } else if (strcmp(candidate->operation, "g3c4.layer-norm.forward") == 0) {
    const size_t t=(size_t)candidate->shape[1];
    shapes[0][0]=1u; shapes[0][1]=t; shapes[0][2]=4u; shapes[1][0]=4u;
    memcpy(shapes[2],shapes[1],sizeof(shapes[2])); memcpy(shapes[6],shapes[0],sizeof(shapes[6]));
    inputs[0]=view(f0,t*4u*sizeof(float),"f32",3u,shapes[0]);
    inputs[1]=view(f3,4u*sizeof(float),"f32",1u,shapes[1]);
    inputs[2]=view(f2,4u*sizeof(float),"f32",1u,shapes[2]);
    inputs[3]=view(&epsilon,sizeof(float),"f32",0u,NULL);
    input_count=4u; output_elements=t*4u;
    outputs[0]=view(output,output_elements*sizeof(float),"f32",3u,shapes[6]);
  } else if (strcmp(candidate->operation, "g3c4.gelu.forward") == 0) {
    const size_t t=(size_t)candidate->shape[1], d=(size_t)candidate->shape[2];
    shapes[0][0]=1u; shapes[0][1]=t; shapes[0][2]=d; memcpy(shapes[6],shapes[0],sizeof(shapes[6]));
    inputs[0]=view(f0,t*d*sizeof(float),"f32",3u,shapes[0]);
    input_count=1u; output_elements=t*d;
    outputs[0]=view(output,output_elements*sizeof(float),"f32",3u,shapes[6]);
  } else if (strcmp(candidate->operation, "g3c4.residual.forward") == 0) {
    memcpy(shapes[0],candidate->shape,3u*sizeof(uint64_t)); memcpy(shapes[1],shapes[0],sizeof(shapes[1]));
    memcpy(shapes[6],shapes[0],sizeof(shapes[6])); output_elements=elements(3u,shapes[0]);
    inputs[0]=view(f0,output_elements*sizeof(float),"f32",3u,shapes[0]);
    inputs[1]=view(f1,output_elements*sizeof(float),"f32",3u,shapes[1]);
    input_count=2u; outputs[0]=view(output,output_elements*sizeof(float),"f32",3u,shapes[6]);
  } else if (strstr(candidate->operation, "heads.") != NULL) {
    const size_t t=(size_t)candidate->shape[1], h=(size_t)candidate->shape[2], dh=(size_t)candidate->shape[3];
    const int split=strstr(candidate->operation,"split")!=NULL;
    if (split) { shapes[0][0]=1u; shapes[0][1]=t; shapes[0][2]=h*dh; shapes[6][0]=1u; shapes[6][1]=h; shapes[6][2]=t; shapes[6][3]=dh; }
    else { shapes[0][0]=1u; shapes[0][1]=h; shapes[0][2]=t; shapes[0][3]=dh; shapes[6][0]=1u; shapes[6][1]=t; shapes[6][2]=h*dh; }
    output_elements=t*h*dh; inputs[0]=view(f0,output_elements*sizeof(float),"f32",split?3u:4u,shapes[0]);
    input_count=1u; outputs[0]=view(output,output_elements*sizeof(float),"f32",split?4u:3u,shapes[6]);
  } else {
    const size_t hq=(size_t)candidate->shape[1],hkv=(size_t)candidate->shape[2];
    const size_t tq=(size_t)candidate->shape[3],tk=(size_t)candidate->shape[4],dh=(size_t)candidate->shape[5];
    shapes[0][0]=1u; shapes[0][1]=hq; shapes[0][2]=tq; shapes[0][3]=dh;
    shapes[1][0]=1u; shapes[1][1]=hkv; shapes[1][2]=tk; shapes[1][3]=dh; memcpy(shapes[2],shapes[1],sizeof(shapes[2]));
    shapes[3][0]=1u; shapes[3][1]=tq; shapes[4][0]=1u; shapes[4][1]=tk;
    shapes[5][0]=1u; shapes[5][1]=tq; shapes[5][2]=tk; memcpy(shapes[6],shapes[0],sizeof(shapes[6]));
    inputs[0]=view(f0,hq*tq*dh*sizeof(float),"f32",4u,shapes[0]);
    inputs[1]=view(f1,hkv*tk*dh*sizeof(float),"f32",4u,shapes[1]);
    inputs[2]=view(f2,hkv*tk*dh*sizeof(float),"f32",4u,shapes[2]);
    inputs[3]=view(i0,tq*sizeof(*i0),"i64",2u,shapes[3]);
    inputs[4]=view(i1,tk*sizeof(*i1),"i64",2u,shapes[4]);
    inputs[5]=view(keep,tq*tk,"bool",3u,shapes[5]);
    input_count=6u; output_elements=hq*tq*dh;
    outputs[0]=view(output,output_elements*sizeof(float),"f32",4u,shapes[6]);
  }

  et_kernel_request_v1 request={sizeof(request),candidate->operation,"f32","cpu",candidate->rank,candidate->shape,1u,{0}};
  et_kernel_call_v1 call={sizeof(call),candidate->capability,&request,input_count,sizeof(inputs[0]),input_count*sizeof(inputs[0]),inputs,1u,sizeof(outputs[0]),sizeof(outputs[0]),outputs};
  et_kernel_error error;
  REQUIRE(et_kernel_runtime_dispatch(runtime,&call,&error)==0);
  for (size_t i=0u;i<output_elements;++i) REQUIRE(isfinite(output[i]));
  if (strcmp(candidate->operation,"g3c4.residual.forward")==0)
    for (size_t i=0u;i<output_elements;++i) REQUIRE(output[i]==f0[i]+f1[i]);
  memcpy(before,output,output_elements*sizeof(float));
  uint64_t adjacent[6]; memcpy(adjacent,candidate->shape,candidate->rank*sizeof(uint64_t)); adjacent[candidate->rank-1u]++;
  request.shape=adjacent;
  REQUIRE(et_kernel_runtime_dispatch(runtime,&call,&error)!=0);
  REQUIRE(error.category==ET_KERNEL_ERROR_UNSUPPORTED);
  REQUIRE(memcmp(before,output,output_elements*sizeof(float))==0);
}

int64_t et_g3c4_test_provider_transport_v1(void) {
  et_kernel_runtime *runtime=NULL;
  et_kernel_error error;
  REQUIRE(et_kernel_runtime_discover(resolve,NULL,&runtime,&error)==0);
  for (size_t i=0u;i<sizeof(rows)/sizeof(rows[0]);++i) exercise(runtime,&rows[i]);
  et_kernel_runtime_destroy(runtime);
  return (int64_t)(sizeof(rows)/sizeof(rows[0]));
}

#if defined(ET_G3C4_AOT_BRIDGE_TESTING)
int main(void) { return et_g3c4_test_provider_transport_v1()==28 ? 0 : 1; }
#endif
