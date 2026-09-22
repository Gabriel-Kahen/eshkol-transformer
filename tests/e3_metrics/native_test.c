#include "eshkol_transformer/e3_evaluation_metrics_abi.h"

#include <fenv.h>
#include <errno.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
static size_t checks;
static unsigned short x87_control(void);
static unsigned short x87_status(void);
#define CHECK(c) do { ++checks; if (!(c)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

enum { ACCUMULATE, CORRECT, FINALIZE };
static const char *const operations[] = {
  "e3.evaluation-metrics.accumulate", "e3.evaluation-metrics.correct.bool",
  "e3.evaluation-metrics.finalize"
};

typedef union storage { uint64_t alignment; unsigned char bytes[2080]; } storage;
typedef struct fixture {
  et_kernel_call_v1 call;
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 inputs[9], outputs[5];
  uint64_t request_shape[3], input_shapes[9][3];
  _Alignas(8) char capability[128], operation[128], f32[128], boolean[128], cpu[128], i64[128];
  storage input_data[9], output_data[5];
} fixture;

static void put_bits(void *data, size_t index, uint32_t bits) {
  memcpy((unsigned char *)data + index * 4u, &bits, 4u);
}
static uint32_t get_bits(const void *data, size_t index) {
  uint32_t bits; memcpy(&bits, (const unsigned char *)data + index * 4u, 4u); return bits;
}
static void put_float(fixture *f, size_t index, float value) { memcpy(f->inputs[index].data, &value, 4u); }
static void put_i64(fixture *f, size_t index, size_t element, int64_t value) {
  memcpy((unsigned char *)f->inputs[index].data + element*8u, &value, 8u);
}
static int64_t get_i64(const void *p) { int64_t n; memcpy(&n,p,8u); return n; }
static void *payload(storage *s) { return s->bytes + 8u; }
static et_kernel_tensor_view_v1 view(fixture *f, void *data, const char *dtype,
                                     size_t rank, const uint64_t *shape, size_t bytes) {
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),data,bytes,dtype,
    f->cpu,ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0u,rank,shape};
}
static void setup(fixture *f, size_t operation) {
  memset(f, 0, sizeof(*f));
  strcpy(f->capability, "e3.evaluation-metrics"); strcpy(f->operation, operations[operation]);
  strcpy(f->f32, "f32"); strcpy(f->boolean, "bool"); strcpy(f->cpu, "cpu"); strcpy(f->i64,"i64");
  f->request_shape[0]=1; f->request_shape[1]=2; f->request_shape[2]=256;
  f->request = (et_kernel_request_v1){sizeof(f->request), f->operation,f->f32,f->cpu,
    3u,f->request_shape,1u,{0}};
  const size_t ni=operation==ACCUMULATE?9u:3u, no=operation==ACCUMULATE?5u:operation==CORRECT?2u:4u;
  for (size_t i=0;i<9u;++i) {
    memset(f->input_data[i].bytes,0xa5,sizeof(storage));
    f->input_shapes[i][0]=1;f->input_shapes[i][1]=2;f->input_shapes[i][2]=256;
    const int integer=operation==ACCUMULATE && (i==3u||i==4u||i==8u);
    f->inputs[i]=view(f,payload(&f->input_data[i]),integer?f->i64:f->f32,0u,NULL,integer?8u:4u);
    memset(f->inputs[i].data,0,f->inputs[i].byte_length);
  }
  if(operation==CORRECT) {
    f->inputs[0]=view(f,payload(&f->input_data[0]),f->f32,3u,f->input_shapes[0],2048u);
    f->inputs[1]=view(f,payload(&f->input_data[1]),f->i64,2u,f->input_shapes[1],16u);
    f->inputs[2]=view(f,payload(&f->input_data[2]),f->boolean,2u,f->input_shapes[2],2u);
    memset(f->inputs[0].data,0,2048u);memset(f->inputs[1].data,0,16u);memset(f->inputs[2].data,1,2u);
  } else if(operation==ACCUMULATE) {
    put_float(f,5,2.0f);put_float(f,6,1.0f);put_float(f,7,1.0f);put_i64(f,8,0,1);
  } else { put_float(f,0,7.0f);put_float(f,1,3.0f);put_float(f,2,2.0f); }
  for(size_t o=0;o<5u;++o) {
    memset(f->output_data[o].bytes,0x5a,sizeof(storage));
    const int integer=(operation==ACCUMULATE&&o>=3u)||(operation==CORRECT&&o==1u);
    f->outputs[o]=view(f,payload(&f->output_data[o]),integer?f->i64:f->f32,0u,NULL,integer?8u:4u);
  }
  f->call=(et_kernel_call_v1){sizeof(f->call),f->capability,&f->request,ni,sizeof(f->inputs[0]),
    ni*sizeof(f->inputs[0]),f->inputs,no,sizeof(f->outputs[0]),no*sizeof(f->outputs[0]),f->outputs};
}
static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  CHECK(context == NULL);
  CHECK(strcmp(symbol, et_kernel_provider_symbol()) == 0);
  return et_e3_metrics_kernel_provider_v1();
}

/* The error record is separate trusted K1 control storage. Never exercise an
 * error/data alias as a recoverable provider rejection: K1 writes it first. */
static void reject(et_kernel_runtime *runtime, fixture *f, uint32_t category,
                   uint32_t code) {
  fixture before; memcpy(&before, f, sizeof(before));
  et_kernel_error error;
  memset(&error, 0x71, sizeof(error));
  const unsigned short control = x87_control(), status_word = x87_status();
  const unsigned mxcsr = _mm_getcsr();
  errno = E2BIG;
  const int32_t status = et_kernel_runtime_dispatch(runtime, &f->call, &error);
  CHECK(errno == E2BIG);
  CHECK(x87_control() == control); CHECK(x87_status() == status_word);
  CHECK(_mm_getcsr() == mxcsr);
  if (status != (int32_t)category || (code != UINT32_MAX && error.code != code))
    fprintf(stderr, "%s: got %d/%u expected %u/%u (%s)\n", f->operation,
      status, error.code, category, code, error.message);
  CHECK(status == (int32_t)category); CHECK(error.category == category);
  CHECK(error.code != ET_KERNEL_CODE_OK);
  CHECK(error.operation[0] && error.message[0]);
  if (code == ET_KERNEL_CODE_PROVIDER_REJECTED || !strncmp(error.operation,"e3.",3))
    CHECK(strcmp(error.operation,f->request.operation)==0);
  if (code != UINT32_MAX) CHECK(error.code == code);
  CHECK(memcmp(f, &before, sizeof(before)) == 0);
}
static void succeed(et_kernel_runtime *runtime, fixture *f) {
  fixture before; memcpy(&before, f, sizeof(before));
  et_kernel_error error;
  errno = E2BIG;
  CHECK(et_kernel_runtime_dispatch(runtime, &f->call, &error) == 0);
  CHECK(errno == E2BIG);
  CHECK(error.category == ET_KERNEL_ERROR_NONE && error.code == ET_KERNEL_CODE_OK);
  /* Exempt only exact declared output spans; metadata, inputs and surrounding
   * guard bytes remain covered, including when outputs share adjacent backing. */
  for (size_t i = 0; i < f->call.output_count; ++i) {
    const uintptr_t offset = (uintptr_t)f->outputs[i].data - (uintptr_t)f;
    CHECK(offset <= sizeof(*f) - f->outputs[i].byte_length);
    memcpy((unsigned char *)&before + offset, f->outputs[i].data,
           f->outputs[i].byte_length);
  }
  CHECK(memcmp(f, &before, sizeof(before)) == 0);
}
static void output_is(fixture *f, size_t output, size_t index, uint32_t bits) {
  CHECK(get_bits(f->outputs[output].data, index) == bits);
}

static void capability_tests(et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *p=et_e3_metrics_kernel_provider_v1();
  CHECK(p==et_e3_metrics_kernel_provider_v1()); CHECK(p->capability_count==1);
  CHECK(p->abi_major==1 && p->abi_minor==0 && p->required_features==0);
  CHECK(strcmp(p->name,"e3.cpu-f32-bool.serial")==0);
  CHECK(strcmp(p->version,"1.0")==0);CHECK(strcmp(p->evidence,"E3:cpu-f32-bool-metrics-v1")==0);
  const et_kernel_capability_v1 *c=et_kernel_runtime_capability_find(runtime,"e3.evaluation-metrics");
  CHECK(c && c->status==ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(strcmp(c->implementation,p->name)==0 && strcmp(c->version,p->version)==0 && strcmp(c->evidence,p->evidence)==0);
  CHECK(c->operation_count==3 && c->deterministic==1);
  CHECK(c->dtype_count==1 && strcmp(c->dtypes[0],"f32")==0);
  CHECK(c->device_count==1 && strcmp(c->devices[0],"cpu")==0);
  CHECK(c->shape_range_count==1 && c->shape_ranges[0].rank==3);
  const uint64_t dims[]={1,2,256};
  for(size_t d=0;d<3;++d) {
    const et_kernel_dimension_range_v1 *r=&c->shape_ranges[0].dimensions[d];
    CHECK(r->minimum==dims[d] && r->maximum==dims[d] && r->maximum_unbounded==0);
  }
  size_t verified=0;
  for(size_t i=0;i<et_kernel_runtime_capability_count(runtime);++i)
    verified+=et_kernel_runtime_capability_at(runtime,i)->status==ET_KERNEL_CAPABILITY_VERIFIED;
  CHECK(verified==1);
  et_kernel_runtime *baseline=NULL;et_kernel_error e;
  CHECK(et_kernel_runtime_baseline(&baseline,&e)==0);
  for(size_t op=0;op<3;++op) {
    CHECK(strcmp(c->operations[op],operations[op])==0);
    fixture f;setup(&f,op);
    reject(baseline,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    for(unsigned deterministic=0;deterministic<2;++deterministic) {
      f.request.deterministic=(uint8_t)deterministic;
      CHECK(et_kernel_runtime_capability_require(runtime,f.capability,&f.request,NULL,&e)==0);
    }
    for(size_t d=0;d<3;++d) for(size_t side=0;side<2;++side) {
      setup(&f,op);f.request_shape[d]=side?dims[d]+1:dims[d]-1;
      reject(runtime,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    }
    setup(&f,op);f.request.rank=2;
    reject(runtime,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f,op);f.request.dtype=f.boolean;
    reject(runtime,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f,op);strcpy(f.operation,"e3.evaluation-metrics.correct.f32");
    reject(runtime,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f,op);strcpy(f.cpu,"cuda:0");
    reject(runtime,&f,ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  }
  et_kernel_runtime_destroy(baseline);
}
static void positive_tests(et_kernel_runtime *runtime) {
  fixture f;
  for(size_t op=0;op<3;++op) {
    setup(&f,op);succeed(runtime,&f);
    storage expected[5];memcpy(expected,f.output_data,sizeof(expected));
    for(size_t n=0;n<8;++n) { succeed(runtime,&f);CHECK(memcmp(expected,f.output_data,sizeof(expected))==0); }
    setup(&f,op);
    for(size_t o=0;o<f.call.output_count;++o)
      memset(f.outputs[o].data,0xff,f.outputs[o].byte_length);
    succeed(runtime,&f);
  }
  for(unsigned mask=1;mask<4;++mask) for(size_t which=0;which<4;++which) {
    setup(&f,CORRECT);unsigned char *m=f.inputs[2].data;m[0]=(mask&1u)!=0;m[1]=(mask&2u)!=0;
    put_i64(&f,1,0,which&1u?255:0);put_i64(&f,1,1,which&2u?255:0);
    succeed(runtime,&f);CHECK(get_i64(f.outputs[1].data)==(int64_t)(m[0]+m[1]));
    float expected=(float)(m[0]*!(which&1u)+m[1]*!(which&2u));
    CHECK(memcmp(f.outputs[0].data,&expected,4)==0);
  }
  for(size_t t=0;t<2;++t) {
    setup(&f,CORRECT);put_bits(f.inputs[0].data,256*t+17,0x3f800000u);
    put_bits(f.inputs[0].data,256*t+255,0x3f800000u);put_i64(&f,1,t,17);
    succeed(runtime,&f);output_is(&f,0,0,0x40000000u);
    put_i64(&f,1,t,255);succeed(runtime,&f);output_is(&f,0,0,0x3f800000u);
    setup(&f,CORRECT);
    for(size_t j=0;j<512;++j) put_bits(f.inputs[0].data,j,j%2?0u:0x80000000u);
    succeed(runtime,&f);output_is(&f,0,0,0x40000000u);
    setup(&f,CORRECT);put_bits(f.inputs[0].data,256*t+255,1u);put_i64(&f,1,t,255);
    succeed(runtime,&f);output_is(&f,0,0,0x40000000u);
  }
  setup(&f,CORRECT);f.inputs[2].data=(unsigned char *)f.inputs[2].data+1;memset(f.inputs[2].data,1,2);
  succeed(runtime,&f);
  setup(&f,ACCUMULATE);succeed(runtime,&f);
  output_is(&f,0,0,0x40000000u);output_is(&f,1,0,0x3f800000u);output_is(&f,2,0,0x3f800000u);
  CHECK(get_i64(f.outputs[3].data)==1 && get_i64(f.outputs[4].data)==1);
  setup(&f,ACCUMULATE);put_float(&f,0,7);put_float(&f,1,3);put_float(&f,2,2);put_i64(&f,3,0,3);put_i64(&f,4,0,2);
  succeed(runtime,&f);output_is(&f,0,0,0x41100000u);output_is(&f,1,0,0x40800000u);output_is(&f,2,0,0x40400000u);
  CHECK(get_i64(f.outputs[3].data)==4 && get_i64(f.outputs[4].data)==3);
  setup(&f,ACCUMULATE);put_float(&f,1,16382);put_i64(&f,3,0,16382);put_i64(&f,4,0,8191);
  put_float(&f,6,2);put_i64(&f,8,0,2);succeed(runtime,&f);
  output_is(&f,1,0,0x46800000u);CHECK(get_i64(f.outputs[3].data)==16384 && get_i64(f.outputs[4].data)==8192);
  setup(&f,FINALIZE);succeed(runtime,&f);output_is(&f,0,0,0x40155555u);output_is(&f,1,0,0x40400000u);output_is(&f,3,0,0x3f2aaaabu);
  setup(&f,FINALIZE);put_float(&f,0,0);put_float(&f,1,16384);put_float(&f,2,16384);
  succeed(runtime,&f);output_is(&f,0,0,0);output_is(&f,2,0,0x3f800000u);output_is(&f,3,0,0x3f800000u);
  const uint32_t small[]={0u,0x80000000u,1u,0x007fffffu,0x00800000u};
  for(size_t b=0;b<COUNT(small);++b) {
    setup(&f,ACCUMULATE);put_bits(f.inputs[5].data,0,small[b]);succeed(runtime,&f);
    output_is(&f,0,0,small[b]==0x80000000u?0u:small[b]);
    setup(&f,FINALIZE);put_bits(f.inputs[0].data,0,small[b]);put_float(&f,1,1);put_float(&f,2,-0.0f);
    succeed(runtime,&f);output_is(&f,0,0,small[b]==0x80000000u?0u:small[b]);output_is(&f,3,0,0u);output_is(&f,2,0,0x3f800000u);
  }
  setup(&f,ACCUMULATE);
  for(size_t i=0;i<3;++i) put_float(&f,i,-0.0f);
  put_float(&f,5,-0.0f);put_float(&f,7,-0.0f);succeed(runtime,&f);output_is(&f,0,0,0u);output_is(&f,2,0,0u);
  setup(&f,FINALIZE);put_bits(f.inputs[0].data,0,1);put_float(&f,1,16384);put_float(&f,2,1);
  succeed(runtime,&f);output_is(&f,0,0,0);output_is(&f,2,0,0x3f800000u);output_is(&f,3,0,0x38800000u);
}
#define DOMAIN ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_PROVIDER_REJECTED
#define CAP ET_KERNEL_ERROR_UNSUPPORTED,ET_KERNEL_CODE_PROVIDER_REJECTED
static void value_rejections(et_kernel_runtime *runtime) {
  fixture f;
  const uint32_t bad[]={0x7fc00001u,0x7f800001u,0xff800001u,0x7f800000u,0xff800000u,0xbf800000u,0x80000001u};
  for(size_t op=0;op<3;++op) {
    setup(&f,op);const size_t ni=f.call.input_count;
    for(size_t i=0;i<ni;++i) {
      setup(&f,op);if(strcmp(f.inputs[i].dtype,"f32")) continue;
      for(size_t b=0;b<COUNT(bad);++b) {
        if(op==CORRECT && b>=5) continue; /* finite negative logits are valid */
        const size_t count=op==CORRECT?512:1;
        for(size_t j=0;j<count;++j) {
          setup(&f,op);put_bits(f.inputs[i].data,j,bad[b]);reject(runtime,&f,DOMAIN);
          if(op==CORRECT) { ((uint8_t *)f.inputs[2].data)[j/256]=0;reject(runtime,&f,DOMAIN); }
        }
      }
    }
  }
  for(size_t t=0;t<2;++t) for(size_t b=0;b<2;++b) {
    setup(&f,CORRECT);put_i64(&f,1,t,b?256:-1);
    reject(runtime,&f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE);
    ((uint8_t *)f.inputs[2].data)[t]=0;
    reject(runtime,&f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE);
    setup(&f,CORRECT);((uint8_t *)f.inputs[2].data)[t]=b?255:2;reject(runtime,&f,DOMAIN);
  }
  setup(&f,CORRECT);memset(f.inputs[2].data,0,2);reject(runtime,&f,DOMAIN);
  /* Target precedes bad logit and bad mask; logits precede mask. */
  put_i64(&f,1,1,256);put_bits(f.inputs[0].data,0,0x7f800001);reject(runtime,&f,ET_KERNEL_ERROR_SHAPE_MISMATCH,ET_KERNEL_CODE_INVALID_SHAPE);
  const size_t counters[]={3,4,8};
  for(size_t i=0;i<3;++i) {setup(&f,ACCUMULATE);put_i64(&f,counters[i],0,-1);reject(runtime,&f,DOMAIN);}
  for(int64_t active=0;active<=3;active+=3) {setup(&f,ACCUMULATE);put_i64(&f,8,0,active);reject(runtime,&f,DOMAIN);}
  for(size_t i=3;i<=4;++i) {
    setup(&f,ACCUMULATE);put_i64(&f,i,0,INT64_MAX);put_float(&f,1,0.5f);reject(runtime,&f,CAP);
    put_float(&f,5,-1);reject(runtime,&f,DOMAIN); /* float domain precedes caps */
  }
  setup(&f,ACCUMULATE);put_i64(&f,4,0,8193);reject(runtime,&f,CAP);
  setup(&f,ACCUMULATE);put_i64(&f,3,0,16385);reject(runtime,&f,CAP);
  setup(&f,ACCUMULATE);put_i64(&f,4,0,1);reject(runtime,&f,DOMAIN);
  setup(&f,ACCUMULATE);put_i64(&f,3,0,3);put_i64(&f,4,0,1);put_float(&f,1,3);reject(runtime,&f,DOMAIN);
  setup(&f,ACCUMULATE);put_float(&f,1,1);reject(runtime,&f,DOMAIN);
  setup(&f,ACCUMULATE);put_float(&f,0,1);reject(runtime,&f,DOMAIN);
  setup(&f,ACCUMULATE);put_float(&f,6,2);reject(runtime,&f,DOMAIN);
  for(size_t i=2;i<=7;i+=5) for(size_t b=0;b<2;++b) {
    setup(&f,ACCUMULATE);put_float(&f,0,1);put_float(&f,1,1);put_i64(&f,3,0,1);put_i64(&f,4,0,1);
    put_float(&f,i,b?2:0.5f);reject(runtime,&f,DOMAIN);
  }
  setup(&f,ACCUMULATE);put_float(&f,1,8192);put_i64(&f,3,0,8192);put_i64(&f,4,0,8192);
  put_float(&f,0,FLT_MAX);put_float(&f,5,FLT_MAX);reject(runtime,&f,CAP); /* cap before sum overflow */
  setup(&f,ACCUMULATE);put_float(&f,0,FLT_MAX);put_float(&f,1,1);put_i64(&f,3,0,1);put_i64(&f,4,0,1);
  put_float(&f,5,FLT_MAX);reject(runtime,&f,DOMAIN);
  setup(&f,ACCUMULATE);put_float(&f,1,16384);put_i64(&f,3,0,16384);put_i64(&f,4,0,8192);reject(runtime,&f,CAP);
  for(size_t i=0;i<6;++i) {
    setup(&f,FINALIZE);
    if(i==0) put_float(&f,1,0);
    if(i==1) put_float(&f,1,0.5f);
    if(i==2) put_float(&f,2,0.5f);
    if(i==3) put_float(&f,2,4);
    if(i==4) put_float(&f,0,FLT_MAX);
    if(i==5) {put_float(&f,1,16384.5f);put_float(&f,2,16385);}
    if(i==5) reject(runtime,&f,CAP); else reject(runtime,&f,DOMAIN);
  }
}
static void schema_rejections(et_kernel_runtime *runtime) {
  for (size_t op = 0; op < 3u; ++op) {
    fixture f; setup(&f, op);
    const size_t ni = f.call.input_count, no = f.call.output_count;
    for (size_t side = 0; side < 2u; ++side) for (size_t index = 0; index < (side ? no : ni); ++index) {
#define RESET_VIEW() setup(&f, op); v = side ? &f.outputs[index] : &f.inputs[index]
      et_kernel_tensor_view_v1 *v;
      RESET_VIEW(); v->struct_size = sizeof(*v) - 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      RESET_VIEW(); v->struct_size = sizeof(*v) + 8u;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      RESET_VIEW(); v->byte_length--;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->byte_length++;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->data = NULL;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->data = (void *)(UINTPTR_MAX - 1u);
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
      RESET_VIEW(); v->layout = 0u;
      reject(runtime, &f, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->offset_bytes = 4u;
      reject(runtime, &f, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->dtype = "f16"; v->byte_length = v->rank == 3u ? 1024u : v->rank ? 4u : 2u;
      reject(runtime, &f, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT);
      RESET_VIEW(); v->device = "cuda:0";
      reject(runtime, &f, ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW();
      if (strcmp(v->dtype, "bool") != 0) {
        v->data = (unsigned char *)v->data + 1u;
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      }
      RESET_VIEW();
      if (v->rank != 0u) {
        uint64_t *shape = f.input_shapes[index];
        shape[0] = 2u; shape[1] = 1u;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->rank = 1u; v->shape = &f.request_shape[1];
        v->byte_length = strcmp(v->dtype,"i64")==0 ? 16u : strcmp(v->dtype,"bool")==0 ? 2u : 8u;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); f.input_shapes[index][0]=0;v->byte_length=0;v->data=NULL;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->shape = NULL;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW();
        f.input_shapes[index][0] = UINT64_MAX;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INTEGER_OVERFLOW);
        RESET_VIEW(); v->shape = (const uint64_t *)((const unsigned char *)v->shape + 1u);
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      } else {
        v->shape = f.request_shape;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->rank = 1u; v->shape = f.request_shape;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
      }
#undef RESET_VIEW
    }
    setup(&f, op); f.call.struct_size--;
    reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
    setup(&f, op); f.request.struct_size--;
    reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
    setup(&f, op); f.request.deterministic = 2u;
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_TEXT);
    for (size_t side = 0; side < 2u; ++side) {
      setup(&f, op);
      if (side) --f.call.output_stride; else --f.call.input_stride;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      setup(&f, op);
      if (side) --f.call.output_bytes; else --f.call.input_bytes;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) f.call.outputs = (unsigned char *)f.outputs + 1u;
      else f.call.inputs = (unsigned char *)f.inputs + 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      setup(&f, op);
      if (side) f.call.output_count = SIZE_MAX; else f.call.input_count = SIZE_MAX;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) f.call.output_stride = SIZE_MAX - 7u; else f.call.input_stride = SIZE_MAX - 7u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) { f.call.output_count = 0u; f.call.output_stride = 0u; f.call.output_bytes = 0u; f.call.outputs = NULL; }
      else { f.call.input_count = 0u; f.call.input_stride = 0u; f.call.input_bytes = 0u; f.call.inputs = NULL; }
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
    }
  }
}

static void alias_rejections(et_kernel_runtime *runtime) {
  for(size_t op=0;op<3;++op) {
    fixture f;setup(&f,op);const size_t ni=f.call.input_count,no=f.call.output_count;
    for(size_t o=0;o<no;++o) for(size_t i=0;i<ni;++i) for(size_t shift=0;shift<2;++shift) {
      setup(&f,op);f.outputs[o].data=(unsigned char *)f.inputs[i].data+shift;
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    for(size_t a=0;a<no;++a) for(size_t b=a+1;b<no;++b) for(size_t shift=0;shift<2;++shift) {
      setup(&f,op);f.outputs[b].data=(unsigned char *)f.outputs[a].data+shift;
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    for(size_t a=0;a<ni;++a) for(size_t b=a+1;b<ni;++b) {
      setup(&f,op);f.inputs[b].data=f.inputs[a].data;
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
      setup(&f,op);
      const size_t wa=!strcmp(f.inputs[a].dtype,"i64")?8u:!strcmp(f.inputs[a].dtype,"bool")?1u:4u;
      const size_t wb=!strcmp(f.inputs[b].dtype,"i64")?8u:!strcmp(f.inputs[b].dtype,"bool")?1u:4u;
      if(f.inputs[a].byte_length>wb) {
        f.inputs[b].data=(unsigned char *)f.inputs[a].data+wb;
        reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
      } else if(f.inputs[b].byte_length>wa) {
        f.inputs[a].data=(unsigned char *)f.inputs[b].data+wa;
        reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
      }
    }
    for(size_t o=0;o<no;++o) for(size_t metadata=0;metadata<(op==CORRECT?14u:11u);++metadata) {
      setup(&f,op);
      void *spans[]={&f.call,&f.request,f.inputs,f.outputs,f.request_shape,
        f.capability,f.operation,f.f32,f.cpu,f.i64,f.boolean,
        f.input_shapes[0],f.input_shapes[1],f.input_shapes[2]};
      /* Unused text is not live metadata; give each distinct symbol a real role. */
      if(metadata==9 && op==FINALIZE) continue;
      if(metadata==10 && op!=CORRECT) continue;
      f.outputs[o].data=spans[metadata];
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    /* Place the terminating NUL alone at an aligned output start. The live
     * symbol substring is shifted within a real buffer, never read out of range. */
    for(size_t symbol=0;symbol<5;++symbol) {
      setup(&f,op);
      char *base=symbol==0?f.capability:symbol==1?f.operation:symbol==2?f.f32:symbol==3?f.cpu:f.i64;
      if(symbol==4 && op==FINALIZE) continue;
      const size_t len=strlen(base), shift=(8u-len%8u)%8u;
      memmove(base+shift,base,len+1u);
      const char *moved=base+shift;
      if(symbol==0) f.call.capability=moved;
      if(symbol==1) f.request.operation=moved;
      if(symbol==2) f.request.dtype=moved;
      if(symbol==3) f.request.device=moved;
      for(size_t side=0;side<2;++side) for(size_t i=0;i<(side?no:ni);++i) {
        et_kernel_tensor_view_v1 *v=side?&f.outputs[i]:&f.inputs[i];
        if(v->dtype==base) v->dtype=moved;
        if(v->device==base) v->device=moved;
      }
      f.outputs[0].data=base+shift+len;
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    setup(&f,op);
    void *adjacent=(unsigned char *)f.inputs[0].data+f.inputs[0].byte_length;
    memcpy(adjacent,f.inputs[1].data,f.inputs[1].byte_length);f.inputs[1].data=adjacent;
    succeed(runtime,&f);
    setup(&f,op);f.outputs[0].data=(unsigned char *)f.inputs[0].data+f.inputs[0].byte_length;
    succeed(runtime,&f);
    setup(&f,op);
    size_t offset=8;
    for(size_t o=0;o<no;++o) {
      offset=(offset+7u)/8u*8u;f.outputs[o].data=f.output_data[0].bytes+offset;offset+=f.outputs[o].byte_length;
    }
    succeed(runtime,&f);
  }
}
static void unique_view_text_tests(et_kernel_runtime *runtime) {
  for(size_t op=0;op<3;++op) {
    fixture f;setup(&f,op);
    _Alignas(8) char text[14][2][16];memset(text,0x35,sizeof(text));
    const size_t ni=f.call.input_count,no=f.call.output_count;
    for(size_t i=0;i<ni+no;++i) {
      et_kernel_tensor_view_v1 *v=i<ni?&f.inputs[i]:&f.outputs[i-ni];
      strcpy(text[i][0],v->dtype);strcpy(text[i][1],v->device);
      v->dtype=text[i][0];v->device=text[i][1];
    }
    unsigned char before[sizeof(text)];memcpy(before,text,sizeof(text));
    succeed(runtime,&f);
    for(size_t o=0;o<no;++o) for(size_t i=0;i<ni+no;++i) for(size_t field=0;field<2;++field) {
      void *saved=f.outputs[o].data;f.outputs[o].data=text[i][field];
      reject(runtime,&f,ET_KERNEL_ERROR_INVALID_ARGUMENT,ET_KERNEL_CODE_ALIASING_OUTPUT);
      CHECK(!memcmp(before,text,sizeof(text)));f.outputs[o].data=saved;
    }
  }
}
static void compatible_table_tests(et_kernel_runtime *runtime) {
  typedef struct extended_view {et_kernel_tensor_view_v1 prefix;unsigned char extension[8];} extended_view;
  for(size_t op=0;op<3;++op) {
    fixture f;setup(&f,op);
    struct {et_kernel_call_v1 prefix;unsigned char extension[8];} call;
    struct {et_kernel_request_v1 prefix;unsigned char extension[8];} request;
    extended_view inputs[9],outputs[5];
    memset(&call,0xb6,sizeof(call));memset(&request,0xe9,sizeof(request));
    memset(inputs,0xc7,sizeof(inputs));memset(outputs,0xd8,sizeof(outputs));
    call.prefix=f.call;call.prefix.struct_size=sizeof(call);
    request.prefix=f.request;request.prefix.struct_size=sizeof(request);call.prefix.request=&request.prefix;
    for(size_t i=0;i<f.call.input_count;++i) {inputs[i].prefix=f.inputs[i];inputs[i].prefix.struct_size=sizeof(inputs[i]);}
    for(size_t i=0;i<f.call.output_count;++i) {outputs[i].prefix=f.outputs[i];outputs[i].prefix.struct_size=sizeof(outputs[i]);}
    call.prefix.inputs=inputs;call.prefix.input_stride=sizeof(inputs[0]);call.prefix.input_bytes=f.call.input_count*sizeof(inputs[0]);
    call.prefix.outputs=outputs;call.prefix.output_stride=sizeof(outputs[0]);call.prefix.output_bytes=f.call.output_count*sizeof(outputs[0]);
    unsigned char saved_call[sizeof(call)],saved_request[sizeof(request)],saved_inputs[sizeof(inputs)],saved_outputs[sizeof(outputs)];
    memcpy(saved_call,&call,sizeof(call));memcpy(saved_request,&request,sizeof(request));
    memcpy(saved_inputs,inputs,sizeof(inputs));memcpy(saved_outputs,outputs,sizeof(outputs));
    et_kernel_error error;CHECK(et_kernel_runtime_dispatch(runtime,&call.prefix,&error)==0);
    CHECK(!memcmp(saved_call,&call,sizeof(call))&&!memcmp(saved_request,&request,sizeof(request)));
    CHECK(!memcmp(saved_inputs,inputs,sizeof(inputs))&&!memcmp(saved_outputs,outputs,sizeof(outputs)));
    for(size_t o=0;o<f.call.output_count;++o) for(size_t m=0;m<4;++m) {
      void *extra[]={call.extension,request.extension,inputs[f.call.input_count-1].extension,outputs[f.call.output_count-1].extension};
      outputs[o].prefix.data=extra[m];memcpy(saved_outputs,outputs,sizeof(outputs));
      fixture before;memcpy(&before,&f,sizeof(f));
      CHECK(et_kernel_runtime_dispatch(runtime,&call.prefix,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
      CHECK(error.code==ET_KERNEL_CODE_ALIASING_OUTPUT);
      CHECK(!memcmp(saved_call,&call,sizeof(call))&&!memcmp(saved_request,&request,sizeof(request)));
      CHECK(!memcmp(saved_inputs,inputs,sizeof(inputs))&&!memcmp(saved_outputs,outputs,sizeof(outputs)));
      CHECK(!memcmp(&before,&f,sizeof(f)));
      outputs[o].prefix.data=f.outputs[o].data;
    }
  }
}
static unsigned short x87_control(void) {unsigned short x;__asm__ volatile("fnstcw %0":"=m"(x));return x;}
static unsigned short x87_status(void) {unsigned short x;__asm__ volatile("fnstsw %0":"=am"(x));return x;}
static void set_x87_control(unsigned short x) {__asm__ volatile("fldcw %0"::"m"(x));}
static void invalid_value(fixture *f) {put_bits(f->inputs[0].data,0,0x7f800001u);}
static void environment_tests(et_kernel_runtime *runtime) {
  fenv_t saved;CHECK(fegetenv(&saved)==0);CHECK(feclearexcept(FE_ALL_EXCEPT)==0);
  const unsigned short cw=x87_control();const unsigned mx=_mm_getcsr();
  for(size_t op=0;op<3;++op) {
    fixture f;
    const unsigned bits[]={0x40u,0x8000u,0x2000u,0x4000u,0x6000u};
    for(size_t b=0;b<COUNT(bits);++b) {
      setup(&f,op);invalid_value(&f);_mm_setcsr(mx|bits[b]);reject(runtime,&f,CAP);_mm_setcsr(mx);
    }
    for(size_t b=0;b<6;++b) {
      setup(&f,op);invalid_value(&f);_mm_setcsr(mx&~(1u<<(7+b)));reject(runtime,&f,CAP);_mm_setcsr(mx);
      set_x87_control((unsigned short)(cw&~(1u<<b)));reject(runtime,&f,CAP);set_x87_control(cw);
    }
    for(unsigned b=0x400;b<=0xc00;b+=0x400) {
      setup(&f,op);invalid_value(&f);set_x87_control((unsigned short)(cw|b));reject(runtime,&f,CAP);set_x87_control(cw);
    }
    /* Preserve independent sticky flags, not just the merged FE_ALL_EXCEPT view. */
    for(size_t flags=0;flags<3;++flags) for(size_t failure=0;failure<3;++failure) {
      setup(&f,op);
      if(failure==1) invalid_value(&f);
      if(failure==2) {
        if(op==CORRECT) memset(f.inputs[2].data,0,2);
        if(op==ACCUMULATE) put_float(&f,5,-1);
        if(op==FINALIZE) put_float(&f,0,FLT_MAX); /* expf sets ERANGE internally */
      }
      CHECK(feclearexcept(FE_ALL_EXCEPT)==0);
      if(flags!=1) __asm__ volatile("fldz\n\tfldz\n\tfdivp\n\tfstp %%st(0)":::"st");
      _mm_setcsr((_mm_getcsr()&~0x3fu)|(flags?0x04u:0u));
      const unsigned short bc=x87_control(),bs=x87_status();const unsigned bm=_mm_getcsr();
      fixture before;memcpy(&before,&f,sizeof(f));et_kernel_error error;errno=EILSEQ;
      const int32_t rc=et_e3_metrics_kernel_provider_v1()->validate_call(&f.call,&error);
      CHECK(errno==EILSEQ);CHECK((rc!=0)==(failure!=0));
      CHECK(x87_control()==bc&&x87_status()==bs&&_mm_getcsr()==bm);CHECK(!memcmp(&before,&f,sizeof(f)));
      if(failure) reject(runtime,&f,DOMAIN);else {succeed(runtime,&f);CHECK(x87_control()==bc);CHECK((_mm_getcsr()&~0x3fu)==(bm&~0x3fu));}
      CHECK(feclearexcept(FE_ALL_EXCEPT)==0);_mm_setcsr(mx);
    }
  }
  /* Direct callback prefix cases are defensive only, not an alternative API. */
  et_kernel_error error;const et_kernel_provider_v1 *p=et_e3_metrics_kernel_provider_v1();
  errno=E2BIG;CHECK(p->validate_call(NULL,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(errno==E2BIG);
  CHECK(error.code==ET_KERNEL_CODE_NULL_ARGUMENT&&strcmp(error.operation,"e3.metrics.provider")==0);
  fixture f;setup(&f,FINALIZE);f.call.request=NULL;
  CHECK(p->validate_call(&f.call,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);CHECK(error.code==ET_KERNEL_CODE_NULL_ARGUMENT);
  setup(&f,FINALIZE);f.call.struct_size--;
  CHECK(p->validate_call(&f.call,&error)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(error.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  setup(&f,FINALIZE);f.request.struct_size--;
  CHECK(p->validate_call(&f.call,&error)==ET_KERNEL_ERROR_VERSION_MISMATCH);CHECK(error.code==ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  CHECK(fesetenv(&saved)==0);
}
int main(void) {
  checks=0;fenv_t original;CHECK(fegetenv(&original)==0);CHECK(fesetenv(FE_DFL_ENV)==0);
  et_kernel_runtime *runtime=NULL;et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve,NULL,&runtime,&error)==0);
  capability_tests(runtime);positive_tests(runtime);value_rejections(runtime);schema_rejections(runtime);
  alias_rejections(runtime);unique_view_text_tests(runtime);compatible_table_tests(runtime);environment_tests(runtime);
  et_kernel_runtime_destroy(runtime);CHECK(fesetenv(&original)==0);
  printf("E3-METRICS native: %zu checks; schemas, exact bytes, caps, fenv/errno PASS\n",checks);
  return 0;
}
