#include "eshkol_transformer/e3_evaluation_metrics_abi.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

/* Link this test alone with --wrap=fegetenv and --wrap=fesetenv. The production
 * source/archive has no failpoint. A failed restoration deliberately does not
 * promise restored FP state; the harness restores its saved environment itself. */
extern int __real_fegetenv(fenv_t *environment);
extern int __real_fesetenv(const fenv_t *environment);
static unsigned fault, saves, restores;
enum { FAIL_SAVE = 1u, FAIL_RESTORE = 2u };
int __wrap_fegetenv(fenv_t *environment) {
  ++saves;
  if (fault & FAIL_SAVE) { errno = ENOSYS; return 1; }
  return __real_fegetenv(environment);
}
int __wrap_fesetenv(const fenv_t *environment) {
  ++restores;
  if (fault & FAIL_RESTORE) { errno = EIO; return 1; }
  return __real_fesetenv(environment);
}

static size_t checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); \
} } while (0)

typedef struct fixture {
  et_kernel_call_v1 call;
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 inputs[3], outputs[4];
  uint64_t shape[3];
  float input_data[3];
  uint32_t before_guard;
  float output_data[4];
  uint32_t after_guard;
} fixture;

static void setup(fixture *f) {
  memset(f,0,sizeof(*f));
  f->shape[0]=1;f->shape[1]=2;f->shape[2]=256;
  f->request=(et_kernel_request_v1){sizeof(f->request),
    "e3.evaluation-metrics.finalize","f32","cpu",3,f->shape,1,{0}};
  f->input_data[0]=7.0f;f->input_data[1]=3.0f;f->input_data[2]=2.0f;
  for (size_t i=0;i<3;++i)
    f->inputs[i]=(et_kernel_tensor_view_v1){sizeof(f->inputs[i]),
      &f->input_data[i],sizeof(float),"f32","cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,0,NULL};
  for (size_t i=0;i<4;++i)
    f->outputs[i]=(et_kernel_tensor_view_v1){sizeof(f->outputs[i]),
      &f->output_data[i],sizeof(float),"f32","cpu",ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,0,0,NULL};
  memset(f->output_data,0x5a,sizeof(f->output_data));
  f->before_guard=UINT32_C(0x12345678);f->after_guard=UINT32_C(0xabcdef09);
  f->call=(et_kernel_call_v1){sizeof(f->call),"e3.evaluation-metrics",&f->request,
    3,sizeof(f->inputs[0]),sizeof(f->inputs),f->inputs,
    4,sizeof(f->outputs[0]),sizeof(f->outputs),f->outputs};
}
static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  CHECK(context==NULL);CHECK(strcmp(symbol,et_kernel_provider_symbol())==0);
  return et_e3_metrics_kernel_provider_v1();
}
static void run_fault(et_kernel_runtime *runtime, unsigned injection, int invalid) {
  fixture f,before;setup(&f);
  if (invalid==1) f.input_data[1]=0.0f;
  if (invalid==2) f.input_data[0]=FLT_MAX; /* expf fails after arithmetic. */
  memcpy(&before,&f,sizeof(f));
  et_kernel_error error;memset(&error,0xa5,sizeof(error));
  fenv_t environment;CHECK(__real_fegetenv(&environment)==0);
  fault=injection;saves=0;restores=0;errno=E2BIG;
  const int32_t rc=et_kernel_runtime_dispatch(runtime,&f.call,&error);
  const int observed_errno=errno;
  fault=0;
  CHECK(__real_fesetenv(&environment)==0);
  const uint32_t category=(injection&FAIL_SAVE)?ET_KERNEL_ERROR_UNSUPPORTED:
    ET_KERNEL_ERROR_INTERNAL;
  CHECK(rc==(int32_t)category);CHECK(error.category==category);
  CHECK(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(strcmp(error.operation,"e3.metrics.provider")==0);
  CHECK(error.message[0]!=0 && memchr(error.message,0,sizeof(error.message))!=NULL);
  CHECK(observed_errno==E2BIG);CHECK(saves==1);
  CHECK(restores==((injection&FAIL_SAVE)?0u:1u));
  CHECK(memcmp(&before,&f,sizeof(f))==0);
}
static void passthrough(et_kernel_runtime *runtime) {
  fixture f,before;setup(&f);memcpy(&before,&f,sizeof(f));
  const et_kernel_provider_v1 *p=et_e3_metrics_kernel_provider_v1();
  et_kernel_error error;memset(&error,0xa5,sizeof(error));
  fault=0;saves=0;restores=0;errno=EILSEQ;
  CHECK(p->validate_call(&f.call,&error)==0);
  CHECK(errno==EILSEQ);CHECK(saves==1 && restores==1);
  CHECK(error.category==ET_KERNEL_ERROR_NONE && error.code==ET_KERNEL_CODE_OK);
  CHECK(memcmp(&before,&f,sizeof(f))==0);
  CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&error)==0);
  CHECK(errno==EILSEQ);CHECK(saves==2 && restores==2);
  CHECK(f.output_data[0]==7.0f/3.0f && f.output_data[1]==3.0f);
  CHECK(f.output_data[2]>10.0f && f.output_data[2]<11.0f && f.output_data[3]==2.0f/3.0f);
  memcpy(before.output_data,f.output_data,sizeof(f.output_data));
  CHECK(memcmp(&before,&f,sizeof(f))==0);
  setup(&f);f.input_data[1]=0.0f;memcpy(&before,&f,sizeof(f));
  errno=EILSEQ;
  CHECK(et_kernel_runtime_dispatch(runtime,&f.call,&error)==ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(strcmp(error.operation,"e3.evaluation-metrics.finalize")==0);
  CHECK(errno==EILSEQ);CHECK(saves==3 && restores==3);
  CHECK(memcmp(&before,&f,sizeof(f))==0);
}
static void prefix_precedence(void) {
  const et_kernel_provider_v1 *p=et_e3_metrics_kernel_provider_v1();
  fixture f,before;setup(&f);f.call.struct_size=sizeof(f.call)-1;
  memcpy(&before,&f,sizeof(f));
  et_kernel_error error;
  fenv_t environment;CHECK(__real_fegetenv(&environment)==0);
  fault=FAIL_RESTORE;saves=0;restores=0;errno=E2BIG;
  CHECK(p->validate_call(&f.call,&error)==ET_KERNEL_ERROR_INTERNAL);
  CHECK(errno==E2BIG);CHECK(saves==1 && restores==1);
  CHECK(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(strcmp(error.operation,"e3.metrics.provider")==0);
  CHECK(memcmp(&before,&f,sizeof(f))==0);
  fault=FAIL_SAVE;saves=0;restores=0;errno=E2BIG;
  CHECK(p->validate_call(NULL,&error)==ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(errno==E2BIG);CHECK(saves==1 && restores==0);
  CHECK(error.code==ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(strcmp(error.operation,"e3.metrics.provider")==0);
  fault=0;CHECK(__real_fesetenv(&environment)==0);
}
int main(void) {
  fenv_t original;CHECK(__real_fegetenv(&original)==0);
  CHECK(__real_fesetenv(FE_DFL_ENV)==0);
  et_kernel_runtime *runtime=NULL;et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve,NULL,&runtime,&error)==0);
  passthrough(runtime);
  for (unsigned injection=1;injection<=3;++injection)
    for (int invalid=0;invalid<=2;++invalid) run_fault(runtime,injection,invalid);
  prefix_precedence();
  et_kernel_runtime_destroy(runtime);CHECK(__real_fesetenv(&original)==0);
  printf("E3-METRICS fenv faults: %zu checks; save/restore precedence and errno PASS\n",checks);
  return 0;
}
