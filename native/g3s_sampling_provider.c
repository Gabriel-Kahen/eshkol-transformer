#include "eshkol_transformer/g3s_sampling_abi.h"
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if !defined(__x86_64__)
#error "G3-S v1 requires the reviewed x86-64 binary32 lane"
#endif
#include <xmmintrin.h>
#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4, "G3-S requires eight-bit bytes/binary32");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
               FLT_EVAL_METHOD == 0, "G3-S requires standard IEEE binary32 evaluation");
/* Private structural, signed-word, Philox and control helpers adapted from
 * native/n3k_primitives_provider.c at 4353bd2. No predecessor is linked here. */
static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int pointer_span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int aligned_pointer(const void *pointer, size_t alignment) {
  return pointer != NULL && (uintptr_t)pointer % alignment == 0u;
}

static int ranges_overlap(const void *left, size_t left_bytes,
                          const void *right, size_t right_bytes) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left_bytes == 0u || right_bytes == 0u ||
      left_start > UINTPTR_MAX - left_bytes ||
      right_start > UINTPTR_MAX - right_bytes) {
    return left_bytes != 0u && right_bytes != 0u;
  }
  return left_start < right_start + right_bytes &&
         right_start < left_start + left_bytes;
}

static const et_kernel_tensor_view_v1 *input_at(
    const et_kernel_call_v1 *call, size_t index) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)call->inputs +
                                             index * call->input_stride);
}

static et_kernel_tensor_view_v1 *output_at(const et_kernel_call_v1 *call,
                                           size_t index) {
  return (et_kernel_tensor_view_v1 *)((unsigned char *)call->outputs +
                                      index * call->output_stride);
}

static int32_t set_error(et_kernel_error *error,
                         et_kernel_error_category category,
                         et_kernel_error_code code, const char *operation,
                         const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int checked_product(size_t *result, size_t left, size_t right) {
  if (right != 0u && left > SIZE_MAX / right) {
    return 0;
  }
  *result = left * right;
  return 1;
}

static int32_t table_views(const et_kernel_call_v1 *call, int output,
                           size_t expected, const char *operation,
                           const et_kernel_tensor_view_v1 **views,
                           et_kernel_error *error) {
  const size_t count = output ? call->output_count : call->input_count;
  const size_t stride = output ? call->output_stride : call->input_stride;
  const size_t bytes = output ? call->output_bytes : call->input_bytes;
  const void *base = output ? call->outputs : call->inputs;
  if (count != expected) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     output ? "output count does not match operation schema"
                            : "input count does not match operation schema");
  }
  if (base == NULL) return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_NULL_ARGUMENT, operation, "tensor table is required");
  if (stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                     "tensor table stride truncates the K1 v1 prefix");
  }
  if (stride % _Alignof(et_kernel_tensor_view_v1) != 0u ||
      !aligned_pointer(base, _Alignof(et_kernel_tensor_view_v1)) ||
      bytes == 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor table span or alignment is invalid");
  }
  if (expected > SIZE_MAX / stride || !pointer_span_fits(base, bytes))
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INTEGER_OVERFLOW, operation, "tensor table span overflows");
  if (bytes != expected * stride) return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
    ET_KERNEL_CODE_INVALID_BUFFER, operation, "tensor table byte count differs");
  for (size_t index = 0; index < expected; index++) {
    const et_kernel_tensor_view_v1 *view =
        (const et_kernel_tensor_view_v1 *)((const unsigned char *)base +
                                            index * stride);
    if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
        view->struct_size > stride) {
      return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                       ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                       "tensor descriptor does not fit its table stride");
    }
    views[index] = view;
  }
  return 0;
}

static int32_t validate_view(const et_kernel_tensor_view_v1 *view,
                             const char *dtype, size_t element_size,
                             size_t alignment, size_t rank,
                             const size_t *shape, const char *operation,
                             et_kernel_error *error) {
  size_t elements = 1u;
  size_t expected_bytes;
  if (!exact_text(view->dtype, dtype)) {
    return set_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, operation,
                     "tensor dtype does not match operation schema");
  }
  if (!exact_text(view->device, "cpu")) {
    return set_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, operation,
                     "G3-S v1 accepts only CPU tensor views");
  }
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "G3-S v1 requires dense zero-offset tensor views");
  }
  if (view->rank != rank || (rank != 0u && view->shape == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, operation,
                     "tensor rank does not match operation schema");
  }
  if (rank != 0u &&
      (!aligned_pointer(view->shape, _Alignof(uint64_t)) ||
       rank > SIZE_MAX / sizeof(*view->shape) ||
       !pointer_span_fits(view->shape, rank * sizeof(*view->shape)))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor shape storage is invalid");
  }
  for (size_t index = 0; index < rank; index++) {
    if (view->shape[index] != (uint64_t)shape[index] ||
        !checked_product(&elements, elements, shape[index])) {
      return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                       ET_KERNEL_CODE_INVALID_SHAPE, operation,
                       "tensor extent does not match operation schema");
    }
  }
  if (!checked_product(&expected_bytes, elements, element_size) ||
      view->byte_length != expected_bytes ||
      !aligned_pointer(view->data, alignment)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "tensor data alignment or byte span is invalid");
  }
  if (!pointer_span_fits(view->data, expected_bytes))
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_PROVIDER_REJECTED, operation, "tensor data address wraps");
  return 0;
}

static uint64_t decode_signed_word(int64_t value) {
  if (value >= 0) return (uint64_t)value;
  return UINT64_MAX - (uint64_t)(-(value + INT64_C(1)));
}

static int64_t encode_signed_word(uint64_t value) {
  if (value <= (uint64_t)INT64_MAX) return (int64_t)value;
  return -(int64_t)(UINT64_MAX - value) - INT64_C(1);
}

static void philox4x32_10(uint64_t key, uint64_t counter_low,
                          uint64_t counter_high, uint32_t output[4]) {
  uint32_t c0 = (uint32_t)counter_low;
  uint32_t c1 = (uint32_t)(counter_low >> 32);
  uint32_t c2 = (uint32_t)counter_high;
  uint32_t c3 = (uint32_t)(counter_high >> 32);
  uint32_t k0 = (uint32_t)key;
  uint32_t k1 = (uint32_t)(key >> 32);
  for (size_t round = 0; round < 10u; round++) {
    const uint64_t product0 = UINT64_C(0xd2511f53) * (uint64_t)c0;
    const uint64_t product1 = UINT64_C(0xcd9e8d57) * (uint64_t)c2;
    const uint32_t next0 = (uint32_t)(product1 >> 32) ^ c1 ^ k0;
    const uint32_t next1 = (uint32_t)product1;
    const uint32_t next2 = (uint32_t)(product0 >> 32) ^ c3 ^ k1;
    const uint32_t next3 = (uint32_t)product0;
    c0 = next0;
    c1 = next1;
    c2 = next2;
    c3 = next3;
    if (round != 9u) {
      k0 += UINT32_C(0x9e3779b9);
      k1 += UINT32_C(0xbb67ae85);
    }
  }
  output[0] = c0;
  output[1] = c1;
  output[2] = c2;
  output[3] = c3;
}

/* Controls are checked before any floating value (including sNaN) is read. */
static int fp_environment_supported(void) {
  unsigned short control;
  __asm__ volatile("fnstcw %0" : "=m"(control));
  const unsigned int mxcsr = _mm_getcsr();
  return (control & 0x003fu) == 0x003fu && (control & 0x0c00u) == 0u &&
         (mxcsr & 0x00001f80u) == 0x00001f80u &&
         (mxcsr & 0x00006040u) == 0u && (mxcsr & 0x00008000u) == 0u;
}

typedef struct sampling_scratch {
  float a[256], b[256];
  size_t order[256], count;
  float weight;
} sampling_scratch;

/* Every arithmetic statement rounds once to binary32. Insertion sort is bounded,
 * stable by ID, allocation-free and compares rounded probabilities only. */
static int distribution(const float *logits, float temperature, size_t k,
                        float p, sampling_scratch *s) {
  float maximum=logits[0];
  for (size_t i=0;i<256;++i) {
    if (!isfinite(logits[i])) return 0;
    if (logits[i]>maximum) maximum=logits[i];
  }
  float sum=0.0f;
  for (size_t i=0;i<256;++i) {
    const float difference=logits[i]-maximum;
    if (!isfinite(difference)) return 0;
    const float scaled=difference/temperature;
    if (!isfinite(scaled)) return 0;
    s->a[i]=expf(scaled);
    if (!isfinite(s->a[i])) return 0;
    sum=sum+s->a[i];
    if (!isfinite(sum)) return 0;
  }
  for (size_t i=0;i<256;++i) {
    s->a[i]=s->a[i]/sum;
    if (!isfinite(s->a[i])) return 0;
    s->order[i]=i;
    s->b[i]=0.0f;
  }
  for (size_t i=1;i<256;++i) {
    const size_t id=s->order[i];
    size_t j=i;
    while (j>0 && (s->a[id]>s->a[s->order[j-1]] ||
           (s->a[id]==s->a[s->order[j-1]] && id<s->order[j-1]))) {
      s->order[j]=s->order[j-1]; --j;
    }
    s->order[j]=id;
  }
  sum=0.0f;
  for (size_t i=0;i<k;++i) {
    sum=sum+s->a[s->order[i]];
    if (!isfinite(sum)) return 0;
  }
  for (size_t i=0;i<k;++i) {
    s->b[i]=s->a[s->order[i]]/sum;
    if (!isfinite(s->b[i])) return 0;
  }
  float cumulative=0.0f;
  s->count=k;
  for (size_t i=0;i<k;++i) {
    cumulative=cumulative+s->b[i];
    if (!isfinite(cumulative)) return 0;
    if (cumulative>=p) { s->count=i+1; break; }
  }
  s->weight=0.0f;
  for (size_t i=0;i<s->count;++i) {
    s->weight=s->weight+s->b[i];
    if (!isfinite(s->weight)) return 0;
  }
  return s->weight>0.0f;
}
static int64_t select_token(const sampling_scratch *s,float u) {
  const float r=u*s->weight;
  float cdf=0.0f;
  int64_t last=0;
  for (size_t i=0;i<s->count;++i) {
    if (s->b[i]>0.0f) last=(int64_t)s->order[i];
    cdf=cdf+s->b[i];
    if (cdf>r) return (int64_t)s->order[i];
  }
  return last;
}
static int categorical(const et_kernel_call_v1 *call) {
  return exact_text(call->request->operation,"g3s.categorical.forward");
}
/* Complete local candidate computation precedes both publication stores. */
static int run_operation(const et_kernel_call_v1 *call,int cat,
                         int64_t *token,int64_t next[4]) {
  const float *logits=input_at(call,0)->data;
  const int64_t *state=input_at(call,cat?4u:1u)->data;
  memcpy(next,state,4*sizeof(int64_t));
  if (!cat) {
    *token=0;
    for (size_t i=0;i<256;++i) {
      if (!isfinite(logits[i])) return 0;
      if (logits[i]>logits[(size_t)*token]) *token=(int64_t)i;
    }
    return 1;
  }
  sampling_scratch scratch;
  if (!distribution(logits,*(const float *)input_at(call,1)->data,
      (size_t)*(const int64_t *)input_at(call,2)->data,
      *(const float *)input_at(call,3)->data,&scratch)) return 0;
  const uint64_t low=decode_signed_word(state[2]);
  const uint64_t high=decode_signed_word(state[3]);
  uint32_t lanes[4];
  philox4x32_10((uint64_t)state[1]^UINT64_C(0x4733434154454731),low,high,lanes);
  const float u=(float)(lanes[0]>>8)*0x1p-24f;
  *token=select_token(&scratch,u);
  const uint64_t successor=low+UINT64_C(1);
  next[2]=encode_signed_word(successor);
  next[3]=encode_signed_word(high+(uint64_t)(successor==0));
  return 1;
}
static int32_t validate_inner(const et_kernel_call_v1 *call,et_kernel_error *error) {
  const char *op="g3s.provider";
#define FAIL(category,code,message) return set_error(error,ET_KERNEL_ERROR_##category,ET_KERNEL_CODE_##code,op,message)
  if (!call) { FAIL(INVALID_ARGUMENT,NULL_ARGUMENT,"call is required"); }
  if (!aligned_pointer(call,_Alignof(et_kernel_call_v1))) { FAIL(INVALID_ARGUMENT,INVALID_BUFFER,"call is misaligned"); }
  if (call->struct_size<ET_KERNEL_CALL_V1_0_SIZE) { FAIL(VERSION_MISMATCH,INVALID_STRUCT_SIZE,"call truncates K1 prefix"); }
  const et_kernel_request_v1 *r=call->request;
  if (!r) { FAIL(INVALID_ARGUMENT,NULL_ARGUMENT,"request is required"); }
  if (!aligned_pointer(r,_Alignof(et_kernel_request_v1))) { FAIL(INVALID_ARGUMENT,INVALID_BUFFER,"request is misaligned"); }
  if (r->struct_size<ET_KERNEL_REQUEST_V1_0_SIZE) { FAIL(VERSION_MISMATCH,INVALID_STRUCT_SIZE,"request truncates K1 prefix"); }
  if (!r->operation) { FAIL(INVALID_ARGUMENT,NULL_ARGUMENT,"operation is required"); }
  /* Unknown G3-S operations keep their attempted name; foreign text cannot
   * impersonate another provider in this provider-local diagnostic. */
  static const char prefix[]="g3s.";
  size_t prefix_length=0;
  while (prefix_length<sizeof(prefix)-1 &&
         r->operation[prefix_length]==prefix[prefix_length]) ++prefix_length;
  if (prefix_length==sizeof(prefix)-1) op=r->operation;
  const int cat=categorical(call);
  if ((!cat && !exact_text(op,"g3s.greedy.forward")) ||
      !exact_text(call->capability,cat?"g3s.categorical":"g3s.greedy") ||
      !exact_text(r->dtype,"f32") || !exact_text(r->device,"cpu") || r->rank!=2 || r->deterministic>1) {
    FAIL(UNSUPPORTED,PROVIDER_REJECTED,"request is outside exact G3-S evidence");
  }
  for (size_t i=0;i<sizeof(r->reserved);++i) if (r->reserved[i]) { FAIL(INVALID_ARGUMENT,INVALID_BUFFER,"request reserved bytes must be zero"); }
  if (!r->shape) { FAIL(SHAPE_MISMATCH,INVALID_SHAPE,"request shape is required"); }
  if (!aligned_pointer(r->shape,_Alignof(uint64_t)) || !pointer_span_fits(r->shape,2*sizeof(uint64_t))) { FAIL(INVALID_ARGUMENT,INVALID_BUFFER,"request shape span is invalid"); }
  if (r->shape[0]!=1 || r->shape[1]!=256) { FAIL(UNSUPPORTED,PROVIDER_REJECTED,"request row is outside exact G3-S evidence"); }
  const size_t ni=cat?5u:2u;
  const et_kernel_tensor_view_v1 *in[5],*out[2];
  int32_t rc=table_views(call,0,ni,op,in,error);
  if (!rc) rc=table_views(call,1,2,op,out,error);
  if (rc) return rc;
  const size_t logits_shape[2]={1,256},state_shape[1]={4},token_shape[1]={1};
  for (size_t i=0;i<ni+2;++i) {
    const int output=i>=ni;
    const size_t index=output?i-ni:i;
    const int f=!output && (index==0 || (cat && (index==1 || index==3)));
    const size_t rank=output?1u:(index==0?2u:(index==(cat?4u:1u)?1u:0u));
    const size_t *shape=output?(index==0?token_shape:state_shape):(index==0?logits_shape:state_shape);
    rc=validate_view(output?out[index]:in[index],f?"f32":"i64",f?4u:8u,f?_Alignof(float):_Alignof(int64_t),rank,shape,op,error);
    if (rc) return rc;
  }
  for (size_t i=0;i<2;++i) {
    for (size_t j=0;j<ni;++j) if (ranges_overlap(out[i]->data,out[i]->byte_length,in[j]->data,in[j]->byte_length)) { FAIL(INVALID_ARGUMENT,ALIASING_OUTPUT,"output overlaps input"); }
    if (i && ranges_overlap(out[0]->data,out[0]->byte_length,out[i]->data,out[i]->byte_length)) { FAIL(INVALID_ARGUMENT,ALIASING_OUTPUT,"outputs overlap"); }
  }
  for (size_t i=0;i<ni;++i) for (size_t j=i+1;j<ni;++j)
    if (ranges_overlap(in[i]->data,in[i]->byte_length,in[j]->data,in[j]->byte_length)) { FAIL(INVALID_ARGUMENT,PROVIDER_REJECTED,"inputs overlap"); }
  if (!fp_environment_supported()) { FAIL(UNSUPPORTED,PROVIDER_REJECTED,"unsupported x87 or MXCSR controls"); }
  const int64_t *state=in[cat?4u:1u]->data;
  if (state[0]!=1) { FAIL(VERSION_MISMATCH,PROVIDER_REJECTED,"unknown RNG version"); }
  if (state[1]<0) { FAIL(INVALID_ARGUMENT,PROVIDER_REJECTED,"negative RNG seed"); }
  /* Required draw admission precedes even floating payload classification. */
  if (cat && state[2]==-1 && state[3]==-1) { FAIL(INVALID_ARGUMENT,PROVIDER_REJECTED,"categorical RNG exhausted"); }
  if (cat) {
    const float temperature=*(const float *)in[1]->data,p=*(const float *)in[3]->data;
    const int64_t k=*(const int64_t *)in[2]->data;
    if (!isfinite(temperature) || temperature<=0.0f || !isfinite(p) || p<=0.0f || p>1.0f || k<1 || k>256) { FAIL(INVALID_ARGUMENT,PROVIDER_REJECTED,"invalid categorical policy"); }
  }
  int64_t token,next[4];
  if (!run_operation(call,cat,&token,next)) { FAIL(INVALID_ARGUMENT,PROVIDER_REJECTED,"nonfinite input or prospective arithmetic"); }
  et_kernel_error_clear(error);
  return 0;
#undef FAIL
}
static int32_t provider_validate(const et_kernel_call_v1 *call,et_kernel_error *error) {
  fenv_t environment;
  if (fegetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_UNSUPPORTED,
    ET_KERNEL_CODE_PROVIDER_REJECTED,"g3s.provider","cannot observe floating environment");
  const int32_t rc=validate_inner(call,error);
  if (fesetenv(&environment)!=0) return set_error(error,ET_KERNEL_ERROR_INTERNAL,
    ET_KERNEL_CODE_PROVIDER_REJECTED,"g3s.provider","cannot restore floating environment");
  return rc;
}
static void provider_invoke(const et_kernel_call_v1 *call) {
  int64_t token,next[4];
  /* Complete preflight and stable inputs/controls make this recomputation nonfailing. */
  (void)run_operation(call,categorical(call),&token,next);
  memcpy(output_at(call,0)->data,&token,sizeof(token));
  memcpy(output_at(call,1)->data,next,sizeof(next));
}
static const et_kernel_dimension_range_v1 dimensions[]={{1,1,0,{0}},{256,256,0,{0}}};
static const et_kernel_shape_range_v1 rows[]={{2,dimensions}};
static const char *const greedy_ops[]={"g3s.greedy.forward"};
static const char *const categorical_ops[]={"g3s.categorical.forward"};
static const char *const dtypes[]={"f32"},*const devices[]={"cpu"};
static const et_kernel_capability_v1 capabilities[]={
  {sizeof(et_kernel_capability_v1),"g3s.categorical",ET_KERNEL_CAPABILITY_VERIFIED,
   "g3s.cpu-f32.serial","1.0","G3-S:cpu-f32-v256-v1",1,{0},1,categorical_ops,1,dtypes,1,devices,1,rows},
  {sizeof(et_kernel_capability_v1),"g3s.greedy",ET_KERNEL_CAPABILITY_VERIFIED,
   "g3s.cpu-f32.serial","1.0","G3-S:cpu-f32-v256-v1",1,{0},1,greedy_ops,1,dtypes,1,devices,1,rows}
};
static const et_kernel_provider_v1 provider={
  sizeof(et_kernel_provider_v1),ET_KERNEL_ABI_MAJOR,ET_KERNEL_ABI_MINOR,0,
  "g3s.cpu-f32.serial","1.0","G3-S:cpu-f32-v256-v1",COUNT(capabilities),
  sizeof(et_kernel_capability_v1),sizeof(capabilities),capabilities,provider_validate,provider_invoke
};
const et_kernel_provider_v1 *et_g3s_kernel_provider_v1(void) { return &provider; }
