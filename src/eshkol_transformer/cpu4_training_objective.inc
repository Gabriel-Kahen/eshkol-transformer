/* Included only by the fixed CPU4 transport with its private objective flag.
 * Eshkol owns model traversal; this bridge only invokes reviewed L2/L3S kernels.
 * All caller-visible writes follow complete numerical staging. */
#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"
typedef struct c4o_bytevector { int64_t length; unsigned char payload[4096]; } c4o_bytevector;
static int c4o_same(const char *left,const char *right) {
 return left && right && strcmp(left,right)==0;
}
static unsigned char *c4o_bytes(void *lease,void *header,size_t bytes) {
 size_t extent=8+bytes;
 if(!header || (uintptr_t)header>UINTPTR_MAX-extent) {
  c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;
 }
 if(c4t_cerror(et_g3c4_training_span_disjoint_internal(lease,header,extent)))return NULL;
 for(c4t_record *r=c4t_registry;r;r=r->next)
  if(c4t_overlap(header,extent,r,r->extent)) {
   c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;
  }
 if(et_f32_tensor_private_owned_span_overlap_v1(header,extent)) {
  c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);return NULL;
 }
 int64_t length;memcpy(&length,header,8);
 if(length!=(int64_t)bytes) {c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);return NULL;}
 return (unsigned char *)header+8;
}
int64_t et_cpu4_private_batch_copy_v1(void *lease,const et_kernel_tensor_view_v1 *view,
                                     void *bytes,int64_t kind) {
 c4t_clear();
 if(kind<0 || kind>2)return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SELECTOR);
 size_t width=kind==2?1:8, count=4*width;
 if(!view || view->struct_size<ET_KERNEL_TENSOR_VIEW_V1_0_SIZE || !view->shape ||
    view->rank!=2 || view->shape[0]!=1 || view->shape[1]!=4 || view->byte_length!=count ||
    !view->data || (uintptr_t)view->data>UINTPTR_MAX-count)
  return c4t_bad(ET_M3T_SHAPE_MISMATCH,ET_M3T_CODE_SHAPE);
 if(!c4o_same(view->dtype,kind==2?"bool":"i64"))
  return c4t_fail(1,ET_KERNEL_ERROR_DTYPE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);
 if(!c4o_same(view->device,"cpu"))return c4t_fail(1,ET_KERNEL_ERROR_DEVICE_MISMATCH,ET_KERNEL_CODE_INVALID_BUFFER);
 if(view->layout!=ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || view->offset_bytes ||
    (uintptr_t)view->data%width)return c4t_fail(1,ET_KERNEL_ERROR_NONCONTIGUOUS,ET_KERNEL_CODE_INVALID_BUFFER);
 unsigned char *payload=c4o_bytes(lease,bytes,count);if(!payload)return c4t_category;
 if(c4t_overlap(view->data,count,bytes,count+8) ||
    c4t_overlap(view,sizeof(*view),bytes,count+8) ||
    c4t_overlap(view->shape,2*sizeof(uint64_t),bytes,count+8))
  return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 for(size_t i=0;i<4;i++) {
  int64_t value;
  if(kind==2)value=((const unsigned char *)view->data)[i];
  else memcpy(&value,(const unsigned char *)view->data+8*i,8);
  if(value<0 || value>(kind==2?1:255))return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_SHAPE);
 }
 memcpy(payload,view->data,count);return 0;
}
static const et_kernel_provider_v1 *c4o_resolver(void *context,
                                             const char *symbol) {
  return c4o_same(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1)
             ? (const et_kernel_provider_v1 *)context
             : NULL;
}

static int64_t c4o_kernel_status(int32_t status, const et_kernel_error *error) {
 return c4t_kerror(status,error);
}

static et_kernel_tensor_view_v1 c4o_tensor(void *data, size_t bytes,
                                      const char *dtype, size_t rank,
                                      const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(result),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = rank,
      .shape = shape,
  };
  return result;
}

static et_kernel_call_v1 c4o_call(const char *capability,
                              const et_kernel_request_v1 *request,
                              size_t input_count,
                              const et_kernel_tensor_view_v1 *inputs,
                              size_t output_count,
                              et_kernel_tensor_view_v1 *outputs) {
  et_kernel_call_v1 result = {
      .struct_size = sizeof(result),
      .capability = capability,
      .request = request,
      .input_count = input_count,
      .input_stride = sizeof(*inputs),
      .input_bytes = input_count * sizeof(*inputs),
      .inputs = inputs,
      .output_count = output_count,
      .output_stride = sizeof(*outputs),
      .output_bytes = output_count * sizeof(*outputs),
      .outputs = outputs,
  };
  return result;
}

static int64_t c4o_dispatch(const et_kernel_runtime *runtime,
                        const et_kernel_call_v1 *operation) {
  et_kernel_error error;
  return c4o_kernel_status(et_kernel_runtime_dispatch(runtime, operation, &error),
                       &error);
}

static void c4o_put_u32_le(unsigned char *destination, uint32_t value) {
  for (size_t index = 0; index < 4u; ++index)
    destination[index] = (unsigned char)(value >> (8u * index));
}

int64_t et_cpu4_private_objective_prepare_v1(void *logits,void *seed,void *targets,
                                            void *mask,void *observation) {
 c4t_clear();
 c4t_record *source=c4t_admit(logits,C4T_LOGITS,0);if(!source)return c4t_category;
 c4t_record *destination=c4t_admit(seed,C4T_LOGITS,0);if(!destination)return c4t_category;
 if(source==destination || source->lease!=destination->lease)
  return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 unsigned char *target_payload=c4o_bytes(source->lease,targets,32);if(!target_payload)return c4t_category;
 unsigned char *mask_payload=c4o_bytes(source->lease,mask,4);if(!mask_payload)return c4t_category;
 unsigned char *observation_payload=c4o_bytes(source->lease,observation,12);if(!observation_payload)return c4t_category;
 if(c4t_overlap(targets,40,mask,12) || c4t_overlap(targets,40,observation,20) ||
    c4t_overlap(mask,12,observation,20))
  return c4t_bad(ET_M3T_INVALID_ARGUMENT,ET_M3T_CODE_IDENTITY);
 int64_t status=0;
  c4o_bytevector logits_header = {.length = 4096};
  c4o_bytevector gradient_header = {.length = 4096};
  status = et_cpu4_private_logits_copy_to_v1(logits, &logits_header, 4096);
  if (status != 0) return status;

  float losses[4] = {0.0f, 0.0f};
  float reduction[3] = {0.0f, 0.0f, 0.0f};
  float upstream[4] = {0.0f, 0.0f};
  const uint64_t shape3[3] = {1u, 4u, 256u};
  const uint64_t shape2[2] = {1u, 4u};

  et_kernel_runtime *l2 = NULL;
  et_kernel_runtime *l3s = NULL;
  et_kernel_error error;
  status = c4o_kernel_status(et_kernel_runtime_discover(
      c4o_resolver, (void *)et_l2_indexed_cross_entropy_provider_v1(), &l2,
      &error), &error);
  if (status != 0) goto cleanup;
  status = c4o_kernel_status(et_kernel_runtime_discover(
      c4o_resolver, (void *)et_l3s_kernel_provider_v1(), &l3s, &error), &error);
  if (status != 0) goto cleanup;

  et_kernel_request_v1 l2_forward_request = {
      sizeof(l2_forward_request), ET_L2_INDEXED_CROSS_ENTROPY_FORWARD,
      "f32", "cpu", 3u, shape3, 1u, {0}};
  et_kernel_tensor_view_v1 l2_forward_inputs[2] = {
      c4o_tensor(logits_header.payload, sizeof(logits_header.payload), "f32", 3u,
             shape3),
      c4o_tensor(target_payload, 32, "i64", 2u, shape2)};
  et_kernel_tensor_view_v1 l2_forward_outputs[1] = {
      c4o_tensor(losses, sizeof(losses), "f32", 2u, shape2)};
  et_kernel_call_v1 l2_forward = c4o_call(
      ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY, &l2_forward_request, 2u,
      l2_forward_inputs, 1u, l2_forward_outputs);
  status = c4o_dispatch(l2, &l2_forward);
  if (status != 0) goto cleanup;

  et_kernel_request_v1 reduce_request = {
      sizeof(reduce_request), "l3s.masked-objective.reduce.bool", "f32",
      "cpu", 2u, shape2, 1u, {0}};
  et_kernel_tensor_view_v1 reduce_inputs[2] = {
      c4o_tensor(losses, sizeof(losses), "f32", 2u, shape2),
      c4o_tensor(mask_payload, 4, "bool", 2u, shape2)};
  et_kernel_tensor_view_v1 reduce_outputs[3] = {
      c4o_tensor(&reduction[0], sizeof(float), "f32", 0u, NULL),
      c4o_tensor(&reduction[1], sizeof(float), "f32", 0u, NULL),
      c4o_tensor(&reduction[2], sizeof(float), "f32", 0u, NULL)};
  et_kernel_call_v1 reduce = c4o_call("l3s.masked-objective", &reduce_request, 2u,
                                  reduce_inputs, 3u, reduce_outputs);
  status = c4o_dispatch(l3s, &reduce);
  if (status != 0) goto cleanup;

  et_kernel_request_v1 numerator_request = {
      sizeof(numerator_request), "l3s.masked-objective.numerator-seed.bool",
      "f32", "cpu", 2u, shape2, 1u, {0}};
  et_kernel_tensor_view_v1 numerator_inputs[1] = {
      c4o_tensor(mask_payload, 4, "bool", 2u, shape2)};
  et_kernel_tensor_view_v1 numerator_outputs[1] = {
      c4o_tensor(upstream, sizeof(upstream), "f32", 2u, shape2)};
  et_kernel_call_v1 numerator = c4o_call(
      "l3s.masked-objective", &numerator_request, 1u, numerator_inputs, 1u,
      numerator_outputs);
  status = c4o_dispatch(l3s, &numerator);
  if (status != 0) goto cleanup;

  et_kernel_request_v1 l2_backward_request = {
      sizeof(l2_backward_request), ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD,
      "f32", "cpu", 3u, shape3, 1u, {0}};
  et_kernel_tensor_view_v1 l2_backward_inputs[3] = {
      c4o_tensor(logits_header.payload, sizeof(logits_header.payload), "f32", 3u,
             shape3),
      c4o_tensor(target_payload, 32, "i64", 2u, shape2),
      c4o_tensor(upstream, sizeof(upstream), "f32", 2u, shape2)};
  et_kernel_tensor_view_v1 l2_backward_outputs[1] = {
      c4o_tensor(gradient_header.payload, sizeof(gradient_header.payload), "f32",
             3u, shape3)};
  et_kernel_call_v1 l2_backward = c4o_call(
      ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY, &l2_backward_request, 3u,
      l2_backward_inputs, 1u, l2_backward_outputs);
  status = c4o_dispatch(l2, &l2_backward);
  if (status != 0) goto cleanup;

cleanup:
 et_kernel_runtime_destroy(l3s);et_kernel_runtime_destroy(l2);
 if(status)return status;
 status=et_cpu4_private_logits_copy_from_v1(seed,&gradient_header,4096);
 if(status)return status;
 for(size_t i=0;i<3;i++) {uint32_t bits;memcpy(&bits,&reduction[i],4);c4o_put_u32_le(observation_payload+4*i,bits);}
 return 0;
}
