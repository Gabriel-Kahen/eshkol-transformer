#include "tr3b_objective_bridge.h"

#include "eshkol_transformer/indexed_cross_entropy.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"
#include "../src/eshkol_transformer/m3_model.h"
#include "../src/eshkol_transformer/m3t_transport.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum {
  TR3B_ROWS = 2,
  TR3B_VOCAB = 256,
  TR3B_LOGIT_COUNT = TR3B_ROWS * TR3B_VOCAB,
  TR3B_LOGIT_BYTES = TR3B_LOGIT_COUNT * (int)sizeof(float),
  TR3B_TARGET_BYTES = TR3B_ROWS * (int)sizeof(int64_t),
  TR3B_MASK_BYTES = TR3B_ROWS,
  TR3B_OBSERVATION_BYTES = 3 * (int)sizeof(float)
};

typedef struct tr3b_bytevector_2048 {
  int64_t length;
  unsigned char payload[TR3B_LOGIT_BYTES];
} tr3b_bytevector_2048;

static int span(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int overlaps(const void *left, size_t left_bytes, const void *right,
                    size_t right_bytes) {
  if (left_bytes == 0u || right_bytes == 0u) return 0;
  if (!span(left, left_bytes) || !span(right, right_bytes)) return 1;
  return (uintptr_t)left < (uintptr_t)right + right_bytes &&
         (uintptr_t)right < (uintptr_t)left + left_bytes;
}

static int64_t bytevector(void *header, int64_t expected,
                          unsigned char **payload) {
  int64_t encoded = -1;
  if (payload == NULL || expected < 0 || header == NULL ||
      !span(header, sizeof(encoded) + (size_t)expected))
    return ET_TR3B_INVALID_ARGUMENT;
  memcpy(&encoded, header, sizeof(encoded));
  if (encoded != expected) return ET_TR3B_SHAPE_MISMATCH;
  *payload = (unsigned char *)header + sizeof(encoded);
  return ET_TR3B_OK;
}

static int same(const char *left, const char *right) {
  return left != NULL && right != NULL && strcmp(left, right) == 0;
}

static int64_t view(const et_kernel_tensor_view_v1 *candidate,
                    const char *dtype, size_t width) {
  if (candidate == NULL ||
      candidate->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      candidate->shape == NULL || !span(candidate->shape, 2u * sizeof(uint64_t)) ||
      candidate->rank != 2u || candidate->shape[0] != 1u ||
      candidate->shape[1] != 2u || candidate->byte_length != 2u * width ||
      !span(candidate->data, candidate->byte_length))
    return ET_TR3B_SHAPE_MISMATCH;
  if (!same(candidate->dtype, dtype)) return ET_TR3B_DTYPE_MISMATCH;
  if (!same(candidate->device, "cpu")) return ET_TR3B_DEVICE_MISMATCH;
  if (candidate->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      candidate->offset_bytes != 0u ||
      (uintptr_t)candidate->data % width != 0u)
    return ET_TR3B_NONCONTIGUOUS;
  return ET_TR3B_OK;
}

static int64_t m3t_status(int64_t status) {
  if (status == ET_M3T_OK) return ET_TR3B_OK;
  switch (et_m3t_private_last_error_category_v1()) {
    case ET_M3T_INVALID_ARGUMENT: return ET_TR3B_INVALID_ARGUMENT;
    case ET_M3T_INVALID_STATE: return ET_TR3B_INVALID_STATE;
    case ET_M3T_SHAPE_MISMATCH: return ET_TR3B_SHAPE_MISMATCH;
    case ET_M3T_UNSUPPORTED: return ET_TR3B_UNSUPPORTED;
    default: return ET_TR3B_INTERNAL;
  }
}

int64_t et_tr3b_private_stage_input_v1(
    void *m3t_input, const et_kernel_tensor_view_v1 *d2_input) {
  int64_t status = view(d2_input, "i64", sizeof(int64_t));
  if (status != ET_TR3B_OK) return status;
  const int64_t *ids = (const int64_t *)d2_input->data;
  if (ids[0] < 0 || ids[0] >= TR3B_VOCAB ||
      ids[1] < 0 || ids[1] >= TR3B_VOCAB)
    return ET_TR3B_SHAPE_MISMATCH;
  return m3t_status(et_m3t_private_input_copy_v1(m3t_input, ids[0], ids[1]));
}

static int64_t copy_view(const et_kernel_tensor_view_v1 *source,
                         const char *dtype, size_t width, void *destination,
                         int64_t destination_bytes) {
  unsigned char *payload = NULL;
  int64_t status = view(source, dtype, width);
  if (status != ET_TR3B_OK) return status;
  status = bytevector(destination, destination_bytes, &payload);
  if (status != ET_TR3B_OK) return status;
  if (overlaps(source->data, source->byte_length, payload,
               (size_t)destination_bytes))
    return ET_TR3B_INVALID_ARGUMENT;
  memcpy(payload, source->data, (size_t)destination_bytes);
  return ET_TR3B_OK;
}

int64_t et_tr3b_private_copy_targets_v1(
    const et_kernel_tensor_view_v1 *d2_targets, void *target_bytes) {
  return copy_view(d2_targets, "i64", sizeof(int64_t), target_bytes,
                   TR3B_TARGET_BYTES);
}

int64_t et_tr3b_private_copy_mask_v1(
    const et_kernel_tensor_view_v1 *d2_mask, void *mask_bytes) {
  return copy_view(d2_mask, "bool", sizeof(unsigned char), mask_bytes,
                   TR3B_MASK_BYTES);
}

static const et_kernel_provider_v1 *resolver(void *context,
                                             const char *symbol) {
  return same(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1)
             ? (const et_kernel_provider_v1 *)context
             : NULL;
}

static int64_t kernel_status(int32_t status, const et_kernel_error *error) {
  if (status == 0) return ET_TR3B_OK;
  if (error == NULL) return ET_TR3B_INTERNAL;
  switch (error->category) {
    case ET_KERNEL_ERROR_INVALID_ARGUMENT: return ET_TR3B_INVALID_ARGUMENT;
    case ET_KERNEL_ERROR_SHAPE_MISMATCH: return ET_TR3B_SHAPE_MISMATCH;
    case ET_KERNEL_ERROR_DTYPE_MISMATCH: return ET_TR3B_DTYPE_MISMATCH;
    case ET_KERNEL_ERROR_DEVICE_MISMATCH: return ET_TR3B_DEVICE_MISMATCH;
    case ET_KERNEL_ERROR_NONCONTIGUOUS: return ET_TR3B_NONCONTIGUOUS;
    case ET_KERNEL_ERROR_UNSUPPORTED: return ET_TR3B_UNSUPPORTED;
    case ET_KERNEL_ERROR_VERSION_MISMATCH: return ET_TR3B_VERSION_MISMATCH;
    default: return ET_TR3B_INTERNAL;
  }
}

static et_kernel_tensor_view_v1 tensor(void *data, size_t bytes,
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

static et_kernel_call_v1 call(const char *capability,
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

static int64_t dispatch(const et_kernel_runtime *runtime,
                        const et_kernel_call_v1 *operation) {
  et_kernel_error error;
  return kernel_status(et_kernel_runtime_dispatch(runtime, operation, &error),
                       &error);
}

static void put_u32_le(unsigned char *destination, uint32_t value) {
  for (size_t index = 0; index < 4u; ++index)
    destination[index] = (unsigned char)(value >> (8u * index));
}

int64_t et_tr3b_private_objective_prepare_v1(
    void *m3_graph_logits, void *m3t_seed, void *target_bytes,
    void *mask_bytes, void *observation_bytes) {
  unsigned char *target_payload = NULL;
  unsigned char *mask_payload = NULL;
  unsigned char *observation_payload = NULL;
  int64_t status = bytevector(target_bytes, TR3B_TARGET_BYTES, &target_payload);
  if (status != ET_TR3B_OK) return status;
  status = bytevector(mask_bytes, TR3B_MASK_BYTES, &mask_payload);
  if (status != ET_TR3B_OK) return status;
  status = bytevector(observation_bytes, TR3B_OBSERVATION_BYTES,
                      &observation_payload);
  if (status != ET_TR3B_OK) return status;
  if (overlaps(target_payload, TR3B_TARGET_BYTES, mask_payload, TR3B_MASK_BYTES) ||
      overlaps(target_payload, TR3B_TARGET_BYTES, observation_payload,
               TR3B_OBSERVATION_BYTES) ||
      overlaps(mask_payload, TR3B_MASK_BYTES, observation_payload,
               TR3B_OBSERVATION_BYTES))
    return ET_TR3B_INVALID_ARGUMENT;

  tr3b_bytevector_2048 logits_header = {.length = TR3B_LOGIT_BYTES};
  tr3b_bytevector_2048 gradient_header = {.length = TR3B_LOGIT_BYTES};
  status = m3t_status(et_m3_private_model_logits_copy_to_v1(
      m3_graph_logits, &logits_header, TR3B_LOGIT_BYTES));
  if (status != ET_TR3B_OK) return status;

  float losses[TR3B_ROWS] = {0.0f, 0.0f};
  float reduction[3] = {0.0f, 0.0f, 0.0f};
  float upstream[TR3B_ROWS] = {0.0f, 0.0f};
  const uint64_t shape3[3] = {1u, 2u, 256u};
  const uint64_t shape2[2] = {1u, 2u};

  et_kernel_runtime *l2 = NULL;
  et_kernel_runtime *l3s = NULL;
  et_kernel_error error;
  status = kernel_status(et_kernel_runtime_discover(
      resolver, (void *)et_l2_indexed_cross_entropy_provider_v1(), &l2,
      &error), &error);
  if (status != ET_TR3B_OK) goto cleanup;
  status = kernel_status(et_kernel_runtime_discover(
      resolver, (void *)et_l3s_kernel_provider_v1(), &l3s, &error), &error);
  if (status != ET_TR3B_OK) goto cleanup;

  et_kernel_request_v1 l2_forward_request = {
      sizeof(l2_forward_request), ET_L2_INDEXED_CROSS_ENTROPY_FORWARD,
      "f32", "cpu", 3u, shape3, 1u, {0}};
  et_kernel_tensor_view_v1 l2_forward_inputs[2] = {
      tensor(logits_header.payload, sizeof(logits_header.payload), "f32", 3u,
             shape3),
      tensor(target_payload, TR3B_TARGET_BYTES, "i64", 2u, shape2)};
  et_kernel_tensor_view_v1 l2_forward_outputs[1] = {
      tensor(losses, sizeof(losses), "f32", 2u, shape2)};
  et_kernel_call_v1 l2_forward = call(
      ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY, &l2_forward_request, 2u,
      l2_forward_inputs, 1u, l2_forward_outputs);
  status = dispatch(l2, &l2_forward);
  if (status != ET_TR3B_OK) goto cleanup;

  et_kernel_request_v1 reduce_request = {
      sizeof(reduce_request), "l3s.masked-objective.reduce.bool", "f32",
      "cpu", 2u, shape2, 1u, {0}};
  et_kernel_tensor_view_v1 reduce_inputs[2] = {
      tensor(losses, sizeof(losses), "f32", 2u, shape2),
      tensor(mask_payload, TR3B_MASK_BYTES, "bool", 2u, shape2)};
  et_kernel_tensor_view_v1 reduce_outputs[3] = {
      tensor(&reduction[0], sizeof(float), "f32", 0u, NULL),
      tensor(&reduction[1], sizeof(float), "f32", 0u, NULL),
      tensor(&reduction[2], sizeof(float), "f32", 0u, NULL)};
  et_kernel_call_v1 reduce = call("l3s.masked-objective", &reduce_request, 2u,
                                  reduce_inputs, 3u, reduce_outputs);
  status = dispatch(l3s, &reduce);
  if (status != ET_TR3B_OK) goto cleanup;

  et_kernel_request_v1 numerator_request = {
      sizeof(numerator_request), "l3s.masked-objective.numerator-seed.bool",
      "f32", "cpu", 2u, shape2, 1u, {0}};
  et_kernel_tensor_view_v1 numerator_inputs[1] = {
      tensor(mask_payload, TR3B_MASK_BYTES, "bool", 2u, shape2)};
  et_kernel_tensor_view_v1 numerator_outputs[1] = {
      tensor(upstream, sizeof(upstream), "f32", 2u, shape2)};
  et_kernel_call_v1 numerator = call(
      "l3s.masked-objective", &numerator_request, 1u, numerator_inputs, 1u,
      numerator_outputs);
  status = dispatch(l3s, &numerator);
  if (status != ET_TR3B_OK) goto cleanup;

  et_kernel_request_v1 l2_backward_request = {
      sizeof(l2_backward_request), ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD,
      "f32", "cpu", 3u, shape3, 1u, {0}};
  et_kernel_tensor_view_v1 l2_backward_inputs[3] = {
      tensor(logits_header.payload, sizeof(logits_header.payload), "f32", 3u,
             shape3),
      tensor(target_payload, TR3B_TARGET_BYTES, "i64", 2u, shape2),
      tensor(upstream, sizeof(upstream), "f32", 2u, shape2)};
  et_kernel_tensor_view_v1 l2_backward_outputs[1] = {
      tensor(gradient_header.payload, sizeof(gradient_header.payload), "f32",
             3u, shape3)};
  et_kernel_call_v1 l2_backward = call(
      ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY, &l2_backward_request, 3u,
      l2_backward_inputs, 1u, l2_backward_outputs);
  status = dispatch(l2, &l2_backward);
  if (status != ET_TR3B_OK) goto cleanup;

cleanup:
  et_kernel_runtime_destroy(l3s);
  et_kernel_runtime_destroy(l2);
  if (status != ET_TR3B_OK) return status;

  /* The genuine seed is the only owner written by preparation. Observation
   * publication follows it as an infallible disjoint 12-byte memcpy. */
  status = m3t_status(et_m3t_private_logits_copy_from_v1(
      m3t_seed, &gradient_header, TR3B_LOGIT_BYTES));
  if (status != ET_TR3B_OK) return status;
  for (size_t index = 0; index < 3u; ++index) {
    uint32_t bits = 0u;
    memcpy(&bits, &reduction[index], sizeof(bits));
    put_u32_le(observation_payload + index * sizeof(bits), bits);
  }
  return ET_TR3B_OK;
}
