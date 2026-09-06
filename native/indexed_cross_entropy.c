#include "eshkol_transformer/indexed_cross_entropy.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

_Static_assert(sizeof(float) == 4u && FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
                   FLT_MAX_EXP == 128,
               "L2 ABI v1 requires IEEE-754 binary32 float storage");

static const char *const l2_operations[] = {
    ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD,
    ET_L2_INDEXED_CROSS_ENTROPY_FORWARD,
};
static const char *const l2_dtypes[] = {"f32"};
static const char *const l2_devices[] = {"cpu"};
static const et_kernel_dimension_range_v1 l2_dimensions[] = {
    {.minimum = 1u, .maximum = ET_L2_INDEXED_CROSS_ENTROPY_MAX_EXTENT,
     .maximum_unbounded = 0u, .reserved = {0}},
    {.minimum = 1u, .maximum = ET_L2_INDEXED_CROSS_ENTROPY_MAX_EXTENT,
     .maximum_unbounded = 0u, .reserved = {0}},
    {.minimum = 1u, .maximum = ET_L2_INDEXED_CROSS_ENTROPY_MAX_EXTENT,
     .maximum_unbounded = 0u, .reserved = {0}},
};
static const et_kernel_shape_range_v1 l2_shapes[] = {
    {.rank = 3u, .dimensions = l2_dimensions},
};

static int32_t l2_error(et_kernel_error *error,
                        et_kernel_error_category category,
                        et_kernel_error_code code, const char *operation,
                        const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static const et_kernel_tensor_view_v1 *l2_tensor_at(const void *base,
                                                     size_t stride,
                                                     size_t index) {
  return (const et_kernel_tensor_view_v1 *)((const unsigned char *)base +
                                             index * stride);
}

static int l2_text_equal(const char *left, const char *right) {
  return left != NULL && right != NULL && strcmp(left, right) == 0;
}

static int l2_aligned_span(const et_kernel_tensor_view_v1 *tensor,
                           size_t alignment) {
  const uintptr_t start = (uintptr_t)tensor->data;
  return start % alignment == 0u &&
         start <= UINTPTR_MAX - tensor->byte_length;
}

static int l2_overlaps(const et_kernel_tensor_view_v1 *left,
                       const et_kernel_tensor_view_v1 *right) {
  const uintptr_t left_start = (uintptr_t)left->data;
  const uintptr_t right_start = (uintptr_t)right->data;
  const uintptr_t left_end = left_start + left->byte_length;
  const uintptr_t right_end = right_start + right->byte_length;
  return left_start < right_end && right_start < left_end;
}

static int32_t l2_expect_tensor(const et_kernel_tensor_view_v1 *tensor,
                                const char *dtype, size_t rank,
                                const uint64_t *shape,
                                et_kernel_error *error) {
  if (!l2_text_equal(tensor->dtype, dtype)) {
    return l2_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                    ET_KERNEL_CODE_PROVIDER_REJECTED,
                    "indexed-cross-entropy", "operand dtype is invalid");
  }
  if (tensor->rank != rank) {
    return l2_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                    ET_KERNEL_CODE_PROVIDER_REJECTED,
                    "indexed-cross-entropy", "operand rank is invalid");
  }
  for (size_t dimension = 0; dimension < rank; dimension++) {
    if (tensor->shape[dimension] != shape[dimension]) {
      return l2_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                      ET_KERNEL_CODE_PROVIDER_REJECTED,
                      "indexed-cross-entropy",
                      "operand shape does not match the request");
    }
  }
  if (!l2_aligned_span(tensor,
                       l2_text_equal(dtype, "i64") ? _Alignof(int64_t)
                                                    : _Alignof(float))) {
    return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_INVALID_BUFFER,
                    "indexed-cross-entropy",
                    "operand storage is misaligned or its range overflows");
  }
  return 0;
}

static float l2_row_maximum(const float *row, size_t vocabulary) {
  float maximum = row[0];
  for (size_t token = 1u; token < vocabulary; token++) {
    if (row[token] > maximum) {
      maximum = row[token];
    }
  }
  return maximum;
}

static float l2_row_exponential_sum(const float *row, size_t vocabulary,
                                    float maximum) {
  float sum = 0.0f;
  for (size_t token = 0u; token < vocabulary; token++) {
    sum += expf(row[token] - maximum);
  }
  return sum;
}

static int32_t l2_validate_call(const et_kernel_call_v1 *call,
                                et_kernel_error *error) {
  const et_kernel_request_v1 *request = call->request;
  const int forward = l2_text_equal(request->operation,
                                    ET_L2_INDEXED_CROSS_ENTROPY_FORWARD);
  const int backward = l2_text_equal(request->operation,
                                     ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD);
  const et_kernel_tensor_view_v1 *logits;
  const et_kernel_tensor_view_v1 *targets;
  const et_kernel_tensor_view_v1 *upstream = NULL;
  const et_kernel_tensor_view_v1 *output;
  uint64_t leading_shape[2];
  size_t rows;
  size_t vocabulary;
  int32_t result;

  if ((!forward && !backward) || request->rank != 3u ||
      request->shape[0] == 0u || request->shape[1] == 0u ||
      request->shape[2] == 0u) {
    return l2_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                    ET_KERNEL_CODE_PROVIDER_REJECTED,
                    "indexed-cross-entropy", "operation or request shape is invalid");
  }
  if (call->input_count != (forward ? 2u : 3u) || call->output_count != 1u) {
    return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_PROVIDER_REJECTED,
                    request->operation, "operand count is invalid");
  }

  logits = l2_tensor_at(call->inputs, call->input_stride, 0u);
  targets = l2_tensor_at(call->inputs, call->input_stride, 1u);
  output = l2_tensor_at(call->outputs, call->output_stride, 0u);
  leading_shape[0] = request->shape[0];
  leading_shape[1] = request->shape[1];
  result = l2_expect_tensor(logits, "f32", 3u, request->shape, error);
  if (result != 0) {
    return result;
  }
  result = l2_expect_tensor(targets, "i64", 2u, leading_shape, error);
  if (result != 0) {
    return result;
  }
  if (backward) {
    upstream = l2_tensor_at(call->inputs, call->input_stride, 2u);
    result = l2_expect_tensor(upstream, "f32", 2u, leading_shape, error);
    if (result != 0) {
      return result;
    }
    result = l2_expect_tensor(output, "f32", 3u, request->shape, error);
    if (result != 0) {
      return result;
    }
  } else {
    result = l2_expect_tensor(output, "f32", 2u, leading_shape, error);
    if (result != 0) {
      return result;
    }
  }
  if (l2_overlaps(logits, targets) ||
      (backward && (l2_overlaps(logits, upstream) ||
                    l2_overlaps(targets, upstream)))) {
    return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                    ET_KERNEL_CODE_ALIASING_OUTPUT, request->operation,
                    "input tensor storage must be pairwise disjoint");
  }

  rows = logits->byte_length / (sizeof(float) * (size_t)request->shape[2]);
  vocabulary = (size_t)request->shape[2];
  for (size_t row_index = 0u; row_index < rows; row_index++) {
    const float *row = (const float *)logits->data + row_index * vocabulary;
    const int64_t target = ((const int64_t *)targets->data)[row_index];
    float maximum;
    float sum;
    if (target < 0 || (uint64_t)target >= request->shape[2]) {
      return l2_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                      ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                      "target index is outside the vocabulary");
    }
    for (size_t token = 0u; token < vocabulary; token++) {
      if (!isfinite(row[token])) {
        return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                        ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                        "logits must be finite f32 values");
      }
    }
    maximum = l2_row_maximum(row, vocabulary);
    sum = l2_row_exponential_sum(row, vocabulary, maximum);
    if (!isfinite(sum) || !(sum > 0.0f)) {
      return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                      ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                      "softmax normalization is not finite and positive");
    }
    if (forward) {
      const float loss = logf(sum) + (maximum - row[(size_t)target]);
      if (!isfinite(loss)) {
        return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                        ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                        "per-token loss is not representable as finite f32");
      }
    } else {
      const float incoming = ((const float *)upstream->data)[row_index];
      if (!isfinite(incoming)) {
        return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                        ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                        "upstream gradients must be finite f32 values");
      }
      for (size_t token = 0u; token < vocabulary; token++) {
        const float probability = expf(row[token] - maximum) / sum;
        const float gradient =
            incoming * (probability - (token == (size_t)target ? 1.0f : 0.0f));
        if (!isfinite(gradient)) {
          return l2_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                          ET_KERNEL_CODE_PROVIDER_REJECTED, request->operation,
                          "logit gradient is not representable as finite f32");
        }
      }
    }
  }
  et_kernel_error_clear(error);
  return 0;
}

static void l2_invoke_call(const et_kernel_call_v1 *call) {
  const et_kernel_request_v1 *request = call->request;
  const int backward = l2_text_equal(request->operation,
                                     ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD);
  const et_kernel_tensor_view_v1 *logits =
      l2_tensor_at(call->inputs, call->input_stride, 0u);
  const et_kernel_tensor_view_v1 *targets =
      l2_tensor_at(call->inputs, call->input_stride, 1u);
  const et_kernel_tensor_view_v1 *upstream =
      backward ? l2_tensor_at(call->inputs, call->input_stride, 2u) : NULL;
  const et_kernel_tensor_view_v1 *output =
      l2_tensor_at(call->outputs, call->output_stride, 0u);
  const size_t vocabulary = (size_t)request->shape[2];
  const size_t rows = logits->byte_length / (sizeof(float) * vocabulary);

  for (size_t row_index = 0u; row_index < rows; row_index++) {
    const float *row = (const float *)logits->data + row_index * vocabulary;
    const size_t target = (size_t)((const int64_t *)targets->data)[row_index];
    const float maximum = l2_row_maximum(row, vocabulary);
    const float sum = l2_row_exponential_sum(row, vocabulary, maximum);
    if (!backward) {
      ((float *)output->data)[row_index] =
          logf(sum) + (maximum - row[target]);
    } else {
      const float incoming = ((const float *)upstream->data)[row_index];
      float *gradient = (float *)output->data + row_index * vocabulary;
      for (size_t token = 0u; token < vocabulary; token++) {
        const float probability = expf(row[token] - maximum) / sum;
        gradient[token] = incoming *
                          (probability - (token == target ? 1.0f : 0.0f));
      }
    }
  }
}

static const et_kernel_capability_v1 l2_capability = {
    .struct_size = sizeof(et_kernel_capability_v1),
    .name = ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY,
    .status = ET_KERNEL_CAPABILITY_VERIFIED,
    .implementation = "eshkol-transformer-l2-cpu",
    .version = "1.0.0",
    .evidence = "L2:cpu-f32-indexed-cross-entropy-v1",
    .deterministic = 1u,
    .reserved = {0},
    .operation_count = sizeof(l2_operations) / sizeof(l2_operations[0]),
    .operations = l2_operations,
    .dtype_count = sizeof(l2_dtypes) / sizeof(l2_dtypes[0]),
    .dtypes = l2_dtypes,
    .device_count = sizeof(l2_devices) / sizeof(l2_devices[0]),
    .devices = l2_devices,
    .shape_range_count = sizeof(l2_shapes) / sizeof(l2_shapes[0]),
    .shape_ranges = l2_shapes,
};

static const et_kernel_provider_v1 l2_provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = ET_KERNEL_ABI_MINOR,
    .required_features = 0u,
    .name = "eshkol-transformer-l2-cpu",
    .version = "1.0.0",
    .evidence = "L2:cpu-f32-indexed-cross-entropy-v1",
    .capability_count = 1u,
    .capability_stride = sizeof(et_kernel_capability_v1),
    .capability_bytes = sizeof(et_kernel_capability_v1),
    .capabilities = &l2_capability,
    .validate_call = l2_validate_call,
    .invoke_call = l2_invoke_call,
};

int32_t et_l2_indexed_cross_entropy_abi_major_v1(void) {
  return (int32_t)ET_L2_INDEXED_CROSS_ENTROPY_ABI_MAJOR;
}

int32_t et_l2_indexed_cross_entropy_abi_minor_v1(void) {
  return (int32_t)ET_L2_INDEXED_CROSS_ENTROPY_ABI_MINOR;
}

const et_kernel_provider_v1 *et_l2_indexed_cross_entropy_provider_v1(void) {
  return &l2_provider;
}
