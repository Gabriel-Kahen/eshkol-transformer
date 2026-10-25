#ifndef ET_M3T_C4_LINEAR_VJP_PRIVATE
#error "C4 linear VJP is a feature-private leaf"
#endif

#include "m3t_c4_linear_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int row_admitted(size_t input_width, size_t output_width) {
  return (input_width == 4u &&
          (output_width == 4u || output_width == 8u || output_width == 256u)) ||
         (input_width == 8u && output_width == 4u);
}

static int check_view(const et_kernel_tensor_view_v1 *view, size_t rank,
                      const uint64_t *shape, size_t bytes) {
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT;
  if (view->rank != rank || view->byte_length != bytes ||
      (uintptr_t)view->shape % _Alignof(uint64_t) != 0u ||
      memcmp(view->shape, shape, rank * sizeof(uint64_t)) != 0)
    return ET_M3T_C4_LINEAR_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_LINEAR_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_LINEAR_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_LINEAR_VJP_LAYOUT;
  return ET_M3T_C4_LINEAR_VJP_OK;
}

static int overlaps(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
  const uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
  return a > UINTPTR_MAX - left_bytes || b > UINTPTR_MAX - right_bytes ||
         (a < b + right_bytes && b < a + left_bytes);
}

int et_m3t_c4_linear_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *weight,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient,
    et_kernel_tensor_view_v1 *weight_gradient) {
  if (!input || !weight || !upstream || !input_gradient || !weight_gradient ||
      !input->shape || !weight->shape || !upstream->shape ||
      !input_gradient->shape || !weight_gradient->shape ||
      input->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      weight->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      (uintptr_t)input->shape % _Alignof(uint64_t) != 0u ||
      (uintptr_t)weight->shape % _Alignof(uint64_t) != 0u ||
      input->rank != 3u || weight->rank != 2u)
    return ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT;
  const size_t input_width = (size_t)input->shape[2];
  const size_t output_width = (size_t)weight->shape[0];
  if (!row_admitted(input_width, output_width))
    return ET_M3T_C4_LINEAR_VJP_SHAPE;
  const uint64_t input_shape[3] = {1u, 4u, input_width};
  const uint64_t weight_shape[2] = {output_width, input_width};
  const uint64_t upstream_shape[3] = {1u, 4u, output_width};
  const size_t input_bytes = 16u * input_width;
  const size_t weight_bytes = 4u * output_width * input_width;
  const size_t upstream_bytes = 16u * output_width;
  const et_kernel_tensor_view_v1 *views[5] = {
      input, weight, upstream, input_gradient, weight_gradient};
  const uint64_t *shapes[5] = {
      input_shape, weight_shape, upstream_shape, input_shape, weight_shape};
  const size_t ranks[5] = {3u, 2u, 3u, 3u, 2u};
  const size_t bytes[5] = {
      input_bytes, weight_bytes, upstream_bytes, input_bytes, weight_bytes};
  for (size_t i = 0; i < 5u; ++i) {
    const int status = check_view(views[i], ranks[i], shapes[i], bytes[i]);
    if (status) return status;
  }
  for (size_t destination = 3u; destination < 5u; ++destination)
    for (size_t other = 0u; other < destination; ++other)
      if (overlaps(views[destination]->data, bytes[destination],
                   views[other]->data, bytes[other]))
        return ET_M3T_C4_LINEAR_VJP_OVERLAP;

  const float *x = input->data, *w = weight->data, *u = upstream->data;
  for (size_t i = 0; i < 4u * input_width; ++i)
    if (!isfinite(x[i])) return ET_M3T_C4_LINEAR_VJP_VALUE;
  for (size_t i = 0; i < output_width * input_width; ++i)
    if (!isfinite(w[i])) return ET_M3T_C4_LINEAR_VJP_VALUE;
  for (size_t i = 0; i < 4u * output_width; ++i)
    if (!isfinite(u[i])) return ET_M3T_C4_LINEAR_VJP_VALUE;

  float staged_input[32], staged_weight[1024];
  for (size_t t = 0; t < 4u; ++t) {
    for (size_t o = 0; o < output_width; ++o) {
      float forward = 0.0f;
      for (size_t i = 0; i < input_width; ++i) {
        const float product = x[t * input_width + i] *
                              w[o * input_width + i];
        forward += product;
        if (!isfinite(product) || !isfinite(forward))
          return ET_M3T_C4_LINEAR_VJP_VALUE;
      }
    }
    for (size_t i = 0; i < input_width; ++i) {
      float sum = 0.0f;
      for (size_t o = 0; o < output_width; ++o) {
        const float product = u[t * output_width + o] *
                              w[o * input_width + i];
        sum += product;
        if (!isfinite(product) || !isfinite(sum))
          return ET_M3T_C4_LINEAR_VJP_VALUE;
      }
      staged_input[t * input_width + i] = sum;
    }
  }
  for (size_t o = 0; o < output_width; ++o)
    for (size_t i = 0; i < input_width; ++i) {
      float sum = 0.0f;
      for (size_t t = 0; t < 4u; ++t) {
        const float product = u[t * output_width + o] *
                              x[t * input_width + i];
        sum += product;
        if (!isfinite(product) || !isfinite(sum))
          return ET_M3T_C4_LINEAR_VJP_VALUE;
      }
      staged_weight[o * input_width + i] = sum;
    }
  memcpy(input_gradient->data, staged_input, input_bytes);
  memcpy(weight_gradient->data, staged_weight, weight_bytes);
  return ET_M3T_C4_LINEAR_VJP_OK;
}
