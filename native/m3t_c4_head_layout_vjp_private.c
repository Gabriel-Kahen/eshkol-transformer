#ifndef ET_M3T_C4_HEAD_LAYOUT_VJP_PRIVATE
#error "C4 head-layout VJP is a feature-private leaf"
#endif

#include "m3t_c4_head_layout_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int check_view(const et_kernel_tensor_view_v1 *view, size_t rank,
                      const uint64_t *shape) {
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT;
  if (view->rank != rank || view->byte_length != 64u ||
      (uintptr_t)view->shape % _Alignof(uint64_t) != 0u ||
      memcmp(view->shape, shape, rank * sizeof(uint64_t)) != 0)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_LAYOUT;
  return ET_M3T_C4_HEAD_LAYOUT_VJP_OK;
}

static int overlaps(const void *left, const void *right) {
  const uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
  return a > UINTPTR_MAX - 64u || b > UINTPTR_MAX - 64u ||
         (a < b + 64u && b < a + 64u);
}

int et_m3t_c4_head_layout_vjp_private_v1(
    int operation, const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient) {
  if (operation != ET_M3T_C4_HEAD_LAYOUT_SPLIT &&
      operation != ET_M3T_C4_HEAD_LAYOUT_MERGE)
    return ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT;
  const uint64_t token_shape[3] = {1u, 4u, 4u};
  const uint64_t head_shape[4] = {1u, 2u, 4u, 2u};
  const int split = operation == ET_M3T_C4_HEAD_LAYOUT_SPLIT;
  const uint64_t *input_shape = split ? token_shape : head_shape;
  const uint64_t *upstream_shape = split ? head_shape : token_shape;
  const size_t input_rank = split ? 3u : 4u;
  const size_t upstream_rank = split ? 4u : 3u;
  int status = check_view(input, input_rank, input_shape);
  if (status) return status;
  status = check_view(upstream, upstream_rank, upstream_shape);
  if (status) return status;
  status = check_view(input_gradient, input_rank, input_shape);
  if (status) return status;
  if (overlaps(input_gradient->data, input->data) ||
      overlaps(input_gradient->data, upstream->data))
    return ET_M3T_C4_HEAD_LAYOUT_VJP_OVERLAP;
  const float *x = input->data, *u = upstream->data;
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(x[i]) || !isfinite(u[i]))
      return ET_M3T_C4_HEAD_LAYOUT_VJP_VALUE;

  float staged[16];
  for (size_t t = 0; t < 4u; ++t)
    for (size_t h = 0; h < 2u; ++h)
      for (size_t d = 0; d < 2u; ++d) {
        const size_t token = (t * 2u + h) * 2u + d;
        const size_t head = (h * 4u + t) * 2u + d;
        memcpy(staged + (split ? token : head),
               u + (split ? head : token), sizeof(float));
      }
  memcpy(input_gradient->data, staged, sizeof(staged));
  return ET_M3T_C4_HEAD_LAYOUT_VJP_OK;
}
