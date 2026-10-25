#ifndef ET_M3T_C4_RESIDUAL_VJP_PRIVATE
#error "C4 residual VJP is a feature-private leaf"
#endif

#include "m3t_c4_residual_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int overlaps(const void *a, const void *b) {
  const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x > UINTPTR_MAX - 64u || y > UINTPTR_MAX - 64u ||
         (x < y + 64u && y < x + 64u);
}

static int check_view(const et_kernel_tensor_view_v1 *view) {
  static const uint64_t shape[3] = {1u, 4u, 4u};
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT;
  if (view->rank != 3u || view->byte_length != 64u ||
      memcmp(view->shape, shape, sizeof(shape)) != 0)
    return ET_M3T_C4_RESIDUAL_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_RESIDUAL_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_RESIDUAL_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_RESIDUAL_VJP_LAYOUT;
  return ET_M3T_C4_RESIDUAL_VJP_OK;
}

int et_m3t_c4_residual_vjp_private_v1(
    const et_kernel_tensor_view_v1 *left,
    const et_kernel_tensor_view_v1 *right,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *left_gradient,
    et_kernel_tensor_view_v1 *right_gradient) {
  const et_kernel_tensor_view_v1 *views[5] = {
      left, right, upstream, left_gradient, right_gradient};
  for (size_t i = 0; i < 5u; ++i) {
    const int status = check_view(views[i]);
    if (status != ET_M3T_C4_RESIDUAL_VJP_OK) return status;
  }
  for (size_t destination = 3u; destination < 5u; ++destination)
    for (size_t other = 0u; other < destination; ++other)
      if (overlaps(views[destination]->data, views[other]->data))
        return ET_M3T_C4_RESIDUAL_VJP_OVERLAP;
  const float *a = left->data, *b = right->data, *g = upstream->data;
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(g[i]) ||
        !isfinite(a[i] + b[i]))
      return ET_M3T_C4_RESIDUAL_VJP_VALUE;
  memcpy(left_gradient->data, g, 64u);
  memcpy(right_gradient->data, g, 64u);
  return ET_M3T_C4_RESIDUAL_VJP_OK;
}
