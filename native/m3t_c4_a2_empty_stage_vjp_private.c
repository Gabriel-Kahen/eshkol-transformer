#ifndef ET_M3T_C4_A2_EMPTY_STAGE_VJP_PRIVATE
#error "C4 A2 empty-stage VJP is a feature-private leaf"
#endif

#include "m3t_c4_a2_empty_stage_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int check_view(const et_kernel_tensor_view_v1 *view) {
  static const uint64_t shape[4] = {1u, 2u, 4u, 2u};
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_A2_EMPTY_STAGE_VJP_INVALID_ARGUMENT;
  if (view->rank != 4u || view->byte_length != 64u ||
      (uintptr_t)view->shape % _Alignof(uint64_t) != 0u ||
      memcmp(view->shape, shape, sizeof(shape)) != 0)
    return ET_M3T_C4_A2_EMPTY_STAGE_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_A2_EMPTY_STAGE_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_A2_EMPTY_STAGE_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u ||
      (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_A2_EMPTY_STAGE_VJP_LAYOUT;
  return ET_M3T_C4_A2_EMPTY_STAGE_VJP_OK;
}

static int span_overlaps(const void *a, size_t a_bytes,
                         const void *b, size_t b_bytes) {
  const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x > UINTPTR_MAX - a_bytes || y > UINTPTR_MAX - b_bytes ||
         (x < y + b_bytes && y < x + a_bytes);
}

int et_m3t_c4_a2_empty_stage_vjp_private_v1(
    const et_kernel_tensor_view_v1 *cache_key_gradient,
    const et_kernel_tensor_view_v1 *cache_value_gradient,
    et_kernel_tensor_view_v1 *staged_key_gradient,
    et_kernel_tensor_view_v1 *staged_value_gradient) {
  const et_kernel_tensor_view_v1 *views[4] = {
      cache_key_gradient, cache_value_gradient,
      staged_key_gradient, staged_value_gradient};
  for (size_t i = 0u; i < 4u; ++i) {
    const int status = check_view(views[i]);
    if (status != ET_M3T_C4_A2_EMPTY_STAGE_VJP_OK) return status;
  }
  for (size_t i = 0u; i < 4u; ++i)
    for (size_t j = i + 1u; j < 4u; ++j)
      if (span_overlaps(views[i]->data, views[i]->byte_length,
                        views[j]->data, views[j]->byte_length))
        return ET_M3T_C4_A2_EMPTY_STAGE_VJP_OVERLAP;
  for (size_t destination = 2u; destination < 4u; ++destination)
    for (size_t control = 0u; control < 4u; ++control)
      if (span_overlaps(views[destination]->data, 64u,
                        views[control], views[control]->struct_size) ||
          span_overlaps(views[destination]->data, 64u,
                        views[control]->shape, 4u * sizeof(uint64_t)) ||
          span_overlaps(views[destination]->data, 64u,
                        views[control]->dtype, sizeof("f32")) ||
          span_overlaps(views[destination]->data, 64u,
                        views[control]->device, sizeof("cpu")))
        return ET_M3T_C4_A2_EMPTY_STAGE_VJP_OVERLAP;
  const float *dk = cache_key_gradient->data;
  const float *dv = cache_value_gradient->data;
  for (size_t i = 0u; i < 16u; ++i)
    if (!isfinite(dk[i]) || !isfinite(dv[i]))
      return ET_M3T_C4_A2_EMPTY_STAGE_VJP_VALUE;
  memcpy(staged_key_gradient->data, dk, 64u);
  memcpy(staged_value_gradient->data, dv, 64u);
  return ET_M3T_C4_A2_EMPTY_STAGE_VJP_OK;
}
