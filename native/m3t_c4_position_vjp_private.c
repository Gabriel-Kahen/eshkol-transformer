#ifndef ET_M3T_C4_POSITION_VJP_PRIVATE
#error "C4 position VJP is a feature-private leaf"
#endif

#include "m3t_c4_position_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int span_overlaps(const void *a, size_t an, const void *b, size_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn) return 1;
  return x < y + bn && y < x + an;
}

static int check_view(const et_kernel_tensor_view_v1 *v, const char *dtype,
                      size_t rank, const uint64_t *shape, size_t bytes,
                      size_t alignment) {
  if (!v || v->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !v->data || !v->shape || !v->dtype || !v->device)
    return ET_M3T_C4_POSITION_VJP_INVALID_ARGUMENT;
  if (v->rank != rank || v->byte_length != bytes)
    return ET_M3T_C4_POSITION_VJP_SHAPE;
  for (size_t i = 0; i < rank; ++i)
    if (v->shape[i] != shape[i]) return ET_M3T_C4_POSITION_VJP_SHAPE;
  if (strcmp(v->dtype, dtype)) return ET_M3T_C4_POSITION_VJP_DTYPE;
  if (strcmp(v->device, "cpu")) return ET_M3T_C4_POSITION_VJP_DEVICE;
  if (v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || v->offset_bytes ||
      (uintptr_t)v->data % alignment)
    return ET_M3T_C4_POSITION_VJP_LAYOUT;
  return ET_M3T_C4_POSITION_VJP_OK;
}

int et_m3t_c4_position_vjp_private_v1(
    const et_kernel_tensor_view_v1 *ids,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *gradient) {
  static const uint64_t ids_shape[2] = {1, 4};
  static const uint64_t up_shape[3] = {1, 4, 4};
  static const uint64_t grad_shape[2] = {4, 4};
  int status = check_view(ids, "i64", 2, ids_shape, 32, 8);
  if (status) return status;
  status = check_view(upstream, "f32", 3, up_shape, 64, 4);
  if (status) return status;
  status = check_view(gradient, "f32", 2, grad_shape, 64, 4);
  if (status) return status;
  if (span_overlaps(gradient->data, 64, ids->data, 32) ||
      span_overlaps(gradient->data, 64, upstream->data, 64))
    return ET_M3T_C4_POSITION_VJP_OVERLAP;
  const int64_t *position = ids->data;
  for (int64_t i = 0; i < 4; ++i)
    if (position[i] != i) return ET_M3T_C4_POSITION_VJP_VALUE;
  const float *source = upstream->data;
  for (size_t i = 0; i < 16; ++i)
    if (!isfinite(source[i])) return ET_M3T_C4_POSITION_VJP_VALUE;
  memcpy(gradient->data, source, 64);
  return ET_M3T_C4_POSITION_VJP_OK;
}
