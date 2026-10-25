#ifndef ET_M3T_C4_EMBEDDING_VJP_PRIVATE
#error "C4 embedding VJP is a feature-private leaf"
#endif

#include "m3t_c4_embedding_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int check_view(const et_kernel_tensor_view_v1 *v, const char *dtype,
                      size_t rank, const uint64_t *shape, size_t bytes,
                      size_t alignment) {
  if (!v || v->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !v->data || !v->shape || !v->dtype || !v->device)
    return ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT;
  if (v->rank != rank || v->byte_length != bytes ||
      (uintptr_t)v->shape % _Alignof(uint64_t) != 0u ||
      memcmp(v->shape, shape, rank * sizeof(uint64_t)) != 0)
    return ET_M3T_C4_EMBEDDING_VJP_SHAPE;
  if (strcmp(v->dtype, dtype) != 0)
    return ET_M3T_C4_EMBEDDING_VJP_DTYPE;
  if (strcmp(v->device, "cpu") != 0)
    return ET_M3T_C4_EMBEDDING_VJP_DEVICE;
  if (v->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR || v->offset_bytes ||
      (uintptr_t)v->data % alignment != 0u)
    return ET_M3T_C4_EMBEDDING_VJP_LAYOUT;
  return ET_M3T_C4_EMBEDDING_VJP_OK;
}

static int overlaps(const void *a, size_t an, const void *b, size_t bn) {
  const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn ||
         (x < y + bn && y < x + an);
}

int et_m3t_c4_embedding_vjp_private_v1(
    const et_kernel_tensor_view_v1 *ids,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *gradient) {
  const uint64_t ids_shape[2] = {1u, 4u};
  const uint64_t up_shape[3] = {1u, 4u, 4u};
  const uint64_t grad_shape[2] = {256u, 4u};
  int status = check_view(ids, "i64", 2u, ids_shape, 32u, _Alignof(int64_t));
  if (status) return status;
  status = check_view(upstream, "f32", 3u, up_shape, 64u, _Alignof(float));
  if (status) return status;
  status = check_view(gradient, "f32", 2u, grad_shape, 4096u,
                      _Alignof(float));
  if (status) return status;
  if (overlaps(gradient->data, 4096u, ids->data, 32u) ||
      overlaps(gradient->data, 4096u, upstream->data, 64u))
    return ET_M3T_C4_EMBEDDING_VJP_OVERLAP;
  const int64_t *id = ids->data;
  const float *u = upstream->data;
  for (size_t t = 0; t < 4u; ++t)
    if (id[t] < 0 || id[t] >= 256)
      return ET_M3T_C4_EMBEDDING_VJP_VALUE;
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(u[i])) return ET_M3T_C4_EMBEDDING_VJP_VALUE;
  float staged[1024] = {0};
  for (size_t t = 0; t < 4u; ++t)
    for (size_t d = 0; d < 4u; ++d) {
      float *target = &staged[(size_t)id[t] * 4u + d];
      *target += u[t * 4u + d];
      if (!isfinite(*target)) return ET_M3T_C4_EMBEDDING_VJP_VALUE;
    }
  memcpy(gradient->data, staged, sizeof(staged));
  return ET_M3T_C4_EMBEDDING_VJP_OK;
}
