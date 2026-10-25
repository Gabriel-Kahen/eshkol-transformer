#ifndef ET_M3T_C4_GELU_VJP_PRIVATE
#error "C4 GELU VJP is a feature-private leaf"
#endif

#include "m3t_c4_gelu_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int overlaps(const void *a, const void *b) {
  const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x > UINTPTR_MAX - 128u || y > UINTPTR_MAX - 128u ||
         (x < y + 128u && y < x + 128u);
}

static int check_view(const et_kernel_tensor_view_v1 *view) {
  static const uint64_t shape[3] = {1u, 4u, 8u};
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT;
  if (view->rank != 3u || view->byte_length != 128u ||
      memcmp(view->shape, shape, sizeof(shape)) != 0)
    return ET_M3T_C4_GELU_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_GELU_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_GELU_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_GELU_VJP_LAYOUT;
  return ET_M3T_C4_GELU_VJP_OK;
}

int et_m3t_c4_gelu_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *gradient) {
  int status = check_view(input);
  if (status) return status;
  status = check_view(upstream);
  if (status) return status;
  status = check_view(gradient);
  if (status) return status;
  if (overlaps(gradient->data, input->data) ||
      overlaps(gradient->data, upstream->data))
    return ET_M3T_C4_GELU_VJP_OVERLAP;
  const float *x = input->data, *g = upstream->data;
  float staged[32];
  const float half = 0x1p-1f;
  const float inv_sqrt_2 = 0x1.6a09e6p-1f;
  const float inv_sqrt_pi = 0x1.20dd76p-1f;
  for (size_t i = 0; i < 32u; ++i) {
    if (!isfinite(x[i]) || !isfinite(g[i])) return ET_M3T_C4_GELU_VJP_VALUE;
    const float z = x[i] * inv_sqrt_2;
    const float erf_value = erff(z);
    const float gaussian = (float)exp(-(double)z * (double)z);
    const float slope = half * (1.0f + erf_value) +
                        ((x[i] * inv_sqrt_2) * inv_sqrt_pi) * gaussian;
    staged[i] = slope * g[i];
    if (!isfinite(z) || !isfinite(erf_value) || !isfinite(gaussian) ||
        !isfinite(slope) || !isfinite(staged[i]))
      return ET_M3T_C4_GELU_VJP_VALUE;
  }
  memcpy(gradient->data, staged, sizeof(staged));
  return ET_M3T_C4_GELU_VJP_OK;
}
