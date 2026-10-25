#ifndef ET_M3T_C4_LAYERNORM_VJP_PRIVATE
#error "C4 layernorm VJP is a feature-private leaf"
#endif

#include "m3t_c4_layernorm_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int check_view(const et_kernel_tensor_view_v1 *view, size_t rank,
                      const uint64_t *shape, size_t bytes) {
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_LAYERNORM_VJP_INVALID_ARGUMENT;
  if (view->rank != rank || view->byte_length != bytes ||
      (uintptr_t)view->shape % _Alignof(uint64_t) != 0u ||
      (rank && memcmp(view->shape, shape, rank * sizeof(uint64_t)) != 0))
    return ET_M3T_C4_LAYERNORM_VJP_SHAPE;
  if (strcmp(view->dtype, "f32") != 0)
    return ET_M3T_C4_LAYERNORM_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_LAYERNORM_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % _Alignof(float) != 0u)
    return ET_M3T_C4_LAYERNORM_VJP_LAYOUT;
  return ET_M3T_C4_LAYERNORM_VJP_OK;
}

static int overlaps(const void *left, size_t left_bytes,
                    const void *right, size_t right_bytes) {
  const uintptr_t a = (uintptr_t)left, b = (uintptr_t)right;
  return a > UINTPTR_MAX - left_bytes || b > UINTPTR_MAX - right_bytes ||
         (a < b + right_bytes && b < a + left_bytes);
}

int et_m3t_c4_layernorm_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *gamma,
    const et_kernel_tensor_view_v1 *beta,
    const et_kernel_tensor_view_v1 *epsilon,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient,
    et_kernel_tensor_view_v1 *gamma_gradient,
    et_kernel_tensor_view_v1 *beta_gradient) {
  const et_kernel_tensor_view_v1 *views[8] = {
      input, gamma, beta, epsilon, upstream,
      input_gradient, gamma_gradient, beta_gradient};
  const uint64_t tensor_shape[3] = {1u, 4u, 4u};
  const uint64_t channel_shape[1] = {4u};
  const uint64_t *shapes[8] = {tensor_shape, channel_shape, channel_shape,
      NULL, tensor_shape, tensor_shape, channel_shape, channel_shape};
  const size_t ranks[8] = {3u, 1u, 1u, 0u, 3u, 3u, 1u, 1u};
  const size_t bytes[8] = {64u, 16u, 16u, 4u, 64u, 64u, 16u, 16u};
  for (size_t i = 0; i < 8u; ++i) {
    const int status = check_view(views[i], ranks[i], shapes[i], bytes[i]);
    if (status) return status;
  }
  for (size_t destination = 5u; destination < 8u; ++destination)
    for (size_t other = 0u; other < destination; ++other)
      if (overlaps(views[destination]->data, bytes[destination],
                   views[other]->data, bytes[other]))
        return ET_M3T_C4_LAYERNORM_VJP_OVERLAP;

  const float *x = input->data, *g = gamma->data, *b = beta->data;
  const float *u = upstream->data;
  const float eps = *(const float *)epsilon->data;
  if (!isfinite(eps) || eps <= 0.0f)
    return ET_M3T_C4_LAYERNORM_VJP_VALUE;
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(x[i]) || !isfinite(u[i]))
      return ET_M3T_C4_LAYERNORM_VJP_VALUE;
  for (size_t i = 0; i < 4u; ++i)
    if (!isfinite(g[i]) || !isfinite(b[i]))
      return ET_M3T_C4_LAYERNORM_VJP_VALUE;

  float staged_x[16], staged_g[4] = {0}, staged_b[4] = {0};
  for (size_t t = 0; t < 4u; ++t) {
    float sum = 0.0f, variance_sum = 0.0f, delta[4], normalized[4];
    float v[4], mean_v = 0.0f, mean_v_delta = 0.0f;
    for (size_t d = 0; d < 4u; ++d) {
      sum += x[4u * t + d];
      if (!isfinite(sum)) return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    }
    const float mean = sum / 4.0f;
    if (!isfinite(mean)) return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    for (size_t d = 0; d < 4u; ++d) {
      delta[d] = x[4u * t + d] - mean;
      const float square = delta[d] * delta[d];
      variance_sum += square;
      if (!isfinite(delta[d]) || !isfinite(square) || !isfinite(variance_sum))
        return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    }
    const float variance = variance_sum / 4.0f;
    const float adjusted = variance + eps;
    const float root = sqrtf(adjusted);
    const float rstd = 1.0f / root;
    const float rstd_squared = rstd * rstd;
    if (!isfinite(variance) || !isfinite(adjusted) || !isfinite(root) ||
        !isfinite(rstd) || !isfinite(rstd_squared))
      return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    for (size_t d = 0; d < 4u; ++d) {
      const float scaled = (delta[d] * rstd) * g[d];
      const float value = scaled + b[d];
      normalized[d] = delta[d] * rstd;
      v[d] = u[4u * t + d] * g[d];
      const float vd = v[d] * delta[d];
      mean_v += v[d];
      mean_v_delta += vd;
      if (!isfinite(normalized[d]) || !isfinite(scaled) ||
          !isfinite(value) || !isfinite(v[d]) || !isfinite(vd) ||
          !isfinite(mean_v) || !isfinite(mean_v_delta))
        return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    }
    mean_v /= 4.0f;
    mean_v_delta /= 4.0f;
    for (size_t d = 0; d < 4u; ++d) {
      const size_t i = 4u * t + d;
      const float centered = v[d] - mean_v;
      const float variance_term = delta[d] * rstd_squared * mean_v_delta;
      staged_x[i] = rstd * (centered - variance_term);
      staged_g[d] += u[i] * normalized[d];
      staged_b[d] += u[i];
      if (!isfinite(mean_v) || !isfinite(mean_v_delta) ||
          !isfinite(centered) || !isfinite(variance_term) ||
          !isfinite(staged_x[i]) || !isfinite(staged_g[d]) ||
          !isfinite(staged_b[d]))
        return ET_M3T_C4_LAYERNORM_VJP_VALUE;
    }
  }
  memcpy(input_gradient->data, staged_x, sizeof(staged_x));
  memcpy(gamma_gradient->data, staged_g, sizeof(staged_g));
  memcpy(beta_gradient->data, staged_b, sizeof(staged_b));
  return ET_M3T_C4_LAYERNORM_VJP_OK;
}
