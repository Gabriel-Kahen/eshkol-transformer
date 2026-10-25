#ifndef ET_M3T_C4_ATTENTION_VJP_PRIVATE
#error "C4 attention VJP is a feature-private leaf"
#endif

#include "m3t_c4_attention_vjp_private.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int check_view(const et_kernel_tensor_view_v1 *view, const char *dtype,
                      size_t rank, const uint64_t *shape, size_t bytes,
                      size_t alignment) {
  if (!view || view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      !view->data || !view->shape || !view->dtype || !view->device)
    return ET_M3T_C4_ATTENTION_VJP_INVALID_ARGUMENT;
  if (view->rank != rank || view->byte_length != bytes ||
      (uintptr_t)view->shape % _Alignof(uint64_t) != 0u ||
      memcmp(view->shape, shape, rank * sizeof(uint64_t)) != 0)
    return ET_M3T_C4_ATTENTION_VJP_SHAPE;
  if (strcmp(view->dtype, dtype) != 0)
    return ET_M3T_C4_ATTENTION_VJP_DTYPE;
  if (strcmp(view->device, "cpu") != 0)
    return ET_M3T_C4_ATTENTION_VJP_DEVICE;
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u || (uintptr_t)view->data % alignment != 0u)
    return ET_M3T_C4_ATTENTION_VJP_LAYOUT;
  return ET_M3T_C4_ATTENTION_VJP_OK;
}

static int overlaps(const et_kernel_tensor_view_v1 *a,
                    const et_kernel_tensor_view_v1 *b) {
  const uintptr_t x = (uintptr_t)a->data, y = (uintptr_t)b->data;
  return x > UINTPTR_MAX - a->byte_length ||
         y > UINTPTR_MAX - b->byte_length ||
         (x < y + b->byte_length && y < x + a->byte_length);
}

static size_t head_index(size_t h, size_t t, size_t d) {
  return (h * 4u + t) * 2u + d;
}

int et_m3t_c4_attention_vjp_private_v1(
    const et_kernel_tensor_view_v1 *query,
    const et_kernel_tensor_view_v1 *key,
    const et_kernel_tensor_view_v1 *value,
    const et_kernel_tensor_view_v1 *query_positions,
    const et_kernel_tensor_view_v1 *key_positions,
    const et_kernel_tensor_view_v1 *keep,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *query_gradient,
    et_kernel_tensor_view_v1 *key_gradient,
    et_kernel_tensor_view_v1 *value_gradient) {
  const uint64_t head_shape[4] = {1u, 2u, 4u, 2u};
  const uint64_t position_shape[2] = {1u, 4u};
  const uint64_t mask_shape[3] = {1u, 4u, 4u};
  const et_kernel_tensor_view_v1 *views[10] = {
    query, key, value, query_positions, key_positions, keep, upstream,
    query_gradient, key_gradient, value_gradient};
  for (size_t i = 0; i < 10u; ++i) {
    const int pos = i == 3u || i == 4u, mask = i == 5u;
    const int status = check_view(views[i], pos ? "i64" : mask ? "bool" : "f32",
        pos ? 2u : mask ? 3u : 4u,
        pos ? position_shape : mask ? mask_shape : head_shape,
        pos ? 32u : mask ? 16u : 64u,
        pos ? _Alignof(int64_t) : mask ? 1u : _Alignof(float));
    if (status) return status;
  }
  for (size_t i = 0; i < 10u; ++i)
    for (size_t j = i + 1u; j < 10u; ++j)
      if (overlaps(views[i], views[j]) &&
          !(i == 3u && j == 4u &&
            query_positions->data == key_positions->data))
        return ET_M3T_C4_ATTENTION_VJP_OVERLAP;

  const float *q = query->data, *k = key->data, *v = value->data;
  const float *u = upstream->data;
  const int64_t *qp = query_positions->data, *kp = key_positions->data;
  const uint8_t *mask = keep->data;
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(q[i]) || !isfinite(k[i]) || !isfinite(v[i]) ||
        !isfinite(u[i])) return ET_M3T_C4_ATTENTION_VJP_VALUE;
  for (size_t i = 0; i < 4u; ++i)
    if (qp[i] != (int64_t)i || kp[i] != (int64_t)i)
      return ET_M3T_C4_ATTENTION_VJP_VALUE;
  for (size_t i = 0; i < 16u; ++i)
    if (mask[i] > 1u) return ET_M3T_C4_ATTENTION_VJP_VALUE;

  float dq[16] = {0}, dk[16] = {0}, dv[16] = {0};
  const float scale = 1.0f / sqrtf(2.0f);
  for (size_t h = 0; h < 2u; ++h) {
    for (size_t t = 0; t < 4u; ++t) {
      float score[4] = {0}, p[4] = {0}, gp[4] = {0};
      float maximum = 0.0f, denominator = 0.0f, weighted = 0.0f;
      int any = 0;
      for (size_t j = 0; j < 4u; ++j) {
        if (!mask[t * 4u + j] || kp[j] > qp[t]) continue;
        float dot = 0.0f;
        for (size_t d = 0; d < 2u; ++d)
          dot += q[head_index(h, t, d)] * k[head_index(h, j, d)];
        score[j] = dot * scale;
        if (!isfinite(score[j])) return ET_M3T_C4_ATTENTION_VJP_VALUE;
        if (!any || score[j] > maximum) maximum = score[j];
        any = 1;
      }
      if (!any) continue;
      for (size_t j = 0; j < 4u; ++j) {
        if (!mask[t * 4u + j] || kp[j] > qp[t]) continue;
        p[j] = expf(score[j] - maximum);
        denominator += p[j];
      }
      if (!isfinite(denominator) || denominator <= 0.0f)
        return ET_M3T_C4_ATTENTION_VJP_VALUE;
      for (size_t j = 0; j < 4u; ++j) {
        if (!mask[t * 4u + j] || kp[j] > qp[t]) continue;
        p[j] /= denominator;
        for (size_t d = 0; d < 2u; ++d) {
          gp[j] += u[head_index(h, t, d)] * v[head_index(h, j, d)];
          dv[head_index(h, j, d)] += p[j] * u[head_index(h, t, d)];
        }
        weighted += p[j] * gp[j];
      }
      for (size_t j = 0; j < 4u; ++j) {
        if (!mask[t * 4u + j] || kp[j] > qp[t]) continue;
        const float ds = p[j] * (gp[j] - weighted) * scale;
        for (size_t d = 0; d < 2u; ++d) {
          dq[head_index(h, t, d)] += ds * k[head_index(h, j, d)];
          dk[head_index(h, j, d)] += ds * q[head_index(h, t, d)];
        }
      }
    }
  }
  for (size_t i = 0; i < 16u; ++i)
    if (!isfinite(dq[i]) || !isfinite(dk[i]) || !isfinite(dv[i]))
      return ET_M3T_C4_ATTENTION_VJP_VALUE;
  memcpy(query_gradient->data, dq, sizeof(dq));
  memcpy(key_gradient->data, dk, sizeof(dk));
  memcpy(value_gradient->data, dv, sizeof(dv));
  return ET_M3T_C4_ATTENTION_VJP_OK;
}
