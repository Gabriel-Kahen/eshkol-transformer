#ifndef PA4_WIDER_TEST_FIXTURE_H
#define PA4_WIDER_TEST_FIXTURE_H

#include "eshkol_transformer/pa4_wider_attention_abi.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { REF, ONLINE, PROVIDERS };
enum { FORWARD, BACKWARD, JOINT };
enum { OUTPUT, DQ, DK, DV, OUTPUTS };
enum { Q, K, V, QP, KP, MASK, UPSTREAM };
enum { MAX_TOKENS = 64, MAX_ELEMENTS = 2 * 4 * MAX_TOKENS * 16 };

static const uint64_t workload_shapes[][6] = {
    {1, 4, 2, 8, 8, 8},    {1, 4, 2, 16, 16, 16}, {1, 4, 2, 32, 32, 16},
    {1, 4, 2, 64, 64, 16}, {2, 4, 2, 16, 32, 16}, {1, 4, 1, 32, 32, 16}};

typedef struct {
  uint64_t shape[6], q_shape[4], kv_shape[4], qp_shape[2], kp_shape[2],
      mask_shape[3];
  size_t q_count, kv_count;
  float q[MAX_ELEMENTS], k[MAX_ELEMENTS], v[MAX_ELEMENTS],
      upstream[MAX_ELEMENTS];
  int64_t qp[2 * MAX_TOKENS], kp[2 * MAX_TOKENS];
  uint8_t mask[2 * MAX_TOKENS * MAX_TOKENS];
  float result[OUTPUTS][MAX_ELEMENTS];
  et_kernel_tensor_view_v1 in[7], out[OUTPUTS];
  et_kernel_request_v1 request[2];
  et_kernel_call_v1 call[2];
} fixture;

typedef struct {
  double absolute;
  double relative;
  uint32_t ulps;
} error_metrics;

static inline void require(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "PA4-Wider FAIL: %s\n", message);
    exit(EXIT_FAILURE);
  }
}

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0 ? context : NULL;
}

static inline const et_kernel_provider_v1 *provider(int variant) {
  return variant == REF ? et_pa4_wider_ref_provider_v1()
                        : et_pa4_wider_online_provider_v1();
}

static inline et_kernel_runtime *discover(int variant) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  const int32_t status = et_kernel_runtime_discover(
      resolve, (void *)provider(variant), &runtime, &error);
  if (status != 0)
    fprintf(stderr, "%s: %s\n", error.operation, error.message);
  require(status == 0 && runtime != NULL, "explicit K1 provider discovery");
  return runtime;
}

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                     const char *dtype, size_t rank,
                                     const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){.struct_size =
                                        sizeof(et_kernel_tensor_view_v1),
                                    .data = data,
                                    .byte_length = bytes,
                                    .dtype = dtype,
                                    .device = "cpu",
                                    .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    .rank = rank,
                                    .shape = shape};
}

static float sample_value(size_t index, uint32_t seed) {
  uint32_t bits = ((uint32_t)index + 1u) * UINT32_C(0x9e3779b1) + seed;
  bits = (bits ^ (bits >> 16u)) * UINT32_C(0x85ebca6b);
  bits = (bits ^ (bits >> 13u)) * UINT32_C(0xc2b2ae35);
  return (float)((int32_t)((bits ^ (bits >> 16u)) % 257u) - 128) / 128.0f;
}

static inline void init_fixture(fixture *f, const uint64_t *shape) {
  memset(f, 0, sizeof(*f));
  memcpy(f->shape, shape, sizeof(f->shape));
  const size_t n = shape[0], hq = shape[1], hkv = shape[2];
  const size_t tq = shape[3], tk = shape[4], dh = shape[5];
  memcpy(f->q_shape, (uint64_t[]){n, hq, tq, dh}, sizeof(f->q_shape));
  memcpy(f->kv_shape, (uint64_t[]){n, hkv, tk, dh}, sizeof(f->kv_shape));
  memcpy(f->qp_shape, (uint64_t[]){n, tq}, sizeof(f->qp_shape));
  memcpy(f->kp_shape, (uint64_t[]){n, tk}, sizeof(f->kp_shape));
  memcpy(f->mask_shape, (uint64_t[]){n, tq, tk}, sizeof(f->mask_shape));
  f->q_count = n * hq * tq * dh;
  f->kv_count = n * hkv * tk * dh;
  for (size_t i = 0; i < f->q_count; ++i) {
    f->q[i] = sample_value(i, 17u);
    f->upstream[i] = sample_value(i, 29u);
  }
  for (size_t i = 0; i < f->kv_count; ++i) {
    f->k[i] = sample_value(i, 37u);
    f->v[i] = sample_value(i, 53u);
  }
  for (size_t b = 0; b < n; ++b) {
    const size_t prefix = tk > tq ? tk - tq : 0u;
    for (size_t i = 0; i < tq; ++i)
      f->qp[b * tq + i] = (int64_t)(67u * b + 3u * (i + prefix) + 1u);
    for (size_t j = 0; j < tk; ++j)
      f->kp[b * tk + j] = (int64_t)(67u * b + 3u * j + 1u);
    for (size_t i = 0; i < tq; ++i)
      for (size_t j = 0; j < tk; ++j)
        f->mask[(b * tq + i) * tk + j] =
            (uint8_t)(i != 3u && (i + 2u * j + 3u * b) % 11u != 7u);
  }
  const size_t q_bytes = f->q_count * sizeof(float);
  const size_t kv_bytes = f->kv_count * sizeof(float);
  f->in[Q] = view(f->q, q_bytes, "f32", 4u, f->q_shape);
  f->in[K] = view(f->k, kv_bytes, "f32", 4u, f->kv_shape);
  f->in[V] = view(f->v, kv_bytes, "f32", 4u, f->kv_shape);
  f->in[QP] = view(f->qp, n * tq * sizeof(int64_t), "i64", 2u, f->qp_shape);
  f->in[KP] = view(f->kp, n * tk * sizeof(int64_t), "i64", 2u, f->kp_shape);
  f->in[MASK] = view(f->mask, n * tq * tk, "bool", 3u, f->mask_shape);
  f->in[UPSTREAM] = view(f->upstream, q_bytes, "f32", 4u, f->q_shape);
  f->out[OUTPUT] = view(f->result[OUTPUT], q_bytes, "f32", 4u, f->q_shape);
  f->out[DQ] = view(f->result[DQ], q_bytes, "f32", 4u, f->q_shape);
  f->out[DK] = view(f->result[DK], kv_bytes, "f32", 4u, f->kv_shape);
  f->out[DV] = view(f->result[DV], kv_bytes, "f32", 4u, f->kv_shape);

  for (int op = FORWARD; op <= BACKWARD; ++op) {
    f->request[op] = (et_kernel_request_v1){
        .struct_size = sizeof(et_kernel_request_v1),
        .operation = op == FORWARD ? "causal-attention.forward"
                                   : "causal-attention.backward",
        .dtype = "f32",
        .device = "cpu",
        .rank = 6u,
        .shape = f->shape,
        .deterministic = 1u};
    const size_t inputs = op == FORWARD ? 6u : 7u;
    const size_t outputs = op == FORWARD ? 1u : 3u;
    f->call[op] =
        (et_kernel_call_v1){.struct_size = sizeof(et_kernel_call_v1),
                            .capability = "kernel.causal-attention",
                            .request = &f->request[op],
                            .input_count = inputs,
                            .input_stride = sizeof(*f->in),
                            .input_bytes = inputs * sizeof(*f->in),
                            .inputs = f->in,
                            .output_count = outputs,
                            .output_stride = sizeof(*f->out),
                            .output_bytes = outputs * sizeof(*f->out),
                            .outputs = f->out + (op == FORWARD ? OUTPUT : DQ)};
  }
}

static inline void dispatch(const et_kernel_runtime *runtime,
                            const et_kernel_call_v1 *call) {
  et_kernel_error error;
  const int32_t status = et_kernel_runtime_dispatch(runtime, call, &error);
  if (status != 0)
    fprintf(stderr, "%s: %s\n", error.operation, error.message);
  require(status == 0, "K1 dispatch");
}

static uint32_t ordered_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
  return (bits >> 31u) ? UINT32_C(0x80000000) - magnitude
                       : UINT32_C(0x80000000) + magnitude;
}

static inline int compare(float ref, float online, error_metrics *metrics) {
  if (!isfinite(ref) || !isfinite(online))
    return 0;
  const double absolute = fabs((double)ref - online);
  const double scale = fmax(fabs((double)ref), fabs((double)online));
  const double relative = scale != 0.0 ? absolute / scale : 0.0;
  const uint32_t a = ordered_bits(ref), b = ordered_bits(online);
  const uint32_t ulps = a > b ? a - b : b - a;
  if (absolute > metrics->absolute)
    metrics->absolute = absolute;
  if (relative > metrics->relative)
    metrics->relative = relative;
  if (ulps > metrics->ulps)
    metrics->ulps = ulps;
  return absolute <= 3e-7 + 3e-6 * scale;
}

#endif
