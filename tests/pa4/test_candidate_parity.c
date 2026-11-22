#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/pa4_attention_candidate.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "PA4 FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                      const char *dtype, size_t rank,
                                      const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(result), .data = data, .byte_length = bytes,
      .dtype = dtype, .device = "cpu", .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = rank, .shape = shape};
  return result;
}

typedef enum fixture {
  regular, fully_masked, score_overflow, dp_overflow, dq_overflow,
  dk_overflow, dv_overflow, wider_keys
} fixture;

static void run_shape(const uint64_t ext[6], fixture variant) {
  const size_t n = (size_t)ext[0], hq = (size_t)ext[1];
  const size_t hkv = (size_t)ext[2], tq = (size_t)ext[3];
  const size_t tk = (size_t)ext[4], dh = (size_t)ext[5];
  const size_t q_count = n * hq * tq * dh;
  const size_t kv_count = n * hkv * tk * dh;
  const size_t mask_count = n * tq * tk;
  uint64_t qs[4] = {ext[0], ext[1], ext[3], ext[5]};
  uint64_t ks[4] = {ext[0], ext[2], ext[4], ext[5]};
  uint64_t qps[2] = {ext[0], ext[3]}, kps[2] = {ext[0], ext[4]};
  uint64_t ms[3] = {ext[0], ext[3], ext[4]};
  float q[96], k[96], v[96], upstream[96];
  float results[2][4][96];
  float sentinel[96];
  const float zero[96] = {0};
  int64_t qp[8], kp[8];
  uint8_t mask[24];
  et_kernel_tensor_view_v1 inputs[7], outputs[3];
  et_kernel_request_v1 req = {
      .struct_size = sizeof(req), .dtype = "f32", .device = "cpu",
      .rank = 6u, .shape = ext, .deterministic = 1u};
  et_kernel_call_v1 call = {
      .struct_size = sizeof(call), .capability = "kernel.causal-attention",
      .request = &req, .input_count = 6u,
      .input_stride = sizeof(et_kernel_tensor_view_v1),
      .input_bytes = 6u * sizeof(et_kernel_tensor_view_v1), .inputs = inputs,
      .output_count = 1u, .output_stride = sizeof(et_kernel_tensor_view_v1),
      .output_bytes = sizeof(et_kernel_tensor_view_v1), .outputs = outputs};
  const et_kernel_provider_v1 *providers[2] = {
      et_a2_kernel_provider_v1(), et_pa4_kernel_provider_v1()};
  et_kernel_error error;
  CHECK(q_count <= 96u && kv_count <= 96u && n * tq <= 8u &&
        n * tk <= 8u && mask_count <= 24u);
  for (size_t i = 0; i < q_count; i++) {
    q[i] = (float)((int)(i % 13u) - 6) * 0.17f;
    upstream[i] = (float)((int)(i % 11u) - 5) * 0.13f;
  }
  for (size_t i = 0; i < kv_count; i++) {
    k[i] = (float)((int)(i % 17u) - 8) * 0.11f;
    v[i] = (float)((int)(i % 19u) - 9) * 0.09f;
  }
  for (size_t b = 0; b < n; b++) {
    for (size_t t = 0; t < tq; t++) qp[b * tq + t] = (int64_t)(b + t + 1u);
    for (size_t t = 0; t < tk; t++) kp[b * tk + t] = (int64_t)(b + t);
  }
  for (size_t i = 0; i < mask_count; i++) mask[i] = (uint8_t)(i % 5u != 2u);
  if (variant == fully_masked) {
    memset(mask, 0, mask_count);
    q[0] = k[0] = v[0] = upstream[0] = FLT_MAX;
  } else if (variant == score_overflow) {
    q[0] = FLT_MAX;
    k[0] = 2.0f;
  } else if (variant == dp_overflow) {
    v[0] = FLT_MAX;
    upstream[0] = 2.0f;
  } else if (variant == dq_overflow || variant == dk_overflow ||
             variant == dv_overflow) {
    memset(mask, 1, mask_count);
    memset(q, 0, q_count * sizeof(float));
    memset(k, 0, kv_count * sizeof(float));
    memset(v, 0, kv_count * sizeof(float));
    for (size_t i = 0; i < q_count; i++) {
      q[i] = variant == dk_overflow ? FLT_MAX : 0.0f;
      upstream[i] = variant == dv_overflow ? 0.0f : 2.0f;
    }
    if (variant == dv_overflow) {
      /* Overflow the accumulation in the final dimension of a wider tile. */
      for (size_t i = dh - 1; i < q_count; i += dh) upstream[i] = FLT_MAX;
    } else {
      v[0] = 1.0f;
      v[1] = -1.0f;
      if (variant == dq_overflow) {
        k[0] = FLT_MAX;
        k[1] = -FLT_MAX;
      }
    }
  }
  inputs[0] = view(q, q_count * sizeof(float), "f32", 4u, qs);
  inputs[1] = view(k, kv_count * sizeof(float), "f32", 4u, ks);
  inputs[2] = view(v, kv_count * sizeof(float), "f32", 4u, ks);
  inputs[3] = view(qp, n * tq * sizeof(int64_t), "i64", 2u, qps);
  inputs[4] = view(kp, n * tk * sizeof(int64_t), "i64", 2u, kps);
  inputs[5] = view(mask, mask_count, "bool", 3u, ms);
  inputs[6] = view(upstream, q_count * sizeof(float), "f32", 4u, qs);
  memset(sentinel, 0x5a, sizeof(sentinel));
  for (size_t backward = 0; backward < 2u; backward++) {
    req.operation = backward ? "causal-attention.backward" :
                               "causal-attention.forward";
    call.input_count = 6u + backward;
    call.input_bytes = call.input_count * call.input_stride;
    call.output_count = backward ? 3u : 1u;
    call.output_bytes = call.output_count * call.output_stride;
    const int overflow = variant == score_overflow ||
        (backward && variant >= dp_overflow && variant <= dv_overflow);
    for (size_t p = 0; p < 2u; p++) {
      float (*result)[96] = results[p] + backward;
      outputs[0] = view(result[0], q_count * sizeof(float), "f32", 4u, qs);
      if (backward) {
        outputs[1] = view(result[1], kv_count * sizeof(float), "f32", 4u, ks);
        outputs[2] = view(result[2], kv_count * sizeof(float), "f32", 4u, ks);
      }
      for (size_t i = 0; i < call.output_count; i++) {
        memcpy(outputs[i].data, sentinel, outputs[i].byte_length);
      }
      const int wider = p == 1u && variant == wider_keys;
      const int status = providers[p]->validate_call(&call, &error);
      if (overflow || wider) {
        CHECK(status == (wider ? ET_KERNEL_ERROR_UNSUPPORTED :
                                ET_KERNEL_ERROR_INVALID_ARGUMENT));
        CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
        for (size_t i = 0; i < call.output_count; i++) {
          CHECK(memcmp(outputs[i].data, sentinel, outputs[i].byte_length) == 0);
        }
      } else {
        CHECK(status == 0);
        providers[p]->invoke_call(&call);
        for (size_t i = 0; i < call.output_count; i++) {
          if (variant == fully_masked) {
            CHECK(memcmp(outputs[i].data, zero, outputs[i].byte_length) == 0);
          }
          if (p == 1u) {
            CHECK(memcmp(result[i], results[0][backward + i],
                         outputs[i].byte_length) == 0);
          }
        }
      }
    }
  }
}

int main(void) {
  static const uint64_t shapes[][6] = {
      {1,2,2,1,1,1}, {1,2,2,2,2,1}, {1,2,2,1,2,1},
      {1,2,2,1,2,2}, {1,2,2,2,2,2}, {1,4,2,1,1,2},
      {1,4,2,3,3,2}, {2,4,2,2,3,4}, {2,4,2,3,3,2},
      {2,4,2,1,3,2}};
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    run_shape(shapes[i], regular);
  run_shape(shapes[8], fully_masked);
  run_shape(shapes[0], score_overflow);
  run_shape(shapes[0], dp_overflow);
  const uint64_t grouped[] = {1,4,2,2,2,1};
  const uint64_t wide_head[] = {1,4,2,2,2,5};
  const uint64_t wide_keys[] = {1,4,2,2,4,2};
  run_shape(grouped, dq_overflow);
  run_shape(grouped, dk_overflow);
  run_shape(wide_head, dv_overflow);
  run_shape(wide_head, regular);
  run_shape(wide_keys, wider_keys);
  puts("PA4 candidate bitwise parity PASS (10 advertised shapes and direct-call edge cases)");
  return 0;
}
