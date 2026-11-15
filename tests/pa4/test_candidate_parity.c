#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/pa4_attention_candidate.h"

#include <math.h>
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

static void run_shape(const uint64_t ext[6]) {
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
  float base_out[96], pa4_out[96], base_dq[96], pa4_dq[96];
  float base_dk[96], pa4_dk[96], base_dv[96], pa4_dv[96];
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
  inputs[0] = view(q, q_count * sizeof(float), "f32", 4u, qs);
  inputs[1] = view(k, kv_count * sizeof(float), "f32", 4u, ks);
  inputs[2] = view(v, kv_count * sizeof(float), "f32", 4u, ks);
  inputs[3] = view(qp, n * tq * sizeof(int64_t), "i64", 2u, qps);
  inputs[4] = view(kp, n * tk * sizeof(int64_t), "i64", 2u, kps);
  inputs[5] = view(mask, mask_count, "bool", 3u, ms);
  inputs[6] = view(upstream, q_count * sizeof(float), "f32", 4u, qs);
  for (size_t p = 0; p < 2u; p++) {
    float *out = p == 0u ? base_out : pa4_out;
    float *dq = p == 0u ? base_dq : pa4_dq;
    float *dk = p == 0u ? base_dk : pa4_dk;
    float *dv = p == 0u ? base_dv : pa4_dv;
    req.operation = "causal-attention.forward";
    outputs[0] = view(out, q_count * sizeof(float), "f32", 4u, qs);
    CHECK(providers[p]->validate_call(&call, &error) == 0);
    providers[p]->invoke_call(&call);
    req.operation = "causal-attention.backward";
    call.input_count = 7u;
    call.input_bytes = 7u * sizeof(et_kernel_tensor_view_v1);
    call.output_count = 3u;
    call.output_bytes = 3u * sizeof(et_kernel_tensor_view_v1);
    outputs[0] = view(dq, q_count * sizeof(float), "f32", 4u, qs);
    outputs[1] = view(dk, kv_count * sizeof(float), "f32", 4u, ks);
    outputs[2] = view(dv, kv_count * sizeof(float), "f32", 4u, ks);
    CHECK(providers[p]->validate_call(&call, &error) == 0);
    providers[p]->invoke_call(&call);
    call.input_count = 6u;
    call.input_bytes = 6u * sizeof(et_kernel_tensor_view_v1);
    call.output_count = 1u;
    call.output_bytes = sizeof(et_kernel_tensor_view_v1);
  }
  CHECK(memcmp(base_out, pa4_out, q_count * sizeof(float)) == 0);
  CHECK(memcmp(base_dq, pa4_dq, q_count * sizeof(float)) == 0);
  CHECK(memcmp(base_dk, pa4_dk, kv_count * sizeof(float)) == 0);
  CHECK(memcmp(base_dv, pa4_dv, kv_count * sizeof(float)) == 0);
  /* Direct-provider rejection is explicit even before capability resolution. */
  uint64_t wider[6] = {ext[0], ext[1], ext[2], ext[3], 4u, ext[5]};
  float sentinel[96];
  memcpy(sentinel, pa4_dq, q_count * sizeof(float));
  req.shape = wider;
  req.operation = "causal-attention.forward";
  CHECK(providers[1]->validate_call(&call, &error) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(pa4_dq, sentinel, q_count * sizeof(float)) == 0);
}

int main(void) {
  static const uint64_t shapes[][6] = {
      {1,2,2,1,1,1}, {1,2,2,2,2,1}, {1,2,2,1,2,1},
      {1,2,2,1,2,2}, {1,2,2,2,2,2}, {1,4,2,1,1,2},
      {1,4,2,3,3,2}, {2,4,2,2,3,4}, {2,4,2,3,3,2},
      {2,4,2,1,3,2}};
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
    run_shape(shapes[i]);
  puts("PA4 candidate bitwise parity PASS (10 advertised shapes, forward/backward)");
  return 0;
}
