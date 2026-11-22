#define _POSIX_C_SOURCE 200809L
#include "eshkol_transformer/a2_attention_abi.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Exact A2-admitted rows only; no general shape or precision fallback. */
static const uint64_t shapes[][6] = {
    {1,2,2,1,1,1}, {1,2,2,2,2,1}, {1,2,2,1,2,1},
    {1,2,2,1,2,2}, {1,2,2,2,2,2}, {1,4,2,1,1,2},
    {1,4,2,3,3,2}, {2,4,2,2,3,4}, {2,4,2,3,3,2},
    {2,4,2,1,3,2}};

static void fail(const char *message) {
  (void)fprintf(stderr, "attention benchmark: %s\n", message);
  exit(1);
}

static unsigned number(const char *text, unsigned maximum) {
  char *end;
  errno = 0;
  if (*text < '0' || *text > '9') fail("invalid unsigned argument");
  unsigned long value = strtoul(text, &end, 10);
  if (errno || *end || value > maximum) fail("argument out of range");
  return (unsigned)value;
}

static uint64_t now_ns(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value)) fail("CLOCK_MONOTONIC unavailable");
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_a2_kernel_provider_v1() : NULL;
}

static et_kernel_tensor_view_v1 view(void *data, size_t bytes, const char *dtype,
                                     size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(result), .data = data, .byte_length = bytes,
      .dtype = dtype, .device = "cpu", .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0, .rank = rank, .shape = shape};
  return result;
}

static void fill(float *values, size_t count, unsigned role) {
  for (size_t i = 0; i < count; ++i)
    values[i] = (float)((int)((i * 7 + role * 3) % 23) - 11) / 16.0f;
}

int main(int argc, char **argv) {
  if (argc != 5) fail("usage: attention-v1 ROW forward|backward WARMUP ITERATIONS");
  unsigned row = number(argv[1], 9), warmup = number(argv[3], 100000);
  unsigned iterations = number(argv[4], 1000000);
  int backward = strcmp(argv[2], "backward") == 0;
  if ((!backward && strcmp(argv[2], "forward")) || !iterations)
    fail("unsupported operation or zero iterations");
  const uint64_t *s = shapes[row];
  size_t n = s[0], hq = s[1], hkv = s[2], tq = s[3], tk = s[4], d = s[5];
  size_t nq = n * hq * tq * d, nk = n * hkv * tk * d;
  uint64_t qs[] = {n,hq,tq,d}, ks[] = {n,hkv,tk,d};
  uint64_t qps[] = {n,tq}, kps[] = {n,tk}, ms[] = {n,tq,tk};
  /* Maximum storage is bounded by the exact table, independent of CLI counts. */
  float q[64], k[64], v[64], upstream[64], y[64], dq[64], dk[64], dv[64];
  int64_t qp[6], kp[6];
  uint8_t mask[18];
  fill(q, nq, 0); fill(k, nk, 1); fill(v, nk, 2); fill(upstream, nq, 3);
  for (size_t b = 0; b < n; ++b) {
    for (size_t j = 0; j < tq; ++j) qp[b*tq+j] = (int64_t)(j+1+b);
    for (size_t j = 0; j < tk; ++j) kp[b*tk+j] = (int64_t)(j+b);
    for (size_t j = 0; j < tq; ++j)
      for (size_t a = 0; a < tk; ++a)
        mask[(b*tq+j)*tk+a] = (uint8_t)(!(tq > 1 && j == tq-1) && (j+a+b)%3 != 1);
  }
  et_kernel_tensor_view_v1 inputs[] = {
      view(q,nq*4,"f32",4,qs), view(k,nk*4,"f32",4,ks),
      view(v,nk*4,"f32",4,ks), view(qp,n*tq*8,"i64",2,qps),
      view(kp,n*tk*8,"i64",2,kps), view(mask,n*tq*tk,"bool",3,ms),
      view(upstream,nq*4,"f32",4,qs)};
  et_kernel_tensor_view_v1 outputs[] = {
      view(backward ? dq : y,nq*4,"f32",4,qs),
      view(dk,nk*4,"f32",4,ks), view(dv,nk*4,"f32",4,ks)};
  et_kernel_request_v1 request = {
      .struct_size = sizeof(request),
      .operation = backward ? "causal-attention.backward" : "causal-attention.forward",
      .dtype = "f32", .device = "cpu", .rank = 6, .shape = s, .deterministic = 1};
  et_kernel_call_v1 call = {
      .struct_size = sizeof(call), .capability = "kernel.causal-attention",
      .request = &request, .input_count = backward ? 7 : 6,
      .input_stride = sizeof(inputs[0]),
      .input_bytes = (backward ? 7 : 6)*sizeof(inputs[0]), .inputs = inputs,
      .output_count = backward ? 3 : 1, .output_stride = sizeof(outputs[0]),
      .output_bytes = (backward ? 3 : 1)*sizeof(outputs[0]), .outputs = outputs};
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  const et_kernel_capability_v1 *entry = NULL;
  if (et_kernel_runtime_discover(resolve,NULL,&runtime,&error) ||
      et_kernel_runtime_capability_require(runtime,call.capability,&request,&entry,&error))
    fail(error.message);
  if (!entry || strcmp(et_a2_kernel_provider_v1()->name,"a2.cpu-f32.serial"))
    fail("unexpected provider");
  for (unsigned i = 0; i < warmup; ++i)
    if (et_kernel_runtime_dispatch(runtime,&call,&error)) fail(error.message);
  uint64_t start = now_ns();
  for (unsigned i = 0; i < iterations; ++i)
    if (et_kernel_runtime_dispatch(runtime,&call,&error)) fail(error.message);
  uint64_t elapsed = now_ns() - start;
  if (!elapsed) fail("nonpositive dispatch duration");
  (void)printf("{\"dispatch_elapsed_ns\":%" PRIu64 ",\"output_bits\":\"",elapsed);
  for (size_t t = 0; t < call.output_count; ++t) {
    const float *values = outputs[t].data;
    for (size_t i = 0; i < outputs[t].byte_length/4; ++i) {
      uint32_t bits;
      if (!isfinite(values[i])) fail("nonfinite output");
      memcpy(&bits,&values[i],4);
      (void)printf("%08" PRIx32,bits);
    }
  }
  (void)puts("\",\"provider\":\"a2.cpu-f32.serial\",\"version\":1}");
  et_kernel_runtime_destroy(runtime);
  return 0;
}
