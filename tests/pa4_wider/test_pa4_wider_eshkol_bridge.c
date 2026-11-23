#include "eshkol_transformer/pa4_wider_attention_abi.h"
#include "pa4_wider_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tracking;
static size_t heap_calls;

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);

void *__wrap_malloc(size_t size) {
  heap_calls += tracking;
  return __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size) {
  heap_calls += tracking;
  return __real_calloc(count, size);
}

void *__wrap_realloc(void *ptr, size_t size) {
  heap_calls += tracking;
  return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr) {
  heap_calls += tracking;
  __real_free(ptr);
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum { MAX_ELEMENTS = 2 * 4 * 64 * 16 };
static float q[MAX_ELEMENTS], k[MAX_ELEMENTS], v[MAX_ELEMENTS];
static float dout[MAX_ELEMENTS], out[MAX_ELEMENTS], dq[MAX_ELEMENTS];
static float dk[MAX_ELEMENTS], dv[MAX_ELEMENTS];
static int64_t qpos[2 * 64], kpos[2 * 64];
static uint8_t mask[2 * 64 * 64];

static void check_marshaling(void) {
  float buf[2];
  int64_t positions[2];
  uint8_t masks[2];
  et_pa4_wider_aot_store_f32(buf, 0, 1.0 + 0x1p-24);
  et_pa4_wider_aot_store_f32(buf, 1, -0.0);
  CHECK(et_pa4_wider_aot_load_f32(buf, 0) == 1.0);
  CHECK(et_pa4_wider_aot_f32_bits(buf, 0) == INT64_C(0x3f800000));
  CHECK(et_pa4_wider_aot_f32_bits(buf, 1) == INT64_C(0x80000000));
  et_pa4_wider_aot_store_f32(buf, 0, 0x1p-149);
  CHECK(et_pa4_wider_aot_load_f32(buf, 0) == 0x1p-149);
  CHECK(et_pa4_wider_aot_f32_bits(buf, 0) == 1);
  et_pa4_wider_aot_store_f32(buf, 1, -DBL_MAX);
  CHECK(et_pa4_wider_aot_f32_bits(buf, 1) == INT64_C(0xff800000));
  et_pa4_wider_aot_store_i64(positions, 1, INT64_MAX);
  et_pa4_wider_aot_store_u8(masks, 1, UINT8_MAX);
  CHECK(positions[1] == INT64_MAX && masks[1] == UINT8_MAX);
}

static void check_shape(et_kernel_runtime *runtime, int n, int hq, int hkv,
                        int tq, int tk, int dh) {
  const int group = hq / hkv;
  for (int b = 0; b < n; ++b) {
    for (int i = 0; i < tq; ++i) {
      qpos[b * tq + i] = 2 * i + 1;
      memset(mask + (b * tq + i) * tk, i != 0 || tq == 1, (size_t)tk);
    }
    for (int j = 0; j < tk; ++j)
      kpos[b * tk + j] = j;
    for (int h = 0; h < hkv; ++h)
      for (int l = 0; l < tk * dh; ++l)
        v[(b * hkv + h) * tk * dh + l] = 0.125f * (1 + b * hkv + h);
  }
  for (int i = 0; i < n * hq * tq * dh; ++i)
    dout[i] = 0.25f;

  CHECK(et_pa4_wider_aot_forward(runtime, n, hq, hkv, tq, tk, dh, q, k, v, qpos,
                                 kpos, mask, out) == 0);
  CHECK(et_pa4_wider_aot_backward(runtime, n, hq, hkv, tq, tk, dh, q, k, v,
                                  qpos, kpos, mask, dout, dq, dk, dv) == 0);
  for (int b = 0; b < n; ++b) {
    for (int h = 0; h < hq; ++h) {
      const float expected = 0.125f * (1 + b * hkv + h / group);
      for (int i = 0; i < tq; ++i) {
        for (int d = 0; d < dh; ++d) {
          const int offset = ((b * hq + h) * tq + i) * dh + d;
          if (i == 0 && tq > 1)
            CHECK(et_pa4_wider_aot_f32_bits(out, offset) == 0);
          else
            CHECK(fabsf(out[offset] - expected) <= 2e-6f);
          CHECK(dq[offset] == 0.0f);
        }
      }
    }
    for (int h = 0; h < hkv; ++h) {
      for (int d = 0; d < dh; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < tk; ++j) {
          const int offset = ((b * hkv + h) * tk + j) * dh + d;
          CHECK(dk[offset] == 0.0f && dv[offset] >= 0.0f);
          sum += dv[offset];
        }
        const int active_rows = tq == 1 ? 1 : tq - 1;
        CHECK(fabsf(sum - 0.25f * group * active_rows) <= 3e-5f);
      }
    }
  }
}

static void expect_sentinel(const float *buffer, size_t count) {
  const unsigned char *bytes = (const unsigned char *)buffer;
  for (size_t i = 0; i < count * sizeof(*buffer); ++i)
    CHECK(bytes[i] == 0xa5);
}

static void check_bad_calls(et_kernel_runtime *runtime, int variant) {
  qpos[0] = kpos[0] = 0;
  qpos[1] = kpos[1] = 1;
  memset(mask, 1, 4);
  memset(out, 0xa5, 8 * sizeof(*out));
  memset(dq, 0xa5, 8 * sizeof(*dq));
  memset(dk, 0xa5, 4 * sizeof(*dk));
  memset(dv, 0xa5, 4 * sizeof(*dv));

  const int invalid = ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const int bad_shape = ET_KERNEL_ERROR_UNSUPPORTED;
  et_kernel_runtime *unchanged = runtime;
  CHECK(et_pa4_wider_aot_open(-1, &unchanged) == invalid);
  CHECK(unchanged == runtime);
  CHECK(et_pa4_wider_aot_open(variant, NULL) == invalid);
  _Alignas(et_kernel_runtime *) unsigned char cell[sizeof(runtime) + 1] = {0};
  CHECK(et_pa4_wider_aot_open(variant, (void *)(cell + 1)) == invalid);
  CHECK(et_pa4_wider_aot_forward(NULL, 1, 2, 1, 2, 2, 2, q, k, v, qpos, kpos,
                                 mask, out) == invalid);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, -1, 2, 2, q, k, v, qpos,
                                 kpos, mask, out) == bad_shape);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, INT64_MAX, 2, q, k, v,
                                 qpos, kpos, mask, out) == bad_shape);
  CHECK(et_pa4_wider_aot_backward(runtime, 1, 3, 2, 2, 2, 2, q, k, v, qpos,
                                  kpos, mask, dout, dq, dk, dv) == bad_shape);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, 2, 2, NULL, k, v, qpos,
                                 kpos, mask,
                                 out) == ET_KERNEL_ERROR_SHAPE_MISMATCH);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, 2, 2, q, k, k, qpos, kpos,
                                 mask, out) == invalid);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos, kpos,
                                 mask, q) == invalid);
  CHECK(et_pa4_wider_aot_backward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos,
                                  kpos, mask, dout, dq, dk, dk) == invalid);
  float *misaligned = (float *)((unsigned char *)dq + 1);
  CHECK(et_pa4_wider_aot_backward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos,
                                  kpos, mask, dout, misaligned, dk,
                                  dv) == invalid);

  qpos[1] = 0;
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos, kpos,
                                 mask, out) == invalid);
  qpos[1] = 1;
  mask[3] = 2;
  CHECK(et_pa4_wider_aot_backward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos,
                                  kpos, mask, dout, dq, dk, dv) == invalid);
  mask[3] = 1;
  dout[7] = NAN;
  CHECK(et_pa4_wider_aot_backward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos,
                                  kpos, mask, dout, dq, dk, dv) == invalid);
  dout[7] = 0.25f;
  CHECK(fesetround(FE_DOWNWARD) == 0);
  CHECK(et_pa4_wider_aot_forward(runtime, 1, 2, 1, 2, 2, 2, q, k, v, qpos, kpos,
                                 mask, out) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(fesetround(FE_TONEAREST) == 0);

  expect_sentinel(out, 8);
  expect_sentinel(dq, 8);
  expect_sentinel(dk, 4);
  expect_sentinel(dv, 4);
  for (size_t i = 0; i < 8; ++i)
    CHECK(q[i] == 0.0f);

  /* Equal scores and near-FLT_MAX values overflow the online numerator.
   * This distinguishes the selected provider and detects a fallback.
   */
  qpos[0] = 1;
  v[0] = v[1] = FLT_MAX;
  const int status = et_pa4_wider_aot_forward(runtime, 1, 1, 1, 1, 2, 1, q, k,
                                              v, qpos, kpos, mask, out);
  if (variant == 0)
    CHECK(status == 0 && out[0] == FLT_MAX);
  else {
    CHECK(status == invalid);
    expect_sentinel(out, 8);
  }
}

int main(void) {
  check_marshaling();
  for (int variant = 0; variant <= 1; ++variant) {
    et_kernel_runtime *runtime = NULL;
    CHECK(et_pa4_wider_aot_open(variant, &runtime) == 0 && runtime != NULL);
    const et_kernel_provider_v1 *provider =
        variant == 0 ? et_pa4_wider_ref_provider_v1()
                     : et_pa4_wider_online_provider_v1();
    const et_kernel_capability_v1 *provided = provider->capabilities;
    const et_kernel_capability_v1 *cap =
        et_kernel_runtime_capability_find(runtime, "kernel.causal-attention");
    CHECK(cap != NULL &&
          strcmp(cap->implementation, provided->implementation) == 0);
    tracking = 1;
    check_shape(runtime, 1, 1, 1, 1, 1, 1);
    for (int hq = 1; hq <= 4; ++hq)
      for (int hkv = 1; hkv <= hq; ++hkv)
        if (hq % hkv == 0)
          check_shape(runtime, 2, hq, hkv, 5, 7, 3);
    check_shape(runtime, 2, 4, 4, 64, 64, 16);
    check_bad_calls(runtime, variant);
    tracking = 0;
    et_pa4_wider_aot_close(runtime);
  }
  CHECK(heap_calls == 0);
  puts("PA4-Wider AOT C11 bridge, malformed statuses and zero-allocation "
       "dispatch: PASS");
  return 0;
}
