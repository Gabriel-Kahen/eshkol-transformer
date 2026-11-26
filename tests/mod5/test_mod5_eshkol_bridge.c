#include "mod5_eshkol_bridge.h"
#include "mod5_test.h"

#include <float.h>

/* K1 owns heap-backed discovery metadata. Dispatch may use none of the
 * allocator entry points imported by K1. The provider has its own nm audit. */
static int dispatching;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!dispatching);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!dispatching);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!dispatching);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!dispatching);
  __real_free(ptr);
}

int main(void) {
  et_kernel_runtime *runtime = NULL, *baseline = NULL;
  et_kernel_error error;
  CHECK(et_mod5_aot_open(NULL) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_aot_open(&runtime) == 0 && runtime != NULL);
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);

  float x[128], gamma[16], dy[256], k[64] = {0}, v[64], slopes[4] = {0};
  float gate[256], up[256], y[256], dx[256], dg[256], dv[64];
  float epsilon = 0.125f, bad_epsilon = -1.0f, overflow;
  et_mod5_aot_store_f32(&overflow, 0, DBL_MAX);
  int64_t qpos[8], kpos[8];
  uint8_t mask[64];
  memset(mask, 1, sizeof(mask));
  for (size_t i = 0; i < ARRAY_COUNT(dy); ++i) {
    dy[i] = 1.0f;
    gate[i] = 0.5f;
    up[i] = 2.0f;
  }
  for (size_t i = 0; i < ARRAY_COUNT(x); ++i)
    et_mod5_aot_store_f32(x, (int64_t)i, 1.0);
  for (size_t i = 0; i < ARRAY_COUNT(gamma); ++i)
    gamma[i] = 2.0f;
  for (size_t i = 0; i < ARRAY_COUNT(v); ++i)
    v[i] = 0.75f;
  for (size_t i = 0; i < ARRAY_COUNT(qpos); ++i)
    qpos[i] = kpos[i] = (int64_t)i;

  dispatching = 1;
  CHECK(et_mod5_aot_rms_forward(runtime, x, gamma, &epsilon, y) == 0);
  check_close(et_mod5_aot_load_f32(y, 127), 2.0 / sqrt(1.125), 1e-6, 0);
  CHECK(et_mod5_aot_rms_backward(runtime, x, gamma, &epsilon, dy, dx, dg) == 0);
  check_close(dg[15], 8.0 / sqrt(1.125), 1e-6, 0);
  CHECK(et_mod5_aot_swiglu_forward(runtime, gate, up, y) == 0);
  check_close(y[255], 1.0 / (1.0 + exp(-0.5)), 1e-6, 0);
  CHECK(et_mod5_aot_swiglu_backward(runtime, gate, up, dy, dx, dg) == 0);
  check_close(dg[255], 0.5 / (1.0 + exp(-0.5)), 1e-6, 0);
  for (int64_t heads = 1; heads <= 2; ++heads) {
    CHECK(et_mod5_aot_attention_forward(runtime, heads, x, k, v, qpos, kpos,
                                        mask, slopes, 1, y) == 0);
    CHECK(y[127] == 0.75f);
    CHECK(et_mod5_aot_attention_backward(runtime, heads, x, k, v, qpos, kpos,
                                         mask, slopes, 1, dy, dx, dg, dv) == 0);
    CHECK(dx[127] == 0.0f && dg[32 * heads - 1] == 0.0f);
    CHECK(dv[32 * heads - 1] == (float)(4 / heads));
  }

  memset(y, 0xa5, sizeof(y));
  unsigned char before[sizeof(y)];
  memcpy(before, y, sizeof(y));
  CHECK(et_mod5_aot_rms_forward(baseline, x, gamma, &epsilon, y) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mod5_aot_rms_forward(runtime, x, gamma, &bad_epsilon, y) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_aot_rms_forward(runtime, x, gamma, &overflow, y) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_aot_swiglu_forward(runtime, gate, up, NULL) ==
        ET_KERNEL_ERROR_SHAPE_MISMATCH);
  CHECK(et_mod5_aot_swiglu_backward(runtime, gate, up, dy, y, y) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mod5_aot_attention_forward(runtime, 3, x, k, v, qpos, kpos, mask,
                                      slopes, 1,
                                      y) == ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mod5_aot_attention_forward(runtime, 2, x, k, v, qpos, kpos, mask,
                                      slopes, -1,
                                      y) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(y, before, sizeof(y)) == 0);
  dispatching = 0;

  et_mod5_aot_close(baseline);
  et_mod5_aot_close(runtime);
  puts("MOD5 AOT bridge ASan/UBSan/LSan and allocation-free dispatch: PASS");
  return 0;
}
