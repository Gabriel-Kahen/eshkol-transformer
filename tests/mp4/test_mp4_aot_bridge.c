#include "mp4_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
              #cond);                                                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#ifdef ET_MP4_BRIDGE_WRAP_ALLOC
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
#define SET_DISPATCHING(v) (dispatching = (v))
#else
#define SET_DISPATCHING(v) ((void)(v))
#endif

static uint32_t f32_bits(float v) {
  uint32_t u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

static void test_sizing(void) {
  CHECK(et_mp4_aot_element_bytes(ET_MP4_AOT_DTYPE_F32) == 4);
  CHECK(et_mp4_aot_element_bytes(ET_MP4_AOT_DTYPE_F16) == 2);
  CHECK(et_mp4_aot_element_bytes(ET_MP4_AOT_DTYPE_BF16) == 2);
  CHECK(et_mp4_aot_element_bytes(ET_MP4_AOT_DTYPE_BOOL) == 1);
  CHECK(et_mp4_aot_element_bytes(-1) == -1);
  CHECK(et_mp4_aot_element_bytes(99) == -1);

  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_F16, 256) == 512);
  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_BF16, 256) == 512);
  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_F32, 256) == 1024);
  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_F16, 0) == -1);
  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_F16, -8) == -1);
  CHECK(et_mp4_aot_tensor_bytes(ET_MP4_AOT_DTYPE_F32, INT64_MAX) == -1);
  CHECK(et_mp4_aot_scaler_bytes() == (int64_t)sizeof(et_mp4_aot_scaler));
}

static void test_marshal_helpers(void) {
  float fbuf[4] = {0};
  uint16_t u16buf[4] = {0};
  uint8_t u8buf[4] = {0};
  unsigned char raw[32];
  memset(raw, 0xa5, sizeof(raw));
  unsigned char raw_before[32];
  memcpy(raw_before, raw, sizeof(raw));

  /* Valid stores and loads, including -0.0 and out-of-f32-range doubles. */
  CHECK(et_mp4_aot_store_f32(fbuf, 4, 0, -0.0) == 0);
  CHECK(et_mp4_aot_load_f32_bits(fbuf, 4, 0) == INT64_C(0x80000000));
  CHECK(et_mp4_aot_store_f32(fbuf, 4, 1, 1.5) == 0);
  CHECK(et_mp4_aot_load_f32(fbuf, 4, 1) == 1.5);
  CHECK(et_mp4_aot_store_f32(fbuf, 4, 2, DBL_MAX) == 0);
  CHECK(et_mp4_aot_load_f32_bits(fbuf, 4, 2) == INT64_C(0x7f800000));
  CHECK(et_mp4_aot_store_f32(fbuf, 4, 3, -DBL_MAX) == 0);
  CHECK(et_mp4_aot_load_f32_bits(fbuf, 4, 3) == INT64_C(0xff800000));

  CHECK(et_mp4_aot_store_f32_bits(fbuf, 4, 0, INT64_C(0x3f801000)) == 0);
  CHECK(et_mp4_aot_load_f32_bits(fbuf, 4, 0) == INT64_C(0x3f801000));
  CHECK(et_mp4_aot_store_u16(u16buf, 4, 2, 0x7bff) == 0);
  CHECK(et_mp4_aot_load_u16(u16buf, 4, 2) == 0x7bff);
  CHECK(et_mp4_aot_store_u8(u8buf, 4, 3, 1) == 0);
  CHECK(et_mp4_aot_load_u8(u8buf, 4, 3) == 1);

  /* Negative: null, misaligned (via void *), bounds, invalid bit range. */
  void *misaligned1 = raw + 1;
  void *misaligned2 = raw + 2;
  CHECK(et_mp4_aot_store_f32(NULL, 4, 0, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32(misaligned1, 4, 0, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32(misaligned2, 4, 0, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32(fbuf, 0, 0, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32(fbuf, 4, -1, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32(fbuf, 4, 4, 1.0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(isnan(et_mp4_aot_load_f32(NULL, 4, 0)));
  CHECK(isnan(et_mp4_aot_load_f32(misaligned1, 4, 0)));
  CHECK(isnan(et_mp4_aot_load_f32(fbuf, 4, -1)));
  CHECK(isnan(et_mp4_aot_load_f32(fbuf, 4, 4)));

  CHECK(et_mp4_aot_store_f32_bits(misaligned1, 4, 0, 0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32_bits(fbuf, 4, 0, -1) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_f32_bits(fbuf, 4, 0, INT64_C(0x100000000)) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_load_f32_bits(misaligned1, 4, 0) == -1);
  CHECK(et_mp4_aot_load_f32_bits(fbuf, 4, 4) == -1);

  CHECK(et_mp4_aot_store_u16(NULL, 4, 0, 0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u16(misaligned1, 4, 0, 0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u16(u16buf, 4, 4, 0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u16(u16buf, 4, 0, -1) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u16(u16buf, 4, 0, 0x10000) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_load_u16(NULL, 4, 0) == -1);
  CHECK(et_mp4_aot_load_u16(misaligned1, 4, 0) == -1);
  CHECK(et_mp4_aot_load_u16(u16buf, 4, -1) == -1);
  CHECK(et_mp4_aot_load_u16(u16buf, 4, 4) == -1);

  CHECK(et_mp4_aot_store_u8(NULL, 4, 0, 0) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u8(u8buf, 4, 4, 0) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u8(u8buf, 4, 0, -1) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_store_u8(u8buf, 4, 0, 256) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_load_u8(NULL, 4, 0) == -1);
  CHECK(et_mp4_aot_load_u8(u8buf, 4, 4) == -1);

  CHECK(memcmp(raw, raw_before, sizeof(raw)) == 0);
}

static void test_casts_and_negative_boundaries(const et_kernel_runtime *rt,
                                               const et_kernel_runtime *base) {
  static const uint32_t f16_in_bits[8] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0x3f801000u,
      0x3f803000u, 0x477fe000u, 0xbfc00000u, 0x33800000u};
  static const uint16_t f16_exp_narrow[8] = {
      0x0000u, 0x8000u, 0x3c00u, 0x3c00u, 0x3c02u, 0x7bffu, 0xbe00u, 0x0001u};
  static const uint32_t f16_exp_wide[8] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0x3f800000u,
      0x3f804000u, 0x477fe000u, 0xbfc00000u, 0x33800000u};

  static const uint32_t bf16_in_bits[8] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0x3f808000u,
      0x3f818000u, 0x477fe000u, 0xbfc00000u, 0x3f808001u};
  static const uint16_t bf16_exp_narrow[8] = {
      0x0000u, 0x8000u, 0x3f80u, 0x3f80u, 0x3f82u, 0x4780u, 0xbfc0u, 0x3f81u};
  static const uint32_t bf16_exp_wide[8] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0x3f800000u,
      0x3f820000u, 0x47800000u, 0xbfc00000u, 0x3f810000u};

  uint16_t narrow[8] = {0};
  uint32_t wide[8] = {0};

  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, f16_in_bits, narrow) == 0);
  CHECK(memcmp(narrow, f16_exp_narrow, sizeof(narrow)) == 0);
  CHECK(et_mp4_aot_cast_f16_to_f32(rt, 8, narrow, wide) == 0);
  CHECK(memcmp(wide, f16_exp_wide, sizeof(wide)) == 0);

  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, bf16_in_bits, narrow) == 0);
  CHECK(memcmp(narrow, bf16_exp_narrow, sizeof(narrow)) == 0);
  CHECK(et_mp4_aot_cast_bf16_to_f32(rt, 8, narrow, wide) == 0);
  CHECK(memcmp(wide, bf16_exp_wide, sizeof(wide)) == 0);

  /* Negative boundary tests with sentinel preservation. */
  memset(narrow, 0x5a, sizeof(narrow));
  memset(wide, 0x5a, sizeof(wide));
  uint16_t narrow_before[8];
  uint32_t wide_before[8];
  memcpy(narrow_before, narrow, sizeof(narrow));
  memcpy(wide_before, wide, sizeof(wide));

  /* Null runtime / baseline runtime / null buffers / unsupported shapes. */
  CHECK(et_mp4_aot_cast_f32_to_f16(NULL, 8, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_f16(base, 8, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, NULL, narrow) != 0);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, f16_in_bits, NULL) != 0);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 0, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, -1, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 3, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_UNSUPPORTED);
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 1024, f16_in_bits, narrow) ==
        ET_KERNEL_ERROR_UNSUPPORTED);

  /* Misaligned pointers passed strictly as void * (zero C11 UB). */
  unsigned char raw[64];
  memset(raw, 0, sizeof(raw));
  void *mis1 = raw + 1;
  void *mis2 = raw + 2;
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, mis2, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, f16_in_bits, mis1) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f16_to_f32(rt, 8, mis1, wide) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_bf16_to_f32(rt, 8, bf16_exp_narrow, mis2) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* Overlapping input and output spans. */
  uint32_t alias_buf[16] = {0};
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, alias_buf, alias_buf) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, alias_buf,
                                    (unsigned char *)alias_buf + 4) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f16_to_f32(rt, 8, alias_buf, alias_buf) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  /* Non-finite and out-of-range inputs rejected before any write. */
  uint32_t bad_f32[8] = {0};
  bad_f32[7] = 0x7f800000u; /* +Inf */
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  bad_f32[7] = 0x7fc00000u; /* qNaN */
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  bad_f32[7] = 0x477fe001u; /* 65504 + 1 ULP, exceeds f16 max */
  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  bad_f32[7] = 0x7f7fc000u; /* rounds to bf16 +Inf */
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, bad_f32, narrow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  uint16_t bad_f16[8] = {0};
  bad_f16[3] = 0x7c00u; /* f16 +Inf */
  CHECK(et_mp4_aot_cast_f16_to_f32(rt, 8, bad_f16, wide) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  bad_f16[3] = 0x7e01u; /* f16 NaN */
  CHECK(et_mp4_aot_cast_f16_to_f32(rt, 8, bad_f16, wide) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  uint16_t bad_bf16[8] = {0};
  bad_bf16[3] = 0x7f80u; /* bf16 +Inf */
  CHECK(et_mp4_aot_cast_bf16_to_f32(rt, 8, bad_bf16, wide) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  bad_bf16[3] = 0xffc1u; /* bf16 -NaN */
  CHECK(et_mp4_aot_cast_bf16_to_f32(rt, 8, bad_bf16, wide) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);

  CHECK(memcmp(narrow, narrow_before, sizeof(narrow)) == 0);
  CHECK(memcmp(wide, wide_before, sizeof(wide)) == 0);
}

static void test_scaler_and_unscale(const et_kernel_runtime *rt) {
  et_mp4_aot_scaler scaler;
  memset(&scaler, 0, sizeof(scaler));

  /* Uninitialized scaler is rejected explicitly. */
  CHECK(isnan(et_mp4_aot_scaler_get_scale(&scaler)));
  CHECK(et_mp4_aot_scaler_get_growth_tracker(&scaler) == -1);
  CHECK(et_mp4_aot_scaler_get_overflow_count(&scaler) == -1);
  CHECK(isnan(et_mp4_aot_scaler_scale(&scaler, 1.0)));
  CHECK(et_mp4_aot_scaler_step(&scaler, 0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);

  /* Misaligned scaler pointer via void * without typed cast UB. */
  unsigned char raw_scaler[sizeof(et_mp4_aot_scaler) + 16];
  memset(raw_scaler, 0, sizeof(raw_scaler));
  void *mis_scaler = raw_scaler + 1;
  CHECK(et_mp4_aot_scaler_init(mis_scaler, 32.0, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_init(NULL, 32.0, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_step(mis_scaler, 0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);

  /* Invalid init parameters rejected. */
  CHECK(et_mp4_aot_scaler_init(&scaler, 32.0, 1.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_init(&scaler, 32.0, 2.0, 1.0, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_init(&scaler, 32.0, 2.0, 0.5, 0, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_init(&scaler, DBL_MAX, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_init(&scaler, NAN, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);

  /* Valid init and failed re-init atomicity. */
  CHECK(et_mp4_aot_scaler_init(&scaler, 32.0, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(et_mp4_aot_scaler_get_scale(&scaler) == 32.0);
  CHECK(et_mp4_aot_scaler_get_growth_tracker(&scaler) == 0);
  CHECK(et_mp4_aot_scaler_get_overflow_count(&scaler) == 0);
  const et_mp4_aot_scaler snap = scaler;
  CHECK(et_mp4_aot_scaler_init(&scaler, 0.5, 2.0, 0.5, 2, 1.0, 256.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(memcmp(&scaler, &snap, sizeof(scaler)) == 0);

  /* Scale buffer positive and negative checks. */
  float src[8] = {0.5f, -1.0f, 1.5f, -2.0f, 0.25f, -0.5f, 0.75f, -1.0f};
  float dst[8] = {0};
  CHECK(et_mp4_aot_scaler_scale(&scaler, 0.5) == 16.0);
  CHECK(isnan(et_mp4_aot_scaler_scale(&scaler, NAN)));
  CHECK(et_mp4_aot_scaler_scale_f32(&scaler, 8, src, dst) == 0);
  CHECK(dst[0] == 16.0f && dst[3] == -64.0f);

  float dst_before[8];
  memcpy(dst_before, dst, sizeof(dst));
  float bad_src[8] = {0};
  bad_src[5] = INFINITY;
  CHECK(et_mp4_aot_scaler_scale_f32(&scaler, 8, bad_src, dst) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_scale_f32(&scaler, 8, dst, dst) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_scale_f32(&scaler, 8, raw_scaler + 1, dst) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(dst, dst_before, sizeof(dst)) == 0);

  /* Unscale clean and overflow paths for both f16 and bf16. */
  uint16_t g16[8] = {0};
  uint16_t gbf16[8] = {0};
  float out_grads[8] = {0};
  uint8_t overflow = 0xffu;

  CHECK(et_mp4_aot_cast_f32_to_f16(rt, 8, dst, g16) == 0);
  CHECK(et_mp4_aot_cast_f32_to_bf16(rt, 8, dst, gbf16) == 0);
  CHECK(et_mp4_aot_scaler_unscale_f16(rt, &scaler, 8, g16, out_grads,
                                      &overflow) == 0);
  CHECK(overflow == 0u && out_grads[0] == 0.5f && out_grads[3] == -2.0f);

  overflow = 0xffu;
  CHECK(et_mp4_aot_scaler_unscale_bf16(rt, &scaler, 8, gbf16, out_grads,
                                       &overflow) == 0);
  CHECK(overflow == 0u && out_grads[0] == 0.5f && out_grads[3] == -2.0f);

  /* Invalid inv_scale rejected directly without touching outputs. */
  float bad_inv = 0.0f;
  overflow = 0x5au;
  CHECK(et_mp4_aot_grad_unscale_f16(rt, 8, g16, &bad_inv, out_grads,
                                    &overflow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(overflow == 0x5au && out_grads[0] == 0.5f);
  bad_inv = -1.0f;
  CHECK(et_mp4_aot_grad_unscale_bf16(rt, 8, gbf16, &bad_inv, out_grads,
                                     &overflow) ==
        ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(overflow == 0x5au);

  /* Step clean twice -> growth 32 -> 64. */
  CHECK(et_mp4_aot_scaler_step(&scaler, 0) ==
        (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(et_mp4_aot_scaler_get_scale(&scaler) == 32.0);
  CHECK(et_mp4_aot_scaler_get_growth_tracker(&scaler) == 1);
  CHECK(et_mp4_aot_scaler_step(&scaler, 0) ==
        (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(et_mp4_aot_scaler_get_scale(&scaler) == 64.0);
  CHECK(et_mp4_aot_scaler_get_growth_tracker(&scaler) == 0);

  /* Inject f16 NaN -> unscale sets overflow=1 and zeroes master grads to +0.0f. */
  g16[7] = 0x7c01u;
  for (size_t i = 0; i < 8; ++i)
    out_grads[i] = -3.0f;
  overflow = 0u;
  CHECK(et_mp4_aot_scaler_unscale_f16(rt, &scaler, 8, g16, out_grads,
                                      &overflow) == 0);
  CHECK(overflow == 1u);
  for (size_t i = 0; i < 8; ++i)
    CHECK(f32_bits(out_grads[i]) == 0u);

  /* Step with overflow=1 -> backoff 64 -> 32, overflow_count == 1. */
  CHECK(et_mp4_aot_scaler_step(&scaler, 1) ==
        (int32_t)ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
  CHECK(et_mp4_aot_scaler_get_scale(&scaler) == 32.0);
  CHECK(et_mp4_aot_scaler_get_growth_tracker(&scaler) == 0);
  CHECK(et_mp4_aot_scaler_get_overflow_count(&scaler) == 1);

  /* Invalid overflow flag leaves state byte-for-byte unchanged. */
  const et_mp4_aot_scaler before_bad = scaler;
  CHECK(et_mp4_aot_scaler_step(&scaler, 2) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_scaler_step(&scaler, -1) ==
        (int32_t)ET_MP4_LOSS_SCALER_INVALID_ARGUMENT);
  CHECK(memcmp(&scaler, &before_bad, sizeof(scaler)) == 0);

  /* Underflow fault leaves scaler byte-for-byte unchanged. */
  CHECK(et_mp4_aot_scaler_init(&scaler, 2.0, 2.0, 0.5, 2, 2.0, 16.0) ==
        (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  CHECK(et_mp4_aot_scaler_step(&scaler, 0) ==
        (int32_t)ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
  const et_mp4_aot_scaler before_underflow = scaler;
  CHECK(et_mp4_aot_scaler_step(&scaler, 1) ==
        (int32_t)ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT);
  CHECK(memcmp(&scaler, &before_underflow, sizeof(scaler)) == 0);
}

int main(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_runtime *baseline = NULL;
  et_kernel_error error;

  CHECK(et_mp4_aot_open(NULL) == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_mp4_aot_open(&runtime) == 0 && runtime != NULL);
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0 && baseline != NULL);

  SET_DISPATCHING(1);
  test_sizing();
  test_marshal_helpers();
  test_casts_and_negative_boundaries(runtime, baseline);
  test_scaler_and_unscale(runtime);
  SET_DISPATCHING(0);

  et_mp4_aot_close(NULL);
  et_mp4_aot_close(baseline);
  et_mp4_aot_close(runtime);
  puts("MP4 AOT bridge C11 tests and negative boundaries: PASS");
  return 0;
}
