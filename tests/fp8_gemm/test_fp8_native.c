#include "eshkol_transformer/fp8_gemm_abi.h"
#include "fp8_gemm_eshkol_bridge.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Allocation trap: enforces ZERO heap allocations across all FP8 core & bridge
 * calls when compiled with -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free.
 */
static int forbid_alloc = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!forbid_alloc);
  __real_free(ptr);
}

static uint32_t f32_bits(float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

static float bits_f32(uint32_t u) {
  float v = 0.0f;
  memcpy(&v, &u, sizeof(v));
  return v;
}

static void fill_pattern_matrix(float *dst, size_t rows, size_t cols, int salt,
                                int with_outliers) {
  for (size_t r = 0u; r < rows; ++r) {
    for (size_t c = 0u; c < cols; ++c) {
      const size_t idx = r * cols + c;
      const int bucket =
          (int)((31u * (unsigned)idx + 17u * (unsigned)salt + (unsigned)(idx / 5u)) %
                33u) -
          16;
      if (with_outliers) {
        /* Outlier-heavy activation tensor: a single massive channel outlier at
         * (row=0, col=0) = 224.0 alongside small normal activations
         * (~0.0003..0.0048). Per-tensor E4M3FN (scale = 0.5) crushes all other
         * entries into subnormals/zero, while 1x16 block scaling isolates the
         * outlier to a single block and preserves full precision elsewhere. */
        float v = (float)bucket * 0.0003f;
        if (c == 0u && r == 0u) {
          v = 224.0f;
        }
        dst[idx] = v;
      } else {
        float v = (float)bucket * 0.125f;
        if (c == 0u) {
          v = ((r % 2u) == 0u) ? 0.03125f : -0.03125f;
        } else if (c == 1u) {
          v = 2.0f;
        }
        dst[idx] = v;
      }
    }
  }
}

static void print_f32_vec_hex(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

static void print_u8_vec_hex(const uint8_t *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%02x", (unsigned)v[i]);
  }
  putchar('\n');
}

/* ---- Oracle Emitter Mode (--oracle) ---- */

static void run_oracle_dump(void) {
  forbid_alloc = 1;

  /* 1. Exhaustive 256-codepoint E4M3FN decode & re-encode. */
  puts("SECTION E4M3_ALL256");
  for (unsigned c = 0u; c < 256u; ++c) {
    float val = 0.0f;
    const et_fp8_status dst = et_fp8_e4m3_decode_scalar_v1((uint8_t)c, &val);
    uint8_t re = 0u;
    et_fp8_status rst = ET_FP8_OK;
    if (dst == ET_FP8_OK) {
      rst = et_fp8_e4m3_encode_scalar_v1(val, ET_FP8_OVERFLOW_REJECT, &re);
    }
    forbid_alloc = 0;
    printf("%u %d %08x %d %u\n", c, (int)dst,
           dst == ET_FP8_OK ? f32_bits(val) : 0u, (int)rst, (unsigned)re);
    forbid_alloc = 1;
  }

  /* 2. Exhaustive 256-codepoint E5M2 decode & re-encode. */
  puts("SECTION E5M2_ALL256");
  for (unsigned c = 0u; c < 256u; ++c) {
    float val = 0.0f;
    const et_fp8_status dst = et_fp8_e5m2_decode_scalar_v1((uint8_t)c, &val);
    uint8_t re = 0u;
    et_fp8_status rst = ET_FP8_OK;
    if (dst == ET_FP8_OK) {
      rst = et_fp8_e5m2_encode_scalar_v1(val, ET_FP8_OVERFLOW_REJECT, &re);
    }
    forbid_alloc = 0;
    printf("%u %d %08x %d %u\n", c, (int)dst,
           dst == ET_FP8_OK ? f32_bits(val) : 0u, (int)rst, (unsigned)re);
    forbid_alloc = 1;
  }

  /* 3. RNE tie-to-even & overflow boundary bit patterns for E4M3FN. */
  const uint32_t e4m3_cases[] = {
      0x00000000u, /* +0.0 */
      0x80000000u, /* -0.0 */
      0x00000001u, /* smallest f32 subnormal -> +0.0 */
      0x80000001u, /* -smallest f32 subnormal -> -0.0 */
      0x3A7FFFFFu, /* nextafter(2^-10, 0) -> 0x00 */
      0x3A800000u, /* 2^-10 exact half of 0x01, tie to even 0x00 */
      0x3A800001u, /* nextafter(2^-10, +inf) -> 0x01 */
      0xBA800000u, /* -2^-10 -> 0x80 */
      0xBA800001u, /* -nextafter(2^-10, +inf) -> 0x81 */
      0x3B400000u, /* 1.5 * 2^-9 -> tie to even 0x02 */
      0x3BA00000u, /* 2.5 * 2^-9 -> tie to even 0x02 */
      0x3BE00000u, /* 3.5 * 2^-9 -> tie to even 0x04 */
      0x3C500000u, /* 6.5 * 2^-9 -> tie to even 0x06 */
      0x3C700000u, /* 7.5 * 2^-9 -> tie to even 0x08 (min normal!) */
      0x3C7FFFFFu, /* just below 2^-6 -> 0x08 */
      0x3C800000u, /* 2^-6 exact -> 0x08 */
      0x3F880000u, /* 1.0625 = half between 1.0 (0x38 even) and 1.125 (0x39) -> 0x38 */
      0x3F880001u, /* > 1.0625 -> 0x39 */
      0x3F980000u, /* 1.1875 = half between 1.125 (0x39 odd) and 1.25 (0x3A) -> 0x3A */
      0x3FF80000u, /* 1.9375 = half between 1.875 (0x3F odd) and 2.0 (0x40) -> 0x40 */
      0x43E00000u, /* 448.0 -> 0x7E */
      0x43E7FFFFu, /* nextafter(464.0, 0) -> 0x7E */
      0x43E80000u, /* 464.0 exact half between 448 (even) & 480 (NaN) -> 0x7E */
      0x43E80001u, /* nextafter(464.0, +inf) -> OVERFLOW / saturate 0x7E */
      0x43F00000u, /* 480.0 -> OVERFLOW / saturate 0x7E */
      0xC3E80000u, /* -464.0 -> 0xFE */
      0xC3E80001u, /* -nextafter(464.0, +inf) -> OVERFLOW / saturate 0xFE */
      0x7F7FFFFFu, /* FLT_MAX -> OVERFLOW / saturate 0x7E */
      0x7F800000u, /* +Inf -> NONFINITE */
      0xFF800000u, /* -Inf -> NONFINITE */
      0x7FC00000u, /* qNaN -> NONFINITE */
      0x7FA00001u, /* sNaN -> NONFINITE */
  };
  forbid_alloc = 0;
  printf("SECTION E4M3_RNE %zu\n", ARRAY_COUNT(e4m3_cases));
  forbid_alloc = 1;
  for (size_t i = 0u; i < ARRAY_COUNT(e4m3_cases); ++i) {
    uint8_t q_sat = 0u, q_rej = 0u;
    const et_fp8_status s_sat = et_fp8_e4m3_encode_scalar_v1(
        bits_f32(e4m3_cases[i]), ET_FP8_OVERFLOW_SATURATE_TO_MAX, &q_sat);
    const et_fp8_status s_rej = et_fp8_e4m3_encode_scalar_v1(
        bits_f32(e4m3_cases[i]), ET_FP8_OVERFLOW_REJECT, &q_rej);
    forbid_alloc = 0;
    printf("%08x %d %u %d %u\n", e4m3_cases[i], (int)s_sat, (unsigned)q_sat,
           (int)s_rej, (unsigned)q_rej);
    forbid_alloc = 1;
  }

  /* 4. RNE tie-to-even & overflow boundary bit patterns for E5M2. */
  const uint32_t e5m2_cases[] = {
      0x00000000u, /* +0.0 */
      0x80000000u, /* -0.0 */
      0x36FFFFFFu, /* nextafter(2^-17, 0) -> 0x00 */
      0x37000000u, /* 2^-17 exact half of 0x01, tie to even 0x00 */
      0x37000001u, /* nextafter(2^-17, +inf) -> 0x01 */
      0xB7000000u, /* -2^-17 -> 0x80 */
      0xB7000001u, /* -nextafter(2^-17, +inf) -> 0x81 */
      0x37C00000u, /* 1.5 * 2^-16 -> tie to even 0x02 */
      0x38200000u, /* 2.5 * 2^-16 -> tie to even 0x02 */
      0x38600000u, /* 3.5 * 2^-16 -> tie to even 0x04 (min normal!) */
      0x38800000u, /* 2^-14 exact -> 0x04 */
      0x3F900000u, /* 1.125 = half between 1.0 (0x3C even) and 1.25 (0x3D) -> 0x3C */
      0x3F900001u, /* > 1.125 -> 0x3D */
      0x3FB00000u, /* 1.375 = half between 1.25 (0x3D odd) and 1.5 (0x3E) -> 0x3E */
      0x3FF00000u, /* 1.875 = half between 1.75 (0x3F odd) and 2.0 (0x40) -> 0x40 */
      0x47600000u, /* 57344.0 -> 0x7B */
      0x476FFFFFu, /* nextafter(61440.0, 0) -> 0x7B */
      0x47700000u, /* 61440.0 exact half between 57344 (odd) and 65536 (Inf) -> OVERFLOW / saturate 0x7B */
      0x47700001u, /* nextafter(61440.0, +inf) -> OVERFLOW / saturate 0x7B */
      0xC76FFFFFu, /* -nextafter(61440.0, 0) -> 0xFB */
      0xC7700000u, /* -61440.0 -> OVERFLOW / saturate 0xFB */
      0x7F7FFFFFu, /* FLT_MAX -> OVERFLOW / saturate 0x7B */
      0x7F800000u, /* +Inf -> NONFINITE */
      0xFF800000u, /* -Inf -> NONFINITE */
      0x7FC00000u, /* qNaN -> NONFINITE */
  };
  forbid_alloc = 0;
  printf("SECTION E5M2_RNE %zu\n", ARRAY_COUNT(e5m2_cases));
  forbid_alloc = 1;
  for (size_t i = 0u; i < ARRAY_COUNT(e5m2_cases); ++i) {
    uint8_t q_sat = 0u, q_rej = 0u;
    const et_fp8_status s_sat = et_fp8_e5m2_encode_scalar_v1(
        bits_f32(e5m2_cases[i]), ET_FP8_OVERFLOW_SATURATE_TO_MAX, &q_sat);
    const et_fp8_status s_rej = et_fp8_e5m2_encode_scalar_v1(
        bits_f32(e5m2_cases[i]), ET_FP8_OVERFLOW_REJECT, &q_rej);
    forbid_alloc = 0;
    printf("%08x %d %u %d %u\n", e5m2_cases[i], (int)s_sat, (unsigned)q_sat,
           (int)s_rej, (unsigned)q_rej);
    forbid_alloc = 1;
  }

  /* 5. Delayed-scaling trajectory across 10 steps (E4M3FN & E5M2). */
  const float amax_seq[10] = {0.0f,  2.0f,  64.0f, 8.0f,  4.0f,
                              2.0f,  1.0f,  0.5f,  0.0f,  128.0f};
  for (int t = 0; t < 2; ++t) {
    const et_fp8_format fmt = (t == 0) ? ET_FP8_FORMAT_E4M3FN : ET_FP8_FORMAT_E5M2;
    const uint32_t hlen = (t == 0) ? 4u : 5u;
    const uint32_t margin = (t == 0) ? 1u : 2u;
    const float eps = 1e-6f;
    const float min_s = 1e-4f;
    const float max_s = 65536.0f;
    et_fp8_delayed_scale_state_v1 st;
    CHECK(et_fp8_delayed_scale_init(&st, fmt, hlen, margin, eps, min_s, max_s) ==
          ET_FP8_OK);
    forbid_alloc = 0;
    printf("SECTION DELAYED_SCALE %u %u %u %a %a %a %zu\n", (unsigned)fmt,
           (unsigned)hlen, (unsigned)margin, (double)eps, (double)min_s,
           (double)max_s, ARRAY_COUNT(amax_seq));
    forbid_alloc = 1;
    for (size_t s = 0u; s < ARRAY_COUNT(amax_seq); ++s) {
      CHECK(et_fp8_delayed_scale_update(&st, amax_seq[s]) == ET_FP8_OK);
      float sc = 0.0f, ha = 0.0f;
      CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
      forbid_alloc = 0;
      printf("%a %a %a\n", (double)amax_seq[s], (double)sc, (double)ha);
      forbid_alloc = 1;
    }
  }

  /* 6. Block-scaled W8A8 GEMM + STE backward cases. */
  const struct {
    size_t m, n, k, bk, bn;
    int has_bias;
    int outliers;
    int salt;
  } gemm_cases[] = {
      {4u, 8u, 32u, 16u, 1u, 1, 0, 3},
      {8u, 16u, 32u, 16u, 16u, 1, 1, 7},
      {6u, 32u, 64u, 32u, 32u, 0, 0, 11},
      {8u, 16u, 32u, 16u, 1u, 1, 1, 19},
  };
  forbid_alloc = 0;
  printf("SECTION BLOCK_GEMM %zu\n", ARRAY_COUNT(gemm_cases));
  for (size_t ci = 0u; ci < ARRAY_COUNT(gemm_cases); ++ci) {
    const size_t M = gemm_cases[ci].m;
    const size_t N = gemm_cases[ci].n;
    const size_t K = gemm_cases[ci].k;
    const size_t BK = gemm_cases[ci].bk;
    const size_t BN = gemm_cases[ci].bn;
    const size_t KB = K / BK;
    const size_t NB = N / BN;
    float X[8 * 64], W[32 * 64], bias[32], Y[8 * 32];
    float dY[8 * 32], dX[8 * 64], dW[32 * 64], dB[32];
    uint8_t Xq[8 * 64], Wq[32 * 64];
    float Sx[8 * 4], Sw[32 * 4];

    fill_pattern_matrix(X, M, K, gemm_cases[ci].salt, gemm_cases[ci].outliers);
    fill_pattern_matrix(W, N, K, gemm_cases[ci].salt + 5, 0);
    fill_pattern_matrix(dY, M, N, gemm_cases[ci].salt + 9, 0);
    for (size_t i = 0u; i < N; ++i)
      bias[i] = ((float)((int)(i % 7u) - 3)) * 0.125f;

    forbid_alloc = 1;
    CHECK(et_fp8_block_quantize_e4m3_v1(X, M, K, 1u, BK, Xq, Sx) == ET_FP8_OK);
    CHECK(et_fp8_block_quantize_e4m3_v1(W, N, K, BN, BK, Wq, Sw) == ET_FP8_OK);
    CHECK(et_fp8_block_gemm_e4m3_v1(M, N, K, BK, BN, Xq, Sx, Wq, Sw,
                                    gemm_cases[ci].has_bias ? bias : NULL,
                                    Y) == ET_FP8_OK);
    CHECK(et_fp8_block_gemm_e4m3_backward_v1(M, N, K, BK, BN, dY, Xq, Sx, Wq,
                                             Sw, dX, dW, dB) == ET_FP8_OK);
    forbid_alloc = 0;

    printf("CASE %zu %zu %zu %zu %zu %d\n", M, N, K, BK, BN,
           gemm_cases[ci].has_bias);
    print_u8_vec_hex(Xq, M * K);
    print_f32_vec_hex(Sx, M * KB);
    print_u8_vec_hex(Wq, N * K);
    print_f32_vec_hex(Sw, NB * KB);
    if (gemm_cases[ci].has_bias)
      print_f32_vec_hex(bias, N);
    print_f32_vec_hex(Y, M * N);
    print_f32_vec_hex(dY, M * N);
    print_f32_vec_hex(dX, M * K);
    print_f32_vec_hex(dW, N * K);
    print_f32_vec_hex(dB, N);
  }
  forbid_alloc = 0;
}

/* ---- Native C11 Verification Suite ---- */

static void test_exhaustive_codecs(void) {
  forbid_alloc = 1;

  /* E4M3FN: 254 finite codes + 2 NaN codes (0x7F, 0xFF). */
  int e4m3_finite = 0;
  int e4m3_rejected = 0;
  float prev_pos = -1.0f;
  for (unsigned c = 0u; c < 256u; ++c) {
    float val = 1234.5f;
    const et_fp8_status st = et_fp8_e4m3_decode_scalar_v1((uint8_t)c, &val);
    if (c == 0x7Fu || c == 0xFFu) {
      CHECK(st == ET_FP8_ERR_NONFINITE);
      CHECK(val == 1234.5f); /* untouched on error */
      e4m3_rejected += 1;
    } else {
      CHECK(st == ET_FP8_OK);
      CHECK(isfinite(val));
      e4m3_finite += 1;
      if (c < 0x7Fu) {
        CHECK(val > prev_pos);
        prev_pos = val;
      }
      uint8_t re = 0xAAu;
      CHECK(et_fp8_e4m3_encode_scalar_v1(val, ET_FP8_OVERFLOW_REJECT, &re) ==
            ET_FP8_OK);
      CHECK(re == (uint8_t)c);
    }
  }
  CHECK(e4m3_finite == 254 && e4m3_rejected == 2);

  /* Key E4M3FN landmarks. */
  float v = 0.0f;
  CHECK(et_fp8_e4m3_decode_scalar_v1(0x00u, &v) == ET_FP8_OK &&
        f32_bits(v) == 0x00000000u);
  CHECK(et_fp8_e4m3_decode_scalar_v1(0x80u, &v) == ET_FP8_OK &&
        f32_bits(v) == 0x80000000u);
  CHECK(et_fp8_e4m3_decode_scalar_v1(0x01u, &v) == ET_FP8_OK &&
        v == 0.001953125f);
  CHECK(et_fp8_e4m3_decode_scalar_v1(0x7Eu, &v) == ET_FP8_OK && v == 448.0f);
  CHECK(et_fp8_e4m3_decode_scalar_v1(0xFEu, &v) == ET_FP8_OK && v == -448.0f);

  /* E5M2: 248 finite codes + 8 Inf/NaN codes (0x7C..0x7F, 0xFC..0xFF). */
  int e5m2_finite = 0;
  int e5m2_rejected = 0;
  prev_pos = -1.0f;
  for (unsigned c = 0u; c < 256u; ++c) {
    float val = 5678.5f;
    const et_fp8_status st = et_fp8_e5m2_decode_scalar_v1((uint8_t)c, &val);
    if ((c & 0x7Cu) == 0x7Cu) {
      CHECK(st == ET_FP8_ERR_NONFINITE);
      CHECK(val == 5678.5f);
      e5m2_rejected += 1;
    } else {
      CHECK(st == ET_FP8_OK);
      CHECK(isfinite(val));
      e5m2_finite += 1;
      if (c < 0x7Cu) {
        CHECK(val > prev_pos);
        prev_pos = val;
      }
      uint8_t re = 0xAAu;
      CHECK(et_fp8_e5m2_encode_scalar_v1(val, ET_FP8_OVERFLOW_REJECT, &re) ==
            ET_FP8_OK);
      CHECK(re == (uint8_t)c);
    }
  }
  CHECK(e5m2_finite == 248 && e5m2_rejected == 8);

  CHECK(et_fp8_e5m2_decode_scalar_v1(0x01u, &v) == ET_FP8_OK &&
        v == 0.0000152587890625f);
  CHECK(et_fp8_e5m2_decode_scalar_v1(0x7Bu, &v) == ET_FP8_OK && v == 57344.0f);
  CHECK(et_fp8_e5m2_decode_scalar_v1(0xFBu, &v) == ET_FP8_OK && v == -57344.0f);

  /* RNE half-way tie-to-even spot checks + overflow modes. */
  uint8_t q = 0x55u;
  CHECK(et_fp8_e4m3_encode_scalar_v1(464.0f, ET_FP8_OVERFLOW_REJECT, &q) ==
            ET_FP8_OK &&
        q == 0x7Eu);
  q = 0x55u;
  CHECK(et_fp8_e4m3_encode_scalar_v1(bits_f32(0x43E80001u),
                                     ET_FP8_OVERFLOW_REJECT,
                                     &q) == ET_FP8_ERR_OVERFLOW &&
        q == 0x55u);
  CHECK(et_fp8_e4m3_encode_scalar_v1(bits_f32(0x43E80001u),
                                     ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                                     &q) == ET_FP8_OK &&
        q == 0x7Eu);
  q = 0x55u;
  CHECK(et_fp8_e5m2_encode_scalar_v1(61440.0f, ET_FP8_OVERFLOW_REJECT, &q) ==
            ET_FP8_ERR_OVERFLOW &&
        q == 0x55u);
  CHECK(et_fp8_e5m2_encode_scalar_v1(61440.0f, ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                                     &q) == ET_FP8_OK &&
        q == 0x7Bu);
  CHECK(et_fp8_e5m2_encode_scalar_v1(bits_f32(0x476FFFFFu),
                                     ET_FP8_OVERFLOW_REJECT, &q) == ET_FP8_OK &&
        q == 0x7Bu);

  /* Non-finite f32 always rejected even in SATURATE_TO_MAX mode. */
  const uint32_t nonfinites[] = {0x7F800000u, 0xFF800000u, 0x7FC00000u,
                                 0x7FA00001u};
  for (size_t i = 0u; i < ARRAY_COUNT(nonfinites); ++i) {
    q = 0x33u;
    CHECK(et_fp8_e4m3_encode_scalar_v1(bits_f32(nonfinites[i]),
                                       ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                                       &q) == ET_FP8_ERR_NONFINITE &&
          q == 0x33u);
    CHECK(et_fp8_e5m2_encode_scalar_v1(bits_f32(nonfinites[i]),
                                       ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                                       &q) == ET_FP8_ERR_NONFINITE &&
          q == 0x33u);
  }

  /* Alignment-safe bulk encode/decode on unaligned +1 byte offset! */
  uint8_t raw_f32[64];
  uint8_t raw_q[16];
  uint8_t raw_out[64];
  const float sample[8] = {0.0f, -0.0f, 0.001953125f, 1.5f,
                           -3.25f, 448.0f, -448.0f, 16.0f};
  memcpy(raw_f32 + 1, sample, sizeof(sample));
  CHECK(et_fp8_e4m3_encode_v1((const float *)(const void *)(raw_f32 + 1), 8u,
                              ET_FP8_OVERFLOW_REJECT, raw_q + 3) == ET_FP8_OK);
  CHECK(et_fp8_e4m3_decode_v1(raw_q + 3, 8u, (float *)(void *)(raw_out + 2)) ==
        ET_FP8_OK);
  CHECK(memcmp(raw_out + 2, sample, sizeof(sample)) == 0);

  forbid_alloc = 0;
}

static void test_delayed_scaling(void) {
  forbid_alloc = 1;

  et_fp8_delayed_scale_state_v1 st;
  CHECK(et_fp8_delayed_scale_init(&st, ET_FP8_FORMAT_E4M3FN, 4u, 1u, 1e-6f,
                                  1e-4f, 65536.0f) == ET_FP8_OK);
  /* Before any update: history_amax == 0 => clamped to max_scale. */
  float sc = 0.0f, ha = -1.0f;
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 0.0f && sc == 65536.0f);

  /* Step 1: zero amax safety. */
  CHECK(et_fp8_delayed_scale_update(&st, 0.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 0.0f && sc == 65536.0f);

  /* Step 2: spike 112.0 => scale = (448 / 112) * 0.5 = 2.0. */
  CHECK(et_fp8_delayed_scale_update(&st, 112.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 112.0f && fabsf(sc - 2.0f) <= 1e-6f);

  /* Steps 3..5: smaller values while 112.0 stays in 4-step window. */
  CHECK(et_fp8_delayed_scale_update(&st, 8.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_update(&st, 4.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_update(&st, 2.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 112.0f && fabsf(sc - 2.0f) <= 1e-6f);

  /* Step 6: 112.0 evicted from 4-step ring buffer! Window is {8, 4, 2, 1}.
   * scale = (448 / 8) * 0.5 = 28.0. */
  CHECK(et_fp8_delayed_scale_update(&st, 1.0f) == ET_FP8_OK);
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 8.0f && fabsf(sc - 28.0f) <= 1e-6f);

  /* Tensor update path. */
  const float tensor[4] = {-14.0f, 3.5f, 16.0f, -0.25f};
  float obs = 0.0f;
  CHECK(et_fp8_delayed_scale_update_tensor(&st, tensor, 4u, &obs) == ET_FP8_OK);
  CHECK(obs == 16.0f);
  CHECK(et_fp8_delayed_scale_compute(&st, &sc, &ha) == ET_FP8_OK);
  CHECK(ha == 16.0f && fabsf(sc - 14.0f) <= 1e-6f);

  /* Failure-atomicity on negative/non-finite amax. */
  const et_fp8_delayed_scale_state_v1 snapshot = st;
  CHECK(et_fp8_delayed_scale_update(&st, -1.0f) == ET_FP8_ERR_INVALID_ARGUMENT);
  CHECK(memcmp(&st, &snapshot, sizeof(st)) == 0);
  CHECK(et_fp8_delayed_scale_update(&st, bits_f32(0x7F800000u)) ==
        ET_FP8_ERR_NONFINITE);
  CHECK(memcmp(&st, &snapshot, sizeof(st)) == 0);

  forbid_alloc = 0;
}

static void test_block_gemm_parity_outliers_and_ste_fd(void) {
  const size_t M = 8u, N = 16u, K = 64u, BK = 16u;
  float X[8 * 64], W[16 * 64], bias[16];
  fill_pattern_matrix(X, M, K, 13, 1); /* outlier-heavy */
  fill_pattern_matrix(W, N, K, 29, 0);
  for (size_t i = 0u; i < N; ++i)
    bias[i] = ((float)((int)(i % 5u) - 2)) * 0.25f;

  /* Exact binary64 reference Y_f32 = X @ W^T + bias. */
  float Y_f32[8 * 16];
  for (size_t mi = 0u; mi < M; ++mi) {
    for (size_t ni = 0u; ni < N; ++ni) {
      double acc = 0.0;
      for (size_t ki = 0u; ki < K; ++ki)
        acc += (double)X[mi * K + ki] * (double)W[ni * K + ki];
      acc += (double)bias[ni];
      Y_f32[mi * N + ni] = (float)acc;
    }
  }

  /* Per-tensor baseline quantization & GEMM. */
  uint8_t Xq_pt[8 * 64], Wq_pt[16 * 64];
  float sx_pt = 0.0f, sw_pt = 0.0f;
  forbid_alloc = 1;
  CHECK(et_fp8_per_tensor_quantize_e4m3_v1(X, M * K, Xq_pt, &sx_pt) ==
        ET_FP8_OK);
  CHECK(et_fp8_per_tensor_quantize_e4m3_v1(W, N * K, Wq_pt, &sw_pt) ==
        ET_FP8_OK);
  float Xdeq_pt[8 * 64], Wdeq_pt[16 * 64];
  CHECK(et_fp8_e4m3_decode_v1(Xq_pt, M * K, Xdeq_pt) == ET_FP8_OK);
  CHECK(et_fp8_e4m3_decode_v1(Wq_pt, N * K, Wdeq_pt) == ET_FP8_OK);
  forbid_alloc = 0;

  double pertensor_max_err = 0.0;
  double pertensor_l2 = 0.0;
  for (size_t mi = 0u; mi < M; ++mi) {
    for (size_t ni = 0u; ni < N; ++ni) {
      double acc = 0.0;
      for (size_t ki = 0u; ki < K; ++ki)
        acc += ((double)Xdeq_pt[mi * K + ki] * (double)sx_pt) *
               ((double)Wdeq_pt[ni * K + ki] * (double)sw_pt);
      acc += (double)bias[ni];
      const double err = fabs((double)(float)acc - (double)Y_f32[mi * N + ni]);
      if (err > pertensor_max_err)
        pertensor_max_err = err;
      pertensor_l2 += err * err;
    }
  }
  const double pertensor_rmse = sqrt(pertensor_l2 / (double)(M * N));

  /* Test both block_n = 1 (1x16) and block_n = 16 (16x16 DeepSeek-V3 tile). */
  const size_t bn_list[2] = {1u, 16u};
  for (size_t bi = 0u; bi < 2u; ++bi) {
    const size_t BN = bn_list[bi];
    uint8_t Xq[8 * 64], Wq[16 * 64];
    float Sx[8 * 4], Sw[16 * 4];
    float Xdeq[8 * 64], Wdeq[16 * 64], Y_blk[8 * 16];

    forbid_alloc = 1;
    CHECK(et_fp8_block_quantize_e4m3_v1(X, M, K, 1u, BK, Xq, Sx) == ET_FP8_OK);
    CHECK(et_fp8_block_quantize_e4m3_v1(W, N, K, BN, BK, Wq, Sw) == ET_FP8_OK);
    CHECK(et_fp8_block_dequantize_e4m3_v1(Xq, Sx, M, K, 1u, BK, Xdeq) ==
          ET_FP8_OK);
    CHECK(et_fp8_block_dequantize_e4m3_v1(Wq, Sw, N, K, BN, BK, Wdeq) ==
          ET_FP8_OK);
    CHECK(et_fp8_block_gemm_e4m3_v1(M, N, K, BK, BN, Xq, Sx, Wq, Sw, bias,
                                    Y_blk) == ET_FP8_OK);
    forbid_alloc = 0;

    /* Exact parity against dequant(X_q, S_x) @ dequant(W_q, S_w)^T + bias. */
    double parity_max_err = 0.0;
    double blk_max_err = 0.0;
    double blk_l2 = 0.0;
    for (size_t mi = 0u; mi < M; ++mi) {
      for (size_t ni = 0u; ni < N; ++ni) {
        double ref = 0.0;
        for (size_t ki = 0u; ki < K; ++ki)
          ref += (double)Xdeq[mi * K + ki] * (double)Wdeq[ni * K + ki];
        ref += (double)bias[ni];
        const double p_err = fabs((double)Y_blk[mi * N + ni] - (double)(float)ref);
        if (p_err > parity_max_err)
          parity_max_err = p_err;
        const double q_err =
            fabs((double)Y_blk[mi * N + ni] - (double)Y_f32[mi * N + ni]);
        if (q_err > blk_max_err)
          blk_max_err = q_err;
        blk_l2 += q_err * q_err;
      }
    }
    const double blk_rmse = sqrt(blk_l2 / (double)(M * N));

    /* Central finite-difference numerical gradient check for STE backward. */
    float dY[8 * 16], dX[8 * 64], dW[16 * 64], dB[16];
    fill_pattern_matrix(dY, M, N, 41 + (int)bi, 0);
    forbid_alloc = 1;
    CHECK(et_fp8_block_gemm_e4m3_backward_v1(M, N, K, BK, BN, dY, Xq, Sx, Wq,
                                             Sw, dX, dW, dB) == ET_FP8_OK);
    forbid_alloc = 0;

    const double eps = 1e-3;
    double max_fd_err = 0.0;
    for (size_t mi = 0u; mi < M; ++mi) {
      for (size_t ki = 0u; ki < K; ki += 11u) {
        double lp = 0.0, lm = 0.0;
        for (size_t ni = 0u; ni < N; ++ni) {
          lp += (double)dY[mi * N + ni] *
                ((double)Xdeq[mi * K + ki] + eps) * (double)Wdeq[ni * K + ki];
          lm += (double)dY[mi * N + ni] *
                ((double)Xdeq[mi * K + ki] - eps) * (double)Wdeq[ni * K + ki];
        }
        const double fd = (lp - lm) / (2.0 * eps);
        const double err = fabs((double)dX[mi * K + ki] - fd);
        if (err > max_fd_err)
          max_fd_err = err;
      }
    }
    for (size_t ni = 0u; ni < N; ni += 3u) {
      for (size_t ki = 0u; ki < K; ki += 11u) {
        double lp = 0.0, lm = 0.0;
        for (size_t mi = 0u; mi < M; ++mi) {
          lp += (double)dY[mi * N + ni] * (double)Xdeq[mi * K + ki] *
                ((double)Wdeq[ni * K + ki] + eps);
          lm += (double)dY[mi * N + ni] * (double)Xdeq[mi * K + ki] *
                ((double)Wdeq[ni * K + ki] - eps);
        }
        const double fd = (lp - lm) / (2.0 * eps);
        const double err = fabs((double)dW[ni * K + ki] - fd);
        if (err > max_fd_err)
          max_fd_err = err;
      }
    }
    for (size_t ni = 0u; ni < N; ++ni) {
      double lp = 0.0, lm = 0.0;
      for (size_t mi = 0u; mi < M; ++mi) {
        lp += (double)dY[mi * N + ni] * ((double)bias[ni] + eps);
        lm += (double)dY[mi * N + ni] * ((double)bias[ni] - eps);
      }
      const double fd = (lp - lm) / (2.0 * eps);
      const double err = fabs((double)dB[ni] - fd);
      if (err > max_fd_err)
        max_fd_err = err;
    }
    printf("FP8 NATIVE BLOCK_GEMM bn=%zu parity_err=%.3e block_vs_f32_max=%.5f "
           "pertensor_vs_f32_max=%.5f block_rmse=%.5f pertensor_rmse=%.5f "
           "ste_fd_err=%.3e PASS\n",
           BN, parity_max_err, blk_max_err, pertensor_max_err, blk_rmse,
           pertensor_rmse, max_fd_err);
    CHECK(parity_max_err <= 1e-6);
    CHECK(blk_max_err < pertensor_max_err * 0.5);
    CHECK(blk_rmse < pertensor_rmse * 0.35);
    CHECK(max_fd_err <= 1e-4);
  }
}

static void test_negative_and_atomicity(void) {
  forbid_alloc = 1;

  /* 1. Bulk codec two-pass failure atomicity (error on last element leaves dst untouched). */
  float src_f[4] = {1.0f, 2.0f, 3.0f, 500.0f};
  uint8_t dst_q[4] = {0xA1u, 0xB2u, 0xC3u, 0xD4u};
  const uint8_t dst_q_orig[4] = {0xA1u, 0xB2u, 0xC3u, 0xD4u};
  CHECK(et_fp8_e4m3_encode_v1(src_f, 4u, ET_FP8_OVERFLOW_REJECT, dst_q) ==
        ET_FP8_ERR_OVERFLOW);
  CHECK(memcmp(dst_q, dst_q_orig, sizeof(dst_q)) == 0);

  src_f[3] = bits_f32(0x7FC00000u); /* NaN on last element */
  CHECK(et_fp8_e4m3_encode_v1(src_f, 4u, ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                              dst_q) == ET_FP8_ERR_NONFINITE);
  CHECK(memcmp(dst_q, dst_q_orig, sizeof(dst_q)) == 0);

  uint8_t bad_codes[4] = {0x38u, 0x40u, 0x44u, 0x7Fu};
  float dst_f[4] = {9.0f, 8.0f, 7.0f, 6.0f};
  const float dst_f_orig[4] = {9.0f, 8.0f, 7.0f, 6.0f};
  CHECK(et_fp8_e4m3_decode_v1(bad_codes, 4u, dst_f) == ET_FP8_ERR_NONFINITE);
  CHECK(memcmp(dst_f, dst_f_orig, sizeof(dst_f)) == 0);

  bad_codes[3] = 0x7Cu; /* Inf in E5M2 */
  CHECK(et_fp8_e5m2_decode_v1(bad_codes, 4u, dst_f) == ET_FP8_ERR_NONFINITE);
  CHECK(memcmp(dst_f, dst_f_orig, sizeof(dst_f)) == 0);

  /* 2. Aliasing rejections. */
  CHECK(et_fp8_e4m3_encode_v1(src_f, 1u, ET_FP8_OVERFLOW_SATURATE_TO_MAX,
                              (uint8_t *)(void *)src_f) == ET_FP8_ERR_ALIAS);
  CHECK(et_fp8_e4m3_decode_v1((const uint8_t *)(const void *)dst_f, 1u,
                              dst_f) == ET_FP8_ERR_ALIAS);

  /* 3. GEMM shape, block, NaN code, invalid scale, aliasing, extreme overflow atomicity. */
  uint8_t xq[32], wq[32];
  memset(xq, 0x38, sizeof(xq)); /* 1.0 in E4M3FN */
  memset(wq, 0x38, sizeof(wq));
  float sx[2] = {1.0f, 1.0f};
  float sw[2] = {1.0f, 1.0f};
  float y[4] = {77.0f, 88.0f, 99.0f, 111.0f};
  const float y_orig[4] = {77.0f, 88.0f, 99.0f, 111.0f};

  /* Non-multiple K (24 with block_k=16). */
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 1u, 24u, 16u, 1u, xq, sx, wq, sw, NULL,
                                  y) == ET_FP8_ERR_SHAPE);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);

  /* Unsupported block_k (8). */
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 1u, 16u, 8u, 1u, xq, sx, wq, sw, NULL,
                                  y) == ET_FP8_ERR_UNSUPPORTED_BLOCK);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);

  /* Non-multiple N with block_n=16. */
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 8u, 16u, 16u, 16u, xq, sx, wq, sw, NULL,
                                  y) == ET_FP8_ERR_SHAPE);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);

  /* Output aliasing S_x. */
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 1u, 16u, 16u, 1u, xq, sx, wq, sw, NULL,
                                  sx) == ET_FP8_ERR_ALIAS);

  /* NaN E4M3FN code in W_q. */
  wq[15] = 0xFFu;
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 1u, 16u, 16u, 1u, xq, sx, wq, sw, NULL,
                                  y) == ET_FP8_ERR_NONFINITE);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);
  wq[15] = 0x38u;

  /* Zero / negative scale rejected. */
  sx[0] = 0.0f;
  CHECK(et_fp8_block_gemm_e4m3_v1(1u, 1u, 16u, 16u, 1u, xq, sx, wq, sw, NULL,
                                  y) == ET_FP8_ERR_INVALID_ARGUMENT);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);

  /* Extreme-scale f32 overflow on second output preserves first output untouched! */
  uint8_t xq2[32];
  memset(xq2, 0x7E, sizeof(xq2)); /* 448.0 */
  float sx2[2] = {1.0f, 1e35f};   /* row 0 safe, row 1 overflows f32! */
  float sw2[1] = {1e10f};
  CHECK(et_fp8_block_gemm_e4m3_v1(2u, 1u, 16u, 16u, 1u, xq2, sx2, xq2, sw2,
                                  NULL, y) == ET_FP8_ERR_OVERFLOW);
  CHECK(memcmp(y, y_orig, sizeof(y)) == 0);

  /* 4. Non-standard FP environment rejected explicitly (rounding mode + FTZ/DAZ). */
  const int saved_round = fegetround();
  CHECK(fesetround(FE_TOWARDZERO) == 0);
  uint8_t dummy_q = 0u;
  CHECK(et_fp8_e4m3_encode_scalar_v1(1.0f, ET_FP8_OVERFLOW_REJECT, &dummy_q) ==
        ET_FP8_ERR_ENVIRONMENT);
  CHECK(fesetround(saved_round) == 0);

#if defined(__x86_64__)
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_fp8_e4m3_encode_scalar_v1(1.0f, ET_FP8_OVERFLOW_REJECT, &dummy_q) ==
        ET_FP8_ERR_ENVIRONMENT);
  _mm_setcsr(saved_mxcsr | 0x0040u); /* DAZ */
  CHECK(et_fp8_e5m2_encode_scalar_v1(1.0f, ET_FP8_OVERFLOW_REJECT, &dummy_q) ==
        ET_FP8_ERR_ENVIRONMENT);
  _mm_setcsr(saved_mxcsr);
#endif

  /* 5. Bridge zero-allocation & misaligned raw-pointer checks. */
  uint8_t raw_state[sizeof(et_fp8_delayed_scale_state_v1) + 4u];
  void *unaligned_state = raw_state + 1u;
  CHECK(et_fp8_aot_delayed_scale_init(unaligned_state, ET_FP8_FORMAT_E4M3FN, 4,
                                      1, 1e-6, 1e-4, 65536.0) == ET_FP8_OK);
  CHECK(et_fp8_aot_delayed_scale_update(unaligned_state, 16.0) == ET_FP8_OK);
  uint8_t raw_sc[8], raw_am[8];
  CHECK(et_fp8_aot_delayed_scale_compute(unaligned_state, raw_sc + 1,
                                         raw_am + 1) == ET_FP8_OK);
  CHECK(fabs(et_fp8_aot_load_f32(raw_sc + 1, 0) - 14.0) <= 1e-6);
  CHECK(fabs(et_fp8_aot_load_f32(raw_am + 1, 0) - 16.0) <= 1e-6);

  forbid_alloc = 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }
  test_exhaustive_codecs();
  test_delayed_scaling();
  test_block_gemm_parity_outliers_and_ste_fd();
  test_negative_and_atomicity();
  puts("FP8 NATIVE SUITE (ZERO-ALLOC WRAPPED): PASS");
  return 0;
}
