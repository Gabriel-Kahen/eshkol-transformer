#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/w4a32_quant_abi.h"
#include "w4a32_quant_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Allocation trap enforcing ZERO heap allocations on hot quant/dequant/GEMM
 * kernels and buffer-level container serialize/parse. */
static int hot_kernel_active = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!hot_kernel_active);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!hot_kernel_active);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!hot_kernel_active);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!hot_kernel_active);
  __real_free(ptr);
}

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31u);
}

static float det_uniform(uint64_t *rng, float lo, float hi) {
  const uint64_t w = splitmix64(rng);
  const double u = (double)(w >> 11u) * (1.0 / 9007199254740992.0);
  return (float)((double)lo + u * ((double)hi - (double)lo));
}

static void print_f32_hex_vec(const float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (i != 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

static void print_bytes_hex(const uint8_t *b, size_t n) {
  for (size_t i = 0; i < n; ++i)
    printf("%02x", (unsigned)b[i]);
  putchar('\n');
}

/* ---- 1. Exact IEEE-754 binary16 edge cases ---- */
static void test_f16_codec(void) {
  uint16_t bits = 0u;
  float val = 0.0f;

  hot_kernel_active = 1;
  CHECK(et_w4a32_f32_to_f16(0.0f, &bits) == ET_W4A32_OK && bits == 0x0000u);
  CHECK(et_w4a32_f32_to_f16(-0.0f, &bits) == ET_W4A32_OK && bits == 0x8000u);
  CHECK(et_w4a32_f32_to_f16(1.0f, &bits) == ET_W4A32_OK && bits == 0x3c00u);
  CHECK(et_w4a32_f32_to_f16(-2.0f, &bits) == ET_W4A32_OK && bits == 0xc000u);
  CHECK(et_w4a32_f32_to_f16(65504.0f, &bits) == ET_W4A32_OK && bits == 0x7bffu);
  CHECK(et_w4a32_f32_to_f16(-65504.0f, &bits) == ET_W4A32_OK &&
        bits == 0xfbffu);
  /* 65519.0f rounds down to 65504.0f (finite), 65520.0f overflows to Inf. */
  CHECK(et_w4a32_f32_to_f16(65519.0f, &bits) == ET_W4A32_OK && bits == 0x7bffu);
  CHECK(et_w4a32_f32_to_f16(65520.0f, &bits) == ET_W4A32_ERR_OVERFLOW);
  CHECK(et_w4a32_f32_to_f16(-65520.0f, &bits) == ET_W4A32_ERR_OVERFLOW);
  CHECK(et_w4a32_f32_to_f16(INFINITY, &bits) == ET_W4A32_ERR_NONFINITE);
  CHECK(et_w4a32_f32_to_f16(-INFINITY, &bits) == ET_W4A32_ERR_NONFINITE);
  CHECK(et_w4a32_f32_to_f16(NAN, &bits) == ET_W4A32_ERR_NONFINITE);

  /* Subnormal boundaries: 2^-24 = 0x0001, 2^-25 ties to even -> 0x0000. */
  CHECK(et_w4a32_f32_to_f16(ldexpf(1.0f, -24), &bits) == ET_W4A32_OK &&
        bits == 0x0001u);
  CHECK(et_w4a32_f32_to_f16(ldexpf(1.0f, -25), &bits) == ET_W4A32_OK &&
        bits == 0x0000u);
  CHECK(et_w4a32_f16_to_f32(0x0001u, &val) == ET_W4A32_OK &&
        val == ldexpf(1.0f, -24));
  CHECK(et_w4a32_f16_to_f32(0x7bffu, &val) == ET_W4A32_OK && val == 65504.0f);
  CHECK(et_w4a32_f16_to_f32(0x7c00u, &val) == ET_W4A32_ERR_NONFINITE);
  CHECK(et_w4a32_f16_to_f32(0xfc00u, &val) == ET_W4A32_ERR_NONFINITE);
  CHECK(et_w4a32_f16_to_f32(0x7e00u, &val) == ET_W4A32_ERR_NONFINITE);
  hot_kernel_active = 0;
}

/* ---- 2. Explicit GGML Q4_0 split-half nibble bit-layout verification ---- */
static void test_q4_0_exact_bit_layout(void) {
  /* Construct a 1x32 weight vector with exact integer multiples of d = +0.5f:
   * Make w[0] = -4.0f (largest magnitude, negative -> signed_max = -4.0f,
   * d = -4.0 / -8.0 = +0.5f -> f16 0x3800 LE bytes {0x00, 0x38}).
   * Elements 0..15 encode (q0 - 8) = [-8, -7, ..., +7] -> q0 = 0..15.
   * Elements 16..31 encode (q1 - 8) = [+7, +6, ..., -8] -> q1 = 15..0.
   * Wait: only w[0] and w[31] have magnitude 4.0f; since w[0] = -4.0f is seen
   * first with strict '>', signed_max = -4.0f -> d = +0.5f exact! */
  float w[32];
  for (uint32_t j = 0; j < 16u; ++j) {
    w[j] = 0.5f * (float)((int32_t)j - 8);
    w[j + 16u] = 0.5f * (float)(7 - (int32_t)j);
  }
  uint8_t blk[18];
  float deq[32];

  hot_kernel_active = 1;
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, w, blk, sizeof(blk)) == ET_W4A32_OK);
  /* Verify LE f16 scale = +0.5f (0x3800). */
  CHECK(blk[0] == 0x00u && blk[1] == 0x38u);
  /* Verify split-half nibble layout: low nibble = elem j, high = elem j+16. */
  for (uint32_t j = 0; j < 16u; ++j) {
    const uint8_t expected_lo = (uint8_t)j;
    const uint8_t expected_hi = (uint8_t)(15u - j);
    CHECK((blk[2u + j] & 0x0Fu) == expected_lo);
    CHECK(((blk[2u + j] >> 4u) & 0x0Fu) == expected_hi);
  }
  CHECK(et_w4a32_q4_0_dequantize_v1(1u, 32u, blk, sizeof(blk), deq) ==
        ET_W4A32_OK);
  hot_kernel_active = 0;

  /* Exact reconstruction with zero error on grid values! */
  for (uint32_t i = 0; i < 32u; ++i)
    CHECK(deq[i] == w[i]);

  /* Also verify all-zero block produces d = 0x0000 and qs = 0x88. */
  float zeros[32] = {0.0f};
  hot_kernel_active = 1;
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, zeros, blk, sizeof(blk)) ==
        ET_W4A32_OK);
  CHECK(blk[0] == 0x00u && blk[1] == 0x00u);
  for (uint32_t j = 0; j < 16u; ++j)
    CHECK(blk[2u + j] == 0x88u);
  CHECK(et_w4a32_q4_0_dequantize_v1(1u, 32u, blk, sizeof(blk), deq) ==
        ET_W4A32_OK);
  hot_kernel_active = 0;
  for (uint32_t i = 0; i < 32u; ++i)
    CHECK(deq[i] == 0.0f);
}

/* ---- 3. Q4_ASYM exact layout & skewed-distribution advantage over Q4_0 ---- */
static void test_asym_layout_and_skewed_advantage(void) {
  /* Skewed positive weights in [+10.0, +13.75]:
   * Q4_ASYM uses m = 10.0, d = 3.75 / 15 = 0.25 (exact f16!) -> 0.0 error!
   * Q4_0 must cover [-13.75, +13.75] with step ~1.72 -> huge error! */
  const uint32_t groups[3] = {32u, 64u, 128u};
  for (size_t gi = 0; gi < 3u; ++gi) {
    const uint32_t G = groups[gi];
    float w[128];
    for (uint32_t i = 0; i < G; ++i)
      w[i] = 10.0f + 0.25f * (float)(i % 16u);
    size_t asym_bytes = 0u;
    size_t q4_bytes = 0u;
    CHECK(et_w4a32_packed_bytes(ET_W4A32_FORMAT_Q4_ASYM, G, 1u, G,
                                &asym_bytes) == ET_W4A32_OK);
    CHECK(et_w4a32_packed_bytes(ET_W4A32_FORMAT_Q4_0, 32u, 1u, G, &q4_bytes) ==
          ET_W4A32_OK);
    uint8_t asym_buf[128];
    uint8_t q4_buf[128];
    float deq_asym[128];
    float deq_q4[128];

    hot_kernel_active = 1;
    CHECK(et_w4a32_asym_quantize_v1(1u, G, G, w, asym_buf, asym_bytes) ==
          ET_W4A32_OK);
    CHECK(et_w4a32_asym_dequantize_v1(1u, G, G, asym_buf, asym_bytes,
                                      deq_asym) == ET_W4A32_OK);
    CHECK(et_w4a32_q4_0_quantize_v1(1u, G, w, q4_buf, q4_bytes) == ET_W4A32_OK);
    CHECK(et_w4a32_q4_0_dequantize_v1(1u, G, q4_buf, q4_bytes, deq_q4) ==
          ET_W4A32_OK);
    hot_kernel_active = 0;

    /* Check exact f16 header bytes for d = 0.25 (0x3400), m = 10.0 (0x4900). */
    CHECK(asym_buf[0] == 0x00u && asym_buf[1] == 0x34u);
    CHECK(asym_buf[2] == 0x00u && asym_buf[3] == 0x49u);

    double err_asym = 0.0;
    double err_q4 = 0.0;
    for (uint32_t i = 0; i < G; ++i) {
      const double ea = fabs((double)deq_asym[i] - (double)w[i]);
      const double eq = fabs((double)deq_q4[i] - (double)w[i]);
      if (ea > err_asym)
        err_asym = ea;
      if (eq > err_q4)
        err_q4 = eq;
    }
    CHECK(err_asym == 0.0);
    CHECK(err_q4 > 0.5);
  }
}

/* ---- 4. Fused GEMV (M=1) & GEMM (M>1) parity with dequant-F32 matmul ---- */
static void test_fused_gemv_and_gemm_parity(void) {
  uint64_t rng = UINT64_C(0x123456789abcdef0);
  const uint32_t ms[3] = {1u, 4u, 8u};
  const uint32_t N = 24u;
  const uint32_t K = 128u;

  float *x = (float *)calloc(8u * K, sizeof(float));
  float *w = (float *)calloc(N * K, sizeof(float));
  float *bias = (float *)calloc(N, sizeof(float));
  float *w_deq = (float *)calloc(N * K, sizeof(float));
  float *y_fused = (float *)calloc(8u * N, sizeof(float));
  float *y_ref = (float *)calloc(8u * N, sizeof(float));
  CHECK(x && w && bias && w_deq && y_fused && y_ref);

  for (size_t i = 0; i < 8u * K; ++i)
    x[i] = det_uniform(&rng, -1.5f, 1.5f);
  for (size_t i = 0; i < (size_t)N * K; ++i)
    w[i] = det_uniform(&rng, -0.75f, 1.25f);
  for (size_t i = 0; i < N; ++i)
    bias[i] = det_uniform(&rng, -0.5f, 0.5f);

  /* Test Q4_0 and Q4_ASYM (G=32, 64, 128), with and without bias. */
  const struct {
    et_w4a32_format fmt;
    uint32_t g;
  } configs[4] = {
      {ET_W4A32_FORMAT_Q4_0, 32u},
      {ET_W4A32_FORMAT_Q4_ASYM, 32u},
      {ET_W4A32_FORMAT_Q4_ASYM, 64u},
      {ET_W4A32_FORMAT_Q4_ASYM, 128u},
  };

  double max_parity_err = 0.0;
  for (size_t ci = 0; ci < 4u; ++ci) {
    size_t packed_bytes = 0u;
    CHECK(et_w4a32_packed_bytes(configs[ci].fmt, configs[ci].g, N, K,
                                &packed_bytes) == ET_W4A32_OK);
    uint8_t *packed = (uint8_t *)calloc(packed_bytes, 1u);
    CHECK(packed != NULL);

    hot_kernel_active = 1;
    if (configs[ci].fmt == ET_W4A32_FORMAT_Q4_0) {
      CHECK(et_w4a32_q4_0_quantize_v1(N, K, w, packed, packed_bytes) ==
            ET_W4A32_OK);
      CHECK(et_w4a32_q4_0_dequantize_v1(N, K, packed, packed_bytes, w_deq) ==
            ET_W4A32_OK);
    } else {
      CHECK(et_w4a32_asym_quantize_v1(N, K, configs[ci].g, w, packed,
                                      packed_bytes) == ET_W4A32_OK);
      CHECK(et_w4a32_asym_dequantize_v1(N, K, configs[ci].g, packed,
                                        packed_bytes, w_deq) == ET_W4A32_OK);
    }

    for (size_t mi = 0; mi < 3u; ++mi) {
      const uint32_t M = ms[mi];
      for (int use_bias = 0; use_bias <= 1; ++use_bias) {
        const float *bptr = use_bias ? bias : NULL;
        if (configs[ci].fmt == ET_W4A32_FORMAT_Q4_0) {
          CHECK(et_w4a32_q4_0_gemm_v1(M, N, K, x, packed, packed_bytes, bptr,
                                      y_fused) == ET_W4A32_OK);
        } else {
          CHECK(et_w4a32_asym_gemm_v1(M, N, K, configs[ci].g, x, packed,
                                      packed_bytes, bptr, y_fused) ==
                ET_W4A32_OK);
        }
        CHECK(et_w4a32_f32_gemm_v1(M, N, K, x, w_deq, bptr, y_ref) ==
              ET_W4A32_OK);
        for (size_t idx = 0; idx < (size_t)M * N; ++idx) {
          const double err = fabs((double)y_fused[idx] - (double)y_ref[idx]);
          if (err > max_parity_err)
            max_parity_err = err;
          CHECK(err <= 1e-6);
        }
      }
    }
    hot_kernel_active = 0;
    free(packed);
  }

  printf("W4A32 fused GEMV/GEMM parity max_abs_err=%.3e: PASS\n",
         max_parity_err);

  free(x);
  free(w);
  free(bias);
  free(w_deq);
  free(y_fused);
  free(y_ref);
}

/* ---- 5. Container ETQ41000 save/load, tamper, symlink, & error preservation ---- */
static void test_container_and_negatives(void) {
  float w[2 * 64];
  for (uint32_t i = 0; i < 128u; ++i)
    w[i] = -1.0f + 0.02f * (float)i;

  size_t q4_bytes = 0u;
  size_t ckpt_bytes = 0u;
  hot_kernel_active = 1;
  CHECK(et_w4a32_packed_bytes(ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &q4_bytes) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_container_size(ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u,
                                &ckpt_bytes) == ET_W4A32_OK);
  CHECK(q4_bytes == 72u && ckpt_bytes == 152u);

  uint8_t packed[72];
  uint8_t wire[152];
  uint8_t parsed[72];
  size_t out_size = 0u;
  et_w4a32_container_meta meta;

  CHECK(et_w4a32_q4_0_quantize_v1(2u, 64u, w, packed, sizeof(packed)) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_container_serialize(ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, packed,
                                     sizeof(packed), wire, sizeof(wire),
                                     &out_size) == ET_W4A32_OK);
  CHECK(out_size == sizeof(wire));
  memset(parsed, 0xEE, sizeof(parsed));
  CHECK(et_w4a32_container_parse(wire, sizeof(wire), ET_W4A32_FORMAT_Q4_0, 32u,
                                 2u, 64u, &meta, parsed, sizeof(parsed)) ==
        ET_W4A32_OK);
  CHECK(memcmp(parsed, packed, sizeof(packed)) == 0);
  CHECK(meta.version_major == 1u && meta.version_minor == 0u &&
        meta.format == ET_W4A32_FORMAT_Q4_0 && meta.group_size == 32u &&
        meta.rows_n == 2u && meta.cols_k == 64u && meta.payload_bytes == 72u);

  /* Buffer-level tamper rejections preserve destination bytes. */
  uint8_t bad_wire[152];
  uint8_t sentinel[72];
  memset(sentinel, 0x5Au, sizeof(sentinel));

  /* Corrupt payload byte -> checksum mismatch. */
  memcpy(bad_wire, wire, sizeof(wire));
  bad_wire[ET_W4A32_CONTAINER_HEADER_BYTES] ^= 0x80u;
  memcpy(parsed, sentinel, sizeof(sentinel));
  CHECK(et_w4a32_container_parse(bad_wire, sizeof(bad_wire),
                                 ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &meta,
                                 parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_CHECKSUM);
  CHECK(memcmp(parsed, sentinel, sizeof(sentinel)) == 0);

  /* Truncated buffer -> truncated error. */
  CHECK(et_w4a32_container_parse(wire, sizeof(wire) - 1u, ET_W4A32_FORMAT_Q4_0,
                                 32u, 2u, 64u, &meta, parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_TRUNCATED);
  CHECK(memcmp(parsed, sentinel, sizeof(sentinel)) == 0);

  /* Bad magic -> format error. */
  memcpy(bad_wire, wire, sizeof(wire));
  bad_wire[0] = 'Z';
  CHECK(et_w4a32_container_parse(bad_wire, sizeof(bad_wire),
                                 ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &meta,
                                 parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_FORMAT);

  /* Bad version -> version error. */
  memcpy(bad_wire, wire, sizeof(wire));
  bad_wire[8] = 2u;
  CHECK(et_w4a32_container_parse(bad_wire, sizeof(bad_wire),
                                 ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &meta,
                                 parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_VERSION);

  /* Non-zero reserved flags with recomputed valid SHA-256 -> format error. */
  memcpy(bad_wire, wire, sizeof(wire));
  bad_wire[40] = 1u;
  CHECK(et_w4a32_sha256_bytes(bad_wire, sizeof(bad_wire) - 32u,
                              bad_wire + sizeof(bad_wire) - 32u) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_container_parse(bad_wire, sizeof(bad_wire),
                                 ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &meta,
                                 parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_FORMAT);

  /* Non-finite f16 scale inside payload with valid SHA-256 -> nonfinite error. */
  memcpy(bad_wire, wire, sizeof(wire));
  bad_wire[ET_W4A32_CONTAINER_HEADER_BYTES] = 0x00u;
  bad_wire[ET_W4A32_CONTAINER_HEADER_BYTES + 1u] = 0x7cu; /* +Inf f16 */
  CHECK(et_w4a32_sha256_bytes(bad_wire, sizeof(bad_wire) - 32u,
                              bad_wire + sizeof(bad_wire) - 32u) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_container_parse(bad_wire, sizeof(bad_wire),
                                 ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u, &meta,
                                 parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(parsed, sentinel, sizeof(sentinel)) == 0);
  hot_kernel_active = 0;

  /* Atomic file save + load + permission check (0600, non-executable) + symlink. */
  char dir_tpl[] = "/tmp/w4a32-native-test.XXXXXX";
  char *dir = mkdtemp(dir_tpl);
  CHECK(dir != NULL);
  char file_path[256];
  char link_path[256];
  CHECK(snprintf(file_path, sizeof(file_path), "%s/weights.etq4", dir) > 0);
  CHECK(snprintf(link_path, sizeof(link_path), "%s/weights.link", dir) > 0);

  CHECK(et_w4a32_container_save(file_path, ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u,
                                packed, sizeof(packed)) == ET_W4A32_OK);
  struct stat st;
  CHECK(stat(file_path, &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  CHECK((st.st_mode & 0111) == 0);

  memset(parsed, 0, sizeof(parsed));
  CHECK(et_w4a32_container_load(file_path, ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u,
                                &meta, parsed, sizeof(parsed)) == ET_W4A32_OK);
  CHECK(memcmp(parsed, packed, sizeof(packed)) == 0);

  /* Symlink rejection. */
  CHECK(symlink(file_path, link_path) == 0);
  memcpy(parsed, sentinel, sizeof(sentinel));
  CHECK(et_w4a32_container_load(link_path, ET_W4A32_FORMAT_Q4_0, 32u, 2u, 64u,
                                &meta, parsed, sizeof(parsed)) ==
        ET_W4A32_ERR_SYMLINK);
  CHECK(memcmp(parsed, sentinel, sizeof(sentinel)) == 0);

  CHECK(unlink(link_path) == 0);
  CHECK(unlink(file_path) == 0);
  CHECK(rmdir(dir) == 0);
}

/* ---- 6. Negative kernel tests: shape, non-finite, overflow, aliasing, FP env ---- */
static void test_negative_kernels(void) {
  float w[64] = {0.0f};
  float x[64] = {0.0f};
  float bias[2] = {0.0f, 0.0f};
  uint8_t q_buf[40];
  float y[4] = {11.0f, 22.0f, 33.0f, 44.0f};
  const float y_orig[4] = {11.0f, 22.0f, 33.0f, 44.0f};
  uint8_t q_orig[40];
  memset(q_buf, 0x3Cu, sizeof(q_buf));
  memcpy(q_orig, q_buf, sizeof(q_orig));

  hot_kernel_active = 1;

  /* K % 32 != 0 for Q4_0 and invalid group sizes for Q4_ASYM. */
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 30u, w, q_buf, 18u) == ET_W4A32_ERR_SHAPE);
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 48u, w, q_buf, 18u) == ET_W4A32_ERR_SHAPE);
  CHECK(et_w4a32_asym_quantize_v1(1u, 64u, 16u, w, q_buf, 40u) ==
        ET_W4A32_ERR_SHAPE);
  CHECK(et_w4a32_asym_quantize_v1(1u, 96u, 64u, w, q_buf, 40u) ==
        ET_W4A32_ERR_SHAPE);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);

  /* Non-finite input in second block preserves first block untouched (two-pass!). */
  w[63] = NAN;
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 64u, w, q_buf, 36u) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);
  CHECK(et_w4a32_asym_quantize_v1(1u, 64u, 32u, w, q_buf, 40u) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);
  w[63] = 0.0f;

  /* f16 scale overflow in second block preserves output untouched.
   * Q4_0 overflows when max_abs/8 >= 65520 (w=600000 -> d=75000).
   * Q4_ASYM overflows when (max-min)/15 >= 65520 (w=1e6 -> d=66666.7)
   * or when |w_min| >= 65520 (w=-70000 -> m=-70000). */
  w[40] = 600000.0f;
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 64u, w, q_buf, 36u) ==
        ET_W4A32_ERR_OVERFLOW);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);
  w[40] = 1000000.0f;
  CHECK(et_w4a32_asym_quantize_v1(1u, 64u, 32u, w, q_buf, 40u) ==
        ET_W4A32_ERR_OVERFLOW);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);
  w[40] = -70000.0f;
  CHECK(et_w4a32_asym_quantize_v1(1u, 64u, 32u, w, q_buf, 40u) ==
        ET_W4A32_ERR_OVERFLOW);
  CHECK(memcmp(q_buf, q_orig, sizeof(q_orig)) == 0);
  w[40] = 0.0f;

  /* Valid quantize for subsequent GEMM negative tests. */
  for (uint32_t i = 0; i < 64u; ++i)
    w[i] = 0.125f * (float)((int32_t)(i % 9u) - 4);
  CHECK(et_w4a32_q4_0_quantize_v1(2u, 32u, w, q_buf, 36u) == ET_W4A32_OK);

  /* Non-finite X preserves Y. */
  x[15] = INFINITY;
  CHECK(et_w4a32_q4_0_gemm_v1(2u, 2u, 32u, x, q_buf, 36u, bias, y) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(y, y_orig, sizeof(y_orig)) == 0);
  x[15] = 0.0f;

  /* Non-finite bias preserves Y. */
  bias[1] = -NAN;
  CHECK(et_w4a32_q4_0_gemm_v1(2u, 2u, 32u, x, q_buf, 36u, bias, y) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(y, y_orig, sizeof(y_orig)) == 0);
  bias[1] = 0.0f;

  /* Non-finite f16 scale in block 1 preserves Y. */
  uint8_t bad_q[36];
  memcpy(bad_q, q_buf, 36u);
  bad_q[18] = 0x01u;
  bad_q[19] = 0x7cu; /* NaN f16 */
  CHECK(et_w4a32_q4_0_gemm_v1(2u, 2u, 32u, x, bad_q, 36u, bias, y) ==
        ET_W4A32_ERR_NONFINITE);
  CHECK(memcmp(y, y_orig, sizeof(y_orig)) == 0);

  /* Output f32 overflow on second row preserves first row (dry-run pass!). */
  float x_huge[64] = {0.0f};
  x_huge[0] = 1.0f;
  x_huge[32] = 1.0e38f; /* row 1 overflows f32 when multiplied by 4.0 * 8 */
  uint8_t q_big[36];
  float w_big[64] = {0.0f};
  w_big[0] = 32.0f;
  w_big[32] = 32.0f;
  CHECK(et_w4a32_q4_0_quantize_v1(2u, 32u, w_big, q_big, 36u) == ET_W4A32_OK);
  CHECK(et_w4a32_q4_0_gemm_v1(2u, 2u, 32u, x_huge, q_big, 36u, NULL, y) ==
        ET_W4A32_ERR_OVERFLOW);
  CHECK(memcmp(y, y_orig, sizeof(y_orig)) == 0);

  /* Aliased buffers rejected and output preserved. */
  CHECK(et_w4a32_q4_0_gemm_v1(1u, 2u, 32u, x, q_buf, 36u, bias, x) ==
        ET_W4A32_ERR_ALIAS);
  CHECK(et_w4a32_q4_0_gemm_v1(1u, 2u, 32u, x, q_buf, 36u, y, y) ==
        ET_W4A32_ERR_ALIAS);
  CHECK(memcmp(y, y_orig, sizeof(y_orig)) == 0);

  /* Misaligned float buffer rejected; odd-offset packed byte buffer succeeds
   * cleanly (proving byte-wise little-endian loads/stores with zero UB). */
  uint8_t unaligned[256] = {0u};
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, (const float *)(unaligned + 1u),
                                  q_buf, 18u) == ET_W4A32_ERR_ALIGNMENT);
  uint8_t odd_packed[19] = {0u};
  float odd_deq[32] = {0.0f};
  float odd_y[1] = {0.0f};
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, w, odd_packed + 1u, 18u) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_q4_0_dequantize_v1(1u, 32u, odd_packed + 1u, 18u, odd_deq) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, x, odd_packed + 1u, 18u, NULL,
                              odd_y) == ET_W4A32_OK);

  /* Non-default FP rounding mode rejected explicitly. */
  const int saved_round = fegetround();
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, w, q_buf, 18u) ==
        ET_W4A32_ERR_FP_ENV);
  CHECK(fesetround(saved_round) == 0);

  hot_kernel_active = 0;
}

/* ---- 7. Bridge zero-alloc hot dispatch verification ---- */
static void test_bridge_hot_paths(void) {
  et_w4a32_aot_ctx *ctx = NULL;
  CHECK(et_w4a32_aot_create(4, 64, 64, 32, ET_W4A32_FORMAT_Q4_ASYM, 64, &ctx) ==
        ET_W4A32_OK);
  for (int64_t i = 0; i < et_w4a32_aot_x_count(ctx); ++i)
    CHECK(et_w4a32_aot_set_x(ctx, i, 0.05 * (double)((i % 11) - 5)) ==
          ET_W4A32_OK);
  for (int64_t i = 0; i < et_w4a32_aot_w1_count(ctx); ++i)
    CHECK(et_w4a32_aot_set_w1(ctx, i, 0.03 * (double)((i % 13) - 6)) ==
          ET_W4A32_OK);
  for (int64_t i = 0; i < et_w4a32_aot_w2_count(ctx); ++i)
    CHECK(et_w4a32_aot_set_w2(ctx, i, 0.25 + 0.04 * (double)(i % 17)) ==
          ET_W4A32_OK);
  for (int64_t i = 0; i < et_w4a32_aot_b1_count(ctx); ++i)
    CHECK(et_w4a32_aot_set_b1(ctx, i, 0.01 * (double)(i % 7)) == ET_W4A32_OK);
  for (int64_t i = 0; i < et_w4a32_aot_b2_count(ctx); ++i)
    CHECK(et_w4a32_aot_set_b2(ctx, i, -0.02 * (double)(i % 5)) == ET_W4A32_OK);

  /* Bridge quantize, dequantize, fused MLP, and ref MLP allocate zero heap. */
  hot_kernel_active = 1;
  CHECK(et_w4a32_aot_quantize_all(ctx) == ET_W4A32_OK);
  CHECK(et_w4a32_aot_dequantize_all(ctx) == ET_W4A32_OK);
  CHECK(et_w4a32_aot_mlp_forward_fused(ctx, 1) == ET_W4A32_OK);
  CHECK(et_w4a32_aot_mlp_forward_deq_ref(ctx, 1) == ET_W4A32_OK);
  CHECK(et_w4a32_aot_mlp_forward_fused(ctx, 4) == ET_W4A32_OK);
  CHECK(et_w4a32_aot_mlp_forward_deq_ref(ctx, 4) == ET_W4A32_OK);
  hot_kernel_active = 0;

  for (int64_t i = 0; i < et_w4a32_aot_y_count(ctx, 4); ++i) {
    const double diff =
        fabs(et_w4a32_aot_get_y_fused(ctx, i) - et_w4a32_aot_get_y_ref(ctx, i));
    CHECK(diff <= 1e-5);
  }
  et_w4a32_aot_destroy(ctx);
}

/* ---- 8. Oracle output emitter for independent Python reference ---- */
static void run_oracle_mode(void) {
  const struct {
    et_w4a32_format fmt;
    uint32_t g;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    int skewed;
    int use_bias;
  } cases[8] = {
      {ET_W4A32_FORMAT_Q4_0, 32u, 1u, 4u, 32u, 0, 0},
      {ET_W4A32_FORMAT_Q4_0, 32u, 1u, 6u, 64u, 0, 1},
      {ET_W4A32_FORMAT_Q4_0, 32u, 5u, 8u, 96u, 1, 1},
      {ET_W4A32_FORMAT_Q4_0, 32u, 8u, 12u, 128u, 0, 1},
      {ET_W4A32_FORMAT_Q4_ASYM, 32u, 1u, 6u, 64u, 1, 1},
      {ET_W4A32_FORMAT_Q4_ASYM, 64u, 4u, 8u, 128u, 1, 1},
      {ET_W4A32_FORMAT_Q4_ASYM, 128u, 1u, 6u, 128u, 1, 0},
      {ET_W4A32_FORMAT_Q4_ASYM, 128u, 8u, 10u, 256u, 0, 1},
  };

  uint64_t rng = UINT64_C(0x57344133324f5241);
  for (size_t ci = 0; ci < 8u; ++ci) {
    const uint32_t M = cases[ci].m;
    const uint32_t N = cases[ci].n;
    const uint32_t K = cases[ci].k;
    const uint32_t G = cases[ci].g;
    const et_w4a32_format fmt = cases[ci].fmt;

    size_t packed_bytes = 0u;
    CHECK(et_w4a32_packed_bytes(fmt, G, N, K, &packed_bytes) == ET_W4A32_OK);

    float *x = (float *)calloc((size_t)M * K, sizeof(float));
    float *w = (float *)calloc((size_t)N * K, sizeof(float));
    float *bias = (float *)calloc(N, sizeof(float));
    uint8_t *packed = (uint8_t *)calloc(packed_bytes, 1u);
    float *deq = (float *)calloc((size_t)N * K, sizeof(float));
    float *y_fused = (float *)calloc((size_t)M * N, sizeof(float));
    CHECK(x && w && bias && packed && deq && y_fused);

    for (size_t i = 0; i < (size_t)M * K; ++i)
      x[i] = det_uniform(&rng, -2.0f, 2.0f);
    for (size_t i = 0; i < (size_t)N * K; ++i)
      w[i] = cases[ci].skewed ? det_uniform(&rng, 0.5f, 3.5f)
                              : det_uniform(&rng, -1.5f, 1.5f);
    for (size_t i = 0; i < N; ++i)
      bias[i] = det_uniform(&rng, -0.5f, 0.5f);

    hot_kernel_active = 1;
    if (fmt == ET_W4A32_FORMAT_Q4_0) {
      CHECK(et_w4a32_q4_0_quantize_v1(N, K, w, packed, packed_bytes) ==
            ET_W4A32_OK);
      CHECK(et_w4a32_q4_0_dequantize_v1(N, K, packed, packed_bytes, deq) ==
            ET_W4A32_OK);
      CHECK(et_w4a32_q4_0_gemm_v1(M, N, K, x, packed, packed_bytes,
                                  cases[ci].use_bias ? bias : NULL,
                                  y_fused) == ET_W4A32_OK);
    } else {
      CHECK(et_w4a32_asym_quantize_v1(N, K, G, w, packed, packed_bytes) ==
            ET_W4A32_OK);
      CHECK(et_w4a32_asym_dequantize_v1(N, K, G, packed, packed_bytes, deq) ==
            ET_W4A32_OK);
      CHECK(et_w4a32_asym_gemm_v1(M, N, K, G, x, packed, packed_bytes,
                                  cases[ci].use_bias ? bias : NULL,
                                  y_fused) == ET_W4A32_OK);
    }
    hot_kernel_active = 0;

    printf("CASE %u %u %u %u %u %d\n", (unsigned)fmt, (unsigned)G, (unsigned)M,
           (unsigned)N, (unsigned)K, cases[ci].use_bias);
    print_f32_hex_vec(x, (size_t)M * K);
    print_f32_hex_vec(w, (size_t)N * K);
    if (cases[ci].use_bias)
      print_f32_hex_vec(bias, N);
    else
      puts("NONE");
    print_bytes_hex(packed, packed_bytes);
    print_f32_hex_vec(deq, (size_t)N * K);
    print_f32_hex_vec(y_fused, (size_t)M * N);

    free(x);
    free(w);
    free(bias);
    free(packed);
    free(deq);
    free(y_fused);
  }

  /* Emit a serialized ETQ41000 container + negative status vector for Python
   * oracle verification of SHA-256 wire format and negative contract codes. */
  float w_ct[32];
  for (uint32_t i = 0; i < 32u; ++i)
    w_ct[i] = 0.25f * (float)((int32_t)(i % 16u) - 8);
  uint8_t blk_ct[18];
  uint8_t wire_ct[98];
  size_t wire_len = 0u;
  hot_kernel_active = 1;
  CHECK(et_w4a32_q4_0_quantize_v1(1u, 32u, w_ct, blk_ct, sizeof(blk_ct)) ==
        ET_W4A32_OK);
  CHECK(et_w4a32_container_serialize(ET_W4A32_FORMAT_Q4_0, 32u, 1u, 32u, blk_ct,
                                     sizeof(blk_ct), wire_ct, sizeof(wire_ct),
                                     &wire_len) == ET_W4A32_OK);
  CHECK(wire_len == sizeof(wire_ct));

  float w_bad_k[30] = {0.0f};
  float w_ovf[32] = {0.0f};
  w_ovf[0] = 600000.0f;
  float x_nan[32] = {0.0f};
  x_nan[0] = NAN;
  uint8_t bad_scale[18] = {0u};
  bad_scale[0] = 0x00u;
  bad_scale[1] = 0x7cu;
  float y_tmp[1] = {0.0f};
  uint8_t out_tmp[18] = {0u};
  et_w4a32_container_meta meta_tmp;
  uint8_t wire_tamper[98];
  memcpy(wire_tamper, wire_ct, sizeof(wire_ct));
  wire_tamper[48] ^= 0x01u;

  const int32_t s_k_mod =
      et_w4a32_q4_0_quantize_v1(1u, 30u, w_bad_k, out_tmp, sizeof(out_tmp));
  const int32_t s_ovf =
      et_w4a32_q4_0_quantize_v1(1u, 32u, w_ovf, out_tmp, sizeof(out_tmp));
  const int32_t s_nan_x =
      et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, x_nan, blk_ct, sizeof(blk_ct), NULL, y_tmp);
  const int32_t s_inf_d =
      et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, w_ct, bad_scale, sizeof(bad_scale), NULL, y_tmp);
  const int32_t s_alias =
      et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, w_ct, blk_ct, sizeof(blk_ct), NULL, w_ct);
  const int32_t s_cksum = et_w4a32_container_parse(
      wire_tamper, sizeof(wire_tamper), ET_W4A32_FORMAT_Q4_0, 32u, 1u, 32u,
      &meta_tmp, out_tmp, sizeof(out_tmp));
  const int32_t s_trunc = et_w4a32_container_parse(
      wire_ct, sizeof(wire_ct) - 3u, ET_W4A32_FORMAT_Q4_0, 32u, 1u, 32u,
      &meta_tmp, out_tmp, sizeof(out_tmp));
  hot_kernel_active = 0;

  puts("CONTAINER");
  print_bytes_hex(blk_ct, sizeof(blk_ct));
  print_bytes_hex(wire_ct, sizeof(wire_ct));
  printf("NEGATIVES %d %d %d %d %d %d %d\n", (int)s_k_mod, (int)s_ovf,
         (int)s_nan_x, (int)s_inf_d, (int)s_alias, (int)s_cksum, (int)s_trunc);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_mode();
    return 0;
  }
  test_f16_codec();
  test_q4_0_exact_bit_layout();
  test_asym_layout_and_skewed_advantage();
  test_fused_gemv_and_gemm_parity();
  test_container_and_negatives();
  test_negative_kernels();
  test_bridge_hot_paths();
  puts("W4A32 native C11 tests (zero-alloc hot kernels + layout + container + negatives): PASS");
  return 0;
}
