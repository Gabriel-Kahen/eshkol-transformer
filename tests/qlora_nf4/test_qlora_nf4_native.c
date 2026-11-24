#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/qlora_nf4_abi.h"
#include "qlora_nf4_eshkol_bridge.h"

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

/* Allocation trap enforcing ZERO heap allocations on hot quant/dequant/forward/
 * backward/merge/AdamW/train_step/serialize/parse kernels. */
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

static double rand_unit_open(uint64_t *rng) {
  return ((double)(splitmix64(rng) >> 11u) + 0.5) * (1.0 / 9007199254740992.0);
}

static void fill_gaussian(float *dst, size_t n, double std_dev, uint64_t *rng) {
  static const double two_pi = 6.283185307179586476925286766559;
  size_t i = 0u;
  while (i < n) {
    const double u1 = rand_unit_open(rng);
    const double u2 = rand_unit_open(rng);
    const double mag = std_dev * sqrt(-2.0 * log(u1));
    dst[i++] = (float)(mag * cos(two_pi * u2));
    if (i < n)
      dst[i++] = (float)(mag * sin(two_pi * u2));
  }
}

static double calc_rmse(const float *a, const float *b, size_t n) {
  double acc = 0.0;
  for (size_t i = 0u; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    acc += d * d;
  }
  return sqrt(acc / (double)n);
}

static double uniform_int4_rmse(const float *w, size_t n, size_t bs) {
  double sse = 0.0;
  const size_t blocks = n / bs;
  for (size_t b = 0u; b < blocks; ++b) {
    const float *blk = w + b * bs;
    float absmax = 0.0f;
    for (size_t j = 0u; j < bs; ++j) {
      const float a = fabsf(blk[j]);
      if (a > absmax)
        absmax = a;
    }
    if (absmax == 0.0f)
      continue;
    const double scale = (double)absmax / 7.0;
    for (size_t j = 0u; j < bs; ++j) {
      double q = floor((double)blk[j] / scale + 0.5);
      if (q < -7.0)
        q = -7.0;
      if (q > 7.0)
        q = 7.0;
      const double w_hat = (double)(float)(q * scale);
      const double d = (double)blk[j] - w_hat;
      sse += d * d;
    }
  }
  return sqrt(sse / (double)n);
}

static void print_f32_hex_vec(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i != 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

static void print_bytes_hex(const uint8_t *b, size_t n) {
  for (size_t i = 0u; i < n; ++i)
    printf("%02x", (unsigned)b[i]);
  putchar('\n');
}

/* ---- 1. NF4 codebook, packing, midpoint tie-break & zero blocks ---- */
static void test_nf4_codebook_and_packing(void) {
  static const float expected[16] = {
      -1.0f,
      -0.6961928009986877f,
      -0.5250730514526367f,
      -0.39491748809814453f,
      -0.28444138169288635f,
      -0.18477343022823334f,
      -0.09105003625154495f,
      0.0f,
      0.07958029955625534f,
      0.16093020141124725f,
      0.24611230194568634f,
      0.33791524171829224f,
      0.44070982933044434f,
      0.5626170039176941f,
      0.7229568362236023f,
      1.0f};
  for (int i = 0; i < 16; ++i) {
    CHECK(ET_NF4_CODEBOOK[i] == expected[i]);
    if (i > 0)
      CHECK(ET_NF4_CODEBOOK[i] > ET_NF4_CODEBOOK[i - 1]);
  }

  /* Exact codebook round-trip & low-nibble-first packing check (block_size=32). */
  float w[32];
  for (int i = 0; i < 16; ++i) {
    w[i] = ET_NF4_CODEBOOK[i] * 2.0f;
    w[16 + i] = ET_NF4_CODEBOOK[15 - i] * 2.0f;
  }
  et_qlora_quant_spec sq = {1u, 32u, 32u, 0u, ET_QLORA_QUANT_NF4_SQ, 0u};
  uint8_t packed[20];
  float deq[32];

  hot_kernel_active = 1;
  CHECK(et_qlora_nf4_quantize(&sq, w, packed, sizeof(packed)) == ET_QLORA_OK);
  /* Pair p=0 has nibbles (0, 1) -> low=0, high=1 => 0x10. */
  for (int p = 0; p < 8; ++p) {
    const uint8_t lo = (uint8_t)(2 * p);
    const uint8_t hi = (uint8_t)(2 * p + 1);
    CHECK(packed[p] == (uint8_t)(lo | (hi << 4u)));
  }
  CHECK(et_qlora_nf4_dequantize(&sq, packed, sizeof(packed), deq) ==
        ET_QLORA_OK);
  hot_kernel_active = 0;
  for (int i = 0; i < 32; ++i)
    CHECK(deq[i] == w[i]);

  /* Strict midpoint tie-break to larger index (using exact binary32 midpoints
   * with absmax = 1.0f). */
  float w_mid[32];
  w_mid[0] = -1.0f;
  w_mid[1] = 1.0f;
  for (int i = 0; i < 15; ++i) {
    const double mid =
        0.5 * ((double)ET_NF4_CODEBOOK[i] + (double)ET_NF4_CODEBOOK[i + 1]);
    /* Just above midpoint -> i+1; just below midpoint -> i. */
    w_mid[2 + 2 * i] = nextafterf((float)mid, 2.0f);
    w_mid[3 + 2 * i] = nextafterf((float)mid, -2.0f);
  }
  hot_kernel_active = 1;
  CHECK(et_qlora_nf4_quantize(&sq, w_mid, packed, sizeof(packed)) ==
        ET_QLORA_OK);
  hot_kernel_active = 0;
  for (int i = 0; i < 15; ++i) {
    const uint8_t byte = packed[1 + i];
    CHECK((byte & 0x0Fu) == (uint8_t)(i + 1));
    CHECK(((byte >> 4u) & 0x0Fu) == (uint8_t)i);
  }

  /* Exact midpoint where (double)float_mid == exact_mid: e.g. 0.5*(-1 + -0.6961928)
   * or 0.5*(0.0 + 0.0795803). Verify exact tie goes to larger index. */
  float w_tie[32];
  memset(w_tie, 0, sizeof(w_tie));
  w_tie[0] = 1.0f;
  const float exact_mid_7_8 = (float)(0.5 * ((double)ET_NF4_CODEBOOK[7] +
                                             (double)ET_NF4_CODEBOOK[8]));
  /* Check exact representability in binary32 (halving a binary32 float is exact!). */
  CHECK((double)exact_mid_7_8 ==
        0.5 * ((double)ET_NF4_CODEBOOK[7] + (double)ET_NF4_CODEBOOK[8]));
  w_tie[1] = exact_mid_7_8;
  hot_kernel_active = 1;
  CHECK(et_qlora_nf4_quantize(&sq, w_tie, packed, sizeof(packed)) ==
        ET_QLORA_OK);
  hot_kernel_active = 0;
  CHECK((packed[0] & 0x0Fu) == 15u);
  CHECK(((packed[0] >> 4u) & 0x0Fu) == 8u); /* tie-break to larger index 8! */

  /* All-zero block -> nibbles 0x77, scale 0.0f, exact 0.0f dequant. */
  float w_zero[32];
  memset(w_zero, 0, sizeof(w_zero));
  hot_kernel_active = 1;
  CHECK(et_qlora_nf4_quantize(&sq, w_zero, packed, sizeof(packed)) ==
        ET_QLORA_OK);
  for (int p = 0; p < 16; ++p)
    CHECK(packed[p] == 0x77u);
  CHECK(et_qlora_nf4_dequantize(&sq, packed, sizeof(packed), deq) ==
        ET_QLORA_OK);
  hot_kernel_active = 0;
  for (int i = 0; i < 32; ++i)
    CHECK(deq[i] == 0.0f);
}

/* ---- 2. Single-Quant vs Double-Quant footprint & Gaussian RMSE vs INT4 ---- */
static void test_sq_dq_footprint_and_gaussian_rmse(void) {
  const uint32_t rows = 128u, cols = 128u;
  const size_t n = (size_t)rows * (size_t)cols;
  float *w = (float *)malloc(n * sizeof(float));
  float *deq_sq = (float *)malloc(n * sizeof(float));
  float *deq_dq = (float *)malloc(n * sizeof(float));
  CHECK(w != NULL && deq_sq != NULL && deq_dq != NULL);

  uint64_t rng = UINT64_C(0x1122334455667788);
  fill_gaussian(w, n, 0.25, &rng);

  const et_qlora_quant_spec sq = {rows, cols, 64u, 0u, ET_QLORA_QUANT_NF4_SQ,
                                  0u};
  const et_qlora_quant_spec dq = {rows, cols, 64u, 256u, ET_QLORA_QUANT_NF4_DQ,
                                  0u};
  et_qlora_quant_footprint fp_sq, fp_dq;
  CHECK(et_qlora_packed_footprint(&sq, &fp_sq) == ET_QLORA_OK);
  CHECK(et_qlora_packed_footprint(&dq, &fp_dq) == ET_QLORA_OK);

  CHECK(fp_sq.f32_bytes == 65536u);
  CHECK(fp_sq.packed_bytes == 9216u);
  CHECK(fabs(fp_sq.bits_per_param - 4.5) < 1e-12);

  CHECK(fp_dq.f32_bytes == 65536u);
  CHECK(fp_dq.packed_bytes == 8456u);
  CHECK(fabs(fp_dq.bits_per_param - 4.12890625) < 1e-12);
  CHECK((double)fp_dq.f32_bytes / (double)fp_dq.packed_bytes > 7.7);

  uint8_t *p_sq = (uint8_t *)malloc((size_t)fp_sq.packed_bytes);
  uint8_t *p_dq = (uint8_t *)malloc((size_t)fp_dq.packed_bytes);
  CHECK(p_sq != NULL && p_dq != NULL);

  hot_kernel_active = 1;
  CHECK(et_qlora_nf4_quantize(&sq, w, p_sq, (size_t)fp_sq.packed_bytes) ==
        ET_QLORA_OK);
  CHECK(et_qlora_nf4_dequantize(&sq, p_sq, (size_t)fp_sq.packed_bytes,
                                deq_sq) == ET_QLORA_OK);
  CHECK(et_qlora_nf4_quantize(&dq, w, p_dq, (size_t)fp_dq.packed_bytes) ==
        ET_QLORA_OK);
  CHECK(et_qlora_nf4_dequantize(&dq, p_dq, (size_t)fp_dq.packed_bytes,
                                deq_dq) == ET_QLORA_OK);
  hot_kernel_active = 0;

  const double rmse_sq = calc_rmse(w, deq_sq, n);
  const double rmse_dq = calc_rmse(w, deq_dq, n);
  const double rmse_u4 = uniform_int4_rmse(w, n, 64u);
  CHECK(rmse_sq > 0.0 && rmse_dq > 0.0 && rmse_u4 > 0.0);
  CHECK(rmse_sq < rmse_u4);
  CHECK(rmse_dq < rmse_u4);
  /* DQ scale quantization adds negligible RMSE overhead (< 2%). */
  CHECK(rmse_dq < rmse_sq * 1.02);

  printf("QLORA NATIVE GAUSSIAN RMSE: nf4_sq=%.6e nf4_dq=%.6e unif_int4=%.6e "
         "dq_bpp=%.4f ratio=%.3fx\n",
         rmse_sq, rmse_dq, rmse_u4, fp_dq.bits_per_param,
         (double)fp_dq.f32_bytes / (double)fp_dq.packed_bytes);

  /* Also verify all valid block_size x dq_block_size combinations (including
   * partial DQ groups). */
  const uint32_t bss[3] = {32u, 64u, 128u};
  const uint32_t dqss[4] = {16u, 32u, 64u, 256u};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      et_qlora_quant_spec sp = {64u, 128u, bss[i], dqss[j],
                                ET_QLORA_QUANT_NF4_DQ, 0u};
      et_qlora_quant_footprint fp;
      CHECK(et_qlora_packed_footprint(&sp, &fp) == ET_QLORA_OK);
      uint8_t *buf = (uint8_t *)malloc((size_t)fp.packed_bytes);
      float *out = (float *)malloc(64u * 128u * sizeof(float));
      CHECK(buf != NULL && out != NULL);
      hot_kernel_active = 1;
      CHECK(et_qlora_nf4_quantize(&sp, w, buf, (size_t)fp.packed_bytes) ==
            ET_QLORA_OK);
      CHECK(et_qlora_nf4_dequantize(&sp, buf, (size_t)fp.packed_bytes, out) ==
            ET_QLORA_OK);
      hot_kernel_active = 0;
      CHECK(calc_rmse(w, out, 64u * 128u) < uniform_int4_rmse(w, 64u * 128u, bss[i]));
      free(buf);
      free(out);
    }
  }

  free(w);
  free(deq_sq);
  free(deq_dq);
  free(p_sq);
  free(p_dq);
}

/* ---- 3. Fused Forward vs Merge Parity + Central FD Gradient Checks ---- */
static void test_fused_linear_parity_and_fd(void) {
  const et_qlora_quant_mode modes[2] = {ET_QLORA_QUANT_NF4_SQ,
                                        ET_QLORA_QUANT_NF4_DQ};
  const uint32_t bss[2] = {32u, 64u};
  const uint32_t dqss[2] = {0u, 16u};

  double global_max_fd_err = 0.0;
  double global_max_merge_err = 0.0;
  int total_fd_checks = 0;

  for (int c = 0; c < 2; ++c) {
    et_qlora_linear_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.quant.rows_out = 16u;
    cfg.quant.cols_in = 64u;
    cfg.quant.block_size = bss[c];
    cfg.quant.dq_block_size = dqss[c];
    cfg.quant.mode = modes[c];
    cfg.m = 4u;
    cfg.r = 4u;
    cfg.alpha = 8.0f;

    et_qlora_quant_footprint fp;
    CHECK(et_qlora_packed_footprint(&cfg.quant, &fp) == ET_QLORA_OK);

    const size_t x_n = (size_t)cfg.m * cfg.quant.cols_in;
    const size_t w_n = (size_t)cfg.quant.rows_out * cfg.quant.cols_in;
    const size_t y_n = (size_t)cfg.m * cfg.quant.rows_out;
    const size_t a_n = (size_t)cfg.r * cfg.quant.cols_in;
    const size_t b_n = (size_t)cfg.quant.rows_out * cfg.r;

    float *w_f32 = (float *)malloc(w_n * sizeof(float));
    float *w_merged = (float *)malloc(w_n * sizeof(float));
    float *x = (float *)malloc(x_n * sizeof(float));
    float *bias = (float *)malloc(cfg.quant.rows_out * sizeof(float));
    float *a = (float *)malloc(a_n * sizeof(float));
    float *b = (float *)malloc(b_n * sizeof(float));
    float *y_base = (float *)malloc(y_n * sizeof(float));
    float *y_fused = (float *)malloc(y_n * sizeof(float));
    float *y_merged = (float *)malloc(y_n * sizeof(float));
    float *dy = (float *)malloc(y_n * sizeof(float));
    float *dx = (float *)malloc(x_n * sizeof(float));
    float *da = (float *)malloc(a_n * sizeof(float));
    float *db = (float *)malloc(b_n * sizeof(float));
    float *dbias = (float *)malloc(cfg.quant.rows_out * sizeof(float));
    uint8_t *w_nf4 = (uint8_t *)malloc((size_t)fp.packed_bytes);
    uint8_t *w_nf4_copy = (uint8_t *)malloc((size_t)fp.packed_bytes);
    CHECK(w_f32 && w_merged && x && bias && a && b && y_base && y_fused &&
          y_merged && dy && dx && da && db && dbias && w_nf4 && w_nf4_copy);

    uint64_t rng = UINT64_C(0xabcdef0123456789) + (uint64_t)c * 997u;
    fill_gaussian(w_f32, w_n, 0.3, &rng);
    fill_gaussian(x, x_n, 0.4, &rng);
    fill_gaussian(bias, cfg.quant.rows_out, 0.1, &rng);
    fill_gaussian(dy, y_n, 0.3, &rng);

    hot_kernel_active = 1;
    CHECK(et_qlora_nf4_quantize(&cfg.quant, w_f32, w_nf4,
                                (size_t)fp.packed_bytes) == ET_QLORA_OK);
    CHECK(et_qlora_init_lora(cfg.quant.cols_in, cfg.quant.rows_out, cfg.r,
                             42u + (uint64_t)c, a, b) == ET_QLORA_OK);

    /* Zero-init B must match frozen NF4 base-only forward bitwise! */
    CHECK(et_qlora_linear_forward(&cfg, x, w_nf4, (size_t)fp.packed_bytes, bias,
                                  NULL, NULL, y_base) == ET_QLORA_OK);
    CHECK(et_qlora_linear_forward(&cfg, x, w_nf4, (size_t)fp.packed_bytes, bias,
                                  a, b, y_fused) == ET_QLORA_OK);
    hot_kernel_active = 0;
    CHECK(memcmp(y_base, y_fused, y_n * sizeof(float)) == 0);

    /* Non-zero B: test fused forward vs merge_to_f32 parity and backward FD. */
    fill_gaussian(b, b_n, 0.2, &rng);
    memcpy(w_nf4_copy, w_nf4, (size_t)fp.packed_bytes);
    uint8_t sha_before[32], sha_after[32];
    CHECK(et_qlora_sha256_bytes(w_nf4, (size_t)fp.packed_bytes, sha_before) ==
          ET_QLORA_OK);

    hot_kernel_active = 1;
    CHECK(et_qlora_linear_forward(&cfg, x, w_nf4, (size_t)fp.packed_bytes, bias,
                                  a, b, y_fused) == ET_QLORA_OK);
    CHECK(et_qlora_merge_to_f32(&cfg, w_nf4, (size_t)fp.packed_bytes, a, b,
                                w_merged) == ET_QLORA_OK);
    CHECK(et_qlora_f32_linear_forward(cfg.m, cfg.quant.rows_out,
                                      cfg.quant.cols_in, x, w_merged, bias,
                                      y_merged) == ET_QLORA_OK);
    CHECK(et_qlora_linear_backward(&cfg, x, w_nf4, (size_t)fp.packed_bytes, a,
                                   b, dy, dx, da, db, dbias) == ET_QLORA_OK);
    hot_kernel_active = 0;

    CHECK(memcmp(w_nf4, w_nf4_copy, (size_t)fp.packed_bytes) == 0);
    CHECK(et_qlora_sha256_bytes(w_nf4, (size_t)fp.packed_bytes, sha_after) ==
          ET_QLORA_OK);
    CHECK(memcmp(sha_before, sha_after, 32u) == 0);

    for (size_t i = 0u; i < y_n; ++i) {
      const double diff = fabs((double)y_fused[i] - (double)y_merged[i]);
      if (diff > global_max_merge_err)
        global_max_merge_err = diff;
      CHECK(diff < 1e-6);
    }

    /* Central finite-difference checks on dX, dA, dB, dbias.
     * L = <dY, Y> is exact-linear in each of X, A, B, bias individually
     * (zero truncation error); h = 2^-4 avoids f32 cancellation noise. */
    const float eps = 0.0625f;
    float *y_tmp = (float *)malloc(y_n * sizeof(float));
    CHECK(y_tmp != NULL);

    float *targets[4] = {x, a, b, bias};
    const float *analytic[4] = {dx, da, db, dbias};
    const size_t lengths[4] = {x_n, a_n, b_n, cfg.quant.rows_out};
    for (int t = 0; t < 4; ++t) {
      const size_t step = (lengths[t] > 24u) ? (lengths[t] / 16u) : 1u;
      for (size_t idx = 0u; idx < lengths[t]; idx += step) {
        const float orig = targets[t][idx];

        targets[t][idx] = orig + eps;
        hot_kernel_active = 1;
        CHECK(et_qlora_linear_forward(&cfg, x, w_nf4, (size_t)fp.packed_bytes,
                                      bias, a, b, y_tmp) == ET_QLORA_OK);
        hot_kernel_active = 0;
        double lp = 0.0;
        for (size_t i = 0u; i < y_n; ++i)
          lp += (double)dy[i] * (double)y_tmp[i];

        targets[t][idx] = orig - eps;
        hot_kernel_active = 1;
        CHECK(et_qlora_linear_forward(&cfg, x, w_nf4, (size_t)fp.packed_bytes,
                                      bias, a, b, y_tmp) == ET_QLORA_OK);
        hot_kernel_active = 0;
        double lm = 0.0;
        for (size_t i = 0u; i < y_n; ++i)
          lm += (double)dy[i] * (double)y_tmp[i];

        targets[t][idx] = orig;
        const double fd = (lp - lm) / (2.0 * (double)eps);
        const double err = fabs((double)analytic[t][idx] - fd);
        if (err > global_max_fd_err)
          global_max_fd_err = err;
        CHECK(err < 1e-5);
        ++total_fd_checks;
      }
    }

    free(y_tmp);
    free(w_f32);
    free(w_merged);
    free(x);
    free(bias);
    free(a);
    free(b);
    free(y_base);
    free(y_fused);
    free(y_merged);
    free(dy);
    free(dx);
    free(da);
    free(db);
    free(dbias);
    free(w_nf4);
    free(w_nf4_copy);
  }

  printf("QLORA NATIVE LINEAR PARITY & FD: checks=%d max_fd_err=%.3e "
         "max_merge_diff=%.3e PASS\n",
         total_fd_checks, global_max_fd_err, global_max_merge_err);
}

/* ---- 4. 2-Layer QLoRA MLP AdamW 12-Step & 6+6 Checkpoint Resume ---- */
static void test_mlp_train_and_checkpoint_resume(void) {
  et_qlora_mlp_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.m = 4u;
  cfg.d_in = 64u;
  cfg.d_hidden = 64u;
  cfg.d_out = 32u;
  cfg.r = 4u;
  cfg.block_size = 64u;
  cfg.dq_block_size = 16u;
  cfg.quant_mode = ET_QLORA_QUANT_NF4_DQ;
  cfg.alpha = 8.0f;

  et_qlora_quant_spec s1 = {cfg.d_hidden, cfg.d_in, cfg.block_size,
                            cfg.dq_block_size, cfg.quant_mode, 0u};
  et_qlora_quant_spec s2 = {cfg.d_out, cfg.d_hidden, cfg.block_size,
                            cfg.dq_block_size, cfg.quant_mode, 0u};
  et_qlora_quant_footprint fp1, fp2;
  CHECK(et_qlora_packed_footprint(&s1, &fp1) == ET_QLORA_OK);
  CHECK(et_qlora_packed_footprint(&s2, &fp2) == ET_QLORA_OK);

  size_t scratch_n = 0u, ckpt_bytes = 0u;
  CHECK(et_qlora_mlp_scratch_floats(&cfg, &scratch_n) == ET_QLORA_OK);
  CHECK(et_qlora_checkpoint_size(&cfg, &ckpt_bytes) == ET_QLORA_OK);

  const size_t w1_n = (size_t)cfg.d_hidden * cfg.d_in;
  const size_t w2_n = (size_t)cfg.d_out * cfg.d_hidden;
  const size_t x_n = (size_t)cfg.m * cfg.d_in;
  const size_t y_n = (size_t)cfg.m * cfg.d_out;
  const size_t h_n = (size_t)cfg.m * cfg.d_hidden;
  const size_t a1_n = (size_t)cfg.r * cfg.d_in;
  const size_t b1_n = (size_t)cfg.d_hidden * cfg.r;
  const size_t a2_n = (size_t)cfg.r * cfg.d_hidden;
  const size_t b2_n = (size_t)cfg.d_out * cfg.r;

  float *w1_f32 = (float *)malloc(w1_n * sizeof(float));
  float *w2_f32 = (float *)malloc(w2_n * sizeof(float));
  float *b1_f = (float *)malloc(cfg.d_hidden * sizeof(float));
  float *b2_f = (float *)malloc(cfg.d_out * sizeof(float));
  float *x = (float *)malloc(x_n * sizeof(float));
  float *target = (float *)malloc(y_n * sizeof(float));
  uint8_t *w1_nf4 = (uint8_t *)malloc((size_t)fp1.packed_bytes);
  uint8_t *w2_nf4 = (uint8_t *)malloc((size_t)fp2.packed_bytes);
  float *scratch = (float *)malloc(scratch_n * sizeof(float));
  float *z1 = (float *)malloc(h_n * sizeof(float));
  float *h1 = (float *)malloc(h_n * sizeof(float));
  float *y_full = (float *)malloc(y_n * sizeof(float));
  float *y_res = (float *)malloc(y_n * sizeof(float));
  CHECK(w1_f32 && w2_f32 && b1_f && b2_f && x && target && w1_nf4 && w2_nf4 &&
        scratch && z1 && h1 && y_full && y_res);

  uint64_t rng = UINT64_C(0x42424242abcdef01);
  fill_gaussian(w1_f32, w1_n, 0.25, &rng);
  fill_gaussian(w2_f32, w2_n, 0.25, &rng);
  fill_gaussian(b1_f, cfg.d_hidden, 0.05, &rng);
  fill_gaussian(b2_f, cfg.d_out, 0.05, &rng);
  fill_gaussian(x, x_n, 0.5, &rng);

  CHECK(et_qlora_nf4_quantize(&s1, w1_f32, w1_nf4, (size_t)fp1.packed_bytes) ==
        ET_QLORA_OK);
  CHECK(et_qlora_nf4_quantize(&s2, w2_f32, w2_nf4, (size_t)fp2.packed_bytes) ==
        ET_QLORA_OK);

  uint8_t base_sha_init[32], base_sha_after[32];
  CHECK(et_qlora_mlp_base_sha256(&cfg, w1_nf4, (size_t)fp1.packed_bytes, w2_nf4,
                                 (size_t)fp2.packed_bytes,
                                 base_sha_init) == ET_QLORA_OK);

  CHECK(et_qlora_mlp_forward(&cfg, x, w1_nf4, (size_t)fp1.packed_bytes, b1_f,
                             NULL, NULL, w2_nf4, (size_t)fp2.packed_bytes, b2_f,
                             NULL, NULL, z1, h1, y_full) == ET_QLORA_OK);
  for (size_t i = 0u; i < y_n; ++i)
    target[i] = y_full[i] + (float)((rand_unit_open(&rng) - 0.5) * 0.8);

  /* Run A: uninterrupted 12-step AdamW training. */
  float *a1 = (float *)calloc(a1_n, sizeof(float));
  float *b1 = (float *)calloc(b1_n, sizeof(float));
  float *a2 = (float *)calloc(a2_n, sizeof(float));
  float *b2 = (float *)calloc(b2_n, sizeof(float));
  float *ma1 = (float *)calloc(a1_n, sizeof(float));
  float *va1 = (float *)calloc(a1_n, sizeof(float));
  float *mb1 = (float *)calloc(b1_n, sizeof(float));
  float *vb1 = (float *)calloc(b1_n, sizeof(float));
  float *ma2 = (float *)calloc(a2_n, sizeof(float));
  float *va2 = (float *)calloc(a2_n, sizeof(float));
  float *mb2 = (float *)calloc(b2_n, sizeof(float));
  float *vb2 = (float *)calloc(b2_n, sizeof(float));
  CHECK(a1 && b1 && a2 && b2 && ma1 && va1 && mb1 && vb1 && ma2 && va2 && mb2 &&
        vb2);

  CHECK(et_qlora_init_lora(cfg.d_in, cfg.d_hidden, cfg.r, 1001u, a1, b1) ==
        ET_QLORA_OK);
  CHECK(et_qlora_init_lora(cfg.d_hidden, cfg.d_out, cfg.r, 2002u, a2, b2) ==
        ET_QLORA_OK);

  et_qlora_adamw_config opt_full = {0.003f, 0.9f, 0.999f, 1e-8f, 0.01f, 0u, 0u};
  double losses_full[12];
  for (int s = 0; s < 12; ++s) {
    hot_kernel_active = 1;
    CHECK(et_qlora_mlp_train_step(
              &cfg, &opt_full, x, target, w1_nf4, (size_t)fp1.packed_bytes,
              b1_f, w2_nf4, (size_t)fp2.packed_bytes, b2_f, a1, b1, a2, b2, ma1,
              va1, mb1, vb1, ma2, va2, mb2, vb2, scratch, scratch_n,
              &losses_full[s]) == ET_QLORA_OK);
    hot_kernel_active = 0;
    if (s > 0)
      CHECK(losses_full[s] < losses_full[s - 1]);
  }
  CHECK(opt_full.step == 12u);
  CHECK(et_qlora_mlp_forward(&cfg, x, w1_nf4, (size_t)fp1.packed_bytes, b1_f,
                             a1, b1, w2_nf4, (size_t)fp2.packed_bytes, b2_f, a2,
                             b2, z1, h1, y_full) == ET_QLORA_OK);
  CHECK(et_qlora_mlp_base_sha256(&cfg, w1_nf4, (size_t)fp1.packed_bytes, w2_nf4,
                                 (size_t)fp2.packed_bytes,
                                 base_sha_after) == ET_QLORA_OK);
  CHECK(memcmp(base_sha_init, base_sha_after, 32u) == 0);

  /* Run B: 6 steps -> save ETQL1000 -> poison -> reload -> 6 steps. */
  float *ra1 = (float *)calloc(a1_n, sizeof(float));
  float *rb1 = (float *)calloc(b1_n, sizeof(float));
  float *ra2 = (float *)calloc(a2_n, sizeof(float));
  float *rb2 = (float *)calloc(b2_n, sizeof(float));
  float *rma1 = (float *)calloc(a1_n, sizeof(float));
  float *rva1 = (float *)calloc(a1_n, sizeof(float));
  float *rmb1 = (float *)calloc(b1_n, sizeof(float));
  float *rvb1 = (float *)calloc(b1_n, sizeof(float));
  float *rma2 = (float *)calloc(a2_n, sizeof(float));
  float *rva2 = (float *)calloc(a2_n, sizeof(float));
  float *rmb2 = (float *)calloc(b2_n, sizeof(float));
  float *rvb2 = (float *)calloc(b2_n, sizeof(float));
  CHECK(ra1 && rb1 && ra2 && rb2 && rma1 && rva1 && rmb1 && rvb1 && rma2 &&
        rva2 && rmb2 && rvb2);

  CHECK(et_qlora_init_lora(cfg.d_in, cfg.d_hidden, cfg.r, 1001u, ra1, rb1) ==
        ET_QLORA_OK);
  CHECK(et_qlora_init_lora(cfg.d_hidden, cfg.d_out, cfg.r, 2002u, ra2, rb2) ==
        ET_QLORA_OK);
  et_qlora_adamw_config opt_res = {0.003f, 0.9f, 0.999f, 1e-8f, 0.01f, 0u, 0u};

  double losses_res[12];
  for (int s = 0; s < 6; ++s) {
    CHECK(et_qlora_mlp_train_step(
              &cfg, &opt_res, x, target, w1_nf4, (size_t)fp1.packed_bytes, b1_f,
              w2_nf4, (size_t)fp2.packed_bytes, b2_f, ra1, rb1, ra2, rb2, rma1,
              rva1, rmb1, rvb1, rma2, rva2, rmb2, rvb2, scratch, scratch_n,
              &losses_res[s]) == ET_QLORA_OK);
    CHECK(losses_res[s] == losses_full[s]);
  }
  CHECK(opt_res.step == 6u);

  /* Buffer-level serialize/parse with zero heap allocations. */
  uint8_t *ckpt_buf = (uint8_t *)malloc(ckpt_bytes);
  CHECK(ckpt_buf != NULL);
  size_t ser_size = 0u;
  hot_kernel_active = 1;
  CHECK(et_qlora_checkpoint_serialize(
            &cfg, &opt_res, base_sha_init, ra1, rb1, ra2, rb2, rma1, rva1, rmb1,
            rvb1, rma2, rva2, rmb2, rvb2, ckpt_buf, ckpt_bytes,
            &ser_size) == ET_QLORA_OK);
  hot_kernel_active = 0;
  CHECK(ser_size == ckpt_bytes);

  /* File-level atomic 0600 save + O_NOFOLLOW reload. */
  char dir_tmpl[] = "/tmp/etql_test_XXXXXX";
  CHECK(mkdtemp(dir_tmpl) != NULL);
  char ckpt_path[256], bad_path[256], sym_path[256];
  snprintf(ckpt_path, sizeof(ckpt_path), "%s/step6.etql", dir_tmpl);
  snprintf(bad_path, sizeof(bad_path), "%s/bad.etql", dir_tmpl);
  snprintf(sym_path, sizeof(sym_path), "%s/sym.etql", dir_tmpl);

  CHECK(et_qlora_checkpoint_save(ckpt_path, &cfg, &opt_res, base_sha_init, ra1,
                                 rb1, ra2, rb2, rma1, rva1, rmb1, rvb1, rma2,
                                 rva2, rmb2, rvb2) == ET_QLORA_OK);
  struct stat sb;
  CHECK(stat(ckpt_path, &sb) == 0);
  CHECK((sb.st_mode & 0777) == 0600);
  CHECK((size_t)sb.st_size == ckpt_bytes);

  /* Poison all trainable weights and AdamW state before reload. */
  for (size_t i = 0u; i < a1_n; ++i)
    ra1[i] = rma1[i] = rva1[i] = 99.0f;
  for (size_t i = 0u; i < b1_n; ++i)
    rb1[i] = rmb1[i] = rvb1[i] = 99.0f;
  for (size_t i = 0u; i < a2_n; ++i)
    ra2[i] = rma2[i] = rva2[i] = 99.0f;
  for (size_t i = 0u; i < b2_n; ++i)
    rb2[i] = rmb2[i] = rvb2[i] = 99.0f;
  opt_res.step = 999u;

  et_qlora_checkpoint_meta meta;
  CHECK(et_qlora_checkpoint_load(ckpt_path, &cfg, base_sha_init, &meta,
                                 &opt_res, ra1, rb1, ra2, rb2, rma1, rva1, rmb1,
                                 rvb1, rma2, rva2, rmb2, rvb2) == ET_QLORA_OK);
  CHECK(meta.version_major == 1u && meta.version_minor == 0u);
  CHECK(meta.adamw_step == 6u && opt_res.step == 6u);
  CHECK(memcmp(meta.base_nf4_sha256, base_sha_init, 32u) == 0);

  for (int s = 6; s < 12; ++s) {
    CHECK(et_qlora_mlp_train_step(
              &cfg, &opt_res, x, target, w1_nf4, (size_t)fp1.packed_bytes, b1_f,
              w2_nf4, (size_t)fp2.packed_bytes, b2_f, ra1, rb1, ra2, rb2, rma1,
              rva1, rmb1, rvb1, rma2, rva2, rmb2, rvb2, scratch, scratch_n,
              &losses_res[s]) == ET_QLORA_OK);
    CHECK(losses_res[s] == losses_full[s]);
  }
  CHECK(et_qlora_mlp_forward(&cfg, x, w1_nf4, (size_t)fp1.packed_bytes, b1_f,
                             ra1, rb1, w2_nf4, (size_t)fp2.packed_bytes, b2_f,
                             ra2, rb2, z1, h1, y_res) == ET_QLORA_OK);
  CHECK(memcmp(a1, ra1, a1_n * sizeof(float)) == 0);
  CHECK(memcmp(b1, rb1, b1_n * sizeof(float)) == 0);
  CHECK(memcmp(a2, ra2, a2_n * sizeof(float)) == 0);
  CHECK(memcmp(b2, rb2, b2_n * sizeof(float)) == 0);
  CHECK(memcmp(ma1, rma1, a1_n * sizeof(float)) == 0);
  CHECK(memcmp(va1, rva1, a1_n * sizeof(float)) == 0);
  CHECK(memcmp(y_full, y_res, y_n * sizeof(float)) == 0);

  /* ---- Checkpoint tamper, mismatch, symlink & non-regular negatives ---- */
  /* Corrupt checksum (flip payload bit). */
  ckpt_buf[ET_QLORA_CHECKPOINT_HEADER_BYTES] ^= 0x80u;
  hot_kernel_active = 1;
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes, &cfg, base_sha_init,
                                  &meta, &opt_res, ra1, rb1, ra2, rb2, rma1,
                                  rva1, rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_CHECKSUM);
  hot_kernel_active = 0;
  ckpt_buf[ET_QLORA_CHECKPOINT_HEADER_BYTES] ^= 0x80u;

  /* Truncated. */
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes - 1u, &cfg,
                                  base_sha_init, &meta, &opt_res, ra1, rb1, ra2,
                                  rb2, rma1, rva1, rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_TRUNCATED);

  /* Bad magic. */
  ckpt_buf[0] = 'Z';
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes, &cfg, base_sha_init,
                                  &meta, &opt_res, ra1, rb1, ra2, rb2, rma1,
                                  rva1, rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_FORMAT);
  ckpt_buf[0] = 'E';

  /* Bad version. */
  ckpt_buf[8] = 2u;
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes, &cfg, base_sha_init,
                                  &meta, &opt_res, ra1, rb1, ra2, rb2, rma1,
                                  rva1, rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_VERSION);
  ckpt_buf[8] = 1u;

  /* Base hash mismatch. */
  uint8_t wrong_sha[32];
  memcpy(wrong_sha, base_sha_init, 32u);
  wrong_sha[31] ^= 0x01u;
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes, &cfg, wrong_sha, &meta,
                                  &opt_res, ra1, rb1, ra2, rb2, rma1, rva1,
                                  rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_BASE_HASH);

  /* Wrong expected rank/config. */
  et_qlora_mlp_config wrong_cfg = cfg;
  wrong_cfg.alpha = 16.0f;
  CHECK(et_qlora_checkpoint_parse(ckpt_buf, ckpt_bytes, &wrong_cfg,
                                  base_sha_init, &meta, &opt_res, ra1, rb1, ra2,
                                  rb2, rma1, rva1, rmb1, rvb1, rma2, rva2, rmb2,
                                  rvb2) == ET_QLORA_ERR_SHAPE);

  /* Symlink rejection. */
  CHECK(symlink(ckpt_path, sym_path) == 0);
  CHECK(et_qlora_checkpoint_load(sym_path, &cfg, base_sha_init, &meta, &opt_res,
                                 ra1, rb1, ra2, rb2, rma1, rva1, rmb1, rvb1,
                                 rma2, rva2, rmb2,
                                 rvb2) == ET_QLORA_ERR_SYMLINK);

  /* Non-regular file (/dev/null) rejection. */
  CHECK(et_qlora_checkpoint_load("/dev/null", &cfg, base_sha_init, &meta,
                                 &opt_res, ra1, rb1, ra2, rb2, rma1, rva1, rmb1,
                                 rvb1, rma2, rva2, rmb2,
                                 rvb2) == ET_QLORA_ERR_FORMAT);

  /* Oversized file rejection. */
  FILE *f = fopen(bad_path, "wb");
  CHECK(f != NULL);
  CHECK(fwrite(ckpt_buf, 1u, ckpt_bytes, f) == ckpt_bytes);
  fputc(0, f);
  CHECK(fclose(f) == 0);
  CHECK(et_qlora_checkpoint_load(bad_path, &cfg, base_sha_init, &meta, &opt_res,
                                 ra1, rb1, ra2, rb2, rma1, rva1, rmb1, rvb1,
                                 rma2, rva2, rmb2,
                                 rvb2) == ET_QLORA_ERR_FORMAT);

  (void)unlink(sym_path);
  (void)unlink(bad_path);
  (void)unlink(ckpt_path);
  (void)rmdir(dir_tmpl);

  printf("QLORA NATIVE 12-STEP & 6+6 RESUME: init=%.6e final=%.6e "
         "bitwise_resume=PASS\n",
         losses_full[0], losses_full[11]);

  free(ckpt_buf);
  free(ra1);
  free(rb1);
  free(ra2);
  free(rb2);
  free(rma1);
  free(rva1);
  free(rmb1);
  free(rvb1);
  free(rma2);
  free(rva2);
  free(rmb2);
  free(rvb2);
  free(a1);
  free(b1);
  free(a2);
  free(b2);
  free(ma1);
  free(va1);
  free(mb1);
  free(vb1);
  free(ma2);
  free(va2);
  free(mb2);
  free(vb2);
  free(w1_f32);
  free(w2_f32);
  free(b1_f);
  free(b2_f);
  free(x);
  free(target);
  free(w1_nf4);
  free(w2_nf4);
  free(scratch);
  free(z1);
  free(h1);
  free(y_full);
  free(y_res);
}

/* ---- 5. Boundary, Overflow Atomicity, Aliasing & FP-Env Negatives ---- */
static void test_negatives_and_atomicity(void) {
  /* Invalid specs. */
  et_qlora_quant_spec bad = {16u, 64u, 48u, 0u, ET_QLORA_QUANT_NF4_SQ, 0u};
  CHECK(et_qlora_validate_quant_spec(&bad) == ET_QLORA_ERR_SHAPE);
  bad.block_size = 64u;
  bad.dq_block_size = 64u; /* must be 0 for SQ */
  CHECK(et_qlora_validate_quant_spec(&bad) == ET_QLORA_ERR_SHAPE);
  bad.mode = ET_QLORA_QUANT_NF4_DQ;
  bad.dq_block_size = 128u; /* not in {16,32,64,256} */
  CHECK(et_qlora_validate_quant_spec(&bad) == ET_QLORA_ERR_SHAPE);
  bad.mode = 99u;
  CHECK(et_qlora_validate_quant_spec(&bad) == ET_QLORA_ERR_FORMAT);
  bad.mode = ET_QLORA_QUANT_NF4_SQ;
  bad.dq_block_size = 0u;
  bad.cols_in = 48u; /* not multiple of 64 */
  CHECK(et_qlora_validate_quant_spec(&bad) == ET_QLORA_ERR_SHAPE);

  /* Overflow atomicity: huge X triggers dry-run overflow -> Y untouched. */
  et_qlora_linear_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.quant.rows_out = 2u;
  cfg.quant.cols_in = 32u;
  cfg.quant.block_size = 32u;
  cfg.quant.mode = ET_QLORA_QUANT_NF4_SQ;
  cfg.m = 1u;
  cfg.r = 1u;
  cfg.alpha = 2.0f;

  float w[64], x[32], a[32], b[2], y[2];
  for (int i = 0; i < 64; ++i)
    w[i] = 1e20f;
  for (int i = 0; i < 32; ++i) {
    x[i] = 1e20f;
    a[i] = 0.0f;
  }
  b[0] = b[1] = 0.0f;
  y[0] = 42.25f;
  y[1] = -17.5f;
  uint8_t packed[40];
  CHECK(et_qlora_nf4_quantize(&cfg.quant, w, packed, sizeof(packed)) ==
        ET_QLORA_OK);
  CHECK(et_qlora_linear_forward(&cfg, x, packed, sizeof(packed), NULL, a, b,
                                y) == ET_QLORA_ERR_NONFINITE);
  CHECK(y[0] == 42.25f && y[1] == -17.5f);

  /* Negative scale in packed NF4 rejected. */
  uint32_t neg_one = 0xbf800000u;
  memcpy(packed + 32u, &neg_one, 4u);
  for (int i = 0; i < 32; ++i)
    x[i] = 1.0f;
  CHECK(et_qlora_linear_forward(&cfg, x, packed, sizeof(packed), NULL, NULL,
                                NULL, y) == ET_QLORA_ERR_NONFINITE);
  CHECK(y[0] == 42.25f && y[1] == -17.5f);

  /* Aliasing rejected. */
  CHECK(et_qlora_nf4_quantize(&cfg.quant, w, packed, sizeof(packed)) ==
        ET_QLORA_OK);
  CHECK(et_qlora_f32_linear_forward(1u, 2u, 32u, x, w, NULL, x) ==
        ET_QLORA_ERR_ALIAS);

  /* Non-default FP rounding mode rejected cleanly and restored. */
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_qlora_validate_quant_spec(&cfg.quant) == ET_QLORA_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);
  CHECK(et_qlora_validate_quant_spec(&cfg.quant) == ET_QLORA_OK);

  /* Negative AdamW v_state rejected without mutating params. */
  et_qlora_adamw_config opt = {0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 0u, 0u};
  float p[2] = {1.5f, -2.5f}, g[2] = {0.1f, -0.2f}, m[2] = {0.0f, 0.0f},
        v[2] = {0.0f, -1e-6f};
  CHECK(et_qlora_adamw_step(2u, &opt, p, g, m, v) == ET_QLORA_ERR_NONFINITE);
  CHECK(p[0] == 1.5f && p[1] == -2.5f);

  printf("QLORA NATIVE NEGATIVES & ATOMICITY: PASS\n");
}

/* ---- 6. Oracle Emitter (--oracle) for Independent Python Verification ---- */
static void run_oracle_emitter(void) {
  const struct {
    uint32_t m, rows_out, cols_in, r, bs, dqs;
    et_qlora_quant_mode mode;
    float alpha;
  } cases[4] = {
      {3u, 8u, 32u, 2u, 32u, 0u, ET_QLORA_QUANT_NF4_SQ, 4.0f},
      {4u, 12u, 64u, 4u, 64u, 0u, ET_QLORA_QUANT_NF4_SQ, 8.0f},
      {3u, 12u, 32u, 3u, 32u, 16u, ET_QLORA_QUANT_NF4_DQ, 6.0f},
      {4u, 16u, 64u, 4u, 64u, 16u, ET_QLORA_QUANT_NF4_DQ, 8.0f},
  };

  for (int c = 0; c < 4; ++c) {
    et_qlora_linear_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.quant.rows_out = cases[c].rows_out;
    cfg.quant.cols_in = cases[c].cols_in;
    cfg.quant.block_size = cases[c].bs;
    cfg.quant.dq_block_size = cases[c].dqs;
    cfg.quant.mode = cases[c].mode;
    cfg.m = cases[c].m;
    cfg.r = cases[c].r;
    cfg.alpha = cases[c].alpha;

    et_qlora_quant_footprint fp;
    CHECK(et_qlora_packed_footprint(&cfg.quant, &fp) == ET_QLORA_OK);

    const size_t x_n = (size_t)cfg.m * cfg.quant.cols_in;
    const size_t w_n = (size_t)cfg.quant.rows_out * cfg.quant.cols_in;
    const size_t y_n = (size_t)cfg.m * cfg.quant.rows_out;
    const size_t a_n = (size_t)cfg.r * cfg.quant.cols_in;
    const size_t b_n = (size_t)cfg.quant.rows_out * cfg.r;

    float *w_f32 = (float *)malloc(w_n * sizeof(float));
    float *w_deq = (float *)malloc(w_n * sizeof(float));
    float *w_merged = (float *)malloc(w_n * sizeof(float));
    float *x = (float *)malloc(x_n * sizeof(float));
    float *bias = (float *)malloc(cfg.quant.rows_out * sizeof(float));
    float *a = (float *)malloc(a_n * sizeof(float));
    float *b = (float *)malloc(b_n * sizeof(float));
    float *dy = (float *)malloc(y_n * sizeof(float));
    float *y = (float *)malloc(y_n * sizeof(float));
    float *dx = (float *)malloc(x_n * sizeof(float));
    float *da = (float *)malloc(a_n * sizeof(float));
    float *db = (float *)malloc(b_n * sizeof(float));
    float *dbias = (float *)malloc(cfg.quant.rows_out * sizeof(float));
    uint8_t *packed = (uint8_t *)malloc((size_t)fp.packed_bytes);
    CHECK(w_f32 && w_deq && w_merged && x && bias && a && b && dy && y && dx &&
          da && db && dbias && packed);

    uint64_t rng = UINT64_C(0x1020304050607080) + (uint64_t)c * 7919u;
    fill_gaussian(w_f32, w_n, 0.35, &rng);
    fill_gaussian(x, x_n, 0.4, &rng);
    fill_gaussian(bias, cfg.quant.rows_out, 0.15, &rng);
    fill_gaussian(a, a_n, 0.2, &rng);
    fill_gaussian(b, b_n, 0.2, &rng);
    fill_gaussian(dy, y_n, 0.25, &rng);

    CHECK(et_qlora_nf4_quantize(&cfg.quant, w_f32, packed,
                                (size_t)fp.packed_bytes) == ET_QLORA_OK);
    CHECK(et_qlora_nf4_dequantize(&cfg.quant, packed, (size_t)fp.packed_bytes,
                                  w_deq) == ET_QLORA_OK);
    CHECK(et_qlora_linear_forward(&cfg, x, packed, (size_t)fp.packed_bytes,
                                  bias, a, b, y) == ET_QLORA_OK);
    CHECK(et_qlora_linear_backward(&cfg, x, packed, (size_t)fp.packed_bytes, a,
                                   b, dy, dx, da, db, dbias) == ET_QLORA_OK);
    CHECK(et_qlora_merge_to_f32(&cfg, packed, (size_t)fp.packed_bytes, a, b,
                                w_merged) == ET_QLORA_OK);

    printf("LINEAR %u %u %u %u %u %u %u %a\n", cfg.m, cfg.quant.rows_out,
           cfg.quant.cols_in, cfg.r, cfg.quant.block_size,
           cfg.quant.dq_block_size, cfg.quant.mode, (double)cfg.alpha);
    print_f32_hex_vec(w_f32, w_n);
    print_f32_hex_vec(x, x_n);
    print_f32_hex_vec(bias, cfg.quant.rows_out);
    print_f32_hex_vec(a, a_n);
    print_f32_hex_vec(b, b_n);
    print_f32_hex_vec(dy, y_n);
    print_bytes_hex(packed, (size_t)fp.packed_bytes);
    print_f32_hex_vec(w_deq, w_n);
    print_f32_hex_vec(y, y_n);
    print_f32_hex_vec(dx, x_n);
    print_f32_hex_vec(da, a_n);
    print_f32_hex_vec(db, b_n);
    print_f32_hex_vec(dbias, cfg.quant.rows_out);
    print_f32_hex_vec(w_merged, w_n);

    free(w_f32);
    free(w_deq);
    free(w_merged);
    free(x);
    free(bias);
    free(a);
    free(b);
    free(dy);
    free(y);
    free(dx);
    free(da);
    free(db);
    free(dbias);
    free(packed);
  }

  /* Emit 12-step 2-layer QLoRA MLP AdamW trajectory on DQ frozen base. */
  et_qlora_mlp_config mcfg;
  memset(&mcfg, 0, sizeof(mcfg));
  mcfg.m = 3u;
  mcfg.d_in = 32u;
  mcfg.d_hidden = 32u;
  mcfg.d_out = 16u;
  mcfg.r = 4u;
  mcfg.block_size = 32u;
  mcfg.dq_block_size = 16u;
  mcfg.quant_mode = ET_QLORA_QUANT_NF4_DQ;
  mcfg.alpha = 8.0f;

  et_qlora_quant_spec s1 = {mcfg.d_hidden, mcfg.d_in, mcfg.block_size,
                            mcfg.dq_block_size, mcfg.quant_mode, 0u};
  et_qlora_quant_spec s2 = {mcfg.d_out, mcfg.d_hidden, mcfg.block_size,
                            mcfg.dq_block_size, mcfg.quant_mode, 0u};
  et_qlora_quant_footprint fp1, fp2;
  CHECK(et_qlora_packed_footprint(&s1, &fp1) == ET_QLORA_OK);
  CHECK(et_qlora_packed_footprint(&s2, &fp2) == ET_QLORA_OK);
  size_t scratch_n = 0u;
  CHECK(et_qlora_mlp_scratch_floats(&mcfg, &scratch_n) == ET_QLORA_OK);

  const size_t w1_n = (size_t)mcfg.d_hidden * mcfg.d_in;
  const size_t w2_n = (size_t)mcfg.d_out * mcfg.d_hidden;
  const size_t x_n = (size_t)mcfg.m * mcfg.d_in;
  const size_t h_n = (size_t)mcfg.m * mcfg.d_hidden;
  const size_t y_n = (size_t)mcfg.m * mcfg.d_out;
  const size_t a1_n = (size_t)mcfg.r * mcfg.d_in;
  const size_t b1_n = (size_t)mcfg.d_hidden * mcfg.r;
  const size_t a2_n = (size_t)mcfg.r * mcfg.d_hidden;
  const size_t b2_n = (size_t)mcfg.d_out * mcfg.r;

  float *w1_f = (float *)malloc(w1_n * sizeof(float));
  float *w2_f = (float *)malloc(w2_n * sizeof(float));
  float *b1_f = (float *)malloc(mcfg.d_hidden * sizeof(float));
  float *b2_f = (float *)malloc(mcfg.d_out * sizeof(float));
  float *x = (float *)malloc(x_n * sizeof(float));
  float *target = (float *)malloc(y_n * sizeof(float));
  uint8_t *w1_q = (uint8_t *)malloc((size_t)fp1.packed_bytes);
  uint8_t *w2_q = (uint8_t *)malloc((size_t)fp2.packed_bytes);
  float *a1 = (float *)calloc(a1_n, sizeof(float));
  float *b1 = (float *)calloc(b1_n, sizeof(float));
  float *a2 = (float *)calloc(a2_n, sizeof(float));
  float *b2 = (float *)calloc(b2_n, sizeof(float));
  float *ma1 = (float *)calloc(a1_n, sizeof(float));
  float *va1 = (float *)calloc(a1_n, sizeof(float));
  float *mb1 = (float *)calloc(b1_n, sizeof(float));
  float *vb1 = (float *)calloc(b1_n, sizeof(float));
  float *ma2 = (float *)calloc(a2_n, sizeof(float));
  float *va2 = (float *)calloc(a2_n, sizeof(float));
  float *mb2 = (float *)calloc(b2_n, sizeof(float));
  float *vb2 = (float *)calloc(b2_n, sizeof(float));
  float *scratch = (float *)calloc(scratch_n, sizeof(float));
  float *z1 = (float *)calloc(h_n, sizeof(float));
  float *h1 = (float *)calloc(h_n, sizeof(float));
  float *y = (float *)calloc(y_n, sizeof(float));
  CHECK(w1_f && w2_f && b1_f && b2_f && x && target && w1_q && w2_q && a1 &&
        b1 && a2 && b2 && ma1 && va1 && mb1 && vb1 && ma2 && va2 && mb2 &&
        vb2 && scratch && z1 && h1 && y);

  uint64_t rng = UINT64_C(0x9988776655443322);
  fill_gaussian(w1_f, w1_n, 0.25, &rng);
  fill_gaussian(w2_f, w2_n, 0.25, &rng);
  fill_gaussian(b1_f, mcfg.d_hidden, 0.05, &rng);
  fill_gaussian(b2_f, mcfg.d_out, 0.05, &rng);
  fill_gaussian(x, x_n, 0.4, &rng);
  fill_gaussian(target, y_n, 0.4, &rng);
  CHECK(et_qlora_nf4_quantize(&s1, w1_f, w1_q, (size_t)fp1.packed_bytes) ==
        ET_QLORA_OK);
  CHECK(et_qlora_nf4_quantize(&s2, w2_f, w2_q, (size_t)fp2.packed_bytes) ==
        ET_QLORA_OK);
  CHECK(et_qlora_init_lora(mcfg.d_in, mcfg.d_hidden, mcfg.r, 555u, a1, b1) ==
        ET_QLORA_OK);
  CHECK(et_qlora_init_lora(mcfg.d_hidden, mcfg.d_out, mcfg.r, 666u, a2, b2) ==
        ET_QLORA_OK);

  et_qlora_adamw_config opt = {0.02f, 0.9f, 0.999f, 1e-8f, 0.01f, 0u, 0u};
  printf("MLP %u %u %u %u %u %u %u %u %a %a %a %a %a %a 12\n", mcfg.m,
         mcfg.d_in, mcfg.d_hidden, mcfg.d_out, mcfg.r, mcfg.block_size,
         mcfg.dq_block_size, mcfg.quant_mode, (double)mcfg.alpha,
         (double)opt.lr, (double)opt.beta1, (double)opt.beta2, (double)opt.eps,
         (double)opt.weight_decay);
  print_f32_hex_vec(w1_f, w1_n);
  print_f32_hex_vec(w2_f, w2_n);
  print_f32_hex_vec(b1_f, mcfg.d_hidden);
  print_f32_hex_vec(b2_f, mcfg.d_out);
  print_f32_hex_vec(x, x_n);
  print_f32_hex_vec(target, y_n);
  print_f32_hex_vec(a1, a1_n);
  print_f32_hex_vec(b1, b1_n);
  print_f32_hex_vec(a2, a2_n);
  print_f32_hex_vec(b2, b2_n);

  double losses[12];
  for (int s = 0; s < 12; ++s) {
    CHECK(et_qlora_mlp_train_step(
              &mcfg, &opt, x, target, w1_q, (size_t)fp1.packed_bytes, b1_f,
              w2_q, (size_t)fp2.packed_bytes, b2_f, a1, b1, a2, b2, ma1, va1,
              mb1, vb1, ma2, va2, mb2, vb2, scratch, scratch_n,
              &losses[s]) == ET_QLORA_OK);
  }
  CHECK(et_qlora_mlp_forward(&mcfg, x, w1_q, (size_t)fp1.packed_bytes, b1_f, a1,
                             b1, w2_q, (size_t)fp2.packed_bytes, b2_f, a2, b2,
                             z1, h1, y) == ET_QLORA_OK);
  for (int s = 0; s < 12; ++s) {
    if (s != 0)
      putchar(' ');
    printf("%a", losses[s]);
  }
  putchar('\n');
  print_f32_hex_vec(a1, a1_n);
  print_f32_hex_vec(b1, b1_n);
  print_f32_hex_vec(a2, a2_n);
  print_f32_hex_vec(b2, b2_n);
  print_f32_hex_vec(y, y_n);

  free(w1_f);
  free(w2_f);
  free(b1_f);
  free(b2_f);
  free(x);
  free(target);
  free(w1_q);
  free(w2_q);
  free(a1);
  free(b1);
  free(a2);
  free(b2);
  free(ma1);
  free(va1);
  free(mb1);
  free(vb1);
  free(ma2);
  free(va2);
  free(mb2);
  free(vb2);
  free(scratch);
  free(z1);
  free(h1);
  free(y);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_emitter();
    return 0;
  }
  test_nf4_codebook_and_packing();
  test_sq_dq_footprint_and_gaussian_rmse();
  test_fused_linear_parity_and_fd();
  test_mlp_train_and_checkpoint_resume();
  test_negatives_and_atomicity();
  printf("QLORA NATIVE ALL TESTS: PASS\n");
  return 0;
}
