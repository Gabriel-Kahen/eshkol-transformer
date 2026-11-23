#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/lora_adapter_abi.h"
#include "lora_eshkol_bridge.h"

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

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static float det_val(size_t index, int salt, float scale) {
  const int v = (int)((index * 31u + (size_t)salt * 17u + index / 5u) % 37u) - 18;
  return (float)v * scale;
}

static void fill_det(float *buf, size_t count, int salt, float scale) {
  for (size_t i = 0; i < count; ++i)
    buf[i] = det_val(i, salt, scale);
}

static void fill_sentinel(float *buf, size_t count, float sentinel) {
  for (size_t i = 0; i < count; ++i)
    buf[i] = sentinel;
}

static int check_sentinel(const float *buf, size_t count, float sentinel) {
  uint32_t expected_bits = 0u;
  memcpy(&expected_bits, &sentinel, sizeof(expected_bits));
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0u;
    memcpy(&bits, &buf[i], sizeof(bits));
    if (bits != expected_bits)
      return 0;
  }
  return 1;
}

static void print_hex_vec(const float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i)
    printf("%s%a", i == 0u ? "" : " ", (double)buf[i]);
  putchar('\n');
}

/* ---- Oracle vector emitter (--oracle) ---- */

static int run_oracle(void) {
  static const et_lora_config cases[] = {
      {1u, 1u, 1u, 1u, 1.0f},
      {1u, 4u, 3u, 1u, 2.0f},
      {2u, 5u, 4u, 2u, 4.0f},
      {4u, 8u, 6u, 3u, 6.0f},
      {8u, 16u, 12u, 4u, 8.0f},
      {6u, 12u, 16u, 8u, 12.0f},
      {16u, 32u, 24u, 16u, 32.0f},
      {64u, 64u, 64u, 16u, 16.0f},
  };
  float x[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float w[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float dy[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float y[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dx[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float da[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float db[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float w_merged[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_restored[ET_LORA_MAX_N * ET_LORA_MAX_K];

  for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
    const et_lora_config cfg = cases[c];
    const size_t x_n = (size_t)cfg.m * cfg.k;
    const size_t w_n = (size_t)cfg.n * cfg.k;
    const size_t a_n = (size_t)cfg.r * cfg.k;
    const size_t b_n = (size_t)cfg.n * cfg.r;
    const size_t y_n = (size_t)cfg.m * cfg.n;

    const float s_base = (c % 2u == 0u) ? 0.03125f : 0.019f;
    const float s_w = (c % 2u == 0u) ? 0.015625f : 0.017f;
    fill_det(x, x_n, (int)(c * 7u + 1u), s_base);
    fill_det(w, w_n, (int)(c * 7u + 2u), s_w);
    fill_det(a, a_n, (int)(c * 7u + 3u), s_base);
    fill_det(b, b_n, (int)(c * 7u + 4u), s_base);
    fill_det(dy, y_n, (int)(c * 7u + 5u), s_base);

    REQUIRE(et_lora_forward(&cfg, x, w, a, b, y) == ET_LORA_OK);
    REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK);
    REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK);
    REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK);

    printf("CASE %u %u %u %u %a\n", cfg.m, cfg.k, cfg.n, cfg.r,
           (double)cfg.alpha);
    print_hex_vec(x, x_n);
    print_hex_vec(w, w_n);
    print_hex_vec(a, a_n);
    print_hex_vec(b, b_n);
    print_hex_vec(dy, y_n);
    print_hex_vec(y, y_n);
    print_hex_vec(dx, x_n);
    print_hex_vec(da, a_n);
    print_hex_vec(db, b_n);
    print_hex_vec(w_merged, w_n);
    print_hex_vec(w_restored, w_n);
  }
  return 0;
}

/* ---- 1. Zero-init B bitwise parity & frozen-base immutability ---- */

static void test_zero_init_and_frozen_base(void) {
  static const et_lora_config shapes[] = {
      {1u, 1u, 1u, 1u, 1.0f},
      {4u, 8u, 6u, 2u, 4.0f},
      {8u, 16u, 12u, 4u, 8.0f},
      {16u, 32u, 24u, 8u, 16.0f},
      {64u, 64u, 64u, 16u, 32.0f},
  };
  float x[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float w[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_snapshot[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float y_base[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float y_lora[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dy[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dx[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float da[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float db[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float w_merged[ET_LORA_MAX_N * ET_LORA_MAX_K];
  uint8_t digest_before[ET_LORA_SHA256_BYTES];
  uint8_t digest_after[ET_LORA_SHA256_BYTES];

  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
    const et_lora_config cfg = shapes[s];
    const size_t x_n = (size_t)cfg.m * cfg.k;
    const size_t w_n = (size_t)cfg.n * cfg.k;
    const size_t a_n = (size_t)cfg.r * cfg.k;
    const size_t b_n = (size_t)cfg.n * cfg.r;
    const size_t y_n = (size_t)cfg.m * cfg.n;

    fill_det(x, x_n, (int)(s * 5u + 1u), 0.0625f);
    fill_det(w, w_n, (int)(s * 5u + 2u), 0.03125f);
    fill_det(dy, y_n, (int)(s * 5u + 3u), 0.0625f);
    memcpy(w_snapshot, w, w_n * sizeof(float));
    REQUIRE(et_lora_base_weight_sha256(cfg.k, cfg.n, w, digest_before) ==
            ET_LORA_OK);

    REQUIRE(et_lora_init_adapter(&cfg, UINT64_C(100) + s, a, b) == ET_LORA_OK);
    for (size_t i = 0; i < b_n; ++i)
      REQUIRE(b[i] == 0.0f && !signbit(b[i]));
    int any_a_nonzero = 0;
    for (size_t i = 0; i < a_n; ++i)
      if (a[i] != 0.0f)
        any_a_nonzero = 1;
    REQUIRE(any_a_nonzero);

    REQUIRE(et_lora_base_forward(&cfg, x, w, y_base) == ET_LORA_OK);
    REQUIRE(et_lora_forward(&cfg, x, w, a, b, y_lora) == ET_LORA_OK);
    /* Zero-init B must match base forward bitwise! */
    REQUIRE(memcmp(y_base, y_lora, y_n * sizeof(float)) == 0);

    /* Merge with B=0 must also match W_base bitwise! */
    REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK);
    REQUIRE(memcmp(w, w_merged, w_n * sizeof(float)) == 0);

    /* Run backward with B=0 and verify W is untouched. */
    REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK);
    for (size_t i = 0; i < a_n; ++i)
      REQUIRE(da[i] == 0.0f);
    REQUIRE(memcmp(w, w_snapshot, w_n * sizeof(float)) == 0);
    REQUIRE(et_lora_base_weight_sha256(cfg.k, cfg.n, w, digest_after) ==
            ET_LORA_OK);
    REQUIRE(memcmp(digest_before, digest_after, ET_LORA_SHA256_BYTES) == 0);
  }
  printf("LORA native zero-init bitwise parity & frozen base: PASS\n");
}

/* ---- 2. Central finite-difference gradient checks on dX, dA, dB ---- */

static double scalar_objective(const et_lora_config *cfg, const float *x,
                               const float *w, const float *a, const float *b,
                               const float *dy) {
  float y[ET_LORA_MAX_M * ET_LORA_MAX_N];
  REQUIRE(et_lora_forward(cfg, x, w, a, b, y) == ET_LORA_OK);
  const size_t y_n = (size_t)cfg->m * cfg->n;
  double sum = 0.0;
  for (size_t i = 0; i < y_n; ++i)
    sum += (double)y[i] * (double)dy[i];
  return sum;
}

static void test_finite_differences(void) {
  static const et_lora_config shapes[] = {
      {1u, 1u, 1u, 1u, 1.5f},
      {2u, 4u, 3u, 2u, 4.0f},
      {4u, 7u, 5u, 3u, 6.0f},
      {5u, 10u, 8u, 4u, 8.0f},
  };
  float x[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float w[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_orig[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float dy[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dx[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float da[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float db[ET_LORA_MAX_N * ET_LORA_MAX_R];

  const float eps = 1.0f / 512.0f;
  size_t total_checks = 0u;
  double max_abs_err = 0.0;

  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
    const et_lora_config cfg = shapes[s];
    const size_t x_n = (size_t)cfg.m * cfg.k;
    const size_t w_n = (size_t)cfg.n * cfg.k;
    const size_t a_n = (size_t)cfg.r * cfg.k;
    const size_t b_n = (size_t)cfg.n * cfg.r;
    const size_t y_n = (size_t)cfg.m * cfg.n;

    const float x_scale = (s == 3u) ? 0.019f : 0.0625f;
    const float w_scale = (s == 3u) ? 0.017f : 0.03125f;
    const float a_scale = (s == 3u) ? 0.023f : 0.0625f;
    const float b_scale = (s == 3u) ? 0.021f : 0.0625f;
    const float dy_scale = (s == 3u) ? 0.013f : 0.0625f;
    fill_det(x, x_n, (int)(s * 11u + 1u), x_scale);
    fill_det(w, w_n, (int)(s * 11u + 2u), w_scale);
    fill_det(a, a_n, (int)(s * 11u + 3u), a_scale);
    fill_det(b, b_n, (int)(s * 11u + 4u), b_scale);
    fill_det(dy, y_n, (int)(s * 11u + 5u), dy_scale);
    memcpy(w_orig, w, w_n * sizeof(float));

    REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK);
    REQUIRE(memcmp(w, w_orig, w_n * sizeof(float)) == 0);

    for (size_t i = 0; i < x_n; ++i) {
      const float orig = x[i];
      x[i] = orig + eps;
      const double plus = scalar_objective(&cfg, x, w, a, b, dy);
      x[i] = orig - eps;
      const double minus = scalar_objective(&cfg, x, w, a, b, dy);
      x[i] = orig;
      const double num = (plus - minus) / (2.0 * (double)eps);
      const double err = fabs((double)dx[i] - num);
      if (err > max_abs_err)
        max_abs_err = err;
      REQUIRE(err <= 1e-4 + 1e-3 * fabs(num));
      ++total_checks;
    }
    for (size_t i = 0; i < a_n; ++i) {
      const float orig = a[i];
      a[i] = orig + eps;
      const double plus = scalar_objective(&cfg, x, w, a, b, dy);
      a[i] = orig - eps;
      const double minus = scalar_objective(&cfg, x, w, a, b, dy);
      a[i] = orig;
      const double num = (plus - minus) / (2.0 * (double)eps);
      const double err = fabs((double)da[i] - num);
      if (err > max_abs_err)
        max_abs_err = err;
      REQUIRE(err <= 1e-4 + 1e-3 * fabs(num));
      ++total_checks;
    }
    for (size_t i = 0; i < b_n; ++i) {
      const float orig = b[i];
      b[i] = orig + eps;
      const double plus = scalar_objective(&cfg, x, w, a, b, dy);
      b[i] = orig - eps;
      const double minus = scalar_objective(&cfg, x, w, a, b, dy);
      b[i] = orig;
      const double num = (plus - minus) / (2.0 * (double)eps);
      const double err = fabs((double)db[i] - num);
      if (err > max_abs_err)
        max_abs_err = err;
      REQUIRE(err <= 1e-4 + 1e-3 * fabs(num));
      ++total_checks;
    }
  }
  printf("LORA native finite-difference gradients: PASS (%zu checks, "
         "max_abs_err=%.3e)\n",
         total_checks, max_abs_err);
}

/* ---- 3. Merge/unmerge forward parity and lossless restoration ---- */

static void test_merge_unmerge(void) {
  static const et_lora_config shapes[] = {
      {1u, 4u, 4u, 2u, 4.0f},
      {8u, 16u, 12u, 4u, 8.0f},
      {16u, 32u, 24u, 8u, 16.0f},
      {64u, 64u, 64u, 16u, 16.0f},
  };
  float x[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float w[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float w_merged[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_restored[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float y_adapter[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float y_merged[ET_LORA_MAX_M * ET_LORA_MAX_N];

  double max_fwd_diff = 0.0;
  double max_unmerge_diff = 0.0;

  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
    const et_lora_config cfg = shapes[s];
    const size_t x_n = (size_t)cfg.m * cfg.k;
    const size_t w_n = (size_t)cfg.n * cfg.k;
    const size_t a_n = (size_t)cfg.r * cfg.k;
    const size_t b_n = (size_t)cfg.n * cfg.r;
    const size_t y_n = (size_t)cfg.m * cfg.n;

    /* General non-dyadic values with Kaiming-style 1/sqrt(K) scale so
     * outputs are O(1) and binary32 merge/unmerge parity stays <= 1e-6. */
    const float inv_root_k = 1.0f / sqrtf((float)cfg.k);
    fill_det(x, x_n, (int)(s * 13u + 1u), 0.038f * inv_root_k);
    fill_det(w, w_n, (int)(s * 13u + 2u), 0.034f * inv_root_k);
    fill_det(a, a_n, (int)(s * 13u + 3u), 0.046f * inv_root_k);
    fill_det(b, b_n, (int)(s * 13u + 4u), 0.042f * inv_root_k);

    REQUIRE(et_lora_forward(&cfg, x, w, a, b, y_adapter) == ET_LORA_OK);
    REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK);
    REQUIRE(et_lora_base_forward(&cfg, x, w_merged, y_merged) == ET_LORA_OK);
    REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK);

    for (size_t i = 0; i < y_n; ++i) {
      const double d = fabs((double)y_adapter[i] - (double)y_merged[i]);
      if (d > max_fwd_diff)
        max_fwd_diff = d;
      REQUIRE(d <= 1e-6);
    }
    for (size_t i = 0; i < w_n; ++i) {
      const double d = fabs((double)w[i] - (double)w_restored[i]);
      if (d > max_unmerge_diff)
        max_unmerge_diff = d;
      REQUIRE(d <= 1e-6);
    }

    /* Exact dyadic values -> bit-exact lossless merge/unmerge! */
    fill_det(w, w_n, (int)(s + 21u), 0.125f);
    fill_det(a, a_n, (int)(s + 22u), 0.125f);
    fill_det(b, b_n, (int)(s + 23u), 0.125f);
    REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK);
    REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK);
    REQUIRE(memcmp(w, w_restored, w_n * sizeof(float)) == 0);
  }
  printf("LORA native merge/unmerge parity: PASS (max_fwd_diff=%.3e, "
         "max_unmerge_diff=%.3e, dyadic_exact=PASS)\n",
         max_fwd_diff, max_unmerge_diff);
}

/* ---- 4. Checkpoint save/reload & complete tamper/negative suite ---- */

static void write_file_raw(const char *path, const uint8_t *data, size_t len) {
  const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  REQUIRE(fd >= 0);
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, data + off, len - off);
    REQUIRE(n > 0);
    off += (size_t)n;
  }
  REQUIRE(close(fd) == 0);
}

static size_t read_file_raw(const char *path, uint8_t *buf, size_t cap) {
  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  REQUIRE(fd >= 0);
  size_t used = 0u;
  while (used < cap) {
    const ssize_t n = read(fd, buf + used, cap - used);
    if (n == 0)
      break;
    REQUIRE(n > 0);
    used += (size_t)n;
  }
  REQUIRE(close(fd) == 0);
  return used;
}

/* Independent SHA-256 for crafting a valid-checksum non-finite tamper test. */
static uint32_t test_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}
static void test_sha256(const uint8_t *msg, size_t len, uint8_t out[32]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  const size_t padded = ((len + 9u + 63u) / 64u) * 64u;
  uint8_t *buf = (uint8_t *)calloc(1u, padded);
  REQUIRE(buf != NULL);
  memcpy(buf, msg, len);
  buf[len] = 0x80u;
  const uint64_t bits = (uint64_t)len * UINT64_C(8);
  for (size_t i = 0; i < 8u; ++i)
    buf[padded - 1u - i] = (uint8_t)(bits >> (8u * i));
  for (size_t off = 0; off < padded; off += 64u) {
    uint32_t w[64];
    for (size_t i = 0; i < 16u; ++i) {
      w[i] = ((uint32_t)buf[off + i * 4u] << 24u) |
             ((uint32_t)buf[off + i * 4u + 1u] << 16u) |
             ((uint32_t)buf[off + i * 4u + 2u] << 8u) |
             (uint32_t)buf[off + i * 4u + 3u];
    }
    for (size_t i = 16u; i < 64u; ++i) {
      const uint32_t s0 =
          test_rotr32(w[i - 15u], 7u) ^ test_rotr32(w[i - 15u], 18u) ^
          (w[i - 15u] >> 3u);
      const uint32_t s1 =
          test_rotr32(w[i - 2u], 17u) ^ test_rotr32(w[i - 2u], 19u) ^
          (w[i - 2u] >> 10u);
      w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (size_t i = 0; i < 64u; ++i) {
      const uint32_t s1 =
          test_rotr32(e, 6u) ^ test_rotr32(e, 11u) ^ test_rotr32(e, 25u);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 =
          test_rotr32(a, 2u) ^ test_rotr32(a, 13u) ^ test_rotr32(a, 22u);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  free(buf);
  for (size_t i = 0; i < 8u; ++i) {
    out[i * 4u] = (uint8_t)(h[i] >> 24u);
    out[i * 4u + 1u] = (uint8_t)(h[i] >> 16u);
    out[i * 4u + 2u] = (uint8_t)(h[i] >> 8u);
    out[i * 4u + 3u] = (uint8_t)h[i];
  }
}

static void test_checkpoint_and_tampers(void) {
  char dir_tpl[] = "/tmp/et-lora-test-XXXXXX";
  char *dir = mkdtemp(dir_tpl);
  REQUIRE(dir != NULL);

  char ckpt_path[256];
  char bad_path[256];
  char sym_path[256];
  snprintf(ckpt_path, sizeof(ckpt_path), "%s/adapter.etlr", dir);
  snprintf(bad_path, sizeof(bad_path), "%s/tampered.etlr", dir);
  snprintf(sym_path, sizeof(sym_path), "%s/link.etlr", dir);

  const et_lora_config cfg = {4u, 12u, 8u, 3u, 6.0f};
  float w[8u * 12u];
  float w_wrong[8u * 12u];
  float a[3u * 12u];
  float b[8u * 3u];
  float a_out[3u * 12u];
  float b_out[8u * 3u];

  fill_det(w, 8u * 12u, 31, 0.03125f);
  fill_det(a, 3u * 12u, 32, 0.0625f);
  fill_det(b, 8u * 3u, 33, 0.0625f);
  memcpy(w_wrong, w, sizeof(w));
  w_wrong[5] += 0.25f;

  /* Valid save -> verify non-executable regular file & exact byte size. */
  REQUIRE(et_lora_checkpoint_save(ckpt_path, &cfg, w, a, b) == ET_LORA_OK);
  struct stat st;
  REQUIRE(stat(ckpt_path, &st) == 0);
  REQUIRE(S_ISREG(st.st_mode));
  REQUIRE((st.st_mode & 0111) == 0);
  size_t expected_size = 0u;
  REQUIRE(et_lora_checkpoint_size(cfg.k, cfg.n, cfg.r, &expected_size) ==
          ET_LORA_OK);
  REQUIRE((size_t)st.st_size == expected_size);

  /* Valid load -> exact bitwise match of A, B, alpha, and digests. */
  et_lora_checkpoint_meta meta;
  memset(&meta, 0, sizeof(meta));
  fill_sentinel(a_out, 3u * 12u, -77.0f);
  fill_sentinel(b_out, 8u * 3u, -88.0f);
  REQUIRE(et_lora_checkpoint_load(ckpt_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_OK);
  REQUIRE(meta.version_major == 1u && meta.version_minor == 0u);
  REQUIRE(meta.k == cfg.k && meta.n == cfg.n && meta.r == cfg.r);
  REQUIRE(meta.alpha == cfg.alpha);
  REQUIRE(memcmp(a, a_out, sizeof(a)) == 0);
  REQUIRE(memcmp(b, b_out, sizeof(b)) == 0);
  uint8_t w_digest[ET_LORA_SHA256_BYTES];
  REQUIRE(et_lora_base_weight_sha256(cfg.k, cfg.n, w, w_digest) == ET_LORA_OK);
  REQUIRE(memcmp(meta.base_weight_sha256, w_digest, ET_LORA_SHA256_BYTES) == 0);

  uint8_t raw[ET_LORA_CHECKPOINT_MAX_BYTES + 16u];
  const size_t raw_len = read_file_raw(ckpt_path, raw, sizeof(raw));
  REQUIRE(raw_len == expected_size);

  /* 1. Base hash mismatch. */
  fill_sentinel(a_out, 3u * 12u, 13.0f);
  fill_sentinel(b_out, 8u * 3u, 17.0f);
  REQUIRE(et_lora_checkpoint_load(ckpt_path, cfg.k, cfg.n, cfg.r, w_wrong,
                                  &meta, a_out, b_out) ==
          ET_LORA_ERR_BASE_HASH);
  REQUIRE(check_sentinel(a_out, 3u * 12u, 13.0f));
  REQUIRE(check_sentinel(b_out, 8u * 3u, 17.0f));

  /* 2. Rank / shape mismatch. */
  REQUIRE(et_lora_checkpoint_load(ckpt_path, cfg.k, cfg.n, 2u, w, &meta, a_out,
                                  b_out) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_checkpoint_load(ckpt_path, 11u, cfg.n, cfg.r, w, &meta, a_out,
                                  b_out) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_checkpoint_load(ckpt_path, cfg.k, 7u, cfg.r, w, &meta, a_out,
                                  b_out) == ET_LORA_ERR_SHAPE);

  /* 3. Truncated files (10 bytes, 64 bytes, len - 1 bytes). */
  write_file_raw(bad_path, raw, 10u);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_TRUNCATED);
  write_file_raw(bad_path, raw, 64u);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_TRUNCATED);
  write_file_raw(bad_path, raw, raw_len - 1u);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_TRUNCATED);

  /* 4. Extra trailing byte -> FORMAT. */
  uint8_t extra[ET_LORA_CHECKPOINT_MAX_BYTES + 16u];
  memcpy(extra, raw, raw_len);
  extra[raw_len] = 0x5au;
  write_file_raw(bad_path, extra, raw_len + 1u);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_FORMAT);

  /* 5. Corrupt checksum (payload bit flip and trailer bit flip). */
  uint8_t mod[ET_LORA_CHECKPOINT_MAX_BYTES];
  memcpy(mod, raw, raw_len);
  mod[ET_LORA_CHECKPOINT_HEADER_BYTES] ^= 0x80u;
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_CHECKSUM);
  memcpy(mod, raw, raw_len);
  mod[raw_len - 1u] ^= 0x01u;
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_CHECKSUM);

  /* 6. Bad magic. */
  memcpy(mod, raw, raw_len);
  mod[0] = 'Z';
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_FORMAT);

  /* 7. Version mismatch (major=2 and minor=1). */
  memcpy(mod, raw, raw_len);
  mod[8] = 2u;
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_VERSION);
  memcpy(mod, raw, raw_len);
  mod[12] = 1u;
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_VERSION);

  /* 8. Valid checksum but NaN inside A payload -> NONFINITE. */
  memcpy(mod, raw, raw_len);
  mod[ET_LORA_CHECKPOINT_HEADER_BYTES + 0u] = 0x01u;
  mod[ET_LORA_CHECKPOINT_HEADER_BYTES + 1u] = 0x00u;
  mod[ET_LORA_CHECKPOINT_HEADER_BYTES + 2u] = 0xc0u;
  mod[ET_LORA_CHECKPOINT_HEADER_BYTES + 3u] = 0x7fu;
  test_sha256(mod, raw_len - ET_LORA_SHA256_BYTES,
              mod + (raw_len - ET_LORA_SHA256_BYTES));
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_NONFINITE);

  /* 9. Valid checksum but +Inf in alpha -> NONFINITE. */
  memcpy(mod, raw, raw_len);
  mod[28] = 0x00u;
  mod[29] = 0x00u;
  mod[30] = 0x80u;
  mod[31] = 0x7fu;
  test_sha256(mod, raw_len - ET_LORA_SHA256_BYTES,
              mod + (raw_len - ET_LORA_SHA256_BYTES));
  write_file_raw(bad_path, mod, raw_len);
  REQUIRE(et_lora_checkpoint_load(bad_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_NONFINITE);

  /* 10. Symlink rejection (O_NOFOLLOW -> SYMLINK). */
  REQUIRE(symlink(ckpt_path, sym_path) == 0);
  REQUIRE(et_lora_checkpoint_load(sym_path, cfg.k, cfg.n, cfg.r, w, &meta,
                                  a_out, b_out) == ET_LORA_ERR_SYMLINK);

  /* 11. Directory and non-existent file rejection -> IO. */
  REQUIRE(et_lora_checkpoint_load(dir, cfg.k, cfg.n, cfg.r, w, &meta, a_out,
                                  b_out) == ET_LORA_ERR_IO);
  REQUIRE(et_lora_checkpoint_save(dir, &cfg, w, a, b) == ET_LORA_ERR_IO);
  REQUIRE(et_lora_checkpoint_save("", &cfg, w, a, b) == ET_LORA_ERR_ARGUMENT);

  /* All failures must leave caller buffers untouched. */
  REQUIRE(check_sentinel(a_out, 3u * 12u, 13.0f));
  REQUIRE(check_sentinel(b_out, 8u * 3u, 17.0f));

  (void)unlink(sym_path);
  (void)unlink(bad_path);
  (void)unlink(ckpt_path);
  (void)rmdir(dir);
  printf("LORA native checkpoint save/reload & tamper suite: PASS\n");
}

/* ---- 5. Bounds, alignment (via void *), aliasing, non-finite, FP env ---- */

static void test_negative_boundaries(void) {
  const et_lora_config valid = {2u, 4u, 4u, 2u, 4.0f};
  float x[8] = {0.5f, -0.25f, 0.75f, 0.125f, -0.5f, 0.25f, 0.5f, -0.125f};
  float w[16];
  float a[8];
  float b[8];
  float dy[8];
  float y[8];
  float dx[8];
  float da[8];
  float db[8];
  fill_det(w, 16u, 41, 0.0625f);
  fill_det(a, 8u, 42, 0.0625f);
  fill_det(b, 8u, 43, 0.0625f);
  fill_det(dy, 8u, 44, 0.0625f);

  /* Invalid configs. */
  REQUIRE(et_lora_validate_config(NULL) == ET_LORA_ERR_ARGUMENT);
  const et_lora_config bad_m0 = {0u, 4u, 4u, 2u, 1.0f};
  const et_lora_config bad_m65 = {65u, 4u, 4u, 2u, 1.0f};
  const et_lora_config bad_k0 = {2u, 0u, 4u, 1u, 1.0f};
  const et_lora_config bad_k65 = {2u, 65u, 4u, 1u, 1.0f};
  const et_lora_config bad_n0 = {2u, 4u, 0u, 1u, 1.0f};
  const et_lora_config bad_n65 = {2u, 4u, 65u, 1u, 1.0f};
  const et_lora_config bad_r0 = {2u, 4u, 4u, 0u, 1.0f};
  const et_lora_config bad_r17 = {2u, 32u, 32u, 17u, 1.0f};
  const et_lora_config bad_r_gt_min = {2u, 3u, 8u, 4u, 1.0f};
  const et_lora_config bad_alpha0 = {2u, 4u, 4u, 2u, 0.0f};
  const et_lora_config bad_alphaneg = {2u, 4u, 4u, 2u, -1.0f};
  const et_lora_config bad_alphanan = {2u, 4u, 4u, 2u, NAN};
  const et_lora_config bad_alphainf = {2u, 4u, 4u, 2u, INFINITY};

  REQUIRE(et_lora_validate_config(&bad_m0) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_m65) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_k0) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_k65) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_n0) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_n65) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_r0) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_r17) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_r_gt_min) == ET_LORA_ERR_SHAPE);
  REQUIRE(et_lora_validate_config(&bad_alpha0) == ET_LORA_ERR_ARGUMENT);
  REQUIRE(et_lora_validate_config(&bad_alphaneg) == ET_LORA_ERR_ARGUMENT);
  REQUIRE(et_lora_validate_config(&bad_alphanan) == ET_LORA_ERR_NONFINITE);
  REQUIRE(et_lora_validate_config(&bad_alphainf) == ET_LORA_ERR_NONFINITE);

  /* Misaligned pointer boundary checks using void * (zero misaligned casts!). */
  _Alignas(16) unsigned char raw_buf[128];
  memset(raw_buf, 0, sizeof(raw_buf));
  void *misaligned = raw_buf + 1u;
  fill_sentinel(y, 8u, 5.0f);
  REQUIRE(et_lora_forward_raw(&valid, misaligned, w, a, b, y) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_forward_raw(&valid, x, misaligned, a, b, y) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_forward_raw(&valid, x, w, misaligned, b, y) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_forward_raw(&valid, x, w, a, misaligned, y) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_forward_raw(&valid, x, w, a, b, misaligned) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_backward_raw(&valid, x, w, a, b, dy, misaligned, da, db) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_merge_raw(&valid, w, a, b, misaligned) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_unmerge_raw(&valid, w, a, b, misaligned) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(et_lora_init_adapter_raw(&valid, 1u, misaligned, b) ==
          ET_LORA_ERR_ALIGNMENT);
  REQUIRE(check_sentinel(y, 8u, 5.0f));

  /* Aliasing checks (output aliases input / input aliases input / output-output). */
  REQUIRE(et_lora_forward(&valid, x, w, a, b, x) == ET_LORA_ERR_ALIAS);
  REQUIRE(et_lora_forward(&valid, x, w, a, a, y) == ET_LORA_ERR_ALIAS);
  REQUIRE(et_lora_backward(&valid, x, w, a, b, dy, dx, dx, db) ==
          ET_LORA_ERR_ALIAS);
  REQUIRE(et_lora_merge(&valid, w, a, b, w) == ET_LORA_ERR_ALIAS);
  REQUIRE(et_lora_init_adapter(&valid, 1u, a, a) == ET_LORA_ERR_ALIAS);

  /* Non-finite operand & late intermediate overflow with failure atomicity. */
  float x_nan[8];
  memcpy(x_nan, x, sizeof(x));
  x_nan[3] = NAN;
  fill_sentinel(y, 8u, 9.0f);
  REQUIRE(et_lora_forward(&valid, x_nan, w, a, b, y) == ET_LORA_ERR_NONFINITE);
  REQUIRE(check_sentinel(y, 8u, 9.0f));

  float x_huge[8];
  float w_huge[16];
  fill_sentinel(x_huge, 8u, FLT_MAX * 0.5f);
  fill_sentinel(w_huge, 16u, 4.0f);
  fill_sentinel(y, 8u, 11.0f);
  REQUIRE(et_lora_forward(&valid, x_huge, w_huge, a, b, y) ==
          ET_LORA_ERR_NONFINITE);
  REQUIRE(check_sentinel(y, 8u, 11.0f));

  fill_sentinel(dx, 8u, 3.0f);
  fill_sentinel(da, 8u, 4.0f);
  fill_sentinel(db, 8u, 6.0f);
  REQUIRE(et_lora_backward(&valid, x_huge, w_huge, a, b, dy, dx, da, db) ==
          ET_LORA_ERR_NONFINITE);
  REQUIRE(check_sentinel(dx, 8u, 3.0f));
  REQUIRE(check_sentinel(da, 8u, 4.0f));
  REQUIRE(check_sentinel(db, 8u, 6.0f));

  /* FP environment checks: rounding mode, x86-64 MXCSR FTZ/DAZ, x87 control. */
  const int saved_round = fegetround();
  REQUIRE(fesetround(FE_TOWARDZERO) == 0);
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_ERR_FP_ENV);
  REQUIRE(fesetround(saved_round) == 0);
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_OK);

#if defined(__x86_64__)
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* FTZ */
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_ERR_FP_ENV);
  _mm_setcsr(saved_mxcsr | 0x0040u); /* DAZ */
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_ERR_FP_ENV);
  _mm_setcsr(saved_mxcsr);
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_OK);

  uint16_t saved_x87 = 0u;
  __asm__ volatile("fnstcw %0" : "=m"(saved_x87));
  uint16_t mod_x87 = (uint16_t)(saved_x87 | 0x0c00u); /* x87 round toward zero */
  __asm__ volatile("fldcw %0" : : "m"(mod_x87));
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_ERR_FP_ENV);
  __asm__ volatile("fldcw %0" : : "m"(saved_x87));
  REQUIRE(et_lora_forward(&valid, x, w, a, b, y) == ET_LORA_OK);
#endif

  printf("LORA native bounds/alignment/aliasing/nonfinite/FP-env: PASS\n");
}

/* ---- 6. AOT bridge smoke test under normal & ASan/UBSan/LSan ---- */

static void test_aot_bridge(void) {
  et_lora_aot_ctx *ctx = NULL;
  REQUIRE(et_lora_aot_create(0, 8, 8, 2, 4.0, &ctx) == ET_LORA_ERR_SHAPE);
  REQUIRE(ctx == NULL);
  REQUIRE(et_lora_aot_create(4, 8, 6, 2, 4.0, &ctx) == ET_LORA_OK);
  REQUIRE(ctx != NULL);
  REQUIRE(et_lora_aot_x_count(ctx) == 32);
  REQUIRE(et_lora_aot_w_count(ctx) == 48);
  REQUIRE(et_lora_aot_a_count(ctx) == 16);
  REQUIRE(et_lora_aot_b_count(ctx) == 12);
  REQUIRE(et_lora_aot_y_count(ctx) == 24);
  for (int64_t i = 0; i < 32; ++i)
    REQUIRE(et_lora_aot_set_x(ctx, i, det_val((size_t)i, 1, 0.0625f)) ==
            ET_LORA_OK);
  for (int64_t i = 0; i < 48; ++i)
    REQUIRE(et_lora_aot_set_w_base(ctx, i, det_val((size_t)i, 2, 0.03125f)) ==
            ET_LORA_OK);
  REQUIRE(et_lora_aot_init_adapter(ctx, 7) == ET_LORA_OK);
  REQUIRE(et_lora_aot_base_forward(ctx) == ET_LORA_OK);
  REQUIRE(et_lora_aot_forward(ctx) == ET_LORA_OK);
  for (int64_t i = 0; i < 24; ++i)
    REQUIRE(et_lora_aot_y_bits(ctx, i) == et_lora_aot_y_base_bits(ctx, i));
  REQUIRE(et_lora_aot_compute_base_sha256(ctx) == ET_LORA_OK);
  REQUIRE(et_lora_aot_compute_y_sha256(ctx) == ET_LORA_OK);
  et_lora_aot_destroy(ctx);
  printf("LORA AOT bridge native smoke: PASS\n");
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return run_oracle();
  test_zero_init_and_frozen_base();
  test_finite_differences();
  test_merge_unmerge();
  test_checkpoint_and_tampers();
  test_negative_boundaries();
  test_aot_bridge();
  return 0;
}
