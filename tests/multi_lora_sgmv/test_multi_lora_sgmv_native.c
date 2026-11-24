#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/multi_lora_sgmv_abi.h"
#include "multi_lora_sgmv_eshkol_bridge.h"

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

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Allocation trap: enforces ZERO heap allocations across ALL hot SGMV forward,
 * backward, AdamW, and adapter-registry pool operations when linked with
 * -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free. */
static int forbid_alloc = 0;
static size_t alloc_calls_while_forbidden = 0u;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  if (forbid_alloc) {
    ++alloc_calls_while_forbidden;
    CHECK(!forbid_alloc);
  }
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  if (forbid_alloc) {
    ++alloc_calls_while_forbidden;
    CHECK(!forbid_alloc);
  }
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  if (forbid_alloc) {
    ++alloc_calls_while_forbidden;
    CHECK(!forbid_alloc);
  }
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  if (forbid_alloc) {
    ++alloc_calls_while_forbidden;
    CHECK(!forbid_alloc);
  }
  __real_free(ptr);
}

static uint32_t f32_bits(float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

static void fill_det(float *dst, size_t count, unsigned salt, float scale) {
  for (size_t i = 0u; i < count; ++i) {
    const int bucket =
        (int)((31u * (unsigned)i + 17u * salt + (unsigned)(i / 3u)) % 37u) - 18;
    dst[i] = (float)bucket * scale;
  }
}

static void print_f32_hex(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

/* ---- Test 1: Heterogeneous ranks r=0(base),1,2,4,8,16 + unsorted SGMV ---- */

static void test_heterogeneous_ranks_and_parity(void) {
  const uint32_t in_dim = 20u;
  const uint32_t out_dim = 16u;
  const uint32_t max_adapters = 8u;
  const uint32_t max_rank = 16u;
  const uint32_t M = 18u;

  float w_base[16u * 20u];
  float bias[16u];
  float x[18u * 20u];
  fill_det(w_base, sizeof(w_base) / sizeof(w_base[0]), 11u, 0.03125f);
  fill_det(bias, sizeof(bias) / sizeof(bias[0]), 19u, 0.0625f);
  fill_det(x, sizeof(x) / sizeof(x[0]), 29u, 0.0625f);

  const uint32_t ranks[5] = {1u, 2u, 4u, 8u, 16u};
  const int32_t ids[5] = {10, 20, 30, 40, 50};
  const float alphas[5] = {2.0f, 4.0f, 8.0f, 16.0f, 32.0f};

  /* Unsorted batch mixing base-only (-1, rank 0) and ranks 1, 2, 4, 8, 16. */
  const int32_t row_ids[18] = {50, -1, 20, 10, 40, 30, -1, 50, 10,
                               30, 20, 40, 10, 50, -1, 40, 30, 20};

  for (uint32_t has_bias = 0u; has_bias <= 1u; ++has_bias) {
    const et_mlora_registry_config cfg = {in_dim,   out_dim,  max_adapters,
                                          max_rank, has_bias, 0u};
    et_mlora_registry *reg = NULL;
    CHECK(et_mlora_registry_create_v1(&cfg, w_base, has_bias ? bias : NULL,
                                      &reg) == ET_MLORA_OK);

    /* Zero-alloc hot section begins. */
    forbid_alloc = 1;

    /* Reject rank == 0 explicitly. */
    CHECK(et_mlora_register_seeded_adapter_v1(reg, 99, 0u, 2.0f, 1u, NULL) ==
          ET_MLORA_ERR_SHAPE);

    for (size_t k = 0u; k < 5u; ++k) {
      float a[16u * 20u];
      float b[16u * 16u];
      fill_det(a, (size_t)ranks[k] * in_dim, (unsigned)(100u + k * 7u), 0.05f);
      fill_det(b, (size_t)out_dim * ranks[k], (unsigned)(200u + k * 11u),
               0.05f);
      et_mlora_adapter_info info;
      CHECK(et_mlora_register_adapter_v1(reg, ids[k], ranks[k], alphas[k], a, b,
                                         &info) == ET_MLORA_OK);
      CHECK(info.adapter_id == ids[k]);
      CHECK(info.rank == ranks[k]);
      CHECK(info.generation == 1u);
      CHECK(info.pin_count == 0u);
    }

    float y_sgmv[18u * 16u];
    float y_ref[18u * 16u];
    et_mlora_sgmv_stats stats;
    CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y_sgmv, &stats) ==
          ET_MLORA_OK);
    CHECK(stats.batch_rows == M);
    CHECK(stats.base_only_rows == 3u);
    CHECK(stats.adapted_rows == 15u);
    CHECK(stats.active_segments == 5u);

    for (uint32_t i = 0u; i < M; ++i) {
      CHECK(et_mlora_single_adapter_forward_v1(
                reg, row_ids[i], 1u, &x[(size_t)i * in_dim],
                &y_ref[(size_t)i * out_dim]) == ET_MLORA_OK);
    }

    /* Bitwise 0.000e+00 parity between unsorted SGMV and single-row reference. */
    CHECK(memcmp(y_sgmv, y_ref, sizeof(y_sgmv)) == 0);

    forbid_alloc = 0;
    et_mlora_registry_destroy_v1(reg);
  }
}

/* ---- Test 2: Central FD checks (dX, dA, dB) & Zero Cross-Adapter Bleed ---- */

static double eval_weighted_loss(const et_mlora_registry *reg, uint32_t M,
                                 const float *x, const int32_t *row_ids,
                                 const float *dy, uint32_t out_dim) {
  float y[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y, NULL) == ET_MLORA_OK);
  double acc = 0.0;
  for (size_t i = 0u; i < (size_t)M * (size_t)out_dim; ++i)
    acc += (double)y[i] * (double)dy[i];
  return acc;
}

static void test_backward_fd_and_zero_bleed(double *out_max_fd_err) {
  const uint32_t in_dim = 16u;
  const uint32_t out_dim = 12u;
  const uint32_t max_adapters = 8u;
  const uint32_t max_rank = 12u;
  const uint32_t M = 14u;

  const et_mlora_registry_config cfg = {in_dim,   out_dim, max_adapters,
                                        max_rank, 1u,      0u};
  float w_base[12u * 16u];
  float bias[12u];
  float x[14u * 16u];
  float dy[14u * 12u];
  fill_det(w_base, sizeof(w_base) / sizeof(w_base[0]), 31u, 0.03125f);
  fill_det(bias, sizeof(bias) / sizeof(bias[0]), 41u, 0.0625f);
  fill_det(x, sizeof(x) / sizeof(x[0]), 53u, 0.0625f);
  fill_det(dy, sizeof(dy) / sizeof(dy[0]), 67u, 0.0625f);

  et_mlora_registry *reg = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w_base, bias, &reg) == ET_MLORA_OK);

  uint8_t base_before[ET_MLORA_SHA256_BYTES];
  uint8_t base_after[ET_MLORA_SHA256_BYTES];

  forbid_alloc = 1;
  CHECK(et_mlora_get_base_sha256_v1(reg, base_before) == ET_MLORA_OK);

  const uint32_t ranks[5] = {1u, 2u, 4u, 8u, 4u};
  const int32_t ids[5] = {101, 202, 303, 404, 505}; /* 505 is IDLE! */
  const float alphas[5] = {2.0f, 4.0f, 8.0f, 16.0f, 6.0f};
  for (size_t k = 0u; k < 5u; ++k) {
    float a[12u * 16u];
    float b[12u * 12u];
    fill_det(a, (size_t)ranks[k] * in_dim, (unsigned)(300u + k * 13u), 0.0625f);
    fill_det(b, (size_t)out_dim * ranks[k], (unsigned)(400u + k * 17u), 0.0625f);
    CHECK(et_mlora_register_adapter_v1(reg, ids[k], ranks[k], alphas[k], a, b,
                                       NULL) == ET_MLORA_OK);
  }

  /* First poison idle adapter 505's gradient buffers via a prior batch so we
   * prove the subsequent batch without 505 resets 505's grads to exact +0.0f. */
  const int32_t poison_ids[1] = {505};
  float dx_poison[16u];
  CHECK(et_mlora_sgmv_backward_v1(reg, 1u, x, poison_ids, dy, dx_poison, NULL) ==
        ET_MLORA_OK);

  /* Heterogeneous unsorted batch WITHOUT adapter 505. */
  const int32_t row_ids[14] = {404, -1,  101, 303, 202, 404, 101,
                               -1,  202, 303, 404, 303, 202, 101};
  float dx[14u * 16u];
  et_mlora_sgmv_stats stats;
  CHECK(et_mlora_sgmv_backward_v1(reg, M, x, row_ids, dy, dx, &stats) ==
        ET_MLORA_OK);
  CHECK(stats.base_only_rows == 2u);
  CHECK(stats.adapted_rows == 12u);
  CHECK(stats.active_segments == 4u);

  /* Verify frozen base SHA-256 unchanged. */
  CHECK(et_mlora_get_base_sha256_v1(reg, base_after) == ET_MLORA_OK);
  CHECK(memcmp(base_before, base_after, sizeof(base_before)) == 0);

  /* Verify ZERO cross-adapter gradient bleed on idle adapter 505 (all bits 0). */
  float da_idle[4u * 16u];
  float db_idle[12u * 4u];
  CHECK(et_mlora_get_adapter_grads_v1(reg, 505, da_idle, db_idle) ==
        ET_MLORA_OK);
  for (size_t i = 0u; i < 4u * 16u; ++i)
    CHECK(f32_bits(da_idle[i]) == 0u);
  for (size_t i = 0u; i < 12u * 4u; ++i)
    CHECK(f32_bits(db_idle[i]) == 0u);

  /* Verify bitwise gradient isolation: each active adapter's (dA, dB) equals
   * the exact gradient from running backward on a compact sub-batch containing
   * ONLY that adapter's own rows! Save full-batch gradients first because each
   * single-adapter sub-batch backward zeroes all non-participating adapters. */
  float da_batch_all[4][12u * 16u];
  float db_batch_all[4][12u * 12u];
  for (size_t k = 0u; k < 4u; ++k) {
    CHECK(et_mlora_get_adapter_grads_v1(reg, ids[k], da_batch_all[k],
                                        db_batch_all[k]) == ET_MLORA_OK);
  }
  for (size_t k = 0u; k < 4u; ++k) {
    const int32_t aid = ids[k];
    const uint32_t r = ranks[k];

    float sub_x[14u * 16u];
    float sub_dy[14u * 12u];
    int32_t sub_ids[14];
    uint32_t sub_m = 0u;
    for (uint32_t i = 0u; i < M; ++i) {
      if (row_ids[i] == aid) {
        memcpy(&sub_x[(size_t)sub_m * in_dim], &x[(size_t)i * in_dim],
               in_dim * sizeof(float));
        memcpy(&sub_dy[(size_t)sub_m * out_dim], &dy[(size_t)i * out_dim],
               out_dim * sizeof(float));
        sub_ids[sub_m] = aid;
        ++sub_m;
      }
    }
    float sub_dx[14u * 16u];
    CHECK(et_mlora_sgmv_backward_v1(reg, sub_m, sub_x, sub_ids, sub_dy, sub_dx,
                                    NULL) == ET_MLORA_OK);
    float da_sub[12u * 16u];
    float db_sub[12u * 12u];
    CHECK(et_mlora_get_adapter_grads_v1(reg, aid, da_sub, db_sub) ==
          ET_MLORA_OK);
    CHECK(memcmp(da_batch_all[k], da_sub, (size_t)r * in_dim * sizeof(float)) ==
          0);
    CHECK(memcmp(db_batch_all[k], db_sub, (size_t)out_dim * r * sizeof(float)) ==
          0);
  }

  /* Restore full-batch gradients for central finite-difference checks. */
  CHECK(et_mlora_sgmv_backward_v1(reg, M, x, row_ids, dy, dx, NULL) ==
        ET_MLORA_OK);

  const float eps = 1.0f / 512.0f; /* exact power of two -> exact +/- eps */
  double max_err = 0.0;

  /* Central FD on dX (all M * in_dim elements). */
  float x_pert[14u * 16u];
  memcpy(x_pert, x, sizeof(x));
  for (size_t i = 0u; i < (size_t)M * in_dim; ++i) {
    const float orig = x_pert[i];
    x_pert[i] = orig + eps;
    const double lp = eval_weighted_loss(reg, M, x_pert, row_ids, dy, out_dim);
    x_pert[i] = orig - eps;
    const double lm = eval_weighted_loss(reg, M, x_pert, row_ids, dy, out_dim);
    x_pert[i] = orig;
    const double num = (lp - lm) / (2.0 * (double)eps);
    const double err = fabs((double)dx[i] - num);
    if (err > max_err)
      max_err = err;
    CHECK(err < 1e-5);
  }

  /* Central FD on dA and dB across ALL 4 active heterogeneous ranks. */
  for (size_t k = 0u; k < 4u; ++k) {
    const int32_t aid = ids[k];
    const uint32_t r = ranks[k];
    float a[12u * 16u];
    float b[12u * 12u];
    float da[12u * 16u];
    float db[12u * 12u];
    CHECK(et_mlora_get_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
    CHECK(et_mlora_get_adapter_grads_v1(reg, aid, da, db) == ET_MLORA_OK);

    for (size_t i = 0u; i < (size_t)r * in_dim; ++i) {
      const float orig = a[i];
      a[i] = orig + eps;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double lp = eval_weighted_loss(reg, M, x, row_ids, dy, out_dim);
      a[i] = orig - eps;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double lm = eval_weighted_loss(reg, M, x, row_ids, dy, out_dim);
      a[i] = orig;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double err = fabs((double)da[i] - num);
      if (err > max_err)
        max_err = err;
      CHECK(err < 1e-5);
    }
    for (size_t i = 0u; i < (size_t)out_dim * r; ++i) {
      const float orig = b[i];
      b[i] = orig + eps;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double lp = eval_weighted_loss(reg, M, x, row_ids, dy, out_dim);
      b[i] = orig - eps;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double lm = eval_weighted_loss(reg, M, x, row_ids, dy, out_dim);
      b[i] = orig;
      CHECK(et_mlora_set_adapter_weights_v1(reg, aid, a, b) == ET_MLORA_OK);
      const double num = (lp - lm) / (2.0 * (double)eps);
      const double err = fabs((double)db[i] - num);
      if (err > max_err)
        max_err = err;
      CHECK(err < 1e-5);
    }
  }

  forbid_alloc = 0;
  et_mlora_registry_destroy_v1(reg);
  *out_max_fd_err = max_err;
}

/* ---- Test 3: 12-Step Multi-Task AdamW Training + Pin/Hot-Swap + ETML1000 ---- */

static void test_training_hotswap_and_bundle(void) {
  const uint32_t in_dim = 16u;
  const uint32_t out_dim = 12u;
  const uint32_t max_adapters = 6u;
  const uint32_t max_rank = 8u;
  const uint32_t M = 12u;
  const et_mlora_registry_config cfg = {in_dim,   out_dim, max_adapters,
                                        max_rank, 1u,      0u};

  float w_base[12u * 16u];
  float bias[12u];
  float x[12u * 16u];
  fill_det(w_base, sizeof(w_base) / sizeof(w_base[0]), 7u, 0.03125f);
  fill_det(bias, sizeof(bias) / sizeof(bias[0]), 13u, 0.0625f);
  fill_det(x, sizeof(x) / sizeof(x[0]), 23u, 0.0625f);

  et_mlora_registry *reg = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w_base, bias, &reg) == ET_MLORA_OK);

  const int32_t row_ids[12] = {303, -1,  101, 202, 303, -1,
                               202, 101, -1,  303, 101, 202};
  const int32_t active_ids[3] = {101, 202, 303};
  const uint32_t active_ranks[3] = {2u, 4u, 8u};
  const float active_alphas[3] = {4.0f, 8.0f, 16.0f};

  uint8_t base_sha_init[ET_MLORA_SHA256_BYTES];
  uint8_t base_sha_now[ET_MLORA_SHA256_BYTES];

  forbid_alloc = 1;

  CHECK(et_mlora_get_base_sha256_v1(reg, base_sha_init) == ET_MLORA_OK);
  for (size_t k = 0u; k < 3u; ++k) {
    CHECK(et_mlora_register_seeded_adapter_v1(reg, active_ids[k],
                                              active_ranks[k], active_alphas[k],
                                              1000u + k, NULL) == ET_MLORA_OK);
  }
  /* Idle 4th adapter. */
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 404, 4u, 8.0f, 9999u, NULL) ==
        ET_MLORA_OK);

  /* Capture idle adapter initial weights to verify zero mutation. */
  float idle_a0[4u * 16u];
  float idle_b0[12u * 4u];
  CHECK(et_mlora_get_adapter_weights_v1(reg, 404, idle_a0, idle_b0) ==
        ET_MLORA_OK);

  /* Build distinct per-adapter targets = y_base + task_delta_a. */
  float y_base[12u * 12u];
  CHECK(et_mlora_single_adapter_forward_v1(reg, ET_MLORA_BASE_ONLY_ID, M, x,
                                           y_base) == ET_MLORA_OK);
  float target[12u * 12u];
  for (uint32_t i = 0u; i < M; ++i) {
    unsigned salt = 71u;
    float scale = 0.0f;
    if (row_ids[i] == 101) {
      salt = 83u;
      scale = 0.05f;
    } else if (row_ids[i] == 202) {
      salt = 97u;
      scale = -0.06f;
    } else if (row_ids[i] == 303) {
      salt = 109u;
      scale = 0.07f;
    }
    for (uint32_t o = 0u; o < out_dim; ++o) {
      const size_t idx = (size_t)i * out_dim + o;
      const int b = (int)((29u * (unsigned)idx + 13u * salt) % 31u) - 15;
      target[idx] = y_base[idx] + scale * (float)b;
    }
  }

  /* 12-step heterogeneous multi-adapter AdamW training. */
  const et_mlora_adamw_config acfg = {0.01f, 0.9f, 0.999f, 1e-8f, 0.001f};
  double prev_loss[3] = {1e30, 1e30, 1e30};
  float y[12u * 12u];
  float dy[12u * 12u];
  float dx[12u * 16u];

  for (int step = 0; step <= 12; ++step) {
    CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y, NULL) == ET_MLORA_OK);
    double loss[3] = {0.0, 0.0, 0.0};
    memset(dy, 0, sizeof(dy));
    for (uint32_t i = 0u; i < M; ++i) {
      int task = -1;
      if (row_ids[i] == 101)
        task = 0;
      else if (row_ids[i] == 202)
        task = 1;
      else if (row_ids[i] == 303)
        task = 2;
      if (task < 0)
        continue;
      const double inv = 1.0 / (double)(3u * out_dim);
      for (uint32_t o = 0u; o < out_dim; ++o) {
        const size_t idx = (size_t)i * out_dim + o;
        const double diff = (double)y[idx] - (double)target[idx];
        loss[task] += 0.5 * inv * diff * diff;
        dy[idx] = (float)(inv * diff);
      }
    }
    if (step > 0) {
      for (int t = 0; t < 3; ++t)
        CHECK(loss[t] < prev_loss[t]);
    }
    for (int t = 0; t < 3; ++t)
      prev_loss[t] = loss[t];

    if (step < 12) {
      CHECK(et_mlora_sgmv_backward_v1(reg, M, x, row_ids, dy, dx, NULL) ==
            ET_MLORA_OK);
      for (int t = 0; t < 3; ++t)
        CHECK(et_mlora_adamw_step_adapter_v1(reg, active_ids[t], &acfg) ==
              ET_MLORA_OK);
    }

    CHECK(et_mlora_get_base_sha256_v1(reg, base_sha_now) == ET_MLORA_OK);
    CHECK(memcmp(base_sha_init, base_sha_now, sizeof(base_sha_init)) == 0);
  }

  /* Verify idle adapter 404 weights and step remained untouched. */
  float idle_a1[4u * 16u];
  float idle_b1[12u * 4u];
  et_mlora_adapter_info idle_info;
  CHECK(et_mlora_get_adapter_weights_v1(reg, 404, idle_a1, idle_b1) ==
        ET_MLORA_OK);
  CHECK(memcmp(idle_a0, idle_a1, sizeof(idle_a0)) == 0);
  CHECK(memcmp(idle_b0, idle_b1, sizeof(idle_b0)) == 0);
  CHECK(et_mlora_get_adapter_info_v1(reg, 404, &idle_info) == ET_MLORA_OK);
  CHECK(idle_info.adamw_step == 0u);

  /* Pin / unpin / hot-swap lifecycle on adapter 404 (zero-alloc!). */
  uint32_t pins = 0u;
  const uint32_t old_slot = idle_info.slot_index;
  const uint32_t old_gen = idle_info.generation;
  CHECK(et_mlora_pin_adapter_v1(reg, 404, &pins) == ET_MLORA_OK && pins == 1u);
  CHECK(et_mlora_pin_adapter_v1(reg, 404, &pins) == ET_MLORA_OK && pins == 2u);
  CHECK(et_mlora_unregister_adapter_v1(reg, 404) == ET_MLORA_ERR_BUSY);
  CHECK(et_mlora_unpin_adapter_v1(reg, 404, &pins) == ET_MLORA_OK &&
        pins == 1u);
  CHECK(et_mlora_unregister_adapter_v1(reg, 404) == ET_MLORA_ERR_BUSY);
  CHECK(et_mlora_unpin_adapter_v1(reg, 404, &pins) == ET_MLORA_OK &&
        pins == 0u);
  /* Double unpin -> ET_MLORA_ERR_UNDERFLOW. */
  CHECK(et_mlora_unpin_adapter_v1(reg, 404, &pins) == ET_MLORA_ERR_UNDERFLOW);
  CHECK(et_mlora_unregister_adapter_v1(reg, 404) == ET_MLORA_OK);
  CHECK(et_mlora_unregister_adapter_v1(reg, 404) == ET_MLORA_ERR_NOT_FOUND);

  /* Hot-load replacement adapter 505 into freed slot and check generation++. */
  et_mlora_adapter_info new_info;
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 505, 3u, 6.0f, 5555u,
                                            &new_info) == ET_MLORA_OK);
  CHECK(new_info.slot_index == old_slot);
  CHECK(new_info.generation == old_gen + 1u);

  /* Duplicate ID and pool-full checks. */
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 505, 2u, 4.0f, 1u, NULL) ==
        ET_MLORA_ERR_ALREADY_EXISTS);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 606, 2u, 4.0f, 2u, NULL) ==
        ET_MLORA_OK);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 707, 2u, 4.0f, 3u, NULL) ==
        ET_MLORA_OK);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 808, 2u, 4.0f, 4u, NULL) ==
        ET_MLORA_ERR_FULL);
  CHECK(et_mlora_unregister_adapter_v1(reg, 606) == ET_MLORA_OK);
  CHECK(et_mlora_unregister_adapter_v1(reg, 707) == ET_MLORA_OK);

  /* Capture post-training batched Y before bundle save. */
  float y_before_save[12u * 12u];
  CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y_before_save, NULL) ==
        ET_MLORA_OK);

  forbid_alloc = 0;

  /* ETML1000 v1.0 bundle save (with AdamW state) -> reload into fresh registry. */
  char tmpdir[] = "/tmp/mlora-test-XXXXXX";
  CHECK(mkdtemp(tmpdir) != NULL);
  char bpath[256];
  char tpath[256];
  char spath[256];
  snprintf(bpath, sizeof(bpath), "%s/adapters.etml", tmpdir);
  snprintf(tpath, sizeof(tpath), "%s/tampered.etml", tmpdir);
  snprintf(spath, sizeof(spath), "%s/symlink.etml", tmpdir);

  CHECK(et_mlora_bundle_save_v1(bpath, reg, 1u) == ET_MLORA_OK);

  struct stat st;
  CHECK(stat(bpath, &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);

  et_mlora_registry *fresh = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w_base, bias, &fresh) == ET_MLORA_OK);
  et_mlora_bundle_meta meta;
  CHECK(et_mlora_bundle_load_v1(bpath, fresh, &meta) == ET_MLORA_OK);
  CHECK(meta.version_major == 1u && meta.version_minor == 0u);
  CHECK(meta.adapter_count == 4u);
  CHECK((meta.flags & ET_MLORA_BUNDLE_FLAG_HAS_BIAS) != 0u);
  CHECK((meta.flags & ET_MLORA_BUNDLE_FLAG_HAS_ADAMW) != 0u);
  CHECK(memcmp(meta.base_sha256, base_sha_init, ET_MLORA_SHA256_BYTES) == 0);

  /* Verify bitwise identical batched outputs AND AdamW continuation parity. */
  forbid_alloc = 1;
  float y_after_load[12u * 12u];
  CHECK(et_mlora_sgmv_forward_v1(fresh, M, x, row_ids, y_after_load, NULL) ==
        ET_MLORA_OK);
  CHECK(memcmp(y_before_save, y_after_load, sizeof(y_before_save)) == 0);

  /* One more backward + AdamW step on both reg and fresh -> bitwise identical! */
  CHECK(et_mlora_sgmv_backward_v1(reg, M, x, row_ids, dy, dx, NULL) ==
        ET_MLORA_OK);
  CHECK(et_mlora_sgmv_backward_v1(fresh, M, x, row_ids, dy, dx, NULL) ==
        ET_MLORA_OK);
  for (int t = 0; t < 3; ++t) {
    CHECK(et_mlora_adamw_step_adapter_v1(reg, active_ids[t], &acfg) ==
          ET_MLORA_OK);
    CHECK(et_mlora_adamw_step_adapter_v1(fresh, active_ids[t], &acfg) ==
          ET_MLORA_OK);
  }
  CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y_before_save, NULL) ==
        ET_MLORA_OK);
  CHECK(et_mlora_sgmv_forward_v1(fresh, M, x, row_ids, y_after_load, NULL) ==
        ET_MLORA_OK);
  CHECK(memcmp(y_before_save, y_after_load, sizeof(y_before_save)) == 0);

  /* Reject loading over a pinned adapter (failure-atomic!). */
  CHECK(et_mlora_pin_adapter_v1(fresh, 101, NULL) == ET_MLORA_OK);
  forbid_alloc = 0;
  CHECK(et_mlora_bundle_load_v1(bpath, fresh, NULL) == ET_MLORA_ERR_BUSY);
  CHECK(et_mlora_unpin_adapter_v1(fresh, 101, NULL) == ET_MLORA_OK);

  /* Tamper / truncation / symlink / base-hash mismatch checks. */
  CHECK(symlink(bpath, spath) == 0);
  CHECK(et_mlora_bundle_load_v1(spath, fresh, NULL) == ET_MLORA_ERR_SYMLINK);
  CHECK(unlink(spath) == 0);

  /* Base-hash mismatch on altered W_base. */
  float w_alt[12u * 16u];
  memcpy(w_alt, w_base, sizeof(w_base));
  w_alt[0] += 0.25f;
  et_mlora_registry *alt_reg = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w_alt, bias, &alt_reg) == ET_MLORA_OK);
  CHECK(et_mlora_bundle_load_v1(bpath, alt_reg, NULL) == ET_MLORA_ERR_BASE_HASH);
  et_mlora_registry_destroy_v1(alt_reg);

  /* Buffer-level serialize/parse tamper checks. */
  size_t bsize = 0u;
  CHECK(et_mlora_bundle_size_v1(reg, 0u, &bsize) == ET_MLORA_OK);
  uint8_t *buf = (uint8_t *)malloc(bsize + 4u);
  CHECK(buf != NULL);
  size_t written = 0u;
  CHECK(et_mlora_bundle_serialize_v1(reg, 0u, buf, bsize, &written) ==
        ET_MLORA_OK);
  CHECK(written == bsize);

  /* 1. Checksum tamper. */
  buf[bsize - 1u] ^= 0xffu;
  CHECK(et_mlora_bundle_parse_v1(buf, bsize, fresh, NULL) ==
        ET_MLORA_ERR_CHECKSUM);
  buf[bsize - 1u] ^= 0xffu;

  /* 2. Truncated. */
  CHECK(et_mlora_bundle_parse_v1(buf, bsize - 5u, fresh, NULL) ==
        ET_MLORA_ERR_TRUNCATED);

  /* 3. Trailing bytes. */
  buf[bsize] = 0u;
  CHECK(et_mlora_bundle_parse_v1(buf, bsize + 1u, fresh, NULL) ==
        ET_MLORA_ERR_FORMAT);

  /* 4. Bad magic. */
  buf[0] = 'Z';
  CHECK(et_mlora_bundle_parse_v1(buf, bsize, fresh, NULL) ==
        ET_MLORA_ERR_FORMAT);
  buf[0] = 'E';

  /* 5. Bad version. */
  buf[8] = 2u;
  CHECK(et_mlora_bundle_parse_v1(buf, bsize, fresh, NULL) ==
        ET_MLORA_ERR_VERSION);
  buf[8] = 1u;

  /* Confirm failure atomicity: fresh still produces y_after_load bitwise. */
  float y_check[12u * 12u];
  CHECK(et_mlora_sgmv_forward_v1(fresh, M, x, row_ids, y_check, NULL) ==
        ET_MLORA_OK);
  CHECK(memcmp(y_after_load, y_check, sizeof(y_check)) == 0);

  free(buf);
  (void)unlink(bpath);
  (void)unlink(tpath);
  (void)rmdir(tmpdir);
  et_mlora_registry_destroy_v1(fresh);
  et_mlora_registry_destroy_v1(reg);
}

/* ---- Test 4: Negative Boundary, Failure Atomicity & FP-Env Traps ---- */

static void test_negatives_and_fp_env(void) {
  const et_mlora_registry_config bad_dim = {0u, 8u, 4u, 4u, 0u, 0u};
  const et_mlora_registry_config bad_rank = {8u, 8u, 4u, 9u, 0u, 0u};
  const et_mlora_registry_config bad_res = {8u, 8u, 4u, 4u, 0u, 1u};
  CHECK(et_mlora_validate_registry_config_v1(&bad_dim) == ET_MLORA_ERR_SHAPE);
  CHECK(et_mlora_validate_registry_config_v1(&bad_rank) == ET_MLORA_ERR_SHAPE);
  CHECK(et_mlora_validate_registry_config_v1(&bad_res) == ET_MLORA_ERR_ARGUMENT);

  const et_mlora_registry_config cfg = {4u, 4u, 4u, 2u, 0u, 0u};
  float w[16u];
  fill_det(w, 16u, 5u, 0.125f);
  et_mlora_registry *reg = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w, NULL, &reg) == ET_MLORA_OK);

  forbid_alloc = 1;
  CHECK(et_mlora_register_seeded_adapter_v1(reg, -1, 2u, 4.0f, 1u, NULL) ==
        ET_MLORA_ERR_ARGUMENT);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 1, 0u, 4.0f, 1u, NULL) ==
        ET_MLORA_ERR_SHAPE);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 1, 3u, 4.0f, 1u, NULL) ==
        ET_MLORA_ERR_SHAPE);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 1, 2u, 0.0f, 1u, NULL) ==
        ET_MLORA_ERR_ARGUMENT);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 1, 2u, NAN, 1u, NULL) ==
        ET_MLORA_ERR_NONFINITE);
  CHECK(et_mlora_register_seeded_adapter_v1(reg, 1, 2u, 4.0f, 1u, NULL) ==
        ET_MLORA_OK);

  /* Failure atomicity on non-finite X and unknown adapter id: Y untouched! */
  float x[8u];
  fill_det(x, 8u, 9u, 0.25f);
  float y[8u];
  for (size_t i = 0u; i < 8u; ++i)
    y[i] = 42.5f;
  const int32_t bad_ids[2] = {1, 999};
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, bad_ids, y, NULL) ==
        ET_MLORA_ERR_NOT_FOUND);
  for (size_t i = 0u; i < 8u; ++i)
    CHECK(y[i] == 42.5f);

  const int32_t neg_ids[2] = {1, -2};
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, neg_ids, y, NULL) ==
        ET_MLORA_ERR_ARGUMENT);

  const int32_t ok_ids[2] = {1, -1};
  x[3] = INFINITY;
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, ok_ids, y, NULL) ==
        ET_MLORA_ERR_NONFINITE);
  for (size_t i = 0u; i < 8u; ++i)
    CHECK(y[i] == 42.5f);
  x[3] = 0.25f;

  /* Aliasing rejection (x == y). */
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, ok_ids, x, NULL) ==
        ET_MLORA_ERR_ALIAS);

  /* FP-environment trap: FE_UPWARD rejected, then restored. */
  const int saved_round = fegetround();
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, ok_ids, y, NULL) ==
        ET_MLORA_ERR_FP_ENV);
  CHECK(fesetround(saved_round) == 0);

#if defined(__x86_64__)
  /* x86_64 MXCSR FTZ trap. */
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u);
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, ok_ids, y, NULL) ==
        ET_MLORA_ERR_FP_ENV);
  _mm_setcsr(saved_mxcsr);
#endif

  /* Confirm normal call succeeds again after FP-env restoration. */
  CHECK(et_mlora_sgmv_forward_v1(reg, 2u, x, ok_ids, y, NULL) == ET_MLORA_OK);

  forbid_alloc = 0;
  et_mlora_registry_destroy_v1(reg);
}

/* ---- Oracle Data Emitter (--oracle) ---- */

static void emit_oracle_data(void) {
  /* Part 1: 4 heterogeneous SGMV forward/backward cases (covering has_bias 0/1,
   * ranks 1,2,4,8,16, base-only -1, and an idle adapter). */
  typedef struct case_spec {
    uint32_t M, in_dim, out_dim, has_bias, num_adapters;
    int32_t ids[6];
    uint32_t ranks[6];
    float alphas[6];
    int32_t row_ids[18];
  } case_spec;

  const case_spec cases[3] = {
      {12u,
       16u,
       12u,
       1u,
       4u,
       {101, 202, 303, 404, 0, 0},
       {2u, 4u, 8u, 4u, 0u, 0u},
       {4.0f, 8.0f, 16.0f, 8.0f, 0.0f, 0.0f},
       {303, -1, 101, 202, 303, -1, 202, 101, -1, 303, 101, 202, 0, 0, 0, 0, 0,
        0}},
      {18u,
       20u,
       16u,
       1u,
       6u,
       {10, 20, 30, 40, 50, 60},
       {1u, 2u, 4u, 8u, 16u, 4u},
       {2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 8.0f},
       {50, -1, 20, 10, 40, 30, -1, 50, 10, 30, 20, 40, 10, 50, -1, 40, 30,
        20}},
      {8u,
       12u,
       10u,
       0u,
       3u,
       {5, 15, 25, 0, 0, 0},
       {1u, 3u, 5u, 0u, 0u, 0u},
       {1.5f, 6.0f, 10.0f, 0.0f, 0.0f, 0.0f},
       {25, 5, -1, 25, -1, 5, 25, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
  };

  for (size_t c = 0u; c < 3u; ++c) {
    const case_spec *sp = &cases[c];
    const et_mlora_registry_config cfg = {
        sp->in_dim, sp->out_dim, 8u, 16u < sp->out_dim ? 16u : sp->out_dim,
        sp->has_bias, 0u};
    float w_base[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_IN_DIM];
    float bias[ET_MLORA_MAX_OUT_DIM];
    float x[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
    float dy[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
    fill_det(w_base, (size_t)sp->out_dim * sp->in_dim, (unsigned)(11u + c * 17u),
             0.03125f);
    fill_det(bias, sp->out_dim, (unsigned)(19u + c * 17u), 0.0625f);
    fill_det(x, (size_t)sp->M * sp->in_dim, (unsigned)(29u + c * 17u), 0.0625f);
    fill_det(dy, (size_t)sp->M * sp->out_dim, (unsigned)(37u + c * 17u),
             0.0625f);

    et_mlora_registry *reg = NULL;
    CHECK(et_mlora_registry_create_v1(&cfg, w_base, sp->has_bias ? bias : NULL,
                                      &reg) == ET_MLORA_OK);

    printf("SGMV_CASE %u %u %u %u %u\n", sp->M, sp->in_dim, sp->out_dim,
           sp->has_bias, sp->num_adapters);
    print_f32_hex(w_base, (size_t)sp->out_dim * sp->in_dim);
    if (sp->has_bias)
      print_f32_hex(bias, sp->out_dim);
    print_f32_hex(x, (size_t)sp->M * sp->in_dim);
    print_f32_hex(dy, (size_t)sp->M * sp->out_dim);
    for (uint32_t i = 0u; i < sp->M; ++i)
      printf("%s%d", i ? " " : "", sp->row_ids[i]);
    putchar('\n');

    for (uint32_t k = 0u; k < sp->num_adapters; ++k) {
      float a[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
      float b[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
      fill_det(a, (size_t)sp->ranks[k] * sp->in_dim,
               (unsigned)(100u + c * 50u + k * 7u), 0.0625f);
      fill_det(b, (size_t)sp->out_dim * sp->ranks[k],
               (unsigned)(200u + c * 50u + k * 11u), 0.0625f);
      CHECK(et_mlora_register_adapter_v1(reg, sp->ids[k], sp->ranks[k],
                                         sp->alphas[k], a, b, NULL) ==
            ET_MLORA_OK);
      printf("ADAPTER %d %u %a\n", sp->ids[k], sp->ranks[k],
             (double)sp->alphas[k]);
      print_f32_hex(a, (size_t)sp->ranks[k] * sp->in_dim);
      print_f32_hex(b, (size_t)sp->out_dim * sp->ranks[k]);
    }

    float y[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
    float dx[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
    CHECK(et_mlora_sgmv_forward_v1(reg, sp->M, x, sp->row_ids, y, NULL) ==
          ET_MLORA_OK);
    CHECK(et_mlora_sgmv_backward_v1(reg, sp->M, x, sp->row_ids, dy, dx, NULL) ==
          ET_MLORA_OK);

    print_f32_hex(y, (size_t)sp->M * sp->out_dim);
    print_f32_hex(dx, (size_t)sp->M * sp->in_dim);
    for (uint32_t k = 0u; k < sp->num_adapters; ++k) {
      float da[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
      float db[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
      CHECK(et_mlora_get_adapter_grads_v1(reg, sp->ids[k], da, db) ==
            ET_MLORA_OK);
      print_f32_hex(da, (size_t)sp->ranks[k] * sp->in_dim);
      print_f32_hex(db, (size_t)sp->out_dim * sp->ranks[k]);
    }
    et_mlora_registry_destroy_v1(reg);
  }

  /* Part 2: 12-step multi-adapter AdamW trajectory for Python oracle parity. */
  const uint32_t M = 12u, in_dim = 16u, out_dim = 12u;
  const et_mlora_registry_config cfg = {in_dim, out_dim, 6u, 8u, 1u, 0u};
  float w_base[12u * 16u], bias[12u], x[12u * 16u];
  fill_det(w_base, sizeof(w_base) / sizeof(w_base[0]), 7u, 0.03125f);
  fill_det(bias, sizeof(bias) / sizeof(bias[0]), 13u, 0.0625f);
  fill_det(x, sizeof(x) / sizeof(x[0]), 23u, 0.0625f);
  const int32_t row_ids[12] = {303, -1,  101, 202, 303, -1,
                               202, 101, -1,  303, 101, 202};
  const int32_t ids[3] = {101, 202, 303};
  const uint32_t ranks[3] = {2u, 4u, 8u};
  const float alphas[3] = {4.0f, 8.0f, 16.0f};
  const et_mlora_adamw_config acfg = {0.01f, 0.9f, 0.999f, 1e-8f, 0.001f};

  et_mlora_registry *reg = NULL;
  CHECK(et_mlora_registry_create_v1(&cfg, w_base, bias, &reg) == ET_MLORA_OK);

  float y_base[12u * 12u], target[12u * 12u];
  CHECK(et_mlora_single_adapter_forward_v1(reg, ET_MLORA_BASE_ONLY_ID, M, x,
                                           y_base) == ET_MLORA_OK);
  for (uint32_t i = 0u; i < M; ++i) {
    unsigned salt = 71u;
    float scale = 0.0f;
    if (row_ids[i] == 101) {
      salt = 83u;
      scale = 0.05f;
    } else if (row_ids[i] == 202) {
      salt = 97u;
      scale = -0.06f;
    } else if (row_ids[i] == 303) {
      salt = 109u;
      scale = 0.07f;
    }
    for (uint32_t o = 0u; o < out_dim; ++o) {
      const size_t idx = (size_t)i * out_dim + o;
      const int b = (int)((29u * (unsigned)idx + 13u * salt) % 31u) - 15;
      target[idx] = y_base[idx] + scale * (float)b;
    }
  }

  printf("ADAMW_TRAJ %u %u %u 12 %a %a %a %a %a\n", M, in_dim, out_dim,
         (double)acfg.lr, (double)acfg.beta1, (double)acfg.beta2,
         (double)acfg.eps, (double)acfg.weight_decay);
  print_f32_hex(w_base, sizeof(w_base) / sizeof(w_base[0]));
  print_f32_hex(bias, sizeof(bias) / sizeof(bias[0]));
  print_f32_hex(x, sizeof(x) / sizeof(x[0]));
  print_f32_hex(target, sizeof(target) / sizeof(target[0]));
  for (uint32_t i = 0u; i < M; ++i)
    printf("%s%d", i ? " " : "", row_ids[i]);
  putchar('\n');

  for (int t = 0; t < 3; ++t) {
    CHECK(et_mlora_register_seeded_adapter_v1(reg, ids[t], ranks[t], alphas[t],
                                              1000u + (uint64_t)t, NULL) ==
          ET_MLORA_OK);
    float a[8u * 16u], b[12u * 8u];
    CHECK(et_mlora_get_adapter_weights_v1(reg, ids[t], a, b) == ET_MLORA_OK);
    printf("TRAJ_ADAPTER %d %u %a\n", ids[t], ranks[t], (double)alphas[t]);
    print_f32_hex(a, (size_t)ranks[t] * in_dim);
    print_f32_hex(b, (size_t)out_dim * ranks[t]);
  }

  float y[12u * 12u], dy[12u * 12u], dx[12u * 16u];
  for (int step = 0; step <= 12; ++step) {
    CHECK(et_mlora_sgmv_forward_v1(reg, M, x, row_ids, y, NULL) == ET_MLORA_OK);
    double loss[3] = {0.0, 0.0, 0.0};
    memset(dy, 0, sizeof(dy));
    for (uint32_t i = 0u; i < M; ++i) {
      int task = -1;
      if (row_ids[i] == 101)
        task = 0;
      else if (row_ids[i] == 202)
        task = 1;
      else if (row_ids[i] == 303)
        task = 2;
      if (task < 0)
        continue;
      const double inv = 1.0 / (double)(3u * out_dim);
      for (uint32_t o = 0u; o < out_dim; ++o) {
        const size_t idx = (size_t)i * out_dim + o;
        const double diff = (double)y[idx] - (double)target[idx];
        loss[task] += 0.5 * inv * diff * diff;
        dy[idx] = (float)(inv * diff);
      }
    }
    printf("STEP %d %a %a %a\n", step, loss[0], loss[1], loss[2]);
    if (step < 12) {
      CHECK(et_mlora_sgmv_backward_v1(reg, M, x, row_ids, dy, dx, NULL) ==
            ET_MLORA_OK);
      for (int t = 0; t < 3; ++t)
        CHECK(et_mlora_adamw_step_adapter_v1(reg, ids[t], &acfg) ==
              ET_MLORA_OK);
    }
  }
  print_f32_hex(y, sizeof(y) / sizeof(y[0]));
  et_mlora_registry_destroy_v1(reg);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    emit_oracle_data();
    return 0;
  }
  double max_fd_err = 0.0;
  test_heterogeneous_ranks_and_parity();
  test_backward_fd_and_zero_bleed(&max_fd_err);
  test_training_hotswap_and_bundle();
  test_negatives_and_fp_env();
  CHECK(alloc_calls_while_forbidden == 0u);
  printf("MULTI-LORA SGMV native C11 tests: PASS "
         "(ranks=0,1,2,4,8,16 bitwise_parity=0.000e+00 max_fd_err=%.3e "
         "zero_bleed=0.000e+00 hot_path_allocs=0)\n",
         max_fd_err);
  return 0;
}
