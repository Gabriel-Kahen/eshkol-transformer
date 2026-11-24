#define _GNU_SOURCE

#include "eshkol_transformer/mod5_adamw_abi.h"

#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg));        \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static void make_base_config(et_mod5_adamw_config_v1 *cfg, uint64_t n) {
  const char *model_tag = "mod5-native-test-model-v1";
  memset(cfg, 0, sizeof(*cfg));
  cfg->param_count = n;
  cfg->beta1 = 0.9f;
  cfg->beta2 = 0.999f;
  cfg->eps = 1e-8f;
  cfg->weight_decay = 0.01f;
  cfg->max_grad_norm = 0.5f;
  cfg->schedule.kind = ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY;
  cfg->schedule.base_lr = 0.05;
  cfg->schedule.min_lr = 0.005;
  cfg->schedule.warmup_steps = 4u;
  cfg->schedule.total_steps = 16u;
  cfg->tokenizer_id = "mod5-native-token-v1";
  CHECK(et_mod5_adamw_sha256_bytes((const uint8_t *)model_tag,
                                   strlen(model_tag),
                                   cfg->model_config_hash) == ET_MOD5_ADAMW_OK,
        "hash model config");
}

static uint8_t *read_raw_file(const char *path, size_t *out_len) {
  FILE *f = fopen(path, "rb");
  long sz;
  uint8_t *buf;
  if (f == NULL) {
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  sz = ftell(f);
  if (sz < 0 || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  buf = (uint8_t *)malloc((size_t)sz + 1u);
  if (buf == NULL) {
    fclose(f);
    return NULL;
  }
  if (sz > 0 && fread(buf, 1u, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_len = (size_t)sz;
  return buf;
}

static int write_raw_file(const char *path, const uint8_t *buf, size_t len) {
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    return -1;
  }
  if (len > 0u && fwrite(buf, 1u, len, f) != len) {
    fclose(f);
    return -1;
  }
  return fclose(f);
}

static void resign_snapshot(uint8_t *buf, size_t len) {
  uint8_t digest[32];
  CHECK(len >= ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES, "resign len");
  CHECK(et_mod5_adamw_sha256_bytes(
            buf, len - ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES, digest) ==
            ET_MOD5_ADAMW_OK,
        "resign sha256");
  memcpy(buf + len - ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES, digest, 32u);
}

static void test_lr_schedules(void) {
  et_mod5_lr_schedule_config_v1 sched;
  double lr = -1.0;

  /* CONSTANT */
  memset(&sched, 0, sizeof(sched));
  sched.kind = ET_MOD5_LR_CONSTANT;
  sched.base_lr = 0.02;
  sched.min_lr = 0.01;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) == ET_MOD5_ADAMW_OK &&
            lr == 0.02,
        "constant lr step 1");
  CHECK(et_mod5_adamw_compute_lr(&sched, 100u, &lr) == ET_MOD5_ADAMW_OK &&
            lr == 0.02,
        "constant lr step 100");
  CHECK(et_mod5_adamw_compute_lr(&sched, 0u, &lr) ==
            ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "step 0 rejected");

  /* LINEAR_WARMUP_LINEAR_DECAY */
  sched.kind = ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY;
  sched.base_lr = 0.1;
  sched.min_lr = 0.02;
  sched.warmup_steps = 4u;
  sched.total_steps = 12u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 2u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.05) < 1e-15,
        "linear warmup midpoint");
  CHECK(et_mod5_adamw_compute_lr(&sched, 4u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.1) < 1e-15,
        "linear warmup peak");
  CHECK(et_mod5_adamw_compute_lr(&sched, 8u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.06) < 1e-15,
        "linear decay midpoint");
  CHECK(et_mod5_adamw_compute_lr(&sched, 12u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.02) < 1e-15,
        "linear decay end");
  CHECK(et_mod5_adamw_compute_lr(&sched, 20u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.02) < 1e-15,
        "linear post-total clamps to min_lr");

  /* LINEAR_WARMUP_COSINE_DECAY */
  sched.kind = ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.025) < 1e-15,
        "cosine warmup step 1");
  CHECK(et_mod5_adamw_compute_lr(&sched, 4u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.1) < 1e-15,
        "cosine warmup peak");
  CHECK(et_mod5_adamw_compute_lr(&sched, 8u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.06) < 1e-15,
        "cosine decay midpoint exact 0.5");
  CHECK(et_mod5_adamw_compute_lr(&sched, 12u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.02) < 1e-15,
        "cosine decay end");
  CHECK(et_mod5_adamw_compute_lr(&sched, 99u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.02) < 1e-15,
        "cosine post-total clamps to min_lr");

  /* Edge case: warmup == 0 */
  sched.warmup_steps = 0u;
  sched.total_steps = 4u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 2u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.06) < 1e-15,
        "cosine zero-warmup midpoint");

  /* Edge case: warmup == total */
  sched.warmup_steps = 4u;
  sched.total_steps = 4u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 4u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.1) < 1e-15,
        "warmup==total at peak");
  CHECK(et_mod5_adamw_compute_lr(&sched, 5u, &lr) == ET_MOD5_ADAMW_OK &&
            fabs(lr - 0.02) < 1e-15,
        "warmup==total post-total");

  /* Invalid schedule configs */
  sched.min_lr = 0.2;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) ==
            ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "min_lr > base_lr rejected");
  sched.min_lr = 0.01;
  sched.warmup_steps = 5u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) ==
            ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "warmup > total rejected");
  sched.warmup_steps = 0u;
  sched.total_steps = 0u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) ==
            ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "total==0 rejected");
  sched.total_steps = 4u;
  sched.kind = 99u;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) ==
            ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "unknown schedule kind rejected");
  sched.kind = ET_MOD5_LR_CONSTANT;
  sched.base_lr = NAN;
  CHECK(et_mod5_adamw_compute_lr(&sched, 1u, &lr) ==
            ET_MOD5_ADAMW_ERR_NONFINITE,
        "NaN base_lr rejected");
}

static void test_invalid_hyperparams(void) {
  et_mod5_adamw_config_v1 cfg;
  et_mod5_adamw_state *st = NULL;

  make_base_config(&cfg, 8u);
  cfg.param_count = 0u;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG &&
            st == NULL,
        "zero param_count rejected");

  make_base_config(&cfg, 8u);
  cfg.beta1 = 1.0f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "beta1==1 rejected");

  make_base_config(&cfg, 8u);
  cfg.beta1 = -0.1f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "beta1<0 rejected");

  make_base_config(&cfg, 8u);
  cfg.beta2 = 1.0f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "beta2==1 rejected");

  make_base_config(&cfg, 8u);
  cfg.eps = 0.0f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "eps==0 rejected");

  make_base_config(&cfg, 8u);
  cfg.weight_decay = -0.01f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "negative weight_decay rejected");

  make_base_config(&cfg, 8u);
  cfg.max_grad_norm = -1.0f;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "negative max_grad_norm rejected");

  make_base_config(&cfg, 8u);
  cfg.beta2 = (float)NAN;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_NONFINITE,
        "NaN beta2 rejected");

  make_base_config(&cfg, 8u);
  cfg.tokenizer_id = "";
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "empty tokenizer_id rejected");

  make_base_config(&cfg, 8u);
  char too_long[ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES + 2u];
  memset(too_long, 'a', sizeof(too_long) - 1u);
  too_long[sizeof(too_long) - 1u] = '\0';
  cfg.tokenizer_id = too_long;
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_ERR_INVALID_ARG,
        "oversized tokenizer_id rejected");
}

static void test_step_clipping_and_atomic_rollback(void) {
  et_mod5_adamw_config_v1 cfg;
  et_mod5_adamw_state *st = NULL;
  const float init_p[4] = {1.0f, -2.0f, 0.5f, 3.0f};
  const float big_g[4] = {3.0f, 4.0f, 0.0f, 0.0f}; /* L2 norm = 5.0 > 0.5 */
  const float exact_g[4] = {0.25f, 0.25f, 0.25f, 0.25f}; /* L2 norm = 0.5 == 0.5 exact */
  const float zero_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  et_mod5_adamw_step_info_v1 info;
  float p_before[4], m_before[4], v_before[4];
  float p_after[4], m_after[4], v_after[4];

  make_base_config(&cfg, 4u);
  CHECK(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_OK, "create state");
  CHECK(et_mod5_adamw_set_params(st, init_p, 4u) == ET_MOD5_ADAMW_OK,
        "set params");

  /* Step 1: clipped (norm 5.0 > 0.5 -> clip_scale = 0.1) */
  CHECK(et_mod5_adamw_step(st, big_g, 4u, 8u, &info) == ET_MOD5_ADAMW_OK,
        "step 1 ok");
  CHECK(info.step == 1u && info.clipped == 1u &&
            fabs(info.grad_norm - 5.0) < 1e-12 &&
            fabs(info.clip_scale - 0.1) < 1e-12,
        "step 1 clipping telemetry");

  /* Step 2: exact boundary (norm 0.5 == 0.5 -> unclipped) */
  CHECK(et_mod5_adamw_step(st, exact_g, 4u, 8u, &info) == ET_MOD5_ADAMW_OK,
        "step 2 ok");
  CHECK(info.step == 2u && info.clipped == 0u &&
            fabs(info.grad_norm - 0.5) < 1e-12 && info.clip_scale == 1.0,
        "step 2 exact boundary unclipped");

  /* Step 3: zero gradients still apply decoupled weight decay & advance step */
  CHECK(et_mod5_adamw_get_params(st, p_before, 4u) == ET_MOD5_ADAMW_OK,
        "get params before zero step");
  CHECK(et_mod5_adamw_step(st, zero_g, 4u, 8u, &info) == ET_MOD5_ADAMW_OK,
        "step 3 zero grad ok");
  CHECK(et_mod5_adamw_get_params(st, p_after, 4u) == ET_MOD5_ADAMW_OK,
        "get params after zero step");
  CHECK(info.step == 3u && info.clipped == 0u &&
            fabsf(p_after[3]) < fabsf(p_before[3]),
        "zero grad still applies decoupled decay");

  /* Snapshot live state before negative step failures */
  CHECK(et_mod5_adamw_get_params(st, p_before, 4u) == ET_MOD5_ADAMW_OK,
        "snap p");
  CHECK(et_mod5_adamw_get_moments(st, m_before, v_before, 4u) ==
            ET_MOD5_ADAMW_OK,
        "snap mv");

  /* NaN gradient rejected atomically */
  {
    const float nan_g[4] = {0.1f, (float)NAN, 0.2f, 0.3f};
    CHECK(et_mod5_adamw_step(st, nan_g, 4u, 8u, &info) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "NaN grad rejected");
  }
  /* +Inf gradient rejected atomically */
  {
    const float inf_g[4] = {0.1f, 0.2f, (float)INFINITY, 0.3f};
    CHECK(et_mod5_adamw_step(st, inf_g, 4u, 8u, &info) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "+Inf grad rejected");
  }
  /* Huge gradient overflowing f32 g*g when clipping disabled */
  {
    et_mod5_adamw_config_v1 noclip_cfg;
    et_mod5_adamw_state *nc = NULL;
    const float huge_g[4] = {0.1f, 1e25f, 0.0f, 0.0f};
    float nc_p0[4], nc_p1[4], nc_m1[4], nc_v1[4];
    make_base_config(&noclip_cfg, 4u);
    noclip_cfg.max_grad_norm = 0.0f;
    CHECK(et_mod5_adamw_create(&noclip_cfg, &nc) == ET_MOD5_ADAMW_OK,
          "create noclip");
    CHECK(et_mod5_adamw_set_params(nc, init_p, 4u) == ET_MOD5_ADAMW_OK,
          "set noclip params");
    CHECK(et_mod5_adamw_get_params(nc, nc_p0, 4u) == ET_MOD5_ADAMW_OK,
          "snap noclip");
    CHECK(et_mod5_adamw_step(nc, huge_g, 4u, 8u, &info) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "g*g overflow rejected");
    CHECK(et_mod5_adamw_get_params(nc, nc_p1, 4u) == ET_MOD5_ADAMW_OK &&
              et_mod5_adamw_get_moments(nc, nc_m1, nc_v1, 4u) ==
                  ET_MOD5_ADAMW_OK &&
              memcmp(nc_p0, nc_p1, sizeof(nc_p0)) == 0 &&
              nc_m1[0] == 0.0f && nc_v1[0] == 0.0f &&
              et_mod5_adamw_get_step(nc) == 0u &&
              et_mod5_adamw_get_cumulative_tokens(nc) == 0u,
          "g*g overflow leaves state bitwise untouched");
    et_mod5_adamw_destroy(nc);
  }
  /* Wrong FP rounding mode rejected with DETERMINISM error */
  {
    const int saved_round = fegetround();
    CHECK(fesetround(FE_UPWARD) == 0, "set FE_UPWARD");
    CHECK(et_mod5_adamw_step(st, exact_g, 4u, 8u, &info) ==
              ET_MOD5_ADAMW_ERR_DETERMINISM,
          "non-RNE rounding rejected");
    CHECK(fesetround(saved_round) == 0, "restore rounding");
  }

  /* Verify live state after all rejected steps is bitwise unchanged */
  CHECK(et_mod5_adamw_get_params(st, p_after, 4u) == ET_MOD5_ADAMW_OK &&
            et_mod5_adamw_get_moments(st, m_after, v_after, 4u) ==
                ET_MOD5_ADAMW_OK &&
            memcmp(p_before, p_after, sizeof(p_before)) == 0 &&
            memcmp(m_before, m_after, sizeof(m_before)) == 0 &&
            memcmp(v_before, v_after, sizeof(v_before)) == 0 &&
            et_mod5_adamw_get_step(st) == 3u &&
            et_mod5_adamw_get_cumulative_tokens(st) == 24u,
        "all failed steps preserved bitwise state");

  et_mod5_adamw_destroy(st);
}

static void test_snapshot_roundtrip_and_negatives(const char *dir) {
  char snap_path[512];
  char bad_path[512];
  char sym_path[512];
  et_mod5_adamw_config_v1 cfg;
  et_mod5_adamw_state *s1 = NULL;
  et_mod5_adamw_state *s2 = NULL;
  const float init_p[6] = {0.5f, -1.25f, 2.0f, -0.75f, 0.125f, 1.5f};
  const float g1[6] = {0.3f, -0.4f, 0.5f, -0.2f, 0.1f, -0.6f};
  const float g2[6] = {-0.2f, 0.3f, -0.1f, 0.4f, -0.5f, 0.2f};
  const et_mod5_dataset_cursor_v1 cur = {3u, 64u, 2u};
  const uint64_t rng[4] = {UINT64_C(0x1111222233334444),
                           UINT64_C(0x5555666677778888),
                           UINT64_C(0x9999aaaabbbbcccc),
                           UINT64_C(0xddddeeeeffff0001)};
  struct stat st_meta;
  uint8_t *orig = NULL;
  size_t orig_len = 0u;
  char hash_before[65];
  char hash_after[65];

  snprintf(snap_path, sizeof(snap_path), "%s/valid.etsnap", dir);
  snprintf(bad_path, sizeof(bad_path), "%s/bad.etsnap", dir);
  snprintf(sym_path, sizeof(sym_path), "%s/sym.etsnap", dir);

  make_base_config(&cfg, 6u);
  CHECK(et_mod5_adamw_create(&cfg, &s1) == ET_MOD5_ADAMW_OK, "create s1");
  CHECK(et_mod5_adamw_create(&cfg, &s2) == ET_MOD5_ADAMW_OK, "create s2");
  CHECK(et_mod5_adamw_set_params(s1, init_p, 6u) == ET_MOD5_ADAMW_OK,
        "set s1 params");
  CHECK(et_mod5_adamw_set_cursor(s1, &cur) == ET_MOD5_ADAMW_OK, "set cursor");
  CHECK(et_mod5_adamw_set_rng(s1, rng) == ET_MOD5_ADAMW_OK, "set rng");
  CHECK(et_mod5_adamw_step(s1, g1, 6u, 16u, NULL) == ET_MOD5_ADAMW_OK,
        "step s1 #1");
  CHECK(et_mod5_adamw_step(s1, g2, 6u, 16u, NULL) == ET_MOD5_ADAMW_OK,
        "step s1 #2");
  {
    uint64_t rval = 0u;
    CHECK(et_mod5_adamw_rng_next_u64(s1, &rval) == ET_MOD5_ADAMW_OK,
          "advance rng");
  }

  CHECK(et_mod5_adamw_save_snapshot(s1, snap_path) == ET_MOD5_ADAMW_OK,
        "save valid snapshot");
  CHECK(stat(snap_path, &st_meta) == 0 && S_ISREG(st_meta.st_mode) &&
            (st_meta.st_mode & 0111) == 0 && (st_meta.st_mode & 0777) == 0600,
        "snapshot file is regular non-executable 0600");

  CHECK(et_mod5_adamw_load_snapshot(s2, snap_path) == ET_MOD5_ADAMW_OK,
        "load valid snapshot");
  CHECK(et_mod5_adamw_aot_state_sha256_hex(s1, hash_before,
                                           sizeof(hash_before)) ==
            ET_MOD5_ADAMW_OK,
        "hash s1");
  CHECK(et_mod5_adamw_aot_state_sha256_hex(s2, hash_after, sizeof(hash_after)) ==
            ET_MOD5_ADAMW_OK,
        "hash s2");
  CHECK(strcmp(hash_before, hash_after) == 0,
        "loaded snapshot state hash bitwise matches original");

  /* Symlink rejection on both load and save */
  CHECK(symlink(snap_path, sym_path) == 0, "create symlink");
  CHECK(et_mod5_adamw_load_snapshot(s2, sym_path) == ET_MOD5_ADAMW_ERR_IO,
        "O_NOFOLLOW rejects symlink load");
  CHECK(et_mod5_adamw_save_snapshot(s1, sym_path) == ET_MOD5_ADAMW_ERR_IO,
        "save refuses symlink destination");
  CHECK(et_mod5_adamw_load_snapshot(s2, dir) == ET_MOD5_ADAMW_ERR_IO,
        "load rejects directory");
  (void)unlink(sym_path);

  /* Read raw valid snapshot bytes for mutation tests */
  orig = read_raw_file(snap_path, &orig_len);
  CHECK(orig != NULL && orig_len > ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES + 32u,
        "read raw snapshot");

  /* 1. Truncated at multiple offsets */
  {
    const size_t cuts[] = {0u, 4u, 10u, 100u, orig_len - 1u};
    for (size_t i = 0u; i < sizeof(cuts) / sizeof(cuts[0]); ++i) {
      CHECK(write_raw_file(bad_path, orig, cuts[i]) == 0, "write truncated");
      CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
                ET_MOD5_ADAMW_ERR_TRUNCATED,
            "truncated snapshot rejected");
    }
  }

  /* 2. Trailing byte */
  {
    uint8_t *extra = (uint8_t *)malloc(orig_len + 1u);
    CHECK(extra != NULL, "alloc extra");
    memcpy(extra, orig, orig_len);
    extra[orig_len] = 0x5au;
    CHECK(write_raw_file(bad_path, extra, orig_len + 1u) == 0,
          "write trailing");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_TRUNCATED,
          "trailing byte rejected");
    free(extra);
  }

  /* 3. Bad magic */
  {
    uint8_t *mut = (uint8_t *)malloc(orig_len);
    memcpy(mut, orig, orig_len);
    mut[0] = 'X';
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write bad magic");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_BAD_MAGIC,
          "bad magic rejected");
    free(mut);
  }

  /* 4. Version mismatch (major=2, minor=0 and major=1, minor=1) */
  {
    uint8_t *mut = (uint8_t *)malloc(orig_len);
    memcpy(mut, orig, orig_len);
    mut[8] = 2u;
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write bad major");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_VERSION_MISMATCH,
          "major version mismatch rejected");

    memcpy(mut, orig, orig_len);
    mut[10] = 1u;
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write bad minor");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_VERSION_MISMATCH,
          "minor version mismatch rejected");
    free(mut);
  }

  /* 5. Corrupt checksum (header flip, payload flip, digest flip) */
  {
    const size_t flips[] = {24u, ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES + 22u,
                            orig_len - 1u};
    uint8_t *mut = (uint8_t *)malloc(orig_len);
    for (size_t i = 0u; i < sizeof(flips) / sizeof(flips[0]); ++i) {
      memcpy(mut, orig, orig_len);
      mut[flips[i]] ^= 0x01u;
      CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write corrupt");
      CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
                ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH,
            "corrupt checksum rejected");
    }
    free(mut);
  }

  /* 6. Non-finite param / m / v and negative v (re-signed) */
  {
    uint8_t *mut = (uint8_t *)malloc(orig_len);
    const size_t payload_off =
        ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES + strlen(cfg.tokenizer_id);
    const uint32_t qnan = 0x7fc00000u;
    const uint32_t neg_one = 0xbf800000u;
    /* NaN in params[0] */
    memcpy(mut, orig, orig_len);
    memcpy(mut + payload_off, &qnan, 4u);
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write nan param");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "NaN param in snapshot rejected");
    /* NaN in m[0] */
    memcpy(mut, orig, orig_len);
    memcpy(mut + payload_off + 6u * 4u, &qnan, 4u);
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write nan m");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "NaN m in snapshot rejected");
    /* Negative v[0] */
    memcpy(mut, orig, orig_len);
    memcpy(mut + payload_off + 12u * 4u, &neg_one, 4u);
    resign_snapshot(mut, orig_len);
    CHECK(write_raw_file(bad_path, mut, orig_len) == 0, "write neg v");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_NONFINITE,
          "negative v in snapshot rejected");
    free(mut);
  }

  /* 7. Config mismatch (different weight_decay, different tokenizer, different
   * param_count) */
  {
    et_mod5_adamw_config_v1 cfg_wd;
    et_mod5_adamw_state *s_wd = NULL;
    make_base_config(&cfg_wd, 6u);
    cfg_wd.weight_decay = 0.02f;
    CHECK(et_mod5_adamw_create(&cfg_wd, &s_wd) == ET_MOD5_ADAMW_OK,
          "create mismatch wd");
    CHECK(et_mod5_adamw_save_snapshot(s_wd, bad_path) == ET_MOD5_ADAMW_OK,
          "save mismatch wd");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH,
          "weight_decay mismatch rejected");
    et_mod5_adamw_destroy(s_wd);

    et_mod5_adamw_config_v1 cfg_tok;
    et_mod5_adamw_state *s_tok = NULL;
    make_base_config(&cfg_tok, 6u);
    cfg_tok.tokenizer_id = "other-tokenizer-v2";
    CHECK(et_mod5_adamw_create(&cfg_tok, &s_tok) == ET_MOD5_ADAMW_OK,
          "create mismatch tok");
    CHECK(et_mod5_adamw_save_snapshot(s_tok, bad_path) == ET_MOD5_ADAMW_OK,
          "save mismatch tok");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH,
          "tokenizer mismatch rejected");
    et_mod5_adamw_destroy(s_tok);

    et_mod5_adamw_config_v1 cfg_n;
    et_mod5_adamw_state *s_n = NULL;
    make_base_config(&cfg_n, 4u);
    CHECK(et_mod5_adamw_create(&cfg_n, &s_n) == ET_MOD5_ADAMW_OK,
          "create mismatch n");
    CHECK(et_mod5_adamw_save_snapshot(s_n, bad_path) == ET_MOD5_ADAMW_OK,
          "save mismatch n");
    CHECK(et_mod5_adamw_load_snapshot(s2, bad_path) ==
              ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH,
          "param_count mismatch rejected");
    et_mod5_adamw_destroy(s_n);
  }

  /* Verify live state s2 is STILL bitwise identical to hash_before after EVERY
   * single rejected load above! */
  CHECK(et_mod5_adamw_aot_state_sha256_hex(s2, hash_after, sizeof(hash_after)) ==
            ET_MOD5_ADAMW_OK &&
            strcmp(hash_before, hash_after) == 0,
        "all rejected snapshot loads preserved live state bitwise");

  /* D1 shard write/read + corruption test */
  {
    char shard_path[512];
    const int64_t toks[9] = {1, 4, 1, 5, 9, 2, 6, 5, 3};
    int64_t loaded[16];
    int64_t count = 0;
    uint8_t *sraw = NULL;
    size_t slen = 0u;
    snprintf(shard_path, sizeof(shard_path), "%s/shard0.ets", dir);
    CHECK(et_mod5_adamw_aot_write_d1_shard(shard_path, 0, 16,
                                           "mod5-native-token-v1", toks,
                                           9) == ET_MOD5_ADAMW_OK,
          "write d1 shard");
    CHECK(et_mod5_adamw_aot_read_d1_shard(shard_path, 0, 16,
                                          "mod5-native-token-v1", loaded, 16,
                                          &count) == ET_MOD5_ADAMW_OK &&
              count == 9 && memcmp(toks, loaded, sizeof(toks)) == 0,
          "read d1 shard exact");
    sraw = read_raw_file(shard_path, &slen);
    CHECK(sraw != NULL && slen > 80u, "read raw shard");
    sraw[slen - 33u] ^= 0x01u;
    CHECK(write_raw_file(shard_path, sraw, slen) == 0, "corrupt shard");
    CHECK(et_mod5_adamw_aot_read_d1_shard(shard_path, 0, 16,
                                          "mod5-native-token-v1", loaded, 16,
                                          &count) ==
              ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH,
          "corrupt d1 shard rejected");
    free(sraw);
    (void)unlink(shard_path);
  }

  free(orig);
  (void)unlink(snap_path);
  (void)unlink(bad_path);
  et_mod5_adamw_destroy(s1);
  et_mod5_adamw_destroy(s2);
}

int main(void) {
  char dir_template[] = "/tmp/et_mod5_adamw_native_XXXXXX";
  char *dir = mkdtemp(dir_template);
  if (dir == NULL) {
    perror("mkdtemp");
    return 1;
  }

  test_lr_schedules();
  test_invalid_hyperparams();
  test_step_clipping_and_atomic_rollback();
  test_snapshot_roundtrip_and_negatives(dir);

  (void)rmdir(dir);
  if (g_failures != 0) {
    fprintf(stderr, "MOD5 AdamW native tests: %d failure(s)\n", g_failures);
    return 1;
  }
  printf("MOD5 AdamW native tests: PASS\n");
  return 0;
}
