#define _POSIX_C_SOURCE 200809L

#include "wave4_composed_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#pragma STDC FP_CONTRACT OFF

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(stderr, "%s:%d: FAIL: %s\n", __FILE__, __LINE__, #expr);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static float sample_lcg(uint32_t *state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return (float)(*state >> 16u) * 0x1p-15f - 1.0f;
}

static void populate_rank(et_wave4_composed_aot_context *ctx, int64_t layers,
                          int64_t rank) {
  const int64_t p = et_wave4_composed_aot_parameter_count(ctx);
  const int64_t x = et_wave4_composed_aot_activation_count(ctx);
  CHECK(p == layers * 104 && x == 8);

  uint32_t rng = 241u;
  for (int64_t i = 0; i < p; ++i) {
    const double val = rank == 0 ? 0.20 * (double)sample_lcg(&rng) : -7.0;
    CHECK(et_wave4_composed_aot_set_master(ctx, i, val) ==
          ET_WAVE4_COMPOSED_OK);
  }
  if (rank == 0) {
    for (int64_t l = 0; l < layers; ++l) {
      for (int64_t i = 0; i < 4; ++i) {
        CHECK(et_wave4_composed_aot_set_master(ctx, l * 104 + i, 1.0) ==
              ET_WAVE4_COMPOSED_OK);
        CHECK(et_wave4_composed_aot_set_master(ctx, l * 104 + 52 + i, 1.0) ==
              ET_WAVE4_COMPOSED_OK);
      }
    }
  }

  rng = 61u;
  for (int64_t i = 0; i < x; ++i) {
    const float common = sample_lcg(&rng);
    const float offset = sample_lcg(&rng);
    const float in = 0.75f * common + (float)rank * 0.025f * offset;
    const float tgt = 0.70f * in - 0.20f + 0.15f * sample_lcg(&rng);
    CHECK(et_wave4_composed_aot_set_input(ctx, i, (double)in) ==
          ET_WAVE4_COMPOSED_OK);
    CHECK(et_wave4_composed_aot_set_target(ctx, i, (double)tgt) ==
          ET_WAVE4_COMPOSED_OK);
  }
}

static void test_null_and_invalid_arguments(void) {
  et_wave4_composed_aot_destroy(NULL);

  /* Null out_ctx */
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125,
                                     412775, 0.22, 0.5, 64.0, 2.0, 0.5, 32.0,
                                     128.0, 5, NULL) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);

  /* Null ctx on all mutators / operations / accessors */
  CHECK(et_wave4_composed_aot_set_master(NULL, 0, 1.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_input(NULL, 0, 1.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_target(NULL, 0, 1.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_overflow_inject(NULL, 0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_activation_limit(NULL, 0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_bootstrap(NULL) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_step(NULL, 0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_evaluate(NULL, "/tmp/x") ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_verify_manifest(NULL, "/tmp/x") ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_parameter_count(NULL) == -1);
  CHECK(et_wave4_composed_aot_activation_count(NULL) == -1);
  CHECK(et_wave4_composed_aot_planned_activation_peak_bytes(NULL) == -1);
  CHECK(et_wave4_composed_aot_planned_workspace_bytes(NULL) == -1);
  CHECK(et_wave4_composed_aot_narrow_parameter_bytes(NULL) == -1);
  CHECK(et_wave4_composed_aot_last_status(NULL) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(isnan(et_wave4_composed_aot_last_loss(NULL)));
  CHECK(et_wave4_composed_aot_last_local_overflow(NULL) == -1);
  CHECK(et_wave4_composed_aot_live_activation_peak_bytes(NULL) == -1);
  CHECK(et_wave4_composed_aot_workspace_bytes(NULL) == -1);
  CHECK(isnan(et_wave4_composed_aot_scaler_scale(NULL)));
  CHECK(et_wave4_composed_aot_scaler_clean_steps(NULL) == -1);
  CHECK(isnan(et_wave4_composed_aot_get_master(NULL, 0)));
  CHECK(et_wave4_composed_aot_master_bits(NULL, 0) == -1);
  CHECK(et_wave4_composed_aot_param_digest_byte(NULL, 0) == -1);
  CHECK(isnan(et_wave4_composed_aot_eval_mean_nll(NULL)));
  CHECK(isnan(et_wave4_composed_aot_eval_perplexity(NULL)));
  CHECK(et_wave4_composed_aot_eval_contaminated_documents(NULL) == -1);
  CHECK(et_wave4_composed_aot_manifest_digest_byte(NULL, 0) == -1);
  CHECK(et_wave4_composed_aot_read_byte(NULL, 0) == -1);
  CHECK(et_wave4_composed_aot_read_byte("abc", -1) == -1);
  CHECK(et_wave4_composed_aot_read_byte("abc", 1) == 'b');

  char hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  CHECK(et_wave4_composed_aot_param_digest_hex(NULL, hex, sizeof(hex)) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_manifest_digest_hex(NULL, hex, sizeof(hex)) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);

  /* Invalid layer, dtype, policy, checkpoint_every, rank/world, sock_dir,
   * scaler params; out_ctx must stay unchanged on every failure. */
  et_wave4_composed_aot_context *sentinel =
      (et_wave4_composed_aot_context *)(uintptr_t)0xdeadbeefu;
  et_wave4_composed_aot_context *out = sentinel;

  /* Invalid layers: 0, 2, 5, 16 */
  const int64_t bad_layers[] = {0, 2, 5, 16};
  for (size_t i = 0; i < sizeof(bad_layers) / sizeof(bad_layers[0]); ++i) {
    CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, bad_layers[i],
                                       ET_WAVE4_COMPOSED_F16,
                                       ET_AC4_POLICY_NONE, 0, 0.125, 0.03125,
                                       1, 0.22, 0.5, 64.0, 2.0, 0.5, 32.0,
                                       128.0, 5, &out) ==
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
    CHECK(out == sentinel);
  }

  /* Invalid dtypes: 0, 3 */
  const int64_t bad_dtypes[] = {0, 3};
  for (size_t i = 0; i < sizeof(bad_dtypes) / sizeof(bad_dtypes[0]); ++i) {
    CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, bad_dtypes[i],
                                       ET_AC4_POLICY_NONE, 0, 0.125, 0.03125,
                                       1, 0.22, 0.5, 64.0, 2.0, 0.5, 32.0,
                                       128.0, 5, &out) ==
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
    CHECK(out == sentinel);
  }

  /* Invalid policies: -1, 4; invalid checkpoint_every */
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     -1, 0, 0.125, 0.03125, 1, 0.22, 0.5, 64.0,
                                     2.0, 0.5, 32.0, 128.0, 5, &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     4, 0, 0.125, 0.03125, 1, 0.22, 0.5, 64.0,
                                     2.0, 0.5, 32.0, 128.0, 5, &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 2, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_EVERY_K, 0, 0.125, 0.03125,
                                     1, 0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0,
                                     5, &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);

  /* Invalid rank/world/timeout/sock_dir */
  CHECK(et_wave4_composed_aot_create(NULL, 0, 0, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 1, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 2, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create("", 0, 2, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  char huge_dir[256];
  memset(huge_dir, 'a', sizeof(huge_dir) - 1u);
  huge_dir[sizeof(huge_dir) - 1u] = '\0';
  CHECK(et_wave4_composed_aot_create(huge_dir, 0, 2, 5000, 4,
                                     ET_WAVE4_COMPOSED_F16, ET_AC4_POLICY_NONE,
                                     0, 0.125, 0.03125, 1, 0.22, 0.5, 64.0, 2.0,
                                     0.5, 32.0, 128.0, 5, &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 0, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);

  /* Invalid scaler config */
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 0.0, 128.0, 5,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, 64.0, 2.0, 0.5, 32.0, 128.0, 0,
                                     &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
                                     ET_AC4_POLICY_NONE, 0, 0.125, 0.03125, 1,
                                     0.22, 0.5, (double)NAN, 2.0, 0.5, 32.0,
                                     128.0, 5, &out) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(out == sentinel);
}

static void test_single_rank_faults_and_short_buffers(void) {
  et_wave4_composed_aot_context *ctx = NULL;
  CHECK(et_wave4_composed_aot_create(NULL, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_BF16,
                                     ET_AC4_POLICY_SQRT_SEGMENTS, 0, 0.125,
                                     0.03125, 412775, 0.22, 0.5, 64.0, 2.0, 0.5,
                                     32.0, 128.0, 5, &ctx) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(ctx != NULL);
  CHECK(et_wave4_composed_aot_planned_activation_peak_bytes(ctx) == 576);
  CHECK(et_wave4_composed_aot_planned_workspace_bytes(ctx) == 6464);
  CHECK(et_wave4_composed_aot_narrow_parameter_bytes(ctx) == 832);

  /* Must reject step/evaluate before bootstrap */
  CHECK(et_wave4_composed_aot_step(ctx, 0) ==
        ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED);
  CHECK(et_wave4_composed_aot_evaluate(ctx, "/tmp/unused.eval5") ==
        ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED);
  CHECK(et_wave4_composed_aot_last_status(ctx) ==
        ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED);

  /* OOB indices and invalid toggles */
  CHECK(et_wave4_composed_aot_set_master(ctx, -1, 0.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_master(ctx, 416, 0.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_input(ctx, -1, 0.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_input(ctx, 8, 0.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_target(ctx, 8, 0.0) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_overflow_inject(ctx, -1) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_overflow_inject(ctx, 2) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_set_activation_limit(ctx, -1) ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_param_digest_byte(ctx, -1) == -1);
  CHECK(et_wave4_composed_aot_param_digest_byte(ctx, 32) == -1);
  CHECK(et_wave4_composed_aot_manifest_digest_byte(ctx, 0) == -1);

  /* Short digest buffer must fail without modifying canary bytes */
  char short_buf[64];
  memset(short_buf, 0x5a, sizeof(short_buf));
  char expected_canary[64];
  memset(expected_canary, 0x5a, sizeof(expected_canary));
  CHECK(et_wave4_composed_aot_param_digest_hex(ctx, short_buf, 64) ==
        ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL);
  CHECK(memcmp(short_buf, expected_canary, sizeof(short_buf)) == 0);
  CHECK(et_wave4_composed_aot_param_digest_hex(ctx, short_buf, 0) ==
        ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL);
  CHECK(memcmp(short_buf, expected_canary, sizeof(short_buf)) == 0);

  populate_rank(ctx, 4, 0);
  CHECK(et_wave4_composed_aot_bootstrap(ctx) == ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_scaler_scale(ctx) == 64.0);
  CHECK(et_wave4_composed_aot_scaler_clean_steps(ctx) == 0);

  /* Activation limit below peak rejects cleanly */
  CHECK(et_wave4_composed_aot_set_activation_limit(ctx, 575) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_step(ctx, 0) ==
        ET_WAVE4_COMPOSED_ACTIVATION_LIMIT);
  CHECK(et_wave4_composed_aot_set_activation_limit(ctx, 576) ==
        ET_WAVE4_COMPOSED_OK);

  /* Clean step 0 and 1 */
  CHECK(et_wave4_composed_aot_step(ctx, 0) == ET_WAVE4_COMPOSED_OK);
  const double loss0 = et_wave4_composed_aot_last_loss(ctx);
  CHECK(et_wave4_composed_aot_step(ctx, 1) == ET_WAVE4_COMPOSED_OK);
  const double loss1 = et_wave4_composed_aot_last_loss(ctx);
  CHECK(loss1 < loss0);

  /* Step 2: overflow backoff (64 -> 32) preserves weights bitwise */
  char pre_fault_hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  CHECK(et_wave4_composed_aot_param_digest_hex(ctx, pre_fault_hex,
                                               sizeof(pre_fault_hex)) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_set_overflow_inject(ctx, 1) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_step(ctx, 2) ==
        ET_WAVE4_COMPOSED_OVERFLOW_BACKOFF);
  CHECK(et_wave4_composed_aot_last_local_overflow(ctx) == 1);
  CHECK(et_wave4_composed_aot_scaler_scale(ctx) == 32.0);
  CHECK(et_wave4_composed_aot_scaler_clean_steps(ctx) == 0);

  char post_backoff_hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  CHECK(et_wave4_composed_aot_param_digest_hex(ctx, post_backoff_hex,
                                               sizeof(post_backoff_hex)) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(strcmp(pre_fault_hex, post_backoff_hex) == 0);

  /* Retry at min_scale=32 triggers UNDERFLOW_FAULT leaving weights, scaler,
   * and last_loss completely untouched */
  const double backoff_loss = et_wave4_composed_aot_last_loss(ctx);
  CHECK(et_wave4_composed_aot_step(ctx, 2) ==
        ET_WAVE4_COMPOSED_UNDERFLOW_FAULT);
  CHECK(et_wave4_composed_aot_last_status(ctx) ==
        ET_WAVE4_COMPOSED_UNDERFLOW_FAULT);
  CHECK(et_wave4_composed_aot_scaler_scale(ctx) == 32.0);
  CHECK(et_wave4_composed_aot_scaler_clean_steps(ctx) == 0);
  CHECK(memcmp(&backoff_loss, &(double){et_wave4_composed_aot_last_loss(ctx)},
               sizeof(double)) == 0);

  char post_underflow_hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  CHECK(et_wave4_composed_aot_param_digest_hex(ctx, post_underflow_hex,
                                               sizeof(post_underflow_hex)) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(strcmp(pre_fault_hex, post_underflow_hex) == 0);

  /* Recover cleanly on step 3 */
  CHECK(et_wave4_composed_aot_set_overflow_inject(ctx, 0) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_step(ctx, 3) == ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_last_loss(ctx) < loss1);

  /* Evaluate and verify manifest + tamper detection + short manifest buffer */
  char dir[] = "/tmp/eshkol-w4-bridge.XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char path[128];
  CHECK(snprintf(path, sizeof(path), "%s/fixture.eval5", dir) > 0);
  CHECK(et_wave4_composed_aot_evaluate(ctx, "") ==
        ET_WAVE4_COMPOSED_INVALID_ARGUMENT);
  CHECK(et_wave4_composed_aot_evaluate(ctx, path) == ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_verify_manifest(ctx, path) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(et_wave4_composed_aot_eval_contaminated_documents(ctx) == 0);
  CHECK(et_wave4_composed_aot_eval_perplexity(ctx) > 1.0);
  CHECK(et_wave4_composed_aot_manifest_digest_hex(ctx, short_buf, 64) ==
        ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL);
  char mhex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  CHECK(et_wave4_composed_aot_manifest_digest_hex(ctx, mhex, sizeof(mhex)) ==
        ET_WAVE4_COMPOSED_OK);
  CHECK(strlen(mhex) == 64u);

  /* Tamper with manifest and confirm checksum rejection */
  FILE *f = fopen(path, "r+");
  CHECK(f != NULL);
  char buf[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  const size_t n = fread(buf, 1u, sizeof(buf) - 1u, f);
  CHECK(n > 0u);
  buf[n] = '\0';
  char *hit = strstr(buf, "split=held-out");
  CHECK(hit != NULL);
  CHECK(fseek(f, (long)(hit - buf) + (long)strlen("split="), SEEK_SET) == 0);
  CHECK(fputc('w', f) == 'w' && fclose(f) == 0);
  CHECK(et_wave4_composed_aot_verify_manifest(ctx, path) ==
        (int32_t)(ET_WAVE4_COMPOSED_EVAL5_ERROR_BASE - ET_EVAL5_ERR_CHECKSUM));

  CHECK(unlink(path) == 0);
  CHECK(rmdir(dir) == 0);
  et_wave4_composed_aot_destroy(ctx);
}

static void test_two_rank_unix_socket_all_policies(void) {
  const et_ac4_policy policies[] = {ET_AC4_POLICY_NONE,
                                    ET_AC4_POLICY_EVERY_LAYER,
                                    ET_AC4_POLICY_SQRT_SEGMENTS,
                                    ET_AC4_POLICY_EVERY_K};
  char baseline_hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES] = {0};

  for (size_t pi = 0; pi < sizeof(policies) / sizeof(policies[0]); ++pi) {
    char sock_dir[] = "/tmp/eshkol-w4-sock.XXXXXX";
    CHECK(mkdtemp(sock_dir) != NULL);

    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
      et_wave4_composed_aot_context *c1 = NULL;
      const int64_t k = policies[pi] == ET_AC4_POLICY_EVERY_K ? 2 : 0;
      CHECK(et_wave4_composed_aot_create(sock_dir, 1, 2, 10000, 4,
                                         ET_WAVE4_COMPOSED_F16, policies[pi], k,
                                         0.125, 0.03125, 412776, 0.22, 0.5,
                                         64.0, 2.0, 0.5, 32.0, 128.0, 5,
                                         &c1) == ET_WAVE4_COMPOSED_OK);
      populate_rank(c1, 4, 1);
      CHECK(et_wave4_composed_aot_bootstrap(c1) == ET_WAVE4_COMPOSED_OK);
      CHECK(et_wave4_composed_aot_step(c1, 0) == ET_WAVE4_COMPOSED_OK);
      CHECK(et_wave4_composed_aot_set_overflow_inject(c1, 1) ==
            ET_WAVE4_COMPOSED_OK);
      CHECK(et_wave4_composed_aot_step(c1, 2) ==
            ET_WAVE4_COMPOSED_OVERFLOW_BACKOFF);
      CHECK(et_wave4_composed_aot_step(c1, 2) ==
            ET_WAVE4_COMPOSED_UNDERFLOW_FAULT);
      CHECK(et_wave4_composed_aot_set_overflow_inject(c1, 0) ==
            ET_WAVE4_COMPOSED_OK);
      CHECK(et_wave4_composed_aot_step(c1, 3) == ET_WAVE4_COMPOSED_OK);
      et_wave4_composed_aot_destroy(c1);
      exit(EXIT_SUCCESS);
    }

    et_wave4_composed_aot_context *c0 = NULL;
    const int64_t k = policies[pi] == ET_AC4_POLICY_EVERY_K ? 2 : 0;
    CHECK(et_wave4_composed_aot_create(sock_dir, 0, 2, 10000, 4,
                                       ET_WAVE4_COMPOSED_F16, policies[pi], k,
                                       0.125, 0.03125, 412775, 0.22, 0.5, 64.0,
                                       2.0, 0.5, 32.0, 128.0, 5, &c0) ==
          ET_WAVE4_COMPOSED_OK);
    populate_rank(c0, 4, 0);
    CHECK(et_wave4_composed_aot_bootstrap(c0) == ET_WAVE4_COMPOSED_OK);
    CHECK(et_wave4_composed_aot_step(c0, 0) == ET_WAVE4_COMPOSED_OK);
    CHECK(et_wave4_composed_aot_step(c0, 2) ==
          ET_WAVE4_COMPOSED_OVERFLOW_BACKOFF);
    CHECK(et_wave4_composed_aot_last_local_overflow(c0) == 0);
    CHECK(et_wave4_composed_aot_scaler_scale(c0) == 32.0);
    CHECK(et_wave4_composed_aot_step(c0, 2) ==
          ET_WAVE4_COMPOSED_UNDERFLOW_FAULT);
    CHECK(et_wave4_composed_aot_step(c0, 3) == ET_WAVE4_COMPOSED_OK);

    char hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
    CHECK(et_wave4_composed_aot_param_digest_hex(c0, hex, sizeof(hex)) ==
          ET_WAVE4_COMPOSED_OK);
    if (pi == 0)
      memcpy(baseline_hex, hex, sizeof(hex));
    else
      CHECK(strcmp(baseline_hex, hex) == 0);

    et_wave4_composed_aot_destroy(c0);

    int status = 0;
    CHECK(waitpid(child, &status, 0) == child && WIFEXITED(status) &&
          WEXITSTATUS(status) == 0);
    CHECK(rmdir(sock_dir) == 0);
  }
}

int main(void) {
  test_null_and_invalid_arguments();
  test_single_rank_faults_and_short_buffers();
  test_two_rank_unix_socket_all_policies();
  puts("Wave 4/5 composed AOT bridge C11 tests: PASS");
  return EXIT_SUCCESS;
}
