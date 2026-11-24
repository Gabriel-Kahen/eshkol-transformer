#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/e2e_lifecycle_abi.h"
#include "e2e_lifecycle_eshkol_bridge.h"

#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *msg) {
  fprintf(stderr, "E2E NATIVE FAIL: %s\n", msg);
  exit(1);
}

static void require_true(int cond, const char *msg) {
  if (!cond) {
    die(msg);
  }
}

static void flip_byte_in_file(const char *src, const char *dst, size_t offset) {
  int fd_in = open(src, O_RDONLY | O_NOFOLLOW);
  int fd_out;
  struct stat st;
  uint8_t *buf;
  size_t len;
  ssize_t r;
  ssize_t w;

  require_true(fd_in >= 0, "open src for corruption");
  require_true(fstat(fd_in, &st) == 0 && st.st_size > (off_t)offset,
               "fstat src");
  len = (size_t)st.st_size;
  buf = (uint8_t *)calloc(1u, len);
  require_true(buf != NULL, "calloc tamper buf");
  r = read(fd_in, buf, len);
  (void)close(fd_in);
  require_true(r == (ssize_t)len, "read src");

  buf[offset] ^= 0x5Au;

  (void)unlink(dst);
  fd_out = open(dst, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  require_true(fd_out >= 0, "open dst");
  w = write(fd_out, buf, len);
  (void)close(fd_out);
  free(buf);
  require_true(w == (ssize_t)len, "write tampered dst");
}

static void test_stage3_fd_gradients(void) {
  double max_err = 1.0;
  et_e2e_status st = et_e2e_stage3_check_fd_gradients(&max_err);
  require_true(st == ET_E2E_OK, "Stage 3 FD gradient check failed");
  require_true(isfinite(max_err) && max_err <= 2.0e-3,
               "Stage 3 FD max error out of tolerance");
  printf("E2E_NATIVE_FD_GRADIENTS max_abs_err=%.9g PASS\n", max_err);
}

static void test_full_pipeline_and_determinism(const char *dir_a,
                                               const char *dir_b) {
  et_e2e_stage_metrics_v1 ma;
  et_e2e_stage_metrics_v1 mb;
  memset(&ma, 0, sizeof(ma));
  memset(&mb, 0, sizeof(mb));

  require_true(et_e2e_run_full_in_process(dir_a, &ma) == ET_E2E_OK,
               "run A failed");
  require_true(et_e2e_run_full_in_process(dir_b, &mb) == ET_E2E_OK,
               "run B failed");

  /* Exact deterministic parity across two fresh directories. */
  require_true(memcmp(&ma, &mb, sizeof(ma)) == 0,
               "metrics differ across two identical fresh runs");

  /* Stage 1 assertions */
  require_true(ma.s1_input_docs == 11u && ma.s1_kept_docs == 6u,
               "stage1 kept docs");
  require_true(ma.s1_dropped_exact == 1u && ma.s1_dropped_near == 1u &&
                   ma.s1_dropped_utf8 == 1u && ma.s1_dropped_length == 1u &&
                   ma.s1_dropped_quality == 1u,
               "stage1 drop breakdown");

  /* Stage 2 assertions */
  require_true(ma.s2_vocab_size > 256u && ma.s2_merge_count > 0u,
               "stage2 merges");
  require_true(ma.s2_lossless_verified == 1u && ma.s2_reload_sha_match == 1u,
               "stage2 lossless & reload");
  require_true(ma.s2_compression_ratio > 1.5, "stage2 compression > 1.5x");

  /* Stage 3 assertions */
  require_true(ma.s3_param_count > 0u && (ma.s3_param_count % 32u) == 0u,
               "stage3 param alignment");
  require_true(ma.s3_loss_step8 < ma.s3_loss_step4 &&
                   ma.s3_loss_step4 < ma.s3_loss_step1,
               "stage3 strict loss decrease");
  require_true(ma.s3_weights_bitwise_equal == 1u &&
                   ma.s3_snapshot_sha_equal == 1u &&
                   ma.s3_corrupt_rejected == 1u,
               "stage3 bitwise resume parity");
  require_true(
      memcmp(ma.s3_full_snap_sha256, ma.s3_resumed_snap_sha256, 32u) == 0,
      "stage3 snapshot sha equality");

  /* Stage 4 assertions */
  require_true(ma.s4_rank == 4u && ma.s4_final_loss < ma.s4_initial_loss,
               "stage4 lora loss decrease");
  require_true(ma.s4_base_frozen_equal == 1u && ma.s4_reload_equal == 1u,
               "stage4 frozen base & reload");
  require_true(ma.s4_merge_max_abs_err <= 1.0e-6 &&
                   ma.s4_unmerge_max_abs_err <= 1.0e-6,
               "stage4 merge/unmerge parity");

  /* Stage 5 assertions */
  require_true(ma.s5_final_loss < ma.s5_initial_loss &&
                   ma.s5_final_margin > ma.s5_initial_margin &&
                   ma.s5_final_margin > 0.0 && ma.s5_final_accuracy == 1.0 &&
                   ma.s5_ref_frozen_equal == 1u,
               "stage5 dpo alignment invariants");

  /* Stage 6 assertions */
  require_true(ma.s6_contam_status == ET_EVAL5_ERR_CONTAMINATED &&
                   ma.s6_contam_manifest_absent == 1u &&
                   ma.s6_contam_docs_found == 1u &&
                   ma.s6_clean_contam_docs == 0u &&
                   ma.s6_fresh_reload_verified == 1u &&
                   isfinite(ma.s6_lm_mean_nll) && ma.s6_lm_perplexity > 1.0 &&
                   isfinite(ma.s6_mc_accuracy),
               "stage6 eval5 contamination gate & clean metrics");

  /* Stage 7 assertions */
  require_true(ma.s7_greedy_exact_match == 1u &&
                   ma.s7_stoch_replay_match == 1u && ma.s7_eos_hit == 1u &&
                   ma.s7_eos_emitted == 2u,
               "stage7 speculative decode invariants");

  /* Stage 8 assertions */
  require_true(ma.s8_weight_count == ma.s3_param_count &&
                   ma.s8_f16_max_abs_err <= 1.0e-3 &&
                   ma.s8_bf16_max_abs_err <= 1.0e-2 &&
                   ma.s8_q8_0_rel_l2_err <= 1.0e-2,
               "stage8 io5 quant error bounds");

  printf("E2E_NATIVE_FULL_PIPELINE kept=%llu/%llu vocab=%u ratio=%.4f "
         "s3_loss=%.6f->%.6f lora_loss=%.6f->%.6f dpo_loss=%.6f->%.6f "
         "dpo_margin=%.6f->%.6f ppl=%.4f f16=%.3e bf16=%.3e q8_rel=%.3e PASS\n",
         (unsigned long long)ma.s1_kept_docs,
         (unsigned long long)ma.s1_input_docs, ma.s2_vocab_size,
         ma.s2_compression_ratio, ma.s3_loss_step1, ma.s3_loss_step8,
         ma.s4_initial_loss, ma.s4_final_loss, ma.s5_initial_loss,
         ma.s5_final_loss, ma.s5_initial_margin, ma.s5_final_margin,
         ma.s6_lm_perplexity, ma.s8_f16_max_abs_err, ma.s8_bf16_max_abs_err,
         ma.s8_q8_0_rel_l2_err);
}

static void test_negative_propagation(const char *valid_dir,
                                      const char *neg_dir) {
  et_e2e_aot_ctx *ctx = NULL;
  char hex[65];
  char p1[1024];
  char p2[1024];
  et_e2e_stage_metrics_v1 m;

  /* 1. NULL and empty argument rejection. */
  require_true(et_e2e_aot_create(NULL) == ET_E2E_ERR_ARGUMENT,
               "create NULL out");
  require_true(et_e2e_run_full_in_process(NULL, &m) == ET_E2E_ERR_ARGUMENT,
               "full NULL dir");
  require_true(et_e2e_run_full_in_process("", &m) == ET_E2E_ERR_ARGUMENT,
               "full empty dir");
  require_true(et_e2e_run_full_in_process(valid_dir, NULL) ==
                   ET_E2E_ERR_ARGUMENT,
               "full NULL metrics");
  require_true(et_e2e_stage3_check_fd_gradients(NULL) == ET_E2E_ERR_ARGUMENT,
               "fd NULL out");
  require_true(et_e2e_aot_get_i64(NULL, ET_E2E_M_S1_KEPT_DOCS) == -1,
               "get_i64 NULL");
  require_true(isnan(et_e2e_aot_get_f64(NULL, ET_E2E_F_S3_LOSS_STEP1)),
               "get_f64 NULL");
  require_true(et_e2e_aot_read_byte(NULL, 0) == -1, "read_byte NULL");
  require_true(et_e2e_aot_read_byte("abc", -1) == -1, "read_byte neg");

  require_true(et_e2e_aot_create(&ctx) == ET_E2E_OK, "create ctx for neg");
  require_true(et_e2e_aot_get_i64(ctx, 99999) == -1, "invalid i64 metric");
  require_true(isnan(et_e2e_aot_get_f64(ctx, 99999)), "invalid f64 metric");
  require_true(et_e2e_aot_get_sha256_hex(ctx, 999, hex, 65) ==
                   ET_E2E_ERR_ARGUMENT,
               "invalid digest id");
  require_true(et_e2e_aot_get_sha256_hex(ctx, ET_E2E_DIGEST_S1_REPORT, hex,
                                         16) == ET_E2E_ERR_ARGUMENT,
               "short digest cap");

  /* 2. Missing artifacts in empty neg_dir fail cleanly without leaks. */
  require_true(et_e2e_aot_run_stage2_train_save(ctx, neg_dir) != ET_E2E_OK,
               "stage2 missing corpus must fail");
  require_true(et_e2e_aot_run_stage2_reload_encode(ctx, neg_dir) != ET_E2E_OK,
               "stage2 reload missing must fail");
  require_true(et_e2e_aot_run_stage3_pretrain_resume(ctx, neg_dir) != ET_E2E_OK,
               "stage3 resume missing must fail");
  require_true(et_e2e_aot_run_stage4_lora_reload(ctx, neg_dir) != ET_E2E_OK,
               "stage4 reload missing must fail");
  require_true(et_e2e_aot_run_stage6_verify_fresh(ctx, neg_dir) != ET_E2E_OK,
               "stage6 verify missing must fail");

  /* 3. Tampered Stage 1 corpus bundle is rejected by checksum. */
  snprintf(p1, sizeof(p1), "%s/stage1_clean_corpus.bin", valid_dir);
  snprintf(p2, sizeof(p2), "%s/stage1_clean_corpus.bin", neg_dir);
  flip_byte_in_file(p1, p2, 16u);
  require_true(et_e2e_aot_run_stage2_train_save(ctx, neg_dir) ==
                   ET_E2E_ERR_STAGE1_DATA4,
               "tampered corpus bundle must fail");

  /* 4. Tampered Stage 6 manifest is rejected on fresh reload. */
  snprintf(p1, sizeof(p1), "%s/stage6_clean.manifest", valid_dir);
  snprintf(p2, sizeof(p2), "%s/stage6_clean.manifest", neg_dir);
  flip_byte_in_file(p1, p2, 24u);
  require_true(et_e2e_aot_run_stage6_verify_fresh(ctx, neg_dir) ==
                   ET_E2E_ERR_STAGE6_EVAL5,
               "tampered eval5 manifest must fail");

  /* 5. Symlinked Stage 2 tokenizer is rejected (O_NOFOLLOW). */
  snprintf(p1, sizeof(p1), "%s/stage2_tokenizer.etbpe", valid_dir);
  snprintf(p2, sizeof(p2), "%s/stage2_tokenizer.etbpe", neg_dir);
  require_true(symlink(p1, p2) == 0, "create symlink for neg");
  require_true(et_e2e_aot_run_stage2_reload_encode(ctx, neg_dir) != ET_E2E_OK,
               "symlinked tokenizer must fail");

  et_e2e_aot_destroy(ctx);
  printf("E2E_NATIVE_NEGATIVE_TESTS PASS\n");
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: test_e2e_lifecycle_native <dir_a> <dir_b> <neg>\n");
    return 2;
  }
  test_stage3_fd_gradients();
  test_full_pipeline_and_determinism(argv[1], argv[2]);
  test_negative_propagation(argv[1], argv[3]);
  return 0;
}
