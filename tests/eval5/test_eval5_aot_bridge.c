#define _POSIX_C_SOURCE 200809L

#include "eval5_eshkol_bridge.h"

#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static void make_path(char *out, size_t cap, const char *dir,
                      const char *name) {
  int n = snprintf(out, cap, "%s/%s", dir, name);
  CHECK(n > 0 && (size_t)n < cap);
}

static void test_null_and_bounds_negative(void) {
  et_eval5_aot_corpus *corpus = NULL;
  et_eval5_aot_mc_batch *mc = NULL;
  et_eval5_aot_manifest *manifest = NULL;
  int64_t count = -1;
  int64_t indices[4] = {99, 99, 99, 99};
  int64_t summary[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN] = {99, 99, 99, 99, 99, 99};
  double rate = 99.0;
  double lm[ET_EVAL5_AOT_LM_METRICS_LEN] = {99.0, 99.0, 99.0, 99.0};
  uint8_t digest[ET_EVAL5_SHA256_BYTES] = {0};
  int32_t eq = -1;

  CHECK(et_eval5_aot_corpus_create(NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_mc_create(NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_create(NULL) == ET_EVAL5_ERR_ARGUMENT);

  et_eval5_aot_corpus_destroy(NULL);
  et_eval5_aot_mc_destroy(NULL);
  et_eval5_aot_manifest_destroy(NULL);

  CHECK(et_eval5_aot_corpus_create(&corpus) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_doc_count(NULL, &count) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_doc_count(corpus, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_doc_count(corpus, &count) == ET_EVAL5_OK &&
        count == 0);
  CHECK(et_eval5_aot_corpus_add_doc(NULL, NULL, 0, NULL, 0) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_add_doc(corpus, NULL, 1, NULL, 0) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_add_doc(corpus, NULL, 0, NULL, 1) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_add_doc(corpus, NULL, -1, NULL, 0) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_corpus_add_doc(corpus, NULL, 0, NULL, -1) ==
        ET_EVAL5_ERR_ARGUMENT);

  /* Empty eval corpus rejected */
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(NULL, corpus, ET_EVAL5_NGRAM_TOKENS, 2,
                                         1, 0.25, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, NULL, ET_EVAL5_NGRAM_TOKENS, 2,
                                         1, 0.25, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);

  const uint8_t doc_bytes[] = "abcdef";
  const uint32_t doc_tokens[] = {1u, 2u, 3u, 4u};
  CHECK(et_eval5_aot_corpus_add_doc(corpus, doc_bytes, 6, doc_tokens, 4) ==
        ET_EVAL5_OK);

  /* Invalid thresholds & modes & bounds */
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, 0, 2, 1, 0.25, 0.0,
                                         indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, -1, 2, 1, 0.25, 0.0,
                                         indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         0, 1, 0.25, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 2, 0.25, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, -0.01, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 1.01, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, NAN, 0.0, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, -0.1, indices, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, INFINITY, indices, 4,
                                         summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, 0.0, indices, 0, summary,
                                         &rate) == ET_EVAL5_ERR_BOUNDS);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, 0.0, NULL, 4, summary,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, 0.0, indices, 4, NULL,
                                         &rate) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_audit_contamination(corpus, corpus, ET_EVAL5_NGRAM_TOKENS,
                                         2, 1, 0.25, 0.0, indices, 4, summary,
                                         NULL) == ET_EVAL5_ERR_ARGUMENT);
  /* Outputs remained untouched on invalid argument / bounds */
  CHECK(indices[0] == 99 && summary[0] == 99 && rate == 99.0);

  /* LM scorer negative tests */
  float logits[4] = {0.0f, 1.0f, 0.0f, 1.0f};
  uint32_t targets[2] = {0u, 1u};
  CHECK(et_eval5_aot_score_lm(NULL, 4, targets, 2, 2, 4, lm) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_lm(logits, 4, NULL, 2, 2, 4, lm) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_lm(logits, 4, targets, 2, 2, 4, NULL) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_lm(logits, 4, targets, 2, 2, 0, lm) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_lm(logits, 3, targets, 2, 2, 4, lm) ==
        ET_EVAL5_ERR_BOUNDS);
  targets[1] = 2u;
  CHECK(et_eval5_aot_score_lm(logits, 4, targets, 2, 2, 4, lm) ==
        ET_EVAL5_ERR_TARGET);
  targets[1] = 1u;
  logits[0] = NAN;
  CHECK(et_eval5_aot_score_lm(logits, 4, targets, 2, 2, 4, lm) ==
        ET_EVAL5_ERR_NONFINITE);
  logits[0] = 0.0f;
  CHECK(lm[0] == 99.0);

  /* MC batch negative tests */
  CHECK(et_eval5_aot_mc_create(&mc) == ET_EVAL5_OK);
  int64_t mc_corr = 99, mc_tot = 99;
  double mc_acc = 99.0;
  CHECK(et_eval5_aot_mc_add_choice(mc, logits, 4, targets, 2, 2) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_mc(mc, 1, &mc_corr, &mc_tot, &mc_acc) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_mc_begin_item(mc, -1) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_mc_begin_item(mc, 2) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_mc_add_choice(mc, logits, 3, targets, 2, 2) ==
        ET_EVAL5_ERR_BOUNDS);
  CHECK(et_eval5_aot_mc_add_choice(mc, logits, 4, targets, 2, 2) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_score_mc(mc, 2, &mc_corr, &mc_tot, &mc_acc) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_score_mc(mc, 1, &mc_corr, &mc_tot, &mc_acc) ==
        ET_EVAL5_ERR_TARGET);
  CHECK(mc_corr == 99 && mc_tot == 99 && mc_acc == 99.0);

  /* Manifest & helper negative tests */
  CHECK(et_eval5_aot_manifest_create(&manifest) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_set_identity(NULL, "s", "t", "p", 8, 0) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_set_identity(manifest, "s", "t", "p", 0, 0) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_set_identity(manifest, "", "t", "p", 8, 0) ==
        ET_EVAL5_ERR_FORMAT);
  CHECK(et_eval5_aot_manifest_set_digests(manifest, NULL, digest, digest,
                                          digest) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_verify_clean(NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_verify_clean(manifest) == ET_EVAL5_ERR_FORMAT);
  CHECK(et_eval5_aot_manifest_write_atomic(NULL, manifest) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_write_atomic("", manifest) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_read_file(NULL, manifest) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_manifest_read_file("", manifest) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_sha256_bytes(NULL, 1, digest) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_sha256_bytes(doc_bytes, 6, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_sha256_file(NULL, digest) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_sha256_file("", digest) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_store_u32(NULL, 1, 0, 1) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_store_u32(targets, 2, 2, 1) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_store_u32(targets, 2, 0, -1) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_store_f32(NULL, 1, 0, 1.0) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_store_f32(logits, 4, 4, 1.0) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(isnan(et_eval5_aot_load_f64(NULL, 1, 0)));
  CHECK(isnan(et_eval5_aot_load_f64(lm, 4, 4)));
  CHECK(et_eval5_aot_load_i64(NULL, 1, 0) == INT64_MIN);
  CHECK(et_eval5_aot_load_i64(summary, 6, -1) == INT64_MIN);
  CHECK(et_eval5_aot_bytes_equal(NULL, digest, 1, &eq) ==
        ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_aot_bytes_equal(digest, digest, 32, NULL) ==
        ET_EVAL5_ERR_ARGUMENT);

  et_eval5_aot_manifest_destroy(manifest);
  et_eval5_aot_mc_destroy(mc);
  et_eval5_aot_corpus_destroy(corpus);
}

static void test_clean_contaminated_symlink_and_tamper(const char *dir) {
  et_eval5_aot_corpus *train = NULL;
  et_eval5_aot_corpus *clean_eval = NULL;
  et_eval5_aot_corpus *exact_eval = NULL;
  et_eval5_aot_corpus *ngram_eval = NULL;
  CHECK(et_eval5_aot_corpus_create(&train) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_create(&clean_eval) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_create(&exact_eval) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_create(&ngram_eval) == ET_EVAL5_OK);

  const uint32_t tr0[] = {1u, 4u, 1u, 5u, 9u, 2u, 6u, 5u, 3u};
  const uint32_t tr1[] = {2u, 7u, 1u, 8u, 2u, 8u, 1u, 8u, 2u};
  const uint32_t ev_clean0[] = {0u, 3u, 10u, 11u, 12u, 13u, 14u, 15u, 0u};
  const uint32_t ev_clean1[] = {15u, 14u, 13u, 12u, 11u, 10u, 3u, 0u, 15u};
  const uint32_t ev_ngram0[] = {0u, 3u, 10u, 1u, 5u, 9u, 2u, 6u, 14u};

  CHECK(et_eval5_aot_corpus_add_doc(train, (const uint8_t *)"train-0", 7, tr0,
                                    9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(train, (const uint8_t *)"train-1", 7, tr1,
                                    9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(clean_eval, (const uint8_t *)"clean-0", 7,
                                    ev_clean0, 9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(clean_eval, (const uint8_t *)"clean-1", 7,
                                    ev_clean1, 9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(exact_eval, (const uint8_t *)"train-0", 7,
                                    tr0, 9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(exact_eval, (const uint8_t *)"clean-1", 7,
                                    ev_clean1, 9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(ngram_eval, (const uint8_t *)"ngram-0", 7,
                                    ev_ngram0, 9) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_corpus_add_doc(ngram_eval, (const uint8_t *)"clean-1", 7,
                                    ev_clean1, 9) == ET_EVAL5_OK);

  int64_t indices[2] = {-1, -1};
  int64_t clean_summary[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN] = {0};
  int64_t exact_summary[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN] = {0};
  int64_t ngram_summary[ET_EVAL5_AOT_AUDIT_SUMMARY_LEN] = {0};
  double clean_rate = -1.0, exact_rate = -1.0, ngram_rate = -1.0;

  CHECK(et_eval5_aot_audit_contamination(train, clean_eval,
                                         ET_EVAL5_NGRAM_TOKENS, 3, 1, 0.25, 0.0,
                                         indices, 2, clean_summary,
                                         &clean_rate) == ET_EVAL5_OK);
  CHECK(clean_summary[ET_EVAL5_AOT_AUDIT_EVAL_DOCS] == 2 &&
        clean_summary[ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS] == 0 &&
        clean_summary[ET_EVAL5_AOT_AUDIT_TOTAL_NGRAMS] == 14 &&
        clean_summary[ET_EVAL5_AOT_AUDIT_MATCHED_NGRAMS] == 0 &&
        clean_rate == 0.0);

  CHECK(et_eval5_aot_audit_contamination(train, exact_eval,
                                         ET_EVAL5_NGRAM_TOKENS, 3, 1, 0.25, 0.0,
                                         indices, 2, exact_summary,
                                         &exact_rate) ==
        ET_EVAL5_ERR_CONTAMINATED);
  CHECK(exact_summary[ET_EVAL5_AOT_AUDIT_EXACT_MATCHES] == 1 &&
        exact_summary[ET_EVAL5_AOT_AUDIT_CONTAMINATED_DOCS] == 1 &&
        indices[0] == 0 && exact_rate == 0.5);

  CHECK(et_eval5_aot_audit_contamination(train, ngram_eval,
                                         ET_EVAL5_NGRAM_TOKENS, 3, 1, 0.25, 0.0,
                                         indices, 2, ngram_summary,
                                         &ngram_rate) ==
        ET_EVAL5_ERR_CONTAMINATED);
  CHECK(ngram_summary[ET_EVAL5_AOT_AUDIT_EXACT_MATCHES] == 0 &&
        ngram_summary[ET_EVAL5_AOT_AUDIT_NGRAM_DOCS] == 1 &&
        ngram_summary[ET_EVAL5_AOT_AUDIT_MATCHED_NGRAMS] == 3 &&
        indices[0] == 0 && ngram_rate == 0.5);

  /* Score LM and MC */
  const float lm_logits[8] = {0.2f, -0.1f, 0.4f, -0.3f,
                              -0.5f, 0.6f, 0.1f, -0.2f};
  const uint32_t lm_targets[2] = {2u, 1u};
  double lm_metrics[ET_EVAL5_AOT_LM_METRICS_LEN] = {0};
  CHECK(et_eval5_aot_score_lm(lm_logits, 8, lm_targets, 2, 4, 4, lm_metrics) ==
        ET_EVAL5_OK);
  CHECK(lm_metrics[ET_EVAL5_AOT_LM_TOTAL_NLL] > 0.0 &&
        lm_metrics[ET_EVAL5_AOT_LM_PERPLEXITY] >= 1.0 &&
        lm_metrics[ET_EVAL5_AOT_LM_BITS_PER_BYTE] > 0.0);

  et_eval5_aot_mc_batch *mc = NULL;
  CHECK(et_eval5_aot_mc_create(&mc) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_mc_begin_item(mc, 1) == ET_EVAL5_OK);
  const float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const uint32_t t0[1] = {0u};
  const float c1[4] = {0.0f, 2.0f, 0.0f, 0.0f};
  const uint32_t t1[1] = {1u};
  CHECK(et_eval5_aot_mc_add_choice(mc, c0, 4, t0, 1, 4) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_mc_add_choice(mc, c1, 4, t1, 1, 4) == ET_EVAL5_OK);
  int64_t mc_corr = 0, mc_tot = 0;
  double mc_acc = 0.0;
  CHECK(et_eval5_aot_score_mc(mc, 1, &mc_corr, &mc_tot, &mc_acc) ==
        ET_EVAL5_OK);
  CHECK(mc_corr == 1 && mc_tot == 1 && mc_acc == 1.0);

  /* Build manifest */
  et_eval5_aot_manifest *manifest = NULL;
  et_eval5_aot_manifest *reloaded = NULL;
  CHECK(et_eval5_aot_manifest_create(&manifest) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_create(&reloaded) == ET_EVAL5_OK);

  uint8_t d_tr[32], d_ev[32], d_tok[32], d_ckpt[32];
  CHECK(et_eval5_aot_sha256_bytes((const uint8_t *)"train", 5, d_tr) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_sha256_bytes((const uint8_t *)"eval", 4, d_ev) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_sha256_bytes((const uint8_t *)"tok", 3, d_tok) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_sha256_bytes((const uint8_t *)"ckpt", 4, d_ckpt) ==
        ET_EVAL5_OK);

  CHECK(et_eval5_aot_manifest_set_identity(manifest, "eval5-bridge-test",
                                           "lm-mc", "clean.v1", 8, 0) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_set_digests(manifest, d_tr, d_ev, d_tok,
                                          d_ckpt) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_set_lm(manifest, 2, 4, lm_metrics) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_set_mc(manifest, 1, mc_corr, mc_tot, mc_acc) ==
        ET_EVAL5_OK);

  /* Contaminated fail-closed rejection on manifest publication */
  char rejected_path[4096];
  make_path(rejected_path, sizeof(rejected_path), dir, "rejected.manifest");
  CHECK(et_eval5_aot_manifest_set_audit(manifest, ET_EVAL5_NGRAM_TOKENS, 3, 1,
                                        0.25, 0.0, exact_summary,
                                        exact_rate) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_verify_clean(manifest) ==
        ET_EVAL5_ERR_CONTAMINATED);
  CHECK(et_eval5_aot_manifest_write_atomic(rejected_path, manifest) ==
        ET_EVAL5_ERR_CONTAMINATED);
  struct stat st;
  CHECK(lstat(rejected_path, &st) != 0);

  CHECK(et_eval5_aot_manifest_set_audit(manifest, ET_EVAL5_NGRAM_TOKENS, 3, 1,
                                        0.25, 0.0, ngram_summary,
                                        ngram_rate) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_verify_clean(manifest) ==
        ET_EVAL5_ERR_CONTAMINATED);
  CHECK(et_eval5_aot_manifest_write_atomic(rejected_path, manifest) ==
        ET_EVAL5_ERR_CONTAMINATED);
  CHECK(lstat(rejected_path, &st) != 0);

  /* Clean atomic publication and verified reload */
  CHECK(et_eval5_aot_manifest_set_audit(manifest, ET_EVAL5_NGRAM_TOKENS, 3, 1,
                                        0.25, 0.0, clean_summary,
                                        clean_rate) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_verify_clean(manifest) == ET_EVAL5_OK);

  char clean_path[4096];
  make_path(clean_path, sizeof(clean_path), dir, "clean.manifest");
  CHECK(et_eval5_aot_manifest_write_atomic(clean_path, manifest) ==
        ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_read_file(clean_path, reloaded) == ET_EVAL5_OK);
  CHECK(et_eval5_aot_manifest_verify_clean(reloaded) == ET_EVAL5_OK);

  int64_t r_counts[ET_EVAL5_AOT_MANIFEST_COUNTS_LEN] = {0};
  double r_metrics[ET_EVAL5_AOT_MANIFEST_METRICS_LEN] = {0};
  uint8_t r_sha[32] = {0};
  CHECK(et_eval5_aot_manifest_get_metrics(reloaded, r_counts, r_metrics,
                                          r_sha) == ET_EVAL5_OK);
  CHECK(r_counts[0] == (ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC));
  CHECK(r_counts[2] == 2 && r_counts[3] == 4);
  CHECK(r_counts[4] == 1 && r_counts[5] == 1);
  CHECK(r_counts[6] == 2 && r_counts[7] == 0);
  CHECK(r_metrics[0] == lm_metrics[0] && r_metrics[2] == lm_metrics[2] &&
        r_metrics[4] == 1.0 && r_metrics[5] == 0.0);

  uint8_t rt[32], re[32], rk[32], rc[32];
  CHECK(et_eval5_aot_manifest_get_digests(reloaded, rt, re, rk, rc) ==
        ET_EVAL5_OK);
  int32_t eq = 0;
  CHECK(et_eval5_aot_bytes_equal(rt, d_tr, 32, &eq) == ET_EVAL5_OK && eq == 1);
  CHECK(et_eval5_aot_bytes_equal(re, d_ev, 32, &eq) == ET_EVAL5_OK && eq == 1);
  CHECK(et_eval5_aot_bytes_equal(rk, d_tok, 32, &eq) == ET_EVAL5_OK && eq == 1);
  CHECK(et_eval5_aot_bytes_equal(rc, d_ckpt, 32, &eq) == ET_EVAL5_OK &&
        eq == 1);

  /* File SHA-256 helper checks regular file and rejects symlinks */
  uint8_t file_sha[32] = {0};
  CHECK(et_eval5_aot_sha256_file(clean_path, file_sha) == ET_EVAL5_OK);

  char link_path[4096];
  make_path(link_path, sizeof(link_path), dir, "link.manifest");
  CHECK(symlink("clean.manifest", link_path) == 0);
  CHECK(et_eval5_aot_manifest_read_file(link_path, reloaded) ==
        ET_EVAL5_ERR_IO);
  CHECK(et_eval5_aot_manifest_write_atomic(link_path, manifest) ==
        ET_EVAL5_ERR_IO);
  CHECK(et_eval5_aot_sha256_file(link_path, file_sha) == ET_EVAL5_ERR_IO);
  CHECK(unlink(link_path) == 0);

  /* Tampered manifest rejection (in-memory and on-disk) */
  int64_t fmt_size = 0;
  CHECK(et_eval5_aot_manifest_format(manifest, NULL, 0, &fmt_size) ==
        ET_EVAL5_OK);
  CHECK(fmt_size > 0 && fmt_size <= (int64_t)ET_EVAL5_MANIFEST_MAX_BYTES);
  char buf[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  int64_t written = 0;
  CHECK(et_eval5_aot_manifest_format(manifest, buf, (int64_t)sizeof(buf),
                                     &written) == ET_EVAL5_OK);
  CHECK(written == fmt_size);
  CHECK(et_eval5_aot_manifest_parse(buf, written, reloaded) == ET_EVAL5_OK);

  char *suite_field = strstr(buf, "suite=");
  CHECK(suite_field != NULL);
  suite_field[6] = (suite_field[6] == 'z') ? 'a' : 'z';
  CHECK(et_eval5_aot_manifest_parse(buf, written, reloaded) ==
        ET_EVAL5_ERR_CHECKSUM);

  char tampered_path[4096];
  make_path(tampered_path, sizeof(tampered_path), dir, "tampered.manifest");
  int fd = open(tampered_path, O_CREAT | O_EXCL | O_WRONLY, 0600);
  CHECK(fd >= 0);
  CHECK(write(fd, buf, (size_t)written) == (ssize_t)written);
  CHECK(close(fd) == 0);
  CHECK(et_eval5_aot_manifest_read_file(tampered_path, reloaded) ==
        ET_EVAL5_ERR_CHECKSUM);
  CHECK(unlink(tampered_path) == 0);

  et_eval5_aot_manifest_destroy(reloaded);
  et_eval5_aot_manifest_destroy(manifest);
  et_eval5_aot_mc_destroy(mc);
  et_eval5_aot_corpus_destroy(ngram_eval);
  et_eval5_aot_corpus_destroy(exact_eval);
  et_eval5_aot_corpus_destroy(clean_eval);
  et_eval5_aot_corpus_destroy(train);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  test_null_and_bounds_negative();
  test_clean_contaminated_symlink_and_tamper(argv[1]);
  puts("EVAL5 AOT bridge tests (negative, clean, contaminated, symlink, tamper): PASS");
  return 0;
}
