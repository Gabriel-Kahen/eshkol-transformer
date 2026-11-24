#include "eshkol_transformer/data4_pipeline_abi.h"
#include "data4_pipeline_eshkol_bridge.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *msg, int line) {
  fprintf(stderr, "DATA4 pipeline test failure at line %d: %s\n", line, msg);
  exit(1);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) die(#cond, __LINE__);                                         \
  } while (0)

static void test_utf8_edge_cases(void) {
  uint64_t cp = 0u, pr = 0u, al = 0u;

  /* Empty input is valid UTF-8 with 0 codepoints. */
  CHECK(et_data4_validate_utf8(NULL, 0u, &cp, &pr, &al) == ET_DATA4_OK);
  CHECK(cp == 0u && pr == 0u && al == 0u);

  /* Valid 1, 2, 3, and 4 byte UTF-8 sequences including boundary values. */
  const uint8_t valid_all[] = {
      'A', '0', ' ', '\t', '\n', '\r',
      0xC2u, 0xA0u,                     /* U+00A0 NBSP (min 2-byte, printable) */
      0xC3u, 0xA9u,                     /* U+00E9 'é' (alnum + printable) */
      0xE0u, 0xA0u, 0x80u,              /* U+0800 (min 3-byte, printable) */
      0xE2u, 0x82u, 0xACu,              /* U+20AC '€' (printable) */
      0xEDu, 0x9Fu, 0xBFu,              /* U+D7FF (just below surrogate) */
      0xEEu, 0x80u, 0x80u,              /* U+E000 (just above surrogate) */
      0xE6u, 0x97u, 0xA5u,              /* U+65E5 '日' (CJK alnum + printable) */
      0xF0u, 0x90u, 0x80u, 0x80u,       /* U+10000 (min 4-byte, printable) */
      0xF0u, 0x9Fu, 0x98u, 0x80u,       /* U+1F600 emoji (printable) */
      0xF4u, 0x8Fu, 0xBFu, 0xBFu        /* U+10FFFF (max scalar, nonchar!) */
  };
  CHECK(et_data4_validate_utf8(valid_all, sizeof(valid_all), &cp, &pr, &al) ==
        ET_DATA4_OK);
  CHECK(cp == 16u);
  /* U+10FFFF is a noncharacter so not printable; other 15 are printable. */
  CHECK(pr == 15u);
  /* Alnum: 'A', '0', 'é', '日' = 4. */
  CHECK(al == 4u);

  /* Embedded NUL and C1 control + U+FDD0 noncharacter: valid UTF-8, not printable. */
  const uint8_t ctrl_and_nonchar[] = {
      0x00u, 0x7Fu, 0xC2u, 0x80u, 0xC2u, 0x9Fu, 0xEFu, 0xB7u, 0x90u, 0xEFu,
      0xBFu, 0xBEu};
  CHECK(et_data4_validate_utf8(ctrl_and_nonchar, sizeof(ctrl_and_nonchar), &cp,
                               &pr, &al) == ET_DATA4_OK);
  CHECK(cp == 6u && pr == 0u && al == 0u);

  /* Invalid UTF-8 sequences: overlong, surrogates, >U+10FFFF, stray, truncated. */
  struct bad_case {
    const uint8_t bytes[8];
    size_t len;
  };
  const struct bad_case bad_cases[] = {
      {{0x80u}, 1u},                               /* stray continuation */
      {{0xBFu}, 1u},                               /* stray continuation */
      {{0xC0u, 0xAFu}, 2u},                        /* overlong '/' */
      {{0xC1u, 0xBFu}, 2u},                        /* overlong 0x7F */
      {{0xC2u}, 1u},                               /* truncated 2-byte */
      {{0xC2u, 0x20u}, 2u},                        /* bad continuation 2-byte */
      {{0xE0u, 0x80u, 0xAFu}, 3u},                 /* overlong 3-byte */
      {{0xE0u, 0x9Fu, 0xBFu}, 3u},                 /* overlong U+07FF */
      {{0xEDu, 0xA0u, 0x80u}, 3u},                 /* surrogate U+D800 */
      {{0xEDu, 0xADu, 0xBFu}, 3u},                 /* surrogate U+DB7F */
      {{0xEDu, 0xBFu, 0xBFu}, 3u},                 /* surrogate U+DFFF */
      {{0xE2u, 0x82u}, 2u},                        /* truncated 3-byte */
      {{0xE2u, 0x82u, 0x20u}, 3u},                 /* bad cont in 3-byte */
      {{0xF0u, 0x80u, 0x80u, 0xAFu}, 4u},          /* overlong 4-byte */
      {{0xF0u, 0x8Fu, 0xBFu, 0xBFu}, 4u},          /* overlong U+FFFF */
      {{0xF4u, 0x90u, 0x80u, 0x80u}, 4u},          /* U+110000 (> max) */
      {{0xF5u, 0x80u, 0x80u, 0x80u}, 4u},          /* > 0xF4 */
      {{0xF0u, 0x9Fu, 0x98u}, 3u},                 /* truncated 4-byte */
      {{0xF0u, 0x9Fu, 0x98u, 0x20u}, 4u},          /* bad cont in 4-byte */
      {{0xFEu}, 1u},                               /* invalid byte */
      {{0xFFu}, 1u},                               /* invalid byte */
  };
  for (size_t i = 0u; i < sizeof(bad_cases) / sizeof(*bad_cases); ++i) {
    CHECK(et_data4_validate_utf8(bad_cases[i].bytes, bad_cases[i].len, &cp, &pr,
                                 &al) == ET_DATA4_ERR_FORMAT);
  }

  /* Aliasing and argument guards. */
  CHECK(et_data4_validate_utf8(NULL, 4u, &cp, &pr, &al) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_validate_utf8(valid_all, sizeof(valid_all), &cp, &cp, &al) ==
        ET_DATA4_ERR_ALIASING);
  uint64_t aliased_buf[2] = {0u, 0u};
  CHECK(et_data4_validate_utf8((const uint8_t *)aliased_buf,
                               sizeof(aliased_buf), &aliased_buf[0], NULL,
                               NULL) == ET_DATA4_ERR_ALIASING);
}

static void test_false_positive_protection(void) {
  /* Construct two documents that share a single common 5-gram and DO collide
   * in at least one MinHash permutation (bands == num_perms => rows_per_band=1)
   * yet have low exact Jaccard similarity (< 0.80). Exact Jaccard verification
   * MUST prevent a false-positive near-duplicate drop. */
  const char *doc_a =
      "SHARED_ANCHOR_TOKEN alpha beta gamma delta epsilon zeta eta theta iota "
      "kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi";
  const char *doc_b =
      "SHARED_ANCHOR_TOKEN mercury venus earth mars jupiter saturn uranus "
      "neptune pluto ceres eris haumea makemake sedna orcus quaoar gonggong";

  uint64_t sig_a[128];
  uint64_t sig_b[128];
  uint64_t uniq_a = 0u, uniq_b = 0u;
  CHECK(et_data4_compute_minhash((const uint8_t *)doc_a, strlen(doc_a), 5u,
                                 128u, UINT64_C(0x123456789abcdef0), sig_a,
                                 128u, &uniq_a) == ET_DATA4_OK);
  CHECK(et_data4_compute_minhash((const uint8_t *)doc_b, strlen(doc_b), 5u,
                                 128u, UINT64_C(0x123456789abcdef0), sig_b,
                                 128u, &uniq_b) == ET_DATA4_OK);

  int collisions = 0;
  for (int k = 0; k < 128; ++k) {
    if (sig_a[k] == sig_b[k]) ++collisions;
  }
  CHECK(collisions >= 1); /* Guarantee LSH candidate pair WAS proposed! */

  uint64_t jnum = 0u, jden = 0u;
  CHECK(et_data4_exact_jaccard((const uint8_t *)doc_a, strlen(doc_a),
                               (const uint8_t *)doc_b, strlen(doc_b), 5u, &jnum,
                               &jden) == ET_DATA4_OK);
  CHECK(jnum > 0u && jden > jnum && (double)jnum / (double)jden < 0.25);

  et_data4_pipeline_config cfg;
  CHECK(et_data4_config_default(&cfg) == ET_DATA4_OK);
  cfg.minhash_num_perms = 128u;
  cfg.minhash_bands = 128u; /* rows_per_band == 1 => every collision proposes */
  cfg.minhash_seed = UINT64_C(0x123456789abcdef0);
  cfg.near_dup_jaccard_threshold = 0.80;

  const et_data4_document docs[2] = {
      {"fp-doc-a", (const uint8_t *)doc_a, strlen(doc_a)},
      {"fp-doc-b", (const uint8_t *)doc_b, strlen(doc_b)},
  };
  et_data4_doc_result results[2];
  et_data4_pipeline_summary summary;
  CHECK(et_data4_pipeline_run(docs, 2u, &cfg, results, 2u, &summary) ==
        ET_DATA4_OK);
  CHECK(summary.kept_documents == 2u);
  CHECK(summary.duplicate_near_documents == 0u);
  CHECK(results[0].disposition == ET_DATA4_DISP_KEPT);
  CHECK(results[1].disposition == ET_DATA4_DISP_KEPT);
}

static void test_rejected_does_not_suppress_later(void) {
  /* Doc 0 fails quality filter (low printable ratio due to control bytes),
   * while Doc 1 is clean text containing the same text core. Doc 0 must NOT
   * enter the dedup index or suppress Doc 1. */
  const uint8_t bad_doc[] =
      "\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13"
      "The deterministic Eshkol transformer corpus preparation pipeline "
      "validates UTF-8 documents carefully.";
  const char *clean_doc =
      "The deterministic Eshkol transformer corpus preparation pipeline "
      "validates UTF-8 documents carefully.";

  et_data4_pipeline_config cfg;
  CHECK(et_data4_config_default(&cfg) == ET_DATA4_OK);
  cfg.minhash_num_perms = 64u;
  cfg.minhash_bands = 32u;
  cfg.near_dup_jaccard_threshold = 0.60;

  const et_data4_document docs[2] = {
      {"noisy-first", bad_doc, sizeof(bad_doc) - 1u},
      {"clean-second", (const uint8_t *)clean_doc, strlen(clean_doc)},
  };
  et_data4_doc_result results[2];
  et_data4_pipeline_summary summary;
  CHECK(et_data4_pipeline_run(docs, 2u, &cfg, results, 2u, &summary) ==
        ET_DATA4_OK);
  CHECK(results[0].disposition == ET_DATA4_DISP_REJECTED_QUALITY);
  CHECK(results[1].disposition == ET_DATA4_DISP_KEPT);
  CHECK(summary.kept_documents == 1u);
}

static void test_empty_and_zero_length_and_negatives(void) {
  et_data4_pipeline_config cfg;
  CHECK(et_data4_config_default(&cfg) == ET_DATA4_OK);

  /* Empty corpus succeeds cleanly and formats a verifiable report. */
  et_data4_pipeline_summary empty_sum;
  CHECK(et_data4_pipeline_run(NULL, 0u, &cfg, NULL, 0u, &empty_sum) ==
        ET_DATA4_OK);
  CHECK(empty_sum.input_documents == 0u && empty_sum.kept_documents == 0u &&
        empty_sum.input_bytes == 0u && empty_sum.kept_bytes == 0u);
  size_t empty_rep_len = 0u;
  uint8_t empty_sha[ET_DATA4_SHA256_BYTES];
  CHECK(et_data4_report_format(&cfg, &empty_sum, NULL, 0u, NULL, 0u,
                               &empty_rep_len, empty_sha) == ET_DATA4_OK);
  char *empty_rep = (char *)malloc(empty_rep_len + 1u);
  CHECK(empty_rep != NULL);
  size_t empty_written = 0u;
  CHECK(et_data4_report_format(&cfg, &empty_sum, NULL, 0u, empty_rep,
                               empty_rep_len + 1u, &empty_written,
                               empty_sha) == ET_DATA4_OK);
  CHECK(empty_written == empty_rep_len);
  et_data4_pipeline_summary verified_empty;
  CHECK(et_data4_report_verify(empty_rep, empty_written, NULL, &verified_empty,
                               NULL, 0u) == ET_DATA4_OK);
  CHECK(memcmp(verified_empty.report_sha256, empty_sha,
               ET_DATA4_SHA256_BYTES) == 0);
  free(empty_rep);

  /* Zero-length doc: rejected under default min_bytes=1, kept when min=0. */
  const et_data4_document zdoc[2] = {
      {"zero-a", NULL, 0u},
      {"zero-b", (const uint8_t *)"", 0u},
  };
  et_data4_doc_result zres[2];
  et_data4_pipeline_summary zsum;
  CHECK(et_data4_pipeline_run(zdoc, 2u, &cfg, zres, 2u, &zsum) == ET_DATA4_OK);
  CHECK(zres[0].disposition == ET_DATA4_DISP_REJECTED_LENGTH);
  CHECK(zres[1].disposition == ET_DATA4_DISP_REJECTED_LENGTH);

  et_data4_pipeline_config zero_ok_cfg = cfg;
  zero_ok_cfg.min_bytes = 0u;
  zero_ok_cfg.min_codepoints = 0u;
  zero_ok_cfg.min_printable_ratio = 0.0;
  zero_ok_cfg.min_alnum_ratio = 0.0;
  CHECK(et_data4_pipeline_run(zdoc, 2u, &zero_ok_cfg, zres, 2u, &zsum) ==
        ET_DATA4_OK);
  CHECK(zres[0].disposition == ET_DATA4_DISP_KEPT);
  CHECK(zres[1].disposition == ET_DATA4_DISP_DUPLICATE_EXACT);
  CHECK(strcmp(zres[1].matched_source_id, "zero-a") == 0);

  /* Negative config tests. */
  et_data4_pipeline_config bad = cfg;
  bad.min_bytes = 100u;
  bad.max_bytes = 50u;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.min_codepoints = 20u;
  bad.max_codepoints = 10u;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.min_printable_ratio = NAN;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.min_alnum_ratio = -0.0;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.max_dup_line_ratio = 1.01;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.max_repeated_ngram_ratio = INFINITY;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.rep_ngram_size = 0u;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.minhash_bands = 15u; /* 64 % 15 != 0 */
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.near_dup_jaccard_threshold = 0.0; /* must be > 0 when near_dup_enabled */
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  bad = cfg;
  bad.reserved0 = 1u;
  CHECK(et_data4_pipeline_run(NULL, 0u, &bad, NULL, 0u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  /* Null, bounds, invalid/duplicate source IDs, aliasing. */
  const char *hello = "hello world corpus sample text 12345";
  const et_data4_document dup_ids[2] = {
      {"same-id", (const uint8_t *)hello, strlen(hello)},
      {"same-id", (const uint8_t *)hello, strlen(hello)},
  };
  CHECK(et_data4_pipeline_run(dup_ids, 2u, &cfg, zres, 2u, &zsum) ==
        ET_DATA4_ERR_DUPLICATE_SOURCE_ID);

  const et_data4_document bad_id_space[1] = {
      {"bad id", (const uint8_t *)hello, strlen(hello)},
  };
  CHECK(et_data4_pipeline_run(bad_id_space, 1u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_INVALID_SOURCE_ID);

  const et_data4_document bad_id_empty[1] = {
      {"", (const uint8_t *)hello, strlen(hello)},
  };
  CHECK(et_data4_pipeline_run(bad_id_empty, 1u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_INVALID_SOURCE_ID);

  const et_data4_document bad_bytes_null[1] = {
      {"ok-id", NULL, 5u},
  };
  CHECK(et_data4_pipeline_run(bad_bytes_null, 1u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);

  CHECK(et_data4_pipeline_run(NULL, 1u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_pipeline_run(zdoc, 2u, &cfg, NULL, 2u, &zsum) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_pipeline_run(zdoc, 2u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_BOUNDS);

  /* Aliased doc bytes overlapping results_out. */
  const et_data4_document alias_doc[1] = {
      {"alias-id", (const uint8_t *)&zres[0], 16u},
  };
  CHECK(et_data4_pipeline_run(alias_doc, 1u, &cfg, zres, 1u, &zsum) ==
        ET_DATA4_ERR_ALIASING);

  /* Aliased config/summary overlap via union. */
  union {
    et_data4_pipeline_config c;
    et_data4_pipeline_summary s;
  } overlap;
  CHECK(et_data4_config_default(&overlap.c) == ET_DATA4_OK);
  CHECK(et_data4_pipeline_run(NULL, 0u, &overlap.c, NULL, 0u, &overlap.s) ==
        ET_DATA4_ERR_ALIASING);

  /* Misaligned AOT bridge pointer (void * + 1, zero typed-pointer UB). */
  void *session = et_data4_aot_session_create();
  CHECK(session != NULL);
  void *misaligned = (void *)((unsigned char *)session + 1u);
  CHECK(et_data4_aot_run(misaligned) == ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_set_limits(misaligned, 1, 100, 1, 100) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_get_stat(misaligned, ET_DATA4_AOT_STAT_KEPT_DOCS) == -1);
  CHECK(et_data4_aot_get_stat(session, ET_DATA4_AOT_STAT_KEPT_DOCS) == -1);
  CHECK(et_data4_aot_set_quality(session, 1.5, 0.5, 0.3, 0.4, 5) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_set_near_dedup(session, 1, 5, 64, 15, 1, 0.8) ==
        ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_add_doc(session, "a\0b", 3, "xyz", 3) ==
        ET_DATA4_ERR_INVALID_SOURCE_ID);
  CHECK(et_data4_aot_buf_store_u8(NULL, 8, 0, 1) == ET_DATA4_ERR_ARGUMENT);
  uint8_t tiny[4] = {0};
  CHECK(et_data4_aot_buf_store_u8(tiny, 4, 4, 1) == ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_buf_store_u8(tiny, 4, 0, 256) == ET_DATA4_ERR_ARGUMENT);
  CHECK(et_data4_aot_buf_load_u8(tiny, 4, -1) == -1);
  et_data4_aot_session_destroy(session);
}

static void run_oracle_corpus(const char *out_dir) {
  et_data4_pipeline_config cfg;
  CHECK(et_data4_config_default(&cfg) == ET_DATA4_OK);
  cfg.min_bytes = 24u;
  cfg.max_bytes = 512u;
  cfg.min_codepoints = 20u;
  cfg.max_codepoints = 512u;
  cfg.min_printable_ratio = 0.90;
  cfg.min_alnum_ratio = 0.50;
  cfg.max_dup_line_ratio = 0.25;
  cfg.max_repeated_ngram_ratio = 0.35;
  cfg.rep_ngram_size = 5u;
  cfg.near_dup_enabled = 1u;
  cfg.dedup_ngram_size = 5u;
  cfg.minhash_num_perms = 64u;
  cfg.minhash_bands = 16u;
  cfg.minhash_seed = UINT64_C(0x4441544134534545);
  cfg.near_dup_jaccard_threshold = 0.75;

  const char *d0 =
      "Eshkol transformer training requires deterministic corpus filtering and "
      "exact provenance verification across shards.";
  const uint8_t d1[] = {
      'b', 'a', 'd', ' ', 'u', 't', 'f', '8', ' ', 0xEDu, 0xA0u, 0x80u, ' ',
      's', 'u', 'r', 'r', 'o', 'g', 'a', 't', 'e', ' ', 't', 'e', 'x', 't'};
  const char *d2 = "too short doc";
  const char *d3 = "!!!@@@###$$$%%%^^^&&&***((()))___+++===---///|||~~~";
  const char *d4 =
      "Repeated spam line for testing.\n"
      "Repeated spam line for testing.\n"
      "Repeated spam line for testing.\n"
      "One distinct closing sentence.\n";
  const char *d5 =
      "abcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcde1234";
  const char *d6 = d0; /* exact duplicate of d0 */
  const char *d7 =
      "Eshkol transformer training requires deterministic corpus filtering and "
      "exact provenance verification across shards!"; /* near-dup of d0 */
  const char *d8 =
      "Multilingual UTF-8 corpus sample: caf\xC3\xA9 na\xC3\xAFve "
      "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E "
      "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 "
      "with high alphanumeric density.";
  const char *d9 =
      "Completely distinct technical note on memory-safe C11 ring buffers, "
      "checksummed manifests, and reproducible LLVM builds.";

  const et_data4_document docs[10] = {
      {"src.clean.0", (const uint8_t *)d0, strlen(d0)},
      {"src.bad_utf8.1", d1, sizeof(d1)},
      {"src.short.2", (const uint8_t *)d2, strlen(d2)},
      {"src.symbols.3", (const uint8_t *)d3, strlen(d3)},
      {"src.dup_lines.4", (const uint8_t *)d4, strlen(d4)},
      {"src.rep_ngram.5", (const uint8_t *)d5, strlen(d5)},
      {"src.exact_dup.6", (const uint8_t *)d6, strlen(d6)},
      {"src.near_dup.7", (const uint8_t *)d7, strlen(d7)},
      {"src.utf8_multi.8", (const uint8_t *)d8, strlen(d8)},
      {"src.distinct.9", (const uint8_t *)d9, strlen(d9)},
  };

  et_data4_doc_result results[10];
  et_data4_pipeline_summary summary;
  CHECK(et_data4_pipeline_run(docs, 10u, &cfg, results, 10u, &summary) ==
        ET_DATA4_OK);

  CHECK(summary.input_documents == 10u);
  CHECK(summary.kept_documents == 3u);
  CHECK(summary.rejected_utf8_documents == 1u);
  CHECK(summary.rejected_length_documents == 1u);
  CHECK(summary.rejected_quality_documents == 1u);
  CHECK(summary.rejected_repetition_documents == 2u);
  CHECK(summary.duplicate_exact_documents == 1u);
  CHECK(summary.duplicate_near_documents == 1u);
  CHECK(results[0].disposition == ET_DATA4_DISP_KEPT);
  CHECK(results[1].disposition == ET_DATA4_DISP_REJECTED_UTF8);
  CHECK(results[2].disposition == ET_DATA4_DISP_REJECTED_LENGTH);
  CHECK(results[3].disposition == ET_DATA4_DISP_REJECTED_QUALITY);
  CHECK(results[4].disposition == ET_DATA4_DISP_REJECTED_REPETITION);
  CHECK(results[5].disposition == ET_DATA4_DISP_REJECTED_REPETITION);
  CHECK(results[6].disposition == ET_DATA4_DISP_DUPLICATE_EXACT);
  CHECK(strcmp(results[6].matched_source_id, "src.clean.0") == 0);
  CHECK(results[7].disposition == ET_DATA4_DISP_DUPLICATE_NEAR);
  CHECK(strcmp(results[7].matched_source_id, "src.clean.0") == 0);
  CHECK(results[7].jaccard_den > 0u &&
        results[7].jaccard_num < results[7].jaccard_den);
  CHECK(results[8].disposition == ET_DATA4_DISP_KEPT);
  CHECK(results[9].disposition == ET_DATA4_DISP_KEPT);

  size_t report_len = 0u;
  uint8_t report_sha[ET_DATA4_SHA256_BYTES];
  CHECK(et_data4_report_format(&cfg, &summary, results, 10u, NULL, 0u,
                               &report_len, report_sha) == ET_DATA4_OK);
  char *report = (char *)malloc(report_len + 1u);
  CHECK(report != NULL);
  size_t written = 0u;
  CHECK(et_data4_report_format(&cfg, &summary, results, 10u, report,
                               report_len + 1u, &written,
                               report_sha) == ET_DATA4_OK);
  CHECK(written == report_len);

  /* Round-trip verify and tamper rejection checks. */
  et_data4_pipeline_config vcfg;
  et_data4_pipeline_summary vsum;
  et_data4_doc_result vres[10];
  CHECK(et_data4_report_verify(report, report_len, &vcfg, &vsum, vres, 10u) ==
        ET_DATA4_OK);
  CHECK(memcmp(vsum.report_sha256, report_sha, ET_DATA4_SHA256_BYTES) == 0);
  CHECK(vsum.kept_documents == 3u && vsum.kept_bytes == summary.kept_bytes);
  CHECK(vres[7].jaccard_num == results[7].jaccard_num &&
        vres[7].jaccard_den == results[7].jaccard_den);

  /* Tamper body byte -> ET_DATA4_ERR_CHECKSUM. */
  char *tampered = (char *)malloc(report_len + 1u);
  CHECK(tampered != NULL);
  memcpy(tampered, report, report_len + 1u);
  char *pos = strstr(tampered, "src.clean.0");
  CHECK(pos != NULL);
  pos[0] = 'X';
  CHECK(et_data4_report_verify(tampered, report_len, NULL, NULL, NULL, 0u) ==
        ET_DATA4_ERR_CHECKSUM);

  /* Tamper checksum hex digit -> ET_DATA4_ERR_CHECKSUM. */
  memcpy(tampered, report, report_len + 1u);
  tampered[report_len - 2u] =
      tampered[report_len - 2u] == 'a' ? 'b' : 'a';
  CHECK(et_data4_report_verify(tampered, report_len, NULL, NULL, NULL, 0u) ==
        ET_DATA4_ERR_CHECKSUM);

  /* Bad version -> ET_DATA4_ERR_VERSION. */
  memcpy(tampered, report, report_len + 1u);
  char *ver = strstr(tampered, "version\t1\t0\n");
  CHECK(ver != NULL);
  ver[8] = '2';
  CHECK(et_data4_report_verify(tampered, report_len, NULL, NULL, NULL, 0u) ==
        ET_DATA4_ERR_VERSION);
  free(tampered);

  /* Write report and MinHash/Jaccard trace if out_dir provided. */
  if (out_dir != NULL) {
    char path[1024];
    int pn = snprintf(path, sizeof(path), "%s/report.tsv", out_dir);
    CHECK(pn > 0 && (size_t)pn < sizeof(path));
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL);
    CHECK(fwrite(report, 1u, report_len, fp) == report_len);
    CHECK(fclose(fp) == 0);

    pn = snprintf(path, sizeof(path), "%s/minhash.txt", out_dir);
    CHECK(pn > 0 && (size_t)pn < sizeof(path));
    fp = fopen(path, "wb");
    CHECK(fp != NULL);
    uint64_t sig[64];
    uint64_t uniq = 0u;
    CHECK(et_data4_compute_minhash(docs[0].bytes, docs[0].byte_count, 5u, 64u,
                                   cfg.minhash_seed, sig, 64u, &uniq) ==
          ET_DATA4_OK);
    fprintf(fp, "uniq %" PRIu64 "\n", uniq);
    for (int k = 0; k < 64; ++k) fprintf(fp, "%016" PRIx64 "\n", sig[k]);
    CHECK(fclose(fp) == 0);
  }

  printf("DATA4 NATIVE SUMMARY input=%" PRIu64 " kept=%" PRIu64
         " utf8=%" PRIu64 " len=%" PRIu64 " qual=%" PRIu64 " rep=%" PRIu64
         " exact=%" PRIu64 " near=%" PRIu64 " near_j=%" PRIu64 "/%" PRIu64
         " report_bytes=%zu PASS\n",
         summary.input_documents, summary.kept_documents,
         summary.rejected_utf8_documents, summary.rejected_length_documents,
         summary.rejected_quality_documents,
         summary.rejected_repetition_documents,
         summary.duplicate_exact_documents, summary.duplicate_near_documents,
         results[7].jaccard_num, results[7].jaccard_den, report_len);
  free(report);
}

static int verify_report_file_status(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) return 2;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return 2;
  }
  long sz = ftell(fp);
  if (sz < 0 || fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return 2;
  }
  char *buf = (char *)malloc((size_t)sz + 1u);
  if (buf == NULL) {
    fclose(fp);
    return 2;
  }
  size_t got = sz > 0 ? fread(buf, 1u, (size_t)sz, fp) : 0u;
  fclose(fp);
  if (got != (size_t)sz) {
    free(buf);
    return 2;
  }
  et_data4_status st =
      et_data4_report_verify(buf, (size_t)sz, NULL, NULL, NULL, 0u);
  free(buf);
  printf("%" PRId32 "\n", st);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--verify-report") == 0) {
    return verify_report_file_status(argv[2]);
  }
  const char *out_dir = NULL;
  if (argc == 3 && strcmp(argv[1], "--oracle-out") == 0) {
    out_dir = argv[2];
  } else if (argc != 1) {
    fprintf(stderr, "usage: %s [--oracle-out DIR | --verify-report FILE]\n",
            argv[0]);
    return 2;
  }

  test_utf8_edge_cases();
  test_false_positive_protection();
  test_rejected_does_not_suppress_later();
  test_empty_and_zero_length_and_negatives();
  run_oracle_corpus(out_dir);
  return 0;
}
