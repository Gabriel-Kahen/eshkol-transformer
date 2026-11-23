#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/eval5_benchmark_abi.h"

#include <dirent.h>
#include <fenv.h>
#include <float.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#define CHECK(condition) do { \
  if (!(condition)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
    exit(1); \
  } \
} while (0)

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define BYTE_DOC(text) {(const uint8_t *)(text), sizeof(text) - 1, NULL, 0}
#define TOKEN_DOC(text, ids) {(const uint8_t *)(text), sizeof(text) - 1, ids, COUNT(ids)}
#define SEQUENCE(values, targets, width) {values, targets, COUNT(targets), width, COUNT(values)}

static void print_audit(const char *label, uint32_t order, et_eval5_status status,
                        const et_eval5_contamination_summary *summary,
                        const uint64_t *indices) {
  printf("%s %" PRIu32 " %" PRId32 " %" PRIu64 " %" PRIu64 " %" PRIu64
         " %" PRIu64 " %" PRIu64 " %" PRIu64 " %a",
         label, order, status, summary->eval_documents, summary->contaminated_documents,
         summary->exact_doc_matches, summary->ngram_contaminated_docs,
         summary->total_eval_ngrams, summary->matched_eval_ngrams, summary->contamination_rate);
  for (uint64_t i = 0; i < summary->contaminated_documents; ++i)
    printf(" %" PRIu64, indices[i]);
  puts("");
}

static void test_byte_audit(et_eval5_manifest *manifest) {
  static const et_eval5_document train[] = {
      BYTE_DOC("abcabc"), BYTE_DOC("xy"), BYTE_DOC("abcabc"),
      BYTE_DOC("pqr"), BYTE_DOC("stu"), {NULL, 0, NULL, 0},
  };
  static const et_eval5_document eval[] = {
      BYTE_DOC("zzabcabcq"), BYTE_DOC("xy"), BYTE_DOC("abcabc"), BYTE_DOC("pqrstu"),
      BYTE_DOC("xyab"), BYTE_DOC(""), BYTE_DOC("xxx"), BYTE_DOC("stuabc"),
  };
  et_eval5_contamination_config config = {
      ET_EVAL5_NGRAM_BYTES, 3, 0, 0.5, 0.75,
  };
  uint64_t indices[COUNT(eval)];
  et_eval5_contamination_summary summary;
  et_eval5_status status = et_eval5_audit_contamination(
      train, COUNT(train), eval, COUNT(eval), &config, indices, COUNT(indices), &summary);
  CHECK(status == ET_EVAL5_OK);
  const uint64_t expected[] = {0, 1, 2, 3, 5, 7};
  CHECK(summary.eval_documents == COUNT(eval) && summary.contaminated_documents == COUNT(expected));
  CHECK(summary.exact_doc_matches == 3 && summary.ngram_contaminated_docs == 4);
  CHECK(summary.total_eval_ngrams == 22 && summary.matched_eval_ngrams == 12);
  CHECK(summary.contamination_rate == 0.75);
  CHECK(memcmp(indices, expected, sizeof(expected)) == 0);
  print_audit("byte", config.ngram_order, status, &summary, indices);
  manifest->contamination = config;
  manifest->audit = summary;

  config.max_allowed_rate = nextafter(0.75, 0.0);
  CHECK(et_eval5_audit_contamination(train, COUNT(train), eval, COUNT(eval), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_ERR_CONTAMINATED);
  CHECK(summary.contamination_rate == 0.75 && memcmp(indices, expected, sizeof(expected)) == 0);
  config.max_allowed_rate = 1.0;
  config.forbid_exact_match = 1;
  CHECK(et_eval5_audit_contamination(train, COUNT(train), eval, COUNT(eval), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_ERR_CONTAMINATED);

  config.min_ngram_overlap = 1.0;
  config.forbid_exact_match = 0;
  CHECK(et_eval5_audit_contamination(train, COUNT(train), eval, COUNT(eval), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_OK);
  CHECK(summary.ngram_contaminated_docs == 1 && summary.contaminated_documents == 3);
  CHECK(summary.matched_eval_ngrams == 12 && indices[0] == 1 && indices[1] == 2 && indices[2] == 5);

  indices[0] = UINT64_MAX;
  CHECK(et_eval5_audit_contamination(NULL, 0, eval, COUNT(eval), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_OK);
  CHECK(summary.contaminated_documents == 0 && summary.matched_eval_ngrams == 0 &&
        summary.total_eval_ngrams == 22 && summary.contamination_rate == 0.0);
  CHECK(indices[0] == UINT64_MAX);

  const et_eval5_document binary[] = {BYTE_DOC("\0a\xff"), BYTE_DOC("\0a\xfe")};
  CHECK(et_eval5_audit_contamination(binary, 1, binary, COUNT(binary), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_OK);
  CHECK(summary.exact_doc_matches == 1 && summary.ngram_contaminated_docs == 1 &&
        summary.matched_eval_ngrams == 1 && indices[0] == 0);
}

static void test_token_audit(void) {
  static const uint32_t a[] = {1, 257};
  static const uint32_t b[] = {2, 0};
  static const uint32_t c[] = {UINT32_MAX, 0, 3, 5, 7, 8, 9, 7, 8, 9, 7, 8, 9, 4, 6};
  static const uint32_t d[] = {7, 8, 9, 7, 8, 9, 7, 8, 9, 4, 6, 1, 257, 0, 1};
  static const uint32_t e[] = {UINT32_MAX, 0, 3, 5, 7, 8, 9, 7, 8, 9, 7, 8, 9, 4, 5};
  static const et_eval5_document train[] = {
      TOKEN_DOC("collider-a", a), TOKEN_DOC("long-train", c),
      TOKEN_DOC("duplicate-ids", a),
  };
  static const et_eval5_document eval[] = {
      TOKEN_DOC("collider-b", b), TOKEN_DOC("collider-a-different-bytes", a),
      TOKEN_DOC("near", e), TOKEN_DOC("partial", d),
      TOKEN_DOC("collider-a", b), BYTE_DOC("long-train"), BYTE_DOC("other"),
  };
  const uint32_t orders[] = {1, 2, 5, 8, 13, UINT32_MAX};
  uint64_t indices[COUNT(eval)];
  et_eval5_contamination_summary summary;
  for (size_t i = 0; i < COUNT(orders); ++i) {
    et_eval5_contamination_config config = {
        ET_EVAL5_NGRAM_TOKENS, orders[i], 0, 0.0, 1.0,
    };
    et_eval5_status status = et_eval5_audit_contamination(
        train, COUNT(train), eval, COUNT(eval), &config, indices, COUNT(indices), &summary);
    CHECK(status == ET_EVAL5_OK && summary.exact_doc_matches == 2);
    if (orders[i] == 2) {
      /* With base 257, [1,257] and [2,0] share the same uint64 key. */
      CHECK(summary.contaminated_documents == 5 && summary.ngram_contaminated_docs == 3);
      CHECK(indices[0] == 1 && indices[1] == 2 && indices[2] == 3 &&
            indices[3] == 4 && indices[4] == 5);
    }
    print_audit("token", config.ngram_order, status, &summary, indices);
  }

  const et_eval5_document both[] = {TOKEN_DOC("a", a), TOKEN_DOC("b", b)};
  const et_eval5_document checked[] = {TOKEN_DOC("check-a", a), TOKEN_DOC("check-b", b)};
  et_eval5_contamination_config config = {ET_EVAL5_NGRAM_TOKENS, 2, 0, 1.0, 1.0};
  CHECK(et_eval5_audit_contamination(both, COUNT(both), checked, COUNT(checked), &config,
                                    indices, COUNT(indices), &summary) == ET_EVAL5_OK);
  CHECK(summary.exact_doc_matches == 0 && summary.ngram_contaminated_docs == 2 &&
        summary.matched_eval_ngrams == 2 && indices[0] == 0 && indices[1] == 1);
}

static void test_invalid_audit(void) {
  const et_eval5_document document = BYTE_DOC("valid");
  const et_eval5_document invalid_bytes = {NULL, 1, NULL, 0};
  const et_eval5_document invalid_tokens = {NULL, 0, NULL, 1};
  const et_eval5_document huge_bytes = {document.bytes, SIZE_MAX, NULL, 0};
  const uint32_t token = 0;
  const et_eval5_document huge_tokens = {NULL, 0, &token, SIZE_MAX};
  et_eval5_contamination_config config = {ET_EVAL5_NGRAM_BYTES, 2, 0, 0.0, 1.0};
  et_eval5_contamination_summary summary = {.eval_documents = 99};
  uint64_t index = 99;

  CHECK(et_eval5_audit_contamination(NULL, 1, &document, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, NULL, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, 0, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, 1, NULL,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, 1, &config,
                                    NULL, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, 1, &config,
                                    &index, 1, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, 1, &config,
                                    &index, 0, &summary) == ET_EVAL5_ERR_BOUNDS);
  CHECK(et_eval5_audit_contamination(&document, SIZE_MAX, &document, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_OVERFLOW);
  CHECK(et_eval5_audit_contamination(&document, 1, &document, SIZE_MAX, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_OVERFLOW);
  CHECK(et_eval5_audit_contamination(&invalid_bytes, 1, &document, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_audit_contamination(&document, 1, &huge_bytes, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_OVERFLOW);
  config.ngram_mode = ET_EVAL5_NGRAM_TOKENS;
  CHECK(et_eval5_audit_contamination(NULL, 0, &huge_tokens, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_OVERFLOW);
  CHECK(et_eval5_audit_contamination(NULL, 0, &invalid_tokens, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);

  const et_eval5_contamination_config bad[] = {
      {0, 2, 0, 0.0, 1.0}, {ET_EVAL5_NGRAM_BYTES, 0, 0, 0.0, 1.0},
      {ET_EVAL5_NGRAM_BYTES, 2, 2, 0.0, 1.0}, {ET_EVAL5_NGRAM_BYTES, 2, 0, NAN, 1.0},
      {ET_EVAL5_NGRAM_BYTES, 2, 0, 0.0, INFINITY}, {ET_EVAL5_NGRAM_BYTES, 2, 0, 1.1, 1.0},
      {ET_EVAL5_NGRAM_BYTES, 2, 0, 0.0, -0.1},
  };
  for (size_t i = 0; i < COUNT(bad); ++i) {
    CHECK(et_eval5_audit_contamination(NULL, 0, &document, 1, &bad[i],
                                      &index, 1, &summary) == ET_EVAL5_ERR_ARGUMENT);
  }
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_eval5_audit_contamination(NULL, 0, &document, 1, &config,
                                    &index, 1, &summary) == ET_EVAL5_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);
  CHECK(index == 99 && summary.eval_documents == 99);
}

static void test_lm(et_eval5_lm_metrics *metrics) {
  static const float logits[] = {
      0.1f, -0.2f, 0.3f, -0.4f,
      -100.0f, 100.0f, 99.0f, -100.0f,
      -40.0f, 0.0f, -50.0f, -60.0f,
      -1.0f, -1.0f, -1.0f, -1.0f,
  };
  static const uint32_t targets[] = {2, 2, 1, 0};
  const et_eval5_sequence sequence = SEQUENCE(logits, targets, 4);
  CHECK(et_eval5_score_lm(&sequence, 31, metrics) == ET_EVAL5_OK);
  CHECK(metrics->token_count == 4 && metrics->byte_count == 31);
  printf("lm %" PRIu64 " %" PRIu64 " %a %a %a %a\n", metrics->token_count,
         metrics->byte_count, metrics->total_nll, metrics->mean_nll, metrics->perplexity,
         metrics->bits_per_byte);

  const float near_zero[] = {0, -40};
  const uint32_t zero[] = {0};
  const et_eval5_sequence near = SEQUENCE(near_zero, zero, 2);
  et_eval5_lm_metrics result;
  CHECK(et_eval5_score_lm(&near, 1, &result) == ET_EVAL5_OK);
  CHECK(result.total_nll == log1p(exp(-40.0)) && result.total_nll > 0 && result.perplexity == 1.0);
  const float uniform[] = {0, 0, 0, 0};
  const et_eval5_sequence equal = SEQUENCE(uniform, zero, 4);
  CHECK(et_eval5_score_lm(&equal, 2, &result) == ET_EVAL5_OK);
  CHECK(result.total_nll == log(4.0) && result.perplexity == 4.0 && result.bits_per_byte == 1.0);
  const float singleton[] = {-FLT_MAX};
  const et_eval5_sequence one = SEQUENCE(singleton, zero, 1);
  CHECK(et_eval5_score_lm(&one, 1, &result) == ET_EVAL5_OK);
  CHECK(result.total_nll == 0.0 && result.perplexity == 1.0 && result.bits_per_byte == 0.0);
}

static void test_mc(et_eval5_mc_metrics *normalized) {
  static const float short_logits[] = {0.0f, 0.0f};
  static const uint32_t short_target[] = {0};
  static const float long_logits[] = {0.0f, 0.2f, 0.0f, 0.2f};
  static const uint32_t long_target[] = {1, 1};
  static const et_eval5_sequence different[] = {
      SEQUENCE(short_logits, short_target, 2), SEQUENCE(long_logits, long_target, 2),
  };
  static const et_eval5_sequence tied[] = {
      SEQUENCE(short_logits, short_target, 2), SEQUENCE(short_logits, short_target, 2),
  };
  const et_eval5_choice_item items[] = {
      {different, COUNT(different), 1}, {tied, COUNT(tied), 0}, {tied, COUNT(tied), 1},
  };
  for (uint32_t mode = 0; mode <= 1; ++mode) {
    et_eval5_mc_metrics result;
    CHECK(et_eval5_score_mc(items, COUNT(items), mode, &result) == ET_EVAL5_OK);
    CHECK(result.correct_count == 1 + mode && result.total_count == COUNT(items));
    CHECK(result.accuracy == (double)(1 + mode) / COUNT(items));
    printf("mc %" PRIu32 " %" PRIu64 " %" PRIu64 " %a\n",
           mode, result.correct_count, result.total_count, result.accuracy);
    if (mode == 1) *normalized = result;
  }
}

static void test_invalid_metrics(void) {
  float logits[] = {0, 1};
  uint32_t targets[] = {0};
  et_eval5_sequence sequence = SEQUENCE(logits, targets, 2);
  et_eval5_lm_metrics lm = {.token_count = 99};
  et_eval5_mc_metrics mc = {.correct_count = 99};
  et_eval5_choice_item choice = {&sequence, 1, 0};

  CHECK(et_eval5_score_lm(NULL, 1, &lm) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_lm(&sequence, 0, &lm) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_lm(&sequence, 1, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_mc(NULL, 1, 0, &mc) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_mc(&choice, 0, 0, &mc) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_mc(&choice, 1, 2, &mc) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_mc(&choice, 1, 0, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_score_mc(&choice, SIZE_MAX, 0, &mc) == ET_EVAL5_ERR_OVERFLOW);

  et_eval5_sequence invalid[] = {
      {NULL, targets, 1, 2, 2}, {logits, NULL, 1, 2, 2},
      {logits, targets, 0, 2, 2}, {logits, targets, 1, 0, 2},
  };
  for (size_t i = 0; i < COUNT(invalid); ++i)
    CHECK(et_eval5_score_lm(&invalid[i], 1, &lm) == ET_EVAL5_ERR_ARGUMENT);
  sequence.logit_count = 1;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_BOUNDS);
  sequence.logit_count = 3;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_BOUNDS);
  sequence.logit_count = 2;
  sequence.token_count = SIZE_MAX;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_OVERFLOW);
  sequence.token_count = 1;
  sequence.vocab_size = SIZE_MAX;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_OVERFLOW);
  sequence.vocab_size = 2;
  targets[0] = 2;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_TARGET);
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_TARGET);
  targets[0] = 0;

  const float bad[] = {NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < COUNT(bad); ++i) {
    logits[1] = bad[i];
    CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_NONFINITE);
    CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_NONFINITE);
  }
  logits[1] = FLT_MAX;
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_OVERFLOW);
  logits[1] = 1000.0f;
  CHECK(feclearexcept(FE_OVERFLOW) == 0);
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_OVERFLOW);
  CHECK(fetestexcept(FE_OVERFLOW) == 0);
#if defined(__x86_64__)
  const unsigned saved_mxcsr = _mm_getcsr();
  const unsigned unmasked = saved_mxcsr & ~(_MM_MASK_OVERFLOW | _MM_EXCEPT_OVERFLOW);
  _mm_setcsr(unmasked);
  et_eval5_status lm_status = et_eval5_score_lm(&sequence, 1, &lm);
  et_eval5_status mc_status = et_eval5_score_mc(&choice, 1, 0, &mc);
  const unsigned after_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr);
  CHECK(lm_status == ET_EVAL5_ERR_FP_ENV && mc_status == ET_EVAL5_ERR_FP_ENV);
  CHECK(after_mxcsr == unmasked);
#endif
  logits[1] = 1.0f;

  choice.choice_count = 0;
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_ARGUMENT);
  choice.choice_count = SIZE_MAX;
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_OVERFLOW);
  choice.choice_count = 1;
  choice.correct_index = 1;
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_TARGET);
  choice.correct_index = 0;
  choice.choices = NULL;
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_ARGUMENT);
  choice.choices = &sequence;

  CHECK(fesetround(FE_DOWNWARD) == 0);
  CHECK(et_eval5_score_lm(&sequence, 1, &lm) == ET_EVAL5_ERR_FP_ENV);
  CHECK(et_eval5_score_mc(&choice, 1, 0, &mc) == ET_EVAL5_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);
  CHECK(lm.token_count == 99 && mc.correct_count == 99);
}

static void make_path(char *out, size_t capacity, const char *directory,
                      const char *name) {
  int length = snprintf(out, capacity, "%s/%s", directory, name);
  CHECK(length > 0 && (size_t)length < capacity);
}

static void replace_float_field(char *record, const char *key, double value) {
  char *field = strstr(record, key);
  CHECK(field != NULL);
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  char hex[17];
  CHECK(snprintf(hex, sizeof(hex), "%016" PRIx64, bits) == 16);
  memcpy(field + strlen(key), hex, 16);
}

static void test_invalid_lm_manifest(const et_eval5_manifest *manifest,
                                    const char *path) {
  const double ln2 = 0x1.62e42fefa39efp-1;
  const struct {
    et_eval5_lm_metrics lm;
    const char *checksum;
  } cases[] = {
      /* Digests independently computed with Python hashlib.sha256. */
      {{1, 1, 0, 0, 2, 1},
       "03333a702c729b18176ad580bcb2e6a77a12e8d7ca8dad0ad5ed783fa9264480"},
      {{1, 1, ln2, ln2, 2, 1.25},
       "cf3047503a5b0d5c585e90c71801421c81422b3042e961ddc0dd9abc7126342d"},
      {{1, 1, ln2, ln2, 3, 1},
       "1727079365b786559dde1e7052ee9850549fb1b62c08b55a65eb0989c6e68ec7"},
      {{1, 1, 1000, 1000, DBL_MAX, 0x1.68ac7b8cc7decp+10},
       "d2d17bfcc7f598ab232f2c1702c88563c442c232132373e4f668ba257579a12a"},
  };
  et_eval5_manifest record = *manifest;
  record.lm = (et_eval5_lm_metrics){1, 1, 0, 0, 1, 0};
  char canonical[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  char changed[sizeof(canonical)];
  size_t size;
  et_eval5_manifest parsed;
  CHECK(et_eval5_manifest_format(&record, canonical, sizeof(canonical), &size) == ET_EVAL5_OK);
  CHECK(et_eval5_manifest_parse(canonical, size, &parsed) == ET_EVAL5_OK);

  for (size_t i = 0; i < COUNT(cases); ++i) {
    record.lm = cases[i].lm;
    size_t rejected_size = 0;
    CHECK(et_eval5_manifest_format(&record, changed, sizeof(changed),
                                  &rejected_size) == ET_EVAL5_ERR_FORMAT);
    CHECK(et_eval5_manifest_write_atomic(path, &record) == ET_EVAL5_ERR_FORMAT);

    memcpy(changed, canonical, size + 1);
    replace_float_field(changed, "lm.total_nll=", record.lm.total_nll);
    replace_float_field(changed, "lm.mean_nll=", record.lm.mean_nll);
    replace_float_field(changed, "lm.perplexity=", record.lm.perplexity);
    replace_float_field(changed, "lm.bits_per_byte=", record.lm.bits_per_byte);
    CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_CHECKSUM);
    const char key[] = "manifest_sha256=";
    char *checksum = strstr(changed, key);
    CHECK(checksum != NULL);
    memcpy(checksum + sizeof(key) - 1, cases[i].checksum, 2 * ET_EVAL5_SHA256_BYTES);
    CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_FORMAT);
  }
}

static void test_manifest(et_eval5_manifest *manifest, const char *directory) {
  char canonical[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  char changed[sizeof(canonical)];
  size_t size = 0;
  size_t other_size = 0;

  CHECK(et_eval5_manifest_format(manifest, NULL, 0, &size) == ET_EVAL5_OK);
  CHECK(size > 0 && size <= ET_EVAL5_MANIFEST_MAX_BYTES);
  CHECK(et_eval5_manifest_format(manifest, canonical, sizeof(canonical), &other_size) == ET_EVAL5_OK);
  CHECK(size == other_size && strlen(canonical) == size);
  changed[0] = '!';
  CHECK(et_eval5_manifest_format(manifest, changed, size, &other_size) == ET_EVAL5_ERR_BOUNDS);
  CHECK(size == other_size && changed[0] == '!');
  CHECK(et_eval5_manifest_format(NULL, canonical, sizeof(canonical), &other_size) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_format(manifest, NULL, 1, &other_size) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_format(manifest, NULL, 0, NULL) == ET_EVAL5_ERR_ARGUMENT);

  et_eval5_manifest parsed = {.version_major = 99};
  for (size_t cut = 0; cut < size; ++cut)
    CHECK(et_eval5_manifest_parse(canonical, cut, &parsed) != ET_EVAL5_OK);
  CHECK(et_eval5_manifest_parse(canonical, sizeof(canonical), &parsed) == ET_EVAL5_ERR_BOUNDS);
  CHECK(et_eval5_manifest_parse(NULL, size, &parsed) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_parse(canonical, size, NULL) == ET_EVAL5_ERR_ARGUMENT);
  memcpy(changed, canonical, size + 1);
  strstr(changed, "suite=")[6] = 'z';
  CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_CHECKSUM);
  memcpy(changed, canonical, size + 1);
  strstr(changed, "version=")[8] = '2';
  CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_VERSION);
  memcpy(changed, canonical, size + 1);
  strstr(changed, "checkpoint_sha256=")[18] = 'G';
  CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_FORMAT);
  memcpy(changed, canonical, size + 1);
  strstr(changed, "manifest_sha256=")[16] = ':';
  CHECK(et_eval5_manifest_parse(changed, size, &parsed) == ET_EVAL5_ERR_FORMAT);
  memcpy(changed, canonical, size + 1);
  changed[size] = '\n';
  CHECK(et_eval5_manifest_parse(changed, size + 1, &parsed) == ET_EVAL5_ERR_FORMAT);
  CHECK(parsed.version_major == 99);

  CHECK(et_eval5_manifest_parse(canonical, size, &parsed) == ET_EVAL5_OK);
  CHECK(strcmp(parsed.suite, manifest->suite) == 0 && parsed.mc.accuracy == manifest->mc.accuracy);
  CHECK(memcmp(parsed.manifest_sha256, manifest->manifest_sha256, ET_EVAL5_SHA256_BYTES) != 0);
  CHECK(et_eval5_manifest_format(&parsed, changed, sizeof(changed), &other_size) == ET_EVAL5_OK);
  CHECK(size == other_size && memcmp(canonical, changed, size + 1) == 0);

  et_eval5_manifest invalid = *manifest;
  invalid.version_minor = 1;
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_VERSION);
  invalid = *manifest;
  memset(invalid.task, 'a', sizeof(invalid.task));
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_FORMAT);
  invalid.task[ET_EVAL5_NAME_MAX] = '\0';
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_OK);
  invalid = *manifest;
  invalid.audit.ngram_contaminated_docs = 0;
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_FORMAT);
  invalid = *manifest;
  invalid.mc.accuracy = -0.0;
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_FORMAT);
  invalid = *manifest;
  invalid.lm.mean_nll = nextafter(invalid.lm.mean_nll, INFINITY);
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_FORMAT);
  invalid = *manifest;
  invalid.metric_mask = 4;
  CHECK(et_eval5_manifest_format(&invalid, NULL, 0, &other_size) == ET_EVAL5_ERR_FORMAT);
  invalid = *manifest;
  invalid.metric_mask = ET_EVAL5_METRIC_LM;
  invalid.mc = (et_eval5_mc_metrics){0};
  invalid.mc_normalize_by_length = 0;
  CHECK(et_eval5_manifest_format(&invalid, changed, sizeof(changed), &other_size) == ET_EVAL5_OK);
  CHECK(et_eval5_manifest_parse(changed, other_size, &parsed) == ET_EVAL5_OK);
  CHECK(parsed.metric_mask == ET_EVAL5_METRIC_LM && parsed.mc.total_count == 0);

  invalid = *manifest;
  invalid.metric_mask = ET_EVAL5_METRIC_MC;
  invalid.lm = (et_eval5_lm_metrics){0};
  invalid.contamination.forbid_exact_match = 1;
  CHECK(et_eval5_manifest_format(&invalid, changed, sizeof(changed), &other_size) == ET_EVAL5_OK);
  CHECK(et_eval5_manifest_parse(changed, other_size, &parsed) == ET_EVAL5_OK);
  CHECK(parsed.metric_mask == ET_EVAL5_METRIC_MC && parsed.lm.token_count == 0 &&
        parsed.contamination.forbid_exact_match == 1 && parsed.audit.exact_doc_matches != 0);

  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_eval5_manifest_format(manifest, NULL, 0, &other_size) == ET_EVAL5_ERR_FP_ENV);
  CHECK(et_eval5_manifest_parse(canonical, size, &parsed) == ET_EVAL5_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);

  char path[4096];
  char alternate[4096];
  make_path(path, sizeof(path), directory, "benchmark.manifest");
  CHECK(et_eval5_manifest_write_atomic(NULL, manifest) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_write_atomic(path, NULL) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_write_atomic(path, manifest) == ET_EVAL5_OK);
  int first = open(path, O_RDONLY);
  CHECK(first >= 0);
  struct stat original;
  CHECK(fstat(first, &original) == 0 && (original.st_mode & 077) == 0);
  CHECK(et_eval5_manifest_write_atomic(path, manifest) == ET_EVAL5_OK);
  struct stat replacement;
  CHECK(stat(path, &replacement) == 0 && original.st_ino != replacement.st_ino);
  CHECK(read(first, changed, size) == (ssize_t)size && memcmp(changed, canonical, size) == 0);
  CHECK(close(first) == 0);

  invalid = *manifest;
  invalid.context_length = 0;
  CHECK(et_eval5_manifest_write_atomic(path, &invalid) == ET_EVAL5_ERR_FORMAT);
  test_invalid_lm_manifest(manifest, path);
  CHECK(et_eval5_manifest_read_file(path, &parsed) == ET_EVAL5_OK);
  CHECK(et_eval5_manifest_format(&parsed, changed, sizeof(changed), &other_size) == ET_EVAL5_OK);
  CHECK(size == other_size && memcmp(canonical, changed, size + 1) == 0);
  CHECK(et_eval5_manifest_read_file(NULL, &parsed) == ET_EVAL5_ERR_ARGUMENT);
  CHECK(et_eval5_manifest_read_file(path, NULL) == ET_EVAL5_ERR_ARGUMENT);

  make_path(alternate, sizeof(alternate), directory, "link");
  CHECK(symlink("benchmark.manifest", alternate) == 0);
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_IO);
  CHECK(unlink(alternate) == 0);
  CHECK(symlink("missing.manifest", alternate) == 0);
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_IO);
  CHECK(unlink(alternate) == 0);
  CHECK(mkfifo(alternate, 0600) == 0);
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_IO);
  CHECK(unlink(alternate) == 0);
  CHECK(mkdir(alternate, 0700) == 0);
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_IO);
  CHECK(et_eval5_manifest_write_atomic(alternate, manifest) == ET_EVAL5_ERR_IO);
  CHECK(rmdir(alternate) == 0);

  int malformed = open(alternate, O_CREAT | O_EXCL | O_WRONLY, 0600);
  CHECK(malformed >= 0);
  CHECK(write(malformed, canonical, size - 1) == (ssize_t)(size - 1));
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_FORMAT);
  CHECK(ftruncate(malformed, ET_EVAL5_MANIFEST_MAX_BYTES + 1) == 0);
  CHECK(close(malformed) == 0);
  CHECK(et_eval5_manifest_read_file(alternate, &parsed) == ET_EVAL5_ERR_BOUNDS);
  CHECK(unlink(alternate) == 0);

  DIR *entries = opendir(directory);
  CHECK(entries != NULL);
  for (struct dirent *entry = readdir(entries); entry != NULL; entry = readdir(entries))
    CHECK(strncmp(entry->d_name, ".eval5-manifest.", 16) != 0);
  CHECK(closedir(entries) == 0);
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "--read-status") == 0) {
    et_eval5_manifest manifest;
    printf("%" PRId32 "\n", et_eval5_manifest_read_file(argv[2], &manifest));
    return 0;
  }
  CHECK(argc == 2);
  et_eval5_manifest manifest = {
      .version_major = ET_EVAL5_VERSION_MAJOR, .version_minor = ET_EVAL5_VERSION_MINOR,
      .suite = "fixture-suite", .task = "lm-and-mc", .split = "validation.v1",
      .context_length = 2048, .max_samples = 100,
      .metric_mask = ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC,
      .mc_normalize_by_length = 1,
  };
  for (size_t i = 0; i < ET_EVAL5_SHA256_BYTES; ++i) {
    manifest.train_dataset_sha256[i] = (uint8_t)i;
    manifest.eval_dataset_sha256[i] = (uint8_t)(0x20u + i);
    manifest.tokenizer_sha256[i] = (uint8_t)(0x40u + i);
    manifest.checkpoint_sha256[i] = (uint8_t)(0x60u + i);
  }

  test_byte_audit(&manifest);
  test_token_audit();
  test_invalid_audit();
  test_lm(&manifest.lm);
  test_mc(&manifest.mc);
  test_invalid_metrics();
  test_manifest(&manifest, argv[1]);
  return 0;
}
