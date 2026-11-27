#include "mod5_v256_manifest_bridge.h"
#include "mod5_v256_score_bridge.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
  if (!(condition)) { fprintf(stderr, "check failed at line %d\n", __LINE__); return 1; } \
} while (0)

static int nibble(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  return -1;
}

static int digest(const char hex[65], uint8_t out[32]) {
  if (strlen(hex) != 64u) return 0;
  for (size_t i = 0; i < 32u; ++i) {
    int high = nibble(hex[2u * i]), low = nibble(hex[2u * i + 1u]);
    if (high < 0 || low < 0) return 0;
    out[i] = (uint8_t)(16 * high + low);
  }
  return 1;
}

static int32_t compare_runner(
    const uint8_t fingerprints[288], const uint8_t digests[96],
    const uint8_t documents[36], const uint32_t train_tokens[18],
    const uint32_t eval_tokens[18], const float logits[4][8 * 256],
    const char *expected, size_t size) {
  return et_eval5_v256_compare_runner(
      fingerprints, digests, documents, train_tokens, eval_tokens,
      logits[0], logits[1], logits[2], logits[3], expected, size);
}

static int test_runner(et_eval5_v256_manifest_input input, uint8_t digests[96],
                       const uint8_t train_bytes[18], const uint8_t eval_bytes[18],
                       const uint32_t train_tokens[18], const uint32_t eval_tokens[18]) {
  static const float logits[4][8 * 256] = {
      {[200] = 0.125f}, {[15] = -0.25f}, {[8 * 256 - 1] = 0.5f}, {[0] = -1.0f},
  };
  float probes[8];
  CHECK(et_eval5_v256_score(logits[0], logits[1], logits[2], logits[3], eval_tokens,
                           input.metrics, input.mc_counts, probes) == ET_EVAL5_OK);
  char expected[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t size = 0;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, expected, sizeof(expected),
                                         &size) == ET_EVAL5_OK);

  uint8_t fingerprints[288], documents[36];
  for (size_t i = 0; i < 3u; ++i)
    memcpy(fingerprints + 96u * i, input.tokenizer_fingerprint, 96u);
  memcpy(documents, train_bytes, 18u);
  memcpy(documents + 18u, eval_bytes, 18u);
  CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                       logits, expected, size) == ET_EVAL5_OK);

  for (size_t i = 0; i < 3u; ++i) {
    fingerprints[96u * i + 95u] = '5';
    CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                         logits, expected, size) == ET_EVAL5_ERR_IDENTITY);
    fingerprints[96u * i + 95u] = '4';
    digests[32u * i] ^= 1u;
    CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                         logits, expected, size) == ET_EVAL5_ERR_FORMAT);
    digests[32u * i] ^= 1u;
  }
  CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                       logits, expected, size - 1u) == ET_EVAL5_ERR_FORMAT);
  expected[size] = '!';
  CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                       logits, expected, size) == ET_EVAL5_ERR_FORMAT);
  expected[size] = '\0';
  documents[18] ^= 1u;
  CHECK(compare_runner(fingerprints, digests, documents, train_tokens, eval_tokens,
                       logits, expected, size) == ET_EVAL5_ERR_IDENTITY);
  memcpy(documents + 18u, train_bytes, 18u);
  CHECK(compare_runner(fingerprints, digests, documents, train_tokens, train_tokens,
                       logits, expected, size) == ET_EVAL5_ERR_CONTAMINATED);
  return 0;
}

int main(int argc, char **argv) {
  static const char fingerprint[] =
      "sha256:eshkol-byte-tokenizer-v1:"
      "aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704";
  static const uint8_t train_bytes[18] = {
      1, 4, 1, 5, 9, 2, 6, 5, 3, 2, 7, 1, 8, 2, 8, 1, 8, 2};
  static const uint8_t eval_bytes[18] = {
      200, 3, 10, 11, 12, 13, 14, 15, 201,
      201, 15, 14, 13, 12, 11, 10, 3, 200};
  uint32_t train_tokens[18], eval_tokens[18];
  for (size_t i = 0; i < 18u; ++i) {
    train_tokens[i] = train_bytes[i];
    eval_tokens[i] = eval_bytes[i];
  }
  et_eval5_v256_manifest_input input = {0};
  input.tokenizer_fingerprint = (const uint8_t *)fingerprint;
  input.tokenizer_length = sizeof(fingerprint) - 1u;
  input.corpus_fingerprint = (const uint8_t *)fingerprint;
  input.corpus_length = sizeof(fingerprint) - 1u;
  CHECK(digest("71e4c68f07866a317aa091575ef555ecda37c04347381dbc795adcb5371023b6",
               input.train_manifest_sha256));
  CHECK(digest("b2e31837e5094b79d95c5ac30a799a54e1c117f60e12511d3853f5ef5bfd69e2",
               input.eval_manifest_sha256));
  CHECK(digest("505358be27f676998816c9d2ed194358edb017be4ccb2050069c1e22d9279eae",
               input.checkpoint_sha256));
  CHECK(et_eval5_v256_audit(train_bytes, train_tokens, eval_bytes, eval_tokens,
                             input.audit_counts) == ET_EVAL5_OK);
  /* Fixed values from the sealed ab088449 development AOT receipt. This C
   * check is a formatter test, not a new compiled-model execution. */
  input.metrics[0] = 90.780767317731659;
  input.metrics[1] = 5.6737979573582287;
  input.metrics[2] = 291.13816763515604;
  input.metrics[3] = 8.1855601760866463;
  input.metrics[4] = 0.5;
  input.mc_counts[0] = 2;
  input.mc_counts[1] = 4;

  char formatted[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  size_t size = 0;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, sizeof(formatted),
                                          &size) == ET_EVAL5_OK);
  CHECK(size > 0u && size <= ET_EVAL5_MANIFEST_MAX_BYTES);
  et_eval5_manifest parsed;
  CHECK(et_eval5_manifest_parse(formatted, size, &parsed) == ET_EVAL5_OK);
  CHECK(parsed.version_major == 1u && parsed.version_minor == 0u);
  CHECK(strcmp(parsed.suite, "mod5-v256-development") == 0);
  CHECK(parsed.metric_mask == (ET_EVAL5_METRIC_LM | ET_EVAL5_METRIC_MC));
  CHECK(parsed.lm.total_nll == input.metrics[0] && parsed.mc.correct_count == 2u);
  CHECK(parsed.audit.eval_documents == 2u && parsed.audit.matched_eval_ngrams == 0u);
  CHECK(memcmp(parsed.train_dataset_sha256, input.train_manifest_sha256, 32u) == 0);
  CHECK(memcmp(parsed.eval_dataset_sha256, input.eval_manifest_sha256, 32u) == 0);
  CHECK(memcmp(parsed.checkpoint_sha256, input.checkpoint_sha256, 32u) == 0);
  uint8_t expected_tokenizer[32];
  CHECK(digest("aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704",
               expected_tokenizer));
  CHECK(memcmp(parsed.tokenizer_sha256, expected_tokenizer, 32u) == 0);
  if (argc == 2 && strcmp(argv[1], "--emit") == 0) {
    CHECK(fwrite(formatted, 1u, size, stdout) == size);
    return 0;
  }

  uint8_t digests[96];
  memcpy(digests, input.train_manifest_sha256, 32u);
  memcpy(digests + 32u, input.eval_manifest_sha256, 32u);
  memcpy(digests + 64u, input.checkpoint_sha256, 32u);
  CHECK(test_runner(input, digests, train_bytes, eval_bytes,
                    train_tokens, eval_tokens) == 0);
  int64_t audit_record[9] = {0};
  for (size_t i = 0; i < 8u; ++i)
    audit_record[i + 1u] = (int64_t)input.audit_counts[i];
  char live_output[ET_EVAL5_MANIFEST_MAX_BYTES + 1u];
  uint64_t live_size = 0;
  CHECK(et_eval5_v256_manifest_roundtrip_live(
      (const uint8_t *)fingerprint, (const uint8_t *)fingerprint,
      digests, audit_record, input.metrics, input.mc_counts,
      live_output, &live_size) == ET_EVAL5_OK);
  CHECK(live_size == size && memcmp(live_output, formatted, size + 1u) == 0);

  memset(live_output, '!', sizeof(live_output));
  live_size = 777u;
  audit_record[0] = ET_EVAL5_ERR_CONTAMINATED;
  CHECK(et_eval5_v256_manifest_roundtrip_live(
      (const uint8_t *)fingerprint, (const uint8_t *)fingerprint,
      digests, audit_record, input.metrics, input.mc_counts,
      live_output, &live_size) == ET_EVAL5_ERR_CONTAMINATED);
  for (size_t i = 0; i < sizeof(live_output); ++i) CHECK(live_output[i] == '!');
  CHECK(live_size == 777u);
  audit_record[0] = 0;

  char untouched[sizeof(formatted)];
  memset(untouched, '!', sizeof(untouched));
  memcpy(formatted, untouched, sizeof(formatted));
  size = 777u;
  input.audit_status = ET_EVAL5_ERR_CONTAMINATED;
  CHECK(et_eval5_v256_audit(train_bytes, train_tokens, train_bytes, train_tokens,
                             input.audit_counts) == ET_EVAL5_ERR_CONTAMINATED);
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, sizeof(formatted),
                                          &size) == ET_EVAL5_ERR_CONTAMINATED);
  CHECK(memcmp(formatted, untouched, sizeof(formatted)) == 0 && size == 777u);
  input.audit_status = ET_EVAL5_OK;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, sizeof(formatted),
                                          &size) == ET_EVAL5_ERR_CONTAMINATED);
  CHECK(memcmp(formatted, untouched, sizeof(formatted)) == 0 && size == 777u);
  CHECK(et_eval5_v256_audit(train_bytes, train_tokens, eval_bytes, eval_tokens,
                             input.audit_counts) == ET_EVAL5_OK);

  char wrong_corpus[sizeof(fingerprint)];
  memcpy(wrong_corpus, fingerprint, sizeof(fingerprint));
  wrong_corpus[sizeof(fingerprint) - 2u] = '5';
  input.corpus_fingerprint = (const uint8_t *)wrong_corpus;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, sizeof(formatted),
                                          &size) == ET_EVAL5_ERR_IDENTITY);
  CHECK(memcmp(formatted, untouched, sizeof(formatted)) == 0 && size == 777u);
  CHECK(et_eval5_v256_manifest_roundtrip_live(
      (const uint8_t *)fingerprint, (const uint8_t *)wrong_corpus,
      digests, audit_record, input.metrics, input.mc_counts,
      live_output, &live_size) == ET_EVAL5_ERR_IDENTITY);
  for (size_t i = 0; i < sizeof(live_output); ++i) CHECK(live_output[i] == '!');
  CHECK(live_size == 777u);
  input.corpus_fingerprint = (const uint8_t *)fingerprint;
  input.metrics[1] += 1.0;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, sizeof(formatted),
                                          &size) == ET_EVAL5_ERR_FORMAT);
  CHECK(memcmp(formatted, untouched, sizeof(formatted)) == 0 && size == 777u);
  input.metrics[1] -= 1.0;
  CHECK(et_eval5_v256_manifest_roundtrip(&input, formatted, 1u,
                                          &size) == ET_EVAL5_ERR_BOUNDS);
  CHECK(memcmp(formatted, untouched, sizeof(formatted)) == 0 && size == 777u);
  puts("EVAL5 V256 in-memory manifest round-trip and one-shot parity: PASS");
  return 0;
}
