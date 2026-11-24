#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/constrained_decode_abi.h"
#include "constrained_decode_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* ---- Allocation Trap for Hot-Path Zero-Allocation Verification ---- */

static int hot_path_active = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!hot_path_active);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!hot_path_active);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!hot_path_active);
  __real_free(ptr);
}

static uint32_t f32_bits(float v) {
  uint32_t b = 0u;
  memcpy(&b, &v, sizeof(b));
  return b;
}

static uint32_t accepts_str(const et_cd_dfa *dfa, const char *s) {
  uint32_t acc = 99u;
  hot_path_active = 1;
  CHECK(et_cd_dfa_accepts_bytes(dfa, (const uint8_t *)s, strlen(s), &acc) ==
        ET_CD_OK);
  hot_path_active = 0;
  return acc;
}

/* ---- Test 1: Raw DFA Co-Accessibility Pruning and Dead-End Rejection ---- */

static void test_coaccessibility_pruning(void) {
  /* 6-state DFA:
   *   0 (start) --'a'--> 1 (live) --'b'--> 2 (accept)
   *   0 --'x'--> 3 (dead-end trap) --'y'--> 4 (trap loop -> 3)
   *   1 --'z'--> 3 (branch into dead trap -> must be pruned!)
   *   5 (unreachable accept state, still co-accessible by itself)
   */
  uint16_t trans[6u * ET_CD_ALPHABET_SIZE];
  memset(trans, 0xFF, sizeof(trans));
  trans[0u * 256u + (uint8_t)'a'] = 1u;
  trans[1u * 256u + (uint8_t)'b'] = 2u;
  trans[0u * 256u + (uint8_t)'x'] = 3u;
  trans[3u * 256u + (uint8_t)'y'] = 4u;
  trans[4u * 256u + (uint8_t)'y'] = 3u;
  trans[1u * 256u + (uint8_t)'z'] = 3u;

  const uint16_t accepts[2] = {2u, 5u};
  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_build_raw(6u, 0u, accepts, 2u, trans, &dfa) == ET_CD_OK);
  CHECK(dfa != NULL);

  et_cd_dfa_info info;
  hot_path_active = 1;
  CHECK(et_cd_dfa_info_get(dfa, &info) == ET_CD_OK);
  CHECK(info.num_states == 6u);
  CHECK(info.start_state == 0u);
  CHECK(info.live_state_count == 4u); /* {0, 1, 2, 5} */
  CHECK(info.accepting_state_count == 2u);

  CHECK(et_cd_dfa_is_live(dfa, 0u) == 1u);
  CHECK(et_cd_dfa_is_live(dfa, 1u) == 1u);
  CHECK(et_cd_dfa_is_live(dfa, 2u) == 1u);
  CHECK(et_cd_dfa_is_live(dfa, 3u) == 0u);
  CHECK(et_cd_dfa_is_live(dfa, 4u) == 0u);
  CHECK(et_cd_dfa_is_live(dfa, 5u) == 1u);

  /* Transitions into dead states 3 and 4 MUST be pruned to ET_CD_DEAD_STATE! */
  CHECK(et_cd_dfa_next_state(dfa, 0u, (uint8_t)'a') == 1u);
  CHECK(et_cd_dfa_next_state(dfa, 1u, (uint8_t)'b') == 2u);
  CHECK(et_cd_dfa_next_state(dfa, 0u, (uint8_t)'x') == ET_CD_DEAD_STATE);
  CHECK(et_cd_dfa_next_state(dfa, 1u, (uint8_t)'z') == ET_CD_DEAD_STATE);
  CHECK(et_cd_dfa_next_state(dfa, 3u, (uint8_t)'y') == ET_CD_DEAD_STATE);
  CHECK(et_cd_dfa_next_state(dfa, 4u, (uint8_t)'y') == ET_CD_DEAD_STATE);
  hot_path_active = 0;

  CHECK(accepts_str(dfa, "ab") == 1u);
  CHECK(accepts_str(dfa, "a") == 0u);
  CHECK(accepts_str(dfa, "xy") == 0u);
  et_cd_dfa_destroy(dfa);

  /* Empty language: start state 0 cannot reach accept state 2 -> reject! */
  memset(trans, 0xFF, sizeof(trans));
  trans[0u * 256u + (uint8_t)'a'] = 1u;
  trans[1u * 256u + (uint8_t)'b'] = 0u;
  const uint16_t unreach_accept[1] = {2u};
  et_cd_dfa *empty_dfa = (et_cd_dfa *)(uintptr_t)0x1;
  CHECK(et_cd_dfa_build_raw(3u, 0u, unreach_accept, 1u, trans, &empty_dfa) ==
        ET_CD_ERR_EMPTY_LANGUAGE);
  CHECK(empty_dfa == (et_cd_dfa *)(uintptr_t)0x1); /* untouched on error */

  /* Zero accepting states -> ET_CD_ERR_EMPTY_LANGUAGE. */
  CHECK(et_cd_dfa_build_raw(3u, 0u, NULL, 0u, trans, &empty_dfa) ==
        ET_CD_ERR_EMPTY_LANGUAGE);

  /* Out-of-bounds transition target -> ET_CD_ERR_STATE. */
  trans[0u * 256u + (uint8_t)'c'] = 3u;
  CHECK(et_cd_dfa_build_raw(3u, 0u, unreach_accept, 1u, trans, &empty_dfa) ==
        ET_CD_ERR_STATE);
}

/* ---- Test 2: Literal Choices Compiler ---- */

static void test_literal_choices(void) {
  const uint8_t *choices[4] = {(const uint8_t *)"yes", (const uint8_t *)"no",
                               (const uint8_t *)"none",
                               (const uint8_t *)"maybe"};
  const size_t lens[4] = {3u, 2u, 4u, 5u};

  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_compile_literal_choices(choices, lens, 4u, &dfa) == ET_CD_OK);
  CHECK(accepts_str(dfa, "yes") == 1u);
  CHECK(accepts_str(dfa, "no") == 1u);
  CHECK(accepts_str(dfa, "none") == 1u);
  CHECK(accepts_str(dfa, "maybe") == 1u);

  CHECK(accepts_str(dfa, "") == 0u);
  CHECK(accepts_str(dfa, "y") == 0u);
  CHECK(accepts_str(dfa, "n") == 0u);
  CHECK(accepts_str(dfa, "non") == 0u);
  CHECK(accepts_str(dfa, "none!") == 0u);
  CHECK(accepts_str(dfa, "YES") == 0u);
  et_cd_dfa_destroy(dfa);

  /* Negative schema tests: empty list, empty choice, duplicate choice. */
  et_cd_dfa *bad = NULL;
  CHECK(et_cd_dfa_compile_literal_choices(choices, lens, 0u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const size_t zero_lens[1] = {0u};
  CHECK(et_cd_dfa_compile_literal_choices(choices, zero_lens, 1u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const uint8_t *dups[2] = {(const uint8_t *)"yes", (const uint8_t *)"yes"};
  const size_t dup_lens[2] = {3u, 3u};
  CHECK(et_cd_dfa_compile_literal_choices(dups, dup_lens, 2u, &bad) ==
        ET_CD_ERR_SCHEMA);
}

/* ---- Test 3: Flat JSON Object Schema Compiler ---- */

static void test_flat_json_schema(void) {
  const char *enums[4] = {"yes", "no", "none", "line1\n\"q\""};
  const et_cd_json_field_spec fields[4] = {
      {"answer", ET_CD_JSON_ENUM, enums, 4u},
      {"score", ET_CD_JSON_INTEGER, NULL, 0u},
      {"ok", ET_CD_JSON_BOOLEAN, NULL, 0u},
      {"note", ET_CD_JSON_STRING, NULL, 0u},
  };

  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_compile_flat_json_schema(fields, 4u, &dfa) == ET_CD_OK);

  /* Valid instances covering every type, escaping, UTF-8, negative/zero ints. */
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":0,\"ok\":true,\"note\":\"\"}") ==
        1u);
  CHECK(accepts_str(dfa, "{\"answer\":\"no\",\"score\":-0,\"ok\":false,\"note\":\"hi\\n\\t\\\"\\\\\\/\\u0041\"}") ==
        1u);
  CHECK(accepts_str(dfa, "{\"answer\":\"none\",\"score\":-987654,\"ok\":true,\"note\":\"caf\\u00e9 \xC3\xA9 \xE2\x82\xAC \xF0\x9F\x98\x80\"}") ==
        1u);
  CHECK(accepts_str(dfa, "{\"answer\":\"line1\\n\\\"q\\\"\",\"score\":42,\"ok\":false,\"note\":\"ok\"}") ==
        1u);

  /* Invalid JSON instances MUST be rejected. */
  CHECK(accepts_str(dfa, "{\"answer\":\"bad\",\"score\":0,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":01,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":-01,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":-,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":+5,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1.5,\"ok\":true,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":True,\"note\":\"\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"bad\nesc\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\\x41\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\\u123\"}") ==
        0u);
  /* Invalid UTF-8 inside string: lone continuation 0x80, overlong 0xC0 0x80, surrogate 0xED 0xA0 0x80. */
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\x80\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\xC0\x80\"}") ==
        0u);
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\xED\xA0\x80\"}") ==
        0u);
  /* Extra trailing byte or wrong field order. */
  CHECK(accepts_str(dfa, "{\"answer\":\"yes\",\"score\":1,\"ok\":true,\"note\":\"\"} ") ==
        0u);
  CHECK(accepts_str(dfa, "{\"score\":1,\"answer\":\"yes\",\"ok\":true,\"note\":\"\"}") ==
        0u);

  et_cd_dfa_destroy(dfa);

  /* Malformed schema specifications MUST be rejected. */
  et_cd_dfa *bad = NULL;
  CHECK(et_cd_dfa_compile_flat_json_schema(fields, 0u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const et_cd_json_field_spec dup_keys[2] = {
      {"x", ET_CD_JSON_INTEGER, NULL, 0u},
      {"x", ET_CD_JSON_BOOLEAN, NULL, 0u},
  };
  CHECK(et_cd_dfa_compile_flat_json_schema(dup_keys, 2u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const et_cd_json_field_spec bad_key[1] = {
      {"bad\"key", ET_CD_JSON_INTEGER, NULL, 0u},
  };
  CHECK(et_cd_dfa_compile_flat_json_schema(bad_key, 1u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const et_cd_json_field_spec bad_enum_empty[1] = {
      {"k", ET_CD_JSON_ENUM, NULL, 0u},
  };
  CHECK(et_cd_dfa_compile_flat_json_schema(bad_enum_empty, 1u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const et_cd_json_field_spec bad_type[1] = {
      {"k", 99, NULL, 0u},
  };
  CHECK(et_cd_dfa_compile_flat_json_schema(bad_type, 1u, &bad) ==
        ET_CD_ERR_SCHEMA);
  const char *bad_utf8_enum[1] = {"\xED\xA0\x80"};
  const et_cd_json_field_spec bad_enum_utf8[1] = {
      {"k", ET_CD_JSON_ENUM, bad_utf8_enum, 1u},
  };
  CHECK(et_cd_dfa_compile_flat_json_schema(bad_enum_utf8, 1u, &bad) ==
        ET_CD_ERR_SCHEMA);
}

/* ---- Test 4: Vocab Index, Zero-Alloc Hot Path, Bit-Exactness, Errors ---- */

static void test_hot_path_and_negatives(void) {
  const uint8_t *choices[2] = {(const uint8_t *)"yes", (const uint8_t *)"no"};
  const size_t lens[2] = {3u, 2u};
  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_compile_literal_choices(choices, lens, 2u, &dfa) == ET_CD_OK);

  /* Vocab: 0="y", 1="es", 2="no", 3="bad", 4="" (empty non-EOS -> disallowed), 5=EOS. */
  const uint8_t *toks[6] = {(const uint8_t *)"y", (const uint8_t *)"es",
                            (const uint8_t *)"no", (const uint8_t *)"bad",
                            (const uint8_t *)"", NULL};
  const size_t tlens[6] = {1u, 2u, 2u, 3u, 0u, 0u};

  et_cd_vocab_index *idx = NULL;
  CHECK(et_cd_vocab_index_build(dfa, toks, tlens, 6u, 5u, &idx) == ET_CD_OK);

  et_cd_vocab_index_info vinfo;
  CHECK(et_cd_vocab_index_info_get(idx, &vinfo) == ET_CD_OK);
  CHECK(vinfo.vocab_size == 6u && vinfo.eos_token_id == 5u &&
        vinfo.mask_words == 1u);

  /* Include -0.0f to verify bit-exact sign-bit preservation! */
  float neg_zero = -0.0f;
  float logits[6] = {neg_zero, 5.5f, -1.25f, 100.0f, 50.0f, 99.0f};
  float masked[6] = {0};

  /* Enter zero-allocation zone! Any malloc/calloc/realloc/free fails the test! */
  hot_path_active = 1;

  uint16_t state = vinfo.start_state;
  CHECK(et_cd_apply_logit_mask_v1(idx, state, logits, masked) == ET_CD_OK);

  /* At start_state: only "y" (0) and "no" (2) are allowed. */
  CHECK(f32_bits(masked[0]) == f32_bits(neg_zero));
  CHECK(f32_bits(masked[1]) == UINT32_C(0xFF800000));
  CHECK(f32_bits(masked[2]) == f32_bits(logits[2]));
  CHECK(f32_bits(masked[3]) == UINT32_C(0xFF800000));
  CHECK(f32_bits(masked[4]) == UINT32_C(0xFF800000));
  CHECK(f32_bits(masked[5]) == UINT32_C(0xFF800000)); /* EOS disallowed! */

  /* In-place masking matches bit-for-bit. */
  float inplace[6];
  memcpy(inplace, logits, sizeof(logits));
  CHECK(et_cd_apply_logit_mask_v1(idx, state, inplace, inplace) == ET_CD_OK);
  CHECK(memcmp(inplace, masked, sizeof(masked)) == 0);

  /* Partial overlap rejected with ET_CD_ERR_ALIASING. */
  float overlap_buf[7] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  CHECK(et_cd_apply_logit_mask_v1(idx, state, overlap_buf, overlap_buf + 1) ==
        ET_CD_ERR_ALIASING);

  /* Illegal token advance rejected; outputs untouched. */
  uint16_t next_s = 0x7777u;
  uint32_t fin = 42u;
  CHECK(et_cd_advance_token_v1(idx, state, 3u, &next_s, &fin) ==
        ET_CD_ERR_ILLEGAL_TOKEN);
  CHECK(next_s == 0x7777u && fin == 42u);
  CHECK(et_cd_advance_token_v1(idx, state, 5u, &next_s, &fin) ==
        ET_CD_ERR_ILLEGAL_TOKEN);
  CHECK(next_s == 0x7777u && fin == 42u);

  /* Advance "y" (0), then "es" (1), then EOS (5). */
  CHECK(et_cd_advance_token_v1(idx, state, 0u, &next_s, &fin) == ET_CD_OK);
  CHECK(fin == 0u && next_s != ET_CD_DEAD_STATE);
  state = next_s;

  CHECK(et_cd_apply_logit_mask_v1(idx, state, logits, masked) == ET_CD_OK);
  CHECK(f32_bits(masked[1]) == f32_bits(logits[1])); /* only "es" allowed */
  CHECK(f32_bits(masked[0]) == UINT32_C(0xFF800000));
  CHECK(f32_bits(masked[5]) == UINT32_C(0xFF800000));

  CHECK(et_cd_advance_token_v1(idx, state, 1u, &next_s, &fin) == ET_CD_OK);
  CHECK(fin == 0u);
  state = next_s;

  /* Now at accepting state: ONLY EOS (5) allowed! */
  CHECK(et_cd_apply_logit_mask_v1(idx, state, logits, masked) == ET_CD_OK);
  for (uint32_t t = 0u; t < 5u; ++t)
    CHECK(f32_bits(masked[t]) == UINT32_C(0xFF800000));
  CHECK(f32_bits(masked[5]) == f32_bits(logits[5]));

  CHECK(et_cd_advance_token_v1(idx, state, 5u, &next_s, &fin) == ET_CD_OK);
  CHECK(fin == 1u && next_s == ET_CD_DEAD_STATE);

  /* Any further mask or advance after EOS fails with ET_CD_ERR_STATE! */
  CHECK(et_cd_apply_logit_mask_v1(idx, next_s, logits, masked) ==
        ET_CD_ERR_STATE);
  CHECK(et_cd_advance_token_v1(idx, next_s, 0u, &next_s, &fin) ==
        ET_CD_ERR_STATE);

  /* Non-finite logit (NaN, +Inf, -Inf) fails with ET_CD_ERR_NONFINITE and
   * preserves masked_logits untouched. */
  float sentinel[6] = {1.25f, 2.25f, 3.25f, 4.25f, 5.25f, 6.25f};
  float bad_logits[6];
  memcpy(bad_logits, logits, sizeof(logits));
  const uint32_t nonfin_bits[3] = {UINT32_C(0x7FC00001), UINT32_C(0x7F800000),
                                   UINT32_C(0xFF800000)};
  for (size_t k = 0u; k < 3u; ++k) {
    memcpy(bad_logits, logits, sizeof(logits));
    memcpy(&bad_logits[5], &nonfin_bits[k], sizeof(float));
    float out_copy[6];
    memcpy(out_copy, sentinel, sizeof(sentinel));
    CHECK(et_cd_apply_logit_mask_v1(idx, vinfo.start_state, bad_logits,
                                    out_copy) == ET_CD_ERR_NONFINITE);
    CHECK(memcmp(out_copy, sentinel, sizeof(sentinel)) == 0);
  }

  /* Non-default FP rounding mode -> ET_CD_ERR_FP_ENV. */
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_cd_apply_logit_mask_v1(idx, vinfo.start_state, logits, masked) ==
        ET_CD_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);

  hot_path_active = 0;
  et_cd_vocab_index_destroy(idx);

  /* Empty allowed mask state: vocab only has "y" and EOS (missing "es" and "no"),
   * so after advancing "y", NO token is valid -> ET_CD_ERR_NO_VALID_TOKENS! */
  const uint8_t *partial_toks[2] = {(const uint8_t *)"y", NULL};
  const size_t partial_lens[2] = {1u, 0u};
  CHECK(et_cd_vocab_index_build(dfa, partial_toks, partial_lens, 2u, 1u,
                                &idx) == ET_CD_OK);
  hot_path_active = 1;
  uint16_t after_y = 0u;
  uint32_t after_fin = 0u;
  CHECK(et_cd_advance_token_v1(idx, vinfo.start_state, 0u, &after_y,
                               &after_fin) == ET_CD_OK);
  float small_logits[2] = {1.0f, 2.0f};
  float small_out[2] = {9.5f, -4.5f};
  CHECK(et_cd_apply_logit_mask_v1(idx, after_y, small_logits, small_out) ==
        ET_CD_ERR_NO_VALID_TOKENS);
  CHECK(small_out[0] == 9.5f && small_out[1] == -4.5f); /* preserved! */
  hot_path_active = 0;

  et_cd_vocab_index_destroy(idx);
  et_cd_dfa_destroy(dfa);
}

/* ---- Test 5: ETCD1000 v1.0 Checksummed Artifact Roundtrip & Tampers ---- */

static void test_artifact_io(void) {
  const char *enums[3] = {"yes", "no", "maybe"};
  const et_cd_json_field_spec fields[3] = {
      {"answer", ET_CD_JSON_ENUM, enums, 3u},
      {"score", ET_CD_JSON_INTEGER, NULL, 0u},
      {"ok", ET_CD_JSON_BOOLEAN, NULL, 0u},
  };
  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_compile_flat_json_schema(fields, 3u, &dfa) == ET_CD_OK);

  size_t need = 0u;
  CHECK(et_cd_dfa_artifact_size(dfa, &need) == ET_CD_OK);
  uint8_t *buf1 = (uint8_t *)malloc(need);
  uint8_t *buf2 = (uint8_t *)malloc(need);
  CHECK(buf1 != NULL && buf2 != NULL);

  size_t sz1 = 0u, sz2 = 0u;
  CHECK(et_cd_dfa_serialize(dfa, buf1, need, &sz1) == ET_CD_OK);
  CHECK(et_cd_dfa_serialize(dfa, buf2, need, &sz2) == ET_CD_OK);
  CHECK(sz1 == need && sz2 == need && memcmp(buf1, buf2, need) == 0);

  /* Atomic save + O_NOFOLLOW load roundtrip in temp dir. */
  char dir_tpl[] = "/tmp/etcd-test.XXXXXX";
  char *tdir = mkdtemp(dir_tpl);
  CHECK(tdir != NULL);
  char file_path[256];
  char link_path[256];
  snprintf(file_path, sizeof(file_path), "%s/grammar.etcd", tdir);
  snprintf(link_path, sizeof(link_path), "%s/grammar.link", tdir);

  CHECK(et_cd_dfa_save(file_path, dfa) == ET_CD_OK);
  CHECK(symlink(file_path, link_path) == 0);

  et_cd_dfa *reloaded = NULL;
  uint8_t sha_mem[ET_CD_SHA256_BYTES];
  uint8_t sha_disk[ET_CD_SHA256_BYTES];
  CHECK(et_cd_dfa_parse(buf1, sz1, &reloaded, sha_mem) == ET_CD_OK);
  et_cd_dfa_destroy(reloaded);
  reloaded = NULL;

  CHECK(et_cd_dfa_load(file_path, &reloaded, sha_disk) == ET_CD_OK);
  CHECK(memcmp(sha_mem, sha_disk, ET_CD_SHA256_BYTES) == 0);
  CHECK(accepts_str(reloaded, "{\"answer\":\"maybe\",\"score\":-7,\"ok\":false}") ==
        1u);
  et_cd_dfa_destroy(reloaded);

  /* Symlink rejection via O_NOFOLLOW. */
  et_cd_dfa *sym_dfa = NULL;
  CHECK(et_cd_dfa_load(link_path, &sym_dfa, NULL) == ET_CD_ERR_SYMLINK);
  CHECK(sym_dfa == NULL);

  /* Cleanup temp files. */
  CHECK(unlink(link_path) == 0);
  CHECK(unlink(file_path) == 0);
  CHECK(rmdir(tdir) == 0);

  /* Negative parse tests: bad magic, bad version, truncated, extra byte, checksum. */
  et_cd_dfa *bad = NULL;
  buf2[0] = 'X';
  CHECK(et_cd_dfa_parse(buf2, sz1, &bad, NULL) == ET_CD_ERR_FORMAT);
  memcpy(buf2, buf1, sz1);
  buf2[8] = 2u;
  CHECK(et_cd_dfa_parse(buf2, sz1, &bad, NULL) == ET_CD_ERR_VERSION);
  memcpy(buf2, buf1, sz1);
  CHECK(et_cd_dfa_parse(buf2, sz1 - 1u, &bad, NULL) == ET_CD_ERR_TRUNCATED);
  CHECK(et_cd_dfa_parse(buf2, 4u, &bad, NULL) == ET_CD_ERR_TRUNCATED);
  buf2[ET_CD_ARTIFACT_HEADER_BYTES] ^= 0x80u;
  CHECK(et_cd_dfa_parse(buf2, sz1, &bad, NULL) == ET_CD_ERR_CHECKSUM);

  uint8_t *extra = (uint8_t *)malloc(sz1 + 1u);
  CHECK(extra != NULL);
  memcpy(extra, buf1, sz1);
  extra[sz1] = 0u;
  CHECK(et_cd_dfa_parse(extra, sz1 + 1u, &bad, NULL) == ET_CD_ERR_FORMAT);
  free(extra);

  free(buf1);
  free(buf2);
  et_cd_dfa_destroy(dfa);
}

/* ---- Oracle Dump Modes for Independent Python Verification ---- */

static uint64_t xorshift64(uint64_t *state) {
  uint64_t x = *state;
  x ^= x << 13u;
  x ^= x >> 7u;
  x ^= x << 17u;
  *state = x;
  return x;
}

static int run_oracle_prune(void) {
  /* Build a 7-state DFA with dead cycles, dead branches, and live paths. */
  uint16_t trans[7u * ET_CD_ALPHABET_SIZE];
  memset(trans, 0xFF, sizeof(trans));
  trans[0u * 256u + (uint8_t)'a'] = 1u;
  trans[0u * 256u + (uint8_t)'d'] = 4u; /* dead branch */
  trans[1u * 256u + (uint8_t)'b'] = 2u; /* accept */
  trans[1u * 256u + (uint8_t)'c'] = 5u; /* dead cycle */
  trans[2u * 256u + (uint8_t)'a'] = 1u; /* loop back */
  trans[2u * 256u + (uint8_t)'x'] = 6u; /* dead sink */
  trans[4u * 256u + (uint8_t)'e'] = 5u;
  trans[5u * 256u + (uint8_t)'f'] = 4u;
  trans[3u * 256u + (uint8_t)'g'] = 2u; /* unreachable from 0, but co-accessible */

  const uint16_t accepts[1] = {2u};
  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_build_raw(7u, 0u, accepts, 1u, trans, &dfa) == ET_CD_OK);

  et_cd_dfa_info info;
  CHECK(et_cd_dfa_info_get(dfa, &info) == ET_CD_OK);
  printf("PRUNE_INFO %u %u %u %u\n", info.num_states, info.start_state,
         info.live_state_count, info.accepting_state_count);
  for (uint16_t s = 0u; s < info.num_states; ++s) {
    printf("STATE %u live=%u accept=%u", s, et_cd_dfa_is_live(dfa, s),
           et_cd_dfa_is_accepting(dfa, s));
    for (uint32_t b = 0u; b < 256u; ++b) {
      const uint16_t nxt = et_cd_dfa_next_state(dfa, s, (uint8_t)b);
      if (nxt != ET_CD_DEAD_STATE)
        printf(" %u:%u", b, nxt);
    }
    printf("\n");
  }
  et_cd_dfa_destroy(dfa);
  return 0;
}

static int run_oracle_json(uint64_t seed, uint32_t num_trajectories) {
  const char *answers[3] = {"yes", "no", "maybe"};
  const et_cd_json_field_spec fields[4] = {
      {"answer", ET_CD_JSON_ENUM, answers, 3u},
      {"score", ET_CD_JSON_INTEGER, NULL, 0u},
      {"ok", ET_CD_JSON_BOOLEAN, NULL, 0u},
      {"note", ET_CD_JSON_STRING, NULL, 0u},
  };
  et_cd_dfa *dfa = NULL;
  CHECK(et_cd_dfa_compile_flat_json_schema(fields, 4u, &dfa) == ET_CD_OK);

  /* Vocabulary: 95 printable ASCII bytes + 18 multi-byte/adversarial tokens + EOS = 114 tokens. */
  static const char *const extras[18] = {
      "{\"answer\":\"", "yes",          "no",        "maybe",
      "\",\"score\":", ",\"ok\":",     ",\"note\":\"", "true",
      "false",         "42",           "-105",      "\\n",
      "\\u0041",       "\"}\"",        "INVALID\n{{", "007",
      "\"\"broken",    "\xC3\xA9", /* UTF-8 'é' */
  };
  const uint32_t V = 95u + 18u + 1u;
  const uint32_t EOS = V - 1u;
  uint8_t ascii[95];
  const uint8_t *tptrs[114];
  size_t tlens[114];

  for (uint32_t i = 0u; i < 95u; ++i) {
    ascii[i] = (uint8_t)(0x20u + i);
    tptrs[i] = &ascii[i];
    tlens[i] = 1u;
  }
  for (uint32_t i = 0u; i < 18u; ++i) {
    tptrs[95u + i] = (const uint8_t *)extras[i];
    tlens[95u + i] = strlen(extras[i]);
  }
  tptrs[EOS] = NULL;
  tlens[EOS] = 0u;

  et_cd_vocab_index *idx = NULL;
  CHECK(et_cd_vocab_index_build(dfa, tptrs, tlens, V, EOS, &idx) == ET_CD_OK);

  et_cd_dfa_info dinfo;
  et_cd_vocab_index_info vinfo;
  CHECK(et_cd_dfa_info_get(dfa, &dinfo) == ET_CD_OK);
  CHECK(et_cd_vocab_index_info_get(idx, &vinfo) == ET_CD_OK);

  printf("META %u %u %u %u %u\n", dinfo.num_states, dinfo.start_state,
         dinfo.live_state_count, vinfo.vocab_size, vinfo.eos_token_id);

  /* Dump vocabulary in hex for unambiguous Python parsing. */
  for (uint32_t t = 0u; t < V; ++t) {
    printf("TOK %u %zu ", t, tlens[t]);
    for (size_t k = 0u; k < tlens[t]; ++k)
      printf("%02x", tptrs[t][k]);
    printf("\n");
  }

  /* Dump DFA transitions and precomputed allowed bitsets for every state. */
  for (uint16_t s = 0u; s < dinfo.num_states; ++s) {
    printf("DFA_STATE %u %u %u", s, et_cd_dfa_is_live(dfa, s),
           et_cd_dfa_is_accepting(dfa, s));
    for (uint32_t b = 0u; b < 256u; ++b) {
      const uint16_t nxt = et_cd_dfa_next_state(dfa, s, (uint8_t)b);
      if (nxt != ET_CD_DEAD_STATE)
        printf(" %u:%u", b, nxt);
    }
    printf("\n");

    uint64_t words[ET_CD_VOCAB_WORDS_MAX];
    CHECK(et_cd_vocab_index_copy_mask_words(idx, s, words, vinfo.mask_words) ==
          ET_CD_OK);
    printf("MASK %u", s);
    for (uint32_t w = 0u; w < vinfo.mask_words; ++w)
      printf(" %016llx", (unsigned long long)words[w]);
    printf("\n");
  }

  /* Run randomized adversarial-logit trajectories under zero-alloc trap! */
  uint64_t rng = seed ? seed : UINT64_C(0x123456789abcdef0);
  float logits[114];
  float masked[114];
  uint8_t out_buf[4096];

  for (uint32_t traj = 0u; traj < num_trajectories; ++traj) {
    uint16_t state = vinfo.start_state;
    uint32_t finished = 0u;
    size_t out_len = 0u;
    uint32_t steps = 0u;
    uint32_t adversarial_blocks = 0u;

    hot_path_active = 1;
    while (!finished && steps < 96u) {
      for (uint32_t t = 0u; t < V; ++t) {
        const uint64_t r = xorshift64(&rng);
        logits[t] = (float)((int32_t)(r % 2001u) - 1000) * 0.01f;
      }
      /* Adversarial spikes on definitely-invalid / tricky tokens. */
      logits[95u + 14u] = 100.0f; /* "INVALID\n{{" (unescaped LF: invalid everywhere!) */
      logits[95u + 16u] = 99.0f;  /* "\"\"broken" (double quote: invalid everywhere!) */
      logits[95u + 13u] = 98.0f;  /* "\"}\"" */
      logits[95u + 15u] = 97.0f;  /* "007" */
      /* After 24 steps, bias closing delimiters (',' '"' '}') and EOS above
       * normal tokens to ensure finite termination of unbounded INTEGER/STRING loops. */
      if (steps >= 24u) {
        logits[(uint32_t)(',' - 0x20)] = 150.0f;
        logits[(uint32_t)('"' - 0x20)] = 150.0f;
        logits[(uint32_t)('}' - 0x20)] = 150.0f;
        logits[EOS] = 160.0f;
      }

      /* Record unconstrained argmax to measure adversarial intervention. */
      uint32_t raw_best = 0u;
      for (uint32_t t = 1u; t < V; ++t) {
        if (logits[t] > logits[raw_best])
          raw_best = t;
      }

      CHECK(et_cd_apply_logit_mask_v1(idx, state, logits, masked) == ET_CD_OK);
      if (!isfinite(masked[raw_best]))
        adversarial_blocks += 1u;

      uint32_t best = V;
      for (uint32_t t = 0u; t < V; ++t) {
        if (isfinite(masked[t]) && (best == V || masked[t] > masked[best]))
          best = t;
      }
      CHECK(best < V);

      uint16_t next_s = ET_CD_DEAD_STATE;
      CHECK(et_cd_advance_token_v1(idx, state, best, &next_s, &finished) ==
            ET_CD_OK);
      if (!finished) {
        CHECK(out_len + tlens[best] <= sizeof(out_buf));
        memcpy(out_buf + out_len, tptrs[best], tlens[best]);
        out_len += tlens[best];
      }
      state = next_s;
      steps += 1u;
    }
    hot_path_active = 0;

    CHECK(finished == 1u);
    printf("TRAJ %u %u %u ", traj, steps, adversarial_blocks);
    for (size_t k = 0u; k < out_len; ++k)
      printf("%02x", out_buf[k]);
    printf("\n");
  }

  et_cd_vocab_index_destroy(idx);
  et_cd_dfa_destroy(dfa);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--oracle-prune") == 0)
    return run_oracle_prune();
  if (argc >= 4 && strcmp(argv[1], "--oracle-json") == 0) {
    const uint64_t seed = (uint64_t)strtoull(argv[2], NULL, 10);
    const uint32_t count = (uint32_t)strtoul(argv[3], NULL, 10);
    return run_oracle_json(seed, count);
  }

  test_coaccessibility_pruning();
  test_literal_choices();
  test_flat_json_schema();
  test_hot_path_and_negatives();
  test_artifact_io();

  printf("CONSTRAINED_DECODE native C11 tests (zero-alloc trapped): PASS\n");
  return 0;
}
