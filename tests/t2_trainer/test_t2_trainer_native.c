#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/t2_bpe_trainer_abi.h"
#include "t2_bpe_trainer_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, msg)                                                 \
  do {                                                                   \
    if (!(cond)) {                                                       \
      fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

/* Local SHA-256 helper solely for crafting re-signed malicious artifact tests. */
typedef struct {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} test_sha256_ctx;

static uint32_t test_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void test_sha256_block(test_sha256_ctx *s, const uint8_t b[64]) {
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
  uint32_t w[64];
  for (size_t i = 0u; i < 16u; ++i) {
    w[i] = ((uint32_t)b[i * 4u] << 24u) | ((uint32_t)b[i * 4u + 1u] << 16u) |
           ((uint32_t)b[i * 4u + 2u] << 8u) | (uint32_t)b[i * 4u + 3u];
  }
  for (size_t i = 16u; i < 64u; ++i) {
    uint32_t s0 =
        test_rotr32(w[i - 15u], 7u) ^ test_rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    uint32_t s1 =
        test_rotr32(w[i - 2u], 17u) ^ test_rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = s->h[0], bv = s->h[1], c = s->h[2], d = s->h[3];
  uint32_t e = s->h[4], f = s->h[5], g = s->h[6], h = s->h[7];
  for (size_t i = 0u; i < 64u; ++i) {
    uint32_t s1 = test_rotr32(e, 6u) ^ test_rotr32(e, 11u) ^ test_rotr32(e, 25u);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t t1 = h + s1 + ch + k[i] + w[i];
    uint32_t s0 = test_rotr32(a, 2u) ^ test_rotr32(a, 13u) ^ test_rotr32(a, 22u);
    uint32_t maj = (a & bv) ^ (a & c) ^ (bv & c);
    uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1;
    d = c; c = bv; bv = a; a = t1 + t2;
  }
  s->h[0] += a; s->h[1] += bv; s->h[2] += c; s->h[3] += d;
  s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

static void test_sha256(const uint8_t *data, size_t size, uint8_t out[32]) {
  static const uint32_t init[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                   0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                   0x1f83d9abu, 0x5be0cd19u};
  test_sha256_ctx ctx;
  memcpy(ctx.h, init, sizeof(init));
  ctx.length = (uint64_t)size;
  ctx.used = 0u;
  while (size != 0u) {
    size_t take = 64u - ctx.used;
    if (take > size) take = size;
    memcpy(ctx.block + ctx.used, data, take);
    ctx.used += take;
    data += take;
    size -= take;
    if (ctx.used == 64u) {
      test_sha256_block(&ctx, ctx.block);
      ctx.used = 0u;
    }
  }
  ctx.block[ctx.used++] = 0x80u;
  if (ctx.used > 56u) {
    memset(ctx.block + ctx.used, 0, 64u - ctx.used);
    test_sha256_block(&ctx, ctx.block);
    ctx.used = 0u;
  }
  memset(ctx.block + ctx.used, 0, 56u - ctx.used);
  const uint64_t bits = ctx.length * UINT64_C(8);
  for (size_t i = 0u; i < 8u; ++i) {
    ctx.block[63u - i] = (uint8_t)(bits >> (8u * i));
  }
  test_sha256_block(&ctx, ctx.block);
  for (size_t i = 0u; i < 8u; ++i) {
    out[i * 4u] = (uint8_t)(ctx.h[i] >> 24u);
    out[i * 4u + 1u] = (uint8_t)(ctx.h[i] >> 16u);
    out[i * 4u + 2u] = (uint8_t)(ctx.h[i] >> 8u);
    out[i * 4u + 3u] = (uint8_t)ctx.h[i];
  }
}

static void put_le16(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static void put_le32(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void resign_artifact(uint8_t *buf, size_t size) {
  test_sha256(buf, size - ET_T2_BPE_SHA256_BYTES,
              buf + (size - ET_T2_BPE_SHA256_BYTES));
}

static void assert_roundtrip(const et_t2_bpe_model *model,
                             const uint8_t *input, size_t len,
                             const char *label) {
  size_t tok_count = 0u;
  CHECK(et_t2_bpe_encode(model, input, len, NULL, 0u, &tok_count) ==
            ET_T2_BPE_OK,
        label);
  uint32_t *toks = tok_count != 0u ? malloc(tok_count * sizeof(*toks)) : NULL;
  size_t actual_toks = 0u;
  CHECK(et_t2_bpe_encode(model, input, len, toks, tok_count, &actual_toks) ==
            ET_T2_BPE_OK,
        label);
  CHECK(actual_toks == tok_count, label);

  size_t dec_len = 0u;
  CHECK(et_t2_bpe_decode(model, toks, actual_toks, NULL, 0u, &dec_len) ==
            ET_T2_BPE_OK,
        label);
  CHECK(dec_len == len, label);

  uint8_t *decoded = dec_len != 0u ? malloc(dec_len) : NULL;
  size_t actual_dec = 0u;
  CHECK(et_t2_bpe_decode(model, toks, actual_toks, decoded, dec_len,
                         &actual_dec) == ET_T2_BPE_OK,
        label);
  CHECK(actual_dec == len, label);
  if (len != 0u && decoded != NULL) {
    CHECK(memcmp(input, decoded, len) == 0, label);
  }
  free(decoded);
  free(toks);
}

static void test_determinism_and_tie_breaking(void) {
  /* Ties:
   * doc0 = "dc", doc1 = "dc" -> ('d','c') freq=2
   * doc2 = "ba", doc3 = "ba" -> ('b','a') freq=2
   * doc4 = "ab", doc5 = "ab" -> ('a','b') freq=2
   * doc6 = "ac", doc7 = "ac" -> ('a','c') freq=2
   * Lowest lexicographic (left, right) must win in exact order:
   * rank 0: ('a','b') -> 256
   * rank 1: ('a','c') -> 257
   * rank 2: ('b','a') -> 258
   * rank 3: ('d','c') -> 259
   * Also confirms document boundaries NEVER form cross-doc pairs. */
  const et_t2_bpe_document docs[] = {
      {(const uint8_t *)"dc", 2u}, {(const uint8_t *)"dc", 2u},
      {(const uint8_t *)"ba", 2u}, {(const uint8_t *)"ba", 2u},
      {(const uint8_t *)"ab", 2u}, {(const uint8_t *)"ab", 2u},
      {(const uint8_t *)"ac", 2u}, {(const uint8_t *)"ac", 2u},
  };
  const et_t2_bpe_train_config cfg = {
      .target_vocab_size = 300u,
      .min_frequency = 2u,
      .special_mask = ET_T2_BPE_SPECIAL_ALL,
  };

  et_t2_bpe_model *m1 = NULL;
  et_t2_bpe_model *m2 = NULL;
  CHECK(et_t2_bpe_train(docs, 8u, &cfg, &m1) == ET_T2_BPE_OK, "train m1");
  CHECK(et_t2_bpe_train(docs, 8u, &cfg, &m2) == ET_T2_BPE_OK, "train m2");
  CHECK(m1 != NULL && m2 != NULL, "models non-null");

  et_t2_bpe_info info1, info2;
  CHECK(et_t2_bpe_get_info(m1, &info1) == ET_T2_BPE_OK, "info1");
  CHECK(et_t2_bpe_get_info(m2, &info2) == ET_T2_BPE_OK, "info2");
  CHECK(info1.merge_count == 4u, "early stop at exactly 4 merges");
  CHECK(info1.vocab_size == 256u + 4u + 4u, "vocab_size = 256 + merges + 4");
  CHECK(info1.bos_id == 260u && info1.eos_id == 261u &&
            info1.pad_id == 262u && info1.unk_id == 263u,
        "special IDs contiguous after merges");
  CHECK(memcmp(&info1, &info2, sizeof(info1)) == 0, "deterministic info");

  const uint32_t expected[4][3] = {
      {'a', 'b', 256u},
      {'a', 'c', 257u},
      {'b', 'a', 258u},
      {'d', 'c', 259u},
  };
  for (uint32_t r = 0u; r < 4u; ++r) {
    et_t2_bpe_merge mg1, mg2;
    CHECK(et_t2_bpe_get_merge(m1, r, &mg1) == ET_T2_BPE_OK, "get merge 1");
    CHECK(et_t2_bpe_get_merge(m2, r, &mg2) == ET_T2_BPE_OK, "get merge 2");
    CHECK(mg1.left == expected[r][0] && mg1.right == expected[r][1] &&
              mg1.new_id == expected[r][2],
          "canonical lexicographic tie-breaking");
    CHECK(memcmp(&mg1, &mg2, sizeof(mg1)) == 0, "deterministic merges");
  }

  /* Token spelling inspection */
  uint8_t tok_buf[8];
  size_t tok_len = 0u;
  CHECK(et_t2_bpe_token_bytes(m1, 256u, tok_buf, sizeof(tok_buf), &tok_len) ==
            ET_T2_BPE_OK,
        "token_bytes 256");
  CHECK(tok_len == 2u && tok_buf[0] == 'a' && tok_buf[1] == 'b',
        "token 256 == ab");
  CHECK(et_t2_bpe_token_bytes(m1, info1.bos_id, tok_buf, sizeof(tok_buf),
                              &tok_len) == ET_T2_BPE_OK &&
            tok_len == 0u,
        "special token has 0 decoded bytes");

  /* Serialize parity */
  size_t sz1 = 0u, sz2 = 0u;
  uint8_t sha1[32], sha2[32];
  CHECK(et_t2_bpe_serialize(m1, NULL, 0u, &sz1, NULL) == ET_T2_BPE_OK, "sz1");
  CHECK(et_t2_bpe_serialize(m2, NULL, 0u, &sz2, NULL) == ET_T2_BPE_OK, "sz2");
  CHECK(sz1 == sz2 && sz1 == ET_T2_BPE_HEADER_BYTES + 4u * 12u + 32u,
        "exact artifact size");
  uint8_t *b1 = malloc(sz1);
  uint8_t *b2 = malloc(sz2);
  CHECK(et_t2_bpe_serialize(m1, b1, sz1, &sz1, sha1) == ET_T2_BPE_OK, "ser1");
  CHECK(et_t2_bpe_serialize(m2, b2, sz2, &sz2, sha2) == ET_T2_BPE_OK, "ser2");
  CHECK(memcmp(b1, b2, sz1) == 0 && memcmp(sha1, sha2, 32u) == 0,
        "byte-identical serialized artifacts");

  free(b1);
  free(b2);
  et_t2_bpe_destroy(m1);
  et_t2_bpe_destroy(m2);
}

static void test_compression_and_lossless_roundtrip(const char *tmp_dir) {
  static const char corpus_text[] =
      "eshkol transformer deterministic byte pair encoding trainer "
      "eshkol transformer deterministic byte pair encoding trainer "
      "eshkol transformer deterministic byte pair encoding trainer "
      "transformer attention normalization tokenizer checkpoint "
      "transformer attention normalization tokenizer checkpoint "
      "transformer attention normalization tokenizer checkpoint "
      "deterministic checksum verified artifact reload parity "
      "deterministic checksum verified artifact reload parity\n";
  const et_t2_bpe_document docs[] = {
      {(const uint8_t *)corpus_text, sizeof(corpus_text) - 1u},
      {(const uint8_t *)"\x00\x01\x02\x00\x01\x02\xff\xfe\xff\xfe", 10u},
  };
  const et_t2_bpe_train_config cfg = {
      .target_vocab_size = 340u,
      .min_frequency = 2u,
      .special_mask = ET_T2_BPE_SPECIAL_BOS | ET_T2_BPE_SPECIAL_EOS,
  };

  et_t2_bpe_model *model = NULL;
  CHECK(et_t2_bpe_train(docs, 2u, &cfg, &model) == ET_T2_BPE_OK,
        "train structured corpus");
  CHECK(model != NULL, "model trained");

  /* Compression check on structured text */
  static const char probe[] =
      "eshkol transformer deterministic tokenizer checkpoint artifact";
  const size_t probe_len = sizeof(probe) - 1u;
  size_t encoded_len = 0u;
  CHECK(et_t2_bpe_encode(model, (const uint8_t *)probe, probe_len, NULL, 0u,
                         &encoded_len) == ET_T2_BPE_OK,
        "encode size query");
  CHECK(encoded_len < probe_len, "compression: encoded_len < raw_len");
  CHECK(encoded_len * 2u < probe_len, "compression ratio > 2.0x on structured text");

  /* 100% lossless round-trip suite */
  assert_roundtrip(model, (const uint8_t *)"", 0u, "empty input");
  assert_roundtrip(model, (const uint8_t *)probe, probe_len, "structured ASCII");

  /* All 256 raw byte values individually */
  for (unsigned v = 0u; v < 256u; ++v) {
    uint8_t single = (uint8_t)v;
    assert_roundtrip(model, &single, 1u, "single raw byte");
  }

  /* Full 0x00..0xFF ramp and reversed ramp */
  uint8_t ramp[256];
  uint8_t rev_ramp[256];
  for (size_t i = 0u; i < 256u; ++i) {
    ramp[i] = (uint8_t)i;
    rev_ramp[i] = (uint8_t)(255u - i);
  }
  assert_roundtrip(model, ramp, sizeof(ramp), "all 256 raw bytes 0x00..0xFF");
  assert_roundtrip(model, rev_ramp, sizeof(rev_ramp),
                   "all 256 raw bytes 0xFF..0x00");

  /* NUL-embedded bytes */
  static const uint8_t nul_payload[] = {0x00, 'e',  's',  'h',  'k',  'o',
                                        'l',  0x00, 0x00, 0x01, 0x02, 0x00,
                                        't',  'r',  'a',  'n',  's',  0x00};
  assert_roundtrip(model, nul_payload, sizeof(nul_payload), "NUL-embedded");

  /* Multi-byte UTF-8 (Latin, Greek, CJK, 4-byte emoji) */
  static const uint8_t utf8_payload[] =
      "caf\xc3\xa9 \xce\xb1\xce\xb2\xce\xb3 "
      "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e "
      "\xf0\x9f\x9a\x80\xf0\x9f\xa7\xaa "
      "eshkol transformer";
  assert_roundtrip(model, utf8_payload, sizeof(utf8_payload) - 1u,
                   "multi-byte UTF-8");

  /* Invalid non-UTF-8 byte sequence */
  static const uint8_t invalid_utf8[] = {0xff, 0xfe, 0x80, 0xc0, 0xaf,
                                         0xed, 0xa0, 0x80, 0xf8, 0x88,
                                         0x80, 0x80, 0x80, 0x00, 0xff};
  assert_roundtrip(model, invalid_utf8, sizeof(invalid_utf8),
                   "invalid non-UTF-8 bytes");

  /* encode_ex with BOS/EOS + decode omitting specials */
  et_t2_bpe_info info;
  CHECK(et_t2_bpe_get_info(model, &info) == ET_T2_BPE_OK, "info");
  uint32_t wrapped[128];
  size_t wrapped_len = 0u;
  CHECK(et_t2_bpe_encode_ex(model, (const uint8_t *)probe, probe_len, 1u, 1u,
                            wrapped, 128u, &wrapped_len) == ET_T2_BPE_OK,
        "encode_ex BOS+EOS");
  CHECK(wrapped_len == encoded_len + 2u, "BOS+EOS adds 2 tokens");
  CHECK(wrapped[0] == info.bos_id && wrapped[wrapped_len - 1u] == info.eos_id,
        "BOS at start and EOS at end");
  uint8_t dec_wrapped[128];
  size_t dec_wrapped_len = 0u;
  CHECK(et_t2_bpe_decode(model, wrapped, wrapped_len, dec_wrapped,
                         sizeof(dec_wrapped), &dec_wrapped_len) == ET_T2_BPE_OK,
        "decode BOS+EOS");
  CHECK(dec_wrapped_len == probe_len &&
            memcmp(dec_wrapped, probe, probe_len) == 0,
        "decode omits BOS/EOS and restores exact bytes");

  /* Save atomically to file and reload in fresh model */
  char artifact_path[512];
  snprintf(artifact_path, sizeof(artifact_path), "%s/model.etbpe", tmp_dir);
  uint8_t save_sha[32], load_sha[32];
  CHECK(et_t2_bpe_save_file(model, artifact_path, save_sha) == ET_T2_BPE_OK,
        "atomic save_file");

  struct stat st;
  CHECK(stat(artifact_path, &st) == 0 && S_ISREG(st.st_mode) &&
            (st.st_mode & 0111) == 0,
        "saved artifact is regular non-executable file");

  et_t2_bpe_model *reloaded = NULL;
  CHECK(et_t2_bpe_load_file(artifact_path, &reloaded, load_sha) == ET_T2_BPE_OK,
        "load_file");
  CHECK(reloaded != NULL, "reloaded non-null");
  CHECK(memcmp(save_sha, load_sha, 32u) == 0, "save/load SHA-256 parity");

  uint32_t orig_toks[128], rel_toks[128];
  size_t orig_n = 0u, rel_n = 0u;
  CHECK(et_t2_bpe_encode(model, (const uint8_t *)probe, probe_len, orig_toks,
                         128u, &orig_n) == ET_T2_BPE_OK,
        "encode orig");
  CHECK(et_t2_bpe_encode(reloaded, (const uint8_t *)probe, probe_len, rel_toks,
                         128u, &rel_n) == ET_T2_BPE_OK,
        "encode reloaded");
  CHECK(orig_n == rel_n && memcmp(orig_toks, rel_toks, orig_n * sizeof(uint32_t)) == 0,
        "save/reload encode parity");
  assert_roundtrip(reloaded, ramp, sizeof(ramp), "reloaded roundtrip ramp");
  assert_roundtrip(reloaded, nul_payload, sizeof(nul_payload),
                   "reloaded roundtrip NUL");

  et_t2_bpe_destroy(reloaded);
  et_t2_bpe_destroy(model);
}

static void test_negative_tamper_and_symlink(const char *tmp_dir) {
  const et_t2_bpe_document doc = {(const uint8_t *)"ababababcdcdcd", 14u};
  et_t2_bpe_train_config cfg = {270u, 2u, ET_T2_BPE_SPECIAL_NONE};
  et_t2_bpe_model *model = NULL;

  /* Argument validation */
  CHECK(et_t2_bpe_train(NULL, 1u, &cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "null docs with doc_count=1");
  CHECK(et_t2_bpe_train(&doc, 1u, NULL, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "null config");
  CHECK(et_t2_bpe_train(&doc, 1u, &cfg, NULL) == ET_T2_BPE_ERR_ARGUMENT,
        "null out_model");

  et_t2_bpe_train_config bad_cfg = cfg;
  bad_cfg.target_vocab_size = 255u;
  CHECK(et_t2_bpe_train(&doc, 1u, &bad_cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "vocab < 256 rejected");
  bad_cfg.target_vocab_size = 4097u;
  CHECK(et_t2_bpe_train(&doc, 1u, &bad_cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "vocab > 4096 rejected");
  bad_cfg.target_vocab_size = 256u;
  bad_cfg.special_mask = ET_T2_BPE_SPECIAL_BOS;
  CHECK(et_t2_bpe_train(&doc, 1u, &bad_cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "vocab < 256 + specials rejected");
  bad_cfg.target_vocab_size = 300u;
  bad_cfg.special_mask = 0x10u;
  CHECK(et_t2_bpe_train(&doc, 1u, &bad_cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "unknown special bit rejected");
  bad_cfg.special_mask = 0u;
  bad_cfg.min_frequency = 0u;
  CHECK(et_t2_bpe_train(&doc, 1u, &bad_cfg, &model) == ET_T2_BPE_ERR_ARGUMENT,
        "min_frequency == 0 rejected");

  /* Valid base model for negative encode/decode/artifact tests */
  CHECK(et_t2_bpe_train(&doc, 1u, &cfg, &model) == ET_T2_BPE_OK, "valid train");

  /* Unconfigured BOS/EOS in encode_ex rejected explicitly */
  size_t out_n = 0u;
  CHECK(et_t2_bpe_encode_ex(model, doc.bytes, doc.byte_count, 1u, 0u, NULL, 0u,
                            &out_n) == ET_T2_BPE_ERR_ARGUMENT,
        "unconfigured BOS rejected");
  CHECK(et_t2_bpe_encode_ex(model, doc.bytes, doc.byte_count, 0u, 1u, NULL, 0u,
                            &out_n) == ET_T2_BPE_ERR_ARGUMENT,
        "unconfigured EOS rejected");

  /* Out-of-range token IDs on decode */
  et_t2_bpe_info info;
  CHECK(et_t2_bpe_get_info(model, &info) == ET_T2_BPE_OK, "get_info");
  uint32_t bad_toks[2] = {65u, info.vocab_size};
  uint8_t out_buf[16] = {0xAA};
  size_t out_len = 0u;
  CHECK(et_t2_bpe_decode(model, bad_toks, 2u, out_buf, sizeof(out_buf),
                         &out_len) == ET_T2_BPE_ERR_TOKEN_ID,
        "out-of-range token ID rejected");
  CHECK(out_buf[0] == 0xAA, "failed decode leaves output buffer untouched");
  bad_toks[1] = UINT32_MAX;
  CHECK(et_t2_bpe_decode(model, bad_toks, 2u, out_buf, sizeof(out_buf),
                         &out_len) == ET_T2_BPE_ERR_TOKEN_ID,
        "UINT32_MAX token ID rejected");

  /* Bounds check leaves output untouched */
  uint32_t small_toks[1] = {0xDEADBEEFu};
  CHECK(et_t2_bpe_encode(model, (const uint8_t *)"xyz", 3u, small_toks, 1u,
                         &out_n) == ET_T2_BPE_ERR_BOUNDS &&
            out_n == 3u && small_toks[0] == 0xDEADBEEFu,
        "encode bounds check leaves buffer untouched");

  /* Serialize valid artifact, then tamper in every required way */
  size_t sz = 0u;
  CHECK(et_t2_bpe_serialize(model, NULL, 0u, &sz, NULL) == ET_T2_BPE_OK, "sz");
  uint8_t *valid = malloc(sz);
  CHECK(et_t2_bpe_serialize(model, valid, sz, &sz, NULL) == ET_T2_BPE_OK, "ser");

  et_t2_bpe_model *bad_model = NULL;
  uint8_t *mut = malloc(sz + 4u);

  /* 1. Corrupt checksum (flip 1 bit in merge table without re-signing) */
  memcpy(mut, valid, sz);
  mut[ET_T2_BPE_HEADER_BYTES] ^= 0x01u;
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_CHECKSUM &&
            bad_model == NULL,
        "corrupt payload checksum rejected");

  /* 2. Corrupt checksum trailer directly */
  memcpy(mut, valid, sz);
  mut[sz - 1u] ^= 0x80u;
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_CHECKSUM,
        "corrupt trailer checksum rejected");

  /* 3. Truncated artifacts */
  const size_t trunc_sizes[] = {0u, 4u, 11u, 44u, 75u, sz - 1u, sz - 12u};
  for (size_t i = 0u; i < sizeof(trunc_sizes) / sizeof(trunc_sizes[0]); ++i) {
    CHECK(et_t2_bpe_deserialize(valid, trunc_sizes[i], &bad_model, NULL) ==
              ET_T2_BPE_ERR_FORMAT,
          "truncated artifact rejected");
  }

  /* 4. Trailing bytes */
  memcpy(mut, valid, sz);
  mut[sz] = 0u;
  CHECK(et_t2_bpe_deserialize(mut, sz + 1u, &bad_model, NULL) ==
            ET_T2_BPE_ERR_FORMAT,
        "trailing bytes rejected");

  /* 5. Bad magic */
  memcpy(mut, valid, sz);
  mut[0] = 'X';
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) == ET_T2_BPE_ERR_MAGIC,
        "bad magic rejected");

  /* 6. Version mismatch (major=2, minor=0 and major=1, minor=1) */
  memcpy(mut, valid, sz);
  put_le16(mut + 8u, 2u);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_VERSION,
        "major version mismatch rejected");
  memcpy(mut, valid, sz);
  put_le16(mut + 10u, 1u);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_VERSION,
        "minor version mismatch rejected");

  /* 7. Re-signed semantic attacks: non-zero flags */
  memcpy(mut, valid, sz);
  put_le32(mut + 12u, 1u);
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_FORMAT,
        "re-signed nonzero flags rejected");

  /* 8. Re-signed self-referential merge (left == new_id) */
  memcpy(mut, valid, sz);
  put_le32(mut + ET_T2_BPE_HEADER_BYTES, 256u); /* rank 0 new_id is 256 */
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_INVALID_MERGE,
        "self-referential merge rejected");

  /* 9. Re-signed forward/cyclic merge (right > new_id) */
  memcpy(mut, valid, sz);
  put_le32(mut + ET_T2_BPE_HEADER_BYTES + 4u, 257u);
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_INVALID_MERGE,
        "forward/cyclic merge rejected");

  /* 10. Re-signed out-of-range merge operand */
  memcpy(mut, valid, sz);
  put_le32(mut + ET_T2_BPE_HEADER_BYTES + 4u, 9999u);
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_INVALID_MERGE,
        "out-of-range merge operand rejected");

  /* 11. Re-signed wrong new_id */
  memcpy(mut, valid, sz);
  put_le32(mut + ET_T2_BPE_HEADER_BYTES + 8u, 260u);
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_INVALID_MERGE,
        "wrong new_id rejected");

  /* 12. Re-signed duplicate merge pair */
  if (info.merge_count >= 2u) {
    memcpy(mut, valid, sz);
    memcpy(mut + ET_T2_BPE_HEADER_BYTES + 12u, mut + ET_T2_BPE_HEADER_BYTES, 8u);
    resign_artifact(mut, sz);
    CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
              ET_T2_BPE_ERR_INVALID_MERGE,
          "duplicate merge pair rejected");
  }

  /* 13. Re-signed zip-bomb exponential doubling chain (>256 decoded bytes) */
  {
    const uint32_t mcount = 9u; /* 2^9 = 512 > 256 */
    const size_t zsz = ET_T2_BPE_HEADER_BYTES + mcount * 12u + 32u;
    uint8_t *zbuf = calloc(1u, zsz);
    memcpy(zbuf, ET_T2_BPE_MAGIC, 8u);
    put_le16(zbuf + 8u, 1u);
    put_le16(zbuf + 10u, 0u);
    put_le32(zbuf + 12u, 0u);
    put_le32(zbuf + 16u, 256u + mcount);
    put_le32(zbuf + 20u, mcount);
    put_le32(zbuf + 24u, 0u);
    put_le32(zbuf + 28u, ET_T2_BPE_TOKEN_NONE);
    put_le32(zbuf + 32u, ET_T2_BPE_TOKEN_NONE);
    put_le32(zbuf + 36u, ET_T2_BPE_TOKEN_NONE);
    put_le32(zbuf + 40u, ET_T2_BPE_TOKEN_NONE);
    uint32_t prev = 'a';
    for (uint32_t r = 0u; r < mcount; ++r) {
      size_t off = ET_T2_BPE_HEADER_BYTES + r * 12u;
      put_le32(zbuf + off, prev);
      put_le32(zbuf + off + 4u, prev);
      put_le32(zbuf + off + 8u, 256u + r);
      prev = 256u + r;
    }
    resign_artifact(zbuf, zsz);
    CHECK(et_t2_bpe_deserialize(zbuf, zsz, &bad_model, NULL) ==
              ET_T2_BPE_ERR_INVALID_MERGE,
          "zip-bomb token > 256 bytes rejected");
    free(zbuf);
  }

  /* 14. Re-signed mismatched special ID */
  memcpy(mut, valid, sz);
  put_le32(mut + 28u, 100u); /* bos_id set while special_mask==0 */
  resign_artifact(mut, sz);
  CHECK(et_t2_bpe_deserialize(mut, sz, &bad_model, NULL) ==
            ET_T2_BPE_ERR_FORMAT,
        "mismatched special ID rejected");

  /* 15. Symlink rejection on BOTH load_file (O_NOFOLLOW) and save_file */
  char real_path[512];
  char link_path[512];
  snprintf(real_path, sizeof(real_path), "%s/real.etbpe", tmp_dir);
  snprintf(link_path, sizeof(link_path), "%s/symlink.etbpe", tmp_dir);
  CHECK(et_t2_bpe_save_file(model, real_path, NULL) == ET_T2_BPE_OK,
        "save real file");
  CHECK(symlink(real_path, link_path) == 0, "create symlink");
  CHECK(et_t2_bpe_load_file(link_path, &bad_model, NULL) == ET_T2_BPE_ERR_IO &&
            bad_model == NULL,
        "load_file rejects symlink via O_NOFOLLOW");
  CHECK(et_t2_bpe_save_file(model, link_path, NULL) == ET_T2_BPE_ERR_IO,
        "save_file refuses overwriting symlink");
  CHECK(et_t2_bpe_load_file(tmp_dir, &bad_model, NULL) == ET_T2_BPE_ERR_IO,
        "load_file rejects directory");

  free(mut);
  free(valid);
  et_t2_bpe_destroy(model);
}

static void test_bridge_api(const char *tmp_dir) {
  et_t2_bpe_aot_corpus *corpus = NULL;
  CHECK(et_t2_bpe_aot_corpus_create(&corpus) == ET_T2_BPE_OK && corpus != NULL,
        "bridge corpus create");
  CHECK(et_t2_bpe_aot_corpus_add_cstr(
            corpus, "banana bandana banana bandana banana") == ET_T2_BPE_OK,
        "bridge add cstr");
  const uint8_t bin_doc[] = {0x00, 0xff, 0x00, 0xff, 0x00, 0xff};
  CHECK(et_t2_bpe_aot_corpus_add_bytes(corpus, bin_doc, 6) == ET_T2_BPE_OK,
        "bridge add bytes");
  int64_t doc_count = 0;
  CHECK(et_t2_bpe_aot_corpus_doc_count(corpus, &doc_count) == ET_T2_BPE_OK &&
            doc_count == 2,
        "bridge doc_count == 2");

  et_t2_bpe_model *model = NULL;
  CHECK(et_t2_bpe_aot_train(corpus, 290, 2, ET_T2_BPE_SPECIAL_ALL, &model) ==
            ET_T2_BPE_OK &&
            model != NULL,
        "bridge train");
  int64_t vocab = 0, merges = 0;
  int64_t specials[4] = {0};
  int64_t m0[3] = {0};
  CHECK(et_t2_bpe_aot_vocab_size(model, &vocab) == ET_T2_BPE_OK && vocab > 260,
        "bridge vocab_size");
  CHECK(et_t2_bpe_aot_merge_count(model, &merges) == ET_T2_BPE_OK &&
            merges > 0 && vocab == 256 + merges + 4,
        "bridge merge_count");
  CHECK(et_t2_bpe_aot_special_ids(model, specials) == ET_T2_BPE_OK &&
            specials[0] == 256 + merges && specials[3] == 256 + merges + 3,
        "bridge special_ids");
  CHECK(et_t2_bpe_aot_get_merge(model, 0, m0) == ET_T2_BPE_OK && m0[2] == 256,
        "bridge get_merge 0");

  char path[512];
  snprintf(path, sizeof(path), "%s/bridge.etbpe", tmp_dir);
  uint8_t sha_save[32], sha_load[32];
  CHECK(et_t2_bpe_aot_save(model, path, sha_save) == ET_T2_BPE_OK, "bridge save");
  et_t2_bpe_model *reloaded = NULL;
  CHECK(et_t2_bpe_aot_load(path, &reloaded, sha_load) == ET_T2_BPE_OK &&
            reloaded != NULL,
        "bridge load");
  int32_t eq = 0;
  CHECK(et_t2_bpe_aot_bytes_equal(sha_save, sha_load, 32, &eq) == ET_T2_BPE_OK &&
            eq == 1,
        "bridge bytes_equal sha256");

  int64_t tok_n = 0;
  uint32_t toks[64];
  uint8_t dec[64];
  int64_t dec_n = 0;
  CHECK(et_t2_bpe_aot_count_tokens_cstr(reloaded, "banana bandana", &tok_n) ==
            ET_T2_BPE_OK &&
            tok_n < 14,
        "bridge count_tokens_cstr compresses");
  CHECK(et_t2_bpe_aot_encode_cstr(reloaded, "banana bandana", toks, 64, &tok_n) ==
            ET_T2_BPE_OK,
        "bridge encode_cstr");
  CHECK(et_t2_bpe_aot_decode_bytes(reloaded, toks, tok_n, dec, 64, &dec_n) ==
            ET_T2_BPE_OK,
        "bridge decode_bytes");
  CHECK(et_t2_bpe_aot_cstr_bytes_equal("banana bandana", dec, dec_n, &eq) ==
            ET_T2_BPE_OK &&
            eq == 1,
        "bridge lossless cstr roundtrip");

  /* Buffer helpers bounds checks */
  uint8_t u8buf[4] = {0};
  uint32_t u32buf[4] = {0};
  CHECK(et_t2_bpe_aot_store_u8(u8buf, 4, 2, 0xAB) == ET_T2_BPE_OK &&
            et_t2_bpe_aot_load_u8(u8buf, 4, 2) == 0xAB,
        "bridge u8 store/load");
  CHECK(et_t2_bpe_aot_store_u8(u8buf, 4, 4, 1) == ET_T2_BPE_ERR_ARGUMENT &&
            et_t2_bpe_aot_load_u8(u8buf, 4, 4) == -1,
        "bridge u8 bounds");
  CHECK(et_t2_bpe_aot_store_u32(u32buf, 4, 1, 300) == ET_T2_BPE_OK &&
            et_t2_bpe_aot_load_u32(u32buf, 4, 1) == 300 &&
            et_t2_bpe_aot_load_u32(u32buf, 4, -1) == -1,
        "bridge u32 store/load/bounds");

  /* train_bytes negative check */
  et_t2_bpe_model *bad = NULL;
  CHECK(et_t2_bpe_aot_train_bytes(bin_doc, -1, 260, 1, 0, &bad) ==
            ET_T2_BPE_ERR_ARGUMENT,
        "bridge negative byte_count rejected");

  et_t2_bpe_aot_destroy(reloaded);
  et_t2_bpe_aot_destroy(model);
  et_t2_bpe_aot_corpus_destroy(corpus);
}

/* Oracle CLI helper: reads length-prefixed docs + input file, trains, emits
 * JSON-free TSV summary of merges, encoded tokens, decoded bytes hex, and
 * writes artifact to out_artifact_path for exact Python oracle verification. */
static int read_entire_file(const char *path, uint8_t **out_data,
                            size_t *out_len) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) return 0;
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return 0;
  }
  long sz = ftell(fp);
  if (sz < 0 || fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    return 0;
  }
  uint8_t *buf = malloc((size_t)sz + 1u);
  if (buf == NULL) {
    fclose(fp);
    return 0;
  }
  if (sz > 0 && fread(buf, 1u, (size_t)sz, fp) != (size_t)sz) {
    free(buf);
    fclose(fp);
    return 0;
  }
  fclose(fp);
  *out_data = buf;
  *out_len = (size_t)sz;
  return 1;
}

static int run_oracle_cli(int argc, char **argv) {
  /* --oracle-cli <docs_blob> <probe_file> <target_vocab> <min_freq> <special_mask> <artifact_out> */
  if (argc != 8) {
    fprintf(stderr, "bad --oracle-cli arguments\n");
    return 2;
  }
  uint8_t *blob = NULL;
  size_t blob_len = 0u;
  uint8_t *probe = NULL;
  size_t probe_len = 0u;
  if (!read_entire_file(argv[2], &blob, &blob_len) ||
      !read_entire_file(argv[3], &probe, &probe_len)) {
    free(blob);
    free(probe);
    return 3;
  }

  /* Parse docs blob: [u32_le doc_count] followed by doc_count * ([u32_le len][bytes]). */
  if (blob_len < 4u) {
    free(blob);
    free(probe);
    return 4;
  }
  const uint32_t doc_count =
      (uint32_t)blob[0] | ((uint32_t)blob[1] << 8u) |
      ((uint32_t)blob[2] << 16u) | ((uint32_t)blob[3] << 24u);
  et_t2_bpe_document *docs =
      doc_count != 0u ? calloc(doc_count, sizeof(*docs)) : NULL;
  size_t off = 4u;
  for (uint32_t d = 0u; d < doc_count; ++d) {
    if (off + 4u > blob_len) {
      free(docs);
      free(blob);
      free(probe);
      return 4;
    }
    const uint32_t dlen =
        (uint32_t)blob[off] | ((uint32_t)blob[off + 1u] << 8u) |
        ((uint32_t)blob[off + 2u] << 16u) | ((uint32_t)blob[off + 3u] << 24u);
    off += 4u;
    if (off + dlen > blob_len) {
      free(docs);
      free(blob);
      free(probe);
      return 4;
    }
    docs[d].bytes = blob + off;
    docs[d].byte_count = dlen;
    off += dlen;
  }

  const et_t2_bpe_train_config cfg = {
      .target_vocab_size = (uint32_t)strtoul(argv[4], NULL, 10),
      .min_frequency = (uint32_t)strtoul(argv[5], NULL, 10),
      .special_mask = (uint32_t)strtoul(argv[6], NULL, 10),
  };

  et_t2_bpe_model *model = NULL;
  if (et_t2_bpe_train(docs, doc_count, &cfg, &model) != ET_T2_BPE_OK) {
    free(docs);
    free(blob);
    free(probe);
    return 5;
  }

  uint8_t sha[32];
  if (et_t2_bpe_save_file(model, argv[7], sha) != ET_T2_BPE_OK) {
    et_t2_bpe_destroy(model);
    free(docs);
    free(blob);
    free(probe);
    return 6;
  }

  et_t2_bpe_info info;
  et_t2_bpe_get_info(model, &info);
  printf("INFO\t%" PRIu32 "\t%" PRIu32 "\t%" PRIu32 "\t%" PRIu32 "\t%" PRIu32
         "\t%" PRIu32 "\t%" PRIu32 "\n",
         info.vocab_size, info.merge_count, info.special_mask, info.bos_id,
         info.eos_id, info.pad_id, info.unk_id);
  for (uint32_t r = 0u; r < info.merge_count; ++r) {
    et_t2_bpe_merge m;
    et_t2_bpe_get_merge(model, r, &m);
    printf("MERGE\t%" PRIu32 "\t%" PRIu32 "\t%" PRIu32 "\t%" PRIu32 "\n", r,
           m.left, m.right, m.new_id);
  }

  size_t tok_n = 0u;
  et_t2_bpe_encode(model, probe, probe_len, NULL, 0u, &tok_n);
  uint32_t *toks = tok_n != 0u ? malloc(tok_n * sizeof(*toks)) : NULL;
  et_t2_bpe_encode(model, probe, probe_len, toks, tok_n, &tok_n);
  printf("TOKENS");
  for (size_t i = 0u; i < tok_n; ++i) {
    printf("\t%" PRIu32, toks[i]);
  }
  printf("\n");

  size_t dec_n = 0u;
  et_t2_bpe_decode(model, toks, tok_n, NULL, 0u, &dec_n);
  uint8_t *dec = dec_n != 0u ? malloc(dec_n) : NULL;
  et_t2_bpe_decode(model, toks, tok_n, dec, dec_n, &dec_n);
  printf("DECODED\t");
  for (size_t i = 0u; i < dec_n; ++i) {
    printf("%02x", dec[i]);
  }
  printf("\nSHA256\t");
  for (size_t i = 0u; i < 32u; ++i) {
    printf("%02x", sha[i]);
  }
  printf("\n");

  free(dec);
  free(toks);
  et_t2_bpe_destroy(model);
  free(docs);
  free(blob);
  free(probe);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--oracle-cli") == 0) {
    return run_oracle_cli(argc, argv);
  }
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <tmp_dir>\n", argv[0]);
    return 2;
  }

  test_determinism_and_tie_breaking();
  test_compression_and_lossless_roundtrip(argv[1]);
  test_negative_tamper_and_symlink(argv[1]);
  test_bridge_api(argv[1]);

  if (failures != 0) {
    fprintf(stderr, "T2 BPE trainer native tests FAILED (%d failures)\n",
            failures);
    return 1;
  }
  printf("T2 BPE trainer native C11 + bridge + negative/tamper/symlink tests: PASS\n");
  return 0;
}
