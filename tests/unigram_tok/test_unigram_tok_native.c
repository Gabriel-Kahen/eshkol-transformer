#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/unigram_tok_abi.h"
#include "unigram_tok_eshkol_bridge.h"

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
#pragma STDC FP_CONTRACT OFF

static void die_msg(const char *msg, int line) {
  fprintf(stderr, "UNIGRAM_NATIVE_FAIL line=%d: %s\n", line, msg);
  exit(1);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) die_msg(#cond, __LINE__);                                     \
  } while (0)

/* ---- Standalone SHA-256 helper for crafting re-signed corrupt artifacts ---- */

static uint32_t test_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void test_sha256(const uint8_t *data, size_t size, uint8_t digest[32]) {
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
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  const size_t total = ((size + 9u + 63u) / 64u) * 64u;
  uint8_t *buf = calloc(total, 1u);
  CHECK(buf != NULL);
  if (size > 0u) memcpy(buf, data, size);
  buf[size] = 0x80u;
  const uint64_t bits = (uint64_t)size * UINT64_C(8);
  for (size_t i = 0u; i < 8u; ++i) {
    buf[total - 1u - i] = (uint8_t)(bits >> (8u * i));
  }
  for (size_t off = 0u; off < total; off += 64u) {
    uint32_t w[64];
    for (size_t i = 0u; i < 16u; ++i) {
      const size_t j = off + i * 4u;
      w[i] = ((uint32_t)buf[j] << 24u) | ((uint32_t)buf[j + 1u] << 16u) |
             ((uint32_t)buf[j + 2u] << 8u) | (uint32_t)buf[j + 3u];
    }
    for (size_t i = 16u; i < 64u; ++i) {
      const uint32_t s0 =
          test_rotr32(w[i - 15u], 7u) ^ test_rotr32(w[i - 15u], 18u) ^
          (w[i - 15u] >> 3u);
      const uint32_t s1 =
          test_rotr32(w[i - 2u], 17u) ^ test_rotr32(w[i - 2u], 19u) ^
          (w[i - 2u] >> 10u);
      w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (size_t i = 0u; i < 64u; ++i) {
      const uint32_t s1 =
          test_rotr32(e, 6u) ^ test_rotr32(e, 11u) ^ test_rotr32(e, 25u);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 =
          test_rotr32(a, 2u) ^ test_rotr32(a, 13u) ^ test_rotr32(a, 22u);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  free(buf);
  for (size_t i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)h[i];
  }
}

static void put_le32_t(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void put_f64_le_t(uint8_t *dst, double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  for (size_t i = 0u; i < 8u; ++i) {
    dst[i] = (uint8_t)((bits >> (8u * i)) & UINT64_C(0xff));
  }
}

static uint64_t f64_to_u64(double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static void print_hex(const uint8_t *data, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    printf("%02x", (unsigned)data[i]);
  }
}

/* ---- Test 1: Viterbi & Forward-Backward vs Exhaustive Brute Force ---- */

static double test_log_add_exp(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  if (a < b) {
    const double t = a;
    a = b;
    b = t;
  }
  return a + log1p(exp(b - a));
}

static void test_viterbi_and_fb_bruteforce(void) {
  uint8_t raw_bytes[256];
  et_unigram_piece_spec_v1 specs[264];
  for (uint32_t i = 0u; i < 256u; ++i) {
    raw_bytes[i] = (uint8_t)i;
    specs[i].bytes = &raw_bytes[i];
    specs[i].byte_count = 1u;
    specs[i].log_prob = -6.0;
  }
  /* Make a, b, c, d distinct scores plus overlapping multi-byte pieces,
   * including deliberate exact score tie: "ab" (-3.0) + "cd" (-3.0) == "abcd" (-6.0). */
  specs['a'].log_prob = -2.0;
  specs['b'].log_prob = -2.5;
  specs['c'].log_prob = -1.5;
  specs['d'].log_prob = -2.25;

  const char *extras[8] = {"ab", "bc", "cd", "abc", "bcd", "abcd", "ba", "aba"};
  const double escores[8] = {-3.0, -3.5, -3.0, -4.25, -4.5, -6.0, -3.25, -4.75};
  for (uint32_t k = 0u; k < 8u; ++k) {
    specs[256u + k].bytes = (const uint8_t *)extras[k];
    specs[256u + k].byte_count = strlen(extras[k]);
    specs[256u + k].log_prob = escores[k];
  }

  et_unigram_model *model = NULL;
  CHECK(et_unigram_create_v1(specs, 264u, ET_UNIGRAM_SPECIAL_NONE, &model) ==
        ET_UNIGRAM_OK);

  const char *probes[5] = {"abcd", "ababcd", "abababa", "cdabbc", "a"};
  for (size_t p = 0u; p < 5u; ++p) {
    const uint8_t *s = (const uint8_t *)probes[p];
    const size_t n = strlen(probes[p]);
    CHECK(n >= 1u && n <= 10u);

    uint32_t vit_toks[32];
    size_t vit_cnt = 0u;
    double vit_score = 0.0;
    CHECK(et_unigram_encode_v1(model, s, n, vit_toks, 32u, &vit_cnt,
                               &vit_score) == ET_UNIGRAM_OK);

    double fb_log_z = 0.0;
    double fb_exp[264];
    CHECK(et_unigram_forward_backward_v1(model, s, n, &fb_log_z, fb_exp,
                                         264u) == ET_UNIGRAM_OK);

    /* Exhaustively enumerate all 2^(n-1) boundary masks. */
    const uint32_t num_masks = 1u << (n - 1u);
    double bf_best = -INFINITY;
    double bf_log_z = -INFINITY;
    double *seg_scores = malloc((size_t)num_masks * sizeof(double));
    uint32_t *seg_valid = calloc((size_t)num_masks, sizeof(uint32_t));
    CHECK(seg_scores != NULL && seg_valid != NULL);

    for (uint32_t mask = 0u; mask < num_masks; ++mask) {
      size_t start = 0u;
      double sc = 0.0;
      int valid = 1;
      for (size_t cut = 0u; cut < n; ++cut) {
        if (cut + 1u == n || ((mask >> cut) & 1u) != 0u) {
          const size_t seg_len = cut + 1u - start;
          int found = -1;
          for (uint32_t tid = 0u; tid < 264u; ++tid) {
            if (specs[tid].byte_count == seg_len &&
                memcmp(specs[tid].bytes, s + start, seg_len) == 0) {
              found = (int)tid;
              break;
            }
          }
          if (found < 0) {
            valid = 0;
            break;
          }
          sc += specs[found].log_prob;
          start = cut + 1u;
        }
      }
      if (valid) {
        seg_valid[mask] = 1u;
        seg_scores[mask] = sc;
        if (sc > bf_best) bf_best = sc;
        bf_log_z = test_log_add_exp(bf_log_z, sc);
      }
    }

    CHECK(fabs(vit_score - bf_best) < 1e-12);
    CHECK(fabs(fb_log_z - bf_log_z) < 1e-12);

    /* Verify exact tie-break on "abcd": "abcd" (-6.0, len 4) beats "ab"+"cd" (-6.0, len 2). */
    if (p == 0u) {
      CHECK(vit_cnt == 1u);
      CHECK(vit_toks[0] == 256u + 5u); /* "abcd" */
    }

    /* Verify brute-force expected counts against Forward-Backward E[c_i]. */
    double bf_exp[264];
    memset(bf_exp, 0, sizeof(bf_exp));
    for (uint32_t mask = 0u; mask < num_masks; ++mask) {
      if (!seg_valid[mask]) continue;
      const double prob = exp(seg_scores[mask] - bf_log_z);
      size_t start = 0u;
      for (size_t cut = 0u; cut < n; ++cut) {
        if (cut + 1u == n || ((mask >> cut) & 1u) != 0u) {
          const size_t seg_len = cut + 1u - start;
          for (uint32_t tid = 0u; tid < 264u; ++tid) {
            if (specs[tid].byte_count == seg_len &&
                memcmp(specs[tid].bytes, s + start, seg_len) == 0) {
              bf_exp[tid] += prob;
              break;
            }
          }
          start = cut + 1u;
        }
      }
    }

    double sum_len_weighted = 0.0;
    for (uint32_t tid = 0u; tid < 264u; ++tid) {
      CHECK(fabs(fb_exp[tid] - bf_exp[tid]) < 1e-11);
      sum_len_weighted += fb_exp[tid] * (double)specs[tid].byte_count;
    }
    /* Conservation identity: sum_i E[c_i] * len(i) == N. */
    CHECK(fabs(sum_len_weighted - (double)n) < 1e-11);

    free(seg_scores);
    free(seg_valid);
  }

  et_unigram_destroy_v1(model);
}

/* ---- Test 2: Forward-Backward Central Finite Difference Gradient Check ---- */

static void check_fb_central_fd(const et_unigram_model *model,
                                const char *text) {
  et_unigram_info_v1 info;
  CHECK(et_unigram_get_info_v1(model, &info) == ET_UNIGRAM_OK);
  double *expected = calloc((size_t)info.active_count, sizeof(double));
  CHECK(expected != NULL);

  double log_z = 0.0;
  CHECK(et_unigram_forward_backward_v1(model, (const uint8_t *)text,
                                       strlen(text), &log_z, expected,
                                       (size_t)info.active_count) ==
        ET_UNIGRAM_OK);

  const double eps = 1e-6;
  double max_err = 0.0;
  size_t nonzero_checked = 0u;
  for (uint32_t tid = 0u; tid < info.active_count; ++tid) {
    /* Check all active pieces with nonzero expected count plus every 16th zero piece. */
    if (expected[tid] == 0.0 && (tid % 16u) != 0u) continue;
    double lz_plus = 0.0;
    double lz_minus = 0.0;
    CHECK(et_unigram_aot_perturbed_log_z_cstr(model, text, (int64_t)tid, eps,
                                              &lz_plus) == ET_UNIGRAM_OK);
    CHECK(et_unigram_aot_perturbed_log_z_cstr(model, text, (int64_t)tid, -eps,
                                              &lz_minus) == ET_UNIGRAM_OK);
    const double fd = (lz_plus - lz_minus) / (2.0 * eps);
    const double err = fabs(fd - expected[tid]);
    if (err > max_err) max_err = err;
    CHECK(err < 1e-6);
    if (expected[tid] > 1e-6) ++nonzero_checked;
  }
  CHECK(nonzero_checked >= 8u);
  CHECK(max_err < 1e-6);
  free(expected);
}

/* ---- Test 3: Training Determinism, Convergence, Both M-steps, FFBS, Round-Trip ---- */

static void test_training_ffbs_and_roundtrip(const char *work_dir) {
  const char *doc_strs[4] = {
      "eshkol transformer unigram viterbi lattice tokenizer forward backward em "
      "eshkol transformer unigram viterbi lattice tokenizer forward backward em",
      "deterministic sentencepiece subword regularization expectation maximization "
      "deterministic sentencepiece subword regularization expectation maximization",
      "eshkol transformer attention rmsnorm swiglu checkpoint artifact sha256 "
      "eshkol transformer attention rmsnorm swiglu checkpoint artifact sha256",
      "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xf0\x9f\x9a\x80 eshkol "
      "caf\xc3\xa9 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xf0\x9f\x9a\x80 eshkol",
  };
  et_unigram_document_v1 docs[4];
  for (size_t i = 0u; i < 4u; ++i) {
    docs[i].bytes = (const uint8_t *)doc_strs[i];
    docs[i].byte_count = strlen(doc_strs[i]);
  }

  for (uint32_t mode = 0u; mode <= 1u; ++mode) {
    et_unigram_train_config_v1 cfg;
    cfg.struct_size = (uint32_t)sizeof(et_unigram_train_config_v1);
    cfg.target_vocab_size = 320u;
    cfg.seed_vocab_size = 1024u;
    cfg.max_piece_bytes = 16u;
    cfg.min_frequency = 2u;
    cfg.num_em_sub_iters = 2u;
    cfg.final_em_iters = 5u;
    cfg.mstep_mode = mode;
    cfg.special_mask = ET_UNIGRAM_SPECIAL_ALL;
    cfg.reserved = 0u;
    cfg.shrinking_factor = 0.75;
    cfg.smoothing_alpha = (mode == ET_UNIGRAM_MSTEP_DIGAMMA_VB) ? 0.5 : 1e-3;

    et_unigram_model *m1 = NULL;
    et_unigram_model *m2 = NULL;
    et_unigram_train_stats_v1 st1;
    et_unigram_train_stats_v1 st2;
    CHECK(et_unigram_train_v1(docs, 4u, &cfg, &m1, &st1) == ET_UNIGRAM_OK);
    CHECK(et_unigram_train_v1(docs, 4u, &cfg, &m2, &st2) == ET_UNIGRAM_OK);

    CHECK(st1.final_active_count == 316u); /* 320 - 4 specials */
    CHECK(st1.pruning_rounds > 0u);
    CHECK(st1.final_em_end_ll >= st1.final_em_start_ll - 1e-9);
    CHECK(f64_to_u64(st1.final_em_end_ll) == f64_to_u64(st2.final_em_end_ll));

    /* Verify normalized probability sum == 1.0 and bitwise determinism. */
    et_unigram_info_v1 info;
    CHECK(et_unigram_get_info_v1(m1, &info) == ET_UNIGRAM_OK);
    CHECK(info.vocab_size == 320u);
    CHECK(info.active_count == 316u);
    CHECK(info.learned_count == 60u);
    CHECK(info.bos_id == 316u && info.eos_id == 317u &&
          info.pad_id == 318u && info.unk_id == 319u);

    double sum_p = 0.0;
    for (uint32_t i = 0u; i < info.active_count; ++i) {
      double lp = 0.0;
      CHECK(et_unigram_get_piece_v1(m1, i, NULL, 0u, NULL, &lp) ==
            ET_UNIGRAM_OK);
      CHECK(isfinite(lp) && lp < 0.0);
      sum_p += exp(lp);
    }
    CHECK(fabs(sum_p - 1.0) < 1e-11);

    size_t sz1 = 0u, sz2 = 0u;
    CHECK(et_unigram_serialize_v1(m1, NULL, 0u, &sz1, NULL) == ET_UNIGRAM_OK);
    CHECK(et_unigram_serialize_v1(m2, NULL, 0u, &sz2, NULL) == ET_UNIGRAM_OK);
    CHECK(sz1 == sz2);
    uint8_t *b1 = malloc(sz1);
    uint8_t *b2 = malloc(sz2);
    uint8_t sha1[32], sha2[32];
    CHECK(b1 != NULL && b2 != NULL);
    CHECK(et_unigram_serialize_v1(m1, b1, sz1, &sz1, sha1) == ET_UNIGRAM_OK);
    CHECK(et_unigram_serialize_v1(m2, b2, sz2, &sz2, sha2) == ET_UNIGRAM_OK);
    CHECK(memcmp(b1, b2, sz1) == 0);
    CHECK(memcmp(sha1, sha2, 32u) == 0);
    free(b1);
    free(b2);

    /* Central FD verification on the trained model. */
    const char *probe =
        "eshkol transformer unigram viterbi lattice tokenizer checkpoint";
    check_fb_central_fd(m1, probe);

    /* Compression check (> 1.5x) & encode_ex <bos>/<eos> framing. */
    uint32_t toks[256];
    size_t tcnt = 0u;
    double vscore = 0.0;
    CHECK(et_unigram_encode_ex_v1(m1, (const uint8_t *)probe, strlen(probe), 1u,
                                  1u, toks, 256u, &tcnt, &vscore) ==
          ET_UNIGRAM_OK);
    CHECK(toks[0] == info.bos_id && toks[tcnt - 1u] == info.eos_id);
    CHECK((double)strlen(probe) / (double)(tcnt - 2u) > 1.5);

    uint8_t dec[256];
    size_t dcnt = 0u;
    CHECK(et_unigram_decode_v1(m1, toks, tcnt, dec, 256u, &dcnt) ==
          ET_UNIGRAM_OK);
    CHECK(dcnt == strlen(probe) && memcmp(dec, probe, dcnt) == 0);

    /* Kudo (2018) FFBS subword regularization: deterministic & 100% lossless. */
    uint32_t s1[256], s2[256], s3[256];
    size_t sc1 = 0u, sc2 = 0u, sc3 = 0u;
    double ss1 = 0.0, ss2 = 0.0, ss3 = 0.0;
    CHECK(et_unigram_sample_v1(m1, (const uint8_t *)probe, strlen(probe), 0.5,
                               UINT64_C(12345), s1, 256u, &sc1, &ss1) ==
          ET_UNIGRAM_OK);
    CHECK(et_unigram_sample_v1(m1, (const uint8_t *)probe, strlen(probe), 0.5,
                               UINT64_C(12345), s2, 256u, &sc2, &ss2) ==
          ET_UNIGRAM_OK);
    CHECK(sc1 == sc2 && memcmp(s1, s2, sc1 * sizeof(uint32_t)) == 0);
    CHECK(f64_to_u64(ss1) == f64_to_u64(ss2));
    CHECK(ss1 <= vscore + 1e-12); /* Viterbi is the global optimum! */

    CHECK(et_unigram_decode_v1(m1, s1, sc1, dec, 256u, &dcnt) == ET_UNIGRAM_OK);
    CHECK(dcnt == strlen(probe) && memcmp(dec, probe, dcnt) == 0);

    /* Sample a second seed and confirm lossless decode. */
    CHECK(et_unigram_sample_v1(m1, (const uint8_t *)probe, strlen(probe), 0.25,
                               UINT64_C(999999), s3, 256u, &sc3, &ss3) ==
          ET_UNIGRAM_OK);
    CHECK(et_unigram_decode_v1(m1, s3, sc3, dec, 256u, &dcnt) == ET_UNIGRAM_OK);
    CHECK(dcnt == strlen(probe) && memcmp(dec, probe, dcnt) == 0);

    /* All 256 raw bytes + embedded NULs + invalid UTF-8 + multibyte UTF-8. */
    uint8_t blob[300];
    size_t blen = 0u;
    blob[blen++] = 0x00u;
    blob[blen++] = 0x00u;
    for (uint32_t b = 0u; b < 256u; ++b) blob[blen++] = (uint8_t)b;
    /* Invalid UTF-8 surrogates / overlong / lone continuation bytes. */
    const uint8_t bad_utf8[8] = {0x80u, 0xC0u, 0xC1u, 0xEDu,
                                 0xA0u, 0x80u, 0xFEu, 0xFFu};
    memcpy(blob + blen, bad_utf8, 8u);
    blen += 8u;
    const char *utf8_tail = "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xf0\x9f\x9a\x80";
    memcpy(blob + blen, utf8_tail, strlen(utf8_tail));
    blen += strlen(utf8_tail);
    blob[blen++] = 0x00u;

    uint32_t btoks[512];
    size_t btcnt = 0u;
    uint8_t bdec[512];
    size_t bdcnt = 0u;
    CHECK(et_unigram_encode_v1(m1, blob, blen, btoks, 512u, &btcnt, NULL) ==
          ET_UNIGRAM_OK);
    CHECK(et_unigram_decode_v1(m1, btoks, btcnt, bdec, 512u, &bdcnt) ==
          ET_UNIGRAM_OK);
    CHECK(bdcnt == blen && memcmp(bdec, blob, blen) == 0);

    /* Empty input round-trip. */
    size_t ecnt = 99u;
    double escore = 99.0;
    CHECK(et_unigram_encode_v1(m1, NULL, 0u, btoks, 512u, &ecnt, &escore) ==
          ET_UNIGRAM_OK);
    CHECK(ecnt == 0u && escore == 0.0);
    size_t edcnt = 99u;
    CHECK(et_unigram_decode_v1(m1, NULL, 0u, bdec, 512u, &edcnt) ==
          ET_UNIGRAM_OK);
    CHECK(edcnt == 0u);

    /* Save/reload round-trip + permissions + tamper + symlink checks (mode 0). */
    if (mode == 0u) {
      char path[512];
      char sym_path[512];
      snprintf(path, sizeof(path), "%s/model.etug", work_dir);
      snprintf(sym_path, sizeof(sym_path), "%s/model_link.etug", work_dir);

      uint8_t save_sha[32], load_sha[32];
      CHECK(et_unigram_save_file_v1(m1, path, save_sha) == ET_UNIGRAM_OK);

      struct stat st_meta;
      CHECK(lstat(path, &st_meta) == 0);
      CHECK(S_ISREG(st_meta.st_mode));
      CHECK((st_meta.st_mode & 0777) == 0600);

      et_unigram_model *reloaded = NULL;
      CHECK(et_unigram_load_file_v1(path, &reloaded, load_sha) ==
            ET_UNIGRAM_OK);
      CHECK(memcmp(save_sha, load_sha, 32u) == 0);

      uint32_t rtoks[256];
      size_t rtcnt = 0u;
      double rscore = 0.0;
      CHECK(et_unigram_encode_ex_v1(reloaded, (const uint8_t *)probe,
                                    strlen(probe), 1u, 1u, rtoks, 256u, &rtcnt,
                                    &rscore) == ET_UNIGRAM_OK);
      CHECK(rtcnt == tcnt && memcmp(rtoks, toks, tcnt * sizeof(uint32_t)) == 0);
      CHECK(f64_to_u64(rscore) == f64_to_u64(vscore));

      /* Symlink rejection on both load and save overwrite. */
      (void)unlink(sym_path);
      CHECK(symlink(path, sym_path) == 0);
      et_unigram_model *bad = (et_unigram_model *)0x1;
      CHECK(et_unigram_load_file_v1(sym_path, &bad, NULL) == ET_UNIGRAM_ERR_IO);
      CHECK(bad == NULL);
      CHECK(et_unigram_save_file_v1(m1, sym_path, NULL) == ET_UNIGRAM_ERR_IO);
      (void)unlink(sym_path);

      /* Directory rejection on load/save. */
      CHECK(et_unigram_load_file_v1(work_dir, &bad, NULL) == ET_UNIGRAM_ERR_IO);
      CHECK(et_unigram_save_file_v1(m1, work_dir, NULL) == ET_UNIGRAM_ERR_IO);

      et_unigram_destroy_v1(reloaded);
    }

    et_unigram_destroy_v1(m1);
    et_unigram_destroy_v1(m2);
  }
}

/* ---- Test 4: Tamper, Re-Signed Semantic Corruption, Bounds, & FP-Env Traps ---- */

static void test_negatives_and_fp_env(void) {
  uint8_t raw[256];
  et_unigram_piece_spec_v1 specs[258];
  for (uint32_t i = 0u; i < 256u; ++i) {
    raw[i] = (uint8_t)i;
    specs[i].bytes = &raw[i];
    specs[i].byte_count = 1u;
    specs[i].log_prob = -6.0;
  }
  const uint8_t p1[2] = {'a', 'b'};
  const uint8_t p2[2] = {'c', 'd'};
  specs[256].bytes = p1;
  specs[256].byte_count = 2u;
  specs[256].log_prob = -3.0;
  specs[257].bytes = p2;
  specs[257].byte_count = 2u;
  specs[257].log_prob = -3.5;

  et_unigram_model *m = NULL;
  CHECK(et_unigram_create_v1(specs, 258u, ET_UNIGRAM_SPECIAL_NONE, &m) ==
        ET_UNIGRAM_OK);

  /* Serialize valid model to mutate in-memory for deserialization negatives. */
  size_t sz = 0u;
  CHECK(et_unigram_serialize_v1(m, NULL, 0u, &sz, NULL) == ET_UNIGRAM_OK);
  uint8_t *art = malloc(sz);
  uint8_t *mut = malloc(sz + 4u);
  CHECK(art != NULL && mut != NULL);
  CHECK(et_unigram_serialize_v1(m, art, sz, &sz, NULL) == ET_UNIGRAM_OK);

  et_unigram_model *bad = NULL;

  /* 1. Truncated -> FORMAT. */
  CHECK(et_unigram_deserialize_v1(art, sz - 5u, &bad, NULL) ==
        ET_UNIGRAM_ERR_FORMAT);
  CHECK(bad == NULL);

  /* 2. Trailing garbage -> FORMAT. */
  memcpy(mut, art, sz);
  mut[sz] = 0u;
  CHECK(et_unigram_deserialize_v1(mut, sz + 1u, &bad, NULL) ==
        ET_UNIGRAM_ERR_FORMAT);

  /* 3. Bad magic -> MAGIC. */
  memcpy(mut, art, sz);
  mut[0] = 'X';
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) == ET_UNIGRAM_ERR_MAGIC);

  /* 4. Bad version -> VERSION. */
  memcpy(mut, art, sz);
  mut[8] = 2u;
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_VERSION);

  /* 5. Bit-flip payload without re-signing -> CHECKSUM. */
  memcpy(mut, art, sz);
  mut[48] ^= 0x01u;
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_CHECKSUM);

  /* 6. Re-signed missing/corrupted byte fallback (piece 0 byte = 0x01) -> INVALID_VOCAB. */
  memcpy(mut, art, sz);
  mut[48u + 10u] = 0x01u;
  test_sha256(mut, sz - 32u, mut + sz - 32u);
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);

  /* 7. Re-signed duplicate piece (make piece 257 == "ab" same as piece 256) -> INVALID_VOCAB. */
  memcpy(mut, art, sz);
  const size_t p257_bytes_off = 48u + (256u * 11u) + 12u + 10u;
  mut[p257_bytes_off] = 'a';
  mut[p257_bytes_off + 1u] = 'b';
  test_sha256(mut, sz - 32u, mut + sz - 32u);
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);

  /* 8. Re-signed NaN score -> NONFINITE. */
  memcpy(mut, art, sz);
  put_f64_le_t(mut + 48u, (double)NAN);
  test_sha256(mut, sz - 32u, mut + sz - 32u);
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_NONFINITE);

  /* 9. Re-signed positive log_prob (+0.5) -> INVALID_VOCAB. */
  memcpy(mut, art, sz);
  put_f64_le_t(mut + 48u, 0.5);
  test_sha256(mut, sz - 32u, mut + sz - 32u);
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);

  /* 10. Re-signed non-zero flags -> FORMAT. */
  memcpy(mut, art, sz);
  put_le32_t(mut + 12u, 1u);
  test_sha256(mut, sz - 32u, mut + sz - 32u);
  CHECK(et_unigram_deserialize_v1(mut, sz, &bad, NULL) ==
        ET_UNIGRAM_ERR_FORMAT);

  free(art);
  free(mut);

  /* Direct create_v1 negatives: <256 pieces, duplicate piece, NaN, +Inf, positive. */
  CHECK(et_unigram_create_v1(specs, 255u, 0u, &bad) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);
  specs[257].bytes = p1; /* duplicate */
  CHECK(et_unigram_create_v1(specs, 258u, 0u, &bad) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);
  specs[257].bytes = p2;
  specs[257].log_prob = (double)INFINITY;
  CHECK(et_unigram_create_v1(specs, 258u, 0u, &bad) ==
        ET_UNIGRAM_ERR_NONFINITE);
  specs[257].log_prob = 0.25;
  CHECK(et_unigram_create_v1(specs, 258u, 0u, &bad) ==
        ET_UNIGRAM_ERR_INVALID_VOCAB);
  specs[257].log_prob = -3.5;

  /* Train config boundary checks: vocab < 256, vocab > 4096, NaN alpha. */
  et_unigram_train_config_v1 cfg = {
      (uint32_t)sizeof(et_unigram_train_config_v1),
      200u, 512u, 12u, 2u, 2u, 4u, ET_UNIGRAM_MSTEP_MLE, 0u, 0u, 0.75, 1e-3};
  CHECK(et_unigram_train_v1(NULL, 0u, &cfg, &bad, NULL) ==
        ET_UNIGRAM_ERR_ARGUMENT);
  cfg.target_vocab_size = 5000u;
  CHECK(et_unigram_train_v1(NULL, 0u, &cfg, &bad, NULL) ==
        ET_UNIGRAM_ERR_BOUNDS);
  cfg.target_vocab_size = 256u;
  cfg.smoothing_alpha = (double)NAN;
  CHECK(et_unigram_train_v1(NULL, 0u, &cfg, &bad, NULL) ==
        ET_UNIGRAM_ERR_NONFINITE);

  /* Decode OOB token ID leaves output buffer untouched. */
  uint8_t sentinel_out[8];
  memset(sentinel_out, 0xAA, sizeof(sentinel_out));
  const uint32_t oob_toks[2] = {65u, 999u};
  size_t out_n = 0u;
  CHECK(et_unigram_decode_v1(m, oob_toks, 2u, sentinel_out, 8u, &out_n) ==
        ET_UNIGRAM_ERR_TOKEN_ID);
  for (size_t i = 0u; i < 8u; ++i) CHECK(sentinel_out[i] == 0xAAu);

  /* Insufficient encode capacity leaves output buffer untouched. */
  uint32_t tok_sent[2] = {0xDEADBEEFu, 0xCAFEBABEu};
  size_t need = 0u;
  CHECK(et_unigram_encode_v1(m, (const uint8_t *)"abcdef", 6u, tok_sent, 1u,
                             &need, NULL) == ET_UNIGRAM_ERR_BOUNDS);
  CHECK(need == 4u);
  CHECK(tok_sent[0] == 0xDEADBEEFu && tok_sent[1] == 0xCAFEBABEu);

  /* FP-env trap check (FE_UPWARD -> ET_UNIGRAM_ERR_FP_ENV, then restore). */
  const int saved_round = fegetround();
  CHECK(fesetround(FE_UPWARD) == 0);
  double lz = 0.0;
  CHECK(et_unigram_encode_v1(m, (const uint8_t *)"abcd", 4u, NULL, 0u, &need,
                             NULL) == ET_UNIGRAM_ERR_FP_ENV);
  CHECK(et_unigram_forward_backward_v1(m, (const uint8_t *)"abcd", 4u, &lz,
                                       NULL, 0u) == ET_UNIGRAM_ERR_FP_ENV);
  CHECK(et_unigram_sample_v1(m, (const uint8_t *)"abcd", 4u, 1.0, 1u, NULL, 0u,
                             &need, NULL) == ET_UNIGRAM_ERR_FP_ENV);
  CHECK(et_unigram_create_v1(specs, 258u, 0u, &bad) == ET_UNIGRAM_ERR_FP_ENV);
  CHECK(fesetround(saved_round) == 0);

  et_unigram_destroy_v1(m);
}

/* ---- Oracle CLI Mode for Independent Python Verification ---- */

static uint8_t *read_whole_file(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  CHECK(f != NULL);
  CHECK(fseek(f, 0, SEEK_END) == 0);
  const long sz = ftell(f);
  CHECK(sz >= 0);
  CHECK(fseek(f, 0, SEEK_SET) == 0);
  uint8_t *buf = malloc((size_t)sz + 1u);
  CHECK(buf != NULL);
  if (sz > 0) CHECK(fread(buf, 1u, (size_t)sz, f) == (size_t)sz);
  fclose(f);
  *out_size = (size_t)sz;
  return buf;
}

static uint32_t rd_le32(const uint8_t *p) {
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static int run_oracle_cli(int argc, char **argv) {
  if (argc != 15) {
    fprintf(stderr, "invalid --oracle-cli arg count: %d\n", argc);
    return 2;
  }
  size_t docs_sz = 0u, probe_sz = 0u;
  uint8_t *docs_blob = read_whole_file(argv[2], &docs_sz);
  uint8_t *probe_blob = read_whole_file(argv[3], &probe_sz);

  CHECK(docs_sz >= 4u);
  const uint32_t doc_count = rd_le32(docs_blob);
  et_unigram_document_v1 *docs =
      calloc((size_t)doc_count + 1u, sizeof(et_unigram_document_v1));
  CHECK(docs != NULL);
  size_t pos = 4u;
  for (uint32_t d = 0u; d < doc_count; ++d) {
    CHECK(pos + 4u <= docs_sz);
    const uint32_t dlen = rd_le32(docs_blob + pos);
    pos += 4u;
    CHECK(pos + (size_t)dlen <= docs_sz);
    docs[d].bytes = docs_blob + pos;
    docs[d].byte_count = (size_t)dlen;
    pos += (size_t)dlen;
  }
  CHECK(pos == docs_sz);

  et_unigram_train_config_v1 cfg;
  cfg.struct_size = (uint32_t)sizeof(et_unigram_train_config_v1);
  cfg.target_vocab_size = (uint32_t)strtoul(argv[4], NULL, 10);
  cfg.seed_vocab_size = (uint32_t)strtoul(argv[5], NULL, 10);
  cfg.max_piece_bytes = (uint32_t)strtoul(argv[6], NULL, 10);
  cfg.min_frequency = (uint32_t)strtoul(argv[7], NULL, 10);
  cfg.num_em_sub_iters = (uint32_t)strtoul(argv[8], NULL, 10);
  cfg.final_em_iters = (uint32_t)strtoul(argv[9], NULL, 10);
  cfg.mstep_mode = (uint32_t)strtoul(argv[10], NULL, 10);
  cfg.special_mask = (uint32_t)strtoul(argv[11], NULL, 10);
  cfg.reserved = 0u;
  cfg.shrinking_factor = strtod(argv[12], NULL);
  cfg.smoothing_alpha = strtod(argv[13], NULL);
  const char *art_path = argv[14];

  et_unigram_model *model = NULL;
  et_unigram_train_stats_v1 stats;
  CHECK(et_unigram_train_v1(docs, (size_t)doc_count, &cfg, &model, &stats) ==
        ET_UNIGRAM_OK);

  uint8_t sha[32];
  CHECK(et_unigram_save_file_v1(model, art_path, sha) == ET_UNIGRAM_OK);

  et_unigram_info_v1 info;
  CHECK(et_unigram_get_info_v1(model, &info) == ET_UNIGRAM_OK);
  printf("INFO\t%u\t%u\t%u\t%u\t%u\t%u\t%u\t%u\n", info.vocab_size,
         info.active_count, info.learned_count, info.special_mask, info.bos_id,
         info.eos_id, info.pad_id, info.unk_id);
  printf("STATS\t%u\t%u\t%u\t%u\t%016llx\t%016llx\t%016llx\n",
         stats.seed_active_count, stats.final_active_count,
         stats.pruning_rounds, stats.total_em_iters,
         (unsigned long long)f64_to_u64(stats.initial_log_likelihood),
         (unsigned long long)f64_to_u64(stats.final_em_start_ll),
         (unsigned long long)f64_to_u64(stats.final_em_end_ll));

  for (uint32_t i = 0u; i < info.active_count; ++i) {
    uint8_t pbuf[ET_UNIGRAM_MAX_PIECE_BYTES];
    size_t plen = 0u;
    double lp = 0.0;
    CHECK(et_unigram_get_piece_v1(model, i, pbuf, sizeof(pbuf), &plen, &lp) ==
          ET_UNIGRAM_OK);
    printf("PIECE\t%u\t%016llx\t", i, (unsigned long long)f64_to_u64(lp));
    print_hex(pbuf, plen);
    putchar('\n');
  }

  size_t tok_n = 0u;
  double vscore = 0.0;
  CHECK(et_unigram_encode_v1(model, probe_blob, probe_sz, NULL, 0u, &tok_n,
                             &vscore) == ET_UNIGRAM_OK);
  uint32_t *toks = malloc((tok_n + 1u) * sizeof(uint32_t));
  CHECK(toks != NULL);
  CHECK(et_unigram_encode_v1(model, probe_blob, probe_sz, toks, tok_n, &tok_n,
                             &vscore) == ET_UNIGRAM_OK);

  printf("TOKENS");
  for (size_t i = 0u; i < tok_n; ++i) printf("\t%u", toks[i]);
  putchar('\n');
  printf("VSCORE\t%016llx\n", (unsigned long long)f64_to_u64(vscore));

  size_t dec_n = 0u;
  CHECK(et_unigram_decode_v1(model, toks, tok_n, NULL, 0u, &dec_n) ==
        ET_UNIGRAM_OK);
  uint8_t *dec = malloc(dec_n + 1u);
  CHECK(dec != NULL);
  CHECK(et_unigram_decode_v1(model, toks, tok_n, dec, dec_n, &dec_n) ==
        ET_UNIGRAM_OK);
  printf("DECODED\t");
  print_hex(dec, dec_n);
  putchar('\n');

  double fb_lz = 0.0;
  double *fb_exp = calloc((size_t)info.active_count, sizeof(double));
  CHECK(fb_exp != NULL);
  CHECK(et_unigram_forward_backward_v1(model, probe_blob, probe_sz, &fb_lz,
                                       fb_exp, (size_t)info.active_count) ==
        ET_UNIGRAM_OK);
  printf("FBLOGZ\t%016llx\n", (unsigned long long)f64_to_u64(fb_lz));
  for (uint32_t i = 0u; i < info.active_count; ++i) {
    if (fb_exp[i] > 0.0) {
      printf("FBEXP\t%u\t%016llx\n", i,
             (unsigned long long)f64_to_u64(fb_exp[i]));
    }
  }
  printf("SHA256\t");
  print_hex(sha, 32u);
  putchar('\n');

  free(fb_exp);
  free(dec);
  free(toks);
  et_unigram_destroy_v1(model);
  free(docs);
  free(docs_blob);
  free(probe_blob);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--oracle-cli") == 0) {
    return run_oracle_cli(argc, argv);
  }
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <work_dir>\n", argv[0]);
    return 2;
  }

  test_viterbi_and_fb_bruteforce();
  test_training_ffbs_and_roundtrip(argv[1]);
  test_negatives_and_fp_env();

  printf("UNIGRAM_NATIVE_ALL_TESTS: PASS\n");
  return 0;
}
