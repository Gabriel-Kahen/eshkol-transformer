#define _GNU_SOURCE

#include "eshkol_transformer/kv_cache_quant_abi.h"
#include "kv_cache_quant_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* Allocation trap: enforces ZERO heap allocations across all hot-path KV
 * append/dequant/decode/truncate/reset/stats/codec calls when linked with
 * -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free. */
static int forbid_alloc = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!forbid_alloc);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!forbid_alloc);
  __real_free(ptr);
}

static uint32_t f32_bits(float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

static float bits_f32(uint32_t u) {
  float v = 0.0f;
  memcpy(&v, &u, sizeof(v));
  return v;
}

static float det_val(uint32_t seed, uint32_t idx, float step) {
  const int bucket =
      (int)((37u * seed + 19u * idx + (idx / 3u) * 7u) % 41u) - 20;
  return (float)bucket * step;
}

static void fill_kv_slice(float *k, float *v, uint32_t hkv, uint32_t hd,
                          uint32_t layer, uint32_t pos, int outlier_head1) {
  for (uint32_t h = 0u; h < hkv; ++h) {
    for (uint32_t d = 0u; d < hd; ++d) {
      const size_t idx = (size_t)h * (size_t)hd + (size_t)d;
      const uint32_t kseed = 11u + layer * 101u + pos * 7u + h * 29u;
      const uint32_t vseed = 53u + layer * 131u + pos * 13u + h * 31u;
      if (outlier_head1 && h == 1u) {
        /* Realistic LLM KV channel outlier: d=0 large outlier (~12.0),
         * d=1..hd-1 small activations (~0.01..0.20). */
        if (d == 0u) {
          k[idx] = ((pos % 2u) == 0u) ? 12.0f : -11.5f;
          v[idx] = ((pos % 3u) == 0u) ? -12.0f : 11.0f;
        } else {
          k[idx] = det_val(kseed, d, 0.01f);
          v[idx] = det_val(vseed, d, 0.01f);
        }
      } else {
        /* Normalized RMSNorm-scale KV activations in [-0.5, 0.5]. */
        k[idx] = det_val(kseed, d, 0.025f);
        v[idx] = det_val(vseed, d, 0.025f);
      }
    }
  }
}

static void fill_q(float *q, uint32_t hq, uint32_t hd, uint32_t seed) {
  for (uint32_t qh = 0u; qh < hq; ++qh) {
    for (uint32_t d = 0u; d < hd; ++d) {
      const size_t idx = (size_t)qh * (size_t)hd + (size_t)d;
      q[idx] = det_val(seed + qh * 17u, d, 0.0625f);
    }
  }
}

static int compute_top1_logit(const float *attn_out, uint32_t n_elems,
                              uint32_t vocab_size) {
  int best = 0;
  double best_val = -INFINITY;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    double logit = 0.0;
    for (uint32_t i = 0u; i < n_elems; ++i) {
      double w = (double)det_val(300u + v * 43u, i, 0.05f);
      if ((i % vocab_size) == v)
        w += 0.5;
      logit += w * (double)attn_out[i];
    }
    if (logit > best_val) {
      best_val = logit;
      best = (int)v;
    }
  }
  return best;
}

static void print_f32_hex_vec(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%08x", (unsigned)f32_bits(v[i]));
  }
  putchar('\n');
}

static void print_u8_hex_vec(const uint8_t *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%02x", (unsigned)v[i]);
  }
  putchar('\n');
}

/* ---- Mode: --oracle (dumps exact test vectors for Python reference) ---- */
static void run_oracle_dump(void) {
  /* 1. All 256 FP8 E4M3FN codepoint decodings. */
  puts("E4M3_DECODE_256");
  for (uint32_t c = 0u; c < 256u; ++c) {
    float fv = 0.0f;
    const et_kvq_status st = et_kvq_fp8_e4m3_decode_scalar((uint8_t)c, &fv);
    printf("%02x %d %08x\n", (unsigned)c, (int)st,
           (unsigned)(st == ET_KVQ_OK ? f32_bits(fv) : 0u));
  }

  /* 2. Key IEEE-754 boundary/tie f32 encodings into E4M3FN. */
  static const uint32_t enc_bits[] = {
      0x00000000u, /* +0.0 */
      0x80000000u, /* -0.0 */
      0x00000001u, /* min f32 subnormal */
      0x3A800000u, /* 2^-10 (exact tie between 0x00 even and 0x01 odd -> 0x00) */
      0x3A800001u, /* just above 2^-10 -> 0x01 */
      0x3B000000u, /* 2^-9 = 0x01 */
      0x3B400000u, /* 3 * 2^-10 (exact tie between 0x01 odd and 0x02 even -> 0x02) */
      0x3BC00000u, /* 7.5 * 2^-9? no, 6*2^-10 = 3*2^-9 */
      0x3C700000u, /* 7.5 * 2^-9 (tie between 0x07 odd and 0x08 even -> 0x08) */
      0x3C800000u, /* 2^-6 = min normal 0x08 */
      0x3F800000u, /* 1.0 = 0x38 */
      0xBF800000u, /* -1.0 = 0xB8 */
      0x3F880000u, /* 1.0625 (exact tie between 1.0 even and 1.125 odd -> 0x38) */
      0x3F980000u, /* 1.1875 (exact tie between 1.125 odd and 1.25 even -> 0x3A) */
      0x43E00000u, /* 448.0 = 0x7E */
      0x43E80000u, /* 464.0 (tie to even 0x7E) */
      0x43E80001u, /* > 464.0 (saturate to 0x7E) */
      0xC3F00000u, /* -480.0 (saturate to 0xFE) */
      0x7F800000u, /* +Inf -> NONFINITE */
      0xFF800000u, /* -Inf -> NONFINITE */
      0x7FC00000u, /* qNaN -> NONFINITE */
  };
  printf("E4M3_ENCODE %zu\n", ARRAY_COUNT(enc_bits));
  for (size_t i = 0u; i < ARRAY_COUNT(enc_bits); ++i) {
    uint8_t code = 0xAAu;
    const et_kvq_status st =
        et_kvq_fp8_e4m3_encode_scalar(bits_f32(enc_bits[i]), &code);
    printf("%08x %d %02x\n", (unsigned)enc_bits[i], (int)st,
           (unsigned)(st == ET_KVQ_OK ? code : 0u));
  }

  /* 3. Vector quantization + dequantization cases for INT8_SYM and FP8_E4M3FN. */
  const float vec0[8] = {0.0f, -0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  const float vec1[8] = {127.0f, -127.0f, 0.5f, 1.5f,
                         2.5f,   -0.5f,   -1.5f, -2.5f};
  const float vec2[8] = {24.0f,   0.015f, -0.031f, 0.092f,
                         -0.145f, 0.004f, -0.008f, 0.111f};
  const float vec3[8] = {bits_f32(1u),  bits_f32(2u),  bits_f32(5u),
                         bits_f32(10u), -bits_f32(3u), -bits_f32(7u),
                         0.0f,          bits_f32(12u)};
  const float *vecs[4] = {vec0, vec1, vec2, vec3};
  printf("VEC_CASES %zu\n", ARRAY_COUNT(vecs));
  for (size_t ci = 0u; ci < ARRAY_COUNT(vecs); ++ci) {
    int8_t qi8[8];
    uint8_t qf8[8];
    float sc_i8 = 0.0f, sc_f8 = 0.0f;
    float deq_i8[8], deq_f8[8];
    CHECK(et_kvq_int8_sym_quantize_vec(vecs[ci], 8u, qi8, &sc_i8) == ET_KVQ_OK);
    CHECK(et_kvq_int8_sym_dequantize_vec(qi8, 8u, sc_i8, deq_i8) == ET_KVQ_OK);
    CHECK(et_kvq_fp8_e4m3_quantize_vec(vecs[ci], 8u, qf8, &sc_f8) == ET_KVQ_OK);
    CHECK(et_kvq_fp8_e4m3_dequantize_vec(qf8, 8u, sc_f8, deq_f8) == ET_KVQ_OK);

    printf("VEC %zu 8\n", ci);
    print_f32_hex_vec(vecs[ci], 8u);
    printf("%08x\n", (unsigned)f32_bits(sc_i8));
    print_u8_hex_vec((const uint8_t *)qi8, 8u);
    print_f32_hex_vec(deq_i8, 8u);
    printf("%08x\n", (unsigned)f32_bits(sc_f8));
    print_u8_hex_vec(qf8, 8u);
    print_f32_hex_vec(deq_f8, 8u);
  }

  /* 4. End-to-end quantized KV cache GQA decode cases across FP32, INT8, FP8,
   *    full causal and sliding-window, with outlier head 1. */
  const et_kvq_format fmts[3] = {ET_KVQ_FMT_FP32, ET_KVQ_FMT_INT8_SYM,
                                 ET_KVQ_FMT_FP8_E4M3FN};
  const uint32_t windows[2] = {0u, 4u};
  const uint32_t nl = 2u, hkv = 2u, hq = 8u, hd = 16u, seq_len = 10u;
  const float scale = 1.0f / sqrtf((float)hd);
  printf("GQA_CASES %u\n", 6u);

  for (size_t fi = 0u; fi < 3u; ++fi) {
    const et_kvq_config cfg = {fmts[fi], nl, hkv, hd, 16u, 0u};
    et_kvq_cache *cache = NULL;
    CHECK(et_kvq_cache_create(&cfg, NULL, 0u, &cache) == ET_KVQ_OK);

    float k_all[2][10][2 * 16];
    float v_all[2][10][2 * 16];
    for (uint32_t l = 0u; l < nl; ++l) {
      for (uint32_t pos = 0u; pos < seq_len; ++pos) {
        fill_kv_slice(k_all[l][pos], v_all[l][pos], hkv, hd, l, pos, 1);
        CHECK(et_kvq_append_token_v1(cache, l, k_all[l][pos], v_all[l][pos]) ==
              ET_KVQ_OK);
      }
    }

    float q[8 * 16];
    float out[8 * 16];
    fill_q(q, hq, hd, 77u + (uint32_t)fi * 19u);

    for (size_t wi = 0u; wi < 2u; ++wi) {
      const uint32_t layer = (uint32_t)wi;
      const uint32_t win = windows[wi];
      CHECK(et_kvq_decode_gqa_v1(cache, layer, hq, scale, win, q, out) ==
            ET_KVQ_OK);
      printf("GQA %u %u %u %u %u %u %08x\n", (unsigned)fmts[fi],
             (unsigned)seq_len, (unsigned)hq, (unsigned)hkv, (unsigned)hd,
             (unsigned)win, (unsigned)f32_bits(scale));
      print_f32_hex_vec(q, (size_t)hq * (size_t)hd);
      for (uint32_t pos = 0u; pos < seq_len; ++pos) {
        float dk[2 * 16], dv[2 * 16];
        CHECK(et_kvq_dequant_slice_v1(cache, layer, pos, dk, dv) == ET_KVQ_OK);
        print_f32_hex_vec(k_all[layer][pos], (size_t)hkv * (size_t)hd);
        print_f32_hex_vec(v_all[layer][pos], (size_t)hkv * (size_t)hd);
        print_f32_hex_vec(dk, (size_t)hkv * (size_t)hd);
        print_f32_hex_vec(dv, (size_t)hkv * (size_t)hd);
        if (fmts[fi] != ET_KVQ_FMT_FP32) {
          for (uint32_t side = 0u; side < 2u; ++side) {
            for (uint32_t h = 0u; h < hkv; ++h) {
              uint8_t raw[16];
              float sc = 0.0f;
              CHECK(et_kvq_read_raw_vector(cache, layer, side, pos, h, raw,
                                           sizeof(raw), &sc) == ET_KVQ_OK);
              printf("%08x ", (unsigned)f32_bits(sc));
              print_u8_hex_vec(raw, hd);
            }
          }
        }
      }
      print_f32_hex_vec(out, (size_t)hq * (size_t)hd);
    }

    et_kvq_cache_destroy(cache);
  }
}

/* ---- Test 1: Exhaustive 256-codepoint FP8 E4M3FN & INT8 RNE codecs ---- */
static void test_codecs_exhaustive(void) {
  forbid_alloc = 1;

  uint32_t finite_count = 0u;
  uint32_t reject_count = 0u;
  for (uint32_t c = 0u; c < 256u; ++c) {
    float val = 1234.0f;
    const et_kvq_status dst = et_kvq_fp8_e4m3_decode_scalar((uint8_t)c, &val);
    if (c == 0x7Fu || c == 0xFFu) {
      CHECK(dst == ET_KVQ_ERR_NONFINITE);
      CHECK(f32_bits(val) == f32_bits(1234.0f));
      ++reject_count;
    } else {
      CHECK(dst == ET_KVQ_OK);
      CHECK(isfinite(val));
      uint8_t reenc = 0x55u;
      CHECK(et_kvq_fp8_e4m3_encode_scalar(val, &reenc) == ET_KVQ_OK);
      CHECK(reenc == (uint8_t)c);
      ++finite_count;
    }
  }
  CHECK(finite_count == 254u);
  CHECK(reject_count == 2u);

  /* Check exact RNE ties and saturation boundaries. */
  uint8_t code = 0u;
  CHECK(et_kvq_fp8_e4m3_encode_scalar(448.0f, &code) == ET_KVQ_OK &&
        code == 0x7Eu);
  CHECK(et_kvq_fp8_e4m3_encode_scalar(464.0f, &code) == ET_KVQ_OK &&
        code == 0x7Eu);
  CHECK(et_kvq_fp8_e4m3_encode_scalar(1000.0f, &code) == ET_KVQ_OK &&
        code == 0x7Eu);
  CHECK(et_kvq_fp8_e4m3_encode_scalar(-1000.0f, &code) == ET_KVQ_OK &&
        code == 0xFEu);
  CHECK(et_kvq_fp8_e4m3_encode_scalar(bits_f32(0x3A800000u), &code) ==
            ET_KVQ_OK &&
        code == 0x00u); /* 2^-10 ties to even 0 */
  CHECK(et_kvq_fp8_e4m3_encode_scalar(bits_f32(0x3B400000u), &code) ==
            ET_KVQ_OK &&
        code == 0x02u); /* 3*2^-10 ties to even 2 */
  CHECK(et_kvq_fp8_e4m3_encode_scalar(bits_f32(0x7F800000u), &code) ==
        ET_KVQ_ERR_NONFINITE);
  CHECK(et_kvq_fp8_e4m3_encode_scalar(bits_f32(0x7FC00000u), &code) ==
        ET_KVQ_ERR_NONFINITE);

  /* Symmetric INT8 RNE ties & zero-vector & subnormal vector. */
  const float ties[8] = {127.0f, -127.0f, 0.5f, 1.5f,
                         2.5f,   -0.5f,   -1.5f, -2.5f};
  int8_t qi[8];
  float sc = -1.0f;
  CHECK(et_kvq_int8_sym_quantize_vec(ties, 8u, qi, &sc) == ET_KVQ_OK);
  CHECK(sc == 1.0f);
  CHECK(qi[0] == 127 && qi[1] == -127 && qi[2] == 0 && qi[3] == 2 &&
        qi[4] == 2 && qi[5] == 0 && qi[6] == -2 && qi[7] == -2);

  const float zeros[4] = {0.0f, -0.0f, 0.0f, -0.0f};
  int8_t qz[4] = {9, 9, 9, 9};
  float zsc = 5.0f;
  CHECK(et_kvq_int8_sym_quantize_vec(zeros, 4u, qz, &zsc) == ET_KVQ_OK);
  CHECK(zsc == 0.0f && qz[0] == 0 && qz[1] == 0 && qz[2] == 0 && qz[3] == 0);

  uint8_t qfz[4] = {9, 9, 9, 9};
  zsc = 5.0f;
  CHECK(et_kvq_fp8_e4m3_quantize_vec(zeros, 4u, qfz, &zsc) == ET_KVQ_OK);
  CHECK(zsc == 0.0f && qfz[0] == 0 && qfz[1] == 0 && qfz[2] == 0 &&
        qfz[3] == 0);

  /* Reject -128 in INT8 dequantize and non-zero q with zero scale. */
  int8_t bad_q[4] = {0, -128, 0, 0};
  float out_buf[4] = {7.0f, 7.0f, 7.0f, 7.0f};
  CHECK(et_kvq_int8_sym_dequantize_vec(bad_q, 4u, 1.0f, out_buf) ==
        ET_KVQ_ERR_CORRUPT);
  CHECK(out_buf[0] == 7.0f && out_buf[1] == 7.0f);
  int8_t bad_z[4] = {0, 1, 0, 0};
  CHECK(et_kvq_int8_sym_dequantize_vec(bad_z, 4u, 0.0f, out_buf) ==
        ET_KVQ_ERR_CORRUPT);
  CHECK(out_buf[0] == 7.0f);

  forbid_alloc = 0;
}

/* ---- Test 2: Compression stats, external fixed pool, hot zero-alloc,
 *              bitwise parity, outlier dominance, top-1 & sliding window ---- */
static void test_kvq_core_and_decode(void) {
  const uint32_t nl = 2u, hkv = 2u, hq = 8u, hd = 64u, max_seq = 32u;
  const float scale = 1.0f / sqrtf((float)hd);

  const et_kvq_config cfg_fp32 = {ET_KVQ_FMT_FP32, nl, hkv, hd, max_seq, 0u};
  const et_kvq_config cfg_int8 = {ET_KVQ_FMT_INT8_SYM, nl, hkv, hd, max_seq,
                                  0u};
  const et_kvq_config cfg_fp8 = {ET_KVQ_FMT_FP8_E4M3FN, nl, hkv, hd, max_seq,
                                 0u};

  /* Verify caller-supplied fixed backing pool for FP8 cache! */
  size_t fp8_req = 0u;
  CHECK(et_kvq_storage_bytes_required(&cfg_fp8, &fp8_req) == ET_KVQ_OK);
  CHECK(fp8_req == (size_t)nl * 2u * max_seq * hkv * (hd + 4u));
  CHECK(fp8_req == 17408u);

  uint8_t *ext_pool = (uint8_t *)__real_calloc(1u, fp8_req + 16u);
  CHECK(ext_pool != NULL);
  /* Misaligned external pool must be rejected. */
  et_kvq_cache *bad_c = NULL;
  CHECK(et_kvq_cache_create(&cfg_fp8, ext_pool + 1u, fp8_req, &bad_c) ==
        ET_KVQ_ERR_ALIGNMENT);
  /* Undersized external pool must be rejected. */
  CHECK(et_kvq_cache_create(&cfg_fp8, ext_pool, fp8_req - 1u, &bad_c) ==
        ET_KVQ_ERR_CAPACITY);

  et_kvq_cache *c_fp32 = NULL;
  et_kvq_cache *c_int8 = NULL;
  et_kvq_cache *c_fp8 = NULL;
  et_kvq_cache *c_deq_int8 = NULL;
  et_kvq_cache *c_deq_fp8 = NULL;

  CHECK(et_kvq_cache_create(&cfg_fp32, NULL, 0u, &c_fp32) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_int8, NULL, 0u, &c_int8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_fp8, ext_pool, fp8_req, &c_fp8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_fp32, NULL, 0u, &c_deq_int8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_fp32, NULL, 0u, &c_deq_fp8) == ET_KVQ_OK);

  /* Enter strict ZERO-ALLOC hot-path audit! */
  forbid_alloc = 1;

  float k_buf[2 * 64];
  float v_buf[2 * 64];
  float dk_buf[2 * 64];
  float dv_buf[2 * 64];
  float q_buf[8 * 64];
  float out_fp32[8 * 64];
  float out_int8[8 * 64];
  float out_fp8[8 * 64];
  float out_deq_int8[8 * 64];
  float out_deq_fp8[8 * 64];

  for (uint32_t pos = 0u; pos < max_seq; ++pos) {
    for (uint32_t l = 0u; l < nl; ++l) {
      fill_kv_slice(k_buf, v_buf, hkv, hd, l, pos, 1);
      CHECK(et_kvq_append_token_v1(c_fp32, l, k_buf, v_buf) == ET_KVQ_OK);
      CHECK(et_kvq_append_token_v1(c_int8, l, k_buf, v_buf) == ET_KVQ_OK);
      CHECK(et_kvq_append_token_v1(c_fp8, l, k_buf, v_buf) == ET_KVQ_OK);

      CHECK(et_kvq_dequant_slice_v1(c_int8, l, pos, dk_buf, dv_buf) ==
            ET_KVQ_OK);
      CHECK(et_kvq_append_token_v1(c_deq_int8, l, dk_buf, dv_buf) == ET_KVQ_OK);

      CHECK(et_kvq_dequant_slice_v1(c_fp8, l, pos, dk_buf, dv_buf) ==
            ET_KVQ_OK);
      CHECK(et_kvq_append_token_v1(c_deq_fp8, l, dk_buf, dv_buf) == ET_KVQ_OK);
    }

    /* Multi-step decode top-1 argmax preservation check at t = 8, 16, 24, 32. */
    if (((pos + 1u) % 8u) == 0u) {
      fill_q(q_buf, hq, hd, 901u + pos * 11u);
      CHECK(et_kvq_decode_gqa_v1(c_fp32, 0u, hq, scale, 0u, q_buf, out_fp32) ==
            ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_int8, 0u, hq, scale, 0u, q_buf, out_int8) ==
            ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, out_fp8) ==
            ET_KVQ_OK);

      const int top_fp32 = compute_top1_logit(out_fp32, hq * hd, 16u);
      const int top_int8 = compute_top1_logit(out_int8, hq * hd, 16u);
      const int top_fp8 = compute_top1_logit(out_fp8, hq * hd, 16u);
      CHECK(top_int8 == top_fp32);
      CHECK(top_fp8 == top_fp32);
    }
  }

  /* Verify exact 3.7647x compression stats (65536 B -> 17408 B). */
  et_kvq_memory_stats st_fp32, st_int8, st_fp8;
  CHECK(et_kvq_cache_memory_stats(c_fp32, &st_fp32) == ET_KVQ_OK);
  CHECK(et_kvq_cache_memory_stats(c_int8, &st_int8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_memory_stats(c_fp8, &st_fp8) == ET_KVQ_OK);
  CHECK(st_fp32.fp32_baseline_bytes == 65536u &&
        st_fp32.quantized_bytes == 65536u && st_fp32.compression_ratio == 1.0);
  CHECK(st_int8.fp32_baseline_bytes == 65536u &&
        st_int8.quantized_bytes == 17408u);
  CHECK(st_fp8.fp32_baseline_bytes == 65536u &&
        st_fp8.quantized_bytes == 17408u);
  CHECK(fabs(st_int8.compression_ratio - (64.0 / 17.0)) < 1e-12);
  CHECK(fabs(st_fp8.compression_ratio - (64.0 / 17.0)) < 1e-12);
  CHECK(st_fp8.min_seq_len == 32u && st_fp8.max_seq_len_used == 32u);

  /* Bitwise 0.0 parity: fused quantized decode == explicit dequant decode
   * across both layers and both full-causal & sliding-window (window_size=8). */
  fill_q(q_buf, hq, hd, 4242u);
  for (uint32_t l = 0u; l < nl; ++l) {
    const uint32_t wins[2] = {0u, 8u};
    for (size_t wi = 0u; wi < 2u; ++wi) {
      CHECK(et_kvq_decode_gqa_v1(c_fp32, l, hq, scale, wins[wi], q_buf,
                                 out_fp32) == ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_int8, l, hq, scale, wins[wi], q_buf,
                                 out_int8) == ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_deq_int8, l, hq, scale, wins[wi], q_buf,
                                 out_deq_int8) == ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_fp8, l, hq, scale, wins[wi], q_buf,
                                 out_fp8) == ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(c_deq_fp8, l, hq, scale, wins[wi], q_buf,
                                 out_deq_fp8) == ET_KVQ_OK);

      CHECK(memcmp(out_int8, out_deq_int8, sizeof(out_int8)) == 0);
      CHECK(memcmp(out_fp8, out_deq_fp8, sizeof(out_fp8)) == 0);

      if (l == 0u && wins[wi] == 0u) {
        /* Measure RMSE on normal heads (qh 0..3 -> kv_h 0) and outlier heads
         * (qh 4..7 -> kv_h 1, coordinates d=1..63). */
        double sse_i8_norm = 0.0, sse_f8_norm = 0.0;
        double sse_i8_out = 0.0, sse_f8_out = 0.0;
        for (uint32_t qh = 0u; qh < 4u; ++qh) {
          for (uint32_t d = 0u; d < hd; ++d) {
            const size_t i = (size_t)qh * hd + d;
            const double ei = (double)out_int8[i] - (double)out_fp32[i];
            const double ef = (double)out_fp8[i] - (double)out_fp32[i];
            sse_i8_norm += ei * ei;
            sse_f8_norm += ef * ef;
          }
        }
        for (uint32_t qh = 4u; qh < 8u; ++qh) {
          for (uint32_t d = 1u; d < hd; ++d) {
            const size_t i = (size_t)qh * hd + d;
            const double ei = (double)out_int8[i] - (double)out_fp32[i];
            const double ef = (double)out_fp8[i] - (double)out_fp32[i];
            sse_i8_out += ei * ei;
            sse_f8_out += ef * ef;
          }
        }
        const double rmse_i8_norm = sqrt(sse_i8_norm / (4.0 * (double)hd));
        const double rmse_f8_norm = sqrt(sse_f8_norm / (4.0 * (double)hd));
        const double rmse_i8_out = sqrt(sse_i8_out / (4.0 * (double)(hd - 1u)));
        const double rmse_f8_out = sqrt(sse_f8_out / (4.0 * (double)(hd - 1u)));
        CHECK(rmse_i8_norm < 1e-2);
        CHECK(rmse_f8_norm < 1e-2);
        /* FP8 E4M3FN preserves small channels next to a 24.0 outlier far better
         * than uniform linear INT8! */
        CHECK(rmse_f8_out < rmse_i8_out);
      }
    }
  }

  /* Verify sliding-window output actually differs from full causal output. */
  float out_full[8 * 64], out_win[8 * 64];
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, out_full) ==
        ET_KVQ_OK);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 8u, q_buf, out_win) ==
        ET_KVQ_OK);
  CHECK(memcmp(out_full, out_win, sizeof(out_full)) != 0);

  /* Truncate (32 -> 20) + re-append (20..31) -> bitwise identical decode! */
  CHECK(et_kvq_cache_truncate(c_fp8, 33u) == ET_KVQ_ERR_OUT_OF_RANGE);
  CHECK(et_kvq_cache_truncate(c_fp8, 20u) == ET_KVQ_OK);
  uint32_t sl_check = 0u;
  CHECK(et_kvq_cache_layer_seq_len(c_fp8, 0u, &sl_check) == ET_KVQ_OK &&
        sl_check == 20u);
  CHECK(et_kvq_dequant_slice_v1(c_fp8, 0u, 20u, dk_buf, dv_buf) ==
        ET_KVQ_ERR_OUT_OF_RANGE);
  for (uint32_t pos = 20u; pos < max_seq; ++pos) {
    for (uint32_t l = 0u; l < nl; ++l) {
      fill_kv_slice(k_buf, v_buf, hkv, hd, l, pos, 1);
      CHECK(et_kvq_append_token_v1(c_fp8, l, k_buf, v_buf) == ET_KVQ_OK);
    }
  }
  float out_reapp[8 * 64];
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, out_reapp) ==
        ET_KVQ_OK);
  CHECK(memcmp(out_full, out_reapp, sizeof(out_full)) == 0);

  /* Capacity overflow must fail atomically without mutating seq_len. */
  CHECK(et_kvq_append_token_v1(c_fp8, 0u, k_buf, v_buf) == ET_KVQ_ERR_CAPACITY);
  CHECK(et_kvq_cache_layer_seq_len(c_fp8, 0u, &sl_check) == ET_KVQ_OK &&
        sl_check == 32u);

  /* Reset + non-finite input atomicity checks (still under forbid_alloc=1!). */
  CHECK(et_kvq_cache_reset(c_int8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_layer_seq_len(c_int8, 0u, &sl_check) == ET_KVQ_OK &&
        sl_check == 0u);
  k_buf[3] = bits_f32(0x7FC00000u); /* qNaN */
  CHECK(et_kvq_append_token_v1(c_int8, 0u, k_buf, v_buf) ==
        ET_KVQ_ERR_NONFINITE);
  CHECK(et_kvq_cache_layer_seq_len(c_int8, 0u, &sl_check) == ET_KVQ_OK &&
        sl_check == 0u);

  /* Non-finite Q must leave out untouched. */
  float sentinel[8 * 64];
  for (size_t i = 0u; i < ARRAY_COUNT(sentinel); ++i)
    sentinel[i] = -999.25f;
  q_buf[5] = bits_f32(0x7F800000u); /* +Inf */
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_NONFINITE);
  CHECK(sentinel[0] == -999.25f && sentinel[511] == -999.25f);

  /* Invalid num_q_heads not divisible by num_kv_heads. */
  fill_q(q_buf, hq, hd, 4242u);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, 3u, scale, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_SHAPE);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, 0.0f, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_INVALID_ARG);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, -1.0f, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_INVALID_ARG);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, bits_f32(0x7F800000u), 0u, q_buf,
                             sentinel) == ET_KVQ_ERR_NONFINITE);

  /* Aliased q == out rejected. */
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, q_buf) ==
        ET_KVQ_ERR_ALIAS);

  /* FP-env trap (FE_TOWARDZERO and x86_64 MXCSR FTZ/DAZ). */
  const int saved_rnd = fegetround();
  CHECK(fesetround(FE_TOWARDZERO) == 0);
  CHECK(et_kvq_append_token_v1(c_int8, 0u, v_buf, v_buf) == ET_KVQ_ERR_FP_ENV);
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_FP_ENV);
  CHECK(fesetround(saved_rnd) == 0);

#if defined(__x86_64__)
  const unsigned int saved_mxcsr = _mm_getcsr();
  _mm_setcsr(saved_mxcsr | 0x8000u); /* FTZ */
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_FP_ENV);
  _mm_setcsr(saved_mxcsr | 0x0040u); /* DAZ */
  CHECK(et_kvq_decode_gqa_v1(c_fp8, 0u, hq, scale, 0u, q_buf, sentinel) ==
        ET_KVQ_ERR_FP_ENV);
  _mm_setcsr(saved_mxcsr);
#endif

  forbid_alloc = 0;

  et_kvq_cache_destroy(c_fp32);
  et_kvq_cache_destroy(c_int8);
  et_kvq_cache_destroy(c_fp8);
  et_kvq_cache_destroy(c_deq_int8);
  et_kvq_cache_destroy(c_deq_fp8);
  __real_free(ext_pool);
}

/* ---- Test 3: ETKQ1000 SHA-256 Snapshot Save/Load & Security Negatives ---- */
static void test_snapshots(void) {
  char tmpl[] = "/tmp/etkq_test_XXXXXX";
  char *tdir = mkdtemp(tmpl);
  CHECK(tdir != NULL);

  const uint32_t nl = 2u, hkv = 2u, hq = 8u, hd = 32u, seq_len = 12u;
  const float scale = 1.0f / sqrtf((float)hd);
  const et_kvq_format fmts[3] = {ET_KVQ_FMT_FP32, ET_KVQ_FMT_INT8_SYM,
                                 ET_KVQ_FMT_FP8_E4M3FN};

  float k_buf[2 * 32], v_buf[2 * 32], q_buf[8 * 32];
  fill_q(q_buf, hq, hd, 515u);

  char path_i8[512], path_f8[512], path_sym[512], path_trunc[512];
  snprintf(path_i8, sizeof(path_i8), "%s/snap_i8.bin", tdir);
  snprintf(path_f8, sizeof(path_f8), "%s/snap_f8.bin", tdir);
  snprintf(path_sym, sizeof(path_sym), "%s/snap_sym.bin", tdir);
  snprintf(path_trunc, sizeof(path_trunc), "%s/snap_trunc.bin", tdir);

  for (size_t fi = 0u; fi < 3u; ++fi) {
    const et_kvq_config cfg = {fmts[fi], nl, hkv, hd, 16u, 0u};
    et_kvq_cache *src = NULL;
    et_kvq_cache *dst = NULL;
    CHECK(et_kvq_cache_create(&cfg, NULL, 0u, &src) == ET_KVQ_OK);
    CHECK(et_kvq_cache_create(&cfg, NULL, 0u, &dst) == ET_KVQ_OK);

    for (uint32_t pos = 0u; pos < seq_len; ++pos) {
      for (uint32_t l = 0u; l < nl; ++l) {
        fill_kv_slice(k_buf, v_buf, hkv, hd, l, pos, 1);
        CHECK(et_kvq_append_token_v1(src, l, k_buf, v_buf) == ET_KVQ_OK);
      }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/snap_fmt_%u.bin", tdir,
             (unsigned)fmts[fi]);
    CHECK(et_kvq_cache_save_snapshot(src, path) == ET_KVQ_OK);

    /* Check strict 0600 permissions. */
    struct stat st;
    CHECK(stat(path, &st) == 0);
    CHECK((st.st_mode & 0777) == 0600);

    CHECK(et_kvq_cache_load_snapshot(dst, path) == ET_KVQ_OK);

    float out_src[8 * 32], out_dst[8 * 32];
    for (uint32_t l = 0u; l < nl; ++l) {
      CHECK(et_kvq_decode_gqa_v1(src, l, hq, scale, 0u, q_buf, out_src) ==
            ET_KVQ_OK);
      CHECK(et_kvq_decode_gqa_v1(dst, l, hq, scale, 0u, q_buf, out_dst) ==
            ET_KVQ_OK);
      CHECK(memcmp(out_src, out_dst, sizeof(out_src)) == 0);
    }

    if (fmts[fi] == ET_KVQ_FMT_INT8_SYM)
      CHECK(et_kvq_cache_save_snapshot(src, path_i8) == ET_KVQ_OK);
    if (fmts[fi] == ET_KVQ_FMT_FP8_E4M3FN)
      CHECK(et_kvq_cache_save_snapshot(src, path_f8) == ET_KVQ_OK);

    (void)unlink(path);
    et_kvq_cache_destroy(src);
    et_kvq_cache_destroy(dst);
  }

  /* Format mismatch, shape mismatch, capacity mismatch, symlink, tamper,
   * truncation, trailing-byte rejection (all preserving target cache state!). */
  const et_kvq_config cfg_f8 = {ET_KVQ_FMT_FP8_E4M3FN, nl, hkv, hd, 16u, 0u};
  const et_kvq_config cfg_small = {ET_KVQ_FMT_INT8_SYM, nl, hkv, hd, 4u, 0u};
  const et_kvq_config cfg_hd16 = {ET_KVQ_FMT_INT8_SYM, nl, hkv, 16u, 16u, 0u};
  const et_kvq_config cfg_i8 = {ET_KVQ_FMT_INT8_SYM, nl, hkv, hd, 16u, 0u};

  et_kvq_cache *c_f8 = NULL, *c_small = NULL, *c_hd16 = NULL, *c_i8 = NULL;
  CHECK(et_kvq_cache_create(&cfg_f8, NULL, 0u, &c_f8) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_small, NULL, 0u, &c_small) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_hd16, NULL, 0u, &c_hd16) == ET_KVQ_OK);
  CHECK(et_kvq_cache_create(&cfg_i8, NULL, 0u, &c_i8) == ET_KVQ_OK);

  /* Put 1 token into c_f8 to verify failed load leaves c_f8 untouched. */
  fill_kv_slice(k_buf, v_buf, hkv, hd, 0u, 0u, 0);
  CHECK(et_kvq_append_token_v1(c_f8, 0u, k_buf, v_buf) == ET_KVQ_OK);

  /* Format mismatch: INT8 snapshot -> FP8 cache. */
  CHECK(et_kvq_cache_load_snapshot(c_f8, path_i8) == ET_KVQ_ERR_FORMAT_MISMATCH);
  uint32_t sl0 = 99u;
  CHECK(et_kvq_cache_layer_seq_len(c_f8, 0u, &sl0) == ET_KVQ_OK && sl0 == 1u);

  /* Shape mismatch & capacity mismatch. */
  CHECK(et_kvq_cache_load_snapshot(c_hd16, path_i8) == ET_KVQ_ERR_SHAPE);
  CHECK(et_kvq_cache_load_snapshot(c_small, path_i8) == ET_KVQ_ERR_CAPACITY);

  /* Symlink rejection on BOTH save and load. */
  CHECK(symlink(path_i8, path_sym) == 0);
  CHECK(et_kvq_cache_load_snapshot(c_i8, path_sym) == ET_KVQ_ERR_IO);
  CHECK(et_kvq_cache_save_snapshot(c_f8, path_sym) == ET_KVQ_ERR_IO);

  /* Truncation rejection (copy all but last byte). */
  struct stat st_i8;
  CHECK(stat(path_i8, &st_i8) == 0);
  uint8_t *raw = (uint8_t *)__real_malloc((size_t)st_i8.st_size + 1u);
  CHECK(raw != NULL);
  int fd = open(path_i8, O_RDONLY);
  CHECK(fd >= 0);
  CHECK(read(fd, raw, (size_t)st_i8.st_size) == st_i8.st_size);
  close(fd);

  fd = open(path_trunc, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  CHECK(fd >= 0);
  CHECK(write(fd, raw, (size_t)st_i8.st_size - 1u) == st_i8.st_size - 1);
  close(fd);
  CHECK(et_kvq_cache_load_snapshot(c_i8, path_trunc) == ET_KVQ_ERR_CORRUPT);

  /* Trailing extra byte rejection. */
  raw[st_i8.st_size] = 0xAAu;
  fd = open(path_trunc, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  CHECK(fd >= 0);
  CHECK(write(fd, raw, (size_t)st_i8.st_size + 1u) == st_i8.st_size + 1);
  close(fd);
  CHECK(et_kvq_cache_load_snapshot(c_i8, path_trunc) == ET_KVQ_ERR_CORRUPT);

  /* Payload bit-flip tamper rejection -> ET_KVQ_ERR_CHECKSUM. */
  raw[80] ^= 0x01u;
  fd = open(path_trunc, O_WRONLY | O_CREAT | O_TRUNC, 0600);
  CHECK(fd >= 0);
  CHECK(write(fd, raw, (size_t)st_i8.st_size) == st_i8.st_size);
  close(fd);
  CHECK(et_kvq_cache_load_snapshot(c_i8, path_trunc) == ET_KVQ_ERR_CHECKSUM);

  __real_free(raw);
  (void)unlink(path_i8);
  (void)unlink(path_f8);
  (void)unlink(path_sym);
  (void)unlink(path_trunc);
  (void)rmdir(tdir);

  et_kvq_cache_destroy(c_f8);
  et_kvq_cache_destroy(c_small);
  et_kvq_cache_destroy(c_hd16);
  et_kvq_cache_destroy(c_i8);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_dump();
    return 0;
  }

  test_codecs_exhaustive();
  test_kvq_core_and_decode();
  test_snapshots();
  puts("KVQ native C11 tests (codecs + zero-alloc hot path + parity + "
       "snapshots + negatives): PASS");
  return 0;
}
