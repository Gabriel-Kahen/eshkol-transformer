#define _GNU_SOURCE

#include "eshkol_transformer/data4_pipeline_abi.h"
#include "eshkol_transformer/dpo_alignment_abi.h"
#include "eshkol_transformer/io5_quant_abi.h"
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/lora_adapter_abi.h"
#include "eshkol_transformer/mod5_adamw_abi.h"
#include "eshkol_transformer/mod5_mla_abi.h"
#include "eshkol_transformer/mod5_moe_abi.h"
#include "eshkol_transformer/mod5_rope_abi.h"
#include "eshkol_transformer/o2_muon_abi.h"
#include "eshkol_transformer/spec_decode_abi.h"
#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FP_CONTRACT OFF

_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && sizeof(float) == 4 &&
                   DBL_MANT_DIG == 53 && sizeof(double) == 8,
               "b0_wave5_bench requires IEEE binary32 and binary64");

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define REQUIRE(cond, msg)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "b0_wave5_bench FAIL (%s:%d): %s\n", __FILE__, __LINE__, \
              (msg));                                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static volatile uint32_t g_dce_sink = 0u;

/* ========================================================================= */
/* Timing, deterministic seeding, and SHA-256 helpers                        */
/* ========================================================================= */

static uint64_t now_ns(void) {
  struct timespec ts;
  REQUIRE(clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0,
          "clock_gettime(CLOCK_MONOTONIC_RAW)");
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
  const uint64_t x = *(const uint64_t *)a;
  const uint64_t y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

static float det_f32(size_t idx, uint32_t seed) {
  uint32_t x = ((uint32_t)idx + 1u) * UINT32_C(0x9e3779b1) + seed;
  x = (x ^ (x >> 16u)) * UINT32_C(0x85ebca6b);
  x = (x ^ (x >> 13u)) * UINT32_C(0xc2b2ae35);
  x ^= x >> 16u;
  return (float)((int32_t)(x % 257u) - 128) / 256.0f;
}

static uint32_t load_u32_head(const void *buf) {
  uint32_t v = 0u;
  memcpy(&v, buf, sizeof(v));
  return v;
}

static void check_finite_nonzero_f32(const float *buf, size_t n,
                                     const char *label) {
  int nonzero = 0;
  for (size_t i = 0; i < n; ++i) {
    REQUIRE(isfinite(buf[i]), label);
    if (buf[i] != 0.0f)
      nonzero = 1;
  }
  REQUIRE(nonzero, label);
}

typedef struct {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} sha256_ctx;

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_block(sha256_ctx *ctx, const uint8_t block[64]) {
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
  for (size_t i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = ctx->h[0], b = ctx->h[1], c = ctx->h[2], d = ctx->h[3];
  uint32_t e = ctx->h[4], f = ctx->h[5], g = ctx->h[6], h = ctx->h[7];
  for (size_t i = 0; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  ctx->h[0] += a;
  ctx->h[1] += b;
  ctx->h[2] += c;
  ctx->h[3] += d;
  ctx->h[4] += e;
  ctx->h[5] += f;
  ctx->h[6] += g;
  ctx->h[7] += h;
}

static void sha256_init(sha256_ctx *ctx) {
  static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                 0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                 0x1f83d9abu, 0x5be0cd19u};
  memcpy(ctx->h, iv, sizeof(iv));
  ctx->length = 0u;
  ctx->used = 0u;
}

static void sha256_update(sha256_ctx *ctx, const void *data, size_t len) {
  const uint8_t *p = (const uint8_t *)data;
  ctx->length += (uint64_t)len;
  while (len != 0u) {
    size_t take = 64u - ctx->used;
    if (take > len)
      take = len;
    memcpy(ctx->block + ctx->used, p, take);
    ctx->used += take;
    p += take;
    len -= take;
    if (ctx->used == 64u) {
      sha256_block(ctx, ctx->block);
      ctx->used = 0u;
    }
  }
}

static void sha256_final_hex(sha256_ctx *ctx, char hex[65]) {
  static const char digits[] = "0123456789abcdef";
  const uint64_t bits = ctx->length * UINT64_C(8);
  ctx->block[ctx->used++] = 0x80u;
  if (ctx->used > 56u) {
    memset(ctx->block + ctx->used, 0, 64u - ctx->used);
    sha256_block(ctx, ctx->block);
    ctx->used = 0u;
  }
  memset(ctx->block + ctx->used, 0, 56u - ctx->used);
  for (size_t i = 0; i < 8u; ++i)
    ctx->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(ctx, ctx->block);
  for (size_t i = 0; i < 8u; ++i) {
    for (size_t j = 0; j < 4u; ++j) {
      const uint8_t byte = (uint8_t)(ctx->h[i] >> (24u - 8u * j));
      hex[(i * 4u + j) * 2u] = digits[byte >> 4u];
      hex[(i * 4u + j) * 2u + 1u] = digits[byte & 0x0fu];
    }
  }
  hex[64] = '\0';
}

static void sha256_hex_buf(const void *data, size_t len, char hex[65]) {
  sha256_ctx ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, data, len);
  sha256_final_hex(&ctx, hex);
}

static void sha256_file_hex(const char *path, char hex[65]) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "b0_wave5_bench FAIL: cannot open archive '%s': %s\n", path,
            strerror(errno));
    exit(EXIT_FAILURE);
  }
  sha256_ctx ctx;
  sha256_init(&ctx);
  uint8_t buf[16384];
  size_t total = 0u;
  for (;;) {
    const size_t n = fread(buf, 1u, sizeof(buf), fp);
    if (n > 0u) {
      sha256_update(&ctx, buf, n);
      total += n;
    }
    if (n < sizeof(buf)) {
      REQUIRE(!ferror(fp), "fread archive for sha256");
      break;
    }
  }
  fclose(fp);
  REQUIRE(total > 0u, "empty archive file");
  sha256_final_hex(&ctx, hex);
}

static uint64_t checksum_from_sha256_hex(const char hex[65]) {
  uint64_t v = 0u;
  for (size_t i = 0; i < 16u; ++i) {
    const char c = hex[i];
    uint64_t nib = 0u;
    if (c >= '0' && c <= '9')
      nib = (uint64_t)(c - '0');
    else if (c >= 'a' && c <= 'f')
      nib = (uint64_t)(10 + (c - 'a'));
    else
      REQUIRE(0, "invalid sha256 hex char");
    v = (v << 4u) | nib;
  }
  v &= UINT64_C(0x3fffffffffffffff);
  return v != 0u ? v : UINT64_C(1);
}

/* ========================================================================= */
/* K1 helpers (for MOD5 RoPE, MOD5 MoE, MOD5 MLA)                            */
/* ========================================================================= */

typedef struct {
  const et_kernel_provider_v1 *provider;
} provider_ctx;

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  const provider_ctx *ctx = (const provider_ctx *)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0 ? ctx->provider
                                                           : NULL;
}

static et_kernel_runtime *discover_provider(const et_kernel_provider_v1 *prov) {
  provider_ctx ctx = {prov};
  et_kernel_runtime *rt = NULL;
  et_kernel_error err;
  const int32_t st =
      et_kernel_runtime_discover(resolve_provider, &ctx, &rt, &err);
  if (st != 0)
    fprintf(stderr, "K1 discover error %s: %s\n", err.operation, err.message);
  REQUIRE(st == 0 && rt != NULL, "K1 provider discovery");
  return rt;
}

static void k1_dispatch(const et_kernel_runtime *rt,
                        const et_kernel_call_v1 *call) {
  et_kernel_error err;
  const int32_t st = et_kernel_runtime_dispatch(rt, call, &err);
  if (st != 0)
    fprintf(stderr, "K1 dispatch error %s (%u/%u): %s\n", err.operation,
            err.category, err.code, err.message);
  REQUIRE(st == 0 && err.code == ET_KERNEL_CODE_OK, "K1 dispatch");
}

static et_kernel_tensor_view_v1 make_view(void *data, size_t bytes,
                                          const char *dtype, size_t rank,
                                          const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){.struct_size =
                                        sizeof(et_kernel_tensor_view_v1),
                                    .data = data,
                                    .byte_length = bytes,
                                    .dtype = dtype,
                                    .device = "cpu",
                                    .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                    .offset_bytes = 0u,
                                    .rank = rank,
                                    .shape = shape};
}

static et_kernel_call_v1 make_call(const char *cap,
                                   const et_kernel_request_v1 *req, size_t ni,
                                   const void *in, size_t no, void *out) {
  const size_t stride = sizeof(et_kernel_tensor_view_v1);
  return (et_kernel_call_v1){.struct_size = sizeof(et_kernel_call_v1),
                             .capability = cap,
                             .request = req,
                             .input_count = ni,
                             .input_stride = stride,
                             .input_bytes = ni * stride,
                             .inputs = in,
                             .output_count = no,
                             .output_stride = stride,
                             .output_bytes = no * stride,
                             .outputs = out};
}

/* ========================================================================= */
/* Unified benchmark result storage                                          */
/* ========================================================================= */

enum {
  NUM_GROUPS = 11,
  MAX_CASES_PER_GROUP = 8,
  MAX_SAMPLES = 32
};

typedef enum {
  GRP_DATA4_PIPELINE = 0,
  GRP_DPO = 1,
  GRP_IO5_QUANT = 2,
  GRP_LORA = 3,
  GRP_MOD5_ADAMW = 4,
  GRP_MOD5_MLA = 5,
  GRP_MOD5_MOE = 6,
  GRP_MOD5_ROPE = 7,
  GRP_O2_MUON = 8,
  GRP_SPEC_DECODE = 9,
  GRP_T2_TRAINER = 10
} group_id;

typedef struct {
  const char *key;
  const char *engine_label;
  const char *archive_name;
  const char *archive_relpath;
} group_descriptor;

/* Strictly sorted by key (ASCII lexicographical order). */
static const group_descriptor g_group_descs[NUM_GROUPS] = {
    {"data4_pipeline", "DATA4 Pipeline",
     "libeshkol_transformer_data4_pipeline.a",
     "data4_pipeline/libeshkol_transformer_data4_pipeline.a"},
    {"dpo", "DPO/cDPO", "libeshkol_transformer_dpo.a",
     "dpo/libeshkol_transformer_dpo.a"},
    {"io5_quant", "IO5 Quant Codec", "libeshkol_transformer_io5_quant.a",
     "io5_quant/libeshkol_transformer_io5_quant.a"},
    {"lora", "LoRA", "libeshkol_transformer_lora.a",
     "lora/libeshkol_transformer_lora.a"},
    {"mod5_adamw", "MOD5 AdamW", "libeshkol_transformer_mod5_adamw.a",
     "mod5_adamw/libeshkol_transformer_mod5_adamw.a"},
    {"mod5_mla", "MOD5 MLA", "libeshkol_transformer_mod5_mla.a",
     "mod5_mla/libeshkol_transformer_mod5_mla.a"},
    {"mod5_moe", "MOD5 MoE", "libeshkol_transformer_mod5_moe.a",
     "mod5_moe/libeshkol_transformer_mod5_moe.a"},
    {"mod5_rope", "MOD5 RoPE", "libeshkol_transformer_mod5_rope.a",
     "mod5_rope/libeshkol_transformer_mod5_rope.a"},
    {"o2_muon", "O2 Muon+WSD", "libeshkol_transformer_o2_muon.a",
     "o2_muon/libeshkol_transformer_o2_muon.a"},
    {"spec_decode", "Speculative Decoding",
     "libeshkol_transformer_spec_decode.a",
     "spec_decode/libeshkol_transformer_spec_decode.a"},
    {"t2_trainer", "T2 BPE Trainer", "libeshkol_transformer_t2_trainer.a",
     "t2_trainer/libeshkol_transformer_t2_trainer.a"}};

typedef struct {
  const char *name;
  const char *operation;
  const char *shape;
  const char *throughput_unit;
  uint64_t work_units_per_iter;
  uint32_t warmup;
  uint32_t samples;
  uint32_t iterations;
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t p95_ns;
  uint64_t max_ns;
  double mean_ns;
  double median_ns_per_iter;
  double throughput;
  uint64_t checksum;
  char output_sha256[65];
} bench_case_result;

typedef struct {
  int executed;
  size_t case_count;
  bench_case_result cases[MAX_CASES_PER_GROUP];
} bench_group_result;

static bench_group_result g_results[NUM_GROUPS];

static void record_case(group_id gid, const char *name, const char *operation,
                        const char *shape, const char *throughput_unit,
                        uint64_t work_units_per_iter, uint32_t warmup,
                        uint32_t samples, uint32_t iterations,
                        const uint64_t *raw_times, const char witness_hex[65]) {
  REQUIRE(samples >= 1u && samples <= MAX_SAMPLES, "invalid sample count");
  REQUIRE(iterations >= 1u, "invalid iteration count");
  REQUIRE(work_units_per_iter >= 1u, "invalid work_units_per_iter");

  bench_group_result *gr = &g_results[gid];
  REQUIRE(gr->case_count < MAX_CASES_PER_GROUP, "case capacity exceeded");
  gr->executed = 1;

  uint64_t sorted[MAX_SAMPLES];
  uint64_t sum = 0u;
  for (uint32_t i = 0; i < samples; ++i) {
    REQUIRE(raw_times[i] > 0u, "nonpositive measured duration");
    sorted[i] = raw_times[i];
    sum += raw_times[i];
  }
  qsort(sorted, samples, sizeof(sorted[0]), cmp_u64);

  const size_t med_idx = (size_t)((samples - 1u) / 2u);
  const size_t p95_idx = (size_t)((95u * (samples - 1u) + 99u) / 100u);
  REQUIRE(med_idx <= p95_idx && p95_idx < samples, "percentile index bounds");

  bench_case_result *r = &gr->cases[gr->case_count++];
  memset(r, 0, sizeof(*r));
  r->name = name;
  r->operation = operation;
  r->shape = shape;
  r->throughput_unit = throughput_unit;
  r->work_units_per_iter = work_units_per_iter;
  r->warmup = warmup;
  r->samples = samples;
  r->iterations = iterations;
  r->min_ns = sorted[0];
  r->median_ns = sorted[med_idx];
  r->p95_ns = sorted[p95_idx];
  r->max_ns = sorted[samples - 1u];
  r->mean_ns = (double)sum / (double)samples;
  r->median_ns_per_iter = (double)r->median_ns / (double)iterations;
  r->throughput = ((double)work_units_per_iter * (double)iterations * 1e9) /
                  (double)r->median_ns;
  REQUIRE(isfinite(r->mean_ns) && r->mean_ns > 0.0, "mean_ns finite positive");
  REQUIRE(isfinite(r->throughput) && r->throughput > 0.0,
          "throughput finite positive");
  memcpy(r->output_sha256, witness_hex, 65u);
  r->checksum = checksum_from_sha256_hex(witness_hex);
  REQUIRE(r->checksum != 0u, "non-zero checksum");
}

/* ========================================================================= */
/* 1. MOD5 RoPE                                                              */
/* ========================================================================= */

static void bench_mod5_rope(int smoke) {
  enum { N = 2, H = 4, T = 64, DH = 16, ELEMS = N * H * T * DH };
  static float x[ELEMS], dy[ELEMS], y[ELEMS], dx[ELEMS];
  static int64_t pos[N * T];
  uint64_t shape[4] = {N, H, T, DH};
  uint64_t pos_shape[2] = {N, T};
  float theta = 10000.0f;
  float scale = 1.0f;

  for (size_t i = 0; i < ELEMS; ++i) {
    x[i] = det_f32(i, 11u);
    dy[i] = det_f32(i, 23u);
  }
  for (size_t b = 0; b < N; ++b)
    for (size_t t = 0; t < T; ++t)
      pos[b * T + t] = (int64_t)(b * T + t);

  et_kernel_tensor_view_v1 fwd_in[4] = {
      make_view(x, sizeof(x), "f32", 4u, shape),
      make_view(pos, sizeof(pos), "i64", 2u, pos_shape),
      make_view(&theta, sizeof(theta), "f32", 0u, NULL),
      make_view(&scale, sizeof(scale), "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 fwd_out = make_view(y, sizeof(y), "f32", 4u, shape);
  et_kernel_request_v1 fwd_req = {sizeof(et_kernel_request_v1), "rope.forward",
                                  "f32", "cpu", 4u, shape, 1u, {0}};
  et_kernel_call_v1 fwd_call =
      make_call("kernel.rope", &fwd_req, 4u, fwd_in, 1u, &fwd_out);

  et_kernel_tensor_view_v1 bwd_in[4] = {
      make_view(dy, sizeof(dy), "f32", 4u, shape),
      make_view(pos, sizeof(pos), "i64", 2u, pos_shape),
      make_view(&theta, sizeof(theta), "f32", 0u, NULL),
      make_view(&scale, sizeof(scale), "f32", 0u, NULL)};
  et_kernel_tensor_view_v1 bwd_out =
      make_view(dx, sizeof(dx), "f32", 4u, shape);
  et_kernel_request_v1 bwd_req = {sizeof(et_kernel_request_v1), "rope.backward",
                                  "f32", "cpu", 4u, shape, 1u, {0}};
  et_kernel_call_v1 bwd_call =
      make_call("kernel.rope", &bwd_req, 4u, bwd_in, 1u, &bwd_out);

  et_kernel_runtime *rt = discover_provider(et_mod5_rope_kernel_provider_v1());
  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 2u : 20u;
  uint64_t times[MAX_SAMPLES];

  /* Forward */
  memset(y, 0, sizeof(y));
  k1_dispatch(rt, &fwd_call);
  check_finite_nonzero_f32(y, ELEMS, "mod5_rope forward");
  char witness[65];
  sha256_hex_buf(y, sizeof(y), witness);

  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &fwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(y, 0, sizeof(y));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      k1_dispatch(rt, &fwd_call);
      acc ^= load_u32_head(&y[i % ELEMS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(y, sizeof(y), check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_rope forward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_ROPE, "rope_forward_2x4x64x16", "rope.forward",
              "N=2,H=4,T=64,Dh=16", "elems_per_sec", ELEMS, warmup, samples,
              iters, times, witness);

  /* Backward */
  memset(dx, 0, sizeof(dx));
  k1_dispatch(rt, &bwd_call);
  check_finite_nonzero_f32(dx, ELEMS, "mod5_rope backward");
  sha256_hex_buf(dx, sizeof(dx), witness);

  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &bwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dx, 0, sizeof(dx));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      k1_dispatch(rt, &bwd_call);
      acc ^= load_u32_head(&dx[i % ELEMS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(dx, sizeof(dx), check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_rope backward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_ROPE, "rope_backward_2x4x64x16", "rope.backward",
              "N=2,H=4,T=64,Dh=16", "elems_per_sec", ELEMS, warmup, samples,
              iters, times, witness);

  et_kernel_runtime_destroy(rt);
}

/* ========================================================================= */
/* 2. MOD5 MoE                                                               */
/* ========================================================================= */

static void bench_mod5_moe(int smoke) {
  enum { N = 2, T = 16, D = 32, F = 64, E = 8, K = 2 };
  static float x[N * T * D], dy[N * T * D];
  static float w_gate[E * D], w_gate_exp[E * F * D], w_up_exp[E * F * D],
      w_down_exp[E * D * F];
  static float y[N * T * D], aux_loss;
  static int64_t expert_counts[E], active_evals;
  static float dx[N * T * D], dw_gate[E * D], dw_gate_exp[E * F * D],
      dw_up_exp[E * F * D], dw_down_exp[E * D * F];
  float alpha_aux = 0.01f;

  for (size_t i = 0; i < ARRAY_COUNT(x); ++i) {
    x[i] = det_f32(i, 101u);
    dy[i] = det_f32(i, 103u);
  }
  for (size_t i = 0; i < ARRAY_COUNT(w_gate); ++i)
    w_gate[i] = det_f32(i, 107u);
  for (size_t i = 0; i < ARRAY_COUNT(w_gate_exp); ++i) {
    w_gate_exp[i] = det_f32(i, 109u);
    w_up_exp[i] = det_f32(i, 113u);
    w_down_exp[i] = det_f32(i, 127u);
  }

  uint64_t req_shape[6] = {N, T, D, F, E, K};
  uint64_t x_shape[3] = {N, T, D};
  uint64_t wg_shape[2] = {E, D};
  uint64_t wgu_shape[3] = {E, F, D};
  uint64_t wd_shape[3] = {E, D, F};
  uint64_t ec_shape[1] = {E};

  et_kernel_tensor_view_v1 in_views[7] = {
      make_view(x, sizeof(x), "f32", 3u, x_shape),
      make_view(w_gate, sizeof(w_gate), "f32", 2u, wg_shape),
      make_view(w_gate_exp, sizeof(w_gate_exp), "f32", 3u, wgu_shape),
      make_view(w_up_exp, sizeof(w_up_exp), "f32", 3u, wgu_shape),
      make_view(w_down_exp, sizeof(w_down_exp), "f32", 3u, wd_shape),
      make_view(&alpha_aux, sizeof(alpha_aux), "f32", 0u, NULL),
      make_view(dy, sizeof(dy), "f32", 3u, x_shape)};
  et_kernel_tensor_view_v1 fwd_out[4] = {
      make_view(y, sizeof(y), "f32", 3u, x_shape),
      make_view(&aux_loss, sizeof(aux_loss), "f32", 0u, NULL),
      make_view(expert_counts, sizeof(expert_counts), "i64", 1u, ec_shape),
      make_view(&active_evals, sizeof(active_evals), "i64", 0u, NULL)};
  et_kernel_tensor_view_v1 bwd_out[5] = {
      make_view(dx, sizeof(dx), "f32", 3u, x_shape),
      make_view(dw_gate, sizeof(dw_gate), "f32", 2u, wg_shape),
      make_view(dw_gate_exp, sizeof(dw_gate_exp), "f32", 3u, wgu_shape),
      make_view(dw_up_exp, sizeof(dw_up_exp), "f32", 3u, wgu_shape),
      make_view(dw_down_exp, sizeof(dw_down_exp), "f32", 3u, wd_shape)};

  et_kernel_request_v1 fwd_req = {sizeof(et_kernel_request_v1), "moe.forward",
                                  "f32", "cpu", 6u, req_shape, 1u, {0}};
  et_kernel_request_v1 bwd_req = {sizeof(et_kernel_request_v1), "moe.backward",
                                  "f32", "cpu", 6u, req_shape, 1u, {0}};
  et_kernel_call_v1 fwd_call =
      make_call("kernel.moe", &fwd_req, 6u, in_views, 4u, fwd_out);
  et_kernel_call_v1 bwd_call =
      make_call("kernel.moe", &bwd_req, 7u, in_views, 5u, bwd_out);

  et_kernel_runtime *rt = discover_provider(et_mod5_moe_kernel_provider_v1());
  const uint32_t warmup = smoke ? 1u : 4u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 1u : 5u;
  uint64_t times[MAX_SAMPLES];

  /* Forward */
  memset(y, 0, sizeof(y));
  aux_loss = 0.0f;
  memset(expert_counts, 0, sizeof(expert_counts));
  active_evals = 0;
  k1_dispatch(rt, &fwd_call);
  check_finite_nonzero_f32(y, ARRAY_COUNT(y), "mod5_moe forward y");
  REQUIRE(isfinite(aux_loss) && aux_loss >= 0.0f, "mod5_moe aux_loss");
  REQUIRE(active_evals == (int64_t)(N * T * K), "mod5_moe sparse evals");
  char witness[65];
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, y, sizeof(y));
    sha256_update(&sc, &aux_loss, sizeof(aux_loss));
    sha256_update(&sc, expert_counts, sizeof(expert_counts));
    sha256_update(&sc, &active_evals, sizeof(active_evals));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &fwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(y, 0, sizeof(y));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      k1_dispatch(rt, &fwd_call);
      acc ^= load_u32_head(&y[i % ARRAY_COUNT(y)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, y, sizeof(y));
    sha256_update(&sc, &aux_loss, sizeof(aux_loss));
    sha256_update(&sc, expert_counts, sizeof(expert_counts));
    sha256_update(&sc, &active_evals, sizeof(active_evals));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_moe forward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_MOE, "moe_forward_2x16x32x64_e8k2", "moe.forward",
              "N=2,T=16,D=32,F=64,E=8,K=2", "tokens_per_sec", N * T, warmup,
              samples, iters, times, witness);

  /* Backward */
  memset(dx, 0, sizeof(dx));
  memset(dw_gate, 0, sizeof(dw_gate));
  memset(dw_gate_exp, 0, sizeof(dw_gate_exp));
  memset(dw_up_exp, 0, sizeof(dw_up_exp));
  memset(dw_down_exp, 0, sizeof(dw_down_exp));
  k1_dispatch(rt, &bwd_call);
  check_finite_nonzero_f32(dx, ARRAY_COUNT(dx), "mod5_moe backward dx");
  check_finite_nonzero_f32(dw_gate, ARRAY_COUNT(dw_gate),
                           "mod5_moe backward dw_gate");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, dw_gate, sizeof(dw_gate));
    sha256_update(&sc, dw_gate_exp, sizeof(dw_gate_exp));
    sha256_update(&sc, dw_up_exp, sizeof(dw_up_exp));
    sha256_update(&sc, dw_down_exp, sizeof(dw_down_exp));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &bwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dx, 0, sizeof(dx));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      k1_dispatch(rt, &bwd_call);
      acc ^= load_u32_head(&dx[i % ARRAY_COUNT(dx)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, dw_gate, sizeof(dw_gate));
    sha256_update(&sc, dw_gate_exp, sizeof(dw_gate_exp));
    sha256_update(&sc, dw_up_exp, sizeof(dw_up_exp));
    sha256_update(&sc, dw_down_exp, sizeof(dw_down_exp));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_moe backward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_MOE, "moe_backward_2x16x32x64_e8k2", "moe.backward",
              "N=2,T=16,D=32,F=64,E=8,K=2", "tokens_per_sec", N * T, warmup,
              samples, iters, times, witness);

  et_kernel_runtime_destroy(rt);
}

/* ========================================================================= */
/* 3. MOD5 MLA                                                               */
/* ========================================================================= */

static void bench_mod5_mla(int smoke) {
  enum { N = 2, T = 16, D = 32, H = 4, DH = 16, DC = 8, HD = H * DH };
  static float x[N * T * D], dy[N * T * D], out[N * T * D], dx[N * T * D];
  static float w_q[HD * D], w_dkv[DC * D], w_uk[HD * DC], w_uv[HD * DC],
      w_o[D * HD];
  static float dw_q[HD * D], dw_dkv[DC * D], dw_uk[HD * DC], dw_uv[HD * DC],
      dw_o[D * HD];

  for (size_t i = 0; i < ARRAY_COUNT(x); ++i) {
    x[i] = det_f32(i, 211u);
    dy[i] = det_f32(i, 223u);
  }
  for (size_t i = 0; i < ARRAY_COUNT(w_q); ++i)
    w_q[i] = det_f32(i, 227u);
  for (size_t i = 0; i < ARRAY_COUNT(w_dkv); ++i)
    w_dkv[i] = det_f32(i, 229u);
  for (size_t i = 0; i < ARRAY_COUNT(w_uk); ++i) {
    w_uk[i] = det_f32(i, 233u);
    w_uv[i] = det_f32(i, 239u);
  }
  for (size_t i = 0; i < ARRAY_COUNT(w_o); ++i)
    w_o[i] = det_f32(i, 241u);

  const double comp = et_mod5_mla_kv_compression_ratio(H, DH, DC);
  REQUIRE(fabs(comp - 16.0) < 1e-12, "mod5_mla 16:1 KV compression ratio");

  uint64_t req_shape[6] = {N, T, D, H, DH, DC};
  uint64_t x_shape[3] = {N, T, D};
  uint64_t wq_shape[2] = {HD, D};
  uint64_t wdkv_shape[2] = {DC, D};
  uint64_t wukv_shape[2] = {HD, DC};
  uint64_t wo_shape[2] = {D, HD};

  et_kernel_tensor_view_v1 in_views[7] = {
      make_view(x, sizeof(x), "f32", 3u, x_shape),
      make_view(w_q, sizeof(w_q), "f32", 2u, wq_shape),
      make_view(w_dkv, sizeof(w_dkv), "f32", 2u, wdkv_shape),
      make_view(w_uk, sizeof(w_uk), "f32", 2u, wukv_shape),
      make_view(w_uv, sizeof(w_uv), "f32", 2u, wukv_shape),
      make_view(w_o, sizeof(w_o), "f32", 2u, wo_shape),
      make_view(dy, sizeof(dy), "f32", 3u, x_shape)};
  et_kernel_tensor_view_v1 fwd_out =
      make_view(out, sizeof(out), "f32", 3u, x_shape);
  et_kernel_tensor_view_v1 bwd_out[6] = {
      make_view(dx, sizeof(dx), "f32", 3u, x_shape),
      make_view(dw_q, sizeof(dw_q), "f32", 2u, wq_shape),
      make_view(dw_dkv, sizeof(dw_dkv), "f32", 2u, wdkv_shape),
      make_view(dw_uk, sizeof(dw_uk), "f32", 2u, wukv_shape),
      make_view(dw_uv, sizeof(dw_uv), "f32", 2u, wukv_shape),
      make_view(dw_o, sizeof(dw_o), "f32", 2u, wo_shape)};

  et_kernel_request_v1 fwd_req = {sizeof(et_kernel_request_v1), "mla.forward",
                                  "f32", "cpu", 6u, req_shape, 1u, {0}};
  et_kernel_request_v1 bwd_req = {sizeof(et_kernel_request_v1), "mla.backward",
                                  "f32", "cpu", 6u, req_shape, 1u, {0}};
  et_kernel_call_v1 fwd_call =
      make_call("kernel.mla", &fwd_req, 6u, in_views, 1u, &fwd_out);
  et_kernel_call_v1 bwd_call =
      make_call("kernel.mla", &bwd_req, 7u, in_views, 6u, bwd_out);

  et_kernel_runtime *rt = discover_provider(et_mod5_mla_kernel_provider_v1());
  const uint32_t warmup = smoke ? 1u : 4u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t fwd_iters = smoke ? 1u : 6u;
  const uint32_t bwd_iters = smoke ? 1u : 3u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Forward */
  memset(out, 0, sizeof(out));
  k1_dispatch(rt, &fwd_call);
  check_finite_nonzero_f32(out, ARRAY_COUNT(out), "mod5_mla forward");
  sha256_hex_buf(out, sizeof(out), witness);
  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &fwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(out, 0, sizeof(out));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < fwd_iters; ++i) {
      k1_dispatch(rt, &fwd_call);
      acc ^= load_u32_head(&out[i % ARRAY_COUNT(out)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(out, sizeof(out), check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_mla forward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_MLA, "mla_forward_2x16x32_h4d16_dc8", "mla.forward",
              "N=2,T=16,D=32,H=4,Dh=16,dc=8", "tokens_per_sec", N * T, warmup,
              samples, fwd_iters, times, witness);

  /* Backward */
  memset(dx, 0, sizeof(dx));
  k1_dispatch(rt, &bwd_call);
  check_finite_nonzero_f32(dx, ARRAY_COUNT(dx), "mod5_mla backward dx");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, dw_q, sizeof(dw_q));
    sha256_update(&sc, dw_dkv, sizeof(dw_dkv));
    sha256_update(&sc, dw_uk, sizeof(dw_uk));
    sha256_update(&sc, dw_uv, sizeof(dw_uv));
    sha256_update(&sc, dw_o, sizeof(dw_o));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t w = 0; w < warmup; ++w)
    k1_dispatch(rt, &bwd_call);
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dx, 0, sizeof(dx));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < bwd_iters; ++i) {
      k1_dispatch(rt, &bwd_call);
      acc ^= load_u32_head(&dx[i % ARRAY_COUNT(dx)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, dw_q, sizeof(dw_q));
    sha256_update(&sc, dw_dkv, sizeof(dw_dkv));
    sha256_update(&sc, dw_uk, sizeof(dw_uk));
    sha256_update(&sc, dw_uv, sizeof(dw_uv));
    sha256_update(&sc, dw_o, sizeof(dw_o));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_mla backward witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_MLA, "mla_backward_2x16x32_h4d16_dc8", "mla.backward",
              "N=2,T=16,D=32,H=4,Dh=16,dc=8", "tokens_per_sec", N * T, warmup,
              samples, bwd_iters, times, witness);

  /* Compressed latent KV-cache prefill (8 tokens) + incremental decode (8) */
  static float c_kv[1 * T * DC];
  static float out_cache[T * D];
  et_mod5_mla_dims cdims = {1u, T, D, H, DH, DC};
  et_mod5_mla_kv_cache cache;
  et_kernel_error err;
  const size_t expected_latent = et_mod5_mla_latent_kv_bytes(1u, T, DC);
  REQUIRE(expected_latent == sizeof(c_kv), "mla latent kv bytes");
  REQUIRE(et_mod5_mla_cache_init(&cache, &cdims, c_kv, sizeof(c_kv), &err) == 0,
          "mla cache init");

  const uint32_t cache_iters = smoke ? 1u : 6u;
  /* Witness */
  REQUIRE(et_mod5_mla_cache_reset(&cache, &err) == 0, "mla cache reset");
  REQUIRE(et_mod5_mla_cache_prefill_f32(&cache, 8u, x, w_q, w_dkv, w_uk, w_uv,
                                        w_o, out_cache, &err) == 0,
          "mla cache prefill");
  for (size_t step = 8u; step < T; ++step) {
    REQUIRE(et_mod5_mla_cache_step_f32(&cache, x + step * D, w_q, w_dkv, w_uk,
                                       w_uv, w_o, out_cache + step * D,
                                       &err) == 0,
            "mla cache step");
  }
  check_finite_nonzero_f32(out_cache, T * D, "mod5_mla cache output");
  sha256_hex_buf(out_cache, sizeof(out_cache), witness);

  for (uint32_t w = 0; w < warmup; ++w) {
    REQUIRE(et_mod5_mla_cache_reset(&cache, &err) == 0, "mla warmup reset");
    REQUIRE(et_mod5_mla_cache_prefill_f32(&cache, 8u, x, w_q, w_dkv, w_uk, w_uv,
                                          w_o, out_cache, &err) == 0,
            "mla warmup prefill");
    for (size_t step = 8u; step < T; ++step) {
      REQUIRE(et_mod5_mla_cache_step_f32(&cache, x + step * D, w_q, w_dkv, w_uk,
                                         w_uv, w_o, out_cache + step * D,
                                         &err) == 0,
              "mla warmup step");
    }
  }
  for (uint32_t s = 0; s < samples; ++s) {
    memset(out_cache, 0, sizeof(out_cache));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < cache_iters; ++i) {
      REQUIRE(et_mod5_mla_cache_reset(&cache, &err) == 0, "mla timed reset");
      REQUIRE(et_mod5_mla_cache_prefill_f32(&cache, 8u, x, w_q, w_dkv, w_uk,
                                            w_uv, w_o, out_cache, &err) == 0,
              "mla timed prefill");
      for (size_t step = 8u; step < T; ++step) {
        REQUIRE(et_mod5_mla_cache_step_f32(&cache, x + step * D, w_q, w_dkv,
                                           w_uk, w_uv, w_o,
                                           out_cache + step * D, &err) == 0,
                "mla timed step");
      }
      acc ^= load_u32_head(&out_cache[(i * 7u) % (T * D)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(out_cache, sizeof(out_cache), check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_mla cache witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_MOD5_MLA, "mla_cache_prefill_decode_1x16_dc8",
              "mla.cache_prefill_decode", "N=1,prefill=8,decode=8,D=32,dc=8",
              "tokens_per_sec", T, warmup, samples, cache_iters, times,
              witness);

  et_kernel_runtime_destroy(rt);
}

/* ========================================================================= */
/* 4. MOD5 AdamW                                                             */
/* ========================================================================= */

static void bench_mod5_adamw_case(int smoke, const char *name, const char *op,
                                  const char *shape, size_t param_count,
                                  uint32_t sched_kind) {
  float *init_params = (float *)malloc(param_count * sizeof(float));
  float *grads = (float *)malloc(param_count * sizeof(float));
  float *out_p = (float *)malloc(param_count * sizeof(float));
  float *out_m = (float *)malloc(param_count * sizeof(float));
  float *out_v = (float *)malloc(param_count * sizeof(float));
  REQUIRE(init_params && grads && out_p && out_m && out_v, "adamw malloc");

  for (size_t i = 0; i < param_count; ++i) {
    init_params[i] = det_f32(i, 307u);
    grads[i] = det_f32(i, 311u) * 0.25f;
  }

  et_mod5_adamw_config_v1 cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.param_count = (uint64_t)param_count;
  cfg.beta1 = 0.9f;
  cfg.beta2 = 0.999f;
  cfg.eps = 1e-8f;
  cfg.weight_decay = 0.01f;
  cfg.max_grad_norm = 1.0f;
  cfg.schedule.kind = sched_kind;
  cfg.schedule.base_lr = 1e-3;
  cfg.schedule.min_lr = 1e-5;
  cfg.schedule.warmup_steps = 4u;
  cfg.schedule.total_steps = 100u;
  for (size_t i = 0; i < 32u; ++i)
    cfg.model_config_hash[i] = (uint8_t)(i * 7u + 3u);
  cfg.tokenizer_id = "wave5-bpe-v1";

  uint8_t cfg_hash[32];
  REQUIRE(et_mod5_adamw_compute_config_hash(&cfg, cfg_hash) == ET_MOD5_ADAMW_OK,
          "adamw config hash");

  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 3u : 15u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Untimed witness run of `iters` steps from clean state */
  et_mod5_adamw_state *st = NULL;
  REQUIRE(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_OK && st != NULL,
          "adamw create witness");
  REQUIRE(et_mod5_adamw_set_params(st, init_params, param_count) ==
              ET_MOD5_ADAMW_OK,
          "adamw set_params witness");
  et_mod5_adamw_step_info_v1 info;
  for (uint32_t i = 0; i < iters; ++i) {
    REQUIRE(et_mod5_adamw_step(st, grads, param_count, 64u, &info) ==
                ET_MOD5_ADAMW_OK,
            "adamw witness step");
  }
  REQUIRE(et_mod5_adamw_get_params(st, out_p, param_count) == ET_MOD5_ADAMW_OK,
          "adamw get_params witness");
  REQUIRE(et_mod5_adamw_get_moments(st, out_m, out_v, param_count) ==
              ET_MOD5_ADAMW_OK,
          "adamw get_moments witness");
  check_finite_nonzero_f32(out_p, param_count, "adamw out_p");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_p, param_count * sizeof(float));
    sha256_update(&sc, out_m, param_count * sizeof(float));
    sha256_update(&sc, out_v, param_count * sizeof(float));
    sha256_update(&sc, cfg_hash, sizeof(cfg_hash));
    sha256_final_hex(&sc, witness);
  }
  et_mod5_adamw_destroy(st);

  /* Warmup */
  REQUIRE(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_OK,
          "adamw create warmup");
  REQUIRE(et_mod5_adamw_set_params(st, init_params, param_count) ==
              ET_MOD5_ADAMW_OK,
          "adamw set_params warmup");
  for (uint32_t w = 0; w < warmup; ++w) {
    REQUIRE(et_mod5_adamw_step(st, grads, param_count, 64u, &info) ==
                ET_MOD5_ADAMW_OK,
            "adamw warmup step");
  }
  et_mod5_adamw_destroy(st);

  /* Measured samples */
  for (uint32_t s = 0; s < samples; ++s) {
    REQUIRE(et_mod5_adamw_create(&cfg, &st) == ET_MOD5_ADAMW_OK,
            "adamw create sample");
    REQUIRE(et_mod5_adamw_set_params(st, init_params, param_count) ==
                ET_MOD5_ADAMW_OK,
            "adamw set_params sample");
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_mod5_adamw_step(st, grads, param_count, 64u, &info) ==
                  ET_MOD5_ADAMW_OK,
              "adamw timed step");
      acc ^= (uint32_t)(info.step & UINT64_C(0xffffffff));
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    REQUIRE(et_mod5_adamw_get_params(st, out_p, param_count) ==
                ET_MOD5_ADAMW_OK,
            "adamw get_params sample");
    REQUIRE(et_mod5_adamw_get_moments(st, out_m, out_v, param_count) ==
                ET_MOD5_ADAMW_OK,
            "adamw get_moments sample");
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_p, param_count * sizeof(float));
    sha256_update(&sc, out_m, param_count * sizeof(float));
    sha256_update(&sc, out_v, param_count * sizeof(float));
    sha256_update(&sc, cfg_hash, sizeof(cfg_hash));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "mod5_adamw witness drift");
    et_mod5_adamw_destroy(st);
    times[s] = t1 - t0;
  }

  record_case(GRP_MOD5_ADAMW, name, op, shape, "elems_per_sec",
              (uint64_t)param_count, warmup, samples, iters, times, witness);

  free(out_v);
  free(out_m);
  free(out_p);
  free(grads);
  free(init_params);
}

static void bench_mod5_adamw(int smoke) {
  bench_mod5_adamw_case(smoke, "adamw_step_4096_cosine", "adamw.step_cosine",
                        "params=4096,clip=1.0", 4096u,
                        ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY);
  bench_mod5_adamw_case(smoke, "adamw_step_16384_linear", "adamw.step_linear",
                        "params=16384,clip=1.0", 16384u,
                        ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY);
}

/* ========================================================================= */
/* 5. O2 Muon + WSD                                                          */
/* ========================================================================= */

static void bench_o2_muon(int smoke) {
  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Case 1: Newton-Schulz 5-step orthogonalization on 32x32 matrix */
  {
    enum { R = 32, C = 32, ELEMS = R * C };
    static float g[ELEMS], u[ELEMS];
    for (size_t i = 0; i < ELEMS; ++i)
      g[i] = det_f32(i, 401u) + ((i / C == i % C) ? 0.5f : 0.0f);

    const uint32_t iters = smoke ? 2u : 10u;
    memset(u, 0, sizeof(u));
    REQUIRE(et_muon_newton_schulz5_f32(R, C, g, ET_MUON_DEFAULT_NS_STEPS, 1e-7,
                                       u) == ET_MUON_OK,
            "muon ns5 witness");
    double init_ortho_err = 0.0;
    REQUIRE(et_muon_orthogonality_error_f32(R, C, g, &init_ortho_err) ==
                ET_MUON_OK,
            "muon init ortho err");
    double ortho_err = 0.0;
    REQUIRE(et_muon_orthogonality_error_f32(R, C, u, &ortho_err) == ET_MUON_OK,
            "muon ortho err");
    REQUIRE(isfinite(ortho_err) && ortho_err < init_ortho_err,
            "muon ortho improvement");
    check_finite_nonzero_f32(u, ELEMS, "muon ns5 u");
    sha256_hex_buf(u, sizeof(u), witness);

    for (uint32_t w = 0; w < warmup; ++w)
      REQUIRE(et_muon_newton_schulz5_f32(R, C, g, ET_MUON_DEFAULT_NS_STEPS,
                                         1e-7, u) == ET_MUON_OK,
              "muon ns5 warmup");

    for (uint32_t s = 0; s < samples; ++s) {
      memset(u, 0, sizeof(u));
      uint32_t acc = 0u;
      const uint64_t t0 = now_ns();
      for (uint32_t i = 0; i < iters; ++i) {
        REQUIRE(et_muon_newton_schulz5_f32(R, C, g, ET_MUON_DEFAULT_NS_STEPS,
                                           1e-7, u) == ET_MUON_OK,
                "muon ns5 timed");
        acc ^= load_u32_head(&u[i % ELEMS]);
      }
      const uint64_t t1 = now_ns();
      g_dce_sink ^= acc;
      char check[65];
      sha256_hex_buf(u, sizeof(u), check);
      REQUIRE(strcmp(check, witness) == 0, "muon ns5 witness drift");
      times[s] = t1 - t0;
    }
    record_case(GRP_O2_MUON, "muon_ns5_32x32", "muon.newton_schulz5",
                "rows=32,cols=32,ns_steps=5", "elems_per_sec", ELEMS, warmup,
                samples, iters, times, witness);
  }

  /* Case 2: Hybrid Muon (2D) + AdamW (1D) step with WSD cosine schedule */
  {
    enum { N0 = 32 * 32, N1 = 32 * 16, N2 = 256 };
    static float p0_init[N0], g0[N0], p0[N0], m0[N0];
    static float p1_init[N1], g1[N1], p1[N1], m1[N1];
    static float p2_init[N2], g2[N2], p2[N2], m2[N2], v2[N2];

    for (size_t i = 0; i < N0; ++i) {
      p0_init[i] = det_f32(i, 409u);
      g0[i] = det_f32(i, 419u) * 0.2f;
    }
    for (size_t i = 0; i < N1; ++i) {
      p1_init[i] = det_f32(i, 421u);
      g1[i] = det_f32(i, 431u) * 0.2f;
    }
    for (size_t i = 0; i < N2; ++i) {
      p2_init[i] = det_f32(i, 433u);
      g2[i] = det_f32(i, 439u) * 0.2f;
    }

    et_wsd_config wsd = {.warmup_steps = 4u,
                         .stable_steps = 12u,
                         .total_steps = 24u,
                         .max_lr = 0.02,
                         .min_lr = 0.001,
                         .decay_kind = ET_WSD_DECAY_COSINE,
                         .reserved = 0u};
    et_muon_step_config scfg = {.lr = 0.02,
                                .step_index = 1u,
                                .muon_beta = 0.95,
                                .muon_eps = 1e-7,
                                .muon_weight_decay = 0.01,
                                .muon_ns_steps = ET_MUON_DEFAULT_NS_STEPS,
                                .muon_nesterov = 1u,
                                .adamw_beta1 = 0.9,
                                .adamw_beta2 = 0.999,
                                .adamw_eps = 1e-8,
                                .adamw_weight_decay = 0.0,
                                .adamw_lr_scale = 0.1};
    et_muon_param_group groups[3] = {
        {ET_MUON_GROUP_MUON_2D, 32u, 32u, 1u, p0, g0, m0, NULL},
        {ET_MUON_GROUP_MUON_2D, 32u, 16u, 1u, p1, g1, m1, NULL},
        {ET_MUON_GROUP_ADAMW_1D, 256u, 1u, 0u, p2, g2, m2, v2}};

    const uint32_t iters = smoke ? 2u : 6u;
    double out_lr = 0.0;
    uint32_t out_phase = 0u;

    /* Witness */
    memcpy(p0, p0_init, sizeof(p0));
    memcpy(p1, p1_init, sizeof(p1));
    memcpy(p2, p2_init, sizeof(p2));
    memset(m0, 0, sizeof(m0));
    memset(m1, 0, sizeof(m1));
    memset(m2, 0, sizeof(m2));
    memset(v2, 0, sizeof(v2));
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_muon_hybrid_step_wsd(&wsd, i, &scfg, 3u, groups, &out_lr,
                                      &out_phase) == ET_MUON_OK,
              "muon hybrid witness");
    }
    check_finite_nonzero_f32(p0, N0, "muon hybrid p0");
    check_finite_nonzero_f32(p2, N2, "muon hybrid p2");
    {
      sha256_ctx sc;
      sha256_init(&sc);
      sha256_update(&sc, p0, sizeof(p0));
      sha256_update(&sc, p1, sizeof(p1));
      sha256_update(&sc, p2, sizeof(p2));
      sha256_update(&sc, m0, sizeof(m0));
      sha256_update(&sc, m2, sizeof(m2));
      sha256_update(&sc, v2, sizeof(v2));
      sha256_final_hex(&sc, witness);
    }

    /* Warmup */
    memcpy(p0, p0_init, sizeof(p0));
    memcpy(p1, p1_init, sizeof(p1));
    memcpy(p2, p2_init, sizeof(p2));
    memset(m0, 0, sizeof(m0));
    memset(m1, 0, sizeof(m1));
    memset(m2, 0, sizeof(m2));
    memset(v2, 0, sizeof(v2));
    for (uint32_t w = 0; w < warmup; ++w) {
      REQUIRE(et_muon_hybrid_step_wsd(&wsd, w, &scfg, 3u, groups, &out_lr,
                                      &out_phase) == ET_MUON_OK,
              "muon hybrid warmup");
    }

    for (uint32_t s = 0; s < samples; ++s) {
      memcpy(p0, p0_init, sizeof(p0));
      memcpy(p1, p1_init, sizeof(p1));
      memcpy(p2, p2_init, sizeof(p2));
      memset(m0, 0, sizeof(m0));
      memset(m1, 0, sizeof(m1));
      memset(m2, 0, sizeof(m2));
      memset(v2, 0, sizeof(v2));
      uint32_t acc = 0u;
      const uint64_t t0 = now_ns();
      for (uint32_t i = 0; i < iters; ++i) {
        REQUIRE(et_muon_hybrid_step_wsd(&wsd, i, &scfg, 3u, groups, &out_lr,
                                        &out_phase) == ET_MUON_OK,
                "muon hybrid timed");
        acc ^= load_u32_head(&p0[i % N0]) ^ out_phase;
      }
      const uint64_t t1 = now_ns();
      g_dce_sink ^= acc;
      char check[65];
      sha256_ctx sc;
      sha256_init(&sc);
      sha256_update(&sc, p0, sizeof(p0));
      sha256_update(&sc, p1, sizeof(p1));
      sha256_update(&sc, p2, sizeof(p2));
      sha256_update(&sc, m0, sizeof(m0));
      sha256_update(&sc, m2, sizeof(m2));
      sha256_update(&sc, v2, sizeof(v2));
      sha256_final_hex(&sc, check);
      REQUIRE(strcmp(check, witness) == 0, "muon hybrid witness drift");
      times[s] = t1 - t0;
    }
    record_case(GRP_O2_MUON, "muon_hybrid_wsd_step", "muon.hybrid_step_wsd",
                "groups=32x32+32x16+256x1,ns=5", "steps_per_sec", 1u, warmup,
                samples, iters, times, witness);
  }
}

/* ========================================================================= */
/* 6. LoRA                                                                   */
/* ========================================================================= */

static void bench_lora(int smoke) {
  enum { M = 32, K = 64, N = 64, R = 8 };
  et_lora_config cfg = {M, K, N, R, 16.0f};
  static float x[M * K], w[N * K], a[R * K], b[N * R], dy[M * N];
  static float y[M * N], dx[M * K], da[R * K], db[N * R];
  static float w_merged[N * K], w_restored[N * K];

  for (size_t i = 0; i < ARRAY_COUNT(x); ++i)
    x[i] = det_f32(i, 503u);
  for (size_t i = 0; i < ARRAY_COUNT(w); ++i)
    w[i] = det_f32(i, 509u);
  for (size_t i = 0; i < ARRAY_COUNT(dy); ++i)
    dy[i] = det_f32(i, 521u);
  REQUIRE(et_lora_init_adapter(&cfg, UINT64_C(0x5eed10ca12345678), a, b) ==
              ET_LORA_OK,
          "lora init");
  for (size_t i = 0; i < ARRAY_COUNT(b); ++i)
    b[i] = det_f32(i, 523u) * 0.25f;

  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 2u : 15u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Forward */
  memset(y, 0, sizeof(y));
  REQUIRE(et_lora_forward(&cfg, x, w, a, b, y) == ET_LORA_OK,
          "lora fwd witness");
  check_finite_nonzero_f32(y, ARRAY_COUNT(y), "lora forward y");
  sha256_hex_buf(y, sizeof(y), witness);
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_lora_forward(&cfg, x, w, a, b, y) == ET_LORA_OK,
            "lora fwd warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(y, 0, sizeof(y));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_lora_forward(&cfg, x, w, a, b, y) == ET_LORA_OK,
              "lora fwd timed");
      acc ^= load_u32_head(&y[i % ARRAY_COUNT(y)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(y, sizeof(y), check);
    REQUIRE(strcmp(check, witness) == 0, "lora fwd witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_LORA, "lora_forward_32x64x64_r8", "lora.forward",
              "M=32,K=64,N=64,r=8", "elems_per_sec", M * N, warmup, samples,
              iters, times, witness);

  /* Backward */
  memset(dx, 0, sizeof(dx));
  memset(da, 0, sizeof(da));
  memset(db, 0, sizeof(db));
  REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK,
          "lora bwd witness");
  check_finite_nonzero_f32(dx, ARRAY_COUNT(dx), "lora bwd dx");
  check_finite_nonzero_f32(da, ARRAY_COUNT(da), "lora bwd da");
  check_finite_nonzero_f32(db, ARRAY_COUNT(db), "lora bwd db");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, da, sizeof(da));
    sha256_update(&sc, db, sizeof(db));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK,
            "lora bwd warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dx, 0, sizeof(dx));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_lora_backward(&cfg, x, w, a, b, dy, dx, da, db) == ET_LORA_OK,
              "lora bwd timed");
      acc ^= load_u32_head(&da[i % ARRAY_COUNT(da)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, dx, sizeof(dx));
    sha256_update(&sc, da, sizeof(da));
    sha256_update(&sc, db, sizeof(db));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "lora bwd witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_LORA, "lora_backward_32x64x64_r8", "lora.backward",
              "M=32,K=64,N=64,r=8", "elems_per_sec", M * N, warmup, samples,
              iters, times, witness);

  /* Merge + Unmerge */
  memset(w_merged, 0, sizeof(w_merged));
  memset(w_restored, 0, sizeof(w_restored));
  REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK,
          "lora merge witness");
  REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK,
          "lora unmerge witness");
  check_finite_nonzero_f32(w_merged, ARRAY_COUNT(w_merged), "lora w_merged");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, w_merged, sizeof(w_merged));
    sha256_update(&sc, w_restored, sizeof(w_restored));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi) {
    REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK,
            "lora merge warmup");
    REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK,
            "lora unmerge warmup");
  }
  for (uint32_t s = 0; s < samples; ++s) {
    memset(w_merged, 0, sizeof(w_merged));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_lora_merge(&cfg, w, a, b, w_merged) == ET_LORA_OK,
              "lora merge timed");
      REQUIRE(et_lora_unmerge(&cfg, w_merged, a, b, w_restored) == ET_LORA_OK,
              "lora unmerge timed");
      acc ^= load_u32_head(&w_merged[i % ARRAY_COUNT(w_merged)]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, w_merged, sizeof(w_merged));
    sha256_update(&sc, w_restored, sizeof(w_restored));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "lora merge/unmerge witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_LORA, "lora_merge_unmerge_64x64_r8", "lora.merge_unmerge",
              "K=64,N=64,r=8", "elems_per_sec", N * K, warmup, samples, iters,
              times, witness);
}

/* ========================================================================= */
/* 7. DPO / cDPO                                                             */
/* ========================================================================= */

static void bench_dpo(int smoke) {
  enum { B = 4, T = 32, V = 128, LOGITS = B * T * V, TOKENS = B * T };
  static float pol_c[LOGITS], pol_r[LOGITS], ref_c[LOGITS], ref_r[LOGITS];
  static float d_c[LOGITS], d_r[LOGITS];
  static int64_t tgt_c[TOKENS], tgt_r[TOKENS];
  static uint8_t msk_c[TOKENS], msk_r[TOKENS];

  for (size_t i = 0; i < LOGITS; ++i) {
    pol_c[i] = det_f32(i, 601u);
    pol_r[i] = det_f32(i, 607u);
    ref_c[i] = det_f32(i, 613u);
    ref_r[i] = det_f32(i, 617u);
  }
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      const size_t idx = b * T + t;
      tgt_c[idx] = (int64_t)((idx * 17u + 3u) % V);
      tgt_r[idx] = (int64_t)((idx * 29u + 7u) % V);
      msk_c[idx] = (uint8_t)(t >= 4u ? 1u : 0u);
      msk_r[idx] = (uint8_t)(t >= 4u ? 1u : 0u);
    }
  }

  et_dpo_config cfg_dpo = {B, T, V, 0u, 0.1, 0.0};
  et_dpo_config cfg_cdpo = {B, T, V, 0u, 0.1, 0.1};
  et_dpo_telemetry tel;

  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 2u : 12u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* DPO forward */
  memset(&tel, 0, sizeof(tel));
  REQUIRE(et_dpo_forward(&cfg_dpo, pol_c, pol_r, ref_c, ref_r, tgt_c, tgt_r,
                         msk_c, msk_r, &tel) == ET_DPO_OK,
          "dpo fwd witness");
  REQUIRE(isfinite(tel.mean_loss) && tel.mean_loss > 0.0, "dpo fwd loss");
  sha256_hex_buf(&tel, sizeof(tel), witness);
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_dpo_forward(&cfg_dpo, pol_c, pol_r, ref_c, ref_r, tgt_c, tgt_r,
                           msk_c, msk_r, &tel) == ET_DPO_OK,
            "dpo fwd warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(&tel, 0, sizeof(tel));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_dpo_forward(&cfg_dpo, pol_c, pol_r, ref_c, ref_r, tgt_c, tgt_r,
                             msk_c, msk_r, &tel) == ET_DPO_OK,
              "dpo fwd timed");
      acc ^= load_u32_head(&tel.mean_loss);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(&tel, sizeof(tel), check);
    REQUIRE(strcmp(check, witness) == 0, "dpo fwd witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_DPO, "dpo_forward_4x32x128", "dpo.forward",
              "B=4,T=32,V=128,beta=0.1", "tokens_per_sec", TOKENS * 2u, warmup,
              samples, iters, times, witness);

  /* cDPO forward + backward */
  memset(&tel, 0, sizeof(tel));
  memset(d_c, 0, sizeof(d_c));
  memset(d_r, 0, sizeof(d_r));
  REQUIRE(et_dpo_forward_backward(&cfg_cdpo, pol_c, pol_r, ref_c, ref_r, tgt_c,
                                  tgt_r, msk_c, msk_r, &tel, d_c,
                                  d_r) == ET_DPO_OK,
          "cdpo fwd_bwd witness");
  check_finite_nonzero_f32(d_c, LOGITS, "cdpo d_c");
  check_finite_nonzero_f32(d_r, LOGITS, "cdpo d_r");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, &tel, sizeof(tel));
    sha256_update(&sc, d_c, sizeof(d_c));
    sha256_update(&sc, d_r, sizeof(d_r));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_dpo_forward_backward(&cfg_cdpo, pol_c, pol_r, ref_c, ref_r,
                                    tgt_c, tgt_r, msk_c, msk_r, &tel, d_c,
                                    d_r) == ET_DPO_OK,
            "cdpo fwd_bwd warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(d_c, 0, sizeof(d_c));
    memset(d_r, 0, sizeof(d_r));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_dpo_forward_backward(&cfg_cdpo, pol_c, pol_r, ref_c, ref_r,
                                      tgt_c, tgt_r, msk_c, msk_r, &tel, d_c,
                                      d_r) == ET_DPO_OK,
              "cdpo fwd_bwd timed");
      acc ^= load_u32_head(&d_c[(i * 131u) % LOGITS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, &tel, sizeof(tel));
    sha256_update(&sc, d_c, sizeof(d_c));
    sha256_update(&sc, d_r, sizeof(d_r));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "cdpo fwd_bwd witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_DPO, "cdpo_forward_backward_4x32x128",
              "cdpo.forward_backward", "B=4,T=32,V=128,beta=0.1,ls=0.1",
              "tokens_per_sec", TOKENS * 2u, warmup, samples, iters, times,
              witness);
}

/* ========================================================================= */
/* 8. Speculative Decoding                                                   */
/* ========================================================================= */

static void bench_spec_decode(int smoke) {
  enum { K = 8, V = 1024 };
  static float draft_logits[K * V];
  static float target_logits[(K + 1) * V];
  uint32_t draft_tokens[K];
  uint32_t out_tokens[K + 1];
  et_spec_verify_result_v1 res;

  for (size_t i = 0; i < K * V; ++i)
    draft_logits[i] = det_f32(i, 701u);
  for (size_t i = 0; i < (K + 1) * V; ++i)
    target_logits[i] = det_f32(i, 709u);

  /* Make first 5 draft tokens strongly favored by both draft and target, and
     divergent on step 5 so both acceptance and correction are exercised. */
  for (size_t k = 0; k < K; ++k) {
    const uint32_t tok = (uint32_t)((k * 97u + 19u) % V);
    draft_tokens[k] = tok;
    draft_logits[k * V + tok] = 4.5f;
    if (k < 5u) {
      target_logits[k * V + tok] = 4.2f;
    } else {
      const uint32_t alt = (tok + 13u) % V;
      target_logits[k * V + alt] = 4.8f;
    }
  }
  target_logits[K * V + 42u] = 4.0f;

  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t greedy_iters = smoke ? 4u : 80u;
  const uint32_t sample_iters = smoke ? 2u : 15u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Greedy verification */
  memset(out_tokens, 0, sizeof(out_tokens));
  memset(&res, 0, sizeof(res));
  REQUIRE(et_spec_verify_greedy_v1(draft_tokens, target_logits, K, V,
                                   ET_SPEC_NO_EOS, out_tokens, K + 1u,
                                   &res) == ET_SPEC_OK,
          "spec greedy witness");
  REQUIRE(res.accepted_count == 5u && res.emitted_count == 6u,
          "spec greedy expected acceptance");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_tokens, res.emitted_count * sizeof(uint32_t));
    sha256_update(&sc, &res, sizeof(res));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_spec_verify_greedy_v1(draft_tokens, target_logits, K, V,
                                     ET_SPEC_NO_EOS, out_tokens, K + 1u,
                                     &res) == ET_SPEC_OK,
            "spec greedy warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(out_tokens, 0, sizeof(out_tokens));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < greedy_iters; ++i) {
      REQUIRE(et_spec_verify_greedy_v1(draft_tokens, target_logits, K, V,
                                       ET_SPEC_NO_EOS, out_tokens, K + 1u,
                                       &res) == ET_SPEC_OK,
              "spec greedy timed");
      acc ^= out_tokens[0] ^ res.emitted_count;
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_tokens, res.emitted_count * sizeof(uint32_t));
    sha256_update(&sc, &res, sizeof(res));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "spec greedy witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_SPEC_DECODE, "spec_verify_greedy_k8_v1024",
              "spec_decode.verify_greedy", "K=8,V=1024", "tokens_per_sec", K,
              warmup, samples, greedy_iters, times, witness);

  /* Stochastic rejection-sampling verification */
  et_spec_sampling_params_v1 sp = {.temperature = 0.8,
                                   .top_p = 0.95,
                                   .top_k = 64u,
                                   .eos_token_id = ET_SPEC_NO_EOS};
  et_spec_rng_state_v1 rng = {.seed = UINT64_C(0x123456789abcdef1),
                              .counter = 0u};

  memset(out_tokens, 0, sizeof(out_tokens));
  memset(&res, 0, sizeof(res));
  REQUIRE(et_spec_verify_sample_v1(draft_tokens, draft_logits, target_logits, K,
                                   V, &sp, &rng, out_tokens, K + 1u,
                                   &res) == ET_SPEC_OK,
          "spec sample witness");
  REQUIRE(res.emitted_count >= 1u && res.emitted_count <= K + 1u,
          "spec sample emitted bounds");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_tokens, res.emitted_count * sizeof(uint32_t));
    sha256_update(&sc, &res, sizeof(res));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_spec_verify_sample_v1(draft_tokens, draft_logits, target_logits,
                                     K, V, &sp, &rng, out_tokens, K + 1u,
                                     &res) == ET_SPEC_OK,
            "spec sample warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    memset(out_tokens, 0, sizeof(out_tokens));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < sample_iters; ++i) {
      REQUIRE(et_spec_verify_sample_v1(draft_tokens, draft_logits,
                                       target_logits, K, V, &sp, &rng,
                                       out_tokens, K + 1u, &res) == ET_SPEC_OK,
              "spec sample timed");
      acc ^= out_tokens[0] ^ (uint32_t)res.rng_draws;
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, out_tokens, res.emitted_count * sizeof(uint32_t));
    sha256_update(&sc, &res, sizeof(res));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "spec sample witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_SPEC_DECODE, "spec_verify_sample_k8_v1024",
              "spec_decode.verify_sample", "K=8,V=1024,top_k=64,top_p=0.95",
              "tokens_per_sec", K, warmup, samples, sample_iters, times,
              witness);
}

/* ========================================================================= */
/* 9. T2 BPE Trainer                                                         */
/* ========================================================================= */

static void bench_t2_trainer(int smoke) {
  enum { DOC_COUNT = 16, MAX_DOC_BYTES = 320 };
  static uint8_t doc_bufs[DOC_COUNT][MAX_DOC_BYTES];
  et_t2_bpe_document docs[DOC_COUNT];

  static const char *const phrases[8] = {
      "deterministic transformer native kernel benchmark ",
      "rotary position embeddings and latent attention ",
      "sparse mixture of experts load balancing router ",
      "adamw optimizer state snapshot checksum resume ",
      "muon newton schulz orthogonalization schedule ",
      "low rank adapter merge unmerge fine tuning ",
      "direct preference optimization reward margin ",
      "speculative decoding rejection sampling verification "};

  size_t total_corpus_bytes = 0u;
  for (size_t d = 0; d < DOC_COUNT; ++d) {
    char *dst = (char *)doc_bufs[d];
    const int n = snprintf(dst, MAX_DOC_BYTES,
                           "doc_%02zu: %s%s%s%siteration_%02zu end.", d,
                           phrases[d % 8u], phrases[(d * 3u + 1u) % 8u],
                           phrases[(d * 5u + 2u) % 8u],
                           phrases[(d + 4u) % 8u], d * 7u + 1u);
    REQUIRE(n > 0 && (size_t)n < MAX_DOC_BYTES, "t2 doc snprintf");
    docs[d].bytes = doc_bufs[d];
    docs[d].byte_count = (size_t)n;
    total_corpus_bytes += (size_t)n;
  }
  REQUIRE(total_corpus_bytes > 2000u, "t2 corpus size");

  et_t2_bpe_train_config tcfg = {.target_vocab_size = 384u,
                                 .min_frequency = 2u,
                                 .special_mask = ET_T2_BPE_SPECIAL_ALL};

  const uint32_t warmup = smoke ? 1u : 3u;
  const uint32_t samples = smoke ? 5u : 21u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Case 1: BPE training */
  et_t2_bpe_model *ref_model = NULL;
  REQUIRE(et_t2_bpe_train(docs, DOC_COUNT, &tcfg, &ref_model) == ET_T2_BPE_OK &&
              ref_model != NULL,
          "t2 train witness");
  et_t2_bpe_info info;
  REQUIRE(et_t2_bpe_get_info(ref_model, &info) == ET_T2_BPE_OK, "t2 get_info");
  REQUIRE(info.merge_count > 16u, "t2 non-trivial merges");

  static uint8_t ser_buf[ET_T2_BPE_ARTIFACT_MAX_BYTES];
  size_t ser_len = 0u;
  uint8_t ser_sha[32];
  REQUIRE(et_t2_bpe_serialize(ref_model, ser_buf, sizeof(ser_buf), &ser_len,
                              ser_sha) == ET_T2_BPE_OK,
          "t2 serialize witness");
  sha256_hex_buf(ser_buf, ser_len, witness);

  for (uint32_t wi = 0; wi < warmup; ++wi) {
    et_t2_bpe_model *m = NULL;
    REQUIRE(et_t2_bpe_train(docs, DOC_COUNT, &tcfg, &m) == ET_T2_BPE_OK,
            "t2 train warmup");
    et_t2_bpe_destroy(m);
  }
  for (uint32_t s = 0; s < samples; ++s) {
    et_t2_bpe_model *m = NULL;
    const uint64_t t0 = now_ns();
    REQUIRE(et_t2_bpe_train(docs, DOC_COUNT, &tcfg, &m) == ET_T2_BPE_OK,
            "t2 train timed");
    const uint64_t t1 = now_ns();
    size_t chk_len = 0u;
    REQUIRE(et_t2_bpe_serialize(m, ser_buf, sizeof(ser_buf), &chk_len, NULL) ==
                ET_T2_BPE_OK,
            "t2 serialize sample");
    char check[65];
    sha256_hex_buf(ser_buf, chk_len, check);
    REQUIRE(strcmp(check, witness) == 0, "t2 train witness drift");
    g_dce_sink ^= (uint32_t)chk_len;
    et_t2_bpe_destroy(m);
    times[s] = t1 - t0;
  }
  record_case(GRP_T2_TRAINER, "t2_bpe_train_v384", "t2_bpe.train",
              "docs=16,target_vocab=384,min_freq=2", "docs_per_sec", DOC_COUNT,
              warmup, samples, 1u, times, witness);

  /* Case 2: BPE encode + lossless decode over all 16 documents */
  static uint32_t tok_buf[MAX_DOC_BYTES];
  static uint8_t dec_buf[MAX_DOC_BYTES];
  const uint32_t enc_iters = smoke ? 2u : 12u;

  uint64_t total_tokens = 0u;
  {
    sha256_ctx sc;
    sha256_init(&sc);
    for (size_t d = 0; d < DOC_COUNT; ++d) {
      size_t ntok = 0u, ndec = 0u;
      REQUIRE(et_t2_bpe_encode(ref_model, docs[d].bytes, docs[d].byte_count,
                               tok_buf, ARRAY_COUNT(tok_buf),
                               &ntok) == ET_T2_BPE_OK,
              "t2 encode witness");
      REQUIRE(et_t2_bpe_decode(ref_model, tok_buf, ntok, dec_buf,
                               sizeof(dec_buf), &ndec) == ET_T2_BPE_OK,
              "t2 decode witness");
      REQUIRE(ndec == docs[d].byte_count &&
                  memcmp(dec_buf, docs[d].bytes, ndec) == 0,
              "t2 lossless roundtrip");
      total_tokens += (uint64_t)ntok;
      sha256_update(&sc, tok_buf, ntok * sizeof(uint32_t));
    }
    sha256_final_hex(&sc, witness);
  }
  REQUIRE(total_tokens > 0u, "t2 nonzero tokens");

  for (uint32_t wi = 0; wi < warmup; ++wi) {
    for (size_t d = 0; d < DOC_COUNT; ++d) {
      size_t ntok = 0u, ndec = 0u;
      REQUIRE(et_t2_bpe_encode(ref_model, docs[d].bytes, docs[d].byte_count,
                               tok_buf, ARRAY_COUNT(tok_buf),
                               &ntok) == ET_T2_BPE_OK,
              "t2 encode warmup");
      REQUIRE(et_t2_bpe_decode(ref_model, tok_buf, ntok, dec_buf,
                               sizeof(dec_buf), &ndec) == ET_T2_BPE_OK,
              "t2 decode warmup");
    }
  }
  for (uint32_t s = 0; s < samples; ++s) {
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t it = 0; it < enc_iters; ++it) {
      for (size_t d = 0; d < DOC_COUNT; ++d) {
        size_t ntok = 0u, ndec = 0u;
        REQUIRE(et_t2_bpe_encode(ref_model, docs[d].bytes, docs[d].byte_count,
                                 tok_buf, ARRAY_COUNT(tok_buf),
                                 &ntok) == ET_T2_BPE_OK,
                "t2 encode timed");
        REQUIRE(et_t2_bpe_decode(ref_model, tok_buf, ntok, dec_buf,
                                 sizeof(dec_buf), &ndec) == ET_T2_BPE_OK,
                "t2 decode timed");
        acc ^= tok_buf[0] ^ (uint32_t)ndec;
      }
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    sha256_ctx sc;
    sha256_init(&sc);
    for (size_t d = 0; d < DOC_COUNT; ++d) {
      size_t ntok = 0u;
      REQUIRE(et_t2_bpe_encode(ref_model, docs[d].bytes, docs[d].byte_count,
                               tok_buf, ARRAY_COUNT(tok_buf),
                               &ntok) == ET_T2_BPE_OK,
              "t2 encode verify");
      sha256_update(&sc, tok_buf, ntok * sizeof(uint32_t));
    }
    char check[65];
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "t2 encode/decode witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_T2_TRAINER, "t2_bpe_encode_decode_corpus",
              "t2_bpe.encode_decode", "docs=16,vocab=384", "tokens_per_sec",
              total_tokens, warmup, samples, enc_iters, times, witness);

  et_t2_bpe_destroy(ref_model);
}

/* ========================================================================= */
/* 10. DATA4 Pipeline                                                        */
/* ========================================================================= */

static void bench_data4_pipeline(int smoke) {
  enum { DOC_COUNT = 32, MAX_DOC = 320 };
  static char src_ids[DOC_COUNT][32];
  static uint8_t payloads[DOC_COUNT][MAX_DOC];
  et_data4_document docs[DOC_COUNT];

  static const char *const topics[20] = {
      "alpha compiler diagnostics verify static types and deterministic passes",
      "bravo matrix kernels compute exact binary64 reductions for stability",
      "charlie checkpoint headers bind versioned sha256 manifests atomically",
      "delta tokenizer merges preserve raw byte roundtrip equivalence always",
      "echo rotary embeddings scale angles smoothly across sequence positions",
      "foxtrot sparse routers balance token assignments among active experts",
      "golf latent caches compress key value projections into low rank vectors",
      "hotel optimizers track momentum variance and gradient clipping norms",
      "india orthogonal updates refine weight matrices with quintic iterations",
      "juliet low rank adapters train compact delta factors without base edits",
      "kilo preference objectives separate chosen responses from rejected text",
      "lima speculative verifiers accept matching prefixes with lossless math",
      "mike quantization codecs pack blocks into compact eight bit scales",
      "november corpus filters reject invalid sequences and noisy repetition",
      "oscar minhash signatures cluster near duplicate articles deterministically",
      "papa jaccard verification confirms exact byte ngram overlap thresholds",
      "quebec provenance reports record canonical dispositions with checksums",
      "romeo build scripts audit exported symbols and forbidden runtime imports",
      "sierra sanitizer harnesses catch leaks and undefined behavior early",
      "tango benchmark runners measure monotonic clock percentiles honestly"};

  for (size_t i = 0; i < DOC_COUNT; ++i) {
    snprintf(src_ids[i], sizeof(src_ids[i]), "wave5-doc-%02zu", i);
    docs[i].source_id = src_ids[i];
    if (i < 20u) {
      const int n = snprintf(
          (char *)payloads[i], MAX_DOC,
          "Record %02zu: %s.\nSecondary clause %02zu confirms unique content "
          "marker %zu with clean prose.",
          i, topics[i], i, i * 97u + 13u);
      REQUIRE(n > 0 && (size_t)n < MAX_DOC, "data4 unique doc");
      docs[i].bytes = payloads[i];
      docs[i].byte_count = (size_t)n;
    } else if (i < 24u) {
      /* Exact duplicates of docs 0..3 */
      const size_t src = i - 20u;
      memcpy(payloads[i], payloads[src], docs[src].byte_count);
      docs[i].bytes = payloads[i];
      docs[i].byte_count = docs[src].byte_count;
    } else if (i < 28u) {
      /* Near duplicates of docs 4..7 (minor tail edit) */
      const size_t src = (i - 24u) + 4u;
      const int n = snprintf((char *)payloads[i], MAX_DOC, "%.*s [v2]",
                             (int)docs[src].byte_count, (const char *)payloads[src]);
      REQUIRE(n > 0 && (size_t)n < MAX_DOC, "data4 near dup doc");
      docs[i].bytes = payloads[i];
      docs[i].byte_count = (size_t)n;
    } else if (i == 28u) {
      /* Too short */
      const char *s = "short";
      memcpy(payloads[i], s, 5u);
      docs[i].bytes = payloads[i];
      docs[i].byte_count = 5u;
    } else if (i == 29u) {
      /* Low alnum quality */
      const char *s = "!!!@@@###$$$%%%^^^&&&***((()))___+++---===///|||:::;;;";
      const size_t len = strlen(s);
      memcpy(payloads[i], s, len);
      docs[i].bytes = payloads[i];
      docs[i].byte_count = len;
    } else if (i == 30u) {
      /* Duplicate-line repetition */
      const char *s = "repeated quality line here\nrepeated quality line here\n"
                      "repeated quality line here\nrepeated quality line here\n";
      const size_t len = strlen(s);
      memcpy(payloads[i], s, len);
      docs[i].bytes = payloads[i];
      docs[i].byte_count = len;
    } else {
      /* Invalid UTF-8 */
      payloads[i][0] = 0xffu;
      payloads[i][1] = 0xfeu;
      payloads[i][2] = 0x80u;
      payloads[i][3] = 'a';
      docs[i].bytes = payloads[i];
      docs[i].byte_count = 4u;
    }
  }

  et_data4_pipeline_config cfg;
  REQUIRE(et_data4_config_default(&cfg) == ET_DATA4_OK, "data4 default config");
  cfg.min_bytes = 32u;
  cfg.max_bytes = 4096u;
  cfg.min_codepoints = 32u;
  cfg.max_codepoints = 4096u;
  cfg.min_printable_ratio = 0.85;
  cfg.min_alnum_ratio = 0.50;
  cfg.max_dup_line_ratio = 0.40;
  cfg.max_repeated_ngram_ratio = 0.50;
  cfg.rep_ngram_size = 5u;
  cfg.near_dup_enabled = 1u;
  cfg.dedup_ngram_size = 5u;
  cfg.minhash_num_perms = 64u;
  cfg.minhash_bands = 16u;
  cfg.minhash_seed = UINT64_C(0x123456789abcdef0);
  cfg.near_dup_jaccard_threshold = 0.5;

  static et_data4_doc_result results[DOC_COUNT];
  et_data4_pipeline_summary summary;

  const uint32_t warmup = smoke ? 1u : 4u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t run_iters = smoke ? 1u : 5u;
  const uint32_t rep_iters = smoke ? 2u : 15u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* Case 1: Pipeline run (UTF-8 + quality + exact + MinHash/LSH near-dedup) */
  REQUIRE(et_data4_pipeline_run(docs, DOC_COUNT, &cfg, results, DOC_COUNT,
                                &summary) == ET_DATA4_OK,
          "data4 pipeline witness");
  REQUIRE(summary.input_documents == DOC_COUNT, "data4 input count");
  REQUIRE(summary.kept_documents == 20u, "data4 kept count");
  REQUIRE(summary.duplicate_exact_documents == 4u, "data4 exact dup count");
  REQUIRE(summary.duplicate_near_documents == 4u, "data4 near dup count");
  REQUIRE(summary.rejected_utf8_documents == 1u, "data4 utf8 reject count");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, &summary, sizeof(summary));
    for (size_t i = 0; i < DOC_COUNT; ++i) {
      sha256_update(&sc, &results[i].disposition, sizeof(results[i].disposition));
      sha256_update(&sc, results[i].sha256, ET_DATA4_SHA256_BYTES);
    }
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi)
    REQUIRE(et_data4_pipeline_run(docs, DOC_COUNT, &cfg, results, DOC_COUNT,
                                  &summary) == ET_DATA4_OK,
            "data4 pipeline warmup");
  for (uint32_t s = 0; s < samples; ++s) {
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < run_iters; ++i) {
      REQUIRE(et_data4_pipeline_run(docs, DOC_COUNT, &cfg, results, DOC_COUNT,
                                    &summary) == ET_DATA4_OK,
              "data4 pipeline timed");
      acc ^= (uint32_t)summary.kept_bytes;
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, &summary, sizeof(summary));
    for (size_t i = 0; i < DOC_COUNT; ++i) {
      sha256_update(&sc, &results[i].disposition, sizeof(results[i].disposition));
      sha256_update(&sc, results[i].sha256, ET_DATA4_SHA256_BYTES);
    }
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "data4 pipeline witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_DATA4_PIPELINE, "data4_pipeline_run_32docs",
              "data4.pipeline_run", "docs=32,perms=64,bands=16,ngram=5",
              "docs_per_sec", DOC_COUNT, warmup, samples, run_iters, times,
              witness);

  /* Case 2: Canonical TSV provenance report format + verify */
  size_t req_len = 0u;
  uint8_t rep_sha[ET_DATA4_SHA256_BYTES];
  REQUIRE(et_data4_report_format(&cfg, &summary, results, DOC_COUNT, NULL, 0u,
                                 &req_len, rep_sha) == ET_DATA4_OK,
          "data4 report size query");
  char *rep_buf = (char *)malloc(req_len + 1u);
  REQUIRE(rep_buf != NULL, "data4 report malloc");
  size_t out_len = 0u;
  REQUIRE(et_data4_report_format(&cfg, &summary, results, DOC_COUNT, rep_buf,
                                 req_len + 1u, &out_len,
                                 rep_sha) == ET_DATA4_OK,
          "data4 report format witness");
  et_data4_pipeline_summary verified_sum;
  REQUIRE(et_data4_report_verify(rep_buf, out_len, NULL, &verified_sum, NULL,
                                 0u) == ET_DATA4_OK,
          "data4 report verify witness");
  REQUIRE(memcmp(verified_sum.report_sha256, rep_sha, ET_DATA4_SHA256_BYTES) ==
              0,
          "data4 report sha256 match");
  sha256_hex_buf(rep_buf, out_len, witness);

  for (uint32_t wi = 0; wi < warmup; ++wi) {
    REQUIRE(et_data4_report_format(&cfg, &summary, results, DOC_COUNT, rep_buf,
                                   req_len + 1u, &out_len,
                                   rep_sha) == ET_DATA4_OK,
            "data4 report format warmup");
    REQUIRE(et_data4_report_verify(rep_buf, out_len, NULL, &verified_sum, NULL,
                                   0u) == ET_DATA4_OK,
            "data4 report verify warmup");
  }
  for (uint32_t s = 0; s < samples; ++s) {
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < rep_iters; ++i) {
      REQUIRE(et_data4_report_format(&cfg, &summary, results, DOC_COUNT,
                                     rep_buf, req_len + 1u, &out_len,
                                     rep_sha) == ET_DATA4_OK,
              "data4 report format timed");
      REQUIRE(et_data4_report_verify(rep_buf, out_len, NULL, &verified_sum,
                                     NULL, 0u) == ET_DATA4_OK,
              "data4 report verify timed");
      acc ^= (uint32_t)out_len ^ (uint32_t)rep_sha[0];
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_hex_buf(rep_buf, out_len, check);
    REQUIRE(strcmp(check, witness) == 0, "data4 report witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_DATA4_PIPELINE, "data4_report_format_verify_32docs",
              "data4.report_format_verify", "docs=32", "docs_per_sec",
              DOC_COUNT, warmup, samples, rep_iters, times, witness);
  free(rep_buf);
}

/* ========================================================================= */
/* 11. IO5 Quant Codec                                                       */
/* ========================================================================= */

static void bench_io5_quant(int smoke) {
  enum { ELEMS = 4096 };
  static float src[ELEMS], dst[ELEMS];
  static uint8_t narrow[ELEMS * 2u];
  size_t q8_bytes = 0u;
  REQUIRE(et_io5_q8_0_byte_size(ELEMS, &q8_bytes) == ET_IO5_QUANT_OK,
          "io5 q8_0 size");
  uint8_t *q8_buf = (uint8_t *)malloc(q8_bytes);
  REQUIRE(q8_buf != NULL, "io5 q8_0 malloc");

  for (size_t i = 0; i < ELEMS; ++i)
    src[i] = det_f32(i, 907u) * 8.0f;

  const uint32_t warmup = smoke ? 2u : 5u;
  const uint32_t samples = smoke ? 5u : 21u;
  const uint32_t iters = smoke ? 4u : 80u;
  uint64_t times[MAX_SAMPLES];
  char witness[65];

  /* F16 encode + decode */
  memset(narrow, 0, sizeof(narrow));
  memset(dst, 0, sizeof(dst));
  REQUIRE(et_io5_encode_f16(src, sizeof(src), narrow, sizeof(narrow), ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 f16 encode witness");
  REQUIRE(et_io5_decode_f16(narrow, sizeof(narrow), dst, sizeof(dst), ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 f16 decode witness");
  check_finite_nonzero_f32(dst, ELEMS, "io5 f16 dst");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, narrow, sizeof(narrow));
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi) {
    REQUIRE(et_io5_encode_f16(src, sizeof(src), narrow, sizeof(narrow),
                              ELEMS) == ET_IO5_QUANT_OK,
            "io5 f16 enc warmup");
    REQUIRE(et_io5_decode_f16(narrow, sizeof(narrow), dst, sizeof(dst),
                              ELEMS) == ET_IO5_QUANT_OK,
            "io5 f16 dec warmup");
  }
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dst, 0, sizeof(dst));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_io5_encode_f16(src, sizeof(src), narrow, sizeof(narrow),
                                ELEMS) == ET_IO5_QUANT_OK,
              "io5 f16 enc timed");
      REQUIRE(et_io5_decode_f16(narrow, sizeof(narrow), dst, sizeof(dst),
                                ELEMS) == ET_IO5_QUANT_OK,
              "io5 f16 dec timed");
      acc ^= load_u32_head(&dst[i % ELEMS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, narrow, sizeof(narrow));
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "io5 f16 witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_IO5_QUANT, "io5_f16_roundtrip_4096", "io5.f16_roundtrip",
              "elems=4096", "elems_per_sec", ELEMS, warmup, samples, iters,
              times, witness);

  /* BF16 encode + decode */
  memset(narrow, 0, sizeof(narrow));
  memset(dst, 0, sizeof(dst));
  REQUIRE(et_io5_encode_bf16(src, sizeof(src), narrow, sizeof(narrow), ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 bf16 encode witness");
  REQUIRE(et_io5_decode_bf16(narrow, sizeof(narrow), dst, sizeof(dst), ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 bf16 decode witness");
  check_finite_nonzero_f32(dst, ELEMS, "io5 bf16 dst");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, narrow, sizeof(narrow));
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi) {
    REQUIRE(et_io5_encode_bf16(src, sizeof(src), narrow, sizeof(narrow),
                               ELEMS) == ET_IO5_QUANT_OK,
            "io5 bf16 enc warmup");
    REQUIRE(et_io5_decode_bf16(narrow, sizeof(narrow), dst, sizeof(dst),
                               ELEMS) == ET_IO5_QUANT_OK,
            "io5 bf16 dec warmup");
  }
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dst, 0, sizeof(dst));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_io5_encode_bf16(src, sizeof(src), narrow, sizeof(narrow),
                                 ELEMS) == ET_IO5_QUANT_OK,
              "io5 bf16 enc timed");
      REQUIRE(et_io5_decode_bf16(narrow, sizeof(narrow), dst, sizeof(dst),
                                 ELEMS) == ET_IO5_QUANT_OK,
              "io5 bf16 dec timed");
      acc ^= load_u32_head(&dst[i % ELEMS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, narrow, sizeof(narrow));
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "io5 bf16 witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_IO5_QUANT, "io5_bf16_roundtrip_4096", "io5.bf16_roundtrip",
              "elems=4096", "elems_per_sec", ELEMS, warmup, samples, iters,
              times, witness);

  /* Q8_0 quantize + dequantize */
  memset(q8_buf, 0, q8_bytes);
  memset(dst, 0, sizeof(dst));
  REQUIRE(et_io5_quantize_q8_0(src, sizeof(src), q8_buf, q8_bytes, ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 q8_0 quant witness");
  REQUIRE(et_io5_dequantize_q8_0(q8_buf, q8_bytes, dst, sizeof(dst), ELEMS) ==
              ET_IO5_QUANT_OK,
          "io5 q8_0 dequant witness");
  check_finite_nonzero_f32(dst, ELEMS, "io5 q8_0 dst");
  {
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, q8_buf, q8_bytes);
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, witness);
  }
  for (uint32_t wi = 0; wi < warmup; ++wi) {
    REQUIRE(et_io5_quantize_q8_0(src, sizeof(src), q8_buf, q8_bytes, ELEMS) ==
                ET_IO5_QUANT_OK,
            "io5 q8_0 q warmup");
    REQUIRE(et_io5_dequantize_q8_0(q8_buf, q8_bytes, dst, sizeof(dst), ELEMS) ==
                ET_IO5_QUANT_OK,
            "io5 q8_0 dq warmup");
  }
  for (uint32_t s = 0; s < samples; ++s) {
    memset(dst, 0, sizeof(dst));
    uint32_t acc = 0u;
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      REQUIRE(et_io5_quantize_q8_0(src, sizeof(src), q8_buf, q8_bytes, ELEMS) ==
                  ET_IO5_QUANT_OK,
              "io5 q8_0 q timed");
      REQUIRE(et_io5_dequantize_q8_0(q8_buf, q8_bytes, dst, sizeof(dst),
                                     ELEMS) == ET_IO5_QUANT_OK,
              "io5 q8_0 dq timed");
      acc ^= load_u32_head(&dst[i % ELEMS]);
    }
    const uint64_t t1 = now_ns();
    g_dce_sink ^= acc;
    char check[65];
    sha256_ctx sc;
    sha256_init(&sc);
    sha256_update(&sc, q8_buf, q8_bytes);
    sha256_update(&sc, dst, sizeof(dst));
    sha256_final_hex(&sc, check);
    REQUIRE(strcmp(check, witness) == 0, "io5 q8_0 witness drift");
    times[s] = t1 - t0;
  }
  record_case(GRP_IO5_QUANT, "io5_q8_0_roundtrip_4096", "io5.q8_0_roundtrip",
              "elems=4096,blocks=128", "elems_per_sec", ELEMS, warmup, samples,
              iters, times, witness);

  free(q8_buf);
}

/* ========================================================================= */
/* Canonical JSON builder & host metadata                                    */
/* ========================================================================= */

typedef struct {
  char *data;
  size_t len;
  size_t cap;
} strbuf;

static void sb_init(strbuf *sb) {
  sb->cap = 16384u;
  sb->len = 0u;
  sb->data = (char *)malloc(sb->cap);
  REQUIRE(sb->data != NULL, "strbuf malloc");
  sb->data[0] = '\0';
}

static void sb_free(strbuf *sb) {
  free(sb->data);
  sb->data = NULL;
  sb->len = 0u;
  sb->cap = 0u;
}

static void sb_reserve(strbuf *sb, size_t extra) {
  if (sb->len + extra + 1u <= sb->cap)
    return;
  size_t next = sb->cap;
  while (next < sb->len + extra + 1u)
    next *= 2u;
  char *p = (char *)realloc(sb->data, next);
  REQUIRE(p != NULL, "strbuf realloc");
  sb->data = p;
  sb->cap = next;
}

static void sb_append_mem(strbuf *sb, const char *s, size_t n) {
  sb_reserve(sb, n);
  memcpy(sb->data + sb->len, s, n);
  sb->len += n;
  sb->data[sb->len] = '\0';
}

static void sb_append(strbuf *sb, const char *s) {
  sb_append_mem(sb, s, strlen(s));
}

static void sb_append_char(strbuf *sb, char c) { sb_append_mem(sb, &c, 1u); }

static void sb_append_json_str(strbuf *sb, const char *s) {
  sb_append_char(sb, '"');
  for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; ++p) {
    const unsigned char c = *p;
    if (c == '"')
      sb_append(sb, "\\\"");
    else if (c == '\\')
      sb_append(sb, "\\\\");
    else if (c == '\b')
      sb_append(sb, "\\b");
    else if (c == '\f')
      sb_append(sb, "\\f");
    else if (c == '\n')
      sb_append(sb, "\\n");
    else if (c == '\r')
      sb_append(sb, "\\r");
    else if (c == '\t')
      sb_append(sb, "\\t");
    else if (c < 0x20u) {
      char esc[7];
      snprintf(esc, sizeof(esc), "\\u%04x", (unsigned)c);
      sb_append(sb, esc);
    } else {
      sb_append_char(sb, (char)c);
    }
  }
  sb_append_char(sb, '"');
}

static void sb_append_u64(strbuf *sb, uint64_t v) {
  char buf[32];
  const int n = snprintf(buf, sizeof(buf), "%" PRIu64, v);
  REQUIRE(n > 0 && (size_t)n < sizeof(buf), "u64 format");
  sb_append_mem(sb, buf, (size_t)n);
}

/* Format finite double to match Python json.dumps canonical representation. */
static void sb_append_double(strbuf *sb, double v) {
  REQUIRE(isfinite(v), "non-finite double in report");
  if (v == 0.0) {
    sb_append(sb, "0.0");
    return;
  }
  char buf[64];
  const double mag = fabs(v);
  if (mag >= 0.00009999995 && mag < 1e8) {
    const int n = snprintf(buf, sizeof(buf), "%.6f", v);
    REQUIRE(n > 0 && (size_t)n < sizeof(buf), "double fixed6 format");
    size_t len = (size_t)n;
    while (len > 1u && buf[len - 1u] == '0' && buf[len - 2u] != '.')
      --len;
    sb_append_mem(sb, buf, len);
  } else if (mag >= 1e8 && mag < 1e12) {
    const int n = snprintf(buf, sizeof(buf), "%.2f", v);
    REQUIRE(n > 0 && (size_t)n < sizeof(buf), "double fixed2 format");
    size_t len = (size_t)n;
    while (len > 1u && buf[len - 1u] == '0' && buf[len - 2u] != '.')
      --len;
    sb_append_mem(sb, buf, len);
  } else if (mag >= 1e12 && mag < 9.9999995e15) {
    const int n = snprintf(buf, sizeof(buf), "%.1f", v);
    REQUIRE(n > 0 && (size_t)n < sizeof(buf), "double fixed1 format");
    sb_append_mem(sb, buf, (size_t)n);
  } else {
    const int n = snprintf(buf, sizeof(buf), "%.6e", v);
    REQUIRE(n > 0 && (size_t)n < sizeof(buf), "double sci format");
    char *e = strchr(buf, 'e');
    REQUIRE(e != NULL, "double sci exponent");
    size_t mlen = (size_t)(e - buf);
    while (mlen > 1u && buf[mlen - 1u] == '0')
      --mlen;
    if (mlen > 1u && buf[mlen - 1u] == '.')
      --mlen;
    sb_append_mem(sb, buf, mlen);
    sb_append(sb, e);
  }
}

static void read_os_release(char out[128]) {
  snprintf(out, 128u, "Linux");
  FILE *fp = fopen("/etc/os-release", "r");
  if (!fp)
    return;
  char line[256];
  while (fgets(line, (int)sizeof(line), fp) != NULL) {
    if (strncmp(line, "PRETTY_NAME=", 12u) == 0) {
      char *val = line + 12u;
      size_t len = strlen(val);
      while (len > 0u && (val[len - 1u] == '\n' || val[len - 1u] == '\r'))
        val[--len] = '\0';
      if (len >= 2u && val[0] == '"' && val[len - 1u] == '"') {
        val[len - 1u] = '\0';
        ++val;
      }
      if (*val != '\0')
        snprintf(out, 128u, "%s", val);
      break;
    }
  }
  fclose(fp);
}

static void read_cpu_model(char out[128]) {
  snprintf(out, 128u, "unknown");
  FILE *fp = fopen("/proc/cpuinfo", "r");
  if (!fp)
    return;
  char line[256];
  while (fgets(line, (int)sizeof(line), fp) != NULL) {
    if (strncmp(line, "model name", 10u) == 0) {
      char *colon = strchr(line, ':');
      if (colon) {
        char *val = colon + 1;
        while (*val == ' ' || *val == '\t')
          ++val;
        size_t len = strlen(val);
        while (len > 0u && (val[len - 1u] == '\n' || val[len - 1u] == '\r' ||
                            val[len - 1u] == ' '))
          val[--len] = '\0';
        if (*val != '\0')
          snprintf(out, 128u, "%s", val);
      }
      break;
    }
  }
  fclose(fp);
}

static void format_utc_now(char out[32]) {
  struct timespec ts;
  REQUIRE(clock_gettime(CLOCK_REALTIME, &ts) == 0, "clock_gettime realtime");
  struct tm utc;
  REQUIRE(gmtime_r(&ts.tv_sec, &utc) != NULL, "gmtime_r");
  const size_t n = strftime(out, 32u, "%Y-%m-%dT%H:%M:%SZ", &utc);
  REQUIRE(n > 0u, "strftime utc");
}

static void write_file_atomic(const char *path, const char *data, size_t len) {
  const size_t path_len = strlen(path);
  char *tmp = (char *)malloc(path_len + 64u);
  REQUIRE(tmp != NULL, "atomic path malloc");
  snprintf(tmp, path_len + 64u, "%s.tmp.%ld", path, (long)getpid());

  const int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    free(tmp);
    REQUIRE(0, "open atomic temp file");
  }
  size_t written = 0u;
  while (written < len) {
    const ssize_t n = write(fd, data + written, len - written);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      close(fd);
      unlink(tmp);
      free(tmp);
      REQUIRE(0, "write atomic temp file");
    }
    written += (size_t)n;
  }
  if (fsync(fd) != 0 || close(fd) != 0) {
    unlink(tmp);
    free(tmp);
    REQUIRE(0, "fsync/close atomic temp file");
  }
  if (rename(tmp, path) != 0) {
    unlink(tmp);
    free(tmp);
    REQUIRE(0, "rename atomic report");
  }
  free(tmp);
}

/* ========================================================================= */
/* CLI & Canonical Body Construction                                         */
/* ========================================================================= */

typedef struct {
  const char *output_path;
  const char *archive_dir;
  const char *selected_group;
  const char *compiler_id;
  const char *cflags;
  const char *build_mode;
  const char *git_commit;
  int git_dirty;
  int smoke;
  int selected_group_idx; /* -1 for all */
} cli_config;

static int find_group_index(const char *name) {
  for (int i = 0; i < NUM_GROUPS; ++i) {
    if (strcmp(name, g_group_descs[i].key) == 0)
      return i;
  }
  return -1;
}

static void run_group(int idx, int smoke) {
  switch ((group_id)idx) {
  case GRP_DATA4_PIPELINE:
    bench_data4_pipeline(smoke);
    break;
  case GRP_DPO:
    bench_dpo(smoke);
    break;
  case GRP_IO5_QUANT:
    bench_io5_quant(smoke);
    break;
  case GRP_LORA:
    bench_lora(smoke);
    break;
  case GRP_MOD5_ADAMW:
    bench_mod5_adamw(smoke);
    break;
  case GRP_MOD5_MLA:
    bench_mod5_mla(smoke);
    break;
  case GRP_MOD5_MOE:
    bench_mod5_moe(smoke);
    break;
  case GRP_MOD5_ROPE:
    bench_mod5_rope(smoke);
    break;
  case GRP_O2_MUON:
    bench_o2_muon(smoke);
    break;
  case GRP_SPEC_DECODE:
    bench_spec_decode(smoke);
    break;
  case GRP_T2_TRAINER:
    bench_t2_trainer(smoke);
    break;
  default:
    REQUIRE(0, "invalid group id");
  }
}

static void build_canonical_body(strbuf *p, const cli_config *cfg,
                                 uint64_t self_rss_kib,
                                 const char archive_shas[NUM_GROUPS][65],
                                 const char k1_sha[65]) {
  struct utsname u;
  REQUIRE(uname(&u) == 0, "uname");
  char os_rel[128], cpu_model[128], utc[32];
  read_os_release(os_rel);
  read_cpu_model(cpu_model);
  format_utc_now(utc);
  long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpus < 1L)
    ncpus = 1L;

  /* Top-level body keys (strictly sorted):
     environment, groups, measurement, process, schema, toolchain, version */
  sb_append(p, "{\"environment\":{\"architecture\":");
  sb_append_json_str(p, u.machine);
  sb_append(p, ",\"cpu_model\":");
  sb_append_json_str(p, cpu_model);
  sb_append(p, ",\"kernel\":");
  sb_append_json_str(p, u.release);
  sb_append(p, ",\"logical_cpus\":");
  sb_append_u64(p, (uint64_t)ncpus);
  sb_append(p, ",\"observed_at_utc\":");
  sb_append_json_str(p, utc);
  sb_append(p, ",\"os_release\":");
  sb_append_json_str(p, os_rel);

  sb_append(p, "},\"groups\":{");
  int first_group = 1;
  for (int g = 0; g < NUM_GROUPS; ++g) {
    if (!g_results[g].executed)
      continue;
    if (!first_group)
      sb_append_char(p, ',');
    first_group = 0;
    sb_append_json_str(p, g_group_descs[g].key);
    sb_append(p, ":{\"archive\":");
    sb_append_json_str(p, g_group_descs[g].archive_name);
    sb_append(p, ",\"cases\":[");
    for (size_t i = 0; i < g_results[g].case_count; ++i) {
      const bench_case_result *r = &g_results[g].cases[i];
      if (i > 0u)
        sb_append_char(p, ',');
      sb_append(p, "{\"checksum\":");
      sb_append_u64(p, r->checksum);
      sb_append(p, ",\"iterations\":");
      sb_append_u64(p, r->iterations);
      sb_append(p, ",\"max_ns\":");
      sb_append_u64(p, r->max_ns);
      sb_append(p, ",\"mean_ns\":");
      sb_append_double(p, r->mean_ns);
      sb_append(p, ",\"median_ns\":");
      sb_append_u64(p, r->median_ns);
      sb_append(p, ",\"median_ns_per_iter\":");
      sb_append_double(p, r->median_ns_per_iter);
      sb_append(p, ",\"min_ns\":");
      sb_append_u64(p, r->min_ns);
      sb_append(p, ",\"name\":");
      sb_append_json_str(p, r->name);
      sb_append(p, ",\"operation\":");
      sb_append_json_str(p, r->operation);
      sb_append(p, ",\"output_sha256\":");
      sb_append_json_str(p, r->output_sha256);
      sb_append(p, ",\"p95_ns\":");
      sb_append_u64(p, r->p95_ns);
      sb_append(p, ",\"samples\":");
      sb_append_u64(p, r->samples);
      sb_append(p, ",\"shape\":");
      sb_append_json_str(p, r->shape);
      sb_append(p, ",\"throughput\":");
      sb_append_double(p, r->throughput);
      sb_append(p, ",\"throughput_unit\":");
      sb_append_json_str(p, r->throughput_unit);
      sb_append(p, ",\"warmup\":");
      sb_append_u64(p, r->warmup);
      sb_append(p, ",\"work_units_per_iter\":");
      sb_append_u64(p, r->work_units_per_iter);
      sb_append_char(p, '}');
    }
    sb_append(p, "],\"engine\":");
    sb_append_json_str(p, g_group_descs[g].engine_label);
    sb_append_char(p, '}');
  }

  sb_append(p, "},\"measurement\":{\"backend\":{\"acceleration\":false,"
               "\"device\":\"cpu\",\"evidence\":\"direct-native-c11-link\","
               "\"name\":\"host-cpu\"},\"clock\":\"clock_monotonic_raw\","
               "\"elapsed_unit\":\"nanosecond\",\"limitations\":["
               "\"CPU-only serial execution; no GPU, HIP/ROCm, or CUDA kernels\","
               "\"All 11 native Wave 5 engines execute bounded verified CPU ABI shapes only\","
               "\"Single-machine unpinned timings reflect host load and frequency scaling\""
               "],\"memory_source\":\"linux_getrusage_rusage_self\","
               "\"peak_rss_unit\":\"kibibyte\",\"selected_group\":");
  sb_append_json_str(p, cfg->selected_group);
  sb_append(p, ",\"smoke_mode\":");
  sb_append(p, cfg->smoke ? "true" : "false");

  sb_append(p, "},\"process\":{\"self_peak_rss_kib\":");
  sb_append_u64(p, self_rss_kib);

  sb_append(p, "},\"schema\":\"wave5-bench-v1\",\"toolchain\":{\"archives\":{");
  for (int g = 0; g < NUM_GROUPS; ++g) {
    if (g > 0)
      sb_append_char(p, ',');
    sb_append_json_str(p, g_group_descs[g].key);
    sb_append(p, ":{\"filename\":");
    sb_append_json_str(p, g_group_descs[g].archive_name);
    sb_append(p, ",\"sha256\":");
    sb_append_json_str(p, archive_shas[g]);
    sb_append_char(p, '}');
  }
  sb_append(p, "},\"build_mode\":");
  sb_append_json_str(p, cfg->build_mode);
  sb_append(p, ",\"cflags\":");
  sb_append_json_str(p, cfg->cflags);
  sb_append(p, ",\"compiler\":");
  sb_append_json_str(p, cfg->compiler_id);
  sb_append(p, ",\"git_commit\":");
  sb_append_json_str(p, cfg->git_commit);
  sb_append(p, ",\"git_dirty\":");
  sb_append(p, cfg->git_dirty ? "true" : "false");
  sb_append(p, ",\"k1_archive\":{\"filename\":\"libeshkol_transformer_k1.a\","
               "\"sha256\":");
  sb_append_json_str(p, k1_sha);
  sb_append(p, "}},\"version\":1}");
}

static void print_usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s --output <path> [--smoke | --full] [--group <name|all>] "
          "[--archive-dir <dir>] [--compiler-id <str>] [--cflags <str>] "
          "[--build-mode normal|sanitize] [--git-commit <sha>] "
          "[--git-dirty 0|1]\n",
          prog);
}

int main(int argc, char **argv) {
  cli_config cfg = {.output_path = NULL,
                    .archive_dir = "build",
                    .selected_group = "all",
                    .compiler_id = __VERSION__,
                    .cflags = "c11",
                    .build_mode = "normal",
                    .git_commit = "unknown",
                    .git_dirty = 0,
                    .smoke = 0,
                    .selected_group_idx = -1};
  int seen_smoke = 0;
  int seen_full = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--output") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.output_path = argv[++i];
    } else if (strcmp(argv[i], "--archive-dir") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.archive_dir = argv[++i];
    } else if (strcmp(argv[i], "--group") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.selected_group = argv[++i];
    } else if (strcmp(argv[i], "--smoke") == 0) {
      seen_smoke = 1;
      cfg.smoke = 1;
    } else if (strcmp(argv[i], "--full") == 0) {
      seen_full = 1;
      cfg.smoke = 0;
    } else if (strcmp(argv[i], "--compiler-id") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.compiler_id = argv[++i];
    } else if (strcmp(argv[i], "--cflags") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.cflags = argv[++i];
    } else if (strcmp(argv[i], "--build-mode") == 0) {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.build_mode = argv[++i];
    } else if (strcmp(argv[i], "--git-commit") == 0) {
      if (i + 1 >= argc || argv[i + 1][0] == '\0') {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.git_commit = argv[++i];
    } else if (strcmp(argv[i], "--git-dirty") == 0) {
      if (i + 1 >= argc ||
          (strcmp(argv[i + 1], "0") != 0 && strcmp(argv[i + 1], "1") != 0)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
      }
      cfg.git_dirty = strcmp(argv[++i], "1") == 0 ? 1 : 0;
    } else {
      fprintf(stderr, "b0_wave5_bench: unknown argument '%s'\n", argv[i]);
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (seen_smoke && seen_full) {
    fprintf(stderr, "b0_wave5_bench: --smoke and --full are mutually exclusive\n");
    return EXIT_FAILURE;
  }
  if (cfg.output_path == NULL || cfg.output_path[0] == '\0') {
    fprintf(stderr, "b0_wave5_bench: missing required --output <path>\n");
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (strcmp(cfg.build_mode, "normal") != 0 &&
      strcmp(cfg.build_mode, "sanitize") != 0) {
    fprintf(stderr, "b0_wave5_bench: invalid --build-mode '%s'\n",
            cfg.build_mode);
    return EXIT_FAILURE;
  }
  if (strcmp(cfg.selected_group, "all") == 0) {
    cfg.selected_group_idx = -1;
  } else {
    cfg.selected_group_idx = find_group_index(cfg.selected_group);
    if (cfg.selected_group_idx < 0) {
      fprintf(stderr, "b0_wave5_bench: unknown --group '%s'\n",
              cfg.selected_group);
      return EXIT_FAILURE;
    }
  }

  /* Compute exact SHA-256 of all 11 linked Wave 5 archives + K1 archive. */
  char archive_shas[NUM_GROUPS][65];
  char k1_sha[65];
  char path_buf[1024];
  for (int g = 0; g < NUM_GROUPS; ++g) {
    const int n = snprintf(path_buf, sizeof(path_buf), "%s/%s", cfg.archive_dir,
                           g_group_descs[g].archive_relpath);
    REQUIRE(n > 0 && (size_t)n < sizeof(path_buf), "archive path overflow");
    sha256_file_hex(path_buf, archive_shas[g]);
  }
  {
    const int n = snprintf(path_buf, sizeof(path_buf),
                           "%s/k1/libeshkol_transformer_k1.a", cfg.archive_dir);
    REQUIRE(n > 0 && (size_t)n < sizeof(path_buf), "k1 archive path overflow");
    sha256_file_hex(path_buf, k1_sha);
  }

  /* Execute requested engine benchmarks. */
  size_t executed_groups = 0u;
  size_t total_cases = 0u;
  if (cfg.selected_group_idx >= 0) {
    run_group(cfg.selected_group_idx, cfg.smoke);
    executed_groups = 1u;
    total_cases = g_results[cfg.selected_group_idx].case_count;
  } else {
    for (int g = 0; g < NUM_GROUPS; ++g) {
      run_group(g, cfg.smoke);
      executed_groups += 1u;
      total_cases += g_results[g].case_count;
    }
  }

  struct rusage ru_self;
  REQUIRE(getrusage(RUSAGE_SELF, &ru_self) == 0, "getrusage self");
  REQUIRE(ru_self.ru_maxrss > 0L, "nonpositive self ru_maxrss");
  const uint64_t self_rss_kib = (uint64_t)ru_self.ru_maxrss;

  /* Build canonical body without artifact_sha256 and hash it. */
  strbuf body;
  sb_init(&body);
  build_canonical_body(&body, &cfg, self_rss_kib, archive_shas, k1_sha);
  REQUIRE(body.len >= 2u && body.data[0] == '{', "canonical body framing");

  char artifact_sha256[65];
  sha256_hex_buf(body.data, body.len, artifact_sha256);

  /* Insert artifact_sha256 in strict lexicographical top-level key order. */
  strbuf doc;
  sb_init(&doc);
  sb_append(&doc, "{\"artifact_sha256\":");
  sb_append_json_str(&doc, artifact_sha256);
  sb_append_char(&doc, ',');
  sb_append_mem(&doc, body.data + 1u, body.len - 1u);
  sb_append_char(&doc, '\n');

  write_file_atomic(cfg.output_path, doc.data, doc.len);

  printf("wave5-bench-v1 PASS: mode=%s smoke=%d group=%s groups=%zu cases=%zu "
         "self_rss_kib=%" PRIu64 " artifact_sha256=%s report=%s\n",
         cfg.build_mode, cfg.smoke, cfg.selected_group, executed_groups,
         total_cases, self_rss_kib, artifact_sha256, cfg.output_path);

  sb_free(&doc);
  sb_free(&body);
  return EXIT_SUCCESS;
}
