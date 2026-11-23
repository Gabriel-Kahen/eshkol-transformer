#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include "eshkol_transformer/dist5_collectives_abi.h"
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/mod5_primitives_abi.h"
#include "eshkol_transformer/mp4_mixed_precision_abi.h"
#include "eshkol_transformer/pa4_wider_attention_abi.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FP_CONTRACT OFF

_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && sizeof(float) == 4 &&
                   DBL_MANT_DIG == 53 && sizeof(double) == 8,
               "bench_wave45 requires IEEE binary32 and binary64");

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define REQUIRE(cond, msg)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "bench_wave45 FAIL (%s:%d): %s\n", __FILE__, __LINE__,   \
              (msg));                                                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/* ========================================================================= */
/* Timing & SHA-256 helpers                                                  */
/* ========================================================================= */

static uint64_t now_ns(void) {
  struct timespec ts;
  REQUIRE(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "clock_gettime monotonic");
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
  const uint64_t x = *(const uint64_t *)a;
  const uint64_t y = *(const uint64_t *)b;
  return (x > y) - (x < y);
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

/* ========================================================================= */
/* K1 helpers                                                                */
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
  const int32_t st = et_kernel_runtime_discover(resolve_provider, &ctx, &rt, &err);
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

static void check_finite_nonzero(const float *buf, size_t count,
                                 const char *label) {
  int nonzero = 0;
  for (size_t i = 0; i < count; ++i) {
    REQUIRE(isfinite(buf[i]), label);
    if (buf[i] != 0.0f)
      nonzero = 1;
  }
  REQUIRE(nonzero, label);
}

/* ========================================================================= */
/* Result storage structs (static/stack-friendly to keep LSan clean on fork) */
/* ========================================================================= */

typedef struct {
  const char *kernel;
  const char *operation;
  size_t rank;
  uint64_t shape[6];
  uint32_t warmup;
  uint32_t samples;
  uint32_t iterations;
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t max_ns;
  double median_ns_per_iter;
  char output_sha256[65];
} mod5_case_result;

typedef struct {
  const char *operation;
  uint64_t shape[6];
  uint32_t pairs;
  uint32_t warmup;
  uint32_t iterations;
  uint64_t ref_min_ns;
  uint64_t ref_median_ns;
  uint64_t ref_max_ns;
  double ref_median_ns_per_iter;
  uint64_t online_min_ns;
  uint64_t online_median_ns;
  uint64_t online_max_ns;
  double online_median_ns_per_iter;
  double online_vs_ref_ratio;
  double speedup_pct;
  double ab_ratio_min;
  double ab_ratio_max;
  double ba_ratio_min;
  double ba_ratio_max;
  double max_abs_diff;
  double max_rel_diff;
  char ref_output_sha256[65];
  char online_output_sha256[65];
} pa4_case_result;

typedef struct {
  const char *capability;
  const char *operation;
  const char *narrow_dtype;
  size_t rank;
  uint64_t shape[4];
  size_t element_count;
  size_t narrow_bytes;
  size_t wide_bytes;
  uint32_t warmup;
  uint32_t samples;
  uint32_t iterations;
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t max_ns;
  double median_ns_per_iter;
  double throughput_melem_per_s;
  char output_sha256[65];
} mp4_case_result;

typedef struct {
  uint32_t layers;
  const char *policy;
  uint32_t checkpoint_every;
  size_t planned_activation_peak_bytes;
  size_t live_activation_peak_bytes;
  size_t baseline_none_peak_bytes;
  size_t saved_activation_bytes;
  double memory_reduction_pct;
  size_t gradient_staging_bytes;
  size_t workspace_bytes;
  uint32_t retained_checkpoints;
  uint32_t retained_tapes;
  uint32_t rematerialized_layers;
  uint32_t warmup;
  uint32_t samples;
  uint32_t iterations;
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t max_ns;
  double median_ns_per_iter;
  double time_vs_none_ratio;
  char output_sha256[65];
} ac4_case_result;

typedef struct {
  const char *operation;
  uint32_t world_size;
  size_t element_count;
  size_t payload_bytes;
  uint32_t warmup;
  uint32_t samples;
  uint32_t iterations;
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t max_ns;
  double median_ns_per_iter;
  char output_sha256[65];
} dist5_case_result;

enum {
  MAX_MOD5_CASES = 48,
  MAX_PA4_CASES = 12,
  MAX_MP4_CASES = 16,
  MAX_AC4_CASES = 16,
  MAX_DIST5_CASES = 16,
  MAX_SAMPLES = 16
};

static mod5_case_result g_mod5_results[MAX_MOD5_CASES];
static size_t g_mod5_count = 0;

static pa4_case_result g_pa4_results[MAX_PA4_CASES];
static size_t g_pa4_count = 0;

static mp4_case_result g_mp4_results[MAX_MP4_CASES];
static size_t g_mp4_count = 0;
static int g_mp4_loss_scaler_verified = 0;

static ac4_case_result g_ac4_results[MAX_AC4_CASES];
static size_t g_ac4_count = 0;

static dist5_case_result g_dist5_results[MAX_DIST5_CASES];
static size_t g_dist5_count = 0;

/* ========================================================================= */
/* 1. MOD5 Kernel Benchmark                                                  */
/* ========================================================================= */

typedef struct {
  et_kernel_request_v1 fwd_req, bwd_req;
  et_kernel_call_v1 fwd, bwd;
  et_kernel_tensor_view_v1 in[9], out, grads[3];
  uint64_t shape[6], q_shape[4], kv_shape[4], qp_shape[2], kp_shape[2],
      mask_shape[3];
  float operands[3][1024], upstream[1024], slopes[4], epsilon;
  float result[1024], gradients[3][1024];
  int64_t query_pos[16], key_pos[16], window;
  uint8_t mask[256];
} mod5_fixture;

static void mod5_digest_outputs(const mod5_fixture *f, int mode,
                                char hex[65]) {
  sha256_ctx ctx;
  sha256_init(&ctx);
  if (mode == 0 || mode == 2) {
    check_finite_nonzero(f->result, f->out.byte_length / sizeof(float),
                         "mod5 forward output");
    sha256_update(&ctx, f->result, f->out.byte_length);
  }
  if (mode == 1 || mode == 2) {
    for (size_t g = 0; g < f->bwd.output_count; ++g) {
      check_finite_nonzero(f->gradients[g],
                           f->grads[g].byte_length / sizeof(float),
                           "mod5 backward grad");
      sha256_update(&ctx, f->gradients[g], f->grads[g].byte_length);
    }
  }
  sha256_final_hex(&ctx, hex);
}

static void mod5_time_op(const et_kernel_runtime *rt, mod5_fixture *f,
                         const char *kernel, size_t rank, const uint64_t *shape,
                         int mode, uint32_t warmup, uint32_t samples,
                         uint32_t iters) {
  static const char *const op_names[] = {"forward", "backward",
                                         "forward_backward"};
  REQUIRE(g_mod5_count < MAX_MOD5_CASES, "mod5 capacity");
  REQUIRE(samples <= MAX_SAMPLES, "mod5 samples cap");

  /* Untimed witness verification first */
  memset(f->result, 0, f->out.byte_length);
  for (size_t g = 0; g < f->bwd.output_count; ++g)
    memset(f->gradients[g], 0, f->grads[g].byte_length);
  if (mode == 0 || mode == 2)
    k1_dispatch(rt, &f->fwd);
  if (mode == 1 || mode == 2)
    k1_dispatch(rt, &f->bwd);
  char witness[65];
  mod5_digest_outputs(f, mode, witness);

  for (uint32_t w = 0; w < warmup; ++w) {
    if (mode == 0 || mode == 2)
      k1_dispatch(rt, &f->fwd);
    if (mode == 1 || mode == 2)
      k1_dispatch(rt, &f->bwd);
  }

  uint64_t times[MAX_SAMPLES];
  for (uint32_t s = 0; s < samples; ++s) {
    memset(f->result, 0, f->out.byte_length);
    for (size_t g = 0; g < f->bwd.output_count; ++g)
      memset(f->gradients[g], 0, f->grads[g].byte_length);
    const uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iters; ++i) {
      if (mode == 0 || mode == 2)
        k1_dispatch(rt, &f->fwd);
      if (mode == 1 || mode == 2)
        k1_dispatch(rt, &f->bwd);
    }
    const uint64_t elapsed = now_ns() - t0;
    REQUIRE(elapsed > 0u, "mod5 nonpositive elapsed");
    char check[65];
    mod5_digest_outputs(f, mode, check);
    REQUIRE(strcmp(check, witness) == 0, "mod5 witness drift");
    times[s] = elapsed;
  }

  qsort(times, samples, sizeof(times[0]), cmp_u64);
  mod5_case_result *r = &g_mod5_results[g_mod5_count++];
  memset(r, 0, sizeof(*r));
  r->kernel = kernel;
  r->operation = op_names[mode];
  r->rank = rank;
  memcpy(r->shape, shape, rank * sizeof(uint64_t));
  r->warmup = warmup;
  r->samples = samples;
  r->iterations = iters;
  r->min_ns = times[0];
  r->median_ns = times[(samples - 1u) / 2u];
  r->max_ns = times[samples - 1u];
  r->median_ns_per_iter = (double)r->median_ns / (double)iters;
  memcpy(r->output_sha256, witness, 65u);
}

static void mod5_seed(mod5_fixture *f) {
  for (size_t field = 0; field < f->bwd.output_count; ++field) {
    const size_t n = f->grads[field].byte_length / sizeof(float);
    for (size_t i = 0; i < n; ++i) {
      const size_t v = (i * (7u + 4u * field) + 5u * field + i / 3u) % 31u;
      f->operands[field][i] = ((float)v - 15.0f) * 0.0625f;
    }
  }
  const size_t out_n = f->out.byte_length / sizeof(float);
  for (size_t i = 0; i < out_n; ++i)
    f->upstream[i] = ((float)((i * 13u + i / 5u + 3u) % 23u) - 11.0f) * 0.0625f;
}

static void run_mod5_benchmarks(int smoke) {
  static mod5_fixture f;
  static const uint64_t rms_shapes[][3] = {{1, 2, 4}, {2, 8, 16}, {1, 16, 32}};
  static const uint64_t swi_shapes[][3] = {{1, 2, 4}, {2, 8, 32}, {1, 16, 64}};
  static const uint64_t attn_shapes[][6] = {{1, 4, 1, 8, 8, 4},
                                            {1, 4, 1, 16, 16, 8},
                                            {1, 4, 2, 8, 8, 4},
                                            {1, 4, 2, 16, 16, 8}};

  et_kernel_runtime *rt = discover_provider(et_mod5_kernel_provider_v1());
  const uint32_t warmup = smoke ? 2u : 10u;
  const uint32_t samples = smoke ? 3u : 7u;
  const uint32_t elem_iters = smoke ? 4u : 200u;
  const uint32_t attn_iters = smoke ? 4u : 80u;

  /* RMSNorm */
  for (size_t i = 0; i < ARRAY_COUNT(rms_shapes); ++i) {
    memset(&f, 0, sizeof(f));
    memcpy(f.shape, rms_shapes[i], 3u * sizeof(uint64_t));
    const size_t total = (size_t)(f.shape[0] * f.shape[1] * f.shape[2]);
    const size_t d = (size_t)f.shape[2];
    f.epsilon = 0.0625f;
    f.in[0] = make_view(f.operands[0], total * sizeof(float), "f32", 3u, f.shape);
    f.in[1] = make_view(f.operands[1], d * sizeof(float), "f32", 1u, f.shape + 2u);
    f.in[2] = make_view(&f.epsilon, sizeof(float), "f32", 0u, NULL);
    f.in[3] = make_view(f.upstream, total * sizeof(float), "f32", 3u, f.shape);
    f.out = make_view(f.result, total * sizeof(float), "f32", 3u, f.shape);
    f.grads[0] = make_view(f.gradients[0], total * sizeof(float), "f32", 3u, f.shape);
    f.grads[1] = make_view(f.gradients[1], d * sizeof(float), "f32", 1u, f.shape + 2u);
    f.fwd_req = (et_kernel_request_v1){sizeof(et_kernel_request_v1), "rmsnorm.forward",
                                       "f32", "cpu", 3u, f.shape, 1u, {0}};
    f.bwd_req = f.fwd_req;
    f.bwd_req.operation = "rmsnorm.backward";
    f.fwd = make_call("kernel.rmsnorm", &f.fwd_req, 3u, f.in, 1u, &f.out);
    f.bwd = make_call("kernel.rmsnorm", &f.bwd_req, 4u, f.in, 2u, f.grads);
    mod5_seed(&f);
    for (int mode = 0; mode <= 2; ++mode)
      mod5_time_op(rt, &f, "rmsnorm", 3u, f.shape, mode, warmup, samples, elem_iters);
  }

  /* SwiGLU */
  for (size_t i = 0; i < ARRAY_COUNT(swi_shapes); ++i) {
    memset(&f, 0, sizeof(f));
    memcpy(f.shape, swi_shapes[i], 3u * sizeof(uint64_t));
    const size_t total = (size_t)(f.shape[0] * f.shape[1] * f.shape[2]);
    f.in[0] = make_view(f.operands[0], total * sizeof(float), "f32", 3u, f.shape);
    f.in[1] = make_view(f.operands[1], total * sizeof(float), "f32", 3u, f.shape);
    f.in[2] = make_view(f.upstream, total * sizeof(float), "f32", 3u, f.shape);
    f.out = make_view(f.result, total * sizeof(float), "f32", 3u, f.shape);
    f.grads[0] = make_view(f.gradients[0], total * sizeof(float), "f32", 3u, f.shape);
    f.grads[1] = make_view(f.gradients[1], total * sizeof(float), "f32", 3u, f.shape);
    f.fwd_req = (et_kernel_request_v1){sizeof(et_kernel_request_v1), "swiglu.forward",
                                       "f32", "cpu", 3u, f.shape, 1u, {0}};
    f.bwd_req = f.fwd_req;
    f.bwd_req.operation = "swiglu.backward";
    f.fwd = make_call("kernel.swiglu", &f.fwd_req, 2u, f.in, 1u, &f.out);
    f.bwd = make_call("kernel.swiglu", &f.bwd_req, 3u, f.in, 2u, f.grads);
    mod5_seed(&f);
    for (int mode = 0; mode <= 2; ++mode)
      mod5_time_op(rt, &f, "swiglu", 3u, f.shape, mode, warmup, samples, elem_iters);
  }

  /* Variant attention (MQA and GQA) */
  for (size_t i = 0; i < ARRAY_COUNT(attn_shapes); ++i) {
    memset(&f, 0, sizeof(f));
    memcpy(f.shape, attn_shapes[i], 6u * sizeof(uint64_t));
    const uint64_t n = f.shape[0], hq = f.shape[1], hkv = f.shape[2];
    const uint64_t tq = f.shape[3], tk = f.shape[4], dh = f.shape[5];
    f.q_shape[0] = n;
    f.q_shape[1] = hq;
    f.q_shape[2] = tq;
    f.q_shape[3] = dh;
    f.kv_shape[0] = n;
    f.kv_shape[1] = hkv;
    f.kv_shape[2] = tk;
    f.kv_shape[3] = dh;
    f.qp_shape[0] = n;
    f.qp_shape[1] = tq;
    f.kp_shape[0] = n;
    f.kp_shape[1] = tk;
    f.mask_shape[0] = n;
    f.mask_shape[1] = tq;
    f.mask_shape[2] = tk;
    f.window = 8;
    for (size_t b = 0; b < n; ++b) {
      for (size_t t = 0; t < tq; ++t) {
        f.query_pos[b * tq + t] = (int64_t)(t * 2u + b);
        for (size_t k = 0; k < tk; ++k)
          f.mask[(b * tq + t) * tk + k] = (uint8_t)((b + t + k) % 5u != 2u);
      }
      for (size_t k = 0; k < tk; ++k)
        f.key_pos[b * tk + k] = (int64_t)(k * 2u + b);
    }
    for (size_t h = 0; h < hq; ++h)
      f.slopes[h] = ((float)h - 1.25f) * 0.0625f;
    const size_t q_bytes = (size_t)(n * hq * tq * dh) * sizeof(float);
    const size_t kv_bytes = (size_t)(n * hkv * tk * dh) * sizeof(float);
    f.in[0] = make_view(f.operands[0], q_bytes, "f32", 4u, f.q_shape);
    f.in[1] = make_view(f.operands[1], kv_bytes, "f32", 4u, f.kv_shape);
    f.in[2] = make_view(f.operands[2], kv_bytes, "f32", 4u, f.kv_shape);
    f.in[3] = make_view(f.query_pos, (size_t)(n * tq) * sizeof(int64_t), "i64",
                        2u, f.qp_shape);
    f.in[4] = make_view(f.key_pos, (size_t)(n * tk) * sizeof(int64_t), "i64",
                        2u, f.kp_shape);
    f.in[5] = make_view(f.mask, (size_t)(n * tq * tk), "bool", 3u, f.mask_shape);
    f.in[6] = make_view(f.slopes, (size_t)hq * sizeof(float), "f32", 1u, f.shape + 1u);
    f.in[7] = make_view(&f.window, sizeof(int64_t), "i64", 0u, NULL);
    f.in[8] = make_view(f.upstream, q_bytes, "f32", 4u, f.q_shape);
    f.out = make_view(f.result, q_bytes, "f32", 4u, f.q_shape);
    f.grads[0] = make_view(f.gradients[0], q_bytes, "f32", 4u, f.q_shape);
    f.grads[1] = make_view(f.gradients[1], kv_bytes, "f32", 4u, f.kv_shape);
    f.grads[2] = make_view(f.gradients[2], kv_bytes, "f32", 4u, f.kv_shape);
    f.fwd_req = (et_kernel_request_v1){sizeof(et_kernel_request_v1),
                                       "variant-attention.forward",
                                       "f32", "cpu", 6u, f.shape, 1u, {0}};
    f.bwd_req = f.fwd_req;
    f.bwd_req.operation = "variant-attention.backward";
    f.fwd = make_call("kernel.variant-attention", &f.fwd_req, 8u, f.in, 1u, &f.out);
    f.bwd = make_call("kernel.variant-attention", &f.bwd_req, 9u, f.in, 3u, f.grads);
    mod5_seed(&f);
    const char *kname = hkv == 1u ? "variant_attention_mqa" : "variant_attention_gqa";
    for (int mode = 0; mode <= 2; ++mode)
      mod5_time_op(rt, &f, kname, 6u, f.shape, mode, warmup, samples, attn_iters);
  }

  et_kernel_runtime_destroy(rt);
}

/* ========================================================================= */
/* 2. PA4-Wider AB/BA Benchmark                                              */
/* ========================================================================= */

enum { PA4_MAX_TOKENS = 64, PA4_MAX_ELEMS = 2 * 4 * PA4_MAX_TOKENS * 16 };

typedef struct {
  uint64_t shape[6], q_shape[4], kv_shape[4], qp_shape[2], kp_shape[2],
      mask_shape[3];
  size_t q_count, kv_count;
  float q[PA4_MAX_ELEMS], k[PA4_MAX_ELEMS], v[PA4_MAX_ELEMS],
      upstream[PA4_MAX_ELEMS];
  int64_t qp[2 * PA4_MAX_TOKENS], kp[2 * PA4_MAX_TOKENS];
  uint8_t mask[2 * PA4_MAX_TOKENS * PA4_MAX_TOKENS];
  float result[4][PA4_MAX_ELEMS];
  et_kernel_tensor_view_v1 in[7], out[4];
  et_kernel_request_v1 req[2];
  et_kernel_call_v1 call[2];
} pa4_fixture;

static float pa4_sample_val(size_t index, uint32_t seed) {
  uint32_t bits = ((uint32_t)index + 1u) * UINT32_C(0x9e3779b1) + seed;
  bits = (bits ^ (bits >> 16u)) * UINT32_C(0x85ebca6b);
  bits = (bits ^ (bits >> 13u)) * UINT32_C(0xc2b2ae35);
  return (float)((int32_t)((bits ^ (bits >> 16u)) % 257u) - 128) / 128.0f;
}

static void pa4_init(pa4_fixture *f, const uint64_t shape[6]) {
  memset(f, 0, sizeof(*f));
  memcpy(f->shape, shape, sizeof(f->shape));
  const size_t n = (size_t)shape[0], hq = (size_t)shape[1],
               hkv = (size_t)shape[2], tq = (size_t)shape[3],
               tk = (size_t)shape[4], dh = (size_t)shape[5];
  f->q_shape[0] = n;
  f->q_shape[1] = hq;
  f->q_shape[2] = tq;
  f->q_shape[3] = dh;
  f->kv_shape[0] = n;
  f->kv_shape[1] = hkv;
  f->kv_shape[2] = tk;
  f->kv_shape[3] = dh;
  f->qp_shape[0] = n;
  f->qp_shape[1] = tq;
  f->kp_shape[0] = n;
  f->kp_shape[1] = tk;
  f->mask_shape[0] = n;
  f->mask_shape[1] = tq;
  f->mask_shape[2] = tk;
  f->q_count = n * hq * tq * dh;
  f->kv_count = n * hkv * tk * dh;
  for (size_t i = 0; i < f->q_count; ++i) {
    f->q[i] = pa4_sample_val(i, 17u);
    f->upstream[i] = pa4_sample_val(i, 29u);
  }
  for (size_t i = 0; i < f->kv_count; ++i) {
    f->k[i] = pa4_sample_val(i, 37u);
    f->v[i] = pa4_sample_val(i, 53u);
  }
  for (size_t b = 0; b < n; ++b) {
    for (size_t i = 0; i < tq; ++i)
      f->qp[b * tq + i] = (int64_t)(67u * b + 3u * i + 1u);
    for (size_t j = 0; j < tk; ++j)
      f->kp[b * tk + j] = (int64_t)(67u * b + 3u * j + 1u);
    for (size_t i = 0; i < tq; ++i)
      for (size_t j = 0; j < tk; ++j)
        f->mask[(b * tq + i) * tk + j] =
            (uint8_t)(i != 3u && (i + 2u * j + 3u * b) % 11u != 7u);
  }
  const size_t q_bytes = f->q_count * sizeof(float);
  const size_t kv_bytes = f->kv_count * sizeof(float);
  f->in[0] = make_view(f->q, q_bytes, "f32", 4u, f->q_shape);
  f->in[1] = make_view(f->k, kv_bytes, "f32", 4u, f->kv_shape);
  f->in[2] = make_view(f->v, kv_bytes, "f32", 4u, f->kv_shape);
  f->in[3] = make_view(f->qp, n * tq * sizeof(int64_t), "i64", 2u, f->qp_shape);
  f->in[4] = make_view(f->kp, n * tk * sizeof(int64_t), "i64", 2u, f->kp_shape);
  f->in[5] = make_view(f->mask, n * tq * tk, "bool", 3u, f->mask_shape);
  f->in[6] = make_view(f->upstream, q_bytes, "f32", 4u, f->q_shape);
  f->out[0] = make_view(f->result[0], q_bytes, "f32", 4u, f->q_shape);
  f->out[1] = make_view(f->result[1], q_bytes, "f32", 4u, f->q_shape);
  f->out[2] = make_view(f->result[2], kv_bytes, "f32", 4u, f->kv_shape);
  f->out[3] = make_view(f->result[3], kv_bytes, "f32", 4u, f->kv_shape);
  for (int op = 0; op <= 1; ++op) {
    f->req[op] = (et_kernel_request_v1){
        .struct_size = sizeof(et_kernel_request_v1),
        .operation =
            op == 0 ? "causal-attention.forward" : "causal-attention.backward",
        .dtype = "f32",
        .device = "cpu",
        .rank = 6u,
        .shape = f->shape,
        .deterministic = 1u};
    f->call[op] = make_call("kernel.causal-attention", &f->req[op],
                            op == 0 ? 6u : 7u, f->in, op == 0 ? 1u : 3u,
                            f->out + (op == 0 ? 0u : 1u));
  }
}

static void pa4_digest(const pa4_fixture *f, int op, char hex[65]) {
  const int first = op == 1 ? 1 : 0;
  const int last = op == 0 ? 0 : 3;
  sha256_ctx ctx;
  sha256_init(&ctx);
  for (int j = first; j <= last; ++j)
    sha256_update(&ctx, f->result[j], f->out[j].byte_length);
  sha256_final_hex(&ctx, hex);
}

static void run_pa4_wider_benchmarks(int smoke) {
  static pa4_fixture f;
  static float witness[2][4][PA4_MAX_ELEMS];
  static const uint64_t shapes[][6] = {{1, 4, 2, 16, 16, 16},
                                       {1, 4, 2, 32, 32, 16},
                                       {1, 4, 2, 64, 64, 16}};
  static const char *const op_names[] = {"forward", "backward", "joint"};

  et_kernel_runtime *rts[2] = {
      discover_provider(et_pa4_wider_ref_provider_v1()),
      discover_provider(et_pa4_wider_online_provider_v1())};

  const uint32_t pairs = smoke ? 2u : 8u;
  const uint32_t warmup = smoke ? 1u : 4u;
  REQUIRE(pairs <= MAX_SAMPLES, "pa4 pairs cap");

  for (size_t si = 0; si < ARRAY_COUNT(shapes); ++si) {
    pa4_init(&f, shapes[si]);
    const uint32_t iters = smoke ? 2u : (si == 0u ? 30u : (si == 1u ? 20u : 10u));

    /* Pre-verify outputs & parity BEFORE any timing! */
    double op_max_abs[3] = {0.0, 0.0, 0.0};
    double op_max_rel[3] = {0.0, 0.0, 0.0};
    char witness_hex[2][3][65];

    for (int p = 0; p <= 1; ++p) {
      for (int j = 0; j < 4; ++j)
        memset(f.result[j], 0, f.out[j].byte_length);
      k1_dispatch(rts[p], &f.call[0]);
      k1_dispatch(rts[p], &f.call[1]);
      for (int j = 0; j < 4; ++j) {
        const size_t cnt = f.out[j].byte_length / sizeof(float);
        check_finite_nonzero(f.result[j], cnt, "pa4 witness");
        memcpy(witness[p][j], f.result[j], f.out[j].byte_length);
      }
      for (int op = 0; op <= 2; ++op)
        pa4_digest(&f, op, witness_hex[p][op]);
    }

    for (int j = 0; j < 4; ++j) {
      const size_t cnt = f.out[j].byte_length / sizeof(float);
      for (size_t i = 0; i < cnt; ++i) {
        const double a = (double)witness[0][j][i];
        const double b = (double)witness[1][j][i];
        const double abs_diff = fabs(a - b);
        const double scale = fmax(fabs(a), fabs(b));
        const double rel_diff = scale != 0.0 ? abs_diff / scale : 0.0;
        REQUIRE(abs_diff <= 3e-7 + 3e-6 * scale, "pa4 ref vs online parity");
        const int bucket = j == 0 ? 0 : 1;
        if (abs_diff > op_max_abs[bucket])
          op_max_abs[bucket] = abs_diff;
        if (rel_diff > op_max_rel[bucket])
          op_max_rel[bucket] = rel_diff;
        if (abs_diff > op_max_abs[2])
          op_max_abs[2] = abs_diff;
        if (rel_diff > op_max_rel[2])
          op_max_rel[2] = rel_diff;
      }
    }

    for (int op = 0; op <= 2; ++op) {
      REQUIRE(g_pa4_count < MAX_PA4_CASES, "pa4 capacity");
      uint64_t times[2][MAX_SAMPLES];
      double min_ratio[2] = {INFINITY, INFINITY};
      double max_ratio[2] = {0.0, 0.0};

      for (uint32_t pair = 0; pair < pairs; ++pair) {
        const int first = (pair % 2u == 0u) ? 0 : 1;
        const int second = 1 - first;
        const int order[2] = {first, second};
        for (int arm_idx = 0; arm_idx < 2; ++arm_idx) {
          const int arm = order[arm_idx];
          for (uint32_t w = 0; w < warmup; ++w) {
            if (op == 0 || op == 2)
              k1_dispatch(rts[arm], &f.call[0]);
            if (op == 1 || op == 2)
              k1_dispatch(rts[arm], &f.call[1]);
          }
          for (int j = 0; j < 4; ++j)
            memset(f.result[j], 0, f.out[j].byte_length);
          const uint64_t t0 = now_ns();
          for (uint32_t it = 0; it < iters; ++it) {
            if (op == 0 || op == 2)
              k1_dispatch(rts[arm], &f.call[0]);
            if (op == 1 || op == 2)
              k1_dispatch(rts[arm], &f.call[1]);
          }
          const uint64_t elapsed = now_ns() - t0;
          REQUIRE(elapsed > 0u, "pa4 nonpositive elapsed");
          char dig[65];
          pa4_digest(&f, op, dig);
          REQUIRE(strcmp(dig, witness_hex[arm][op]) == 0,
                  "pa4 timed output digest mismatch");
          times[arm][pair] = elapsed;
        }
        const double ratio =
            (double)times[1][pair] / (double)times[0][pair];
        const size_t ord_slot = pair % 2u;
        if (ratio < min_ratio[ord_slot])
          min_ratio[ord_slot] = ratio;
        if (ratio > max_ratio[ord_slot])
          max_ratio[ord_slot] = ratio;
      }

      qsort(times[0], pairs, sizeof(times[0][0]), cmp_u64);
      qsort(times[1], pairs, sizeof(times[1][0]), cmp_u64);
      pa4_case_result *r = &g_pa4_results[g_pa4_count++];
      memset(r, 0, sizeof(*r));
      r->operation = op_names[op];
      memcpy(r->shape, shapes[si], sizeof(r->shape));
      r->pairs = pairs;
      r->warmup = warmup;
      r->iterations = iters;
      r->ref_min_ns = times[0][0];
      r->ref_median_ns = times[0][(pairs - 1u) / 2u];
      r->ref_max_ns = times[0][pairs - 1u];
      r->ref_median_ns_per_iter = (double)r->ref_median_ns / (double)iters;
      r->online_min_ns = times[1][0];
      r->online_median_ns = times[1][(pairs - 1u) / 2u];
      r->online_max_ns = times[1][pairs - 1u];
      r->online_median_ns_per_iter =
          (double)r->online_median_ns / (double)iters;
      r->online_vs_ref_ratio =
          (double)r->online_median_ns / (double)r->ref_median_ns;
      r->speedup_pct = 100.0 *
                       ((double)r->ref_median_ns - (double)r->online_median_ns) /
                       (double)r->ref_median_ns;
      r->ab_ratio_min = min_ratio[0];
      r->ab_ratio_max = max_ratio[0];
      r->ba_ratio_min = min_ratio[1];
      r->ba_ratio_max = max_ratio[1];
      r->max_abs_diff = op_max_abs[op];
      r->max_rel_diff = op_max_rel[op];
      memcpy(r->ref_output_sha256, witness_hex[0][op], 65u);
      memcpy(r->online_output_sha256, witness_hex[1][op], 65u);
    }
  }

  et_kernel_runtime_destroy(rts[0]);
  et_kernel_runtime_destroy(rts[1]);
}

/* ========================================================================= */
/* 3. MP4 Mixed-Precision Cast & Unscale Benchmark                           */
/* ========================================================================= */

static void run_mp4_benchmarks(int smoke) {
  typedef struct {
    const char *dtype;
    const char *cap_cast;
    const char *op_narrow;
    const char *op_widen;
    const char *op_unscale;
  } mp4_fmt;

  static const mp4_fmt fmts[] = {
      {"f16", "kernel.cast-f16", "cast.f32-to-f16", "cast.f16-to-f32",
       "grad.unscale-f16-to-f32"},
      {"bf16", "kernel.cast-bf16", "cast.f32-to-bf16", "cast.bf16-to-f32",
       "grad.unscale-bf16-to-f32"}};

  typedef struct {
    size_t rank;
    uint64_t dims[3];
    size_t count;
  } mp4_shape;

  static const mp4_shape shapes[] = {{3u, {2, 8, 16}, 256u},
                                     {2u, {256, 4, 0}, 1024u}};

  et_kernel_runtime *rt = discover_provider(et_mp4_kernel_provider_v1());

  /* Verify loss scaler state machine as part of MP4 validation */
  et_mp4_loss_scaler_v1 scaler;
  REQUIRE(et_mp4_loss_scaler_init(&scaler, 64.0f, 2.0f, 0.5f, 2u, 1.0f,
                                  1024.0f) == ET_MP4_LOSS_SCALER_STEP_ACCEPTED,
          "mp4 scaler init");
  REQUIRE(et_mp4_loss_scaler_update(&scaler, 0u) ==
              ET_MP4_LOSS_SCALER_STEP_ACCEPTED &&
              scaler.clean_steps == 1u && scaler.scale == 64.0f,
          "mp4 scaler step 1");
  REQUIRE(et_mp4_loss_scaler_update(&scaler, 0u) ==
              ET_MP4_LOSS_SCALER_STEP_ACCEPTED &&
              scaler.clean_steps == 0u && scaler.scale == 128.0f,
          "mp4 scaler growth");
  REQUIRE(et_mp4_loss_scaler_update(&scaler, 1u) ==
              ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF &&
              scaler.clean_steps == 0u && scaler.scale == 64.0f,
          "mp4 scaler backoff");
  g_mp4_loss_scaler_verified = 1;

  const uint32_t warmup = smoke ? 2u : 15u;
  const uint32_t samples = smoke ? 3u : 7u;
  const uint32_t iters = smoke ? 5u : 500u;

  float src_f32[1024];
  float dst_f32[1024];
  uint16_t narrow_buf[1024];
  for (size_t i = 0; i < 1024u; ++i)
    src_f32[i] = ((float)((i * 19u + 7u) % 257u) - 128.0f) * 0.125f;

  for (size_t si = 0; si < ARRAY_COUNT(shapes); ++si) {
    const mp4_shape *sh = &shapes[si];
    const size_t n = sh->count;
    const size_t narrow_bytes = n * 2u;
    const size_t wide_bytes = n * 4u;
    REQUIRE(wide_bytes == 2u * narrow_bytes, "mp4 2:1 byte ratio");

    for (size_t fi = 0; fi < ARRAY_COUNT(fmts); ++fi) {
      const mp4_fmt *fmt = &fmts[fi];

      /* 1) Narrow: f32 -> narrow */
      {
        et_kernel_request_v1 req = {sizeof(et_kernel_request_v1),
                                    fmt->op_narrow, "f32", "cpu", sh->rank,
                                    sh->dims, 1u, {0}};
        et_kernel_tensor_view_v1 in =
            make_view(src_f32, wide_bytes, "f32", sh->rank, sh->dims);
        et_kernel_tensor_view_v1 out =
            make_view(narrow_buf, narrow_bytes, fmt->dtype, sh->rank, sh->dims);
        et_kernel_call_v1 call = make_call(fmt->cap_cast, &req, 1u, &in, 1u, &out);

        memset(narrow_buf, 0, narrow_bytes);
        k1_dispatch(rt, &call);
        char witness[65];
        sha256_hex_buf(narrow_buf, narrow_bytes, witness);

        for (uint32_t w = 0; w < warmup; ++w)
          k1_dispatch(rt, &call);
        uint64_t times[MAX_SAMPLES];
        for (uint32_t s = 0; s < samples; ++s) {
          memset(narrow_buf, 0, narrow_bytes);
          const uint64_t t0 = now_ns();
          for (uint32_t it = 0; it < iters; ++it)
            k1_dispatch(rt, &call);
          const uint64_t elapsed = now_ns() - t0;
          REQUIRE(elapsed > 0u, "mp4 narrow elapsed");
          char dig[65];
          sha256_hex_buf(narrow_buf, narrow_bytes, dig);
          REQUIRE(strcmp(dig, witness) == 0, "mp4 narrow digest");
          times[s] = elapsed;
        }
        qsort(times, samples, sizeof(times[0]), cmp_u64);
        mp4_case_result *r = &g_mp4_results[g_mp4_count++];
        memset(r, 0, sizeof(*r));
        r->capability = fmt->cap_cast;
        r->operation = fmt->op_narrow;
        r->narrow_dtype = fmt->dtype;
        r->rank = sh->rank;
        memcpy(r->shape, sh->dims, sh->rank * sizeof(uint64_t));
        r->element_count = n;
        r->narrow_bytes = narrow_bytes;
        r->wide_bytes = wide_bytes;
        r->warmup = warmup;
        r->samples = samples;
        r->iterations = iters;
        r->min_ns = times[0];
        r->median_ns = times[(samples - 1u) / 2u];
        r->max_ns = times[samples - 1u];
        r->median_ns_per_iter = (double)r->median_ns / (double)iters;
        r->throughput_melem_per_s =
            ((double)n * (double)iters * 1000.0) / (double)r->median_ns;
        memcpy(r->output_sha256, witness, 65u);
      }

      /* 2) Widen: narrow -> f32 (using the verified narrow_buf from above) */
      {
        et_kernel_request_v1 req = {sizeof(et_kernel_request_v1),
                                    fmt->op_widen, "f32", "cpu", sh->rank,
                                    sh->dims, 1u, {0}};
        et_kernel_tensor_view_v1 in =
            make_view(narrow_buf, narrow_bytes, fmt->dtype, sh->rank, sh->dims);
        et_kernel_tensor_view_v1 out =
            make_view(dst_f32, wide_bytes, "f32", sh->rank, sh->dims);
        et_kernel_call_v1 call = make_call(fmt->cap_cast, &req, 1u, &in, 1u, &out);

        memset(dst_f32, 0, wide_bytes);
        k1_dispatch(rt, &call);
        check_finite_nonzero(dst_f32, n, "mp4 widen output");
        char witness[65];
        sha256_hex_buf(dst_f32, wide_bytes, witness);

        for (uint32_t w = 0; w < warmup; ++w)
          k1_dispatch(rt, &call);
        uint64_t times[MAX_SAMPLES];
        for (uint32_t s = 0; s < samples; ++s) {
          memset(dst_f32, 0, wide_bytes);
          const uint64_t t0 = now_ns();
          for (uint32_t it = 0; it < iters; ++it)
            k1_dispatch(rt, &call);
          const uint64_t elapsed = now_ns() - t0;
          REQUIRE(elapsed > 0u, "mp4 widen elapsed");
          char dig[65];
          sha256_hex_buf(dst_f32, wide_bytes, dig);
          REQUIRE(strcmp(dig, witness) == 0, "mp4 widen digest");
          times[s] = elapsed;
        }
        qsort(times, samples, sizeof(times[0]), cmp_u64);
        mp4_case_result *r = &g_mp4_results[g_mp4_count++];
        memset(r, 0, sizeof(*r));
        r->capability = fmt->cap_cast;
        r->operation = fmt->op_widen;
        r->narrow_dtype = fmt->dtype;
        r->rank = sh->rank;
        memcpy(r->shape, sh->dims, sh->rank * sizeof(uint64_t));
        r->element_count = n;
        r->narrow_bytes = narrow_bytes;
        r->wide_bytes = wide_bytes;
        r->warmup = warmup;
        r->samples = samples;
        r->iterations = iters;
        r->min_ns = times[0];
        r->median_ns = times[(samples - 1u) / 2u];
        r->max_ns = times[samples - 1u];
        r->median_ns_per_iter = (double)r->median_ns / (double)iters;
        r->throughput_melem_per_s =
            ((double)n * (double)iters * 1000.0) / (double)r->median_ns;
        memcpy(r->output_sha256, witness, 65u);
      }

      /* 3) Gradient unscale: narrow + inv_scale -> f32 + overflow */
      {
        float inv_scale = 0.015625f; /* 1/64 */
        uint8_t overflow = 0xffu;
        et_kernel_request_v1 req = {sizeof(et_kernel_request_v1),
                                    fmt->op_unscale, "f32", "cpu", sh->rank,
                                    sh->dims, 1u, {0}};
        et_kernel_tensor_view_v1 in[2] = {
            make_view(narrow_buf, narrow_bytes, fmt->dtype, sh->rank, sh->dims),
            make_view(&inv_scale, sizeof(float), "f32", 0u, NULL)};
        et_kernel_tensor_view_v1 out[2] = {
            make_view(dst_f32, wide_bytes, "f32", sh->rank, sh->dims),
            make_view(&overflow, 1u, "bool", 0u, NULL)};
        et_kernel_call_v1 call =
            make_call("kernel.grad-unscale", &req, 2u, in, 2u, out);

        memset(dst_f32, 0, wide_bytes);
        overflow = 0xffu;
        k1_dispatch(rt, &call);
        REQUIRE(overflow == 0u, "mp4 unscale clean overflow");
        check_finite_nonzero(dst_f32, n, "mp4 unscale output");
        char witness[65];
        sha256_hex_buf(dst_f32, wide_bytes, witness);

        for (uint32_t w = 0; w < warmup; ++w)
          k1_dispatch(rt, &call);
        uint64_t times[MAX_SAMPLES];
        for (uint32_t s = 0; s < samples; ++s) {
          memset(dst_f32, 0, wide_bytes);
          overflow = 0xffu;
          const uint64_t t0 = now_ns();
          for (uint32_t it = 0; it < iters; ++it)
            k1_dispatch(rt, &call);
          const uint64_t elapsed = now_ns() - t0;
          REQUIRE(elapsed > 0u, "mp4 unscale elapsed");
          REQUIRE(overflow == 0u, "mp4 unscale timed overflow");
          char dig[65];
          sha256_hex_buf(dst_f32, wide_bytes, dig);
          REQUIRE(strcmp(dig, witness) == 0, "mp4 unscale digest");
          times[s] = elapsed;
        }
        qsort(times, samples, sizeof(times[0]), cmp_u64);
        mp4_case_result *r = &g_mp4_results[g_mp4_count++];
        memset(r, 0, sizeof(*r));
        r->capability = "kernel.grad-unscale";
        r->operation = fmt->op_unscale;
        r->narrow_dtype = fmt->dtype;
        r->rank = sh->rank;
        memcpy(r->shape, sh->dims, sh->rank * sizeof(uint64_t));
        r->element_count = n;
        r->narrow_bytes = narrow_bytes;
        r->wide_bytes = wide_bytes;
        r->warmup = warmup;
        r->samples = samples;
        r->iterations = iters;
        r->min_ns = times[0];
        r->median_ns = times[(samples - 1u) / 2u];
        r->max_ns = times[samples - 1u];
        r->median_ns_per_iter = (double)r->median_ns / (double)iters;
        r->throughput_melem_per_s =
            ((double)n * (double)iters * 1000.0) / (double)r->median_ns;
        memcpy(r->output_sha256, witness, 65u);
      }
    }
  }

  et_kernel_runtime_destroy(rt);
}

/* ========================================================================= */
/* 4. AC4 Activation Checkpointing Benchmark                                 */
/* ========================================================================= */

static float ac4_lcg(uint32_t *state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return (float)(*state >> 16) * 0x1p-15f - 1.0f;
}

static void run_ac4_benchmarks(int smoke) {
  enum { AC4_MAX_X = 48, AC4_MAX_P = ET_AC4_MAX_LAYERS * 464 };
  static const uint32_t depths[] = {4u, 8u, 16u};
  static const et_ac4_policy policies[] = {
      ET_AC4_POLICY_NONE, ET_AC4_POLICY_EVERY_LAYER,
      ET_AC4_POLICY_SQRT_SEGMENTS, ET_AC4_POLICY_EVERY_K};
  static const char *const policy_names[] = {"NONE", "EVERY_LAYER",
                                             "SQRT_SEGMENTS", "EVERY_K"};

  static float input[AC4_MAX_X], target[AC4_MAX_X], params[AC4_MAX_P];
  static float output[4][AC4_MAX_X], dx[4][AC4_MAX_X], dparams[4][AC4_MAX_P];

  const uint32_t warmup = smoke ? 1u : 6u;
  const uint32_t samples = smoke ? 3u : 7u;
  const uint32_t iters = smoke ? 2u : 60u;

  for (size_t di = 0; di < ARRAY_COUNT(depths); ++di) {
    const et_ac4_shape_v1 shape = {depths[di], 2u, 3u, 4u, 2u, 2u};
    const size_t width = (size_t)shape.query_heads * shape.head_dim;
    const size_t kv_width = (size_t)shape.kv_heads * shape.head_dim;
    const size_t gamma2 = width + 2u * width * width + 2u * kv_width * width;
    const size_t per_layer = gamma2 + width + 3u * width * width;
    const size_t x_cnt = (size_t)shape.batch * shape.tokens * width;
    const size_t p_cnt = (size_t)shape.layers * per_layer;
    REQUIRE(x_cnt <= AC4_MAX_X && p_cnt <= AC4_MAX_P, "ac4 buffer bounds");

    uint32_t st = 41u + depths[di];
    for (size_t i = 0; i < x_cnt; ++i) {
      input[i] = 0.65f * ac4_lcg(&st);
      target[i] = 0.62f * input[i] + 0.27f * ac4_lcg(&st);
    }
    for (size_t i = 0; i < p_cnt; ++i)
      params[i] = 0.28f * ac4_lcg(&st);
    for (uint32_t l = 0; l < shape.layers; ++l) {
      float *lp = params + l * per_layer;
      for (size_t i = 0; i < width; ++i) {
        lp[i] = 0.92f + 0.16f * ac4_lcg(&st);
        lp[gamma2 + i] = 0.87f + 0.16f * ac4_lcg(&st);
      }
    }

    et_ac4_config_v1 cfgs[4];
    et_ac4_memory_v1 plans[4];
    et_ac4_result_v1 results[4];
    et_ac4_io_v1 ios[4];
    float *workspaces[4] = {NULL, NULL, NULL, NULL};
    uint64_t times[4][MAX_SAMPLES];
    char digests[4][65];

    for (size_t p = 0; p < 4u; ++p) {
      cfgs[p] = (et_ac4_config_v1){.shape = shape,
                                   .mode = ET_AC4_FORWARD_BACKWARD,
                                   .policy = policies[p],
                                   .checkpoint_every = p == 3u ? shape.layers : 0u,
                                   .epsilon = 0.125f,
                                   .dropout_rate = 0.2f,
                                   .seed = UINT64_C(0x570efdb000311fb0),
                                   .step = 0u,
                                   .activation_limit_bytes = 0u};
      REQUIRE(et_ac4_plan_v1(&cfgs[p], &plans[p]) == ET_AC4_OK, "ac4 plan");
      cfgs[p].activation_limit_bytes = plans[p].activation_peak_bytes;
      workspaces[p] = (float *)malloc(plans[p].workspace_bytes);
      REQUIRE(workspaces[p] != NULL, "ac4 workspace malloc");
      ios[p] = (et_ac4_io_v1){.input = {input, x_cnt},
                              .target = {target, x_cnt},
                              .parameters = {params, p_cnt},
                              .output = {output[p], x_cnt},
                              .d_input = {dx[p], x_cnt},
                              .d_parameters = {dparams[p], p_cnt},
                              .workspace = workspaces[p],
                              .workspace_bytes = plans[p].workspace_bytes};

      /* Untimed verification + parity check */
      REQUIRE(et_ac4_run_v1(&cfgs[p], &ios[p], &results[p]) == ET_AC4_OK,
              "ac4 untimed run");
      REQUIRE(isfinite(results[p].loss) && results[p].loss > 0.0, "ac4 loss");
      REQUIRE(results[p].memory.activation_peak_bytes ==
                  plans[p].activation_peak_bytes,
              "ac4 live peak matches plan");
      check_finite_nonzero(output[p], x_cnt, "ac4 output");
      check_finite_nonzero(dx[p], x_cnt, "ac4 d_input");
      check_finite_nonzero(dparams[p], p_cnt, "ac4 d_params");

      sha256_ctx ctx;
      sha256_init(&ctx);
      sha256_update(&ctx, &results[p].loss, sizeof(double));
      sha256_update(&ctx, output[p], x_cnt * sizeof(float));
      sha256_update(&ctx, dx[p], x_cnt * sizeof(float));
      sha256_update(&ctx, dparams[p], p_cnt * sizeof(float));
      sha256_final_hex(&ctx, digests[p]);

      if (p > 0u) {
        REQUIRE(memcmp(&results[0].loss, &results[p].loss, sizeof(double)) == 0,
                "ac4 loss parity");
        REQUIRE(memcmp(output[0], output[p], x_cnt * sizeof(float)) == 0,
                "ac4 output parity");
        REQUIRE(memcmp(dx[0], dx[p], x_cnt * sizeof(float)) == 0,
                "ac4 dx parity");
        REQUIRE(memcmp(dparams[0], dparams[p], p_cnt * sizeof(float)) == 0,
                "ac4 dparams parity");
        REQUIRE(strcmp(digests[0], digests[p]) == 0, "ac4 digest parity");
      }

      for (uint32_t w = 0; w < warmup; ++w)
        REQUIRE(et_ac4_run_v1(&cfgs[p], &ios[p], &results[p]) == ET_AC4_OK,
                "ac4 warmup");
    }

    /* Interleaved timed samples across policies */
    for (uint32_t s = 0; s < samples; ++s) {
      for (size_t j = 0; j < 4u; ++j) {
        const size_t p = (j + s) % 4u;
        const uint64_t t0 = now_ns();
        for (uint32_t it = 0; it < iters; ++it)
          REQUIRE(et_ac4_run_v1(&cfgs[p], &ios[p], &results[p]) == ET_AC4_OK,
                  "ac4 timed run");
        const uint64_t elapsed = now_ns() - t0;
        REQUIRE(elapsed > 0u, "ac4 nonpositive elapsed");
        REQUIRE(results[p].memory.activation_peak_bytes ==
                    plans[p].activation_peak_bytes,
                "ac4 timed peak check");
        times[p][s] = elapsed;
      }
    }

    uint64_t medians[4];
    for (size_t p = 0; p < 4u; ++p) {
      qsort(times[p], samples, sizeof(times[p][0]), cmp_u64);
      medians[p] = times[p][(samples - 1u) / 2u];
    }

    const size_t base_peak = results[0].memory.activation_peak_bytes;
    for (size_t p = 0; p < 4u; ++p) {
      REQUIRE(g_ac4_count < MAX_AC4_CASES, "ac4 capacity");
      const et_ac4_memory_v1 *m = &results[p].memory;
      if (p > 0u)
        REQUIRE(m->activation_peak_bytes < base_peak,
                "ac4 strict peak memory reduction");
      ac4_case_result *r = &g_ac4_results[g_ac4_count++];
      memset(r, 0, sizeof(*r));
      r->layers = shape.layers;
      r->policy = policy_names[p];
      r->checkpoint_every = cfgs[p].checkpoint_every;
      r->planned_activation_peak_bytes = plans[p].activation_peak_bytes;
      r->live_activation_peak_bytes = m->activation_peak_bytes;
      r->baseline_none_peak_bytes = base_peak;
      r->saved_activation_bytes = base_peak - m->activation_peak_bytes;
      r->memory_reduction_pct =
          100.0 * (double)r->saved_activation_bytes / (double)base_peak;
      r->gradient_staging_bytes = m->gradient_staging_bytes;
      r->workspace_bytes = m->workspace_bytes;
      r->retained_checkpoints = m->retained_checkpoints;
      r->retained_tapes = m->retained_tapes;
      r->rematerialized_layers = m->rematerialized_layers;
      r->warmup = warmup;
      r->samples = samples;
      r->iterations = iters;
      r->min_ns = times[p][0];
      r->median_ns = medians[p];
      r->max_ns = times[p][samples - 1u];
      r->median_ns_per_iter = (double)r->median_ns / (double)iters;
      r->time_vs_none_ratio = (double)medians[p] / (double)medians[0];
      memcpy(r->output_sha256, digests[p], 65u);
      free(workspaces[p]);
    }
  }
}

/* ========================================================================= */
/* 5. DIST5 Multi-Process Collective Benchmark (fork + AF_UNIX sockets)      */
/* ========================================================================= */

typedef struct {
  uint64_t min_ns;
  uint64_t median_ns;
  uint64_t max_ns;
  char output_sha256[65];
} dist5_wire_report;

static void fd_transfer_all(int fd, void *buf, size_t len, int writing) {
  uint8_t *p = (uint8_t *)buf;
  while (len != 0u) {
    const ssize_t n = writing ? write(fd, p, len) : read(fd, p, len);
    if (n < 0 && errno == EINTR)
      continue;
    REQUIRE(n > 0, writing ? "pipe write" : "pipe read");
    p += (size_t)n;
    len -= (size_t)n;
  }
}

static void dist5_worker(uint32_t rank, uint32_t world, const int *fds,
                         size_t fd_count, int report_fd, int is_sync_grad,
                         size_t elem_count, uint32_t warmup, uint32_t samples,
                         uint32_t iters) {
  enum { DIST5_MAX_ELEMS = 4096, DIST5_TIMEOUT_MS = 10000 };
  static float sendbuf[DIST5_MAX_ELEMS];
  static float recvbuf[DIST5_MAX_ELEMS];
  static float expected[DIST5_MAX_ELEMS];
  REQUIRE(elem_count <= DIST5_MAX_ELEMS, "dist5 elem cap");

  et_dist5_comm_t *comm = NULL;
  REQUIRE(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, rank,
                               world, fd_count == 0u ? NULL : fds, fd_count,
                               &comm) == ET_DIST5_OK,
          "dist5 comm create");
  for (size_t i = 0; i < fd_count; ++i)
    REQUIRE(close(fds[i]) == 0, "dist5 close borrowed fd");

  for (size_t i = 0; i < elem_count; ++i) {
    sendbuf[i] = ((float)((i * 7u + rank * 13u) % 31u) - 15.0f) * 0.125f;
    float acc = ((float)((i * 7u) % 31u) - 15.0f) * 0.125f;
    for (uint32_t r = 1u; r < world; ++r)
      acc += ((float)((i * 7u + r * 13u) % 31u) - 15.0f) * 0.125f;
    expected[i] = is_sync_grad ? (acc / (float)world) : acc;
  }

  REQUIRE(et_dist5_barrier(comm, 1u, DIST5_TIMEOUT_MS) == ET_DIST5_OK,
          "dist5 initial barrier");

  /* Warmup + initial bitwise check */
  for (uint32_t w = 0; w < warmup; ++w) {
    memset(recvbuf, 0, elem_count * sizeof(float));
    if (is_sync_grad) {
      REQUIRE(et_dist5_sync_gradients_f32(comm, sendbuf, recvbuf, elem_count,
                                          100u + w, DIST5_TIMEOUT_MS) ==
                  ET_DIST5_OK,
              "dist5 warmup sync_gradients");
    } else {
      REQUIRE(et_dist5_allreduce(comm, sendbuf, recvbuf, elem_count,
                                 ET_DIST5_F32, ET_DIST5_SUM, 100u + w,
                                 DIST5_TIMEOUT_MS) == ET_DIST5_OK,
              "dist5 warmup allreduce");
    }
    REQUIRE(memcmp(recvbuf, expected, elem_count * sizeof(float)) == 0,
            "dist5 warmup bitwise parity");
  }

  uint64_t times[MAX_SAMPLES];
  for (uint32_t s = 0; s < samples; ++s) {
    REQUIRE(et_dist5_barrier(comm, 200u + s, DIST5_TIMEOUT_MS) == ET_DIST5_OK,
            "dist5 sample barrier");
    memset(recvbuf, 0, elem_count * sizeof(float));
    const uint64_t t0 = now_ns();
    for (uint32_t it = 0; it < iters; ++it) {
      if (is_sync_grad) {
        REQUIRE(et_dist5_sync_gradients_f32(comm, sendbuf, recvbuf, elem_count,
                                            1000u + s * 1000u + it,
                                            DIST5_TIMEOUT_MS) == ET_DIST5_OK,
                "dist5 timed sync_gradients");
      } else {
        REQUIRE(et_dist5_allreduce(comm, sendbuf, recvbuf, elem_count,
                                   ET_DIST5_F32, ET_DIST5_SUM,
                                   1000u + s * 1000u + it,
                                   DIST5_TIMEOUT_MS) == ET_DIST5_OK,
                "dist5 timed allreduce");
      }
    }
    const uint64_t elapsed = now_ns() - t0;
    REQUIRE(elapsed > 0u, "dist5 nonpositive elapsed");
    REQUIRE(memcmp(recvbuf, expected, elem_count * sizeof(float)) == 0,
            "dist5 timed bitwise parity");
    times[s] = elapsed;
  }

  REQUIRE(et_dist5_barrier(comm, 999999u, DIST5_TIMEOUT_MS) == ET_DIST5_OK,
          "dist5 final barrier");
  et_dist5_comm_destroy(comm);

  if (rank == 0u) {
    qsort(times, samples, sizeof(times[0]), cmp_u64);
    dist5_wire_report wire;
    memset(&wire, 0, sizeof(wire));
    wire.min_ns = times[0];
    wire.median_ns = times[(samples - 1u) / 2u];
    wire.max_ns = times[samples - 1u];
    sha256_hex_buf(recvbuf, elem_count * sizeof(float), wire.output_sha256);
    fd_transfer_all(report_fd, &wire, sizeof(wire), 1);
    REQUIRE(close(report_fd) == 0, "dist5 close report_fd");
  }
}

static void run_dist5_case(uint32_t world, int is_sync_grad, size_t elem_count,
                           uint32_t warmup, uint32_t samples, uint32_t iters) {
  REQUIRE(world >= 1u && world <= 4u, "dist5 world range");
  REQUIRE(g_dist5_count < MAX_DIST5_CASES, "dist5 capacity");
  REQUIRE(fflush(NULL) == 0, "fflush before fork");

  int pairs[3][2];
  for (uint32_t r = 1u; r < world; ++r)
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, pairs[r - 1u]) == 0,
            "dist5 socketpair");

  int pipefd[2];
  REQUIRE(pipe(pipefd) == 0, "dist5 report pipe");

  pid_t pids[4];
  for (uint32_t rank = 0u; rank < world; ++rank) {
    pids[rank] = fork();
    REQUIRE(pids[rank] >= 0, "dist5 fork");
    if (pids[rank] == 0) {
      alarm(30u);
      REQUIRE(close(pipefd[0]) == 0, "child close pipe read");
      if (rank != 0u)
        REQUIRE(close(pipefd[1]) == 0, "worker close pipe write");
      int fds[3];
      size_t fd_count = 0u;
      for (uint32_t r = 1u; r < world; ++r) {
        if (rank == 0u)
          fds[fd_count++] = pairs[r - 1u][0];
        else
          REQUIRE(close(pairs[r - 1u][0]) == 0, "worker close coord end");
        if (rank == r)
          fds[fd_count++] = pairs[r - 1u][1];
        else
          REQUIRE(close(pairs[r - 1u][1]) == 0, "close unused worker end");
      }
      dist5_worker(rank, world, fds, fd_count, rank == 0u ? pipefd[1] : -1,
                   is_sync_grad, elem_count, warmup, samples, iters);
      exit(EXIT_SUCCESS);
    }
  }

  for (uint32_t r = 1u; r < world; ++r) {
    REQUIRE(close(pairs[r - 1u][0]) == 0, "parent close pair[0]");
    REQUIRE(close(pairs[r - 1u][1]) == 0, "parent close pair[1]");
  }
  REQUIRE(close(pipefd[1]) == 0, "parent close pipe write");

  dist5_wire_report wire;
  memset(&wire, 0, sizeof(wire));
  fd_transfer_all(pipefd[0], &wire, sizeof(wire), 0);
  REQUIRE(close(pipefd[0]) == 0, "parent close pipe read");

  int all_ok = 1;
  for (uint32_t rank = 0u; rank < world; ++rank) {
    int status = 0;
    pid_t got;
    do {
      got = waitpid(pids[rank], &status, 0);
    } while (got < 0 && errno == EINTR);
    REQUIRE(got == pids[rank], "dist5 waitpid");
    if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
      fprintf(stderr, "dist5 rank %u/%u failed (status %#x)\n", rank, world,
              status);
      all_ok = 0;
    }
  }
  REQUIRE(all_ok, "dist5 all ranks exited cleanly");

  dist5_case_result *r = &g_dist5_results[g_dist5_count++];
  memset(r, 0, sizeof(*r));
  r->operation = is_sync_grad ? "sync_gradients_f32" : "allreduce_f32";
  r->world_size = world;
  r->element_count = elem_count;
  r->payload_bytes = elem_count * sizeof(float);
  r->warmup = warmup;
  r->samples = samples;
  r->iterations = iters;
  r->min_ns = wire.min_ns;
  r->median_ns = wire.median_ns;
  r->max_ns = wire.max_ns;
  r->median_ns_per_iter = (double)r->median_ns / (double)iters;
  memcpy(r->output_sha256, wire.output_sha256, 65u);
}

static void run_dist5_benchmarks(int smoke) {
  static const uint32_t worlds[] = {1u, 2u, 4u};
  static const size_t counts[] = {256u, 4096u};
  const uint32_t warmup = smoke ? 1u : 5u;
  const uint32_t samples = smoke ? 3u : 7u;
  const uint32_t iters = smoke ? 3u : 40u;

  for (size_t wi = 0; wi < ARRAY_COUNT(worlds); ++wi) {
    for (size_t ci = 0; ci < ARRAY_COUNT(counts); ++ci) {
      run_dist5_case(worlds[wi], 0, counts[ci], warmup, samples, iters);
      run_dist5_case(worlds[wi], 1, counts[ci], warmup, samples, iters);
    }
  }
}

/* ========================================================================= */
/* Host metadata & canonical JSON builder                                    */
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

/* Format finite double to match Python's canonical json.dumps representation */
static void sb_append_double(strbuf *sb, double v) {
  REQUIRE(isfinite(v), "non-finite double in report");
  if (v == 0.0) {
    sb_append(sb, "0.0");
    return;
  }
  char buf[64];
  const double mag = fabs(v);
  if (mag >= 0.00009999995 && mag < 1e15) {
    const int n = snprintf(buf, sizeof(buf), "%.6f", v);
    REQUIRE(n > 0 && (size_t)n < sizeof(buf), "double fixed format");
    size_t len = (size_t)n;
    while (len > 1u && buf[len - 1u] == '0' && buf[len - 2u] != '.')
      --len;
    sb_append_mem(sb, buf, len);
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

static void sb_append_u64_array(strbuf *sb, const uint64_t *vals, size_t n) {
  sb_append_char(sb, '[');
  for (size_t i = 0; i < n; ++i) {
    if (i > 0u)
      sb_append_char(sb, ',');
    sb_append_u64(sb, vals[i]);
  }
  sb_append_char(sb, ']');
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
/* Canonical JSON payload assembly (strict lexicographical key order!)       */
/* ========================================================================= */

typedef struct {
  const char *output_path;
  const char *compiler_id;
  const char *cflags;
  const char *build_mode;
  const char *git_commit;
  int git_dirty;
  int smoke;
} cli_config;

static void build_canonical_payload(strbuf *p, const cli_config *cfg,
                                    uint64_t self_rss_kib,
                                    uint64_t child_rss_kib) {
  struct utsname u;
  REQUIRE(uname(&u) == 0, "uname");
  char os_rel[128], cpu_model[128], utc[32];
  read_os_release(os_rel);
  read_cpu_model(cpu_model);
  format_utc_now(utc);
  long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpus < 1L)
    ncpus = 1L;

  /* payload keys: ac4, dist5, environment, measurement, mod5, mp4, pa4_wider,
     process, toolchain */
  sb_append(p, "{\"ac4\":{\"base_shape\":{\"batch\":2,\"head_dim\":2,"
               "\"kv_heads\":2,\"query_heads\":4,\"tokens\":3},\"cases\":[");
  for (size_t i = 0; i < g_ac4_count; ++i) {
    const ac4_case_result *r = &g_ac4_results[i];
    if (i > 0u)
      sb_append_char(p, ',');
    sb_append(p, "{\"baseline_none_peak_bytes\":");
    sb_append_u64(p, (uint64_t)r->baseline_none_peak_bytes);
    sb_append(p, ",\"checkpoint_every\":");
    sb_append_u64(p, r->checkpoint_every);
    sb_append(p, ",\"gradient_staging_bytes\":");
    sb_append_u64(p, (uint64_t)r->gradient_staging_bytes);
    sb_append(p, ",\"iterations\":");
    sb_append_u64(p, r->iterations);
    sb_append(p, ",\"layers\":");
    sb_append_u64(p, r->layers);
    sb_append(p, ",\"live_activation_peak_bytes\":");
    sb_append_u64(p, (uint64_t)r->live_activation_peak_bytes);
    sb_append(p, ",\"max_ns\":");
    sb_append_u64(p, r->max_ns);
    sb_append(p, ",\"median_ns\":");
    sb_append_u64(p, r->median_ns);
    sb_append(p, ",\"median_ns_per_iter\":");
    sb_append_double(p, r->median_ns_per_iter);
    sb_append(p, ",\"memory_reduction_pct\":");
    sb_append_double(p, r->memory_reduction_pct);
    sb_append(p, ",\"min_ns\":");
    sb_append_u64(p, r->min_ns);
    sb_append(p, ",\"output_sha256\":");
    sb_append_json_str(p, r->output_sha256);
    sb_append(p, ",\"parity_with_none_bitwise\":true");
    sb_append(p, ",\"planned_activation_peak_bytes\":");
    sb_append_u64(p, (uint64_t)r->planned_activation_peak_bytes);
    sb_append(p, ",\"policy\":");
    sb_append_json_str(p, r->policy);
    sb_append(p, ",\"rematerialized_layers\":");
    sb_append_u64(p, r->rematerialized_layers);
    sb_append(p, ",\"retained_checkpoints\":");
    sb_append_u64(p, r->retained_checkpoints);
    sb_append(p, ",\"retained_tapes\":");
    sb_append_u64(p, r->retained_tapes);
    sb_append(p, ",\"samples\":");
    sb_append_u64(p, r->samples);
    sb_append(p, ",\"saved_activation_bytes\":");
    sb_append_u64(p, (uint64_t)r->saved_activation_bytes);
    sb_append(p, ",\"time_vs_none_ratio\":");
    sb_append_double(p, r->time_vs_none_ratio);
    sb_append(p, ",\"warmup\":");
    sb_append_u64(p, r->warmup);
    sb_append(p, ",\"workspace_bytes\":");
    sb_append_u64(p, (uint64_t)r->workspace_bytes);
    sb_append_char(p, '}');
  }
  sb_append(p, "],\"dropout_rate\":0.2},\"dist5\":{\"cases\":[");
  for (size_t i = 0; i < g_dist5_count; ++i) {
    const dist5_case_result *r = &g_dist5_results[i];
    if (i > 0u)
      sb_append_char(p, ',');
    sb_append(p, "{\"element_count\":");
    sb_append_u64(p, (uint64_t)r->element_count);
    sb_append(p, ",\"iterations\":");
    sb_append_u64(p, r->iterations);
    sb_append(p, ",\"max_ns\":");
    sb_append_u64(p, r->max_ns);
    sb_append(p, ",\"median_ns\":");
    sb_append_u64(p, r->median_ns);
    sb_append(p, ",\"median_ns_per_iter\":");
    sb_append_double(p, r->median_ns_per_iter);
    sb_append(p, ",\"min_ns\":");
    sb_append_u64(p, r->min_ns);
    sb_append(p, ",\"operation\":");
    sb_append_json_str(p, r->operation);
    sb_append(p, ",\"output_sha256\":");
    sb_append_json_str(p, r->output_sha256);
    sb_append(p, ",\"payload_bytes\":");
    sb_append_u64(p, (uint64_t)r->payload_bytes);
    sb_append(p, ",\"samples\":");
    sb_append_u64(p, r->samples);
    sb_append(p, ",\"verified_bitwise\":true,\"warmup\":");
    sb_append_u64(p, r->warmup);
    sb_append(p, ",\"world_size\":");
    sb_append_u64(p, r->world_size);
    sb_append_char(p, '}');
  }
  sb_append(p, "],\"transport\":\"af_unix_sock_stream\"},\"environment\":{"
               "\"architecture\":");
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

  sb_append(p, "},\"measurement\":{\"backend\":{\"acceleration\":false,"
               "\"device\":\"cpu\",\"evidence\":\"direct-native-c11-link\","
               "\"name\":\"host-cpu\"},\"clock\":\"clock_monotonic\","
               "\"elapsed_unit\":\"nanosecond\",\"limitations\":["
               "\"CPU-only serial execution; no GPU or accelerator kernels\","
               "\"DIST5 uses single-host AF_UNIX SOCK_STREAM IPC over forked processes\","
               "\"MOD5/PA4-Wider/MP4/AC4 cover bounded verified CPU-f32 ABI shapes only\""
               "],\"memory_source\":\"linux_getrusage_rusage_self\","
               "\"peak_rss_unit\":\"kibibyte\",\"smoke_mode\":");
  sb_append(p, cfg->smoke ? "true" : "false");

  sb_append(p, "},\"mod5\":{\"cases\":[");
  for (size_t i = 0; i < g_mod5_count; ++i) {
    const mod5_case_result *r = &g_mod5_results[i];
    if (i > 0u)
      sb_append_char(p, ',');
    sb_append(p, "{\"iterations\":");
    sb_append_u64(p, r->iterations);
    sb_append(p, ",\"kernel\":");
    sb_append_json_str(p, r->kernel);
    sb_append(p, ",\"max_ns\":");
    sb_append_u64(p, r->max_ns);
    sb_append(p, ",\"median_ns\":");
    sb_append_u64(p, r->median_ns);
    sb_append(p, ",\"median_ns_per_iter\":");
    sb_append_double(p, r->median_ns_per_iter);
    sb_append(p, ",\"min_ns\":");
    sb_append_u64(p, r->min_ns);
    sb_append(p, ",\"operation\":");
    sb_append_json_str(p, r->operation);
    sb_append(p, ",\"output_sha256\":");
    sb_append_json_str(p, r->output_sha256);
    sb_append(p, ",\"samples\":");
    sb_append_u64(p, r->samples);
    sb_append(p, ",\"shape\":");
    sb_append_u64_array(p, r->shape, r->rank);
    sb_append(p, ",\"warmup\":");
    sb_append_u64(p, r->warmup);
    sb_append_char(p, '}');
  }
  sb_append(p, "],\"provider\":\"mod5.cpu-f32.serial\"},\"mp4\":{\"cases\":[");
  for (size_t i = 0; i < g_mp4_count; ++i) {
    const mp4_case_result *r = &g_mp4_results[i];
    if (i > 0u)
      sb_append_char(p, ',');
    sb_append(p, "{\"capability\":");
    sb_append_json_str(p, r->capability);
    sb_append(p, ",\"element_count\":");
    sb_append_u64(p, (uint64_t)r->element_count);
    sb_append(p, ",\"iterations\":");
    sb_append_u64(p, r->iterations);
    sb_append(p, ",\"max_ns\":");
    sb_append_u64(p, r->max_ns);
    sb_append(p, ",\"median_ns\":");
    sb_append_u64(p, r->median_ns);
    sb_append(p, ",\"median_ns_per_iter\":");
    sb_append_double(p, r->median_ns_per_iter);
    sb_append(p, ",\"min_ns\":");
    sb_append_u64(p, r->min_ns);
    sb_append(p, ",\"narrow_bytes\":");
    sb_append_u64(p, (uint64_t)r->narrow_bytes);
    sb_append(p, ",\"narrow_dtype\":");
    sb_append_json_str(p, r->narrow_dtype);
    sb_append(p, ",\"operation\":");
    sb_append_json_str(p, r->operation);
    sb_append(p, ",\"output_sha256\":");
    sb_append_json_str(p, r->output_sha256);
    sb_append(p, ",\"samples\":");
    sb_append_u64(p, r->samples);
    sb_append(p, ",\"shape\":");
    sb_append_u64_array(p, r->shape, r->rank);
    sb_append(p, ",\"storage_byte_ratio\":2.0,\"storage_ratio_denominator\":1,"
                 "\"storage_ratio_numerator\":2,\"throughput_melem_per_s\":");
    sb_append_double(p, r->throughput_melem_per_s);
    sb_append(p, ",\"warmup\":");
    sb_append_u64(p, r->warmup);
    sb_append(p, ",\"wide_bytes\":");
    sb_append_u64(p, (uint64_t)r->wide_bytes);
    sb_append_char(p, '}');
  }
  sb_append(p, "],\"loss_scaler_verified\":");
  sb_append(p, g_mp4_loss_scaler_verified ? "true" : "false");
  sb_append(p, ",\"provider\":\"mp4.cpu-f32.serial\"},\"pa4_wider\":{\"cases\":[");
  for (size_t i = 0; i < g_pa4_count; ++i) {
    const pa4_case_result *r = &g_pa4_results[i];
    if (i > 0u)
      sb_append_char(p, ',');
    sb_append(p, "{\"ab_ratio_max\":");
    sb_append_double(p, r->ab_ratio_max);
    sb_append(p, ",\"ab_ratio_min\":");
    sb_append_double(p, r->ab_ratio_min);
    sb_append(p, ",\"ba_ratio_max\":");
    sb_append_double(p, r->ba_ratio_max);
    sb_append(p, ",\"ba_ratio_min\":");
    sb_append_double(p, r->ba_ratio_min);
    sb_append(p, ",\"iterations\":");
    sb_append_u64(p, r->iterations);
    sb_append(p, ",\"max_abs_diff\":");
    sb_append_double(p, r->max_abs_diff);
    sb_append(p, ",\"max_rel_diff\":");
    sb_append_double(p, r->max_rel_diff);
    sb_append(p, ",\"online_max_ns\":");
    sb_append_u64(p, r->online_max_ns);
    sb_append(p, ",\"online_median_ns\":");
    sb_append_u64(p, r->online_median_ns);
    sb_append(p, ",\"online_median_ns_per_iter\":");
    sb_append_double(p, r->online_median_ns_per_iter);
    sb_append(p, ",\"online_min_ns\":");
    sb_append_u64(p, r->online_min_ns);
    sb_append(p, ",\"online_output_sha256\":");
    sb_append_json_str(p, r->online_output_sha256);
    sb_append(p, ",\"online_vs_ref_ratio\":");
    sb_append_double(p, r->online_vs_ref_ratio);
    sb_append(p, ",\"operation\":");
    sb_append_json_str(p, r->operation);
    sb_append(p, ",\"pairs\":");
    sb_append_u64(p, r->pairs);
    sb_append(p, ",\"parity_verified\":true,\"ref_max_ns\":");
    sb_append_u64(p, r->ref_max_ns);
    sb_append(p, ",\"ref_median_ns\":");
    sb_append_u64(p, r->ref_median_ns);
    sb_append(p, ",\"ref_median_ns_per_iter\":");
    sb_append_double(p, r->ref_median_ns_per_iter);
    sb_append(p, ",\"ref_min_ns\":");
    sb_append_u64(p, r->ref_min_ns);
    sb_append(p, ",\"ref_output_sha256\":");
    sb_append_json_str(p, r->ref_output_sha256);
    sb_append(p, ",\"scope\":\"dispatch\",\"shape\":");
    sb_append_u64_array(p, r->shape, 6u);
    sb_append(p, ",\"speedup_pct\":");
    sb_append_double(p, r->speedup_pct);
    sb_append(p, ",\"warmup\":");
    sb_append_u64(p, r->warmup);
    sb_append_char(p, '}');
  }
  sb_append(p, "],\"providers\":{\"online\":\"pa4-wider.online\",\"ref\":"
               "\"pa4-wider.ref\"}},\"process\":{\"children_peak_rss_kib\":");
  sb_append_u64(p, child_rss_kib);
  sb_append(p, ",\"self_peak_rss_kib\":");
  sb_append_u64(p, self_rss_kib);
  sb_append(p, "},\"toolchain\":{\"build_mode\":");
  sb_append_json_str(p, cfg->build_mode);
  sb_append(p, ",\"cflags\":");
  sb_append_json_str(p, cfg->cflags);
  sb_append(p, ",\"compiler\":");
  sb_append_json_str(p, cfg->compiler_id);
  sb_append(p, ",\"git_commit\":");
  sb_append_json_str(p, cfg->git_commit);
  sb_append(p, ",\"git_dirty\":");
  sb_append(p, cfg->git_dirty ? "true" : "false");
  sb_append(p, ",\"libraries\":[\"libeshkol_transformer_k1.a\","
               "\"libeshkol_transformer_mod5.a\","
               "\"libeshkol_transformer_pa4_wider.a\","
               "\"libeshkol_transformer_mp4.a\","
               "\"libeshkol_ac4.a\","
               "\"libeshkol_dist5.a\"]}}");
}

int main(int argc, char **argv) {
  cli_config cfg = {.output_path = NULL,
                    .compiler_id = __VERSION__,
                    .cflags = "c11",
                    .build_mode = "normal",
                    .git_commit = "unknown",
                    .git_dirty = 0,
                    .smoke = 0};

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      cfg.output_path = argv[++i];
    } else if (strcmp(argv[i], "--compiler-id") == 0 && i + 1 < argc) {
      cfg.compiler_id = argv[++i];
    } else if (strcmp(argv[i], "--cflags") == 0 && i + 1 < argc) {
      cfg.cflags = argv[++i];
    } else if (strcmp(argv[i], "--build-mode") == 0 && i + 1 < argc) {
      cfg.build_mode = argv[++i];
    } else if (strcmp(argv[i], "--git-commit") == 0 && i + 1 < argc) {
      cfg.git_commit = argv[++i];
    } else if (strcmp(argv[i], "--git-dirty") == 0 && i + 1 < argc) {
      cfg.git_dirty = strcmp(argv[++i], "1") == 0 ? 1 : 0;
    } else if (strcmp(argv[i], "--smoke") == 0) {
      cfg.smoke = 1;
    } else {
      fprintf(stderr, "Usage: %s --output <path> [--smoke] [--compiler-id S] "
                      "[--cflags S] [--build-mode normal|sanitize] "
                      "[--git-commit SHA] [--git-dirty 0|1]\n",
              argv[0]);
      return EXIT_FAILURE;
    }
  }
  REQUIRE(cfg.output_path != NULL && cfg.output_path[0] != '\0',
          "missing required --output <path>");
  REQUIRE(strcmp(cfg.build_mode, "normal") == 0 ||
              strcmp(cfg.build_mode, "sanitize") == 0,
          "invalid --build-mode");

  /* Run all 5 native Wave 4/5 benchmark suites (no heap held across fork!) */
  run_mod5_benchmarks(cfg.smoke);
  run_pa4_wider_benchmarks(cfg.smoke);
  run_mp4_benchmarks(cfg.smoke);
  run_ac4_benchmarks(cfg.smoke);
  run_dist5_benchmarks(cfg.smoke);

  struct rusage ru_self, ru_children;
  REQUIRE(getrusage(RUSAGE_SELF, &ru_self) == 0, "getrusage self");
  REQUIRE(getrusage(RUSAGE_CHILDREN, &ru_children) == 0, "getrusage children");
  REQUIRE(ru_self.ru_maxrss > 0L, "nonpositive self ru_maxrss");
  const uint64_t self_rss_kib = (uint64_t)ru_self.ru_maxrss;
  const uint64_t child_rss_kib =
      ru_children.ru_maxrss > 0L ? (uint64_t)ru_children.ru_maxrss : 0u;

  strbuf payload;
  sb_init(&payload);
  build_canonical_payload(&payload, &cfg, self_rss_kib, child_rss_kib);

  /* Compute SHA-256 over canonical integrity object:
     {"format":"wave45-bench-v1","payload":<payload>,"version":1} */
  strbuf integrity;
  sb_init(&integrity);
  sb_append(&integrity, "{\"format\":\"wave45-bench-v1\",\"payload\":");
  sb_append_mem(&integrity, payload.data, payload.len);
  sb_append(&integrity, ",\"version\":1}");

  char digest_hex[65];
  sha256_hex_buf(integrity.data, integrity.len, digest_hex);

  /* Assemble canonical outer envelope + single trailing LF */
  strbuf envelope;
  sb_init(&envelope);
  sb_append(&envelope, "{\"checksum\":{\"algorithm\":\"sha256\",\"digest\":");
  sb_append_json_str(&envelope, digest_hex);
  sb_append(&envelope, "},\"format\":\"wave45-bench-v1\",\"payload\":");
  sb_append_mem(&envelope, payload.data, payload.len);
  sb_append(&envelope, ",\"version\":1}\n");

  write_file_atomic(cfg.output_path, envelope.data, envelope.len);

  printf("wave45-bench-v1 PASS: mode=%s smoke=%d mod5=%zu pa4=%zu mp4=%zu "
         "ac4=%zu dist5=%zu self_rss_kib=%" PRIu64 " sha256=%s report=%s\n",
         cfg.build_mode, cfg.smoke, g_mod5_count, g_pa4_count, g_mp4_count,
         g_ac4_count, g_dist5_count, self_rss_kib, digest_hex, cfg.output_path);

  sb_free(&envelope);
  sb_free(&integrity);
  sb_free(&payload);
  return EXIT_SUCCESS;
}
