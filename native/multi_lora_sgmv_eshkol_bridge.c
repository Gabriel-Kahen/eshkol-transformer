#define _POSIX_C_SOURCE 200809L

#include "multi_lora_sgmv_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

/* Standalone SHA-256 for batched Y digest verification inside the AOT bridge. */
typedef struct bridge_sha256 {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} bridge_sha256;

static uint32_t br_rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void br_sha256_block(bridge_sha256 *st, const uint8_t block[64]) {
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
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0u; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = br_rotr32(w[i - 15u], 7u) ^
                        br_rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 = br_rotr32(w[i - 2u], 17u) ^
                        br_rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = st->h[0];
  b = st->h[1];
  c = st->h[2];
  d = st->h[3];
  e = st->h[4];
  f = st->h[5];
  g = st->h[6];
  h = st->h[7];
  for (i = 0u; i < 64u; ++i) {
    const uint32_t s1 = br_rotr32(e, 6u) ^ br_rotr32(e, 11u) ^ br_rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = br_rotr32(a, 2u) ^ br_rotr32(a, 13u) ^ br_rotr32(a, 22u);
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
  st->h[0] += a;
  st->h[1] += b;
  st->h[2] += c;
  st->h[3] += d;
  st->h[4] += e;
  st->h[5] += f;
  st->h[6] += g;
  st->h[7] += h;
}

static void br_hash_f32_le(const float *data, size_t count,
                           uint8_t digest[32]) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  bridge_sha256 st;
  memcpy(st.h, initial, sizeof(initial));
  st.length = 0u;
  st.used = 0u;

  for (size_t i = 0u; i < count; ++i) {
    uint32_t bits = 0u;
    uint8_t le[4];
    memcpy(&bits, &data[i], sizeof(bits));
    le[0] = (uint8_t)(bits & 0xffu);
    le[1] = (uint8_t)((bits >> 8u) & 0xffu);
    le[2] = (uint8_t)((bits >> 16u) & 0xffu);
    le[3] = (uint8_t)((bits >> 24u) & 0xffu);
    st.length += 4u;
    size_t rem = 4u;
    const uint8_t *p = le;
    while (rem != 0u) {
      size_t take = 64u - st.used;
      if (take > rem)
        take = rem;
      memcpy(st.block + st.used, p, take);
      st.used += take;
      p += take;
      rem -= take;
      if (st.used == 64u) {
        br_sha256_block(&st, st.block);
        st.used = 0u;
      }
    }
  }
  const uint64_t bit_len = st.length * UINT64_C(8);
  st.block[st.used++] = 0x80u;
  if (st.used > 56u) {
    memset(st.block + st.used, 0, 64u - st.used);
    br_sha256_block(&st, st.block);
    st.used = 0u;
  }
  memset(st.block + st.used, 0, 56u - st.used);
  for (size_t i = 0u; i < 8u; ++i)
    st.block[63u - i] = (uint8_t)(bit_len >> (8u * i));
  br_sha256_block(&st, st.block);
  for (size_t i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(st.h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(st.h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(st.h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)st.h[i];
  }
}

static uint32_t br_f32_bits(float v) {
  uint32_t u = 0u;
  memcpy(&u, &v, sizeof(u));
  return u;
}

static int64_t digest_word_be64(const uint8_t digest[32], int64_t word_index) {
  if (word_index < 0 || word_index >= 4)
    return 0;
  const size_t off = (size_t)word_index * 8u;
  uint64_t w = 0u;
  for (size_t i = 0u; i < 8u; ++i)
    w = (w << 8u) | (uint64_t)digest[off + i];
  /* Mask to 62-bit positive integer for clean Eshkol i64 display/compare. */
  return (int64_t)(w & UINT64_C(0x3fffffffffffffff));
}

struct et_mlora_aot_ctx {
  uint32_t m;
  et_mlora_registry_config cfg;
  et_mlora_registry *reg;
  float w_base[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_IN_DIM];
  float bias[ET_MLORA_MAX_OUT_DIM];
  float x[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
  float dy[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  float target[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  float y[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  float y_ref[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  float dx[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
  int32_t row_adapter_ids[ET_MLORA_MAX_M];
  et_mlora_sgmv_stats last_stats;
  uint8_t base_sha256[ET_MLORA_SHA256_BYTES];
  uint8_t y_sha256[ET_MLORA_SHA256_BYTES];
};

int32_t et_mlora_aot_create(int64_t m, int64_t in_dim, int64_t out_dim,
                            int64_t max_adapters, int64_t max_rank,
                            int64_t has_bias, et_mlora_aot_ctx **out_ctx) {
  if (out_ctx == NULL || m < 1 || m > (int64_t)ET_MLORA_MAX_M ||
      in_dim < 1 || in_dim > (int64_t)ET_MLORA_MAX_IN_DIM ||
      out_dim < 1 || out_dim > (int64_t)ET_MLORA_MAX_OUT_DIM ||
      max_adapters < 1 || max_adapters > (int64_t)ET_MLORA_MAX_ADAPTERS ||
      max_rank < 1 || max_rank > (int64_t)ET_MLORA_MAX_RANK ||
      (has_bias != 0 && has_bias != 1))
    return ET_MLORA_ERR_ARGUMENT;

  et_mlora_registry_config cfg;
  cfg.in_dim = (uint32_t)in_dim;
  cfg.out_dim = (uint32_t)out_dim;
  cfg.max_adapters = (uint32_t)max_adapters;
  cfg.max_rank = (uint32_t)max_rank;
  cfg.has_bias = (uint32_t)has_bias;
  cfg.reserved = 0u;

  et_mlora_status status = et_mlora_validate_registry_config_v1(&cfg);
  if (status != ET_MLORA_OK)
    return status;

  et_mlora_aot_ctx *ctx =
      (et_mlora_aot_ctx *)calloc(1u, sizeof(et_mlora_aot_ctx));
  if (ctx == NULL)
    return ET_MLORA_ERR_NOMEM;

  ctx->m = (uint32_t)m;
  ctx->cfg = cfg;
  for (uint32_t i = 0u; i < ctx->m; ++i)
    ctx->row_adapter_ids[i] = ET_MLORA_BASE_ONLY_ID;

  *out_ctx = ctx;
  return ET_MLORA_OK;
}

void et_mlora_aot_destroy(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  et_mlora_registry_destroy_v1(ctx->reg);
  free(ctx);
}

int32_t et_mlora_aot_set_w_base(et_mlora_aot_ctx *ctx, int64_t index,
                                double value) {
  if (ctx == NULL || ctx->reg != NULL)
    return ET_MLORA_ERR_ARGUMENT;
  const int64_t count = (int64_t)ctx->cfg.out_dim * (int64_t)ctx->cfg.in_dim;
  if (index < 0 || index >= count)
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  ctx->w_base[index] = (float)value;
  return ET_MLORA_OK;
}

int32_t et_mlora_aot_set_bias(et_mlora_aot_ctx *ctx, int64_t index,
                              double value) {
  if (ctx == NULL || ctx->reg != NULL || ctx->cfg.has_bias == 0u)
    return ET_MLORA_ERR_ARGUMENT;
  if (index < 0 || index >= (int64_t)ctx->cfg.out_dim)
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  ctx->bias[index] = (float)value;
  return ET_MLORA_OK;
}

int32_t et_mlora_aot_finalize_base(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg != NULL)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_registry_create_v1(
      &ctx->cfg, ctx->w_base, ctx->cfg.has_bias != 0u ? ctx->bias : NULL,
      &ctx->reg);
}

int64_t et_mlora_aot_x_count(const et_mlora_aot_ctx *ctx) {
  if (ctx == NULL)
    return 0;
  return (int64_t)ctx->m * (int64_t)ctx->cfg.in_dim;
}

int64_t et_mlora_aot_w_count(const et_mlora_aot_ctx *ctx) {
  if (ctx == NULL)
    return 0;
  return (int64_t)ctx->cfg.out_dim * (int64_t)ctx->cfg.in_dim;
}

int64_t et_mlora_aot_y_count(const et_mlora_aot_ctx *ctx) {
  if (ctx == NULL)
    return 0;
  return (int64_t)ctx->m * (int64_t)ctx->cfg.out_dim;
}

int32_t et_mlora_aot_set_x(et_mlora_aot_ctx *ctx, int64_t index, double value) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_x_count(ctx))
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  ctx->x[index] = (float)value;
  return ET_MLORA_OK;
}

double et_mlora_aot_get_x(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_x_count(ctx))
    return 0.0;
  return (double)ctx->x[index];
}

int32_t et_mlora_aot_set_dy(et_mlora_aot_ctx *ctx, int64_t index,
                            double value) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  ctx->dy[index] = (float)value;
  return ET_MLORA_OK;
}

int32_t et_mlora_aot_set_target(et_mlora_aot_ctx *ctx, int64_t index,
                                double value) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  ctx->target[index] = (float)value;
  return ET_MLORA_OK;
}

double et_mlora_aot_get_target(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return 0.0;
  return (double)ctx->target[index];
}

int32_t et_mlora_aot_set_row_adapter(et_mlora_aot_ctx *ctx, int64_t row,
                                     int64_t adapter_id) {
  if (ctx == NULL || row < 0 || row >= (int64_t)ctx->m ||
      adapter_id < INT32_MIN || adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  ctx->row_adapter_ids[row] = (int32_t)adapter_id;
  return ET_MLORA_OK;
}

int64_t et_mlora_aot_get_row_adapter(const et_mlora_aot_ctx *ctx, int64_t row) {
  if (ctx == NULL || row < 0 || row >= (int64_t)ctx->m)
    return -999;
  return (int64_t)ctx->row_adapter_ids[row];
}

double et_mlora_aot_get_y(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return 0.0;
  return (double)ctx->y[index];
}

double et_mlora_aot_get_y_ref(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return 0.0;
  return (double)ctx->y_ref[index];
}

double et_mlora_aot_get_dx(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_x_count(ctx))
    return 0.0;
  return (double)ctx->dx[index];
}

int64_t et_mlora_aot_y_bits(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return -1;
  return (int64_t)br_f32_bits(ctx->y[index]);
}

int64_t et_mlora_aot_y_ref_bits(const et_mlora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_mlora_aot_y_count(ctx))
    return -1;
  return (int64_t)br_f32_bits(ctx->y_ref[index]);
}

int32_t et_mlora_aot_register_seeded(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                                     int64_t rank, double alpha, int64_t seed) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  if (adapter_id < INT32_MIN || adapter_id > INT32_MAX || rank < 0 ||
      rank > (int64_t)UINT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_register_seeded_adapter_v1(
      ctx->reg, (int32_t)adapter_id, (uint32_t)rank, (float)alpha,
      (uint64_t)seed, NULL);
}

int32_t et_mlora_aot_unregister(et_mlora_aot_ctx *ctx, int64_t adapter_id) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < INT32_MIN ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_unregister_adapter_v1(ctx->reg, (int32_t)adapter_id);
}

int32_t et_mlora_aot_pin(et_mlora_aot_ctx *ctx, int64_t adapter_id) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < INT32_MIN ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_pin_adapter_v1(ctx->reg, (int32_t)adapter_id, NULL);
}

int32_t et_mlora_aot_unpin(et_mlora_aot_ctx *ctx, int64_t adapter_id) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < INT32_MIN ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_unpin_adapter_v1(ctx->reg, (int32_t)adapter_id, NULL);
}

int64_t et_mlora_aot_adapter_rank(const et_mlora_aot_ctx *ctx,
                                  int64_t adapter_id) {
  et_mlora_adapter_info info;
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  return (int64_t)info.rank;
}

int64_t et_mlora_aot_adapter_generation(const et_mlora_aot_ctx *ctx,
                                        int64_t adapter_id) {
  et_mlora_adapter_info info;
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  return (int64_t)info.generation;
}

int64_t et_mlora_aot_adapter_pin_count(const et_mlora_aot_ctx *ctx,
                                       int64_t adapter_id) {
  et_mlora_adapter_info info;
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  return (int64_t)info.pin_count;
}

int64_t et_mlora_aot_adapter_step(const et_mlora_aot_ctx *ctx,
                                  int64_t adapter_id) {
  et_mlora_adapter_info info;
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  return (int64_t)info.adamw_step;
}

int32_t et_mlora_aot_set_a(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index, double value) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  et_mlora_adapter_info info;
  et_mlora_status status =
      et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info);
  if (status != ET_MLORA_OK)
    return status;
  const int64_t a_count = (int64_t)info.rank * (int64_t)ctx->cfg.in_dim;
  if (index < 0 || index >= a_count)
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  float a_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float b_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  status = et_mlora_get_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                           b_buf);
  if (status != ET_MLORA_OK)
    return status;
  a_buf[index] = (float)value;
  return et_mlora_set_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                         b_buf);
}

int32_t et_mlora_aot_set_b(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index, double value) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  et_mlora_adapter_info info;
  et_mlora_status status =
      et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info);
  if (status != ET_MLORA_OK)
    return status;
  const int64_t b_count = (int64_t)ctx->cfg.out_dim * (int64_t)info.rank;
  if (index < 0 || index >= b_count)
    return ET_MLORA_ERR_ARGUMENT;
  if (!isfinite(value) || !isfinite((float)value))
    return ET_MLORA_ERR_NONFINITE;
  float a_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float b_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  status = et_mlora_get_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                           b_buf);
  if (status != ET_MLORA_OK)
    return status;
  b_buf[index] = (float)value;
  return et_mlora_set_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                         b_buf);
}

double et_mlora_aot_get_a(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                          int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return 0.0;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return 0.0;
  const int64_t a_count = (int64_t)info.rank * (int64_t)ctx->cfg.in_dim;
  if (index < 0 || index >= a_count)
    return 0.0;
  float a_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float b_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                      b_buf) != ET_MLORA_OK)
    return 0.0;
  return (double)a_buf[index];
}

double et_mlora_aot_get_b(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                          int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return 0.0;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return 0.0;
  const int64_t b_count = (int64_t)ctx->cfg.out_dim * (int64_t)info.rank;
  if (index < 0 || index >= b_count)
    return 0.0;
  float a_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float b_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_weights_v1(ctx->reg, (int32_t)adapter_id, a_buf,
                                      b_buf) != ET_MLORA_OK)
    return 0.0;
  return (double)b_buf[index];
}

double et_mlora_aot_get_da(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return 0.0;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return 0.0;
  const int64_t a_count = (int64_t)info.rank * (int64_t)ctx->cfg.in_dim;
  if (index < 0 || index >= a_count)
    return 0.0;
  float da_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float db_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_grads_v1(ctx->reg, (int32_t)adapter_id, da_buf,
                                    db_buf) != ET_MLORA_OK)
    return 0.0;
  return (double)da_buf[index];
}

double et_mlora_aot_get_db(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                           int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return 0.0;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return 0.0;
  const int64_t b_count = (int64_t)ctx->cfg.out_dim * (int64_t)info.rank;
  if (index < 0 || index >= b_count)
    return 0.0;
  float da_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float db_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_grads_v1(ctx->reg, (int32_t)adapter_id, da_buf,
                                    db_buf) != ET_MLORA_OK)
    return 0.0;
  return (double)db_buf[index];
}

int64_t et_mlora_aot_da_bits(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                             int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  const int64_t a_count = (int64_t)info.rank * (int64_t)ctx->cfg.in_dim;
  if (index < 0 || index >= a_count)
    return -1;
  float da_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float db_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_grads_v1(ctx->reg, (int32_t)adapter_id, da_buf,
                                    db_buf) != ET_MLORA_OK)
    return -1;
  return (int64_t)br_f32_bits(da_buf[index]);
}

int64_t et_mlora_aot_db_bits(const et_mlora_aot_ctx *ctx, int64_t adapter_id,
                             int64_t index) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return -1;
  et_mlora_adapter_info info;
  if (et_mlora_get_adapter_info_v1(ctx->reg, (int32_t)adapter_id, &info) !=
      ET_MLORA_OK)
    return -1;
  const int64_t b_count = (int64_t)ctx->cfg.out_dim * (int64_t)info.rank;
  if (index < 0 || index >= b_count)
    return -1;
  float da_buf[ET_MLORA_MAX_RANK * ET_MLORA_MAX_IN_DIM];
  float db_buf[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_RANK];
  if (et_mlora_get_adapter_grads_v1(ctx->reg, (int32_t)adapter_id, da_buf,
                                    db_buf) != ET_MLORA_OK)
    return -1;
  return (int64_t)br_f32_bits(db_buf[index]);
}

int32_t et_mlora_aot_sgmv_forward(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_sgmv_forward_v1(ctx->reg, ctx->m, ctx->x,
                                  ctx->row_adapter_ids, ctx->y,
                                  &ctx->last_stats);
}

int32_t et_mlora_aot_single_row_ref_forward(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  for (uint32_t i = 0u; i < ctx->m; ++i) {
    const et_mlora_status status = et_mlora_single_adapter_forward_v1(
        ctx->reg, ctx->row_adapter_ids[i], 1u,
        &ctx->x[(size_t)i * ctx->cfg.in_dim],
        &ctx->y_ref[(size_t)i * ctx->cfg.out_dim]);
    if (status != ET_MLORA_OK)
      return status;
  }
  return ET_MLORA_OK;
}

int32_t et_mlora_aot_sgmv_backward(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_sgmv_backward_v1(ctx->reg, ctx->m, ctx->x,
                                   ctx->row_adapter_ids, ctx->dy, ctx->dx,
                                   &ctx->last_stats);
}

int32_t et_mlora_aot_adamw_step(et_mlora_aot_ctx *ctx, int64_t adapter_id,
                                double lr, double beta1, double beta2,
                                double eps, double weight_decay) {
  if (ctx == NULL || ctx->reg == NULL || adapter_id < 0 ||
      adapter_id > INT32_MAX)
    return ET_MLORA_ERR_ARGUMENT;
  et_mlora_adamw_config cfg;
  cfg.lr = (float)lr;
  cfg.beta1 = (float)beta1;
  cfg.beta2 = (float)beta2;
  cfg.eps = (float)eps;
  cfg.weight_decay = (float)weight_decay;
  return et_mlora_adamw_step_adapter_v1(ctx->reg, (int32_t)adapter_id, &cfg);
}

int64_t et_mlora_aot_last_base_rows(const et_mlora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_stats.base_only_rows;
}

int64_t et_mlora_aot_last_adapted_rows(const et_mlora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_stats.adapted_rows;
}

int64_t et_mlora_aot_last_active_segments(const et_mlora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->last_stats.active_segments;
}

int32_t et_mlora_aot_compute_base_sha256(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_get_base_sha256_v1(ctx->reg, ctx->base_sha256);
}

int64_t et_mlora_aot_base_sha256_word(const et_mlora_aot_ctx *ctx,
                                      int64_t word_index) {
  if (ctx == NULL)
    return 0;
  return digest_word_be64(ctx->base_sha256, word_index);
}

int32_t et_mlora_aot_compute_y_sha256(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  br_hash_f32_le(ctx->y, (size_t)ctx->m * (size_t)ctx->cfg.out_dim,
                 ctx->y_sha256);
  return ET_MLORA_OK;
}

int64_t et_mlora_aot_y_sha256_word(const et_mlora_aot_ctx *ctx,
                                   int64_t word_index) {
  if (ctx == NULL)
    return 0;
  return digest_word_be64(ctx->y_sha256, word_index);
}

int64_t et_mlora_aot_env_phase(void) {
  const char *phase = getenv("MLORA_AOT_PHASE");
  if (phase == NULL || phase[0] == '\0' || strcmp(phase, "all") == 0)
    return 0;
  if (strcmp(phase, "train") == 0)
    return 1;
  if (strcmp(phase, "reload") == 0)
    return 2;
  return -1;
}

int32_t et_mlora_aot_save_env_bundle(const et_mlora_aot_ctx *ctx,
                                     int64_t include_adamw) {
  if (ctx == NULL || ctx->reg == NULL ||
      (include_adamw != 0 && include_adamw != 1))
    return ET_MLORA_ERR_ARGUMENT;
  const char *path = getenv("MLORA_AOT_BUNDLE");
  if (path == NULL || path[0] == '\0')
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_bundle_save_v1(path, ctx->reg, (uint32_t)include_adamw);
}

int32_t et_mlora_aot_load_env_bundle(et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  const char *path = getenv("MLORA_AOT_BUNDLE");
  if (path == NULL || path[0] == '\0')
    return ET_MLORA_ERR_ARGUMENT;
  return et_mlora_bundle_load_v1(path, ctx->reg, NULL);
}

static int br_write_file(const char *path, const uint8_t *buf, size_t size) {
  const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0)
    return 0;
  size_t off = 0u;
  while (off < size) {
    const ssize_t w = write(fd, buf + off, size - off);
    if (w < 0 && errno == EINTR)
      continue;
    if (w <= 0) {
      (void)close(fd);
      return 0;
    }
    off += (size_t)w;
  }
  return close(fd) == 0;
}

int32_t et_mlora_aot_check_tampered_env(const et_mlora_aot_ctx *ctx,
                                        int64_t tamper_kind) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  const char *bundle_path = getenv("MLORA_AOT_BUNDLE");
  const char *tamper_path = getenv("MLORA_AOT_TAMPER");
  if (bundle_path == NULL || bundle_path[0] == '\0' ||
      tamper_path == NULL || tamper_path[0] == '\0')
    return ET_MLORA_ERR_ARGUMENT;

  /* Read valid bundle bytes first. */
  const int fd = open(bundle_path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return ET_MLORA_ERR_IO;

  uint8_t *buf = (uint8_t *)malloc(ET_MLORA_BUNDLE_MAX_BYTES);
  if (buf == NULL) {
    (void)close(fd);
    return ET_MLORA_ERR_NOMEM;
  }
  size_t len = 0u;
  while (len < ET_MLORA_BUNDLE_MAX_BYTES) {
    const ssize_t n = read(fd, buf + len, ET_MLORA_BUNDLE_MAX_BYTES - len);
    if (n == 0)
      break;
    if (n < 0) {
      if (errno == EINTR)
        continue;
      (void)close(fd);
      free(buf);
      return ET_MLORA_ERR_IO;
    }
    len += (size_t)n;
  }
  (void)close(fd);
  if (len < ET_MLORA_BUNDLE_MIN_BYTES) {
    free(buf);
    return ET_MLORA_ERR_IO;
  }

  /* Create a temporary registry (perturbed W_base for kind 6, identical else). */
  float w_alt[ET_MLORA_MAX_OUT_DIM * ET_MLORA_MAX_IN_DIM];
  const size_t w_count = (size_t)ctx->cfg.out_dim * (size_t)ctx->cfg.in_dim;
  memcpy(w_alt, ctx->w_base, w_count * sizeof(float));
  if (tamper_kind == 6)
    w_alt[0] += 0.125f;

  et_mlora_registry *tmp_reg = NULL;
  et_mlora_status status = et_mlora_registry_create_v1(
      &ctx->cfg, w_alt, ctx->cfg.has_bias != 0u ? ctx->bias : NULL, &tmp_reg);
  if (status != ET_MLORA_OK) {
    free(buf);
    return status;
  }

  if (tamper_kind == 6) {
    status = et_mlora_bundle_load_v1(bundle_path, tmp_reg, NULL);
    et_mlora_registry_destroy_v1(tmp_reg);
    free(buf);
    return status;
  }

  (void)unlink(tamper_path);
  if (tamper_kind == 5) {
    if (symlink(bundle_path, tamper_path) != 0) {
      et_mlora_registry_destroy_v1(tmp_reg);
      free(buf);
      return ET_MLORA_ERR_IO;
    }
    status = et_mlora_bundle_load_v1(tamper_path, tmp_reg, NULL);
    (void)unlink(tamper_path);
    et_mlora_registry_destroy_v1(tmp_reg);
    free(buf);
    return status;
  }

  size_t out_len = len;
  if (tamper_kind == 1) {
    /* Flip payload/checksum byte. */
    buf[len - 1u] ^= 0x5au;
  } else if (tamper_kind == 2) {
    /* Truncate by 7 bytes. */
    out_len = len - 7u;
  } else if (tamper_kind == 3) {
    /* Bad magic. */
    buf[0] = 'X';
  } else if (tamper_kind == 4) {
    /* Bad version major. */
    buf[8] = 9u;
  } else {
    et_mlora_registry_destroy_v1(tmp_reg);
    free(buf);
    return ET_MLORA_ERR_ARGUMENT;
  }

  if (!br_write_file(tamper_path, buf, out_len)) {
    et_mlora_registry_destroy_v1(tmp_reg);
    free(buf);
    return ET_MLORA_ERR_IO;
  }
  status = et_mlora_bundle_load_v1(tamper_path, tmp_reg, NULL);
  (void)unlink(tamper_path);
  et_mlora_registry_destroy_v1(tmp_reg);
  free(buf);
  return status;
}

int32_t et_mlora_aot_check_nonfinite_x(const et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  float x_bad[ET_MLORA_MAX_M * ET_MLORA_MAX_IN_DIM];
  float y_tmp[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  const size_t x_count = (size_t)ctx->m * (size_t)ctx->cfg.in_dim;
  memcpy(x_bad, ctx->x, x_count * sizeof(float));
  const uint32_t qnan_bits = UINT32_C(0x7fc00001);
  memcpy(&x_bad[0], &qnan_bits, sizeof(qnan_bits));
  return et_mlora_sgmv_forward_v1(ctx->reg, ctx->m, x_bad, ctx->row_adapter_ids,
                                  y_tmp, NULL);
}

int32_t et_mlora_aot_check_fp_env_trap(const et_mlora_aot_ctx *ctx) {
  if (ctx == NULL || ctx->reg == NULL)
    return ET_MLORA_ERR_ARGUMENT;
  float y_tmp[ET_MLORA_MAX_M * ET_MLORA_MAX_OUT_DIM];
  const int saved = fegetround();
  if (fesetround(FE_UPWARD) != 0)
    return ET_MLORA_ERR_FP_ENV;
  const et_mlora_status status = et_mlora_sgmv_forward_v1(
      ctx->reg, ctx->m, ctx->x, ctx->row_adapter_ids, y_tmp, NULL);
  (void)fesetround(saved);
  return status;
}
