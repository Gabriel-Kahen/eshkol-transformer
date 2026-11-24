#define _POSIX_C_SOURCE 200809L

#include "w4a32_quant_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct et_w4a32_aot_ctx {
  uint32_t max_m;
  uint32_t d_in;
  uint32_t d_hidden;
  uint32_t d_out;
  et_w4a32_format format;
  uint32_t group_size;
  size_t w1_packed_bytes;
  size_t w2_packed_bytes;
  float *x;
  float *w1;
  float *b1;
  float *w2;
  float *b2;
  float *w1_deq;
  float *w2_deq;
  uint8_t *w1_q;
  uint8_t *w2_q;
  float *h_buf;
  float *y_fused;
  float *y_ref;
  uint8_t y_sha256[ET_W4A32_SHA256_BYTES];
  uint8_t packed_sha256[ET_W4A32_SHA256_BYTES];
};

static float f32_from_f64(double value) {
  if (!isfinite(value))
    return (float)value;
  if (value > (double)FLT_MAX)
    return INFINITY;
  if (value < -(double)FLT_MAX)
    return -INFINITY;
  return (float)value;
}

static int64_t f32_bits_i64(float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return (int64_t)bits;
}

static int64_t digest_word_i64(const uint8_t digest[ET_W4A32_SHA256_BYTES],
                               int64_t word_index) {
  if (word_index < 0 || word_index >= 4)
    return -1;
  const size_t base = (size_t)word_index * 8u;
  uint64_t word = 0u;
  for (size_t i = 0; i < 8u; ++i)
    word = (word << 8u) | (uint64_t)digest[base + i];
  return (int64_t)word;
}

void et_w4a32_aot_destroy(et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  free(ctx->x);
  free(ctx->w1);
  free(ctx->b1);
  free(ctx->w2);
  free(ctx->b2);
  free(ctx->w1_deq);
  free(ctx->w2_deq);
  free(ctx->w1_q);
  free(ctx->w2_q);
  free(ctx->h_buf);
  free(ctx->y_fused);
  free(ctx->y_ref);
  free(ctx);
}

int32_t et_w4a32_aot_create(int64_t max_m, int64_t d_in, int64_t d_hidden,
                            int64_t d_out, int64_t format, int64_t group_size,
                            et_w4a32_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  if (max_m < 1 || max_m > (int64_t)ET_W4A32_MAX_M || d_in < 1 ||
      d_in > (int64_t)ET_W4A32_MAX_K || d_hidden < 1 ||
      d_hidden > (int64_t)ET_W4A32_MAX_N || d_out < 1 ||
      d_out > (int64_t)ET_W4A32_MAX_N || group_size < 1 ||
      group_size > (int64_t)ET_W4A32_MAX_K)
    return ET_W4A32_ERR_SHAPE;
  if (format != (int64_t)ET_W4A32_FORMAT_Q4_0 &&
      format != (int64_t)ET_W4A32_FORMAT_Q4_ASYM)
    return ET_W4A32_ERR_FORMAT;

  size_t w1_bytes = 0u;
  size_t w2_bytes = 0u;
  et_w4a32_status status = et_w4a32_packed_bytes(
      (et_w4a32_format)format, (uint32_t)group_size, (uint32_t)d_hidden,
      (uint32_t)d_in, &w1_bytes);
  if (status != ET_W4A32_OK)
    return status;
  status = et_w4a32_packed_bytes((et_w4a32_format)format, (uint32_t)group_size,
                                 (uint32_t)d_out, (uint32_t)d_hidden,
                                 &w2_bytes);
  if (status != ET_W4A32_OK)
    return status;

  et_w4a32_aot_ctx *ctx = (et_w4a32_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_W4A32_ERR_NOMEM;

  ctx->max_m = (uint32_t)max_m;
  ctx->d_in = (uint32_t)d_in;
  ctx->d_hidden = (uint32_t)d_hidden;
  ctx->d_out = (uint32_t)d_out;
  ctx->format = (et_w4a32_format)format;
  ctx->group_size = (uint32_t)group_size;
  ctx->w1_packed_bytes = w1_bytes;
  ctx->w2_packed_bytes = w2_bytes;

  ctx->x = (float *)calloc((size_t)ctx->max_m * ctx->d_in, sizeof(float));
  ctx->w1 = (float *)calloc((size_t)ctx->d_hidden * ctx->d_in, sizeof(float));
  ctx->b1 = (float *)calloc(ctx->d_hidden, sizeof(float));
  ctx->w2 = (float *)calloc((size_t)ctx->d_out * ctx->d_hidden, sizeof(float));
  ctx->b2 = (float *)calloc(ctx->d_out, sizeof(float));
  ctx->w1_deq =
      (float *)calloc((size_t)ctx->d_hidden * ctx->d_in, sizeof(float));
  ctx->w2_deq =
      (float *)calloc((size_t)ctx->d_out * ctx->d_hidden, sizeof(float));
  ctx->w1_q = (uint8_t *)calloc(w1_bytes, 1u);
  ctx->w2_q = (uint8_t *)calloc(w2_bytes, 1u);
  ctx->h_buf =
      (float *)calloc((size_t)ctx->max_m * ctx->d_hidden, sizeof(float));
  ctx->y_fused =
      (float *)calloc((size_t)ctx->max_m * ctx->d_out, sizeof(float));
  ctx->y_ref = (float *)calloc((size_t)ctx->max_m * ctx->d_out, sizeof(float));

  if (ctx->x == NULL || ctx->w1 == NULL || ctx->b1 == NULL || ctx->w2 == NULL ||
      ctx->b2 == NULL || ctx->w1_deq == NULL || ctx->w2_deq == NULL ||
      ctx->w1_q == NULL || ctx->w2_q == NULL || ctx->h_buf == NULL ||
      ctx->y_fused == NULL || ctx->y_ref == NULL) {
    et_w4a32_aot_destroy(ctx);
    return ET_W4A32_ERR_NOMEM;
  }

  *out_ctx = ctx;
  return ET_W4A32_OK;
}

int64_t et_w4a32_aot_x_count(const et_w4a32_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->max_m * (int64_t)ctx->d_in;
}

int64_t et_w4a32_aot_w1_count(const et_w4a32_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->d_hidden * (int64_t)ctx->d_in;
}

int64_t et_w4a32_aot_w2_count(const et_w4a32_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->d_out * (int64_t)ctx->d_hidden;
}

int64_t et_w4a32_aot_b1_count(const et_w4a32_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->d_hidden;
}

int64_t et_w4a32_aot_b2_count(const et_w4a32_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->d_out;
}

int64_t et_w4a32_aot_y_count(const et_w4a32_aot_ctx *ctx, int64_t m) {
  if (ctx == NULL || m < 1 || m > (int64_t)ctx->max_m)
    return -1;
  return m * (int64_t)ctx->d_out;
}

int64_t et_w4a32_aot_f32_weight_bytes(const et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  const int64_t elems =
      et_w4a32_aot_w1_count(ctx) + et_w4a32_aot_w2_count(ctx);
  return elems * (int64_t)sizeof(float);
}

int64_t et_w4a32_aot_packed_weight_bytes(const et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1;
  return (int64_t)(ctx->w1_packed_bytes + ctx->w2_packed_bytes);
}

static int32_t set_elem(float *buf, int64_t count, int64_t index,
                        double value) {
  if (buf == NULL || index < 0 || index >= count)
    return ET_W4A32_ERR_ARGUMENT;
  buf[index] = f32_from_f64(value);
  return ET_W4A32_OK;
}

static double get_elem(const float *buf, int64_t count, int64_t index) {
  if (buf == NULL || index < 0 || index >= count)
    return NAN;
  return (double)buf[index];
}

int32_t et_w4a32_aot_set_x(et_w4a32_aot_ctx *ctx, int64_t index, double value) {
  return ctx == NULL ? ET_W4A32_ERR_ARGUMENT
                     : set_elem(ctx->x, et_w4a32_aot_x_count(ctx), index, value);
}

int32_t et_w4a32_aot_set_w1(et_w4a32_aot_ctx *ctx, int64_t index,
                            double value) {
  return ctx == NULL
             ? ET_W4A32_ERR_ARGUMENT
             : set_elem(ctx->w1, et_w4a32_aot_w1_count(ctx), index, value);
}

int32_t et_w4a32_aot_set_b1(et_w4a32_aot_ctx *ctx, int64_t index,
                            double value) {
  return ctx == NULL
             ? ET_W4A32_ERR_ARGUMENT
             : set_elem(ctx->b1, et_w4a32_aot_b1_count(ctx), index, value);
}

int32_t et_w4a32_aot_set_w2(et_w4a32_aot_ctx *ctx, int64_t index,
                            double value) {
  return ctx == NULL
             ? ET_W4A32_ERR_ARGUMENT
             : set_elem(ctx->w2, et_w4a32_aot_w2_count(ctx), index, value);
}

int32_t et_w4a32_aot_set_b2(et_w4a32_aot_ctx *ctx, int64_t index,
                            double value) {
  return ctx == NULL
             ? ET_W4A32_ERR_ARGUMENT
             : set_elem(ctx->b2, et_w4a32_aot_b2_count(ctx), index, value);
}

double et_w4a32_aot_get_w1(const et_w4a32_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->w1, et_w4a32_aot_w1_count(ctx), index);
}

double et_w4a32_aot_get_w1_deq(const et_w4a32_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->w1_deq, et_w4a32_aot_w1_count(ctx), index);
}

double et_w4a32_aot_get_w2(const et_w4a32_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->w2, et_w4a32_aot_w2_count(ctx), index);
}

double et_w4a32_aot_get_w2_deq(const et_w4a32_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->w2_deq, et_w4a32_aot_w2_count(ctx), index);
}

int32_t et_w4a32_aot_quantize_all(et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  et_w4a32_status status;
  if (ctx->format == ET_W4A32_FORMAT_Q4_0) {
    status = et_w4a32_q4_0_quantize_v1(ctx->d_hidden, ctx->d_in, ctx->w1,
                                       ctx->w1_q, ctx->w1_packed_bytes);
    if (status != ET_W4A32_OK)
      return status;
    return et_w4a32_q4_0_quantize_v1(ctx->d_out, ctx->d_hidden, ctx->w2,
                                     ctx->w2_q, ctx->w2_packed_bytes);
  }
  status = et_w4a32_asym_quantize_v1(ctx->d_hidden, ctx->d_in, ctx->group_size,
                                     ctx->w1, ctx->w1_q, ctx->w1_packed_bytes);
  if (status != ET_W4A32_OK)
    return status;
  return et_w4a32_asym_quantize_v1(ctx->d_out, ctx->d_hidden, ctx->group_size,
                                   ctx->w2, ctx->w2_q, ctx->w2_packed_bytes);
}

int32_t et_w4a32_aot_dequantize_all(et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  et_w4a32_status status;
  if (ctx->format == ET_W4A32_FORMAT_Q4_0) {
    status = et_w4a32_q4_0_dequantize_v1(ctx->d_hidden, ctx->d_in, ctx->w1_q,
                                         ctx->w1_packed_bytes, ctx->w1_deq);
    if (status != ET_W4A32_OK)
      return status;
    return et_w4a32_q4_0_dequantize_v1(ctx->d_out, ctx->d_hidden, ctx->w2_q,
                                       ctx->w2_packed_bytes, ctx->w2_deq);
  }
  status = et_w4a32_asym_dequantize_v1(ctx->d_hidden, ctx->d_in,
                                       ctx->group_size, ctx->w1_q,
                                       ctx->w1_packed_bytes, ctx->w1_deq);
  if (status != ET_W4A32_OK)
    return status;
  return et_w4a32_asym_dequantize_v1(ctx->d_out, ctx->d_hidden, ctx->group_size,
                                     ctx->w2_q, ctx->w2_packed_bytes,
                                     ctx->w2_deq);
}

static void apply_relu_inplace(float *buf, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (buf[i] < 0.0f)
      buf[i] = 0.0f;
  }
}

int32_t et_w4a32_aot_mlp_forward_fused(et_w4a32_aot_ctx *ctx, int64_t m) {
  if (ctx == NULL || m < 1 || m > (int64_t)ctx->max_m)
    return ET_W4A32_ERR_SHAPE;
  const uint32_t mu = (uint32_t)m;
  et_w4a32_status status;
  if (ctx->format == ET_W4A32_FORMAT_Q4_0) {
    status = et_w4a32_q4_0_gemm_v1(mu, ctx->d_hidden, ctx->d_in, ctx->x,
                                   ctx->w1_q, ctx->w1_packed_bytes, ctx->b1,
                                   ctx->h_buf);
    if (status != ET_W4A32_OK)
      return status;
    apply_relu_inplace(ctx->h_buf, (size_t)mu * ctx->d_hidden);
    return et_w4a32_q4_0_gemm_v1(mu, ctx->d_out, ctx->d_hidden, ctx->h_buf,
                                 ctx->w2_q, ctx->w2_packed_bytes, ctx->b2,
                                 ctx->y_fused);
  }
  status = et_w4a32_asym_gemm_v1(mu, ctx->d_hidden, ctx->d_in, ctx->group_size,
                                 ctx->x, ctx->w1_q, ctx->w1_packed_bytes,
                                 ctx->b1, ctx->h_buf);
  if (status != ET_W4A32_OK)
    return status;
  apply_relu_inplace(ctx->h_buf, (size_t)mu * ctx->d_hidden);
  return et_w4a32_asym_gemm_v1(mu, ctx->d_out, ctx->d_hidden, ctx->group_size,
                               ctx->h_buf, ctx->w2_q, ctx->w2_packed_bytes,
                               ctx->b2, ctx->y_fused);
}

int32_t et_w4a32_aot_mlp_forward_deq_ref(et_w4a32_aot_ctx *ctx, int64_t m) {
  if (ctx == NULL || m < 1 || m > (int64_t)ctx->max_m)
    return ET_W4A32_ERR_SHAPE;
  const uint32_t mu = (uint32_t)m;
  et_w4a32_status status = et_w4a32_f32_gemm_v1(
      mu, ctx->d_hidden, ctx->d_in, ctx->x, ctx->w1_deq, ctx->b1, ctx->h_buf);
  if (status != ET_W4A32_OK)
    return status;
  apply_relu_inplace(ctx->h_buf, (size_t)mu * ctx->d_hidden);
  return et_w4a32_f32_gemm_v1(mu, ctx->d_out, ctx->d_hidden, ctx->h_buf,
                              ctx->w2_deq, ctx->b2, ctx->y_ref);
}

double et_w4a32_aot_get_y_fused(const et_w4a32_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL)
    return NAN;
  return get_elem(ctx->y_fused, (int64_t)ctx->max_m * (int64_t)ctx->d_out,
                  index);
}

double et_w4a32_aot_get_y_ref(const et_w4a32_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL)
    return NAN;
  return get_elem(ctx->y_ref, (int64_t)ctx->max_m * (int64_t)ctx->d_out, index);
}

int64_t et_w4a32_aot_y_fused_bits(const et_w4a32_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 ||
      index >= (int64_t)ctx->max_m * (int64_t)ctx->d_out)
    return -1;
  return f32_bits_i64(ctx->y_fused[index]);
}

int32_t et_w4a32_aot_compute_y_sha256(et_w4a32_aot_ctx *ctx, int64_t m) {
  if (ctx == NULL || m < 1 || m > (int64_t)ctx->max_m)
    return ET_W4A32_ERR_SHAPE;
  const size_t count = (size_t)m * (size_t)ctx->d_out;
  uint8_t *le = (uint8_t *)malloc(count * 4u);
  if (le == NULL)
    return ET_W4A32_ERR_NOMEM;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0u;
    memcpy(&bits, &ctx->y_fused[i], sizeof(bits));
    le[i * 4u + 0u] = (uint8_t)(bits & 0xffu);
    le[i * 4u + 1u] = (uint8_t)((bits >> 8u) & 0xffu);
    le[i * 4u + 2u] = (uint8_t)((bits >> 16u) & 0xffu);
    le[i * 4u + 3u] = (uint8_t)((bits >> 24u) & 0xffu);
  }
  const et_w4a32_status status =
      et_w4a32_sha256_bytes(le, count * 4u, ctx->y_sha256);
  free(le);
  return status;
}

int64_t et_w4a32_aot_y_sha256_word(const et_w4a32_aot_ctx *ctx,
                                   int64_t word_index) {
  if (ctx == NULL)
    return -1;
  return digest_word_i64(ctx->y_sha256, word_index);
}

int32_t et_w4a32_aot_compute_packed_sha256(et_w4a32_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  const size_t total = ctx->w1_packed_bytes + ctx->w2_packed_bytes;
  uint8_t *concat = (uint8_t *)malloc(total);
  if (concat == NULL)
    return ET_W4A32_ERR_NOMEM;
  memcpy(concat, ctx->w1_q, ctx->w1_packed_bytes);
  memcpy(concat + ctx->w1_packed_bytes, ctx->w2_q, ctx->w2_packed_bytes);
  const et_w4a32_status status =
      et_w4a32_sha256_bytes(concat, total, ctx->packed_sha256);
  free(concat);
  return status;
}

int64_t et_w4a32_aot_packed_sha256_word(const et_w4a32_aot_ctx *ctx,
                                        int64_t word_index) {
  if (ctx == NULL)
    return -1;
  return digest_word_i64(ctx->packed_sha256, word_index);
}

int64_t et_w4a32_aot_env_phase(void) {
  const char *phase = getenv("W4A32_AOT_PHASE");
  if (phase == NULL || phase[0] == '\0' || strcmp(phase, "both") == 0)
    return 0;
  if (strcmp(phase, "save") == 0)
    return 1;
  if (strcmp(phase, "reload") == 0)
    return 2;
  return -1;
}

static int format_slot_path(char *out, size_t cap, int64_t pair_id,
                            int layer_idx) {
  const char *dir = getenv("W4A32_AOT_DIR");
  if (dir == NULL || dir[0] == '\0')
    return 0;
  const int n = snprintf(out, cap, "%s/pair_%ld_layer_%d.etq4", dir,
                         (long)pair_id, layer_idx);
  return n > 0 && (size_t)n < cap;
}

int32_t et_w4a32_aot_save_env_pair(const et_w4a32_aot_ctx *ctx,
                                   int64_t pair_id) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  char p1[512];
  char p2[512];
  if (!format_slot_path(p1, sizeof(p1), pair_id, 1) ||
      !format_slot_path(p2, sizeof(p2), pair_id, 2))
    return ET_W4A32_ERR_ARGUMENT;
  et_w4a32_status status = et_w4a32_container_save(
      p1, ctx->format, ctx->group_size, ctx->d_hidden, ctx->d_in, ctx->w1_q,
      ctx->w1_packed_bytes);
  if (status != ET_W4A32_OK)
    return status;
  return et_w4a32_container_save(p2, ctx->format, ctx->group_size, ctx->d_out,
                                 ctx->d_hidden, ctx->w2_q,
                                 ctx->w2_packed_bytes);
}

int32_t et_w4a32_aot_load_env_pair(et_w4a32_aot_ctx *ctx, int64_t pair_id) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;
  char p1[512];
  char p2[512];
  if (!format_slot_path(p1, sizeof(p1), pair_id, 1) ||
      !format_slot_path(p2, sizeof(p2), pair_id, 2))
    return ET_W4A32_ERR_ARGUMENT;
  /* Poison destination packed buffers first so reload must populate them. */
  memset(ctx->w1_q, 0xA5, ctx->w1_packed_bytes);
  memset(ctx->w2_q, 0x5A, ctx->w2_packed_bytes);
  et_w4a32_container_meta meta1;
  et_w4a32_container_meta meta2;
  et_w4a32_status status = et_w4a32_container_load(
      p1, ctx->format, ctx->group_size, ctx->d_hidden, ctx->d_in, &meta1,
      ctx->w1_q, ctx->w1_packed_bytes);
  if (status != ET_W4A32_OK)
    return status;
  return et_w4a32_container_load(p2, ctx->format, ctx->group_size, ctx->d_out,
                                 ctx->d_hidden, &meta2, ctx->w2_q,
                                 ctx->w2_packed_bytes);
}

static int read_file_raw(const char *path, uint8_t *buf, size_t cap,
                         size_t *out_len) {
  const int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return 0;
  size_t used = 0u;
  int ok = 1;
  while (used < cap) {
    const ssize_t n = read(fd, buf + used, cap - used);
    if (n == 0)
      break;
    if (n < 0) {
      if (errno == EINTR)
        continue;
      ok = 0;
      break;
    }
    used += (size_t)n;
  }
  if (close(fd) != 0)
    ok = 0;
  if (ok)
    *out_len = used;
  return ok;
}

static int write_file_raw(const char *path, const uint8_t *buf, size_t len) {
  const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0)
    return 0;
  size_t off = 0u;
  int ok = 1;
  while (off < len) {
    const ssize_t n = write(fd, buf + off, len - off);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      ok = 0;
      break;
    }
    off += (size_t)n;
  }
  if (close(fd) != 0)
    ok = 0;
  return ok;
}

int32_t et_w4a32_aot_check_negative(const et_w4a32_aot_ctx *ctx,
                                    int64_t pair_id, int64_t kind) {
  if (ctx == NULL)
    return ET_W4A32_ERR_ARGUMENT;

  if (kind >= 1 && kind <= 7) {
    char p1[512];
    char tamper[512];
    const char *dir = getenv("W4A32_AOT_DIR");
    if (dir == NULL || dir[0] == '\0' ||
        !format_slot_path(p1, sizeof(p1), pair_id, 1))
      return ET_W4A32_ERR_ARGUMENT;
    const int tn = snprintf(tamper, sizeof(tamper), "%s/tamper_%ld.etq4", dir,
                            (long)kind);
    if (tn <= 0 || (size_t)tn >= sizeof(tamper))
      return ET_W4A32_ERR_ARGUMENT;

    uint8_t *dst = (uint8_t *)malloc(ctx->w1_packed_bytes);
    if (dst == NULL)
      return ET_W4A32_ERR_NOMEM;
    memset(dst, 0xCC, ctx->w1_packed_bytes);
    et_w4a32_container_meta meta;

    if (kind == 5) {
      (void)unlink(tamper);
      if (symlink(p1, tamper) != 0) {
        free(dst);
        return ET_W4A32_ERR_IO;
      }
      const et_w4a32_status rc = et_w4a32_container_load(
          tamper, ctx->format, ctx->group_size, ctx->d_hidden, ctx->d_in, &meta,
          dst, ctx->w1_packed_bytes);
      (void)unlink(tamper);
      free(dst);
      return rc;
    }
    if (kind == 6) {
      const et_w4a32_format wrong_fmt =
          (ctx->format == ET_W4A32_FORMAT_Q4_0) ? ET_W4A32_FORMAT_Q4_ASYM
                                                : ET_W4A32_FORMAT_Q4_0;
      const uint32_t wrong_g =
          (wrong_fmt == ET_W4A32_FORMAT_Q4_0) ? 32u : ctx->group_size;
      size_t wrong_bytes = 0u;
      (void)et_w4a32_packed_bytes(wrong_fmt, wrong_g, ctx->d_hidden, ctx->d_in,
                                  &wrong_bytes);
      uint8_t *dst2 = (uint8_t *)malloc(wrong_bytes);
      if (dst2 == NULL) {
        free(dst);
        return ET_W4A32_ERR_NOMEM;
      }
      const et_w4a32_status rc = et_w4a32_container_load(
          p1, wrong_fmt, wrong_g, ctx->d_hidden, ctx->d_in, &meta, dst2,
          wrong_bytes);
      free(dst2);
      free(dst);
      return rc;
    }
    if (kind == 7) {
      const uint32_t wrong_n = ctx->d_hidden + 1u;
      size_t wrong_bytes = 0u;
      (void)et_w4a32_packed_bytes(ctx->format, ctx->group_size, wrong_n,
                                  ctx->d_in, &wrong_bytes);
      uint8_t *dst2 = (uint8_t *)malloc(wrong_bytes);
      if (dst2 == NULL) {
        free(dst);
        return ET_W4A32_ERR_NOMEM;
      }
      const et_w4a32_status rc = et_w4a32_container_load(
          p1, ctx->format, ctx->group_size, wrong_n, ctx->d_in, &meta, dst2,
          wrong_bytes);
      free(dst2);
      free(dst);
      return rc;
    }

    const size_t cap = ET_W4A32_CONTAINER_MIN_BYTES + ctx->w1_packed_bytes + 16u;
    uint8_t *raw = (uint8_t *)malloc(cap);
    if (raw == NULL) {
      free(dst);
      return ET_W4A32_ERR_NOMEM;
    }
    size_t len = 0u;
    if (!read_file_raw(p1, raw, cap, &len) || len < ET_W4A32_CONTAINER_MIN_BYTES) {
      free(raw);
      free(dst);
      return ET_W4A32_ERR_IO;
    }
    if (kind == 1)
      raw[ET_W4A32_CONTAINER_HEADER_BYTES] ^= 0x01u;
    else if (kind == 2)
      len -= 7u;
    else if (kind == 3)
      raw[0] = 'X';
    else if (kind == 4)
      raw[8] = 2u;

    (void)unlink(tamper);
    if (!write_file_raw(tamper, raw, len)) {
      free(raw);
      free(dst);
      return ET_W4A32_ERR_IO;
    }
    const et_w4a32_status rc = et_w4a32_container_load(
        tamper, ctx->format, ctx->group_size, ctx->d_hidden, ctx->d_in, &meta,
        dst, ctx->w1_packed_bytes);
    (void)unlink(tamper);
    free(raw);
    free(dst);
    return rc;
  }

  if (kind == 8) {
    /* K % 32 != 0 rejection. */
    float w_bad[30] = {0.0f};
    uint8_t out_bad[18] = {0u};
    return et_w4a32_q4_0_quantize_v1(1u, 30u, w_bad, out_bad, sizeof(out_bad));
  }
  if (kind == 9) {
    /* f16 scale overflow rejection: max_abs / 8 >= 65520 -> max_abs = 600000. */
    float w_big[32] = {0.0f};
    uint8_t out_q[18] = {0u};
    w_big[0] = 600000.0f;
    return et_w4a32_q4_0_quantize_v1(1u, 32u, w_big, out_q, sizeof(out_q));
  }
  if (kind == 10) {
    /* Non-finite f16 scale (+Inf = 0x7c00) in GEMM; verify sentinel preserved. */
    float x_small[32] = {0.0f};
    uint8_t blk[18] = {0u};
    blk[0] = 0x00u;
    blk[1] = 0x7cu;
    float y_out[1] = {42.5f};
    const et_w4a32_status rc =
        et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, x_small, blk, sizeof(blk), NULL, y_out);
    if (y_out[0] != 42.5f)
      return ET_W4A32_ERR_IO;
    return rc;
  }
  if (kind == 11) {
    /* Non-finite X in GEMM; verify sentinel preserved. */
    float x_nan[32] = {0.0f};
    x_nan[5] = NAN;
    uint8_t blk[18] = {0u};
    float y_out[1] = {-13.25f};
    const et_w4a32_status rc =
        et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, x_nan, blk, sizeof(blk), NULL, y_out);
    if (y_out[0] != -13.25f)
      return ET_W4A32_ERR_IO;
    return rc;
  }
  if (kind == 12) {
    /* Aliased X and Y in GEMM; verify sentinel preserved. */
    float shared[32] = {0.0f};
    shared[0] = 7.75f;
    uint8_t blk[18] = {0u};
    const et_w4a32_status rc = et_w4a32_q4_0_gemm_v1(1u, 1u, 32u, shared, blk,
                                                     sizeof(blk), NULL, shared);
    if (shared[0] != 7.75f)
      return ET_W4A32_ERR_IO;
    return rc;
  }

  return ET_W4A32_ERR_ARGUMENT;
}
