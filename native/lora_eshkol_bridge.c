#define _POSIX_C_SOURCE 200809L

#include "lora_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct et_lora_aot_ctx {
  et_lora_config cfg;
  float x[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float w_base[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_merged[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float w_restored[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float target[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  float y[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float y_base[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float y_merged[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dy[ET_LORA_MAX_M * ET_LORA_MAX_N];
  float dx[ET_LORA_MAX_M * ET_LORA_MAX_K];
  float da[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float db[ET_LORA_MAX_N * ET_LORA_MAX_R];
  uint8_t base_sha256[ET_LORA_SHA256_BYTES];
  uint8_t y_sha256[ET_LORA_SHA256_BYTES];
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

static int64_t digest_word_i64(const uint8_t digest[ET_LORA_SHA256_BYTES],
                               int64_t word_index) {
  if (word_index < 0 || word_index >= 4)
    return -1;
  const size_t base = (size_t)word_index * 8u;
  uint64_t word = 0u;
  for (size_t i = 0; i < 8u; ++i)
    word = (word << 8u) | (uint64_t)digest[base + i];
  return (int64_t)word;
}

int32_t et_lora_aot_create(int64_t m, int64_t k, int64_t n, int64_t r,
                           double alpha, et_lora_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  if (m < 1 || m > (int64_t)ET_LORA_MAX_M || k < 1 ||
      k > (int64_t)ET_LORA_MAX_K || n < 1 || n > (int64_t)ET_LORA_MAX_N ||
      r < 1 || r > (int64_t)ET_LORA_MAX_R)
    return ET_LORA_ERR_SHAPE;

  et_lora_config cfg;
  cfg.m = (uint32_t)m;
  cfg.k = (uint32_t)k;
  cfg.n = (uint32_t)n;
  cfg.r = (uint32_t)r;
  cfg.alpha = f32_from_f64(alpha);
  const et_lora_status status = et_lora_validate_config(&cfg);
  if (status != ET_LORA_OK)
    return status;

  et_lora_aot_ctx *ctx = (et_lora_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_LORA_ERR_NOMEM;
  ctx->cfg = cfg;
  *out_ctx = ctx;
  return ET_LORA_OK;
}

void et_lora_aot_destroy(et_lora_aot_ctx *ctx) { free(ctx); }

int64_t et_lora_aot_x_count(const et_lora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->cfg.m * (int64_t)ctx->cfg.k;
}

int64_t et_lora_aot_w_count(const et_lora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->cfg.n * (int64_t)ctx->cfg.k;
}

int64_t et_lora_aot_a_count(const et_lora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->cfg.r * (int64_t)ctx->cfg.k;
}

int64_t et_lora_aot_b_count(const et_lora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->cfg.n * (int64_t)ctx->cfg.r;
}

int64_t et_lora_aot_y_count(const et_lora_aot_ctx *ctx) {
  return ctx == NULL ? -1 : (int64_t)ctx->cfg.m * (int64_t)ctx->cfg.n;
}

int32_t et_lora_aot_init_adapter(et_lora_aot_ctx *ctx, int64_t seed) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_init_adapter(&ctx->cfg, (uint64_t)seed, ctx->a, ctx->b);
}

static int32_t set_elem(float *buf, int64_t count, int64_t index,
                        double value) {
  if (buf == NULL || index < 0 || index >= count)
    return ET_LORA_ERR_ARGUMENT;
  buf[index] = f32_from_f64(value);
  return ET_LORA_OK;
}

static double get_elem(const float *buf, int64_t count, int64_t index) {
  if (buf == NULL || index < 0 || index >= count)
    return NAN;
  return (double)buf[index];
}

int32_t et_lora_aot_set_x(et_lora_aot_ctx *ctx, int64_t index, double value) {
  return ctx == NULL ? ET_LORA_ERR_ARGUMENT
                     : set_elem(ctx->x, et_lora_aot_x_count(ctx), index, value);
}

int32_t et_lora_aot_set_w_base(et_lora_aot_ctx *ctx, int64_t index,
                               double value) {
  return ctx == NULL
             ? ET_LORA_ERR_ARGUMENT
             : set_elem(ctx->w_base, et_lora_aot_w_count(ctx), index, value);
}

int32_t et_lora_aot_set_target(et_lora_aot_ctx *ctx, int64_t index,
                               double value) {
  return ctx == NULL
             ? ET_LORA_ERR_ARGUMENT
             : set_elem(ctx->target, et_lora_aot_y_count(ctx), index, value);
}

int32_t et_lora_aot_set_a(et_lora_aot_ctx *ctx, int64_t index, double value) {
  return ctx == NULL ? ET_LORA_ERR_ARGUMENT
                     : set_elem(ctx->a, et_lora_aot_a_count(ctx), index, value);
}

int32_t et_lora_aot_set_b(et_lora_aot_ctx *ctx, int64_t index, double value) {
  return ctx == NULL ? ET_LORA_ERR_ARGUMENT
                     : set_elem(ctx->b, et_lora_aot_b_count(ctx), index, value);
}

int32_t et_lora_aot_set_dy(et_lora_aot_ctx *ctx, int64_t index, double value) {
  return ctx == NULL
             ? ET_LORA_ERR_ARGUMENT
             : set_elem(ctx->dy, et_lora_aot_y_count(ctx), index, value);
}

double et_lora_aot_get_a(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->a, et_lora_aot_a_count(ctx), index);
}

double et_lora_aot_get_b(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->b, et_lora_aot_b_count(ctx), index);
}

double et_lora_aot_get_y(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->y, et_lora_aot_y_count(ctx), index);
}

double et_lora_aot_get_y_base(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->y_base, et_lora_aot_y_count(ctx), index);
}

double et_lora_aot_get_y_merged(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->y_merged, et_lora_aot_y_count(ctx), index);
}

double et_lora_aot_get_target(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->target, et_lora_aot_y_count(ctx), index);
}

double et_lora_aot_get_dx(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->dx, et_lora_aot_x_count(ctx), index);
}

double et_lora_aot_get_da(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->da, et_lora_aot_a_count(ctx), index);
}

double et_lora_aot_get_db(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN : get_elem(ctx->db, et_lora_aot_b_count(ctx), index);
}

double et_lora_aot_get_w_base(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL ? NAN
                     : get_elem(ctx->w_base, et_lora_aot_w_count(ctx), index);
}

double et_lora_aot_get_w_restored(const et_lora_aot_ctx *ctx, int64_t index) {
  return ctx == NULL
             ? NAN
             : get_elem(ctx->w_restored, et_lora_aot_w_count(ctx), index);
}

int64_t et_lora_aot_w_base_bits(const et_lora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_lora_aot_w_count(ctx))
    return -1;
  return f32_bits_i64(ctx->w_base[index]);
}

int64_t et_lora_aot_y_bits(const et_lora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_lora_aot_y_count(ctx))
    return -1;
  return f32_bits_i64(ctx->y[index]);
}

int64_t et_lora_aot_y_base_bits(const et_lora_aot_ctx *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= et_lora_aot_y_count(ctx))
    return -1;
  return f32_bits_i64(ctx->y_base[index]);
}

int32_t et_lora_aot_base_forward(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_base_forward(&ctx->cfg, ctx->x, ctx->w_base, ctx->y_base);
}

int32_t et_lora_aot_forward(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_forward(&ctx->cfg, ctx->x, ctx->w_base, ctx->a, ctx->b,
                         ctx->y);
}

int32_t et_lora_aot_backward(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_backward(&ctx->cfg, ctx->x, ctx->w_base, ctx->a, ctx->b,
                          ctx->dy, ctx->dx, ctx->da, ctx->db);
}

int32_t et_lora_aot_merge_and_forward(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  et_lora_status status =
      et_lora_merge(&ctx->cfg, ctx->w_base, ctx->a, ctx->b, ctx->w_merged);
  if (status != ET_LORA_OK)
    return status;
  return et_lora_base_forward(&ctx->cfg, ctx->x, ctx->w_merged, ctx->y_merged);
}

int32_t et_lora_aot_unmerge(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_unmerge(&ctx->cfg, ctx->w_merged, ctx->a, ctx->b,
                         ctx->w_restored);
}

int32_t et_lora_aot_compute_base_sha256(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_base_weight_sha256(ctx->cfg.k, ctx->cfg.n, ctx->w_base,
                                    ctx->base_sha256);
}

int64_t et_lora_aot_base_sha256_word(const et_lora_aot_ctx *ctx,
                                     int64_t word_index) {
  if (ctx == NULL)
    return -1;
  return digest_word_i64(ctx->base_sha256, word_index);
}

int32_t et_lora_aot_compute_y_sha256(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  /* Reuse canonical little-endian binary32 SHA-256 over [m, n] output tensor. */
  return et_lora_base_weight_sha256(ctx->cfg.n, ctx->cfg.m, ctx->y,
                                    ctx->y_sha256);
}

int64_t et_lora_aot_y_sha256_word(const et_lora_aot_ctx *ctx,
                                  int64_t word_index) {
  if (ctx == NULL)
    return -1;
  return digest_word_i64(ctx->y_sha256, word_index);
}

int64_t et_lora_aot_env_phase(void) {
  const char *phase = getenv("LORA_AOT_PHASE");
  if (phase == NULL || phase[0] == '\0' || strcmp(phase, "both") == 0)
    return 0;
  if (strcmp(phase, "train") == 0)
    return 1;
  if (strcmp(phase, "reload") == 0)
    return 2;
  return -1;
}

int32_t et_lora_aot_save_env_checkpoint(const et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  const char *path = getenv("LORA_AOT_CKPT");
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  return et_lora_checkpoint_save(path, &ctx->cfg, ctx->w_base, ctx->a, ctx->b);
}

int32_t et_lora_aot_load_env_checkpoint(et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  const char *path = getenv("LORA_AOT_CKPT");
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  et_lora_checkpoint_meta meta;
  const et_lora_status status =
      et_lora_checkpoint_load(path, ctx->cfg.k, ctx->cfg.n, ctx->cfg.r,
                              ctx->w_base, &meta, ctx->a, ctx->b);
  if (status != ET_LORA_OK)
    return status;
  ctx->cfg.alpha = meta.alpha;
  return ET_LORA_OK;
}

int32_t et_lora_aot_check_wrong_base_hash(const et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  const char *path = getenv("LORA_AOT_CKPT");
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  float wrong_w[ET_LORA_MAX_N * ET_LORA_MAX_K];
  float tmp_a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float tmp_b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  const size_t w_count = (size_t)ctx->cfg.n * (size_t)ctx->cfg.k;
  memcpy(wrong_w, ctx->w_base, w_count * sizeof(float));
  wrong_w[0] = wrong_w[0] + 0.125f;
  et_lora_checkpoint_meta meta;
  return et_lora_checkpoint_load(path, ctx->cfg.k, ctx->cfg.n, ctx->cfg.r,
                                 wrong_w, &meta, tmp_a, tmp_b);
}

int32_t et_lora_aot_check_wrong_rank(const et_lora_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  const char *path = getenv("LORA_AOT_CKPT");
  if (path == NULL || path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;
  const uint32_t alt_r = ctx->cfg.r > 1u ? (ctx->cfg.r - 1u) : 2u;
  float tmp_a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float tmp_b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  et_lora_checkpoint_meta meta;
  return et_lora_checkpoint_load(path, ctx->cfg.k, ctx->cfg.n, alt_r,
                                 ctx->w_base, &meta, tmp_a, tmp_b);
}

static int read_entire_file(const char *path, uint8_t *buf, size_t cap,
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

static int write_entire_file(const char *path, const uint8_t *buf, size_t len) {
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

/* tamper_kind:
 *   1 = corrupt payload byte (ET_LORA_ERR_CHECKSUM)
 *   2 = truncated file (ET_LORA_ERR_TRUNCATED)
 *   3 = bad magic (ET_LORA_ERR_FORMAT)
 *   4 = bad version (ET_LORA_ERR_VERSION)
 *   5 = symlink path (ET_LORA_ERR_SYMLINK)
 */
int32_t et_lora_aot_check_tampered_env(const et_lora_aot_ctx *ctx,
                                       int64_t tamper_kind) {
  if (ctx == NULL)
    return ET_LORA_ERR_ARGUMENT;
  const char *ckpt_path = getenv("LORA_AOT_CKPT");
  const char *tamper_path = getenv("LORA_AOT_TAMPER");
  if (ckpt_path == NULL || ckpt_path[0] == '\0' || tamper_path == NULL ||
      tamper_path[0] == '\0')
    return ET_LORA_ERR_ARGUMENT;

  float tmp_a[ET_LORA_MAX_R * ET_LORA_MAX_K];
  float tmp_b[ET_LORA_MAX_N * ET_LORA_MAX_R];
  et_lora_checkpoint_meta meta;

  if (tamper_kind == 5) {
    (void)unlink(tamper_path);
    if (symlink(ckpt_path, tamper_path) != 0)
      return ET_LORA_ERR_IO;
    const et_lora_status rc = et_lora_checkpoint_load(
        tamper_path, ctx->cfg.k, ctx->cfg.n, ctx->cfg.r, ctx->w_base, &meta,
        tmp_a, tmp_b);
    (void)unlink(tamper_path);
    return rc;
  }

  uint8_t raw[ET_LORA_CHECKPOINT_MAX_BYTES];
  size_t len = 0u;
  if (!read_entire_file(ckpt_path, raw, sizeof(raw), &len) ||
      len < ET_LORA_CHECKPOINT_MIN_BYTES)
    return ET_LORA_ERR_IO;

  if (tamper_kind == 1) {
    raw[ET_LORA_CHECKPOINT_HEADER_BYTES] ^= 0x01u;
  } else if (tamper_kind == 2) {
    len -= 5u;
  } else if (tamper_kind == 3) {
    raw[0] = 'X';
  } else if (tamper_kind == 4) {
    raw[8] = 2u;
  } else {
    return ET_LORA_ERR_ARGUMENT;
  }

  (void)unlink(tamper_path);
  if (!write_entire_file(tamper_path, raw, len))
    return ET_LORA_ERR_IO;
  const et_lora_status rc = et_lora_checkpoint_load(
      tamper_path, ctx->cfg.k, ctx->cfg.n, ctx->cfg.r, ctx->w_base, &meta,
      tmp_a, tmp_b);
  (void)unlink(tamper_path);
  return rc;
}
