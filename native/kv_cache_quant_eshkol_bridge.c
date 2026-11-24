#define _GNU_SOURCE

#include "kv_cache_quant_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

struct et_kvq_aot_ctx {
  et_kvq_config cfg;
  et_kvq_cache *cache;
  float *k_stage;   /* [num_kv_heads * head_dim] */
  float *v_stage;   /* [num_kv_heads * head_dim] */
  float *deq_k;     /* [num_kv_heads * head_dim] */
  float *deq_v;     /* [num_kv_heads * head_dim] */
  float *q_stage;   /* [ET_KVQ_MAX_Q_HEADS * head_dim] */
  float *out_stage; /* [ET_KVQ_MAX_Q_HEADS * head_dim] */
};

static int build_snapshot_slot_path(int64_t slot, char *buf, size_t buf_len) {
  if (slot < 0 || slot > 7)
    return -1;
  const char *dir = getenv("ET_KVQ_SNAPSHOT_DIR");
  if (dir == NULL || dir[0] == '\0')
    return -1;
  const int n = snprintf(buf, buf_len, "%s/etkq_slot_%ld.bin", dir, (long)slot);
  if (n <= 0 || (size_t)n >= buf_len)
    return -1;
  return 0;
}

int32_t et_kvq_aot_create(int64_t format, int64_t num_layers,
                          int64_t num_kv_heads, int64_t head_dim,
                          int64_t max_seq_len, et_kvq_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  if (format < 0 || format > (int64_t)UINT32_MAX || num_layers < 0 ||
      num_layers > (int64_t)UINT32_MAX || num_kv_heads < 0 ||
      num_kv_heads > (int64_t)UINT32_MAX || head_dim < 0 ||
      head_dim > (int64_t)UINT32_MAX || max_seq_len < 0 ||
      max_seq_len > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;

  et_kvq_config cfg;
  cfg.format = (et_kvq_format)format;
  cfg.num_layers = (uint32_t)num_layers;
  cfg.num_kv_heads = (uint32_t)num_kv_heads;
  cfg.head_dim = (uint32_t)head_dim;
  cfg.max_seq_len = (uint32_t)max_seq_len;
  cfg.reserved = 0u;

  const et_kvq_status vst = et_kvq_validate_config(&cfg);
  if (vst != ET_KVQ_OK)
    return vst;

  et_kvq_aot_ctx *ctx = (et_kvq_aot_ctx *)calloc(1u, sizeof(et_kvq_aot_ctx));
  if (ctx == NULL)
    return ET_KVQ_ERR_ALLOC;

  const size_t kv_floats = (size_t)cfg.num_kv_heads * (size_t)cfg.head_dim;
  const size_t q_floats = (size_t)ET_KVQ_MAX_Q_HEADS * (size_t)cfg.head_dim;

  ctx->cfg = cfg;
  ctx->k_stage = (float *)calloc(kv_floats, sizeof(float));
  ctx->v_stage = (float *)calloc(kv_floats, sizeof(float));
  ctx->deq_k = (float *)calloc(kv_floats, sizeof(float));
  ctx->deq_v = (float *)calloc(kv_floats, sizeof(float));
  ctx->q_stage = (float *)calloc(q_floats, sizeof(float));
  ctx->out_stage = (float *)calloc(q_floats, sizeof(float));

  if (ctx->k_stage == NULL || ctx->v_stage == NULL || ctx->deq_k == NULL ||
      ctx->deq_v == NULL || ctx->q_stage == NULL || ctx->out_stage == NULL) {
    et_kvq_aot_destroy(ctx);
    return ET_KVQ_ERR_ALLOC;
  }

  const et_kvq_status cst = et_kvq_cache_create(&cfg, NULL, 0u, &ctx->cache);
  if (cst != ET_KVQ_OK) {
    et_kvq_aot_destroy(ctx);
    return cst;
  }

  *out_ctx = ctx;
  return ET_KVQ_OK;
}

void et_kvq_aot_destroy(et_kvq_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  et_kvq_cache_destroy(ctx->cache);
  free(ctx->k_stage);
  free(ctx->v_stage);
  free(ctx->deq_k);
  free(ctx->deq_v);
  free(ctx->q_stage);
  free(ctx->out_stage);
  free(ctx);
}

int32_t et_kvq_aot_reset(et_kvq_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_cache_reset(ctx->cache);
}

int32_t et_kvq_aot_truncate(et_kvq_aot_ctx *ctx, int64_t new_seq_len) {
  if (ctx == NULL || new_seq_len < 0 || new_seq_len > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_cache_truncate(ctx->cache, (uint32_t)new_seq_len);
}

int64_t et_kvq_aot_layer_seq_len(const et_kvq_aot_ctx *ctx, int64_t layer) {
  if (ctx == NULL || layer < 0 || layer > (int64_t)UINT32_MAX)
    return (int64_t)ET_KVQ_ERR_INVALID_ARG;
  uint32_t len = 0u;
  const et_kvq_status st =
      et_kvq_cache_layer_seq_len(ctx->cache, (uint32_t)layer, &len);
  return (st == ET_KVQ_OK) ? (int64_t)len : (int64_t)st;
}

int64_t et_kvq_aot_stat_fp32_bytes(const et_kvq_aot_ctx *ctx) {
  if (ctx == NULL)
    return (int64_t)ET_KVQ_ERR_INVALID_ARG;
  et_kvq_memory_stats st_mem;
  const et_kvq_status st = et_kvq_cache_memory_stats(ctx->cache, &st_mem);
  return (st == ET_KVQ_OK) ? (int64_t)st_mem.fp32_baseline_bytes : (int64_t)st;
}

int64_t et_kvq_aot_stat_quant_bytes(const et_kvq_aot_ctx *ctx) {
  if (ctx == NULL)
    return (int64_t)ET_KVQ_ERR_INVALID_ARG;
  et_kvq_memory_stats st_mem;
  const et_kvq_status st = et_kvq_cache_memory_stats(ctx->cache, &st_mem);
  return (st == ET_KVQ_OK) ? (int64_t)st_mem.quantized_bytes : (int64_t)st;
}

double et_kvq_aot_stat_compression_ratio(const et_kvq_aot_ctx *ctx) {
  if (ctx == NULL)
    return -1.0;
  et_kvq_memory_stats st_mem;
  if (et_kvq_cache_memory_stats(ctx->cache, &st_mem) != ET_KVQ_OK)
    return -1.0;
  return st_mem.compression_ratio;
}

int32_t et_kvq_aot_set_k(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                         double value) {
  if (ctx == NULL || kv_head < 0 || dim < 0 ||
      (uint64_t)kv_head >= (uint64_t)ctx->cfg.num_kv_heads ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  ctx->k_stage[(size_t)kv_head * (size_t)ctx->cfg.head_dim + (size_t)dim] =
      (float)value;
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_set_v(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                         double value) {
  if (ctx == NULL || kv_head < 0 || dim < 0 ||
      (uint64_t)kv_head >= (uint64_t)ctx->cfg.num_kv_heads ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  ctx->v_stage[(size_t)kv_head * (size_t)ctx->cfg.head_dim + (size_t)dim] =
      (float)value;
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_set_k_bits(et_kvq_aot_ctx *ctx, int64_t kv_head, int64_t dim,
                              int64_t f32_bits) {
  if (ctx == NULL || kv_head < 0 || dim < 0 ||
      (uint64_t)kv_head >= (uint64_t)ctx->cfg.num_kv_heads ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  const uint32_t u = (uint32_t)(f32_bits & INT64_C(0xFFFFFFFF));
  float fv = 0.0f;
  memcpy(&fv, &u, sizeof(fv));
  ctx->k_stage[(size_t)kv_head * (size_t)ctx->cfg.head_dim + (size_t)dim] = fv;
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_append_token(et_kvq_aot_ctx *ctx, int64_t layer) {
  if (ctx == NULL || layer < 0 || layer > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_append_token_v1(ctx->cache, (uint32_t)layer, ctx->k_stage,
                                ctx->v_stage);
}

int32_t et_kvq_aot_dequant_slice(et_kvq_aot_ctx *ctx, int64_t layer,
                                 int64_t pos) {
  if (ctx == NULL || layer < 0 || layer > (int64_t)UINT32_MAX || pos < 0 ||
      pos > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_dequant_slice_v1(ctx->cache, (uint32_t)layer, (uint32_t)pos,
                                 ctx->deq_k, ctx->deq_v);
}

double et_kvq_aot_get_deq_k(const et_kvq_aot_ctx *ctx, int64_t kv_head,
                            int64_t dim) {
  if (ctx == NULL || kv_head < 0 || dim < 0 ||
      (uint64_t)kv_head >= (uint64_t)ctx->cfg.num_kv_heads ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return 0.0;
  return (double)ctx->deq_k[(size_t)kv_head * (size_t)ctx->cfg.head_dim +
                            (size_t)dim];
}

double et_kvq_aot_get_deq_v(const et_kvq_aot_ctx *ctx, int64_t kv_head,
                            int64_t dim) {
  if (ctx == NULL || kv_head < 0 || dim < 0 ||
      (uint64_t)kv_head >= (uint64_t)ctx->cfg.num_kv_heads ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return 0.0;
  return (double)ctx->deq_v[(size_t)kv_head * (size_t)ctx->cfg.head_dim +
                            (size_t)dim];
}

int32_t et_kvq_aot_copy_dequant_to_fp32(et_kvq_aot_ctx *src_ctx,
                                        et_kvq_aot_ctx *dst_fp32_ctx,
                                        int64_t layer, int64_t pos) {
  if (src_ctx == NULL || dst_fp32_ctx == NULL || layer < 0 ||
      layer > (int64_t)UINT32_MAX || pos < 0 || pos > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;
  if (dst_fp32_ctx->cfg.format != ET_KVQ_FMT_FP32)
    return ET_KVQ_ERR_FORMAT_MISMATCH;
  if (src_ctx->cfg.num_kv_heads != dst_fp32_ctx->cfg.num_kv_heads ||
      src_ctx->cfg.head_dim != dst_fp32_ctx->cfg.head_dim)
    return ET_KVQ_ERR_SHAPE;

  const et_kvq_status dst =
      et_kvq_dequant_slice_v1(src_ctx->cache, (uint32_t)layer, (uint32_t)pos,
                              src_ctx->deq_k, src_ctx->deq_v);
  if (dst != ET_KVQ_OK)
    return dst;
  return et_kvq_append_token_v1(dst_fp32_ctx->cache, (uint32_t)layer,
                                src_ctx->deq_k, src_ctx->deq_v);
}

int32_t et_kvq_aot_set_q(et_kvq_aot_ctx *ctx, int64_t q_head, int64_t dim,
                         double value) {
  if (ctx == NULL || q_head < 0 || dim < 0 ||
      (uint64_t)q_head >= (uint64_t)ET_KVQ_MAX_Q_HEADS ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  ctx->q_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim + (size_t)dim] =
      (float)value;
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_set_q_bits(et_kvq_aot_ctx *ctx, int64_t q_head, int64_t dim,
                              int64_t f32_bits) {
  if (ctx == NULL || q_head < 0 || dim < 0 ||
      (uint64_t)q_head >= (uint64_t)ET_KVQ_MAX_Q_HEADS ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return ET_KVQ_ERR_OUT_OF_RANGE;
  const uint32_t u = (uint32_t)(f32_bits & INT64_C(0xFFFFFFFF));
  float fv = 0.0f;
  memcpy(&fv, &u, sizeof(fv));
  ctx->q_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim + (size_t)dim] = fv;
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_decode_gqa(et_kvq_aot_ctx *ctx, int64_t layer,
                              int64_t num_q_heads, double scale,
                              int64_t window_size) {
  if (ctx == NULL || layer < 0 || layer > (int64_t)UINT32_MAX ||
      num_q_heads < 0 || num_q_heads > (int64_t)ET_KVQ_MAX_Q_HEADS ||
      window_size < 0 || window_size > (int64_t)UINT32_MAX)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_decode_gqa_v1(ctx->cache, (uint32_t)layer,
                              (uint32_t)num_q_heads, (float)scale,
                              (uint32_t)window_size, ctx->q_stage,
                              ctx->out_stage);
}

double et_kvq_aot_get_out(const et_kvq_aot_ctx *ctx, int64_t q_head,
                          int64_t dim) {
  if (ctx == NULL || q_head < 0 || dim < 0 ||
      (uint64_t)q_head >= (uint64_t)ET_KVQ_MAX_Q_HEADS ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return 0.0;
  return (double)ctx->out_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim +
                                (size_t)dim];
}

int64_t et_kvq_aot_get_out_bits(const et_kvq_aot_ctx *ctx, int64_t q_head,
                                int64_t dim) {
  if (ctx == NULL || q_head < 0 || dim < 0 ||
      (uint64_t)q_head >= (uint64_t)ET_KVQ_MAX_Q_HEADS ||
      (uint64_t)dim >= (uint64_t)ctx->cfg.head_dim)
    return -1;
  const float fv =
      ctx->out_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim + (size_t)dim];
  uint32_t u = 0u;
  memcpy(&u, &fv, sizeof(u));
  return (int64_t)u;
}

int32_t et_kvq_aot_save_snapshot_slot(const et_kvq_aot_ctx *ctx, int64_t slot) {
  if (ctx == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  char path[PATH_MAX];
  if (build_snapshot_slot_path(slot, path, sizeof(path)) != 0)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_cache_save_snapshot(ctx->cache, path);
}

int32_t et_kvq_aot_load_snapshot_slot(et_kvq_aot_ctx *ctx, int64_t slot) {
  if (ctx == NULL)
    return ET_KVQ_ERR_INVALID_ARG;
  char path[PATH_MAX];
  if (build_snapshot_slot_path(slot, path, sizeof(path)) != 0)
    return ET_KVQ_ERR_INVALID_ARG;
  return et_kvq_cache_load_snapshot(ctx->cache, path);
}

int32_t et_kvq_aot_tamper_snapshot_slot(int64_t slot, int64_t byte_offset) {
  if (byte_offset < 0)
    return ET_KVQ_ERR_INVALID_ARG;
  char path[PATH_MAX];
  if (build_snapshot_slot_path(slot, path, sizeof(path)) != 0)
    return ET_KVQ_ERR_INVALID_ARG;

  const int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return ET_KVQ_ERR_IO;

  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) ||
      byte_offset >= (int64_t)st.st_size) {
    close(fd);
    return ET_KVQ_ERR_IO;
  }

  uint8_t b = 0u;
  if (pread(fd, &b, 1u, (off_t)byte_offset) != 1) {
    close(fd);
    return ET_KVQ_ERR_IO;
  }
  b ^= 0x5Au;
  if (pwrite(fd, &b, 1u, (off_t)byte_offset) != 1) {
    close(fd);
    return ET_KVQ_ERR_IO;
  }
  close(fd);
  return ET_KVQ_OK;
}

int32_t et_kvq_aot_check_fp_env_trap(et_kvq_aot_ctx *ctx, int64_t layer,
                                     int64_t num_q_heads, double scale) {
  if (ctx == NULL || layer < 0 || layer > (int64_t)UINT32_MAX ||
      num_q_heads < 1 || num_q_heads > (int64_t)ET_KVQ_MAX_Q_HEADS)
    return ET_KVQ_ERR_INVALID_ARG;

  const int saved_round = fegetround();
  if (fesetround(FE_TOWARDZERO) != 0)
    return ET_KVQ_ERR_STATE;

  const et_kvq_status st_append = et_kvq_append_token_v1(
      ctx->cache, (uint32_t)layer, ctx->k_stage, ctx->v_stage);
  const et_kvq_status st_decode = et_kvq_decode_gqa_v1(
      ctx->cache, (uint32_t)layer, (uint32_t)num_q_heads, (float)scale, 0u,
      ctx->q_stage, ctx->out_stage);

  (void)fesetround(saved_round);
  if (st_append != ET_KVQ_ERR_FP_ENV || st_decode != ET_KVQ_ERR_FP_ENV)
    return ET_KVQ_ERR_STATE;
  return ET_KVQ_OK;
}
