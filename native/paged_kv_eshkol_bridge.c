#define _POSIX_C_SOURCE 200809L

#include "paged_kv_eshkol_bridge.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ET_PAGED_KV_AOT_HASH_SLOTS 4u

struct et_paged_kv_aot_ctx {
  et_paged_kv_pool *pool;
  et_paged_kv_config cfg;
  size_t token_floats;
  float *k_stage;
  float *v_stage;
  float q_stage[ET_PAGED_KV_MAX_Q_HEADS * ET_PAGED_KV_MAX_HEAD_DIM];
  float out_stage[ET_PAGED_KV_MAX_Q_HEADS * ET_PAGED_KV_MAX_HEAD_DIM];
  uint32_t token_id_stage[ET_PAGED_KV_MAX_BLOCK_SIZE];
  uint8_t hashes[ET_PAGED_KV_AOT_HASH_SLOTS][ET_PAGED_KV_HASH_BYTES];
  uint32_t last_num_q_heads;
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

int32_t et_paged_kv_aot_create(int64_t num_blocks, int64_t num_layers,
                               int64_t num_kv_heads, int64_t block_size,
                               int64_t head_dim, int64_t max_seqs,
                               int64_t max_blocks_per_seq,
                               et_paged_kv_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (num_blocks < 1 || num_blocks > (int64_t)ET_PAGED_KV_MAX_BLOCKS ||
      num_layers < 1 || num_layers > (int64_t)ET_PAGED_KV_MAX_LAYERS ||
      num_kv_heads < 1 || num_kv_heads > (int64_t)ET_PAGED_KV_MAX_KV_HEADS ||
      block_size < 1 || block_size > (int64_t)ET_PAGED_KV_MAX_BLOCK_SIZE ||
      head_dim < 1 || head_dim > (int64_t)ET_PAGED_KV_MAX_HEAD_DIM ||
      max_seqs < 1 || max_seqs > (int64_t)ET_PAGED_KV_MAX_SEQS ||
      max_blocks_per_seq < 1 ||
      max_blocks_per_seq > (int64_t)ET_PAGED_KV_MAX_BLOCKS_PER_SEQ)
    return ET_PAGED_KV_ERR_SHAPE;

  et_paged_kv_config cfg;
  cfg.num_blocks = (uint32_t)num_blocks;
  cfg.num_layers = (uint32_t)num_layers;
  cfg.num_kv_heads = (uint32_t)num_kv_heads;
  cfg.block_size = (uint32_t)block_size;
  cfg.head_dim = (uint32_t)head_dim;
  cfg.max_seqs = (uint32_t)max_seqs;
  cfg.max_blocks_per_seq = (uint32_t)max_blocks_per_seq;
  cfg.reserved = 0u;

  et_paged_kv_aot_ctx *ctx = (et_paged_kv_aot_ctx *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_NOMEM;

  const et_paged_kv_status st =
      et_paged_kv_create(&cfg, NULL, 0u, &ctx->pool);
  if (st != ET_PAGED_KV_OK) {
    free(ctx);
    return st;
  }

  ctx->cfg = cfg;
  ctx->token_floats = (size_t)cfg.num_layers * (size_t)cfg.num_kv_heads *
                      (size_t)cfg.head_dim;
  ctx->k_stage = (float *)calloc(ctx->token_floats, sizeof(float));
  ctx->v_stage = (float *)calloc(ctx->token_floats, sizeof(float));
  if (ctx->k_stage == NULL || ctx->v_stage == NULL) {
    et_paged_kv_aot_destroy(ctx);
    return ET_PAGED_KV_ERR_NOMEM;
  }

  *out_ctx = ctx;
  return ET_PAGED_KV_OK;
}

void et_paged_kv_aot_destroy(et_paged_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  et_paged_kv_destroy(ctx->pool);
  free(ctx->k_stage);
  free(ctx->v_stage);
  free(ctx);
}

int32_t et_paged_kv_aot_reset(et_paged_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  const int32_t st = et_paged_kv_reset(ctx->pool);
  if (st != ET_PAGED_KV_OK)
    return st;
  memset(ctx->k_stage, 0, ctx->token_floats * sizeof(float));
  memset(ctx->v_stage, 0, ctx->token_floats * sizeof(float));
  memset(ctx->q_stage, 0, sizeof(ctx->q_stage));
  memset(ctx->out_stage, 0, sizeof(ctx->out_stage));
  memset(ctx->token_id_stage, 0, sizeof(ctx->token_id_stage));
  memset(ctx->hashes, 0, sizeof(ctx->hashes));
  ctx->last_num_q_heads = 0u;
  return ET_PAGED_KV_OK;
}

int64_t et_paged_kv_aot_seq_alloc(et_paged_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  uint32_t seq_id = 0u;
  const et_paged_kv_status st = et_paged_kv_seq_alloc(ctx->pool, &seq_id);
  return st == ET_PAGED_KV_OK ? (int64_t)seq_id : (int64_t)st;
}

int32_t et_paged_kv_aot_seq_free(et_paged_kv_aot_ctx *ctx, int64_t seq_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  return et_paged_kv_seq_free(ctx->pool, (uint32_t)seq_id);
}

int64_t et_paged_kv_aot_seq_fork(et_paged_kv_aot_ctx *ctx,
                                 int64_t parent_seq_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (parent_seq_id < 0 || parent_seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  uint32_t child_id = 0u;
  const et_paged_kv_status st =
      et_paged_kv_seq_fork(ctx->pool, (uint32_t)parent_seq_id, &child_id);
  return st == ET_PAGED_KV_OK ? (int64_t)child_id : (int64_t)st;
}

int32_t et_paged_kv_aot_seq_truncate(et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                                     int64_t new_len) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (new_len < 0 || new_len > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  return et_paged_kv_seq_truncate(ctx->pool, (uint32_t)seq_id,
                                  (uint32_t)new_len);
}

int64_t et_paged_kv_aot_seq_len(const et_paged_kv_aot_ctx *ctx,
                                int64_t seq_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  uint32_t len = 0u;
  const et_paged_kv_status st =
      et_paged_kv_seq_len(ctx->pool, (uint32_t)seq_id, &len);
  return st == ET_PAGED_KV_OK ? (int64_t)len : (int64_t)st;
}

int64_t et_paged_kv_aot_seq_num_blocks(const et_paged_kv_aot_ctx *ctx,
                                       int64_t seq_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  uint32_t n = 0u;
  const et_paged_kv_status st =
      et_paged_kv_seq_block_table(ctx->pool, (uint32_t)seq_id, NULL, 0u, &n);
  return st == ET_PAGED_KV_OK ? (int64_t)n : (int64_t)st;
}

int64_t et_paged_kv_aot_seq_phys_block(const et_paged_kv_aot_ctx *ctx,
                                       int64_t seq_id, int64_t logical_block) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (logical_block < 0 ||
      logical_block >= (int64_t)ctx->cfg.max_blocks_per_seq)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  uint32_t table[ET_PAGED_KV_MAX_BLOCKS_PER_SEQ];
  uint32_t n = 0u;
  const et_paged_kv_status st = et_paged_kv_seq_block_table(
      ctx->pool, (uint32_t)seq_id, table, ET_PAGED_KV_MAX_BLOCKS_PER_SEQ, &n);
  if (st != ET_PAGED_KV_OK)
    return (int64_t)st;
  if ((uint32_t)logical_block >= n)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  return (int64_t)table[logical_block];
}

int64_t et_paged_kv_aot_block_refcount(const et_paged_kv_aot_ctx *ctx,
                                       int64_t block_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (block_id < 0 || block_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  uint32_t rc = 0u;
  const et_paged_kv_status st =
      et_paged_kv_block_refcount(ctx->pool, (uint32_t)block_id, &rc);
  return st == ET_PAGED_KV_OK ? (int64_t)rc : (int64_t)st;
}

static int32_t set_kv_stage(et_paged_kv_aot_ctx *ctx, float *buf, int64_t layer,
                            int64_t kv_head, int64_t dim, double value) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (layer < 0 || layer >= (int64_t)ctx->cfg.num_layers || kv_head < 0 ||
      kv_head >= (int64_t)ctx->cfg.num_kv_heads || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim)
    return ET_PAGED_KV_ERR_SHAPE;
  const size_t idx = ((size_t)layer * (size_t)ctx->cfg.num_kv_heads +
                      (size_t)kv_head) *
                         (size_t)ctx->cfg.head_dim +
                     (size_t)dim;
  buf[idx] = f32_from_f64(value);
  return ET_PAGED_KV_OK;
}

int32_t et_paged_kv_aot_set_k(et_paged_kv_aot_ctx *ctx, int64_t layer,
                              int64_t kv_head, int64_t dim, double value) {
  return set_kv_stage(ctx, ctx == NULL ? NULL : ctx->k_stage, layer, kv_head,
                      dim, value);
}

int32_t et_paged_kv_aot_set_v(et_paged_kv_aot_ctx *ctx, int64_t layer,
                              int64_t kv_head, int64_t dim, double value) {
  return set_kv_stage(ctx, ctx == NULL ? NULL : ctx->v_stage, layer, kv_head,
                      dim, value);
}

int32_t et_paged_kv_aot_append_token(et_paged_kv_aot_ctx *ctx, int64_t seq_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  return et_paged_kv_append_token(ctx->pool, (uint32_t)seq_id, ctx->k_stage,
                                  ctx->v_stage);
}

int32_t et_paged_kv_aot_set_q(et_paged_kv_aot_ctx *ctx, int64_t q_head,
                              int64_t dim, double value) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (q_head < 0 || q_head >= (int64_t)ET_PAGED_KV_MAX_Q_HEADS || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim)
    return ET_PAGED_KV_ERR_SHAPE;
  ctx->q_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim + (size_t)dim] =
      f32_from_f64(value);
  return ET_PAGED_KV_OK;
}

int32_t et_paged_kv_aot_decode_attention(et_paged_kv_aot_ctx *ctx,
                                         int64_t seq_id, int64_t layer_idx,
                                         int64_t num_q_heads, double scale) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (layer_idx < 0 || layer_idx > (int64_t)UINT32_MAX || num_q_heads < 1 ||
      num_q_heads > (int64_t)ET_PAGED_KV_MAX_Q_HEADS)
    return ET_PAGED_KV_ERR_SHAPE;
  const et_paged_kv_status st = et_paged_kv_decode_attention_v1(
      ctx->pool, (uint32_t)seq_id, (uint32_t)layer_idx, (uint32_t)num_q_heads,
      f32_from_f64(scale), ctx->q_stage, ctx->out_stage);
  if (st == ET_PAGED_KV_OK)
    ctx->last_num_q_heads = (uint32_t)num_q_heads;
  return st;
}

double et_paged_kv_aot_get_out(const et_paged_kv_aot_ctx *ctx, int64_t q_head,
                               int64_t dim) {
  if (ctx == NULL || q_head < 0 || q_head >= (int64_t)ctx->last_num_q_heads ||
      dim < 0 || dim >= (int64_t)ctx->cfg.head_dim)
    return NAN;
  return (double)ctx->out_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim +
                                (size_t)dim];
}

int64_t et_paged_kv_aot_get_out_bits(const et_paged_kv_aot_ctx *ctx,
                                     int64_t q_head, int64_t dim) {
  if (ctx == NULL || q_head < 0 || q_head >= (int64_t)ctx->last_num_q_heads ||
      dim < 0 || dim >= (int64_t)ctx->cfg.head_dim)
    return -1;
  return f32_bits_i64(
      ctx->out_stage[(size_t)q_head * (size_t)ctx->cfg.head_dim + (size_t)dim]);
}

double et_paged_kv_aot_read_kv(const et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                               int64_t pos, int64_t layer_idx,
                               int64_t kv_head, int64_t is_v, int64_t dim) {
  if (ctx == NULL || seq_id < 0 || seq_id > (int64_t)UINT32_MAX || pos < 0 ||
      pos > (int64_t)UINT32_MAX || layer_idx < 0 ||
      layer_idx >= (int64_t)ctx->cfg.num_layers || kv_head < 0 ||
      kv_head >= (int64_t)ctx->cfg.num_kv_heads || (is_v != 0 && is_v != 1) ||
      dim < 0 || dim >= (int64_t)ctx->cfg.head_dim)
    return NAN;
  float k_buf[ET_PAGED_KV_MAX_HEAD_DIM];
  float v_buf[ET_PAGED_KV_MAX_HEAD_DIM];
  const et_paged_kv_status st = et_paged_kv_read_token(
      ctx->pool, (uint32_t)seq_id, (uint32_t)pos, (uint32_t)layer_idx,
      (uint32_t)kv_head, k_buf, v_buf);
  if (st != ET_PAGED_KV_OK)
    return NAN;
  return (double)(is_v == 0 ? k_buf[dim] : v_buf[dim]);
}

int32_t et_paged_kv_aot_set_token_id(et_paged_kv_aot_ctx *ctx, int64_t slot,
                                     int64_t token_id) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (slot < 0 || slot >= (int64_t)ctx->cfg.block_size || token_id < 0 ||
      token_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  ctx->token_id_stage[slot] = (uint32_t)token_id;
  return ET_PAGED_KV_OK;
}

int32_t et_paged_kv_aot_hash_block(et_paged_kv_aot_ctx *ctx, int64_t prev_slot,
                                   int64_t out_slot) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (out_slot < 0 || out_slot >= (int64_t)ET_PAGED_KV_AOT_HASH_SLOTS)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  if (prev_slot < -1 || prev_slot >= (int64_t)ET_PAGED_KV_AOT_HASH_SLOTS)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  const uint8_t *prev = prev_slot < 0 ? NULL : ctx->hashes[prev_slot];
  return et_paged_kv_hash_token_block(prev, ctx->token_id_stage,
                                      ctx->cfg.block_size,
                                      ctx->hashes[out_slot]);
}

int32_t et_paged_kv_aot_prefix_register(et_paged_kv_aot_ctx *ctx,
                                        int64_t seq_id, int64_t logical_block,
                                        int64_t hash_slot) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (logical_block < 0 || logical_block > (int64_t)UINT32_MAX ||
      hash_slot < 0 || hash_slot >= (int64_t)ET_PAGED_KV_AOT_HASH_SLOTS)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  return et_paged_kv_prefix_cache_register(ctx->pool, (uint32_t)seq_id,
                                           (uint32_t)logical_block,
                                           ctx->hashes[hash_slot]);
}

int64_t et_paged_kv_aot_prefix_attach(et_paged_kv_aot_ctx *ctx, int64_t seq_id,
                                      int64_t hash_slot) {
  if (ctx == NULL)
    return ET_PAGED_KV_ERR_ARGUMENT;
  if (seq_id < 0 || seq_id > (int64_t)UINT32_MAX)
    return ET_PAGED_KV_ERR_INVALID_SEQ;
  if (hash_slot < 0 || hash_slot >= (int64_t)ET_PAGED_KV_AOT_HASH_SLOTS)
    return ET_PAGED_KV_ERR_INVALID_RANGE;
  uint32_t hit = 0u;
  const et_paged_kv_status st = et_paged_kv_prefix_cache_attach(
      ctx->pool, (uint32_t)seq_id, ctx->hashes[hash_slot], &hit);
  return st == ET_PAGED_KV_OK ? (int64_t)hit : (int64_t)st;
}

static int get_stats(const et_paged_kv_aot_ctx *ctx, et_paged_kv_stats *out) {
  if (ctx == NULL)
    return 0;
  return et_paged_kv_stats_get(ctx->pool, out) == ET_PAGED_KV_OK;
}

int64_t et_paged_kv_aot_stat_total_blocks(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.total_blocks : -1;
}

int64_t et_paged_kv_aot_stat_free_blocks(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.free_blocks : -1;
}

int64_t et_paged_kv_aot_stat_used_blocks(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.used_blocks : -1;
}

int64_t et_paged_kv_aot_stat_active_seqs(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.active_seqs : -1;
}

int64_t et_paged_kv_aot_stat_shared_blocks(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.shared_blocks : -1;
}

int64_t et_paged_kv_aot_stat_prefix_cached(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.prefix_cached_blocks : -1;
}

int64_t et_paged_kv_aot_stat_cow_copies(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.cow_copies : -1;
}

int64_t et_paged_kv_aot_stat_prefix_hits(const et_paged_kv_aot_ctx *ctx) {
  et_paged_kv_stats s;
  return get_stats(ctx, &s) ? (int64_t)s.prefix_cache_hits : -1;
}
