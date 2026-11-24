#define _POSIX_C_SOURCE 200809L

#include "radix_kv_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct et_radix_kv_aot_ctx {
  et_radix_kv_config cfg;
  et_radix_kv_tree *tree;

  uint32_t *tokens_buf;     /* [max_kv_tokens] */
  uint32_t *kv_slots_buf;   /* [max_kv_tokens] */
  float *k_stage;           /* [max_kv_tokens, n_layers, n_kv_heads, head_dim] */
  float *v_stage;           /* [max_kv_tokens, n_layers, n_kv_heads, head_dim] */
  float *read_k_buf;        /* [max_kv_tokens, n_kv_heads, head_dim] */
  float *read_v_buf;        /* [max_kv_tokens, n_kv_heads, head_dim] */
  float *q_stage;           /* [max_kv_tokens, ET_RADIX_KV_MAX_Q_HEADS, head_dim] */
  float *out_buf;           /* [max_kv_tokens, ET_RADIX_KV_MAX_Q_HEADS, head_dim] */
  float *suffix_k_layer;    /* [max_kv_tokens, n_kv_heads, head_dim] */
  float *suffix_v_layer;    /* [max_kv_tokens, n_kv_heads, head_dim] */

  et_radix_kv_handle last_handle;
  uint32_t last_matched_len;
  uint32_t last_did_split;
  uint32_t last_inserted_tokens;
  uint32_t last_read_len;
  uint32_t last_q_len;
  uint32_t last_n_q_heads;
};

static size_t layer_kv_floats(const et_radix_kv_config *cfg) {
  return (size_t)cfg->n_kv_heads * (size_t)cfg->head_dim;
}

static size_t token_side_floats(const et_radix_kv_config *cfg) {
  return (size_t)cfg->n_layers * layer_kv_floats(cfg);
}

int32_t et_radix_kv_aot_create(int64_t max_kv_tokens, int64_t max_nodes,
                               int64_t n_layers, int64_t n_kv_heads,
                               int64_t head_dim,
                               et_radix_kv_aot_ctx **out_ctx) {
  if (out_ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  *out_ctx = NULL;

  if (max_kv_tokens <= 0 || max_kv_tokens > (int64_t)ET_RADIX_KV_MAX_TOKENS ||
      max_nodes < 2 || max_nodes > (int64_t)ET_RADIX_KV_MAX_NODES ||
      n_layers <= 0 || n_layers > (int64_t)ET_RADIX_KV_MAX_LAYERS ||
      n_kv_heads <= 0 || n_kv_heads > (int64_t)ET_RADIX_KV_MAX_KV_HEADS ||
      head_dim <= 0 || head_dim > (int64_t)ET_RADIX_KV_MAX_HEAD_DIM)
    return ET_RADIX_ERR_SHAPE;

  et_radix_kv_config cfg;
  cfg.max_kv_tokens = (uint32_t)max_kv_tokens;
  cfg.max_nodes = (uint32_t)max_nodes;
  cfg.n_layers = (uint32_t)n_layers;
  cfg.n_kv_heads = (uint32_t)n_kv_heads;
  cfg.head_dim = (uint32_t)head_dim;
  cfg.reserved = 0u;

  et_radix_kv_aot_ctx *ctx =
      (et_radix_kv_aot_ctx *)calloc(1u, sizeof(et_radix_kv_aot_ctx));
  if (ctx == NULL)
    return ET_RADIX_ERR_NOMEM;
  ctx->cfg = cfg;

  const et_radix_kv_status st = et_radix_kv_create(&cfg, &ctx->tree);
  if (st != ET_RADIX_OK) {
    free(ctx);
    return st;
  }

  const size_t max_t = (size_t)cfg.max_kv_tokens;
  const size_t lkv = layer_kv_floats(&cfg);
  const size_t tside = token_side_floats(&cfg);
  const size_t q_cap = max_t * (size_t)ET_RADIX_KV_MAX_Q_HEADS * cfg.head_dim;
  if (q_cap > (size_t)ET_RADIX_KV_MAX_POOL_FLOATS) {
    et_radix_kv_aot_destroy(ctx);
    return ET_RADIX_ERR_SHAPE;
  }

  ctx->tokens_buf = (uint32_t *)calloc(max_t, sizeof(uint32_t));
  ctx->kv_slots_buf = (uint32_t *)calloc(max_t, sizeof(uint32_t));
  ctx->k_stage = (float *)calloc(max_t * tside, sizeof(float));
  ctx->v_stage = (float *)calloc(max_t * tside, sizeof(float));
  ctx->read_k_buf = (float *)calloc(max_t * lkv, sizeof(float));
  ctx->read_v_buf = (float *)calloc(max_t * lkv, sizeof(float));
  ctx->q_stage = (float *)calloc(q_cap, sizeof(float));
  ctx->out_buf = (float *)calloc(q_cap, sizeof(float));
  ctx->suffix_k_layer = (float *)calloc(max_t * lkv, sizeof(float));
  ctx->suffix_v_layer = (float *)calloc(max_t * lkv, sizeof(float));

  if (ctx->tokens_buf == NULL || ctx->kv_slots_buf == NULL ||
      ctx->k_stage == NULL || ctx->v_stage == NULL ||
      ctx->read_k_buf == NULL || ctx->read_v_buf == NULL ||
      ctx->q_stage == NULL || ctx->out_buf == NULL ||
      ctx->suffix_k_layer == NULL || ctx->suffix_v_layer == NULL) {
    et_radix_kv_aot_destroy(ctx);
    return ET_RADIX_ERR_NOMEM;
  }

  ctx->last_handle = et_radix_kv_root_handle(ctx->tree);
  *out_ctx = ctx;
  return ET_RADIX_OK;
}

void et_radix_kv_aot_destroy(et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return;
  et_radix_kv_destroy(ctx->tree);
  free(ctx->tokens_buf);
  free(ctx->kv_slots_buf);
  free(ctx->k_stage);
  free(ctx->v_stage);
  free(ctx->read_k_buf);
  free(ctx->read_v_buf);
  free(ctx->q_stage);
  free(ctx->out_buf);
  free(ctx->suffix_k_layer);
  free(ctx->suffix_v_layer);
  free(ctx);
}

int32_t et_radix_kv_aot_reset(et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  const et_radix_kv_status st = et_radix_kv_reset(ctx->tree);
  if (st == ET_RADIX_OK) {
    ctx->last_handle = et_radix_kv_root_handle(ctx->tree);
    ctx->last_matched_len = 0u;
    ctx->last_did_split = 0u;
    ctx->last_inserted_tokens = 0u;
    ctx->last_read_len = 0u;
    ctx->last_q_len = 0u;
    ctx->last_n_q_heads = 0u;
  }
  return st;
}

int64_t et_radix_kv_aot_root_handle(const et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return (int64_t)ET_RADIX_ERR_ARGUMENT;
  return (int64_t)et_radix_kv_root_handle(ctx->tree);
}

int32_t et_radix_kv_aot_set_token(et_radix_kv_aot_ctx *ctx, int64_t pos,
                                  int64_t token_id) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (pos < 0 || pos >= (int64_t)ctx->cfg.max_kv_tokens || token_id < 0 ||
      token_id > (int64_t)UINT32_MAX)
    return ET_RADIX_ERR_SHAPE;
  ctx->tokens_buf[(size_t)pos] = (uint32_t)token_id;
  return ET_RADIX_OK;
}

int32_t et_radix_kv_aot_set_kv(et_radix_kv_aot_ctx *ctx, int64_t pos,
                               int64_t layer, int64_t kv_head, int64_t dim,
                               int64_t is_v, double value) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (pos < 0 || pos >= (int64_t)ctx->cfg.max_kv_tokens || layer < 0 ||
      layer >= (int64_t)ctx->cfg.n_layers || kv_head < 0 ||
      kv_head >= (int64_t)ctx->cfg.n_kv_heads || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim || (is_v != 0 && is_v != 1))
    return ET_RADIX_ERR_SHAPE;
  const size_t idx =
      (size_t)pos * token_side_floats(&ctx->cfg) +
      ((size_t)layer * ctx->cfg.n_kv_heads + (size_t)kv_head) *
          ctx->cfg.head_dim +
      (size_t)dim;
  if (is_v == 0)
    ctx->k_stage[idx] = (float)value;
  else
    ctx->v_stage[idx] = (float)value;
  return ET_RADIX_OK;
}

int32_t et_radix_kv_aot_set_q(et_radix_kv_aot_ctx *ctx, int64_t q_pos,
                              int64_t q_head, int64_t dim, double value) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (q_pos < 0 || q_pos >= (int64_t)ctx->cfg.max_kv_tokens || q_head < 0 ||
      q_head >= (int64_t)ET_RADIX_KV_MAX_Q_HEADS || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim)
    return ET_RADIX_ERR_SHAPE;
  const size_t idx =
      ((size_t)q_pos * ET_RADIX_KV_MAX_Q_HEADS + (size_t)q_head) *
          ctx->cfg.head_dim +
      (size_t)dim;
  ctx->q_stage[idx] = (float)value;
  return ET_RADIX_OK;
}

int32_t et_radix_kv_aot_match_prefix(et_radix_kv_aot_ctx *ctx,
                                     int64_t seq_len) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (seq_len < 0 || seq_len > (int64_t)ctx->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;
  et_radix_kv_match_result res;
  memset(&res, 0, sizeof(res));
  const et_radix_kv_status st = et_radix_kv_match_prefix(
      ctx->tree, ctx->tokens_buf, (uint32_t)seq_len, &res, ctx->kv_slots_buf,
      ctx->cfg.max_kv_tokens);
  if (st == ET_RADIX_OK) {
    ctx->last_handle = res.handle;
    ctx->last_matched_len = res.matched_len;
    ctx->last_did_split = res.did_split;
  }
  return st;
}

int32_t et_radix_kv_aot_insert(et_radix_kv_aot_ctx *ctx, int64_t seq_len) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (seq_len < 0 || seq_len > (int64_t)ctx->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;
  et_radix_kv_handle h = ET_RADIX_KV_HANDLE_INVALID;
  uint32_t matched = 0u;
  const et_radix_kv_status st =
      et_radix_kv_insert(ctx->tree, ctx->tokens_buf, (uint32_t)seq_len,
                         ctx->k_stage, ctx->v_stage, &h, &matched);
  if (st == ET_RADIX_OK) {
    ctx->last_handle = h;
    ctx->last_matched_len = matched;
    ctx->last_inserted_tokens = (uint32_t)seq_len - matched;
  }
  return st;
}

int32_t et_radix_kv_aot_insert_suffix(et_radix_kv_aot_ctx *ctx,
                                      int64_t prefix_handle,
                                      int64_t suffix_len) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (suffix_len < 0 || suffix_len > (int64_t)ctx->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;
  et_radix_kv_handle h = ET_RADIX_KV_HANDLE_INVALID;
  uint32_t inserted = 0u;
  const et_radix_kv_status st = et_radix_kv_insert_suffix(
      ctx->tree, (et_radix_kv_handle)prefix_handle, ctx->tokens_buf,
      (uint32_t)suffix_len, ctx->k_stage, ctx->v_stage, &h, &inserted);
  if (st == ET_RADIX_OK) {
    ctx->last_handle = h;
    ctx->last_inserted_tokens = inserted;
  }
  return st;
}

int32_t et_radix_kv_aot_pin(et_radix_kv_aot_ctx *ctx, int64_t handle) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  return et_radix_kv_pin(ctx->tree, (et_radix_kv_handle)handle);
}

int32_t et_radix_kv_aot_unpin(et_radix_kv_aot_ctx *ctx, int64_t handle) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  return et_radix_kv_unpin(ctx->tree, (et_radix_kv_handle)handle);
}

int64_t et_radix_kv_aot_evict_lru(et_radix_kv_aot_ctx *ctx,
                                  int64_t target_free_tokens) {
  if (ctx == NULL)
    return (int64_t)ET_RADIX_ERR_ARGUMENT;
  if (target_free_tokens < 0 ||
      target_free_tokens > (int64_t)ctx->cfg.max_kv_tokens)
    return (int64_t)ET_RADIX_ERR_SHAPE;
  uint32_t evicted = 0u;
  const et_radix_kv_status st =
      et_radix_kv_evict_lru(ctx->tree, (uint32_t)target_free_tokens, &evicted);
  if (st != ET_RADIX_OK)
    return (int64_t)st;
  return (int64_t)evicted;
}

int32_t et_radix_kv_aot_verify_invariants(const et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  return et_radix_kv_verify_invariants(ctx->tree);
}

int64_t et_radix_kv_aot_last_handle(const et_radix_kv_aot_ctx *ctx) {
  return ctx == NULL ? (int64_t)ET_RADIX_ERR_ARGUMENT
                     : (int64_t)ctx->last_handle;
}

int64_t et_radix_kv_aot_last_matched_len(const et_radix_kv_aot_ctx *ctx) {
  return ctx == NULL ? (int64_t)ET_RADIX_ERR_ARGUMENT
                     : (int64_t)ctx->last_matched_len;
}

int64_t et_radix_kv_aot_last_did_split(const et_radix_kv_aot_ctx *ctx) {
  return ctx == NULL ? (int64_t)ET_RADIX_ERR_ARGUMENT
                     : (int64_t)ctx->last_did_split;
}

int64_t et_radix_kv_aot_last_inserted_tokens(const et_radix_kv_aot_ctx *ctx) {
  return ctx == NULL ? (int64_t)ET_RADIX_ERR_ARGUMENT
                     : (int64_t)ctx->last_inserted_tokens;
}

int64_t et_radix_kv_aot_last_kv_slot(const et_radix_kv_aot_ctx *ctx,
                                     int64_t pos) {
  if (ctx == NULL)
    return (int64_t)ET_RADIX_ERR_ARGUMENT;
  if (pos < 0 || pos >= (int64_t)ctx->last_matched_len)
    return (int64_t)ET_RADIX_ERR_SHAPE;
  return (int64_t)ctx->kv_slots_buf[(size_t)pos];
}

int64_t et_radix_kv_aot_handle_depth(const et_radix_kv_aot_ctx *ctx,
                                     int64_t handle) {
  if (ctx == NULL)
    return (int64_t)ET_RADIX_ERR_ARGUMENT;
  uint32_t depth = 0u;
  const et_radix_kv_status st = et_radix_kv_handle_info(
      ctx->tree, (et_radix_kv_handle)handle, &depth, NULL, NULL, NULL);
  return st == ET_RADIX_OK ? (int64_t)depth : (int64_t)st;
}

int64_t et_radix_kv_aot_handle_pin_count(const et_radix_kv_aot_ctx *ctx,
                                         int64_t handle) {
  if (ctx == NULL)
    return (int64_t)ET_RADIX_ERR_ARGUMENT;
  uint32_t pins = 0u;
  const et_radix_kv_status st = et_radix_kv_handle_info(
      ctx->tree, (et_radix_kv_handle)handle, NULL, &pins, NULL, NULL);
  return st == ET_RADIX_OK ? (int64_t)pins : (int64_t)st;
}

int32_t et_radix_kv_aot_read_kv(et_radix_kv_aot_ctx *ctx, int64_t handle,
                                int64_t layer) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (layer < 0 || layer >= (int64_t)ctx->cfg.n_layers)
    return ET_RADIX_ERR_SHAPE;
  const size_t cap =
      (size_t)ctx->cfg.max_kv_tokens * layer_kv_floats(&ctx->cfg);
  uint32_t seq_len = 0u;
  const et_radix_kv_status st = et_radix_kv_read_kv(
      ctx->tree, (et_radix_kv_handle)handle, (uint32_t)layer, ctx->read_k_buf,
      ctx->read_v_buf, cap, &seq_len);
  if (st == ET_RADIX_OK)
    ctx->last_read_len = seq_len;
  return st;
}

double et_radix_kv_aot_get_read_kv(const et_radix_kv_aot_ctx *ctx, int64_t pos,
                                   int64_t kv_head, int64_t dim, int64_t is_v) {
  if (ctx == NULL || pos < 0 || pos >= (int64_t)ctx->last_read_len ||
      kv_head < 0 || kv_head >= (int64_t)ctx->cfg.n_kv_heads || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim || (is_v != 0 && is_v != 1))
    return 0.0;
  const size_t idx =
      ((size_t)pos * ctx->cfg.n_kv_heads + (size_t)kv_head) *
          ctx->cfg.head_dim +
      (size_t)dim;
  return is_v == 0 ? (double)ctx->read_k_buf[idx]
                   : (double)ctx->read_v_buf[idx];
}

int32_t et_radix_kv_aot_attend_gqa(et_radix_kv_aot_ctx *ctx,
                                   int64_t prefix_handle, int64_t layer,
                                   int64_t n_q_heads, int64_t q_len,
                                   int64_t suffix_len, double scale) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  if (layer < 0 || layer >= (int64_t)ctx->cfg.n_layers || n_q_heads <= 0 ||
      n_q_heads > (int64_t)ET_RADIX_KV_MAX_Q_HEADS || q_len <= 0 ||
      q_len > (int64_t)ctx->cfg.max_kv_tokens || suffix_len < 0 ||
      suffix_len > (int64_t)ctx->cfg.max_kv_tokens)
    return ET_RADIX_ERR_SHAPE;

  /* Pack contiguous Q[q_len, n_q_heads, head_dim] into out_buf? No, pack into
   * a contiguous view via temporary compacting or directly allocate a small
   * stack/compact buffer? We already have q_stage indexed with stride
   * ET_RADIX_KV_MAX_Q_HEADS; pack into suffix_k_layer? No, pack in a scratch
   * buffer allocated on demand? Keep zero alloc: pack q into a compact slice
   * of q_stage only if n_q_heads == MAX_Q_HEADS, or allocate? Better: use a
   * dedicated contiguous q_compact buffer! Let's compact into malloc'd temp or
   * check if we can store q_stage packed at set_q? Wait: set_q doesn't know
   * n_q_heads yet. A clean compact buffer allocated via malloc for q_compact
   * and freed immediately is tiny and safe! */
  const size_t hd = (size_t)ctx->cfg.head_dim;
  const size_t q_floats = (size_t)q_len * (size_t)n_q_heads * hd;
  float *q_compact = (float *)malloc(q_floats * sizeof(float));
  float *out_compact = (float *)malloc(q_floats * sizeof(float));
  if (q_compact == NULL || out_compact == NULL) {
    free(q_compact);
    free(out_compact);
    return ET_RADIX_ERR_NOMEM;
  }

  for (uint32_t qi = 0u; qi < (uint32_t)q_len; ++qi) {
    for (uint32_t qh = 0u; qh < (uint32_t)n_q_heads; ++qh) {
      const float *src =
          ctx->q_stage +
          ((size_t)qi * ET_RADIX_KV_MAX_Q_HEADS + (size_t)qh) * hd;
      float *dst = q_compact + ((size_t)qi * (size_t)n_q_heads + (size_t)qh) * hd;
      memcpy(dst, src, hd * sizeof(float));
    }
  }

  const float *sk = NULL;
  const float *sv = NULL;
  if (suffix_len > 0) {
    const size_t lkv = layer_kv_floats(&ctx->cfg);
    const size_t tside = token_side_floats(&ctx->cfg);
    const size_t loff = (size_t)layer * lkv;
    for (uint32_t t = 0u; t < (uint32_t)suffix_len; ++t) {
      memcpy(ctx->suffix_k_layer + (size_t)t * lkv,
             ctx->k_stage + (size_t)t * tside + loff, lkv * sizeof(float));
      memcpy(ctx->suffix_v_layer + (size_t)t * lkv,
             ctx->v_stage + (size_t)t * tside + loff, lkv * sizeof(float));
    }
    sk = ctx->suffix_k_layer;
    sv = ctx->suffix_v_layer;
  }

  const et_radix_kv_status st = et_radix_kv_attend_gqa_f32(
      ctx->tree, (et_radix_kv_handle)prefix_handle, (uint32_t)layer,
      (uint32_t)n_q_heads, (uint32_t)q_len, (uint32_t)suffix_len, (float)scale,
      q_compact, sk, sv, out_compact);

  if (st == ET_RADIX_OK) {
    for (uint32_t qi = 0u; qi < (uint32_t)q_len; ++qi) {
      for (uint32_t qh = 0u; qh < (uint32_t)n_q_heads; ++qh) {
        const float *src =
            out_compact + ((size_t)qi * (size_t)n_q_heads + (size_t)qh) * hd;
        float *dst = ctx->out_buf +
                     ((size_t)qi * ET_RADIX_KV_MAX_Q_HEADS + (size_t)qh) * hd;
        memcpy(dst, src, hd * sizeof(float));
      }
    }
    ctx->last_q_len = (uint32_t)q_len;
    ctx->last_n_q_heads = (uint32_t)n_q_heads;
  }

  free(q_compact);
  free(out_compact);
  return st;
}

double et_radix_kv_aot_get_out(const et_radix_kv_aot_ctx *ctx, int64_t q_pos,
                               int64_t q_head, int64_t dim) {
  if (ctx == NULL || q_pos < 0 || q_pos >= (int64_t)ctx->last_q_len ||
      q_head < 0 || q_head >= (int64_t)ctx->last_n_q_heads || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim)
    return 0.0;
  const size_t idx =
      ((size_t)q_pos * ET_RADIX_KV_MAX_Q_HEADS + (size_t)q_head) *
          ctx->cfg.head_dim +
      (size_t)dim;
  return (double)ctx->out_buf[idx];
}

int64_t et_radix_kv_aot_get_out_bits(const et_radix_kv_aot_ctx *ctx,
                                     int64_t q_pos, int64_t q_head,
                                     int64_t dim) {
  if (ctx == NULL || q_pos < 0 || q_pos >= (int64_t)ctx->last_q_len ||
      q_head < 0 || q_head >= (int64_t)ctx->last_n_q_heads || dim < 0 ||
      dim >= (int64_t)ctx->cfg.head_dim)
    return 0;
  const size_t idx =
      ((size_t)q_pos * ET_RADIX_KV_MAX_Q_HEADS + (size_t)q_head) *
          ctx->cfg.head_dim +
      (size_t)dim;
  uint32_t bits = 0u;
  memcpy(&bits, &ctx->out_buf[idx], sizeof(bits));
  return (int64_t)bits;
}

int32_t et_radix_kv_aot_snapshot_save_env(const et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  const char *path = getenv("ET_RADIX_KV_AOT_SNAPSHOT_PATH");
  if (path == NULL || path[0] == '\0')
    return ET_RADIX_ERR_ARGUMENT;
  return et_radix_kv_snapshot_save(ctx->tree, path);
}

int32_t et_radix_kv_aot_snapshot_reload_env(et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  const char *path = getenv("ET_RADIX_KV_AOT_SNAPSHOT_PATH");
  if (path == NULL || path[0] == '\0')
    return ET_RADIX_ERR_ARGUMENT;
  et_radix_kv_tree *loaded = NULL;
  const et_radix_kv_status st = et_radix_kv_snapshot_load(path, &loaded);
  if (st != ET_RADIX_OK)
    return st;
  et_radix_kv_config loaded_cfg;
  if (et_radix_kv_config_get(loaded, &loaded_cfg) != ET_RADIX_OK ||
      memcmp(&loaded_cfg, &ctx->cfg, sizeof(loaded_cfg)) != 0) {
    et_radix_kv_destroy(loaded);
    return ET_RADIX_ERR_SHAPE;
  }
  et_radix_kv_destroy(ctx->tree);
  ctx->tree = loaded;
  ctx->last_handle = et_radix_kv_root_handle(ctx->tree);
  return ET_RADIX_OK;
}

int32_t et_radix_kv_aot_snapshot_tamper_check_env(
    const et_radix_kv_aot_ctx *ctx) {
  if (ctx == NULL)
    return ET_RADIX_ERR_ARGUMENT;
  const char *path = getenv("ET_RADIX_KV_AOT_SNAPSHOT_PATH");
  if (path == NULL || path[0] == '\0')
    return ET_RADIX_ERR_ARGUMENT;

  const size_t path_len = strlen(path);
  char *bad_path = (char *)malloc(path_len + 16u);
  if (bad_path == NULL)
    return ET_RADIX_ERR_NOMEM;
  memcpy(bad_path, path, path_len);
  memcpy(bad_path + path_len, ".tamper", 8u);

  const int in_fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (in_fd < 0) {
    free(bad_path);
    return ET_RADIX_ERR_IO;
  }
  struct stat meta;
  if (fstat(in_fd, &meta) != 0 || meta.st_size < 96) {
    (void)close(in_fd);
    free(bad_path);
    return ET_RADIX_ERR_IO;
  }
  const size_t sz = (size_t)meta.st_size;
  uint8_t *buf = (uint8_t *)malloc(sz);
  if (buf == NULL) {
    (void)close(in_fd);
    free(bad_path);
    return ET_RADIX_ERR_NOMEM;
  }
  size_t got = 0u;
  while (got < sz) {
    const ssize_t r = read(in_fd, buf + got, sz - got);
    if (r < 0 && errno == EINTR)
      continue;
    if (r <= 0)
      break;
    got += (size_t)r;
  }
  (void)close(in_fd);
  if (got != sz) {
    free(buf);
    free(bad_path);
    return ET_RADIX_ERR_IO;
  }

  /* 1. Flip 1 bit in payload and confirm ET_RADIX_ERR_CHECKSUM. */
  buf[ET_RADIX_KV_SNAPSHOT_HEADER_BYTES] ^= 0x01u;
  int out_fd = open(bad_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out_fd < 0) {
    free(buf);
    free(bad_path);
    return ET_RADIX_ERR_IO;
  }
  size_t put = 0u;
  while (put < sz) {
    const ssize_t w = write(out_fd, buf + put, sz - put);
    if (w < 0 && errno == EINTR)
      continue;
    if (w <= 0)
      break;
    put += (size_t)w;
  }
  (void)close(out_fd);
  et_radix_kv_tree *bad_tree = NULL;
  const et_radix_kv_status st_tamper =
      et_radix_kv_snapshot_load(bad_path, &bad_tree);
  if (bad_tree != NULL)
    et_radix_kv_destroy(bad_tree);

  /* 2. Truncate snapshot and confirm ET_RADIX_ERR_TRUNCATED. */
  buf[ET_RADIX_KV_SNAPSHOT_HEADER_BYTES] ^= 0x01u;
  out_fd = open(bad_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out_fd >= 0) {
    const size_t trunc_sz = sz - 10u;
    size_t tput = 0u;
    while (tput < trunc_sz) {
      const ssize_t w = write(out_fd, buf + tput, trunc_sz - tput);
      if (w < 0 && errno == EINTR)
        continue;
      if (w <= 0)
        break;
      tput += (size_t)w;
    }
    (void)close(out_fd);
  }
  bad_tree = NULL;
  const et_radix_kv_status st_trunc =
      et_radix_kv_snapshot_load(bad_path, &bad_tree);
  if (bad_tree != NULL)
    et_radix_kv_destroy(bad_tree);

  (void)unlink(bad_path);
  free(buf);
  free(bad_path);

  if (st_tamper != ET_RADIX_ERR_CHECKSUM || st_trunc != ET_RADIX_ERR_TRUNCATED)
    return ET_RADIX_ERR_INVARIANT;
  return ET_RADIX_OK;
}

static et_radix_kv_stats fetch_stats(const et_radix_kv_aot_ctx *ctx) {
  et_radix_kv_stats st;
  memset(&st, 0, sizeof(st));
  if (ctx != NULL && ctx->tree != NULL)
    (void)et_radix_kv_stats_get(ctx->tree, &st);
  return st;
}

int64_t et_radix_kv_aot_stat_used_tokens(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).used_tokens;
}
int64_t et_radix_kv_aot_stat_free_tokens(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).free_tokens;
}
int64_t et_radix_kv_aot_stat_pinned_tokens(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).pinned_tokens;
}
int64_t et_radix_kv_aot_stat_evictable_tokens(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).evictable_tokens;
}
int64_t et_radix_kv_aot_stat_active_nodes(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).active_nodes;
}
int64_t et_radix_kv_aot_stat_lru_tick(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).lru_tick;
}
int64_t et_radix_kv_aot_stat_total_matched_tokens(
    const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).total_matched_tokens;
}
int64_t et_radix_kv_aot_stat_total_inserted_tokens(
    const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).total_inserted_tokens;
}
int64_t et_radix_kv_aot_stat_total_evicted_tokens(
    const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).total_evicted_tokens;
}
int64_t et_radix_kv_aot_stat_total_edge_splits(const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).total_edge_splits;
}
int64_t et_radix_kv_aot_stat_total_oom_rejections(
    const et_radix_kv_aot_ctx *ctx) {
  return (int64_t)fetch_stats(ctx).total_oom_rejections;
}
