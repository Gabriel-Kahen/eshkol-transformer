#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/paged_kv_abi.h"
#include "paged_kv_eshkol_bridge.h"

#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

static void fail(const char *msg, int line) {
  fprintf(stderr, "FAIL (line %d): %s\n", line, msg);
  exit(1);
}

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond))                                                               \
      fail(#cond, __LINE__);                                                   \
  } while (0)

static float det_f32(uint32_t seed, uint32_t idx) {
  uint32_t x = seed * 747796405u + idx * 2891336453u + 277803737u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  const int32_t signed_bucket = (int32_t)(x % 1025u) - 512;
  return (float)signed_bucket * 0.0029296875f; /* ~[-1.5, 1.5], non-dyadic */
}

static void fill_token_kv(uint32_t seed, uint32_t num_layers,
                          uint32_t num_kv_heads, uint32_t head_dim, float *k,
                          float *v) {
  const size_t n = (size_t)num_layers * (size_t)num_kv_heads * (size_t)head_dim;
  for (size_t i = 0u; i < n; ++i) {
    k[i] = det_f32(seed, (uint32_t)(2u * i));
    v[i] = det_f32(seed + 17u, (uint32_t)(2u * i + 1u));
  }
}

/* Independent contiguous dense GQA attention reference in binary64. */
static void dense_gqa_reference(uint32_t seq_len, uint32_t num_layers,
                                uint32_t layer_idx, uint32_t num_q_heads,
                                uint32_t num_kv_heads, uint32_t head_dim,
                                float scale, const float *q,
                                const float *dense_k, const float *dense_v,
                                float *out_ref) {
  const uint32_t group_size = num_q_heads / num_kv_heads;
  const size_t tok_stride =
      (size_t)num_layers * (size_t)num_kv_heads * (size_t)head_dim;
  double *scores = (double *)calloc(seq_len, sizeof(double));
  CHECK(scores != NULL);

  for (uint32_t qh = 0u; qh < num_q_heads; ++qh) {
    const uint32_t kv_h = qh / group_size;
    const float *q_head = q + (size_t)qh * (size_t)head_dim;
    double max_s = -INFINITY;
    for (uint32_t t = 0u; t < seq_len; ++t) {
      const float *k_vec =
          dense_k + (size_t)t * tok_stride +
          ((size_t)layer_idx * (size_t)num_kv_heads + (size_t)kv_h) *
              (size_t)head_dim;
      double dot = 0.0;
      for (uint32_t d = 0u; d < head_dim; ++d)
        dot += (double)q_head[d] * (double)k_vec[d];
      scores[t] = dot * (double)scale;
      if (scores[t] > max_s)
        max_s = scores[t];
    }
    double denom = 0.0;
    for (uint32_t t = 0u; t < seq_len; ++t) {
      scores[t] = exp(scores[t] - max_s);
      denom += scores[t];
    }
    for (uint32_t d = 0u; d < head_dim; ++d) {
      double acc = 0.0;
      for (uint32_t t = 0u; t < seq_len; ++t) {
        const float *v_vec =
            dense_v + (size_t)t * tok_stride +
            ((size_t)layer_idx * (size_t)num_kv_heads + (size_t)kv_h) *
                (size_t)head_dim;
        acc += (scores[t] / denom) * (double)v_vec[d];
      }
      out_ref[(size_t)qh * (size_t)head_dim + (size_t)d] = (float)acc;
    }
  }
  free(scores);
}

static double max_abs_diff(const float *a, const float *b, size_t n) {
  double m = 0.0;
  for (size_t i = 0u; i < n; ++i) {
    const double d = fabs((double)a[i] - (double)b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

static void print_hex_vec(const float *v, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i != 0u)
      putchar(' ');
    printf("%a", (double)v[i]);
  }
  putchar('\n');
}

/* Test 1: Pool-owned & external-buffer lifecycle, CoW fork, diverging
 * continuations, speculative truncation, prefix deduplication, and GQA parity. */
static double test_cow_prefix_truncate_and_parity(void) {
  const et_paged_kv_config cfg = {
      24u, /* num_blocks */
      3u,  /* num_layers */
      2u,  /* num_kv_heads */
      4u,  /* block_size */
      16u, /* head_dim */
      8u,  /* max_seqs */
      8u,  /* max_blocks_per_seq */
      0u};
  size_t pool_floats = 0u;
  CHECK(et_paged_kv_pool_float_count(&cfg, &pool_floats) == ET_PAGED_KV_OK);
  float *ext_buf = (float *)calloc(pool_floats, sizeof(float));
  CHECK(ext_buf != NULL);

  et_paged_kv_pool *pool = NULL;
  CHECK(et_paged_kv_create(&cfg, ext_buf, pool_floats, &pool) ==
        ET_PAGED_KV_OK);

  const uint32_t tok_floats = cfg.num_layers * cfg.num_kv_heads * cfg.head_dim;
  const uint32_t max_toks = cfg.max_blocks_per_seq * cfg.block_size;
  float *dense_k[4];
  float *dense_v[4];
  uint32_t dense_len[4] = {0u, 0u, 0u, 0u};
  for (int i = 0; i < 4; ++i) {
    dense_k[i] = (float *)calloc((size_t)max_toks * tok_floats, sizeof(float));
    dense_v[i] = (float *)calloc((size_t)max_toks * tok_floats, sizeof(float));
    CHECK(dense_k[i] != NULL && dense_v[i] != NULL);
  }

  /* Allocate parent seq0 and prefill 10 tokens (2 full blocks + 2 partial). */
  uint32_t s0 = 99u;
  CHECK(et_paged_kv_seq_alloc(pool, &s0) == ET_PAGED_KV_OK && s0 == 0u);
  for (uint32_t t = 0u; t < 10u; ++t) {
    float *kt = dense_k[0] + (size_t)t * tok_floats;
    float *vt = dense_v[0] + (size_t)t * tok_floats;
    fill_token_kv(100u + t, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, kt,
                  vt);
    CHECK(et_paged_kv_append_token(pool, s0, kt, vt) == ET_PAGED_KV_OK);
  }
  dense_len[0] = 10u;

  /* Register full blocks 0 and 1 in prefix cache using chained SHA-256. */
  const uint32_t blk0_toks[4] = {10u, 20u, 30u, 40u};
  const uint32_t blk1_toks[4] = {50u, 60u, 70u, 80u};
  uint8_t h0[ET_PAGED_KV_HASH_BYTES];
  uint8_t h1[ET_PAGED_KV_HASH_BYTES];
  uint8_t h_other[ET_PAGED_KV_HASH_BYTES];
  CHECK(et_paged_kv_hash_token_block(NULL, blk0_toks, 4u, h0) ==
        ET_PAGED_KV_OK);
  CHECK(et_paged_kv_hash_token_block(h0, blk1_toks, 4u, h1) == ET_PAGED_KV_OK);
  /* Verify chaining changes hash when prev_hash differs. */
  CHECK(et_paged_kv_hash_token_block(NULL, blk1_toks, 4u, h_other) ==
        ET_PAGED_KV_OK);
  CHECK(memcmp(h1, h_other, ET_PAGED_KV_HASH_BYTES) != 0);

  CHECK(et_paged_kv_prefix_cache_register(pool, s0, 0u, h0) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_prefix_cache_register(pool, s0, 1u, h1) == ET_PAGED_KV_OK);
  /* Partial block 2 cannot be registered. */
  CHECK(et_paged_kv_prefix_cache_register(pool, s0, 2u, h_other) ==
        ET_PAGED_KV_ERR_INVALID_RANGE);

  /* Independent sequence attaches both prefix blocks with zero copies. */
  uint32_t s_prefix = 99u;
  uint32_t hit = 0u;
  CHECK(et_paged_kv_seq_alloc(pool, &s_prefix) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_prefix_cache_attach(pool, s_prefix, h_other, &hit) ==
            ET_PAGED_KV_OK &&
        hit == 0u);
  CHECK(et_paged_kv_prefix_cache_attach(pool, s_prefix, h0, &hit) ==
            ET_PAGED_KV_OK &&
        hit == 1u);
  CHECK(et_paged_kv_prefix_cache_attach(pool, s_prefix, h1, &hit) ==
            ET_PAGED_KV_OK &&
        hit == 1u);

  uint32_t len_p = 0u;
  CHECK(et_paged_kv_seq_len(pool, s_prefix, &len_p) == ET_PAGED_KV_OK &&
        len_p == 8u);
  uint32_t bt_s0[8], bt_sp[8];
  uint32_t nb0 = 0u, nbp = 0u;
  CHECK(et_paged_kv_seq_block_table(pool, s0, bt_s0, 8u, &nb0) ==
            ET_PAGED_KV_OK &&
        nb0 == 3u);
  CHECK(et_paged_kv_seq_block_table(pool, s_prefix, bt_sp, 8u, &nbp) ==
            ET_PAGED_KV_OK &&
        nbp == 2u);
  CHECK(bt_sp[0] == bt_s0[0] && bt_sp[1] == bt_s0[1]);

  /* Truncate s_prefix into middle of block 1 (len 6) and append a new token:
   * must trigger CoW on block 1 WITHOUT corrupting s0 or prefix cache! */
  CHECK(et_paged_kv_seq_truncate(pool, s_prefix, 6u) == ET_PAGED_KV_OK);
  float *tmp_k = (float *)calloc(tok_floats, sizeof(float));
  float *tmp_v = (float *)calloc(tok_floats, sizeof(float));
  CHECK(tmp_k != NULL && tmp_v != NULL);
  fill_token_kv(9001u, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, tmp_k,
                tmp_v);
  CHECK(et_paged_kv_append_token(pool, s_prefix, tmp_k, tmp_v) ==
        ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_block_table(pool, s_prefix, bt_sp, 8u, &nbp) ==
            ET_PAGED_KV_OK &&
        nbp == 2u);
  CHECK(bt_sp[0] == bt_s0[0] && bt_sp[1] != bt_s0[1]);
  /* Verify s0 slot 6 (block 1, slot 2) is bitwise untouched! */
  float rk[16], rv[16];
  CHECK(et_paged_kv_read_token(pool, s0, 6u, 0u, 0u, rk, rv) == ET_PAGED_KV_OK);
  CHECK(memcmp(rk, dense_k[0] + (size_t)6u * tok_floats, 16u * sizeof(float)) ==
        0);
  CHECK(memcmp(rv, dense_v[0] + (size_t)6u * tok_floats, 16u * sizeof(float)) ==
        0);
  CHECK(et_paged_kv_seq_free(pool, s_prefix) == ET_PAGED_KV_OK);

  /* Edge case: shared prefix-cached block -> one seq truncates into middle ->
   * other seq frees (ref_count drops 2 -> 1) -> remaining seq appends:
   * prefix cache MUST auto-evict before in-place mutation so no stale read
   * can ever occur. */
  {
    uint32_t s_a = 99u, s_b = 99u;
    uint32_t h_hit = 0u;
    CHECK(et_paged_kv_seq_alloc(pool, &s_a) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_prefix_cache_attach(pool, s_a, h0, &h_hit) ==
              ET_PAGED_KV_OK &&
          h_hit == 1u);
    CHECK(et_paged_kv_seq_truncate(pool, s_a, 2u) == ET_PAGED_KV_OK);
    /* Temporarily truncate s0 to 0 to release s0's ref on blk0 while s_a holds
     * it with ref_count == 1, then verify append on s_a evicts h0 cleanly. */
    uint32_t s_save = 99u;
    CHECK(et_paged_kv_seq_fork(pool, s0, &s_save) == ET_PAGED_KV_OK);
    /* Actually test with a fresh dedicated block on s_b so s0 stays intact! */
    CHECK(et_paged_kv_seq_free(pool, s_save) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_free(pool, s_a) == ET_PAGED_KV_OK);

    CHECK(et_paged_kv_seq_alloc(pool, &s_a) == ET_PAGED_KV_OK);
    for (uint32_t t = 0u; t < 4u; ++t) {
      fill_token_kv(9500u + t, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
                    tmp_k, tmp_v);
      CHECK(et_paged_kv_append_token(pool, s_a, tmp_k, tmp_v) ==
            ET_PAGED_KV_OK);
    }
    const uint32_t uniq_toks[4] = {901u, 902u, 903u, 904u};
    uint8_t h_uniq[ET_PAGED_KV_HASH_BYTES];
    CHECK(et_paged_kv_hash_token_block(NULL, uniq_toks, 4u, h_uniq) ==
          ET_PAGED_KV_OK);
    CHECK(et_paged_kv_prefix_cache_register(pool, s_a, 0u, h_uniq) ==
          ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_fork(pool, s_a, &s_b) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_truncate(pool, s_a, 2u) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_free(pool, s_b) == ET_PAGED_KV_OK);
    /* s_a now exclusively owns the block (ref_count==1); appending must evict
     * h_uniq from the prefix cache before mutating slot 2. */
    CHECK(et_paged_kv_append_token(pool, s_a, tmp_k, tmp_v) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_alloc(pool, &s_b) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_prefix_cache_attach(pool, s_b, h_uniq, &h_hit) ==
              ET_PAGED_KV_OK &&
          h_hit == 0u);
    CHECK(et_paged_kv_seq_free(pool, s_b) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_free(pool, s_a) == ET_PAGED_KV_OK);
  }
  free(tmp_k);
  free(tmp_v);

  et_paged_kv_stats st;
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.used_blocks == 3u && st.shared_blocks == 0u && st.cow_copies == 1u &&
        st.prefix_cache_hits == 3u && st.prefix_cached_blocks == 2u);

  /* Fork s0 into s1, s2, s3 (zero copies at fork time). */
  uint32_t seqs[4] = {s0, 0u, 0u, 0u};
  for (int i = 1; i < 4; ++i) {
    CHECK(et_paged_kv_seq_fork(pool, s0, &seqs[i]) == ET_PAGED_KV_OK);
    memcpy(dense_k[i], dense_k[0], (size_t)10u * tok_floats * sizeof(float));
    memcpy(dense_v[i], dense_v[0], (size_t)10u * tok_floats * sizeof(float));
    dense_len[i] = 10u;
  }
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.used_blocks == 3u && st.shared_blocks == 3u && st.cow_copies == 1u &&
        st.active_seqs == 4u);

  /* Append diverging batch continuations to s1 (+5), s2 (+7), s3 (+1). */
  const uint32_t add_counts[4] = {0u, 5u, 7u, 1u};
  for (int i = 1; i < 4; ++i) {
    const uint32_t add = add_counts[i];
    float *k_batch = dense_k[i] + (size_t)dense_len[i] * tok_floats;
    float *v_batch = dense_v[i] + (size_t)dense_len[i] * tok_floats;
    for (uint32_t t = 0u; t < add; ++t) {
      fill_token_kv(500u * (uint32_t)i + t, cfg.num_layers, cfg.num_kv_heads,
                    cfg.head_dim, k_batch + (size_t)t * tok_floats,
                    v_batch + (size_t)t * tok_floats);
    }
    CHECK(et_paged_kv_append_tokens(pool, seqs[i], add, k_batch, v_batch) ==
          ET_PAGED_KV_OK);
    dense_len[i] += add;
  }

  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  /* 3 CoW splits on partial block 2 -> total cow_copies = 1 + 3 = 4.
   * Blocks 0 and 1 remain shared by all 4 sequences (shared_blocks == 2). */
  CHECK(st.cow_copies == 4u && st.shared_blocks == 2u);
  uint32_t rc0 = 0u, rc1 = 0u, rc2 = 0u;
  CHECK(et_paged_kv_block_refcount(pool, bt_s0[0], &rc0) == ET_PAGED_KV_OK &&
        rc0 == 4u);
  CHECK(et_paged_kv_block_refcount(pool, bt_s0[1], &rc1) == ET_PAGED_KV_OK &&
        rc1 == 4u);
  CHECK(et_paged_kv_block_refcount(pool, bt_s0[2], &rc2) == ET_PAGED_KV_OK &&
        rc2 == 1u);

  /* Speculative append on s2 (+5 -> len 22), then truncate back to 13! */
  for (uint32_t t = 0u; t < 5u; ++t) {
    float *kt = dense_k[2] + (size_t)(dense_len[2] + t) * tok_floats;
    float *vt = dense_v[2] + (size_t)(dense_len[2] + t) * tok_floats;
    fill_token_kv(8800u + t, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, kt,
                  vt);
    CHECK(et_paged_kv_append_token(pool, seqs[2], kt, vt) == ET_PAGED_KV_OK);
  }
  dense_len[2] += 5u;
  CHECK(et_paged_kv_seq_truncate(pool, seqs[2], 13u) == ET_PAGED_KV_OK);
  dense_len[2] = 13u;

  /* Verify exact Paged GQA Causal Decode Attention across all 4 sequences,
   * all 3 layers, and GQA ratios (num_q_heads = 2, 4, 8). */
  const uint32_t q_head_cases[3] = {2u, 4u, 8u};
  const float scale = 1.0f / sqrtf((float)cfg.head_dim);
  double worst_err = 0.0;

  for (int si = 0; si < 4; ++si) {
    for (uint32_t layer = 0u; layer < cfg.num_layers; ++layer) {
      for (int qi = 0; qi < 3; ++qi) {
        const uint32_t hq = q_head_cases[qi];
        float q[8u * 16u];
        float out_paged[8u * 16u];
        float out_ref[8u * 16u];
        for (uint32_t i = 0u; i < hq * cfg.head_dim; ++i)
          q[i] = det_f32(3000u + (uint32_t)si * 97u + layer * 13u + hq, i);
        CHECK(et_paged_kv_decode_attention_v1(pool, seqs[si], layer, hq, scale,
                                              q, out_paged) == ET_PAGED_KV_OK);
        dense_gqa_reference(dense_len[si], cfg.num_layers, layer, hq,
                            cfg.num_kv_heads, cfg.head_dim, scale, q,
                            dense_k[si], dense_v[si], out_ref);
        const double err = max_abs_diff(out_paged, out_ref, hq * cfg.head_dim);
        if (err > worst_err)
          worst_err = err;
        CHECK(err <= 1e-6);
      }
    }
  }

  /* Free all sequences and verify zero block leaks & prefix cache empty. */
  for (int i = 0; i < 4; ++i)
    CHECK(et_paged_kv_seq_free(pool, seqs[i]) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.active_seqs == 0u && st.used_blocks == 0u &&
        st.free_blocks == st.total_blocks && st.shared_blocks == 0u &&
        st.prefix_cached_blocks == 0u);

  /* Reset and verify counters reset to 0 cleanly. */
  CHECK(et_paged_kv_reset(pool) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.cow_copies == 0u && st.prefix_cache_hits == 0u &&
        st.free_blocks == st.total_blocks);

  for (int i = 0; i < 4; ++i) {
    free(dense_k[i]);
    free(dense_v[i]);
  }
  et_paged_kv_destroy(pool);
  free(ext_buf);
  return worst_err;
}

/* Test 2: OOM block exhaustion rollback safety (single-token CoW, boundary
 * alloc, and multi-block batch append), verifying zero leaked blocks and
 * bitwise-unchanged sibling attention output. */
static void test_oom_rollback_safety(void) {
  const et_paged_kv_config cfg = {
      4u, /* num_blocks */
      2u, /* num_layers */
      2u, /* num_kv_heads */
      4u, /* block_size */
      8u, /* head_dim */
      4u, /* max_seqs */
      4u, /* max_blocks_per_seq */
      0u};
  et_paged_kv_pool *pool = NULL;
  CHECK(et_paged_kv_create(&cfg, NULL, 0u, &pool) == ET_PAGED_KV_OK);

  const uint32_t tok_floats = cfg.num_layers * cfg.num_kv_heads * cfg.head_dim;
  float *k_batch = (float *)calloc(12u * tok_floats, sizeof(float));
  float *v_batch = (float *)calloc(12u * tok_floats, sizeof(float));
  CHECK(k_batch != NULL && v_batch != NULL);
  for (uint32_t t = 0u; t < 12u; ++t) {
    fill_token_kv(700u + t, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
                  k_batch + (size_t)t * tok_floats,
                  v_batch + (size_t)t * tok_floats);
  }

  uint32_t s0 = 0u, s1 = 0u, s2 = 0u;
  CHECK(et_paged_kv_seq_alloc(pool, &s0) == ET_PAGED_KV_OK);
  /* Append 6 tokens -> uses 2 blocks (blk 0 full, blk 1 partial). */
  CHECK(et_paged_kv_append_tokens(pool, s0, 6u, k_batch, v_batch) ==
        ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_fork(pool, s0, &s1) == ET_PAGED_KV_OK);

  /* Snapshot s0 attention output before OOM stress. */
  float q[4u * 8u];
  float out_before[4u * 8u];
  float out_after[4u * 8u];
  for (uint32_t i = 0u; i < 32u; ++i)
    q[i] = det_f32(999u, i);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.35355339f, q,
                                        out_before) == ET_PAGED_KV_OK);

  /* Only 2 free blocks remain. Batch-appending 7 tokens to s1 requires:
   * 1 CoW block (for shared partial blk 1) + 2 new blocks (to reach len 13 ->
   * 4 logical blocks) = 3 blocks > 2 free! Must fail atomically with OOM! */
  CHECK(et_paged_kv_append_tokens(pool, s1, 7u, k_batch, v_batch) ==
        ET_PAGED_KV_ERR_OOM_BLOCKS);
  et_paged_kv_stats st;
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.free_blocks == 2u && st.used_blocks == 2u && st.cow_copies == 0u);
  uint32_t len1 = 0u;
  CHECK(et_paged_kv_seq_len(pool, s1, &len1) == ET_PAGED_KV_OK && len1 == 6u);

  /* Now consume the remaining 2 blocks via s2 (8 tokens). */
  CHECK(et_paged_kv_seq_alloc(pool, &s2) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_append_tokens(pool, s2, 8u, k_batch, v_batch) ==
        ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK &&
        st.free_blocks == 0u);

  /* Single-token CoW append on s1 must fail cleanly with OOM. */
  CHECK(et_paged_kv_append_token(pool, s1, k_batch, v_batch) ==
        ET_PAGED_KV_ERR_OOM_BLOCKS);
  /* Boundary single-token append on s2 must fail cleanly with OOM. */
  CHECK(et_paged_kv_append_token(pool, s2, k_batch, v_batch) ==
        ET_PAGED_KV_ERR_OOM_BLOCKS);

  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.free_blocks == 0u && st.cow_copies == 0u && st.shared_blocks == 2u);
  CHECK(et_paged_kv_seq_len(pool, s1, &len1) == ET_PAGED_KV_OK && len1 == 6u);

  /* Verify sibling s0 output is BITWISE IDENTICAL after all OOM failures! */
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.35355339f, q,
                                        out_after) == ET_PAGED_KV_OK);
  CHECK(memcmp(out_before, out_after, sizeof(out_before)) == 0);

  CHECK(et_paged_kv_seq_free(pool, s2) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_free(pool, s1) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_free(pool, s0) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &st) == ET_PAGED_KV_OK);
  CHECK(st.free_blocks == st.total_blocks && st.used_blocks == 0u);

  free(k_batch);
  free(v_batch);
  et_paged_kv_destroy(pool);
}

/* Test 3: Exhaustive negative contract checks (bad shapes, invalid seq_id,
 * fork-of-freed-seq, truncate beyond seq_len, non-finite K/V/Q/scale,
 * aliasing, misaligned raw pointers, FP environment rejection). */
static void test_negative_contracts(void) {
  et_paged_kv_config bad = {8u, 2u, 2u, 4u, 8u, 2u, 4u, 0u};
  et_paged_kv_pool *pool = NULL;

  /* Config validation negatives. */
  CHECK(et_paged_kv_validate_config(NULL) == ET_PAGED_KV_ERR_ARGUMENT);
  bad.reserved = 1u;
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_ARGUMENT);
  bad.reserved = 0u;

  bad.block_size = 0u;
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_SHAPE);
  bad.block_size = ET_PAGED_KV_MAX_BLOCK_SIZE + 1u;
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_SHAPE);
  bad.block_size = 4u;

  bad.num_layers = 0u;
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_SHAPE);
  bad.num_layers = 2u;

  bad.max_blocks_per_seq = 9u; /* > num_blocks (8) */
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_SHAPE);
  bad.max_blocks_per_seq = 2u;

  /* External pool size / alignment / alias negatives. */
  size_t req = 0u;
  CHECK(et_paged_kv_pool_float_count(&bad, &req) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_create(&bad, NULL, 1u, &pool) == ET_PAGED_KV_ERR_ARGUMENT);
  uint8_t *raw_ext = (uint8_t *)calloc(req * sizeof(float) + 16u, 1u);
  CHECK(raw_ext != NULL);
  CHECK(et_paged_kv_create_raw(&bad, raw_ext + 1u, req, &pool) ==
        ET_PAGED_KV_ERR_ALIGNMENT);
  CHECK(et_paged_kv_create_raw(&bad, raw_ext, req - 1u, &pool) ==
        ET_PAGED_KV_ERR_SHAPE);

  /* Create valid pool with external buffer. */
  CHECK(et_paged_kv_create_raw(&bad, raw_ext, req, &pool) == ET_PAGED_KV_OK);

  /* OOM seqs + invalid seq_id + fork-of-freed-seq. */
  uint32_t s0 = 0u, s1 = 0u, s2 = 0u;
  CHECK(et_paged_kv_seq_alloc(pool, &s0) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(pool, &s1) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(pool, &s2) == ET_PAGED_KV_ERR_OOM_SEQS);
  CHECK(et_paged_kv_seq_fork(pool, s0, &s2) == ET_PAGED_KV_ERR_OOM_SEQS);
  CHECK(et_paged_kv_seq_free(pool, s1) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_free(pool, s1) == ET_PAGED_KV_ERR_INVALID_SEQ);
  CHECK(et_paged_kv_seq_fork(pool, s1, &s2) == ET_PAGED_KV_ERR_INVALID_SEQ);
  CHECK(et_paged_kv_seq_fork(pool, 99u, &s2) == ET_PAGED_KV_ERR_INVALID_SEQ);

  const uint32_t tok_floats = bad.num_layers * bad.num_kv_heads * bad.head_dim;
  uint8_t *k_raw = (uint8_t *)calloc(tok_floats * sizeof(float) + 16u, 1u);
  uint8_t *v_raw = (uint8_t *)calloc(tok_floats * sizeof(float) + 16u, 1u);
  CHECK(k_raw != NULL && v_raw != NULL);
  float *kf = (float *)k_raw;
  float *vf = (float *)v_raw;
  fill_token_kv(42u, bad.num_layers, bad.num_kv_heads, bad.head_dim, kf, vf);

  /* Empty seq decode rejected. */
  float q[4u * 8u];
  float out[4u * 8u];
  for (uint32_t i = 0u; i < 32u; ++i) {
    q[i] = 0.25f;
    out[i] = -123.5f;
  }
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_INVALID_RANGE);
  CHECK(out[0] == -123.5f);

  /* Misaligned raw K/V rejected. */
  CHECK(et_paged_kv_append_token_raw(pool, s0, k_raw + 1u, v_raw) ==
        ET_PAGED_KV_ERR_ALIGNMENT);
  /* Aliasing K == V and K overlapping external pool rejected. */
  CHECK(et_paged_kv_append_token(pool, s0, kf, kf) == ET_PAGED_KV_ERR_ALIAS);
  CHECK(et_paged_kv_append_token(pool, s0, (const float *)raw_ext, vf) ==
        ET_PAGED_KV_ERR_ALIAS);

  /* Non-finite K and V rejected with zero mutation. */
  kf[3] = NAN;
  CHECK(et_paged_kv_append_token(pool, s0, kf, vf) ==
        ET_PAGED_KV_ERR_NONFINITE);
  kf[3] = 0.25f;
  vf[5] = INFINITY;
  CHECK(et_paged_kv_append_token(pool, s0, kf, vf) ==
        ET_PAGED_KV_ERR_NONFINITE);
  vf[5] = -INFINITY;
  CHECK(et_paged_kv_append_token(pool, s0, kf, vf) ==
        ET_PAGED_KV_ERR_NONFINITE);
  vf[5] = 0.5f;
  uint32_t len0 = 99u;
  CHECK(et_paged_kv_seq_len(pool, s0, &len0) == ET_PAGED_KV_OK && len0 == 0u);

  /* Valid append of 1 token, then truncate beyond seq_len rejected. */
  CHECK(et_paged_kv_append_token(pool, s0, kf, vf) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_truncate(pool, s0, 2u) ==
        ET_PAGED_KV_ERR_INVALID_RANGE);
  CHECK(et_paged_kv_seq_truncate(pool, s1, 0u) == ET_PAGED_KV_ERR_INVALID_SEQ);

  /* Seq full check (max_blocks_per_seq = 2 -> max 8 tokens). */
  for (uint32_t i = 0u; i < 7u; ++i)
    CHECK(et_paged_kv_append_token(pool, s0, kf, vf) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_append_token(pool, s0, kf, vf) == ET_PAGED_KV_ERR_SEQ_FULL);

  /* Decode attention negative checks: out must stay untouched (-123.5f). */
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 2u, 4u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_SHAPE); /* layer_idx out of range */
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 3u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_SHAPE); /* num_q_heads not multiple of num_kv_heads */
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 0u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_SHAPE);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.0f, q, out) ==
        ET_PAGED_KV_ERR_ARGUMENT);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, -1.0f, q, out) ==
        ET_PAGED_KV_ERR_ARGUMENT);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, NAN, q, out) ==
        ET_PAGED_KV_ERR_NONFINITE);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, q) ==
        ET_PAGED_KV_ERR_ALIAS);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q,
                                        (float *)raw_ext) ==
        ET_PAGED_KV_ERR_ALIAS);
  uint8_t q_unaligned[4u * 8u * sizeof(float) + 8u];
  memset(q_unaligned, 0, sizeof(q_unaligned));
  CHECK(et_paged_kv_decode_attention_v1_raw(pool, s0, 0u, 4u, 0.5f,
                                            q_unaligned + 1u, out) ==
        ET_PAGED_KV_ERR_ALIGNMENT);
  q[7] = NAN;
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_NONFINITE);
  q[7] = 0.25f;
  CHECK(out[0] == -123.5f);

  /* FP environment check: non-FE_TONEAREST rejected cleanly. */
  CHECK(fesetround(FE_UPWARD) == 0);
  CHECK(et_paged_kv_validate_config(&bad) == ET_PAGED_KV_ERR_FP_ENV);
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, out) ==
        ET_PAGED_KV_ERR_FP_ENV);
  CHECK(fesetround(FE_TONEAREST) == 0);
  CHECK(out[0] == -123.5f);

  /* Clean decode after restoring FP env succeeds. */
  CHECK(et_paged_kv_decode_attention_v1(pool, s0, 0u, 4u, 0.5f, q, out) ==
        ET_PAGED_KV_OK);

  free(k_raw);
  free(v_raw);
  et_paged_kv_destroy(pool);
  free(raw_ext);
}

/* Oracle mode: emit exact hex-float cases covering single-block, multi-block,
 * partial-block, CoW-forked siblings, prefix-cache attached, truncated &
 * re-appended, and multi-layer / multi-GQA-ratio decode attention. */
static int run_oracle_dump(void) {
  struct {
    uint32_t num_layers;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t block_size;
    uint32_t head_dim;
    uint32_t prefill_len;
    uint32_t fork_add;
    uint32_t trunc_len;
    uint32_t reappend_len;
    uint32_t layer_idx;
  } cases[8] = {
      {1u, 2u, 2u, 4u, 8u, 1u, 0u, 1u, 0u, 0u},    /* single token, MHA */
      {2u, 4u, 2u, 4u, 8u, 4u, 0u, 4u, 0u, 1u},    /* exact 1 block, GQA 2x */
      {2u, 8u, 2u, 4u, 16u, 11u, 0u, 11u, 0u, 0u}, /* 2 full + 3 partial, GQA 4x */
      {3u, 6u, 2u, 3u, 12u, 5u, 4u, 9u, 0u, 2u},   /* CoW fork across block */
      {2u, 4u, 1u, 5u, 16u, 7u, 6u, 9u, 0u, 1u},   /* CoW fork + MQA + truncate */
      {2u, 8u, 4u, 4u, 8u, 8u, 3u, 6u, 5u, 0u},    /* truncate + reappend */
      {3u, 4u, 2u, 2u, 24u, 9u, 2u, 11u, 0u, 2u},  /* small block_size=2 */
      {2u, 16u, 2u, 8u, 32u, 19u, 5u, 24u, 0u, 1u} /* wide GQA 8x, 3 blocks */
  };

  for (int ci = 0; ci < 8; ++ci) {
    const uint32_t nl = cases[ci].num_layers;
    const uint32_t hq = cases[ci].num_q_heads;
    const uint32_t hkv = cases[ci].num_kv_heads;
    const uint32_t bs = cases[ci].block_size;
    const uint32_t hd = cases[ci].head_dim;
    const uint32_t max_len =
        cases[ci].prefill_len + cases[ci].fork_add + cases[ci].reappend_len + 4u;
    const uint32_t max_blks = (max_len + bs - 1u) / bs + 4u;
    const et_paged_kv_config cfg = {max_blks * 3u, nl, hkv,      bs,
                                    hd,            4u, max_blks, 0u};

    et_paged_kv_pool *pool = NULL;
    CHECK(et_paged_kv_create(&cfg, NULL, 0u, &pool) == ET_PAGED_KV_OK);

    const size_t tok_floats = (size_t)nl * (size_t)hkv * (size_t)hd;
    float *dk = (float *)calloc((size_t)max_len * tok_floats, sizeof(float));
    float *dv = (float *)calloc((size_t)max_len * tok_floats, sizeof(float));
    CHECK(dk != NULL && dv != NULL);

    uint32_t s0 = 0u;
    CHECK(et_paged_kv_seq_alloc(pool, &s0) == ET_PAGED_KV_OK);
    for (uint32_t t = 0u; t < cases[ci].prefill_len; ++t) {
      fill_token_kv(1000u * (uint32_t)(ci + 1) + t, nl, hkv, hd,
                    dk + (size_t)t * tok_floats, dv + (size_t)t * tok_floats);
      CHECK(et_paged_kv_append_token(pool, s0, dk + (size_t)t * tok_floats,
                                     dv + (size_t)t * tok_floats) ==
            ET_PAGED_KV_OK);
    }
    uint32_t active_seq = s0;
    uint32_t cur_len = cases[ci].prefill_len;

    if (cases[ci].fork_add > 0u) {
      uint32_t child = 0u;
      CHECK(et_paged_kv_seq_fork(pool, s0, &child) == ET_PAGED_KV_OK);
      for (uint32_t t = 0u; t < cases[ci].fork_add; ++t) {
        fill_token_kv(5000u * (uint32_t)(ci + 1) + t, nl, hkv, hd,
                      dk + (size_t)(cur_len + t) * tok_floats,
                      dv + (size_t)(cur_len + t) * tok_floats);
        CHECK(et_paged_kv_append_token(
                  pool, child, dk + (size_t)(cur_len + t) * tok_floats,
                  dv + (size_t)(cur_len + t) * tok_floats) == ET_PAGED_KV_OK);
      }
      cur_len += cases[ci].fork_add;
      active_seq = child;
    }

    if (cases[ci].trunc_len < cur_len) {
      CHECK(et_paged_kv_seq_truncate(pool, active_seq, cases[ci].trunc_len) ==
            ET_PAGED_KV_OK);
      cur_len = cases[ci].trunc_len;
    }

    for (uint32_t t = 0u; t < cases[ci].reappend_len; ++t) {
      fill_token_kv(9000u * (uint32_t)(ci + 1) + t, nl, hkv, hd,
                    dk + (size_t)(cur_len + t) * tok_floats,
                    dv + (size_t)(cur_len + t) * tok_floats);
      CHECK(et_paged_kv_append_token(
                pool, active_seq, dk + (size_t)(cur_len + t) * tok_floats,
                dv + (size_t)(cur_len + t) * tok_floats) == ET_PAGED_KV_OK);
    }
    cur_len += cases[ci].reappend_len;

    const float scale = 1.0f / sqrtf((float)hd);
    float *q = (float *)calloc((size_t)hq * (size_t)hd, sizeof(float));
    float *out = (float *)calloc((size_t)hq * (size_t)hd, sizeof(float));
    float *layer_k =
        (float *)calloc((size_t)cur_len * (size_t)hkv * (size_t)hd,
                        sizeof(float));
    float *layer_v =
        (float *)calloc((size_t)cur_len * (size_t)hkv * (size_t)hd,
                        sizeof(float));
    CHECK(q != NULL && out != NULL && layer_k != NULL && layer_v != NULL);

    for (size_t i = 0u; i < (size_t)hq * (size_t)hd; ++i)
      q[i] = det_f32(777u * (uint32_t)(ci + 1), (uint32_t)i);

    CHECK(et_paged_kv_decode_attention_v1(pool, active_seq, cases[ci].layer_idx,
                                          hq, scale, q, out) == ET_PAGED_KV_OK);

    const size_t lstride = (size_t)hkv * (size_t)hd;
    for (uint32_t t = 0u; t < cur_len; ++t) {
      memcpy(layer_k + (size_t)t * lstride,
             dk + (size_t)t * tok_floats + (size_t)cases[ci].layer_idx * lstride,
             lstride * sizeof(float));
      memcpy(layer_v + (size_t)t * lstride,
             dv + (size_t)t * tok_floats + (size_t)cases[ci].layer_idx * lstride,
             lstride * sizeof(float));
    }

    printf("CASE %u %u %u %u %a\n", cur_len, hq, hkv, hd, (double)scale);
    print_hex_vec(q, (size_t)hq * (size_t)hd);
    print_hex_vec(layer_k, (size_t)cur_len * lstride);
    print_hex_vec(layer_v, (size_t)cur_len * lstride);
    print_hex_vec(out, (size_t)hq * (size_t)hd);

    free(q);
    free(out);
    free(layer_k);
    free(layer_v);
    free(dk);
    free(dv);
    et_paged_kv_destroy(pool);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0)
    return run_oracle_dump();
  if (argc != 1)
    return 2;

  const double max_err = test_cow_prefix_truncate_and_parity();
  test_oom_rollback_safety();
  test_negative_contracts();
  printf("PAGED_KV native C11 tests: PASS (max_attn_abs_err=%.3e)\n", max_err);
  return 0;
}
