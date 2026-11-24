#define _POSIX_C_SOURCE 200809L

#include "eshkol_transformer/radix_kv_abi.h"
#include "radix_kv_eshkol_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static float det_val(uint32_t seed, uint32_t idx) {
  const int32_t bucket =
      (int32_t)(((seed * 37u + idx * 19u + (idx / 3u)) % 41u)) - 20;
  return 0.0625f * (float)bucket;
}

static void fill_seq_kv(uint32_t seq_len, uint32_t n_layers,
                        uint32_t n_kv_heads, uint32_t head_dim, uint32_t seed,
                        float *k, float *v) {
  const size_t per_tok = (size_t)n_layers * n_kv_heads * head_dim;
  for (uint32_t t = 0u; t < seq_len; ++t) {
    for (size_t i = 0u; i < per_tok; ++i) {
      k[(size_t)t * per_tok + i] = det_val(seed + t * 11u, (uint32_t)i);
      v[(size_t)t * per_tok + i] = det_val(seed + t * 11u + 7u, (uint32_t)i);
    }
  }
}

static void extract_layer_kv(uint32_t seq_len, uint32_t n_layers,
                             uint32_t n_kv_heads, uint32_t head_dim,
                             uint32_t layer, const float *k_all,
                             const float *v_all, float *k_l, float *v_l) {
  const size_t lkv = (size_t)n_kv_heads * head_dim;
  const size_t per_tok = (size_t)n_layers * lkv;
  for (uint32_t t = 0u; t < seq_len; ++t) {
    memcpy(k_l + (size_t)t * lkv, k_all + (size_t)t * per_tok + layer * lkv,
           lkv * sizeof(float));
    memcpy(v_l + (size_t)t * lkv, v_all + (size_t)t * per_tok + layer * lkv,
           lkv * sizeof(float));
  }
}

static void check_stats_conservation(const et_radix_kv_tree *tree) {
  et_radix_kv_stats st;
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.used_tokens + st.free_tokens == st.max_kv_tokens);
  CHECK(st.evictable_tokens + st.pinned_tokens == st.used_tokens);
  CHECK(st.active_nodes + st.free_nodes == st.max_nodes);
  CHECK(st.active_nodes >= 1u);
  CHECK(et_radix_kv_verify_invariants(tree) == ET_RADIX_OK);
}

static void test_config_and_lifecycle(void) {
  et_radix_kv_config bad = {0u, 8u, 2u, 2u, 4u, 0u};
  CHECK(et_radix_kv_validate_config(NULL) == ET_RADIX_ERR_ARGUMENT);
  CHECK(et_radix_kv_validate_config(&bad) == ET_RADIX_ERR_SHAPE);
  bad.max_kv_tokens = 16u;
  bad.max_nodes = 1u; /* must be >= 2 */
  CHECK(et_radix_kv_validate_config(&bad) == ET_RADIX_ERR_SHAPE);
  bad.max_nodes = 8u;
  bad.reserved = 1u;
  CHECK(et_radix_kv_validate_config(&bad) == ET_RADIX_ERR_ARGUMENT);
  bad.reserved = 0u;
  bad.max_kv_tokens = ET_RADIX_KV_MAX_TOKENS;
  bad.n_layers = ET_RADIX_KV_MAX_LAYERS;
  bad.n_kv_heads = ET_RADIX_KV_MAX_KV_HEADS;
  bad.head_dim = ET_RADIX_KV_MAX_HEAD_DIM;
  CHECK(et_radix_kv_validate_config(&bad) == ET_RADIX_ERR_SHAPE);

  const et_radix_kv_config cfg = {64u, 32u, 2u, 2u, 8u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);
  CHECK(tree != NULL);
  check_stats_conservation(tree);

  et_radix_kv_config got;
  CHECK(et_radix_kv_config_get(tree, &got) == ET_RADIX_OK);
  CHECK(memcmp(&got, &cfg, sizeof(cfg)) == 0);
  CHECK(et_radix_kv_root_handle(tree) != ET_RADIX_KV_HANDLE_INVALID);
  CHECK(et_radix_kv_root_handle(NULL) == ET_RADIX_KV_HANDLE_INVALID);
  et_radix_kv_destroy(tree);
  et_radix_kv_destroy(NULL);
}

static void test_splits_branching_and_pins(void) {
  const et_radix_kv_config cfg = {64u, 32u, 2u, 2u, 8u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  /* Insert 24-token sequence A = sys[0..15] + userA[100..107]. */
  uint32_t seq_a[24];
  for (uint32_t i = 0u; i < 16u; ++i)
    seq_a[i] = 1000u + i;
  for (uint32_t i = 0u; i < 8u; ++i)
    seq_a[16u + i] = 2000u + i;
  float k_a[24 * 32], v_a[24 * 32];
  fill_seq_kv(24u, 2u, 2u, 8u, 10u, k_a, v_a);

  et_radix_kv_handle h_a = ET_RADIX_KV_HANDLE_INVALID;
  uint32_t matched_ins = 99u;
  CHECK(et_radix_kv_insert(tree, seq_a, 24u, k_a, v_a, &h_a, &matched_ins) ==
        ET_RADIX_OK);
  CHECK(matched_ins == 0u);
  check_stats_conservation(tree);

  /* Pin h_a before any split happens! */
  CHECK(et_radix_kv_pin(tree, h_a) == ET_RADIX_OK);
  et_radix_kv_stats st;
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.used_tokens == 24u && st.pinned_tokens == 24u &&
        st.evictable_tokens == 0u && st.active_nodes == 2u);

  /* Match seq_b = sys[0..15] + userB[3000..3005] -> splits edge at 16! */
  uint32_t seq_b[22];
  for (uint32_t i = 0u; i < 16u; ++i)
    seq_b[i] = 1000u + i;
  for (uint32_t i = 0u; i < 6u; ++i)
    seq_b[16u + i] = 3000u + i;
  et_radix_kv_match_result mres;
  uint32_t slots_sys[22];
  CHECK(et_radix_kv_match_prefix(tree, seq_b, 22u, &mres, slots_sys, 22u) ==
        ET_RADIX_OK);
  CHECK(mres.matched_len == 16u && mres.did_split == 1u);
  check_stats_conservation(tree);

  /* Verify h_a handle is STILL valid with depth=24, pin_count=1, and new split
   * parent mres.handle has depth=16, pin_count=1, self_pin_count=0! */
  uint32_t depth = 0u, pins = 0u, self_pins = 0u;
  CHECK(et_radix_kv_handle_info(tree, h_a, &depth, &pins, &self_pins, NULL) ==
        ET_RADIX_OK);
  CHECK(depth == 24u && pins == 1u && self_pins == 1u);
  CHECK(et_radix_kv_handle_info(tree, mres.handle, &depth, &pins, &self_pins,
                                NULL) == ET_RADIX_OK);
  CHECK(depth == 16u && pins == 1u && self_pins == 0u);

  /* Unpinning split parent (self_pin_count == 0, pin_count == 1) MUST fail! */
  CHECK(et_radix_kv_unpin(tree, mres.handle) == ET_RADIX_ERR_DOUBLE_UNPIN);
  check_stats_conservation(tree);

  /* Insert suffix of seq_b under mres.handle (6 tokens only, zero copy!). */
  float k_b_suf[6 * 32], v_b_suf[6 * 32];
  fill_seq_kv(6u, 2u, 2u, 8u, 50u, k_b_suf, v_b_suf);
  et_radix_kv_handle h_b = ET_RADIX_KV_HANDLE_INVALID;
  uint32_t inserted = 0u;
  CHECK(et_radix_kv_insert_suffix(tree, mres.handle, seq_b + 16u, 6u, k_b_suf,
                                  v_b_suf, &h_b, &inserted) == ET_RADIX_OK);
  CHECK(inserted == 6u);
  check_stats_conservation(tree);
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.used_tokens == 30u && st.pinned_tokens == 24u &&
        st.evictable_tokens == 6u && st.active_nodes == 4u);

  /* Split interior edge [0..15] at offset 8 via seq_c = sys[0..7] + [4000]! */
  uint32_t seq_c[9];
  for (uint32_t i = 0u; i < 8u; ++i)
    seq_c[i] = 1000u + i;
  seq_c[8] = 4000u;
  uint32_t slots_c[9];
  et_radix_kv_match_result mres_c;
  CHECK(et_radix_kv_match_prefix(tree, seq_c, 9u, &mres_c, slots_c, 9u) ==
        ET_RADIX_OK);
  CHECK(mres_c.matched_len == 8u && mres_c.did_split == 1u);
  for (uint32_t i = 0u; i < 8u; ++i)
    CHECK(slots_c[i] == slots_sys[i]);
  check_stats_conservation(tree);

  /* Unpin h_a and verify double-unpin fails cleanly. */
  CHECK(et_radix_kv_unpin(tree, h_a) == ET_RADIX_OK);
  CHECK(et_radix_kv_unpin(tree, h_a) == ET_RADIX_ERR_DOUBLE_UNPIN);
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.pinned_tokens == 0u && st.evictable_tokens == 30u);

  et_radix_kv_destroy(tree);
}

static void test_lru_eviction_cascade_and_oom_rollback(void) {
  /* Tight pool: max_kv_tokens = 10, max_nodes = 10. */
  const et_radix_kv_config cfg = {10u, 10u, 1u, 2u, 4u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  float k[10 * 8], v[10 * 8];
  fill_seq_kv(10u, 1u, 2u, 4u, 100u, k, v);

  /* Build: P([1,2]) -> C1([10,11]) and C2([20,21]) (6 tokens), plus Q([3,4,5,6]) (4 tokens). */
  const uint32_t seq_c1[4] = {1u, 2u, 10u, 11u};
  const uint32_t seq_c2[4] = {1u, 2u, 20u, 21u};
  const uint32_t seq_q[4] = {3u, 4u, 5u, 6u};
  et_radix_kv_handle h_c1 = 0u, h_c2 = 0u, h_q = 0u;
  CHECK(et_radix_kv_insert(tree, seq_c1, 4u, k, v, &h_c1, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_insert(tree, seq_c2, 4u, k, v, &h_c2, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_insert(tree, seq_q, 4u, k, v, &h_q, NULL) == ET_RADIX_OK);
  check_stats_conservation(tree);

  /* Verify ticks: C1 < C2 < Q (and P has C2's tick). */
  uint64_t t_c1 = 0u, t_c2 = 0u, t_q = 0u;
  CHECK(et_radix_kv_handle_info(tree, h_c1, NULL, NULL, NULL, &t_c1) == ET_RADIX_OK);
  CHECK(et_radix_kv_handle_info(tree, h_c2, NULL, NULL, NULL, &t_c2) == ET_RADIX_OK);
  CHECK(et_radix_kv_handle_info(tree, h_q, NULL, NULL, NULL, &t_q) == ET_RADIX_OK);
  CHECK(t_c1 < t_c2 && t_c2 < t_q);

  /* Insert R([90..95], 6 tokens): MUST evict C1 (2) -> C2 (2) -> newly exposed
   * parent leaf P (2) = 6 tokens total, keeping Q untouched! */
  const uint32_t seq_r[6] = {90u, 91u, 92u, 93u, 94u, 95u};
  et_radix_kv_handle h_r = 0u;
  CHECK(et_radix_kv_insert(tree, seq_r, 6u, k, v, &h_r, NULL) == ET_RADIX_OK);
  check_stats_conservation(tree);

  /* C1 and C2 must be evicted (stale handles rejected), Q and R alive! */
  CHECK(et_radix_kv_handle_info(tree, h_c1, NULL, NULL, NULL, NULL) ==
        ET_RADIX_ERR_INVALID_HANDLE);
  CHECK(et_radix_kv_handle_info(tree, h_c2, NULL, NULL, NULL, NULL) ==
        ET_RADIX_ERR_INVALID_HANDLE);
  CHECK(et_radix_kv_handle_info(tree, h_q, NULL, NULL, NULL, NULL) ==
        ET_RADIX_OK);
  CHECK(et_radix_kv_handle_info(tree, h_r, NULL, NULL, NULL, NULL) ==
        ET_RADIX_OK);

  et_radix_kv_stats st;
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.used_tokens == 10u && st.free_tokens == 0u &&
        st.total_evicted_tokens == 6u && st.total_evicted_nodes == 3u);

  /* Now pin BOTH Q and R (all 10 tokens pinned) and test atomic OOM rollback! */
  CHECK(et_radix_kv_pin(tree, h_q) == ET_RADIX_OK);
  CHECK(et_radix_kv_pin(tree, h_r) == ET_RADIX_OK);
  et_radix_kv_stats before;
  CHECK(et_radix_kv_stats_get(tree, &before) == ET_RADIX_OK);
  CHECK(before.pinned_tokens == 10u && before.evictable_tokens == 0u);

  /* OOM on fresh sequence AND OOM on mid-edge split insert! */
  const uint32_t seq_oom1[2] = {70u, 71u};
  const uint32_t seq_oom2[4] = {90u, 91u, 70u, 71u};
  et_radix_kv_handle h_fail = ET_RADIX_KV_HANDLE_INVALID;
  CHECK(et_radix_kv_insert(tree, seq_oom1, 2u, k, v, &h_fail, NULL) ==
        ET_RADIX_ERR_OOM);
  CHECK(et_radix_kv_insert(tree, seq_oom2, 4u, k, v, &h_fail, NULL) ==
        ET_RADIX_ERR_OOM);

  et_radix_kv_stats after;
  CHECK(et_radix_kv_stats_get(tree, &after) == ET_RADIX_OK);
  CHECK(after.used_tokens == before.used_tokens &&
        after.free_tokens == before.free_tokens &&
        after.pinned_tokens == before.pinned_tokens &&
        after.active_nodes == before.active_nodes &&
        after.lru_tick == before.lru_tick &&
        after.total_edge_splits == before.total_edge_splits &&
        after.total_evicted_tokens == before.total_evicted_tokens &&
        after.total_oom_rejections == before.total_oom_rejections + 2u);
  check_stats_conservation(tree);

  /* Unpin Q -> now 4 tokens evictable, inserting 5 still OOMs atomically
   * without evicting Q! */
  CHECK(et_radix_kv_unpin(tree, h_q) == ET_RADIX_OK);
  const uint32_t seq_oom5[5] = {80u, 81u, 82u, 83u, 84u};
  CHECK(et_radix_kv_insert(tree, seq_oom5, 5u, k, v, &h_fail, NULL) ==
        ET_RADIX_ERR_OOM);
  CHECK(et_radix_kv_handle_info(tree, h_q, NULL, NULL, NULL, NULL) ==
        ET_RADIX_OK);
  check_stats_conservation(tree);

  /* Node exhaustion OOM on match_prefix split when max_nodes == 2. */
  et_radix_kv_destroy(tree);
  const et_radix_kv_config tight_nodes = {8u, 2u, 1u, 2u, 4u, 0u};
  CHECK(et_radix_kv_create(&tight_nodes, &tree) == ET_RADIX_OK);
  const uint32_t s4[4] = {1u, 2u, 3u, 4u};
  et_radix_kv_handle hs4 = 0u;
  CHECK(et_radix_kv_insert(tree, s4, 4u, k, v, &hs4, NULL) == ET_RADIX_OK);
  const uint32_t s2[3] = {1u, 2u, 9u};
  et_radix_kv_match_result mr;
  CHECK(et_radix_kv_match_prefix(tree, s2, 3u, &mr, NULL, 0u) ==
        ET_RADIX_ERR_OOM);
  check_stats_conservation(tree);
  et_radix_kv_destroy(tree);
}

static void test_gqa_bitwise_parity_and_read_kv(void) {
  const et_radix_kv_config cfg = {64u, 32u, 2u, 2u, 16u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  const uint32_t full_len = 20u;
  const uint32_t prefix_len = 12u;
  const uint32_t suffix_len = full_len - prefix_len;
  const uint32_t n_q_heads = 6u; /* GQA ratio 3 */
  const float scale = 1.0f / sqrtf(16.0f);

  uint32_t tokens[20];
  for (uint32_t i = 0u; i < full_len; ++i)
    tokens[i] = 500u + i;
  float k_all[20 * 2 * 2 * 16], v_all[20 * 2 * 2 * 16];
  fill_seq_kv(full_len, 2u, 2u, 16u, 333u, k_all, v_all);

  /* Insert prefix + extra tail to force a mid-edge split when matching prefix_len. */
  uint32_t dec_tokens[16];
  for (uint32_t i = 0u; i < 12u; ++i)
    dec_tokens[i] = tokens[i];
  for (uint32_t i = 12u; i < 16u; ++i)
    dec_tokens[i] = 9900u + i;
  et_radix_kv_handle h_dec = 0u;
  CHECK(et_radix_kv_insert(tree, dec_tokens, 16u, k_all, v_all, &h_dec, NULL) ==
        ET_RADIX_OK);

  et_radix_kv_match_result mres;
  CHECK(et_radix_kv_match_prefix(tree, tokens, full_len, &mres, NULL, 0u) ==
        ET_RADIX_OK);
  CHECK(mres.matched_len == prefix_len && mres.did_split == 1u);
  CHECK(et_radix_kv_pin(tree, mres.handle) == ET_RADIX_OK);

  float q_full[20 * 6 * 16];
  for (size_t i = 0u; i < 20u * 6u * 16u; ++i)
    q_full[i] = det_val(777u, (uint32_t)i);

  for (uint32_t layer = 0u; layer < 2u; ++layer) {
    float k_l[20 * 2 * 16], v_l[20 * 2 * 16];
    extract_layer_kv(full_len, 2u, 2u, 16u, layer, k_all, v_all, k_l, v_l);

    /* Verify read_kv gathers exact bitwise K/V for cached prefix. */
    float rk[12 * 2 * 16], rv[12 * 2 * 16];
    uint32_t rlen = 0u;
    CHECK(et_radix_kv_read_kv(tree, mres.handle, layer, rk, rv, 12u * 32u,
                              &rlen) == ET_RADIX_OK);
    CHECK(rlen == prefix_len);
    CHECK(memcmp(rk, k_l, (size_t)prefix_len * 32u * sizeof(float)) == 0);
    CHECK(memcmp(rv, v_l, (size_t)prefix_len * 32u * sizeof(float)) == 0);

    /* Full uncached prefill vs cached-prefix + suffix prefill. */
    float out_full[20 * 6 * 16];
    float out_cached[8 * 6 * 16];
    const et_radix_kv_handle root_h = et_radix_kv_root_handle(tree);
    CHECK(et_radix_kv_attend_gqa_f32(tree, root_h, layer, n_q_heads, full_len,
                                     full_len, scale, q_full, k_l, v_l,
                                     out_full) == ET_RADIX_OK);
    CHECK(et_radix_kv_attend_gqa_f32(
              tree, mres.handle, layer, n_q_heads, suffix_len, suffix_len,
              scale, q_full + (size_t)prefix_len * 6u * 16u,
              k_l + (size_t)prefix_len * 32u, v_l + (size_t)prefix_len * 32u,
              out_cached) == ET_RADIX_OK);

    /* Bitwise 0.0 parity! */
    CHECK(memcmp(out_cached, out_full + (size_t)prefix_len * 6u * 16u,
                 sizeof(out_cached)) == 0);

    /* Single-token decode step (q_len = 1, suffix_len = 1) vs full prefill last
     * token after inserting tokens[0..full_len-2]. */
    float out_dec1[6 * 16];
    et_radix_kv_handle h_full = 0u;
    CHECK(et_radix_kv_insert_suffix(
              tree, mres.handle, tokens + prefix_len, suffix_len,
              k_all + (size_t)prefix_len * 64u,
              v_all + (size_t)prefix_len * 64u, &h_full, NULL) == ET_RADIX_OK);
    CHECK(et_radix_kv_attend_gqa_f32(
              tree, h_full, layer, n_q_heads, 1u, 0u, scale,
              q_full + (size_t)(full_len - 1u) * 6u * 16u, NULL, NULL,
              out_dec1) == ET_RADIX_OK);
    CHECK(memcmp(out_dec1, out_full + (size_t)(full_len - 1u) * 6u * 16u,
                 sizeof(out_dec1)) == 0);
  }

  check_stats_conservation(tree);
  et_radix_kv_destroy(tree);
}

static void test_snapshot_roundtrip_and_security(void) {
  char dir_tmpl[] = "/tmp/radix-snap-XXXXXX";
  char *dir = mkdtemp(dir_tmpl);
  CHECK(dir != NULL);

  char snap1[256], snap2[256], bad_path[256], link_path[256];
  snprintf(snap1, sizeof(snap1), "%s/cache1.etrx", dir);
  snprintf(snap2, sizeof(snap2), "%s/cache2.etrx", dir);
  snprintf(bad_path, sizeof(bad_path), "%s/corrupt.etrx", dir);
  snprintf(link_path, sizeof(link_path), "%s/symlink.etrx", dir);

  const et_radix_kv_config cfg = {32u, 16u, 2u, 2u, 4u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  const uint32_t s1[6] = {1u, 2u, 3u, 10u, 11u, 12u};
  const uint32_t s2[6] = {1u, 2u, 3u, 20u, 21u, 22u};
  const uint32_t s3[4] = {9u, 8u, 7u, 6u};
  float k1[6 * 16], v1[6 * 16], k2[6 * 16], v2[6 * 16], k3[4 * 16], v3[4 * 16];
  fill_seq_kv(6u, 2u, 2u, 4u, 11u, k1, v1);
  fill_seq_kv(6u, 2u, 2u, 4u, 22u, k2, v2);
  fill_seq_kv(4u, 2u, 2u, 4u, 33u, k3, v3);

  et_radix_kv_handle h1 = 0u, h2 = 0u, h3 = 0u;
  CHECK(et_radix_kv_insert(tree, s1, 6u, k1, v1, &h1, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_insert(tree, s2, 6u, k2, v2, &h2, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_insert(tree, s3, 4u, k3, v3, &h3, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_pin(tree, h1) == ET_RADIX_OK);

  float q[4 * 4];
  for (size_t i = 0u; i < 16u; ++i)
    q[i] = det_val(44u, (uint32_t)i);
  float out_orig[4 * 4];
  CHECK(et_radix_kv_attend_gqa_f32(tree, h2, 1u, 4u, 1u, 0u, 0.5f, q, NULL,
                                   NULL, out_orig) == ET_RADIX_OK);

  /* Save snapshot and check 0600 permissions. */
  CHECK(et_radix_kv_snapshot_save(tree, snap1) == ET_RADIX_OK);
  struct stat st_file;
  CHECK(stat(snap1, &st_file) == 0);
  CHECK((st_file.st_mode & 0777) == 0600);

  /* Load into fresh tree and verify exact match, KV parity, and canonical
   * byte-identical re-save. */
  et_radix_kv_tree *loaded = NULL;
  CHECK(et_radix_kv_snapshot_load(snap1, &loaded) == ET_RADIX_OK);
  CHECK(loaded != NULL);
  check_stats_conservation(loaded);

  et_radix_kv_stats st_orig, st_load;
  CHECK(et_radix_kv_stats_get(tree, &st_orig) == ET_RADIX_OK);
  CHECK(et_radix_kv_stats_get(loaded, &st_load) == ET_RADIX_OK);
  CHECK(st_load.used_tokens == st_orig.used_tokens &&
        st_load.active_nodes == st_orig.active_nodes &&
        st_load.lru_tick == st_orig.lru_tick &&
        st_load.pinned_tokens == 0u);

  CHECK(et_radix_kv_snapshot_save(loaded, snap2) == ET_RADIX_OK);
  FILE *f1 = fopen(snap1, "rb");
  FILE *f2 = fopen(snap2, "rb");
  CHECK(f1 != NULL && f2 != NULL);
  const size_t snap_sz = (size_t)st_file.st_size;
  uint8_t *b1 = (uint8_t *)malloc(snap_sz);
  uint8_t *b2 = (uint8_t *)malloc(snap_sz);
  CHECK(b1 != NULL && b2 != NULL);
  CHECK(fread(b1, 1u, snap_sz, f1) == snap_sz);
  CHECK(fread(b2, 1u, snap_sz, f2) == snap_sz);
  fclose(f1);
  fclose(f2);
  CHECK(memcmp(b1, b2, snap_sz) == 0);

  et_radix_kv_match_result mr;
  CHECK(et_radix_kv_match_prefix(loaded, s2, 6u, &mr, NULL, 0u) == ET_RADIX_OK);
  CHECK(mr.matched_len == 6u && mr.did_split == 0u);
  float out_loaded[4 * 4];
  CHECK(et_radix_kv_attend_gqa_f32(loaded, mr.handle, 1u, 4u, 1u, 0u, 0.5f, q,
                                   NULL, NULL, out_loaded) == ET_RADIX_OK);
  CHECK(memcmp(out_orig, out_loaded, sizeof(out_orig)) == 0);

  /* Symlink rejection (O_NOFOLLOW). */
  CHECK(symlink(snap1, link_path) == 0);
  et_radix_kv_tree *bad = NULL;
  CHECK(et_radix_kv_snapshot_load(link_path, &bad) == ET_RADIX_ERR_SYMLINK);
  CHECK(bad == NULL);

  /* Bit-flip checksum rejection. */
  b1[70] ^= 0x04u;
  FILE *fb = fopen(bad_path, "wb");
  CHECK(fb != NULL && fwrite(b1, 1u, snap_sz, fb) == snap_sz);
  fclose(fb);
  CHECK(et_radix_kv_snapshot_load(bad_path, &bad) == ET_RADIX_ERR_CHECKSUM);
  CHECK(bad == NULL);
  b1[70] ^= 0x04u;

  /* Truncation rejection. */
  fb = fopen(bad_path, "wb");
  CHECK(fb != NULL && fwrite(b1, 1u, snap_sz - 7u, fb) == snap_sz - 7u);
  fclose(fb);
  CHECK(et_radix_kv_snapshot_load(bad_path, &bad) == ET_RADIX_ERR_TRUNCATED);

  /* Extra trailing byte rejection. */
  fb = fopen(bad_path, "wb");
  CHECK(fb != NULL && fwrite(b1, 1u, snap_sz, fb) == snap_sz);
  uint8_t extra = 0xAAu;
  CHECK(fwrite(&extra, 1u, 1u, fb) == 1u);
  fclose(fb);
  CHECK(et_radix_kv_snapshot_load(bad_path, &bad) == ET_RADIX_ERR_FORMAT);

  /* Bad magic rejection. */
  b1[0] = 'X';
  fb = fopen(bad_path, "wb");
  CHECK(fb != NULL && fwrite(b1, 1u, snap_sz, fb) == snap_sz);
  fclose(fb);
  CHECK(et_radix_kv_snapshot_load(bad_path, &bad) == ET_RADIX_ERR_FORMAT);

  free(b1);
  free(b2);
  unlink(snap1);
  unlink(snap2);
  unlink(bad_path);
  unlink(link_path);
  rmdir(dir);
  et_radix_kv_destroy(loaded);
  et_radix_kv_destroy(tree);
}

static void test_negative_boundaries(void) {
  const et_radix_kv_config cfg = {16u, 8u, 2u, 2u, 4u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  const uint32_t tok[2] = {1u, 2u};
  float k[2 * 16], v[2 * 16];
  fill_seq_kv(2u, 2u, 2u, 4u, 5u, k, v);
  et_radix_kv_handle h = 0u;

  /* Non-finite K and V rejected atomically. */
  k[3] = NAN;
  CHECK(et_radix_kv_insert(tree, tok, 2u, k, v, &h, NULL) ==
        ET_RADIX_ERR_NONFINITE);
  k[3] = 0.25f;
  v[7] = INFINITY;
  CHECK(et_radix_kv_insert(tree, tok, 2u, k, v, &h, NULL) ==
        ET_RADIX_ERR_NONFINITE);
  v[7] = -0.25f;

  /* Valid insert, then negative checks on attend / read_kv / match / pin. */
  CHECK(et_radix_kv_insert(tree, tok, 2u, k, v, &h, NULL) == ET_RADIX_OK);
  CHECK(et_radix_kv_pin(tree, ET_RADIX_KV_HANDLE_INVALID) ==
        ET_RADIX_ERR_INVALID_HANDLE);
  CHECK(et_radix_kv_pin(tree, UINT64_C(0xdeadbeef00000001)) ==
        ET_RADIX_ERR_INVALID_HANDLE);

  et_radix_kv_match_result mr;
  uint32_t one_slot[1];
  CHECK(et_radix_kv_match_prefix(tree, tok, 2u, &mr, one_slot, 1u) ==
        ET_RADIX_ERR_SHAPE);
  CHECK(et_radix_kv_match_prefix(tree, tok, 2u, &mr, NULL, 4u) ==
        ET_RADIX_ERR_ARGUMENT);

  float rk[16], rv[16];
  CHECK(et_radix_kv_read_kv(tree, h, 2u, rk, rv, 16u, NULL) ==
        ET_RADIX_ERR_SHAPE);
  CHECK(et_radix_kv_read_kv(tree, h, 0u, rk, rk, 16u, NULL) ==
        ET_RADIX_ERR_ALIAS);
  CHECK(et_radix_kv_read_kv(tree, h, 0u, rk, rv, 8u, NULL) ==
        ET_RADIX_ERR_SHAPE);

  float q[16], out[16];
  for (size_t i = 0u; i < 16u; ++i)
    q[i] = 0.125f;
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 3u, 1u, 0u, 0.5f, q, NULL, NULL,
                                   out) == ET_RADIX_ERR_SHAPE);
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 4u, 3u, 0u, 0.5f, q, NULL, NULL,
                                   out) == ET_RADIX_ERR_SHAPE);
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 4u, 1u, 0u, 0.0f, q, NULL, NULL,
                                   out) == ET_RADIX_ERR_NONFINITE);
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 4u, 1u, 0u, 0.5f, q, NULL, NULL,
                                   q) == ET_RADIX_ERR_ALIAS);
  q[2] = NAN;
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 4u, 1u, 0u, 0.5f, q, NULL, NULL,
                                   out) == ET_RADIX_ERR_NONFINITE);

  /* Unaligned pointer check. */
  uint8_t raw[128];
  memset(raw, 0, sizeof(raw));
  const float *misaligned = (const float *)(const void *)(raw + 1u);
  CHECK(et_radix_kv_attend_gqa_f32(tree, h, 0u, 4u, 1u, 0u, 0.5f, misaligned,
                                   NULL, NULL, out) == ET_RADIX_ERR_ALIGNMENT);

  check_stats_conservation(tree);
  et_radix_kv_destroy(tree);
}

static uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13u;
  x ^= x >> 17u;
  x ^= x << 5u;
  *state = x;
  return x;
}

static void test_fuzz_stress_invariants(void) {
  const et_radix_kv_config cfg = {48u, 24u, 1u, 2u, 4u, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  uint32_t rng = 0x12345678u;
  et_radix_kv_handle tracked[32];
  uint32_t pinned_flags[32];
  uint32_t tracked_cnt = 0u;
  memset(tracked, 0, sizeof(tracked));
  memset(pinned_flags, 0, sizeof(pinned_flags));

  uint32_t tokens[12];
  float k[12 * 8], v[12 * 8];
  float q[16], out[16];
  for (size_t i = 0u; i < 16u; ++i)
    q[i] = 0.0625f * (float)((int)(i % 7u) - 3);

  for (uint32_t step = 0u; step < 500u; ++step) {
    const uint32_t op = xorshift32(&rng) % 6u;
    if (op == 0u || op == 1u) {
      /* Insert structured hierarchical sequence (encourages prefix sharing + splits). */
      const uint32_t len = 1u + (xorshift32(&rng) % 10u);
      const uint32_t sys_id = xorshift32(&rng) % 3u;
      const uint32_t branch_id = xorshift32(&rng) % 4u;
      for (uint32_t i = 0u; i < len; ++i) {
        if (i < 3u)
          tokens[i] = 10u * sys_id + i;
        else if (i < 6u)
          tokens[i] = 100u * branch_id + i;
        else
          tokens[i] = 1000u + (xorshift32(&rng) % 6u);
      }
      fill_seq_kv(len, 1u, 2u, 4u, step + 1u, k, v);
      et_radix_kv_handle h = 0u;
      const et_radix_kv_status st =
          et_radix_kv_insert(tree, tokens, len, k, v, &h, NULL);
      CHECK(st == ET_RADIX_OK || st == ET_RADIX_ERR_OOM);
      if (st == ET_RADIX_OK && tracked_cnt < 32u) {
        int dup = 0;
        for (uint32_t j = 0u; j < tracked_cnt; ++j) {
          if (tracked[j] == h) {
            dup = 1;
            break;
          }
        }
        if (!dup) {
          tracked[tracked_cnt] = h;
          pinned_flags[tracked_cnt] = 0u;
          tracked_cnt++;
        }
      }
    } else if (op == 2u) {
      /* Match prefix (may split interior edge). */
      const uint32_t len = 1u + (xorshift32(&rng) % 8u);
      const uint32_t sys_id = xorshift32(&rng) % 3u;
      for (uint32_t i = 0u; i < len; ++i)
        tokens[i] = (i < 3u) ? (10u * sys_id + i) : (100u * (xorshift32(&rng) % 4u) + i);
      et_radix_kv_match_result mr;
      uint32_t slots[12];
      const et_radix_kv_status st =
          et_radix_kv_match_prefix(tree, tokens, len, &mr, slots, 12u);
      CHECK(st == ET_RADIX_OK || st == ET_RADIX_ERR_OOM);
      if (st == ET_RADIX_OK && mr.matched_len > 0u) {
        CHECK(et_radix_kv_attend_gqa_f32(tree, mr.handle, 0u, 4u, 1u, 0u, 0.5f,
                                         q, NULL, NULL, out) == ET_RADIX_OK);
      }
    } else if (op == 3u && tracked_cnt > 0u) {
      /* Pin a tracked handle (may have been evicted!). */
      const uint32_t idx = xorshift32(&rng) % tracked_cnt;
      const et_radix_kv_status st = et_radix_kv_pin(tree, tracked[idx]);
      if (st == ET_RADIX_OK)
        pinned_flags[idx]++;
      else
        CHECK(st == ET_RADIX_ERR_INVALID_HANDLE);
    } else if (op == 4u && tracked_cnt > 0u) {
      /* Unpin a tracked handle. */
      const uint32_t idx = xorshift32(&rng) % tracked_cnt;
      const et_radix_kv_status st = et_radix_kv_unpin(tree, tracked[idx]);
      if (pinned_flags[idx] > 0u) {
        CHECK(st == ET_RADIX_OK);
        pinned_flags[idx]--;
      } else {
        CHECK(st == ET_RADIX_ERR_DOUBLE_UNPIN ||
              st == ET_RADIX_ERR_INVALID_HANDLE);
      }
    } else if (op == 5u) {
      const uint32_t target = xorshift32(&rng) % 24u;
      uint32_t evicted = 0u;
      CHECK(et_radix_kv_evict_lru(tree, target, &evicted) == ET_RADIX_OK);
    }

    /* Full invariant check after EVERY SINGLE fuzz operation! */
    check_stats_conservation(tree);
  }

  /* Release all remaining pins and evict everything down to 0 used tokens. */
  for (uint32_t i = 0u; i < tracked_cnt; ++i) {
    while (pinned_flags[i] > 0u) {
      CHECK(et_radix_kv_unpin(tree, tracked[i]) == ET_RADIX_OK);
      pinned_flags[i]--;
    }
  }
  uint32_t evicted_all = 0u;
  CHECK(et_radix_kv_evict_lru(tree, cfg.max_kv_tokens, &evicted_all) ==
        ET_RADIX_OK);
  et_radix_kv_stats st;
  CHECK(et_radix_kv_stats_get(tree, &st) == ET_RADIX_OK);
  CHECK(st.used_tokens == 0u && st.free_tokens == cfg.max_kv_tokens &&
        st.active_nodes == 1u);
  check_stats_conservation(tree);
  et_radix_kv_destroy(tree);
}

/* 50-op deterministic multi-request trace emitted for independent Python
 * RadixTree + LRU + binary64 GQA oracle cross-checking. */
static void print_hex_floats(const float *arr, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    if (i > 0u)
      putchar(' ');
    printf("%a", (double)arr[i]);
  }
  putchar('\n');
}

static void run_oracle_trace(void) {
  const uint32_t max_tok = 32u;
  const uint32_t max_nodes = 24u;
  const uint32_t n_layers = 1u;
  const uint32_t n_kv_heads = 2u;
  const uint32_t n_q_heads = 4u;
  const uint32_t head_dim = 4u;
  const float scale = 0.5f;

  const et_radix_kv_config cfg = {max_tok, max_nodes, n_layers, n_kv_heads,
                                  head_dim, 0u};
  et_radix_kv_tree *tree = NULL;
  CHECK(et_radix_kv_create(&cfg, &tree) == ET_RADIX_OK);

  printf("CONFIG %u %u %u %u %u %a\n", max_tok, max_nodes, n_q_heads,
         n_kv_heads, head_dim, (double)scale);

  uint32_t rng = 0x51a7f00du;
  et_radix_kv_handle handles[64];
  uint32_t handle_count = 0u;
  memset(handles, 0, sizeof(handles));

  for (uint32_t step = 0u; step < 50u; ++step) {
    const uint32_t kind = step < 6u ? 0u : (xorshift32(&rng) % 5u);
    if (kind == 0u || kind == 1u) {
      /* INSERT */
      const uint32_t len = 2u + (xorshift32(&rng) % 7u);
      const uint32_t sys = xorshift32(&rng) % 2u;
      const uint32_t sub = xorshift32(&rng) % 3u;
      uint32_t toks[12];
      for (uint32_t i = 0u; i < len; ++i) {
        if (i < 2u)
          toks[i] = 10u * sys + i;
        else if (i < 4u)
          toks[i] = 100u * sub + i;
        else
          toks[i] = 500u + (xorshift32(&rng) % 4u);
      }
      float k[12 * 8], v[12 * 8];
      fill_seq_kv(len, n_layers, n_kv_heads, head_dim, 1000u + step * 3u, k, v);
      et_radix_kv_handle h = 0u;
      uint32_t matched_ins = 0u;
      const et_radix_kv_status st =
          et_radix_kv_insert(tree, toks, len, k, v, &h, &matched_ins);
      check_stats_conservation(tree);
      int32_t hid = -1;
      if (st == ET_RADIX_OK) {
        hid = (int32_t)handle_count;
        handles[handle_count++] = h;
      }
      et_radix_kv_stats stats;
      CHECK(et_radix_kv_stats_get(tree, &stats) == ET_RADIX_OK);
      printf("OP %u INSERT %d %u %d %u %u %u %u %u %llu %llu", step, st,
             matched_ins, hid, stats.used_tokens, stats.free_tokens,
             stats.pinned_tokens, stats.evictable_tokens, stats.active_nodes,
             (unsigned long long)stats.total_evicted_tokens,
             (unsigned long long)stats.lru_tick);
      printf(" TOKENS %u", len);
      for (uint32_t i = 0u; i < len; ++i)
        printf(" %u", toks[i]);
      putchar('\n');
      print_hex_floats(k, (size_t)len * 8u);
      print_hex_floats(v, (size_t)len * 8u);
    } else if (kind == 2u) {
      /* MATCH + GQA ATTEND if matched > 0 */
      const uint32_t len = 1u + (xorshift32(&rng) % 7u);
      const uint32_t sys = xorshift32(&rng) % 2u;
      const uint32_t sub = xorshift32(&rng) % 3u;
      uint32_t toks[12];
      for (uint32_t i = 0u; i < len; ++i) {
        if (i < 2u)
          toks[i] = 10u * sys + i;
        else if (i < 4u)
          toks[i] = 100u * sub + i;
        else
          toks[i] = 500u + (xorshift32(&rng) % 4u);
      }
      et_radix_kv_match_result mr;
      memset(&mr, 0, sizeof(mr));
      const et_radix_kv_status st =
          et_radix_kv_match_prefix(tree, toks, len, &mr, NULL, 0u);
      check_stats_conservation(tree);
      int32_t hid = -1;
      if (st == ET_RADIX_OK) {
        hid = (int32_t)handle_count;
        handles[handle_count++] = mr.handle;
      }
      et_radix_kv_stats stats;
      CHECK(et_radix_kv_stats_get(tree, &stats) == ET_RADIX_OK);
      const uint32_t do_attn =
          (st == ET_RADIX_OK && mr.matched_len > 0u) ? 1u : 0u;
      printf("OP %u MATCH %d %u %u %d %u %u %u %u %u %llu %llu %u TOKENS %u",
             step, st, mr.matched_len, mr.did_split, hid, stats.used_tokens,
             stats.free_tokens, stats.pinned_tokens, stats.evictable_tokens,
             stats.active_nodes, (unsigned long long)stats.total_evicted_tokens,
             (unsigned long long)stats.lru_tick, do_attn, len);
      for (uint32_t i = 0u; i < len; ++i)
        printf(" %u", toks[i]);
      putchar('\n');
      if (do_attn) {
        /* 2-token suffix GQA attention over cached prefix + 2 new suffix tokens. */
        float q[2 * 4 * 4], sk[2 * 2 * 4], sv[2 * 2 * 4], out[2 * 4 * 4];
        for (size_t i = 0u; i < 32u; ++i)
          q[i] = det_val(2000u + step, (uint32_t)i);
        fill_seq_kv(2u, n_layers, n_kv_heads, head_dim, 3000u + step, sk, sv);
        CHECK(et_radix_kv_attend_gqa_f32(tree, mr.handle, 0u, n_q_heads, 2u, 2u,
                                         scale, q, sk, sv, out) == ET_RADIX_OK);
        print_hex_floats(q, 32u);
        print_hex_floats(sk, 16u);
        print_hex_floats(sv, 16u);
        print_hex_floats(out, 32u);
      }
    } else if (kind == 3u && handle_count > 0u) {
      /* PIN */
      const uint32_t hid = xorshift32(&rng) % handle_count;
      const et_radix_kv_status st = et_radix_kv_pin(tree, handles[hid]);
      check_stats_conservation(tree);
      et_radix_kv_stats stats;
      CHECK(et_radix_kv_stats_get(tree, &stats) == ET_RADIX_OK);
      printf("OP %u PIN %d %u %u %u %u %u %u %llu %llu\n", step, st, hid,
             stats.used_tokens, stats.free_tokens, stats.pinned_tokens,
             stats.evictable_tokens, stats.active_nodes,
             (unsigned long long)stats.total_evicted_tokens,
             (unsigned long long)stats.lru_tick);
    } else if (handle_count > 0u) {
      /* UNPIN */
      const uint32_t hid = xorshift32(&rng) % handle_count;
      const et_radix_kv_status st = et_radix_kv_unpin(tree, handles[hid]);
      check_stats_conservation(tree);
      et_radix_kv_stats stats;
      CHECK(et_radix_kv_stats_get(tree, &stats) == ET_RADIX_OK);
      printf("OP %u UNPIN %d %u %u %u %u %u %u %llu %llu\n", step, st, hid,
             stats.used_tokens, stats.free_tokens, stats.pinned_tokens,
             stats.evictable_tokens, stats.active_nodes,
             (unsigned long long)stats.total_evicted_tokens,
             (unsigned long long)stats.lru_tick);
    }
  }

  et_radix_kv_destroy(tree);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--oracle") == 0) {
    run_oracle_trace();
    return 0;
  }

  test_config_and_lifecycle();
  test_splits_branching_and_pins();
  test_lru_eviction_cascade_and_oom_rollback();
  test_gqa_bitwise_parity_and_read_kv();
  test_snapshot_roundtrip_and_security();
  test_negative_boundaries();
  test_fuzz_stress_invariants();
  printf("RADIX_KV native C11 tests (splits, LRU cascade, OOM rollback, GQA "
         "parity, ETRX1000 snapshot, 500-op fuzz): PASS\n");
  return 0;
}
