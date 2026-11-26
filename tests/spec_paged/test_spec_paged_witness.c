/* Development-only composition witness. KV values come from a deterministic
 * mock producer; this does not call a draft or target model. */
#include "eshkol_transformer/paged_kv_abi.h"
#include "eshkol_transformer/spec_decode_abi.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #x); \
  return 1; \
} } while (0)

enum { BLOCKS = 8, PREFIX = 3, HORIZON = 2, DIM = 2 };

static void mock_kv(uint32_t token, uint32_t pos, float k[DIM], float v[DIM]) {
  k[0] = (float)(token * 16u + pos);
  k[1] = (float)(token * 16u + pos + 1u);
  v[0] = -k[0];
  v[1] = -k[1];
}

static int append(et_paged_kv_pool *pool, uint32_t seq, uint32_t token) {
  uint32_t pos = UINT32_MAX;
  float k[DIM], v[DIM];
  CHECK(et_paged_kv_seq_len(pool, seq, &pos) == ET_PAGED_KV_OK);
  mock_kv(token, pos, k, v);
  CHECK(et_paged_kv_append_token(pool, seq, k, v) == ET_PAGED_KV_OK);
  return 0;
}

static int check_tokens(const et_paged_kv_pool *pool, uint32_t seq,
                        const uint32_t *tokens, uint32_t count) {
  uint32_t len = UINT32_MAX;
  CHECK(et_paged_kv_seq_len(pool, seq, &len) == ET_PAGED_KV_OK);
  CHECK(len == count);
  for (uint32_t pos = 0; pos < count; ++pos) {
    float expected_k[DIM], expected_v[DIM], k[DIM], v[DIM];
    mock_kv(tokens[pos], pos, expected_k, expected_v);
    CHECK(et_paged_kv_read_token(pool, seq, pos, 0u, 0u, k, v) == ET_PAGED_KV_OK);
    CHECK(memcmp(k, expected_k, sizeof k) == 0);
    CHECK(memcmp(v, expected_v, sizeof v) == 0);
  }
  return 0;
}

static int check_ownership(const et_paged_kv_pool *pool,
                           const uint32_t seqs[3]) {
  uint32_t refs[BLOCKS] = {0};
  uint32_t tables[3][4], counts[3], used = 0, shared = 0;
  et_paged_kv_stats stats;
  for (uint32_t i = 0; i < 3u; ++i) {
    CHECK(et_paged_kv_seq_block_table(pool, seqs[i], tables[i], 4u,
                                      &counts[i]) == ET_PAGED_KV_OK);
    for (uint32_t j = 0; j < counts[i]; ++j) {
      CHECK(tables[i][j] < BLOCKS);
      ++refs[tables[i][j]];
    }
  }
  /* The complete first block is shared; writes to the partial second block
   * split the anchor, control and speculative sequences by CoW. */
  CHECK(tables[0][0] == tables[1][0] && tables[1][0] == tables[2][0]);
  CHECK(tables[0][1] != tables[1][1]);
  CHECK(tables[0][1] != tables[2][1]);
  CHECK(tables[1][1] != tables[2][1]);
  CHECK(counts[1] == counts[2]);
  if (counts[1] == 3u) CHECK(tables[1][2] != tables[2][2]);
  for (uint32_t b = 0; b < BLOCKS; ++b) {
    uint32_t actual = UINT32_MAX;
    CHECK(et_paged_kv_block_refcount(pool, b, &actual) == ET_PAGED_KV_OK);
    CHECK(actual == refs[b]);
    used += refs[b] != 0u;
    shared += refs[b] > 1u;
  }
  CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
  CHECK(stats.active_seqs == 3u);
  CHECK(stats.total_blocks == BLOCKS);
  CHECK(stats.used_blocks == used && stats.free_blocks == BLOCKS - used);
  CHECK(stats.shared_blocks == shared);
  CHECK(stats.prefix_cached_blocks == 0u);
  return 0;
}

static int run_case(int reject, int stochastic) {
  const et_paged_kv_config cfg = {BLOCKS, 1u, 1u, 2u, DIM, 3u, 4u, 0u};
  const uint32_t prefix[PREFIX] = {1u, 2u, 3u};
  const uint32_t draft[HORIZON] = {4u, reject ? 9u : 5u};
  const uint32_t target[HORIZON + 1] = {4u, reject ? 7u : 5u,
                                               reject ? 8u : 6u};
  uint32_t expected[PREFIX + HORIZON + 1] = {1u, 2u, 3u, 4u,
                                             reject ? 7u : 5u, 6u};
  float target_logits[(HORIZON + 1) * 16u];
  float draft_logits[HORIZON * 16u];
  uint32_t emitted[HORIZON + 1] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  const et_spec_sampling_params_v1 params = {1.0, 1.0, 1u, ET_SPEC_NO_EOS};
  const et_spec_rng_state_v1 rng = {17u, 0u};
  et_spec_verify_result_v1 result;
  et_paged_kv_pool *pool = NULL;
  uint32_t seqs[3]; /* anchor, uninterrupted control, speculative sequence */
  float draft_k[HORIZON * DIM], draft_v[HORIZON * DIM];
  uint32_t final_count;
  uint32_t draft_blocks[4], draft_block_count, tail_refcount;
  et_paged_kv_stats before_truncate, after_truncate;

  for (uint32_t row = 0; row < HORIZON + 1u; ++row)
    for (uint32_t col = 0; col < 16u; ++col)
      target_logits[row * 16u + col] = col == target[row] ? 10.0f : -10.0f;
  for (uint32_t row = 0; row < HORIZON; ++row)
    for (uint32_t col = 0; col < 16u; ++col)
      draft_logits[row * 16u + col] = col == draft[row] ? 10.0f : -10.0f;

  CHECK(et_paged_kv_create(&cfg, NULL, 0u, &pool) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(pool, &seqs[0]) == ET_PAGED_KV_OK);
  for (uint32_t i = 0; i < PREFIX; ++i) CHECK(append(pool, seqs[0], prefix[i]) == 0);
  CHECK(et_paged_kv_seq_fork(pool, seqs[0], &seqs[1]) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_fork(pool, seqs[0], &seqs[2]) == ET_PAGED_KV_OK);

  final_count = reject ? PREFIX + 2u : PREFIX + 3u;
  for (uint32_t i = PREFIX; i < final_count; ++i)
    CHECK(append(pool, seqs[1], expected[i]) == 0);
  for (uint32_t i = 0; i < HORIZON; ++i)
    mock_kv(draft[i], PREFIX + i, &draft_k[i * DIM], &draft_v[i * DIM]);
  CHECK(et_paged_kv_append_tokens(pool, seqs[2], HORIZON, draft_k, draft_v) ==
        ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_block_table(pool, seqs[2], draft_blocks, 4u,
                                    &draft_block_count) == ET_PAGED_KV_OK);
  CHECK(draft_block_count == 3u);
  CHECK(et_paged_kv_stats_get(pool, &before_truncate) == ET_PAGED_KV_OK);

  if (stochastic) {
    CHECK(et_spec_verify_sample_v1(draft, draft_logits, target_logits, HORIZON,
                                    16u, &params, &rng, emitted, HORIZON + 1u,
                                    &result) == ET_SPEC_OK);
    CHECK(result.rng_draws == 3u && result.next_rng.counter == 3u);
    CHECK(result.next_rng.seed == rng.seed);
  } else {
    CHECK(et_spec_verify_greedy_v1(draft, target_logits, HORIZON, 16u,
                                    ET_SPEC_NO_EOS, emitted, HORIZON + 1u,
                                    &result) == ET_SPEC_OK);
    CHECK(result.rng_draws == 0u);
  }
  CHECK(result.accepted_count == (reject ? 1u : 2u));
  CHECK(result.rejected == (uint32_t)reject);
  CHECK(result.emitted_count == (reject ? 2u : 3u));
  CHECK(result.eos_hit == 0u);
  for (uint32_t i = 0; i < result.emitted_count; ++i)
    CHECK(emitted[i] == expected[PREFIX + i]);

  /* Discard unaccepted draft KV. The emitted tail is a mock target KV value,
   * staged only after the verifier has selected its correction or bonus. */
  CHECK(et_paged_kv_seq_truncate(pool, seqs[2],
                                 PREFIX + result.accepted_count) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &after_truncate) == ET_PAGED_KV_OK);
  if (reject) {
    CHECK(et_paged_kv_block_refcount(pool, draft_blocks[2], &tail_refcount) ==
          ET_PAGED_KV_OK);
    CHECK(tail_refcount == 0u);
    CHECK(after_truncate.free_blocks == before_truncate.free_blocks + 1u);
    CHECK(after_truncate.used_blocks + 1u == before_truncate.used_blocks);
  }
  CHECK(check_tokens(pool, seqs[2], expected,
                     PREFIX + result.accepted_count) == 0);
  for (uint32_t i = result.accepted_count; i < result.emitted_count; ++i)
    CHECK(append(pool, seqs[2], emitted[i]) == 0);

  CHECK(check_tokens(pool, seqs[0], prefix, PREFIX) == 0);
  CHECK(check_tokens(pool, seqs[1], expected, final_count) == 0);
  CHECK(check_tokens(pool, seqs[2], expected, final_count) == 0);
  CHECK(check_ownership(pool, seqs) == 0);
  for (uint32_t i = 0; i < 3u; ++i)
    CHECK(et_paged_kv_seq_free(pool, seqs[i]) == ET_PAGED_KV_OK);
  et_paged_kv_stats stats;
  CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
  CHECK(stats.active_seqs == 0u && stats.used_blocks == 0u &&
        stats.shared_blocks == 0u && stats.free_blocks == BLOCKS);
  et_paged_kv_destroy(pool);
  printf("SPEC_PAGED %s %s accepted=%u emitted=%u rollback=%u ownership=PASS\n",
         stochastic ? "sample" : "greedy", reject ? "reject" : "accept",
         result.accepted_count, result.emitted_count, reject ? 1u : 0u);
  return 0;
}

int main(void) {
  CHECK(run_case(0, 0) == 0);
  CHECK(run_case(1, 0) == 0);
  CHECK(run_case(0, 1) == 0);
  CHECK(run_case(1, 1) == 0);
  return 0;
}
