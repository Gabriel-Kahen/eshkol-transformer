#include "eshkol_transformer/spec_paged_round_abi.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x)                                                                       \
  do {                                                                                 \
    if (!(x)) {                                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);                          \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

enum {
  DRAFT,
  TARGET,
  MODELS,
  BLOCKS = 8,
  BLOCK_SIZE = 2,
  PREFIX = 3,
  HORIZON = 3,
  VOCAB = 16,
  DRAFT_WIDTH = 2,
  MAX_WIDTH = 4,
  MAX_TOKENS = PREFIX + HORIZON + 1,
  DRAFT_BLOCK_FLOATS = 2 * BLOCK_SIZE * DRAFT_WIDTH
};

static const uint32_t prefix[PREFIX] = {1u, 2u, 3u};
static const uint32_t draft_tokens[HORIZON] = {4u, 5u, 6u};
static const uint32_t full_target[HORIZON + 1] = {4u, 5u, 6u, 7u};

typedef struct fixture {
  et_spec_paged_round_request_v1 request;
  et_paged_kv_pool *pool[MODELS];
  uint32_t parent[MODELS];
  uint32_t parent_table[MODELS][2];
  size_t width[MODELS];
  float draft_storage[BLOCKS * DRAFT_BLOCK_FLOATS];
  float candidate_k[MODELS][HORIZON * MAX_WIDTH];
  float candidate_v[MODELS][HORIZON * MAX_WIDTH];
  float logits[(HORIZON + 1) * VOCAB];
  float tail_k[MODELS][MAX_WIDTH];
  float tail_v[MODELS][MAX_WIDTH];
  uint32_t expected[MAX_TOKENS];
  uint32_t expected_tail_pos;
  uint32_t callback_calls;
  int32_t callback_status;
  int bad_tail_shape;
} fixture;

static void produce_kv(size_t model, uint32_t token, uint32_t pos, size_t width,
                       float *keys, float *values) {
  for (size_t i = 0; i < width; ++i) {
    keys[i] = (float)(model * 4096u + token * 64u + pos * 4u + i);
    values[i] = -keys[i];
  }
}

static int check_sequence(const et_paged_kv_pool *pool, uint32_t seq, size_t model,
                          const uint32_t *tokens, uint32_t count) {
  et_paged_kv_config cfg;
  uint32_t len;
  CHECK(et_paged_kv_config_get(pool, &cfg) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_len(pool, seq, &len) == ET_PAGED_KV_OK);
  CHECK(len == count);
  for (uint32_t pos = 0; pos < count; ++pos) {
    float expected_k[MAX_WIDTH], expected_v[MAX_WIDTH];
    produce_kv(model, tokens[pos], pos,
               (size_t)cfg.num_layers * cfg.num_kv_heads * cfg.head_dim, expected_k,
               expected_v);
    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
      for (uint32_t head = 0; head < cfg.num_kv_heads; ++head) {
        float keys[MAX_WIDTH], values[MAX_WIDTH];
        const size_t offset = (layer * cfg.num_kv_heads + head) * cfg.head_dim;
        CHECK(et_paged_kv_read_token(pool, seq, pos, layer, head, keys, values) ==
              ET_PAGED_KV_OK);
        CHECK(memcmp(keys, expected_k + offset, cfg.head_dim * sizeof(float)) == 0);
        CHECK(memcmp(values, expected_v + offset, cfg.head_dim * sizeof(float)) == 0);
      }
    }
  }
  return 0;
}

static int32_t recompute(void *context, uint32_t token, uint32_t position,
                         et_spec_paged_round_prefix_v1 draft,
                         et_spec_paged_round_prefix_v1 target,
                         et_spec_paged_kv_span_v1 *out_draft,
                         et_spec_paged_kv_span_v1 *out_target) {
  fixture *f = context;
  const et_spec_paged_round_prefix_v1 views[MODELS] = {draft, target};
  ++f->callback_calls;
  CHECK(position == f->expected_tail_pos && position < MAX_TOKENS);
  CHECK(token == f->expected[position]);
  for (size_t i = 0; i < MODELS; ++i) {
    CHECK(views[i].pool == f->pool[i] && views[i].seq_id != f->parent[i]);
    CHECK(check_sequence(views[i].pool, views[i].seq_id, i, f->expected, position) ==
          0);
    produce_kv(i, token, position, f->width[i], f->tail_k[i], f->tail_v[i]);
  }
  *out_draft =
      (et_spec_paged_kv_span_v1){f->tail_k[DRAFT], f->tail_v[DRAFT], f->width[DRAFT]};
  *out_target = (et_spec_paged_kv_span_v1){f->tail_k[TARGET], f->tail_v[TARGET],
                                           f->width[TARGET] - (f->bad_tail_shape != 0)};
  return f->callback_status;
}

static int init_fixture(fixture *f, uint32_t target_blocks) {
  const et_paged_kv_config cfg[MODELS] = {
      {BLOCKS, 1u, 1u, BLOCK_SIZE, DRAFT_WIDTH, 2u, BLOCKS, 0u},
      {target_blocks, 2u, 2u, BLOCK_SIZE, 1u, 3u, target_blocks, 0u}};
  et_spec_paged_round_model_v1 *models[MODELS] = {&f->request.draft,
                                                  &f->request.target};
  uint8_t prefix_hash[ET_PAGED_KV_HASH_BYTES];
  CHECK(et_paged_kv_hash_token_block(NULL, prefix, BLOCK_SIZE, prefix_hash) ==
        ET_PAGED_KV_OK);
  for (size_t i = 0; i < MODELS; ++i) {
    f->width[i] = (size_t)cfg[i].num_layers * cfg[i].num_kv_heads * cfg[i].head_dim;
    float *storage = i == DRAFT ? f->draft_storage : NULL;
    const size_t storage_count = i == DRAFT ? BLOCKS * DRAFT_BLOCK_FLOATS : 0u;
    CHECK(et_paged_kv_create(&cfg[i], storage, storage_count, &f->pool[i]) ==
          ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_alloc(f->pool[i], &f->parent[i]) == ET_PAGED_KV_OK);
    for (uint32_t pos = 0; pos < PREFIX; ++pos) {
      float keys[MAX_WIDTH], values[MAX_WIDTH];
      produce_kv(i, prefix[pos], pos, f->width[i], keys, values);
      CHECK(et_paged_kv_append_token(f->pool[i], f->parent[i], keys, values) ==
            ET_PAGED_KV_OK);
    }
    uint32_t blocks;
    CHECK(et_paged_kv_seq_block_table(f->pool[i], f->parent[i], f->parent_table[i], 2u,
                                      &blocks) == ET_PAGED_KV_OK &&
          blocks == 2u);
    CHECK(et_paged_kv_prefix_cache_register(f->pool[i], f->parent[i], 0u,
                                            prefix_hash) == ET_PAGED_KV_OK);
    for (uint32_t j = 0; j < HORIZON; ++j) {
      const size_t offset = j * f->width[i];
      produce_kv(i, draft_tokens[j], PREFIX + j, f->width[i],
                 f->candidate_k[i] + offset, f->candidate_v[i] + offset);
    }
    *models[i] = (et_spec_paged_round_model_v1){
        f->pool[i],
        f->parent[i],
        {f->candidate_k[i], f->candidate_v[i], HORIZON * f->width[i]}};
  }
  f->request.draft_tokens = draft_tokens;
  f->request.target_logits = f->logits;
  f->request.target_logit_count = (HORIZON + 1u) * VOCAB;
  f->request.horizon = HORIZON;
  f->request.vocab_size = VOCAB;
  f->request.recompute = recompute;
  f->request.context = f;
  memcpy(f->expected, prefix, sizeof prefix);
  return 0;
}

static void set_case(fixture *f, const uint32_t target[HORIZON + 1], uint32_t eos,
                     uint32_t accepted, uint32_t emitted) {
  f->request.eos_token_id = eos;
  f->expected_tail_pos = PREFIX + accepted;
  memcpy(f->expected + PREFIX, target, emitted * sizeof(uint32_t));
  for (uint32_t row = 0; row <= HORIZON; ++row)
    for (uint32_t col = 0; col < VOCAB; ++col)
      f->logits[row * VOCAB + col] = col == target[row] ? 10.0f : -10.0f;
}

static int check_refs(const et_paged_kv_pool *pool, const uint32_t *ids, uint32_t count,
                      uint32_t registered) {
  et_paged_kv_stats stats;
  uint32_t refs[BLOCKS] = {0};
  uint32_t used = 0, shared = 0;
  CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
  CHECK(stats.active_seqs == count && stats.prefix_cached_blocks == registered);
  CHECK(stats.total_blocks <= BLOCKS);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t table[BLOCKS], blocks;
    CHECK(et_paged_kv_seq_block_table(pool, ids[i], table, BLOCKS, &blocks) ==
          ET_PAGED_KV_OK);
    for (uint32_t j = 0; j < blocks; ++j) {
      CHECK(table[j] < stats.total_blocks);
      ++refs[table[j]];
    }
  }
  for (uint32_t i = 0; i < stats.total_blocks; ++i) {
    uint32_t actual;
    CHECK(et_paged_kv_block_refcount(pool, i, &actual) == ET_PAGED_KV_OK);
    CHECK(actual == refs[i]);
    used += actual != 0u;
    shared += actual > 1u;
  }
  CHECK(stats.used_blocks == used && stats.free_blocks == stats.total_blocks - used);
  CHECK(stats.shared_blocks == shared);
  return 0;
}

static int check_parent(const fixture *f, size_t model) {
  uint32_t actual[2], blocks;
  CHECK(et_paged_kv_seq_block_table(f->pool[model], f->parent[model], actual, 2u,
                                    &blocks) == ET_PAGED_KV_OK &&
        blocks == 2u);
  CHECK(memcmp(actual, f->parent_table[model], sizeof actual) == 0);
  CHECK(check_sequence(f->pool[model], f->parent[model], model, prefix, PREFIX) == 0);
  return 0;
}

static int cleanup(fixture *f) {
  for (size_t i = 0; i < MODELS; ++i) {
    CHECK(et_paged_kv_seq_free(f->pool[i], f->parent[i]) == ET_PAGED_KV_OK);
    CHECK(check_refs(f->pool[i], NULL, 0u, 0u) == 0);
    et_paged_kv_destroy(f->pool[i]);
  }
  return 0;
}

static int run_success(fixture *f, uint32_t accepted, uint32_t emitted,
                       uint32_t rejected, uint32_t eos_hit) {
  const uint32_t calls_before = f->callback_calls;
  et_spec_paged_round_result_v1 result;
  et_spec_paged_round_status_v1 status =
      et_spec_paged_round_greedy_v1(&f->request, &result);
  CHECK(status.kind == ET_SPEC_PAGED_ROUND_OK && status.code == 0);
  CHECK(result.verification.accepted_count == accepted);
  CHECK(result.verification.emitted_count == emitted);
  CHECK(result.verification.rejected == rejected &&
        result.verification.eos_hit == eos_hit);
  CHECK(result.verification.rng_draws == 0u &&
        result.verification.next_rng.counter == 0u);
  CHECK(memcmp(result.tokens, f->expected + PREFIX, emitted * sizeof(uint32_t)) == 0);
  CHECK(f->callback_calls == calls_before + (emitted > accepted));

  const uint32_t children[MODELS] = {result.draft_seq_id, result.target_seq_id};
  for (size_t i = 0; i < MODELS; ++i) {
    const uint32_t ids[2] = {f->parent[i], children[i]};
    CHECK(ids[0] != ids[1]);
    CHECK(check_parent(f, i) == 0);
    CHECK(check_sequence(f->pool[i], children[i], i, f->expected, PREFIX + emitted) ==
          0);
    uint32_t child_table[BLOCKS], blocks;
    CHECK(et_paged_kv_seq_block_table(f->pool[i], children[i], child_table, BLOCKS,
                                      &blocks) == ET_PAGED_KV_OK &&
          blocks >= 2u);
    CHECK(child_table[0] == f->parent_table[i][0]);
    CHECK(child_table[1] != f->parent_table[i][1]);
    CHECK(check_refs(f->pool[i], ids, 2u, 1u) == 0);
    CHECK(et_paged_kv_seq_free(f->pool[i], children[i]) == ET_PAGED_KV_OK);
    CHECK(check_parent(f, i) == 0);
    CHECK(check_refs(f->pool[i], &f->parent[i], 1u, 1u) == 0);
  }
  return 0;
}

static int check_failure(fixture *f, int32_t kind, int32_t code,
                         uint32_t callback_calls, const uint32_t *extra_target,
                         uint32_t extra_count) {
  et_spec_paged_round_result_v1 result;
  unsigned char initial[sizeof result];
  memset(&result, 0xa5, sizeof result);
  memcpy(initial, &result, sizeof initial);
  const uint32_t calls_before = f->callback_calls;
  et_spec_paged_round_status_v1 status =
      et_spec_paged_round_greedy_v1(&f->request, &result);
  CHECK(status.kind == kind && status.code == code);
  CHECK(memcmp(&result, initial, sizeof initial) == 0);
  CHECK(f->callback_calls == calls_before + callback_calls);
  for (size_t i = 0; i < MODELS; ++i)
    CHECK(check_parent(f, i) == 0);
  CHECK(check_refs(f->pool[DRAFT], &f->parent[DRAFT], 1u, 1u) == 0);
  uint32_t target_ids[3] = {f->parent[TARGET]};
  for (uint32_t i = 0; i < extra_count; ++i)
    target_ids[i + 1u] = extra_target[i];
  CHECK(check_refs(f->pool[TARGET], target_ids, 1u + extra_count, 1u) == 0);
  return 0;
}

static int test_accept_and_reject(void) {
  fixture f = {0};
  const uint32_t partial[HORIZON + 1] = {4u, 9u, 6u, 7u};
  const uint32_t first[HORIZON + 1] = {9u, 5u, 6u, 7u};
  CHECK(init_fixture(&f, BLOCKS) == 0);
  set_case(&f, full_target, ET_SPEC_NO_EOS, 3u, 4u);
  CHECK(run_success(&f, 3u, 4u, 0u, 0u) == 0);
  set_case(&f, partial, ET_SPEC_NO_EOS, 1u, 2u);
  CHECK(run_success(&f, 1u, 2u, 1u, 0u) == 0);
  set_case(&f, full_target, 4u, 1u, 1u);
  f.request.recompute = NULL;
  CHECK(run_success(&f, 1u, 1u, 0u, 1u) == 0);
  f.request.recompute = recompute;
  set_case(&f, first, 9u, 0u, 1u);
  CHECK(run_success(&f, 0u, 1u, 1u, 1u) == 0);
  set_case(&f, full_target, 7u, 3u, 4u);
  CHECK(run_success(&f, 3u, 4u, 0u, 1u) == 0);
  CHECK(cleanup(&f) == 0);
  return 0;
}

static int test_validate_and_retry(void) {
  fixture f = {0};
  const uint32_t partial[HORIZON + 1] = {4u, 9u, 6u, 7u};
  CHECK(init_fixture(&f, BLOCKS) == 0);
  set_case(&f, partial, ET_SPEC_NO_EOS, 1u, 2u);

  f.request.target.pool = f.pool[DRAFT];
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_ARGUMENT, 0, 0u, NULL, 0u) == 0);
  f.request.target.pool = f.pool[TARGET];
  --f.request.target_logit_count;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_VERIFY, ET_SPEC_ERR_BOUNDS, 0u, NULL,
                      0u) == 0);
  ++f.request.target_logit_count;
  f.request.recompute = NULL;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_ARGUMENT, 0, 0u, NULL, 0u) == 0);
  f.request.recompute = recompute;
  uint32_t empty;
  CHECK(et_paged_kv_seq_alloc(f.pool[TARGET], &empty) == ET_PAGED_KV_OK);
  f.request.target.parent_seq_id = empty;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_PREFIX_LENGTH, 0, 0u, &empty, 1u) ==
        0);
  f.request.target.parent_seq_id = f.parent[TARGET];
  CHECK(et_paged_kv_seq_free(f.pool[TARGET], empty) == ET_PAGED_KV_OK);

  const size_t final_kv = HORIZON * f.width[TARGET] - 1u;
  float saved = f.candidate_v[TARGET][final_kv];
  f.candidate_v[TARGET][final_kv] = NAN;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE,
                      ET_PAGED_KV_ERR_NONFINITE, 0u, NULL, 0u) == 0);
  f.candidate_v[TARGET][final_kv] = saved;
  const size_t final_logit = (HORIZON + 1u) * VOCAB - 1u;
  saved = f.logits[final_logit];
  f.logits[final_logit] = NAN;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_VERIFY, ET_SPEC_ERR_NONFINITE, 0u,
                      NULL, 0u) == 0);
  f.logits[final_logit] = saved;
  f.callback_status = -74;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_RECOMPUTE, -74, 1u, NULL, 0u) == 0);
  f.callback_status = 0;
  f.bad_tail_shape = 1;
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE, ET_PAGED_KV_ERR_SHAPE,
                      1u, NULL, 0u) == 0);
  f.bad_tail_shape = 0;
  CHECK(run_success(&f, 1u, 2u, 1u, 0u) == 0);
  CHECK(cleanup(&f) == 0);
  return 0;
}

static int test_pool_exhaustion(void) {
  fixture f = {0};
  uint32_t extra[2];
  CHECK(init_fixture(&f, 5u) == 0);
  set_case(&f, full_target, ET_SPEC_NO_EOS, 3u, 4u);
  CHECK(et_paged_kv_seq_alloc(f.pool[TARGET], &extra[0]) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(f.pool[TARGET], &extra[1]) == ET_PAGED_KV_OK);
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE,
                      ET_PAGED_KV_ERR_OOM_SEQS, 0u, extra, 2u) == 0);
  CHECK(et_paged_kv_seq_free(f.pool[TARGET], extra[1]) == ET_PAGED_KV_OK);

  float keys[MAX_WIDTH], values[MAX_WIDTH];
  produce_kv(TARGET, 12u, 0u, f.width[TARGET], keys, values);
  CHECK(et_paged_kv_append_token(f.pool[TARGET], extra[0], keys, values) ==
        ET_PAGED_KV_OK);
  /* The target has enough room to stage but needs one more block for the
   * bonus. Draft's corresponding append has already succeeded. */
  CHECK(check_failure(&f, ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE,
                      ET_PAGED_KV_ERR_OOM_BLOCKS, 1u, extra, 1u) == 0);
  /* Confirm that draft actually appended the bonus before target failed:
   * its first-slot K/V survives in an unreferenced external-storage block. */
  uint32_t discarded_bonus_blocks = 0u;
  for (uint32_t b = 0; b < BLOCKS; ++b) {
    const float *block = f.draft_storage + b * DRAFT_BLOCK_FLOATS;
    if (memcmp(block, f.tail_k[DRAFT], DRAFT_WIDTH * sizeof(float)) == 0 &&
        memcmp(block + BLOCK_SIZE * DRAFT_WIDTH, f.tail_v[DRAFT],
               DRAFT_WIDTH * sizeof(float)) == 0) {
      uint32_t refs;
      CHECK(et_paged_kv_block_refcount(f.pool[DRAFT], b, &refs) == ET_PAGED_KV_OK);
      CHECK(refs == 0u);
      ++discarded_bonus_blocks;
    }
  }
  CHECK(discarded_bonus_blocks == 1u);
  uint32_t extra_token = 12u;
  CHECK(check_sequence(f.pool[TARGET], extra[0], TARGET, &extra_token, 1u) == 0);
  CHECK(et_paged_kv_seq_free(f.pool[TARGET], extra[0]) == ET_PAGED_KV_OK);
  CHECK(run_success(&f, 3u, 4u, 0u, 0u) == 0);
  CHECK(cleanup(&f) == 0);

  fixture short_pool = {0};
  CHECK(init_fixture(&short_pool, 3u) == 0);
  set_case(&short_pool, full_target, ET_SPEC_NO_EOS, 3u, 4u);
  CHECK(check_failure(&short_pool, ET_SPEC_PAGED_ROUND_ERR_TARGET_CACHE,
                      ET_PAGED_KV_ERR_OOM_BLOCKS, 0u, NULL, 0u) == 0);
  CHECK(cleanup(&short_pool) == 0);
  return 0;
}

int main(void) {
  CHECK(test_accept_and_reject() == 0);
  CHECK(test_validate_and_retry() == 0);
  CHECK(test_pool_exhaustion() == 0);
  puts("SPEC_PAGED_ROUND accept, reject, EOS, rollback and retry: PASS");
  return 0;
}
