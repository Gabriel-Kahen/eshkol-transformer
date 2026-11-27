#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROVIDER_ROUTES_PRIVATE 1
#define ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE 1
#define ET_G3C4_SAMPLER_TRANSPORT_PRIVATE 1
#define ET_G3C4_TOKEN_FRAME_PRIVATE 1
#define ET_G3C4_TOKEN_FORWARD_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include "eshkol_transformer/spec_paged_round_abi.h"

#include <math.h>
#include <stdio.h>

#define CHECK(condition) do {                                                  \
  if (!(condition)) {                                                          \
    fprintf(stderr, "FAIL %s:%d: %s [G3 domain=%lld category=%lld code=%lld]\n",  \
            __FILE__, __LINE__, #condition,                                     \
            (long long)et_g3c4_private_last_error_domain_v1(),                    \
            (long long)et_g3c4_private_last_error_category_v1(),                  \
            (long long)et_g3c4_private_last_error_code_v1());                     \
    exit(1);                                                                   \
  }                                                                            \
} while (0)
#define OK(call) CHECK((call) == 0)

enum { VOCAB = 256, CAPACITY = 4, HEADS = 2, WIDTH = 2, BLOCKS = 4,
       BLOCK_SIZE = 2, MAX_HORIZON = 2, TOKEN_FLOATS = HEADS * WIDTH, INJECTED_ERROR = -17 };
enum { DRAFT, TARGET, MODELS };
typedef float logit_row[VOCAB];

typedef struct {
  float key[TOKEN_FLOATS];
  float value[TOKEN_FLOATS];
} token_kv;

typedef struct {
  uint32_t length;
  token_kv tokens[CAPACITY];
} cache_prefix;

typedef struct {
  et_g3c4_model_owner_internal *owner;
  et_g3c4_context_internal *control;
  et_paged_kv_pool *pool;
  uint32_t anchor;
  uint32_t branch;
} model;

typedef struct {
  float key[MAX_HORIZON * TOKEN_FLOATS];
  float value[MAX_HORIZON * TOKEN_FLOATS];
} candidate_kv;

typedef struct {
  et_g3c4_model_owner_internal *owners[MODELS];
  const float *target_logits;
  uint32_t horizon;
  int32_t injected_status;
  uint32_t calls;
  cache_prefix prefixes[MODELS];
  token_kv kv[MODELS];
  logit_row logits[MODELS];
} round_scratch;

static et_g3c4_model_owner_internal *new_owner(int64_t seed, char ids[14]) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner != NULL);
  for (int64_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_bind_v1(owner, i, &ids[i]));
  for (int64_t i = 0; i < 14; ++i)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, i));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static et_g3c4_context_internal *new_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), VOCAB, INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  return context;
}

static uint32_t greedy(const logit_row logits) {
  uint32_t token = 0;
  for (uint32_t i = 0; i < VOCAB; ++i) {
    CHECK(isfinite(logits[i]));
    if (logits[i] > logits[token]) token = i;
  }
  return token;
}

static void forward(et_g3c4_context_internal *context, uint32_t token,
                    const logit_row selection, logit_row output) {
  float carrier[CAPACITY * VOCAB] = {0};
  float *last = carrier + (CAPACITY - 1) * VOCAB;
  int64_t initial_rng[4], selected = -1, published = -1;
  CHECK(token < VOCAB);
  /* The private frame selects from its last row. Only the initial prompt
     needs a token carrier, since it has no preceding model logits. */
  if (selection == NULL) last[token] = 1.0f;
  else memcpy(last, selection, sizeof(logit_row));
  memcpy(initial_rng, context->generator_rng_words, sizeof(initial_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(context, carrier, &selected));
  CHECK(selected == token);
  OK(et_g3c4_private_token_forward_v1(context, selected, output));
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == selected);
  OK(et_g3c4_private_call_finish_v1(context));
  CHECK(memcmp(initial_rng, context->generator_rng_words,
               sizeof(initial_rng)) == 0);
}

static cache_prefix read_a2(et_g3c4_context_internal *context) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *key = NULL, *value = NULL;
  const et_kernel_tensor_view_v1 *length = NULL, *keep = NULL;
  et_kernel_error error;
  cache_prefix prefix = {0};
  OK(et_a2_kv_cache_read_borrow_begin_v1(context->cache, &borrow, &error));
  OK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0, &key, &value, &length, &keep, &error));
  const int64_t n = *(const int64_t *)length->data;
  CHECK(n >= 0 && n <= CAPACITY);
  prefix.length = (uint32_t)n;
  for (uint32_t pos = 0; pos < CAPACITY; ++pos) {
    CHECK(((const uint8_t *)keep->data)[pos] == (pos < prefix.length));
    if (pos >= prefix.length) continue;
    for (uint32_t h = 0; h < HEADS; ++h) {
      const size_t offset = (h * CAPACITY + pos) * WIDTH;
      memcpy(&prefix.tokens[pos].key[h * WIDTH],
             (const float *)key->data + offset, WIDTH * sizeof(float));
      memcpy(&prefix.tokens[pos].value[h * WIDTH],
             (const float *)value->data + offset, WIDTH * sizeof(float));
    }
  }
  OK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error));
  return prefix;
}

static cache_prefix read_paged(const et_paged_kv_pool *pool, uint32_t seq) {
  cache_prefix prefix = {0};
  OK(et_paged_kv_seq_len(pool, seq, &prefix.length));
  CHECK(prefix.length <= CAPACITY);
  for (uint32_t pos = 0; pos < prefix.length; ++pos)
    for (uint32_t h = 0; h < HEADS; ++h)
      OK(et_paged_kv_read_token(pool, seq, pos, 0, h,
          &prefix.tokens[pos].key[h * WIDTH], &prefix.tokens[pos].value[h * WIDTH]));
  return prefix;
}

/* PKV reads [position,head,feature]. A2 staging takes [1,head,length,feature]. */
static void restore_a2(et_g3c4_context_internal *context,
                       const cache_prefix *prefix) {
  int64_t n = prefix->length;
  const uint64_t count_shape[] = {1};
  const uint64_t kv_shape[] = {1, HEADS, (uint64_t)n, WIDTH};
  float key[HEADS * CAPACITY * WIDTH], value[HEADS * CAPACITY * WIDTH];
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_error error;
  CHECK(n > 0 && n < CAPACITY);
  for (size_t h = 0; h < HEADS; ++h)
    for (size_t pos = 0; pos < (size_t)n; ++pos) {
      const size_t offset = (h * (size_t)n + pos) * WIDTH;
      memcpy(key + offset, &prefix->tokens[pos].key[h * WIDTH],
             WIDTH * sizeof(float));
      memcpy(value + offset, &prefix->tokens[pos].value[h * WIDTH],
             WIDTH * sizeof(float));
    }
  const size_t bytes = HEADS * (size_t)n * WIDTH * sizeof(float);
  et_kernel_tensor_view_v1 counts = et_g3c4_view(&n, sizeof(n), "i64", 1, count_shape);
  et_kernel_tensor_view_v1 keys = et_g3c4_view(key, bytes, "f32", 4, kv_shape);
  et_kernel_tensor_view_v1 values = et_g3c4_view(value, bytes, "f32", 4, kv_shape);
  OK(et_a2_kv_cache_transaction_begin_v1(
      context->cache, (uint64_t)n, &counts, &transaction, &error));
  OK(et_a2_kv_cache_transaction_stage_layer_v1(
      transaction, 0, &keys, &values, &error));
  OK(et_a2_kv_cache_transaction_commit_v1(&transaction, &error));
}

static void check_prefix(const cache_prefix *actual, const cache_prefix *expected) {
  CHECK(actual->length == expected->length);
  for (uint32_t pos = 0; pos < actual->length; ++pos) {
    CHECK(memcmp(actual->tokens[pos].key, expected->tokens[pos].key,
                 sizeof(actual->tokens[pos].key)) == 0);
    CHECK(memcmp(actual->tokens[pos].value, expected->tokens[pos].value,
                 sizeof(actual->tokens[pos].value)) == 0);
  }
}

static token_kv forward_from_prefix(et_g3c4_model_owner_internal *owner,
                                    const cache_prefix *prefix, uint32_t token,
                                    const logit_row selection, logit_row output) {
  et_g3c4_context_internal *context = new_generator(owner);
  restore_a2(context, prefix);
  forward(context, token, selection, output);
  cache_prefix committed = read_a2(context);
  CHECK(committed.length == prefix->length + 1);
  committed.length = prefix->length;
  check_prefix(&committed, prefix);
  OK(et_g3c4_private_generator_close_v1(context));
  return committed.tokens[prefix->length];
}

static int32_t recompute(void *opaque, uint32_t token, uint32_t position,
                         et_spec_paged_round_prefix_v1 draft,
                         et_spec_paged_round_prefix_v1 target,
                         et_spec_paged_kv_span_v1 *out_draft,
                         et_spec_paged_kv_span_v1 *out_target) {
  round_scratch *scratch = opaque;
  const et_spec_paged_round_prefix_v1 views[MODELS] = {draft, target};
  et_spec_paged_kv_span_v1 *const out[MODELS] = {out_draft, out_target};
  CHECK(position > 0 && position <= scratch->horizon + 1 && position < CAPACITY);
  const float *target_row = scratch->target_logits + (position - 1) * VOCAB;
  ++scratch->calls;
  for (size_t i = 0; i < MODELS; ++i) {
    const et_spec_paged_round_prefix_v1 view = views[i];
    scratch->prefixes[i] = read_paged(view.pool, view.seq_id);
    CHECK(scratch->prefixes[i].length == position);

    /* At position two the accepted child fits one block. A rejected
       position-two candidate would have occupied a second child block. */
    et_paged_kv_stats stats;
    OK(et_paged_kv_stats_get(view.pool, &stats));
    CHECK(stats.active_seqs == 2 && stats.shared_blocks == 0 &&
          stats.used_blocks == 1 + (position + BLOCK_SIZE - 1) / BLOCK_SIZE);

    token_kv *kv = &scratch->kv[i];
    *kv = forward_from_prefix(scratch->owners[i], &scratch->prefixes[i],
                              token, target_row, scratch->logits[i]);
    *out[i] = (et_spec_paged_kv_span_v1){kv->key, kv->value, TOKEN_FLOATS};
  }
  return scratch->injected_status;
}

static void continue_paged(const model *m, uint32_t token,
                           const logit_row selection, logit_row output) {
  const cache_prefix prefix = read_paged(m->pool, m->branch);
  const token_kv kv = forward_from_prefix(m->owner, &prefix, token, selection, output);
  OK(et_paged_kv_append_token(m->pool, m->branch, kv.key, kv.value));
}

static void start_model(model *m, et_g3c4_model_owner_internal *owner,
                        uint32_t prompt, logit_row logits) {
  const et_paged_kv_config cfg = {
    .num_blocks = BLOCKS, .block_size = BLOCK_SIZE, .num_layers = 1,
    .num_kv_heads = HEADS, .head_dim = WIDTH, .max_seqs = 2,
    .max_blocks_per_seq = 2,
  };
  m->owner = owner;
  m->control = new_generator(owner);
  forward(m->control, prompt, NULL, logits);
  const cache_prefix prefix = read_a2(m->control);
  CHECK(prefix.length == 1);
  OK(et_paged_kv_create(&cfg, NULL, 0, &m->pool));
  OK(et_paged_kv_seq_alloc(m->pool, &m->anchor));
  OK(et_paged_kv_append_token(m->pool, m->anchor,
      prefix.tokens[0].key, prefix.tokens[0].value));
}

static uint32_t anchor_block(const model *m) {
  uint32_t block, count;
  OK(et_paged_kv_seq_block_table(m->pool, m->anchor, &block, 1, &count));
  CHECK(count == 1);
  return block;
}

static void check_failed_round(const model *m, uint32_t original_block) {
  const cache_prefix actual = read_paged(m->pool, m->anchor);
  const cache_prefix expected = read_a2(m->control);
  et_paged_kv_stats stats;
  check_prefix(&actual, &expected);
  CHECK(anchor_block(m) == original_block);
  OK(et_paged_kv_stats_get(m->pool, &stats));
  CHECK(stats.active_seqs == 1 && stats.used_blocks == 1 &&
        stats.free_blocks == BLOCKS - 1 && stats.shared_blocks == 0 &&
        stats.cow_copies == 1);
  for (uint32_t block = 0; block < BLOCKS; ++block) {
    uint32_t refs;
    OK(et_paged_kv_block_refcount(m->pool, block, &refs));
    CHECK(refs == (uint32_t)(block == original_block));
  }
}

static void finish_model(const model *m, uint64_t expected_copies) {
  et_paged_kv_stats stats;
  const cache_prefix anchor = read_paged(m->pool, m->anchor);
  cache_prefix expected = read_a2(m->control);
  CHECK(expected.length == CAPACITY);
  expected.length = 1;
  check_prefix(&anchor, &expected);
  OK(et_paged_kv_stats_get(m->pool, &stats));
  CHECK(stats.active_seqs == 2 && stats.used_blocks == 3 &&
        stats.shared_blocks == 0 && stats.cow_copies == expected_copies);
  OK(et_paged_kv_seq_free(m->pool, m->branch));
  OK(et_paged_kv_seq_free(m->pool, m->anchor));
  OK(et_paged_kv_stats_get(m->pool, &stats));
  CHECK(stats.active_seqs == 0 && stats.used_blocks == 0 &&
        stats.free_blocks == BLOCKS && stats.shared_blocks == 0);
  et_paged_kv_destroy(m->pool);
  OK(et_g3c4_private_generator_close_v1(m->control));
}

static void check_control(const model *m, const logit_row actual,
                          const logit_row expected) {
  CHECK(memcmp(actual, expected, sizeof(logit_row)) == 0);
  const cache_prefix paged = read_paged(m->pool, m->branch);
  const cache_prefix uninterrupted = read_a2(m->control);
  check_prefix(&paged, &uninterrupted);
}

static void run_case(const char *label, et_g3c4_model_owner_internal *target_owner,
                     et_g3c4_model_owner_internal *draft_owner,
                     uint32_t prompt, uint32_t horizon, uint32_t accepted) {
  model models[MODELS];
  float target_logits[(MAX_HORIZON + 1) * VOCAB];
  logit_row draft_rows[MAX_HORIZON + 1], control_rows[CAPACITY], draft_control;
  uint32_t proposals[MAX_HORIZON];
  int64_t committed[CAPACITY] = {prompt};
  candidate_kv candidates[MODELS];
  et_g3c4_context_internal *producers[MODELS];

  CHECK(horizon > 0 && horizon <= MAX_HORIZON && accepted > 0 && accepted <= horizon);
  start_model(&models[DRAFT], draft_owner, prompt, draft_rows[0]);
  start_model(&models[TARGET], target_owner, prompt, target_logits);
  memcpy(control_rows[0], target_logits, sizeof(logit_row));

  for (size_t i = 0; i < MODELS; ++i) {
    producers[i] = new_generator(models[i].owner);
    const cache_prefix prefix = read_paged(models[i].pool, models[i].anchor);
    restore_a2(producers[i], &prefix);
  }
  for (uint32_t i = 0; i < horizon; ++i) {
    proposals[i] = greedy(draft_rows[i]);
    forward(producers[DRAFT], proposals[i], draft_rows[i], draft_rows[i + 1]);
    forward(producers[TARGET], proposals[i], draft_rows[i], target_logits + (i + 1) * VOCAB);
  }
  for (size_t i = 0; i < MODELS; ++i) {
    const cache_prefix prefix = read_a2(producers[i]);
    CHECK(prefix.length == horizon + 1);
    for (uint32_t candidate = 0; candidate < horizon; ++candidate) {
      const token_kv *kv = &prefix.tokens[candidate + 1];
      memcpy(candidates[i].key + candidate * TOKEN_FLOATS, kv->key, sizeof(kv->key));
      memcpy(candidates[i].value + candidate * TOKEN_FLOATS, kv->value, sizeof(kv->value));
    }
    OK(et_g3c4_private_generator_close_v1(producers[i]));
  }

  round_scratch scratch = {
    .owners = {draft_owner, target_owner},
    .target_logits = target_logits,
    .horizon = horizon,
  };
  const et_spec_paged_round_request_v1 request = {
    .draft = {models[DRAFT].pool, models[DRAFT].anchor,
              {candidates[DRAFT].key, candidates[DRAFT].value, horizon * TOKEN_FLOATS}},
    .target = {models[TARGET].pool, models[TARGET].anchor,
               {candidates[TARGET].key, candidates[TARGET].value, horizon * TOKEN_FLOATS}},
    .draft_tokens = proposals,
    .target_logits = target_logits,
    .target_logit_count = (horizon + 1) * VOCAB,
    .horizon = horizon,
    .vocab_size = VOCAB,
    .eos_token_id = UINT32_MAX,
    .recompute = recompute,
    .context = &scratch,
  };
  et_spec_paged_round_result_v1 result;
  const uint32_t rejected = accepted < horizon;
  if (rejected) {
    unsigned char sentinel[sizeof(result)];
    memset(sentinel, 0xA5, sizeof(sentinel));
    memcpy(&result, sentinel, sizeof(result));
    const uint32_t draft_block = anchor_block(&models[DRAFT]);
    const uint32_t target_block = anchor_block(&models[TARGET]);
    scratch.injected_status = INJECTED_ERROR;
    const et_spec_paged_round_status_v1 status =
        et_spec_paged_round_greedy_v1(&request, &result);
    CHECK(status.kind == ET_SPEC_PAGED_ROUND_ERR_RECOMPUTE && status.code == INJECTED_ERROR);
    CHECK(scratch.calls == 1 && memcmp(&result, sentinel, sizeof(result)) == 0);
    check_failed_round(&models[DRAFT], draft_block);
    check_failed_round(&models[TARGET], target_block);
    scratch.injected_status = 0;
  }
  const et_spec_paged_round_status_v1 status =
      et_spec_paged_round_greedy_v1(&request, &result);
  CHECK(status.kind == ET_SPEC_PAGED_ROUND_OK && status.code == 0);
  CHECK(scratch.calls == 1u + rejected);
  const et_spec_verify_result_v1 verify = result.verification;
  CHECK(verify.accepted_count == accepted && verify.emitted_count == accepted + 1);
  CHECK(verify.rejected == rejected && verify.eos_hit == 0);
  CHECK(verify.rng_draws == 0 && verify.next_rng.seed == 0 && verify.next_rng.counter == 0);
  model *const draft = &models[DRAFT], *const target = &models[TARGET];
  draft->branch = result.draft_seq_id;
  target->branch = result.target_seq_id;

  /* Drive the controls from the target alone and compare them at the
     accepted prefix, after the round, and after any continuation. */
  for (uint32_t i = 0; i < CAPACITY - 1; ++i) {
    const uint32_t control_token = greedy(control_rows[i]);
    if (i < verify.emitted_count) CHECK(result.tokens[i] == control_token);
    if (i < accepted) {
      CHECK(result.tokens[i] == proposals[i]);
    } else if (i == accepted) {
      if (rejected) CHECK(result.tokens[i] != proposals[i]);
      for (size_t m = 0; m < MODELS; ++m) {
        const cache_prefix uninterrupted = read_a2(models[m].control);
        check_prefix(&scratch.prefixes[m], &uninterrupted);
      }
      CHECK(memcmp(target_logits + i * VOCAB, control_rows[i], sizeof(logit_row)) == 0);
      CHECK(memcmp(draft_rows[i], draft_control, sizeof(logit_row)) == 0);
    } else {
      const uint32_t token = greedy(scratch.logits[TARGET]);
      CHECK(token == control_token);
      logit_row selection;
      memcpy(selection, scratch.logits[TARGET], sizeof(selection));
      continue_paged(target, token, selection, scratch.logits[TARGET]);
      continue_paged(draft, token, selection, scratch.logits[DRAFT]);
    }
    committed[i + 1] = control_token;
    forward(target->control, control_token, control_rows[i], control_rows[i + 1]);
    forward(draft->control, control_token, control_rows[i], draft_control);
    if (i >= accepted) {
      check_control(target, scratch.logits[TARGET], control_rows[i + 1]);
      check_control(draft, scratch.logits[DRAFT], draft_control);
    }
  }

  float batch[CAPACITY * VOCAB];
  et_g3c4_context_internal *reference = new_generator(target_owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 0, 0));
  OK(et_g3c4_private_full_prefix_forward_v1(reference, committed, batch));
  OK(et_g3c4_private_call_finish_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
  CHECK(memcmp(batch, &control_rows, sizeof(batch)) == 0);
  finish_model(draft, scratch.calls);
  finish_model(target, scratch.calls);
  printf("SPEC5 live %s: proposed=%u accepted=%u emitted=%u continuation=%u\n",
         label, horizon, accepted, verify.emitted_count,
         CAPACITY - 1 - verify.emitted_count);
  if (rejected) puts("SPEC5 live callback rollback and retry: PASS");
}

int main(void) {
  static char target_ids[14], draft_ids[14], matching_ids[14];
  et_g3c4_model_owner_internal *target = new_owner(1729, target_ids);
  et_g3c4_model_owner_internal *draft = new_owner(38, draft_ids);
  et_g3c4_model_owner_internal *matching = new_owner(1729, matching_ids);
  run_case("late rejection", target, draft, 71, 2, 1);
  run_case("bonus and continuation", target, draft, 71, 1, 1);
  run_case("full acceptance", target, matching, 71, 2, 2);
  return 0;
}
