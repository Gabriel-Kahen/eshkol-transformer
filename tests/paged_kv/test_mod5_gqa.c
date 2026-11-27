/* Share the admitted native MOD5 fixture, including its projections and
 * downstream residual/FFN path. Only this witness selects paged attention. */
#define main mod5_training_smoke_main
#include "../mod5/test_mod5_training_smoke.c"
#undef main

#include "eshkol_transformer/paged_kv_abi.h"

enum {
  KV_HEADS = 2,
  KV_WIDTH = KV_HEADS * HEAD_WIDTH,
  GROUP = HEADS / KV_HEADS,
  BLOCK_SIZE = 2,
  SEQ_BLOCKS = (TOKENS + BLOCK_SIZE - 1) / BLOCK_SIZE,
  POOL_BLOCKS = BATCHES * SEQ_BLOCKS,
  POOL_WORDS = POOL_BLOCKS * BLOCK_SIZE * KV_WIDTH * 2
};
static const et_paged_kv_config config = {
    .num_blocks = POOL_BLOCKS, .num_layers = 1, .num_kv_heads = KV_HEADS,
    .block_size = BLOCK_SIZE, .head_dim = HEAD_WIDTH, .max_seqs = BATCHES,
    .max_blocks_per_seq = SEQ_BLOCKS};

typedef struct prefix_state {
  et_paged_kv_stats stats;
  uint32_t lengths[BATCHES], counts[BATCHES], table[BATCHES][SEQ_BLOCKS];
  uint32_t refs[POOL_BLOCKS];
  float storage[POOL_WORDS];
} prefix_state;

static void capture_prefix(const et_paged_kv_pool *pool,
                           const uint32_t sequences[BATCHES],
                           const float storage[POOL_WORDS], prefix_state *out) {
  memset(out, 0, sizeof(*out));
  CHECK(et_paged_kv_stats_get(pool, &out->stats) == ET_PAGED_KV_OK);
  for (size_t b = 0; b < BATCHES; ++b) {
    CHECK(et_paged_kv_seq_len(pool, sequences[b], &out->lengths[b]) ==
          ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_block_table(pool, sequences[b], out->table[b],
                                     SEQ_BLOCKS, &out->counts[b]) ==
          ET_PAGED_KV_OK);
  }
  for (uint32_t block = 0; block < POOL_BLOCKS; ++block)
    CHECK(et_paged_kv_block_refcount(pool, block, &out->refs[block]) ==
          ET_PAGED_KV_OK);
  memcpy(out->storage, storage, sizeof(out->storage));
}

static void check_prefix(const et_paged_kv_pool *pool,
                         const uint32_t sequences[BATCHES],
                         const uint32_t visible[BATCHES], const activations *a,
                         const prefix_state *state) {
  uint32_t refs[POOL_BLOCKS] = {0};
  uint32_t used = 0;
  for (size_t b = 0; b < BATCHES; ++b) {
    CHECK(state->lengths[b] == visible[b]);
    CHECK(state->counts[b] == (visible[b] + BLOCK_SIZE - 1u) / BLOCK_SIZE);
    used += state->counts[b];
    for (size_t i = 0; i < state->counts[b]; ++i) {
      const uint32_t id = state->table[b][i];
      CHECK(id < POOL_BLOCKS && refs[id] == 0);
      refs[id] = 1;
    }
    for (uint32_t t = 0; t < visible[b]; ++t) {
      for (uint32_t h = 0; h < KV_HEADS; ++h) {
        float key[HEAD_WIDTH], value[HEAD_WIDTH];
        const size_t index = (b * TOKENS + t) * KV_WIDTH + h * HEAD_WIDTH;
        CHECK(et_paged_kv_read_token(pool, sequences[b], t, 0, h, key, value) ==
              ET_PAGED_KV_OK);
        CHECK(memcmp(key, a->k + index, sizeof(key)) == 0);
        CHECK(memcmp(value, a->v + index, sizeof(value)) == 0);
      }
    }
  }
  CHECK(memcmp(state->refs, refs, sizeof(refs)) == 0);
  CHECK(state->stats.active_seqs == BATCHES && state->stats.used_blocks == used &&
        state->stats.free_blocks == POOL_BLOCKS - used &&
        state->stats.shared_blocks == 0 && state->stats.cow_copies == 0);
}

static void dense_gqa(const et_kernel_runtime *runtime, size_t kv_heads,
                      activations *a, activations *grad) {
  CHECK(kv_heads == KV_HEADS && grad == NULL);
  attention_policy policy = {.window = TOKENS};
  for (size_t t = 0; t < TOKENS; ++t) {
    policy.key_pos[t] = (int64_t)t;
    for (size_t b = 0; b < BATCHES; ++b)
      policy.query_pos[b][t] = (int64_t)t;
    for (size_t k = 0; k < TOKENS; ++k)
      policy.keep[t * TOKENS + k] = (uint8_t)(k <= t);
  }
  attend_with_policy(runtime, kv_heads, a, NULL, &policy);
}

static void paged_gqa(const et_kernel_runtime *runtime, size_t kv_heads,
                      activations *a, activations *grad) {
  (void)runtime;
  CHECK(kv_heads == KV_HEADS && grad == NULL);
  et_paged_kv_pool *pool = NULL;
  float storage[POOL_WORDS] = {0};
  size_t floats = 0;
  CHECK(et_paged_kv_pool_float_count(&config, &floats) == ET_PAGED_KV_OK &&
        floats == ARRAY_COUNT(storage));
  CHECK(et_paged_kv_create(&config, storage, floats, &pool) == ET_PAGED_KV_OK);

  uint32_t sequences[BATCHES], visible[BATCHES] = {0};
  for (size_t b = 0; b < BATCHES; ++b)
    CHECK(et_paged_kv_seq_alloc(pool, &sequences[b]) == ET_PAGED_KV_OK);
  CHECK(sequences[0] != sequences[1]);
  const float scale = 1.0f / sqrtf((float)HEAD_WIDTH);
  prefix_state before, after;
  for (uint32_t t = 0; t < TOKENS; ++t) {
    for (size_t b = 0; b < BATCHES; ++b) {
      const size_t row = b * TOKENS + t;
      const float *query = a->q + row * WIDTH;
      const float *keys = a->k + row * KV_WIDTH;
      const float *values = a->v + row * KV_WIDTH;
      float *out = a->attention + row * WIDTH;
      CHECK(et_paged_kv_append_token(pool, sequences[b], keys, values) ==
            ET_PAGED_KV_OK);
      ++visible[b];
      capture_prefix(pool, sequences, storage, &before);
      check_prefix(pool, sequences, visible, a, &before);
      CHECK(et_paged_kv_decode_attention_v1(pool, sequences[b], 0, HEADS,
                                            scale, query, out) ==
            ET_PAGED_KV_OK);
      capture_prefix(pool, sequences, storage, &after);
      CHECK(memcmp(&before, &after, sizeof(before)) == 0);
      if (t == 0) {
        /* One visible token isolates the Q→KV map even with different queries. */
        CHECK(memcmp(values, values + HEAD_WIDTH,
                     HEAD_WIDTH * sizeof(*values)) != 0);
        for (size_t h = 0; h < HEADS; ++h)
          CHECK(memcmp(out + h * HEAD_WIDTH,
                       values + (h / GROUP) * HEAD_WIDTH,
                       HEAD_WIDTH * sizeof(*out)) == 0);
      }
    }
  }

  const float sentinel[WIDTH] = {
      -9.0f, 8.0f, -7.0f, 6.0f, -5.0f, 4.0f, -3.0f, 2.0f};
  float rejected[WIDTH];
  memcpy(rejected, sentinel, sizeof(sentinel));
  CHECK(et_paged_kv_decode_attention_v1(pool, sequences[0], 0, HEADS - 1u,
                                        scale, a->q, rejected) ==
        ET_PAGED_KV_ERR_SHAPE);
  CHECK(memcmp(rejected, sentinel, sizeof(rejected)) == 0);
  capture_prefix(pool, sequences, storage, &after);
  CHECK(memcmp(&before, &after, sizeof(before)) == 0);

  for (size_t b = 0; b < BATCHES; ++b)
    CHECK(et_paged_kv_seq_free(pool, sequences[b]) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &after.stats) == ET_PAGED_KV_OK);
  CHECK(after.stats.active_seqs == 0 && after.stats.used_blocks == 0 &&
        after.stats.free_blocks == POOL_BLOCKS && after.stats.shared_blocks == 0);
  for (uint32_t block = 0; block < POOL_BLOCKS; ++block) {
    uint32_t refs;
    CHECK(et_paged_kv_block_refcount(pool, block, &refs) == ET_PAGED_KV_OK && refs == 0);
  }
  et_paged_kv_destroy(pool);
}

static float compare(const float *actual, const float *expected, float bound) {
  float maximum = 0;
  for (size_t i = 0; i < ELEMENTS; ++i) {
    CHECK(isfinite(actual[i]) && isfinite(expected[i]));
    const float error = fabsf(actual[i] - expected[i]);
    CHECK(error <= bound);
    maximum = fmaxf(maximum, error);
  }
  return maximum;
}

static int differs(const float *a, const float *b, size_t words) {
  for (size_t i = 0; i < words; ++i)
    if (fabsf(a[i] - b[i]) > 1e-6f)
      return 1;
  return 0;
}

static void write_trace(const char *path, const activations *base,
                        const activations *changed) {
  FILE *trace = fopen(path, "wb");
  CHECK(trace != NULL);
  const activations *cases[] = {base, changed};
  for (size_t i = 0; i < ARRAY_COUNT(cases); ++i) {
    const activations *a = cases[i];
    CHECK(fwrite(a->q, sizeof(float), ELEMENTS, trace) == ELEMENTS);
    CHECK(fwrite(a->k, sizeof(float), ROWS * KV_WIDTH, trace) == ROWS * KV_WIDTH);
    CHECK(fwrite(a->v, sizeof(float), ROWS * KV_WIDTH, trace) == ROWS * KV_WIDTH);
    CHECK(fwrite(a->attention, sizeof(float), ELEMENTS, trace) == ELEMENTS);
    CHECK(fwrite(a->output, sizeof(float), ELEMENTS, trace) == ELEMENTS);
  }
  CHECK(fclose(trace) == 0);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  et_kernel_runtime *runtime = discover_mod5();
  weights w;
  float input[ELEMENTS];
  initialize_fixture(KV_HEADS, &w, input);
  const weights initial = w;
  activations dense = {0}, paged = {0}, changed_dense = {0}, changed_paged = {0};
  block_forward_with_attention(runtime, KV_HEADS, &w, input, &dense, dense_gqa);
  block_forward_with_attention(runtime, KV_HEADS, &w, input, &paged, paged_gqa);

  /* At least one multi-token row distinguishes both queries in each KV group. */
  for (size_t h = 0; h < KV_HEADS; ++h) {
    int distinct = 0;
    for (size_t b = 0; b < BATCHES; ++b) {
      for (size_t t = 1; t < TOKENS; ++t) {
        const float *head = paged.attention + (b * TOKENS + t) * WIDTH +
                            h * GROUP * HEAD_WIDTH;
        distinct |= differs(head, head + HEAD_WIDTH, HEAD_WIDTH);
      }
    }
    CHECK(distinct);
  }
  const size_t suffix = (TOKENS - 1u) * WIDTH;
  input[suffix] += 0.75f;
  input[suffix + 3u] -= 0.5f;
  block_forward_with_attention(runtime, KV_HEADS, &w, input, &changed_dense, dense_gqa);
  block_forward_with_attention(runtime, KV_HEADS, &w, input, &changed_paged, paged_gqa);
  CHECK(memcmp(&initial, &w, sizeof(w)) == 0);

  float attention_error = compare(paged.attention, dense.attention, 1e-6f);
  attention_error = fmaxf(attention_error,
      compare(changed_paged.attention, changed_dense.attention, 1e-6f));
  float block_error = compare(paged.output, dense.output, 2e-6f);
  block_error = fmaxf(block_error, compare(changed_paged.output, changed_dense.output, 2e-6f));

  CHECK(memcmp(paged.attention, changed_paged.attention, suffix * sizeof(float)) == 0);
  CHECK(memcmp(paged.output, changed_paged.output, suffix * sizeof(float)) == 0);
  CHECK(differs(paged.attention + suffix, changed_paged.attention + suffix, WIDTH));
  CHECK(differs(paged.output + suffix, changed_paged.output + suffix, WIDTH));
  const size_t batch = TOKENS * WIDTH;
  CHECK(memcmp(paged.attention + batch, changed_paged.attention + batch,
               batch * sizeof(float)) == 0);
  CHECK(memcmp(paged.output + batch, changed_paged.output + batch,
               batch * sizeof(float)) == 0);

  write_trace(argv[1], &paged, &changed_paged);
  et_kernel_runtime_destroy(runtime);
  printf("MOD5/PKV5 live GQA PASS: %u query heads, %u KV heads, %u decode rows, "
         "%u attention and block words each, abs errors %.9g %.9g\n",
         (unsigned)HEADS, (unsigned)KV_HEADS, 2u * BATCHES * TOKENS,
         2u * ELEMENTS, (double)attention_error, (double)block_error);
  return EXIT_SUCCESS;
}
