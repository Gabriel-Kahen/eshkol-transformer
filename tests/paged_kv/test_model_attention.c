#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROVIDER_ROUTES_PRIVATE 1
#define ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include "eshkol_transformer/data4_pipeline_abi.h"
#include "eshkol_transformer/paged_kv_abi.h"

#include <math.h>
#include <stdio.h>

enum { TOKENS = 4, HEADS = 2, WIDTH = 2, VOCAB = 256, BLOCK_SIZE = 2 };
enum {
  ATTENTION_WORDS = HEADS * TOKENS * WIDTH, LOGIT_WORDS = TOKENS * VOCAB,
  SEQ_BLOCKS = TOKENS / BLOCK_SIZE, POOL_BLOCKS = 3,
  POOL_WORDS = POOL_BLOCKS * BLOCK_SIZE * HEADS * WIDTH * 2,
  MODEL_BYTES = 4768
};

enum route { DENSE, FRESH, DONOR, REUSE };

static const et_paged_kv_config reuse_config = {
    .num_blocks = POOL_BLOCKS, .num_layers = 1, .num_kv_heads = HEADS,
    .block_size = BLOCK_SIZE, .head_dim = WIDTH, .max_seqs = 2,
    .max_blocks_per_seq = SEQ_BLOCKS};
static const uint32_t norm_epsilon_bits = UINT32_C(0x3727c5ac);

#define CHECK(condition) do {                                           \
  if (!(condition)) {                                                    \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);  \
    exit(1);                                                             \
  }                                                                      \
} while (0)

static struct {
  enum route route;
  et_paged_kv_pool *pool;
  uint32_t donor, recipient;
  float storage[POOL_WORDS];
  float donor_q[ATTENTION_WORDS], donor_attention[ATTENTION_WORDS];
  float fresh_attention[ATTENTION_WORDS];
  size_t dense_calls, paged_calls, attention_words, logit_words;
  size_t reuse_calls, reuse_attention_words, reuse_logit_words, reuse_kv_words;
  float dense[ATTENTION_WORDS], dense_k[ATTENTION_WORDS], dense_v[ATTENTION_WORDS];
  float attention_error, logit_error;
} witness;

static void compare(const float *actual, const float *expected, size_t count,
                    float tolerance, float *maximum_error) {
  for (size_t i = 0; i < count; i++) {
    const float error = fabsf(actual[i] - expected[i]);
    CHECK(isfinite(error) && error <= tolerance);
    *maximum_error = fmaxf(*maximum_error, error);
  }
}

static void check_view(const et_kernel_tensor_view_v1 *view,
                       const char *dtype, size_t rank, const uint64_t *shape,
                       size_t bytes) {
  CHECK(view->struct_size == sizeof(*view) && view->data != NULL);
  CHECK(view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR &&
        view->offset_bytes == 0 && view->byte_length == bytes);
  CHECK(strcmp(view->dtype, dtype) == 0 && strcmp(view->device, "cpu") == 0);
  CHECK(view->rank == rank);
  if (rank != 0)
    CHECK(view->shape != NULL &&
          memcmp(view->shape, shape, rank * sizeof(shape[0])) == 0);
}

static void hash(const void *bytes, size_t size, uint8_t *digest) {
  CHECK(et_data4_sha256(bytes, size, digest) == ET_DATA4_OK);
}

static void write_u32_le(uint8_t *out, uint32_t value) {
  for (size_t i = 0; i < sizeof(value); i++)
    out[i] = (uint8_t)(value >> (8u * i));
}

static void model_image(void *context, uint8_t image[MODEL_BYTES]) {
  const et_g3c4_context_internal *active = et_g3c4_admit_active_call(context);
  CHECK(active != NULL);
  size_t offset = 0;
  const size_t count = sizeof(active->pins.views) / sizeof(active->pins.views[0]);
  for (size_t i = 0; i < count; i++) {
    const et_kernel_tensor_view_v1 *view = &active->pins.views[i];
    const size_t words = et_g3c4_parameter_elements(i);
    check_view(view, "f32", et_g3c4_parameter_rank(i),
               et_g3c4_parameter_shapes[i], words * sizeof(uint32_t));
    CHECK(view->byte_length <= MODEL_BYTES - offset);
    for (size_t j = 0; j < words; j++) {
      uint32_t bits;
      memcpy(&bits, (const uint8_t *)view->data + j * sizeof(bits), sizeof(bits));
      write_u32_le(image + offset, bits);
      offset += sizeof(bits);
    }
  }
  CHECK(offset == MODEL_BYTES);
}

static void producer_digest(uint32_t position_base, uint8_t *digest) {
  // Fixed G3-C4 CPU f32 full-prefix producer, inference, absolute positions.
  static const uint8_t profile[] = "g3c4-fixed-cpu-f32-absolute-causal-v1";
  const uint32_t settings[] = {
      1, 1, TOKENS, HEADS, VOCAB, reuse_config.num_layers,
      reuse_config.num_kv_heads, reuse_config.block_size, reuse_config.head_dim,
      position_base, norm_epsilon_bits
  };
  uint8_t data[sizeof(profile) + sizeof(settings)];
  memcpy(data, profile, sizeof(profile));
  for (size_t i = 0; i < sizeof(settings) / sizeof(settings[0]); i++)
    write_u32_le(data + sizeof(profile) + i * sizeof(settings[0]), settings[i]);
  hash(data, sizeof(data), digest);
}

static et_paged_kv_prefix_key_v1 prefix_key(
    const et_paged_kv_prefix_context_v1 *context, const int64_t tokens[TOKENS]) {
  uint32_t ids[BLOCK_SIZE];
  for (size_t i = 0; i < BLOCK_SIZE; i++) {
    CHECK(tokens[i] >= 0 && tokens[i] < VOCAB);
    ids[i] = (uint32_t)tokens[i];
  }
  et_paged_kv_prefix_key_v1 key;
  CHECK(et_paged_kv_hash_token_block_with_context_v1(context, NULL, ids,
                                                   BLOCK_SIZE, &key) ==
        ET_PAGED_KV_OK);
  return key;
}

typedef struct {
  et_paged_kv_stats stats;
  uint32_t lengths[2], blocks[2][SEQ_BLOCKS], block_counts[2], refs[POOL_BLOCKS];
  float storage[POOL_WORDS];
} pool_snapshot;

static void capture_pool(pool_snapshot *snapshot) {
  const uint32_t sequences[] = {witness.donor, witness.recipient};
  memset(snapshot, 0, sizeof(*snapshot));
  CHECK(et_paged_kv_stats_get(witness.pool, &snapshot->stats) == ET_PAGED_KV_OK);
  for (size_t i = 0; i < 2; i++) {
    CHECK(et_paged_kv_seq_len(witness.pool, sequences[i], &snapshot->lengths[i]) ==
          ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_block_table(witness.pool, sequences[i],
                                     snapshot->blocks[i], SEQ_BLOCKS,
                                     &snapshot->block_counts[i]) ==
          ET_PAGED_KV_OK);
    CHECK(snapshot->block_counts[i] <= SEQ_BLOCKS);
    for (size_t j = 0; j < snapshot->block_counts[i]; j++)
      CHECK(snapshot->blocks[i][j] < POOL_BLOCKS);
  }
  for (uint32_t block = 0; block < POOL_BLOCKS; block++)
    CHECK(et_paged_kv_block_refcount(witness.pool, block, &snapshot->refs[block]) ==
          ET_PAGED_KV_OK);
  memcpy(snapshot->storage, witness.storage, sizeof(witness.storage));
}

static void check_rejected_context(const et_paged_kv_prefix_key_v1 *key,
                                   const pool_snapshot *before) {
  const et_paged_kv_prefix_key_v1 saved = *key;
  pool_snapshot after;
  uint32_t hit = 73;
  CHECK(et_paged_kv_prefix_cache_attach_with_context_v1(
            witness.pool, witness.recipient, key, &hit) == ET_PAGED_KV_ERR_CONTEXT);
  CHECK(hit == 73 && memcmp(key, &saved, sizeof(saved)) == 0);
  capture_pool(&after);
  CHECK(memcmp(before, &after, sizeof(after)) == 0);

  // Attempt registration on the donor's full but unindexed suffix block.
  CHECK(et_paged_kv_prefix_cache_register_with_context_v1(
            witness.pool, witness.donor, 1, key) == ET_PAGED_KV_ERR_CONTEXT);
  CHECK(memcmp(key, &saved, sizeof(saved)) == 0);
  capture_pool(&after);
  CHECK(memcmp(before, &after, sizeof(after)) == 0);
}

static void attach_prefix(const et_paged_kv_prefix_context_v1 *context,
                          const int64_t tokens[TOKENS],
                          const et_paged_kv_prefix_context_v1 foreign[3]) {
  pool_snapshot before, after;
  capture_pool(&before);
  CHECK(before.lengths[0] == TOKENS && before.lengths[1] == 0);
  CHECK(before.block_counts[0] == SEQ_BLOCKS && before.block_counts[1] == 0);
  const uint32_t prefix_block = before.blocks[0][0];
  CHECK(before.refs[prefix_block] == 1 && before.stats.used_blocks == SEQ_BLOCKS);
  CHECK(before.stats.prefix_cache_hits == 0 && before.stats.cow_copies == 0 &&
        before.stats.shared_blocks == 0 && before.stats.prefix_cached_blocks == 1);
  for (size_t i = 0; i < 3; i++) {
    const et_paged_kv_prefix_key_v1 key = prefix_key(&foreign[i], tokens);
    check_rejected_context(&key, &before);
  }

  const et_paged_kv_prefix_key_v1 key = prefix_key(context, tokens);
  uint32_t hit = 0;
  CHECK(et_paged_kv_prefix_cache_attach_with_context_v1(
            witness.pool, witness.recipient, &key, &hit) == ET_PAGED_KV_OK);
  CHECK(hit == 1);
  // Only the recipient block table, length, reference count and hit telemetry
  // should change. The entire physical KV storage must stay byte-identical.
  before.lengths[1] = BLOCK_SIZE;
  before.block_counts[1] = 1;
  before.blocks[1][0] = prefix_block;
  before.refs[prefix_block]++;
  before.stats.shared_blocks++;
  before.stats.prefix_cache_hits++;
  capture_pool(&after);
  CHECK(memcmp(&before, &after, sizeof(after)) == 0);
}

static void check_cached_kv(const et_paged_kv_pool *pool, uint32_t sequence,
                            const float *keys, const float *values) {
  for (uint32_t token = 0; token < TOKENS; token++) {
    for (uint32_t head = 0; head < HEADS; head++) {
      const size_t index = (head * TOKENS + token) * WIDTH;
      float k[WIDTH], v[WIDTH];
      CHECK(et_paged_kv_read_token(pool, sequence, token, 0, head, k, v) ==
            ET_PAGED_KV_OK);
      CHECK(memcmp(k, keys + index, sizeof(k)) == 0 &&
            memcmp(v, values + index, sizeof(v)) == 0);
    }
  }
}

static void paged_attention(const et_kernel_call_v1 *call,
                            float result[ATTENTION_WORDS]) {
  const uint64_t request_shape[] = {1, HEADS, HEADS, TOKENS, TOKENS, WIDTH};
  const uint64_t head_shape[] = {1, HEADS, TOKENS, WIDTH};
  const uint64_t position_shape[] = {1, TOKENS};
  const uint64_t mask_shape[] = {1, TOKENS, TOKENS};
  const et_kernel_tensor_view_v1 *inputs = call->inputs;
  CHECK(call->request->rank == 6 &&
        memcmp(call->request->shape, request_shape, sizeof(request_shape)) == 0);
  CHECK(call->request->deterministic == 1 &&
        strcmp(call->request->dtype, "f32") == 0 &&
        strcmp(call->request->device, "cpu") == 0);
  CHECK(call->input_count == 6 && call->input_stride == sizeof(*inputs));
  CHECK(call->output_count == 1 && call->output_stride == sizeof(*inputs));
  for (size_t i = 0; i < 3; i++)
    check_view(&inputs[i], "f32", 4, head_shape, sizeof(witness.dense));
  for (size_t i = 3; i < 5; i++)
    check_view(&inputs[i], "i64", 2, position_shape, TOKENS * sizeof(int64_t));
  check_view(&inputs[5], "bool", 3, mask_shape, TOKENS * TOKENS);
  check_view(call->outputs, "f32", 4, head_shape, sizeof(witness.dense));

  // The paged kernel sees the entire live sequence and takes no explicit mask.
  const int64_t *query_positions = inputs[3].data;
  const int64_t *key_positions = inputs[4].data;
  const uint8_t *keep = inputs[5].data;
  for (size_t query = 0; query < TOKENS; query++) {
    CHECK(query_positions[query] == (int64_t)query);
    CHECK(key_positions[query] == (int64_t)query);
    for (size_t key = 0; key < TOKENS; key++)
      CHECK(keep[query * TOKENS + key] == (uint8_t)(key <= query));
  }

  const et_paged_kv_config fresh_config = {
      .num_blocks = SEQ_BLOCKS, .num_layers = 1,
      .num_kv_heads = HEADS, .block_size = BLOCK_SIZE, .head_dim = WIDTH,
      .max_seqs = 1, .max_blocks_per_seq = SEQ_BLOCKS};
  const int fresh = witness.route == FRESH;
  et_paged_kv_pool *pool = witness.pool;
  uint32_t sequence = witness.donor;
  if (fresh) {
    CHECK(et_paged_kv_create(&fresh_config, NULL, 0, &pool) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_seq_alloc(pool, &sequence) == ET_PAGED_KV_OK);
  }

  const float *q = inputs[0].data, *k = inputs[1].data, *v = inputs[2].data;
  const size_t prefix_bytes = BLOCK_SIZE * WIDTH * sizeof(float);
  size_t start = 0;
  if (witness.route == REUSE) {
    sequence = witness.recipient;
    start = BLOCK_SIZE;
    for (size_t head = 0; head < HEADS; head++) {
      const size_t index = head * TOKENS * WIDTH;
      CHECK(memcmp(q + index, witness.donor_q + index, prefix_bytes) == 0);
      memcpy(result + index, witness.donor_attention + index, prefix_bytes);
    }
  }
  const float scale = 1.0f / sqrtf((float)WIDTH);
  for (size_t token = start; token < TOKENS; token++) {
    float query[HEADS * WIDTH], key[HEADS * WIDTH], value[HEADS * WIDTH];
    float row[HEADS * WIDTH];
    for (size_t head = 0; head < HEADS; head++) {
      const size_t index = (head * TOKENS + token) * WIDTH;
      memcpy(query + head * WIDTH, q + index, WIDTH * sizeof(float));
      memcpy(key + head * WIDTH, k + index, WIDTH * sizeof(float));
      memcpy(value + head * WIDTH, v + index, WIDTH * sizeof(float));
    }
    CHECK(et_paged_kv_append_token(pool, sequence, key, value) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_decode_attention_v1(pool, sequence, 0, HEADS, scale,
                                         query, row) == ET_PAGED_KV_OK);
    for (size_t head = 0; head < HEADS; head++)
      memcpy(result + (head * TOKENS + token) * WIDTH, row + head * WIDTH,
             WIDTH * sizeof(float));
  }
  check_cached_kv(pool, sequence, witness.dense_k, witness.dense_v);
  if (fresh) {
    et_paged_kv_stats stats;
    CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
    CHECK(stats.active_seqs == 1 && stats.used_blocks == SEQ_BLOCKS);
    CHECK(et_paged_kv_seq_free(pool, sequence) == ET_PAGED_KV_OK);
    CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
    CHECK(stats.active_seqs == 0 && stats.free_blocks == SEQ_BLOCKS);
    et_paged_kv_destroy(pool);
    memcpy(witness.fresh_attention, result, sizeof(witness.fresh_attention));
  } else if (witness.route == DONOR) {
    memcpy(witness.donor_q, q, sizeof(witness.donor_q));
    memcpy(witness.donor_attention, result, sizeof(witness.donor_attention));
  } else {
    CHECK(memcmp(result, witness.fresh_attention, sizeof(witness.fresh_attention)) == 0);
    witness.reuse_kv_words += 2 * ATTENTION_WORDS;
  }
}

int32_t __real_et_kernel_runtime_dispatch(
    const et_kernel_runtime *, const et_kernel_call_v1 *, et_kernel_error *);

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (strcmp(call->capability, "g3c4.layer-norm") == 0) {
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    uint32_t epsilon_bits;
    CHECK(call->input_count == 4 && call->input_stride == sizeof(*inputs));
    check_view(&inputs[3], "f32", 0, NULL, sizeof(epsilon_bits));
    memcpy(&epsilon_bits, inputs[3].data, sizeof(epsilon_bits));
    CHECK(epsilon_bits == norm_epsilon_bits);
  }
  if (strcmp(call->capability, "g3c4.causal-attention") != 0)
    return __real_et_kernel_runtime_dispatch(runtime, call, error);

  CHECK(strcmp(call->request->operation, "g3c4.causal-attention.forward") == 0);
  const et_kernel_tensor_view_v1 *output = call->outputs;
  if (witness.route == DENSE) {
    const int32_t status = __real_et_kernel_runtime_dispatch(runtime, call, error);
    CHECK(status == 0 && output->byte_length == sizeof(witness.dense));
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const uint64_t head_shape[] = {1, HEADS, TOKENS, WIDTH};
    check_view(&inputs[1], "f32", 4, head_shape, sizeof(witness.dense_k));
    check_view(&inputs[2], "f32", 4, head_shape, sizeof(witness.dense_v));
    memcpy(witness.dense_k, inputs[1].data, sizeof(witness.dense_k));
    memcpy(witness.dense_v, inputs[2].data, sizeof(witness.dense_v));
    memcpy(witness.dense, output->data, sizeof(witness.dense));
    witness.dense_calls++;
    return status;
  }

  float result[ATTENTION_WORDS];
  paged_attention(call, result);
  compare(result, witness.dense, ATTENTION_WORDS, 1e-6f,
          &witness.attention_error);
  memcpy(output->data, result, sizeof(result));
  if (witness.route == REUSE) {
    witness.reuse_attention_words += ATTENTION_WORDS;
    witness.reuse_calls++;
  } else {
    witness.attention_words += ATTENTION_WORDS;
    witness.paged_calls++;
  }
  et_kernel_error_clear(error);
  return 0;
}

static void forward(void *context, const int64_t tokens[TOKENS],
                    enum route route, float dense[LOGIT_WORDS],
                    float paged[LOGIT_WORDS]) {
  const size_t dense_calls = witness.dense_calls;
  const size_t paged_calls = witness.paged_calls;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, tokens, dense) == 0);
  CHECK(witness.dense_calls == dense_calls + 1 &&
        witness.paged_calls == paged_calls);
  witness.route = route;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, tokens, paged) == 0);
  witness.route = DENSE;
  CHECK(witness.paged_calls == paged_calls + 1 &&
        witness.dense_calls == dense_calls + 1);
  compare(paged, dense, LOGIT_WORDS, 2e-6f, &witness.logit_error);
  witness.logit_words += LOGIT_WORDS;
}

static void read_mapping(const char *path, uint8_t mapping[VOCAB]) {
  FILE *file = fopen(path, "rb");
  CHECK(file != NULL);
  CHECK(fread(mapping, 1, VOCAB, file) == VOCAB);
  CHECK(fgetc(file) == EOF && !ferror(file) && fclose(file) == 0);
}

static void prompt_ids(const uint8_t mapping[VOCAB], const uint8_t bytes[TOKENS],
                       int64_t tokens[TOKENS]) {
  for (size_t i = 0; i < TOKENS; i++) tokens[i] = mapping[bytes[i]];
}

static void check_pool_cleanup(void) {
  pool_snapshot current;
  capture_pool(&current);
  CHECK(current.lengths[0] == TOKENS && current.lengths[1] == TOKENS);
  CHECK(current.block_counts[0] == SEQ_BLOCKS && current.block_counts[1] == SEQ_BLOCKS);
  CHECK(current.blocks[0][0] == current.blocks[1][0] &&
        current.blocks[0][1] != current.blocks[1][1]);
  const uint32_t prefix = current.blocks[1][0];
  CHECK(current.refs[prefix] == 2 && current.refs[current.blocks[0][1]] == 1 &&
        current.refs[current.blocks[1][1]] == 1);
  CHECK(current.stats.used_blocks == POOL_BLOCKS && current.stats.active_seqs == 2 &&
        current.stats.shared_blocks == 1 && current.stats.prefix_cached_blocks == 1 &&
        current.stats.cow_copies == 0 && current.stats.prefix_cache_hits == 1);
  CHECK(et_paged_kv_seq_free(witness.pool, witness.donor) == ET_PAGED_KV_OK);

  uint32_t refs;
  CHECK(et_paged_kv_block_refcount(witness.pool, prefix, &refs) == ET_PAGED_KV_OK &&
        refs == 1);
  CHECK(et_paged_kv_stats_get(witness.pool, &current.stats) == ET_PAGED_KV_OK);
  CHECK(current.stats.used_blocks == SEQ_BLOCKS && current.stats.active_seqs == 1 &&
        current.stats.shared_blocks == 0 && current.stats.prefix_cached_blocks == 1);
  check_cached_kv(witness.pool, witness.recipient, witness.dense_k, witness.dense_v);
  CHECK(et_paged_kv_seq_free(witness.pool, witness.recipient) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(witness.pool, &current.stats) == ET_PAGED_KV_OK);
  CHECK(current.stats.free_blocks == POOL_BLOCKS && current.stats.used_blocks == 0 &&
        current.stats.active_seqs == 0 && current.stats.shared_blocks == 0 &&
        current.stats.prefix_cached_blocks == 0 && current.stats.cow_copies == 0 &&
        current.stats.prefix_cache_hits == 1);
  for (uint32_t block = 0; block < POOL_BLOCKS; block++) {
    CHECK(et_paged_kv_block_refcount(witness.pool, block, &refs) == ET_PAGED_KV_OK &&
          refs == 0);
  }
  et_paged_kv_destroy(witness.pool);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  uint8_t mapping[VOCAB];
  read_mapping(argv[1], mapping);
  static unsigned char identities[14];
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (size_t i = 0; i < sizeof(identities); i++)
    CHECK(et_g3c4_private_model_owner_bind_v1(
              owner, (int64_t)i, &identities[i]) == 0);
  for (size_t i = 0; i < sizeof(identities); i++)
    CHECK(et_g3c4_private_model_owner_initialize_v1(owner, (int64_t)i) == 0);
  const size_t gamma_indices[] = {7, 9, 12};
  for (size_t i = 0; i < sizeof(gamma_indices) / sizeof(gamma_indices[0]); i++) {
    float *gamma = owner->parameters[gamma_indices[i]]->value->data;
    for (size_t j = 0; j < HEADS * WIDTH; j++) gamma[j] = 1.0f;
  }
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner) == 0);
  CHECK(et_g3c4_private_model_owner_commit_seal_v1(owner) == 0);
  void *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), VOCAB, INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) == 0);

  const uint8_t first_bytes[TOKENS] = {17, 99, 23, 27};
  const uint8_t other_bytes[TOKENS] = {17, 99, 24, 31};
  int64_t first_tokens[TOKENS], other_tokens[TOKENS];
  prompt_ids(mapping, first_bytes, first_tokens);
  prompt_ids(mapping, other_bytes, other_tokens);
  uint8_t image[MODEL_BYTES];
  model_image(context, image);
  et_paged_kv_prefix_context_v1 prefix_context;
  hash(image, sizeof(image), prefix_context.model);
  hash(mapping, sizeof(mapping), prefix_context.tokenizer);
  producer_digest(0, prefix_context.producer_config);

  // Derive one foreign key per changed identity input; all prompt IDs stay fixed.
  et_paged_kv_prefix_context_v1 foreign[] = {
      prefix_context, prefix_context, prefix_context};
  image[0] ^= 1u;
  hash(image, sizeof(image), foreign[0].model);
  const uint8_t last = mapping[VOCAB - 1];
  mapping[VOCAB - 1] = mapping[VOCAB - 2];
  mapping[VOCAB - 2] = last;
  hash(mapping, sizeof(mapping), foreign[1].tokenizer);
  producer_digest(1, foreign[2].producer_config);

  CHECK(et_paged_kv_create_with_prefix_context_v1(
            &reuse_config, &prefix_context, witness.storage, POOL_WORDS,
            &witness.pool) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(witness.pool, &witness.donor) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(witness.pool, &witness.recipient) == ET_PAGED_KV_OK);
  float first[LOGIT_WORDS], other[LOGIT_WORDS], dense[LOGIT_WORDS];
  forward(context, first_tokens, DONOR, dense, first);
  const et_paged_kv_prefix_key_v1 donor_key = prefix_key(&prefix_context,
                                                        first_tokens);
  CHECK(et_paged_kv_prefix_cache_register_with_context_v1(
            witness.pool, witness.donor, 0, &donor_key) == ET_PAGED_KV_OK);

  const int64_t invalid_tokens[TOKENS] = {17, 99, 24, VOCAB};
  memcpy(other, first, sizeof(other));
  witness.route = FRESH;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, invalid_tokens, other) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_SELECTOR);
  CHECK(witness.paged_calls == 1 && memcmp(other, first, sizeof(other)) == 0);
  witness.route = DENSE;

  forward(context, other_tokens, FRESH, dense, other);
  const size_t prefix_words = BLOCK_SIZE * VOCAB;
  CHECK(memcmp(first, other, prefix_words * sizeof(float)) == 0);
  CHECK(memcmp(first + prefix_words, other + prefix_words,
               (LOGIT_WORDS - prefix_words) * sizeof(float)) != 0);

  attach_prefix(&prefix_context, other_tokens, foreign);
  float reused[LOGIT_WORDS];
  witness.route = REUSE;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, other_tokens, reused) == 0);
  witness.route = DENSE;
  CHECK(witness.reuse_calls == 1 && witness.dense_calls == 2 &&
        witness.paged_calls == 2);
  CHECK(memcmp(reused, other, sizeof(reused)) == 0);
  compare(reused, dense, LOGIT_WORDS, 2e-6f, &witness.logit_error);
  witness.reuse_logit_words += LOGIT_WORDS;
  check_pool_cleanup();

  CHECK(et_g3c4_private_call_finish_v1(context) == 0);
  CHECK(et_g3c4_private_generator_close_v1(context) == 0);
  printf("PKV5 model attention PASS: dense=%zu paged=%zu attention=%zu logits=%zu "
         "max_attention_error=%a max_logit_error=%a\n", witness.dense_calls,
         witness.paged_calls, witness.attention_words, witness.logit_words,
         (double)witness.attention_error, (double)witness.logit_error);
  printf("PKV5 live context prefix PASS: hits=1 copies=0 rejections=6 "
         "attention=%zu logits=%zu kv=%zu suffix_tokens=%u\n",
         witness.reuse_attention_words, witness.reuse_logit_words,
         witness.reuse_kv_words, (unsigned)(TOKENS - BLOCK_SIZE));
  return 0;
}
