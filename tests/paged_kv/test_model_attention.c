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
#include "eshkol_transformer/paged_kv_abi.h"

#include <math.h>
#include <stdio.h>

enum { TOKENS = 4, HEADS = 2, WIDTH = 2, VOCAB = 256, BLOCK_SIZE = 2 };
enum { ATTENTION_WORDS = HEADS * TOKENS * WIDTH, LOGIT_WORDS = TOKENS * VOCAB };

#define CHECK(condition) do {                                           \
  if (!(condition)) {                                                    \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);  \
    exit(1);                                                             \
  }                                                                      \
} while (0)

static struct {
  int use_paged;
  size_t dense_calls, paged_calls, attention_words, logit_words;
  float dense[ATTENTION_WORDS];
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
  CHECK(view->rank == rank && view->shape != NULL &&
        memcmp(view->shape, shape, rank * sizeof(shape[0])) == 0);
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

  const et_paged_kv_config config = {
      .num_blocks = TOKENS / BLOCK_SIZE, .num_layers = 1,
      .num_kv_heads = HEADS, .block_size = BLOCK_SIZE, .head_dim = WIDTH,
      .max_seqs = 1, .max_blocks_per_seq = TOKENS / BLOCK_SIZE};
  et_paged_kv_pool *pool = NULL;
  uint32_t sequence;
  CHECK(et_paged_kv_create(&config, NULL, 0, &pool) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_seq_alloc(pool, &sequence) == ET_PAGED_KV_OK);

  const float *q = inputs[0].data, *k = inputs[1].data, *v = inputs[2].data;
  const float scale = 1.0f / sqrtf((float)WIDTH);
  for (size_t token = 0; token < TOKENS; token++) {
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
  et_paged_kv_stats stats;
  CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
  CHECK(stats.active_seqs == 1 && stats.used_blocks == config.num_blocks);
  CHECK(et_paged_kv_seq_free(pool, sequence) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_stats_get(pool, &stats) == ET_PAGED_KV_OK);
  CHECK(stats.active_seqs == 0 && stats.free_blocks == config.num_blocks);
  et_paged_kv_destroy(pool);
}

int32_t __real_et_kernel_runtime_dispatch(
    const et_kernel_runtime *, const et_kernel_call_v1 *, et_kernel_error *);

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (strcmp(call->capability, "g3c4.causal-attention") != 0)
    return __real_et_kernel_runtime_dispatch(runtime, call, error);

  CHECK(strcmp(call->request->operation, "g3c4.causal-attention.forward") == 0);
  const et_kernel_tensor_view_v1 *output = call->outputs;
  if (!witness.use_paged) {
    const int32_t status = __real_et_kernel_runtime_dispatch(runtime, call, error);
    CHECK(status == 0 && output->byte_length == sizeof(witness.dense));
    memcpy(witness.dense, output->data, sizeof(witness.dense));
    witness.dense_calls++;
    return status;
  }

  float result[ATTENTION_WORDS];
  paged_attention(call, result);
  compare(result, witness.dense, ATTENTION_WORDS, 1e-6f,
          &witness.attention_error);
  memcpy(output->data, result, sizeof(result));
  witness.attention_words += ATTENTION_WORDS;
  witness.paged_calls++;
  et_kernel_error_clear(error);
  return 0;
}

static void forward(void *context, const int64_t tokens[TOKENS],
                    float paged[LOGIT_WORDS]) {
  float dense[LOGIT_WORDS];
  const size_t dense_calls = witness.dense_calls;
  const size_t paged_calls = witness.paged_calls;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, tokens, dense) == 0);
  CHECK(witness.dense_calls == dense_calls + 1 &&
        witness.paged_calls == paged_calls);
  witness.use_paged = 1;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, tokens, paged) == 0);
  witness.use_paged = 0;
  CHECK(witness.paged_calls == paged_calls + 1 &&
        witness.dense_calls == dense_calls + 1);
  compare(paged, dense, LOGIT_WORDS, 2e-6f, &witness.logit_error);
  witness.logit_words += LOGIT_WORDS;
}

int main(void) {
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

  const int64_t first_tokens[TOKENS] = {17, 99, 23, 27};
  const int64_t other_tokens[TOKENS] = {17, 99, 24, 31};
  float first[LOGIT_WORDS], other[LOGIT_WORDS];
  forward(context, first_tokens, first);

  const int64_t invalid_tokens[TOKENS] = {17, 99, 24, VOCAB};
  memcpy(other, first, sizeof(other));
  witness.use_paged = 1;
  CHECK(et_g3c4_private_full_prefix_forward_v1(context, invalid_tokens, other) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_SELECTOR);
  CHECK(witness.paged_calls == 1 && memcmp(other, first, sizeof(other)) == 0);
  witness.use_paged = 0;

  forward(context, other_tokens, other);
  const size_t prefix_words = BLOCK_SIZE * VOCAB;
  CHECK(memcmp(first, other, prefix_words * sizeof(float)) == 0);
  CHECK(memcmp(first + prefix_words, other + prefix_words,
               (LOGIT_WORDS - prefix_words) * sizeof(float)) != 0);

  CHECK(et_g3c4_private_call_finish_v1(context) == 0);
  CHECK(et_g3c4_private_generator_close_v1(context) == 0);
  printf("PKV5 model attention PASS: dense=%zu paged=%zu attention=%zu logits=%zu "
         "max_attention_error=%a max_logit_error=%a\n", witness.dense_calls,
         witness.paged_calls, witness.attention_words, witness.logit_words,
         (double)witness.attention_error, (double)witness.logit_error);
  return 0;
}
