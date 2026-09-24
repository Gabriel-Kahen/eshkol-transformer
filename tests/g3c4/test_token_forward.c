#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROVIDER_ROUTES_PRIVATE 1
#define ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE 1
#define ET_G3C4_SAMPLER_TRANSPORT_PRIVATE 1
#define ET_G3C4_TOKEN_FRAME_PRIVATE 1
#define ET_G3C4_TOKEN_FORWARD_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <math.h>
#include <stdio.h>

void et_m3t_test_k1_fail_after(size_t allowed);
void et_m3t_test_k1_allocator_reset(void);

static size_t checks;
#define CHECK(condition) do { \
  checks++; \
  if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s [domain=%lld category=%lld code=%lld]\n", \
        __FILE__, __LINE__, #condition, \
        (long long)et_g3c4_private_last_error_domain_v1(), \
        (long long)et_g3c4_private_last_error_category_v1(), \
        (long long)et_g3c4_private_last_error_code_v1()); \
    exit(1); \
  } \
} while (0)
#define OK(expression) CHECK((expression) == 0)

static void check_k1_error(int64_t category, int64_t code) {
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
}

int32_t __real_et_kernel_runtime_dispatch(
    const et_kernel_runtime *, const et_kernel_call_v1 *, et_kernel_error *);

static const char *const expected_capabilities[21] = {
  "g3n.embedding-forward", "g3c4.embedding-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward",
  "g3n.linear-forward", "g3n.linear-forward", "g3n.linear-forward",
  "g3n.head-layout-forward", "g3n.head-layout-forward",
  "g3n.head-layout-forward", "g3c4.causal-attention",
  "g3n.head-layout-forward", "g3n.linear-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward",
  "g3n.linear-forward", "kernel.activation", "g3n.linear-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward", "g3n.linear-forward"
};
static const char *const expected_operations[21] = {
  "g3n.embedding.forward", "g3c4.embedding.forward",
  "g3n.residual.forward", "g3n.layer-norm.forward",
  "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias",
  "g3n.linear.forward-no-bias", "g3n.heads.split.forward",
  "g3n.heads.split.forward", "g3n.heads.split.forward",
  "g3c4.causal-attention.forward", "g3n.heads.merge.forward",
  "g3n.linear.forward-no-bias", "g3n.residual.forward",
  "g3n.layer-norm.forward", "g3n.linear.forward-no-bias",
  "gelu.forward", "g3n.linear.forward-no-bias",
  "g3n.residual.forward", "g3n.layer-norm.forward",
  "g3n.linear.forward-no-bias"
};
static int record_forward;
static size_t forward_dispatches;
static size_t fail_forward_at = SIZE_MAX;
static int64_t expected_position;
static float staged_key[4], staged_value[4];

static void check_request_shape(size_t index, const et_kernel_request_v1 *r) {
  static const uint64_t rows[21][6] = {
    {1,1,256,4}, {1,1,4,4}, {1,1,4}, {1,1,4},
    {1,1,4,4}, {1,1,4,4}, {1,1,4,4},
    {1,1,2,2}, {1,1,2,2}, {1,1,2,2},
    {1,2,2,1,4,2}, {1,1,2,2}, {1,1,4,4}, {1,1,4},
    {1,1,4}, {1,1,4,8}, {1,1,8}, {1,1,8,4},
    {1,1,4}, {1,1,4}, {1,1,4,256}
  };
  static const size_t ranks[21] = {
    4,4,3,3,4,4,4,4,4,4,6,4,4,3,3,4,3,4,3,3,4
  };
  CHECK(r->rank == ranks[index]);
  for (size_t dim = 0u; dim < ranks[index]; dim++)
    CHECK(r->shape[dim] == rows[index][dim]);
}

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_forward && strncmp(call->capability, "g3s.", 4u) != 0) {
    const size_t index = forward_dispatches++;
    CHECK(index < 21u);
    CHECK(strcmp(call->capability, expected_capabilities[index]) == 0);
    CHECK(strcmp(call->request->operation, expected_operations[index]) == 0);
    CHECK(call->request->deterministic == 1u);
    check_request_shape(index, call->request);
    if (index == 10u) {
      const et_kernel_tensor_view_v1 *inputs = call->inputs;
      CHECK(*(const int64_t *)inputs[3].data == expected_position);
      for (size_t key = 0u; key < 4u; key++) {
        CHECK(((const int64_t *)inputs[4].data)[key] == (int64_t)key);
        CHECK(((const uint8_t *)inputs[5].data)[key] ==
              (key <= (size_t)expected_position ? 1u : 0u));
      }
      for (size_t head = 0u; head < 2u; head++)
        for (size_t dim = 0u; dim < 2u; dim++) {
          const size_t cache_index = (head * 4u +
              (size_t)expected_position) * 2u + dim;
          const size_t local_index = head * 2u + dim;
          staged_key[local_index] =
              ((const float *)inputs[1].data)[cache_index];
          staged_value[local_index] =
              ((const float *)inputs[2].data)[cache_index];
        }
    }
    if (index == fail_forward_at) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.token-forward-cut");
      strcpy(error->message, "injected token forward failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return __real_et_kernel_runtime_dispatch(runtime, call, error);
}

static unsigned char identities[14];

static et_g3c4_model_owner_internal *create_owner(void) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static et_g3c4_context_internal *create_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context =
      et_g3c4_private_generator_seed_v1(
          owner, 7, 0, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  return context;
}

typedef struct cache_snapshot {
  float keys[16];
  float values[16];
  int64_t length;
  uint8_t keep[4];
} cache_snapshot;

static void snapshot_cache(et_a2_kv_cache *cache, cache_snapshot *snapshot) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
            borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  memcpy(snapshot->keys, keys->data, sizeof(snapshot->keys));
  memcpy(snapshot->values, values->data, sizeof(snapshot->values));
  snapshot->length = *(const int64_t *)lengths->data;
  memcpy(snapshot->keep, keep->data, sizeof(snapshot->keep));
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
}

static void check_snapshot(
    et_a2_kv_cache *cache, const cache_snapshot *expected) {
  cache_snapshot actual;
  snapshot_cache(cache, &actual);
  CHECK(memcmp(actual.keys, expected->keys, sizeof(actual.keys)) == 0);
  CHECK(memcmp(actual.values, expected->values, sizeof(actual.values)) == 0);
  CHECK(actual.length == expected->length);
  CHECK(memcmp(actual.keep, expected->keep, sizeof(actual.keep)) == 0);
}

static void prefill_cache(
    et_a2_kv_cache *cache, int64_t length, float marker) {
  static const uint64_t count_shape[1] = {1u};
  uint64_t kv_shape[4] = {1u, 2u, 0u, 2u};
  float keys[12], values[12];
  et_kernel_tensor_view_v1 counts, key_view, value_view;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_error error;
  CHECK(length >= 1 && length <= 3);
  kv_shape[2] = (uint64_t)length;
  for (size_t index = 0u; index < (size_t)length * 4u; index++) {
    keys[index] = marker + (float)index;
    values[index] = marker + 20.0f + (float)index;
  }
  counts = et_g3c4_view(&length, sizeof(length), "i64", 1u, count_shape);
  key_view = et_g3c4_view(
      keys, (size_t)length * 4u * sizeof(float), "f32", 4u, kv_shape);
  value_view = et_g3c4_view(
      values, (size_t)length * 4u * sizeof(float), "f32", 4u, kv_shape);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            cache, (uint64_t)length, &counts, &transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
            transaction, 0u, &key_view, &value_view, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_commit_v1(&transaction, &error) == 0);
}

static void prepare_sample_logits(float logits[1024], int64_t token) {
  for (size_t index = 0u; index < 1024u; index++) logits[index] = 0.0f;
  logits[3u * 256u + (size_t)token] = 10.0f;
}

static void successful_position(
    et_g3c4_model_owner_internal *owner, int64_t position,
    float marker, float output_logits[256]) {
  float sample_logits[1024];
  int64_t speculative = -1, published = -1;
  int64_t saved_rng[4];
  et_g3c4_context_internal *context = create_generator(owner);
  cache_snapshot before, after;
  if (position != 0) prefill_cache(context->cache, position, marker);
  snapshot_cache(context->cache, &before);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  prepare_sample_logits(sample_logits, 37);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative));
  CHECK(speculative == 37);
  CHECK(context->token_frame_position == position);
  expected_position = position;
  forward_dispatches = 0u;
  fail_forward_at = SIZE_MAX;
  record_forward = 1;
  OK(et_g3c4_private_token_forward_v1(
      context, speculative, output_logits));
  record_forward = 0;
  CHECK(forward_dispatches == 21u);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  for (size_t index = 0u; index < 256u; index++)
    CHECK(isfinite(output_logits[index]));
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == speculative);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  snapshot_cache(context->cache, &after);
  CHECK(after.length == position + 1);
  for (size_t index = 0u; index < 4u; index++)
    CHECK(after.keep[index] == (index <= (size_t)position ? 1u : 0u));
  for (size_t head = 0u; head < 2u; head++)
    for (size_t dim = 0u; dim < 2u; dim++) {
      const size_t cache_index =
          (head * 4u + (size_t)position) * 2u + dim;
      const size_t local_index = head * 2u + dim;
      CHECK(after.keys[cache_index] == staged_key[local_index]);
      CHECK(after.values[cache_index] == staged_value[local_index]);
    }
  for (size_t head = 0u; head < 2u; head++)
    for (size_t prior = 0u; prior < (size_t)position; prior++)
      for (size_t dim = 0u; dim < 2u; dim++) {
        const size_t index = (head * 4u + prior) * 2u + dim;
        CHECK(after.keys[index] == before.keys[index]);
        CHECK(after.values[index] == before.values[index]);
      }
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void dispatch_failure_cuts(et_g3c4_model_owner_internal *owner) {
  for (size_t cut = 0u; cut < 21u; cut++) {
    float sample_logits[1024], output[256];
    int64_t speculative = -1;
    int64_t saved_rng[4];
    cache_snapshot before;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill_cache(context->cache, 3, 100.0f + (float)cut);
    snapshot_cache(context->cache, &before);
    memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
    prepare_sample_logits(sample_logits, 37);
    for (size_t index = 0u; index < 256u; index++) output[index] = -123.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    OK(et_g3c4_private_token_frame_begin_v1(
        context, sample_logits, &speculative));
    expected_position = 3;
    forward_dispatches = 0u;
    fail_forward_at = cut;
    record_forward = 1;
    CHECK(et_g3c4_private_token_forward_v1(
              context, speculative, output) != 0);
    record_forward = 0;
    check_k1_error(
        ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(forward_dispatches == cut + 1u);
    CHECK(et_g3c4_token_frame_idle(context));
    CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
    for (size_t index = 0u; index < 256u; index++) CHECK(output[index] == -123.0f);
    check_snapshot(context->cache, &before);
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  fail_forward_at = SIZE_MAX;
}

static void allocation_and_rejection_cuts(
    et_g3c4_model_owner_internal *owner) {
  float sample_logits[1024], output[256];
  int64_t speculative = -1;
  int64_t saved_rng[4];
  size_t begin_failure_cuts = 0u;
  cache_snapshot before;
  et_g3c4_context_internal *context;
  prepare_sample_logits(sample_logits, 37);

  for (size_t allowed = 0u; allowed <= 8u; allowed++) {
    context = create_generator(owner);
    snapshot_cache(context->cache, &before);
    memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    speculative = -1;
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    int64_t status = et_g3c4_private_token_frame_begin_v1(
        context, sample_logits, &speculative);
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status == 0) {
      CHECK(allowed == 8u);
      CHECK(speculative == 37);
      OK(et_g3c4_private_token_frame_abort_v1(context));
    } else {
      begin_failure_cuts++;
      check_k1_error(
          ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED);
      CHECK(speculative == -1);
      CHECK(et_g3c4_token_frame_idle(context));
    }
    CHECK(memcmp(context->generator_rng_words,
                 saved_rng, sizeof(saved_rng)) == 0);
    check_snapshot(context->cache, &before);
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  CHECK(begin_failure_cuts == 8u);

  context = create_generator(owner);
  snapshot_cache(context->cache, &before);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative));
  for (size_t index = 0u; index < 256u; index++) output[index] = -123.0f;
  et_m3t_test_k1_fail_after(0u);
  CHECK(et_g3c4_private_token_forward_v1(
            context, speculative, output) != 0);
  et_m3t_test_k1_allocator_reset();
  check_k1_error(
      ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  for (size_t index = 0u; index < 256u; index++) CHECK(output[index] == -123.0f);
  check_snapshot(context->cache, &before);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner);
  snapshot_cache(context->cache, &before);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_a2_kv_cache_test_fail_alloc_after_v1(8u);
  OK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative));
  for (size_t index = 0u; index < 256u; index++) output[index] = -123.0f;
  CHECK(et_g3c4_private_token_forward_v1(
            context, speculative, output) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  check_k1_error(
      ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  for (size_t index = 0u; index < 256u; index++) CHECK(output[index] == -123.0f);
  check_snapshot(context->cache, &before);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative));
  CHECK(et_g3c4_private_token_forward_v1(
            context, speculative == 255 ? 254 : speculative + 1,
            output) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(et_g3c4_private_token_forward_v1(
            context, speculative, (float *)context) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  OK(et_g3c4_private_token_frame_abort_v1(context));
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void cross_provider_prefix_parity(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[4] = {3, 7, 11, 37};
  float reference[1024], sample_logits[1024], actual[256];
  et_g3c4_context_internal *reference_context = create_generator(owner);
  et_g3c4_context_internal *token_context = create_generator(owner);
  int64_t speculative, published;
  cache_snapshot final;

  OK(et_g3c4_private_call_acquire_v1(reference_context, 0, 0));
  OK(et_g3c4_private_full_prefix_forward_v1(
      reference_context, tokens, reference));
  OK(et_g3c4_private_call_finish_v1(reference_context));
  OK(et_g3c4_private_generator_close_v1(reference_context));

  for (size_t position = 0u; position < 4u; position++) {
    prepare_sample_logits(sample_logits, tokens[position]);
    OK(et_g3c4_private_call_acquire_v1(token_context, 2, 1));
    OK(et_g3c4_private_token_frame_begin_v1(
        token_context, sample_logits, &speculative));
    CHECK(speculative == tokens[position]);
    OK(et_g3c4_private_token_forward_v1(
        token_context, speculative, actual));
    CHECK(memcmp(actual, reference + position * 256u, sizeof(actual)) == 0);
    OK(et_g3c4_private_call_prepare_end_v1(token_context));
    OK(et_g3c4_private_token_frame_publish_v1(token_context, &published));
    CHECK(published == tokens[position]);
    OK(et_g3c4_private_call_finish_v1(token_context));
  }
  snapshot_cache(token_context->cache, &final);
  CHECK(final.length == 4);
  OK(et_g3c4_private_generator_close_v1(token_context));
}

int main(void) {
  float empty_logits[256], continued_logits[256], repeated_logits[256];
  et_g3c4_model_owner_internal *owner = create_owner();
  successful_position(owner, 0, 0.0f, empty_logits);
  successful_position(owner, 3, 50.0f, continued_logits);
  successful_position(owner, 3, 50.0f, repeated_logits);
  CHECK(memcmp(continued_logits, repeated_logits,
               sizeof(continued_logits)) == 0);
  CHECK(memcmp(empty_logits, continued_logits, sizeof(empty_logits)) != 0);
  cross_provider_prefix_parity(owner);
  dispatch_failure_cuts(owner);
  allocation_and_rejection_cuts(owner);
  printf("G3-C4 token forward PASS: checks=%zu roles=21 cuts=31 positions=2\n",
         checks);
  return 0;
}
