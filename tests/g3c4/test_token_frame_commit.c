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

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <math.h>
#include <stdio.h>

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

int32_t __real_et_kernel_runtime_dispatch(
    const et_kernel_runtime *, const et_kernel_call_v1 *, et_kernel_error *);
static int fail_sampler;
static size_t sampler_dispatches;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (strcmp(call->capability, "g3s.greedy") == 0 ||
      strcmp(call->capability, "g3s.categorical") == 0) {
    sampler_dispatches++;
    if (fail_sampler) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.token-frame-cut");
      strcpy(error->message, "injected sampler failure");
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
    et_g3c4_model_owner_internal *owner, int64_t mode) {
  et_g3c4_context_internal *context =
      et_g3c4_private_generator_seed_v1(
          owner, 7, mode, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  CHECK(et_g3c4_token_frame_idle(context));
  return context;
}

static void check_cache(
    et_a2_kv_cache *cache, int64_t expected_length,
    const float *expected_keys, const float *expected_values) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *keep = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
            borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  CHECK(*(const int64_t *)lengths->data == expected_length);
  for (size_t index = 0u; index < 4u; index++)
    CHECK(((const uint8_t *)keep->data)[index] ==
          (index < (size_t)expected_length ? 1u : 0u));
  for (size_t index = 0u; index < 16u; index++) {
    float expected_key = 0.0f;
    float expected_value = 0.0f;
    if (expected_length == 1 && index < 2u) {
      expected_key = expected_keys[index];
      expected_value = expected_values[index];
    } else if (expected_length == 1 && index >= 8u && index < 10u) {
      expected_key = expected_keys[index - 6u];
      expected_value = expected_values[index - 6u];
    }
    CHECK(((const float *)keys->data)[index] == expected_key);
    CHECK(((const float *)values->data)[index] == expected_value);
  }
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
}

static void check_cache_busy(et_a2_kv_cache *cache) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            cache, &borrow, &error) != 0);
  CHECK(borrow == NULL);
  CHECK(error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
}

static void categorical_commit(et_g3c4_model_owner_internal *owner) {
  float logits[1024] = {0};
  float keys[4], values[4];
  int64_t speculative = -17;
  int64_t published = -19;
  int64_t saved_rng[4];
  int64_t expected_successor[4];
  et_g3c4_context_internal *context = create_generator(owner, 1);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(context, logits, &speculative));
  CHECK(speculative >= 0 && speculative <= 255);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(context->token_frame_transaction != NULL);
  check_cache_busy(context->cache);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  memcpy(expected_successor, context->token_frame_successor,
         sizeof(expected_successor));
  CHECK(expected_successor[0] == 1 && expected_successor[1] == 7);
  CHECK(expected_successor[2] == 1 && expected_successor[3] == 0);
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  CHECK(et_g3c4_private_token_frame_stage_v1(
            context, speculative == 255 ? 254 : speculative + 1,
            keys, values) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  for (size_t index = 0u; index < 4u; index++) {
    keys[index] = (float)(speculative + 1 + (int64_t)index);
    values[index] = (float)(speculative + 11 + (int64_t)index);
  }
  OK(et_g3c4_private_token_frame_stage_v1(
      context, speculative, keys, values));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  check_cache_busy(context->cache);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  CHECK(et_g3c4_private_token_frame_publish_v1(
            context, &context->generator_rng_words[0]) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == speculative);
  CHECK(context->budget == 0);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, expected_successor,
               sizeof(expected_successor)) == 0);
  check_cache(context->cache, 1, keys, values);
  CHECK(et_g3c4_private_token_frame_publish_v1(context, &published) != 0);
  check_cache(context->cache, 1, keys, values);
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void greedy_commit(et_g3c4_model_owner_internal *owner) {
  float logits[1024] = {0};
  float keys[4] = {1, 2, 3, 4};
  float values[4] = {5, 6, 7, 8};
  int64_t speculative = -17;
  int64_t published = -19;
  int64_t saved_rng[4];
  et_g3c4_context_internal *context = create_generator(owner, 0);
  logits[3u * 256u + 37u] = 10.0f;
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_token_frame_begin_v1(
      context, logits, &speculative));
  CHECK(speculative == 37);
  CHECK(memcmp(context->token_frame_successor, saved_rng,
               sizeof(saved_rng)) == 0);
  OK(et_g3c4_private_token_frame_stage_v1(
      context, speculative, keys, values));
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == 37);
  CHECK(memcmp(context->generator_rng_words, saved_rng,
               sizeof(saved_rng)) == 0);
  check_cache(context->cache, 1, keys, values);
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void failure_cuts(et_g3c4_model_owner_internal *owner) {
  float logits[1024] = {0};
  float keys[4] = {1, 2, 3, 4};
  float values[4] = {5, 6, 7, 8};
  int64_t speculative;
  int64_t saved_rng[4];
  et_g3c4_context_internal *context;

  context = create_generator(owner, 0);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  speculative = -17;
  fail_sampler = 1;
  CHECK(et_g3c4_private_token_frame_begin_v1(
            context, logits, &speculative) != 0);
  fail_sampler = 0;
  CHECK(speculative == -17);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  check_cache(context->cache, 0, keys, values);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner, 0);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  speculative = -17;
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_token_frame_begin_v1(
            context, logits, &speculative) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(speculative == -17);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  check_cache(context->cache, 0, keys, values);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner, 0);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  speculative = -17;
  OK(et_g3c4_private_token_frame_begin_v1(context, logits, &speculative));
  keys[0] = NAN;
  CHECK(et_g3c4_private_token_frame_stage_v1(
            context, speculative, keys, values) != 0);
  keys[0] = 1.0f;
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  check_cache(context->cache, 0, keys, values);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner, 0);
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  speculative = -17;
  OK(et_g3c4_private_token_frame_begin_v1(context, logits, &speculative));
  OK(et_g3c4_private_token_frame_stage_v1(
      context, speculative, keys, values));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  check_cache_busy(context->cache);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  check_cache(context->cache, 0, keys, values);
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  categorical_commit(owner);
  greedy_commit(owner);
  failure_cuts(owner);
  printf("G3-C4 token frame commit PASS: checks=%zu sampler-dispatches=%zu modes=2 cuts=4\n",
         checks, sampler_dispatches);
  return 0;
}
