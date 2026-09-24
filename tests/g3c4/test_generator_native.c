#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1
#define ET_G3C4_GENERATOR_PRIVATE 1

#ifdef ET_G3C4_GENERATOR_A2_TU
#include "../../native/a2_kv_cache.c"

size_t et_g3c4_generator_test_a2_cache_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache *item = live_caches;
       item != NULL; item = item->registry_next) count++;
  return count;
}
#else
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <stdio.h>

size_t et_g3c4_generator_test_a2_cache_count(void);

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

static unsigned char identities[14];

static void expect_error(int64_t domain, int64_t category, int64_t code) {
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
}

static size_t transport_count(uint32_t kind) {
  size_t count = 0u;
  for (et_g3c4_transport_header_internal *item = et_g3c4_transport_registry;
       item != NULL; item = item->registry_next)
    if (item->kind == kind) count++;
  return count;
}

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

static et_g3c4_context_internal *seed_generator(
    et_g3c4_model_owner_internal *owner, int64_t seed) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)et_g3c4_private_generator_seed_v1(
          owner, seed, 0, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  CHECK(context->transport.magic == ET_G3C4_CONTEXT_MAGIC);
  CHECK(context->transport.kind == ET_G3C4_CONTEXT_KIND);
  CHECK(context->transport.state == ET_G3C4_CONTEXT_LIVE);
  CHECK(context->transport.busy == ET_G3C4_CALL_IDLE);
  CHECK(context->generator_kind == 1u && context->generator_ready == 1u);
  CHECK(context->generator_rng_words[0] == 1);
  CHECK(context->generator_rng_words[1] == seed);
  CHECK(context->generator_rng_words[2] == 0);
  CHECK(context->generator_rng_words[3] == 0);
  return context;
}

static void layout_and_macro_surface(void) {
  CHECK(sizeof(et_g3c4_transport_header_internal) == 32u);
  CHECK(sizeof(et_g3c4_context_internal) == 1520u);
  CHECK(sizeof(et_g3c4_rng_internal) == 64u);
  CHECK(offsetof(et_g3c4_context_internal, transport) == 0u);
  CHECK(offsetof(et_g3c4_rng_internal, transport) == 0u);
}

static void rng_lifetime(void) {
  int foreign = 0;
  CHECK(et_g3c4_private_rng_seed_v1(-1) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_SELECTOR);
  CHECK(et_g3c4_private_rng_clone_v1(&foreign) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_rng_word_v1(&foreign, 0) == 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_rng_release_v1(&foreign) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);

  const size_t baseline = transport_count(ET_G3C4_RNG_KIND);
  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_rng_seed_v1(0) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  CHECK(transport_count(ET_G3C4_RNG_KIND) == baseline);
  et_g3c4_context_allocation_limit = SIZE_MAX;

  et_g3c4_rng_internal *rng =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(INT64_MAX);
  CHECK(rng != NULL);
  et_g3c4_rng_internal forged = *rng;
  CHECK(et_g3c4_private_rng_clone_v1(&forged) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  rng->transport.magic ^= UINT64_C(1);
  CHECK(et_g3c4_private_rng_clone_v1(rng) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  rng->transport.magic ^= UINT64_C(1);
  rng->transport.busy = 1u;
  CHECK(et_g3c4_private_rng_clone_v1(rng) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  rng->transport.busy = 0u;
  rng->transport.state = 99u;
  CHECK(et_g3c4_private_rng_clone_v1(rng) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  rng->transport.state = ET_G3C4_CONTEXT_LIVE;
  CHECK(et_g3c4_private_rng_word_v1(rng, 0) == 1);
  CHECK(et_g3c4_private_rng_word_v1(rng, 1) == INT64_MAX);
  CHECK(et_g3c4_private_rng_word_v1(rng, 2) == 0);
  CHECK(et_g3c4_private_rng_word_v1(rng, 3) == 0);
  CHECK(et_g3c4_private_rng_word_v1(rng, -1) == 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_SELECTOR);
  CHECK(et_g3c4_private_rng_word_v1(rng, 4) == 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_SELECTOR);

  rng->words[2] = INT64_MIN;
  rng->words[3] = INT64_MAX;
  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_rng_clone_v1(rng) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_rng_internal *clone =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_clone_v1(rng);
  CHECK(clone != NULL && clone != rng);
  CHECK(memcmp(clone->words, rng->words, sizeof(rng->words)) == 0);
  rng->words[2] = 7;
  CHECK(clone->words[2] == INT64_MIN);
  rng->words[2] = INT64_MIN;
  OK(et_g3c4_private_rng_release_v1(rng));
  OK(et_g3c4_private_rng_release_v1(rng));
  CHECK(et_g3c4_private_rng_word_v1(rng, 0) == 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_rng_clone_v1(rng) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(clone->words[2] == INT64_MIN && clone->words[3] == INT64_MAX);
  OK(et_g3c4_private_rng_release_v1(clone));
  clone->words[0] = 1;
  CHECK(et_g3c4_private_rng_release_v1(clone) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  clone->words[0] = 0;
  OK(et_g3c4_private_rng_release_v1(clone));
}

static void policy_and_construction(
    et_g3c4_model_owner_internal *owner) {
  const int64_t invalid[][6] = {
    {-1, INT64_C(0x3f800000), 256, INT64_C(0x3f800000), 1, -1},
    {2, INT64_C(0x3f800000), 256, INT64_C(0x3f800000), 1, -1},
    {0, 0, 256, INT64_C(0x3f800000), 1, -1},
    {0, INT64_C(0x7f800000), 256, INT64_C(0x3f800000), 1, -1},
    {0, INT64_C(0x3f800000), 255, INT64_C(0x3f800000), 1, -1},
    {1, INT64_C(0x3f800000), 0, INT64_C(0x3f800000), 1, -1},
    {1, INT64_C(0x3f800000), 257, INT64_C(0x3f800000), 1, -1},
    {1, INT64_C(0x3f800000), 1, 0, 1, -1},
    {1, INT64_C(0x3f800000), 1, INT64_C(0x3f800001), 1, -1},
    {1, INT64_C(0x3f800000), 1, INT64_C(0x3f000000), -1, -1},
    {1, INT64_C(0x3f800000), 1, INT64_C(0x3f000000), 2, -1},
    {1, INT64_C(0x3f800000), 1, INT64_C(0x3f000000), 1, -2},
    {1, INT64_C(0x3f800000), 1, INT64_C(0x3f000000), 1, 256}
  };
  for (size_t index = 0u; index < sizeof(invalid) / sizeof(invalid[0]); index++) {
    CHECK(et_g3c4_private_generator_seed_v1(
              owner, 0, invalid[index][0], invalid[index][1],
              invalid[index][2], invalid[index][3], invalid[index][4],
              invalid[index][5]) == NULL);
    expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
                 ET_G3C4_CODE_SELECTOR);
  }
  CHECK(et_g3c4_private_generator_seed_v1(
            owner, -1, 0, INT64_C(0x3f800000), 256,
            INT64_C(0x3f800000), 1, -1) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_SELECTOR);

  const size_t contexts = transport_count(ET_G3C4_CONTEXT_KIND);
  const size_t caches = et_g3c4_generator_test_a2_cache_count();
  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_generator_seed_v1(
            owner, 0, 0, INT64_C(0x3f800000), 256,
            INT64_C(0x3f800000), 1, -1) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  CHECK(transport_count(ET_G3C4_CONTEXT_KIND) == contexts);
  CHECK(et_g3c4_generator_test_a2_cache_count() == caches);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  for (size_t allowed = 0u; allowed < 5u; allowed++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    CHECK(et_g3c4_private_generator_seed_v1(
              owner, 0, 0, INT64_C(0x3f800000), 256,
              INT64_C(0x3f800000), 1, -1) == NULL);
    expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
                 ET_KERNEL_CODE_ALLOCATION_FAILED);
    CHECK(transport_count(ET_G3C4_CONTEXT_KIND) == contexts);
    CHECK(et_g3c4_generator_test_a2_cache_count() == caches);
  }
  et_a2_kv_cache_test_reset_allocator_v1();

  et_g3c4_rng_internal *rng =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(41);
  CHECK(rng != NULL);
  rng->words[2] = INT64_MIN;
  rng->words[3] = INT64_MAX;
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)et_g3c4_private_generator_rng_v1(
          owner, rng, 1, INT64_C(0x00800000), 1,
          INT64_C(0x3f000000), 0, 255);
  CHECK(context != NULL);
  CHECK(context->generator_policy[0] == 1);
  CHECK(context->generator_policy[1] == INT64_C(0x00800000));
  CHECK(context->generator_policy[2] == 1);
  CHECK(context->generator_policy[3] == INT64_C(0x3f000000));
  CHECK(context->generator_policy[4] == 0);
  CHECK(context->generator_policy[5] == 255);
  CHECK(memcmp(context->generator_rng_words, rng->words,
               sizeof(rng->words)) == 0);
  OK(et_g3c4_private_rng_release_v1(rng));
  CHECK(context->generator_rng_words[2] == INT64_MIN);
  CHECK(context->generator_rng_words[3] == INT64_MAX);
  OK(et_g3c4_private_generator_close_v1(context));

  rng = (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(43);
  CHECK(rng != NULL);
  OK(et_g3c4_private_rng_release_v1(rng));
  CHECK(et_g3c4_private_generator_rng_v1(
            owner, rng, 0, INT64_C(0x3f800000), 256,
            INT64_C(0x3f800000), 1, -1) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
}

static void type_boundaries(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *base =
      (et_g3c4_context_internal *)et_g3c4_private_context_create_v1(owner);
  CHECK(base != NULL);
  CHECK(base->generator_kind == 0u && base->generator_ready == 0u);
  CHECK(et_g3c4_private_generator_close_v1(base) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_call_acquire_v1(base, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_rng_release_v1(base) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  OK(et_g3c4_private_context_close_v1(base));
  CHECK(et_g3c4_private_generator_close_v1(base) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);

  et_g3c4_context_internal *generator = seed_generator(owner, 3);
  CHECK(et_g3c4_private_context_close_v1(generator) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_rng_clone_v1(generator) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  OK(et_g3c4_private_generator_close_v1(generator));
  CHECK(et_g3c4_private_context_close_v1(generator) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);

  et_g3c4_rng_internal *rng =
      (et_g3c4_rng_internal *)et_g3c4_private_rng_seed_v1(7);
  CHECK(rng != NULL);
  CHECK(et_g3c4_private_generator_close_v1(rng) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  OK(et_g3c4_private_rng_release_v1(rng));
  CHECK(et_g3c4_private_generator_close_v1(rng) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
}

static void close_retry_and_scrub(et_g3c4_model_owner_internal *owner) {
  int64_t count = 1;
  float keys[4] = {0};
  float values[4] = {0};
  uint64_t count_shape[1] = {1u};
  uint64_t tensor_shape[4] = {1u, 2u, 1u, 2u};
  et_kernel_tensor_view_v1 counts = {
    sizeof(counts), &count, sizeof(count), "i64", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 1u, count_shape};
  et_kernel_tensor_view_v1 key_view = {
    sizeof(key_view), keys, sizeof(keys), "f32", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape};
  et_kernel_tensor_view_v1 value_view = {
    sizeof(value_view), values, sizeof(values), "f32", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape};
  et_kernel_error error;

  et_g3c4_context_internal *context = seed_generator(owner, 11);
  context->transport.magic ^= UINT64_C(1);
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->transport.magic ^= UINT64_C(1);
  context->generator_policy[2] = 0;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->generator_policy[2] = 256;
  context->generator_rng_words[0] = 2;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->generator_rng_words[0] = 1;
  et_a2_kv_cache_transaction *transaction = NULL;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  et_g3c4_context_internal before = *context;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(context, &before, sizeof(before)) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);

  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
            transaction, 0u, &key_view, &value_view, &error) == 0);
  et_a2_kv_cache_transaction_view *view = NULL;
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
            transaction, 0u, &view, &error) == 0);
  before = *context;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(context, &before, sizeof(before)) == 0);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(&view, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);

  et_a2_kv_cache_read_borrow *borrow = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            context->cache, &borrow, &error) == 0);
  before = *context;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(context, &before, sizeof(before)) == 0);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  before = *context;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(memcmp(context, &before, sizeof(before)) == 0);
  OK(et_g3c4_private_call_finish_v1(context));

  OK(et_g3c4_private_generator_close_v1(context));
  CHECK(context->transport.state == ET_G3C4_CONTEXT_DEAD);
  CHECK(context->generator_kind == 1u && context->generator_ready == 0u);
  CHECK(context->owner == NULL && context->cache == NULL);
  CHECK(et_g3c4_zero_i64_words(context->generator_policy, 6u));
  CHECK(et_g3c4_zero_i64_words(context->generator_rng_words, 4u));
  OK(et_g3c4_private_generator_close_v1(context));
  context->generator_rng_words[3] = 1;
  CHECK(et_g3c4_private_generator_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->generator_rng_words[3] = 0;
  OK(et_g3c4_private_generator_close_v1(context));
}

static void retention(et_g3c4_model_owner_internal *owner) {
  size_t baseline = transport_count(ET_G3C4_CONTEXT_KIND);
  size_t count_1024 = 0u;
  for (size_t cycle = 1u; cycle <= 8192u; cycle++) {
    et_g3c4_context_internal *context = seed_generator(owner, (int64_t)cycle);
    OK(et_g3c4_private_generator_close_v1(context));
    CHECK(et_g3c4_generator_test_a2_cache_count() == 0u);
    if (cycle == 1024u)
      count_1024 = transport_count(ET_G3C4_CONTEXT_KIND) - baseline;
  }
  size_t count_8192 = transport_count(ET_G3C4_CONTEXT_KIND) - baseline;
  CHECK(count_1024 == 1024u && count_8192 == 8192u);
  CHECK(count_1024 * sizeof(et_g3c4_context_internal) == 1556480u);
  CHECK(count_8192 * sizeof(et_g3c4_context_internal) == 12451840u);

  baseline = transport_count(ET_G3C4_RNG_KIND);
  count_1024 = 0u;
  for (size_t cycle = 1u; cycle <= 8192u; cycle++) {
    void *rng = et_g3c4_private_rng_seed_v1((int64_t)cycle);
    CHECK(rng != NULL);
    OK(et_g3c4_private_rng_release_v1(rng));
    if (cycle == 1024u)
      count_1024 = transport_count(ET_G3C4_RNG_KIND) - baseline;
  }
  count_8192 = transport_count(ET_G3C4_RNG_KIND) - baseline;
  CHECK(count_1024 == 1024u && count_8192 == 8192u);
  CHECK(count_1024 * sizeof(et_g3c4_rng_internal) == 65536u);
  CHECK(count_8192 * sizeof(et_g3c4_rng_internal) == 524288u);

  void *source = et_g3c4_private_rng_seed_v1(99);
  CHECK(source != NULL);
  baseline = transport_count(ET_G3C4_RNG_KIND);
  count_1024 = 0u;
  for (size_t cycle = 1u; cycle <= 8192u; cycle++) {
    void *clone = et_g3c4_private_rng_clone_v1(source);
    CHECK(clone != NULL);
    OK(et_g3c4_private_rng_release_v1(clone));
    if (cycle == 1024u)
      count_1024 = transport_count(ET_G3C4_RNG_KIND) - baseline;
  }
  count_8192 = transport_count(ET_G3C4_RNG_KIND) - baseline;
  CHECK(count_1024 == 1024u && count_8192 == 8192u);
  CHECK(count_1024 * sizeof(et_g3c4_rng_internal) == 65536u);
  CHECK(count_8192 * sizeof(et_g3c4_rng_internal) == 524288u);
  OK(et_g3c4_private_rng_release_v1(source));

  printf("G3-C4 generator retention: contexts_1024=1024 bytes_1024=1556480 "
         "contexts_8192=8192 bytes_8192=12451840 rng_1024=1024 "
         "rng_bytes_1024=65536 rng_8192=8192 rng_bytes_8192=524288\n");
}

int main(void) {
  layout_and_macro_surface();
  rng_lifetime();
  et_g3c4_model_owner_internal *owner = create_owner();
  policy_and_construction(owner);
  type_boundaries(owner);
  close_retry_and_scrub(owner);
  retention(owner);
  CHECK(et_g3c4_generator_test_a2_cache_count() == 0u);
  printf("G3-C4 native generator PASS: checks=%zu context_bytes=%zu "
         "rng_bytes=%zu\n", checks, sizeof(et_g3c4_context_internal),
         sizeof(et_g3c4_rng_internal));
  return 0;
}
#endif
