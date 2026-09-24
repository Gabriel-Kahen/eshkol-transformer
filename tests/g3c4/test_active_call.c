#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1

#ifdef ET_G3C4_ACTIVE_CALL_A2_TU
#include "../../native/a2_kv_cache.c"

size_t et_g3c4_test_a2_cache_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache *item = live_caches;
       item != NULL; item = item->registry_next) count++;
  return count;
}
size_t et_g3c4_test_a2_transaction_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache_transaction *item = live_transactions;
       item != NULL; item = item->registry_next) count++;
  return count;
}
size_t et_g3c4_test_a2_view_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache_transaction_view *item = live_views;
       item != NULL; item = item->registry_next) count++;
  return count;
}
size_t et_g3c4_test_a2_borrow_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache_read_borrow *item = live_borrows;
       item != NULL; item = item->registry_next) count++;
  return count;
}
#else
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <stdio.h>

size_t et_g3c4_test_a2_cache_count(void);
size_t et_g3c4_test_a2_transaction_count(void);
size_t et_g3c4_test_a2_view_count(void);
size_t et_g3c4_test_a2_borrow_count(void);

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

typedef struct cache_image {
  float keys[16];
  float values[16];
  int64_t length;
  uint8_t mask[4];
} cache_image;

typedef struct owner_image {
  et_g3c4_model_owner_internal record;
  uint32_t values[1192];
  uint32_t gradients[1192];
} owner_image;

static void expect_error(int64_t domain, int64_t category, int64_t code) {
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
}

static size_t context_count(void) {
  size_t count = 0u;
  for (et_g3c4_context_internal *item = et_g3c4_context_registry;
       item != NULL; item = item->registry_next) count++;
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

static et_g3c4_context_internal *create_context(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)et_g3c4_private_context_create_v1(owner);
  CHECK(context != NULL);
  CHECK(context->state == ET_G3C4_CONTEXT_LIVE);
  CHECK(context->busy == ET_G3C4_CALL_IDLE);
  CHECK(context->acquired_mask == 0u);
  CHECK(context->call_kind == 0 && context->budget == 0);
  CHECK(et_g3c4_pins_idle(&context->pins));
  return context;
}

static void owner_read(
    et_g3c4_model_owner_internal *owner, owner_image *image) {
  size_t offset = 0u;
  image->record = *owner;
  for (size_t index = 0u; index < 14u; index++) {
    size_t count = et_g3c4_parameter_elements(index);
    memcpy(image->values + offset, owner->parameters[index]->value->data,
           count * sizeof(*image->values));
    memcpy(image->gradients + offset, owner->parameters[index]->gradient->data,
           count * sizeof(*image->gradients));
    offset += count;
  }
  CHECK(offset == 1192u);
}

static void check_owner_unchanged(
    et_g3c4_model_owner_internal *owner, const owner_image *expected) {
  owner_image actual;
  owner_read(owner, &actual);
  CHECK(memcmp(&actual.record, &expected->record,
               sizeof(actual.record)) == 0);
  CHECK(memcmp(actual.values, expected->values,
               sizeof(actual.values)) == 0);
  CHECK(memcmp(actual.gradients, expected->gradients,
               sizeof(actual.gradients)) == 0);
}

static void check_idle(
    const et_g3c4_context_internal *context,
    const et_g3c4_model_owner_internal *owner) {
  CHECK(context->busy == ET_G3C4_CALL_IDLE);
  CHECK(context->acquired_mask == 0u);
  CHECK(context->call_kind == 0 && context->budget == 0);
  CHECK(et_g3c4_pins_idle(&context->pins));
  CHECK(owner->active == NULL);
  for (size_t index = 0u; index < 14u; index++) {
    CHECK(owner->parameters[index]->plan_pins == 0);
    CHECK(owner->parameters[index]->value->active_borrow == NULL);
    CHECK(owner->parameters[index]->gradient->active_borrow == NULL);
    CHECK(owner->parameters[index]->value->plan_pins == 0);
    CHECK(owner->parameters[index]->gradient->plan_pins == 0);
  }
}

static void check_active(
    const et_g3c4_context_internal *context,
    const et_g3c4_model_owner_internal *owner,
    int64_t call_kind, int64_t budget) {
  CHECK(context->busy == ET_G3C4_CALL_ACTIVE);
  CHECK(context->acquired_mask == ET_G3C4_ACQUIRED_FULL);
  CHECK(context->call_kind == call_kind && context->budget == budget);
  CHECK(context->pins.self == &context->pins);
  CHECK(context->pins.held_mask == UINT16_C(0x3fff));
  CHECK(owner->active == context);
}

static void cache_read(et_a2_kv_cache *cache, cache_image *image) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  et_kernel_error error;
  memset(image, 0, sizeof(*image));
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
            borrow, 0u, &keys, &values, &lengths, &mask, &error) == 0);
  memcpy(image->keys, keys->data, sizeof(image->keys));
  memcpy(image->values, values->data, sizeof(image->values));
  memcpy(&image->length, lengths->data, sizeof(image->length));
  memcpy(image->mask, mask->data, sizeof(image->mask));
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
}

static void check_cache_unchanged(
    et_a2_kv_cache *cache, const cache_image *expected) {
  cache_image actual;
  cache_read(cache, &actual);
  CHECK(memcmp(&actual, expected, sizeof(actual)) == 0);
}

static void accepted_tuples(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_context_internal *context) {
  const int64_t tuples[4][2] = {{0,0}, {1,0}, {2,0}, {2,1}};
  for (size_t index = 0u; index < 4u; index++) {
    OK(et_g3c4_private_call_acquire_v1(
        context, tuples[index][0], tuples[index][1]));
    check_active(context, owner, tuples[index][0], tuples[index][1]);
    const et_g3c4_context_internal before = *context;
    OK(et_g3c4_private_call_prepare_end_v1(context));
    CHECK(memcmp(context, &before, sizeof(*context)) == 0);
    OK(index % 2u == 0u
        ? et_g3c4_private_call_finish_v1(context)
        : et_g3c4_private_call_abort_v1(context));
    check_idle(context, owner);
  }
}

static void rejected_tuples(et_g3c4_context_internal *context) {
  const int64_t tuples[][2] = {
    {-1,0}, {0,-1}, {0,1}, {1,1}, {2,-1}, {2,2}, {3,0},
    {INT64_MIN,INT64_MAX}, {INT64_MAX,INT64_MIN}
  };
  for (size_t index = 0u; index < sizeof(tuples) / sizeof(tuples[0]); index++) {
    CHECK(et_g3c4_private_call_acquire_v1(
              context, tuples[index][0], tuples[index][1]) != 0);
    expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
                 ET_G3C4_CODE_SELECTOR);
    check_idle(context, context->owner);
  }
}

static void a2_allocation_failures(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_context_internal *context) {
  cache_image image;
  cache_read(context->cache, &image);
  for (size_t allowed = 0u; allowed < 3u; allowed++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    CHECK(et_g3c4_private_call_acquire_v1(context, 2, 1) != 0);
    expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
                 ET_KERNEL_CODE_ALLOCATION_FAILED);
    check_idle(context, owner);
    CHECK(et_g3c4_test_a2_cache_count() == 1u);
    CHECK(et_g3c4_test_a2_transaction_count() == 0u);
    CHECK(et_g3c4_test_a2_view_count() == 0u);
    CHECK(et_g3c4_test_a2_borrow_count() == 0u);
  }
  et_a2_kv_cache_test_reset_allocator_v1();
  for (size_t allowed = 0u; allowed < 3u; allowed++) {
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
    expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
                 ET_KERNEL_CODE_ALLOCATION_FAILED);
    check_active(context, owner, 2, 1);
    CHECK(et_g3c4_test_a2_cache_count() == 1u);
    CHECK(et_g3c4_test_a2_transaction_count() == 0u);
    CHECK(et_g3c4_test_a2_view_count() == 0u);
    CHECK(et_g3c4_test_a2_borrow_count() == 0u);
    et_a2_kv_cache_test_reset_allocator_v1();
    OK(et_g3c4_private_call_abort_v1(context));
  }
  for (size_t allowed = 0u; allowed < 3u; allowed++) {
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    CHECK(et_g3c4_private_call_abort_v1(context) != 0);
    expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
                 ET_KERNEL_CODE_ALLOCATION_FAILED);
    check_active(context, owner, 2, 1);
    CHECK(et_g3c4_test_a2_cache_count() == 1u);
    CHECK(et_g3c4_test_a2_transaction_count() == 0u);
    CHECK(et_g3c4_test_a2_view_count() == 0u);
    CHECK(et_g3c4_test_a2_borrow_count() == 0u);
    et_a2_kv_cache_test_reset_allocator_v1();
    OK(et_g3c4_private_call_abort_v1(context));
  }
  check_cache_unchanged(context->cache, &image);
}

static void pin_and_publication_failures(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_context_internal *context) {
  for (size_t prefix = 0u; prefix < 14u; prefix++) {
    m3_call_test_fail_after = prefix;
    CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) != 0);
    expect_error(ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_ERROR_INTERNAL,
                 ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    check_idle(context, owner);
    CHECK(m3_call_test_release_count == prefix);
    for (size_t index = 0u; index < prefix; index++)
      CHECK(m3_call_test_release_order[index] == prefix - index - 1u);
  }
  m3_call_test_fail_after = SIZE_MAX;
  for (size_t completed = 1u; completed <= 4u; completed++) {
    et_g3c4_active_call_fail_after = completed;
    CHECK(et_g3c4_private_call_acquire_v1(context, 1, 0) != 0);
    expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
                 ET_G3C4_CODE_INVARIANT);
    check_idle(context, owner);
    CHECK(m3_call_test_release_count == 14u);
    for (size_t index = 0u; index < 14u; index++)
      CHECK(m3_call_test_release_order[index] == 13u - index);
  }
  et_g3c4_active_call_fail_after = SIZE_MAX;
}

static void nested_leases(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_context_internal *context) {
  int64_t count = 1;
  float keys[4] = {0};
  float values[4] = {0};
  uint64_t count_shape[1] = {1u};
  uint64_t tensor_shape[4] = {1u,2u,1u,2u};
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

  et_a2_kv_cache_transaction *transaction = NULL;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_g3c4_private_call_acquire_v1(context, 2, 1) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  check_idle(context, owner);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);

  et_a2_kv_cache_read_borrow *borrow = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            context->cache, &borrow, &error) == 0);
  CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  check_idle(context, owner);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  check_active(context, owner, 2, 1);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
            transaction, 0u, &key_view, &value_view, &error) == 0);
  et_a2_kv_cache_transaction_view *view = NULL;
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
            transaction, 0u, &view, &error) == 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(&view, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);

  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  borrow = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            context->cache, &borrow, &error) == 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_test_a2_transaction_count() == 0u);
  CHECK(et_g3c4_test_a2_view_count() == 0u);
  CHECK(et_g3c4_test_a2_borrow_count() == 0u);
}

static void identity_lifecycle_and_corruption(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_context_internal *context) {
  int foreign = 0;
  CHECK(et_g3c4_private_call_acquire_v1(&foreign, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_call_prepare_end_v1(&foreign) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);

  et_g3c4_context_internal *second = create_context(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_call_acquire_v1(second, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_call_finish_v1(context));
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);

  owner->state = ET_G3C4_OWNER_OPEN;
  CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  owner->state = ET_G3C4_OWNER_SEALED;

  context->acquired_mask = ET_G3C4_ACQUIRED_PINS;
  CHECK(et_g3c4_private_call_acquire_v1(context, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->acquired_mask = 0u;
  context->pins.held_mask = UINT16_C(1);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->pins.held_mask = 0u;

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  context->acquired_mask ^= ET_G3C4_ACQUIRED_METADATA;
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->acquired_mask ^= ET_G3C4_ACQUIRED_METADATA;
  owner->active = NULL;
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  owner->active = context;
  context->pins.held_mask ^= UINT16_C(1);
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->pins.held_mask ^= UINT16_C(1);
  context->call_kind = 7;
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->call_kind = 2;
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_context_close_v1(second));

  et_g3c4_context_internal *dead = create_context(owner);
  OK(et_g3c4_private_context_close_v1(dead));
  dead->call_kind = 1;
  CHECK(et_g3c4_private_context_close_v1(dead) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  dead->call_kind = 0;
  OK(et_g3c4_private_context_close_v1(dead));
  CHECK(et_g3c4_private_call_acquire_v1(dead, 0, 0) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_call_prepare_end_v1(dead) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_call_finish_v1(dead) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_call_abort_v1(dead) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
}

static void retention(et_g3c4_model_owner_internal *owner) {
  const size_t baseline = context_count();
  size_t count_1024 = 0u;
  for (size_t cycle = 1u; cycle <= 8192u; cycle++) {
    et_g3c4_context_internal *context = create_context(owner);
    OK(et_g3c4_private_context_close_v1(context));
    CHECK(et_g3c4_test_a2_cache_count() == 1u);
    if (cycle == 1024u) count_1024 = context_count() - baseline;
  }
  const size_t count_8192 = context_count() - baseline;
  CHECK(sizeof(et_g3c4_context_internal) == 1432u);
  CHECK(count_1024 == 1024u && count_8192 == 8192u);
  printf("G3-C4 active retention: registry_1024=%zu bytes_1024=%zu "
         "registry_8192=%zu bytes_8192=%zu\n",
         count_1024, count_1024 * sizeof(et_g3c4_context_internal),
         count_8192, count_8192 * sizeof(et_g3c4_context_internal));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context = create_context(owner);
  owner_image owner_original;
  cache_image original;
  owner_read(owner, &owner_original);
  cache_read(context->cache, &original);
  accepted_tuples(owner, context);
  check_owner_unchanged(owner, &owner_original);
  rejected_tuples(context);
  check_owner_unchanged(owner, &owner_original);
  a2_allocation_failures(owner, context);
  check_owner_unchanged(owner, &owner_original);
  pin_and_publication_failures(owner, context);
  check_owner_unchanged(owner, &owner_original);
  nested_leases(owner, context);
  check_owner_unchanged(owner, &owner_original);
  identity_lifecycle_and_corruption(owner, context);
  check_owner_unchanged(owner, &owner_original);
  check_cache_unchanged(context->cache, &original);
  retention(owner);
  check_owner_unchanged(owner, &owner_original);
  OK(et_g3c4_private_context_close_v1(context));
  CHECK(et_g3c4_test_a2_cache_count() == 0u);
  printf("G3-C4 active call PASS: checks=%zu record_bytes=%zu\n",
         checks, sizeof(et_g3c4_context_internal));
  return 0;
}
#endif
