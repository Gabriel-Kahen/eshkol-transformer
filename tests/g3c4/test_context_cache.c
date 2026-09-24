#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1

#ifdef ET_G3C4_A2_INTRUSIVE_TU
#include "../../native/a2_kv_cache.c"

size_t et_g3c4_test_a2_cache_registry_count(void) {
  size_t count = 0u;
  for (et_a2_kv_cache *cache = live_caches;
       cache != NULL; cache = cache->registry_next)
    count++;
  return count;
}

int et_g3c4_test_a2_cache_is_live(const et_a2_kv_cache *candidate) {
  return find_cache(candidate) == candidate;
}
#else
#include "../../src/eshkol_transformer/m3t_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <stdio.h>

size_t et_g3c4_test_a2_cache_registry_count(void);
int et_g3c4_test_a2_cache_is_live(const et_a2_kv_cache *candidate);

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

static unsigned char identities[4][14];

static size_t context_registry_count(void) {
  size_t count = 0u;
  for (et_g3c4_context_internal *context = et_g3c4_context_registry;
       context != NULL; context = context->registry_next)
    count++;
  return count;
}

static et_g3c4_model_owner_internal *create_owner(size_t row, int64_t seed) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[row][(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static void expect_error(int64_t domain, int64_t category, int64_t code) {
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
}

static void owner_bytes(
    et_g3c4_model_owner_internal *owner,
    uint32_t values[1192], uint32_t gradients[1192]) {
  size_t offset = 0u;
  for (size_t index = 0u; index < 14u; index++) {
    size_t count = et_g3c4_parameter_elements(index);
    memcpy(values + offset, owner->parameters[index]->value->data,
           count * sizeof(*values));
    memcpy(gradients + offset, owner->parameters[index]->gradient->data,
           count * sizeof(*gradients));
    offset += count;
  }
  CHECK(offset == 1192u);
}

static void owner_unchanged(
    et_g3c4_model_owner_internal *owner,
    const et_g3c4_model_owner_internal *record,
    const uint32_t values[1192], const uint32_t gradients[1192]) {
  uint32_t actual_values[1192];
  uint32_t actual_gradients[1192];
  CHECK(memcmp(owner, record, sizeof(*owner)) == 0);
  owner_bytes(owner, actual_values, actual_gradients);
  CHECK(memcmp(values, actual_values, sizeof(actual_values)) == 0);
  CHECK(memcmp(gradients, actual_gradients, sizeof(actual_gradients)) == 0);
}

static void construction_failures(et_g3c4_model_owner_internal *owner) {
  const size_t baseline = context_registry_count();
  const size_t cache_baseline = et_g3c4_test_a2_cache_registry_count();
  const et_g3c4_model_owner_internal owner_record = *owner;
  uint32_t values[1192];
  uint32_t gradients[1192];
  owner_bytes(owner, values, gradients);

  et_g3c4_context_allocation_limit =
      et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  CHECK(context_registry_count() == baseline);
  CHECK(et_g3c4_test_a2_cache_registry_count() == cache_baseline);
  owner_unchanged(owner, &owner_record, values, gradients);
  et_g3c4_context_allocation_limit = SIZE_MAX;

  for (size_t allowed = 0u; allowed < 5u; allowed++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
    expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
                 ET_KERNEL_CODE_ALLOCATION_FAILED);
    CHECK(context_registry_count() == baseline);
    CHECK(et_g3c4_test_a2_cache_registry_count() == cache_baseline);
    owner_unchanged(owner, &owner_record, values, gradients);
  }
  et_a2_kv_cache_test_reset_allocator_v1();
}

static et_g3c4_context_internal *create_context(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)et_g3c4_private_context_create_v1(owner);
  CHECK(context != NULL);
  CHECK(context->magic == ET_G3C4_CONTEXT_MAGIC);
  CHECK(context->kind == ET_G3C4_CONTEXT_KIND);
  CHECK(context->state == ET_G3C4_CONTEXT_LIVE);
  CHECK(context->busy == 0u);
  CHECK(context->owner == owner);
  CHECK(context->cache != NULL);
  CHECK(et_g3c4_test_a2_cache_is_live(context->cache));
  return context;
}

static void cache_geometry_and_multiplicity(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *first = create_context(owner);
  et_g3c4_context_internal *second = create_context(owner);
  CHECK(first != second);
  CHECK(first->cache != second->cache);

  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            first->cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
            borrow, 0u, &keys, &values, &lengths, &mask, &error) == 0);
  CHECK(keys != NULL && values != NULL && lengths != NULL && mask != NULL);
  CHECK(keys->rank == 4u && values->rank == 4u);
  CHECK(keys->shape[0] == 1u && keys->shape[1] == 2u &&
        keys->shape[2] == 4u && keys->shape[3] == 2u);
  CHECK(values->shape[0] == 1u && values->shape[1] == 2u &&
        values->shape[2] == 4u && values->shape[3] == 2u);
  CHECK(lengths->rank == 1u && lengths->shape[0] == 1u);
  CHECK(mask->rank == 2u && mask->shape[0] == 1u && mask->shape[1] == 4u);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_context_close_v1(first));
  OK(et_g3c4_private_context_close_v1(second));
  CHECK(et_g3c4_test_a2_cache_registry_count() == 0u);
}

static void transaction_busy(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_context(owner);
  int64_t count = 1;
  uint64_t shape[1] = {1u};
  et_kernel_tensor_view_v1 counts = {
    sizeof(counts), &count, sizeof(count), "i64", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 1u, shape
  };
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(context->state == ET_G3C4_CONTEXT_LIVE);
  CHECK(context->cache != NULL);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);
  OK(et_g3c4_private_context_close_v1(context));
}

static void view_busy(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_context(owner);
  int64_t count = 1;
  float keys[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  uint64_t count_shape[1] = {1u};
  uint64_t tensor_shape[4] = {1u, 2u, 1u, 2u};
  et_kernel_tensor_view_v1 counts = {
    sizeof(counts), &count, sizeof(count), "i64", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 1u, count_shape
  };
  et_kernel_tensor_view_v1 key_view = {
    sizeof(key_view), keys, sizeof(keys), "f32", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape
  };
  et_kernel_tensor_view_v1 value_view = {
    sizeof(value_view), values, sizeof(values), "f32", "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape
  };
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *view = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
            context->cache, 1u, &counts, &transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
            transaction, 0u, &key_view, &value_view, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
            transaction, 0u, &view, &error) == 0);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(&view, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error) == 0);
  OK(et_g3c4_private_context_close_v1(context));
}

static void borrow_busy(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = create_context(owner);
  et_a2_kv_cache_read_borrow *borrow = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            context->cache, &borrow, &error) == 0);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_context_close_v1(context));
}

static void admission_and_close(et_g3c4_model_owner_internal *owner) {
  int foreign = 0;
  CHECK(et_g3c4_private_context_create_v1(&foreign) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_context_close_v1(&foreign) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT,
               ET_G3C4_CODE_IDENTITY);

  et_g3c4_context_internal *context = create_context(owner);
  et_a2_kv_cache *cache = context->cache;
  context->busy = 1u;
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  context->busy = 0u;
  owner->active = context;
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  owner->active = NULL;
  OK(et_g3c4_private_context_close_v1(context));
  CHECK(context->state == ET_G3C4_CONTEXT_DEAD);
  CHECK(context->owner == NULL);
  CHECK(context->cache == NULL);
  OK(et_g3c4_private_context_close_v1(context));
  et_a2_kv_cache *stale = cache;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_destroy_v1(&stale, &error) != 0);
  CHECK(stale == cache);

  context->magic ^= UINT64_C(1);
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->magic ^= UINT64_C(1);
  context->kind = 0u;
  CHECK(et_g3c4_private_context_close_v1(context) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  context->kind = ET_G3C4_CONTEXT_KIND;
  OK(et_g3c4_private_context_close_v1(context));
}

static void wrong_lifecycle(void) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(11);
  CHECK(owner != NULL);
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_model_owner_abort_v1(owner));
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);

  owner = et_g3c4_private_model_owner_create_seeded_v1(13);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[1][(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  CHECK(et_g3c4_private_context_create_v1(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
               ET_G3C4_CODE_LIFECYCLE);
  OK(et_g3c4_private_model_owner_abort_v1(owner));
}

static void retention(et_g3c4_model_owner_internal *owner) {
  const size_t baseline = context_registry_count();
  size_t count_1024 = 0u;
  size_t bytes_1024 = 0u;
  for (size_t cycle = 1u; cycle <= 8192u; cycle++) {
    et_g3c4_context_internal *context = create_context(owner);
    OK(et_g3c4_private_context_close_v1(context));
    CHECK(et_g3c4_test_a2_cache_registry_count() == 0u);
    if (cycle == 1024u) {
      count_1024 = context_registry_count() - baseline;
      bytes_1024 = count_1024 * sizeof(*context);
    }
  }
  const size_t count_8192 = context_registry_count() - baseline;
  const size_t bytes_8192 = count_8192 * sizeof(et_g3c4_context_internal);
  CHECK(count_1024 == 1024u);
  CHECK(count_8192 == 8192u);
  CHECK(bytes_1024 == 1024u * sizeof(et_g3c4_context_internal));
  CHECK(bytes_8192 == 8192u * sizeof(et_g3c4_context_internal));
  printf("G3-C4 context retention: registry_1024=%zu bytes_1024=%zu "
         "registry_8192=%zu bytes_8192=%zu\n",
         count_1024, bytes_1024, count_8192, bytes_8192);
}

int main(void) {
  wrong_lifecycle();
  et_g3c4_model_owner_internal *owner = create_owner(0u, 1729);
  const et_g3c4_model_owner_internal owner_record = *owner;
  uint32_t values[1192];
  uint32_t gradients[1192];
  owner_bytes(owner, values, gradients);
  construction_failures(owner);
  cache_geometry_and_multiplicity(owner);
  admission_and_close(owner);
  transaction_busy(owner);
  view_busy(owner);
  borrow_busy(owner);
  retention(owner);
  owner_unchanged(owner, &owner_record, values, gradients);
  printf("G3-C4 context cache PASS: checks=%zu record_bytes=%zu\n",
         checks, sizeof(et_g3c4_context_internal));
  return 0;
}
#endif
