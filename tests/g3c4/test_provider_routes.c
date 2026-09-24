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

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

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

static unsigned char identities[14];

static void expect_error(int64_t domain, int64_t category, int64_t code) {
  CHECK(et_g3c4_private_last_error_domain_v1() == domain);
  CHECK(et_g3c4_private_last_error_category_v1() == category);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
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

static void *seed_generator(et_g3c4_model_owner_internal *owner) {
  return et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1);
}

static void exact_route_table(void) {
  CHECK(sizeof(et_g3c4_forward_routes) /
            sizeof(et_g3c4_forward_routes[0]) == 12u);
  CHECK(sizeof(et_g3c4_sampler_routes) /
            sizeof(et_g3c4_sampler_routes[0]) == 2u);
  CHECK(et_g3c4_forward_routes[0].rank == 4u);
  CHECK(et_g3c4_forward_routes[0].shape[1] == 4u);
  CHECK(et_g3c4_forward_routes[1].shape[2] == 256u);
  CHECK(et_g3c4_forward_routes[11].rank == 6u);
  CHECK(et_g3c4_forward_routes[11].shape[3] == 4u);
  CHECK(et_g3c4_forward_routes[11].shape[4] == 4u);
  CHECK(et_g3c4_sampler_routes[0].shape[1] == 256u);
  CHECK(et_g3c4_sampler_routes[1].shape[1] == 256u);
}

static void staged_rejection_is_atomic(void) {
  et_kernel_provider_v1 rejected = *et_g3s_kernel_provider_v1();
  et_kernel_runtime *forward = NULL;
  et_kernel_runtime *sampler = NULL;
  rejected.capability_count = 1u;
  CHECK(et_g3c4_stage_provider_routes(
            et_g3c4_kernel_provider_v1(), &rejected,
            &forward, &sampler) != 0);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_INVARIANT);
  CHECK(forward == NULL && sampler == NULL);
  CHECK(et_g3c4_forward_runtime == NULL);
  CHECK(et_g3c4_sampler_runtime == NULL);
}

static void construction_atomicity(et_g3c4_model_owner_internal *owner) {
  et_m3t_test_k1_fail_after(0u);
  CHECK(seed_generator(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
               ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(et_g3c4_forward_runtime == NULL);
  CHECK(et_g3c4_sampler_runtime == NULL);
  et_m3t_test_k1_allocator_reset();

  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(seed_generator(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  CHECK(et_g3c4_forward_runtime == NULL);
  CHECK(et_g3c4_sampler_runtime == NULL);
  et_g3c4_context_allocation_limit = SIZE_MAX;

  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(seed_generator(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL,
               ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(et_g3c4_forward_runtime == NULL);
  CHECK(et_g3c4_sampler_runtime == NULL);
  et_a2_kv_cache_test_reset_allocator_v1();
}

static void accepted_routes_are_reused(
    et_g3c4_model_owner_internal *owner) {
  void *first = seed_generator(owner);
  et_kernel_runtime *forward = et_g3c4_forward_runtime;
  et_kernel_runtime *sampler = et_g3c4_sampler_runtime;
  CHECK(first != NULL);
  CHECK(forward != NULL && sampler != NULL && forward != sampler);
  OK(et_g3c4_private_generator_close_v1(first));

  void *second = seed_generator(owner);
  CHECK(second != NULL);
  CHECK(et_g3c4_forward_runtime == forward);
  CHECK(et_g3c4_sampler_runtime == sampler);
  OK(et_g3c4_private_generator_close_v1(second));

  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(seed_generator(owner) == NULL);
  expect_error(ET_G3C4_DOMAIN_C4, ET_G3C4_INTERNAL,
               ET_G3C4_CODE_ALLOCATION);
  CHECK(et_g3c4_forward_runtime == forward);
  CHECK(et_g3c4_sampler_runtime == sampler);
  et_g3c4_context_allocation_limit = SIZE_MAX;
}

int main(void) {
  exact_route_table();
  staged_rejection_is_atomic();
  et_g3c4_model_owner_internal *owner = create_owner();
  construction_atomicity(owner);
  accepted_routes_are_reused(owner);
  printf("G3-C4 private provider routes PASS: checks=%zu forward=12 sampler=2\n",
         checks);
  return 0;
}
