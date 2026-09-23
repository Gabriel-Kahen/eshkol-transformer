#define ET_F32_TENSOR_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1

#include "../../src/eshkol_transformer/m3t_f32_integration.c"
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

static unsigned char identities[8][14];

static et_f32_test_live_counts_v1 live_counts(void) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  return counts;
}

static void same_live(et_f32_test_live_counts_v1 expected) {
  et_f32_test_live_counts_v1 actual = live_counts();
  CHECK(memcmp(&actual, &expected, sizeof(actual)) == 0);
}

static size_t owner_registry_count(void) {
  size_t count = 0u;
  for (et_g3c4_model_owner_internal *owner = et_g3c4_owner_registry;
       owner != NULL; owner = owner->registry_next)
    count++;
  return count;
}

static et_g3c4_model_owner_internal *create_bound(
    int row, int64_t seed) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[row][index]));
  return owner;
}

static et_g3c4_model_owner_internal *create_initialized(
    int row, int64_t seed) {
  et_g3c4_model_owner_internal *owner = create_bound(row, seed);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  return owner;
}

static void dispatch_expected(
    size_t index, int64_t state[4], uint32_t *bits) {
  static const uint64_t state_shape[1] = {4u};
  int64_t next[4] = {0, 0, 0, 0};
  const size_t count = et_g3c4_parameter_elements(index);
  et_kernel_tensor_view_v1 input = et_g3c4_view(
      state, 4u * sizeof(int64_t), "i64", 1u, state_shape);
  et_kernel_tensor_view_v1 outputs[2] = {
    et_g3c4_view(bits, count * sizeof(uint32_t), "f32", 2u,
                 et_g3c4_parameter_shapes[index]),
    et_g3c4_view(next, sizeof(next), "i64", 1u, state_shape)};
  et_kernel_request_v1 request = {
    sizeof(request), "n3k.matrix-init.uniform", "f32", "cpu", 2u,
    et_g3c4_parameter_shapes[index], 1u, {0}};
  et_kernel_call_v1 call = {
    sizeof(call), "n3k.matrix-init", &request,
    1u, sizeof(input), sizeof(input), &input,
    2u, sizeof(outputs[0]), sizeof(outputs), outputs};
  et_kernel_error error;
  OK(et_kernel_runtime_dispatch(
      et_g3c4_initializer_runtime, &call, &error));
  memcpy(state, next, sizeof(next));
}

static void test_error_snapshot(void) {
  const et_g3c4_error_state_internal valid[] = {
    {ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY},
    {ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
     ET_KERNEL_CODE_NULL_ARGUMENT},
    {ET_G3C4_DOMAIN_I1, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
     ET_I64_TENSOR_CODE_NULL_ARGUMENT},
    {ET_G3C4_DOMAIN_I2, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
     ET_F32_TENSOR_CODE_NULL_ARGUMENT}
  };
  const et_g3c4_error_state_internal invalid[] = {
    {-1, 1, 1}, {4, 1, 1},
    {ET_G3C4_DOMAIN_C4, 7, 1},
    {ET_G3C4_DOMAIN_K1, ET_KERNEL_ERROR_INTERNAL + 1, 1},
    {ET_G3C4_DOMAIN_I1, 1, ET_I64_TENSOR_CODE_PROVIDER_REJECTED + 1},
    {ET_G3C4_DOMAIN_I2, 1, ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT + 1},
    {ET_G3C4_DOMAIN_C4, 0, 1},
    {ET_G3C4_DOMAIN_C4, 1, 0}
  };
  et_g3c4_error_reset_internal();
  CHECK(et_g3c4_private_last_error_domain_v1() == 0);
  CHECK(et_g3c4_private_last_error_category_v1() == 0);
  CHECK(et_g3c4_private_last_error_code_v1() == 0);
  for (size_t index = 0u; index < sizeof(valid) / sizeof(valid[0]); index++) {
    et_g3c4_error_set_internal(
        valid[index].domain, valid[index].category, valid[index].code);
    et_g3c4_error_state_internal observed =
        et_g3c4_error_snapshot_internal();
    CHECK(memcmp(&observed, &valid[index], sizeof(observed)) == 0);
    et_g3c4_error_reset_internal();
    et_g3c4_error_restore_internal(valid[index]);
    observed = et_g3c4_error_snapshot_internal();
    CHECK(memcmp(&observed, &valid[index], sizeof(observed)) == 0);
  }
  for (size_t index = 0u;
       index < sizeof(invalid) / sizeof(invalid[0]); index++) {
    et_g3c4_error_set_internal(
        invalid[index].domain, invalid[index].category, invalid[index].code);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4);
    CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
  }
  et_kernel_error kernel_error = {
    ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED, "x", "y"};
  et_f32_tensor_error f32_error = {
    ET_F32_TENSOR_ERROR_INVALID_STATE, ET_F32_TENSOR_CODE_ACTIVE_BORROW,
    "x", "y"};
  CHECK(et_g3c4_capture_kernel(1, &kernel_error) == 1);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        ET_KERNEL_ERROR_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_KERNEL_CODE_ALLOCATION_FAILED);
  CHECK(et_g3c4_capture_f32(1, &f32_error) == 1);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I2);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_F32_TENSOR_CODE_ACTIVE_BORROW);

  et_kernel_runtime *runtime = NULL;
  et_kernel_error discover_error;
  OK(et_kernel_runtime_discover(
      et_g3c4_resolve_provider, (void *)et_n3k_kernel_provider_v1(),
      &runtime, &discover_error));
  const uint64_t bad_shape[2] = {3u, 3u};
  et_kernel_request_v1 bad_request = {
    sizeof(bad_request), "n3k.matrix-init.uniform", "f32", "cpu", 2u,
    bad_shape, 1u, {0}};
  const et_kernel_capability_v1 *entry = NULL;
  et_kernel_error actual_kernel_error;
  int32_t status = et_kernel_runtime_capability_require(
      runtime, "n3k.matrix-init", &bad_request,
      &entry, &actual_kernel_error);
  CHECK(status != 0);
  CHECK(et_g3c4_capture_kernel(status, &actual_kernel_error) == status);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        actual_kernel_error.category);
  CHECK(et_g3c4_private_last_error_code_v1() == actual_kernel_error.code);
  et_kernel_runtime_destroy(runtime);
}

static void test_allocation_rollback(void) {
  const et_f32_test_live_counts_v1 baseline = live_counts();
  const size_t registry_baseline = owner_registry_count();

  et_g3c4_allocation_limit = 0u;
  et_g3c4_successful_allocations = 0u;
  size_t k1_limit;
  for (k1_limit = 0u; k1_limit < 256u; k1_limit++) {
    et_m3t_test_k1_fail_after(k1_limit);
    et_g3c4_model_owner_internal *owner =
        et_g3c4_private_model_owner_create_seeded_v1(1);
    et_m3t_test_k1_allocator_reset();
    CHECK(owner == NULL);
    CHECK(et_g3c4_initializer_runtime == NULL);
    if (et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4) {
      CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
      CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
      break;
    }
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_KERNEL_CODE_ALLOCATION_FAILED);
    same_live(baseline);
    CHECK(owner_registry_count() == registry_baseline);
  }
  CHECK(k1_limit < 256u);
  same_live(baseline);
  CHECK(owner_registry_count() == registry_baseline);
  et_g3c4_allocation_limit = SIZE_MAX;
  et_g3c4_model_owner_internal *warm =
      et_g3c4_private_model_owner_create_seeded_v1(1);
  CHECK(warm != NULL);
  OK(et_g3c4_private_model_owner_abort_v1(warm));
  const size_t warmed_registry = owner_registry_count();
  CHECK(warmed_registry == registry_baseline + 1u);
  printf("seeded owner K1 allocation failpoints: %zu\n", k1_limit);

  et_g3c4_allocation_limit = 0u;
  et_g3c4_successful_allocations = 0u;
  CHECK(et_g3c4_private_model_owner_create_seeded_v1(2) == NULL);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  same_live(baseline);
  CHECK(owner_registry_count() == warmed_registry);
  et_g3c4_allocation_limit = SIZE_MAX;

  size_t limit;
  for (limit = 0u; limit < 512u; limit++) {
    et_f32_tensor_test_fail_alloc_after_v1(limit);
    et_g3c4_model_owner_internal *owner =
        et_g3c4_private_model_owner_create_seeded_v1(3);
    et_f32_tensor_test_reset_allocator_v1();
    if (owner != NULL) {
      OK(et_g3c4_private_model_owner_abort_v1(owner));
      break;
    }
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I2);
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    same_live(baseline);
    CHECK(owner_registry_count() == warmed_registry);
  }
  CHECK(limit < 512u);
  same_live(baseline);
  printf("seeded owner I2 allocation failpoints: %zu\n", limit);
}

static void test_admission_and_order(void) {
  int foreign = 0;
  CHECK(et_g3c4_private_model_owner_create_seeded_v1(-1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_model_owner_parameter_v1(&foreign, 0) == NULL);

  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(17);
  CHECK(owner != NULL);
  CHECK(et_g3c4_private_model_owner_create_seeded_v1(18) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_model_owner_parameter_v1(owner, -1) == NULL);
  CHECK(et_g3c4_private_model_owner_parameter_v1(owner, 14) == NULL);
  CHECK(et_g3c4_private_model_owner_bind_v1(owner, 0, NULL));
  OK(et_g3c4_private_model_owner_bind_v1(owner, 0, &identities[0][0]));
  CHECK(et_g3c4_private_model_owner_bind_v1(
      owner, 0, &identities[0][1]));
  CHECK(et_g3c4_private_model_owner_bind_v1(
      owner, 1, &identities[0][0]));
  CHECK(et_g3c4_private_model_owner_initialize_v1(owner, 0));
  for (int64_t index = 1; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[0][index]));
  CHECK(et_g3c4_private_model_owner_initialize_v1(owner, 1));
  CHECK(et_g3c4_private_model_owner_word_v1(owner, 4) == 0);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_READINESS);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  CHECK(et_g3c4_private_model_owner_initialize_v1(owner, 13));
  et_f32_parameter *stale_parameter = owner->parameters[0];
  OK(et_g3c4_private_model_owner_abort_v1(owner));
  OK(et_g3c4_private_model_owner_abort_v1(owner));
  CHECK(et_g3c4_private_model_owner_parameter_v1(owner, 0) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);

  et_g3c4_model_owner_internal *next =
      et_g3c4_private_model_owner_create_seeded_v1(19);
  CHECK(next != NULL);
  et_f32_tensor_error error;
  CHECK(et_g3c4_construction_parameter_preflight_internal(
      next, stale_parameter, NULL, &error));
  OK(et_g3c4_private_model_owner_abort_v1(next));
}

static void test_values_and_schedule(void) {
  const int64_t seed = 1729;
  et_g3c4_model_owner_internal *owner = create_initialized(1, seed);
  int64_t expected_state[4] = {1, seed, 0, 0};
  const float *storage[28] = {0};

  CHECK(et_g3c4_private_model_owner_word_v1(owner, 0) == 1);
  CHECK(et_g3c4_private_model_owner_word_v1(owner, 1) == seed);
  CHECK(et_g3c4_private_model_owner_word_v1(owner, 2) == 0);
  CHECK(et_g3c4_private_model_owner_word_v1(owner, 3) == 0);

  for (size_t index = 0u; index < 14u; index++) {
    et_f32_parameter *parameter = owner->parameters[index];
    et_f32_parameter *prior_parameter = NULL;
    const et_f32_tensor *value = NULL;
    et_f32_tensor *gradient = find_parameter(parameter)->gradient;
    et_f32_tensor_error error;
    et_f32_gradient_metadata_v1 metadata = {sizeof(metadata), 0u, 0u, 0u};
    size_t rank = 0u;
    size_t count = 0u;
    uint32_t actual[1024] = {0};
    uint32_t expected[1024] = {0};

    OK(et_f32_parameter_value_tensor_v1(parameter, &value, &error));
    OK(et_f32_tensor_rank_v1(value, &rank, &error));
    CHECK(rank == et_g3c4_parameter_rank(index));
    for (size_t dimension = 0u; dimension < rank; dimension++) {
      uint64_t extent = 0u;
      OK(et_f32_tensor_shape_at_v1(value, dimension, &extent, &error));
      CHECK(extent == et_g3c4_parameter_shapes[index][dimension]);
    }
    OK(et_f32_tensor_element_count_v1(value, &count, &error));
    CHECK(count == et_g3c4_parameter_elements(index));
    OK(et_f32_tensor_copy_bits_to_v1(value, actual, count, &error));
    if (rank == 2u) {
      dispatch_expected(index, expected_state, expected);
    } else if (index == 7u || index == 9u || index == 12u) {
      for (size_t element = 0u; element < count; element++)
        expected[element] = UINT32_C(0x3f800000);
    }
    CHECK(memcmp(actual, expected, count * sizeof(actual[0])) == 0);

    memset(actual, 0xff, sizeof(actual));
    OK(et_f32_tensor_copy_bits_to_v1(gradient, actual, count, &error));
    for (size_t element = 0u; element < count; element++)
      CHECK(actual[element] == 0u);
    OK(et_f32_parameter_gradient_metadata_v1(
        parameter, &metadata, &error));
    CHECK(metadata.state == ET_F32_GRADIENT_ABSENT);
    CHECK(metadata.contribution_count == 0u);
    CHECK(metadata.normalization_weight_bits == 0u);

    storage[2u * index] = et_f32_tensor_test_data_storage_v1(value);
    storage[2u * index + 1u] =
        et_f32_tensor_test_data_storage_v1(gradient);
    CHECK(storage[2u * index] != NULL);
    CHECK(storage[2u * index + 1u] != NULL);
    for (size_t prior = 0u; prior < 2u * index + 1u; prior++)
      CHECK(storage[prior] != storage[2u * index + 1u]);
    for (size_t prior = 0u; prior < 2u * index; prior++)
      CHECK(storage[prior] != storage[2u * index]);
    if (index != 0u) {
      prior_parameter = owner->parameters[index - 1u];
      CHECK(parameter != prior_parameter);
    }
  }
  CHECK(expected_state[0] == 1);
  CHECK(expected_state[1] == seed);
  CHECK(expected_state[2] == 292);
  CHECK(expected_state[3] == 0);
  for (int64_t index = 0; index < 4; index++)
    CHECK(et_g3c4_private_model_owner_word_v1(owner, index + 4) ==
          expected_state[index]);
  OK(et_g3c4_private_model_owner_abort_v1(owner));
}

typedef struct held_control {
  et_f32_tensor_borrow *borrow;
  et_f32_scoped_guard_internal scoped;
  et_f32_tensor_copy_plan *copy;
  et_f32_gradient_reset_plan *reset;
  et_f32_gradient_plan *gradient;
  et_f32_tensor *numerator;
} held_control;

static void hold_control(
    et_g3c4_model_owner_internal *owner, size_t index,
    int kind, held_control *held) {
  et_f32_parameter *parameter = owner->parameters[index];
  et_f32_tensor *value = parameter->value;
  et_f32_tensor *gradient = parameter->gradient;
  et_f32_tensor_error error;
  if (kind == 0)
    OK(et_f32_tensor_borrow_begin_v1(value, &held->borrow, &error));
  else if (kind == 1)
    OK(et_f32_tensor_borrow_begin_v1(gradient, &held->borrow, &error));
  else if (kind == 2)
    OK(et_f32_tensor_scoped_begin_internal(value, &held->scoped, &error));
  else if (kind == 3)
    OK(et_f32_tensor_scoped_begin_internal(gradient, &held->scoped, &error));
  else if (kind == 4) {
    et_f32_tensor_copy_assignment_v1 assignment = {
      sizeof(assignment), value, gradient};
    OK(et_f32_tensor_copy_plan_prepare_v1(
        1u, &assignment, &held->copy, &error));
  } else if (kind == 5) {
    OK(et_f32_gradient_reset_plan_prepare_v1(
        1u, &parameter, &held->reset, &error));
  } else {
    et_f32_gradient_contribution_v1 contribution = {
      sizeof(contribution), parameter, NULL, 0u};
    OK(et_f32_tensor_clone_v1(value, &held->numerator, &error));
    contribution.weighted_numerator = held->numerator;
    OK(et_f32_gradient_plan_prepare_v1(
        1u, &contribution, UINT32_C(0x3f800000),
        &held->gradient, &error));
  }
}

static void release_control(held_control *held) {
  et_f32_tensor_error error;
  OK(et_f32_tensor_borrow_end_v1(&held->borrow, &error));
  OK(et_f32_tensor_scoped_end_internal(&held->scoped));
  OK(et_f32_tensor_copy_plan_release_v1(&held->copy, &error));
  OK(et_f32_gradient_reset_plan_release_v1(&held->reset, &error));
  OK(et_f32_gradient_plan_release_v1(&held->gradient, &error));
  OK(et_f32_tensor_destroy_v1(&held->numerator, &error));
}

static void test_atomic_preflight(void) {
  et_g3c4_model_owner_internal *owner = create_initialized(2, 31);
  et_f32_test_live_counts_v1 baseline = live_counts();
  et_f32_tensor_error error;

  for (size_t index = 0u; index < 14u; index++) {
    for (int kind = 0; kind < 7; kind++) {
      held_control held = {0};
      hold_control(owner, index, kind, &held);
      et_f32_test_live_counts_v1 with_control = live_counts();
      CHECK(et_g3c4_construction_parameter_preflight_internal(
          owner, owner->parameters[0], owner->handles[0], &error));
      CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
      CHECK(owner->state == ET_G3C4_OWNER_OPEN);
      CHECK(et_g3c4_private_model_owner_abort_v1(owner));
      CHECK(owner->state == ET_G3C4_OWNER_OPEN);
      for (size_t member = 0u; member < 14u; member++)
        CHECK(et_f32_parameter_is_live_v1(owner->parameters[member]));
      same_live(with_control);

      if (kind == 0) {
        et_f32_tensor_error m3t_error;
        CHECK(et_m3t_f32_parameter_preflight_internal(
            owner->parameters[index], &m3t_error));
        CHECK(strcmp(m3t_error.operation, "m3t-parameter-preflight") == 0);
        CHECK(et_f32_parameter_idle_preflight_internal(
            owner->parameters[index], &error));
        CHECK(strcmp(error.operation, "f32-parameter-idle-preflight") == 0);
      }
      release_control(&held);
      same_live(baseline);
      OK(et_g3c4_construction_parameter_preflight_internal(
          owner, owner->parameters[index], owner->handles[index], &error));
    }
  }

  CHECK(et_g3c4_construction_parameter_preflight_internal(
      owner, owner->parameters[0], &identities[2][1], &error));
  CHECK(et_g3c4_construction_parameter_preflight_internal(
      owner, &error, owner->handles[0], &error));
  CHECK(et_g3c4_construction_parameter_preflight_internal(
      &error, owner->parameters[0], owner->handles[0], &error));

  et_f32_parameter *saved_parameter = owner->parameters[1];
  owner->parameters[1] = owner->parameters[0];
  CHECK(et_g3c4_construction_parameter_preflight_internal(
      owner, owner->parameters[0], owner->handles[0], &error));
  CHECK(et_g3c4_private_model_owner_abort_v1(owner));
  for (size_t member = 0u; member < 14u; member++)
    CHECK(et_f32_parameter_is_live_v1(
        member == 1u ? saved_parameter : owner->parameters[member]));
  owner->parameters[1] = saved_parameter;

  const void *saved_handle = owner->handles[1];
  owner->handles[1] = owner->handles[0];
  CHECK(et_g3c4_construction_parameter_preflight_internal(
      owner, owner->parameters[0], owner->handles[0], &error));
  CHECK(et_g3c4_private_model_owner_abort_v1(owner));
  owner->handles[1] = saved_handle;

  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  held_control prepared = {0};
  hold_control(owner, 13u, 4, &prepared);
  CHECK(et_g3c4_private_model_owner_abort_v1(owner));
  CHECK(owner->state == ET_G3C4_OWNER_PREPARED);
  release_control(&prepared);
  OK(et_g3c4_private_model_owner_abort_v1(owner));
  same_live((et_f32_test_live_counts_v1){
      .struct_size = sizeof(et_f32_test_live_counts_v1)});
}

static void test_prepare_and_commit(void) {
  et_g3c4_model_owner_internal *owner = create_initialized(3, 47);
  et_f32_tensor_error error;
  uint32_t saved[1024];
  size_t count = 0u;

  owner->active = owner;
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  owner->active = NULL;

  et_f32_tensor *value = owner->parameters[0]->value;
  OK(et_f32_tensor_element_count_v1(value, &count, &error));
  OK(et_f32_tensor_copy_bits_to_v1(value, saved, count, &error));
  uint32_t nonfinite[1024];
  memcpy(nonfinite, saved, count * sizeof(saved[0]));
  nonfinite[0] = UINT32_C(0x7f800000);
  OK(et_f32_tensor_copy_bits_from_v1(value, nonfinite, count, &error));
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_READINESS);
  OK(et_f32_tensor_copy_bits_from_v1(value, saved, count, &error));

  et_f32_parameter_test_set_metadata_v1(
      owner->parameters[1], ET_F32_GRADIENT_PRESENT, 1u,
      UINT32_C(0x3f800000));
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  et_f32_parameter_test_set_metadata_v1(
      owner->parameters[1], ET_F32_GRADIENT_ABSENT, 0u, 0u);

  owner->successor[2] = 291;
  CHECK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
  owner->successor[2] = 292;

  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  CHECK(owner->state == ET_G3C4_OWNER_PREPARED);
  CHECK(et_g3c4_private_model_owner_initialize_v1(owner, 0));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  CHECK(owner->state == ET_G3C4_OWNER_SEALED);
  CHECK(staged_c4_owner == NULL);
  CHECK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  CHECK(et_g3c4_private_model_owner_abort_v1(owner));
  CHECK(owner->state == ET_G3C4_OWNER_SEALED);
  CHECK(et_g3c4_private_model_owner_parameter_v1(owner, 13) ==
        owner->parameters[13]);
}

int main(void) {
  test_error_snapshot();
  test_allocation_rollback();
  test_admission_and_order();
  test_values_and_schedule();
  test_atomic_preflight();
  test_prepare_and_commit();
  printf("G3-C4 seeded native owner: %zu checks\n", checks);
  return 0;
}
