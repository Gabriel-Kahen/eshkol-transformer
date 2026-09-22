#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,     \
              #condition);                                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static et_f32_tensor *make_tensor(const uint32_t bits[2]) {
  const uint64_t shape[1] = {2u};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(1u, shape, &tensor, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, 2u, &error) == 0);
  return tensor;
}

static void check_bits(const et_f32_tensor *tensor,
                       const uint32_t expected[2]) {
  et_f32_tensor_error error;
  uint32_t actual[2] = {UINT32_MAX, UINT32_MAX};
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
}

static et_f32_test_live_counts_v1 live_counts(void) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  return counts;
}

static void release_copy(et_f32_tensor_copy_plan **plan) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_copy_plan_release_v1(plan, &error) == 0);
  CHECK(*plan == NULL);
}

static void release_reset(et_f32_gradient_reset_plan **plan) {
  et_f32_tensor_error error;
  CHECK(et_f32_gradient_reset_plan_release_v1(plan, &error) == 0);
  CHECK(*plan == NULL);
}

static void destroy_tensor(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

int main(void) {
  const uint32_t initial_bits[2] = {UINT32_C(0x3f800000),
                                     UINT32_C(0x40000000)};
  const uint32_t gradient_bits[2] = {UINT32_C(0x40a00000),
                                      UINT32_C(0x40c00000)};
  const uint32_t zero_bits[2] = {0u, 0u};
  const uint32_t next_parameter_bits[2] = {UINT32_C(0x40400000),
                                            UINT32_C(0x40800000)};
  const uint32_t next_avg_bits[2] = {UINT32_C(0x3dcccccd),
                                      UINT32_C(0x3e4ccccd)};
  const uint32_t next_avg_sq_bits[2] = {UINT32_C(0x3a83126f),
                                         UINT32_C(0x3b03126f)};
  et_f32_tensor_error error;
  et_f32_test_live_counts_v1 baseline = live_counts();
  et_f32_tensor *initial = make_tensor(initial_bits);
  et_f32_tensor *gradient_source = make_tensor(gradient_bits);
  et_f32_tensor *zero_source = make_tensor(zero_bits);
  et_f32_tensor *exp_avg = make_tensor(zero_bits);
  et_f32_tensor *exp_avg_sq = make_tensor(zero_bits);
  et_f32_tensor *next_parameter = make_tensor(next_parameter_bits);
  et_f32_tensor *next_avg = make_tensor(next_avg_bits);
  et_f32_tensor *next_avg_sq = make_tensor(next_avg_sq_bits);
  et_f32_parameter *parameter = NULL;
  const et_f32_tensor *value_before = NULL;
  const et_f32_tensor *value_after = NULL;
  et_f32_gradient_plan *gradient_plan = NULL;
  et_f32_tensor_copy_plan *copy_plan = NULL;
  et_f32_gradient_reset_plan *reset_plan = NULL;
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  et_f32_test_live_counts_v1 before_plans;
  et_f32_test_live_counts_v1 with_plans;
  int identity = 0;

  CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &value_before, &error) ==
        0);

  {
    et_f32_gradient_contribution_v1 contribution = {
        .struct_size = sizeof(contribution),
        .destination = parameter,
        .weighted_numerator = gradient_source,
        .expected_ordinal = 0u,
    };
    CHECK(et_f32_gradient_plan_prepare_v1(
              1u, &contribution, UINT32_C(0x3f800000), &gradient_plan,
              &error) == 0);
    CHECK(et_f32_gradient_plan_commit_v1(gradient_plan, &error) == 0);
    CHECK(et_f32_gradient_plan_release_v1(&gradient_plan, &error) == 0);
  }
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
        0);
  CHECK(metadata.state == ET_F32_GRADIENT_PRESENT);
  CHECK(metadata.contribution_count == 1u);
  CHECK(metadata.normalization_weight_bits == UINT32_C(0x3f800000));

  {
    et_f32_tensor_copy_assignment_v1 assignments[3] = {
        {ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE,
         (et_f32_tensor *)value_before, next_parameter},
        {ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE, exp_avg, next_avg},
        {ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE, exp_avg_sq, next_avg_sq},
    };
    et_f32_parameter *parameters[1] = {parameter};

    before_plans = live_counts();
    CHECK(et_f32_tensor_copy_plan_prepare_v1(3u, assignments, &copy_plan,
                                             &error) == 0);
    CHECK(et_f32_gradient_reset_plan_prepare_v1(
              1u, parameters, &reset_plan, &error) == 0);
    with_plans = live_counts();
    CHECK(with_plans.copy_plans == before_plans.copy_plans + 1u);
    CHECK(with_plans.reset_plans == before_plans.reset_plans + 1u);

    /* Abort both prepared plans: neither plan has mutated its destination. */
    release_reset(&reset_plan);
    release_copy(&copy_plan);
    check_bits(value_before, initial_bits);
    check_bits(exp_avg, zero_bits);
    check_bits(exp_avg_sq, zero_bits);
    memset(&metadata, 0, sizeof(metadata));
    metadata.struct_size = sizeof(metadata);
    CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
          0);
    CHECK(metadata.state == ET_F32_GRADIENT_PRESENT);
    CHECK(metadata.contribution_count == 1u);

    /* Retry, then prove that the exact commit/release tail allocates nothing. */
    CHECK(et_f32_tensor_copy_plan_prepare_v1(3u, assignments, &copy_plan,
                                             &error) == 0);
    CHECK(et_f32_gradient_reset_plan_prepare_v1(
              1u, parameters, &reset_plan, &error) == 0);
    et_f32_tensor_test_fail_alloc_after_v1(0u);
    CHECK(et_f32_tensor_copy_plan_commit_v1(copy_plan, &error) == 0);
    CHECK(et_f32_gradient_reset_plan_commit_v1(reset_plan, &error) == 0);
    release_copy(&copy_plan);
    release_reset(&reset_plan);
    et_f32_tensor_test_reset_allocator_v1();
  }

  CHECK(et_f32_parameter_value_tensor_v1(parameter, &value_after, &error) ==
        0);
  CHECK(value_after == value_before);
  check_bits(value_after, next_parameter_bits);
  check_bits(exp_avg, next_avg_bits);
  check_bits(exp_avg_sq, next_avg_sq_bits);
  memset(&metadata, 0, sizeof(metadata));
  metadata.struct_size = sizeof(metadata);
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
        0);
  CHECK(metadata.state == ET_F32_GRADIENT_ABSENT);
  CHECK(metadata.contribution_count == 0u);
  CHECK(metadata.normalization_weight_bits == 0u);
  /* The absent backing is not publicly borrowable. Re-admit an exact-zero
   * contribution, inspect it, then return the slot to absent. This proves the
   * reset cleared bytes as well as metadata. */
  {
    et_f32_gradient_contribution_v1 contribution = {
        .struct_size = sizeof(contribution),
        .destination = parameter,
        .weighted_numerator = zero_source,
        .expected_ordinal = 0u,
    };
    et_f32_tensor *snapshot = NULL;
    et_f32_parameter *parameters[1] = {parameter};
    CHECK(et_f32_gradient_plan_prepare_v1(
              1u, &contribution, UINT32_C(0x3f800000), &gradient_plan,
              &error) == 0);
    CHECK(et_f32_gradient_plan_commit_v1(gradient_plan, &error) == 0);
    CHECK(et_f32_gradient_plan_release_v1(&gradient_plan, &error) == 0);
    CHECK(et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot,
                                                &error) == 0);
    check_bits(snapshot, zero_bits);
    destroy_tensor(&snapshot);
    CHECK(et_f32_gradient_reset_plan_prepare_v1(
              1u, parameters, &reset_plan, &error) == 0);
    CHECK(et_f32_gradient_reset_plan_commit_v1(reset_plan, &error) == 0);
    release_reset(&reset_plan);
  }

  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  destroy_tensor(&initial);
  destroy_tensor(&gradient_source);
  destroy_tensor(&zero_source);
  destroy_tensor(&exp_avg);
  destroy_tensor(&exp_avg_sq);
  destroy_tensor(&next_parameter);
  destroy_tensor(&next_avg);
  destroy_tensor(&next_avg_sq);
  {
    et_f32_test_live_counts_v1 final = live_counts();
    CHECK(final.tensors == baseline.tensors);
    CHECK(final.parameters == baseline.parameters);
    CHECK(final.borrows == baseline.borrows);
    CHECK(final.copy_plans == baseline.copy_plans);
    CHECK(final.gradient_plans == baseline.gradient_plans);
    CHECK(final.reset_plans == baseline.reset_plans);
    CHECK(final.owned_clones == baseline.owned_clones);
  }

  printf("TR3-O I2 plan coexistence PASS: %zu checks\n", checks);
  return 0;
}
