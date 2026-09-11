#include "o2_optimizer_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,       \
              #condition);                                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static et_f32_parameter *make_parameter(const uint32_t *bits, size_t count,
                                        const void *identity) {
  uint64_t shape[1] = {(uint64_t)count};
  et_f32_tensor *initial = NULL;
  et_f32_parameter *parameter = NULL;
  et_f32_tensor_error error;
  memset(&error, 0, sizeof(error));
  CHECK(et_f32_tensor_create_v1(1u, shape, &initial, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(initial, bits, count, &error) == 0);
  CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, identity, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  return parameter;
}

static void accumulate(et_f32_parameter *parameter, const uint32_t *bits,
                       size_t count, uint64_t ordinal, uint32_t weight_bits) {
  uint64_t shape[1] = {(uint64_t)count};
  et_f32_tensor *source = NULL;
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_contribution_v1 contribution;
  et_f32_tensor_error error;
  memset(&error, 0, sizeof(error));
  CHECK(et_f32_tensor_create_v1(1u, shape, &source, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(source, bits, count, &error) == 0);
  contribution.struct_size = sizeof(contribution);
  contribution.destination = parameter;
  contribution.weighted_numerator = source;
  contribution.expected_ordinal = ordinal;
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &contribution, weight_bits, &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&source, &error) == 0);
}

static void parameter_bits(et_f32_parameter *parameter, uint32_t *bits,
                           size_t count) {
  et_f32_tensor *snapshot = NULL;
  et_f32_tensor_error error;
  memset(&error, 0, sizeof(error));
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(snapshot, bits, count, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&snapshot, &error) == 0);
}

static et_o2_optimizer *make_optimizer(et_f32_parameter **parameters,
                                       const void **identities, size_t count) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 error;
  size_t index;
  memset(&error, 0, sizeof(error));
  CHECK(et_o2_optimizer_builder_create_v1(
            count, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            UINT32_C(0x3f800000), &builder, &error) == 0);
  for (index = 0u; index < count; ++index) {
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, index, parameters[index], identities[index],
              UINT32_C(0x3dcccccd),         /* 0.1 */
              0u, 0u, UINT32_C(0x3f800000), /* epsilon 1 */
              0u, &error) == 0);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  CHECK(builder == NULL && optimizer != NULL);
  return optimizer;
}

static float f32(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static int close_f32(uint32_t actual, uint32_t expected) {
  float delta = f32(actual) - f32(expected);
  float scale = f32(expected);
  if (delta < 0.0f) {
    delta = -delta;
  }
  if (scale < 0.0f) {
    scale = -scale;
  }
  return delta <= 2.0e-6f + 2.0e-5f * scale;
}

static void test_frozen_adamw_reference(void) {
  static int identity;
  const uint32_t initial[] = {UINT32_C(0x3f800000), UINT32_C(0xc0000000),
                              UINT32_C(0x3f000000)};
  const uint32_t gradients[2][3] = {
      {UINT32_C(0x3e800000), UINT32_C(0xbf000000), 0u},
      {UINT32_C(0xbe000000), UINT32_C(0x3e800000), UINT32_C(0x3f800000)}};
  const uint32_t expected_parameter[2][3] = {
      {UINT32_C(0x3f7d2f1b), UINT32_C(0xbffe76c9), UINT32_C(0x3effbe77)},
      {UINT32_C(0x3f7c3fbf), UINT32_C(0xbffdde5f), UINT32_C(0x3efbada4)}};
  const uint32_t expected_avg[2][3] = {
      {UINT32_C(0x3cccccd0), UINT32_C(0xbd4cccd0), 0u},
      {UINT32_C(0x3c23d70c), UINT32_C(0xbca3d70c), UINT32_C(0x3dccccd0)}};
  const uint32_t expected_avg_sq[2][3] = {
      {UINT32_C(0x38831200), UINT32_C(0x39831200), 0u},
      {UINT32_C(0x38a3b4f2), UINT32_C(0x39a3b4f2), UINT32_C(0x3a831200)}};
  et_f32_parameter *parameter = make_parameter(initial, 3u, &identity);
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 error;
  uint32_t actual[3];
  size_t step;
  size_t index;
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            UINT32_C(0x3f800000), &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameter, &identity, UINT32_C(0x3c23d70a),
            UINT32_C(0x3f666666), UINT32_C(0x3f7fbe77), UINT32_C(0x322bcc77),
            UINT32_C(0x3dcccccd), &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  for (step = 0u; step < 2u; ++step) {
    accumulate(parameter, gradients[step], 3u, 0u, UINT32_C(0x3f800000));
    CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
    parameter_bits(parameter, actual, 3u);
    for (index = 0u; index < 3u; ++index) {
      CHECK(close_f32(actual[index], expected_parameter[step][index]));
    }
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              optimizer, 0u, ET_O2_MOMENT_EXP_AVG, actual, 3u, &error) == 0);
    for (index = 0u; index < 3u; ++index) {
      CHECK(close_f32(actual[index], expected_avg[step][index]));
    }
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 3u, &error) == 0);
    for (index = 0u; index < 3u; ++index) {
      CHECK(close_f32(actual[index], expected_avg_sq[step][index]));
    }
    if (step == 0u) {
      CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &error) == 0);
    }
  }
}

static void test_frozen_linear_schedule(void) {
  static int identity;
  const uint32_t initial[] = {UINT32_C(0x3f800000)};
  const uint32_t expected[] = {UINT32_C(0x3f000000), UINT32_C(0x3f800000),
                               UINT32_C(0x3f466666), UINT32_C(0x3f0ccccd),
                               UINT32_C(0x3ea66666), UINT32_C(0x3dcccccd),
                               UINT32_C(0x3dcccccd), UINT32_C(0x3dcccccd)};
  et_f32_parameter *parameter = make_parameter(initial, 1u, &identity);
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 error;
  uint32_t factor;
  size_t index;
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_LINEAR, 2u, 6u,
            UINT32_C(0x3dcccccd), &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(builder, 0u, parameter, &identity,
                                       UINT32_C(0x3ca3d70a), 0u, 0u,
                                       UINT32_C(0x3f800000), 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  for (index = 0u; index < sizeof(expected) / sizeof(expected[0]); ++index) {
    CHECK(et_o2_test_optimizer_set_completed_updates_v1(optimizer,
                                                        (uint64_t)index) == 0);
    CHECK(et_o2_optimizer_schedule_factor_bits_v1(optimizer, &factor, &error) ==
          0);
    CHECK(factor == expected[index]);
  }
}

static et_o2_optimizer *make_linear_schedule_optimizer(uint64_t warmup,
                                                       uint64_t total,
                                                       uint32_t minimum_ratio,
                                                       const void *identity) {
  const uint32_t initial[] = {UINT32_C(0x3f800000)};
  et_f32_parameter *parameter = make_parameter(initial, 1u, identity);
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 error;
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_LINEAR, warmup, total,
            minimum_ratio, &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(builder, 0u, parameter, identity, 0u, 0u,
                                       0u, UINT32_C(0x3f800000), 0u,
                                       &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  return optimizer;
}

static void check_schedule_at(et_o2_optimizer *optimizer, uint64_t update,
                              uint32_t expected) {
  et_o2_error_v1 error;
  uint32_t actual = UINT32_MAX;
  CHECK(update > 0u);
  CHECK(et_o2_test_optimizer_set_completed_updates_v1(optimizer, update - 1u) ==
        0);
  CHECK(et_o2_optimizer_schedule_factor_bits_v1(optimizer, &actual, &error) ==
        0);
  CHECK(actual == expected);
}

static void test_schedule_exact_rounding_full_domain(void) {
  static int identities[3];
  const uint64_t i64_maximum = (uint64_t)INT64_MAX;
  et_o2_optimizer *ties = make_linear_schedule_optimizer(
      UINT64_C(33554432), UINT64_C(33554433), 0u, &identities[0]);
  et_o2_optimizer *large = make_linear_schedule_optimizer(
      i64_maximum - 1u, i64_maximum, 0u, &identities[1]);
  et_o2_optimizer *subnormal =
      make_linear_schedule_optimizer(0u, i64_maximum, 1u, &identities[2]);

  /* Exact midpoints between adjacent binary32 values round to the even low
   * or high significand, respectively. */
  check_schedule_at(ties, UINT64_C(16777217), UINT32_C(0x3f000000));
  check_schedule_at(ties, UINT64_C(16777219), UINT32_C(0x3f000002));

  /* Ratios retain full signed-i64 counter information rather than narrowing
   * through binary64 or binary32 counter conversion. */
  check_schedule_at(large, UINT64_C(4611686018427387903), UINT32_C(0x3f000000));
  check_schedule_at(large, i64_maximum - 2u, UINT32_C(0x3f800000));
  check_schedule_at(subnormal, i64_maximum - 1u, UINT32_C(0x20000000));
  check_schedule_at(subnormal, i64_maximum, 1u);
}

static void check_gradient_state(et_f32_parameter *parameter, uint32_t state,
                                 uint64_t count, uint32_t weight) {
  et_f32_gradient_metadata_v1 metadata;
  et_f32_tensor_error error;
  memset(&metadata, 0, sizeof(metadata));
  metadata.struct_size = sizeof(metadata);
  memset(&error, 0, sizeof(error));
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
        0);
  CHECK(metadata.state == state);
  CHECK(metadata.contribution_count == count);
  CHECK(metadata.normalization_weight_bits == weight);
}

static void test_step_zero_and_state_lifetime(void) {
  static int identity;
  const uint32_t initial[] = {UINT32_C(0x3f800000)};
  const uint32_t gradient[] = {UINT32_C(0x3f800000)};
  et_f32_parameter *parameter = make_parameter(initial, 1u, &identity);
  et_f32_parameter *parameters[] = {parameter};
  const void *identities[] = {&identity};
  et_o2_optimizer *optimizer = make_optimizer(parameters, identities, 1u);
  et_o2_optimizer_state *state = NULL;
  et_o2_optimizer_state *sibling = NULL;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_o2_test_live_counts_v1 counts;
  et_o2_error_v1 error;
  uint32_t bits[1];
  uint64_t completed = 0u;

  accumulate(parameter, gradient, 1u, 0u, UINT32_C(0x3f800000));
  memset(&error, 0, sizeof(error));
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  CHECK(et_o2_optimizer_completed_updates_v1(optimizer, &completed, &error) ==
        0);
  CHECK(completed == 1u);
  parameter_bits(parameter, bits, 1u);
  CHECK(bits[0] == UINT32_C(0x3f733333)); /* 1 - 0.1 / (1 + 1) */
  CHECK(et_o2_test_optimizer_moment_bits_v1(optimizer, 0u, ET_O2_MOMENT_EXP_AVG,
                                            bits, 1u, &error) == 0);
  CHECK(bits[0] == UINT32_C(0x3f800000));
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, bits, 1u, &error) == 0);
  CHECK(bits[0] == UINT32_C(0x3f800000));
  check_gradient_state(parameter, ET_F32_GRADIENT_PRESENT, 1u,
                       UINT32_C(0x3f800000));

  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, &error) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(state == NULL);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &error) == 0);
  check_gradient_state(parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);

  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, &error) == 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &sibling, &error) == 0);
  memset(&counts, 0, sizeof(counts));
  counts.struct_size = sizeof(counts);
  et_o2_test_live_counts_snapshot_v1(&counts);
  CHECK(counts.live_states == 2u);
  CHECK(counts.live_state_handles == 4u);
  CHECK(counts.owned_state_clones == 4u);

  CHECK(et_o2_optimizer_state_moment_handle_v1(state, 0u, ET_O2_MOMENT_EXP_AVG,
                                               &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                              &error) == 0);
  CHECK(view != NULL && view->byte_length == sizeof(uint32_t));
  CHECK(et_o2_optimizer_state_release_v1(state, &error) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);

  et_o2_test_fail_release_after_v1(0u);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) ==
        ET_O2_STATUS_INTERNAL);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  CHECK(et_o2_optimizer_state_entry_count_v1(state, &(size_t){0}, &error) ==
        ET_O2_STATUS_INVALID_STATE);
  et_o2_test_reset_failpoints_v1();
  parameter_bits(parameter, bits, 1u);
  CHECK(bits[0] == UINT32_C(0x3f733333));
  CHECK(et_o2_test_optimizer_moment_bits_v1(optimizer, 0u, ET_O2_MOMENT_EXP_AVG,
                                            bits, 1u, &error) == 0);
  CHECK(bits[0] == UINT32_C(0x3f800000));

  memset(&counts, 0, sizeof(counts));
  counts.struct_size = sizeof(counts);
  et_o2_test_live_counts_snapshot_v1(&counts);
  CHECK(counts.live_states == 1u);
  CHECK(counts.owned_state_clones == 2u);
  CHECK(et_o2_optimizer_state_release_v1(sibling, &error) == 0);
  memset(&counts, 0, sizeof(counts));
  counts.struct_size = sizeof(counts);
  et_o2_test_live_counts_snapshot_v1(&counts);
  CHECK(counts.live_states == 0u);
  CHECK(counts.owned_state_clones == 0u);
}

static void test_atomic_missing_gradient(void) {
  static int first_identity;
  static int second_identity;
  const uint32_t first_initial[] = {UINT32_C(0x3f800000)};
  const uint32_t second_initial[] = {UINT32_C(0x40000000)};
  const uint32_t gradient[] = {UINT32_C(0x3f800000)};
  et_f32_parameter *parameters[2];
  const void *identities[2] = {&first_identity, &second_identity};
  et_o2_optimizer *optimizer;
  et_o2_error_v1 error;
  uint32_t bits[1];
  uint64_t completed = 99u;
  parameters[0] = make_parameter(first_initial, 1u, identities[0]);
  parameters[1] = make_parameter(second_initial, 1u, identities[1]);
  optimizer = make_optimizer(parameters, identities, 2u);
  accumulate(parameters[0], gradient, 1u, 0u, UINT32_C(0x3f800000));
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) ==
        ET_O2_STATUS_INVALID_STATE);
  parameter_bits(parameters[0], bits, 1u);
  CHECK(bits[0] == first_initial[0]);
  parameter_bits(parameters[1], bits, 1u);
  CHECK(bits[0] == second_initial[0]);
  CHECK(et_o2_optimizer_completed_updates_v1(optimizer, &completed, &error) ==
        0);
  CHECK(completed == 0u);
  CHECK(et_o2_test_optimizer_set_completed_updates_v1(
            optimizer, (uint64_t)INT64_MAX) == 0);
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(error.code == ET_O2_CODE_COUNTER_OVERFLOW);
  parameter_bits(parameters[0], bits, 1u);
  CHECK(bits[0] == first_initial[0]);
}

static void test_snapshot_allocation_rollback_and_cross_owner(void) {
  static int identity;
  const uint32_t initial[] = {UINT32_C(0x3f800000)};
  et_f32_parameter *parameter = make_parameter(initial, 1u, &identity);
  et_f32_parameter *parameters[] = {parameter};
  const void *identities[] = {&identity};
  et_o2_optimizer *optimizer = make_optimizer(parameters, identities, 1u);
  et_o2_optimizer_state *first = NULL;
  et_o2_optimizer_state *second = NULL;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_o2_test_live_counts_v1 counts;
  et_o2_error_v1 error;

  et_o2_test_fail_alloc_after_v1(3u);
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &first, &error) ==
        ET_O2_STATUS_INTERNAL);
  CHECK(first == NULL);
  et_o2_test_reset_failpoints_v1();
  memset(&counts, 0, sizeof(counts));
  counts.struct_size = sizeof(counts);
  et_o2_test_live_counts_snapshot_v1(&counts);
  CHECK(counts.owned_state_clones == 0u);
  CHECK(counts.live_states == 0u);

  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &first, &error) == 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &second, &error) == 0);
  CHECK(et_o2_optimizer_state_moment_handle_v1(first, 0u, ET_O2_MOMENT_EXP_AVG,
                                               &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(second, handle, &borrow, &view,
                                              &error) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(borrow == NULL && view == NULL);
  CHECK(et_o2_optimizer_state_release_v1(first, &error) == 0);
  CHECK(et_o2_optimizer_state_release_v1(second, &error) == 0);
}

int main(void) {
  et_o2_test_reset_failpoints_v1();
  test_frozen_adamw_reference();
  test_frozen_linear_schedule();
  test_schedule_exact_rounding_full_domain();
  test_step_zero_and_state_lifetime();
  test_atomic_missing_gradient();
  test_snapshot_allocation_rollback_and_cross_owner();
  puts("O2 native optimizer PASS");
  return 0;
}
