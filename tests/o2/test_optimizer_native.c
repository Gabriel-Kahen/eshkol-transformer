#include "o2_optimizer_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>

static int checks;
static int failures;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                   \
    if (!(condition)) {                                                         \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                                \
      failures++;                                                               \
    }                                                                           \
  } while (0)

typedef struct fixture {
  et_f32_tensor *initial;
  et_f32_parameter *parameter;
  et_o2_optimizer *optimizer;
} fixture;

static void expect_o2_error(int32_t result, const et_o2_error_v1 *error,
                            uint32_t category, uint32_t code) {
  if (result != (int32_t)category || error->category != category ||
      error->code != code) {
    (void)fprintf(stderr,
                  "error mismatch: result=%d category=%u code=%u op=%s "
                  "expected=%u/%u\n",
                  result, error->category, error->code, error->operation,
                  category, code);
  }
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
}

static et_f32_tensor *make_tensor(const uint32_t *bits, size_t count) {
  const uint64_t shape[] = {(uint64_t)count};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(1u, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, count, &error) == 0);
  return tensor;
}

static void destroy_tensor(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void check_tensor_bits(const et_f32_tensor *tensor,
                              const uint32_t *expected, size_t count) {
  et_f32_tensor_error error;
  uint32_t actual[16];
  CHECK(count <= sizeof(actual) / sizeof(actual[0]));
  if (count > sizeof(actual) / sizeof(actual[0])) {
    return;
  }
  memset(actual, 0xa5, count * sizeof(*actual));
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, actual, count, &error) == 0);
  CHECK(memcmp(actual, expected, count * sizeof(*actual)) == 0);
}

static void check_parameter_bits(et_f32_parameter *parameter,
                                 const uint32_t *expected, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *snapshot = NULL;
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) == 0);
  CHECK(snapshot != NULL);
  check_tensor_bits(snapshot, expected, count);
  destroy_tensor(&snapshot);
}

static void read_parameter_bits(et_f32_parameter *parameter, uint32_t *bits,
                                size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *snapshot = NULL;
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(snapshot, bits, count, &error) == 0);
  destroy_tensor(&snapshot);
}

static void check_gradient_bits(et_f32_parameter *parameter,
                                const uint32_t *expected, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  CHECK(et_f32_parameter_gradient_borrow_begin_v1(parameter, &borrow, &error) ==
        0);
  CHECK(borrow != NULL);
  CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL);
  CHECK(view->byte_length == count * sizeof(*expected));
  CHECK(memcmp(view->data, expected, count * sizeof(*expected)) == 0);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
}

static et_f32_gradient_metadata_v1 gradient_metadata(
    et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) ==
        0);
  return metadata;
}

static void set_gradient(et_f32_parameter *parameter, const uint32_t *bits,
                         size_t count, uint64_t contributions,
                         uint32_t weight_bits) {
  et_f32_parameter_test_set_gradient_bits_v1(parameter, bits, count);
  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        contributions, weight_bits);
}

static void accumulate_gradient(et_f32_parameter *parameter,
                                const uint32_t *numerator_bits, size_t count,
                                uint64_t ordinal, uint32_t weight_bits) {
  et_f32_tensor_error error;
  et_f32_tensor *numerator = make_tensor(numerator_bits, count);
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_contribution_v1 contribution = {
      .struct_size = sizeof(contribution),
      .destination = parameter,
      .weighted_numerator = numerator,
      .expected_ordinal = ordinal};
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &contribution, weight_bits, &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  destroy_tensor(&numerator);
}

static et_o2_optimizer *make_optimizer(et_f32_parameter *parameter,
                                       const void *identity,
                                       uint32_t learning_rate_bits,
                                       uint32_t beta1_bits,
                                       uint32_t beta2_bits,
                                       uint32_t epsilon_bits,
                                       uint32_t weight_decay_bits,
                                       uint32_t clip_kind,
                                       uint32_t clip_max_bits,
                                       uint32_t schedule_kind,
                                       uint64_t warmup, uint64_t total,
                                       uint32_t minimum_ratio_bits) {
  et_o2_error_v1 error;
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, clip_kind, clip_max_bits, schedule_kind, warmup, total,
            minimum_ratio_bits, &builder, &error) == 0);
  CHECK(builder != NULL);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameter, identity, learning_rate_bits, beta1_bits,
            beta2_bits, epsilon_bits, weight_decay_bits, &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  CHECK(builder == NULL);
  CHECK(optimizer != NULL);
  return optimizer;
}

static fixture make_fixture(uint32_t initial_bits, uint32_t learning_rate_bits,
                            uint32_t beta1_bits, uint32_t beta2_bits,
                            uint32_t epsilon_bits, uint32_t weight_decay_bits,
                            uint32_t clip_kind, uint32_t clip_max_bits,
                            uint32_t schedule_kind, uint64_t warmup,
                            uint64_t total, uint32_t minimum_ratio_bits) {
  et_f32_tensor_error error;
  fixture result;
  result.initial = make_tensor(&initial_bits, 1u);
  result.parameter = NULL;
  result.optimizer = NULL;
  CHECK(et_f32_parameter_create_v1(result.initial, &result.parameter, &error) ==
        0);
  CHECK(et_f32_parameter_bind_identity_v1(result.parameter, result.parameter,
                                          &error) == 0);
  result.optimizer = make_optimizer(
      result.parameter, result.parameter, learning_rate_bits, beta1_bits,
      beta2_bits, epsilon_bits, weight_decay_bits, clip_kind, clip_max_bits,
      schedule_kind, warmup, total, minimum_ratio_bits);
  return result;
}

static uint64_t completed_updates(const et_o2_optimizer *optimizer) {
  et_o2_error_v1 error;
  uint64_t value = UINT64_MAX;
  CHECK(et_o2_optimizer_completed_updates_v1(optimizer, &value, &error) == 0);
  return value;
}

static uint32_t schedule_factor(const et_o2_optimizer *optimizer) {
  et_o2_error_v1 error;
  uint32_t value = UINT32_MAX;
  CHECK(et_o2_optimizer_schedule_factor_bits_v1(optimizer, &value, &error) ==
        0);
  return value;
}

static void check_gradient_metadata(et_f32_parameter *parameter,
                                    uint32_t state, uint64_t count,
                                    uint32_t weight_bits) {
  et_f32_gradient_metadata_v1 actual = gradient_metadata(parameter);
  CHECK(actual.state == state);
  CHECK(actual.contribution_count == count);
  CHECK(actual.normalization_weight_bits == weight_bits);
}

static void check_o2_live_counts_equal(const et_o2_test_live_counts_v1 *actual,
                                       const et_o2_test_live_counts_v1 *expected) {
  CHECK(actual->builders == expected->builders);
  CHECK(actual->optimizers == expected->optimizers);
  CHECK(actual->live_states == expected->live_states);
  CHECK(actual->dead_states == expected->dead_states);
  CHECK(actual->live_state_handles == expected->live_state_handles);
  CHECK(actual->dead_state_handles == expected->dead_state_handles);
  CHECK(actual->state_borrows == expected->state_borrows);
  CHECK(actual->owned_state_clones == expected->owned_state_clones);
}

static void check_i2_live_counts_equal(const et_f32_test_live_counts_v1 *actual,
                                       const et_f32_test_live_counts_v1 *expected) {
  CHECK(actual->tensors == expected->tensors);
  CHECK(actual->parameters == expected->parameters);
  CHECK(actual->borrows == expected->borrows);
  CHECK(actual->copy_plans == expected->copy_plans);
  CHECK(actual->gradient_plans == expected->gradient_plans);
  CHECK(actual->reset_plans == expected->reset_plans);
  CHECK(actual->owned_clones == expected->owned_clones);
}

static void test_builder_rejections(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *initial = make_tensor(&one, 1u);
  et_f32_parameter *parameter = NULL;
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  int identity = 0;

  CHECK(et_f32_parameter_create_v1(initial, &parameter, &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &tensor_error) ==
        0);

  expect_o2_error(et_o2_optimizer_builder_create_v1(
                      0u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u,
                      0u, one, &builder, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(builder == NULL);
  expect_o2_error(et_o2_optimizer_builder_create_v1(
                      1u, 99u, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one,
                      &builder, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);
  expect_o2_error(et_o2_optimizer_builder_create_v1(
                      1u, ET_O2_CLIP_GLOBAL_L2, 0u, ET_O2_SCHEDULE_CONSTANT,
                      0u, 0u, one, &builder, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);
  expect_o2_error(et_o2_optimizer_builder_create_v1(
                      1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_LINEAR, 2u, 2u,
                      one, &builder, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);

  CHECK(et_o2_optimizer_builder_create_v1(
            2u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one,
            &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameter, &identity, UINT32_C(0x3dcccccd),
            UINT32_C(0x3f666666), UINT32_C(0x3f7d70a4),
            UINT32_C(0x322bcc77), 0u, &error) == 0);
  expect_o2_error(
      et_o2_optimizer_builder_set_v1(
          builder, 1u, parameter, &identity, UINT32_C(0x3dcccccd),
          UINT32_C(0x3f666666), UINT32_C(0x3f7d70a4),
          UINT32_C(0x322bcc77), 0u, &error),
      &error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_DUPLICATE_PARAMETER);
  expect_o2_error(et_o2_optimizer_builder_finish_v1(&builder, &optimizer,
                                                    &error),
                  &error, ET_O2_STATUS_SHAPE_MISMATCH,
                  ET_O2_CODE_MISSING_PARAMETER);
  CHECK(optimizer == NULL);
  CHECK(et_o2_optimizer_builder_abort_v1(&builder, &error) == 0);
  CHECK(builder == NULL);

  /* Every scalar domain is checked from bits without implicit conversion. */
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one,
            &builder, &error) == 0);
  expect_o2_error(et_o2_optimizer_builder_set_v1(
                      builder, 0u, parameter, &identity,
                      UINT32_C(0x7f800000), UINT32_C(0x3f000000),
                      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u,
                      &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(et_o2_optimizer_builder_abort_v1(&builder, &error) == 0);

  CHECK(et_f32_parameter_destroy_v1(&parameter, &tensor_error) == 0);
  destroy_tensor(&initial);
}

static void test_exact_parameter_ceiling(void) {
  static et_f32_parameter *parameters[1365];
  static int identities[1365];
  const uint32_t one = UINT32_C(0x3f800000);
  const size_t ceiling = 1365u;
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *initial = make_tensor(&one, 1u);
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_optimizer_state *state = NULL;
  et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  fixture small;
  size_t index;

  CHECK(ET_O2_MAX_PARAMETERS == ceiling);
  et_o2_test_live_counts_snapshot_v1(&before);
  expect_o2_error(et_o2_optimizer_builder_create_v1(
                      ceiling + 1u, ET_O2_CLIP_NONE, 0u,
                      ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one, &builder,
                      &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(builder == NULL);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.builders == before.builders);
  CHECK(after.optimizers == before.optimizers);
  CHECK(after.owned_state_clones == before.owned_state_clones);

  /* An over-ceiling rejection leaves the registries usable immediately. */
  small = make_fixture(
      one, UINT32_C(0x3dcccccd), 0u, 0u, one, 0u, ET_O2_CLIP_NONE, 0u,
      ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  set_gradient(small.parameter, &one, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(small.optimizer, &error) == 0);
  CHECK(completed_updates(small.optimizer) == 1u);
  CHECK(et_o2_optimizer_zero_grad_v1(small.optimizer, &error) == 0);
  check_gradient_metadata(small.parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);

  CHECK(et_o2_optimizer_builder_create_v1(
            ceiling, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            one, &builder, &error) == 0);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.builders == before.builders + 1u);
  CHECK(after.optimizers == before.optimizers + 1u);
  for (index = 0u; index < ceiling; index++) {
    CHECK(et_f32_parameter_create_v1(initial, &parameters[index],
                                     &tensor_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(parameters[index],
                                            &identities[index],
                                            &tensor_error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, index, parameters[index], &identities[index],
              UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
              UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, &error) ==
          0);
    set_gradient(parameters[index], &one, 1u, 1u, one);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  CHECK(builder == NULL && optimizer != NULL);
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  CHECK(completed_updates(optimizer) == 1u);
  check_gradient_metadata(parameters[0], ET_F32_GRADIENT_PRESENT, 1u, one);
  check_gradient_metadata(parameters[ceiling - 1u], ET_F32_GRADIENT_PRESENT,
                          1u, one);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &error) == 0);
  check_gradient_metadata(parameters[0], ET_F32_GRADIENT_ABSENT, 0u, 0u);
  check_gradient_metadata(parameters[ceiling - 1u], ET_F32_GRADIENT_ABSENT,
                          0u, 0u);
  before.struct_size = sizeof(before);
  et_o2_test_live_counts_snapshot_v1(&before);
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, &error) == 0);
  after.struct_size = sizeof(after);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.live_states == before.live_states + 1u);
  CHECK(after.live_state_handles == before.live_state_handles + 2730u);
  CHECK(after.owned_state_clones == before.owned_state_clones + 2730u);
  CHECK(et_o2_optimizer_load_state_v1(optimizer, state, &error) == 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.live_states == before.live_states);
  CHECK(after.live_state_handles == before.live_state_handles);
  CHECK(after.owned_state_clones == before.owned_state_clones);
  destroy_tensor(&initial);
}

static void test_step_metadata_atomicity_and_zero(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t two = UINT32_C(0x40000000);
  const uint32_t expected = UINT32_C(0x3f600000); /* 1 - .25 / 2 */
  const uint32_t nan = UINT32_C(0x7fc00000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3e800000), 0u, 0u, one, 0u, ET_O2_CLIP_NONE, 0u,
      ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);

  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_ABSENT);
  CHECK(completed_updates(item.optimizer) == 0u);
  CHECK(schedule_factor(item.optimizer) == one);
  check_parameter_bits(item.parameter, &one, 1u);

  set_gradient(item.parameter, &nan, 1u, 1u, one);
  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE);
  CHECK(completed_updates(item.optimizer) == 0u);
  check_parameter_bits(item.parameter, &one, 1u);

  set_gradient(item.parameter, &one, 1u, 0u, one);
  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_METADATA);
  set_gradient(item.parameter, &one, 1u, 1u, 0u);
  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_METADATA);
  set_gradient(item.parameter, &one, 1u, 1u, UINT32_C(0x7f800000));
  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_METADATA);
  CHECK(completed_updates(item.optimizer) == 0u);
  check_parameter_bits(item.parameter, &one, 1u);

  set_gradient(item.parameter, &two, 1u, 2u, two);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(completed_updates(item.optimizer) == 1u);
  check_parameter_bits(item.parameter, &expected, 1u);
  /* A step consumes a read-only normalized view and retains the I2 slot. */
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 2u, two);

  CHECK(et_o2_optimizer_zero_grad_v1(item.optimizer, &error) == 0);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);
  CHECK(et_o2_optimizer_zero_grad_v1(item.optimizer, &error) == 0);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);
  CHECK(completed_updates(item.optimizer) == 1u);
}

static void test_present_zero_decay_and_linear_schedule(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t zero = 0u;
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), 0u, 0u, one, UINT32_C(0x3f000000),
      ET_O2_CLIP_GLOBAL_L2, one, ET_O2_SCHEDULE_LINEAR, 2u, 4u,
      UINT32_C(0x3e800000));

  CHECK(schedule_factor(item.optimizer) == UINT32_C(0x3f000000));
  set_gradient(item.parameter, &zero, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(completed_updates(item.optimizer) == 1u);
  CHECK(schedule_factor(item.optimizer) == one);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 1u, one);

  CHECK(et_o2_optimizer_zero_grad_v1(item.optimizer, &error) == 0);
  set_gradient(item.parameter, &zero, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(completed_updates(item.optimizer) == 2u);
  CHECK(schedule_factor(item.optimizer) == UINT32_C(0x3f200000));
}

static void test_global_norm_exact_boundary(void) {
  static int identities[4];
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t three = UINT32_C(0x40400000);
  const uint32_t four = UINT32_C(0x40800000);
  const uint32_t five = UINT32_C(0x40a00000);
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_parameter *parameters[4] = {NULL, NULL, NULL, NULL};
  et_o2_optimizer_builder *builders[2] = {NULL, NULL};
  et_o2_optimizer *optimizers[2] = {NULL, NULL};
  size_t index;

  for (index = 0u; index < 4u; index++) {
    et_f32_tensor *initial = make_tensor(&one, 1u);
    CHECK(et_f32_parameter_create_v1(initial, &parameters[index],
                                     &tensor_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(parameters[index],
                                            &identities[index],
                                            &tensor_error) == 0);
    destroy_tensor(&initial);
  }
  for (index = 0u; index < 2u; index++) {
    CHECK(et_o2_optimizer_builder_create_v1(
              2u, index == 0u ? ET_O2_CLIP_NONE : ET_O2_CLIP_GLOBAL_L2,
              index == 0u ? 0u : five, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
              one, &builders[index], &error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
              builders[index], 0u, parameters[index * 2u],
              &identities[index * 2u], UINT32_C(0x3e800000), 0u, 0u, one,
              0u, &error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
              builders[index], 1u, parameters[index * 2u + 1u],
              &identities[index * 2u + 1u], UINT32_C(0x3e800000), 0u, 0u,
              one, 0u, &error) == 0);
    CHECK(et_o2_optimizer_builder_finish_v1(
              &builders[index], &optimizers[index], &error) == 0);
  }
  set_gradient(parameters[0], &three, 1u, 1u, one);
  set_gradient(parameters[1], &four, 1u, 1u, one);
  set_gradient(parameters[2], &three, 1u, 1u, one);
  set_gradient(parameters[3], &four, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(optimizers[0], &error) == 0);
  CHECK(et_o2_optimizer_step_v1(optimizers[1], &error) == 0);
  {
    et_f32_tensor *unclipped = NULL;
    et_f32_tensor *boundary = NULL;
    int32_t equal = 0;
    CHECK(et_f32_parameter_value_snapshot_v1(parameters[0], &unclipped,
                                             &tensor_error) == 0);
    CHECK(et_f32_parameter_value_snapshot_v1(parameters[2], &boundary,
                                             &tensor_error) == 0);
    CHECK(et_f32_tensor_bits_equal_v1(unclipped, boundary, &equal,
                                      &tensor_error) == 0);
    CHECK(equal == 1);
    destroy_tensor(&unclipped);
    destroy_tensor(&boundary);
    CHECK(et_f32_parameter_value_snapshot_v1(parameters[1], &unclipped,
                                             &tensor_error) == 0);
    CHECK(et_f32_parameter_value_snapshot_v1(parameters[3], &boundary,
                                             &tensor_error) == 0);
    CHECK(et_f32_tensor_bits_equal_v1(unclipped, boundary, &equal,
                                      &tensor_error) == 0);
    CHECK(equal == 1);
    destroy_tensor(&unclipped);
    destroy_tensor(&boundary);
  }
}

static void test_clip_ratio_gradual_underflow_to_positive_zero(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t two = UINT32_C(0x40000000);
  const uint32_t half = UINT32_C(0x3f000000);
  const uint32_t minimum_subnormal = UINT32_C(0x00000001);
  const uint32_t maximum_finite = UINT32_C(0x7f7fffff);
  const uint32_t expected_parameter = UINT32_C(0x3fc00000);
  const uint32_t positive_zero = 0u;
  et_o2_error_v1 error;
  fixture item = make_fixture(
      two, half, half, half, one, half, ET_O2_CLIP_GLOBAL_L2,
      minimum_subnormal, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  uint32_t moment = UINT32_MAX;

  set_gradient(item.parameter, &maximum_finite, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(completed_updates(item.optimizer) == 1u);
  CHECK(schedule_factor(item.optimizer) == one);
  check_parameter_bits(item.parameter, &expected_parameter, 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            item.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &moment, 1u, &error) ==
        0);
  CHECK(moment == positive_zero);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            item.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &moment, 1u,
            &error) == 0);
  CHECK(moment == positive_zero);
  check_gradient_bits(item.parameter, &maximum_finite, 1u);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 1u, one);
}

static void test_global_norm_above_multi_tensor_frozen_parity(void) {
  static int identities[2];
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t initial0[] = {one, one};
  const uint32_t initial1[] = {one, one, one};
  const uint32_t gradient0[] = {UINT32_C(0x40400000),
                                UINT32_C(0x40800000)};
  const uint32_t gradient1[] = {0u, UINT32_C(0x41400000),
                                UINT32_C(0x80000000)};
  const uint32_t expected_parameter0[] = {UINT32_C(0x3f400000),
                                          UINT32_C(0x3f400000)};
  const uint32_t expected_parameter1[] = {one, UINT32_C(0x3f400000), one};
  const uint32_t expected_avg0[] = {UINT32_C(0x3fc00000),
                                    UINT32_C(0x40000000)};
  const uint32_t expected_avg1[] = {0u, UINT32_C(0x40c00000), 0u};
  const uint32_t expected_avg_sq0[] = {UINT32_C(0x40100000),
                                       UINT32_C(0x40800000)};
  const uint32_t expected_avg_sq1[] = {0u, UINT32_C(0x42100000), 0u};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *initial_tensor0 = make_tensor(initial0, 2u);
  et_f32_tensor *initial_tensor1 = make_tensor(initial1, 3u);
  et_f32_parameter *parameters[2] = {NULL, NULL};
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  uint32_t actual[3];

  CHECK(et_f32_parameter_create_v1(initial_tensor0, &parameters[0],
                                   &tensor_error) == 0);
  CHECK(et_f32_parameter_create_v1(initial_tensor1, &parameters[1],
                                   &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameters[0], &identities[0],
                                          &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameters[1], &identities[1],
                                          &tensor_error) == 0);
  destroy_tensor(&initial_tensor0);
  destroy_tensor(&initial_tensor1);
  CHECK(et_o2_optimizer_builder_create_v1(
            2u, ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x40d00000),
            ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one, &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameters[0], &identities[0],
            UINT32_C(0x3e800000), 0u, 0u, 1u, 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 1u, parameters[1], &identities[1],
            UINT32_C(0x3e800000), 0u, 0u, 1u, 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  set_gradient(parameters[0], gradient0, 2u, 7u, one);
  set_gradient(parameters[1], gradient1, 3u, 7u, one);

  /* The unique-tensor norm is 13, so max=6.5 scales every gradient by 0.5. */
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  CHECK(completed_updates(optimizer) == 1u);
  check_parameter_bits(parameters[0], expected_parameter0, 2u);
  check_parameter_bits(parameters[1], expected_parameter1, 3u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected_avg0, sizeof(expected_avg0)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected_avg_sq0, sizeof(expected_avg_sq0)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 1u, ET_O2_MOMENT_EXP_AVG, actual, 3u, &error) == 0);
  CHECK(memcmp(actual, expected_avg1, sizeof(expected_avg1)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 1u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 3u, &error) == 0);
  CHECK(memcmp(actual, expected_avg_sq1, sizeof(expected_avg_sq1)) == 0);
  check_gradient_bits(parameters[0], gradient0, 2u);
  check_gradient_bits(parameters[1], gradient1, 3u);
  check_gradient_metadata(parameters[0], ET_F32_GRADIENT_PRESENT, 7u, one);
  check_gradient_metadata(parameters[1], ET_F32_GRADIENT_PRESENT, 7u, one);
}

static void test_global_norm_scaled_accumulation_avoids_sum_overflow(void) {
  static int identity;
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t initial[] = {one, one};
  const uint32_t gradient[] = {UINT32_C(0x5f502ab5),
                               UINT32_C(0x5f502ab5)};
  const uint32_t expected_avg[] = {UINT32_C(0x5ed02ab5),
                                   UINT32_C(0x5ed02ab5)};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *initial_tensor = make_tensor(initial, 2u);
  et_f32_parameter *parameter = NULL;
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  uint32_t actual[2];

  CHECK(et_f32_parameter_create_v1(initial_tensor, &parameter,
                                   &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity,
                                          &tensor_error) == 0);
  destroy_tensor(&initial_tensor);
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x7f7fffff),
            ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one, &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameter, &identity, 1u, UINT32_C(0x3f000000),
            UINT32_C(0x3f000000), one, 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  set_gradient(parameter, gradient, 2u, 1u, one);

  /* Each square is finite but their naive binary32 sum overflows. */
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  CHECK(completed_updates(optimizer) == 1u);
  check_parameter_bits(parameter, initial, 2u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected_avg, sizeof(expected_avg)) == 0);
  check_gradient_bits(parameter, gradient, 2u);
  check_gradient_metadata(parameter, ET_F32_GRADIENT_PRESENT, 1u, one);
}

static void test_distinct_parameter_groups_frozen_parity(void) {
  static int identities[2];
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t initial0[] = {UINT32_C(0x40000000), UINT32_C(0xbf800000)};
  const uint32_t initial1[] = {UINT32_C(0x3e800000), UINT32_C(0xbf400000),
                               UINT32_C(0x3fc00000)};
  const uint32_t gradient0[] = {UINT32_C(0x3f000000), UINT32_C(0xbe800000)};
  const uint32_t gradient1[] = {UINT32_C(0xbf800000), 0u,
                                UINT32_C(0x3e000000)};
  const uint32_t expected_p0[] = {UINT32_C(0x3ffd70a4),
                                  UINT32_C(0xbf7ae149)};
  const uint32_t expected_m0[] = {UINT32_C(0x3dcccccc),
                                  UINT32_C(0xbd4ccccc)};
  const uint32_t expected_v0[] = {UINT32_C(0x3c4cccd0),
                                  UINT32_C(0x3b4cccd0)};
  const uint32_t expected_p1[] = {UINT32_C(0x3e826e96),
                                  UINT32_C(0xbf3fced9),
                                  UINT32_C(0x3fbf2b05)};
  const uint32_t expected_m1[] = {UINT32_C(0xbe99999a), 0u,
                                  UINT32_C(0x3d19999a)};
  const uint32_t expected_v1[] = {UINT32_C(0x3dccccd0), 0u,
                                  UINT32_C(0x3accccd0)};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *tensor0 = make_tensor(initial0, 2u);
  et_f32_tensor *tensor1 = make_tensor(initial1, 3u);
  et_f32_parameter *parameters[2] = {NULL, NULL};
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  uint32_t actual[3];

  CHECK(et_f32_parameter_create_v1(tensor0, &parameters[0], &tensor_error) ==
        0);
  CHECK(et_f32_parameter_create_v1(tensor1, &parameters[1], &tensor_error) ==
        0);
  CHECK(et_f32_parameter_bind_identity_v1(parameters[0], &identities[0],
                                          &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameters[1], &identities[1],
                                          &tensor_error) == 0);
  destroy_tensor(&tensor0);
  destroy_tensor(&tensor1);
  CHECK(et_o2_optimizer_builder_create_v1(
            2u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one,
            &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameters[0], &identities[0],
            UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
            UINT32_C(0x3f733333), UINT32_C(0x358637bd), 0u, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 1u, parameters[1], &identities[1],
            UINT32_C(0x3ba3d70a), UINT32_C(0x3f333333),
            UINT32_C(0x3f666666), UINT32_C(0x3727c5ac),
            UINT32_C(0x3e4ccccd), &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  set_gradient(parameters[0], gradient0, 2u, 1u, one);
  set_gradient(parameters[1], gradient1, 3u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  check_parameter_bits(parameters[0], expected_p0, 2u);
  check_parameter_bits(parameters[1], expected_p1, 3u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected_m0, sizeof(expected_m0)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 2u, &error) ==
        0);
  CHECK(memcmp(actual, expected_v0, sizeof(expected_v0)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 1u, ET_O2_MOMENT_EXP_AVG, actual, 3u, &error) == 0);
  CHECK(memcmp(actual, expected_m1, sizeof(expected_m1)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 1u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 3u, &error) ==
        0);
  CHECK(memcmp(actual, expected_v1, sizeof(expected_v1)) == 0);
}

static void test_unequal_weight_accumulation_frozen_parity(void) {
  static int identity;
  const uint32_t initial[] = {UINT32_C(0x3f000000), UINT32_C(0xbf800000)};
  const uint32_t numerator0[] = {UINT32_C(0xbfc00000),
                                 UINT32_C(0xc0400000)};
  const uint32_t numerator1[] = {UINT32_C(0xc0560000),
                                 UINT32_C(0xc0e80000)};
  const uint32_t expected_parameter[] = {UINT32_C(0x3f026e98),
                                         UINT32_C(0xbf7d2f1b)};
  const uint32_t expected_m[] = {UINT32_C(0xbe255558),
                                 UINT32_C(0xbeaeeef2)};
  const uint32_t expected_v[] = {UINT32_C(0x3b2ad79e),
                                 UINT32_C(0x3c3f420c)};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_tensor *initial_tensor = make_tensor(initial, 2u);
  et_f32_parameter *parameter = NULL;
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  uint32_t actual[2];

  CHECK(et_f32_parameter_create_v1(initial_tensor, &parameter,
                                   &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity,
                                          &tensor_error) == 0);
  destroy_tensor(&initial_tensor);
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            UINT32_C(0x3f800000), &builder, &error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, parameter, &identity, UINT32_C(0x3c23d70a),
            UINT32_C(0x3f666666), UINT32_C(0x3f7fbe77),
            UINT32_C(0x322bcc77), UINT32_C(0x3dcccccd), &error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  accumulate_gradient(parameter, numerator0, 2u, 0u, UINT32_C(0x3f800000));
  accumulate_gradient(parameter, numerator1, 2u, 1u, UINT32_C(0x40000000));
  check_gradient_metadata(parameter, ET_F32_GRADIENT_PRESENT, 2u,
                          UINT32_C(0x40400000));
  CHECK(et_o2_optimizer_step_v1(optimizer, &error) == 0);
  check_parameter_bits(parameter, expected_parameter, 2u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG, actual, 2u, &error) == 0);
  CHECK(memcmp(actual, expected_m, sizeof(expected_m)) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, actual, 2u, &error) ==
        0);
  CHECK(memcmp(actual, expected_v, sizeof(expected_v)) == 0);
  check_gradient_metadata(parameter, ET_F32_GRADIENT_PRESENT, 2u,
                          UINT32_C(0x40400000));
}

static void test_allocation_failures_are_atomic(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  size_t allowed;
  int saw_failure = 0;
  int saw_success = 0;

  set_gradient(item.parameter, &one, 1u, 1u, one);
  for (allowed = 0u; allowed < 128u; allowed++) {
    int32_t result;
    et_o2_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_step_v1(item.optimizer, &error);
    et_o2_test_reset_failpoints_v1();
    if (result == 0) {
      saw_success = 1;
      CHECK(completed_updates(item.optimizer) == 1u);
      break;
    }
    saw_failure = 1;
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(completed_updates(item.optimizer) == 0u);
    check_parameter_bits(item.parameter, &one, 1u);
    check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 1u, one);
  }
  CHECK(saw_failure != 0);
  CHECK(saw_success != 0);
}

static void test_i2_step_allocation_failures_are_atomic(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  size_t allowed;
  int saw_failure = 0;
  int saw_success = 0;

  set_gradient(item.parameter, &one, 1u, 1u, one);
  for (allowed = 0u; allowed < 128u; allowed++) {
    int32_t result;
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_step_v1(item.optimizer, &error);
    et_f32_tensor_test_reset_allocator_v1();
    if (result == 0) {
      saw_success = 1;
      CHECK(completed_updates(item.optimizer) == 1u);
      break;
    }
    saw_failure = 1;
    if (error.category != ET_O2_STATUS_INTERNAL) {
      (void)fprintf(stderr,
                    "I2 allocation failpoint %zu mapped to %u/%u (%s: %s)\n",
                    allowed, error.category, error.code, error.operation,
                    error.message);
    }
    CHECK(error.category == ET_O2_STATUS_INTERNAL);
    CHECK(completed_updates(item.optimizer) == 0u);
    check_parameter_bits(item.parameter, &one, 1u);
    check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 1u, one);
  }
  CHECK(saw_failure != 0);
  CHECK(saw_success != 0);
}

static void test_counter_overflow_is_atomic(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  set_gradient(item.parameter, &one, 1u, 1u, one);
  et_o2_test_optimizer_set_completed_updates_v1(
      item.optimizer, (uint64_t)INT64_MAX);
  expect_o2_error(et_o2_optimizer_step_v1(item.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_COUNTER_OVERFLOW);
  CHECK(completed_updates(item.optimizer) == (uint64_t)INT64_MAX);
  CHECK(schedule_factor(item.optimizer) == one);
  check_parameter_bits(item.parameter, &one, 1u);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 1u, one);
}

static uint32_t state_moment_bits(et_o2_optimizer_state *state, size_t index,
                                  uint32_t kind) {
  et_o2_error_v1 error;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  uint32_t result = UINT32_MAX;
  CHECK(et_o2_optimizer_state_moment_handle_v1(state, index, kind, &handle,
                                                &error) == 0);
  CHECK(handle != NULL);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                               &error) == 0);
  CHECK(borrow != NULL);
  CHECK(view != NULL);
  CHECK(view->byte_length == sizeof(result));
  memcpy(&result, view->data, sizeof(result));
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
  return result;
}

static void test_state_round_trip_and_release(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t gradient = UINT32_C(0x3f000000);
  et_o2_error_v1 error;
  fixture source = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  fixture target;
  et_o2_optimizer_state *state = NULL;
  et_o2_optimizer_state *sibling = NULL;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  size_t entry_count = 0u;
  uint64_t state_updates = UINT64_MAX;
  uint32_t option = 0u;
  uint32_t source_m;
  uint32_t source_v;
  uint32_t snapshot_m;
  uint32_t boundary_parameter;
  uint32_t source_parameter;
  uint32_t target_parameter;
  const uint32_t next_gradient = UINT32_C(0xbe800000);
  et_f32_test_live_counts_v1 i2_baseline = {.struct_size =
                                                sizeof(i2_baseline)};
  et_f32_test_live_counts_v1 i2_current = {.struct_size =
                                               sizeof(i2_current)};
  size_t allowed;
  int saw_load_failure = 0;
  int saw_load_success = 0;
  int saw_i2_load_failure = 0;
  int saw_i2_load_success = 0;

  set_gradient(source.parameter, &gradient, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(source.optimizer, &error) == 0);
  read_parameter_bits(source.parameter, &boundary_parameter, 1u);
  target = make_fixture(
      boundary_parameter, UINT32_C(0x3d4ccccd), UINT32_C(0x3e800000),
      UINT32_C(0x3e800000), UINT32_C(0x3dcccccd), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &state, &error) ==
        0);
  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &sibling, &error) ==
        0);
  CHECK(state != NULL && sibling != NULL && state != sibling);
  CHECK(et_o2_optimizer_state_entry_count_v1(state, &entry_count, &error) == 0);
  CHECK(entry_count == 1u);
  CHECK(et_o2_optimizer_state_completed_updates_v1(state, &state_updates,
                                                    &error) == 0);
  CHECK(state_updates == 1u);
  CHECK(et_o2_optimizer_state_option_bits_v1(
            state, 0u, ET_O2_OPTION_LEARNING_RATE, &option, &error) == 0);
  CHECK(option == UINT32_C(0x3dcccccd));
  source_m = state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG);
  source_v = state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG_SQ);
  snapshot_m = source_m;

  for (allowed = 0u; allowed < 128u; allowed++) {
    int32_t result;
    uint32_t target_m = UINT32_MAX;
    uint32_t target_v = UINT32_MAX;
    et_o2_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_load_state_v1(target.optimizer, state, &error);
    et_o2_test_reset_failpoints_v1();
    if (result == 0) {
      saw_load_success = 1;
      break;
    }
    saw_load_failure = 1;
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(completed_updates(target.optimizer) == 0u);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &target_m, 1u,
              &error) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &target_v, 1u,
              &error) == 0);
    CHECK(target_m == 0u && target_v == 0u);
    CHECK(state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG) == source_m);
    CHECK(state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG_SQ) == source_v);
  }
  CHECK(saw_load_failure != 0);
  CHECK(saw_load_success != 0);
  CHECK(completed_updates(target.optimizer) == 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &option, 1u,
            &error) == 0);
  CHECK(option == source_m);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &option, 1u,
            &error) == 0);
  CHECK(option == source_v);

  /* I2 copy-plan allocation failures also leave the installed state exact. */
  et_f32_test_live_counts_snapshot_v1(&i2_baseline);
  for (allowed = 0u; allowed < 128u; allowed++) {
    int32_t result;
    uint32_t target_m = UINT32_MAX;
    uint32_t target_v = UINT32_MAX;
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_load_state_v1(target.optimizer, state, &error);
    et_f32_tensor_test_reset_allocator_v1();
    if (result == 0) {
      saw_i2_load_success = 1;
      break;
    }
    saw_i2_load_failure = 1;
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(completed_updates(target.optimizer) == 1u);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &target_m, 1u,
              &error) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &target_v, 1u,
              &error) == 0);
    CHECK(target_m == source_m && target_v == source_v);
    i2_current.struct_size = sizeof(i2_current);
    et_f32_test_live_counts_snapshot_v1(&i2_current);
    CHECK(i2_current.tensors == i2_baseline.tensors);
    CHECK(i2_current.borrows == i2_baseline.borrows);
    CHECK(i2_current.copy_plans == i2_baseline.copy_plans);
    CHECK(i2_current.owned_clones == i2_baseline.owned_clones);
  }
  CHECK(saw_i2_load_failure != 0);
  CHECK(saw_i2_load_success != 0);

  set_gradient(source.parameter, &next_gradient, 1u, 1u, one);
  set_gradient(target.parameter, &next_gradient, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_step_v1(target.optimizer, &error) == 0);
  CHECK(completed_updates(source.optimizer) == 2u);
  CHECK(completed_updates(target.optimizer) == 2u);
  CHECK(schedule_factor(source.optimizer) == schedule_factor(target.optimizer));
  read_parameter_bits(source.parameter, &source_parameter, 1u);
  read_parameter_bits(target.parameter, &target_parameter, 1u);
  CHECK(source_parameter == target_parameter);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            source.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &source_m, 1u,
            &error) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &option, 1u,
            &error) == 0);
  CHECK(source_m == option);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            source.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &source_v, 1u,
            &error) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &option, 1u,
            &error) == 0);
  CHECK(source_v == option);

  CHECK(et_o2_optimizer_state_moment_handle_v1(
            state, 0u, ET_O2_MOMENT_EXP_AVG, &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                               &error) == 0);
  expect_o2_error(et_o2_optimizer_state_release_v1(state, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  /* The exact registered dead receiver is the sole idempotent release case. */
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  expect_o2_error(et_o2_optimizer_state_entry_count_v1(
                      state, &entry_count, &error),
                  &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  expect_o2_error(et_o2_optimizer_state_borrow_begin_v1(
                      state, handle, &borrow, &view, &error),
                  &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);

  /* Releasing one clone cannot affect a sibling or live optimizer storage. */
  CHECK(state_moment_bits(sibling, 0u, ET_O2_MOMENT_EXP_AVG) == snapshot_m);
  CHECK(completed_updates(source.optimizer) == 2u);
  CHECK(completed_updates(target.optimizer) == 2u);
  CHECK(et_o2_optimizer_state_release_v1(sibling, &error) == 0);

  CHECK(et_o2_optimizer_zero_grad_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(target.optimizer, &error) == 0);
  set_gradient(source.parameter, &one, 1u, 1u, one);
  set_gradient(target.parameter, &one, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_step_v1(target.optimizer, &error) == 0);
  read_parameter_bits(source.parameter, &source_parameter, 1u);
  read_parameter_bits(target.parameter, &target_parameter, 1u);
  CHECK(source_parameter == target_parameter);
  CHECK(completed_updates(source.optimizer) == 3u);
  CHECK(completed_updates(target.optimizer) == 3u);
}

static void test_state_load_restores_schedule_clip_and_next_update(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t first_gradient = UINT32_C(0x3f000000);
  const uint32_t second_gradient = UINT32_C(0xbf400000);
  const uint32_t next_gradient = one;
  et_o2_error_v1 error;
  fixture source = make_fixture(
      one, UINT32_C(0x3c23d70a), UINT32_C(0x3f666666),
      UINT32_C(0x3f7fbe77), UINT32_C(0x322bcc77),
      UINT32_C(0x3dcccccd), ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x3e800000),
      ET_O2_SCHEDULE_LINEAR, 2u, 6u, UINT32_C(0x3dcccccd));
  fixture target;
  et_o2_optimizer_state *state = NULL;
  uint32_t boundary_parameter;
  uint32_t before_load_parameter;
  uint32_t after_load_parameter;
  uint32_t source_parameter;
  uint32_t target_parameter;
  uint32_t source_m;
  uint32_t target_m;
  uint32_t source_v;
  uint32_t target_v;

  set_gradient(source.parameter, &first_gradient, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(source.optimizer, &error) == 0);
  set_gradient(source.parameter, &second_gradient, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(source.optimizer, &error) == 0);
  CHECK(completed_updates(source.optimizer) == 2u);
  CHECK(schedule_factor(source.optimizer) == UINT32_C(0x3f466666));
  read_parameter_bits(source.parameter, &boundary_parameter, 1u);

  target = make_fixture(
      boundary_parameter, UINT32_C(0x3e4ccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3dcccccd), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  set_gradient(target.parameter, &next_gradient, 1u, 9u,
               UINT32_C(0x40000000));
  CHECK(et_o2_optimizer_zero_grad_v1(target.optimizer, &error) == 0);
  check_gradient_metadata(target.parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);
  read_parameter_bits(target.parameter, &before_load_parameter, 1u);

  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &state, &error) ==
        0);
  CHECK(et_o2_optimizer_load_state_v1(target.optimizer, state, &error) == 0);
  read_parameter_bits(target.parameter, &after_load_parameter, 1u);
  CHECK(after_load_parameter == before_load_parameter);
  check_gradient_metadata(target.parameter, ET_F32_GRADIENT_ABSENT, 0u, 0u);
  CHECK(completed_updates(target.optimizer) == 2u);
  CHECK(schedule_factor(target.optimizer) == UINT32_C(0x3f466666));

  set_gradient(source.parameter, &next_gradient, 1u, 3u,
               UINT32_C(0x40000000));
  set_gradient(target.parameter, &next_gradient, 1u, 3u,
               UINT32_C(0x40000000));
  CHECK(et_o2_optimizer_step_v1(source.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_step_v1(target.optimizer, &error) == 0);
  CHECK(completed_updates(source.optimizer) == 3u);
  CHECK(completed_updates(target.optimizer) == 3u);
  CHECK(schedule_factor(source.optimizer) == UINT32_C(0x3f0ccccd));
  CHECK(schedule_factor(target.optimizer) == UINT32_C(0x3f0ccccd));
  read_parameter_bits(source.parameter, &source_parameter, 1u);
  read_parameter_bits(target.parameter, &target_parameter, 1u);
  CHECK(source_parameter == target_parameter);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            source.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &source_m, 1u,
            &error) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &target_m, 1u,
            &error) == 0);
  CHECK(source_m == target_m);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            source.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &source_v, 1u,
            &error) == 0);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &target_v, 1u,
            &error) == 0);
  CHECK(source_v == target_v);
  check_gradient_bits(source.parameter, &next_gradient, 1u);
  check_gradient_bits(target.parameter, &next_gradient, 1u);
  check_gradient_metadata(source.parameter, ET_F32_GRADIENT_PRESENT, 3u,
                          UINT32_C(0x40000000));
  check_gradient_metadata(target.parameter, ET_F32_GRADIENT_PRESENT, 3u,
                          UINT32_C(0x40000000));
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
}

static void test_cross_owner_handle_and_snapshot_failpoints(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *left = NULL;
  et_o2_optimizer_state *right = NULL;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_o2_test_live_counts_v1 baseline = {.struct_size = sizeof(baseline)};
  et_o2_test_live_counts_v1 current = {.struct_size = sizeof(current)};
  size_t allowed;
  int saw_failure = 0;
  int saw_success = 0;

  et_o2_test_live_counts_snapshot_v1(&baseline);
  for (allowed = 0u; allowed < 128u; allowed++) {
    et_o2_optimizer_state *attempt = NULL;
    int32_t result;
    et_o2_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_state_snapshot_v1(item.optimizer, &attempt,
                                                &error);
    et_o2_test_reset_failpoints_v1();
    if (result == 0) {
      saw_success = 1;
      left = attempt;
      break;
    }
    saw_failure = 1;
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(attempt == NULL);
    et_o2_test_live_counts_snapshot_v1(&current);
    CHECK(current.live_states == baseline.live_states);
    CHECK(current.live_state_handles == baseline.live_state_handles);
    CHECK(current.state_borrows == baseline.state_borrows);
    CHECK(current.owned_state_clones == baseline.owned_state_clones);
  }
  CHECK(saw_failure != 0);
  CHECK(saw_success != 0);
  CHECK(et_o2_optimizer_state_release_v1(left, &error) == 0);
  left = NULL;
  baseline.struct_size = sizeof(baseline);
  et_o2_test_live_counts_snapshot_v1(&baseline);
  saw_failure = 0;
  saw_success = 0;
  for (allowed = 0u; allowed < 128u; allowed++) {
    et_o2_optimizer_state *attempt = NULL;
    int32_t result;
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    result = et_o2_optimizer_state_snapshot_v1(item.optimizer, &attempt,
                                                &error);
    et_f32_tensor_test_reset_allocator_v1();
    if (result == 0) {
      saw_success = 1;
      left = attempt;
      break;
    }
    saw_failure = 1;
    CHECK(error.category == ET_O2_STATUS_INTERNAL);
    CHECK(attempt == NULL);
    current.struct_size = sizeof(current);
    et_o2_test_live_counts_snapshot_v1(&current);
    CHECK(current.live_states == baseline.live_states);
    CHECK(current.live_state_handles == baseline.live_state_handles);
    CHECK(current.state_borrows == baseline.state_borrows);
    CHECK(current.owned_state_clones == baseline.owned_state_clones);
  }
  CHECK(saw_failure != 0);
  CHECK(saw_success != 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(item.optimizer, &right, &error) ==
        0);
  CHECK(et_o2_optimizer_state_moment_handle_v1(
            left, 0u, ET_O2_MOMENT_EXP_AVG, &handle, &error) == 0);
  expect_o2_error(et_o2_optimizer_state_borrow_begin_v1(
                      right, handle, &borrow, &view, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_HANDLE);
  CHECK(borrow == NULL && view == NULL);
  CHECK(et_o2_optimizer_state_release_v1(left, &error) == 0);
  CHECK(et_o2_optimizer_state_release_v1(right, &error) == 0);
}

static void test_nonfinite_snapshot_load_rejected_atomically(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t nan = UINT32_C(0x7fc00000);
  const uint32_t infinities[] = {UINT32_C(0x7f800000),
                                 UINT32_C(0xff800000)};
  const uint32_t negative_one = UINT32_C(0xbf800000);
  const uint32_t zero = 0u;
  et_o2_error_v1 error;
  fixture source = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  fixture target = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *state = NULL;
  et_o2_optimizer_state_handle *handle = NULL;
  et_o2_optimizer_state_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  uint32_t moment = UINT32_MAX;
  size_t nonfinite_index;

  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &state, &error) ==
        0);
  CHECK(et_o2_optimizer_state_moment_handle_v1(
            state, 0u, ET_O2_MOMENT_EXP_AVG, &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                               &error) == 0);
  CHECK(view != NULL && view->byte_length == sizeof(nan));
  memcpy(view->data, &nan, sizeof(nan));
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA, ET_O2_CODE_NONFINITE);
  CHECK(completed_updates(target.optimizer) == 0u);
  check_parameter_bits(target.parameter, &one, 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &moment, 1u,
            &error) == 0);
  CHECK(moment == 0u);
  CHECK(state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG) == nan);

  for (nonfinite_index = 0u;
       nonfinite_index < sizeof(infinities) / sizeof(infinities[0]);
       nonfinite_index++) {
    view = NULL;
    CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                                 &error) == 0);
    memcpy(view->data, &infinities[nonfinite_index],
           sizeof(infinities[nonfinite_index]));
    CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
    expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                  &error),
                    &error, ET_O2_STATUS_CORRUPT_DATA,
                    ET_O2_CODE_NONFINITE);
    CHECK(completed_updates(target.optimizer) == 0u);
  }

  handle = NULL;
  view = NULL;
  CHECK(et_o2_optimizer_state_moment_handle_v1(
            state, 0u, ET_O2_MOMENT_EXP_AVG, &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                               &error) == 0);
  memcpy(view->data, &zero, sizeof(zero));
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  handle = NULL;
  view = NULL;
  CHECK(et_o2_optimizer_state_moment_handle_v1(
            state, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &handle, &error) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(state, handle, &borrow, &view,
                                               &error) == 0);
  memcpy(view->data, &negative_one, sizeof(negative_one));
  CHECK(et_o2_optimizer_state_borrow_end_v1(&borrow, &error) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA,
                  ET_O2_CODE_INVALID_MOMENT);
  CHECK(completed_updates(target.optimizer) == 0u);
  check_parameter_bits(target.parameter, &one, 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ, &moment, 1u,
            &error) == 0);
  CHECK(moment == 0u);
  CHECK(state_moment_bits(state, 0u, ET_O2_MOMENT_EXP_AVG_SQ) ==
        negative_one);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
}

static void test_protected_state_metadata_load_validation(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture source = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  fixture target = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *state = NULL;
  uint32_t moment = UINT32_MAX;

  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &state, &error) ==
        0);
  CHECK(et_o2_test_state_set_config_v1(
            state, 99u, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(et_o2_test_state_set_config_v1(
            state, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            one) == 0);

  CHECK(et_o2_test_state_set_option_bits_v1(
            state, 0u, ET_O2_OPTION_LEARNING_RATE,
            UINT32_C(0x7f800000)) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(et_o2_test_state_set_option_bits_v1(
            state, 0u, ET_O2_OPTION_LEARNING_RATE,
            UINT32_C(0x3dcccccd)) == 0);

  CHECK(et_o2_test_state_set_completed_updates_v1(
            state, (uint64_t)INT64_MAX + UINT64_C(1)) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA,
                  ET_O2_CODE_COUNTER_OVERFLOW);
  CHECK(et_o2_test_state_set_completed_updates_v1(state, 0u) == 0);
  CHECK(completed_updates(target.optimizer) == 0u);
  check_parameter_bits(target.parameter, &one, 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            target.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &moment, 1u,
            &error) == 0);
  CHECK(moment == 0u);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
}

static void test_load_requires_absent_destination_before_state_borrow(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture source = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  fixture target = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *state = NULL;

  CHECK(et_o2_optimizer_state_snapshot_v1(source.optimizer, &state, &error) ==
        0);
  CHECK(et_o2_test_state_set_config_v1(
            state, 99u, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one) == 0);
  set_gradient(target.parameter, &one, 1u, 5u, one);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_INVALID_STATE,
                  ET_O2_CODE_GRADIENT_METADATA);
  check_gradient_bits(target.parameter, &one, 1u);
  check_gradient_metadata(target.parameter, ET_F32_GRADIENT_PRESENT, 5u, one);
  CHECK(et_o2_optimizer_zero_grad_v1(target.optimizer, &error) == 0);
  expect_o2_error(et_o2_optimizer_load_state_v1(target.optimizer, state,
                                                &error),
                  &error, ET_O2_STATUS_CORRUPT_DATA,
                  ET_O2_CODE_INVALID_OPTION);
  CHECK(et_o2_test_state_set_config_v1(
            state, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u,
            one) == 0);
  CHECK(et_o2_optimizer_load_state_v1(target.optimizer, state, &error) == 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
}

static void test_release_defect_finishes_exact_once_tail(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *state = NULL;
  et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_o2_test_live_counts_v1 dead = {.struct_size = sizeof(dead)};
  size_t entry_count = 0u;

  et_o2_test_live_counts_snapshot_v1(&before);
  CHECK(et_o2_optimizer_state_snapshot_v1(item.optimizer, &state, &error) ==
        0);
  et_o2_test_live_counts_snapshot_v1(&live);
  CHECK(live.live_states == before.live_states + 1u);
  CHECK(live.owned_state_clones == before.owned_state_clones + 2u);

  et_o2_test_fail_release_after_v1(0u);
  expect_o2_error(et_o2_optimizer_state_release_v1(state, &error), &error,
                  ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT);
  et_o2_test_reset_failpoints_v1();
  et_o2_test_live_counts_snapshot_v1(&dead);
  CHECK(dead.live_states == before.live_states);
  CHECK(dead.dead_states == before.dead_states + 1u);
  CHECK(dead.owned_state_clones == before.owned_state_clones);
  expect_o2_error(et_o2_optimizer_state_entry_count_v1(
                      state, &entry_count, &error),
                  &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  /* No second callback: the exact native dead receiver succeeds despite the
   * still-armed provider-defect failpoint. */
  et_o2_test_fail_release_after_v1(0u);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  et_o2_test_reset_failpoints_v1();
}

static void test_release_admission_corruption_is_nonmutating(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *state = NULL;
  et_o2_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};

  CHECK(et_o2_optimizer_state_snapshot_v1(item.optimizer, &state, &error) ==
        0);
  et_o2_test_live_counts_snapshot_v1(&live);
  et_o2_test_state_set_provider_version_v1(state, 99u, 0u);
  expect_o2_error(et_o2_optimizer_state_release_v1(state, &error), &error,
                  ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_MISMATCH);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.live_states == live.live_states);
  CHECK(after.owned_state_clones == live.owned_state_clones);
  et_o2_test_state_set_provider_version_v1(
      state, ET_O2_PROVIDER_ABI_MAJOR, ET_O2_PROVIDER_ABI_MINOR);

  et_o2_test_state_set_owned_clone_count_v1(state, 1u);
  expect_o2_error(et_o2_optimizer_state_release_v1(state, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(after.live_states == live.live_states);
  CHECK(after.owned_state_clones == live.owned_state_clones);
  et_o2_test_state_set_owned_clone_count_v1(state, 2u);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
}

static void test_release_exact_owner_ledger_is_nonmutating(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  const uint32_t first_gradient = UINT32_C(0x3f000000);
  const uint32_t next_gradient = UINT32_C(0xbf000000);
  et_o2_error_v1 error;
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);
  et_o2_optimizer_state *left = NULL;
  et_o2_optimizer_state *right = NULL;
  et_o2_test_live_counts_v1 o2_before = {.struct_size = sizeof(o2_before)};
  et_o2_test_live_counts_v1 o2_snapshots = {
      .struct_size = sizeof(o2_snapshots)};
  et_o2_test_live_counts_v1 o2_live = {.struct_size = sizeof(o2_live)};
  et_o2_test_live_counts_v1 o2_after = {.struct_size = sizeof(o2_after)};
  et_f32_test_live_counts_v1 i2_before = {.struct_size = sizeof(i2_before)};
  et_f32_test_live_counts_v1 i2_snapshots = {
      .struct_size = sizeof(i2_snapshots)};
  et_f32_test_live_counts_v1 i2_live = {.struct_size = sizeof(i2_live)};
  et_f32_test_live_counts_v1 i2_after = {.struct_size = sizeof(i2_after)};
  uint32_t parameter_before;
  uint32_t optimizer_moment_before;
  uint32_t left_moment;
  uint32_t right_moment;
  uint32_t lifecycle = 0u;

  set_gradient(item.parameter, &first_gradient, 1u, 1u, one);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(item.optimizer, &error) == 0);
  et_o2_test_live_counts_snapshot_v1(&o2_before);
  et_f32_test_live_counts_snapshot_v1(&i2_before);
  CHECK(et_o2_optimizer_state_snapshot_v1(item.optimizer, &left, &error) == 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(item.optimizer, &right, &error) ==
        0);
  et_o2_test_live_counts_snapshot_v1(&o2_snapshots);
  et_f32_test_live_counts_snapshot_v1(&i2_snapshots);
  CHECK(o2_snapshots.builders == o2_before.builders);
  CHECK(o2_snapshots.optimizers == o2_before.optimizers);
  CHECK(o2_snapshots.live_states == o2_before.live_states + 2u);
  CHECK(o2_snapshots.dead_states == o2_before.dead_states);
  CHECK(o2_snapshots.live_state_handles ==
        o2_before.live_state_handles + 4u);
  CHECK(o2_snapshots.dead_state_handles == o2_before.dead_state_handles);
  CHECK(o2_snapshots.state_borrows == o2_before.state_borrows);
  CHECK(o2_snapshots.owned_state_clones ==
        o2_before.owned_state_clones + 4u);
  CHECK(i2_snapshots.tensors == i2_before.tensors + 4u);
  CHECK(i2_snapshots.parameters == i2_before.parameters);
  CHECK(i2_snapshots.borrows == i2_before.borrows);
  CHECK(i2_snapshots.copy_plans == i2_before.copy_plans);
  CHECK(i2_snapshots.gradient_plans == i2_before.gradient_plans);
  CHECK(i2_snapshots.reset_plans == i2_before.reset_plans);
  CHECK(i2_snapshots.owned_clones == i2_before.owned_clones + 4u);

  left_moment = state_moment_bits(left, 0u, ET_O2_MOMENT_EXP_AVG);
  right_moment = state_moment_bits(right, 0u, ET_O2_MOMENT_EXP_AVG);
  CHECK(left_moment == right_moment);
  set_gradient(item.parameter, &next_gradient, 1u, 3u,
               UINT32_C(0x40000000));
  read_parameter_bits(item.parameter, &parameter_before, 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            item.optimizer, 0u, ET_O2_MOMENT_EXP_AVG,
            &optimizer_moment_before, 1u, &error) == 0);
  et_o2_test_live_counts_snapshot_v1(&o2_live);
  et_f32_test_live_counts_snapshot_v1(&i2_live);

  et_o2_test_fail_release_after_v1(0u);
  CHECK(et_o2_test_state_set_resolution_from_state_v1(
            left, 0u, ET_O2_MOMENT_EXP_AVG, right, 0u,
            ET_O2_MOMENT_EXP_AVG) == 0);
  expect_o2_error(et_o2_optimizer_state_release_v1(left, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT);
  CHECK(et_o2_optimizer_state_lifecycle_v1(left, &lifecycle, &error) == 0);
  CHECK(lifecycle == ET_O2_OPTIMIZER_STATE_LIVE);
  et_o2_test_live_counts_snapshot_v1(&o2_after);
  et_f32_test_live_counts_snapshot_v1(&i2_after);
  check_o2_live_counts_equal(&o2_after, &o2_live);
  check_i2_live_counts_equal(&i2_after, &i2_live);
  CHECK(state_moment_bits(right, 0u, ET_O2_MOMENT_EXP_AVG) == right_moment);
  CHECK(et_o2_test_state_set_resolution_from_state_v1(
            left, 0u, ET_O2_MOMENT_EXP_AVG, left, 0u,
            ET_O2_MOMENT_EXP_AVG) == 0);
  CHECK(state_moment_bits(left, 0u, ET_O2_MOMENT_EXP_AVG) == left_moment);

  CHECK(et_o2_test_state_set_resolution_from_optimizer_v1(
            left, 0u, ET_O2_MOMENT_EXP_AVG, item.optimizer, 0u,
            ET_O2_MOMENT_EXP_AVG) == 0);
  expect_o2_error(et_o2_optimizer_state_release_v1(left, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_OWNER_CONFLICT);
  CHECK(et_o2_optimizer_state_lifecycle_v1(left, &lifecycle, &error) == 0);
  CHECK(lifecycle == ET_O2_OPTIMIZER_STATE_LIVE);
  et_o2_test_live_counts_snapshot_v1(&o2_after);
  et_f32_test_live_counts_snapshot_v1(&i2_after);
  check_o2_live_counts_equal(&o2_after, &o2_live);
  check_i2_live_counts_equal(&i2_after, &i2_live);
  check_parameter_bits(item.parameter, &parameter_before, 1u);
  check_gradient_bits(item.parameter, &next_gradient, 1u);
  check_gradient_metadata(item.parameter, ET_F32_GRADIENT_PRESENT, 3u,
                          UINT32_C(0x40000000));
  CHECK(completed_updates(item.optimizer) == 1u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(
            item.optimizer, 0u, ET_O2_MOMENT_EXP_AVG, &left_moment, 1u,
            &error) == 0);
  CHECK(left_moment == optimizer_moment_before);
  CHECK(et_o2_test_state_set_resolution_from_state_v1(
            left, 0u, ET_O2_MOMENT_EXP_AVG, left, 0u,
            ET_O2_MOMENT_EXP_AVG) == 0);
  CHECK(state_moment_bits(left, 0u, ET_O2_MOMENT_EXP_AVG) == right_moment);

  et_o2_test_reset_failpoints_v1();
  CHECK(et_o2_optimizer_state_release_v1(left, &error) == 0);
  CHECK(state_moment_bits(right, 0u, ET_O2_MOMENT_EXP_AVG) == right_moment);
  CHECK(et_o2_optimizer_state_release_v1(right, &error) == 0);
  et_o2_test_live_counts_snapshot_v1(&o2_after);
  et_f32_test_live_counts_snapshot_v1(&i2_after);
  CHECK(o2_after.live_states == o2_before.live_states);
  CHECK(o2_after.dead_states == o2_before.dead_states + 2u);
  CHECK(o2_after.live_state_handles == o2_before.live_state_handles);
  CHECK(o2_after.dead_state_handles == o2_before.dead_state_handles + 4u);
  CHECK(o2_after.owned_state_clones == o2_before.owned_state_clones);
  CHECK(i2_after.tensors == i2_before.tensors);
  CHECK(i2_after.parameters == i2_before.parameters);
  CHECK(i2_after.borrows == i2_before.borrows);
  CHECK(i2_after.copy_plans == i2_before.copy_plans);
  CHECK(i2_after.gradient_plans == i2_before.gradient_plans);
  CHECK(i2_after.reset_plans == i2_before.reset_plans);
  CHECK(i2_after.owned_clones == i2_before.owned_clones);
  CHECK(et_o2_optimizer_step_v1(item.optimizer, &error) == 0);
  CHECK(completed_updates(item.optimizer) == 2u);
}

static void test_release_first_middle_last_defects(void) {
  static int identities[2];
  const uint32_t one = UINT32_C(0x3f800000);
  const size_t release_limits[] = {0u, 1u, 3u};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 error;
  et_f32_parameter *parameters[2] = {NULL, NULL};
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  size_t index;

  for (index = 0u; index < 2u; index++) {
    et_f32_tensor *initial = make_tensor(&one, 1u);
    CHECK(et_f32_parameter_create_v1(initial, &parameters[index],
                                     &tensor_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(parameters[index],
                                            &identities[index],
                                            &tensor_error) == 0);
    destroy_tensor(&initial);
  }
  CHECK(et_o2_optimizer_builder_create_v1(
            2u, ET_O2_CLIP_NONE, 0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one,
            &builder, &error) == 0);
  for (index = 0u; index < 2u; index++) {
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, index, parameters[index], &identities[index],
              UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
              UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, &error) ==
          0);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &error) == 0);
  for (index = 0u; index < sizeof(release_limits) / sizeof(release_limits[0]);
       index++) {
    et_o2_optimizer_state *state = NULL;
    et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
    et_o2_test_live_counts_v1 live = {.struct_size = sizeof(live)};
    et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};
    uint32_t lifecycle = 0u;
    et_o2_test_live_counts_snapshot_v1(&before);
    CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, &error) == 0);
    et_o2_test_live_counts_snapshot_v1(&live);
    CHECK(live.owned_state_clones == before.owned_state_clones + 4u);
    et_o2_test_fail_release_after_v1(release_limits[index]);
    expect_o2_error(et_o2_optimizer_state_release_v1(state, &error), &error,
                    ET_O2_STATUS_INTERNAL, ET_O2_CODE_PROVIDER_DEFECT);
    CHECK(et_o2_optimizer_state_lifecycle_v1(state, &lifecycle, &error) == 0);
    CHECK(lifecycle == ET_O2_OPTIMIZER_STATE_DEAD);
    et_o2_test_live_counts_snapshot_v1(&after);
    CHECK(after.live_states == before.live_states);
    CHECK(after.owned_state_clones == before.owned_state_clones);
    CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
    et_o2_test_reset_failpoints_v1();
  }
}

static void test_noncanonical_float_environment_category(void) {
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_error_v1 error;
  uint32_t factor = 0u;
  const uint32_t saved_mxcsr = _mm_getcsr();
  fixture item = make_fixture(
      one, UINT32_C(0x3dcccccd), UINT32_C(0x3f000000),
      UINT32_C(0x3f000000), UINT32_C(0x3a83126f), 0u, ET_O2_CLIP_NONE,
      0u, ET_O2_SCHEDULE_CONSTANT, 0u, 0u, one);

  _mm_setcsr(saved_mxcsr | UINT32_C(0x00008000));
  expect_o2_error(et_o2_optimizer_schedule_factor_bits_v1(
                      item.optimizer, &factor, &error),
                  &error, ET_O2_STATUS_DETERMINISM_UNAVAILABLE,
                  ET_O2_CODE_FLOAT_ENVIRONMENT);
  _mm_setcsr(saved_mxcsr);
  CHECK(et_o2_optimizer_schedule_factor_bits_v1(
            item.optimizer, &factor, &error) == 0);
  CHECK(factor == one);
}

int main(void) {
  test_builder_rejections();
  test_exact_parameter_ceiling();
  test_step_metadata_atomicity_and_zero();
  test_present_zero_decay_and_linear_schedule();
  test_global_norm_exact_boundary();
  test_clip_ratio_gradual_underflow_to_positive_zero();
  test_global_norm_above_multi_tensor_frozen_parity();
  test_global_norm_scaled_accumulation_avoids_sum_overflow();
  test_distinct_parameter_groups_frozen_parity();
  test_unequal_weight_accumulation_frozen_parity();
  test_allocation_failures_are_atomic();
  test_i2_step_allocation_failures_are_atomic();
  test_counter_overflow_is_atomic();
  test_state_round_trip_and_release();
  test_state_load_restores_schedule_clip_and_next_update();
  test_cross_owner_handle_and_snapshot_failpoints();
  test_nonfinite_snapshot_load_rejected_atomically();
  test_protected_state_metadata_load_validation();
  test_load_requires_absent_destination_before_state_borrow();
  test_release_defect_finishes_exact_once_tail();
  test_release_admission_corruption_is_nonmutating();
  test_release_exact_owner_ledger_is_nonmutating();
  test_release_first_middle_last_defects();
  test_noncanonical_float_environment_category();
  if (failures != 0) {
    (void)fprintf(stderr, "O2 native adversarial FAIL: %d/%d checks failed\n",
                  failures, checks);
    return 1;
  }
  (void)printf("O2 native adversarial PASS: %d checks\n", checks);
  return 0;
}
