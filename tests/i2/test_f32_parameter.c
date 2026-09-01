#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

static int checks;
static int failures;

#define CHECK(condition)                                                        \
  do {                                                                          \
    checks++;                                                                    \
    if (!(condition)) {                                                          \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                    #condition);                                                 \
      failures++;                                                                \
    }                                                                            \
  } while (0)

_Static_assert(sizeof(et_f32_gradient_metadata_v1) == 24u,
               "I2 gradient metadata layout changed");
_Static_assert(offsetof(et_f32_gradient_metadata_v1, state) == 8u,
               "I2 gradient state offset changed");
_Static_assert(offsetof(et_f32_gradient_metadata_v1,
                        normalization_weight_bits) == 12u,
               "I2 gradient weight offset changed");
_Static_assert(offsetof(et_f32_gradient_metadata_v1, contribution_count) ==
                   16u,
               "I2 gradient count offset changed");
_Static_assert(sizeof(et_f32_gradient_contribution_v1) == 32u,
               "I2 contribution layout changed");

static void expect_error(int32_t result, const et_f32_tensor_error *error,
                         et_f32_tensor_error_category category,
                         et_f32_tensor_error_code code) {
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

static et_f32_tensor *make_tensor(const uint64_t *shape, size_t rank,
                                  const uint32_t *bits, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, count, &error) == 0);
  return tensor;
}

static void destroy_tensor(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void read_bits(const et_f32_tensor *tensor, uint32_t *bits,
                      size_t count) {
  et_f32_tensor_error error;
  memset(bits, 0xa5, count * sizeof(*bits));
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, bits, count, &error) == 0);
}

static void check_tensor_bits(const et_f32_tensor *tensor,
                              const uint32_t *expected, size_t count) {
  uint32_t actual[128];
  CHECK(count <= sizeof(actual) / sizeof(actual[0]));
  if (count > sizeof(actual) / sizeof(actual[0])) {
    return;
  }
  read_bits(tensor, actual, count);
  if (memcmp(actual, expected, count * sizeof(*expected)) != 0) {
    (void)fprintf(stderr, "bit mismatch:");
    for (size_t index = 0u; index < count; index++) {
      (void)fprintf(stderr, " %08x/%08x", actual[index], expected[index]);
    }
    (void)fprintf(stderr, "\n");
  }
  CHECK(memcmp(actual, expected, count * sizeof(*expected)) == 0);
}

static et_f32_parameter *make_parameter(const et_f32_tensor *initial) {
  et_f32_tensor_error error;
  et_f32_parameter *parameter = NULL;
  CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  CHECK(parameter != NULL);
  return parameter;
}

static void destroy_parameter(et_f32_parameter **parameter) {
  et_f32_tensor_error error;
  CHECK(et_f32_parameter_destroy_v1(parameter, &error) == 0);
  CHECK(*parameter == NULL);
}

static et_f32_gradient_metadata_v1 metadata(
    const et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  et_f32_gradient_metadata_v1 result;
  memset(&result, 0, sizeof(result));
  result.struct_size = sizeof(result);
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &result, &error) ==
        0);
  CHECK(result.struct_size == sizeof(result));
  return result;
}

static void check_absent(et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  et_f32_gradient_metadata_v1 actual = metadata(parameter);
  int32_t scalar = 0x51515151;
  et_f32_tensor *snapshot = NULL;
  et_f32_tensor_borrow *borrow = NULL;

  CHECK(actual.state == ET_F32_GRADIENT_ABSENT);
  CHECK(actual.contribution_count == 0u);
  CHECK(actual.normalization_weight_bits == UINT32_C(0x00000000));
  expect_error(et_f32_parameter_gradient_finite_v1(parameter, &scalar,
                                                   &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(scalar == 0x51515151);
  expect_error(et_f32_parameter_gradient_exact_positive_zero_v1(
                   parameter, &scalar, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(scalar == 0x51515151);
  expect_error(et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot,
                                                     &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(snapshot == NULL);
  expect_error(et_f32_parameter_gradient_borrow_begin_v1(parameter, &borrow,
                                                         &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(borrow == NULL);
}

static et_f32_tensor *gradient_snapshot(et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  et_f32_tensor *snapshot = NULL;
  CHECK(et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot, &error) ==
        0);
  CHECK(snapshot != NULL);
  return snapshot;
}

static void check_present(et_f32_parameter *parameter, uint64_t count,
                          uint32_t weight_bits, const uint32_t *expected,
                          size_t element_count, int32_t exact_zero) {
  et_f32_tensor_error error;
  et_f32_gradient_metadata_v1 actual = metadata(parameter);
  et_f32_tensor *snapshot;
  int32_t scalar = -1;

  CHECK(actual.state == ET_F32_GRADIENT_PRESENT);
  CHECK(actual.contribution_count == count);
  CHECK(actual.normalization_weight_bits == weight_bits);
  CHECK(et_f32_parameter_gradient_finite_v1(parameter, &scalar, &error) == 0);
  CHECK(scalar == 1);
  scalar = -1;
  CHECK(et_f32_parameter_gradient_exact_positive_zero_v1(
            parameter, &scalar, &error) == 0);
  CHECK(scalar == exact_zero);
  snapshot = gradient_snapshot(parameter);
  check_tensor_bits(snapshot, expected, element_count);
  destroy_tensor(&snapshot);
}

static void test_parameter_identity_value_and_detachment(void) {
  const uint64_t shape[] = {3u};
  const uint32_t initial_bits[] = {UINT32_C(0x3f800000),
                                   UINT32_C(0x80000000),
                                   UINT32_C(0x7fc12345)};
  const uint32_t changed_bits[] = {UINT32_C(0x40000000),
                                   UINT32_C(0x40400000),
                                   UINT32_C(0x40800000)};
  et_f32_tensor_error error;
  et_f32_tensor *initial = make_tensor(shape, 1u, initial_bits, 3u);
  et_f32_parameter *parameter = make_parameter(initial);
  et_f32_parameter *other = make_parameter(initial);
  et_f32_tensor *snapshot = NULL;
  et_f32_tensor *fresh = NULL;
  et_f32_tensor_borrow *left_borrow = NULL;
  et_f32_tensor_borrow *right_borrow = NULL;
  const et_kernel_tensor_view_v1 *left_view = NULL;
  const et_kernel_tensor_view_v1 *right_view = NULL;
  const et_f32_tensor *live_value = NULL;
  const et_f32_tensor *saved_live_value;
  const void *identity = NULL;
  int identity_token = 1;
  int other_identity = 2;

  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) ==
        0);
  check_tensor_bits(snapshot, initial_bits, 3u);
  CHECK(et_f32_tensor_copy_bits_from_v1(snapshot, changed_bits, 3u, &error) ==
        0);
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &fresh, &error) == 0);
  check_tensor_bits(fresh, initial_bits, 3u);
  destroy_tensor(&snapshot);
  destroy_tensor(&fresh);

  CHECK(et_f32_parameter_value_tensor_v1(parameter, &live_value, &error) ==
        0);
  CHECK(live_value != NULL);
  check_tensor_bits(live_value, initial_bits, 3u);
  saved_live_value = live_value;
  expect_error(et_f32_parameter_value_tensor_v1(parameter, &live_value,
                                                &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(live_value == saved_live_value);

  CHECK(et_f32_tensor_copy_bits_from_v1(initial, changed_bits, 3u, &error) ==
        0);
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &fresh, &error) == 0);
  check_tensor_bits(fresh, initial_bits, 3u);
  destroy_tensor(&fresh);

  expect_error(et_f32_parameter_bind_identity_v1(parameter, NULL, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity_token,
                                          &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity_token,
                                          &error) == 0);
  expect_error(et_f32_parameter_bind_identity_v1(parameter, &other_identity,
                                                 &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_parameter_identity_v1(parameter, &identity, &error) == 0);
  CHECK(identity == &identity_token);

  CHECK(et_f32_parameter_value_borrow_begin_v1(parameter, &left_borrow,
                                               &error) == 0);
  CHECK(et_f32_parameter_value_borrow_begin_v1(other, &right_borrow,
                                               &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(left_borrow, &left_view, &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(right_borrow, &right_view, &error) == 0);
  CHECK(left_view->data != right_view->data);
  CHECK(et_f32_tensor_borrow_end_v1(&left_borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_end_v1(&right_borrow, &error) == 0);

  check_absent(parameter);
  check_absent(other);
  destroy_parameter(&parameter);
  {
    size_t rank = 99u;
    expect_error(et_f32_tensor_rank_v1(saved_live_value, &rank, &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                 ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(rank == 99u);
  }
  destroy_parameter(&other);
  destroy_tensor(&initial);
}

static void test_tensor_batch_copy_atomicity(void) {
  const uint64_t shape[] = {2u};
  const uint64_t wrong_shape[] = {3u};
  const uint32_t a_bits[] = {UINT32_C(0x3f800000), UINT32_C(0x40000000)};
  const uint32_t b_bits[] = {UINT32_C(0x40400000), UINT32_C(0x40800000)};
  const uint32_t old_bits[] = {UINT32_C(0xdeadbeef), UINT32_C(0xabcdef01)};
  const uint32_t wrong_bits[] = {1u, 2u, 3u};
  et_f32_tensor_error error;
  et_f32_tensor *source_a = make_tensor(shape, 1u, a_bits, 2u);
  et_f32_tensor *source_b = make_tensor(shape, 1u, b_bits, 2u);
  et_f32_tensor *destination_a = make_tensor(shape, 1u, old_bits, 2u);
  et_f32_tensor *destination_b = make_tensor(shape, 1u, old_bits, 2u);
  et_f32_tensor *wrong = make_tensor(wrong_shape, 1u, wrong_bits, 3u);
  et_f32_tensor_copy_assignment_v1 assignments[2];
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor *saved;

  memset(assignments, 0, sizeof(assignments));
  for (size_t index = 0u; index < 2u; index++) {
    assignments[index].struct_size = sizeof(assignments[index]);
  }
  assignments[0].destination = destination_a;
  assignments[0].source = source_a;
  assignments[1].destination = destination_b;
  assignments[1].source = wrong;
  expect_error(et_f32_tensor_copy_plan_prepare_v1(2u, assignments, &plan,
                                                  &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INVALID_SHAPE);
  CHECK(plan == NULL);
  check_tensor_bits(destination_a, old_bits, 2u);
  check_tensor_bits(destination_b, old_bits, 2u);

  assignments[1].source = source_b;
  assignments[1].destination = destination_a;
  expect_error(et_f32_tensor_copy_plan_prepare_v1(2u, assignments, &plan,
                                                  &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);
  check_tensor_bits(destination_a, old_bits, 2u);

  assignments[0].destination = destination_a;
  assignments[0].source = destination_a;
  assignments[1].destination = destination_b;
  expect_error(et_f32_tensor_copy_plan_prepare_v1(2u, assignments, &plan,
                                                  &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);
  assignments[0].source = source_a;

  {
    et_f32_tensor_borrow *source_borrow = NULL;
    CHECK(et_f32_tensor_borrow_begin_v1(source_a, &source_borrow, &error) ==
          0);
    expect_error(et_f32_tensor_copy_plan_prepare_v1(2u, assignments, &plan,
                                                    &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                 ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(plan == NULL);
    check_tensor_bits(destination_a, old_bits, 2u);
    check_tensor_bits(destination_b, old_bits, 2u);
    CHECK(et_f32_tensor_borrow_end_v1(&source_borrow, &error) == 0);
  }

  CHECK(et_f32_tensor_copy_plan_prepare_v1(2u, assignments, &plan, &error) ==
        0);
  CHECK(plan != NULL);
  expect_error(et_f32_tensor_copy_bits_from_v1(source_a, old_bits, 2u,
                                               &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  check_tensor_bits(source_a, a_bits, 2u);
  expect_error(et_f32_tensor_copy_bits_from_v1(destination_a, a_bits, 2u,
                                               &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  check_tensor_bits(destination_a, old_bits, 2u);
  saved = destination_a;
  expect_error(et_f32_tensor_destroy_v1(&destination_a, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(destination_a == saved);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_fail_alloc_after_v1(0u);
#endif
  CHECK(et_f32_tensor_copy_plan_commit_v1(plan, &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_reset_allocator_v1();
#endif
  check_tensor_bits(destination_a, a_bits, 2u);
  check_tensor_bits(destination_b, b_bits, 2u);
  expect_error(et_f32_tensor_copy_plan_commit_v1(plan, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  check_tensor_bits(destination_a, a_bits, 2u);
  check_tensor_bits(destination_b, b_bits, 2u);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);
  CHECK(plan == NULL);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);

  destroy_tensor(&source_a);
  destroy_tensor(&source_b);
  destroy_tensor(&destination_a);
  destroy_tensor(&destination_b);
  destroy_tensor(&wrong);
}

static void test_gradient_accumulation_and_reset(void) {
  const uint64_t shape[] = {3u};
  const uint64_t wrong_shape[] = {2u};
  const uint32_t initial[] = {UINT32_C(0x3f800000),
                              UINT32_C(0x40000000),
                              UINT32_C(0x40400000)};
  const uint32_t first_a[] = {UINT32_C(0x3f800000),
                              UINT32_C(0xc0000000),
                              UINT32_C(0x00000000)};
  const uint32_t first_b[] = {UINT32_C(0x40400000),
                              UINT32_C(0x40800000),
                              UINT32_C(0x00000000)};
  const uint32_t zeros[] = {0u, 0u, 0u};
  const uint32_t second_a[] = {UINT32_C(0x3f000000),
                               UINT32_C(0x40000000),
                               UINT32_C(0x80000000)};
  const uint32_t second_b[] = {UINT32_C(0xbf800000),
                               UINT32_C(0xc0800000),
                               UINT32_C(0x00000000)};
  const uint32_t accumulated_a[] = {UINT32_C(0x3fc00000), 0u, 0u};
  const uint32_t accumulated_b[] = {UINT32_C(0x40000000), 0u, 0u};
  const uint32_t wrong_bits[] = {0u, 0u};
  const uint32_t nonfinite_bits[] = {UINT32_C(0x7f800000), 0u, 0u};
  et_f32_tensor_error error;
  et_f32_tensor *initial_tensor = make_tensor(shape, 1u, initial, 3u);
  et_f32_tensor *a1 = make_tensor(shape, 1u, first_a, 3u);
  et_f32_tensor *b1 = make_tensor(shape, 1u, first_b, 3u);
  et_f32_tensor *z1 = make_tensor(shape, 1u, zeros, 3u);
  et_f32_tensor *a2 = make_tensor(shape, 1u, second_a, 3u);
  et_f32_tensor *b2 = make_tensor(shape, 1u, second_b, 3u);
  et_f32_tensor *wrong = make_tensor(wrong_shape, 1u, wrong_bits, 2u);
  et_f32_tensor *nonfinite =
      make_tensor(shape, 1u, nonfinite_bits, 3u);
  et_f32_parameter *a = make_parameter(initial_tensor);
  et_f32_parameter *b = make_parameter(initial_tensor);
  et_f32_parameter *z = make_parameter(initial_tensor);
  et_f32_gradient_contribution_v1 contributions[3];
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_reset_plan *reset = NULL;
  et_f32_parameter *parameters[] = {a, b, z};
  et_f32_parameter *saved_parameter;
  et_f32_tensor *detached;
  et_f32_tensor *fresh;
  et_f32_tensor_borrow *borrow = NULL;
  int identity_a = 1;
  int identity_b = 2;
  int identity_z = 3;

  CHECK(et_f32_parameter_bind_identity_v1(a, &identity_a, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(b, &identity_b, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(z, &identity_z, &error) == 0);
  check_absent(a);
  check_absent(b);
  check_absent(z);

  memset(contributions, 0, sizeof(contributions));
  for (size_t index = 0u; index < 3u; index++) {
    contributions[index].struct_size = sizeof(contributions[index]);
    contributions[index].destination = parameters[index];
    contributions[index].expected_ordinal = 0u;
  }
  contributions[0].weighted_numerator = a1;
  contributions[1].weighted_numerator = b1;
  contributions[2].weighted_numerator = z1;
  CHECK(et_f32_gradient_plan_prepare_v1(3u, contributions,
                                        UINT32_C(0x3fc00000), &plan,
                                        &error) == 0);
  CHECK(plan != NULL);
  saved_parameter = a;
  expect_error(et_f32_parameter_destroy_v1(&a, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(a == saved_parameter);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_fail_alloc_after_v1(0u);
#endif
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_reset_allocator_v1();
#endif
  expect_error(et_f32_gradient_plan_commit_v1(plan, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  check_present(a, 1u, UINT32_C(0x3fc00000), first_a, 3u, 0);
  check_present(b, 1u, UINT32_C(0x3fc00000), first_b, 3u, 0);
  check_present(z, 1u, UINT32_C(0x3fc00000), zeros, 3u, 1);

  detached = gradient_snapshot(a);
  CHECK(et_f32_tensor_copy_bits_from_v1(detached, zeros, 3u, &error) == 0);
  fresh = gradient_snapshot(a);
  check_tensor_bits(fresh, first_a, 3u);
  destroy_tensor(&detached);
  destroy_tensor(&fresh);

  CHECK(et_f32_parameter_gradient_borrow_begin_v1(a, &borrow, &error) == 0);
  expect_error(et_f32_gradient_reset_plan_prepare_v1(3u, parameters, &reset,
                                                     &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(reset == NULL);
  check_present(a, 1u, UINT32_C(0x3fc00000), first_a, 3u, 0);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);

  contributions[0].weighted_numerator = a2;
  contributions[1].weighted_numerator = b2;
  contributions[2].weighted_numerator = z1;
  for (size_t index = 0u; index < 3u; index++) {
    contributions[index].expected_ordinal = 1u;
  }
  CHECK(et_f32_gradient_plan_prepare_v1(3u, contributions,
                                        UINT32_C(0x3f000000), &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  check_present(a, 2u, UINT32_C(0x40000000), accumulated_a, 3u, 0);
  check_present(b, 2u, UINT32_C(0x40000000), accumulated_b, 3u, 0);
  check_present(z, 2u, UINT32_C(0x40000000), zeros, 3u, 1);

  for (size_t index = 0u; index < 3u; index++) {
    contributions[index].expected_ordinal = 2u;
  }

  contributions[1].destination = a;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   3u, contributions, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);
  contributions[1].destination = b;
  check_present(a, 2u, UINT32_C(0x40000000), accumulated_a, 3u, 0);

  contributions[1].weighted_numerator = a2;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   3u, contributions, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);
  contributions[1].weighted_numerator = b2;
  check_present(a, 2u, UINT32_C(0x40000000), accumulated_a, 3u, 0);
  check_present(b, 2u, UINT32_C(0x40000000), accumulated_b, 3u, 0);

  contributions[0].expected_ordinal = 1u;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   3u, contributions, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(plan == NULL);
  contributions[0].expected_ordinal = 2u;

  contributions[0].weighted_numerator = wrong;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   3u, contributions, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INVALID_SHAPE);
  CHECK(plan == NULL);
  contributions[0].weighted_numerator = nonfinite;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   3u, contributions, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);
  contributions[0].weighted_numerator = a2;
  for (size_t index = 0u; index < 4u; index++) {
    const uint32_t invalid_weight[] = {UINT32_C(0x00000000),
                                       UINT32_C(0x80000000),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0x7fc00000)};
    expect_error(et_f32_gradient_plan_prepare_v1(
                     3u, contributions, invalid_weight[index], &plan, &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_F32_TENSOR_CODE_INVALID_BUFFER);
    CHECK(plan == NULL);
  }
  check_present(a, 2u, UINT32_C(0x40000000), accumulated_a, 3u, 0);
  check_present(b, 2u, UINT32_C(0x40000000), accumulated_b, 3u, 0);

  {
    et_f32_parameter *duplicates[] = {a, a};
    expect_error(et_f32_gradient_reset_plan_prepare_v1(2u, duplicates,
                                                       &reset, &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_F32_TENSOR_CODE_INVALID_BUFFER);
    CHECK(reset == NULL);
  }
  CHECK(et_f32_gradient_reset_plan_prepare_v1(3u, parameters, &reset,
                                              &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_fail_alloc_after_v1(0u);
#endif
  CHECK(et_f32_gradient_reset_plan_commit_v1(reset, &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_tensor_test_reset_allocator_v1();
#endif
  expect_error(et_f32_gradient_reset_plan_commit_v1(reset, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset, &error) == 0);
  check_absent(a);
  check_absent(b);
  check_absent(z);

  CHECK(et_f32_gradient_reset_plan_prepare_v1(3u, parameters, &reset,
                                              &error) == 0);
  CHECK(et_f32_gradient_reset_plan_commit_v1(reset, &error) == 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset, &error) == 0);
  check_absent(a);
  check_absent(b);
  check_absent(z);

  destroy_parameter(&a);
  destroy_parameter(&b);
  destroy_parameter(&z);
  destroy_tensor(&initial_tensor);
  destroy_tensor(&a1);
  destroy_tensor(&b1);
  destroy_tensor(&z1);
  destroy_tensor(&a2);
  destroy_tensor(&b2);
  destroy_tensor(&wrong);
  destroy_tensor(&nonfinite);
}

static void test_binary32_environment_and_edge_rounding(void) {
  const uint64_t shape[] = {1u};
  const uint32_t initial_bits[] = {0u};
  const uint32_t subnormal_bits[] = {1u};
  const uint32_t one_bits[] = {UINT32_C(0x3f800000)};
  const uint32_t half_ulp_bits[] = {UINT32_C(0x33800000)};
  et_f32_tensor_error error;
  et_f32_tensor *initial = make_tensor(shape, 1u, initial_bits, 1u);
  et_f32_tensor *subnormal =
      make_tensor(shape, 1u, subnormal_bits, 1u);
  et_f32_tensor *one = make_tensor(shape, 1u, one_bits, 1u);
  et_f32_tensor *half_ulp = make_tensor(shape, 1u, half_ulp_bits, 1u);
  et_f32_parameter *parameter = make_parameter(initial);
  et_f32_gradient_contribution_v1 item = {
      .struct_size = sizeof(item),
      .destination = parameter,
      .weighted_numerator = subnormal,
      .expected_ordinal = 0u,
  };
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_reset_plan *reset = NULL;
  et_f32_parameter *parameters[] = {parameter};
  uint32_t saved_control = _mm_getcsr();
  int identity = 7;

  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
  {
    const uint32_t hostile_controls[] = {
        saved_control | UINT32_C(0x00002000),
        saved_control | UINT32_C(0x00008000),
        saved_control | UINT32_C(0x00000040),
        saved_control & ~UINT32_C(0x00000080),
    };
    for (size_t index = 0u;
         index < sizeof(hostile_controls) / sizeof(hostile_controls[0]);
         index++) {
      _mm_setcsr(hostile_controls[index]);
      expect_error(et_f32_gradient_plan_prepare_v1(
                       1u, &item, UINT32_C(0x3f800000), &plan, &error),
                   &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                   ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT);
      CHECK(plan == NULL);
      check_absent(parameter);
    }
  }
  _mm_setcsr(saved_control);

  CHECK(et_f32_gradient_plan_prepare_v1(1u, &item,
                                        UINT32_C(0x3f800000), &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  check_present(parameter, 1u, UINT32_C(0x3f800000), subnormal_bits, 1u, 0);
  CHECK(_mm_getcsr() == saved_control);

  CHECK(et_f32_gradient_reset_plan_prepare_v1(1u, parameters, &reset,
                                              &error) == 0);
  CHECK(et_f32_gradient_reset_plan_commit_v1(reset, &error) == 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset, &error) == 0);
  item.weighted_numerator = one;
  item.expected_ordinal = 0u;
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &item,
                                        UINT32_C(0x3f800000), &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  item.weighted_numerator = half_ulp;
  item.expected_ordinal = 1u;
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &item,
                                        UINT32_C(0x3f800000), &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  check_present(parameter, 2u, UINT32_C(0x40000000), one_bits, 1u, 0);
  CHECK(_mm_getcsr() == saved_control);

  destroy_parameter(&parameter);
  destroy_tensor(&half_ulp);
  destroy_tensor(&one);
  destroy_tensor(&subnormal);
  destroy_tensor(&initial);
}

#ifdef ET_F32_TENSOR_TESTING
static void test_zero_count_live_error_aliases(void) {
  const uint64_t shape[] = {66u};
  uint32_t zero[66] = {0};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = make_tensor(shape, 1u, zero, 66u);
  et_f32_tensor_error *aliased_error =
      (et_f32_tensor_error *)(void *)
          et_f32_tensor_test_data_storage_v1(tensor);
  et_f32_tensor_copy_plan *copy_plan = NULL;
  et_f32_gradient_plan *gradient_plan = NULL;
  et_f32_gradient_reset_plan *reset_plan = NULL;

  CHECK(et_f32_tensor_copy_plan_prepare_v1(0u, NULL, &copy_plan,
                                           aliased_error) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(copy_plan == NULL);
  check_tensor_bits(tensor, zero, 66u);
  CHECK(et_f32_gradient_plan_prepare_v1(0u, NULL, UINT32_C(0x3f800000),
                                        &gradient_plan, aliased_error) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(gradient_plan == NULL);
  check_tensor_bits(tensor, zero, 66u);
  CHECK(et_f32_gradient_reset_plan_prepare_v1(0u, NULL, &reset_plan,
                                              aliased_error) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(reset_plan == NULL);
  check_tensor_bits(tensor, zero, 66u);

  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, zero, 66u, &error) == 0);
  destroy_tensor(&tensor);
}

static void test_malformed_gradient_state_and_overflow(void) {
  const uint64_t shape[] = {2u};
  const uint32_t initial[] = {UINT32_C(0x3f800000),
                              UINT32_C(0x40000000)};
  const uint32_t contribution_bits[] = {UINT32_C(0x3f000000),
                                        UINT32_C(0xbf000000)};
  const uint32_t nonfinite[] = {UINT32_C(0x7f800000), 0u};
  et_f32_tensor_error error;
  et_f32_tensor *initial_tensor = make_tensor(shape, 1u, initial, 2u);
  et_f32_tensor *contribution =
      make_tensor(shape, 1u, contribution_bits, 2u);
  et_f32_parameter *parameter = make_parameter(initial_tensor);
  et_f32_gradient_contribution_v1 item = {
      .struct_size = sizeof(item),
      .destination = parameter,
      .weighted_numerator = contribution,
      .expected_ordinal = UINT64_MAX,
  };
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_metadata_v1 actual;
  int identity = 1;

  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
  et_f32_parameter_test_set_gradient_bits_v1(parameter, contribution_bits,
                                             2u);
  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        (uint64_t)INT64_MAX - 1u,
                                        UINT32_C(0x3f800000));
  item.expected_ordinal = (uint64_t)INT64_MAX - 1u;
  CHECK(et_f32_gradient_plan_prepare_v1(
            1u, &item, UINT32_C(0x3f800000), &plan, &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  actual = metadata(parameter);
  CHECK(actual.contribution_count == (uint64_t)INT64_MAX);
  item.expected_ordinal = (uint64_t)INT64_MAX;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(plan == NULL);
  actual = metadata(parameter);
  CHECK(actual.contribution_count == (uint64_t)INT64_MAX);

  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        UINT64_MAX, UINT32_C(0x3f800000));
  item.expected_ordinal = UINT64_MAX;
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(plan == NULL);
  actual = metadata(parameter);
  CHECK(actual.state == ET_F32_GRADIENT_PRESENT);
  CHECK(actual.contribution_count == UINT64_MAX);
  CHECK(actual.normalization_weight_bits == UINT32_C(0x3f800000));

  item.expected_ordinal = 1u;
  et_f32_parameter_test_set_metadata_v1(parameter, 99u, 1u,
                                        UINT32_C(0x3f800000));
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(plan == NULL);

  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_ABSENT,
                                        1u, 0u);
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(plan == NULL);

  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        1u, UINT32_C(0x3f800000));
  et_f32_parameter_test_set_gradient_bits_v1(parameter, nonfinite, 2u);
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x3f800000), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(plan == NULL);

  et_f32_parameter_test_set_gradient_bits_v1(parameter, contribution_bits,
                                             2u);
  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        1u, UINT32_C(0x7f7fffff));
  expect_error(et_f32_gradient_plan_prepare_v1(
                   1u, &item, UINT32_C(0x7f7fffff), &plan, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(plan == NULL);

  et_f32_parameter_test_set_gradient_bits_v1(parameter, (uint32_t[2]){0u, 0u},
                                             2u);
  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_ABSENT,
                                        0u, 0u);
  check_absent(parameter);
  destroy_parameter(&parameter);
  destroy_tensor(&contribution);
  destroy_tensor(&initial_tensor);
}

static void test_parameter_and_plan_failpoints(void) {
  const uint64_t shape[] = {2u};
  const uint32_t initial[] = {UINT32_C(0x3f800000),
                              UINT32_C(0x40000000)};
  const uint32_t contribution_bits[] = {UINT32_C(0x3f000000),
                                        UINT32_C(0xbf000000)};
  et_f32_tensor_error error;
  et_f32_tensor *initial_tensor = make_tensor(shape, 1u, initial, 2u);
  et_f32_tensor *contribution =
      make_tensor(shape, 1u, contribution_bits, 2u);
  int saw_parameter_success = 0;

  for (size_t allowed = 0u; allowed < 16u; allowed++) {
    et_f32_parameter *parameter = NULL;
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    int32_t result =
        et_f32_parameter_create_v1(initial_tensor, &parameter, &error);
    if (result == 0) {
      saw_parameter_success = 1;
      et_f32_tensor_test_reset_allocator_v1();
      destroy_parameter(&parameter);
      break;
    }
    expect_error(result, &error, ET_F32_TENSOR_ERROR_INTERNAL,
                 ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(parameter == NULL);
  }
  CHECK(saw_parameter_success);
  et_f32_tensor_test_reset_allocator_v1();

  {
    et_f32_parameter *parameter = make_parameter(initial_tensor);
    et_f32_gradient_contribution_v1 item = {
        .struct_size = sizeof(item),
        .destination = parameter,
        .weighted_numerator = contribution,
        .expected_ordinal = 0u,
    };
    et_f32_gradient_plan *plan = NULL;
    int identity = 1;
    int saw_plan_success = 0;
    CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) ==
          0);
    for (size_t allowed = 0u; allowed < 16u; allowed++) {
      et_f32_tensor_test_fail_alloc_after_v1(allowed);
      int32_t result = et_f32_gradient_plan_prepare_v1(
          1u, &item, UINT32_C(0x3f800000), &plan, &error);
      if (result == 0) {
        saw_plan_success = 1;
        et_f32_tensor_test_reset_allocator_v1();
        CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
        break;
      }
      expect_error(result, &error, ET_F32_TENSOR_ERROR_INTERNAL,
                   ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      CHECK(plan == NULL);
      check_absent(parameter);
    }
    CHECK(saw_plan_success);
    et_f32_tensor_test_reset_allocator_v1();
    check_absent(parameter);
    destroy_parameter(&parameter);
  }

  {
    const uint32_t old_bits[] = {UINT32_C(0xdeadbeef),
                                 UINT32_C(0xabcdef01)};
    et_f32_tensor *destination =
        make_tensor(shape, 1u, old_bits, 2u);
    et_f32_tensor_copy_assignment_v1 assignment = {
        .struct_size = sizeof(assignment),
        .destination = destination,
        .source = contribution,
    };
    et_f32_tensor_copy_plan *copy_plan = NULL;
    int saw_copy_success = 0;
    for (size_t allowed = 0u; allowed < 8u; allowed++) {
      et_f32_tensor_test_fail_alloc_after_v1(allowed);
      int32_t result = et_f32_tensor_copy_plan_prepare_v1(
          1u, &assignment, &copy_plan, &error);
      if (result == 0) {
        saw_copy_success = 1;
        et_f32_tensor_test_reset_allocator_v1();
        CHECK(et_f32_tensor_copy_plan_release_v1(&copy_plan, &error) == 0);
        break;
      }
      expect_error(result, &error, ET_F32_TENSOR_ERROR_INTERNAL,
                   ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      CHECK(copy_plan == NULL);
      check_tensor_bits(destination, old_bits, 2u);
    }
    CHECK(saw_copy_success);
    et_f32_tensor_test_reset_allocator_v1();
    check_tensor_bits(destination, old_bits, 2u);
    destroy_tensor(&destination);
  }

  {
    et_f32_parameter *parameter = make_parameter(initial_tensor);
    et_f32_gradient_contribution_v1 item = {
        .struct_size = sizeof(item),
        .destination = parameter,
        .weighted_numerator = contribution,
        .expected_ordinal = 0u,
    };
    et_f32_gradient_plan *gradient_plan = NULL;
    et_f32_gradient_reset_plan *reset_plan = NULL;
    et_f32_parameter *parameters[] = {parameter};
    int identity = 2;
    int saw_reset_success = 0;
    CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) ==
          0);
    CHECK(et_f32_gradient_plan_prepare_v1(1u, &item,
                                          UINT32_C(0x3f800000),
                                          &gradient_plan, &error) == 0);
    CHECK(et_f32_gradient_plan_commit_v1(gradient_plan, &error) == 0);
    CHECK(et_f32_gradient_plan_release_v1(&gradient_plan, &error) == 0);
    check_present(parameter, 1u, UINT32_C(0x3f800000), contribution_bits,
                  2u, 0);
    for (size_t allowed = 0u; allowed < 8u; allowed++) {
      et_f32_tensor_test_fail_alloc_after_v1(allowed);
      int32_t result = et_f32_gradient_reset_plan_prepare_v1(
          1u, parameters, &reset_plan, &error);
      if (result == 0) {
        saw_reset_success = 1;
        et_f32_tensor_test_reset_allocator_v1();
        CHECK(et_f32_gradient_reset_plan_release_v1(&reset_plan, &error) ==
              0);
        break;
      }
      expect_error(result, &error, ET_F32_TENSOR_ERROR_INTERNAL,
                   ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      CHECK(reset_plan == NULL);
      et_f32_tensor_test_reset_allocator_v1();
      check_present(parameter, 1u, UINT32_C(0x3f800000), contribution_bits,
                    2u, 0);
    }
    CHECK(saw_reset_success);
    et_f32_tensor_test_reset_allocator_v1();
    check_present(parameter, 1u, UINT32_C(0x3f800000), contribution_bits, 2u,
                  0);
    CHECK(et_f32_gradient_reset_plan_prepare_v1(1u, parameters, &reset_plan,
                                                &error) == 0);
    et_f32_tensor_test_fail_alloc_after_v1(0u);
    CHECK(et_f32_gradient_reset_plan_commit_v1(reset_plan, &error) == 0);
    et_f32_tensor_test_reset_allocator_v1();
    CHECK(et_f32_gradient_reset_plan_release_v1(&reset_plan, &error) == 0);
    check_absent(parameter);
    destroy_parameter(&parameter);
  }

  destroy_tensor(&contribution);
  destroy_tensor(&initial_tensor);
}
#endif

int main(void) {
  test_parameter_identity_value_and_detachment();
  test_tensor_batch_copy_atomicity();
  test_gradient_accumulation_and_reset();
  test_binary32_environment_and_edge_rounding();
#ifdef ET_F32_TENSOR_TESTING
  test_zero_count_live_error_aliases();
  test_malformed_gradient_state_and_overflow();
  test_parameter_and_plan_failpoints();
  {
    et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
    et_f32_test_live_counts_snapshot_v1(&counts);
    CHECK(counts.tensors == 0u);
    CHECK(counts.parameters == 0u);
    CHECK(counts.borrows == 0u);
    CHECK(counts.copy_plans == 0u);
    CHECK(counts.gradient_plans == 0u);
    CHECK(counts.reset_plans == 0u);
  }
#endif
  if (failures != 0) {
    (void)fprintf(stderr, "I2 parameter FAIL: %d of %d checks failed\n",
                  failures, checks);
    return 1;
  }
  (void)printf("I2 parameter PASS: %d identity, gradient, and atomic checks\n",
               checks);
  return 0;
}
