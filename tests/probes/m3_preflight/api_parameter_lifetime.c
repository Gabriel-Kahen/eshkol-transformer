/* Diagnostic only: no provider or model API. Exercise existing I2 ownership. */
#include "f32_parameter_internal.h"
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "API lifetime FAIL line %d: %s (category=%u code=%u)\n", \
          __LINE__, #condition, error.category, error.code); return 1; \
} } while (0)

int main(void) {
  et_f32_tensor_error error = {0};
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_v1 baseline = {.struct_size = sizeof(baseline)};
  et_f32_test_live_counts_v1 final = {.struct_size = sizeof(final)};
  et_f32_test_live_counts_snapshot_v1(&baseline);
#endif
  et_f32_tensor *initial = NULL, *replacement = NULL, *snapshot = NULL;
  et_f32_parameter *parameter = NULL;
  et_f32_tensor_borrow *borrow = NULL, *second = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  const et_f32_tensor *value = NULL, *after = NULL;
  uint64_t shape[] = {2};
  uint32_t ones[] = {0x3f800000, 0x3f800000};
  uint32_t twos[] = {0x40000000, 0x40000000}, bits[2] = {0};
  int identity = 0;
  CHECK(et_f32_tensor_create_v1(1, shape, &initial, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(initial, ones, 2, &error) == 0);
  CHECK(et_f32_tensor_create_v1(1, shape, &replacement, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(replacement, twos, 2, &error) == 0);
  CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &value, &error) == 0);
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) == 0);
  CHECK(et_f32_parameter_value_borrow_begin_v1(parameter, &borrow, &error) == 0);
  CHECK(et_f32_parameter_value_borrow_begin_v1(parameter, &second, &error) != 0);
  CHECK(error.code == ET_F32_TENSOR_CODE_ACTIVE_BORROW && second == NULL);
  et_f32_tensor_copy_assignment_v1 assignment = {
    ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE, (et_f32_tensor *)value, replacement};
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1, &assignment, &plan, &error) != 0);
  CHECK(error.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
        error.code == ET_F32_TENSOR_CODE_INVALID_HANDLE && plan == NULL);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) != 0);
  CHECK(error.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
        error.code == ET_F32_TENSOR_CODE_INVALID_HANDLE && parameter != NULL);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1, &assignment, &plan, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &after, &error) == 0);
  CHECK(value == after);
  CHECK(et_f32_parameter_validate_identity_v1(parameter, &identity, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(value, bits, 2, &error) == 0);
  CHECK(bits[0] == twos[0] && bits[1] == twos[1]);
  CHECK(et_f32_tensor_copy_bits_to_v1(snapshot, bits, 2, &error) == 0);
  CHECK(bits[0] == ones[0] && bits[1] == ones[1]);
  CHECK(et_f32_tensor_destroy_v1(&snapshot, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&initial, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&replacement, &error) == 0);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_snapshot_v1(&final);
  CHECK(memcmp(&baseline, &final, sizeof(baseline)) == 0);
#endif
  puts("M3 API LIFETIME PASS: exclusive borrow blocks mutation/destruction; stable identity permits later value change; detached snapshot preserves primals");
  return 0;
}
