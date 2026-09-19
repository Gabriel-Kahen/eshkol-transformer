#include "f32_parameter_internal.h"
#include <stdint.h>
#include "eshkol_transformer/m3t_f32_scoped.h"
#include <string.h>
static et_f32_tensor_borrow *held;
static et_f32_scoped_guard_internal scoped;
static et_f32_tensor_copy_plan *copy;
static et_f32_gradient_reset_plan *reset;
static et_f32_gradient_plan *gradient;
static et_f32_tensor *numerator;
static et_f32_test_live_counts_v1 held_counts;
extern et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);
int64_t et_i2_construction_test_count_v1(int64_t index) {
  et_f32_test_live_counts_v1 counts = {0};
  counts.struct_size = sizeof(counts);
  et_f32_test_live_counts_snapshot_v1(&counts);
  if (index == 0) return (int64_t)counts.parameters;
  if (index == 1) return (int64_t)counts.tensors;
  if (index == 2) return (int64_t)counts.borrows;
  return -1;
}
int64_t et_i2_construction_test_hold_v1(void *raw, int64_t mode) {
  et_f32_parameter *parameter = raw;
  const et_f32_tensor *value = NULL;
  et_f32_tensor *grad = et_m3t_test_parameter_gradient(parameter);
  if (held || scoped.owner || copy || reset || gradient || numerator || !grad)
    return -1;
  memset(&held_counts, 0, sizeof(held_counts));
  held_counts.struct_size = sizeof(held_counts);
  et_f32_test_live_counts_snapshot_v1(&held_counts);
  int32_t status = et_f32_parameter_value_tensor_v1(parameter, &value, NULL);
  if (status) return status;
  if (mode == 0) status = et_f32_parameter_value_borrow_begin_v1(parameter, &held, NULL);
  else if (mode == 1) status = et_f32_tensor_borrow_begin_v1(grad, &held, NULL);
  else if (mode == 2) status = et_f32_tensor_scoped_begin_internal((et_f32_tensor *)value, &scoped, NULL);
  else if (mode == 3) status = et_f32_tensor_scoped_begin_internal(grad, &scoped, NULL);
  else if (mode == 4) {
    et_f32_tensor_copy_assignment_v1 assignment = {sizeof(assignment), (et_f32_tensor *)value, grad};
    status = et_f32_tensor_copy_plan_prepare_v1(1, &assignment, &copy, NULL);
  } else if (mode == 5) status = et_f32_gradient_reset_plan_prepare_v1(1, &parameter, &reset, NULL);
  else if (mode == 6) {
    status = et_f32_tensor_clone_v1(value, &numerator, NULL);
    if (!status) {
      et_f32_gradient_contribution_v1 contribution = {sizeof(contribution), parameter, numerator, 0};
      status = et_f32_gradient_plan_prepare_v1(1, &contribution, UINT32_C(0x3f800000), &gradient, NULL);
    }
  } else return -1;
  if (status) {
    (void)et_f32_tensor_destroy_v1(&numerator, NULL);
    return status;
  }
  et_f32_test_live_counts_snapshot_v1(&held_counts);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  return status;
}
int64_t et_i2_construction_test_unchanged_v1(void) {
  et_f32_test_live_counts_v1 counts = {0};
  counts.struct_size = sizeof(counts);
  et_f32_test_live_counts_snapshot_v1(&counts);
  return memcmp(&counts, &held_counts, sizeof(counts)) == 0;
}
int64_t et_i2_construction_test_unhold_v1(void) {
  int32_t status = et_f32_tensor_borrow_end_v1(&held, NULL);
  status |= et_f32_tensor_scoped_end_internal(&scoped);
  status |= et_f32_tensor_copy_plan_release_v1(&copy, NULL);
  status |= et_f32_gradient_reset_plan_release_v1(&reset, NULL);
  status |= et_f32_gradient_plan_release_v1(&gradient, NULL);
  status |= et_f32_tensor_destroy_v1(&numerator, NULL);
  et_f32_tensor_test_reset_allocator_v1();
  return status;
}

/* Include the unchanged native identity implementation through a test-only
 * allocator shim. Only its four construction allocation boundaries can fail;
 * the Eshkol runtime and providers keep their ordinary allocators. */
#include <stdlib.h>
static int64_t construction_allocations = -1;
static void *construction_calloc(size_t count, size_t bytes) {
  if (construction_allocations == 0) {
    construction_allocations = -1;
    return NULL;
  }
  if (construction_allocations > 0) --construction_allocations;
  return calloc(count, bytes);
}
int64_t et_i2_construction_test_fail_after_v1(int64_t count) {
  construction_allocations = count;
  return 0;
}
#define ET_P1_TRUSTED_BUILD 1
#define ET_P1_TEST_HOOKS 1
#define calloc construction_calloc
#include "../../native/p1_identity.c"
#undef calloc
