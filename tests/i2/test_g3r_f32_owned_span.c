#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_F32_TENSOR_TESTING 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#include "../../native/f32_tensor.c"

#include <stdio.h>

static size_t checks;
#define CHECK(condition) do { ++checks; if (!(condition)) {                    \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); exit(1);    \
} } while (0)

static et_f32_tensor *make_tensor(void) {
  et_f32_tensor *tensor = NULL;
  const uint64_t shape[] = {80u}; /* 320 genuine writable data bytes. */
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_create_v1(1u, shape, &tensor, &error) == 0);
  return tensor;
}

int main(void) {
  et_f32_tensor_error error;
  et_f32_tensor *source = make_tensor();
  et_f32_tensor *target = make_tensor();
  et_f32_tensor *clone = NULL;
  et_f32_parameter *parameter = NULL;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_copy_plan *copy_plan = NULL;
  et_f32_gradient_plan *gradient_plan = NULL;
  et_f32_gradient_reset_plan *reset_plan = NULL;
  et_f32_test_tensor_metadata_v1 original = {
      .struct_size = sizeof(original)};
  unsigned char caller[264];
  unsigned char caller_before[264];
  memset(caller, 0xa5, sizeof(caller));
  memcpy(caller_before, caller, sizeof(caller));
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(NULL, 264u) == 2);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller, 0u) == 2);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      (const void *)(uintptr_t)(UINTPTR_MAX - 3u), 8u) == 2);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);
  CHECK(memcmp(caller, caller_before, sizeof(caller)) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1((void *)(uintptr_t)1u,
                                                     1u) == 0);
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(source, &original) == 0);
  CHECK(original.byte_length >= sizeof(caller));
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(original.data_storage,
                                                     sizeof(caller)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      (const unsigned char *)original.data_storage + original.byte_length - 1u,
      1u) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(original.shape_storage,
                                                     sizeof(uint64_t)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(original.stride_storage,
                                                     sizeof(size_t)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(source,
                                                     sizeof(*source)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      f32_destroy_record_find(source), sizeof(f32_destroy_provenance)) == 1);

  /* The old descriptor query misses this exact genuine payload. */
  float *decoy = (float *)malloc(original.byte_length);
  CHECK(decoy != NULL);
  et_f32_test_tensor_metadata_v1 forged = original;
  forged.data_storage = decoy;
  CHECK(et_f32_tensor_test_metadata_restore_v1(source, &forged) == 0);
  unsigned char data_before[264];
  memcpy(data_before, original.data_storage, sizeof(data_before));
  CHECK(et_f32_tensor_private_storage_overlap_v1(
      original.data_storage, sizeof(data_before)) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      original.data_storage, sizeof(data_before)) == 1);
  CHECK(memcmp(data_before, original.data_storage, sizeof(data_before)) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  CHECK(et_f32_tensor_test_metadata_restore_v1(source, &original) == 0);
  free(decoy);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);
  for (uint32_t member = ET_F32_TEST_TENSOR_METADATA_RANK;
       member <= ET_F32_TEST_TENSOR_METADATA_DATA; ++member) {
    CHECK(et_f32_tensor_test_metadata_corrupt_v1(source, member, 0u) == 0);
    CHECK(et_f32_tensor_private_owned_span_overlap_v1(
        original.data_storage, sizeof(caller)) == 1);
    CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                       sizeof(caller)) == 3);
    CHECK(et_f32_tensor_test_metadata_restore_v1(source, &original) == 0);
    CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                       sizeof(caller)) == 0);
  }
  f32_destroy_provenance *record = f32_destroy_record_find(source);
  CHECK(record != NULL);
  record->bytes++;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  record->bytes--;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);

  CHECK(et_f32_parameter_create_v1(source, &parameter, &error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, parameter, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(parameter,
                                                     sizeof(*parameter)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(parameter->value->data,
                                                     sizeof(caller)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      parameter->gradient->data, sizeof(caller)) == 1);
  et_f32_tensor *saved_value = parameter->value;
  parameter->value = (et_f32_tensor *)(uintptr_t)1u;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(saved_value->data,
                                                     sizeof(caller)) == 1);
  parameter->value = saved_value;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);
  CHECK(et_f32_owned_tensor_clone_v1(source, &clone, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(clone->data,
                                                     sizeof(caller)) == 1);
  CHECK(et_f32_tensor_borrow_begin_v1(source, &borrow, &error) == 0);
  et_f32_tensor_borrow *retired_borrow = borrow;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(borrow,
                                                     sizeof(*borrow)) == 1);
  et_f32_tensor *saved_borrow_owner = borrow->owner;
  borrow->owner = (et_f32_tensor *)(uintptr_t)1u;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  borrow->owner = saved_borrow_owner;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(retired_borrow,
                                                     sizeof(*retired_borrow)) == 1);

  et_f32_tensor_copy_assignment_v1 assignment = {
      .struct_size = sizeof(assignment), .destination = target,
      .source = source};
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &assignment, &copy_plan,
                                            &error) == 0);
  et_f32_tensor_copy_plan *retired_copy_plan = copy_plan;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(copy_plan,
                                                     sizeof(*copy_plan)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      copy_plan->assignments, sizeof(*copy_plan->assignments)) == 3);
  et_f32_tensor_copy_assignment_v1 *saved_assignments = copy_plan->assignments;
  copy_plan->assignments = NULL;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  copy_plan->assignments = saved_assignments;
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  CHECK(et_f32_tensor_copy_plan_release_v1(&copy_plan, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(retired_copy_plan,
                                                     sizeof(*retired_copy_plan)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);

  et_f32_parameter *parameters[] = {parameter};
  CHECK(et_f32_gradient_reset_plan_prepare_v1(1u, parameters, &reset_plan,
                                               &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(reset_plan,
                                                     sizeof(*reset_plan)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset_plan, &error) == 0);

  et_f32_gradient_contribution_v1 contribution = {
      .struct_size = sizeof(contribution), .destination = parameter,
      .weighted_numerator = source, .expected_ordinal = 0u};
  CHECK(et_f32_gradient_plan_prepare_v1(
      1u, &contribution, UINT32_C(0x3f800000), &gradient_plan, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      gradient_plan, sizeof(*gradient_plan)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      gradient_plan->entries, sizeof(*gradient_plan->entries)) == 3);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      gradient_plan->entries[0].prepared, sizeof(caller)) == 3);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 3);
  CHECK(et_f32_gradient_plan_release_v1(&gradient_plan, &error) == 0);

  et_f32_test_borrow_event_counts_v1 events_before = {
      .struct_size = sizeof(events_before)};
  et_f32_test_borrow_event_counts_v1 events_after = {
      .struct_size = sizeof(events_after)};
  et_f32_test_live_counts_v1 live_before = {.struct_size = sizeof(live_before)};
  et_f32_test_live_counts_v1 live_after = {.struct_size = sizeof(live_after)};
  et_f32_test_retired_counts_v1 dead_before = {
      .struct_size = sizeof(dead_before)};
  et_f32_test_retired_counts_v1 dead_after = {
      .struct_size = sizeof(dead_after)};
  et_f32_test_borrow_event_counts_snapshot_v1(&events_before);
  et_f32_test_live_counts_snapshot_v1(&live_before);
  et_f32_test_retired_counts_snapshot_v1(&dead_before);
  et_f32_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(caller,
                                                     sizeof(caller)) == 0);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == 0u);
  et_f32_tensor_test_reset_allocator_v1();
  et_f32_test_borrow_event_counts_snapshot_v1(&events_after);
  et_f32_test_live_counts_snapshot_v1(&live_after);
  et_f32_test_retired_counts_snapshot_v1(&dead_after);
  CHECK(memcmp(&events_before, &events_after, sizeof(events_before)) == 0);
  CHECK(memcmp(&live_before, &live_after, sizeof(live_before)) == 0);
  CHECK(memcmp(&dead_before, &dead_after, sizeof(dead_before)) == 0);
  CHECK(memcmp(caller, caller_before, sizeof(caller)) == 0);

  et_f32_tensor *stale_clone = clone;
  CHECK(et_f32_owned_tensor_release_v1(clone, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(stale_clone,
                                                     sizeof(*stale_clone)) == 1);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  CHECK(et_f32_tensor_destroy_v1(&target, &error) == 0);
  uintptr_t old_data = (uintptr_t)original.data_storage;
  et_f32_tensor *stale_source = source;
  CHECK(et_f32_tensor_destroy_v1(&source, &error) == 0);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(stale_source,
                                                     sizeof(*stale_source)) == 1);
  CHECK(et_f32_tensor_private_owned_span_overlap_v1(
      (const void *)old_data, sizeof(caller)) == 0);
  printf("G3R f32 owned span PASS: %zu checks\n", checks);
  return 0;
}
