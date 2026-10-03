#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(expr)                                                            \
  do {                                                                         \
    ++checks;                                                                   \
    if (!(expr)) {                                                              \
      ++failures;                                                               \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);         \
    }                                                                           \
  } while (0)

static void expect_status(const et_f32_tensor *tensor, uint32_t category,
                          uint32_t code) {
  et_f32_tensor_error error;
  memset(&error, 0xa5, sizeof(error));
  int32_t actual = et_f32_tensor_private_destroy_ready_v1(tensor, &error);
  if (actual != (int32_t)category || error.category != category ||
      error.code != code) {
    fprintf(stderr, "STATUS %p got %d/%u/%u expected %u/%u\n",
            (const void *)tensor, actual, error.category, error.code,
            category, code);
  }
  CHECK(actual == (int32_t)category);
  CHECK(error.category == category);
  CHECK(error.code == code);
}

static et_f32_tensor *make_tensor(uint64_t extent) {
  et_f32_tensor *result = NULL;
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_create_v1(1u, &extent, &result, &error) == 0);
  CHECK(result != NULL);
  return result;
}

static void release(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static et_f32_test_live_counts_v1 live_counts(void) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  return counts;
}

static void basic_and_lifetime(void) {
  et_f32_tensor *scalar = NULL;
  et_f32_tensor *empty;
  et_f32_tensor *tensor;
  et_f32_tensor *stale;
  et_f32_tensor_error error;
  et_f32_test_live_counts_v1 before;
  et_f32_test_borrow_event_counts_v1 events_before = {
      .struct_size = sizeof(events_before)};
  et_f32_test_borrow_event_counts_v1 events_after = {
      .struct_size = sizeof(events_after)};
  uint32_t bits[80] = {0};
  uint32_t after[80] = {0};
  int foreign = 0;
  CHECK(et_f32_tensor_create_v1(0u, NULL, &scalar, &error) == 0);
  empty = make_tensor(0u);
  tensor = make_tensor(80u);
  for (size_t i = 0u; i < 80u; ++i) bits[i] = UINT32_C(0x3f800000) + (uint32_t)i;
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, 80u, &error) == 0);
  before = live_counts();
  CHECK(et_f32_tensor_test_destroy_record_count_v1() == before.tensors);
  CHECK(et_f32_tensor_test_destroy_record_bytes_v1() > 0u);
  et_f32_test_borrow_event_counts_snapshot_v1(&events_before);
  size_t allocations = et_f32_tensor_test_successful_allocations_v1();
  expect_status(scalar, ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_CODE_OK);
  expect_status(empty, ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_CODE_OK);
  expect_status(tensor, ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_CODE_OK);
  CHECK(et_f32_tensor_private_destroy_ready_v1(tensor, NULL) == 0);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == allocations);
  et_f32_test_borrow_event_counts_snapshot_v1(&events_after);
  CHECK(memcmp(&events_before, &events_after, sizeof(events_before)) == 0);
  CHECK(live_counts().tensors == before.tensors);
  CHECK(et_f32_tensor_test_destroy_record_count_v1() == before.tensors);
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, after, 80u, &error) == 0);
  CHECK(memcmp(bits, after, sizeof(bits)) == 0);
  expect_status(NULL, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                ET_F32_TENSOR_CODE_NULL_ARGUMENT);
  expect_status((const et_f32_tensor *)&foreign,
                ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  stale = tensor;
  release(&tensor);
  expect_status(stale, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  et_f32_tensor_error saved;
  size_t control_bytes = et_f32_tensor_test_control_bytes_v1();
  CHECK(control_bytes <= sizeof(saved));
  memcpy(&saved, stale, control_bytes);
  CHECK(et_f32_tensor_private_destroy_ready_v1(
            scalar, (et_f32_tensor_error *)stale) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&saved, stale, control_bytes) == 0);
  release(&empty);
  release(&scalar);
}

static void associations_and_aliases(void) {
  et_f32_tensor *source = make_tensor(80u);
  et_f32_tensor *other = make_tensor(80u);
  et_f32_tensor *private_clone = NULL;
  et_f32_parameter *parameter = NULL;
  const et_f32_tensor *value = NULL;
  et_f32_tensor_error error;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_copy_plan *plan = NULL;
  et_f32_tensor_copy_assignment_v1 assignment = {
      .struct_size = sizeof(assignment), .destination = other, .source = source};
  CHECK(et_f32_tensor_borrow_begin_v1(source, &borrow, &error) == 0);
  expect_status(source, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_f32_tensor_copy_plan_prepare_v1(1u, &assignment, &plan, &error) == 0);
  expect_status(source, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_tensor_copy_plan_release_v1(&plan, &error) == 0);
  CHECK(et_f32_parameter_create_v1(source, &parameter, &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &value, &error) == 0);
  expect_status(value, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  expect_status(et_f32_parameter_test_gradient_tensor_v1(parameter),
                ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_owned_tensor_clone_v1(source, &private_clone, &error) == 0);
  expect_status(private_clone, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_owned_tensor_release_v1(private_clone, &error) == 0);
  CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);

  et_f32_test_tensor_metadata_v1 original = {.struct_size = sizeof(original)};
  et_f32_test_tensor_metadata_v1 aliased = {.struct_size = sizeof(aliased)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(other, &original) == 0);
  aliased = original;
  aliased.data_storage = (float *)et_f32_tensor_test_data_storage_v1(source);
  CHECK(et_f32_tensor_test_metadata_restore_v1(other, &aliased) == 0);
  expect_status(source, ET_F32_TENSOR_ERROR_INTERNAL,
                ET_F32_TENSOR_CODE_PROVIDER_REJECTED);
  CHECK(et_f32_tensor_test_metadata_restore_v1(other, &original) == 0);
  expect_status(source, ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_CODE_OK);
  aliased = original;
  aliased.shape_storage = (uint64_t *)et_f32_tensor_test_shape_storage_v1(source);
  CHECK(et_f32_tensor_test_metadata_restore_v1(other, &aliased) == 0);
  expect_status(source, ET_F32_TENSOR_ERROR_INTERNAL,
                ET_F32_TENSOR_CODE_PROVIDER_REJECTED);
  CHECK(et_f32_tensor_test_metadata_restore_v1(other, &original) == 0);

  et_f32_tensor_error saved;
  const float *data = et_f32_tensor_test_data_storage_v1(source);
  memcpy(&saved, data, sizeof(saved));
  CHECK(et_f32_tensor_private_destroy_ready_v1(
            source, (et_f32_tensor_error *)(void *)data) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&saved, data, sizeof(saved)) == 0);
  size_t control_bytes = et_f32_tensor_test_control_bytes_v1();
  CHECK(control_bytes <= sizeof(saved));
  memcpy(&saved, source, control_bytes);
  CHECK(et_f32_tensor_private_destroy_ready_v1(
            source, (et_f32_tensor_error *)source) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&saved, source, control_bytes) == 0);
  release(&other);
  release(&source);
}

static void metadata_and_allocations(void) {
  et_f32_tensor *tensor = make_tensor(80u);
  et_f32_test_tensor_metadata_v1 original = {.struct_size = sizeof(original)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(tensor, &original) == 0);
  et_f32_test_tensor_metadata_v1 shifted = original;
  shifted.data_storage = original.data_storage + 1;
  CHECK(et_f32_tensor_test_metadata_restore_v1(tensor, &shifted) == 0);
  expect_status(tensor, ET_F32_TENSOR_ERROR_INTERNAL,
                ET_F32_TENSOR_CODE_PROVIDER_REJECTED);
  CHECK(et_f32_tensor_test_metadata_restore_v1(tensor, &original) == 0);
  for (uint32_t member = ET_F32_TEST_TENSOR_METADATA_RANK;
       member <= ET_F32_TEST_TENSOR_METADATA_DATA; ++member) {
    CHECK(et_f32_tensor_test_metadata_corrupt_v1(tensor, member, 0u) == 0);
    expect_status(tensor, ET_F32_TENSOR_ERROR_INTERNAL,
                  ET_F32_TENSOR_CODE_PROVIDER_REJECTED);
    CHECK(et_f32_tensor_test_metadata_restore_v1(tensor, &original) == 0);
    expect_status(tensor, ET_F32_TENSOR_ERROR_NONE, ET_F32_TENSOR_CODE_OK);
  }
  release(&tensor);
  et_f32_test_live_counts_v1 baseline = live_counts();
  size_t record_bytes = et_f32_tensor_test_destroy_record_bytes_v1();
  for (size_t cut = 0u; cut < 5u; ++cut) {
    uint64_t extent = 1u;
    tensor = NULL;
    et_f32_tensor_error error;
    et_f32_tensor_test_fail_alloc_after_v1(cut);
    CHECK(et_f32_tensor_create_v1(1u, &extent, &tensor, &error) ==
          ET_F32_TENSOR_ERROR_INTERNAL);
    CHECK(error.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(tensor == NULL);
    CHECK(live_counts().tensors == baseline.tensors);
    CHECK(et_f32_tensor_test_destroy_record_bytes_v1() == record_bytes);
    et_f32_tensor_test_reset_allocator_v1();
  }
  tensor = make_tensor(1u);
  CHECK(et_f32_tensor_test_destroy_record_count_v1() ==
        live_counts().tensors);
  release(&tensor);
  CHECK(et_f32_tensor_test_destroy_record_bytes_v1() == record_bytes);
}

static void owned_clone_allocation_cuts(void) {
  et_f32_tensor *source = make_tensor(4u);
  const uint32_t bits[4] = {UINT32_C(0x80000000), UINT32_C(0x3f800000),
                            UINT32_C(0x7fc01234), UINT32_C(0xff7fffff)};
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_copy_bits_from_v1(source, bits, 4u, &error) == 0);
  et_f32_test_tensor_metadata_v1 original = {.struct_size = sizeof(original)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(source, &original) == 0);
  const et_f32_test_live_counts_v1 baseline = live_counts();
  const size_t record_count = et_f32_tensor_test_destroy_record_count_v1();
  const size_t record_bytes = et_f32_tensor_test_destroy_record_bytes_v1();

  /* Rank-one, nonempty clone: control, shape, stride, data, provenance. */
  for (size_t cut = 0u; cut < 5u; ++cut) {
    et_f32_tensor *clone = NULL;
    et_f32_test_tensor_metadata_v1 observed = {
        .struct_size = sizeof(observed)};
    et_f32_tensor_test_fail_alloc_after_v1(cut);
    CHECK(et_f32_owned_tensor_clone_v1(source, &clone, &error) ==
          ET_F32_TENSOR_ERROR_INTERNAL);
    CHECK(error.category == ET_F32_TENSOR_ERROR_INTERNAL);
    CHECK(error.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(clone == NULL);
    CHECK(et_f32_tensor_test_successful_allocations_v1() == cut);
    CHECK(live_counts().tensors == baseline.tensors);
    CHECK(live_counts().owned_clones == baseline.owned_clones);
    CHECK(et_f32_tensor_test_destroy_record_count_v1() == record_count);
    CHECK(et_f32_tensor_test_destroy_record_bytes_v1() == record_bytes);
    CHECK(et_f32_tensor_test_metadata_snapshot_v1(source, &observed) == 0);
    CHECK(memcmp(&original, &observed, sizeof(original)) == 0);
    CHECK(memcmp(et_f32_tensor_test_data_storage_v1(source), bits,
                 sizeof(bits)) == 0);
    et_f32_tensor_test_reset_allocator_v1();
  }

  et_f32_tensor *clone = NULL;
  uint32_t cloned_bits[4] = {0};
  CHECK(et_f32_owned_tensor_clone_v1(source, &clone, &error) == 0);
  CHECK(clone != NULL);
  CHECK(live_counts().tensors == baseline.tensors + 1u);
  CHECK(live_counts().owned_clones == baseline.owned_clones + 1u);
  CHECK(et_f32_tensor_test_destroy_record_count_v1() == record_count + 1u);
  CHECK(et_f32_tensor_copy_bits_to_v1(clone, cloned_bits, 4u, &error) == 0);
  CHECK(memcmp(cloned_bits, bits, sizeof(bits)) == 0);
  expect_status(clone, ET_F32_TENSOR_ERROR_INVALID_STATE,
                ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_f32_owned_tensor_release_v1(clone, &error) == 0);
  CHECK(live_counts().tensors == baseline.tensors);
  CHECK(live_counts().owned_clones == baseline.owned_clones);
  CHECK(et_f32_tensor_test_destroy_record_count_v1() == record_count);
  CHECK(et_f32_tensor_test_destroy_record_bytes_v1() == record_bytes);
  release(&source);
}

int main(void) {
  basic_and_lifetime();
  associations_and_aliases();
  metadata_and_allocations();
  owned_clone_allocation_cuts();
  if (failures != 0u) return 1;
  printf("G3R f32 destroy readiness PASS: %u checks\n", checks);
  return 0;
}
