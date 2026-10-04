#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#define ET_O2_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#include "../../native/o2_optimizer.c"

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); \
} } while (0)

typedef struct fixture {
  et_f32_tensor *initial[ET_O2_CANDIDATE_COUNT];
  et_f32_parameter *parameter[ET_O2_CANDIDATE_COUNT];
  et_o2_optimizer *optimizer;
} fixture;

static void make_fixture(fixture *f) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_error_v1 o2_error;
  et_f32_tensor_error f32_error;
  const uint64_t shape[1] = {80u}; /* Genuine 320-byte writable payload. */
  memset(f, 0, sizeof(*f));
  CHECK(et_o2_optimizer_builder_create_v1(
      ET_O2_CANDIDATE_COUNT, ET_O2_CLIP_NONE, 0u,
      ET_O2_SCHEDULE_CONSTANT, 0u, 0u, UINT32_C(0x3f800000),
      &builder, &o2_error) == 0);
  for (size_t i = 0u; i < ET_O2_CANDIDATE_COUNT; ++i) {
    CHECK(et_f32_tensor_create_v1(1u, shape, &f->initial[i], &f32_error) == 0);
    CHECK(et_f32_parameter_create_v1(f->initial[i], &f->parameter[i],
                                     &f32_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(f->parameter[i],
                                           f->parameter[i], &f32_error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
        builder, i, f->parameter[i], f->parameter[i], UINT32_C(0x3ca3d70a),
        UINT32_C(0x3f4ccccd), UINT32_C(0x3f7d70a4),
        UINT32_C(0x322bcc77), 0u, &o2_error) == 0);
  }
  et_o2_test_fail_alloc_after_v1(1u); /* Optimizer succeeds, roster fails. */
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &f->optimizer,
                                           &o2_error) == ET_O2_STATUS_INTERNAL);
  CHECK(o2_error.code == ET_O2_CODE_ALLOCATION_FAILED);
  CHECK(builder != NULL && f->optimizer == NULL);
  et_o2_test_reset_failpoints_v1();
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &f->optimizer,
                                           &o2_error) == 0);
}

static void cleanup_fixture(fixture *f) {
  et_f32_tensor_error error;
  for (size_t i = 0u; i < ET_O2_CANDIDATE_COUNT; ++i) {
    CHECK(et_f32_parameter_destroy_v1(&f->parameter[i], &error) == 0);
    CHECK(et_f32_tensor_destroy_v1(&f->initial[i], &error) == 0);
  }
}

static void expect_failure(et_o2_optimizer *candidate, uint32_t category,
                           uint32_t code) {
  et_o2_error_v1 error;
  int32_t result = et_o2_private_candidate_retire_preflight_v1(candidate,
                                                                &error);
  if (result != (int32_t)category)
    fprintf(stderr, "expected %u/%u, got %d/%u at %p\n", category, code,
            result, error.code, (void *)candidate);
  CHECK(result == (int32_t)category);
  CHECK(error.category == category);
  CHECK(error.code == code);
  CHECK(strcmp(error.operation, "candidate-optimizer-retire") == 0);
}

static void expect_fail_closed(et_o2_optimizer *candidate) {
  et_o2_error_v1 error;
  memset(&error, 0xa5, sizeof(error));
  CHECK(et_o2_private_candidate_retire_preflight_v1(candidate, &error) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  const unsigned char *bytes = (const unsigned char *)&error;
  for (size_t i = 0u; i < sizeof(error); ++i) CHECK(bytes[i] == 0xa5u);
}

int main(void) {
  fixture first, second;
  et_o2_error_v1 error;
  et_f32_tensor_error f32_error;
  et_f32_tensor_borrow *borrow = NULL;
  et_o2_optimizer_state *state = NULL;
  et_f32_tensor *saved;
  et_f32_tensor *retired_moment;
  et_o2_retire_provenance *retired_record;
  const et_f32_tensor *parameter_value = NULL;
  et_o2_entry *entries;
  size_t count;
  unsigned char payload_before[sizeof(et_o2_error_v1)];
  et_f32_test_tensor_metadata_v1 original_metadata = {
      .struct_size = sizeof(original_metadata)};
  et_f32_test_tensor_metadata_v1 forged_metadata;
  float decoy[80] = {0};
  et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_o2_test_live_counts_v1 o2_before = {.struct_size = sizeof(o2_before)};
  et_o2_test_live_counts_v1 o2_after = {.struct_size = sizeof(o2_after)};

  make_fixture(&first);
  et_f32_test_live_counts_snapshot_v1(&before);
  et_o2_test_live_counts_snapshot_v1(&o2_before);
  CHECK(o2_before.optimizers == 1u);
  CHECK(o2_before.live_entry_bytes ==
        ET_O2_CANDIDATE_COUNT * sizeof(et_o2_entry));
  CHECK(o2_before.live_optimizer_moments == ET_O2_CANDIDATE_COUNT * 2u);
  CHECK(o2_before.live_optimizer_moment_bytes ==
        ET_O2_CANDIDATE_COUNT * 2u * 80u * sizeof(float));
  size_t f32_alloc_before = et_f32_tensor_test_successful_allocations_v1();
  et_o2_test_fail_alloc_after_v1(0u);
  size_t o2_alloc_before = et_o2_successful_allocations;
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) == 0);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == f32_alloc_before);
  CHECK(et_o2_successful_allocations == o2_alloc_before);
  et_o2_test_reset_failpoints_v1();
  saved = first.optimizer->entries[0].exp_avg;
  const float *moment_data = et_f32_tensor_test_data_storage_v1(saved);
  CHECK(moment_data != NULL &&
        (uintptr_t)moment_data % _Alignof(et_o2_error_v1) == 0u);
  memcpy(payload_before, moment_data, sizeof(payload_before));
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)(uintptr_t)moment_data) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, moment_data, sizeof(payload_before)) == 0);
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(saved,
                                                 &original_metadata) == 0);
  forged_metadata = original_metadata;
  forged_metadata.data_storage = decoy;
  CHECK(et_f32_tensor_test_metadata_restore_v1(saved, &forged_metadata) == 0);
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)(uintptr_t)moment_data) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, moment_data, sizeof(payload_before)) == 0);
  memset(&error, 0xa5, sizeof(error));
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  const unsigned char *error_bytes = (const unsigned char *)&error;
  for (size_t i = 0u; i < sizeof(error); ++i)
    CHECK(error_bytes[i] == 0xa5u);
  CHECK(et_f32_tensor_test_metadata_restore_v1(saved,
                                                &original_metadata) == 0);
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) == 0);
  CHECK(et_f32_parameter_value_tensor_v1(first.parameter[0], &parameter_value,
                                          &f32_error) == 0);
  const float *parameter_data =
      et_f32_tensor_test_data_storage_v1(parameter_value);
  CHECK(parameter_data != NULL &&
        (uintptr_t)parameter_data % _Alignof(et_o2_error_v1) == 0u);
  memcpy(payload_before, parameter_data, sizeof(payload_before));
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)(uintptr_t)parameter_data) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, parameter_data, sizeof(payload_before)) == 0);
  CHECK(ET_O2_CANDIDATE_COUNT * sizeof(et_o2_entry) >= sizeof(payload_before));
  memcpy(payload_before, first.optimizer->entries, sizeof(payload_before));
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)first.optimizer->entries) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, first.optimizer->entries,
               sizeof(payload_before)) == 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(first.optimizer, &state,
                                           &error) == 0);
  CHECK(state->count * sizeof(*state->entries) >= sizeof(payload_before));
  memcpy(payload_before, state->entries, sizeof(payload_before));
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)state->entries) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, state->entries, sizeof(payload_before)) == 0);
  CHECK(state->lifecycle == ET_O2_OPTIMIZER_STATE_LIVE);
  expect_fail_closed(first.optimizer);
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     NULL) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_optimizer_state_release_v1(state, &error) == 0);
  expect_failure(NULL, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_INVALID_HANDLE);
  expect_failure((et_o2_optimizer *)(uintptr_t)1u, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_INVALID_HANDLE);
  count = first.optimizer->count;
  first.optimizer->count = 13u;
  expect_fail_closed(first.optimizer);
  first.optimizer->count = 15u;
  expect_fail_closed(first.optimizer);
  first.optimizer->count = count;
  et_o2_entry *original_entries = first.optimizer->entries;
  et_o2_entry decoy_entries[ET_O2_CANDIDATE_COUNT];
  memcpy(decoy_entries, original_entries, sizeof(decoy_entries));
  memcpy(payload_before, original_entries, sizeof(payload_before));
  first.optimizer->entries = decoy_entries;
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)original_entries) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, original_entries, sizeof(payload_before)) == 0);
  expect_fail_closed(first.optimizer);
  first.optimizer->entries = original_entries;
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) == 0);
  first.optimizer->provider_major++;
  expect_failure(first.optimizer, ET_O2_STATUS_INTERNAL,
                 ET_O2_CODE_PROVIDER_DEFECT);
  first.optimizer->provider_major--;
  first.optimizer->busy = 1u;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_ACTIVE_BORROW);
  first.optimizer->busy = 0u;
#ifdef ET_TR3_O2_PRIVATE_OPERATIONS
  first.optimizer->active_operation = first.optimizer;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_ACTIVE_BORROW);
  first.optimizer->active_operation = NULL;
#endif
  for (size_t i = 0u; i < ET_O2_CANDIDATE_COUNT; ++i) {
    for (size_t kind = 0u; kind < 2u; ++kind) {
      size_t peer = (i + 1u) % ET_O2_CANDIDATE_COUNT;
      et_f32_tensor **slot = kind == 0u ? &first.optimizer->entries[i].exp_avg
                                        : &first.optimizer->entries[i].exp_avg_sq;
      saved = *slot;
      *slot = first.optimizer->entries[peer].exp_avg;
      expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                     ET_O2_CODE_INVALID_HANDLE);
      *slot = saved;
    }
  }
  et_f32_parameter *original_parameter = first.optimizer->entries[2].parameter;
  first.optimizer->entries[2].parameter = first.optimizer->entries[1].parameter;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_INVALID_HANDLE);
  first.optimizer->entries[2].parameter = original_parameter;
  const void *original_handle = first.optimizer->entries[2].p1_handle;
  first.optimizer->entries[2].p1_handle = first.optimizer->entries[1].p1_handle;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_INVALID_HANDLE);
  first.optimizer->entries[2].p1_handle = original_handle;
  const et_f32_tensor *original_storage =
      first.optimizer->entries[2].storage_owner;
  first.optimizer->entries[2].storage_owner =
      first.optimizer->entries[1].storage_owner;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_INVALID_HANDLE);
  first.optimizer->entries[2].storage_owner = original_storage;
  saved = first.optimizer->entries[0].exp_avg;
  CHECK(et_f32_tensor_borrow_begin_v1(saved, &borrow, &f32_error) == 0);
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(error.category == ET_O2_STATUS_INVALID_STATE &&
        error.code == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_f32_tensor_is_live_v1(saved) == 1);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &f32_error) == 0);
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) == 0);
  make_fixture(&second);
  saved = second.optimizer->entries[0].exp_avg;
  second.optimizer->entries[0].exp_avg = first.optimizer->entries[0].exp_avg;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_OWNER_CONFLICT);
  second.optimizer->entries[0].exp_avg = saved;
  original_parameter = second.optimizer->entries[0].parameter;
  second.optimizer->entries[0].parameter = first.optimizer->entries[0].parameter;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_OWNER_CONFLICT);
  second.optimizer->entries[0].parameter = original_parameter;
  original_handle = second.optimizer->entries[0].p1_handle;
  second.optimizer->entries[0].p1_handle = first.optimizer->entries[0].p1_handle;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_OWNER_CONFLICT);
  second.optimizer->entries[0].p1_handle = original_handle;
  saved = (et_f32_tensor *)second.optimizer->entries[0].storage_owner;
  second.optimizer->entries[0].storage_owner =
      first.optimizer->entries[0].storage_owner;
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_OWNER_CONFLICT);
  second.optimizer->entries[0].storage_owner = saved;
  CHECK(et_o2_private_candidate_retire_preflight_v1(first.optimizer,
                                                     &error) == 0);
  entries = first.optimizer->entries;
  retired_moment = first.optimizer->entries[0].exp_avg;
  f32_alloc_before = et_f32_tensor_test_successful_allocations_v1();
  et_o2_test_fail_alloc_after_v1(0u);
  o2_alloc_before = et_o2_successful_allocations;
  CHECK(et_o2_private_candidate_retire_commit_v1(first.optimizer,
                                                  &error) == 0);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == f32_alloc_before);
  CHECK(et_o2_successful_allocations == o2_alloc_before);
  et_o2_test_reset_failpoints_v1();
  CHECK(first.optimizer->entries == NULL && first.optimizer->count == 0u);
  CHECK(first.optimizer->magic == ET_O2_RETIRED_OPTIMIZER_MAGIC);
  CHECK(entries != first.optimizer->entries);
  expect_failure(first.optimizer, ET_O2_STATUS_INVALID_STATE,
                 ET_O2_CODE_INVALID_HANDLE);
  retired_record = et_o2_retire_record_find(first.optimizer);
  CHECK(retired_record != NULL &&
        sizeof(*retired_record) + retired_record->count * sizeof(et_o2_entry) >=
            sizeof(payload_before));
  memcpy(payload_before, retired_record, sizeof(payload_before));
  CHECK(et_o2_private_candidate_retire_preflight_v1(
      first.optimizer, (et_o2_error_v1 *)retired_record) ==
        ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(payload_before, retired_record, sizeof(payload_before)) == 0);
  CHECK(et_f32_tensor_is_live_v1(retired_moment) == 0);
  et_f32_test_live_counts_snapshot_v1(&after);
  et_o2_test_live_counts_snapshot_v1(&o2_after);
  CHECK(o2_after.optimizers == 1u && o2_after.retired_optimizers == 1u);
  CHECK(o2_after.live_entry_bytes ==
        ET_O2_CANDIDATE_COUNT * sizeof(et_o2_entry));
  CHECK(o2_after.live_optimizer_moments == ET_O2_CANDIDATE_COUNT * 2u);
  CHECK(o2_after.live_optimizer_moment_bytes ==
        ET_O2_CANDIDATE_COUNT * 2u * 80u * sizeof(float));
  CHECK(before.tensors + ET_O2_CANDIDATE_COUNT * 5u ==
        after.tensors + ET_O2_CANDIDATE_COUNT * 2u);
  CHECK(et_o2_private_candidate_retire_commit_v1(second.optimizer,
                                                  &error) == 0);
  cleanup_fixture(&first);
  cleanup_fixture(&second);

  /* A defect after the first free is terminal, never a retryable result. */
  make_fixture(&first);
  pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    et_o2_test_retire_fail_after_v1(1u);
    (void)et_o2_private_candidate_retire_commit_v1(first.optimizer, &error);
    _Exit(99);
  }
  int status = 0;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 134);
  CHECK(et_o2_private_candidate_retire_commit_v1(first.optimizer,
                                                  &error) == 0);
  cleanup_fixture(&first);
  printf("G3R O2 retirement PASS: %zu checks\n", checks);
  return 0;
}
