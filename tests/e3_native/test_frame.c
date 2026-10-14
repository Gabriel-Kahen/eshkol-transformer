#include "eshkol_transformer/kernel_abi.h"

int32_t et_e3_test_observed_runtime_discover_v1(
    et_kernel_provider_resolver_v1 resolver, void *context,
    et_kernel_runtime **runtime_output, et_kernel_error *error);
void et_e3_test_observed_runtime_destroy_v1(et_kernel_runtime *runtime);
int32_t et_e3_test_observed_kernel_dispatch_v1(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error);

#define ET_E3_TESTING 1
#define ET_E3_FAULT_TESTING 1
#define ET_E3_PARTIAL_PIN_TESTING 1
#define et_f32_tensor_destroy_v1 et_e3_test_observed_tensor_destroy_v1
#define et_kernel_runtime_dispatch et_e3_test_observed_kernel_dispatch_v1
#define et_kernel_runtime_discover et_e3_test_observed_runtime_discover_v1
#define et_kernel_runtime_destroy et_e3_test_observed_runtime_destroy_v1
#include "../../src/eshkol_transformer/e3_frame.c"
#undef et_f32_tensor_destroy_v1
#undef et_kernel_runtime_dispatch
#undef et_kernel_runtime_discover
#undef et_kernel_runtime_destroy

#include <math.h>
#include <stdio.h>

int32_t et_f32_tensor_destroy_v1(et_f32_tensor **tensor,
                                 et_f32_tensor_error *error);
int32_t et_kernel_runtime_dispatch(const et_kernel_runtime *runtime,
                                   const et_kernel_call_v1 *call,
                                   et_kernel_error *error);

static size_t observed_dispatches;
static et_kernel_runtime *live_runtimes[64];
static size_t live_runtime_count, runtime_creates, runtime_destroys;

int32_t et_e3_test_observed_runtime_discover_v1(
    et_kernel_provider_resolver_v1 resolver, void *context,
    et_kernel_runtime **runtime_output, et_kernel_error *error) {
  const int32_t rc = et_kernel_runtime_discover(
      resolver, context, runtime_output, error);
  if (rc == 0) {
    if (!runtime_output || !*runtime_output ||
        live_runtime_count == sizeof(live_runtimes) / sizeof(live_runtimes[0])) {
      fprintf(stderr, "runtime track create capacity %zu\n", live_runtime_count);
      abort();
    }
    for (size_t i = 0; i < live_runtime_count; ++i)
      if (live_runtimes[i] == *runtime_output) {
        fprintf(stderr, "runtime track duplicate\n");
        abort();
      }
    live_runtimes[live_runtime_count++] = *runtime_output;
    ++runtime_creates;
  }
  return rc;
}

void et_e3_test_observed_runtime_destroy_v1(et_kernel_runtime *runtime) {
  if (runtime) {
    size_t index = 0;
    while (index < live_runtime_count && live_runtimes[index] != runtime)
      ++index;
    if (index == live_runtime_count) {
      fprintf(stderr, "runtime track unknown destroy %p count %zu\n",
              (void *)runtime, live_runtime_count);
      abort();
    }
    live_runtimes[index] = live_runtimes[--live_runtime_count];
    ++runtime_destroys;
  }
  et_kernel_runtime_destroy(runtime);
}

int32_t et_e3_test_observed_kernel_dispatch_v1(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  ++observed_dispatches;
  return et_kernel_runtime_dispatch(runtime, call, error);
}

static et_f32_tensor *destroyed_tensors[64];
static size_t destroyed_tensor_count;

int32_t et_e3_test_observed_tensor_destroy_v1(
    et_f32_tensor **tensor, et_f32_tensor_error *error) {
  if (tensor && *tensor && destroyed_tensor_count < 64)
    destroyed_tensors[destroyed_tensor_count++] = *tensor;
  return et_f32_tensor_destroy_v1(tensor, error);
}

static size_t checks;
#define CHECK(value) do { ++checks; if (!(value)) { \
  fprintf(stderr, "FAIL line %d: %s [domain=%lld category=%lld code=%lld]\n", \
      __LINE__, #value, (long long)e3_last_domain, \
      (long long)e3_last_category, (long long)e3_last_code); \
  exit(1); \
} } while (0)
#define OK(value) CHECK((value) == 0)

static unsigned char identities[14];
static et_f32_tensor_error tensor_error;

static void expect_local_invariant(int64_t status) {
  CHECK(status == E3_INTERNAL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_INVARIANT);
}

static void test_diagnostic_mapping(void) {
  CHECK(e3_dependency(1, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                      ET_KERNEL_CODE_NULL_ARGUMENT) == E3_INVALID_ARGUMENT);
  CHECK(e3_last_domain == 1 &&
        e3_last_category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
        e3_last_code == ET_KERNEL_CODE_NULL_ARGUMENT);
  CHECK(e3_dependency(1, ET_KERNEL_ERROR_UNSUPPORTED,
                      ET_KERNEL_CODE_PROVIDER_REJECTED) == E3_UNSUPPORTED);
  expect_local_invariant(e3_dependency(0, ET_KERNEL_ERROR_INTERNAL,
                                      ET_KERNEL_CODE_PROVIDER_REJECTED));
  expect_local_invariant(e3_dependency(1, ET_KERNEL_ERROR_NONE,
                                      ET_KERNEL_CODE_NULL_ARGUMENT));
  expect_local_invariant(e3_dependency(1, ET_KERNEL_ERROR_INTERNAL + 1,
                                      ET_KERNEL_CODE_NULL_ARGUMENT));
  expect_local_invariant(e3_dependency(1, ET_KERNEL_ERROR_INTERNAL,
                                      ET_KERNEL_CODE_OK));
  expect_local_invariant(e3_dependency(1, ET_KERNEL_ERROR_INTERNAL,
                                      ET_KERNEL_CODE_PROVIDER_REJECTED + 1));

  CHECK(e3_dependency(2, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                      ET_I64_TENSOR_CODE_NULL_ARGUMENT) == E3_INVALID_ARGUMENT);
  CHECK(e3_dependency(2, ET_I64_TENSOR_ERROR_NONCONTIGUOUS,
                      ET_I64_TENSOR_CODE_INVALID_BUFFER) == E3_INVALID_ARGUMENT);
  CHECK(e3_dependency(2, ET_I64_TENSOR_ERROR_INVALID_STATE,
                      ET_I64_TENSOR_CODE_ACTIVE_BORROW) == E3_INVALID_STATE);
  CHECK(e3_dependency(2, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
                      ET_I64_TENSOR_CODE_INVALID_SHAPE) == E3_SHAPE_MISMATCH);
  CHECK(e3_dependency(2, ET_I64_TENSOR_ERROR_VERSION_MISMATCH,
                      ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH) == E3_INTERNAL);
  CHECK(e3_last_domain == 2 &&
        e3_last_category == ET_I64_TENSOR_ERROR_VERSION_MISMATCH &&
        e3_last_code == ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH);
  expect_local_invariant(e3_dependency(2, ET_I64_TENSOR_ERROR_INTERNAL + 1,
                                      ET_I64_TENSOR_CODE_NULL_ARGUMENT));
  expect_local_invariant(e3_dependency(2, ET_I64_TENSOR_ERROR_INTERNAL,
                                      ET_I64_TENSOR_CODE_PROVIDER_REJECTED + 1));

  CHECK(e3_dependency(3, ET_F32_TENSOR_ERROR_INVALID_STATE,
                      ET_F32_TENSOR_CODE_ACTIVE_BORROW) == E3_INVALID_STATE);
  CHECK(e3_dependency(3, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                      ET_F32_TENSOR_CODE_INVALID_SHAPE) == E3_SHAPE_MISMATCH);
  expect_local_invariant(e3_dependency(3, ET_F32_TENSOR_ERROR_INTERNAL + 1,
                                      ET_F32_TENSOR_CODE_NULL_ARGUMENT));
  expect_local_invariant(e3_dependency(3, ET_F32_TENSOR_ERROR_INTERNAL,
                                      ET_F32_TENSOR_CODE_FLOAT_ENVIRONMENT + 1));

  static const struct {
    int64_t category, code, status, local_code;
  } valid_m3[] = {
    {ET_M3T_INVALID_ARGUMENT, ET_M3T_CODE_IDENTITY,
     E3_INVALID_ARGUMENT, E3_CODE_IDENTITY},
    {ET_M3T_INVALID_STATE, ET_M3T_CODE_LIFECYCLE,
     E3_INVALID_STATE, E3_CODE_PHASE},
    {ET_M3T_INTERNAL, ET_M3T_CODE_ALLOCATION,
     E3_INTERNAL, E3_CODE_ALLOCATION},
    {ET_M3T_SHAPE_MISMATCH, ET_M3T_CODE_SHAPE,
     E3_SHAPE_MISMATCH, E3_CODE_SHAPE},
    {ET_M3T_INVALID_STATE, ET_M3T_CODE_REENTRANCY,
     E3_INVALID_STATE, E3_CODE_REENTRANCY},
    {ET_M3T_INTERNAL, ET_M3T_CODE_INTERNAL,
     E3_INTERNAL, E3_CODE_INVARIANT},
  };
  for (size_t i = 0; i < sizeof(valid_m3) / sizeof(valid_m3[0]); ++i) {
    error_domain = 0;
    error_category = valid_m3[i].category;
    error_code = valid_m3[i].code;
    CHECK(e3_m3_error() == valid_m3[i].status);
    CHECK(e3_last_domain == 0 && e3_last_category == valid_m3[i].status &&
          e3_last_code == valid_m3[i].local_code);
  }
  error_domain = 0;
  error_category = ET_M3T_INVALID_ARGUMENT;
  error_code = ET_M3T_CODE_ALLOCATION;
  expect_local_invariant(e3_m3_error());
  error_category = ET_M3T_INTERNAL + 1;
  error_code = ET_M3T_CODE_INTERNAL;
  expect_local_invariant(e3_m3_error());
  error_category = ET_M3T_INTERNAL;
  error_code = ET_M3T_CODE_INTERNAL + 1;
  expect_local_invariant(e3_m3_error());

  error_domain = 2;
  error_category = ET_I64_TENSOR_ERROR_NONCONTIGUOUS;
  error_code = ET_I64_TENSOR_CODE_INVALID_BUFFER;
  CHECK(e3_m3_error() == E3_INVALID_ARGUMENT);
  CHECK(e3_last_domain == 2 &&
        e3_last_category == ET_I64_TENSOR_ERROR_NONCONTIGUOUS &&
        e3_last_code == ET_I64_TENSOR_CODE_INVALID_BUFFER);
  error_domain = 4;
  error_category = 1;
  error_code = 1;
  expect_local_invariant(e3_m3_error());
}

static owner *make_model(void) {
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial != NULL);
  OK(et_m3t_private_initializer_seal_v1(initial));
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model != NULL);
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_bind_v1(model, (int64_t)i, &identities[i]));
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_initialize_v1(model, (int64_t)i));
  OK(et_m3t_private_owner_seal_v1(model));
  return model;
}

static et_f32_tensor *scalar(void) {
  et_f32_tensor *tensor = NULL;
  OK(et_f32_tensor_create_v1(0, NULL, &tensor, &tensor_error));
  return tensor;
}

static void destroy_tensor(et_f32_tensor **tensor) {
  OK(et_f32_tensor_destroy_v1(tensor, &tensor_error));
}

static void read_scalar(et_f32_tensor *tensor, uint32_t *bits) {
  OK(et_f32_tensor_copy_bits_to_v1(tensor, bits, 1, &tensor_error));
}

static void write_scalar(et_f32_tensor *tensor, uint32_t bits) {
  OK(et_f32_tensor_copy_bits_from_v1(tensor, &bits, 1, &tensor_error));
}

static et_kernel_tensor_view_v1 i64_pair(int64_t words[2]) {
  return e3_native_view(words, 16, "i64", 2, e3_shape_pair);
}

static et_kernel_tensor_view_v1 bool_pair(unsigned char words[2]) {
  return e3_native_view(words, 2, "bool", 2, e3_shape_pair);
}

static void expect_input_alias_unchanged(
    et_e3_frame_internal *frame, const et_kernel_tensor_view_v1 *view) {
  et_e3_frame_internal before;
  uint32_t destinations_before[4];
  memcpy(&before, frame, sizeof(before));
  for (size_t i = 0; i < 4; ++i)
    read_scalar(frame->destination[i], &destinations_before[i]);
  CHECK(et_e3_private_stage_inputs_v1(frame, view) == E3_INVALID_ARGUMENT);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INVALID_ARGUMENT &&
        e3_last_code == E3_CODE_DESTINATION);
  CHECK(memcmp(frame, &before, sizeof(before)) == 0);
  for (size_t i = 0; i < 4; ++i) {
    uint32_t after;
    read_scalar(frame->destination[i], &after);
    CHECK(after == destinations_before[i]);
  }
  CHECK(frame->phase == E3_ACQUIRED && frame->stage_plane == 0);
}

static void test_stage_aliases(et_e3_frame_internal *frame) {
  int64_t words[2] = {0, 1};
  uint64_t shape[2] = {1, 2};
  et_kernel_tensor_view_v1 view = e3_native_view(words, sizeof(words), "i64",
                                                 2, shape);

  expect_input_alias_unchanged(
      frame, (const et_kernel_tensor_view_v1 *)(const void *)&frame->counts);

  view.shape = (const uint64_t *)(const void *)&frame->counts;
  expect_input_alias_unchanged(frame, &view);
  view.shape = shape;
  view.data = frame->ids;
  expect_input_alias_unchanged(frame, &view);

  view.data = words;
  expect_input_alias_unchanged(
      frame, (const et_kernel_tensor_view_v1 *)(const void *)frame->destination[0]);
  view.shape = (const uint64_t *)(const void *)frame->destination[0];
  expect_input_alias_unchanged(frame, &view);
  view.shape = shape;
  view.data = frame->destination[0];
  expect_input_alias_unchanged(frame, &view);
  et_f32_scoped_guard_internal guard = {0};
  OK(et_f32_tensor_scoped_begin_internal(frame->destination[0], &guard,
                                          &tensor_error));
  view.data = guard.view.data;
  OK(et_f32_tensor_scoped_end_internal(&guard));
  expect_input_alias_unchanged(frame, &view);

  view.data = words;
  view.shape = (const uint64_t *)(const void *)&view;
  expect_input_alias_unchanged(frame, &view);
  view.shape = shape;
  view.data = &view;
  expect_input_alias_unchanged(frame, &view);
  view.data = shape;
  expect_input_alias_unchanged(frame, &view);
}

static void stage(et_e3_frame_internal *frame, int64_t first, int64_t second,
                  int64_t target0, int64_t target1,
                  unsigned char mask0, unsigned char mask1) {
  int64_t inputs[2] = {first, second};
  int64_t targets[2] = {target0, target1};
  unsigned char masks[2] = {mask0, mask1};
  et_kernel_tensor_view_v1 input_view = i64_pair(inputs);
  et_kernel_tensor_view_v1 target_view = i64_pair(targets);
  et_kernel_tensor_view_v1 mask_view = bool_pair(masks);
  OK(et_e3_private_stage_inputs_v1(frame, &input_view));
  OK(et_e3_private_stage_targets_v1(frame, &target_view));
  OK(et_e3_private_stage_mask_v1(frame, &mask_view));
}

static void forward_batch(et_e3_frame_internal *frame) {
  for (int64_t role = 0; role < 21; ++role) {
    const int64_t rc = et_e3_private_forward_role_v1(frame, role);
    if (rc) fprintf(stderr, "forward role %lld failed\n", (long long)role);
    OK(rc);
    et_f32_scoped_guard_internal guard = {0};
    OK(et_f32_tensor_scoped_begin_internal(frame->forward[role], &guard,
                                            &tensor_error));
    const float *values = guard.view.data;
    for (size_t i = 0; i < guard.view.byte_length / sizeof(float); ++i)
      CHECK(isfinite(values[i]));
    OK(et_f32_tensor_scoped_end_internal(&guard));
  }
  OK(et_e3_private_ce_v1(frame));
  OK(et_e3_private_reduce_v1(frame));
  OK(et_e3_private_correct_v1(frame));
  OK(et_e3_private_accumulate_v1(frame));
}

static void abort_call(et_e3_frame_internal *frame) {
  OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);
  et_e3_private_finish_v1(frame);
  CHECK(frame->phase == E3_IDLE && frame->model->active == NULL);
}

static void expect_metric_bits_error(void *frame, int64_t selector,
                                     int64_t domain, int64_t category,
                                     int64_t code) {
  const int64_t actual =
      et_e3_private_selected_metric_bits_ref_v1(frame, selector);
  if (actual != -1)
    fprintf(stderr, "metric bits selector %lld unexpectedly returned %lld\n",
            (long long)selector, (long long)actual);
  CHECK(actual == -1);
  CHECK(e3_last_domain == domain && e3_last_category == category &&
        e3_last_code == code);
}

static void test_create_rejections(owner *model, et_f32_tensor *destination[4]) {
  CHECK(et_e3_private_frame_create_v1((void *)(uintptr_t)1,
      destination[0], destination[1], destination[2], destination[3]) == NULL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INVALID_ARGUMENT &&
        e3_last_code == E3_CODE_IDENTITY);
  CHECK(et_e3_private_frame_create_v1(model, destination[0], destination[0],
      destination[2], destination[3]) == NULL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INVALID_ARGUMENT &&
        e3_last_code == E3_CODE_DESTINATION);

  const uint64_t shape[] = {1};
  et_f32_tensor *wrong = NULL;
  OK(et_f32_tensor_create_v1(1, shape, &wrong, &tensor_error));
  CHECK(et_e3_private_frame_create_v1(model, wrong, destination[1],
      destination[2], destination[3]) == NULL);
  CHECK(e3_last_code == E3_CODE_DESTINATION);
  destroy_tensor(&wrong);

  et_f32_tensor_borrow *borrow = NULL;
  OK(et_f32_tensor_borrow_begin_v1(destination[0], &borrow, &tensor_error));
  CHECK(et_e3_private_frame_create_v1(model, destination[0], destination[1],
      destination[2], destination[3]) == NULL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INVALID_STATE &&
        e3_last_code == E3_CODE_DESTINATION);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &tensor_error));

  et_e3_test_frame_fail_alloc_after_v1(0);
  CHECK(et_e3_private_frame_create_v1(model, destination[0], destination[1],
      destination[2], destination[3]) == NULL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_ALLOCATION);
  et_e3_test_frame_fail_alloc_after_v1(SIZE_MAX);
}

static void test_i2_allocation_rollback(owner *model,
                                        et_f32_tensor *destination[4]) {
  et_f32_test_live_counts_v1 baseline = {.struct_size = sizeof(baseline)};
  et_f32_test_live_counts_snapshot_v1(&baseline);
  size_t failpoint = 0;
  for (;; ++failpoint) {
    et_f32_tensor_test_fail_alloc_after_v1(failpoint);
    et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
        model, destination[0], destination[1], destination[2], destination[3]);
    et_f32_tensor_test_reset_allocator_v1();
    if (frame) {
      CHECK(failpoint > 37);
      OK(et_e3_private_frame_destroy_preflight_v1(frame));
      et_e3_private_frame_destroy_v1(frame);
      break;
    }
    CHECK(e3_last_domain == 3);
    et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
    et_f32_test_live_counts_snapshot_v1(&after);
    CHECK(after.tensors == baseline.tensors);
    CHECK(after.parameters == baseline.parameters);
    CHECK(after.borrows == baseline.borrows);
    CHECK(after.copy_plans == baseline.copy_plans);
    CHECK(after.gradient_plans == baseline.gradient_plans);
    CHECK(after.reset_plans == baseline.reset_plans);
    CHECK(failpoint < 512);
  }
}

static void test_transaction(owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  expect_metric_bits_error((void *)(uintptr_t)17, 0, 0,
                           E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
  expect_metric_bits_error((void *)(uintptr_t)17, 2, 0,
                           E3_INVALID_ARGUMENT, E3_CODE_IDENTITY);
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  expect_metric_bits_error(frame, 2, 0, E3_INVALID_ARGUMENT, E3_CODE_PHASE);
  CHECK(frame->lifecycle == 1 && frame->phase == E3_IDLE);
  CHECK(e3_all_tensors(frame, (et_f32_tensor *[37]){0}) == 37);
  CHECK(et_e3_private_frame_check_v1(frame) == 0);
  frame->positions[1] = 2;
  CHECK(et_e3_private_frame_check_v1(frame) == E3_INTERNAL);
  CHECK(e3_last_code == E3_CODE_INVARIANT);
  frame->positions[1] = 1;
  write_scalar(frame->epsilon, UINT32_C(0x3f800000));
  CHECK(et_e3_private_frame_check_v1(frame) == E3_INTERNAL);
  write_scalar(frame->epsilon, UINT32_C(0x3727c5ac));
  frame->model->active = destination[0];
  CHECK(et_e3_private_frame_check_v1(frame) == E3_INVALID_STATE);
  CHECK(e3_last_code == E3_CODE_REENTRANCY);
  frame->model->active = NULL;
  CHECK(et_e3_private_frame_check_v1(frame) == 0);
  CHECK(et_e3_private_frame_check_v1((void *)(uintptr_t)17) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_private_counter_ref_v1(frame, 0) == -1);
  CHECK(e3_last_category == E3_INVALID_STATE);

  OK(et_e3_private_acquire_v1(frame));
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  CHECK(frame->model->active == frame && frame->pins.held_mask == UINT16_C(0x3fff));
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_private_forward_role_v1(frame, -1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_private_forward_role_v1(frame, 21) == E3_INVALID_ARGUMENT);
  int64_t premature_targets[2] = {3, 4};
  et_kernel_tensor_view_v1 premature = i64_pair(premature_targets);
  CHECK(et_e3_private_stage_targets_v1(frame, &premature) == E3_INVALID_STATE);
  test_stage_aliases(frame);

  int64_t bad_ids[2] = {0, 256};
  et_kernel_tensor_view_v1 bad = i64_pair(bad_ids);
  CHECK(et_e3_private_stage_inputs_v1(frame, &bad) == E3_SHAPE_MISMATCH);
  CHECK(frame->phase == E3_ACQUIRED && frame->stage_plane == 0);
  bad_ids[1] = 1;
  bad.dtype = "f32";
  CHECK(et_e3_private_stage_inputs_v1(frame, &bad) == E3_UNSUPPORTED);
  CHECK(frame->phase == E3_ACQUIRED && frame->stage_plane == 0);
  bad.dtype = "i64";
  bad.data = frame->ids;
  CHECK(et_e3_private_stage_inputs_v1(frame, &bad) == E3_INVALID_ARGUMENT);
  CHECK(e3_last_code == E3_CODE_DESTINATION);
  CHECK(frame->phase == E3_ACQUIRED && frame->stage_plane == 0);

  stage(frame, 3, 197, 197, 0, 1, 1);
  CHECK(et_e3_private_forward_role_v1(frame, 1) == E3_INVALID_STATE);
  forward_batch(frame);
  stage(frame, 0, 255, 255, 7, 1, 1);
  forward_batch(frame);
  stage(frame, 7, 7, 7, 0, 1, 0);
  forward_batch(frame);
  OK(et_e3_private_finalize_v1(frame));
  CHECK(frame->staged_counts[0] == 5 && frame->staged_counts[1] == 3);
  OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);
  OK(et_e3_private_publish_v1(frame));
  et_e3_private_finish_v1(frame);

  uint32_t first[4];
  for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &first[i]);
  float loss, perplexity;
  memcpy(&loss, &first[0], sizeof(loss));
  memcpy(&perplexity, &first[2], sizeof(perplexity));
  CHECK(fabsf(loss - 5.5432668f) <= 0.00005f);
  CHECK(first[1] == UINT32_C(0x40a00000));
  CHECK(fabsf(perplexity - 255.51134f) <= 0.005f);
  CHECK(first[3] == UINT32_C(0x00000000));
  CHECK(first[1] == UINT32_C(0x40a00000));
  et_f32_test_live_counts_v1 accessor_before = {
      .struct_size = sizeof(accessor_before),
  };
  et_f32_test_live_counts_v1 accessor_after = {
      .struct_size = sizeof(accessor_after),
  };
  et_f32_test_live_counts_snapshot_v1(&accessor_before);
  et_e3_test_frame_fail_alloc_after_v1(0);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame, 0) == first[0]);
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame, 1) == first[1]);
  et_f32_tensor_test_reset_allocator_v1();
  et_e3_test_frame_fail_alloc_after_v1(SIZE_MAX);
  et_f32_test_live_counts_snapshot_v1(&accessor_after);
  CHECK(memcmp(&accessor_before, &accessor_after,
               sizeof(accessor_before)) == 0);
  expect_metric_bits_error(frame, -1, 0, E3_INVALID_ARGUMENT, E3_CODE_PHASE);
  expect_metric_bits_error(frame, 2, 0, E3_INVALID_ARGUMENT, E3_CODE_PHASE);

  frame->published_valid = 0;
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  frame->published_valid = 1;
  frame->committed = 0;
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  frame->committed = 2;

  frame->model->active = frame;
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE,
                           E3_CODE_REENTRANCY);
  frame->model->active = NULL;
  frame->cleanup_ready = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->cleanup_ready = 0;
  frame->next_role = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->next_role = 0;
  frame->stage_plane = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->stage_plane = 0;
  frame->sum_bank = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->sum_bank = 0;
  frame->active = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->active = 0;
  frame->staged_counts[0] = 1;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->staged_counts[0] = 0;
  frame->pins.self = &frame->pins;
  expect_metric_bits_error(frame, 0, 0, E3_INTERNAL, E3_CODE_INVARIANT);
  frame->pins.self = NULL;

  et_f32_tensor_borrow *metric_borrow = NULL;
  OK(et_f32_tensor_borrow_begin_v1(destination[0], &metric_borrow,
                                   &tensor_error));
  expect_metric_bits_error(frame, 0, 3, ET_F32_TENSOR_ERROR_INVALID_STATE,
                           ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  OK(et_f32_tensor_borrow_end_v1(&metric_borrow, &tensor_error));
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame, 0) == first[0]);

  et_f32_tensor *metric_source = scalar();
  et_f32_tensor_copy_assignment_v1 metric_assignment = {
      .struct_size = sizeof(metric_assignment),
      .destination = destination[0],
      .source = metric_source,
  };
  et_f32_tensor_copy_plan *metric_plan = NULL;
  OK(et_f32_tensor_copy_plan_prepare_v1(1, &metric_assignment, &metric_plan,
                                        &tensor_error));
  expect_metric_bits_error(frame, 0, 3, ET_F32_TENSOR_ERROR_INVALID_STATE,
                           ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  OK(et_f32_tensor_copy_plan_release_v1(&metric_plan, &tensor_error));
  destroy_tensor(&metric_source);
  CHECK(et_e3_private_selected_metric_bits_ref_v1(frame, 0) == first[0]);
  CHECK(et_e3_private_counter_ref_v1(frame, 0) == 5);
  CHECK(et_e3_private_counter_ref_v1(frame, 1) == 3);
  CHECK(et_e3_private_counter_ref_v1(frame, 2) == -1);
  CHECK(e3_last_category == E3_INVALID_ARGUMENT);

  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 1, 2, 2, 3, 1, 1);
  CHECK(et_e3_private_forward_role_v1(frame, 0) == 0);
  abort_call(frame);
  for (size_t i = 0; i < 4; ++i) {
    uint32_t bits;
    read_scalar(destination[i], &bits);
    CHECK(bits == first[i]);
  }
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  CHECK(et_e3_private_counter_ref_v1(frame, 0) == 5);
  CHECK(et_e3_private_counter_ref_v1(frame, 1) == 3);

  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  OK(et_e3_private_finalize_v1(frame));
  OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);
  et_f32_tensor_borrow *borrow = NULL;
  OK(et_f32_tensor_borrow_begin_v1(destination[2], &borrow, &tensor_error));
  CHECK(et_e3_private_publish_v1(frame) == E3_INVALID_STATE);
  CHECK(e3_last_domain == 0 && e3_last_code == E3_CODE_DESTINATION);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &tensor_error));
  et_e3_private_finish_v1(frame);
  for (size_t i = 0; i < 4; ++i) {
    uint32_t bits;
    read_scalar(destination[i], &bits);
    CHECK(bits == first[i]);
  }
  CHECK(et_e3_private_counter_ref_v1(frame, 0) == 5);

  OK(et_e3_private_acquire_v1(frame));
  frame->phase = E3_CORRECTED;
  frame->counts[0][0] = 16384;
  frame->counts[0][1] = 8192;
  frame->active = 1;
  write_scalar(frame->sum[0][0], 0);
  write_scalar(frame->sum[0][1], UINT32_C(0x46800000));
  write_scalar(frame->sum[0][2], 0);
  write_scalar(frame->batch[0], 0);
  write_scalar(frame->batch[1], UINT32_C(0x3f800000));
  write_scalar(frame->batch[3], 0);
  CHECK(et_e3_private_accumulate_v1(frame) == E3_UNSUPPORTED);
  CHECK(e3_last_domain == 1 && e3_last_category == ET_KERNEL_ERROR_UNSUPPORTED);
  abort_call(frame);

  et_f32_tensor_borrow *destroy_borrow = NULL;
  OK(et_f32_tensor_borrow_begin_v1(destination[0], &destroy_borrow,
                                   &tensor_error));
  CHECK(et_e3_private_frame_destroy_preflight_v1(frame) == E3_INVALID_STATE);
  CHECK(e3_last_code == E3_CODE_DESTINATION);
  OK(et_f32_tensor_borrow_end_v1(&destroy_borrow, &tensor_error));
  et_f32_tensor *creation_order[37];
  CHECK(e3_all_tensors(frame, creation_order) == 37);
  destroyed_tensor_count = 0;
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(destroyed_tensor_count == 37);
  for (size_t i = 0; i < 37; ++i)
    CHECK(destroyed_tensors[i] == creation_order[36 - i]);
  CHECK(et_e3_private_frame_check_v1(frame) == E3_INVALID_STATE);
  expect_metric_bits_error(frame, 0, 0, E3_INVALID_STATE, E3_CODE_PHASE);
  CHECK(et_e3_private_frame_destroy_preflight_v1(frame) == E3_INVALID_STATE);
}

static void test_discovery_bind_rollback(owner *model,
                                         et_f32_tensor *destination[4]) {
  const uint32_t sentinels[4] = {
      UINT32_C(0x3f000001), UINT32_C(0x3f000002),
      UINT32_C(0x3f000003), UINT32_C(0x3f000004)};
  for (size_t i = 0; i < 4; ++i) write_scalar(destination[i], sentinels[i]);
  for (int64_t selected = 0; selected < 6; ++selected) {
    owner model_before;
    memcpy(&model_before, model, sizeof(model_before));
    et_e3_frame_internal *registry_before = e3_registry;
    et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
    et_f32_test_live_counts_snapshot_v1(&before);
    const size_t creates_before = runtime_creates;
    const size_t destroys_before = runtime_destroys;
    const size_t live_before = live_runtime_count;
    et_kernel_runtime *baseline_runtimes[64];
    memcpy(baseline_runtimes, live_runtimes,
           live_before * sizeof(baseline_runtimes[0]));

    e3_test_discovery_fail_index = selected;
    CHECK(et_e3_private_frame_create_v1(model, destination[0], destination[1],
        destination[2], destination[3]) == NULL);
    CHECK(e3_test_discovery_fail_index == -1);
    CHECK(runtime_creates == creates_before + (size_t)selected);
    CHECK(runtime_destroys == destroys_before + (size_t)selected);
    CHECK(live_runtime_count == live_before);
    for (size_t i = 0; i < live_before; ++i) {
      size_t found = 0;
      while (found < live_runtime_count &&
             live_runtimes[found] != baseline_runtimes[i]) ++found;
      CHECK(found < live_runtime_count);
    }
    CHECK(e3_last_domain == 1 &&
          e3_last_category == ET_KERNEL_ERROR_UNSUPPORTED &&
          e3_last_code == ET_KERNEL_CODE_SYMBOL_MISSING);
    CHECK(et_e3_private_last_error_domain_v1() == 1 &&
          et_e3_private_last_error_category_v1() ==
              ET_KERNEL_ERROR_UNSUPPORTED &&
          et_e3_private_last_error_code_v1() ==
              ET_KERNEL_CODE_SYMBOL_MISSING);
    CHECK(e3_dependency_status(e3_last_domain, e3_last_category) ==
          E3_UNSUPPORTED);
    CHECK(e3_registry == registry_before && model->active == NULL);
    CHECK(memcmp(model, &model_before, sizeof(model_before)) == 0);
    et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
    et_f32_test_live_counts_snapshot_v1(&after);
    CHECK(after.tensors == before.tensors);
    CHECK(after.parameters == before.parameters);
    CHECK(after.owned_clones == before.owned_clones);
    CHECK(after.borrows == before.borrows);
    CHECK(after.copy_plans == before.copy_plans);
    CHECK(after.gradient_plans == before.gradient_plans);
    CHECK(after.reset_plans == before.reset_plans);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t bits;
      read_scalar(destination[i], &bits);
      CHECK(bits == sentinels[i]);
    }

    /* The one-shot fault is gone: bind and use a newly constructed frame. */
    et_e3_frame_internal *retry = et_e3_private_frame_create_v1(
        model, destination[0], destination[1], destination[2], destination[3]);
    CHECK(retry != NULL && e3_registry == retry &&
          retry->next == registry_before);
    CHECK(runtime_creates == creates_before + (size_t)selected + 6 &&
          runtime_destroys == destroys_before + (size_t)selected &&
          live_runtime_count == live_before + 6);
    OK(et_e3_private_frame_check_v1(retry));
    OK(et_e3_private_acquire_v1(retry));
    stage(retry, 3, 197, 197, 0, 1, 1);
    forward_batch(retry);
    abort_call(retry);
    OK(et_e3_private_frame_destroy_preflight_v1(retry));
    et_e3_private_frame_destroy_v1(retry);
    CHECK(e3_registry == retry && retry->lifecycle == 2 &&
          retry->next == registry_before && model->active == NULL);
    CHECK(runtime_destroys == destroys_before + (size_t)selected + 6 &&
          live_runtime_count == live_before);
    et_f32_test_live_counts_snapshot_v1(&after);
    CHECK(after.tensors == before.tensors &&
          after.parameters == before.parameters &&
          after.owned_clones == before.owned_clones &&
          after.borrows == before.borrows &&
          after.copy_plans == before.copy_plans &&
          after.gradient_plans == before.gradient_plans &&
          after.reset_plans == before.reset_plans);
  }
}

static void test_forward_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  et_e3_frame_internal *other = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(other != NULL);
  OK(et_e3_test_fault_arm_v1(other, 1, 20, 8192));
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);
  CHECK(et_e3_test_fault_ref_v1(other, 2) == 20);
  CHECK(et_e3_test_fault_ref_v1(other, 3) == 8192);
  OK(et_e3_test_fault_reset_v1(other));
  OK(et_e3_private_frame_destroy_preflight_v1(other));
  et_e3_private_frame_destroy_v1(other);
  CHECK(et_e3_test_fault_arm_v1((void *)(uintptr_t)17, 1, 0, 1) ==
        E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 0, 0, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 21, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 8193) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);
  CHECK(et_e3_test_fault_ref_v1(frame, 5) == -1);

  OK(et_e3_test_fault_arm_v1(frame, 1, 0, 1));
  frame->positions[1] = 2;
  CHECK(et_e3_private_acquire_v1(frame) == E3_INTERNAL);
  frame->positions[1] = 1;
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  OK(et_e3_test_fault_reset_v1(frame));

  OK(et_e3_test_fault_arm_v1(frame, 1, 1, 1));
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 1) == E3_INVALID_STATE);
  OK(et_e3_private_acquire_v1(frame));
  CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == -1);
  stage(frame, 3, 197, 197, 0, 1, 1);
  CHECK(et_e3_private_forward_role_v1(frame, 1) == E3_INVALID_STATE);
  CHECK(frame->fault_reached == 0);
  const size_t before_role_zero = observed_dispatches;
  OK(et_e3_private_forward_role_v1(frame, 0));
  CHECK(frame->next_role == 1 && observed_dispatches == before_role_zero + 1);
  CHECK(et_e3_private_forward_role_v1(frame, 1) == E3_INTERNAL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_INVARIANT);
  CHECK(frame->next_role == 1 && frame->fault_state == 2 &&
        frame->fault_reached == 1 &&
        observed_dispatches == before_role_zero + 1);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
  CHECK(et_e3_test_fault_ref_v1(frame, 1) == 1);
  CHECK(et_e3_test_fault_ref_v1(frame, 2) == 1);
  CHECK(et_e3_test_fault_ref_v1(frame, 3) == 1);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 1) == E3_INVALID_STATE);
  CHECK(et_e3_private_frame_destroy_preflight_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  /* The second visit to role 0 occurs only after a complete first batch. */
  OK(et_e3_test_fault_arm_v1(frame, 1, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  CHECK(frame->fault_reached == 1 && frame->fault_state == 1);
  stage(frame, 0, 255, 255, 7, 1, 1);
  const size_t before_second_role_zero = observed_dispatches;
  CHECK(et_e3_private_forward_role_v1(frame, 0) == E3_INTERNAL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_INVARIANT);
  CHECK(frame->next_role == 0 && frame->fault_state == 2 &&
        frame->fault_reached == 2 &&
        observed_dispatches == before_second_role_zero);
  abort_call(frame);
  OK(et_e3_test_fault_reset_v1(frame));

  /* An unreached occurrence is retained as failed test state until reset. */
  OK(et_e3_test_fault_arm_v1(frame, 1, 20, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));

  /* Feature-enabled, unarmed dispatch remains the ordinary role sequence. */
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  CHECK(frame->next_role == 0 && frame->phase == E3_ACQUIRED);
  abort_call(frame);
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 1) == E3_INVALID_STATE);
}

static void test_partial_pin_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  et_e3_frame_internal *other = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(other != NULL);
  OK(et_e3_test_fault_arm_v1(other, 4, 2, 1));
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);
  CHECK(et_e3_test_fault_ref_v1(other, 2) == 2);
  OK(et_e3_test_fault_reset_v1(other));
  OK(et_e3_private_frame_destroy_preflight_v1(other));
  et_e3_private_frame_destroy_v1(other);
  CHECK(et_e3_test_fault_arm_v1(frame, 4, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 4, 14, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 4, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 4, 0, 2) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1((void *)(uintptr_t)17, 4, 0, 1) ==
        E3_INVALID_ARGUMENT);
  for (int64_t index = 0; index < 14; ++index) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 4, index, 1));
    CHECK(et_e3_test_fault_arm_v1(frame, 4, index, 1) == E3_INVALID_STATE);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INTERNAL);
    CHECK(e3_last_domain == 3 && e3_last_category == 6 && e3_last_code == 6);
    CHECK(frame->phase == E3_IDLE && !frame->model->active &&
          e3_pins_idle(&frame->pins));
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 4);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == index);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
    for (size_t i = 0; i < 14; ++i) {
      const et_f32_tensor *value = NULL;
      size_t pins = SIZE_MAX;
      et_f32_tensor_borrow *borrow = NULL;
      et_f32_tensor_error error;
      OK(et_f32_parameter_value_tensor_v1(model->p[i], &value, &error));
      OK(et_f32_tensor_test_plan_pins_v1(value, &pins));
      CHECK(pins == 0);
      OK(et_f32_parameter_value_borrow_begin_v1(model->p[i], &borrow, &error));
      OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
    }
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    OK(et_e3_test_fault_reset_v1(frame));
  }
  OK(et_e3_test_fault_arm_v1(frame, 4, 13, 1));
  frame->positions[1] = 2;
  CHECK(et_e3_private_acquire_v1(frame) == E3_INTERNAL);
  frame->positions[1] = 1;
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 0);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_acquire_v1(frame));
  CHECK(et_e3_test_fault_arm_v1(frame, 4, 0, 1) == E3_INVALID_STATE);
  abort_call(frame);
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 4, 0, 1) == E3_INVALID_STATE);
}

static void test_target_staging_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 10, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 10, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 10, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 10, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 10, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 10, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      int64_t inputs[2] = {visit == 1 ? 3 : 0, visit == 1 ? 197 : 255};
      int64_t targets[2] = {visit == 1 ? 197 : 255, visit == 1 ? 0 : 7};
      et_kernel_tensor_view_v1 input_view = i64_pair(inputs);
      et_kernel_tensor_view_v1 target_view = i64_pair(targets);
      const int64_t old_targets[2] = {frame->targets[0], frame->targets[1]};
      OK(et_e3_private_stage_inputs_v1(frame, &input_view));
      CHECK(frame->ids[0] == inputs[0] && frame->ids[1] == inputs[1] &&
            frame->phase == E3_STAGING && frame->stage_plane == 1);
      const size_t before_target = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_stage_targets_v1(frame, &target_view));
        CHECK(frame->targets[0] == targets[0] &&
              frame->targets[1] == targets[1] &&
              frame->stage_plane == 2 && observed_dispatches == before_target);
        unsigned char masks[2] = {1, 1};
        et_kernel_tensor_view_v1 mask_view = bool_pair(masks);
        OK(et_e3_private_stage_mask_v1(frame, &mask_view));
        forward_batch(frame);
        CHECK(frame->fault_state == 1 && frame->fault_reached == 1);
      } else {
        CHECK(et_e3_private_stage_targets_v1(frame, &target_view) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(frame->targets[0] == old_targets[0] &&
              frame->targets[1] == old_targets[1] &&
              frame->phase == E3_STAGING && frame->stage_plane == 1 &&
              observed_dispatches == before_target &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 10);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  /* The armed cut precedes even target-view validation. */
  OK(et_e3_test_fault_arm_v1(frame, 10, 0, 1));
  OK(et_e3_private_acquire_v1(frame));
  int64_t input_only[2] = {3, 197};
  et_kernel_tensor_view_v1 input_only_view = i64_pair(input_only);
  OK(et_e3_private_stage_inputs_v1(frame, &input_only_view));
  CHECK(et_e3_private_stage_targets_v1(frame, NULL) == E3_INTERNAL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_INVARIANT && frame->stage_plane == 1);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
  OK(et_e3_test_fault_reset_v1(frame));

  OK(et_e3_test_fault_arm_v1(frame, 10, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 10, 0, 1) == E3_INVALID_STATE);
}

static void test_mask_staging_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 11, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 11, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 11, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 11, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 11, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 11, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    const size_t initial_dispatches = observed_dispatches;
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      int64_t inputs[2] = {visit == 1 ? 3 : 0, visit == 1 ? 197 : 255};
      int64_t targets[2] = {visit == 1 ? 197 : 255, visit == 1 ? 0 : 7};
      et_kernel_tensor_view_v1 input_view = i64_pair(inputs);
      et_kernel_tensor_view_v1 target_view = i64_pair(targets);
      unsigned char old_mask[2] = {frame->mask[0], frame->mask[1]};
      OK(et_e3_private_stage_inputs_v1(frame, &input_view));
      const size_t before_target = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_stage_targets_v1(frame, &target_view));
        unsigned char masks[2] = {1, 1};
        et_kernel_tensor_view_v1 mask_view = bool_pair(masks);
        OK(et_e3_private_stage_mask_v1(frame, &mask_view));
        forward_batch(frame);
        CHECK(observed_dispatches > initial_dispatches &&
              frame->fault_state == 1 && frame->fault_reached == 1);
      } else {
        CHECK(et_e3_private_stage_targets_v1(frame, &target_view) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(frame->ids[0] == inputs[0] && frame->ids[1] == inputs[1] &&
              frame->targets[0] == targets[0] &&
              frame->targets[1] == targets[1]);
        CHECK(frame->mask[0] == old_mask[0] &&
              frame->mask[1] == old_mask[1] &&
              frame->phase == E3_STAGING && frame->stage_plane == 2 &&
              observed_dispatches == before_target &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 11);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  OK(et_e3_test_fault_arm_v1(frame, 11, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 11, 0, 1) == E3_INVALID_STATE);
}

static void test_ce_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 5, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 5, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 5, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 5, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 5, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 5, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      stage(frame, visit == 1 ? 3 : 0, visit == 1 ? 197 : 255,
            visit == 1 ? 197 : 255, visit == 1 ? 0 : 7, 1, 1);
      for (int64_t role = 0; role < 21; ++role)
        OK(et_e3_private_forward_role_v1(frame, role));
      CHECK(frame->phase == E3_CE_READY && frame->next_role == 21);
      const size_t before_ce = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_ce_v1(frame));
        CHECK(observed_dispatches == before_ce + 1);
        OK(et_e3_private_reduce_v1(frame));
        OK(et_e3_private_correct_v1(frame));
        OK(et_e3_private_accumulate_v1(frame));
        CHECK(frame->fault_state == 1 && frame->fault_reached == 1);
      } else {
        CHECK(et_e3_private_ce_v1(frame) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(observed_dispatches == before_ce && frame->next_role == 21 &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 5);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  OK(et_e3_test_fault_arm_v1(frame, 5, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 5, 0, 1) == E3_INVALID_STATE);
}

static void test_reduce_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 6, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 6, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 6, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 6, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 6, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 6, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      stage(frame, visit == 1 ? 3 : 0, visit == 1 ? 197 : 255,
            visit == 1 ? 197 : 255, visit == 1 ? 0 : 7, 1, 1);
      for (int64_t role = 0; role < 21; ++role)
        OK(et_e3_private_forward_role_v1(frame, role));
      CHECK(frame->phase == E3_CE_READY && frame->next_role == 21);
      const size_t before_ce = observed_dispatches;
      OK(et_e3_private_ce_v1(frame));
      CHECK(observed_dispatches == before_ce + 1);
      CHECK(frame->phase == E3_CE_READY && frame->next_role == 22);
      const size_t before_reduce = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_reduce_v1(frame));
        CHECK(observed_dispatches == before_reduce + 1);
        OK(et_e3_private_correct_v1(frame));
        OK(et_e3_private_accumulate_v1(frame));
        CHECK(frame->fault_state == 1 && frame->fault_reached == 1);
      } else {
        CHECK(et_e3_private_reduce_v1(frame) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(observed_dispatches == before_reduce &&
              frame->phase == E3_CE_READY && frame->next_role == 22 &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 6);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  OK(et_e3_test_fault_arm_v1(frame, 6, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 6, 0, 1) == E3_INVALID_STATE);
}

static void test_correct_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 7, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 7, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 7, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 7, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 7, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 7, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      stage(frame, visit == 1 ? 3 : 0, visit == 1 ? 197 : 255,
            visit == 1 ? 197 : 255, visit == 1 ? 0 : 7, 1, 1);
      for (int64_t role = 0; role < 21; ++role)
        OK(et_e3_private_forward_role_v1(frame, role));
      CHECK(frame->phase == E3_CE_READY && frame->next_role == 21);
      const size_t before_ce = observed_dispatches;
      OK(et_e3_private_ce_v1(frame));
      CHECK(observed_dispatches == before_ce + 1);
      CHECK(frame->phase == E3_CE_READY && frame->next_role == 22);
      OK(et_e3_private_reduce_v1(frame));
      CHECK(frame->phase == E3_REDUCED &&
            observed_dispatches == before_ce + 2);
      const size_t before_correct = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_correct_v1(frame));
        CHECK(observed_dispatches == before_correct + 1);
        OK(et_e3_private_accumulate_v1(frame));
        CHECK(frame->fault_state == 1 && frame->fault_reached == 1);
      } else {
        CHECK(et_e3_private_correct_v1(frame) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(observed_dispatches == before_correct &&
              frame->phase == E3_REDUCED && frame->next_role == 22 &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 7);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  OK(et_e3_test_fault_arm_v1(frame, 7, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 7, 0, 1) == E3_INVALID_STATE);
}

static void test_accumulate_preinvoke_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_arm_v1(frame, 8, -1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 8, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 8, 0, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 8, 0, 3) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 0);

  for (int64_t occurrence = 1; occurrence <= 2; ++occurrence) {
    uint32_t words[4];
    for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &words[i]);
    const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
    const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
    OK(et_e3_test_fault_arm_v1(frame, 8, 0, occurrence));
    CHECK(et_e3_test_fault_arm_v1(frame, 8, 0, occurrence) == E3_INVALID_STATE);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_fault_reset_v1(frame) == E3_INVALID_STATE);
    for (int64_t visit = 1; visit <= occurrence; ++visit) {
      stage(frame, visit == 1 ? 3 : 0, visit == 1 ? 197 : 255,
            visit == 1 ? 197 : 255, visit == 1 ? 0 : 7, 1, 1);
      for (int64_t role = 0; role < 21; ++role)
        OK(et_e3_private_forward_role_v1(frame, role));
      OK(et_e3_private_ce_v1(frame));
      OK(et_e3_private_reduce_v1(frame));
      const size_t before_correct = observed_dispatches;
      OK(et_e3_private_correct_v1(frame));
      CHECK(observed_dispatches == before_correct + 1 &&
            frame->phase == E3_CORRECTED);
      const size_t before_accumulate = observed_dispatches;
      if (visit < occurrence) {
        OK(et_e3_private_accumulate_v1(frame));
        CHECK(observed_dispatches == before_accumulate + 1 &&
              frame->fault_state == 1 && frame->fault_reached == 1 &&
              frame->counts[frame->sum_bank][1] == 1);
      } else {
        const uint32_t bank = frame->sum_bank;
        int64_t counts[2][2];
        uint32_t sums[2][3];
        memcpy(counts, frame->counts, sizeof(counts));
        for (size_t side = 0; side < 2; ++side)
          for (size_t i = 0; i < 3; ++i)
            read_scalar(frame->sum[side][i], &sums[side][i]);
        CHECK(et_e3_private_accumulate_v1(frame) == E3_INTERNAL);
        CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
              e3_last_code == E3_CODE_INVARIANT);
        CHECK(observed_dispatches == before_accumulate &&
              frame->phase == E3_CORRECTED && frame->sum_bank == bank &&
              frame->fault_state == 2 && frame->fault_reached == occurrence);
        CHECK(memcmp(counts, frame->counts, sizeof(counts)) == 0);
        for (size_t side = 0; side < 2; ++side)
          for (size_t i = 0; i < 3; ++i) {
            uint32_t after;
            read_scalar(frame->sum[side][i], &after);
            CHECK(after == sums[side][i]);
          }
      }
    }
    abort_call(frame);
    CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
    CHECK(et_e3_test_fault_ref_v1(frame, 1) == 8);
    CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
    CHECK(et_e3_test_fault_ref_v1(frame, 3) == occurrence);
    CHECK(et_e3_test_fault_ref_v1(frame, 4) == occurrence);
    CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t after;
      read_scalar(destination[i], &after);
      CHECK(after == words[i]);
    }
    CHECK(et_e3_private_counter_ref_v1(frame, 0) == tokens &&
          et_e3_private_counter_ref_v1(frame, 1) == batches);
    OK(et_e3_test_fault_reset_v1(frame));
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    abort_call(frame);
  }

  OK(et_e3_test_fault_arm_v1(frame, 8, 0, 2));
  OK(et_e3_private_acquire_v1(frame));
  stage(frame, 3, 197, 197, 0, 1, 1);
  forward_batch(frame);
  abort_call(frame);
  CHECK(et_e3_test_fault_ref_v1(frame, 0) == 3);
  CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  OK(et_e3_test_fault_reset_v1(frame));
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_test_fault_arm_v1(frame, 8, 0, 1) == E3_INVALID_STATE);
}

static void test_publication_precommit_selector(
    owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_fault_seed_publication_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_test_fault_arm_v1(frame, 2, 1, 1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 2, 0, 2) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_fault_arm_v1(frame, 3, 0, 1) == E3_INVALID_ARGUMENT);
  uint32_t ordinary[4];
  const uint32_t sentinels[4] = {UINT32_C(0x3f800000), UINT32_C(0x40000000),
                                  UINT32_C(0x40400000), UINT32_C(0x40800000)};
  for (int invocation = 0; invocation < 3; ++invocation) {
    if (invocation == 1) {
      OK(et_e3_test_fault_seed_publication_v1(frame));
      CHECK(et_e3_private_counter_ref_v1(frame, 0) == 7);
      CHECK(et_e3_private_counter_ref_v1(frame, 1) == 4);
      for (size_t i = 0; i < 4; ++i) {
        uint32_t bits;
        read_scalar(destination[i], &bits);
        CHECK(bits == sentinels[i] && bits != ordinary[i]);
      }
      OK(et_e3_test_fault_arm_v1(frame, 2, 0, 1));
      CHECK(et_e3_test_fault_arm_v1(frame, 1, 0, 1) == E3_INVALID_STATE);
    }
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    stage(frame, 0, 255, 255, 7, 1, 1);
    forward_batch(frame);
    stage(frame, 7, 7, 7, 0, 1, 0);
    forward_batch(frame);
    OK(et_e3_private_finalize_v1(frame));
    OK(et_e3_private_cleanup_preflight_v1(frame));
    et_e3_private_drain_v1(frame);
    if (invocation == 1) {
      et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
      et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
      et_f32_test_live_counts_snapshot_v1(&before);
      CHECK(et_e3_private_publish_v1(frame) == E3_INTERNAL);
      CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
            e3_last_code == E3_CODE_INVARIANT);
      et_f32_test_live_counts_snapshot_v1(&after);
      CHECK(after.borrows == before.borrows);
      CHECK(frame->fault_state == 2 && frame->fault_reached == 1 &&
            frame->committed == 1);
    } else {
      OK(et_e3_private_publish_v1(frame));
    }
    et_e3_private_finish_v1(frame);
    if (invocation == 0) {
      for (size_t i = 0; i < 4; ++i) read_scalar(destination[i], &ordinary[i]);
      CHECK(frame->published_counts[0] == 5 && frame->published_counts[1] == 3);
    } else if (invocation == 1) {
      CHECK(et_e3_test_fault_ref_v1(frame, 0) == 2);
      CHECK(et_e3_test_fault_ref_v1(frame, 1) == 2);
      CHECK(et_e3_test_fault_ref_v1(frame, 2) == 0);
      CHECK(et_e3_test_fault_ref_v1(frame, 3) == 1);
      CHECK(et_e3_test_fault_ref_v1(frame, 4) == 1);
      CHECK(et_e3_private_counter_ref_v1(frame, 0) == 7);
      CHECK(et_e3_private_counter_ref_v1(frame, 1) == 4);
      for (size_t i = 0; i < 4; ++i) {
        uint32_t bits;
        read_scalar(destination[i], &bits);
        CHECK(bits == sentinels[i]);
      }
      OK(et_e3_test_fault_reset_v1(frame));
    } else {
      CHECK(et_e3_private_counter_ref_v1(frame, 0) == 5);
      CHECK(et_e3_private_counter_ref_v1(frame, 1) == 3);
      for (size_t i = 0; i < 4; ++i) {
        uint32_t bits;
        read_scalar(destination[i], &bits);
        CHECK(bits == ordinary[i] && bits != sentinels[i]);
      }
    }
  }
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
}

int main(void) {
  test_diagnostic_mapping();
  owner *model = make_model();
  et_f32_tensor *destination[4] = {scalar(), scalar(), scalar(), scalar()};
  test_create_rejections(model, destination);
  test_i2_allocation_rollback(model, destination);
  test_transaction(model, destination);
  test_discovery_bind_rollback(model, destination);
  test_forward_preinvoke_selector(model, destination);
  test_target_staging_preinvoke_selector(model, destination);
  test_mask_staging_preinvoke_selector(model, destination);
  test_ce_preinvoke_selector(model, destination);
  test_reduce_preinvoke_selector(model, destination);
  test_correct_preinvoke_selector(model, destination);
  test_accumulate_preinvoke_selector(model, destination);
  test_partial_pin_selector(model, destination);
  test_publication_precommit_selector(model, destination);
  for (size_t i = 0; i < 4; ++i) destroy_tensor(&destination[i]);
  printf("E3 native frame: %zu checks passed\n", checks);
  return 0;
}
