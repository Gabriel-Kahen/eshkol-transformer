#define ET_E3_TESTING 1
#include "../../src/eshkol_transformer/e3_frame.c"

#include <math.h>
#include <stdio.h>

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

  e3_test_fail_frame_alloc_after(0);
  CHECK(et_e3_private_frame_create_v1(model, destination[0], destination[1],
      destination[2], destination[3]) == NULL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_ALLOCATION);
  e3_test_fail_frame_alloc_after(SIZE_MAX);
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
  CHECK(frame->model->active == frame && frame->pins.held_mask == UINT16_C(0x3fff));
  CHECK(et_e3_private_acquire_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_private_forward_role_v1(frame, -1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_private_forward_role_v1(frame, 21) == E3_INVALID_ARGUMENT);
  int64_t premature_targets[2] = {3, 4};
  et_kernel_tensor_view_v1 premature = i64_pair(premature_targets);
  CHECK(et_e3_private_stage_targets_v1(frame, &premature) == E3_INVALID_STATE);

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
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  CHECK(et_e3_private_frame_check_v1(frame) == E3_INVALID_STATE);
  CHECK(et_e3_private_frame_destroy_preflight_v1(frame) == E3_INVALID_STATE);
}

int main(void) {
  owner *model = make_model();
  et_f32_tensor *destination[4] = {scalar(), scalar(), scalar(), scalar()};
  test_create_rejections(model, destination);
  test_i2_allocation_rollback(model, destination);
  test_transaction(model, destination);
  for (size_t i = 0; i < 4; ++i) destroy_tensor(&destination[i]);
  printf("E3 native frame: %zu checks passed\n", checks);
  return 0;
}
