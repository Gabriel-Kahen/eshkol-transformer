#define ET_E3_R1_PREFLIGHT_TESTING 1
#define main et_e3_base_native_main
#include "test_frame.c"
#undef main

static uint32_t first_word(et_f32_tensor *tensor) {
  et_f32_scoped_guard_internal guard = {0};
  uint32_t word;
  OK(et_f32_tensor_scoped_begin_internal(tensor, &guard, &tensor_error));
  CHECK(guard.view.byte_length >= sizeof(word));
  memcpy(&word, guard.view.data, sizeof(word));
  OK(et_f32_tensor_scoped_end_internal(&guard));
  return word;
}

static et_f32_test_live_counts_v1 live_counts(void) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  return counts;
}

static void check_cut(et_e3_frame_internal *frame, et_f32_tensor *output,
                      uint32_t before, et_f32_test_live_counts_v1 live_before,
                      int64_t status) {
  CHECK(status == E3_INTERNAL);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INTERNAL &&
        e3_last_code == E3_CODE_TEST_PREFLIGHT);
  CHECK(!frame->preinvoke_armed && first_word(output) == before);
  et_f32_test_live_counts_v1 live_after = live_counts();
  CHECK(memcmp(&live_before, &live_after, sizeof(live_before)) == 0);
  CHECK(frame->model->active == frame &&
        frame->pins.held_mask == UINT16_C(0x3fff));
  CHECK(et_g3t_model_pins_check_internal(&frame->pins, &tensor_error) == 0);
}

static void test_preinvoke_cuts(owner *model, et_f32_tensor *destination[4]) {
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  CHECK(et_e3_test_preinvoke_arm_v1((void *)(uintptr_t)1, 7, 0) ==
        E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_preinvoke_arm_v1(frame, 6, -1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_preinvoke_arm_v1(frame, 7, -1) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_preinvoke_arm_v1(frame, 7, 21) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_preinvoke_arm_v1(frame, 10, 0) == E3_INVALID_ARGUMENT);
  CHECK(et_e3_test_preinvoke_arm_v1(frame, 13, -1) == E3_INVALID_ARGUMENT);
  CHECK(!frame->preinvoke_armed);

  struct { int64_t stage, role; } cases[26];
  for (int64_t role = 0; role < 21; ++role) {
    cases[role].stage = 7;
    cases[role].role = role;
  }
  for (int64_t stage_id = 8; stage_id <= 12; ++stage_id) {
    cases[21 + stage_id - 8].stage = stage_id;
    cases[21 + stage_id - 8].role = -1;
  }
  for (size_t cut = 0; cut < sizeof(cases) / sizeof(cases[0]); ++cut) {
    const int64_t stage_id = cases[cut].stage, role_id = cases[cut].role;
    const uint32_t sentinels[4] = {
        UINT32_C(0x3f100000), UINT32_C(0x3f200000),
        UINT32_C(0x3f300000), UINT32_C(0x3f400000)};
    for (size_t i = 0; i < 4; ++i) write_scalar(destination[i], sentinels[i]);
    int64_t published_before[2];
    memcpy(published_before, frame->published_counts, sizeof(published_before));
    const uint32_t valid_before = frame->published_valid;
    et_kernel_runtime *runtimes_before[6];
    memcpy(runtimes_before, frame->runtime, sizeof(runtimes_before));
    OK(et_e3_test_preinvoke_arm_v1(frame, stage_id, role_id));
    CHECK(et_e3_test_preinvoke_arm_v1(frame, stage_id, role_id) ==
          E3_INVALID_STATE);
    CHECK(frame->preinvoke_armed);
    OK(et_e3_private_acquire_v1(frame));
    CHECK(et_e3_test_preinvoke_arm_v1(frame, stage_id, role_id) ==
          E3_INVALID_STATE);
    stage(frame, 3, 197, 197, 0, 1, 1);
    for (int64_t role = 0; role < 21; ++role) {
      if (stage_id == 7 && role == role_id) {
        et_f32_tensor *output = frame->forward[role];
        const uint32_t before = first_word(output);
        const et_f32_test_live_counts_v1 live_before = live_counts();
        check_cut(frame, output, before, live_before,
                  et_e3_private_forward_role_v1(frame, role));
        CHECK(frame->phase == E3_FORWARD && frame->next_role == role);
        goto injected;
      }
      OK(et_e3_private_forward_role_v1(frame, role));
      CHECK(frame->preinvoke_armed);
    }
#define CUT_STAGE(number, output, call) do { \
  et_f32_tensor *cut_output = (output); \
  uint32_t before = first_word(cut_output); \
  et_f32_test_live_counts_v1 live_before = live_counts(); \
  int64_t status = (call); \
  if (stage_id == (number)) { \
    check_cut(frame, cut_output, before, live_before, status); \
    goto injected; \
  } \
  OK(status); \
  CHECK(frame->preinvoke_armed); \
} while (0)
    CUT_STAGE(8, frame->ce, et_e3_private_ce_v1(frame));
    CUT_STAGE(9, frame->batch[0], et_e3_private_reduce_v1(frame));
    CUT_STAGE(10, frame->batch[3], et_e3_private_correct_v1(frame));
    CUT_STAGE(11, frame->sum[frame->sum_bank ^ 1u][0],
              et_e3_private_accumulate_v1(frame));
    CUT_STAGE(12, frame->result[0], et_e3_private_finalize_v1(frame));
#undef CUT_STAGE
    CHECK(0);
injected:
    CHECK(memcmp(frame->runtime, runtimes_before, sizeof(runtimes_before)) == 0);
    CHECK(frame->published_valid == valid_before &&
          memcmp(frame->published_counts, published_before,
                 sizeof(published_before)) == 0);
    for (size_t i = 0; i < 4; ++i) {
      uint32_t actual;
      read_scalar(destination[i], &actual);
      CHECK(actual == sentinels[i]);
    }
    abort_call(frame);
    CHECK(!frame->preinvoke_armed);
    OK(et_e3_private_acquire_v1(frame));
    stage(frame, 3, 197, 197, 0, 1, 1);
    forward_batch(frame);
    OK(et_e3_private_finalize_v1(frame));
    OK(et_e3_private_cleanup_preflight_v1(frame));
    et_e3_private_drain_v1(frame);
    OK(et_e3_private_publish_v1(frame));
    et_e3_private_finish_v1(frame);
    CHECK(frame->published_counts[0] == 2 && frame->published_counts[1] == 1);
  }
  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
}

int main(void) {
  owner *model = make_model();
  et_f32_tensor *destination[4] = {scalar(), scalar(), scalar(), scalar()};
  test_preinvoke_cuts(model, destination);
  for (size_t i = 0; i < 4; ++i) destroy_tensor(&destination[i]);
  printf("E3 R1 native preinvoke: %zu checks passed\n", checks);
  return 0;
}
