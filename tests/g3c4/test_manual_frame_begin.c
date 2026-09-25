#define ET_G3C4_MANUAL_FRAME_BEGIN_PRIVATE 1
#define ET_G3C4_LOGITS_RESERVATION_TEST_MAIN \
  et_g3c4_logits_reservation_predecessor_main
#include "test_logits_reservation.c"
#undef ET_G3C4_LOGITS_RESERVATION_TEST_MAIN

static void check_frame(
    et_g3c4_context_internal *context, int64_t kind,
    const int64_t *ids, int64_t length) {
  CHECK(context->manual_frame != NULL);
  CHECK(context->manual_frame->frame_kind == kind);
  CHECK(context->manual_frame->input_length == length);
  CHECK(context->manual_frame->next_ordinal == 0);
  CHECK(memcmp(context->manual_frame->input_ids, ids,
               (size_t)length * sizeof(ids[0])) == 0);
}

static void manual_prefill(
    et_g3c4_model_owner_internal *owner, int64_t length) {
  const int64_t ids[2] = {41, 43};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_logits_internal *logits;
  cache_snapshot before, after;
  int64_t rng[4];
  snapshot_cache(context->cache, &before);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 1) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_frame_begin_v1(context, NULL, 1) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 2) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 3) ==
        ET_G3C4_INVALID_ARGUMENT);
  logits = et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL);
  CHECK(et_g3c4_private_frame_begin_v1(context, logits, 1) ==
        ET_G3C4_INVALID_ARGUMENT);
  et_g3c4_context_allocation_limit = et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 1) ==
        ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  CHECK(context->manual_frame == NULL);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  OK(et_g3c4_private_frame_begin_v1(context, input, 1));
  check_frame(context, 1, ids, length);
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 1) ==
        ET_G3C4_INVALID_STATE);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  check_logits(logits, context);
  OK(et_g3c4_private_tensor_release_v1(input));
  check_frame(context, 1, ids, length);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_tensor_release_v1(logits));
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void borrowed_input(et_g3c4_model_owner_internal *owner) {
  const int64_t ids[1] = {49};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, 1);
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_logits_reserve_v1(context) != NULL);
  OK(et_i64_tensor_borrow_begin_v1(input->tensor, &borrow, &error));
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 1) != 0);
  CHECK(context->manual_frame == NULL);
  OK(et_i64_tensor_borrow_end_v1(&borrow, &error));
  OK(et_g3c4_private_frame_begin_v1(context, input, 1));
  check_frame(context, 1, ids, 1);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void manual_decode(et_g3c4_model_owner_internal *owner) {
  const int64_t prefix[1] = {41}, next[1] = {47};
  float numeric[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(prefix, 1);
  et_g3c4_input_internal *decode = create_prompt(next, 1);
  et_g3c4_input_internal *too_long = create_prompt((const int64_t[2]){47, 49}, 2);
  et_g3c4_logits_internal *logits;
  et_g3c4_output_internal *output;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error error;
  cache_snapshot before, after;
  CHECK(et_g3c4_private_call_acquire_v1(context, 1, 0) == 0);
  logits = et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL);
  CHECK(et_g3c4_private_frame_begin_v1(context, too_long, 2) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_frame_begin_v1(context, decode, 2) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  output = et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, numeric));
  OK(et_g3c4_private_call_abort_v1(context));
  snapshot_cache(context->cache, &before);
  CHECK(before.length == 1);
  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  logits = et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL);
  CHECK(et_g3c4_private_frame_begin_v1(context, too_long, 2) ==
        ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_frame_begin_v1(context, decode, 2));
  check_frame(context, 2, next, 1);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  OK(et_f32_tensor_borrow_begin_v1(logits->tensor, &borrow, &error));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  check_frame(context, 2, next, 1);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_tensor_release_v1(decode));
  OK(et_g3c4_private_tensor_release_v1(too_long));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void generate_rejects_frame(et_g3c4_model_owner_internal *owner) {
  const int64_t ids[1] = {53};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, 1);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 1) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_frame_begin_v1(context, input, 2) ==
        ET_G3C4_INVALID_STATE);
  CHECK(context->manual_frame == NULL);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

#ifndef ET_G3C4_MANUAL_FRAME_BEGIN_TEST_MAIN
#define ET_G3C4_MANUAL_FRAME_BEGIN_TEST_MAIN main
#endif
int ET_G3C4_MANUAL_FRAME_BEGIN_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  manual_prefill(owner, 1);
  manual_prefill(owner, 2);
  borrowed_input(owner);
  manual_decode(owner);
  generate_rejects_frame(owner);
  printf("G3-C4 manual frame begin PASS: checks=%zu prefill=2 decode=1 allocation=1 borrow=2\n", checks);
  return 0;
}
