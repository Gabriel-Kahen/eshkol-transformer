#define ET_G3C4_MANUAL_FRAME_COMMIT_PRIVATE 1
#include "test_manual_tail.c"

static void publication_case(et_g3c4_model_owner_internal *owner,
                             int64_t kind, int64_t length,
                             int abort_after_prepare) {
  const int64_t ids[2] = {41, 43}, prefix_id[1] = {47};
  float ignored[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_logits_internal *pending;
  et_g3c4_manual_frame_internal *frame;
  et_a2_kv_cache *old_cache;
  et_a2_kv_cache_transaction_view *nested = NULL;
  et_kernel_error kernel_error;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error f32_error;
  cache_snapshot before, after;
  binding_snapshot binding, binding_after;
  int64_t rng[4];
  int64_t committed_length = 0;
  uint32_t copied[256], expected[256];

  if (kind == 2) {
    et_g3c4_input_internal *prefix = create_prompt(prefix_id, 1);
    et_g3c4_output_internal *output;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    output = et_g3c4_private_output_reserve_v1(context, 1);
    CHECK(output != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, prefix, ignored));
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_tensor_release_v1(prefix));
  }
  OK(et_g3c4_private_call_acquire_v1(context, kind - 1, 0));
  pending = et_g3c4_private_logits_reserve_v1(context);
  CHECK(pending != NULL);
  CHECK(et_g3c4_private_frame_prepare_v1(context, pending) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_manual_role0_run(context));
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++)
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
  OK(et_g3c4_manual_a2_run(context));
  OK(et_g3c4_manual_at_run(context));
  OK(et_g3c4_manual_ao_run(context));
  OK(et_g3c4_manual_r_run(context));
  for (int64_t ordinal = 14; ordinal <= 20; ordinal++)
    OK(et_g3c4_manual_tail_run(context, ordinal));
  frame = context->manual_frame;
  CHECK(frame->next_ordinal == 21 && frame->publication_state == 0u);
  old_cache = context->cache;
  snapshot_cache(old_cache, &before);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  check_logits(pending, context);
  CHECK(et_g3c4_private_frame_prepare_v1(context, input) ==
        ET_G3C4_INVALID_ARGUMENT);
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_frame_prepare_v1(context, pending) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(frame->publication_state == 0u);
  check_logits(pending, context);
  OK(et_f32_tensor_borrow_begin_v1(pending->tensor, &borrow, &f32_error));
  CHECK(et_g3c4_private_frame_prepare_v1(context, pending) != 0);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &f32_error));
  OK(et_g3c4_private_frame_prepare_v1(context, pending));
  CHECK(frame->publication_state == 1u &&
        et_g3c4_private_frame_prepare_v1(context, pending) ==
          ET_G3C4_INVALID_STATE);
  memcpy(expected, frame->z + (size_t)(length - 1) * 256u, sizeof(expected));
  OK(et_f32_tensor_copy_bits_to_v1(pending->tensor, copied, 256u, &f32_error));
  CHECK(memcmp(expected, copied, sizeof(expected)) == 0);
  if (abort_after_prepare) {
    OK(et_g3c4_private_call_abort_v1(context));
    CHECK(context->manual_frame == NULL &&
          pending->transport.state == ET_G3C4_CONTEXT_DEAD);
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&before, &after);
    snapshot_binding(context, &binding_after);
    check_binding_snapshot_equal(&binding, &binding_after);
    CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
    OK(et_g3c4_private_generator_close_v1(context));
    return;
  }
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_frame_commit_v1(context) == ET_G3C4_INVALID_STATE);
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(frame->publication_state == 1u);
  snapshot_cache(old_cache, &after);
  check_cache_snapshot_equal(&before, &after);
  snapshot_binding(context, &binding_after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  CHECK(frame->publication_state == 2u);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) ==
        ET_G3C4_INVALID_STATE);
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_frame_commit_v1(context) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(frame->publication_state == 2u && context->cache == old_cache);
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      frame->a2_transaction, 0u, &nested, &kernel_error));
  CHECK(et_g3c4_private_frame_commit_v1(context) != 0);
  CHECK(frame->publication_state == 2u && context->cache == old_cache);
  OK(et_a2_kv_cache_transaction_view_end_v1(&nested, &kernel_error));
  OK(et_g3c4_private_frame_commit_v1(context));
  CHECK(frame->publication_state == 3u && context->cache != old_cache &&
        frame->a2_candidate == NULL && frame->a2_transaction == NULL &&
        pending->parent_ctx == NULL &&
        pending->transport.state == ET_G3C4_LOGITS_PUBLISHED);
  CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_frame_commit_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_finish_v1(context));
  CHECK(context->manual_frame == NULL &&
        et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_output_committed_cache_length(context, &committed_length));
  CHECK(committed_length == (kind == 2 ? 2 : length));
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  OK(et_f32_tensor_copy_bits_to_v1(pending->tensor, copied, 256u, &f32_error));
  CHECK(memcmp(expected, copied, sizeof(expected)) == 0);
  if (kind == 1) {
    CHECK(context->prefill_binding_ready == 1u &&
          context->prefill_tokens[0] == ids[0] &&
          context->prefill_tokens[length - 1] == ids[length - 1]);
  } else {
    snapshot_binding(context, &binding_after);
    check_binding_snapshot_equal(&binding, &binding_after);
  }
  OK(et_g3c4_private_tensor_release_v1(pending));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  tail_case(owner, 1, 1);
  publication_case(owner, 1, 1, 0);
  publication_case(owner, 1, 2, 0);
  publication_case(owner, 2, 1, 0);
  publication_case(owner, 1, 1, 1);
  printf("G3-C4 private manual frame commit PASS: checks=%zu routes=4\n", checks);
  return 0;
}
