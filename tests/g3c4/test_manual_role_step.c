#define ET_G3C4_MANUAL_ROLE_STEP_PRIVATE 1
#include "test_manual_frame_commit.c"

static size_t provider_retries;

static void role_step_case(et_g3c4_model_owner_internal *owner,
                           int64_t kind, int64_t length) {
  const int64_t ids[2] = {41, 43}, prefix_id[1] = {47};
  float ignored[256], slot_before[512];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_logits_internal *pending;
  et_g3c4_manual_frame_internal *frame;
  et_a2_kv_cache *old_cache;
  cache_snapshot before, after;
  binding_snapshot binding, binding_after;
  at_candidate_snapshot staged_before, staged_after;
  int64_t rng[4], committed_length = 0;
  uint32_t copied[256], expected[256];
  et_f32_tensor_error f32_error;

  whole_reference(owner, kind, length);
  CHECK(et_g3c4_private_role_step_v1(NULL, 0) != 0);
  CHECK(et_g3c4_private_role_step_v1(context, 0) ==
        ET_G3C4_INVALID_STATE);
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
  CHECK(et_g3c4_private_role_step_v1(context, 0) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  frame = context->manual_frame;
  old_cache = context->cache;
  snapshot_cache(old_cache, &before);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  CHECK(et_g3c4_private_role_step_v1(context, -1) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_role_step_v1(context, 21) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_role_step_v1(pending, 0) != 0);
  for (int64_t ordinal = 0; ordinal <= 20; ordinal++) {
    CHECK(frame->next_ordinal == ordinal && frame->publication_state == 0u);
    if (ordinal > 0)
      CHECK(et_g3c4_private_role_step_v1(context, ordinal - 1) ==
            ET_G3C4_INVALID_STATE);
    if (ordinal < 20)
      CHECK(et_g3c4_private_role_step_v1(context, ordinal + 1) ==
            ET_G3C4_INVALID_STATE);
    CHECK(frame->next_ordinal == ordinal);
    if (ordinal == 10) {
      et_a2_kv_cache_test_fail_alloc_after_v1(0u);
      CHECK(et_g3c4_private_role_step_v1(context, ordinal) != 0);
      et_a2_kv_cache_test_reset_allocator_v1();
      CHECK(frame->next_ordinal == ordinal && frame->a2_candidate == NULL &&
            frame->a2_transaction == NULL && context->cache == old_cache);
    }
    if (ordinal >= 14) {
      float *slot = tail_slot(frame, ordinal);
      const size_t bytes = (size_t)length * tail_width(ordinal) * sizeof(float);
      memcpy(slot_before, slot, bytes);
      snapshot_at_candidate(frame, &staged_before);
      tail_context = context;
      tail_dispatches = 0u;
      record_tail = fail_tail = 1;
      CHECK(et_g3c4_private_role_step_v1(context, ordinal) != 0);
      record_tail = fail_tail = 0;
      CHECK(tail_dispatches == 1u && frame->next_ordinal == ordinal &&
            memcmp(slot_before, slot, bytes) == 0);
      CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
            et_g3c4_private_last_error_code_v1() ==
                ET_KERNEL_CODE_PROVIDER_REJECTED);
      snapshot_at_candidate(frame, &staged_after);
      CHECK(memcmp(&staged_before, &staged_after, sizeof(staged_before)) == 0);
      tail_dispatches = 0u;
      record_tail = 1;
      OK(et_g3c4_private_role_step_v1(context, ordinal));
      record_tail = 0;
      CHECK(tail_dispatches == 1u && memcmp(slot, tail_output, bytes) == 0);
      provider_retries++;
    } else {
      OK(et_g3c4_private_role_step_v1(context, ordinal));
    }
    CHECK(frame->next_ordinal == ordinal + 1);
  }
  CHECK(memcmp(frame->z + (size_t)(length - 1) * 256u,
               whole_outputs[6] + (size_t)(length - 1) * 256u,
               256u * sizeof(float)) == 0);
  CHECK(et_g3c4_private_role_step_v1(context, 20) ==
        ET_G3C4_INVALID_STATE);
  check_logits(pending, context);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  snapshot_binding(context, &binding_after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  memcpy(expected, frame->z + (size_t)(length - 1) * 256u,
         sizeof(expected));
  OK(et_g3c4_private_frame_prepare_v1(context, pending));
  CHECK(et_g3c4_private_role_step_v1(context, 20) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_frame_commit_v1(context));
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_output_committed_cache_length(context, &committed_length));
  CHECK(committed_length == (kind == 2 ? 2 : length));
  OK(et_f32_tensor_copy_bits_to_v1(pending->tensor, copied, 256u, &f32_error));
  CHECK(memcmp(copied, expected, sizeof(copied)) == 0);
  CHECK(et_g3c4_private_role_step_v1(context, 0) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_tensor_release_v1(pending));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  tail_case(owner, 1, 1);
  publication_case(owner, 1, 1, 0);
  role_step_case(owner, 1, 1);
  role_step_case(owner, 1, 2);
  role_step_case(owner, 2, 1);
  CHECK(provider_retries == 21u);
  printf("G3-C4 private manual role step PASS: checks=%zu routes=3 retries=%zu\n",
         checks, provider_retries);
  return 0;
}
