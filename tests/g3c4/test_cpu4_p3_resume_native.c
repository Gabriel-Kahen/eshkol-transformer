#define ET_CPU4_C4_P3_RESUME_PRIVATE 1
#define ET_CPU4_C4_HISTORY_TEST_MAIN cpu4_history_predecessor_main
#include "test_cpu4_history_native.c"

static void resume_parity(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *before, *after;
  et_g3c4_input_internal *before_input, *after_input;
  et_g3c4_output_internal *before_output, *after_output;
  int64_t ids3[3], rng[4], final_rng[4];
  float logits[256], replay_logits[256];
  cache_snapshot prefix, replay;
  terminal_snapshot original, resumed;
  bridge_carrier carrier;

  start_prefix(owner, prompt, 1, -1,
               &before, &before_input, &before_output);
  memcpy(ids3, before->cpu4_history.ids, sizeof(ids3));
  memcpy(rng, before->generator_rng_words, sizeof(rng));
  memcpy(logits, before->p2g2_prefix_next_logits, sizeof(logits));
  snapshot_cache(before->cache, &prefix);
  carrier = new_carrier();
  OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
      before, before_output, &carrier, 16));
  terminal_raw first = raw_id(ids3[2]);
  terminal_raw second = raw_id(before->token_frame_candidate);
  OK(et_g3c4_private_p2g2_terminal_commit_v1(
      before, before_output, &first, 9, &second, 9));
  memcpy(final_rng, before->generator_rng_words, sizeof(final_rng));
  memset(&original, 0, sizeof(original));
  original.length = 64;
  OK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      before_output, &original, sizeof(original)));
  OK(et_g3c4_private_tensor_release_v1(before_input));
  OK(et_g3c4_private_output_release_v1(before_output));
  OK(et_g3c4_private_generator_close_v1(before));

  /* Controlled native state fixture; checked file import is a later gate. */
  void *imported_rng = et_g3c4_private_rng_seed_v1(rng[1]);
  CHECK(imported_rng != NULL);
  memcpy(((et_g3c4_rng_internal *)imported_rng)->words,
         rng, sizeof(rng));
  after = et_g3c4_private_generator_rng_v1(
      owner, imported_rng, 1, INT64_C(0x3f800000), 17,
      INT64_C(0x3f000000), 2, -1);
  CHECK(after != NULL);
  OK(et_g3c4_private_rng_release_v1(imported_rng));
  after_input = create_prompt(prompt, 2);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(
      after, after_input, 2));
  OK(et_g3c4_private_call_acquire_v1(after, 2, 2));
  after_output = et_g3c4_private_output_reserve_v1(after, 2);
  CHECK(after_output != NULL);
  size_t draws = history_samples;
  cache_snapshot empty_before, empty_after;
  cpu4_history_record history_before = after->cpu4_history;
  snapshot_cache(after->cache, &empty_before);
  CHECK(et_g3c4_private_prefill3_v1(after, ids3, replay_logits) != 0);
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, before_output, ids3, rng, replay_logits) != 0);
  union { int64_t ids[128]; float logits[256]; } alias = {0};
  memcpy(alias.ids, ids3, sizeof(ids3));
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, alias.ids, rng, alias.logits) != 0);
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error ids_error;
  OK(et_i64_tensor_borrow_begin_v1(after_output->ids,
                                    &borrow, &ids_error));
  OK(et_i64_tensor_borrow_view_v1(borrow, &view, &ids_error));
  float *owned_logits = (float *)view->data;
  OK(et_i64_tensor_borrow_end_v1(&borrow, &ids_error));
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, rng, owned_logits) != 0);
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, (const int64_t *)owned_logits,
      replay_logits) != 0);
  int64_t wrong_rng[4];
  memcpy(wrong_rng, rng, sizeof(wrong_rng));
  wrong_rng[2]++;
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, wrong_rng, replay_logits) != 0);
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, NULL, replay_logits) != 0);
  record_prefill = 1;
  prefill_dispatches = 0u;
  fail_prefill_at = 10u;
  CHECK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, rng, replay_logits) != 0);
  record_prefill = 0;
  fail_prefill_at = SIZE_MAX;
  snapshot_cache(after->cache, &empty_after);
  CHECK(memcmp(empty_before.keys, empty_after.keys,
               sizeof(empty_before.keys)) == 0 &&
        memcmp(empty_before.values, empty_after.values,
               sizeof(empty_before.values)) == 0 &&
        memcmp(empty_before.keep, empty_after.keep,
               sizeof(empty_before.keep)) == 0 &&
        empty_before.length == empty_after.length);
  CHECK(memcmp(&history_before, &after->cpu4_history,
               sizeof(history_before)) == 0 &&
        memcmp(rng, after->generator_rng_words, sizeof(rng)) == 0 &&
        after->p2g2_prefix_committed == 0u &&
        after->prefill_binding_ready == 0u &&
        history_samples == draws);
  assert_pending(after_output);
  OK(et_cpu4_private_c4_p3_resume_prefix_v1(
      after, after_output, ids3, rng, replay_logits));
  CHECK(history_samples == draws);
  CHECK(memcmp(logits, replay_logits, sizeof(logits)) == 0);
  CHECK(memcmp(rng, after->generator_rng_words, sizeof(rng)) == 0);
  CHECK(after->cpu4_history.count == 3u &&
        after->cpu4_history.prompt_count == 2u &&
        after->cpu4_history.generated_count == 1u &&
        after->cpu4_history.origin[2] == 2u &&
        after->cpu4_history.commit_epoch == 2u);
  snapshot_cache(after->cache, &replay);
  CHECK(memcmp(prefix.keys, replay.keys, sizeof(prefix.keys)) == 0 &&
        memcmp(prefix.values, replay.values, sizeof(prefix.values)) == 0 &&
        memcmp(prefix.keep, replay.keep, sizeof(prefix.keep)) == 0 &&
        prefix.length == replay.length);
  carrier = new_carrier();
  OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
      after, after_output, &carrier, 16));
  first = raw_id(ids3[2]);
  second = raw_id(after->token_frame_candidate);
  OK(et_g3c4_private_p2g2_terminal_commit_v1(
      after, after_output, &first, 9, &second, 9));
  CHECK(memcmp(final_rng, after->generator_rng_words,
               sizeof(final_rng)) == 0);
  memset(&resumed, 0, sizeof(resumed));
  resumed.length = 64;
  OK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      after_output, &resumed, sizeof(resumed)));
  CHECK(memcmp(&original, &resumed, sizeof(original)) == 0);
  OK(et_g3c4_private_tensor_release_v1(after_input));
  OK(et_g3c4_private_output_release_v1(after_output));
  OK(et_g3c4_private_generator_close_v1(after));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  resume_parity(owner);
  printf("CPU4 C4 P3 resume native PASS: checks=%zu\n", checks);
  return 0;
}
