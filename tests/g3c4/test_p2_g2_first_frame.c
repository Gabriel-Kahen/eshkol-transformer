#define ET_G3C4_P2_G2_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G1_PENDING_TEST_MAIN et_g3c4_p2g1_pending_predecessor_main
#include "test_p2_g1_pending.c"
#undef ET_G3C4_P2_G1_PENDING_TEST_MAIN

static et_g3c4_context_internal *p2g2_generator(
    et_g3c4_model_owner_internal *owner, int categorical) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, categorical ? 1729 : 7, categorical, INT64_C(0x3f800000),
      categorical ? 17 : 256,
      categorical ? INT64_C(0x3f000000) : INT64_C(0x3f800000),
      2, -1);
  CHECK(context != NULL);
  return context;
}

static void first_frame(et_g3c4_model_owner_internal *owner,
                        int categorical) {
  const int64_t prompt[2] = {0, 255};
  int64_t reference_ids[3] = {0, 255, -1};
  et_g3c4_context_internal *context = p2g2_generator(owner, categorical);
  et_g3c4_input_internal *input = create_prompt(prompt, 2);
  float last[256], next[256], reference_logits[256];
  int64_t token = -1, untouched = -73, before_rng[4];
  int64_t legacy_rng[4] = {-3, -4, -5, -6};
  cache_snapshot committed, after_abort, reference_cache;
  et_i64_tensor_error ids_error;
  uint64_t ids_extent = 0;
  memcpy(before_rng, context->generator_rng_words, sizeof(before_rng));
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_i64_tensor_shape_at_v1(output->ids, 0u, &ids_extent, &ids_error));
  CHECK(ids_extent == 2u);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  snapshot_cache(context->cache, &committed);
  CHECK(committed.length == 2);
  CHECK(memcmp(committed.keep, (const uint8_t[4]){1, 1, 0, 0}, 4u) == 0);
  float legacy_logits[1024] = {0};
  CHECK(et_g3c4_private_sample_last_v1(
      context, legacy_logits, &untouched, legacy_rng) ==
        ET_G3C4_INVALID_STATE);
  CHECK(memcmp(legacy_rng, (const int64_t[4]){-3, -4, -5, -6},
               sizeof(legacy_rng)) == 0);
  CHECK(et_g3c4_private_token_frame_begin_v1(
      context, legacy_logits, &untouched) == ET_G3C4_INVALID_STATE);
  CHECK(untouched == -73);
  CHECK(et_g3c4_private_token_forward_v1(context, 0, next) ==
        ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  CHECK(token >= 0 && token <= 255);
  print_oracle_case(categorical ? "categorical" : "greedy",
                    last, token, before_rng,
                    context->token_frame_successor);
  CHECK(context->token_frame_position == 2);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(output->numeric_ready == 0u && output->ids_copied == 0u &&
        output->text_ready == 0u);
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  CHECK(*(const int64_t *)lengths->data == 3);
  CHECK(memcmp(keep->data, (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
  float staged_keys[16], staged_values[16];
  memcpy(staged_keys, keys->data, sizeof(staged_keys));
  memcpy(staged_values, values->data, sizeof(staged_values));
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_token_frame_publish_v1(context, &untouched) ==
        ET_G3C4_INVALID_STATE);
  CHECK(untouched == -73);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) ==
        ET_G3C4_INVALID_STATE);
  CHECK(output->numeric_ready == 0u && output->length == 0 &&
        output->cache_length == 0);
  CHECK(memcmp(context->generator_rng_words, before_rng,
               sizeof(before_rng)) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort);
  check_cache_snapshot_equal(&committed, &after_abort);
  CHECK(memcmp(context->generator_rng_words, before_rng,
               sizeof(before_rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  reference_ids[2] = token;
  et_g3c4_context_internal *reference = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 1));
  OK(et_g3c4_private_prefill3_v1(
      reference, reference_ids, reference_logits));
  CHECK(memcmp(next, reference_logits, sizeof(next)) == 0);
  snapshot_cache(reference->cache, &reference_cache);
  CHECK(reference_cache.length == 3);
  CHECK(memcmp(staged_keys, reference_cache.keys,
               sizeof(staged_keys)) == 0);
  CHECK(memcmp(staged_values, reference_cache.values,
               sizeof(staged_values)) == 0);
  OK(et_g3c4_private_call_abort_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
}

static void rejection_and_retry(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  const int64_t short_prompt[1] = {7};
  et_g3c4_context_internal *context = p2g2_generator(owner, 0);
  et_g3c4_input_internal *input = create_prompt(prompt, 2);
  et_g3c4_input_internal *short_input = create_prompt(short_prompt, 1);
  float logits[256], next[256];
  int64_t token = -1, original_rng[4];
  cache_snapshot before, after;
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, short_input, 2) == ET_G3C4_SHAPE_MISMATCH);
  snapshot_cache(context->cache, &before);
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  memset(logits, 0x7f, sizeof(logits));
  /* The legacy raw-pointer entry cannot commit P2 under a budget-two call,
   * even if its bytes happen to match the owned I1 prompt. */
  CHECK(et_g3c4_private_prefill2_v1(context, prompt, logits) ==
        ET_G3C4_INVALID_STATE);
  for (size_t i = 0; i < sizeof(logits); i++)
    CHECK(((const unsigned char *)logits)[i] == 0x7f);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  CHECK(et_g3c4_private_output_reserve_v1(context, 1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_SHAPE_MISMATCH);
  et_i64_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_output_reserve_v1(context, 2) == NULL);
  et_i64_tensor_test_reset_allocator_v1();
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  int64_t *ids = (int64_t *)et_i64_tensor_test_data_storage_v1(input->tensor);
  CHECK(ids != NULL);
  CHECK(et_g3c4_private_prefill2_v1(context, ids, logits) ==
        ET_G3C4_INVALID_STATE);
  CHECK(output->numeric_ready == 0u && output->ids_copied == 0u &&
        output->text_ready == 0u);
  for (size_t i = 0; i < sizeof(logits); i++)
    CHECK(((const unsigned char *)logits)[i] == 0x7f);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  for (size_t position = 0; position < 2u; position++) {
    for (size_t bad = 0; bad < 2u; bad++) {
      int64_t original = ids[position];
      ids[position] = bad ? 256 : -1;
      CHECK(et_g3c4_private_prompt_prefill_v1(context, input, logits) != 0);
      ids[position] = original;
      snapshot_cache(context->cache, &after);
      check_cache_snapshot_equal(&before, &after);
      CHECK(memcmp(context->generator_rng_words, original_rng,
                   sizeof(original_rng)) == 0);
    }
  }
  for (int mode = 1; mode <= 4; mode++) {
    malformed_input_view = mode;
    CHECK(et_g3c4_private_prompt_prefill_v1(context, input, logits) != 0);
    malformed_input_view = 0;
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&before, &after);
  }
  et_i64_tensor_borrow *held_input = NULL;
  et_i64_tensor_error ids_error;
  OK(et_i64_tensor_borrow_begin_v1(input->tensor, &held_input, &ids_error));
  CHECK(et_g3c4_private_prompt_prefill_v1(context, input, logits) != 0);
  OK(et_i64_tensor_borrow_end_v1(&held_input, &ids_error));
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  et_a2_kv_cache_read_borrow *held_cache = NULL;
  et_kernel_error kernel_error;
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held_cache, &kernel_error));
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, logits, &token) != 0);
  OK(et_a2_kv_cache_read_borrow_end_v1(&held_cache, &kernel_error));
  CHECK(et_g3c4_token_frame_idle(context));
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, logits, &token) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  forge_sample = 1;
  forged_token = 256;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, logits, &token) != 0);
  forge_sample = 0;
  CHECK(token == -1);
  CHECK(et_g3c4_token_frame_idle(context));
  OK(et_g3c4_private_token_frame_begin_last_v1(context, logits, &token));
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(short_input));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner);
  CHECK(et_g3c4_private_call_acquire_v1(context, 2, 2) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_pins_idle(&context->pins));
  OK(et_g3c4_private_generator_close_v1(context));
}

#ifndef ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN
#define ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN main
#endif
int ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  first_frame(owner, 0);
  first_frame(owner, 1);
  rejection_and_retry(owner);
  printf("G3-C4 P2/G2 first frame PASS: checks=%zu\n", checks);
  return 0;
}
