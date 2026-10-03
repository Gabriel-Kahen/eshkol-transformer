#define ET_G3C4_P2_G1_PUBLICATION_PRIVATE 1
#define ET_G3C4_P2_G1_PENDING_TEST_MAIN \
  et_g3c4_p2g1_pending_predecessor_main
#include "test_p2_g1_pending.c"
#undef ET_G3C4_P2_G1_PENDING_TEST_MAIN

typedef struct publication_fixture {
  et_g3c4_context_internal *context;
  et_g3c4_output_internal *output;
  void *input;
  int64_t token;
  int64_t prior_rng[4];
  int64_t successor[4];
  float last[256], next[256];
  cache_snapshot p2;
  binding_snapshot binding;
} publication_fixture;

static publication_fixture publication_ready(
    et_g3c4_model_owner_internal *owner, int categorical) {
  const int64_t prompt[2] = {0, 255};
  publication_fixture fixture;
  memset(&fixture, 0, sizeof(fixture));
  fixture.context = categorical
      ? p2g1_categorical_generator(owner) : create_generator(owner);
  fixture.input = create_prompt(prompt, 2);
  memcpy(fixture.prior_rng, fixture.context->generator_rng_words,
         sizeof(fixture.prior_rng));
  OK(et_g3c4_private_call_acquire_v1(fixture.context, 2, 1));
  fixture.output = et_g3c4_private_output_reserve_v1(fixture.context, 2);
  CHECK(fixture.output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(
      fixture.context, fixture.input, fixture.last));
  snapshot_cache(fixture.context->cache, &fixture.p2);
  snapshot_binding(fixture.context, &fixture.binding);
  CHECK(fixture.p2.length == 2);
  fixture.token = -1;
  OK(et_g3c4_private_token_frame_begin_last_v1(
      fixture.context, fixture.last, &fixture.token));
  OK(et_g3c4_private_token_forward_v1(
      fixture.context, fixture.token, fixture.next));
  memcpy(fixture.successor, fixture.context->token_frame_successor,
         sizeof(fixture.successor));
  OK(et_g3c4_private_output_prepare_v1(fixture.context, fixture.output));
  decode_staging staging = {8, {0}};
  raw_carrier raw = {1, {(unsigned char)fixture.token}};
  OK(et_g3c4_private_output_copy_decode_ids_v1(
      fixture.context, fixture.output, &staging));
  OK(et_g3c4_private_output_accept_text_v1(
      fixture.context, fixture.output, &raw));
  CHECK(staging.bytes[0] == (unsigned char)fixture.token);
  return fixture;
}

static void check_pending_unchanged(const publication_fixture *fixture) {
  binding_snapshot binding;
  snapshot_binding(fixture->context, &binding);
  check_binding_snapshot_equal(&fixture->binding, &binding);
  CHECK(memcmp(fixture->context->generator_rng_words, fixture->prior_rng,
               sizeof(fixture->prior_rng)) == 0);
  CHECK(fixture->context->token_frame_transaction != NULL);
  CHECK(fixture->output->transport.state == 0u);
  CHECK(fixture->output->parent_ctx == fixture->context);
  CHECK(fixture->output->numeric_ready == 1u &&
        fixture->output->ids_copied == 1u &&
        fixture->output->text_ready == 1u);
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      fixture->context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  CHECK(keys != NULL && values != NULL && lengths != NULL && keep != NULL);
  CHECK(*(const int64_t *)lengths->data == 3);
  CHECK(memcmp(keep->data, (const uint8_t[4]){1,1,1,0}, 4u) == 0);
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
}

static void publication_success(
    et_g3c4_model_owner_internal *owner, int categorical) {
  publication_fixture fixture = publication_ready(owner, categorical);
  et_g3c4_context_internal *context = fixture.context;
  et_g3c4_output_internal *output = fixture.output;
  int64_t stray = -77;
  CHECK(et_g3c4_private_generation_frame_commit_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_token_frame_publish_v1(context, &stray) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_token_frame_abort_v1(context) ==
        ET_G3C4_INVALID_STATE);
  CHECK(stray == -77);
  check_pending_unchanged(&fixture);
  OK(et_g3c4_private_generation_frame_prepare_v1(context, output));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_PREPARED);
  CHECK(et_g3c4_private_output_release_v1(output) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_END_READY);
  check_pending_unchanged(&fixture);
  OK(et_g3c4_private_generation_frame_commit_v1(context, output));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_COMMITTED);
  CHECK(context->token_frame_transaction == NULL && context->budget == 0);
  CHECK(memcmp(context->generator_rng_words, fixture.successor,
               sizeof(fixture.successor)) == 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);
#ifdef ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
#endif
  CHECK(output->parent_ctx == NULL && output->ids != NULL);
  CHECK(output->length == 1 && output->cache_length == 3);
  CHECK(memcmp(output->rng, fixture.successor,
               sizeof(fixture.successor)) == 0);
  CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_token_frame_abort_v1(context) ==
        ET_G3C4_INVALID_STATE);
  cache_snapshot committed, reference_cache;
  snapshot_cache(context->cache, &committed);
  CHECK(committed.length == 3);
  CHECK(memcmp(committed.keep, (const uint8_t[4]){1,1,1,0}, 4u) == 0);
  OK(et_g3c4_private_call_finish_v1(context));
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_tensor_release_v1(fixture.input));
  OK(et_g3c4_private_generator_close_v1(context));
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);

  uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      output->ids);
  size_t *stride = (size_t *)et_i64_tensor_test_stride_storage_v1(
      output->ids);
  CHECK(shape != NULL && stride != NULL);
  shape[0] = 2u;
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);
  shape[0] = 1u;
  stride[0] = 16u;
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);
  stride[0] = sizeof(int64_t);
  et_i64_tensor *owned_ids = output->ids;
  output->ids = (et_i64_tensor *)(uintptr_t)0x1000u;
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);
  output->ids = owned_ids;

  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error ids_error;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &ids_error));
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_PUBLISHED);
  OK(et_i64_tensor_borrow_end_v1(&borrow, &ids_error));
  OK(et_g3c4_private_output_release_v1(output));
  check_dead_output(output);
  OK(et_g3c4_private_output_release_v1(output));

  const int64_t ids[3] = {0, 255, fixture.token};
  float reference_logits[256];
  et_g3c4_context_internal *reference = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 1));
  OK(et_g3c4_private_prefill3_v1(reference, ids, reference_logits));
  CHECK(memcmp(fixture.next, reference_logits, sizeof(fixture.next)) == 0);
  snapshot_cache(reference->cache, &reference_cache);
  check_cache_snapshot_equal(&committed, &reference_cache);
  print_oracle_case(categorical ? "categorical" : "greedy",
                    fixture.last, fixture.token, fixture.prior_rng,
                    fixture.successor);
  OK(et_g3c4_private_call_abort_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
}

static void publication_rejections(et_g3c4_model_owner_internal *owner) {
  publication_fixture fixture = publication_ready(owner, 0);
  et_g3c4_context_internal *context = fixture.context;
  et_g3c4_output_internal *output = fixture.output;
  int64_t *id = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
  uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      output->ids);
  size_t *stride = (size_t *)et_i64_tensor_test_stride_storage_v1(
      output->ids);
  CHECK(id != NULL && shape != NULL && stride != NULL);
  for (int mode = 0; mode < 8; mode++) {
    if (mode == 0) *id = -1;
    if (mode == 1) *id = 256;
    if (mode == 2) shape[0] = 2u;
    if (mode >= 3 && mode <= 5) malformed_output_view = mode - 2;
    if (mode == 6) output->rng[2] ^= 1;
    if (mode == 7) stride[0] = 16u;
    CHECK(et_g3c4_private_generation_frame_prepare_v1(
        context, output) != 0);
    *id = fixture.token;
    shape[0] = 1u;
    stride[0] = sizeof(int64_t);
    malformed_output_view = 0;
    memcpy(output->rng, fixture.successor, sizeof(output->rng));
    CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
    check_pending_unchanged(&fixture);
  }
  CHECK(et_g3c4_private_generation_frame_prepare_v1(
      context, (void *)(uintptr_t)0x1000u) != 0);
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error ids_error;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &ids_error));
  CHECK(et_g3c4_private_generation_frame_prepare_v1(context, output) != 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  check_pending_unchanged(&fixture);
  OK(et_i64_tensor_borrow_end_v1(&borrow, &ids_error));
  et_a2_kv_cache_transaction_view *view = NULL;
  et_kernel_error cache_error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &view, &cache_error));
  CHECK(et_g3c4_private_generation_frame_prepare_v1(context, output) != 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(context->token_frame_transaction != NULL);
  CHECK(output->transport.state == 0u && output->parent_ctx == context);
  CHECK(memcmp(context->generator_rng_words, fixture.prior_rng,
               sizeof(fixture.prior_rng)) == 0);
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &cache_error));
  check_pending_unchanged(&fixture);

  OK(et_g3c4_private_generation_frame_prepare_v1(context, output));
  CHECK(et_g3c4_private_generation_frame_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  et_a2_kv_cache_test_reset_allocator_v1();
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_PREPARED);
  check_pending_unchanged(&fixture);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  context->token_frame_candidate ^= 1;
  CHECK(et_g3c4_private_generation_frame_commit_v1(context, output) != 0);
  context->token_frame_candidate ^= 1;
  context->prefill_binding_values[0] ^= 1u;
  CHECK(et_g3c4_private_generation_frame_commit_v1(context, output) != 0);
  context->prefill_binding_values[0] ^= 1u;
  et_a2_kv_cache_transaction_view *commit_view = NULL;
  et_kernel_error commit_error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &commit_view, &commit_error));
  CHECK(et_g3c4_private_generation_frame_commit_v1(context, output) != 0);
  OK(et_a2_kv_cache_transaction_view_end_v1(
      &commit_view, &commit_error));
  output->length = 2;
  CHECK(et_g3c4_private_generation_frame_commit_v1(context, output) != 0);
  output->length = 1;
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_END_READY);
  check_pending_unchanged(&fixture);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  cache_snapshot after;
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&fixture.p2, &after);
  CHECK(memcmp(context->generator_rng_words, fixture.prior_rng,
               sizeof(fixture.prior_rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(fixture.input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void sampled_abort_and_retry(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {0, 255};
  float last[256], next[256];
  int64_t token = -1, original_rng[4];
  cache_snapshot p2, after;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(prompt, 2);
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  snapshot_cache(context->cache, &p2);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_token_frame_idle(context));
  check_dead_output(output);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&p2, &after);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  output = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  OK(et_g3c4_private_output_prepare_v1(context, output));
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner);
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_prefill2_v1(context, prompt, last));
  snapshot_cache(context->cache, &p2);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  OK(et_g3c4_private_token_frame_abort_v1(context));
  CHECK(et_g3c4_token_frame_idle(context));
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&p2, &after);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

#ifndef ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN
#define ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN main
#endif
int ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN(void) {
  CHECK(et_g3c4_p2g1_pending_predecessor_main() == 0);
  et_g3c4_model_owner_internal *owner = p2g1_test_owner;
  publication_success(owner, 0);
  publication_success(owner, 1);
  publication_rejections(owner);
  sampled_abort_and_retry(owner);
  printf("G3-C4 P2/G1 publication PASS: checks=%zu\n", checks);
  return 0;
}
