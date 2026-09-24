#define ET_I64_TENSOR_TESTING 1
#define ET_G3C4_PREFILL1_PRIVATE 1
#define ET_G3C4_PREFILL2_PRIVATE 1
#define ET_G3C4_PROMPT_T1_BORROW_PRIVATE 1
#define ET_G3C4_PROMPT_PREFILL_PRIVATE 1
#define ET_G3C4_OUTPUT_RESERVATION_PRIVATE 1
#define main et_g3c4_prefill3_predecessor_main
#include "test_prefill3.c"
#undef main

static void *create_prompt(const int64_t *tokens, int64_t length) {
  void *shell = et_t1_i64_shell_create_v1(length);
  CHECK(shell != NULL);
  for (int64_t index = 0; index < length; index++)
    CHECK(et_t1_i64_shell_write_v1(shell, index, tokens[index]) == 0);
  CHECK(et_t1_i64_shell_seal_v1(shell) == 0);
  void *input = et_g3c4_private_input_from_t1_v1(shell);
  CHECK(input != NULL);
  return input;
}

static size_t output_count(void) {
  size_t count = 0u;
  et_g3c4_transport_header_internal *header;
  for (header = et_g3c4_transport_registry;
       header != NULL; header = header->registry_next)
    if (header->kind == ET_G3C4_OUTPUT_KIND) count++;
  return count;
}

static void check_active(
    et_g3c4_context_internal *context, int64_t budget) {
  CHECK(ET_G3C4_CONTEXT_BUSY(context) == ET_G3C4_CALL_ACTIVE);
  CHECK(context->call_kind == 2);
  CHECK(context->budget == budget);
  CHECK(context->acquired_mask == ET_G3C4_ACQUIRED_FULL);
  CHECK(context->owner->active == context);
}

static void check_idle(et_g3c4_context_internal *context) {
  CHECK(ET_G3C4_CONTEXT_BUSY(context) == ET_G3C4_CALL_IDLE);
  CHECK(context->call_kind == 0);
  CHECK(context->budget == 0);
  CHECK(context->acquired_mask == 0u);
  CHECK(context->owner->active == NULL);
}

static void check_pending_output(
    et_g3c4_output_internal *output,
    et_g3c4_context_internal *context, int64_t p, int64_t g) {
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(output->transport.magic == ET_G3C4_OUTPUT_MAGIC);
  CHECK(output->transport.kind == ET_G3C4_OUTPUT_KIND);
  CHECK(output->transport.state == 0u);
  CHECK(output->parent_ctx == context);
  CHECK(output->prompt_length == p);
  CHECK(output->generated_length == g);
  CHECK(output->ids != NULL);
  CHECK(output->length == 0);
  CHECK(output->cache_length == 0);
  CHECK(memcmp(output->rng, (const int64_t[4]){0,0,0,0},
               sizeof(output->rng)) == 0);
  CHECK(output->numeric_ready == 0u);
  CHECK(output->ids_copied == 0u);
  CHECK(output->text_ready == 0u);
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view->rank == 1u);
  CHECK(view->shape[0] == (uint64_t)g);
  CHECK(view->byte_length == (size_t)g * sizeof(int64_t));
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
}

static void check_dead_output(et_g3c4_output_internal *output) {
  CHECK(output->transport.state == ET_G3C4_CONTEXT_DEAD);
  CHECK(output->parent_ctx == NULL);
  CHECK(output->prompt_length == 0);
  CHECK(output->generated_length == 0);
  CHECK(output->ids == NULL);
  CHECK(output->length == 0);
  CHECK(output->cache_length == 0);
  CHECK(memcmp(output->rng, (const int64_t[4]){0,0,0,0},
               sizeof(output->rng)) == 0);
  CHECK(output->numeric_ready == 0u);
  CHECK(output->ids_copied == 0u);
  CHECK(output->text_ready == 0u);
}

static void reserved_prefill_abort(
    et_g3c4_model_owner_internal *owner, int64_t p, int64_t g) {
  const int64_t tokens[2] = {0, 255};
  float logits[256];
  int64_t rng[4];
  cache_snapshot committed_cache, after_abort_cache;
  binding_snapshot committed_binding, after_abort_binding;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(tokens, p);

  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, g));
  OK(et_g3c4_private_call_acquire_v1(context, 2, g));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, p);
  CHECK(output != NULL);
  check_pending_output(output, context, p, g);
  CHECK(et_g3c4_private_output_reserve_v1(context, p) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);

  if (p == 1) {
    prefill1_dispatches = 0u;
    record_prefill1 = 1;
  } else {
    prefill2_dispatches = 0u;
    record_prefill2 = 1;
  }
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  record_prefill1 = 0;
  record_prefill2 = 0;
  CHECK(p == 1 ? prefill1_dispatches == 21u
               : prefill2_dispatches == 21u);
  check_pending_output(output, context, p, g);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  snapshot_cache(context->cache, &committed_cache);
  snapshot_binding(context, &committed_binding);
  CHECK(committed_cache.length == p);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_finish_v1(context) == ET_G3C4_INVALID_STATE);
  check_active(context, g);

  OK(et_g3c4_private_call_abort_v1(context));
  check_idle(context);
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort_cache);
  snapshot_binding(context, &after_abort_binding);
  CHECK(memcmp(committed_cache.keys, after_abort_cache.keys,
               sizeof(committed_cache.keys)) == 0);
  CHECK(memcmp(committed_cache.values, after_abort_cache.values,
               sizeof(committed_cache.values)) == 0);
  CHECK(committed_cache.length == after_abort_cache.length);
  CHECK(memcmp(committed_cache.keep, after_abort_cache.keep,
               sizeof(committed_cache.keep)) == 0);
  CHECK(memcmp(committed_binding.identities, after_abort_binding.identities,
               sizeof(committed_binding.identities)) == 0);
  CHECK(memcmp(committed_binding.values, after_abort_binding.values,
               sizeof(committed_binding.values)) == 0);
  CHECK(memcmp(committed_binding.tokens, after_abort_binding.tokens,
               sizeof(committed_binding.tokens)) == 0);
  CHECK(committed_binding.ready == after_abort_binding.ready);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  OK(et_g3c4_private_output_release_v1(output));
  OK(et_g3c4_private_output_release_v1(output));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void admission_and_linkage(
    et_g3c4_model_owner_internal *owner) {
  const int64_t p1[1] = {7};
  const int64_t p2[2] = {7, 11};
  float output[256];
  cache_snapshot cache;
  binding_snapshot binding;
  int64_t rng[4];
  et_g3c4_context_internal *context = create_generator(owner);
  void *input1 = create_prompt(p1, 1);
  void *input2 = create_prompt(p2, 2);

  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  for (size_t index = 0u; index < 256u; index++) output[index] = -31.0f;
  CHECK(et_g3c4_private_prompt_prefill_v1(context, input1, output) ==
        ET_G3C4_INVALID_STATE);
  for (size_t index = 0u; index < 256u; index++)
    CHECK(output[index] == -31.0f);
  check_preserved(context, &cache, &binding, rng);
  CHECK(et_g3c4_private_output_reserve_v1(context, 0) == NULL);
  CHECK(et_g3c4_private_output_reserve_v1(context, 3) == NULL);
  et_g3c4_output_internal *mismatch =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(mismatch != NULL);
  CHECK(et_g3c4_private_prompt_prefill_v1(context, input1, output) ==
        ET_G3C4_INVALID_STATE);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_output_release_v1(mismatch));
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_output_reserve_v1(context, 2) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_SHAPE_MISMATCH);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_output_reserve_v1(context, 1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_abort_v1(context));

  CHECK(et_g3c4_private_output_release_v1(input1) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_release_v1(context) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_release_v1(&cache) ==
        ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_tensor_release_v1(input1));
  OK(et_g3c4_private_tensor_release_v1(input2));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void output_alias_rejection(
    et_g3c4_model_owner_internal *owner) {
  const int64_t token[1] = {29};
  cache_snapshot cache;
  binding_snapshot binding;
  int64_t rng[4];
  float safe_logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(token, 1);

  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 1));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  et_i64_tensor_test_fail_alloc_after_v1(0u);
  prefill1_dispatches = 0u;
  record_prefill1 = 1;
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input, safe_logits) != 0);
  record_prefill1 = 0;
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
  CHECK(prefill1_dispatches == 0u);
  check_preserved(context, &cache, &binding, rng);
  check_pending_output(output, context, 1, 1);

  prefill1_dispatches = 0u;
  record_prefill1 = 1;
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input, (float *)(void *)output) == ET_G3C4_INVALID_ARGUMENT);
  record_prefill1 = 0;
  CHECK(prefill1_dispatches == 0u);
  check_preserved(context, &cache, &binding, rng);
  check_pending_output(output, context, 1, 1);

  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  float *ids_alias = (float *)(void *)view->data;
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  prefill1_dispatches = 0u;
  record_prefill1 = 1;
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input, ids_alias) == ET_G3C4_INVALID_ARGUMENT);
  record_prefill1 = 0;
  CHECK(prefill1_dispatches == 0u);
  check_preserved(context, &cache, &binding, rng);
  check_pending_output(output, context, 1, 1);

  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void nonidle_frame_borrow_atomicity(
    et_g3c4_model_owner_internal *owner) {
  const int64_t token[1] = {31};
  float logits[256];
  float full_logits[1024] = {0};
  cache_snapshot committed_cache, after_abort_cache;
  binding_snapshot committed_binding, after_abort_binding;
  int64_t rng[4];
  int64_t speculative = -1;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(token, 1);

  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 1));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  snapshot_cache(context->cache, &committed_cache);
  snapshot_binding(context, &committed_binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  memcpy(full_logits + 3u * 256u, logits, sizeof(logits));
  OK(et_g3c4_private_token_frame_begin_v1(
      context, full_logits, &speculative));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  et_a2_kv_cache_transaction *saved_transaction =
      context->token_frame_transaction;
  int64_t saved_candidate = context->token_frame_candidate;
  int64_t saved_successor[4];
  memcpy(saved_successor, context->token_frame_successor,
         sizeof(saved_successor));
  int64_t saved_position = context->token_frame_position;

  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(context->token_frame_transaction == saved_transaction);
  CHECK(context->token_frame_candidate == saved_candidate);
  CHECK(memcmp(context->token_frame_successor, saved_successor,
               sizeof(saved_successor)) == 0);
  CHECK(context->token_frame_position == saved_position);
  CHECK(output->transport.state == 0u);
  CHECK(output->parent_ctx == context);
  check_active(context, 1);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);

  OK(et_g3c4_private_call_abort_v1(context));
  check_idle(context);
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort_cache);
  snapshot_binding(context, &after_abort_binding);
  CHECK(memcmp(committed_cache.keys, after_abort_cache.keys,
               sizeof(committed_cache.keys)) == 0);
  CHECK(memcmp(committed_cache.values, after_abort_cache.values,
               sizeof(committed_cache.values)) == 0);
  CHECK(committed_cache.length == after_abort_cache.length);
  CHECK(memcmp(committed_cache.keep, after_abort_cache.keep,
               sizeof(committed_cache.keep)) == 0);
  CHECK(memcmp(committed_binding.identities, after_abort_binding.identities,
               sizeof(committed_binding.identities)) == 0);
  CHECK(memcmp(committed_binding.values, after_abort_binding.values,
               sizeof(committed_binding.values)) == 0);
  CHECK(memcmp(committed_binding.tokens, after_abort_binding.tokens,
               sizeof(committed_binding.tokens)) == 0);
  CHECK(committed_binding.ready == after_abort_binding.ready);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static size_t i1_allocation_cuts(
    et_g3c4_model_owner_internal *owner, int64_t p, int64_t g) {
  const int64_t tokens[2] = {13, 17};
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 5u; allowed++) {
    et_g3c4_context_internal *context = create_generator(owner);
    void *input = create_prompt(tokens, p);
    size_t before = output_count();
    OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, g));
    OK(et_g3c4_private_call_acquire_v1(context, 2, g));
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    et_g3c4_output_internal *output =
        et_g3c4_private_output_reserve_v1(context, p);
    et_i64_tensor_test_reset_allocator_v1();
    if (output == NULL) {
      failures++;
      CHECK(output_count() == before);
      CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
    } else {
      CHECK(output_count() == before + 1u);
      check_pending_output(output, context, p, g);
      OK(et_g3c4_private_output_release_v1(output));
    }
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  return failures;
}

static void allocation_and_borrow_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t token[1] = {23};
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(token, 1);
  size_t before = output_count();
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_context_allocation_limit = 0u;
  et_g3c4_context_successful_allocations = 0u;
  CHECK(et_g3c4_private_output_reserve_v1(context, 1) == NULL);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_context_successful_allocations = 0u;
  CHECK(output_count() == before);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  check_active(context, 1);

  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  et_a2_kv_cache_read_borrow *cache_borrow = NULL;
  et_kernel_error cache_error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &cache_borrow, &cache_error) == 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_KERNEL_CODE_PROVIDER_REJECTED);
  check_pending_output(output, context, 1, 1);
  check_active(context, 1);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      &cache_borrow, &cache_error) == 0);

  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(output->transport.state == 0u);
  CHECK(output->parent_ctx == context);
  CHECK(output->prompt_length == 1);
  CHECK(output->generated_length == 1);
  CHECK(output->ids != NULL);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(output->transport.state == 0u);
  CHECK(output->parent_ctx == context);
  CHECK(output->ids != NULL);
  check_active(context, 1);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  check_pending_output(output, context, 1, 1);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

#ifndef ET_G3C4_OUTPUT_RESERVATION_TEST_MAIN
#define ET_G3C4_OUTPUT_RESERVATION_TEST_MAIN main
#endif

int ET_G3C4_OUTPUT_RESERVATION_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  reserved_prefill_abort(owner, 2, 0);
  reserved_prefill_abort(owner, 1, 1);
  admission_and_linkage(owner);
  output_alias_rejection(owner);
  size_t g0_cuts = i1_allocation_cuts(owner, 2, 0);
  size_t g1_cuts = i1_allocation_cuts(owner, 1, 1);
  CHECK(g0_cuts == 3u);
  CHECK(g1_cuts == 4u);
  allocation_and_borrow_cuts(owner);
  nonidle_frame_borrow_atomicity(owner);
  printf("G3-C4 output reservation PASS: checks=%zu routes=2 "
         "owner-cuts=1 g0-i1-cuts=%zu g1-i1-cuts=%zu "
         "prefill-i1-cuts=1 borrow-cuts=4 alias-cuts=2\n",
         checks, g0_cuts, g1_cuts);
  return 0;
}
