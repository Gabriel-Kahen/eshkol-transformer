#define ET_G3C4_P2_G2_CALL_A2_READ_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_predecessor
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

static void check_a2_error(int64_t category) {
  CHECK(category == ET_KERNEL_ERROR_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_KERNEL_CODE_PROVIDER_REJECTED);
}

static void check_held_cache(et_a2_kv_cache_read_borrow *held,
                             const cache_snapshot *expected) {
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  CHECK(held != NULL);
  OK(et_a2_kv_cache_read_borrow_layer_v1(
      held, 0u, &keys, &values, &lengths, &keep, &error));
  CHECK(memcmp(keys->data, expected->keys, sizeof(expected->keys)) == 0);
  CHECK(memcmp(values->data, expected->values, sizeof(expected->values)) == 0);
  CHECK(*(const int64_t *)lengths->data == expected->length);
  CHECK(memcmp(keep->data, expected->keep, sizeof(expected->keep)) == 0);
}

static void committed_read(et_g3c4_model_owner_internal *owner,
                           et_g3c4_model_owner_internal *foreign_owner,
                           int phase) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  et_g3c4_context_internal *foreign = p2g2_generator(foreign_owner, 0);
  et_a2_kv_cache *foreign_cache = foreign->cache;
  OK(et_g3c4_private_call_acquire_v1(foreign, 2, 2));
  et_g3c4_output_internal *foreign_output =
      et_g3c4_private_output_reserve_v1(foreign, 2);
  CHECK(foreign_output != NULL);
  float logits[256], original_logits[256];
  int64_t rng[4];
  binding_snapshot binding;
  cache_snapshot cache;
  if (phase == 3)
    start_prefix(owner, prompt, 0, -1, &context, &input, &output);
  else {
    active_pair(owner, 0, prompt, &context, &input, &output, NULL);
    if (phase == 2)
      OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  }
  et_a2_kv_cache *original_cache = context->cache;
  CHECK(context->p2g2_a2_read.cache == original_cache &&
        context->p2g2_a2_read.output == output && foreign_cache != original_cache);
  snapshot_cache(original_cache, &cache);
  CHECK(cache.length == phase);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  et_g3c4_output_internal copy = *output;
  CHECK(et_g3c4_p2g2_a2_read_begin(context, &copy) != 0);
  CHECK(et_g3c4_p2g2_a2_read_begin(foreign, output) != 0);
  context->cache = foreign_cache;
  CHECK(et_g3c4_p2g2_a2_read_begin(context, output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  context->cache = original_cache;

  et_kernel_error error;
  et_a2_kv_cache_read_borrow *outside = NULL;
  OK(et_a2_kv_cache_read_borrow_begin_v1(original_cache, &outside, &error));
  CHECK(et_g3c4_p2g2_a2_read_end(context, output) != 0);
  check_a2_error(et_g3c4_p2g2_a2_read_begin(context, output));
  CHECK(context->p2g2_a2_read.used == 0u && context->p2g2_a2_read.borrow == NULL);
  check_held_cache(outside, &cache);
  OK(et_a2_kv_cache_read_borrow_end_v1(&outside, &error));

  for (size_t cut = 0u; cut < 3u; cut++) {
    et_a2_kv_cache_test_fail_alloc_after_v1(cut);
    CHECK(et_g3c4_p2g2_a2_read_begin(context, output) != 0);
    et_a2_kv_cache_test_reset_allocator_v1();
    CHECK(context->p2g2_a2_read.used == 0u && context->p2g2_a2_read.borrow == NULL);
  }
  OK(et_g3c4_p2g2_a2_read_begin(context, output));
  et_a2_kv_cache_read_borrow *held = context->p2g2_a2_read.borrow;
  CHECK(et_g3c4_p2g2_a2_read_begin(context, output) != 0);
  CHECK(et_g3c4_p2g2_a2_read_end(context, &copy) != 0);
  CHECK(et_g3c4_p2g2_a2_read_end(foreign, output) != 0);

  unsigned char context_bytes[sizeof(*context)], output_bytes[sizeof(*output)];
  memcpy(context_bytes, context, sizeof(context_bytes));
  memcpy(output_bytes, output, sizeof(output_bytes));
  check_a2_error(et_g3c4_private_call_abort_v1(context));
  check_a2_error(et_g3c4_private_call_finish_v1(context));
  check_a2_error(et_g3c4_private_output_release_v1(output));
  memset(logits, 0xa5, sizeof(logits));
  memcpy(original_logits, logits, sizeof(logits));
  if (phase == 0) {
    check_a2_error(et_g3c4_private_prompt_prefill_v1(context, input, logits));
    CHECK(memcmp(logits, original_logits, sizeof(logits)) == 0);
  } else if (phase == 2) {
    int64_t selected = -1;
    check_a2_error(et_g3c4_private_token_frame_begin_last_v1(
        context, logits, &selected));
    CHECK(selected == -1);
  } else {
    bridge_carrier carrier = new_carrier(), unchanged = carrier;
    check_a2_error(et_g3c4_private_p2g2_second_frame_carrier_v1(
        context, output, &carrier, 16));
    CHECK(memcmp(&carrier, &unchanged, sizeof(carrier)) == 0);
  }
  CHECK(memcmp(context, context_bytes, sizeof(context_bytes)) == 0);
  CHECK(memcmp(output, output_bytes, sizeof(output_bytes)) == 0);
  CHECK(context->owner->active == context);

  context->cache = foreign_cache;
  if (phase == 0) {
    CHECK(et_g3c4_private_prompt_prefill_v1(context, input, logits) ==
          ET_G3C4_INVALID_STATE);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
    CHECK(memcmp(logits, original_logits, sizeof(logits)) == 0);
  }
  CHECK(et_g3c4_p2g2_a2_read_end(context, output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_output_release_v1(output) == ET_G3C4_INVALID_STATE);
  context->cache = original_cache;
  CHECK(memcmp(context, context_bytes, sizeof(context_bytes)) == 0);
  CHECK(memcmp(output, output_bytes, sizeof(output_bytes)) == 0);
  output->parent_ctx = foreign;
  foreign_output->parent_ctx = context;
  CHECK(et_g3c4_p2g2_a2_read_end(context, foreign_output) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_output_release_v1(output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  output->parent_ctx = context;
  foreign_output->parent_ctx = foreign;
  check_held_cache(held, &cache);
  cache_snapshot other_cache;
  snapshot_cache(foreign_cache, &other_cache);
  OK(et_g3c4_p2g2_a2_read_begin(foreign, foreign_output));
  CHECK(et_g3c4_p2g2_a2_read_end(context, foreign_output) != 0);
  CHECK(et_g3c4_p2g2_a2_read_end(foreign, output) != 0);
  OK(et_g3c4_p2g2_a2_read_end(context, output));
  check_held_cache(foreign->p2g2_a2_read.borrow, &other_cache);
  OK(et_g3c4_p2g2_a2_read_end(foreign, foreign_output));
  CHECK(et_g3c4_p2g2_a2_read_end(context, output) != 0);
  CHECK(et_g3c4_p2g2_a2_read_begin(context, output) != 0);
  OK(et_g3c4_private_call_abort_v1(context));
  check_preserved(context, &cache, &binding, rng);
  CHECK(et_g3c4_admit_idle_call(context) == context);

  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  et_g3c4_output_internal *next = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(next != NULL && next != output);
  OK(et_g3c4_p2g2_a2_read_begin(context, next));
  CHECK(et_g3c4_p2g2_a2_read_end(context, output) != 0);
  check_held_cache(context->p2g2_a2_read.borrow, &cache);
  OK(et_g3c4_p2g2_a2_read_end(context, next));
  close_pair(context, input, next);
  OK(et_g3c4_private_call_abort_v1(foreign));
  OK(et_g3c4_private_generator_close_v1(foreign));
}

static void cache_authority_starts_at_construction(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_model_owner_internal *foreign_owner) {
  const int64_t prompt[3] = {7, 11, 13};
  et_g3c4_context_internal *context = p2g2_generator(owner, 0);
  et_g3c4_context_internal *foreign = p2g2_generator(foreign_owner, 0);
  et_a2_kv_cache *original_cache = context->cache;
  CHECK(context->p2g2_a2_read.cache == original_cache);
  CHECK(foreign->cache != original_cache);
  cache_snapshot original, other, actual;
  snapshot_cache(original_cache, &original);
  snapshot_cache(foreign->cache, &other);
  check_cache_snapshot_equal(&original, &other);
  et_g3c4_input_internal *input = create_prompt(prompt, 2);
  context->cache = foreign->cache;
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_call_acquire_v1(context, 2, 2) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(et_g3c4_private_generator_close_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  CHECK(context->p2g2_a2_read.cache == original_cache &&
        context->owner->active == NULL && context->acquired_mask == 0u);
  context->cache = original_cache;
  snapshot_cache(context->cache, &actual);
  check_cache_snapshot_equal(&original, &actual);
  snapshot_cache(foreign->cache, &actual);
  check_cache_snapshot_equal(&other, &actual);

  /* A budget-two generator may legitimately replace its cache in another
   * budget, then enter a later budget-two call against that exact cache. */
  float logits[256];
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  OK(et_g3c4_private_prefill1_v1(context, prompt[0], logits));
  CHECK(context->p2g2_a2_read.cache == context->cache);
  snapshot_cache(context->cache, &actual);
  CHECK(actual.length == 1);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->p2g2_a2_read.cache == context->cache);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  OK(et_g3c4_private_prefill3_v1(context, prompt, logits));
  CHECK(context->p2g2_a2_read.cache == context->cache);
  snapshot_cache(context->cache, &actual);
  CHECK(actual.length == 3);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  et_g3c4_output_internal *output = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_p2g2_a2_read_begin(context, output));
  check_held_cache(context->p2g2_a2_read.borrow, &actual);
  OK(et_g3c4_p2g2_a2_read_end(context, output));
  close_pair(context, input, output);
  CHECK(context->p2g2_a2_read.cache == NULL);
  OK(et_g3c4_private_generator_close_v1(context));
  OK(et_g3c4_private_generator_close_v1(foreign));
}

static void replacement_and_retagged_release(
    et_g3c4_model_owner_internal *owner,
    et_g3c4_model_owner_internal *foreign_owner) {
  et_g3c4_context_internal *context = p2g2_generator(owner, 0);
  et_g3c4_context_internal *foreign = create_generator(foreign_owner);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 2));
  et_g3c4_output_internal *first = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(first != NULL);
  OK(et_g3c4_private_call_acquire_v1(foreign, 2, 1));
  et_g3c4_output_internal *unrelated = et_g3c4_private_output_reserve_v1(foreign, 2);
  CHECK(unrelated != NULL);
  cache_snapshot cache;
  snapshot_cache(context->cache, &cache);
  OK(et_g3c4_p2g2_a2_read_begin(context, first));

  first->generated_length = 1;
  check_a2_error(et_g3c4_private_output_release_v1(first));
  first->parent_ctx = foreign;
  CHECK(et_g3c4_private_output_release_v1(first) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  first->parent_ctx = context;
  first->generated_length = 2;
  OK(et_g3c4_private_output_release_v1(unrelated));
  check_dead_output(unrelated);
  check_held_cache(context->p2g2_a2_read.borrow, &cache);

  OK(et_g3c4_p2g2_a2_read_end(context, first));
  OK(et_g3c4_private_output_release_v1(first));
  check_dead_output(first);
  OK(et_g3c4_private_output_release_v1(first));
  et_g3c4_output_internal *next = et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(next != NULL && next != first);
  CHECK(et_g3c4_p2g2_a2_read_begin(context, next) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  context->cache = foreign->cache;
  CHECK(et_g3c4_private_output_release_v1(next) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  context->cache = context->p2g2_a2_read.cache;
  OK(et_g3c4_private_output_release_v1(next));
  check_dead_output(next);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_abort_v1(foreign));
  OK(et_g3c4_private_generator_close_v1(context));
  OK(et_g3c4_private_generator_close_v1(foreign));
}

static void staged_transaction_is_distinct(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  terminal_raw first, second;
  const int64_t prompt[2] = {7, 11};
  terminal_ready(owner, prompt, 0, &context, &input, &output, &first, &second);
  CHECK(et_g3c4_p2g2_a2_read_begin(context, output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_LIFECYCLE);
  CHECK(context->p2g2_a2_read.borrow == NULL);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  et_a2_kv_cache *original_cache = context->cache;
  unsigned char context_bytes[sizeof(*context)], output_bytes[sizeof(*output)];
  memcpy(context_bytes, context, sizeof(context_bytes));
  memcpy(output_bytes, output, sizeof(output_bytes));
  tamper_terminal_view_owner = context;
  tamper_terminal_view_at = 2;
  terminal_view_ends = 0;
  fail_terminal_cache_commit = 1;
  fail_terminal_copy = 1;
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
        context, output, &first, 9, &second, 9) != 0);
  CHECK(terminal_view_ends == 2);
  fail_terminal_copy = 0;
  fail_terminal_cache_commit = 0;
  tamper_terminal_view_owner = NULL;
  context->cache = original_cache;
  CHECK(memcmp(context, context_bytes, sizeof(context_bytes)) == 0);
  CHECK(memcmp(output, output_bytes, sizeof(output_bytes)) == 0);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
        context->token_frame_transaction, original_cache, 3) == 1);
  close_pair(context, input, output);
}

#ifdef ET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE
static void release_a2_while_i1_is_held(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  OK(et_g3c4_p2g2_call_i1_lease_begin(context, output));
  OK(et_g3c4_p2g2_a2_read_begin(context, output));
  check_a2_error(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_p2g2_a2_read_end(context, output));
  CHECK(et_g3c4_private_call_abort_v1(context) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  OK(et_g3c4_p2g2_call_i1_lease_end(context, output));
  close_pair(context, input, output);
}
#endif

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_model_owner_internal *foreign_owner = create_foreign_owner();
  committed_read(owner, foreign_owner, 0);
  committed_read(owner, foreign_owner, 2);
  committed_read(owner, foreign_owner, 3);
  cache_authority_starts_at_construction(owner, foreign_owner);
  replacement_and_retagged_release(owner, foreign_owner);
  staged_transaction_is_distinct(owner);
#ifdef ET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE
  release_a2_while_i1_is_held(owner);
#endif
  printf("G3-C4 P2/G2 call A2 committed read PASS: checks=%zu\n", checks);
  return 0;
}
