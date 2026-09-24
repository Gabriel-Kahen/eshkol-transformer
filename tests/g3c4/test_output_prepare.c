#define ET_G3C4_LAST_LOGIT_FRAME_PRIVATE 1
#define ET_G3C4_OUTPUT_PREPARE_PRIVATE 1
#define ET_G3C4_OUTPUT_RESERVATION_TEST_MAIN \
  et_g3c4_output_reservation_predecessor_main
#define __wrap_et_kernel_runtime_dispatch et_g3c4_output_prepare_base_dispatch
#include "test_output_reservation.c"
#undef __wrap_et_kernel_runtime_dispatch
#undef ET_G3C4_OUTPUT_RESERVATION_TEST_MAIN

static int record_token_forward;
static size_t token_forward_dispatches;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_token_forward) token_forward_dispatches++;
  return et_g3c4_output_prepare_base_dispatch(runtime, call, error);
}

static void check_numeric_output(
    et_g3c4_output_internal *output, int64_t p, int64_t g,
    int64_t expected_id, const int64_t expected_rng[4]) {
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(output->transport.state == 0u);
  CHECK(output->prompt_length == p);
  CHECK(output->generated_length == g);
  CHECK(output->length == g);
  CHECK(output->cache_length == p + g);
  CHECK(memcmp(output->rng, expected_rng, sizeof(output->rng)) == 0);
  CHECK(output->numeric_ready == 1u);
  CHECK(output->ids_copied == 0u);
  CHECK(output->text_ready == 0u);
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view->rank == 1u);
  CHECK(view->shape[0] == (uint64_t)g);
  CHECK(view->byte_length == (size_t)g * sizeof(int64_t));
  if (g == 1) CHECK(*(const int64_t *)view->data == expected_id);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
}

static void check_cache_snapshot_equal(
    const cache_snapshot *expected, const cache_snapshot *actual) {
  CHECK(memcmp(expected->keys, actual->keys, sizeof(expected->keys)) == 0);
  CHECK(memcmp(expected->values, actual->values,
               sizeof(expected->values)) == 0);
  CHECK(expected->length == actual->length);
  CHECK(memcmp(expected->keep, actual->keep, sizeof(expected->keep)) == 0);
}

static void check_binding_snapshot_equal(
    const binding_snapshot *expected, const binding_snapshot *actual) {
  CHECK(memcmp(expected->identities, actual->identities,
               sizeof(expected->identities)) == 0);
  CHECK(memcmp(expected->values, actual->values,
               sizeof(expected->values)) == 0);
  CHECK(memcmp(expected->tokens, actual->tokens,
               sizeof(expected->tokens)) == 0);
  CHECK(expected->ready == actual->ready);
}

static void prepare_g0(et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[2] = {41, 43};
  float logits[256];
  int64_t rng[4];
  cache_snapshot committed, after_prepare, after_abort;
  binding_snapshot binding, binding_after;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(tokens, 2);

  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  snapshot_cache(context->cache, &committed);
  snapshot_binding(context, &binding);
  CHECK(committed.length == 2);

  OK(et_g3c4_private_output_prepare_v1(context, output));
  check_numeric_output(output, 2, 0, 0, rng);
  snapshot_cache(context->cache, &after_prepare);
  check_cache_snapshot_equal(&committed, &after_prepare);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_numeric_output(output, 2, 0, 0, rng);

  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&committed, &after_abort);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void prepare_g1(et_g3c4_model_owner_internal *owner) {
  const int64_t token[1] = {47};
  float prefill_logits[256], token_logits[256];
  int64_t rng[4], successor[4];
  int64_t speculative = -1;
  cache_snapshot committed, after_abort;
  binding_snapshot binding, binding_after;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(token, 1);

  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 1));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, prefill_logits));
  snapshot_cache(context->cache, &committed);
  snapshot_binding(context, &binding);
  CHECK(committed.length == 1);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(output, context, 1, 1);

  OK(et_g3c4_private_token_frame_begin_last_v1(
      context, prefill_logits, &speculative));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(output, context, 1, 1);

  token_forward_dispatches = 0u;
  record_token_forward = 1;
  OK(et_g3c4_private_token_forward_v1(
      context, speculative, token_logits));
  record_token_forward = 0;
  CHECK(token_forward_dispatches == 21u);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  et_a2_kv_cache_transaction *transaction = context->token_frame_transaction;
  memcpy(successor, context->token_frame_successor, sizeof(successor));

  OK(et_g3c4_private_output_prepare_v1(context, output));
  check_numeric_output(output, 1, 1, speculative, successor);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(context->token_frame_transaction == transaction);
  CHECK(context->token_frame_candidate == speculative);
  CHECK(memcmp(context->token_frame_successor, successor,
               sizeof(successor)) == 0);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_numeric_output(output, 1, 1, speculative, successor);

  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&committed, &after_abort);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static size_t g0_cache_allocation_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[2] = {53, 59};
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 4u; allowed++) {
    float logits[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    void *input = create_prompt(tokens, 2);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    et_g3c4_output_internal *output =
        et_g3c4_private_output_reserve_v1(context, 2);
    CHECK(output != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    int64_t status = et_g3c4_private_output_prepare_v1(context, output);
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status != 0) {
      failures++;
      CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_ALLOCATION_FAILED);
      check_pending_output(output, context, 2, 0);
      check_preserved(context, &cache, &binding, rng);
    } else {
      check_numeric_output(output, 2, 0, 0, rng);
    }
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  return failures;
}

static et_g3c4_model_owner_internal *create_foreign_owner(void) {
  static unsigned char foreign_identities[14];
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(1730);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &foreign_identities[(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static void g1_borrow_cut_and_admission(
    et_g3c4_model_owner_internal *owner) {
  const int64_t token[1] = {61};
  float prefill_logits[256], token_logits[256];
  int64_t speculative = -1;
  int64_t rng[4], successor[4];
  cache_snapshot cache;
  binding_snapshot binding;
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_model_owner_internal *foreign_owner = create_foreign_owner();
  et_g3c4_context_internal *foreign_context = create_generator(foreign_owner);
  void *input = create_prompt(token, 1);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  CHECK(et_g3c4_private_output_prepare_v1(context, context) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_prepare_v1(output, output) ==
        ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_call_acquire_v1(foreign_context, 2, 1));
  et_g3c4_output_internal *foreign_output =
      et_g3c4_private_output_reserve_v1(foreign_context, 1);
  CHECK(foreign_output != NULL);
  CHECK(et_g3c4_private_output_prepare_v1(foreign_context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(foreign_output, foreign_context, 1, 1);
  OK(et_g3c4_private_call_abort_v1(foreign_context));

  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(output, context, 1, 1);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, prefill_logits));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(output, context, 1, 1);

  unsigned char saved_parameter =
      ((unsigned char *)context->pins.views[0].data)[0];
  ((unsigned char *)context->pins.views[0].data)[0] ^= 1u;
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_G3C4_CODE_STALE_BINDING);
  check_pending_output(output, context, 1, 1);
  ((unsigned char *)context->pins.views[0].data)[0] = saved_parameter;

  OK(et_g3c4_private_token_frame_begin_last_v1(
      context, prefill_logits, &speculative));
  CHECK(et_g3c4_private_output_prepare_v1(context, output) ==
        ET_G3C4_INVALID_STATE);
  check_pending_output(output, context, 1, 1);
  OK(et_g3c4_private_token_forward_v1(
      context, speculative, token_logits));
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  et_a2_kv_cache_transaction *transaction =
      context->token_frame_transaction;
  int64_t position = context->token_frame_position;
  et_i64_tensor *ids = output->ids;
  int64_t output_length = output->length;
  int64_t output_cache_length = output->cache_length;
  int64_t output_rng[4];
  memcpy(output_rng, output->rng, sizeof(output_rng));

  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  int64_t saved_id = *(const int64_t *)view->data;
  CHECK(et_g3c4_private_output_prepare_v1(context, output) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(output->ids == ids);
  CHECK(output->length == output_length);
  CHECK(output->cache_length == output_cache_length);
  CHECK(memcmp(output->rng, output_rng, sizeof(output_rng)) == 0);
  CHECK(output->numeric_ready == 0u);
  CHECK(output->ids_copied == 0u);
  CHECK(output->text_ready == 0u);
  CHECK(*(const int64_t *)view->data == saved_id);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(context->token_frame_transaction == transaction);
  CHECK(context->token_frame_candidate == speculative);
  CHECK(memcmp(context->token_frame_successor, successor,
               sizeof(successor)) == 0);
  CHECK(context->token_frame_position == position);
  binding_snapshot binding_during;
  snapshot_binding(context, &binding_during);
  check_binding_snapshot_equal(&binding, &binding_during);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_output_prepare_v1(context, output));
  check_numeric_output(output, 1, 1, speculative, successor);

  OK(et_g3c4_private_call_abort_v1(context));
  cache_snapshot cache_after;
  binding_snapshot binding_after;
  snapshot_cache(context->cache, &cache_after);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&cache, &cache_after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  OK(et_g3c4_private_generator_close_v1(foreign_context));
}

#ifndef ET_G3C4_OUTPUT_PREPARE_TEST_MAIN
#define ET_G3C4_OUTPUT_PREPARE_TEST_MAIN main
#endif

int ET_G3C4_OUTPUT_PREPARE_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  prepare_g0(owner);
  prepare_g1(owner);
  size_t cache_cuts = g0_cache_allocation_cuts(owner);
  CHECK(cache_cuts == 3u);
  g1_borrow_cut_and_admission(owner);
  printf("G3-C4 output prepare PASS: checks=%zu routes=2 "
         "token-dispatches=21 cache-cuts=%zu borrow-cuts=1 "
         "ownership-cuts=3 repeat-cuts=2 readiness-cuts=4\n",
         checks, cache_cuts);
  return 0;
}
