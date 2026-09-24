#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE 1
#define ET_G3C4_OUTPUT_PREPARE_TEST_MAIN \
  et_g3c4_output_prepare_predecessor_main
#include "test_output_prepare.c"
#undef ET_G3C4_OUTPUT_PREPARE_TEST_MAIN

typedef struct decode_staging {
  int64_t length;
  unsigned char bytes[8];
} decode_staging;

typedef struct prepared_output {
  et_g3c4_context_internal *context;
  et_g3c4_output_internal *output;
  et_i64_tensor *ids;
  void *input;
  int64_t token;
  int64_t initial_rng[4];
  int64_t result_rng[4];
  et_a2_kv_cache_transaction *transaction;
  int64_t position;
  binding_snapshot binding;
} prepared_output;

static prepared_output make_prepared(
    et_g3c4_model_owner_internal *owner, int64_t generated) {
  const int64_t tokens[2] = {67, 71};
  float prefill_logits[256], token_logits[256];
  prepared_output result;
  memset(&result, 0, sizeof(result));
  result.token = -1;
  result.context = create_generator(owner);
  result.input = create_prompt(tokens, generated == 0 ? 2 : 1);
  memcpy(result.initial_rng, result.context->generator_rng_words,
         sizeof(result.initial_rng));
  OK(et_g3c4_private_call_acquire_v1(result.context, 2, generated));
  result.output = et_g3c4_private_output_reserve_v1(
      result.context, generated == 0 ? 2 : 1);
  CHECK(result.output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(
      result.context, result.input, prefill_logits));
  if (generated == 1) {
    OK(et_g3c4_private_token_frame_begin_last_v1(
        result.context, prefill_logits, &result.token));
    OK(et_g3c4_private_token_forward_v1(
        result.context, result.token, token_logits));
    memcpy(result.result_rng, result.context->token_frame_successor,
           sizeof(result.result_rng));
  } else {
    memcpy(result.result_rng, result.initial_rng, sizeof(result.result_rng));
  }
  OK(et_g3c4_private_output_prepare_v1(result.context, result.output));
  result.ids = result.output->ids;
  result.transaction = result.context->token_frame_transaction;
  result.position = result.context->token_frame_position;
  snapshot_binding(result.context, &result.binding);
  return result;
}

static void discard_prepared(prepared_output *prepared) {
  OK(et_g3c4_private_call_abort_v1(prepared->context));
  OK(et_g3c4_private_tensor_release_v1(prepared->input));
  OK(et_g3c4_private_generator_close_v1(prepared->context));
}

static void check_decode_pending(
    const prepared_output *prepared, uint32_t copied) {
  CHECK(prepared->output->transport.state == 0u);
  CHECK(prepared->output->parent_ctx == prepared->context);
  CHECK(prepared->output->ids == prepared->ids);
  CHECK(prepared->output->prompt_length ==
        (prepared->output->generated_length == 0 ? 2 : 1));
  CHECK(prepared->output->generated_length ==
        (prepared->token < 0 ? 0 : 1));
  CHECK(prepared->output->numeric_ready == 1u);
  CHECK(prepared->output->ids_copied == copied);
  CHECK(prepared->output->text_ready == 0u);
  CHECK(prepared->output->length == prepared->output->generated_length);
  CHECK(prepared->output->cache_length ==
        prepared->output->prompt_length +
            prepared->output->generated_length);
  CHECK(memcmp(prepared->output->rng, prepared->result_rng,
               sizeof(prepared->output->rng)) == 0);
  CHECK(memcmp(prepared->context->generator_rng_words,
               prepared->initial_rng, sizeof(prepared->initial_rng)) == 0);
  binding_snapshot binding;
  snapshot_binding(prepared->context, &binding);
  check_binding_snapshot_equal(&prepared->binding, &binding);
  if (prepared->output->generated_length == 1) {
    CHECK(prepared->context->token_frame_state ==
          ET_G3C4_TOKEN_FRAME_READY);
    CHECK(prepared->context->token_frame_transaction ==
          prepared->transaction);
    CHECK(prepared->context->token_frame_candidate == prepared->token);
    CHECK(memcmp(prepared->context->token_frame_successor,
                 prepared->result_rng, sizeof(prepared->result_rng)) == 0);
    CHECK(prepared->context->token_frame_position == prepared->position);
  } else {
    CHECK(et_g3c4_token_frame_idle(prepared->context));
  }
}

static void check_staging_unchanged(
    const decode_staging *staging, int64_t length,
    unsigned char byte) {
  CHECK(staging->length == length);
  for (size_t index = 0u; index < sizeof(staging->bytes); index++)
    CHECK(staging->bytes[index] == byte);
}

static void g0_route_and_aliases(et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[2] = {73, 79};
  float logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(tokens, 2);
  int64_t initial_rng[4];
  memcpy(initial_rng, context->generator_rng_words, sizeof(initial_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));

  decode_staging early = {0, {0}};
  memset(early.bytes, 0xa1, sizeof(early.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, &early) == ET_G3C4_INVALID_STATE);
  check_staging_unchanged(&early, 0, 0xa1);
  check_pending_output(output, context, 2, 0);

  OK(et_g3c4_private_output_prepare_v1(context, output));
  prepared_output prepared;
  memset(&prepared, 0, sizeof(prepared));
  prepared.context = context;
  prepared.output = output;
  prepared.ids = output->ids;
  prepared.input = input;
  prepared.token = -1;
  memcpy(prepared.initial_rng, initial_rng, sizeof(initial_rng));
  memcpy(prepared.result_rng, output->rng, sizeof(prepared.result_rng));
  snapshot_binding(context, &prepared.binding);
  check_decode_pending(&prepared, 0u);

  decode_staging malformed = {8, {0}};
  memset(malformed.bytes, 0xa2, sizeof(malformed.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, &malformed) == ET_G3C4_SHAPE_MISMATCH);
  check_staging_unchanged(&malformed, 8, 0xa2);
  check_decode_pending(&prepared, 0u);

  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, context) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, output) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  et_a2_kv_cache_read_borrow *cache_borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *keep = NULL;
  et_kernel_error kernel_error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &cache_borrow, &kernel_error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      cache_borrow, 0u, &keys, &values, &lengths, &keep,
      &kernel_error) == 0);
  void *cache_alias = keys->data;
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      &cache_borrow, &kernel_error) == 0);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, cache_alias) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  decode_staging exact = {0, {0}};
  memset(exact.bytes, 0xa3, sizeof(exact.bytes));
  OK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, &exact));
  check_staging_unchanged(&exact, 0, 0xa3);
  check_decode_pending(&prepared, 1u);

  decode_staging repeat = {0, {0}};
  memset(repeat.bytes, 0xa4, sizeof(repeat.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      context, output, &repeat) == ET_G3C4_INVALID_STATE);
  check_staging_unchanged(&repeat, 0, 0xa4);
  check_decode_pending(&prepared, 1u);
  discard_prepared(&prepared);
}

static void g1_failures_and_success(
    et_g3c4_model_owner_internal *owner) {
  prepared_output prepared = make_prepared(owner, 1);
  et_i64_tensor_error error;
  check_decode_pending(&prepared, 0u);

  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, NULL) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output,
      (void *)(uintptr_t)(UINTPTR_MAX - 7u)) ==
      ET_G3C4_INVALID_ARGUMENT);
  check_decode_pending(&prepared, 0u);
  for (int64_t bad_length = -1; bad_length <= 9; bad_length += 8) {
    decode_staging bad = {bad_length, {0}};
    memset(bad.bytes, 0xb1, sizeof(bad.bytes));
    CHECK(et_g3c4_private_output_copy_decode_ids_v1(
        prepared.context, prepared.output, &bad) ==
        ET_G3C4_SHAPE_MISMATCH);
    check_staging_unchanged(&bad, bad_length, 0xb1);
    check_decode_pending(&prepared, 0u);
  }

  unsigned char saved_parameter =
      ((unsigned char *)prepared.context->pins.views[0].data)[0];
  ((unsigned char *)prepared.context->pins.views[0].data)[0] ^= 1u;
  decode_staging stale = {8, {0}};
  memset(stale.bytes, 0xb2, sizeof(stale.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, &stale) ==
      ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_G3C4_CODE_STALE_BINDING);
  check_staging_unchanged(&stale, 8, 0xb2);
  ((unsigned char *)prepared.context->pins.views[0].data)[0] =
      saved_parameter;

  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output,
      prepared.context->pins.views[0].data) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  const int64_t alias_tokens[2] = {8, 199};
  et_g3c4_input_internal *alias_input =
      create_prompt(alias_tokens, 2);
  et_i64_tensor_borrow *alias_borrow = NULL;
  const et_kernel_tensor_view_v1 *alias_view = NULL;
  CHECK(et_i64_tensor_borrow_begin_v1(
      alias_input->tensor, &alias_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(
      alias_borrow, &alias_view, &error) == 0);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, alias_view->data) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(((const int64_t *)alias_view->data)[0] == alias_tokens[0]);
  CHECK(((const int64_t *)alias_view->data)[1] == alias_tokens[1]);
  check_decode_pending(&prepared, 0u);
  CHECK(et_i64_tensor_borrow_end_v1(&alias_borrow, &error) == 0);
  OK(et_g3c4_private_tensor_release_v1(alias_input));

  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  CHECK(et_i64_tensor_borrow_begin_v1(
      prepared.output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  void *ids_alias = view->data;
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, ids_alias) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  decode_staging held = {8, {0}};
  memset(held.bytes, 0xb3, sizeof(held.bytes));
  view = NULL;
  CHECK(et_i64_tensor_borrow_begin_v1(
      prepared.output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  int64_t held_id = *(const int64_t *)view->data;
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, &held) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  check_staging_unchanged(&held, 8, 0xb3);
  check_decode_pending(&prepared, 0u);
  CHECK(*(const int64_t *)view->data == held_id);
  CHECK(prepared.context->token_frame_state ==
        ET_G3C4_TOKEN_FRAME_READY);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);

  int64_t invalid_id = 256;
  OK(et_i64_tensor_copy_from_v1(
      prepared.output->ids, &invalid_id, 1u, &error));
  decode_staging invalid = {8, {0}};
  memset(invalid.bytes, 0xb4, sizeof(invalid.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, &invalid) ==
      ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_G3C4_CODE_INVARIANT);
  check_staging_unchanged(&invalid, 8, 0xb4);
  check_decode_pending(&prepared, 0u);
  OK(et_i64_tensor_copy_from_v1(
      prepared.output->ids, &prepared.token, 1u, &error));

  decode_staging exact = {8, {0}};
  memset(exact.bytes, 0xb5, sizeof(exact.bytes));
  OK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.output, &exact));
  CHECK(exact.length == 8);
  CHECK(exact.bytes[0] == (unsigned char)prepared.token);
  for (size_t index = 1u; index < sizeof(exact.bytes); index++)
    CHECK(exact.bytes[index] == 0u);
  check_decode_pending(&prepared, 1u);
  discard_prepared(&prepared);
}

static size_t borrow_allocation_cuts(
    et_g3c4_model_owner_internal *owner) {
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 2u; allowed++) {
    prepared_output prepared = make_prepared(owner, 1);
    decode_staging staging = {8, {0}};
    memset(staging.bytes, 0xc1, sizeof(staging.bytes));
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    int64_t status = et_g3c4_private_output_copy_decode_ids_v1(
        prepared.context, prepared.output, &staging);
    et_i64_tensor_test_reset_allocator_v1();
    if (status != 0) {
      failures++;
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
      check_staging_unchanged(&staging, 8, 0xc1);
      check_decode_pending(&prepared, 0u);
    } else {
      CHECK(staging.bytes[0] == (unsigned char)prepared.token);
      check_decode_pending(&prepared, 1u);
    }
    discard_prepared(&prepared);
  }
  return failures;
}

static void ownership_cut(et_g3c4_model_owner_internal *owner) {
  prepared_output prepared = make_prepared(owner, 1);
  et_g3c4_model_owner_internal *foreign_owner = create_foreign_owner();
  et_g3c4_context_internal *foreign_context =
      create_generator(foreign_owner);
  OK(et_g3c4_private_call_acquire_v1(foreign_context, 2, 1));
  et_g3c4_output_internal *foreign_output =
      et_g3c4_private_output_reserve_v1(foreign_context, 1);
  CHECK(foreign_output != NULL);
  decode_staging staging = {8, {0}};
  memset(staging.bytes, 0xd1, sizeof(staging.bytes));
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      foreign_context, prepared.output, &staging) ==
      ET_G3C4_INVALID_STATE);
  check_staging_unchanged(&staging, 8, 0xd1);
  check_decode_pending(&prepared, 0u);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.output, prepared.output, &staging) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared.context, prepared.context, &staging) ==
      ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_call_abort_v1(foreign_context));
  OK(et_g3c4_private_generator_close_v1(foreign_context));
  discard_prepared(&prepared);
}

#ifndef ET_G3C4_OUTPUT_DECODE_IDS_TEST_MAIN
#define ET_G3C4_OUTPUT_DECODE_IDS_TEST_MAIN main
#endif

int ET_G3C4_OUTPUT_DECODE_IDS_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  g0_route_and_aliases(owner);
  g1_failures_and_success(owner);
  size_t allocation_cuts = borrow_allocation_cuts(owner);
  CHECK(allocation_cuts == 1u);
  ownership_cut(owner);
  printf("G3-C4 output decode IDs PASS: checks=%zu routes=2 "
         "borrow-allocation-cuts=%zu active-borrow-cuts=1 "
         "alias-cuts=6 ownership-cuts=3 readiness-cuts=2 "
         "malformed-cuts=5 repeat-cuts=1\n",
         checks, allocation_cuts);
  return 0;
}
