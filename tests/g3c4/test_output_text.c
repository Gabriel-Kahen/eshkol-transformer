#define ET_G3C4_OUTPUT_TEXT_PRIVATE 1
#define ET_G3C4_OUTPUT_DECODE_IDS_TEST_MAIN \
  et_g3c4_output_decode_ids_predecessor_main
#include "test_output_decode_ids.c"
#undef ET_G3C4_OUTPUT_DECODE_IDS_TEST_MAIN

typedef struct raw_carrier {
  int64_t length;
  unsigned char bytes[8];
} raw_carrier;

static void copy_ids(prepared_output *prepared) {
  decode_staging staging = {
    prepared->output->generated_length * (int64_t)sizeof(int64_t), {0}};
  OK(et_g3c4_private_output_copy_decode_ids_v1(
      prepared->context, prepared->output, &staging));
  check_decode_pending(prepared, 1u);
}

static void check_text_pending(
    const prepared_output *prepared, uint32_t ready) {
  CHECK(prepared->output->numeric_ready == 1u);
  CHECK(prepared->output->ids_copied == 1u);
  CHECK(prepared->output->text_ready == ready);
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
}

static void g0_text(et_g3c4_model_owner_internal *owner) {
  prepared_output prepared = make_prepared(owner, 0);
  raw_carrier raw = {0, {0xa1}};
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_INVALID_STATE);
  CHECK(raw.bytes[0] == 0xa1);
  copy_ids(&prepared);

  raw.length = 1;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_SHAPE_MISMATCH);
  CHECK(raw.bytes[0] == 0xa1);
  check_text_pending(&prepared, 0u);
  raw.length = 0;
  et_a2_kv_cache_read_borrow *cache_borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *keep = NULL;
  et_kernel_error kernel_error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      prepared.context->cache, &cache_borrow, &kernel_error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      cache_borrow, 0u, &keys, &values, &lengths, &keep,
      &kernel_error) == 0);
  void *cache_alias = keys->data;
  CHECK(et_a2_kv_cache_read_borrow_end_v1(
      &cache_borrow, &kernel_error) == 0);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, cache_alias) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  check_text_pending(&prepared, 0u);
  OK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw));
  CHECK(raw.bytes[0] == 0xa1);
  check_text_pending(&prepared, 1u);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_INVALID_STATE);
  check_text_pending(&prepared, 1u);
  discard_prepared(&prepared);
}

static void g1_text_failures_and_success(
    et_g3c4_model_owner_internal *owner) {
  prepared_output prepared = make_prepared(owner, 1);
  copy_ids(&prepared);
  raw_carrier raw = {1, {0}};

  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, NULL) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output,
      (void *)(uintptr_t)(UINTPTR_MAX - 7u)) == ET_G3C4_INVALID_ARGUMENT);
  raw.length = -1;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_SHAPE_MISMATCH);
  raw.length = 2;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_SHAPE_MISMATCH);
  raw.length = 1;

  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, prepared.context) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, prepared.output) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, prepared.context->owner) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output,
      prepared.context->pins.views[0].data) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  const int64_t alias_tokens[2] = {31, 37};
  et_g3c4_input_internal *alias_input = create_prompt(alias_tokens, 2);
  et_i64_tensor_borrow *alias_borrow = NULL;
  const et_kernel_tensor_view_v1 *alias_view = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(
      alias_input->tensor, &alias_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(
      alias_borrow, &alias_view, &error) == 0);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, alias_view->data) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);
  CHECK(et_i64_tensor_borrow_end_v1(&alias_borrow, &error) == 0);
  OK(et_g3c4_private_tensor_release_v1(alias_input));

  et_i64_tensor_borrow *ids_borrow = NULL;
  const et_kernel_tensor_view_v1 *ids_view = NULL;
  CHECK(et_i64_tensor_borrow_begin_v1(
      prepared.output->ids, &ids_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(
      ids_borrow, &ids_view, &error) == 0);
  void *ids_alias = ids_view->data;
  CHECK(et_i64_tensor_borrow_end_v1(&ids_borrow, &error) == 0);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, ids_alias) ==
      ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALIAS);

  unsigned char saved_parameter =
      ((unsigned char *)prepared.context->pins.views[0].data)[0];
  ((unsigned char *)prepared.context->pins.views[0].data)[0] ^= 1u;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_G3C4_CODE_STALE_BINDING);
  ((unsigned char *)prepared.context->pins.views[0].data)[0] =
      saved_parameter;
  check_text_pending(&prepared, 0u);

  raw.bytes[0] = (unsigned char)prepared.token ^ 1u;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_SHAPE);
  check_text_pending(&prepared, 0u);

  et_i64_tensor_test_fail_alloc_after_v1(0u);
  raw.bytes[0] = (unsigned char)prepared.token;
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
  et_i64_tensor_test_reset_allocator_v1();
  check_text_pending(&prepared, 0u);

  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  CHECK(et_i64_tensor_borrow_begin_v1(
      prepared.output->ids, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  check_text_pending(&prepared, 0u);

  OK(et_g3c4_private_output_accept_text_v1(
      prepared.context, prepared.output, &raw));
  check_text_pending(&prepared, 1u);
  discard_prepared(&prepared);
}

static void text_ownership_cut(et_g3c4_model_owner_internal *owner) {
  prepared_output prepared = make_prepared(owner, 1);
  copy_ids(&prepared);
  et_g3c4_model_owner_internal *foreign_owner = create_foreign_owner();
  et_g3c4_context_internal *foreign_context = create_generator(foreign_owner);
  OK(et_g3c4_private_call_acquire_v1(foreign_context, 2, 1));
  et_g3c4_output_internal *foreign_output =
      et_g3c4_private_output_reserve_v1(foreign_context, 1);
  CHECK(foreign_output != NULL);
  raw_carrier raw = {1, {(unsigned char)prepared.token}};
  CHECK(et_g3c4_private_output_accept_text_v1(
      foreign_context, prepared.output, &raw) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_output_accept_text_v1(
      prepared.context, foreign_output, &raw) == ET_G3C4_INVALID_STATE);
  check_text_pending(&prepared, 0u);
  OK(et_g3c4_private_call_abort_v1(foreign_context));
  OK(et_g3c4_private_generator_close_v1(foreign_context));
  discard_prepared(&prepared);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  g0_text(owner);
  g1_text_failures_and_success(owner);
  text_ownership_cut(owner);
  printf("G3-C4 output text PASS: checks=%zu routes=2 "
         "readiness-cuts=3 malformed-cuts=4 alias-cuts=7 "
         "borrow-cuts=2 ownership-cuts=2 mutation-cuts=1\n", checks);
  return 0;
}
