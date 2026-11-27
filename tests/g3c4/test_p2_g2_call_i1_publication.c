#define ET_G3C4_P2_G2_CALL_I1_LEASE_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_predecessor
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

typedef enum publication_path {
  PREFIX, SECOND_FRAME, TERMINAL_FIRST_EOS, TERMINAL_SECOND_TOKEN
} publication_path;

static void snapshot_pending_cache(et_g3c4_context_internal *context,
                                   cache_snapshot *snapshot) {
  if (context->token_frame_transaction == NULL) {
    snapshot_cache(context->cache, snapshot);
    return;
  }
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  memcpy(snapshot->keys, keys->data, sizeof(snapshot->keys));
  memcpy(snapshot->values, values->data, sizeof(snapshot->values));
  snapshot->length = *(const int64_t *)lengths->data;
  memcpy(snapshot->keep, keep->data, sizeof(snapshot->keep));
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
}

static int64_t run_publication(publication_path path,
                               et_g3c4_context_internal *context,
                               et_g3c4_output_internal *output,
                               const terminal_raw *first,
                               const terminal_raw *second,
                               bridge_carrier *stage) {
  switch (path) {
    case PREFIX:
      return et_g3c4_private_p2g2_prefix_commit_v1(context, output, first);
    case SECOND_FRAME:
      return et_g3c4_private_p2g2_second_frame_carrier_v1(
          context, output, stage, 16);
    case TERMINAL_FIRST_EOS:
      return et_g3c4_private_p2g2_terminal_commit_v1(
          context, output, first, 9, NULL, 0);
    case TERMINAL_SECOND_TOKEN:
      return et_g3c4_private_p2g2_terminal_commit_v1(
          context, output, first, 9, second, 9);
  }
  abort();
}

static void reject_held_ids_substitution(et_g3c4_model_owner_internal *owner,
                                         publication_path path) {
  const int64_t prompt[2] = {7, 11};
  const uint64_t shape[1] = {2u};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  et_i64_tensor *substitute = NULL;
  et_i64_tensor_error error;
  terminal_raw first = raw_id(0), second = raw_id(0);
  bridge_carrier stage = new_carrier();
  if (path == PREFIX) {
    active_pair(owner, 0, prompt, &context, &input, &output, NULL);
    OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, &stage, 16));
    first = raw_id(context->token_frame_candidate);
  } else if (path == SECOND_FRAME) {
    start_prefix(owner, prompt, 0, -1, &context, &input, &output);
  } else {
    terminal_ready(owner, prompt, path == TERMINAL_FIRST_EOS,
                   &context, &input, &output, &first, &second);
  }

  et_i64_tensor *original = output->ids;
  OK(et_i64_tensor_create_v1(1u, shape, &substitute, &error));
  CHECK(substitute != original);
  OK(et_m3_private_i64_unborrowed_v1(substitute, &error));
  OK(et_g3c4_p2g2_call_i1_lease_begin(context, output));
  et_i64_tensor_borrow *held = context->p2g2_i1_lease.borrow;
  CHECK(held != NULL);
  cache_snapshot before, after;
  snapshot_pending_cache(context, &before);

  const bridge_carrier stage_before = stage;
  const terminal_raw first_before = first, second_before = second;
  unsigned char context_before[sizeof(*context)], output_before[sizeof(*output)];
  memcpy(context_before, context, sizeof(context_before));
  memcpy(output_before, output, sizeof(output_before));

  /* Calls must return a recoverable error before either irreversible write. */
  fail_terminal_copy = 1;
  fail_terminal_cache_commit = 1;
  CHECK(run_publication(path, context, output, &first, &second, &stage) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(memcmp(context, context_before, sizeof(context_before)) == 0);
  CHECK(memcmp(output, output_before, sizeof(output_before)) == 0);

  output->ids = substitute;
  memcpy(output_before, output, sizeof(output_before));
  CHECK(run_publication(path, context, output, &first, &second, &stage) ==
        ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_C4);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_IDENTITY);
  fail_terminal_copy = 0;
  fail_terminal_cache_commit = 0;
  CHECK(memcmp(context, context_before, sizeof(context_before)) == 0);
  CHECK(memcmp(output, output_before, sizeof(output_before)) == 0);
  CHECK(memcmp(&stage, &stage_before, sizeof(stage)) == 0);
  CHECK(memcmp(&first, &first_before, sizeof(first)) == 0);
  CHECK(memcmp(&second, &second_before, sizeof(second)) == 0);
  CHECK(context->p2g2_i1_lease.borrow == held &&
        context->owner->active == context);
  snapshot_pending_cache(context, &after);
  check_cache_snapshot_equal(&after, &before);
  int64_t ids[2] = {-1, -1};
  OK(et_i64_tensor_copy_to_v1(substitute, ids, 2u, &error));
  CHECK(ids[0] == 0 && ids[1] == 0);

  output->ids = original;
  const et_kernel_tensor_view_v1 *ids_view = NULL;
  OK(et_i64_tensor_borrow_view_v1(held, &ids_view, &error));
  CHECK(ids_view->byte_length == 16u);
  CHECK(memcmp(ids_view->data, (const int64_t[2]){0, 0}, 16u) == 0);
  CHECK(et_g3c4_private_call_abort_v1(context) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  OK(et_g3c4_p2g2_call_i1_lease_end(context, output));
  CHECK(context->p2g2_i1_lease.borrow == NULL);
  OK(et_i64_tensor_destroy_v1(&substitute, &error));
  close_pair(context, input, output);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  reject_held_ids_substitution(owner, PREFIX);
  reject_held_ids_substitution(owner, SECOND_FRAME);
  reject_held_ids_substitution(owner, TERMINAL_FIRST_EOS);
  reject_held_ids_substitution(owner, TERMINAL_SECOND_TOKEN);
  printf("G3-C4 P2/G2 call I1 publication guard PASS: checks=%zu\n", checks);
  return 0;
}
