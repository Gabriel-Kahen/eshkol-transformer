#include "output_envelope_native.c"

static et_i64_tensor_borrow *protected_ids_borrow;
static et_a2_kv_cache_read_borrow *protected_cache_borrow;
static et_a2_kv_cache_transaction_view *protected_transaction_view;

int64_t et_g3c4_p2g2_protected_test_stage_v1(void *context_candidate) {
  const int64_t tokens[2] = {0, 255};
  float prefill_logits[256], next_logits[256];
  int64_t selected = -1;
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_input_internal *input;
  int64_t status;
  if (context == NULL || context->call_kind != 2 || context->budget != 2)
    return -1;
  input = create_prompt(tokens, 2);
  if (input == NULL) return -1;
  status = et_g3c4_private_prompt_prefill_v1(
      context, input, prefill_logits);
  if (status == 0)
    status = et_g3c4_private_token_frame_begin_last_v1(
        context, prefill_logits, &selected);
  if (status == 0)
    status = et_g3c4_private_token_forward_v1(
        context, selected, next_logits);
  if (et_g3c4_private_tensor_release_v1(input) != 0) abort();
  return status;
}

int64_t et_g3c4_p2g2_protected_test_ids_lease_v1(
    void *output_candidate, int64_t active) {
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  et_i64_tensor_error error;
  if (output == NULL || output->ids == NULL) return -1;
  if (active == 1 && protected_ids_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        output->ids, &protected_ids_borrow, &error);
  if (active == 0 && protected_ids_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&protected_ids_borrow, &error);
  return -1;
}

int64_t et_g3c4_p2g2_protected_test_cache_lease_v1(
    void *context_candidate, int64_t active) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_kernel_error error;
  if (context == NULL) return -1;
  if (active == 1 && protected_cache_borrow == NULL &&
      protected_transaction_view == NULL) {
    if (context->token_frame_transaction != NULL)
      return et_a2_kv_cache_transaction_view_begin_v1(
          context->token_frame_transaction, 0u,
          &protected_transaction_view, &error);
    return et_a2_kv_cache_read_borrow_begin_v1(
        context->cache, &protected_cache_borrow, &error);
  }
  if (active == 0 && protected_transaction_view != NULL)
    return et_a2_kv_cache_transaction_view_end_v1(
        &protected_transaction_view, &error);
  if (active == 0 && protected_cache_borrow != NULL)
    return et_a2_kv_cache_read_borrow_end_v1(
        &protected_cache_borrow, &error);
  return -1;
}
