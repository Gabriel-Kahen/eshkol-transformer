#define ET_G3C4_MANUAL_AT_PRIVATE 1
#include "test_manual_at_predecessor.inc"

static int record_at, fail_at;
static size_t at_dispatches;
static et_g3c4_context_internal *at_context;

typedef struct at_candidate_snapshot {
  float keys[8], values[8];
  int64_t length;
  uint8_t keep[2];
} at_candidate_snapshot;

static void snapshot_at_candidate(et_g3c4_manual_frame_internal *frame,
                                  at_candidate_snapshot *snapshot) {
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  memset(snapshot, 0, sizeof(*snapshot));
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      frame->a2_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  CHECK(keys->rank == 4u && keys->shape[2] == 2u &&
        values->rank == 4u && values->shape[2] == 2u);
  memcpy(snapshot->keys, keys->data, sizeof(snapshot->keys));
  memcpy(snapshot->values, values->data, sizeof(snapshot->values));
  snapshot->length = ((const int64_t *)lengths->data)[0];
  memcpy(snapshot->keep, keep->data, sizeof(snapshot->keep));
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
}

int32_t et_g3c4_manual_ao_base_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_at) {
    const et_g3c4_manual_frame_internal *frame = at_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const int t2 = frame->input_length == 2;
    at_dispatches++;
    CHECK(at_dispatches == 1u && frame->next_ordinal == 11);
    CHECK(strcmp(call->capability,
                 t2 ? "n3k.head-layout" : "g3n.head-layout-forward") == 0);
    CHECK(strcmp(call->request->operation,
                 t2 ? "n3k.heads.merge.forward" :
                      "g3n.heads.merge.forward") == 0);
    CHECK(call->request->deterministic == 1u && call->request->rank == 4u &&
          call->request->shape[0] == 1u &&
          call->request->shape[1] == (uint64_t)frame->input_length &&
          call->request->shape[2] == 2u && call->request->shape[3] == 2u);
    CHECK(call->input_count == 1u && call->output_count == 1u &&
          inputs[0].data == frame->ah && inputs[0].rank == 4u &&
          inputs[0].shape[0] == 1u && inputs[0].shape[1] == 2u &&
          inputs[0].shape[2] == (uint64_t)frame->input_length &&
          inputs[0].shape[3] == 2u &&
          inputs[0].byte_length == (size_t)frame->input_length * 4u * sizeof(float));
    CHECK(outputs[0].data != frame->at && outputs[0].rank == 3u &&
          outputs[0].shape[0] == 1u &&
          outputs[0].shape[1] == (uint64_t)frame->input_length &&
          outputs[0].shape[2] == 4u &&
          outputs[0].byte_length == (size_t)frame->input_length * 4u * sizeof(float));
    if (fail_at) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.manual-at-cut");
      strcpy(error->message, "injected attention merge failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return et_g3c4_manual_at_base_dispatch(runtime, call, error);
}

static void at_case(et_g3c4_model_owner_internal *owner,
                    int64_t frame_kind, int64_t length) {
  const int64_t ids[2] = {41, 43}, prefix[1] = {47};
  float logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_input_internal *prefix_input = NULL;
  et_g3c4_logits_internal *pending;
  et_g3c4_output_internal *output;
  et_g3c4_manual_frame_internal *frame;
  et_a2_kv_cache *candidate;
  et_a2_kv_cache_transaction *transaction;
  et_f32_tensor_borrow *logits_borrow = NULL;
  et_f32_tensor_error f32_error;
  cache_snapshot before, after;
  binding_snapshot binding, binding_after;
  at_candidate_snapshot staged_before, staged_after;
  int64_t rng[4];
  CHECK(et_g3c4_manual_at_run(context) == ET_G3C4_INVALID_STATE);
  if (frame_kind == 2) {
    prefix_input = create_prompt(prefix, 1);
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    output = et_g3c4_private_output_reserve_v1(context, 1);
    CHECK(output != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, prefix_input, logits));
    OK(et_g3c4_private_call_abort_v1(context));
  }
  OK(et_g3c4_private_call_acquire_v1(context, frame_kind - 1, 0));
  pending = et_g3c4_private_logits_reserve_v1(context);
  CHECK(pending != NULL);
  CHECK(et_g3c4_manual_at_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, frame_kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  CHECK(et_g3c4_manual_at_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_manual_role0_run(context));
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++)
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
  CHECK(et_g3c4_manual_at_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_manual_a2_run(context));
  frame = context->manual_frame;
  CHECK(frame->next_ordinal == 11 && frame->a2_candidate != NULL &&
        frame->a2_transaction != NULL);
  candidate = frame->a2_candidate;
  transaction = frame->a2_transaction;
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  snapshot_at_candidate(frame, &staged_before);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  at_context = context;
  at_dispatches = 0u;
  record_at = fail_at = 1;
  CHECK(et_g3c4_manual_at_run(context) != 0);
  record_at = fail_at = 0;
  CHECK(at_dispatches == 1u && frame->next_ordinal == 11 &&
        frame->a2_candidate == candidate && frame->a2_transaction == transaction);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
        et_g3c4_private_last_error_code_v1() == ET_KERNEL_CODE_PROVIDER_REJECTED);
  for (size_t i = 0u; i < 8u; i++) CHECK(frame->at[i] == 0.0f);
  snapshot_at_candidate(frame, &staged_after);
  CHECK(memcmp(&staged_before, &staged_after, sizeof(staged_before)) == 0);
  check_logits(pending, context);
  at_dispatches = 0u;
  record_at = 1;
  OK(et_g3c4_manual_at_run(context));
  record_at = 0;
  CHECK(at_dispatches == 1u && frame->next_ordinal == 12 &&
        frame->a2_candidate == candidate && frame->a2_transaction == transaction);
  for (size_t position = 0u; position < (size_t)length; position++) {
    CHECK(memcmp(frame->at + position * 4u, frame->ah + position * 2u,
                 2u * sizeof(float)) == 0);
    CHECK(memcmp(frame->at + position * 4u + 2u,
                 frame->ah + (size_t)length * 2u + position * 2u,
                 2u * sizeof(float)) == 0);
  }
  if (length == 1)
    for (size_t i = 4u; i < 8u; i++) CHECK(frame->at[i] == 0.0f);
  CHECK(et_g3c4_manual_at_run(context) == ET_G3C4_INVALID_STATE);
  snapshot_at_candidate(frame, &staged_after);
  CHECK(memcmp(&staged_before, &staged_after, sizeof(staged_before)) == 0);
  snapshot_cache(context->cache, &after);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&before, &after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  check_logits(pending, context);
  OK(et_f32_tensor_borrow_begin_v1(pending->tensor, &logits_borrow,
                                    &f32_error));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(context->manual_frame == frame && frame->a2_candidate == candidate &&
        frame->a2_transaction == transaction);
  OK(et_f32_tensor_borrow_end_v1(&logits_borrow, &f32_error));
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  if (prefix_input != NULL) OK(et_g3c4_private_tensor_release_v1(prefix_input));
  OK(et_g3c4_private_generator_close_v1(context));
}

int et_g3c4_manual_at_predecessor_main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  at_case(owner, 1, 1);
  at_case(owner, 1, 2);
  at_case(owner, 2, 1);
  printf("G3-C4 internal manual AT PASS: checks=%zu routes=3 ordinal=11\n", checks);
  return 0;
}
