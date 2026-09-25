#define ET_G3C4_MANUAL_R_PRIVATE 1
#include "test_manual_r_predecessor.inc"

static int record_r, fail_r;
static size_t r_dispatches;
static et_g3c4_context_internal *r_context;
static float r_output[8];

int32_t et_g3c4_manual_tail_base_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  int32_t status;
  if (record_r) {
    const et_g3c4_manual_frame_internal *frame = r_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const int t2 = frame->input_length == 2;
    r_dispatches++;
    CHECK(r_dispatches == 1u && frame->next_ordinal == 13);
    CHECK(strcmp(call->capability,
                 t2 ? "n3k.residual" : "g3n.residual-forward") == 0);
    CHECK(strcmp(call->request->operation,
                 t2 ? "n3k.residual.forward" : "g3n.residual.forward") == 0);
    CHECK(call->request->deterministic == 1u && call->request->rank == 3u &&
          call->request->shape[0] == 1u &&
          call->request->shape[1] == (uint64_t)frame->input_length &&
          call->request->shape[2] == 4u);
    CHECK(call->input_count == 2u && call->output_count == 1u &&
          inputs[0].data == frame->x && inputs[1].data == frame->ao &&
          inputs[0].data != inputs[1].data);
    for (size_t index = 0u; index < 2u; index++)
      CHECK(inputs[index].rank == 3u && inputs[index].shape[0] == 1u &&
            inputs[index].shape[1] == (uint64_t)frame->input_length &&
            inputs[index].shape[2] == 4u &&
            inputs[index].byte_length ==
                (size_t)frame->input_length * 4u * sizeof(float));
    CHECK(outputs[0].data != frame->r && outputs[0].rank == 3u &&
          outputs[0].shape[0] == 1u &&
          outputs[0].shape[1] == (uint64_t)frame->input_length &&
          outputs[0].shape[2] == 4u &&
          outputs[0].byte_length ==
              (size_t)frame->input_length * 4u * sizeof(float));
    if (fail_r) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.manual-r-cut");
      strcpy(error->message, "injected attention residual failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  status = et_g3c4_manual_r_base_dispatch(runtime, call, error);
  if (record_r && status == 0)
    memcpy(r_output, ((const et_kernel_tensor_view_v1 *)call->outputs)[0].data,
           (size_t)r_context->manual_frame->input_length * 4u * sizeof(float));
  return status;
}

static void r_case(et_g3c4_model_owner_internal *owner,
                   int64_t frame_kind, int64_t length) {
  const int64_t ids[2] = {41, 43}, prefix[1] = {47};
  float logits[256], x_before[8], ao_before[8];
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
  CHECK(et_g3c4_manual_r_run(context) == ET_G3C4_INVALID_STATE);
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
  CHECK(et_g3c4_manual_r_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, frame_kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_manual_role0_run(context));
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++)
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
  OK(et_g3c4_manual_a2_run(context));
  OK(et_g3c4_manual_at_run(context));
  CHECK(et_g3c4_manual_r_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_manual_ao_run(context));
  frame = context->manual_frame;
  CHECK(frame->next_ordinal == 13 && frame->a2_candidate != NULL &&
        frame->a2_transaction != NULL);
  candidate = frame->a2_candidate;
  transaction = frame->a2_transaction;
  memcpy(x_before, frame->x, sizeof(x_before));
  memcpy(ao_before, frame->ao, sizeof(ao_before));
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  snapshot_at_candidate(frame, &staged_before);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  r_context = context;
  r_dispatches = 0u;
  record_r = fail_r = 1;
  CHECK(et_g3c4_manual_r_run(context) != 0);
  record_r = fail_r = 0;
  CHECK(r_dispatches == 1u && frame->next_ordinal == 13 &&
        frame->a2_candidate == candidate && frame->a2_transaction == transaction);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
        et_g3c4_private_last_error_code_v1() == ET_KERNEL_CODE_PROVIDER_REJECTED);
  for (size_t i = 0u; i < 8u; i++) CHECK(frame->r[i] == 0.0f);
  CHECK(memcmp(x_before, frame->x, sizeof(x_before)) == 0 &&
        memcmp(ao_before, frame->ao, sizeof(ao_before)) == 0);
  snapshot_at_candidate(frame, &staged_after);
  CHECK(memcmp(&staged_before, &staged_after, sizeof(staged_before)) == 0);
  check_logits(pending, context);
  r_dispatches = 0u;
  record_r = 1;
  OK(et_g3c4_manual_r_run(context));
  record_r = 0;
  CHECK(r_dispatches == 1u && frame->next_ordinal == 14 &&
        frame->a2_candidate == candidate && frame->a2_transaction == transaction);
  CHECK(memcmp(frame->r, r_output, (size_t)length * 4u * sizeof(float)) == 0);
  for (size_t i = 0u; i < (size_t)length * 4u; i++) {
    const float expected = x_before[i] + ao_before[i];
    CHECK(memcmp(&frame->r[i], &expected, sizeof(expected)) == 0);
  }
  if (length == 1)
    for (size_t i = 4u; i < 8u; i++) CHECK(frame->r[i] == 0.0f);
  CHECK(et_g3c4_manual_r_run(context) == ET_G3C4_INVALID_STATE);
  CHECK(memcmp(x_before, frame->x, sizeof(x_before)) == 0 &&
        memcmp(ao_before, frame->ao, sizeof(ao_before)) == 0);
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

int et_g3c4_manual_r_predecessor_main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  r_case(owner, 1, 1);
  r_case(owner, 1, 2);
  r_case(owner, 2, 1);
  printf("G3-C4 internal manual R PASS: checks=%zu routes=3 ordinal=13\n", checks);
  return 0;
}
