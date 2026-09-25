#define ET_G3C4_MANUAL_A2_PRIVATE 1
#include "test_manual_a2_predecessor.inc"

static int record_a2, fail_a2;
static size_t a2_dispatches;
static size_t a2_allocation_cuts;
static et_g3c4_context_internal *a2_context;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_a2) {
    const et_g3c4_manual_frame_internal *frame = a2_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const uint8_t *mask = inputs[5].data;
    const int64_t *positions = inputs[3].data;
    const int64_t *keys = inputs[4].data;
    const int64_t t = frame->input_length;
    a2_dispatches++;
    CHECK(a2_dispatches == 1u && frame->next_ordinal == 10);
    CHECK(strcmp(call->capability, "kernel.causal-attention") == 0);
    CHECK(strcmp(call->request->operation, "causal-attention.forward") == 0);
    CHECK(call->request->rank == 6u && call->input_count == 6u &&
          call->output_count == 1u && call->request->deterministic == 1u);
    CHECK(call->request->shape[0] == 1u && call->request->shape[1] == 2u &&
          call->request->shape[2] == 2u &&
          call->request->shape[3] == (uint64_t)t &&
          call->request->shape[4] == 2u && call->request->shape[5] == 2u);
    CHECK(inputs[0].data == frame->qh && inputs[0].rank == 4u &&
          inputs[0].shape[2] == (uint64_t)t);
    CHECK(inputs[1].rank == 4u && inputs[1].shape[2] == 2u &&
          inputs[1].byte_length == 8u * sizeof(float));
    CHECK(inputs[2].rank == 4u && inputs[2].shape[2] == 2u &&
          inputs[2].data != inputs[1].data);
    CHECK(inputs[3].rank == 2u && inputs[3].shape[1] == (uint64_t)t &&
          inputs[4].rank == 2u && inputs[4].shape[1] == 2u &&
          keys[0] == 0 && keys[1] == 1);
    CHECK(inputs[5].rank == 3u && inputs[5].shape[1] == (uint64_t)t &&
          inputs[5].shape[2] == 2u && outputs[0].data != frame->ah);
    CHECK(positions[0] == (frame->frame_kind == 2 ? 1 : 0));
    CHECK(mask[0] == 1u && mask[1] == (frame->frame_kind == 2));
    if (t == 2) CHECK(positions[1] == 1 && mask[2] == 1u && mask[3] == 1u);
    if (fail_a2) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.manual-a2-cut");
      strcpy(error->message, "injected A2 failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return et_g3c4_manual_a2_base_dispatch(runtime, call, error);
}

static void a2_case(et_g3c4_model_owner_internal *owner,
                    int64_t frame_kind, int64_t length) {
  const int64_t ids[2] = {41, 43}, prefix[1] = {47};
  float logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_input_internal *prefix_input = NULL;
  et_g3c4_logits_internal *pending;
  et_g3c4_output_internal *output;
  et_g3c4_manual_frame_internal *frame;
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL;
  const et_kernel_tensor_view_v1 *effective = NULL, *keep = NULL;
  et_f32_tensor_borrow *logits_borrow = NULL;
  et_f32_tensor_error f32_error;
  et_kernel_error error;
  cache_snapshot before, after;
  binding_snapshot binding, binding_after;
  int64_t rng[4];
  CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INVALID_STATE);
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
  CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, frame_kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_manual_role0_run(context));
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++)
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
  frame = context->manual_frame;
  CHECK(frame->next_ordinal == 10 && frame->a2_candidate == NULL &&
        frame->a2_transaction == NULL);
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  a2_context = context;
  a2_dispatches = 0u;
  record_a2 = fail_a2 = 1;
  CHECK(et_g3c4_manual_a2_run(context) != 0);
  record_a2 = fail_a2 = 0;
  CHECK(a2_dispatches == 1u && frame->next_ordinal == 10 &&
        frame->a2_candidate == NULL && frame->a2_transaction == NULL);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
        et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_PROVIDER_REJECTED);
  for (size_t i = 0u; i < 8u; i++) CHECK(frame->ah[i] == 0.0f);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  for (size_t cut = 0u; cut < 64u; cut++) {
    int64_t status;
    a2_dispatches = 0u;
    record_a2 = 1;
    et_a2_kv_cache_test_fail_alloc_after_v1(cut);
    status = et_g3c4_manual_a2_run(context);
    et_a2_kv_cache_test_reset_allocator_v1();
    record_a2 = 0;
    if (status == 0) break;
    a2_allocation_cuts++;
    CHECK(a2_dispatches == 0u && frame->next_ordinal == 10 &&
          frame->a2_candidate == NULL && frame->a2_transaction == NULL);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
          et_g3c4_private_last_error_code_v1() ==
              ET_KERNEL_CODE_ALLOCATION_FAILED);
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&before, &after);
    check_logits(pending, context);
  }
  CHECK(a2_dispatches == 1u && frame->next_ordinal == 11 &&
        frame->a2_candidate != NULL && frame->a2_transaction != NULL);
  CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      frame->a2_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &k, &v, &effective, &keep, &error));
  CHECK(k->shape[2] == 2u && v->shape[2] == 2u &&
        ((const int64_t *)effective->data)[0] ==
            (frame_kind == 2 ? 2 : length));
  CHECK(((const uint8_t *)keep->data)[0] == 1u &&
        ((const uint8_t *)keep->data)[1] ==
            (frame_kind == 2 || length == 2));
  for (size_t head = 0u; head < 2u; head++) {
    for (size_t key = 0u; key < 2u; key++) {
      const size_t destination = head * 4u + key * 2u;
      const float *expected_k = NULL, *expected_v = NULL;
      const int64_t old = frame_kind == 2 ? 1 : 0;
      if ((int64_t)key < old) {
        expected_k = before.keys + head * 8u + key * 2u;
        expected_v = before.values + head * 8u + key * 2u;
      } else if ((int64_t)key < old + length) {
        const size_t source = head * (size_t)length * 2u +
                              (key - (size_t)old) * 2u;
        expected_k = frame->kh + source;
        expected_v = frame->vh + source;
      }
      if (expected_k != NULL) {
        CHECK(memcmp((const float *)k->data + destination, expected_k,
                     2u * sizeof(float)) == 0);
        CHECK(memcmp((const float *)v->data + destination, expected_v,
                     2u * sizeof(float)) == 0);
      } else {
        CHECK(((const float *)k->data)[destination] == 0.0f &&
              ((const float *)k->data)[destination + 1u] == 0.0f &&
              ((const float *)v->data)[destination] == 0.0f &&
              ((const float *)v->data)[destination + 1u] == 0.0f);
      }
    }
  }
  for (size_t i = 0u; i < (size_t)length * 4u; i++)
    CHECK(isfinite(frame->ah[i]));
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
  snapshot_cache(context->cache, &after);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&before, &after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  check_logits(pending, context);
  {
    et_a2_kv_cache_transaction *saved_transaction = frame->a2_transaction;
    et_a2_kv_cache *saved_candidate = frame->a2_candidate;
    frame->a2_transaction = NULL;
    CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
    CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
    CHECK(context->manual_frame == frame && frame->a2_candidate == saved_candidate);
    frame->a2_transaction = saved_transaction;
    frame->a2_candidate = NULL;
    CHECK(et_g3c4_manual_a2_run(context) == ET_G3C4_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
    CHECK(et_g3c4_private_call_abort_v1(context) == ET_G3C4_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_INVARIANT);
    CHECK(context->manual_frame == frame &&
          frame->a2_transaction == saved_transaction);
    frame->a2_candidate = saved_candidate;
  }
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  check_logits(pending, context);
  OK(et_f32_tensor_borrow_begin_v1(pending->tensor, &logits_borrow,
                                    &f32_error));
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(context->manual_frame == frame && frame->a2_transaction != NULL);
  OK(et_f32_tensor_borrow_end_v1(&logits_borrow, &f32_error));
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  if (prefix_input != NULL) OK(et_g3c4_private_tensor_release_v1(prefix_input));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  a2_case(owner, 1, 1);
  a2_case(owner, 1, 2);
  a2_case(owner, 2, 1);
  printf("G3-C4 internal manual A2 PASS: checks=%zu routes=3 cuts=%zu\n",
         checks, a2_allocation_cuts);
  return 0;
}
