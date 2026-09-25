#define ET_G3C4_MANUAL_TAIL_PRIVATE 1
#include "test_manual_tail_predecessor.inc"

static int record_tail, fail_tail, record_whole;
static size_t tail_dispatches, whole_dispatches;
static et_g3c4_context_internal *tail_context;
static float tail_output[512], whole_outputs[7][512];
static size_t whole_width;

static float *tail_slot(et_g3c4_manual_frame_internal *frame, int64_t ordinal) {
  switch (ordinal) {
    case 14: return frame->n2;
    case 15: return frame->fu;
    case 16: return frame->fg;
    case 17: return frame->fd;
    case 18: return frame->y;
    case 19: return frame->nf;
    default: return frame->z;
  }
}

static size_t tail_width(int64_t ordinal) {
  return ordinal == 15 || ordinal == 16 ? 8u :
         (ordinal == 20 ? 256u : 4u);
}

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  int32_t status;
  if (record_tail) {
    static const char *const t1_capability[7] = {
      "g3n.layer-norm-forward", "g3n.linear-forward", "kernel.activation",
      "g3n.linear-forward", "g3n.residual-forward",
      "g3n.layer-norm-forward", "g3n.linear-forward"};
    static const char *const t1_operation[7] = {
      "g3n.layer-norm.forward", "g3n.linear.forward-no-bias", "gelu.forward",
      "g3n.linear.forward-no-bias", "g3n.residual.forward",
      "g3n.layer-norm.forward", "g3n.linear.forward-no-bias"};
    static const char *const t2_capability[7] = {
      "kernel.norm", "n3k.linear", "n3k.gelu", "n3k.linear",
      "n3k.residual", "kernel.norm", "n3k.linear"};
    static const char *const t2_operation[7] = {
      "layer-norm.forward", "n3k.linear.forward-no-bias", "n3k.gelu.forward",
      "n3k.linear.forward-no-bias", "n3k.residual.forward",
      "layer-norm.forward", "n3k.linear.forward-no-bias"};
    const et_g3c4_manual_frame_internal *frame = tail_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const int64_t ordinal = frame->next_ordinal;
    const size_t index = (size_t)(ordinal - 14);
    const int t2 = frame->input_length == 2;
    const size_t width = tail_width(ordinal);
    tail_dispatches++;
    CHECK(tail_dispatches == 1u && ordinal >= 14 && ordinal <= 20);
    CHECK(strcmp(call->capability,
                 (t2 ? t2_capability : t1_capability)[index]) == 0);
    CHECK(strcmp(call->request->operation,
                 (t2 ? t2_operation : t1_operation)[index]) == 0);
    CHECK(call->request->deterministic == 1u &&
          call->request->rank == (ordinal == 15 || ordinal == 17 ||
                                 ordinal == 20 ? 4u : 3u) &&
          call->request->shape[0] == 1u &&
          call->request->shape[1] == (uint64_t)frame->input_length);
    CHECK(call->output_count == 1u && outputs[0].data != tail_slot(
              tail_context->manual_frame, ordinal) &&
          outputs[0].rank == 3u && outputs[0].shape[0] == 1u &&
          outputs[0].shape[1] == (uint64_t)frame->input_length &&
          outputs[0].shape[2] == width &&
          outputs[0].byte_length ==
              (size_t)frame->input_length * width * sizeof(float));
    if (ordinal == 14 || ordinal == 19) {
      uint32_t epsilon_bits;
      CHECK(call->input_count == 4u &&
            inputs[0].data == (ordinal == 14 ? frame->r : frame->y) &&
            inputs[1].data == tail_context->pins.views[
                ordinal == 14 ? 9u : 12u].data &&
            inputs[2].data == tail_context->pins.views[
                ordinal == 14 ? 8u : 11u].data &&
            inputs[3].rank == 0u);
      memcpy(&epsilon_bits, inputs[3].data, sizeof(epsilon_bits));
      CHECK(epsilon_bits == UINT32_C(0x3727c5ac));
      CHECK(call->request->shape[2] == 4u);
    } else if (ordinal == 15 || ordinal == 17 || ordinal == 20) {
      const size_t weight = ordinal == 15 ? 5u :
                            (ordinal == 17 ? 4u : 10u);
      CHECK(call->input_count == 2u &&
            inputs[0].data == (ordinal == 15 ? frame->n2 :
                              (ordinal == 17 ? frame->fg : frame->nf)) &&
            inputs[1].data == tail_context->pins.views[weight].data &&
            call->request->shape[2] == (ordinal == 17 ? 8u : 4u) &&
            call->request->shape[3] == width);
    } else if (ordinal == 16) {
      CHECK(call->input_count == 1u && inputs[0].data == frame->fu &&
            call->request->shape[2] == 8u);
    } else {
      CHECK(call->input_count == 2u && inputs[0].data == frame->r &&
            inputs[1].data == frame->fd && call->request->shape[2] == 4u);
    }
    if (fail_tail) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.manual-tail-cut");
      strcpy(error->message, "injected tail-role failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  status = et_g3c4_manual_tail_base_dispatch(runtime, call, error);
  if (record_tail && status == 0)
    memcpy(tail_output, ((const et_kernel_tensor_view_v1 *)call->outputs)[0].data,
           (size_t)tail_context->manual_frame->input_length *
               tail_width(tail_context->manual_frame->next_ordinal) *
               sizeof(float));
  if (record_whole && status == 0) {
    CHECK(whole_dispatches < 21u);
    if (whole_dispatches >= 14u) {
      const size_t ordinal = whole_dispatches;
      const size_t bytes = whole_width * tail_width((int64_t)ordinal) *
                           sizeof(float);
      CHECK(((const et_kernel_tensor_view_v1 *)call->outputs)[0].byte_length ==
            bytes);
      memcpy(whole_outputs[ordinal - 14u],
             ((const et_kernel_tensor_view_v1 *)call->outputs)[0].data, bytes);
    }
    whole_dispatches++;
  }
  return status;
}

static void whole_reference(et_g3c4_model_owner_internal *owner,
                            int64_t frame_kind, int64_t length) {
  const int64_t ids[2] = {41, 43}, prefix[1] = {47};
  float logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  whole_dispatches = 0u;
  whole_width = (size_t)length;
  if (frame_kind == 1) {
    OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
    record_whole = 1;
    if (length == 1)
      OK(et_g3c4_private_prefill1_v1(context, ids[0], logits));
    else
      OK(et_g3c4_private_prefill2_v1(context, ids, logits));
    record_whole = 0;
    OK(et_g3c4_private_call_finish_v1(context));
  } else {
    float sample_logits[1024] = {0};
    int64_t speculative = -1;
    et_g3c4_input_internal *input = create_prompt(prefix, 1);
    et_g3c4_output_internal *output;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    output = et_g3c4_private_output_reserve_v1(context, 1);
    CHECK(output != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
    OK(et_g3c4_private_call_abort_v1(context));
    OK(et_g3c4_private_tensor_release_v1(input));
    sample_logits[3u * 256u + (size_t)ids[0]] = 10.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    OK(et_g3c4_private_token_frame_begin_v1(
        context, sample_logits, &speculative));
    CHECK(speculative == ids[0]);
    record_whole = 1;
    OK(et_g3c4_private_token_forward_v1(context, speculative, logits));
    record_whole = 0;
    OK(et_g3c4_private_call_abort_v1(context));
  }
  CHECK(whole_dispatches == 21u);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void tail_case(et_g3c4_model_owner_internal *owner,
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
  unsigned char prefix_before[offsetof(et_g3c4_manual_frame_internal, n2) -
                              offsetof(et_g3c4_manual_frame_internal, et)];
  int64_t rng[4];
  whole_reference(owner, frame_kind, length);
  CHECK(et_g3c4_manual_tail_run(context, 14) == ET_G3C4_INVALID_STATE);
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
  OK(et_g3c4_private_frame_begin_v1(context, input, frame_kind));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_manual_role0_run(context));
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++)
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
  OK(et_g3c4_manual_a2_run(context));
  OK(et_g3c4_manual_at_run(context));
  OK(et_g3c4_manual_ao_run(context));
  OK(et_g3c4_manual_r_run(context));
  frame = context->manual_frame;
  CHECK(frame->next_ordinal == 14 && frame->a2_candidate != NULL &&
        frame->a2_transaction != NULL);
  candidate = frame->a2_candidate;
  transaction = frame->a2_transaction;
  memcpy(prefix_before, (const unsigned char *)frame +
         offsetof(et_g3c4_manual_frame_internal, et), sizeof(prefix_before));
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  snapshot_at_candidate(frame, &staged_before);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  tail_context = context;
  CHECK(et_g3c4_manual_tail_run(context, 13) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_manual_tail_run(context, 21) == ET_G3C4_INVALID_ARGUMENT);
  for (int64_t ordinal = 14; ordinal <= 20; ordinal++) {
    float *slot = tail_slot(frame, ordinal);
    const size_t width = tail_width(ordinal);
    const size_t bytes = (size_t)length * width * sizeof(float);
    CHECK(frame->next_ordinal == ordinal);
    if (ordinal < 20)
      CHECK(et_g3c4_manual_tail_run(context, ordinal + 1) ==
            ET_G3C4_INVALID_STATE);
    tail_dispatches = 0u;
    record_tail = fail_tail = 1;
    CHECK(et_g3c4_manual_tail_run(context, ordinal) != 0);
    record_tail = fail_tail = 0;
    CHECK(tail_dispatches == 1u && frame->next_ordinal == ordinal &&
          frame->a2_candidate == candidate &&
          frame->a2_transaction == transaction);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1 &&
          et_g3c4_private_last_error_code_v1() ==
              ET_KERNEL_CODE_PROVIDER_REJECTED);
    for (size_t index = 0u; index < width * 2u; index++) CHECK(slot[index] == 0.0f);
    CHECK(memcmp(prefix_before, (const unsigned char *)frame +
          offsetof(et_g3c4_manual_frame_internal, et),
          sizeof(prefix_before)) == 0);
    snapshot_at_candidate(frame, &staged_after);
    CHECK(memcmp(&staged_before, &staged_after, sizeof(staged_before)) == 0);
    check_logits(pending, context);
    tail_dispatches = 0u;
    record_tail = 1;
    OK(et_g3c4_manual_tail_run(context, ordinal));
    record_tail = 0;
    CHECK(tail_dispatches == 1u && frame->next_ordinal == ordinal + 1);
    CHECK(memcmp(slot, tail_output, bytes) == 0);
    CHECK(memcmp(slot, whole_outputs[ordinal - 14], bytes) == 0);
    for (size_t index = 0u; index < (size_t)length * width; index++)
      CHECK(isfinite(slot[index]));
    if (length == 1)
      for (size_t index = width; index < width * 2u; index++)
        CHECK(slot[index] == 0.0f);
  }
  CHECK(frame->next_ordinal == 21);
  CHECK(et_g3c4_manual_tail_run(context, 20) == ET_G3C4_INVALID_STATE);
  CHECK(memcmp(prefix_before, (const unsigned char *)frame +
        offsetof(et_g3c4_manual_frame_internal, et), sizeof(prefix_before)) == 0);
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

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  tail_case(owner, 1, 1);
  tail_case(owner, 1, 2);
  tail_case(owner, 2, 1);
  printf("G3-C4 internal manual tail PASS: checks=%zu routes=3 cuts=21 ordinals=14-20\n",
         checks);
  return 0;
}
