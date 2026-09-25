#define ET_G3C4_MANUAL_PRE_A2_PRIVATE 1
#define ET_G3C4_MANUAL_ROLE0_TEST_MAIN et_g3c4_manual_role0_predecessor_main
#define ET_G3C4_MANUAL_ROLE0_FINAL_DISPATCH et_g3c4_pre_a2_base_dispatch
#include "test_manual_role0.c"
#undef ET_G3C4_MANUAL_ROLE0_FINAL_DISPATCH
#undef ET_G3C4_MANUAL_ROLE0_TEST_MAIN

static int record_pre_a2, fail_pre_a2;
static int record_reference;
static size_t pre_a2_dispatches;
static size_t reference_dispatches;
static et_g3c4_context_internal *pre_a2_context;
static float pre_a2_output[8];
static float reference_outputs[10][8];
static int64_t reference_length;

static float *pre_a2_slot(et_g3c4_manual_frame_internal *frame,
                          int64_t ordinal) {
  switch (ordinal) {
    case 1: return frame->ep;
    case 2: return frame->x;
    case 3: return frame->n1;
    case 4: return frame->qt;
    case 5: return frame->kt;
    case 6: return frame->vt;
    case 7: return frame->qh;
    case 8: return frame->kh;
    default: return frame->vh;
  }
}

int32_t et_g3c4_manual_a2_base_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  int32_t status;
  if (record_pre_a2) {
    static const char *const g3n_capabilities[9] = {
      "g3n.embedding-forward", "g3n.residual-forward",
      "g3n.layer-norm-forward", "g3n.linear-forward",
      "g3n.linear-forward", "g3n.linear-forward",
      "g3n.head-layout-forward", "g3n.head-layout-forward",
      "g3n.head-layout-forward"};
    static const char *const g3n_operations[9] = {
      "g3n.embedding.forward", "g3n.residual.forward",
      "g3n.layer-norm.forward", "g3n.linear.forward-no-bias",
      "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias",
      "g3n.heads.split.forward", "g3n.heads.split.forward",
      "g3n.heads.split.forward"};
    static const char *const t2_capabilities[9] = {
      "n3k.embedding-forward", "n3k.residual", "kernel.norm",
      "n3k.linear", "n3k.linear", "n3k.linear",
      "n3k.head-layout", "n3k.head-layout", "n3k.head-layout"};
    static const char *const t2_operations[9] = {
      "n3k.embedding.forward", "n3k.residual.forward",
      "layer-norm.forward", "n3k.linear.forward-no-bias",
      "n3k.linear.forward-no-bias", "n3k.linear.forward-no-bias",
      "n3k.heads.split.forward", "n3k.heads.split.forward",
      "n3k.heads.split.forward"};
    et_g3c4_manual_frame_internal *frame = pre_a2_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const int64_t ordinal = frame->next_ordinal;
    const int t2 = frame->input_length == 2;
    const size_t rank = ordinal == 2 || ordinal == 3 ? 3u : 4u;
    pre_a2_dispatches++;
    CHECK(pre_a2_dispatches == 1u && ordinal >= 1 && ordinal <= 9);
    CHECK(strcmp(call->capability,
                 (t2 ? t2_capabilities : g3n_capabilities)[ordinal - 1]) == 0);
    CHECK(strcmp(call->request->operation,
                 (t2 ? t2_operations : g3n_operations)[ordinal - 1]) == 0);
    CHECK(call->request->rank == rank &&
          call->request->shape[0] == 1u &&
          call->request->shape[1] == (uint64_t)frame->input_length &&
          call->request->deterministic == 1u);
    CHECK(outputs[0].data != pre_a2_slot(frame, ordinal));
    CHECK(outputs[0].rank == (ordinal >= 7 ? 4u : 3u));
    CHECK(outputs[0].shape[ordinal >= 7 ? 2u : 1u] ==
          (uint64_t)frame->input_length);
    if (ordinal == 1) {
      const int64_t *positions = inputs[0].data;
      CHECK(positions[0] == (frame->frame_kind == 2 ? 1 : 0));
      if (t2) CHECK(positions[1] == 1);
      CHECK(inputs[1].data == pre_a2_context->pins.views[13].data);
      CHECK(inputs[1].byte_length == 8u * sizeof(float) &&
            inputs[1].rank == 2u && inputs[1].shape[0] == 2u &&
            inputs[1].shape[1] == 4u);
      CHECK(call->request->shape[2] == 2u && call->request->shape[3] == 4u);
    } else if (ordinal == 2) {
      CHECK(inputs[0].data == frame->et && inputs[1].data == frame->ep);
    } else if (ordinal == 3) {
      uint32_t bits;
      CHECK(inputs[0].data == frame->x);
      CHECK(inputs[1].data == pre_a2_context->pins.views[7].data);
      CHECK(inputs[2].data == pre_a2_context->pins.views[6].data);
      memcpy(&bits, inputs[3].data, sizeof(bits));
      CHECK(bits == UINT32_C(0x3727c5ac));
    } else if (ordinal <= 6) {
      CHECK(inputs[0].data == frame->n1);
      CHECK(inputs[1].data == pre_a2_context->pins.views[
          ordinal == 4 ? 2u : (ordinal == 5 ? 0u : 3u)].data);
      CHECK(call->request->shape[2] == 4u && call->request->shape[3] == 4u);
    } else {
      CHECK(inputs[0].data == pre_a2_slot(frame, ordinal - 3));
      CHECK(call->request->shape[2] == 2u && call->request->shape[3] == 2u);
      CHECK(outputs[0].shape[1] == 2u && outputs[0].shape[3] == 2u);
    }
    if (fail_pre_a2) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.pre-a2-cut");
      strcpy(error->message, "injected pre-A2 failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  status = et_g3c4_pre_a2_base_dispatch(runtime, call, error);
  if (record_reference && status == 0 && reference_dispatches < 10u)
    memcpy(reference_outputs[reference_dispatches++],
           ((const et_kernel_tensor_view_v1 *)call->outputs)[0].data,
           (size_t)reference_length * 4u * sizeof(float));
  if (record_pre_a2 && status == 0)
    memcpy(pre_a2_output,
           ((const et_kernel_tensor_view_v1 *)call->outputs)[0].data,
           (size_t)pre_a2_context->manual_frame->input_length *
               4u * sizeof(float));
  return status;
}

static void pre_a2_case(et_g3c4_model_owner_internal *owner,
                        int64_t frame_kind, int64_t length,
                        int64_t fail_ordinal) {
  const int64_t ids[2] = {41, 43}, prefix[1] = {47};
  float logits[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_context_internal *reference = NULL;
  et_g3c4_input_internal *input = create_prompt(ids, length);
  et_g3c4_input_internal *prefix_input = NULL;
  et_g3c4_logits_internal *pending;
  et_g3c4_output_internal *output;
  cache_snapshot before, after;
  binding_snapshot binding, binding_after;
  int64_t rng[4];
  if (frame_kind == 1 && fail_ordinal == 0) {
    reference = create_generator(owner);
    reference_length = length;
    reference_dispatches = 0u;
    record_reference = 1;
    OK(et_g3c4_private_call_acquire_v1(reference, 0, 0));
    if (length == 1)
      OK(et_g3c4_private_prefill1_v1(reference, ids[0], logits));
    else
      OK(et_g3c4_private_prefill2_v1(reference, ids, logits));
    record_reference = 0;
    CHECK(reference_dispatches == 10u);
    OK(et_g3c4_private_call_finish_v1(reference));
    OK(et_g3c4_private_generator_close_v1(reference));
  }
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
  if (reference != NULL)
    CHECK(memcmp(context->manual_frame->et, reference_outputs[0],
                 (size_t)length * 4u * sizeof(float)) == 0);
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  pre_a2_context = context;
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++) {
    et_g3c4_manual_frame_internal *frame = context->manual_frame;
    float *slot = pre_a2_slot(frame, ordinal);
    CHECK(et_g3c4_manual_pre_a2_run(context, ordinal + 1) ==
          (ordinal == 9 ? ET_G3C4_INVALID_ARGUMENT : ET_G3C4_INVALID_STATE));
    CHECK(frame->next_ordinal == ordinal);
    if (ordinal == fail_ordinal) {
      pre_a2_dispatches = 0u;
      fail_pre_a2 = record_pre_a2 = 1;
      CHECK(et_g3c4_manual_pre_a2_run(context, ordinal) != 0);
      fail_pre_a2 = record_pre_a2 = 0;
      CHECK(pre_a2_dispatches == 1u && frame->next_ordinal == ordinal);
      CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_PROVIDER_REJECTED);
      for (size_t index = 0u; index < 8u; index++) CHECK(slot[index] == 0.0f);
      check_logits(pending, context);
    }
    pre_a2_dispatches = 0u;
    record_pre_a2 = 1;
    OK(et_g3c4_manual_pre_a2_run(context, ordinal));
    record_pre_a2 = 0;
    CHECK(pre_a2_dispatches == 1u && frame->next_ordinal == ordinal + 1);
    CHECK(memcmp(slot, pre_a2_output,
                 (size_t)length * 4u * sizeof(float)) == 0);
    if (reference != NULL)
      CHECK(memcmp(slot, reference_outputs[ordinal],
                   (size_t)length * 4u * sizeof(float)) == 0);
    for (size_t index = 0u; index < (size_t)length * 4u; index++)
      CHECK(isfinite(slot[index]));
    if (length == 1)
      for (size_t index = 4u; index < 8u; index++) CHECK(slot[index] == 0.0f);
  }
  CHECK(context->manual_frame->next_ordinal == 10);
  CHECK(et_g3c4_manual_pre_a2_run(context, 10) == ET_G3C4_INVALID_ARGUMENT);
  snapshot_cache(context->cache, &after);
  snapshot_binding(context, &binding_after);
  check_cache_snapshot_equal(&before, &after);
  check_binding_snapshot_equal(&binding, &binding_after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  check_logits(pending, context);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  if (prefix_input != NULL) OK(et_g3c4_private_tensor_release_v1(prefix_input));
  OK(et_g3c4_private_generator_close_v1(context));
}

int et_g3c4_manual_pre_a2_predecessor_main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  pre_a2_case(owner, 1, 1, 0);
  pre_a2_case(owner, 1, 2, 0);
  pre_a2_case(owner, 2, 1, 0);
  for (int64_t ordinal = 1; ordinal <= 9; ordinal++) {
    pre_a2_case(owner, 1, 1, ordinal);
    pre_a2_case(owner, 1, 2, ordinal);
    pre_a2_case(owner, 2, 1, ordinal);
  }
  printf("G3-C4 internal manual pre-A2 PASS: checks=%zu routes=3 cuts=27 ordinals=1-9\n", checks);
  return 0;
}
