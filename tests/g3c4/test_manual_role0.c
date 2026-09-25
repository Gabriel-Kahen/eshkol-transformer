#define ET_G3C4_MANUAL_ROLE0_PRIVATE 1
#define ET_G3C4_MANUAL_FRAME_BEGIN_TEST_MAIN \
  et_g3c4_manual_frame_begin_predecessor_main
#define ET_G3C4_OUTPUT_PREPARE_FINAL_DISPATCH et_g3c4_role0_base_dispatch
#include "test_manual_frame_begin.c"
#undef ET_G3C4_OUTPUT_PREPARE_FINAL_DISPATCH
#undef ET_G3C4_MANUAL_FRAME_BEGIN_TEST_MAIN

static int record_role0;
static int fail_role0;
static size_t role0_dispatches;
static et_g3c4_context_internal *role0_context;

#ifndef ET_G3C4_MANUAL_ROLE0_FINAL_DISPATCH
#define ET_G3C4_MANUAL_ROLE0_FINAL_DISPATCH __wrap_et_kernel_runtime_dispatch
#endif
int32_t ET_G3C4_MANUAL_ROLE0_FINAL_DISPATCH(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_role0) {
    const et_g3c4_manual_frame_internal *frame = role0_context->manual_frame;
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    const int t2 = frame->input_length == 2;
    role0_dispatches++;
    CHECK(role0_dispatches == 1u);
    CHECK(strcmp(call->capability, t2 ? "n3k.embedding-forward" :
                                        "g3n.embedding-forward") == 0);
    CHECK(strcmp(call->request->operation,
                 t2 ? "n3k.embedding.forward" :
                      "g3n.embedding.forward") == 0);
    CHECK(call->request->rank == 4u && call->request->deterministic == 1u);
    CHECK(call->request->shape[0] == 1u &&
          call->request->shape[1] == (uint64_t)frame->input_length &&
          call->request->shape[2] == 256u &&
          call->request->shape[3] == 4u);
    CHECK(call->input_count == 2u && call->output_count == 1u);
    CHECK(inputs[0].data == frame->input_ids);
    CHECK(inputs[0].rank == 2u &&
          inputs[0].shape[1] == (uint64_t)frame->input_length);
    CHECK(inputs[1].data == role0_context->pins.views[10].data);
    CHECK(outputs[0].data != frame->et && outputs[0].rank == 3u &&
          outputs[0].shape[1] == (uint64_t)frame->input_length &&
          outputs[0].shape[2] == 4u);
    if (fail_role0) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.manual-role0-cut");
      strcpy(error->message, "injected role0 failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return et_g3c4_role0_base_dispatch(runtime, call, error);
}

static void role0_run(
    et_g3c4_context_internal *context, int64_t length,
    const int64_t ids[2]) {
  cache_snapshot before, after;
  int64_t rng[4];
  et_g3c4_manual_frame_internal *frame = context->manual_frame;
  const float *table = (const float *)context->pins.views[10].data;
  snapshot_cache(context->cache, &before);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  role0_context = context;
  role0_dispatches = 0u;
  record_role0 = 1;
  OK(et_g3c4_manual_role0_run(context));
  record_role0 = 0;
  CHECK(role0_dispatches == 1u && frame->next_ordinal == 1);
  for (int64_t position = 0; position < length; position++)
    CHECK(memcmp(frame->et + 4 * position, table + 4 * ids[position],
                 4u * sizeof(float)) == 0);
  if (length == 1)
    for (size_t index = 4u; index < 8u; index++) CHECK(frame->et[index] == 0.0f);
  CHECK(et_g3c4_manual_role0_run(context) == ET_G3C4_INVALID_STATE);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&before, &after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
}

static void prefill_role0(
    et_g3c4_model_owner_internal *owner, int64_t length) {
  const int64_t ids[2] = {41, 43};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, length);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_logits_reserve_v1(context) != NULL);
  OK(et_g3c4_private_frame_begin_v1(context, input, 1));
  OK(et_g3c4_private_tensor_release_v1(input));
  role0_run(context, length, ids);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(context->manual_frame == NULL);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void dispatch_cut(et_g3c4_model_owner_internal *owner) {
  const int64_t ids[2] = {47, 49};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, 2);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_logits_reserve_v1(context) != NULL);
  OK(et_g3c4_private_frame_begin_v1(context, input, 1));
  role0_context = context;
  role0_dispatches = 0u;
  fail_role0 = record_role0 = 1;
  CHECK(et_g3c4_manual_role0_run(context) != 0);
  record_role0 = fail_role0 = 0;
  CHECK(role0_dispatches == 1u);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(context->manual_frame->next_ordinal == 0);
  for (size_t index = 0u; index < 8u; index++)
    CHECK(context->manual_frame->et[index] == 0.0f);
  role0_run(context, 2, ids);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void decode_role0(et_g3c4_model_owner_internal *owner) {
  const int64_t prefix[1] = {41}, ids[2] = {53, 0};
  float numeric[256];
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(prefix, 1);
  et_g3c4_input_internal *decode = create_prompt(ids, 1);
  et_g3c4_output_internal *output;
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  output = et_g3c4_private_output_reserve_v1(context, 1);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, numeric));
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  CHECK(et_g3c4_private_logits_reserve_v1(context) != NULL);
  OK(et_g3c4_private_frame_begin_v1(context, decode, 2));
  role0_run(context, 1, ids);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_tensor_release_v1(decode));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void malformed_role0(et_g3c4_model_owner_internal *owner) {
  const int64_t ids[1] = {59};
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_input_internal *input = create_prompt(ids, 1);
  et_g3c4_logits_internal *logits;
  CHECK(et_g3c4_manual_role0_run(&checks) == ET_G3C4_INVALID_ARGUMENT);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_manual_role0_run(context) == ET_G3C4_INVALID_STATE);
  logits = et_g3c4_private_logits_reserve_v1(context);
  CHECK(logits != NULL);
  CHECK(et_g3c4_manual_role0_run(context) == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_frame_begin_v1(context, input, 1));
  OK(et_g3c4_private_tensor_release_v1(logits));
  CHECK(et_g3c4_manual_role0_run(context) == ET_G3C4_INVALID_STATE);
  CHECK(context->manual_frame->next_ordinal == 0);
  for (size_t index = 0u; index < 8u; index++)
    CHECK(context->manual_frame->et[index] == 0.0f);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

#ifndef ET_G3C4_MANUAL_ROLE0_TEST_MAIN
#define ET_G3C4_MANUAL_ROLE0_TEST_MAIN main
#endif
int ET_G3C4_MANUAL_ROLE0_TEST_MAIN(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  prefill_role0(owner, 1);
  prefill_role0(owner, 2);
  dispatch_cut(owner);
  decode_role0(owner);
  malformed_role0(owner);
  printf("G3-C4 internal manual role0 PASS: checks=%zu routes=3 failure-retry=1\n", checks);
  return 0;
}
