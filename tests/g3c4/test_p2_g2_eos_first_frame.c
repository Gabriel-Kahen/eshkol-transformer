#define ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_CARRIER_TEST_MAIN carrier_compatibility_main
#define ET_G3C4_P2_G1_TEST_DISPATCH carrier_dispatch
#include "test_p2_g2_carrier_bridge.c"

static bridge_carrier *tampered_carrier;
static et_g3c4_context_internal *tampered_context;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  int32_t status = carrier_dispatch(runtime, call, error);
  if (status == 0 && tampered_carrier != NULL &&
      tampered_context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED) {
    tampered_carrier->length = 9;
    tampered_carrier = NULL;
  }
  return status;
}

static et_g3c4_context_internal *eos_generator(
    et_g3c4_model_owner_internal *owner, int categorical, int64_t eos) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, categorical ? 1729 : 7, categorical, INT64_C(0x3f800000),
      categorical ? 17 : 256,
      categorical ? INT64_C(0x3f000000) : INT64_C(0x3f800000), 2, eos);
  CHECK(context != NULL);
  return context;
}

static void begin_eos_pair(
    et_g3c4_model_owner_internal *owner, const int64_t prompt[2],
    int categorical, int64_t eos, et_g3c4_context_internal **context,
    et_g3c4_input_internal **input, et_g3c4_output_internal **output) {
  *context = eos_generator(owner, categorical, eos);
  *input = create_prompt(prompt, 2);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(*context, *input, 2));
  OK(et_g3c4_private_call_acquire_v1(*context, 2, 2));
  *output = et_g3c4_private_output_reserve_v1(*context, 2);
  CHECK(*output != NULL);
}

static void assert_pending(const et_g3c4_output_internal *output) {
  CHECK(output->numeric_ready == 0u && output->ids_copied == 0u &&
        output->text_ready == 0u && output->length == 0 &&
        output->cache_length == 0);
}

static void eos_parity(et_g3c4_model_owner_internal *owner,
                       const int64_t prompt[2], int categorical,
                       int selected_eos) {
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  int64_t selected = -1, initial_rng[4], successor[4];
  float last[256], reference_next[256], actual_next[256];
  float actual_keys[16], actual_values[16];
  cache_snapshot prefilled, after_abort, full_prefix;
  bridge_carrier carrier = new_carrier();

  /* A separate genuine non-EOS request determines the comparison draw.
   * Never replay an EOS-rejected frame on the request under test. */
  active_pair(owner, categorical, prompt, &context, &input, &output, NULL);
  memcpy(initial_rng, context->generator_rng_words, sizeof(initial_rng));
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  snapshot_cache(context->cache, &prefilled);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &selected));
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  close_pair(context, input, output);

  begin_eos_pair(owner, prompt, categorical, selected_eos ? selected : -1,
                 &context, &input, &output);
  et_kernel_tensor_view_v1 saved_views[14];
  unsigned char parameter_bytes[1192u * sizeof(float)];
  size_t offset = 0u;
  memcpy(saved_views, context->pins.views, sizeof(saved_views));
  for (size_t i = 0u; i < 14u; i++) {
    CHECK(context->pins.views[i].byte_length <= sizeof(parameter_bytes) - offset);
    memcpy(parameter_bytes + offset, context->pins.views[i].data,
           (size_t)context->pins.views[i].byte_length);
    offset += (size_t)context->pins.views[i].byte_length;
  }
  CHECK(offset == sizeof(parameter_bytes));
  OK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 16));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(context->token_frame_position == 2 &&
        context->token_frame_candidate == selected);
  CHECK((selected == context->generator_policy[5]) == selected_eos);
  CHECK(context->p2g2_prefix_committed == 0u && context->budget == 2);
  CHECK(carrier.length == 8 && carrier.bytes[0] == (unsigned char)selected);
  for (size_t i = 1u; i < 8u; i++) CHECK(carrier.bytes[i] == 0u);
  for (size_t i = 0u; i < 8u; i++) CHECK(carrier.canary[i] == 0x5au);
  assert_pending(output);
  CHECK(memcmp(context->generator_rng_words, initial_rng, sizeof(initial_rng)) == 0);
  CHECK(memcmp(context->token_frame_successor, successor, sizeof(successor)) == 0);
  if (selected_eos)
    print_oracle_case(categorical ? "categorical" : "greedy", last,
                     context->token_frame_candidate, initial_rng,
                     context->token_frame_successor);
  memcpy(actual_next, context->p2g2_frame_next_logits, sizeof(actual_next));
  memcpy(actual_keys, context->p2g2_frame_keys, sizeof(actual_keys));
  memcpy(actual_values, context->p2g2_frame_values, sizeof(actual_values));

  /* This leaf deliberately cannot turn EOS into a non-EOS prefix/output. */
  struct { int64_t length; unsigned char raw; } raw = {1, (unsigned char)selected};
  if (selected_eos)
    CHECK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &raw) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_INVALID_STATE);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(memcmp(actual_next, context->p2g2_frame_next_logits, sizeof(actual_next)) == 0);
  CHECK(memcmp(successor, context->token_frame_successor, sizeof(successor)) == 0);
  assert_pending(output);

  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  CHECK(*(const int64_t *)lengths->data == 3);
  CHECK(memcmp(keep->data, (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(memcmp(saved_views, context->pins.views, sizeof(saved_views)) == 0);
  assert_pending(output);
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
  OK(et_g3c4_private_call_abort_v1(context));
  snapshot_cache(context->cache, &after_abort);
  check_cache_snapshot_equal(&after_abort, &prefilled);
  CHECK(memcmp(context->generator_rng_words, initial_rng, sizeof(initial_rng)) == 0);
  CHECK(et_g3c4_pins_idle(&context->pins));
  offset = 0u;
  for (size_t i = 0u; i < 14u; i++) {
    CHECK(memcmp(parameter_bytes + offset, owner->parameters[i]->value->data,
                 (size_t)saved_views[i].byte_length) == 0);
    offset += (size_t)saved_views[i].byte_length;
  }
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  context = create_generator(owner);
  const int64_t full_ids[3] = {prompt[0], prompt[1], selected};
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  OK(et_g3c4_private_prefill3_v1(context, full_ids, reference_next));
  snapshot_cache(context->cache, &full_prefix);
  CHECK(memcmp(actual_next, reference_next, sizeof(actual_next)) == 0);
  CHECK(memcmp(actual_keys, full_prefix.keys, sizeof(actual_keys)) == 0);
  CHECK(memcmp(actual_values, full_prefix.values, sizeof(actual_values)) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void eos_rejections(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  bridge_carrier carrier;
  cache_snapshot entry, after;
  int64_t rng[4];
  begin_eos_pair(owner, prompt, 0, 56, &context, &input, &output);
  carrier = new_carrier();
  snapshot_cache(context->cache, &entry);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 15) == ET_G3C4_INVALID_ARGUMENT);
  carrier.length = 9;
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_SHAPE_MISMATCH);
  carrier.length = 8;
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, (unsigned char *)&carrier + 1, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, context, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, (void *)context->pins.views[0].data, 16) == ET_G3C4_INVALID_ARGUMENT);
  check_carrier_untouched(&carrier);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&entry, &after);
  close_pair(context, input, output);

  for (int cut = 0; cut < 3; cut++) {
    begin_eos_pair(owner, prompt, 0, 56, &context, &input, &output);
    carrier = new_carrier();
    fail_bridge_sample = cut == 0;
    fail_bridge_forward = cut == 1;
    forge_sample = cut == 2;
    forged_token = 256;
    CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
        context, input, &carrier, 16) != 0);
    fail_bridge_sample = fail_bridge_forward = forge_sample = 0;
    check_carrier_untouched(&carrier);
    CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
    assert_pending(output);
    OK(et_g3c4_private_call_abort_v1(context));
    snapshot_cache(context->cache, &after);
    CHECK(after.length == 2);
    CHECK(memcmp(after.keep, (const uint8_t[4]){1, 1, 0, 0}, 4u) == 0);
    check_dead_output(output);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
  begin_eos_pair(owner, prompt, 0, 56, &context, &input, &output);
  carrier = new_carrier();
  tampered_carrier = &carrier;
  tampered_context = context;
  CHECK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_SHAPE_MISMATCH);
  CHECK(tampered_carrier == NULL && carrier.length == 9);
  tampered_context = NULL;
  for (size_t i = 0u; i < 8u; i++) {
    CHECK(carrier.bytes[i] == 0xa5u && carrier.canary[i] == 0x5au);
  }
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  assert_pending(output);
  close_pair(context, input, output);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  parity_case(owner, (const int64_t[2]){0, 255}, 0);
  parity_case(owner, (const int64_t[2]){7, 11}, 1);
  rejection_case(owner);
  printf("G3-C4 P2/G2 carrier compatibility PASS: checks=%zu\n", checks);
  const int64_t prompts[2][2] = {{0, 255}, {7, 11}};
  for (int pair = 0; pair < 2; pair++) {
    printf("EOS-ORACLE-PAIR %d\n", pair);
    for (int categorical = 0; categorical < 2; categorical++)
      eos_parity(owner, prompts[pair], categorical, 1);
  }
  for (int categorical = 0; categorical < 2; categorical++)
    eos_parity(owner, prompts[0], categorical, 0);
  eos_rejections(owner);
  printf("G3-C4 P2/G2 EOS first frame PASS: checks=%zu pending-only=1\n", checks);
  return 0;
}
