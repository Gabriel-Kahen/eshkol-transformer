#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_EOS_TEST_MAIN eos_predecessor_main
#define ET_G3C4_P2_G2_EOS_TEST_DISPATCH eos_dispatch
#include "test_p2_g2_eos_first_frame.c"

static int capture_t4;
static cache_snapshot reference_t4;
static bridge_carrier *second_tamper;
static et_g3c4_context_internal *second_context;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  int32_t result = eos_dispatch(runtime, call, error);
  if (result == 0 && second_tamper != NULL &&
      second_context->p2g2_second_frame == 1u) {
    second_tamper->length = 9;
    second_tamper = NULL;
  }
  return result;
}

int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (capture_t4 && transaction != NULL && *transaction != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    OK(et_a2_kv_cache_transaction_view_begin_v1(*transaction, 0u, &view, &local));
    OK(et_a2_kv_cache_transaction_view_tensors_v1(
        view, &keys, &values, &lengths, &keep, &local));
    memcpy(reference_t4.keys, keys->data, sizeof(reference_t4.keys));
    memcpy(reference_t4.values, values->data, sizeof(reference_t4.values));
    reference_t4.length = *(const int64_t *)lengths->data;
    memcpy(reference_t4.keep, keep->data, sizeof(reference_t4.keep));
    CHECK(reference_t4.length == 4);
    OK(et_a2_kv_cache_transaction_view_end_v1(&view, &local));
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(transaction, error);
}

static void pending_ids_zero(et_g3c4_output_internal *output) {
  const int64_t *ids = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(ids != NULL);
  CHECK(memcmp(ids, (const int64_t[2]){0, 0}, 16u) == 0);
  assert_pending(output);
}

static void start_prefix(et_g3c4_model_owner_internal *owner,
                         const int64_t prompt[2], int categorical,
                         int counter_case, int64_t eos, et_g3c4_context_internal **context,
                         et_g3c4_input_internal **input,
                         et_g3c4_output_internal **output) {
  if (eos >= 0) begin_eos_pair(owner, prompt, categorical, eos, context, input, output);
  else active_pair(owner, categorical, prompt, context, input, output, NULL);
  /* Test-only starting Philox counters in a genuine registered owner. */
  if (counter_case > 0 && counter_case <= 3) {
    (*context)->generator_rng_words[2] = counter_case == 2 ? -3 : -2;
    (*context)->generator_rng_words[3] = counter_case == 1 ? 0 : -1;
  }
  if (counter_case == 4) (*context)->generator_policy[2] = 1;
  bridge_carrier first = new_carrier();
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(*context, *input, &first, 16));
  struct { int64_t length; unsigned char byte; } raw = {1, first.bytes[0]};
  OK(et_g3c4_private_p2g2_prefix_commit_v1(*context, *output, &raw));
  CHECK((*context)->p2g2_prefix_committed == 1u);
  CHECK((*context)->p2g2_second_frame == 0u);
  pending_ids_zero(*output);
}

typedef struct frozen_parameters {
  unsigned char values[4768], gradients[4768];
  uint8_t present[14];
  et_f32_gradient_metadata_v1 metadata[14];
} frozen_parameters;

static void freeze_parameters(et_g3c4_model_owner_internal *owner,
                              frozen_parameters *snapshot) {
  memset(snapshot, 0, sizeof(*snapshot));
  size_t offset = 0u;
  et_f32_tensor_error error;
  for (size_t i = 0u; i < 14u; i++) {
    et_f32_parameter *parameter = owner->parameters[i];
    size_t bytes = parameter->value->byte_length;
    CHECK(bytes <= sizeof(snapshot->values) - offset);
    memcpy(snapshot->values + offset, parameter->value->data, bytes);
    if (parameter->gradient != NULL) {
      CHECK(parameter->gradient->byte_length == bytes);
      snapshot->present[i] = 1u;
      memcpy(snapshot->gradients + offset, parameter->gradient->data, bytes);
    }
    snapshot->metadata[i].struct_size = sizeof(snapshot->metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(parameter, &snapshot->metadata[i], &error));
    offset += bytes;
  }
  CHECK(offset == sizeof(snapshot->values));
}

static void check_frozen_parameters(et_g3c4_model_owner_internal *owner,
                                    const frozen_parameters *snapshot) {
  frozen_parameters now;
  freeze_parameters(owner, &now);
  CHECK(memcmp(&now, snapshot, sizeof(now)) == 0);
}

static void second_parity(et_g3c4_model_owner_internal *owner,
                          const int64_t prompt[2], int categorical,
                          int counter_case, int64_t eos, int oracle) {
  et_g3c4_context_internal *context, *reference;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  cache_snapshot prefix, after;
  bridge_carrier second = new_carrier();
  float last[256], actual[256], t4[1024];
  float keys[16], values[16];
  int64_t rng[4], successor[4], first;
  start_prefix(owner, prompt, categorical, counter_case, eos, &context, &input, &output);
  frozen_parameters parameters;
  freeze_parameters(owner, &parameters);
  first = context->p2g2_prefix_token;
  memcpy(last, context->p2g2_prefix_next_logits, sizeof(last));
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  snapshot_cache(context->cache, &prefix);
  CHECK(prefix.length == 3);
  CHECK(memcmp(prefix.keep, (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
  int64_t fake_id = -17;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(context, last, &fake_id) != 0);
  CHECK(fake_id == -17 && et_g3c4_token_frame_idle(context));
  OK(et_g3c4_private_p2g2_second_frame_carrier_v1(context, output, &second, 16));
  CHECK(context->p2g2_second_frame == 1u && context->token_frame_position == 3);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  int64_t selected = context->token_frame_candidate;
  if (eos >= 0) CHECK(selected == eos);
  CHECK(second.length == 8 && second.bytes[0] == (unsigned char)selected);
  for (size_t i = 1u; i < 8u; i++) CHECK(second.bytes[i] == 0u);
  for (size_t i = 0u; i < 8u; i++) CHECK(second.canary[i] == 0x5au);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  CHECK(memcmp(last, context->p2g2_prefix_next_logits, sizeof(last)) == 0);
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  if (oracle) print_oracle_case(counter_case == 4 ? "categorical-singleton" :
                               (categorical ? "categorical" : "greedy"),
                               last, selected, rng, successor);
  memcpy(actual, context->p2g2_frame_next_logits, sizeof(actual));
  memcpy(keys, context->p2g2_frame_keys, sizeof(keys));
  memcpy(values, context->p2g2_frame_values, sizeof(values));
  pending_ids_zero(output);
  check_frozen_parameters(owner, &parameters);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) != 0);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) != 0);
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(context, output, &second, 16) != 0);
  pending_ids_zero(output);
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *k = NULL, *v = NULL, *length = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(view, &k, &v, &length, &keep, &error));
  CHECK(*(const int64_t *)length->data == 4);
  CHECK(memcmp(keep->data, (const uint8_t[4]){1, 1, 1, 1}, 4u) == 0);
  for (size_t head = 0u; head < 2u; head++) {
    CHECK(memcmp((const float *)k->data + head * 8u,
                 prefix.keys + head * 8u, 6u * sizeof(float)) == 0);
    CHECK(memcmp((const float *)v->data + head * 8u,
                 prefix.values + head * 8u, 6u * sizeof(float)) == 0);
  }
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  pending_ids_zero(output);
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&prefix, &after);
  CHECK(context->p2g2_prefix_committed == 0u && context->p2g2_second_frame == 0u);
  check_dead_output(output);
  check_frozen_parameters(owner, &parameters);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  reference = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 0, 0));
  const int64_t ids[4] = {prompt[0], prompt[1], first, selected};
  capture_t4 = 1;
  OK(et_g3c4_private_full_prefix_forward_v1(reference, ids, t4));
  capture_t4 = 0;
  CHECK(memcmp(actual, t4 + 768u, sizeof(actual)) == 0);
  CHECK(memcmp(keys, reference_t4.keys, sizeof(keys)) == 0);
  CHECK(memcmp(values, reference_t4.values, sizeof(values)) == 0);
  OK(et_g3c4_private_call_finish_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
}

static void second_negatives(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  for (int cut = 0; cut < 10; cut++) {
    et_g3c4_context_internal *context;
    et_g3c4_input_internal *input;
    et_g3c4_output_internal *output;
    start_prefix(owner, prompt, 1, cut == 8 ? 3 : 0, -1, &context, &input, &output);
    bridge_carrier carrier = new_carrier();
    cache_snapshot before, after;
    int64_t rng[4]; memcpy(rng, context->generator_rng_words, sizeof(rng));
    snapshot_cache(context->cache, &before);
    fail_bridge_sample = cut == 0;
    fail_bridge_forward = cut == 1;
    forge_sample = cut == 2 || cut == 3;
    forged_token = cut == 2 ? -1 : 256;
    if (cut == 4) carrier.length = 9;
    if (cut == 5) et_a2_kv_cache_test_fail_alloc_after_v1(0u);
    if (cut == 6) { second_tamper = &carrier; second_context = context; }
    int64_t *ids = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
    if (cut == 7) ids[1] = 1;
    CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(
        context, cut == 9 ? (void *)input : (void *)output, &carrier, 16) != 0);
    fail_bridge_sample = fail_bridge_forward = forge_sample = 0;
    second_tamper = NULL;
    et_a2_kv_cache_test_reset_allocator_v1();
    if (cut == 7) ids[1] = 0;
    CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
    CHECK(carrier.bytes[0] == 0xa5u);
    if (et_g3c4_token_frame_idle(context)) {
      snapshot_cache(context->cache, &after);
      check_cache_snapshot_equal(&before, &after);
    }
    pending_ids_zero(output);
    OK(et_g3c4_private_call_abort_v1(context));
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&before, &after);
    check_dead_output(output);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
}

static void second_preflight_negatives(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  for (int cut = 0; cut < 7; cut++) {
    et_g3c4_context_internal *context;
    et_g3c4_input_internal *input;
    et_g3c4_output_internal *output;
    start_prefix(owner, prompt, 0, 0, -1, &context, &input, &output);
    bridge_carrier carrier = new_carrier();
    cache_snapshot before, after;
    snapshot_cache(context->cache, &before);
    void *pointer = &carrier;
    int64_t bytes = 16;
    if (cut == 0) bytes = 15;
    if (cut == 1) pointer = (unsigned char *)&carrier + 1;
    if (cut == 2) pointer = context;
    if (cut == 3) pointer = (void *)et_i64_tensor_test_data_storage_v1(output->ids);
    if (cut == 4) pointer = (void *)context->pins.views[0].data;
    if (cut == 5) context->p2g2_second_frame = 1u;
    et_i64_tensor_borrow *held = NULL;
    et_i64_tensor_error error;
    if (cut == 6) OK(et_i64_tensor_borrow_begin_v1(output->ids, &held, &error));
    CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(context, output, pointer, bytes) != 0);
    if (cut == 5) context->p2g2_second_frame = 0u;
    if (cut == 6) OK(et_i64_tensor_borrow_end_v1(&held, &error));
    check_carrier_untouched(&carrier);
    CHECK(et_g3c4_token_frame_idle(context));
    pending_ids_zero(output);
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&before, &after);
    close_pair(context, input, output);
  }
  /* Genuine first EOS never manufactures a non-EOS committed prefix. */
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  bridge_carrier carrier = new_carrier();
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(context, input, &carrier, 16));
  int64_t eos = context->token_frame_candidate;
  close_pair(context, input, output);
  begin_eos_pair(owner, prompt, 0, eos, &context, &input, &output);
  carrier = new_carrier();
  OK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(context, input, &carrier, 16));
  bridge_carrier second = new_carrier();
  CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(context, output, &second, 16) != 0);
  check_carrier_untouched(&second);
  pending_ids_zero(output);
  CHECK(context->p2g2_prefix_committed == 0u && context->token_frame_position == 2);
  close_pair(context, input, output);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  const int64_t prompts[2][2] = {{0, 255}, {7, 11}};
  for (int pair = 0; pair < 2; pair++) {
    printf("SECOND-ORACLE-PAIR %d\n", pair);
    for (int categorical = 0; categorical < 2; categorical++)
      second_parity(owner, prompts[pair], categorical, 0, -1, 1);
  }
  for (int counter = 1; counter <= 2; counter++)
    second_parity(owner, prompts[0], 1, counter, -1, 1);
  second_parity(owner, prompts[0], 1, 4, -1, 1);
  /* Discover second EOS on a separate complete speculative request. */
  et_g3c4_context_internal *probe;
  et_g3c4_input_internal *probe_input;
  et_g3c4_output_internal *probe_output;
  start_prefix(owner, prompts[0], 0, 0, -1, &probe, &probe_input, &probe_output);
  bridge_carrier eos_probe = new_carrier();
  OK(et_g3c4_private_p2g2_second_frame_carrier_v1(probe, probe_output, &eos_probe, 16));
  int64_t second_eos = probe->token_frame_candidate;
  CHECK(second_eos != probe->p2g2_prefix_token);
  close_pair(probe, probe_input, probe_output);
  second_parity(owner, prompts[0], 0, 0, second_eos, 1);
  second_negatives(owner);
  second_preflight_negatives(owner);
  printf("G3-C4 P2/G2 second frame PASS: checks=%zu pending-only=1\n", checks);
  return 0;
}
