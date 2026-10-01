#define ET_G3C4_P2_G2_PREFIX_COMMIT_PRIVATE 1
#define ET_G3C4_P2_G2_CARRIER_BRIDGE_PRIVATE 1
#define ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN et_g3c4_p2g2_bridge_predecessor_main
#include "test_p2_g2_first_frame.c"
#undef ET_G3C4_P2_G2_FIRST_FRAME_TEST_MAIN

typedef struct bridge_carrier {
  int64_t length;
  unsigned char bytes[8];
  unsigned char canary[8];
} bridge_carrier;
_Static_assert(offsetof(bridge_carrier, canary) == 16u,
               "bridge carrier payload must end at byte 16");

static void check_carrier_untouched(const bridge_carrier *carrier) {
  CHECK(carrier->length == 8);
  for (size_t i = 0u; i < 8u; i++) {
    CHECK(carrier->bytes[i] == 0xa5u);
    CHECK(carrier->canary[i] == 0x5au);
  }
}

static bridge_carrier new_carrier(void) {
  bridge_carrier result = {8, {0}, {0}};
  memset(result.bytes, 0xa5, sizeof(result.bytes));
  memset(result.canary, 0x5a, sizeof(result.canary));
  return result;
}

static void active_pair(
    et_g3c4_model_owner_internal *owner, int categorical,
    const int64_t prompt[2], et_g3c4_context_internal **context,
    et_g3c4_input_internal **input, et_g3c4_output_internal **output,
    cache_snapshot *entry) {
  *context = p2g2_generator(owner, categorical);
  *input = create_prompt(prompt, 2);
  if (entry != NULL) snapshot_cache((*context)->cache, entry);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(*context, *input, 2));
  OK(et_g3c4_private_call_acquire_v1(*context, 2, 2));
  *output = et_g3c4_private_output_reserve_v1(*context, 2);
  CHECK(*output != NULL);
}

static void close_pair(
    et_g3c4_context_internal *context, et_g3c4_input_internal *input,
    et_g3c4_output_internal *output) {
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void parity_case(et_g3c4_model_owner_internal *owner,
                        const int64_t prompt[2], int categorical) {
  et_g3c4_context_internal *context, *reference;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  bridge_carrier carrier = new_carrier();
  cache_snapshot actual_cache, reference_cache;
  float actual_next[256], last[256], reference_next[256];
  float actual_keys[16], actual_values[16];
  int64_t original_rng[4], successor[4], reference_id = -1;

  active_pair(owner, categorical, prompt, &context, &input, &output, NULL);
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16));
  CHECK(carrier.length == 8);
  for (size_t i = 1u; i < sizeof(carrier.bytes); i++)
    CHECK(carrier.bytes[i] == 0u);
  for (size_t i = 0u; i < sizeof(carrier.canary); i++)
    CHECK(carrier.canary[i] == 0x5au);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  CHECK(context->token_frame_candidate == (int64_t)carrier.bytes[0]);
  CHECK(context->budget == 2 && context->p2g2_prefix_committed == 0u);
  CHECK(output->numeric_ready == 0u && output->ids_copied == 0u &&
        output->text_ready == 0u && output->length == 0);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  memcpy(actual_next, context->p2g2_frame_next_logits, sizeof(actual_next));
  memcpy(actual_keys, context->p2g2_frame_keys, sizeof(actual_keys));
  memcpy(actual_values, context->p2g2_frame_values, sizeof(actual_values));
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  OK(et_g3c4_private_call_abort_v1(context));
  snapshot_cache(context->cache, &actual_cache);
  CHECK(actual_cache.length == 2);
  CHECK(memcmp(actual_cache.keep, (const uint8_t[4]){1, 1, 0, 0}, 4u) == 0);
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  reference = p2g2_generator(owner, categorical);
  input = create_prompt(prompt, 2);
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 2));
  output = et_g3c4_private_output_reserve_v1(reference, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(reference, input, last));
  snapshot_cache(reference->cache, &reference_cache);
  check_cache_snapshot_equal(&actual_cache, &reference_cache);
  OK(et_g3c4_private_token_frame_begin_last_v1(
      reference, last, &reference_id));
  CHECK(reference_id == (int64_t)carrier.bytes[0]);
  print_oracle_case(categorical ? "categorical" : "greedy", last,
                    reference_id, original_rng,
                    reference->token_frame_successor);
  CHECK(memcmp(successor, reference->token_frame_successor,
               sizeof(successor)) == 0);
  OK(et_g3c4_private_token_forward_v1(
      reference, reference_id, reference_next));
  CHECK(memcmp(actual_next, reference_next, sizeof(actual_next)) == 0);
  CHECK(memcmp(actual_keys, reference->p2g2_frame_keys,
               sizeof(actual_keys)) == 0);
  CHECK(memcmp(actual_values, reference->p2g2_frame_values,
               sizeof(actual_values)) == 0);
  close_pair(reference, input, output);
}

static void rejection_case(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  bridge_carrier carrier = new_carrier();
  cache_snapshot entry, after;
  int64_t rng[4];

  active_pair(owner, 0, prompt, &context, &input, &output, &entry);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  et_g3c4_input_internal *short_input = create_prompt(
      (const int64_t[1]){7}, 1);
  et_g3c4_input_internal *dead_input = create_prompt(prompt, 2);
  OK(et_g3c4_private_tensor_release_v1(dead_input));
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, short_input, &carrier, 16) == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, dead_input, &carrier, 16) != 0);
  OK(et_g3c4_private_tensor_release_v1(short_input));
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 15) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 17) == ET_G3C4_INVALID_ARGUMENT);
  for (int64_t bad = -1; bad <= 9; bad += 8) {
    carrier.length = bad;
    CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, &carrier, 16) == ET_G3C4_SHAPE_MISMATCH);
  }
  carrier.length = 8;
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (unsigned char *)&carrier + 1, 16) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, context, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, context->owner, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, input, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, output, 16) == ET_G3C4_INVALID_ARGUMENT);
  const void *output_ids = et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(output_ids != NULL);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (void *)output_ids, 16) == ET_G3C4_INVALID_ARGUMENT);
  const void *ids = et_i64_tensor_test_data_storage_v1(input->tensor);
  CHECK(ids != NULL);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (void *)ids, 16) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (void *)context->pins.views[0].data, 16) ==
        ET_G3C4_INVALID_ARGUMENT);
  et_f32_tensor *spare_f32 = NULL;
  et_f32_tensor_error f32_error;
  OK(et_f32_tensor_create_v1(
      1u, (const uint64_t[1]){4u}, &spare_f32, &f32_error));
  const float *f32_data = et_f32_tensor_test_data_storage_v1(spare_f32);
  CHECK(f32_data != NULL);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (void *)f32_data, 16) == ET_G3C4_INVALID_ARGUMENT);
  OK(et_f32_tensor_destroy_v1(&spare_f32, &f32_error));
  et_a2_kv_cache_read_borrow *held = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error cache_error;
  OK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &held, &cache_error));
  OK(et_a2_kv_cache_read_borrow_layer_v1(
      held, 0u, &keys, &values, &lengths, &keep, &cache_error));
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, (void *)keys->data, 16) == ET_G3C4_INVALID_ARGUMENT);
  OK(et_a2_kv_cache_read_borrow_end_v1(&held, &cache_error));
  check_carrier_untouched(&carrier);
  snapshot_cache(context->cache, &after);
  check_cache_snapshot_equal(&entry, &after);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);

  for (size_t cut = 0u; cut <= 20u; cut += 10u) {
    prefill2_dispatches = 0u;
    fail_prefill2_at = cut;
    record_prefill2 = 1;
    CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, &carrier, 16) != 0);
    record_prefill2 = 0;
    fail_prefill2_at = SIZE_MAX;
    check_carrier_untouched(&carrier);
    snapshot_cache(context->cache, &after);
    check_cache_snapshot_equal(&entry, &after);
    CHECK(context->prefill_binding_ready == 0u);
    CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  }
  close_pair(context, input, output);

  for (int mode = 0; mode < 3; mode++) {
    active_pair(owner, 0, prompt, &context, &input, &output, &entry);
    fail_bridge_sample = mode == 0;
    forge_sample = mode != 0;
    forged_token = mode == 1 ? -1 : 256;
    CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, &carrier, 16) != 0);
    fail_bridge_sample = 0;
    forge_sample = 0;
    check_carrier_untouched(&carrier);
    CHECK(context->prefill_binding_ready == 1u);
    CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
    OK(et_g3c4_private_call_abort_v1(context));
    snapshot_cache(context->cache, &after);
    CHECK(after.length == 2);
    check_dead_output(output);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }

  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  fail_bridge_forward = 1;
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16) != 0);
  fail_bridge_forward = 0;
  check_carrier_untouched(&carrier);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  snapshot_cache(context->cache, &after);
  CHECK(after.length == 2);
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  context->generator_policy[5] = 56;
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_INVALID_STATE);
  check_carrier_untouched(&carrier);
  CHECK(context->token_frame_candidate == 56);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_SAMPLED);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  snapshot_cache(context->cache, &after);
  CHECK(after.length == 2);
  CHECK(memcmp(after.keep, (const uint8_t[4]){1, 1, 0, 0}, 4u) == 0);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  check_dead_output(output);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16));
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_INVALID_STATE);
  close_pair(context, input, output);
  CHECK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16) == ET_G3C4_INVALID_STATE);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  parity_case(owner, (const int64_t[2]){0, 255}, 0);
  parity_case(owner, (const int64_t[2]){7, 11}, 1);
  rejection_case(owner);
  printf("G3-C4 P2/G2 carrier bridge PASS: checks=%zu\n", checks);
  return 0;
}
