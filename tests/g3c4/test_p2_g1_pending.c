#define ET_G3C4_P2_G1_PENDING_PRIVATE 1
#define ET_A2_KV_CACHE_TESTING 1
#define ET_G3C4_OUTPUT_PREPARE_FINAL_DISPATCH et_g3c4_p2g1_base_dispatch
#define ET_G3C4_OUTPUT_TEXT_TEST_MAIN et_g3c4_p2g1_predecessor_main
#include "test_output_text.c"
#undef ET_G3C4_OUTPUT_TEXT_TEST_MAIN

static int forge_sample;
static int64_t forged_token;
static int forge_descriptor;
static int malformed_input_view;

int32_t __real_et_i64_tensor_borrow_view_v1(
    const et_i64_tensor_borrow *, const et_kernel_tensor_view_v1 **,
    et_i64_tensor_error *);

int32_t __wrap_et_i64_tensor_borrow_view_v1(
    const et_i64_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 **result,
    et_i64_tensor_error *error) {
  int32_t status = __real_et_i64_tensor_borrow_view_v1(
      borrow, result, error);
  if (status == 0 && malformed_input_view && (*result)->rank == 2u &&
      (*result)->shape[0] == 1u && (*result)->shape[1] == 2u) {
    static et_kernel_tensor_view_v1 changed;
    static uint64_t shape[2];
    changed = **result;
    shape[0] = 1u;
    shape[1] = 2u;
    changed.shape = shape;
    switch (malformed_input_view) {
      case 1: changed.rank = 1u; break;
      case 2: shape[1] = 3u; break;
      case 3: changed.data = NULL; break;
      case 4: changed.byte_length = 8u; break;
      default: break;
    }
    *result = &changed;
  }
  return status;
}

static et_g3c4_context_internal *p2g1_categorical_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 1729, 1, INT64_C(0x3f800000), 17,
      INT64_C(0x3f000000), 1, -1);
  CHECK(context != NULL);
  return context;
}

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (forge_sample && call->capability != NULL &&
      strncmp(call->capability, "g3s.", 4u) == 0) {
    if (forge_descriptor) {
      et_kernel_call_v1 changed = *call;
      et_kernel_tensor_view_v1 outputs[2];
      memcpy(outputs, call->outputs, sizeof(outputs));
      outputs[0].rank = 0u;
      changed.outputs = outputs;
      return et_g3c4_p2g1_base_dispatch(runtime, &changed, error);
    }
    int32_t result = et_g3c4_p2g1_base_dispatch(runtime, call, error);
    if (result == 0) {
      et_kernel_tensor_view_v1 *outputs = call->outputs;
      *(int64_t *)outputs[0].data = forged_token;
    }
    return result;
  }
  return et_g3c4_p2g1_base_dispatch(runtime, call, error);
}

static void print_oracle_case(
    const char *mode, const float logits[256], int64_t token,
    const int64_t before[4], const int64_t after[4]) {
  printf("ORACLE %s %lld %lld %lld %lld %lld %lld %lld %lld %lld",
         mode, (long long)token,
         (long long)before[0], (long long)before[1],
         (long long)before[2], (long long)before[3],
         (long long)after[0], (long long)after[1],
         (long long)after[2], (long long)after[3]);
  for (size_t i = 0u; i < 256u; i++) {
    uint32_t bits;
    memcpy(&bits, &logits[i], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
}

static void pending_parity(
    et_g3c4_model_owner_internal *owner, int categorical) {
  const int64_t prompt[2] = {0, 255};
  int64_t reference_ids[3] = {0, 255, -1};
  float last[256], next[256], reference_logits[256];
  int64_t token = -1, original_rng[4], staged_rng[4];
  cache_snapshot committed, after_abort, reference_cache;
  binding_snapshot binding, after_binding;
  et_g3c4_context_internal *context = categorical
      ? p2g1_categorical_generator(owner) : create_generator(owner);
  void *input = create_prompt(prompt, 2);
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 1) == 0);
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_g3c4_output_internal *output =
      et_g3c4_private_output_reserve_v1(context, 2);
  CHECK(output != NULL);
  OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
  snapshot_cache(context->cache, &committed);
  snapshot_binding(context, &binding);
  CHECK(committed.length == 2);
  CHECK(memcmp(committed.keep, (const uint8_t[4]){1,1,0,0}, 4u) == 0);
  OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
  CHECK(token >= 0 && token <= 255);
  memcpy(staged_rng, context->token_frame_successor, sizeof(staged_rng));
  OK(et_g3c4_private_token_forward_v1(context, token, next));
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY);
  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      context->token_frame_transaction, 0u, &view, &error));
  OK(et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error));
  CHECK(*(const int64_t *)lengths->data == 3);
  CHECK(memcmp(keep->data, (const uint8_t[4]){1,1,1,0}, 4u) == 0);
  float staged_keys[16], staged_values[16];
  memcpy(staged_keys, keys->data, sizeof(staged_keys));
  memcpy(staged_values, values->data, sizeof(staged_values));
  OK(et_a2_kv_cache_transaction_view_end_v1(&view, &error));
  OK(et_g3c4_private_output_prepare_v1(context, output));
  check_numeric_output(output, 2, 1, token, staged_rng);
  decode_staging staging = {8, {0}};
  raw_carrier raw = {1, {(unsigned char)token}};
  OK(et_g3c4_private_output_copy_decode_ids_v1(context, output, &staging));
  OK(et_g3c4_private_output_accept_text_v1(context, output, &raw));
  CHECK(staging.bytes[0] == (unsigned char)token);
  CHECK(et_g3c4_private_call_prepare_end_v1(context) == ET_G3C4_INVALID_STATE);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  print_oracle_case(categorical ? "categorical" : "greedy",
                    last, token, original_rng, staged_rng);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  snapshot_cache(context->cache, &after_abort);
  snapshot_binding(context, &after_binding);
  check_cache_snapshot_equal(&committed, &after_abort);
  check_binding_snapshot_equal(&binding, &after_binding);
  CHECK(memcmp(context->generator_rng_words, original_rng,
               sizeof(original_rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));

  reference_ids[2] = token;
  et_g3c4_context_internal *reference = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 1));
  OK(et_g3c4_private_prefill3_v1(
      reference, reference_ids, reference_logits));
  CHECK(memcmp(next, reference_logits, sizeof(next)) == 0);
  snapshot_cache(reference->cache, &reference_cache);
  CHECK(reference_cache.length == 3);
  CHECK(memcmp(reference_cache.keep,
               (const uint8_t[4]){1,1,1,0}, 4u) == 0);
  CHECK(memcmp(staged_keys, reference_cache.keys,
               sizeof(staged_keys)) == 0);
  CHECK(memcmp(staged_values, reference_cache.values,
               sizeof(staged_values)) == 0);
  OK(et_g3c4_private_call_abort_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
}

static void prompt_failure_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  for (int mode = 1; mode <= 12; mode++) {
    float logits[256];
    cache_snapshot before, after;
    binding_snapshot bound, bound_after;
    int64_t rng[4];
    et_g3c4_context_internal *context = create_generator(owner);
    et_g3c4_input_internal *input = create_prompt(prompt, 2);
    uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
        input->tensor);
    size_t *strides = (size_t *)et_i64_tensor_test_stride_storage_v1(
        input->tensor);
    int64_t *ids = (int64_t *)et_i64_tensor_test_data_storage_v1(
        input->tensor);
    CHECK(shape != NULL && strides != NULL && ids != NULL);
    snapshot_cache(context->cache, &before);
    snapshot_binding(context, &bound);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_g3c4_output_internal *output =
        et_g3c4_private_output_reserve_v1(context, 2);
    CHECK(output != NULL);
    for (size_t i = 0u; i < 256u; i++) logits[i] = -47.0f;
    if (mode <= 4) malformed_input_view = mode;
    else if (mode == 5) shape[1] = 3u;
    else if (mode == 6) strides[1] = 16u;
    else if (mode <= 10)
      ids[(mode - 7) / 2] = mode % 2 == 1 ? -1 : 256;
    else if (mode == 11) et_i64_tensor_test_fail_alloc_after_v1(0u);
    else {
      prefill2_dispatches = 0u;
      fail_prefill2_at = 0u;
      record_prefill2 = 1;
    }
    int64_t rejected = et_g3c4_private_prompt_prefill_v1(context, input, logits);
    CHECK(rejected != 0);
    if (mode == 11) et_i64_tensor_test_reset_allocator_v1();
    if (mode == 12) {
      record_prefill2 = 0;
      fail_prefill2_at = SIZE_MAX;
      CHECK(prefill2_dispatches == 1u);
    }
    malformed_input_view = 0;
    shape[1] = 2u;
    strides[1] = 8u;
    ids[0] = prompt[0];
    ids[1] = prompt[1];
    for (size_t i = 0u; i < 256u; i++) CHECK(logits[i] == -47.0f);
    snapshot_cache(context->cache, &after);
    snapshot_binding(context, &bound_after);
    check_cache_snapshot_equal(&before, &after);
    check_binding_snapshot_equal(&bound, &bound_after);
    CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
    CHECK(et_g3c4_token_frame_idle(context));
    check_pending_output(output, context, 2, 1);
    OK(et_g3c4_private_call_abort_v1(context));
    check_dead_output(output);
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_g3c4_output_internal *retry =
        et_g3c4_private_output_reserve_v1(context, 2);
    CHECK(retry != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
    int64_t token = -1;
    float next[256];
    OK(et_g3c4_private_token_frame_begin_last_v1(context, logits, &token));
    OK(et_g3c4_private_token_forward_v1(context, token, next));
    OK(et_g3c4_private_output_prepare_v1(context, retry));
    OK(et_g3c4_private_call_abort_v1(context));
    check_dead_output(retry);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
}

static void rejected_budget_two(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {0, 255};
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(prompt, 2);
  cache_snapshot before, after;
  binding_snapshot binding, after_binding;
  int64_t rng[4];
  snapshot_cache(context->cache, &before);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 2) ==
        ET_G3C4_SHAPE_MISMATCH);
  CHECK(et_g3c4_pins_idle(&context->pins));
  CHECK(context->owner->active == NULL);
  CHECK(et_g3c4_private_call_acquire_v1(context, 2, 2) != 0);
  snapshot_cache(context->cache, &after);
  snapshot_binding(context, &after_binding);
  check_cache_snapshot_equal(&before, &after);
  check_binding_snapshot_equal(&binding, &after_binding);
  CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void forged_sampler_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {37, 43};
  for (int mode = 0; mode < 3; mode++) {
    float last[256], next[256];
    cache_snapshot before, after;
    binding_snapshot binding, after_binding;
    int64_t rng[4], token = -777;
    et_g3c4_context_internal *context = create_generator(owner);
    void *input = create_prompt(prompt, 2);
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_g3c4_output_internal *output =
        et_g3c4_private_output_reserve_v1(context, 2);
    CHECK(output != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
    snapshot_cache(context->cache, &before);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    forge_sample = 1;
    forge_descriptor = mode == 2;
    forged_token = mode == 0 ? -1 : 256;
    CHECK(et_g3c4_private_token_frame_begin_last_v1(
        context, last, &token) != 0);
    forge_sample = 0;
    forge_descriptor = 0;
    CHECK(token == -777);
    CHECK(et_g3c4_token_frame_idle(context));
    check_pending_output(output, context, 2, 1);
    snapshot_cache(context->cache, &after);
    snapshot_binding(context, &after_binding);
    check_cache_snapshot_equal(&before, &after);
    check_binding_snapshot_equal(&binding, &after_binding);
    CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
    OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
    OK(et_g3c4_private_token_forward_v1(context, token, next));
    OK(et_g3c4_private_output_prepare_v1(context, output));
    OK(et_g3c4_private_call_abort_v1(context));
    check_dead_output(output);
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_g3c4_output_internal *fresh =
        et_g3c4_private_output_reserve_v1(context, 2);
    CHECK(fresh != NULL);
    OK(et_g3c4_private_prompt_prefill_v1(context, input, last));
    OK(et_g3c4_private_token_frame_begin_last_v1(context, last, &token));
    OK(et_g3c4_private_token_forward_v1(context, token, next));
    OK(et_g3c4_private_output_prepare_v1(context, fresh));
    OK(et_g3c4_private_call_abort_v1(context));
    check_dead_output(fresh);
    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(context));
  }
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  pending_parity(owner, 0);
  pending_parity(owner, 1);
  prompt_failure_cuts(owner);
  forged_sampler_cuts(owner);
  rejected_budget_two(owner);
  printf("G3-C4 P2/G1 pending PASS: checks=%zu\n", checks);
  return 0;
}
