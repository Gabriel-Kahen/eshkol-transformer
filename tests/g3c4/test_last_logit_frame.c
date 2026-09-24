#define ET_G3C4_LAST_LOGIT_FRAME_PRIVATE 1
#define main et_g3c4_prefill3_predecessor_main
#define __wrap_et_kernel_runtime_dispatch et_g3c4_prefill3_base_dispatch
#include "test_prefill3.c"
#undef __wrap_et_kernel_runtime_dispatch
#undef main

static int record_adapter;
static int fail_adapter;
static size_t adapter_dispatches;
static size_t total_adapter_dispatches;
static const float *expected_last_logits;
static unsigned char additional_identities[2][14];
static size_t additional_owner_count;

static et_g3c4_model_owner_internal *create_additional_owner(void) {
  unsigned char *owner_identities;
  et_g3c4_model_owner_internal *owner;
  CHECK(additional_owner_count < 2u);
  owner_identities = additional_identities[additional_owner_count++];
  owner = et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &owner_identities[(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_adapter && strncmp(call->capability, "g3s.", 4u) == 0) {
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    adapter_dispatches++;
    total_adapter_dispatches++;
    CHECK(strcmp(call->capability,
                 call->input_count == 2u ? "g3s.greedy" :
                                           "g3s.categorical") == 0);
    CHECK(strcmp(call->request->operation,
                 call->input_count == 2u ? "g3s.greedy.forward" :
                                           "g3s.categorical.forward") == 0);
    CHECK(call->request->rank == 2u);
    CHECK(call->request->shape[0] == 1u);
    CHECK(call->request->shape[1] == 256u);
    CHECK(inputs[0].byte_length == 256u * sizeof(float));
    CHECK(memcmp(inputs[0].data, expected_last_logits,
                 256u * sizeof(float)) == 0);
    if (fail_adapter) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.last-logit-frame-cut");
      strcpy(error->message, "injected last-logit sampler failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return et_g3c4_prefill3_base_dispatch(runtime, call, error);
}

static et_g3c4_context_internal *create_categorical_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 1729, 1, INT64_C(0x3f800000), 17,
      INT64_C(0x3f000000), 1, -1);
  CHECK(context != NULL);
  return context;
}

static void check_idle_preserved(
    et_g3c4_context_internal *context, const cache_snapshot *cache,
    const binding_snapshot *binding, const int64_t rng[4], int64_t output) {
  check_preserved(context, cache, binding, rng);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(context->budget == 1);
  CHECK(output == INT64_C(-777));
}

static void exact_prefill_to_committed_token(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[3] = {3, 7, 11};
  int64_t reference_tokens[4] = {3, 7, 11, -1};
  float last[256], next[256], reference[1024];
  int64_t speculative = -1, published = -1;
  cache_snapshot cache;
  et_g3c4_context_internal *context = create_generator(owner);
  et_g3c4_context_internal *reference_context;

  prefill_success(context, prompt, last);
  expected_last_logits = last;
  adapter_dispatches = 0u;
  fail_adapter = 0;
  record_adapter = 1;
  OK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &speculative));
  record_adapter = 0;
  CHECK(adapter_dispatches == 1u);
  CHECK(speculative >= 0 && speculative <= 255);
  CHECK(context->token_frame_position == 3);
  reference_tokens[3] = speculative;

  reference_context = create_generator(create_additional_owner());
  OK(et_g3c4_private_call_acquire_v1(reference_context, 0, 0));
  OK(et_g3c4_private_full_prefix_forward_v1(
      reference_context, reference_tokens, reference));
  OK(et_g3c4_private_call_finish_v1(reference_context));
  OK(et_g3c4_private_generator_close_v1(reference_context));

  OK(et_g3c4_private_token_forward_v1(context, speculative, next));
  CHECK(memcmp(next, reference + 3u * 256u, sizeof(next)) == 0);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == speculative);
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  CHECK(cache.length == 4);
  CHECK(memcmp(cache.keep, (const uint8_t[4]){1,1,1,1}, 4u) == 0);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void categorical_legacy_parity(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[3] = {3, 7, 11};
  float last_a[256], last_b[256], carrier[1024];
  int64_t token_a = -1, token_b = -1;
  et_g3c4_context_internal *a = create_categorical_generator(owner);
  et_g3c4_context_internal *b =
      create_categorical_generator(create_additional_owner());

  prefill_success(a, prompt, last_a);
  prefill_success(b, prompt, last_b);
  CHECK(memcmp(last_a, last_b, sizeof(last_a)) == 0);
  for (size_t index = 0u; index < 1024u; index++) carrier[index] = -1.0f;
  memcpy(carrier + 3u * 256u, last_b, sizeof(last_b));
  expected_last_logits = last_a;
  adapter_dispatches = 0u;
  record_adapter = 1;
  OK(et_g3c4_private_token_frame_begin_last_v1(a, last_a, &token_a));
  record_adapter = 0;
  OK(et_g3c4_private_token_frame_begin_v1(b, carrier, &token_b));
  CHECK(adapter_dispatches == 1u);
  CHECK(token_a == token_b);
  CHECK(memcmp(a->token_frame_successor, b->token_frame_successor,
               sizeof(a->token_frame_successor)) == 0);
  CHECK(a->token_frame_position == b->token_frame_position);
  OK(et_g3c4_private_token_frame_abort_v1(a));
  OK(et_g3c4_private_token_frame_abort_v1(b));
  close_after_abort(a);
  close_after_abort(b);
}

static void sampler_and_allocation_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[3] = {3, 7, 11};
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 8u; allowed++) {
    float last[256];
    int64_t output = -777, rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill_success(context, prompt, last);
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    int64_t status = et_g3c4_private_token_frame_begin_last_v1(
        context, last, &output);
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status != 0) {
      failures++;
      CHECK(allowed < 8u);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_ALLOCATION_FAILED);
      check_idle_preserved(context, &cache, &binding, rng, output);
    } else {
      CHECK(allowed == 8u);
      CHECK(output >= 0 && output <= 255);
      OK(et_g3c4_private_token_frame_abort_v1(context));
    }
    close_after_abort(context);
  }
  CHECK(failures == 8u);

  float last[256];
  int64_t output = -777, rng[4];
  cache_snapshot cache;
  binding_snapshot binding;
  et_g3c4_context_internal *context = create_generator(owner);
  prefill_success(context, prompt, last);
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  expected_last_logits = last;
  adapter_dispatches = 0u;
  fail_adapter = 1;
  record_adapter = 1;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &output) != 0);
  record_adapter = 0;
  fail_adapter = 0;
  CHECK(adapter_dispatches == 1u);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_KERNEL_CODE_PROVIDER_REJECTED);
  check_idle_preserved(context, &cache, &binding, rng, output);
  close_after_abort(context);
}

static void stale_nonfinite_and_state_rejections(
    et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[3] = {3, 7, 11};
  float last[256];
  int64_t output = -777;
  et_g3c4_context_internal *context = create_generator(owner);
  prefill_success(context, prompt, last);
  unsigned char saved = ((unsigned char *)context->pins.views[0].data)[0];
  ((unsigned char *)context->pins.views[0].data)[0] ^= 1u;
  adapter_dispatches = 0u;
  record_adapter = 1;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &output) != 0);
  record_adapter = 0;
  CHECK(adapter_dispatches == 0u);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_STALE_BINDING);
  ((unsigned char *)context->pins.views[0].data)[0] = saved;
  last[0] = NAN;
  output = -777;
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &output) != 0);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(output == -777);
  close_after_abort(context);

  context = create_generator(owner);
  prefill_success(context, prompt, last);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &output) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, &output) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  close_after_abort(context);
}

static void alias_rejections(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[3] = {3, 7, 11};
  union { float logits[256]; int64_t words[128]; } shared;
  float last[256];
  int64_t output = -777;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  et_g3c4_context_internal *context = create_generator(owner);
  prefill_success(context, prompt, last);
  memcpy(shared.logits, last, sizeof(last));
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, shared.logits, shared.words) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, (int64_t *)(void *)context) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, (const float *)(const void *)context, &output) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, (int64_t *)(void *)context->owner) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, (const float *)(const void *)context->owner, &output) != 0);
  for (size_t index = 0u; index < 14u; index++) {
    CHECK(et_g3c4_private_token_frame_begin_last_v1(
        context, last,
        (int64_t *)(void *)context->pins.views[index].data) != 0);
    CHECK(et_g3c4_private_token_frame_begin_last_v1(
        context,
        (const float *)(const void *)context->pins.views[index].data,
        &output) != 0);
  }
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, (int64_t *)(void *)context->cache) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, (const float *)(const void *)context->cache, &output) != 0);
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  int64_t *cache_alias = (int64_t *)(void *)keys->data;
  const float *cache_input_alias = (const float *)(const void *)keys->data;
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, last, cache_alias) != 0);
  CHECK(et_g3c4_private_token_frame_begin_last_v1(
      context, cache_input_alias, &output) != 0);
  CHECK(et_g3c4_token_frame_idle(context));
  CHECK(output == -777);
  close_after_abort(context);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  exact_prefill_to_committed_token(owner);
  categorical_legacy_parity(owner);
  sampler_and_allocation_cuts(owner);
  stale_nonfinite_and_state_rejections(owner);
  alias_rejections(owner);
  printf("G3-C4 last-logit frame PASS: checks=%zu sampler-dispatches=%zu allocation-cuts=8\n",
         checks, total_adapter_dispatches);
  return 0;
}
