#define ET_I64_TENSOR_TESTING 1
#define ET_G3C4_PREFILL1_PRIVATE 1
#define ET_G3C4_PREFILL2_PRIVATE 1
#define ET_G3C4_PROMPT_T1_BORROW_PRIVATE 1
#define ET_G3C4_PROMPT_PREFILL_PRIVATE 1
#define main et_g3c4_prefill2_predecessor_main
#include "test_prefill3.c"
#undef main

static void *create_prompt(const int64_t *tokens, int64_t length) {
  void *shell = et_t1_i64_shell_create_v1(length);
  CHECK(shell != NULL);
  for (int64_t index = 0; index < length; index++)
    CHECK(et_t1_i64_shell_write_v1(shell, index, tokens[index]) == 0);
  CHECK(et_t1_i64_shell_seal_v1(shell) == 0);
  void *input = et_g3c4_private_input_from_t1_v1(shell);
  CHECK(input != NULL);
  return input;
}

static void check_idle(et_g3c4_context_internal *context) {
  CHECK(ET_G3C4_CONTEXT_BUSY(context) == ET_G3C4_CALL_IDLE);
  CHECK(context->call_kind == 0);
  CHECK(context->budget == 0);
  CHECK(context->acquired_mask == 0u);
  CHECK(context->owner->active == NULL);
  CHECK(et_g3c4_pins_idle(&context->pins));
}

static void check_borrow_available(et_g3c4_input_internal *input) {
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(input->tensor, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
}

static void route_parity(et_g3c4_model_owner_internal *owner) {
  const int64_t p1[1] = {255};
  const int64_t p2[2] = {0, 255};
  const int64_t *routes[2] = {p1, p2};
  const int64_t lengths[2] = {1, 2};
  const int64_t budgets[2] = {1, 0};

  for (size_t route = 0u; route < 2u; route++) {
    float direct_logits[256], bound_logits[256];
    cache_snapshot direct_cache, bound_cache;
    binding_snapshot direct_binding, bound_binding;
    et_g3c4_context_internal *direct = create_generator(owner);
    et_g3c4_context_internal *bound = create_generator(owner);
    void *input = create_prompt(routes[route], lengths[route]);

    OK(et_g3c4_private_call_acquire_v1(direct, 2, budgets[route]));
    if (lengths[route] == 1) {
      prefill1_dispatches = 0u;
      fail_prefill1_at = SIZE_MAX;
      record_prefill1 = 1;
      OK(et_g3c4_private_prefill1_v1(
          direct, routes[route][0], direct_logits));
      record_prefill1 = 0;
      CHECK(prefill1_dispatches == 21u);
    } else {
      prefill2_dispatches = 0u;
      fail_prefill2_at = SIZE_MAX;
      record_prefill2 = 1;
      OK(et_g3c4_private_prefill2_v1(
          direct, routes[route], direct_logits));
      record_prefill2 = 0;
      CHECK(prefill2_dispatches == 21u);
    }
    OK(et_g3c4_private_call_finish_v1(direct));

    OK(et_g3c4_private_prompt_prefill_preflight_v1(
        bound, input, budgets[route]));
    check_idle(bound);
    OK(et_g3c4_private_call_acquire_v1(bound, 2, budgets[route]));
    if (lengths[route] == 1) {
      prefill1_dispatches = 0u;
      record_prefill1 = 1;
    } else {
      prefill2_dispatches = 0u;
      record_prefill2 = 1;
    }
    OK(et_g3c4_private_prompt_prefill_v1(bound, input, bound_logits));
    record_prefill1 = 0;
    record_prefill2 = 0;
    CHECK(lengths[route] == 1 ? prefill1_dispatches == 21u
                              : prefill2_dispatches == 21u);
    check_borrow_available((et_g3c4_input_internal *)input);
    OK(et_g3c4_private_call_finish_v1(bound));

    CHECK(memcmp(direct_logits, bound_logits, sizeof(direct_logits)) == 0);
    snapshot_cache(direct->cache, &direct_cache);
    snapshot_cache(bound->cache, &bound_cache);
    CHECK(memcmp(direct_cache.keys, bound_cache.keys,
                 sizeof(direct_cache.keys)) == 0);
    CHECK(memcmp(direct_cache.values, bound_cache.values,
                 sizeof(direct_cache.values)) == 0);
    CHECK(direct_cache.length == bound_cache.length);
    CHECK(memcmp(direct_cache.keep, bound_cache.keep,
                 sizeof(direct_cache.keep)) == 0);
    snapshot_binding(direct, &direct_binding);
    snapshot_binding(bound, &bound_binding);
    CHECK(memcmp(direct_binding.identities, bound_binding.identities,
                 sizeof(direct_binding.identities)) == 0);
    CHECK(memcmp(direct_binding.values, bound_binding.values,
                 sizeof(direct_binding.values)) == 0);
    CHECK(memcmp(direct_binding.tokens, bound_binding.tokens,
                 sizeof(direct_binding.tokens)) == 0);
    CHECK(direct_binding.ready == bound_binding.ready);
    CHECK(bound_cache.length == lengths[route]);
    CHECK(memcmp(bound->prefill_tokens,
                 lengths[route] == 1
                   ? (const int64_t[3]){255, 0, 0}
                   : (const int64_t[3]){0, 255, 0},
                 sizeof(bound->prefill_tokens)) == 0);

    OK(et_g3c4_private_tensor_release_v1(input));
    OK(et_g3c4_private_generator_close_v1(direct));
    OK(et_g3c4_private_generator_close_v1(bound));
  }
}

static void pre_acquire_admission(et_g3c4_model_owner_internal *owner) {
  const int64_t p1[1] = {7};
  const int64_t p2[2] = {7, 11};
  et_g3c4_context_internal *context = create_generator(owner);
  void *input1 = create_prompt(p1, 1);
  void *input2 = create_prompt(p2, 2);
  void *dead = create_prompt(p1, 1);
  float output[256];
  cache_snapshot cache;
  binding_snapshot binding;
  int64_t rng[4];

  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 0));
  check_idle(context);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 1));
  check_idle(context);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input2, 0));
  check_idle(context);
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, input2, 1) == ET_G3C4_SHAPE_MISMATCH);
  check_idle(context);
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  for (size_t index = 0u; index < 256u; index++) output[index] = -47.0f;
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input2, output) == ET_G3C4_INVALID_STATE);
  for (size_t index = 0u; index < 256u; index++)
    CHECK(output[index] == -47.0f);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, input1, -1) == ET_G3C4_SHAPE_MISMATCH);
  check_idle(context);
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, input1, 2) == ET_G3C4_SHAPE_MISMATCH);
  check_idle(context);
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, context, 0) == ET_G3C4_INVALID_ARGUMENT);
  check_idle(context);
  OK(et_g3c4_private_tensor_release_v1(dead));
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(
      context, dead, 0) == ET_G3C4_INVALID_STATE);
  check_idle(context);

  OK(et_g3c4_private_tensor_release_v1(input1));
  OK(et_g3c4_private_tensor_release_v1(input2));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void adapter_failure_cuts(
    et_g3c4_model_owner_internal *owner, int64_t length) {
  const int64_t initial[2] = {3, 7};
  const int64_t next[2] = {11, 37};
  for (size_t cut = 0u; cut < 21u; cut++) {
    float first[256], output[256];
    cache_snapshot cache;
    binding_snapshot binding;
    int64_t rng[4];
    et_g3c4_context_internal *context = create_generator(owner);
    void *first_input = create_prompt(initial, length);
    void *next_input = create_prompt(next, length);

    OK(et_g3c4_private_prompt_prefill_preflight_v1(
        context, first_input, 0));
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    OK(et_g3c4_private_prompt_prefill_v1(context, first_input, first));
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t index = 0u; index < 256u; index++) output[index] = -91.0f;

    OK(et_g3c4_private_prompt_prefill_preflight_v1(
        context, next_input, 0));
    OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
    if (length == 1) {
      prefill1_dispatches = 0u;
      fail_prefill1_at = cut;
      record_prefill1 = 1;
    } else {
      prefill2_dispatches = 0u;
      fail_prefill2_at = cut;
      record_prefill2 = 1;
    }
    CHECK(et_g3c4_private_prompt_prefill_v1(
        context, next_input, output) != 0);
    record_prefill1 = 0;
    record_prefill2 = 0;
    fail_prefill1_at = SIZE_MAX;
    fail_prefill2_at = SIZE_MAX;
    CHECK(length == 1 ? prefill1_dispatches == cut + 1u
                      : prefill2_dispatches == cut + 1u);
    for (size_t index = 0u; index < 256u; index++)
      CHECK(output[index] == -91.0f);
    check_preserved(context, &cache, &binding, rng);
    check_borrow_available((et_g3c4_input_internal *)next_input);
    close_after_abort(context);
    OK(et_g3c4_private_tensor_release_v1(first_input));
    OK(et_g3c4_private_tensor_release_v1(next_input));
  }
}

static void borrow_and_alias_failures(et_g3c4_model_owner_internal *owner) {
  const int64_t p1[1] = {19};
  const int64_t p2[2] = {23, 29};
  float first[256], output[256];
  cache_snapshot cache;
  binding_snapshot binding;
  int64_t rng[4];
  et_i64_tensor_borrow *external = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  et_g3c4_context_internal *context = create_generator(owner);
  void *input1 = create_prompt(p1, 1);
  void *input2 = create_prompt(p2, 2);

  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  OK(et_g3c4_private_prompt_prefill_v1(context, input1, first));
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  for (size_t index = 0u; index < 256u; index++) output[index] = -73.0f;
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  et_i64_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_prompt_prefill_v1(context, input1, output) != 0);
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
  for (size_t index = 0u; index < 256u; index++)
    CHECK(output[index] == -73.0f);
  check_preserved(context, &cache, &binding, rng);
  check_borrow_available((et_g3c4_input_internal *)input1);
  OK(et_g3c4_private_call_abort_v1(context));

  CHECK(et_i64_tensor_borrow_begin_v1(
      ((et_g3c4_input_internal *)input1)->tensor, &external, &error) == 0);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_prompt_prefill_v1(context, input1, output) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  check_preserved(context, &cache, &binding, rng);
  CHECK(et_i64_tensor_borrow_end_v1(&external, &error) == 0);
  OK(et_g3c4_private_call_abort_v1(context));

  CHECK(et_i64_tensor_borrow_begin_v1(
      ((et_g3c4_input_internal *)input2)->tensor, &external, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(external, &view, &error) == 0);
  const void *input_data = view->data;
  CHECK(et_i64_tensor_borrow_end_v1(&external, &error) == 0);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input2, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input2, (float *)input_data) == ET_G3C4_INVALID_ARGUMENT);
  check_preserved(context, &cache, &binding, rng);
  check_borrow_available((et_g3c4_input_internal *)input2);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_prompt_prefill_preflight_v1(context, input1, 0));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 0));
  CHECK(et_g3c4_private_prompt_prefill_v1(
      context, input1, (float *)input1) == ET_G3C4_INVALID_ARGUMENT);
  check_preserved(context, &cache, &binding, rng);
  close_after_abort(context);

  OK(et_g3c4_private_tensor_release_v1(input1));
  OK(et_g3c4_private_tensor_release_v1(input2));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  route_parity(owner);
  pre_acquire_admission(owner);
  adapter_failure_cuts(owner, 1);
  adapter_failure_cuts(owner, 2);
  borrow_and_alias_failures(owner);
  printf("G3-C4 prompt/prefill binding PASS: checks=%zu routes=2 "
         "pre-acquire-cuts=5 numerical-cuts=42 borrow-cuts=3\n",
         checks);
  return 0;
}
