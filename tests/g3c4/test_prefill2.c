#define ET_G3C4_PREFILL1_PRIVATE 1
#define ET_G3C4_PREFILL2_PRIVATE 1
#define main et_g3c4_prefill3_predecessor_main
#include "test_prefill3.c"
#undef main

static void prefill2_success(
    et_g3c4_context_internal *context, const int64_t tokens[2],
    float logits[256]) {
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  prefill2_dispatches = 0u;
  fail_prefill2_at = SIZE_MAX;
  record_prefill2 = 1;
  OK(et_g3c4_private_prefill2_v1(context, tokens, logits));
  record_prefill2 = 0;
  CHECK(prefill2_dispatches == 21u);
}

static void numerical_parity_and_commit(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[4] = {3, 7, 11, 37};
  const int64_t committed_tokens[3] = {3, 7, 0};
  float full[1024], actual[256];
  int64_t rng[4];
  cache_snapshot cache;
  et_g3c4_context_internal *reference = create_generator(owner);
  et_g3c4_context_internal *context = create_generator(owner);

  OK(et_g3c4_private_call_acquire_v1(reference, 0, 0));
  OK(et_g3c4_private_full_prefix_forward_v1(reference, tokens, full));
  OK(et_g3c4_private_call_finish_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));

  memcpy(rng, context->generator_rng_words, sizeof(rng));
  prefill2_success(context, tokens, actual);
  CHECK(memcmp(actual, full + 256u, sizeof(actual)) == 0);
  for (size_t index = 0u; index < 256u; index++)
    CHECK(isfinite(actual[index]));
  CHECK(memcmp(context->generator_rng_words, rng, sizeof(rng)) == 0);
  CHECK(context->prefill_binding_ready == 1u);
  CHECK(memcmp(context->prefill_tokens, committed_tokens,
               sizeof(committed_tokens)) == 0);
  snapshot_cache(context->cache, &cache);
  CHECK(cache.length == 2);
  CHECK(memcmp(cache.keep, (const uint8_t[4]){1,1,0,0}, 4u) == 0);
  for (size_t head = 0u; head < 2u; head++)
    for (size_t position = 0u; position < 2u; position++)
      for (size_t dimension = 0u; dimension < 2u; dimension++) {
        const size_t cache_index = (head * 4u + position) * 2u + dimension;
        const size_t local_index = (head * 2u + position) * 2u + dimension;
        CHECK(cache.keys[cache_index] == prefill2_staged_keys[local_index]);
        CHECK(cache.values[cache_index] == prefill2_staged_values[local_index]);
      }
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void dispatch_failure_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t first_tokens[2] = {3, 7};
  const int64_t next_tokens[2] = {11, 37};
  for (size_t cut = 0u; cut < 21u; cut++) {
    float first[256], output[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill2_success(context, first_tokens, first);
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t index = 0u; index < 256u; index++) output[index] = -123.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    prefill2_dispatches = 0u;
    fail_prefill2_at = cut;
    record_prefill2 = 1;
    CHECK(et_g3c4_private_prefill2_v1(context, next_tokens, output) != 0);
    record_prefill2 = 0;
    CHECK(prefill2_dispatches == cut + 1u);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_KERNEL_CODE_PROVIDER_REJECTED);
    for (size_t index = 0u; index < 256u; index++)
      CHECK(output[index] == -123.0f);
    check_preserved(context, &cache, &binding, rng);
    close_after_abort(context);
  }
  fail_prefill2_at = SIZE_MAX;
}

static void a2_allocation_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t first_tokens[2] = {3, 7};
  const int64_t next_tokens[2] = {11, 37};
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 11u; allowed++) {
    float first[256], output[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill2_success(context, first_tokens, first);
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t index = 0u; index < 256u; index++) output[index] = -321.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    int64_t status =
        et_g3c4_private_prefill2_v1(context, next_tokens, output);
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status != 0) {
      failures++;
      CHECK(allowed < 11u);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_ALLOCATION_FAILED);
      for (size_t index = 0u; index < 256u; index++)
        CHECK(output[index] == -321.0f);
      check_preserved(context, &cache, &binding, rng);
      close_after_abort(context);
    } else {
      CHECK(allowed == 11u);
      CHECK(memcmp(context->prefill_tokens,
                   (const int64_t[3]){11,37,0},
                   sizeof(context->prefill_tokens)) == 0);
      OK(et_g3c4_private_call_finish_v1(context));
      OK(et_g3c4_private_generator_close_v1(context));
    }
  }
  CHECK(failures == 11u);
}

static void runtime_allocation_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t first_tokens[2] = {3, 7};
  const int64_t next_tokens[2] = {11, 37};
  for (size_t allowed = 0u; allowed < 2u; allowed++) {
    float first[256], output[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill2_success(context, first_tokens, first);
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t index = 0u; index < 256u; index++) output[index] = -456.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_m3t_test_k1_fail_after(allowed);
    CHECK(et_g3c4_private_prefill2_v1(context, next_tokens, output) != 0);
    et_m3t_test_k1_allocator_reset();
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_KERNEL_CODE_ALLOCATION_FAILED);
    for (size_t index = 0u; index < 256u; index++)
      CHECK(output[index] == -456.0f);
    check_preserved(context, &cache, &binding, rng);
    close_after_abort(context);
  }
}

static void alias_and_admission_rejections(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[2] = {3, 7};
  float first[256], output[256];
  int64_t rng[4];
  cache_snapshot cache;
  binding_snapshot binding;
  unsigned char parameter[4096];
  union {
    int64_t words[128];
    float logits[256];
  } overlap;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  et_g3c4_context_internal *context = create_generator(owner);

  prefill2_success(context, tokens, first);
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  memcpy(parameter, context->pins.views[10].data, sizeof(parameter));
  CHECK(et_g3c4_private_prefill2_v1(
      context, tokens, (float *)context->pins.views[10].data) != 0);
  CHECK(memcmp(parameter, context->pins.views[10].data,
               sizeof(parameter)) == 0);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill2_v1(
      context, tokens, (float *)(void *)context) != 0);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  float *cache_alias = (float *)keys->data;
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill2_v1(context, tokens, cache_alias) != 0);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  memset(&overlap, 0, sizeof(overlap));
  overlap.words[0] = 3;
  overlap.words[1] = 7;
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill2_v1(
      context, overlap.words, overlap.logits) != 0);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  for (size_t index = 0u; index < 256u; index++) output[index] = -789.0f;
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill2_v1(
      context, (const int64_t[2]){-1,7}, output) != 0);
  CHECK(et_g3c4_private_prefill2_v1(
      context, (const int64_t[2]){3,256}, output) != 0);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  CHECK(et_g3c4_private_prefill2_v1(context, tokens, output) != 0);
  check_preserved(context, &cache, &binding, rng);
  close_after_abort(context);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  numerical_parity_and_commit(owner);
  dispatch_failure_cuts(owner);
  a2_allocation_cuts(owner);
  runtime_allocation_cuts(owner);
  alias_and_admission_rejections(owner);
  printf("G3-C4 prefill2 PASS: checks=%zu roles=21 dispatch-cuts=21 "
         "a2-allocation-cuts=11 runtime-allocation-cuts=2\n", checks);
  return 0;
}
