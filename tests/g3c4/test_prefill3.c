#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROVIDER_ROUTES_PRIVATE 1
#define ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE 1
#define ET_G3C4_SAMPLER_TRANSPORT_PRIVATE 1
#define ET_G3C4_TOKEN_FRAME_PRIVATE 1
#define ET_G3C4_TOKEN_FORWARD_PRIVATE 1
#define ET_G3C4_PREFILL3_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <math.h>
#include <stdio.h>

void et_m3t_test_k1_fail_after(size_t allowed);
void et_m3t_test_k1_allocator_reset(void);

static size_t checks;
#define CHECK(condition) do { \
  checks++; \
  if (!(condition)) { \
    fprintf(stderr, "FAIL %s:%d: %s [domain=%lld category=%lld code=%lld]\n", \
        __FILE__, __LINE__, #condition, \
        (long long)et_g3c4_private_last_error_domain_v1(), \
        (long long)et_g3c4_private_last_error_category_v1(), \
        (long long)et_g3c4_private_last_error_code_v1()); \
    exit(1); \
  } \
} while (0)
#define OK(expression) CHECK((expression) == 0)

int32_t __real_et_kernel_runtime_dispatch(
    const et_kernel_runtime *, const et_kernel_call_v1 *, et_kernel_error *);

static const char *const expected_capabilities[21] = {
  "g3c4.embedding-forward", "g3c4.embedding-forward", "kernel.residual",
  "g3c4.layer-norm", "g3c4.linear", "g3c4.linear", "g3c4.linear",
  "g3c4.head-layout", "g3c4.head-layout", "g3c4.head-layout",
  "g3c4.causal-attention", "g3c4.head-layout", "g3c4.linear",
  "kernel.residual", "g3c4.layer-norm", "g3c4.linear", "g3c4.gelu",
  "g3c4.linear", "kernel.residual", "g3c4.layer-norm", "g3c4.linear"
};
static const char *const expected_operations[21] = {
  "g3c4.embedding.forward", "g3c4.embedding.forward", "residual.forward",
  "g3c4.layer-norm.forward", "g3c4.linear.forward-no-bias",
  "g3c4.linear.forward-no-bias", "g3c4.linear.forward-no-bias",
  "g3c4.heads.split.forward", "g3c4.heads.split.forward",
  "g3c4.heads.split.forward", "g3c4.causal-attention.forward",
  "g3c4.heads.merge.forward", "g3c4.linear.forward-no-bias",
  "residual.forward", "g3c4.layer-norm.forward",
  "g3c4.linear.forward-no-bias", "g3c4.gelu.forward",
  "g3c4.linear.forward-no-bias", "residual.forward",
  "g3c4.layer-norm.forward", "g3c4.linear.forward-no-bias"
};
static int record_prefill;
static size_t prefill_dispatches;
static size_t fail_prefill_at = SIZE_MAX;

#ifdef ET_G3C4_PREFILL1_PRIVATE
static const char *const expected_p1_capabilities[21] = {
  "g3n.embedding-forward", "g3c4.embedding-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward",
  "g3n.linear-forward", "g3n.linear-forward", "g3n.linear-forward",
  "g3n.head-layout-forward", "g3n.head-layout-forward",
  "g3n.head-layout-forward", "g3c4.causal-attention",
  "g3n.head-layout-forward", "g3n.linear-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward",
  "g3n.linear-forward", "kernel.activation", "g3n.linear-forward",
  "g3n.residual-forward", "g3n.layer-norm-forward", "g3n.linear-forward"
};
static const char *const expected_p1_operations[21] = {
  "g3n.embedding.forward", "g3c4.embedding.forward",
  "g3n.residual.forward", "g3n.layer-norm.forward",
  "g3n.linear.forward-no-bias", "g3n.linear.forward-no-bias",
  "g3n.linear.forward-no-bias", "g3n.heads.split.forward",
  "g3n.heads.split.forward", "g3n.heads.split.forward",
  "g3c4.causal-attention.forward", "g3n.heads.merge.forward",
  "g3n.linear.forward-no-bias", "g3n.residual.forward",
  "g3n.layer-norm.forward", "g3n.linear.forward-no-bias",
  "gelu.forward", "g3n.linear.forward-no-bias",
  "g3n.residual.forward", "g3n.layer-norm.forward",
  "g3n.linear.forward-no-bias"
};
static int record_prefill1;
static size_t prefill1_dispatches;
static size_t fail_prefill1_at = SIZE_MAX;
static float prefill1_staged_keys[4];
static float prefill1_staged_values[4];

#ifdef ET_G3C4_PREFILL2_PRIVATE
static const char *const expected_p2_capabilities[21] = {
  "n3k.embedding-forward", "g3c4.embedding-forward",
  "n3k.residual", "kernel.norm",
  "n3k.linear", "n3k.linear", "n3k.linear",
  "n3k.head-layout", "n3k.head-layout", "n3k.head-layout",
  "g3c4.causal-attention", "n3k.head-layout", "n3k.linear",
  "n3k.residual", "kernel.norm", "n3k.linear", "n3k.gelu",
  "n3k.linear", "n3k.residual", "kernel.norm", "n3k.linear"
};
static const char *const expected_p2_operations[21] = {
  "n3k.embedding.forward", "g3c4.embedding.forward",
  "n3k.residual.forward", "layer-norm.forward",
  "n3k.linear.forward-no-bias", "n3k.linear.forward-no-bias",
  "n3k.linear.forward-no-bias", "n3k.heads.split.forward",
  "n3k.heads.split.forward", "n3k.heads.split.forward",
  "g3c4.causal-attention.forward", "n3k.heads.merge.forward",
  "n3k.linear.forward-no-bias", "n3k.residual.forward",
  "layer-norm.forward", "n3k.linear.forward-no-bias",
  "n3k.gelu.forward", "n3k.linear.forward-no-bias",
  "n3k.residual.forward", "layer-norm.forward",
  "n3k.linear.forward-no-bias"
};
static int record_prefill2;
static size_t prefill2_dispatches;
static size_t fail_prefill2_at = SIZE_MAX;
static float prefill2_staged_keys[8];
static float prefill2_staged_values[8];
#endif
#endif

static void check_request_shape(size_t index, const et_kernel_request_v1 *r) {
  static const uint64_t rows[21][6] = {
    {1,3,256,4}, {1,3,4,4}, {1,3,4}, {1,3,4},
    {1,3,4,4}, {1,3,4,4}, {1,3,4,4},
    {1,3,2,2}, {1,3,2,2}, {1,3,2,2},
    {1,2,2,3,4,2}, {1,3,2,2}, {1,3,4,4}, {1,3,4},
    {1,3,4}, {1,3,4,8}, {1,3,8}, {1,3,8,4},
    {1,3,4}, {1,3,4}, {1,3,4,256}
  };
  static const size_t ranks[21] = {
    4,4,3,3,4,4,4,4,4,4,6,4,4,3,3,4,3,4,3,3,4
  };
  CHECK(r->rank == ranks[index]);
  for (size_t dim = 0u; dim < ranks[index]; dim++)
    CHECK(r->shape[dim] == rows[index][dim]);
}

#ifdef ET_G3C4_PREFILL1_PRIVATE
static void check_p1_request_shape(
    size_t index, const et_kernel_request_v1 *request) {
  static const uint64_t rows[21][6] = {
    {1,1,256,4}, {1,1,4,4}, {1,1,4}, {1,1,4},
    {1,1,4,4}, {1,1,4,4}, {1,1,4,4},
    {1,1,2,2}, {1,1,2,2}, {1,1,2,2},
    {1,2,2,1,4,2}, {1,1,2,2}, {1,1,4,4}, {1,1,4},
    {1,1,4}, {1,1,4,8}, {1,1,8}, {1,1,8,4},
    {1,1,4}, {1,1,4}, {1,1,4,256}
  };
  static const size_t ranks[21] = {
    4,4,3,3,4,4,4,4,4,4,6,4,4,3,3,4,3,4,3,3,4
  };
  CHECK(request->rank == ranks[index]);
  for (size_t dimension = 0u; dimension < ranks[index]; dimension++)
    CHECK(request->shape[dimension] == rows[index][dimension]);
}

#ifdef ET_G3C4_PREFILL2_PRIVATE
static void check_p2_request_shape(
    size_t index, const et_kernel_request_v1 *request) {
  static const uint64_t rows[21][6] = {
    {1,2,256,4}, {1,2,4,4}, {1,2,4}, {1,2,4},
    {1,2,4,4}, {1,2,4,4}, {1,2,4,4},
    {1,2,2,2}, {1,2,2,2}, {1,2,2,2},
    {1,2,2,2,4,2}, {1,2,2,2}, {1,2,4,4}, {1,2,4},
    {1,2,4}, {1,2,4,8}, {1,2,8}, {1,2,8,4},
    {1,2,4}, {1,2,4}, {1,2,4,256}
  };
  static const size_t ranks[21] = {
    4,4,3,3,4,4,4,4,4,4,6,4,4,3,3,4,3,4,3,3,4
  };
  CHECK(request->rank == ranks[index]);
  for (size_t dimension = 0u; dimension < ranks[index]; dimension++)
    CHECK(request->shape[dimension] == rows[index][dimension]);
}
#endif
#endif

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
#ifdef ET_G3C4_PREFILL1_PRIVATE
#ifdef ET_G3C4_PREFILL2_PRIVATE
  if (record_prefill2) {
    const size_t index = prefill2_dispatches++;
    CHECK(index < 21u);
    CHECK(strcmp(call->capability, expected_p2_capabilities[index]) == 0);
    CHECK(strcmp(call->request->operation, expected_p2_operations[index]) == 0);
    CHECK(call->request->deterministic == 1u);
    check_p2_request_shape(index, call->request);
    if (index == 10u) {
      const et_kernel_tensor_view_v1 *inputs = call->inputs;
      const int64_t *query = (const int64_t *)inputs[3].data;
      for (size_t position = 0u; position < 2u; position++) {
        CHECK(query[position] == (int64_t)position);
        for (size_t key = 0u; key < 4u; key++) {
          CHECK(((const int64_t *)inputs[4].data)[key] == (int64_t)key);
          CHECK(((const uint8_t *)inputs[5].data)[position * 4u + key] ==
                (key <= position ? 1u : 0u));
        }
      }
      for (size_t head = 0u; head < 2u; head++)
        for (size_t position = 0u; position < 2u; position++)
          for (size_t dimension = 0u; dimension < 2u; dimension++) {
            const size_t cache_index = (head * 4u + position) * 2u + dimension;
            const size_t local_index = (head * 2u + position) * 2u + dimension;
            prefill2_staged_keys[local_index] =
                ((const float *)inputs[1].data)[cache_index];
            prefill2_staged_values[local_index] =
                ((const float *)inputs[2].data)[cache_index];
          }
    }
    if (index == fail_prefill2_at) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.prefill2-cut");
      strcpy(error->message, "injected prefill2 failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
#endif
  if (record_prefill1) {
    const size_t index = prefill1_dispatches++;
    CHECK(index < 21u);
    CHECK(strcmp(call->capability, expected_p1_capabilities[index]) == 0);
    CHECK(strcmp(call->request->operation, expected_p1_operations[index]) == 0);
    CHECK(call->request->deterministic == 1u);
    check_p1_request_shape(index, call->request);
    if (index == 10u) {
      const et_kernel_tensor_view_v1 *inputs = call->inputs;
      CHECK(*(const int64_t *)inputs[3].data == 0);
      for (size_t key = 0u; key < 4u; key++) {
        CHECK(((const int64_t *)inputs[4].data)[key] == (int64_t)key);
        CHECK(((const uint8_t *)inputs[5].data)[key] ==
              (key == 0u ? 1u : 0u));
      }
      for (size_t head = 0u; head < 2u; head++)
        for (size_t dimension = 0u; dimension < 2u; dimension++) {
          const size_t cache_index = head * 8u + dimension;
          const size_t local_index = head * 2u + dimension;
          prefill1_staged_keys[local_index] =
              ((const float *)inputs[1].data)[cache_index];
          prefill1_staged_values[local_index] =
              ((const float *)inputs[2].data)[cache_index];
        }
    }
    if (index == fail_prefill1_at) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.prefill1-cut");
      strcpy(error->message, "injected prefill1 failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
#endif
  if (record_prefill) {
    const size_t index = prefill_dispatches++;
    CHECK(index < 21u);
    CHECK(strcmp(call->capability, expected_capabilities[index]) == 0);
    CHECK(strcmp(call->request->operation, expected_operations[index]) == 0);
    CHECK(call->request->deterministic == 1u);
    check_request_shape(index, call->request);
    if (index == 10u) {
      const et_kernel_tensor_view_v1 *inputs = call->inputs;
      const int64_t *query = (const int64_t *)inputs[3].data;
      const int64_t *keys = (const int64_t *)inputs[4].data;
      const uint8_t *keep = (const uint8_t *)inputs[5].data;
      for (size_t q = 0u; q < 3u; q++) {
        CHECK(query[q] == (int64_t)q);
        for (size_t k = 0u; k < 4u; k++) {
          CHECK(keys[k] == (int64_t)k);
          CHECK(keep[q * 4u + k] == (k <= q ? 1u : 0u));
        }
      }
    }
    if (index == fail_prefill_at) {
      memset(error, 0, sizeof(*error));
      error->category = ET_KERNEL_ERROR_INTERNAL;
      error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
      strcpy(error->operation, "g3c4.prefill3-cut");
      strcpy(error->message, "injected prefill failure");
      return ET_KERNEL_ERROR_INTERNAL;
    }
  }
  return __real_et_kernel_runtime_dispatch(runtime, call, error);
}

static unsigned char identities[14];

static et_g3c4_model_owner_internal *create_owner(void) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static et_g3c4_context_internal *create_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  return context;
}

typedef struct cache_snapshot {
  float keys[16], values[16];
  int64_t length;
  uint8_t keep[4];
} cache_snapshot;

typedef struct binding_snapshot {
  const void *identities[14];
  unsigned char values[4768];
  int64_t tokens[3];
  uint32_t ready;
} binding_snapshot;

static void snapshot_cache(et_a2_kv_cache *cache, cache_snapshot *snapshot) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  memcpy(snapshot->keys, keys->data, sizeof(snapshot->keys));
  memcpy(snapshot->values, values->data, sizeof(snapshot->values));
  snapshot->length = *(const int64_t *)lengths->data;
  memcpy(snapshot->keep, keep->data, sizeof(snapshot->keep));
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
}

static void snapshot_binding(
    const et_g3c4_context_internal *context, binding_snapshot *snapshot) {
  memcpy(snapshot->identities, context->prefill_binding_identities,
         sizeof(snapshot->identities));
  memcpy(snapshot->values, context->prefill_binding_values,
         sizeof(snapshot->values));
  memcpy(snapshot->tokens, context->prefill_tokens, sizeof(snapshot->tokens));
  snapshot->ready = context->prefill_binding_ready;
}

static void check_preserved(
    et_g3c4_context_internal *context, const cache_snapshot *cache,
    const binding_snapshot *binding, const int64_t rng[4]) {
  cache_snapshot actual_cache;
  binding_snapshot actual_binding;
  snapshot_cache(context->cache, &actual_cache);
  snapshot_binding(context, &actual_binding);
  CHECK(memcmp(actual_cache.keys, cache->keys, sizeof(cache->keys)) == 0);
  CHECK(memcmp(actual_cache.values, cache->values,
               sizeof(cache->values)) == 0);
  CHECK(actual_cache.length == cache->length);
  CHECK(memcmp(actual_cache.keep, cache->keep, sizeof(cache->keep)) == 0);
  CHECK(memcmp(actual_binding.identities, binding->identities,
               sizeof(binding->identities)) == 0);
  CHECK(memcmp(actual_binding.values, binding->values,
               sizeof(binding->values)) == 0);
  CHECK(memcmp(actual_binding.tokens, binding->tokens,
               sizeof(binding->tokens)) == 0);
  CHECK(actual_binding.ready == binding->ready);
  CHECK(memcmp(context->generator_rng_words, rng, 4u * sizeof(rng[0])) == 0);
}

static void close_after_abort(et_g3c4_context_internal *context) {
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void prefill_success(
    et_g3c4_context_internal *context, const int64_t tokens[3],
    float logits[256]) {
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  prefill_dispatches = 0u;
  fail_prefill_at = SIZE_MAX;
  record_prefill = 1;
  OK(et_g3c4_private_prefill3_v1(context, tokens, logits));
  record_prefill = 0;
  CHECK(prefill_dispatches == 21u);
}

static void numerical_parity_and_continuation(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[4] = {3, 7, 11, 37};
  float reference[1024], last[256], next[256], sample_logits[1024];
  int64_t speculative = -1, published = -1, saved_rng[4];
  cache_snapshot cache;
  et_g3c4_context_internal *reference_context = create_generator(owner);
  et_g3c4_context_internal *context = create_generator(owner);

  OK(et_g3c4_private_call_acquire_v1(reference_context, 0, 0));
  OK(et_g3c4_private_full_prefix_forward_v1(
      reference_context, tokens, reference));
  OK(et_g3c4_private_call_finish_v1(reference_context));
  OK(et_g3c4_private_generator_close_v1(reference_context));

  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  prefill_success(context, tokens, last);
  CHECK(memcmp(last, reference + 2u * 256u, sizeof(last)) == 0);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  CHECK(context->prefill_binding_ready == 1u);
  CHECK(memcmp(context->prefill_tokens, tokens, 3u * sizeof(tokens[0])) == 0);
  snapshot_cache(context->cache, &cache);
  CHECK(cache.length == 3);
  CHECK(memcmp(cache.keep, (const uint8_t[4]){1,1,1,0}, 4u) == 0);

  memset(sample_logits, 0, sizeof(sample_logits));
  sample_logits[3u * 256u + 37u] = 10.0f;
  OK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative));
  CHECK(speculative == 37);
  OK(et_g3c4_private_token_forward_v1(context, speculative, next));
  CHECK(memcmp(next, reference + 3u * 256u, sizeof(next)) == 0);
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_token_frame_publish_v1(context, &published));
  CHECK(published == 37);
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  CHECK(cache.length == 4);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void replacement_and_dispatch_cuts(
    et_g3c4_model_owner_internal *owner) {
  const int64_t first_tokens[3] = {3, 7, 11};
  const int64_t next_tokens[3] = {13, 17, 19};
  for (size_t cut = 0u; cut < 21u; cut++) {
    float first_logits[256], output[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill_success(context, first_tokens, first_logits);
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t i = 0u; i < 256u; i++) output[i] = -123.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    prefill_dispatches = 0u;
    fail_prefill_at = cut;
    record_prefill = 1;
    CHECK(et_g3c4_private_prefill3_v1(context, next_tokens, output) != 0);
    record_prefill = 0;
    CHECK(prefill_dispatches == cut + 1u);
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
    CHECK(et_g3c4_private_last_error_code_v1() == ET_KERNEL_CODE_PROVIDER_REJECTED);
    for (size_t i = 0u; i < 256u; i++) CHECK(output[i] == -123.0f);
    check_preserved(context, &cache, &binding, rng);
    close_after_abort(context);
  }
  fail_prefill_at = SIZE_MAX;
}

static void allocation_cuts(et_g3c4_model_owner_internal *owner) {
  const int64_t first_tokens[3] = {3, 7, 11};
  const int64_t next_tokens[3] = {13, 17, 19};
  size_t failures = 0u;
  for (size_t allowed = 0u; allowed <= 11u; allowed++) {
    float first_logits[256], output[256];
    int64_t rng[4];
    cache_snapshot cache;
    binding_snapshot binding;
    et_g3c4_context_internal *context = create_generator(owner);
    prefill_success(context, first_tokens, first_logits);
    OK(et_g3c4_private_call_finish_v1(context));
    snapshot_cache(context->cache, &cache);
    snapshot_binding(context, &binding);
    memcpy(rng, context->generator_rng_words, sizeof(rng));
    for (size_t i = 0u; i < 256u; i++) output[i] = -321.0f;
    OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    int64_t status = et_g3c4_private_prefill3_v1(context, next_tokens, output);
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status != 0) {
      failures++;
      CHECK(allowed < 11u);
      CHECK(et_g3c4_private_last_error_code_v1() ==
            ET_KERNEL_CODE_ALLOCATION_FAILED);
      for (size_t i = 0u; i < 256u; i++) CHECK(output[i] == -321.0f);
      check_preserved(context, &cache, &binding, rng);
      close_after_abort(context);
    } else {
      CHECK(allowed == 11u);
      CHECK(context->prefill_binding_ready == 1u);
      CHECK(memcmp(context->prefill_tokens, next_tokens,
                   sizeof(context->prefill_tokens)) == 0);
      OK(et_g3c4_private_call_finish_v1(context));
      OK(et_g3c4_private_generator_close_v1(context));
    }
  }
  CHECK(failures == 11u);
}

static void owned_alias_rejections(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[3] = {3, 7, 11};
  float logits[256];
  int64_t rng[4];
  cache_snapshot cache;
  binding_snapshot binding;
  unsigned char parameter[4096];
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  et_g3c4_context_internal *context = create_generator(owner);

  prefill_success(context, tokens, logits);
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  memcpy(parameter, context->pins.views[10].data, sizeof(parameter));
  CHECK(et_g3c4_private_prefill3_v1(
      context, tokens, (float *)context->pins.views[10].data) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_ARGUMENT);
  CHECK(memcmp(parameter, context->pins.views[10].data,
               sizeof(parameter)) == 0);
  CHECK(et_g3c4_prefill_binding_matches_pins(context));
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill3_v1(
      context, tokens, (float *)(void *)context->cache) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_ARGUMENT);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      context->cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  float *cache_alias = (float *)keys->data;
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill3_v1(context, tokens, cache_alias) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_ARGUMENT);
  check_preserved(context, &cache, &binding, rng);
  close_after_abort(context);
}

static void runtime_stale_and_rejections(
    et_g3c4_model_owner_internal *owner) {
  const int64_t tokens[3] = {3, 7, 11};
  float logits[256], output[256], sample_logits[1024];
  int64_t rng[4], speculative = -1;
  cache_snapshot cache;
  binding_snapshot binding;
  et_g3c4_context_internal *context = create_generator(owner);

  prefill_success(context, tokens, logits);
  OK(et_g3c4_private_call_finish_v1(context));
  snapshot_cache(context->cache, &cache);
  snapshot_binding(context, &binding);
  memcpy(rng, context->generator_rng_words, sizeof(rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  et_m3t_test_k1_fail_after(0u);
  CHECK(et_g3c4_private_prefill3_v1(context, tokens, output) != 0);
  et_m3t_test_k1_allocator_reset();
  CHECK(et_g3c4_private_last_error_code_v1() == ET_KERNEL_CODE_ALLOCATION_FAILED);
  check_preserved(context, &cache, &binding, rng);
  OK(et_g3c4_private_call_abort_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  unsigned char saved = ((unsigned char *)context->pins.views[0].data)[0];
  ((unsigned char *)context->pins.views[0].data)[0] ^= 1u;
  memset(sample_logits, 0, sizeof(sample_logits));
  sample_logits[3u * 256u + 37u] = 10.0f;
  CHECK(et_g3c4_private_token_frame_begin_v1(
      context, sample_logits, &speculative) != 0);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_STALE_BINDING);
  CHECK(speculative == -1);
  ((unsigned char *)context->pins.views[0].data)[0] = saved;
  OK(et_g3c4_private_token_frame_begin_v1(context, sample_logits, &speculative));
  OK(et_g3c4_private_token_frame_abort_v1(context));
  close_after_abort(context);

  context = create_generator(owner);
  for (size_t i = 0u; i < 256u; i++) output[i] = -456.0f;
  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  CHECK(et_g3c4_private_prefill3_v1(context, tokens, output) != 0);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  close_after_abort(context);

  context = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_prefill3_v1(
      context, (const int64_t *)((unsigned char *)context + 1u), output) != 0);
  close_after_abort(context);
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  numerical_parity_and_continuation(owner);
  replacement_and_dispatch_cuts(owner);
  allocation_cuts(owner);
  owned_alias_rejections(owner);
  runtime_stale_and_rejections(owner);
  printf("G3-C4 prefill3 PASS: checks=%zu roles=21 dispatch-cuts=21 allocation-cuts=11\n",
         checks);
  return 0;
}
