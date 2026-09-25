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

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <math.h>
#include <stdio.h>

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

typedef struct expected_step {
  const char *capability;
  const char *operation;
  size_t rank;
  uint64_t shape[6];
} expected_step;

static const expected_step expected[21] = {
  {"g3c4.embedding-forward", "g3c4.embedding.forward", 4u, {1,4,256,4}},
  {"g3c4.embedding-forward", "g3c4.embedding.forward", 4u, {1,4,4,4}},
  {"g3c4.residual", "g3c4.residual.forward", 3u, {1,4,4}},
  {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3u, {1,4,4}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,4}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,4}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,4}},
  {"g3c4.head-layout", "g3c4.heads.split.forward", 4u, {1,4,2,2}},
  {"g3c4.head-layout", "g3c4.heads.split.forward", 4u, {1,4,2,2}},
  {"g3c4.head-layout", "g3c4.heads.split.forward", 4u, {1,4,2,2}},
  {"g3c4.causal-attention", "g3c4.causal-attention.forward", 6u,
   {1,2,2,4,4,2}},
  {"g3c4.head-layout", "g3c4.heads.merge.forward", 4u, {1,4,2,2}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,4}},
  {"g3c4.residual", "g3c4.residual.forward", 3u, {1,4,4}},
  {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3u, {1,4,4}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,8}},
  {"g3c4.gelu", "g3c4.gelu.forward", 3u, {1,4,8}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,8,4}},
  {"g3c4.residual", "g3c4.residual.forward", 3u, {1,4,4}},
  {"g3c4.layer-norm", "g3c4.layer-norm.forward", 3u, {1,4,4}},
  {"g3c4.linear", "g3c4.linear.forward-no-bias", 4u, {1,4,4,256}},
};

static int record_dispatch;
static size_t dispatch_count;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_dispatch) {
    CHECK(dispatch_count < 21u);
    const expected_step *step = &expected[dispatch_count++];
    CHECK(strcmp(call->capability, step->capability) == 0);
    CHECK(strcmp(call->request->operation, step->operation) == 0);
    CHECK(call->request->rank == step->rank);
    if (memcmp(call->request->shape, step->shape,
               step->rank * sizeof(step->shape[0])) != 0) {
      fprintf(stderr, "shape mismatch at step %zu\n", dispatch_count - 1u);
      CHECK(0);
    }
    CHECK(strcmp(call->request->dtype, "f32") == 0);
    CHECK(strcmp(call->request->device, "cpu") == 0);
    CHECK(call->request->deterministic == 1u);
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
  for (size_t role = 0u; role < 3u; role++) {
    static const size_t gamma_indices[3] = {7u, 9u, 12u};
    float *gamma = owner->parameters[gamma_indices[role]]->value->data;
    for (size_t index = 0u; index < 4u; index++) gamma[index] = 1.0f;
  }
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}

static et_g3c4_context_internal *create_generator(
    et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context =
      et_g3c4_private_generator_seed_v1(
          owner, 7, 0, INT64_C(0x3f800000), 256,
          INT64_C(0x3f800000), 1, -1);
  CHECK(context != NULL);
  return context;
}

static void check_cache_empty(et_a2_kv_cache *cache) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL;
  const et_kernel_tensor_view_v1 *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL;
  const et_kernel_tensor_view_v1 *keep = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
            cache, &borrow, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
            borrow, 0u, &keys, &values, &lengths, &keep, &error) == 0);
  CHECK(*(const int64_t *)lengths->data == 0);
  for (size_t index = 0u; index < 16u; index++) {
    CHECK(((const float *)keys->data)[index] == 0.0f);
    CHECK(((const float *)values->data)[index] == 0.0f);
  }
  for (size_t index = 0u; index < 4u; index++)
    CHECK(((const uint8_t *)keep->data)[index] == 0u);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
}

static void fill(float output[1024], float value) {
  for (size_t index = 0u; index < 1024u; index++) output[index] = value;
}

static int all_equal(const float output[1024], float value) {
  for (size_t index = 0u; index < 1024u; index++)
    if (output[index] != value) return 0;
  return 1;
}

int main(void) {
  const int64_t tokens[4] = {1, 2, 3, 4};
  const int64_t changed_tokens[4] = {4, 3, 2, 1};
  const int64_t invalid_tokens[4] = {1, 2, 3, 256};
  float first[1024], second[1024], changed[1024], guarded[1024];
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context = create_generator(owner);
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));

  record_dispatch = 1;
  dispatch_count = 0u;
  OK(et_g3c4_private_full_prefix_forward_v1(context, tokens, first));
  record_dispatch = 0;
  CHECK(dispatch_count == 21u);
  for (size_t index = 0u; index < 1024u; index++) CHECK(isfinite(first[index]));
  check_cache_empty(context->cache);

  record_dispatch = 1;
  dispatch_count = 0u;
  OK(et_g3c4_private_full_prefix_forward_v1(context, tokens, second));
  record_dispatch = 0;
  CHECK(dispatch_count == 21u);
  CHECK(memcmp(first, second, sizeof(first)) == 0);

  record_dispatch = 1;
  dispatch_count = 0u;
  OK(et_g3c4_private_full_prefix_forward_v1(
      context, changed_tokens, changed));
  record_dispatch = 0;
  CHECK(dispatch_count == 21u);
  CHECK(memcmp(first, changed, sizeof(first)) != 0);
  check_cache_empty(context->cache);

  fill(guarded, 12345.0f);
  record_dispatch = 1;
  dispatch_count = 0u;
  CHECK(et_g3c4_private_full_prefix_forward_v1(
            context, invalid_tokens, guarded) != 0);
  record_dispatch = 0;
  CHECK(dispatch_count == 0u);
  CHECK(all_equal(guarded, 12345.0f));
  check_cache_empty(context->cache);

  size_t failure_cuts = 0u;
  for (size_t allowed = 0u; allowed < 16u; allowed++) {
    fill(guarded, 12345.0f);
    et_a2_kv_cache_test_fail_alloc_after_v1(allowed);
    record_dispatch = 1;
    dispatch_count = 0u;
    int64_t status = et_g3c4_private_full_prefix_forward_v1(
        context, tokens, guarded);
    record_dispatch = 0;
    et_a2_kv_cache_test_reset_allocator_v1();
    if (status == 0) {
      CHECK(dispatch_count == 21u);
      CHECK(memcmp(first, guarded, sizeof(first)) == 0);
      break;
    }
    failure_cuts++;
    CHECK(dispatch_count == 10u);
    CHECK(all_equal(guarded, 12345.0f));
    CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
    CHECK(et_g3c4_private_last_error_category_v1() ==
          ET_KERNEL_ERROR_INTERNAL);
    CHECK(et_g3c4_private_last_error_code_v1() ==
          ET_KERNEL_CODE_ALLOCATION_FAILED);
    check_cache_empty(context->cache);
  }
  CHECK(failure_cuts >= 2u);
  OK(et_g3c4_private_call_finish_v1(context));

  OK(et_g3c4_private_call_acquire_v1(context, 1, 0));
  fill(guarded, 12345.0f);
  CHECK(et_g3c4_private_full_prefix_forward_v1(
            context, tokens, guarded) != 0);
  CHECK(all_equal(guarded, 12345.0f));
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));

  printf("G3-C4 fixed full-prefix forward PASS: checks=%zu steps=21 cuts=%zu\n",
         checks, failure_cuts);
  return 0;
}
