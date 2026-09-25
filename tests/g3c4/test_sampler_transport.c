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

static int record_dispatch;
static int force_dispatch_failure;
static int expected_categorical;
static const float *expected_full_logits;
static const int64_t *expected_rng;
static size_t dispatch_count;

int32_t __wrap_et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *error) {
  if (record_dispatch) {
    const et_kernel_tensor_view_v1 *inputs = call->inputs;
    const et_kernel_tensor_view_v1 *outputs = call->outputs;
    dispatch_count++;
    CHECK(dispatch_count == 1u);
    CHECK(strcmp(call->capability,
                 expected_categorical ? "g3s.categorical" : "g3s.greedy") == 0);
    CHECK(strcmp(call->request->operation,
                 expected_categorical ? "g3s.categorical.forward" :
                                        "g3s.greedy.forward") == 0);
    CHECK(call->request->rank == 2u);
    CHECK(call->request->shape[0] == 1u &&
          call->request->shape[1] == 256u);
    CHECK(call->request->deterministic == 1u);
    CHECK(call->input_count == (expected_categorical ? 5u : 2u));
    CHECK(call->output_count == 2u);
    CHECK(inputs[0].data != expected_full_logits + 3u * 256u);
    CHECK(memcmp(inputs[0].data, expected_full_logits + 3u * 256u,
                 256u * sizeof(float)) == 0);
    CHECK(memcmp(inputs[expected_categorical ? 4u : 1u].data,
                 expected_rng, 4u * sizeof(int64_t)) == 0);
    CHECK(outputs[0].data != NULL && outputs[1].data != NULL);
    if (expected_categorical) {
      uint32_t bits;
      memcpy(&bits, inputs[1].data, sizeof(bits));
      CHECK(bits == UINT32_C(0x3f800000));
      CHECK(*(const int64_t *)inputs[2].data == 256);
      memcpy(&bits, inputs[3].data, sizeof(bits));
      CHECK(bits == UINT32_C(0x3f800000));
    }
  }
  if (record_dispatch && force_dispatch_failure) {
    memset(error, 0, sizeof(*error));
    error->category = ET_KERNEL_ERROR_INTERNAL;
    error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
    strcpy(error->operation, "g3s.test-cut");
    strcpy(error->message, "injected sampler dispatch failure");
    return ET_KERNEL_ERROR_INTERNAL;
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
    et_g3c4_model_owner_internal *owner, int64_t mode) {
  et_g3c4_context_internal *context =
      et_g3c4_private_generator_seed_v1(
          owner, 7, mode, INT64_C(0x3f800000), 256,
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
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) == 0);
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

static void prepare_logits(float logits[1024]) {
  for (size_t index = 0u; index < 1024u; index++) logits[index] = 0.0f;
  logits[5] = 1000.0f;
  logits[3u * 256u + 37u] = 10.0f;
}

static void expect_unchanged(
    int64_t token, const int64_t successor[4],
    const int64_t context_rng[4], et_g3c4_context_internal *context) {
  CHECK(token == -17);
  for (size_t index = 0u; index < 4u; index++)
    CHECK(successor[index] == INT64_C(-23));
  CHECK(memcmp(context_rng, context->generator_rng_words,
               4u * sizeof(int64_t)) == 0);
  check_cache_empty(context->cache);
}

static void greedy_cases(et_g3c4_model_owner_internal *owner) {
  float logits[1024];
  int64_t token = -17;
  int64_t successor[4] = {-23, -23, -23, -23};
  et_g3c4_context_internal *context = create_generator(owner, 0);
  int64_t saved_rng[4];
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  prepare_logits(logits);
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));

  expected_categorical = 0;
  expected_full_logits = logits;
  expected_rng = saved_rng;
  dispatch_count = 0u;
  record_dispatch = 1;
  OK(et_g3c4_private_sample_last_v1(
      context, logits, &token, successor));
  record_dispatch = 0;
  CHECK(dispatch_count == 1u);
  CHECK(token == 37);
  CHECK(memcmp(successor, saved_rng, sizeof(successor)) == 0);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  check_cache_empty(context->cache);

  token = -17;
  for (size_t index = 0u; index < 4u; index++) successor[index] = -23;
  dispatch_count = 0u;
  force_dispatch_failure = 1;
  record_dispatch = 1;
  CHECK(et_g3c4_private_sample_last_v1(
            context, logits, &token, successor) != 0);
  record_dispatch = 0;
  force_dispatch_failure = 0;
  CHECK(dispatch_count == 1u);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_K1);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_KERNEL_ERROR_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_KERNEL_CODE_PROVIDER_REJECTED);
  expect_unchanged(token, successor, saved_rng, context);

  token = -17;
  for (size_t index = 0u; index < 4u; index++) successor[index] = -23;
  logits[3u * 256u] = NAN;
  CHECK(et_g3c4_private_sample_last_v1(
            context, logits, &token, successor) != 0);
  expect_unchanged(token, successor, saved_rng, context);
  logits[3u * 256u] = 0.0f;

  CHECK(et_g3c4_private_sample_last_v1(
            context, logits, &context->generator_rng_words[0], successor) != 0);
  CHECK(memcmp(context->generator_rng_words, saved_rng, sizeof(saved_rng)) == 0);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void categorical_cases(et_g3c4_model_owner_internal *owner) {
  float logits[1024] = {0};
  int64_t first_token = -17, second_token = -17;
  int64_t first_successor[4] = {-23, -23, -23, -23};
  int64_t second_successor[4] = {-23, -23, -23, -23};
  et_g3c4_context_internal *context = create_generator(owner, 1);
  int64_t saved_rng[4];
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  expected_categorical = 1;
  expected_full_logits = logits;
  expected_rng = saved_rng;
  dispatch_count = 0u;
  record_dispatch = 1;
  OK(et_g3c4_private_sample_last_v1(
      context, logits, &first_token, first_successor));
  record_dispatch = 0;
  CHECK(dispatch_count == 1u);
  CHECK(first_token >= 0 && first_token <= 255);
  CHECK(first_successor[0] == 1 && first_successor[1] == 7);
  CHECK(first_successor[2] == 1 && first_successor[3] == 0);
  CHECK(memcmp(saved_rng, context->generator_rng_words, sizeof(saved_rng)) == 0);

  OK(et_g3c4_private_sample_last_v1(
      context, logits, &second_token, second_successor));
  CHECK(second_token == first_token);
  CHECK(memcmp(second_successor, first_successor,
               sizeof(first_successor)) == 0);
  CHECK(memcmp(saved_rng, context->generator_rng_words, sizeof(saved_rng)) == 0);
  check_cache_empty(context->cache);

  context->generator_rng_words[2] = -1;
  context->generator_rng_words[3] = -1;
  int64_t exhausted_rng[4];
  memcpy(exhausted_rng, context->generator_rng_words, sizeof(exhausted_rng));
  first_token = -17;
  for (size_t index = 0u; index < 4u; index++) first_successor[index] = -23;
  CHECK(et_g3c4_private_sample_last_v1(
            context, logits, &first_token, first_successor) != 0);
  expect_unchanged(first_token, first_successor, exhausted_rng, context);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

static void call_kind_rejection(et_g3c4_model_owner_internal *owner) {
  float logits[1024] = {0};
  int64_t token = -17;
  int64_t successor[4] = {-23, -23, -23, -23};
  et_g3c4_context_internal *context = create_generator(owner, 0);
  int64_t saved_rng[4];
  memcpy(saved_rng, context->generator_rng_words, sizeof(saved_rng));
  OK(et_g3c4_private_call_acquire_v1(context, 0, 0));
  CHECK(et_g3c4_private_sample_last_v1(
            context, logits, &token, successor) != 0);
  expect_unchanged(token, successor, saved_rng, context);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  greedy_cases(owner);
  categorical_cases(owner);
  call_kind_rejection(owner);
  printf("G3-C4 sampler transport PASS: checks=%zu dispatches=1 modes=2\n",
         checks);
  return 0;
}
