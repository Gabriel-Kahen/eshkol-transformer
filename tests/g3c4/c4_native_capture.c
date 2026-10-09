/* Test-only observation of the real private C4 owner and numerical routes. */
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
#define ET_G3C4_PREFILL1_PRIVATE 1
#define ET_G3C4_PREFILL2_PRIVATE 1
#define ET_G3C4_PROMPT_T1_BORROW_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(test) do { if (!(test)) { \
  fprintf(stderr, "C4 capture rejected at %s:%d: %s\n", \
          __FILE__, __LINE__, #test); exit(1); \
} } while (0)

static const uint64_t parameter_shapes[14][2] = {
    {4,4}, {4,4}, {4,4}, {4,4}, {4,8}, {8,4}, {4,0}, {4,0},
    {4,0}, {4,0}, {256,4}, {4,0}, {4,0}, {4,4}};
static unsigned char identities[3][14];
static int target_t4_abort;
static unsigned target_abort_count, outside_abort_count;
static int32_t target_abort_result;
static uint32_t target_keys[16], target_values[16];

static uint32_t bits(float value) {
  uint32_t word;
  REQUIRE(isfinite(value));
  memcpy(&word, &value, sizeof(word));
  return word;
}

static void check_view(const et_kernel_tensor_view_v1 *view,
                       const char *dtype, size_t rank,
                       const uint64_t *shape, size_t bytes) {
  REQUIRE(view != NULL && view->struct_size >= ET_KERNEL_TENSOR_VIEW_V1_0_SIZE);
  REQUIRE(view->data != NULL && view->byte_length == bytes);
  REQUIRE(view->dtype != NULL && strcmp(view->dtype, dtype) == 0);
  REQUIRE(view->device != NULL && strcmp(view->device, "cpu") == 0);
  REQUIRE(view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  REQUIRE(view->offset_bytes == 0u && view->rank == rank && view->shape != NULL);
  for (size_t dimension = 0u; dimension < rank; dimension++)
    REQUIRE(view->shape[dimension] == shape[dimension]);
}

/* A2 exposes capacity four. Only active P positions enter the transcript. */
static void copy_active_cache(const et_kernel_tensor_view_v1 *keys,
                              const et_kernel_tensor_view_v1 *values,
                              const et_kernel_tensor_view_v1 *lengths,
                              const et_kernel_tensor_view_v1 *keep,
                              int64_t prefix, uint32_t out_keys[16],
                              uint32_t out_values[16]) {
  const uint64_t kv_shape[4] = {1,2,4,2};
  const uint64_t length_shape[1] = {1};
  const uint64_t keep_shape[2] = {1,4};
  const float *k, *v;
  const uint8_t *mask;
  REQUIRE(prefix >= 1 && prefix <= 4);
  check_view(keys, "f32", 4u, kv_shape, 16u * sizeof(float));
  check_view(values, "f32", 4u, kv_shape, 16u * sizeof(float));
  check_view(lengths, "i64", 1u, length_shape, sizeof(int64_t));
  check_view(keep, "bool", 2u, keep_shape, 4u);
  REQUIRE(*(const int64_t *)lengths->data == prefix);
  mask = (const uint8_t *)keep->data;
  for (size_t position = 0u; position < 4u; position++)
    REQUIRE(mask[position] == (uint8_t)(position < (size_t)prefix));
  k = (const float *)keys->data;
  v = (const float *)values->data;
  for (size_t head = 0u; head < 2u; head++)
    for (size_t position = 0u; position < (size_t)prefix; position++)
      for (size_t dimension = 0u; dimension < 2u; dimension++) {
        const size_t source = (head * 4u + position) * 2u + dimension;
        const size_t target = (head * (size_t)prefix + position) * 2u + dimension;
        out_keys[target] = bits(k[source]);
        out_values[target] = bits(v[source]);
      }
}

int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  if (target_t4_abort) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    REQUIRE(++target_abort_count == 1u && slot != NULL && *slot != NULL);
    REQUIRE(et_a2_kv_cache_transaction_view_begin_v1(
        *slot, 0u, &view, &local) == 0);
    REQUIRE(et_a2_kv_cache_transaction_view_tensors_v1(
        view, &keys, &values, &lengths, &keep, &local) == 0);
    copy_active_cache(keys, values, lengths, keep, 4,
                      target_keys, target_values);
    REQUIRE(et_a2_kv_cache_transaction_view_end_v1(&view, &local) == 0);
    target_abort_result = __real_et_a2_kv_cache_transaction_abort_v1(slot, error);
    return target_abort_result;
  }
  outside_abort_count++;
  return __real_et_a2_kv_cache_transaction_abort_v1(slot, error);
}

static void print_words(const char *tag, const uint32_t *words, size_t count) {
  printf("%s %zu", tag, count);
  for (size_t index = 0u; index < count; index++) printf(" %08" PRIx32, words[index]);
  putchar('\n');
}

static et_g3c4_model_owner_internal *create_owner(int64_t seed, size_t slot) {
  et_g3c4_model_owner_internal *owner =
      et_g3c4_private_model_owner_create_seeded_v1(seed);
  REQUIRE(owner != NULL && slot < 3u);
  for (int64_t index = 0; index < 14; index++)
    REQUIRE(et_g3c4_private_model_owner_bind_v1(
        owner, index, &identities[slot][(size_t)index]) == 0);
  for (int64_t index = 0; index < 14; index++)
    REQUIRE(et_g3c4_private_model_owner_initialize_v1(owner, index) == 0);
  REQUIRE(et_g3c4_private_model_owner_prepare_seal_v1(owner) == 0);
  REQUIRE(et_g3c4_private_model_owner_commit_seal_v1(owner) == 0);
  return owner;
}

static void snapshot_model(et_g3c4_model_owner_internal *owner, int64_t seed,
                           uint32_t image[1192], int64_t state[4]) {
  size_t offset = 0u;
  et_f32_tensor_error error;
  for (size_t index = 0u; index < 14u; index++) {
    et_f32_tensor_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *view = NULL;
    const size_t rank = parameter_shapes[index][1] ? 2u : 1u;
    const size_t count = parameter_shapes[index][0] *
                         (rank == 2u ? parameter_shapes[index][1] : 1u);
    REQUIRE(et_f32_parameter_value_borrow_begin_v1(
        owner->parameters[index], &borrow, &error) == 0);
    REQUIRE(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
    check_view(view, "f32", rank, parameter_shapes[index], count * sizeof(float));
    REQUIRE(offset + count <= 1192u);
    for (size_t element = 0u; element < count; element++)
      image[offset + element] = bits(((const float *)view->data)[element]);
    offset += count;
    REQUIRE(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  }
  REQUIRE(offset == 1192u);
  for (int64_t index = 0; index < 4; index++)
    state[(size_t)index] = et_g3c4_private_model_owner_word_v1(owner, index + 4);
  REQUIRE(state[0] == 1 && state[1] == seed && state[2] == 292 && state[3] == 0);
}

static void require_model_unchanged(et_g3c4_model_owner_internal *owner,
                                    int64_t seed, const uint32_t image[1192],
                                    const int64_t state[4]) {
  uint32_t current_image[1192];
  int64_t current_state[4];
  snapshot_model(owner, seed, current_image, current_state);
  REQUIRE(memcmp(current_image, image, 1192u * sizeof(uint32_t)) == 0);
  REQUIRE(memcmp(current_state, state, 4u * sizeof(int64_t)) == 0);
}

static void print_model(int64_t seed, const uint32_t image[1192],
                        const int64_t state[4]) {
  printf("MODEL %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 "\n",
         seed, state[0], state[1], state[2], state[3]);
  print_words("PARAMETERS", image, 1192u);
}

static et_g3c4_context_internal *new_context(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *context = et_g3c4_private_generator_seed_v1(
      owner, 7, 0, INT64_C(0x3f800000), 256, INT64_C(0x3f800000), 1, -1);
  REQUIRE(context != NULL);
  return context;
}

static void capture_prefill(et_g3c4_model_owner_internal *owner,
                            const int64_t tokens[4], size_t prefix,
                            uint32_t logits[256], uint32_t keys[16],
                            uint32_t values[16], int need_cache) {
  et_g3c4_context_internal *context = new_context(owner);
  float output[256];
  REQUIRE(et_g3c4_private_call_acquire_v1(context, 2, 1) == 0);
  if (prefix == 1u)
    REQUIRE(et_g3c4_private_prefill1_v1(context, tokens[0], output) == 0);
  else if (prefix == 2u)
    REQUIRE(et_g3c4_private_prefill2_v1(context, tokens, output) == 0);
  else {
    REQUIRE(prefix == 3u);
    REQUIRE(et_g3c4_private_prefill3_v1(context, tokens, output) == 0);
  }
  for (size_t index = 0u; index < 256u; index++) logits[index] = bits(output[index]);
  if (need_cache) {
    et_a2_kv_cache_read_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *k = NULL, *v = NULL, *lengths = NULL, *keep = NULL;
    et_kernel_error error;
    REQUIRE(et_a2_kv_cache_read_borrow_begin_v1(context->cache, &borrow, &error) == 0);
    REQUIRE(et_a2_kv_cache_read_borrow_layer_v1(
        borrow, 0u, &k, &v, &lengths, &keep, &error) == 0);
    copy_active_cache(k, v, lengths, keep, (int64_t)prefix, keys, values);
    REQUIRE(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  }
  REQUIRE(et_g3c4_private_call_finish_v1(context) == 0);
  REQUIRE(et_g3c4_private_generator_close_v1(context) == 0);
}

static void capture_case(et_g3c4_model_owner_internal *owner, int64_t seed,
                         size_t prefix, const int64_t tokens[4]) {
  uint32_t logits[1024], keys[16] = {0}, values[16] = {0};
  REQUIRE(prefix >= 1u && prefix <= 4u);
  printf("CASE %" PRId64 " %zu", seed, prefix);
  for (size_t index = 0u; index < prefix; index++) {
    REQUIRE(tokens[index] >= 0 && tokens[index] <= 255);
    printf(" %" PRId64, tokens[index]);
  }
  putchar('\n');
  if (prefix < 4u) {
    for (size_t row = 1u; row <= prefix; row++)
      capture_prefill(owner, tokens, row, logits + (row - 1u) * 256u,
                      keys, values, row == prefix);
  } else {
    et_g3c4_context_internal *context = new_context(owner);
    float output[1024];
    const unsigned before = outside_abort_count;
    REQUIRE(et_g3c4_private_call_acquire_v1(context, 0, 0) == 0);
    target_abort_count = 0u;
    target_abort_result = -1;
    target_t4_abort = 1;
    REQUIRE(et_g3c4_private_full_prefix_forward_v1(context, tokens, output) == 0);
    target_t4_abort = 0;
    REQUIRE(target_abort_count == 1u && target_abort_result == 0);
    REQUIRE(outside_abort_count == before);
    memcpy(keys, target_keys, sizeof(keys));
    memcpy(values, target_values, sizeof(values));
    for (size_t index = 0u; index < 1024u; index++) logits[index] = bits(output[index]);
    REQUIRE(et_g3c4_private_call_finish_v1(context) == 0);
    REQUIRE(et_g3c4_private_generator_close_v1(context) == 0);
    puts("T4ABORT 1 0");
  }
  print_words("LOGITS", logits, prefix * 256u);
  print_words("KEYS", keys, prefix * 4u);
  print_words("VALUES", values, prefix * 4u);
  puts("ENDCASE");
}

int main(void) {
  static const int64_t seeds[3] = {0, 1729, INT64_MAX};
  size_t model_count, case_count;
  char end[16];
  REQUIRE(scanf("C4CAP1 %zu", &model_count) == 1 && model_count == 3u);
  puts("C4CAP1");
  for (size_t model = 0u; model < 3u; model++) {
    int64_t seed;
    et_g3c4_model_owner_internal *owner;
    uint32_t image[1192];
    int64_t state[4];
    REQUIRE(scanf(" MODEL %" SCNd64 " %zu", &seed, &case_count) == 2);
    REQUIRE(seed == seeds[model] && case_count > 0u && case_count <= 32u);
    owner = create_owner(seed, model);
    snapshot_model(owner, seed, image, state);
    print_model(seed, image, state);
    for (size_t row = 0u; row < case_count; row++) {
      size_t prefix;
      int64_t tokens[4] = {0, 0, 0, 0};
      require_model_unchanged(owner, seed, image, state);
      REQUIRE(scanf(" CASE %zu", &prefix) == 1 && prefix >= 1u && prefix <= 4u);
      for (size_t index = 0u; index < prefix; index++)
        REQUIRE(scanf(" %" SCNd64, &tokens[index]) == 1);
      capture_case(owner, seed, prefix, tokens);
      require_model_unchanged(owner, seed, image, state);
    }
  }
  REQUIRE(scanf(" %15s", end) == 1 && strcmp(end, "END") == 0);
  REQUIRE(scanf(" %15s", end) == EOF);
  puts("ENDCAPTURE");
  REQUIRE(fflush(stdout) == 0);
  return 0;
}
