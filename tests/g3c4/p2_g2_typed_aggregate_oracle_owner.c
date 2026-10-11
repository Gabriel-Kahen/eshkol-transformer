/* Test-only owner translation unit for the private same-call oracle.
 * Build this instead of g3c4_model_owner.c with the same feature macros.
 * Link --wrap for the four P2/G2 calls and A2 transaction_abort_v1. */
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#if !defined(ET_G3C4_P2_G2_TERMINAL_PRIVATE) || \
    !defined(ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE) || \
    !defined(ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE)
#error "typed aggregate oracle requires the complete private terminal path"
#endif

typedef struct aggregate_cache {
  float keys[16];
  float values[16];
  int64_t length;
  uint8_t keep[4];
} aggregate_cache;

typedef struct aggregate_record {
  et_g3c4_context_internal *context;
  et_g3c4_model_owner_internal *owner;
  et_g3c4_output_internal *output;
  int64_t prompt[2], policy[6], entry_rng[4];
  int64_t first, second, first_after[4], second_before[4], second_after[4];
  float second_logits[256];
  aggregate_cache published;
  int64_t case_id;
  int first_seen, prefix_seen, second_seen, terminal_seen;
} aggregate_record;

static aggregate_record aggregate;
static aggregate_cache aggregate_t4;
static int aggregate_capture_t4, aggregate_t4_captures;

static int aggregate_cache_read(et_a2_kv_cache *cache,
                                aggregate_cache *destination) {
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  if (et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) != 0)
    return -1;
  if (et_a2_kv_cache_read_borrow_layer_v1(
          borrow, 0u, &keys, &values, &lengths, &keep, &error) != 0 ||
      keys == NULL || values == NULL || lengths == NULL || keep == NULL ||
      keys->data == NULL || values->data == NULL ||
      lengths->data == NULL || keep->data == NULL) {
    (void)et_a2_kv_cache_read_borrow_end_v1(&borrow, &error);
    return -1;
  }
  memcpy(destination->keys, keys->data, sizeof(destination->keys));
  memcpy(destination->values, values->data, sizeof(destination->values));
  memcpy(&destination->length, lengths->data, sizeof(destination->length));
  memcpy(destination->keep, keep->data, sizeof(destination->keep));
  return et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0 ? 0 : -1;
}

static int aggregate_cache_equal(const aggregate_cache *a,
                                 const aggregate_cache *b) {
  return a->length == b->length &&
         memcmp(a->keys, b->keys, sizeof(a->keys)) == 0 &&
         memcmp(a->values, b->values, sizeof(a->values)) == 0 &&
         memcmp(a->keep, b->keep, sizeof(a->keep)) == 0;
}

int64_t et_g3c4_p2g2_aggregate_oracle_mark_v1(
    void *candidate, int64_t case_id, int64_t prompt0, int64_t prompt1) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(candidate);
  if (context == NULL || case_id < 0 || case_id > 3 ||
      prompt0 < 0 || prompt0 > 255 || prompt1 < 0 || prompt1 > 255 ||
      (aggregate.context != NULL && !aggregate.terminal_seen)) return -1;
  memset(&aggregate, 0, sizeof(aggregate));
  aggregate.context = context;
  aggregate.owner = context->owner;
  aggregate.case_id = case_id;
  aggregate.prompt[0] = prompt0;
  aggregate.prompt[1] = prompt1;
  aggregate.first = aggregate.second = -1;
  memcpy(aggregate.policy, context->generator_policy,
         sizeof(aggregate.policy));
  memcpy(aggregate.entry_rng, context->generator_rng_words,
         sizeof(aggregate.entry_rng));
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *candidate, void *input, void *carrier, int64_t extent) {
  int64_t status = __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      candidate, input, carrier, extent);
  et_g3c4_context_internal *context = candidate;
  if (status == 0 && context == aggregate.context &&
      !aggregate.first_seen) {
    aggregate.first = context->token_frame_candidate;
    memcpy(aggregate.first_after, context->token_frame_successor,
           sizeof(aggregate.first_after));
    aggregate.first_seen = 1;
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_prefix_commit_v1(
    void *, void *, const void *);
int64_t __wrap_et_g3c4_private_p2g2_prefix_commit_v1(
    void *candidate, void *output, const void *raw) {
  int64_t status = __real_et_g3c4_private_p2g2_prefix_commit_v1(
      candidate, output, raw);
  et_g3c4_context_internal *context = candidate;
  if (status == 0 && context == aggregate.context &&
      aggregate.first_seen && !aggregate.prefix_seen) {
    memcpy(aggregate.second_before, context->generator_rng_words,
           sizeof(aggregate.second_before));
    memcpy(aggregate.second_logits, context->p2g2_prefix_next_logits,
           sizeof(aggregate.second_logits));
    aggregate.prefix_seen = 1;
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *candidate, void *output, void *carrier, int64_t extent) {
  int64_t status = __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
      candidate, output, carrier, extent);
  et_g3c4_context_internal *context = candidate;
  if (status == 0 && context == aggregate.context &&
      aggregate.prefix_seen && !aggregate.second_seen) {
    aggregate.second = context->token_frame_candidate;
    memcpy(aggregate.second_after, context->token_frame_successor,
           sizeof(aggregate.second_after));
    aggregate.second_seen = 1;
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *candidate, void *output, const void *first, int64_t first_extent,
    const void *second, int64_t second_extent) {
  int64_t status = __real_et_g3c4_private_p2g2_terminal_commit_v1(
      candidate, output, first, first_extent, second, second_extent);
  et_g3c4_context_internal *context = candidate;
  if (status == 0 && context == aggregate.context &&
      !aggregate.terminal_seen) {
    aggregate.output = output;
    aggregate.terminal_seen = aggregate_cache_read(
        context->cache, &aggregate.published) == 0 ? 1 : -1;
  }
  return status;
}

int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (aggregate_capture_t4 && transaction != NULL &&
      *transaction != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    if (et_a2_kv_cache_transaction_view_begin_v1(
            *transaction, 0u, &view, &local) != 0 ||
        et_a2_kv_cache_transaction_view_tensors_v1(
            view, &keys, &values, &lengths, &keep, &local) != 0 ||
        keys == NULL || values == NULL || lengths == NULL || keep == NULL ||
        keys->data == NULL || values->data == NULL ||
        lengths->data == NULL || keep->data == NULL) abort();
    memcpy(aggregate_t4.keys, keys->data, sizeof(aggregate_t4.keys));
    memcpy(aggregate_t4.values, values->data, sizeof(aggregate_t4.values));
    memcpy(&aggregate_t4.length, lengths->data, sizeof(aggregate_t4.length));
    memcpy(aggregate_t4.keep, keep->data, sizeof(aggregate_t4.keep));
    if (et_a2_kv_cache_transaction_view_end_v1(&view, &local) != 0) abort();
    aggregate_t4_captures++;
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(transaction, error);
}

static void aggregate_print_row(int64_t step, const float logits[256],
                                int64_t selected, const int64_t before[4],
                                const int64_t after[4]) {
  printf("AGG_ORACLE %lld %lld %s %lld %lld %lld %lld %lld",
         (long long)aggregate.case_id, (long long)step,
         aggregate.policy[0] == 0 ? "greedy" : "categorical",
         (long long)aggregate.policy[1],
         (long long)aggregate.policy[2],
         (long long)aggregate.policy[3],
         (long long)aggregate.policy[5],
         (long long)selected);
  for (size_t i = 0; i < 4; i++) printf(" %lld", (long long)before[i]);
  for (size_t i = 0; i < 4; i++) printf(" %lld", (long long)after[i]);
  for (size_t i = 0; i < 256; i++) {
    uint32_t bits;
    memcpy(&bits, &logits[i], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
}

int64_t et_g3c4_p2g2_aggregate_oracle_verify_v1(
    void *candidate, void *output_candidate, int64_t expected_count) {
  et_g3c4_context_internal *context = candidate;
  et_g3c4_output_internal *output = output_candidate;
  et_g3c4_context_internal *reference = NULL;
  aggregate_cache expected;
  float first_logits[256], second_logits[256], full_logits[1024];
  int64_t p3[3], p4[4];
  int64_t status = -1;
  if (context != aggregate.context || output != aggregate.output ||
      aggregate.terminal_seen != 1 || aggregate.first_seen != 1 ||
      (expected_count != 1 && expected_count != 2) ||
      aggregate.prefix_seen != (expected_count == 2) ||
      aggregate.second_seen != (expected_count == 2) ||
      output->length != expected_count ||
      output->cache_length != expected_count + 2 ||
      aggregate.published.length != expected_count + 2 ||
      output->transport.state != ET_G3C4_OUTPUT_TERMINAL ||
      memcmp(output->rng, context->generator_rng_words,
             sizeof(output->rng)) != 0 ||
      memcmp(output->rng, expected_count == 1 ? aggregate.first_after :
             aggregate.second_after, sizeof(output->rng)) != 0 ||
      (expected_count == 2 && memcmp(aggregate.second_before,
             aggregate.first_after, sizeof(aggregate.first_after)) != 0))
    return -1;

  reference = et_g3c4_private_generator_seed_v1(
      aggregate.owner, aggregate.entry_rng[1], aggregate.policy[0],
      aggregate.policy[1], aggregate.policy[2], aggregate.policy[3],
      aggregate.policy[4], aggregate.policy[5]);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 2, 1) != 0 ||
      et_g3c4_private_prefill2_v1(
          reference, aggregate.prompt, first_logits) != 0 ||
      et_g3c4_private_call_abort_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0)
    return -1;
  reference = NULL;
  p3[0] = aggregate.prompt[0];
  p3[1] = aggregate.prompt[1];
  p3[2] = aggregate.first;
  reference = et_g3c4_private_generator_seed_v1(
      aggregate.owner, aggregate.entry_rng[1], aggregate.policy[0],
      aggregate.policy[1], aggregate.policy[2], aggregate.policy[3],
      aggregate.policy[4], aggregate.policy[5]);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 2, 1) != 0 ||
      et_g3c4_private_prefill3_v1(
          reference, p3, second_logits) != 0 ||
      aggregate_cache_read(reference->cache, &expected) != 0 ||
      et_g3c4_private_call_abort_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0)
    return -1;
  reference = NULL;
  if (expected_count == 2) {
    if (memcmp(second_logits, aggregate.second_logits,
               sizeof(second_logits)) != 0) return -1;
    p4[0] = p3[0]; p4[1] = p3[1];
    p4[2] = p3[2]; p4[3] = aggregate.second;
    reference = et_g3c4_private_generator_seed_v1(
        aggregate.owner, aggregate.entry_rng[1], aggregate.policy[0],
        aggregate.policy[1], aggregate.policy[2], aggregate.policy[3],
        aggregate.policy[4], aggregate.policy[5]);
    if (reference == NULL ||
        et_g3c4_private_call_acquire_v1(reference, 0, 0) != 0)
      return -1;
    aggregate_t4_captures = 0;
    aggregate_capture_t4 = 1;
    status = et_g3c4_private_full_prefix_forward_v1(
        reference, p4, full_logits);
    aggregate_capture_t4 = 0;
    if (status != 0 || aggregate_t4_captures != 1 ||
        et_g3c4_private_call_finish_v1(reference) != 0 ||
        et_g3c4_private_generator_close_v1(reference) != 0)
      return -1;
    expected = aggregate_t4;
  }
  if (!aggregate_cache_equal(&aggregate.published, &expected)) return -1;
  aggregate_print_row(0, first_logits, aggregate.first,
                      aggregate.entry_rng, aggregate.first_after);
  if (expected_count == 2)
    aggregate_print_row(1, second_logits, aggregate.second,
                        aggregate.second_before, aggregate.second_after);
  printf("AGG_A2 %lld %lld\n", (long long)aggregate.case_id,
         (long long)expected_count);
  aggregate.context = NULL;
  return 0;
}
