/* Link the same private aggregate as the accepted prefix witness, now with
 * EOS, second-frame, terminal and logical-I1 features enabled together. */
#define ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE 1
#define ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE 1
#include "p2_g2_t1_prefix_coordinator_native.c"

static int composer_capture_t4;
static cache_snapshot composer_t4;
int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (composer_capture_t4 && transaction != NULL && *transaction != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    if (et_a2_kv_cache_transaction_view_begin_v1(
          *transaction, 0u, &view, &local) != 0 ||
        et_a2_kv_cache_transaction_view_tensors_v1(
          view, &keys, &values, &lengths, &keep, &local) != 0)
      return -1;
    memcpy(composer_t4.keys, keys->data, sizeof(composer_t4.keys));
    memcpy(composer_t4.values, values->data, sizeof(composer_t4.values));
    composer_t4.length = *(const int64_t *)lengths->data;
    memcpy(composer_t4.keep, keep->data, sizeof(composer_t4.keep));
    if (et_a2_kv_cache_transaction_view_end_v1(&view, &local) != 0)
      return -1;
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(transaction, error);
}

int64_t et_g3c4_p2g2_composer_test_full_prefix_v1(
    void *context_candidate, int64_t prompt0, int64_t prompt1,
    int64_t id0, int64_t id1, int64_t count) {
  et_g3c4_context_internal *actual =
      et_g3c4_admit_idle_call(context_candidate);
  et_g3c4_context_internal *reference;
  cache_snapshot observed, expected;
  float logits[1024];
  const int64_t prefix[4] = {prompt0, prompt1, id0, id1};
  if (actual == NULL || (count != 1 && count != 2) ||
      id0 < 0 || id0 > 255 || (count == 2 && (id1 < 0 || id1 > 255)) ||
      (count == 1 && id1 != 0))
    return -1;
  snapshot_cache(actual->cache, &observed);
  reference = create_generator(actual->owner);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 0, 0) != 0)
    return -1;
  if (count == 1) {
    if (et_g3c4_private_prefill3_v1(reference, prefix, logits) != 0)
      return -1;
    snapshot_cache(reference->cache, &expected);
  } else {
    composer_capture_t4 = 1;
    if (et_g3c4_private_full_prefix_forward_v1(
          reference, prefix, logits) != 0)
      return -1;
    composer_capture_t4 = 0;
    expected = composer_t4;
  }
  if (!coordinator_cache_equal(&observed, &expected) ||
      et_g3c4_private_call_finish_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0)
    return -1;
  return 0;
}

int64_t et_g3c4_p2g2_composer_test_second_oracle_v1(
    void *context_candidate, int64_t selected) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  if (context == NULL || !coordinator_saw_prefix ||
      selected < 0 || selected > 255 || !coordinator_owner_unchanged())
    return -1;
  printf("COMPOSER_SECOND %s %lld %lld",
         coordinator_policy[0] == 0 ? "greedy" : "categorical",
         (long long)coordinator_selected, (long long)selected);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)coordinator_successor_rng[index]);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)context->generator_rng_words[index]);
  for (size_t index = 0u; index < 256u; index++) {
    uint32_t bits;
    memcpy(&bits, &coordinator_next_logits[index], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
  return 0;
}

static int composer_fault_mode;
static int composer_read_fault;
int64_t et_g3c4_p2g2_composer_test_fault_v1(int64_t active) {
  if (active < 0 || active > 3) return -1;
  composer_fault_mode = (int)active;
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_read_fault_v1(int64_t active) {
  if (active < 0 || active > 3) return -1;
  composer_read_fault = (int)active;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *output, void *destination, int64_t bytes) {
  if (composer_read_fault == 1) {
    composer_read_fault = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, destination, bytes);
}

int64_t __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *, int64_t, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *clone, int64_t count, void *destination, int64_t bytes) {
  if (composer_read_fault == 3) {
    int64_t *declared = (int64_t *)destination;
    int64_t result;
    composer_read_fault = 0;
    *declared += 8;
    result = __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
        clone, count, destination, bytes);
    *declared -= 8;
    if (result == 0) abort();
    return result;
  }
  if (composer_read_fault == 2) {
    composer_read_fault = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      clone, count, destination, bytes);
}

int64_t et_g3c4_p2g2_composer_test_clone_count_v1(int64_t state) {
  int64_t count = 0;
  const et_g3c4_transport_header_internal *record;
  if (state != ET_G3C4_CONTEXT_LIVE && state != ET_G3C4_CONTEXT_DEAD)
    return -1;
  for (record = et_g3c4_transport_registry; record != NULL;
       record = record->registry_next)
    if (record->kind == ET_G3C4_TERMINAL_IDS_KIND && record->state == state)
      count++;
  return count;
}

int64_t __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *context, void *input, void *staging, int64_t bytes) {
  int64_t result;
  et_g3c4_output_internal *pending = NULL;
  if (composer_fault_mode == 1) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  result = __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, staging, bytes);
  if (result == 0 &&
      (et_g3c4_pending_output_lookup(context, &pending) != 0 ||
       pending == NULL ||
       et_g3c4_p2g2_coordinator_test_observe_v1(
           context, pending, 2,
           ((et_g3c4_context_internal *)context)->token_frame_candidate) != 0))
    return -1;
  return result;
}

int64_t et_g3c4_p2g2_composer_test_after_abort_v1(void *context) {
  const int64_t expected = coordinator_saw_prefix ? 3 :
      coordinator_precommit_cache_owner != NULL ? 2 : 0;
  return et_g3c4_p2g2_coordinator_test_after_abort_v1(context, expected);
}

int64_t __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *context, void *output, void *staging, int64_t bytes) {
  et_g3c4_context_internal *actual = (et_g3c4_context_internal *)context;
  if (et_g3c4_p2g2_coordinator_test_prefix_v1(
          context, output, actual->p2g2_prefix_token) != 0)
    return -1;
  if (composer_fault_mode == 2) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
      context, output, staging, bytes);
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  if (composer_fault_mode == 3) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, first, first_bytes, second, second_bytes);
}

static et_i64_tensor_borrow *composer_clone_borrow;
int64_t et_g3c4_p2g2_composer_test_clone_lease_v1(
    void *candidate, int64_t active) {
  et_g3c4_terminal_ids_clone_internal *clone =
      et_g3c4_terminal_ids_admit(candidate, 0);
  et_i64_tensor_error error;
  if (clone == NULL || clone->ids == NULL) return -1;
  if (active == 1 && composer_clone_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        clone->ids, &composer_clone_borrow, &error);
  if (active == 0 && composer_clone_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&composer_clone_borrow, &error);
  return -1;
}
