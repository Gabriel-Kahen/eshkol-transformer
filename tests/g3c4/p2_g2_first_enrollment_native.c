/* Test instrumentation for the additive Eshkol reservation witness. */
#include "p2_g2_protected_scope_native.c"

static et_g3c4_context_internal *snapshot_context;
static et_a2_kv_cache *snapshot_cache_owner;
static cache_snapshot entry_cache;
static binding_snapshot entry_binding;
static int64_t entry_rng[4];
static et_g3c4_context_internal *lease_context;
static et_g3c4_output_internal *lease_owner;
static et_i64_tensor_borrow *lease_slot;
static int64_t prefill_calls;
static int64_t sample_calls;

int64_t __wrap_et_g3c4_private_prompt_prefill_v1(
    void *context, void *input, float logits[256]) {
  (void)context;
  (void)input;
  (void)logits;
  ++prefill_calls;
  abort();
}

int64_t __wrap_et_g3c4_private_token_frame_begin_last_v1(
    void *context, const float logits[256], int64_t *selected) {
  (void)context;
  (void)logits;
  (void)selected;
  ++sample_calls;
  abort();
}

int64_t et_g3c4_first_enrollment_test_snapshot_v1(void *candidate) {
  et_g3c4_context_internal *context = et_g3c4_admit_idle_call(candidate);
  if (context == NULL) return -1;
  snapshot_context = context;
  snapshot_cache_owner = context->cache;
  snapshot_cache(context->cache, &entry_cache);
  snapshot_binding(context, &entry_binding);
  memcpy(entry_rng, context->generator_rng_words, sizeof(entry_rng));
  return 0;
}

int64_t et_g3c4_first_enrollment_test_preserved_v1(
    void *candidate, int64_t active) {
  et_g3c4_context_internal *context = active
      ? et_g3c4_admit_active_call(candidate) : et_g3c4_admit_idle_call(candidate);
  if (context == NULL || context != snapshot_context ||
      context->cache != snapshot_cache_owner || !et_g3c4_token_frame_idle(context) ||
      prefill_calls != 0 || sample_calls != 0)
    return 0;
  check_preserved(context, &entry_cache, &entry_binding, entry_rng);
  return 1;
}

int64_t et_g3c4_first_enrollment_test_pending_v1(void *candidate) {
  et_g3c4_context_internal *context = et_g3c4_admit_context(candidate);
  et_g3c4_output_internal *owner = NULL;
  if (context == NULL || et_g3c4_pending_output_lookup(context, &owner) != 0)
    return -1;
  return owner == NULL ? 0 : 1;
}

int64_t et_g3c4_first_enrollment_test_lease_v1(
    void *candidate, void *output_candidate, int64_t acquire) {
  et_g3c4_context_internal *context = et_g3c4_admit_active_call(candidate);
  et_g3c4_output_internal *owner = et_g3c4_admit_output(output_candidate, 0);
  et_i64_tensor_error error;
  if (context == NULL || owner == NULL || owner->parent_ctx != context ||
      owner->transport.state != 0u || owner->ids == NULL)
    return -1;
  if (acquire == 1 && lease_slot == NULL) {
    if (et_i64_tensor_borrow_begin_v1(owner->ids, &lease_slot, &error) != 0)
      return -1;
    lease_context = context;
    lease_owner = owner;
    return 0;
  }
  if (acquire == 0 && context == lease_context && owner == lease_owner &&
      lease_slot != NULL) {
    if (et_i64_tensor_borrow_end_v1(&lease_slot, &error) != 0) return -1;
    lease_context = NULL;
    lease_owner = NULL;
    return 0;
  }
  return -1;
}

int64_t et_g3c4_first_enrollment_test_foreign_release_v1(void *candidate) {
  et_g3c4_context_internal *foreign = et_g3c4_admit_idle_call(candidate);
  if (foreign == NULL || foreign == lease_context || lease_slot == NULL)
    return -1;
  OK(et_g3c4_private_call_acquire_v1(foreign, 2, 2));
  void *owner = et_g3c4_private_output_reserve_v1(foreign, 2);
  CHECK(owner != NULL);
  int64_t released = et_g3c4_first_enrollment_test_lease_v1(foreign, owner, 0);
  CHECK(lease_slot != NULL);
  OK(et_g3c4_private_call_abort_v1(foreign));
  return released == -1 ? 1 : 0;
}

int64_t et_g3c4_first_enrollment_test_fault_v1(int64_t kind, int64_t cut) {
  if (cut < -1) return -1;
  switch (kind) {
    case 0:
      et_g3c4_active_call_fail_after = cut < 0 ? SIZE_MAX : (size_t)cut;
      return 0;
    case 1:
      return cut < 0 ? et_g3c4_output_envelope_test_owner_fail_reset_v1()
                     : et_g3c4_output_envelope_test_owner_fail_after_v1(cut);
    case 2:
      if (cut < 0) et_i64_tensor_test_reset_allocator_v1();
      else et_i64_tensor_test_fail_alloc_after_v1((size_t)cut);
      return 0;
    default:
      return -1;
  }
}
