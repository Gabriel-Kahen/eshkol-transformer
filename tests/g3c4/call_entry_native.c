#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_TESTING 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3C4_PROMPT_T1_BORROW_PRIVATE 1
#define ET_A2_KV_CACHE_TESTING 1

#define et_g3c4_private_call_finish_v1 \
  et_g3c4_call_entry_real_finish_v1
#define et_g3c4_private_call_abort_v1 \
  et_g3c4_call_entry_real_abort_v1

#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

#undef et_g3c4_private_call_finish_v1
#undef et_g3c4_private_call_abort_v1

static int64_t et_g3c4_call_entry_fail_finish;
static int64_t et_g3c4_call_entry_fail_abort;

int64_t et_g3c4_private_call_finish_v1(void *candidate) {
  if (et_g3c4_call_entry_fail_finish) return 1;
  return et_g3c4_call_entry_real_finish_v1(candidate);
}

int64_t et_g3c4_private_call_abort_v1(void *candidate) {
  if (et_g3c4_call_entry_fail_abort) return 1;
  return et_g3c4_call_entry_real_abort_v1(candidate);
}

int64_t et_g3c4_call_entry_test_failstop_v1(int64_t selector) {
  et_g3c4_call_entry_fail_finish = selector == 1;
  et_g3c4_call_entry_fail_abort = selector == 2;
  return 0;
}

int64_t et_g3c4_call_entry_test_owner_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_allocation_limit = (size_t)allowed;
  et_g3c4_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_call_entry_test_owner_reset_v1(void) {
  et_g3c4_allocation_limit = SIZE_MAX;
  et_g3c4_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_call_entry_test_context_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_context_allocation_limit = (size_t)allowed;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_call_entry_test_context_reset_v1(void) {
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_call_entry_test_a2_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_a2_kv_cache_test_fail_alloc_after_v1((size_t)allowed);
  return 0;
}

int64_t et_g3c4_call_entry_test_a2_reset_v1(void) {
  et_a2_kv_cache_test_reset_allocator_v1();
  return 0;
}

int64_t et_g3c4_call_entry_test_acquire_fail_after_v1(int64_t completed) {
  if (completed < 0) et_g3c4_active_call_fail_after = SIZE_MAX;
  else et_g3c4_active_call_fail_after = (size_t)completed;
  return 0;
}

int64_t et_g3c4_call_entry_test_transport_count_v1(int64_t kind) {
  int64_t count = 0;
  et_g3c4_transport_header_internal *item;
  for (item = et_g3c4_transport_registry; item != NULL;
       item = item->registry_next)
    if ((int64_t)item->kind == kind) count++;
  return count;
}

int64_t et_g3c4_call_entry_test_context_field_v1(
    void *candidate, int64_t selector) {
  et_g3c4_context_internal *context = NULL;
  et_g3c4_transport_header_internal *item;
  for (item = et_g3c4_transport_registry; item != NULL;
       item = item->registry_next)
    if ((void *)item == candidate && item->kind == ET_G3C4_CONTEXT_KIND)
      context = (et_g3c4_context_internal *)item;
  if (context == NULL) return -1;
  switch (selector) {
    case 0: return (int64_t)ET_G3C4_CONTEXT_BUSY(context);
    case 1: return (int64_t)context->acquired_mask;
    case 2: return (int64_t)context->pins.held_mask;
    case 3: return context->call_kind;
    case 4: return context->budget;
    case 5: return context->owner->active == context ? 1 : 0;
    case 6: return (int64_t)ET_G3C4_CONTEXT_STATE(context);
    case 7: return (int64_t)context->generator_ready;
    default: return -1;
  }
}

int64_t et_g3c4_call_entry_test_owner_state_v1(void *candidate) {
  et_g3c4_model_owner_internal *owner;
  for (owner = et_g3c4_owner_registry; owner != NULL;
       owner = owner->registry_next)
    if ((void *)owner == candidate) return (int64_t)owner->state;
  return -1;
}

int64_t et_g3c4_call_entry_test_context_bytes_v1(void) {
  return (int64_t)sizeof(et_g3c4_context_internal);
}
