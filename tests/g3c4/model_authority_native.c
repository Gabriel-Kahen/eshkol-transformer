#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#include "../../src/eshkol_transformer/g3c4_model_owner.c"

int64_t et_g3c4_model_test_owner_fail_after_v1(int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_allocation_limit = (size_t)allowed;
  et_g3c4_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_model_test_owner_reset_v1(void) {
  et_g3c4_allocation_limit = SIZE_MAX;
  et_g3c4_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_model_test_owner_registry_count_v1(void) {
  int64_t count = 0;
  for (et_g3c4_model_owner_internal *owner = et_g3c4_owner_registry;
       owner != NULL; owner = owner->registry_next) {
    count++;
  }
  return count;
}

int64_t et_g3c4_model_test_owner_state_v1(void *candidate) {
  for (et_g3c4_model_owner_internal *owner = et_g3c4_owner_registry;
       owner != NULL; owner = owner->registry_next) {
    if (owner == candidate) return owner->state;
  }
  return -1;
}
