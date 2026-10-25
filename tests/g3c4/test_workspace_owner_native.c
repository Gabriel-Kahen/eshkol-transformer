/* Intrusive native source test; the delivered ABI is private. */
#define ET_F32_TENSOR_TESTING 1
#define ET_M3_CALL_TESTING 1
#define ET_G3C4_CONTEXT_TESTING 1
#define ET_G3C4_NATIVE_OWNER_PRIVATE 1
#define ET_G3C4_NATIVE_PINS_PRIVATE 1
#define ET_G3C4_CONTEXT_PRIVATE 1
#define ET_G3C4_ACTIVE_CALL_PRIVATE 1
#define ET_G3C4_GENERATOR_PRIVATE 1
#define ET_G3R_C4_OWNER_TOPOLOGY_PRIVATE 1
#define ET_G3R_C4_OWNER_ASSOCIATION_PRIVATE 1
#define ET_G3R_C4_OWNER_PARAMETER_LEASE_PRIVATE 1
#define ET_G3R_C4_CONTEXT_EXCLUSION_PRIVATE 1
#define ET_G3R_C4_WORKSPACE_PRIVATE 1
#define ET_G3R_C4_WORKSPACE_TESTING 1
#ifdef ET_G3R_C4_WORKSPACE_A2_TU
#include "../../native/a2_kv_cache.c"
#else
#include "../../src/eshkol_transformer/m3_call_f32_integration.c"
#include "../../src/eshkol_transformer/g3c4_model_owner.c"
#include <stdio.h>
#define CHECK(x) do { checks++; if (!(x)) { \
  fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define OK(x) CHECK((x) == 0)
static int checks;
static unsigned char first_ids[14], second_ids[14], first_bits[4768];
static et_g3c4_model_owner_internal *make_owner(unsigned char ids[14], int64_t seed) {
  et_g3c4_model_owner_internal *owner = et_g3c4_private_model_owner_create_seeded_v1(seed);
  CHECK(owner != NULL);
  for (int64_t i = 0; i < 14; i++) OK(et_g3c4_private_model_owner_bind_v1(owner, i, &ids[(size_t)i]));
  for (int64_t i = 0; i < 14; i++) OK(et_g3c4_private_model_owner_initialize_v1(owner, i));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(owner));
  return owner;
}
static void unchanged(et_g3c4_model_owner_internal *owner) {
  size_t at = 0;
  for (size_t i = 0; i < 14; i++) {
    size_t bytes = et_g3c4_parameter_elements(i) * sizeof(uint32_t);
    CHECK(memcmp(first_bits + at, owner->parameters[i]->value->data, bytes) == 0);
    at += bytes;
  }
  CHECK(at == sizeof(first_bits));
}
static void status(int64_t got, int64_t wanted, int64_t code) {
  CHECK(got == wanted);
  CHECK(et_g3c4_private_last_error_category_v1() == wanted);
  CHECK(et_g3c4_private_last_error_code_v1() == code);
}
int main(void) {
  int foreign = 0;
  et_g3c4_model_owner_internal *first = make_owner(first_ids, 1729);
  et_g3c4_model_owner_internal *second = make_owner(second_ids, 1730);
  size_t at = 0;
  for (size_t i = 0; i < 14; i++) {
    size_t bytes = et_g3c4_parameter_elements(i) * sizeof(uint32_t);
    memcpy(first_bits + at, first->parameters[i]->value->data, bytes);
    at += bytes;
  }
  CHECK(at == sizeof(first_bits));
  et_g3c4_private_workspace_test_fail_allocation_once_v1();
  CHECK(et_g3c4_private_workspace_stage_v1(first) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_ALLOCATION);
  CHECK(first->active == NULL && et_g3c4_workspace_staged == NULL);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == 0u);
  unchanged(first);
  et_g3c4_context_internal *idle = (et_g3c4_context_internal *)
      et_g3c4_private_generator_seed_v1(first, 4, 0, INT64_C(0x3f800000),
          256, INT64_C(0x3f800000), 1, -1);
  CHECK(idle != NULL && idle->cache != NULL && first->active == NULL);
  CHECK(et_g3c4_private_workspace_stage_v1(first) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_g3c4_private_generator_close_v1(idle));
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error error;
  OK(et_f32_parameter_value_borrow_begin_v1(first->parameters[0], &borrow, &error));
  CHECK(et_g3c4_private_workspace_stage_v1(first) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(first->active == NULL && et_g3c4_workspace_staged == NULL);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  et_g3c4_workspace_internal *stage = et_g3c4_private_workspace_stage_v1(first);
  CHECK(stage != NULL && first->active == NULL);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == sizeof(*stage));
  int64_t nonce = et_g3c4_private_workspace_staged_nonce_v1(stage);
  CHECK(nonce > 0 && et_g3c4_private_last_error_category_v1() == 0);
  status(et_g3c4_private_workspace_prepare_commit_v1(second, stage, nonce),
      ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  status(et_g3c4_private_workspace_prepare_commit_v1(first, stage, nonce + 1),
      ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  status(et_g3c4_private_workspace_abort_v1(&foreign),
      ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  CHECK(first->active == NULL && et_g3c4_workspace_staged == stage);
  OK(et_g3c4_private_workspace_prepare_commit_v1(first, stage, nonce));
  et_g3c4_private_workspace_test_fail_commit_once_v1();
  status(et_g3c4_private_workspace_commit_v1(first, stage, nonce),
      ET_G3C4_INTERNAL, ET_G3C4_CODE_ALLOCATION);
  CHECK(first->active == NULL && et_g3c4_workspace_staged == stage &&
        et_g3c4_workspace_live == NULL);
  OK(et_g3c4_private_workspace_abort_v1(stage));
  CHECK(first->active == NULL && et_g3c4_workspace_staged == NULL);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == 0u);
  unchanged(first);
  stage = et_g3c4_private_workspace_stage_v1(first);
  CHECK(stage != NULL);
  nonce = et_g3c4_private_workspace_staged_nonce_v1(stage);
  OK(et_g3c4_private_workspace_prepare_commit_v1(first, stage, nonce));
  OK(et_g3c4_private_workspace_commit_v1(first, stage, nonce));
  CHECK(first->active == stage && et_g3c4_workspace_staged == NULL);
  OK(et_g3c4_private_workspace_live_status_v1(first, stage, nonce));
  et_g3c4_workspace_internal *other = et_g3c4_private_workspace_stage_v1(second);
  CHECK(other != NULL);
  int64_t other_nonce = et_g3c4_private_workspace_staged_nonce_v1(other);
  OK(et_g3c4_private_workspace_prepare_commit_v1(second, other, other_nonce));
  OK(et_g3c4_private_workspace_commit_v1(second, other, other_nonce));
  CHECK(second->active == other && first->active == stage);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == 2u * sizeof(*stage));
  OK(et_g3c4_private_workspace_release_v1(second, other, other_nonce));
  CHECK(second->active == NULL && first->active == stage);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == sizeof(*stage));
  status(et_g3c4_private_workspace_live_status_v1(second, stage, nonce),
      ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  status(et_g3c4_private_workspace_live_status_v1(first, stage, nonce + 1),
      ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  status(et_g3c4_private_workspace_binding_status_v1(
      first, stage, nonce, 0, &first_ids[1], first->parameters[0]),
      ET_G3C4_INVALID_STATE, ET_G3C4_CODE_STALE_BINDING);
  for (int64_t i = 0; i < 14; i++)
    OK(et_g3c4_private_workspace_binding_status_v1(first, stage, nonce, i,
        &first_ids[(size_t)i], first->parameters[(size_t)i]));
  CHECK(et_g3c4_private_context_create_v1(first) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  CHECK(et_g3c4_private_generator_seed_v1(first, 5, 0, INT64_C(0x3f800000),
      256, INT64_C(0x3f800000), 1, -1) == NULL);
  CHECK(et_g3c4_private_last_error_category_v1() == ET_G3C4_INVALID_STATE);
  OK(et_f32_parameter_value_borrow_begin_v1(first->parameters[0], &borrow, &error));
  status(et_g3c4_private_workspace_release_v1(first, stage, nonce),
      ET_F32_TENSOR_ERROR_INVALID_STATE, ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(first->active == stage);
  OK(et_f32_tensor_borrow_end_v1(&borrow, &error));
  unchanged(first);
  OK(et_g3c4_private_workspace_release_v1(first, stage, nonce));
  CHECK(first->active == NULL && et_g3c4_workspace_live == NULL);
  CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == 0u);
  status(et_g3c4_private_workspace_live_status_v1(first, stage, nonce),
      ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
  for (int round = 0; round < 32; round++) {
    stage = et_g3c4_private_workspace_stage_v1(first);
    CHECK(stage != NULL);
    int64_t next = et_g3c4_private_workspace_staged_nonce_v1(stage);
    CHECK(next > nonce);
    status(et_g3c4_private_workspace_live_status_v1(first, stage, nonce),
        ET_G3C4_INVALID_ARGUMENT, ET_G3C4_CODE_IDENTITY);
    OK(et_g3c4_private_workspace_prepare_commit_v1(first, stage, next));
    OK(et_g3c4_private_workspace_commit_v1(first, stage, next));
    OK(et_g3c4_private_workspace_release_v1(first, stage, next));
    CHECK(et_g3c4_workspace_live == NULL && first->active == NULL);
    CHECK(et_g3c4_private_workspace_test_retained_bytes_v1() == 0u);
    nonce = next;
  }
  unchanged(first);
  printf("G3-R C4 workspace native PASS checks=%d\n", checks);
  return 0;
}
#endif
