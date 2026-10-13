/* Host-only native M3/O2 composition. No Eshkol constructor is inferred. */
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_O2_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#define ET_G3R_O2_CREATION_TESTING 1
#define ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE 1
#define ET_G3R_COMMITTED_HISTORY_PRIVATE 1
#define ET_G3T_MANUAL_DECODE_PRIVATE 1
#define ET_G3T_MANUAL_P2_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_P1_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_LOGITS_PRIVATE 1
#define ET_G3T_OWNED_TOKEN_INPUT_PRIVATE 1
#define ET_G3T_P2_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_OUTPUT_TEXT_PRIVATE 1
#define ET_G3T_FINAL_PUBLICATION_PRIVATE 1
#define ET_G3T_INPUT_FROM_T1_PRIVATE 1
#define ET_G3T_PREFILL_SAMPLE_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#include "src/eshkol_transformer/g3t_transport.c"
#include "native/o2_optimizer.c"
#include "native/g3r_o2_creation_bridge.h"
#include <stdio.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, \
    "creation bridge:%d: %s\n", __LINE__, #x); return 1; } } while (0)

static int64_t err(void *hold, int64_t index) {
  return et_g3r_private_o2_creation_error_word_v1(hold, index);
}

int main(void) {
  static unsigned char identities[14];
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial && et_m3t_private_initializer_seal_v1(initial) == 0);
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_bind_v1(model, i, &identities[i]) == 0);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_initialize_v1(model, i) == 0);
  CHECK(et_m3t_private_owner_seal_v1(model) == 0);
  CHECK(et_m3t_private_initializer_seal_v1(model->successor) == 0);

  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 o2;
  CHECK(et_o2_optimizer_builder_create_v1(
      14, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT,
      0, 0, UINT32_C(0x3f800000), &builder, &o2) == 0);
  for (size_t i = 0; i < 14; ++i)
    CHECK(et_o2_optimizer_builder_set_v1(
        builder, i, model->p[i], model->handles[i],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
        UINT32_C(0x3f7d70a4), UINT32_C(0x322bcc77), 0, &o2) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &o2) == 0);
  const size_t counts[14] = {16,16,16,16,32,32,4,4,4,4,1024,4,4,8};
  uint32_t gradient[1024];
  for (size_t i = 0; i < 1024; ++i) gradient[i] = UINT32_C(0x3f800000);
  for (size_t i = 0; i < 14; ++i) {
    et_f32_parameter_test_set_gradient_bits_v1(
        model->p[i], gradient, counts[i]);
    et_f32_parameter_test_set_metadata_v1(
        model->p[i], ET_F32_GRADIENT_PRESENT, 1,
        UINT32_C(0x3f800000));
  }
  CHECK(et_o2_optimizer_step_v1(optimizer, &o2) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &o2) == 0);
  uint64_t updates = 0;
  CHECK(et_o2_optimizer_completed_updates_v1(optimizer, &updates, &o2) == 0 &&
        updates == 1);

  void *hold = et_g3r_private_o2_creation_begin_v1(
      (void *)(uintptr_t)1u, optimizer);
  CHECK(!hold && err(NULL, 0) == 1 && optimizer->busy == 0);
  et_g3r_o2_creation_test_fail_shell_once_v1();
  hold = et_g3r_private_o2_creation_begin_v1(model, optimizer);
  CHECK(!hold && err(NULL, 0) == 4 && err(NULL, 3) == 4 &&
        optimizer->busy == 0);

  et_f32_parameter_test_set_gradient_bits_v1(
      model->p[0], gradient, counts[0]);
  et_f32_parameter_test_set_metadata_v1(
      model->p[0], ET_F32_GRADIENT_PRESENT, 1, UINT32_C(0x3f800000));
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(optimizer->busy == 0 && err(NULL, 0) == 2);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &o2) == 0);
  optimizer->busy = 1; /* Owned negative for a concurrently reserved O2. */
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(optimizer->busy == 1 && err(NULL, 0) == 2);
  optimizer->busy = 0;

  /* Allow two genuine moment clones, then deny a later F32 allocation.
   * O2 must drain those partial baselines before the bridge returns NULL. */
  et_f32_test_live_counts_v1 partial_before = {.struct_size =
                                                sizeof(partial_before)};
  et_f32_test_live_counts_v1 partial_after = {.struct_size =
                                               sizeof(partial_after)};
  et_f32_test_live_counts_snapshot_v1(&partial_before);
  et_f32_tensor_test_fail_alloc_after_v1(10);
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == 10);
  CHECK(optimizer->busy == 0 && err(NULL, 0) == 2 &&
        err(NULL, 2) == ET_O2_STATUS_INTERNAL &&
        err(NULL, 3) == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  et_f32_test_live_counts_snapshot_v1(&partial_after);
  CHECK(partial_after.tensors == partial_before.tensors &&
        partial_after.owned_clones == partial_before.owned_clones);
  et_f32_tensor_test_reset_allocator_v1();

  /* The accepted O2 held reader permits a preexisting read-only value borrow;
   * it rejects conflicting writes and O2 busy instead. */
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_error f32;
  CHECK(et_f32_parameter_value_borrow_begin_v1(
      model->p[0], &borrow, &f32) == 0);
  hold = et_g3r_private_o2_creation_begin_v1(model, optimizer);
  CHECK(hold && optimizer->busy == 1);
  CHECK(et_g3r_private_o2_creation_check_v1(hold) == 0);
  CHECK(err(hold, 0) == 0 && err(NULL, 0) == 0);
  CHECK(err((void *)(uintptr_t)1u, 0) == 4 &&
        err((void *)(uintptr_t)1u, 3) == 2);
  CHECK(err(hold, 4) == INT64_MIN && err(NULL, 0) == 0);
  CHECK(et_o2_optimizer_step_v1(optimizer, &o2) == ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &o2) == ET_O2_STATUS_INVALID_STATE);
  et_o2_optimizer_state *ordinary = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &ordinary, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(ordinary == NULL);
  et_f32_test_live_counts_v1 before = {0}, during = {0}, after = {0};
  before.struct_size = during.struct_size = after.struct_size = sizeof(before);
  et_f32_test_live_counts_snapshot_v1(&during);
  CHECK(during.tensors >= 84);
  CHECK(et_g3r_private_o2_creation_end_v1(hold) == 0);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(optimizer->busy == 0 &&
        after.tensors + 28 == during.tensors);
  CHECK(et_g3r_o2_creation_test_tombstone_cleared_v1(hold) == 1);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &f32) == 0);
  CHECK(et_g3r_private_o2_creation_end_v1(hold) == -1);
  CHECK(err(NULL, 0) == 4 && err(NULL, 3) == 2 && err(hold, 0) == 0);

  void *old = model->handles[0];
  model->handles[0] = model->handles[1];
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(optimizer->busy == 0 && err(NULL, 0) == 1);
  model->handles[0] = old;
  et_o2_optimizer *foreign = NULL;
  et_o2_optimizer_builder *other = NULL;
  CHECK(et_o2_optimizer_builder_create_v1(
      14, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT,
      0, 0, UINT32_C(0x3f800000), &other, &o2) == 0);
  for (size_t i = 0; i < 14; ++i)
    CHECK(et_o2_optimizer_builder_set_v1(
        other, i, model->p[13-i], model->handles[13-i],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
        UINT32_C(0x3f7d70a4), UINT32_C(0x322bcc77), 0, &o2) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&other, &foreign, &o2) == 0);
  CHECK(et_g3r_private_o2_creation_begin_v1(model, foreign) == NULL);
  CHECK(foreign->busy == 0 && err(NULL, 0) == 2);

  et_o2_test_fail_alloc_after_v1(0);
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(optimizer->busy == 0 && err(NULL, 0) == 2);
  et_o2_test_reset_failpoints_v1();
  hold = et_g3r_private_o2_creation_begin_v1(model, optimizer);
  CHECK(hold && optimizer->busy == 1);
  et_f32_test_live_counts_snapshot_v1(&before);
  et_f32_tensor *moment = optimizer->entries[0].exp_avg;
  float *bits = (float *)(uintptr_t)et_f32_tensor_test_data_storage_v1(moment);
  CHECK(bits != NULL);
  uint32_t original;
  memcpy(&original, bits, 4);
  uint32_t changed = original ^ 1u;
  memcpy(bits, &changed, 4);
  CHECK(et_g3r_private_o2_creation_check_v1(hold) == -1);
  int64_t first[4];
  for (int i = 0; i < 4; ++i) first[i] = err(hold, i);
  CHECK(first[0] == 2 && err(NULL, 0) == 2);
  CHECK(et_g3r_private_o2_creation_end_v1(hold) == -1);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(optimizer->busy == 0 &&
        after.tensors + 28 == before.tensors);
  CHECK(et_g3r_o2_creation_test_tombstone_cleared_v1(hold) == 1);
  for (int i = 0; i < 4; ++i) CHECK(err(hold, i) == first[i]);
  CHECK(et_g3r_private_o2_creation_end_v1(hold) == -1);
  CHECK(err(NULL, 0) == 4 && err(hold, 0) == 2);
  memcpy(bits, &original, 4);
  hold = et_g3r_private_o2_creation_begin_v1(model, optimizer);
  CHECK(hold && et_g3r_private_o2_creation_end_v1(hold) == 0);

  for (unsigned i = 0; i < 61; ++i) {
    hold = et_g3r_private_o2_creation_begin_v1(model, optimizer);
    CHECK(hold && et_g3r_private_o2_creation_end_v1(hold) == 0);
  }
  CHECK(et_g3r_private_o2_creation_begin_v1(model, optimizer) == NULL);
  CHECK(err(NULL, 0) == 4 && err(NULL, 3) == 3 && optimizer->busy == 0);
  printf("creation O2 bridge PASS: %u checks\n", checks);
  return 0;
}
