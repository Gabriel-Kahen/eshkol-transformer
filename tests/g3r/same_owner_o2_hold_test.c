#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_O2_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#include "../../native/o2_optimizer.c"
#include <stdio.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, \
    "same-owner O2:%d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void) {
  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_f32_parameter *parameters[14] = {0};
  const void *identities[14] = {0};
  et_f32_tensor *initial[14] = {0};
  et_o2_error_v1 o2;
  et_f32_tensor_error f32;
  const uint64_t shape[1] = {2};
  CHECK(et_o2_optimizer_builder_create_v1(
      14, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT,
      0, 0, UINT32_C(0x3f800000), &builder, &o2) == 0);
  for (size_t i = 0; i < 14; ++i) {
    CHECK(et_f32_tensor_create_v1(1, shape, &initial[i], &f32) == 0);
    CHECK(et_f32_parameter_create_v1(initial[i], &parameters[i], &f32) == 0);
    identities[i] = parameters[i];
    CHECK(et_f32_parameter_bind_identity_v1(
        parameters[i], identities[i], &f32) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
        builder, i, parameters[i], identities[i],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
        UINT32_C(0x3f7d70a4), UINT32_C(0x322bcc77), 0, &o2) == 0);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, &o2) == 0);
  et_o2_g3r_save_hold *hold = NULL;
  et_o2_test_fail_alloc_after_v1(0);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) != 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  et_o2_test_reset_failpoints_v1();
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) == 0);
  CHECK(hold != NULL && optimizer->busy == 1);
  CHECK(et_o2_optimizer_step_v1(optimizer, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  et_o2_optimizer_state *ordinary = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &ordinary, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(ordinary == NULL);
  et_f32_tensor *p1 = NULL;
  CHECK(et_f32_parameter_value_snapshot_v1(parameters[0], &p1, &f32) == 0);
  CHECK(et_f32_tensor_destroy_v1(&p1, &f32) == 0);
  et_o2_optimizer_state *state = NULL;
  et_o2_test_fail_alloc_after_v1(0);
  CHECK(et_g3r_private_o2_save_state_v1(hold, &state, &o2) != 0);
  CHECK(state == NULL && optimizer->busy == 1);
  et_o2_test_reset_failpoints_v1();
  CHECK(et_g3r_private_o2_save_state_v1(hold, &state, &o2) == 0);
  CHECK(state != NULL && state->count == 14);
  CHECK(et_g3r_private_o2_save_check_v1(hold, &o2) == 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &o2) == 0);
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) == 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) != 0);
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &ordinary, &o2) == 0);
  CHECK(et_o2_optimizer_state_release_v1(ordinary, &o2) == 0);
  identities[0] = parameters[1];
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) != 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  identities[0] = parameters[0];
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) == 0);
  et_f32_tensor *saved = optimizer->entries[0].exp_avg;
  float *bits = (float *)(uintptr_t)et_f32_tensor_test_data_storage_v1(saved);
  CHECK(bits != NULL);
  uint32_t before;
  memcpy(&before, bits, sizeof(before));
  uint32_t after = before ^ UINT32_C(1);
  memcpy(bits, &after, sizeof(after));
  CHECK(et_g3r_private_o2_save_check_v1(hold, &o2) != 0);
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) != 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  memcpy(bits, &before, sizeof(before));
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) == 0);
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) == 0);
  for (size_t i = 0; i < 61; ++i) {
    CHECK(et_g3r_private_o2_save_begin_v1(
        optimizer, parameters, identities, &hold, &o2) == 0);
    CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) == 0);
  }
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) ==
      ET_O2_STATUS_UNSUPPORTED);
  CHECK(hold == NULL && optimizer->busy == 0);
  printf("same-owner O2 hold PASS: %u checks\n", checks);
  return 0;
}
