#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_O2_TESTING 1
#define ET_F32_TENSOR_TESTING 1
#include "../../include/eshkol_transformer/f32_tensor.h"
static int32_t injected_bits_equal(const et_f32_tensor *,
                                   const et_f32_tensor *, int32_t *,
                                   et_f32_tensor_error *);
#define et_f32_tensor_bits_equal_v1 injected_bits_equal
#include "../../native/o2_optimizer.c"
#undef et_f32_tensor_bits_equal_v1
#include <stdio.h>

static size_t compare_calls;
static int fault_issued_compare;
static int32_t injected_bits_equal(const et_f32_tensor *left,
                                   const et_f32_tensor *right, int32_t *equal,
                                   et_f32_tensor_error *error) {
  if (fault_issued_compare && ++compare_calls == 29u) {
    memset(error, 0, sizeof(*error));
    error->category = ET_F32_TENSOR_ERROR_INTERNAL;
    error->code = ET_F32_TENSOR_CODE_PROVIDER_REJECTED;
    return -1;
  }
  return et_f32_tensor_bits_equal_v1(left, right, equal, error);
}

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
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) == 0);
  uint32_t *option_fields[5] = {
      &state->entries[0].options.learning_rate_bits,
      &state->entries[0].options.beta1_bits,
      &state->entries[0].options.beta2_bits,
      &state->entries[0].options.epsilon_bits,
      &state->entries[0].options.weight_decay_bits};
  for (size_t i = 0; i < 5; ++i) {
    uint32_t before = *option_fields[i];
    *option_fields[i] ^= 1u;
    CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) != 0);
    *option_fields[i] = before;
  }
  uint32_t before_clip = state->config.clip_kind;
  state->config.clip_kind ^= 1u;
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) != 0);
  state->config.clip_kind = before_clip;
  state->completed_updates++;
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) != 0);
  state->completed_updates--;
  for (size_t kind = 0; kind < 2; ++kind) {
    et_f32_tensor *issued = state->owned_clones[kind];
    float *data = (float *)(uintptr_t)et_f32_tensor_test_data_storage_v1(issued);
    uint32_t before = 0, changed;
    CHECK(data != NULL);
    memcpy(&before, data, sizeof(before));
    changed = before ^ 1u;
    memcpy(data, &changed, sizeof(changed));
    CHECK(et_g3r_private_o2_save_check_v1(hold, &o2) == 0);
    CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) != 0);
    memcpy(data, &before, sizeof(before));
  }
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) == 0);
  compare_calls = 0;
  fault_issued_compare = 1;
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) != 0);
  CHECK(compare_calls == 29u && o2.category == ET_O2_STATUS_INTERNAL &&
        o2.code == ET_F32_TENSOR_CODE_PROVIDER_REJECTED);
  fault_issued_compare = 0;
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) == 0);
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
