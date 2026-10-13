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
  et_o2_optimizer_state *ordinary = NULL;
  unsigned char optimizer_before[sizeof(*optimizer)];
  unsigned char entries_before[14 * sizeof(*optimizer->entries)];
  memcpy(optimizer_before, optimizer, sizeof(optimizer_before));
  memcpy(entries_before, optimizer->entries, sizeof(entries_before));
  const size_t initial_tombstones = et_o2_g3r_save_tombstones;
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities,
      (et_o2_g3r_save_hold **)(void *)&optimizer->completed_updates,
      &o2) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold,
      (et_o2_error_v1 *)optimizer->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold,
      (et_o2_error_v1 *)optimizer) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(hold == NULL && et_o2_g3r_save_tombstones == initial_tombstones &&
        memcmp(optimizer, optimizer_before, sizeof(optimizer_before)) == 0 &&
        memcmp(optimizer->entries, entries_before, sizeof(entries_before)) == 0);
  et_o2_test_fail_alloc_after_v1(0);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) != 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  et_o2_test_reset_failpoints_v1();
  const size_t tombstones_before_partial = et_o2_g3r_save_tombstones;
  et_f32_tensor_test_fail_alloc_after_v1(4);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) != 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  CHECK(et_o2_g3r_save_tombstones == tombstones_before_partial);
  et_f32_tensor_test_reset_allocator_v1();
  union {
    et_o2_error_v1 error;
    et_o2_g3r_save_hold *slot;
  } begin_alias = {0};
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &begin_alias.slot,
      &begin_alias.error) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(begin_alias.slot == NULL && optimizer->busy == 0 &&
        et_o2_g3r_save_tombstones == tombstones_before_partial);
  CHECK(et_g3r_private_o2_save_check_v1(
      (et_o2_g3r_save_hold *)(uintptr_t)1, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  et_o2_g3r_save_hold *foreign = (et_o2_g3r_save_hold *)(uintptr_t)1;
  CHECK(et_g3r_private_o2_save_end_v1(&foreign, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  CHECK(foreign == (et_o2_g3r_save_hold *)(uintptr_t)1);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold, &o2) == 0);
  CHECK(hold != NULL && optimizer->busy == 1);
  et_o2_g3r_save_hold *first_hold = hold;
  et_o2_g3r_save_hold *second_hold = NULL;
  unsigned char shell_before[sizeof(*hold)];
  memcpy(shell_before, hold, sizeof(shell_before));
  CHECK(et_g3r_private_o2_save_check_v1(
      hold, (et_o2_error_v1 *)hold) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_check_v1(
      hold, (et_o2_error_v1 *)&hold->optimizer) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(hold, shell_before, sizeof(shell_before)) == 0);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &hold->next, &o2) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(hold, shell_before, sizeof(shell_before)) == 0);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &second_hold,
      (et_o2_error_v1 *)hold) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(hold, shell_before, sizeof(shell_before)) == 0);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &second_hold, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  CHECK(second_hold == NULL && hold == first_hold && optimizer->busy == 1);
  memcpy(entries_before, optimizer->entries, sizeof(entries_before));
  CHECK(et_g3r_private_o2_save_check_v1(
      hold, (et_o2_error_v1 *)optimizer->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_end_v1(
      &hold, (et_o2_error_v1 *)optimizer->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_end_v1(
      (et_o2_g3r_save_hold **)(void *)&optimizer->completed_updates,
      &o2) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, (et_o2_optimizer_state **)(void *)&optimizer->completed_updates,
      &o2) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, &ordinary, (et_o2_error_v1 *)optimizer->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(hold == first_hold && optimizer->busy == 1u &&
        optimizer->completed_updates == 0u &&
        memcmp(optimizer->entries, entries_before, sizeof(entries_before)) == 0);
  CHECK(et_o2_optimizer_step_v1(optimizer, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  ordinary = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &ordinary, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  CHECK(ordinary == NULL);
  et_f32_tensor *p1 = NULL;
  CHECK(et_f32_parameter_value_snapshot_v1(parameters[0], &p1, &f32) == 0);
  CHECK(et_f32_tensor_destroy_v1(&p1, &f32) == 0);
  et_o2_optimizer_state *state = NULL;
  union {
    et_o2_error_v1 error;
    et_o2_optimizer_state *slot;
  } state_alias = {0};
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, &state_alias.slot, &state_alias.error) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(state_alias.slot == NULL && optimizer->busy == 1);
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, &state, (et_o2_error_v1 *)hold) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(state == NULL && optimizer->busy == 1);
  et_o2_test_fail_alloc_after_v1(0);
  CHECK(et_g3r_private_o2_save_state_v1(hold, &state, &o2) != 0);
  CHECK(state == NULL && optimizer->busy == 1);
  et_o2_test_reset_failpoints_v1();
  CHECK(et_g3r_private_o2_save_state_v1(hold, &state, &o2) == 0);
  CHECK(state != NULL && state->count == 14);
  CHECK(et_g3r_private_o2_save_check_v1(hold, &o2) == 0);
  CHECK(et_g3r_private_o2_save_issued_match_v1(hold, state, &o2) == 0);
  unsigned char state_before[sizeof(*state)];
  unsigned char state_entries_before[14 * sizeof(*state->entries)];
  unsigned char handles_before[28 * sizeof(*state->handles)];
  unsigned char clones_before[28 * sizeof(*state->owned_clones)];
  unsigned char handle_before[sizeof(*state->handles[0])];
  memcpy(state_before, state, sizeof(state_before));
  memcpy(state_entries_before, state->entries, sizeof(state_entries_before));
  memcpy(handles_before, state->handles, sizeof(handles_before));
  memcpy(clones_before, state->owned_clones, sizeof(clones_before));
  memcpy(handle_before, state->handles[0], sizeof(handle_before));
  CHECK(et_g3r_private_o2_save_check_v1(
      hold, (et_o2_error_v1 *)state->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)state->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)state) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)state->handles) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)state->owned_clones) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)state->handles[0]) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_end_v1(
      &hold, (et_o2_error_v1 *)state->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, &ordinary, (et_o2_error_v1 *)state->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_begin_v1(
      optimizer, parameters, identities, &second_hold,
      (et_o2_error_v1 *)state->entries) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_state_v1(
      hold, (et_o2_optimizer_state **)(void *)&state->completed_updates,
      &o2) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(hold == first_hold && optimizer->busy == 1u &&
        memcmp(state, state_before, sizeof(state_before)) == 0 &&
        memcmp(state->entries, state_entries_before,
               sizeof(state_entries_before)) == 0 &&
        memcmp(state->handles, handles_before, sizeof(handles_before)) == 0 &&
        memcmp(state->owned_clones, clones_before, sizeof(clones_before)) == 0 &&
        memcmp(state->handles[0], handle_before, sizeof(handle_before)) == 0);
  memcpy(shell_before, hold, sizeof(shell_before));
  CHECK(et_g3r_private_o2_save_issued_match_v1(
      hold, state, (et_o2_error_v1 *)hold) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_end_v1(
      &hold, (et_o2_error_v1 *)hold) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(et_g3r_private_o2_save_end_v1(&hold->next, &o2) ==
      ET_O2_STATUS_INVALID_ARGUMENT);
  union {
    et_o2_error_v1 error;
    et_o2_g3r_save_hold *slot;
  } end_alias = {.slot = hold};
  CHECK(et_g3r_private_o2_save_end_v1(
      &end_alias.slot, &end_alias.error) == ET_O2_STATUS_INVALID_ARGUMENT);
  CHECK(end_alias.slot == hold && optimizer->busy == 1);
  CHECK(memcmp(hold, shell_before, sizeof(shell_before)) == 0);
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
  state = NULL;
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) == 0);
  CHECK(hold == NULL && optimizer->busy == 0);
  CHECK(et_g3r_private_o2_save_check_v1(first_hold, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  CHECK(et_g3r_private_o2_save_state_v1(first_hold, &state, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  CHECK(state == NULL);
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
  optimizer->config.total_updates++;
  CHECK(et_g3r_private_o2_save_check_v1(hold, &o2) != 0);
  CHECK(et_g3r_private_o2_save_end_v1(&hold, &o2) ==
      ET_O2_STATUS_INVALID_STATE);
  CHECK(o2.code == ET_O2_CODE_OWNER_CONFLICT);
  CHECK(hold == NULL && optimizer->busy == 0);
  optimizer->config.total_updates--;
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
