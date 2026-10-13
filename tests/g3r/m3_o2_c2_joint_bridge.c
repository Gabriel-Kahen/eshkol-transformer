/* Test-only joint bridge: the sealed M3T registry issues the exact roster
 * before the same optimizer may enter its O2 hold. No caller-built roster. */
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_O2_TESTING 1
#include "../../native/o2_optimizer.c"
#include "../../src/eshkol_transformer/m3t_transport.h"

int64_t et_g3r_test_m3_o2_roster_match_v1(
    void *model_owner, void *candidate) {
  et_o2_optimizer *optimizer = et_o2_find_optimizer((et_o2_optimizer *)candidate);
  void *parameters[14];
  const void *identities[14];
  et_o2_error_v1 error;
  if (!optimizer ||
      et_o2_require_optimizer(optimizer, "g3r-test-roster-match", &error) != 0 ||
      optimizer->count != 14u ||
      et_m3t_private_owner_save_roster_v1(
          model_owner, parameters, identities) != 0)
    return 0;
  for (size_t i = 0; i < 14u; ++i)
    if (parameters[i] != optimizer->entries[i].parameter ||
        identities[i] != optimizer->entries[i].p1_handle)
      return 0;
  return 1;
}

void *et_g3r_test_m3_o2_hold_begin_v1(
    void *model_owner, void *candidate) {
  et_o2_optimizer *optimizer = (et_o2_optimizer *)candidate;
  void *raw_parameters[14];
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_o2_g3r_save_hold *hold = NULL;
  et_o2_error_v1 error;
  if (et_m3t_private_owner_save_roster_v1(
          model_owner, raw_parameters, identities) != 0)
    return NULL;
  for (size_t i = 0; i < 14u; ++i)
    parameters[i] = (et_f32_parameter *)raw_parameters[i];
  if (et_g3r_private_o2_save_begin_v1(
          optimizer, parameters, identities, &hold, &error) != 0)
    return NULL;
  return hold;
}

void *et_g3r_test_o2_hold_issue_v1(void *candidate) {
  et_o2_optimizer_state *issued = NULL;
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_state_v1(
          (et_o2_g3r_save_hold *)candidate, &issued, &error) != 0)
    return NULL;
  return issued;
}

int64_t et_g3r_test_o2_hold_match_v1(void *candidate, void *issued) {
  et_o2_error_v1 error;
  return et_g3r_private_o2_save_issued_match_v1(
      (et_o2_g3r_save_hold *)candidate,
      (et_o2_optimizer_state *)issued, &error);
}

int64_t et_g3r_test_o2_hold_end_v1(void *candidate) {
  et_o2_g3r_save_hold *hold = (et_o2_g3r_save_hold *)candidate;
  et_o2_error_v1 error;
  return et_g3r_private_o2_save_end_v1(&hold, &error);
}

int64_t et_g3r_test_o2_live_count_v1(int64_t index) {
  et_o2_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_o2_test_live_counts_snapshot_v1(&counts);
  switch (index) {
  case 0: return (int64_t)counts.optimizers;
  case 1: return (int64_t)counts.live_states;
  case 2: return (int64_t)counts.state_borrows;
  case 3: return (int64_t)counts.owned_state_clones;
  default: return -1;
  }
}

/* Prove the foreign shell really owns equal bits on distinct parameters. */
int64_t et_g3r_test_o2_equal_foreign_v1(void *first, void *second) {
  et_o2_optimizer *left = et_o2_find_optimizer((et_o2_optimizer *)first);
  et_o2_optimizer *right = et_o2_find_optimizer((et_o2_optimizer *)second);
  et_f32_tensor_error error;
  if (!left || !right || left == right || left->count != 14u ||
      right->count != 14u || left->completed_updates != right->completed_updates ||
      left->config.clip_kind != right->config.clip_kind ||
      left->config.clip_max_bits != right->config.clip_max_bits ||
      left->config.schedule_kind != right->config.schedule_kind ||
      left->config.minimum_ratio_bits != right->config.minimum_ratio_bits ||
      left->config.warmup_updates != right->config.warmup_updates ||
      left->config.total_updates != right->config.total_updates)
    return 0;
  for (size_t i = 0; i < 14u; ++i) {
    const et_o2_entry *a = &left->entries[i], *b = &right->entries[i];
    const et_f32_tensor *av = NULL, *bv = NULL;
    int32_t equal = 0;
    if (a->parameter == b->parameter || a->p1_handle == b->p1_handle ||
        memcmp(&a->options, &b->options, sizeof(a->options)) != 0 ||
        et_f32_parameter_value_tensor_v1(a->parameter, &av, &error) != 0 ||
        et_f32_parameter_value_tensor_v1(b->parameter, &bv, &error) != 0 ||
        et_f32_tensor_bits_equal_v1(av, bv, &equal, &error) != 0 || !equal)
      return 0;
    const et_f32_tensor *am[2] = {a->exp_avg, a->exp_avg_sq};
    const et_f32_tensor *bm[2] = {b->exp_avg, b->exp_avg_sq};
    for (size_t j = 0; j < 2u; ++j) {
      equal = 0;
      if (et_f32_tensor_bits_equal_v1(am[j], bm[j], &equal, &error) != 0 ||
          !equal)
        return 0;
    }
  }
  return 1;
}

static struct {
  et_o2_g3r_save_hold *hold;
  et_o2_optimizer *optimizer;
  et_o2_optimizer_state *issued;
  unsigned char hold_shell[sizeof(et_o2_g3r_save_hold)];
  unsigned char optimizer_shell[sizeof(et_o2_optimizer)];
  unsigned char issued_shell[sizeof(et_o2_optimizer_state)];
  et_o2_entry optimizer_entries[14];
  et_o2_state_entry issued_entries[14];
  et_o2_optimizer_state_handle *handles[28];
  et_f32_tensor *clones[28];
  et_o2_optimizer_state_handle handle_shells[28];
  int ready;
} held_bytes;

/* Exact byte witness for the held control shells and registered ledgers. */
int64_t et_g3r_test_o2_held_bytes_v1(int64_t action, void *hold_ptr,
                                      void *optimizer_ptr, void *issued_ptr) {
  et_o2_g3r_save_hold *hold = (et_o2_g3r_save_hold *)hold_ptr;
  et_o2_optimizer *optimizer = (et_o2_optimizer *)optimizer_ptr;
  et_o2_optimizer_state *issued = (et_o2_optimizer_state *)issued_ptr;
  if (action == 0) {
    et_o2_error_v1 error;
    if (et_g3r_private_o2_save_adopt_preflight_v1(
            hold, optimizer, issued, &error) != 0)
      return 0;
    held_bytes.hold = hold;
    held_bytes.optimizer = optimizer;
    held_bytes.issued = issued;
    memcpy(held_bytes.hold_shell, hold, sizeof(*hold));
    memcpy(held_bytes.optimizer_shell, optimizer, sizeof(*optimizer));
    memcpy(held_bytes.issued_shell, issued, sizeof(*issued));
    memcpy(held_bytes.optimizer_entries, optimizer->entries,
           sizeof(held_bytes.optimizer_entries));
    memcpy(held_bytes.issued_entries, issued->entries,
           sizeof(held_bytes.issued_entries));
    memcpy(held_bytes.handles, issued->handles, sizeof(held_bytes.handles));
    memcpy(held_bytes.clones, issued->owned_clones, sizeof(held_bytes.clones));
    for (size_t i = 0; i < 28u; ++i)
      memcpy(&held_bytes.handle_shells[i], issued->handles[i],
             sizeof(held_bytes.handle_shells[i]));
    held_bytes.ready = 1;
    return 1;
  }
  if (action != 1 || !held_bytes.ready || hold != held_bytes.hold ||
      optimizer != held_bytes.optimizer || issued != held_bytes.issued ||
      memcmp(held_bytes.hold_shell, hold, sizeof(*hold)) != 0 ||
      memcmp(held_bytes.optimizer_shell, optimizer, sizeof(*optimizer)) != 0 ||
      memcmp(held_bytes.issued_shell, issued, sizeof(*issued)) != 0 ||
      memcmp(held_bytes.optimizer_entries, optimizer->entries,
             sizeof(held_bytes.optimizer_entries)) != 0 ||
      memcmp(held_bytes.issued_entries, issued->entries,
             sizeof(held_bytes.issued_entries)) != 0 ||
      memcmp(held_bytes.handles, issued->handles, sizeof(held_bytes.handles)) != 0 ||
      memcmp(held_bytes.clones, issued->owned_clones, sizeof(held_bytes.clones)) != 0)
    return 0;
  for (size_t i = 0; i < 28u; ++i)
    if (memcmp(&held_bytes.handle_shells[i], issued->handles[i],
               sizeof(held_bytes.handle_shells[i])) != 0)
      return 0;
  return 1;
}
