/* Test-only bridge: one native O2 provider for the Eshkol adoption/C2 gate.
 * The M3 roster issuer is outside this bounded seam; read the actual
 * registered optimizer entries here, then exercise the accepted O2 hold. */
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_O2_TESTING 1
#include "../../native/o2_optimizer.c"

void *et_g3r_test_o2_hold_begin_v1(void *candidate) {
  et_o2_optimizer *optimizer = (et_o2_optimizer *)candidate;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_o2_g3r_save_hold *hold = NULL;
  et_o2_error_v1 error;
  if (et_o2_find_optimizer(optimizer) != optimizer ||
      optimizer->count != 14u)
    return NULL;
  for (size_t i = 0; i < 14u; ++i) {
    parameters[i] = optimizer->entries[i].parameter;
    identities[i] = optimizer->entries[i].p1_handle;
  }
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
