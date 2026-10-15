/* Host causal cuts for the feature-private roster-to-O2 hold bridge. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "o2_optimizer_internal.h"
#include "eshkol_transformer/m3t_transport.h"

extern void *et_g3r_private_trainer_o2_save_begin_v1(void *, void *);
extern int64_t et_g3r_private_trainer_o2_save_end_v1(void *);
extern int64_t et_g3r_private_trainer_o2_save_last_source_v1(void);
extern int64_t et_g3r_private_o2_adopter_issued_identity_v1(
    void *, void *, void *, void *);

static int model, optimizer, hold, issued_state, parameter[14], identity[14];
static int model_live = 1, optimizer_busy, native_live;
static int roster_calls, begin_calls, end_calls, owner_calls, issue_calls;

int64_t et_m3t_private_owner_save_roster_v1(
    void *candidate, void *parameters[14], const void *identities[14]) {
  ++roster_calls;
  if (candidate != &model || !model_live) return 2;
  for (int i = 0; i < 14; ++i) {
    parameters[i] = &parameter[i];
    identities[i] = &identity[i];
  }
  return 0;
}

int32_t et_g3r_private_o2_save_begin_v1(
    et_o2_optimizer *candidate, et_f32_parameter *const parameters[14],
    const void *const identities[14], et_o2_g3r_save_hold **output,
    et_o2_error_v1 *error) {
  ++begin_calls;
  if ((void *)candidate != &optimizer || optimizer_busy) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    return ET_O2_STATUS_INVALID_STATE;
  }
  for (int i = 0; i < 14; ++i)
    if ((void *)parameters[i] != &parameter[i] ||
        identities[i] != &identity[i]) {
      error->category = ET_O2_STATUS_INVALID_STATE;
      return ET_O2_STATUS_INVALID_STATE;
    }
  native_live = 1;
  *output = (et_o2_g3r_save_hold *)&hold;
  return 0;
}

int32_t et_g3r_private_o2_save_end_v1(
    et_o2_g3r_save_hold **candidate, et_o2_error_v1 *error) {
  ++end_calls;
  if ((void *)*candidate != &hold || !native_live) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    return ET_O2_STATUS_INVALID_STATE;
  }
  native_live = 0;
  *candidate = NULL;
  return 0;
}

int32_t et_g3r_private_o2_save_owner_match_v1(
    et_o2_g3r_save_hold *candidate, const et_o2_optimizer *owner,
    et_o2_error_v1 *error) {
  ++owner_calls;
  if ((void *)candidate == &hold && (void *)owner == &optimizer && native_live)
    return 0;
  error->category = ET_O2_STATUS_INVALID_STATE;
  return ET_O2_STATUS_INVALID_STATE;
}

int32_t et_g3r_private_o2_save_issued_identity_v1(
    et_o2_g3r_save_hold *candidate, const et_o2_optimizer_state *issued,
    et_o2_error_v1 *error) {
  ++issue_calls;
  if ((void *)candidate == &hold && (void *)issued == &issued_state &&
      native_live)
    return 0;
  error->category = ET_O2_STATUS_INVALID_STATE;
  return ET_O2_STATUS_INVALID_STATE;
}

int main(void) {
  int foreign;
  assert(et_g3r_private_trainer_o2_save_begin_v1(&optimizer, &foreign) == NULL);
  assert(roster_calls == 1 && begin_calls == 0 && !native_live);
  assert(et_g3r_private_trainer_o2_save_last_source_v1() == 1);
  model_live = 0;
  assert(et_g3r_private_trainer_o2_save_begin_v1(&optimizer, &model) == NULL);
  assert(begin_calls == 0 && !native_live);
  model_live = 1;
  optimizer_busy = 1;
  assert(et_g3r_private_trainer_o2_save_begin_v1(&optimizer, &model) == NULL);
  assert(begin_calls == 1 && !native_live);
  assert(et_g3r_private_trainer_o2_save_last_source_v1() == 2);
  optimizer_busy = 0;
  void *published = et_g3r_private_trainer_o2_save_begin_v1(&optimizer, &model);
  assert(published == &hold && native_live && begin_calls == 2);
  assert(et_g3r_private_trainer_o2_save_last_source_v1() == 0);
  assert(et_g3r_private_o2_adopter_issued_identity_v1(
      &hold, &optimizer, &issued_state, &issued_state) == 0);
  assert(owner_calls == 1 && issue_calls == 1);
  assert(et_g3r_private_o2_adopter_issued_identity_v1(
      &hold, &optimizer, &issued_state, &foreign) ==
      ET_O2_STATUS_INVALID_STATE);
  assert(owner_calls == 1 && issue_calls == 1);
  assert(et_g3r_private_o2_adopter_issued_identity_v1(
      &hold, &foreign, &issued_state, &issued_state) ==
      ET_O2_STATUS_INVALID_STATE);
  assert(owner_calls == 2 && issue_calls == 1);
  assert(et_g3r_private_trainer_o2_save_end_v1(published) == 0);
  assert(!native_live && end_calls == 1);
  assert(et_g3r_private_trainer_o2_save_end_v1(published) ==
         ET_O2_STATUS_INVALID_STATE);
  assert(end_calls == 2 && roster_calls == 4);
  puts("G3R-ORIGINAL-O2-BRIDGE-PASS");
  return 0;
}
