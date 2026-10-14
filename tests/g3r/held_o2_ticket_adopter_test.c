#include "o2_optimizer_internal.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int64_t et_g3r_private_o2_adopter_owner_match_v2(int64_t, void *);
void *et_g3r_private_o2_adopter_issue_v2(int64_t);
int64_t et_g3r_private_o2_adopter_issued_match_v2(
    int64_t, void *, void *, void *);
int64_t et_o2_private_last_error_category_v1(void);

static int original_optimizer, equal_other_optimizer;
static int issued_state = 7, equal_other_state = 7;
static int64_t live_ticket = 41;
static unsigned owner_calls, issue_calls, issued_match_calls;

int32_t et_g3r_private_o2_save_owner_match_v2(
    int64_t ticket, const et_o2_optimizer *optimizer, et_o2_error_v1 *error) {
  ++owner_calls;
  if (ticket == live_ticket &&
      (const void *)optimizer == &original_optimizer) return 0;
  memset(error, 0, sizeof(*error));
  error->category = ET_O2_STATUS_INVALID_STATE;
  error->code = ticket == live_ticket ? ET_O2_CODE_OWNER_CONFLICT :
                                        ET_O2_CODE_INVALID_HANDLE;
  return ET_O2_STATUS_INVALID_STATE;
}
int32_t et_g3r_private_o2_save_state_v2(
    int64_t ticket, et_o2_optimizer_state **state, et_o2_error_v1 *error) {
  ++issue_calls;
  assert(*state == NULL);
  if (ticket == live_ticket) {
    *state = (et_o2_optimizer_state *)&issued_state;
    return 0;
  }
  memset(error, 0, sizeof(*error));
  error->category = ET_O2_STATUS_INVALID_STATE;
  error->code = ET_O2_CODE_INVALID_HANDLE;
  return ET_O2_STATUS_INVALID_STATE;
}
int32_t et_g3r_private_o2_save_issued_match_v2(
    int64_t ticket, const et_o2_optimizer_state *state,
    et_o2_error_v1 *error) {
  ++issued_match_calls;
  if (ticket == live_ticket && (const void *)state == &issued_state) return 0;
  memset(error, 0, sizeof(*error));
  error->category = ET_O2_STATUS_INVALID_STATE;
  error->code = ET_O2_CODE_INVALID_HANDLE;
  return ET_O2_STATUS_INVALID_STATE;
}

int main(void) {
  assert(issued_state == equal_other_state);
  assert(et_g3r_private_o2_adopter_owner_match_v2(
      live_ticket, &original_optimizer) == 0);
  assert(et_g3r_private_o2_adopter_owner_match_v2(
      live_ticket, &equal_other_optimizer) == ET_O2_STATUS_INVALID_STATE);
  assert(et_g3r_private_o2_adopter_issue_v2(live_ticket) == &issued_state);
  assert(et_g3r_private_o2_adopter_issued_match_v2(
      live_ticket, &original_optimizer, &issued_state,
      &issued_state) == 0);
  const unsigned before = issued_match_calls;
  assert(et_g3r_private_o2_adopter_issued_match_v2(
      live_ticket, &original_optimizer, &issued_state,
      &equal_other_state) == ET_O2_STATUS_INVALID_STATE);
  assert(issued_match_calls == before);
  assert(et_o2_private_last_error_category_v1() ==
         ET_O2_STATUS_INVALID_STATE);
  assert(et_g3r_private_o2_adopter_issued_match_v2(
      live_ticket, &equal_other_optimizer, &issued_state,
      &issued_state) == ET_O2_STATUS_INVALID_STATE);
  live_ticket = 42; /* A later live hold cannot revive the old ticket. */
  assert(et_g3r_private_o2_adopter_owner_match_v2(
      41, &original_optimizer) == ET_O2_STATUS_INVALID_STATE);
  assert(et_g3r_private_o2_adopter_issue_v2(41) == NULL);
  assert(et_g3r_private_o2_adopter_issued_match_v2(
      41, &original_optimizer, &issued_state,
      &issued_state) == ET_O2_STATUS_INVALID_STATE);
  assert(et_g3r_private_o2_adopter_owner_match_v2(
      0, &original_optimizer) == ET_O2_STATUS_INVALID_STATE);
  assert(owner_calls == 7 && issue_calls == 2 && issued_match_calls == 1);
  puts("held O2 ticket adopter PASS");
  return 0;
}
