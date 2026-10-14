/* Causal provider-contract fixture; the compiled joint aggregate remains a
 * separate gate. Mock functions enforce the reviewed M3/G3/O2 C signatures. */
#include "g3r_same_owner_hold_bridge.h"
#include "o2_optimizer_internal.h"
#include "eshkol_transformer/g3t_transport.h"
#include "eshkol_transformer/m3t_transport.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int model, generator, optimizer, equal_optimizer, g3_hold, o2_hold;
static int parameters[14], identities[14], g3_releases, o2_releases;
static int fail_o2_begin, fail_o2_check, fail_g3_end, forge_roster;
static int64_t g3_category, g3_code, m3_category, m3_code;

int64_t et_m3t_private_last_error_category_v1(void) { return m3_category; }
int64_t et_m3t_private_last_error_code_v1(void) { return m3_code; }
int64_t et_g3t_private_last_error_category_v1(void) { return g3_category; }
int64_t et_g3t_private_last_error_code_v1(void) { return g3_code; }
int64_t et_m3t_private_owner_save_roster_v1(
    void *owner, void *out[14], const void *ids[14]) {
  if (owner != &model) { m3_category = 2; m3_code = 7; return 2; }
  for (int i = 0; i < 14; ++i) {
    out[i] = &parameters[i]; ids[i] = &identities[i];
  }
  if (forge_roster) ids[0] = &identities[1];
  return 0;
}
void *et_g3r_private_g3t_save_begin_v1(
    void *owner, void *const roster[14], const void *const ids[14]) {
  if (owner != &generator) { g3_category = 2; g3_code = 2; return NULL; }
  for (int i = 0; i < 14; ++i)
    if (roster[i] != &parameters[i] || ids[i] != &identities[i]) {
      g3_category = 2; g3_code = 3; return NULL;
    }
  return &g3_hold;
}
int64_t et_g3r_private_g3t_save_check_v1(void *hold) {
  assert(hold == &g3_hold); return 0;
}
int64_t et_g3r_private_g3t_save_end_v1(void *hold) {
  assert(hold == &g3_hold); ++g3_releases;
  if (fail_g3_end) { g3_category = 2; g3_code = 9; return 2; }
  return 0;
}
int32_t et_g3r_private_o2_save_begin_v1(
    et_o2_optimizer *owner, et_f32_parameter *const roster[14],
    const void *const ids[14], et_o2_g3r_save_hold **out,
    et_o2_error_v1 *error) {
  assert(*out == NULL);
  if ((void *)owner != &optimizer || fail_o2_begin) {
    if (fail_o2_begin == 2) return ET_O2_STATUS_INVALID_ARGUMENT;
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_OWNER_CONFLICT;
    return ET_O2_STATUS_INVALID_STATE;
  }
  for (int i = 0; i < 14; ++i)
    if ((void *)roster[i] != &parameters[i] || ids[i] != &identities[i]) {
      error->category = ET_O2_STATUS_INVALID_STATE;
      error->code = ET_O2_CODE_OWNER_CONFLICT;
      return ET_O2_STATUS_INVALID_STATE;
    }
  *out = (et_o2_g3r_save_hold *)&o2_hold;
  return 0;
}
int32_t et_g3r_private_o2_save_owner_match_v1(
    et_o2_g3r_save_hold *hold, const et_o2_optimizer *owner,
    et_o2_error_v1 *error) {
  assert((void *)hold == &o2_hold);
  if ((const void *)owner != &optimizer) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_OWNER_CONFLICT;
    return ET_O2_STATUS_INVALID_STATE;
  }
  return 0;
}
int32_t et_g3r_private_o2_save_check_v1(
    et_o2_g3r_save_hold *hold, et_o2_error_v1 *error) {
  assert((void *)hold == &o2_hold);
  if (fail_o2_check) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_INVALID_MOMENT;
    return ET_O2_STATUS_INVALID_STATE;
  }
  return 0;
}
int32_t et_g3r_private_o2_save_end_v1(
    et_o2_g3r_save_hold **slot, et_o2_error_v1 *error) {
  assert((void *)*slot == &o2_hold);
  int32_t status = et_g3r_private_o2_save_check_v1(*slot, error);
  ++o2_releases; *slot = NULL; return status;
}

int main(void) {
  assert(!et_g3r_private_same_owner_holds_begin_v1(
      &generator, &equal_optimizer, &optimizer));
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 1);
  assert(!et_g3r_private_same_owner_holds_begin_v1(
      &equal_optimizer, &model, &optimizer));
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 2);
  forge_roster = 1;
  assert(!et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer));
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 2);
  forge_roster = 0;
  fail_o2_begin = 1;
  assert(!et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer));
  assert(g3_releases == 1 && o2_releases == 0);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  assert(et_g3r_private_same_owner_holds_error_code_v1() ==
         ET_O2_CODE_OWNER_CONFLICT);
  fail_o2_begin = 2;
  assert(!et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer));
  assert(g3_releases == 2);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  assert(et_g3r_private_same_owner_holds_error_category_v1() ==
         ET_O2_STATUS_INVALID_ARGUMENT);
  fail_o2_begin = 0;
  void *pair = et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer);
  assert(pair && et_g3r_private_same_owner_holds_g3_v1(pair) == &g3_hold);
  assert(et_g3r_private_same_owner_holds_o2_v1(pair) == &o2_hold);
  equal_optimizer = optimizer; /* bit-equal, distinct owner address */
  assert(et_g3r_private_same_owner_holds_check_v1(
      pair, &generator, &model, &equal_optimizer) == 2);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 0);
  assert(et_g3r_private_same_owner_holds_check_v1(
      pair, &generator, &model, &optimizer) == 0);
  fail_o2_check = 1;
  fail_g3_end = 1;
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 7);
  assert(o2_releases == 1 && g3_releases == 3);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  assert(et_g3r_private_same_owner_holds_error_code_v1() ==
         ET_O2_CODE_INVALID_MOMENT);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 2);
  assert(o2_releases == 1 && g3_releases == 3);
  assert(!et_g3r_private_same_owner_holds_o2_v1(pair));
  fail_o2_check = fail_g3_end = 0;
  pair = et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer);
  assert(pair);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 0);
  assert(o2_releases == 2 && g3_releases == 4);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 2);
  puts("same-owner hold bridge fixture PASS");
}
