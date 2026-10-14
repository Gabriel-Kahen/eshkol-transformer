/* Causal provider-contract fixture; the compiled joint aggregate remains a
 * separate gate. Mock functions enforce the reviewed M3/G3/O2 C signatures. */
#include "../../native/g3r_same_owner_hold_bridge.c"
#include "o2_optimizer_internal.h"
#include "eshkol_transformer/g3t_transport.h"
#include "eshkol_transformer/m3t_transport.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int model, generator, optimizer, equal_optimizer, g3_hold, o2_hold;
static int parameters[14], identities[14], g3_releases, o2_releases;
static int fail_o2_begin, fail_o2_check, fail_g3_end, fail_g3_copy;
static int forge_roster, g3_copies;
static int64_t g3_category, g3_code, m3_category, m3_code;
static int64_t next_g3_ticket = 1, next_o2_ticket = 1;
static int64_t live_g3_ticket, live_o2_ticket;

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
int64_t et_g3r_private_g3t_save_copy_v1(
    void *hold, et_g3r_g3t_held_words_v1 *destination,
    int64_t destination_bytes) {
  assert(hold == &g3_hold && destination_bytes == 136);
  if (fail_g3_copy) { g3_category = 2; g3_code = 9; return 2; }
  ++g3_copies;
  for (int i = 0; i < 17; ++i) destination->word[i] = 100 + i;
  return 0;
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
int64_t et_g3r_private_g3t_save_begin_v2(
    void *owner, void *const roster[14], const void *const ids[14]) {
  if (owner != &generator || live_g3_ticket) {
    g3_category = 2; g3_code = 2; return 0;
  }
  for (int i = 0; i < 14; ++i)
    if (roster[i] != &parameters[i] || ids[i] != &identities[i]) {
      g3_category = 2; g3_code = 3; return 0;
    }
  live_g3_ticket = next_g3_ticket++;
  return live_g3_ticket;
}
int64_t et_g3r_private_g3t_save_check_v2(int64_t ticket) {
  if (ticket == live_g3_ticket && ticket > 0) return 0;
  g3_category = 1; g3_code = 1; return 1;
}
int64_t et_g3r_private_g3t_save_copy_v2(
    int64_t ticket, et_g3r_g3t_held_words_v1 *destination,
    int64_t destination_bytes) {
  if (et_g3r_private_g3t_save_check_v2(ticket)) return g3_category;
  assert(destination_bytes == 136);
  if (fail_g3_copy) { g3_category = 2; g3_code = 9; return 2; }
  ++g3_copies;
  for (int i = 0; i < 17; ++i) destination->word[i] = 100 + i;
  return 0;
}
int64_t et_g3r_private_g3t_save_end_v2(int64_t ticket) {
  if (et_g3r_private_g3t_save_check_v2(ticket)) return g3_category;
  live_g3_ticket = 0; ++g3_releases;
  if (fail_g3_end) { g3_category = 2; g3_code = 9; return 2; }
  return 0;
}
int32_t et_g3r_private_o2_save_begin_v2(
    et_o2_optimizer *owner, et_f32_parameter *const roster[14],
    const void *const ids[14], int64_t *out, et_o2_error_v1 *error) {
  assert(*out == 0);
  if ((void *)owner != &optimizer || fail_o2_begin || live_o2_ticket) {
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
  *out = live_o2_ticket = next_o2_ticket++;
  return 0;
}
int32_t et_g3r_private_o2_save_check_v2(
    int64_t ticket, et_o2_error_v1 *error) {
  if (ticket != live_o2_ticket || ticket <= 0) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_INVALID_HANDLE;
    return ET_O2_STATUS_INVALID_STATE;
  }
  if (fail_o2_check) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_INVALID_MOMENT;
    return ET_O2_STATUS_INVALID_STATE;
  }
  return 0;
}
int32_t et_g3r_private_o2_save_owner_match_v2(
    int64_t ticket, const et_o2_optimizer *owner, et_o2_error_v1 *error) {
  int32_t status = et_g3r_private_o2_save_check_v2(ticket, error);
  if (status) return status;
  if ((const void *)owner != &optimizer) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_OWNER_CONFLICT;
    return ET_O2_STATUS_INVALID_STATE;
  }
  return 0;
}
int32_t et_g3r_private_o2_save_end_v2(
    int64_t *slot, et_o2_error_v1 *error) {
  if (*slot != live_o2_ticket || *slot <= 0) {
    error->category = ET_O2_STATUS_INVALID_STATE;
    error->code = ET_O2_CODE_INVALID_HANDLE;
    return ET_O2_STATUS_INVALID_STATE;
  }
  int32_t status = et_g3r_private_o2_save_check_v2(*slot, error);
  live_o2_ticket = 0; *slot = 0; ++o2_releases;
  return status;
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
  int64_t pair = et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer);
  assert(pair > 0 && et_g3r_private_same_owner_holds_g3_v1(pair) == &g3_hold);
  assert(et_g3r_private_same_owner_holds_o2_v1(pair) == &o2_hold);
  equal_optimizer = optimizer; /* bit-equal, distinct owner address */
  assert(et_g3r_private_same_owner_holds_check_v1(
      pair, &generator, &model, &equal_optimizer) == 2);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 0);
  assert(et_g3r_private_same_owner_holds_check_v1(
      pair, &generator, &model, &optimizer) == 0);
  struct { int64_t length; int64_t words[17]; } image;
  memset(&image, 0x5a, sizeof(image));
  image.length = 136;
  const int64_t before = image.words[0];
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 135) == 1);
  assert(image.words[0] == before && g3_copies == 0);
  image.length = 135;
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 136) == 1);
  assert(image.words[0] == before && g3_copies == 0);
  image.length = 136;
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, (unsigned char *)&image + 1, 136) == 1);
  assert(image.words[0] == before && g3_copies == 0);
  fail_o2_check = 1;
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 136) == 7);
  assert(image.words[0] == before && g3_copies == 0);
  fail_o2_check = 0;
  fail_g3_copy = 1;
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 136) == 2);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 2);
  assert(image.words[0] == before && g3_copies == 0);
  fail_g3_copy = 0;
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 136) == 0);
  assert(g3_copies == 1 && image.words[0] == 100 && image.words[16] == 116);
  fail_o2_check = 1;
  fail_g3_end = 1;
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 7);
  assert(o2_releases == 1 && g3_releases == 3);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  assert(et_g3r_private_same_owner_holds_error_code_v1() ==
         ET_O2_CODE_INVALID_MOMENT);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 2);
  assert(o2_releases == 1 && g3_releases == 3);
  assert(et_g3r_private_same_owner_holds_copy_g3_v1(
      pair, &image, 136) == 2);
  assert(g3_copies == 1);
  assert(!et_g3r_private_same_owner_holds_o2_v1(pair));
  fail_o2_check = fail_g3_end = 0;
  pair = et_g3r_private_same_owner_holds_begin_v1(
      &generator, &model, &optimizer);
  assert(pair > 0);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 0);
  assert(o2_releases == 2 && g3_releases == 4);
  assert(et_g3r_private_same_owner_holds_end_v1(pair) == 2);
  const int64_t stale = pair;
  /* More than the old lifetime cap, while allocator addresses may recycle. */
  for (int i = 0; i < 128; ++i) {
    pair = et_g3r_private_same_owner_holds_begin_v1(
        &generator, &model, &optimizer);
    assert(pair > stale);
    assert(et_g3r_private_same_owner_holds_check_v1(
        stale, &generator, &model, &optimizer) == 2);
    assert(et_g3r_private_same_owner_holds_end_v1(pair) == 0);
    assert(et_g3r_private_same_owner_holds_end_v1(pair) == 2);
  }
  assert(o2_releases == 130 && g3_releases == 132);
  assert(et_g3r_private_same_owner_holds_check_v2(
      0, &generator, &model, &optimizer) == 2);
  assert(et_g3r_private_same_owner_holds_g3_v2(INT64_MAX) == 0);
  assert(et_g3r_private_same_owner_holds_o2_v2(INT64_MAX) == 0);
  forge_roster = 1;
  assert(!et_g3r_private_same_owner_holds_begin_v2(
      &generator, &model, &optimizer));
  assert(!live_g3_ticket && !live_o2_ticket);
  forge_roster = 0;
  fail_o2_begin = 1;
  const int g3_before_failed_begin = g3_releases;
  assert(!et_g3r_private_same_owner_holds_begin_v2(
      &generator, &model, &optimizer));
  assert(g3_releases == g3_before_failed_begin + 1);
  assert(!live_g3_ticket && !live_o2_ticket);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  fail_o2_begin = 0;
  pair = et_g3r_private_same_owner_holds_begin_v2(
      &generator, &model, &optimizer);
  assert(pair > stale && live_g3_ticket && live_o2_ticket);
  assert(et_g3r_private_same_owner_holds_g3_v2(pair) == live_g3_ticket);
  assert(et_g3r_private_same_owner_holds_o2_v2(pair) == live_o2_ticket);
  assert(et_g3r_private_same_owner_holds_check_v1(
      pair, &generator, &model, &optimizer) == 2);
  assert(!et_g3r_private_same_owner_holds_begin_v2(
      &generator, &model, &optimizer));
  assert(et_g3r_private_same_owner_holds_check_v2(
      pair, &generator, &model, &equal_optimizer) == 2);
  assert(et_g3r_private_same_owner_holds_check_v2(
      pair, &generator, &model, &optimizer) == 0);
  memset(&image, 0x5a, sizeof(image)); image.length = 136;
  const int64_t v2_before = image.words[0];
  assert(et_g3r_private_same_owner_holds_copy_g3_v2(
      pair, &image, 135) == 1);
  assert(image.words[0] == v2_before);
  fail_g3_copy = 1;
  assert(et_g3r_private_same_owner_holds_copy_g3_v2(
      pair, &image, 136) == 2);
  assert(image.words[0] == v2_before);
  fail_g3_copy = 0;
  assert(et_g3r_private_same_owner_holds_copy_g3_v2(
      pair, &image, 136) == 0);
  assert(image.words[0] == 100 && image.words[16] == 116);
  fail_o2_check = 1; fail_g3_end = 1;
  const int g3_before_failed_end = g3_releases;
  const int o2_before_failed_end = o2_releases;
  assert(et_g3r_private_same_owner_holds_end_v2(pair) == 7);
  assert(g3_releases == g3_before_failed_end + 1 &&
         o2_releases == o2_before_failed_end + 1);
  assert(et_g3r_private_same_owner_holds_error_source_v1() == 3);
  assert(et_g3r_private_same_owner_holds_error_code_v1() ==
         ET_O2_CODE_INVALID_MOMENT);
  assert(et_g3r_private_same_owner_holds_end_v2(pair) == 2);
  assert(!live_g3_ticket && !live_o2_ticket);
  fail_o2_check = fail_g3_end = 0;
  const int64_t old_g3 = next_g3_ticket - 1;
  const int64_t old_o2 = next_o2_ticket - 1;
  const int64_t old_pair = pair;
  for (int i = 0; i < 128; ++i) {
    pair = et_g3r_private_same_owner_holds_begin_v2(
        &generator, &model, &optimizer);
    assert(pair > old_pair);
    assert(et_g3r_private_same_owner_holds_check_v2(
        old_pair, &generator, &model, &optimizer) == 2);
    assert(et_g3r_private_g3t_save_check_v2(old_g3) != 0);
    et_o2_error_v1 old_error = {0};
    assert(et_g3r_private_o2_save_check_v2(old_o2, &old_error) ==
           ET_O2_STATUS_INVALID_STATE);
    assert(et_g3r_private_same_owner_holds_end_v2(pair) == 0);
    assert(et_g3r_private_same_owner_holds_end_v2(pair) == 2);
  }
  assert(!live_g3_ticket && !live_o2_ticket);
  next_g3_ticket = 136;
  pair = et_g3r_private_same_owner_holds_begin_v2(
      &generator, &model, &optimizer);
  assert(pair > old_pair && pairs && pairs->g3_ticket == 136);
  const int copies_before_overlap = g3_copies;
  assert(et_g3r_private_same_owner_holds_copy_g3_v2(
      pair, &pairs->g3_ticket, 136) == 1);
  assert(g3_copies == copies_before_overlap && pairs->g3_ticket == 136);
  assert(et_g3r_private_same_owner_holds_end_v2(pair) == 0);
  puts("same-owner hold bridge fixture PASS");
}
