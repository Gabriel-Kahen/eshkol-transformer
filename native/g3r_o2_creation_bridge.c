#include "g3r_o2_creation_bridge.h"
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
#include "o2_optimizer_internal.h"
#include "../src/eshkol_transformer/m3t_transport.h"
#include <stdlib.h>
#include <string.h>

enum { ET_G3R_CREATION_O2_MAX_TOMBSTONES = 64 };
typedef struct creation_hold {
  struct creation_hold *next;
  void *owner;
  et_o2_optimizer *optimizer;
  et_o2_g3r_save_hold *o2;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_g3r_same_owner_save_error_v1 first;
  unsigned live;
} creation_hold;
static creation_hold *holds;
static unsigned tombstones;
static et_g3r_same_owner_save_error_v1 latest;
#ifdef ET_G3R_O2_CREATION_TESTING
static unsigned fail_shell_once;
void et_g3r_o2_creation_test_fail_shell_once_v1(void) { fail_shell_once = 1; }
#endif

static et_g3r_same_owner_save_error_v1 stale(void) {
  return (et_g3r_same_owner_save_error_v1){4, 0, 1, 2};
}
static void coord(creation_hold *hold, int64_t code) {
  et_g3r_same_owner_save_error_v1 value = {4, 0, 1, code};
  if (hold) {
    if (!hold->first.source) hold->first = value;
    latest = hold->first;
  } else latest = value;
}
static void m3_error(creation_hold *hold) {
  et_g3r_same_owner_save_error_v1 value = {
      1, et_m3t_private_last_error_domain_v1(),
      et_m3t_private_last_error_category_v1(),
      et_m3t_private_last_error_code_v1()};
  if (hold) {
    if (!hold->first.source) hold->first = value;
    latest = hold->first;
  } else latest = value;
}
static void o2_error(creation_hold *hold, const et_o2_error_v1 *error) {
  et_g3r_same_owner_save_error_v1 value = {2, 0, error->category,
                                             error->code};
  if (hold) {
    if (!hold->first.source) hold->first = value;
    latest = hold->first;
  } else latest = value;
}
/* Address-first admission: no caller pointer is dereferenced by this walk. */
static creation_hold *find(const void *candidate) {
  for (creation_hold *hold = holds; hold; hold = hold->next)
    if (hold == candidate) return hold;
  return NULL;
}
#ifdef ET_G3R_O2_CREATION_TESTING
int64_t et_g3r_o2_creation_test_tombstone_cleared_v1(void *candidate) {
  creation_hold *hold = find(candidate);
  if (!hold || hold->live || hold->owner || hold->optimizer || hold->o2)
    return 0;
  for (size_t i = 0; i < 14; ++i)
    if (hold->parameters[i] || hold->identities[i]) return 0;
  return 1;
}
#endif
static int roster(void *owner, et_f32_parameter *typed[14],
                  const void *identities[14]) {
  void *raw[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(owner, raw, identities) != 0)
    return -1;
  for (size_t i = 0; i < 14; ++i)
    typed[i] = (et_f32_parameter *)raw[i];
  return 0;
}

void *et_g3r_private_o2_creation_begin_v1(void *owner, void *optimizer) {
  memset(&latest, 0, sizeof(latest));
  if (!owner || !optimizer) { coord(NULL, 1); return NULL; }
  if (tombstones >= ET_G3R_CREATION_O2_MAX_TOMBSTONES) {
    coord(NULL, 3); return NULL;
  }
  et_f32_parameter *typed[14] = {0};
  const void *identities[14] = {0};
  if (roster(owner, typed, identities) != 0) {
    m3_error(NULL); return NULL;
  }
#ifdef ET_G3R_O2_CREATION_TESTING
  if (fail_shell_once) { fail_shell_once = 0; coord(NULL, 4); return NULL; }
#endif
  creation_hold *hold = calloc(1, sizeof(*hold));
  if (!hold) { coord(NULL, 4); return NULL; }
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_begin_v1(
          (et_o2_optimizer *)optimizer, typed, identities,
          &hold->o2, &error) != 0) {
    o2_error(NULL, &error);
    if (hold->o2 != NULL) abort();
    free(hold);
    return NULL;
  }
  hold->owner = owner;
  hold->optimizer = (et_o2_optimizer *)optimizer;
  memcpy(hold->parameters, typed, sizeof(typed));
  memcpy(hold->identities, identities, sizeof(identities));
  hold->live = 1;
  hold->next = holds;
  holds = hold;
  ++tombstones;
  return hold;
}

int64_t et_g3r_private_o2_creation_check_v1(void *candidate) {
  creation_hold *hold = find(candidate);
  if (!hold || !hold->live) { latest = stale(); return -1; }
  if (hold->first.source) { latest = hold->first; return -1; }
  memset(&latest, 0, sizeof(latest));
  et_f32_parameter *typed[14] = {0};
  const void *identities[14] = {0};
  if (roster(hold->owner, typed, identities) != 0) {
    m3_error(hold); return -1;
  }
  if (memcmp(typed, hold->parameters, sizeof(typed)) != 0 ||
      memcmp(identities, hold->identities, sizeof(identities)) != 0) {
    coord(hold, 2); return -1;
  }
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_check_v1(hold->o2, &error) != 0) {
    o2_error(hold, &error); return -1;
  }
  return 0;
}

int64_t et_g3r_private_o2_creation_end_v1(void *candidate) {
  creation_hold *hold = find(candidate);
  if (!hold || !hold->live) { latest = stale(); return -1; }
  if (!hold->first.source) (void)et_g3r_private_o2_creation_check_v1(hold);
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_end_v1(&hold->o2, &error) != 0)
    o2_error(hold, &error);
  if (hold->o2 != NULL) abort(); /* Cannot claim a drained live lease. */
  hold->live = 0;
  hold->owner = NULL;
  hold->optimizer = NULL;
  memset(hold->parameters, 0, sizeof(hold->parameters));
  memset(hold->identities, 0, sizeof(hold->identities));
  latest = hold->first;
  return hold->first.source ? -1 : 0;
}

int64_t et_g3r_private_o2_creation_error_word_v1(
    void *candidate, int64_t index) {
  if (index < 0 || index > 3) return INT64_MIN;
  creation_hold *hold = candidate ? find(candidate) : NULL;
  et_g3r_same_owner_save_error_v1 value = candidate
      ? (hold ? hold->first : stale()) : latest;
  const int64_t words[4] = {
      value.source, value.domain, value.category, value.code};
  return words[index];
}
#endif
