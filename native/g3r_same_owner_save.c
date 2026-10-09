#include "g3r_same_owner_save.h"
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
#include "o2_optimizer_internal.h"
#include "../src/eshkol_transformer/m3t_transport.h"
#include "../src/eshkol_transformer/g3t_transport.h"
#include <stdlib.h>
#include <string.h>

enum { ET_G3R_SAVE_MAX_TOMBSTONES = 64 };
typedef struct et_g3r_same_owner_hold {
  struct et_g3r_same_owner_hold *next;
  void *generator;
  void *m3t_owner;
  void *g3_hold;
  et_o2_g3r_save_hold *o2_hold;
  void *parameters[14];
  const void *identities[14];
  unsigned live;
} et_g3r_same_owner_hold;
static et_g3r_same_owner_hold *holds;
static unsigned tombstones;

static et_g3r_same_owner_hold *admit(const void *candidate) {
  for (et_g3r_same_owner_hold *hold = holds; hold; hold = hold->next)
    if (hold == candidate && hold->live) return hold;
  return NULL;
}

void *et_g3r_private_same_owner_save_begin_v1(
    void *generator, void *m3t_owner, void *optimizer) {
  if (!generator || !m3t_owner || !optimizer ||
      tombstones >= ET_G3R_SAVE_MAX_TOMBSTONES)
    return NULL;
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(
          m3t_owner, parameters, identities) != 0)
    return NULL;
  et_g3r_same_owner_hold *hold = calloc(1, sizeof(*hold));
  if (!hold) return NULL;
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_begin_v1(
          optimizer, (et_f32_parameter *const *)parameters, identities,
          &hold->o2_hold, &error) != 0) {
    free(hold);
    return NULL;
  }
  hold->g3_hold = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!hold->g3_hold) {
    if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0)
      abort(); /* No source mutation occurred under this new O2 hold. */
    free(hold);
    return NULL;
  }
  hold->generator = generator;
  hold->m3t_owner = m3t_owner;
  memcpy(hold->parameters, parameters, sizeof(parameters));
  memcpy(hold->identities, identities, sizeof(identities));
  hold->live = 1u;
  hold->next = holds;
  holds = hold;
  ++tombstones;
  return hold;
}

int64_t et_g3r_private_same_owner_save_check_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) return -1;
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(
          hold->m3t_owner, parameters, identities) != 0)
    return -1;
  if (memcmp(parameters, hold->parameters, sizeof(parameters)) ||
      memcmp(identities, hold->identities, sizeof(identities)))
    return -1;
  et_o2_error_v1 error;
  if (et_g3r_private_g3t_save_check_v1(hold->g3_hold) != 0 ||
      et_g3r_private_o2_save_check_v1(hold->o2_hold, &error) != 0)
    return -1;
  return 0;
}

int64_t et_g3r_private_same_owner_save_copy_v1(
    void *candidate, et_g3r_g3t_idle_snapshot_words_v1 *destination,
    int64_t destination_bytes) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold || et_g3r_private_same_owner_save_check_v1(hold) != 0)
    return -1;
  if (et_g3r_private_g3t_save_copy_v1(
          hold->g3_hold, destination, destination_bytes) != 0)
    return -1;
  return et_g3r_private_same_owner_save_check_v1(hold);
}

int64_t et_g3r_private_same_owner_save_o2_state_v1(
    void *candidate, void **destination) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold || !destination || *destination ||
      et_g3r_private_same_owner_save_check_v1(hold) != 0)
    return -1;
  et_o2_optimizer_state *state = NULL;
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_state_v1(
          hold->o2_hold, &state, &error) != 0)
    return -1;
  if (et_g3r_private_same_owner_save_check_v1(hold) != 0) {
    if (et_o2_optimizer_state_release_v1(state, &error) != 0)
      abort(); /* Detached new state has no external borrower. */
    return -1;
  }
  *destination = state;
  return 0;
}

int64_t et_g3r_private_same_owner_save_end_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) return -1;
  int64_t first = et_g3r_private_same_owner_save_check_v1(hold);
  if (et_g3r_private_g3t_save_end_v1(hold->g3_hold) != 0 && !first)
    first = -1;
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0 && !first)
    first = -1;
  hold->g3_hold = NULL;
  hold->generator = NULL;
  hold->m3t_owner = NULL;
  memset(hold->parameters, 0, sizeof(hold->parameters));
  memset(hold->identities, 0, sizeof(hold->identities));
  hold->live = 0u; /* Retain inert shell, no pointer reuse into authority. */
  return first;
}
#endif
