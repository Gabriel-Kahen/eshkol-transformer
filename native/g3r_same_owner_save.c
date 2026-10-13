#include "g3r_same_owner_save.h"
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
#include "o2_optimizer_internal.h"
#include "../src/eshkol_transformer/m3t_transport.h"
#include "../src/eshkol_transformer/g3t_transport.h"
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(et_g3r_g3t_held_words_v1) ==
                   17u * sizeof(int64_t),
               "same-owner scalar snapshot must remain seventeen words");

enum { ET_G3R_SAVE_MAX_TOMBSTONES = 64 };
static const uint64_t save_parameter_shape[14][2] = {
    {4,4}, {4,4}, {4,4}, {4,4}, {4,8}, {8,4}, {4,0},
    {4,0}, {4,0}, {4,0}, {256,4}, {4,0}, {4,0}, {2,4}};
enum {
  ET_G3R_O2_UNISSUED,
  ET_G3R_O2_ATTEMPTED,
  ET_G3R_O2_PENDING,
  ET_G3R_O2_CLAIMED,
  ET_G3R_O2_CLEANUP_PENDING,
  ET_G3R_O2_ENDED
};
typedef struct et_g3r_same_owner_hold {
  struct et_g3r_same_owner_hold *next;
  void *generator;
  void *m3t_owner;
  void *g3_hold;
  et_o2_g3r_save_hold *o2_hold;
  void *parameters[14];
  const void *identities[14];
  et_f32_parameter_save_witness *witness[14];
  et_g3r_g3t_held_words_v1 captured_words;
  et_g3r_same_owner_save_error_v1 first;
  et_o2_optimizer_state *issued_state;
  unsigned issue_phase, capture_attempted, capture_ready, claimed_once;
  unsigned live;
} et_g3r_same_owner_hold;
static et_g3r_same_owner_hold *holds;
static unsigned tombstones;
static et_g3r_same_owner_save_error_v1 last_error;

static void clear_error(void) { memset(&last_error, 0, sizeof(last_error)); }

static void latch(et_g3r_same_owner_hold *hold,
                  et_g3r_same_owner_save_error_v1 value) {
  if (hold) {
    if (!hold->first.source) hold->first = value;
    last_error = hold->first;
  } else {
    last_error = value;
  }
}

static void coordinator_error(et_g3r_same_owner_hold *hold, int64_t code) {
  latch(hold, (et_g3r_same_owner_save_error_v1){
      ET_G3R_SAVE_SOURCE_COORDINATOR, 0, 1, code});
}

static void m3t_error(void) {
  latch(NULL, (et_g3r_same_owner_save_error_v1){
      ET_G3R_SAVE_SOURCE_M3T, et_m3t_private_last_error_domain_v1(),
      et_m3t_private_last_error_category_v1(),
      et_m3t_private_last_error_code_v1()});
}

static void o2_error(et_g3r_same_owner_hold *hold,
                     const et_o2_error_v1 *error) {
  latch(hold, (et_g3r_same_owner_save_error_v1){
      ET_G3R_SAVE_SOURCE_O2, 0, error->category, error->code});
}

static void g3t_error(et_g3r_same_owner_hold *hold) {
  latch(hold, (et_g3r_same_owner_save_error_v1){
      ET_G3R_SAVE_SOURCE_G3T, et_g3t_private_last_error_domain_v1(),
      et_g3t_private_last_error_category_v1(),
      et_g3t_private_last_error_code_v1()});
}

static void f32_save_error(et_g3r_same_owner_hold *hold,
                           const et_f32_tensor_error *error) {
  latch(hold, (et_g3r_same_owner_save_error_v1){
      ET_G3R_SAVE_SOURCE_F32, 3, error->category, error->code});
}

int64_t et_g3r_private_same_owner_save_last_error_v1(
    et_g3r_same_owner_save_error_v1 *destination) {
  if (!destination) return -1;
  *destination = last_error;
  return 0;
}

static et_g3r_same_owner_hold *admit(const void *candidate) {
  for (et_g3r_same_owner_hold *hold = holds; hold; hold = hold->next)
    if (hold == candidate && hold->live) return hold;
  return NULL;
}

/* Retained shells are searched by address before any caller dereference. */
static et_g3r_same_owner_hold *find_hold(const void *candidate) {
  for (et_g3r_same_owner_hold *hold = holds; hold; hold = hold->next)
    if (hold == candidate) return hold;
  return NULL;
}

int64_t et_g3r_private_same_owner_save_error_word_v1(
    void *candidate, int64_t index) {
  if (index < 0 || index > 3) return INT64_MIN;
  et_g3r_same_owner_hold *hold = candidate ? find_hold(candidate) : NULL;
  const et_g3r_same_owner_save_error_v1 error = !candidate ? last_error
      : hold ? hold->first
             : (et_g3r_same_owner_save_error_v1){
                   ET_G3R_SAVE_SOURCE_COORDINATOR, 0, 1,
                   ET_G3R_SAVE_COORD_STALE};
  const int64_t words[4] = {
      error.source, error.domain, error.category, error.code};
  return words[index];
}

int64_t et_g3r_private_same_owner_save_drain_state_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = candidate ? find_hold(candidate) : NULL;
  if (!hold) return -1;
  if (hold->live) {
    if (!hold->o2_hold || !hold->g3_hold || !hold->generator ||
        !hold->m3t_owner ||
        ((hold->issue_phase == ET_G3R_O2_PENDING ||
          hold->issue_phase == ET_G3R_O2_CLEANUP_PENDING) !=
         (hold->issued_state != NULL))) abort();
    for (size_t i = 0; i < 14u; ++i)
      if (!hold->witness[i] || !hold->parameters[i] ||
          !hold->identities[i]) abort();
    return hold->issue_phase == ET_G3R_O2_CLEANUP_PENDING ? 1 : 0;
  }
  if (hold->issue_phase != ET_G3R_O2_ENDED || hold->issued_state ||
      hold->o2_hold || hold->g3_hold || hold->generator ||
      hold->m3t_owner || hold->capture_ready) abort();
  for (size_t i = 0; i < 14u; ++i)
    if (hold->witness[i] || hold->parameters[i] || hold->identities[i]) abort();
  return hold->claimed_once ? 3 : 2;
}

void *et_g3r_private_same_owner_save_begin_v1(
    void *generator, void *m3t_owner, void *optimizer) {
  clear_error();
  if (!generator || !m3t_owner || !optimizer) {
    coordinator_error(NULL, ET_G3R_SAVE_COORD_ARGUMENT);
    return NULL;
  }
  if (tombstones >= ET_G3R_SAVE_MAX_TOMBSTONES) {
    coordinator_error(NULL, ET_G3R_SAVE_COORD_EXHAUSTED);
    return NULL;
  }
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(
          m3t_owner, parameters, identities) != 0) {
    m3t_error();
    return NULL;
  }
  et_g3r_same_owner_hold *hold = calloc(1, sizeof(*hold));
  if (!hold) {
    coordinator_error(NULL, ET_G3R_SAVE_COORD_ALLOCATION);
    return NULL;
  }
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_begin_v1(
          optimizer, (et_f32_parameter *const *)parameters, identities,
          &hold->o2_hold, &error) != 0) {
    o2_error(NULL, &error);
    free(hold);
    return NULL;
  }
  hold->g3_hold = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!hold->g3_hold) {
    g3t_error(NULL); /* Capture before O2 cleanup can change leaf state. */
    if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0 &&
        hold->o2_hold != NULL)
      abort(); /* Failed to drain an owned hold. */
    free(hold);
    return NULL;
  }
  if (et_g3r_private_g3t_save_check_v1(hold->g3_hold) != 0) {
    g3t_error(NULL);
    (void)et_g3r_private_g3t_save_end_v1(hold->g3_hold);
    if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0 &&
        hold->o2_hold != NULL) abort();
    free(hold);
    return NULL;
  }
  et_f32_tensor_error f32_error;
  size_t acquired = 0u;
  for (; acquired < 14u; ++acquired) {
    const size_t rank = save_parameter_shape[acquired][1] ? 2u : 1u;
    if (et_f32_parameter_save_witness_begin_v1(
            (et_f32_parameter *)parameters[acquired], identities[acquired],
            rank, save_parameter_shape[acquired], &hold->witness[acquired],
            &f32_error) != 0) {
      f32_save_error(NULL, &f32_error); /* Save before other leaf cleanup. */
      while (acquired > 0u)
        if (et_f32_parameter_save_witness_end_v1(
                &hold->witness[--acquired], &f32_error) != 0) abort();
      (void)et_g3r_private_g3t_save_end_v1(hold->g3_hold);
      if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0 &&
          hold->o2_hold != NULL) abort();
      free(hold);
      return NULL;
    }
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
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->first.source) {
    last_error = hold->first;
    return -1;
  }
  clear_error();
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(
          hold->m3t_owner, parameters, identities) != 0) {
    latch(hold, (et_g3r_same_owner_save_error_v1){
        ET_G3R_SAVE_SOURCE_M3T, et_m3t_private_last_error_domain_v1(),
        et_m3t_private_last_error_category_v1(),
        et_m3t_private_last_error_code_v1()});
    return -1;
  }
  if (memcmp(parameters, hold->parameters, sizeof(parameters)) ||
      memcmp(identities, hold->identities, sizeof(identities))) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  et_o2_error_v1 error;
  if (et_g3r_private_g3t_save_check_v1(hold->g3_hold) != 0) {
    g3t_error(hold);
    return -1;
  }
  et_f32_tensor_error f32_error;
  for (size_t i = 0u; i < 14u; ++i)
    if (et_f32_parameter_save_witness_check_v1(
            hold->witness[i], &f32_error) != 0) {
      f32_save_error(hold, &f32_error);
      return -1;
    }
  if (et_g3r_private_o2_save_check_v1(hold->o2_hold, &error) != 0) {
    o2_error(hold, &error);
    return -1;
  }
  return 0;
}

int64_t et_g3r_private_same_owner_save_copy_v1(
    void *candidate, et_g3r_g3t_held_words_v1 *destination,
    int64_t destination_bytes) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (et_g3r_private_same_owner_save_check_v1(hold) != 0)
    return -1;
  if (et_g3r_private_g3t_save_copy_v1(
          hold->g3_hold, destination, destination_bytes) != 0) {
    g3t_error(hold);
    return -1;
  }
  return et_g3r_private_same_owner_save_check_v1(hold);
}

int64_t et_g3r_private_same_owner_save_capture_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->capture_attempted) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  hold->capture_attempted = 1u;
  et_g3r_g3t_held_words_v1 staged = {0};
  if (et_g3r_private_same_owner_save_copy_v1(
          hold, &staged, (int64_t)sizeof(staged)) != 0 ||
      et_g3r_private_same_owner_save_check_v1(hold) != 0)
    return -1;
  hold->captured_words = staged;
  hold->capture_ready = 1u;
  return 0;
}

int64_t et_g3r_private_same_owner_save_snapshot_word_v1(
    void *candidate, int64_t index) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold || !hold->capture_ready || index < 0 || index >= 17)
    return INT64_MIN;
  return hold->captured_words.word[index];
}

void *et_g3r_private_same_owner_save_o2_issue_v1(void *candidate) {
  void *state = NULL;
  if (et_g3r_private_same_owner_save_o2_state_v1(candidate, &state) != 0)
    return NULL;
  return state;
}

int64_t et_g3r_private_same_owner_save_o2_state_v1(
    void *candidate, void **destination) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->issue_phase != ET_G3R_O2_UNISSUED) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  hold->issue_phase = ET_G3R_O2_ATTEMPTED;
  if (!destination || *destination) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ARGUMENT);
    return -1;
  }
  if (et_g3r_private_same_owner_save_check_v1(hold) != 0)
    return -1;
  et_o2_optimizer_state *state = NULL;
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_state_v1(
          hold->o2_hold, &state, &error) != 0) {
    o2_error(hold, &error);
    return -1;
  }
  hold->issued_state = state;
  hold->issue_phase = ET_G3R_O2_PENDING;
  if (et_g3r_private_o2_save_issued_match_v1(
          hold->o2_hold, state, &error) != 0) {
    o2_error(hold, &error);
    (void)et_g3r_private_same_owner_save_o2_cancel_v1(hold);
    return -1;
  }
  if (et_g3r_private_same_owner_save_check_v1(hold) != 0) {
    (void)et_g3r_private_same_owner_save_o2_cancel_v1(hold);
    return -1;
  }
  *destination = state;
  return 0;
}

static int64_t release_issued(et_g3r_same_owner_hold *hold) {
  et_o2_error_v1 error;
  et_o2_optimizer_state *state = hold->issued_state;
  if (!state) return 0;
  if (et_o2_optimizer_state_release_v1(state, &error) == 0) {
    hold->issued_state = NULL;
    hold->issue_phase = ET_G3R_O2_ATTEMPTED;
    return 0;
  }
  o2_error(hold, &error); /* Preserve before lifecycle query changes O2 error. */
  uint32_t lifecycle = 0;
  if (et_o2_optimizer_state_lifecycle_v1(state, &lifecycle, &error) == 0 &&
      lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    hold->issued_state = NULL;
    hold->issue_phase = ET_G3R_O2_ATTEMPTED;
    return -1;
  }
  hold->issue_phase = ET_G3R_O2_CLEANUP_PENDING;
  return -1;
}

int64_t et_g3r_private_same_owner_save_o2_claim_v1(
    void *candidate, void *state) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->issue_phase != ET_G3R_O2_PENDING ||
      !state || state != hold->issued_state) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  et_o2_error_v1 error;
  if (et_g3r_private_same_owner_save_check_v1(hold) != 0) return -1;
  if (et_g3r_private_o2_save_issued_match_v1(
          hold->o2_hold, (et_o2_optimizer_state *)state, &error) != 0) {
    o2_error(hold, &error);
    return -1;
  }
  hold->issued_state = NULL;
  hold->issue_phase = ET_G3R_O2_CLAIMED;
  hold->claimed_once = 1u;
  return 0;
}

int64_t et_g3r_private_same_owner_save_o2_cancel_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->issue_phase == ET_G3R_O2_CLAIMED) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  if (hold->issue_phase == ET_G3R_O2_CLEANUP_PENDING) {
    if (hold->first.source) last_error = hold->first;
    return -1; /* Only the checked retry may revisit uncertain cleanup. */
  }
  if (hold->issue_phase == ET_G3R_O2_UNISSUED)
    hold->issue_phase = ET_G3R_O2_ATTEMPTED;
  if (hold->issue_phase == ET_G3R_O2_PENDING)
    (void)release_issued(hold);
  return hold->first.source ? -1 : 0;
}

int64_t et_g3r_private_same_owner_save_cleanup_retry_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->issue_phase != ET_G3R_O2_CLEANUP_PENDING ||
      !hold->issued_state) {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  et_o2_error_v1 error;
  uint32_t lifecycle = 0;
  if (et_o2_optimizer_state_lifecycle_v1(
          hold->issued_state, &lifecycle, &error) != 0) {
    o2_error(hold, &error);
    return -1;
  }
  if (lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    hold->issued_state = NULL;
    hold->issue_phase = ET_G3R_O2_ATTEMPTED;
  } else if (lifecycle == ET_O2_OPTIMIZER_STATE_LIVE) {
    (void)release_issued(hold);
  } else {
    coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1;
  }
  if (hold->issued_state) return -1;
  return et_g3r_private_same_owner_save_end_v1(hold);
}

int64_t et_g3r_private_same_owner_save_end_v1(void *candidate) {
  et_g3r_same_owner_hold *hold = admit(candidate);
  if (!hold) {
    clear_error();
    coordinator_error(NULL, ET_G3R_SAVE_COORD_STALE);
    return -1;
  }
  if (hold->issue_phase == ET_G3R_O2_PENDING)
    (void)release_issued(hold);
  if (hold->issued_state) {
    if (hold->issue_phase != ET_G3R_O2_CLEANUP_PENDING)
      coordinator_error(hold, ET_G3R_SAVE_COORD_ISSUE);
    return -1; /* Keep both leases and the native state rooted for retry. */
  }
  (void)et_g3r_private_same_owner_save_check_v1(hold);
  et_f32_tensor_error f32_error;
  for (size_t i = 14u; i > 0u; --i)
    if (et_f32_parameter_save_witness_end_v1(
            &hold->witness[i - 1u], &f32_error) != 0) abort();
  if (et_g3r_private_g3t_save_end_v1(hold->g3_hold) != 0 &&
      !hold->first.source)
    g3t_error(hold);
  et_o2_error_v1 error;
  if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0 &&
      !hold->first.source)
    o2_error(hold, &error);
  if (hold->o2_hold != NULL) abort(); /* Must not lose the live O2 lease. */
  hold->g3_hold = NULL;
  memset(&hold->captured_words, 0, sizeof(hold->captured_words));
  hold->capture_ready = 0u;
  hold->generator = NULL;
  hold->m3t_owner = NULL;
  memset(hold->parameters, 0, sizeof(hold->parameters));
  memset(hold->identities, 0, sizeof(hold->identities));
  hold->live = 0u; /* Retain inert shell, no pointer reuse into authority. */
  hold->issue_phase = ET_G3R_O2_ENDED;
  if (hold->first.source) last_error = hold->first;
  return hold->first.source ? -1 : 0;
}
#endif
