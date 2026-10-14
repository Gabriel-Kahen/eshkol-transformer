/* Source-private same-owner G3/M3/A2 and O2 hold prerequisite. This file is
 * deliberately absent from public roots until a reviewed aggregate links it. */
#include "g3r_joint_hold_internal.h"
#include "eshkol_transformer/m3t_transport.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { G3R_JOINT_NONE = 0, G3R_JOINT_M3 = 1, G3R_JOINT_G3 = 2,
       G3R_JOINT_O2 = 3, G3R_JOINT_INTERNAL = 4 };
enum { G3R_JOINT_MAX_TOMBSTONES = 64 };

struct et_g3r_joint_hold {
  struct et_g3r_joint_hold *next;
  void *generator;
  void *model;
  et_o2_optimizer *optimizer;
  void *g3_hold;
  et_o2_g3r_save_hold *o2_hold;
  et_o2_optimizer_state *issued;
  uint32_t issued_once;
  uint32_t live;
};

static et_g3r_joint_hold *g3r_joint_registry;
static unsigned g3r_joint_tombstones;
static _Thread_local int64_t g3r_joint_domain;
static _Thread_local int64_t g3r_joint_category;
static _Thread_local int64_t g3r_joint_code;

static void joint_clear(void) {
  g3r_joint_domain = g3r_joint_category = g3r_joint_code = 0;
}
static void joint_m3_error(void) {
  g3r_joint_domain = G3R_JOINT_M3;
  g3r_joint_category = et_m3t_private_last_error_category_v1();
  g3r_joint_code = et_m3t_private_last_error_code_v1();
}
static void joint_g3_error(void) {
  g3r_joint_domain = G3R_JOINT_G3;
  g3r_joint_category = et_g3t_private_last_error_category_v1();
  g3r_joint_code = et_g3t_private_last_error_code_v1();
}
static void joint_o2_error(const et_o2_error_v1 *error) {
  g3r_joint_domain = G3R_JOINT_O2;
  g3r_joint_category = error->category;
  g3r_joint_code = error->code;
}
static int64_t joint_invalid(void) {
  g3r_joint_domain = G3R_JOINT_INTERNAL;
  g3r_joint_category = ET_O2_STATUS_INVALID_STATE;
  g3r_joint_code = ET_O2_CODE_INVALID_HANDLE;
  return -1;
}
static et_g3r_joint_hold *joint_find(const et_g3r_joint_hold *candidate) {
  for (et_g3r_joint_hold *hold = g3r_joint_registry; hold; hold = hold->next)
    if (hold == candidate && hold->live == 1u)
      return hold;
  return NULL;
}

et_g3r_joint_hold *et_g3r_joint_hold_begin_v1(
    void *generator, void *sealed_model, et_o2_optimizer *optimizer) {
  void *parameters[14] = {0};
  et_f32_parameter *typed_parameters[14] = {0};
  const void *identities[14] = {0};
  et_o2_error_v1 o2_error = {0};
  et_g3r_joint_hold *hold;
  joint_clear();
  if (!generator || !sealed_model || !optimizer ||
      g3r_joint_tombstones >= G3R_JOINT_MAX_TOMBSTONES) {
    joint_invalid();
    return NULL;
  }
  /* Allocate the permanent registry shell before acquiring either hold. */
  hold = calloc(1, sizeof(*hold));
  if (!hold) {
    joint_invalid();
    return NULL;
  }
  if (et_m3t_private_owner_save_roster_v1(
          sealed_model, parameters, identities) != 0) {
    joint_m3_error();
    free(hold);
    return NULL;
  }
  for (size_t i = 0; i < 14; ++i)
    typed_parameters[i] = parameters[i];
  hold->g3_hold = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!hold->g3_hold) {
    joint_g3_error();
    free(hold);
    return NULL;
  }
  if (et_g3r_private_o2_save_begin_v1(
          optimizer, typed_parameters, identities,
          &hold->o2_hold, &o2_error) != 0) {
    joint_o2_error(&o2_error);
    /* G3 end drains its borrow/pins even when its final check reports a
     * defect. The O2 error remains the first typed failure. */
    (void)et_g3r_private_g3t_save_end_v1(hold->g3_hold);
    free(hold);
    return NULL;
  }
  hold->generator = generator;
  hold->model = sealed_model;
  hold->optimizer = optimizer;
  hold->live = 1u;
  hold->next = g3r_joint_registry;
  g3r_joint_registry = hold;
  ++g3r_joint_tombstones;
  return hold;
}

int64_t et_g3r_joint_hold_check_v1(et_g3r_joint_hold *candidate) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  et_o2_error_v1 error = {0};
  joint_clear();
  if (!hold) return joint_invalid();
  if (et_g3r_private_g3t_save_check_v1(hold->g3_hold) != 0) {
    joint_g3_error();
    return -g3r_joint_category;
  }
  if (et_g3r_private_o2_save_owner_match_v1(
          hold->o2_hold, hold->optimizer, &error) != 0) {
    joint_o2_error(&error);
    return g3r_joint_category;
  }
  return 0;
}

int64_t et_g3r_joint_hold_optimizer_match_v1(
    et_g3r_joint_hold *candidate, const et_o2_optimizer *optimizer) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  joint_clear();
  if (!hold || !optimizer || optimizer != hold->optimizer)
    return joint_invalid();
  return et_g3r_joint_hold_check_v1(hold);
}

int64_t et_g3r_joint_hold_owner_match_v1(
    et_g3r_joint_hold *candidate, const void *generator,
    const void *sealed_model, const et_o2_optimizer *optimizer) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  joint_clear();
  if (!hold || !generator || !sealed_model || !optimizer ||
      generator != hold->generator || sealed_model != hold->model ||
      optimizer != hold->optimizer)
    return joint_invalid();
  return et_g3r_joint_hold_check_v1(hold);
}

et_o2_optimizer_state *et_g3r_joint_hold_issue_v1(
    et_g3r_joint_hold *candidate) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  et_o2_error_v1 error = {0};
  joint_clear();
  if (!hold || hold->issued || hold->issued_once) {
    joint_invalid();
    return NULL;
  }
  if (et_g3r_joint_hold_check_v1(hold) != 0) return NULL;
  et_o2_optimizer_state *issued = NULL;
  if (et_g3r_private_o2_save_state_v1(
          hold->o2_hold, &issued, &error) != 0) {
    if (issued != NULL) abort(); /* Failed issue returned live authority. */
    joint_o2_error(&error);
    return NULL;
  }
  if (issued == NULL) abort(); /* Successful issue must name exact authority. */
  /* The registered coordinator roots the exact return before any recheck. */
  hold->issued = issued;
  hold->issued_once = 1u;
  if (et_g3r_joint_hold_issued_match_v1(hold, issued) != 0) {
    const int64_t first_domain = g3r_joint_domain;
    const int64_t first_category = g3r_joint_category;
    const int64_t first_code = g3r_joint_code;
    const int64_t release = et_g3r_joint_hold_release_issued_v1(hold);
    if (release == -1) abort(); /* Still LIVE: no safe unnamed caller. */
    g3r_joint_domain = first_domain;
    g3r_joint_category = first_category;
    g3r_joint_code = first_code;
    return NULL;
  }
  return issued;
}

int64_t et_g3r_joint_hold_issued_match_v1(
    et_g3r_joint_hold *candidate, const et_o2_optimizer_state *state) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  et_o2_error_v1 error = {0};
  joint_clear();
  if (!hold || !state || state != hold->issued)
    return joint_invalid(); /* Bit-equal foreign states are insufficient. */
  const int64_t check = et_g3r_joint_hold_check_v1(hold);
  if (check != 0) return check;
  if (et_g3r_private_o2_save_issued_match_v1(
          hold->o2_hold, state, &error) != 0) {
    joint_o2_error(&error);
    return g3r_joint_category;
  }
  return 0;
}

int64_t et_g3r_joint_hold_copy_words_v1(
    et_g3r_joint_hold *candidate, et_g3r_g3t_held_words_v1 *destination) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  joint_clear();
  if (!hold || !destination) return joint_invalid();
  const int64_t check = et_g3r_joint_hold_check_v1(hold);
  if (check != 0) return check;
  if (et_g3r_private_g3t_save_copy_v1(
          hold->g3_hold, destination, sizeof(*destination)) != 0) {
    joint_g3_error();
    return -g3r_joint_category;
  }
  return 0;
}

int64_t et_g3r_joint_hold_release_issued_v1(et_g3r_joint_hold *candidate) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  et_o2_error_v1 error = {0};
  uint32_t lifecycle = 0;
  joint_clear();
  if (!hold || !hold->issued) return joint_invalid();
  const int32_t status = et_o2_optimizer_state_release_v1(hold->issued, &error);
  if (status == 0) {
    hold->issued = NULL;
    return 0;
  }
  joint_o2_error(&error); /* Preserve first typed release defect. */
  et_o2_error_v1 lifecycle_error = {0};
  if (et_o2_optimizer_state_lifecycle_v1(
          hold->issued, &lifecycle, &lifecycle_error) != 0)
    abort();
  if (lifecycle == ET_O2_OPTIMIZER_STATE_LIVE) return -1;
  if (lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    hold->issued = NULL;
    return -2;
  }
  abort(); /* RELEASING or unknown authority cannot escape to Scheme. */
}

int64_t et_g3r_joint_hold_transfer_issued_v1(
    et_g3r_joint_hold *candidate, const et_o2_optimizer_state *state) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  joint_clear();
  if (!hold || !state || state != hold->issued) return joint_invalid();
  /* Require the exact original pair and still-LIVE issued bits immediately
   * before dropping native custody. No native release occurs in this step. */
  if (et_g3r_joint_hold_issued_match_v1(hold, state) != 0)
    return g3r_joint_category ? g3r_joint_category : -1;
  hold->issued = NULL;
  return 0;
}

int64_t et_g3r_joint_hold_end_v1(et_g3r_joint_hold *candidate) {
  et_g3r_joint_hold *hold = joint_find(candidate);
  et_o2_error_v1 error = {0};
  int64_t first_domain = 0, first_category = 0, first_code = 0;
  joint_clear();
  if (!hold || hold->issued) return joint_invalid();
  /* Both native ends drain even when their final recheck reports a defect. */
  if (et_g3r_private_o2_save_end_v1(&hold->o2_hold, &error) != 0) {
    joint_o2_error(&error);
    first_domain = g3r_joint_domain;
    first_category = g3r_joint_category;
    first_code = g3r_joint_code;
  }
  if (hold->o2_hold != NULL) abort(); /* End did not consume the exact hold. */
  const int64_t g3_end = et_g3r_private_g3t_save_end_v1(hold->g3_hold);
  if (g3_end != 0 && !first_domain) {
    joint_g3_error();
    first_domain = g3r_joint_domain;
    first_category = g3r_joint_category;
    first_code = g3r_joint_code;
  }
  if (et_g3r_private_g3t_save_hold_lifecycle_v1(hold->g3_hold) != 2)
    abort(); /* A rejected G3 end did not drain the original owner. */
  hold->g3_hold = NULL;
  hold->generator = hold->model = NULL;
  hold->optimizer = NULL;
  hold->live = 0u; /* Keep the registered shell; never recycle its address. */
  g3r_joint_domain = first_domain;
  g3r_joint_category = first_category;
  g3r_joint_code = first_code;
  return first_domain ? (first_domain == G3R_JOINT_O2
                             ? first_category : -first_category) : 0;
}

int64_t et_g3r_joint_hold_lifecycle_v1(
    const et_g3r_joint_hold *candidate) {
  for (const et_g3r_joint_hold *hold = g3r_joint_registry;
       hold; hold = hold->next)
    if (hold == candidate) {
      if (hold->live == 1u) return 1;
      if (hold->live == 0u) return 2;
      abort();
    }
  return 0;
}

int64_t et_g3r_joint_hold_last_domain_v1(void) { return g3r_joint_domain; }
int64_t et_g3r_joint_hold_last_category_v1(void) { return g3r_joint_category; }
int64_t et_g3r_joint_hold_last_code_v1(void) { return g3r_joint_code; }
