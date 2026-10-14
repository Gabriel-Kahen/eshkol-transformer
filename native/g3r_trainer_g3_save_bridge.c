/* Feature-private same-owner G3 hold entry. The Eshkol caller first owns an
 * exact TR3 snapshot lease; this bridge never grants trainer authority. */
#if !defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE) || \
    !defined(ET_G3R_COMMITTED_HISTORY_PRIVATE)
#error "trainer G3 save bridge requires the held G3/M3T feature contracts"
#endif
#include "eshkol_transformer/g3t_transport.h"
#include "eshkol_transformer/m3t_transport.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct g3r_trainer_g3_hold {
  struct g3r_trainer_g3_hold *next;
  void *native;
  int live;
} g3r_trainer_g3_hold;

static g3r_trainer_g3_hold *g3r_trainer_g3_holds;
static _Thread_local int64_t g3r_trainer_g3_error_source;

static g3r_trainer_g3_hold *g3r_trainer_g3_find(void *candidate) {
  for (g3r_trainer_g3_hold *item = g3r_trainer_g3_holds;
       item; item = item->next)
    if (item == candidate) return item;
  return NULL;
}

/* 1=M3T roster, 2=G3 held owner, 3=bridge allocation/authority. */
int64_t et_g3r_private_trainer_g3_save_last_source_v1(void) {
  return g3r_trainer_g3_error_source;
}

void *et_g3r_private_trainer_g3_save_begin_v1(void *generator,
                                              void *sealed_model) {
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  g3r_trainer_g3_error_source = 0;
  if (et_m3t_private_owner_save_roster_v1(
          sealed_model, parameters, identities) != 0) {
    g3r_trainer_g3_error_source = 1;
    return NULL;
  }
  g3r_trainer_g3_hold *item = calloc(1, sizeof(*item));
  if (!item) {
    g3r_trainer_g3_error_source = 3;
    return NULL;
  }
  item->native = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!item->native) {
    g3r_trainer_g3_error_source = 2;
    free(item);
    return NULL;
  }
  item->live = 1;
  item->next = g3r_trainer_g3_holds;
  g3r_trainer_g3_holds = item;
  return item;
}

int64_t et_g3r_private_trainer_g3_save_check_v1(void *candidate) {
  g3r_trainer_g3_hold *item = g3r_trainer_g3_find(candidate);
  if (!item || !item->live || !item->native) {
    g3r_trainer_g3_error_source = 3;
    return -1;
  }
  int64_t status = et_g3r_private_g3t_save_check_v1(item->native);
  g3r_trainer_g3_error_source = status ? 2 : 0;
  return status;
}

/* Native end drains a valid hold even when its final invariant check fails.
 * The Eshkol owner treats every nonzero result as uncertain and fails stop. */
int64_t et_g3r_private_trainer_g3_save_end_v1(void *candidate) {
  g3r_trainer_g3_hold *item = g3r_trainer_g3_find(candidate);
  if (!item || !item->live || !item->native) {
    g3r_trainer_g3_error_source = 3;
    return -1;
  }
  int64_t status = et_g3r_private_g3t_save_end_v1(item->native);
  item->native = NULL;
  item->live = 0;
  g3r_trainer_g3_error_source = status ? 2 : 0;
  return status;
}
