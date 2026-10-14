#include "g3r_same_owner_hold_bridge.h"
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
#include "o2_optimizer_internal.h"
#include "eshkol_transformer/g3t_transport.h"
#include "eshkol_transformer/m3t_transport.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* One immutable shell per address: stale and double-release pointers never
 * become a later live pair. Caller serializes the entire feature. */
typedef struct et_g3r_hold_pair {
  struct et_g3r_hold_pair *next;
  void *generator, *model, *optimizer;
  void *g3;
  et_o2_g3r_save_hold *o2;
  unsigned live;
} et_g3r_hold_pair;
static et_g3r_hold_pair *pairs;
static unsigned pair_count;
static _Thread_local int64_t error_source, error_category, error_code;

enum { BRIDGE = 0, M3 = 1, G3 = 2, O2 = 3 };
static int64_t bridge_error(int64_t category, int64_t code) {
  error_source = BRIDGE; error_category = category; error_code = code;
  return category;
}
static void m3_error(void) {
  error_source = M3;
  error_category = et_m3t_private_last_error_category_v1();
  error_code = et_m3t_private_last_error_code_v1();
}
static void g3_error(void) {
  error_source = G3;
  error_category = et_g3t_private_last_error_category_v1();
  error_code = et_g3t_private_last_error_code_v1();
}
static void o2_error(const et_o2_error_v1 *error, int32_t status) {
  error_source = O2;
  /* O2's active-builder/buffer rejection deliberately leaves the error
   * record untouched. The returned status still carries its typed class. */
  error_category = error->category ? error->category : status;
  error_code = error->code ? error->code : ET_O2_CODE_OWNER_CONFLICT;
}
static void clear_error(void) {
  error_source = error_category = error_code = 0;
}
static et_g3r_hold_pair *admit(void *candidate) {
  for (et_g3r_hold_pair *p = pairs; p; p = p->next)
    if (p == candidate && p->live && p->g3 && p->o2) return p;
  bridge_error(2, 2); /* invalid-state, stale/foreign handle */
  return NULL;
}

void *et_g3r_private_same_owner_holds_begin_v1(
    void *generator, void *model, void *optimizer) {
  clear_error();
  if (!generator || !model || !optimizer || pair_count == 64u) {
    bridge_error(!generator || !model || !optimizer ? 1 : 4, 1);
    return NULL;
  }
  et_g3r_hold_pair *pair = calloc(1, sizeof(*pair));
  if (!pair) { bridge_error(5, 5); return NULL; }
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(model, parameters, identities)) {
    m3_error(); free(pair); return NULL;
  }
  pair->g3 = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!pair->g3) { g3_error(); free(pair); return NULL; }
  et_f32_parameter *o2_parameters[14];
  for (size_t i = 0; i < 14; ++i)
    o2_parameters[i] = (et_f32_parameter *)parameters[i];
  et_o2_error_v1 error = {0};
  const int32_t o2_status = et_g3r_private_o2_save_begin_v1(
      (et_o2_optimizer *)optimizer, o2_parameters, identities,
      &pair->o2, &error);
  if (o2_status) {
    o2_error(&error, o2_status);
    const int64_t first_source = error_source, first_category = error_category;
    const int64_t first_code = error_code;
    if (pair->o2) abort(); /* Failed begin promised an empty output slot. */
    /* G3 end drains even if its check reports a defect; ambiguity aborts. */
    (void)et_g3r_private_g3t_save_end_v1(pair->g3);
    error_source = first_source; error_category = first_category;
    error_code = first_code;
    free(pair);
    return NULL;
  }
  if (!pair->o2) abort();
  pair->generator = generator; pair->model = model; pair->optimizer = optimizer;
  pair->live = 1u;
  pair->next = pairs; pairs = pair; ++pair_count;
  return pair;
}

int64_t et_g3r_private_same_owner_holds_check_v1(
    void *candidate, void *generator, void *model, void *optimizer) {
  clear_error();
  et_g3r_hold_pair *pair = admit(candidate);
  if (!pair) return error_category;
  if (pair->generator != generator || pair->model != model ||
      pair->optimizer != optimizer)
    return bridge_error(2, 3);
  if (et_g3r_private_g3t_save_check_v1(pair->g3)) {
    g3_error(); return error_category;
  }
  et_o2_error_v1 error = {0};
  int32_t status = et_g3r_private_o2_save_owner_match_v1(
      pair->o2, (et_o2_optimizer *)optimizer, &error);
  if (status) { o2_error(&error, status); return error_category; }
  status = et_g3r_private_o2_save_check_v1(pair->o2, &error);
  if (status) { o2_error(&error, status); return error_category; }
  return 0;
}

void *et_g3r_private_same_owner_holds_g3_v1(void *candidate) {
  clear_error();
  et_g3r_hold_pair *pair = admit(candidate);
  return pair ? pair->g3 : NULL;
}
void *et_g3r_private_same_owner_holds_o2_v1(void *candidate) {
  clear_error();
  et_g3r_hold_pair *pair = admit(candidate);
  return pair ? pair->o2 : NULL;
}
int64_t et_g3r_private_same_owner_holds_copy_g3_v1(
    void *candidate, void *bytevector_header, int64_t payload_bytes) {
  clear_error();
  et_g3r_hold_pair *pair = admit(candidate);
  if (!pair) return error_category;
  enum { HELD_BYTES = 17 * 8, CARRIER_BYTES = 8 + HELD_BYTES };
  _Static_assert(sizeof(et_g3r_g3t_held_words_v1) == HELD_BYTES,
                 "G3 held image ABI changed");
  const uintptr_t address = (uintptr_t)bytevector_header;
  if (!bytevector_header || payload_bytes != HELD_BYTES ||
      address % _Alignof(int64_t) != 0u ||
      address > UINTPTR_MAX - CARRIER_BYTES)
    return bridge_error(1, 1);
  int64_t declared = -1;
  memcpy(&declared, bytevector_header, sizeof(declared));
  if (declared != HELD_BYTES) return bridge_error(1, 1);
  const uintptr_t end = address + CARRIER_BYTES;
  const uintptr_t pair_begin = (uintptr_t)pair;
  if (address < pair_begin + sizeof(*pair) && pair_begin < end)
    return bridge_error(1, 1);
  if (et_g3r_private_same_owner_holds_check_v1(
          pair, pair->generator, pair->model, pair->optimizer))
    return error_category;
  /* G3's copy rejects payload overlap with its original owner, A2, F32,
   * and held controls before writing any byte. The Eshkol wrapper allocates
   * this exact carrier; no detached snapshot or temporary lease is used. */
  void *payload = (unsigned char *)bytevector_header + sizeof(int64_t);
  if (et_g3r_private_g3t_save_copy_v1(
          pair->g3, (et_g3r_g3t_held_words_v1 *)payload,
          HELD_BYTES)) {
    g3_error(); return error_category;
  }
  clear_error();
  return 0;
}
int64_t et_g3r_private_same_owner_holds_end_v1(void *candidate) {
  clear_error();
  et_g3r_hold_pair *pair = admit(candidate);
  if (!pair) return error_category;
  /* Check before reverse-order draining. Preserve the earliest typed error. */
  (void)et_g3r_private_same_owner_holds_check_v1(
      pair, pair->generator, pair->model, pair->optimizer);
  int64_t first_source = error_source, first_category = error_category;
  int64_t first_code = error_code;
  et_o2_error_v1 error = {0};
  const int32_t o2_status = et_g3r_private_o2_save_end_v1(&pair->o2, &error);
  if (pair->o2) abort(); /* O2 end must prove the lease drained. */
  if (o2_status && !first_category) {
    o2_error(&error, o2_status);
    first_source = error_source; first_category = error_category;
    first_code = error_code;
  }
  const int64_t g3_status = et_g3r_private_g3t_save_end_v1(pair->g3);
  if (g3_status && !first_category) {
    g3_error();
    first_source = error_source; first_category = error_category;
    first_code = error_code;
  }
  pair->g3 = NULL; pair->generator = pair->model = pair->optimizer = NULL;
  pair->live = 0u;
  error_source = first_source; error_category = first_category;
  error_code = first_code;
  return first_category;
}
int64_t et_g3r_private_same_owner_holds_error_source_v1(void) { return error_source; }
int64_t et_g3r_private_same_owner_holds_error_category_v1(void) { return error_category; }
int64_t et_g3r_private_same_owner_holds_error_code_v1(void) { return error_code; }
void et_g3r_private_same_owner_holds_fail_stop_v1(void) { abort(); }
#endif
