#ifndef ET_G3C4_P2_G2_HELD_CASE_WITNESS_H
#define ET_G3C4_P2_G2_HELD_CASE_WITNESS_H

#include <stddef.h>
#include <stdint.h>

/* Test-only authority for the current held case, including pointer reuse. */
typedef struct {
  uint64_t case_id;
  uint64_t marked_case_id;
  const void *context;
  const void *cache;
} et_g3c4_p2g2_held_case_witness;

static void et_g3c4_p2g2_held_case_begin(
    et_g3c4_p2g2_held_case_witness *witness) {
  witness->case_id++;
  witness->marked_case_id = 0;
  witness->context = NULL;
  witness->cache = NULL;
}

static void et_g3c4_p2g2_held_case_mark(
    et_g3c4_p2g2_held_case_witness *witness,
    const void *context, const void *cache) {
  witness->context = context;
  witness->cache = cache;
  witness->marked_case_id = witness->case_id;
}

static int et_g3c4_p2g2_held_case_current(
    const et_g3c4_p2g2_held_case_witness *witness,
    const void *context, const void *cache) {
  return witness->case_id != 0 &&
         witness->marked_case_id == witness->case_id &&
         witness->context == context && witness->cache == cache;
}

#endif
