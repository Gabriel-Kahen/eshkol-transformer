// Test-only Eshkol vector/cons cut points; no G3-T ownership authority.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include "arena_memory.h"
#include <cstdint>

extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" void *__real_arena_allocate_cons_with_header(arena_t *);

static int64_t kind, before, seen, consumed;

extern "C" int64_t et_g3r_rng_alloc_arm_v1(int64_t requested_kind,
                                              int64_t requested_before) {
  if (kind || requested_kind < 1 || requested_kind > 2 ||
      requested_before < 0) return -1;
  kind = requested_kind;
  before = requested_before;
  seen = consumed = 0;
  return 0;
}

static bool should_fail(int64_t requested_kind) {
  if (kind != requested_kind) return false;
  if (seen++ != before) return false;
  kind = 0;
  consumed = 1;
  return true;
}

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  return should_fail(1) ? nullptr
                        : __real_arena_allocate_vector_with_header(arena,
                                                                    capacity);
}

extern "C" void *__wrap_arena_allocate_cons_with_header(arena_t *arena) {
  return should_fail(2) ? nullptr
                        : __real_arena_allocate_cons_with_header(arena);
}

extern "C" int64_t et_g3r_rng_alloc_consumed_v1(void) { return consumed; }
extern "C" int64_t et_g3r_rng_alloc_disarm_v1(void) {
  const int64_t count = seen;
  kind = consumed = 0;
  return count;
}
