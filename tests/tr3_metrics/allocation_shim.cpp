#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "arena_memory.h"
#pragma clang diagnostic pop

#include <cstdint>
#include <cstdlib>

extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);

namespace {
int mode = 0;
int hit = 0;
}

extern "C" int64_t et_tr3_metrics_fault_arm_v1() {
  if (mode != 0) std::abort();
  mode = 1;
  return 1;
}

extern "C" int64_t et_tr3_metrics_fault_arm_promotion_v1() {
  if (mode != 0) std::abort();
  mode = 2;
  return 1;
}

extern "C" int64_t et_tr3_metrics_fault_hit_v1() { return hit; }

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  if (mode != 1) return __real_arena_allocate_vector_with_header(arena, capacity);
  mode = 0;
  hit = 1;
  arena_block_t *block = arena->current_block;
  const bool prior_bounded = arena->bounded;
  const size_t prior_used = block->used;
  arena->bounded = true;
  block->used = block->size;
  void *result = __real_arena_allocate_vector_with_header(arena, capacity);
  arena->bounded = prior_bounded;
  block->used = prior_used;
  if (result != nullptr) std::abort();
  return result;
}

extern "C" void *real_region_allocate_quiet(arena_t *, size_t, size_t)
    asm("__real__Z28eshkol_region_allocate_quietP5arenamm");
extern "C" void *wrap_region_allocate_quiet(arena_t *, size_t, size_t)
    asm("__wrap__Z28eshkol_region_allocate_quietP5arenamm");
extern "C" void *wrap_region_allocate_quiet(
    arena_t *arena, size_t bytes, size_t alignment) {
  if (mode != 2) return real_region_allocate_quiet(arena, bytes, alignment);
  mode = 0;
  hit = 2;
  arena_block_t *block = arena->current_block;
  const bool prior_bounded = arena->bounded;
  const size_t prior_used = block->used;
  arena->bounded = true;
  block->used = block->size;
  void *result = real_region_allocate_quiet(arena, bytes, alignment);
  arena->bounded = prior_bounded;
  block->used = prior_used;
  if (result != nullptr) std::abort();
  return result;
}
