// Test-only controls for the final-runtime E3/P1 acceptance profile.
// They use the runtime's real bounded-arena policy; no allocator is replaced.
#include "arena_memory.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *);

namespace {
struct ArenaLimit {
  arena_t *arena = nullptr;
  arena_block_t *block = nullptr;
  size_t used = 0;
  size_t limited_used = 0;
  size_t total = 0;
  size_t blocks = 0;
  bool bounded = false;
};

ArenaLimit active;
int64_t last_condition = 0;
int64_t last_used_delta = 0;
int64_t last_total_delta = 0;
int64_t last_block_delta = 0;

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "E3-P1 final-runtime shim failure: %s\n", message);
  std::abort();
}

size_t parse_budget() {
  const char *text = std::getenv("E3_P1_FAIL_BYTES");
  if (!text || !*text) fail("E3_P1_FAIL_BYTES is absent");
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno || !end || *end != '\0' || value > SIZE_MAX)
    fail("E3_P1_FAIL_BYTES is invalid");
  return static_cast<size_t>(value);
}

void restore_limit() {
  if (!active.arena || !active.block) fail("no arena limit is armed");
  if (active.arena->current_block != active.block)
    fail("bounded arena unexpectedly changed its current block");
  const size_t now_used = arena_get_used_memory(active.arena);
  const size_t now_total = arena_get_total_memory(active.arena);
  const size_t now_blocks = arena_get_block_count(active.arena);
  last_used_delta = static_cast<int64_t>(now_used) -
                    static_cast<int64_t>(active.limited_used);
  last_total_delta = static_cast<int64_t>(now_total) - static_cast<int64_t>(active.total);
  last_block_delta = static_cast<int64_t>(now_blocks) - static_cast<int64_t>(active.blocks);
  active.block->used = active.used;
  active.arena->bounded = active.bounded;
  active = {};
}
}  // namespace

extern "C" int64_t et_e3_p1_test_arena_used_v1() {
  return static_cast<int64_t>(arena_get_used_memory(get_global_arena_shared()));
}

extern "C" int64_t et_e3_p1_test_arena_total_v1() {
  return static_cast<int64_t>(arena_get_total_memory(get_global_arena_shared()));
}

extern "C" int64_t et_e3_p1_test_arena_blocks_v1() {
  return static_cast<int64_t>(arena_get_block_count(get_global_arena_shared()));
}

extern "C" int64_t et_e3_p1_test_disable_allocations_v1() {
  if (active.arena) fail("nested arena limit");
  arena_t *arena = get_global_arena_shared();
  if (!arena || !arena->current_block) fail("global arena unavailable");
  const size_t used = arena_get_used_memory(arena);
  active = {arena, arena->current_block, arena->current_block->used,
            used - arena->current_block->used + arena->current_block->size,
            arena_get_total_memory(arena), arena_get_block_count(arena),
            arena->bounded};
  arena->bounded = true;
  arena->current_block->used = arena->current_block->size;
  return 0;
}

extern "C" int64_t et_e3_p1_test_enable_allocations_v1() {
  restore_limit();
  return 0;
}

extern "C" int64_t et_e3_p1_test_fail_arm_v1() {
  if (active.arena) fail("nested failure injection");
  const char *scope = std::getenv("E3_P1_FAIL_SCOPE");
  eshkol_region_t *region = region_current();
  if (!scope || !region) fail("failure scope or current region unavailable");
  arena_t *arena = nullptr;
  if (!std::strcmp(scope, "region")) arena = region->arena;
  else if (!std::strcmp(scope, "root")) arena = region->escape_base;
  else fail("E3_P1_FAIL_SCOPE must be region or root");
  if (!arena || !arena->current_block) fail("selected arena unavailable");
  const size_t budget = parse_budget();
  if (budget > arena->current_block->size - arena->current_block->used)
    fail("failure budget exceeds current block capacity");
  const size_t used = arena_get_used_memory(arena);
  active = {arena, arena->current_block, arena->current_block->used,
            used - arena->current_block->used + arena->current_block->size - budget,
            arena_get_total_memory(arena), arena_get_block_count(arena),
            arena->bounded};
  last_condition = 0;
  last_used_delta = last_total_delta = last_block_delta = 0;
  arena->bounded = true;
  arena->current_block->used = arena->current_block->size - budget;
  return static_cast<int64_t>(budget);
}

extern "C" int64_t et_e3_p1_test_fail_caught_v1() {
  const char *scope = std::getenv("E3_P1_FAIL_SCOPE");
  const bool root = scope && !std::strcmp(scope, "root");
  restore_limit();
  eshkol_tagged_value_t value{};
  eshkol_get_raised_value(&value);
  if (value.type != ESHKOL_VALUE_HEAP_PTR || value.flags || value.reserved)
    fail("caught allocation condition has a noncanonical tag");
  auto *exception = reinterpret_cast<eshkol_exception_t *>(value.data.ptr_val);
  if (!exception || exception != g_current_exception)
    fail("caught allocation condition identity differs from runtime state");
  const char *expected = root ? "region promotion allocation failed"
                              : "object or exception-handler allocation failed";
  if (std::strcmp(exception->message, expected))
    fail("caught allocation condition message differs");
  last_condition = root ? 2 : 1;
  return last_condition;
}

extern "C" int64_t et_e3_p1_test_fail_disarm_v1() {
  restore_limit();
  return 0;
}

extern "C" int64_t et_e3_p1_test_fail_condition_v1() {
  return last_condition;
}

extern "C" int64_t et_e3_p1_test_fail_used_delta_v1() {
  return last_used_delta;
}

extern "C" int64_t et_e3_p1_test_fail_total_delta_v1() {
  return last_total_delta;
}

extern "C" int64_t et_e3_p1_test_fail_block_delta_v1() {
  return last_block_delta;
}
