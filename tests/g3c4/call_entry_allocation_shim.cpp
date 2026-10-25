// Test-only constructor and handler allocation interception.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include "arena_memory.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void *__real_malloc(size_t);
extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" void *__real_arena_allocate_cons_with_header(arena_t *);
extern "C" void *__real_arena_allocate_with_header(
    arena_t *, size_t, uint8_t, uint8_t);
extern "C" void *__real_arena_allocate_aligned(arena_t *, size_t, size_t);
extern "C" void __real_eshkol_push_exception_handler(void *);

static int allocation_kind;
static int64_t allocation_before;
static int64_t allocation_seen;
static int allocation_consumed;
static int capture_active;
static void *captured_bytevectors[2];
static int captured_bytevector_count;
static int handler_preparing;
static int handler_fresh;
static int handler_armed;
static int handler_in_push;
static int64_t handler_pushes;
static void require(bool condition, const char *message);

// Test-only provenance. Capture the allocator request and its backing bump
// allocation at construction, before an Eshkol record can expose a carrier.
struct extent_entry {
  void *payload;
  void *raw;
  size_t requested;
  size_t allocated;
  size_t alignment;
  uint8_t flags;
};
static extent_entry extent_entries[8];
static int extent_active;
static int extent_inside_bytevector;
static int extent_count;
static int extent_overflow;
static void *extent_raw;
static size_t extent_allocated;
static size_t extent_alignment;

extern "C" int64_t et_g3c4_terminal_extent_arm_v1(void) {
  require(!extent_active && !extent_inside_bytevector, "extent arm idle");
  for (auto &entry : extent_entries) entry = {};
  extent_count = 0;
  extent_overflow = 0;
  extent_active = 1;
  return 0;
}

extern "C" int64_t et_g3c4_terminal_extent_seal_v1(void) {
  extent_active = 0;
  if (extent_overflow || extent_count != 8) return 0;
  for (int i = 0; i < extent_count; ++i) {
    if (!extent_entries[i].payload || !extent_entries[i].raw) return 0;
    for (int j = 0; j < i; ++j)
      if (extent_entries[i].payload == extent_entries[j].payload) return 0;
  }
  return 1;
}

// All comparisons use addresses and sizes captured by wrappers. In particular,
// no checker derives an object header by subtracting from an input pointer.
extern "C" int64_t et_g3c4_terminal_extent_match_v1(
    int64_t slot, void *candidate, int64_t requested, int64_t allocated) {
  if (extent_active || extent_overflow || extent_count != 8 || slot < 0 ||
      slot >= 8 || requested < 0 || allocated < 0) return 0;
  const extent_entry &entry = extent_entries[slot];
  const auto raw = reinterpret_cast<std::uintptr_t>(entry.raw);
  return entry.payload != nullptr && entry.raw != nullptr &&
         candidate == entry.payload && entry.requested == size_t(requested) &&
         entry.allocated == size_t(allocated) && entry.alignment == 8 &&
         entry.flags == 0 &&
         raw <= UINTPTR_MAX - 8 &&
         reinterpret_cast<std::uintptr_t>(candidate) == raw + 8 ? 1 : 0;
}

extern "C" void *__wrap_arena_allocate_aligned(
    arena_t *arena, size_t size, size_t alignment) {
  void *raw = __real_arena_allocate_aligned(arena, size, alignment);
  if (extent_inside_bytevector) {
    extent_raw = raw;
    extent_allocated = size;
    extent_alignment = alignment;
  }
  return raw;
}

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr,
                 "G3-C4 allocation shim FAIL: %s kind=%d before=%lld seen=%lld "
                 "consumed=%d pushes=%lld\n",
                 message, allocation_kind, (long long)allocation_before,
                 (long long)allocation_seen, allocation_consumed,
                 (long long)handler_pushes);
    std::abort();
  }
}

extern "C" int64_t et_g3c4_call_entry_alloc_arm_v1(
    int64_t kind, int64_t before) {
  require(!allocation_kind && !handler_armed && kind >= 1 && kind <= 3 &&
              before >= 0,
          "valid vector/cons arm");
  allocation_kind = (int)kind;
  allocation_before = before;
  allocation_seen = 0;
  allocation_consumed = 0;
  capture_active = 1;
  captured_bytevectors[0] = nullptr;
  captured_bytevectors[1] = nullptr;
  captured_bytevector_count = 0;
  return 0;
}

static bool allocation_should_fail(int kind) {
  if (allocation_kind != kind) return false;
  if (allocation_seen++ != allocation_before) return false;
  allocation_kind = 0;
  allocation_consumed = 1;
  return true;
}

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  return allocation_should_fail(1)
             ? nullptr
             : __real_arena_allocate_vector_with_header(arena, capacity);
}

extern "C" void *__wrap_arena_allocate_cons_with_header(arena_t *arena) {
  return allocation_should_fail(2)
             ? nullptr
             : __real_arena_allocate_cons_with_header(arena);
}

extern "C" void *__wrap_arena_allocate_with_header(
    arena_t *arena, size_t data_size, uint8_t subtype, uint8_t flags) {
  if (subtype == HEAP_SUBTYPE_BYTEVECTOR && allocation_should_fail(3))
    return nullptr;
  const bool record = extent_active && subtype == HEAP_SUBTYPE_BYTEVECTOR;
  if (record) {
    extent_inside_bytevector = 1;
    extent_raw = nullptr;
    extent_allocated = 0;
    extent_alignment = 0;
  }
  void *result = __real_arena_allocate_with_header(
      arena, data_size, subtype, flags);
  if (record) {
    extent_inside_bytevector = 0;
    if (extent_count == 8) {
      extent_overflow = 1;
    } else {
      extent_entries[extent_count++] =
          {result, extent_raw, data_size, extent_allocated, extent_alignment,
           flags};
    }
  }
  if (capture_active && subtype == HEAP_SUBTYPE_BYTEVECTOR && result != nullptr &&
      captured_bytevector_count < 2)
    captured_bytevectors[captured_bytevector_count++] = result;
  return result;
}

extern "C" int64_t et_g3c4_call_entry_alloc_capture_zero_v1(
    int64_t slot, int64_t expected_length) {
  if (slot < 0 || slot > 1 || expected_length < -1) return 0;
  if (expected_length == -1)
    return captured_bytevectors[slot] == nullptr ? 1 : 0;
  const void *candidate = captured_bytevectors[slot];
  if (candidate == nullptr || *(const int64_t *)candidate != expected_length)
    return 0;
  const unsigned char *bytes = (const unsigned char *)candidate + 8;
  for (int64_t i = 0; i < expected_length; ++i)
    if (bytes[i] != 0u) return 0;
  return 1;
}

extern "C" int64_t et_g3c4_call_entry_alloc_consumed_v1(void) {
  return allocation_consumed;
}

extern "C" int64_t et_g3c4_call_entry_alloc_disarm_v1(void) {
  const int64_t seen = allocation_seen;
  allocation_kind = 0;
  allocation_consumed = 0;
  capture_active = 0;
  return seen;
}

extern "C" void *__wrap_malloc(size_t bytes) {
  if (bytes == sizeof(eshkol_exception_handler_t) && handler_in_push) {
    if (handler_preparing) handler_fresh = 1;
    if (handler_armed && handler_pushes == 3) {
      handler_armed = 0;
      allocation_consumed = 1;
      return nullptr;
    }
  }
  return __real_malloc(bytes);
}

extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  handler_in_push = 1;
  if (handler_armed) ++handler_pushes;
  __real_eshkol_push_exception_handler(buffer);
  handler_in_push = 0;
}

extern "C" int64_t et_g3c4_call_entry_handler_prepare_v1(void) {
  require(!allocation_kind && !handler_preparing && !handler_armed,
          "handler preparation starts idle");
  handler_preparing = 1;
  handler_fresh = 0;
  handler_pushes = 0;
  allocation_consumed = 0;
  return 0;
}

extern "C" int64_t et_g3c4_call_entry_handler_fresh_v1(void) {
  return handler_fresh;
}

extern "C" int64_t et_g3c4_call_entry_handler_arm_v1(void) {
  require(handler_preparing && handler_fresh && !handler_armed,
          "handler arm follows a fresh allocation");
  handler_preparing = 0;
  handler_armed = 1;
  handler_pushes = 0;
  return 0;
}

extern "C" int64_t et_g3c4_call_entry_handler_caught_v1(int64_t idle) {
  handler_in_push = 0;
  require(!handler_armed && allocation_consumed && handler_pushes == 3 && idle,
          "third production handler allocation failed before enrollment");
  allocation_consumed = 0;
  return 0;
}
