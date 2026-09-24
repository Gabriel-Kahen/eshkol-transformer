// Test-only allocation interception for the genuine TR3-C joint transaction.
// The failpoint is armed after all fixtures and snapshots exist.  On a hit it
// denies the real allocation once, disarms before production cleanup, and
// reports the last production stage reached.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "arena_memory.h"
#pragma clang diagnostic pop

#include <eshkol/eshkol.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" arena_tagged_cons_cell_t *__real_arena_allocate_cons_with_header(
    arena_t *);
extern "C" eshkol_tensor_t *__real_arena_allocate_tensor_with_header(arena_t *);
extern "C" ad_node_t *__real_arena_allocate_ad_node_with_header(arena_t *);
extern "C" void *__real_arena_allocate(arena_t *, size_t);
extern "C" void *__real_arena_allocate_with_header(
    arena_t *, size_t, uint8_t, uint8_t);
extern "C" eshkol_closure_t *__real_arena_allocate_closure_with_header(
    arena_t *, uint64_t, size_t, uint64_t, uint64_t, const char *);
extern "C" char *__real_arena_allocate_string_with_header(arena_t *, size_t);
namespace {
bool armed = false;
bool hit = false;
uint64_t allocation_class = 0;
uint64_t target = 0;
uint64_t attempts = 0;
uint64_t hit_stage = 0;
uint64_t current_stage = 0;
const char *hit_kind = nullptr;
std::array<uint64_t, 14> stage_attempts{};
uint64_t arena_attempts = 0;

constexpr const char *stage_name(uint64_t stage) {
  switch (stage) {
    case 1: return "parent";
    case 2: return "lease-enter";
    case 3: return "source-borrow";
    case 4: return "p1-clone";
    case 5: return "o2-moment-decode";
    case 6: return "controls-copy";
    case 7: return "source-close";
    case 8: return "d2-prepare";
    case 9: return "o2-prepare";
    case 10: return "i2-compose";
    case 11: return "cross-validate";
    case 12: return "i2-prepare-seal";
    case 13: return "irreversible-tail";
    default: return "unreached";
  }
}

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr,
      "TR3 joint allocation shim FAIL: %s target=%llu attempts=%llu "
      "stage=%llu hit=%d armed=%d\n",
      message, static_cast<unsigned long long>(target),
      static_cast<unsigned long long>(attempts),
      static_cast<unsigned long long>(current_stage), hit, armed);
  std::abort();
}

void require(bool condition, const char *message) {
  if (!condition) fail(message);
}

bool deny(const char *kind, uint64_t &kind_count) {
  if (!armed || allocation_class != 0) return false;
  require(current_stage >= 1 && current_stage <= 13,
          "allocation outside attributed transaction stage");
  ++kind_count;
  ++stage_attempts[current_stage];
  const uint64_t ordinal = attempts++;
  if (ordinal != target) return false;
  hit = true;
  hit_stage = current_stage;
  hit_kind = kind;
  armed = false;
  std::fprintf(stderr,
      "TR3-JOINT-ALLOC-DENY ordinal=%llu reached-stage=%s allocator=%s\n",
      static_cast<unsigned long long>(target), stage_name(hit_stage), hit_kind);
  return true;
}

template <class Result, class Call>
Result bounded_arena_failure(arena_t *arena, Call call) {
  require(arena != nullptr && arena->current_block != nullptr,
          "bounded arena allocation has no current block");
  arena_block_t *const block = arena->current_block;
  const bool prior_bounded = arena->bounded;
  const size_t prior_used = block->used;
  arena->bounded = true;
  block->used = block->size;
  Result result = call();
  arena->bounded = prior_bounded;
  block->used = prior_used;
  require(result == nullptr, "real bounded arena allocation did not fail");
  return result;
}
}  // namespace

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  if (deny("arena-vector", arena_attempts))
    return bounded_arena_failure<void *>(arena, [&]() {
      return __real_arena_allocate_vector_with_header(arena, capacity);
    });
  return __real_arena_allocate_vector_with_header(arena, capacity);
}

extern "C" arena_tagged_cons_cell_t *__wrap_arena_allocate_cons_with_header(
    arena_t *arena) {
  if (deny("arena-cons", arena_attempts))
    return bounded_arena_failure<arena_tagged_cons_cell_t *>(arena, [&]() {
      return __real_arena_allocate_cons_with_header(arena);
    });
  return __real_arena_allocate_cons_with_header(arena);
}

extern "C" eshkol_tensor_t *__wrap_arena_allocate_tensor_with_header(
    arena_t *arena) {
  if (deny("arena-tensor", arena_attempts))
    return bounded_arena_failure<eshkol_tensor_t *>(arena, [&]() {
      return __real_arena_allocate_tensor_with_header(arena);
    });
  return __real_arena_allocate_tensor_with_header(arena);
}

extern "C" ad_node_t *__wrap_arena_allocate_ad_node_with_header(
    arena_t *arena) {
  if (deny("arena-ad-node", arena_attempts))
    return bounded_arena_failure<ad_node_t *>(arena, [&]() {
      return __real_arena_allocate_ad_node_with_header(arena);
    });
  return __real_arena_allocate_ad_node_with_header(arena);
}

extern "C" void *__wrap_arena_allocate(arena_t *arena, size_t size) {
  if (deny("arena-raw", arena_attempts))
    return bounded_arena_failure<void *>(arena, [&]() {
      return __real_arena_allocate(arena, size);
    });
  return __real_arena_allocate(arena, size);
}

extern "C" void *__wrap_arena_allocate_with_header(
    arena_t *arena, size_t size, uint8_t subtype, uint8_t flags) {
  if (deny("arena-header", arena_attempts))
    return bounded_arena_failure<void *>(arena, [&]() {
      return __real_arena_allocate_with_header(arena, size, subtype, flags);
    });
  return __real_arena_allocate_with_header(arena, size, subtype, flags);
}

extern "C" eshkol_closure_t *__wrap_arena_allocate_closure_with_header(
    arena_t *arena, uint64_t fn, size_t captures, uint64_t sexpr,
    uint64_t return_type, const char *name) {
  if (deny("arena-closure", arena_attempts))
    return bounded_arena_failure<eshkol_closure_t *>(arena, [&]() {
      return __real_arena_allocate_closure_with_header(
          arena, fn, captures, sexpr, return_type, name);
    });
  return __real_arena_allocate_closure_with_header(
      arena, fn, captures, sexpr, return_type, name);
}

extern "C" char *__wrap_arena_allocate_string_with_header(
    arena_t *arena, size_t length) {
  if (deny("arena-string", arena_attempts))
    return bounded_arena_failure<char *>(arena, [&]() {
      return __real_arena_allocate_string_with_header(arena, length);
    });
  return __real_arena_allocate_string_with_header(arena, length);
}

extern "C" int64_t et_tr3_joint_allocation_arm_v1(
    int64_t class_code, int64_t ordinal) {
  require(class_code >= 0 && class_code <= 3 && ordinal >= 0 &&
              !armed && !hit && attempts == 0,
          "invalid allocation prefix arm");
  allocation_class = static_cast<uint64_t>(class_code);
  target = static_cast<uint64_t>(ordinal);
  armed = true;
  return 1;
}

extern "C" int64_t et_tr3_joint_allocation_stage_v1(int64_t stage) {
  if (!armed) return 1;
  require(stage >= 1 && stage <= 13, "invalid transaction stage");
  const uint64_t next = static_cast<uint64_t>(stage);
  if (next > current_stage) current_stage = next;
  return 1;
}

extern "C" int64_t et_tr3_joint_allocation_finish_v1(int64_t caught) {
  if (allocation_class != 0) {
    require(armed && !hit, "native allocation class changed shim state");
    armed = false;
    if (caught == 1) {
      require(current_stage >= 1 && current_stage <= 12,
              "native allocation failure escaped its recoverable prefix");
      hit = true;
      hit_stage = current_stage;
      hit_kind = allocation_class == 1 ? "f32" :
                 allocation_class == 2 ? "i2" : "o2";
      std::printf(
          "TR3-JOINT-ALLOC-CUT ordinal=%llu stage=%s allocator=%s\n",
          static_cast<unsigned long long>(target), stage_name(hit_stage),
          hit_kind);
      return static_cast<int64_t>(hit_stage);
    }
    require(caught == 0 && current_stage == 13,
            "native allocation prefix ended before the tail");
    return 0;
  }
  if (hit) {
    require(!armed && caught == 1 && hit_kind != nullptr && hit_stage != 0,
            "denied allocation did not unwind to caller");
    std::printf(
        "TR3-JOINT-ALLOC-CUT ordinal=%llu stage=%s allocator=%s\n",
        static_cast<unsigned long long>(target), stage_name(hit_stage), hit_kind);
    return static_cast<int64_t>(hit_stage);
  }
  require(armed && caught == 0 && current_stage == 13,
          "uninjected transaction did not reach irreversible tail");
  armed = false;
  std::printf(
      "TR3-JOINT-ALLOC-CENSUS total=%llu arena=%llu terminal-stage=%s\n",
      static_cast<unsigned long long>(attempts),
      static_cast<unsigned long long>(arena_attempts),
      stage_name(current_stage));
  return 0;
}

extern "C" int64_t et_tr3_joint_allocation_done_v1() {
  require(!armed, "allocation prefix remains armed");
  if (!hit) require(current_stage == 13, "successful prefix missed tail");
  return 1;
}
