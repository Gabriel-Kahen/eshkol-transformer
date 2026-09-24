// Test-only linker interception around the frozen production runtime.  It
// denies actual allocation calls; it does not manufacture exception objects or
// replace the checked-promotion implementation.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "arena_memory.h"
#include "runtime_region_promotion_internal.h"
#pragma clang diagnostic pop

#define ET_F32_TENSOR_TESTING 1
#define ET_O2_TESTING 1
#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <eshkol/eshkol.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" et_f32_tensor *et_m3t_test_parameter_gradient(et_f32_parameter *);

extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" arena_tagged_cons_cell_t *__real_arena_allocate_cons_with_header(arena_t *);
extern "C" eshkol_tensor_t *__real_arena_allocate_tensor_with_header(arena_t *);
extern "C" ad_node_t *__real_arena_allocate_ad_node_with_header(arena_t *);
extern "C" void *__real_arena_allocate(arena_t *, size_t);
extern "C" void *__real_arena_allocate_with_header(
    arena_t *, size_t, uint8_t, uint8_t);
extern "C" eshkol_closure_t *__real_arena_allocate_closure_with_header(
    arena_t *, uint64_t, size_t, uint64_t, uint64_t, const char *);
extern "C" char *__real_arena_allocate_string_with_header(arena_t *, size_t);
extern "C" void *real_region_allocate_quiet(arena_t *, size_t, size_t)
    asm("__real__Z28eshkol_region_allocate_quietP5arenamm");
extern "C" void *__real__Znwm(size_t);
extern "C" void *__real_malloc(size_t);
extern "C" void __real_eshkol_push_exception_handler(void *);
extern "C" void __real_eshkol_get_raised_value(eshkol_tagged_value_t *);
extern "C" int32_t __real_eshkol_region_write_barrier_checked_v1(
    eshkol_tagged_value_t *, const void *, const eshkol_tagged_value_t *);

namespace {
enum class Mode { idle, object, all_object, promotion, target, handler, poststage };
Mode mode = Mode::idle;
void *lease_root = nullptr;
uint64_t fail_after = 0;
uint64_t attempts = 0;
uint64_t vector_attempts = 0;
uint64_t cons_attempts = 0;
uint64_t tensor_attempts = 0;
uint64_t ad_node_attempts = 0;
uint64_t raw_attempts = 0;
uint64_t header_attempts = 0;
uint64_t closure_attempts = 0;
uint64_t string_attempts = 0;
uint64_t promotion_barriers = 0;
uint64_t object_failures = 0;
uint64_t promotion_failures = 0;
uint64_t target_failures = 0;
uint64_t handler_failures = 0;
uint64_t poststage_allocations = 0;
uint64_t poststage_root_barriers = 0;
bool hit = false;
bool observed = false;
bool promotion_active = false;
bool target_active = false;
bool poststage_window = false;
bool poststage_complete = false;
bool preparing_handler = false;
bool fresh_handler = false;
bool in_push = false;
bool all_object_armed = false;

constexpr size_t parameter_count = 14;
constexpr size_t max_elements = 1024;
struct ParameterSnapshot {
  size_t elements;
  uint32_t value[max_elements];
  uint32_t gradient[max_elements];
  et_f32_gradient_metadata_v1 metadata;
};
struct OptimizerSnapshot {
  uint64_t completed_updates;
  size_t elements[parameter_count];
  uint32_t moments[parameter_count][2][max_elements];
};
ParameterSnapshot parameters[parameter_count];
OptimizerSnapshot optimizer;
bool have_snapshot = false;

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr,
      "TR3 lease failure shim FAIL: %s mode=%d after=%llu attempts=%llu "
      "objects=%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu "
      "promotions=%llu hit=%d observed=%d\n",
      message, static_cast<int>(mode),
      static_cast<unsigned long long>(fail_after),
      static_cast<unsigned long long>(attempts),
      static_cast<unsigned long long>(vector_attempts),
      static_cast<unsigned long long>(cons_attempts),
      static_cast<unsigned long long>(tensor_attempts),
      static_cast<unsigned long long>(ad_node_attempts),
      static_cast<unsigned long long>(raw_attempts),
      static_cast<unsigned long long>(header_attempts),
      static_cast<unsigned long long>(closure_attempts),
      static_cast<unsigned long long>(string_attempts),
      static_cast<unsigned long long>(promotion_barriers), hit, observed);
  std::abort();
}
void require(bool condition, const char *message) {
  if (!condition) fail(message);
}
void arm(Mode next, uint64_t ordinal) {
  require(mode == Mode::idle && !hit, "failpoint already active");
  mode = next;
  all_object_armed = next == Mode::all_object;
  fail_after = ordinal;
  attempts = vector_attempts = cons_attempts = tensor_attempts = 0;
  ad_node_attempts = 0;
  raw_attempts = header_attempts = closure_attempts = string_attempts = 0;
  promotion_barriers = 0;
  hit = observed = false;
}
bool deny_object(uint64_t &kind_attempts) {
  if (mode != Mode::object && mode != Mode::all_object) return false;
  ++kind_attempts;
  const uint64_t current = attempts++;
  if (current != fail_after) return false;
  hit = true;
  ++object_failures;
  // Cleanup and exception observation must use the ordinary runtime.
  mode = Mode::idle;
  return true;
}
void note_poststage_allocation() {
  if (mode == Mode::poststage && poststage_window) ++poststage_allocations;
}
template <class Result, class Call>
Result bounded_object_failure(arena_t *arena, Call call) {
  require(arena && arena->current_block, "bounded object arena");
  arena_block_t *block = arena->current_block;
  const bool prior_bounded = arena->bounded;
  const size_t prior_used = block->used;
  arena->bounded = true;
  block->used = block->size;
  Result result = call();
  arena->bounded = prior_bounded;
  block->used = prior_used;
  require(result == nullptr, "real bounded object allocation did not fail");
  return result;
}
void copy_parameter(ParameterSnapshot *out, et_f32_parameter *parameter) {
  require(out && parameter, "parameter snapshot operands");
  const et_f32_tensor *value = nullptr;
  et_f32_tensor_error error{};
  require(et_f32_parameter_value_tensor_v1(parameter, &value, &error) == 0,
          "parameter value lookup");
  et_f32_tensor *gradient = et_m3t_test_parameter_gradient(parameter);
  require(gradient, "parameter gradient lookup");
  size_t value_count = 0;
  size_t gradient_count = 0;
  require(et_f32_tensor_element_count_v1(value, &value_count, &error) == 0 &&
              et_f32_tensor_element_count_v1(
                  gradient, &gradient_count, &error) == 0 &&
              value_count == gradient_count && value_count <= max_elements,
          "parameter element count");
  std::memset(out, 0, sizeof(*out));
  out->elements = value_count;
  require(et_f32_tensor_copy_bits_to_v1(
              value, out->value, value_count, &error) == 0 &&
              et_f32_tensor_copy_bits_to_v1(
                  gradient, out->gradient, gradient_count, &error) == 0,
          "parameter exact bit copy");
  out->metadata.struct_size = sizeof(out->metadata);
  require(et_f32_parameter_gradient_metadata_v1(
              parameter, &out->metadata, &error) == 0,
          "parameter gradient metadata");
}
void copy_optimizer(OptimizerSnapshot *out, et_o2_optimizer *candidate) {
  require(out && candidate, "optimizer snapshot operands");
  std::memset(out, 0, sizeof(*out));
  et_o2_error_v1 error{};
  require(et_o2_optimizer_completed_updates_v1(
              candidate, &out->completed_updates, &error) == 0,
          "optimizer completed-update observation");
  for (size_t index = 0; index < parameter_count; ++index) {
    out->elements[index] = parameters[index].elements;
    for (uint32_t kind = 0; kind < 2; ++kind) {
      require(et_o2_test_optimizer_moment_bits_v1(
                  candidate, index, kind, out->moments[index][kind],
                  out->elements[index], &error) == 0,
              "optimizer exact moment copy");
    }
  }
}
} // namespace

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  note_poststage_allocation();
  if (deny_object(vector_attempts))
    return bounded_object_failure<void *>(arena, [&]() {
      return __real_arena_allocate_vector_with_header(arena, capacity);
    });
  return __real_arena_allocate_vector_with_header(arena, capacity);
}
extern "C" arena_tagged_cons_cell_t *__wrap_arena_allocate_cons_with_header(
    arena_t *arena) {
  note_poststage_allocation();
  if (deny_object(cons_attempts))
    return bounded_object_failure<arena_tagged_cons_cell_t *>(arena, [&]() {
      return __real_arena_allocate_cons_with_header(arena);
    });
  return __real_arena_allocate_cons_with_header(arena);
}
extern "C" eshkol_tensor_t *__wrap_arena_allocate_tensor_with_header(
    arena_t *arena) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(tensor_attempts))
    return bounded_object_failure<eshkol_tensor_t *>(arena, [&]() {
      return __real_arena_allocate_tensor_with_header(arena);
    });
  if (mode == Mode::object) ++tensor_attempts;
  return __real_arena_allocate_tensor_with_header(arena);
}
extern "C" ad_node_t *__wrap_arena_allocate_ad_node_with_header(
    arena_t *arena) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(ad_node_attempts))
    return bounded_object_failure<ad_node_t *>(arena, [&]() {
      return __real_arena_allocate_ad_node_with_header(arena);
    });
  if (mode == Mode::object) ++ad_node_attempts;
  return __real_arena_allocate_ad_node_with_header(arena);
}
extern "C" void *__wrap_arena_allocate(arena_t *arena, size_t size) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(raw_attempts))
    return bounded_object_failure<void *>(arena, [&]() {
      return __real_arena_allocate(arena, size);
    });
  if (mode == Mode::object) ++raw_attempts;
  return __real_arena_allocate(arena, size);
}
extern "C" void *__wrap_arena_allocate_with_header(
    arena_t *arena, size_t size, uint8_t subtype, uint8_t flags) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(header_attempts))
    return bounded_object_failure<void *>(arena, [&]() {
      return __real_arena_allocate_with_header(arena, size, subtype, flags);
    });
  if (mode == Mode::object) ++header_attempts;
  return __real_arena_allocate_with_header(
      arena, size, subtype, flags);
}
extern "C" eshkol_closure_t *__wrap_arena_allocate_closure_with_header(
    arena_t *arena, uint64_t fn, size_t captures, uint64_t sexpr,
    uint64_t return_type, const char *name) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(closure_attempts))
    return bounded_object_failure<eshkol_closure_t *>(arena, [&]() {
      return __real_arena_allocate_closure_with_header(
          arena, fn, captures, sexpr, return_type, name);
    });
  if (mode == Mode::object) ++closure_attempts;
  return __real_arena_allocate_closure_with_header(
      arena, fn, captures, sexpr, return_type, name);
}
extern "C" char *__wrap_arena_allocate_string_with_header(
    arena_t *arena, size_t length) {
  note_poststage_allocation();
  if (mode == Mode::all_object && deny_object(string_attempts))
    return bounded_object_failure<char *>(arena, [&]() {
      return __real_arena_allocate_string_with_header(arena, length);
    });
  if (mode == Mode::object) ++string_attempts;
  return __real_arena_allocate_string_with_header(arena, length);
}

extern "C" void *__wrap__Znwm(size_t bytes) {
  note_poststage_allocation();
  if (mode == Mode::promotion && promotion_active) {
    const uint64_t current = attempts++;
    if (current == fail_after) {
      hit = true;
      ++promotion_failures;
      throw std::bad_alloc();
    }
  }
  return __real__Znwm(bytes);
}

extern "C" void *wrap_region_allocate_quiet(
    arena_t *arena, size_t bytes, size_t alignment)
    asm("__wrap__Z28eshkol_region_allocate_quietP5arenamm");
extern "C" void *wrap_region_allocate_quiet(
    arena_t *arena, size_t bytes, size_t alignment) {
  note_poststage_allocation();
  if (mode == Mode::target && target_active) {
    const uint64_t current = attempts++;
    if (current == fail_after) {
      hit = true;
      arena_block_t *block = arena->current_block;
      const bool prior_bounded = arena->bounded;
      const size_t prior_used = block->used;
      arena->bounded = true;
      block->used = block->size;
      void *result = real_region_allocate_quiet(arena, bytes, alignment);
      arena->bounded = prior_bounded;
      block->used = prior_used;
      require(result == nullptr,
              "real bounded promotion target allocation did not fail");
      return result;
    }
  }
  return real_region_allocate_quiet(arena, bytes, alignment);
}

extern "C" int32_t __wrap_eshkol_region_write_barrier_checked_v1(
    eshkol_tagged_value_t *out, const void *owner,
    const eshkol_tagged_value_t *input) {
  if (lease_root && owner == lease_root &&
      (mode == Mode::promotion || mode == Mode::target)) {
    ++promotion_barriers;
    if (mode == Mode::promotion) {
      promotion_active = true;
      const int32_t status = __real_eshkol_region_write_barrier_checked_v1(
          out, owner, input);
      promotion_active = false;
      if (status != 0) {
        require(hit && status == 1, "promotion prefix returned wrong status");
        mode = Mode::idle;
      }
      return status;
    }

    target_active = true;
    const int32_t status = __real_eshkol_region_write_barrier_checked_v1(
        out, owner, input);
    target_active = false;
    if (status != 0) {
      require(hit && status == 1,
              "target prefix returned wrong promotion status");
      ++target_failures;
      mode = Mode::idle;
    }
    return status;
  }
  if (lease_root && owner == lease_root && mode == Mode::poststage) {
    const int32_t status = __real_eshkol_region_write_barrier_checked_v1(
        out, owner, input);
    require(status == 0, "post-staging registry barrier failed");
    ++poststage_root_barriers;
    if (poststage_root_barriers == 1) poststage_window = true;
    else if (poststage_root_barriers == 3) {
      poststage_complete = true;
    }
    return status;
  }
  return __real_eshkol_region_write_barrier_checked_v1(out, owner, input);
}

extern "C" void *__wrap_malloc(size_t bytes) {
  note_poststage_allocation();
  if (bytes == sizeof(eshkol_exception_handler_t) && in_push) {
    if (preparing_handler) fresh_handler = true;
    if (mode == Mode::handler) {
      hit = true;
      ++handler_failures;
      mode = Mode::idle;
      return nullptr;
    }
  }
  return __real_malloc(bytes);
}
extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  in_push = true;
  __real_eshkol_push_exception_handler(buffer);
  in_push = false;
}
extern "C" void __wrap_eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  __real_eshkol_get_raised_value(value);
  in_push = false; // a failed push transfers around the wrapper epilogue
  if (hit && !observed) {
    require(value && value->type == ESHKOL_VALUE_HEAP_PTR &&
                value->data.ptr_val != 0 && g_current_exception,
            "allocation condition tag");
    auto *exception = reinterpret_cast<eshkol_exception_t *>(
        value->data.ptr_val);
    require(exception == g_current_exception && exception->message,
            "allocation condition identity");
    const bool constructor =
        std::strcmp(exception->message,
                    "object or exception-handler allocation failed") == 0;
    const bool promotion =
        std::strcmp(exception->message,
                    "region promotion allocation failed") == 0;
    // The E1 shell is opaque here. The Eshkol guard checks its actual
    // category and operation before treating it as an expected failure.
    const bool e1_shell =
        std::strcmp(exception->message, "transformer.error_internal:v1") == 0;
    require(constructor || promotion || (all_object_armed && e1_shell),
            "unexpected allocation condition");
    observed = true;
  }
}

extern "C" int64_t et_tr3_lf_root_v1(void *root) {
  require(mode == Mode::idle && root, "lease root registration");
  lease_root = root;
  return 1;
}
extern "C" int64_t et_tr3_lf_object_arm_v1(int64_t ordinal) {
  require(ordinal >= 0, "object prefix ordinal");
  arm(Mode::object, static_cast<uint64_t>(ordinal));
  return 1;
}
extern "C" int64_t et_tr3_lf_all_arm_v1(int64_t ordinal) {
  require(ordinal >= 0, "all-object prefix ordinal");
  arm(Mode::all_object, static_cast<uint64_t>(ordinal));
  return 1;
}
extern "C" int64_t et_tr3_lf_promotion_arm_v1(int64_t ordinal) {
  require(ordinal >= 0, "promotion prefix ordinal");
  arm(Mode::promotion, static_cast<uint64_t>(ordinal));
  return 1;
}
extern "C" int64_t et_tr3_lf_target_arm_v1(int64_t ordinal) {
  require(ordinal >= 0, "target prefix ordinal");
  arm(Mode::target, static_cast<uint64_t>(ordinal));
  return 1;
}
extern "C" int64_t et_tr3_lf_handler_prepare_v1() {
  require(mode == Mode::idle && !preparing_handler, "handler preparation");
  preparing_handler = true;
  fresh_handler = false;
  return 1;
}
extern "C" int64_t et_tr3_lf_handler_fresh_v1() {
  return fresh_handler ? 1 : 0;
}
extern "C" int64_t et_tr3_lf_handler_arm_v1() {
  require(preparing_handler && fresh_handler && mode == Mode::idle,
          "handler pool exhaustion");
  preparing_handler = false;
  arm(Mode::handler, 0);
  return 1;
}
extern "C" int64_t et_tr3_lf_poststage_arm_v1() {
  require(mode == Mode::idle && !poststage_window,
          "post-staging census already active");
  mode = Mode::poststage;
  poststage_allocations = 0;
  poststage_root_barriers = 0;
  poststage_complete = false;
  return 1;
}
extern "C" int64_t et_tr3_lf_poststage_finish_v1() {
  require(mode == Mode::poststage && poststage_complete &&
              poststage_window && poststage_root_barriers == 3,
          "post-staging interval did not reach enrollment");
  require(poststage_allocations == 0,
          "post-staging interval attempted an allocation");
  poststage_window = false;
  mode = Mode::idle;
  return 1;
}
extern "C" int64_t et_tr3_lf_finish_v1(int64_t caught) {
  const bool failed = hit;
  if (failed) {
    require(caught == 1 && observed, "failure was not caught as real condition");
  } else {
    require(caught == 0, "uninjected attempt unexpectedly raised");
    require(mode == Mode::object || mode == Mode::all_object ||
                mode == Mode::promotion ||
                mode == Mode::target,
            "non-matrix failpoint unexpectedly missed");
    if (mode == Mode::object) {
      require(tensor_attempts == 0 && ad_node_attempts == 0,
              "unreviewed tensor or AD constructor executed in object census");
      require(raw_attempts > 0 && closure_attempts > 0 &&
                  string_attempts > 0,
              "unsafe final81298 allocator census unexpectedly empty");
      std::printf(
          "TR3-LEASE-UNSAFE-ALLOCATOR-CENSUS raw=%llu header=%llu "
          "closure=%llu string=%llu tensor=%llu ad-node=%llu BLOCKED\n",
          static_cast<unsigned long long>(raw_attempts),
          static_cast<unsigned long long>(header_attempts),
          static_cast<unsigned long long>(closure_attempts),
          static_cast<unsigned long long>(string_attempts),
          static_cast<unsigned long long>(tensor_attempts),
          static_cast<unsigned long long>(ad_node_attempts));
    } else if (mode == Mode::all_object) {
      require(vector_attempts > 0 && cons_attempts > 0 &&
                  raw_attempts > 0 && closure_attempts > 0 &&
                  string_attempts > 0,
              "repaired-runtime allocator census unexpectedly empty");
      std::printf(
          "TR3-LEASE-ALL-ALLOCATOR-CENSUS vector=%llu cons=%llu raw=%llu "
          "header=%llu closure=%llu string=%llu tensor=%llu ad-node=%llu "
          "PASS\n",
          static_cast<unsigned long long>(vector_attempts),
          static_cast<unsigned long long>(cons_attempts),
          static_cast<unsigned long long>(raw_attempts),
          static_cast<unsigned long long>(header_attempts),
          static_cast<unsigned long long>(closure_attempts),
          static_cast<unsigned long long>(string_attempts),
          static_cast<unsigned long long>(tensor_attempts),
          static_cast<unsigned long long>(ad_node_attempts));
    }
  }
  mode = Mode::idle;
  all_object_armed = false;
  hit = observed = promotion_active = target_active = false;
  return failed ? 1 : 0;
}
extern "C" int64_t et_tr3_lf_snapshot_begin_v1() {
  require(mode == Mode::idle, "snapshot while armed");
  std::memset(parameters, 0, sizeof(parameters));
  std::memset(&optimizer, 0, sizeof(optimizer));
  have_snapshot = false;
  return 1;
}
extern "C" int64_t et_tr3_lf_snapshot_parameter_v1(
    int64_t index, void *parameter) {
  require(index >= 0 && index < static_cast<int64_t>(parameter_count),
          "parameter snapshot index");
  copy_parameter(&parameters[index],
                 static_cast<et_f32_parameter *>(parameter));
  return 1;
}
extern "C" int64_t et_tr3_lf_snapshot_optimizer_v1(void *candidate) {
  copy_optimizer(&optimizer, static_cast<et_o2_optimizer *>(candidate));
  have_snapshot = true;
  return 1;
}
extern "C" int64_t et_tr3_lf_verify_parameter_v1(
    int64_t index, void *parameter) {
  require(have_snapshot && index >= 0 &&
              index < static_cast<int64_t>(parameter_count),
          "parameter verification index");
  ParameterSnapshot now{};
  copy_parameter(&now, static_cast<et_f32_parameter *>(parameter));
  return std::memcmp(&now, &parameters[index], sizeof(now)) == 0 ? 1 : 0;
}
extern "C" int64_t et_tr3_lf_verify_optimizer_v1(void *candidate) {
  require(have_snapshot, "optimizer verification without snapshot");
  OptimizerSnapshot now{};
  copy_optimizer(&now, static_cast<et_o2_optimizer *>(candidate));
  return std::memcmp(&now, &optimizer, sizeof(now)) == 0 ? 1 : 0;
}
extern "C" int64_t et_tr3_lf_done_v1() {
  require(mode == Mode::idle && !all_object_armed &&
              !preparing_handler && !promotion_active &&
              !target_active && !poststage_window,
          "unfinished failpoint");
  require(object_failures + promotion_failures + target_failures +
              handler_failures <= 1,
          "isolated process observed multiple failures");
  return 1;
}
