#include <atomic>
#include <setjmp.h>

#include <eshkol/core/runtime.h>
#include "arena_memory.h"
#include "tr3_c_private_initializer_bridge.h"

#undef setjmp
extern "C" int setjmp(jmp_buf environment) __attribute__((returns_twice));

extern "C" void __eshkol_lib_init__(void *arena);
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *value);
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t *value);

namespace {
enum : unsigned { kUninitialized, kInitializing, kReady };
std::atomic<unsigned> readiness{kUninitialized};
}

extern "C" int et_tr3_c_private_ready_internal_v1(void) {
  return readiness.load(std::memory_order_acquire) == kReady;
}

extern "C" int et_tr3_c_private_initialize_v1(void) {
  unsigned expected = kUninitialized;
  if (!readiness.compare_exchange_strong(expected, kInitializing,
                                         std::memory_order_acq_rel)) {
    return expected == kReady ? ET_TR3_C_INIT_READY_V1
                              : ET_TR3_C_INIT_BUSY_V1;
  }

  if (eshkol_runtime_init() != 0) {
    readiness.store(kUninitialized, std::memory_order_release);
    return ET_TR3_C_INIT_RUNTIME_V1;
  }
  arena_t *const arena = get_global_arena_shared();
  if (arena == nullptr) {
    readiness.store(kUninitialized, std::memory_order_release);
    return ET_TR3_C_INIT_ARENA_V1;
  }
  arena_t *slot = nullptr;
  if (!__repl_shared_arena.compare_exchange_strong(slot, arena,
                                                    std::memory_order_acq_rel) &&
      slot != arena) {
    readiness.store(kUninitialized, std::memory_order_release);
    return ET_TR3_C_INIT_ARENA_V1;
  }

  eshkol_exception_handler_t *const prior = g_exception_handler_stack;
  jmp_buf handler;
  eshkol_push_exception_handler(&handler);
  if (g_exception_handler_stack == prior) {
    readiness.store(kUninitialized, std::memory_order_release);
    return ET_TR3_C_INIT_HANDLER_V1;
  }
  if (setjmp(handler) != 0) {
    eshkol_tagged_value_t raised;
    eshkol_get_raised_value(&raised);
    eshkol_parallel_scope_end();
    eshkol_pop_exception_handler();
    eshkol_set_raised_value(&raised);
    readiness.store(kUninitialized, std::memory_order_release);
    return ET_TR3_C_INIT_EXCEPTION_V1;
  }

  eshkol_parallel_scope_begin();
  __eshkol_lib_init__(arena);
  eshkol_parallel_scope_end();
  eshkol_pop_exception_handler();
  readiness.store(kReady, std::memory_order_release);
  return ET_TR3_C_INIT_READY_V1;
}
