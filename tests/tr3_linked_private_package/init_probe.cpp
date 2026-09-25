#include <atomic>
#include <csetjmp>
#include <cstdio>

#include <eshkol/core/runtime.h>
#include "arena_memory.h"

extern "C" void __eshkol_lib_init__(void *arena);
extern "C" eshkol_exception_t *eshkol_get_current_exception(void);
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *value);

int main() {
  if (eshkol_runtime_init() != 0) return 1;
  arena_t *const arena = get_global_arena_shared();
  if (!arena) return 2;
  // Match AOT main's runtime slot before invoking the package initializer.
  __repl_shared_arena.store(arena);
  if (__repl_shared_arena.load() != arena) return 3;
  const size_t used_before = arena_get_used_memory(arena);
  eshkol_exception_handler_t *const prior = g_exception_handler_stack;
  jmp_buf handler;
  eshkol_push_exception_handler(&handler);
  if (g_exception_handler_stack == prior) return 4;
  if (setjmp(handler) != 0) {
    eshkol_tagged_value_t raised;
    eshkol_get_raised_value(&raised);
    std::fprintf(stderr, "init raised type=%u exc=%p\n", raised.type,
                 static_cast<void *>(eshkol_get_current_exception()));
    eshkol_parallel_scope_end();
    eshkol_pop_exception_handler();
    return 5;
  }
  eshkol_parallel_scope_begin();
  __eshkol_lib_init__(arena);
  eshkol_parallel_scope_end();
  eshkol_pop_exception_handler();
  const size_t used_after = arena_get_used_memory(arena);
  if (g_exception_handler_stack != prior ||
      __repl_shared_arena.load() != arena || used_after <= used_before) {
    return 6;
  }
  std::printf("TR3 linked private initializer PASS: root_used_before=%zu root_used_after=%zu\n",
              used_before, used_after);
  eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);
  return 0;
}
