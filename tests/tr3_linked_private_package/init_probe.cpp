#include <atomic>
#include <csetjmp>
#include <cstdio>

#include <eshkol/core/runtime.h>
#include "arena_memory.h"
#include "tr3_c_private_initializer_bridge.h"

int main() {
  arena_t *const arena = get_global_arena_shared();
  if (!arena || __repl_shared_arena.load() != nullptr) return 1;
  const size_t used_before = arena_get_used_memory(arena);
  eshkol_exception_handler_t *const prior = g_exception_handler_stack;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_READY_V1) return 2;
  const size_t used_after = arena_get_used_memory(arena);
  if (g_exception_handler_stack != prior ||
      __repl_shared_arena.load() != arena || used_after <= used_before) {
    return 3;
  }
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_READY_V1 ||
      arena_get_used_memory(arena) != used_after) return 4;
  std::printf("TR3 linked private initializer PASS: root_used_before=%zu root_used_after=%zu\n",
              used_before, used_after);
  return 0;
}
