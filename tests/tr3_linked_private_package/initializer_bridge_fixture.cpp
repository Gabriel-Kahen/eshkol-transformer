#include <atomic>
#include <csetjmp>
#include <cstdint>

#include "arena_memory.h"
#include "tr3_c_private_initializer_bridge.h"

extern "C" {
eshkol_exception_handler_t *g_exception_handler_stack = nullptr;
}
extern "C" std::atomic<arena_t *> __repl_shared_arena{nullptr};

static arena_t root{};
static arena_t foreign{};
static eshkol_exception_handler_t frame{};
static eshkol_tagged_value_t raised{};
static bool runtime_fails;
static bool arena_missing;
static bool handler_fails;
static bool initializer_raises;
static bool initializer_reenters;
static int initializer_calls;
static int parallel_depth;
static int reentrant_status;

extern "C" int eshkol_runtime_init(void) { return runtime_fails ? -1 : 0; }
extern "C" arena_t *get_global_arena_shared(void) {
  return arena_missing ? nullptr : &root;
}
extern "C" void eshkol_push_exception_handler(void *buffer) {
  if (handler_fails) return;
  frame.jmp_buf_ptr = buffer;
  frame.prev = g_exception_handler_stack;
  g_exception_handler_stack = &frame;
}
extern "C" void eshkol_pop_exception_handler(void) {
  g_exception_handler_stack = g_exception_handler_stack->prev;
}
extern "C" void eshkol_parallel_scope_begin(void) { ++parallel_depth; }
extern "C" void eshkol_parallel_scope_end(void) { --parallel_depth; }
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  *value = raised;
}
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t *value) {
  raised = *value;
}
extern "C" void __eshkol_lib_init__(void *arena) {
  if (arena != &root || parallel_depth != 1 ||
      g_exception_handler_stack != &frame) __builtin_trap();
  ++initializer_calls;
  if (initializer_reenters) {
    initializer_reenters = false;
    reentrant_status = et_tr3_c_private_initialize_v1();
  }
  if (initializer_raises) {
    initializer_raises = false;
    raised.type = ESHKOL_VALUE_INT64;
    raised.data.int_val = INT64_C(0x13579bdf);
    longjmp(*static_cast<jmp_buf *>(frame.jmp_buf_ptr), 1);
  }
}

int main() {
  runtime_fails = true;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_RUNTIME_V1) return 1;
  runtime_fails = false;
  arena_missing = true;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_ARENA_V1) return 2;
  arena_missing = false;
  __repl_shared_arena.store(&foreign);
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_ARENA_V1 ||
      __repl_shared_arena.load() != &foreign) return 3;
  __repl_shared_arena.store(nullptr);
  handler_fails = true;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_HANDLER_V1) return 4;
  handler_fails = false;
  initializer_raises = true;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_EXCEPTION_V1 ||
      g_exception_handler_stack != nullptr || parallel_depth != 0 ||
      raised.type != ESHKOL_VALUE_INT64 ||
      raised.data.int_val != INT64_C(0x13579bdf) ||
      __repl_shared_arena.load() != &root || initializer_calls != 1) return 5;
  initializer_reenters = true;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_READY_V1 ||
      reentrant_status != ET_TR3_C_INIT_BUSY_V1 ||
      initializer_calls != 2 || g_exception_handler_stack != nullptr ||
      parallel_depth != 0) return 6;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_READY_V1 ||
      initializer_calls != 2) return 7;
  return 0;
}
