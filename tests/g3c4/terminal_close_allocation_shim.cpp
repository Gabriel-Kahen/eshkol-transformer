// Preserve the accepted call-entry failure controls while adding a test-only
// fence from native terminal zero through the Eshkol m3-call return.
#define __wrap_arena_allocate_vector_with_header terminal_prior_vector
#define __wrap_arena_allocate_cons_with_header terminal_prior_cons
#define __wrap_arena_allocate_with_header terminal_prior_bytes
#define __wrap_malloc terminal_prior_malloc
#define __wrap_eshkol_push_exception_handler terminal_prior_handler
#include "call_entry_allocation_shim.cpp"
#undef __wrap_arena_allocate_vector_with_header
#undef __wrap_arena_allocate_cons_with_header
#undef __wrap_arena_allocate_with_header
#undef __wrap_malloc
#undef __wrap_eshkol_push_exception_handler

static bool terminal_tail_armed;

extern "C" int64_t et_g3c4_terminal_tail_arm_v1(void) {
  if (terminal_tail_armed) std::abort();
  terminal_tail_armed = true;
  return 0;
}

extern "C" int64_t et_g3c4_terminal_tail_disarm_v1(void) {
  if (!terminal_tail_armed) return -1;
  terminal_tail_armed = false;
  return 0;
}

extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  if (terminal_tail_armed) std::abort();
  return terminal_prior_vector(arena, capacity);
}

extern "C" void *__wrap_arena_allocate_cons_with_header(arena_t *arena) {
  if (terminal_tail_armed) std::abort();
  return terminal_prior_cons(arena);
}

extern "C" void *__wrap_arena_allocate_with_header(
    arena_t *arena, size_t bytes, uint8_t subtype, uint8_t flags) {
  if (terminal_tail_armed) std::abort();
  return terminal_prior_bytes(arena, bytes, subtype, flags);
}

extern "C" void *__wrap_malloc(size_t bytes) {
  if (terminal_tail_armed) std::abort();
  return terminal_prior_malloc(bytes);
}

extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  if (terminal_tail_armed) std::abort();
  terminal_prior_handler(buffer);
}
