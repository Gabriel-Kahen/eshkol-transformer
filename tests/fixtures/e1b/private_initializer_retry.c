#include <setjmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#include "e1b_error_consumer_bridge.h"

static eshkol_exception_handler_t handler_frames[4];
static int handler_count;
static int fail_next_handler_push;
static int parallel_depth;
static int parallel_begin_count;
static int parallel_end_count;
static int initializer_calls;
static eshkol_tagged_value_t raised_value;
static eshkol_exception_t init_exception;
eshkol_exception_handler_t *g_exception_handler_stack;

void eshkol_push_exception_handler(void *buffer) {
  eshkol_exception_handler_t *frame;

  if (fail_next_handler_push) {
    fail_next_handler_push = 0;
    return;
  }
  if (handler_count == 4) {
    abort();
  }
  frame = &handler_frames[handler_count++];
  frame->jmp_buf_ptr = buffer;
  frame->prev = g_exception_handler_stack;
  g_exception_handler_stack = frame;
}

void eshkol_pop_exception_handler(void) {
  if (handler_count == 0) {
    abort();
  }
  g_exception_handler_stack = g_exception_handler_stack->prev;
  --handler_count;
}

void eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  *value = raised_value;
}

void eshkol_set_raised_value(const eshkol_tagged_value_t *value) {
  raised_value = *value;
}

eshkol_exception_t *eshkol_get_current_exception(void) {
  return &init_exception;
}

void eshkol_raise(eshkol_exception_t *exception) {
  if (exception != &init_exception || handler_count == 0) {
    abort();
  }
  longjmp(*(jmp_buf *)g_exception_handler_stack->jmp_buf_ptr, 1);
}

void eshkol_parallel_scope_begin(void) {
  ++parallel_depth;
  ++parallel_begin_count;
}

void eshkol_parallel_scope_end(void) {
  if (parallel_depth == 0) {
    abort();
  }
  --parallel_depth;
  ++parallel_end_count;
}

void *get_global_arena_shared(void) {
  return &init_exception;
}

void __eshkol_lib_init__(void *arena) {
  ++initializer_calls;
  if (arena != &init_exception) {
    abort();
  }
  if (initializer_calls == 1) {
    eshkol_tagged_value_t failure = {0};
    failure.type = ESHKOL_VALUE_INT64;
    failure.data.int_val = INT64_C(0x13579bdf);
    eshkol_set_raised_value(&failure);
    eshkol_raise(&init_exception);
  }
}

#define PRIVATE_ACCESSOR_STUB(name)                                        \
  eshkol_tagged_value_t name(eshkol_tagged_value_t value) { return value; }

PRIVATE_ACCESSOR_STUB(et_e1b_private_error_predicate_cabi_v1)
PRIVATE_ACCESSOR_STUB(et_e1b_private_error_category_cabi_v1)
PRIVATE_ACCESSOR_STUB(et_e1b_private_error_operation_cabi_v1)
PRIVATE_ACCESSOR_STUB(et_e1b_private_error_message_cabi_v1)
PRIVATE_ACCESSOR_STUB(et_e1b_private_error_details_cabi_v1)
PRIVATE_ACCESSOR_STUB(et_e1b_private_error_cause_cabi_v1)

_Noreturn eshkol_tagged_value_t et_e1b_private_raise_cabi_v1(
    eshkol_tagged_value_t category, eshkol_tagged_value_t operation,
    eshkol_tagged_value_t message, eshkol_tagged_value_t details,
    eshkol_tagged_value_t cause) {
  (void)category;
  (void)operation;
  (void)message;
  (void)details;
  (void)cause;
  abort();
}

int main(void) {
  jmp_buf outer_handler;
  pid_t child;
  int child_status;

  child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
    fail_next_handler_push = 1;
    et_e1b_ensure_private_initialized_v1();
    _exit(0);
  }
  if (waitpid(child, &child_status, 0) != child ||
      !WIFSIGNALED(child_status)) {
    return 1;
  }

  eshkol_push_exception_handler(&outer_handler);
  if (setjmp(outer_handler) == 0) {
    et_e1b_ensure_private_initialized_v1();
    return 1;
  }

  if (parallel_depth != 0 || parallel_begin_count != 1 ||
      parallel_end_count != 1 || handler_count != 1 ||
      raised_value.type != ESHKOL_VALUE_INT64 ||
      raised_value.data.int_val != INT64_C(0x13579bdf)) {
    return 2;
  }
  eshkol_pop_exception_handler();

  et_e1b_ensure_private_initialized_v1();
  if (parallel_depth != 0 || parallel_begin_count != 2 ||
      parallel_end_count != 2 || initializer_calls != 2 ||
      handler_count != 0) {
    return 3;
  }

  et_e1b_ensure_private_initialized_v1();
  if (parallel_begin_count != 2 || parallel_end_count != 2 ||
      initializer_calls != 2 || handler_count != 0) {
    return 4;
  }
  return 0;
}
