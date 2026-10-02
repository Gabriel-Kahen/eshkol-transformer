/* Isolated I2 guard witness. Never linked into an installed package. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if ET_E3_I2_GUARD_FAIL_LENGTH != 1 && ET_E3_I2_GUARD_FAIL_LENGTH != 3
#error "I2 guard fault must select exactly result vector 1 or 3"
#endif

extern int64_t __real_et_i2_private_copy_builder_prepare_v1(void *);
extern int64_t __real_et_i2_private_copy_builder_abort_v1(void *);
extern int64_t __real_et_i2_private_copy_builder_commit_v1(void *);
extern void *__real_arena_allocate_vector_with_header(void *, size_t);
extern void __real_eshkol_get_raised_value(eshkol_tagged_value_t *);

static void *prepared_builder;
static void *raised_condition;
static int prepare_count, abort_count, commit_count, vector_count;
static int consumed, raised_count, finished;

static void require_witness(int condition, const char *message) {
  if (!condition) {
    static const char prefix[] = "E3-I2-GUARD-WRAP-FAIL: ";
    (void)write(2, prefix, sizeof(prefix) - 1u);
    (void)write(2, message, strlen(message));
    (void)write(2, "\n", 1u);
    abort();
  }
}

int64_t __wrap_et_i2_private_copy_builder_prepare_v1(void *builder) {
  const int64_t status = __real_et_i2_private_copy_builder_prepare_v1(builder);
  if (finished) return status;
  require_witness(status == 0 && !prepared_builder && builder,
                  "one real registered builder prepared");
  prepared_builder = builder;
  ++prepare_count;
  return status;
}

void *__wrap_arena_allocate_vector_with_header(void *arena, size_t length) {
  if (!finished && prepared_builder && !consumed) {
    require_witness(length == (vector_count == 0 ? 1u : 3u) &&
                    vector_count < 2, "exact envelope allocation order");
    ++vector_count;
    if (length == ET_E3_I2_GUARD_FAIL_LENGTH) {
      consumed = 1;
      return NULL;
    }
  }
  return __real_arena_allocate_vector_with_header(arena, length);
}

int64_t __wrap_et_i2_private_copy_builder_abort_v1(void *builder) {
  if (!finished && prepared_builder) {
    require_witness(builder == prepared_builder && consumed && !abort_count,
                    "guard aborts the same prepared builder once");
    ++abort_count;
  }
  const int64_t status = __real_et_i2_private_copy_builder_abort_v1(builder);
  if (!finished && prepared_builder)
    require_witness(status == 0, "real native builder abort succeeded");
  return status;
}

int64_t __wrap_et_i2_private_copy_builder_commit_v1(void *builder) {
  if (!finished && builder == prepared_builder) ++commit_count;
  return __real_et_i2_private_copy_builder_commit_v1(builder);
}

void __wrap_eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  __real_eshkol_get_raised_value(value);
  if (finished || !consumed) return;
  require_witness(value->type == ESHKOL_VALUE_HEAP_PTR &&
                  !value->flags && !value->reserved && value->data.ptr_val,
                  "actual emergency condition tag");
  eshkol_exception_t *exception = (eshkol_exception_t *)value->data.ptr_val;
  require_witness(exception->message &&
                  strcmp(exception->message,
                         "object or exception-handler allocation failed") == 0,
                  "actual emergency condition message");
  if (!raised_condition) raised_condition = exception;
  else require_witness(raised_condition == exception,
                       "I2 guard rethrows unchanged condition");
  ++raised_count;
  if (raised_count == 2) {
    require_witness(prepare_count == 1 && abort_count == 1 &&
                    commit_count == 0 && vector_count ==
                      (ET_E3_I2_GUARD_FAIL_LENGTH == 1 ? 1 : 2),
                    "one prepare, consumed cut, one abort, no commit");
    static const char marker[] = "E3-I2-GUARD-WRAP-PASS\n";
    require_witness(write(1, marker, sizeof(marker) - 1u) ==
                      (ssize_t)(sizeof(marker) - 1u), "witness marker write");
    finished = 1;
  }
}
