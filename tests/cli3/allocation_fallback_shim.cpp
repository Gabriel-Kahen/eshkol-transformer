// Test-only linker interception for the production CLI entry and aggregate.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

extern "C" void *__real_malloc(std::size_t);
extern "C" void *__real_arena_allocate_vector_with_header(void *, std::size_t);
extern "C" int64_t __real_et_e1b_public_cli3_dispatch_fallback_v1(void);
extern "C" void __real_eshkol_push_exception_handler(void *);
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *);

enum class injection_mode { invalid, handler, rollback };

static injection_mode selected_mode = injection_mode::invalid;
static bool initialized;
static bool blocked;
static bool in_push;
static int trigger_count;
static int fallback_count;
static int trigger_depth = -1;
static int push_count;

[[noreturn]] static void fail(const char *message) {
  static const char prefix[] = "CLI3 allocation fallback shim FAIL: ";
  (void)write(STDERR_FILENO, prefix, sizeof(prefix) - 1u);
  (void)write(STDERR_FILENO, message, std::strlen(message));
  (void)write(STDERR_FILENO, "\n", 1u);
  _Exit(98);
}

static injection_mode mode() {
  if (!initialized) {
    const char *value = std::getenv("CLI3_ALLOCATION_FALLBACK_MODE");
    if (value && !std::strcmp(value, "handler")) {
      selected_mode = injection_mode::handler;
    } else if (value && !std::strcmp(value, "rollback")) {
      selected_mode = injection_mode::rollback;
    }
    initialized = true;
  }
  return selected_mode;
}

static int handler_depth() {
  int depth = 0;
  for (eshkol_exception_handler_t *frame = g_exception_handler_stack; frame;
       frame = frame->prev) {
    ++depth;
    if (depth > 64) {
      fail("exception-handler stack is malformed");
    }
  }
  return depth;
}

static bool path_exists(const char *path) {
  return path && path[0] != '\0' && access(path, F_OK) == 0;
}

extern "C" void *__wrap_malloc(std::size_t bytes) {
  if (mode() == injection_mode::handler &&
      in_push && bytes == sizeof(eshkol_exception_handler_t)) {
    const int depth = handler_depth();
    if (depth == 1) {
      if (!blocked) {
        if (push_count != 2) {
          fail("private-dispatch guard was not the second real push");
        }
        blocked = true;
        ++trigger_count;
        trigger_depth = depth;
      }
      return nullptr;
    }
  }
  return __real_malloc(bytes);
}

extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  if (mode() == injection_mode::handler) {
    ++push_count;
    in_push = true;
  }
  __real_eshkol_push_exception_handler(buffer);
  in_push = false;
}

extern "C" void *__wrap_arena_allocate_vector_with_header(void *arena,
                                                             std::size_t count) {
  if (mode() == injection_mode::rollback) {
    const char *lock = std::getenv("CLI3_ALLOCATION_FALLBACK_LOCK");
    const char *shard = std::getenv("CLI3_ALLOCATION_FALLBACK_SHARD0");
    if (blocked || (path_exists(lock) && path_exists(shard))) {
      if (!blocked) {
        blocked = true;
        ++trigger_count;
        trigger_depth = handler_depth();
      }
      return nullptr;
    }
  }
  return __real_arena_allocate_vector_with_header(arena, count);
}

static void write_marker() {
  const char *path = std::getenv("CLI3_ALLOCATION_FALLBACK_MARKER");
  if (!path || path[0] == '\0') {
    fail("marker path is missing");
  }
  const char *record = nullptr;
  std::size_t length = 0;
  if (selected_mode == injection_mode::handler && trigger_depth == 1 &&
      push_count == 2 && in_push) {
    static const char text[] =
        "mode=handler trigger=1 fallback=1 depth=1 push=2 condition=5\n";
    record = text;
    length = sizeof(text) - 1u;
  } else if (selected_mode == injection_mode::rollback && trigger_depth == 3) {
    static const char text[] =
        "mode=rollback trigger=1 fallback=1 depth=3 condition=5\n";
    record = text;
    length = sizeof(text) - 1u;
  } else {
    fail("unexpected failure mode or handler depth");
  }
  const int descriptor = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    fail("cannot create marker");
  }
  const ssize_t amount = write(descriptor, record, length);
  const int saved_errno = errno;
  if (close(descriptor) != 0 || amount < 0 ||
      static_cast<std::size_t>(amount) != length) {
    errno = saved_errno;
    fail("cannot complete marker");
  }
}

extern "C" int64_t __wrap_et_e1b_public_cli3_dispatch_fallback_v1(void) {
  ++fallback_count;
  if (!blocked || trigger_count != 1 || fallback_count != 1) {
    fail("fallback did not follow exactly one injected allocation failure");
  }
  eshkol_tagged_value_t raised{};
  eshkol_get_raised_value(&raised);
  if (raised.type != ESHKOL_VALUE_HEAP_PTR || raised.flags != 0u ||
      raised.reserved != 0u ||
      reinterpret_cast<eshkol_exception_t *>(raised.data.ptr_val) !=
          g_current_exception ||
      !g_current_exception ||
      std::strcmp(g_current_exception->message,
                  "object or exception-handler allocation failed")) {
    fail("outer fallback did not receive canonical runtime condition 5");
  }
  const int64_t status =
      __real_et_e1b_public_cli3_dispatch_fallback_v1();
  if (status != 70) {
    fail("production static fallback did not retain internal status 70");
  }
  write_marker();
  return status;
}
