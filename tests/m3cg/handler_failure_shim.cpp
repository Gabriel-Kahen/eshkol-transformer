// The frozen C runtime header uses C99 flexible arrays and anonymous unions.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void *__real_malloc(size_t);
extern "C" void __real_eshkol_push_exception_handler(void *);
extern "C" void __real_eshkol_get_raised_value(eshkol_tagged_value_t *);
static eshkol_exception_t *condition5;
static eshkol_exception_handler_t *outer_handler, *fixture_parent;
static eshkol_exception_handler_t *boundary_handler, *cleanup_handler;
static bool preparing, fresh, armed, failing, hit;
static int mode, pushes, in_push, body, catches, reads, failed_mallocs;
static void check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr,
                 "FAIL M3 handler: %s (mode=%d pushes=%d hit=%d reads=%d "
                 "failed-mallocs=%d body=%d catches=%d)\n",
                 message, mode, pushes, hit, reads, failed_mallocs, body, catches);
    std::abort();
  }
}
extern "C" void *__wrap_malloc(size_t bytes) {
  if (bytes == sizeof(eshkol_exception_handler_t)) {
    if (preparing && in_push == -1) fresh = true;
    if (armed && mode == 1 && in_push == 2 && !failing) {
      check(pushes == 2 && boundary_handler &&
            g_exception_handler_stack == boundary_handler &&
            boundary_handler->prev == outer_handler, "cleanup follows established boundary");
      failing = true;
      hit = true;
    }
    if (failing) {
      ++failed_mallocs;
      return nullptr;
    }
  }
  return __real_malloc(bytes);
}
extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  if (armed) in_push = ++pushes;
  else if (preparing) in_push = -1;
  __real_eshkol_push_exception_handler(buffer);
  if (armed && pushes == 1) {
    boundary_handler = g_exception_handler_stack;
    check(boundary_handler && boundary_handler != outer_handler &&
          boundary_handler->prev == outer_handler, "production boundary installed above outer catch");
  } else if (armed && pushes == 2 && mode == 2) {
    cleanup_handler = g_exception_handler_stack;
    check(cleanup_handler && cleanup_handler != boundary_handler &&
          cleanup_handler->prev == boundary_handler,
          "cleanup installed above production boundary");
  }
  in_push = 0;
}
extern "C" void __wrap_eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  __real_eshkol_get_raised_value(value);
  if (failing) {
    check(value->type == ESHKOL_VALUE_HEAP_PTR && !value->flags && !value->reserved &&
          reinterpret_cast<eshkol_exception_t *>(value->data.ptr_val) == condition5 &&
          g_current_exception == condition5, "every boundary catches exact static condition5");
    eshkol_exception_handler_t *expected = nullptr;
    if (mode == 1) expected = reads == 0 ? outer_handler : fixture_parent;
    else if (reads == 0) expected = boundary_handler;
    else if (reads == 1) expected = outer_handler;
    else expected = fixture_parent;
    check(g_exception_handler_stack == expected, "condition5 crosses the expected guard boundary");
    ++reads;
    in_push = 0;
  }
}
extern "C" int64_t et_m3cg_handler_calibrate() {
  eshkol_runtime_emergency_raise_v1(5);
  std::abort();
}
extern "C" int64_t et_m3cg_handler_calibrated() {
  eshkol_tagged_value_t value{};
  __real_eshkol_get_raised_value(&value);
  check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
        "calibration has canonical tag");
  condition5 = reinterpret_cast<eshkol_exception_t *>(value.data.ptr_val);
  check(condition5 && condition5 == g_current_exception &&
        !std::strcmp(condition5->message, "object or exception-handler allocation failed"),
        "condition5 calibration identity");
  return 1;
}
extern "C" int64_t et_m3cg_handler_prepare(int64_t selected_mode) {
  check(condition5 && (selected_mode == 1 || selected_mode == 2) && !preparing && !armed,
        "valid independent experiment");
  mode = static_cast<int>(selected_mode);
  preparing = true;
  fresh = false;
  failing = false;
  hit = false;
  pushes = 0;
  in_push = 0;
  body = 0;
  reads = 0;
  failed_mallocs = 0;
  return 1;
}
extern "C" int64_t et_m3cg_handler_fresh() { return fresh; }
extern "C" int64_t et_m3cg_handler_arm() {
  check(preparing && fresh && !armed, "recycled handlers occupied");
  outer_handler = g_exception_handler_stack;
  check(outer_handler, "outer Scheme catch is installed");
  fixture_parent = outer_handler->prev;
  preparing = false;
  armed = true;
  pushes = 0;
  return 1;
}
extern "C" int64_t et_m3cg_handler_body() {
  ++body;
  if (mode == 2) {
    check(armed && pushes == 2 && cleanup_handler && !failing,
          "body failure begins below both production guards");
    failing = true;
    hit = true;
    return 2;
  }
  return 1;
}
extern "C" int64_t et_m3cg_handler_caught(int64_t busy) {
  const int expected_reads = mode == 1 ? 2 : 3;
  const int expected_body = mode == 1 ? 0 : 1;
  check(hit && failing && armed && pushes == 2 && reads == expected_reads &&
        failed_mallocs == 1 && body == expected_body && !busy,
        "outer catch sees exact condition5, idle guard and persistent allocation failure");
  failing = false;
  armed = false;
  ++catches;
  return 1;
}
extern "C" int64_t et_m3cg_handler_finish(int64_t selected_mode, int64_t busy) {
  check(selected_mode == mode && catches == mode && hit && !failing && !armed && !busy,
        "retry ends idle");
  if (mode == 1)
    std::puts("M3CG handler: case=prepublish pushes=2 condition5-reads=2 body=0 idle=1 retry=1");
  else
    std::puts("M3CG handler: case=body pushes=2 condition5-reads=3 body=1 idle=1 retry=1");
  return 1;
}
