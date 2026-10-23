// Development witness: adapt actual M3 handler wrapper mechanics, no runtime patch.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void *__real_malloc(size_t);
extern "C" void __real_eshkol_push_exception_handler(void *);
extern "C" void __real_eshkol_get_raised_value(eshkol_tagged_value_t *);
extern "C" void *__real_et_g3c4_private_training_lease_create_v1(void *);
extern "C" int64_t __real_et_g3c4_private_training_lease_prepare_release_v1(void *);
extern "C" int64_t __real_et_g3c4_private_training_lease_commit_release_v1(void *);
extern "C" int64_t et_g3c4_private_training_lease_check_v1(void *);
static eshkol_exception_t *condition5;
static eshkol_exception_handler_t *outer_handler, *fixture_parent, *constructor_handler;
static bool preparing, fresh, armed, failing, hit, tracking, dead_checked;
static int mode, pushes, in_push, reads, failed_mallocs;
static int create_attempts, create_success, prepare_success, commit_success, failure_attempts;
static void *owner_expected, *observed_lease;
static void check(bool good, const char *message) {
  if (!good) {
    std::fprintf(stderr, "FAIL CPU4 binding witness: %s mode=%d pushes=%d reads=%d hit=%d mallocfails=%d creates=%d\n",
                 message, mode, pushes, reads, hit, failed_mallocs, create_attempts);
    std::abort();
  }
}
extern "C" void *__wrap_malloc(size_t bytes) {
  if (bytes == sizeof(eshkol_exception_handler_t)) {
    if (preparing && in_push == -1) fresh = true;
    if (armed && mode != 3 && in_push == 1 && !failing) {
      check(g_exception_handler_stack == outer_handler, "failed first constructor push has previous active frame");
      failing = true;
      hit = true;
    }
    if (failing) {
      hit = true;
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
    constructor_handler = g_exception_handler_stack;
    check(mode == 3 && constructor_handler && constructor_handler->prev == outer_handler,
          "postpublication case installed genuine constructor handler");
  }
  in_push = 0;
}
extern "C" void __wrap_eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  __real_eshkol_get_raised_value(value);
  if (failing) {
    check(value->type == ESHKOL_VALUE_HEAP_PTR && !value->flags && !value->reserved &&
          reinterpret_cast<eshkol_exception_t *>(value->data.ptr_val) == condition5 &&
          g_current_exception == condition5, "exact calibrated static condition5 at every caught boundary");
    auto *expected = mode == 3 && reads == 0 ? outer_handler : fixture_parent;
    check(g_exception_handler_stack == expected, "real constructor/fixture guard stack crossing");
    ++reads;
    in_push = 0;
  }
}
extern "C" int64_t et_cpu4bw_calibrate() {
  eshkol_runtime_emergency_raise_v1(5);
  std::abort();
}
extern "C" int64_t et_cpu4bw_calibrated() {
  eshkol_tagged_value_t value{};
  __real_eshkol_get_raised_value(&value);
  check(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved, "canonical calibration tag");
  condition5 = reinterpret_cast<eshkol_exception_t *>(value.data.ptr_val);
  check(condition5 && condition5 == g_current_exception &&
        !std::strcmp(condition5->message, "object or exception-handler allocation failed"), "runtime condition5 identity");
  return 1;
}
extern "C" int64_t et_cpu4bw_prepare(int64_t selected, void *owner) {
  check(condition5 && selected >= 1 && selected <= 3 && owner && !preparing && !armed && !tracking,
        "independent prepared case");
  mode = static_cast<int>(selected);
  owner_expected = owner;
  observed_lease = nullptr;
  outer_handler = fixture_parent = constructor_handler = nullptr;
  preparing = tracking = true;
  fresh = armed = failing = hit = dead_checked = false;
  pushes = in_push = reads = failed_mallocs = 0;
  create_attempts = create_success = prepare_success = commit_success = failure_attempts = 0;
  return 1;
}
extern "C" int64_t et_cpu4bw_fresh() { return fresh; }
extern "C" int64_t et_cpu4bw_arm() {
  check(preparing && fresh && !armed, "actual handler malloc observed after occupying recycled frames");
  outer_handler = g_exception_handler_stack;
  check(outer_handler, "fixture catch active before constructor");
  fixture_parent = outer_handler->prev;
  preparing = false;
  armed = true;
  pushes = 0;
  return 1;
}
extern "C" int64_t et_cpu4bw_post() {
  check(armed && mode == 3 && pushes == 1 && !failing && constructor_handler &&
        g_exception_handler_stack == constructor_handler, "genuine P1 capture returned under actual constructor guard");
  // reserve129 must actually attempt sizeof(handler) malloc and raise condition5.
  // If enough recycled frames exist, Scheme reaches exit72 and witness FAILS.
  failing = true;
  return 1;
}
extern "C" int64_t et_cpu4bw_caught() {
  check(armed && hit && failing && pushes == 1 && reads == (mode == 3 ? 2 : 1) &&
        failed_mallocs == 1 && create_attempts == 0, "fixture caught genuine allocation failure before native lease");
  failure_attempts = create_attempts;
  armed = failing = false;
  return 1;
}
extern "C" int64_t et_cpu4bw_failure_attempts() { return failure_attempts; }
extern "C" void *__wrap_et_g3c4_private_training_lease_create_v1(void *owner) {
  if (tracking) {
    check(owner == owner_expected && !observed_lease, "actual create uses same canonical native owner");
    ++create_attempts;
  }
  void *lease = __real_et_g3c4_private_training_lease_create_v1(owner);
  if (tracking && lease) { ++create_success; observed_lease = lease; }
  return lease;
}
extern "C" int64_t __wrap_et_g3c4_private_training_lease_prepare_release_v1(void *lease) {
  if (tracking) check(lease == observed_lease && lease, "prepare exact observed lease");
  const auto status = __real_et_g3c4_private_training_lease_prepare_release_v1(lease);
  if (tracking && status == 0) ++prepare_success;
  return status;
}
extern "C" int64_t __wrap_et_g3c4_private_training_lease_commit_release_v1(void *lease) {
  if (tracking) check(lease == observed_lease && lease, "commit exact observed lease");
  const auto status = __real_et_g3c4_private_training_lease_commit_release_v1(lease);
  if (tracking && status == 0) ++commit_success;
  return status;
}
extern "C" int64_t et_cpu4bw_dead(void *lease) {
  check(tracking && lease == observed_lease && lease && prepare_success == 1 && commit_success == 1,
        "one real prepare/commit before dead check");
  check(et_g3c4_private_training_lease_check_v1(lease) != 0, "same released lease rejects check");
  dead_checked = true;
  return 1;
}
extern "C" int64_t et_cpu4bw_finish(int64_t selected, int64_t pending, int64_t blocked) {
  const bool baseline = mode == 1;
  check(tracking && selected == mode && !armed && !failing && hit &&
        pending == (baseline ? 1 : 0) && blocked == (baseline ? 1 : 0) &&
        create_attempts == (baseline ? 0 : 1) && create_success == (baseline ? 0 : 1) &&
        prepare_success == (baseline ? 0 : 1) && commit_success == (baseline ? 0 : 1) &&
        dead_checked == !baseline, "causal pending/retry and actual lease lifecycle");
  std::printf("CPU4-BINDING-WITNESS case=%d pushes=%d condition5-reads=%d failure-create-attempts=%d pending=%lld retry-blocked=%lld retry-create=%d prepare=%d commit=%d dead-check=%d\n",
    mode, pushes, reads, failure_attempts, static_cast<long long>(pending), static_cast<long long>(blocked),
    create_success, prepare_success, commit_success, dead_checked);
  tracking = false;
  return 1;
}
