// Test-only linker interception: no replacement reserve/push implementation.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <eshkol/eshkol.h>
#pragma clang diagnostic pop
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10
#define ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10 0
#endif
#if ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10 < 0 || \
    ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10 > 1
#error "ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10 must be 0 or 1"
#endif

extern "C" void *__real_malloc(size_t);
extern "C" void __real_eshkol_push_exception_handler(void *);
extern "C" void __real_eshkol_get_raised_value(eshkol_tagged_value_t *);
extern "C" int64_t __real_eshkol_runtime_reserve_exception_handlers_v1(int64_t);
extern "C" int64_t __real_et_p1_private_state_bind_v1(void *, void *, const void *);
extern "C" int64_t __real_et_i2_private_owned_release_v1(void *);
extern "C" int64_t __real_et_o2_private_optimizer_state_release_v1(void *);
extern "C" int64_t __real_et_p1_private_state_release_begin_v1(void *, void *);

static eshkol_exception_t *condition5;
static eshkol_exception_handler_t *outer;
static bool preparing, fresh, armed, in_push, in_reserve, blocked, observed;
static bool defect, bind_defect;
static int64_t site, mode, reserves, requested_reserve, actual_reserve;
static int64_t attempts, hits, pushes, active_on_attempt;
static int64_t binds, p1, i2, o2, events[3], completed;
static void require(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr,
                 "C2 handler FAIL: %s site=%lld mode=%lld reserve=%lld/%lld calls=%lld malloc=%lld hit=%lld push=%lld active=%lld bind=%lld provider=%lld/%lld/%lld acquire=%lld/%lld/%lld\n",
                 message, (long long)site, (long long)mode, (long long)requested_reserve,
                 (long long)actual_reserve, (long long)reserves,
                 (long long)attempts, (long long)hits, (long long)pushes,
                 (long long)active_on_attempt,
                 (long long)binds,
                 (long long)p1, (long long)i2, (long long)o2,
                 (long long)events[0], (long long)events[1], (long long)events[2]);
    std::abort();
  }
}
extern "C" void *__wrap_malloc(size_t bytes) {
  if (bytes == sizeof(eshkol_exception_handler_t) && (in_push || in_reserve)) {
    if (preparing && in_push) fresh = true;
    if (armed) {
      ++attempts;
      active_on_attempt = 0;
      for (eshkol_exception_handler_t *frame = g_exception_handler_stack;
           frame && frame != outer; frame = frame->prev) {
        ++active_on_attempt;
      }
#if ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10
      if (site == 3 && mode == 3 && blocked) {
        ++hits;
        require(requested_reserve == 11 && actual_reserve == 10 && pushes == 19 &&
                    active_on_attempt == 10 && binds == 1 && !p1 && i2 == 1 && !o2 &&
                    events[0] == 1 && !events[1] && !events[2],
                "reserve-10 diagnostic reached the exact LOAD dual-fault push");
        std::fprintf(
            stderr,
            "C2-HANDLER-DIAGNOSTIC case=3 mode=3 reserve=11/10 push=19 active=10 bind=1 provider=0/1/0 acquire=1/0/0 EXPECTED\n");
        std::fflush(stderr);
        std::_Exit(97);
      }
#endif
      if ((mode == 0 && in_reserve) || blocked) {
        ++hits;
        return nullptr;
      }
    }
  }
  return __real_malloc(bytes);
}
extern "C" void __wrap_eshkol_push_exception_handler(void *buffer) {
  in_push = true;
  if (armed) ++pushes;
  __real_eshkol_push_exception_handler(buffer);
  in_push = false;
}
extern "C" int64_t __wrap_eshkol_runtime_reserve_exception_handlers_v1(int64_t count) {
  if (armed) {
    if (reserves == 0) {
      require(count == (site == 1 ? 5 : site == 2 ? 6 : 11), "exact initial reserve");
      require(g_exception_handler_stack == outer, "reserve precedes operation handlers");
      requested_reserve = count;
      actual_reserve = count -
          (ET_C2_HANDLER_DIAGNOSTIC_LOAD_DUAL_RESERVE10 && site == 3 && mode == 3 ? 1 : 0);
    }
    ++reserves;
  }
  in_reserve = true;
  int64_t status = __real_eshkol_runtime_reserve_exception_handlers_v1(
      armed && reserves == 1 ? actual_reserve : count);
  in_reserve = false;
  if (armed && reserves == 1) {
    require(mode != 0, "reserve failure injection must hit real malloc");
    require(status == 0, "reserve success");
    // Subsequent attempts remain blocked until the operation ends, including
    // nested reserves during LOAD rollback. Do not disarm on the first catch.
    attempts = hits = 0;
    blocked = true;
  }
  return status;
}
extern "C" void __wrap_eshkol_get_raised_value(eshkol_tagged_value_t *value) {
  __real_eshkol_get_raised_value(value);
  in_push = in_reserve = false; // nonlocal transfer bypassed wrapper epilogues
  if (armed && mode == 0 && hits) {
    require(value->type == ESHKOL_VALUE_HEAP_PTR && !value->flags && !value->reserved &&
            reinterpret_cast<eshkol_exception_t *>(value->data.ptr_val) == condition5 &&
            g_current_exception == condition5, "exact runtime condition5");
    observed = true;
  }
}
extern "C" int64_t __wrap_et_p1_private_state_release_begin_v1(void *context, void *state) {
  if (armed) ++p1;
  return __real_et_p1_private_state_release_begin_v1(context, state);
}
extern "C" int64_t __wrap_et_p1_private_state_bind_v1(
    void *context, void *state, const void *provider) {
  if (armed) ++binds;
  int64_t status = __real_et_p1_private_state_bind_v1(context, state, provider);
  if (armed && bind_defect) {
    require(site == 3 && mode == 3 && status == 0 && blocked,
            "real P1 bind transfers before adoption defect");
    bind_defect = false;
    // The real bind consumed native entry authority. A nonzero status makes
    // P1 adoption fail after its caller ledger has already transferred.
    return 9;
  }
  return status;
}
extern "C" int64_t __wrap_et_i2_private_owned_release_v1(void *value) {
  if (armed) ++i2;
  int64_t status = __real_et_i2_private_owned_release_v1(value);
  if (armed && defect) {
    require(status == 0 && blocked, "provider consumes real owner before defect");
    defect = false;
    // The bridge cleared its error on success. Nonzero status therefore takes
    // i2-native-fail's INTERNAL branch, after actual native owner consumption.
    return 9;
  }
  return status;
}
extern "C" int64_t __wrap_et_o2_private_optimizer_state_release_v1(void *value) {
  if (armed) ++o2;
  return __real_et_o2_private_optimizer_state_release_v1(value);
}
extern "C" int64_t et_c2h_calibrate() {
  eshkol_runtime_emergency_raise_v1(5);
  std::abort();
}
extern "C" int64_t et_c2h_calibrated() {
  eshkol_tagged_value_t value{};
  __real_eshkol_get_raised_value(&value);
  require(value.type == ESHKOL_VALUE_HEAP_PTR && !value.flags && !value.reserved,
          "calibration tag");
  condition5 = reinterpret_cast<eshkol_exception_t *>(value.data.ptr_val);
  require(condition5 == g_current_exception &&
          !std::strcmp(condition5->message, "object or exception-handler allocation failed"),
          "calibration identity");
  return 1;
}
extern "C" int64_t et_c2h_prepare(int64_t selected, int64_t behavior) {
  require(!armed && selected >= 1 && selected <= 3 && behavior >= 0 && behavior <= 3 &&
              (behavior != 3 || selected == 3),
          "valid independent case");
  site = selected; mode = behavior;
  reserves = requested_reserve = actual_reserve = 0;
  attempts = hits = pushes = active_on_attempt = binds = p1 = i2 = o2 = 0;
  for (auto &event : events) event = 0;
  observed = blocked = false;
  defect = mode == 2 || mode == 3;
  bind_defect = mode == 3;
  preparing = true; fresh = false;
  return 1;
}
extern "C" int64_t et_c2h_fresh() { return fresh; }
extern "C" int64_t et_c2h_arm() {
  require(condition5 && preparing && fresh && !armed, "free list exhausted by live guards");
  outer = g_exception_handler_stack;
  require(outer, "outer catch installed");
  preparing = false; armed = true;
  return 1;
}
extern "C" int64_t et_c2h_event(int64_t index) {
  require(index >= 0 && index < 3, "event index");
  if (armed) ++events[index];
  return 1;
}
extern "C" int64_t et_c2h_finish(int64_t caught) {
  require(armed && reserves > 0, "operation reached actual reserve");
  if (mode == 0) {
    require(caught && observed && attempts == 1 && hits == 1 && !pushes && !p1 && !i2 && !o2 &&
            !events[0] && !events[1] && !events[2], "failed reserve has zero ownership effects");
  } else {
    require(blocked && attempts == 0 && hits == 0 && pushes > 0,
            "cleanup uses reserved handlers with malloc persistently disabled");
    require(caught == (mode == 2 || mode == 3),
            "expected success or consumed-provider defect");
    if (mode == 3) {
      require(site == 3 && binds == 1 && !p1 && i2 == 1 && !o2,
              "dual fault binds once and cleans only exact I2 ownership");
    } else if (site < 3 || mode == 2) {
      require(p1 == 1 && i2 == 1 && o2 == (site == 1 ? 0 : 1),
              "exactly one P1/I2 cleanup and the expected O2 cleanup");
    } else {
      require(!p1 && !i2 && !o2, "successful LOAD retains its owner");
    }
    if (mode == 3) {
      require(events[0] == 1 && !events[1] && !events[2],
              "dual fault stops after the exact decode acquisition");
    } else {
      require(events[0] == (site == 3) && events[1] == (site == 3) &&
                  events[2] == (site == 3),
              "exact acquisition/compose prefix");
    }
    require((mode != 2 && mode != 3) || !defect,
            "real provider defect was exercised");
    require(mode != 3 || !bind_defect,
            "post-transfer bind defect was exercised");
  }
  armed = blocked = false;
  in_push = in_reserve = false;
  ++completed;
  std::printf("C2-HANDLER case=%lld mode=%lld hit=%lld bind=%lld provider=%lld/%lld/%lld acquire=%lld/%lld/%lld PASS\n",
              (long long)site, (long long)mode, (long long)hits,
              (long long)binds,
              (long long)p1, (long long)i2, (long long)o2,
              (long long)events[0], (long long)events[1], (long long)events[2]);
  return 1;
}
extern "C" int64_t et_c2h_done() {
  require(completed == 10 && !armed && !preparing, "all ten cases completed");
  return 1;
}
