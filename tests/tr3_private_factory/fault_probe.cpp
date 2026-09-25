// Linked, test-only faults against the pinned runtime and native producers.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "arena_memory.h"
#pragma clang diagnostic pop

#define ET_D2_NATIVE_TESTING 1
#include "d2_native.h"
#define ET_O2_TESTING 1
#include "o2_optimizer_internal.h"
#include "tr3_c_private_factory_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <eshkol/eshkol.h>

extern "C" void et_m3t_test_fail_alloc_after(size_t);
extern "C" int64_t __real_et_tr3_c_factory_stage_v1(int64_t);
extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" int64_t __real_et_d2_dataset_open_v1(const void *, int64_t);
extern "C" int64_t __real_et_d2_dataset_close_v1(const void *);
extern "C" eshkol_tagged_value_t factory_retained
    __asm__("tr3-c-factory-retained");

namespace {
enum class Mode {
  idle, t2, m3t, o2, lease, cleanup, retention, lease_busy, close_busy
};
Mode mode = Mode::idle;
int64_t stage = 0;
bool denied = false;
unsigned opens = 0;
unsigned closes = 0;
unsigned refused_closes = 0;
int64_t peak_live = 0;
const void *dataset_owner = nullptr;
int64_t active_generation = 0;
int64_t active_borrow = 0;

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "TR3 factory fault probe FAIL: %s stage=%lld "
               "denied=%d opens=%u closes=%u refused=%u live=%lld peak=%lld\n",
               message, static_cast<long long>(stage), denied, opens, closes,
               refused_closes,
               static_cast<long long>(et_d2_dataset_test_live_count_v1()),
               static_cast<long long>(peak_live));
  std::abort();
}
void check(bool condition, const char *message) {
  if (!condition) fail(message);
}
template <class Call> void *bounded_denial(arena_t *arena, Call call) {
  check(arena && arena->current_block, "arena exists");
  arena_block_t *block = arena->current_block;
  const bool old_bounded = arena->bounded;
  const size_t old_used = block->used;
  arena->bounded = true;
  block->used = block->size;
  void *answer = call();
  arena->bounded = old_bounded;
  block->used = old_used;
  check(answer == nullptr, "real bounded vector allocation denied");
  return answer;
}
bool digest(const char *directory, uint8_t out[32]) {
  char path[4096];
  const int length = std::snprintf(path, sizeof(path), "%s/manifest.etm", directory);
  if (length < 0 || length >= static_cast<int>(sizeof(path))) return false;
  FILE *file = std::fopen(path, "rb");
  if (!file) return false;
  bool ok = std::fseek(file, -32, SEEK_END) == 0 &&
            std::fread(out, 1, 32, file) == 32;
  if (std::fclose(file) != 0) ok = false;
  return ok;
}
size_t optimizer_count() {
  et_o2_test_live_counts_v1 counts{};
  counts.struct_size = sizeof(counts);
  et_o2_test_live_counts_snapshot_v1(&counts);
  check(counts.builders == 0, "no staged O2 builder");
  return counts.optimizers;
}
void stage_live_borrow() {
  check(dataset_owner != nullptr && active_generation == 0,
        "exact opened D2 owner available");
  active_generation = et_d2_batch_create_v1(dataset_owner, 1, 1, 17);
  check(active_generation > 0, "native D2 batch created");
  struct alignas(8) Bytevector {
    int64_t length;
    uint8_t data[8];
  } token{8, {1, 0, 0, 0, 0, 0, 0, 0}};
  check(et_d2_batch_write_pair_i64le_span_v1(
            dataset_owner, active_generation, 0, &token, 8, &token, 8, 1) == 0,
        "native D2 batch filled");
  check(et_d2_batch_seal_v1(dataset_owner, active_generation) == 0,
        "native D2 batch sealed");
  active_borrow = et_d2_batch_borrow_begin_v1(dataset_owner,
                                                active_generation);
  check(active_borrow > 0 && et_d2_batch_test_borrow_count_v1() == 1,
        "real D2 active borrow staged");
}
void stage_unborrowed_batch() {
  check(dataset_owner != nullptr && active_generation == 0,
        "exact opened D2 owner available for busy preflight");
  active_generation = et_d2_batch_create_v1(dataset_owner, 1, 1, 17);
  check(active_generation > 0 &&
            et_d2_batch_test_live_count_v1() == 1,
        "real unborrowed D2 batch staged");
}
void retained_snapshot(eshkol_tagged_value_t out[6]) {
  check(ESHKOL_IS_VECTOR_COMPAT(factory_retained),
        "source process root is a vector");
  const void *const data = reinterpret_cast<const void *>(
      static_cast<uintptr_t>(factory_retained.data.ptr_val));
  int64_t length = 0;
  std::memcpy(&length, data, sizeof(length));
  check(length == 6, "source process root has six children");
  std::memcpy(out, static_cast<const uint8_t *>(data) + sizeof(length),
              6 * sizeof(*out));
  for (unsigned i = 0; i < 6; ++i) {
    check((out[i].type == ESHKOL_VALUE_HEAP_PTR ||
           out[i].type == ESHKOL_VALUE_CALLABLE) &&
              out[i].data.ptr_val != 0,
          "source child remains rooted");
  }
}
}  // namespace

extern "C" int64_t __wrap_et_tr3_c_factory_stage_v1(int64_t value) {
  stage = value;
  if (mode == Mode::cleanup && value == ET_TR3_C_STAGE_M3T_INITIALIZER)
    stage_live_borrow();
  if (mode == Mode::lease_busy && value == ET_TR3_C_STAGE_LEASE)
    stage_unborrowed_batch();
  return __real_et_tr3_c_factory_stage_v1(value);
}
extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  if (((mode == Mode::t2 && stage == ET_TR3_C_STAGE_T2) ||
       (mode == Mode::lease && stage == ET_TR3_C_STAGE_LEASE)) && !denied) {
    denied = true;
    mode = Mode::idle;  // Leave exception construction and cleanup unmodified.
    return bounded_denial(arena, [&]() {
      return __real_arena_allocate_vector_with_header(arena, capacity);
    });
  }
  return __real_arena_allocate_vector_with_header(arena, capacity);
}
extern "C" int64_t __wrap_et_d2_dataset_open_v1(
    const void *owner, int64_t shuffle_slots) {
  const int64_t answer = __real_et_d2_dataset_open_v1(owner, shuffle_slots);
  if (answer == 0) {
    ++opens;
    dataset_owner = owner;
    const int64_t live = et_d2_dataset_test_live_count_v1();
    if (live > peak_live) peak_live = live;
  }
  return answer;
}
extern "C" int64_t __wrap_et_d2_dataset_close_v1(const void *owner) {
  const int64_t answer = __real_et_d2_dataset_close_v1(owner);
  if (answer == 0) ++closes;
  if (answer == ET_D2_NATIVE_STATUS_INVALID_STATE) ++refused_closes;
  return answer;
}

int main(int argc, char **argv) {
  check(argc == 3, "mode and corpus arguments");
  if (std::strcmp(argv[1], "t2") == 0) mode = Mode::t2;
  else if (std::strcmp(argv[1], "m3t") == 0) mode = Mode::m3t;
  else if (std::strcmp(argv[1], "o2") == 0) mode = Mode::o2;
  else if (std::strcmp(argv[1], "lease") == 0) mode = Mode::lease;
  else if (std::strcmp(argv[1], "cleanup") == 0) mode = Mode::cleanup;
  else if (std::strcmp(argv[1], "retention") == 0) mode = Mode::retention;
  else if (std::strcmp(argv[1], "lease-busy") == 0) mode = Mode::lease_busy;
  else if (std::strcmp(argv[1], "close-busy") == 0) mode = Mode::close_busy;
  else fail("unknown mode");
  check(et_tr3_c_private_initialize_v1() == ET_TR3_C_INIT_READY_V1,
        "private initializer");
  check(et_d2_dataset_test_live_count_v1() == 0, "initial D2 live baseline");
  static const uint8_t json[] =
      "{\"config-schema-major\":1,\"config-schema-minor\":0,"
      "\"model.context-length\":2,\"model.hidden-size\":4,"
      "\"model.layer-count\":1,\"model.query-head-count\":2,"
      "\"model.vocabulary-size\":256,\"run.seed\":1729}";
  et_tr3_c_create_request_v1 request{};
  request.size = sizeof(request);
  request.major = 1;
  request.x1_json = json;
  request.x1_len = sizeof(json) - 1;
  request.directory = reinterpret_cast<const uint8_t *>(argv[2]);
  request.directory_len = static_cast<uint32_t>(std::strlen(argv[2]));
  request.batch_size = 1;
  request.sequence_length = 2;
  request.maximum_manifest_bytes = 1048576;
  request.maximum_shard_bytes = 1048576;
  request.maximum_total_tokens = 4;
  request.maximum_batch_bytes = 34;
  request.shuffle_window_rows = 1;
  request.packing = 1;
  request.adamw_f32_bits[0] = UINT32_C(0x3dcccccd);
  request.adamw_f32_bits[1] = UINT32_C(0x3f000000);
  request.adamw_f32_bits[2] = UINT32_C(0x3f000000);
  request.adamw_f32_bits[3] = UINT32_C(0x3a83126f);
  check(digest(argv[2], request.expected_manifest_sha256), "manifest digest");
  if (mode == Mode::m3t || mode == Mode::cleanup)
    et_m3t_test_fail_alloc_after(0);
  if (mode == Mode::o2) et_o2_test_fail_alloc_after_v1(0);
  const et_tr3_c_result_v1 first =
      et_tr3_c_private_trainer_create_v1(&request);
  et_m3t_test_fail_alloc_after(SIZE_MAX);
  et_o2_test_reset_failpoints_v1();
  if (std::strcmp(argv[1], "close-busy") == 0) {
    check(first.status == ET_TR3_C_OK && first.stage == ET_TR3_C_STAGE_NONE &&
              first.handle != nullptr, "trainer created before close-busy cut");
    stage_unborrowed_batch();
    const et_tr3_c_result_v1 busy =
        et_tr3_c_private_trainer_close_v1(first.handle);
    check(busy.status == ET_TR3_C_INVALID_STATE &&
              busy.stage == ET_TR3_C_STAGE_CLOSE &&
              busy.reason == ET_TR3_C_REASON_BUSY &&
              busy.original_category == 0 && busy.handle == nullptr,
          "native D2 busy preflight preserves live handle");
    check(opens == 1 && closes == 0 &&
              et_d2_dataset_test_live_count_v1() == 1 &&
              et_d2_batch_test_live_count_v1() == 1 &&
              optimizer_count() == 1,
          "busy close left both receivers live");
    check(et_d2_batch_release_preflight_v1(dataset_owner,
                                           active_generation) == 0 &&
              et_d2_batch_release_v1(dataset_owner,
                                     active_generation) == 0 &&
              et_d2_batch_test_live_count_v1() == 0,
          "accepted D2 release restores idle");
    const et_tr3_c_result_v1 ended =
        et_tr3_c_private_trainer_close_v1(first.handle);
    check(ended.status == ET_TR3_C_OK && ended.stage == ET_TR3_C_STAGE_NONE &&
              ended.reason == ET_TR3_C_REASON_RAISED_E1 &&
              ended.original_category == 0 && ended.handle == nullptr,
          "same handle closes after busy dependency clears");
    const et_tr3_c_result_v1 repeated =
        et_tr3_c_private_trainer_close_v1(first.handle);
    check(repeated.status == ET_TR3_C_INVALID_STATE &&
              repeated.stage == ET_TR3_C_STAGE_CLOSE &&
              repeated.reason == ET_TR3_C_REASON_ALREADY_CLOSED &&
              repeated.original_category == 0 && repeated.handle == nullptr,
          "retried close leaves exact tombstone");
    std::printf("TR3 factory fault close-busy busy=7/10/2/0 "
                "retry=0/0/0/0 repeat=7/10/5/0 D2=1/0/1 PASS\n");
    return 0;
  }
  if (std::strcmp(argv[1], "retention") == 0) {
    check(first.status == ET_TR3_C_OK && first.stage == ET_TR3_C_STAGE_NONE &&
              first.reason == ET_TR3_C_REASON_RAISED_E1 &&
              first.original_category == 0 && first.handle != nullptr,
          "published exact handle");
    eshkol_tagged_value_t before[6]{};
    eshkol_tagged_value_t after[6]{};
    retained_snapshot(before);
    check(opens == 1 && closes == 0 &&
              et_d2_dataset_test_live_count_v1() == 1 &&
              optimizer_count() == 1,
          "children live before close");
    et_tr3_c_handle_v1 *const alias = first.handle;
    const et_tr3_c_result_v1 forged = et_tr3_c_private_trainer_close_v1(
        reinterpret_cast<et_tr3_c_handle_v1 *>(&request));
    check(forged.status == ET_TR3_C_INVALID_ARGUMENT &&
              forged.stage == ET_TR3_C_STAGE_CLOSE &&
              forged.reason == ET_TR3_C_REASON_BAD_HANDLE &&
              forged.original_category == 0 && forged.handle == nullptr,
          "forged address rejected before dereference");
    const et_tr3_c_result_v1 ended = et_tr3_c_private_trainer_close_v1(alias);
    check(ended.status == ET_TR3_C_OK && ended.stage == ET_TR3_C_STAGE_NONE &&
              ended.reason == ET_TR3_C_REASON_RAISED_E1 &&
              ended.original_category == 0 && ended.handle == nullptr,
          "exact alias closes lease");
    retained_snapshot(after);
    for (unsigned i = 0; i < 6; ++i)
      check(before[i].type == after[i].type &&
                before[i].data.raw_val == after[i].data.raw_val,
            "all six source children retained through close");
    check(opens == 1 && closes == 0 &&
              et_d2_dataset_test_live_count_v1() == 1 &&
              optimizer_count() == 1,
          "native children remain live after lease close");
    const et_tr3_c_result_v1 repeated =
        et_tr3_c_private_trainer_close_v1(first.handle);
    check(repeated.status == ET_TR3_C_INVALID_STATE &&
              repeated.stage == ET_TR3_C_STAGE_CLOSE &&
              repeated.reason == ET_TR3_C_REASON_ALREADY_CLOSED &&
              repeated.original_category == 0 && repeated.handle == nullptr,
          "exact handle tombstone rejects repeat close");
    const et_tr3_c_result_v1 again =
        et_tr3_c_private_trainer_create_v1(&request);
    check(again.status == ET_TR3_C_INVALID_STATE &&
              again.stage == ET_TR3_C_STAGE_ADMISSION &&
              again.reason == ET_TR3_C_REASON_ATTEMPT_USED &&
              again.original_category == 0 && again.handle == nullptr,
          "successful attempt also consumed");
    std::printf("TR3 factory fault retention close=0/0/0/0 "
                "repeat=7/10/5/0 D2=1/0/1 O2=1 root=6 PASS\n");
    return 0;
  }
  check(first.status == (std::strcmp(argv[1], "lease-busy") == 0
                             ? ET_TR3_C_INVALID_STATE : ET_TR3_C_INTERNAL) &&
            first.reason == (std::strcmp(argv[1], "cleanup") == 0
                                 ? ET_TR3_C_REASON_CLEANUP_FAILED
                                 : std::strcmp(argv[1], "lease") == 0
                                       ? ET_TR3_C_REASON_FOREIGN_EXCEPTION
                                       : ET_TR3_C_REASON_RAISED_E1) &&
            first.original_category == (std::strcmp(argv[1], "cleanup") == 0
                                            ? ET_TR3_C_INTERNAL : 0) &&
            first.handle == nullptr,
        "producer or lease failure has exact tuple and no handle");
  if (std::strcmp(argv[1], "t2") == 0) {
    check(first.stage == ET_TR3_C_STAGE_T2, "T2 result stage");
    check(denied && stage == ET_TR3_C_STAGE_T2, "actual T2 allocation denial");
    check(opens == 0 && closes == 0 && peak_live == 0,
          "T2 failure precedes D2 open");
  } else if (std::strcmp(argv[1], "m3t") == 0) {
    check(first.stage == ET_TR3_C_STAGE_M3T_INITIALIZER,
          "M3T result stage");
    check(stage == ET_TR3_C_STAGE_M3T_INITIALIZER, "M3T initializer reached");
    check(opens == 1 && closes == 1 && peak_live == 1,
          "one open D2 receiver closed after later failure");
  } else if (std::strcmp(argv[1], "o2") == 0) {
    check(first.stage == ET_TR3_C_STAGE_O2 && stage == ET_TR3_C_STAGE_O2,
          "O2 allocation failure stage");
    check(opens == 1 && closes == 1 && peak_live == 1 &&
              optimizer_count() == 0,
          "failed O2 producer returned no optimizer and D2 closed");
  } else if (std::strcmp(argv[1], "lease") == 0) {
    check(first.stage == ET_TR3_C_STAGE_LEASE && denied,
          "real lease-stage allocation denial");
    check(opens == 1 && closes == 1 && peak_live == 1 &&
              optimizer_count() == 1,
          "lease failure retained completed O2 child and closed D2");
  } else if (std::strcmp(argv[1], "lease-busy") == 0) {
    check(first.status == ET_TR3_C_INVALID_STATE &&
              first.stage == ET_TR3_C_STAGE_LEASE &&
              first.reason == ET_TR3_C_REASON_RAISED_E1 &&
              first.original_category == 0 && first.handle == nullptr,
          "authentic lease busy from native D2 preflight");
    check(active_generation > 0 && opens == 1 && closes == 1 &&
              et_d2_dataset_test_live_count_v1() == 0 &&
              et_d2_batch_test_live_count_v1() == 0 &&
              optimizer_count() == 1,
          "lease busy cut cleans D2 and retains O2");
  } else {
    check(first.stage == ET_TR3_C_STAGE_M3T_INITIALIZER &&
              stage == ET_TR3_C_STAGE_M3T_INITIALIZER &&
              active_generation > 0 && active_borrow > 0,
          "first M3T failure with exact D2 borrow");
    check(opens == 1 && closes == 0 && refused_closes == 1 && peak_live == 1 &&
              et_d2_dataset_test_live_count_v1() == 1 &&
              et_d2_batch_test_borrow_count_v1() == 1,
          "real D2 close refusal preserves retained child");
  }
  if (std::strcmp(argv[1], "cleanup") != 0)
    check(et_d2_dataset_test_live_count_v1() == 0,
          "D2 live count restored after ordinary failure");
  const et_tr3_c_result_v1 again =
      et_tr3_c_private_trainer_create_v1(&request);
  check(again.status == ET_TR3_C_INVALID_STATE &&
            again.stage == ET_TR3_C_STAGE_ADMISSION &&
            again.reason == ET_TR3_C_REASON_ATTEMPT_USED &&
            again.original_category == 0 && again.handle == nullptr,
        "failed attempt is consumed");
  const et_tr3_c_result_v1 absent =
      et_tr3_c_private_trainer_close_v1(first.handle);
  check(absent.status == ET_TR3_C_INVALID_ARGUMENT &&
            absent.stage == ET_TR3_C_STAGE_CLOSE &&
            absent.reason == ET_TR3_C_REASON_BAD_HANDLE &&
            absent.original_category == 0 && absent.handle == nullptr,
        "failed attempt owns no closeable handle");
  std::printf("TR3 factory fault %s first=%u/%u/%u/%u "
              "again=%u/%u/%u/%u D2=%u/%u/%u/%lld O2=%zu PASS\n",
              argv[1], first.status, first.stage, first.reason,
              first.original_category, again.status, again.stage,
              again.reason, again.original_category, opens, closes,
              refused_closes,
              static_cast<long long>(et_d2_dataset_test_live_count_v1()),
              optimizer_count());
  return 0;
}
