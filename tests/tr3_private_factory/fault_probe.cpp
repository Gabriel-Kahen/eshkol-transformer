// Linked, test-only faults against the pinned runtime and native producers.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "arena_memory.h"
#pragma clang diagnostic pop

#define ET_D2_NATIVE_TESTING 1
#include "d2_native.h"
#include "tr3_c_private_factory_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void et_m3t_test_fail_alloc_after(size_t);
extern "C" int64_t __real_et_tr3_c_factory_stage_v1(int64_t);
extern "C" void *__real_arena_allocate_vector_with_header(arena_t *, size_t);
extern "C" int64_t __real_et_d2_dataset_open_v1(const void *, int64_t);
extern "C" int64_t __real_et_d2_dataset_close_v1(const void *);

namespace {
enum class Mode { idle, t2, m3t };
Mode mode = Mode::idle;
int64_t stage = 0;
bool denied = false;
unsigned opens = 0;
unsigned closes = 0;
int64_t peak_live = 0;

[[noreturn]] void fail(const char *message) {
  std::fprintf(stderr, "TR3 factory fault probe FAIL: %s stage=%lld "
               "denied=%d opens=%u closes=%u live=%lld peak=%lld\n",
               message, static_cast<long long>(stage), denied, opens, closes,
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
}  // namespace

extern "C" int64_t __wrap_et_tr3_c_factory_stage_v1(int64_t value) {
  stage = value;
  return __real_et_tr3_c_factory_stage_v1(value);
}
extern "C" void *__wrap_arena_allocate_vector_with_header(
    arena_t *arena, size_t capacity) {
  if (mode == Mode::t2 && stage == ET_TR3_C_STAGE_T2 && !denied) {
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
    const int64_t live = et_d2_dataset_test_live_count_v1();
    if (live > peak_live) peak_live = live;
  }
  return answer;
}
extern "C" int64_t __wrap_et_d2_dataset_close_v1(const void *owner) {
  const int64_t answer = __real_et_d2_dataset_close_v1(owner);
  if (answer == 0) ++closes;
  return answer;
}

int main(int argc, char **argv) {
  check(argc == 3, "mode and corpus arguments");
  if (std::strcmp(argv[1], "t2") == 0) mode = Mode::t2;
  else if (std::strcmp(argv[1], "m3t") == 0) mode = Mode::m3t;
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
  if (mode == Mode::m3t) et_m3t_test_fail_alloc_after(0);
  const et_tr3_c_result_v1 first =
      et_tr3_c_private_trainer_create_v1(&request);
  et_m3t_test_fail_alloc_after(SIZE_MAX);
  check(first.status == ET_TR3_C_INTERNAL &&
            first.reason == ET_TR3_C_REASON_RAISED_E1 &&
            first.original_category == 0 && first.handle == nullptr,
        "authenticated allocation failure has exact tuple and no handle");
  if (std::strcmp(argv[1], "t2") == 0) {
    check(first.stage == ET_TR3_C_STAGE_T2, "T2 result stage");
    check(denied && stage == ET_TR3_C_STAGE_T2, "actual T2 allocation denial");
    check(opens == 0 && closes == 0 && peak_live == 0,
          "T2 failure precedes D2 open");
  } else {
    check(first.stage == ET_TR3_C_STAGE_M3T_INITIALIZER,
          "M3T result stage");
    check(stage == ET_TR3_C_STAGE_M3T_INITIALIZER, "M3T initializer reached");
    check(opens == 1 && closes == 1 && peak_live == 1,
          "one open D2 receiver closed after later failure");
  }
  check(et_d2_dataset_test_live_count_v1() == 0,
        "D2 live count restored after failure");
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
              "again=%u/%u/%u/%u D2=%u/%u/%lld PASS\n",
              argv[1], first.status, first.stage, first.reason,
              first.original_category, again.status, again.stage,
              again.reason, again.original_category, opens, closes,
              static_cast<long long>(et_d2_dataset_test_live_count_v1()));
  return 0;
}
