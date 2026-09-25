#include <atomic>
#include <csetjmp>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <eshkol/eshkol.h>
#include "arena_memory.h"
#include "tr3_c_private_factory_bridge.h"

extern "C" int64_t et_tr3_c_factory_input_v1(int64_t);
extern "C" int64_t et_tr3_c_factory_expected_byte_v1(int64_t);
extern "C" int64_t et_tr3_c_factory_stage_v1(int64_t);
extern "C" int64_t et_tr3_c_factory_report_v1(int64_t, int64_t, int64_t);

extern "C" eshkol_exception_handler_t *g_exception_handler_stack = nullptr;
extern "C" std::atomic<arena_t *> __repl_shared_arena{nullptr};
static arena_t arena{};
static eshkol_exception_handler_t frame{};
static eshkol_tagged_value_t raised{};
static int parallel_depth;
static int copies;
static char copied[7][16385];
static bool source_failure;
static bool close_busy;
static uint64_t active_region_depth;
static int shell_token;

extern "C" uint64_t region_get_depth(void) { return active_region_depth; }

extern "C" int eshkol_runtime_init(void) { return 0; }
extern "C" arena_t *get_global_arena_shared(void) { return &arena; }
extern "C" void eshkol_push_exception_handler(void *buffer) {
  frame.jmp_buf_ptr = buffer;
  frame.prev = g_exception_handler_stack;
  g_exception_handler_stack = &frame;
}
extern "C" void eshkol_pop_exception_handler(void) {
  g_exception_handler_stack = g_exception_handler_stack->prev;
}
extern "C" void eshkol_parallel_scope_begin(void) { ++parallel_depth; }
extern "C" void eshkol_parallel_scope_end(void) { --parallel_depth; }
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *out) {
  *out = raised;
}
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t *value) {
  raised = *value;
}
extern "C" void __eshkol_lib_init__(void *input) {
  if (input != &arena || parallel_depth != 1) std::abort();
}
extern "C" void *eshkol_runtime_copy_string(void *input, const char *text) {
  if (input != &arena || copies >= 7) return nullptr;
  const size_t length = std::strlen(text);
  if (length >= sizeof(copied[0])) return nullptr;
  std::memcpy(copied[copies], text, length + 1);
  return copied[copies++];
}

static eshkol_tagged_value_t boolean(bool value) {
  eshkol_tagged_value_t result{};
  result.type = ESHKOL_VALUE_BOOL;
  result.data.int_val = value ? 1 : 0;
  return result;
}
extern "C" eshkol_tagged_value_t fake_create(
    eshkol_tagged_value_t json, eshkol_tagged_value_t directory,
    eshkol_tagged_value_t rate, eshkol_tagged_value_t beta1,
    eshkol_tagged_value_t beta2, eshkol_tagged_value_t epsilon,
    eshkol_tagged_value_t decay) __asm__("tr3-c-factory-create-source");
extern "C" eshkol_tagged_value_t fake_create(
    eshkol_tagged_value_t json, eshkol_tagged_value_t directory,
    eshkol_tagged_value_t rate, eshkol_tagged_value_t beta1,
    eshkol_tagged_value_t beta2, eshkol_tagged_value_t epsilon,
    eshkol_tagged_value_t decay) {
  if (parallel_depth != 1 || copies != 7 ||
      std::strcmp(reinterpret_cast<const char *>(json.data.ptr_val), "{}") ||
      std::strcmp(reinterpret_cast<const char *>(directory.data.ptr_val), "corpus") ||
      std::strcmp(reinterpret_cast<const char *>(rate.data.ptr_val), "3dcccccd") ||
      std::strcmp(reinterpret_cast<const char *>(beta1.data.ptr_val), "3f000000") ||
      std::strcmp(reinterpret_cast<const char *>(beta2.data.ptr_val), "3f000000") ||
      std::strcmp(reinterpret_cast<const char *>(epsilon.data.ptr_val), "3a83126f") ||
      std::strcmp(reinterpret_cast<const char *>(decay.data.ptr_val), "00000000") ||
      et_tr3_c_factory_input_v1(0) != 1 ||
      et_tr3_c_factory_expected_byte_v1(0) != 0)
    std::abort();
  if (source_failure) {
    et_tr3_c_factory_stage_v1(ET_TR3_C_STAGE_D2);
    et_tr3_c_factory_report_v1(ET_TR3_C_UNSUPPORTED,
                               ET_TR3_C_REASON_RAISED_E1, 0);
    return boolean(false);
  }
  return eshkol_make_ptr(reinterpret_cast<uint64_t>(&shell_token),
                         ESHKOL_VALUE_HEAP_PTR);
}
extern "C" eshkol_tagged_value_t fake_close(eshkol_tagged_value_t)
    __asm__("tr3-c-factory-close-source");
extern "C" eshkol_tagged_value_t fake_close(eshkol_tagged_value_t trainer) {
  if (parallel_depth != 1 ||
      trainer.data.ptr_val != reinterpret_cast<uint64_t>(&shell_token))
    std::abort();
  if (close_busy) {
    close_busy = false;
    et_tr3_c_factory_report_v1(ET_TR3_C_INVALID_STATE,
                               ET_TR3_C_REASON_BUSY, 0);
    return boolean(false);
  }
  return boolean(true);
}

static bool matches(et_tr3_c_result_v1 result, uint32_t status,
                    uint32_t stage, uint32_t reason, bool handle) {
  return result.status == status && result.stage == stage &&
         result.reason == reason && result.original_category == 0 &&
         (result.handle != nullptr) == handle;
}

int main(int argc, char **argv) {
  source_failure = argc == 2 && std::strcmp(argv[1], "failure") == 0;
  static const uint8_t json[] = "{}";
  static const uint8_t directory[] = "corpus";
  et_tr3_c_create_request_v1 request{};
  request.size = sizeof(request);
  request.major = 1;
  request.x1_json = json;
  request.x1_len = 2;
  request.directory = directory;
  request.directory_len = 6;
  request.batch_size = 1;
  request.sequence_length = 2;
  request.maximum_manifest_bytes = 1048576;
  request.maximum_shard_bytes = 1048576;
  request.maximum_total_tokens = 4;
  request.maximum_batch_bytes = 34;
  request.shuffle_window_rows = 1;
  request.packing = 1;
  request.adamw_f32_bits[0] = 0x3dcccccd;
  request.adamw_f32_bits[1] = 0x3f000000;
  request.adamw_f32_bits[2] = 0x3f000000;
  request.adamw_f32_bits[3] = 0x3a83126f;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_NOT_READY, false)) return 1;
  if (et_tr3_c_private_initialize_v1() != ET_TR3_C_INIT_READY_V1) return 2;
  active_region_depth = 1;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_BUSY, false)) return 18;
  active_region_depth = 0;
  if (!matches(et_tr3_c_private_trainer_create_v1(nullptr),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 3;
  request.size = 0;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 4;
  request.size = sizeof(request);
  request.major = 2;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 15;
  request.major = 1;
  const uint8_t embedded_nul[] = {'{', 0, '}'};
  request.x1_json = embedded_nul;
  request.x1_len = 3;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 5;
  request.x1_json = json;
  request.x1_len = 2;
  const uint8_t malformed_utf8[] = {0xc0, 0x80};
  request.directory = malformed_utf8;
  request.directory_len = sizeof(malformed_utf8);
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 16;
  request.directory = directory;
  request.directory_len = 6;
  request.adamw_f32_bits[3] = 0;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_MALFORMED_REQUEST, false)) return 17;
  request.adamw_f32_bits[3] = 0x3a83126f;
  const et_tr3_c_result_v1 first = et_tr3_c_private_trainer_create_v1(&request);
  if (source_failure) {
    if (!matches(first, ET_TR3_C_UNSUPPORTED, ET_TR3_C_STAGE_D2,
                 ET_TR3_C_REASON_RAISED_E1, false)) return 6;
  } else if (!matches(first, ET_TR3_C_OK, ET_TR3_C_STAGE_NONE,
                      ET_TR3_C_REASON_RAISED_E1, true)) return 7;
  if (!matches(et_tr3_c_private_trainer_create_v1(&request),
               ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
               ET_TR3_C_REASON_ATTEMPT_USED, false)) return 8;
  if (source_failure) return parallel_depth == 0 ? 0 : 9;
  if (!matches(et_tr3_c_private_trainer_close_v1(nullptr),
               ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_CLOSE,
               ET_TR3_C_REASON_BAD_HANDLE, false)) return 10;
  close_busy = true;
  if (!matches(et_tr3_c_private_trainer_close_v1(first.handle),
               ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_CLOSE,
               ET_TR3_C_REASON_BUSY, false)) return 11;
  if (!matches(et_tr3_c_private_trainer_close_v1(first.handle),
               ET_TR3_C_OK, ET_TR3_C_STAGE_NONE,
               ET_TR3_C_REASON_RAISED_E1, false)) return 12;
  if (!matches(et_tr3_c_private_trainer_close_v1(first.handle),
               ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_CLOSE,
               ET_TR3_C_REASON_ALREADY_CLOSED, false)) return 13;
  return parallel_depth == 0 ? 0 : 14;
}
