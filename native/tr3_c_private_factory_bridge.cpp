#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <setjmp.h>

#include <eshkol/eshkol.h>
#include <eshkol/core/runtime.h>
#include "arena_memory.h"
#include "tr3_c_private_factory_bridge.h"

#undef setjmp
extern "C" int setjmp(jmp_buf environment) __attribute__((returns_twice));

extern "C" int et_tr3_c_private_ready_internal_v1(void);
extern "C" void eshkol_get_raised_value(eshkol_tagged_value_t *value);
extern "C" void eshkol_set_raised_value(const eshkol_tagged_value_t *value);
extern "C" eshkol_tagged_value_t tr3_c_factory_create_source(
    eshkol_tagged_value_t, eshkol_tagged_value_t, eshkol_tagged_value_t,
    eshkol_tagged_value_t, eshkol_tagged_value_t, eshkol_tagged_value_t,
    eshkol_tagged_value_t) __asm__("tr3-c-factory-create-source");
extern "C" eshkol_tagged_value_t tr3_c_factory_close_source(
    eshkol_tagged_value_t)
    __asm__("tr3-c-factory-close-source");

static_assert(sizeof(et_tr3_c_create_request_v1) == 176);
static_assert(sizeof(et_tr3_c_result_v1) == 24);
static_assert(sizeof(eshkol_tagged_value_t) == 16);

struct et_tr3_c_private_trainer_handle_v1 {
  uint64_t identity;
};

namespace {
std::atomic_flag operation = ATOMIC_FLAG_INIT;
bool attempted = false;
bool live = false;
bool closed = false;
et_tr3_c_private_trainer_handle_v1 handle{UINT64_C(0x4553484b54523343)};
eshkol_tagged_value_t trainer_shell{};
et_tr3_c_create_request_v1 staged{};
char json_bytes[16385]{};
char directory_bytes[4097]{};
char hex_bytes[5][9]{};
uint32_t source_status = ET_TR3_C_OK;
uint32_t source_stage = ET_TR3_C_STAGE_ADMISSION;
uint32_t source_reason = ET_TR3_C_REASON_RAISED_E1;
uint32_t source_original = 0;

et_tr3_c_result_v1 result(uint32_t status, uint32_t stage, uint32_t reason,
                          uint32_t original = 0,
                          et_tr3_c_handle_v1 *published = nullptr) {
  return {status, stage, reason, original, published};
}

bool ascii_json(const uint8_t *text, uint32_t length) {
  if (text == nullptr || length == 0 || length > 16384) return false;
  for (uint32_t i = 0; i < length; ++i)
    if (text[i] == 0 || text[i] > 127) return false;
  return true;
}

bool utf8_directory(const uint8_t *text, uint32_t length) {
  if (text == nullptr || length == 0 || length > 4096) return false;
  for (uint32_t i = 0; i < length;) {
    const uint8_t first = text[i++];
    if (first == 0) return false;
    if (first < 0x80) continue;
    uint32_t tail = 0;
    uint32_t code = 0;
    uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      tail = 1; code = first & 0x1f; minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      tail = 2; code = first & 0x0f; minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      tail = 3; code = first & 0x07; minimum = 0x10000;
    } else return false;
    if (tail > length - i) return false;
    for (uint32_t n = 0; n < tail; ++n) {
      const uint8_t byte = text[i++];
      if ((byte & 0xc0) != 0x80) return false;
      code = (code << 6) | (byte & 0x3f);
    }
    if (code < minimum || code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff)) return false;
  }
  return true;
}

bool nonnegative_finite(uint32_t bits) { return bits < UINT32_C(0x7f800000); }
bool accepted_bits(const uint32_t *bits) {
  return nonnegative_finite(bits[0]) &&
         nonnegative_finite(bits[1]) && bits[1] < UINT32_C(0x3f800000) &&
         nonnegative_finite(bits[2]) && bits[2] < UINT32_C(0x3f800000) &&
         nonnegative_finite(bits[3]) && bits[3] > 0 &&
         nonnegative_finite(bits[4]);
}

bool valid_request(const et_tr3_c_create_request_v1 *request) {
  if (request == nullptr || request->size != sizeof(*request) ||
      request->major != 1 || request->minor != 0 || request->flags != 0)
    return false;
  for (uint8_t byte : request->reserved) if (byte != 0) return false;
  if (!ascii_json(request->x1_json, request->x1_len) ||
      !utf8_directory(request->directory, request->directory_len)) return false;
  if (request->batch_size != 1 || request->sequence_length != 2 ||
      request->maximum_manifest_bytes <= 0 ||
      request->maximum_shard_bytes <= 0 ||
      request->maximum_total_tokens < 0 ||
      request->maximum_batch_bytes <= 0 ||
      request->shuffle_window_rows <= 0 ||
      request->shuffle_seed_present > 1 || request->packing > 1 ||
      (request->shuffle_seed_present == 0 && request->shuffle_seed != 0) ||
      request->shuffle_seed < 0 || !accepted_bits(request->adamw_f32_bits))
    return false;
  return true;
}

void render_hex(uint32_t bits, char *out) {
  static constexpr char digits[] = "0123456789abcdef";
  for (unsigned i = 0; i < 8; ++i)
    out[i] = digits[(bits >> (28 - i * 4)) & 15];
  out[8] = '\0';
}

bool tagged_string(const char *text, eshkol_tagged_value_t *out) {
  void *copy = eshkol_runtime_copy_string(get_global_arena_shared(), text);
  if (copy == nullptr) return false;
  *out = eshkol_make_ptr(reinterpret_cast<uint64_t>(copy),
                         ESHKOL_VALUE_HEAP_PTR);
  return true;
}

bool source_true(eshkol_tagged_value_t value) {
  return value.type == ESHKOL_VALUE_BOOL && value.data.int_val != 0;
}

bool invoke_create() {
  jmp_buf frame;
  eshkol_exception_handler_t *const prior = g_exception_handler_stack;
  eshkol_push_exception_handler(&frame);
  if (g_exception_handler_stack == prior) {
    source_status = ET_TR3_C_INTERNAL;
    source_reason = ET_TR3_C_REASON_FOREIGN_EXCEPTION;
    return false;
  }
  if (setjmp(frame) != 0) {
    eshkol_tagged_value_t raised;
    eshkol_get_raised_value(&raised);
    eshkol_parallel_scope_end();
    eshkol_pop_exception_handler();
    eshkol_set_raised_value(&raised);
    source_status = ET_TR3_C_INTERNAL;
    source_reason = ET_TR3_C_REASON_FOREIGN_EXCEPTION;
    return false;
  }
  eshkol_parallel_scope_begin();
  eshkol_tagged_value_t args[7];
  const char *strings[7] = {json_bytes, directory_bytes, hex_bytes[0],
                            hex_bytes[1], hex_bytes[2], hex_bytes[3],
                            hex_bytes[4]};
  bool prepared = true;
  for (unsigned i = 0; i < 7; ++i)
    if (!tagged_string(strings[i], &args[i])) { prepared = false; break; }
  if (!prepared) {
    source_status = ET_TR3_C_INTERNAL;
    source_reason = ET_TR3_C_REASON_FOREIGN_EXCEPTION;
    eshkol_parallel_scope_end();
    eshkol_pop_exception_handler();
    return false;
  }
  const eshkol_tagged_value_t answer = tr3_c_factory_create_source(
      args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
  const bool published = answer.type == ESHKOL_VALUE_HEAP_PTR &&
                         answer.data.ptr_val != 0;
  if (published) trainer_shell = answer;
  eshkol_parallel_scope_end();
  eshkol_pop_exception_handler();
  return published;
}

bool invoke_close() {
  jmp_buf frame;
  eshkol_exception_handler_t *const prior = g_exception_handler_stack;
  eshkol_push_exception_handler(&frame);
  if (g_exception_handler_stack == prior) {
    source_status = ET_TR3_C_INTERNAL;
    source_reason = ET_TR3_C_REASON_FOREIGN_EXCEPTION;
    return false;
  }
  if (setjmp(frame) != 0) {
    eshkol_tagged_value_t raised;
    eshkol_get_raised_value(&raised);
    eshkol_parallel_scope_end();
    eshkol_pop_exception_handler();
    eshkol_set_raised_value(&raised);
    source_status = ET_TR3_C_INTERNAL;
    source_reason = ET_TR3_C_REASON_FOREIGN_EXCEPTION;
    return false;
  }
  eshkol_parallel_scope_begin();
  const eshkol_tagged_value_t answer = tr3_c_factory_close_source(trainer_shell);
  eshkol_parallel_scope_end();
  eshkol_pop_exception_handler();
  return source_true(answer);
}
}  // namespace

extern "C" int64_t et_tr3_c_factory_input_v1(int64_t index) {
  switch (index) {
    case 0: return staged.batch_size;
    case 1: return staged.sequence_length;
    case 2: return staged.maximum_manifest_bytes;
    case 3: return staged.maximum_shard_bytes;
    case 4: return staged.maximum_total_tokens;
    case 5: return staged.maximum_batch_bytes;
    case 6: return staged.shuffle_seed;
    case 7: return staged.shuffle_window_rows;
    case 8: return staged.shuffle_seed_present;
    case 9: return staged.packing;
    default: return -1;
  }
}
extern "C" int64_t et_tr3_c_factory_expected_byte_v1(int64_t index) {
  if (index < 0 || index >= 32) return -1;
  return staged.expected_manifest_sha256[index];
}
extern "C" int64_t et_tr3_c_factory_stage_v1(int64_t stage) {
  if (stage >= ET_TR3_C_STAGE_X1 && stage <= ET_TR3_C_STAGE_LEASE)
    source_stage = static_cast<uint32_t>(stage);
  return source_stage;
}
extern "C" int64_t et_tr3_c_factory_report_v1(int64_t status, int64_t reason,
                                                  int64_t original) {
  source_status = status >= 0 && status <= ET_TR3_C_INTERNAL
                      ? static_cast<uint32_t>(status) : ET_TR3_C_INTERNAL;
  source_reason = reason >= 0 && reason <= ET_TR3_C_REASON_MALFORMED_REQUEST
                      ? static_cast<uint32_t>(reason)
                      : ET_TR3_C_REASON_FOREIGN_EXCEPTION;
  source_original = original >= 0 && original <= ET_TR3_C_INTERNAL
                        ? static_cast<uint32_t>(original) : ET_TR3_C_INTERNAL;
  return 0;
}
extern "C" int64_t et_tr3_c_factory_reason_v1(void) { return source_reason; }

extern "C" et_tr3_c_result_v1 et_tr3_c_private_trainer_create_v1(
    const et_tr3_c_create_request_v1 *request) {
  if (!et_tr3_c_private_ready_internal_v1())
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_NOT_READY);
  if (operation.test_and_set(std::memory_order_acquire))
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_BUSY);
  if (region_get_depth() != 0) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_BUSY);
  }
  if (!valid_request(request)) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_MALFORMED_REQUEST);
  }
  if (attempted) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_ATTEMPT_USED);
  }
  attempted = true;
  staged = *request;
  std::memcpy(json_bytes, request->x1_json, request->x1_len);
  json_bytes[request->x1_len] = '\0';
  std::memcpy(directory_bytes, request->directory, request->directory_len);
  directory_bytes[request->directory_len] = '\0';
  for (unsigned i = 0; i < 5; ++i)
    render_hex(request->adamw_f32_bits[i], hex_bytes[i]);
  source_status = ET_TR3_C_OK;
  source_stage = ET_TR3_C_STAGE_X1;
  source_reason = ET_TR3_C_REASON_RAISED_E1;
  source_original = 0;
  const bool created = invoke_create();
  et_tr3_c_result_v1 answer;
  if (created && source_status == ET_TR3_C_OK) {
    live = true;
    answer = result(ET_TR3_C_OK, ET_TR3_C_STAGE_NONE,
                    ET_TR3_C_REASON_RAISED_E1, 0, &handle);
  } else {
    answer = result(source_status == ET_TR3_C_OK ? ET_TR3_C_INTERNAL
                                                 : source_status,
                    source_stage, source_reason, source_original);
  }
  operation.clear(std::memory_order_release);
  return answer;
}

extern "C" et_tr3_c_result_v1 et_tr3_c_private_trainer_close_v1(
    et_tr3_c_handle_v1 *candidate) {
  if (!et_tr3_c_private_ready_internal_v1())
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_ADMISSION,
                  ET_TR3_C_REASON_NOT_READY);
  if (operation.test_and_set(std::memory_order_acquire))
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_CLOSE,
                  ET_TR3_C_REASON_BUSY);
  if (region_get_depth() != 0) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_CLOSE,
                  ET_TR3_C_REASON_BUSY);
  }
  if (candidate != &handle || !attempted || (!live && !closed)) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_ARGUMENT, ET_TR3_C_STAGE_CLOSE,
                  ET_TR3_C_REASON_BAD_HANDLE);
  }
  if (closed) {
    operation.clear(std::memory_order_release);
    return result(ET_TR3_C_INVALID_STATE, ET_TR3_C_STAGE_CLOSE,
                  ET_TR3_C_REASON_ALREADY_CLOSED);
  }
  source_status = ET_TR3_C_OK;
  source_stage = ET_TR3_C_STAGE_CLOSE;
  source_reason = ET_TR3_C_REASON_RAISED_E1;
  source_original = 0;
  const bool unlinked = invoke_close();
  et_tr3_c_result_v1 answer;
  if (unlinked && source_status == ET_TR3_C_OK) {
    live = false;
    closed = true;
    answer = result(ET_TR3_C_OK, ET_TR3_C_STAGE_NONE,
                    ET_TR3_C_REASON_RAISED_E1);
  } else {
    answer = result(source_status == ET_TR3_C_OK ? ET_TR3_C_INTERNAL
                                                 : source_status,
                    ET_TR3_C_STAGE_CLOSE, source_reason, source_original);
  }
  operation.clear(std::memory_order_release);
  return answer;
}
