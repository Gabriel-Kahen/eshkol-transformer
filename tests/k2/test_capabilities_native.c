#include "k2_capabilities_internal.h"

#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/kernel_abi.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct test_closure_header {
  uint8_t subtype;
  uint8_t flags;
  uint16_t reference_count;
  uint32_t size;
} test_closure_header;

typedef struct test_closure_body {
  uint64_t function_address;
  const void *environment;
  uint64_t source_expression;
  const char *name;
  uint8_t return_type;
  uint8_t input_arity;
  uint8_t flags;
  uint8_t reserved;
  uint32_t higher_order_type;
} test_closure_body;

typedef struct test_closure {
  test_closure_header header;
  test_closure_body body;
} test_closure;

typedef struct aligned_error {
  int64_t declared_length;
  union {
    et_kernel_error error;
    unsigned char bytes[264];
  };
} aligned_error;

typedef struct tagged_bytes {
  int64_t declared_length;
  unsigned char payload[512];
} tagged_bytes;

_Static_assert(offsetof(aligned_error, bytes) == 8u,
               "test diagnostic payload must follow its i64 header");
_Static_assert(sizeof(aligned_error) == 272u,
               "test diagnostic carrier layout changed");

static int failures;

#ifdef ET_K2_ALLOCATOR_WRAP
typedef struct tracked_allocation {
  void *pointer;
  size_t bytes;
} tracked_allocation;

static tracked_allocation tracked_allocations[256];
static int allocation_tracking;
static size_t allocation_fail_after;
static size_t allocation_calls;
static size_t allocation_requested_bytes;
static size_t allocation_live_blocks;
static size_t allocation_live_bytes;
static size_t allocation_peak_bytes;

void *__real_malloc(size_t bytes);
void *__real_calloc(size_t count, size_t bytes);
void *__real_realloc(void *pointer, size_t bytes);
void __real_free(void *pointer);

static size_t tracked_index(void *pointer) {
  if (pointer == NULL) {
    return 256u;
  }
  for (size_t index = 0; index < 256u; index++) {
    if (tracked_allocations[index].pointer == pointer) {
      return index;
    }
  }
  return 256u;
}

static void tracked_remove(void *pointer) {
  const size_t index = tracked_index(pointer);
  if (index != 256u) {
    allocation_live_blocks--;
    allocation_live_bytes -= tracked_allocations[index].bytes;
    tracked_allocations[index].pointer = NULL;
    tracked_allocations[index].bytes = 0u;
  }
}

static void tracked_add(void *pointer, size_t bytes) {
  if (pointer == NULL) {
    return;
  }
  for (size_t index = 0; index < 256u; index++) {
    if (tracked_allocations[index].pointer == NULL) {
      tracked_allocations[index].pointer = pointer;
      tracked_allocations[index].bytes = bytes;
      allocation_live_blocks++;
      allocation_live_bytes += bytes;
      if (allocation_live_bytes > allocation_peak_bytes) {
        allocation_peak_bytes = allocation_live_bytes;
      }
      return;
    }
  }
  failures++;
}

static int allocation_should_fail(size_t bytes) {
  size_t index;
  if (!allocation_tracking) {
    return 0;
  }
  index = allocation_calls++;
  allocation_requested_bytes += bytes;
  return index == allocation_fail_after;
}

void *__wrap_malloc(size_t bytes) {
  void *pointer;
  if (allocation_should_fail(bytes)) {
    return NULL;
  }
  pointer = __real_malloc(bytes);
  if (allocation_tracking) {
    tracked_add(pointer, bytes);
  }
  return pointer;
}

void *__wrap_calloc(size_t count, size_t bytes) {
  void *pointer;
  size_t total;
  if (count != 0u && bytes > SIZE_MAX / count) {
    total = SIZE_MAX;
  } else {
    total = count * bytes;
  }
  if (allocation_should_fail(total)) {
    return NULL;
  }
  pointer = __real_calloc(count, bytes);
  if (allocation_tracking) {
    tracked_add(pointer, total);
  }
  return pointer;
}

void *__wrap_realloc(void *old_pointer, size_t bytes) {
  void *new_pointer;
  if (allocation_should_fail(bytes)) {
    return NULL;
  }
  new_pointer = __real_realloc(old_pointer, bytes);
  if (new_pointer != NULL) {
    tracked_remove(old_pointer);
    if (allocation_tracking) {
      tracked_add(new_pointer, bytes);
    }
  }
  return new_pointer;
}

void __wrap_free(void *pointer) {
  tracked_remove(pointer);
  __real_free(pointer);
}

static void allocation_begin(size_t fail_after) {
  if (allocation_live_blocks != 0u) {
    failures++;
  }
  memset(tracked_allocations, 0, sizeof(tracked_allocations));
  allocation_fail_after = fail_after;
  allocation_calls = 0u;
  allocation_requested_bytes = 0u;
  allocation_live_bytes = 0u;
  allocation_peak_bytes = 0u;
  allocation_tracking = 1;
}

static void allocation_end(void) { allocation_tracking = 0; }
#endif

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,   \
                    #condition);                                               \
      failures++;                                                              \
    }                                                                          \
  } while (0)

static test_closure make_closure(uint64_t code, size_t *environment) {
  test_closure closure;
  memset(&closure, 0, sizeof(closure));
  closure.header.size = 40u;
  closure.body.function_address = code;
  closure.body.environment = environment;
  closure.body.input_arity = 1u;
  return closure;
}

static void initialize_error(aligned_error *diagnostic, unsigned char fill) {
  diagnostic->declared_length = 264;
  memset(diagnostic->bytes, fill, sizeof(diagnostic->bytes));
}

static void initialize_bytes(tagged_bytes *carrier, const void *bytes,
                             size_t byte_count) {
  CHECK(byte_count <= sizeof(carrier->payload));
  carrier->declared_length = (int64_t)byte_count;
  memset(carrier->payload, 0, sizeof(carrier->payload));
  if (bytes != NULL && byte_count != 0u) {
    memcpy(carrier->payload, bytes, byte_count);
  }
}

static void register_factories(void) {
  static size_t report_environment = (size_t)0x00010001u;
  static size_t request_environment = (size_t)0x00010001u;
  static size_t entry_environment = (size_t)0x00010001u;
  test_closure report = make_closure(UINT64_C(0x1010), &report_environment);
  test_closure request = make_closure(UINT64_C(0x2020), &request_environment);
  test_closure entry = make_closure(UINT64_C(0x3030), &entry_environment);
  CHECK(et_k2_private_report_factory_register_v1(&report.body) == 1);
  CHECK(et_k2_private_request_factory_register_v1(&request.body) == 1);
  CHECK(et_k2_private_entry_factory_register_v1(&entry.body) == 1);
  CHECK(et_k2_private_report_factory_authenticate_v1(&report.body) == 1);
  CHECK(et_k2_private_request_factory_authenticate_v1(&request.body) == 1);
  CHECK(et_k2_private_entry_factory_authenticate_v1(&entry.body) == 1);
}

static int all_bytes(const unsigned char *bytes, size_t count,
                     unsigned char value) {
  for (size_t index = 0; index < count; index++) {
    if (bytes[index] != value) {
      return 0;
    }
  }
  return 1;
}

static int zero_after_nul(const char *text, size_t capacity) {
  const char *terminal = (const char *)memchr(text, '\0', capacity);
  if (terminal == NULL) {
    return 0;
  }
  return all_bytes((const unsigned char *)terminal + 1u,
                   capacity - (size_t)(terminal + 1u - text), 0u);
}

static void test_factory_identity(void) {
  static size_t environment = (size_t)0x00010001u;
  aligned_error diagnostic;
  test_closure first = make_closure(UINT64_C(0x1110), &environment);
  test_closure same_code = make_closure(UINT64_C(0x1110), &environment);
  test_closure different = make_closure(UINT64_C(0x2220), &environment);
  test_closure third = make_closure(UINT64_C(0x3330), &environment);
  test_closure malformed = make_closure(UINT64_C(0x4440), &environment);

  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_private_report_factory_authenticate_v1(&first.body) == -9);
  CHECK(et_k2_private_report_factory_register_v1(&first.body) == -9);
  CHECK(et_k2_private_report_factory_register_v1(NULL) == -1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  malformed.body.flags = 1u;
  CHECK(et_k2_private_report_factory_register_v1(&malformed.body) == -1);
  CHECK(et_k2_private_report_factory_register_v1(&first.body) == 1);
  CHECK(et_k2_private_report_factory_register_v1(&same_code.body) == 1);
  CHECK(et_k2_private_report_factory_register_v1(&different.body) == -9);
  CHECK(et_k2_private_report_factory_authenticate_v1(&first.body) == 1);
  CHECK(et_k2_private_report_factory_authenticate_v1(&different.body) == 0);
  CHECK(et_k2_private_request_factory_register_v1(&same_code.body) == -9);
  CHECK(et_k2_private_request_factory_authenticate_v1(&same_code.body) == -9);
  CHECK(et_k2_private_request_factory_register_v1(&different.body) == 1);
  CHECK(et_k2_private_entry_factory_register_v1(&third.body) == 1);
  CHECK(et_k2_private_request_factory_authenticate_v1(&different.body) == 1);
  CHECK(et_k2_private_entry_factory_authenticate_v1(&third.body) == 1);
}

static void test_diagnostic_admission(void) {
  aligned_error diagnostic;
  tagged_bytes dtype;
  tagged_bytes device;
  tagged_bytes shape;
  _Alignas(int64_t) unsigned char misaligned[274];
  unsigned char original[sizeof(diagnostic)];
  int64_t declared_length = 264;
  int64_t nested_length = 4;

  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  memcpy(original, &diagnostic, sizeof(original));
  CHECK(et_k2_private_runtime_ensure_v1(NULL, 264) == 1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 263) == 1);
  CHECK(memcmp(&diagnostic, original, sizeof(original)) == 0);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 265) == 1);
  CHECK(memcmp(&diagnostic, original, sizeof(original)) == 0);
  diagnostic.declared_length = 263;
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 1);
  CHECK(diagnostic.declared_length == 263);
  CHECK(all_bytes(diagnostic.bytes, sizeof(diagnostic.bytes), 0xa5u));
  diagnostic.declared_length = 264;
  memset(misaligned, 0x5a, sizeof(misaligned));
  memcpy(misaligned + 1u, &declared_length, sizeof(declared_length));
  memcpy(original, misaligned + 1u, sizeof(original));
  CHECK(et_k2_private_runtime_ensure_v1(misaligned + 1u, 264) == 1);
  CHECK(memcmp(misaligned + 1u, original, sizeof(original)) == 0);
  CHECK(et_k2_private_runtime_ensure_v1(
            (void *)(uintptr_t)(UINTPTR_MAX - (uintptr_t)100u), 264) == 1);
  CHECK(et_k2_private_runtime_ensure_v1((void *)et_f32_tensor_provider_v1(),
                                        264) == 1);

  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  CHECK(diagnostic.declared_length == 264);
  CHECK(all_bytes(diagnostic.bytes, sizeof(diagnostic.bytes), 0u));

  initialize_error(&diagnostic, 0x3cu);
  memcpy(diagnostic.bytes, &nested_length, sizeof(nested_length));
  memcpy(diagnostic.bytes + sizeof(nested_length), "abcd", 4u);
  initialize_bytes(&dtype, "f32", 3u);
  initialize_bytes(&device, "cpu", 3u);
  initialize_bytes(&shape, NULL, 0u);
  memcpy(original, &diagnostic, sizeof(original));
  CHECK(et_k2_private_runtime_require_v1(1, 9, diagnostic.bytes, 4, &dtype, 3,
                                         &device, 3, &shape, 0, 0, 1,
                                         &diagnostic, 264) == 1);
  CHECK(memcmp(&diagnostic, original, sizeof(original)) == 0);

  initialize_error(&diagnostic, 0x3cu);
  memcpy(original, &diagnostic, sizeof(original));
  CHECK(et_k2_private_runtime_require_v1(
            1, 9, &diagnostic, -1, &dtype, 3, &device, 3, &shape, 0, 0, 1,
            &diagnostic, 264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(&diagnostic, original, sizeof(original)) == 0);
  CHECK(et_k2_private_runtime_require_v1(
            1, 9, diagnostic.bytes, INT64_MAX, &dtype, 3, &device, 3, &shape, 0,
            0, 1, &diagnostic, 264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(memcmp(&diagnostic, original, sizeof(original)) == 0);

  initialize_error(&diagnostic, 0x3cu);
  initialize_bytes(&dtype, "f32", 3u);
  dtype.declared_length = 2;
  CHECK(et_k2_private_runtime_require_v1(1, 9, &device, 3, &dtype, 3, &device,
                                         3, &shape, 0, 0, 1, &diagnostic,
                                         264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.declared_length == 264);
  CHECK(diagnostic.error.code == ET_K2_CODE_TRANSPORT);

  initialize_error(&diagnostic, 0x3cu);
  CHECK(et_k2_private_runtime_require_v1(
            1, 9, (const void *)(uintptr_t)(UINTPTR_MAX - 2u), 3, &device, 3,
            &device, 3, &shape, 0, 0, 1, &diagnostic,
            264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.error.code == ET_K2_CODE_TRANSPORT);
}

static int64_t require_request(int64_t generation, int64_t entry_index,
                               const char *operation, const char *dtype,
                               const char *device, const uint64_t *shape,
                               int64_t rank, int64_t deterministic,
                               aligned_error *diagnostic) {
  tagged_bytes operation_bytes;
  tagged_bytes dtype_bytes;
  tagged_bytes device_bytes;
  tagged_bytes shape_bytes;
  initialize_bytes(&operation_bytes, operation, strlen(operation));
  initialize_bytes(&dtype_bytes, dtype, strlen(dtype));
  initialize_bytes(&device_bytes, device, strlen(device));
  initialize_bytes(&shape_bytes, NULL, (size_t)rank * 8u);
  for (int64_t index = 0; index < rank; index++) {
    for (unsigned int byte = 0; byte < 8u; byte++) {
      shape_bytes.payload[index * 8 + byte] =
          (unsigned char)(shape[index] >> (byte * 8u));
    }
  }
  return et_k2_private_runtime_require_v1(
      generation, entry_index, &operation_bytes,
      operation_bytes.declared_length, &dtype_bytes,
      dtype_bytes.declared_length, &device_bytes, device_bytes.declared_length,
      &shape_bytes, shape_bytes.declared_length, rank, deterministic,
      diagnostic, 264);
}

static void check_k1_no_match(const aligned_error *diagnostic) {
  CHECK(diagnostic->declared_length == 264);
  CHECK(diagnostic->error.category == ET_K2_STATUS_UNSUPPORTED);
  CHECK(diagnostic->error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  CHECK(strcmp(diagnostic->error.operation, "capability-require") == 0);
  CHECK(zero_after_nul(diagnostic->error.operation,
                       sizeof(diagnostic->error.operation)));
  CHECK(zero_after_nul(diagnostic->error.message,
                       sizeof(diagnostic->error.message)));
}

static void test_k1_category_code_pairs(void) {
  static const struct {
    uint32_t category;
    uint32_t code;
    uint32_t expected_category;
  } valid_pairs[] = {
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_NULL_ARGUMENT,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_TEXT,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_ENUM,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_DUPLICATE_ENTRY,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_BUFFER_TOO_SMALL,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_SHAPE,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT,
       ET_K2_STATUS_INVALID_ARGUMENT},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INTEGER_OVERFLOW,
       ET_K2_STATUS_SHAPE_MISMATCH},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE,
       ET_K2_STATUS_SHAPE_MISMATCH},
      {ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER,
       ET_K2_STATUS_SHAPE_MISMATCH},
      {ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT,
       ET_K2_STATUS_DTYPE_MISMATCH},
      {ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER,
       ET_K2_STATUS_DEVICE_MISMATCH},
      {ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER,
       ET_K2_STATUS_NONCONTIGUOUS},
      {ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_SYMBOL_MISSING,
       ET_K2_STATUS_UNSUPPORTED},
      {ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED,
       ET_K2_STATUS_UNSUPPORTED},
      {ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
       ET_K2_STATUS_VERSION_MISMATCH},
      {ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_ABI_MAJOR_MISMATCH,
       ET_K2_STATUS_VERSION_MISMATCH},
      {ET_KERNEL_ERROR_VERSION_MISMATCH,
       ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE, ET_K2_STATUS_VERSION_MISMATCH},
      {ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED,
       ET_K2_STATUS_INTERNAL},
      {ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_INTEGER_OVERFLOW,
       ET_K2_STATUS_INTERNAL},
      {ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_PROVIDER_REJECTED,
       ET_K2_STATUS_INTERNAL},
  };
  aligned_error diagnostic;

  for (size_t index = 0u; index < sizeof(valid_pairs) / sizeof(valid_pairs[0]);
       index++) {
    initialize_error(&diagnostic, 0xa5u);
    CHECK(et_k2_test_copy_k1_result_v1(
              (int32_t)valid_pairs[index].category, valid_pairs[index].category,
              valid_pairs[index].code, 0, &diagnostic,
              264) == (int64_t)valid_pairs[index].expected_category);
    CHECK(diagnostic.error.category == valid_pairs[index].expected_category);
    CHECK(diagnostic.error.code == valid_pairs[index].code);
    CHECK(strcmp(diagnostic.error.operation, "capability-require") == 0);
    CHECK(strcmp(diagnostic.error.message, "test K1 diagnostic") == 0);
    CHECK(zero_after_nul(diagnostic.error.operation,
                         sizeof(diagnostic.error.operation)));
    CHECK(zero_after_nul(diagnostic.error.message,
                         sizeof(diagnostic.error.message)));
  }

  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_test_copy_k1_result_v1(0, ET_KERNEL_ERROR_NONE, ET_KERNEL_CODE_OK,
                                     0, &diagnostic,
                                     264) == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.category == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_INVARIANT);

  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_test_copy_k1_result_v1(
            ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_ERROR_UNSUPPORTED,
            ET_KERNEL_CODE_NULL_ARGUMENT, 0, &diagnostic,
            264) == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.category == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_INVARIANT);
  CHECK(strcmp(diagnostic.error.operation, "capability-require") == 0);
  CHECK(zero_after_nul(diagnostic.error.operation,
                       sizeof(diagnostic.error.operation)));
  CHECK(zero_after_nul(diagnostic.error.message,
                       sizeof(diagnostic.error.message)));

  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_test_copy_k1_result_v1(
            ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_ERROR_INVALID_ARGUMENT,
            ET_KERNEL_CODE_NULL_ARGUMENT, 0, &diagnostic,
            264) == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_INVARIANT);

  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_test_copy_k1_result_v1(
            ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_ERROR_UNSUPPORTED,
            ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, 1, &diagnostic,
            264) == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_INVARIANT);

  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_test_copy_k1_result_v1(
            ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_ERROR_UNSUPPORTED,
            ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, 2, &diagnostic,
            264) == ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_INVARIANT);
}

static void test_discovery_and_matching(void) {
  aligned_error diagnostic;
  uint64_t rank1[1];
  uint64_t rank2[2] = {1u, 1u};
  tagged_bytes operation_carrier;
  tagged_bytes dtype_carrier;
  tagged_bytes device_carrier;
  tagged_bytes shape_carrier;
  int64_t generation;

  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  CHECK(all_bytes(diagnostic.bytes, sizeof(diagnostic.bytes), 0u));
  CHECK(et_k2_test_discovery_count_v1() == 1u);
  CHECK(et_k2_test_runtime_live_count_v1() == 1u);
  CHECK(et_k2_private_runtime_pid_v1() == (int64_t)getpid());
  generation = et_k2_private_runtime_generation_v1();
  CHECK(generation > 0);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  CHECK(et_k2_test_discovery_count_v1() == 1u);

  register_factories();
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", NULL, 0, 0,
                        &diagnostic) == 0);
  CHECK(all_bytes(diagnostic.bytes, sizeof(diagnostic.bytes), 0u));
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", NULL, 0, 1,
                        &diagnostic) == 0);
  rank1[0] = 0u;
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", rank1, 1,
                        0, &diagnostic) == 0);
  rank1[0] = (uint64_t)(SIZE_MAX / sizeof(float));
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", rank1, 1,
                        1, &diagnostic) == 0);
  rank1[0]++;
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", rank1, 1,
                        1, &diagnostic) == ET_K2_STATUS_UNSUPPORTED);
  check_k1_no_match(&diagnostic);
  CHECK(require_request(generation, 9, "wrong", "f32", "cpu", NULL, 0, 1,
                        &diagnostic) == ET_K2_STATUS_UNSUPPORTED);
  check_k1_no_match(&diagnostic);
  CHECK(require_request(generation, 9, "storage.copy", "wrong", "cpu", NULL, 0,
                        1, &diagnostic) == ET_K2_STATUS_UNSUPPORTED);
  CHECK(require_request(generation, 9, "storage.copy", "f32", "wrong", NULL, 0,
                        1, &diagnostic) == ET_K2_STATUS_UNSUPPORTED);
  CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", rank2, 2,
                        1, &diagnostic) == ET_K2_STATUS_UNSUPPORTED);
  CHECK(require_request(generation, 0, "storage.copy", "f32", "cpu", NULL, 0, 1,
                        &diagnostic) == ET_K2_STATUS_UNSUPPORTED);

  initialize_bytes(&operation_carrier, "Bad", 3u);
  initialize_bytes(&dtype_carrier, "f32", 3u);
  initialize_bytes(&device_carrier, "cpu", 3u);
  initialize_bytes(&shape_carrier, NULL, 0u);
  CHECK(et_k2_private_runtime_require_v1(generation, 9, &operation_carrier, 3,
                                         &dtype_carrier, 3, &device_carrier, 3,
                                         &shape_carrier, 0, 0, 1, &diagnostic,
                                         264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.error.code == ET_K2_CODE_TRANSPORT);
  initialize_bytes(&operation_carrier, "storage.copy", 12u);
  CHECK(et_k2_private_runtime_require_v1(generation, 11, &operation_carrier, 12,
                                         &dtype_carrier, 3, &device_carrier, 3,
                                         &shape_carrier, 0, 0, 1, &diagnostic,
                                         264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.error.code == ET_K2_CODE_ENTRY_INDEX);
  CHECK(et_k2_private_runtime_require_v1(
            generation + 1, 9, &operation_carrier, 12, &dtype_carrier, 3,
            &device_carrier, 3, &shape_carrier, 0, 0, 1, &diagnostic,
            264) == ET_K2_STATUS_INVALID_STATE);
  CHECK(diagnostic.error.code == ET_K2_CODE_STALE);
  CHECK(et_k2_private_runtime_require_v1(generation, 9, &operation_carrier, 12,
                                         &dtype_carrier, 3, &device_carrier, 3,
                                         &shape_carrier, 8, 0, 1, &diagnostic,
                                         264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.error.code == ET_K2_CODE_TRANSPORT);
  CHECK(et_k2_private_runtime_require_v1(generation, 9, &operation_carrier, 12,
                                         &dtype_carrier, 3, &device_carrier, 3,
                                         &shape_carrier, 0, 0, 2, &diagnostic,
                                         264) == ET_K2_STATUS_INVALID_ARGUMENT);
  CHECK(diagnostic.error.code == ET_K2_CODE_TRANSPORT);

#ifdef ET_K2_ALLOCATOR_WRAP
  {
    const size_t calls_before = allocation_calls;
    const size_t requested_before = allocation_requested_bytes;
    allocation_fail_after = SIZE_MAX;
    allocation_tracking = 1;
#endif
    for (size_t query = 0; query < 8192u; query++) {
      CHECK(require_request(generation, 9, "storage.copy", "f32", "cpu", NULL,
                            0, (int64_t)(query & 1u), &diagnostic) == 0);
    }
#ifdef ET_K2_ALLOCATOR_WRAP
    allocation_end();
    CHECK(allocation_calls == calls_before);
    CHECK(allocation_requested_bytes == requested_before);
  }
#endif
  CHECK(et_k2_test_discovery_count_v1() == 1u);
  CHECK(et_k2_test_runtime_live_count_v1() == 1u);
}

static void test_provider_and_partial_failure(void) {
  aligned_error diagnostic;
  et_kernel_provider_v1 altered;
  int64_t old_generation;

  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  old_generation = et_k2_private_runtime_generation_v1();
  CHECK(old_generation == -7);
  et_k2_test_provider_override_v1(NULL, 1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) ==
        ET_K2_STATUS_UNSUPPORTED);
  CHECK(diagnostic.error.code == ET_K2_CODE_PROVIDER_ABSENT);
  CHECK(et_k2_test_runtime_live_count_v1() == 0u);

  altered = *et_f32_tensor_provider_v1();
  altered.abi_major++;
  et_k2_test_provider_override_v1(&altered, 1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) ==
        ET_K2_STATUS_VERSION_MISMATCH);
  CHECK(diagnostic.error.code == ET_K2_CODE_PROVIDER_AUDIT);

  altered = *et_f32_tensor_provider_v1();
  altered.name = "foreign-provider";
  et_k2_test_provider_override_v1(&altered, 1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) ==
        ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_PROVIDER_AUDIT);

  et_k2_test_provider_override_v1(NULL, 0);
  et_k2_test_fail_stage_v1(ET_K2_TEST_FAIL_AFTER_DISCOVER);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) ==
        ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_RUNTIME_AUDIT);
  CHECK(et_k2_test_discovery_count_v1() == 1u);
  CHECK(et_k2_test_destroy_count_v1() == 1u);
  CHECK(et_k2_test_runtime_live_count_v1() == 0u);
  CHECK(et_k2_private_runtime_generation_v1() == -7);
  et_k2_test_fail_stage_v1(ET_K2_TEST_FAIL_RUNTIME_AUDIT);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) ==
        ET_K2_STATUS_INTERNAL);
  CHECK(diagnostic.error.code == ET_K2_CODE_RUNTIME_AUDIT);
  CHECK(et_k2_test_discovery_count_v1() == 2u);
  CHECK(et_k2_test_destroy_count_v1() == 2u);
  CHECK(et_k2_test_runtime_live_count_v1() == 0u);
  et_k2_test_fail_stage_v1(ET_K2_TEST_FAIL_NONE);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  CHECK(et_k2_test_discovery_count_v1() == 3u);
  CHECK(et_k2_test_runtime_live_count_v1() == 1u);

  old_generation = et_k2_private_runtime_generation_v1();
  et_k2_test_runtime_drop_v1();
  CHECK(et_k2_private_runtime_generation_v1() == -7);
  CHECK(require_request(old_generation, 9, "storage.copy", "f32", "cpu", NULL,
                        0, 1, &diagnostic) == ET_K2_STATUS_INVALID_STATE);
  CHECK(diagnostic.error.code == ET_K2_CODE_STALE);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  CHECK(et_k2_private_runtime_generation_v1() == old_generation + 1);
}

static void expect_provider_rejected(const et_kernel_provider_v1 *provider,
                                     int64_t category) {
  aligned_error diagnostic;
  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  et_k2_test_provider_override_v1(provider, 1);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == category);
  CHECK(diagnostic.error.category == (uint32_t)category);
  CHECK(diagnostic.error.code == (provider == NULL
                                      ? ET_K2_CODE_PROVIDER_ABSENT
                                      : ET_K2_CODE_PROVIDER_AUDIT));
  CHECK(et_k2_test_discovery_count_v1() == 0u);
  CHECK(et_k2_test_runtime_live_count_v1() == 0u);
}

static void test_exact_provider_audit(void) {
  const et_kernel_provider_v1 *known = et_f32_tensor_provider_v1();
  const et_kernel_capability_v1 *known_entry =
      (const et_kernel_capability_v1 *)known->capabilities;
  et_kernel_provider_v1 provider;
  et_kernel_capability_v1 entry;
  et_kernel_shape_range_v1 ranges[2];
  et_kernel_dimension_range_v1 dimension;
  const char *operations[1] = {"storage.copy"};
  const char *dtypes[1] = {"f32"};
  const char *devices[1] = {"cpu"};

#define REJECT_PROVIDER(field, value, expected_category)                       \
  do {                                                                         \
    provider = *known;                                                         \
    provider.field = (value);                                                  \
    expect_provider_rejected(&provider, (expected_category));                  \
  } while (0)

  expect_provider_rejected(NULL, ET_K2_STATUS_UNSUPPORTED);
  REJECT_PROVIDER(struct_size, 0u, ET_K2_STATUS_VERSION_MISMATCH);
  REJECT_PROVIDER(abi_major, ET_KERNEL_ABI_MAJOR + 1u,
                  ET_K2_STATUS_VERSION_MISMATCH);
  REJECT_PROVIDER(abi_minor, ET_KERNEL_ABI_MINOR + 1u,
                  ET_K2_STATUS_VERSION_MISMATCH);
  REJECT_PROVIDER(required_features, UINT64_C(1),
                  ET_K2_STATUS_VERSION_MISMATCH);
  REJECT_PROVIDER(version, "2.0", ET_K2_STATUS_VERSION_MISMATCH);
  REJECT_PROVIDER(name, "foreign", ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(evidence, "foreign", ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(capability_count, 2u, ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(capability_stride, 0u, ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(capability_bytes, 0u, ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(capabilities, NULL, ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(validate_call, NULL, ET_K2_STATUS_INTERNAL);
  REJECT_PROVIDER(invoke_call, NULL, ET_K2_STATUS_INTERNAL);

  entry = *known_entry;
  entry.struct_size = 0u;
  provider = *known;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_VERSION_MISMATCH);

  entry = *known_entry;
  entry.reserved[0] = 1u;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);
  entry = *known_entry;
  entry.status = ET_KERNEL_CAPABILITY_UNVERIFIED;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);
  entry = *known_entry;
  entry.deterministic = 0u;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);

  entry = *known_entry;
  operations[0] = "foreign";
  entry.operations = operations;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);
  operations[0] = "storage.copy";
  entry = *known_entry;
  dtypes[0] = "f64";
  entry.dtypes = dtypes;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);
  dtypes[0] = "f32";
  entry = *known_entry;
  devices[0] = "gpu";
  entry.devices = devices;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);

  devices[0] = "cpu";
  entry = *known_entry;
  ranges[0] = known_entry->shape_ranges[0];
  ranges[1] = known_entry->shape_ranges[1];
  dimension = ranges[1].dimensions[0];
  dimension.maximum--;
  ranges[1].dimensions = &dimension;
  entry.shape_ranges = ranges;
  provider.capabilities = &entry;
  expect_provider_rejected(&provider, ET_K2_STATUS_INTERNAL);

#undef REJECT_PROVIDER
}

#ifdef ET_K2_ALLOCATOR_WRAP
static void test_every_k1_discovery_allocation(void) {
  /*
   * Known upstream K1 blocker (issue #73 integration comments 5628645123 and
   * 5628645216): fail_after==1 makes the baseline capability-table calloc
   * return NULL after capability_count was set to 11.  K1 then destroys the
   * partial runtime by iterating that count through a NULL table and crashes.
   * This exhaustive sweep is the regression gate for the authorized K1 fix;
   * fail_after==0 and 2..57 already cleaned up correctly on the broken head.
   */
  aligned_error diagnostic;
  size_t successful_fail_index = SIZE_MAX;
  size_t successful_requested = 0u;
  size_t successful_live_blocks = 0u;
  size_t successful_live_bytes = 0u;
  size_t successful_peak_bytes = 0u;

  for (size_t fail_after = 0u; fail_after < 128u; fail_after++) {
    int64_t result;
    et_k2_test_reset_v1();
    initialize_error(&diagnostic, 0xa5u);
    allocation_begin(fail_after);
    result = et_k2_private_runtime_ensure_v1(&diagnostic, 264);
    allocation_end();
    if (result == ET_K2_STATUS_OK) {
      successful_fail_index = fail_after;
      successful_requested = allocation_requested_bytes;
      successful_live_blocks = allocation_live_blocks;
      successful_live_bytes = allocation_live_bytes;
      successful_peak_bytes = allocation_peak_bytes;
      break;
    }
    CHECK(result == ET_K2_STATUS_INTERNAL);
    CHECK(diagnostic.error.category == ET_K2_STATUS_INTERNAL);
    CHECK(diagnostic.error.code == ET_KERNEL_CODE_ALLOCATION_FAILED);
    CHECK(et_k2_test_runtime_live_count_v1() == 0u);
    CHECK(allocation_live_blocks == 0u);
    CHECK(allocation_live_bytes == 0u);
    CHECK(et_k2_private_runtime_generation_v1() == -7);
  }
  CHECK(successful_fail_index != SIZE_MAX);
  CHECK(successful_fail_index == allocation_calls);
  CHECK(successful_fail_index == 58u);
  CHECK(successful_requested == 2487u);
  CHECK(successful_live_blocks == 54u);
  CHECK(successful_live_bytes == 2407u);
  CHECK(successful_peak_bytes == 2487u);
  (void)printf("K2 K1 allocation measurement: calls=%zu requested=%zu "
               "live-blocks=%zu live-bytes=%zu peak-bytes=%zu\n",
               successful_fail_index, successful_requested,
               successful_live_blocks, successful_live_bytes,
               successful_peak_bytes);
  et_k2_test_reset_v1();
  CHECK(allocation_live_blocks == 0u);
  CHECK(allocation_live_bytes == 0u);
}
#endif

typedef struct child_result {
  int64_t stale_generation;
  int64_t failed_ensure;
  uint32_t failed_code;
  int64_t child_generation;
  int64_t child_pid;
  uint64_t live_count;
} child_result;

static void test_serialized_fork(void) {
  aligned_error diagnostic;
  child_result child;
  int descriptors[2];
  pid_t pid;
  int status = 0;
  int64_t parent_generation;

  et_k2_test_reset_v1();
  initialize_error(&diagnostic, 0xa5u);
  CHECK(et_k2_private_runtime_ensure_v1(&diagnostic, 264) == 0);
  parent_generation = et_k2_private_runtime_generation_v1();
  CHECK(pipe(descriptors) == 0);
  pid = fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    ssize_t written;
    (void)close(descriptors[0]);
    memset(&child, 0, sizeof(child));
    child.stale_generation = et_k2_private_runtime_generation_v1();
    et_k2_test_fail_stage_v1(ET_K2_TEST_FAIL_BEFORE_DISCOVER);
    child.failed_ensure = et_k2_private_runtime_ensure_v1(&diagnostic, 264);
    child.failed_code = diagnostic.error.code;
    et_k2_test_fail_stage_v1(ET_K2_TEST_FAIL_NONE);
    if (et_k2_private_runtime_ensure_v1(&diagnostic, 264) != 0) {
      _exit(3);
    }
    child.child_generation = et_k2_private_runtime_generation_v1();
    child.child_pid = et_k2_private_runtime_pid_v1();
    child.live_count = et_k2_test_runtime_live_count_v1();
    written = write(descriptors[1], &child, sizeof(child));
    (void)close(descriptors[1]);
    _exit(written == (ssize_t)sizeof(child) ? 0 : 4);
  }
  if (pid > 0) {
    ssize_t received;
    (void)close(descriptors[1]);
    received = read(descriptors[0], &child, sizeof(child));
    (void)close(descriptors[0]);
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(received == (ssize_t)sizeof(child));
    CHECK(child.stale_generation == -7);
    CHECK(child.failed_ensure == ET_K2_STATUS_INTERNAL);
    CHECK(child.failed_code == ET_K2_CODE_PROVIDER_AUDIT);
    CHECK(child.child_generation == parent_generation + 1);
    CHECK(child.child_pid == (int64_t)pid);
    CHECK(child.live_count == 1u);
    CHECK(et_k2_private_runtime_generation_v1() == parent_generation);
    CHECK(et_k2_private_runtime_pid_v1() == (int64_t)getpid());
  }
}

int main(void) {
  test_factory_identity();
  test_diagnostic_admission();
  test_k1_category_code_pairs();
  test_discovery_and_matching();
  test_provider_and_partial_failure();
  test_exact_provider_audit();
#ifdef ET_K2_ALLOCATOR_WRAP
  test_every_k1_discovery_allocation();
#endif
  test_serialized_fork();
  et_k2_test_reset_v1();
  if (failures != 0) {
    (void)fprintf(stderr, "%d K2 native checks failed\n", failures);
    return EXIT_FAILURE;
  }
  (void)puts("K2 native capability checks passed");
  return EXIT_SUCCESS;
}
