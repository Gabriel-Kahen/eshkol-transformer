#include "eshkol_transformer/kernel_abi.h"

#include <stddef.h>
#include <stdint.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
static int checks;
static int resolver_calls;
static int validate_calls;
static int invoke_calls;
static int reject_mock_call;
static int malformed_mock_error;

#define CHECK(condition)                                                        \
  do {                                                                          \
    checks++;                                                                    \
    if (!(condition)) {                                                          \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);      \
      failures++;                                                                \
    }                                                                            \
  } while (0)

_Static_assert(ET_KERNEL_ABI_MAJOR == 1u, "K1 ABI major changed");
_Static_assert(ET_KERNEL_ABI_MINOR == 0u, "K1 ABI minor changed");
_Static_assert(sizeof(uint32_t) == 4u, "K1 requires 32-bit uint32_t");
_Static_assert(sizeof(uint64_t) == 8u, "K1 requires 64-bit uint64_t");
_Static_assert(offsetof(et_kernel_provider_v1, struct_size) == 0u,
               "provider struct_size must be first");
_Static_assert(offsetof(et_kernel_capability_v1, struct_size) == 0u,
               "capability struct_size must be first");
_Static_assert(offsetof(et_kernel_request_v1, struct_size) == 0u,
               "request struct_size must be first");
_Static_assert(offsetof(et_kernel_tensor_view_v1, struct_size) == 0u,
               "tensor struct_size must be first");
_Static_assert(offsetof(et_kernel_call_v1, struct_size) == 0u,
               "call struct_size must be first");
_Static_assert(sizeof(void *) == 8u && sizeof(size_t) == 8u,
               "K1 ABI v1 is defined for the supported x86-64 lane");
_Static_assert(sizeof(et_kernel_error) == 264u, "error ABI layout changed");
_Static_assert(sizeof(et_kernel_dimension_range_v1) == 24u,
               "dimension-range ABI layout changed");
_Static_assert(sizeof(et_kernel_shape_range_v1) == 16u,
               "shape-range ABI layout changed");
_Static_assert(sizeof(et_kernel_capability_v1) == 120u,
               "capability ABI layout changed");
_Static_assert(sizeof(et_kernel_tensor_view_v1) == 72u,
               "tensor-view ABI layout changed");
_Static_assert(sizeof(et_kernel_request_v1) == 56u,
               "request ABI layout changed");
_Static_assert(sizeof(et_kernel_call_v1) == 104u,
               "call ABI layout changed");
_Static_assert(sizeof(et_kernel_provider_v1) == 80u,
               "provider ABI layout changed");
_Static_assert(offsetof(et_kernel_error, message) == 72u,
               "error message offset changed");
_Static_assert(offsetof(et_kernel_capability_v1, shape_ranges) == 112u,
               "capability shape-range offset changed");
_Static_assert(offsetof(et_kernel_tensor_view_v1, shape) == 64u,
               "tensor shape offset changed");
_Static_assert(offsetof(et_kernel_request_v1, deterministic) == 48u,
               "request deterministic offset changed");
_Static_assert(offsetof(et_kernel_call_v1, outputs) == 96u,
               "call outputs offset changed");
_Static_assert(offsetof(et_kernel_provider_v1, invoke_call) == 72u,
               "provider callback offset changed");

typedef struct test_provider_v1_1 {
  et_kernel_provider_v1 prefix;
  uint64_t ignored_minor_tail;
} test_provider_v1_1;

static const et_kernel_dimension_range_v1 mock_dimensions[] = {
    {.minimum = 1u, .maximum = 8u, .maximum_unbounded = 0u, .reserved = {0}},
};
static const et_kernel_shape_range_v1 mock_ranges[] = {
    {.rank = 1u, .dimensions = mock_dimensions},
};
static char mock_operation[] = "abi-test";
static const char *const mock_operations[] = {"z-test", mock_operation};
static const char *const mock_dtypes[] = {"f32", "bool"};
static const char *const mock_devices[] = {"test-cpu"};
static const et_kernel_capability_v1 mock_capability = {
    .struct_size = sizeof(et_kernel_capability_v1),
    .name = "test.mock-dispatch",
    .status = ET_KERNEL_CAPABILITY_VERIFIED,
    .implementation = "test-only-mock",
    .version = "1.1-test",
    .evidence = "TEST-ONLY:ABI-CONFORMANCE-NOT-CAPABILITY-EVIDENCE",
    .deterministic = 1u,
    .reserved = {0},
    .operation_count = 2u,
    .operations = mock_operations,
    .dtype_count = 2u,
    .dtypes = mock_dtypes,
    .device_count = 1u,
    .devices = mock_devices,
    .shape_range_count = 1u,
    .shape_ranges = mock_ranges,
};

static int32_t mock_validate(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  (void)call;
  validate_calls++;
  if (!reject_mock_call) {
    return 0;
  }
  et_kernel_error_clear(error);
  if (error != NULL) {
    error->category = ET_KERNEL_ERROR_SHAPE_MISMATCH;
    error->code = ET_KERNEL_CODE_PROVIDER_REJECTED;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   "test.mock-dispatch");
    (void)snprintf(error->message, sizeof(error->message), "%s",
                   "test-only provider rejection before dispatch");
    if (malformed_mock_error == 2) {
      error->category = UINT32_C(99);
    } else if (malformed_mock_error == 3) {
      error->message[0] = '\0';
    }
  }
  return malformed_mock_error == 1 ? 99 : ET_KERNEL_ERROR_SHAPE_MISMATCH;
}

static void mock_invoke(const et_kernel_call_v1 *call) {
  invoke_calls++;
  for (size_t index = 0; index < call->output_count; index++) {
    memset(call->outputs[index].data, 0x5a, call->outputs[index].byte_length);
  }
}

static et_kernel_provider_v1 mock_provider = {
    .struct_size = sizeof(et_kernel_provider_v1),
    .abi_major = ET_KERNEL_ABI_MAJOR,
    .abi_minor = 1u,
    .required_features = 0u,
    .name = "test-only-provider",
    .version = "1.1-test",
    .evidence = "TEST-ONLY:NEVER-INSTALLED-OR-DISCOVERED-BY-DEFAULT",
    .capability_count = 1u,
    .capabilities = &mock_capability,
    .validate_call = mock_validate,
    .invoke_call = mock_invoke,
};

static const et_kernel_provider_v1 *resolve_provider(void *context,
                                                     const char *symbol) {
  resolver_calls++;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return (const et_kernel_provider_v1 *)context;
}

static const et_kernel_provider_v1 *resolve_missing(void *context,
                                                    const char *symbol) {
  (void)context;
  resolver_calls++;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return NULL;
}

static void expect_error(int32_t result, const et_kernel_error *error,
                         et_kernel_error_category category,
                         et_kernel_error_code code) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
}

static char *report_json(const et_kernel_runtime *runtime, size_t *length) {
  et_kernel_error error;
  size_t required = 0;
  char *buffer;
  CHECK(et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) ==
        0);
  CHECK(required > 2u);
  buffer = (char *)malloc(required);
  CHECK(buffer != NULL);
  if (buffer == NULL) {
    return NULL;
  }
  memset(buffer, 0xa5, required);
  CHECK(et_kernel_runtime_report_json(runtime, buffer, required, &required,
                                      &error) == 0);
  CHECK(buffer[required - 2u] == '\n');
  CHECK(buffer[required - 1u] == '\0');
  if (length != NULL) {
    *length = required - 1u;
  }
  return buffer;
}

static et_kernel_request_v1 mock_request(uint64_t *shape) {
  et_kernel_request_v1 request = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = "abi-test",
      .dtype = "f32",
      .device = "test-cpu",
      .rank = 1u,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  return request;
}

static void test_version_and_baseline(void) {
  static const char *const expected_names[] = {
      "autodiff.reverse",
      "kernel.activation",
      "kernel.causal-attention",
      "kernel.embedding-backward",
      "kernel.indexed-cross-entropy",
      "kernel.matmul",
      "kernel.norm",
      "tensor.bool",
      "tensor.contiguous",
      "tensor.f32",
      "tensor.i64",
  };
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  uint64_t shape[] = {2u, 2u};
  et_kernel_request_v1 request = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = "matmul",
      .dtype = "f32",
      .device = "cpu",
      .rank = 2u,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  char *json;

  CHECK(et_kernel_abi_major() == 1);
  CHECK(et_kernel_abi_minor() == 0);
  CHECK(strcmp(et_kernel_provider_symbol(),
               "eshkol_transformer_kernel_provider_v1") == 0);
  CHECK(et_kernel_runtime_baseline(&runtime, &error) == 0);
  CHECK(runtime != NULL);
  CHECK(et_kernel_runtime_capability_count(runtime) ==
        sizeof(expected_names) / sizeof(expected_names[0]));
  for (size_t index = 0; index < sizeof(expected_names) / sizeof(expected_names[0]);
       index++) {
    const et_kernel_capability_v1 *entry =
        et_kernel_runtime_capability_at(runtime, index);
    CHECK(entry != NULL);
    CHECK(strcmp(entry->name, expected_names[index]) == 0);
    CHECK(entry->status == ET_KERNEL_CAPABILITY_UNVERIFIED);
    CHECK(entry->operation_count == 0u);
    CHECK(entry->dtype_count == 0u);
    CHECK(entry->device_count == 0u);
    CHECK(entry->shape_range_count == 0u);
  }
  CHECK(et_kernel_runtime_capability_at(runtime, 1000u) == NULL);
  CHECK(et_kernel_runtime_capability_find(runtime, "device.accelerator") ==
        NULL);
  expect_error(et_kernel_runtime_capability_require(
                   runtime, "kernel.matmul", &request, NULL, &error),
               &error, ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.dtype = "f16";
  expect_error(et_kernel_runtime_capability_require(
                   runtime, "kernel.matmul", &request, NULL, &error),
               &error, ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.dtype = "bf16";
  request.device = "gpu:0";
  expect_error(et_kernel_runtime_capability_require(
                   runtime, "kernel.matmul", &request, NULL, &error),
               &error, ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.operation = "unknown-operation";
  expect_error(et_kernel_runtime_capability_require(
                   runtime, "kernel.matmul", &request, NULL, &error),
               &error, ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);

  json = report_json(runtime, NULL);
  CHECK(json != NULL);
  if (json != NULL) {
    CHECK(strstr(json, "\"provider_abi\":null") != NULL);
    CHECK(strstr(json, "\"status\":\"unsupported\"") == NULL);
    CHECK(strstr(json, "\"status\":\"verified\"") == NULL);
    free(json);
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_discovery_versions_and_descriptors(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  et_kernel_provider_v1 provider;
  test_provider_v1_1 extended_provider;
  et_kernel_capability_v1 capability;
  const char *const duplicate_operations[] = {"abi-test", "abi-test"};
  et_kernel_capability_v1 duplicates[2];
  et_kernel_capability_v1 malformed_pair[2];

  resolver_calls = 0;
  expect_error(et_kernel_runtime_discover(resolve_missing, NULL, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_SYMBOL_MISSING);
  CHECK(runtime == NULL);
  CHECK(resolver_calls == 1);

  provider = mock_provider;
  provider.abi_major = 2u;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_ABI_MAJOR_MISMATCH);
  CHECK(runtime == NULL);

  provider = mock_provider;
  provider.struct_size = offsetof(et_kernel_provider_v1, invoke_call);
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_INVALID_STRUCT_SIZE);

  provider = mock_provider;
  provider.required_features = UINT64_C(1);
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE);

  provider = mock_provider;
  capability = mock_capability;
  capability.struct_size = offsetof(et_kernel_capability_v1, shape_ranges);
  provider.capabilities = &capability;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_INVALID_STRUCT_SIZE);

  provider = mock_provider;
  capability = mock_capability;
  capability.operation_count = 2u;
  capability.operations = duplicate_operations;
  provider.capabilities = &capability;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_DUPLICATE_ENTRY);

  provider = mock_provider;
  malformed_pair[0] = mock_capability;
  malformed_pair[1] = mock_capability;
  malformed_pair[1].name = NULL;
  provider.capability_count = 2u;
  provider.capabilities = malformed_pair;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_INVALID_TEXT);

  provider = mock_provider;
  duplicates[0] = mock_capability;
  duplicates[1] = mock_capability;
  provider.capability_count = 2u;
  provider.capabilities = duplicates;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_DUPLICATE_ENTRY);

  provider = mock_provider;
  provider.validate_call = NULL;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_NULL_ARGUMENT);

  provider = mock_provider;
  provider.abi_minor = 1u;
  CHECK(et_kernel_runtime_discover(resolve_provider, &provider, &runtime, &error) ==
        0);
  CHECK(runtime != NULL);
  CHECK(et_kernel_runtime_capability_count(runtime) == 12u);
  CHECK(et_kernel_runtime_capability_find(runtime, "kernel.matmul")->status ==
        ET_KERNEL_CAPABILITY_UNVERIFIED);
  CHECK(et_kernel_runtime_capability_find(runtime, "test.mock-dispatch")->status ==
        ET_KERNEL_CAPABILITY_VERIFIED);
  et_kernel_runtime_destroy(runtime);
  runtime = NULL;

  extended_provider.prefix = mock_provider;
  extended_provider.prefix.struct_size = sizeof(extended_provider);
  extended_provider.prefix.abi_minor = 1u;
  extended_provider.ignored_minor_tail = UINT64_C(0xfeedfacecafebeef);
  CHECK(et_kernel_runtime_discover(resolve_provider, &extended_provider.prefix,
                                   &runtime, &error) == 0);
  CHECK(runtime != NULL);
  CHECK(extended_provider.ignored_minor_tail == UINT64_C(0xfeedfacecafebeef));
  et_kernel_runtime_destroy(runtime);
  runtime = NULL;

  provider = mock_provider;
  provider.capability_count = ET_KERNEL_MAX_CAPABILITIES + 1u;
  expect_error(et_kernel_runtime_discover(resolve_provider, &provider, &runtime,
                                          &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_INTEGER_OVERFLOW);

  provider = mock_provider;
  capability = mock_capability;
  capability.deterministic = 0u;
  provider.capabilities = &capability;
  CHECK(et_kernel_runtime_discover(resolve_provider, &provider, &runtime, &error) ==
        0);
  {
    uint64_t shape[] = {1u};
    et_kernel_request_v1 request = mock_request(shape);
    expect_error(et_kernel_runtime_capability_require(
                     runtime, "test.mock-dispatch", &request, NULL, &error),
                 &error, ET_KERNEL_ERROR_UNSUPPORTED,
                 ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    request.deterministic = 0u;
    CHECK(et_kernel_runtime_capability_require(
              runtime, "test.mock-dispatch", &request, NULL, &error) == 0);
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_deep_copy_and_canonicalization(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_runtime *second_runtime = NULL;
  et_kernel_error error;
  et_kernel_provider_v1 provider = mock_provider;
  et_kernel_capability_v1 reversed_capability = mock_capability;
  const char *const reversed_operations[] = {"abi-test", "z-test"};
  const char *const reversed_dtypes[] = {"bool", "f32"};
  char *first;
  char *second;
  size_t first_length = 0;
  size_t second_length = 0;

  CHECK(et_kernel_runtime_discover(resolve_provider, &provider, &runtime, &error) ==
        0);
  mock_operation[0] = 'x';
  CHECK(strcmp(et_kernel_runtime_capability_find(runtime, "test.mock-dispatch")
                   ->operations[0],
               "abi-test") == 0);
  mock_operation[0] = 'a';
  first = report_json(runtime, &first_length);

  reversed_capability.operations = reversed_operations;
  reversed_capability.dtypes = reversed_dtypes;
  provider.capabilities = &reversed_capability;
  CHECK(et_kernel_runtime_discover(resolve_provider, &provider, &second_runtime,
                                   &error) == 0);
  second = report_json(second_runtime, &second_length);
  CHECK(first != NULL && second != NULL);
  if (first != NULL && second != NULL) {
    CHECK(first_length == second_length);
    CHECK(memcmp(first, second, first_length) == 0);
    CHECK(strstr(first, "TEST-ONLY:ABI-CONFORMANCE-NOT-CAPABILITY-EVIDENCE") !=
          NULL);
    CHECK(strstr(first, "\"provider_abi\":{\"major\":1,\"minor\":1}") !=
          NULL);
  }
  free(first);
  free(second);
  et_kernel_runtime_destroy(runtime);
  et_kernel_runtime_destroy(second_runtime);
}

static void test_provider_json_canonical_edges(void) {
  static const et_kernel_dimension_range_v1 rank_one_dimensions[] = {
      {.minimum = 0u,
       .maximum = UINT64_MAX,
       .maximum_unbounded = 0u,
       .reserved = {0}},
  };
  static const et_kernel_dimension_range_v1 rank_two_dimensions[] = {
      {.minimum = 0u,
       .maximum = 0u,
       .maximum_unbounded = 1u,
       .reserved = {0}},
      {.minimum = 1u, .maximum = 8u, .maximum_unbounded = 0u, .reserved = {0}},
  };
  static const et_kernel_shape_range_v1 unsorted_ranges[] = {
      {.rank = 2u, .dimensions = rank_two_dimensions},
      {.rank = 1u, .dimensions = rank_one_dimensions},
  };
  static const char *const unsorted_symbols[] = {"z-value", "a-value"};
  const et_kernel_capability_v1 capability = {
      .struct_size = sizeof(et_kernel_capability_v1),
      .name = "test.json",
      .status = ET_KERNEL_CAPABILITY_UNVERIFIED,
      .implementation = "test-only-json",
      .version = "v\n\t\"\\\xc3\xa9",
      .evidence = "TEST-ONLY\n\t\"\\\xc3\xa9",
      .deterministic = 0u,
      .reserved = {0},
      .operation_count = 2u,
      .operations = unsorted_symbols,
      .dtype_count = 2u,
      .dtypes = unsorted_symbols,
      .device_count = 2u,
      .devices = unsorted_symbols,
      .shape_range_count = 2u,
      .shape_ranges = unsorted_ranges,
  };
  const et_kernel_provider_v1 provider = {
      .struct_size = sizeof(et_kernel_provider_v1),
      .abi_major = 1u,
      .abi_minor = 1u,
      .required_features = 0u,
      .name = "test-json-provider",
      .version = "1.1-test",
      .evidence = "TEST-ONLY:JSON-CANONICALIZATION",
      .capability_count = 1u,
      .capabilities = &capability,
      .validate_call = NULL,
      .invoke_call = NULL,
  };
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  char *c_report;
  char *utf8_report;
  size_t c_length = 0;
  size_t utf8_length = 0;

  CHECK(et_kernel_runtime_discover(resolve_provider, (void *)&provider, &runtime,
                                   &error) == 0);
  CHECK(setlocale(LC_ALL, "C") != NULL);
  c_report = report_json(runtime, &c_length);
  CHECK(setlocale(LC_ALL, "C.UTF-8") != NULL);
  utf8_report = report_json(runtime, &utf8_length);
  CHECK(c_report != NULL && utf8_report != NULL);
  if (c_report != NULL && utf8_report != NULL) {
    CHECK(c_length == utf8_length);
    CHECK(memcmp(c_report, utf8_report, c_length) == 0);
    CHECK(strstr(c_report,
                 "\"devices\":[\"a-value\",\"z-value\"]") != NULL);
    CHECK(strstr(c_report,
                 "\"shape_ranges\":[[[0,18446744073709551615]],"
                 "[[0,null],[1,8]]]") != NULL);
    CHECK(strstr(c_report,
                 "\"evidence\":\"TEST-ONLY\\u000a\\u0009\\\"\\\\"
                 "\xc3\xa9\"") != NULL);
    CHECK(strstr(c_report,
                 "\"version\":\"v\\u000a\\u0009\\\"\\\\\xc3\xa9\"") !=
          NULL);
  }
  (void)setlocale(LC_ALL, "C");
  free(c_report);
  free(utf8_report);
  et_kernel_runtime_destroy(runtime);
}

static void test_report_buffer_atomicity(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  char buffer[16];
  size_t required = 0;
  CHECK(et_kernel_runtime_baseline(&runtime, &error) == 0);
  memset(buffer, 0x33, sizeof(buffer));
  expect_error(et_kernel_runtime_report_json(runtime, buffer, sizeof(buffer),
                                             &required, &error),
               &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_BUFFER_TOO_SMALL);
  CHECK(required > sizeof(buffer));
  for (size_t index = 0; index < sizeof(buffer); index++) {
    CHECK((unsigned char)buffer[index] == 0x33u);
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_requests_and_dispatch(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  uint64_t shape[] = {1u};
  uint64_t bad_shape[] = {9u};
  uint32_t input_bits = UINT32_C(0x3f800000);
  uint32_t output_bits = UINT32_C(0x11223344);
  et_kernel_tensor_view_v1 input = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = &input_bits,
      .byte_length = sizeof(input_bits),
      .dtype = "f32",
      .device = "test-cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = 1u,
      .shape = shape,
  };
  et_kernel_tensor_view_v1 output = input;
  et_kernel_request_v1 request = mock_request(shape);
  et_kernel_call_v1 call = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = "test.mock-dispatch",
      .request = request,
      .input_count = 1u,
      .inputs = &input,
      .output_count = 1u,
      .outputs = &output,
  };

  output.data = &output_bits;
  CHECK(et_kernel_runtime_discover(resolve_provider, &mock_provider, &runtime,
                                   &error) == 0);
  validate_calls = 0;
  invoke_calls = 0;
  reject_mock_call = 0;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) == 0);
  CHECK(validate_calls == 1);
  CHECK(invoke_calls == 1);
  CHECK(output_bits == UINT32_C(0x5a5a5a5a));
  CHECK(input_bits == UINT32_C(0x3f800000));

  call.request.operation = "missing-operation";
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  call.request.operation = "abi-test";
  call.request.dtype = "bf16";
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  call.request.dtype = "f32";
  call.request.device = "gpu:0";
  input.device = "gpu:0";
  output.device = "gpu:0";
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  call.request.device = "test-cpu";
  input.device = "test-cpu";
  output.device = "test-cpu";

  output_bits = UINT32_C(0x11223344);
  reject_mock_call = 1;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_SHAPE_MISMATCH,
               ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(validate_calls == 2);
  CHECK(invoke_calls == 1);
  CHECK(output_bits == UINT32_C(0x11223344));
  reject_mock_call = 0;

  for (malformed_mock_error = 1; malformed_mock_error <= 3;
       malformed_mock_error++) {
    output_bits = UINT32_C(0x11223344);
    reject_mock_call = 1;
    expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
                 ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_PROVIDER_REJECTED);
    CHECK(output_bits == UINT32_C(0x11223344));
    CHECK(invoke_calls == 1);
  }
  malformed_mock_error = 0;
  output_bits = UINT32_C(0x11223344);
  CHECK(et_kernel_runtime_dispatch(runtime, &call, NULL) ==
        ET_KERNEL_ERROR_INTERNAL);
  CHECK(output_bits == UINT32_C(0x11223344));
  reject_mock_call = 0;

  request = mock_request(bad_shape);
  call.request = request;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_UNSUPPORTED,
               ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  CHECK(invoke_calls == 1);

  call.request = mock_request(shape);
  output.data = input.data;
  output_bits = UINT32_C(0x11223344);
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(input_bits == UINT32_C(0x3f800000));
  CHECK(invoke_calls == 1);
  output.data = &output_bits;

  input.layout = (et_kernel_layout_v1)0;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_NONCONTIGUOUS,
               ET_KERNEL_CODE_INVALID_BUFFER);
  input.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  input.offset_bytes = 1u;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_NONCONTIGUOUS,
               ET_KERNEL_CODE_INVALID_BUFFER);
  input.offset_bytes = 0u;

  input.byte_length = 3u;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_SHAPE_MISMATCH,
               ET_KERNEL_CODE_INVALID_BUFFER);
  input.byte_length = sizeof(input_bits);
  input.device = "cpu";
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_DEVICE_MISMATCH,
               ET_KERNEL_CODE_INVALID_BUFFER);
  input.device = "test-cpu";
  input.dtype = "f64";
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_DTYPE_MISMATCH,
               ET_KERNEL_CODE_INVALID_TEXT);
  input.dtype = "f32";

  input.struct_size = offsetof(et_kernel_tensor_view_v1, shape);
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  input.struct_size = sizeof(input);

  input.data = NULL;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_SHAPE_MISMATCH,
               ET_KERNEL_CODE_INVALID_BUFFER);
  input.data = &input_bits;

  {
    uint64_t overflow_shape[] = {UINT64_MAX};
    input.shape = overflow_shape;
    expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
                 ET_KERNEL_ERROR_SHAPE_MISMATCH,
                 ET_KERNEL_CODE_INTEGER_OVERFLOW);
    input.shape = shape;
  }

  {
    unsigned char alias_storage[8] = {0};
    input.data = alias_storage;
    output.data = alias_storage + 2;
    expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT,
                 ET_KERNEL_CODE_ALIASING_OUTPUT);
    input.data = &input_bits;
    output.data = &output_bits;
  }

  {
    uint32_t second_output_bits = 0u;
    et_kernel_tensor_view_v1 outputs[2] = {output, output};
    outputs[0].data = &second_output_bits;
    outputs[1].data = ((unsigned char *)&second_output_bits) + 1;
    call.output_count = 2u;
    call.outputs = outputs;
    expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
                 ET_KERNEL_ERROR_INVALID_ARGUMENT,
                 ET_KERNEL_CODE_ALIASING_OUTPUT);
    call.output_count = 1u;
    call.outputs = &output;
  }

  call.input_count = ET_KERNEL_MAX_TENSORS + 1u;
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_INVALID_ARGUMENT,
               ET_KERNEL_CODE_INTEGER_OVERFLOW);
  call.input_count = 1u;

  call.struct_size = offsetof(et_kernel_call_v1, outputs);
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
  call.struct_size = sizeof(call);
  call.request.struct_size = offsetof(et_kernel_request_v1, reserved);
  expect_error(et_kernel_runtime_dispatch(runtime, &call, &error), &error,
               ET_KERNEL_ERROR_VERSION_MISMATCH,
               ET_KERNEL_CODE_INVALID_STRUCT_SIZE);

  CHECK(et_kernel_runtime_capability_require(
            runtime, "test.mock-dispatch", &call.request, NULL, NULL) ==
        ET_KERNEL_ERROR_VERSION_MISMATCH);

  et_kernel_runtime_destroy(runtime);
}

int main(void) {
  test_version_and_baseline();
  test_discovery_versions_and_descriptors();
  test_deep_copy_and_canonicalization();
  test_provider_json_canonical_edges();
  test_report_buffer_atomicity();
  test_requests_and_dispatch();
  if (failures != 0) {
    fprintf(stderr, "K1 FAIL: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  printf("K1 PASS: %d ABI, discovery, report, and malformed-call checks\n",
         checks);
  return 0;
}
