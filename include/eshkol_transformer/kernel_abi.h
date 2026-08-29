#ifndef ESHKOL_TRANSFORMER_KERNEL_ABI_H
#define ESHKOL_TRANSFORMER_KERNEL_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_KERNEL_ABI_MAJOR 1u
#define ET_KERNEL_ABI_MINOR 0u
#define ET_KERNEL_PROVIDER_SYMBOL_V1 "eshkol_transformer_kernel_provider_v1"
#define ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES UINT64_C(0)
#define ET_KERNEL_ERROR_OPERATION_CAPACITY 64u
#define ET_KERNEL_ERROR_MESSAGE_CAPACITY 192u
#define ET_KERNEL_MAX_CAPABILITIES 4096u
#define ET_KERNEL_MAX_LIST_ENTRIES 4096u
#define ET_KERNEL_MAX_SHAPE_RANGES 4096u
#define ET_KERNEL_MAX_RANK 64u
#define ET_KERNEL_MAX_TENSORS 1024u
#define ET_KERNEL_MAX_SYMBOL_BYTES 127u
#define ET_KERNEL_MAX_METADATA_BYTES 1024u
#define ET_KERNEL_CAPABILITY_V1_0_SIZE ((size_t)120u)
#define ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ((size_t)72u)
#define ET_KERNEL_REQUEST_V1_0_SIZE ((size_t)56u)
#define ET_KERNEL_CALL_V1_0_SIZE ((size_t)88u)
#define ET_KERNEL_PROVIDER_V1_0_SIZE ((size_t)96u)

enum {
  ET_KERNEL_CAPABILITY_UNVERIFIED = 0,
  ET_KERNEL_CAPABILITY_VERIFIED = 1,
  ET_KERNEL_CAPABILITY_UNSUPPORTED = 2
};
typedef uint32_t et_kernel_capability_status;

enum {
  ET_KERNEL_ERROR_NONE = 0,
  ET_KERNEL_ERROR_INVALID_ARGUMENT,
  ET_KERNEL_ERROR_SHAPE_MISMATCH,
  ET_KERNEL_ERROR_DTYPE_MISMATCH,
  ET_KERNEL_ERROR_DEVICE_MISMATCH,
  ET_KERNEL_ERROR_NONCONTIGUOUS,
  ET_KERNEL_ERROR_UNSUPPORTED,
  ET_KERNEL_ERROR_VERSION_MISMATCH,
  ET_KERNEL_ERROR_INTERNAL
};
typedef uint32_t et_kernel_error_category;

enum {
  ET_KERNEL_CODE_OK = 0,
  ET_KERNEL_CODE_NULL_ARGUMENT,
  ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
  ET_KERNEL_CODE_INVALID_TEXT,
  ET_KERNEL_CODE_INVALID_ENUM,
  ET_KERNEL_CODE_DUPLICATE_ENTRY,
  ET_KERNEL_CODE_SYMBOL_MISSING,
  ET_KERNEL_CODE_ABI_MAJOR_MISMATCH,
  ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE,
  ET_KERNEL_CODE_ALLOCATION_FAILED,
  ET_KERNEL_CODE_BUFFER_TOO_SMALL,
  ET_KERNEL_CODE_INTEGER_OVERFLOW,
  ET_KERNEL_CODE_INVALID_SHAPE,
  ET_KERNEL_CODE_INVALID_BUFFER,
  ET_KERNEL_CODE_ALIASING_OUTPUT,
  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED,
  ET_KERNEL_CODE_PROVIDER_REJECTED
};
typedef uint32_t et_kernel_error_code;

typedef struct et_kernel_error {
  et_kernel_error_category category;
  et_kernel_error_code code;
  char operation[ET_KERNEL_ERROR_OPERATION_CAPACITY];
  char message[ET_KERNEL_ERROR_MESSAGE_CAPACITY];
} et_kernel_error;

typedef struct et_kernel_dimension_range_v1 {
  uint64_t minimum;
  uint64_t maximum;
  uint8_t maximum_unbounded;
  uint8_t reserved[7];
} et_kernel_dimension_range_v1;

typedef struct et_kernel_shape_range_v1 {
  size_t rank;
  const et_kernel_dimension_range_v1 *dimensions;
} et_kernel_shape_range_v1;

typedef struct et_kernel_capability_v1 {
  size_t struct_size;
  const char *name;
  et_kernel_capability_status status;
  const char *implementation;
  const char *version;
  const char *evidence;
  uint8_t deterministic;
  uint8_t reserved[7];
  size_t operation_count;
  const char *const *operations;
  size_t dtype_count;
  const char *const *dtypes;
  size_t device_count;
  const char *const *devices;
  size_t shape_range_count;
  const et_kernel_shape_range_v1 *shape_ranges;
} et_kernel_capability_v1;

enum {
  ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR = 1
};
typedef uint32_t et_kernel_layout_v1;

typedef struct et_kernel_tensor_view_v1 {
  size_t struct_size;
  void *data;
  size_t byte_length;
  const char *dtype;
  const char *device;
  et_kernel_layout_v1 layout;
  uint64_t offset_bytes;
  size_t rank;
  const uint64_t *shape;
} et_kernel_tensor_view_v1;

typedef struct et_kernel_request_v1 {
  size_t struct_size;
  const char *operation;
  const char *dtype;
  const char *device;
  size_t rank;
  const uint64_t *shape;
  uint8_t deterministic;
  uint8_t reserved[7];
} et_kernel_request_v1;

typedef struct et_kernel_call_v1 {
  size_t struct_size;
  const char *capability;
  const et_kernel_request_v1 *request;
  size_t input_count;
  size_t input_stride;
  size_t input_bytes;
  const void *inputs;
  size_t output_count;
  size_t output_stride;
  size_t output_bytes;
  void *outputs;
} et_kernel_call_v1;

typedef int32_t (*et_kernel_validate_call_v1)(const et_kernel_call_v1 *call,
                                              et_kernel_error *error);
typedef void (*et_kernel_invoke_call_v1)(const et_kernel_call_v1 *call);

typedef struct et_kernel_provider_v1 {
  size_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  uint64_t required_features;
  const char *name;
  const char *version;
  const char *evidence;
  size_t capability_count;
  size_t capability_stride;
  size_t capability_bytes;
  const void *capabilities;
  et_kernel_validate_call_v1 validate_call;
  et_kernel_invoke_call_v1 invoke_call;
} et_kernel_provider_v1;

#if defined(__cplusplus)
#define ET_KERNEL_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ET_KERNEL_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif
ET_KERNEL_STATIC_ASSERT(sizeof(void *) == 8u && sizeof(size_t) == 8u,
                        "kernel ABI v1 requires 64-bit pointers and size_t");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_error) == 264u &&
                            offsetof(et_kernel_error, message) == 72u,
                        "kernel error layout changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_dimension_range_v1) == 24u &&
                            offsetof(et_kernel_dimension_range_v1,
                                     maximum_unbounded) == 16u,
                        "dimension-range layout changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_shape_range_v1) == 16u &&
                            offsetof(et_kernel_shape_range_v1, dimensions) ==
                                8u,
                        "shape-range layout changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_capability_v1) ==
                            ET_KERNEL_CAPABILITY_V1_0_SIZE,
                        "capability v1.0 prefix layout changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_capability_v1, shape_ranges) ==
                            112u,
                        "capability shape-range offset changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_tensor_view_v1) ==
                            ET_KERNEL_TENSOR_VIEW_V1_0_SIZE,
                        "tensor-view v1.0 prefix layout changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_tensor_view_v1, shape) == 64u,
                        "tensor-view shape offset changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_request_v1) ==
                            ET_KERNEL_REQUEST_V1_0_SIZE,
                        "request v1.0 prefix layout changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_request_v1, deterministic) == 48u,
                        "request deterministic offset changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_call_v1) == ET_KERNEL_CALL_V1_0_SIZE,
                        "call v1.0 prefix layout changed");
ET_KERNEL_STATIC_ASSERT(sizeof(et_kernel_provider_v1) ==
                            ET_KERNEL_PROVIDER_V1_0_SIZE,
                        "provider v1.0 prefix layout changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_provider_v1, capability_stride) ==
                            56u,
                        "provider capability stride offset changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_provider_v1, capability_bytes) ==
                            64u &&
                            offsetof(et_kernel_provider_v1, capabilities) ==
                                72u &&
                            offsetof(et_kernel_provider_v1, validate_call) ==
                                80u,
                        "provider capability table offsets changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_provider_v1, invoke_call) == 88u,
                        "provider invoke callback offset changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_call_v1, request) == 16u,
                        "call request pointer offset changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_call_v1, input_count) == 24u &&
                            offsetof(et_kernel_call_v1, input_stride) == 32u &&
                            offsetof(et_kernel_call_v1, input_bytes) == 40u &&
                            offsetof(et_kernel_call_v1, inputs) == 48u &&
                            offsetof(et_kernel_call_v1, output_count) == 56u &&
                            offsetof(et_kernel_call_v1, output_stride) == 64u &&
                            offsetof(et_kernel_call_v1, output_bytes) == 72u,
                        "call tensor table offsets changed");
ET_KERNEL_STATIC_ASSERT(offsetof(et_kernel_call_v1, outputs) == 80u,
                        "call output table offset changed");
#undef ET_KERNEL_STATIC_ASSERT

typedef const et_kernel_provider_v1 *(*et_kernel_provider_resolver_v1)(
    void *context, const char *symbol_name);

typedef struct et_kernel_runtime et_kernel_runtime;

int32_t et_kernel_abi_major(void);
int32_t et_kernel_abi_minor(void);
const char *et_kernel_provider_symbol(void);

void et_kernel_error_clear(et_kernel_error *error);

int32_t et_kernel_runtime_baseline(et_kernel_runtime **runtime,
                                   et_kernel_error *error);
int32_t et_kernel_runtime_discover(et_kernel_provider_resolver_v1 resolver,
                                   void *context,
                                   et_kernel_runtime **runtime,
                                   et_kernel_error *error);
void et_kernel_runtime_destroy(et_kernel_runtime *runtime);

size_t et_kernel_runtime_capability_count(const et_kernel_runtime *runtime);
const et_kernel_capability_v1 *et_kernel_runtime_capability_at(
    const et_kernel_runtime *runtime, size_t index);
const et_kernel_capability_v1 *et_kernel_runtime_capability_find(
    const et_kernel_runtime *runtime, const char *name);

int32_t et_kernel_runtime_capability_require(
    const et_kernel_runtime *runtime, const char *capability,
    const et_kernel_request_v1 *request,
    const et_kernel_capability_v1 **entry, et_kernel_error *error);

int32_t et_kernel_runtime_dispatch(const et_kernel_runtime *runtime,
                                   const et_kernel_call_v1 *call,
                                   et_kernel_error *error);

/* required_capacity includes the terminal NUL; report bytes end in one LF. */
int32_t et_kernel_runtime_report_json(const et_kernel_runtime *runtime,
                                      char *buffer, size_t capacity,
                                      size_t *required_capacity,
                                      et_kernel_error *error);

#ifdef __cplusplus
}
#endif

#endif
