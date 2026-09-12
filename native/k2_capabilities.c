#include "k2_capabilities_internal.h"

#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/kernel_abi.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifdef ET_K2_TESTING
#include <errno.h>
#include <sys/wait.h>
#endif
#include <sys/types.h>
#include <unistd.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "K2 private v1 transport requires a little-endian target"
#endif

#define ET_K2_ERROR_BYTES ((size_t)264u)
#define ET_K2_BYTEVECTOR_HEADER_BYTES ((size_t)8u)
#define ET_K2_ERROR_CARRIER_BYTES                                              \
  (ET_K2_BYTEVECTOR_HEADER_BYTES + ET_K2_ERROR_BYTES)
#define ET_K2_ENTRY_COUNT ((size_t)11u)
#define ET_K2_FACTORY_CLOSURE_BYTES ((size_t)40u)
#define ET_K2_FACTORY_HEADER_BYTES ((size_t)8u)
#define ET_K2_K1_RUNTIME_CONTROL_BYTES ((size_t)40u)
#define ET_K2_CLOSURE_FLAG_VARIADIC ((uint8_t)1u)

#ifdef __cplusplus
#define ET_K2_STATIC_ASSERT(condition, message)                                \
  static_assert(condition, message)
#define ET_K2_ALIGNOF(type) alignof(type)
#else
#define ET_K2_STATIC_ASSERT(condition, message)                                \
  _Static_assert(condition, message)
#define ET_K2_ALIGNOF(type) _Alignof(type)
#endif

ET_K2_STATIC_ASSERT(sizeof(et_kernel_error) == ET_K2_ERROR_BYTES,
                    "K2 diagnostic record must be 264 bytes");
ET_K2_STATIC_ASSERT(sizeof(int64_t) == ET_K2_BYTEVECTOR_HEADER_BYTES,
                    "Eshkol bytevector header must be eight bytes");
ET_K2_STATIC_ASSERT(offsetof(et_kernel_error, category) == 0u,
                    "K2 diagnostic category offset changed");
ET_K2_STATIC_ASSERT(offsetof(et_kernel_error, code) == 4u,
                    "K2 diagnostic code offset changed");
ET_K2_STATIC_ASSERT(offsetof(et_kernel_error, operation) == 8u,
                    "K2 diagnostic operation offset changed");
ET_K2_STATIC_ASSERT(offsetof(et_kernel_error, message) == 72u,
                    "K2 diagnostic message offset changed");

typedef struct et_k2_closure_header {
  uint8_t subtype;
  uint8_t flags;
  uint16_t reference_count;
  uint32_t size;
} et_k2_closure_header;

typedef struct et_k2_closure_body {
  uint64_t function_address;
  const void *environment;
  uint64_t source_expression;
  const char *name;
  uint8_t return_type;
  uint8_t input_arity;
  uint8_t flags;
  uint8_t reserved;
  uint32_t higher_order_type;
} et_k2_closure_body;

typedef struct et_k2_state {
  et_kernel_runtime *runtime;
  int64_t runtime_pid;
  int64_t generation;
  int busy;
  uint64_t report_factory;
  uint64_t request_factory;
  uint64_t entry_factory;
#ifdef ET_K2_TESTING
  const et_kernel_provider_v1 *provider_override;
  int provider_override_enabled;
  int fail_stage;
  uint64_t discovery_count;
  uint64_t destroy_count;
  uint64_t runtime_live_count;
#endif
} et_k2_state;

ET_K2_STATIC_ASSERT(sizeof(et_k2_closure_header) == ET_K2_FACTORY_HEADER_BYTES,
                    "Eshkol closure header layout changed");
ET_K2_STATIC_ASSERT(sizeof(et_k2_closure_body) == ET_K2_FACTORY_CLOSURE_BYTES,
                    "Eshkol closure body layout changed");

static et_k2_state state;
static const unsigned char resolver_context_sentinel = 0x6bu;

static const char *const expected_names[ET_K2_ENTRY_COUNT] = {
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

static int span(const void *pointer, size_t bytes, uintptr_t *begin,
                uintptr_t *end) {
  uintptr_t address;
  if (pointer == NULL && bytes != 0u) {
    return 0;
  }
  address = (uintptr_t)pointer;
  if (address > UINTPTR_MAX - bytes) {
    return 0;
  }
  if (begin != NULL) {
    *begin = address;
  }
  if (end != NULL) {
    *end = address + bytes;
  }
  return 1;
}

static int overlaps(const void *left, size_t left_bytes, const void *right,
                    size_t right_bytes) {
  uintptr_t left_begin;
  uintptr_t left_end;
  uintptr_t right_begin;
  uintptr_t right_end;
  if (left_bytes == 0u || right_bytes == 0u ||
      !span(left, left_bytes, &left_begin, &left_end) ||
      !span(right, right_bytes, &right_begin, &right_end)) {
    return 0;
  }
  return left_begin < right_end && right_begin < left_end;
}

static int protected_span_overlaps_text(const void *protected_span,
                                        size_t protected_bytes,
                                        const char *text) {
  return text != NULL &&
         overlaps(protected_span, protected_bytes, text, strlen(text) + 1u);
}

static int
protected_span_overlaps_capability(const void *protected_span,
                                   size_t protected_bytes,
                                   const et_kernel_capability_v1 *entry) {
  if (entry == NULL ||
      overlaps(protected_span, protected_bytes, entry, sizeof(*entry)) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   entry->name) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   entry->implementation) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   entry->version) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   entry->evidence)) {
    return entry == NULL ? 0 : 1;
  }
  if (entry->operation_count != 0u &&
      overlaps(protected_span, protected_bytes, entry->operations,
               entry->operation_count * sizeof(*entry->operations))) {
    return 1;
  }
  for (size_t index = 0; index < entry->operation_count; index++) {
    if (protected_span_overlaps_text(protected_span, protected_bytes,
                                     entry->operations[index])) {
      return 1;
    }
  }
  if (entry->dtype_count != 0u &&
      overlaps(protected_span, protected_bytes, entry->dtypes,
               entry->dtype_count * sizeof(*entry->dtypes))) {
    return 1;
  }
  for (size_t index = 0; index < entry->dtype_count; index++) {
    if (protected_span_overlaps_text(protected_span, protected_bytes,
                                     entry->dtypes[index])) {
      return 1;
    }
  }
  if (entry->device_count != 0u &&
      overlaps(protected_span, protected_bytes, entry->devices,
               entry->device_count * sizeof(*entry->devices))) {
    return 1;
  }
  for (size_t index = 0; index < entry->device_count; index++) {
    if (protected_span_overlaps_text(protected_span, protected_bytes,
                                     entry->devices[index])) {
      return 1;
    }
  }
  if (entry->shape_range_count != 0u &&
      overlaps(protected_span, protected_bytes, entry->shape_ranges,
               entry->shape_range_count * sizeof(*entry->shape_ranges))) {
    return 1;
  }
  for (size_t index = 0; index < entry->shape_range_count; index++) {
    const et_kernel_shape_range_v1 *range = &entry->shape_ranges[index];
    if (range->rank != 0u &&
        overlaps(protected_span, protected_bytes, range->dimensions,
                 range->rank * sizeof(*range->dimensions))) {
      return 1;
    }
  }
  return 0;
}

static int
protected_span_overlaps_provider(const void *protected_span,
                                 size_t protected_bytes,
                                 const et_kernel_provider_v1 *provider) {
  if (provider == NULL ||
      overlaps(protected_span, protected_bytes, provider, sizeof(*provider)) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   provider->name) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   provider->version) ||
      protected_span_overlaps_text(protected_span, protected_bytes,
                                   provider->evidence)) {
    return provider == NULL ? 0 : 1;
  }
  if (provider->capability_count != 0u &&
      overlaps(protected_span, protected_bytes, provider->capabilities,
               provider->capability_bytes)) {
    return 1;
  }
  for (size_t index = 0; index < provider->capability_count; index++) {
    const et_kernel_capability_v1 *entry =
        (const et_kernel_capability_v1 *)((const unsigned char *)
                                              provider->capabilities +
                                          index * provider->capability_stride);
    if (protected_span_overlaps_capability(protected_span, protected_bytes,
                                           entry)) {
      return 1;
    }
  }
  return 0;
}

static int span_overlaps_protected(const void *protected_span,
                                   size_t protected_bytes) {
  const et_kernel_provider_v1 *provider = et_f32_tensor_provider_v1();
  if (overlaps(protected_span, protected_bytes, &state, sizeof(state)) ||
      overlaps(protected_span, protected_bytes, &resolver_context_sentinel,
               sizeof(resolver_context_sentinel)) ||
      overlaps(protected_span, protected_bytes, expected_names,
               sizeof(expected_names)) ||
      protected_span_overlaps_provider(protected_span, protected_bytes,
                                       provider)) {
    return 1;
  }
  for (size_t index = 0; index < ET_K2_ENTRY_COUNT; index++) {
    if (protected_span_overlaps_text(protected_span, protected_bytes,
                                     expected_names[index])) {
      return 1;
    }
  }
  if (state.runtime != NULL) {
    size_t count;
    /* Exact-head K1 owns a 40-byte opaque runtime control allocation. */
    if (overlaps(protected_span, protected_bytes, state.runtime,
                 ET_K2_K1_RUNTIME_CONTROL_BYTES)) {
      return 1;
    }
    count = et_kernel_runtime_capability_count(state.runtime);
    for (size_t index = 0; index < count; index++) {
      if (protected_span_overlaps_capability(
              protected_span, protected_bytes,
              et_kernel_runtime_capability_at(state.runtime, index))) {
        return 1;
      }
    }
  }
  return 0;
}

/*
 * Every Eshkol bytevector pointer names its aligned signed-i64 length header;
 * the payload begins eight bytes later.  The private diagnostic protocol uses
 * an exact 264-byte payload: u32 category@0, u32 code@4, operation[64]@8,
 * message[192]@72.  NULL, non-exact supplied or declared capacity,
 * insufficient carrier/payload alignment, wrapping carrier storage, input
 * overlap, or protected-metadata overlap is rejected without changing any of
 * the 272-byte carrier.  Every admitted success or failure copies a fully
 * zeroed local record to the payload only, so the length header and any storage
 * outside the exact carrier remain untouched.
 */
static int bytevector_payload(const void *header, int64_t expected_length,
                              const unsigned char **payload,
                              size_t *carrier_bytes) {
  int64_t declared_length;
  size_t total_bytes;
  uintptr_t unused_begin;
  uintptr_t unused_end;
  if (header == NULL || expected_length < 0 ||
      (uint64_t)expected_length >
          (uint64_t)(SIZE_MAX - ET_K2_BYTEVECTOR_HEADER_BYTES)) {
    return 0;
  }
  total_bytes = ET_K2_BYTEVECTOR_HEADER_BYTES + (size_t)expected_length;
  if ((uintptr_t)header % ET_K2_ALIGNOF(int64_t) != 0u ||
      !span(header, total_bytes, &unused_begin, &unused_end)) {
    return 0;
  }
  memcpy(&declared_length, header, sizeof(declared_length));
  if (declared_length != expected_length) {
    return 0;
  }
  *payload = (const unsigned char *)header + ET_K2_BYTEVECTOR_HEADER_BYTES;
  if (carrier_bytes != NULL) {
    *carrier_bytes = total_bytes;
  }
  return 1;
}

static int admit_error(void *error_header, int64_t error_capacity,
                       void **error_payload) {
  const unsigned char *payload;
  size_t carrier_bytes;
  if (error_capacity != (int64_t)ET_K2_ERROR_BYTES ||
      !bytevector_payload(error_header, error_capacity, &payload,
                          &carrier_bytes) ||
      (uintptr_t)payload % ET_K2_ALIGNOF(et_kernel_error) != 0u ||
      span_overlaps_protected(error_header, carrier_bytes)) {
    return 0;
  }
  *error_payload = (void *)(uintptr_t)payload;
  return 1;
}

static int64_t write_error(void *error_bytes, uint32_t category, uint32_t code,
                           const char *operation, const char *message) {
  et_kernel_error local;
  size_t length;
  memset(&local, 0, sizeof(local));
  local.category = category;
  local.code = code;
  if (operation != NULL) {
    length = strlen(operation);
    if (length >= sizeof(local.operation)) {
      length = sizeof(local.operation) - 1u;
    }
    memcpy(local.operation, operation, length);
  }
  if (message != NULL) {
    length = strlen(message);
    if (length >= sizeof(local.message)) {
      length = sizeof(local.message) - 1u;
    }
    memcpy(local.message, message, length);
  }
  memcpy(error_bytes, &local, sizeof(local));
  return (int64_t)category;
}

static int64_t write_success(void *error_bytes) {
  et_kernel_error local;
  memset(&local, 0, sizeof(local));
  memcpy(error_bytes, &local, sizeof(local));
  return ET_K2_STATUS_OK;
}

static int zero_bytes(const void *bytes, size_t count) {
  const unsigned char *values = (const unsigned char *)bytes;
  for (size_t index = 0; index < count; index++) {
    if (values[index] != 0u) {
      return 0;
    }
  }
  return 1;
}

static int canonical_c_text(const char *text, size_t capacity) {
  const char *terminal;
  if (text == NULL || capacity == 0u || text[0] == '\0') {
    return 0;
  }
  terminal = (const char *)memchr(text, '\0', capacity);
  return terminal != NULL &&
         zero_bytes(terminal + 1u, capacity - (size_t)(terminal + 1u - text));
}

static int k1_category_code_pair_valid(uint32_t category, uint32_t code) {
  switch (category) {
  case ET_KERNEL_ERROR_INVALID_ARGUMENT:
    return code == ET_KERNEL_CODE_NULL_ARGUMENT ||
           code == ET_KERNEL_CODE_INVALID_TEXT ||
           code == ET_KERNEL_CODE_INVALID_ENUM ||
           code == ET_KERNEL_CODE_DUPLICATE_ENTRY ||
           code == ET_KERNEL_CODE_BUFFER_TOO_SMALL ||
           code == ET_KERNEL_CODE_INTEGER_OVERFLOW ||
           code == ET_KERNEL_CODE_INVALID_SHAPE ||
           code == ET_KERNEL_CODE_INVALID_BUFFER ||
           code == ET_KERNEL_CODE_ALIASING_OUTPUT;
  case ET_KERNEL_ERROR_SHAPE_MISMATCH:
    return code == ET_KERNEL_CODE_INTEGER_OVERFLOW ||
           code == ET_KERNEL_CODE_INVALID_SHAPE ||
           code == ET_KERNEL_CODE_INVALID_BUFFER;
  case ET_KERNEL_ERROR_DTYPE_MISMATCH:
    return code == ET_KERNEL_CODE_INVALID_TEXT;
  case ET_KERNEL_ERROR_DEVICE_MISMATCH:
  case ET_KERNEL_ERROR_NONCONTIGUOUS:
    return code == ET_KERNEL_CODE_INVALID_BUFFER;
  case ET_KERNEL_ERROR_UNSUPPORTED:
    return code == ET_KERNEL_CODE_SYMBOL_MISSING ||
           code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED;
  case ET_KERNEL_ERROR_VERSION_MISMATCH:
    return code == ET_KERNEL_CODE_INVALID_STRUCT_SIZE ||
           code == ET_KERNEL_CODE_ABI_MAJOR_MISMATCH ||
           code == ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE;
  case ET_KERNEL_ERROR_INTERNAL:
    return code == ET_KERNEL_CODE_ALLOCATION_FAILED ||
           code == ET_KERNEL_CODE_INTEGER_OVERFLOW ||
           code == ET_KERNEL_CODE_PROVIDER_REJECTED;
  default:
    return 0;
  }
}

static int64_t copy_k1_result(void *error_bytes, int32_t result,
                              const et_kernel_error *source,
                              const char *fallback_operation) {
  uint32_t category;
  et_kernel_error local;
  if (result == 0) {
    if (!zero_bytes(source, sizeof(*source))) {
      return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                         ET_K2_CODE_INVARIANT, fallback_operation,
                         "K1 returned malformed success diagnostics");
    }
    return write_success(error_bytes);
  }
  if (result < ET_KERNEL_ERROR_INVALID_ARGUMENT ||
      result > ET_KERNEL_ERROR_INTERNAL ||
      source->category != (uint32_t)result ||
      source->code == ET_KERNEL_CODE_OK ||
      source->code > ET_KERNEL_CODE_PROVIDER_REJECTED ||
      !k1_category_code_pair_valid(source->category, source->code) ||
      !canonical_c_text(source->operation, sizeof(source->operation)) ||
      !canonical_c_text(source->message, sizeof(source->message))) {
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL, ET_K2_CODE_INVARIANT,
                       fallback_operation,
                       "K1 returned malformed failure diagnostics");
  }
  category = source->category;
  if (category == ET_KERNEL_ERROR_VERSION_MISMATCH) {
    category = ET_K2_STATUS_VERSION_MISMATCH;
  } else if (category == ET_KERNEL_ERROR_INTERNAL) {
    category = ET_K2_STATUS_INTERNAL;
  }
  memset(&local, 0, sizeof(local));
  local.category = category;
  local.code = source->code;
  memcpy(local.operation, source->operation, sizeof(local.operation));
  memcpy(local.message, source->message, sizeof(local.message));
  memcpy(error_bytes, &local, sizeof(local));
  return (int64_t)category;
}

static int exact_text(const char *actual, const char *expected) {
  return actual != NULL && strcmp(actual, expected) == 0;
}

static int reserved_zero(const uint8_t *bytes, size_t count) {
  for (size_t index = 0; index < count; index++) {
    if (bytes[index] != 0u) {
      return 0;
    }
  }
  return 1;
}

static int exact_i2_capability(const et_kernel_capability_v1 *entry) {
  const uint64_t maximum = (uint64_t)(SIZE_MAX / sizeof(float));
  if (entry == NULL || entry->struct_size != ET_KERNEL_CAPABILITY_V1_0_SIZE ||
      !exact_text(entry->name, "tensor.f32") ||
      entry->status != ET_KERNEL_CAPABILITY_VERIFIED ||
      !exact_text(entry->implementation, "eshkol-transformer-f32") ||
      !exact_text(entry->version, "1.0") ||
      !exact_text(entry->evidence, "I2:bounded-exact-f32-storage.copy-v1") ||
      entry->deterministic != 1u ||
      !reserved_zero(entry->reserved, sizeof(entry->reserved)) ||
      entry->operation_count != 1u || entry->operations == NULL ||
      !exact_text(entry->operations[0], "storage.copy") ||
      entry->dtype_count != 1u || entry->dtypes == NULL ||
      !exact_text(entry->dtypes[0], "f32") || entry->device_count != 1u ||
      entry->devices == NULL || !exact_text(entry->devices[0], "cpu") ||
      entry->shape_range_count != 2u || entry->shape_ranges == NULL) {
    return 0;
  }
  if (entry->shape_ranges[0].rank != 0u ||
      entry->shape_ranges[0].dimensions != NULL ||
      entry->shape_ranges[1].rank != 1u ||
      entry->shape_ranges[1].dimensions == NULL) {
    return 0;
  }
  return entry->shape_ranges[1].dimensions[0].minimum == 0u &&
         entry->shape_ranges[1].dimensions[0].maximum == maximum &&
         entry->shape_ranges[1].dimensions[0].maximum_unbounded == 0u &&
         reserved_zero(entry->shape_ranges[1].dimensions[0].reserved,
                       sizeof(entry->shape_ranges[1].dimensions[0].reserved));
}

static int exact_i2_provider(const et_kernel_provider_v1 *provider) {
  const et_kernel_provider_v1 *known = et_f32_tensor_provider_v1();
  const et_kernel_capability_v1 *entry;
  if (provider == NULL || known == NULL ||
      provider->struct_size != ET_KERNEL_PROVIDER_V1_0_SIZE ||
      provider->abi_major != ET_KERNEL_ABI_MAJOR ||
      provider->abi_minor != ET_KERNEL_ABI_MINOR ||
      provider->required_features !=
          ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES ||
      !exact_text(provider->name, "eshkol-transformer-f32") ||
      !exact_text(provider->version, "1.0") ||
      !exact_text(provider->evidence, "I2:bounded-exact-f32-storage.copy-v1") ||
      provider->capability_count != 1u ||
      provider->capability_stride != ET_KERNEL_CAPABILITY_V1_0_SIZE ||
      provider->capability_bytes != ET_KERNEL_CAPABILITY_V1_0_SIZE ||
      provider->capabilities == NULL ||
      provider->validate_call != known->validate_call ||
      provider->invoke_call != known->invoke_call) {
    return 0;
  }
  entry = (const et_kernel_capability_v1 *)provider->capabilities;
  return exact_i2_capability(entry);
}

enum et_k2_provider_audit {
  ET_K2_PROVIDER_ABSENT,
  ET_K2_PROVIDER_VERSION_MISMATCH,
  ET_K2_PROVIDER_METADATA_MISMATCH,
  ET_K2_PROVIDER_EXACT
};

static enum et_k2_provider_audit
audit_i2_provider(const et_kernel_provider_v1 *provider) {
  if (provider == NULL) {
    return ET_K2_PROVIDER_ABSENT;
  }
  if (provider->struct_size != ET_KERNEL_PROVIDER_V1_0_SIZE ||
      provider->abi_major != ET_KERNEL_ABI_MAJOR ||
      provider->abi_minor != ET_KERNEL_ABI_MINOR ||
      provider->required_features !=
          ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES ||
      !exact_text(provider->version, "1.0")) {
    return ET_K2_PROVIDER_VERSION_MISMATCH;
  }
  if (provider->capability_count == 1u &&
      provider->capability_stride == ET_KERNEL_CAPABILITY_V1_0_SIZE &&
      provider->capability_bytes == ET_KERNEL_CAPABILITY_V1_0_SIZE &&
      provider->capabilities != NULL &&
      ((const et_kernel_capability_v1 *)provider->capabilities)->struct_size !=
          ET_KERNEL_CAPABILITY_V1_0_SIZE) {
    return ET_K2_PROVIDER_VERSION_MISMATCH;
  }
  return exact_i2_provider(provider) ? ET_K2_PROVIDER_EXACT
                                     : ET_K2_PROVIDER_METADATA_MISMATCH;
}

static int exact_baseline(const et_kernel_capability_v1 *entry,
                          const char *name) {
  return entry != NULL &&
         entry->struct_size == ET_KERNEL_CAPABILITY_V1_0_SIZE &&
         exact_text(entry->name, name) &&
         entry->status == ET_KERNEL_CAPABILITY_UNVERIFIED &&
         exact_text(entry->implementation, "eshkol-core") &&
         exact_text(entry->version, "1.3.4-evolve") &&
         exact_text(entry->evidence,
                    "R0-merged-through-PR15:untested-with-reason") &&
         entry->deterministic == 0u &&
         reserved_zero(entry->reserved, sizeof(entry->reserved)) &&
         entry->operation_count == 0u && entry->operations == NULL &&
         entry->dtype_count == 0u && entry->dtypes == NULL &&
         entry->device_count == 0u && entry->devices == NULL &&
         entry->shape_range_count == 0u && entry->shape_ranges == NULL;
}

static int exact_runtime(const et_kernel_runtime *runtime) {
  if (runtime == NULL ||
      et_kernel_runtime_capability_count(runtime) != ET_K2_ENTRY_COUNT) {
    return 0;
  }
  for (size_t index = 0; index < ET_K2_ENTRY_COUNT; index++) {
    const et_kernel_capability_v1 *entry =
        et_kernel_runtime_capability_at(runtime, index);
    if (index == 9u) {
      if (!exact_i2_capability(entry)) {
        return 0;
      }
    } else if (!exact_baseline(entry, expected_names[index])) {
      return 0;
    }
  }
  return 1;
}

static const et_kernel_provider_v1 *provider_for_discovery(void) {
#ifdef ET_K2_TESTING
  if (state.provider_override_enabled) {
    return state.provider_override;
  }
#endif
  return et_f32_tensor_provider_v1();
}

static const et_kernel_provider_v1 *resolve_i2(void *context,
                                               const char *symbol_name) {
  if (context != &resolver_context_sentinel || symbol_name == NULL ||
      strcmp(symbol_name, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0) {
    return NULL;
  }
  return provider_for_discovery();
}

static void destroy_runtime(et_kernel_runtime *runtime) {
  if (runtime == NULL) {
    return;
  }
  et_kernel_runtime_destroy(runtime);
#ifdef ET_K2_TESTING
  state.destroy_count++;
  if (state.runtime_live_count != 0u) {
    state.runtime_live_count--;
  }
#endif
}

int64_t et_k2_private_runtime_pid_v1(void) {
  const pid_t pid = getpid();
  return pid > 0 ? (int64_t)pid : ET_K2_SCALAR_INTERNAL;
}

int64_t et_k2_private_runtime_generation_v1(void) {
  const int64_t pid = et_k2_private_runtime_pid_v1();
  if (pid <= 0) {
    return ET_K2_SCALAR_INTERNAL;
  }
  if (state.runtime == NULL || state.runtime_pid != pid) {
    return ET_K2_SCALAR_INVALID_STATE;
  }
  if (state.generation <= 0 || !exact_runtime(state.runtime)) {
    return ET_K2_SCALAR_INTERNAL;
  }
  return state.generation;
}

int64_t et_k2_private_runtime_ensure_v1(void *error_bytes,
                                        int64_t error_capacity) {
  void *error_payload;
  et_kernel_runtime *candidate = NULL;
  et_kernel_error k1_error;
  const et_kernel_provider_v1 *provider;
  enum et_k2_provider_audit provider_audit;
  int64_t pid;
  int32_t result;
  if (!admit_error(error_bytes, error_capacity, &error_payload)) {
    return ET_K2_STATUS_INVALID_ARGUMENT;
  }
  error_bytes = error_payload;
  if (state.busy) {
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL, ET_K2_CODE_INVARIANT,
                       "capability-discover",
                       "concurrent or reentrant K2 use is unsupported");
  }
  pid = et_k2_private_runtime_pid_v1();
  if (pid <= 0) {
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL, ET_K2_CODE_INVARIANT,
                       "capability-discover",
                       "cannot obtain a valid process identifier");
  }
  state.busy = 1;
  if (state.runtime != NULL && state.runtime_pid == pid) {
    if (state.generation <= 0 || !exact_runtime(state.runtime)) {
      state.busy = 0;
      return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                         ET_K2_CODE_RUNTIME_AUDIT, "capability-discover",
                         "the published K1 runtime failed an invariant audit");
    }
    state.busy = 0;
    return write_success(error_bytes);
  }
  if (state.runtime != NULL) {
    et_kernel_runtime *inherited = state.runtime;
    state.runtime = NULL;
    state.runtime_pid = 0;
    destroy_runtime(inherited);
  }
  provider = provider_for_discovery();
  provider_audit = audit_i2_provider(provider);
  if (provider_audit == ET_K2_PROVIDER_ABSENT) {
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_UNSUPPORTED,
                       ET_K2_CODE_PROVIDER_ABSENT, "capability-discover",
                       "the fixed I2 provider is absent");
  }
  if (provider_audit == ET_K2_PROVIDER_VERSION_MISMATCH) {
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_VERSION_MISMATCH,
                       ET_K2_CODE_PROVIDER_AUDIT, "capability-discover",
                       "the fixed I2 provider ABI is incompatible");
  }
  if (provider_audit != ET_K2_PROVIDER_EXACT) {
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_PROVIDER_AUDIT, "capability-discover",
                       "the fixed I2 provider descriptor failed exact audit");
  }
#ifdef ET_K2_TESTING
  if (state.fail_stage == ET_K2_TEST_FAIL_BEFORE_DISCOVER) {
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_PROVIDER_AUDIT, "capability-discover",
                       "injected failure before K1 discovery");
  }
  state.discovery_count++;
#endif
  memset(&k1_error, 0, sizeof(k1_error));
  result = et_kernel_runtime_discover(
      resolve_i2, (void *)&resolver_context_sentinel, &candidate, &k1_error);
  if (result != 0) {
    if (candidate != NULL) {
      destroy_runtime(candidate);
    }
    state.busy = 0;
    return copy_k1_result(error_bytes, result, &k1_error,
                          "capability-discover");
  }
#ifdef ET_K2_TESTING
  state.runtime_live_count++;
  if (state.fail_stage == ET_K2_TEST_FAIL_AFTER_DISCOVER ||
      state.fail_stage == ET_K2_TEST_FAIL_RUNTIME_AUDIT) {
    destroy_runtime(candidate);
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_RUNTIME_AUDIT, "capability-discover",
                       "injected failure after K1 discovery");
  }
#endif
  if (!zero_bytes(&k1_error, sizeof(k1_error))) {
    destroy_runtime(candidate);
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL, ET_K2_CODE_INVARIANT,
                       "capability-discover",
                       "K1 returned malformed success diagnostics");
  }
  if (!exact_runtime(candidate)) {
    destroy_runtime(candidate);
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_RUNTIME_AUDIT, "capability-discover",
                       "the eleven-row K1 report failed exact audit");
  }
  if (state.generation == INT64_MAX) {
    destroy_runtime(candidate);
    state.busy = 0;
    return write_error(error_bytes, ET_K2_STATUS_INTERNAL, ET_K2_CODE_INVARIANT,
                       "capability-discover",
                       "K2 runtime generation overflowed");
  }
  state.runtime = candidate;
  state.runtime_pid = pid;
  state.generation++;
  if (state.generation <= 0) {
    state.generation = 1;
  }
  state.busy = 0;
  return write_success(error_bytes);
}

static int valid_symbol_bytes(const void *bytes, int64_t byte_count) {
  const unsigned char *text = (const unsigned char *)bytes;
  if (text == NULL || byte_count <= 0 ||
      byte_count > (int64_t)ET_KERNEL_MAX_SYMBOL_BYTES ||
      text[0] < (unsigned char)'a' || text[0] > (unsigned char)'z') {
    return 0;
  }
  for (int64_t index = 0; index < byte_count; index++) {
    const unsigned char value = text[index];
    if (!((value >= (unsigned char)'a' && value <= (unsigned char)'z') ||
          (value >= (unsigned char)'0' && value <= (unsigned char)'9') ||
          value == (unsigned char)'.' || value == (unsigned char)'_' ||
          value == (unsigned char)'-' || value == (unsigned char)':')) {
      return 0;
    }
  }
  return 1;
}

static int input_carrier_overlaps_error(const void *error_header,
                                        const void *input_header,
                                        int64_t input_bytes) {
  size_t carrier_bytes;
  if (input_header == NULL) {
    return 0;
  }
  if (overlaps(error_header, ET_K2_ERROR_CARRIER_BYTES, input_header,
               ET_K2_BYTEVECTOR_HEADER_BYTES)) {
    return 1;
  }
  if (input_bytes < 0 ||
      (uint64_t)input_bytes >
          (uint64_t)(SIZE_MAX - ET_K2_BYTEVECTOR_HEADER_BYTES)) {
    return 0;
  }
  carrier_bytes = ET_K2_BYTEVECTOR_HEADER_BYTES + (size_t)input_bytes;
  return overlaps(error_header, ET_K2_ERROR_CARRIER_BYTES, input_header,
                  carrier_bytes);
}

static uint64_t decode_u64le(const unsigned char *bytes) {
  uint64_t value = 0u;
  for (unsigned int shift = 0u; shift < 64u; shift += 8u) {
    value |= (uint64_t)bytes[shift / 8u] << shift;
  }
  return value;
}

int64_t et_k2_private_runtime_require_v1(
    int64_t expected_generation, int64_t sorted_entry_index,
    const void *operation, int64_t operation_bytes, const void *dtype,
    int64_t dtype_bytes, const void *device, int64_t device_bytes,
    const void *shape_u64le, int64_t shape_bytes, int64_t rank,
    int64_t deterministic, void *error_bytes, int64_t error_capacity) {
  char operation_text[ET_KERNEL_MAX_SYMBOL_BYTES + 1u];
  char dtype_text[ET_KERNEL_MAX_SYMBOL_BYTES + 1u];
  char device_text[ET_KERNEL_MAX_SYMBOL_BYTES + 1u];
  uint64_t shape[ET_KERNEL_MAX_RANK];
  et_kernel_request_v1 request;
  const et_kernel_capability_v1 *entry = NULL;
  et_kernel_error k1_error;
  const unsigned char *operation_payload;
  const unsigned char *dtype_payload;
  const unsigned char *device_payload;
  const unsigned char *shape_payload;
  void *error_payload;
  size_t unused_carrier_bytes;
  int operation_valid;
  int dtype_valid;
  int device_valid;
  int shape_valid;
  int32_t result;
  int64_t pid;
  if (!admit_error(error_bytes, error_capacity, &error_payload)) {
    return ET_K2_STATUS_INVALID_ARGUMENT;
  }
  if (input_carrier_overlaps_error(error_bytes, operation, operation_bytes) ||
      input_carrier_overlaps_error(error_bytes, dtype, dtype_bytes) ||
      input_carrier_overlaps_error(error_bytes, device, device_bytes) ||
      input_carrier_overlaps_error(error_bytes, shape_u64le, shape_bytes)) {
    return ET_K2_STATUS_INVALID_ARGUMENT;
  }
  operation_valid = bytevector_payload(
      operation, operation_bytes, &operation_payload, &unused_carrier_bytes);
  dtype_valid = bytevector_payload(dtype, dtype_bytes, &dtype_payload,
                                   &unused_carrier_bytes);
  device_valid = bytevector_payload(device, device_bytes, &device_payload,
                                    &unused_carrier_bytes);
  shape_valid = bytevector_payload(shape_u64le, shape_bytes, &shape_payload,
                                   &unused_carrier_bytes);
  if (!operation_valid || !dtype_valid || !device_valid || !shape_valid ||
      rank < 0 || rank > (int64_t)ET_KERNEL_MAX_RANK ||
      (rank > 0 && rank > INT64_MAX / 8) || shape_bytes != rank * 8 ||
      deterministic < 0 || deterministic > 1 ||
      !valid_symbol_bytes(operation_payload, operation_bytes) ||
      !valid_symbol_bytes(dtype_payload, dtype_bytes) ||
      !valid_symbol_bytes(device_payload, device_bytes)) {
    return write_error(error_payload, ET_K2_STATUS_INVALID_ARGUMENT,
                       ET_K2_CODE_TRANSPORT, "capability-require",
                       "K2 request transport is malformed");
  }
  if (expected_generation <= 0) {
    return write_error(error_payload, ET_K2_STATUS_INVALID_ARGUMENT,
                       ET_K2_CODE_TRANSPORT, "capability-require",
                       "expected runtime generation must be positive");
  }
  if (sorted_entry_index < 0 ||
      sorted_entry_index >= (int64_t)ET_K2_ENTRY_COUNT) {
    return write_error(error_payload, ET_K2_STATUS_INVALID_ARGUMENT,
                       ET_K2_CODE_ENTRY_INDEX, "capability-require",
                       "sorted capability entry index is outside 0..10");
  }
  pid = et_k2_private_runtime_pid_v1();
  if (state.busy) {
    return write_error(error_payload, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_INVARIANT, "capability-require",
                       "concurrent or reentrant K2 use is unsupported");
  }
  if (pid <= 0 || state.runtime == NULL || state.runtime_pid != pid ||
      state.generation != expected_generation) {
    return write_error(error_payload, ET_K2_STATUS_INVALID_STATE,
                       ET_K2_CODE_STALE, "capability-require",
                       "K2 runtime origin or generation is stale");
  }
  if (!exact_runtime(state.runtime)) {
    return write_error(
        error_payload, ET_K2_STATUS_INTERNAL, ET_K2_CODE_RUNTIME_AUDIT,
        "capability-require",
        "the eleven-row K1 runtime no longer passes exact audit");
  }
  memcpy(operation_text, operation_payload, (size_t)operation_bytes);
  operation_text[operation_bytes] = '\0';
  memcpy(dtype_text, dtype_payload, (size_t)dtype_bytes);
  dtype_text[dtype_bytes] = '\0';
  memcpy(device_text, device_payload, (size_t)device_bytes);
  device_text[device_bytes] = '\0';
  for (int64_t index = 0; index < rank; index++) {
    shape[index] = decode_u64le(shape_payload + index * 8);
  }
  memset(&request, 0, sizeof(request));
  request.struct_size = sizeof(request);
  request.operation = operation_text;
  request.dtype = dtype_text;
  request.device = device_text;
  request.rank = (size_t)rank;
  request.shape = rank == 0 ? NULL : shape;
  request.deterministic = (uint8_t)deterministic;
  memset(&k1_error, 0, sizeof(k1_error));
  state.busy = 1;
  result = et_kernel_runtime_capability_require(
      state.runtime, expected_names[sorted_entry_index], &request, &entry,
      &k1_error);
  state.busy = 0;
  if (result == 0 &&
      (entry == NULL ||
       entry != et_kernel_runtime_capability_at(state.runtime,
                                                (size_t)sorted_entry_index))) {
    return write_error(error_payload, ET_K2_STATUS_INTERNAL,
                       ET_K2_CODE_RUNTIME_AUDIT, "capability-require",
                       "K1 returned an unexpected capability entry");
  }
  return copy_k1_result(error_payload, result, &k1_error, "capability-require");
}

static int closure_code(const void *closure, uint64_t *code) {
  et_k2_closure_header header;
  et_k2_closure_body body;
  size_t environment_header;
  const uintptr_t address = (uintptr_t)closure;
  uintptr_t unused_begin;
  uintptr_t unused_end;
  if (closure == NULL || code == NULL || address < ET_K2_FACTORY_HEADER_BYTES ||
      address % ET_K2_ALIGNOF(et_k2_closure_body) != 0u ||
      !span((const unsigned char *)closure - ET_K2_FACTORY_HEADER_BYTES,
            ET_K2_FACTORY_HEADER_BYTES + ET_K2_FACTORY_CLOSURE_BYTES,
            &unused_begin, &unused_end)) {
    return 0;
  }
  memcpy(&header, (const unsigned char *)closure - ET_K2_FACTORY_HEADER_BYTES,
         sizeof(header));
  memcpy(&body, closure, sizeof(body));
  if (header.subtype != 0u || header.size != ET_K2_FACTORY_CLOSURE_BYTES ||
      body.function_address == 0u || body.environment == NULL ||
      body.input_arity != 1u ||
      (body.flags & ET_K2_CLOSURE_FLAG_VARIADIC) != 0u ||
      (uintptr_t)body.environment % ET_K2_ALIGNOF(size_t) != 0u ||
      !span(body.environment, sizeof(environment_header), &unused_begin,
            &unused_end)) {
    return 0;
  }
  memcpy(&environment_header, body.environment, sizeof(environment_header));
  if ((environment_header & (size_t)0xffffu) != (size_t)1u ||
      ((environment_header >> 16u) & (size_t)0xffffu) != (size_t)1u ||
      (environment_header >> (sizeof(size_t) * CHAR_BIT - 1u)) != 0u) {
    return 0;
  }
  *code = body.function_address;
  return 1;
}

static int64_t register_factory(const void *closure, uint64_t *slot,
                                const uint64_t *other_a,
                                const uint64_t *other_b) {
  uint64_t code;
  const int64_t pid = et_k2_private_runtime_pid_v1();
  if (!closure_code(closure, &code)) {
    return ET_K2_SCALAR_INVALID_ARGUMENT;
  }
  /* Factory identity freezes only after the exact runtime audit succeeds. */
  if (pid <= 0 || state.runtime == NULL || state.runtime_pid != pid ||
      state.generation <= 0 || !exact_runtime(state.runtime)) {
    return ET_K2_SCALAR_INTERNAL;
  }
  if (*slot == code) {
    return 1;
  }
  if (*slot != 0u || *other_a == code || *other_b == code) {
    return ET_K2_SCALAR_INTERNAL;
  }
  *slot = code;
  return 1;
}

static int64_t authenticate_factory(const void *closure, const uint64_t *slot) {
  uint64_t code;
  if (*slot == 0u) {
    return ET_K2_SCALAR_INTERNAL;
  }
  return closure_code(closure, &code) && code == *slot ? 1 : 0;
}

int64_t et_k2_private_report_factory_register_v1(const void *closure) {
  return register_factory(closure, &state.report_factory,
                          &state.request_factory, &state.entry_factory);
}

int64_t et_k2_private_request_factory_register_v1(const void *closure) {
  return register_factory(closure, &state.request_factory,
                          &state.report_factory, &state.entry_factory);
}

int64_t et_k2_private_entry_factory_register_v1(const void *closure) {
  return register_factory(closure, &state.entry_factory, &state.report_factory,
                          &state.request_factory);
}

int64_t et_k2_private_report_factory_authenticate_v1(const void *closure) {
  return authenticate_factory(closure, &state.report_factory);
}

int64_t et_k2_private_request_factory_authenticate_v1(const void *closure) {
  return authenticate_factory(closure, &state.request_factory);
}

int64_t et_k2_private_entry_factory_authenticate_v1(const void *closure) {
  return authenticate_factory(closure, &state.entry_factory);
}

#ifdef ET_K2_TESTING
void et_k2_test_reset_v1(void) {
  if (state.runtime != NULL) {
    et_kernel_runtime_destroy(state.runtime);
  }
  memset(&state, 0, sizeof(state));
}

void et_k2_test_runtime_drop_v1(void) {
  et_kernel_runtime *runtime = state.runtime;
  state.runtime = NULL;
  state.runtime_pid = 0;
  destroy_runtime(runtime);
}

void et_k2_test_provider_override_v1(const et_kernel_provider_v1 *provider,
                                     int enabled) {
  state.provider_override = provider;
  state.provider_override_enabled = enabled != 0;
}

void et_k2_test_fail_stage_v1(int stage) { state.fail_stage = stage; }

uint64_t et_k2_test_discovery_count_v1(void) { return state.discovery_count; }

uint64_t et_k2_test_destroy_count_v1(void) { return state.destroy_count; }

uint64_t et_k2_test_runtime_live_count_v1(void) {
  return state.runtime_live_count;
}

int64_t et_k2_test_fork_v1(void) {
  const pid_t child = fork();
  return child < 0 ? -1 : (int64_t)child;
}

int64_t et_k2_test_wait_child_v1(int64_t child_pid) {
  int status = 0;
  pid_t child;
  if (child_pid <= 0 || (int64_t)(pid_t)child_pid != child_pid) {
    return 0;
  }
  do {
    child = waitpid((pid_t)child_pid, &status, 0);
  } while (child < 0 && errno == EINTR);
  return child == (pid_t)child_pid && WIFEXITED(status) &&
                 WEXITSTATUS(status) == 0
             ? 1
             : 0;
}

void et_k2_test_exit_child_v1(int64_t status) {
  _exit(status >= 0 && status <= 255 ? (int)status : 255);
}

int64_t et_k2_test_copy_k1_result_v1(int32_t result, uint32_t category,
                                     uint32_t code, uint32_t text_fault,
                                     void *error_bytes,
                                     int64_t error_capacity) {
  et_kernel_error source;
  void *error_payload;
  if (!admit_error(error_bytes, error_capacity, &error_payload)) {
    return ET_K2_STATUS_INVALID_ARGUMENT;
  }
  memset(&source, 0, sizeof(source));
  source.category = category;
  source.code = code;
  memcpy(source.operation, "capability-require", sizeof("capability-require"));
  memcpy(source.message, "test K1 diagnostic", sizeof("test K1 diagnostic"));
  if (text_fault == 1u) {
    memset(source.operation, 'x', sizeof(source.operation));
  } else if (text_fault == 2u) {
    source.message[0] = 'x';
    source.message[1] = '\0';
    source.message[2] = 'y';
  }
  return copy_k1_result(error_payload, result, &source, "capability-require");
}
#endif
