#include "eshkol_transformer/kernel_abi.h"

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ET_ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

struct et_kernel_runtime {
  et_kernel_validate_call_v1 validate_call;
  et_kernel_invoke_call_v1 invoke_call;
  uint32_t provider_minor;
  uint8_t provider_present;
  size_t capability_count;
  et_kernel_capability_v1 *capabilities;
};

static const char *const baseline_names[] = {
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

static int32_t set_error(et_kernel_error *error,
                         et_kernel_error_category category,
                         et_kernel_error_code code, const char *operation,
                         const char *message) {
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s",
                   operation == NULL ? "kernel-abi" : operation);
    (void)snprintf(error->message, sizeof(error->message), "%s",
                   message == NULL ? "kernel ABI error" : message);
  }
  return (int32_t)category;
}

void et_kernel_error_clear(et_kernel_error *error) {
  if (error != NULL) {
    memset(error, 0, sizeof(*error));
  }
}

int32_t et_kernel_abi_major(void) { return (int32_t)ET_KERNEL_ABI_MAJOR; }
int32_t et_kernel_abi_minor(void) { return (int32_t)ET_KERNEL_ABI_MINOR; }
const char *et_kernel_provider_symbol(void) {
  return ET_KERNEL_PROVIDER_SYMBOL_V1;
}

static int bounded_text_length(const char *text, size_t maximum,
                               size_t *length) {
  const char *terminal;
  if (text == NULL || maximum == SIZE_MAX) {
    return 0;
  }
  terminal = (const char *)memchr(text, '\0', maximum + 1u);
  if (terminal == NULL) {
    return 0;
  }
  if (length != NULL) {
    *length = (size_t)(terminal - text);
  }
  return 1;
}

static int valid_utf8(const char *text, size_t maximum) {
  const unsigned char *cursor = (const unsigned char *)text;
  size_t length;
  const unsigned char *end;
  if (!bounded_text_length(text, maximum, &length)) {
    return 0;
  }
  end = cursor + length;
  while (cursor < end) {
    uint32_t codepoint;
    size_t continuation;
    if (*cursor < 0x80u) {
      cursor++;
      continue;
    }
    if ((*cursor & 0xe0u) == 0xc0u) {
      codepoint = (uint32_t)(*cursor & 0x1fu);
      continuation = 1;
      if (codepoint < 2u) {
        return 0;
      }
    } else if ((*cursor & 0xf0u) == 0xe0u) {
      codepoint = (uint32_t)(*cursor & 0x0fu);
      continuation = 2;
    } else if ((*cursor & 0xf8u) == 0xf0u) {
      codepoint = (uint32_t)(*cursor & 0x07u);
      continuation = 3;
    } else {
      return 0;
    }
    cursor++;
    if ((size_t)(end - cursor) < continuation) {
      return 0;
    }
    for (size_t index = 0; index < continuation; index++, cursor++) {
      if ((*cursor & 0xc0u) != 0x80u) {
        return 0;
      }
      codepoint = (codepoint << 6u) | (uint32_t)(*cursor & 0x3fu);
    }
    if ((continuation == 2 && codepoint < 0x800u) ||
        (continuation == 3 && codepoint < 0x10000u) ||
        (codepoint >= 0xd800u && codepoint <= 0xdfffu) ||
        codepoint > 0x10ffffu) {
      return 0;
    }
  }
  return 1;
}

static int valid_symbol(const char *text) {
  const unsigned char *cursor = (const unsigned char *)text;
  size_t length = 0;
  if (!bounded_text_length(text, ET_KERNEL_MAX_SYMBOL_BYTES, &length) ||
      length == 0u || *cursor < 'a' || *cursor > 'z') {
    return 0;
  }
  for (size_t index = 0; index < length; index++, cursor++) {
    const int valid = (*cursor >= 'a' && *cursor <= 'z') ||
                      (*cursor >= '0' && *cursor <= '9') || *cursor == '.' ||
                      *cursor == '_' || *cursor == '-' || *cursor == ':';
    if (!valid) {
      return 0;
    }
  }
  return 1;
}

static char *duplicate_text(const char *text) {
  const size_t length = strlen(text);
  char *copy = (char *)malloc(length + 1u);
  if (copy != NULL) {
    memcpy(copy, text, length + 1u);
  }
  return copy;
}

static void free_string_array(size_t count, const char *const *values) {
  if (values != NULL) {
    for (size_t index = 0; index < count; index++) {
      free((void *)values[index]);
    }
  }
  free((void *)values);
}

static void free_capability(et_kernel_capability_v1 *capability) {
  if (capability == NULL) {
    return;
  }
  free((void *)capability->name);
  free((void *)capability->implementation);
  free((void *)capability->version);
  free((void *)capability->evidence);
  free_string_array(capability->operation_count, capability->operations);
  free_string_array(capability->dtype_count, capability->dtypes);
  free_string_array(capability->device_count, capability->devices);
  if (capability->shape_ranges != NULL) {
    for (size_t index = 0; index < capability->shape_range_count; index++) {
      free((void *)capability->shape_ranges[index].dimensions);
    }
  }
  free((void *)capability->shape_ranges);
  memset(capability, 0, sizeof(*capability));
}

void et_kernel_runtime_destroy(et_kernel_runtime *runtime) {
  if (runtime == NULL) {
    return;
  }
  for (size_t index = 0; index < runtime->capability_count; index++) {
    free_capability(&runtime->capabilities[index]);
  }
  free(runtime->capabilities);
  free(runtime);
}

static int string_pointer_compare(const void *left, const void *right) {
  const char *const *left_text = (const char *const *)left;
  const char *const *right_text = (const char *const *)right;
  return strcmp(*left_text, *right_text);
}

static int capability_compare(const void *left, const void *right) {
  const et_kernel_capability_v1 *left_capability =
      (const et_kernel_capability_v1 *)left;
  const et_kernel_capability_v1 *right_capability =
      (const et_kernel_capability_v1 *)right;
  return strcmp(left_capability->name, right_capability->name);
}

static int shape_range_compare(const void *left, const void *right) {
  const et_kernel_shape_range_v1 *left_range =
      (const et_kernel_shape_range_v1 *)left;
  const et_kernel_shape_range_v1 *right_range =
      (const et_kernel_shape_range_v1 *)right;
  if (left_range->rank != right_range->rank) {
    return left_range->rank < right_range->rank ? -1 : 1;
  }
  for (size_t index = 0; index < left_range->rank; index++) {
    const et_kernel_dimension_range_v1 *left_dimension =
        &left_range->dimensions[index];
    const et_kernel_dimension_range_v1 *right_dimension =
        &right_range->dimensions[index];
    if (left_dimension->minimum != right_dimension->minimum) {
      return left_dimension->minimum < right_dimension->minimum ? -1 : 1;
    }
    if (left_dimension->maximum_unbounded !=
        right_dimension->maximum_unbounded) {
      return left_dimension->maximum_unbounded ? 1 : -1;
    }
    if (!left_dimension->maximum_unbounded &&
        left_dimension->maximum != right_dimension->maximum) {
      return left_dimension->maximum < right_dimension->maximum ? -1 : 1;
    }
  }
  return 0;
}

static int32_t validate_string_array(const char *operation, size_t count,
                                     const char *const *source,
                                     et_kernel_error *error) {
  if (count > ET_KERNEL_MAX_LIST_ENTRIES ||
      count > SIZE_MAX / sizeof(*source)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                     "capability string list exceeds ABI v1 limit");
  }
  if (count == 0u) {
    return source == NULL
               ? 0
               : set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                           ET_KERNEL_CODE_INVALID_BUFFER, operation,
                           "empty capability string list has storage");
  }
  if (source == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                     "nonempty capability string list has null storage");
  }
  if ((uintptr_t)source % _Alignof(const char *) != 0u ||
      (uintptr_t)source > UINTPTR_MAX - count * sizeof(*source)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "capability string list storage is misaligned or overflows");
  }
  for (size_t index = 0; index < count; index++) {
    if (!valid_symbol(source[index])) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_TEXT, operation,
                       "capability list contains an invalid symbol");
    }
    for (size_t earlier = 0; earlier < index; earlier++) {
      if (strcmp(source[earlier], source[index]) == 0) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_DUPLICATE_ENTRY, operation,
                         "capability list contains a duplicate symbol");
      }
    }
  }
  return 0;
}

static int32_t copy_string_array(const char *operation, size_t count,
                                 const char *const *source,
                                 const char *const **destination,
  et_kernel_error *error) {
  const char **copy = NULL;
  int32_t result;
  *destination = NULL;
  result = validate_string_array(operation, count, source, error);
  if (result != 0 || count == 0u) {
    return result;
  }
  copy = (const char **)calloc(count, sizeof(*copy));
  if (copy == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_ALLOCATION_FAILED, operation,
                     "cannot allocate capability string list");
  }
  for (size_t index = 0; index < count; index++) {
    copy[index] = duplicate_text(source[index]);
    if (copy[index] == NULL) {
      free_string_array(index, copy);
      return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                       ET_KERNEL_CODE_ALLOCATION_FAILED, operation,
                       "cannot copy capability symbol");
    }
  }
  qsort(copy, count, sizeof(*copy), string_pointer_compare);
  *destination = copy;
  return 0;
}

static int32_t validate_shape_ranges(
    const char *operation, size_t count,
    const et_kernel_shape_range_v1 *source, et_kernel_error *error) {
  if (count > ET_KERNEL_MAX_SHAPE_RANGES ||
      count > SIZE_MAX / sizeof(*source)) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                     "shape-range list exceeds ABI v1 limit");
  }
  if (count == 0u) {
    return source == NULL
               ? 0
               : set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                           ET_KERNEL_CODE_INVALID_BUFFER, operation,
                           "empty shape-range list has storage");
  }
  if (source == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, operation,
                     "nonempty shape-range list has null storage");
  }
  if ((uintptr_t)source % _Alignof(et_kernel_shape_range_v1) != 0u ||
      (uintptr_t)source > UINTPTR_MAX - count * sizeof(*source)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, operation,
                     "shape-range list storage is misaligned or overflows");
  }
  for (size_t range_index = 0; range_index < count; range_index++) {
    const size_t rank = source[range_index].rank;
    if (rank > ET_KERNEL_MAX_RANK ||
        rank > SIZE_MAX / sizeof(*source[range_index].dimensions) ||
        (rank > 0u && source[range_index].dimensions == NULL)) {
      return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                       rank > ET_KERNEL_MAX_RANK
                           ? ET_KERNEL_CODE_INTEGER_OVERFLOW
                           : ET_KERNEL_CODE_INVALID_SHAPE,
                       operation, "capability contains an invalid shape range");
    }
    if (rank > 0u &&
        ((uintptr_t)source[range_index].dimensions %
                 _Alignof(et_kernel_dimension_range_v1) !=
             0u ||
         (uintptr_t)source[range_index].dimensions >
             UINTPTR_MAX - rank * sizeof(*source[range_index].dimensions))) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_BUFFER, operation,
                       "shape dimension storage is misaligned or overflows");
    }
    for (size_t dimension_index = 0; dimension_index < rank;
         dimension_index++) {
      const et_kernel_dimension_range_v1 *dimension =
          &source[range_index].dimensions[dimension_index];
      if (dimension->maximum_unbounded > 1u ||
          (!dimension->maximum_unbounded &&
           dimension->minimum > dimension->maximum)) {
        return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         ET_KERNEL_CODE_INVALID_SHAPE, operation,
                         "capability contains an invalid shape range");
      }
    }
    for (size_t earlier = 0; earlier < range_index; earlier++) {
      if (shape_range_compare(&source[earlier], &source[range_index]) == 0) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_DUPLICATE_ENTRY, operation,
                         "capability contains a duplicate shape range");
      }
    }
  }
  return 0;
}

static int32_t copy_shape_ranges(const char *operation, size_t count,
                                 const et_kernel_shape_range_v1 *source,
                                 const et_kernel_shape_range_v1 **destination,
                                 et_kernel_error *error) {
  et_kernel_shape_range_v1 *copy = NULL;
  int32_t result;
  *destination = NULL;
  result = validate_shape_ranges(operation, count, source, error);
  if (result != 0 || count == 0u) {
    return result;
  }
  copy = (et_kernel_shape_range_v1 *)calloc(count, sizeof(*copy));
  if (copy == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_ALLOCATION_FAILED, operation,
                     "cannot allocate shape-range list");
  }
  for (size_t range_index = 0; range_index < count; range_index++) {
    const size_t rank = source[range_index].rank;
    et_kernel_dimension_range_v1 *dimensions = NULL;
    copy[range_index].rank = rank;
    if (rank == 0) {
      continue;
    }
    dimensions = (et_kernel_dimension_range_v1 *)calloc(
        rank, sizeof(*dimensions));
    if (dimensions == NULL) {
      for (size_t index = 0; index < range_index; index++) {
        free((void *)copy[index].dimensions);
      }
      free(copy);
      return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                       ET_KERNEL_CODE_ALLOCATION_FAILED, operation,
                       "cannot allocate shape dimensions");
    }
    for (size_t dimension_index = 0; dimension_index < rank;
         dimension_index++) {
      const et_kernel_dimension_range_v1 dimension =
          source[range_index].dimensions[dimension_index];
      dimensions[dimension_index] = dimension;
      memset(dimensions[dimension_index].reserved, 0,
             sizeof(dimensions[dimension_index].reserved));
    }
    copy[range_index].dimensions = dimensions;
  }
  qsort(copy, count, sizeof(*copy), shape_range_compare);
  *destination = copy;
  return 0;
}

static int32_t validate_capability(const et_kernel_capability_v1 *source,
                                   et_kernel_error *error) {
  int32_t result;
  if (source == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-discover",
                     "provider capability is null");
  }
  if (source->struct_size < ET_KERNEL_CAPABILITY_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                     "capability-discover",
                     "provider capability descriptor is truncated");
  }
  if (!valid_symbol(source->name) || !valid_symbol(source->implementation) ||
      !valid_utf8(source->version, ET_KERNEL_MAX_METADATA_BYTES) ||
      !valid_utf8(source->evidence, ET_KERNEL_MAX_METADATA_BYTES)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_TEXT, "capability-discover",
                     "provider capability metadata is invalid");
  }
  if (source->status > ET_KERNEL_CAPABILITY_UNSUPPORTED ||
      source->deterministic > 1u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_ENUM, "capability-discover",
                     "provider capability status is invalid");
  }
  result = validate_string_array("capability-discover",
                                 source->operation_count,
                                 source->operations, error);
  if (result != 0) {
    return result;
  }
  result = validate_string_array("capability-discover", source->dtype_count,
                                 source->dtypes, error);
  if (result != 0) {
    return result;
  }
  result = validate_string_array("capability-discover", source->device_count,
                                 source->devices, error);
  if (result != 0) {
    return result;
  }
  result = validate_shape_ranges("capability-discover",
                                 source->shape_range_count,
                                 source->shape_ranges, error);
  if (result != 0) {
    return result;
  }
  if (source->status == ET_KERNEL_CAPABILITY_VERIFIED &&
      (source->operation_count == 0u || source->dtype_count == 0u ||
       source->device_count == 0u || source->shape_range_count == 0u)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_SHAPE, "capability-discover",
                     "verified capability has incomplete constraints");
  }
  return 0;
}

static int32_t copy_capability(const et_kernel_capability_v1 *source,
                               et_kernel_capability_v1 *destination,
                               et_kernel_error *error) {
  int32_t result;
  memset(destination, 0, sizeof(*destination));
  result = validate_capability(source, error);
  if (result != 0) {
    return result;
  }
  destination->struct_size = sizeof(*destination);
  destination->status = source->status;
  destination->deterministic = source->deterministic;
  destination->name = duplicate_text(source->name);
  destination->implementation = duplicate_text(source->implementation);
  destination->version = duplicate_text(source->version);
  destination->evidence = duplicate_text(source->evidence);
  if (destination->name == NULL || destination->implementation == NULL ||
      destination->version == NULL || destination->evidence == NULL) {
    free_capability(destination);
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_ALLOCATION_FAILED, "capability-discover",
                     "cannot copy provider capability metadata");
  }
  destination->operation_count = source->operation_count;
  result = copy_string_array("capability-discover", source->operation_count,
                             source->operations, &destination->operations,
                             error);
  if (result != 0) {
    goto failure;
  }
  destination->dtype_count = source->dtype_count;
  result = copy_string_array("capability-discover", source->dtype_count,
                             source->dtypes, &destination->dtypes, error);
  if (result != 0) {
    goto failure;
  }
  destination->device_count = source->device_count;
  result = copy_string_array("capability-discover", source->device_count,
                             source->devices, &destination->devices, error);
  if (result != 0) {
    goto failure;
  }
  destination->shape_range_count = source->shape_range_count;
  result = copy_shape_ranges("capability-discover", source->shape_range_count,
                             source->shape_ranges,
                             &destination->shape_ranges, error);
  if (result != 0) {
    goto failure;
  }
  return 0;

failure:
  free_capability(destination);
  return result;
}

static int32_t create_baseline(et_kernel_runtime **output,
                               et_kernel_error *error) {
  et_kernel_runtime *runtime = NULL;
  if (output == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-discover",
                     "runtime output pointer is null");
  }
  *output = NULL;
  runtime = (et_kernel_runtime *)calloc(1, sizeof(*runtime));
  if (runtime == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_ALLOCATION_FAILED, "capability-discover",
                     "cannot allocate capability runtime");
  }
  runtime->capability_count = ET_ARRAY_COUNT(baseline_names);
  runtime->capabilities = (et_kernel_capability_v1 *)calloc(
      runtime->capability_count, sizeof(*runtime->capabilities));
  if (runtime->capabilities == NULL) {
    et_kernel_runtime_destroy(runtime);
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_ALLOCATION_FAILED, "capability-discover",
                     "cannot allocate baseline capabilities");
  }
  for (size_t index = 0; index < runtime->capability_count; index++) {
    et_kernel_capability_v1 source;
    int32_t result;
    memset(&source, 0, sizeof(source));
    source.struct_size = sizeof(source);
    source.name = baseline_names[index];
    source.status = ET_KERNEL_CAPABILITY_UNVERIFIED;
    source.implementation = "eshkol-core";
    source.version = "1.3.4-evolve";
    source.evidence = "R0-merged-through-PR15:untested-with-reason";
    result = copy_capability(&source, &runtime->capabilities[index], error);
    if (result != 0) {
      et_kernel_runtime_destroy(runtime);
      return result;
    }
  }
  *output = runtime;
  et_kernel_error_clear(error);
  return 0;
}

int32_t et_kernel_runtime_baseline(et_kernel_runtime **runtime,
                                   et_kernel_error *error) {
  return create_baseline(runtime, error);
}

static int32_t validate_provider_capability_table(
    const et_kernel_provider_v1 *provider, et_kernel_error *error) {
  const size_t alignment = _Alignof(et_kernel_capability_v1);
  if (provider->capability_count == 0u) {
    if (provider->capabilities != NULL || provider->capability_stride != 0u ||
        provider->capability_bytes != 0u) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_BUFFER, "capability-discover",
                       "empty capability table must have null storage and zero stride");
    }
    return 0;
  }
  if (provider->capabilities == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-discover",
                     "nonempty capability table has null storage");
  }
  if (provider->capability_stride < ET_KERNEL_CAPABILITY_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                     "capability-discover",
                     "capability table stride is smaller than the v1.0 prefix");
  }
  if (provider->capability_stride % alignment != 0u ||
      (uintptr_t)provider->capabilities % alignment != 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "capability-discover",
                     "capability table storage or stride is misaligned");
  }
  if (provider->capability_count > SIZE_MAX / provider->capability_stride ||
      provider->capability_count * provider->capability_stride !=
          provider->capability_bytes) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "capability-discover",
                     "capability table byte span is invalid or overflows");
  }
  if ((uintptr_t)provider->capabilities >
      UINTPTR_MAX - provider->capability_bytes) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "capability-discover",
                     "capability table address range overflows");
  }
  return 0;
}

static int32_t provider_capability_at(
    const et_kernel_provider_v1 *provider, size_t index,
    const et_kernel_capability_v1 **capability, et_kernel_error *error) {
  size_t offset;
  if (capability == NULL || index >= provider->capability_count ||
      (provider->capability_stride != 0u &&
       index > SIZE_MAX / provider->capability_stride)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "capability-discover",
                     "capability table offset is invalid");
  }
  offset = index * provider->capability_stride;
  if (offset > provider->capability_bytes -
                   ET_KERNEL_CAPABILITY_V1_0_SIZE ||
      (uintptr_t)provider->capabilities > UINTPTR_MAX - offset) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "capability-discover",
                     "capability table offset exceeds its byte span");
  }
  *capability = (const et_kernel_capability_v1 *)(
      (const unsigned char *)provider->capabilities + offset);
  if ((*capability)->struct_size < ET_KERNEL_CAPABILITY_V1_0_SIZE ||
      (*capability)->struct_size > provider->capability_stride) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                     "capability-discover",
                     "capability descriptor does not fit its table stride");
  }
  return 0;
}

static int32_t validate_provider_capabilities(
    const et_kernel_provider_v1 *provider, int *has_verified_capability,
    et_kernel_error *error) {
  *has_verified_capability = 0;
  for (size_t index = 0; index < provider->capability_count; index++) {
    const et_kernel_capability_v1 *capability;
    int32_t result = provider_capability_at(provider, index, &capability,
                                            error);
    if (result != 0) {
      return result;
    }
    result = validate_capability(capability, error);
    if (result != 0) {
      return result;
    }
    if (capability->status == ET_KERNEL_CAPABILITY_VERIFIED) {
      *has_verified_capability = 1;
    }
    for (size_t earlier = 0; earlier < index; earlier++) {
      const et_kernel_capability_v1 *earlier_capability;
      result = provider_capability_at(provider, earlier, &earlier_capability,
                                      error);
      if (result != 0) {
        return result;
      }
      if (strcmp(earlier_capability->name, capability->name) == 0) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_DUPLICATE_ENTRY,
                         "capability-discover",
                         "provider contains duplicate capabilities");
      }
    }
  }
  return 0;
}

static int32_t merge_provider(et_kernel_runtime *runtime,
                              const et_kernel_provider_v1 *provider,
                              et_kernel_error *error) {
  for (size_t provider_index = 0; provider_index < provider->capability_count;
       provider_index++) {
    const et_kernel_capability_v1 *source;
    et_kernel_capability_v1 copied;
    size_t destination_index = runtime->capability_count;
    int32_t result;
    result = provider_capability_at(provider, provider_index, &source, error);
    if (result != 0) {
      return result;
    }
    result = copy_capability(source, &copied, error);
    if (result != 0) {
      return result;
    }
    for (size_t index = 0; index < runtime->capability_count; index++) {
      if (strcmp(runtime->capabilities[index].name, copied.name) == 0) {
        destination_index = index;
        break;
      }
    }
    if (destination_index == runtime->capability_count) {
      et_kernel_capability_v1 *resized;
      if (runtime->capability_count == SIZE_MAX / sizeof(*resized)) {
        free_capability(&copied);
        return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                         ET_KERNEL_CODE_INTEGER_OVERFLOW,
                         "capability-discover", "capability count overflows");
      }
      resized = (et_kernel_capability_v1 *)realloc(
          runtime->capabilities,
          (runtime->capability_count + 1u) * sizeof(*resized));
      if (resized == NULL) {
        free_capability(&copied);
        return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                         ET_KERNEL_CODE_ALLOCATION_FAILED,
                         "capability-discover",
                         "cannot grow capability report");
      }
      runtime->capabilities = resized;
      runtime->capability_count++;
    } else {
      free_capability(&runtime->capabilities[destination_index]);
    }
    runtime->capabilities[destination_index] = copied;
  }
  qsort(runtime->capabilities, runtime->capability_count,
        sizeof(*runtime->capabilities), capability_compare);
  return 0;
}

int32_t et_kernel_runtime_discover(et_kernel_provider_resolver_v1 resolver,
                                   void *context,
                                   et_kernel_runtime **runtime_output,
                                   et_kernel_error *error) {
  et_kernel_runtime *runtime = NULL;
  const et_kernel_provider_v1 *provider;
  int has_verified_capability = 0;
  int32_t result;
  if (runtime_output == NULL || resolver == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-discover",
                     "resolver and runtime output are required");
  }
  *runtime_output = NULL;
  provider = resolver(context, ET_KERNEL_PROVIDER_SYMBOL_V1);
  if (provider == NULL) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_SYMBOL_MISSING, "capability-discover",
                     "required provider symbol is unavailable");
  }
  if ((uintptr_t)provider % _Alignof(et_kernel_provider_v1) != 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "capability-discover",
                     "provider descriptor is misaligned");
  }
  if (provider->struct_size < ET_KERNEL_PROVIDER_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE,
                     "capability-discover",
                     "provider descriptor is truncated");
  }
  if (provider->abi_major != ET_KERNEL_ABI_MAJOR) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_ABI_MAJOR_MISMATCH,
                     "capability-discover", "provider ABI major is unsupported");
  }
  if ((provider->required_features &
       ~ET_KERNEL_PROVIDER_KNOWN_REQUIRED_FEATURES) != 0u) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_UNKNOWN_REQUIRED_FEATURE,
                     "capability-discover",
                     "provider requires unknown ABI features");
  }
  if (!valid_symbol(provider->name) ||
      !valid_utf8(provider->version, ET_KERNEL_MAX_METADATA_BYTES) ||
      !valid_utf8(provider->evidence, ET_KERNEL_MAX_METADATA_BYTES)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_TEXT, "capability-discover",
                     "provider descriptor metadata is invalid");
  }
  if (provider->capability_count > ET_KERNEL_MAX_CAPABILITIES) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW,
                     "capability-discover",
                     "provider capability count exceeds ABI v1 limit");
  }
  result = validate_provider_capability_table(provider, error);
  if (result != 0) {
    return result;
  }
  result = validate_provider_capabilities(provider, &has_verified_capability,
                                          error);
  if (result != 0) {
    return result;
  }
  if (has_verified_capability &&
      (provider->validate_call == NULL || provider->invoke_call == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-discover",
                     "verified provider requires validate and invoke callbacks");
  }
  result = create_baseline(&runtime, error);
  if (result != 0) {
    return result;
  }
  result = merge_provider(runtime, provider, error);
  if (result != 0) {
    et_kernel_runtime_destroy(runtime);
    return result;
  }
  runtime->validate_call = provider->validate_call;
  runtime->invoke_call = provider->invoke_call;
  runtime->provider_minor = provider->abi_minor;
  runtime->provider_present = 1u;
  *runtime_output = runtime;
  et_kernel_error_clear(error);
  return 0;
}

size_t et_kernel_runtime_capability_count(const et_kernel_runtime *runtime) {
  return runtime == NULL ? 0u : runtime->capability_count;
}

const et_kernel_capability_v1 *et_kernel_runtime_capability_at(
    const et_kernel_runtime *runtime, size_t index) {
  if (runtime == NULL || index >= runtime->capability_count) {
    return NULL;
  }
  return &runtime->capabilities[index];
}

const et_kernel_capability_v1 *et_kernel_runtime_capability_find(
    const et_kernel_runtime *runtime, const char *name) {
  if (runtime == NULL || !valid_symbol(name)) {
    return NULL;
  }
  size_t low = 0;
  size_t high = runtime->capability_count;
  while (low < high) {
    const size_t middle = low + (high - low) / 2u;
    const int order = strcmp(name, runtime->capabilities[middle].name);
    if (order == 0) {
      return &runtime->capabilities[middle];
    }
    if (order < 0) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return NULL;
}

static int string_array_contains(size_t count, const char *const *values,
                                 const char *needle) {
  for (size_t index = 0; index < count; index++) {
    if (strcmp(values[index], needle) == 0) {
      return 1;
    }
  }
  return 0;
}

static int shape_matches(const et_kernel_capability_v1 *capability,
                         const et_kernel_request_v1 *request) {
  for (size_t range_index = 0;
       range_index < capability->shape_range_count; range_index++) {
    const et_kernel_shape_range_v1 *range =
        &capability->shape_ranges[range_index];
    int matches = range->rank == request->rank;
    for (size_t dimension = 0; matches && dimension < range->rank;
         dimension++) {
      const et_kernel_dimension_range_v1 *constraint =
          &range->dimensions[dimension];
      matches = request->shape[dimension] >= constraint->minimum &&
                (constraint->maximum_unbounded ||
                 request->shape[dimension] <= constraint->maximum);
    }
    if (matches) {
      return 1;
    }
  }
  return 0;
}

static int32_t validate_request(const et_kernel_request_v1 *request,
                                et_kernel_error *error) {
  if (request == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-request",
                     "capability request is null");
  }
  if ((uintptr_t)request % _Alignof(et_kernel_request_v1) != 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "capability-request",
                     "capability request is misaligned");
  }
  if (request->struct_size < ET_KERNEL_REQUEST_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "capability-request",
                     "capability request descriptor is truncated");
  }
  if (!valid_symbol(request->operation) || !valid_symbol(request->dtype) ||
      !valid_symbol(request->device) || request->deterministic > 1u ||
      request->rank > ET_KERNEL_MAX_RANK ||
      (request->rank > 0u && request->shape == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_TEXT, "capability-request",
                     "capability request is malformed");
  }
  if (request->rank > 0u &&
      ((uintptr_t)request->shape % _Alignof(uint64_t) != 0u ||
       (uintptr_t)request->shape >
           UINTPTR_MAX - request->rank * sizeof(*request->shape))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "capability-request",
                     "capability request shape storage is invalid");
  }
  return 0;
}

int32_t et_kernel_runtime_capability_require(
    const et_kernel_runtime *runtime, const char *capability_name,
    const et_kernel_request_v1 *request,
    const et_kernel_capability_v1 **entry, et_kernel_error *error) {
  const et_kernel_capability_v1 *capability;
  int32_t result;
  if (entry != NULL) {
    *entry = NULL;
  }
  if (runtime == NULL || !valid_symbol(capability_name)) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-require",
                     "runtime or capability name is invalid");
  }
  result = validate_request(request, error);
  if (result != 0) {
    return result;
  }
  capability = et_kernel_runtime_capability_find(runtime, capability_name);
  if (capability == NULL ||
      capability->status != ET_KERNEL_CAPABILITY_VERIFIED ||
      !string_array_contains(capability->operation_count,
                             capability->operations, request->operation) ||
      !string_array_contains(capability->dtype_count, capability->dtypes,
                             request->dtype) ||
      !string_array_contains(capability->device_count, capability->devices,
                             request->device) ||
      !shape_matches(capability, request) ||
      (request->deterministic && !capability->deterministic)) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED,
                     "capability-require",
                     "no verified capability matches the exact request");
  }
  if (entry != NULL) {
    *entry = capability;
  }
  et_kernel_error_clear(error);
  return 0;
}

static size_t dtype_size(const char *dtype) {
  if (strcmp(dtype, "bool") == 0) {
    return 1u;
  }
  if (strcmp(dtype, "f16") == 0 || strcmp(dtype, "bf16") == 0) {
    return 2u;
  }
  if (strcmp(dtype, "f32") == 0) {
    return 4u;
  }
  if (strcmp(dtype, "i64") == 0) {
    return 8u;
  }
  return 0u;
}

static int32_t validate_tensor(const et_kernel_tensor_view_v1 *tensor,
                               const char *request_device,
                               et_kernel_error *error) {
  size_t bytes;
  size_t element_size;
  if (tensor == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "kernel-dispatch",
                     "tensor view is null");
  }
  if (tensor->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "kernel-dispatch",
                     "tensor view descriptor is truncated");
  }
  if (!valid_symbol(tensor->dtype) || !valid_symbol(tensor->device)) {
    return set_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, "kernel-dispatch",
                     "tensor dtype or device is unsupported by ABI v1");
  }
  element_size = dtype_size(tensor->dtype);
  if (element_size == 0u) {
    return set_error(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_TEXT, "kernel-dispatch",
                     "tensor dtype or device is unsupported by ABI v1");
  }
  if (strcmp(tensor->device, request_device) != 0) {
    return set_error(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                     "tensor device differs from the request device");
  }
  if (tensor->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      tensor->offset_bytes != 0u) {
    return set_error(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                     "tensor must be dense row-major with zero offset");
  }
  if (tensor->rank > 0u && tensor->shape == NULL) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_SHAPE, "kernel-dispatch",
                     "tensor rank requires a shape pointer");
  }
  if (tensor->rank > ET_KERNEL_MAX_RANK) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                     "tensor rank exceeds ABI v1 limit");
  }
  if (tensor->rank > 0u &&
      ((uintptr_t)tensor->shape % _Alignof(uint64_t) != 0u ||
       (uintptr_t)tensor->shape >
           UINTPTR_MAX - tensor->rank * sizeof(*tensor->shape))) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                     "tensor shape storage is misaligned or overflows");
  }
  bytes = element_size;
  for (size_t dimension = 0; dimension < tensor->rank; dimension++) {
    if (tensor->shape[dimension] == 0u) {
      bytes = 0u;
      break;
    }
    if (tensor->shape[dimension] > SIZE_MAX ||
        bytes > SIZE_MAX / (size_t)tensor->shape[dimension]) {
      return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                       ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                       "tensor byte size overflows address space");
    }
    bytes *= (size_t)tensor->shape[dimension];
  }
  if (tensor->byte_length != bytes || (bytes > 0u && tensor->data == NULL)) {
    return set_error(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                     "tensor byte length does not match dtype and shape");
  }
  return 0;
}

static int buffers_overlap(const et_kernel_tensor_view_v1 *left,
                           const et_kernel_tensor_view_v1 *right) {
  const uintptr_t left_start = (uintptr_t)left->data;
  const uintptr_t right_start = (uintptr_t)right->data;
  uintptr_t left_end;
  uintptr_t right_end;
  if (left->byte_length == 0u || right->byte_length == 0u) {
    return 0;
  }
  if (left_start > UINTPTR_MAX - left->byte_length ||
      right_start > UINTPTR_MAX - right->byte_length) {
    return 1;
  }
  left_end = left_start + left->byte_length;
  right_end = right_start + right->byte_length;
  return left_start < right_end && right_start < left_end;
}

static int32_t validate_tensor_table(size_t count, size_t stride, size_t bytes,
                                     const void *base, const char *which,
                                     et_kernel_error *error) {
  const size_t alignment = _Alignof(et_kernel_tensor_view_v1);
  if (count == 0u) {
    if (base != NULL || stride != 0u || bytes != 0u) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                       which);
    }
    return 0;
  }
  if (base == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "kernel-dispatch", which);
  }
  if (stride < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "kernel-dispatch",
                     which);
  }
  if (stride % alignment != 0u || (uintptr_t)base % alignment != 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch", which);
  }
  if (count > SIZE_MAX / stride || count * stride != bytes ||
      (uintptr_t)base > UINTPTR_MAX - bytes) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                     which);
  }
  return 0;
}

static int32_t tensor_table_at(size_t count, size_t stride, size_t bytes,
                               const void *base, size_t index,
                               const et_kernel_tensor_view_v1 **tensor,
                               et_kernel_error *error) {
  size_t offset;
  if (tensor == NULL || index >= count || index > SIZE_MAX / stride) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                     "tensor table offset is invalid");
  }
  offset = index * stride;
  if (offset > bytes - ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      (uintptr_t)base > UINTPTR_MAX - offset) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                     "tensor table offset exceeds its byte span");
  }
  *tensor = (const et_kernel_tensor_view_v1 *)((const unsigned char *)base +
                                                offset);
  if ((*tensor)->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      (*tensor)->struct_size > stride) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "kernel-dispatch",
                     "tensor descriptor does not fit its table stride");
  }
  return 0;
}

static int32_t validate_call(const et_kernel_call_v1 *call,
                             et_kernel_error *error) {
  int32_t result;
  if (call == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "kernel-dispatch",
                     "kernel call is null");
  }
  if ((uintptr_t)call % _Alignof(et_kernel_call_v1) != 0u) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INVALID_BUFFER, "kernel-dispatch",
                     "kernel call descriptor is misaligned");
  }
  if (call->struct_size < ET_KERNEL_CALL_V1_0_SIZE) {
    return set_error(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                     ET_KERNEL_CODE_INVALID_STRUCT_SIZE, "kernel-dispatch",
                     "kernel call descriptor is truncated");
  }
  if (!valid_symbol(call->capability) || call->request == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "kernel-dispatch",
                     "kernel call arrays or capability are invalid");
  }
  if (call->input_count > ET_KERNEL_MAX_TENSORS ||
      call->output_count > ET_KERNEL_MAX_TENSORS) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW, "kernel-dispatch",
                     "kernel tensor count exceeds ABI v1 limit");
  }
  result = validate_tensor_table(call->input_count, call->input_stride,
                                 call->input_bytes, call->inputs,
                                 "input tensor table is invalid", error);
  if (result != 0) {
    return result;
  }
  result = validate_tensor_table(call->output_count, call->output_stride,
                                 call->output_bytes, call->outputs,
                                 "output tensor table is invalid", error);
  if (result != 0) {
    return result;
  }
  result = validate_request(call->request, error);
  if (result != 0) {
    return result;
  }
  for (size_t index = 0; index < call->input_count; index++) {
    const et_kernel_tensor_view_v1 *input;
    result = tensor_table_at(call->input_count, call->input_stride,
                             call->input_bytes, call->inputs, index, &input,
                             error);
    if (result == 0) {
      result = validate_tensor(input, call->request->device, error);
    }
    if (result != 0) {
      return result;
    }
  }
  for (size_t index = 0; index < call->output_count; index++) {
    const et_kernel_tensor_view_v1 *output;
    result = tensor_table_at(call->output_count, call->output_stride,
                             call->output_bytes, call->outputs, index, &output,
                             error);
    if (result == 0) {
      result = validate_tensor(output, call->request->device, error);
    }
    if (result != 0) {
      return result;
    }
    for (size_t input_index = 0; input_index < call->input_count;
         input_index++) {
      const et_kernel_tensor_view_v1 *input;
      result = tensor_table_at(call->input_count, call->input_stride,
                               call->input_bytes, call->inputs, input_index,
                               &input, error);
      if (result != 0) {
        return result;
      }
      if (buffers_overlap(output, input)) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_ALIASING_OUTPUT, "kernel-dispatch",
                         "output storage aliases borrowed input storage");
      }
    }
    for (size_t output_index = 0; output_index < index; output_index++) {
      const et_kernel_tensor_view_v1 *earlier_output;
      result = tensor_table_at(call->output_count, call->output_stride,
                               call->output_bytes, call->outputs, output_index,
                               &earlier_output, error);
      if (result != 0) {
        return result;
      }
      if (buffers_overlap(output, earlier_output)) {
        return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         ET_KERNEL_CODE_ALIASING_OUTPUT, "kernel-dispatch",
                         "output storage aliases another output");
      }
    }
  }
  return 0;
}

static int provider_error_is_valid(int32_t result,
                                   const et_kernel_error *error) {
  if (error == NULL || error->category < ET_KERNEL_ERROR_INVALID_ARGUMENT ||
      error->category > ET_KERNEL_ERROR_INTERNAL ||
      error->code < ET_KERNEL_CODE_NULL_ARGUMENT ||
      error->code > ET_KERNEL_CODE_PROVIDER_REJECTED ||
      result != (int32_t)error->category ||
      memchr(error->operation, '\0', sizeof(error->operation)) == NULL ||
      memchr(error->message, '\0', sizeof(error->message)) == NULL ||
      error->operation[0] == '\0' || error->message[0] == '\0' ||
      !valid_utf8(error->operation, sizeof(error->operation) - 1u) ||
      !valid_utf8(error->message, sizeof(error->message) - 1u)) {
    return 0;
  }
  return 1;
}

int32_t et_kernel_runtime_dispatch(const et_kernel_runtime *runtime,
                                   const et_kernel_call_v1 *call,
                                   et_kernel_error *error) {
  int32_t result;
  if (runtime == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "kernel-dispatch",
                     "capability runtime is null");
  }
  result = validate_call(call, error);
  if (result != 0) {
    return result;
  }
  result = et_kernel_runtime_capability_require(
      runtime, call->capability, call->request, NULL, error);
  if (result != 0) {
    return result;
  }
  if (!runtime->provider_present || runtime->validate_call == NULL ||
      runtime->invoke_call == NULL) {
    return set_error(error, ET_KERNEL_ERROR_UNSUPPORTED,
                     ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED, "kernel-dispatch",
                     "no registered provider can dispatch the request");
  }
  et_kernel_error_clear(error);
  result = runtime->validate_call(call, error);
  if (result != 0) {
    if (!provider_error_is_valid(result, error)) {
      return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                       ET_KERNEL_CODE_PROVIDER_REJECTED, "kernel-dispatch",
                       "provider returned a malformed validation error");
    }
    return (int32_t)error->category;
  }
  runtime->invoke_call(call);
  et_kernel_error_clear(error);
  return 0;
}

typedef struct json_writer {
  char *buffer;
  size_t capacity;
  size_t length;
  int overflow;
} json_writer;

static void json_bytes(json_writer *writer, const char *bytes, size_t length) {
  if (writer->overflow || length > SIZE_MAX - writer->length) {
    writer->overflow = 1;
    return;
  }
  if (writer->buffer != NULL && writer->length + length <= writer->capacity) {
    memcpy(writer->buffer + writer->length, bytes, length);
  }
  writer->length += length;
}

static void json_literal(json_writer *writer, const char *literal) {
  json_bytes(writer, literal, strlen(literal));
}

static void json_string(json_writer *writer, const char *text) {
  static const char hexadecimal[] = "0123456789abcdef";
  const unsigned char *cursor = (const unsigned char *)text;
  json_literal(writer, "\"");
  while (*cursor != 0) {
    if (*cursor == '"' || *cursor == '\\') {
      const char escaped[2] = {'\\', (char)*cursor};
      json_bytes(writer, escaped, sizeof(escaped));
    } else if (*cursor < 0x20u) {
      char escaped[6] = {'\\', 'u', '0', '0',
                         hexadecimal[*cursor >> 4u],
                         hexadecimal[*cursor & 0x0fu]};
      json_bytes(writer, escaped, sizeof(escaped));
    } else {
      json_bytes(writer, (const char *)cursor, 1u);
    }
    cursor++;
  }
  json_literal(writer, "\"");
}

static void json_u64(json_writer *writer, uint64_t value) {
  char digits[21];
  size_t length = 0;
  do {
    digits[length++] = (char)('0' + value % 10u);
    value /= 10u;
  } while (value != 0u);
  for (size_t index = 0; index < length / 2u; index++) {
    const char temporary = digits[index];
    digits[index] = digits[length - index - 1u];
    digits[length - index - 1u] = temporary;
  }
  json_bytes(writer, digits, length);
}

static void json_string_array(json_writer *writer, size_t count,
                              const char *const *values) {
  json_literal(writer, "[");
  for (size_t index = 0; index < count; index++) {
    if (index != 0u) {
      json_literal(writer, ",");
    }
    json_string(writer, values[index]);
  }
  json_literal(writer, "]");
}

static const char *status_text(et_kernel_capability_status status) {
  switch (status) {
  case ET_KERNEL_CAPABILITY_VERIFIED:
    return "verified";
  case ET_KERNEL_CAPABILITY_UNSUPPORTED:
    return "unsupported";
  case ET_KERNEL_CAPABILITY_UNVERIFIED:
  default:
    return "unverified";
  }
}

static void json_shape_ranges(json_writer *writer,
                              const et_kernel_capability_v1 *capability) {
  json_literal(writer, "[");
  for (size_t range = 0; range < capability->shape_range_count; range++) {
    if (range != 0u) {
      json_literal(writer, ",");
    }
    json_literal(writer, "[");
    for (size_t dimension = 0;
         dimension < capability->shape_ranges[range].rank; dimension++) {
      const et_kernel_dimension_range_v1 *constraint =
          &capability->shape_ranges[range].dimensions[dimension];
      if (dimension != 0u) {
        json_literal(writer, ",");
      }
      json_literal(writer, "[");
      json_u64(writer, constraint->minimum);
      json_literal(writer, ",");
      if (constraint->maximum_unbounded) {
        json_literal(writer, "null");
      } else {
        json_u64(writer, constraint->maximum);
      }
      json_literal(writer, "]");
    }
    json_literal(writer, "]");
  }
  json_literal(writer, "]");
}

static void write_report_json(json_writer *writer,
                              const et_kernel_runtime *runtime) {
  json_literal(writer, "{\"abi\":{\"major\":");
  json_u64(writer, ET_KERNEL_ABI_MAJOR);
  json_literal(writer, ",\"minor\":");
  json_u64(writer, ET_KERNEL_ABI_MINOR);
  json_literal(writer, "},\"entries\":[");
  for (size_t index = 0; index < runtime->capability_count; index++) {
    const et_kernel_capability_v1 *capability =
        &runtime->capabilities[index];
    if (index != 0u) {
      json_literal(writer, ",");
    }
    json_literal(writer, "{\"constraints\":{\"devices\":");
    json_string_array(writer, capability->device_count, capability->devices);
    json_literal(writer, ",\"dtypes\":");
    json_string_array(writer, capability->dtype_count, capability->dtypes);
    json_literal(writer, ",\"operations\":");
    json_string_array(writer, capability->operation_count,
                      capability->operations);
    json_literal(writer, ",\"shape_ranges\":");
    json_shape_ranges(writer, capability);
    json_literal(writer, "},\"deterministic\":");
    json_literal(writer, capability->deterministic ? "true" : "false");
    json_literal(writer, ",\"evidence\":");
    json_string(writer, capability->evidence);
    json_literal(writer, ",\"implementation\":");
    json_string(writer, capability->implementation);
    json_literal(writer, ",\"name\":");
    json_string(writer, capability->name);
    json_literal(writer, ",\"status\":");
    json_string(writer, status_text(capability->status));
    json_literal(writer, ",\"version\":");
    json_string(writer, capability->version);
    json_literal(writer, "}");
  }
  json_literal(writer,
               "],\"format\":\"eshkol-kernel-capabilities\","
               "\"process_local\":true,\"provider_abi\":");
  if (!runtime->provider_present) {
    json_literal(writer, "null");
  } else {
    json_literal(writer, "{\"major\":1,\"minor\":");
    json_u64(writer, runtime->provider_minor);
    json_literal(writer, "}");
  }
  json_literal(writer, ",\"version\":1}\n");
}

int32_t et_kernel_runtime_report_json(const et_kernel_runtime *runtime,
                                      char *buffer, size_t capacity,
                                      size_t *required_capacity,
                                      et_kernel_error *error) {
  json_writer counter = {0};
  json_writer output = {0};
  size_t required;
  if (runtime == NULL || required_capacity == NULL) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_NULL_ARGUMENT, "capability-report-json",
                     "runtime and required-capacity output are required");
  }
  write_report_json(&counter, runtime);
  if (counter.overflow || counter.length == SIZE_MAX) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW,
                     "capability-report-json", "JSON report size overflows");
  }
  required = counter.length + 1u;
  *required_capacity = required;
  if (buffer == NULL) {
    if (capacity != 0u) {
      return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                       ET_KERNEL_CODE_NULL_ARGUMENT,
                       "capability-report-json",
                       "nonzero capacity requires a buffer");
    }
    et_kernel_error_clear(error);
    return 0;
  }
  if (capacity < required) {
    return set_error(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                     ET_KERNEL_CODE_BUFFER_TOO_SMALL,
                     "capability-report-json",
                     "JSON output buffer is too small");
  }
  output.buffer = buffer;
  output.capacity = capacity;
  write_report_json(&output, runtime);
  if (output.overflow || output.length != counter.length) {
    return set_error(error, ET_KERNEL_ERROR_INTERNAL,
                     ET_KERNEL_CODE_INTEGER_OVERFLOW,
                     "capability-report-json",
                     "JSON output length changed unexpectedly");
  }
  buffer[output.length] = '\0';
  et_kernel_error_clear(error);
  return 0;
}
