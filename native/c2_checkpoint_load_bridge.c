#include "c2_checkpoint_load_bridge.h"

#include "c2_checkpoint_core.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define C2_PATH_MAX ((size_t)4096u)
#define C2_CONFIG_FINGERPRINT_BYTES ((size_t)93u)

typedef struct c2_span {
  const uint8_t *header;
  uint8_t *bytes;
  size_t allocation;
  size_t length;
} c2_span;

#ifdef ET_C2_CHECKPOINT_LOAD_TESTING
static int c2_fail_after_validate;
static uint64_t c2_copy_count;

void et_c2_checkpoint_load_test_reset_v1(void) {
  c2_fail_after_validate = 0;
  c2_copy_count = 0u;
}

void et_c2_checkpoint_load_test_fail_after_validate_v1(int64_t enabled) {
  c2_fail_after_validate = enabled != 0;
}

uint64_t et_c2_checkpoint_load_test_copy_count_v1(void) {
  return c2_copy_count;
}
#endif

static void put_u64(uint8_t *p, uint64_t value) {
  unsigned i;
  for (i = 0u; i < 8u; ++i) p[i] = (uint8_t)(value >> (8u * i));
}

static uint32_t get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static uint64_t get_u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

static int span_from_bytevector(void *opaque, size_t maximum, size_t exact,
                                c2_span *out) {
  int64_t declared;
  uintptr_t start;
  if (opaque == NULL || out == NULL) return 0;
  memcpy(&declared, opaque, sizeof(declared));
  if (declared < 0 || (uint64_t)declared > (uint64_t)maximum ||
      (exact != SIZE_MAX && (uint64_t)declared != (uint64_t)exact))
    return 0;
  start = (uintptr_t)opaque;
  if (start > UINTPTR_MAX - sizeof(declared) ||
      (size_t)declared > UINTPTR_MAX - start - sizeof(declared))
    return 0;
  out->header = (const uint8_t *)opaque;
  out->bytes = (uint8_t *)(start + sizeof(declared));
  out->allocation = sizeof(declared) + (size_t)declared;
  out->length = (size_t)declared;
  return 1;
}

static int const_span_from_bytevector(const void *opaque, size_t maximum,
                                      c2_span *out) {
  return span_from_bytevector((void *)(uintptr_t)opaque, maximum, SIZE_MAX, out);
}

static int overlaps(const c2_span *left, const c2_span *right) {
  const uintptr_t a = (uintptr_t)left->header;
  const uintptr_t b = (uintptr_t)right->header;
  if (left->allocation > UINTPTR_MAX - a ||
      right->allocation > UINTPTR_MAX - b)
    return 1;
  return a < b + right->allocation && b < a + left->allocation;
}

static int valid_limits(int64_t file, int64_t metadata, int64_t tensor,
                        int64_t tensors, int64_t profile) {
  return file > 0 && (uint64_t)file <= ET_C2_WIRE_MAX_FILE_BYTES &&
         metadata > 0 &&
         (uint64_t)metadata <= ET_C2_WIRE_MAX_METADATA_BYTES && tensor > 0 &&
         (uint64_t)tensor <= ET_C2_WIRE_MAX_TENSOR_BYTES && tensors > 0 &&
         tensors <= (int64_t)ET_C2_WIRE_MAX_TENSORS &&
         (profile == 0 || profile == 1);
}

static void init_error(et_c2_checkpoint_core_error_v1 *error) {
  memset(error, 0, sizeof(*error));
  error->struct_size = sizeof(*error);
  error->format_error.struct_size = sizeof(error->format_error);
}

static void publish_failure(uint8_t *out,
                            const et_c2_checkpoint_core_error_v1 *error) {
  memset(out, 0, (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES);
  put_u64(out + 136u, error->category);
  put_u64(out + 144u, error->code);
  put_u64(out + 152u, error->offset);
  put_u64(out + 160u, (uint64_t)error->native_status);
  put_u64(out + 168u, error->format_error.category);
  put_u64(out + 176u, error->format_error.code);
  put_u64(out + 184u, error->format_error.offset);
}

static void publish_success(uint8_t *out,
                            const et_c2_checkpoint_view_v1 *view) {
  const uint8_t *h = view->bytes;
  memset(out, 0, (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES);
  put_u64(out, ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC);
  put_u64(out + 8u, view->file_bytes);
  put_u64(out + 16u, view->tokenizer_fingerprint_bytes);
  put_u64(out + 24u, view->x1_bytes);
  put_u64(out + 32u, get_u32(h + 128u));
  put_u64(out + 40u, get_u32(h + 132u));
  put_u64(out + 48u, view->model_container_bytes);
  put_u64(out + 56u, view->optimizer_metadata_bytes);
  put_u64(out + 64u, view->optimizer_payload_bytes);
  put_u64(out + 72u, view->total_tokens);
  put_u64(out + 80u, view->completed_updates);
  put_u64(out + 88u, view->epochs);
  put_u64(out + 96u, view->rng_key_bits);
  put_u64(out + 104u, view->rng_counter_low_bits);
  put_u64(out + 112u, view->rng_counter_high_bits);
  put_u64(out + 120u, view->unique_parameter_count);
  put_u64(out + 128u, view->optimizer_group_count);
}

static int prepare_call(const void *path_opaque, int64_t maximum_file_bytes,
                        int64_t maximum_metadata_bytes,
                        int64_t maximum_tensor_bytes, int64_t maximum_tensors,
                        int64_t enforce_operational_profile, void *result_opaque,
                        c2_span *path_span, c2_span *result_span,
                        char path[C2_PATH_MAX + 1u],
                        et_c2_checkpoint_limits_v1 *limits) {
  if (!const_span_from_bytevector(path_opaque, C2_PATH_MAX, path_span) ||
      path_span->length == 0u ||
      memchr(path_span->bytes, 0, path_span->length) != NULL ||
      !span_from_bytevector(result_opaque,
                            (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES,
                            (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES,
                            result_span) ||
      overlaps(path_span, result_span) ||
      !valid_limits(maximum_file_bytes, maximum_metadata_bytes,
                    maximum_tensor_bytes, maximum_tensors,
                    enforce_operational_profile))
    return 0;
  memcpy(path, path_span->bytes, path_span->length);
  path[path_span->length] = '\0';
  memset(limits, 0, sizeof(*limits));
  limits->struct_size = sizeof(*limits);
  limits->maximum_file_bytes = (uint64_t)maximum_file_bytes;
  limits->maximum_metadata_bytes = (uint64_t)maximum_metadata_bytes;
  limits->maximum_tensor_bytes = (uint64_t)maximum_tensor_bytes;
  limits->maximum_tensors = (uint32_t)maximum_tensors;
  limits->enforce_operational_profile =
      (uint32_t)enforce_operational_profile;
  return 1;
}

static int32_t release_image(et_c2_private_checkpoint_image **image,
                             et_c2_checkpoint_core_error_v1 *error) {
  et_c2_checkpoint_core_error_v1 release_error;
  int32_t status;
  init_error(&release_error);
  status = et_c2_private_checkpoint_image_release_v1(image, &release_error);
  if (status == ET_C2_CORE_OK) return ET_C2_CORE_OK;
  *error = release_error;
  error->category = ET_C2_CORE_INTERNAL;
  return ET_C2_CORE_INTERNAL;
}

int64_t et_c2_private_checkpoint_load_measure_v1(
    const void *path_opaque, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    void *result_opaque) {
  c2_span path_span, result_span;
  char path[C2_PATH_MAX + 1u];
  et_c2_checkpoint_limits_v1 limits;
  et_c2_checkpoint_core_error_v1 error;
  et_c2_private_checkpoint_image *image = NULL;
  const et_c2_checkpoint_view_v1 *view = NULL;
  int32_t status;
  init_error(&error);
  if (!prepare_call(path_opaque, maximum_file_bytes, maximum_metadata_bytes,
                    maximum_tensor_bytes, maximum_tensors,
                    enforce_operational_profile, result_opaque, &path_span,
                    &result_span, path, &limits))
    return ET_C2_CORE_INVALID_ARGUMENT;
  memset(result_span.bytes, 0, result_span.length);
  status = et_c2_private_checkpoint_load_image_v1(
      path, &limits, ET_C2_FORMAT_LOAD, &image, &error);
  if (status == ET_C2_CORE_OK)
    status = et_c2_private_checkpoint_image_view_v1(image, &view, &error);
  if (status == ET_C2_CORE_OK) publish_success(result_span.bytes, view);
  if (image != NULL) {
    int32_t release_status = release_image(&image, &error);
    if (release_status != ET_C2_CORE_OK) status = release_status;
  }
  if (status != ET_C2_CORE_OK) publish_failure(result_span.bytes, &error);
  return status;
}

int64_t et_c2_private_checkpoint_load_stage_v1(
    const void *path_opaque, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    const void *measurement_opaque, void *tokenizer_opaque,
    void *config_opaque, void *x1_opaque,
    void *current_opaque, void *epoch_opaque, void *model_opaque,
    void *optimizer_metadata_opaque, void *optimizer_payload_opaque,
    void *result_opaque) {
  c2_span path_span, result_span, measurement_span, outputs[8];
  char path[C2_PATH_MAX + 1u];
  et_c2_checkpoint_limits_v1 limits;
  et_c2_checkpoint_core_error_v1 error;
  et_c2_private_checkpoint_image *image = NULL;
  const et_c2_checkpoint_view_v1 *view = NULL;
  void *opaque[8] = {tokenizer_opaque, config_opaque, x1_opaque,
                     current_opaque, epoch_opaque, model_opaque,
                     optimizer_metadata_opaque, optimizer_payload_opaque};
  static const size_t measurement_offsets[8] = {16u, 0u, 24u, 32u,
                                                 40u, 48u, 56u, 64u};
  size_t exact[8], requested[8];
  const uint8_t *source[8];
  size_t i, j;
  int32_t status;
  init_error(&error);
  if (!prepare_call(path_opaque, maximum_file_bytes, maximum_metadata_bytes,
                    maximum_tensor_bytes, maximum_tensors,
                    enforce_operational_profile, result_opaque, &path_span,
                    &result_span, path, &limits))
    return ET_C2_CORE_INVALID_ARGUMENT;
  if (!const_span_from_bytevector(
          measurement_opaque, (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES,
          &measurement_span) ||
      measurement_span.length != (size_t)ET_C2_CHECKPOINT_LOAD_RESULT_BYTES ||
      overlaps(&measurement_span, &path_span) ||
      overlaps(&measurement_span, &result_span) ||
      get_u64(measurement_span.bytes) != ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC)
    return ET_C2_CORE_INVALID_ARGUMENT;
  requested[0] = (size_t)get_u64(measurement_span.bytes + 16u);
  requested[1] = C2_CONFIG_FINGERPRINT_BYTES;
  requested[2] = (size_t)get_u64(measurement_span.bytes + 24u);
  requested[3] = (size_t)get_u64(measurement_span.bytes + 32u);
  requested[4] = (size_t)get_u64(measurement_span.bytes + 40u);
  requested[5] = (size_t)get_u64(measurement_span.bytes + 48u);
  requested[6] = (size_t)get_u64(measurement_span.bytes + 56u);
  requested[7] = (size_t)get_u64(measurement_span.bytes + 64u);
  for (i = 0u; i < 8u; ++i) {
    const uint64_t encoded = i == 1u
                                 ? (uint64_t)C2_CONFIG_FINGERPRINT_BYTES
                                 : get_u64(measurement_span.bytes +
                                           measurement_offsets[i]);
    /* Refuse truncation on a narrower size_t and every implausible direct
       measurement before dereferencing an output header or starting I/O. */
    if (encoded > (uint64_t)SIZE_MAX ||
        encoded > (uint64_t)maximum_file_bytes ||
        !span_from_bytevector(opaque[i], requested[i], requested[i],
                              &outputs[i]) ||
        overlaps(&outputs[i], &path_span) ||
        overlaps(&outputs[i], &result_span) ||
        overlaps(&outputs[i], &measurement_span))
      return ET_C2_CORE_INVALID_ARGUMENT;
    for (j = 0u; j < i; ++j)
      if (overlaps(&outputs[i], &outputs[j]))
        return ET_C2_CORE_INVALID_ARGUMENT;
  }
  /* Every caller span is now exact and mutually disjoint. Result publication
     and the authoritative second read cannot alter any destination on a
     pre-copy failure. */
  memset(result_span.bytes, 0, result_span.length);
  status = et_c2_private_checkpoint_load_image_v1(
      path, &limits, ET_C2_FORMAT_LOAD, &image, &error);
  if (status == ET_C2_CORE_OK)
    status = et_c2_private_checkpoint_image_view_v1(image, &view, &error);
  if (status == ET_C2_CORE_OK) {
    const uint8_t *bytes = view->bytes;
    exact[0] = (size_t)view->tokenizer_fingerprint_bytes;
    exact[1] = C2_CONFIG_FINGERPRINT_BYTES;
    exact[2] = (size_t)view->x1_bytes;
    exact[3] = (size_t)get_u32(bytes + 128u);
    exact[4] = (size_t)get_u32(bytes + 132u);
    exact[5] = (size_t)view->model_container_bytes;
    exact[6] = (size_t)view->optimizer_metadata_bytes;
    exact[7] = (size_t)view->optimizer_payload_bytes;
    source[0] = bytes + 256u;
    source[1] = source[0] + exact[0];
    source[2] = bytes + (size_t)view->x1_offset;
    source[3] = bytes + (size_t)view->current_cursor_offset;
    source[4] = bytes + (size_t)view->epoch_cursor_offset;
    source[5] = bytes + (size_t)view->model_container_offset;
    source[6] = bytes + (size_t)view->optimizer_metadata_offset;
    source[7] = bytes + (size_t)view->optimizer_payload_offset;
    /* A changed source with different component sizes is not a malformed
       caller destination. The wrapper allocated from this exact validated
       measurement; classify the cross-pass race deterministically before
       inspecting or copying any output bytevector. */
    if (requested[0] != exact[0] || requested[1] != exact[1] ||
        requested[2] != exact[2] || requested[3] != exact[3] ||
        requested[4] != exact[4] || requested[5] != exact[5] ||
        requested[6] != exact[6] || requested[7] != exact[7]) {
      status = ET_C2_CORE_CORRUPT_DATA;
      error.category = ET_C2_CORE_CORRUPT_DATA;
      error.code = ET_C2_CORE_CODE_PROBED_HEADER_CHANGED;
      error.offset = 0u;
    }
  }
#ifdef ET_C2_CHECKPOINT_LOAD_TESTING
  if (status == ET_C2_CORE_OK && c2_fail_after_validate) {
    status = ET_C2_CORE_INTERNAL;
    error.category = ET_C2_CORE_INTERNAL;
    error.code = ET_C2_CORE_CODE_INVALID_IMAGE;
  }
#endif
  if (status == ET_C2_CORE_OK) {
    for (i = 0u; i < 8u; ++i) {
      if (exact[i] != 0u) memcpy(outputs[i].bytes, source[i], exact[i]);
#ifdef ET_C2_CHECKPOINT_LOAD_TESTING
      ++c2_copy_count;
#endif
    }
    publish_success(result_span.bytes, view);
  }
  if (image != NULL) {
    int32_t release_status = release_image(&image, &error);
    if (release_status != ET_C2_CORE_OK) status = release_status;
  }
  if (status != ET_C2_CORE_OK) publish_failure(result_span.bytes, &error);
  return status;
}
