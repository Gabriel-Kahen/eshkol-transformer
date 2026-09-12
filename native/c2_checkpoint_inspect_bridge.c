#include "c2_checkpoint_inspect_bridge.h"

#include "c2_checkpoint_core.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define C2_PATH_MAX ((size_t)4096u)
#define C2_RESULT_MAGIC UINT64_C(0x4332494e53503130)
#define C2_TOKENIZER_MAX ((size_t)96u)
#define C2_CONFIG_BYTES ((size_t)93u)
#define C2_API_BYTES ((size_t)11u)
#define C2_TOKENIZER_AT ((size_t)112u)
#define C2_CONFIG_AT (C2_TOKENIZER_AT + C2_TOKENIZER_MAX)
#define C2_API_AT (C2_CONFIG_AT + C2_CONFIG_BYTES)

typedef struct c2_projection {
  uint64_t file_bytes;
  uint64_t payload_bytes;
  uint64_t tensor_count;
  uint64_t tokenizer_bytes;
  uint8_t tokenizer[C2_TOKENIZER_MAX];
  uint8_t config[C2_CONFIG_BYTES];
  uint8_t api[C2_API_BYTES];
} c2_projection;

#ifdef ET_C2_CHECKPOINT_INSPECT_TESTING
static int c2_fail_projection;
static int c2_fail_release;
void et_c2_checkpoint_inspect_test_reset_v1(void) {
  c2_fail_projection = 0;
  c2_fail_release = 0;
}
void et_c2_checkpoint_inspect_test_fail_projection_v1(int64_t enabled) {
  c2_fail_projection = enabled != 0;
}
void et_c2_checkpoint_inspect_test_fail_release_v1(int64_t enabled) {
  c2_fail_release = enabled != 0;
}
#endif

static uint64_t get_u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

static void put_u64(uint8_t *p, uint64_t value) {
  unsigned i;
  for (i = 0u; i < 8u; ++i) p[i] = (uint8_t)(value >> (8u * i));
}

static int bytevector(void *opaque, size_t maximum, size_t exact,
                      uint8_t **bytes, size_t *length) {
  int64_t declared;
  uintptr_t start;
  if (opaque == NULL) return 0;
  memcpy(&declared, opaque, sizeof(declared));
  if (declared < 0 || (uint64_t)declared > (uint64_t)maximum ||
      (exact != SIZE_MAX && (uint64_t)declared != (uint64_t)exact))
    return 0;
  start = (uintptr_t)opaque;
  if (start > UINTPTR_MAX - sizeof(declared) ||
      (size_t)declared > UINTPTR_MAX - start - sizeof(declared))
    return 0;
  *bytes = (uint8_t *)(start + sizeof(declared));
  *length = (size_t)declared;
  return 1;
}

static void publish_error(uint8_t *out,
                          const et_c2_checkpoint_core_error_v1 *error) {
  memset(out, 0, (size_t)ET_C2_INSPECT_RESULT_BYTES);
  put_u64(out + 8u, error->category);
  put_u64(out + 16u, error->code);
  put_u64(out + 24u, error->offset);
  put_u64(out + 32u, (uint64_t)error->native_status);
  put_u64(out + 40u, error->format_error.category);
  put_u64(out + 48u, error->format_error.code);
  put_u64(out + 56u, error->format_error.offset);
}

static int projection_copy(const et_c2_checkpoint_view_v1 *view,
                           c2_projection *projection) {
  const uint8_t *bytes = view->bytes;
  size_t tokenizer = (size_t)view->tokenizer_fingerprint_bytes;
  size_t library_at;
  uint64_t c1_payload;
  if (bytes == NULL || (tokenizer != 95u && tokenizer != 96u)) return 0;
  library_at = 256u + tokenizer + C2_CONFIG_BYTES + 19u;
  c1_payload = get_u64(bytes + (size_t)view->model_container_offset + 72u);
  if (c1_payload > UINT64_MAX - view->optimizer_payload_bytes) return 0;
  memset(projection, 0, sizeof(*projection));
  projection->file_bytes = view->file_bytes;
  projection->payload_bytes = c1_payload + view->optimizer_payload_bytes;
  projection->tensor_count = view->total_tensor_count;
  projection->tokenizer_bytes = tokenizer;
  memcpy(projection->tokenizer, bytes + 256u, tokenizer);
  memcpy(projection->config, bytes + 256u + tokenizer, C2_CONFIG_BYTES);
  memcpy(projection->api, bytes + library_at + 19u, C2_API_BYTES);
  return 1;
}

int64_t et_c2_private_checkpoint_inspect_bridge_v1(
    void *path_bytevector, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    void *result_bytevector) {
  et_c2_checkpoint_limits_v1 limits = {0};
  et_c2_checkpoint_core_error_v1 error = {0};
  et_c2_checkpoint_core_error_v1 release_error = {0};
  et_c2_private_checkpoint_image *image = NULL;
  const et_c2_checkpoint_view_v1 *view = NULL;
  c2_projection projection;
  uint8_t *path_bytes, *out;
  size_t path_length, out_length;
  char path[C2_PATH_MAX + 1u];
  int32_t status;
  if (!bytevector(result_bytevector, ET_C2_INSPECT_RESULT_BYTES,
                  ET_C2_INSPECT_RESULT_BYTES, &out, &out_length))
    return ET_C2_CORE_INVALID_ARGUMENT;
  memset(out, 0, out_length);
  error.struct_size = sizeof(error);
  error.format_error.struct_size = sizeof(error.format_error);
  if (!bytevector(path_bytevector, C2_PATH_MAX, SIZE_MAX, &path_bytes,
                  &path_length) || path_length == 0u ||
      memchr(path_bytes, 0, path_length) != NULL ||
      maximum_file_bytes <= 0 || maximum_metadata_bytes <= 0 ||
      maximum_tensor_bytes <= 0 || maximum_tensors <= 0 ||
      maximum_tensors > INT64_C(6826) ||
      (enforce_operational_profile != 0 && enforce_operational_profile != 1)) {
    error.category = ET_C2_CORE_INVALID_ARGUMENT;
    error.code = ET_C2_CORE_CODE_ARGUMENT;
    publish_error(out, &error);
    return ET_C2_CORE_INVALID_ARGUMENT;
  }
  memcpy(path, path_bytes, path_length);
  path[path_length] = '\0';
  limits.struct_size = sizeof(limits);
  limits.maximum_file_bytes = (uint64_t)maximum_file_bytes;
  limits.maximum_metadata_bytes = (uint64_t)maximum_metadata_bytes;
  limits.maximum_tensor_bytes = (uint64_t)maximum_tensor_bytes;
  limits.maximum_tensors = (uint32_t)maximum_tensors;
  limits.enforce_operational_profile = (uint32_t)enforce_operational_profile;
  status = et_c2_private_checkpoint_load_image_v1(
      path, &limits, ET_C2_FORMAT_INSPECT, &image, &error);
  if (status == ET_C2_CORE_OK)
    status = et_c2_private_checkpoint_image_view_v1(image, &view, &error);
  if (status == ET_C2_CORE_OK) {
#ifdef ET_C2_CHECKPOINT_INSPECT_TESTING
    if (c2_fail_projection) {
      status = ET_C2_CORE_INTERNAL;
      error.category = ET_C2_CORE_INTERNAL;
      error.code = ET_C2_INSPECT_CODE_PROJECTION;
    } else
#endif
    if (!projection_copy(view, &projection)) {
      status = ET_C2_CORE_INTERNAL;
      error.category = ET_C2_CORE_INTERNAL;
      error.code = ET_C2_INSPECT_CODE_PROJECTION;
    }
  }
  if (image != NULL) {
    release_error.struct_size = sizeof(release_error);
    release_error.format_error.struct_size = sizeof(release_error.format_error);
    if (et_c2_private_checkpoint_image_release_v1(&image, &release_error) !=
        ET_C2_CORE_OK) {
      status = ET_C2_CORE_INTERNAL;
      error = release_error;
      error.category = ET_C2_CORE_INTERNAL;
    }
#ifdef ET_C2_CHECKPOINT_INSPECT_TESTING
    else if (status == ET_C2_CORE_OK && c2_fail_release) {
      status = ET_C2_CORE_INTERNAL;
      error.category = ET_C2_CORE_INTERNAL;
      error.code = ET_C2_INSPECT_CODE_RELEASE;
    }
#endif
  }
  if (status != ET_C2_CORE_OK) {
    publish_error(out, &error);
    return status;
  }
  memset(out, 0, out_length);
  put_u64(out, C2_RESULT_MAGIC);
  put_u64(out + 64u, projection.file_bytes);
  put_u64(out + 72u, projection.payload_bytes);
  put_u64(out + 80u, projection.tensor_count);
  put_u64(out + 88u, projection.tokenizer_bytes);
  put_u64(out + 96u, C2_CONFIG_BYTES);
  put_u64(out + 104u, C2_API_BYTES);
  memcpy(out + C2_TOKENIZER_AT, projection.tokenizer,
         (size_t)projection.tokenizer_bytes);
  memcpy(out + C2_CONFIG_AT, projection.config, C2_CONFIG_BYTES);
  memcpy(out + C2_API_AT, projection.api, C2_API_BYTES);
  return ET_C2_CORE_OK;
}
