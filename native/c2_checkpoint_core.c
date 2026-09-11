#include "c2_checkpoint_core.h"

#include "c2_checkpoint_reader.h"
#include "checkpoint_io.h"

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define C2_IMAGE_MAGIC UINT64_C(0x45544332494d4731)
#define C2_OUTER_BYTES UINT64_C(256)
#define C2_C1_BYTES UINT64_C(128)
#define C2_DIGEST_BYTES UINT64_C(32)
#define C2_I64_MAX UINT64_C(9223372036854775807)

static const uint8_t c2_core_magic[16] = {
    0x89, 0x45, 0x53, 0x48, 0x4b, 0x54, 0x52, 0x4e,
    0x53, 0x54, 0x41, 0x54, 0x45, 0x0d, 0x0a, 0x00};
static const uint8_t c2_core_c1_magic[16] = {
    0x89, 0x45, 0x53, 0x48, 0x4b, 0x4f, 0x4c, 0x43,
    0x4b, 0x50, 0x54, 0x0d, 0x0a, 0x1a, 0x0a, 0x00};

typedef struct c2_outer_probe {
  uint64_t file_bytes;
  uint64_t metadata_bytes;
  uint64_t payload_offset;
  uint64_t payload_bytes;
  uint64_t model_bytes;
  uint32_t model_tensors;
  uint32_t total_tensors;
} c2_outer_probe;

typedef struct c2_test_bytevector_256 {
  int64_t length;
  uint8_t bytes[256];
} c2_test_bytevector_256;

typedef struct c2_test_bytevector_128 {
  int64_t length;
  uint8_t bytes[128];
} c2_test_bytevector_128;

struct et_c2_private_checkpoint_image {
  uint64_t magic;
  struct et_c2_private_checkpoint_image *registry_next;
  et_c2_checkpoint_view_v1 view;
  int64_t bytevector_length;
  uint8_t bytes[];
};

_Static_assert(offsetof(et_c2_private_checkpoint_image, bytes) ==
                   offsetof(et_c2_private_checkpoint_image, bytevector_length) +
                       sizeof(int64_t),
               "C2 retained image must contain an exact reader bytevector");

static et_c2_private_checkpoint_image *c2_images;

#ifdef ET_C2_CHECKPOINT_CORE_TESTING
static et_c2_checkpoint_core_test_hook_v1 c2_test_hook;
static size_t c2_test_allocation_limit = SIZE_MAX;
static size_t c2_test_successful_allocations;

void et_c2_checkpoint_core_test_reset_v1(void) {
  c2_test_hook = NULL;
  c2_test_allocation_limit = SIZE_MAX;
  c2_test_successful_allocations = 0u;
}

void et_c2_checkpoint_core_test_set_hook_v1(
    et_c2_checkpoint_core_test_hook_v1 hook) {
  c2_test_hook = hook;
}

void et_c2_checkpoint_core_test_fail_alloc_after_v1(
    size_t successful_allocations) {
  c2_test_allocation_limit = successful_allocations;
  c2_test_successful_allocations = 0u;
}

size_t et_c2_checkpoint_core_test_live_images_v1(void) {
  size_t count = 0u;
  et_c2_private_checkpoint_image *image;
  for (image = c2_images; image != NULL; image = image->registry_next) ++count;
  return count;
}

static void run_test_hook(uint32_t stage, const char *path) {
  if (c2_test_hook != NULL) c2_test_hook(stage, path);
}
#else
static void run_test_hook(uint32_t stage, const char *path) {
  (void)stage;
  (void)path;
}
#endif

static void *c2_malloc(size_t bytes) {
#ifdef ET_C2_CHECKPOINT_CORE_TESTING
  if (c2_test_successful_allocations >= c2_test_allocation_limit) {
    errno = ENOMEM;
    return NULL;
  }
  ++c2_test_successful_allocations;
#endif
  return malloc(bytes);
}

static uint16_t load_u16(const uint8_t *bytes) {
  return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u));
}

static uint32_t load_u32(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
         ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
}

static uint64_t load_u64(const uint8_t *bytes) {
  uint64_t value = 0u;
  unsigned index;
  for (index = 0u; index < 8u; ++index)
    value |= (uint64_t)bytes[index] << (8u * index);
  return value;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
  if (left > UINT64_MAX - right) return 0;
  *result = left + right;
  return 1;
}

static int zero_bytes(const uint8_t *bytes, size_t count) {
  size_t index;
  for (index = 0u; index < count; ++index)
    if (bytes[index] != 0u) return 0;
  return 1;
}

static int spans_overlap(const void *left, size_t left_bytes,
                         const void *right, size_t right_bytes) {
  const uintptr_t a = (uintptr_t)left;
  const uintptr_t b = (uintptr_t)right;
  if (left == NULL || right == NULL) return 0;
  if (a > UINTPTR_MAX - left_bytes || b > UINTPTR_MAX - right_bytes) return 1;
  return a < b + right_bytes && b < a + left_bytes;
}

static int pointer_in_span(const void *pointer, const void *span,
                           size_t span_bytes) {
  const uintptr_t p = (uintptr_t)pointer;
  const uintptr_t start = (uintptr_t)span;
  return pointer != NULL && span != NULL && start <= UINTPTR_MAX - span_bytes &&
         p >= start && p < start + span_bytes;
}

static void clear_error(et_c2_checkpoint_core_error_v1 *error) {
  if (error == NULL) return;
  error->category = ET_C2_CORE_OK;
  error->code = ET_C2_CORE_CODE_NONE;
  error->offset = 0u;
  error->native_status = 0;
  error->format_error.struct_size = ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE;
  error->format_error.category = ET_C2_FORMAT_OK;
  error->format_error.code = ET_C2_FORMAT_CODE_NONE;
  error->format_error.offset = 0u;
}

static int32_t core_fail(et_c2_checkpoint_core_error_v1 *error,
                         uint32_t category, uint32_t code, uint64_t offset,
                         int64_t native_status) {
  if (error != NULL) {
    error->category = category;
    error->code = code;
    error->offset = offset;
    error->native_status = native_status;
  }
  return (int32_t)category;
}

static int error_argument_valid(et_c2_checkpoint_core_error_v1 *error) {
  return error == NULL || error->struct_size == ET_C2_CHECKPOINT_CORE_ERROR_V1_SIZE;
}

static int limits_valid(const et_c2_checkpoint_limits_v1 *limits) {
  return limits != NULL &&
         limits->struct_size == ET_C2_CHECKPOINT_LIMITS_V1_SIZE &&
         limits->maximum_file_bytes > 0u &&
         limits->maximum_file_bytes <= ET_C2_WIRE_MAX_FILE_BYTES &&
         limits->maximum_metadata_bytes > 0u &&
         limits->maximum_metadata_bytes <= ET_C2_WIRE_MAX_METADATA_BYTES &&
         limits->maximum_tensor_bytes > 0u &&
         limits->maximum_tensor_bytes <= ET_C2_WIRE_MAX_TENSOR_BYTES &&
         limits->maximum_tensors > 0u &&
         limits->maximum_tensors <= ET_C2_WIRE_MAX_TENSORS &&
         limits->enforce_operational_profile <= 1u;
}

static int32_t admit_physical(uint64_t physical,
                              const et_c2_checkpoint_limits_v1 *limits,
                              et_c2_checkpoint_core_error_v1 *error) {
  if (physical < C2_OUTER_BYTES + C2_DIGEST_BYTES ||
      physical > ET_C2_WIRE_MAX_FILE_BYTES)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u, 0);
  if (physical > limits->maximum_file_bytes)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u, 0);
  if (limits->enforce_operational_profile &&
      physical > ET_C2_PROFILE_MAX_FILE_BYTES)
    return core_fail(error, ET_C2_CORE_UNSUPPORTED,
                     ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u, 0);
  return ET_C2_CORE_OK;
}

static int32_t probe_outer(const uint8_t bytes[256], uint64_t physical,
                           const et_c2_checkpoint_limits_v1 *limits,
                           c2_outer_probe *probe,
                           et_c2_checkpoint_core_error_v1 *error) {
  uint64_t metadata_end, payload_end, file_end;
  uint32_t model, unique, optimizer, total, cursor, epoch, fp, groups;
  if (memcmp(bytes, c2_core_magic, sizeof(c2_core_magic)) != 0)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 0u, 0);
  if (load_u16(bytes + 16u) != 1u || load_u16(bytes + 18u) != 0u ||
      load_u64(bytes + 32u) != 0u)
    return core_fail(error, ET_C2_CORE_VERSION_MISMATCH,
                     ET_C2_CORE_CODE_OUTER_PROBE, 16u, 0);
  if (load_u32(bytes + 20u) != 256u)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 20u, 0);
  if (load_u32(bytes + 24u) != 1u || load_u32(bytes + 28u) != 1u)
    return core_fail(error, ET_C2_CORE_VERSION_MISMATCH,
                     ET_C2_CORE_CODE_OUTER_PROBE, 24u, 0);
  if (!zero_bytes(bytes + 240u, 16u))
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 240u, 0);
  probe->file_bytes = load_u64(bytes + 40u);
  probe->metadata_bytes = load_u64(bytes + 56u);
  probe->payload_offset = load_u64(bytes + 64u);
  probe->payload_bytes = load_u64(bytes + 72u);
  probe->model_bytes = load_u64(bytes + 96u);
  model = load_u32(bytes + 80u);
  unique = load_u32(bytes + 84u);
  optimizer = load_u32(bytes + 88u);
  total = load_u32(bytes + 92u);
  cursor = load_u32(bytes + 128u);
  epoch = load_u32(bytes + 132u);
  fp = load_u32(bytes + 136u);
  groups = load_u32(bytes + 156u);
  probe->model_tensors = model;
  probe->total_tensors = total;
  if (probe->file_bytes != physical || load_u64(bytes + 48u) != 256u ||
      !add_u64(256u, probe->metadata_bytes, &metadata_end) ||
      probe->payload_offset != metadata_end ||
      !add_u64(probe->payload_offset, probe->payload_bytes, &payload_end) ||
      !add_u64(payload_end, 32u, &file_end) || file_end != physical ||
      probe->metadata_bytes > ET_C2_WIRE_MAX_METADATA_BYTES || model > 4096u ||
      unique == 0u || unique > 1365u || optimizer != 2u * unique ||
      total != model + optimizer || total > ET_C2_WIRE_MAX_TENSORS ||
      groups == 0u || groups > 1365u || cursor != epoch ||
      (fp == 95u ? cursor != 303u : (fp == 96u ? cursor != 304u : 1)) ||
      load_u32(bytes + 140u) != 93u || load_u32(bytes + 144u) != 19u ||
      load_u32(bytes + 148u) != 57u || load_u32(bytes + 152u) != 71u ||
      load_u64(bytes + 120u) == 0u || load_u64(bytes + 120u) > 16384u ||
      load_u64(bytes + 184u) > C2_I64_MAX ||
      load_u64(bytes + 192u) > C2_I64_MAX ||
      load_u64(bytes + 200u) > C2_I64_MAX || load_u64(bytes + 208u) != 1u)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 40u, 0);
  { static const uint16_t versions[12] = {1, 0, 2, 0, 1, 0,
                                          1, 0, 1, 0, 1, 0};
    unsigned index;
    for (index = 0u; index < 12u; ++index)
      if (load_u16(bytes + 160u + 2u * index) != versions[index])
        return core_fail(error, ET_C2_CORE_VERSION_MISMATCH,
                         ET_C2_CORE_CODE_OUTER_PROBE,
                         160u + 2u * index, 0);
  }
  if (probe->metadata_bytes > limits->maximum_metadata_bytes ||
      total > limits->maximum_tensors)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 56u, 0);
  if (limits->enforce_operational_profile &&
      (probe->metadata_bytes > ET_C2_PROFILE_MAX_METADATA_BYTES ||
       total > ET_C2_PROFILE_MAX_TENSORS))
    return core_fail(error, ET_C2_CORE_UNSUPPORTED,
                     ET_C2_CORE_CODE_OUTER_PROBE, 56u, 0);
  if (probe->model_bytes < C2_C1_BYTES + C2_DIGEST_BYTES ||
      probe->model_bytes > probe->payload_bytes)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_OUTER_PROBE, 96u, 0);
  return ET_C2_CORE_OK;
}

static int32_t probe_c1(const uint8_t bytes[128], const c2_outer_probe *outer,
                        const et_c2_checkpoint_limits_v1 *limits,
                        et_c2_checkpoint_core_error_v1 *error) {
  uint64_t metadata, payload_offset, payload, unsigned_end, file_end;
  if (memcmp(bytes, c2_core_c1_magic, sizeof(c2_core_c1_magic)) != 0)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset, 0);
  if (load_u16(bytes + 16u) != 1u || load_u16(bytes + 18u) != 0u ||
      load_u64(bytes + 32u) != 0u || load_u16(bytes + 92u) != 1u ||
      load_u16(bytes + 94u) != 0u || load_u16(bytes + 96u) != 2u ||
      load_u16(bytes + 98u) != 0u)
    return core_fail(error, ET_C2_CORE_VERSION_MISMATCH,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset + 16u, 0);
  if (load_u32(bytes + 20u) != 128u || load_u32(bytes + 24u) != 1u ||
      load_u32(bytes + 28u) != 1u || !zero_bytes(bytes + 100u, 28u))
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset + 20u, 0);
  metadata = load_u64(bytes + 56u);
  payload_offset = load_u64(bytes + 64u);
  payload = load_u64(bytes + 72u);
  if (load_u64(bytes + 40u) != outer->model_bytes ||
      load_u64(bytes + 48u) != 128u ||
      !add_u64(128u, metadata, &unsigned_end) || payload_offset != unsigned_end ||
      !add_u64(payload_offset, payload, &unsigned_end) ||
      !add_u64(unsigned_end, 32u, &file_end) || file_end != outer->model_bytes ||
      metadata > ET_C2_WIRE_MAX_METADATA_BYTES ||
      load_u32(bytes + 80u) != outer->model_tensors ||
      load_u32(bytes + 80u) > 4096u || load_u32(bytes + 84u) > 4096u ||
      load_u32(bytes + 88u) != 19u)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset + 40u, 0);
  if (outer->metadata_bytes > limits->maximum_metadata_bytes ||
      metadata > limits->maximum_metadata_bytes - outer->metadata_bytes)
    return core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset + 56u, 0);
  if (limits->enforce_operational_profile &&
      (outer->metadata_bytes > ET_C2_PROFILE_MAX_METADATA_BYTES ||
       metadata > ET_C2_PROFILE_MAX_METADATA_BYTES - outer->metadata_bytes))
    return core_fail(error, ET_C2_CORE_UNSUPPORTED,
                     ET_C2_CORE_CODE_C1_PROBE, outer->payload_offset + 56u, 0);
  return ET_C2_CORE_OK;
}

static et_c2_private_checkpoint_image *find_image(const void *candidate) {
  et_c2_private_checkpoint_image *image;
  for (image = c2_images; image != NULL; image = image->registry_next)
    if ((const void *)image == candidate)
      return image->magic == C2_IMAGE_MAGIC ? image : NULL;
  return NULL;
}

static void unlink_image(et_c2_private_checkpoint_image *image) {
  et_c2_private_checkpoint_image **cursor = &c2_images;
  while (*cursor != NULL && *cursor != image) cursor = &(*cursor)->registry_next;
  if (*cursor == image) *cursor = image->registry_next;
}

int32_t et_c2_private_checkpoint_load_image_v1(
    const char *path, const et_c2_checkpoint_limits_v1 *limits, uint32_t mode,
    et_c2_private_checkpoint_image **image_slot,
    et_c2_checkpoint_core_error_v1 *error) {
  et_c2_checkpoint_reader *reader = NULL;
  et_c2_private_checkpoint_image *image = NULL;
  c2_test_bytevector_256 outer_bytes;
  c2_test_bytevector_128 c1_bytes;
  c2_outer_probe outer;
  et_c2_checkpoint_view_v1 parsed = {0};
  et_c2_checkpoint_format_error_v1 format_error = {0};
  uint64_t physical = 0u;
  int64_t native_status;
  int32_t status;
  size_t allocation_bytes;
  if (!error_argument_valid(error)) return ET_C2_CORE_INVALID_ARGUMENT;
  if (error != NULL &&
      (spans_overlap(error, sizeof(*error), image_slot,
                     image_slot == NULL ? 0u : sizeof(*image_slot)) ||
       spans_overlap(error, sizeof(*error), limits,
                     limits == NULL ? 0u : sizeof(*limits)) ||
       pointer_in_span(path, error, sizeof(*error))))
    return ET_C2_CORE_INVALID_ARGUMENT;
  if (spans_overlap(image_slot, image_slot == NULL ? 0u : sizeof(*image_slot),
                    limits, limits == NULL ? 0u : sizeof(*limits)))
    return ET_C2_CORE_INVALID_ARGUMENT;
  clear_error(error);
  if (path == NULL || path[0] == '\0' || image_slot == NULL ||
      *image_slot != NULL || !limits_valid(limits) ||
      (mode != ET_C2_FORMAT_INSPECT && mode != ET_C2_FORMAT_LOAD))
    return core_fail(error, ET_C2_CORE_INVALID_ARGUMENT,
                     ET_C2_CORE_CODE_ARGUMENT, 0u, 0);
  native_status = et_c2_checkpoint_reader_open_v1(path, &reader, &physical);
  if (native_status != 0)
    return core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_NATIVE_IO, 0u,
                     native_status);
  status = admit_physical(physical, limits, error);
  if (status != ET_C2_CORE_OK) goto cleanup;
  outer_bytes.length = (int64_t)sizeof(outer_bytes.bytes);
  native_status = et_c2_checkpoint_reader_read_exact_v1(
      reader, 0u, &outer_bytes.length, sizeof(outer_bytes.bytes));
  if (native_status != 0) {
    status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_NATIVE_IO, 0u,
                       native_status);
    goto cleanup;
  }
  status = probe_outer(outer_bytes.bytes, physical, limits, &outer, error);
  if (status != ET_C2_CORE_OK) goto cleanup;
  run_test_hook(ET_C2_CORE_TEST_AFTER_OUTER_PROBE, path);
  c1_bytes.length = (int64_t)sizeof(c1_bytes.bytes);
  native_status = et_c2_checkpoint_reader_read_exact_v1(
      reader, outer.payload_offset, &c1_bytes.length, sizeof(c1_bytes.bytes));
  if (native_status != 0) {
    status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_NATIVE_IO,
                       outer.payload_offset, native_status);
    goto cleanup;
  }
  status = probe_c1(c1_bytes.bytes, &outer, limits, error);
  if (status != ET_C2_CORE_OK) goto cleanup;
  run_test_hook(ET_C2_CORE_TEST_AFTER_C1_PROBE, path);
  if (physical > (uint64_t)SIZE_MAX - offsetof(et_c2_private_checkpoint_image, bytes)) {
    status = core_fail(error, ET_C2_CORE_IO,
                       ET_C2_CORE_CODE_STAGING_ALLOCATION, 0u,
                       (int64_t)(EOVERFLOW |
                           (ET_CHECKPOINT_IO_STAGE_ALLOCATE <<
                            ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT)));
    goto cleanup;
  }
  allocation_bytes = offsetof(et_c2_private_checkpoint_image, bytes) +
                     (size_t)physical;
  image = (et_c2_private_checkpoint_image *)c2_malloc(allocation_bytes);
  if (image == NULL) {
    const int allocation_errno = errno == 0 ? ENOMEM : errno;
    status = core_fail(error, ET_C2_CORE_IO,
                       ET_C2_CORE_CODE_STAGING_ALLOCATION, 0u,
                       (int64_t)(allocation_errno |
                           (ET_CHECKPOINT_IO_STAGE_ALLOCATE <<
                            ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT)));
    goto cleanup;
  }
  memset(image, 0, offsetof(et_c2_private_checkpoint_image, bytes));
  image->bytevector_length = (int64_t)physical;
  native_status = et_c2_checkpoint_reader_read_exact_v1(
      reader, 0u, &image->bytevector_length, physical);
  if (native_status != 0) {
    status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_NATIVE_IO, 0u,
                       native_status);
    goto cleanup;
  }
  run_test_hook(ET_C2_CORE_TEST_AFTER_FULL_READ, path);
  if (memcmp(outer_bytes.bytes, image->bytes, sizeof(outer_bytes.bytes)) != 0) {
    status = core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                       ET_C2_CORE_CODE_PROBED_HEADER_CHANGED, 0u, 0);
    goto cleanup;
  }
  if (memcmp(c1_bytes.bytes, image->bytes + outer.payload_offset,
             sizeof(c1_bytes.bytes)) != 0) {
    status = core_fail(error, ET_C2_CORE_CORRUPT_DATA,
                       ET_C2_CORE_CODE_PROBED_HEADER_CHANGED,
                       outer.payload_offset, 0);
    goto cleanup;
  }
  parsed.struct_size = ET_C2_CHECKPOINT_VIEW_V1_SIZE;
  format_error.struct_size = ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE;
  status = et_c2_checkpoint_parse_v1(image->bytes, (size_t)physical, limits,
                                     mode, &parsed, &format_error);
  if (status != ET_C2_FORMAT_OK) {
    if (error != NULL) error->format_error = format_error;
    status = core_fail(error, (uint32_t)status, ET_C2_CORE_CODE_FORMAT,
                       format_error.offset, 0);
    goto cleanup;
  }
  native_status = et_c2_checkpoint_reader_validate_final_v1(reader);
  if (native_status != 0) {
    status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_NATIVE_IO, 0u,
                       native_status);
    goto cleanup;
  }
  native_status = et_c2_checkpoint_reader_close_v1(&reader);
  if (native_status != 0) {
    status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_READER_CLOSE, 0u,
                       native_status);
    goto cleanup;
  }
  image->magic = C2_IMAGE_MAGIC;
  image->view = parsed;
  image->view.bytes = image->bytes;
  image->registry_next = c2_images;
  c2_images = image;
  *image_slot = image;
  return ET_C2_CORE_OK;

cleanup:
  free(image);
  if (reader != NULL) {
    native_status = et_c2_checkpoint_reader_close_v1(&reader);
    if (status == ET_C2_CORE_OK && native_status != 0)
      status = core_fail(error, ET_C2_CORE_IO, ET_C2_CORE_CODE_READER_CLOSE,
                         0u, native_status);
  }
  return status;
}

int32_t et_c2_private_checkpoint_image_view_v1(
    const et_c2_private_checkpoint_image *candidate,
    const et_c2_checkpoint_view_v1 **view,
    et_c2_checkpoint_core_error_v1 *error) {
  et_c2_private_checkpoint_image *image;
  if (!error_argument_valid(error)) return ET_C2_CORE_INVALID_ARGUMENT;
  if (error != NULL &&
      spans_overlap(error, sizeof(*error), view,
                    view == NULL ? 0u : sizeof(*view)))
    return ET_C2_CORE_INVALID_ARGUMENT;
  clear_error(error);
  if (view == NULL || *view != NULL)
    return core_fail(error, ET_C2_CORE_INVALID_ARGUMENT,
                     ET_C2_CORE_CODE_ARGUMENT, 0u, 0);
  image = find_image(candidate);
  if (image == NULL)
    return core_fail(error, ET_C2_CORE_INVALID_ARGUMENT,
                     ET_C2_CORE_CODE_INVALID_IMAGE, 0u, 0);
  *view = &image->view;
  return ET_C2_CORE_OK;
}

int32_t et_c2_private_checkpoint_image_release_v1(
    et_c2_private_checkpoint_image **image_slot,
    et_c2_checkpoint_core_error_v1 *error) {
  et_c2_private_checkpoint_image *image;
  if (!error_argument_valid(error)) return ET_C2_CORE_INVALID_ARGUMENT;
  if (error != NULL &&
      spans_overlap(error, sizeof(*error), image_slot,
                    image_slot == NULL ? 0u : sizeof(*image_slot)))
    return ET_C2_CORE_INVALID_ARGUMENT;
  clear_error(error);
  if (image_slot == NULL)
    return core_fail(error, ET_C2_CORE_INVALID_ARGUMENT,
                     ET_C2_CORE_CODE_ARGUMENT, 0u, 0);
  if (*image_slot == NULL) return ET_C2_CORE_OK;
  image = find_image(*image_slot);
  if (image == NULL)
    return core_fail(error, ET_C2_CORE_INVALID_ARGUMENT,
                     ET_C2_CORE_CODE_INVALID_IMAGE, 0u, 0);
  unlink_image(image);
  *image_slot = NULL;
  image->magic = 0u;
  memset(image->bytes, 0, (size_t)image->bytevector_length);
  free(image);
  return ET_C2_CORE_OK;
}
