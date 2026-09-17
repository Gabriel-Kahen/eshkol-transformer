#include "c2_checkpoint_core.h"
#include "c2_checkpoint_reader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK(condition)                                                     \
  do {                                                                       \
    ++checks;                                                                \
    if (!(condition)) {                                                      \
      fprintf(stderr, "C2 OPERATIONAL NATIVE FAIL line %d: %s\n", __LINE__, \
              #condition);                                                   \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static uint32_t u32(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
         ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
}

static uint64_t u64(const uint8_t *bytes) {
  uint64_t value = 0u;
  unsigned index;
  for (index = 0u; index < 8u; ++index)
    value |= (uint64_t)bytes[index] << (8u * index);
  return value;
}

static uint8_t *read_file(const char *path, size_t *length) {
  FILE *stream = fopen(path, "rb");
  long end;
  uint8_t *bytes;
  CHECK(stream != NULL);
  CHECK(fseek(stream, 0, SEEK_END) == 0);
  end = ftell(stream);
  CHECK(end >= 0);
  CHECK(fseek(stream, 0, SEEK_SET) == 0);
  *length = (size_t)end;
  bytes = (uint8_t *)malloc(*length == 0u ? 1u : *length);
  CHECK(bytes != NULL);
  CHECK(fread(bytes, 1u, *length, stream) == *length);
  CHECK(fclose(stream) == 0);
  return bytes;
}

static et_c2_checkpoint_limits_v1 limits(uint64_t file, uint64_t metadata,
                                          uint64_t tensor, uint32_t tensors) {
  et_c2_checkpoint_limits_v1 result = {
      sizeof(result), file, metadata, tensor, tensors, 1u};
  return result;
}

static void resources_flat(void) {
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);
}

static void expect(const char *path, const et_c2_checkpoint_limits_v1 *policy,
                   uint32_t category, uint32_t code, uint32_t format_code) {
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  int32_t status;
  error.struct_size = sizeof(error);
  status = et_c2_private_checkpoint_load_image_v1(
      path, policy, ET_C2_FORMAT_LOAD, &image, &error);
  if (status != (int32_t)category || error.category != category ||
      error.code != code)
    fprintf(stderr,
            "C2 operational mismatch %s: got %d/%u/%u/%u expected %u/%u/%u\n",
            path, status, error.category, error.code, error.format_error.code,
            category, code, format_code);
  CHECK(status == (int32_t)category);
  CHECK(error.category == category && error.code == code);
  if (format_code != 0u) CHECK(error.format_error.code == format_code);
  CHECK(image == NULL);
  resources_flat();
}

static uint64_t maximum_model_tensor(const uint8_t *bytes, size_t length) {
  uint64_t model = u64(bytes + 64u);
  uint64_t at = model + 128u + UINT64_C(19);
  uint32_t entries = u32(bytes + 80u);
  uint64_t maximum = 0u;
  uint32_t index;
  CHECK(model < length);
  for (index = 0u; index < entries; ++index) {
    uint64_t record = u64(bytes + at);
    uint64_t payload = u64(bytes + at + 16u);
    CHECK(record >= 92u && at <= length && record <= length - at);
    if (payload > maximum) maximum = payload;
    at += record;
  }
  return maximum;
}

static void check_joint_exact(const char *path,
                              const et_c2_checkpoint_limits_v1 *target) {
  et_c2_private_checkpoint_image *image = NULL;
  const et_c2_checkpoint_view_v1 *view = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  uint8_t *bytes;
  size_t length;
  uint64_t model;
  error.struct_size = sizeof(error);
  CHECK(et_c2_private_checkpoint_load_image_v1(
            path, target, ET_C2_FORMAT_LOAD, &image, &error) == 0);
  CHECK(et_c2_private_checkpoint_image_view_v1(image, &view, &error) == 0);
  CHECK(view != NULL && view->file_bytes == ET_C2_PROFILE_MAX_FILE_BYTES);
  CHECK(view->total_tensor_count == ET_C2_PROFILE_MAX_TENSORS);
  bytes = read_file(path, &length);
  CHECK(length == ET_C2_PROFILE_MAX_FILE_BYTES);
  model = u64(bytes + 64u);
  CHECK(u64(bytes + 56u) + u64(bytes + model + 56u) ==
        ET_C2_PROFILE_MAX_METADATA_BYTES);
  CHECK(maximum_model_tensor(bytes, length) ==
        ET_C2_PROFILE_MAX_TENSOR_BYTES);
  free(bytes);
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  resources_flat();
}

static void early_without_allocation(
    const char *path, const et_c2_checkpoint_limits_v1 *policy,
    uint32_t category, uint32_t code) {
  et_c2_checkpoint_core_test_fail_alloc_after_v1(0u);
  expect(path, policy, category, code, 0u);
  et_c2_checkpoint_core_test_reset_v1();
}

int main(int argc, char **argv) {
  et_c2_checkpoint_limits_v1 profile = limits(
      ET_C2_WIRE_MAX_FILE_BYTES, ET_C2_WIRE_MAX_METADATA_BYTES,
      ET_C2_WIRE_MAX_TENSOR_BYTES, ET_C2_WIRE_MAX_TENSORS);
  et_c2_checkpoint_limits_v1 target = limits(
      ET_C2_PROFILE_MAX_FILE_BYTES, ET_C2_PROFILE_MAX_METADATA_BYTES,
      ET_C2_PROFILE_MAX_TENSOR_BYTES, ET_C2_PROFILE_MAX_TENSORS);
  CHECK(argc == 10);
  check_joint_exact(argv[1], &target);
  check_joint_exact(argv[1], &profile);

  /* File, artifact metadata, and aggregate count are rejected from fixed
   * probes even when the full-image allocator is armed to fail immediately. */
  early_without_allocation(argv[2], &profile, ET_C2_CORE_UNSUPPORTED,
                           ET_C2_CORE_CODE_PHYSICAL_SIZE);
  early_without_allocation(argv[4], &profile, ET_C2_CORE_UNSUPPORTED,
                           ET_C2_CORE_CODE_C1_PROBE);
  early_without_allocation(argv[8], &profile, ET_C2_CORE_UNSUPPORTED,
                           ET_C2_CORE_CODE_OUTER_PROBE);

  /* Per-tensor admission is intentionally late: the same failpoint wins
   * before the retained parser can discover the least representable +4. */
  et_c2_checkpoint_core_test_fail_alloc_after_v1(0u);
  expect(argv[6], &profile, ET_C2_CORE_IO,
         ET_C2_CORE_CODE_STAGING_ALLOCATION, 0u);
  et_c2_checkpoint_core_test_reset_v1();
  expect(argv[6], &profile, ET_C2_CORE_UNSUPPORTED, ET_C2_CORE_CODE_FORMAT,
         ET_C2_FORMAT_CODE_PROFILE);

  /* Early dimensions win without guessing the unread late tensor field. */
  {
    et_c2_checkpoint_limits_v1 late_caller = profile;
    late_caller.maximum_tensor_bytes = ET_C2_PROFILE_MAX_TENSOR_BYTES - 1u;
    expect(argv[2], &late_caller, ET_C2_CORE_UNSUPPORTED,
           ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u);
    expect(argv[4], &late_caller, ET_C2_CORE_UNSUPPORTED,
           ET_C2_CORE_CODE_C1_PROBE, 0u);
  }
  {
    et_c2_checkpoint_limits_v1 early_caller = profile;
    early_caller.maximum_file_bytes = 1u;
    expect(argv[6], &early_caller, ET_C2_CORE_CORRUPT_DATA,
           ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u);
  }

  /* When caller maxima equal the target, caller lowering deterministically
   * precedes the fixed profile for every combined violation. */
  expect(argv[2], &target, ET_C2_CORE_CORRUPT_DATA,
         ET_C2_CORE_CODE_PHYSICAL_SIZE, 0u);
  expect(argv[4], &target, ET_C2_CORE_CORRUPT_DATA,
         ET_C2_CORE_CODE_C1_PROBE, 0u);
  expect(argv[6], &target, ET_C2_CORE_CORRUPT_DATA, ET_C2_CORE_CODE_FORMAT,
         ET_C2_FORMAT_CODE_LIMIT);
  expect(argv[8], &target, ET_C2_CORE_CORRUPT_DATA,
         ET_C2_CORE_CODE_OUTER_PROBE, 0u);

  /* Independent exact dimensions remain valid under the complete tuple. */
  check_joint_exact(argv[1], &target);
  {
    const int exact_indices[] = {3, 5, 7};
    size_t index;
    for (index = 0u; index < sizeof(exact_indices) / sizeof(exact_indices[0]);
         ++index) {
      et_c2_private_checkpoint_image *image = NULL;
      et_c2_checkpoint_core_error_v1 error = {0};
      error.struct_size = sizeof(error);
      CHECK(et_c2_private_checkpoint_load_image_v1(
                argv[exact_indices[index]], &target, ET_C2_FORMAT_LOAD,
                &image, &error) == 0);
      CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
      resources_flat();
    }
  }
  expect(argv[9], &profile, ET_C2_CORE_CORRUPT_DATA,
         ET_C2_CORE_CODE_FORMAT, ET_C2_FORMAT_CODE_CHECKSUM);
  printf("C2 operational native PASS: %u checks\n", checks);
  return 0;
}
