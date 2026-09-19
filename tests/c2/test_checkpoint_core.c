#define _GNU_SOURCE
#include "c2_checkpoint_core.h"
#include "c2_checkpoint_reader.h"
#include "checkpoint_io.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static unsigned checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    ++checks;                                                                  \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum mutation_kind { MUTATE_NONE, MUTATE_REPLACE, MUTATE_TRUNCATE, MUTATE_GROW };

static enum mutation_kind mutation;
static uint32_t mutation_stage;
static const uint8_t *replacement_bytes;
static size_t replacement_size;

static uint64_t read_u64(const uint8_t *bytes) {
  uint64_t value = 0u;
  unsigned index;
  for (index = 0u; index < 8u; ++index)
    value |= (uint64_t)bytes[index] << (8u * index);
  return value;
}

static void write_all(const char *path, const uint8_t *bytes, size_t size) {
  size_t completed = 0u;
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  CHECK(fd >= 0);
  while (completed < size) {
    ssize_t amount = write(fd, bytes + completed, size - completed);
    CHECK(amount > 0);
    completed += (size_t)amount;
  }
  CHECK(close(fd) == 0);
}

static uint8_t *read_all(const char *path, size_t *size) {
  struct stat metadata;
  uint8_t *bytes;
  size_t completed = 0u;
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  CHECK(fd >= 0);
  CHECK(fstat(fd, &metadata) == 0 && metadata.st_size >= 0);
  *size = (size_t)metadata.st_size;
  bytes = (uint8_t *)malloc(*size);
  CHECK(bytes != NULL);
  while (completed < *size) {
    ssize_t amount = read(fd, bytes + completed, *size - completed);
    CHECK(amount > 0);
    completed += (size_t)amount;
  }
  CHECK(close(fd) == 0);
  return bytes;
}

static char *join_path(const char *directory, const char *name) {
  size_t left = strlen(directory), right = strlen(name);
  char *result = (char *)malloc(left + right + 2u);
  CHECK(result != NULL);
  memcpy(result, directory, left);
  result[left] = '/';
  memcpy(result + left + 1u, name, right + 1u);
  return result;
}

static void mutation_hook(uint32_t stage, const char *path) {
  int fd;
  if (stage != mutation_stage || mutation == MUTATE_NONE) return;
  if (mutation == MUTATE_REPLACE) {
    write_all(path, replacement_bytes, replacement_size);
  } else if (mutation == MUTATE_TRUNCATE) {
    CHECK(replacement_size > 0u);
    CHECK(truncate(path, (off_t)(replacement_size - 1u)) == 0);
  } else {
    uint8_t byte = 0xa5u;
    fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC);
    CHECK(fd >= 0);
    CHECK(write(fd, &byte, 1u) == 1);
    CHECK(close(fd) == 0);
  }
  mutation = MUTATE_NONE;
}

static et_c2_checkpoint_limits_v1 default_limits(void) {
  et_c2_checkpoint_limits_v1 limits = {
      sizeof(limits), ET_C2_WIRE_MAX_FILE_BYTES,
      ET_C2_WIRE_MAX_METADATA_BYTES, ET_C2_WIRE_MAX_TENSOR_BYTES,
      ET_C2_WIRE_MAX_TENSORS, 1u};
  return limits;
}

static int32_t load(const char *path, const et_c2_checkpoint_limits_v1 *limits,
                    et_c2_private_checkpoint_image **image,
                    et_c2_checkpoint_core_error_v1 *error) {
  error->struct_size = sizeof(*error);
  return et_c2_private_checkpoint_load_image_v1(
      path, limits, ET_C2_FORMAT_LOAD, image, error);
}

static void expect_error(int32_t actual, uint32_t category, uint32_t code,
                         const et_c2_checkpoint_core_error_v1 *error) {
  CHECK(actual == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
}

static void test_success_and_guards(const char *path) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_private_checkpoint_image *stale;
  et_c2_private_checkpoint_image *forged =
      (et_c2_private_checkpoint_image *)(uintptr_t)1u;
  const et_c2_checkpoint_view_v1 *view = NULL;
  const et_c2_checkpoint_view_v1 *occupied =
      (const et_c2_checkpoint_view_v1 *)(uintptr_t)1u;
  et_c2_checkpoint_core_error_v1 error = {0};
  CHECK(load(path, &limits, &image, &error) == ET_C2_CORE_OK);
  CHECK(image != NULL);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 1u);
  error.struct_size = sizeof(error);
  CHECK(et_c2_private_checkpoint_image_view_v1(image, &view, &error) == 0);
  CHECK(view != NULL && view->total_tensor_count == 3u && view->bytes != NULL);
  CHECK(et_c2_private_checkpoint_image_view_v1(image, &occupied, &error) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(occupied == (const et_c2_checkpoint_view_v1 *)(uintptr_t)1u);
  view = NULL;
  CHECK(et_c2_private_checkpoint_image_view_v1(forged, &view, &error) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(view == NULL);
  stale = image;
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  CHECK(image == NULL && et_c2_checkpoint_core_test_live_images_v1() == 0u);
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  CHECK(et_c2_private_checkpoint_image_release_v1(&stale, &error) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(stale != NULL);
}

static void test_arguments_and_alloc(const char *path) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_private_checkpoint_image *occupied =
      (et_c2_private_checkpoint_image *)(uintptr_t)1u;
  et_c2_checkpoint_core_error_v1 error = {0};
  error.struct_size = sizeof(error);
  error.category = 77u;
  CHECK(et_c2_private_checkpoint_load_image_v1(
            path, &limits, ET_C2_FORMAT_LOAD,
            (et_c2_private_checkpoint_image **)(void *)&error, &error) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(error.category == 77u);
  error.struct_size = 0u;
  CHECK(et_c2_private_checkpoint_load_image_v1(
            path, &limits, ET_C2_FORMAT_LOAD, &image, &error) ==
        ET_C2_CORE_INVALID_ARGUMENT);
  CHECK(error.struct_size == 0u);
  error.struct_size = sizeof(error);
  expect_error(et_c2_private_checkpoint_load_image_v1(
                   NULL, &limits, ET_C2_FORMAT_LOAD, &image, &error),
               ET_C2_CORE_INVALID_ARGUMENT, ET_C2_CORE_CODE_ARGUMENT, &error);
  expect_error(et_c2_private_checkpoint_load_image_v1(
                   path, &limits, ET_C2_FORMAT_LOAD, &occupied, &error),
               ET_C2_CORE_INVALID_ARGUMENT, ET_C2_CORE_CODE_ARGUMENT, &error);
  CHECK(occupied == (et_c2_private_checkpoint_image *)(uintptr_t)1u);
  limits.maximum_tensors = 0u;
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_INVALID_ARGUMENT,
               ET_C2_CORE_CODE_ARGUMENT, &error);
  limits = default_limits();
  et_c2_checkpoint_core_test_fail_alloc_after_v1(0u);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_STAGING_ALLOCATION, &error);
  CHECK(et_checkpoint_io_status_stage_v1(error.native_status) ==
        ET_CHECKPOINT_IO_STAGE_ALLOCATE);
  CHECK(image == NULL && et_c2_checkpoint_core_test_live_images_v1() == 0u);
  et_c2_checkpoint_core_test_reset_v1();
}

static void test_exact_limits(const char *path, const uint8_t *base,
                              size_t base_size) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  uint64_t payload_offset = read_u64(base + 64u);
  uint64_t artifact_metadata = read_u64(base + 56u) +
                               read_u64(base + payload_offset + 56u);
  limits.maximum_file_bytes = base_size;
  limits.maximum_metadata_bytes = artifact_metadata;
  limits.maximum_tensor_bytes = 8u;
  limits.maximum_tensors = 3u;
  CHECK(load(path, &limits, &image, &error) == 0);
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  limits.maximum_file_bytes = base_size - 1u;
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_PHYSICAL_SIZE, &error);
  limits.maximum_file_bytes = base_size;
  limits.maximum_metadata_bytes = artifact_metadata - 1u;
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_C1_PROBE, &error);
  limits.maximum_metadata_bytes = artifact_metadata;
  limits.maximum_tensors = 2u;
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_OUTER_PROBE, &error);
  limits.maximum_tensors = 3u;
  limits.maximum_tensor_bytes = 7u;
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_FORMAT, &error);
}

static void set_mutation(enum mutation_kind kind, uint32_t stage,
                         const uint8_t *bytes, size_t size) {
  mutation = kind;
  mutation_stage = stage;
  replacement_bytes = bytes;
  replacement_size = size;
  et_c2_checkpoint_core_test_set_hook_v1(mutation_hook);
}

static void test_mutations(const char *path, const uint8_t *base,
                           size_t base_size, const uint8_t *counter,
                           size_t counter_size, const uint8_t *c1_header,
                           size_t c1_header_size) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  CHECK(base_size == counter_size && base_size == c1_header_size);
  set_mutation(MUTATE_REPLACE, ET_C2_CORE_TEST_AFTER_C1_PROBE,
               counter, counter_size);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_PROBED_HEADER_CHANGED, &error);
  CHECK(error.offset == 0u && image == NULL);
  write_all(path, base, base_size);
  set_mutation(MUTATE_REPLACE, ET_C2_CORE_TEST_AFTER_C1_PROBE,
               c1_header, c1_header_size);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_CORRUPT_DATA,
               ET_C2_CORE_CODE_PROBED_HEADER_CHANGED, &error);
  CHECK(error.offset == read_u64(base + 64u) && image == NULL);
  write_all(path, base, base_size);
  set_mutation(MUTATE_TRUNCATE, ET_C2_CORE_TEST_AFTER_C1_PROBE, NULL, base_size);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_NATIVE_IO, &error);
  CHECK(et_checkpoint_io_status_errno_v1(error.native_status) == ENODATA);
  write_all(path, base, base_size);
  set_mutation(MUTATE_GROW, ET_C2_CORE_TEST_AFTER_FULL_READ, NULL, base_size);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_NATIVE_IO, &error);
  CHECK(et_checkpoint_io_status_stage_v1(error.native_status) ==
        ET_CHECKPOINT_IO_STAGE_STAT_SOURCE);
  write_all(path, base, base_size);
  /* Same-size writes after retention are outside the promised snapshot model. */
  set_mutation(MUTATE_REPLACE, ET_C2_CORE_TEST_AFTER_FULL_READ,
               counter, counter_size);
  CHECK(load(path, &limits, &image, &error) == 0);
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  et_c2_checkpoint_core_test_reset_v1();
  write_all(path, base, base_size);
}

static void test_profile_and_precedence(const char *exact_path,
                                        const char *profile_path,
                                        const char *corrupt_path) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  CHECK(load(exact_path, &limits, &image, &error) == 0);
  CHECK(et_c2_private_checkpoint_image_release_v1(&image, &error) == 0);
  expect_error(load(profile_path, &limits, &image, &error),
               ET_C2_CORE_UNSUPPORTED, ET_C2_CORE_CODE_FORMAT, &error);
  expect_error(load(corrupt_path, &limits, &image, &error),
               ET_C2_CORE_CORRUPT_DATA, ET_C2_CORE_CODE_FORMAT, &error);
  limits.maximum_file_bytes = 1u;
  expect_error(load(profile_path, &limits, &image, &error),
               ET_C2_CORE_CORRUPT_DATA, ET_C2_CORE_CODE_PHYSICAL_SIZE, &error);
}

static void test_reader_failures(const char *path) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, 1u,
                                       EACCES);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_NATIVE_IO, &error);
  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, 1u,
                                       EIO);
  expect_error(load(path, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_READER_CLOSE, &error);
  CHECK(image == NULL);
  et_c2_checkpoint_reader_test_reset_v1();
}

static void test_hostile_kinds(const char *directory, const char *path) {
  et_c2_checkpoint_limits_v1 limits = default_limits();
  et_c2_private_checkpoint_image *image = NULL;
  et_c2_checkpoint_core_error_v1 error = {0};
  char *link = join_path(directory, "link");
  char *fifo = join_path(directory, "fifo");
  CHECK(symlink(path, link) == 0);
  expect_error(load(link, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_NATIVE_IO, &error);
  CHECK(et_checkpoint_io_status_errno_v1(error.native_status) == ELOOP);
  CHECK(mkfifo(fifo, 0600) == 0);
  expect_error(load(fifo, &limits, &image, &error), ET_C2_CORE_IO,
               ET_C2_CORE_CODE_NATIVE_IO, &error);
  CHECK(et_checkpoint_io_status_stage_v1(error.native_status) ==
        ET_CHECKPOINT_IO_STAGE_STAT_SOURCE);
  CHECK(unlink(link) == 0 && unlink(fifo) == 0);
  free(link);
  free(fifo);
}

int main(int argc, char **argv) {
  char directory_template[] = "/tmp/et-c2-core-XXXXXX";
  char *directory;
  char *path;
  uint8_t *base, *counter, *c1_header;
  size_t base_size, counter_size, c1_header_size;
  CHECK(argc == 7);
  base = read_all(argv[1], &base_size);
  counter = read_all(argv[2], &counter_size);
  c1_header = read_all(argv[3], &c1_header_size);
  directory = mkdtemp(directory_template);
  CHECK(directory != NULL);
  path = join_path(directory, "checkpoint.c2");
  write_all(path, base, base_size);
  et_c2_checkpoint_core_test_reset_v1();
  et_c2_checkpoint_reader_test_reset_v1();
  test_success_and_guards(path);
  test_arguments_and_alloc(path);
  test_exact_limits(path, base, base_size);
  test_mutations(path, base, base_size, counter, counter_size,
                 c1_header, c1_header_size);
  test_profile_and_precedence(argv[4], argv[5], argv[6]);
  test_reader_failures(path);
  test_hostile_kinds(directory, path);
  CHECK(et_c2_checkpoint_core_test_live_images_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);
  CHECK(unlink(path) == 0 && rmdir(directory) == 0);
  free(path); free(c1_header); free(counter); free(base);
  printf("C2 private checkpoint core: PASS (%u checks)\n", checks);
  return 0;
}
