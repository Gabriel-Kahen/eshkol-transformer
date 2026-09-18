#define _GNU_SOURCE
#include "c2_checkpoint_reader.h"

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

typedef struct test_bytevector {
  int64_t length;
  uint8_t bytes[];
} test_bytevector;

static test_bytevector *make_bytevector(size_t length, uint8_t fill) {
  test_bytevector *value = (test_bytevector *)malloc(sizeof(*value) + length);
  CHECK(value != NULL);
  value->length = (int64_t)length;
  memset(value->bytes, fill, length);
  return value;
}

static char *join_path(const char *directory, const char *name) {
  const size_t left = strlen(directory);
  const size_t right = strlen(name);
  char *path = (char *)malloc(left + right + 2u);
  CHECK(path != NULL);
  memcpy(path, directory, left);
  path[left] = '/';
  memcpy(path + left + 1u, name, right + 1u);
  return path;
}

static void write_exact_path(const char *path, const void *bytes,
                             size_t length) {
  const uint8_t *source = (const uint8_t *)bytes;
  size_t completed = 0u;
  int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  CHECK(descriptor >= 0);
  while (completed < length) {
    const ssize_t amount =
        write(descriptor, source + completed, length - completed);
    CHECK(amount > 0);
    completed += (size_t)amount;
  }
  CHECK(close(descriptor) == 0);
}

static void overwrite_exact_path(const char *path, off_t offset,
                                 const void *bytes, size_t length) {
  const uint8_t *source = (const uint8_t *)bytes;
  size_t completed = 0u;
  int descriptor = open(path, O_WRONLY | O_CLOEXEC);
  CHECK(descriptor >= 0);
  while (completed < length) {
    const ssize_t amount =
        pwrite(descriptor, source + completed, length - completed,
               offset + (off_t)completed);
    CHECK(amount > 0);
    completed += (size_t)amount;
  }
  CHECK(close(descriptor) == 0);
}

static void expect_failure(int64_t status, uint32_t stage, int error_number) {
  CHECK(status != 0);
  CHECK(et_checkpoint_io_status_stage_v1(status) == stage);
  CHECK(et_checkpoint_io_status_errno_v1(status) == (uint32_t)error_number);
  CHECK(!et_checkpoint_io_status_committed_v1(status));
}

static size_t event_count(uint32_t stage) {
  size_t count = 0u;
  size_t index;
  for (index = 0u; index < et_c2_checkpoint_reader_test_event_count_v1();
       ++index) {
    if (et_c2_checkpoint_reader_test_event_at_v1(index) == stage) {
      ++count;
    }
  }
  return count;
}

static void test_arguments(const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  et_c2_checkpoint_reader *occupied = (et_c2_checkpoint_reader *)(uintptr_t)1u;
  test_bytevector *bytes = make_bytevector(4u, 0x5au);
  uint64_t physical_size = 99u;

  expect_failure(et_c2_checkpoint_reader_open_v1(NULL, &reader, &physical_size),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_open_v1("", &reader, &physical_size),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_open_v1(path, NULL, &physical_size),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_open_v1(path, &reader, NULL),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(
      et_c2_checkpoint_reader_open_v1(path, &occupied, &physical_size),
      ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_read_exact_v1(NULL, 0u, bytes, 4u),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_validate_final_v1(NULL),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  expect_failure(et_c2_checkpoint_reader_close_v1(NULL),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);

  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  CHECK(physical_size == 16u);
  bytes->length = 3;
  expect_failure(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, bytes, 4u),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  bytes->length = 4;
  expect_failure(
      et_c2_checkpoint_reader_read_exact_v1(reader, UINT64_MAX, bytes, 4u),
      ET_CHECKPOINT_IO_STAGE_VALIDATE, EOVERFLOW);
  expect_failure(et_c2_checkpoint_reader_read_exact_v1(reader, 15u, bytes, 4u),
                 ET_CHECKPOINT_IO_STAGE_VALIDATE, EMSGSIZE);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);
  CHECK(reader == NULL);
  free(bytes);
}

static void test_same_fd_and_short_reads(const char *directory,
                                         const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  test_bytevector *probe = make_bytevector(4u, 0u);
  test_bytevector *whole = make_bytevector(16u, 0u);
  uint64_t physical_size = 0u;
  char *replacement = join_path(directory, "replacement.bin");

  et_c2_checkpoint_reader_test_reset_v1();
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  CHECK(physical_size == 16u);
  CHECK(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, probe, 4u) == 0);
  CHECK(memcmp(probe->bytes, "0123", 4u) == 0);

  write_exact_path(replacement, "path-replacement", 16u);
  CHECK(rename(replacement, path) == 0);
  et_c2_checkpoint_reader_test_set_short_read_v1(3u);
  CHECK(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, whole, 16u) == 0);
  CHECK(memcmp(whole->bytes, "0123456789abcdef", 16u) == 0);
  CHECK(event_count(ET_CHECKPOINT_IO_STAGE_READ_SOURCE) >= 6u);
  CHECK(et_c2_checkpoint_reader_validate_final_v1(reader) == 0);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);

  free(replacement);
  free(whole);
  free(probe);
}

static void test_in_place_header_change(const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  test_bytevector *probe = make_bytevector(4u, 0u);
  test_bytevector *whole = make_bytevector(16u, 0u);
  uint64_t physical_size = 0u;

  write_exact_path(path, "0123456789abcdef", 16u);
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  CHECK(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, probe, 4u) == 0);
  overwrite_exact_path(path, 0, "WXYZ", 4u);
  CHECK(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, whole, 16u) == 0);
  CHECK(memcmp(probe->bytes, whole->bytes, 4u) != 0);
  /* Same-size mutation is intentionally exposed to the caller's comparison. */
  CHECK(et_c2_checkpoint_reader_validate_final_v1(reader) == 0);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);
  free(whole);
  free(probe);
}

static void test_size_and_eof_validation(const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  test_bytevector *bytes = make_bytevector(16u, 0x5au);
  uint64_t physical_size = 0u;
  int64_t status;

  write_exact_path(path, "0123456789abcdef", 16u);
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  CHECK(truncate(path, 8) == 0);
  status = et_c2_checkpoint_reader_validate_final_v1(reader);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EMSGSIZE);
  status = et_c2_checkpoint_reader_read_exact_v1(reader, 0u, bytes, 16u);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_READ_SOURCE, ENODATA);
  CHECK(memcmp(bytes->bytes, "01234567", 8u) == 0);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);

  write_exact_path(path, "0123456789abcdef", 16u);
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  CHECK(truncate(path, 17) == 0);
  status = et_c2_checkpoint_reader_validate_final_v1(reader);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EMSGSIZE);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);

  write_exact_path(path, "0123456789abcdef", 16u);
  et_c2_checkpoint_reader_test_reset_v1();
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, 1u,
                                       EINTR);
  CHECK(et_c2_checkpoint_reader_validate_final_v1(reader) == 0);
  CHECK(event_count(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE) == 2u);
  CHECK(event_count(ET_CHECKPOINT_IO_STAGE_READ_SOURCE) == 2u);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);
  free(bytes);
}

static void test_failures_and_consuming_close(const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  test_bytevector *bytes = make_bytevector(16u, 0x5au);
  uint64_t physical_size = 77u;
  int64_t status;

  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, 1u,
                                       EACCES);
  status = et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, EACCES);
  CHECK(reader == NULL);
  CHECK(physical_size == 77u);

  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, 1u,
                                       EIO);
  status = et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EIO);
  CHECK(reader == NULL);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);

  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_ALLOCATE, 1u,
                                       ENOMEM);
  status = et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_ALLOCATE, ENOMEM);
  CHECK(reader == NULL);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);

  et_c2_checkpoint_reader_test_reset_v1();
  CHECK(et_c2_checkpoint_reader_open_v1(path, &reader, &physical_size) == 0);
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, 1u,
                                       EINTR);
  CHECK(et_c2_checkpoint_reader_read_exact_v1(reader, 0u, bytes, 16u) == 0);
  CHECK(memcmp(bytes->bytes, "0123456789abcdef", 16u) == 0);

  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_set_short_read_v1(0u);
  status = et_c2_checkpoint_reader_read_exact_v1(reader, 0u, bytes, 16u);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_READ_SOURCE, ENODATA);

  et_c2_checkpoint_reader_test_reset_v1();
  et_c2_checkpoint_reader_test_fail_v1(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, 1u,
                                       EIO);
  status = et_c2_checkpoint_reader_close_v1(&reader);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, EIO);
  CHECK(reader == NULL);
  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_close_v1(&reader) == 0);
  free(bytes);
}

static void test_hostile_kinds(const char *directory, const char *path) {
  et_c2_checkpoint_reader *reader = NULL;
  uint64_t physical_size = 0u;
  char *link_path = join_path(directory, "checkpoint-link");
  char *fifo_path = join_path(directory, "checkpoint-fifo");
  int64_t status;

  CHECK(symlink(path, link_path) == 0);
  status = et_c2_checkpoint_reader_open_v1(link_path, &reader, &physical_size);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, ELOOP);
  CHECK(mkfifo(fifo_path, 0600) == 0);
  status = et_c2_checkpoint_reader_open_v1(fifo_path, &reader, &physical_size);
  expect_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EINVAL);
  CHECK(unlink(fifo_path) == 0);
  CHECK(unlink(link_path) == 0);
  free(fifo_path);
  free(link_path);
}

int main(void) {
  char directory_template[] = "/tmp/et-c2-reader-XXXXXX";
  char *directory = mkdtemp(directory_template);
  char *path;
  CHECK(directory != NULL);
  path = join_path(directory, "checkpoint.bin");
  write_exact_path(path, "0123456789abcdef", 16u);

  test_arguments(path);
  write_exact_path(path, "0123456789abcdef", 16u);
  test_same_fd_and_short_reads(directory, path);
  test_in_place_header_change(path);
  test_size_and_eof_validation(path);
  write_exact_path(path, "0123456789abcdef", 16u);
  test_failures_and_consuming_close(path);
  test_hostile_kinds(directory, path);

  CHECK(et_c2_checkpoint_reader_test_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_live_count_v1() == 0u);
  CHECK(et_c2_checkpoint_reader_test_fd_peak_count_v1() == 1u);
  CHECK(unlink(path) == 0);
  CHECK(rmdir(directory) == 0);
  free(path);
  printf("C2 same-fd checkpoint reader: PASS (%u checks)\n", checks);
  return 0;
}
