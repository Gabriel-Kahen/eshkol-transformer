#define _GNU_SOURCE
#include "checkpoint_io.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
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
  char *path = (char *)malloc(left + 1u + right + 1u);
  CHECK(path != NULL);
  memcpy(path, directory, left);
  path[left] = '/';
  memcpy(path + left + 1u, name, right + 1u);
  return path;
}

static void raw_write(const char *path, const uint8_t *bytes, size_t length) {
  size_t offset = 0u;
  int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  CHECK(descriptor >= 0);
  while (offset < length) {
    const ssize_t amount = write(descriptor, bytes + offset, length - offset);
    CHECK(amount > 0);
    offset += (size_t)amount;
  }
  CHECK(close(descriptor) == 0);
}

static size_t raw_read(const char *path, uint8_t *bytes, size_t capacity) {
  size_t offset = 0u;
  int descriptor = open(path, O_RDONLY | O_CLOEXEC);
  CHECK(descriptor >= 0);
  for (;;) {
    const ssize_t amount = read(descriptor, bytes + offset, capacity - offset);
    CHECK(amount >= 0);
    if (amount == 0) {
      break;
    }
    offset += (size_t)amount;
    CHECK(offset <= capacity);
  }
  CHECK(close(descriptor) == 0);
  return offset;
}

static void expect_file(const char *path, const uint8_t *bytes, size_t length) {
  uint8_t *actual = (uint8_t *)malloc(length == 0u ? 1u : length);
  CHECK(actual != NULL);
  CHECK(raw_read(path, actual, length == 0u ? 1u : length) == length);
  CHECK(memcmp(actual, bytes, length) == 0);
  free(actual);
}

static size_t temp_count(const char *directory) {
  size_t count = 0u;
  DIR *stream = opendir(directory);
  struct dirent *entry;
  CHECK(stream != NULL);
  while ((entry = readdir(stream)) != NULL) {
    if (strncmp(entry->d_name, ".et-c1-", 7u) == 0) {
      ++count;
    }
  }
  CHECK(closedir(stream) == 0);
  return count;
}

static void assert_failure(int64_t status, uint32_t stage, int error_number,
                           int committed) {
  if (et_checkpoint_io_status_stage_v1(status) != stage ||
      et_checkpoint_io_status_errno_v1(status) != (uint32_t)error_number ||
      et_checkpoint_io_status_committed_v1(status) != committed) {
    fprintf(stderr,
            "status mismatch: got stage=%u errno=%u committed=%d; expected "
            "stage=%u errno=%d committed=%d\n",
            et_checkpoint_io_status_stage_v1(status),
            et_checkpoint_io_status_errno_v1(status),
            et_checkpoint_io_status_committed_v1(status), stage, error_number,
            committed);
  }
  CHECK(status != 0);
  CHECK(et_checkpoint_io_status_stage_v1(status) == stage);
  CHECK(et_checkpoint_io_status_errno_v1(status) == (uint32_t)error_number);
  CHECK(et_checkpoint_io_status_committed_v1(status) == committed);
}

static size_t first_event(uint32_t stage) {
  size_t index;
  for (index = 0u; index < et_checkpoint_io_test_event_count_v1(); ++index) {
    if (et_checkpoint_io_test_event_at_v1(index) == stage) {
      return index;
    }
  }
  return (size_t)-1;
}

static size_t event_count(uint32_t stage) {
  size_t count = 0u;
  size_t index;
  for (index = 0u; index < et_checkpoint_io_test_event_count_v1(); ++index) {
    if (et_checkpoint_io_test_event_at_v1(index) == stage) {
      ++count;
    }
  }
  return count;
}

static void test_validation(const char *path) {
  test_bytevector *value = make_bytevector(4u, 0x5au);
  int64_t status;
  CHECK(et_checkpoint_io_abi_major_v1() == 1);
  CHECK(et_checkpoint_io_abi_minor_v1() == 0);
  status = et_checkpoint_io_atomic_write_v1(NULL, value, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, EINVAL, 0);
  status = et_checkpoint_io_atomic_write_v1("", value, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, EINVAL, 0);
  status = et_checkpoint_io_atomic_write_v1(".", value, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, EINVAL, 0);
  status = et_checkpoint_io_atomic_write_v1("/", value, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, EINVAL, 0);
  status = et_checkpoint_io_atomic_write_v1(path, value, 4u, 2);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL, 0);
  value->length = 3;
  status = et_checkpoint_io_atomic_write_v1(path, value, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL, 0);
  value->length = 4;
  status = et_checkpoint_io_read_exact_v1(path, value, 5u, 4u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL, 0);
  status = et_checkpoint_io_read_exact_v1(path, value, 4u, 0u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL, 0);
  free(value);
}

static void test_round_trip(const char *directory, const char *path) {
  test_bytevector *source = make_bytevector(257u, 0u);
  test_bytevector *destination = make_bytevector(257u, 0xa5u);
  test_bytevector *empty = make_bytevector(0u, 0u);
  char *empty_path = join_path(directory, "empty.bin");
  size_t index;
  for (index = 0u; index < 257u; ++index) {
    source->bytes[index] = (uint8_t)index;
  }
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_short_io_v1(7u);
  CHECK(et_checkpoint_io_atomic_write_v1(path, source, 257u, 0) == 0);
  CHECK(first_event(ET_CHECKPOINT_IO_STAGE_WRITE_TEMP) <
        first_event(ET_CHECKPOINT_IO_STAGE_SYNC_TEMP));
  CHECK(first_event(ET_CHECKPOINT_IO_STAGE_SYNC_TEMP) <
        first_event(ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP));
  CHECK(first_event(ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP) <
        first_event(ET_CHECKPOINT_IO_STAGE_PUBLISH));
  CHECK(first_event(ET_CHECKPOINT_IO_STAGE_PUBLISH) <
        first_event(ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY));
  CHECK(first_event(ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY) <
        first_event(ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY));
  CHECK(et_checkpoint_io_test_event_count_v1() > 40u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_short_io_v1(5u);
  CHECK(et_checkpoint_io_read_exact_v1(path, destination, 257u, 1024u) == 0);
  CHECK(memcmp(source->bytes, destination->bytes, 257u) == 0);
  CHECK(temp_count(directory) == 0u);
  CHECK(et_checkpoint_io_atomic_write_v1(empty_path, empty, 0u, 0) == 0);
  CHECK(et_checkpoint_io_read_exact_v1(empty_path, empty, 0u, 1u) == 0);
  expect_file(empty_path, (const uint8_t *)"", 0u);
  CHECK(unlink(empty_path) == 0);
  free(empty_path);
  free(empty);
  free(destination);
  free(source);
}

static void test_preexisting_and_failpoints(const char *directory,
                                            const char *path) {
  static const uint8_t old_bytes[] = "old-artifact";
  test_bytevector *replacement = make_bytevector(12u, 0u);
  struct stat before;
  struct stat after;
  const uint32_t stages[] = {
      ET_CHECKPOINT_IO_STAGE_ALLOCATE,    ET_CHECKPOINT_IO_STAGE_OPEN_PARENT,
      ET_CHECKPOINT_IO_STAGE_RANDOM_TEMP, ET_CHECKPOINT_IO_STAGE_CREATE_TEMP,
      ET_CHECKPOINT_IO_STAGE_WRITE_TEMP,  ET_CHECKPOINT_IO_STAGE_SYNC_TEMP,
      ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP,  ET_CHECKPOINT_IO_STAGE_PUBLISH};
  size_t index;
  int64_t status;
  memcpy(replacement->bytes, "new-artifact", 12u);
  raw_write(path, old_bytes, sizeof(old_bytes) - 1u);
  CHECK(stat(path, &before) == 0);
  et_checkpoint_io_test_reset_v1();
  status = et_checkpoint_io_atomic_write_v1(path, replacement, 12u, 0);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_PUBLISH, EEXIST, 0);
  CHECK(stat(path, &after) == 0);
  CHECK(before.st_ino == after.st_ino);
  expect_file(path, old_bytes, sizeof(old_bytes) - 1u);
  CHECK(temp_count(directory) == 0u);

  for (index = 0u; index < sizeof(stages) / sizeof(stages[0]); ++index) {
    raw_write(path, old_bytes, sizeof(old_bytes) - 1u);
    CHECK(stat(path, &before) == 0);
    et_checkpoint_io_test_reset_v1();
    et_checkpoint_io_test_fail_v1(stages[index], 1u, EIO);
    status = et_checkpoint_io_atomic_write_v1(path, replacement, 12u, 1);
    assert_failure(status, stages[index], EIO, 0);
    CHECK(stat(path, &after) == 0);
    CHECK(before.st_ino == after.st_ino);
    expect_file(path, old_bytes, sizeof(old_bytes) - 1u);
    CHECK(temp_count(directory) == 0u);
  }

  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_WRITE_TEMP, 1u, EINTR);
  CHECK(et_checkpoint_io_atomic_write_v1(path, replacement, 12u, 1) == 0);
  expect_file(path, replacement->bytes, 12u);

  raw_write(path, old_bytes, sizeof(old_bytes) - 1u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY, 1u, EIO);
  status = et_checkpoint_io_atomic_write_v1(path, replacement, 12u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY, EIO, 1);
  expect_file(path, replacement->bytes, 12u);
  CHECK(temp_count(directory) == 0u);

  raw_write(path, old_bytes, sizeof(old_bytes) - 1u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY, 1u,
                                EIO);
  status = et_checkpoint_io_atomic_write_v1(path, replacement, 12u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY, EIO, 1);
  expect_file(path, replacement->bytes, 12u);
  free(replacement);
}

static void test_zero_progress(const char *directory, const char *write_path,
                               const char *read_path) {
  static const uint8_t old_bytes[] = "old-artifact";
  static const uint8_t read_bytes[] = "read1234";
  test_bytevector *replacement = make_bytevector(12u, 0u);
  test_bytevector *destination = make_bytevector(8u, 0x5au);
  uint8_t original[8];
  struct stat before;
  struct stat after;
  char *temporary;
  int64_t status;

  memcpy(replacement->bytes, "new-artifact", 12u);
  memset(original, 0x5a, sizeof(original));
  raw_write(write_path, old_bytes, sizeof(old_bytes) - 1u);
  CHECK(stat(write_path, &before) == 0);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_short_io_v1(0u);
  status = et_checkpoint_io_atomic_write_v1(write_path, replacement, 12u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_WRITE_TEMP, EIO, 0);
  CHECK(event_count(ET_CHECKPOINT_IO_STAGE_WRITE_TEMP) == 1u);
  CHECK(stat(write_path, &after) == 0);
  CHECK(before.st_ino == after.st_ino);
  expect_file(write_path, old_bytes, sizeof(old_bytes) - 1u);
  temporary = join_path(directory, et_checkpoint_io_test_last_temp_v1());
  CHECK(lstat(temporary, &after) != 0);
  CHECK(errno == ENOENT);
  CHECK(temp_count(directory) == 0u);
  free(temporary);

  raw_write(read_path, read_bytes, sizeof(read_bytes) - 1u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_short_io_v1(0u);
  status = et_checkpoint_io_read_exact_v1(read_path, destination, 8u, 32u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_READ_SOURCE, ENODATA, 0);
  CHECK(event_count(ET_CHECKPOINT_IO_STAGE_READ_SOURCE) == 1u);
  CHECK(memcmp(destination->bytes, original, sizeof(original)) == 0);
  CHECK(temp_count(directory) == 0u);

  /* Reset is a distinct disabled state; it does not inject zero progress. */
  et_checkpoint_io_test_reset_v1();
  CHECK(et_checkpoint_io_read_exact_v1(read_path, destination, 8u, 32u) == 0);
  CHECK(memcmp(destination->bytes, read_bytes, sizeof(read_bytes) - 1u) == 0);
  CHECK(et_checkpoint_io_atomic_write_v1(write_path, replacement, 12u, 1) ==
        0);
  expect_file(write_path, replacement->bytes, 12u);

  free(destination);
  free(replacement);
}

static void test_collision_and_orphan(const char *directory, const char *path) {
  uint8_t random[16] = {0};
  test_bytevector *replacement = make_bytevector(3u, 'n');
  char *collision =
      join_path(directory, ".et-c1-00000000000000000000000000000000.tmp");
  int64_t status;
  struct stat metadata;
  raw_write(path, (const uint8_t *)"old", 3u);
  raw_write(collision, (const uint8_t *)"attacker", 8u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_random_v1(random);
  status = et_checkpoint_io_atomic_write_v1(path, replacement, 3u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_CREATE_TEMP, EEXIST, 0);
  expect_file(path, (const uint8_t *)"old", 3u);
  expect_file(collision, (const uint8_t *)"attacker", 8u);
  CHECK(unlink(collision) == 0);
  free(collision);

  raw_write(path, (const uint8_t *)"old", 3u);
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_set_random_v1(random);
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_CLEANUP_TEMP, 1u,
                                EACCES);
  status = et_checkpoint_io_atomic_write_v1(path, replacement, 3u, 0);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_CLEANUP_TEMP, EACCES, 0);
  collision = join_path(directory, et_checkpoint_io_test_last_temp_v1());
  expect_file(collision, replacement->bytes, 3u);
  CHECK(stat(collision, &metadata) == 0);
  CHECK((metadata.st_mode & 0777) == 0600);
  expect_file(path, (const uint8_t *)"old", 3u);
  CHECK(unlink(collision) == 0);
  free(collision);
  free(replacement);
}

static void test_read_rejection(const char *directory, const char *path) {
  test_bytevector *destination = make_bytevector(8u, 0x5au);
  uint8_t original[8];
  const uint32_t stages[] = {ET_CHECKPOINT_IO_STAGE_OPEN_PARENT,
                             ET_CHECKPOINT_IO_STAGE_ALLOCATE,
                             ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE,
                             ET_CHECKPOINT_IO_STAGE_STAT_SOURCE,
                             ET_CHECKPOINT_IO_STAGE_READ_SOURCE,
                             ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE,
                             ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY};
  size_t index;
  int64_t status;
  char *short_path = join_path(directory, "short.bin");
  char *long_path = join_path(directory, "long.bin");
  char *real_path = join_path(directory, "real.bin");
  char *link_path = join_path(directory, "link.bin");
  char *fifo_path = join_path(directory, "fifo.bin");
  memset(original, 0x5a, sizeof(original));
  raw_write(path, (const uint8_t *)"12345678", 8u);
  for (index = 0u; index < sizeof(stages) / sizeof(stages[0]); ++index) {
    memcpy(destination->bytes, original, sizeof(original));
    et_checkpoint_io_test_reset_v1();
    et_checkpoint_io_test_fail_v1(stages[index], 1u, EIO);
    status = et_checkpoint_io_read_exact_v1(path, destination, 8u, 32u);
    assert_failure(status, stages[index], EIO, 0);
    CHECK(memcmp(destination->bytes, original, sizeof(original)) == 0);
  }
  et_checkpoint_io_test_reset_v1();
  et_checkpoint_io_test_fail_v1(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, 1u, EINTR);
  CHECK(et_checkpoint_io_read_exact_v1(path, destination, 8u, 32u) == 0);
  CHECK(memcmp(destination->bytes, "12345678", 8u) == 0);

  raw_write(short_path, (const uint8_t *)"123", 3u);
  memset(destination->bytes, 0x5a, 8u);
  status = et_checkpoint_io_read_exact_v1(short_path, destination, 8u, 32u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EMSGSIZE, 0);
  CHECK(memcmp(destination->bytes, original, 8u) == 0);
  raw_write(long_path, (const uint8_t *)"123456789", 9u);
  status = et_checkpoint_io_read_exact_v1(long_path, destination, 8u, 32u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EMSGSIZE, 0);

  raw_write(real_path, (const uint8_t *)"12345678", 8u);
  CHECK(symlink(real_path, link_path) == 0);
  status = et_checkpoint_io_read_exact_v1(link_path, destination, 8u, 32u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, ELOOP, 0);
  CHECK(mkfifo(fifo_path, 0600) == 0);
  status = et_checkpoint_io_read_exact_v1(fifo_path, destination, 8u, 32u);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EINVAL, 0);

  CHECK(unlink(short_path) == 0);
  CHECK(unlink(long_path) == 0);
  CHECK(unlink(link_path) == 0);
  CHECK(unlink(real_path) == 0);
  CHECK(unlink(fifo_path) == 0);
  free(short_path);
  free(long_path);
  free(real_path);
  free(link_path);
  free(fifo_path);
  free(destination);
}

static void test_write_replaces_symlink(const char *directory) {
  char *target = join_path(directory, "symlink-target.bin");
  char *link = join_path(directory, "symlink-destination.bin");
  test_bytevector *replacement = make_bytevector(4u, 'N');
  struct stat metadata;
  char *real_directory = join_path(directory, "real-directory");
  char *linked_directory = join_path(directory, "linked-directory");
  char *through_link;
  int64_t status;
  raw_write(target, (const uint8_t *)"OLD", 3u);
  CHECK(symlink(target, link) == 0);
  CHECK(et_checkpoint_io_atomic_write_v1(link, replacement, 4u, 1) == 0);
  CHECK(lstat(link, &metadata) == 0);
  CHECK(S_ISREG(metadata.st_mode));
  expect_file(link, replacement->bytes, 4u);
  expect_file(target, (const uint8_t *)"OLD", 3u);
  CHECK(mkdir(real_directory, 0700) == 0);
  CHECK(symlink(real_directory, linked_directory) == 0);
  through_link = join_path(linked_directory, "blocked.bin");
  status = et_checkpoint_io_atomic_write_v1(through_link, replacement, 4u, 1);
  assert_failure(status, ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, ENOTDIR, 0);
  CHECK(access(through_link, F_OK) != 0);
  CHECK(errno == ENOENT);
  CHECK(unlink(linked_directory) == 0);
  CHECK(rmdir(real_directory) == 0);
  CHECK(unlink(link) == 0);
  CHECK(unlink(target) == 0);
  free(replacement);
  free(through_link);
  free(linked_directory);
  free(real_directory);
  free(link);
  free(target);
}

static int buffer_is_uniform(const uint8_t *bytes, size_t length, uint8_t left,
                             uint8_t right) {
  size_t index;
  if (length == 0u || (bytes[0] != left && bytes[0] != right)) {
    return 0;
  }
  for (index = 1u; index < length; ++index) {
    if (bytes[index] != bytes[0]) {
      return 0;
    }
  }
  return 1;
}

static void test_atomic_visibility(const char *path) {
  const size_t length = 65536u;
  test_bytevector *left = make_bytevector(length, 'A');
  test_bytevector *right = make_bytevector(length, 'B');
  uint8_t *observed = (uint8_t *)malloc(length);
  pid_t child;
  int child_status = 0;
  unsigned observations = 0u;
  CHECK(observed != NULL);
  CHECK(et_checkpoint_io_atomic_write_v1(path, left, length, 1) == 0);
  child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    unsigned iteration;
    for (iteration = 0u; iteration < 16u; ++iteration) {
      test_bytevector *value = (iteration & 1u) == 0u ? right : left;
      if (et_checkpoint_io_atomic_write_v1(path, value, length, 1) != 0) {
        _exit(2);
      }
    }
    _exit(0);
  }
  for (;;) {
    const pid_t result = waitpid(child, &child_status, WNOHANG);
    CHECK(result >= 0);
    CHECK(raw_read(path, observed, length) == length);
    CHECK(buffer_is_uniform(observed, length, 'A', 'B'));
    ++observations;
    if (result == child) {
      break;
    }
  }
  CHECK(WIFEXITED(child_status));
  CHECK(WEXITSTATUS(child_status) == 0);
  CHECK(observations > 0u);
  free(observed);
  free(right);
  free(left);
}

static void remove_directory_contents(const char *directory) {
  DIR *stream = opendir(directory);
  struct dirent *entry;
  CHECK(stream != NULL);
  while ((entry = readdir(stream)) != NULL) {
    char *path;
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    path = join_path(directory, entry->d_name);
    CHECK(unlink(path) == 0);
    free(path);
  }
  CHECK(closedir(stream) == 0);
}

int main(void) {
  char directory_template[] = "/tmp/et-c1-io-XXXXXX";
  char *directory = mkdtemp(directory_template);
  char *roundtrip;
  char *destination;
  char *read_path;
  char *atomic_path;
  CHECK(directory != NULL);
  roundtrip = join_path(directory, "roundtrip.bin");
  destination = join_path(directory, "destination.bin");
  read_path = join_path(directory, "read.bin");
  atomic_path = join_path(directory, "atomic.bin");
  test_validation(destination);
  test_round_trip(directory, roundtrip);
  test_preexisting_and_failpoints(directory, destination);
  test_zero_progress(directory, destination, read_path);
  test_collision_and_orphan(directory, destination);
  test_read_rejection(directory, read_path);
  test_write_replaces_symlink(directory);
  test_atomic_visibility(atomic_path);
  CHECK(temp_count(directory) == 0u);
  remove_directory_contents(directory);
  CHECK(rmdir(directory) == 0);
  free(atomic_path);
  free(read_path);
  free(destination);
  free(roundtrip);
  printf("C1 checkpoint I/O: PASS\n");
  return 0;
}
