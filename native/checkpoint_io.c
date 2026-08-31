#define _GNU_SOURCE

#include "checkpoint_io.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(O_CLOEXEC) || !defined(O_DIRECTORY) ||     \
    !defined(O_NOFOLLOW) || !defined(O_NONBLOCK)
#error "checkpoint I/O ABI 1 requires the reviewed Linux open/rename surface"
#endif
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1u << 0)
#endif

#define ET_TEMP_RANDOM_BYTES 16u
#define ET_TEMP_ATTEMPTS 128u
#define ET_TEMP_NAME_BYTES 44u

_Static_assert(CHAR_BIT == 8 && sizeof(int64_t) == 8u && sizeof(uint64_t) == 8u,
               "checkpoint I/O ABI requires exact eight-byte integers");

typedef struct et_path_parts {
  char *parent;
  char *basename;
} et_path_parts;

#ifdef ET_CHECKPOINT_IO_TESTING
#define ET_TEST_EVENT_CAPACITY 4096u
static struct {
  uint32_t fail_stage;
  uint32_t fail_occurrence;
  uint32_t stage_occurrences[256];
  int32_t fail_errno;
  size_t short_io;
  int deterministic_random;
  uint8_t random_bytes[ET_TEMP_RANDOM_BYTES];
  uint32_t events[ET_TEST_EVENT_CAPACITY];
  size_t event_count;
  char last_temp[ET_TEMP_NAME_BYTES];
} et_test;

void et_checkpoint_io_test_reset_v1(void) {
  memset(&et_test, 0, sizeof(et_test));
}

void et_checkpoint_io_test_fail_v1(uint32_t stage, uint32_t occurrence,
                                   int32_t error_number) {
  et_test.fail_stage = stage;
  et_test.fail_occurrence = occurrence;
  et_test.fail_errno = error_number;
}

void et_checkpoint_io_test_set_short_io_v1(size_t maximum_bytes) {
  et_test.short_io = maximum_bytes;
}

void et_checkpoint_io_test_set_random_v1(const uint8_t bytes[16]) {
  if (bytes == NULL) {
    et_test.deterministic_random = 0;
    memset(et_test.random_bytes, 0, sizeof(et_test.random_bytes));
    return;
  }
  et_test.deterministic_random = 1;
  memcpy(et_test.random_bytes, bytes, sizeof(et_test.random_bytes));
}

size_t et_checkpoint_io_test_event_count_v1(void) {
  return et_test.event_count;
}

uint32_t et_checkpoint_io_test_event_at_v1(size_t index) {
  return index < et_test.event_count ? et_test.events[index] : 0u;
}

const char *et_checkpoint_io_test_last_temp_v1(void) {
  return et_test.last_temp;
}

static int test_should_fail(uint32_t stage) {
  uint32_t occurrence;
  if (et_test.event_count < ET_TEST_EVENT_CAPACITY) {
    et_test.events[et_test.event_count++] = stage;
  }
  occurrence = ++et_test.stage_occurrences[stage & 0xffu];
  if (et_test.fail_stage == stage && et_test.fail_occurrence == occurrence) {
    et_test.fail_stage = 0u;
    errno = et_test.fail_errno > 0 ? et_test.fail_errno : EIO;
    return 1;
  }
  return 0;
}
#else
static int test_should_fail(uint32_t stage) {
  (void)stage;
  return 0;
}
#endif

static int64_t make_status(uint32_t stage, int error_number, int committed) {
  uint64_t encoded_errno;
  uint64_t status;
  if (error_number <= 0 ||
      error_number > (int)ET_CHECKPOINT_IO_STATUS_ERRNO_MASK) {
    error_number = EIO;
  }
  encoded_errno = (uint64_t)(uint32_t)error_number;
  status = encoded_errno |
           ((uint64_t)(stage & (uint32_t)ET_CHECKPOINT_IO_STATUS_STAGE_MASK)
            << ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT);
  if (committed) {
    status |= ET_CHECKPOINT_IO_STATUS_COMMITTED;
  }
  return (int64_t)status;
}

static void *wrapped_payload_malloc(size_t length) {
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_ALLOCATE)) {
    return NULL;
  }
  return malloc(length);
}

static int pointer_span_fits(const void *pointer, uint64_t length) {
  const uintptr_t start = (uintptr_t)pointer;
  return pointer != NULL && length <= (uint64_t)SIZE_MAX - sizeof(int64_t) &&
         start <= UINTPTR_MAX - sizeof(int64_t) - (uintptr_t)length;
}

static int validate_bytevector(const void *header, uint64_t expected_length,
                               const uint8_t **bytes) {
  int64_t observed_length;
  if (expected_length > (uint64_t)INT64_MAX ||
      !pointer_span_fits(header, expected_length)) {
    errno = EOVERFLOW;
    return -1;
  }
  memcpy(&observed_length, header, sizeof(observed_length));
  if (observed_length < 0 || (uint64_t)observed_length != expected_length) {
    errno = EINVAL;
    return -1;
  }
  *bytes = (const uint8_t *)header + sizeof(int64_t);
  return 0;
}

static void free_path_parts(et_path_parts *parts) {
  free(parts->parent);
  free(parts->basename);
  parts->parent = NULL;
  parts->basename = NULL;
}

static int split_path(const char *path, et_path_parts *parts) {
  const char *slash;
  const char *base;
  size_t parent_length;
  size_t base_length;
  memset(parts, 0, sizeof(*parts));
  if (path == NULL || path[0] == '\0') {
    errno = EINVAL;
    return -1;
  }
  slash = strrchr(path, '/');
  base = slash == NULL ? path : slash + 1;
  base_length = strlen(base);
  if (base_length == 0u || (base_length == 1u && base[0] == '.') ||
      (base_length == 2u && base[0] == '.' && base[1] == '.')) {
    errno = EINVAL;
    return -1;
  }
  parts->basename = (char *)malloc(base_length + 1u);
  if (parts->basename == NULL) {
    return -1;
  }
  memcpy(parts->basename, base, base_length + 1u);
  if (slash == NULL) {
    parts->parent = (char *)malloc(2u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, ".", 2u);
    }
  } else if (slash == path) {
    parts->parent = (char *)malloc(2u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, "/", 2u);
    }
  } else {
    parent_length = (size_t)(slash - path);
    while (parent_length > 1u && path[parent_length - 1u] == '/') {
      --parent_length;
    }
    parts->parent = (char *)malloc(parent_length + 1u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, path, parent_length);
      parts->parent[parent_length] = '\0';
    }
  }
  if (parts->parent == NULL) {
    free_path_parts(parts);
    errno = ENOMEM;
    return -1;
  }
  return 0;
}

static int open_parent(const char *path, et_path_parts *parts) {
  int descriptor;
  if (split_path(path, parts) != 0) {
    return -1;
  }
  do {
    if (test_should_fail(ET_CHECKPOINT_IO_STAGE_OPEN_PARENT)) {
      descriptor = -1;
    } else {
      descriptor =
          open(parts->parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    }
  } while (descriptor < 0 && errno == EINTR);
  if (descriptor < 0) {
    free_path_parts(parts);
  }
  return descriptor;
}

static int wrapped_openat(int directory, const char *name, int flags,
                          mode_t mode, uint32_t stage) {
  int descriptor;
  do {
    if (test_should_fail(stage)) {
      descriptor = -1;
    } else {
      descriptor = openat(directory, name, flags, mode);
    }
  } while (descriptor < 0 && errno == EINTR);
  return descriptor;
}

static int wrapped_fstat(int descriptor, struct stat *metadata) {
  int result;
  do {
    if (test_should_fail(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE)) {
      result = -1;
    } else {
      result = fstat(descriptor, metadata);
    }
  } while (result < 0 && errno == EINTR);
  return result;
}

static ssize_t wrapped_read(int descriptor, void *buffer, size_t length) {
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_READ_SOURCE)) {
    return -1;
  }
#ifdef ET_CHECKPOINT_IO_TESTING
  if (et_test.short_io != 0u && length > et_test.short_io) {
    length = et_test.short_io;
  }
#endif
  return read(descriptor, buffer, length);
}

static ssize_t wrapped_write(int descriptor, const void *buffer,
                             size_t length) {
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_WRITE_TEMP)) {
    return -1;
  }
#ifdef ET_CHECKPOINT_IO_TESTING
  if (et_test.short_io != 0u && length > et_test.short_io) {
    length = et_test.short_io;
  }
#endif
  return write(descriptor, buffer, length);
}

static int wrapped_fsync(int descriptor, uint32_t stage) {
  int result;
  do {
    if (test_should_fail(stage)) {
      result = -1;
    } else {
      result = fsync(descriptor);
    }
  } while (result < 0 && errno == EINTR);
  return result;
}

static int wrapped_close(int descriptor, uint32_t stage) {
  if (test_should_fail(stage)) {
    const int saved = errno;
    (void)close(descriptor);
    errno = saved;
    return -1;
  }
  return close(descriptor);
}

static int wrapped_unlinkat(int directory, const char *name) {
  int result;
  do {
    if (test_should_fail(ET_CHECKPOINT_IO_STAGE_CLEANUP_TEMP)) {
      result = -1;
    } else {
      result = unlinkat(directory, name, 0);
    }
  } while (result < 0 && errno == EINTR);
  return result;
}

static ssize_t wrapped_getrandom(void *buffer, size_t length) {
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_RANDOM_TEMP)) {
    return -1;
  }
#ifdef ET_CHECKPOINT_IO_TESTING
  if (et_test.deterministic_random) {
    if (length > sizeof(et_test.random_bytes)) {
      length = sizeof(et_test.random_bytes);
    }
    memcpy(buffer, et_test.random_bytes, length);
    return (ssize_t)length;
  }
#endif
  return getrandom(buffer, length, 0u);
}

static int fill_random(uint8_t bytes[ET_TEMP_RANDOM_BYTES]) {
  size_t offset = 0u;
  while (offset < ET_TEMP_RANDOM_BYTES) {
    const ssize_t amount =
        wrapped_getrandom(bytes + offset, ET_TEMP_RANDOM_BYTES - offset);
    if (amount > 0) {
      offset += (size_t)amount;
    } else if (amount == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static void format_temp_name(char name[ET_TEMP_NAME_BYTES],
                             const uint8_t random[ET_TEMP_RANDOM_BYTES]) {
  static const char hexadecimal[] = "0123456789abcdef";
  size_t index;
  memcpy(name, ".et-c1-", 7u);
  for (index = 0u; index < ET_TEMP_RANDOM_BYTES; ++index) {
    name[7u + index * 2u] = hexadecimal[random[index] >> 4u];
    name[8u + index * 2u] = hexadecimal[random[index] & 15u];
  }
  memcpy(name + 39u, ".tmp", 5u);
#ifdef ET_CHECKPOINT_IO_TESTING
  memcpy(et_test.last_temp, name, ET_TEMP_NAME_BYTES);
#endif
}

static int create_temp(int directory, char name[ET_TEMP_NAME_BYTES],
                       uint32_t *failure_stage) {
  uint8_t random[ET_TEMP_RANDOM_BYTES];
  uint32_t attempt;
  for (attempt = 0u; attempt < ET_TEMP_ATTEMPTS; ++attempt) {
    int descriptor;
    if (fill_random(random) != 0) {
      *failure_stage = ET_CHECKPOINT_IO_STAGE_RANDOM_TEMP;
      return -1;
    }
    format_temp_name(name, random);
    descriptor = wrapped_openat(
        directory, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
        (mode_t)0600, ET_CHECKPOINT_IO_STAGE_CREATE_TEMP);
    if (descriptor >= 0) {
      return descriptor;
    }
    if (errno != EEXIST) {
      *failure_stage = ET_CHECKPOINT_IO_STAGE_CREATE_TEMP;
      return -1;
    }
  }
  errno = EEXIST;
  *failure_stage = ET_CHECKPOINT_IO_STAGE_CREATE_TEMP;
  return -1;
}

static int publish_temp(int directory, const char *temporary,
                        const char *destination, int overwrite) {
  int result;
  do {
    if (test_should_fail(ET_CHECKPOINT_IO_STAGE_PUBLISH)) {
      result = -1;
    } else if (overwrite) {
      result = renameat(directory, temporary, directory, destination);
    } else {
#ifdef SYS_renameat2
      result = (int)syscall(SYS_renameat2, directory, temporary, directory,
                            destination, (unsigned int)RENAME_NOREPLACE);
#else
      errno = ENOTSUP;
      result = -1;
#endif
    }
  } while (result < 0 && errno == EINTR);
  return result;
}

static int exact_read_loop(int descriptor, uint8_t *destination,
                           size_t length) {
  size_t offset = 0u;
  while (offset < length) {
    const ssize_t amount =
        wrapped_read(descriptor, destination + offset, length - offset);
    if (amount > 0) {
      offset += (size_t)amount;
    } else if (amount == 0) {
      errno = ENODATA;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int exact_write_loop(int descriptor, const uint8_t *source,
                            size_t length) {
  size_t offset = 0u;
  while (offset < length) {
    const ssize_t amount =
        wrapped_write(descriptor, source + offset, length - offset);
    if (amount > 0) {
      offset += (size_t)amount;
    } else if (amount == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

int64_t et_checkpoint_io_abi_major_v1(void) {
  return (int64_t)ET_CHECKPOINT_IO_ABI_MAJOR;
}

int64_t et_checkpoint_io_abi_minor_v1(void) {
  return (int64_t)ET_CHECKPOINT_IO_ABI_MINOR;
}

int64_t et_checkpoint_io_read_exact_v1(const char *path,
                                       void *bytevector_header,
                                       uint64_t expected_length,
                                       uint64_t maximum_length) {
  et_path_parts parts;
  const uint8_t *ignored;
  uint8_t *temporary = NULL;
  int directory = -1;
  int source = -1;
  struct stat before;
  struct stat after;
  uint8_t trailing;
  int saved_errno;

  if (maximum_length == 0u || expected_length > maximum_length ||
      expected_length > (uint64_t)SIZE_MAX) {
    errno = EINVAL;
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, errno, 0);
  }
  if (validate_bytevector(bytevector_header, expected_length, &ignored) != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, errno, 0);
  }
  if (expected_length != 0u) {
    temporary = (uint8_t *)wrapped_payload_malloc((size_t)expected_length);
    if (temporary == NULL) {
      return make_status(ET_CHECKPOINT_IO_STAGE_ALLOCATE, errno, 0);
    }
  }
  directory = open_parent(path, &parts);
  if (directory < 0) {
    saved_errno = errno;
    free(temporary);
    return make_status(ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, saved_errno, 0);
  }
  source = wrapped_openat(directory, parts.basename,
                          O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0,
                          ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE);
  if (source < 0) {
    saved_errno = errno;
    goto read_failure;
  }
  if (wrapped_fstat(source, &before) != 0) {
    saved_errno = errno;
    goto read_stat_failure;
  }
  if (!S_ISREG(before.st_mode) || before.st_size < 0 ||
      (uint64_t)before.st_size != expected_length) {
    saved_errno = !S_ISREG(before.st_mode) ? EINVAL : EMSGSIZE;
    goto read_stat_failure;
  }
  if (exact_read_loop(source, temporary, (size_t)expected_length) != 0) {
    saved_errno = errno;
    goto read_failure;
  }
  for (;;) {
    const ssize_t amount = wrapped_read(source, &trailing, 1u);
    if (amount == 0) {
      break;
    }
    if (amount > 0) {
      saved_errno = EMSGSIZE;
      goto read_data_failure;
    }
    if (errno != EINTR) {
      saved_errno = errno;
      goto read_failure;
    }
  }
  if (wrapped_fstat(source, &after) != 0) {
    saved_errno = errno;
    goto read_stat_failure;
  }
  if (after.st_dev != before.st_dev || after.st_ino != before.st_ino ||
      after.st_size != before.st_size) {
    saved_errno = ESTALE;
    goto read_stat_failure;
  }
  if (wrapped_close(source, ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE) != 0) {
    saved_errno = errno;
    goto read_close_failure;
  }
  if (wrapped_close(directory, ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY) != 0) {
    saved_errno = errno;
    goto read_directory_failure;
  }
  if (expected_length != 0u) {
    memcpy((uint8_t *)bytevector_header + sizeof(int64_t), temporary,
           (size_t)expected_length);
  }
  free_path_parts(&parts);
  free(temporary);
  return 0;

read_stat_failure:
  if (source >= 0) {
    (void)close(source);
  }
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(temporary);
  return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, saved_errno, 0);
read_data_failure:
  if (source >= 0) {
    (void)close(source);
  }
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(temporary);
  return make_status(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, saved_errno, 0);
read_close_failure:
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(temporary);
  return make_status(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, saved_errno, 0);
read_directory_failure:
  free_path_parts(&parts);
  free(temporary);
  return make_status(ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY, saved_errno, 0);
read_failure:
  if (source >= 0) {
    (void)close(source);
  }
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(temporary);
  return make_status(source < 0 ? ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE
                                : ET_CHECKPOINT_IO_STAGE_READ_SOURCE,
                     saved_errno, 0);
}

int64_t et_checkpoint_io_atomic_write_v1(const char *path,
                                         const void *bytevector_header,
                                         uint64_t expected_length,
                                         int64_t overwrite) {
  et_path_parts parts;
  const uint8_t *source;
  uint8_t *snapshot = NULL;
  int directory = -1;
  int temporary_fd = -1;
  int temporary_created = 0;
  char temporary_name[ET_TEMP_NAME_BYTES] = {0};
  uint32_t failure_stage = ET_CHECKPOINT_IO_STAGE_NONE;
  int saved_errno = 0;
  int cleanup_error = 0;

  if ((overwrite != 0 && overwrite != 1) ||
      expected_length > (uint64_t)SIZE_MAX) {
    errno = EINVAL;
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, errno, 0);
  }
  if (validate_bytevector(bytevector_header, expected_length, &source) != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, errno, 0);
  }
  if (expected_length != 0u) {
    snapshot = (uint8_t *)wrapped_payload_malloc((size_t)expected_length);
    if (snapshot == NULL) {
      return make_status(ET_CHECKPOINT_IO_STAGE_ALLOCATE, errno, 0);
    }
    memcpy(snapshot, source, (size_t)expected_length);
  }
  directory = open_parent(path, &parts);
  if (directory < 0) {
    saved_errno = errno;
    free(snapshot);
    return make_status(ET_CHECKPOINT_IO_STAGE_OPEN_PARENT, saved_errno, 0);
  }
  temporary_fd = create_temp(directory, temporary_name, &failure_stage);
  if (temporary_fd < 0) {
    saved_errno = errno;
    goto write_failure;
  }
  temporary_created = 1;
  if (exact_write_loop(temporary_fd, snapshot, (size_t)expected_length) != 0) {
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_WRITE_TEMP;
    goto write_failure;
  }
  if (wrapped_fsync(temporary_fd, ET_CHECKPOINT_IO_STAGE_SYNC_TEMP) != 0) {
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_SYNC_TEMP;
    goto write_failure;
  }
  if (wrapped_close(temporary_fd, ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP) != 0) {
    temporary_fd = -1;
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP;
    goto write_failure;
  }
  temporary_fd = -1;
  if (publish_temp(directory, temporary_name, parts.basename, overwrite != 0) !=
      0) {
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_PUBLISH;
    goto write_failure;
  }
  temporary_name[0] = '\0';
  if (wrapped_fsync(directory, ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY) != 0) {
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY;
    goto committed_failure;
  }
  if (wrapped_close(directory, ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY) != 0) {
    directory = -1;
    saved_errno = errno;
    failure_stage = ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY;
    goto committed_failure;
  }
  free_path_parts(&parts);
  free(snapshot);
  return 0;

write_failure:
  if (temporary_fd >= 0) {
    if (wrapped_close(temporary_fd, ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP) != 0 &&
        failure_stage == ET_CHECKPOINT_IO_STAGE_NONE) {
      saved_errno = errno;
      failure_stage = ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP;
    }
  }
  if (temporary_created && temporary_name[0] != '\0' &&
      wrapped_unlinkat(directory, temporary_name) != 0 && errno != ENOENT) {
    cleanup_error = errno;
  }
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(snapshot);
  if (cleanup_error != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_CLEANUP_TEMP, cleanup_error, 0);
  }
  return make_status(failure_stage == ET_CHECKPOINT_IO_STAGE_NONE
                         ? ET_CHECKPOINT_IO_STAGE_CREATE_TEMP
                         : failure_stage,
                     saved_errno, 0);

committed_failure:
  if (directory >= 0) {
    (void)close(directory);
  }
  free_path_parts(&parts);
  free(snapshot);
  return make_status(failure_stage, saved_errno, 1);
}
