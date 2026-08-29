#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum et_d1_io_stage {
  ET_D1_IO_ARGUMENT = 1,
  ET_D1_IO_OPEN = 2,
  ET_D1_IO_WRITE = 3,
  ET_D1_IO_CLOSE = 4,
  ET_D1_IO_CLEANUP = 5,
};

static int64_t et_d1_status(enum et_d1_io_stage stage, int error_number) {
  uint32_t stable_error = (uint32_t)(error_number == 0 ? EIO : error_number);
  return ((int64_t)stage << 32) | (int64_t)stable_error;
}

#ifdef ET_D1_TEST_FAULTS
static uint64_t et_d1_test_invocation;

static uint64_t et_d1_test_fail_call(void) {
  const char *text = getenv("ET_D1_TEST_FAIL_CALL");
  char *end = NULL;
  unsigned long long parsed;
  if (text == NULL || *text == '\0') {
    return 1u;
  }
  errno = 0;
  parsed = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0u) {
    return UINT64_MAX;
  }
  return (uint64_t)parsed;
}

static const char *et_d1_test_fault(uint64_t invocation) {
  const char *fault = getenv("ET_D1_TEST_FAULT");
  if (invocation != et_d1_test_fail_call()) {
    return NULL;
  }
  return fault;
}
#endif

/*
 * Fixed internal ABI for the pinned Eshkol compiler. A bytevector's tagged
 * payload points at an i64 native-length header followed immediately by bytes.
 * The explicit expected length prevents trusting that representation blindly.
 *
 * Return zero on success. Failures are a positive i64 containing a stable
 * stage in the high 32 bits and the captured errno in the low 32 bits.
 */
int64_t et_d1_checked_write_new_v1(const char *path,
                                   const void *bytevector_header,
                                   int64_t expected_length) {
  int fd;
  int saved_error;
  int64_t declared_length;
  const unsigned char *bytes;
  uint64_t remaining;
  int owned = 0;
#ifdef ET_D1_TEST_FAULTS
  const char *fault;
  int short_write_injected = 0;
  uint64_t invocation = ++et_d1_test_invocation;
  fault = et_d1_test_fault(invocation);
#endif

  if (path == NULL || *path == '\0' || bytevector_header == NULL ||
      expected_length < 0) {
    return et_d1_status(ET_D1_IO_ARGUMENT, EINVAL);
  }
  memcpy(&declared_length, bytevector_header, sizeof(declared_length));
  if (declared_length != expected_length) {
    return et_d1_status(ET_D1_IO_ARGUMENT, EINVAL);
  }
  bytes = (const unsigned char *)bytevector_header + sizeof(declared_length);
  remaining = (uint64_t)expected_length;

  fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (fd < 0) {
    return et_d1_status(ET_D1_IO_OPEN, errno);
  }
  owned = 1;

  while (remaining > 0u) {
    size_t request = remaining > (uint64_t)INT64_MAX
                         ? (size_t)INT64_MAX
                         : (size_t)remaining;
    ssize_t written;
#ifdef ET_D1_TEST_FAULTS
    if (fault != NULL && strcmp(fault, "short-write") == 0 &&
        !short_write_injected && request > 1u) {
      request = 1u;
      short_write_injected = 1;
    }
    if (fault != NULL && strcmp(fault, "write-enospc") == 0) {
      errno = ENOSPC;
      written = -1;
    } else if (fault != NULL && strcmp(fault, "write-eio") == 0) {
      errno = EIO;
      written = -1;
    } else {
      written = write(fd, bytes, request);
    }
#else
    written = write(fd, bytes, request);
#endif
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      saved_error = errno;
      (void)close(fd);
      fd = -1;
      if (unlink(path) != 0 && errno != ENOENT) {
        return et_d1_status(ET_D1_IO_CLEANUP, errno);
      }
      return et_d1_status(ET_D1_IO_WRITE, saved_error);
    }
    if (written == 0) {
      (void)close(fd);
      fd = -1;
      if (unlink(path) != 0 && errno != ENOENT) {
        return et_d1_status(ET_D1_IO_CLEANUP, errno);
      }
      return et_d1_status(ET_D1_IO_WRITE, EIO);
    }
    bytes += (size_t)written;
    remaining -= (uint64_t)written;
  }

#ifdef ET_D1_TEST_FAULTS
  if (fault != NULL && strcmp(fault, "close-eio") == 0) {
    (void)close(fd);
    errno = EIO;
    saved_error = errno;
  } else
#endif
  if (close(fd) != 0) {
    saved_error = errno;
  } else {
    owned = 0;
    return 0;
  }
  fd = -1;
  if (owned && unlink(path) != 0 && errno != ENOENT) {
    return et_d1_status(ET_D1_IO_CLEANUP, errno);
  }
  return et_d1_status(ET_D1_IO_CLOSE, saved_error);
}
