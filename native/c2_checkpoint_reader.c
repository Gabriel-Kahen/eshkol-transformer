#define _GNU_SOURCE

#include "c2_checkpoint_reader.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(__linux__) || !defined(O_CLOEXEC) || !defined(O_NOFOLLOW) ||      \
    !defined(O_NONBLOCK)
#error "C2 checkpoint reader ABI 1 requires the reviewed Linux open surface"
#endif

_Static_assert(CHAR_BIT == 8 && sizeof(int64_t) == 8u && sizeof(uint64_t) == 8u,
               "C2 checkpoint reader requires exact eight-byte integers");

struct et_c2_checkpoint_reader {
  struct et_c2_checkpoint_reader *registry_next;
  int descriptor;
  uint64_t physical_size;
};

static et_c2_checkpoint_reader *live_readers;

#ifdef ET_C2_CHECKPOINT_READER_TESTING
#define ET_C2_TEST_EVENT_CAPACITY 1024u
static struct {
  uint32_t fail_stage;
  uint32_t fail_occurrence;
  uint32_t fail_errno;
  uint32_t stage_occurrences[256];
  int short_read_enabled;
  size_t short_read;
  uint32_t events[ET_C2_TEST_EVENT_CAPACITY];
  size_t event_count;
  size_t fd_live;
  size_t fd_peak;
} et_c2_test;

void et_c2_checkpoint_reader_test_reset_v1(void) {
  const size_t fd_live = et_c2_test.fd_live;
  const size_t fd_peak = et_c2_test.fd_peak;
  memset(&et_c2_test, 0, sizeof(et_c2_test));
  et_c2_test.fd_live = fd_live;
  et_c2_test.fd_peak = fd_peak;
}

void et_c2_checkpoint_reader_test_fail_v1(uint32_t stage, uint32_t occurrence,
                                          int32_t error_number) {
  et_c2_test.fail_stage = stage;
  et_c2_test.fail_occurrence = occurrence;
  et_c2_test.fail_errno = error_number > 0 ? (uint32_t)error_number : EIO;
}

void et_c2_checkpoint_reader_test_set_short_read_v1(size_t maximum_bytes) {
  et_c2_test.short_read_enabled = 1;
  et_c2_test.short_read = maximum_bytes;
}

size_t et_c2_checkpoint_reader_test_event_count_v1(void) {
  return et_c2_test.event_count;
}

uint32_t et_c2_checkpoint_reader_test_event_at_v1(size_t index) {
  return index < et_c2_test.event_count ? et_c2_test.events[index] : 0u;
}

size_t et_c2_checkpoint_reader_test_live_count_v1(void) {
  size_t count = 0u;
  const et_c2_checkpoint_reader *reader;
  for (reader = live_readers; reader != NULL; reader = reader->registry_next) {
    ++count;
  }
  return count;
}

size_t et_c2_checkpoint_reader_test_fd_live_count_v1(void) {
  return et_c2_test.fd_live;
}

size_t et_c2_checkpoint_reader_test_fd_peak_count_v1(void) {
  return et_c2_test.fd_peak;
}

static int test_should_fail(uint32_t stage) {
  uint32_t occurrence;
  if (et_c2_test.event_count < ET_C2_TEST_EVENT_CAPACITY) {
    et_c2_test.events[et_c2_test.event_count++] = stage;
  }
  occurrence = ++et_c2_test.stage_occurrences[stage & 0xffu];
  if (et_c2_test.fail_stage == stage &&
      et_c2_test.fail_occurrence == occurrence) {
    et_c2_test.fail_stage = 0u;
    errno = (int)et_c2_test.fail_errno;
    return 1;
  }
  return 0;
}

static void test_fd_opened(void) {
  ++et_c2_test.fd_live;
  if (et_c2_test.fd_live > et_c2_test.fd_peak) {
    et_c2_test.fd_peak = et_c2_test.fd_live;
  }
}

static void test_fd_closed(void) {
  if (et_c2_test.fd_live > 0u) {
    --et_c2_test.fd_live;
  }
}
#else
static int test_should_fail(uint32_t stage) {
  (void)stage;
  return 0;
}

static void test_fd_opened(void) {}
static void test_fd_closed(void) {}
#endif

static int64_t make_status(uint32_t stage, int error_number) {
  uint64_t encoded_errno;
  if (error_number <= 0 ||
      error_number > (int)ET_CHECKPOINT_IO_STATUS_ERRNO_MASK) {
    error_number = EIO;
  }
  encoded_errno = (uint64_t)(uint32_t)error_number;
  return (
      int64_t)(encoded_errno |
               ((uint64_t)(stage & (uint32_t)ET_CHECKPOINT_IO_STATUS_STAGE_MASK)
                << ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT));
}

static et_c2_checkpoint_reader *
find_reader(const et_c2_checkpoint_reader *candidate) {
  et_c2_checkpoint_reader *reader;
  for (reader = live_readers; reader != NULL; reader = reader->registry_next) {
    if (reader == candidate) {
      return reader;
    }
  }
  return NULL;
}

static void unlink_reader(et_c2_checkpoint_reader *reader) {
  et_c2_checkpoint_reader **cursor = &live_readers;
  while (*cursor != NULL && *cursor != reader) {
    cursor = &(*cursor)->registry_next;
  }
  if (*cursor == reader) {
    *cursor = reader->registry_next;
  }
}

static int spans_overlap(const void *left, size_t left_size, const void *right,
                         size_t right_size) {
  const uintptr_t left_start = (uintptr_t)left;
  const uintptr_t right_start = (uintptr_t)right;
  if (left == NULL || right == NULL || left_start > UINTPTR_MAX - left_size ||
      right_start > UINTPTR_MAX - right_size) {
    return 1;
  }
  return left_start < right_start + right_size &&
         right_start < left_start + left_size;
}

static int validate_bytevector(void *header, uint64_t expected_length,
                               uint8_t **bytes) {
  int64_t observed_length;
  const uintptr_t start = (uintptr_t)header;
  if (header == NULL || expected_length > (uint64_t)INT64_MAX ||
      expected_length > (uint64_t)SIZE_MAX - sizeof(int64_t) ||
      start > UINTPTR_MAX - sizeof(int64_t) - (uintptr_t)expected_length) {
    errno = EOVERFLOW;
    return -1;
  }
  memcpy(&observed_length, header, sizeof(observed_length));
  if (observed_length < 0 || (uint64_t)observed_length != expected_length) {
    errno = EINVAL;
    return -1;
  }
  *bytes = (uint8_t *)header + sizeof(int64_t);
  return 0;
}

static ssize_t wrapped_pread(int descriptor, void *bytes, size_t length,
                             off_t offset) {
  size_t amount = length;
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_READ_SOURCE)) {
    return -1;
  }
#ifdef ET_C2_CHECKPOINT_READER_TESTING
  if (et_c2_test.short_read_enabled) {
    if (et_c2_test.short_read == 0u) {
      return 0;
    }
    if (amount > et_c2_test.short_read) {
      amount = et_c2_test.short_read;
    }
  }
#endif
  return pread(descriptor, bytes, amount, offset);
}

int64_t et_c2_checkpoint_reader_open_v1(const char *path,
                                        et_c2_checkpoint_reader **reader_slot,
                                        uint64_t *physical_size) {
  et_c2_checkpoint_reader *reader;
  struct stat metadata;
  int descriptor;
  int saved_errno;

  if (path == NULL || path[0] == '\0' || reader_slot == NULL ||
      physical_size == NULL || *reader_slot != NULL ||
      spans_overlap(reader_slot, sizeof(*reader_slot), physical_size,
                    sizeof(*physical_size))) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  }
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE)) {
    return make_status(ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, errno);
  }
  descriptor = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE, errno);
  }
  test_fd_opened();
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE)) {
    saved_errno = errno;
    (void)close(descriptor);
    test_fd_closed();
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, saved_errno);
  }
  if (fstat(descriptor, &metadata) != 0) {
    saved_errno = errno;
    (void)close(descriptor);
    test_fd_closed();
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, saved_errno);
  }
  if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
    (void)close(descriptor);
    test_fd_closed();
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EINVAL);
  }
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_ALLOCATE)) {
    saved_errno = errno;
    (void)close(descriptor);
    test_fd_closed();
    return make_status(ET_CHECKPOINT_IO_STAGE_ALLOCATE, saved_errno);
  }
  reader = (et_c2_checkpoint_reader *)malloc(sizeof(*reader));
  if (reader == NULL) {
    (void)close(descriptor);
    test_fd_closed();
    return make_status(ET_CHECKPOINT_IO_STAGE_ALLOCATE, ENOMEM);
  }
  reader->descriptor = descriptor;
  reader->physical_size = (uint64_t)metadata.st_size;
  reader->registry_next = live_readers;
  live_readers = reader;
  *physical_size = reader->physical_size;
  *reader_slot = reader;
  return 0;
}

int64_t
et_c2_checkpoint_reader_read_exact_v1(et_c2_checkpoint_reader *candidate,
                                      uint64_t offset, void *bytevector_header,
                                      uint64_t expected_length) {
  et_c2_checkpoint_reader *reader = find_reader(candidate);
  uint8_t *bytes;
  uint64_t completed = 0u;

  if (reader == NULL) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  }
  if (expected_length > UINT64_MAX - offset) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EOVERFLOW);
  }
  if (offset + expected_length > reader->physical_size) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EMSGSIZE);
  }
  if (validate_bytevector(bytevector_header, expected_length, &bytes) != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, errno);
  }
  while (completed < expected_length) {
    const uint64_t remaining = expected_length - completed;
    const size_t request =
        remaining > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)remaining;
    const off_t position = (off_t)(offset + completed);
    ssize_t amount =
        wrapped_pread(reader->descriptor, bytes + completed, request, position);
    if (amount < 0 && errno == EINTR) {
      continue;
    }
    if (amount < 0) {
      return make_status(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, errno);
    }
    if (amount == 0) {
      return make_status(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, ENODATA);
    }
    completed += (uint64_t)amount;
  }
  return 0;
}

int64_t
et_c2_checkpoint_reader_validate_final_v1(et_c2_checkpoint_reader *candidate) {
  et_c2_checkpoint_reader *reader = find_reader(candidate);
  struct stat metadata;
  uint8_t trailing_byte;
  ssize_t amount;

  if (reader == NULL) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  }
  if (test_should_fail(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE)) {
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, errno);
  }
  if (fstat(reader->descriptor, &metadata) != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, errno);
  }
  if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
      (uint64_t)metadata.st_size != reader->physical_size) {
    return make_status(ET_CHECKPOINT_IO_STAGE_STAT_SOURCE, EMSGSIZE);
  }
  do {
    amount = wrapped_pread(reader->descriptor, &trailing_byte, 1u,
                           (off_t)reader->physical_size);
  } while (amount < 0 && errno == EINTR);
  if (amount < 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, errno);
  }
  if (amount != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_READ_SOURCE, EMSGSIZE);
  }
  return 0;
}

int64_t
et_c2_checkpoint_reader_close_v1(et_c2_checkpoint_reader **reader_slot) {
  et_c2_checkpoint_reader *reader;
  int close_result;
  int saved_errno = 0;
  int injected_failure;

  if (reader_slot == NULL) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  }
  if (*reader_slot == NULL) {
    return 0;
  }
  reader = find_reader(*reader_slot);
  if (reader == NULL) {
    return make_status(ET_CHECKPOINT_IO_STAGE_VALIDATE, EINVAL);
  }
  unlink_reader(reader);
  *reader_slot = NULL;
  injected_failure = test_should_fail(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE);
  if (injected_failure) {
    saved_errno = errno;
  }
  close_result = close(reader->descriptor);
  if (close_result != 0) {
    saved_errno = errno;
  }
  test_fd_closed();
  free(reader);
  if (injected_failure) {
    return make_status(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, saved_errno);
  }
  if (close_result != 0) {
    return make_status(ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE, saved_errno);
  }
  return 0;
}
