#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

int64_t et_c2_operational_current_rss_kib_v1(void) {
  FILE *stream = fopen("/proc/self/statm", "r");
  unsigned long total_pages = 0ul, resident_pages = 0ul;
  long page_bytes;
  int scanned, closed;
  uint64_t bytes;
  (void)total_pages;
  if (stream == NULL) return -1;
  scanned = fscanf(stream, "%lu %lu", &total_pages, &resident_pages);
  closed = fclose(stream);
  if (scanned != 2 || closed != 0) return -1;
  page_bytes = sysconf(_SC_PAGESIZE);
  if (page_bytes <= 0 ||
      (uint64_t)resident_pages > UINT64_MAX / (uint64_t)page_bytes)
    return -1;
  bytes = (uint64_t)resident_pages * (uint64_t)page_bytes;
  if (bytes / UINT64_C(1024) > (uint64_t)INT64_MAX) return -1;
  return (int64_t)(bytes / UINT64_C(1024));
}

int64_t et_c2_operational_peak_rss_kib_v1(void) {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0)
    return -1;
  return (int64_t)usage.ru_maxrss;
}

int64_t et_c2_operational_monotonic_millis_v1(void) {
  struct timespec now;
  uint64_t milliseconds;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0 ||
      now.tv_nsec < 0)
    return -1;
  if ((uint64_t)now.tv_sec > UINT64_MAX / UINT64_C(1000)) return -1;
  milliseconds = (uint64_t)now.tv_sec * UINT64_C(1000) +
                 (uint64_t)now.tv_nsec / UINT64_C(1000000);
  return milliseconds > (uint64_t)INT64_MAX ? -1 : (int64_t)milliseconds;
}

/* Test-only evidence: 25 expected markers, at most 32 records/2048 bytes.
 * Create the private file ourselves, then append only to that single-link
 * inode. O_NONBLOCK prevents a planted FIFO from delaying the watchdog. */
void et_c2_operational_progress_v1(int64_t phase) {
  static dev_t device;
  static ino_t inode;
  static off_t expected_size;
  static unsigned records;
  static int created;
  static int unavailable;
  const char *path = getenv("C2_OPERATIONAL_PROGRESS_FILE");
  struct stat state;
  char record[96];
  int fd, length, closed;
  ssize_t written;
  if (path == NULL || path[0] == '\0' || phase < 1 || phase > 41 ||
      unavailable || records >= 32u) return;
  fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW |
                  (created ? 0 : O_CREAT | O_EXCL), 0600);
  if (fd < 0) {
    unavailable = 1;
    return;
  }
  if (fstat(fd, &state) != 0 || !S_ISREG(state.st_mode) ||
      state.st_nlink != 1 || state.st_size < 0 ||
      (created && (state.st_dev != device || state.st_ino != inode ||
                   state.st_size != expected_size)) ||
      (!created && state.st_size != 0)) {
    unavailable = 1;
    (void)close(fd);
    return;
  }
  if (!created) {
    device = state.st_dev;
    inode = state.st_ino;
    created = 1;
  }
  length = snprintf(record, sizeof(record),
                    "C2 OPERATIONAL PROGRESS: phase=%" PRId64
                    " monotonic-ms=%" PRId64 "\n",
                    phase, et_c2_operational_monotonic_millis_v1());
  if (length <= 0 || (size_t)length >= sizeof(record) ||
      state.st_size > 2048 - length) {
    unavailable = 1;
    (void)close(fd);
    return;
  }
  written = write(fd, record, (size_t)length);
  closed = close(fd);
  if (written != length || closed != 0) {
    unavailable = 1;
    return;
  }
  expected_size = state.st_size + length;
  ++records;
}
