#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
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
