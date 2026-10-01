// Test-only global-arena census; linked beside the noninstalled E3 aggregate.
#include "arena_memory.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

extern "C" arena_t *get_global_arena_shared(void);

extern "C" int64_t et_e3_retention_arena_used_v1() {
  return static_cast<int64_t>(arena_get_used_memory(get_global_arena_shared()));
}

extern "C" int64_t et_e3_retention_arena_total_v1() {
  return static_cast<int64_t>(arena_get_total_memory(get_global_arena_shared()));
}

extern "C" int64_t et_e3_retention_arena_blocks_v1() {
  return static_cast<int64_t>(arena_get_block_count(get_global_arena_shared()));
}

extern "C" int64_t et_e3_retention_region_used_v1() {
  eshkol_region_t *region = region_current();
  if (!region || !region->arena) std::abort();
  return static_cast<int64_t>(arena_get_used_memory(region->arena));
}

extern "C" int64_t et_e3_retention_rss_kib_v1() {
  FILE *stream = std::fopen("/proc/self/statm", "r");
  if (!stream) std::abort();
  unsigned long long total = 0, resident = 0;
  const int scanned = std::fscanf(stream, "%llu %llu", &total, &resident);
  if (std::fclose(stream) || scanned != 2) std::abort();
  const long page_bytes = sysconf(_SC_PAGESIZE);
  if (page_bytes <= 0) std::abort();
  return static_cast<int64_t>(resident * static_cast<unsigned long long>(page_bytes) /
                              1024);
}
