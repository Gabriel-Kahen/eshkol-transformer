#define ET_A2_KV_CACHE_TESTING 1
#include "../../native/a2_kv_cache.c"

int64_t et_g3c4_call_entry_test_live_cache_stat_v1(int64_t selector) {
  int64_t count = 0;
  int64_t bytes = 0;
  et_a2_kv_cache *cache;
  for (cache = live_caches; cache != NULL; cache = cache->registry_next) {
    count++;
    bytes += (int64_t)cache->total_bytes;
  }
  if (selector == 0) return count;
  if (selector == 1) return bytes;
  return -1;
}
