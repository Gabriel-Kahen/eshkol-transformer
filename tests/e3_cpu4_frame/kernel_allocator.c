/* Test-only census of allocations owned by the actual K1 implementation. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
static size_t allowed=SIZE_MAX, k1_live_allocations;
void et_m3t_test_k1_fail_after(size_t count) { allowed=count; }
void et_m3t_test_k1_allocator_reset(void) { allowed=SIZE_MAX; }
size_t et_e3_cpu4_test_k1_live_allocations(void) { return k1_live_allocations; }
static int admit(void) {
  if (!allowed) return 0;
  if (allowed!=SIZE_MAX) --allowed;
  return 1;
}
static void *counted_malloc(size_t bytes) {
  void *result=admit()?malloc(bytes):NULL;
  if (result) ++k1_live_allocations;
  return result;
}
static void *counted_calloc(size_t count,size_t bytes) {
  void *result=admit()?calloc(count,bytes):NULL;
  if (result) ++k1_live_allocations;
  return result;
}
static void *counted_realloc(void *old,size_t bytes) {
  if (!admit()) return NULL;
  const int had_old=old!=NULL;
  void *result=realloc(old,bytes);
  if (result && !had_old) ++k1_live_allocations;
  else if (!result && bytes==0 && had_old) --k1_live_allocations;
  return result;
}
static void counted_free(void *pointer) {
  if (pointer) { if (!k1_live_allocations) abort(); --k1_live_allocations; }
  free(pointer);
}
#define malloc counted_malloc
#define calloc counted_calloc
#define realloc counted_realloc
#define free counted_free
#include "../../native/kernel_abi.c"
#undef malloc
#undef calloc
#undef realloc
#undef free
