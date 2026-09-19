/* Test-only discovery allocation injection. The production K1 source and public
 * artifact are unchanged; failed discovery must free every partial snapshot. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
static size_t remaining=SIZE_MAX;
void et_m3t_test_k1_fail_after(size_t allowed) {remaining=allowed;}
void et_m3t_test_k1_allocator_reset(void) {remaining=SIZE_MAX;}
static int admit_allocation(void) {
 if(!remaining)return 0;
 if(remaining!=SIZE_MAX)--remaining;
 return 1;
}
static void *test_malloc(size_t bytes) {return admit_allocation()?malloc(bytes):NULL;}
static void *test_calloc(size_t count,size_t bytes) {return admit_allocation()?calloc(count,bytes):NULL;}
static void *test_realloc(void *old,size_t bytes) {return admit_allocation()?realloc(old,bytes):NULL;}
#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#include "../../native/kernel_abi.c"
#undef malloc
#undef calloc
#undef realloc
