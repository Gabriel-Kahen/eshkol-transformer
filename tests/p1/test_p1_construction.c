#define ET_P1_PRIVATE_API 1
#include "p1_identity_internal.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void *__real_calloc(size_t, size_t);
static int fail_after = -1;
void *__wrap_calloc(size_t count, size_t size) {
  if (fail_after == 0) { fail_after = -1; return NULL; }
  if (fail_after > 0) --fail_after;
  return __real_calloc(count, size);
}
static void *result(void *context) { return et_p1_private_result_ptr_v1(context); }
static int64_t live(void *context) {
  assert(et_p1_private_live_entry_count_v1(context) == 0);
  return et_p1_private_result_i64_v1(context);
}
int main(void) {
  void *context = et_p1_private_context_create_v1();
  void *construction, *module, *handle, *old, *other;
  int i;
  assert(context);
  assert(et_p1_private_module_create_v1(context) == 0);
  old = result(context);
  for (i = 0; i < 2; ++i) {
    fail_after = i;
    assert(et_p1_private_construction_begin_v1(context) == ET_P1_STATUS_INTERNAL);
    assert(result(context) == NULL && live(context) == 1);
  }
  assert(et_p1_private_construction_begin_v1(context) == 0);
  construction = result(context);
  assert(et_p1_private_construction_begin_v1(context) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_v1(context, (void *)1) == ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_private_construction_module_create_v1(context, old) == ET_P1_STATUS_INVALID_ARGUMENT);
  for (i = 0; i < 2; ++i) {
    fail_after = i;
    assert(et_p1_private_construction_handle_create_v1(context, construction) == ET_P1_STATUS_INTERNAL);
    assert(live(context) == 1);
  }
  assert(et_p1_private_construction_module_create_v1(context, construction) == 0);
  module = result(context);
  assert(et_p1_private_construction_handle_create_v1(context, construction) == 0);
  handle = result(context);
  assert(live(context) == 3);
  /* Last-record corruption must reject before any earlier enrollment dies. */
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  assert(et_p1_private_construction_abort_v1(context, construction) == ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_public_token_live_v1(module) == 1);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  fail_after = 0;
  assert(et_p1_private_construction_abort_v1(context, construction) == 0);
  assert(fail_after == 0);
  fail_after = -1;
  assert(et_p1_private_construction_abort_v1(context, construction) == 0);
  assert(et_p1_public_token_kind_v1(module) == -ET_P1_TOKEN_MODULE);
  assert(et_p1_public_token_kind_v1(handle) == -ET_P1_TOKEN_PARAMETER_HANDLE);
  assert(et_p1_public_token_live_v1(old) == 1 && live(context) == 1);
  assert(et_p1_private_construction_seal_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_begin_v1(context) == 0);
  other = result(context);
  assert(et_p1_private_construction_commit_prepared_v1(context, other) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_prepared_v1(context, other) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_module_create_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_v1(context, construction) == 0);
  assert(et_p1_private_construction_module_create_v1(context, other) == 0);
  module = result(context);
  fail_after = 0;
  assert(et_p1_private_construction_seal_v1(context, other) == 0);
  assert(fail_after == 0);
  fail_after = -1;
  assert(et_p1_private_construction_abort_v1(context, other) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_public_token_live_v1(module) == 1);
  assert(et_p1_private_construction_begin_v1(context) == 0);
  construction = result(context);
  assert(et_p1_private_construction_module_create_v1(context, construction) == 0);
  module = result(context);
  assert(et_p1_private_construction_handle_create_v1(context, construction) == 0);
  handle = result(context);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  assert(et_p1_private_construction_prepare_v1(context, construction) == ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_public_token_live_v1(module) == 1);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  fail_after = 0;
  assert(et_p1_private_construction_prepare_v1(context, construction) == 0);
  assert(fail_after == 0);
  assert(et_p1_private_construction_prepare_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_begin_v1(context) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_module_create_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_handle_create_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_seal_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_context_release_v1(context) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_commit_prepared_v1(context, old) == ET_P1_STATUS_INVALID_ARGUMENT);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  assert(et_p1_private_construction_commit_prepared_v1(context, construction) == ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_public_token_live_v1(module) == 1);
  assert(et_p1_private_construction_abort_prepared_v1(context, construction) == ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_public_token_live_v1(module) == 1);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  fail_after = 0;
  assert(et_p1_private_construction_abort_prepared_v1(context, construction) == 0);
  assert(fail_after == 0);
  fail_after = -1;
  assert(et_p1_public_token_kind_v1(module) == -ET_P1_TOKEN_MODULE);
  assert(et_p1_public_token_kind_v1(handle) == -ET_P1_TOKEN_PARAMETER_HANDLE);
  assert(et_p1_private_construction_abort_prepared_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_v1(context, construction) == 0);
  assert(et_p1_private_construction_begin_v1(context) == 0);
  construction = result(context);
  assert(et_p1_private_construction_module_create_v1(context, construction) == 0);
  module = result(context);
  assert(et_p1_private_construction_prepare_v1(context, construction) == 0);
  fail_after = 0;
  assert(et_p1_private_construction_commit_prepared_v1(context, construction) == 0);
  assert(fail_after == 0);
  fail_after = -1;
  assert(et_p1_public_token_live_v1(module) == 1);
  assert(et_p1_private_construction_commit_prepared_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_prepared_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_construction_abort_v1(context, construction) == ET_P1_STATUS_INVALID_STATE);
  for (i = 0; i < 100; ++i) {
    assert(et_p1_private_construction_begin_v1(context) == 0);
    construction = result(context);
    assert(et_p1_private_construction_module_create_v1(context, construction) == 0);
    assert(et_p1_private_construction_handle_create_v1(context, construction) == 0);
    assert(et_p1_private_construction_abort_v1(context, construction) == 0);
    assert(live(context) == 3);
  }
  puts("P1 construction PASS: open/prepared split, failure atomicity, exact enrollment, stale/foreign scope, live baseline");
  return 0;
}
