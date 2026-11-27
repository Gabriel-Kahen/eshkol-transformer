#include "eshkol_transformer/paged_kv_abi.h"
#include "paged_kv_eshkol_bridge.h"

#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static size_t call_count;
static size_t fail_at;

void *__real_calloc(size_t count, size_t size);
void *__wrap_calloc(size_t count, size_t size) {
  ++call_count;
  if (call_count == fail_at)
    return NULL;
  return __real_calloc(count, size);
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                         \
      fprintf(stderr, "allocation failure check failed at line %d\n",       \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static et_paged_kv_status create_test_pool(
    const et_paged_kv_config *cfg, const et_paged_kv_prefix_context_v1 *context,
    float *storage, size_t floats, et_paged_kv_pool **pool) {
  if (context != NULL)
    return et_paged_kv_create_with_prefix_context_v1(cfg, context, storage,
                                                     floats, pool);
  return et_paged_kv_create(cfg, storage, floats, pool);
}

static int test_core_create(int external, int context_bound) {
  const et_paged_kv_config cfg = {4u, 1u, 1u, 2u, 2u, 2u, 2u, 0u};
  const et_paged_kv_prefix_context_v1 context = {
      .model = {1u}, .tokenizer = {2u}, .producer_config = {3u}};
  size_t floats = 0u;
  CHECK(et_paged_kv_pool_float_count(&cfg, &floats) == ET_PAGED_KV_OK);
  float *storage = external ? (float *)calloc(floats, sizeof(float)) : NULL;
  CHECK(!external || storage != NULL);
  if (external)
    storage[0] = 17.0f;

  const et_paged_kv_prefix_context_v1 *binding = context_bound ? &context : NULL;
  const size_t storage_floats = external ? floats : 0u;
  et_paged_kv_pool *pool = NULL;
  call_count = 0u;
  fail_at = 0u;
  CHECK(create_test_pool(&cfg, binding, storage, storage_floats, &pool) ==
        ET_PAGED_KV_OK);
  const size_t allocations = call_count;
  et_paged_kv_destroy(pool);
  CHECK(allocations >= 8u);
  if (external)
    storage[0] = 17.0f;

  for (size_t cut = 1u; cut <= allocations; ++cut) {
    pool = (et_paged_kv_pool *)(uintptr_t)1u;
    call_count = 0u;
    fail_at = cut;
    const et_paged_kv_status st =
        create_test_pool(&cfg, binding, storage, storage_floats, &pool);
    fail_at = 0u;
    CHECK(st == ET_PAGED_KV_ERR_NOMEM);
    CHECK(pool == (et_paged_kv_pool *)(uintptr_t)1u);
    CHECK(call_count >= cut && call_count <= allocations);
    if (external)
      CHECK(storage[0] == 17.0f);
  }
  free(storage);
  return 0;
}

static int test_bridge_create(void) {
  et_paged_kv_aot_ctx *ctx = NULL;
  call_count = 0u;
  fail_at = 0u;
  CHECK(et_paged_kv_aot_create(4, 1, 1, 2, 2, 2, 2, &ctx) == ET_PAGED_KV_OK);
  const size_t allocations = call_count;
  et_paged_kv_aot_destroy(ctx);
  CHECK(allocations >= 11u);

  for (size_t cut = 1u; cut <= allocations; ++cut) {
    ctx = (et_paged_kv_aot_ctx *)(uintptr_t)1u;
    call_count = 0u;
    fail_at = cut;
    const int32_t st = et_paged_kv_aot_create(4, 1, 1, 2, 2, 2, 2, &ctx);
    fail_at = 0u;
    CHECK(st == ET_PAGED_KV_ERR_NOMEM);
    CHECK(ctx == (et_paged_kv_aot_ctx *)(uintptr_t)1u);
    CHECK(call_count >= cut && call_count <= allocations);
  }
  return 0;
}

static int test_bridge_reset_failure(void) {
  et_paged_kv_aot_ctx *ctx = NULL;
  CHECK(et_paged_kv_aot_create(4, 1, 1, 2, 2, 2, 2, &ctx) == ET_PAGED_KV_OK);
  const int64_t seq = et_paged_kv_aot_seq_alloc(ctx);
  CHECK(seq >= 0);
  CHECK(et_paged_kv_aot_set_k(ctx, 0, 0, 0, 3.5) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_aot_set_v(ctx, 0, 0, 0, -2.25) == ET_PAGED_KV_OK);
  CHECK(fesetround(FE_UPWARD) == 0);
  const int32_t st = et_paged_kv_aot_reset(ctx);
  CHECK(fesetround(FE_TONEAREST) == 0);
  CHECK(st == ET_PAGED_KV_ERR_FP_ENV);
  CHECK(et_paged_kv_aot_seq_len(ctx, seq) == 0);
  CHECK(et_paged_kv_aot_append_token(ctx, seq) == ET_PAGED_KV_OK);
  CHECK(et_paged_kv_aot_read_kv(ctx, seq, 0, 0, 0, 0, 0) == 3.5);
  CHECK(et_paged_kv_aot_read_kv(ctx, seq, 0, 0, 0, 1, 0) == -2.25);
  et_paged_kv_aot_destroy(ctx);
  return 0;
}

int main(void) {
  CHECK(test_core_create(0, 0) == 0);
  CHECK(test_core_create(1, 0) == 0);
  CHECK(test_core_create(0, 1) == 0);
  CHECK(test_core_create(1, 1) == 0);
  CHECK(test_bridge_create() == 0);
  CHECK(test_bridge_reset_failure() == 0);
  puts("PAGED_KV allocation cuts and reset rollback: PASS");
  return 0;
}
