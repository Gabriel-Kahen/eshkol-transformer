#include "eshkol_transformer/a2_kv_cache.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CACHE_MAGIC UINT64_C(0x41324b5643414348)
#define TXN_MAGIC UINT64_C(0x41324b565452414e)
#define VIEW_MAGIC UINT64_C(0x41324b5656494557)
#define BORROW_MAGIC UINT64_C(0x41324b56424f5252)

struct et_a2_kv_cache {
  uint64_t magic;
  et_a2_kv_cache *registry_next;
  size_t layers, batch, kv_heads, capacity, head_dimension;
  size_t layer_elements, total_elements, total_bytes;
  float *keys;
  float *values;
  int64_t *lengths;
  uint8_t *key_keep_mask;
  et_a2_kv_cache_transaction *active_transaction;
  et_a2_kv_cache_read_borrow *active_borrow;
};

struct et_a2_kv_cache_transaction {
  uint64_t magic;
  et_a2_kv_cache_transaction *registry_next;
  et_a2_kv_cache *cache;
  size_t append_width;
  int64_t *counts;
  int64_t *effective_lengths;
  uint8_t *effective_key_keep_mask;
  uint8_t *staged;
  et_a2_kv_cache_transaction_view *active_view;
};

struct et_a2_kv_cache_transaction_view {
  uint64_t magic;
  et_a2_kv_cache_transaction_view *registry_next;
  et_a2_kv_cache_transaction *transaction;
  size_t layer;
  uint64_t tensor_shape[4];
  uint64_t lengths_shape[1];
  uint64_t mask_shape[2];
  et_kernel_tensor_view_v1 keys;
  et_kernel_tensor_view_v1 values;
  et_kernel_tensor_view_v1 lengths;
  et_kernel_tensor_view_v1 key_keep_mask;
};

struct et_a2_kv_cache_read_borrow {
  uint64_t magic;
  et_a2_kv_cache_read_borrow *registry_next;
  et_a2_kv_cache *cache;
  uint64_t tensor_shape[4];
  uint64_t lengths_shape[1];
  uint64_t mask_shape[2];
  et_kernel_tensor_view_v1 *keys;
  et_kernel_tensor_view_v1 *values;
  et_kernel_tensor_view_v1 lengths;
  et_kernel_tensor_view_v1 key_keep_mask;
};

static et_a2_kv_cache *live_caches;
static et_a2_kv_cache_transaction *live_transactions;
static et_a2_kv_cache_transaction_view *live_views;
static et_a2_kv_cache_read_borrow *live_borrows;

#ifdef ET_A2_KV_CACHE_TESTING
static size_t allocation_limit = SIZE_MAX;
static size_t successful_allocations;

void et_a2_kv_cache_test_fail_alloc_after_v1(size_t allowed) {
  allocation_limit = allowed;
  successful_allocations = 0u;
}
void et_a2_kv_cache_test_reset_allocator_v1(void) {
  allocation_limit = SIZE_MAX;
  successful_allocations = 0u;
}
#endif

static void *a2_calloc(size_t count, size_t size) {
  void *result;
#ifdef ET_A2_KV_CACHE_TESTING
  if (successful_allocations >= allocation_limit) return NULL;
#endif
  result = calloc(count, size);
#ifdef ET_A2_KV_CACHE_TESTING
  if (result != NULL) successful_allocations++;
#endif
  return result;
}

static int span_fits(const void *pointer, size_t bytes) {
  return bytes == 0u ||
         (pointer != NULL && (uintptr_t)pointer <= UINTPTR_MAX - bytes);
}

static int overlaps(const void *a, size_t an, const void *b, size_t bn) {
  uintptr_t as = (uintptr_t)a, bs = (uintptr_t)b;
  if (an == 0u || bn == 0u) return 0;
  if (!span_fits(a, an) || !span_fits(b, bn)) return 1;
  return as < bs + bn && bs < as + an;
}

static int mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a) return 0;
  *out = a * b;
  return 1;
}

static et_a2_kv_cache *find_cache(const et_a2_kv_cache *target) {
  et_a2_kv_cache *it;
  for (it = live_caches; it != NULL; it = it->registry_next)
    if (it == target) return it;
  return NULL;
}
static et_a2_kv_cache_transaction *find_transaction(
    const et_a2_kv_cache_transaction *target) {
  et_a2_kv_cache_transaction *it;
  for (it = live_transactions; it != NULL; it = it->registry_next)
    if (it == target) return it;
  return NULL;
}
static et_a2_kv_cache_transaction_view *find_view(
    const et_a2_kv_cache_transaction_view *target) {
  et_a2_kv_cache_transaction_view *it;
  for (it = live_views; it != NULL; it = it->registry_next)
    if (it == target) return it;
  return NULL;
}
static et_a2_kv_cache_read_borrow *find_borrow(
    const et_a2_kv_cache_read_borrow *target) {
  et_a2_kv_cache_read_borrow *it;
  for (it = live_borrows; it != NULL; it = it->registry_next)
    if (it == target) return it;
  return NULL;
}

static int live_storage_overlap(const void *p, size_t n) {
  et_a2_kv_cache *c;
  et_a2_kv_cache_transaction *t;
  et_a2_kv_cache_transaction_view *v;
  et_a2_kv_cache_read_borrow *b;
  for (c = live_caches; c != NULL; c = c->registry_next) {
    if (overlaps(p, n, c, sizeof(*c)) ||
        overlaps(p, n, c->keys, c->total_bytes) ||
        overlaps(p, n, c->values, c->total_bytes) ||
        overlaps(p, n, c->lengths, c->batch * sizeof(*c->lengths)) ||
        overlaps(p, n, c->key_keep_mask, c->batch * c->capacity)) return 1;
  }
  for (t = live_transactions; t != NULL; t = t->registry_next) {
    if (overlaps(p, n, t, sizeof(*t)) ||
        overlaps(p, n, t->counts, t->cache->batch * sizeof(*t->counts)) ||
        overlaps(p, n, t->effective_lengths,
                 t->cache->batch * sizeof(*t->effective_lengths)) ||
        overlaps(p, n, t->effective_key_keep_mask,
                 t->cache->batch * t->cache->capacity) ||
        overlaps(p, n, t->staged, t->cache->layers)) return 1;
  }
  for (v = live_views; v != NULL; v = v->registry_next)
    if (overlaps(p, n, v, sizeof(*v))) return 1;
  for (b = live_borrows; b != NULL; b = b->registry_next)
    if (overlaps(p, n, b, sizeof(*b)) ||
        overlaps(p, n, b->keys, b->cache->layers * sizeof(*b->keys)) ||
        overlaps(p, n, b->values, b->cache->layers * sizeof(*b->values))) return 1;
  return 0;
}

static int32_t raw_reject(void) {
  return (int32_t)ET_KERNEL_ERROR_INVALID_ARGUMENT;
}

static int error_writable(const et_kernel_error *error) {
  return error == NULL || (!live_storage_overlap(error, sizeof(*error)) &&
                           span_fits(error, sizeof(*error)));
}

static int32_t fail(et_kernel_error *error, et_kernel_error_category category,
                    et_kernel_error_code code, const char *operation,
                    const char *message) {
  if (!error_writable(error)) return raw_reject();
  if (error != NULL) {
    et_kernel_error_clear(error);
    error->category = category;
    error->code = code;
    (void)snprintf(error->operation, sizeof(error->operation), "%s", operation);
    (void)snprintf(error->message, sizeof(error->message), "%s", message);
  }
  return (int32_t)category;
}

static int32_t success_preflighted(et_kernel_error *error) {
  et_kernel_error_clear(error);
  return 0;
}

static int32_t preflight_error_sink(const et_kernel_error *error) {
  return error_writable(error) ? 0 : raw_reject();
}

static int output_slot_ok(const void *slot, et_kernel_error *error,
                          const char *operation) {
  if (slot == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation, "output slot is null");
  if (error != NULL && overlaps(slot, sizeof(void *), error, sizeof(*error)))
    return raw_reject();
  if (!span_fits(slot, sizeof(void *)) ||
      live_storage_overlap(slot, sizeof(void *)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, operation,
                "output slot aliases live or error storage");
  return 0;
}

static int view_descriptor_error_overlap(const et_kernel_tensor_view_v1 *view,
                                         const et_kernel_error *error) {
  return error != NULL &&
         overlaps(view, sizeof(*view), error, sizeof(*error));
}

/* Returns 1 for an exact bounded symbol, 0 for invalid text, and -1 when the
 * logical text span reaches error and therefore no diagnostic may be written. */
static int exact_symbol(const char *symbol, const char *expected,
                        const et_kernel_error *error) {
  size_t index;
  size_t expected_length = strlen(expected);
  int matches = 1;
  if (symbol == NULL ||
      !span_fits(symbol, ET_KERNEL_MAX_SYMBOL_BYTES + 1u)) return 0;
  for (index = 0u; index <= ET_KERNEL_MAX_SYMBOL_BYTES; index++) {
    const char *byte = symbol + index;
    char value;
    if (error != NULL && overlaps(byte, 1u, error, sizeof(*error))) return -1;
    if (live_storage_overlap(byte, 1u)) return 0;
    value = *byte;
    if (value == '\0') return matches && index == expected_length;
    if (index >= expected_length || value != expected[index]) matches = 0;
  }
  return 0;
}

static int32_t validate_view(const et_kernel_tensor_view_v1 *view,
                             const char *dtype, size_t rank,
                             const size_t *shape, size_t element_size,
                             int reject_live_data, et_kernel_error *error,
                             const char *operation) {
  size_t i, elements = 1u, bytes;
  int symbol_result;
  if (view == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_NULL_ARGUMENT, operation, "tensor view is null");
  if (!span_fits(view, sizeof(*view)) ||
      view_descriptor_error_overlap(view, error)) return raw_reject();
  if (live_storage_overlap(view, sizeof(*view)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "tensor descriptor aliases live A2 cache storage");
  if (view->struct_size < ET_KERNEL_TENSOR_VIEW_V1_0_SIZE)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_INVALID_STRUCT_SIZE, operation,
                "tensor view struct size is not v1.0");
  symbol_result = exact_symbol(view->dtype, dtype, error);
  if (symbol_result < 0) return raw_reject();
  if (symbol_result == 0)
    return fail(error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation, "tensor dtype mismatch");
  symbol_result = exact_symbol(view->device, "cpu", error);
  if (symbol_result < 0) return raw_reject();
  if (symbol_result == 0)
    return fail(error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                ET_KERNEL_CODE_INVALID_BUFFER, operation, "tensor device mismatch");
  if (view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0u)
    return fail(error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "tensor must be dense row-major with zero offset");
  if (view->rank != rank || view->shape == NULL)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, operation, "tensor rank mismatch");
  if (!span_fits(view->shape, rank * sizeof(*view->shape)) ||
      (error != NULL && overlaps(view->shape, rank * sizeof(*view->shape),
                                 error, sizeof(*error)))) return raw_reject();
  if (live_storage_overlap(view->shape, rank * sizeof(*view->shape)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "tensor shape aliases live A2 cache storage");
  for (i = 0u; i < rank; i++) {
    if (view->shape[i] != shape[i])
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, operation, "tensor shape mismatch");
    if (!mul_size(elements, shape[i], &elements))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                  "tensor element count overflows size_t");
  }
  if (!mul_size(elements, element_size, &bytes))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INTEGER_OVERFLOW, operation,
                "tensor byte length overflows size_t");
  if (view->byte_length != bytes || view->data == NULL ||
      !span_fits(view->data, bytes) ||
      (uintptr_t)view->data % element_size != 0u)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "tensor data span or alignment is invalid");
  if (error != NULL && overlaps(view->data, bytes, error, sizeof(*error)))
    return raw_reject();
  if (overlaps(view->data, bytes, view, sizeof(*view)) ||
      overlaps(view->data, bytes, view->shape, rank * sizeof(*view->shape)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "tensor data aliases its descriptor metadata");
  if (reject_live_data && live_storage_overlap(view->data, bytes))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, operation,
                "borrowed tensor aliases live A2 cache storage");
  return 0;
}

static void unlink_cache(et_a2_kv_cache *target) {
  et_a2_kv_cache **it = &live_caches;
  while (*it != NULL) { if (*it == target) { *it = target->registry_next; return; } it = &(*it)->registry_next; }
}
static void unlink_transaction(et_a2_kv_cache_transaction *target) {
  et_a2_kv_cache_transaction **it = &live_transactions;
  while (*it != NULL) { if (*it == target) { *it = target->registry_next; return; } it = &(*it)->registry_next; }
}
static void unlink_view(et_a2_kv_cache_transaction_view *target) {
  et_a2_kv_cache_transaction_view **it = &live_views;
  while (*it != NULL) { if (*it == target) { *it = target->registry_next; return; } it = &(*it)->registry_next; }
}
static void unlink_borrow(et_a2_kv_cache_read_borrow *target) {
  et_a2_kv_cache_read_borrow **it = &live_borrows;
  while (*it != NULL) { if (*it == target) { *it = target->registry_next; return; } it = &(*it)->registry_next; }
}

int32_t et_a2_kv_cache_abi_major_v1(void) { return ET_A2_KV_CACHE_ABI_MAJOR; }
int32_t et_a2_kv_cache_abi_minor_v1(void) { return ET_A2_KV_CACHE_ABI_MINOR; }
int32_t et_a2_kv_cache_abi_require_v1(uint32_t major, uint32_t minor,
                                      et_kernel_error *error) {
  if (preflight_error_sink(error) != 0) return raw_reject();
  if (major != ET_A2_KV_CACHE_ABI_MAJOR || minor > ET_A2_KV_CACHE_ABI_MINOR)
    return fail(error, ET_KERNEL_ERROR_VERSION_MISMATCH,
                ET_KERNEL_CODE_ABI_MAJOR_MISMATCH, "a2-kv-cache.abi-require",
                "unsupported A2 KV-cache ABI version");
  return success_preflighted(error);
}

int32_t et_a2_kv_cache_create_v1(uint64_t layers64, uint64_t batch64,
                                 uint64_t heads64, uint64_t capacity64,
                                 uint64_t dimension64,
                                 et_a2_kv_cache **output,
                                 et_kernel_error *error) {
  const char *op = "a2-kv-cache.create";
  size_t elements, bytes;
  et_a2_kv_cache *cache;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r = output_slot_ok(output, error, op);
  if (r != 0) return r;
  if (*output != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "output slot must contain NULL");
  if (layers64 == 0u || batch64 == 0u || heads64 == 0u || capacity64 == 0u ||
      dimension64 == 0u || layers64 > SIZE_MAX || batch64 > SIZE_MAX ||
      heads64 > SIZE_MAX || capacity64 > SIZE_MAX || dimension64 > SIZE_MAX)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, op,
                "all cache dimensions must be positive and representable");
  elements = (size_t)layers64;
  if (!mul_size(elements, (size_t)batch64, &elements) ||
      !mul_size(elements, (size_t)heads64, &elements) ||
      !mul_size(elements, (size_t)capacity64, &elements) ||
      !mul_size(elements, (size_t)dimension64, &elements) ||
      !mul_size(elements, sizeof(float), &bytes) ||
      (size_t)batch64 > SIZE_MAX / sizeof(int64_t))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INTEGER_OVERFLOW, op, "cache size overflows size_t");
  cache = a2_calloc(1u, sizeof(*cache));
  if (cache == NULL) goto allocation_failed;
  cache->keys = a2_calloc(elements, sizeof(float));
  if (cache->keys == NULL) goto allocation_failed;
  cache->values = a2_calloc(elements, sizeof(float));
  if (cache->values == NULL) goto allocation_failed;
  cache->lengths = a2_calloc((size_t)batch64, sizeof(int64_t));
  if (cache->lengths == NULL) goto allocation_failed;
  cache->key_keep_mask = a2_calloc((size_t)batch64, (size_t)capacity64);
  if (cache->key_keep_mask == NULL) goto allocation_failed;
  cache->magic = CACHE_MAGIC;
  cache->layers = (size_t)layers64; cache->batch = (size_t)batch64;
  cache->kv_heads = (size_t)heads64; cache->capacity = (size_t)capacity64;
  cache->head_dimension = (size_t)dimension64;
  cache->total_elements = elements; cache->total_bytes = bytes;
  cache->layer_elements = elements / cache->layers;
  cache->registry_next = live_caches; live_caches = cache;
  *output = cache;
  return success_preflighted(error);
allocation_failed:
  if (cache != NULL) { free(cache->key_keep_mask); free(cache->lengths); free(cache->values); free(cache->keys); free(cache); }
  return fail(error, ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED,
              op, "cache allocation failed");
}

int32_t et_a2_kv_cache_destroy_v1(et_a2_kv_cache **slot,
                                  et_kernel_error *error) {
  const char *op = "a2-kv-cache.destroy";
  et_a2_kv_cache *cache;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r = output_slot_ok(slot, error, op);
  if (r != 0) return r;
  cache = find_cache(*slot);
  if (cache == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "cache handle is invalid");
  if (cache->active_transaction != NULL || cache->active_borrow != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op, "cache has a live lease");
  unlink_cache(cache); cache->magic = 0u;
  free(cache->key_keep_mask); free(cache->lengths); free(cache->values); free(cache->keys); free(cache);
  *slot = NULL;
  return success_preflighted(error);
}

int32_t et_a2_kv_cache_transaction_begin_v1(
    et_a2_kv_cache *handle, uint64_t width64,
    const et_kernel_tensor_view_v1 *counts_view,
    et_a2_kv_cache_transaction **output, et_kernel_error *error) {
  const char *op = "a2-kv-cache.transaction-begin";
  et_a2_kv_cache *cache;
  et_a2_kv_cache_transaction *txn = NULL;
  size_t shape[1], i;
  const int64_t *counts;
  int any = 0, r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  cache = find_cache(handle);
  r = output_slot_ok(output, error, op);
  if (r != 0) return r;
  if (*output != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "output slot must contain NULL");
  if (cache == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "cache handle is invalid");
  if (cache->active_transaction != NULL || cache->active_borrow != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op, "cache already has a live lease");
  if (width64 == 0u || width64 > SIZE_MAX)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, op, "append width must be positive");
  shape[0] = cache->batch;
  r = validate_view(counts_view, "i64", 1u, shape, sizeof(int64_t), 1,
                    error, op);
  if (r != 0) return r;
  if (overlaps(output, sizeof(*output), counts_view, sizeof(*counts_view)) ||
      overlaps(output, sizeof(*output), counts_view->shape,
               sizeof(*counts_view->shape)) ||
      overlaps(output, sizeof(*output), counts_view->data,
               counts_view->byte_length))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, op,
                "transaction output aliases append-count input");
  counts = (const int64_t *)counts_view->data;
  for (i = 0u; i < cache->batch; i++) {
    uint64_t count;
    if (counts[i] < 0)
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, op, "append count is negative");
    count = (uint64_t)counts[i];
    if (count > width64)
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_SHAPE, op, "append count exceeds width");
    if ((uint64_t)cache->lengths[i] > UINT64_MAX - count ||
        (uint64_t)cache->lengths[i] + count > cache->capacity)
      return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INTEGER_OVERFLOW, op,
                  "append exceeds cache capacity");
    any |= count != 0u;
  }
  if (!any)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, op,
                "at least one append count must be positive");
  txn = a2_calloc(1u, sizeof(*txn));
  if (txn == NULL) goto allocation_failed;
  txn->counts = a2_calloc(cache->batch, sizeof(*txn->counts));
  if (txn->counts == NULL) goto allocation_failed;
  txn->effective_lengths = a2_calloc(cache->batch, sizeof(*txn->effective_lengths));
  if (txn->effective_lengths == NULL) goto allocation_failed;
  txn->effective_key_keep_mask = a2_calloc(cache->batch, cache->capacity);
  if (txn->effective_key_keep_mask == NULL) goto allocation_failed;
  txn->staged = a2_calloc(cache->layers, sizeof(*txn->staged));
  if (txn->staged == NULL) goto allocation_failed;
  txn->magic = TXN_MAGIC; txn->cache = cache; txn->append_width = (size_t)width64;
  for (i = 0u; i < cache->batch; i++) {
    size_t position;
    txn->counts[i] = counts[i];
    txn->effective_lengths[i] = cache->lengths[i] + counts[i];
    for (position = 0u; position < (size_t)txn->effective_lengths[i]; position++)
      txn->effective_key_keep_mask[i * cache->capacity + position] = 1u;
  }
  txn->registry_next = live_transactions; live_transactions = txn;
  cache->active_transaction = txn; *output = txn;
  return success_preflighted(error);
allocation_failed:
  if (txn != NULL) { free(txn->staged); free(txn->effective_key_keep_mask); free(txn->effective_lengths); free(txn->counts); free(txn); }
  return fail(error, ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED,
              op, "transaction allocation failed");
}

int32_t et_a2_kv_cache_transaction_stage_layer_v1(
    et_a2_kv_cache_transaction *handle, uint64_t layer64,
    const et_kernel_tensor_view_v1 *keys,
    const et_kernel_tensor_view_v1 *values, et_kernel_error *error) {
  const char *op = "a2-kv-cache.stage-layer";
  et_a2_kv_cache_transaction *txn;
  et_a2_kv_cache *c;
  size_t shape[4], source_bytes, layer, n, h, a;
  const float *src_k, *src_v;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  txn = find_transaction(handle);
  if (txn == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "transaction handle is invalid");
  c = txn->cache;
  if (txn->active_view != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op, "transaction view is live");
  if (layer64 >= c->layers)
    return fail(error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                ET_KERNEL_CODE_INVALID_SHAPE, op, "layer is out of range");
  layer = (size_t)layer64;
  if (txn->staged[layer])
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_DUPLICATE_ENTRY, op, "layer was already staged");
  shape[0] = c->batch; shape[1] = c->kv_heads;
  shape[2] = txn->append_width; shape[3] = c->head_dimension;
  r = validate_view(keys, "f32", 4u, shape, sizeof(float), 1, error, op);
  if (r != 0) return r;
  r = validate_view(values, "f32", 4u, shape, sizeof(float), 1, error, op);
  if (r != 0) return r;
  source_bytes = keys->byte_length;
  if (overlaps(keys->data, source_bytes, values->data, source_bytes) ||
      overlaps(keys, sizeof(*keys), values, sizeof(*values)) ||
      overlaps(keys->data, source_bytes, values, sizeof(*values)) ||
      overlaps(values->data, source_bytes, keys, sizeof(*keys)) ||
      overlaps(keys->data, source_bytes, values->shape,
               4u*sizeof(*values->shape)) ||
      overlaps(values->data, source_bytes, keys->shape,
               4u*sizeof(*keys->shape)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, op, "key and value sources alias");
  src_k = (const float *)keys->data; src_v = (const float *)values->data;
  for (n = 0u; n < source_bytes / sizeof(float); n++)
    if (!isfinite(src_k[n]) || !isfinite(src_v[n]))
      return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER, op,
                  "key and value sources must be finite f32");
  for (n = 0u; n < c->batch; n++)
    for (h = 0u; h < c->kv_heads; h++)
      for (a = 0u; a < (size_t)txn->counts[n]; a++) {
        size_t src = (((n * c->kv_heads + h) * txn->append_width + a) *
                      c->head_dimension);
        size_t dst = ((((layer * c->batch + n) * c->kv_heads + h) *
                       c->capacity + (size_t)c->lengths[n] + a) *
                      c->head_dimension);
        memcpy(c->keys + dst, src_k + src, c->head_dimension * sizeof(float));
        memcpy(c->values + dst, src_v + src, c->head_dimension * sizeof(float));
      }
  txn->staged[layer] = 1u;
  return success_preflighted(error);
}

static void fill_descriptors(et_a2_kv_cache *c, size_t layer,
                             const int64_t *length_data,
                             const uint8_t *mask_data,
                             uint64_t tensor_shape[4], uint64_t length_shape[1],
                             uint64_t mask_shape[2],
                             et_kernel_tensor_view_v1 *keys,
                             et_kernel_tensor_view_v1 *values,
                             et_kernel_tensor_view_v1 *lengths,
                             et_kernel_tensor_view_v1 *key_keep_mask) {
  size_t layer_bytes = c->layer_elements * sizeof(float);
  tensor_shape[0] = c->batch; tensor_shape[1] = c->kv_heads;
  tensor_shape[2] = c->capacity; tensor_shape[3] = c->head_dimension;
  length_shape[0] = c->batch;
  mask_shape[0] = c->batch; mask_shape[1] = c->capacity;
  *keys = (et_kernel_tensor_view_v1){sizeof(*keys), c->keys + layer*c->layer_elements,
      layer_bytes, "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape};
  *values = (et_kernel_tensor_view_v1){sizeof(*values), c->values + layer*c->layer_elements,
      layer_bytes, "f32", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 4u, tensor_shape};
  *lengths = (et_kernel_tensor_view_v1){sizeof(*lengths), (void *)length_data,
      c->batch*sizeof(int64_t), "i64", "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      0u, 1u, length_shape};
  *key_keep_mask = (et_kernel_tensor_view_v1){sizeof(*key_keep_mask),
      (void *)mask_data, c->batch*c->capacity, "bool", "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, 2u, mask_shape};
}

int32_t et_a2_kv_cache_transaction_view_begin_v1(
    et_a2_kv_cache_transaction *handle, uint64_t layer64,
    et_a2_kv_cache_transaction_view **output, et_kernel_error *error) {
  const char *op = "a2-kv-cache.transaction-view-begin";
  et_a2_kv_cache_transaction *txn;
  et_a2_kv_cache_transaction_view *view;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  txn = find_transaction(handle);
  r = output_slot_ok(output, error, op);
  if (r != 0) return r;
  if (*output != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "output slot must contain NULL");
  if (txn == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "transaction handle is invalid");
  if (txn->active_view != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op, "transaction view is already live");
  if (layer64 >= txn->cache->layers || !txn->staged[layer64])
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_PROVIDER_REJECTED, op, "layer is not staged");
  view = a2_calloc(1u, sizeof(*view));
  if (view == NULL)
    return fail(error, ET_KERNEL_ERROR_INTERNAL, ET_KERNEL_CODE_ALLOCATION_FAILED,
                op, "transaction view allocation failed");
  view->magic = VIEW_MAGIC; view->transaction = txn; view->layer = (size_t)layer64;
  fill_descriptors(txn->cache, view->layer, txn->effective_lengths,
                   txn->effective_key_keep_mask, view->tensor_shape,
                   view->lengths_shape, view->mask_shape,
                   &view->keys, &view->values, &view->lengths,
                   &view->key_keep_mask);
  view->registry_next = live_views; live_views = view;
  txn->active_view = view; *output = view;
  return success_preflighted(error);
}

static int descriptor_slots_ok(const void *a, const void *b, const void *c,
                               const void *d,
                               et_kernel_error *error, const char *op) {
  int r = output_slot_ok(a, error, op); if (r != 0) return r;
  r = output_slot_ok(b, error, op); if (r != 0) return r;
  r = output_slot_ok(c, error, op); if (r != 0) return r;
  r = output_slot_ok(d, error, op); if (r != 0) return r;
  if (overlaps(a,sizeof(void*),b,sizeof(void*)) ||
      overlaps(a,sizeof(void*),c,sizeof(void*)) ||
      overlaps(a,sizeof(void*),d,sizeof(void*)) ||
      overlaps(b,sizeof(void*),c,sizeof(void*)) ||
      overlaps(b,sizeof(void*),d,sizeof(void*)) ||
      overlaps(c,sizeof(void*),d,sizeof(void*)))
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_ALIASING_OUTPUT, op, "output slots alias");
  return 0;
}

int32_t et_a2_kv_cache_transaction_view_tensors_v1(
    const et_a2_kv_cache_transaction_view *handle,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **key_keep_mask,
    et_kernel_error *error) {
  const char *op = "a2-kv-cache.transaction-view-tensors";
  et_a2_kv_cache_transaction_view *view;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  view = find_view(handle);
  r = descriptor_slots_ok(keys, values, lengths, key_keep_mask, error, op);
  if (r != 0) return r;
  if (*keys != NULL || *values != NULL || *lengths != NULL ||
      *key_keep_mask != NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "output slots must contain NULL");
  if (view == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "view handle is invalid");
  *keys = &view->keys; *values = &view->values; *lengths = &view->lengths;
  *key_keep_mask = &view->key_keep_mask;
  return success_preflighted(error);
}

int32_t et_a2_kv_cache_transaction_view_end_v1(
    et_a2_kv_cache_transaction_view **slot, et_kernel_error *error) {
  const char *op = "a2-kv-cache.transaction-view-end";
  et_a2_kv_cache_transaction_view *view;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r = output_slot_ok(slot, error, op); if (r != 0) return r;
  view = find_view(*slot);
  if (view == NULL)
    return fail(error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                ET_KERNEL_CODE_INVALID_BUFFER, op, "view handle is invalid");
  view->transaction->active_view = NULL;
  unlink_view(view); view->magic = 0u; free(view); *slot = NULL;
  return success_preflighted(error);
}

static void scrub_transaction(et_a2_kv_cache_transaction *txn) {
  et_a2_kv_cache *c = txn->cache;
  size_t l,n,h,a;
  for (l=0u;l<c->layers;l++) if (txn->staged[l])
    for(n=0u;n<c->batch;n++) for(h=0u;h<c->kv_heads;h++)
      for(a=0u;a<(size_t)txn->counts[n];a++) {
        size_t dst=((((l*c->batch+n)*c->kv_heads+h)*c->capacity+
                    (size_t)c->lengths[n]+a)*c->head_dimension);
        memset(c->keys+dst,0,c->head_dimension*sizeof(float));
        memset(c->values+dst,0,c->head_dimension*sizeof(float));
      }
}

static void free_transaction(et_a2_kv_cache_transaction *txn) {
  txn->cache->active_transaction = NULL; unlink_transaction(txn); txn->magic=0u;
  free(txn->staged); free(txn->effective_key_keep_mask);
  free(txn->effective_lengths); free(txn->counts); free(txn);
}

int32_t et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  const char *op = "a2-kv-cache.transaction-commit";
  et_a2_kv_cache_transaction *txn; size_t i;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r=output_slot_ok(slot,error,op); if(r!=0)return r;
  txn=find_transaction(*slot);
  if(txn==NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"transaction handle is invalid");
  if(txn->active_view!=NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_PROVIDER_REJECTED,op,"transaction view is live");
  for(i=0u;i<txn->cache->layers;i++) if(!txn->staged[i])
    return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"not every layer is staged");
  for(i=0u;i<txn->cache->batch;i++)txn->cache->lengths[i]=txn->effective_lengths[i];
  memcpy(txn->cache->key_keep_mask, txn->effective_key_keep_mask,
         txn->cache->batch * txn->cache->capacity);
  free_transaction(txn); *slot=NULL; return success_preflighted(error);
}

int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  const char *op="a2-kv-cache.transaction-abort";
  et_a2_kv_cache_transaction *txn; int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r=output_slot_ok(slot,error,op);
  if(r!=0)return r;
  txn=find_transaction(*slot);
  if(txn==NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"transaction handle is invalid");
  if(txn->active_view!=NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_PROVIDER_REJECTED,op,"transaction view is live");
  scrub_transaction(txn); free_transaction(txn); *slot=NULL;
  return success_preflighted(error);
}

int32_t et_a2_kv_cache_read_borrow_begin_v1(
    et_a2_kv_cache *handle, et_a2_kv_cache_read_borrow **output,
    et_kernel_error *error) {
  const char *op="a2-kv-cache.read-borrow-begin";
  et_a2_kv_cache *c; et_a2_kv_cache_read_borrow *b;
  size_t layer;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  c=find_cache(handle);
  r=output_slot_ok(output,error,op); if(r!=0)return r;
  if(*output!=NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"output slot must contain NULL");
  if(c==NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"cache handle is invalid");
  if(c->active_transaction!=NULL||c->active_borrow!=NULL)
    return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_PROVIDER_REJECTED,op,"cache already has a live lease");
  b=a2_calloc(1u,sizeof(*b));
  if(b==NULL)goto allocation_failed;
  b->keys=a2_calloc(c->layers,sizeof(*b->keys));
  if(b->keys==NULL)goto allocation_failed;
  b->values=a2_calloc(c->layers,sizeof(*b->values));
  if(b->values==NULL)goto allocation_failed;
  b->magic=BORROW_MAGIC;b->cache=c;
  for(layer=0u;layer<c->layers;layer++) {
    et_kernel_tensor_view_v1 ignored_lengths;
    et_kernel_tensor_view_v1 ignored_mask;
    fill_descriptors(c,layer,c->lengths,c->key_keep_mask,b->tensor_shape,
                     b->lengths_shape,b->mask_shape,&b->keys[layer],
                     &b->values[layer],&ignored_lengths,&ignored_mask);
  }
  fill_descriptors(c,0u,c->lengths,c->key_keep_mask,b->tensor_shape,
                   b->lengths_shape,b->mask_shape,&b->keys[0],&b->values[0],
                   &b->lengths,&b->key_keep_mask);
  b->registry_next=live_borrows;live_borrows=b;c->active_borrow=b;*output=b;
  return success_preflighted(error);
allocation_failed:
  if (b != NULL) { free(b->values); free(b->keys); free(b); }
  return fail(error,ET_KERNEL_ERROR_INTERNAL,
      ET_KERNEL_CODE_ALLOCATION_FAILED,op,"read borrow allocation failed");
}

int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *handle, uint64_t layer64,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **key_keep_mask,
    et_kernel_error *error) {
  const char *op="a2-kv-cache.read-borrow-layer";
  et_a2_kv_cache_read_borrow *b;
  int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  b=find_borrow(handle);
  r=descriptor_slots_ok(keys,values,lengths,key_keep_mask,error,op);if(r!=0)return r;
  if(*keys!=NULL||*values!=NULL||*lengths!=NULL||*key_keep_mask!=NULL)
    return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
        ET_KERNEL_CODE_INVALID_BUFFER,op,"output slots must contain NULL");
  if(b==NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"read borrow handle is invalid");
  if(layer64>=b->cache->layers)return fail(error,ET_KERNEL_ERROR_SHAPE_MISMATCH,
      ET_KERNEL_CODE_INVALID_SHAPE,op,"layer is out of range");
  *keys=&b->keys[layer64];*values=&b->values[layer64];*lengths=&b->lengths;
  *key_keep_mask=&b->key_keep_mask;
  return success_preflighted(error);
}

int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **slot, et_kernel_error *error) {
  const char *op="a2-kv-cache.read-borrow-end";
  et_a2_kv_cache_read_borrow *b;int r;
  if (preflight_error_sink(error) != 0) return raw_reject();
  r=output_slot_ok(slot,error,op);if(r!=0)return r;
  b=find_borrow(*slot);if(b==NULL)return fail(error,ET_KERNEL_ERROR_INVALID_ARGUMENT,
      ET_KERNEL_CODE_INVALID_BUFFER,op,"read borrow handle is invalid");
  b->cache->active_borrow=NULL;unlink_borrow(b);b->magic=0u;
  free(b->values);free(b->keys);free(b);*slot=NULL;
  return success_preflighted(error);
}
