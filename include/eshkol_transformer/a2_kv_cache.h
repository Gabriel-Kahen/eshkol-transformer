#ifndef ESHKOL_TRANSFORMER_A2_KV_CACHE_H
#define ESHKOL_TRANSFORMER_A2_KV_CACHE_H

#include "eshkol_transformer/kernel_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_A2_KV_CACHE_ABI_MAJOR 1u
#define ET_A2_KV_CACHE_ABI_MINOR 0u

typedef struct et_a2_kv_cache et_a2_kv_cache;
typedef struct et_a2_kv_cache_transaction et_a2_kv_cache_transaction;
typedef struct et_a2_kv_cache_transaction_view
    et_a2_kv_cache_transaction_view;
typedef struct et_a2_kv_cache_read_borrow et_a2_kv_cache_read_borrow;

int32_t et_a2_kv_cache_abi_major_v1(void);
int32_t et_a2_kv_cache_abi_minor_v1(void);
int32_t et_a2_kv_cache_abi_require_v1(uint32_t major,
                                      uint32_t minimum_minor,
                                      et_kernel_error *error);

/*
 * Creates distinct, zero-initialized CPU-f32 K/V storage with physical shape
 * [layers,batch,kv_heads,capacity,head_dimension] and exact i64 lengths[batch].
 * All dimensions are positive. Storage identity and capacity never change.
 * Pointer-valued output slots must be non-NULL, disjoint, and initially NULL.
 */
int32_t et_a2_kv_cache_create_v1(uint64_t layers, uint64_t batch,
                                 uint64_t kv_heads, uint64_t capacity,
                                 uint64_t head_dimension,
                                 et_a2_kv_cache **cache,
                                 et_kernel_error *error);
int32_t et_a2_kv_cache_destroy_v1(et_a2_kv_cache **cache,
                                  et_kernel_error *error);

/*
 * append_counts is borrowed only for this call and must be a dense zero-offset
 * CPU i64 K1 view of shape [batch]. append_width must be positive; every count
 * is in [0,append_width], at least one is positive, and old_length+count must
 * not exceed capacity. Beginning a transaction allocates only control/snapshot
 * state; it never reallocates cache storage.
 */
int32_t et_a2_kv_cache_transaction_begin_v1(
    et_a2_kv_cache *cache, uint64_t append_width,
    const et_kernel_tensor_view_v1 *append_counts,
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);

/*
 * Each layer is staged exactly once from distinct, borrowed, dense zero-offset
 * CPU-f32 K1 views [batch,kv_heads,append_width,head_dimension]. Only the first
 * append_counts[n] positions of each row are copied. Validation is complete
 * before mutation and sources are never retained or modified.
 */
int32_t et_a2_kv_cache_transaction_stage_layer_v1(
    et_a2_kv_cache_transaction *transaction, uint64_t layer,
    const et_kernel_tensor_view_v1 *keys,
    const et_kernel_tensor_view_v1 *values, et_kernel_error *error);

/*
 * A view lease may begin only for an already-staged layer. It exposes immutable
 * full-capacity dense K/V views [batch,kv_heads,capacity,head_dimension], an
 * immutable dense i64 effective-length view [batch], and an immutable dense
 * bool key-keep mask [batch,capacity]. Mask bytes are exactly 1 before each
 * effective length and 0 otherwise. While live, the lease
 * blocks staging, commit, abort, and cache destruction. Returned descriptors
 * and their storage are valid only until view_end.
 */
int32_t et_a2_kv_cache_transaction_view_begin_v1(
    et_a2_kv_cache_transaction *transaction, uint64_t layer,
    et_a2_kv_cache_transaction_view **view, et_kernel_error *error);
int32_t et_a2_kv_cache_transaction_view_tensors_v1(
    const et_a2_kv_cache_transaction_view *view,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **effective_lengths,
    const et_kernel_tensor_view_v1 **key_keep_mask,
    et_kernel_error *error);
int32_t et_a2_kv_cache_transaction_view_end_v1(
    et_a2_kv_cache_transaction_view **view, et_kernel_error *error);

/* Commit requires every layer staged. Abort scrubs every staged append slot. */
int32_t et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);
int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);

/*
 * One committed read borrow excludes transactions and destruction. A layer
 * query exposes immutable full-capacity dense K/V views, committed lengths, and
 * the exact dense bool key-keep mask [batch,capacity]; descriptors and storage
 * remain valid until read_borrow_end.
 */
int32_t et_a2_kv_cache_read_borrow_begin_v1(
    et_a2_kv_cache *cache, et_a2_kv_cache_read_borrow **borrow,
    et_kernel_error *error);
int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *borrow, uint64_t layer,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **key_keep_mask,
    et_kernel_error *error);
int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **borrow, et_kernel_error *error);

#ifdef ET_A2_KV_CACHE_TESTING
void et_a2_kv_cache_test_fail_alloc_after_v1(size_t successful_allocations);
void et_a2_kv_cache_test_reset_allocator_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
