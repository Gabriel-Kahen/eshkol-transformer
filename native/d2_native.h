#ifndef ESHKOL_TRANSFORMER_D2_NATIVE_H
#define ESHKOL_TRANSFORMER_D2_NATIVE_H

#include "eshkol_transformer/kernel_abi.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private, uninstalled, statically pinned D2 scalar/ptr transport ABI. */
enum {
  ET_D2_NATIVE_STATUS_OK = 0,
  ET_D2_NATIVE_STATUS_INVALID_ARGUMENT = 1,
  ET_D2_NATIVE_STATUS_INVALID_STATE = 2,
  ET_D2_NATIVE_STATUS_RANGE = 3,
  ET_D2_NATIVE_STATUS_ALLOCATION_FAILED = 4,
  ET_D2_NATIVE_STATUS_INTERNAL = 5,
  ET_D2_NATIVE_STATUS_UNSUPPORTED = 6
};

enum {
  ET_D2_EXACT_READ_ARGUMENT = 1,
  ET_D2_EXACT_READ_OPEN = 2,
  ET_D2_EXACT_READ_READ = 3,
  ET_D2_EXACT_READ_CLOSE = 4
};

enum {
  ET_D2_TEST_READ_FAIL_NONE = 0,
  ET_D2_TEST_READ_FAIL_READ = 1,
  ET_D2_TEST_READ_FAIL_CLOSE = 2
};

/*
 * Eshkol bytevectors are an exact i64 native-length header followed by bytes.
 * Reads may partially change the payload on failure; the caller must discard
 * it. Failure is encoded as (stage << 32) | errno, matching private D1 I/O.
 */
int64_t et_d2_exact_read_v1(const char *path, int64_t path_bytes,
                            int64_t offset, void *bytevector_header,
                            int64_t expected_length);

/*
 * Open is called only after the Eshkol loader has completely validated its
 * configuration, manifest, and every shard. It publishes one native registry
 * entry for owner. Close rejects before mutation during an active scoped borrow;
 * otherwise it unregisters that entry before destroying its current carrier and
 * is a nonallocating, nonrecoverable tail after admission. Public idempotence and
 * exact shell authentication remain Eshkol duties.
 */
int64_t et_d2_dataset_open_v1(const void *owner);
int64_t et_d2_dataset_close_v1(const void *owner);

/*
 * owner is a stable Eshkol dataset identity, compared but never dereferenced.
 * Create returns a positive, owner-qualified generation or zero on failure.
 * Every later operation requires owner plus generation, preventing allocator
 * address ABA while retaining no released native control/tombstone.
 */
int64_t et_d2_batch_create_v1(const void *owner, int64_t rows, int64_t columns,
                              int64_t max_payload);
int64_t et_d2_batch_last_status_v1(void);

/*
 * Both inputs are tagged Eshkol bytevectors containing exactly 8*count i64-le
 * bytes. A validated bounded span writes both planes and sets the identical
 * bool mask span true. Unwritten calloc-zeroed positions remain masked filler.
 * Token/range/duplicate-write semantics remain trusted Eshkol responsibilities.
 */
int64_t et_d2_batch_write_pair_i64le_span_v1(
    const void *owner, int64_t batch_generation, int64_t destination,
    const void *inputs_header, int64_t input_bytes, const void *targets_header,
    int64_t target_bytes, int64_t count);
int64_t et_d2_batch_seal_v1(const void *owner, int64_t batch_generation);

/*
 * One embedded downstream lease may be live. Begin returns a positive unique
 * lease generation or zero on failure. The three accessors return stable,
 * logically read-only dense CPU K1 views until borrow_end.
 */
int64_t et_d2_batch_borrow_begin_v1(const void *owner,
                                    int64_t batch_generation);
const et_kernel_tensor_view_v1 *
et_d2_batch_borrow_inputs_v1(const void *owner, int64_t batch_generation,
                             int64_t lease_generation);
const et_kernel_tensor_view_v1 *
et_d2_batch_borrow_targets_v1(const void *owner, int64_t batch_generation,
                              int64_t lease_generation);
const et_kernel_tensor_view_v1 *
et_d2_batch_borrow_loss_mask_v1(const void *owner, int64_t batch_generation,
                                int64_t lease_generation);
int64_t et_d2_batch_borrow_end_v1(const void *owner, int64_t batch_generation,
                                  int64_t lease_generation);

/*
 * The public wrapper invalidates the dataset's current generation before this
 * exact release. Authentic caller-retained shell aliases remain ordinary GC
 * values; wrapper validation makes their repeated release idempotent and rejects
 * every other stale use before this native boundary.
 */
int64_t et_d2_batch_release_v1(const void *owner, int64_t batch_generation);

enum {
  ET_D2_TEST_FAIL_NONE = 0,
  ET_D2_TEST_FAIL_CONTROL = 1,
  ET_D2_TEST_FAIL_AFTER_INPUT_TENSOR = 2,
  ET_D2_TEST_FAIL_AFTER_INPUT_BORROW = 3,
  ET_D2_TEST_FAIL_AFTER_TARGET_TENSOR = 4,
  ET_D2_TEST_FAIL_AFTER_TARGET_BORROW = 5,
  ET_D2_TEST_FAIL_MASK = 6,
  ET_D2_TEST_FAIL_BEFORE_PUBLISH = 7,
  ET_D2_TEST_FAIL_DATASET = 8
};

#ifdef ET_D2_NATIVE_TESTING
void et_d2_batch_test_fail_stage_v1(int64_t stage);
void et_d2_exact_read_test_fail_stage_v1(int64_t stage);
int64_t et_d2_dataset_test_live_count_v1(void);
int64_t et_d2_batch_test_live_count_v1(void);
int64_t et_d2_batch_test_borrow_count_v1(void);
int64_t et_d2_test_owned_allocation_count_v1(void);
size_t et_d2_dataset_test_control_bytes_v1(void);
size_t et_d2_batch_test_control_bytes_v1(void);
size_t et_d2_batch_test_borrow_bytes_v1(void);
size_t et_d2_batch_test_view_metadata_bytes_v1(void);
size_t et_d2_batch_test_payload_bytes_v1(const void *owner, int64_t generation);
void et_d2_batch_test_exhaust_generations_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
