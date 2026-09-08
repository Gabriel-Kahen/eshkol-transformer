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
  ET_D2_SHELL_FACTORY_DATASET = 1,
  ET_D2_SHELL_FACTORY_BATCH = 2
};

/*
 * Register and validate the generated-code identity of the two private D2
 * shell constructors. Eshkol checks procedure? before either call. The native
 * side retains only two code addresses, never a shell/environment pointer, so
 * this authentication does not extend any per-dataset or per-batch lifetime.
 */
int64_t et_d2_shell_factory_register_v1(const void *closure,
                                        int64_t factory_kind);
int64_t et_d2_shell_factory_validate_v1(const void *closure,
                                        int64_t factory_kind);

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
 * configuration, manifest, and every shard. shuffle_slots is the already
 * admitted exact min(B,M) count of signed-i64 permutation ordinals. Open first
 * allocates an unpublished control and then exactly shuffle_slots native i64
 * slots; either allocation failure publishes nothing. On success the dataset
 * registry owns both allocations until close. Store/load are allocation-free,
 * require the exact live owner and an index in [0, shuffle_slots), and never
 * disclose the backing pointer. Load reports failure through
 * et_d2_batch_last_status_v1 and returns zero only as a non-authoritative value.
 *
 * Close rejects before mutation during an active scoped borrow; otherwise it
 * unregisters that entry before destroying its shuffle storage and current
 * carrier and is a nonallocating, nonrecoverable tail after admission. Public
 * idempotence remains an Eshkol duty; shell constructor identity is admitted
 * above before Eshkol discloses a private query token.
 */
int64_t et_d2_dataset_open_v1(const void *owner, int64_t shuffle_slots);
int64_t et_d2_dataset_close_v1(const void *owner);
int64_t et_d2_shuffle_window_store_v1(const void *owner, int64_t index,
                                      int64_t value);
int64_t et_d2_shuffle_window_load_v1(const void *owner, int64_t index);

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
 * The legacy entry accepts tagged Eshkol bytevectors containing exactly 8*count
 * i64-le bytes. The source-span entry accepts larger tagged bytevectors and
 * selects 8*count bytes at each nonnegative payload-relative byte offset. Both
 * fully validate owner/state, destination, source subtraction bounds, and both
 * bytevectors before writing either plane or mask. Unwritten calloc-zeroed
 * positions remain masked filler. Token/range/duplicate-write semantics remain
 * trusted Eshkol responsibilities.
 */
int64_t et_d2_batch_write_pair_i64le_span_v1(
    const void *owner, int64_t batch_generation, int64_t destination,
    const void *inputs_header, int64_t input_bytes, const void *targets_header,
    int64_t target_bytes, int64_t count);
int64_t et_d2_batch_write_pair_i64le_source_span_v1(
    const void *owner, int64_t batch_generation, int64_t destination,
    const void *inputs_header, int64_t input_bytes, int64_t input_offset,
    const void *targets_header, int64_t target_bytes, int64_t target_offset,
    int64_t count);
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
 * Preflight rejects a missing generation or active scoped borrow without receiver
 * mutation. After it succeeds, serialized Eshkol code invalidates the current
 * generation and all three shells, then calls release as the fixed, nonallocating
 * destruction tail. Release is also valid for a freshly created unpublished
 * batch known by its trusted owner to have no borrow; violating either admission
 * rule is an internal invariant failure, not a recoverable release result.
 * Authentic caller-retained shell aliases remain caller-region-managed Eshkol
 * values; wrapper validation makes their repeated release idempotent and
 * rejects every other stale use before this native boundary.
 */
int64_t et_d2_batch_release_preflight_v1(const void *owner,
                                         int64_t batch_generation);
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
  ET_D2_TEST_FAIL_DATASET = 8,
  ET_D2_TEST_FAIL_SHUFFLE = 9
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
