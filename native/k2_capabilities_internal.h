#ifndef ESHKOL_TRANSFORMER_K2_CAPABILITIES_INTERNAL_H
#define ESHKOL_TRANSFORMER_K2_CAPABILITIES_INTERNAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Private, same-aggregate K2 transport.  Every pointer corresponding to an
 * Eshkol bytevector names its aligned signed-i64 declared-length header; the
 * payload immediately follows that eight-byte header.  Supplied byte counts
 * must equal the declared payload length.  Closure pointers are the only
 * exception.  Diagnostic-returning functions use the E1-aligned category
 * namespace below; other functions use the explicit scalar statuses.
 */
enum {
  ET_K2_STATUS_OK = 0,
  ET_K2_STATUS_INVALID_ARGUMENT = 1,
  ET_K2_STATUS_SHAPE_MISMATCH = 2,
  ET_K2_STATUS_DTYPE_MISMATCH = 3,
  ET_K2_STATUS_DEVICE_MISMATCH = 4,
  ET_K2_STATUS_NONCONTIGUOUS = 5,
  ET_K2_STATUS_UNSUPPORTED = 6,
  ET_K2_STATUS_INVALID_STATE = 7,
  ET_K2_STATUS_VERSION_MISMATCH = 8,
  ET_K2_STATUS_INTERNAL = 9
};

enum {
  ET_K2_SCALAR_INVALID_ARGUMENT = -1,
  ET_K2_SCALAR_INVALID_STATE = -7,
  ET_K2_SCALAR_INTERNAL = -9
};

enum {
  /* 001 is emitted by the Eshkol bridge because native leaves a rejected
   * diagnostic destination unchanged: category INVALID_ARGUMENT. */
  ET_K2_CODE_DIAGNOSTIC_ADMISSION = 0x4b320001u,
  /* INVALID_ARGUMENT */
  ET_K2_CODE_TRANSPORT = 0x4b320002u,
  /* INVALID_STATE */
  ET_K2_CODE_STALE = 0x4b320003u,
  /* INVALID_ARGUMENT */
  ET_K2_CODE_ENTRY_INDEX = 0x4b320004u,
  /* UNSUPPORTED */
  ET_K2_CODE_PROVIDER_ABSENT = 0x4b320005u,
  /* VERSION_MISMATCH for ABI incompatibility; otherwise INTERNAL. */
  ET_K2_CODE_PROVIDER_AUDIT = 0x4b320006u,
  /* INTERNAL */
  ET_K2_CODE_RUNTIME_AUDIT = 0x4b320007u,
  /* Scalar -9; bridge category INTERNAL. */
  ET_K2_CODE_FACTORY_CONFLICT = 0x4b320008u,
  /* Scalar -9; bridge category INTERNAL. */
  ET_K2_CODE_FACTORY_UNREGISTERED = 0x4b320009u,
  /* INTERNAL */
  ET_K2_CODE_INVARIANT = 0x4b32000au
};

/* error_bytes names a length-264 carrier; only its 264-byte payload is written.
 */
int64_t et_k2_private_runtime_ensure_v1(void *error_bytes,
                                        int64_t error_capacity);
/* Returns the current PID (>0), or ET_K2_SCALAR_INTERNAL. */
int64_t et_k2_private_runtime_pid_v1(void);
/* Returns an eligible runtime generation (>0), or ET_K2_SCALAR_INVALID_STATE.
 */
int64_t et_k2_private_runtime_generation_v1(void);
int64_t et_k2_private_runtime_require_v1(
    int64_t expected_generation, int64_t sorted_entry_index,
    const void *operation, int64_t operation_bytes, const void *dtype,
    int64_t dtype_bytes, const void *device, int64_t device_bytes,
    const void *shape_u64le, int64_t shape_bytes, int64_t rank,
    int64_t deterministic, void *error_bytes, int64_t error_capacity);

/* Register: 1 for new/idempotent; -1 malformed; -9 conflict/invariant. */
int64_t et_k2_private_report_factory_register_v1(const void *closure);
int64_t et_k2_private_request_factory_register_v1(const void *closure);
int64_t et_k2_private_entry_factory_register_v1(const void *closure);
/* Authenticate: 1 authentic; 0 unauthentic; -9 unregistered invariant. */
int64_t et_k2_private_report_factory_authenticate_v1(const void *closure);
int64_t et_k2_private_request_factory_authenticate_v1(const void *closure);
int64_t et_k2_private_entry_factory_authenticate_v1(const void *closure);

#ifdef ET_K2_TESTING
#include "eshkol_transformer/kernel_abi.h"

enum {
  ET_K2_TEST_FAIL_NONE = 0,
  ET_K2_TEST_FAIL_BEFORE_DISCOVER = 1,
  ET_K2_TEST_FAIL_AFTER_DISCOVER = 2,
  ET_K2_TEST_FAIL_RUNTIME_AUDIT = 3
};

void et_k2_test_reset_v1(void);
void et_k2_test_runtime_drop_v1(void);
void et_k2_test_provider_override_v1(const et_kernel_provider_v1 *provider,
                                     int enabled);
void et_k2_test_fail_stage_v1(int stage);
uint64_t et_k2_test_discovery_count_v1(void);
uint64_t et_k2_test_destroy_count_v1(void);
uint64_t et_k2_test_runtime_live_count_v1(void);
int64_t et_k2_test_fork_v1(void);
int64_t et_k2_test_wait_child_v1(int64_t child_pid);
void et_k2_test_exit_child_v1(int64_t status);
int64_t et_k2_test_copy_k1_result_v1(int32_t result, uint32_t category,
                                     uint32_t code, uint32_t text_fault,
                                     void *error_bytes, int64_t error_capacity);
#endif

#ifdef __cplusplus
}
#endif

#endif
