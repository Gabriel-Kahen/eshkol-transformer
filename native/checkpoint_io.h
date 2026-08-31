#ifndef ESHKOL_TRANSFORMER_CHECKPOINT_IO_H
#define ESHKOL_TRANSFORMER_CHECKPOINT_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_CHECKPOINT_IO_ABI_MAJOR 1u
#define ET_CHECKPOINT_IO_ABI_MINOR 0u

/*
 * Status layout is stable for ABI 1: errno occupies bits 0..15, stage bits
 * 16..23, and bit 24 records that atomic rename already committed.  Zero is
 * success.  A committed failure means the new complete artifact is visible,
 * but its directory entry was not proved durable.
 */
#define ET_CHECKPOINT_IO_STATUS_ERRNO_MASK UINT64_C(0xffff)
#define ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT 16u
#define ET_CHECKPOINT_IO_STATUS_STAGE_MASK UINT64_C(0xff)
#define ET_CHECKPOINT_IO_STATUS_COMMITTED UINT64_C(0x01000000)

enum et_checkpoint_io_stage {
  ET_CHECKPOINT_IO_STAGE_NONE = 0,
  ET_CHECKPOINT_IO_STAGE_VALIDATE = 1,
  ET_CHECKPOINT_IO_STAGE_ALLOCATE = 2,
  ET_CHECKPOINT_IO_STAGE_OPEN_PARENT = 3,
  ET_CHECKPOINT_IO_STAGE_OPEN_SOURCE = 4,
  ET_CHECKPOINT_IO_STAGE_STAT_SOURCE = 5,
  ET_CHECKPOINT_IO_STAGE_READ_SOURCE = 6,
  ET_CHECKPOINT_IO_STAGE_CLOSE_SOURCE = 7,
  ET_CHECKPOINT_IO_STAGE_RANDOM_TEMP = 8,
  ET_CHECKPOINT_IO_STAGE_CREATE_TEMP = 9,
  ET_CHECKPOINT_IO_STAGE_WRITE_TEMP = 10,
  ET_CHECKPOINT_IO_STAGE_SYNC_TEMP = 11,
  ET_CHECKPOINT_IO_STAGE_CLOSE_TEMP = 12,
  ET_CHECKPOINT_IO_STAGE_PUBLISH = 13,
  ET_CHECKPOINT_IO_STAGE_SYNC_DIRECTORY = 14,
  ET_CHECKPOINT_IO_STAGE_CLEANUP_TEMP = 15,
  ET_CHECKPOINT_IO_STAGE_CLOSE_DIRECTORY = 16
};

static inline uint32_t et_checkpoint_io_status_errno_v1(int64_t status) {
  return (uint32_t)((uint64_t)status & ET_CHECKPOINT_IO_STATUS_ERRNO_MASK);
}

static inline uint32_t et_checkpoint_io_status_stage_v1(int64_t status) {
  return (uint32_t)(((uint64_t)status >> ET_CHECKPOINT_IO_STATUS_STAGE_SHIFT) &
                    ET_CHECKPOINT_IO_STATUS_STAGE_MASK);
}

static inline int et_checkpoint_io_status_committed_v1(int64_t status) {
  return ((uint64_t)status & ET_CHECKPOINT_IO_STATUS_COMMITTED) != 0u;
}

/*
 * bytevector_header points at pinned-Eshkol's bytevector payload header:
 * signed i64 length at offset 0 followed by exact bytes at offset 8.
 * The caller supplies the independently checked expected length.  These calls
 * are unsynchronized and accept no concurrent mutation of path, bytevector, or
 * containing directory. Parent-directory symlinks are rejected at the final
 * component; intermediate path components remain in the caller's trust domain.
 */
int64_t et_checkpoint_io_abi_major_v1(void);
int64_t et_checkpoint_io_abi_minor_v1(void);
int64_t et_checkpoint_io_read_exact_v1(const char *path,
                                       void *bytevector_header,
                                       uint64_t expected_length,
                                       uint64_t maximum_length);
int64_t et_checkpoint_io_atomic_write_v1(const char *path,
                                         const void *bytevector_header,
                                         uint64_t expected_length,
                                         int64_t overwrite);

#ifdef ET_CHECKPOINT_IO_TESTING
void et_checkpoint_io_test_reset_v1(void);
void et_checkpoint_io_test_fail_v1(uint32_t stage, uint32_t occurrence,
                                   int32_t error_number);
/* Reset disables short-I/O injection. Setting zero injects zero progress. */
void et_checkpoint_io_test_set_short_io_v1(size_t maximum_bytes);
void et_checkpoint_io_test_set_random_v1(const uint8_t bytes[16]);
size_t et_checkpoint_io_test_event_count_v1(void);
uint32_t et_checkpoint_io_test_event_at_v1(size_t index);
const char *et_checkpoint_io_test_last_temp_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
