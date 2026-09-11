#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_READER_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_READER_H

#include "checkpoint_io.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Private, uninstalled C2 same-descriptor checkpoint reader.  Calls are
 * unsynchronized.  The returned pointer is a linear capability owned by its
 * output slot and must not be copied.  A failed read may have changed a payload
 * prefix; its caller must discard the unpublished bytevector.
 */
typedef struct et_c2_checkpoint_reader et_c2_checkpoint_reader;

int64_t et_c2_checkpoint_reader_open_v1(const char *path,
                                        et_c2_checkpoint_reader **reader,
                                        uint64_t *physical_size);
int64_t et_c2_checkpoint_reader_read_exact_v1(et_c2_checkpoint_reader *reader,
                                              uint64_t offset,
                                              void *bytevector_header,
                                              uint64_t expected_length);

/*
 * Re-check the opened regular file's size, then require EOF at the initially
 * observed physical size.  This detects truncation and growth; it does not
 * make the descriptor an immutable snapshot or detect every in-place write.
 */
int64_t
et_c2_checkpoint_reader_validate_final_v1(et_c2_checkpoint_reader *reader);

/* A close error still consumes the live reader and nulls the caller's slot. */
int64_t et_c2_checkpoint_reader_close_v1(et_c2_checkpoint_reader **reader);

#ifdef ET_C2_CHECKPOINT_READER_TESTING
void et_c2_checkpoint_reader_test_reset_v1(void);
void et_c2_checkpoint_reader_test_fail_v1(uint32_t stage, uint32_t occurrence,
                                          int32_t error_number);
/* Reset disables short-read injection.  Setting zero injects zero progress. */
void et_c2_checkpoint_reader_test_set_short_read_v1(size_t maximum_bytes);
size_t et_c2_checkpoint_reader_test_event_count_v1(void);
uint32_t et_c2_checkpoint_reader_test_event_at_v1(size_t index);
size_t et_c2_checkpoint_reader_test_live_count_v1(void);
size_t et_c2_checkpoint_reader_test_fd_live_count_v1(void);
size_t et_c2_checkpoint_reader_test_fd_peak_count_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
