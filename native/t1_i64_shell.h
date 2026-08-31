#ifndef ESHKOL_TRANSFORMER_T1_I64_SHELL_H
#define ESHKOL_TRANSFORMER_T1_I64_SHELL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_T1_I64_SHELL_ABI_MAJOR 1
#define ET_T1_I64_SHELL_ABI_MINOR 0

enum {
  ET_T1_I64_SHELL_STATUS_OK = 0,
  ET_T1_I64_SHELL_STATUS_INVALID_ARGUMENT = 1,
  ET_T1_I64_SHELL_STATUS_INVALID_STATE = 2,
  ET_T1_I64_SHELL_STATUS_RANGE = 3,
  ET_T1_I64_SHELL_STATUS_INTERNAL = 4,
  ET_T1_I64_SHELL_STATUS_ALLOCATION_FAILED = 5
};

/*
 * Private T1 transport ABI 1.0.  This is deliberately not installed.  A
 * returned pointer is admitted by pointer value before every dereference.
 */
void *et_t1_i64_shell_create_v1(int64_t length);
int64_t et_t1_i64_shell_length_v1(const void *candidate);
int64_t et_t1_i64_shell_write_v1(void *candidate, int64_t index,
                                 int64_t value);
int64_t et_t1_i64_shell_read_v1(const void *candidate, int64_t index);
int64_t et_t1_i64_shell_seal_v1(void *candidate);
int64_t et_t1_i64_shell_abort_v1(void *candidate);
int64_t et_t1_i64_shell_last_status_v1(void);

#ifdef ET_T1_I64_SHELL_TESTING
enum {
  ET_T1_I64_SHELL_TEST_FAIL_NONE = 0,
  ET_T1_I64_SHELL_TEST_FAIL_SHELL_ALLOC = 1,
  ET_T1_I64_SHELL_TEST_FAIL_AFTER_TENSOR = 2,
  ET_T1_I64_SHELL_TEST_FAIL_AFTER_BORROW = 3
};
void et_t1_i64_shell_test_fail_stage_v1(int64_t stage);
int64_t et_t1_i64_shell_test_live_count_v1(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
