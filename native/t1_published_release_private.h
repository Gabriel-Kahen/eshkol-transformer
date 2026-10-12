#ifndef ESHKOL_TRANSFORMER_T1_PUBLISHED_RELEASE_PRIVATE_H
#define ESHKOL_TRANSFORMER_T1_PUBLISHED_RELEASE_PRIVATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef ET_T1_PUBLISHED_RELEASE_PRIVATE
/* Owner-only. A dead pointer remains reserved and cannot name a new shell. */
int64_t et_t1_i64_shell_private_release_preflight_v1(void *candidate);
int64_t et_t1_i64_shell_private_release_v1(void *candidate);
#ifdef ET_T1_I64_SHELL_TESTING
void et_t1_i64_shell_test_fail_release_after_borrow_v1(int enabled);
#endif
#endif

#ifdef __cplusplus
}
#endif

#endif
