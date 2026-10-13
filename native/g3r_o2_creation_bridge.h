#ifndef ET_G3R_O2_CREATION_BRIDGE_H
#define ET_G3R_O2_CREATION_BRIDGE_H
#include <stdint.h>
#include "g3r_same_owner_save.h"
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
#ifdef __cplusplus
extern "C" {
#endif
/* Private creation-only O2 hold. The Eshkol caller must root and recheck its
 * original graph; this bridge accepts only the native sealed M3T roster. */
void *et_g3r_private_o2_creation_begin_v1(void *m3t_owner, void *optimizer);
int64_t et_g3r_private_o2_creation_check_v1(void *hold);
/* Both return values certify a drained O2 lease; -1 retains the first error. */
int64_t et_g3r_private_o2_creation_end_v1(void *hold);
int64_t et_g3r_private_o2_creation_error_word_v1(void *hold_or_null,
                                                 int64_t index);
#ifdef ET_G3R_O2_CREATION_TESTING
void et_g3r_o2_creation_test_fail_shell_once_v1(void);
int64_t et_g3r_o2_creation_test_tombstone_cleared_v1(void *hold);
#endif
#ifdef __cplusplus
}
#endif
#endif
#endif
