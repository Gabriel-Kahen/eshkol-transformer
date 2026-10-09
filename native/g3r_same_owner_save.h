#ifndef ET_G3R_SAME_OWNER_SAVE_H
#define ET_G3R_SAME_OWNER_SAVE_H
#include <stdint.h>
#include "eshkol_transformer/g3t_transport.h"
#if defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE) && \
    !defined(ET_G3R_G3T_SAVE_LEDGER_PRIVATE)
#error "same-owner SAVE requires the G3R committed-history ledger"
#endif
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
/* Private native composition. Eshkol must authenticate/root all source
 * identities before calling this and recheck them before any publication. */
void *et_g3r_private_same_owner_save_begin_v1(
    void *generator, void *m3t_owner, void *optimizer);
int64_t et_g3r_private_same_owner_save_copy_v1(
    void *hold, et_g3r_g3t_idle_snapshot_words_v1 *destination,
    int64_t destination_bytes);
int64_t et_g3r_private_same_owner_save_o2_state_v1(
    void *hold, void **state);
int64_t et_g3r_private_same_owner_save_check_v1(void *hold);
int64_t et_g3r_private_same_owner_save_end_v1(void *hold);
#endif
#endif
