#ifndef ET_G3R_SAME_OWNER_HOLD_BRIDGE_H
#define ET_G3R_SAME_OWNER_HOLD_BRIDGE_H
#include <stdint.h>

#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
/* Source-private, serialized Eshkol-callable bridge. No public C2 receipt. */
void *et_g3r_private_same_owner_holds_begin_v1(
    void *generator, void *model, void *optimizer);
int64_t et_g3r_private_same_owner_holds_check_v1(
    void *pair, void *generator, void *model, void *optimizer);
void *et_g3r_private_same_owner_holds_g3_v1(void *pair);
void *et_g3r_private_same_owner_holds_o2_v1(void *pair);
int64_t et_g3r_private_same_owner_holds_end_v1(void *pair);
int64_t et_g3r_private_same_owner_holds_error_source_v1(void);
int64_t et_g3r_private_same_owner_holds_error_category_v1(void);
int64_t et_g3r_private_same_owner_holds_error_code_v1(void);
void et_g3r_private_same_owner_holds_fail_stop_v1(void);
#endif
#endif
