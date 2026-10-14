#ifndef ET_G3R_SAME_OWNER_HOLD_BRIDGE_H
#define ET_G3R_SAME_OWNER_HOLD_BRIDGE_H
#include <stdint.h>

#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
/* Source-private, serialized Eshkol-callable bridge. No public C2 receipt. */
/* Positive monotonic ticket; zero denotes failure. Never reuse a dead ticket. */
int64_t et_g3r_private_same_owner_holds_begin_v1(
    void *generator, void *model, void *optimizer);
int64_t et_g3r_private_same_owner_holds_check_v1(
    int64_t pair, void *generator, void *model, void *optimizer);
void *et_g3r_private_same_owner_holds_g3_v1(int64_t pair);
void *et_g3r_private_same_owner_holds_o2_v1(int64_t pair);
/* Copy the held 17-word image to an exact 136-byte Eshkol bytevector. */
int64_t et_g3r_private_same_owner_holds_copy_g3_v1(
    int64_t pair, void *bytevector_header, int64_t payload_bytes);
int64_t et_g3r_private_same_owner_holds_end_v1(int64_t pair);
/* Ticket-backed successor. The v1 pointer-backed pair remains source-private
 * and unchanged until every existing consumer is migrated. */
int64_t et_g3r_private_same_owner_holds_begin_v2(
    void *generator, void *model, void *optimizer);
int64_t et_g3r_private_same_owner_holds_check_v2(
    int64_t pair, void *generator, void *model, void *optimizer);
int64_t et_g3r_private_same_owner_holds_g3_v2(int64_t pair);
int64_t et_g3r_private_same_owner_holds_o2_v2(int64_t pair);
int64_t et_g3r_private_same_owner_holds_copy_g3_v2(
    int64_t pair, void *bytevector_header, int64_t payload_bytes);
int64_t et_g3r_private_same_owner_holds_end_v2(int64_t pair);
int64_t et_g3r_private_same_owner_holds_error_source_v1(void);
int64_t et_g3r_private_same_owner_holds_error_category_v1(void);
int64_t et_g3r_private_same_owner_holds_error_code_v1(void);
void et_g3r_private_same_owner_holds_fail_stop_v1(void);
#endif
#endif
