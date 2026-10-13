#ifndef ET_G3R_JOINT_HOLD_INTERNAL_H
#define ET_G3R_JOINT_HOLD_INTERNAL_H

#include "o2_optimizer_internal.h"
#include "eshkol_transformer/g3t_transport.h"

#if !defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE)
#error "joint G3/O2 hold requires the same-owner private feature"
#endif
#if !defined(ET_G3R_COMMITTED_HISTORY_PRIVATE)
#error "joint G3/O2 hold requires current-history G3 capture"
#endif

typedef struct et_g3r_joint_hold et_g3r_joint_hold;

/* Private source ABI. All calls require the aggregate's external
 * serialization. A negative result denotes a G3/M3 failure; positive denotes
 * O2; zero is success. last_domain/category/code retain the first failure. */
et_g3r_joint_hold *et_g3r_joint_hold_begin_v1(
    void *generator, void *sealed_model, et_o2_optimizer *optimizer);
int64_t et_g3r_joint_hold_check_v1(et_g3r_joint_hold *hold);
/* Match the exact registered Scheme optimizer record's native pointer to the
 * optimizer held by this joint record, then recheck both original holds. */
int64_t et_g3r_joint_hold_optimizer_match_v1(
    et_g3r_joint_hold *hold, const et_o2_optimizer *optimizer);
et_o2_optimizer_state *et_g3r_joint_hold_issue_v1(et_g3r_joint_hold *hold);
int64_t et_g3r_joint_hold_issued_match_v1(
    et_g3r_joint_hold *hold, const et_o2_optimizer_state *candidate);
int64_t et_g3r_joint_hold_copy_words_v1(
    et_g3r_joint_hold *hold, et_g3r_g3t_held_words_v1 *destination);
/* Returns 0 on consumed success, -1 on proved LIVE, -2 on DEAD with defect.
 * Indeterminate lifecycle fails stop. This is pre-C2-claim authority only. */
int64_t et_g3r_joint_hold_release_issued_v1(et_g3r_joint_hold *hold);
/* Reverse O2 then G3 drain; rejects a still-owned issued state. */
int64_t et_g3r_joint_hold_end_v1(et_g3r_joint_hold *hold);
/* Exact registered joint identity: 1 live, 2 dead, 0 foreign. This query does
 * not clear the saved first error from a preceding end. */
int64_t et_g3r_joint_hold_lifecycle_v1(const et_g3r_joint_hold *hold);
int64_t et_g3r_joint_hold_last_domain_v1(void);
int64_t et_g3r_joint_hold_last_category_v1(void);
int64_t et_g3r_joint_hold_last_code_v1(void);

#endif
