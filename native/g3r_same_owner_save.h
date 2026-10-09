#ifndef ET_G3R_SAME_OWNER_SAVE_H
#define ET_G3R_SAME_OWNER_SAVE_H
#include <stdint.h>
#include "eshkol_transformer/g3t_transport.h"
#if defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE) && \
    !defined(ET_G3R_G3T_SAVE_LEDGER_PRIVATE)
#error "same-owner SAVE requires the G3R committed-history ledger"
#endif
#ifdef ET_G3R_SAME_OWNER_SAVE_PRIVATE
enum {
  ET_G3R_SAVE_SOURCE_NONE = 0,
  ET_G3R_SAVE_SOURCE_M3T = 1,
  ET_G3R_SAVE_SOURCE_O2 = 2,
  ET_G3R_SAVE_SOURCE_G3T = 3,
  ET_G3R_SAVE_SOURCE_COORDINATOR = 4
};
enum {
  ET_G3R_SAVE_COORD_ARGUMENT = 1,
  ET_G3R_SAVE_COORD_STALE = 2,
  ET_G3R_SAVE_COORD_EXHAUSTED = 3,
  ET_G3R_SAVE_COORD_ALLOCATION = 4,
  ET_G3R_SAVE_COORD_ISSUE = 5
};
typedef struct et_g3r_same_owner_save_error_v1 {
  int64_t source, domain, category, code;
} et_g3r_same_owner_save_error_v1;
#ifdef __cplusplus
static_assert(sizeof(et_g3r_same_owner_save_error_v1) == 32,
              "same-owner error record layout changed");
#else
_Static_assert(sizeof(et_g3r_same_owner_save_error_v1) == 32,
               "same-owner error record layout changed");
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Private native composition. Eshkol must authenticate/root all source
 * identities before calling this and recheck them before any publication.
 * External serialization is required. Leaf errors are copied immediately;
 * O2 has no native domain, represented as domain zero. A failed live token
 * remains poisoned through end; read-only getter calls never clear error.
 * A future Eshkol transaction must copy the error before another operation. */
void *et_g3r_private_same_owner_save_begin_v1(
    void *generator, void *m3t_owner, void *optimizer);
int64_t et_g3r_private_same_owner_save_copy_v1(
    void *hold, et_g3r_g3t_idle_snapshot_words_v1 *destination,
    int64_t destination_bytes);
int64_t et_g3r_private_same_owner_save_o2_state_v1(
    void *hold, void **state);
/* One issued state per hold. Raw pointer aliases carry no separate authority. */
int64_t et_g3r_private_same_owner_save_o2_claim_v1(
    void *hold, void *state);
int64_t et_g3r_private_same_owner_save_o2_cancel_v1(void *hold);
/* Retains the original error and hold until pending native cleanup closes. */
int64_t et_g3r_private_same_owner_save_cleanup_retry_v1(void *hold);
int64_t et_g3r_private_same_owner_save_check_v1(void *hold);
int64_t et_g3r_private_same_owner_save_end_v1(void *hold);
int64_t et_g3r_private_same_owner_save_last_error_v1(
    et_g3r_same_owner_save_error_v1 *destination);
#ifdef __cplusplus
}
#endif
#endif
#endif
