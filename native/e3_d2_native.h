#ifndef ESHKOL_TRANSFORMER_E3_D2_NATIVE_H
#define ESHKOL_TRANSFORMER_E3_D2_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private, uninstalled E3 seam; substitutes the original D2 TU, never adds one.
 * Owner is compared only. Return and D2 last_status are 1 for NULL/unenrolled,
 * 2 for any current batch (including unpublished/leased), otherwise 0.
 * Only last_status changes; generation exhaustion does not prevent idle. */
int64_t et_e3_d2_dataset_idle_preflight_v1(const void *owner);

#ifdef __cplusplus
}
#endif

#endif
