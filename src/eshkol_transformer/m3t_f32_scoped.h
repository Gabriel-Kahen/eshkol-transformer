#ifndef ET_M3T_F32_SCOPED_H
#define ET_M3T_F32_SCOPED_H
#include "../../native/f32_parameter_internal.h"
/* Private synchronous stack guard. Initialize to zero; never copy an active guard.
 * self authenticates the exact stack address, owner authenticates the installed
 * active_borrow sentinel. Neither sentinel nor guard is a public I2 lease. */
typedef struct et_f32_scoped_guard_internal {
  et_kernel_tensor_view_v1 view;
  et_f32_tensor *owner;
  struct et_f32_scoped_guard_internal *self;
} et_f32_scoped_guard_internal;
int32_t et_f32_tensor_scoped_begin_internal(et_f32_tensor *,
    et_f32_scoped_guard_internal *, et_f32_tensor_error *);
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *);
#ifdef ET_G3C4_NATIVE_OWNER_PRIVATE
/* Neutral C4 construction teardown preflight. This declaration is absent from
 * ordinary and predecessor aggregates. */
int32_t et_f32_parameter_idle_preflight_internal(
    et_f32_parameter *, et_f32_tensor_error *);
#endif
/* Read-only abort preflight, used only with the fixed M3T owner collection. */
int32_t et_m3t_f32_parameter_preflight_internal(et_f32_parameter *, et_f32_tensor_error *);
#endif
