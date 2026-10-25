#ifndef ET_M3T_C4_A2_EMPTY_STAGE_VJP_PRIVATE_H
#define ET_M3T_C4_A2_EMPTY_STAGE_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_A2_EMPTY_STAGE_VJP_PRIVATE
enum et_m3t_c4_a2_empty_stage_vjp_status {
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_OK = 0,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_SHAPE = 2,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_DTYPE = 3,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_DEVICE = 4,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_LAYOUT = 5,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_VALUE = 6,
  ET_M3T_C4_A2_EMPTY_STAGE_VJP_OVERLAP = 7
};

/* Mathematical reverse of the fixed empty-cache, full-four-token A2 stage:
 * dCacheK/V[1,2,4,2] -> dStagedK/V[1,2,4,2]. Borrowed CPU f32 dense views;
 * status only, no output write on rejection. This does not authenticate an
 * A2 transaction or bind a gradient to a C4 owner. */
int et_m3t_c4_a2_empty_stage_vjp_private_v1(
    const et_kernel_tensor_view_v1 *cache_key_gradient,
    const et_kernel_tensor_view_v1 *cache_value_gradient,
    et_kernel_tensor_view_v1 *staged_key_gradient,
    et_kernel_tensor_view_v1 *staged_value_gradient);
#endif

#endif
