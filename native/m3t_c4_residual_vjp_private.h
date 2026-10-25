#ifndef ET_M3T_C4_RESIDUAL_VJP_PRIVATE_H
#define ET_M3T_C4_RESIDUAL_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_RESIDUAL_VJP_PRIVATE
enum et_m3t_c4_residual_vjp_status {
  ET_M3T_C4_RESIDUAL_VJP_OK = 0,
  ET_M3T_C4_RESIDUAL_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_RESIDUAL_VJP_SHAPE = 2,
  ET_M3T_C4_RESIDUAL_VJP_DTYPE = 3,
  ET_M3T_C4_RESIDUAL_VJP_DEVICE = 4,
  ET_M3T_C4_RESIDUAL_VJP_LAYOUT = 5,
  ET_M3T_C4_RESIDUAL_VJP_VALUE = 6,
  ET_M3T_C4_RESIDUAL_VJP_OVERLAP = 7
};

/* Borrowed, synchronous f32[1,4,4] residual VJP. No output is written on
 * rejection. Gradients are separate caller-owned buffers. */
int et_m3t_c4_residual_vjp_private_v1(
    const et_kernel_tensor_view_v1 *left,
    const et_kernel_tensor_view_v1 *right,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *left_gradient,
    et_kernel_tensor_view_v1 *right_gradient);
#endif

#endif
