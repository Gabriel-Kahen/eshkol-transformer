#ifndef ET_M3T_C4_LINEAR_VJP_PRIVATE_H
#define ET_M3T_C4_LINEAR_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_LINEAR_VJP_PRIVATE
enum et_m3t_c4_linear_vjp_status {
  ET_M3T_C4_LINEAR_VJP_OK = 0,
  ET_M3T_C4_LINEAR_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_LINEAR_VJP_SHAPE = 2,
  ET_M3T_C4_LINEAR_VJP_DTYPE = 3,
  ET_M3T_C4_LINEAR_VJP_DEVICE = 4,
  ET_M3T_C4_LINEAR_VJP_LAYOUT = 5,
  ET_M3T_C4_LINEAR_VJP_VALUE = 6,
  ET_M3T_C4_LINEAR_VJP_OVERLAP = 7
};

/* Synchronous borrowed CPU f32 views for the accepted four-token C4
 * no-bias rows. Inputs: X[1,4,I], W[O,I], U[1,4,O]. Outputs: dX[1,4,I],
 * dW[O,I]. Returns status only; no output is written on rejection. */
int et_m3t_c4_linear_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *weight,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient,
    et_kernel_tensor_view_v1 *weight_gradient);
#endif

#endif
