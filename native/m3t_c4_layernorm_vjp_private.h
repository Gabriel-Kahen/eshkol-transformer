#ifndef ET_M3T_C4_LAYERNORM_VJP_PRIVATE_H
#define ET_M3T_C4_LAYERNORM_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_LAYERNORM_VJP_PRIVATE
enum et_m3t_c4_layernorm_vjp_status {
  ET_M3T_C4_LAYERNORM_VJP_OK = 0,
  ET_M3T_C4_LAYERNORM_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_LAYERNORM_VJP_SHAPE = 2,
  ET_M3T_C4_LAYERNORM_VJP_DTYPE = 3,
  ET_M3T_C4_LAYERNORM_VJP_DEVICE = 4,
  ET_M3T_C4_LAYERNORM_VJP_LAYOUT = 5,
  ET_M3T_C4_LAYERNORM_VJP_VALUE = 6,
  ET_M3T_C4_LAYERNORM_VJP_OVERLAP = 7
};

/* Exact C4 row: X,U,dX[1,4,4], gamma,beta,dGamma,dBeta[4],
 * epsilon scalar f32. Synchronous borrowed CPU f32 dense views. No output
 * bytes change on rejection; no pointer or owner is retained. */
int et_m3t_c4_layernorm_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *gamma,
    const et_kernel_tensor_view_v1 *beta,
    const et_kernel_tensor_view_v1 *epsilon,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient,
    et_kernel_tensor_view_v1 *gamma_gradient,
    et_kernel_tensor_view_v1 *beta_gradient);
#endif

#endif
