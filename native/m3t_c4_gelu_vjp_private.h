#ifndef ET_M3T_C4_GELU_VJP_PRIVATE_H
#define ET_M3T_C4_GELU_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_GELU_VJP_PRIVATE
enum et_m3t_c4_gelu_vjp_status {
  ET_M3T_C4_GELU_VJP_OK = 0,
  ET_M3T_C4_GELU_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_GELU_VJP_SHAPE = 2,
  ET_M3T_C4_GELU_VJP_DTYPE = 3,
  ET_M3T_C4_GELU_VJP_DEVICE = 4,
  ET_M3T_C4_GELU_VJP_LAYOUT = 5,
  ET_M3T_C4_GELU_VJP_VALUE = 6,
  ET_M3T_C4_GELU_VJP_OVERLAP = 7
};

/* Borrowed, synchronous C4 GELU VJP. The destination is unchanged on error. */
int et_m3t_c4_gelu_vjp_private_v1(
    const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *gradient);
#endif

#endif
