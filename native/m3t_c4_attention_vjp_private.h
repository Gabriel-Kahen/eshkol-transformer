#ifndef ET_M3T_C4_ATTENTION_VJP_PRIVATE_H
#define ET_M3T_C4_ATTENTION_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_ATTENTION_VJP_PRIVATE
enum et_m3t_c4_attention_vjp_status {
  ET_M3T_C4_ATTENTION_VJP_OK = 0,
  ET_M3T_C4_ATTENTION_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_ATTENTION_VJP_SHAPE = 2,
  ET_M3T_C4_ATTENTION_VJP_DTYPE = 3,
  ET_M3T_C4_ATTENTION_VJP_DEVICE = 4,
  ET_M3T_C4_ATTENTION_VJP_LAYOUT = 5,
  ET_M3T_C4_ATTENTION_VJP_VALUE = 6,
  ET_M3T_C4_ATTENTION_VJP_OVERLAP = 7
};

/* Fixed C4 self-attention: Q,K,V,U,dQ,dK,dV [1,2,4,2], positions
 * [1,4], keep [1,4,4]. Synchronous borrowed CPU dense views. The
 * caller owns storage; failure leaves all gradients untouched. */
int et_m3t_c4_attention_vjp_private_v1(
    const et_kernel_tensor_view_v1 *query,
    const et_kernel_tensor_view_v1 *key,
    const et_kernel_tensor_view_v1 *value,
    const et_kernel_tensor_view_v1 *query_positions,
    const et_kernel_tensor_view_v1 *key_positions,
    const et_kernel_tensor_view_v1 *keep,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *query_gradient,
    et_kernel_tensor_view_v1 *key_gradient,
    et_kernel_tensor_view_v1 *value_gradient);
#endif

#endif
