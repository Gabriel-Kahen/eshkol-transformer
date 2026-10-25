#ifndef ET_M3T_C4_EMBEDDING_VJP_PRIVATE_H
#define ET_M3T_C4_EMBEDDING_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_EMBEDDING_VJP_PRIVATE
enum et_m3t_c4_embedding_vjp_status {
  ET_M3T_C4_EMBEDDING_VJP_OK = 0,
  ET_M3T_C4_EMBEDDING_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_EMBEDDING_VJP_SHAPE = 2,
  ET_M3T_C4_EMBEDDING_VJP_DTYPE = 3,
  ET_M3T_C4_EMBEDDING_VJP_DEVICE = 4,
  ET_M3T_C4_EMBEDDING_VJP_LAYOUT = 5,
  ET_M3T_C4_EMBEDDING_VJP_VALUE = 6,
  ET_M3T_C4_EMBEDDING_VJP_OVERLAP = 7
};

/* Raw V256 IDs [1,4], upstream [1,4,4], tied token-table gradient
 * [256,4]. Synchronous borrowed CPU views; status only, no write on error. */
int et_m3t_c4_embedding_vjp_private_v1(
    const et_kernel_tensor_view_v1 *ids,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *gradient);
#endif

#endif
