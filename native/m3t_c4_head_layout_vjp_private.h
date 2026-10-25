#ifndef ET_M3T_C4_HEAD_LAYOUT_VJP_PRIVATE_H
#define ET_M3T_C4_HEAD_LAYOUT_VJP_PRIVATE_H

#include "eshkol_transformer/kernel_abi.h"

#ifdef ET_M3T_C4_HEAD_LAYOUT_VJP_PRIVATE
enum et_m3t_c4_head_layout_operation {
  ET_M3T_C4_HEAD_LAYOUT_SPLIT = 1,
  ET_M3T_C4_HEAD_LAYOUT_MERGE = 2
};

enum et_m3t_c4_head_layout_vjp_status {
  ET_M3T_C4_HEAD_LAYOUT_VJP_OK = 0,
  ET_M3T_C4_HEAD_LAYOUT_VJP_INVALID_ARGUMENT = 1,
  ET_M3T_C4_HEAD_LAYOUT_VJP_SHAPE = 2,
  ET_M3T_C4_HEAD_LAYOUT_VJP_DTYPE = 3,
  ET_M3T_C4_HEAD_LAYOUT_VJP_DEVICE = 4,
  ET_M3T_C4_HEAD_LAYOUT_VJP_LAYOUT = 5,
  ET_M3T_C4_HEAD_LAYOUT_VJP_VALUE = 6,
  ET_M3T_C4_HEAD_LAYOUT_VJP_OVERLAP = 7
};

/* Exact C4 `[1,4,2,2]` split/merge rows. Split takes X,dX[1,4,4]
 * and U[1,2,4,2]; merge takes X,dX[1,2,4,2] and U[1,4,4]. Borrowed
 * synchronous CPU f32 dense views; status only; no output write on failure. */
int et_m3t_c4_head_layout_vjp_private_v1(
    int operation, const et_kernel_tensor_view_v1 *input,
    const et_kernel_tensor_view_v1 *upstream,
    et_kernel_tensor_view_v1 *input_gradient);
#endif

#endif
