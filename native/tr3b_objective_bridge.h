#ifndef ESHKOL_TRANSFORMER_TR3B_OBJECTIVE_BRIDGE_H
#define ESHKOL_TRANSFORMER_TR3B_OBJECTIVE_BRIDGE_H

#include "eshkol_transformer/kernel_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ET_TR3B_OK = 0,
  ET_TR3B_INVALID_ARGUMENT = 1,
  ET_TR3B_SHAPE_MISMATCH = 2,
  ET_TR3B_DTYPE_MISMATCH = 3,
  ET_TR3B_DEVICE_MISMATCH = 4,
  ET_TR3B_NONCONTIGUOUS = 5,
  ET_TR3B_UNSUPPORTED = 6,
  ET_TR3B_VERSION_MISMATCH = 7,
  ET_TR3B_INTERNAL = 8,
  ET_TR3B_INVALID_STATE = 9
};

/* Private fixed-profile source ABI. These calls are available only inside the
 * source-composed TR3-B aggregate and require serialized trusted callers. */
int64_t et_tr3b_private_stage_input_v1(
    void *m3t_input, const et_kernel_tensor_view_v1 *d2_input);
int64_t et_tr3b_private_copy_targets_v1(
    const et_kernel_tensor_view_v1 *d2_targets, void *target_bytes);
int64_t et_tr3b_private_copy_mask_v1(
    const et_kernel_tensor_view_v1 *d2_mask, void *mask_bytes);
int64_t et_tr3b_private_objective_prepare_v1(
    void *m3_graph_logits, void *m3t_seed, void *target_bytes,
    void *mask_bytes, void *observation_bytes);

#ifdef __cplusplus
}
#endif
#endif
