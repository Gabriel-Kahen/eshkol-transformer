#ifndef ET_M3_CALL_PINS_H
#define ET_M3_CALL_PINS_H
#include "../../native/f32_parameter_internal.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct et_g3t_model_pins_internal {
  struct et_g3t_model_pins_internal *self;
  et_f32_parameter *parameters[14];
  const void *identities[14];
  et_f32_tensor *values[14];
  et_kernel_tensor_view_v1 views[14];
  uint16_t held_mask;
} et_g3t_model_pins_internal;
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error);
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins);
#ifdef __cplusplus
}
#endif
#endif
