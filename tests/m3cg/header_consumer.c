#include "../../src/eshkol_transformer/m3_call_pins.h"
int m3cg_header_consumer(et_f32_parameter *const parameters[14],
    const void *const identities[14], et_g3t_model_pins_internal *pins,
    et_f32_tensor_error *error) {
  int32_t result = et_g3t_model_pins_begin_internal(parameters, identities, pins, error);
  if (result == 0) {
    result = et_g3t_model_pins_check_internal(pins, error);
    et_g3t_model_pins_end_internal(pins);
  }
  return result;
}
