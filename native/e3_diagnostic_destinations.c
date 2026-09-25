/* Source-private E3 diagnostic destinations. Never expose these pointers. */
#include <stdint.h>

#include "eshkol_transformer/f32_tensor.h"

void *et_e3_diagnostic_destination_create_v1(void) {
  et_f32_tensor *tensor = NULL;
  et_f32_tensor_error error;

  et_f32_tensor_error_clear_v1(&error);
  if (et_f32_tensor_create_v1(0u, NULL, &tensor, &error) != 0)
    return NULL;
  return tensor;
}

int64_t et_e3_diagnostic_destination_bits_v1(void *value) {
  uint32_t bits = 0u;
  et_f32_tensor_error error;

  et_f32_tensor_error_clear_v1(&error);
  if (value == NULL ||
      et_f32_tensor_copy_bits_to_v1((const et_f32_tensor *)value, &bits, 1u,
                                    &error) != 0)
    return INT64_C(-1);
  return (int64_t)bits;
}

int64_t et_e3_diagnostic_destination_destroy_v1(void *value) {
  et_f32_tensor *tensor = (et_f32_tensor *)value;
  et_f32_tensor_error error;

  et_f32_tensor_error_clear_v1(&error);
  if (tensor == NULL)
    return INT64_C(1);
  return (int64_t)et_f32_tensor_destroy_v1(&tensor, &error);
}
