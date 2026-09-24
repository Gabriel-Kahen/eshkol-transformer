/* Fixed private E3 test bridge. The inherited boxed M3 bridge is included once. */
#include "m3_package_bridge.c"

#include <stdint.h>

#include "eshkol_transformer/f32_tensor.h"

extern eshkol_tagged_value_t
et_e3_private_test_run_cabi_v1(eshkol_tagged_value_t scenario,
                               eshkol_tagged_value_t corpus_path);

void et_e3_test_run_v1(void *scenario, void *corpus_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_test_run_cabi_v1(
      *et_e1b_box_value_v1(scenario), *et_e1b_box_value_v1(corpus_path));
}

#ifdef ET_E3_TESTING
void *et_e3_test_destination_create_v1(void) {
  et_f32_tensor *tensor = NULL;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (et_f32_tensor_create_v1(0u, NULL, &tensor, &error) != 0)
    return NULL;
  return tensor;
}

int64_t et_e3_test_destination_destroy_v1(void *value) {
  et_f32_tensor *tensor = (et_f32_tensor *)value;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (tensor == NULL)
    return 1;
  return (int64_t)et_f32_tensor_destroy_v1(&tensor, &error);
}

int64_t et_e3_test_destination_bits_v1(void *value) {
  uint32_t bits = 0u;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (value == NULL ||
      et_f32_tensor_copy_bits_to_v1((const et_f32_tensor *)value, &bits, 1u,
                                    &error) != 0)
    return INT64_C(-1);
  return (int64_t)bits;
}
#endif
