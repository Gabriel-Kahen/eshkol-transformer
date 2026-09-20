/* Fixed M3 boxed wrappers; trusted implementation is localized. */
#include "m3t_package_bridge.c"

extern eshkol_tagged_value_t et_e1b_private_m3_model_forward_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second, eshkol_tagged_value_t third);
void et_e1b_public_m3_model_forward_v1(void *first, void *second, void *third, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_model_forward_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second), *et_e1b_box_value_v1(third));
}

extern eshkol_tagged_value_t et_e1b_private_m3_model_output_logits_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_model_output_logits_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_model_output_logits_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3_model_output_loss_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_model_output_loss_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_model_output_loss_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3_model_output_rng_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_model_output_rng_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_model_output_rng_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3_model_output_release_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_model_output_release_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_model_output_release_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3_diagnostic_output_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second, eshkol_tagged_value_t third, eshkol_tagged_value_t fourth);
void et_e1b_public_m3_diagnostic_output_vjp_v1(void *first, void *second, void *third, void *fourth, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_diagnostic_output_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second), *et_e1b_box_value_v1(third), *et_e1b_box_value_v1(fourth));
}

extern eshkol_tagged_value_t et_e1b_private_m3_diagnostic_model_logits_bits_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_diagnostic_model_logits_bits_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_diagnostic_model_logits_bits_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3_diagnostic_model_logits_release_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3_diagnostic_model_logits_release_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3_diagnostic_model_logits_release_cabi_v1(*et_e1b_box_value_v1(first));
}
