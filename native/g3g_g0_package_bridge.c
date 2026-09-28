/* Fixed G3-G revision-2 G0/G1 public boxed wrappers. */
#include "m3_package_bridge.c"
extern eshkol_tagged_value_t et_e1b_private_g3_generator_create_cabi_v1(eshkol_tagged_value_t arg0, eshkol_tagged_value_t arg1, eshkol_tagged_value_t arg2);
void et_e1b_public_g3_generator_create_v1(void *arg0, void *arg1, void *arg2, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generator_create_cabi_v1(*et_e1b_box_value_v1(arg0), *et_e1b_box_value_v1(arg1), *et_e1b_box_value_v1(arg2));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generator_generate_cabi_v1(eshkol_tagged_value_t arg0, eshkol_tagged_value_t arg1);
void et_e1b_public_g3_generator_generate_v1(void *arg0, void *arg1, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generator_generate_cabi_v1(*et_e1b_box_value_v1(arg0), *et_e1b_box_value_v1(arg1));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_input_create_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_input_create_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_input_create_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_token_input_create_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_token_input_create_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_token_input_create_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_ids_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_ids_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_ids_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_lengths_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_lengths_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_lengths_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_text_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_text_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_text_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_rng_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_rng_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_rng_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_cache_lengths_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_cache_lengths_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_cache_lengths_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_tensor_release_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_tensor_release_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_tensor_release_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_output_release_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_output_release_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_output_release_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generation_rng_release_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generation_rng_release_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generation_rng_release_cabi_v1(*et_e1b_box_value_v1(arg0));
}
extern eshkol_tagged_value_t et_e1b_private_g3_generator_close_cabi_v1(eshkol_tagged_value_t arg0);
void et_e1b_public_g3_generator_close_v1(void *arg0, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_g3_generator_close_cabi_v1(*et_e1b_box_value_v1(arg0));
}
