/* Fixed M3T boxed wrappers. All trusted definitions are localized. */
#include "e1b_error_consumer_bridge.h"
#include "i2_wave2_package_bridge.c"

extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_state_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_initializer_state_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_initializer_state_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_algorithm_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_initializer_algorithm_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_initializer_algorithm_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_words_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_initializer_words_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_initializer_words_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_model_create_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_model_create_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_model_create_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_model_profile_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_model_profile_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_model_profile_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_model_initializer_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_model_initializer_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_model_initializer_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_input_create_cabi_v1(void);
void et_e1b_public_m3t_input_create_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_input_create_cabi_v1();
}

extern eshkol_tagged_value_t et_e1b_private_m3t_input_copy_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_input_copy_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_input_copy_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_input_copy_tokenizer_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_input_copy_tokenizer_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_input_copy_tokenizer_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_input_release_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_input_release_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_input_release_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_logits_create_cabi_v1(void);
void et_e1b_public_m3t_logits_create_v1(void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_logits_create_cabi_v1();
}

extern eshkol_tagged_value_t et_e1b_private_m3t_logits_copy_bits_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_logits_copy_bits_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_logits_copy_bits_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_logits_bits_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_logits_bits_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_logits_bits_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_logits_release_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_logits_release_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_logits_release_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_create_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_workspace_create_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_create_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_begin_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_workspace_begin_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_begin_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_vjp_begin_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_workspace_vjp_begin_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_vjp_begin_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_copy_logits_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_workspace_copy_logits_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_copy_logits_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_check_primals_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_workspace_check_primals_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_check_primals_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_reset_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_workspace_reset_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_reset_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_release_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_workspace_release_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_workspace_release_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_embedding_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_embedding_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_embedding_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_linear_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_linear_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_linear_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_layer_norm_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_layer_norm_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_layer_norm_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_gelu_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_gelu_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_gelu_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_residual_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_residual_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_residual_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_heads_split_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_heads_split_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_heads_split_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_heads_merge_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_heads_merge_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_heads_merge_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_attention_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_attention_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_attention_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_embedding_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_embedding_vjp_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_embedding_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_linear_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_linear_vjp_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_linear_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_layer_norm_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_layer_norm_vjp_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_layer_norm_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_gelu_vjp_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_gelu_vjp_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_gelu_vjp_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_residual_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_residual_vjp_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_residual_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_heads_split_vjp_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_heads_split_vjp_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_heads_split_vjp_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_heads_merge_vjp_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_heads_merge_vjp_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_heads_merge_vjp_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_attention_vjp_cabi_v1(eshkol_tagged_value_t first);
void et_e1b_public_m3t_attention_vjp_v1(void *first, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_attention_vjp_cabi_v1(*et_e1b_box_value_v1(first));
}

extern eshkol_tagged_value_t et_e1b_private_m3t_sum_cabi_v1(eshkol_tagged_value_t first, eshkol_tagged_value_t second);
void et_e1b_public_m3t_sum_v1(void *first, void *second, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_m3t_sum_cabi_v1(*et_e1b_box_value_v1(first), *et_e1b_box_value_v1(second));
}
