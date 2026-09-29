/* Installed trainer facet extends the reviewed same-aggregate release bridge. */
#include "tr3_public_release_candidate_bridge.c"
#include "d1_e1b_package_bridge.c"

extern eshkol_tagged_value_t et_e1b_private_c2_trainer_state_release_cabi_v1(
    eshkol_tagged_value_t state);
extern eshkol_tagged_value_t et_e1b_private_x1_config_validate_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_canonical_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_fingerprint_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_ref_cabi_v1(
    eshkol_tagged_value_t resolved, eshkol_tagged_value_t key);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_load_cabi_v1(
    eshkol_tagged_value_t path, eshkol_tagged_value_t policy);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_save_cabi_v1(
    eshkol_tagged_value_t tokenizer, eshkol_tagged_value_t path,
    eshkol_tagged_value_t policy);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_encode_cabi_v1(
    eshkol_tagged_value_t tokenizer, eshkol_tagged_value_t input);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_decode_cabi_v1(
    eshkol_tagged_value_t tokenizer, eshkol_tagged_value_t ids);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_vocab_size_cabi_v1(
    eshkol_tagged_value_t tokenizer);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_fingerprint_cabi_v1(
    eshkol_tagged_value_t tokenizer);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_special_token_id_cabi_v1(
    eshkol_tagged_value_t tokenizer, eshkol_tagged_value_t name);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_cursor_cabi_v1(
    eshkol_tagged_value_t dataset);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_end_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_seek_cabi_v1(
    eshkol_tagged_value_t dataset, eshkol_tagged_value_t cursor);
extern eshkol_tagged_value_t et_e1b_private_d2_token_batch_inputs_cabi_v1(
    eshkol_tagged_value_t batch);
extern eshkol_tagged_value_t et_e1b_private_d2_token_batch_targets_cabi_v1(
    eshkol_tagged_value_t batch);
extern eshkol_tagged_value_t et_e1b_private_d2_token_batch_loss_mask_cabi_v1(
    eshkol_tagged_value_t batch);
extern eshkol_tagged_value_t et_e1b_private_d2_token_batch_validate_cabi_v1(
    eshkol_tagged_value_t batch);
extern eshkol_tagged_value_t et_e1b_private_tr3_metrics_ref_cabi_v1(
    eshkol_tagged_value_t metrics, eshkol_tagged_value_t key);
extern eshkol_tagged_value_t et_e1b_private_tr3_trainer_step_cabi_v1(
    eshkol_tagged_value_t trainer);
extern eshkol_tagged_value_t et_e1b_private_tr3_trainer_stop_policy_cabi_v1(
    eshkol_tagged_value_t max_tokens, eshkol_tagged_value_t max_updates,
    eshkol_tagged_value_t max_epochs);
extern eshkol_tagged_value_t et_e1b_private_tr3_trainer_train_cabi_v1(
    eshkol_tagged_value_t trainer, eshkol_tagged_value_t policy);

#if !defined(ESHKOL_HAS_F32_SCALAR_ABI_V1) || ESHKOL_HAS_F32_SCALAR_ABI_V1 != 1
#error "public metrics-ref requires the accepted true-f32 runtime ABI"
#endif

void et_e1b_public_tr3_metrics_ref_v1(void *metrics, void *key, void *output) {
  eshkol_tagged_value_t result;
  int64_t code;
  et_e1b_ensure_private_initialized_v1();
  result = et_e1b_private_tr3_metrics_ref_cabi_v1(
      *et_e1b_box_value_v1(metrics), *et_e1b_box_value_v1(key));
  if (result.type != ESHKOL_VALUE_INT64 ||
      result.flags != ESHKOL_VALUE_EXACT_FLAG || result.reserved != 0)
    __builtin_trap();
  code = eshkol_unpack_int64(&result);
  if (code < 0) {
    if (code < -INT64_C(4294967296) ||
        eshkol_value_f32_from_bits_v1(&result, (uint32_t)(-code - 1)) !=
            ESHKOL_VALUE_F32_OK)
      __builtin_trap();
  }
  *et_e1b_box_value_v1(output) = result;
}

TR3_PUBLIC_UNARY(et_e1b_public_c2_trainer_state_release_v1,
                 et_e1b_private_c2_trainer_state_release_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_tr3_trainer_step_v1,
                 et_e1b_private_tr3_trainer_step_cabi_v1)
void et_e1b_public_tr3_trainer_stop_policy_v1(
    void *max_tokens, void *max_updates, void *max_epochs, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_tr3_trainer_stop_policy_cabi_v1(
      *et_e1b_box_value_v1(max_tokens), *et_e1b_box_value_v1(max_updates),
      *et_e1b_box_value_v1(max_epochs));
}
TR3_PUBLIC_BINARY(et_e1b_public_tr3_trainer_train_v1,
                  et_e1b_private_tr3_trainer_train_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_x1_config_validate_v1,
                 et_e1b_private_x1_config_validate_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_x1_config_canonical_v1,
                 et_e1b_private_x1_config_canonical_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_x1_config_fingerprint_v1,
                 et_e1b_private_x1_config_fingerprint_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_x1_config_ref_v1,
                  et_e1b_private_x1_config_ref_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_load_v1,
                  et_e1b_private_t1_tokenizer_load_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_encode_v1,
                  et_e1b_private_t1_tokenizer_encode_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_decode_v1,
                  et_e1b_private_t1_tokenizer_decode_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_vocab_size_v1,
                 et_e1b_private_t1_tokenizer_vocab_size_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_fingerprint_v1,
                 et_e1b_private_t1_tokenizer_fingerprint_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_t1_tokenizer_special_token_id_v1,
                  et_e1b_private_t1_tokenizer_special_token_id_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_cursor_v1,
                 et_e1b_private_d2_token_dataset_cursor_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_end_v1,
                 et_e1b_private_d2_token_dataset_end_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_d2_token_dataset_seek_v1,
                  et_e1b_private_d2_token_dataset_seek_cabi_v1)

/* The canonical D2 facade proves procedure? before passing raw callables. */
#define TR3_PUBLIC_D2_CALLABLE_UNARY(name, target)                          \
  void name(void *callable, void *output) {                                  \
    eshkol_tagged_value_t input;                                             \
    et_e1b_ensure_private_initialized_v1();                                 \
    input = eshkol_make_ptr((uint64_t)(uintptr_t)callable,                   \
                            ESHKOL_VALUE_CALLABLE);                          \
    *et_e1b_box_value_v1(output) = target(input);                            \
  }

TR3_PUBLIC_D2_CALLABLE_UNARY(et_e1b_public_d2_token_batch_inputs_v1,
                             et_e1b_private_d2_token_batch_inputs_cabi_v1)
TR3_PUBLIC_D2_CALLABLE_UNARY(et_e1b_public_d2_token_batch_targets_v1,
                             et_e1b_private_d2_token_batch_targets_cabi_v1)
TR3_PUBLIC_D2_CALLABLE_UNARY(et_e1b_public_d2_token_batch_loss_mask_v1,
                             et_e1b_private_d2_token_batch_loss_mask_cabi_v1)

void et_e1b_public_d2_token_batch_validate_v1(void *input, int64_t operation,
                                               void *output) {
  eshkol_tagged_value_t value;
  et_e1b_ensure_private_initialized_v1();
  value = *et_e1b_box_value_v1(input);
  switch (operation) {
  case 0:
    value = et_e1b_private_d2_token_batch_validate_cabi_v1(value);
    break;
  case 1:
    value = et_e1b_private_d2_token_batch_inputs_cabi_v1(value);
    break;
  case 2:
    value = et_e1b_private_d2_token_batch_targets_cabi_v1(value);
    break;
  case 3:
    value = et_e1b_private_d2_token_batch_loss_mask_cabi_v1(value);
    break;
  case 4:
    value = et_e1b_private_d2_token_batch_release_cabi_v1(value);
    break;
  default:
    __builtin_trap();
  }
  *et_e1b_box_value_v1(output) = value;
}

void et_e1b_public_t1_tokenizer_save_v1(void *tokenizer, void *path,
                                        void *policy, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_t1_tokenizer_save_cabi_v1(
      *et_e1b_box_value_v1(tokenizer), *et_e1b_box_value_v1(path),
      *et_e1b_box_value_v1(policy));
}

/* Complete unchanged diagnostic transport facade, in this registry. */
#define TR3_PUBLIC_NULLARY(name, target) \
  void name(void *output) { \
    et_e1b_ensure_private_initialized_v1(); \
    *et_e1b_box_value_v1(output) = target(); \
  }

extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_algorithm_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_initializer_algorithm_v1,
                  et_e1b_private_m3t_initializer_algorithm_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_words_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_initializer_words_v1,
                  et_e1b_private_m3t_initializer_words_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_model_profile_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_model_profile_v1,
                  et_e1b_private_m3t_model_profile_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_model_initializer_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_model_initializer_v1,
                  et_e1b_private_m3t_model_initializer_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_input_create_cabi_v1(void);
TR3_PUBLIC_NULLARY(et_e1b_public_m3t_input_create_v1,
                  et_e1b_private_m3t_input_create_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_input_copy_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_input_copy_v1,
                  et_e1b_private_m3t_input_copy_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_input_copy_tokenizer_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_input_copy_tokenizer_v1,
                  et_e1b_private_m3t_input_copy_tokenizer_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_input_release_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_input_release_v1,
                  et_e1b_private_m3t_input_release_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_logits_create_cabi_v1(void);
TR3_PUBLIC_NULLARY(et_e1b_public_m3t_logits_create_v1,
                  et_e1b_private_m3t_logits_create_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_logits_copy_bits_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_logits_copy_bits_v1,
                  et_e1b_private_m3t_logits_copy_bits_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_logits_bits_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_logits_bits_v1,
                  et_e1b_private_m3t_logits_bits_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_logits_release_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_logits_release_v1,
                  et_e1b_private_m3t_logits_release_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_create_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_workspace_create_v1,
                  et_e1b_private_m3t_workspace_create_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_begin_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_workspace_begin_v1,
                  et_e1b_private_m3t_workspace_begin_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_vjp_begin_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_workspace_vjp_begin_v1,
                  et_e1b_private_m3t_workspace_vjp_begin_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_copy_logits_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_workspace_copy_logits_v1,
                  et_e1b_private_m3t_workspace_copy_logits_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_check_primals_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_workspace_check_primals_v1,
                  et_e1b_private_m3t_workspace_check_primals_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_reset_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_workspace_reset_v1,
                  et_e1b_private_m3t_workspace_reset_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_workspace_release_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_workspace_release_v1,
                  et_e1b_private_m3t_workspace_release_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_embedding_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_embedding_v1,
                  et_e1b_private_m3t_embedding_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_linear_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_linear_v1,
                  et_e1b_private_m3t_linear_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_layer_norm_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_layer_norm_v1,
                  et_e1b_private_m3t_layer_norm_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_gelu_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_gelu_v1,
                  et_e1b_private_m3t_gelu_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_residual_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_residual_v1,
                  et_e1b_private_m3t_residual_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_heads_split_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_heads_split_v1,
                  et_e1b_private_m3t_heads_split_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_heads_merge_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_heads_merge_v1,
                  et_e1b_private_m3t_heads_merge_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_attention_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_attention_v1,
                  et_e1b_private_m3t_attention_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_embedding_vjp_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_embedding_vjp_v1,
                  et_e1b_private_m3t_embedding_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_linear_vjp_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_linear_vjp_v1,
                  et_e1b_private_m3t_linear_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_layer_norm_vjp_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_layer_norm_vjp_v1,
                  et_e1b_private_m3t_layer_norm_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_gelu_vjp_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_gelu_vjp_v1,
                  et_e1b_private_m3t_gelu_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_residual_vjp_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_residual_vjp_v1,
                  et_e1b_private_m3t_residual_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_heads_split_vjp_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_heads_split_vjp_v1,
                  et_e1b_private_m3t_heads_split_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_heads_merge_vjp_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_heads_merge_vjp_v1,
                  et_e1b_private_m3t_heads_merge_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_attention_vjp_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_m3t_attention_vjp_v1,
                  et_e1b_private_m3t_attention_vjp_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_m3t_sum_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_m3t_sum_v1,
                  et_e1b_private_m3t_sum_cabi_v1)
#undef TR3_PUBLIC_NULLARY

/* Complete unchanged P1 module and O2 optimizer facades. */
extern eshkol_tagged_value_t et_e1b_private_p1_module_buffers_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_buffers_v1,
                  et_e1b_private_p1_module_buffers_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_module_state_dict_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_state_dict_v1,
                  et_e1b_private_p1_module_state_dict_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_module_load_state_dict_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_p1_module_load_state_dict_v1,
                  et_e1b_private_p1_module_load_state_dict_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_module_train_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_train_v1,
                  et_e1b_private_p1_module_train_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_module_eval_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_eval_v1,
                  et_e1b_private_p1_module_eval_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_module_zero_grad_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_zero_grad_v1,
                  et_e1b_private_p1_module_zero_grad_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_tree_tie_groups_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_tree_tie_groups_v1,
                  et_e1b_private_p1_parameter_tree_tie_groups_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_handle_path_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_path_v1,
                  et_e1b_private_p1_parameter_handle_path_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_handle_shape_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_shape_v1,
                  et_e1b_private_p1_parameter_handle_shape_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_handle_dtype_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_dtype_v1,
                  et_e1b_private_p1_parameter_handle_dtype_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_handle_device_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_device_v1,
                  et_e1b_private_p1_parameter_handle_device_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_state_dict_paths_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_state_dict_paths_v1,
                  et_e1b_private_p1_state_dict_paths_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_state_dict_tensor_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_p1_state_dict_tensor_v1,
                  et_e1b_private_p1_state_dict_tensor_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_state_dict_alias_groups_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_state_dict_alias_groups_v1,
                  et_e1b_private_p1_state_dict_alias_groups_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_p1_state_dict_release_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_p1_state_dict_release_v1,
                  et_e1b_private_p1_state_dict_release_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_step_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_o2_optimizer_step_v1,
                  et_e1b_private_o2_optimizer_step_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_zero_grad_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_o2_optimizer_zero_grad_v1,
                  et_e1b_private_o2_optimizer_zero_grad_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_state_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_v1,
                  et_e1b_private_o2_optimizer_state_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_load_state_cabi_v1(eshkol_tagged_value_t, eshkol_tagged_value_t);
TR3_PUBLIC_BINARY(et_e1b_public_o2_optimizer_load_state_v1,
                  et_e1b_private_o2_optimizer_load_state_cabi_v1)
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_state_release_cabi_v1(eshkol_tagged_value_t);
TR3_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_release_v1,
                  et_e1b_private_o2_optimizer_state_release_cabi_v1)
