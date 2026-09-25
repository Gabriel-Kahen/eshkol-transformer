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

TR3_PUBLIC_UNARY(et_e1b_public_c2_trainer_state_release_v1,
                 et_e1b_private_c2_trainer_state_release_cabi_v1)
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
