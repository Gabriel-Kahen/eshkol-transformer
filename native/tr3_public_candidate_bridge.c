/* Candidate boxed facade. All five receivers must come from this object. */
#include "e1b_error_consumer_bridge.h"

extern eshkol_tagged_value_t et_e1b_private_x1_config_parse_cabi_v1(
    eshkol_tagged_value_t text);
extern eshkol_tagged_value_t et_e1b_private_x1_config_resolve_cabi_v1(
    eshkol_tagged_value_t config, eshkol_tagged_value_t overrides);
extern eshkol_tagged_value_t et_e1b_private_t1_tokenizer_byte_cabi_v1(
    eshkol_tagged_value_t config);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_open_cabi_v1(
    eshkol_tagged_value_t options, eshkol_tagged_value_t tokenizer);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_close_cabi_v1(
    eshkol_tagged_value_t dataset);

extern eshkol_tagged_value_t et_e1b_private_m3t_initializer_state_cabi_v1(
    eshkol_tagged_value_t seed);
extern eshkol_tagged_value_t et_e1b_private_m3t_model_create_cabi_v1(
    eshkol_tagged_value_t config, eshkol_tagged_value_t initializer);
extern eshkol_tagged_value_t et_e1b_private_tr3_trainer_create_cabi_v1(
    eshkol_tagged_value_t resolved, eshkol_tagged_value_t tokenizer,
    eshkol_tagged_value_t dataset, eshkol_tagged_value_t model,
    eshkol_tagged_value_t optimizer);
extern eshkol_tagged_value_t et_e1b_private_p1_module_parameters_cabi_v1(
    eshkol_tagged_value_t model);
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_tree_paths_cabi_v1(
    eshkol_tagged_value_t tree);
extern eshkol_tagged_value_t et_e1b_private_p1_parameter_tree_handle_cabi_v1(
    eshkol_tagged_value_t tree, eshkol_tagged_value_t path);
extern eshkol_tagged_value_t et_e1b_private_o2_optimizer_create_cabi_v1(
    eshkol_tagged_value_t config, eshkol_tagged_value_t parameters);

#define TR3_PUBLIC_UNARY(name, target)                                      \
  void name(void *input, void *output) {                                     \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define TR3_PUBLIC_BINARY(name, target)                                     \
  void name(void *left, void *right, void *output) {                         \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(                                   \
        *et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));            \
  }

TR3_PUBLIC_UNARY(et_e1b_public_x1_config_parse_v1,
                 et_e1b_private_x1_config_parse_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_x1_config_resolve_v1,
                  et_e1b_private_x1_config_resolve_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_t1_tokenizer_byte_v1,
                 et_e1b_private_t1_tokenizer_byte_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_d2_token_dataset_open_v1,
                  et_e1b_private_d2_token_dataset_open_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_close_v1,
                 et_e1b_private_d2_token_dataset_close_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_p1_module_parameters_v1,
                 et_e1b_private_p1_module_parameters_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_p1_parameter_tree_paths_v1,
                 et_e1b_private_p1_parameter_tree_paths_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_p1_parameter_tree_handle_v1,
                  et_e1b_private_p1_parameter_tree_handle_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_o2_optimizer_create_v1,
                  et_e1b_private_o2_optimizer_create_cabi_v1)

TR3_PUBLIC_UNARY(et_e1b_public_m3t_initializer_state_v1,
                 et_e1b_private_m3t_initializer_state_cabi_v1)
TR3_PUBLIC_BINARY(et_e1b_public_m3t_model_create_v1,
                  et_e1b_private_m3t_model_create_cabi_v1)

void et_e1b_public_tr3_trainer_create_v1(void *resolved, void *tokenizer,
                                           void *dataset, void *model,
                                           void *optimizer, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_tr3_trainer_create_cabi_v1(
      *et_e1b_box_value_v1(resolved), *et_e1b_box_value_v1(tokenizer),
      *et_e1b_box_value_v1(dataset), *et_e1b_box_value_v1(model),
      *et_e1b_box_value_v1(optimizer));
}
