#include "e1b_error_consumer_bridge.h"

/* The trusted identity implementation is part of the prelocalized package,
 * never a library supplied by an application after localization. */
#define ET_P1_TRUSTED_BUILD 1
#include "p1_identity.c"

#define P1_DECLARE_UNARY(name)                                               \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define P1_DECLARE_BINARY(name)                                              \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,              \
                                     eshkol_tagged_value_t right)

P1_DECLARE_UNARY(et_e1b_private_p1_module_parameters_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_module_buffers_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_module_state_dict_cabi_v1);
P1_DECLARE_BINARY(et_e1b_private_p1_module_load_state_dict_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_module_train_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_module_eval_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_module_zero_grad_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_tree_paths_cabi_v1);
P1_DECLARE_BINARY(et_e1b_private_p1_parameter_tree_handle_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_tree_tie_groups_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_handle_path_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_handle_shape_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_handle_dtype_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_parameter_handle_device_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_state_dict_paths_cabi_v1);
P1_DECLARE_BINARY(et_e1b_private_p1_state_dict_tensor_cabi_v1);
P1_DECLARE_UNARY(et_e1b_private_p1_state_dict_alias_groups_cabi_v1);

#define P1_PUBLIC_UNARY(name, target)                                        \
  void name(void *input, void *output) {                                     \
    eshkol_tagged_value_t *const input_value = et_e1b_box_value_v1(input);    \
    eshkol_tagged_value_t *const output_value = et_e1b_box_value_v1(output);  \
    et_e1b_ensure_private_initialized_v1();                                  \
    *output_value = target(*input_value);                                    \
  }

#define P1_PUBLIC_BINARY(name, target)                                       \
  void name(void *left, void *right, void *output) {                         \
    eshkol_tagged_value_t *const left_value = et_e1b_box_value_v1(left);      \
    eshkol_tagged_value_t *const right_value = et_e1b_box_value_v1(right);    \
    eshkol_tagged_value_t *const output_value = et_e1b_box_value_v1(output);  \
    et_e1b_ensure_private_initialized_v1();                                  \
    *output_value = target(*left_value, *right_value);                       \
  }

P1_PUBLIC_UNARY(et_e1b_public_p1_module_parameters_v1,
                et_e1b_private_p1_module_parameters_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_module_buffers_v1,
                et_e1b_private_p1_module_buffers_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_module_state_dict_v1,
                et_e1b_private_p1_module_state_dict_cabi_v1)
P1_PUBLIC_BINARY(et_e1b_public_p1_module_load_state_dict_v1,
                 et_e1b_private_p1_module_load_state_dict_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_module_train_v1,
                et_e1b_private_p1_module_train_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_module_eval_v1,
                et_e1b_private_p1_module_eval_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_module_zero_grad_v1,
                et_e1b_private_p1_module_zero_grad_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_tree_paths_v1,
                et_e1b_private_p1_parameter_tree_paths_cabi_v1)
P1_PUBLIC_BINARY(et_e1b_public_p1_parameter_tree_handle_v1,
                 et_e1b_private_p1_parameter_tree_handle_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_tree_tie_groups_v1,
                et_e1b_private_p1_parameter_tree_tie_groups_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_path_v1,
                et_e1b_private_p1_parameter_handle_path_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_shape_v1,
                et_e1b_private_p1_parameter_handle_shape_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_dtype_v1,
                et_e1b_private_p1_parameter_handle_dtype_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_parameter_handle_device_v1,
                et_e1b_private_p1_parameter_handle_device_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_state_dict_paths_v1,
                et_e1b_private_p1_state_dict_paths_cabi_v1)
P1_PUBLIC_BINARY(et_e1b_public_p1_state_dict_tensor_v1,
                 et_e1b_private_p1_state_dict_tensor_cabi_v1)
P1_PUBLIC_UNARY(et_e1b_public_p1_state_dict_alias_groups_v1,
                et_e1b_private_p1_state_dict_alias_groups_cabi_v1)
