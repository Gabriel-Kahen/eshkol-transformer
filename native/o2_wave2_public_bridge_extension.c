/*
 * Reusable O2 public-wrapper suffix.  Its source-composed caller supplies the
 * E1B box helpers and the six renamed Eshkol implementation procedures.
 */
#define ET_O2_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_O2_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,             \
                                     eshkol_tagged_value_t right)

ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_create_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_step_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_zero_grad_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_cabi_v1);
ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_load_state_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_release_cabi_v1);

#define ET_O2_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_O2_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_create_v1,
                    et_e1b_private_o2_optimizer_create_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_step_v1,
                   et_e1b_private_o2_optimizer_step_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_zero_grad_v1,
                   et_e1b_private_o2_optimizer_zero_grad_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_v1,
                   et_e1b_private_o2_optimizer_state_cabi_v1)
ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_load_state_v1,
                    et_e1b_private_o2_optimizer_load_state_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_release_v1,
                   et_e1b_private_o2_optimizer_state_release_cabi_v1)

#undef ET_O2_PUBLIC_BINARY
#undef ET_O2_PUBLIC_UNARY
#undef ET_O2_DECLARE_BINARY
#undef ET_O2_DECLARE_UNARY
