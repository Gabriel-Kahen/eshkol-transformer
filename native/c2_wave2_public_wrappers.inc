/* Reusable C2 wrappers. The including aggregate supplies E1B box helpers and
 * the six renamed C2 Eshkol procedures. Persistence policy keeps the existing
 * Wave 1 ABI symbol and is routed by ET_E1B_PERSISTENCE_POLICY_PRIVATE_TARGET. */
#define ET_C2_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_C2_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t first,            \
                                     eshkol_tagged_value_t second)
#define ET_C2_DECLARE_TERNARY(name)                                         \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t first,            \
                                     eshkol_tagged_value_t second,           \
                                     eshkol_tagged_value_t third)

ET_C2_DECLARE_BINARY(et_e1b_private_c2_checkpoint_inspect_cabi_v1);
ET_C2_DECLARE_BINARY(et_e1b_private_c2_checkpoint_metadata_ref_cabi_v1);
ET_C2_DECLARE_TERNARY(et_e1b_private_c2_checkpoint_load_cabi_v1);
extern eshkol_tagged_value_t et_e1b_private_c2_checkpoint_save_cabi_v1(
    eshkol_tagged_value_t state, eshkol_tagged_value_t path,
    eshkol_tagged_value_t policy, eshkol_tagged_value_t options);
ET_C2_DECLARE_UNARY(et_e1b_private_c2_trainer_state_release_cabi_v1);

#define ET_C2_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_C2_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

ET_C2_PUBLIC_BINARY(et_e1b_public_c2_checkpoint_inspect_v1,
                     et_e1b_private_c2_checkpoint_inspect_cabi_v1)
ET_C2_PUBLIC_BINARY(et_e1b_public_c2_checkpoint_metadata_ref_v1,
                     et_e1b_private_c2_checkpoint_metadata_ref_cabi_v1)

void et_e1b_public_c2_checkpoint_load_v1(void *path, void *policy,
                                         void *report, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_c2_checkpoint_load_cabi_v1(
      *et_e1b_box_value_v1(path), *et_e1b_box_value_v1(policy),
      *et_e1b_box_value_v1(report));
}

void et_e1b_public_c2_checkpoint_save_v1(void *state, void *path,
                                         void *policy, void *options,
                                         void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_c2_checkpoint_save_cabi_v1(
      *et_e1b_box_value_v1(state), *et_e1b_box_value_v1(path),
      *et_e1b_box_value_v1(policy), *et_e1b_box_value_v1(options));
}

ET_C2_PUBLIC_UNARY(et_e1b_public_c2_trainer_state_release_v1,
                    et_e1b_private_c2_trainer_state_release_cabi_v1)

#undef ET_C2_PUBLIC_BINARY
#undef ET_C2_PUBLIC_UNARY
#undef ET_C2_DECLARE_TERNARY
#undef ET_C2_DECLARE_BINARY
#undef ET_C2_DECLARE_UNARY
