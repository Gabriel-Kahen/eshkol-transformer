/* Installed fixed-model E3 diagnostic wrappers. Trusted definitions stay local. */
#include "m3_package_bridge.c"
#include "e3_d2_public_wrappers.inc"

extern eshkol_tagged_value_t
et_e1b_private_e3_diagnostic_evaluate_fixed_cabi_v1(
    eshkol_tagged_value_t model, eshkol_tagged_value_t dataset);
extern eshkol_tagged_value_t
et_e1b_private_e3_diagnostic_evaluation_f32_bits_cabi_v1(
    eshkol_tagged_value_t report, eshkol_tagged_value_t key);
extern eshkol_tagged_value_t
et_e1b_private_e3_diagnostic_evaluation_count_cabi_v1(
    eshkol_tagged_value_t report, eshkol_tagged_value_t key);

#define ET_E3_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));    \
  }

ET_E3_PUBLIC_BINARY(et_e1b_public_e3_diagnostic_evaluate_fixed_v1,
                    et_e1b_private_e3_diagnostic_evaluate_fixed_cabi_v1)
ET_E3_PUBLIC_BINARY(et_e1b_public_e3_diagnostic_evaluation_f32_bits_v1,
                    et_e1b_private_e3_diagnostic_evaluation_f32_bits_cabi_v1)
ET_E3_PUBLIC_BINARY(et_e1b_public_e3_diagnostic_evaluation_count_v1,
                    et_e1b_private_e3_diagnostic_evaluation_count_cabi_v1)
