#include "e1b_error_consumer_bridge.h"

extern eshkol_tagged_value_t et_e1b_private_x1_config_parse_cabi_v1(
    eshkol_tagged_value_t text);
extern eshkol_tagged_value_t et_e1b_private_x1_config_resolve_cabi_v1(
    eshkol_tagged_value_t config, eshkol_tagged_value_t overrides);
extern eshkol_tagged_value_t et_e1b_private_x1_config_validate_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_canonical_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_fingerprint_cabi_v1(
    eshkol_tagged_value_t resolved);
extern eshkol_tagged_value_t et_e1b_private_x1_config_ref_cabi_v1(
    eshkol_tagged_value_t resolved, eshkol_tagged_value_t key);

#define X1_UNARY_PUBLIC(name, target)                                      \
  void name(void *input, void *output) {                                   \
    et_e1b_ensure_private_initialized_v1();                                \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));     \
  }

X1_UNARY_PUBLIC(et_e1b_public_x1_config_parse_v1,
                et_e1b_private_x1_config_parse_cabi_v1)
X1_UNARY_PUBLIC(et_e1b_public_x1_config_validate_v1,
                et_e1b_private_x1_config_validate_cabi_v1)
X1_UNARY_PUBLIC(et_e1b_public_x1_config_canonical_v1,
                et_e1b_private_x1_config_canonical_cabi_v1)
X1_UNARY_PUBLIC(et_e1b_public_x1_config_fingerprint_v1,
                et_e1b_private_x1_config_fingerprint_cabi_v1)

void et_e1b_public_x1_config_resolve_v1(void *config, void *overrides,
                                        void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_x1_config_resolve_cabi_v1(
      *et_e1b_box_value_v1(config), *et_e1b_box_value_v1(overrides));
}

void et_e1b_public_x1_config_ref_v1(void *resolved, void *key, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e1b_private_x1_config_ref_cabi_v1(
      *et_e1b_box_value_v1(resolved), *et_e1b_box_value_v1(key));
}
