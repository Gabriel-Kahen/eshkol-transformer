/* Installed trainer facet extends the reviewed same-aggregate release bridge. */
#include "tr3_public_release_candidate_bridge.c"

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
