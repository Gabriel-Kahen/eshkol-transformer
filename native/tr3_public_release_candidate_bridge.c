/* Non-installed same-aggregate release candidate. */
#include "tr3_public_candidate_bridge.c"

extern eshkol_tagged_value_t et_e1b_private_tr3_trainer_release_cabi_v1(
    eshkol_tagged_value_t trainer);
extern eshkol_tagged_value_t et_e1b_private_d2_token_dataset_next_batch_cabi_v1(
    eshkol_tagged_value_t dataset);
extern eshkol_tagged_value_t et_e1b_private_d2_token_batch_release_cabi_v1(
    eshkol_tagged_value_t batch);

TR3_PUBLIC_UNARY(et_e1b_public_tr3_trainer_release_v1,
                 et_e1b_private_tr3_trainer_release_cabi_v1)
TR3_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_next_batch_v1,
                 et_e1b_private_d2_token_dataset_next_batch_cabi_v1)

/* An accepted D2 batch is a callable, passed by the facade as a raw pointer. */
void et_e1b_public_d2_token_batch_release_v1(void *batch, void *output) {
  eshkol_tagged_value_t input;
  et_e1b_ensure_private_initialized_v1();
  input = eshkol_make_ptr((uint64_t)(uintptr_t)batch, ESHKOL_VALUE_CALLABLE);
  *et_e1b_box_value_v1(output) =
      et_e1b_private_d2_token_batch_release_cabi_v1(input);
}
