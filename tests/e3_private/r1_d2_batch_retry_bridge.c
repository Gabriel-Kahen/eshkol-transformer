/* Separate noninstalled E3 R1 D2 batch retry entry. */
#include "r1_source_retry_bridge.c"

extern eshkol_tagged_value_t
et_e3_private_r1_d2_batch_retry_run_cabi_v1(eshkol_tagged_value_t corpus_path);

void et_e3_r1_d2_batch_retry_run_v1(void *corpus_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_r1_d2_batch_retry_run_cabi_v1(
      *et_e1b_box_value_v1(corpus_path));
}
