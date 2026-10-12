/* Candidate boxed test entry. This bridge is not admitted by package policy. */
#include "../../native/e3_private_bridge.c"

extern eshkol_tagged_value_t
et_e3_n3_zero_test_run_cabi_v1(eshkol_tagged_value_t scenario,
                                 eshkol_tagged_value_t corpus_path);

void et_e3_n3_zero_test_run_v1(void *scenario, void *corpus_path,
                                void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_n3_zero_test_run_cabi_v1(
      *et_e1b_box_value_v1(scenario), *et_e1b_box_value_v1(corpus_path));
}
