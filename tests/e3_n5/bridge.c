/* One noninstalled boxed N5 entry over the unchanged private E3 bridge. */
#include "../../native/e3_private_bridge.c"

extern eshkol_tagged_value_t
et_e3_private_n5_dispatch_cabi_v1(eshkol_tagged_value_t seed_path,
                                   eshkol_tagged_value_t control_path,
                                   eshkol_tagged_value_t cap_path);
extern eshkol_tagged_value_t
et_e3_private_n5_edge_dispatch_cabi_v1(eshkol_tagged_value_t one_path,
                                        eshkol_tagged_value_t empty_path,
                                        eshkol_tagged_value_t singleton_path,
                                        eshkol_tagged_value_t eos_path);

void et_e3_n5_run_v1(void *seed_path, void *control_path,
                     void *cap_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_n5_dispatch_cabi_v1(
      *et_e1b_box_value_v1(seed_path), *et_e1b_box_value_v1(control_path),
      *et_e1b_box_value_v1(cap_path));
}

void et_e3_n5_edge_run_v1(void *one_path, void *empty_path,
                          void *singleton_path, void *eos_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_n5_edge_dispatch_cabi_v1(
      *et_e1b_box_value_v1(one_path), *et_e1b_box_value_v1(empty_path),
      *et_e1b_box_value_v1(singleton_path), *et_e1b_box_value_v1(eos_path));
}
