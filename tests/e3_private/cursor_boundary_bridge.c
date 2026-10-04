/* Exact noninstalled successor: retain the reviewed retention observers. */
#include "retention_bridge.c"

extern eshkol_tagged_value_t et_e3_private_cursor_boundary_dispatch_cabi_v1(
    eshkol_tagged_value_t paths);

void et_e3_cursor_boundary_run_v1(void *paths, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) =
      et_e3_private_cursor_boundary_dispatch_cabi_v1(*et_e1b_box_value_v1(paths));
}
