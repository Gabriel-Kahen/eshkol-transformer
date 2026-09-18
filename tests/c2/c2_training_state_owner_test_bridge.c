#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>
#include <string.h>

static et_o2_optimizer_state_borrow *et_c2_test_o2_borrow;

int64_t et_c2_test_o2_borrow_begin_v1(et_o2_optimizer_state *state) {
  et_o2_optimizer_state_handle *handle = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_o2_error_v1 error;
  int32_t status;

  if (et_c2_test_o2_borrow != NULL) return -1;
  memset(&error, 0, sizeof(error));
  status = et_o2_optimizer_state_moment_handle_v1(
      state, 0u, ET_O2_MOMENT_EXP_AVG, &handle, &error);
  if (status != 0) return status;
  return et_o2_optimizer_state_borrow_begin_v1(
      state, handle, &et_c2_test_o2_borrow, &view, &error);
}

int64_t et_c2_test_o2_borrow_end_v1(void) {
  et_o2_error_v1 error;
  memset(&error, 0, sizeof(error));
  return et_o2_optimizer_state_borrow_end_v1(&et_c2_test_o2_borrow, &error);
}

int64_t et_c2_test_i2_live_count_v1(int64_t index) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  switch (index) {
  case 0: return (int64_t)counts.tensors;
  case 1: return (int64_t)counts.parameters;
  case 2: return (int64_t)counts.borrows;
  case 3: return (int64_t)counts.copy_plans;
  case 4: return (int64_t)counts.gradient_plans;
  case 5: return (int64_t)counts.reset_plans;
  case 6: return (int64_t)counts.owned_clones;
  default: return -1;
  }
}

int64_t et_c2_test_o2_count_v1(int64_t index) {
  et_o2_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_o2_test_live_counts_snapshot_v1(&counts);
  switch (index) {
  case 0: return (int64_t)counts.builders;
  case 1: return (int64_t)counts.optimizers;
  case 2: return (int64_t)counts.live_states;
  case 3: return (int64_t)counts.dead_states;
  case 4: return (int64_t)counts.live_state_handles;
  case 5: return (int64_t)counts.dead_state_handles;
  case 6: return (int64_t)counts.state_borrows;
  case 7: return (int64_t)counts.owned_state_clones;
  case 8: return (int64_t)counts.reconstruct_builders;
  default: return -1;
  }
}
