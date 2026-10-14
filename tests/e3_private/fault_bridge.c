/* One closed fault entry over the retention test bridge. */
#include "retention_bridge.c"
#include "../../native/f32_parameter_internal.h"
#include "../../src/eshkol_transformer/m3t_transport.h"

/* This closed test tuple holds a real model value borrow across one call.
 * The owner and parameter are admitted by M3T/I2; no evaluator hook fires. */
static void *e3_fault_borrow_owner;
static et_f32_parameter *e3_fault_borrow_parameter;
static int64_t e3_fault_borrow_index = -1;
static et_f32_tensor_borrow *e3_fault_borrow;

int64_t et_e3_fault_value_borrow_begin_v1(void *owner, int64_t index) {
  if (e3_fault_borrow || e3_fault_borrow_owner || index < 0 || index >= 14)
    return -1;
  et_f32_parameter *parameter = et_m3t_private_owner_parameter_v1(owner, index);
  if (!parameter) return -1;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (et_f32_parameter_value_borrow_begin_v1(
          parameter, &e3_fault_borrow, &error) != 0)
    return -1;
  e3_fault_borrow_owner = owner;
  e3_fault_borrow_parameter = parameter;
  e3_fault_borrow_index = index;
  return 0;
}

int64_t et_e3_fault_value_borrow_end_v1(void *owner, int64_t index) {
  if (owner != e3_fault_borrow_owner || index != e3_fault_borrow_index ||
      !e3_fault_borrow ||
      et_m3t_private_owner_parameter_v1(owner, index) !=
          e3_fault_borrow_parameter)
    return -1;
  et_f32_tensor_error error;
  et_f32_tensor_error_clear_v1(&error);
  if (et_f32_tensor_borrow_end_v1(&e3_fault_borrow, &error) != 0) return -1;
  e3_fault_borrow_owner = NULL;
  e3_fault_borrow_parameter = NULL;
  e3_fault_borrow_index = -1;
  return 0;
}

extern eshkol_tagged_value_t
et_e3_private_fault_dispatch_cabi_v1(eshkol_tagged_value_t scenario,
                                      eshkol_tagged_value_t corpus_path);

void et_e3_fault_run_v1(void *scenario, void *corpus_path, void *output) {
  et_e1b_ensure_private_initialized_v1();
  *et_e1b_box_value_v1(output) = et_e3_private_fault_dispatch_cabi_v1(
      *et_e1b_box_value_v1(scenario), *et_e1b_box_value_v1(corpus_path));
}
