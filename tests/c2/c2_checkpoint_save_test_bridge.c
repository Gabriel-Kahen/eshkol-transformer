#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>

void et_i2_test_live_builder_counts_v1(size_t *, size_t *, size_t *);
void et_i2_test_retired_builder_counts_v1(size_t *, size_t *, size_t *,
                                          size_t *);
int64_t et_p1_test_live_entry_count_v1(void);
int64_t et_p1_test_tombstone_count_v1(void);

static int64_t count(size_t value) {
  return value > (size_t)INT64_MAX ? -1 : (int64_t)value;
}

int64_t et_c2_checkpoint_save_test_counter_v1(int64_t index) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_f32_test_borrow_event_counts_v1 events = {.struct_size = sizeof(events)};
  et_o2_test_live_counts_v1 o2 = {.struct_size = sizeof(o2)};
  size_t live_copy = 0u, live_reset = 0u, live_decode = 0u;
  size_t retired_copy = 0u, retired_reset = 0u, retired_decode = 0u;
  size_t retained = 0u;
  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&events);
  et_i2_test_live_builder_counts_v1(&live_copy, &live_reset, &live_decode);
  et_i2_test_retired_builder_counts_v1(&retired_copy, &retired_reset,
                                       &retired_decode, &retained);
  et_o2_test_live_counts_snapshot_v1(&o2);
  switch (index) {
  case 0: return count(live.tensors);
  case 1: return count(live.parameters);
  case 2: return count(live.borrows);
  case 3: return count(live.copy_plans);
  case 4: return count(live.gradient_plans);
  case 5: return count(live.reset_plans);
  case 6: return count(live.owned_clones);
  case 7: return count(retired.tensors);
  case 8: return count(retired.parameters);
  case 9: return count(retired.borrows);
  case 10: return count(retired.copy_plans);
  case 11: return count(retired.gradient_plans);
  case 12: return count(retired.reset_plans);
  case 13: return count(retired.retained_control_bytes);
  case 14: return count(events.begin_calls);
  case 15: return count(events.view_calls);
  case 16: return count(events.end_calls);
  case 17: return count(live_copy);
  case 18: return count(live_reset);
  case 19: return count(live_decode);
  case 20: return count(retired_copy);
  case 21: return count(retired_reset);
  case 22: return count(retired_decode);
  case 23: return count(retained);
  case 24: return count(o2.builders);
  case 25: return count(o2.optimizers);
  case 26: return count(o2.live_states);
  case 27: return count(o2.dead_states);
  case 28: return count(o2.live_state_handles);
  case 29: return count(o2.dead_state_handles);
  case 30: return count(o2.state_borrows);
  case 31: return count(o2.owned_state_clones);
  case 32: return count(o2.reconstruct_builders);
  case 33: return et_p1_test_live_entry_count_v1();
  case 34: return et_p1_test_tombstone_count_v1();
  default: return -1;
  }
}
