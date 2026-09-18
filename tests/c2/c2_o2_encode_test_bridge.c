#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>

void et_i2_test_live_builder_counts_v1(size_t *copy_count,
                                       size_t *reset_count,
                                       size_t *decode_count);
void et_i2_test_retired_builder_counts_v1(size_t *copy_count,
                                          size_t *reset_count,
                                          size_t *decode_count,
                                          size_t *retained_control_bytes);

static int64_t checked_count(size_t value) {
  return value > (size_t)INT64_MAX ? -1 : (int64_t)value;
}

int64_t et_c2_o2_encode_test_counter_v1(int64_t index) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_f32_test_borrow_event_counts_v1 events = {
      .struct_size = sizeof(events),
  };
  et_o2_test_live_counts_v1 o2 = {.struct_size = sizeof(o2)};
  size_t live_copy = 0u, live_reset = 0u, live_decode = 0u;
  size_t dead_copy = 0u, dead_reset = 0u, dead_decode = 0u, retained = 0u;

  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&events);
  et_i2_test_live_builder_counts_v1(&live_copy, &live_reset, &live_decode);
  et_i2_test_retired_builder_counts_v1(
      &dead_copy, &dead_reset, &dead_decode, &retained);
  et_o2_test_live_counts_snapshot_v1(&o2);

  switch (index) {
  case 0: return checked_count(live.tensors);
  case 1: return checked_count(live.parameters);
  case 2: return checked_count(live.borrows);
  case 3: return checked_count(live.copy_plans);
  case 4: return checked_count(live.gradient_plans);
  case 5: return checked_count(live.reset_plans);
  case 6: return checked_count(live.owned_clones);
  case 7: return checked_count(retired.tensors);
  case 8: return checked_count(retired.parameters);
  case 9: return checked_count(retired.borrows);
  case 10: return checked_count(retired.copy_plans);
  case 11: return checked_count(retired.gradient_plans);
  case 12: return checked_count(retired.reset_plans);
  case 13: return checked_count(retired.retained_control_bytes);
  case 14: return checked_count(events.begin_calls);
  case 15: return checked_count(events.view_calls);
  case 16: return checked_count(events.end_calls);
  case 17: return checked_count(live_copy);
  case 18: return checked_count(live_reset);
  case 19: return checked_count(live_decode);
  case 20: return checked_count(dead_copy);
  case 21: return checked_count(dead_reset);
  case 22: return checked_count(dead_decode);
  case 23: return checked_count(retained);
  case 24: return checked_count(o2.builders);
  case 25: return checked_count(o2.optimizers);
  case 26: return checked_count(o2.live_states);
  case 27: return checked_count(o2.dead_states);
  case 28: return checked_count(o2.live_state_handles);
  case 29: return checked_count(o2.dead_state_handles);
  case 30: return checked_count(o2.state_borrows);
  case 31: return checked_count(o2.owned_state_clones);
  case 32: return checked_count(o2.reconstruct_builders);
  default: return -1;
  }
}
