#define _POSIX_C_SOURCE 200809L
#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>
#include <time.h>

int64_t et_tr3_cycle_test_monotonic_millis_v1(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return -1;
  return (int64_t)now.tv_sec * INT64_C(1000) + now.tv_nsec / 1000000;
}

int64_t et_p1_test_live_entry_count_v1(void);
int64_t et_p1_test_tombstone_count_v1(void);

static int64_t bounded(size_t value) {
  return value > (size_t)INT64_MAX ? -1 : (int64_t)value;
}

/* Test-only observable native authority and retained-control census. */
int64_t et_tr3_cycle_test_count_v1(int64_t index) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_o2_test_live_counts_v1 o2 = {.struct_size = sizeof(o2)};
  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_o2_test_live_counts_snapshot_v1(&o2);
  switch (index) {
  case 0: return bounded(live.tensors);
  case 1: return bounded(live.parameters);
  case 2: return bounded(live.borrows);
  case 3: return bounded(live.copy_plans);
  case 4: return bounded(live.gradient_plans);
  case 5: return bounded(live.reset_plans);
  case 6: return bounded(live.owned_clones);
  case 7: return bounded(retired.tensors);
  case 8: return bounded(retired.parameters);
  case 9: return bounded(retired.borrows);
  case 10: return bounded(retired.copy_plans);
  case 11: return bounded(retired.gradient_plans);
  case 12: return bounded(retired.reset_plans);
  case 13: return bounded(retired.retained_control_bytes);
  case 24: return bounded(o2.builders);
  case 25: return bounded(o2.optimizers);
  case 26: return bounded(o2.live_states);
  case 27: return bounded(o2.dead_states);
  case 28: return bounded(o2.live_state_handles);
  case 29: return bounded(o2.dead_state_handles);
  case 30: return bounded(o2.state_borrows);
  case 31: return bounded(o2.owned_state_clones);
  case 32: return bounded(o2.reconstruct_builders);
  case 33: return et_p1_test_live_entry_count_v1();
  case 34: return et_p1_test_tombstone_count_v1();
  default: return 0;
  }
}
