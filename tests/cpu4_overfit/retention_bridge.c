/* Read-only test instrumentation; no training algorithm or failure injection. */
#include "f32_parameter_internal.h"
#include "tr3_o2_step_clear_internal.h"
#include <stdint.h>
#include <string.h>

static et_f32_test_live_counts_v1 baseline;
static et_f32_test_retired_counts_v1 initial_retired;
static int captured;

int64_t et_cpu4_overfit_test_capture(void) {
  memset(&baseline, 0, sizeof(baseline));
  baseline.struct_size = sizeof(baseline);
  initial_retired.struct_size = sizeof(initial_retired);
  et_f32_test_live_counts_snapshot_v1(&baseline);
  et_f32_test_retired_counts_snapshot_v1(&initial_retired);
  captured = 1;
  return baseline.parameters == 14u && baseline.tensors == 56u;
}
int64_t et_cpu4_overfit_test_payload_equal(void) {
  et_f32_test_live_counts_v1 now = {.struct_size = sizeof(now)};
  et_f32_test_live_counts_snapshot_v1(&now);
  return captured && memcmp(&baseline, &now, sizeof(now)) == 0;
}
int64_t et_cpu4_overfit_test_count(int64_t selector) {
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_o2_trainer_step_clear_test_counts_v1 plans = {.struct_size = sizeof(plans)};
  size_t value;
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_o2_trainer_step_clear_test_counts_snapshot_v1(&plans);
  switch (selector) {
    case 0: value = baseline.tensors; break;
    case 1: value = initial_retired.retained_control_bytes; break;
    case 2: value = retired.retained_control_bytes; break;
    case 3: value = plans.live_plans; break;
    case 4: value = plans.committed_plans; break;
    case 5: value = plans.aborted_plans; break;
    case 6: value = plans.live_stages; break;
    case 7: value = plans.retained_control_bytes; break;
    default: return -1;
  }
  return value > INT64_MAX ? -1 : (int64_t)value;
}
