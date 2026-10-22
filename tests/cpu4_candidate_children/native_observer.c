/* Test-only scalar observations; production custody never reads these. */
#include <stdint.h>
#include "o2_optimizer_internal.h"
#include "f32_parameter_internal.h"
int64_t et_cpu4_child_test_o2_count(int64_t kind) {
  et_o2_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_o2_test_live_counts_snapshot_v1(&counts);
  et_f32_test_live_counts_v1 f32 = {.struct_size = sizeof(f32)};
  et_f32_test_live_counts_snapshot_v1(&f32);
  switch (kind) {
    case 0: return (int64_t)counts.builders;
    case 1: return (int64_t)counts.optimizers;
    case 2: return (int64_t)counts.live_states;
    case 3: return (int64_t)counts.live_optimizer_moments;
    case 4: return (int64_t)f32.tensors;
    case 5: return (int64_t)f32.parameters;
    default: return -1;
  }
}
