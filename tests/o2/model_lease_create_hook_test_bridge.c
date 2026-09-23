#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>

enum {
  ET_O2_HOOK_TEST_BUILDERS = 0,
  ET_O2_HOOK_TEST_OPTIMIZERS = 1,
  ET_O2_HOOK_TEST_LIVE_STATES = 2,
  ET_O2_HOOK_TEST_DEAD_STATES = 3,
  ET_O2_HOOK_TEST_LIVE_STATE_HANDLES = 4,
  ET_O2_HOOK_TEST_DEAD_STATE_HANDLES = 5,
  ET_O2_HOOK_TEST_STATE_BORROWS = 6,
  ET_O2_HOOK_TEST_OWNED_STATE_CLONES = 7,
  ET_O2_HOOK_TEST_RECONSTRUCT_BUILDERS = 8,
  ET_O2_HOOK_TEST_TENSORS = 9,
  ET_O2_HOOK_TEST_PARAMETERS = 10,
  ET_O2_HOOK_TEST_BORROWS = 11,
  ET_O2_HOOK_TEST_COPY_PLANS = 12,
  ET_O2_HOOK_TEST_GRADIENT_PLANS = 13,
  ET_O2_HOOK_TEST_RESET_PLANS = 14,
  ET_O2_HOOK_TEST_OWNED_CLONES = 15
};

int64_t et_o2_model_lease_hook_test_live_count_v1(int64_t kind) {
  et_o2_test_live_counts_v1 o2 = {.struct_size = sizeof(o2)};
  et_f32_test_live_counts_v1 i2 = {.struct_size = sizeof(i2)};

  et_o2_test_live_counts_snapshot_v1(&o2);
  et_f32_test_live_counts_snapshot_v1(&i2);
  switch (kind) {
    case ET_O2_HOOK_TEST_BUILDERS:
      return (int64_t)o2.builders;
    case ET_O2_HOOK_TEST_OPTIMIZERS:
      return (int64_t)o2.optimizers;
    case ET_O2_HOOK_TEST_LIVE_STATES:
      return (int64_t)o2.live_states;
    case ET_O2_HOOK_TEST_DEAD_STATES:
      return (int64_t)o2.dead_states;
    case ET_O2_HOOK_TEST_LIVE_STATE_HANDLES:
      return (int64_t)o2.live_state_handles;
    case ET_O2_HOOK_TEST_DEAD_STATE_HANDLES:
      return (int64_t)o2.dead_state_handles;
    case ET_O2_HOOK_TEST_STATE_BORROWS:
      return (int64_t)o2.state_borrows;
    case ET_O2_HOOK_TEST_OWNED_STATE_CLONES:
      return (int64_t)o2.owned_state_clones;
    case ET_O2_HOOK_TEST_RECONSTRUCT_BUILDERS:
      return (int64_t)o2.reconstruct_builders;
    case ET_O2_HOOK_TEST_TENSORS:
      return (int64_t)i2.tensors;
    case ET_O2_HOOK_TEST_PARAMETERS:
      return (int64_t)i2.parameters;
    case ET_O2_HOOK_TEST_BORROWS:
      return (int64_t)i2.borrows;
    case ET_O2_HOOK_TEST_COPY_PLANS:
      return (int64_t)i2.copy_plans;
    case ET_O2_HOOK_TEST_GRADIENT_PLANS:
      return (int64_t)i2.gradient_plans;
    case ET_O2_HOOK_TEST_RESET_PLANS:
      return (int64_t)i2.reset_plans;
    case ET_O2_HOOK_TEST_OWNED_CLONES:
      return (int64_t)i2.owned_clones;
    default:
      return -1;
  }
}
