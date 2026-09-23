#include "f32_parameter_internal.h"
#include "o2_optimizer_internal.h"

#include <stdint.h>

enum {
  ET_O2_HOOK_TEST_BUILDERS = 0,
  ET_O2_HOOK_TEST_OPTIMIZERS = 1,
  ET_O2_HOOK_TEST_TENSORS = 2,
  ET_O2_HOOK_TEST_OWNED_CLONES = 3
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
    case ET_O2_HOOK_TEST_TENSORS:
      return (int64_t)i2.tensors;
    case ET_O2_HOOK_TEST_OWNED_CLONES:
      return (int64_t)i2.owned_clones;
    default:
      return -1;
  }
}
