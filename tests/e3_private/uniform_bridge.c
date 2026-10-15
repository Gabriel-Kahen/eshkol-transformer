/* Exact private boxed entry plus one numeric I2 builder census adapter. */
#include "../../native/e3_private_bridge.c"

#include <stddef.h>
#include <stdint.h>
#include "../../native/f32_parameter_internal.h"

extern void et_i2_test_live_builder_counts_v1(size_t *, size_t *, size_t *);
extern void et_i2_test_retired_builder_counts_v1(size_t *, size_t *, size_t *, size_t *);
extern size_t et_i2_test_decode_builder_control_bytes_v1(void);

int64_t et_e3_uniform_i2_live_builders_v1(void) {
  size_t copies = 0u, resets = 0u, decodes = 0u;
  et_i2_test_live_builder_counts_v1(&copies, &resets, &decodes);
  return (int64_t)(copies + resets + decodes);
}

int64_t et_e3_uniform_i2_retired_builder_controls_v1(void) {
  size_t copies = 0u, resets = 0u, decodes = 0u, bytes = 0u;
  et_i2_test_retired_builder_counts_v1(&copies, &resets, &decodes, &bytes);
  return (int64_t)bytes;
}

int64_t et_e3_uniform_i2_decode_builder_bytes_v1(void) {
  return (int64_t)et_i2_test_decode_builder_control_bytes_v1();
}

/* Prebind test census: a C1 failure must not leave an I2 borrow or plan. */
int64_t et_e3_uniform_live_leases_v1(void) {
  et_f32_test_live_counts_v1 counts = {.struct_size = sizeof(counts)};
  et_f32_test_live_counts_snapshot_v1(&counts);
  return (int64_t)(counts.borrows + counts.copy_plans +
                   counts.gradient_plans + counts.reset_plans);
}
