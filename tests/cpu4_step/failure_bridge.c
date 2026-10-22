/* Test-only access to existing O2 failpoints and exact full-moment words.
 * No optimizer algorithm, production ABI or tensor mutation lives here. */
#include "o2_optimizer_internal.h"
#include "tr3_o2_step_clear_internal.h"

#include <stdint.h>
#include <string.h>

void et_cpu4_step_test_fail_o2_after(int64_t allowed) {
  if (allowed < 0) et_o2_test_reset_failpoints_v1();
  else et_o2_test_fail_alloc_after_v1((size_t)allowed);
}
void et_cpu4_step_test_fail_plan_after(int64_t allowed) {
  if (allowed < 0) et_o2_trainer_step_clear_test_reset_failpoints_v1();
  else et_o2_trainer_step_clear_test_fail_alloc_after_v1((size_t)allowed);
}

int64_t et_cpu4_step_test_moments_equal(void *left, void *right) {
  static const size_t elements[14] =
      {16,16,16,16,32,32,4,4,4,4,1024,4,4,16};
  et_o2_error_v1 error;
  uint32_t av[1024], bv[1024];
  uint64_t ac = 0, bc = 0;
  int64_t equal;
  /* Real snapshots correctly require absent gradients. The existing read-only
   * ET_O2_TESTING getter permits inspection during expected accumulation;
   * the genuine fixture's C4/P1 binding supplies the exact14 shapes/owners. */
  if (et_o2_optimizer_completed_updates_v1(left, &ac, &error) != 0 ||
      et_o2_optimizer_completed_updates_v1(right, &bc, &error) != 0)
    return -1;
  equal = ac == bc;
  for (size_t index = 0; index < 14; ++index) {
    for (uint32_t kind = 0; kind < 2; ++kind) {
      if (et_o2_test_optimizer_moment_bits_v1(left,index,kind,av,elements[index],&error) != 0 ||
          et_o2_test_optimizer_moment_bits_v1(right,index,kind,bv,elements[index],&error) != 0)
        return -1;
      if (memcmp(av,bv,elements[index]*sizeof(*av)) != 0) equal = 0;
    }
  }
  return equal;
}
