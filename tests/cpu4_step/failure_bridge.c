/* Test-only access to existing O2 failpoints and exact full-moment snapshots.
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
  static const uint64_t shapes[14][2] =
      {{4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},
       {4,0},{4,0},{256,4},{4,0},{4,0},{4,4}};
  et_o2_optimizer_state *a = NULL, *b = NULL;
  et_o2_error_v1 error;
  uint32_t av[1024], bv[1024];
  uint64_t ac = 0, bc = 0;
  size_t an = 0, bn = 0;
  int64_t equal = -1;
  if (et_o2_optimizer_state_snapshot_v1(left, &a, &error) != 0 ||
      et_o2_optimizer_state_snapshot_v1(right, &b, &error) != 0 ||
      et_o2_optimizer_state_entry_count_v1(a, &an, &error) != 0 ||
      et_o2_optimizer_state_entry_count_v1(b, &bn, &error) != 0 ||
      et_o2_optimizer_state_completed_updates_v1(a, &ac, &error) != 0 ||
      et_o2_optimizer_state_completed_updates_v1(b, &bc, &error) != 0)
    goto done;
  if (an != 14 || bn != 14) goto done;
  equal = ac == bc;
  for (size_t index = 0; index < 14; ++index) {
    const size_t rank = shapes[index][1] == 0 ? 1u : 2u;
    if (et_o2_optimizer_state_require_shape_v1(a,index,rank,shapes[index],&error) != 0 ||
        et_o2_optimizer_state_require_shape_v1(b,index,rank,shapes[index],&error) != 0) {
      equal = -1;
      goto done;
    }
    for (uint32_t kind = 0; kind < 2; ++kind) {
      if (et_o2_optimizer_state_copy_moment_bits_v1(a,index,kind,av,elements[index],&error) != 0 ||
          et_o2_optimizer_state_copy_moment_bits_v1(b,index,kind,bv,elements[index],&error) != 0) {
        equal = -1;
        goto done;
      }
      if (memcmp(av,bv,elements[index]*sizeof(*av)) != 0) equal = 0;
    }
  }
done:
  if (a != NULL && et_o2_optimizer_state_release_v1(a,&error) != 0) equal = -1;
  if (b != NULL && et_o2_optimizer_state_release_v1(b,&error) != 0) equal = -1;
  return equal;
}
