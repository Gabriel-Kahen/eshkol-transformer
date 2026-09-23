#include "../../native/tr3b_objective_bridge.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "TR3-B native check failed at line %d\n", __LINE__); \
  return 1; \
} } while (0)

typedef struct bytes16 { int64_t length; unsigned char payload[16]; } bytes16;
typedef struct bytes2 { int64_t length; unsigned char payload[2]; } bytes2;

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                     const char *dtype,
                                     const uint64_t shape[2]) {
  et_kernel_tensor_view_v1 result = {
    sizeof(result), data, bytes, dtype, "cpu",
    ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0, 2, shape
  };
  return result;
}

int main(void) {
  const uint64_t shape[2] = {1, 2};
  int64_t source[2] = {197, 41};
  bytes16 destination = {16, {0}};
  et_kernel_tensor_view_v1 candidate =
      view(source, sizeof(source), "i64", shape);
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_OK);
  CHECK(memcmp(source, destination.payload, sizeof(source)) == 0);

  candidate.dtype = "f32";
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_DTYPE_MISMATCH);
  candidate.dtype = "i64";
  candidate.device = "gpu";
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_DEVICE_MISMATCH);
  candidate.device = "cpu";
  candidate.rank = 1;
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_SHAPE_MISMATCH);
  candidate.rank = 2;
  destination.length = 15;
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_SHAPE_MISMATCH);
  destination.length = 16;

  candidate.data = destination.payload;
  CHECK(et_tr3b_private_copy_targets_v1(&candidate, &destination) ==
        ET_TR3B_INVALID_ARGUMENT);

  _Alignas(8) unsigned char overlap[40] = {0};
  bytes2 mask = {2, {1, 1}};
  int64_t target_length = 16;
  int64_t observation_length = 12;
  memcpy(overlap, &target_length, sizeof(target_length));
  memcpy(overlap + 8, &observation_length, sizeof(observation_length));
  unsigned char before[12];
  memcpy(before, overlap + 16, sizeof(before));
  /* target payload [8,24) and observation payload [16,28) overlap. The
   * native bridge must reject before dereferencing either opaque owner. */
  CHECK(et_tr3b_private_objective_prepare_v1(
            NULL, NULL, overlap, &mask, overlap + 8) ==
        ET_TR3B_INVALID_ARGUMENT);
  CHECK(memcmp(before, overlap + 16, sizeof(before)) == 0);

  puts("TR3B-NATIVE-VALIDATION-PASS");
  return 0;
}
