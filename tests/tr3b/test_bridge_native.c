#include "../../native/tr3b_objective_bridge.h"
#include "../../src/eshkol_transformer/m3t_transport.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "TR3-B native check failed at line %d\n", __LINE__); \
  return 1; \
} } while (0)

typedef struct bytes16 { int64_t length; unsigned char payload[16]; } bytes16;
typedef struct bytes12 { int64_t length; unsigned char payload[12]; } bytes12;
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

static int preserved_objective(int64_t wanted, void *targets, void *mask,
                               void *observation) {
  bytes12 *bytes = observation;
  unsigned char before[12];
  memcpy(before, bytes->payload, sizeof(before));
  int64_t status = et_tr3b_private_objective_prepare_v1(
      NULL, NULL, targets, mask, observation);
  return status == wanted && memcmp(before, bytes->payload, sizeof(before)) == 0;
}

int main(void) {
  const uint64_t shape[2] = {1, 2};
  int64_t source[2] = {197, 41};
  et_kernel_tensor_view_v1 candidate =
      view(source, sizeof(source), "i64", shape);

  /* Genuine M3T input ingress succeeds before the boundary negatives. */
  void *input = et_m3t_private_input_create_v1();
  CHECK(input != NULL);
  CHECK(et_tr3b_private_stage_input_v1(input, &candidate) == ET_TR3B_OK);
  source[1] = 256;
  CHECK(et_tr3b_private_stage_input_v1(input, &candidate) ==
        ET_TR3B_SHAPE_MISMATCH);
  source[1] = 41;
  CHECK(et_m3t_private_input_release_v1(input) == ET_M3T_OK);
  CHECK(et_tr3b_private_stage_input_v1(input, &candidate) ==
        ET_TR3B_INVALID_STATE);

  bytes16 destination = {16, {0}};
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

  unsigned char mask_source[2] = {1, 0};
  bytes2 mask_destination = {2, {9, 9}};
  et_kernel_tensor_view_v1 mask_view =
      view(mask_source, sizeof(mask_source), "bool", shape);
  CHECK(et_tr3b_private_copy_mask_v1(&mask_view, &mask_destination) ==
        ET_TR3B_OK);
  CHECK(memcmp(mask_source, mask_destination.payload, sizeof(mask_source)) == 0);
  mask_destination.length = 1;
  CHECK(et_tr3b_private_copy_mask_v1(&mask_view, &mask_destination) ==
        ET_TR3B_SHAPE_MISMATCH);
  mask_destination.length = 2;
  mask_view.data = mask_destination.payload;
  CHECK(et_tr3b_private_copy_mask_v1(&mask_view, &mask_destination) ==
        ET_TR3B_INVALID_ARGUMENT);

  bytes16 targets = {16, {0}};
  bytes2 mask = {2, {1, 1}};
  bytes12 observation = {12, {51, 51, 51, 51, 51, 51,
                              51, 51, 51, 51, 51, 51}};
  bytes16 malformed_targets = {15, {0}};
  bytes2 malformed_mask = {1, {1, 1}};
  CHECK(preserved_objective(ET_TR3B_SHAPE_MISMATCH, &malformed_targets,
                            &mask, &observation));
  CHECK(preserved_objective(ET_TR3B_SHAPE_MISMATCH, &targets,
                            &malformed_mask, &observation));

  /* Target payload [8,24) overlaps mask payload [16,18). */
  _Alignas(8) unsigned char target_mask[32] = {0};
  int64_t length16 = 16, length2 = 2, length12 = 12;
  memcpy(target_mask, &length16, sizeof(length16));
  memcpy(target_mask + 8, &length2, sizeof(length2));
  CHECK(preserved_objective(ET_TR3B_INVALID_ARGUMENT, target_mask,
                            target_mask + 8, &observation));

  /* Target payload [8,24) overlaps observation payload [16,28). */
  _Alignas(8) unsigned char target_observation[40] = {0};
  memcpy(target_observation, &length16, sizeof(length16));
  memcpy(target_observation + 8, &length12, sizeof(length12));
  CHECK(preserved_objective(ET_TR3B_INVALID_ARGUMENT, target_observation,
                            &mask, target_observation + 8));

  /* Observation payload [8,20) overlaps mask payload [16,18). */
  _Alignas(8) unsigned char mask_observation[32] = {0};
  memcpy(mask_observation, &length12, sizeof(length12));
  memcpy(mask_observation + 8, &length2, sizeof(length2));
  CHECK(preserved_objective(ET_TR3B_INVALID_ARGUMENT, &targets,
                            mask_observation + 8, mask_observation));

  /* All three scratch objects share one backing span. Validation must reject
   * before dereferencing the opaque graph or seed and preserve observation. */
  _Alignas(8) unsigned char all_alias[48] = {0};
  memcpy(all_alias, &length16, sizeof(length16));
  memcpy(all_alias + 8, &length12, sizeof(length12));
  memcpy(all_alias + 16, &length2, sizeof(length2));
  CHECK(preserved_objective(ET_TR3B_INVALID_ARGUMENT, all_alias,
                            all_alias + 16, all_alias + 8));

  puts("TR3B-NATIVE-VALIDATION-PASS");
  return 0;
}
