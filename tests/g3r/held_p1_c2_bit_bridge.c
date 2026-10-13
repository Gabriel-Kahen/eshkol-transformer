/* Test-only exact bit baselines for all original fixed14 values and absent
 * gradients. A caller first authenticates the live joint G3/M3/O2 hold. */
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#include "f32_parameter_internal.h"
#include "eshkol_transformer/m3t_transport.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static const uint64_t shapes[14][2] = {
    {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 8}, {8, 4}, {4, 0},
    {4, 0}, {4, 0}, {4, 0}, {256, 4}, {4, 0}, {4, 0}, {2, 4}};
static et_f32_parameter_save_witness *witnesses[14];
static void *owner;

static void release_all(void) {
  et_f32_tensor_error error;
  for (size_t index = 14; index > 0; --index)
    if (witnesses[index - 1] &&
        et_f32_parameter_save_witness_end_v1(
            &witnesses[index - 1], &error) != 0)
      abort();
  owner = NULL;
}

int64_t et_g3r_test_held_bits_begin_v1(void *model) {
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  et_f32_tensor_error error;
  if (owner || !model ||
      et_m3t_private_owner_save_roster_v1(model, parameters, identities) != 0)
    return -1;
  owner = model;
  for (size_t index = 0; index < 14; ++index) {
    const size_t rank = shapes[index][1] ? 2u : 1u;
    if (et_f32_parameter_save_witness_begin_v1(
            (et_f32_parameter *)parameters[index], identities[index], rank,
            shapes[index], &witnesses[index], &error) != 0) {
      release_all();
      return -1;
    }
  }
  return 0;
}

int64_t et_g3r_test_held_bits_check_v1(void *model) {
  et_f32_tensor_error error;
  if (!owner || owner != model) return -1;
  for (size_t index = 0; index < 14; ++index)
    if (!witnesses[index] ||
        et_f32_parameter_save_witness_check_v1(
            witnesses[index], &error) != 0)
      return -1;
  return 0;
}

int64_t et_g3r_test_held_bits_end_v1(void *model) {
  if (et_g3r_test_held_bits_check_v1(model) != 0) return -1;
  release_all();
  return 0;
}
