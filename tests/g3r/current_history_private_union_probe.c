/* Test-only bridge. The Eshkol aggregate supplies genuine native owners. */
#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_G3R_COMMITTED_HISTORY_PRIVATE 1
#include "../../native/f32_parameter_internal.h"
#include "../../src/eshkol_transformer/g3t_transport.h"
#include "../../src/eshkol_transformer/m3t_transport.h"

#include <stdint.h>
#include <string.h>

/* Exact fixed14 diagnostic profile from m3_call_shapes in the same tree. */
static const uint64_t shapes[14][2] = {
  {4, 4}, {4, 4}, {4, 4}, {4, 4}, {4, 8}, {8, 4}, {4, 0},
  {4, 0}, {4, 0}, {4, 0}, {256, 4}, {4, 0}, {4, 0}, {2, 4}
};

/* 0 means every assertion passed; other values identify the failed cut. */
int64_t et_g3r_current_private_union_probe_v1(
    void *generator, void *model, int64_t pending_g1) {
  void *parameters[14] = {0};
  const void *identities[14] = {0};
  void *aliased[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(
          model, aliased, (const void **)aliased) == 0)
    return 1;
  for (size_t i = 0; i < 14; ++i)
    if (aliased[i]) return 2;
  if (et_m3t_private_owner_save_roster_v1(
          model, parameters, identities) != 0)
    return 3;

  if (pending_g1) {
    void *rejected = et_g3r_private_g3t_save_begin_v1(
        generator, parameters, identities);
    if (rejected) {
      (void)et_g3r_private_g3t_save_end_v1(rejected);
      return 4;
    }
    return 0;
  }

  const void *wrong[14];
  memcpy(wrong, identities, sizeof(wrong));
  wrong[0] = identities[1];
  void *rejected = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, wrong);
  if (rejected) {
    (void)et_g3r_private_g3t_save_end_v1(rejected);
    return 5;
  }

  void *hold = et_g3r_private_g3t_save_begin_v1(
      generator, parameters, identities);
  if (!hold) return 6;
  et_f32_parameter_save_witness *witnesses[14] = {0};
  et_f32_tensor_error error;
  int64_t result = 0;
  et_f32_parameter_save_witness *invalid = NULL;
  if (et_f32_parameter_save_witness_begin_v1(
          parameters[0], identities[1], 2, shapes[0], &invalid, &error) == 0 ||
      invalid)
    result = 7;
  for (size_t i = 0; i < 14 && !result; ++i) {
    const size_t rank = shapes[i][1] ? 2u : 1u;
    if (et_f32_parameter_save_witness_begin_v1(
            parameters[i], identities[i], rank, shapes[i],
            &witnesses[i], &error) != 0 ||
        et_f32_parameter_save_witness_check_v1(witnesses[i], &error) != 0)
      result = 8;
  }

  et_g3r_g3t_held_words_v1 output = {{0}};
  if (!result && et_g3r_private_g3t_save_copy_v1(
          hold, &output, sizeof(output)) != 0)
    result = 9;
  if (!result && (output.word[0] != 1 || output.word[10] != 65 ||
                  output.word[12] != 2 || output.word[13] != 1 ||
                  output.word[14] != 1 || output.word[15] != 0))
    result = 10;
  et_g3r_g3t_held_words_v1 sentinel;
  memset(&sentinel, 0x5a, sizeof(sentinel));
  et_g3r_g3t_held_words_v1 baseline = sentinel;
  if (!result && (et_g3r_private_g3t_save_copy_v1(
          hold, &sentinel, sizeof(sentinel) - 1) == 0 ||
                  memcmp(&sentinel, &baseline, sizeof(sentinel))))
    result = 11;
  unsigned char owner_before[136];
  if (!result) {
    memcpy(owner_before, model, sizeof(owner_before));
    if (et_g3r_private_g3t_save_copy_v1(
            hold, (et_g3r_g3t_held_words_v1 *)model,
            sizeof(output)) == 0 ||
        memcmp(owner_before, model, sizeof(owner_before)))
      result = 12;
  }
  if (!result && et_g3r_private_g3t_save_check_v1(hold) != 0)
    result = 13;
  for (size_t i = 14; i > 0; --i)
    if (witnesses[i - 1] &&
        et_f32_parameter_save_witness_end_v1(&witnesses[i - 1], &error) != 0)
      result = result ? result : 14;
  if (et_g3r_private_g3t_save_end_v1(hold) != 0)
    result = result ? result : 15;
  return result;
}
