#include "mod5_v256_c2_transfer.h"
#include "o2_optimizer_internal.h"

#include <stdlib.h>
#include <string.h>

enum { SLOTS = 12, TOKENS = 8, MAX_WORDS = 4096 };

int64_t et_mod5_v256_adamw_row_v1(const et_kernel_tensor_view_v1 *view,
                                  int64_t mask, uint8_t *output) {
  if ((mask != 0 && mask != 1) || view == NULL ||
      view->struct_size != ET_KERNEL_TENSOR_VIEW_V1_0_SIZE ||
      view->dtype == NULL || strcmp(view->dtype, mask ? "bool" : "i64") != 0 ||
      view->device == NULL || strcmp(view->device, "cpu") != 0 ||
      view->layout != ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR ||
      view->offset_bytes != 0 || view->rank != 2 || view->shape == NULL ||
      view->shape[0] != 1 || view->shape[1] != TOKENS || view->data == NULL ||
      view->byte_length != TOKENS * (mask ? sizeof(uint8_t) : sizeof(int64_t)))
    return -1;
  const uint8_t *data = view->data;
  if (mask) {
    for (size_t index = 0; index < TOKENS; ++index)
      if (data[index] != 1)
        return -1;
    return 0;
  }
  if (output == NULL)
    return -1;
  uint8_t tokens[TOKENS];
  for (size_t index = 0; index < TOKENS; ++index) {
    int64_t token;
    memcpy(&token, data + index * sizeof(token), sizeof(token));
    if (token < 0 || token > UINT8_MAX)
      return -1;
    tokens[index] = (uint8_t)token;
  }
  memcpy(output, tokens, sizeof(tokens));
  return 0;
}

int64_t et_mod5_v256_adamw_set_slot_v1(et_f32_parameter **slots, int64_t index,
                                       et_f32_parameter *parameter) {
  if (slots == NULL || index < 0 || index >= SLOTS || parameter == NULL)
    return -1;
  slots[index] = parameter;
  return 0;
}

int64_t et_mod5_v256_adamw_gradient_v1(const et_f32_parameter *parameter,
                                      uint32_t *expected, int64_t count) {
  et_f32_gradient_metadata_v1 meta = {.struct_size = sizeof(meta)};
  if (count < 0 || count > MAX_WORDS ||
      et_f32_parameter_gradient_metadata_v1(parameter, &meta, NULL) != 0)
    return -1;
  if (count == 0)
    return meta.contribution_count == 0 && meta.normalization_weight_bits == 0
               ? 0 : -1;
  if (expected == NULL || meta.contribution_count != 1 ||
      meta.normalization_weight_bits != UINT32_C(0x3f800000))
    return -1;

  et_f32_tensor *snapshot = NULL;
  uint32_t bits[MAX_WORDS];
  if (et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot, NULL) != 0)
    return -1;
  const int32_t copy =
      et_f32_tensor_copy_bits_to_v1(snapshot, bits, (size_t)count, NULL);
  if (et_f32_tensor_destroy_v1(&snapshot, NULL) != 0)
    abort();
  if (copy != 0)
    return -1;
  for (size_t index = 0; index < (size_t)count; ++index) {
    /* An absent numerator starts at +0, so I2 can canonicalize -0 on add. */
    if (bits[index] != expected[index] &&
        !(bits[index] == 0 && expected[index] == UINT32_C(0x80000000)))
      return -1;
  }
  memcpy(expected, bits, (size_t)count * sizeof(*bits));
  return 0;
}

int64_t et_mod5_v256_adamw_moment_v1(const et_o2_optimizer_state *state,
                                    int64_t slot, int64_t moment,
                                    uint32_t *bits, int64_t count) {
  if (slot < 0 || slot >= SLOTS || moment < 0 || moment > 1 ||
      count <= 0 || count > MAX_WORDS || bits == NULL)
    return -1;
  return et_o2_optimizer_state_copy_moment_bits_v1(
      state, (size_t)slot, (uint32_t)moment, bits, (size_t)count, NULL);
}
