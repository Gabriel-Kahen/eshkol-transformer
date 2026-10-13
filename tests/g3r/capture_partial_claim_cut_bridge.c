/* Read-only, test-only witnesses over the authentic original owner graph. */
#include "native/f32_parameter_internal.h"
#include "native/o2_optimizer_internal.h"
#include "src/eshkol_transformer/g3t_transport.h"
#include "src/eshkol_transformer/m3t_transport.h"
#include <stdint.h>
#include <string.h>

static const size_t words[14] = {
    16, 16, 16, 16, 32, 32, 4, 4, 4, 4, 1024, 4, 4, 8};

static uint32_t *destination(void *header, int64_t count) {
  int64_t encoded = -1;
  if (header == NULL || count < 0) return NULL;
  memcpy(&encoded, header, sizeof(encoded));
  if (encoded != count) return NULL;
  return (uint32_t *)(void *)((unsigned char *)header + sizeof(encoded));
}

int64_t et_g3r_partial_gradient_copy_v1(void *model, int64_t index,
                                        void *header, int64_t bytes) {
  void *raw[14] = {0};
  const void *identities[14] = {0};
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  et_f32_tensor_error error;
  const et_f32_tensor *gradient;
  size_t length = 0;
  uint32_t *bits;
  if (index < 0 || index >= 14 ||
      bytes != (int64_t)(words[index] * sizeof(uint32_t)) ||
      (bits = destination(header, bytes)) == NULL ||
      et_m3t_private_owner_save_roster_v1(model, raw, identities) != 0 ||
      raw[index] == NULL || identities[index] == NULL ||
      et_f32_parameter_gradient_metadata_v1(raw[index], &metadata, &error) != 0 ||
      metadata.state != ET_F32_GRADIENT_ABSENT ||
      metadata.contribution_count != 0 ||
      metadata.normalization_weight_bits != 0 ||
      (gradient = et_f32_parameter_test_gradient_tensor_v1(raw[index])) == NULL ||
      et_f32_tensor_byte_length_v1(gradient, &length, &error) != 0 ||
      length != (size_t)bytes ||
      et_f32_tensor_copy_bits_to_v1(gradient, bits, words[index], &error) != 0)
    return -1;
  for (size_t word = 0; word < words[index]; ++word)
    if (bits[word] != UINT32_C(0)) return -1;
  return 0;
}

int64_t et_g3r_partial_moment_copy_v1(void *optimizer, int64_t index,
                                      int64_t kind, void *header, int64_t bytes) {
  et_o2_error_v1 error;
  uint32_t *bits;
  if (index < 0 || index >= 14 || kind < 0 || kind > 1 ||
      bytes != (int64_t)(words[index] * sizeof(uint32_t)) ||
      (bits = destination(header, bytes)) == NULL ||
      et_o2_test_optimizer_moment_bits_v1(optimizer, (size_t)index,
                                          (uint32_t)kind, bits, words[index],
                                          &error) != 0)
    return -1;
  return 0;
}

int64_t et_g3r_partial_history_copy_v1(void *generator, void *header,
                                       int64_t bytes) {
  et_g3r_committed_history_snapshot_v1 image = {0};
  uint32_t *bits;
  if (bytes != (int64_t)sizeof(image) ||
      (bits = destination(header, bytes)) == NULL ||
      et_g3t_private_committed_history_snapshot_v1(generator, &image) != 0)
    return -1;
  memcpy(bits, &image, sizeof(image));
  return 0;
}
