/* Test-only exact-owner gradient images for the post-adoption SAVE cut. */
#if defined(ET_G3R_SAME_OWNER_SAVE_PRIVATE) && defined(ET_F32_TENSOR_TESTING)
#include "../../native/f32_parameter_internal.h"
#include "../../src/eshkol_transformer/m3t_transport.h"
#include <stdint.h>
#include <string.h>

static const size_t words[14] = {
    16, 16, 16, 16, 32, 32, 4, 4, 4, 4, 1024, 4, 4, 8};

static int roster(void *owner, et_f32_parameter *parameters[14]) {
  void *raw[14] = {0};
  const void *identities[14] = {0};
  if (et_m3t_private_owner_save_roster_v1(owner, raw, identities) != 0)
    return -1;
  for (size_t i = 0; i < 14; ++i) {
    if (raw[i] == NULL || identities[i] == NULL) return -1;
    parameters[i] = raw[i];
  }
  return 0;
}

static int gradient_copy(et_f32_parameter *parameter, size_t index,
                         uint32_t *destination) {
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  et_f32_tensor_error error;
  et_f32_tensor *snapshot = NULL;
  size_t length = 0;
  if (et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &error) != 0 ||
      metadata.state != ET_F32_GRADIENT_PRESENT ||
      metadata.contribution_count != 1 ||
      metadata.normalization_weight_bits != UINT32_C(0x3f800000) ||
      et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot, &error) != 0)
    return -1;
  int status = et_f32_tensor_byte_length_v1(snapshot, &length, &error) == 0 &&
                   length == words[index] * sizeof(uint32_t) &&
                   et_f32_tensor_copy_bits_to_v1(snapshot, destination,
                                                  words[index], &error) == 0
                   ? 0 : -1;
  if (et_f32_tensor_destroy_v1(&snapshot, &error) != 0) return -1;
  return status;
}

int64_t et_g3r_test_save_seed_gradients_v1(void *owner) {
  et_f32_parameter *parameters[14];
  uint32_t image[1024], observed[1024];
  et_f32_tensor_error error;
  if (roster(owner, parameters) != 0) return -1;
  for (size_t i = 0; i < 14; ++i) {
    et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    if (et_f32_parameter_gradient_metadata_v1(parameters[i], &metadata,
                                              &error) != 0 ||
        metadata.state != ET_F32_GRADIENT_ABSENT) return -1;
  }
  for (size_t i = 0; i < 14; ++i) {
    for (size_t j = 0; j < words[i]; ++j)
      image[j] = UINT32_C(0x3f000000) + (uint32_t)(i * 2048u + j);
    et_f32_parameter_test_set_gradient_bits_v1(parameters[i], image, words[i]);
    et_f32_parameter_test_set_metadata_v1(parameters[i],
                                          ET_F32_GRADIENT_PRESENT, 1,
                                          UINT32_C(0x3f800000));
    if (gradient_copy(parameters[i], i, observed) != 0 ||
        memcmp(image, observed, words[i] * sizeof(uint32_t)) != 0) return -1;
  }
  return 0;
}

int64_t et_g3r_test_save_gradient_copy_v1(void *owner, int64_t index,
                                            void *bytevector_header,
                                            int64_t byte_count) {
  et_f32_parameter *parameters[14];
  int64_t encoded = -1;
  if (index < 0 || index >= 14 || bytevector_header == NULL ||
      byte_count != (int64_t)(words[index] * sizeof(uint32_t)) ||
      roster(owner, parameters) != 0) return -1;
  memcpy(&encoded, bytevector_header, sizeof(encoded));
  if (encoded != byte_count) return -1;
  return gradient_copy(parameters[index], (size_t)index,
                       (uint32_t *)(void *)((unsigned char *)bytevector_header + 8));
}
#else
typedef int et_g3r_test_save_bits_feature_off;
#endif
