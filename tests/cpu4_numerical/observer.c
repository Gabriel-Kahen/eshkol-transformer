/* Test-only readout of the canonical lease's fourteen parameter planes. */
#include "eshkol_transformer/g3c4_model_owner_internal.h"
#include "eshkol_transformer/kernel_abi.h"
#include "eshkol_transformer/cpu4_training_transport.h"
#include "f32_parameter_internal.h"
#include <stdint.h>
#include <string.h>

static const size_t words[14] =
    {16, 16, 16, 16, 32, 32, 4, 4, 4, 4, 1024, 4, 4, 16};

int64_t et_cpu4_numerical_plane(void *lease, int64_t parameter,
                               int64_t gradient, void *destination,
                               int64_t byte_count) {
  if (parameter < 0 || parameter >= 14 ||
      (gradient != 0 && gradient != 1) || !destination ||
      byte_count != (int64_t)(4u * words[parameter]))
    return -1;
  struct bytevector { int64_t length; unsigned char bytes[]; } *out = destination;
  if (out->length != byte_count) return -1;
  et_f32_parameter *owner =
      et_g3c4_private_training_lease_parameter_v1(lease, parameter);
  if (!owner) return -1;
  const et_f32_tensor *plane = NULL;
  et_f32_tensor *snapshot = NULL;
  et_f32_tensor_error error;
  if (gradient) {
    if (et_f32_parameter_gradient_snapshot_v1(owner, &snapshot, &error))
      return -1;
    plane = snapshot;
  } else if (et_f32_parameter_value_tensor_v1(owner, &plane, &error)) {
    return -1;
  }
  uint32_t bits[1024];
  int32_t status = et_f32_tensor_copy_bits_to_v1(
      plane, bits, words[parameter], &error);
  if (snapshot && et_f32_owned_tensor_release_v1(snapshot, &error)) return -1;
  if (status) return -1;
  for (size_t i = 0; i < words[parameter]; ++i)
    for (size_t j = 0; j < 4; ++j)
      out->bytes[4u * i + j] = (unsigned char)(bits[i] >> (8u * j));
  return 0;
}

/* No model data moves on a malformed shape, dtype or device request. */
int64_t et_cpu4_numerical_bad_shape(void *lease, int64_t selector) {
  if (selector < 0 || selector > 3) return -1;
  const uint64_t shapes[4][2] = {{1, 2}, {4, 1}, {1, 4}, {1, 4}};
  const int64_t input[4] = {3, 197, 3, 41};
  struct { int64_t length; int64_t values[4]; } output =
      {32, {71, 72, 73, 74}};
  const et_kernel_tensor_view_v1 view = {
      .struct_size = sizeof(view), .data = (void *)input,
      .byte_length = sizeof(input),
      .dtype = selector == 2 ? "f32" : "i64",
      .device = selector == 3 ? "cuda" : "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = 2, .shape = shapes[selector]};
  if (et_cpu4_private_batch_copy_v1(lease, &view, &output, 0) == 0)
    return 0;
  return output.length == 32 && output.values[0] == 71 &&
         output.values[1] == 72 && output.values[2] == 73 &&
         output.values[3] == 74;
}
