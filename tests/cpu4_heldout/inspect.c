/* Development-only observations. No model, optimizer or gradient mutation.
 * Link into the SAME I2/O2/C4 registry aggregate; O2 requires ET_O2_TESTING. */
#include "eshkol_transformer/g3c4_model_owner_internal.h"
#include "o2_optimizer_internal.h"
#include <stdlib.h>
#include <string.h>

#ifndef ET_O2_TESTING
#error "held-out moment observation requires ET_O2_TESTING"
#endif

enum { HELDOUT_WORDS = 1192, HELDOUT_VALUE_BYTES = 4768 };
_Static_assert(HELDOUT_WORDS * sizeof(uint32_t) == HELDOUT_VALUE_BYTES,
               "CPU4 parameter byte contract changed");
static const size_t counts[14] = {16,16,16,16,32,32,4,4,4,4,1024,4,4,16};
static const uint64_t shapes[14][2] = {
  {4,4},{4,4},{4,4},{4,4},{4,8},{8,4},{4,0},{4,0},
  {4,0},{4,0},{256,4},{4,0},{4,0},{4,4}};
typedef struct observation {
  void *lease, *optimizer, *parameters[14];
  const void *identities[14];
  uint64_t updates;
  uint32_t values[HELDOUT_WORDS], moments[2][HELDOUT_WORDS];
} observation;
typedef struct image { struct image *next; observation data; } image;
typedef struct output { struct output *next; et_f32_tensor *tensor; } output;
static image *images;
static output *outputs;

static int capture(void *lease, void *optimizer, observation *out) {
  et_f32_tensor_error ferror;
  et_o2_error_v1 oerror;
  size_t offset = 0;
  memset(out, 0, sizeof(*out));
  if (et_g3c4_private_training_lease_check_v1(lease) != 0 ||
      et_o2_optimizer_completed_updates_v1(optimizer, &out->updates, &oerror) != 0 ||
      et_o2_optimizer_require_absent_gradients_v1(optimizer, &oerror) != 0)
    return -1;
  out->lease = lease;
  out->optimizer = optimizer;
  for (int64_t i = 0; i < 14; ++i) {
    const et_f32_tensor *value = NULL;
    et_f32_parameter *parameter = et_g3c4_private_training_lease_parameter_v1(lease, i);
    const void *identity = et_g3c4_private_training_lease_identity_v1(lease, i);
    et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    size_t rank = 0, count = 0, wanted_rank = shapes[i][1] ? 2u : 1u;
    if (!parameter || !identity ||
        et_f32_parameter_validate_identity_v1(parameter, identity, &ferror) != 0 ||
        et_f32_parameter_gradient_metadata_v1(parameter, &metadata, &ferror) != 0 ||
        metadata.state != ET_F32_GRADIENT_ABSENT || metadata.contribution_count != 0 ||
        metadata.normalization_weight_bits != 0 ||
        et_f32_parameter_value_tensor_v1(parameter, &value, &ferror) != 0 ||
        et_f32_tensor_rank_v1(value, &rank, &ferror) != 0 || rank != wanted_rank ||
        et_f32_tensor_element_count_v1(value, &count, &ferror) != 0 || count != counts[i])
      return -1;
    for (size_t d = 0; d < rank; ++d) {
      uint64_t dimension = 0;
      if (et_f32_tensor_shape_at_v1(value, d, &dimension, &ferror) != 0 ||
          dimension != shapes[i][d]) return -1;
    }
    if (offset + count > HELDOUT_WORDS ||
        et_f32_tensor_copy_bits_to_v1(value, out->values + offset, count, &ferror) != 0)
      return -1;
    for (uint32_t kind = 0; kind < 2; ++kind)
      if (et_o2_test_optimizer_moment_bits_v1(optimizer, (size_t)i, kind,
            out->moments[kind] + offset, count, &oerror) != 0) return -1;
    out->parameters[i] = parameter;
    out->identities[i] = identity;
    offset += count;
  }
  return offset == HELDOUT_WORDS ? 0 : -1;
}

void *et_cpu4_heldout_image_create(void *lease, void *optimizer) {
  image *value = malloc(sizeof(*value));
  if (!value) return NULL;
  if (capture(lease, optimizer, &value->data) != 0) { free(value); return NULL; }
  value->next = images;
  images = value;
  return value;
}
int64_t et_cpu4_heldout_image_equal(void *lease, void *optimizer, void *candidate) {
  image *value = images;
  observation actual;
  while (value && value != candidate) value = value->next;
  if (!value || value->data.lease != lease || value->data.optimizer != optimizer ||
      capture(lease, optimizer, &actual) != 0) return -1;
  return memcmp(&value->data, &actual, sizeof(actual)) == 0 ? 1 : 0;
}
int64_t et_cpu4_heldout_image_destroy(void *candidate) {
  image **slot = &images;
  while (*slot && *slot != candidate) slot = &(*slot)->next;
  if (!*slot) return -1;
  image *value = *slot;
  *slot = value->next;
  free(value);
  return 0;
}
void *et_cpu4_heldout_output_create(void) {
  et_f32_tensor_error error;
  output *value = malloc(sizeof(*value));
  if (!value) return NULL;
  value->tensor = NULL;
  if (et_f32_tensor_create_v1(0, NULL, &value->tensor, &error) != 0) {
    free(value); return NULL;
  }
  value->next = outputs;
  outputs = value;
  return value->tensor;
}
int64_t et_cpu4_heldout_output_bits(void *candidate) {
  output *value = outputs;
  et_f32_tensor_error error;
  size_t rank = 1, count = 0;
  uint32_t bits = 0;
  while (value && value->tensor != candidate) value = value->next;
  if (!value || et_f32_tensor_rank_v1(value->tensor, &rank, &error) != 0 || rank != 0 ||
      et_f32_tensor_element_count_v1(value->tensor, &count, &error) != 0 || count != 1 ||
      et_f32_tensor_copy_bits_to_v1(value->tensor, &bits, 1, &error) != 0) return -1;
  return (int64_t)bits;
}
int64_t et_cpu4_heldout_output_destroy(void *candidate) {
  output **slot = &outputs;
  et_f32_tensor_error error;
  while (*slot && (*slot)->tensor != candidate) slot = &(*slot)->next;
  if (!*slot) return -1;
  output *value = *slot;
  if (et_f32_tensor_destroy_v1(&value->tensor, &error) != 0) return -1;
  *slot = value->next;
  free(value);
  return 0;
}
int64_t et_cpu4_heldout_live_owned(void) {
  int64_t count = 0;
  for (image *value = images; value; value = value->next) ++count;
  for (output *value = outputs; value; value = value->next) ++count;
  return count;
}
