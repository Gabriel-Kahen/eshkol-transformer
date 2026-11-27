#include "mod5_v256_c2_transfer.h"

#include <stddef.h>
#include <stdlib.h>

enum { MOD5_V256_PARAMETERS = 12, MOD5_V256_TIED_PARAMETERS = 11 };

typedef struct shape {
  size_t rank;
  uint64_t dimensions[2];
} shape;

/* C1 path order, including two logical entries for a tied model. */
static const shape roster[MOD5_V256_PARAMETERS] = {
    {2, {8, 16}},   {2, {16, 16}}, {2, {16, 16}}, {2, {8, 16}},
    {2, {16, 32}},  {2, {32, 16}}, {2, {32, 16}}, {1, {16}},
    {1, {16}},      {2, {256, 16}}, {1, {16}},     {2, {256, 16}},
};

static size_t elements(const shape *entry) {
  return (size_t)(entry->dimensions[0] *
                  (entry->rank == 2u ? entry->dimensions[1] : 1u));
}

static int matches(const et_f32_tensor *tensor, const shape *entry) {
  size_t rank;
  if (et_f32_tensor_rank_v1(tensor, &rank, NULL) != 0 || rank != entry->rank)
    return 0;
  for (size_t index = 0; index < rank; ++index) {
    uint64_t dimension;
    if (et_f32_tensor_shape_at_v1(tensor, index, &dimension, NULL) != 0 ||
        dimension != entry->dimensions[index])
      return 0;
  }
  return 1;
}

static et_f32_tensor *tensor_from_words(const shape *entry,
                                        const uint32_t *words) {
  et_f32_tensor *tensor = NULL;
  if (et_f32_tensor_create_v1(entry->rank, entry->dimensions, &tensor, NULL) != 0)
    return NULL;
  if (et_f32_tensor_copy_bits_from_v1(tensor, words, elements(entry), NULL) != 0 &&
      et_f32_tensor_destroy_v1(&tensor, NULL) != 0)
    abort();
  return tensor;
}

et_f32_parameter *et_mod5_v256_test_parameter_v1(
    int64_t slot, const uint32_t *words, int64_t word_count) {
  if (slot < 0 || slot >= MOD5_V256_PARAMETERS ||
      word_count != (int64_t)elements(&roster[slot]) || words == NULL)
    return NULL;
  et_f32_tensor *tensor = tensor_from_words(&roster[slot], words);
  et_f32_parameter *parameter = NULL;
  if (tensor == NULL)
    return NULL;
  (void)et_f32_parameter_create_v1(tensor, &parameter, NULL);
  if (et_f32_tensor_destroy_v1(&tensor, NULL) != 0)
    abort();
  return parameter;
}

int64_t et_mod5_v256_test_parameter_bits_v1(
    int64_t slot, const et_f32_parameter *parameter, uint32_t *words,
    int64_t word_count) {
  const et_f32_tensor *value = NULL;
  if (slot < 0 || slot >= MOD5_V256_PARAMETERS ||
      word_count != (int64_t)elements(&roster[slot]) || words == NULL ||
      et_f32_parameter_value_tensor_v1(parameter, &value, NULL) != 0 ||
      !matches(value, &roster[slot]))
    return -1;
  return et_f32_tensor_copy_bits_to_v1(value, words, (size_t)word_count, NULL);
}

int64_t et_mod5_v256_test_gradient_transfer_v1(
    et_f32_parameter *const *parameters, int64_t parameter_count,
    const uint32_t *words, int64_t word_count, int64_t tied) {
  if ((tied != 0 && tied != 1) || parameters == NULL || words == NULL ||
      parameter_count != (tied ? MOD5_V256_TIED_PARAMETERS : MOD5_V256_PARAMETERS))
    return -1;
  const size_t count = (size_t)parameter_count;
  size_t required = 0u;
  for (size_t index = 0u; index < count; ++index) {
    const et_f32_tensor *value = NULL;
    if (et_f32_parameter_value_tensor_v1(parameters[index], &value, NULL) != 0 ||
        !matches(value, &roster[index]))
      return -1;
    required += elements(&roster[index]);
  }
  if (word_count != (int64_t)required)
    return -1;

  et_f32_tensor *numerators[MOD5_V256_PARAMETERS] = {NULL};
  et_f32_gradient_contribution_v1 contributions[MOD5_V256_PARAMETERS];
  et_f32_gradient_plan *plan = NULL;
  int64_t result = -1;
  size_t offset = 0u;
  for (size_t index = 0u; index < count; ++index) {
    numerators[index] = tensor_from_words(&roster[index], words + offset);
    if (numerators[index] == NULL)
      goto cleanup;
    contributions[index] = (et_f32_gradient_contribution_v1){
        .struct_size = sizeof(contributions[index]),
        .destination = parameters[index],
        .weighted_numerator = numerators[index],
        .expected_ordinal = 0,
    };
    offset += elements(&roster[index]);
  }
  result = et_f32_gradient_plan_prepare_v1(count, contributions, 0x3f800000u,
                                            &plan, NULL);
  if (result == 0)
    result = et_f32_gradient_plan_commit_v1(plan, NULL);
cleanup:
  /* Only this transfer owns these temporary tensors and plan. A release
   * defect is fatal, including after a successful atomic commit. */
  if (et_f32_gradient_plan_release_v1(&plan, NULL) != 0)
    abort();
  for (size_t index = 0u; index < count; ++index)
    if (et_f32_tensor_destroy_v1(&numerators[index], NULL) != 0)
      abort();
  return result;
}
