/* Fixture-only adapter for the admitted short-MQA AC4 row and native I2. */
#include "eshkol_transformer/ac4_checkpointing_abi.h"
#include "eshkol_transformer/kernel_abi.h"
#include "f32_parameter_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { PARAMETERS = 9, PARAMETER_FLOATS = 104, TOKENS = 2, WIDTH = 4 };

static const struct {
  size_t rank, count;
  uint64_t extents[2];
} shapes[PARAMETERS] = {
    {1, 4, {4}},      {2, 16, {4, 4}}, {2, 8, {2, 4}},
    {2, 8, {2, 4}},  {2, 16, {4, 4}}, {1, 4, {4}},
    {2, 16, {4, 4}}, {2, 16, {4, 4}}, {2, 16, {4, 4}},
};

static unsigned char *bytevector_payload(void *header, int64_t length) {
  int64_t actual;
  if (!header)
    return NULL;
  memcpy(&actual, header, sizeof(actual));
  return actual == length ? (unsigned char *)header + sizeof(actual) : NULL;
}

static et_f32_tensor *new_tensor(size_t index, const void *bits) {
  et_f32_tensor *tensor = NULL;
  uint32_t words[WIDTH * WIDTH];
  memcpy(words, bits, shapes[index].count * sizeof(words[0]));
  if (et_f32_tensor_create_v1(shapes[index].rank, shapes[index].extents,
                              &tensor, NULL) != 0)
    return NULL;
  if (et_f32_tensor_copy_bits_from_v1(tensor, words, shapes[index].count,
                                      NULL) != 0)
    et_f32_tensor_destroy_v1(&tensor, NULL);
  return tensor;
}

et_f32_parameter *et_mod5_c2_test_parameter_v1(int64_t parameter, void *header,
                                               int64_t bytes) {
  if (parameter < 0 || parameter >= PARAMETERS)
    return NULL;
  const size_t index = (size_t)parameter;
  const size_t count = shapes[index].count;
  const unsigned char *payload = bytevector_payload(header, bytes);
  float initial[WIDTH * WIDTH];
  if (!payload || (bytes != 0 && bytes != (int64_t)(count * sizeof(float))))
    return NULL;
  if (bytes == 0)
    for (size_t i = 0; i < count; ++i) {
      const int centered = (int)((index * 29 + i * 11 + i / 5) % 37) - 18;
      initial[i] = (shapes[index].rank == 1 ? 1.0f : 0.0f) +
                   (float)centered * 0x1p-7f;
    }
  et_f32_tensor *tensor = new_tensor(index, bytes == 0 ? (const void *)initial
                                                      : (const void *)payload);
  et_f32_parameter *result = NULL;
  if (tensor && et_f32_parameter_create_v1(tensor, &result, NULL) != 0)
    result = NULL;
  et_f32_tensor_destroy_v1(&tensor, NULL);
  return result;
}

static int has_plane(const et_kernel_tensor_view_v1 *view, const char *dtype,
                     size_t bytes) {
  return view && view->struct_size == ET_KERNEL_TENSOR_VIEW_V1_0_SIZE &&
         view->data && view->shape && view->rank == 2 &&
         view->shape[0] == 1 && view->shape[1] == TOKENS &&
         view->byte_length == bytes && view->offset_bytes == 0 &&
         view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR &&
         view->dtype && strcmp(view->dtype, dtype) == 0 &&
         view->device && strcmp(view->device, "cpu") == 0;
}

int64_t et_mod5_c2_test_copy_tokens_v1(const et_kernel_tensor_view_v1 *view,
                                       void *header) {
  const size_t size = TOKENS * sizeof(int64_t);
  unsigned char *payload = bytevector_payload(header, (int64_t)size);
  if (!payload || !has_plane(view, "i64", size))
    return -1;
  memcpy(payload, view->data, size);
  return 0;
}

int64_t
et_mod5_c2_test_full_mask_v1(const et_kernel_tensor_view_v1 *view) {
  static const unsigned char mask[TOKENS] = {1, 1};
  return has_plane(view, "bool", sizeof(mask)) &&
         memcmp(view->data, mask, sizeof(mask)) == 0;
}

static int embed(void *header, float output[TOKENS * WIDTH]) {
  const unsigned char *payload =
      bytevector_payload(header, TOKENS * sizeof(int64_t));
  if (!payload)
    return 0;
  for (size_t t = 0; t < TOKENS; ++t) {
    int64_t token;
    memcpy(&token, payload + t * sizeof(token), sizeof(token));
    if (token < 0 || token > 255)
      return 0;
    for (size_t c = 0; c < WIDTH; ++c) {
      const int feature = (int)((token * (int64_t)(c + 3) + 5 * (int64_t)c) % 31);
      output[t * WIDTH + c] = (float)(feature - 15) * 0x1p-5f;
    }
  }
  return 1;
}

int64_t et_mod5_c2_test_gradients_v1(
    void *inputs, void *targets, void *loss_bytes, et_f32_parameter *gamma1,
    et_f32_parameter *wq,
    et_f32_parameter *wk, et_f32_parameter *wv, et_f32_parameter *wo,
    et_f32_parameter *gamma2, et_f32_parameter *wgate, et_f32_parameter *wup,
    et_f32_parameter *wdown) {
  et_f32_parameter *const parameters[] = {gamma1, wq,    wk,  wv,   wo,
                                         gamma2, wgate, wup, wdown};
  float input[TOKENS * WIDTH], target[TOKENS * WIDTH], output[TOKENS * WIDTH];
  float d_input[TOKENS * WIDTH], weights[PARAMETER_FLOATS];
  float gradients[PARAMETER_FLOATS];
  uint64_t loss_bits;
  _Static_assert(sizeof(loss_bits) == sizeof(double), "binary64 loss required");
  unsigned char *loss = bytevector_payload(loss_bytes, sizeof(loss_bits));
  if (!loss || !embed(inputs, input) || !embed(targets, target))
    return -1;
  size_t offset = 0;
  for (size_t i = 0; i < PARAMETERS; ++i) {
    const et_f32_tensor *value = NULL;
    uint32_t words[WIDTH * WIDTH];
    const size_t count = shapes[i].count;
    if (et_f32_parameter_value_tensor_v1(parameters[i], &value, NULL) != 0 ||
        et_f32_tensor_copy_bits_to_v1(value, words, count, NULL) != 0)
      return -1;
    memcpy(weights + offset, words, count * sizeof(words[0]));
    offset += count;
  }

  const et_ac4_config_v1 config = {.shape = {1, 1, TOKENS, 2, 1, 2},
                                  .mode = ET_AC4_FORWARD_BACKWARD,
                                  .policy = ET_AC4_POLICY_NONE,
                                  .epsilon = 0x1p-12f};
  et_ac4_memory_v1 memory;
  if (et_ac4_plan_v1(&config, &memory) != ET_AC4_OK ||
      memory.activation_floats != TOKENS * WIDTH ||
      memory.parameter_floats != offset || offset != PARAMETER_FLOATS)
    return -1;
  float *workspace = malloc(memory.workspace_bytes);
  if (!workspace)
    return -1;
  const et_ac4_io_v1 io = {
      .input = {input, TOKENS * WIDTH},
      .target = {target, TOKENS * WIDTH},
      .parameters = {weights, PARAMETER_FLOATS},
      .output = {output, TOKENS * WIDTH},
      .d_input = {d_input, TOKENS * WIDTH},
      .d_parameters = {gradients, PARAMETER_FLOATS},
      .workspace = workspace,
      .workspace_bytes = memory.workspace_bytes,
  };
  et_ac4_result_v1 result;
  const et_ac4_status run_status = et_ac4_run_v1(&config, &io, &result);
  free(workspace);
  if (run_status != ET_AC4_OK || !isfinite(result.loss) || result.loss <= 0.0)
    return -1;

  et_f32_tensor *numerators[PARAMETERS] = {NULL};
  et_f32_gradient_contribution_v1 contributions[PARAMETERS];
  et_f32_gradient_plan *plan = NULL;
  int32_t status = -1;
  offset = 0;
  for (size_t i = 0; i < PARAMETERS; ++i) {
    numerators[i] = new_tensor(i, gradients + offset);
    if (!numerators[i])
      goto release;
    contributions[i] = (et_f32_gradient_contribution_v1){
        .struct_size = sizeof(contributions[i]),
        .destination = parameters[i],
        .weighted_numerator = numerators[i],
        .expected_ordinal = 0,
    };
    offset += shapes[i].count;
  }
  status = et_f32_gradient_plan_prepare_v1(PARAMETERS, contributions,
                                           0x3f800000u, &plan, NULL);
  if (status == 0)
    status = et_f32_gradient_plan_commit_v1(plan, NULL);
release:
  if (et_f32_gradient_plan_release_v1(&plan, NULL) != 0)
    status = -1;
  for (size_t i = 0; i < PARAMETERS; ++i)
    if (et_f32_tensor_destroy_v1(&numerators[i], NULL) != 0)
      status = -1;
  if (status != 0)
    return -1;
  memcpy(&loss_bits, &result.loss, sizeof(loss_bits));
  for (size_t i = 0; i < sizeof(loss_bits); ++i)
    loss[i] = (unsigned char)(loss_bits >> (8u * i));
  return 0;
}
