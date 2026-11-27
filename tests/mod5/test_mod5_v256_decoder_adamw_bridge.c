#include "mod5_v256_decoder_adamw_bridge.c"

#include <stdio.h>

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      fprintf(stderr, "%d: %s\n", __LINE__, #condition);                          \
      return EXIT_FAILURE;                                                      \
    }                                                                           \
  } while (0)

int main(void) {
  uint64_t shape[2] = {1, TOKENS};
  int64_t ids[TOKENS] = {0, 255, 3, 3, 5, 6, 127, 200};
  uint8_t mask[TOKENS] = {1, 1, 1, 1, 1, 1, 1, 1};
  uint8_t output[TOKENS] = {0};
  et_kernel_tensor_view_v1 view = {
      .struct_size = ET_KERNEL_TENSOR_VIEW_V1_0_SIZE,
      .dtype = "i64", .device = "cpu", .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .rank = 2, .shape = shape, .data = ids, .byte_length = sizeof(ids),
  };
  CHECK(et_mod5_v256_adamw_row_v1(&view, 0, output) == 0);
  for (size_t i = 0; i < TOKENS; ++i)
    CHECK(output[i] == ids[i]);
  ids[TOKENS - 1] = 256;
  CHECK(et_mod5_v256_adamw_row_v1(&view, 0, output) != 0 && output[TOKENS - 1] == 200);
  view.dtype = "bool";
  view.data = mask;
  view.byte_length = sizeof(mask);
  CHECK(et_mod5_v256_adamw_row_v1(&view, 1, NULL) == 0);
  mask[TOKENS - 1] = 0;
  CHECK(et_mod5_v256_adamw_row_v1(&view, 1, NULL) != 0);
  mask[TOKENS - 1] = 1;
  shape[0] = 2;
  CHECK(et_mod5_v256_adamw_row_v1(&view, 1, NULL) != 0);

  enum { SLOT_WORDS = 128 };
  const uint64_t dimensions[2] = {8, 16};
  uint32_t weights[SLOT_WORDS];
  uint32_t gradient[SLOT_WORDS];
  for (size_t i = 0; i < SLOT_WORDS; ++i) {
    weights[i] = UINT32_C(0x3e800000);
    gradient[i] = UINT32_C(0x3d000000);
  }
  gradient[SLOT_WORDS - 1] = UINT32_C(0x80000000);
  et_f32_parameter *parameter = et_mod5_v256_test_parameter_v1(0, weights, SLOT_WORDS);
  et_f32_parameter *slots[SLOTS] = {NULL};
  CHECK(et_mod5_v256_adamw_set_slot_v1(slots, 0, parameter) == 0 && slots[0] == parameter);
  CHECK(et_mod5_v256_adamw_set_slot_v1(slots, SLOTS, parameter) != 0);
  CHECK(et_f32_parameter_bind_identity_v1(parameter, parameter, NULL) == 0);
  CHECK(et_mod5_v256_adamw_gradient_v1(parameter, NULL, 0) == 0);

  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  CHECK(et_o2_optimizer_builder_create_v1(1, ET_O2_CLIP_NONE, 0,
      ET_O2_SCHEDULE_CONSTANT, 0, 0, UINT32_C(0x3f800000), &builder, NULL) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(builder, 0, parameter, parameter,
      0x3c800000u, 0x3f000000u, 0x3f400000u, 0x35800000u, 0x3c800000u, NULL) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &optimizer, NULL) == 0);

  et_f32_tensor *numerator = NULL;
  et_f32_gradient_plan *plan = NULL;
  CHECK(et_f32_tensor_create_v1(2, dimensions, &numerator, NULL) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(numerator, gradient, SLOT_WORDS, NULL) == 0);
  const et_f32_gradient_contribution_v1 contribution = {
      .struct_size = sizeof(contribution), .destination = parameter,
      .weighted_numerator = numerator, .expected_ordinal = 0,
  };
  CHECK(et_f32_gradient_plan_prepare_v1(1, &contribution, UINT32_C(0x3f800000),
                                       &plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, NULL) == 0);
  CHECK(et_f32_tensor_destroy_v1(&numerator, NULL) == 0);
  CHECK(et_mod5_v256_adamw_gradient_v1(parameter, gradient, SLOT_WORDS - 1) != 0);
  CHECK(et_mod5_v256_adamw_gradient_v1(parameter, gradient, SLOT_WORDS) == 0);
  CHECK(gradient[SLOT_WORDS - 1] == 0);
  CHECK(et_o2_optimizer_step_v1(optimizer, NULL) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(optimizer, NULL) == 0);
  CHECK(et_mod5_v256_adamw_gradient_v1(parameter, NULL, 0) == 0);

  et_o2_optimizer_state *state = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(optimizer, &state, NULL) == 0);
  uint32_t observed[SLOT_WORDS];
  CHECK(et_mod5_v256_test_parameter_bits_v1(0, parameter, observed, SLOT_WORDS) == 0);
  CHECK(memcmp(weights, observed, sizeof(weights)) != 0);
  CHECK(et_mod5_v256_adamw_moment_v1(state, 0, ET_O2_MOMENT_EXP_AVG,
                                   observed, SLOT_WORDS) == 0);
  CHECK(observed[0] == UINT32_C(0x3c800000));
  CHECK(et_mod5_v256_adamw_moment_v1(state, 0, ET_O2_MOMENT_EXP_AVG_SQ,
                                   observed, SLOT_WORDS) == 0);
  CHECK(observed[0] == UINT32_C(0x39800000));
  CHECK(et_mod5_v256_adamw_moment_v1(state, 0, 2, observed, SLOT_WORDS) != 0);
  CHECK(et_o2_optimizer_state_release_v1(state, NULL) == 0);
  puts("MOD5 V256 decoder AdamW native adapters PASS");
  return EXIT_SUCCESS;
}
