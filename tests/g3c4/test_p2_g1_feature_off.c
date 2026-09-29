#define ET_G3C4_OUTPUT_TEXT_TEST_MAIN et_g3c4_p2g1_feature_off_predecessor_main
#include "test_output_text.c"
#undef ET_G3C4_OUTPUT_TEXT_TEST_MAIN

int main(void) {
  const int64_t prompt[2] = {0, 255};
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context = create_generator(owner);
  void *input = create_prompt(prompt, 2);
  CHECK(et_g3c4_private_prompt_prefill_preflight_v1(context, input, 1) ==
        ET_G3C4_SHAPE_MISMATCH);
  CHECK(et_g3c4_pins_idle(&context->pins));
  OK(et_g3c4_private_call_acquire_v1(context, 2, 1));
  CHECK(et_g3c4_private_output_reserve_v1(context, 2) == NULL);
  CHECK(et_g3c4_private_last_error_code_v1() == ET_G3C4_CODE_SHAPE);
  OK(et_g3c4_private_call_abort_v1(context));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  printf("G3-C4 P2/G1 feature-off PASS: checks=%zu\n", checks);
  return 0;
}
