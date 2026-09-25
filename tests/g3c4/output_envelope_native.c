#define ET_G3C4_OUTPUT_TEXT_TEST_MAIN \
  et_g3c4_output_text_predecessor_main
#include "test_output_text.c"
#undef ET_G3C4_OUTPUT_TEXT_TEST_MAIN

int64_t et_g3c4_output_envelope_test_prepare_v1(
    void *context_candidate, void *output_candidate,
    int64_t generated) {
  const int64_t tokens[2] = {83, 89};
  float prefill_logits[256], token_logits[256];
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  if (context == NULL || output == NULL ||
      (generated != 0 && generated != 1) ||
      output->parent_ctx != context ||
      output->generated_length != generated)
    return -1;
  et_g3c4_input_internal *input =
      create_prompt(tokens, generated == 0 ? 2 : 1);
  OK(et_g3c4_private_prompt_prefill_v1(
      context, input, prefill_logits));
  if (generated == 1) {
    int64_t token;
    OK(et_g3c4_private_token_frame_begin_last_v1(
        context, prefill_logits, &token));
    OK(et_g3c4_private_token_forward_v1(
        context, token, token_logits));
  }
  OK(et_g3c4_private_tensor_release_v1(input));
  return 0;
}

int64_t et_g3c4_output_envelope_test_field_v1(
    void *candidate, int64_t field) {
  et_g3c4_output_internal *output = (et_g3c4_output_internal *)candidate;
  if (output == NULL || output->transport.magic != ET_G3C4_OUTPUT_MAGIC)
    return -1;
  switch (field) {
    case 0: return (int64_t)output->transport.state;
    case 1: return (int64_t)output->numeric_ready;
    case 2: return (int64_t)output->ids_copied;
    case 3: return (int64_t)output->text_ready;
    case 4: return output->generated_length;
    default: return -1;
  }
}

int64_t et_g3c4_output_envelope_test_owner_fail_after_v1(
    int64_t allowed) {
  if (allowed < 0) return -1;
  et_g3c4_context_allocation_limit = (size_t)allowed;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}

int64_t et_g3c4_output_envelope_test_owner_fail_reset_v1(void) {
  et_g3c4_context_allocation_limit = SIZE_MAX;
  et_g3c4_context_successful_allocations = 0u;
  return 0;
}
