/* The first real G2 frame may be committed and its pending output aborted.
 * Capture must then see the idle P3 history, cache, RNG and pinned model. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN cpu4_p3_terminal_predecessor_main
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  const int64_t prompt[2] = {3, 197};
  bridge_carrier carrier = new_carrier();
  int64_t initial_rng[4], successor[4];
  begin_eos_pair(owner, prompt, 0, 255, &context, &input, &output);
  memcpy(initial_rng, context->generator_rng_words, sizeof(initial_rng));
  OK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, &carrier, 16));
  CHECK(context->token_frame_candidate != 255);
  int64_t selected = context->token_frame_candidate;
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  terminal_raw first = raw_id(selected);
  OK(et_g3c4_private_p2g2_prefix_commit_v1(context, output, &first));
  CHECK(context->cpu4_history.count == 3u &&
        context->cpu4_history.prompt_count == 2u &&
        context->cpu4_history.generated_count == 1u &&
        context->cpu4_history.ids[2] == selected &&
        context->cpu4_history.origin[2] == 2u &&
        context->cpu4_history.commit_epoch == 2u);
  CHECK(memcmp(context->generator_rng_words, successor, sizeof(successor)) == 0);
  CHECK(et_cpu4_private_c4_capture_create_v1(context) == NULL);
  OK(et_g3c4_private_call_abort_v1(context));
  check_dead_output(output);
  CHECK(context->owner->active == NULL &&
        context->p2g2_prefix_committed == 0u &&
        context->cpu4_history.count == 3u &&
        context->cpu4_history.ids[2] == selected &&
        memcmp(context->generator_rng_words, successor, sizeof(successor)) == 0 &&
        memcmp(context->generator_rng_words, initial_rng, sizeof(successor)) != 0);
  et_cpu4_c4_capture *capture = et_cpu4_private_c4_capture_create_v1(context);
  CHECK(capture != NULL);
  et_cpu4_c4_capture_record_v1 record;
  _Alignas(float) unsigned char values[4768];
  OK(et_cpu4_private_c4_capture_copy_record_v1(capture, &record, sizeof(record)));
  OK(et_cpu4_private_c4_capture_copy_parameters_v1(capture, values, sizeof(values)));
  CHECK(record.count == 3u && record.cache_capacity == 4u &&
        record.prompt_count == 2u && record.generated_count == 1u &&
        record.manual_count == 0u && record.generated_eos_mask == 0u &&
        record.commit_epoch == 2u && record.ids[0] == 3 &&
        record.ids[1] == 197 && record.ids[2] == selected &&
        record.origins[0] == 1u && record.origins[1] == 1u &&
        record.origins[2] == 2u &&
        memcmp(record.rng, successor, sizeof(successor)) == 0 &&
        record.values_bytes == sizeof(values));
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  printf("CPU4 C4 genuine P3 first-commit-abort capture PASS: checks=%zu\n", checks);
  return 0;
}
