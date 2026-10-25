/* The first real G2 frame may be committed and its pending output aborted.
 * Capture must then see the idle P3 history, cache, RNG and pinned model. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_CPU4_C4_P3_WIRE_PRIVATE 1
#define ET_CPU4_C4_P3_BOXED_BRIDGE_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN cpu4_p3_terminal_predecessor_main
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

/* Test-only synthetic fixed-profile record checks ABI conversion. The live
 * history transition in main remains separate from trained B250 evidence. */
typedef struct { int64_t length; unsigned char bytes[224]; } boxed_record;
typedef struct { int64_t length; unsigned char bytes[4768]; } boxed_values;
typedef struct { int64_t length; unsigned char bytes[32]; } boxed_sha;
typedef struct { int64_t length; unsigned char bytes[96]; } boxed_fp;
typedef struct { int64_t length; unsigned char bytes[400]; } boxed_wire;

static void check_boxed_bridge(et_cpu4_c4_capture *capture) {
  boxed_record record = {224, {0}};
  boxed_values values = {4768, {0}};
  boxed_sha sha = {32, {0}};
  boxed_fp fp = {96, {0}};
  boxed_wire wire = {400, {0}};
  memset(record.bytes, 0xa5, sizeof(record.bytes));
  memset(values.bytes, 0xa5, sizeof(values.bytes));
  OK(et_cpu4_private_c4_capture_copy_record_eshkol_v1(capture, &record));
  OK(et_cpu4_private_c4_capture_copy_parameters_eshkol_v1(capture, &values));
  CHECK(record.length == 224 && values.length == 4768);
  CHECK(((et_cpu4_c4_capture_record_v1 *)record.bytes)->size == 224u);
  CHECK(memcmp(values.bytes, ((cpu4_capture_provenance *)cpu4_captures)->payload->values,
               sizeof(values.bytes)) == 0);
  boxed_record saved_record = record;
  boxed_values saved_values = values;
  record.length = 223;
  CHECK(et_cpu4_private_c4_capture_copy_record_eshkol_v1(capture, &record)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(record.length == 223);
  CHECK(memcmp(record.bytes, saved_record.bytes, sizeof(record.bytes)) == 0);
  values.length = -1;
  CHECK(et_cpu4_private_c4_capture_copy_parameters_eshkol_v1(capture, &values)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(values.length == -1 &&
        memcmp(values.bytes, saved_values.bytes, sizeof(values.bytes)) == 0);
  record = saved_record;
  values = saved_values;

  cpu4_capture_payload profile = {0};
  memcpy(&profile.record, record.bytes, 224u);
  memcpy(profile.values, values.bytes, sizeof(profile.values));
  profile.record.cache_capacity = 4u;
  profile.record.ids[0] = 3;
  profile.record.ids[1] = 197;
  profile.record.ids[2] = 3;
  profile.record.ids[3] = 0;
  profile.record.origins[0] = 1u;
  profile.record.origins[1] = 1u;
  profile.record.origins[2] = 2u;
  profile.record.origins[3] = 0u;
  profile.record.policy[0] = 0;
  profile.record.policy[1] = INT64_C(0x3f800000);
  profile.record.policy[2] = 256;
  profile.record.policy[3] = INT64_C(0x3f800000);
  profile.record.policy[4] = 2;
  profile.record.policy[5] = 41;
  profile.record.rng[0] = 1;
  profile.record.rng[1] = 41;
  profile.record.rng[2] = 0;
  profile.record.rng[3] = 0;
  cpu4_capture_digest(&profile, profile.record.digest);
  memcpy(record.bytes, &profile.record, 224u);
  memcpy(fp.bytes, cpu4_p3_t1, 96u);
  for (size_t i = 0u; i < sizeof(sha.bytes); ++i) sha.bytes[i] = (unsigned char)i;
  OK(et_cpu4_private_c4_p3_wire_encode_eshkol_v1(
      &record, &values, &sha, &fp, &wire));
  CHECK(wire.length == 400 && memcmp(wire.bytes, cpu4_p3_magic, 8u) == 0);
  CHECK(memcmp(wire.bytes + 240u, sha.bytes, 32u) == 0);
  boxed_wire saved_wire = wire;
  int64_t *headers[5] = {&record.length, &values.length, &sha.length,
                         &fp.length, &wire.length};
  for (size_t i = 0u; i < 5u; ++i) {
    const int64_t original = *headers[i];
    *headers[i] = original - 1;
    CHECK(et_cpu4_private_c4_p3_wire_encode_eshkol_v1(
        &record, &values, &sha, &fp, &wire) == ET_G3C4_INVALID_ARGUMENT);
    CHECK(memcmp(wire.bytes, saved_wire.bytes, sizeof(wire.bytes)) == 0);
    CHECK(*headers[i] == original - 1);
    *headers[i] = original;
  }
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  const int64_t prompt[2] = {3, 197};
  bridge_carrier carrier = new_carrier();
  int64_t successor[4];
  begin_eos_pair(owner, prompt, 0, 255, &context, &input, &output);
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
  CHECK(context->owner->active == NULL);
  CHECK(context->p2g2_prefix_committed == 0u);
  CHECK(context->cpu4_history.count == 3u);
  CHECK(context->cpu4_history.ids[2] == selected);
  CHECK(memcmp(context->generator_rng_words, successor, sizeof(successor)) == 0);
  /* The deterministic native fixture may leave RNG unchanged. The accepted
   * property is exact publication of the sampled successor words. */
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
  check_boxed_bridge(capture);
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  printf("CPU4 C4 genuine P3 first-commit-abort capture PASS: checks=%zu\n", checks);
  return 0;
}
