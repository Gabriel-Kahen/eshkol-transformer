#define ET_G3C4_P2_G2_TERMINAL_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TESTING 1
#define ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE 1
#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_EOS_TEST_MAIN terminal_predecessor_main
#include "test_p2_g2_eos_first_frame.c"

#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

static int fail_terminal_copy, fail_terminal_cache_commit;
int32_t __real_et_i64_tensor_copy_from_v1(
    et_i64_tensor *handle, const int64_t *source, size_t count,
    et_i64_tensor_error *error);
int32_t __wrap_et_i64_tensor_copy_from_v1(
    et_i64_tensor *handle, const int64_t *source, size_t count,
    et_i64_tensor_error *error) {
  if (fail_terminal_copy) return -1;
  return __real_et_i64_tensor_copy_from_v1(handle, source, count, error);
}
int32_t __real_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);
int32_t __wrap_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (fail_terminal_cache_commit) return -1;
  return __real_et_a2_kv_cache_transaction_commit_v1(transaction, error);
}

static int capture_terminal_t4;
static cache_snapshot terminal_t4;

int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (capture_terminal_t4 && transaction != NULL && *transaction != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    OK(et_a2_kv_cache_transaction_view_begin_v1(
        *transaction, 0u, &view, &local));
    OK(et_a2_kv_cache_transaction_view_tensors_v1(
        view, &keys, &values, &lengths, &keep, &local));
    memcpy(terminal_t4.keys, keys->data, sizeof(terminal_t4.keys));
    memcpy(terminal_t4.values, values->data, sizeof(terminal_t4.values));
    terminal_t4.length = *(const int64_t *)lengths->data;
    memcpy(terminal_t4.keep, keep->data, sizeof(terminal_t4.keep));
    OK(et_a2_kv_cache_transaction_view_end_v1(&view, &local));
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(transaction, error);
}

typedef struct frozen_parameters {
  unsigned char values[4768], gradients[4768];
  uint8_t present[14];
  et_f32_gradient_metadata_v1 metadata[14];
} frozen_parameters;

static void freeze_parameters(et_g3c4_model_owner_internal *owner,
                              frozen_parameters *snapshot) {
  memset(snapshot, 0, sizeof(*snapshot));
  size_t offset = 0u;
  et_f32_tensor_error error;
  for (size_t i = 0u; i < 14u; i++) {
    et_f32_parameter *parameter = owner->parameters[i];
    size_t bytes = parameter->value->byte_length;
    CHECK(bytes <= sizeof(snapshot->values) - offset);
    memcpy(snapshot->values + offset, parameter->value->data, bytes);
    if (parameter->gradient != NULL) {
      CHECK(parameter->gradient->byte_length == bytes);
      snapshot->present[i] = 1u;
      memcpy(snapshot->gradients + offset, parameter->gradient->data, bytes);
    }
    snapshot->metadata[i].struct_size = sizeof(snapshot->metadata[i]);
    OK(et_f32_parameter_gradient_metadata_v1(
        parameter, &snapshot->metadata[i], &error));
    offset += bytes;
  }
  CHECK(offset == sizeof(snapshot->values));
}

static void check_frozen_parameters(et_g3c4_model_owner_internal *owner,
                                    const frozen_parameters *snapshot) {
  frozen_parameters now;
  freeze_parameters(owner, &now);
  CHECK(memcmp(&now, snapshot, sizeof(now)) == 0);
}

static void start_prefix(et_g3c4_model_owner_internal *owner,
                         const int64_t prompt[2], int categorical,
                         int64_t eos, et_g3c4_context_internal **context,
                         et_g3c4_input_internal **input,
                         et_g3c4_output_internal **output) {
  if (eos >= 0) begin_eos_pair(owner, prompt, categorical, eos,
                                context, input, output);
  else active_pair(owner, categorical, prompt, context, input, output, NULL);
  bridge_carrier carrier = new_carrier();
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      *context, *input, &carrier, 16));
  struct { int64_t length; unsigned char byte; } raw =
      {1, carrier.bytes[0]};
  OK(et_g3c4_private_p2g2_prefix_commit_v1(
      *context, *output, &raw));
}

typedef struct terminal_raw {
  int64_t length;
  unsigned char byte;
  unsigned char guard[7];
} terminal_raw;

typedef struct terminal_snapshot {
  int64_t length;
  unsigned char bytes[64];
} terminal_snapshot;

static terminal_raw raw_id(int64_t id) {
  terminal_raw raw;
  memset(&raw, 0xa5, sizeof(raw));
  raw.length = 1;
  raw.byte = (unsigned char)id;
  return raw;
}

static int64_t word(const terminal_snapshot *snapshot, size_t index) {
  uint64_t value = 0;
  for (size_t byte = 0; byte < 8u; byte++)
    value |= (uint64_t)snapshot->bytes[index * 8u + byte] << (byte * 8u);
  return (int64_t)value;
}

static void terminal_ready(et_g3c4_model_owner_internal *owner,
                           const int64_t prompt[2], int first_eos,
                           et_g3c4_context_internal **context,
                           et_g3c4_input_internal **input,
                           et_g3c4_output_internal **output,
                           terminal_raw *first, terminal_raw *second) {
  bridge_carrier carrier = new_carrier();
  if (first_eos) {
    et_g3c4_context_internal *probe;
    et_g3c4_input_internal *probe_input;
    et_g3c4_output_internal *probe_output;
    active_pair(owner, 0, prompt, &probe, &probe_input, &probe_output, NULL);
    OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        probe, probe_input, &carrier, 16));
    int64_t eos = probe->token_frame_candidate;
    close_pair(probe, probe_input, probe_output);
    begin_eos_pair(owner, prompt, 0, eos, context, input, output);
    carrier = new_carrier();
    OK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
        *context, *input, &carrier, 16));
    *first = raw_id((*context)->token_frame_candidate);
    *second = raw_id(0);
  } else {
    start_prefix(owner, prompt, 1, -1, context, input, output);
    *first = raw_id((*context)->p2g2_prefix_token);
    OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
        *context, *output, &carrier, 16));
    *second = raw_id((*context)->token_frame_candidate);
  }
}

static void terminal_failure_matrix(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  for (int first_eos = 0; first_eos <= 1; first_eos++)
    for (int cut = 0; cut < 19; cut++) {
      et_g3c4_context_internal *context;
      et_g3c4_input_internal *input;
      et_g3c4_output_internal *output;
      terminal_raw first, second;
      terminal_ready(owner, prompt, first_eos, &context, &input, &output,
                     &first, &second);
      int64_t rng[4];
      memcpy(rng, context->generator_rng_words, sizeof(rng));
      et_i64_tensor_borrow *id_borrow = NULL;
      et_i64_tensor_error id_error;
      et_a2_kv_cache_transaction_view *cache_view = NULL;
      et_kernel_error cache_error;
      int64_t saved_policy = context->generator_policy[5];
      void *requested_output = output;
      void *requested_first = &first;
      void *requested_second = first_eos ? NULL : &second;
      int64_t first_bytes = 9, second_bytes = first_eos ? 0 : 9;
      switch (cut) {
        case 0: first_bytes = 8; break;
        case 1: requested_first = (unsigned char *)&first + 1; break;
        case 2: first.length = 2; break;
        case 3: first.byte ^= 1u; break;
        case 4: requested_first = output; break;
        case 5: requested_output = input; break;
        case 6: context->generator_policy[5] = 256; break;
        case 7: context->token_frame_successor[0] = 2; break;
        case 8: ((unsigned char *)context->p2g2_frame_keys)[0] ^= 1u; break;
        case 9:
          OK(et_i64_tensor_borrow_begin_v1(
              output->ids, &id_borrow, &id_error));
          break;
        case 10:
          OK(et_a2_kv_cache_transaction_view_begin_v1(
              context->token_frame_transaction, 0u, &cache_view,
              &cache_error));
          break;
        case 11: requested_second = &first; second_bytes = 9; break;
        case 12: requested_second = &second; second_bytes = first_eos ? 9 : 8; break;
        case 13:
          if (first_eos) requested_second = &second;
          else second.byte ^= 1u;
          break;
        case 14: et_a2_kv_cache_test_fail_alloc_after_v1(0u); break;
        case 15: context->prefill_binding_values[0] ^= 1u; break;
        case 16: context->token_frame_candidate ^= 1; break;
        case 17:
          if (first_eos) context->p2g2_prefix_committed = 1u;
          else context->p2g2_prefix_raw ^= 1u;
          break;
        case 18: context->generator_policy[4] = 1; break;
      }
      CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
          context, requested_output, requested_first, first_bytes,
          requested_second, second_bytes) != 0);
      CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY &&
            context->token_frame_transaction != NULL &&
            context->owner->active == context &&
            output->transport.state == 0u && output->parent_ctx == context);
      CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
      const int64_t *ids = et_i64_tensor_test_data_storage_v1(output->ids);
      CHECK(ids != NULL && ids[0] == 0 && ids[1] == 0);
      assert_pending(output);
      if (id_borrow != NULL)
        OK(et_i64_tensor_borrow_end_v1(&id_borrow, &id_error));
      if (cache_view != NULL)
        OK(et_a2_kv_cache_transaction_view_end_v1(
            &cache_view, &cache_error));
      et_a2_kv_cache_test_reset_allocator_v1();
      context->generator_policy[5] = saved_policy;
      if (cut == 7) context->token_frame_successor[0] = 1;
      if (cut == 8) ((unsigned char *)context->p2g2_frame_keys)[0] ^= 1u;
      if (cut == 15) context->prefill_binding_values[0] ^= 1u;
      if (cut == 16) context->token_frame_candidate ^= 1;
      if (cut == 17) {
        if (first_eos) context->p2g2_prefix_committed = 0u;
        else context->p2g2_prefix_raw ^= 1u;
      }
      if (cut == 18) context->generator_policy[4] = 2;
      if (cut == 9 || cut == 10 || cut == 14) {
        OK(et_g3c4_private_p2g2_terminal_commit_v1(
            context, output, &first, 9, first_eos ? NULL : &second,
            first_eos ? 0 : 9));
        OK(et_g3c4_private_tensor_release_v1(input));
        OK(et_g3c4_private_output_release_v1(output));
        OK(et_g3c4_private_generator_close_v1(context));
      } else {
        OK(et_g3c4_private_call_abort_v1(context));
        cache_snapshot rolled_back;
        snapshot_cache(context->cache, &rolled_back);
        CHECK(rolled_back.length == (first_eos ? 2 : 3));
        CHECK(memcmp(rolled_back.keep,
                     first_eos ? (const uint8_t[4]){1, 1, 0, 0} :
                                 (const uint8_t[4]){1, 1, 1, 0}, 4u) == 0);
        CHECK(memcmp(rng, context->generator_rng_words, sizeof(rng)) == 0);
        check_dead_output(output);
        OK(et_g3c4_private_tensor_release_v1(input));
        OK(et_g3c4_private_generator_close_v1(context));
      }
    }
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  terminal_raw first = raw_id(0);
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, NULL, 0) != 0);
  assert_pending(output);
  bridge_carrier carrier = new_carrier();
  OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, &carrier, 16));
  first = raw_id(context->token_frame_candidate);
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, NULL, 0) != 0);
  close_pair(context, input, output);
}

static void terminal_a2_witness(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context, *foreign;
  et_g3c4_input_internal *input, *foreign_input;
  et_g3c4_output_internal *output, *foreign_output;
  terminal_raw first, second, foreign_first, foreign_second;
  terminal_ready(owner, prompt, 1, &context, &input, &output,
                 &first, &second);
  static unsigned char foreign_identities[14];
  et_g3c4_model_owner_internal *other_owner =
      et_g3c4_private_model_owner_create_seeded_v1(1729);
  CHECK(other_owner != NULL);
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_bind_v1(
        other_owner, index, &foreign_identities[(size_t)index]));
  for (int64_t index = 0; index < 14; index++)
    OK(et_g3c4_private_model_owner_initialize_v1(other_owner, index));
  OK(et_g3c4_private_model_owner_prepare_seal_v1(other_owner));
  OK(et_g3c4_private_model_owner_commit_seal_v1(other_owner));
  terminal_ready(other_owner, prompt, 1, &foreign, &foreign_input,
                 &foreign_output, &foreign_first, &foreign_second);
  CHECK(memcmp(context->p2g2_frame_keys, foreign->p2g2_frame_keys,
               sizeof(context->p2g2_frame_keys)) == 0);
  CHECK(memcmp(context->p2g2_frame_values, foreign->p2g2_frame_values,
               sizeof(context->p2g2_frame_values)) == 0);
  et_a2_kv_cache_transaction *authentic = context->token_frame_transaction;
  et_a2_kv_cache_transaction *other = foreign->token_frame_transaction;
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, context->cache, 2) == 1);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, context->cache, 3) == 0);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, foreign->cache, 2) == 0);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      other, context->cache, 2) == 0);
  et_a2_kv_cache_test_fail_alloc_after_v1(0u);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, context->cache, 2) == 1);
  et_a2_kv_cache_test_reset_allocator_v1();
  et_a2_kv_cache_transaction_view *held = NULL;
  et_kernel_error error;
  OK(et_a2_kv_cache_transaction_view_begin_v1(
      authentic, 0u, &held, &error));
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, context->cache, 2) == 0);
  OK(et_a2_kv_cache_transaction_view_end_v1(&held, &error));
  context->token_frame_transaction = other;
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, NULL, 0) != 0);
  CHECK(context->token_frame_state == ET_G3C4_TOKEN_FRAME_READY &&
        output->length == 0 && foreign->token_frame_transaction == other);
  context->token_frame_transaction = authentic;
  close_pair(context, input, output);
  CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
      authentic, foreign->cache, 2) == 0);
  close_pair(foreign, foreign_input, foreign_output);

  active_pair(owner, 0, prompt, &context, &input, &output, NULL);
  float logits[256];
  OK(et_g3c4_private_prompt_prefill_v1(context, input, logits));
  static const uint64_t count_shape[1] = {1u};
  static const uint64_t stage_shape[4] = {1u, 2u, 2u, 2u};
  float staged_k[8] = {0}, staged_v[8] = {0};
  et_kernel_tensor_view_v1 keys =
      et_g3c4_view(staged_k, sizeof(staged_k), "f32", 4u, stage_shape);
  et_kernel_tensor_view_v1 values =
      et_g3c4_view(staged_v, sizeof(staged_v), "f32", 4u, stage_shape);
  for (int64_t count = 1; count <= 2; count++) {
    et_kernel_tensor_view_v1 counts = et_g3c4_view(
        &count, sizeof(count), "i64", 1u, count_shape);
    et_a2_kv_cache_transaction *transaction = NULL;
    OK(et_a2_kv_cache_transaction_begin_v1(
        context->cache, 2u, &counts, &transaction, &error));
    OK(et_a2_kv_cache_transaction_stage_layer_v1(
        transaction, 0u, &keys, &values, &error));
    CHECK(et_a2_kv_cache_private_terminal_transaction_witness_v1(
        transaction, context->cache, 2) == 0);
    OK(et_a2_kv_cache_transaction_abort_v1(&transaction, &error));
  }
  close_pair(context, input, output);
}

static void terminal_failstop(et_g3c4_model_owner_internal *owner) {
  const int64_t prompt[2] = {7, 11};
  for (int cut = 0; cut < 3; cut++) {
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
      struct rlimit limit = {0u, 0u};
      (void)setrlimit(RLIMIT_CORE, &limit);
      et_g3c4_context_internal *context;
      et_g3c4_input_internal *input;
      et_g3c4_output_internal *output;
      terminal_raw first, second;
      terminal_ready(owner, prompt, 0, &context, &input, &output,
                     &first, &second);
      fail_terminal_copy = cut == 0;
      fail_terminal_cache_commit = cut == 1;
      et_g3c4_terminal_test_fail_drain = cut == 2;
      (void)et_g3c4_private_p2g2_terminal_commit_v1(
          context, output, &first, 9, &second, 9);
      _exit(99);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
  }
}

static void terminal_success(et_g3c4_model_owner_internal *owner,
                             const int64_t prompt[2], int categorical,
                             int first_eos, int second_eos) {
  et_g3c4_context_internal *context, *probe;
  et_g3c4_input_internal *input, *probe_input;
  et_g3c4_output_internal *output, *probe_output;
  bridge_carrier carrier = new_carrier();
  terminal_raw first, second;
  terminal_snapshot snapshot;
  cache_snapshot actual, independent;
  frozen_parameters parameters;
  int64_t selected, successor[4], original_rng[4];
  float second_logits[256], first_logits[256];
  int64_t eos = -1;
  if (first_eos) {
    active_pair(owner, categorical, prompt, &probe, &probe_input,
                &probe_output, NULL);
    OK(et_g3c4_private_p2g2_first_frame_carrier_v1(
        probe, probe_input, &carrier, 16));
    eos = probe->token_frame_candidate;
    close_pair(probe, probe_input, probe_output);
  } else if (second_eos) {
    start_prefix(owner, prompt, categorical, -1,
                 &probe, &probe_input, &probe_output);
    carrier = new_carrier();
    OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
        probe, probe_output, &carrier, 16));
    eos = probe->token_frame_candidate;
    CHECK(eos != probe->p2g2_prefix_token);
    close_pair(probe, probe_input, probe_output);
  }
  if (first_eos) {
    begin_eos_pair(owner, prompt, categorical, eos,
                   &context, &input, &output);
    carrier = new_carrier();
    OK(et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
        context, input, &carrier, 16));
    selected = context->token_frame_candidate;
    CHECK(selected == eos);
    first = raw_id(selected);
    second = raw_id(0);
  } else {
    start_prefix(owner, prompt, categorical, eos,
                 &context, &input, &output);
    first = raw_id(context->p2g2_prefix_token);
    carrier = new_carrier();
    OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
        context, output, &carrier, 16));
    selected = context->token_frame_candidate;
    second = raw_id(selected);
    memcpy(second_logits, context->p2g2_prefix_next_logits,
           sizeof(second_logits));
    if (second_eos) CHECK(selected == eos);
  }
  freeze_parameters(owner, &parameters);
  memcpy(successor, context->token_frame_successor, sizeof(successor));
  memcpy(original_rng, context->generator_rng_words, sizeof(original_rng));
  CHECK(et_g3c4_private_call_finish_v1(context) != 0);
  CHECK(et_g3c4_private_output_prepare_v1(context, output) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, first_eos ? NULL : &second,
      first_eos ? 0 : 9) == 0);
  CHECK(context->transport.busy == ET_G3C4_CALL_IDLE &&
        context->owner->active == NULL &&
        context->token_frame_state == ET_G3C4_TOKEN_FRAME_IDLE &&
        context->p2g2_prefix_committed == 0u &&
        context->pins.held_mask == 0u);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_TERMINAL &&
        output->parent_ctx == NULL && output->generated_length == 2 &&
        output->length == (first_eos ? 1 : 2) &&
        output->cache_length == (first_eos ? 3 : 4));
  CHECK(memcmp(output->rng, successor, sizeof(successor)) == 0 &&
        memcmp(context->generator_rng_words, successor, sizeof(successor)) == 0);
  CHECK((memcmp(original_rng, successor, sizeof(successor)) == 0) ==
        !categorical);
  const int64_t *ids = et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(ids != NULL && ids[0] == (int64_t)first.byte &&
        ids[1] == (first_eos ? 0 : (int64_t)second.byte));
  snapshot_cache(context->cache, &actual);
  CHECK(actual.length == output->cache_length);
  CHECK(memcmp(actual.keep,
               first_eos ? (const uint8_t[4]){1, 1, 1, 0} :
                           (const uint8_t[4]){1, 1, 1, 1}, 4u) == 0);
  check_frozen_parameters(owner, &parameters);
  et_g3c4_context_internal *reference = create_generator(owner);
  float reference_logits[1024];
  const int64_t prefix[4] = {prompt[0], prompt[1], ids[0], ids[1]};
  OK(et_g3c4_private_call_acquire_v1(reference, 0, 0));
  if (first_eos) {
    OK(et_g3c4_private_prefill3_v1(reference, prefix, reference_logits));
    snapshot_cache(reference->cache, &independent);
  } else {
    capture_terminal_t4 = 1;
    OK(et_g3c4_private_full_prefix_forward_v1(
        reference, prefix, reference_logits));
    capture_terminal_t4 = 0;
    independent = terminal_t4;
  }
  check_cache_snapshot_equal(&actual, &independent);
  OK(et_g3c4_private_call_finish_v1(reference));
  OK(et_g3c4_private_generator_close_v1(reference));
  reference = p2g2_generator(owner, categorical);
  et_g3c4_input_internal *oracle_input = create_prompt(prompt, 2);
  OK(et_g3c4_private_prompt_prefill_preflight_v1(
      reference, oracle_input, 2));
  OK(et_g3c4_private_call_acquire_v1(reference, 2, 2));
  et_g3c4_output_internal *oracle_output =
      et_g3c4_private_output_reserve_v1(reference, 2);
  CHECK(oracle_output != NULL);
  int64_t first_before[4];
  memcpy(first_before, reference->generator_rng_words,
         sizeof(first_before));
  OK(et_g3c4_private_prompt_prefill_v1(
      reference, oracle_input, first_logits));
  print_oracle_case(categorical ? "categorical" : "greedy",
                    first_logits, first.byte, first_before,
                    first_eos ? successor : original_rng);
  if (!first_eos)
    print_oracle_case(categorical ? "categorical" : "greedy",
                      second_logits, second.byte, original_rng, successor);
  close_pair(reference, oracle_input, oracle_output);
  memset(&snapshot, 0xa5, sizeof(snapshot));
  snapshot.length = 64;
  terminal_snapshot unchanged = snapshot;
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, 71) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      input, &snapshot, sizeof(snapshot)) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, output, sizeof(snapshot)) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, (unsigned char *)&snapshot + 1u,
      sizeof(snapshot)) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  snapshot.length = 63;
  unchanged = snapshot;
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, sizeof(snapshot)) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  snapshot.length = 64;
  int64_t *owned_ids = (int64_t *)et_i64_tensor_test_data_storage_v1(
      output->ids);
  int64_t saved_second = owned_ids[1];
  owned_ids[1] = first_eos ? 1 : 256;
  unchanged = snapshot;
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, sizeof(snapshot)) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  owned_ids[1] = saved_second;
  OK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, sizeof(snapshot)));
  CHECK(word(&snapshot, 0) == ids[0] && word(&snapshot, 1) == ids[1] &&
        word(&snapshot, 2) == output->length &&
        word(&snapshot, 3) == output->cache_length);
  for (size_t index = 0u; index < 4u; index++)
    CHECK(word(&snapshot, index + 4u) == successor[index]);
  CHECK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, first_eos ? NULL : &second,
      first_eos ? 0 : 9) != 0);
  CHECK(et_g3c4_private_call_abort_v1(context) != 0);
  et_i64_tensor_borrow *held = NULL;
  et_i64_tensor_error held_error;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &held, &held_error));
  CHECK(et_g3c4_private_output_release_v1(output) != 0);
  CHECK(output->transport.state == ET_G3C4_OUTPUT_TERMINAL);
  OK(et_i64_tensor_borrow_end_v1(&held, &held_error));
  CHECK(et_g3c4_private_output_release_v1(input) != 0);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_output_release_v1(output));
  OK(et_g3c4_private_output_release_v1(output));
  CHECK(output->transport.state == ET_G3C4_CONTEXT_DEAD);
  unchanged = snapshot;
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, sizeof(snapshot)) != 0);
  CHECK(memcmp(&snapshot, &unchanged, sizeof(snapshot)) == 0);
  OK(et_g3c4_private_generator_close_v1(context));
}

static void terminal_second_counter(et_g3c4_model_owner_internal *owner,
                                    int counter_case) {
  const int64_t prompt[2] = {0, 255};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  start_prefix(owner, prompt, 1, -1, &context, &input, &output);
  if (counter_case == 1) {
    context->generator_rng_words[2] = -1;
    context->generator_rng_words[3] = 0;
  } else if (counter_case == 2) {
    context->generator_rng_words[2] = -2;
    context->generator_rng_words[3] = -1;
  } else if (counter_case == 3) {
    context->generator_policy[2] = 1;
  } else {
    context->generator_rng_words[2] = -1;
    context->generator_rng_words[3] = -1;
  }
  int64_t before[4], after[4];
  float logits[256];
  memcpy(before, context->generator_rng_words, sizeof(before));
  memcpy(logits, context->p2g2_prefix_next_logits, sizeof(logits));
  bridge_carrier carrier = new_carrier();
  if (counter_case == 4) {
    CHECK(et_g3c4_private_p2g2_second_frame_carrier_v1(
        context, output, &carrier, 16) != 0);
    CHECK(memcmp(before, context->generator_rng_words,
                 sizeof(before)) == 0 &&
          et_g3c4_token_frame_idle(context));
    assert_pending(output);
    close_pair(context, input, output);
    return;
  }
  OK(et_g3c4_private_p2g2_second_frame_carrier_v1(
      context, output, &carrier, 16));
  memcpy(after, context->token_frame_successor, sizeof(after));
  terminal_raw first = raw_id(context->p2g2_prefix_token);
  terminal_raw second = raw_id(context->token_frame_candidate);
  print_oracle_case(counter_case == 3 ? "categorical-singleton" :
                    "categorical", logits, second.byte, before, after);
  OK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, &second, 9));
  CHECK(memcmp(context->generator_rng_words, after, sizeof(after)) == 0);
  terminal_snapshot snapshot;
  memset(&snapshot, 0, sizeof(snapshot));
  snapshot.length = 64;
  OK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &snapshot, sizeof(snapshot)));
  for (size_t index = 0u; index < 4u; index++)
    CHECK(word(&snapshot, index + 4u) == after[index]);
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_output_release_v1(output));
  OK(et_g3c4_private_generator_close_v1(context));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  const int64_t prompts[2][2] = {{0, 255}, {7, 11}};
  for (size_t pair = 0u; pair < 2u; pair++)
    for (int categorical = 0; categorical < 2; categorical++) {
      terminal_success(owner, prompts[pair], categorical, 1, 0);
      terminal_success(owner, prompts[pair], categorical, 0, 0);
    }
  terminal_success(owner, prompts[0], 0, 0, 1);
  for (int counter_case = 1; counter_case <= 4; counter_case++)
    terminal_second_counter(owner, counter_case);
  terminal_a2_witness(owner);
  terminal_failure_matrix(owner);
  terminal_failstop(owner);
  printf("G3-C4 P2/G2 terminal PASS: checks=%zu\n", checks);
  return 0;
}
