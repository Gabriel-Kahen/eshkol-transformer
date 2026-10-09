/* Reuse the exact accepted prefix owner/rollback observer. This wrapper
 * forces a native terminal preflight rejection before its zero boundary. */
#include "p2_g2_t1_prefix_coordinator_native.c"

typedef struct terminal_close_parameters {
  unsigned char values[4768], gradients[4768];
  uint8_t present[14];
  et_f32_gradient_metadata_v1 metadata[14];
} terminal_close_parameters;

static int terminal_close_freeze_parameters(terminal_close_parameters *out) {
  size_t offset = 0u;
  et_f32_tensor_error error;
  memset(out, 0, sizeof(*out));
  if (coordinator_owner == NULL) return -1;
  for (size_t i = 0u; i < 14u; i++) {
    et_f32_parameter *parameter = coordinator_owner->parameters[i];
    size_t bytes = parameter->value->byte_length;
    if (bytes > sizeof(out->values) - offset) return -1;
    memcpy(out->values + offset, parameter->value->data, bytes);
    if (parameter->gradient != NULL) {
      if (parameter->gradient->byte_length != bytes) return -1;
      out->present[i] = 1u;
      memcpy(out->gradients + offset, parameter->gradient->data, bytes);
    }
    out->metadata[i].struct_size = sizeof(out->metadata[i]);
    if (et_f32_parameter_gradient_metadata_v1(
            parameter, &out->metadata[i], &error) != 0) return -1;
    offset += bytes;
  }
  return offset == sizeof(out->values) ? 0 : -1;
}

typedef struct terminal_close_snapshot {
  int64_t length;
  unsigned char bytes[64];
} terminal_close_snapshot;

static int64_t terminal_close_word(const terminal_close_snapshot *snapshot,
                                   size_t index) {
  uint64_t value = 0u;
  for (size_t i = 0u; i < 8u; i++)
    value |= (uint64_t)snapshot->bytes[8u * index + i] << (8u * i);
  return (int64_t)value;
}

static int capture_terminal_t4;
static cache_snapshot terminal_t4;
int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  if (capture_terminal_t4 && slot != NULL && *slot != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    if (et_a2_kv_cache_transaction_view_begin_v1(
            *slot, 0u, &view, &local) != 0 ||
        et_a2_kv_cache_transaction_view_tensors_v1(
            view, &keys, &values, &lengths, &keep, &local) != 0)
      abort();
    memcpy(terminal_t4.keys, keys->data, sizeof(terminal_t4.keys));
    memcpy(terminal_t4.values, values->data, sizeof(terminal_t4.values));
    terminal_t4.length = *(const int64_t *)lengths->data;
    memcpy(terminal_t4.keep, keep->data, sizeof(terminal_t4.keep));
    if (et_a2_kv_cache_transaction_view_end_v1(&view, &local) != 0) abort();
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(slot, error);
}

static int terminal_close_fault;
int64_t et_g3c4_terminal_tail_arm_v1(void);
static void *terminal_closed_context, *terminal_closed_output;
static int64_t terminal_closed_count, terminal_closed_ids[2];
static int64_t terminal_before_second[4], terminal_after[4];
static float terminal_second_logits[256];
static binding_snapshot terminal_before_binding;
static terminal_close_parameters terminal_original_parameters;

int64_t et_g3c4_p2g2_terminal_close_test_mark_v1(
    void *context, int64_t first, int64_t second) {
  if (et_g3c4_p2g2_coordinator_test_mark_v1(
          context, first, second) != 0) return -1;
  if (terminal_close_freeze_parameters(&terminal_original_parameters) != 0)
    return -1;
  terminal_closed_context = NULL;
  terminal_closed_output = NULL;
  terminal_closed_count = 0;
  return 0;
}

int64_t et_g3c4_p2g2_terminal_close_test_fault_v1(int64_t active) {
  if (active != 0 && active != 1) return -1;
  terminal_close_fault = (int)active;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);

int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  et_g3c4_context_internal *native = et_g3c4_admit_active_call(context);
  int64_t result;
  if (native == NULL) return -1;
  if (terminal_close_fault) {
    if (native->p2g2_prefix_committed) {
      if (et_g3c4_p2g2_coordinator_test_prefix_v1(
              context, output, native->p2g2_prefix_token) != 0)
        return -1;
    } else if (et_g3c4_p2g2_coordinator_test_observe_v1(
                   context, output, 2, -1) != 0) {
      return -1;
    }
  }
  if (!terminal_close_fault &&
      (first == NULL || first_bytes != 9 ||
       ((second == NULL) != (second_bytes == 0)) ||
       (second != NULL && second_bytes != 9))) return -1;
  if (!terminal_close_fault) {
    terminal_closed_ids[0] = ((const unsigned char *)first)[8];
    terminal_closed_ids[1] = second == NULL ? 0 :
        ((const unsigned char *)second)[8];
    terminal_closed_count = second == NULL ? 1 : 2;
    memcpy(terminal_before_second, native->generator_rng_words,
           sizeof(terminal_before_second));
    if (second != NULL)
      memcpy(terminal_second_logits, native->p2g2_prefix_next_logits,
             sizeof(terminal_second_logits));
    snapshot_binding(native, &terminal_before_binding);
  }
  result = __real_et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, first, terminal_close_fault == 1 ? 8 : first_bytes,
      second, second_bytes);
  if (!terminal_close_fault && result == 0) {
    terminal_closed_context = context;
    terminal_closed_output = output;
    memcpy(terminal_after, native->generator_rng_words,
           sizeof(terminal_after));
    if (et_g3c4_terminal_tail_arm_v1() != 0) abort();
  }
  return result;
}

static void terminal_oracle_row(const char *mode, const char *draw,
                                int64_t selected, const int64_t before[4],
                                const int64_t after[4],
                                const float logits[256]) {
  printf("TERM_ORACLE %s %lld %lld %s %lld", mode,
         (long long)coordinator_prompt[0],
         (long long)coordinator_prompt[1], draw, (long long)selected);
  for (size_t i = 0; i < 4u; i++) printf(" %lld", (long long)before[i]);
  for (size_t i = 0; i < 4u; i++) printf(" %lld", (long long)after[i]);
  for (size_t i = 0; i < 256u; i++) {
    uint32_t bits;
    memcpy(&bits, &logits[i], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
}

/* Called only after the real Eshkol terminal call returned its shell. P2
 * last logits and full P3/T4 K/V are obtained by separate native forwards;
 * no caller-supplied sampled ID or RNG enters the production coordinator. */
int64_t et_g3c4_p2g2_terminal_close_test_reference_v1(
    void *context_candidate, void *output_candidate, int64_t count) {
  et_g3c4_context_internal *context = context_candidate;
  et_g3c4_output_internal *output = output_candidate;
  et_g3c4_context_internal *reference, *oracle;
  et_g3c4_input_internal *oracle_input;
  et_g3c4_output_internal *oracle_output;
  cache_snapshot actual, expected, p3;
  binding_snapshot actual_binding;
  terminal_close_snapshot snapshot = {0};
  float logits[1024], first_logits[256];
  int64_t tokens[4] = {coordinator_prompt[0], coordinator_prompt[1],
                       terminal_closed_ids[0], terminal_closed_ids[1]};
  const char *mode = coordinator_policy[0] == 0 ? "greedy" : "categorical";
  const int64_t *first_after = count == 1 ? terminal_after :
                                                  terminal_before_second;
  terminal_close_parameters current;
  if (context != terminal_closed_context ||
      output != terminal_closed_output ||
      count != terminal_closed_count || (count != 1 && count != 2) ||
      !coordinator_owner_unchanged() ||
      memcmp(context->generator_rng_words, terminal_after,
             sizeof(terminal_after)) != 0 ||
      output->length != count || output->cache_length != count + 2 ||
      output->parent_ctx != NULL) return -1;
  if (terminal_close_freeze_parameters(&current) != 0) return -1;
  if (memcmp(&current, &terminal_original_parameters,
             sizeof(current)) != 0) return -1;
  snapshot.length = 64;
  if (et_g3c4_private_p2g2_terminal_snapshot_v1(
          output, &snapshot, sizeof(snapshot)) != 0 ||
      terminal_close_word(&snapshot, 0) != tokens[2] ||
      terminal_close_word(&snapshot, 1) != tokens[3] ||
      terminal_close_word(&snapshot, 2) != count ||
      terminal_close_word(&snapshot, 3) != count + 2) return -1;
  for (size_t i = 0; i < 4u; i++)
    if (terminal_close_word(&snapshot, i + 4u) != terminal_after[i]) return -1;
  snapshot_cache(context->cache, &actual);
  snapshot_binding(context, &actual_binding);
  reference = create_generator(coordinator_owner);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 0, 0) != 0 ||
      et_g3c4_private_prefill3_v1(reference, tokens, logits) != 0)
    return -1;
  snapshot_cache(reference->cache, &p3);
  if (et_g3c4_private_call_abort_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0) return -1;
  expected = p3;
  if (count == 2) {
    reference = create_generator(coordinator_owner);
    if (reference == NULL ||
        et_g3c4_private_call_acquire_v1(reference, 0, 0) != 0)
      return -1;
    capture_terminal_t4 = 1;
    if (et_g3c4_private_full_prefix_forward_v1(
            reference, tokens, logits) != 0) return -1;
    capture_terminal_t4 = 0;
    expected = terminal_t4;
    if (et_g3c4_private_call_abort_v1(reference) != 0 ||
        et_g3c4_private_generator_close_v1(reference) != 0) return -1;
  }
  if (!coordinator_cache_equal(&actual, &expected) ||
      !coordinator_binding_equal(&actual_binding, &terminal_before_binding))
    return -1;
  if (count == 2) {
    for (size_t head = 0u; head < 2u; head++)
      if (memcmp(actual.keys + head * 8u, p3.keys + head * 8u,
                 6u * sizeof(float)) != 0 ||
          memcmp(actual.values + head * 8u, p3.values + head * 8u,
                 6u * sizeof(float)) != 0) return -1;
    if (memcmp(actual.keep, p3.keep, 3u) != 0) return -1;
  }
  oracle = et_g3c4_private_generator_seed_v1(
      coordinator_owner, coordinator_entry_rng[1],
      coordinator_policy[0], coordinator_policy[1], coordinator_policy[2],
      coordinator_policy[3], coordinator_policy[4], coordinator_policy[5]);
  oracle_input = create_prompt(coordinator_prompt, 2);
  if (oracle == NULL || oracle_input == NULL ||
      et_g3c4_private_prompt_prefill_preflight_v1(oracle, oracle_input, 2) != 0 ||
      et_g3c4_private_call_acquire_v1(oracle, 2, 2) != 0) return -1;
  oracle_output = et_g3c4_private_output_reserve_v1(oracle, 2);
  if (oracle_output == NULL ||
      et_g3c4_private_prompt_prefill_v1(
          oracle, oracle_input, first_logits) != 0) return -1;
  if (et_g3c4_private_call_abort_v1(oracle) != 0 ||
      et_g3c4_private_tensor_release_v1(oracle_input) != 0 ||
      et_g3c4_private_generator_close_v1(oracle) != 0) return -1;
  if (!coordinator_owner_unchanged() ||
      terminal_close_freeze_parameters(&current) != 0 ||
      memcmp(&current, &terminal_original_parameters,
             sizeof(current)) != 0) return -1;
  terminal_oracle_row(mode, "first", tokens[2], coordinator_entry_rng,
                      first_after, first_logits);
  if (count == 2)
    terminal_oracle_row(mode, "second", tokens[3], terminal_before_second,
                        terminal_after, terminal_second_logits);
  printf("TERM_CACHE %s %lld %lld %lld %lld %lld %lld %lld\n", mode,
         (long long)coordinator_prompt[0],
         (long long)coordinator_prompt[1], (long long)count,
         (long long)actual.length, (long long)tokens[2],
         (long long)tokens[3], (long long)coordinator_policy[5]);
  terminal_closed_context = NULL;
  terminal_closed_output = NULL;
  return 0;
}

int64_t et_g3c4_p2g2_terminal_close_test_clone_shape_v1(
    void *candidate, int64_t count) {
  et_g3c4_terminal_ids_clone_internal *clone = candidate;
  et_i64_tensor_error error;
  size_t rank = 0u;
  uint64_t length = 0u;
  if (clone == NULL || (count != 1 && count != 2) ||
      clone->transport.kind != 9u ||
      clone->transport.state != ET_G3C4_CONTEXT_LIVE ||
      clone->logical_length != count || clone->ids == NULL ||
      et_i64_tensor_rank_v1(clone->ids, &rank, &error) != 0 ||
      et_i64_tensor_shape_at_v1(clone->ids, 0u, &length, &error) != 0)
    return -1;
  return rank == 1u && length == (uint64_t)count ? 0 : -1;
}
