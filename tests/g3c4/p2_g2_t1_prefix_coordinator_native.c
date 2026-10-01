#define ET_G3C4_P2_G1_PENDING_TEST_MAIN et_g3c4_p2g1_coordinator_predecessor_main
#include "test_p2_g1_pending.c"
#undef ET_G3C4_P2_G1_PENDING_TEST_MAIN
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int coordinator_frame_fault;
static int coordinator_prefix_fault;
static int coordinator_fail_closed_tail;
static uint32_t coordinator_saved_pin_word;
static int coordinator_pin_is_changed;

int64_t et_g3c4_p2g2_coordinator_test_pin_drift_v1(
    void *context_candidate, int64_t active) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)context_candidate;
  uint32_t *word;
  if (context == NULL || context->owner == NULL ||
      context->owner->parameters[0] == NULL) return -1;
  word = (uint32_t *)context->owner->parameters[0]->value->data;
  if (word == NULL) return -1;
  if (active == 1 && !coordinator_pin_is_changed) {
    coordinator_saved_pin_word = *word;
    *word ^= 1u;
    coordinator_pin_is_changed = 1;
    return 0;
  }
  if (active == 0 && coordinator_pin_is_changed) {
    *word = coordinator_saved_pin_word;
    coordinator_pin_is_changed = 0;
    return 0;
  }
  return -1;
}

int64_t __real_et_g3c4_private_p2g2_first_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_first_frame_carrier_v1(
    void *context, void *input, void *staging, int64_t bytes) {
  int64_t original, result;
  if (coordinator_frame_fault == 2)
    return __real_et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, context, bytes);
  if (coordinator_frame_fault != 1)
    return __real_et_g3c4_private_p2g2_first_frame_carrier_v1(
        context, input, staging, bytes);
  memcpy(&original, staging, sizeof(original));
  memcpy(staging, &(const int64_t){7}, sizeof(original));
  result = __real_et_g3c4_private_p2g2_first_frame_carrier_v1(
      context, input, staging, bytes);
  memcpy(staging, &original, sizeof(original));
  return result;
}

int64_t __real_et_g3c4_private_p2g2_prefix_commit_v1(
    void *, void *, const void *);
int64_t __wrap_et_g3c4_private_p2g2_prefix_commit_v1(
    void *context_candidate, void *output_candidate, const void *raw) {
  et_g3c4_context_internal *context =
      (et_g3c4_context_internal *)context_candidate;
  et_g3c4_output_internal *output =
      (et_g3c4_output_internal *)output_candidate;
  et_i64_tensor_borrow *ids = NULL;
  et_a2_kv_cache_transaction_view *view = NULL;
  et_i64_tensor_error ids_error;
  et_kernel_error cache_error;
  int64_t original, result;
  unsigned char prior;
  if (coordinator_prefix_fault == 4)
    return __real_et_g3c4_private_p2g2_prefix_commit_v1(
        context_candidate, output_candidate, context_candidate);
  if (coordinator_prefix_fault == 3) {
    memcpy(&original, raw, sizeof(original));
    memcpy((void *)raw, &(const int64_t){2}, sizeof(original));
    result = __real_et_g3c4_private_p2g2_prefix_commit_v1(
        context_candidate, output_candidate, raw);
    memcpy((void *)raw, &original, sizeof(original));
    return result;
  }
  if (coordinator_prefix_fault == 5) {
    prior = ((const unsigned char *)raw)[sizeof(int64_t)];
    ((unsigned char *)raw)[sizeof(int64_t)] = (unsigned char)(prior ^ 1u);
    result = __real_et_g3c4_private_p2g2_prefix_commit_v1(
        context_candidate, output_candidate, raw);
    ((unsigned char *)raw)[sizeof(int64_t)] = prior;
    return result;
  }
  if (coordinator_prefix_fault == 1) {
    if (et_i64_tensor_borrow_begin_v1(output->ids, &ids, &ids_error) != 0)
      abort();
    result = __real_et_g3c4_private_p2g2_prefix_commit_v1(
        context_candidate, output_candidate, raw);
    if (et_i64_tensor_borrow_end_v1(&ids, &ids_error) != 0) abort();
    return result;
  }
  if (coordinator_prefix_fault == 2) {
    if (et_a2_kv_cache_transaction_view_begin_v1(
            context->token_frame_transaction, 0u, &view,
            &cache_error) != 0) abort();
    result = __real_et_g3c4_private_p2g2_prefix_commit_v1(
        context_candidate, output_candidate, raw);
    if (et_a2_kv_cache_transaction_view_end_v1(
            &view, &cache_error) != 0) abort();
    return result;
  }
  return __real_et_g3c4_private_p2g2_prefix_commit_v1(
      context_candidate, output_candidate, raw);
}

int32_t __real_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **slot, et_kernel_error *error) {
  if (coordinator_fail_closed_tail) return -1;
  return __real_et_a2_kv_cache_transaction_commit_v1(slot, error);
}

int64_t et_g3c4_p2g2_coordinator_test_failstop_v1(
    void *context, void *output, const void *raw) {
  pid_t child = fork();
  int status = 0;
  if (child < 0) return -1;
  if (child == 0) {
    coordinator_fail_closed_tail = 1;
    (void)et_g3c4_private_p2g2_prefix_commit_v1(context, output, raw);
    _exit(97);
  }
  if (waitpid(child, &status, 0) != child) return -1;
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT ? 0 : -1;
}

static et_i64_tensor_borrow *coordinator_ids_borrow;
static et_a2_kv_cache_read_borrow *coordinator_cache_borrow;
static et_a2_kv_cache_transaction_view *coordinator_transaction_view;

int64_t et_g3c4_p2g2_coordinator_test_ids_lease_v1(
    void *candidate, int64_t active) {
  et_g3c4_output_internal *output = et_g3c4_admit_output(candidate, 0);
  et_i64_tensor_error error;
  if (output == NULL || output->ids == NULL) return -1;
  if (active == 1 && coordinator_ids_borrow == NULL)
    return et_i64_tensor_borrow_begin_v1(
        output->ids, &coordinator_ids_borrow, &error);
  if (active == 0 && coordinator_ids_borrow != NULL)
    return et_i64_tensor_borrow_end_v1(&coordinator_ids_borrow, &error);
  return -1;
}

int64_t et_g3c4_p2g2_coordinator_test_cache_lease_v1(
    void *candidate, int64_t active) {
  et_g3c4_context_internal *context = et_g3c4_admit_active_call(candidate);
  et_kernel_error error;
  if (context == NULL) return -1;
  if (active == 1 && coordinator_cache_borrow == NULL &&
      coordinator_transaction_view == NULL) {
    if (context->token_frame_transaction != NULL)
      return et_a2_kv_cache_transaction_view_begin_v1(
          context->token_frame_transaction, 0u,
          &coordinator_transaction_view, &error);
    return et_a2_kv_cache_read_borrow_begin_v1(
        context->cache, &coordinator_cache_borrow, &error);
  }
  if (active == 0 && coordinator_transaction_view != NULL)
    return et_a2_kv_cache_transaction_view_end_v1(
        &coordinator_transaction_view, &error);
  if (active == 0 && coordinator_cache_borrow != NULL)
    return et_a2_kv_cache_read_borrow_end_v1(
        &coordinator_cache_borrow, &error);
  return -1;
}

/* This fixture observes the same native owner used by the Eshkol scope. */
static et_g3c4_model_owner_internal *coordinator_owner;
static uint32_t coordinator_values[1192], coordinator_gradients[1192];
static int64_t coordinator_entry_rng[4], coordinator_successor_rng[4];
static cache_snapshot coordinator_entry_cache, coordinator_precommit_cache;
static cache_snapshot coordinator_prefix_cache;
static binding_snapshot coordinator_entry_binding;
static binding_snapshot coordinator_precommit_binding;
static binding_snapshot coordinator_prefix_binding;
static float coordinator_next_logits[256];
static int64_t coordinator_selected;
static int64_t coordinator_prompt[2];
static int64_t coordinator_policy[6];
static int coordinator_saw_prefix;

static void coordinator_owner_bytes(uint32_t values[1192],
                                    uint32_t gradients[1192]) {
  size_t offset = 0u;
  for (size_t index = 0u; index < 14u; index++) {
    size_t count = et_g3c4_parameter_elements(index);
    memcpy(values + offset, coordinator_owner->parameters[index]->value->data,
           count * sizeof(*values));
    memcpy(gradients + offset,
           coordinator_owner->parameters[index]->gradient->data,
           count * sizeof(*gradients));
    offset += count;
  }
  if (offset != 1192u) abort();
}

static int coordinator_owner_unchanged(void) {
  uint32_t values[1192], gradients[1192];
  coordinator_owner_bytes(values, gradients);
  return memcmp(values, coordinator_values, sizeof(values)) == 0 &&
         memcmp(gradients, coordinator_gradients, sizeof(gradients)) == 0;
}

static int coordinator_cache_equal(const cache_snapshot *a,
                                   const cache_snapshot *b) {
  return a->length == b->length &&
         memcmp(a->keys, b->keys, sizeof(a->keys)) == 0 &&
         memcmp(a->values, b->values, sizeof(a->values)) == 0 &&
         memcmp(a->keep, b->keep, sizeof(a->keep)) == 0;
}

static int coordinator_binding_equal(const binding_snapshot *a,
                                     const binding_snapshot *b) {
  return a->ready == b->ready &&
         memcmp(a->identities, b->identities,
                sizeof(a->identities)) == 0 &&
         memcmp(a->values, b->values, sizeof(a->values)) == 0 &&
         memcmp(a->tokens, b->tokens, sizeof(a->tokens)) == 0;
}

static int coordinator_pending_zero(et_g3c4_output_internal *output) {
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  int result;
  if (et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) != 0)
    return 0;
  if (et_i64_tensor_borrow_view_v1(borrow, &view, &error) != 0)
    abort();
  result = view != NULL && view->rank == 1u && view->shape != NULL &&
           view->shape[0] == 2u && view->byte_length == 16u &&
           view->data != NULL &&
           memcmp(view->data, (const int64_t[2]){0, 0}, 16u) == 0;
  if (et_i64_tensor_borrow_end_v1(&borrow, &error) != 0) abort();
  return result;
}

int64_t et_g3c4_p2g2_coordinator_test_mark_v1(
    void *candidate, int64_t first, int64_t second) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(candidate);
  if (context == NULL) return -1;
  coordinator_owner = context->owner;
  coordinator_prompt[0] = first;
  coordinator_prompt[1] = second;
  memcpy(coordinator_policy, context->generator_policy,
         sizeof(coordinator_policy));
  coordinator_owner_bytes(coordinator_values, coordinator_gradients);
  memcpy(coordinator_entry_rng, context->generator_rng_words,
         sizeof(coordinator_entry_rng));
  snapshot_cache(context->cache, &coordinator_entry_cache);
  snapshot_binding(context, &coordinator_entry_binding);
  coordinator_saw_prefix = 0;
  return coordinator_entry_cache.length == 0 &&
         coordinator_entry_binding.ready == 0u &&
         coordinator_owner_unchanged() ? 0 : -1;
}

/* 1..5: bridge cuts; 6..10: prefix precommit; 11/12: stage header/alias. */
int64_t et_g3c4_p2g2_coordinator_test_fault_v1(int64_t mode) {
  record_prefill2 = 0;
  fail_prefill2_at = SIZE_MAX;
  fail_bridge_sample = 0;
  fail_bridge_forward = 0;
  forge_sample = 0;
  forge_descriptor = 0;
  coordinator_frame_fault = 0;
  coordinator_prefix_fault = 0;
  if (mode == 0) return 0;
  if (mode == 1) {
    prefill2_dispatches = 0u;
    fail_prefill2_at = 0u;
    record_prefill2 = 1;
  } else if (mode == 2) fail_bridge_sample = 1;
  else if (mode == 3) fail_bridge_forward = 1;
  else if (mode == 4 || mode == 5) {
    forge_sample = 1;
    forged_token = mode == 4 ? -1 : 256;
  } else if (mode >= 6 && mode <= 10)
    coordinator_prefix_fault = (int)(mode - 5);
  else if (mode == 11 || mode == 12)
    coordinator_frame_fault = (int)(mode - 10);
  else return -1;
  return 0;
}

int64_t et_g3c4_p2g2_coordinator_test_observe_v1(
    void *context_candidate, void *output_candidate, int64_t cache_length,
    int64_t selected) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  cache_snapshot cache;
  if (context == NULL || output == NULL ||
      (cache_length != 0 && cache_length != 2) ||
      !coordinator_owner_unchanged() ||
      output->transport.state != 0u || output->generated_length != 2 ||
      output->length != 0 || output->cache_length != 0 ||
      output->numeric_ready != 0u || output->ids_copied != 0u ||
      output->text_ready != 0u || !coordinator_pending_zero(output))
    return -1;
  snapshot_cache(context->cache, &cache);
  if (cache.length != cache_length ||
      memcmp(cache.keep,
             cache_length == 0 ? (const uint8_t[4]){0, 0, 0, 0} :
                                 (const uint8_t[4]){1, 1, 0, 0}, 4u) != 0)
    return -1;
  if (cache_length == 2 &&
      (context->p2g2_prefix_committed != 0u ||
       context->prefill_binding_ready != 1u ||
       memcmp(context->generator_rng_words, coordinator_entry_rng,
              sizeof(coordinator_entry_rng)) != 0)) return -1;
  if (cache_length == 0 &&
      memcmp(context->generator_rng_words, coordinator_entry_rng,
             sizeof(coordinator_entry_rng)) != 0) return -1;
  if (selected >= 0 && context->token_frame_candidate != selected)
    return -1;
  if (cache_length == 2) {
    coordinator_precommit_cache = cache;
    snapshot_binding(context, &coordinator_precommit_binding);
    if (coordinator_precommit_binding.ready != 1u) return -1;
  }
  return 0;
}

int64_t et_g3c4_p2g2_coordinator_test_prefix_v1(
    void *context_candidate, void *output_candidate, int64_t selected) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_active_call(context_candidate);
  et_g3c4_output_internal *output =
      et_g3c4_admit_output(output_candidate, 0);
  if (context == NULL || output == NULL ||
      context->p2g2_prefix_committed != 1u ||
      context->p2g2_prefix_token != selected ||
      context->p2g2_prefix_raw != (unsigned char)selected ||
      output->numeric_ready != 0u || output->ids_copied != 0u ||
      output->text_ready != 0u || output->length != 0 ||
      !coordinator_owner_unchanged() ||
      !coordinator_pending_zero(output)) return -1;
  snapshot_cache(context->cache, &coordinator_prefix_cache);
  snapshot_binding(context, &coordinator_prefix_binding);
  if (coordinator_prefix_cache.length != 3 ||
      memcmp(coordinator_prefix_cache.keep,
             (const uint8_t[4]){1, 1, 1, 0}, 4u) != 0 ||
      coordinator_prefix_binding.ready != 1u)
    return -1;
  memcpy(coordinator_next_logits, context->p2g2_prefix_next_logits,
         sizeof(coordinator_next_logits));
  memcpy(coordinator_successor_rng, context->generator_rng_words,
         sizeof(coordinator_successor_rng));
  coordinator_selected = selected;
  coordinator_saw_prefix = 1;
  return 0;
}

int64_t et_g3c4_p2g2_coordinator_test_after_abort_v1(
    void *context_candidate, int64_t expected_length) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  cache_snapshot actual;
  binding_snapshot actual_binding;
  if (context == NULL ||
      expected_length < 0 || expected_length > 3 ||
      !coordinator_owner_unchanged()) return -1;
  snapshot_cache(context->cache, &actual);
  snapshot_binding(context, &actual_binding);
  if (actual.length != expected_length ||
      memcmp(actual.keep,
             expected_length == 0 ? (const uint8_t[4]){0, 0, 0, 0} :
             expected_length == 2 ? (const uint8_t[4]){1, 1, 0, 0} :
                                    (const uint8_t[4]){1, 1, 1, 0}, 4u) != 0)
    return -1;
  if (expected_length == 0 &&
      (!coordinator_cache_equal(&actual, &coordinator_entry_cache) ||
       !coordinator_binding_equal(&actual_binding,
                                  &coordinator_entry_binding)))
    return -1;
  if (expected_length == 2 &&
      (!coordinator_cache_equal(&actual, &coordinator_precommit_cache) ||
       !coordinator_binding_equal(&actual_binding,
                                  &coordinator_precommit_binding)))
    return -1;
  if (expected_length == 3 &&
      (!coordinator_saw_prefix ||
       !coordinator_cache_equal(&actual, &coordinator_prefix_cache) ||
       !coordinator_binding_equal(&actual_binding,
                                  &coordinator_prefix_binding) ||
       memcmp(context->generator_rng_words, coordinator_successor_rng,
              sizeof(coordinator_successor_rng)) != 0)) return -1;
  if (expected_length < 3 &&
      memcmp(context->generator_rng_words, coordinator_entry_rng,
             sizeof(coordinator_entry_rng)) != 0) return -1;
  return 0;
}

int64_t et_g3c4_p2g2_coordinator_test_dead_output_v1(void *candidate) {
  et_g3c4_output_internal *output = (et_g3c4_output_internal *)candidate;
  if (output == NULL || output->transport.magic != ET_G3C4_OUTPUT_MAGIC)
    return -1;
  return output->transport.state == ET_G3C4_CONTEXT_DEAD &&
         output->parent_ctx == NULL && output->prompt_length == 0 &&
         output->generated_length == 0 && output->ids == NULL &&
         output->length == 0 && output->cache_length == 0 &&
         output->numeric_ready == 0u && output->ids_copied == 0u &&
         output->text_ready == 0u ? 0 : -1;
}

int64_t et_g3c4_p2g2_coordinator_test_reference_v1(void) {
  et_g3c4_context_internal *reference;
  et_g3c4_context_internal *oracle_context;
  et_g3c4_input_internal *oracle_input;
  et_g3c4_output_internal *oracle_output;
  cache_snapshot actual;
  float logits[256];
  float last_logits[256];
  const int64_t ids[3] = {
      coordinator_prompt[0], coordinator_prompt[1], coordinator_selected};
  if (!coordinator_saw_prefix || coordinator_owner == NULL) return -1;
  reference = create_generator(coordinator_owner);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 2, 1) != 0 ||
      et_g3c4_private_prefill3_v1(reference, ids, logits) != 0)
    return -1;
  snapshot_cache(reference->cache, &actual);
  if (memcmp(logits, coordinator_next_logits, sizeof(logits)) != 0 ||
      !coordinator_cache_equal(&actual, &coordinator_prefix_cache))
    return -1;
  if (et_g3c4_private_call_abort_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0)
    return -1;
  if (!coordinator_owner_unchanged()) return -1;

  /* Independent P2 prefill supplies the oracle's 256 last logits. The
   * compared token/RNG words remain those observed from the Eshkol call. */
  oracle_context = et_g3c4_private_generator_seed_v1(
      coordinator_owner, coordinator_entry_rng[1],
      coordinator_policy[0], coordinator_policy[1],
      coordinator_policy[2], coordinator_policy[3],
      coordinator_policy[4], coordinator_policy[5]);
  oracle_input = create_prompt(coordinator_prompt, 2);
  if (oracle_context == NULL || oracle_input == NULL ||
      et_g3c4_private_prompt_prefill_preflight_v1(
          oracle_context, oracle_input, 2) != 0 ||
      et_g3c4_private_call_acquire_v1(oracle_context, 2, 2) != 0)
    return -1;
  oracle_output = et_g3c4_private_output_reserve_v1(oracle_context, 2);
  if (oracle_output == NULL ||
      et_g3c4_private_prompt_prefill_v1(
          oracle_context, oracle_input, last_logits) != 0)
    return -1;
  if (et_g3c4_private_call_abort_v1(oracle_context) != 0 ||
      et_g3c4_private_tensor_release_v1(oracle_input) != 0 ||
      et_g3c4_private_generator_close_v1(oracle_context) != 0 ||
      !coordinator_owner_unchanged())
    return -1;
  printf("COORD_ORACLE %s %lld %lld %lld %lld %lld %lld %lld",
         coordinator_policy[0] == 0 ? "greedy" : "categorical",
         (long long)coordinator_prompt[0],
         (long long)coordinator_prompt[1],
         (long long)coordinator_policy[1],
         (long long)coordinator_policy[2],
         (long long)coordinator_policy[3],
         (long long)coordinator_policy[5],
         (long long)coordinator_selected);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)coordinator_entry_rng[index]);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)coordinator_successor_rng[index]);
  for (size_t index = 0u; index < 256u; index++) {
    uint32_t bits;
    memcpy(&bits, &last_logits[index], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
  return 0;
}
