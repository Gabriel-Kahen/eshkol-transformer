/* Link the same private aggregate as the accepted prefix witness, now with
 * EOS, second-frame, terminal and logical-I1 features enabled together. */
#define ET_G3C4_P2_G2_EOS_FIRST_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_SECOND_FRAME_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE 1
#define ET_A2_KV_CACHE_TERMINAL_WITNESS_PRIVATE 1
#include "p2_g2_t1_prefix_coordinator_native.c"

static int composer_capture_t4;
static cache_snapshot composer_t4;
int32_t __real_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **, et_kernel_error *);
int32_t __wrap_et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (composer_capture_t4 && transaction != NULL && *transaction != NULL) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error local;
    if (et_a2_kv_cache_transaction_view_begin_v1(
          *transaction, 0u, &view, &local) != 0 ||
        et_a2_kv_cache_transaction_view_tensors_v1(
          view, &keys, &values, &lengths, &keep, &local) != 0)
      return -1;
    memcpy(composer_t4.keys, keys->data, sizeof(composer_t4.keys));
    memcpy(composer_t4.values, values->data, sizeof(composer_t4.values));
    composer_t4.length = *(const int64_t *)lengths->data;
    memcpy(composer_t4.keep, keep->data, sizeof(composer_t4.keep));
    if (et_a2_kv_cache_transaction_view_end_v1(&view, &local) != 0)
      return -1;
  }
  return __real_et_a2_kv_cache_transaction_abort_v1(transaction, error);
}

int64_t et_g3c4_p2g2_composer_test_full_prefix_v1(
    void *context_candidate, int64_t prompt0, int64_t prompt1,
    int64_t id0, int64_t id1, int64_t count) {
  et_g3c4_context_internal *actual =
      et_g3c4_admit_idle_call(context_candidate);
  et_g3c4_context_internal *reference;
  cache_snapshot observed, expected;
  float logits[1024];
  const int64_t prefix[4] = {prompt0, prompt1, id0, id1};
  if (actual == NULL || (count != 1 && count != 2) ||
      id0 < 0 || id0 > 255 || (count == 2 && (id1 < 0 || id1 > 255)) ||
      (count == 1 && id1 != 0))
    return -1;
  snapshot_cache(actual->cache, &observed);
  reference = create_generator(actual->owner);
  if (reference == NULL ||
      et_g3c4_private_call_acquire_v1(reference, 0, 0) != 0)
    return -1;
  if (count == 1) {
    if (et_g3c4_private_prefill3_v1(reference, prefix, logits) != 0)
      return -1;
    snapshot_cache(reference->cache, &expected);
  } else {
    composer_capture_t4 = 1;
    if (et_g3c4_private_full_prefix_forward_v1(
          reference, prefix, logits) != 0)
      return -1;
    composer_capture_t4 = 0;
    expected = composer_t4;
  }
  if (!coordinator_cache_equal(&observed, &expected) ||
      et_g3c4_private_call_finish_v1(reference) != 0 ||
      et_g3c4_private_generator_close_v1(reference) != 0)
    return -1;
  return 0;
}

static int composer_counter_case;
static int composer_exhausted_observed;

int64_t et_g3c4_p2g2_composer_test_second_oracle_v1(
    void *context_candidate, int64_t selected) {
  et_g3c4_context_internal *context =
      et_g3c4_admit_idle_call(context_candidate);
  if (context == NULL || !coordinator_saw_prefix ||
      selected < 0 || selected > 255 || !coordinator_owner_unchanged())
    return -1;
  const char *mode = coordinator_policy[0] == 0 ? "greedy" :
      coordinator_policy[2] == 1 ? "categorical-singleton" :
      composer_counter_case == 1 ? "categorical-carry" :
      composer_counter_case == 2 ? "categorical-final" : "categorical";
  printf("COMPOSER_SECOND %s %lld %lld", mode,
         (long long)coordinator_selected, (long long)selected);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)coordinator_successor_rng[index]);
  for (size_t index = 0u; index < 4u; index++)
    printf(" %lld", (long long)context->generator_rng_words[index]);
  for (size_t index = 0u; index < 256u; index++) {
    uint32_t bits;
    memcpy(&bits, &coordinator_next_logits[index], sizeof(bits));
    printf(" %08x", bits);
  }
  putchar('\n');
  return 0;
}

static int composer_fault_mode;
static int composer_read_fault;
int64_t et_g3c4_p2g2_composer_test_counter_v1(
    void *candidate, int64_t counter_case) {
  et_g3c4_context_internal *context = et_g3c4_admit_idle_call(candidate);
  if (context == NULL || counter_case < 0 || counter_case > 3 ||
      context->generator_rng_words[0] != 1 ||
      context->generator_rng_words[2] != 0 ||
      context->generator_rng_words[3] != 0)
    return -1;
  composer_counter_case = (int)counter_case;
  composer_exhausted_observed = 0;
  if (counter_case == 1)
    context->generator_rng_words[2] = -2;
  else if (counter_case == 2) {
    context->generator_rng_words[2] = -3;
    context->generator_rng_words[3] = -1;
  } else if (counter_case == 3) {
    context->generator_rng_words[2] = -1;
    context->generator_rng_words[3] = -1;
  }
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_exhausted_v1(void) {
  return composer_exhausted_observed;
}
int64_t et_g3c4_p2g2_composer_test_fault_v1(int64_t active) {
  if (active < 0 || active > 3) return -1;
  composer_fault_mode = (int)active;
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_read_fault_v1(int64_t active) {
  if (active < 0 || active > 3) return -1;
  composer_read_fault = (int)active;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_snapshot_v1(
    void *output, void *destination, int64_t bytes) {
  if (composer_read_fault == 1) {
    composer_read_fault = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, destination, bytes);
}

int64_t __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *, int64_t, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
    void *clone, int64_t count, void *destination, int64_t bytes) {
  if (composer_read_fault == 3) {
    int64_t *declared = (int64_t *)destination;
    int64_t result;
    composer_read_fault = 0;
    *declared += 8;
    result = __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
        clone, count, destination, bytes);
    *declared -= 8;
    if (result == 0) abort();
    return result;
  }
  if (composer_read_fault == 2) {
    composer_read_fault = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return __real_et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      clone, count, destination, bytes);
}

int64_t et_g3c4_p2g2_composer_test_clone_count_v1(int64_t state) {
  int64_t count = 0;
  const et_g3c4_transport_header_internal *record;
  if (state != ET_G3C4_CONTEXT_LIVE && state != ET_G3C4_CONTEXT_DEAD)
    return -1;
  for (record = et_g3c4_transport_registry; record != NULL;
       record = record->registry_next)
    if (record->kind == ET_G3C4_TERMINAL_IDS_KIND && record->state == state)
      count++;
  return count;
}

/* Test-only proof markers; the production sources have no callback seam. */
static int composer_overlay_active, composer_first_success;
static int composer_second_success, composer_zero_marker;
static et_g3c4_output_internal *composer_overlay_pending;
/* A test-only holder spans a rejected abort inside the original m3-call. */
static et_g3c4_context_internal *composer_lease_context;
static et_g3c4_output_internal *composer_lease_output;
static et_i64_tensor *composer_lease_ids;
static et_a2_kv_cache *composer_lease_cache;
static et_a2_kv_cache_transaction *composer_lease_transaction;
static et_i64_tensor_borrow *composer_lease_i1;
static et_a2_kv_cache_transaction_view *composer_lease_a2;
static int64_t composer_lease_rng[4];
static int composer_lease_kind, composer_lease_phase, composer_lease_calls;
static int64_t composer_lease_first_status, composer_lease_second_status;
static et_g3c4_error_state_internal composer_lease_first_error;
static uint32_t composer_lease_frame_state;
static int64_t composer_lease_frame_position, composer_lease_candidate;
typedef struct composer_lease_cache_image {
  unsigned char keys[64], values[64], lengths[8], keep[4];
} composer_lease_cache_image;
static composer_lease_cache_image composer_lease_image;

static int composer_lease_pending_zero(
    et_g3c4_output_internal *pending, int64_t kind) {
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  if (kind == 2) return coordinator_pending_zero(pending);
  if (kind != 1 || composer_lease_i1 == NULL ||
      et_i64_tensor_borrow_view_v1(
          composer_lease_i1, &view, &error) != 0) return 0;
  return view != NULL && view->rank == 1u && view->shape != NULL &&
         view->shape[0] == 2u && view->byte_length == 16u &&
         view->data != NULL &&
         memcmp(view->data, (const int64_t[2]){0, 0}, 16u) == 0;
}

static int composer_lease_cache_snapshot(
    et_a2_kv_cache_transaction *transaction,
    et_a2_kv_cache_transaction_view *held,
    composer_lease_cache_image *image) {
  et_a2_kv_cache_transaction_view *temporary = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
  et_kernel_error error;
  et_a2_kv_cache_transaction_view *view = held;
  int valid;
  if (view == NULL && et_a2_kv_cache_transaction_view_begin_v1(
          transaction, 0u, &temporary, &error) != 0) return -1;
  if (view == NULL) view = temporary;
  valid = et_a2_kv_cache_transaction_view_tensors_v1(
      view, &keys, &values, &lengths, &keep, &error) == 0 &&
      keys != NULL && values != NULL && lengths != NULL && keep != NULL &&
      keys->data != NULL && values->data != NULL &&
      lengths->data != NULL && keep->data != NULL &&
      keys->byte_length == sizeof(image->keys) &&
      values->byte_length == sizeof(image->values) &&
      lengths->byte_length == sizeof(image->lengths) &&
      keep->byte_length == sizeof(image->keep);
  if (valid) {
    memcpy(image->keys, keys->data, sizeof(image->keys));
    memcpy(image->values, values->data, sizeof(image->values));
    memcpy(image->lengths, lengths->data, sizeof(image->lengths));
    memcpy(image->keep, keep->data, sizeof(image->keep));
  }
  if (temporary != NULL &&
      et_a2_kv_cache_transaction_view_end_v1(&temporary, &error) != 0)
    return -1;
  return valid ? 0 : -1;
}

int64_t __real_et_g3c4_private_call_abort_v1(void *);
int64_t __wrap_et_g3c4_private_call_abort_v1(void *candidate) {
  int64_t result = __real_et_g3c4_private_call_abort_v1(candidate);
  if (composer_lease_phase && candidate == composer_lease_context) {
    composer_lease_calls++;
    if (composer_lease_calls == 1) {
      composer_lease_first_status = result;
      composer_lease_first_error = et_g3c4_error_snapshot_internal();
    } else if (composer_lease_calls == 2) {
      composer_lease_second_status = result;
    }
  }
  return result;
}

int64_t et_g3c4_p2g2_composer_test_lease_begin_v1(
    void *context_candidate, void *output_candidate, int64_t kind) {
  et_g3c4_context_internal *context;
  et_g3c4_output_internal *output = NULL;
  et_i64_tensor_error i1_error;
  et_kernel_error a2_error;
  if (composer_lease_phase || (kind != 1 && kind != 2) ||
      output_candidate == NULL) return -1;
  context = et_g3c4_admit_active_call(context_candidate);
  if (context == NULL ||
      et_g3c4_pending_output_lookup(context, &output) != 0 ||
      output == NULL || output != output_candidate || output->ids == NULL ||
      context->token_frame_state != ET_G3C4_TOKEN_FRAME_READY ||
      context->token_frame_transaction == NULL ||
      !coordinator_owner_unchanged()) return -1;
  if (kind == 1) {
    if (et_i64_tensor_borrow_begin_v1(
          output->ids, &composer_lease_i1, &i1_error) != 0) return -1;
  } else if (et_a2_kv_cache_transaction_view_begin_v1(
                 context->token_frame_transaction, 0u,
                 &composer_lease_a2, &a2_error) != 0) return -1;
  if (composer_lease_cache_snapshot(context->token_frame_transaction,
          composer_lease_a2, &composer_lease_image) != 0) abort();
  composer_lease_context = context;
  composer_lease_output = output;
  composer_lease_ids = output->ids;
  composer_lease_cache = context->cache;
  composer_lease_transaction = context->token_frame_transaction;
  composer_lease_frame_state = context->token_frame_state;
  composer_lease_frame_position = context->token_frame_position;
  composer_lease_candidate = context->token_frame_candidate;
  memcpy(composer_lease_rng, context->generator_rng_words,
         sizeof(composer_lease_rng));
  composer_lease_kind = (int)kind;
  composer_lease_calls = 0;
  composer_lease_first_status = composer_lease_second_status = -1;
  composer_lease_phase = 1;
  return 0;
}

int64_t et_g3c4_p2g2_composer_test_lease_rejected_v1(
    void *context_candidate, void *output_candidate, int64_t kind) {
  et_g3c4_output_internal *pending = NULL;
  et_g3c4_context_internal *context = composer_lease_context;
  const int expected_domain = kind == 1 ? ET_G3C4_DOMAIN_I1 : ET_G3C4_DOMAIN_K1;
  const int expected_category = kind == 1 ?
      ET_I64_TENSOR_ERROR_INVALID_STATE : ET_KERNEL_ERROR_INVALID_ARGUMENT;
  const int expected_code = kind == 1 ?
      ET_I64_TENSOR_CODE_ACTIVE_BORROW : ET_KERNEL_CODE_PROVIDER_REJECTED;
  composer_lease_cache_image current_image;
  if (composer_lease_phase != 1 || composer_lease_kind != kind ||
      context_candidate != context || output_candidate != composer_lease_output ||
      composer_lease_calls != 1 ||
      composer_lease_first_status != expected_category ||
      composer_lease_first_error.domain != expected_domain ||
      composer_lease_first_error.category != expected_category ||
      composer_lease_first_error.code != expected_code ||
      (kind == 1 && composer_lease_i1 == NULL) ||
      (kind == 2 && composer_lease_a2 == NULL) ||
      et_g3c4_admit_active_call(context_candidate) != context ||
      et_g3c4_pending_output_lookup(context, &pending) != 0 ||
      pending != composer_lease_output || pending->ids != composer_lease_ids ||
      pending->transport.state != 0u ||
      !composer_lease_pending_zero(pending, kind) ||
      context->cache != composer_lease_cache ||
      context->token_frame_transaction != composer_lease_transaction ||
      context->token_frame_state != composer_lease_frame_state ||
      context->token_frame_position != composer_lease_frame_position ||
      context->token_frame_candidate != composer_lease_candidate ||
      memcmp(context->generator_rng_words, composer_lease_rng,
             sizeof(composer_lease_rng)) != 0 ||
      composer_lease_cache_snapshot(composer_lease_transaction,
          composer_lease_a2, &current_image) != 0 ||
      memcmp(&current_image, &composer_lease_image,
             sizeof(current_image)) != 0 ||
      !coordinator_owner_unchanged()) return -1;
  return 0;
}

int64_t et_g3c4_p2g2_composer_test_lease_end_v1(
    void *context_candidate, void *output_candidate, int64_t kind) {
  et_i64_tensor_error i1_error;
  et_kernel_error a2_error;
  if (composer_lease_phase != 1 || composer_lease_kind != kind ||
      context_candidate != composer_lease_context ||
      output_candidate != composer_lease_output ||
      et_g3c4_p2g2_composer_test_lease_rejected_v1(
          context_candidate, output_candidate, kind) != 0) return -1;
  if (kind == 1) {
    if (et_i64_tensor_borrow_end_v1(&composer_lease_i1, &i1_error) != 0)
      return -1;
  } else if (et_a2_kv_cache_transaction_view_end_v1(
                 &composer_lease_a2, &a2_error) != 0) return -1;
  composer_lease_phase = 2;
  return 0;
}

int64_t et_g3c4_p2g2_composer_test_lease_after_v1(int64_t kind) {
  if (composer_lease_phase != 2 || composer_lease_kind != kind ||
      composer_lease_calls != 2 || composer_lease_second_status != 0 ||
      composer_lease_i1 != NULL || composer_lease_a2 != NULL)
    return -1;
  composer_lease_phase = 0;
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_overlay_mark_v1(
    void *context, int64_t prompt0, int64_t prompt1) {
  composer_overlay_active = 0;
  composer_first_success = composer_second_success = 0;
  composer_overlay_pending = NULL;
  if (et_g3c4_p2g2_coordinator_test_mark_v1(
          context, prompt0, prompt1) != 0)
    return -1;
  composer_overlay_active = 1;
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_overlay_after_v1(
    void *context, int64_t ordinal) {
  if (!composer_overlay_active || !composer_first_success ||
      (ordinal != 1 && ordinal != 2) ||
      (ordinal == 2 && !composer_second_success) ||
      (ordinal == 1 && composer_second_success) ||
      composer_overlay_pending == NULL ||
      et_g3c4_p2g2_coordinator_test_dead_output_v1(
          composer_overlay_pending) != 0 ||
      et_g3c4_p2g2_coordinator_test_after_abort_v1(
          context, ordinal == 1 ? 2 : 3) != 0)
    return -1;
  return 0;
}
int64_t et_g3c4_p2g2_composer_test_zero_marker_v1(int64_t active) {
  if (active != 0 && active != 1) return -1;
  composer_zero_marker = (int)active;
  return 0;
}

int64_t __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
    void *context, void *input, void *staging, int64_t bytes) {
  int64_t result;
  et_g3c4_output_internal *pending = NULL;
  if (composer_fault_mode == 1) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  result = __real_et_g3c4_private_p2g2_eos_first_frame_carrier_v1(
      context, input, staging, bytes);
  if (result != 0 && composer_counter_case == 3) {
    et_g3c4_context_internal *actual = (et_g3c4_context_internal *)context;
    snapshot_cache(actual->cache, &coordinator_precommit_cache);
    snapshot_binding(actual, &coordinator_precommit_binding);
    coordinator_precommit_cache_owner = actual->cache;
    coordinator_precommit_cache_readable = 1;
    composer_exhausted_observed =
        coordinator_precommit_cache.length == 2 &&
        memcmp(coordinator_precommit_cache.keep,
               (const uint8_t[4]){1, 1, 0, 0}, 4u) == 0 &&
        coordinator_precommit_binding.ready == 1u &&
        memcmp(actual->generator_rng_words, coordinator_entry_rng,
               sizeof(coordinator_entry_rng)) == 0;
  }
  if (result == 0 &&
      (et_g3c4_pending_output_lookup(context, &pending) != 0 ||
       pending == NULL ||
       et_g3c4_p2g2_coordinator_test_observe_v1(
           context, pending, 2,
           ((et_g3c4_context_internal *)context)->token_frame_candidate) != 0))
    return -1;
  if (result == 0 && composer_overlay_active) {
    if (pending == NULL || coordinator_precommit_cache_owner == NULL ||
        coordinator_saw_prefix || composer_first_success ||
        memcmp(((et_g3c4_context_internal *)context)->generator_rng_words,
               coordinator_entry_rng, sizeof(coordinator_entry_rng)) != 0)
      abort();
    composer_overlay_pending = pending;
    composer_first_success = 1;
  }
  return result;
}

int64_t et_g3c4_p2g2_composer_test_after_abort_v1(void *context) {
  const int64_t expected = coordinator_saw_prefix ? 3 :
      coordinator_precommit_cache_owner != NULL ? 2 : 0;
  return et_g3c4_p2g2_coordinator_test_after_abort_v1(context, expected);
}

int64_t __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_second_frame_carrier_v1(
    void *context, void *output, void *staging, int64_t bytes) {
  et_g3c4_context_internal *actual = (et_g3c4_context_internal *)context;
  int64_t result;
  if (et_g3c4_p2g2_coordinator_test_prefix_v1(
          context, output, actual->p2g2_prefix_token) != 0)
    return -1;
  if (composer_fault_mode == 2) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  result = __real_et_g3c4_private_p2g2_second_frame_carrier_v1(
      context, output, staging, bytes);
  if (result == 0 && composer_overlay_active) {
    et_a2_kv_cache_transaction_view *view = NULL;
    const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
    const et_kernel_tensor_view_v1 *lengths = NULL, *keep = NULL;
    et_kernel_error error;
    const unsigned char *staged = (const unsigned char *)staging;
    if (!composer_first_success || composer_second_success ||
        output != composer_overlay_pending ||
        !coordinator_saw_prefix || bytes != 16 ||
        actual->token_frame_state != ET_G3C4_TOKEN_FRAME_READY ||
        actual->token_frame_candidate < 0 ||
        actual->token_frame_candidate > 255 ||
        actual->p2g2_second_frame != 1u ||
        et_a2_kv_cache_private_terminal_transaction_witness_v1(
          actual->token_frame_transaction, actual->cache, 3) != 1 ||
        memcmp(actual->generator_rng_words, coordinator_successor_rng,
               sizeof(coordinator_successor_rng)) != 0 ||
        staged[0] != 8 || staged[8] != actual->token_frame_candidate ||
        memcmp(staged + 9, (const unsigned char[7]){0}, 7u) != 0 ||
        et_a2_kv_cache_transaction_view_begin_v1(
          actual->token_frame_transaction, 0u, &view, &error) != 0 ||
        et_a2_kv_cache_transaction_view_tensors_v1(
          view, &keys, &values, &lengths, &keep, &error) != 0)
      abort();
    if (keys == NULL || values == NULL || lengths == NULL || keep == NULL ||
        keys->data == NULL || values->data == NULL ||
        keys->byte_length != sizeof(actual->p2g2_frame_keys) ||
        values->byte_length != sizeof(actual->p2g2_frame_values) ||
        *(const int64_t *)lengths->data != 4 ||
        memcmp(keep->data, (const uint8_t[4]){1, 1, 1, 1}, 4u) != 0 ||
        memcmp(keys->data, actual->p2g2_frame_keys,
               sizeof(actual->p2g2_frame_keys)) != 0 ||
        memcmp(values->data, actual->p2g2_frame_values,
               sizeof(actual->p2g2_frame_values)) != 0 ||
        et_a2_kv_cache_transaction_view_end_v1(&view, &error) != 0)
      abort();
    composer_second_success = 1;
  }
  return result;
}

int64_t __real_et_g3c4_private_p2g2_terminal_commit_v1(
    void *, void *, const void *, int64_t, const void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  if (composer_fault_mode == 3) {
    composer_fault_mode = 0;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  int64_t result = __real_et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, first, first_bytes, second, second_bytes);
  if (result == 0 && composer_zero_marker && write(3, "Z", 1) != 1)
    abort();
  return result;
}

static et_i64_tensor_borrow *composer_clone_borrow;
static et_g3c4_terminal_ids_clone_internal *composer_clone_owner;
static et_i64_tensor_borrow *composer_source_borrow;
static et_g3c4_output_internal *composer_source_owner;
int64_t et_g3c4_p2g2_composer_test_source_lease_v1(
    void *candidate, int64_t active) {
  et_i64_tensor_error error;
  if (active == 1 && composer_source_borrow == NULL) {
    et_g3c4_output_internal *output = et_g3c4_admit_output(candidate, 0);
    if (output == NULL || output->ids == NULL) return -1;
    if (et_i64_tensor_borrow_begin_v1(
          output->ids, &composer_source_borrow, &error) != 0)
      return -1;
    composer_source_owner = output;
    return 0;
  }
  /* Admission requires an unborrowed terminal I1. Match the owner captured
   * on acquire before ending the genuine lease. */
  if (active == 0 && candidate == composer_source_owner &&
      composer_source_borrow != NULL) {
    if (et_i64_tensor_borrow_end_v1(&composer_source_borrow, &error) != 0)
      return -1;
    composer_source_owner = NULL;
    return 0;
  }
  return -1;
}

int64_t et_g3c4_p2g2_composer_test_clone_lease_v1(
    void *candidate, int64_t active) {
  et_g3c4_terminal_ids_clone_internal *clone;
  et_i64_tensor_error error;
  if (active == 1 && composer_clone_borrow == NULL) {
    clone = et_g3c4_terminal_ids_admit(candidate, 0);
    if (clone == NULL || clone->ids == NULL ||
        et_i64_tensor_borrow_begin_v1(
            clone->ids, &composer_clone_borrow, &error) != 0)
      return -1;
    composer_clone_owner = clone;
    return 0;
  }
  if (active == 0 && candidate == composer_clone_owner &&
      composer_clone_borrow != NULL) {
    if (et_i64_tensor_borrow_end_v1(&composer_clone_borrow, &error) != 0)
      return -1;
    composer_clone_owner = NULL;
    return 0;
  }
  return -1;
}
