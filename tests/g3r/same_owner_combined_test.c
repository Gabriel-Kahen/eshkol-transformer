/* Host-only same-owner native composition; no C2 state or file publication. */
#define ET_O2_TESTING 1
#include "src/eshkol_transformer/g3t_transport.c"
#include "native/o2_optimizer.c"
#include "native/g3r_same_owner_save.h"
#include "native/o2_optimizer_internal.h"
#include <stdio.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, \
    "same-owner combined:%d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void) {
  static unsigned char identities[14];
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial != NULL);
  CHECK(et_m3t_private_initializer_seal_v1(initial) == 0);
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model != NULL);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_bind_v1(
        model, i, &identities[i]) == 0);
  for (int64_t i = 0; i < 14; ++i)
    CHECK(et_m3t_private_owner_initialize_v1(model, i) == 0);
  CHECK(et_m3t_private_owner_seal_v1(model) == 0);
  CHECK(et_m3t_private_initializer_seal_v1(model->successor) == 0);

  et_o2_optimizer_builder *builder = NULL;
  et_o2_optimizer *optimizer = NULL;
  et_o2_error_v1 o2;
  CHECK(et_o2_optimizer_builder_create_v1(
      14, ET_O2_CLIP_NONE, 0, ET_O2_SCHEDULE_CONSTANT,
      0, 0, UINT32_C(0x3f800000), &builder, &o2) == 0);
  for (size_t i = 0; i < 14; ++i)
    CHECK(et_o2_optimizer_builder_set_v1(
        builder, i, model->p[i], model->handles[i],
        UINT32_C(0x3ca3d70a), UINT32_C(0x3f4ccccd),
        UINT32_C(0x3f7d70a4), UINT32_C(0x322bcc77), 0, &o2) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(
      &builder, &optimizer, &o2) == 0);

  g3t_context *c = et_g3t_private_generator_seed_v1(
      model, 1729, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 0, -1);
  CHECK(c != NULL);
  int64_t count_data[1] = {1};
  uint64_t count_shape[1] = {1}, kv_shape[4] = {1, 2, 1, 2};
  float key_data[4] = {0}, value_data[4] = {0};
  et_kernel_tensor_view_v1 count =
      g3t_view(count_data, sizeof(count_data), "i64", 1, count_shape);
  et_kernel_tensor_view_v1 key =
      g3t_view(key_data, sizeof(key_data), "f32", 4, kv_shape);
  et_kernel_tensor_view_v1 value =
      g3t_view(value_data, sizeof(value_data), "f32", 4, kv_shape);
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_error kernel;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      c->cache, 1, &count, &transaction, &kernel) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      transaction, 0, &key, &value, &kernel) == 0);
  CHECK(et_a2_kv_cache_transaction_commit_v1(
      &transaction, &kernel) == 0);
  c->ledger.ids[0] = 41;
  c->ledger.length = c->ledger.prompt = 1;
  c->prefill_committed = c->binding_ready = 1;
  et_f32_tensor_error f32;
  CHECK(et_g3t_model_pins_begin_internal(
      model->p, (const void *const *)model->handles, &c->pins, &f32) == 0);
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *view = &c->pins.views[i];
    CHECK(offset + view->byte_length <= sizeof(c->binding_values));
    c->binding_identities[i] = model->handles[i];
    memcpy(c->binding_values + offset, view->data, view->byte_length);
    offset += view->byte_length;
  }
  CHECK(offset == sizeof(c->binding_values));
  et_g3t_model_pins_end_internal(&c->pins);

  void *roster_parameters[14] = {0};
  const void *roster_identities[14] = {0};
  CHECK(et_m3t_private_owner_save_roster_v1(
      (void *)(uintptr_t)1u, roster_parameters, roster_identities) != 0);
  CHECK(et_m3t_private_owner_save_roster_v1(
      model, roster_parameters, roster_identities) == 0);
  CHECK(roster_parameters[0] == model->p[0] &&
        roster_identities[13] == model->handles[13]);
  void *original_handle = model->handles[13];
  model->handles[13] = NULL;
  CHECK(et_m3t_private_owner_save_roster_v1(
      model, roster_parameters, roster_identities) != 0);
  model->handles[13] = original_handle;
  et_g3r_same_owner_save_error_v1 first = {0}, observed = {0};
  CHECK(et_g3r_private_same_owner_save_begin_v1(
      c, model, (void *)(uintptr_t)1u) == NULL);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 && first.domain == 0 &&
        first.category == ET_O2_STATUS_INVALID_STATE &&
        first.code == ET_O2_CODE_INVALID_HANDLE);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  et_a2_kv_cache_read_borrow *blocked = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      c->cache, &blocked, &kernel) == 0);
  CHECK(et_g3r_private_same_owner_save_begin_v1(
      c, model, optimizer) == NULL);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_G3T && first.domain == 1 &&
        first.category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
        first.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  et_o2_optimizer_state *ordinary = NULL;
  CHECK(et_o2_optimizer_state_snapshot_v1(
      optimizer, &ordinary, &o2) == 0);
  CHECK(et_o2_optimizer_state_release_v1(ordinary, &o2) == 0);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&blocked, &kernel) == 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);

  void *hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_check_v1((void *)(uintptr_t)1u) != 0);
  et_a2_kv_cache_transaction *denied = NULL;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      c->cache, 1, &count, &denied, &kernel) != 0);
  CHECK(denied == NULL);
  et_a2_kv_cache *cache_alias = c->cache;
  CHECK(et_a2_kv_cache_destroy_v1(&cache_alias, &kernel) != 0);
  CHECK(cache_alias == c->cache);
  CHECK(et_g3t_private_call_acquire_v1(c, 0) != 0);
  CHECK(et_o2_optimizer_step_v1(optimizer, &o2) ==
        ET_O2_STATUS_INVALID_STATE);
  const et_f32_tensor *value_owner = NULL;
  CHECK(et_f32_parameter_value_tensor_v1(
      model->p[0], &value_owner, &f32) == 0);
  uint32_t replacement[16] = {0};
  CHECK(et_f32_tensor_copy_bits_from_v1(
      (et_f32_tensor *)(uintptr_t)value_owner,
      replacement, 16, &f32) == ET_F32_TENSOR_ERROR_INVALID_STATE);
  for (size_t i = 0; i < 14; ++i) {
    et_f32_tensor *p1 = NULL;
    int32_t equal = 0;
    CHECK(et_f32_parameter_validate_identity_v1(
        model->p[i], model->handles[i], &f32) == 0);
    CHECK(et_f32_parameter_value_snapshot_v1(model->p[i], &p1, &f32) == 0);
    CHECK(et_f32_tensor_bits_equal_v1(
        p1, c->pins.values[i], &equal, &f32) == 0 && equal);
    CHECK(et_f32_tensor_destroy_v1(&p1, &f32) == 0);
  }
  void *state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(
      hold, &state) == 0);
  CHECK(state != NULL);
  et_g3r_g3t_idle_snapshot_words_v1 words = {0};
  CHECK(et_g3r_private_same_owner_save_copy_v1(
      hold, &words, sizeof(words)) == 0);
  CHECK(words.word[0] == 1 && words.word[8] == 1729 &&
        words.word[11] == 41 && words.word[17] == 1);
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(hold, state) == 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &o2) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      c->cache, 1, &count, &denied, &kernel) == 0);
  CHECK(et_a2_kv_cache_transaction_abort_v1(&denied, &kernel) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_copy_v1(
      hold, &words, sizeof(words) - 1) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_G3T && first.domain == 0 &&
        first.category == G3T_ARGUMENT && first.code == G3T_IDENTITY);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  et_f32_tensor *moment = optimizer->entries[0].exp_avg;
  size_t moment_count = 0;
  uint32_t moment_bits[1192] = {0};
  CHECK(et_f32_tensor_element_count_v1(moment, &moment_count, &f32) == 0);
  CHECK(moment_count > 0 && moment_count <= 1192);
  CHECK(et_f32_tensor_copy_bits_to_v1(
      moment, moment_bits, moment_count, &f32) == 0);
  uint32_t unchanged = moment_bits[0];
  moment_bits[0] ^= 1u;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      moment, moment_bits, moment_count, &f32) == 0);
  const et_kernel_tensor_view_v1 *held_keys = NULL, *held_values = NULL;
  const et_kernel_tensor_view_v1 *held_lengths = NULL, *held_mask = NULL;
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      c->save_hold->cache_borrow, 0, &held_keys, &held_values,
      &held_lengths, &held_mask, &kernel) == 0);
  unsigned char *held_bytes = (unsigned char *)(uintptr_t)held_keys->data;
  held_bytes[0] ^= 1u;
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_G3T && first.domain == 0 &&
        first.category == G3T_STATE && first.code == G3T_INVARIANT);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
  held_bytes[0] ^= 1u;
  moment_bits[0] = unchanged;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      moment, moment_bits, moment_count, &f32) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  et_o2_test_fail_alloc_after_v1(0);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) != 0);
  CHECK(state == NULL);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 && first.domain == 0 &&
        first.category == ET_O2_STATUS_INTERNAL &&
        first.code == ET_O2_CODE_ALLOCATION_FAILED);
  et_o2_test_reset_failpoints_v1();
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  held_keys = held_values = held_lengths = held_mask = NULL;
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      c->save_hold->cache_borrow, 0, &held_keys, &held_values,
      &held_lengths, &held_mask, &kernel) == 0);
  held_bytes = (unsigned char *)(uintptr_t)held_keys->data;
  held_bytes[0] ^= 1u;
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_G3T && first.domain == 0 &&
        first.category == G3T_STATE && first.code == G3T_INVARIANT);
  held_bytes[0] ^= 1u;
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  moment_bits[0] ^= 1u;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      moment, moment_bits, moment_count, &f32) == 0);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 && first.domain == 0 &&
        first.category == ET_O2_STATUS_INVALID_STATE &&
        first.code == ET_O2_CODE_OWNER_CONFLICT);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  moment_bits[0] = unchanged;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      moment, moment_bits, moment_count, &f32) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  c->rng[1]++;
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  c->rng[1]--;
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  void *alias = state;
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(hold, alias) == 0);
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(hold, alias) != 0);
  CHECK(et_o2_optimizer_state_release_v1(state, &o2) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  CHECK(et_g3r_private_same_owner_save_o2_cancel_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) != 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, NULL) != 0);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) != 0);
  CHECK(state == NULL);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(
      hold, (void *)(uintptr_t)1u) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_COORDINATOR &&
        first.code == ET_G3R_SAVE_COORD_ISSUE);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  et_o2_optimizer_state *issued = state;
  uint32_t original_option = issued->entries[0].options.learning_rate_bits;
  issued->entries[0].options.learning_rate_bits ^= 1u;
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(hold, issued) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 &&
        first.code == ET_O2_CODE_OWNER_CONFLICT);
  issued->entries[0].options.learning_rate_bits = original_option;
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_o2_optimizer_state_lifecycle_v1(issued,
        (uint32_t[1]){0}, &o2) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  issued = state;
  et_f32_tensor *issued_moment = issued->owned_clones[0];
  size_t issued_count = 0;
  CHECK(et_f32_tensor_element_count_v1(
      issued_moment, &issued_count, &f32) == 0);
  CHECK(issued_count > 0 && issued_count <= 1192);
  CHECK(et_f32_tensor_copy_bits_to_v1(
      issued_moment, moment_bits, issued_count, &f32) == 0);
  uint32_t issued_first = moment_bits[0];
  moment_bits[0] ^= 1u;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      issued_moment, moment_bits, issued_count, &f32) == 0);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_o2_claim_v1(hold, issued) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 &&
        first.code == ET_O2_CODE_OWNER_CONFLICT);
  moment_bits[0] = issued_first;
  CHECK(et_f32_tensor_copy_bits_from_v1(
      issued_moment, moment_bits, issued_count, &f32) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  et_o2_optimizer_state_handle *moment_handle = NULL;
  et_o2_optimizer_state_borrow *state_borrow = NULL;
  const et_kernel_tensor_view_v1 *state_view = NULL;
  CHECK(et_o2_optimizer_state_moment_handle_v1(
      state, 0, ET_O2_MOMENT_EXP_AVG, &moment_handle, &o2) == 0);
  CHECK(et_o2_optimizer_state_borrow_begin_v1(
      state, moment_handle, &state_borrow, &state_view, &o2) == 0);
  CHECK(state_view != NULL);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(optimizer->busy == 1 && c->save_hold != NULL);
  CHECK(et_g3r_private_same_owner_save_cleanup_retry_v1(hold) != 0);
  CHECK(optimizer->busy == 1 && c->save_hold != NULL);
  CHECK(et_o2_optimizer_state_borrow_end_v1(&state_borrow, &o2) == 0);
  CHECK(et_g3r_private_same_owner_save_cleanup_retry_v1(hold) != 0);
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  et_o2_test_fail_release_after_v1(0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_O2 &&
        first.code == ET_O2_CODE_PROVIDER_DEFECT);
  uint32_t lifecycle = 0;
  CHECK(et_o2_optimizer_state_lifecycle_v1(
      state, &lifecycle, &o2) == 0 &&
      lifecycle == ET_O2_OPTIMIZER_STATE_DEAD);
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
  et_o2_test_reset_failpoints_v1();
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  state = NULL;
  CHECK(et_g3r_private_same_owner_save_o2_state_v1(hold, &state) == 0);
  ((et_o2_optimizer_state *)state)->lifecycle =
      ET_O2_OPTIMIZER_STATE_RELEASING;
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(optimizer->busy == 1 && c->save_hold != NULL);
  CHECK(et_g3r_private_same_owner_save_cleanup_retry_v1(hold) != 0);
  CHECK(optimizer->busy == 1 && c->save_hold != NULL);
  ((et_o2_optimizer_state *)state)->lifecycle = ET_O2_OPTIMIZER_STATE_LIVE;
  CHECK(et_g3r_private_same_owner_save_cleanup_retry_v1(hold) != 0);
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_f32_test_live_counts_v1 during = {.struct_size = sizeof(during)};
  et_f32_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  et_f32_test_live_counts_snapshot_v1(&before);
  et_f32_tensor_test_fail_alloc_after_v1(SIZE_MAX);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  const size_t complete_begin_allocations =
      et_f32_tensor_test_successful_allocations_v1();
  /* O2's 28 clones and fourteen witnesses each use four allocations per
   * tensor; every witness also allocates one token. */
  CHECK(complete_begin_allocations == 28u * 4u + 14u * 9u);
  et_f32_test_live_counts_snapshot_v1(&during);
  CHECK(during.tensors == before.tensors + 56 &&
        during.borrows == before.borrows);
  uint64_t first_shape[2] = {4, 4};
  et_f32_parameter_save_witness *extra = NULL;
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape, &extra, &f32) == 0);
  CHECK(et_f32_parameter_save_witness_check_v1(extra, &f32) == 0);
  const float *extra_value = et_f32_tensor_test_data_storage_v1(value_owner);
  CHECK(extra_value != NULL);
  ((unsigned char *)(uintptr_t)extra_value)[0] ^= 1u;
  CHECK(et_f32_parameter_save_witness_check_v1(extra, &f32) != 0);
  ((unsigned char *)(uintptr_t)extra_value)[0] ^= 1u;
  CHECK(et_f32_parameter_save_witness_check_v1(extra, &f32) == 0);
  et_f32_test_tensor_metadata_v1 extra_metadata = {
      .struct_size = sizeof(extra_metadata)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(value_owner,
                                                &extra_metadata) == 0);
  CHECK(et_f32_tensor_test_metadata_corrupt_v1(
      (et_f32_tensor *)(uintptr_t)value_owner,
      ET_F32_TEST_TENSOR_METADATA_DATA, 0) == 0);
  CHECK(et_f32_parameter_save_witness_check_v1(extra, &f32) != 0);
  CHECK(et_f32_tensor_test_metadata_restore_v1(
      (et_f32_tensor *)(uintptr_t)value_owner, &extra_metadata) == 0);
  CHECK(et_f32_parameter_save_witness_check_v1(extra, &f32) == 0);
  CHECK(et_f32_parameter_save_witness_check_v1(
      (void *)(uintptr_t)1u, &f32) != 0);
  et_f32_parameter_save_witness *peer = NULL, *peer_link_before = NULL;
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape, &peer, &f32) == 0);
  memcpy(&peer_link_before, peer, sizeof(peer_link_before));
  CHECK(peer_link_before == extra);
  et_f32_test_live_counts_v1 alias_before = {
      .struct_size = sizeof(alias_before)};
  et_f32_test_live_counts_snapshot_v1(&alias_before);
  et_f32_parameter_save_witness **peer_link =
      (et_f32_parameter_save_witness **)(void *)peer;
  CHECK(et_f32_parameter_save_witness_end_v1(peer_link, &f32) != 0);
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape, peer_link, &f32) != 0);
  CHECK(et_f32_parameter_save_witness_check_v1(
      extra, (et_f32_tensor_error *)(void *)peer) != 0);
  CHECK(et_f32_parameter_save_witness_end_v1(
      &extra, (et_f32_tensor_error *)(void *)peer) != 0);
  CHECK(et_f32_tensor_clone_v1(
      value_owner, (et_f32_tensor **)(void *)peer, &f32) != 0);
  et_f32_tensor *generic_clone = NULL;
  CHECK(et_f32_tensor_clone_v1(
      value_owner, &generic_clone,
      (et_f32_tensor_error *)(void *)peer) != 0 &&
      generic_clone == NULL);
  et_f32_parameter_save_witness *peer_link_after = NULL;
  memcpy(&peer_link_after, peer, sizeof(peer_link_after));
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(peer_link_after == peer_link_before && extra != NULL &&
        after.tensors == alias_before.tensors &&
        after.borrows == alias_before.borrows &&
        et_f32_parameter_save_witness_check_v1(peer, &f32) == 0 &&
        et_f32_parameter_save_witness_check_v1(extra, &f32) == 0);
  CHECK(et_f32_parameter_save_witness_end_v1(&peer, &f32) == 0 &&
        peer == NULL);
  et_f32_parameter_save_witness *stale = extra;
  CHECK(et_f32_parameter_save_witness_end_v1(&extra, &f32) == 0 &&
        extra == NULL);
  CHECK(et_f32_parameter_save_witness_check_v1(stale, &f32) != 0);
  et_f32_parameter_save_witness *retired_link_before = NULL;
  memcpy(&retired_link_before, stale, sizeof(retired_link_before));
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape,
      (et_f32_parameter_save_witness **)(void *)stale, &f32) != 0);
  CHECK(et_f32_parameter_save_witness_check_v1(
      peer_link_before, (et_f32_tensor_error *)(void *)stale) != 0);
  CHECK(et_f32_tensor_clone_v1(
      value_owner, (et_f32_tensor **)(void *)stale, &f32) != 0);
  et_f32_parameter_save_witness *retired_link_after = NULL;
  memcpy(&retired_link_after, stale, sizeof(retired_link_after));
  CHECK(retired_link_after == retired_link_before);
  et_f32_test_live_counts_v1 before_clone_cut = {
      .struct_size = sizeof(before_clone_cut)};
  et_f32_test_live_counts_snapshot_v1(&before_clone_cut);
  et_f32_tensor_test_fail_alloc_after_v1(5u);
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape, &extra, &f32) != 0);
  CHECK(extra == NULL &&
        et_f32_tensor_test_successful_allocations_v1() == 5u &&
        f32.category == ET_F32_TENSOR_ERROR_INTERNAL &&
        f32.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  et_f32_tensor_test_reset_allocator_v1();
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(after.tensors == before_clone_cut.tensors &&
        after.borrows == before_clone_cut.borrows &&
        optimizer->busy == 1u && c->save_hold != NULL &&
        et_g3r_private_same_owner_save_check_v1(hold) == 0);
  CHECK(et_f32_parameter_save_witness_begin_v1(
      model->p[0], model->handles[0], 2, first_shape, &extra, &f32) == 0);
  CHECK(et_f32_parameter_save_witness_end_v1(&extra, &f32) == 0);
  et_f32_gradient_reset_plan *reset = NULL;
  CHECK(et_f32_gradient_reset_plan_prepare_v1(
      1, &model->p[0], &reset, &f32) != 0 && reset == NULL);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(after.tensors == before.tensors && after.borrows == before.borrows);

  et_f32_parameter_save_witness_test_fail_after_v1(8);
  CHECK(et_g3r_private_same_owner_save_begin_v1(c, model, optimizer) == NULL);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_F32 && first.domain == 3 &&
        first.category == ET_F32_TENSOR_ERROR_INTERNAL &&
        first.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(after.tensors == before.tensors && after.borrows == before.borrows);
  et_f32_parameter_save_witness_test_fail_after_v1(SIZE_MAX);
  /* Cut inside the second baseline clone of the first witness in the real
   * combined begin path; O2/G3 leases and the first clone must be drained. */
  const size_t clone_cut = complete_begin_allocations - 14u * 9u + 5u;
  et_f32_tensor_test_fail_alloc_after_v1(clone_cut);
  CHECK(et_g3r_private_same_owner_save_begin_v1(c, model, optimizer) == NULL);
  CHECK(et_f32_tensor_test_successful_allocations_v1() == clone_cut);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_F32 && first.domain == 3 &&
        first.category == ET_F32_TENSOR_ERROR_INTERNAL &&
        first.code == ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(optimizer->busy == 0 && c->save_hold == NULL);
  et_f32_test_live_counts_snapshot_v1(&after);
  CHECK(after.tensors == before.tensors && after.borrows == before.borrows);
  et_g3t_model_pins_internal reacquired = {0};
  CHECK(et_g3t_model_pins_begin_internal(
      model->p, (const void *const *)model->handles, &reacquired, &f32) == 0);
  CHECK(et_g3t_model_pins_check_internal(&reacquired, &f32) == 0);
  et_g3t_model_pins_end_internal(&reacquired);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);

  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  const float *value_storage = et_f32_tensor_test_data_storage_v1(value_owner);
  CHECK(value_storage != NULL);
  uint32_t original_value_bits = 0;
  memcpy(&original_value_bits, value_storage, sizeof(original_value_bits));
  ((unsigned char *)(uintptr_t)value_storage)[0] ^= 1u;
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0 &&
        first.source == ET_G3R_SAVE_SOURCE_G3T);
  memcpy((void *)(uintptr_t)value_storage, &original_value_bits,
         sizeof(original_value_bits));
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);

  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  et_f32_test_tensor_metadata_v1 value_metadata = {
      .struct_size = sizeof(value_metadata)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(value_owner,
                                                &value_metadata) == 0);
  CHECK(et_f32_tensor_test_metadata_corrupt_v1(
      (et_f32_tensor *)(uintptr_t)value_owner,
      ET_F32_TEST_TENSOR_METADATA_SHAPE, 0) == 0);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_f32_tensor_test_metadata_restore_v1(
      (et_f32_tensor *)(uintptr_t)value_owner, &value_metadata) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);

  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  uint32_t altered_gradient[16] = {0};
  altered_gradient[0] = UINT32_C(0x3f800000);
  et_f32_parameter_test_set_gradient_bits_v1(
      model->p[0], altered_gradient, 16);
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0);
  CHECK(first.source == ET_G3R_SAVE_SOURCE_F32 && first.domain == 3 &&
        first.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
        first.code == ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&observed) == 0 &&
        memcmp(&observed, &first, sizeof(first)) == 0);
  altered_gradient[0] = 0;
  et_f32_parameter_test_set_gradient_bits_v1(
      model->p[0], altered_gradient, 16);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  et_f32_parameter_test_set_metadata_v1(
      model->p[0], ET_F32_GRADIENT_PRESENT, 1, UINT32_C(0x3f800000));
  CHECK(et_g3r_private_same_owner_save_check_v1(hold) != 0);
  CHECK(et_g3r_private_same_owner_save_last_error_v1(&first) == 0 &&
        first.source == ET_G3R_SAVE_SOURCE_F32);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) != 0);
  et_f32_parameter_test_set_metadata_v1(
      model->p[0], ET_F32_GRADIENT_ABSENT, 0, 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);
  hold = et_g3r_private_same_owner_save_begin_v1(c, model, optimizer);
  CHECK(hold != NULL);
  int exhausted = 0;
  for (size_t attempt = 0u; attempt < 897u; ++attempt) {
    extra = NULL;
    if (et_f32_parameter_save_witness_begin_v1(
            model->p[0], model->handles[0], 2, first_shape,
            &extra, &f32) != 0) {
      CHECK(extra == NULL && f32.category == ET_F32_TENSOR_ERROR_INVALID_STATE &&
            f32.code == ET_F32_TENSOR_CODE_INVALID_HANDLE);
      exhausted = 1;
      break;
    }
    CHECK(et_f32_parameter_save_witness_end_v1(&extra, &f32) == 0);
  }
  CHECK(exhausted && et_g3r_private_same_owner_save_check_v1(hold) == 0);
  CHECK(et_g3r_private_same_owner_save_end_v1(hold) == 0);
#endif
  CHECK(et_g3t_private_call_acquire_v1(c, 0) == 0);
  CHECK(et_g3t_private_call_abort_v1(c) == 0);
  CHECK(et_g3t_private_generator_close_v1(c) == 0);
  printf("same-owner combined PASS: %u checks\n", checks);
  return 0;
}
