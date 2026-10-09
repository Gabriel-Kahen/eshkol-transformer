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
  CHECK(et_g3t_private_call_acquire_v1(c, 0) == 0);
  CHECK(et_g3t_private_call_abort_v1(c) == 0);
  CHECK(et_g3t_private_generator_close_v1(c) == 0);
  printf("same-owner combined PASS: %u checks\n", checks);
  return 0;
}
