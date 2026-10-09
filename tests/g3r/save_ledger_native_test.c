/* Owned source test: exercise the real native ledger and snapshot admission. */
#include "src/eshkol_transformer/g3t_transport.c"
#include <stdio.h>

static int checks;
#define CHECK(value) do { \
  ++checks; if (!(value)) { \
    fprintf(stderr, "G3-R ledger test:%d: %s\n", __LINE__, #value); \
    return 1; \
  } \
} while (0)

static void fresh(g3t_context *c, int64_t budget, int64_t eos) {
  memset(c, 0, sizeof(*c));
  c->policy[0] = 0;
  c->policy[1] = INT64_C(0x3f800000);
  c->policy[2] = 256;
  c->policy[3] = INT64_C(0x3f800000);
  c->policy[4] = budget;
  c->policy[5] = eos;
  c->frame.active = 1;
}
static int committed(g3t_context *c) {
  CHECK(g3r_ledger_stage(c) == 0);
  CHECK(g3r_ledger_check(c) == 0);
  memcpy(&c->ledger, &c->frame.candidate_ledger, sizeof(c->ledger));
  memset(&c->frame, 0, sizeof(c->frame));
  return 0;
}
static int ledger_routes(void) {
  g3t_context c;
  fresh(&c, 0, 43);
  c.call_kind = 0; c.frame.kind = 1;
  c.frame.prompt_length = 1; c.frame.token = 41;
  CHECK(g3r_ledger_stage(&c) == 0);
  CHECK(c.ledger.length == 0 && c.frame.candidate_ledger.length == 1);
  c.frame.token = 42;
  CHECK(g3r_ledger_check(&c) != 0);
  CHECK(c.ledger.length == 0);
  c.frame.token = 41;
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.prompt == 1 && c.ledger.ids[0] == 41 && !c.ledger.terminal);

  c.call_kind = 1; c.frame.active = 1; c.frame.kind = 2;
  c.frame.prompt_length = 1; c.frame.token = 43;
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.length == 2 && c.ledger.manual == 1 &&
        c.ledger.generated == 0 && c.ledger.terminal == 0);
  c.frame.active = 1; c.frame.kind = 2; c.frame.token = 44;
  CHECK(g3r_ledger_stage(&c) != 0);

  fresh(&c, 0, 43);
  c.call_kind = 0; c.frame.kind = 1;
  c.frame.prompt_length = 2; c.frame.token = 41;
  c.frame.prompt_ids[0] = 41; c.frame.prompt_ids[1] = 43;
  c.frame.prompt_ids[0] = 42;
  CHECK(g3r_ledger_stage(&c) != 0);
  c.frame.prompt_ids[0] = 41;
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.length == 2 && c.ledger.prompt == 2 &&
        c.ledger.ids[1] == 43 && c.ledger.terminal == 0);

  fresh(&c, 0, 43);
  c.call_kind = 2; c.frame.kind = 1;
  c.frame.prompt_length = 2; c.frame.token = 41;
  c.frame.prompt_ids[0] = 41; c.frame.prompt_ids[1] = 43;
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.length == 2 && c.ledger.generated == 0 &&
        c.ledger.terminal == 0 && !c.ledger.pending_g1);

  fresh(&c, 1, 43);
  c.call_kind = 2; c.frame.kind = 1;
  c.frame.prompt_length = 1; c.frame.token = 41;
  CHECK(g3r_ledger_stage(&c) == 0);
  CHECK(c.ledger.length == 0);
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.length == 1 && c.ledger.pending_g1 &&
        c.ledger.generated == 0);
  c.frame.active = 1; c.frame.kind = 2;
  c.frame.prompt_length = 1; c.frame.token = 43;
  c.sampled = 1; c.sample_token = 43;
  CHECK(committed(&c) == 0);
  CHECK(c.ledger.length == 2 && c.ledger.generated == 1 &&
        !c.ledger.pending_g1 && c.ledger.terminal == 1);
  return 0;
}
static int snapshot_ownership(void) {
  _Static_assert(sizeof(et_g3r_g3t_idle_snapshot_words_v1) == 144,
                 "fixed private host ABI");
  g3r_save_snapshot owner = {0};
  et_g3r_g3t_idle_snapshot_words_v1 copied;
  memset(&copied, 0x5a, sizeof(copied));
  owner.h.kind = G3T_SAVE_SNAPSHOT;
  owner.h.state = G3T_LIVE;
  owner.value.word[0] = 1;
  owner.value.word[7] = 1;
  owner.value.word[8] = INT64_MAX;
  owner.value.word[9] = -1;
  owner.value.word[10] = INT64_MIN;
  g3t_registry = &owner.h;
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      &owner, &copied, 143) != 0);
  CHECK(copied.word[0] == INT64_C(0x5a5a5a5a5a5a5a5a));
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      &owner, &copied, 144) == 0);
  CHECK(copied.word[0] == 1 && copied.word[8] == INT64_MAX &&
        copied.word[9] == -1 && copied.word[10] == INT64_MIN);
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      &owner, &owner.value, 144) != 0);
  CHECK(owner.value.word[0] == 1);
  g3t_record foreign = {0};
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      &foreign, &copied, 144) != 0);
  owner.h.busy = 1;
  CHECK(et_g3r_private_g3t_idle_snapshot_release_v1(&owner) != 0);
  owner.h.busy = 0;
  CHECK(et_g3r_private_g3t_idle_snapshot_release_v1(&owner) == 0);
  CHECK(et_g3r_private_g3t_idle_snapshot_release_v1(&owner) != 0);
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      &owner, &copied, 144) != 0);
  CHECK(owner.value.word[0] == 0);
  g3t_registry = NULL;
  return 0;
}
#ifdef ET_G3R_AUTH_TEST
static int stage_real_first_frame(g3t_context *c) {
  int64_t count_data[1] = {1};
  uint64_t count_shape[1] = {1}, kv_shape[4] = {1, 2, 1, 2};
  float key_data[4] = {0}, value_data[4] = {0};
  et_kernel_tensor_view_v1 count = g3t_view(
      count_data, sizeof(count_data), "i64", 1, count_shape);
  et_kernel_tensor_view_v1 key = g3t_view(
      key_data, sizeof(key_data), "f32", 4, kv_shape);
  et_kernel_tensor_view_v1 value = g3t_view(
      value_data, sizeof(value_data), "f32", 4, kv_shape);
  et_kernel_error error;
  CHECK(et_a2_kv_cache_create_v1(1, 1, 2, 2, 2,
                                &c->frame.candidate_cache, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      c->frame.candidate_cache, 1, &count, &c->frame.transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      c->frame.transaction, 0, &key, &value, &error) == 0);
  c->frame.active = c->frame.prepared = 1;
  c->frame.kind = 1;
  c->frame.next_ordinal = 21;
  c->frame.token = 41;
  c->frame.prompt_length = 1;
  CHECK(g3r_ledger_stage(c) == 0);
  return 0;
}
static int authentic_first_commit(owner *model) {
  et_kernel_error error;
  g3t_context *c = et_g3t_private_generator_seed_v1(
      model, 1729, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1);
  CHECK(c != NULL);
  CHECK(et_g3t_private_call_acquire_v1(c, 2) == 0);
  CHECK(et_g3t_private_output_reserve_v1(c, 1) != NULL);
  CHECK(stage_real_first_frame(c) == 0);
  et_a2_kv_cache_transaction_view *held_view = NULL;
  CHECK(et_a2_kv_cache_transaction_view_begin_v1(
      c->frame.transaction, 0, &held_view, &error) == 0);
  CHECK(et_g3t_private_frame_commit_v1(c) != 0);
  CHECK(g3t_error_domain == 1);
  CHECK(c->ledger.length == 0 && c->frame.transaction != NULL &&
        c->frame.candidate_cache != NULL);
  CHECK(et_a2_kv_cache_transaction_view_end_v1(&held_view, &error) == 0);
  CHECK(et_g3t_private_frame_commit_v1(c) == 0);
  CHECK(c->ledger.length == 1 && c->ledger.prompt == 1 &&
        c->ledger.ids[0] == 41 && c->ledger.pending_g1 == 1 &&
        c->frame.active == 0 && c->frame.candidate_cache == NULL);
  et_a2_kv_cache_read_borrow *committed = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *mask = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      c->cache, &committed, &error) == 0);
  CHECK(et_a2_kv_cache_read_borrow_layer_v1(
      committed, 0, &keys, &values, &lengths, &mask, &error) == 0);
  CHECK(((const int64_t *)lengths->data)[0] == 1 &&
        ((const uint8_t *)mask->data)[0] == 1 &&
        ((const uint8_t *)mask->data)[1] == 0);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&committed, &error) == 0);
  CHECK(et_g3t_private_call_abort_v1(c) == 0);
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  CHECK(et_g3t_private_generator_close_v1(c) == 0);

  c = et_g3t_private_generator_seed_v1(
      model, 1729, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 1, -1);
  CHECK(c != NULL);
  CHECK(et_g3t_private_call_acquire_v1(c, 2) == 0);
  CHECK(et_g3t_private_output_reserve_v1(c, 1) != NULL);
  CHECK(stage_real_first_frame(c) == 0);
  et_a2_kv_cache_read_borrow *held_old = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      c->cache, &held_old, &error) == 0);
  CHECK(et_g3t_private_frame_commit_v1(c) != 0);
  CHECK(g3t_error_domain == 1);
  CHECK(c->ledger.length == 0 && c->frame.transaction == NULL &&
        c->frame.candidate_cache != NULL && c->prefill_committed == 0);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&held_old, &error) == 0);
  CHECK(et_g3t_private_call_abort_v1(c) == 0);
  CHECK(et_g3t_private_generator_close_v1(c) == 0);
  return 0;
}
static int authentic_snapshot(void) {
  static unsigned char identities[14];
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial != NULL);
  CHECK(et_m3t_private_initializer_seal_v1(initial) == 0);
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model != NULL);
  for (int64_t index = 0; index < 14; ++index)
    CHECK(et_m3t_private_owner_bind_v1(
        model, index, &identities[index]) == 0);
  for (int64_t index = 0; index < 14; ++index)
    CHECK(et_m3t_private_owner_initialize_v1(model, index) == 0);
  CHECK(et_m3t_private_owner_seal_v1(model) == 0);
  CHECK(et_m3t_private_initializer_seal_v1(model->successor) == 0);
  g3t_context *c = et_g3t_private_generator_seed_v1(
      model, 1729, 0, INT64_C(0x3f800000), 256,
      INT64_C(0x3f800000), 0, -1);
  CHECK(c != NULL);
  int64_t count_data[1] = {1};
  uint64_t count_shape[1] = {1}, kv_shape[4] = {1, 2, 1, 2};
  float key_data[4] = {0}, value_data[4] = {0};
  et_kernel_tensor_view_v1 count = g3t_view(
      count_data, sizeof(count_data), "i64", 1, count_shape);
  et_kernel_tensor_view_v1 key = g3t_view(
      key_data, sizeof(key_data), "f32", 4, kv_shape);
  et_kernel_tensor_view_v1 value = g3t_view(
      value_data, sizeof(value_data), "f32", 4, kv_shape);
  et_a2_kv_cache_transaction *transaction = NULL;
  et_kernel_error error;
  CHECK(et_a2_kv_cache_transaction_begin_v1(
      c->cache, 1, &count, &transaction, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
      transaction, 0, &key, &value, &error) == 0);
  CHECK(et_a2_kv_cache_transaction_commit_v1(&transaction, &error) == 0);
  /* Test-only committed-state fixture; real frame-commit paths are separate. */
  c->ledger.ids[0] = 41;
  c->ledger.length = c->ledger.prompt = 1;
  c->prefill_committed = c->binding_ready = 1;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  CHECK(et_a2_kv_cache_read_borrow_begin_v1(
      c->cache, &borrow, &error) == 0);
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) == 0);
  c->ledger.pending_g1 = 1;
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  c->ledger.pending_g1 = 0;
  c->h.busy = 1;
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  c->h.busy = 0;
  c->ledger.ids[0] = 256;
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  c->ledger.ids[0] = 41;
  c->ledger.length = 2;
  c->ledger.prompt = 2;
  c->ledger.ids[1] = 42;
  CHECK(et_g3r_private_g3t_idle_snapshot_create_v1(c) == NULL);
  c->ledger.ids[1] = 0;
  c->ledger.prompt = 1;
  c->ledger.length = 1;
  void *snapshot = et_g3r_private_g3t_idle_snapshot_create_v1(c);
  CHECK(snapshot != NULL);
  et_g3r_g3t_idle_snapshot_words_v1 words = {0};
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      snapshot, &words, sizeof(words)) == 0);
  CHECK(words.word[0] == 1 && words.word[8] == 1729 &&
        words.word[11] == 41 && words.word[13] == 1 &&
        words.word[17] == 1);
  CHECK(et_g3t_private_generator_close_v1(c) == 0);
  CHECK(et_g3r_private_g3t_idle_snapshot_copy_v1(
      snapshot, &words, sizeof(words)) == 0);
  CHECK(words.word[11] == 41 && words.word[17] == 1);
  CHECK(et_g3r_private_g3t_idle_snapshot_release_v1(snapshot) == 0);
  CHECK(authentic_first_commit(model) == 0);
  return 0;
}
#endif
int main(void) {
  if (ledger_routes() || snapshot_ownership()) return 1;
#ifdef ET_G3R_AUTH_TEST
  if (authentic_snapshot()) return 1;
#endif
  printf("G3-R save ledger source test PASS: %d checks\n", checks);
  return 0;
}
