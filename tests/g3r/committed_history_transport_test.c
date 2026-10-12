#define ET_G3R_COMMITTED_HISTORY_PRIVATE 1
#define ET_G3T_MANUAL_DECODE_PRIVATE 1
#define ET_G3T_MANUAL_P2_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_P1_PREFILL_PRIVATE 1
#define ET_G3T_MANUAL_LOGITS_PRIVATE 1
#define ET_G3T_OWNED_TOKEN_INPUT_PRIVATE 1
#define ET_G3T_P2_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_ZERO_BUDGET_PRIVATE 1
#define ET_G3T_OUTPUT_TEXT_PRIVATE 1
#define ET_G3T_FINAL_PUBLICATION_PRIVATE 1
#define ET_G3T_INPUT_FROM_T1_PRIVATE 1
#define ET_G3T_PREFILL_SAMPLE_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#include "../../src/eshkol_transformer/g3t_transport.c"
#include <assert.h>
#include <stdio.h>

static int fail_commit;
int32_t et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  if (fail_commit) {
    error->category = 7;
    return 7;
  }
  *transaction = NULL;
  return 0;
}
int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  (void)error;
  *transaction = NULL;
  return 0;
}
int32_t et_a2_kv_cache_destroy_v1(
    et_a2_kv_cache **cache, et_kernel_error *error) {
  (void)error;
  *cache = NULL;
  return 0;
}
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  (void)pins; (void)error;
  return 0;
}
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins) {
  pins->held_mask = 0;
}
int32_t et_f32_parameter_validate_identity_v1(
    const et_f32_parameter *parameter, const void *identity,
    et_f32_tensor_error *error) {
  (void)parameter; (void)identity; (void)error;
  return 0;
}
#define UNUSED(...) do { (void)sizeof(#__VA_ARGS__); } while (0)
int32_t et_i64_tensor_borrow_begin_v1(et_i64_tensor *t,
    et_i64_tensor_borrow **b, et_i64_tensor_error *e) {
  UNUSED(t, b);
  e->category = 7;
  return 7;
}
int32_t et_i64_tensor_borrow_view_v1(const et_i64_tensor_borrow *b,
    const et_kernel_tensor_view_v1 **v, et_i64_tensor_error *e) {
  UNUSED(b, v, e); return 1;
}
int32_t et_i64_tensor_borrow_end_v1(et_i64_tensor_borrow **b,
    et_i64_tensor_error *e) { UNUSED(b, e); return 1; }
int32_t et_i64_tensor_destroy_v1(et_i64_tensor **t,
    et_i64_tensor_error *e) { UNUSED(e); *t = NULL; return 0; }
int32_t et_m3_private_i64_unborrowed_v1(const et_i64_tensor *t,
    et_i64_tensor_error *e) { UNUSED(t, e); return 0; }
int32_t et_f32_tensor_scoped_begin_internal(et_f32_tensor *t,
    et_f32_scoped_guard_internal *g, et_f32_tensor_error *e) {
  UNUSED(t, g, e); return 1;
}
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *g) {
  UNUSED(g); return 1;
}
int32_t et_f32_tensor_destroy_v1(et_f32_tensor **t,
    et_f32_tensor_error *e) { UNUSED(t, e); return 1; }
int32_t et_a2_kv_cache_transaction_view_begin_v1(
    et_a2_kv_cache_transaction *t, uint64_t layer,
    et_a2_kv_cache_transaction_view **v, et_kernel_error *e) {
  UNUSED(t, layer, v, e); return 1;
}
int32_t et_a2_kv_cache_transaction_view_tensors_v1(
    const et_a2_kv_cache_transaction_view *v,
    const et_kernel_tensor_view_v1 **k,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *e) {
  UNUSED(v, k, values, lengths, mask, e); return 1;
}
int32_t et_a2_kv_cache_transaction_view_end_v1(
    et_a2_kv_cache_transaction_view **v, et_kernel_error *e) {
  UNUSED(v, e); return 1;
}
int32_t et_a2_kv_cache_read_borrow_begin_v1(et_a2_kv_cache *c,
    et_a2_kv_cache_read_borrow **b, et_kernel_error *e) {
  UNUSED(c, b, e); return 1;
}
int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *b, uint64_t layer,
    const et_kernel_tensor_view_v1 **k,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *e) {
  UNUSED(b, layer, k, values, lengths, mask, e); return 1;
}
int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **b, et_kernel_error *e) {
  UNUSED(b, e); return 1;
}

int main(void) {
  g3t_context c = {0};
  initializer initial = {0}, next = {0};
  owner model = {0};
  unsigned char parameters[14], handles[14];
  initial.r.kind = next.r.kind = INITIALIZER;
  initial.r.state = next.r.state = 1;
  model.r.kind = OWNER;
  model.r.state = 1;
  model.initialized = 14;
  model.original = &initial;
  model.successor = &next;
  for (int i = 0; i < 14; ++i) {
    model.p[i] = (et_f32_parameter *)&parameters[i];
    model.handles[i] = &handles[i];
  }
  model.r.next = &initial.r;
  initial.r.next = &next.r;
  registry = &model.r;
  c.h.kind = G3T_GENERATOR;
  c.h.state = G3T_LIVE;
  g3t_registry = &c.h;
  c.model = &model;
  c.cache = (et_a2_kv_cache *)&parameters[0];
  c.policy[0] = 0;
  c.policy[1] = c.policy[3] = INT64_C(0x3f800000);
  c.policy[2] = 256;
  c.policy[4] = 1;
  c.policy[5] = 65;
  c.rng[0] = 1;
  c.rng[1] = 1729;
  c.rng[2] = -1;
  c.rng[3] = 7;
  et_g3r_committed_history_snapshot_v1 result;
  memset(&result, 0x5a, sizeof(result));
  et_g3r_committed_history_snapshot_v1 sentinel = result;
  assert(et_g3t_private_committed_history_snapshot_v1((void *)1, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));
  assert(!et_g3t_private_committed_history_snapshot_v1(&c, &result));
  assert(result.length == 0 && result.rng[2] == -1 && result.rng[3] == 7);
  sentinel = result;
  g3t_record wrong_kind = {.kind = G3T_INPUT, .state = G3T_LIVE,
                            .next = g3t_registry};
  g3t_registry = &wrong_kind;
  assert(et_g3t_private_committed_history_snapshot_v1(&wrong_kind, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));
  c.h.busy = 1;
  assert(et_g3t_private_committed_history_snapshot_v1(&c, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));
  c.h.busy = 0;
  c.h.state = G3T_DEAD;
  assert(et_g3t_private_committed_history_snapshot_v1(&c, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));
  c.h.state = G3T_LIVE;
  assert(et_g3t_private_committed_history_snapshot_v1(
      &c, (et_g3r_committed_history_snapshot_v1 *)&c));
  assert(!c.history_length && c.h.state == G3T_LIVE);

  c.h.busy = 1;
  c.call_kind = 2;
  model.active = &c;
  c.pins.held_mask = UINT16_C(0x3fff);
  c.frame.active = 1;
  c.frame.kind = 1;
  c.frame.prompt_length = 1;
  c.frame.token = 65;
  c.frame.next_ordinal = 21;
  c.frame.prepared = 1;
  c.frame.transaction = (et_a2_kv_cache_transaction *)&parameters[1];
  c.frame.candidate_cache = (et_a2_kv_cache *)&parameters[2];
  g3t_output pending = {0};
  pending.h.kind = G3T_OUTPUT;
  pending.h.state = G3T_PENDING;
  pending.h.busy = 1;
  pending.h.next = g3t_registry;
  g3t_registry = &pending.h;
  pending.parent_ctx = &c;
  pending.ids = (et_i64_tensor *)&parameters[3];
  c.pending_output = &pending;
  fail_commit = 1;
  assert(et_g3t_private_frame_commit_v1(&c));
  assert(!c.history_length && c.frame.active && c.cache);
  fail_commit = 0;
  assert(!et_g3t_private_frame_commit_v1(&c));
  assert(c.history_length == 1 && c.history_ids[0] == 65);
  assert(c.history_prompt_length == 1 && !c.history_generated_eos_terminal);
  assert(c.history_pending_g1);
  assert(!et_g3t_private_call_abort_v1(&c));
  assert(pending.h.state == G3T_DEAD && !pending.ids);
  assert(c.history_length == 1 && c.history_ids[0] == 65 &&
         c.history_pending_g1);
  c.pending_output = &pending;
  assert(et_g3t_private_committed_history_snapshot_v1(&c, &result));
  c.pending_output = NULL;
  sentinel = result;
  assert(et_g3t_private_committed_history_snapshot_v1(&c, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));
  c.call_kind = 1;
  c.frame.kind = 2;
  c.frame.token = 66;
  assert(g3t_history_append_preflight(&c));
  c.call_kind = 0;
  memset(&c.frame, 0, sizeof(c.frame));
  g3t_output detached = {0};
  detached.h.kind = G3T_OUTPUT;
  detached.h.state = G3T_LIVE;
  detached.h.next = g3t_registry;
  g3t_registry = &detached.h;
  detached.ids = (et_i64_tensor *)&parameters[4];
  assert(!et_g3t_private_output_release_v1(&detached));
  assert(et_g3t_private_committed_history_snapshot_v1(&c, &result));
  assert(!memcmp(&result, &sentinel, sizeof(result)));

  /* An I1 allocation cut in G0 preflight cannot publish a prompt. */
  g3t_context zero = {0};
  zero.h.kind = G3T_GENERATOR;
  zero.h.state = G3T_LIVE;
  zero.h.busy = 1;
  zero.h.next = g3t_registry;
  g3t_registry = &zero.h;
  zero.model = &model;
  model.active = &zero;
  zero.cache = (et_a2_kv_cache *)&parameters[5];
  memcpy(zero.policy, c.policy, sizeof(c.policy));
  zero.policy[4] = 0;
  memcpy(zero.rng, c.rng, sizeof(c.rng));
  zero.call_kind = 2;
  zero.pins.held_mask = UINT16_C(0x3fff);
  unsigned char binding[4736] = {0};
  for (int i = 0; i < 14; ++i) {
    zero.pins.views[i].data = binding;
    zero.pins.views[i].byte_length = i == 13 ? sizeof(binding) : 0;
    zero.pins.identities[i] = &handles[i];
    zero.frame.binding_identities[i] = &handles[i];
  }
  zero.frame.active = zero.frame.kind = 1;
  zero.frame.token = 65;
  zero.frame.prompt_length = 1;
  zero.frame.next_ordinal = 21;
  zero.frame.prepared = 1;
  zero.frame.transaction = (et_a2_kv_cache_transaction *)&parameters[6];
  zero.frame.candidate_cache = (et_a2_kv_cache *)&parameters[7];
  g3t_output empty = {0};
  empty.h.kind = G3T_OUTPUT;
  empty.h.state = G3T_PENDING;
  empty.h.busy = 1;
  empty.h.next = g3t_registry;
  g3t_registry = &empty.h;
  empty.parent_ctx = &zero;
  empty.P = empty.cache_length = 1;
  empty.numeric_ready = empty.ids_copied = empty.text_ready = 1;
  empty.ids = (et_i64_tensor *)&parameters[8];
  memcpy(empty.rng, zero.rng, sizeof(empty.rng));
  zero.pending_output = &empty;
  assert(et_g3t_private_call_prepare_end_v1(&zero) == 7);
  assert(!zero.history_length && zero.frame.active && !zero.frame.end_prepared);
  assert(et_g3t_private_frame_commit_v1(&zero));
  assert(!zero.history_length);
  assert(!et_g3t_private_call_abort_v1(&zero));
  assert(!zero.history_length && empty.h.state == G3T_DEAD);

  assert(!et_g3t_private_generator_close_v1(&c));
  assert(!c.history_length && !c.history_ids[0] && !c.history_ids[1]);
  assert(!c.history_pending_g1);
  assert(c.h.state == G3T_DEAD);

  /* A completed G1 append clears pending provenance. */
  g3t_context complete = {0};
  complete.h.kind = G3T_GENERATOR;
  complete.h.state = G3T_LIVE;
  complete.h.next = g3t_registry;
  g3t_registry = &complete.h;
  complete.model = &model;
  complete.cache = (et_a2_kv_cache *)&parameters[0];
  memcpy(complete.policy, result.policy, sizeof(complete.policy));
  memcpy(complete.rng, result.rng, sizeof(complete.rng));
  complete.call_kind = 2;
  complete.frame.kind = 1;
  complete.frame.prompt_length = 1;
  complete.frame.token = 66;
  g3t_history_commit_prompt(&complete);
  complete.prefill_committed = 1;
  assert(complete.history_pending_g1);
  complete.frame.kind = 2;
  complete.frame.token = 65;
  assert(!g3t_history_append_preflight(&complete));
  g3t_history_commit_append(&complete, 1);
  complete.call_kind = 0;
  memset(&complete.frame, 0, sizeof(complete.frame));
  assert(!complete.history_pending_g1);
  assert(!et_g3t_private_committed_history_snapshot_v1(&complete, &result));
  assert(result.length == 2 && result.ids[0] == 66 && result.ids[1] == 65);
  assert(result.generated_count == 1 && result.generated_eos_terminal == 1);
  g3t_output completed_output = {0};
  completed_output.h.kind = G3T_OUTPUT;
  completed_output.h.state = G3T_LIVE;
  completed_output.h.next = g3t_registry;
  g3t_registry = &completed_output.h;
  completed_output.ids = (et_i64_tensor *)&parameters[4];
  assert(!et_g3t_private_output_release_v1(&completed_output));
  assert(!et_g3t_private_committed_history_snapshot_v1(&complete, &result));
  assert(result.length == 2 && result.generated_count == 1);
  assert(!et_g3t_private_generator_close_v1(&complete));
  puts("G3-R committed history native transport PASS");
  return 0;
}
