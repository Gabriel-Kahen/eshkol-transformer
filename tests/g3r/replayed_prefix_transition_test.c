#define ET_G3R_REPLAYED_PREFIX_PRIVATE 1
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

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr, "G3-R replay FAIL line %d: %s\n", __LINE__, #x); \
  exit(1); } } while (0)

typedef struct { int64_t token; } fake_i64;
static int fake_borrow, fake_cache_borrow, fake_tx_view, fake_runtime;
static et_i64_tensor *borrowed_tensor;
static uint32_t key_bits[8], value_bits[8];
static int64_t cache_length = 1, tx_length = 2;
static uint8_t cache_mask[2] = {1, 0}, tx_mask[2] = {1, 1};
static et_kernel_tensor_view_v1 key_view, value_view, length_view, mask_view;
static int fail_sample, fail_cache_borrow, dispatches;
static unsigned char parameter_bytes[4736], identities[14];

int32_t et_f32_parameter_validate_identity_v1(
    const et_f32_parameter *p, const void *identity, et_f32_tensor_error *e) {
  (void)p; (void)identity; (void)e; return 0;
}
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14],
    const void *const handles[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *e) {
  (void)parameters; (void)e;
  memset(pins, 0, sizeof(*pins));
  pins->held_mask = UINT16_C(0x3fff);
  for (size_t i = 0; i < 14; ++i) {
    pins->identities[i] = handles[i];
    pins->views[i].data = parameter_bytes;
    pins->views[i].byte_length = i ? 0 : sizeof(parameter_bytes);
  }
  return 0;
}
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *e) {
  (void)e; return pins->held_mask == UINT16_C(0x3fff) ? 0 : 2;
}
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins) {
  pins->held_mask = 0;
}
int32_t et_a2_kv_cache_read_borrow_begin_v1(
    et_a2_kv_cache *cache, et_a2_kv_cache_read_borrow **out,
    et_kernel_error *e) {
  (void)cache;
  if (fail_cache_borrow) { e->category = 4; e->code = 2; return 4; }
  *out = (et_a2_kv_cache_read_borrow *)&fake_cache_borrow; return 0;
}
int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *borrow, uint64_t layer,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *e) {
  (void)borrow; (void)e; CHECK(layer == 0);
  *keys = &key_view; *values = &value_view;
  *lengths = &length_view; *mask = &mask_view;
  return 0;
}
int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **borrow, et_kernel_error *e) {
  (void)e; *borrow = NULL; return 0;
}
int32_t et_i64_tensor_create_v1(uint64_t rank, const uint64_t *shape,
    et_i64_tensor **out, et_i64_tensor_error *e) {
  (void)e; CHECK(rank == 1 && shape[0] == 1);
  *out = (et_i64_tensor *)calloc(1, sizeof(fake_i64));
  return *out ? 0 : 7;
}
int32_t et_f32_tensor_scoped_begin_internal(et_f32_tensor *tensor,
    et_f32_scoped_guard_internal *guard, et_f32_tensor_error *e) {
  (void)tensor; (void)guard; (void)e; return 2;
}
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *guard) {
  (void)guard; return 0;
}
int32_t et_f32_tensor_destroy_v1(et_f32_tensor **tensor,
    et_f32_tensor_error *e) {
  (void)e; *tensor = NULL; return 0;
}
int32_t et_a2_kv_cache_destroy_v1(et_a2_kv_cache **cache,
    et_kernel_error *e) {
  (void)e; *cache = NULL; return 0;
}
int32_t et_m3_private_i64_unborrowed_v1(
    const et_i64_tensor *tensor, et_i64_tensor_error *e) {
  (void)tensor; (void)e; return 0;
}
int32_t et_i64_tensor_destroy_v1(
    et_i64_tensor **tensor, et_i64_tensor_error *e) {
  (void)e; free(*tensor); *tensor = NULL; return 0;
}
int32_t et_i64_tensor_borrow_begin_v1(
    et_i64_tensor *tensor, et_i64_tensor_borrow **out,
    et_i64_tensor_error *e) {
  (void)e; borrowed_tensor = tensor;
  *out = (et_i64_tensor_borrow *)&fake_borrow; return 0;
}
int32_t et_i64_tensor_borrow_view_v1(
    const et_i64_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 **out, et_i64_tensor_error *e) {
  (void)borrow; (void)e;
  static uint64_t shape[1] = {1};
  static et_kernel_tensor_view_v1 view;
  view.rank = 1; view.shape = shape; view.byte_length = sizeof(int64_t);
  view.data = &((fake_i64 *)borrowed_tensor)->token;
  *out = &view; return 0;
}
int32_t et_i64_tensor_borrow_end_v1(
    et_i64_tensor_borrow **borrow, et_i64_tensor_error *e) {
  (void)e; *borrow = NULL; return 0;
}
int32_t et_i64_tensor_copy_to_v1(
    const et_i64_tensor *tensor, int64_t *out, uint64_t n,
    et_i64_tensor_error *e) {
  (void)e; CHECK(n == 1); *out = ((const fake_i64 *)tensor)->token; return 0;
}
int32_t et_kernel_runtime_discover(
    et_kernel_provider_resolver_v1 resolver, void *user,
    et_kernel_runtime **out, et_kernel_error *e) {
  (void)resolver; (void)user; (void)e;
  *out = (et_kernel_runtime *)&fake_runtime; return 0;
}
int32_t et_kernel_runtime_dispatch(
    const et_kernel_runtime *runtime, const et_kernel_call_v1 *call,
    et_kernel_error *e) {
  (void)runtime; ++dispatches;
  CHECK(strcmp(call->capability, "g3s.categorical") == 0);
  const et_kernel_tensor_view_v1 *in = call->inputs;
  const et_kernel_tensor_view_v1 *out = call->outputs;
  CHECK(in[0].data != NULL);
  CHECK(((const float *)in[0].data)[77] == 10.0f);
  CHECK(((const int64_t *)in[4].data)[2] == 5);
  if (fail_sample) { e->category = 7; e->code = 1; return 7; }
  *(int64_t *)out[0].data = 77;
  memcpy(out[1].data, in[4].data, 4 * sizeof(int64_t));
  ((int64_t *)out[1].data)[2] = 6;
  return 0;
}
void et_kernel_runtime_destroy(et_kernel_runtime *runtime) { (void)runtime; }
const et_kernel_provider_v1 *et_g3s_kernel_provider_v1(void) {
  return (const et_kernel_provider_v1 *)1;
}
int32_t et_a2_kv_cache_transaction_view_begin_v1(
    et_a2_kv_cache_transaction *tx, uint64_t layer,
    et_a2_kv_cache_transaction_view **out, et_kernel_error *e) {
  (void)tx; (void)layer; (void)e;
  *out = (et_a2_kv_cache_transaction_view *)&fake_tx_view; return 0;
}
int32_t et_a2_kv_cache_transaction_view_tensors_v1(
    const et_a2_kv_cache_transaction_view *view,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *e) {
  (void)view; (void)e; *keys = &key_view; *values = &value_view;
  *lengths = &length_view; *mask = &mask_view; return 0;
}
int32_t et_a2_kv_cache_transaction_view_end_v1(
    et_a2_kv_cache_transaction_view **view, et_kernel_error *e) {
  (void)e; *view = NULL; return 0;
}
int32_t et_a2_kv_cache_transaction_commit_v1(
    et_a2_kv_cache_transaction **tx, et_kernel_error *e) {
  (void)e; *tx = NULL; return 0;
}
int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **tx, et_kernel_error *e) {
  (void)e; *tx = NULL; return 0;
}

static void views_init(void) {
  static uint64_t kv_shape[4] = {1, 2, 2, 2};
  static uint64_t len_shape[1] = {1}, mask_shape[2] = {1, 2};
  key_view.data = key_bits; key_view.rank = 4; key_view.shape = kv_shape;
  key_view.struct_size = sizeof(key_view); key_view.dtype = "f32";
  key_view.device = "cpu";
  key_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  key_view.byte_length = 32;
  value_view = key_view; value_view.data = value_bits;
  length_view.data = &cache_length; length_view.rank = 1;
  length_view.struct_size = sizeof(length_view); length_view.dtype = "i64";
  length_view.device = "cpu";
  length_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  length_view.shape = len_shape; length_view.byte_length = 8;
  mask_view.data = cache_mask; mask_view.rank = 2;
  mask_view.struct_size = sizeof(mask_view); mask_view.dtype = "bool";
  mask_view.device = "cpu";
  mask_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  mask_view.shape = mask_shape; mask_view.byte_length = 2;
}
static void prefix(g3t_context *c, owner *o, int64_t n) {
  memset(c, 0, sizeof(*c));
  c->h.kind = G3T_GENERATOR; c->h.state = G3T_LIVE;
  c->h.next = g3t_registry; g3t_registry = &c->h;
  c->model = o; c->cache = (et_a2_kv_cache *)1;
  c->policy[0] = 1; c->policy[1] = c->policy[3] = 0x3f800000;
  c->policy[2] = 256; c->policy[4] = 1; c->policy[5] = -1;
  c->rng[0] = 1; c->rng[1] = 1729; c->rng[2] = 5;
  c->frame.kind = 1; c->frame.prompt_length = n;
  c->frame.token = 65; c->frame.prompt_ids[0] = 65;
  c->frame.prompt_ids[1] = 66;
  g3t_history_commit_prompt(c); c->frame.kind = 0;
  c->prefill_committed = c->binding_ready = 1;
  c->last_logits[77] = 10.0f;
  for (size_t i = 0; i < 14; ++i) c->binding_identities[i] = o->handles[i];
  memcpy(c->binding_values, parameter_bytes, sizeof(parameter_bytes));
}
int main(void) {
  views_init();
  static owner a, b;
  static initializer original_a, successor_a, original_b, successor_b;
  original_a.r.kind = successor_a.r.kind = INITIALIZER;
  original_b.r.kind = successor_b.r.kind = INITIALIZER;
  original_a.r.state = successor_a.r.state = 1;
  original_b.r.state = successor_b.r.state = 1;
  a.r.kind = b.r.kind = OWNER; a.r.state = b.r.state = 1;
  a.initialized = b.initialized = 14;
  a.original = &original_a; a.successor = &successor_a;
  b.original = &original_b; b.successor = &successor_b;
  original_a.r.next = (record *)&successor_a;
  successor_a.r.next = (record *)&original_b;
  original_b.r.next = (record *)&successor_b;
  successor_b.r.next = (record *)&a;
  a.r.next = (record *)&b;
  registry = (record *)&original_a;
  for (size_t i = 0; i < 14; ++i) {
    a.p[i] = (et_f32_parameter *)&parameter_bytes[i];
    b.p[i] = (et_f32_parameter *)&parameter_bytes[i + 14];
    a.handles[i] = &identities[i]; b.handles[i] = &parameter_bytes[i + 28];
  }
  static g3t_context c;
  prefix(&c, &a, 1);
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &b) != 0);
  CHECK(!c.h.busy && !c.history_pending_g1 && !c.pins.held_mask);
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, NULL) != 0);
  CHECK(!c.h.busy && !c.history_pending_g1);
  c.history_length = 2; c.history_prompt_length = 2;
  c.history_ids[1] = 66; cache_length = 2; cache_mask[1] = 1;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  CHECK(dispatches == 0 && !c.h.busy);
  c.history_length = c.history_prompt_length = 1;
  c.history_ids[1] = 0; cache_length = 1; cache_mask[1] = 0;
  c.rng[2] = c.rng[3] = -1;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  CHECK(dispatches == 0 && !c.h.busy);
  c.rng[2] = 5; c.rng[3] = 0;
  cache_mask[1] = 1;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  CHECK(!c.h.busy && !c.history_pending_g1);
  cache_mask[1] = 0;
  fail_cache_borrow = 1;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  CHECK(!c.h.busy && !c.history_pending_g1 && !c.pins.held_mask);
  fail_cache_borrow = 0;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) == 0);
  CHECK(c.call_kind == 2 && c.replayed_prefix_call && c.history_pending_g1);
  CHECK(c.h.busy && a.active == &c && c.pins.held_mask == 0x3fff);
  g3t_output *out = et_g3t_private_output_reserve_v1(&c, 1);
  CHECK(out && c.pending_output == out && out->h.state == G3T_PENDING);
  g3t_output *aborted = out;
  fail_sample = 1;
  CHECK(et_g3t_private_sample_v1(&c) != 0);
  CHECK(c.rng[2] == 5 && c.history_length == 1 && c.cache);
  CHECK(et_g3t_private_call_abort_v1(&c) == 0);
  CHECK(!c.h.busy && !c.replayed_prefix_call && !c.history_pending_g1);
  CHECK(!c.pins.held_mask && !a.active && c.rng[2] == 5);
  CHECK(c.last_logits[77] == 10.0f && cache_length == 1);
  CHECK(et_g3t_private_committed_history_snapshot_v1(&c,
        &(et_g3r_committed_history_snapshot_v1){0}) == 0);
  fail_sample = 0;
  c.policy[5] = 77;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) == 0);
  out = et_g3t_private_output_reserve_v1(&c, 1);
  CHECK(out && et_g3t_private_sample_v1(&c) == 0);
  CHECK(c.sampled && c.sample_token == 77 && c.successor_rng[2] == 6);
  CHECK(c.rng[2] == 5 && c.history_length == 1);
  g3t_output *sampled_abort = out;
  CHECK(et_g3t_private_call_abort_v1(&c) == 0);
  CHECK(c.rng[2] == 5 && c.history_length == 1 && !c.history_pending_g1);
  CHECK(c.last_logits[77] == 10.0f && cache_length == 1);
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) == 0);
  out = et_g3t_private_output_reserve_v1(&c, 1);
  CHECK(out && et_g3t_private_sample_v1(&c) == 0);
  CHECK(c.sample_token == 77 && c.successor_rng[2] == 6);
  /* The frame/output fixture stands in for the reviewed 21-role route.
   * The sample, final preflight, A2 commit and history/RNG publication below
   * are the production native functions. */
  c.frame.active = 1; c.frame.kind = 2; c.frame.next_ordinal = 21;
  c.frame.prepared = c.frame.end_prepared = 1;
  c.frame.transaction = (et_a2_kv_cache_transaction *)1;
  c.frame.token = c.sample_token;
  out->length = 1; out->cache_length = 2;
  out->numeric_ready = out->ids_copied = out->text_ready = 1;
  out->staged_id[0] = (uint8_t)c.sample_token;
  ((fake_i64 *)out->ids)->token = c.sample_token;
  memcpy(out->rng, c.successor_rng, sizeof(out->rng));
  length_view.data = &tx_length; mask_view.data = tx_mask;
  CHECK(et_g3t_private_frame_commit_v1(&c) == 0);
  CHECK(c.rng[2] == 6 && c.history_length == 2);
  CHECK(c.history_generated_count == 1 && c.history_ids[1] == 77);
  CHECK(c.history_generated_eos_terminal == 1);
  CHECK(!c.history_pending_g1 && out->h.state == G3T_LIVE);
  CHECK(et_g3t_private_call_finish_v1(&c) == 0);
  CHECK(!c.h.busy && !c.replayed_prefix_call && !a.active);
  CHECK(et_g3t_private_committed_history_snapshot_v1(&c,
        &(et_g3r_committed_history_snapshot_v1){0}) == 0);
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  CHECK(et_g3t_private_output_release_v1(out) == 0);
  c.h.state = G3T_DEAD;
  CHECK(et_g3t_private_replayed_prefix_call_acquire_v1(&c, &a) != 0);
  free(out); free(sampled_abort); free(aborted);
  g3t_registry = NULL;
  printf("G3-R replayed-prefix native PASS: %zu checks\n", checks);
  return 0;
}
