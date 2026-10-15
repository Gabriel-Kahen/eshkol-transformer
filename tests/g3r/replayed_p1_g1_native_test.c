/* Host-only state-machine witness. Real provider sampling stays for the
 * later pinned compiled normal/repeat/sanitizer gate. */
#define ET_G3R_REPLAYED_P1_G1_PRIVATE 1
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

static unsigned char binding[4736], keys[32], values[32];
static uint8_t keep[2] = {1, 0};
static int64_t length = 1;
static uint64_t kv_shape[4] = {1, 2, 2, 2};
static uint64_t length_shape[1] = {1}, keep_shape[2] = {1, 2};
static et_kernel_tensor_view_v1 kv[2], len_view, keep_view;
static int fail_borrow, fail_pins;
static unsigned char borrow_tag;

int32_t et_f32_parameter_validate_identity_v1(
    const et_f32_parameter *p, const void *identity, et_f32_tensor_error *e) {
  (void)p; (void)identity; (void)e; return 0;
}
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const parameters[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  (void)parameters;
  if (fail_pins) { error->category = 8; return 8; }
  pins->self = pins;
  pins->held_mask = UINT16_C(0x3fff);
  for (size_t i = 0; i < 14; ++i) {
    pins->identities[i] = identities[i];
    pins->views[i].data = binding;
    pins->views[i].byte_length = i ? 0 : sizeof(binding);
  }
  return 0;
}
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *error) {
  if (pins->self != pins || pins->held_mask != UINT16_C(0x3fff)) {
    error->category = 8; return 8;
  }
  return 0;
}
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins) {
  memset(pins, 0, sizeof(*pins));
}
int32_t et_a2_kv_cache_read_borrow_begin_v1(
    et_a2_kv_cache *cache, et_a2_kv_cache_read_borrow **borrow,
    et_kernel_error *error) {
  (void)cache;
  if (fail_borrow) { error->category = 7; return 7; }
  *borrow = (et_a2_kv_cache_read_borrow *)&borrow_tag;
  return 0;
}
int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *borrow, uint64_t layer,
    const et_kernel_tensor_view_v1 **k, const et_kernel_tensor_view_v1 **v,
    const et_kernel_tensor_view_v1 **len,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *error) {
  (void)borrow; (void)layer; (void)error;
  *k = &kv[0]; *v = &kv[1]; *len = &len_view; *mask = &keep_view;
  return 0;
}
int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **borrow, et_kernel_error *error) {
  (void)error; *borrow = NULL; return 0;
}
int32_t et_i64_tensor_create_v1(
    uint64_t rank, const uint64_t *shape, et_i64_tensor **tensor,
    et_i64_tensor_error *error) {
  (void)rank; (void)shape; (void)error;
  *tensor = (et_i64_tensor *)&borrow_tag; return 0;
}
int32_t et_i64_tensor_destroy_v1(et_i64_tensor **tensor,
    et_i64_tensor_error *error) { (void)error; *tensor = NULL; return 0; }
int32_t et_m3_private_i64_unborrowed_v1(const et_i64_tensor *tensor,
    et_i64_tensor_error *error) { (void)tensor; (void)error; return 0; }
int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **t, et_kernel_error *e) {
  (void)e; *t = NULL; return 0;
}
int32_t et_a2_kv_cache_destroy_v1(et_a2_kv_cache **cache,
    et_kernel_error *e) { (void)e; *cache = NULL; return 0; }
int32_t et_f32_tensor_scoped_begin_internal(et_f32_tensor *t,
    et_f32_scoped_guard_internal *g, et_f32_tensor_error *e) {
  (void)t; (void)g; (void)e; return 1;
}
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *g) {
  (void)g; return 0;
}
int32_t et_f32_tensor_destroy_v1(et_f32_tensor **t,
    et_f32_tensor_error *e) { (void)e; *t = NULL; return 0; }

int32_t et_i64_tensor_borrow_begin_v1(et_i64_tensor *t,
    et_i64_tensor_borrow **b, et_i64_tensor_error *e) {
  (void)t; (void)b; e->category = 7; return 7;
}
int32_t et_i64_tensor_borrow_view_v1(const et_i64_tensor_borrow *b,
    const et_kernel_tensor_view_v1 **v, et_i64_tensor_error *e) {
  (void)b; (void)v; e->category = 7; return 7;
}
int32_t et_i64_tensor_borrow_end_v1(et_i64_tensor_borrow **b,
    et_i64_tensor_error *e) { (void)b; (void)e; return 0; }
int32_t et_a2_kv_cache_create_v1(uint64_t layers, uint64_t batch,
    uint64_t kv_heads, uint64_t capacity, uint64_t head_dimension,
    et_a2_kv_cache **cache, et_kernel_error *error) {
  (void)layers; (void)batch; (void)kv_heads; (void)capacity;
  (void)head_dimension; (void)cache; error->category = 7; return 7;
}

static void init_view(et_kernel_tensor_view_v1 *v, void *data,
                      const char *dtype, uint64_t rank,
                      const uint64_t *shape, uint64_t bytes) {
  *v = (et_kernel_tensor_view_v1){0};
  v->struct_size = sizeof(*v);
  v->data = data; v->dtype = dtype; v->device = "cpu";
  v->layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  v->rank = rank; v->shape = shape; v->byte_length = bytes;
}
int main(void) {
  init_view(&kv[0], keys, "f32", 4, kv_shape, sizeof(keys));
  init_view(&kv[1], values, "f32", 4, kv_shape, sizeof(values));
  init_view(&len_view, &length, "i64", 1, length_shape, sizeof(length));
  init_view(&keep_view, keep, "bool", 2, keep_shape, sizeof(keep));
  initializer first = {0}, last = {0};
  owner model = {0};
  unsigned char parameter[14], handle[14];
  first.r.kind = last.r.kind = INITIALIZER;
  first.r.state = last.r.state = 1;
  model.r.kind = OWNER; model.r.state = 1; model.initialized = 14;
  model.original = &first; model.successor = &last;
  model.r.next = &first.r; first.r.next = &last.r; registry = &model.r;
  for (size_t i = 0; i < 14; ++i) {
    model.p[i] = (et_f32_parameter *)&parameter[i];
    model.handles[i] = &handle[i];
  }
  g3t_context c = {0};
  c.h.kind = G3T_GENERATOR; c.h.state = G3T_LIVE;
  g3t_registry = &c.h; c.model = &model;
  c.cache = (et_a2_kv_cache *)&borrow_tag;
  c.policy[0] = 0; c.policy[1] = c.policy[3] = INT64_C(0x3f800000);
  c.policy[2] = 256; c.policy[4] = 1; c.policy[5] = 65;
  c.rng[0] = 1; c.rng[1] = 1729;
  c.history_ids[0] = 65; c.history_length = c.history_prompt_length = 1;
  c.prefill_committed = c.binding_ready = 1;
  for (size_t i = 0; i < 14; ++i)
    c.binding_identities[i] = model.handles[i];
  g3t_context original = c;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1((void *)1));
  assert(!et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  assert(!memcmp(&c, &original, sizeof(c)));
  assert(et_g3t_private_call_acquire_v1(&c, 2));
  assert(!c.h.busy && !model.active);
  length = 2; keep[1] = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  length = 1; keep[1] = 0;
  fail_borrow = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  fail_borrow = 0;
  fail_pins = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  fail_pins = 0;
  c.frame.next_ordinal = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  c.frame.next_ordinal = 0;
  binding[0] = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  binding[0] = 0;
  c.history_length = c.history_prompt_length = 2;
  c.history_ids[1] = 66; length = 2; keep[1] = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  c.history_length = c.history_prompt_length = 1;
  c.history_ids[1] = 0; length = 1; keep[1] = 0;
  c.history_length = 2; c.history_generated_count = 1;
  c.history_ids[1] = 65; c.history_generated_eos_terminal = 1;
  length = 2; keep[1] = 1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  c.history_length = 1; c.history_generated_count = 0;
  c.history_ids[1] = 0; c.history_generated_eos_terminal = 0;
  length = 1; keep[1] = 0;
  c.policy[0] = 1; c.rng[2] = c.rng[3] = -1;
  assert(et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  c.policy[0] = 0; c.rng[2] = c.rng[3] = 0;
  assert(!et_g3t_private_replayed_p1_g1_acquire_v1(&c));
  assert(c.h.busy && model.active == &c && c.call_kind == 2 &&
         c.replayed_p1_g1_call && c.pins.self == &c.pins);
  void *output = et_g3t_private_output_reserve_v1(&c, 1);
  assert(output && c.pending_output == output);
  c.frame.kind = 2; c.frame.token = 66;
  assert(!g3t_history_append_preflight(&c));
  c.replayed_p1_g1_call = 0;
  assert(g3t_history_append_preflight(&c));
  c.replayed_p1_g1_call = 1;
  memset(&c.frame, 0, sizeof(c.frame));
  assert(et_g3t_private_frame_begin_v1(&c, NULL, 1));
  assert(!et_g3t_private_call_abort_v1(&c));
  assert(!c.h.busy && !model.active && !c.replayed_p1_g1_call &&
         !c.pending_output && !c.sampled && c.cache == original.cache);
  assert(((g3t_output *)output)->h.state == G3T_DEAD);
  assert(!et_g3t_private_replayed_p1_g1_acquire_v1(&c));
  assert(!et_g3t_private_call_abort_v1(&c));
  assert(!memcmp(c.policy, original.policy, sizeof(c.policy)) &&
         !memcmp(c.rng, original.rng, sizeof(c.rng)) &&
         !memcmp(c.history_ids, original.history_ids, sizeof(c.history_ids)) &&
         c.history_length == original.history_length &&
         c.history_prompt_length == original.history_prompt_length &&
         !memcmp(c.last_logits, original.last_logits, sizeof(c.last_logits)));
  assert(!et_g3t_private_replayed_p1_g1_preflight_v1(&c));
  puts("G3-R replayed P1/G1 native state PASS");
}
