#define ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE 1
#define ET_G3R_COMMITTED_SAVE_HOLD_PRIVATE 1
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
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#include "../../src/eshkol_transformer/g3t_transport.c"
#include <assert.h>
#include <stdio.h>

typedef struct bytes136 { int64_t length; unsigned char data[136]; } bytes136;
typedef struct bytes4872 { int64_t length; unsigned char data[4872]; } bytes4872;
static int64_t cache_length;
static int pin_active, cache_active, fail_pin, fail_cache, fail_layer;
static int malformed_lengths;
static int fake_overlap;
static int fake_i64_overlap;
static unsigned char model_bits[4736];
static int64_t lengths_data[1];
static uint64_t lengths_shape[1] = {1};
static et_kernel_tensor_view_v1 lengths_view;

int32_t et_f32_parameter_validate_identity_v1(
    const et_f32_parameter *p, const void *identity, et_f32_tensor_error *e) {
  (void)p; (void)identity; (void)e; return 0;
}
int32_t et_g3t_model_pins_begin_internal(
    et_f32_parameter *const p[14], const void *const identities[14],
    et_g3t_model_pins_internal *pins, et_f32_tensor_error *e) {
  (void)p;
  if (fail_pin) { e->category = ET_F32_TENSOR_ERROR_INTERNAL; return e->category; }
  assert(!pin_active);
  pin_active = 1;
  pins->self = pins;
  pins->held_mask = UINT16_C(0x3fff);
  for (size_t i = 0; i < 14; ++i) {
    pins->parameters[i] = p[i];
    pins->identities[i] = identities[i];
    pins->views[i].data = model_bits;
    pins->views[i].byte_length = i == 13 ? sizeof(model_bits) : 0;
  }
  return 0;
}
int32_t et_g3t_model_pins_check_internal(
    const et_g3t_model_pins_internal *pins, et_f32_tensor_error *e) {
  (void)e; return !pin_active || pins->held_mask != UINT16_C(0x3fff);
}
void et_g3t_model_pins_end_internal(et_g3t_model_pins_internal *pins) {
  assert(pin_active && pins->held_mask == UINT16_C(0x3fff));
  pin_active = 0; pins->held_mask = 0;
}
int32_t et_a2_kv_cache_read_borrow_begin_v1(
    et_a2_kv_cache *cache, et_a2_kv_cache_read_borrow **borrow,
    et_kernel_error *e) {
  (void)cache;
  if (fail_cache || cache_active) {
    e->category = ET_KERNEL_ERROR_INVALID_ARGUMENT; return e->category;
  }
  cache_active = 1; *borrow = (et_a2_kv_cache_read_borrow *)&cache_active;
  return 0;
}
int32_t et_a2_kv_cache_read_borrow_layer_v1(
    et_a2_kv_cache_read_borrow *borrow, uint64_t layer,
    const et_kernel_tensor_view_v1 **keys,
    const et_kernel_tensor_view_v1 **values,
    const et_kernel_tensor_view_v1 **lengths,
    const et_kernel_tensor_view_v1 **mask, et_kernel_error *e) {
  (void)borrow; (void)layer; (void)keys; (void)values; (void)mask;
  if (fail_layer) { e->category = ET_KERNEL_ERROR_INVALID_ARGUMENT; return e->category; }
  lengths_data[0] = cache_length;
  lengths_view = (et_kernel_tensor_view_v1){0};
  lengths_view.struct_size = sizeof(lengths_view);
  lengths_view.data = lengths_data;
  lengths_view.rank = 1;
  lengths_view.shape = lengths_shape;
  lengths_view.byte_length = sizeof(lengths_data);
  lengths_view.dtype = malformed_lengths ? "f32" : "i64";
  lengths_view.device = "cpu";
  lengths_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  *lengths = &lengths_view;
  return 0;
}
int32_t et_a2_kv_cache_read_borrow_end_v1(
    et_a2_kv_cache_read_borrow **borrow, et_kernel_error *e) {
  (void)e; assert(cache_active && *borrow);
  cache_active = 0; *borrow = NULL; return 0;
}
int32_t et_a2_kv_cache_private_storage_overlap_v1(const void *p, size_t n) {
  (void)p; (void)n; return fake_overlap;
}
int32_t et_f32_tensor_private_storage_overlap_v1(const void *p, size_t n) {
  (void)p; (void)n; return fake_overlap;
}
int32_t et_i64_tensor_private_storage_overlap_v1(const void *p, size_t n) {
  (void)p; (void)n; return fake_i64_overlap;
}
#define UNUSED(...) do { (void)sizeof(#__VA_ARGS__); } while (0)
int32_t et_a2_kv_cache_destroy_v1(
    et_a2_kv_cache **cache, et_kernel_error *error) {
  UNUSED(error); *cache = NULL; return 0;
}
int32_t et_a2_kv_cache_transaction_abort_v1(
    et_a2_kv_cache_transaction **transaction, et_kernel_error *error) {
  UNUSED(error); *transaction = NULL; return 0;
}
int32_t et_m3_private_i64_unborrowed_v1(
    const et_i64_tensor *tensor, et_i64_tensor_error *error) {
  UNUSED(tensor, error); return 0;
}
int32_t et_i64_tensor_destroy_v1(
    et_i64_tensor **tensor, et_i64_tensor_error *error) {
  UNUSED(error); *tensor = NULL; return 0;
}
int32_t et_f32_tensor_destroy_v1(
    et_f32_tensor **tensor, et_f32_tensor_error *error) {
  UNUSED(error); *tensor = NULL; return 0;
}
int32_t et_f32_tensor_scoped_begin_internal(
    et_f32_tensor *tensor, et_f32_scoped_guard_internal *guard,
    et_f32_tensor_error *error) {
  UNUSED(tensor, guard);
  error->category = ET_F32_TENSOR_ERROR_INVALID_STATE;
  error->code = ET_F32_TENSOR_CODE_INVALID_HANDLE;
  return error->category;
}
int32_t et_f32_tensor_scoped_end_internal(et_f32_scoped_guard_internal *guard) {
  UNUSED(guard); return 0;
}

static g3t_context c;
static owner model, foreign;
static initializer initial, next;
static unsigned char parameters[14], handles[14];
static bytes136 destination, staging;
static void unchanged(void) {
  unsigned char expected[136]; memset(expected, 0x5a, sizeof(expected));
  assert(!memcmp(destination.data, expected, sizeof(expected)));
  assert(!memcmp(staging.data, expected, sizeof(expected)));
  assert(!pin_active && !cache_active);
}
static void reject(void) {
  memset(destination.data, 0x5a, 136);
  memset(staging.data, 0x5a, 136);
  assert(et_g3t_private_committed_save_copy_v1(
      &c, &model, &destination, &staging));
  unchanged();
}
static void positive(int64_t first, int64_t second, int64_t length,
                     int64_t prompt, int64_t generated, int64_t manual,
                     int64_t terminal) {
  c.history_ids[0] = first; c.history_ids[1] = second;
  c.history_length = cache_length = length;
  c.history_prompt_length = prompt;
  c.history_generated_count = generated;
  c.history_manual_count = manual;
  c.history_generated_eos_terminal = terminal;
  memset(destination.data, 0x5a, 136);
  memset(staging.data, 0x5a, 136);
  assert(!et_g3t_private_committed_save_copy_v1(
      &c, &model, &destination, &staging));
  unsigned char expected[136]; memset(expected, 0x5a, 136);
  assert(!memcmp(destination.data, expected, 136));
  et_g3r_committed_history_snapshot_v1 result;
  memcpy(&result, staging.data, sizeof(result));
  assert(result.ids[0] == first && result.ids[1] == second);
  assert(result.length == length && result.prompt_length == prompt);
  assert(result.generated_count == generated && result.manual_count == manual);
  assert(result.generated_eos_terminal == terminal);
  for (size_t i = 0; i < 4; ++i) assert(result.rng[i] == c.rng[i]);
  assert(!pin_active && !cache_active);
}
int main(void) {
  initial.r.kind = next.r.kind = INITIALIZER;
  initial.r.state = next.r.state = 1;
  model.r.kind = foreign.r.kind = OWNER;
  model.r.state = foreign.r.state = 1;
  model.initialized = foreign.initialized = 14;
  model.original = foreign.original = &initial;
  model.successor = foreign.successor = &next;
  for (int i = 0; i < 14; ++i) {
    model.p[i] = foreign.p[i] = (et_f32_parameter *)&parameters[i];
    model.handles[i] = foreign.handles[i] = &handles[i];
    c.binding_identities[i] = &handles[i];
  }
  model.r.next = &foreign.r; foreign.r.next = &initial.r;
  initial.r.next = &next.r; registry = &model.r;
  c.h.kind = G3T_GENERATOR; c.h.state = G3T_LIVE; g3t_registry = &c.h;
  c.model = &model; c.cache = (et_a2_kv_cache *)&parameters[0];
  c.policy[0] = 0; c.policy[1] = c.policy[3] = INT64_C(0x3f800000);
  c.policy[2] = 256; c.policy[4] = 1; c.policy[5] = 65;
  c.rng[0] = 1; c.rng[1] = 1729; c.rng[2] = -1; c.rng[3] = 7;
  c.prefill_committed = c.binding_ready = 1;
  destination.length = staging.length = 136;
  positive(66, 0, 1, 1, 0, 0, 0);
  positive(66, 67, 2, 2, 0, 0, 0);
  positive(66, 65, 2, 1, 0, 1, 0);
  positive(66, 65, 2, 1, 1, 0, 1);
  c.policy[5] = -1;
  positive(66, 65, 2, 1, 1, 0, 0);
  c.policy[5] = 65;
  c.policy[0] = 1;
  c.rng[2] = -1; c.rng[3] = 0;
  positive(66, 65, 2, 1, 1, 0, 1);
  c.rng[3] = -1;
  positive(66, 65, 2, 1, 1, 0, 1);
  et_g3r_committed_save_capture_v1 held;
  memset(&held, 0x5a, sizeof(held));
  int64_t hold = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(hold && pin_active && cache_active && c.save_hold &&
         c.save_hold->token == hold &&
         c.h.busy && model.active == &c);
  assert(et_g3t_private_call_abort_v1(&c));
  assert(et_g3t_private_call_acquire_v1(&c, 2));
  assert(!g3t_active(&c));
  assert(et_g3t_private_generator_close_v1(&c));
  assert(et_g3t_private_committed_save_copy_v1(
      &c, &model, &destination, &staging));
  assert(c.save_hold && c.save_hold->token == hold &&
         pin_active && cache_active &&
         c.history_length == 2 && c.rng[3] == -1);
  assert(!et_g3t_private_committed_save_hold_copy_v1(hold, &held));
  assert(held.snapshot.ids[0] == 66 && held.snapshot.ids[1] == 65 &&
         held.snapshot.prompt_length == 1 &&
         held.snapshot.generated_count == 1 &&
         held.snapshot.generated_eos_terminal == 1 &&
         held.snapshot.rng[2] == -1 && held.snapshot.rng[3] == -1 &&
         !memcmp(held.model_bits, model_bits, sizeof(model_bits)));
  et_g3r_committed_save_capture_v1 saved = held;
  bytes4872 encoded = {.length = sizeof(encoded.data)};
  memset(encoded.data, 0x5a, sizeof(encoded.data));
  assert(!et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  assert(!memcmp(encoded.data, &saved, sizeof(saved)));
  memset(encoded.data, 0x5a, sizeof(encoded.data));
  encoded.length = sizeof(encoded.data) - 1;
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  encoded.length = sizeof(encoded.data) + 1;
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  encoded.length = sizeof(encoded.data);
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(0, &encoded));
  unsigned char sentinel[sizeof(encoded.data)];
  memset(sentinel, 0x5a, sizeof(sentinel));
  assert(!memcmp(encoded.data, sentinel, sizeof(sentinel)));
  assert(et_g3t_private_committed_save_hold_copy_v1(
      hold, (et_g3r_committed_save_capture_v1 *)&c));
  fake_overlap = 1;
  assert(et_g3t_private_committed_save_hold_copy_v1(hold, &held));
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  fake_overlap = 0;
  fake_i64_overlap = 1;
  assert(et_g3t_private_committed_save_hold_copy_v1(hold, &held));
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  fake_i64_overlap = 0;
  g3t_save_hold other = {0};
  other.next = g3t_save_holds;
  g3t_save_holds = &other;
  assert(et_g3t_private_committed_save_hold_copy_v1(
      hold, (et_g3r_committed_save_capture_v1 *)&other));
  g3t_save_holds = other.next;
  struct { int64_t length; g3t_save_hold record; } header_overlap = {0};
  header_overlap.length = sizeof(encoded.data);
  header_overlap.record.next = g3t_save_holds;
  g3t_save_holds = &header_overlap.record;
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(
      hold, &header_overlap));
  g3t_save_holds = header_overlap.record.next;
  assert(!memcmp(&held, &saved, sizeof(held)));
  model_bits[0] = 1;
  assert(et_g3t_private_committed_save_hold_copy_v1(hold, &held));
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  assert(!memcmp(encoded.data, sentinel, sizeof(sentinel)));
  assert(!memcmp(&held, &saved, sizeof(held)));
  model_bits[0] = 0;
  assert(!et_g3t_private_committed_save_hold_end_v1(hold));
  assert(!pin_active && !cache_active && !c.save_hold && !c.h.busy &&
         !model.active);
  assert(et_g3t_private_committed_save_hold_copy_v1(hold, &held));
  assert(et_g3t_private_committed_save_hold_end_v1(hold));
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(hold, &encoded));
  int64_t successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor && successor != hold &&
         et_g3t_private_committed_save_hold_end_v1(hold) &&
         c.save_hold && c.save_hold->token == successor &&
         pin_active && cache_active);
  assert(!et_g3t_private_committed_save_hold_end_v1(successor));
  fail_pin = 1;
  assert(!et_g3t_private_committed_save_hold_begin_v1(&c, &model));
  fail_pin = 0;
  assert(!pin_active && !cache_active && !c.save_hold);
  fail_cache = 1;
  assert(!et_g3t_private_committed_save_hold_begin_v1(&c, &model));
  fail_cache = 0;
  assert(!pin_active && !cache_active && !c.save_hold);
  cache_length = 1;
  assert(!et_g3t_private_committed_save_hold_begin_v1(&c, &model));
  cache_length = 2;
  assert(!pin_active && !cache_active && !c.save_hold && !model.active);
  assert(!et_g3t_private_committed_save_hold_begin_v1(&c, &foreign));
  c.history_pending_g1 = 1;
  assert(!et_g3t_private_committed_save_hold_begin_v1(&c, &model));
  c.history_pending_g1 = 0;
  successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor);
  c.binding_values[0] ^= 1u;
  assert(et_g3t_private_committed_save_hold_end_v1(successor));
  c.binding_values[0] ^= 1u;
  assert(!pin_active && !cache_active && !c.save_hold && !model.active &&
         !g3t_save_holds);
  successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor);
  et_a2_kv_cache *original_cache = c.cache;
  c.cache = (et_a2_kv_cache *)&parameters[1];
  assert(et_g3t_private_committed_save_hold_copy_v1(successor, &held));
  assert(et_g3t_private_committed_save_hold_end_v1(successor));
  c.cache = original_cache;
  assert(!pin_active && !cache_active && !c.save_hold && !model.active &&
         !g3t_save_holds);
  successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor);
  c.rng[2] = 0;
  assert(g3t_history_valid(&c));
  assert(et_g3t_private_committed_save_hold_copy_v1(successor, &held));
  assert(et_g3t_private_committed_save_hold_copy_bytes_v1(
      successor, &encoded));
  assert(!memcmp(encoded.data, sentinel, sizeof(sentinel)));
  assert(!memcmp(&held, &saved, sizeof(held)));
  assert(et_g3t_private_committed_save_hold_end_v1(successor));
  c.rng[2] = -1;
  successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor);
  c.policy[0] = 0;
  assert(g3t_history_valid(&c));
  assert(et_g3t_private_committed_save_hold_copy_v1(successor, &held));
  assert(!memcmp(&held, &saved, sizeof(held)));
  assert(et_g3t_private_committed_save_hold_end_v1(successor));
  c.policy[0] = 1;
  successor = et_g3t_private_committed_save_hold_begin_v1(&c, &model);
  assert(successor);
  c.history_ids[1] = 64;
  c.history_generated_eos_terminal = 0;
  assert(g3t_history_valid(&c));
  assert(et_g3t_private_committed_save_hold_copy_v1(successor, &held));
  assert(!memcmp(&held, &saved, sizeof(held)));
  assert(et_g3t_private_committed_save_hold_end_v1(successor));
  c.history_ids[1] = 65;
  c.history_generated_eos_terminal = 1;
  assert(!pin_active && !cache_active && !c.save_hold && !model.active &&
         !g3t_save_holds);
  c.policy[0] = 0; c.rng[3] = 7;
  c.policy[4] = 0;
  positive(65, 0, 1, 1, 0, 0, 0);
  c.policy[4] = 1;
  c.history_pending_g1 = 1; reject(); c.history_pending_g1 = 0;
  c.h.busy = 1; reject(); c.h.busy = 0;
  c.frame.active = 1; reject(); c.frame.active = 0;
  c.pending_output = (g3t_output *)&parameters[0]; reject(); c.pending_output = NULL;
  c.pending_logits = (g3t_logits *)&parameters[0]; reject(); c.pending_logits = NULL;
  c.pins.held_mask = 1; reject(); c.pins.held_mask = 0;
  cache_active = 1;
  assert(et_g3t_private_committed_save_copy_v1(
      &c, &model, &destination, &staging));
  cache_active = 0; unchanged();
  cache_length = 2; reject(); cache_length = 1;
  fail_pin = 1; reject(); fail_pin = 0;
  fail_cache = 1; reject(); fail_cache = 0;
  fail_layer = 1; reject(); fail_layer = 0;
  malformed_lengths = 1; reject(); malformed_lengths = 0;
  fake_overlap = 1; reject(); fake_overlap = 0;
  model_bits[0] = 1; reject(); model_bits[0] = 0;
  c.policy[2] = 0; reject(); c.policy[2] = 256;
  destination.length = 135; reject(); destination.length = 136;
  staging.length = 135; reject(); staging.length = 136;
  assert(et_g3t_private_committed_save_copy_v1(
      &c, &foreign, &destination, &staging)); unchanged();
  assert(et_g3t_private_committed_save_copy_v1(
      &c, &model, &destination, &destination)); unchanged();
  c.history_length = cache_length = 0; c.history_prompt_length = 0;
  c.history_generated_count = c.history_generated_eos_terminal = 0;
  c.history_ids[0] = c.history_ids[1] = 0; c.prefill_committed = 0;
  reject();
  puts("G3-R committed save native copy PASS");
  return 0;
}
