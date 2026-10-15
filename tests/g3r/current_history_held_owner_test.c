#define ET_G3R_SAME_OWNER_SAVE_PRIVATE 1
#define ET_G3R_COMMITTED_SAVE_SNAPSHOT_PRIVATE 1
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
static int64_t cache_length;
static int pin_active, cache_active, fail_pin, fail_cache, fail_layer;
static int malformed_lengths;
static int fake_overlap;
static unsigned char model_bits[4736];
static int64_t lengths_data[1];
static uint64_t lengths_shape[1] = {1};
static et_kernel_tensor_view_v1 lengths_view;
static et_kernel_tensor_view_v1 keys_view, values_view, mask_view;
static float keys_data[8], values_data[8];
static uint8_t mask_data[2];
static uint64_t kv_shape[4] = {1,2,2,2}, mask_shape[2] = {1,2};

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
  keys_view = (et_kernel_tensor_view_v1){0};
  keys_view.struct_size = sizeof(keys_view);
  keys_view.data = keys_data; keys_view.rank = 4; keys_view.shape = kv_shape;
  keys_view.byte_length = sizeof(keys_data); keys_view.dtype = "f32";
  keys_view.device = "cpu"; keys_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  values_view = keys_view; values_view.data = values_data;
  mask_data[0] = 1; mask_data[1] = cache_length == 2;
  mask_view = (et_kernel_tensor_view_v1){0};
  mask_view.struct_size = sizeof(mask_view);
  mask_view.data = mask_data; mask_view.rank = 2; mask_view.shape = mask_shape;
  mask_view.byte_length = sizeof(mask_data); mask_view.dtype = "bool";
  mask_view.device = "cpu"; mask_view.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  *keys = &keys_view; *values = &values_view; *mask = &mask_view;
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

static g3t_context c;
static owner model, foreign;
static struct { owner model; unsigned char tail[136]; } foreign_partial;
static struct { g3t_input input; unsigned char tail[136]; } transport_partial;
static struct { m3_graph graph; unsigned char tail[136]; } graph_partial;
static struct { m3_logits logits; unsigned char tail[136]; } logits_partial;
static workspace native_workspace;
static struct { m3_workspace shell; unsigned char tail[136]; } workspace_partial;
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
  foreign_partial.model = foreign;
  foreign.r.next = &foreign_partial.model.r;
  graph_partial.graph.r.kind = M3_GRAPH;
  graph_partial.graph.r.state = 1;
  logits_partial.logits.r.kind = M3_LOGITS;
  logits_partial.logits.r.state = 1;
  native_workspace.r.kind = WORKSPACE;
  native_workspace.r.state = 1;
  foreign_partial.model.r.next = &graph_partial.graph.r;
  graph_partial.graph.r.next = &logits_partial.logits.r;
  logits_partial.logits.r.next = &native_workspace.r;
  native_workspace.r.next = &initial.r;
  initial.r.next = &next.r; registry = &model.r;
  workspace_partial.shell.value = &native_workspace;
  m3_workspaces = &workspace_partial.shell;
  c.h.kind = G3T_GENERATOR; c.h.state = G3T_LIVE;
  transport_partial.input.h.kind = G3T_INPUT;
  transport_partial.input.h.state = G3T_LIVE;
  c.h.next = &transport_partial.input.h; g3t_registry = &c.h;
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
  c.prefill_committed = 1; c.binding_ready = 1;
  c.history_ids[0] = 66; c.history_ids[1] = 0;
  c.history_length = cache_length = c.history_prompt_length = 1;
  c.history_generated_count = c.history_manual_count = 0;
  c.history_generated_eos_terminal = 0;
  c.history_pending_g1 = 1;
  assert(!et_g3r_private_g3t_save_begin_v1(
      &c, (void *const *)model.p, (const void *const *)model.handles));
  assert(!pin_active && !cache_active);
  c.history_pending_g1 = 0;
  void *hold = et_g3r_private_g3t_save_begin_v1(
      &c, (void *const *)model.p, (const void *const *)model.handles);
  assert(hold && pin_active && cache_active && c.h.busy);
  et_g3r_g3t_held_words_v1 words = {0};
  assert(!et_g3r_private_g3t_save_copy_v1(hold, &words, sizeof(words)));
  assert(words.word[0] == c.policy[0] && words.word[6] == c.rng[0]);
  assert(words.word[10] == 66 && words.word[12] == 1);
  assert(words.word[16] == 0);
  unsigned char foreign_before[sizeof(foreign)];
  unsigned char model_before[sizeof(foreign_partial)];
  unsigned char transport_before[sizeof(transport_partial)];
  unsigned char graph_before[sizeof(graph_partial)];
  unsigned char logits_before[sizeof(logits_partial)];
  unsigned char workspace_before[sizeof(workspace_partial)];
  memcpy(foreign_before, &foreign, sizeof(foreign));
  memcpy(model_before, &foreign_partial, sizeof(foreign_partial));
  memcpy(transport_before, &transport_partial, sizeof(transport_partial));
  memcpy(graph_before, &graph_partial, sizeof(graph_partial));
  memcpy(logits_before, &logits_partial, sizeof(logits_partial));
  memcpy(workspace_before, &workspace_partial, sizeof(workspace_partial));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)&foreign, sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(foreign_before, &foreign, sizeof(foreign)));
  assert(et_g3t_private_last_error_code_v1() == G3T_IDENTITY);
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)
          ((char *)&foreign_partial.model + sizeof(owner) - 8),
      sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(model_before, &foreign_partial, sizeof(foreign_partial)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)
          ((char *)&transport_partial.input + sizeof(g3t_input) - 8),
      sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(transport_before, &transport_partial,
                 sizeof(transport_partial)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)&graph_partial.graph,
      sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(graph_before, &graph_partial, sizeof(graph_partial)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)
          ((char *)&logits_partial.logits + sizeof(m3_logits) - 8),
      sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(logits_before, &logits_partial, sizeof(logits_partial)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)
          ((char *)&workspace_partial.shell + sizeof(m3_workspace) - 8),
      sizeof(words)) == G3T_ARGUMENT);
  assert(!memcmp(workspace_before, &workspace_partial,
                 sizeof(workspace_partial)));
  assert(pin_active && cache_active && c.h.busy);
  assert(!et_g3r_private_g3t_save_check_v1(hold));
  unsigned char held_words_before[sizeof(words)];
  memcpy(held_words_before, &words, sizeof(words));
  assert(!et_g3r_private_g3t_save_begin_v1(
      &c, (void *const *)model.p, (const void *const *)model.handles));
  c.policy[0] ^= 1;
  assert(et_g3r_private_g3t_save_copy_v1(hold, &words, sizeof(words)));
  c.policy[0] ^= 1;
  c.rng[2] ^= 1;
  assert(et_g3r_private_g3t_save_copy_v1(hold, &words, sizeof(words)));
  c.rng[2] ^= 1;
  c.history_generated_eos_terminal = 1;
  assert(et_g3r_private_g3t_save_copy_v1(hold, &words, sizeof(words)));
  c.history_generated_eos_terminal = 0;
  model_bits[0] = 1;
  assert(et_g3r_private_g3t_save_copy_v1(hold, &words, sizeof(words)));
  model_bits[0] = 0;
  assert(!memcmp(held_words_before, &words, sizeof(words)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, (et_g3r_g3t_held_words_v1 *)((char *)hold + 1),
      sizeof(words)));
  assert(et_g3r_private_g3t_save_copy_v1(
      hold, &words, sizeof(words) - 1));
  c.history_pending_g1 = 1;
  assert(et_g3r_private_g3t_save_check_v1(hold));
  c.history_pending_g1 = 0;
  c.history_ids[0] ^= 1;
  assert(et_g3r_private_g3t_save_check_v1(hold));
  c.history_ids[0] ^= 1;
  keys_data[0] = 1.0f;
  assert(et_g3r_private_g3t_save_check_v1(hold));
  keys_data[0] = 0.0f;
  assert(!et_g3r_private_g3t_save_check_v1(hold));
  assert(!et_g3r_private_g3t_save_end_v1(hold));
  assert(!pin_active && !cache_active && !c.h.busy);
  assert(et_g3r_private_g3t_save_check_v1(hold));
  hold = et_g3r_private_g3t_save_begin_v1(
      &c, (void *const *)model.p, (const void *const *)model.handles);
  assert(hold);
  c.rng[2] ^= 1;
  assert(et_g3r_private_g3t_save_end_v1(hold));
  assert(!pin_active && !cache_active && !c.h.busy);
  puts("G3-R current-history held owner PASS");
  return 0;
}
