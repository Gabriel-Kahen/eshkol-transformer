/* Source-private G3-T transport composition. M3T's native registry and owner
 * are included once; no C4 identity or transport source enters this TU. */
#include "m3_model.c"
#include "g3t_transport.h"
#include "m3_call_pins.h"
#include "eshkol_transformer/a2_kv_cache.h"
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/g3n_primitives_abi.h"
#include "eshkol_transformer/g3s_sampling_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#endif
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { G3T_GENERATOR = 1, G3T_INPUT = 2, G3T_OUTPUT = 4,
       G3T_PENDING = 0, G3T_LIVE = 1, G3T_DEAD = -1 };
enum { G3T_ARGUMENT = 1, G3T_STATE = 2, G3T_INTERNAL = 5 };
enum { G3T_IDENTITY = 1, G3T_LIFECYCLE = 2, G3T_CONFIG = 3,
       G3T_TOPOLOGY = 6, G3T_ALLOCATION = 7, G3T_INVARIANT = 10 };
typedef struct g3t_record {
  struct g3t_record *next;
  int kind, state, busy;
} g3t_record;
typedef struct g3t_frame {
  int active, next_ordinal, prepared;
  int64_t token;
  et_a2_kv_cache *candidate_cache;
  et_a2_kv_cache_transaction *transaction;
  float et[4], ep[4], x[4], n1[4], qt[4], kt[4], vt[4];
  float qh[4], kh[4], vh[4], ah[4], at[4], ao[4], r[4], n2[4];
  float fu[8], fg[8], fd[4], y[4], nf[4], z[256];
  const void *binding_identities[14];
  unsigned char binding_values[4736];
} g3t_frame;
typedef struct g3t_input {
  g3t_record h;
  int64_t length, ids[2];
} g3t_input;
struct g3t_context;
typedef struct g3t_output {
  g3t_record h;
  struct g3t_context *parent;
  int64_t prompt_length;
} g3t_output;
typedef struct g3t_context {
  g3t_record h;
  int call_kind;
  owner *model;
  et_a2_kv_cache *cache;
  int64_t policy[6], rng[4];
  et_g3t_model_pins_internal pins;
  g3t_output *pending_output;
  g3t_frame frame;
  float last_logits[256];
  int64_t sample_token, successor_rng[4];
  int prefill_committed, sampled;
  const void *binding_identities[14];
  unsigned char binding_values[4736];
  int binding_ready;
} g3t_context;
static g3t_record *g3t_registry;
static int64_t g3t_error_domain, g3t_error_category, g3t_error_code;

static void g3t_clear(void) {
  g3t_error_domain = g3t_error_category = g3t_error_code = 0;
}
static int64_t g3t_fail(int64_t domain, int64_t category, int64_t code) {
  g3t_error_domain = domain;
  g3t_error_category = category;
  g3t_error_code = code;
  return category;
}
static int64_t g3t_bad(int64_t category, int64_t code) {
  return g3t_fail(0, category, code);
}
static int64_t g3t_f32_failure(const et_f32_tensor_error *error) {
  return g3t_fail(3, error->category, error->code);
}
static int64_t g3t_a2_failure(const et_kernel_error *error) {
  return g3t_fail(1, error->category, error->code);
}
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
static int g3t_capture_kernel(int32_t status, const et_kernel_error *error) {
  if (!status) return 0;
  g3t_a2_failure(error);
  return 1;
}
static et_kernel_tensor_view_v1 g3t_view(
    void *data, size_t bytes, const char *dtype, size_t rank,
    const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes, dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, rank, shape};
}
static const et_kernel_provider_v1 *g3t_resolve_provider(
    void *context, const char *symbol) {
  return symbol && strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? (const et_kernel_provider_v1 *)context : NULL;
}
static int64_t g3t_dispatch(
    const et_kernel_provider_v1 *provider, const char *capability,
    const char *operation, size_t rank, const uint64_t *shape,
    et_kernel_tensor_view_v1 *inputs, size_t input_count,
    et_kernel_tensor_view_v1 output) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  if (g3t_capture_kernel(et_kernel_runtime_discover(
          g3t_resolve_provider, (void *)provider, &runtime, &error), &error))
    return g3t_error_category;
  et_kernel_request_v1 request = {
      sizeof(request), operation, "f32", "cpu", rank, shape, 1u, {0}};
  et_kernel_call_v1 call = {
      sizeof(call), capability, &request, input_count, sizeof(inputs[0]),
      input_count * sizeof(inputs[0]), inputs, 1u, sizeof(output),
      sizeof(output), &output};
  int32_t status = et_kernel_runtime_dispatch(runtime, &call, &error);
  int failed = g3t_capture_kernel(status, &error);
  et_kernel_runtime_destroy(runtime);
  return failed ? g3t_error_category : 0;
}
#endif
int64_t et_g3t_private_last_error_domain_v1(void) { return g3t_error_domain; }
int64_t et_g3t_private_last_error_category_v1(void) { return g3t_error_category; }
int64_t et_g3t_private_last_error_code_v1(void) { return g3t_error_code; }

#ifdef ET_G3T_TESTING
#ifndef ET_A2_KV_CACHE_TESTING
#error "G3-T allocation witness requires A2 cache testing hooks"
#endif
static uint64_t g3t_allocation_limit = UINT64_MAX, g3t_allocations;
int64_t et_g3t_test_fail_alloc_after_v1(uint64_t count) {
  g3t_allocation_limit = count;
  g3t_allocations = 0;
  return 0;
}
int64_t et_g3t_test_fail_a2_after_v1(uint64_t count) {
  et_a2_kv_cache_test_fail_alloc_after_v1((size_t)count);
  return 0;
}
uint64_t et_g3t_test_live_contexts_v1(void) {
  uint64_t count = 0;
  for (g3t_record *r = g3t_registry; r; r = r->next)
    if (r->kind == G3T_GENERATOR && r->state == G3T_LIVE) ++count;
  return count;
}
#endif
static g3t_context *g3t_allocate(void) {
#ifdef ET_G3T_TESTING
  if (g3t_allocations >= g3t_allocation_limit) return NULL;
#endif
  g3t_context *c = calloc(1, sizeof(*c));
#ifdef ET_G3T_TESTING
  if (c) ++g3t_allocations;
#endif
  return c;
}
/* Search by address before reading any candidate bytes. Dead headers remain
 * enrolled, so an old pointer cannot gain authority through address reuse. */
static g3t_record *g3t_admit_record(void *candidate, int kind, int allow_dead) {
  g3t_record *r = g3t_registry;
  while (r && r != candidate) r = r->next;
  if (!r || r->kind != kind) {
    g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
    return NULL;
  }
  if (!allow_dead && r->state != G3T_LIVE) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  return r;
}
static g3t_context *g3t_admit(void *candidate, int allow_dead) {
  return (g3t_context *)g3t_admit_record(candidate, G3T_GENERATOR, allow_dead);
}
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
static g3t_context *g3t_active(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return NULL;
  if (!c->h.busy || !c->model || c->model->active != c ||
      c->pins.held_mask != UINT16_C(0x3fff)) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  return c;
}
#endif
static owner *g3t_model(void *candidate) {
  record *r = registry;
  while (r && r != candidate) r = r->next;
  if (!r || r->kind != OWNER) {
    g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
    return NULL;
  }
  owner *o = (owner *)r;
  record *original = registry, *successor = registry;
  while (original && original != (record *)o->original) original = original->next;
  while (successor && successor != (record *)o->successor) successor = successor->next;
  if (o->r.state != 1 || o->initialized != 14 || !original || !successor ||
      original->kind != INITIALIZER || successor->kind != INITIALIZER ||
      original->state != 1 || successor->state != 1) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  for (size_t i = 0; i < 14; ++i) {
    if (!o->p[i] || !o->handles[i]) {
      g3t_bad(G3T_STATE, G3T_TOPOLOGY);
      return NULL;
    }
    for (size_t j = 0; j < i; ++j)
      if (o->p[i] == o->p[j] || o->handles[i] == o->handles[j]) {
        g3t_bad(G3T_STATE, G3T_TOPOLOGY);
        return NULL;
      }
    et_f32_tensor_error error;
    if (et_f32_parameter_validate_identity_v1(o->p[i], o->handles[i], &error)) {
      g3t_f32_failure(&error);
      return NULL;
    }
  }
  return o;
}
static int g3t_positive_f32_bits(int64_t candidate) {
  if (candidate < 0 || (uint64_t)candidate > UINT32_MAX) return 0;
  uint32_t bits = (uint32_t)candidate;
  return bits != 0 && (bits & UINT32_C(0x80000000)) == 0 &&
         (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}
static int g3t_policy_valid(int64_t mode, int64_t temperature_bits, int64_t k,
                            int64_t p_bits, int64_t max_new, int64_t eos) {
  if ((mode != 0 && mode != 1) ||
      !g3t_positive_f32_bits(temperature_bits) ||
      !g3t_positive_f32_bits(p_bits) ||
      p_bits > INT64_C(0x3f800000) || k < 1 || k > 256 ||
      max_new < 0 || max_new > 1 || eos < -1 || eos > 255) return 0;
  return mode != 0 ||
         (temperature_bits == INT64_C(0x3f800000) && k == 256 &&
          p_bits == INT64_C(0x3f800000));
}
void *et_g3t_private_generator_seed_v1(
    void *model_owner, int64_t seed, int64_t mode, int64_t temperature_bits,
    int64_t k, int64_t p_bits, int64_t max_new, int64_t eos) {
  g3t_clear();
  owner *o = g3t_model(model_owner);
  if (!o) return NULL;
  if (seed < 0 || !g3t_policy_valid(mode, temperature_bits, k, p_bits,
                                    max_new, eos)) {
    g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
    return NULL;
  }
  if (o->active) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  g3t_context *c = g3t_allocate();
  if (!c) {
    g3t_bad(G3T_INTERNAL, G3T_ALLOCATION);
    return NULL;
  }
  et_kernel_error error;
  if (et_a2_kv_cache_create_v1(1, 1, 2, 2, 2, &c->cache, &error)) {
    g3t_a2_failure(&error);
    free(c);
    return NULL;
  }
  c->h.kind = G3T_GENERATOR;
  c->h.state = G3T_LIVE;
  c->model = o;
  c->policy[0] = mode;
  c->policy[1] = temperature_bits;
  c->policy[2] = k;
  c->policy[3] = p_bits;
  c->policy[4] = max_new;
  c->policy[5] = eos;
  c->rng[0] = 1;
  c->rng[1] = seed;
  c->h.next = g3t_registry;
  g3t_registry = &c->h;
  return c;
}
int64_t et_g3t_private_generator_close_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 1);
  if (!c) return g3t_error_category;
  if (c->h.state == G3T_DEAD) return 0;
  owner *o = g3t_model(c->model);
  if (!o) return g3t_error_category;
  if (c->h.busy || o->active || c->pins.held_mask) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_kernel_error error;
  if (et_a2_kv_cache_destroy_v1(&c->cache, &error)) return g3t_a2_failure(&error);
  memset(c->policy, 0, sizeof(c->policy));
  memset(c->rng, 0, sizeof(c->rng));
  memset(c->binding_identities, 0, sizeof(c->binding_identities));
  memset(c->binding_values, 0, sizeof(c->binding_values));
  c->binding_ready = 0;
  c->model = NULL;
  c->h.state = G3T_DEAD;
  return 0;
}
int64_t et_g3t_private_call_acquire_v1(void *candidate, int64_t call_kind) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return g3t_error_category;
  if (call_kind < 0 || call_kind > 2) return g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
  owner *o = g3t_model(c->model);
  if (!o) return g3t_error_category;
  if (c->h.busy || o->active || c->pins.held_mask || !c->cache)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (call_kind == 2 &&
      (c->policy[4] != 1 || c->prefill_committed ||
       (c->policy[0] == 1 && c->rng[2] == -1 && c->rng[3] == -1)))
    return g3t_bad(G3T_STATE, G3T_CONFIG);
  et_f32_tensor_error error;
  if (et_g3t_model_pins_begin_internal(o->p, (const void *const *)o->handles,
                                       &c->pins, &error)) return g3t_f32_failure(&error);
  c->call_kind = (int)call_kind;
  c->h.busy = 1;
  o->active = c;
  return 0;
}
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
void *et_g3t_private_input_from_token_v1(int64_t token) {
  g3t_clear();
  if (token < 0 || token > 255) {
    g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
    return NULL;
  }
  g3t_input *input = calloc(1, sizeof(*input));
  if (!input) {
    g3t_bad(G3T_INTERNAL, G3T_ALLOCATION);
    return NULL;
  }
  input->h.kind = G3T_INPUT;
  input->h.state = G3T_LIVE;
  input->length = 1;
  input->ids[0] = token;
  input->h.next = g3t_registry;
  g3t_registry = &input->h;
  return input;
}
int64_t et_g3t_private_tensor_release_v1(void *candidate) {
  g3t_clear();
  g3t_input *input = (g3t_input *)g3t_admit_record(
      candidate, G3T_INPUT, 1);
  if (!input) return g3t_error_category;
  if (input->h.busy) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  input->length = 0;
  memset(input->ids, 0, sizeof(input->ids));
  input->h.state = G3T_DEAD;
  return 0;
}
void *et_g3t_private_output_reserve_v1(void *candidate, int64_t prompt_length) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return NULL;
  if (c->call_kind != 2 || prompt_length != 1 || c->pending_output ||
      c->prefill_committed || c->frame.active) {
    g3t_bad(G3T_STATE, G3T_CONFIG);
    return NULL;
  }
  g3t_output *output = calloc(1, sizeof(*output));
  if (!output) {
    g3t_bad(G3T_INTERNAL, G3T_ALLOCATION);
    return NULL;
  }
  output->h.kind = G3T_OUTPUT;
  output->h.state = G3T_PENDING;
  output->h.busy = 1;
  output->parent = c;
  output->prompt_length = prompt_length;
  output->h.next = g3t_registry;
  g3t_registry = &output->h;
  c->pending_output = output;
  return output;
}
int64_t et_g3t_private_output_release_v1(void *candidate) {
  g3t_clear();
  g3t_output *output = (g3t_output *)g3t_admit_record(
      candidate, G3T_OUTPUT, 1);
  if (!output) return g3t_error_category;
  if (output->h.busy) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  output->h.state = G3T_DEAD;
  output->parent = NULL;
  output->prompt_length = 0;
  return 0;
}
int64_t et_g3t_private_frame_begin_v1(
    void *candidate, void *input_candidate, int64_t length) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  g3t_input *input = (g3t_input *)g3t_admit_record(
      input_candidate, G3T_INPUT, 0);
  if (!input) return g3t_error_category;
  if (c->call_kind != 2 || !c->pending_output || c->frame.active ||
      c->prefill_committed || c->sampled || length != 1 ||
      input->length != 1 || input->h.busy ||
      c->pending_output->prompt_length != 1)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_kernel_error error;
  et_a2_kv_cache *cache = NULL;
  if (g3t_capture_kernel(et_a2_kv_cache_create_v1(
          1, 1, 2, 2, 2, &cache, &error), &error))
    return g3t_error_category;
  memset(&c->frame, 0, sizeof(c->frame));
  c->frame.candidate_cache = cache;
  c->frame.token = input->ids[0];
  c->frame.active = 1;
  return 0;
}
#include "g3t_prefill_roles.inc"
int64_t et_g3t_private_frame_prepare_v1(
    void *candidate, void *staged_result) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (staged_result) return g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
  if (c->call_kind != 2 || !c->pending_output || !c->frame.active ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      c->frame.prepared || c->prefill_committed || c->sampled)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_f32_tensor_error error;
  if (et_g3t_model_pins_check_internal(&c->pins, &error))
    return g3t_f32_failure(&error);
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *view = &c->pins.views[i];
    if (!view->data || !c->pins.identities[i] ||
        view->byte_length > sizeof(c->frame.binding_values) - offset)
      return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    c->frame.binding_identities[i] = c->pins.identities[i];
    memcpy(c->frame.binding_values + offset, view->data, view->byte_length);
    offset += view->byte_length;
  }
  if (offset != sizeof(c->frame.binding_values))
    return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
  c->frame.prepared = 1;
  return 0;
}
int64_t et_g3t_private_frame_commit_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || !c->pending_output || !c->frame.active ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      !c->frame.prepared ||
      c->prefill_committed || c->sampled)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_kernel_error error;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_commit_v1(
          &c->frame.transaction, &error), &error)) return g3t_error_category;
  /* The old cache is empty and unborrowed. A failed destroy retains the
   * candidate for abort, with no prompt/logit publication. */
  if (g3t_capture_kernel(et_a2_kv_cache_destroy_v1(&c->cache, &error),
                         &error)) return g3t_error_category;
  c->cache = c->frame.candidate_cache;
  c->frame.candidate_cache = NULL;
  memcpy(c->binding_identities, c->frame.binding_identities,
         sizeof(c->binding_identities));
  memcpy(c->binding_values, c->frame.binding_values,
         sizeof(c->binding_values));
  c->binding_ready = 1;
  memcpy(c->last_logits, c->frame.z, sizeof(c->last_logits));
  memset(&c->frame, 0, sizeof(c->frame));
  c->prefill_committed = 1;
  return 0;
}
int64_t et_g3t_private_sample_v1(void *candidate) {
  static const uint64_t logits_shape[2] = {1,256}, rng_shape[1] = {4};
  static const uint64_t token_shape[1] = {1};
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || !c->prefill_committed || c->frame.active ||
      !c->pending_output || c->sampled || !c->binding_ready)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_f32_tensor_error pin_error;
  if (et_g3t_model_pins_check_internal(&c->pins, &pin_error))
    return g3t_f32_failure(&pin_error);
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *view = &c->pins.views[i];
    if (!view->data || c->binding_identities[i] != c->pins.identities[i] ||
        view->byte_length > sizeof(c->binding_values) - offset)
      return g3t_bad(G3T_STATE, G3T_TOPOLOGY);
    if (memcmp(c->binding_values + offset, view->data, view->byte_length))
      return g3t_bad(G3T_STATE, G3T_TOPOLOGY);
    offset += view->byte_length;
  }
  if (offset != sizeof(c->binding_values))
    return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
  uint32_t temperature_bits = (uint32_t)c->policy[1];
  uint32_t p_bits = (uint32_t)c->policy[3];
  float temperature, top_p;
  memcpy(&temperature, &temperature_bits, sizeof(temperature));
  memcpy(&top_p, &p_bits, sizeof(top_p));
  int categorical = c->policy[0] == 1;
  int64_t rng[4], token = -1, next[4] = {0};
  memcpy(rng, c->rng, sizeof(rng));
  int64_t k = c->policy[2];
  et_kernel_tensor_view_v1 inputs[5] = {
      g3t_view(c->last_logits, sizeof(c->last_logits), "f32", 2, logits_shape)};
  size_t count;
  if (categorical) {
    inputs[1] = g3t_view(&temperature, sizeof(temperature), "f32", 0, NULL);
    inputs[2] = g3t_view(&k, sizeof(k), "i64", 0, NULL);
    inputs[3] = g3t_view(&top_p, sizeof(top_p), "f32", 0, NULL);
    inputs[4] = g3t_view(rng, sizeof(rng), "i64", 1, rng_shape);
    count = 5;
  } else {
    inputs[1] = g3t_view(rng, sizeof(rng), "i64", 1, rng_shape);
    count = 2;
  }
  et_kernel_tensor_view_v1 outputs[2] = {
      g3t_view(&token, sizeof(token), "i64", 1, token_shape),
      g3t_view(next, sizeof(next), "i64", 1, rng_shape)};
  et_kernel_request_v1 request = {
      sizeof(request), categorical ? "g3s.categorical.forward" :
          "g3s.greedy.forward", "f32", "cpu", 2, logits_shape, 1u, {0}};
  et_kernel_call_v1 call = {
      sizeof(call), categorical ? "g3s.categorical" : "g3s.greedy",
      &request, count, sizeof(inputs[0]), count * sizeof(inputs[0]), inputs,
      2, sizeof(outputs[0]), sizeof(outputs), outputs};
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  if (g3t_capture_kernel(et_kernel_runtime_discover(g3t_resolve_provider,
          (void *)et_g3s_kernel_provider_v1(), &runtime, &error), &error))
    return g3t_error_category;
  int32_t status = et_kernel_runtime_dispatch(runtime, &call, &error);
  int failed = g3t_capture_kernel(status, &error);
  et_kernel_runtime_destroy(runtime);
  if (failed) return g3t_error_category;
  if (token < 0 || token > 255 || next[0] != 1 || next[1] != rng[1])
    return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
  c->sample_token = token;
  memcpy(c->successor_rng, next, sizeof(next));
  c->sampled = 1;
  return 0;
}
#endif
int64_t et_g3t_private_call_abort_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return g3t_error_category;
  if (!c->h.busy) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  owner *o = g3t_model(c->model);
  if (!o || o->active != c) abort();
  et_f32_tensor_error error;
  if (et_g3t_model_pins_check_internal(&c->pins, &error)) abort();
  et_kernel_error kernel_error;
  if (c->frame.transaction && et_a2_kv_cache_transaction_abort_v1(
          &c->frame.transaction, &kernel_error)) abort();
  if (c->frame.candidate_cache && et_a2_kv_cache_destroy_v1(
          &c->frame.candidate_cache, &kernel_error)) abort();
  memset(&c->frame, 0, sizeof(c->frame));
  if (c->pending_output) {
    c->pending_output->h.busy = 0;
    c->pending_output->h.state = G3T_DEAD;
    c->pending_output->parent = NULL;
    c->pending_output->prompt_length = 0;
    c->pending_output = NULL;
  }
  c->sampled = 0;
  c->sample_token = 0;
  memset(c->successor_rng, 0, sizeof(c->successor_rng));
  et_g3t_model_pins_end_internal(&c->pins);
  o->active = NULL;
  c->h.busy = 0;
  c->call_kind = 0;
  return 0;
}
#ifdef ET_G3T_TESTING
int64_t et_g3t_test_pin_count_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return -1;
  if (!c->h.busy) return c->pins.held_mask ? -1 : 0;
  et_f32_tensor_error error;
  return et_g3t_model_pins_check_internal(&c->pins, &error) ? -1 : 14;
}
int64_t et_g3t_test_cache_empty_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return 0;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL, *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  et_kernel_error error;
  if (et_a2_kv_cache_read_borrow_begin_v1(c->cache, &borrow, &error)) return 0;
  int ok = !et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0, &keys, &values, &lengths, &mask, &error) &&
      keys && values && lengths && mask && ((const int64_t *)lengths->data)[0] == 0;
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error)) abort();
  return ok;
}
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
int64_t et_g3t_test_logit_bits_v1(void *candidate, int64_t index) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || index < 0 || index >= 256 || !c->prefill_committed) return -1;
  uint32_t bits;
  memcpy(&bits, &c->last_logits[index], sizeof(bits));
  return bits;
}
int64_t et_g3t_test_sample_token_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  return c && c->sampled ? c->sample_token : -1;
}
int64_t et_g3t_test_rng_word_v1(void *candidate, int64_t index) {
  g3t_context *c = g3t_admit(candidate, 0);
  return c && index >= 0 && index < 4 ? c->rng[index] : -1;
}
int64_t et_g3t_test_successor_word_v1(void *candidate, int64_t index) {
  g3t_context *c = g3t_admit(candidate, 0);
  return c && c->sampled && index >= 0 && index < 4
             ? c->successor_rng[index] : -1;
}
int64_t et_g3t_test_greedy_argmax_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || !c->prefill_committed) return -1;
  int64_t best = 0;
  for (int64_t i = 1; i < 256; ++i)
    if (c->last_logits[i] > c->last_logits[best]) best = i;
  return best;
}
#endif
#endif
