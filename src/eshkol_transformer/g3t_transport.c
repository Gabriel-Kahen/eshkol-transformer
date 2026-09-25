/* Source-private G3-T transport composition. M3T's native registry and owner
 * are included once; no C4 identity or transport source enters this TU. */
#include "m3_model.c"
#include "g3t_transport.h"
#include "m3_call_pins.h"
#include "eshkol_transformer/a2_kv_cache.h"
#include "eshkol_transformer/i64_tensor.h"
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/g3n_primitives_abi.h"
#include "eshkol_transformer/g3s_sampling_abi.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
#include "eshkol_transformer/n3k_primitives_abi.h"
#endif
#endif
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
#ifndef ET_G3T_OUTPUT_TEXT_PRIVATE
#error "G3-T final publication requires G1 ID and raw-text readiness"
#endif
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
#ifndef ET_G3T_FINAL_PUBLICATION_PRIVATE
#error "G3-T zero budget requires final publication"
#endif
#endif
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
#ifndef ET_G3T_ZERO_BUDGET_PRIVATE
#error "G3-T P2 zero budget requires P1 zero budget"
#endif
#endif
#endif
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef ET_G3T_OUTPUT_TEXT_PRIVATE
#if !defined(ET_G3T_PREFILL_SAMPLE_PRIVATE) || \
    !defined(ET_I64_TENSOR_STORAGE_QUERY_PRIVATE) || \
    !defined(ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE)
#error "G3-T output text requires prefill and I1/A2 storage queries"
#endif
#endif

enum { G3T_GENERATOR = 1, G3T_INPUT = 2, G3T_OUTPUT = 4,
       G3T_PENDING = 0, G3T_LIVE = 1, G3T_DEAD = -1 };
enum { G3T_ARGUMENT = 1, G3T_STATE = 2, G3T_SHAPE = 3,
       G3T_INTERNAL = 5 };
enum { G3T_IDENTITY = 1, G3T_LIFECYCLE = 2, G3T_CONFIG = 3,
       G3T_TOPOLOGY = 6, G3T_ALLOCATION = 7, G3T_INVARIANT = 10 };
typedef struct g3t_record {
  struct g3t_record *next;
  int kind, state, busy;
} g3t_record;
typedef struct g3t_frame {
  int active, kind, next_ordinal, prepared, end_prepared;
  int64_t token, prompt_length;
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
  int64_t prompt_ids[2];
  struct {
    float et[8], ep[8], x[8], n1[8], qt[8], kt[8], vt[8];
    float qh[8], kh[8], vh[8], ah[8], at[8], ao[8], r[8], n2[8];
    float fu[16], fg[16], fd[8], y[8], nf[8], z[512];
  } p2;
#endif
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
  struct g3t_context *parent_ctx;
  int64_t P, G;
  et_i64_tensor *ids;
  int64_t length, cache_length, rng[4];
  int numeric_ready, ids_copied, text_ready;
  unsigned char staged_id[8];
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
  int final_committed;
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
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
static int g3t_zero_prompt(const g3t_context *c, const g3t_output *out) {
  if (!c || !out || out->P != c->frame.prompt_length) return 0;
  return out->P == 1
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
         || out->P == 2
#endif
         ;
}
#endif
static int64_t g3t_f32_failure(const et_f32_tensor_error *error) {
  return g3t_fail(3, error->category, error->code);
}
static int64_t g3t_i64_failure(const et_i64_tensor_error *error) {
  return g3t_fail(2, error->category, error->code);
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
#ifdef ET_I64_TENSOR_TESTING
int64_t et_g3t_test_fail_i1_after_v1(uint64_t count) {
  et_i64_tensor_test_fail_alloc_after_v1((size_t)count);
  return 0;
}
#endif
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
static int64_t g3t_check_binding(g3t_context *c) {
  et_f32_tensor_error pin_error;
  if (et_g3t_model_pins_check_internal(&c->pins, &pin_error))
    return g3t_f32_failure(&pin_error);
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *view = &c->pins.views[i];
    if (!view->data || c->binding_identities[i] != c->pins.identities[i] ||
        view->byte_length > sizeof(c->binding_values) - offset ||
        memcmp(c->binding_values + offset, view->data, view->byte_length))
      return g3t_bad(G3T_STATE, G3T_TOPOLOGY);
    offset += view->byte_length;
  }
  return offset == sizeof(c->binding_values) ? 0 :
         g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
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
      (
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
       (c->policy[4] != 0 && c->policy[4] != 1) ||
#else
       c->policy[4] != 1 ||
#endif
       c->prefill_committed ||
       (c->policy[4] == 1 && c->policy[0] == 1 &&
        c->rng[2] == -1 && c->rng[3] == -1)))
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
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
int64_t et_g3t_private_prompt_preflight_v1(
    void *context_candidate, void *input_candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(context_candidate, 0);
  if (!c) return g3t_error_category;
  g3t_input *input = (g3t_input *)g3t_admit_record(
      input_candidate, G3T_INPUT, 0);
  if (!input) return g3t_error_category;
  if (input->length != 2 || input->length + c->policy[4] > 2)
    return g3t_bad(G3T_SHAPE, G3T_CONFIG);
  if (input->ids[0] < 0 || input->ids[0] > 255 ||
      input->ids[1] < 0 || input->ids[1] > 255)
    return g3t_bad(G3T_ARGUMENT, G3T_CONFIG);
  if (c->h.busy || c->prefill_committed || input->h.busy)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  return 0;
}
void *et_g3t_private_input_from_pair_v1(int64_t first, int64_t second) {
  g3t_clear();
  if (first < 0 || first > 255 || second < 0 || second > 255) {
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
  input->length = 2;
  input->ids[0] = first;
  input->ids[1] = second;
  input->h.next = g3t_registry;
  g3t_registry = &input->h;
  return input;
}
#endif
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
  if (c->call_kind != 2 ||
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
      (prompt_length != 1 && (prompt_length != 2 || c->policy[4] != 0)) ||
#else
      prompt_length != 1 ||
#endif
      c->pending_output ||
      c->prefill_committed || c->frame.active) {
    g3t_bad(G3T_STATE, G3T_CONFIG);
    return NULL;
  }
  g3t_output *output = calloc(1, sizeof(*output));
  if (!output) {
    g3t_bad(G3T_INTERNAL, G3T_ALLOCATION);
    return NULL;
  }
  const uint64_t shape[1] = {(uint64_t)c->policy[4]};
  et_i64_tensor_error error;
  if (et_i64_tensor_create_v1(1, shape, &output->ids, &error)) {
    g3t_i64_failure(&error);
    free(output);
    return NULL;
  }
  output->h.kind = G3T_OUTPUT;
  output->h.state = G3T_PENDING;
  output->h.busy = 1;
  output->parent_ctx = c;
  output->P = prompt_length;
  output->G = c->policy[4];
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
  if (output->h.state == G3T_DEAD) return 0;
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
  if (output->h.state == G3T_LIVE && !output->parent_ctx && output->ids) {
    et_i64_tensor_error error;
    if (et_m3_private_i64_unborrowed_v1(output->ids, &error))
      return g3t_i64_failure(&error);
    if (et_i64_tensor_destroy_v1(&output->ids, &error)) abort();
    output->P = output->G = output->length = output->cache_length = 0;
    memset(output->rng, 0, sizeof(output->rng));
    memset(output->staged_id, 0, sizeof(output->staged_id));
    output->numeric_ready = output->ids_copied = output->text_ready = 0;
    output->h.state = G3T_DEAD;
    return 0;
  }
#endif
  /* Pending outputs are destroyed by call abort; no live output exists yet. */
  return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
}
int64_t et_g3t_private_output_prepare_v1(
    void *context_candidate, void *output_candidate) {
  g3t_clear();
  g3t_context *c = g3t_active(context_candidate);
  if (!c) return g3t_error_category;
  g3t_output *output = (g3t_output *)g3t_admit_record(
      output_candidate, G3T_OUTPUT, 1);
  if (!output) return g3t_error_category;
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (c->policy[4] == 0) {
    if (output->h.state != G3T_PENDING || !output->h.busy ||
        c->call_kind != 2 || c->pending_output != output ||
        output->parent_ctx != c || !g3t_zero_prompt(c, output) ||
        output->G != 0 ||
        !output->ids || output->numeric_ready || output->ids_copied ||
        output->text_ready || output->length || output->cache_length ||
        c->prefill_committed || c->sampled || c->binding_ready ||
        !c->frame.active || c->frame.kind != 1 ||
        c->frame.next_ordinal != 21 || !c->frame.transaction ||
        c->frame.prepared)
      return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    et_f32_tensor_error error;
    if (et_g3t_model_pins_check_internal(&c->pins, &error))
      return g3t_f32_failure(&error);
    output->length = 0;
    output->cache_length = output->P;
    memcpy(output->rng, c->rng, sizeof(output->rng));
    output->numeric_ready = 1;
    return 0;
  }
#endif
  if (output->h.state != G3T_PENDING || !output->h.busy ||
      c->call_kind != 2 || c->pending_output != output ||
      output->parent_ctx != c || output->P != 1 || output->G != 1 ||
      !output->ids || output->numeric_ready || output->ids_copied ||
      output->text_ready || output->length || output->cache_length ||
      !c->prefill_committed || !c->sampled || !c->binding_ready ||
      !c->frame.active || c->frame.kind != 2 ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      c->frame.prepared || c->frame.token != c->sample_token ||
      c->sample_token < 0 || c->sample_token > 255)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (g3t_check_binding(c)) return g3t_error_category;
  /* The exact I1 copy is the last recoverable action. */
  int64_t token = c->frame.token;
  et_i64_tensor_error error;
  if (et_i64_tensor_copy_from_v1(output->ids, &token, 1, &error))
    return g3t_i64_failure(&error);
  output->length = 1;
  output->cache_length = 2;
  memcpy(output->rng, c->successor_rng, sizeof(output->rng));
  output->numeric_ready = 1;
  return 0;
}
#ifdef ET_G3T_OUTPUT_TEXT_PRIVATE
static int g3t_span_ok(const void *pointer, size_t bytes) {
  return pointer && (uintptr_t)pointer <= UINTPTR_MAX - bytes;
}
static int g3t_overlap(const void *left, size_t left_bytes,
                       const void *right, size_t right_bytes) {
  if (!left || !right || !left_bytes || !right_bytes) return 0;
  uintptr_t l = (uintptr_t)left, r = (uintptr_t)right;
  return l < r + right_bytes && r < l + left_bytes;
}
static int g3t_carrier_alias(const g3t_context *c,
                             const void *carrier, size_t bytes) {
  for (const g3t_record *r = g3t_registry; r; r = r->next) {
    size_t length = r->kind == G3T_GENERATOR ? sizeof(g3t_context) :
                    r->kind == G3T_INPUT ? sizeof(g3t_input) :
                    r->kind == G3T_OUTPUT ? sizeof(g3t_output) : sizeof(*r);
    if (g3t_overlap(carrier, bytes, r, length)) return 1;
  }
  if (g3t_overlap(carrier, bytes, c->model, sizeof(*c->model))) return 1;
  for (size_t i = 0; i < 14; ++i)
    if (g3t_overlap(carrier, bytes, c->pins.views[i].data,
                    c->pins.views[i].byte_length)) return 1;
  return et_i64_tensor_private_storage_overlap_v1(carrier, bytes) != 0 ||
         et_a2_kv_cache_private_storage_overlap_v1(carrier, bytes) != 0;
}
static g3t_output *g3t_ready_output(void *context_candidate,
                                    void *output_candidate, int text) {
  g3t_context *c = g3t_active(context_candidate);
  if (!c) return NULL;
  g3t_output *out = (g3t_output *)g3t_admit_record(
      output_candidate, G3T_OUTPUT, 1);
  if (!out) return NULL;
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (out->G == 0) {
    if (out->h.state != G3T_PENDING || !out->h.busy ||
        c->call_kind != 2 || c->pending_output != out ||
        out->parent_ctx != c || !g3t_zero_prompt(c, out) ||
        !out->ids || out->length != 0 || out->cache_length != out->P ||
        out->numeric_ready != 1 || out->ids_copied != text ||
        out->text_ready || c->prefill_committed || c->sampled ||
        c->binding_ready || !c->frame.active || c->frame.kind != 1 ||
        c->frame.next_ordinal != 21 || !c->frame.transaction ||
        c->frame.prepared) {
      g3t_bad(G3T_STATE, G3T_LIFECYCLE);
      return NULL;
    }
    et_f32_tensor_error error;
    if (et_g3t_model_pins_check_internal(&c->pins, &error))
      g3t_f32_failure(&error);
    else return out;
    return NULL;
  }
#endif
  if (out->h.state != G3T_PENDING || !out->h.busy ||
      c->call_kind != 2 || c->pending_output != out ||
      out->parent_ctx != c || out->P != 1 || out->G != 1 ||
      !out->ids || out->length != 1 || out->cache_length != 2 ||
      out->numeric_ready != 1 || out->ids_copied != text ||
      out->text_ready || !c->prefill_committed || !c->sampled ||
      !c->binding_ready || !c->frame.active || c->frame.kind != 2 ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      c->frame.prepared) {
    g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    return NULL;
  }
  if (g3t_check_binding(c)) return NULL;
  return out;
}
static int64_t g3t_decode_carrier(void *context_candidate,
                                  void *output_candidate, void *carrier,
                                  int text) {
  g3t_clear();
  g3t_output *out = g3t_ready_output(context_candidate, output_candidate, text);
  if (!out) return g3t_error_category;
  const size_t payload = out->G == 0 ? 0u : (text ? 1u : 8u);
  const size_t bytes = sizeof(int64_t) + payload;
  if (!g3t_span_ok(carrier, bytes) ||
      g3t_carrier_alias(out->parent_ctx, carrier, bytes))
    return g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
  int64_t declared = 0;
  memcpy(&declared, carrier, sizeof(declared));
  if (declared != (int64_t)payload)
    return g3t_bad(G3T_ARGUMENT, G3T_TOPOLOGY);

#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (out->G == 0) {
    et_i64_tensor_borrow *empty_borrow = NULL;
    const et_kernel_tensor_view_v1 *empty_view = NULL;
    et_i64_tensor_error empty_error;
    if (et_i64_tensor_borrow_begin_v1(out->ids, &empty_borrow, &empty_error))
      return g3t_i64_failure(&empty_error);
    if (et_i64_tensor_borrow_view_v1(empty_borrow, &empty_view, &empty_error)) {
      g3t_i64_failure(&empty_error);
    } else if (!empty_view || empty_view->rank != 1 ||
               !empty_view->shape || empty_view->shape[0] != 0 ||
               empty_view->byte_length != 0) {
      g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    }
    int64_t category = g3t_error_category;
    if (et_i64_tensor_borrow_end_v1(&empty_borrow, &empty_error)) abort();
    if (category) return category;
    if (text) out->text_ready = 1;
    else out->ids_copied = 1;
    return 0;
  }
#endif

  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;
  if (et_i64_tensor_borrow_begin_v1(out->ids, &borrow, &error))
    return g3t_i64_failure(&error);
  if (et_i64_tensor_borrow_view_v1(borrow, &view, &error)) {
    g3t_i64_failure(&error);
    goto fail;
  }
  if (!view || view->rank != 1 || !view->shape || view->shape[0] != 1 ||
      view->byte_length != sizeof(int64_t) || !view->data) {
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    goto fail;
  }
  int64_t token;
  memcpy(&token, view->data, sizeof(token));
  if (token < 0 || token > 255) {
    g3t_bad(G3T_ARGUMENT, G3T_TOPOLOGY);
    goto fail;
  }
  if (text &&
      ((const unsigned char *)carrier)[sizeof(int64_t)] !=
          (unsigned char)token) {
    g3t_bad(G3T_ARGUMENT, G3T_TOPOLOGY);
    goto fail;
  }
  if (et_i64_tensor_borrow_end_v1(&borrow, &error)) abort();
  if (text) {
    out->text_ready = 1;
  } else {
    unsigned char encoded[8];
    for (size_t i = 0; i < 8; ++i)
      encoded[i] = (unsigned char)((uint64_t)token >> (i * 8));
    memcpy((unsigned char *)carrier + sizeof(int64_t), encoded, 8);
    memcpy(out->staged_id, encoded, 8);
    out->ids_copied = 1;
  }
  return 0;
fail: {
    const int64_t domain = g3t_error_domain;
    const int64_t category = g3t_error_category;
    const int64_t code = g3t_error_code;
    if (et_i64_tensor_borrow_end_v1(&borrow, &error)) abort();
    g3t_fail(domain, category, code);
    return category;
  }
}
int64_t et_g3t_private_output_copy_decode_ids_v1(
    void *context, void *output, void *staging_header) {
  return g3t_decode_carrier(context, output, staging_header, 0);
}
int64_t et_g3t_private_output_accept_text_v1(
    void *context, void *output, void *raw_header) {
  return g3t_decode_carrier(context, output, raw_header, 1);
}
#endif
int64_t et_g3t_private_frame_begin_v1(
    void *candidate, void *input_candidate, int64_t frame_kind) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || !c->pending_output || c->frame.active ||
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
      (c->pending_output->P != 1 && c->pending_output->P != 2) ||
#else
      c->pending_output->P != 1 ||
#endif
      (frame_kind != 1 && frame_kind != 2))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  et_a2_kv_cache *cache = NULL;
  int64_t token;
  if (frame_kind == 1) {
    g3t_input *input = (g3t_input *)g3t_admit_record(
        input_candidate, G3T_INPUT, 0);
    if (!input) return g3t_error_category;
    if (c->prefill_committed || c->sampled ||
        input->length != c->pending_output->P ||
        input->length + c->policy[4] > 2 ||
#ifndef ET_G3T_P2_ZERO_BUDGET_PRIVATE
        input->length != 1 ||
#endif
        input->length < 1 || input->length > 2 ||
        input->ids[0] < 0 || input->ids[0] > 255 ||
        (input->length == 2 &&
         (input->ids[1] < 0 || input->ids[1] > 255)) || input->h.busy)
      return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    token = input->ids[0];
    et_kernel_error error;
    if (g3t_capture_kernel(et_a2_kv_cache_create_v1(
            1, 1, 2, 2, 2, &cache, &error), &error))
      return g3t_error_category;
  } else {
    if (input_candidate || !c->prefill_committed || !c->sampled ||
        !c->binding_ready)
      return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    if (g3t_check_binding(c)) return g3t_error_category;
    token = c->sample_token;
  }
  memset(&c->frame, 0, sizeof(c->frame));
  c->frame.candidate_cache = cache;
  c->frame.token = token;
  c->frame.prompt_length = frame_kind == 1 ? c->pending_output->P : 1;
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
  if (frame_kind == 1 && c->frame.prompt_length == 2) {
    g3t_input *input = (g3t_input *)input_candidate;
    memcpy(c->frame.prompt_ids, input->ids, sizeof(c->frame.prompt_ids));
  }
#endif
  c->frame.active = 1;
  c->frame.kind = (int)frame_kind;
  return 0;
}
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
#include "g3t_prefill2_roles.inc"
#endif
#include "g3t_prefill_roles.inc"
int64_t et_g3t_private_frame_prepare_v1(
    void *candidate, void *staged_result) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (staged_result) return g3t_bad(G3T_ARGUMENT, G3T_IDENTITY);
  if (c->call_kind != 2 || !c->pending_output || !c->frame.active ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      c->frame.prepared ||
      (c->frame.kind == 1 ? (c->prefill_committed || c->sampled) :
       (c->frame.kind != 2 || !c->prefill_committed || !c->sampled)))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (c->frame.kind == 2) {
    if (g3t_check_binding(c)) return g3t_error_category;
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
    g3t_output *out = c->pending_output;
    if (out->parent_ctx != c || out->h.state != G3T_PENDING ||
        !out->h.busy || out->P != 1 || out->G != 1 ||
        out->length != 1 || out->cache_length != 2 ||
        !out->ids || out->numeric_ready != 1 ||
        out->ids_copied != 1 || out->text_ready != 1 ||
        c->frame.token != c->sample_token)
      return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    c->frame.prepared = 1;
    return 0;
#else
    /* Final preparation needs text-ready output from the publication leaf. */
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
#endif
  }
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (c->policy[4] == 0) {
    g3t_output *out = c->pending_output;
    if (out->parent_ctx != c || out->h.state != G3T_PENDING ||
        !out->h.busy || !g3t_zero_prompt(c, out) || out->G != 0 ||
        out->length != 0 || out->cache_length != out->P || !out->ids ||
        out->numeric_ready != 1 || out->ids_copied != 1 ||
        out->text_ready != 1 ||
        memcmp(out->rng, c->rng, sizeof(out->rng)))
      return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  }
#endif
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
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
static int64_t g3t_zero_preflight(g3t_context *c) {
  g3t_output *out = c->pending_output;
  if (!out || c->policy[4] != 0 || !c->frame.active ||
      c->frame.kind != 1 || c->frame.next_ordinal != 21 ||
      !c->frame.prepared || !c->frame.transaction ||
      !c->frame.candidate_cache || c->prefill_committed || c->sampled ||
      c->binding_ready || out->h.state != G3T_PENDING || !out->h.busy ||
      out->parent_ctx != c || !g3t_zero_prompt(c, out) || out->G != 0 ||
      out->length != 0 || out->cache_length != out->P ||
      out->numeric_ready != 1 || out->ids_copied != 1 ||
      out->text_ready != 1 || !out->ids ||
      memcmp(out->rng, c->rng, sizeof(out->rng)))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);

  et_f32_tensor_error pin_error;
  if (et_g3t_model_pins_check_internal(&c->pins, &pin_error))
    return g3t_f32_failure(&pin_error);
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    const et_kernel_tensor_view_v1 *v = &c->pins.views[i];
    if (!v->data || c->frame.binding_identities[i] != c->pins.identities[i] ||
        v->byte_length > sizeof(c->frame.binding_values) - offset ||
        memcmp(c->frame.binding_values + offset, v->data, v->byte_length))
      return g3t_bad(G3T_STATE, G3T_TOPOLOGY);
    offset += v->byte_length;
  }
  if (offset != sizeof(c->frame.binding_values))
    return g3t_bad(G3T_INTERNAL, G3T_INVARIANT);

  et_i64_tensor_borrow *id_borrow = NULL;
  const et_kernel_tensor_view_v1 *ids = NULL;
  et_i64_tensor_error id_error;
  if (et_i64_tensor_borrow_begin_v1(out->ids, &id_borrow, &id_error))
    return g3t_i64_failure(&id_error);
  if (et_i64_tensor_borrow_view_v1(id_borrow, &ids, &id_error))
    g3t_i64_failure(&id_error);
  else if (!ids || ids->rank != 1 || !ids->shape || ids->shape[0] != 0 ||
           ids->byte_length != 0)
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
  int64_t category = g3t_error_category, domain = g3t_error_domain;
  int64_t code = g3t_error_code;
  if (et_i64_tensor_borrow_end_v1(&id_borrow, &id_error)) abort();
  if (category) return g3t_fail(domain, category, code);

  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *mask = NULL;
  et_kernel_error error;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_begin_v1(
          c->frame.transaction, 0, &view, &error), &error))
    return g3t_error_category;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_tensors_v1(
          view, &keys, &values, &lengths, &mask, &error), &error))
    goto view_end;
  if (!keys || !values || !lengths || !mask ||
      !keys->data || !values->data || !lengths->data || !mask->data ||
      lengths->rank != 1 || !lengths->shape || lengths->shape[0] != 1 ||
      lengths->byte_length != sizeof(int64_t) ||
      ((const int64_t *)lengths->data)[0] != out->P ||
      mask->rank != 2 || !mask->shape || mask->shape[0] != 1 ||
      mask->shape[1] != 2 || mask->byte_length != 2 ||
      ((const uint8_t *)mask->data)[0] != 1 ||
      ((const uint8_t *)mask->data)[1] != (uint8_t)(out->P == 2))
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
view_end:
  category = g3t_error_category; domain = g3t_error_domain;
  code = g3t_error_code;
  if (et_a2_kv_cache_transaction_view_end_v1(&view, &error)) abort();
  if (category) return g3t_fail(domain, category, code);

  /* A successful read lease proves the old cache has no competing lease.
   * End it before the no-failure destruction tail. */
  et_a2_kv_cache_read_borrow *old = NULL;
  const et_kernel_tensor_view_v1 *old_keys = NULL, *old_values = NULL;
  const et_kernel_tensor_view_v1 *old_lengths = NULL, *old_mask = NULL;
  if (g3t_capture_kernel(et_a2_kv_cache_read_borrow_begin_v1(
          c->cache, &old, &error), &error)) return g3t_error_category;
  if (g3t_capture_kernel(et_a2_kv_cache_read_borrow_layer_v1(
          old, 0, &old_keys, &old_values, &old_lengths, &old_mask,
          &error), &error)) goto old_end;
  if (!old_keys || !old_values || !old_lengths || !old_mask ||
      !old_lengths->data || old_lengths->rank != 1 ||
      !old_lengths->shape || old_lengths->shape[0] != 1 ||
      old_lengths->byte_length != sizeof(int64_t) ||
      ((const int64_t *)old_lengths->data)[0] != 0)
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
old_end:
  category = g3t_error_category; domain = g3t_error_domain;
  code = g3t_error_code;
  if (et_a2_kv_cache_read_borrow_end_v1(&old, &error)) abort();
  return category ? g3t_fail(domain, category, code) : 0;
}
#endif
/* The only fallible final-commit work is done here. A2's view admission
 * proves the sole layer is staged and that no nested view exists. */
static int64_t g3t_final_preflight(g3t_context *c) {
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (c->policy[4] == 0) return g3t_zero_preflight(c);
#endif
  g3t_output *out = c->pending_output;
  if (!out || !c->frame.active || c->frame.kind != 2 ||
      c->frame.next_ordinal != 21 || !c->frame.prepared ||
      !c->frame.transaction || c->frame.candidate_cache ||
      !c->prefill_committed || !c->sampled || !c->binding_ready ||
      out->h.state != G3T_PENDING || !out->h.busy ||
      out->parent_ctx != c || out->P != 1 || out->G != 1 ||
      out->length != 1 || out->cache_length != 2 ||
      out->numeric_ready != 1 || out->ids_copied != 1 ||
      out->text_ready != 1 || !out->ids ||
      c->frame.token != c->sample_token ||
      memcmp(out->rng, c->successor_rng, sizeof(out->rng)))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (g3t_check_binding(c)) return g3t_error_category;
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *ids = NULL;
  et_i64_tensor_error i64_error;
  if (et_i64_tensor_borrow_begin_v1(out->ids, &borrow, &i64_error))
    return g3t_i64_failure(&i64_error);
  if (et_i64_tensor_borrow_view_v1(borrow, &ids, &i64_error)) {
    g3t_i64_failure(&i64_error);
    goto i64_fail;
  }
  int64_t token = -1;
  if (!ids || ids->rank != 1 || !ids->shape || ids->shape[0] != 1 ||
      ids->byte_length != sizeof(token) || !ids->data) {
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    goto i64_fail;
  }
  memcpy(&token, ids->data, sizeof(token));
  if (token != c->sample_token || token < 0 || token > 255 ||
      out->staged_id[0] != (unsigned char)token) {
    g3t_bad(G3T_STATE, G3T_TOPOLOGY);
    goto i64_fail;
  }
  for (size_t i = 1; i < sizeof(out->staged_id); ++i)
    if (out->staged_id[i] != 0) {
      g3t_bad(G3T_STATE, G3T_TOPOLOGY);
      goto i64_fail;
    }
  if (et_i64_tensor_borrow_end_v1(&borrow, &i64_error)) abort();

  et_a2_kv_cache_transaction_view *view = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL;
  const et_kernel_tensor_view_v1 *lengths = NULL, *mask = NULL;
  et_kernel_error error;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_begin_v1(
          c->frame.transaction, 0, &view, &error), &error))
    return g3t_error_category;
  if (g3t_capture_kernel(et_a2_kv_cache_transaction_view_tensors_v1(
          view, &keys, &values, &lengths, &mask, &error), &error))
    goto a2_fail;
  if (!keys || !values || !lengths || !mask ||
      !keys->data || !values->data || !lengths->data || !mask->data ||
      lengths->rank != 1 || !lengths->shape || lengths->shape[0] != 1 ||
      lengths->byte_length != sizeof(int64_t) ||
      ((const int64_t *)lengths->data)[0] != 2 ||
      mask->rank != 2 || !mask->shape ||
      mask->shape[0] != 1 || mask->shape[1] != 2 ||
      mask->byte_length != 2 ||
      ((const uint8_t *)mask->data)[0] != 1 ||
      ((const uint8_t *)mask->data)[1] != 1) {
    g3t_bad(G3T_INTERNAL, G3T_INVARIANT);
    goto a2_fail;
  }
  if (et_a2_kv_cache_transaction_view_end_v1(&view, &error)) abort();
  return 0;
i64_fail: {
    int64_t domain = g3t_error_domain, category = g3t_error_category;
    int64_t code = g3t_error_code;
    if (et_i64_tensor_borrow_end_v1(&borrow, &i64_error)) abort();
    g3t_fail(domain, category, code);
    return category;
  }
a2_fail: {
    int64_t domain = g3t_error_domain, category = g3t_error_category;
    int64_t code = g3t_error_code;
    if (et_a2_kv_cache_transaction_view_end_v1(&view, &error)) abort();
    g3t_fail(domain, category, code);
    return category;
  }
}
int64_t et_g3t_private_call_prepare_end_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || c->frame.end_prepared ||
      c->final_committed)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (g3t_final_preflight(c)) return g3t_error_category;
  c->frame.end_prepared = 1;
  return 0;
}
#endif
int64_t et_g3t_private_frame_commit_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || !c->pending_output || !c->frame.active ||
      c->frame.next_ordinal != 21 || !c->frame.transaction ||
      !c->frame.prepared ||
      (c->frame.kind == 1 ? (c->prefill_committed || c->sampled) :
       (c->frame.kind != 2 || !c->prefill_committed || !c->sampled)))
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
  if (c->frame.kind == 1 && c->policy[4] == 0) {
    if (!c->frame.end_prepared || g3t_final_preflight(c))
      return g3t_error_category ?
          g3t_error_category : g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    et_kernel_error error;
    if (et_a2_kv_cache_transaction_commit_v1(
            &c->frame.transaction, &error)) abort();
    if (et_a2_kv_cache_destroy_v1(&c->cache, &error)) abort();
    c->cache = c->frame.candidate_cache;
    c->frame.candidate_cache = NULL;
    memcpy(c->binding_identities, c->frame.binding_identities,
           sizeof(c->binding_identities));
    memcpy(c->binding_values, c->frame.binding_values,
           sizeof(c->binding_values));
    c->binding_ready = 1;
    memcpy(c->last_logits,
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
           c->frame.prompt_length == 2 ? c->frame.p2.z + 256 :
#endif
           c->frame.z, sizeof(c->last_logits));
    c->prefill_committed = 1;
    g3t_output *out = c->pending_output;
    out->parent_ctx = NULL;
    out->h.state = G3T_LIVE;
    out->h.busy = 0;
    c->pending_output = NULL;
    c->final_committed = 1;
    memset(&c->frame, 0, sizeof(c->frame));
    return 0;
  }
#endif
  if (c->frame.kind == 2) {
    if (!c->frame.end_prepared || g3t_final_preflight(c))
      return g3t_error_category ?
          g3t_error_category : g3t_bad(G3T_STATE, G3T_LIFECYCLE);
    /* All recoverable checks end here. This A2 commit has no allocator;
     * an impossible rejection after preflight is fail-stop. */
    et_kernel_error error;
    if (et_a2_kv_cache_transaction_commit_v1(
            &c->frame.transaction, &error)) abort();
    g3t_output *out = c->pending_output;
    memcpy(c->rng, c->successor_rng, sizeof(c->rng));
    out->parent_ctx = NULL;
    memset(out->staged_id, 0, sizeof(out->staged_id));
    out->h.state = G3T_LIVE;
    out->h.busy = 0;
    c->pending_output = NULL;
    c->final_committed = 1;
    memset(&c->frame, 0, sizeof(c->frame));
    return 0;
  }
#else
  if (c->frame.kind == 2)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
#endif
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
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
int64_t et_g3t_private_call_finish_v1(void *candidate) {
  g3t_clear();
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c) return g3t_error_category;
  if (!c->final_committed) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  owner *o = g3t_model(c->model);
  et_f32_tensor_error error;
  if (!o || o->active != c || !c->h.busy || c->call_kind != 2 ||
      c->pending_output || c->frame.active || c->frame.transaction ||
      et_g3t_model_pins_check_internal(&c->pins, &error))
    abort();
  et_g3t_model_pins_end_internal(&c->pins);
  o->active = NULL;
  c->h.busy = 0;
  c->call_kind = 0;
  c->final_committed = 0;
  c->sampled = 0;
  c->sample_token = 0;
  memset(c->successor_rng, 0, sizeof(c->successor_rng));
  return 0;
}
#endif
int64_t et_g3t_private_sample_v1(void *candidate) {
  static const uint64_t logits_shape[2] = {1,256}, rng_shape[1] = {4};
  static const uint64_t token_shape[1] = {1};
  g3t_clear();
  g3t_context *c = g3t_active(candidate);
  if (!c) return g3t_error_category;
  if (c->call_kind != 2 || !c->prefill_committed || c->frame.active ||
      !c->pending_output || c->sampled || !c->binding_ready)
    return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
  if (g3t_check_binding(c)) return g3t_error_category;
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
#ifdef ET_G3T_FINAL_PUBLICATION_PRIVATE
  if (c->final_committed) return g3t_bad(G3T_STATE, G3T_LIFECYCLE);
#endif
  owner *o = g3t_model(c->model);
  if (!o || o->active != c) abort();
  et_f32_tensor_error error;
  if (et_g3t_model_pins_check_internal(&c->pins, &error)) abort();
  if (c->pending_output) {
    et_i64_tensor_error i64_error;
    if (et_m3_private_i64_unborrowed_v1(
            c->pending_output->ids, &i64_error))
      return g3t_i64_failure(&i64_error);
  }
  et_kernel_error kernel_error;
  if (c->frame.transaction && et_a2_kv_cache_transaction_abort_v1(
          &c->frame.transaction, &kernel_error)) abort();
  if (c->frame.candidate_cache && et_a2_kv_cache_destroy_v1(
          &c->frame.candidate_cache, &kernel_error)) abort();
  memset(&c->frame, 0, sizeof(c->frame));
  if (c->pending_output) {
    g3t_output *output = c->pending_output;
    et_i64_tensor_error i64_error;
    if (et_i64_tensor_destroy_v1(&output->ids, &i64_error)) abort();
    output->parent_ctx = NULL;
    output->P = output->G = output->length = output->cache_length = 0;
    memset(output->rng, 0, sizeof(output->rng));
    memset(output->staged_id, 0, sizeof(output->staged_id));
    output->numeric_ready = output->ids_copied = output->text_ready = 0;
    output->h.busy = 0;
    output->h.state = G3T_DEAD;
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
int64_t et_g3t_test_output_word_v1(void *candidate, int64_t field) {
  g3t_output *output = (g3t_output *)g3t_admit_record(
      candidate, G3T_OUTPUT, 1);
  if (!output) return -1;
  switch (field) {
    case 0: return output->numeric_ready;
    case 1: return output->length;
    case 2: return output->cache_length;
    case 3: return output->ids_copied;
    case 4: return output->text_ready;
    case 5: return output->ids != NULL;
    case 6: {
      int64_t id = -1;
      et_i64_tensor_error error;
      return output->ids &&
                     !et_i64_tensor_copy_to_v1(output->ids, &id, 1, &error)
                 ? id : -1;
    }
    default: return field >= 7 && field < 11 ? output->rng[field - 7] : -1;
  }
}
void *et_g3t_test_output_borrow_begin_v1(void *candidate) {
  g3t_output *output = (g3t_output *)g3t_admit_record(
      candidate, G3T_OUTPUT, 1);
  if (!output || (output->h.state != G3T_PENDING &&
                  output->h.state != G3T_LIVE) || !output->ids)
    return NULL;
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error error;
  return et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &error) ?
         NULL : borrow;
}
int64_t et_g3t_test_output_borrow_end_v1(void *candidate) {
  et_i64_tensor_borrow *borrow = candidate;
  et_i64_tensor_error error;
  return et_i64_tensor_borrow_end_v1(&borrow, &error);
}
int64_t et_g3t_test_output_state_v1(void *candidate) {
  g3t_output *out = (g3t_output *)g3t_admit_record(
      candidate, G3T_OUTPUT, 1);
  return out ? out->h.state : -2;
}
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
int64_t et_g3t_test_input_length_set_v1(void *candidate, int64_t length) {
  g3t_input *input = (g3t_input *)g3t_admit_record(
      candidate, G3T_INPUT, 0);
  if (!input || input->h.busy || length < 1 || length > 3) return -1;
  input->length = length;
  return 0;
}
#endif
int64_t et_g3t_test_output_id_set_v1(void *candidate, int64_t token) {
  g3t_output *out = (g3t_output *)g3t_admit_record(
      candidate, G3T_OUTPUT, 1);
  if (!out || out->h.state != G3T_PENDING || !out->numeric_ready ||
      out->ids_copied || !out->ids) return -1;
  et_i64_tensor_error error;
  return et_i64_tensor_copy_from_v1(out->ids, &token, 1, &error);
}
int64_t et_g3t_test_binding_flip_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || !c->h.busy || !c->binding_ready) return -1;
  c->binding_values[0] ^= 1u;
  return 0;
}
#ifdef ET_G3T_ZERO_BUDGET_PRIVATE
int64_t et_g3t_test_frame_binding_flip_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || !c->h.busy || !c->frame.active ||
      c->frame.kind != 1 || !c->frame.prepared) return -1;
  c->frame.binding_values[0] ^= 1u;
  return 0;
}
#endif
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
int64_t et_g3t_test_frame_logit_bits_v1(void *candidate, int64_t index) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || !c->frame.active || c->frame.next_ordinal != 21 ||
      index < 0 || index >= 256) return -1;
  uint32_t bits;
  memcpy(&bits,
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
         c->frame.kind == 1 && c->frame.prompt_length == 2 ?
             &c->frame.p2.z[256 + index] :
#endif
             &c->frame.z[index], sizeof(bits));
  return bits;
}
int64_t et_g3t_test_cache_length_v1(void *candidate) {
  g3t_context *c = g3t_admit(candidate, 0);
  if (!c || !c->cache) return -1;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys = NULL, *values = NULL, *lengths = NULL;
  const et_kernel_tensor_view_v1 *mask = NULL;
  et_kernel_error error;
  if (et_a2_kv_cache_read_borrow_begin_v1(c->cache, &borrow, &error)) return -1;
  int64_t length = et_a2_kv_cache_read_borrow_layer_v1(
      borrow, 0, &keys, &values, &lengths, &mask, &error) ? -1 :
      ((const int64_t *)lengths->data)[0];
  if (et_a2_kv_cache_read_borrow_end_v1(&borrow, &error)) abort();
  return length;
}
#endif
#endif
